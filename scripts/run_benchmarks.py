"""Replay fixed H-GBP policies. Dataset identity never selects solver parameters."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"
DRIVER_VERSION = "3"
NUMERICAL_PREFIXES = (
    "GBP_", "HGBP_", "OMP_", "KMP_", "GOMP_", "MKL_", "OPENBLAS_",
    "BLIS_", "VECLIB_", "NUMEXPR_", "TBB_", "EIGEN_", "GOTO_", "BLAS_",
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def affinity_mask(value):
    try:
        mask = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer CPU mask, e.g. 0xffff") from exc
    if not 0 < mask < 2**64:
        raise argparse.ArgumentTypeError("affinity mask must be a nonzero 64-bit mask")
    return hex(mask)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be finite positive seconds")
    return seconds


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("suite", choices=("pgo", "ba"))
    parser.add_argument("datasets", nargs="*", help="canonical names; defaults to all PGO / BA main10")
    parser.add_argument("--subset", help="PGO: all, SE2, SE3; BA: main10, alamo")
    parser.add_argument("--input", type=Path, help="arbitrary input, hashed but never reference-verified")
    parser.add_argument("--space", choices=("SE2", "SE3"), help="required for PGO --input only")
    parser.add_argument("--label", help="custom display label only, not a reference lookup")
    thread_group = parser.add_mutually_exclusive_group()
    thread_group.add_argument("--threads", type=int, choices=(1, 16))
    thread_group.add_argument("--compare-threads", action="store_true", help="all 1-thread runs, then all 16-thread runs, sequentially")
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--timeout", type=positive_seconds, default=2000.0, help="maximum process wall seconds per run (default: 2000)")
    parser.add_argument("--build-dir", type=Path, default=PROJECT_ROOT / "build", help="build root with separate pgo/ and ba/ executable/runtime directories")
    parser.add_argument("--data-root", type=Path, help="canonical inputs; fallback HGBP_PGO_DATA_ROOT / HGBP_BA_DATA_ROOT, then data/<suite>")
    parser.add_argument("--runtime-root", type=Path, help="RootBA conda environment; fallback HGBP_ROOTBA_RUNTIME; used only for BA")
    parser.add_argument("--output-root", type=Path, required=True, help="must not exist, including for --dry-run")
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--affinity-mask", type=affinity_mask, default="0xffff", help="hardware selection only (default: 0xffff)")
    hardware.add_argument("--no-affinity", action="store_true", help="do not set process affinity; the solver's worker-placement policy is unchanged")
    parser.add_argument("--dry-run", action="store_true", help="verify files/hashes and write planned commands; never launch solvers")
    args = parser.parse_intermixed_args(argv)
    if args.input:
        if args.datasets or args.subset or args.data_root:
            parser.error("--input cannot be combined with dataset names, --subset or --data-root")
        if args.suite == "pgo" and args.space is None:
            parser.error("PGO --input requires --space SE2 or SE3")
    elif args.space or args.label:
        parser.error("--space and --label require --input")
    if args.suite == "ba" and args.space:
        parser.error("--space is only valid for PGO")
    if args.datasets and args.subset:
        parser.error("choose dataset names or --subset, not both")
    if args.label and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.label):
        parser.error("--label must contain only letters, digits, underscores, dots or hyphens")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate dataset names")
    return args


def select_inputs(args, manifest):
    if args.input:
        return [{"label": args.label or "custom", "space": args.space,
                 "path": args.input.resolve(), "canonical": False, "expected_sha256": None}]
    subsets = manifest["subsets"][args.suite]
    subset = args.subset or subsets["default"]
    if subset == "default" or subset not in subsets:
        raise ValueError(f"unknown {args.suite} subset: {subset}")
    names = args.datasets or subsets[subset]
    fallback = os.environ.get(f"HGBP_{args.suite.upper()}_DATA_ROOT")
    root = (args.data_root or (Path(fallback) if fallback else PROJECT_ROOT / "data" / args.suite)).resolve()
    selected = []
    for name in names:
        record = manifest["datasets"].get(name)
        if record is None or record["suite"] != args.suite:
            raise ValueError(f"unknown {args.suite} dataset: {name}")
        selected.append({"label": name, "space": record.get("space"),
                         "path": root / record["path"], "canonical": True,
                         "expected_sha256": record["sha256"]})
    return selected


def ba_arguments(policy, threads):
    # Matches the accepted run_ba.py:hgbp_arguments boolean conventions.
    result = []
    for name, value in policy.items():
        if name in ("description", "threads"):
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if name == "normalize_bal":
                result.append("--normalize-bal" if value else "--no-normalize-bal")
            elif name == "pair_sample_rescale":
                if not value:
                    result.append("--pair-sample-no-rescale")
            elif value:
                result.append(flag)
        else:
            result.extend((flag, str(value)))
    return result + ["--build-threads", str(threads), "--gbp-threads", str(threads)]


def solver_command(suite, config, space, executable, input_path, output_path, threads, mask):
    command = [str(executable), "--problem-file", str(input_path), "--out-json", str(output_path)]
    if suite == "ba":
        return command + ba_arguments(config["policy"], threads)
    arguments = list(config["profiles"][space]["arguments"])
    if mask is None:
        index = arguments.index("--process-affinity-mask")
        del arguments[index:index + 2]
    return command + [value.format(threads=threads, affinity_mask=mask) for value in arguments]


def environment_for(template, threads, runtime_root=None, inherited=None):
    inherited = os.environ if inherited is None else inherited
    removed = sorted(key for key in inherited if key.upper().startswith(NUMERICAL_PREFIXES))
    environment = {key: value for key, value in inherited.items() if key not in removed}
    overrides = {key: value.format(threads=threads) for key, value in template.items()}
    path_key = next((key for key in environment if key.upper() == "PATH"), "PATH")
    path = environment.pop(path_key, "")
    if runtime_root:
        paths = (runtime_root / "Library" / "bin", runtime_root / "Scripts")
        path = os.pathsep.join([str(p) for p in paths] + [path])
    overrides["PATH"] = path
    environment.update(overrides)
    return environment, {"set": overrides, "removed": removed,
                         "inherited_temp": {k: environment[k] for k in ("TEMP", "TMP", "TMPDIR") if k in environment}}


@contextmanager
def process_affinity(mask):
    """Apply and restore the driver's CPU allocation; every child inherits it."""
    if mask is None:
        yield
        return
    bits = int(mask, 0)
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        previous, available = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(previous), ctypes.byref(available)):
            raise ctypes.WinError(ctypes.get_last_error())
        if bits & available.value != bits:
            raise ValueError("requested affinity is unavailable; select --affinity-mask or --no-affinity")
        def set_mask(value):
            if not kernel.SetProcessAffinityMask(handle, value):
                raise ctypes.WinError(ctypes.get_last_error())
        set_mask(bits)
        restore = lambda: set_mask(previous.value)
    elif hasattr(os, "sched_setaffinity"):
        previous = os.sched_getaffinity(0)
        selected = {cpu for cpu in range(64) if bits & (1 << cpu)}
        if not selected <= previous:
            raise ValueError("requested affinity is unavailable; select --affinity-mask or --no-affinity")
        os.sched_setaffinity(0, selected)
        restore = lambda: os.sched_setaffinity(0, previous)
    else:
        raise ValueError("affinity is unsupported on this OS; use --no-affinity")
    try:
        yield
    finally:
        restore()


def require_fields(actual, expected):
    for key, value in expected.items():
        if key not in actual or actual[key] != value:
            raise ValueError(f"native policy mismatch: {key} expected {value!r}, got {actual.get(key)!r}")


def finite_nonnegative(value, label):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid native metric: {label}")


def read_metrics(suite, output_path, threads, space=None, require_work=False):
    result = load_json(output_path)
    if suite == "pgo":
        history = result["mg_history"]
        config = result["config"]
        if config["sync_num_threads"] != threads or config["num_outer"] != 20:
            raise ValueError("PGO effective thread/outer budget differs from request")
        require_fields(config, {"adaptive_precision": True, "smoother": "gbp", "group_size": 20,
                                "basis_rebuild_period": 1, "robust_huber_delta": 5})
        for key in ("cycle_energy_krylov", "skip_coarse_quality_ablation", "direct_enabled",
                    "direct_reference_enabled", "final_direct_polish_steps", "final_coarse_polish_passes"):
            if config.get(key, False):
                raise ValueError(f"unexpected PGO ablation/Direct policy: {key}")
        if space:
            cycles, sweeps, reduced = (3, 100, 4) if space == "SE2" else (5, 50, 12)
            require_fields(config, {"inner_cycles": cycles, "pre_sweeps": sweeps, "r_reduced": reduced})
        if space == "SE3":
            require_fields(config, {"implicit_fine_operator_threads": threads, "persistent_sweeps": True,
                                    "automatic_coarse": True, "cycle_line_search": True,
                                    "precision_initialization": "warm-transport-balanced",
                                    "coarse_linear_backend": "cholmod_auto", "eta_lift": "precision",
                                    "residual_stop_enabled": True})
        if [row["outer"] for row in history] != list(range(21)):
            raise ValueError("PGO must report the initial state and 20 outer iterations")
        if len(result.get("direct_history", [])) > 1:
            raise ValueError("unexpected Direct iterations")
        if result.get("worker_affinity_failures", 0):
            raise ValueError("native worker_affinity_failures is nonzero")
        for row in history:
            finite_nonnegative(row["nonlinear_objective"], "history cost")
            for key, value in row.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise ValueError(f"nonfinite PGO history field: {key}")
            finite_nonnegative(row["outer_total_sec"], "outer_total_sec")
            for key in ("full_precision_sweeps", "eta_only_sweeps", "coarse_solves"):
                finite_nonnegative(row[key], key)
            if row["outer"]:
                if require_work and (row["full_precision_sweeps"] + row["eta_only_sweeps"] <= 0 or row["coarse_solves"] <= 0):
                    raise ValueError("PGO outer iteration did not execute GBP/coarse work")
        native_key = "hgbp_solver_wall_sec" if "hgbp_solver_wall_sec" in result else "solver_wall_sec"
        metrics = {"native_wall_sec": float(result[native_key]), "native_wall_field": native_key,
                   "outer_count": len(history) - 1, "cost": float(history[-1]["nonlinear_objective"]),
                   "counters": {key: sum(row[key] for row in history[1:]) for key in
                                ("full_precision_sweeps", "eta_only_sweeps", "coarse_solves")}}
    else:
        if result["threads"] != threads or result["build_threads"] != threads:
            raise ValueError("BA effective threads differ from request")
        if result["outer"] != 20 or len(result["costs"]) != 21:
            raise ValueError("BA must report 20 outer attempts and 21 costs")
        require_fields(result, {"coarse_operator": "additive", "linear_controller": "fcg",
                                "aggregation": "connected", "gbp_model": "normalized", "fine_smoother": "gbp",
                                "mg_cycles": 5, "pre_sweeps": 3, "gbp_full_sweeps": 32,
                                "graph_neighbors": 4, "coarse_groups_budget": 24,
                                "variance_rel_tol": 1e-6, "linear_rel_tol": 0.1,
                                "pair_sample_cap": 16, "pair_sample_rescale": True,
                                "message_damping": 1.0, "initial_lambda": 1e-4,
                                "pair_factor_scale": 1.0, "krylov_start_outer": 1,
                                "unreduced_unary": False, "no_pose_scaling": False})
        for value in result["costs"]:
            finite_nonnegative(value, "BA cost history")
        for key, values in result.items():
            if isinstance(values, list):
                for value in values:
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"nonfinite BA history field: {key}")
        for key in ("outer_sec", "full_sweeps", "eta_sweeps", "linear_cycles"):
            if len(result[key]) != 20:
                raise ValueError(f"BA {key} must contain all 20 outer attempts")
            for value in result[key]:
                finite_nonnegative(value, key)
                if require_work and key != "outer_sec" and value <= 0:
                    raise ValueError(f"BA iteration did not execute {key}")
        metrics = {"native_wall_sec": float(result["total_sec"]), "native_wall_field": "total_sec",
                   "outer_count": result["outer"], "cost": float(result["final_cost"]),
                   "mre": float(result["final_are_px"]), "rmse": float(result["final_reprojection_rmse_px"]),
                   "observations": result["num_observations"],
                   "counters": {key: sum(result[key]) for key in ("full_sweeps", "eta_sweeps", "linear_cycles")}}
        config = {key: value for key, value in result.items() if not isinstance(value, (list, dict))}
    for key in ("native_wall_sec", "cost", "mre", "rmse"):
        if key in metrics:
            finite_nonnegative(metrics[key], key)
    if metrics["native_wall_sec"] <= 0:
        raise ValueError("native wall time must be positive")
    # Banded eigensolver failures are recoverable fallbacks, not failed solves.
    metrics["diagnostics"] = {key: result[key] for key in ("basis_band_failures", "worker_affinity_failures") if key in result}
    metrics["warnings"] = ["banded eigensolver used a recoverable fallback"] if result.get("basis_band_failures", 0) else []
    if suite == "ba":
        if metrics["observations"] <= 0 or not math.isclose(metrics["rmse"]**2 * metrics["observations"], metrics["cost"], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("inconsistent BA SSR/RMSE/observation count")
        if metrics["mre"] > metrics["rmse"] + 1e-9:
            raise ValueError("BA MRE exceeds RMSE")
    return metrics, config


def check_quality(metrics, reference, tolerance):
    if reference is None:
        return {"status": "not_applicable", "reason": "custom input: no canonical hash or quality reference"}
    errors = {key: abs(metrics[key] - reference[key]) / max(abs(reference[key]), 1e-300)
              for key in ("cost", "mre", "rmse") if key in reference}
    return {"status": "pass" if all(error <= tolerance for error in errors.values()) else "fail",
            "relative_tolerance": tolerance, "relative_error": errors,
            "time_over_reference": metrics["native_wall_sec"] / reference["time_sec"]}


def version_info():
    def git(*arguments):
        try:
            return subprocess.check_output(["git", *arguments], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"driver_version": DRIVER_VERSION, "driver_sha256": sha256(Path(__file__)),
            "git_head": git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "logical_cpus": os.cpu_count()}


def summarize(records):
    rows = []
    groups = dict.fromkeys((r["dataset"], r["threads"]) for r in records)
    for label, threads in groups:
        runs = [r for r in records if (r["dataset"], r["threads"]) == (label, threads)]
        valid = [r for r in runs if "metrics" in r]
        row = {"dataset": label, "threads": threads, "attempts": len(runs), "completed": len(valid)}
        if valid:
            times = [r["metrics"]["native_wall_sec"] for r in valid]
            row.update(native_wall_median_sec=statistics.median(times), native_wall_min_sec=min(times),
                       native_wall_max_sec=max(times), process_wall_median_sec=statistics.median(r["process_wall_sec"] for r in valid),
                       outer_counts=[r["metrics"]["outer_count"] for r in valid],
                       quality_status="fail" if any(r["status"] != "ok" for r in runs) else valid[0]["quality"]["status"])
            for key in ("cost", "mre", "rmse"):
                if key in valid[0]["metrics"]:
                    row[key + "_median"] = statistics.median(r["metrics"][key] for r in valid)
            if valid[0]["reference"]:
                row["time_over_reference"] = statistics.median(times) / valid[0]["reference"]["time_sec"]
        rows.append(row)
    comparisons = []
    for label in dict.fromkeys(r["dataset"] for r in rows):
        by_thread = {r["threads"]: r for r in rows if r["dataset"] == label and "native_wall_median_sec" in r}
        if set(by_thread) == {1, 16} and by_thread[16]["native_wall_median_sec"] > 0:
            comparisons.append({"dataset": label, "native_time_1_over_16": by_thread[1]["native_wall_median_sec"] / by_thread[16]["native_wall_median_sec"]})
    return {"timing": "Ratios only; no cross-machine speed pass/fail. Quality checked per repeat against its own thread reference.",
            "results": rows, "thread_comparison": comparisons}


def run(args):
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ValueError(f"--output-root must be a new directory: {output_root}")
    config = load_json(CONFIG_ROOT / f"{args.suite}.json")
    if config.get("schema_version") != 3 or "datasets" in config or "shared" in config:
        raise ValueError("obsolete solver configuration: expected unified schema 3")
    manifest = load_json(CONFIG_ROOT / "datasets.json")
    references = load_json(CONFIG_ROOT / "reference.json")
    selected = select_inputs(args, manifest)
    runtime_value = (args.runtime_root or os.environ.get("HGBP_ROOTBA_RUNTIME")) if args.suite == "ba" else None
    runtime = Path(runtime_value).resolve() if runtime_value else None
    if runtime is not None and not runtime.is_dir():
        raise ValueError(f"runtime root does not exist: {runtime}")
    mask = None if args.no_affinity else args.affinity_mask
    thread_counts = [1, 16] if args.compare_threads else [args.threads or 16]
    executables = {}
    for item in selected:
        item["input_sha256"] = sha256(item["path"])
        if item["canonical"] and item["input_sha256"] != item["expected_sha256"]:
            raise ValueError(f"{item['label']}: input SHA-256 mismatch")
        family = item["space"] if args.suite == "pgo" else "ba"
        if family not in executables:
            name = f"{family.lower()}_benchmark" if args.suite == "pgo" else "ba_solver"
            executable = (args.build_dir / args.suite / (name + (".exe" if os.name == "nt" else ""))).resolve()
            # Strict split layout: PGO and BA require different OpenBLAS binaries.
            executables[family] = {"path": str(executable), "sha256": sha256(executable),
                                   "adjacent_dll_sha256": {p.name: sha256(p) for p in sorted(executable.parent.glob("*.dll"))}}
        item["executable"] = executables[family]
        item["reference"] = None
        if item["canonical"]:
            reference = references["datasets"][item["label"]]
            if reference["suite"] != args.suite or reference["input_sha256"] != item["input_sha256"]:
                raise ValueError("reference is not bound to this canonical input")
            item["reference"] = {str(t): reference["threads"][str(t)] for t in thread_counts}

    plan = []
    environments = {}
    for threads in thread_counts:
        for repeat in range(1, args.repeats + 1):
            for item in selected:
                family = item["space"] if args.suite == "pgo" else "ba"
                profile = config["profiles"][family] if args.suite == "pgo" else config
                env, env_record = environment_for(profile["environment"], threads, runtime if args.suite == "ba" else None)
                environments[(family, threads)] = env
                stem = f"{item['label']}_t{threads}_r{repeat}"
                result = output_root / (stem + ".json")
                plan.append({"dataset": item["label"], "suite": args.suite, "space": item["space"],
                             "threads": threads, "repeat": repeat, "canonical": item["canonical"],
                             "input": str(item["path"]), "input_sha256": item["input_sha256"],
                             "canonical_hash_verified": item["canonical"], "executable": item["executable"],
                             "reference": item["reference"][str(threads)] if item["reference"] else None,
                             "reference_status": "available" if item["canonical"] else "custom input: no canonical hash or quality reference",
                             "command": solver_command(args.suite, config, item["space"], item["executable"]["path"], item["path"], result, threads, mask),
                             "environment": env_record, "affinity_mask": mask, "cwd": str(PROJECT_ROOT),
                             "output": str(result), "log": str(output_root / (stem + ".log")), "status": "planned"})
    output_root.mkdir(parents=True, exist_ok=False)
    write_json(output_root / "protocol.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
               "version": version_info(), "config": config, "config_hashes": {name: sha256(CONFIG_ROOT / name) for name in
               (f"{args.suite}.json", "datasets.json", "reference.json")}, "dry_run": args.dry_run,
               "sequential_thread_order": thread_counts, "runtime_root": str(runtime) if runtime else None,
               "runtime_root_applies_to": "ba only", "affinity_mask": mask, "timeout_sec": args.timeout,
               "timing": "Native solver wall includes setup, excludes parsing/output. Process wall recorded separately."})
    write_json(output_root / "runs.json", plan)
    if args.dry_run:
        for record in plan:
            print(json.dumps(record["command"]))
        print(f"Dry run only: {output_root}")
        return 0
    tolerance = references["quality_relative_tolerance"][args.suite]
    with process_affinity(mask):
        for record in plan:
            family = record["space"] if args.suite == "pgo" else "ba"
            record["status"] = "running"
            write_json(output_root / "runs.json", plan)
            start = time.perf_counter()
            try:
                with Path(record["log"]).open("w", encoding="utf-8") as log:
                    process = subprocess.run(record["command"], cwd=PROJECT_ROOT,
                                             env=environments[(family, record["threads"])], stdout=log,
                                             stderr=subprocess.STDOUT, check=False,
                                             timeout=args.timeout,
                                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                record["process_wall_sec"] = time.perf_counter() - start
                record["returncode"] = process.returncode
                if process.returncode:
                    raise ValueError(f"solver exited with code {process.returncode}")
                record["output_sha256"] = sha256(record["output"])
                metrics, effective = read_metrics(args.suite, record["output"], record["threads"], record["space"],
                                                  require_work=record["canonical"])
                record["metrics"], record["effective_config"] = metrics, effective
                record["quality"] = check_quality(metrics, record["reference"], tolerance)
                record["status"] = "quality_failed" if record["quality"]["status"] == "fail" else "ok"
            except subprocess.TimeoutExpired:
                record.update(status="timeout", process_wall_sec=time.perf_counter() - start,
                              error=f"process exceeded {args.timeout:g} seconds", returncode=None)
            except (OSError, ValueError, KeyError, TypeError, OverflowError) as exc:
                record.setdefault("process_wall_sec", time.perf_counter() - start)
                record.update(status="failed", error=str(exc))
            finally:
                write_json(output_root / "runs.json", plan)
                write_json(output_root / "summary.json", summarize(plan))
            print(f"{record['dataset']} t{record['threads']} r{record['repeat']}: {record['status']}", flush=True)
    return 2 if any(record["status"] != "ok" for record in plan) else 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
