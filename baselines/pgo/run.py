"""Replay measured PGO baseline policies, not interchangeable stock solver CLIs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("release_driver_utils", ROOT / "scripts/run_benchmarks.py")
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
POLICIES = core.load_json(HERE / "policies.json")["methods"]
PINS = core.load_json(HERE / "provenance.json")
MANIFEST_PATH = ROOT / "configs/datasets.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("method", choices=POLICIES)
    parser.add_argument("datasets", nargs="*", help="formal-nine canonical names, default all nine")
    parser.add_argument("--data-root", type=Path, help="fallback HGBP_PGO_DATA_ROOT, then data/pgo")
    parser.add_argument("--binary-root", type=Path, required=True, help="directory containing this method's executables and its own runtime")
    parser.add_argument("--library-root", type=Path, help="Linux .so directory; defaults to binary-root")
    parser.add_argument("--output-root", type=Path, required=True, help="new directory only")
    parser.add_argument("--repeats", type=core.positive_int, default=1)
    parser.add_argument("--timeout", type=core.positive_seconds, default=2000)
    parser.add_argument("--require-archived-runtime", action="store_true", help="reject any missing/mismatched archived bundle fingerprint")
    parser.add_argument("--include-unmeasured", action="store_true", help="also attempt historical Globe100k exceptions; never invent a timing")
    hardware = parser.add_mutually_exclusive_group()
    hardware.add_argument("--affinity-mask", type=core.affinity_mask, default="0xffff")
    hardware.add_argument("--no-affinity", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="hash/validate and write commands only; no subprocesses")
    args = parser.parse_intermixed_args(argv)
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("duplicate dataset names")
    if args.library_root and POLICIES[args.method]["platform"] != "linux":
        parser.error("--library-root is only for AMM/Cycle on Linux/WSL")
    return args


def solver_command(method, dataset, space, executable, input_path, run_dir):
    policy = POLICIES[method]
    values = {"input": str(input_path), "result": str(run_dir / "result.json"),
              "poses": str(run_dir / "poses.g2o"), "dimension": 2 if space == "SE2" else 3,
              "iterations": policy.get("iterations", {}).get(dataset, 20)}
    return [str(executable)] + [token.format(**values) for token in policy["arguments"]]


def environment_for(policy, library_root=None):
    prefixes = core.NUMERICAL_PREFIXES + ("G2O_",)
    removed = sorted(k for k in os.environ if k.upper().startswith(prefixes))
    env = {k: v for k, v in os.environ.items() if k not in removed}
    env.update(policy["environment"])
    if library_root:
        env["LD_LIBRARY_PATH"] = str(library_root)
    record = {"set": dict(policy["environment"]), "removed": removed}
    for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "PATH", "TEMP", "TMP", "TMPDIR"):
        if key in env:
            record["set"][key] = env[key]
    # Injected shared libraries invalidate a replay; do not inherit them.
    if "LD_PRELOAD" in env:
        del env["LD_PRELOAD"]
        record["set"].pop("LD_PRELOAD", None)
        record["removed"].append("LD_PRELOAD")
    return env, record


def linux_command(command, mask, timeout):
    # GNU timeout owns the solver process group, including OpenMP descendants.
    prefix = ["/usr/bin/time", "-f", "process_wall_sec %e\nmax_rss_kib %M\nexit_code %x",
              "-o", "process_wall.txt", "/usr/bin/timeout", "--signal=TERM", "--kill-after=10s", f"{timeout:g}s"]
    if mask:
        prefix += ["/usr/bin/taskset", mask]
    return ["bash", "-c", "ulimit -v 25165824 && exec " + shlex.join(prefix + command)]


def fingerprint(method, executable, library_root):
    expected = PINS["archived_runtime_sha256"][method]
    relevant = {name: value for name, value in expected.items()
                if not name.endswith(".exe") or name == executable.name}
    actual = {executable.name: core.sha256(executable)}
    for name in relevant:
        path = executable.parent / name
        if not path.is_file() and library_root:
            path = library_root / name
        if path.is_file():
            actual[name] = core.sha256(path)
    matches = executable.name in expected and all(actual.get(name) == value for name, value in relevant.items())
    return {"sha256": actual, "archived_bundle_match": matches,
            "scope": PINS["runtime_scope"],
            "identity": "archived bundle fingerprint" if matches else "rebuilt or unverified runtime; not an archived-binary replay"}


def require(actual, expected):
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"native policy mismatch: {key}={actual.get(key)!r}, expected {value!r}")


def number(text, label):
    values = re.findall(r"^" + re.escape(label) + r":\s*([-+0-9.eE]+)", text, re.M)
    if not values:
        raise ValueError(f"missing native timer/value: {label}")
    return float(values[-1])


def finite_tree(value):
    if isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError("nonfinite native result/history")


def load_scorer():
    spec = importlib.util.spec_from_file_location("baseline_rescore", HERE / "rescore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launcher_metrics(run_dir, elapsed, linux):
    result = {"launcher_wall_sec": elapsed}
    if linux:
        path = run_dir / "process_wall.txt"
        if path.is_file():
            for key, raw in re.findall(r"^(process_wall_sec|max_rss_kib|exit_code)\s+([-+0-9.eE]+)$", path.read_text(), re.M):
                value = float(raw)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("invalid GNU time metric")
                result["gnu_time_" + key] = value
        result["process_wall_sec"] = result.get("gnu_time_process_wall_sec")
    else:
        result["process_wall_sec"] = elapsed
    return result


def read_metrics(method, run_dir, dataset):
    if method in ("amm16", "cycle16"):
        log = (run_dir / "stdout.txt").read_text(encoding="utf-8", errors="replace")
        if not re.search(r"OMP_NUM_THREADS\s*=\s*['\"]?16\b", log):
            raise ValueError("missing observed OpenMP 16-worker environment")
        if method == "amm16":
            count = len(re.findall(r"^\d+: ", log, re.M))
            if count != POLICIES[method]["iterations"][dataset]:
                raise ValueError("incomplete AMM iteration budget")
            metrics = {"native_wall_sec": number(log, "measured algorithm wall"),
                       "setup_sec": number(log, "measured setup wall"),
                       "optimization_sec": number(log, "measured optimization wall"),
                       "native_chordal_objective": number(log, "final objective"), "iterations": count}
            if not math.isclose(metrics["native_wall_sec"], metrics["setup_sec"] + metrics["optimization_sec"], rel_tol=1e-5, abs_tol=0.002):
                raise ValueError("AMM native timing does not include setup + optimization")
        else:
            summary = (run_dir / "summary.txt").read_text(encoding="utf-8", errors="replace")
            match = re.search(r"time_overall \( cycle_basis: ([+\-0-9.eE]+) sec, ordering: ([+\-0-9.eE]+) sec, solve: ([+\-0-9.eE]+) sec \)\s+sum: ([+\-0-9.eE]+) sec", summary)
            process = re.search(r"time_process \( load data: ([+\-0-9.eE]+)\s+optimization: ([+\-0-9.eE]+)\s+dump data: ([+\-0-9.eE]+)\s+calculate vertices: ([+\-0-9.eE]+) \)", summary)
            count = re.search(r"n_iterations:\s*(\d+)", summary)
            if not match or not process or not count or not 1 <= int(count[1]) <= 20:
                raise ValueError("missing Cycle native timers or invalid iteration count")
            metrics = dict(zip(("cycle_basis_sec", "ordering_sec", "solve_sec", "native_wall_sec"), map(float, match.groups())))
            metrics.update(iterations=int(count[1]), optimization_sec=float(process[2]), vertex_reconstruction_sec=float(process[4]))
            metrics["optimization_and_vertices_sec"] = metrics["optimization_sec"] + metrics["vertex_reconstruction_sec"]
    else:
        data = core.load_json(run_dir / "result.json")
        finite_tree(data)
        cfg = data["configuration"]
        require(cfg, {"num_outer": 20, "huber_delta": 5})
        if method == "pcg16":
            require(cfg, {"requested_threads": 16, "observed_openmp_team": 16, "g2o_openmp": True,
                          "pcg_kernel_parallel": False, "pcg_tolerance": 1e-6, "pcg_max_iterations": -1,
                          "pcg_absolute_tolerance": False, "diagonal_jitter": 0})
            if data["returned_iterations"] != 20 or len(data["iterations"]) != 20:
                raise ValueError("incomplete PCG outer budget")
            metrics = {"native_wall_sec": data["timing"]["solver_wall_sec"], "iterations": 20,
                       "native_raw_objective": data["native_objective"]["final_raw"],
                       "native_huber_objective": data["native_objective"]["final_huber"]}
        else:
            history = data["outer_iterations"]
            if method == "cholmod1":
                require(cfg, {"threads": 1, "symbolic_reuse": True, "block_ordering": True,
                              "diagonal_jitter": 1e-10, "early_outer_exit": False})
                if len(history) != 20:
                    raise ValueError("incomplete CHOLMOD outer budget")
                native = data["timing"]["setup_excluding_parse_sec"] + data["timing"]["solve_sec"]
            else:
                require(cfg, {"threads": 16, "preconditioner": "two_level_additive_schwarz", "schwarz_threads": 16,
                              "schwarz_subdomains": 16, "schwarz_overlap_layers": 1, "schwarz_local_shift": 1e-10,
                              "pcg_max_iterations": -1, "pcg_tolerance": 1e-8, "diagonal_jitter": 1e-10,
                              "line_search": True, "paper_gradient_stopping": True,
                              "gradient_absolute_tolerance": 1e-8, "gradient_relative_tolerance": 1e-6})
                if not 0 <= len(history) <= 20:
                    raise ValueError("invalid Schwarz outer budget")
                native = data["timing"]["total_sec"]
            metrics = {"native_wall_sec": native, "iterations": len(history),
                       "native_raw_objective": data["final_objective"]["raw"],
                       "native_huber_objective": data["final_objective"]["huber"],
                       "solve_sec": data["timing"]["solve_sec"]}
    for key, value in metrics.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid native metric: {key}")
    if metrics["native_wall_sec"] <= 0:
        raise ValueError("native wall time must be positive")
    return metrics


def run(args):
    policy = POLICIES[args.method]
    linux = policy["platform"] == "linux"
    if linux and os.name == "nt" and not args.dry_run:
        raise ValueError("Run AMM/Cycle with Python inside Linux/WSL and Linux source/binary/data paths")
    manifest = core.load_json(MANIFEST_PATH)["datasets"]
    names = args.datasets or [name for name, item in manifest.items() if item["suite"] == "pgo"]
    if any(name not in manifest or manifest[name]["suite"] != "pgo" for name in names):
        raise ValueError("only formal-nine PGO dataset names are supported")
    out = args.output_root.resolve()
    if out.exists():
        raise ValueError("--output-root must be a new directory")
    data_root = (args.data_root or Path(os.environ.get("HGBP_PGO_DATA_ROOT") or ROOT / "data/pgo")).resolve()
    binary_root = args.binary_root.resolve()
    library_root = (args.library_root or binary_root).resolve() if linux else None
    if library_root and not library_root.is_dir():
        raise ValueError("Linux library root is not a directory")
    mask = None if args.no_affinity else args.affinity_mask
    env, env_record = environment_for(policy, library_root)
    plan = []
    fingerprints = {}
    for name in names:
        item = manifest[name]
        input_path = data_root / item["path"]
        digest = core.sha256(input_path)
        if digest != item["sha256"]:
            raise ValueError(f"{name}: canonical SHA-256 mismatch")
        executable = binary_root / (policy["executables"][item["space"]] + (".exe" if os.name == "nt" and not linux else ""))
        if executable not in fingerprints:
            fingerprints[executable] = fingerprint(args.method, executable, library_root)
        identity = fingerprints[executable]
        if args.require_archived_runtime and not identity["archived_bundle_match"]:
            raise ValueError(f"archived runtime mismatch: {executable}")
        for repeat in range(1, args.repeats + 1):
            run_dir = out / f"{name}_r{repeat}"
            command = solver_command(args.method, name, item["space"], executable, input_path, run_dir)
            historical = policy.get("historically_unmeasured", {}).get(name)
            record = {"dataset": name, "space": item["space"], "method": args.method, "method_label": policy["label"],
                      "repeat": repeat, "input": str(input_path), "input_sha256": digest, "runtime": identity,
                      "run_dir": str(run_dir), "command": command, "environment": env_record, "affinity_mask": mask,
                      "launch_command": linux_command(command, mask, args.timeout) if linux else command,
                      "historical_exception": historical, "status": "historically_unmeasured" if historical and not args.include_unmeasured else "planned",
                      "shared_cost_status": "pending independent final-pose rescore"}
            plan.append(record)
    out.mkdir(parents=True, exist_ok=False)
    core.write_json(out / "protocol.json", {"driver_sha256": core.sha256(Path(__file__)), "policy": policy,
                    "policy_sha256": core.sha256(HERE / "policies.json"), "manifest_sha256": core.sha256(MANIFEST_PATH),
                    "python": sys.version, "host": sys.platform, "dry_run": args.dry_run, "timeout_sec": args.timeout,
                    "address_space_limit_gib": 24 if linux else None, "timing": policy["timing"],
                    "note": "Sequential execution; no cross-machine speed pass/fail; native costs are labeled by objective."})
    core.write_json(out / "runs.json", plan)
    if args.dry_run:
        for record in plan:
            print(record["status"], json.dumps(record["launch_command"]))
        return 0
    # Fail before launching a solver if NumPy/SciPy or the retained scorer is unavailable.
    scorer = load_scorer()
    with core.process_affinity(None if linux else mask):
        for record in plan:
            if record["status"] == "historically_unmeasured":
                continue
            run_dir = Path(record["run_dir"])
            run_dir.mkdir()
            record["status"] = "running"
            core.write_json(out / "runs.json", plan)
            start = time.perf_counter()
            try:
                with (run_dir / "stdout.txt").open("w", encoding="utf-8") as log:
                    process = subprocess.run(record["launch_command"], cwd=run_dir, env=env, stdout=log,
                                             stderr=subprocess.STDOUT, check=False, timeout=args.timeout + 30 if linux else args.timeout,
                                             creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                record.update(launcher_metrics(run_dir, time.perf_counter() - start, linux), returncode=process.returncode)
                if process.returncode:
                    record["status"] = "timeout" if linux and process.returncode == 124 else "failed"
                else:
                    record["metrics"] = read_metrics(args.method, run_dir, record["dataset"])
                    scores = scorer.score_run(args.method, Path(record["input"]), record["space"], run_dir, record["input_sha256"])
                    core.write_json(run_dir / "independent_scores.json", scores)
                    for native_key, score_key in (("native_raw_objective", "raw_objective"), ("native_huber_objective", "robust_objective")):
                        if native_key in record["metrics"]:
                            native = record["metrics"][native_key]
                            if abs(scores["scores"][score_key] - native) > 1e-8 * max(abs(native), 1e-6):
                                raise ValueError("independent score disagrees with matched native objective")
                    record["independent_scores"] = scores
                    record["shared_cost_status"] = "independently rescored on canonical original factors"
                    record["status"] = "ok"
            except subprocess.TimeoutExpired:
                record.update(status="timeout", **launcher_metrics(run_dir, time.perf_counter() - start, linux))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                record.update(status="failed", error=str(exc))
            finally:
                record["output_sha256"] = {p.name: core.sha256(p) for p in run_dir.iterdir() if p.is_file()}
                core.write_json(out / "runs.json", plan)
            print(record["dataset"], record["status"], flush=True)
    return 2 if any(r["status"] not in ("ok", "historically_unmeasured") for r in plan) else 0


def main(argv=None):
    try:
        return run(parse_args(argv))
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
