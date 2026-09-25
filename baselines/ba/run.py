"""Sequential measured BA profiles; Python stdlib only, no implicit builds/downloads."""

import argparse
from contextlib import contextmanager
import ctypes
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

HERE = Path(__file__).resolve().parent
RELEASE = HERE.parents[1]
PROFILES = json.loads((HERE / "profiles.json").read_text(encoding="utf-8"))
EVIDENCE = json.loads((HERE / "evidence.json").read_text(encoding="utf-8"))
THREAD_ENV = {
    "OMP_NUM_THREADS": "{threads}", "OMP_THREAD_LIMIT": "{threads}",
    "OMP_DYNAMIC": "FALSE", "OPENBLAS_NUM_THREADS": "{threads}",
    "MKL_NUM_THREADS": "{threads}", "OMP_WAIT_POLICY": "PASSIVE",
    "KMP_BLOCKTIME": "0",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                          encoding="utf-8")


def environment(threads, runtime_dirs):
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("GBP_", "HGBP_"))
           and k.upper() not in THREAD_ENV}
    env.update({k: v.format(threads=threads) for k, v in THREAD_ENV.items()})
    env["PATH"] = os.pathsep.join([*(str(p) for p in runtime_dirs),
                                   env.get("PATH", "")])
    if os.name != "nt":
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            [*(str(p) for p in runtime_dirs), env.get("LD_LIBRARY_PATH", "")])
    return env


def command(executable, run_dir, input_path, profile):
    common = PROFILES["common_arguments"]
    # Keep the measured argv order as well as its values. No dataset dispatch.
    return [str(executable), "-C", str(run_dir), "--input", str(input_path),
            *common[:2], "--num-threads", str(profile["threads"]), *common[2:]]


def finite_nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name}: expected a number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name}: nonfinite or negative")
    return value


def metrics(log, dataset, profile):
    info, solver = log["_static"]["problem_info"], log["_static"]["solver"]
    for key, expected in (("num_cameras", dataset["cameras"]),
                          ("num_landmarks", dataset["points"]),
                          ("num_observations", dataset["observations"])):
        if info[key] != expected:
            raise ValueError(f"problem shape mismatch: {key}")
    for key in ("num_threads_given", "num_threads_used"):
        if solver[key] != profile["threads"]:
            raise ValueError(f"thread mismatch: {key}")
    if solver["solver_type"] != profile["solver_type"]:
        raise ValueError("wrong solver type")
    if solver["termination_type"] not in (0, 1, 3):
        raise ValueError(f"solver failure termination: {solver['termination_type']}")
    json.dumps(solver, allow_nan=False)
    histories = [log[key] for key in ("iteration", "cost", "residual_block_mean")]
    if not histories[0] or len({len(h) for h in histories}) != 1:
        raise ValueError("empty or inconsistent histories")
    iterations, costs, means = histories
    if iterations != list(range(len(iterations))) or iterations[-1] > 20:
        raise ValueError("invalid iteration history or max20 budget exceeded")
    for values, name in ((costs, "cost"), (means, "residual_block_mean")):
        for value in values:
            finite_nonnegative(value, name)
    cost, initial = 2 * costs[-1], 2 * costs[0]
    finite_nonnegative(cost, "SSR")
    finite_nonnegative(initial, "initial SSR")
    rmse = math.sqrt(cost / dataset["observations"])
    if means[-1] > rmse + 1e-9:
        raise ValueError("MRE exceeds 2D-observation RMSE")
    return {"cost": cost, "initial_cost": initial, "rmse": rmse,
            "mre": means[-1], "actual_iterations": iterations[-1],
            "seconds": finite_nonnegative(solver["total_time_in_seconds"], "seconds"),
            "solver_details": solver}


@contextmanager
def affinity(mask):
    """Children inherit the requested mask; restore this process on exit."""
    if mask is None:
        yield
        return
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                                                  ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.POINTER(ctypes.c_size_t)]
        kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        handle = kernel.GetCurrentProcess()
        original, system = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(original),
                                              ctypes.byref(system)):
            raise OSError(ctypes.get_last_error(), "cannot read affinity")
        if mask & system.value != mask or not kernel.SetProcessAffinityMask(handle, mask):
            raise ValueError("requested CPU affinity unavailable")
        try:
            yield
        finally:
            if not kernel.SetProcessAffinityMask(handle, original.value):
                raise OSError(ctypes.get_last_error(), "cannot restore affinity")
    elif hasattr(os, "sched_getaffinity"):
        original = os.sched_getaffinity(0)
        requested = {i for i in range(mask.bit_length()) if mask & (1 << i)}
        if not requested <= original:
            raise ValueError("requested CPU affinity unavailable")
        os.sched_setaffinity(0, requested)
        try:
            yield
        finally:
            os.sched_setaffinity(0, original)
    else:
        raise ValueError("CPU affinity unsupported; explicitly use --affinity-mask none")


def execute(job, dataset, profile, env, timeout):
    folder = Path(job["result_dir"])
    folder.mkdir(parents=True, exist_ok=False)
    result = dict(job)
    start = time.perf_counter()
    try:
        with (folder / "stdout.txt").open("w", encoding="utf-8") as stream:
            proc = subprocess.run(job["command"], cwd=folder, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        result.update(returncode=proc.returncode, status="failed")
        if proc.returncode == 0:
            result["status"] = "validation_failed"
            log = json.loads((folder / "ba_log.json").read_text(encoding="utf-8"))
            result.update(metrics(log, dataset, profile), status="ok")
    except subprocess.TimeoutExpired:
        result.update(status="timeout", error=f"process exceeded {timeout} seconds")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        result.setdefault("status", "failed")
        result["error"] = str(error)
    result["process_seconds"] = time.perf_counter() - start
    return result


def summary(runs):
    rows = []
    for dataset, profile in dict.fromkeys((r["dataset"], r["profile"]) for r in runs):
        group = [r for r in runs if (r["dataset"], r["profile"]) == (dataset, profile)]
        good = [r for r in group if r["status"] == "ok"]
        row = {"dataset": dataset, "profile": profile, "successful": len(good),
               "failed": len(group) - len(good),
               "status": "ok" if len(good) == len(group) else "incomplete"}
        if good:
            for key in ("seconds", "process_seconds", "cost", "rmse", "mre"):
                row[key + "_median_successful"] = statistics.median(r[key] for r in good)
        rows.append(row)
    return rows


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary-root", type=Path, required=True)
    ap.add_argument("--runtime-dir", type=Path, action="append", required=True,
                    help="directory containing DLLs/shared libraries; repeat in search order")
    ap.add_argument("--data-root", type=Path, required=True, help="BA manifest path root")
    ap.add_argument("--manifest", type=Path, default=RELEASE / "configs/datasets.json")
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--profiles", nargs="+", choices=PROFILES["profiles"],
                    default=list(PROFILES["profiles"]))
    ap.add_argument("--output", type=Path, required=True, help="must not exist")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=2000)
    ap.add_argument("--affinity-mask", default="0xffff",
                    help="inherited CPU mask (measured: 0xffff), or explicit 'none'")
    ap.add_argument("--require-measured-binaries", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="validate and print plan; no writes/processes")
    return ap


def prepare(args):
    if args.repeats < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("positive repeats and finite positive timeout required")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.profiles)) != len(args.profiles):
        raise ValueError("duplicate datasets or profiles")
    output = args.output.resolve()
    if output.exists():
        raise ValueError("output already exists; stale logs/configs must not be reused")
    mask = None if args.affinity_mask.lower() == "none" else int(args.affinity_mask, 0)
    if mask is not None and (mask <= 0 or mask.bit_length() > 64):
        raise ValueError("affinity must be a nonzero mask of at most 64 CPUs")
    runtime = [p.resolve(strict=True) for p in args.runtime_dir]
    if any(not p.is_dir() for p in runtime):
        raise ValueError("runtime-dir must name a directory")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    data_root = args.data_root.resolve(strict=True)
    datasets = {}
    for name in args.datasets:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("unsafe dataset identifier")
        record = dict(manifest["datasets"][name])
        if record["suite"] != "ba":
            raise ValueError(f"{name} is not BA")
        path = (data_root / record["path"].replace("\\", "/")).resolve(strict=True)
        if not path.is_relative_to(data_root):
            raise ValueError("dataset path escapes data-root")
        if sha256(path) != record["sha256"].lower():
            raise ValueError(f"{name}: input SHA256 mismatch")
        for key in ("cameras", "points", "observations"):
            if type(record[key]) is not int or record[key] <= 0:
                raise ValueError(f"invalid dataset count: {key}")
        with path.open(encoding="utf-8") as stream:
            header = [int(v) for v in stream.readline().split()]
        if header != [record[k] for k in ("cameras", "points", "observations")]:
            raise ValueError(f"{name}: BAL header mismatch")
        datasets[name] = dict(record, resolved_path=str(path))
    binaries = {}
    for name in args.profiles:
        profile = PROFILES["profiles"][name]
        executable = (args.binary_root / (profile["binary"] + (".exe" if os.name == "nt" else ""))).resolve(strict=True)
        digest = sha256(executable)
        match = digest == EVIDENCE["executable_sha256"][profile["method"]]
        if args.require_measured_binaries and not match:
            raise ValueError(f"{name}: not the archived measured executable")
        binaries[name] = {"path": str(executable), "sha256": digest,
                          "matches_measured_executable": match}
    jobs = []
    for dataset in args.datasets:
        for repeat in range(args.repeats):
            # Rotate selected methods between repeats, as in the measured runner.
            offset = repeat % len(args.profiles)
            for name in args.profiles[offset:] + args.profiles[:offset]:
                profile = PROFILES["profiles"][name]
                folder = output / f"{dataset}_{name}_r{repeat}"
                jobs.append({"dataset": dataset, "profile": name, "repeat": repeat,
                             "threads": profile["threads"], "result_dir": str(folder),
                             "input_sha256": datasets[dataset]["sha256"],
                             "command": command(binaries[name]["path"], folder,
                                                datasets[dataset]["resolved_path"], profile)})
    libraries = {}
    for directory in dict.fromkeys([*runtime, *(Path(b["path"]).parent for b in binaries.values())]):
        for path in sorted(directory.iterdir()):
            if path.is_file() and (path.suffix.lower() in (".dll", ".dylib", ".so") or ".so." in path.name):
                libraries[str(path)] = sha256(path)
    plan = {"schema_version": 1, "dry_run": args.dry_run, "datasets": datasets,
            "profiles": PROFILES, "binaries": binaries, "runtime_libraries_sha256": libraries,
            "runtime_dirs": [str(p) for p in runtime], "jobs": jobs,
            "manifest_sha256": sha256(args.manifest), "runner_sha256": sha256(__file__),
            "profiles_sha256": sha256(HERE / "profiles.json"),
            "affinity_mask": None if mask is None else hex(mask),
            "affinity_applied": False, "timeout_process_seconds": args.timeout,
            "repeats": args.repeats, "platform": platform.platform(),
            "processor": platform.processor(), "python": sys.version,
            "environment": {name: {k: v for k, v in environment(PROFILES['profiles'][name]['threads'], runtime).items()
                                    if k in THREAD_ENV or k in ("PATH", "LD_LIBRARY_PATH", "TEMP", "TMP")}
                            for name in args.profiles},
            "metric_source": "native log, not independent final-geometry rescoring",
            "timing": "native solver total and separate process wall; see README for boundaries"}
    return plan, output, mask, runtime


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        plan, output, mask, runtime = prepare(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2, allow_nan=False))
            return 0
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "protocol.json", plan)
        runs, initial_costs = [], {}
        with affinity(mask):
            plan["affinity_applied"] = mask is not None
            write_json(output / "protocol.json", plan)
            for job in plan["jobs"]:
                profile = PROFILES["profiles"][job["profile"]]
                result = execute(job, plan["datasets"][job["dataset"]], profile,
                                 environment(profile["threads"], runtime), args.timeout)
                if result["status"] == "ok":
                    initial = initial_costs.setdefault(job["dataset"], result["initial_cost"])
                    if abs(result["initial_cost"] - initial) > 1e-8 * max(1, abs(initial)):
                        result.update(status="validation_failed", error="inconsistent initial objective")
                runs.append(result)
                write_json(Path(job["result_dir"]) / "result.json", result)
                write_json(output / "runs.json", runs)
                write_json(output / "summary.json", summary(runs))
        return 0 if all(r["status"] == "ok" for r in runs) else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"BA baseline runner: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
