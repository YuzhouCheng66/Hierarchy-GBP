"""Validate all outer trajectories and summarize interleaved full-solver timings."""
import argparse
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compare(cpu_path, gpu_path):
    cpu, gpu = read(cpu_path), read(gpu_path)
    pose_path = lambda p: Path(p).with_name(Path(p).stem + "_poses.json")
    return compare_payload(cpu, gpu, read(pose_path(cpu_path)), read(pose_path(gpu_path)))


def compare_payload(cpu, gpu, p, q):
    a, b = cpu["mg_history"], gpu["mg_history"]
    if len(a) != 21 or len(b) != 21:
        raise ValueError("Both trajectories must contain outer 0..20")
    if any(not isinstance(row["nonlinear_objective"], (int, float)) or
           not math.isfinite(row["nonlinear_objective"]) for row in a+b):
        raise ValueError("Nonfinite or missing objective")
    keys = ["outer", "execution_path", "num_groups", "coarse_dim", "coarse_lambda_nnz",
            "gbp_sweeps_executed", "jacobi_sweeps_executed", "coarse_solves",
            "full_precision_sweeps", "eta_only_sweeps", "precision_checks",
            "precision_freezes", "precision_thaws", "first_precision_freeze_sweep"]
    mismatches = [f"outer={i}:{key}" for i, (x, y) in enumerate(zip(a, b))
                  for key in keys if x[key] != y[key]]
    cost_error = max(abs(x["nonlinear_objective"] - y["nonlinear_objective"]) /
                     max(abs(x["nonlinear_objective"]), 1e-20) for x, y in zip(a, b))
    if p["num_poses"] != q["num_poses"]:
        raise ValueError("Pose count mismatch")
    ph, qh = p["mg_pose_history"], q["mg_pose_history"]
    if len(ph) != len(qh) or len(ph) < 20:
        raise ValueError("Missing outer poses")
    pose_rel = []
    for x, y in zip(ph, qh):
        if len(x) != len(y):
            raise ValueError("Missing poses")
        if any(len(pose) != 3 or not all(math.isfinite(v) for v in pose) for pose in x+y):
            raise ValueError("Invalid SE2 pose")
        diffs = [math.remainder(aa-bb, 2*math.pi) if j == 2 else aa-bb
                 for xx, yy in zip(x, y) for j, (aa, bb) in enumerate(zip(xx, yy))]
        den = max(1., math.sqrt(math.fsum(v*v for pose in x for v in pose)))
        pose_rel.append(math.sqrt(math.fsum(v*v for v in diffs))/den)
    finite = all(math.isfinite(v) for v in [cost_error, *pose_rel])
    passed = finite and not mismatches and cost_error < 1e-6 and max(pose_rel) < 1e-6
    return dict(passed=passed, policy_mismatches=mismatches,
                max_outer_cost_relative_error=cost_error,
                final_cost_relative_error=abs(a[-1]["nonlinear_objective"]-b[-1]["nonlinear_objective"])/
                    max(abs(a[-1]["nonlinear_objective"]),1e-20),
                max_outer_pose_relative_error=max(pose_rel),
                final_pose_relative_error=pose_rel[-1])


def summarize(directory):
    runs = read(directory / "runs.json")
    if isinstance(runs, dict):
        runs = [runs]
    rows = []
    for dataset in sorted({r["dataset"] for r in runs}):
        selected = [r for r in runs if r["dataset"] == dataset and r["repeat"] > 0 and not r["shadow"]]
        cpu = [r for r in selected if r["backend"] == "reference"]
        gpu = [r for r in selected if r["backend"] == "gpu"]
        if not cpu or len(cpu) != len(gpu):
            raise ValueError(f"Missing paired timing samples for {dataset}")
        checks = [compare(a["json"], b["json"]) for a, b in zip(cpu, gpu)]
        trajectory_passed = all(x["passed"] for x in checks)
        shadow = [r for r in runs if r["dataset"] == dataset and r["shadow"]]
        if len(shadow) != 1 or shadow[0]["lower"]["calls"] != 60:
            raise ValueError("Require a normal batch-shadow run (not single-sweep replay)")
        med = lambda xs: statistics.median(xs)
        cpu_ms, gpu_ms = med(r["solver_s"]*1000 for r in cpu), med(r["solver_s"]*1000 for r in gpu)
        profile = {}
        fields = ["sweeps_sec", "basis_build_sec", "coarse_lambda_sec", "coarse_eta_sec",
                  "coarse_solve_sec", "prolong_inject_sec", "factor_relinearize_sec", "objective_eval_sec"]
        for backend, records in [("cpu", cpu), ("gpu", gpu)]:
            profile[backend] = {field: med(sum(x[field] for x in read(r["json"])["mg_history"])*1000
                                           for r in records) for field in fields}
        lower_fraction = profile["cpu"]["sweeps_sec"]/cpu_ms
        rows.append(dict(dataset=dataset, accepted_speedup=trajectory_passed,
            status="passed" if trajectory_passed else "PROVISIONAL: strict outer-cost gate failed",
            repeats=len(cpu), cpu_ms=cpu_ms, gpu_ms=gpu_ms,
            gpu_over_cpu=gpu_ms/cpu_ms, speedup=cpu_ms/gpu_ms,
            cpu_ms_range=[min(r["solver_s"] for r in cpu)*1000,max(r["solver_s"] for r in cpu)*1000],
            gpu_ms_range=[min(r["solver_s"] for r in gpu)*1000,max(r["solver_s"] for r in gpu)*1000],
            cpu_process_ms=med(r["process_s"]*1000 for r in cpu),
            gpu_process_ms=med(r["process_s"]*1000 for r in gpu),
            cpu_cost=cpu[0]["cost"], gpu_cost=gpu[0]["cost"],
            full_sweeps=gpu[0]["full_sweeps"],eta_sweeps=gpu[0]["eta_sweeps"],
            shadow_max_relative=shadow[0]["lower"]["shadow_max_relative"],
            validation=checks[0],phase_medians_ms=profile,
            lower_fraction=lower_fraction, infinite_lower_speedup_bound=1/(1-lower_fraction),
            cuda_lower={key:med(r["lower"][key] for r in gpu) for key in
                ["wall_ms","upload_ms","kernel_ms","download_ms"]}))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if all(row["accepted_speedup"] for row in result) else 1)
