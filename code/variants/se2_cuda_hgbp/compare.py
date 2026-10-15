"""Shared CPU/GPU outer-trajectory validation; use compare_resident.py for reports."""
import json
import math
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


if __name__ == "__main__":
    raise SystemExit("Use compare_resident.py for the supported run-record schema.")
