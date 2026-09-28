"""Independent CPU16 accuracy and old-GPU equivalence are different gates."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

from compare import compare, read


def upper_gate(record):
    mode = record.get("device_upper")
    if not mode:
        return not str(record.get("variant", "")).startswith("cycle")
    expected = {"mean": (60, 0, 0, 0), "restrict": (60, 60, 0, 0),
                "resident": (60, 60, 60, 20)}
    if mode not in expected:
        return False
    upper = record.get("upper") or {}
    counts = tuple(upper.get(k, -1) for k in ("means", "restrictions", "injections", "finishes"))
    return counts == expected[mode] and all(
        0. <= upper.get(k, math.inf) < 1e-6 for k in ("mean_error", "rhs_error", "inject_error"))


def phase_gate(record):
    if not str(record.get("variant", "")).startswith("phase"):
        return True
    values = record.get("phase")
    if values is None and record.get("json"):
        path = Path(record["json"]).with_suffix(".log")
        if path.is_file():
            lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines()
                     if line.startswith("CUDA_PHASE_SUMMARY ")]
            if len(lines) == 1:
                values = {k: int(v) for k, v in re.findall(r"([a-z_]+)=(\d+)", lines[0])}
    if not isinstance(values, dict):
        return False
    if any(type(values.get(k)) is not int for k in ("batches", "finishes", "full_segments", "eta_segments")):
        return False
    return (values["batches"] == values["finishes"] == 60 and
            1 <= values["full_segments"] <= 6000 and 0 <= values["eta_segments"] <= 6000)


def summarize(directory, shadow_directory=None):
    records = read(directory / "runs.json")
    if shadow_directory is None:
        shadow_records = records
    else:
        directories = [shadow_directory] if isinstance(shadow_directory, Path) else shadow_directory
        shadow_records = [record for path in directories for record in read(path / "runs.json")]
    rows = []
    med = statistics.median
    fields = ["outer_total_sec", "basis_build_sec", "coarse_lambda_sec", "sweeps_sec",
              "coarse_eta_sec", "coarse_solve_sec", "prolong_inject_sec", "factor_relinearize_sec",
              "objective_eval_sec"]
    for dataset in sorted({r["dataset"] for r in records}):
        runs = [r for r in records if r["dataset"] == dataset and r["repeat"] > 0
                and not r["shadow"] and not r.get("sanitizer")
                and not r.get("profile_clock") and not r.get("nsight_systems")]
        index = {(r["variant"], r["repeat"]): r for r in runs}
        if len(index) != len(runs):
            raise ValueError("Duplicate run")
        cpu = sorted((r for r in runs if r["variant"] == "reference"), key=lambda r: r["repeat"])
        if not cpu:
            raise ValueError("CPU16 reference is required")
        for variant in sorted({r["variant"] for r in runs} - {"reference"}):
            group = sorted((r for r in runs if r["variant"] == variant), key=lambda r: r["repeat"])
            if [r["repeat"] for r in group] != [r["repeat"] for r in cpu]:
                raise ValueError("Unpaired benchmark")
            if any(r["exit_code"] or not math.isfinite(r["solver_ms"]) or r["solver_ms"] <= 0 for r in cpu+group):
                raise ValueError("Failed or invalid timing run")
            for sequence in (cpu, group):
                if len({r["sha256"] for r in sequence}) != 1:
                    raise ValueError("Executable changed during benchmark")
            for key in ("dataset_sha256", "reference_commit"):
                if len({r.get(key) for r in cpu+group}) != 1 or not cpu[0].get(key):
                    raise ValueError("Mismatched or missing provenance")
            validation = [compare(c["json"], g["json"]) for c, g in zip(cpu, group)]
            old_gpu = [compare(index["bridge", g["repeat"]]["json"], g["json"])
                       for g in group if ("bridge", g["repeat"]) in index]
            cpu_ms = med(r["solver_ms"] for r in cpu)
            gpu_ms = med(r["solver_ms"] for r in group)
            shadow = [r for r in shadow_records if r["dataset"] == dataset and r["variant"] == variant and r["shadow"]]
            matched_shadow = bool(shadow) and all(
                s.get("sha256") == group[0]["sha256"] and
                s.get("dataset_sha256") == group[0].get("dataset_sha256") and
                s.get("reference_commit") == group[0].get("reference_commit") for s in shadow)
            # Timing equality with an earlier GPU is not certification against CPU.
            shadow_passed = matched_shadow and all(
                s["exit_code"] == 0 and s["lower"]["calls"] == 60 and
                s["lower"].get("fallback_batches", 1) == 0 and
                0. <= s["lower"].get("shadow_max_relative", math.inf) < 1e-6 and
                0. <= s["lower"].get("shadow_mean_max_relative", math.inf) < 1e-6
                for s in shadow)
            upper_passed = all(upper_gate(r) for r in group+shadow)
            phase_passed = all(phase_gate(r) for r in group+shadow)
            certified = all(v["passed"] for v in validation) and shadow_passed and upper_passed and phase_passed
            result = dict(dataset=dataset, variant=variant, repeats=len(group), cpu16_ms=cpu_ms,
                          candidate_ms=gpu_ms, candidate_over_cpu=gpu_ms/cpu_ms, speedup=cpu_ms/gpu_ms,
                          candidate_range_ms=[min(r["solver_ms"] for r in group), max(r["solver_ms"] for r in group)],
                          cpu_range_ms=[min(r["solver_ms"] for r in cpu), max(r["solver_ms"] for r in cpu)],
                          cpu_process_ms=med(r["process_ms"] for r in cpu),
                          cpu_cost=cpu[0]["cost"], candidate_cost=group[0]["cost"],
                          cpu_validation=validation[0], old_gpu_validation=old_gpu[0] if old_gpu else None,
                          all_cpu_outer_gates=all(v["passed"] for v in validation),
                          all_repeat_max_cost_error=max(v["max_outer_cost_relative_error"] for v in validation),
                          all_repeat_max_pose_error=max(v["max_outer_pose_relative_error"] for v in validation),
                          all_old_gpu_outer_gates=bool(old_gpu) and all(v["passed"] for v in old_gpu),
                          certified=certified, accepted_2x=certified and gpu_ms<=cpu_ms/2,
                          matched_shadow=matched_shadow, shadow_passed=shadow_passed,
                          upper_passed=upper_passed, phase_passed=phase_passed,
                          candidate_process_ms=med(r["process_ms"] for r in group),
                          cuda_context_ms=med(r.get("cuda_context_ms",0.) for r in group))
            result["process_over_cpu"] = result["candidate_process_ms"] / result["cpu_process_ms"]
            result["process_speedup"] = 1. / result["process_over_cpu"]
            result["accepted_wall_2x"] = certified and result["process_over_cpu"] <= .5
            result["phase_accounting"] = (
                "basis CPU work overlaps GPU batch 1; basis and sweeps/lower-wall fields overlap and must not be added"
                if group[0].get("overlap_basis") else "sequential host phase timers")
            if group[0].get("lower"):
                stage_modes = {r.get("lower_stage_timing_available", True) for r in group}
                if len(stage_modes) != 1:
                    raise ValueError("Stage timing mode changed during benchmark")
                result["lower_stage_timing_available"] = stage_modes.pop()
                result["lower_medians_ms"] = {
                    k:med(r["lower"][k] for r in group)
                    if k == "wall_ms" or result["lower_stage_timing_available"] else None
                    for k in ["wall_ms","kernel_ms","upload_ms","download_ms"]}
            for key, sequence in (("phase_medians_ms", group), ("cpu_phase_medians_ms", cpu)):
                payload = [read(r["json"]) for r in sequence]
                result[key] = {k:med(sum(x[k] for x in p["mg_history"])*1000
                                    for p in payload) for k in fields}
            trajectory = read(group[0]["json"])["mg_history"]
            result["work_counts"] = {k:sum(x[k] for x in trajectory) for k in
                ("gbp_sweeps_executed", "full_precision_sweeps", "eta_only_sweeps", "coarse_solves")}
            rows.append(result)
    return {"scope":"full outer20; solver includes allocation/transfers, context and process wall reported separately",
            "rows":rows}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    parser.add_argument("--shadow-directory",type=Path,action="append")
    args=parser.parse_args()
    result=summarize(args.directory, args.shadow_directory)
    (args.directory/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    for r in result["rows"]:
        print(f"{r['dataset']:6} {r['variant']:12} CPU16={r['cpu16_ms']:.3f} ms candidate={r['candidate_ms']:.3f} ms "
              f"ratio={r['candidate_over_cpu']:.4f} speedup={r['speedup']:.3f} "
              f"CPU_gate={r['all_cpu_outer_gates']} old_GPU_gate={r['all_old_gpu_outer_gates']} target_2x={r['accepted_2x']}")
