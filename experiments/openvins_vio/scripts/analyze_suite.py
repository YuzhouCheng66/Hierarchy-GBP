#!/usr/bin/env python3
"""Verify recorded artifacts, apply the frozen admission rule, and report every cell."""

from pathlib import Path
import argparse, csv, hashlib, json, math, statistics
from score_euroc import read_estimate, qangle, qmul, qinv, norm, sub


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def finite_states(path):
    fields = (
        "px",
        "py",
        "pz",
        "qx",
        "qy",
        "qz",
        "qw",
        "vx",
        "vy",
        "vz",
        "bgx",
        "bgy",
        "bgz",
        "bax",
        "bay",
        "baz",
        "pos_var_x",
        "pos_var_y",
        "pos_var_z",
        "dt_camimu",
    )
    with path.open() as f:
        return all(
            all(math.isfinite(float(row[key])) for key in fields)
            for row in csv.DictReader(f)
            if row["initialized"] == "1"
        )


def write_json(p, x):
    p.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")


def load(suite):
    manifest = json.loads((suite / "manifest.json").read_text())
    assert manifest["complete"]
    records = json.loads((suite / "results.json").read_text())
    assert len(records) == len(manifest["job_order"])
    for record, job in zip(records, manifest["job_order"]):
        assert [record[k] for k in ("sequence", "clones", "method", "trial")] == job
        for name, expected in record["artifacts_sha256"].items():
            assert sha(suite / record["directory"] / name) == expected, (
                record["directory"],
                name,
            )
    return manifest, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    manifest, records = load(a.suite)
    am, audits = load(a.audit)
    assert manifest["phase"] == "formal" and am["phase"] == "audit"
    assert manifest["duration_s"] == am["duration_s"] == -1
    assert manifest["protocol_sha256"] == am["protocol_sha256"]
    assert manifest["configs_sha256"] == am["configs_sha256"]
    assert (
        manifest["builds"]["integrated"]["binary_sha256"]
        == am["builds"]["integrated"]["binary_sha256"]
    )
    for r in records:
        if r["returncode"] == 0 and r["method"] != "original":
            assert not r["backend"]["audit"] and r["backend"]["audit_events"] == 0
    cells = sorted({(r["sequence"], r["clones"]) for r in records})
    summaries = []
    trials = []
    for sequence, clones in cells:
        all_cell = [
            r for r in records if (r["sequence"], r["clones"]) == (sequence, clones)
        ]
        original = next(
            r for r in all_cell if r["method"] == "original" and r["trial"] == 1
        )
        base_ok = original["returncode"] == 0 and "scoring_error" not in original
        base_poses = (
            read_estimate(a.suite / original["directory"] / "vio_trajectory.txt")
            if base_ok
            else []
        )
        for method in ("original", "root_messages", "enhanced_ekf"):
            rs = sorted(
                [r for r in all_cell if r["method"] == method and r["trial"] > 0],
                key=lambda r: r["trial"],
            )
            assert len(rs) == 3
            checks = []
            poses = []
            for r in rs:
                ok = r["returncode"] == 0 and "scoring_error" not in r and base_ok
                detail = dict(trial=r["trial"], success=ok)
                if ok:
                    rr = r["run"]
                    br = original["run"]
                    s = r["score"]
                    bs = original["score"]
                    ep = read_estimate(a.suite / r["directory"] / "vio_trajectory.txt")
                    poses.append(ep)
                    timestamp_match = [p.ns for p in ep] == [p.ns for p in base_poses]
                    detail["coverage"] = timestamp_match and all(
                        rr[k] == br[k]
                        for k in (
                            "frames",
                            "initialized_frames",
                            "first_initialized_elapsed_s",
                            "imu_samples_fed",
                            "tail_pairs_without_future_imu",
                        )
                    )
                    detail["gt_coverage"] = all(
                        s[k] == bs[k]
                        for k in (
                            "matched_estimates",
                            "excluded_estimates",
                            "first_matched_ns",
                            "last_matched_ns",
                        )
                    )
                    detail["finite_states"] = finite_states(
                        a.suite / r["directory"] / "vio_states.csv"
                    )
                    detail["accuracy"] = {}
                    for name, value, base, floor in [
                        (
                            "ate_m",
                            s["alignments"]["yaw4dof"]["position_m"]["rmse"],
                            bs["alignments"]["yaw4dof"]["position_m"]["rmse"],
                            0.001,
                        ),
                        (
                            "rpe_m",
                            s["rpe"]["position_m"]["rmse"],
                            bs["rpe"]["position_m"]["rmse"],
                            0.001,
                        ),
                        (
                            "rpe_rad",
                            s["rpe"]["rotation_rad"]["rmse"],
                            bs["rpe"]["rotation_rad"]["rmse"],
                            math.radians(0.01),
                        ),
                    ]:
                        limit = base + max(floor, 0.05 * base)
                        detail["accuracy"][name] = dict(
                            value=value,
                            official=base,
                            limit=limit,
                            pass_=value <= limit,
                        )
                    ok = (
                        detail["coverage"]
                        and detail["gt_coverage"]
                        and detail["finite_states"]
                        and all(x["pass_"] for x in detail["accuracy"].values())
                    )
                    if timestamp_match:
                        detail["max_unaligned_position_difference_m"] = max(
                            norm(sub(x.p, y.p)) for x, y in zip(ep, base_poses)
                        )
                        detail["max_unaligned_orientation_difference_rad"] = max(
                            qangle(qmul(qinv(x.q), y.q)) for x, y in zip(ep, base_poses)
                        )
                    trials.append(
                        dict(
                            sequence=sequence,
                            clones=clones,
                            method=method,
                            trial=r["trial"],
                            process_wall_ms=r["process_wall_ms"],
                            replay_wall_ms=rr["replay_wall_ms"],
                            api_compute_ms=rr["total_camera_ms"]
                            + rr["total_imu_feed_ms"],
                            decode_ms=rr["total_decode_ms"],
                            ate_m=s["alignments"]["yaw4dof"]["position_m"]["rmse"],
                            rpe_m=s["rpe"]["position_m"]["rmse"],
                            rpe_rotation_deg=math.degrees(
                                s["rpe"]["rotation_rad"]["rmse"]
                            ),
                            admitted=ok,
                        )
                    )
                detail["admitted"] = ok
                checks.append(detail)
            audit = [
                r
                for r in audits
                if (r["sequence"], r["clones"], r["method"])
                == (sequence, clones, method)
            ]
            audit_ok = method == "original" or (
                len(audit) == 1
                and audit[0]["returncode"] == 0
                and "scoring_error" not in audit[0]
                and audit[0]["backend"]["audit"]
                and audit[0]["backend"]["audit_events"] > 0
                and audit[0]["backend"]["audit_gates"]
                == audit[0]["backend"]["features"]
            )
            rows = [
                r
                for r in trials
                if (r["sequence"], r["clones"], r["method"])
                == (sequence, clones, method)
            ]
            summary = dict(
                sequence=sequence,
                clones=clones,
                method=method,
                admitted=all(x["admitted"] for x in checks) and audit_ok,
                audit_pass=audit_ok,
                checks=checks,
                trajectory_bit_identical_across_trials=len(
                    {r["artifacts_sha256"].get("vio_trajectory.txt") for r in rs}
                )
                == 1,
            )
            if len(rows) == 3:
                for key in (
                    "process_wall_ms",
                    "replay_wall_ms",
                    "api_compute_ms",
                    "decode_ms",
                    "ate_m",
                    "rpe_m",
                    "rpe_rotation_deg",
                ):
                    summary[key] = dict(
                        median=statistics.median(r[key] for r in rows),
                        min=min(r[key] for r in rows),
                        max=max(r[key] for r in rows),
                        trials=[r[key] for r in rows],
                    )
            summaries.append(summary)
        base = next(
            s
            for s in summaries
            if (s["sequence"], s["clones"], s["method"])
            == (sequence, clones, "original")
        )
        for s in summaries:
            if (
                (s["sequence"], s["clones"]) == (sequence, clones)
                and "process_wall_ms" in s
                and "process_wall_ms" in base
            ):
                s["speedup_vs_official"] = {
                    key: base[key]["median"] / s[key]["median"]
                    for key in ("process_wall_ms", "replay_wall_ms", "api_compute_ms")
                }
    result = dict(
        protocol_sha256=manifest["protocol_sha256"],
        all_cells_admitted=all(s["admitted"] for s in summaries),
        cell_count=len(cells),
        formal_runs=len(records),
        audit_runs=len(audits),
        summaries=summaries,
    )
    candidates = [
        s
        for s in summaries
        if s["method"] == "root_messages" and "speedup_vs_official" in s
    ]
    result["candidate_speedup_ranges"] = (
        {
            key: [
                min(s["speedup_vs_official"][key] for s in candidates),
                max(s["speedup_vs_official"][key] for s in candidates),
            ]
            for key in ("process_wall_ms", "replay_wall_ms", "api_compute_ms")
        }
        if candidates
        else {}
    )
    write_json(out / "analysis.json", result)
    with (out / "trials.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trials[0]))
        w.writeheader()
        w.writerows(trials)
    lines = [
        "# Complete-sequence end-to-end results",
        "",
        f"Admission: {'PASS' if result['all_cells_admitted'] else 'FAIL / restricted claims'}; {len(cells)} cells, {len(records)} formal processes including warmups, {len(audits)} untimed audit processes.",
        "",
        "Median of three timed processes per cell/method; lower milliseconds is better. Process wall is the primary end-to-end measure. API excludes PNG decode, startup and output.",
        "",
        "| Sequence | Clones | Method | Process ms | Replay ms | API ms | Process speedup | ATE m | Admission |",
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for s in summaries:
        if "process_wall_ms" not in s:
            lines.append(
                f"| {s['sequence']} | {s['clones']} | {s['method']} | failure | | | | | FAIL |"
            )
            continue
        lines.append(
            f"| {s['sequence']} | {s['clones']} | {s['method']} | {s['process_wall_ms']['median']:.2f} | {s['replay_wall_ms']['median']:.2f} | {s['api_compute_ms']['median']:.2f} | {s['speedup_vs_official']['process_wall_ms']:.3f}x | {s['ate_m']['median']:.6f} | {'PASS' if s['admitted'] else 'FAIL'} |"
        )
    lines += [
        "",
        "The enhanced EKF is our supplementary control. This experiment does not establish a hierarchy-specific benefit or GPU performance. See analysis.json for all admission checks, exact trajectory discrepancies and every trial; trials.csv contains the compact measurements.",
        "",
    ]
    if candidates:
        ranges = result["candidate_speedup_ranges"]
        lines += [
            f"Candidate speedup versus official ranges from {ranges['process_wall_ms'][0]:.3f}x to {ranges['process_wall_ms'][1]:.3f}x for the complete process, and {ranges['api_compute_ms'][0]:.3f}x to {ranges['api_compute_ms'][1]:.3f}x for camera+IMU API compute.",
            "",
            "These are measurements of the specified ROS-free folder replay and configurations, not a claim about every OpenVINS application or device. The three processes per cell do not establish population confidence intervals.",
            "",
        ]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
