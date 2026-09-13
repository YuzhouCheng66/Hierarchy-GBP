"""Offline, scale-fixed EuRoC trajectory scoring; Python standard library only.

GT never enters the estimator. Both inputs describe Hamilton IMU-to-world
orientations. GT positions are interpolated linearly and orientations by
shortest-arc SLERP at exact estimate timestamps; extrapolation is forbidden.
"""

from __future__ import annotations
import argparse
import bisect
import csv
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
import math
from pathlib import Path
import statistics


@dataclass
class Pose:
    ns: int
    p: tuple
    q: tuple  # Hamilton x,y,z,w; maps IMU vectors into world


def dot(a, b):
    return math.fsum(x * y for x, y in zip(a, b))


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def scale(a, s):
    return tuple(s * x for x in a)


def norm(a):
    return math.sqrt(max(0.0, dot(a, a)))


def transpose(a):
    return list(map(list, zip(*a)))


def mm(a, b):
    return [[dot(row, col) for col in zip(*b)] for row in a]


def mv(a, b):
    return tuple(dot(row, b) for row in a)


def identity(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def normalize(q):
    length = norm(q)
    if not length > 1e-12 or not all(math.isfinite(x) for x in q):
        raise ValueError("invalid quaternion")
    return scale(q, 1.0 / length)


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def qinv(q):
    return (-q[0], -q[1], -q[2], q[3])


def qmatrix(q):
    x, y, z, w = normalize(q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def qangle(q):
    q = normalize(q)
    # atan2 is accurate near zero and sign-invariant on SO(3).
    return 2 * math.atan2(norm(q[:3]), abs(q[3]))


def slerp(a, b, alpha):
    cosine = dot(a, b)
    if cosine < 0:
        b = scale(b, -1)
        cosine = -cosine
    cosine = min(1.0, max(-1.0, cosine))
    if cosine > 0.9995:
        return normalize(add(scale(a, 1 - alpha), scale(b, alpha)))
    theta = math.acos(cosine)
    denominator = math.sin(theta)
    return normalize(
        add(
            scale(a, math.sin((1 - alpha) * theta) / denominator),
            scale(b, math.sin(alpha * theta) / denominator),
        )
    )


def unique_order(poses, name):
    poses.sort(key=lambda pose: pose.ns)
    if len(poses) < 3:
        raise ValueError(f"{name}: fewer than three poses")
    if any(a.ns == b.ns for a, b in zip(poses, poses[1:])):
        raise ValueError(f"{name}: duplicate timestamps")
    return poses


def to_ns(seconds):
    return int(
        (Decimal(seconds) * Decimal(1000000000)).to_integral_value(
            rounding=ROUND_HALF_EVEN
        )
    )


def read_estimate(path):
    poses = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise ValueError(f"{path}:{number}: expected TUM 8 columns")
        values = [float(v) for v in fields[1:]]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{path}:{number}: nonfinite estimate")
        poses.append(
            Pose(to_ns(fields[0]), tuple(values[:3]), normalize(tuple(values[3:])))
        )
    return unique_order(poses, str(path))


def read_gt(path):
    poses = []
    with path.open(newline="") as stream:
        for number, fields in enumerate(csv.reader(stream), 1):
            if not fields or fields[0].strip().startswith("#"):
                continue
            if len(fields) != 17:
                raise ValueError(f"{path}:{number}: expected official ASL 17 columns")
            values = [float(v) for v in fields[1:8]]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("nonfinite GT")
            # ASL: timestamp, p_xyz, q_wxyz, velocity, biases.
            poses.append(
                Pose(
                    int(fields[0]),
                    tuple(values[:3]),
                    normalize((values[4], values[5], values[6], values[3])),
                )
            )
    return unique_order(poses, str(path))


def interpolate(poses, times, timestamp, max_gap_ns):
    index = bisect.bisect_left(times, timestamp)
    if index < len(times) and times[index] == timestamp:
        return poses[index], 0, None
    if index == 0 or index == len(times):
        return None, None, "outside_gt_time_range"
    before, after = poses[index - 1], poses[index]
    gap = after.ns - before.ns
    if gap > max_gap_ns:
        return None, gap, "gt_bracket_gap_exceeds_limit"
    alpha = (timestamp - before.ns) / gap
    return (
        Pose(
            timestamp,
            add(scale(before.p, 1 - alpha), scale(after.p, alpha)),
            slerp(before.q, after.q, alpha),
        ),
        gap,
        None,
    )


def symmetric_eigen(a):
    """Jacobi diagonalization of a tiny real symmetric matrix (4x4 here)."""
    a = [list(row) for row in a]
    n = len(a)
    vectors = identity(n)
    for _ in range(128):
        p, q = max(
            ((i, j) for i in range(n) for j in range(i + 1, n)),
            key=lambda ij: abs(a[ij[0]][ij[1]]),
        )
        if abs(a[p][q]) <= 1e-15 * max(1.0, max(abs(a[i][i]) for i in range(n))):
            break
        angle = 0.5 * math.atan2(2 * a[p][q], a[q][q] - a[p][p])
        c, s = math.cos(angle), math.sin(angle)
        app, aqq, apq = a[p][p], a[q][q], a[p][q]
        for k in range(n):
            if k in (p, q):
                continue
            akp, akq = a[k][p], a[k][q]
            a[k][p] = a[p][k] = c * akp - s * akq
            a[k][q] = a[q][k] = s * akp + c * akq
        a[p][p] = c * c * app - 2 * c * s * apq + s * s * aqq
        a[q][q] = s * s * app + 2 * c * s * apq + c * c * aqq
        a[p][q] = a[q][p] = 0.0
        for k in range(n):
            vkp, vkq = vectors[k][p], vectors[k][q]
            vectors[k][p] = c * vkp - s * vkq
            vectors[k][q] = s * vkp + c * vkq
    else:
        raise ValueError("4x4 alignment eigensolver did not converge")
    order = sorted(range(n), key=lambda i: a[i][i], reverse=True)
    return [a[i][i] for i in order], [
        tuple(vectors[k][i] for k in range(n)) for i in order
    ]


def alignment(est, gt, mode):
    size = len(est)
    ce = tuple(statistics.fmean(pose.p[k] for pose in est) for k in range(3))
    cg = tuple(statistics.fmean(pose.p[k] for pose in gt) for k in range(3))
    x, y = [sub(p.p, ce) for p in est], [sub(p.p, cg) for p in gt]
    # Cross covariance H=sum x*y^T. Horn's unit quaternion maximizes y^T R x.
    h = [
        [math.fsum(a[i] * b[j] for a, b in zip(x, y)) / size for j in range(3)]
        for i in range(3)
    ]
    if mode == "se3":
        xx, xy, xz = h[0]
        yx, yy, yz = h[1]
        zx, zy, zz = h[2]
        horn = [
            [xx + yy + zz, yz - zy, zx - xz, xy - yx],
            [yz - zy, xx - yy - zz, xy + yx, zx + xz],
            [zx - xz, xy + yx, -xx + yy - zz, yz + zy],
            [xy - yx, zx + xz, yz + zy, -xx - yy + zz],
        ]
        values, vectors = symmetric_eigen(horn)
        if values[0] - values[1] <= 1e-12 * max(1.0, abs(values[0])):
            raise ValueError(
                "SE3 position alignment is degenerate (insufficient spatial excitation)"
            )
        w, qx, qy, qz = vectors[0]
        q = normalize((qx, qy, qz, w))
        diagnostics = dict(horn_eigenvalue_gap=values[0] - values[1])
    elif mode == "yaw4dof":
        cosine, sine = h[0][0] + h[1][1], h[0][1] - h[1][0]
        if math.hypot(cosine, sine) < 1e-12:
            raise ValueError(
                "yaw alignment is degenerate (insufficient horizontal excitation)"
            )
        yaw = math.atan2(sine, cosine)
        q = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        diagnostics = dict(yaw_rad=yaw, horizontal_correlation=math.hypot(cosine, sine))
    else:
        raise ValueError("unknown alignment mode")
    rotation = qmatrix(q)
    translation = sub(cg, mv(rotation, ce))
    return (
        q,
        translation,
        dict(rotation=rotation, translation=translation, scale=1.0, **diagnostics),
    )


def stats(values):
    if not values:
        return dict(count=0, rmse=None, mean=None, median=None, p95=None, maximum=None)
    ordered = sorted(values)
    p = (len(values) - 1) * 0.95
    lo = int(p)
    hi = min(lo + 1, len(values) - 1)
    return dict(
        count=len(values),
        rmse=math.sqrt(statistics.fmean(v * v for v in values)),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p95=ordered[lo] + (p - lo) * (ordered[hi] - ordered[lo]),
        maximum=max(values),
    )


def audit_runner_states(path, trajectory):
    with path.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["initialized"])]
    if len(rows) != len(trajectory):
        raise ValueError("states/trajectory initialized row count mismatch")
    clock, addition, position, rotation, dt = [], [], [], [], []
    for row, pose in zip(rows, trajectory):
        clock.append(abs(to_ns(row["imu_time"]) - pose.ns) * 1e-9)
        addition.append(
            abs(
                to_ns(row["imu_time"])
                - to_ns(row["state_time"])
                - to_ns(row["dt_camimu"])
            )
            * 1e-9
        )
        p = tuple(float(row[key]) for key in ("px", "py", "pz"))
        q = normalize(tuple(float(row[key]) for key in ("qx", "qy", "qz", "qw")))
        position.append(norm(sub(p, pose.p)))
        rotation.append(qangle(qmul(qinv(q), pose.q)))
        dt.append(float(row["dt_camimu"]))
    if (
        max(clock) > 1e-6
        or max(addition) > 1e-6
        or max(position) > 1e-10
        or max(rotation) > 1e-10
    ):
        raise ValueError(
            "runner states/trajectory convention or timestamp consistency failed"
        )
    return dict(
        initialized_rows=len(rows),
        trajectory_time_vs_imu_time_max_s=max(clock),
        imu_time_vs_state_time_plus_estimated_dt_max_s=max(addition),
        position_difference_max_m=max(position),
        quaternion_difference_max_rad=max(rotation),
        estimated_cam_to_imu_dt_min_s=min(dt),
        estimated_cam_to_imu_dt_max_s=max(dt),
        fitted_clock_offset_s=0.0,
        note="Checks runner output consistency; does not establish calibration accuracy against hardware truth",
    )


def rpe(est, gt, interval_s, max_est_gap_s, full_gt=None, max_gt_gap_s=0.020):
    reference = full_gt if full_gt is not None else gt
    times = [p.ns for p in est]
    gt_times = [p.ns for p in reference]
    position, rotation = [], []
    delta = to_ns(str(interval_s))
    max_gap = to_ns(str(max_est_gap_s))
    skipped = 0
    for e0, g0 in zip(est, gt):
        end = e0.ns + delta
        e1, _, err_e = interpolate(est, times, end, max_gap)
        g1, _, err_g = interpolate(reference, gt_times, end, to_ns(str(max_gt_gap_s)))
        if err_e or err_g:
            skipped += 1
            continue
        # Relative motions expressed in each initial IMU frame; global
        # alignment cancels and scale remains fixed at one.
        te = mv(transpose(qmatrix(e0.q)), sub(e1.p, e0.p))
        tg = mv(transpose(qmatrix(g0.q)), sub(g1.p, g0.p))
        re = qmul(qinv(e0.q), e1.q)
        rg = qmul(qinv(g0.q), g1.q)
        position.append(norm(sub(te, tg)))
        rotation.append(qangle(qmul(qinv(rg), re)))
    return dict(
        interval_s=interval_s,
        position_m=stats(position),
        rotation_rad=stats(rotation),
        skipped_endpoints=skipped,
        note="overlapping intervals are diagnostic samples, not independent trials",
    )


def score(
    estimate_path,
    gt_path,
    output_prefix,
    warmup_s=0.0,
    max_gt_gap_s=0.020,
    rpe_interval_s=1.0,
    max_est_gap_s=0.15,
    states_path=None,
):
    all_est, all_gt = read_estimate(estimate_path), read_gt(gt_path)
    gt_times = [pose.ns for pose in all_gt]
    start = all_est[0].ns + to_ns(str(warmup_s))
    max_gap = to_ns(str(max_gt_gap_s))
    est, gt, gaps, rejected = [], [], [], []
    for pose in all_est:
        if pose.ns < start:
            rejected.append(
                dict(
                    timestamp_ns=pose.ns, reason="requested_post_initialization_warmup"
                )
            )
            continue
        target, gap, reason = interpolate(all_gt, gt_times, pose.ns, max_gap)
        if reason:
            rejected.append(dict(timestamp_ns=pose.ns, reason=reason))
            continue
        est.append(pose)
        gt.append(target)
        gaps.append(gap)
    if len(est) < 3:
        raise ValueError("fewer than three valid interpolated estimate/GT pairs")
    rows = [
        dict(timestamp_ns=e.ns, time_s=e.ns * 1e-9, gt_bracket_gap_s=gap * 1e-9)
        for e, gap in zip(est, gaps)
    ]
    summary = dict(
        estimate=str(estimate_path.resolve()),
        groundtruth=str(gt_path.resolve()),
        sha256={
            "estimate": hashlib.sha256(estimate_path.read_bytes()).hexdigest(),
            "groundtruth": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
        },
        primary_alignment="yaw4dof",
        scale=1.0,
        groundtruth_role="offline scoring only; estimator initialization must be verified from its runner/provenance",
        input_estimates=len(all_est),
        matched_estimates=len(est),
        excluded_estimates=len(rejected),
        first_matched_ns=est[0].ns,
        last_matched_ns=est[-1].ns,
        matched_duration_s=(est[-1].ns - est[0].ns) * 1e-9,
        matching=dict(
            method="position linear / shortest-arc quaternion SLERP",
            extrapolation=False,
            max_gt_bracket_gap_s=max_gt_gap_s,
            maximum_actual_gap_s=max(gaps) * 1e-9,
            optimized_clock_shift_s=0.0,
            warmup_after_first_initialized_estimate_s=warmup_s,
        ),
        alignments={},
    )
    if states_path is not None:
        summary["clock_and_output_audit"] = audit_runner_states(states_path, all_est)
    for mode in ("yaw4dof", "se3"):
        q, t, transform = alignment(est, gt, mode)
        rotation = qmatrix(q)
        position_errors, rotation_errors = [], []
        for index, (e, g) in enumerate(zip(est, gt)):
            position = norm(sub(add(mv(rotation, e.p), t), g.p))
            orientation = qangle(qmul(qinv(g.q), qmul(q, e.q)))
            position_errors.append(position)
            rotation_errors.append(orientation)
            rows[index][mode + "_position_m"] = position
            rows[index][mode + "_rotation_rad"] = orientation
        summary["alignments"][mode] = dict(
            transform=transform,
            position_m=stats(position_errors),
            rotation_rad=stats(rotation_errors),
            fit_scope="all matched scored positions, no orientation fit",
        )
    summary["rpe"] = rpe(est, gt, rpe_interval_s, max_est_gap_s, all_gt, max_gt_gap_s)
    # Direction audit only, never used to choose/modify reported orientation.
    inverted = [Pose(e.ns, e.p, qinv(e.q)) for e in est]
    summary["quaternion_direction_diagnostic"] = dict(
        intended_hamilton_imu_to_world_rotation_rpe_rmse_rad=summary["rpe"][
            "rotation_rad"
        ]["rmse"],
        deliberately_inverted_estimate_rotation_rpe_rmse_rad=rpe(
            inverted, gt, rpe_interval_s, max_est_gap_s, all_gt, max_gt_gap_s
        )["rotation_rad"]["rmse"],
        note="diagnostic comparison; no convention selection or time-offset fitting",
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, table in (("_errors.csv", rows), ("_excluded.csv", rejected)):
        fields = list(table[0]) if table else ["timestamp_ns", "reason"]
        with Path(str(output_prefix) + suffix).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(table)
    Path(str(output_prefix) + "_score.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimate", type=Path, required=True)
    parser.add_argument("--groundtruth", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument(
        "--states",
        type=Path,
        help="optional runner states CSV for independent clock/quaternion output audit",
    )
    parser.add_argument("--warmup-s", type=float, default=0.0)
    parser.add_argument("--max-gt-gap-s", type=float, default=0.020)
    parser.add_argument("--rpe-interval-s", type=float, default=1.0)
    parser.add_argument("--max-est-gap-s", type=float, default=0.15)
    args = parser.parse_args()
    if (
        args.warmup_s < 0
        or min(args.max_gt_gap_s, args.rpe_interval_s, args.max_est_gap_s) <= 0
    ):
        parser.error("invalid nonpositive interval / negative warmup")
    result = score(
        args.estimate,
        args.groundtruth,
        args.output_prefix,
        args.warmup_s,
        args.max_gt_gap_s,
        args.rpe_interval_s,
        args.max_est_gap_s,
        args.states,
    )
    print(
        json.dumps(
            dict(
                matched=result["matched_estimates"],
                excluded=result["excluded_estimates"],
                alignments=result["alignments"],
                rpe=result["rpe"],
                quaternion_direction=result["quaternion_direction_diagnostic"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
