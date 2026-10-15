from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Preset:
    name: str
    rows: int
    cols: int
    target_edges: int
    target_g2o_cost: float
    radius: float


PRESETS = {
    "globe1k": Preset(
        name="Globe1kSynth",
        rows=32,
        cols=32,
        target_edges=2140,
        target_g2o_cost=627.862012536485,
        radius=60.0,
    ),
    "globe10k": Preset(
        name="Globe10kSynth",
        rows=100,
        cols=100,
        target_edges=20899,
        target_g2o_cost=6131.63,
        radius=60.0,
    ),
    "globe100k": Preset(
        name="Globe100kSynth",
        rows=316,
        cols=316,
        target_edges=200395,
        target_g2o_cost=40254.50,
        radius=60.0,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic globe-like SE3 pose graph.")
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--out-g2o", required=True)
    parser.add_argument("--out-meta", default="")
    parser.add_argument(
        "--edge-mode",
        choices=["local_diag", "medium_chords", "mixed_chords", "hemisphere_bridges", "separator_weave"],
        default="local_diag",
        help="Controls how the extra edges beyond the base globe grid are distributed.",
    )
    parser.add_argument(
        "--edge-order",
        choices=["legacy_grid_first", "chain_first"],
        default="legacy_grid_first",
        help="Ordering of EDGE records in the output g2o.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--translation-info", type=float, default=400.0)
    parser.add_argument("--rotation-info", type=float, default=400.0)
    parser.add_argument(
        "--noise-scale-mult",
        type=float,
        default=1.0,
        help="Scales measurement noise relative to the paper-derived default.",
    )
    parser.add_argument(
        "--init-translation-amp",
        type=float,
        default=0.75,
        help="Amplitude of smooth initial-guess translation distortion, in meters.",
    )
    parser.add_argument(
        "--init-rotation-amp-deg",
        type=float,
        default=4.0,
        help="Amplitude of smooth initial-guess rotation distortion, in degrees.",
    )
    parser.add_argument(
        "--init-mode",
        choices=["smooth_distortion", "spanning_tree", "odometry_chain"],
        default="smooth_distortion",
        help="How VERTEX initial poses are generated.",
    )
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        return v.copy()
    return v / n


def normalize_rotation(r: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(r)
    out = u @ vt
    if np.linalg.det(out) < 0.0:
        u[:, -1] *= -1.0
        out = u @ vt
    return out


def rot_from_axis_angle(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    if theta < 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = v / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * k + (1.0 - math.cos(theta)) * (k @ k)


def quat_from_rot(r: np.ndarray) -> np.ndarray:
    r = normalize_rotation(r)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def make_pose(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = normalize_rotation(r)
    out[:3, 3] = t
    return out


def se3_inverse(t: np.ndarray) -> np.ndarray:
    r = t[:3, :3]
    p = t[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -(r.T @ p)
    return out


def se3_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return se3_inverse(a) @ b


def info21(info: np.ndarray) -> list[float]:
    vals: list[float] = []
    for row in range(6):
        for col in range(row, 6):
            vals.append(float(info[row, col]))
    return vals


def grid_id(row: int, col: int, rows: int, cols: int) -> int:
    if row % 2 == 0:
        return row * cols + col
    return row * cols + (cols - 1 - col)


def inverse_grid_id(idx: int, rows: int, cols: int) -> tuple[int, int]:
    row = idx // cols
    offset = idx % cols
    if row % 2 == 0:
        return row, offset
    return row, cols - 1 - offset


def generate_ground_truth(preset: Preset) -> list[np.ndarray]:
    rows = preset.rows
    cols = preset.cols
    poses: list[np.ndarray] = [np.eye(4, dtype=np.float64) for _ in range(rows * cols)]
    for row in range(rows):
        # Avoid the exact poles so the local tangent frame remains stable.
        lat = -0.5 * math.pi + (row + 0.5) * (math.pi / rows)
        for col in range(cols):
            lon = (2.0 * math.pi * col) / cols
            x = math.cos(lat) * math.cos(lon)
            y = math.cos(lat) * math.sin(lon)
            z = math.sin(lat)
            radius_scale = 1.0 + 0.02 * math.sin(3.0 * lon) * math.cos(2.0 * lat)
            center = preset.radius * radius_scale * np.array([x, y, z], dtype=np.float64)

            outward = normalize(center)
            east = normalize(np.array([-math.sin(lon), math.cos(lon), 0.0], dtype=np.float64))
            north = normalize(np.cross(outward, east))
            east = normalize(np.cross(north, outward))
            rot = np.column_stack((east, north, outward))
            poses[grid_id(row, col, rows, cols)] = make_pose(rot, center)
    return poses


def smooth_initial_distortion(gt_poses: list[np.ndarray], preset: Preset, t_amp: float, r_amp_deg: float) -> list[np.ndarray]:
    rows = preset.rows
    cols = preset.cols
    r_amp = math.radians(r_amp_deg)
    out: list[np.ndarray] = []
    for idx, gt in enumerate(gt_poses):
        row, col = inverse_grid_id(idx, rows, cols)
        lat = -0.5 * math.pi + (row + 0.5) * (math.pi / rows)
        lon = (2.0 * math.pi * col) / cols
        trans_delta = t_amp * np.array(
            [
                0.55 * math.sin(2.0 * lon) * math.cos(1.5 * lat),
                0.40 * math.cos(3.0 * lon - 0.4 * lat),
                0.35 * math.sin(2.2 * lat + 0.8 * lon),
            ],
            dtype=np.float64,
        )
        rot_delta = r_amp * np.array(
            [
                0.45 * math.sin(1.5 * lat),
                0.35 * math.cos(2.0 * lon + 0.2 * lat),
                0.25 * math.sin(1.7 * lon - 0.5 * lat),
            ],
            dtype=np.float64,
        )
        init = gt.copy()
        init[:3, 3] += trans_delta
        init[:3, :3] = normalize_rotation(init[:3, :3] @ rot_from_axis_angle(rot_delta))
        if idx == 0:
            init = gt.copy()
        out.append(init)
    return out


def build_extra_candidates(preset: Preset, edge_mode: str) -> list[tuple[int, int]]:
    rows = preset.rows
    cols = preset.cols
    candidates: list[tuple[int, int]] = []

    def push(row: int, col: int, d_row: int, d_col: int) -> None:
        if d_row == 0 and (d_col % cols) in (0, 1, cols - 1):
            return
        if d_row == 1 and (d_col % cols) == 0:
            return
        if row + d_row >= rows:
            return
        a = grid_id(row, col, rows, cols)
        b = grid_id(row + d_row, (col + d_col) % cols, rows, cols)
        if a == b:
            return
        candidates.append((a, b))

    if edge_mode == "local_diag":
        for row in range(rows - 1):
            for col in range(cols):
                push(row, col, 1, 1)
        return candidates

    if edge_mode == "mixed_chords":
        # A blend of medium- and long-range loop closures that keeps the total
        # edge count fixed but widens the elimination graph more aggressively.
        jumps = [
            (2, 5),
            (3, cols // 8 + 3),
            (rows // 8, cols // 6 + 1),
            (rows // 6, cols // 4 - 1),
            (rows // 4, cols // 3 + 5),
            (rows // 3, cols // 2 - 1),
        ]
        for d_row, d_col in jumps:
            for row in range(rows - d_row):
                row_stride = 1 + ((row + d_row) % 3)
                for col in range(0, cols, row_stride):
                    push(row, col, d_row, d_col)
        return candidates

    if edge_mode == "medium_chords":
        # A milder variant that still introduces non-local structure, but keeps
        # the separator growth between local_diag and mixed_chords.
        jumps = [
            (2, 3),
            (3, cols // 10 + 2),
            (rows // 10, cols // 8 + 1),
            (rows // 8, cols // 6 + 3),
            (rows // 7, cols // 7 + 4),
        ]
        for d_row, d_col in jumps:
            for row in range(rows - d_row):
                row_stride = 1 + ((2 * row + d_row) % 4)
                for col in range(0, cols, row_stride):
                    push(row, col, d_row, d_col)
        return candidates

    if edge_mode == "hemisphere_bridges":
        # Connect broad latitude bands across distant longitudes so the graph
        # stays globe-like but gains stronger cross-hemisphere separators.
        jumps = [
            (rows // 5, cols // 2 - 3),
            (rows // 4, cols // 3 + 7),
            (rows // 3, cols // 2 + 11),
            (rows // 2 - 1, cols // 4 + 9),
        ]
        for d_row, d_col in jumps:
            for row in range(rows - d_row):
                for col in range(cols):
                    if (row + 2 * col + d_row) % 7 == 0:
                        push(row, col, d_row, d_col)
        return candidates

    if edge_mode == "separator_weave":
        # Bias extra constraints toward block boundaries so direct factorization
        # sees larger separators while the geometry remains smooth.
        row_blocks = max(rows // 10, 1)
        col_blocks = max(cols // 10, 1)
        for row in range(rows):
            if row % row_blocks not in (0, row_blocks // 2, row_blocks - 1):
                continue
            for col in range(cols):
                if col % col_blocks not in (0, col_blocks // 2, col_blocks - 1):
                    continue
                for d_row, d_col in (
                    (rows // 8, cols // 5 + 3),
                    (rows // 6, cols // 3 + 1),
                    (rows // 4, cols // 2 - 5),
                    (rows // 3, cols // 4 + 7),
                ):
                    push(row, col, d_row, d_col)
        return candidates

    raise ValueError(f"unsupported edge mode: {edge_mode}")


def build_edges(preset: Preset, edge_mode: str) -> list[tuple[int, int]]:
    rows = preset.rows
    cols = preset.cols
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add_edge(a: int, b: int) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return
        seen.add(key)
        edges.append(key)

    # Horizontal ring edges.
    for row in range(rows):
        for col in range(cols):
            add_edge(grid_id(row, col, rows, cols), grid_id(row, (col + 1) % cols, rows, cols))

    # Vertical edges between adjacent latitudes.
    for row in range(rows - 1):
        for col in range(cols):
            add_edge(grid_id(row, col, rows, cols), grid_id(row + 1, col, rows, cols))

    base_edges = len(edges)
    extras_needed = preset.target_edges - base_edges
    if extras_needed < 0:
        raise ValueError(f"Preset {preset.name} requests fewer edges than the base globe grid")

    if extras_needed > 0:
        candidates = build_extra_candidates(preset, edge_mode)
        if len(candidates) < extras_needed:
            raise ValueError(
                f"edge mode {edge_mode} produced only {len(candidates)} candidates, need {extras_needed}"
            )
        picks = np.linspace(0, len(candidates) - 1, extras_needed, dtype=int)
        for idx in picks.tolist():
            add_edge(*candidates[idx])

    if len(edges) != preset.target_edges:
        raise ValueError(f"Failed to hit target edge count for {preset.name}: got {len(edges)}")
    return edges


def reorder_edges_chain_first(
    preset: Preset,
    edges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    rows = preset.rows
    cols = preset.cols
    n = rows * cols

    edge_set = set(edges)
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def push(a: int, b: int) -> None:
        key = (a, b) if a < b else (b, a)
        if key not in edge_set or key in seen:
            return
        seen.add(key)
        ordered.append(key)

    # Official synthetic g2o files put the odometry chain first. With the
    # snake ordering used by grid_id(), consecutive ids form a continuous path.
    for idx in range(n - 1):
        push(idx, idx + 1)

    # Then emit local non-chain loop closures: longitude wrap edges and the
    # remaining inter-row edges of the globe grid.
    for row in range(rows):
        push(grid_id(row, 0, rows, cols), grid_id(row, cols - 1, rows, cols))

    for row in range(rows - 1):
        for col in range(cols):
            push(grid_id(row, col, rows, cols), grid_id(row + 1, col, rows, cols))

    # Finally append any remaining extra edges introduced by edge_mode.
    for a, b in edges:
        push(a, b)

    if len(ordered) != len(edges):
        raise ValueError(f"Chain-first reordering lost edges: {len(ordered)} vs {len(edges)}")
    return ordered


def make_measurement(
    gt_i: np.ndarray,
    gt_j: np.ndarray,
    trans_sigma: float,
    rot_sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    meas = se3_between(gt_i, gt_j)
    meas[:3, 3] += rng.normal(0.0, trans_sigma, size=3)
    meas[:3, :3] = normalize_rotation(meas[:3, :3] @ rot_from_axis_angle(rng.normal(0.0, rot_sigma, size=3)))
    return meas


def spanning_tree_initialization(
    gt_poses: list[np.ndarray],
    edges: list[tuple[int, int]],
    measurements: list[np.ndarray],
) -> list[np.ndarray]:
    n = len(gt_poses)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for edge_idx, (i, j) in enumerate(edges):
        adjacency[i].append((edge_idx, j))
        adjacency[j].append((edge_idx, i))

    poses: list[np.ndarray | None] = [None] * n
    poses[0] = gt_poses[0].copy()
    frontier = [0]

    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            base_pose = poses[node]
            assert base_pose is not None
            for edge_idx, nbr in adjacency[node]:
                if poses[nbr] is not None:
                    continue
                i, j = edges[edge_idx]
                meas = measurements[edge_idx]
                if j == nbr and i == node:
                    poses[nbr] = base_pose @ meas
                elif i == nbr and j == node:
                    poses[nbr] = base_pose @ se3_inverse(meas)
                else:
                    raise RuntimeError("Inconsistent adjacency")
                poses[nbr][:3, :3] = normalize_rotation(poses[nbr][:3, :3])
                next_frontier.append(nbr)
        frontier = next_frontier

    unresolved = [idx for idx, pose in enumerate(poses) if pose is None]
    if unresolved:
        raise RuntimeError(f"Spanning-tree initialization failed, unresolved nodes: {len(unresolved)}")
    return [pose for pose in poses if pose is not None]


def odometry_chain_initialization(
    gt_poses: list[np.ndarray],
    measurements_by_edge: dict[tuple[int, int], np.ndarray],
) -> list[np.ndarray]:
    n = len(gt_poses)
    poses: list[np.ndarray] = [np.eye(4, dtype=np.float64) for _ in range(n)]
    poses[0] = gt_poses[0].copy()
    for idx in range(n - 1):
        key = (idx, idx + 1)
        if key not in measurements_by_edge:
            raise RuntimeError(f"Missing odometry-chain edge {key}")
        poses[idx + 1] = poses[idx] @ measurements_by_edge[key]
        poses[idx + 1][:3, :3] = normalize_rotation(poses[idx + 1][:3, :3])
    return poses


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    rng = np.random.default_rng(args.seed)

    gt_poses = generate_ground_truth(preset)
    edges = build_edges(preset, args.edge_mode)
    if args.edge_order == "chain_first":
        edges = reorder_edges_chain_first(preset, edges)

    n = preset.rows * preset.cols
    m = len(edges)
    chi2_target = preset.target_g2o_cost
    # Our solver uses 0.5 * e^T Omega e, while the table reports g2o chi2.
    objective_target = 0.5 * chi2_target
    dof = (6 * m + 6) - 6 * n
    normalized_scale = math.sqrt(max(chi2_target, 1e-12) / max(dof, 1))
    trans_sigma = args.noise_scale_mult * normalized_scale / math.sqrt(args.translation_info)
    rot_sigma = args.noise_scale_mult * normalized_scale / math.sqrt(args.rotation_info)

    info = np.zeros((6, 6), dtype=np.float64)
    info[:3, :3] = args.translation_info * np.eye(3, dtype=np.float64)
    info[3:, 3:] = args.rotation_info * np.eye(3, dtype=np.float64)

    measurements = [make_measurement(gt_poses[i], gt_poses[j], trans_sigma, rot_sigma, rng) for i, j in edges]
    measurements_by_edge = {(i, j): meas for (i, j), meas in zip(edges, measurements)}
    if args.init_mode == "smooth_distortion":
        init_poses = smooth_initial_distortion(gt_poses, preset, args.init_translation_amp, args.init_rotation_amp_deg)
    elif args.init_mode == "spanning_tree":
        init_poses = spanning_tree_initialization(gt_poses, edges, measurements)
    elif args.init_mode == "odometry_chain":
        init_poses = odometry_chain_initialization(gt_poses, measurements_by_edge)
    else:
        raise ValueError(f"Unsupported init mode: {args.init_mode}")

    out_path = Path(args.out_g2o)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for idx, pose in enumerate(init_poses):
            quat = quat_from_rot(pose[:3, :3])
            t = pose[:3, 3]
            fh.write(
                "VERTEX_SE3:QUAT "
                f"{idx} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f}\n"
            )

        for (i, j), meas in zip(edges, measurements):
            quat = quat_from_rot(meas[:3, :3])
            t = meas[:3, 3]
            info_vals = " ".join(f"{v:.9f}" for v in info21(info))
            fh.write(
                "EDGE_SE3:QUAT "
                f"{i} {j} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{quat[0]:.9f} {quat[1]:.9f} {quat[2]:.9f} {quat[3]:.9f} "
                f"{info_vals}\n"
            )

    meta = {
        "preset": args.preset,
        "dataset_name": preset.name,
        "rows": preset.rows,
        "cols": preset.cols,
        "num_nodes": n,
        "num_edges": m,
        "base_grid_edges": (preset.rows * preset.cols) + ((preset.rows - 1) * preset.cols),
        "extra_edges_added": m - ((preset.rows * preset.cols) + ((preset.rows - 1) * preset.cols)),
        "edge_order": args.edge_order,
        "edge_mode": args.edge_mode,
        "target_g2o_cost": chi2_target,
        "target_objective_half_chi2": objective_target,
        "dof_estimate": dof,
        "normalized_residual_scale": normalized_scale,
        "translation_info": args.translation_info,
        "rotation_info": args.rotation_info,
        "translation_sigma": trans_sigma,
        "rotation_sigma_rad": rot_sigma,
        "rotation_sigma_deg": math.degrees(rot_sigma),
        "noise_scale_mult": args.noise_scale_mult,
        "init_mode": args.init_mode,
        "init_translation_amp": args.init_translation_amp,
        "init_rotation_amp_deg": args.init_rotation_amp_deg,
        "seed": args.seed,
        "out_g2o": str(out_path),
    }
    meta_path = Path(args.out_meta) if args.out_meta else out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out_path)
    print(meta_path)


if __name__ == "__main__":
    main()
