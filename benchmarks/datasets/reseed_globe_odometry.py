#!/usr/bin/env python3
"""Rebuild a Globe graph's initial poses without changing any measurements."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preserve every EDGE_SE3:QUAT measurement and information matrix, "
            "but integrate consecutive-id edges into an odometry-chain initial guess."
        )
    )
    parser.add_argument("--input", required=True, help="Source SE3 g2o graph.")
    parser.add_argument("--output", required=True, help="Converted SE3 g2o graph.")
    parser.add_argument("--meta", default="", help="Optional conversion metadata JSON.")
    return parser.parse_args()


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < 1e-15:
        raise ValueError("Invalid zero or non-finite quaternion")
    q = q / norm
    return q if q[3] >= 0.0 else -q


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return quat_normalize(
        np.array(
            [
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
                lw * rw - lx * rx - ly * ry - lz * rz,
            ],
            dtype=np.float64,
        )
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quat_rotate(q: np.ndarray, point: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    skew = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return point + 2.0 * (w * (skew @ point) + skew @ (skew @ point))


def compose(
    lhs: tuple[np.ndarray, np.ndarray],
    rhs: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    lhs_t, lhs_q = lhs
    rhs_t, rhs_q = rhs
    return lhs_t + quat_rotate(lhs_q, rhs_t), quat_multiply(lhs_q, rhs_q)


def inverse(
    pose: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    t, q = pose
    q_inv = quat_conjugate(q)
    return -quat_rotate(q_inv, t), q_inv


def parse_pose(tokens: list[str], offset: int) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray([float(value) for value in tokens[offset : offset + 3]])
    q = quat_normalize(
        np.asarray([float(value) for value in tokens[offset + 3 : offset + 7]])
    )
    return t, q


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    vertex_lines: dict[int, str] = {}
    vertex_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    edge_records: list[tuple[int, int, tuple[np.ndarray, np.ndarray], str]] = []
    passthrough: list[str] = []

    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        tokens = raw_line.split()
        if not tokens:
            passthrough.append(raw_line)
            continue
        if tokens[0] == "VERTEX_SE3:QUAT":
            vertex_id = int(tokens[1])
            vertex_lines[vertex_id] = raw_line
            vertex_poses[vertex_id] = parse_pose(tokens, 2)
        elif tokens[0] == "EDGE_SE3:QUAT":
            edge_records.append(
                (int(tokens[1]), int(tokens[2]), parse_pose(tokens, 3), raw_line)
            )
        else:
            passthrough.append(raw_line)

    vertex_ids = sorted(vertex_poses)
    if vertex_ids != list(range(len(vertex_ids))):
        raise ValueError("Odometry-chain conversion requires contiguous vertex ids 0..N-1")

    chain_edges: dict[int, tuple[int, tuple[np.ndarray, np.ndarray]]] = {}
    for edge_index, (i, j, measurement, _) in enumerate(edge_records):
        if j == i + 1:
            chain_edges.setdefault(i, (edge_index, measurement))
        elif i == j + 1:
            chain_edges.setdefault(j, (edge_index, inverse(measurement)))

    missing = [index for index in range(len(vertex_ids) - 1) if index not in chain_edges]
    if missing:
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(f"Missing consecutive-id odometry edges at: {preview}")

    odometry_poses = [vertex_poses[0]]
    chain_indices: list[int] = []
    for index in range(len(vertex_ids) - 1):
        edge_index, measurement = chain_edges[index]
        chain_indices.append(edge_index)
        odometry_poses.append(compose(odometry_poses[-1], measurement))

    chain_index_set = set(chain_indices)
    ordered_edge_indices = chain_indices + [
        index for index in range(len(edge_records)) if index not in chain_index_set
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for vertex_id, (translation, quaternion) in enumerate(odometry_poses):
            values = [*translation.tolist(), *quaternion.tolist()]
            encoded = " ".join(format(value, ".17g") for value in values)
            stream.write(f"VERTEX_SE3:QUAT {vertex_id} {encoded}\n")
        for line in passthrough:
            stream.write(f"{line}\n")
        for edge_index in ordered_edge_indices:
            stream.write(f"{edge_records[edge_index][3]}\n")

    metadata = {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "num_vertices": len(vertex_ids),
        "num_edges": len(edge_records),
        "measurement_records_preserved_verbatim": True,
        "edge_order": "consecutive_id_odometry_first",
        "initialization": "integrated_consecutive_id_odometry",
        "root_vertex_record": vertex_lines[0],
    }
    meta_path = Path(args.meta) if args.meta else output_path.with_suffix(".conversion.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output_path)
    print(meta_path)


if __name__ == "__main__":
    main()
