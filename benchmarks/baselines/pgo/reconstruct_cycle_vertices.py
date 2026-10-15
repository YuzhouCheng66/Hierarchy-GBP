#!/usr/bin/env python3
"""Reconstruct vertex poses from Cycle-PGO's cycle-consistent edge estimates."""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def so3_exp(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    omega = skew(vector)
    if theta < 1e-8:
        return np.eye(3) + omega + 0.5 * (omega @ omega)
    a = math.sin(theta) / theta
    b = (1.0 - math.cos(theta)) / (theta * theta)
    return np.eye(3) + a * omega + b * (omega @ omega)


def matrix_from_state(state: np.ndarray, dimension: int) -> np.ndarray:
    if dimension == 2:
        x, y, angle = state
        c = math.cos(float(angle))
        s = math.sin(float(angle))
        matrix = np.eye(3)
        matrix[:2, :2] = [[c, -s], [s, c]]
        matrix[:2, 2] = [x, y]
        return matrix

    matrix = np.eye(4)
    matrix[:3, :3] = so3_exp(state[3:6])
    # Cycle-PGO dumps M2V(T), i.e. matrix translation plus SO(3) logarithm.
    matrix[:3, 3] = state[:3]
    return matrix


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            qw = (matrix[2, 1] - matrix[1, 2]) / scale
            qx = 0.25 * scale
            qy = (matrix[0, 1] + matrix[1, 0]) / scale
            qz = (matrix[0, 2] + matrix[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            qw = (matrix[0, 2] - matrix[2, 0]) / scale
            qx = (matrix[0, 1] + matrix[1, 0]) / scale
            qy = 0.25 * scale
            qz = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            qw = (matrix[1, 0] - matrix[0, 1]) / scale
            qx = (matrix[0, 2] + matrix[2, 0]) / scale
            qy = (matrix[1, 2] + matrix[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.array([qx, qy, qz, qw])
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def reconstruct(args) -> dict:
    state_dimension = 3 if args.dimension == 2 else 6
    adjacency: list[list[tuple[int, np.ndarray]]] = [
        [] for _ in range(args.num_nodes)
    ]
    edge_count = 0
    with args.edges.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] != "Edge" or len(tokens) != 3 + state_dimension:
                raise ValueError(
                    f"invalid Cycle-PGO edge at line {line_number}: {line.rstrip()}"
                )
            source = int(tokens[1])
            target = int(tokens[2])
            if not (0 <= source < args.num_nodes and 0 <= target < args.num_nodes):
                raise ValueError(f"edge endpoint out of range at line {line_number}")
            transform = matrix_from_state(
                np.asarray(tokens[3:], dtype=np.float64), args.dimension
            )
            adjacency[source].append((target, transform))
            adjacency[target].append((source, np.linalg.inv(transform)))
            edge_count += 1

    matrix_size = args.dimension + 1
    poses: list[np.ndarray | None] = [None] * args.num_nodes
    component_count = 0
    for root in range(args.num_nodes):
        if poses[root] is not None:
            continue
        component_count += 1
        poses[root] = np.eye(matrix_size)
        queue = deque([root])
        while queue:
            source = queue.popleft()
            source_pose = poses[source]
            assert source_pose is not None
            for target, relative in adjacency[source]:
                if poses[target] is None:
                    poses[target] = source_pose @ relative
                    queue.append(target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for node, pose in enumerate(poses):
            assert pose is not None
            if args.dimension == 2:
                angle = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
                stream.write(
                    f"VERTEX_SE2 {node} {pose[0, 2]:.17g} "
                    f"{pose[1, 2]:.17g} {angle:.17g}\n"
                )
            else:
                quaternion = quaternion_from_matrix(pose[:3, :3])
                stream.write(
                    f"VERTEX_SE3:QUAT {node} "
                    f"{pose[0, 3]:.17g} {pose[1, 3]:.17g} "
                    f"{pose[2, 3]:.17g} {quaternion[0]:.17g} "
                    f"{quaternion[1]:.17g} {quaternion[2]:.17g} "
                    f"{quaternion[3]:.17g}\n"
                )

    report = {
        "edge_file": str(args.edges),
        "output": str(args.output),
        "dimension": args.dimension,
        "num_nodes": args.num_nodes,
        "num_edges": edge_count,
        "connected_components": component_count,
        "method": "breadth-first spanning-tree integration",
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    print(json.dumps(reconstruct(parse_args())))


if __name__ == "__main__":
    main()
