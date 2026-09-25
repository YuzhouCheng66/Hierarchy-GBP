"""Independent final-pose scores using the retained full-information scorers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import scipy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def helper(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se2 = helper("retained_se2")
se3 = helper("retained_se3")
conversion = helper("retained_poses")
cycle = helper("reconstruct_cycle_vertices")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def poses_from_g2o(path, space):
    tag, width = ("VERTEX_SE2", 3) if space == "SE2" else ("VERTEX_SE3:QUAT", 7)
    poses = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == tag:
                key = int(fields[1])
                if key in poses or len(fields) != width + 2:
                    raise ValueError("duplicate or malformed pose")
                poses[key] = list(map(float, fields[2:]))
    if not poses or sorted(poses) != list(range(len(poses))):
        raise ValueError("canonical graphs require complete contiguous pose IDs")
    result = np.asarray([poses[key] for key in range(len(poses))])
    validate_poses(result, space)
    return result


def validate_poses(poses, space):
    width = 3 if space == "SE2" else 7
    if poses.ndim != 2 or poses.shape[1] != width or not len(poses) or not np.isfinite(poses).all():
        raise ValueError("invalid pose shape or nonfinite values")
    if space == "SE3" and np.any(np.linalg.norm(poses[:, 3:], axis=1) == 0):
        raise ValueError("zero quaternion")


def score_run(method, input_path, space, run_dir, expected_sha256):
    start = time.perf_counter()
    if space not in ("SE2", "SE3"):
        raise ValueError("unknown pose space")
    digest = sha256(input_path)
    if digest != expected_sha256:
        raise ValueError("rescore input SHA-256 mismatch")
    initial = poses_from_g2o(input_path, space)
    planar = space == "SE2"
    graph = se2.load_graph(input_path) if planar else se3.load_g2o(input_path)
    conversion_sec = 0.0
    artifacts = {}
    if method == "amm16":
        pose_path = run_dir / "estimates_huber.txt"
        if not np.isfinite(np.loadtxt(pose_path)).all():
            raise ValueError("nonfinite AMM matrix")
        converted = time.perf_counter()
        poses = conversion.amm_poses(pose_path, initial, planar)
        conversion_sec = time.perf_counter() - converted
    elif method == "cycle16":
        pose_path = run_dir / "reconstructed.g2o"
        edge_path = run_dir / "edges.txt"
        converted = time.perf_counter()
        report = cycle.reconstruct(SimpleNamespace(edges=edge_path, dimension=2 if planar else 3,
                                   num_nodes=len(initial), output=pose_path, report=run_dir / "reconstruction.json"))
        conversion_sec = time.perf_counter() - converted
        if report["connected_components"] != 1:
            raise ValueError("disconnected Cycle reconstruction")
        artifacts[edge_path.name] = sha256(edge_path)
        poses = poses_from_g2o(pose_path, space)
    else:
        paths = {"cholmod1": "result.g2o", "pcg16": "result.json.g2o", "schwarz16": "poses.g2o"}
        if method not in paths:
            raise ValueError("unknown baseline method")
        pose_path = run_dir / paths[method]
        poses = poses_from_g2o(pose_path, space)
    validate_poses(poses, space)
    if poses.shape != initial.shape:
        raise ValueError("solution does not contain all input poses")
    scorer = se2.evaluate if planar else se3.score_se3
    values = scorer(graph, poses)
    for key, value in values.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid independent score: {key}")
    # The soft-anchor diagnostic is not part of the shared factor objective.
    values["anchor_cost_excluded"] = values.pop("anchor_cost")
    artifacts[pose_path.name] = sha256(pose_path)
    sources = [Path(__file__), HERE / "retained_se2.py", HERE / "retained_se3.py",
               HERE / "retained_poses.py", HERE / "reconstruct_cycle_vertices.py"]
    return {"method": method, "space": space, "input_sha256": digest, "poses_sha256": artifacts,
            "objective": "original full-information Lie-log; half-SSR and half-Huber(delta=5); excludes anchor",
            "scores": values, "conversion_wall_sec": conversion_sec,
            "rescore_wall_sec": time.perf_counter() - start,
            "scorer_sha256": {p.name: sha256(p) for p in sources},
            "numpy_version": np.__version__, "scipy_version": scipy.__version__}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("method", choices=("cholmod1", "pcg16", "amm16", "cycle16", "schwarz16"))
    parser.add_argument("dataset")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "configs/datasets.json").read_text(encoding="utf-8"))["datasets"]
    item = manifest.get(args.dataset)
    if not item or item["suite"] != "pgo":
        parser.error("only formal-nine PGO dataset names are supported")
    if args.output.exists():
        parser.error("--output must not already exist")
    data_root = args.data_root or Path(os.environ.get("HGBP_PGO_DATA_ROOT") or ROOT / "data/pgo")
    result = score_run(args.method, data_root / item["path"], item["space"], args.run_dir, item["sha256"])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result["scores"], allow_nan=False))


if __name__ == "__main__":
    main()
