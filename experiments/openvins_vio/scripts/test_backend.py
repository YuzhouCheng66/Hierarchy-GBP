#!/usr/bin/env python3
"""Run the independent dense-reference native test with the recorded library paths."""

from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu", type=int, default=0)
    a = p.parse_args()
    build = a.work_dir.resolve() / "build-integrated"
    record = json.loads((build / "build-manifest.json").read_text())
    binary = build / "test_gaussian_backend"
    assert (
        hashlib.sha256(binary.read_bytes()).hexdigest()
        == record["binary_sha256"][binary.name]
    )
    a.output.mkdir(parents=True, exist_ok=False)
    command = ["taskset", "-c", str(a.cpu), str(binary)]
    with (a.output / "test.log").open("w") as f:
        r = subprocess.run(
            command,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **record["environment"]),
        )
    (a.output / "test.json").write_text(
        json.dumps(
            dict(
                command=command,
                returncode=r.returncode,
                binary_sha256=record["binary_sha256"][binary.name],
            ),
            indent=2,
        )
        + "\n"
    )
    print((a.output / "test.log").read_text())
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
