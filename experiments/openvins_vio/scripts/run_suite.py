#!/usr/bin/env python3
"""Serial raw-data validation/benchmark. Never uses recorded linearization inputs."""

from pathlib import Path
import argparse, datetime, hashlib, json, os, platform, random, subprocess, sys, time
from score_euroc import score

PACKAGE = Path(__file__).resolve().parents[1]
SEQUENCES = ("V1_01_easy", "V1_02_medium", "V1_03_difficult")
METHODS = ("original", "root_messages", "enhanced_ekf")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--phase", choices=("smoke", "audit", "formal"), required=True)
    p.add_argument("--cpu", type=int, default=0)
    p.add_argument("--sequences", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    p.add_argument("--clones", nargs="+", type=int, choices=(11, 21), default=[11, 21])
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    work = a.work_dir.resolve()
    builds = {
        name: json.loads((work / ("build-" + name) / "build-manifest.json").read_text())
        for name in (
            ("integrated",) if a.phase != "formal" else ("original", "integrated")
        )
    }
    for kind, b in builds.items():
        assert b["build_returncode"] == 0
        for name, expected in b["binary_sha256"].items():
            assert sha(work / ("build-" + kind) / name) == expected
    if a.phase == "formal":
        assert not subprocess.check_output(
            ["git", "-C", str(work / "source-original"), "status", "--porcelain"],
            text=True,
        )
    env = dict(os.environ, **builds["integrated"]["environment"])
    env["OV_GAUSSIAN_AUDIT"] = "0" if a.phase == "formal" else "1"
    cells = [(s, c) for s in a.sequences for c in a.clones]
    rng = random.Random(20260913)
    jobs = []
    if a.phase == "formal":
        for s, c in cells:
            warm = list(METHODS)
            rng.shuffle(warm)
            jobs += [(s, c, m, 0) for m in warm]
        for trial in range(1, 4):
            block = [(s, c, m, trial) for s, c in cells for m in METHODS]
            rng.shuffle(block)
            jobs += block
    else:
        jobs = [(s, c, m, 0) for s, c in cells for m in METHODS[1:]]
    meta = dict(
        schema="openvins_live_suite_v1",
        phase=a.phase,
        started_utc=now(),
        cpu=a.cpu,
        seed=20260913,
        duration_s=30 if a.phase == "smoke" else -1,
        system=platform.uname()._asdict(),
        lscpu=subprocess.check_output(["lscpu"], text=True),
        builds=builds,
        environment={k: env[k] for k in builds["integrated"]["environment"]},
        protocol_sha256=sha(PACKAGE / "PROTOCOL.md"),
        runner_sha256=sha(Path(__file__)),
        configs_sha256={
            x.name: sha(x) for x in (PACKAGE / "configs").iterdir() if x.is_file()
        },
        job_order=jobs,
        data_manifests={},
    )
    for s, c in cells:
        if s in meta["data_manifests"]:
            continue
        mav = a.data_dir / s / "mav0"
        meta["data_manifests"][s] = {
            str(f.relative_to(mav)): sha(f)
            for f in [
                mav / "cam0/data.csv",
                mav / "cam1/data.csv",
                mav / "imu0/data.csv",
                mav / "state_groundtruth_estimate0/data.csv",
            ]
        }
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    results = []
    for index, (seq, clones, method, trial) in enumerate(jobs):
        directory = (
            out
            / seq
            / ("c" + str(clones))
            / method
            / ("warmup" if a.phase == "formal" and trial == 0 else "trial" + str(trial))
        )
        directory.mkdir(parents=True)
        prefix = directory / "vio"
        kind = "original" if method == "original" else "integrated"
        env["OV_GAUSSIAN_BACKEND"] = method
        env["OV_GAUSSIAN_AUDIT_DUMP"] = str(directory / "innovation-failure.txt")
        cmd = [
            "taskset",
            "-c",
            str(a.cpu),
            str(work / ("build-" + kind) / "run_euroc_folder"),
            str(PACKAGE / "configs" / ("euroc-c" + str(clones) + ".yaml")),
            str((a.data_dir / seq).resolve()),
            str(prefix),
            str(meta["duration_s"]),
        ]
        started = now()
        start = time.perf_counter()
        with (directory / "process.log").open("w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        elapsed = (time.perf_counter() - start) * 1000
        result = dict(
            sequence=seq,
            clones=clones,
            method=method,
            trial=trial,
            phase=a.phase,
            returncode=r.returncode,
            process_wall_ms=elapsed,
            started_utc=started,
            finished_utc=now(),
            command=cmd,
            directory=str(directory.relative_to(out)),
        )
        if not r.returncode:
            result["run"] = json.loads(Path(str(prefix) + "_run.json").read_text())
            if method != "original":
                result["backend"] = json.loads(
                    Path(str(prefix) + "_backend.json").read_text()
                )
            try:
                result["score"] = score(
                    Path(str(prefix) + "_trajectory.txt"),
                    a.data_dir / seq / "mav0/state_groundtruth_estimate0/data.csv",
                    directory / "evaluation",
                    states_path=Path(str(prefix) + "_states.csv"),
                )
            except Exception as e:
                result["scoring_error"] = repr(e)
        result["artifacts_sha256"] = {
            f.name: sha(f) for f in directory.iterdir() if f.is_file()
        }
        (directory / "record.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
        results.append(result)
        (out / "results.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n"
        )
        print(
            f"{index + 1}/{len(jobs)} {seq} c{clones} {method} trial{trial}: code={r.returncode}, process={elapsed:.1f} ms",
            flush=True,
        )
        if r.returncode:
            print((directory / "process.log").read_text()[-2200:], flush=True)
    meta["finished_utc"] = now()
    meta["complete"] = len(results) == len(jobs)
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return int(any(r["returncode"] or "scoring_error" in r for r in results))


if __name__ == "__main__":
    sys.exit(main())
