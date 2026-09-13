#!/usr/bin/env python3
"""Create a compact collaborator artifact after verification; never includes input images or GT poses."""

from pathlib import Path
import argparse, csv, gzip, hashlib, json, shutil, re


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--formal", default="formal01")
    p.add_argument("--audit", default="audit03")
    p.add_argument("--analysis", default="analysis01")
    p.add_argument("--unit", default="unit03")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--redact",
        action="append",
        default=[],
        help="PREFIX=REPLACEMENT; longest prefix applied first",
    )
    a = p.parse_args()
    work = a.work_dir.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    substitutions = sorted(
        [x.split("=", 1) for x in a.redact], key=lambda x: -len(x[0])
    )

    def text(s):
        for old, new in substitutions:
            s = s.replace(old, new)
        s = re.sub(r'("node":\s*)"[^"]*"', r'\1"<HOST>"', s)
        return s

    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text(source.read_text()))

    sources = {}

    def record(source):
        sources[str(source.relative_to(work))] = sha(source)

    for name in ("analysis.json", "trials.csv", "REPORT.md"):
        source = work / a.analysis / name
        record(source)
        copy(source, out / name)
    if (work / "source-audit.json").exists():
        record(work / "source-audit.json")
        copy(work / "source-audit.json", out / "builds/source-audit.json")
    for phase, directory in (("formal", a.formal), ("audit", a.audit)):
        source = work / directory
        for name in ("manifest.json", "results.json"):
            record(source / name)
            copy(source / name, out / (phase + "-" + name))
        records = json.loads((source / "results.json").read_text())
        for r in records:
            d = source / r["directory"]
            copy(
                d / "process.log",
                out / "process-logs" / phase / r["directory"] / "process.log",
            )
            if phase == "formal" and r["trial"] == 1 and r["returncode"] == 0:
                trajectory = d / "vio_trajectory.txt"
                record(trajectory)
                target = (
                    out
                    / "trajectories"
                    / r["sequence"]
                    / ("c" + str(r["clones"]))
                    / (r["method"] + ".txt")
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(trajectory, target)
        if phase == "formal":
            # All per-frame timings, including warmups, without state/GT data.
            with gzip.open(out / "all-frame-timings.csv.gz", "wt", newline="") as f:
                fields = [
                    "sequence",
                    "clones",
                    "method",
                    "trial",
                    "frame",
                    "cam_time",
                    "elapsed_s",
                    "initialized",
                    "cam_ms",
                    "imu_ms",
                    "decode_ms",
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in records:
                    states = source / r["directory"] / "vio_states.csv"
                    if not states.exists():
                        continue
                    record(states)
                    with states.open() as stream:
                        for row in csv.DictReader(stream):
                            writer.writerow(
                                {k: (r[k] if k in r else row[k]) for k in fields}
                            )
    for source in (work / a.unit).iterdir():
        if source.is_file():
            record(source)
            copy(source, out / "unit" / source.name)
    for kind in ("original", "integrated"):
        directory = work / ("build-" + kind)
        for source in directory.glob("build-manifest*.json"):
            record(source)
            copy(source, out / "builds" / kind / source.name)
        for source in directory.glob("build-*.log"):
            record(source)
            target = out / "builds" / kind / (source.name + ".gz")
            with gzip.open(target, "wt") as f:
                f.write(text(source.read_text()))
    for name in ("audit01", "audit02", "smoke01"):
        source = work / name
        if not source.exists():
            continue
        for file in ("manifest.json", "results.json"):
            if (source / file).exists():
                record(source / file)
                copy(source / file, out / "development" / name / file)
        for log in source.rglob("process.log"):
            record(log)
            copy(log, out / "development" / name / log.relative_to(source))
    diagnostic = work / "audit02/V1_01_easy/c21/root_messages/trial0/innovation-failure"
    for suffix in (".txt", ".json"):
        source = diagnostic.with_suffix(suffix)
        if source.exists():
            record(source)
            copy(source, out / "development" / ("ill-conditioned-gate" + suffix))
    if (work / "check_innovation.py").exists():
        copy(work / "check_innovation.py", out / "development/check_innovation.py")
    index = dict(
        schema="openvins_public_evidence_v1",
        note="Machine-specific prefixes normalized in text metadata. Input sensor/GT CSV hashes retained; no input images or GT poses included. All original artifacts were checked by analyze_suite.py before export.",
        original_artifacts_sha256=sources,
        public_artifacts_sha256={
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file()
        },
    )
    (out / "PUBLIC_MANIFEST.json").write_text(json.dumps(index, indent=2) + "\n")
    print("Exported", len(index["public_artifacts_sha256"]), "files")


if __name__ == "__main__":
    main()
