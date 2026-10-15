"""Archive self-contained timing/trajectory evidence without duplicate poses."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from compare import compare_payload, read
from compare_resident import summarize


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def archive(timing, shadows, destination, profile=None):
    if destination.exists():
        raise ValueError("Refuse to overwrite archived evidence")
    summary = summarize(timing, shadows)
    summary["rows"] = [r for r in summary["rows"] if r["variant"] == "transportsegment"]
    if len(summary["rows"]) != 3 or not all(r["certified"] for r in summary["rows"]):
        raise ValueError("All three datasets must pass CPU/trajectory/shadow gates")
    destination.mkdir(parents=True)
    records, hashes = [], set()
    with zipfile.ZipFile(destination / "raw.zip", "w", zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for directory in [timing, *shadows]:
            for run in read(directory / "runs.json"):
                if run["variant"] not in ("reference", "transportsegment"):
                    continue
                entry = {k: v for k, v in run.items() if k not in ("json", "nsight_systems")}
                entry["cohort"] = "shadow" if run["shadow"] else "timing"
                path = Path(run["json"])
                for key, source in (("result", path), ("poses", path.with_name(path.stem + "_poses.json")),
                                    ("log", path.with_suffix(".log"))):
                    data = source.read_bytes()
                    digest = hashlib.sha256(data).hexdigest()
                    name = digest + source.suffix
                    if name not in hashes:
                        output.writestr(name, data)
                        hashes.add(name)
                    entry[key] = name
                records.append(entry)
    write(destination / "runs.json", records)
    write(destination / "summary.json", summary)
    if profile:
        traces = [r for r in read(profile) if ".transportsegment." in r["trace"]]
        for trace in traces:
            trace["trace"] = Path(trace["trace"].replace("\\", "/")).name
        write(destination / "nsight_summary.json", traces)
    verify(destination)


def verify(directory):
    records = read(directory / "runs.json")
    with zipfile.ZipFile(directory / "raw.zip") as data:
        for name in data.namelist():
            if hashlib.sha256(data.read(name)).hexdigest() != Path(name).stem:
                raise ValueError("Archive hash mismatch: " + name)
        load = lambda name: json.loads(data.read(name).decode("utf-8-sig"))
        index = {(r["dataset"], r["variant"], r["repeat"], r["cohort"]): r for r in records}
        checked = 0
        for r in records:
            if r["variant"] != "transportsegment":
                continue
            cpu = index[r["dataset"], "reference", r["repeat"], r["cohort"]]
            result = compare_payload(load(cpu["result"]), load(r["result"]),
                                     load(cpu["poses"]), load(r["poses"]))
            if not result["passed"]:
                raise ValueError("Archived trajectory failed: " + r["dataset"])
            if r["exit_code"] or r["lower"]["fallback_batches"] or r["lower"]["calls"] != 60:
                raise ValueError("Incomplete GPU run")
            if r["shadow"] and not all(0 <= r["lower"][k] < 1e-6 for k in
                                      ("shadow_max_relative", "shadow_mean_max_relative")):
                raise ValueError("Archived shadow gate failed")
            checked += 1
    print(f"Verified {checked} paired outer trajectories; all blob hashes pass")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--shadow", type=Path, action="append")
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        verify(args.verify)
    elif args.timing and args.shadow and args.destination:
        archive(args.timing, args.shadow, args.destination, args.profile)
    else:
        parser.error("Supply --verify, or --timing --shadow --destination")
