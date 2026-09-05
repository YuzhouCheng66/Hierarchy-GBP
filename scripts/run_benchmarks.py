from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_root(
    roots: dict[str, Any],
    name: str,
    command_line: Path | None,
) -> Path:
    if command_line is not None:
        return command_line.resolve()
    record = roots[name]
    environment_value = os.environ.get(record["environment"])
    candidate = Path(environment_value or record["default"])
    if candidate.is_absolute():
        return candidate.resolve()
    return (PROJECT_ROOT / candidate).resolve()


def validate_config(suite: str, config: dict[str, Any]) -> None:
    expected_schema = 2
    if config.get("schema_version") != expected_schema:
        raise ValueError(
            f"{suite}.json must use schema_version {expected_schema}"
        )
    if not isinstance(config.get("shared"), dict):
        raise ValueError(f"{suite}.json is missing the shared object")
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"{suite}.json is missing dataset configurations")
    if suite == "pgo":
        expected_shared = {
            "num_outer",
            "huber_delta",
            "threads",
            "process_affinity_mask",
            "cost_relative_tolerance",
            "thread_ratio_relative_tolerance",
            "thread_time_regression_relative_tolerance",
            "thread_compare_cooldown_sec",
            "thread_cost_relative_tolerance",
        }
        common_keys = {
            "space",
            "inner_cycles",
            "pre_sweeps",
            "group_size",
            "r_reduced",
            "reference",
        }
        se2_keys = common_keys | {
            "coarse_scale",
            "partial_residual_tol",
            "basis_rebuild_period",
            "basis_rebuild_warmup_outers",
            "jitter",
            "fixed_eta_after_sweeps",
        }
        se3_keys = common_keys | {
            "basis_rebuild_period",
            "coarse_numeric_rebuild_period",
            "coarse_reuse_pcg_iters",
            "fixed_lambda_after_sweeps",
            "partial_basis_max_iters",
            "partial_basis_accept_unconverged",
            "implicit_fine_operator_threads",
            "final_direct_polish_steps",
        }
        if set(config["shared"]) != expected_shared:
            raise ValueError(
                "pgo.json shared keys must be exactly: "
                + ", ".join(sorted(expected_shared))
            )
        for dataset, record in datasets.items():
            if record.get("space") not in {"SE2", "SE3"}:
                raise ValueError(f"{dataset}: space must be SE2 or SE3")
            expected_keys = (
                se2_keys if record["space"] == "SE2" else se3_keys
            )
            if set(record) != expected_keys:
                missing = sorted(expected_keys - set(record))
                extra = sorted(set(record) - expected_keys)
                raise ValueError(
                    f"{dataset}: invalid configuration keys; "
                    f"missing={missing}, extra={extra}"
                )
            validate_hgbp_reference(
                dataset,
                record.get("reference"),
                {"time_sec", "cost"},
                {"direct_time_sec", "direct_cost", "hgbp"},
            )
        return

    expected_shared = {
        "outer",
        "threads",
        "mg_cycles",
        "pre_sweeps",
        "gbp_full_sweeps",
        "message_damping",
        "coarse_scale",
        "normalize_bal",
        "pair_factor_scale",
        "pair_backbone_min_coverage",
        "krylov_start_outer",
        "fine_smoother",
        "cost_relative_tolerance",
        "mre_relative_tolerance",
        "rmse_relative_tolerance",
        "thread_ratio_relative_tolerance",
        "thread_time_regression_relative_tolerance",
        "thread_compare_cooldown_sec",
        "thread_cost_relative_tolerance",
        "thread_mre_relative_tolerance",
        "thread_rmse_relative_tolerance",
    }
    if set(config["shared"]) != expected_shared:
        raise ValueError(
            "ba.json shared keys must be exactly: "
            + ", ".join(sorted(expected_shared))
        )
    dataset_keys = {
        "build_threads",
        "group_size",
        "initial_lambda",
        "min_pair_observations",
        "pair_sample_cap",
        "pair_sample_rescale",
        "unreduced_unary",
        "no_pose_scaling",
        "reference",
    }
    for dataset, record in datasets.items():
        optional_keys = {"mg_cycles", "krylov_start_outer"}
        if not dataset_keys.issubset(record) or not set(record).issubset(
            dataset_keys | optional_keys
        ):
            raise ValueError(
                f"{dataset}: invalid BA configuration keys"
            )
        validate_hgbp_reference(
            dataset,
            record.get("reference"),
            {"time_sec", "cost", "mre", "rmse"},
            {"hgbp"},
        )


def validate_hgbp_reference(
    dataset: str,
    reference: Any,
    entry_keys: set[str],
    reference_keys: set[str],
) -> None:
    if not isinstance(reference, dict) or set(reference) != reference_keys:
        raise ValueError(
            f"{dataset}: reference keys must be "
            + ", ".join(sorted(reference_keys))
        )
    hgbp = reference.get("hgbp")
    if not isinstance(hgbp, dict) or set(hgbp) != {"1", "16"}:
        raise ValueError(
            f"{dataset}: reference.hgbp must contain exactly 1 and 16"
        )
    for threads in ("1", "16"):
        entry = hgbp[threads]
        if not isinstance(entry, dict) or set(entry) != entry_keys:
            raise ValueError(
                f"{dataset}: invalid reference.hgbp.{threads} keys"
            )


def pgo_command(
    executable: Path,
    input_path: Path,
    output_path: Path,
    shared: dict[str, Any],
    config: dict[str, Any],
    include_direct: bool,
    write_poses: bool,
) -> list[str]:
    command = [
        str(executable),
        "--problem-file",
        str(input_path),
        "--out-json",
        str(output_path),
        "--num-outer",
        str(shared["num_outer"]),
        "--inner-cycles",
        str(config["inner_cycles"]),
        "--pre-sweeps",
        str(config["pre_sweeps"]),
        "--group-size",
        str(config["group_size"]),
        "--r-reduced",
        str(config["r_reduced"]),
    ]
    if config["space"] == "SE2":
        command.extend(
            (
                "--threads",
                str(shared["threads"]),
                "--partial-residual-tol",
                str(config["partial_residual_tol"]),
                "--basis-rebuild-period",
                str(config["basis_rebuild_period"]),
                "--basis-rebuild-warmup-outers",
                str(config["basis_rebuild_warmup_outers"]),
                "--huber-delta",
                str(shared["huber_delta"]),
                "--coarse-scale",
                str(config["coarse_scale"]),
                "--jitter",
                str(config["jitter"]),
                "--fixed-eta-after-sweeps",
                str(config["fixed_eta_after_sweeps"]),
                "--process-affinity-mask",
                str(shared["process_affinity_mask"]),
            )
        )
    else:
        command.extend(
            (
                "--threads",
                str(shared["threads"]),
                "--huber-delta",
                str(shared["huber_delta"]),
                "--process-affinity-mask",
                str(shared["process_affinity_mask"]),
                "--basis-rebuild-period",
                str(config["basis_rebuild_period"]),
                "--coarse-numeric-rebuild-period",
                str(config["coarse_numeric_rebuild_period"]),
                "--coarse-reuse-pcg-iters",
                str(config["coarse_reuse_pcg_iters"]),
                "--implicit-fine-operator-threads",
                str(min(config["implicit_fine_operator_threads"], shared["threads"])),
                "--fixed-lambda-after-sweeps",
                str(config["fixed_lambda_after_sweeps"]),
                "--partial-basis-max-iters",
                str(config["partial_basis_max_iters"]),
                "--partial-basis-accept-unconverged",
                "1" if config["partial_basis_accept_unconverged"] else "0",
                "--final-direct-polish-steps",
                str(config["final_direct_polish_steps"]),
            )
        )
        if not include_direct:
            command.append("--skip-direct")
    if write_poses:
        command.append("--write-poses")
    return command


def ba_command(
    executable: Path,
    input_path: Path,
    output_path: Path,
    shared: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    command = [
        str(executable),
        "--problem-file",
        str(input_path),
        "--out-json",
        str(output_path),
        "--outer",
        str(shared["outer"]),
        "--mg-cycles",
        str(config.get("mg_cycles", shared["mg_cycles"])),
        "--pre-sweeps",
        str(shared["pre_sweeps"]),
        "--gbp-full-sweeps",
        str(shared["gbp_full_sweeps"]),
        "--group-size",
        str(config["group_size"]),
        "--coarse-scale",
        str(shared["coarse_scale"]),
        "--build-threads",
        str(min(config["build_threads"], shared["threads"])),
        "--gbp-threads",
        str(shared["threads"]),
        "--message-damping",
        str(shared["message_damping"]),
        "--initial-lambda",
        str(config["initial_lambda"]),
        "--pair-factor-scale",
        str(shared["pair_factor_scale"]),
        "--pair-backbone-min-coverage",
        str(shared["pair_backbone_min_coverage"]),
        "--krylov-start-outer",
        str(config.get("krylov_start_outer", shared["krylov_start_outer"])),
        "--fine-smoother",
        str(shared["fine_smoother"]),
        "--min-pair-observations",
        str(config["min_pair_observations"]),
        "--pair-sample-cap",
        str(config["pair_sample_cap"]),
    ]
    command.append("--normalize-bal" if shared["normalize_bal"] else "--no-normalize-bal")
    if not config["pair_sample_rescale"]:
        command.append("--pair-sample-no-rescale")
    if config["unreduced_unary"]:
        command.append("--unreduced-unary")
    if config["no_pose_scaling"]:
        command.append("--no-pose-scaling")
    return command


def environment_for(
    suite: str,
    rootba_runtime: Path,
    threads: int,
) -> tuple[dict[str, str], dict[str, str]]:
    environment = os.environ.copy()
    overrides: dict[str, str] = {}
    if suite == "ba":
        runtime_paths = (
            rootba_runtime / "Library" / "bin",
            rootba_runtime / "Scripts",
        )
        overrides = {
            "OMP_NUM_THREADS": str(threads),
            "OMP_THREAD_LIMIT": str(threads),
            "OMP_DYNAMIC": "FALSE",
            "OPENBLAS_NUM_THREADS": str(threads),
            "MKL_NUM_THREADS": str(threads),
            "OMP_WAIT_POLICY": "PASSIVE",
            "KMP_BLOCKTIME": "0",
            "PATH": os.pathsep.join(
                [str(path) for path in runtime_paths] + [environment["PATH"]]
            ),
        }
    environment.update(overrides)
    return environment, overrides


def read_metrics(suite: str, output_path: Path) -> dict[str, float]:
    result = load_json(output_path)
    if suite == "pgo":
        history = result["mg_history"]
        return {
            "time_sec": sum(float(row["outer_total_sec"]) for row in history),
            "cost": float(history[-1]["nonlinear_objective"]),
        }
    return {
        "time_sec": float(result["total_sec"]),
        "cost": float(result["final_cost"]),
        "mre": float(result["final_are_px"]),
        "rmse": float(result["final_reprojection_rmse_px"]),
    }


def relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(abs(expected), 1e-300)


def summarize(
    suite: str,
    selected: list[str],
    configs: dict[str, Any],
    run_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    shared = configs["shared"]
    summary: list[dict[str, Any]] = []
    for dataset in selected:
        runs = [row for row in run_records if row["dataset"] == dataset]
        times = [float(row["metrics"]["time_sec"]) for row in runs]
        costs = [float(row["metrics"]["cost"]) for row in runs]
        reference = configs["datasets"][dataset]["reference"]
        thread_key = str(shared["threads"])
        if thread_key not in reference["hgbp"]:
            raise KeyError(
                f"{dataset}: no H-GBP reference for {thread_key} thread(s)"
            )
        hgbp_reference = reference["hgbp"][thread_key]
        reference_time = float(hgbp_reference["time_sec"])
        reference_cost = float(hgbp_reference["cost"])
        median_time = statistics.median(times)
        median_cost = statistics.median(costs)
        row: dict[str, Any] = {
            "dataset": dataset,
            "repeats": len(runs),
            "time_median_sec": median_time,
            "time_min_sec": min(times),
            "time_max_sec": max(times),
            "reference_time_sec": reference_time,
            "time_over_reference": median_time / reference_time,
            "cost": median_cost,
            "reference_cost": reference_cost,
            "cost_relative_error": relative_error(median_cost, reference_cost),
        }
        row["cost_match"] = (
            row["cost_relative_error"] <= shared["cost_relative_tolerance"]
        )
        if suite == "ba":
            mres = [float(run["metrics"]["mre"]) for run in runs]
            rmses = [float(run["metrics"]["rmse"]) for run in runs]
            median_mre = statistics.median(mres)
            median_rmse = statistics.median(rmses)
            row.update(
                {
                    "mre": median_mre,
                    "rmse": median_rmse,
                    "reference_mre": float(hgbp_reference["mre"]),
                    "reference_rmse": float(hgbp_reference["rmse"]),
                    "mre_relative_error": relative_error(
                        median_mre, float(hgbp_reference["mre"])
                    ),
                    "rmse_relative_error": relative_error(
                        median_rmse, float(hgbp_reference["rmse"])
                    ),
                }
            )
            row["mre_match"] = (
                row["mre_relative_error"]
                <= shared["mre_relative_tolerance"]
            )
            row["rmse_match"] = (
                row["rmse_relative_error"]
                <= shared["rmse_relative_tolerance"]
            )
        summary.append(row)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(
        dict.fromkeys(key for row in rows for key in row)
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(suite: str, rows: list[dict[str, Any]]) -> None:
    if suite == "pgo":
        print(
            "dataset          time[s]   ref[s]   time/ref"
            "        cost           rel.err   match"
        )
        for row in rows:
            print(
                f"{row['dataset']:<15} "
                f"{row['time_median_sec']:>8.4f} "
                f"{row['reference_time_sec']:>8.4f} "
                f"{row['time_over_reference']:>10.3f} "
                f"{row['cost']:>14.7g} "
                f"{row['cost_relative_error']:>9.2e} "
                f"{str(row['cost_match']):>7}"
            )
    else:
        print(
            "dataset          time[s]   ref[s]   time/ref"
            "        cost        MRE       RMSE   match"
        )
        for row in rows:
            match = (
                row["cost_match"]
                and row["mre_match"]
                and row["rmse_match"]
            )
            print(
                f"{row['dataset']:<15} "
                f"{row['time_median_sec']:>8.4f} "
                f"{row['reference_time_sec']:>8.4f} "
                f"{row['time_over_reference']:>10.3f} "
                f"{row['cost']:>11.7g} "
                f"{row['mre']:>10.6f} "
                f"{row['rmse']:>10.6f} "
                f"{str(match):>7}"
            )


def compare_threads(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    if args.threads is not None:
        parser.error("--compare-threads cannot be combined with --threads")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT_ROOT
        / "results"
        / "reproduction"
        / f"{args.suite}_scaling_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    def child_command(threads: int) -> list[str]:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            args.suite,
            *args.datasets,
            "--threads",
            str(threads),
            "--repeats",
            str(args.repeats),
            "--output-root",
            str(output_root / f"{threads}-thread"),
        ]
        optional_paths = (
            ("--pgo-data-root", args.pgo_data_root),
            ("--ba-data-root", args.ba_data_root),
            ("--rootba-runtime", args.rootba_runtime),
            ("--se2-exe", args.se2_exe),
            ("--se3-exe", args.se3_exe),
            ("--ba-exe", args.ba_exe),
        )
        for option, value in optional_paths:
            if value is not None:
                command.extend((option, str(value)))
        for enabled, option in (
            (args.include_direct, "--include-direct"),
            (args.write_poses, "--write-poses"),
            (args.skip_input_hash, "--skip-input-hash"),
            (args.no_verify_results, "--no-verify-results"),
            (args.dry_run, "--dry-run"),
        ):
            if enabled:
                command.append(option)
        return command

    comparison_config = load_json(CONFIG_ROOT / f"{args.suite}.json")
    cooldown_seconds = float(
        comparison_config["shared"]["thread_compare_cooldown_sec"]
    )
    for threads in (1, 16):
        command = child_command(threads)
        print(
            f"[thread comparison] {subprocess.list2cmdline(command)}",
            flush=True,
        )
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            return completed.returncode
        if threads == 1 and not args.dry_run and cooldown_seconds > 0.0:
            print(
                f"[thread comparison] cooling down for "
                f"{cooldown_seconds:g} s",
                flush=True,
            )
            time.sleep(cooldown_seconds)

    if args.dry_run:
        print(f"Resolved paired configurations: {output_root}")
        return 0

    config = load_json(CONFIG_ROOT / f"{args.suite}.json")
    validate_config(args.suite, config)
    one_rows = {
        row["dataset"]: row
        for row in load_json(output_root / "1-thread" / "summary.json")
    }
    sixteen_rows = {
        row["dataset"]: row
        for row in load_json(output_root / "16-thread" / "summary.json")
    }
    selected = args.datasets or list(config["datasets"])
    ratio_tolerance = float(
        config["shared"]["thread_ratio_relative_tolerance"]
    )
    time_regression_tolerance = float(
        config["shared"]["thread_time_regression_relative_tolerance"]
    )
    cost_tolerance = float(
        config["shared"]["thread_cost_relative_tolerance"]
    )
    comparison: list[dict[str, Any]] = []
    for dataset in selected:
        one = one_rows[dataset]
        sixteen = sixteen_rows[dataset]
        measured_ratio = (
            float(one["time_median_sec"]) /
            float(sixteen["time_median_sec"])
        )
        reference = config["datasets"][dataset]["reference"]["hgbp"]
        reference_ratio = (
            float(reference["1"]["time_sec"]) /
            float(reference["16"]["time_sec"])
        )
        one_time_over_reference = (
            float(one["time_median_sec"]) /
            float(reference["1"]["time_sec"])
        )
        sixteen_time_over_reference = (
            float(sixteen["time_median_sec"]) /
            float(reference["16"]["time_sec"])
        )
        cost_relative_difference = relative_error(
            float(one["cost"]),
            float(sixteen["cost"]),
        )
        row = {
            "dataset": dataset,
            "hgbp1_time_sec": float(one["time_median_sec"]),
            "hgbp16_time_sec": float(sixteen["time_median_sec"]),
            "hgbp1_over_hgbp16": measured_ratio,
            "reference_hgbp1_over_hgbp16": reference_ratio,
            "ratio_relative_error": relative_error(
                measured_ratio,
                reference_ratio,
            ),
            "hgbp1_time_over_reference": one_time_over_reference,
            "hgbp16_time_over_reference": sixteen_time_over_reference,
            "hgbp1_cost": float(one["cost"]),
            "hgbp16_cost": float(sixteen["cost"]),
            "cost_relative_difference": cost_relative_difference,
        }
        row["ratio_match"] = (
            row["ratio_relative_error"] <= ratio_tolerance
        )
        row["time_regression_match"] = (
            one_time_over_reference <= 1.0 + time_regression_tolerance
            and sixteen_time_over_reference
            <= 1.0 + time_regression_tolerance
        )
        row["cost_match"] = cost_relative_difference <= cost_tolerance
        if args.suite == "ba":
            mre_relative_difference = relative_error(
                float(one["mre"]),
                float(sixteen["mre"]),
            )
            rmse_relative_difference = relative_error(
                float(one["rmse"]),
                float(sixteen["rmse"]),
            )
            row.update(
                {
                    "hgbp1_mre": float(one["mre"]),
                    "hgbp16_mre": float(sixteen["mre"]),
                    "mre_relative_difference": mre_relative_difference,
                    "hgbp1_rmse": float(one["rmse"]),
                    "hgbp16_rmse": float(sixteen["rmse"]),
                    "rmse_relative_difference": rmse_relative_difference,
                    "mre_match": (
                        mre_relative_difference
                        <= float(
                            config["shared"][
                                "thread_mre_relative_tolerance"
                            ]
                        )
                    ),
                    "rmse_match": (
                        rmse_relative_difference
                        <= float(
                            config["shared"][
                                "thread_rmse_relative_tolerance"
                            ]
                        )
                    ),
                }
            )
        comparison.append(row)

    (output_root / "thread_scaling_summary.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_root / "thread_scaling_summary.csv", comparison)
    print(
        "dataset          HGBP-1[s] HGBP-16[s]  1/16"
        "   ref  ratio?  perf?   cost rel.err  match"
    )
    for row in comparison:
        match = row["time_regression_match"] and row["cost_match"]
        if args.suite == "ba":
            match = match and row["mre_match"] and row["rmse_match"]
        print(
            f"{row['dataset']:<15} "
            f"{row['hgbp1_time_sec']:>9.4f} "
            f"{row['hgbp16_time_sec']:>10.4f} "
            f"{row['hgbp1_over_hgbp16']:>6.2f} "
            f"{row['reference_hgbp1_over_hgbp16']:>6.2f} "
            f"{str(row['ratio_match']):>7} "
            f"{str(row['time_regression_match']):>6} "
            f"{row['cost_relative_difference']:>12.2e} "
            f"{str(match):>6}"
        )
    print(f"Thread-scaling results: {output_root}")
    if not args.no_verify_results and any(
        not row["time_regression_match"]
        or not row["cost_match"]
        or (
            args.suite == "ba"
            and (not row["mre_match"] or not row["rmse_match"])
        )
        for row in comparison
    ):
        return 3
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the formal H-GBP PGO or BA results."
    )
    parser.add_argument("suite", choices=("pgo", "ba"))
    parser.add_argument("datasets", nargs="*")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--threads",
        type=int,
        help="override the configured worker count (formal runs use 1 or 16)",
    )
    parser.add_argument(
        "--compare-threads",
        action="store_true",
        help="run H-GBP at both 1 and 16 threads and verify scaling/quality",
    )
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--pgo-data-root", type=Path)
    parser.add_argument("--ba-data-root", type=Path)
    parser.add_argument("--rootba-runtime", type=Path)
    parser.add_argument("--se2-exe", type=Path)
    parser.add_argument("--se3-exe", type=Path)
    parser.add_argument("--ba-exe", type=Path)
    parser.add_argument("--include-direct", action="store_true")
    parser.add_argument("--write-poses", action="store_true")
    parser.add_argument("--skip-input-hash", action="store_true")
    parser.add_argument("--no-verify-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.compare_threads:
        return compare_threads(args, parser)

    configs = copy.deepcopy(load_json(CONFIG_ROOT / f"{args.suite}.json"))
    validate_config(args.suite, configs)
    if args.threads is not None:
        if args.threads < 1:
            parser.error("--threads must be positive")
        if args.threads not in (1, 16):
            parser.error("formal runs support --threads 1 or --threads 16")
        configs["shared"]["threads"] = args.threads
    manifest = load_json(CONFIG_ROOT / "datasets.json")
    available = list(configs["datasets"])
    selected = args.datasets or available
    unknown = sorted(set(selected) - set(available))
    if unknown:
        parser.error(f"unknown {args.suite} datasets: {', '.join(unknown)}")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else PROJECT_ROOT / "results" / "reproduction" / f"{args.suite}_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    roots = manifest["roots"]
    data_root = resolve_root(
        roots,
        args.suite,
        args.pgo_data_root if args.suite == "pgo" else args.ba_data_root,
    )
    rootba_runtime = resolve_root(
        roots, "rootba_runtime", args.rootba_runtime
    )
    executables = {
        "SE2": (args.se2_exe or PROJECT_ROOT / "build" / "se2_benchmark.exe").resolve(),
        "SE3": (args.se3_exe or PROJECT_ROOT / "build" / "se3_benchmark.exe").resolve(),
        "ba": (args.ba_exe or PROJECT_ROOT / "build" / "ba_solver.exe").resolve(),
    }

    resolved: dict[str, Any] = {
        "suite": args.suite,
        "repeats": args.repeats,
        "shared": configs["shared"],
        "data_root": str(data_root),
        "selected": {},
        "executables": {},
    }
    for name, executable in executables.items():
        if executable.exists():
            resolved["executables"][name] = {
                "path": str(executable),
                "sha256": sha256(executable),
            }

    verified_paths: set[Path] = set()
    run_records: list[dict[str, Any]] = []
    for dataset in selected:
        config = configs["datasets"][dataset]
        data_record = manifest["datasets"][dataset]
        input_path = (data_root / data_record["path"]).resolve()
        if not input_path.is_file():
            raise FileNotFoundError(f"{dataset}: input not found: {input_path}")
        if not args.skip_input_hash and input_path not in verified_paths:
            actual_hash = sha256(input_path)
            if actual_hash != data_record["sha256"]:
                raise RuntimeError(
                    f"{dataset}: SHA-256 mismatch\n"
                    f"expected {data_record['sha256']}\nactual   {actual_hash}"
                )
            verified_paths.add(input_path)
            print(f"[input] {dataset}: SHA-256 verified")

        resolved["selected"][dataset] = {
            "input": str(input_path),
            "input_sha256": data_record["sha256"],
            "config": config,
        }
        executable_key = config["space"] if args.suite == "pgo" else "ba"
        executable = executables[executable_key]
        if not executable.is_file():
            raise FileNotFoundError(f"executable not found: {executable}")

        for repeat in range(1, args.repeats + 1):
            output_path = output_root / f"{dataset}_run{repeat}.json"
            log_path = output_root / f"{dataset}_run{repeat}.log"
            if args.suite == "pgo":
                command = pgo_command(
                    executable,
                    input_path,
                    output_path,
                    configs["shared"],
                    config,
                    args.include_direct,
                    args.write_poses,
                )
            else:
                command = ba_command(
                    executable,
                    input_path,
                    output_path,
                    configs["shared"],
                    config,
                )
            environment, overrides = environment_for(
                args.suite,
                rootba_runtime,
                int(configs["shared"]["threads"]),
            )
            print(
                f"[run] {dataset} {repeat}/{args.repeats}: "
                f"{subprocess.list2cmdline(command)}"
            )
            if args.dry_run:
                continue
            start = time.perf_counter()
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            external_wall_sec = time.perf_counter() - start
            record: dict[str, Any] = {
                "dataset": dataset,
                "repeat": repeat,
                "returncode": completed.returncode,
                "command": command,
                "environment_overrides": overrides,
                "input": str(input_path),
                "output": str(output_path),
                "log": str(log_path),
                "external_wall_sec": external_wall_sec,
            }
            if completed.returncode != 0:
                run_records.append(record)
                raise RuntimeError(
                    f"{dataset} repeat {repeat} failed; see {log_path}"
                )
            record["metrics"] = read_metrics(args.suite, output_path)
            run_records.append(record)

    (output_root / "resolved_config.json").write_text(
        json.dumps(resolved, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "runs.json").write_text(
        json.dumps(run_records, indent=2) + "\n", encoding="utf-8"
    )
    if args.dry_run:
        print(f"Resolved configuration: {output_root / 'resolved_config.json'}")
        return 0

    summary = summarize(args.suite, selected, configs, run_records)
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output_root / "summary.csv", summary)
    print_summary(args.suite, summary)
    print(f"Results: {output_root}")

    if not args.no_verify_results:
        failed = [
            row
            for row in summary
            if not row["cost_match"]
            or (
                args.suite == "ba"
                and (not row["mre_match"] or not row["rmse_match"])
            )
        ]
        if failed:
            print(
                "Numerical verification failed for: "
                + ", ".join(row["dataset"] for row in failed),
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
