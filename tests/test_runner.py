"""Driver tests use fake binaries and mocked processes: never run a benchmark."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("runner", ROOT / "scripts" / "run_benchmarks.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PGO = runner.load_json(ROOT / "configs" / "pgo.json")
BA = runner.load_json(ROOT / "configs" / "ba.json")
MANIFEST = runner.load_json(ROOT / "configs" / "datasets.json")
REFERENCES = runner.load_json(ROOT / "configs" / "reference.json")


def native_pgo(threads=1, cost=10.0):
    return {"hgbp_solver_wall_sec": 7.0, "solver_wall_sec": 9.0,
            "config": {"sync_num_threads": threads, "num_outer": 20, "adaptive_precision": True,
                       "smoother": "gbp", "group_size": 20, "basis_rebuild_period": 1, "robust_huber_delta": 5,
                       "inner_cycles": 5, "pre_sweeps": 50, "r_reduced": 12,
                       "implicit_fine_operator_threads": threads, "persistent_sweeps": True,
                       "automatic_coarse": True, "cycle_line_search": True,
                       "precision_initialization": "warm-transport-balanced",
                       "coarse_linear_backend": "cholmod_auto", "eta_lift": "precision", "residual_stop_enabled": True},
            "mg_history": [{"outer": i, "nonlinear_objective": cost,
                            "outer_total_sec": 0.1, "full_precision_sweeps": 1,
                            "eta_only_sweeps": 2, "coarse_solves": 3} for i in range(21)]}


def native_ba(threads=1):
    return {"total_sec": 2.0, "threads": threads, "build_threads": threads,
            "outer": 20, "costs": [4.0] * 21, "final_cost": 4.0,
            "final_are_px": 1.0, "final_reprojection_rmse_px": 2.0,
            "num_observations": 1, "full_sweeps": [32] * 20,
            "eta_sweeps": [3] * 20, "linear_cycles": [5] * 20,
            "coarse_operator": "additive", "linear_controller": "fcg", "aggregation": "connected",
            "gbp_model": "normalized", "fine_smoother": "gbp", "mg_cycles": 5, "pre_sweeps": 3,
            "gbp_full_sweeps": 32, "graph_neighbors": 4, "coarse_groups_budget": 24,
            "variance_rel_tol": 1e-6, "linear_rel_tol": 0.1, "outer_sec": [0.1] * 20,
            "pair_sample_cap": 16, "pair_sample_rescale": True, "message_damping": 1.0,
            "initial_lambda": 1e-4, "pair_factor_scale": 1.0, "krylov_start_outer": 1,
            "unreduced_unary": False, "no_pose_scaling": False}


class PolicyTests(unittest.TestCase):
    def test_pgo_exact_archived_arguments(self):
        expected = {
            "SE2": "--num-outer 20 --inner-cycles 3 --pre-sweeps 100 --group-size 20 --r-reduced 4 --threads 16 --partial-residual-tol 0.0001 --basis-rebuild-period 1 --basis-rebuild-warmup-outers 0 --huber-delta 5.0 --coarse-scale 1.0 --jitter 1e-07 --fixed-eta-after-sweeps -1 --process-affinity-mask 0xffff --write-poses --variance-policy adaptive --message-initialization 2",
            "SE3": "--num-outer 20 --inner-cycles 5 --pre-sweeps 50 --group-size 20 --r-reduced 12 --threads 16 --huber-delta 5.0 --process-affinity-mask 0xffff --basis-rebuild-period 1 --coarse-numeric-rebuild-period 1 --coarse-reuse-pcg-iters 1 --implicit-fine-operator-threads 16 --fixed-lambda-after-sweeps -1 --partial-basis-max-iters 6 --partial-basis-accept-unconverged 0 --final-direct-polish-steps 0 --skip-direct --variance-policy adaptive --message-initialization diagonal --persistent-sweeps --automatic-coarse --precision-initialization warm-transport-balanced --cycle-line-search --write-poses",
        }
        for space, tokens in expected.items():
            with self.subTest(space=space):
                command = runner.solver_command("pgo", PGO, space, "exe", "in", "out", 16, "0xffff")
                self.assertEqual(command[5:], tokens.split())

    def test_all_inputs_use_same_arguments_within_space(self):
        for space in ("SE2", "SE3"):
            names = MANIFEST["subsets"]["pgo"][space] + ["NeverSeenBefore"]
            commands = [runner.solver_command("pgo", PGO, space, "exe", name, name + ".json", 16, "0xffff")[5:] for name in names]
            self.assertTrue(all(command == commands[0] for command in commands))
        self.assertNotIn("datasets", PGO)
        self.assertNotIn("datasets", BA)

    def test_ba_exact_uniform_argument_mapping(self):
        expected = "--outer 20 --mg-cycles 5 --pre-sweeps 3 --gbp-full-sweeps 32 --graph-neighbors 4 --coarse-groups 24 --linear-rel-tol 0.1 --variance-rel-tol 1e-06 --pair-sample-cap 16 --message-damping 1.0 --initial-lambda 0.0001 --pair-factor-scale 1.0 --coarse-scale 1.0 --coarse-operator additive --linear-controller fcg --gbp-model normalized --krylov-start-outer 1 --normalize-bal --aggregation connected --build-threads 16 --gbp-threads 16"
        self.assertEqual(runner.ba_arguments(BA["policy"], 16), expected.split())
        self.assertEqual(runner.ba_arguments({"normalize_bal": False, "pair_sample_rescale": False,
                                            "unreduced_unary": True, "no_pose_scaling": False}, 1),
                         ["--no-normalize-bal", "--pair-sample-no-rescale", "--unreduced-unary",
                          "--build-threads", "1", "--gbp-threads", "1"])

    def test_thread_change_is_hardware_only(self):
        for suite, space, config in (("pgo", "SE2", PGO), ("pgo", "SE3", PGO), ("ba", None, BA)):
            one = runner.solver_command(suite, config, space, "exe", "in", "out", 1, "0xffff")
            sixteen = runner.solver_command(suite, config, space, "exe", "in", "out", 16, "0xffff")
            for flag in ("--threads", "--implicit-fine-operator-threads", "--build-threads", "--gbp-threads"):
                if flag in one:
                    index = one.index(flag) + 1
                    self.assertEqual((one[index], sixteen[index]), ("1", "16"))
                    one[index] = "16"
            self.assertEqual(one, sixteen)

    def test_affinity_opt_out_only_removes_cpu_mask(self):
        for space in ("SE2", "SE3"):
            fixed = runner.solver_command("pgo", PGO, space, "exe", "in", "out", 1, "0xffff")
            free = runner.solver_command("pgo", PGO, space, "exe", "in", "out", 1, None)
            index = fixed.index("--process-affinity-mask")
            del fixed[index:index + 2]
            self.assertEqual(fixed, free)

    def test_environment_is_clean_and_matches_validated_budgets(self):
        inherited = {"Path": "host-path", "GBP_TUNING": "bad", "hgbp_se3_bad": "bad",
                     "OMP_NUM_THREADS": "99", "KMP_AFFINITY": "bad", "GOMP_CPU_AFFINITY": "bad",
                     "OPENBLAS_CORETYPE": "bad", "MKL_CBWR": "bad", "KEEP_ME": "yes"}
        for threads in (1, 16):
            for family in ("SE2", "SE3", "ba"):
                template = BA["environment"] if family == "ba" else PGO["profiles"][family]["environment"]
                env, record = runner.environment_for(template, threads, inherited=inherited)
                self.assertEqual(env["OMP_NUM_THREADS"], str(threads))
                self.assertEqual(env["OMP_THREAD_LIMIT"], str(threads))
                self.assertEqual(env["OPENBLAS_NUM_THREADS"], str(threads) if family == "ba" else "1")
                self.assertEqual(env["MKL_NUM_THREADS"], str(threads) if family == "ba" else "1")
                self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
                self.assertEqual(env["KEEP_ME"], "yes")
                for key in record["removed"]:
                    if key not in template:
                        self.assertNotIn(key, env)
                self.assertEqual(env["PATH"], "host-path")
                if family == "ba":
                    self.assertEqual(env["OMP_WAIT_POLICY"], "PASSIVE")
                    self.assertEqual(env["KMP_BLOCKTIME"], "0")
                elif family == "SE3":
                    self.assertEqual(env["HGBP_SE3_ETA_LIFT"], "precision")
                    self.assertEqual(env["HGBP_SE3_RESIDUAL_STOP"], "1")
                    self.assertEqual(env["HGBP_SE3_COARSE_CHOLMOD"], "1")
        self.assertEqual(inherited["OMP_NUM_THREADS"], "99")

    def test_main10_and_opt_in_alamo(self):
        defaults = runner.select_inputs(runner.parse_args(["ba", "--output-root", "new"]), MANIFEST)
        self.assertEqual(len(defaults), 10)
        self.assertNotIn("OneDSfMAlamo", [r["label"] for r in defaults])
        self.assertNotIn("Ladybug138", MANIFEST["datasets"])
        self.assertNotIn("Final1936", MANIFEST["datasets"])
        alamo = runner.select_inputs(runner.parse_args(["ba", "--subset", "alamo", "--output-root", "new"]), MANIFEST)
        self.assertEqual(alamo[0]["expected_sha256"], "24ad4f705b7f13a48ce25fc4647e139a1b9781c253573548127b212382a84a5c")

    def test_dataset_names_after_and_between_options(self):
        for suite, names in (("pgo", ["FR079", "Cubicle"]), ("ba", ["Ladybug49", "Venice89"])):
            for tokens in (["--threads", "1", "--output-root", "new", *names],
                           [names[0], "--threads", "16", names[1], "--output-root", "new"]):
                with self.subTest(suite=suite, tokens=tokens):
                    args = runner.parse_args([suite, *tokens])
                    self.assertEqual(args.datasets, names)
                    self.assertEqual([item["label"] for item in runner.select_inputs(args, MANIFEST)], names)

    def test_reference_bindings_and_updated_se3(self):
        self.assertEqual(REFERENCES["quality_relative_tolerance"], {"pgo": 1e-6, "ba": 1e-5})
        for name, record in MANIFEST["datasets"].items():
            reference = REFERENCES["datasets"][name]
            self.assertEqual(reference["input_sha256"], record["sha256"])
            self.assertEqual(set(reference["threads"]), {"1", "16"})
        self.assertEqual(MANIFEST["datasets"]["Cubicle"]["sha256"], "f3965db26b97ff860e0dbcecd790495629324abc971f5e2dc89dff33b8bf6539")
        self.assertEqual(REFERENCES["datasets"]["Globe100k"]["threads"]["1"]["time_sec"], 177.8758535)
        self.assertEqual(REFERENCES["datasets"]["ParkingGarage"]["threads"]["1"]["cost"], 0.6341924119375459)

    def test_invalid_and_legacy_cli_flags_rejected(self):
        invalid = [[], ["--threads", "2"], ["--threads", "1", "--compare-threads"],
                   ["--repeats", "0"], ["--repeats", "-1"], ["--skip-input-hash"],
                   ["--no-verify-results"], ["--include-direct"], ["--write-poses"],
                   ["--se2-exe", "old.exe"], ["--rootba-runtime", "old"],
                   ["--pgo-data-root", "old"], ["--inner-cycles", "1"], ["--thread", "1"],
                   ["--input", "x"], ["--space", "SE2"], ["--label", "custom"],
                   ["--affinity-mask", "0"], ["--affinity-mask", "wat"],
                   ["--timeout", "0"], ["--timeout", "-1"], ["--timeout", "nan"], ["--timeout", "inf"],
                   ["--affinity-mask", "-1"], ["--no-affinity", "--affinity-mask", "1"],
                   ["FR079", "FR079"], ["FR079", "--subset", "SE2"],
                   ["--input", "x", "--space", "SE2", "--label", "../escape"],
                   ["FR079", "--input", "x", "--space", "SE2"]]
        for flags in invalid:
            argv = ["pgo", *flags, "--output-root", "new"] if flags else ["pgo"]
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(argv)
        for flags in (["ba", "--input", "x", "--space", "SE3"],):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args([*flags, "--output-root", "new"])
        for flags in (["pgo", "Unknown"], ["ba", "FR079"], ["pgo", "--subset", "main10"]):
            with self.assertRaises(ValueError):
                runner.select_inputs(runner.parse_args([*flags, "--output-root", "new"]), MANIFEST)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.build = self.root / "build"
        for suite, names in (("pgo", ("se2_benchmark", "se3_benchmark")), ("ba", ("ba_solver",))):
            folder = self.build / suite
            folder.mkdir(parents=True)
            for name in names:
                (folder / (name + (".exe" if os.name == "nt" else ""))).write_bytes(b"not executable")
        self.input = self.root / "unseen.txt"
        self.input.write_bytes(b"abc")
        self.out = self.root / "new-output"
        self.addCleanup(mock.patch.stopall)
        self.process = mock.patch.object(runner.subprocess, "run", side_effect=AssertionError("real solver forbidden")).start()
        mock.patch.object(runner, "version_info", return_value={"driver_version": "test"}).start()
        mock.patch.object(runner, "process_affinity", side_effect=lambda mask: contextlib.nullcontext()).start()
        mock.patch.dict(os.environ, {key: "" for key in ("HGBP_ROOTBA_RUNTIME", "HGBP_PGO_DATA_ROOT", "HGBP_BA_DATA_ROOT")}).start()

    def args(self, suite="pgo", extra=()):
        return runner.parse_args([suite, "--input", str(self.input),
                                  *(["--space", "SE3"] if suite == "pgo" else []),
                                  "--build-dir", str(self.build), "--output-root", str(self.out), *extra])

    def run_silently(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return runner.run(args)

    def fake_solver(self, command, **kwargs):
        threads = int(command[command.index("--threads") + 1])
        runner.write_json(command[command.index("--out-json") + 1], native_pgo(threads))
        return SimpleNamespace(returncode=0)

    def test_dry_run_hashes_split_paths_and_custom_collision(self):
        args = self.args(extra=["--dry-run", "--label", "Cubicle", "--compare-threads", "--repeats", "2"])
        self.assertEqual(self.run_silently(args), 0)
        self.process.assert_not_called()
        rows = runner.load_json(self.out / "runs.json")
        self.assertEqual([r["threads"] for r in rows], [1, 1, 16, 16])
        self.assertEqual(rows[0]["input_sha256"], "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        for row in rows:
            self.assertFalse(row["canonical_hash_verified"])
            self.assertIsNone(row["reference"])
            self.assertIn("no canonical hash", row["reference_status"])
            self.assertEqual(Path(row["command"][0]).parent, self.build / "pgo")
            self.assertEqual(row["executable"]["sha256"], runner.sha256(row["command"][0]))
        self.assertTrue((self.out / "protocol.json").exists())

    def test_ba_custom_and_runtime_root(self):
        runtime = self.root / "conda"
        runtime.mkdir()
        self.assertEqual(self.run_silently(self.args("ba", ["--dry-run", "--runtime-root", str(runtime)])), 0)
        row = runner.load_json(self.out / "runs.json")[0]
        self.assertEqual(Path(row["command"][0]).parent, self.build / "ba")
        self.assertTrue(row["environment"]["set"]["PATH"].startswith(str(runtime / "Library" / "bin")))
        self.assertIsNone(row["reference"])

    def test_data_root_environment_and_cli_precedence(self):
        for suite in ("pgo", "ba"):
            with mock.patch.dict(os.environ, {f"HGBP_{suite.upper()}_DATA_ROOT": str(self.root)}):
                args = runner.parse_args([suite, "--output-root", str(self.out)])
                selected = runner.select_inputs(args, MANIFEST)
                self.assertTrue(all(item["path"].is_relative_to(self.root) for item in selected))
                args.data_root = self.build
                selected = runner.select_inputs(args, MANIFEST)
                self.assertTrue(all(item["path"].is_relative_to(self.build) for item in selected))

    def test_runtime_environment_resolved_before_cleanup(self):
        runtime = self.root / "conda"
        runtime.mkdir()
        with mock.patch.dict(os.environ, {"HGBP_ROOTBA_RUNTIME": str(runtime)}):
            self.run_silently(self.args("ba", ["--dry-run"]))
        row = runner.load_json(self.out / "runs.json")[0]
        self.assertTrue(row["environment"]["set"]["PATH"].startswith(str(runtime / "Library" / "bin")))
        self.assertIn("HGBP_ROOTBA_RUNTIME", row["environment"]["removed"])
        self.assertNotIn("HGBP_ROOTBA_RUNTIME", row["environment"]["set"])

    def test_runtime_cli_overrides_environment(self):
        runtime = self.root / "conda"
        runtime.mkdir()
        with mock.patch.dict(os.environ, {"HGBP_ROOTBA_RUNTIME": "does-not-exist"}):
            self.run_silently(self.args("ba", ["--dry-run", "--runtime-root", str(runtime)]))
        protocol = runner.load_json(self.out / "protocol.json")
        self.assertEqual(protocol["runtime_root"], str(runtime))

    def test_pgo_runtime_does_not_add_rootba_to_path(self):
        runtime = self.root / "conda"
        runtime.mkdir()
        self.run_silently(self.args(extra=["--dry-run", "--runtime-root", str(runtime)]))
        row = runner.load_json(self.out / "runs.json")[0]
        self.assertNotIn(str(runtime), row["environment"]["set"]["PATH"])

    def test_pgo_ignores_missing_ba_runtime_from_cli_and_environment(self):
        missing = self.root / "missing-conda"
        self.assertFalse(missing.exists())
        for source in ("environment", "cli"):
            with self.subTest(source=source):
                extra = ["--dry-run"]
                if source == "cli":
                    extra += ["--runtime-root", str(missing)]
                args = self.args(extra=extra)
                args.output_root = self.out / source
                with mock.patch.dict(os.environ, {"HGBP_ROOTBA_RUNTIME": str(missing)}):
                    self.assertEqual(self.run_silently(args), 0)
                protocol = runner.load_json(args.output_root / "protocol.json")
                row = runner.load_json(args.output_root / "runs.json")[0]
                self.assertIsNone(protocol["runtime_root"])
                self.assertNotIn(str(missing), row["environment"]["set"]["PATH"])
                self.assertNotIn("HGBP_ROOTBA_RUNTIME", row["environment"]["set"])
        self.process.assert_not_called()

    def test_ba_still_rejects_missing_runtime(self):
        with mock.patch.dict(os.environ, {"HGBP_ROOTBA_RUNTIME": str(self.root / "missing-conda")}):
            with self.assertRaisesRegex(ValueError, "runtime root does not exist"):
                self.run_silently(self.args("ba", ["--dry-run"]))
        self.assertFalse(self.out.exists())
        self.process.assert_not_called()

    def test_existing_output_never_overwritten(self):
        self.out.mkdir()
        marker = self.out / "keep"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "new directory"):
            self.run_silently(self.args(extra=["--dry-run"]))
        self.assertEqual(marker.read_text(), "unchanged")

    def test_flat_executable_layout_is_not_a_fallback(self):
        executable = self.build / "pgo" / ("se3_benchmark.exe" if os.name == "nt" else "se3_benchmark")
        executable.rename(self.build / executable.name)
        with self.assertRaises(FileNotFoundError):
            self.run_silently(self.args(extra=["--dry-run"]))
        self.assertFalse(self.out.exists())

    def test_canonical_hash_mismatch_rejected_before_output(self):
        (self.root / "FR079.g2o").write_bytes(b"wrong input")
        args = runner.parse_args(["pgo", "FR079", "--data-root", str(self.root),
                                  "--build-dir", str(self.build), "--output-root", str(self.out), "--dry-run"])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self.run_silently(args)
        self.assertFalse(self.out.exists())
        self.process.assert_not_called()

    def test_sequential_compare_threads_and_persisted_metrics(self):
        self.process.side_effect = self.fake_solver
        self.assertEqual(self.run_silently(self.args(extra=["--compare-threads", "--repeats", "2"])), 0)
        rows = runner.load_json(self.out / "runs.json")
        self.assertEqual([r["threads"] for r in rows], [1, 1, 16, 16])
        for row in rows:
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["metrics"]["outer_count"], 20)
            self.assertEqual(row["metrics"]["native_wall_sec"], 7.0)
            self.assertEqual(row["quality"]["status"], "not_applicable")
            self.assertGreaterEqual(row["process_wall_sec"], 0)
        summary = runner.load_json(self.out / "summary.json")
        self.assertEqual(summary["thread_comparison"][0]["native_time_1_over_16"], 1.0)
        self.assertNotIn("speed_pass", json.dumps(summary))

    def test_failed_run_is_retained_and_returns_failure(self):
        self.process.side_effect = None
        self.process.return_value = SimpleNamespace(returncode=9)
        self.assertEqual(self.run_silently(self.args()), 2)
        row = runner.load_json(self.out / "runs.json")[0]
        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["returncode"], 9)
        self.assertIn("process_wall_sec", row)
        self.assertTrue((self.out / "summary.json").exists())

    def test_timeout_is_bounded_and_recorded(self):
        self.process.side_effect = runner.subprocess.TimeoutExpired("fake", 0.25)
        self.assertEqual(self.run_silently(self.args(extra=["--timeout", "0.25"])), 2)
        row = runner.load_json(self.out / "runs.json")[0]
        self.assertEqual(row["status"], "timeout")
        self.assertGreaterEqual(row["process_wall_sec"], 0)
        self.assertEqual(self.process.call_args.kwargs["timeout"], 0.25)
        self.assertEqual(runner.parse_args(["pgo", "--output-root", "new"] ).timeout, 2000)

    def test_banded_fallback_is_warning_not_failure(self):
        runner.write_json(self.input, dict(native_pgo(), basis_band_failures=3))
        metrics, _ = runner.read_metrics("pgo", self.input, 1, "SE3")
        self.assertEqual(metrics["diagnostics"]["basis_band_failures"], 3)
        self.assertTrue(metrics["warnings"])
        self.assertEqual(runner.check_quality(metrics, {"cost": 10, "time_sec": 1}, 1e-6)["status"], "pass")

    def test_history_policy_and_native_wall_guards(self):
        for change in (lambda r: r["mg_history"][3].update(nonlinear_objective=float("nan")),
                       lambda r: r["mg_history"][3].update(outer_total_sec=float("inf")),
                       lambda r: r["mg_history"][3].update(coarse_solves=0),
                       lambda r: r["config"].update(adaptive_precision=False),
                       lambda r: r["config"].update(smoother="jacobi"),
                       lambda r: r["config"].update(automatic_coarse=False),
                       lambda r: r.update(hgbp_solver_wall_sec=0),
                       lambda r: r.update(worker_affinity_failures=1)):
            result = native_pgo()
            change(result)
            self.input.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                runner.read_metrics("pgo", self.input, 1, "SE3", require_work=True)
        for change in (lambda r: r["costs"].__setitem__(3, float("inf")),
                       lambda r: r.update(linear_residual_norm=[1.0, float("nan")]),
                       lambda r: r.update(coarse_operator="none"),
                       lambda r: r.update(full_sweeps=[32] * 19),
                       lambda r: r["eta_sweeps"].__setitem__(3, 0),
                       lambda r: r.update(total_sec=0)):
            result = native_ba()
            change(result)
            self.input.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                runner.read_metrics("ba", self.input, 1, require_work=True)

    def test_zero_work_custom_accepted_but_canonical_rejected(self):
        for suite in ("pgo", "ba"):
            with self.subTest(suite=suite):
                result = native_pgo() if suite == "pgo" else native_ba()
                if suite == "pgo":
                    for row in result["mg_history"]:
                        row.update(full_precision_sweeps=0, eta_only_sweeps=0, coarse_solves=0)
                else:
                    for key in ("full_sweeps", "eta_sweeps", "linear_cycles"):
                        result[key] = [0] * 20
                runner.write_json(self.input, result)
                space = "SE3" if suite == "pgo" else None
                metrics, _ = runner.read_metrics(suite, self.input, 1, space)
                self.assertTrue(all(value == 0 for value in metrics["counters"].values()))
                with self.assertRaisesRegex(ValueError, "did not execute"):
                    runner.read_metrics(suite, self.input, 1, space, require_work=True)

                def zero_work_solver(command, **kwargs):
                    runner.write_json(command[command.index("--out-json") + 1], result)
                    return SimpleNamespace(returncode=0)

                self.process.side_effect = zero_work_solver
                args = self.args(suite, ["--threads", "1"])
                args.output_root = self.out / suite
                self.assertEqual(self.run_silently(args), 0)
                record = runner.load_json(args.output_root / "runs.json")[0]
                self.assertEqual(record["status"], "ok")
                self.assertEqual(record["quality"]["status"], "not_applicable")

    def test_custom_work_counters_remain_finite_and_nonnegative(self):
        for suite in ("pgo", "ba"):
            keys = ("full_precision_sweeps", "eta_only_sweeps", "coarse_solves") if suite == "pgo" else ("full_sweeps", "eta_sweeps", "linear_cycles")
            for key in keys:
                for value in (-1, float("nan"), float("inf")):
                    with self.subTest(suite=suite, key=key, value=value):
                        result = native_pgo() if suite == "pgo" else native_ba()
                        if suite == "pgo":
                            result["mg_history"][3][key] = value
                        else:
                            result[key][3] = value
                        self.input.write_text(json.dumps(result))
                        with self.assertRaises(ValueError):
                            runner.read_metrics(suite, self.input, 1, "SE3" if suite == "pgo" else None)

    def test_custom_policy_checks_remain_strict(self):
        for suite in ("pgo", "ba"):
            result = native_pgo() if suite == "pgo" else native_ba()
            if suite == "pgo":
                result["config"]["adaptive_precision"] = False
            else:
                result["coarse_operator"] = "none"
            runner.write_json(self.input, result)
            with self.assertRaisesRegex(ValueError, "native policy mismatch"):
                runner.read_metrics(suite, self.input, 1, "SE3" if suite == "pgo" else None)

    def test_native_wall_not_sum_of_outer_times(self):
        runner.write_json(self.input, native_pgo())
        metrics, _ = runner.read_metrics("pgo", self.input, 1)
        self.assertEqual(metrics["native_wall_sec"], 7)
        self.assertEqual(metrics["counters"]["full_precision_sweeps"], 20)
        result = native_pgo()
        del result["hgbp_solver_wall_sec"]
        runner.write_json(self.input, result)
        self.assertEqual(runner.read_metrics("pgo", self.input, 1)[0]["native_wall_sec"], 9)
        del result["solver_wall_sec"]
        runner.write_json(self.input, result)
        with self.assertRaises(KeyError):
            runner.read_metrics("pgo", self.input, 1)

    def test_ba_reprojection_and_invalid_native_results(self):
        runner.write_json(self.input, native_ba())
        metrics, _ = runner.read_metrics("ba", self.input, 1)
        self.assertEqual((metrics["cost"], metrics["mre"], metrics["rmse"]), (4, 1, 2))
        for result in (dict(native_ba(), final_reprojection_rmse_px=3),
                       dict(native_ba(), threads=16), dict(native_ba(), outer=19),
                       dict(native_ba(), final_are_px=4), dict(native_pgo(), mg_history=[])):
            runner.write_json(self.input, result)
            with self.assertRaises(ValueError):
                runner.read_metrics("pgo" if "mg_history" in result else "ba", self.input, 1)
        result = native_pgo()
        result["mg_history"][-1]["nonlinear_objective"] = float("nan")
        self.input.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "invalid native metric"):
            runner.read_metrics("pgo", self.input, 1)

    def test_quality_is_thread_specific_not_speed_specific(self):
        reference1 = {"cost": 10, "time_sec": 1}
        reference16 = {"cost": 11, "time_sec": 0.1}
        metrics = {"cost": 10, "native_wall_sec": 1000}
        self.assertEqual(runner.check_quality(metrics, reference1, 1e-6)["status"], "pass")
        self.assertEqual(runner.check_quality(metrics, reference16, 1e-6)["status"], "fail")
        self.assertEqual(runner.check_quality(dict(metrics, cost=10.0001), reference1, 1e-6)["status"], "fail")
        self.assertEqual(runner.check_quality(dict(metrics, cost=10.000001), reference1, 1e-6)["status"], "pass")
        ba_ref = {"cost": 10, "mre": 2, "rmse": 3, "time_sec": 1}
        ba_metrics = {**ba_ref, "native_wall_sec": 1000, "mre": 2.0001}
        self.assertEqual(runner.check_quality(ba_metrics, ba_ref, 1e-5)["status"], "fail")

    def test_one_bad_repeat_cannot_hide_in_median(self):
        reference = {"cost": 10, "time_sec": 1}
        records = [{"dataset": "x", "threads": 1, "status": status,
                    "reference": reference, "metrics": {"cost": cost, "native_wall_sec": 1, "outer_count": 20},
                    "quality": {"status": "pass"}, "process_wall_sec": 2}
                   for status, cost in (("ok", 10), ("ok", 10), ("quality_failed", 20))]
        row = runner.summarize(records)["results"][0]
        self.assertEqual(row["cost_median"], 10)
        self.assertEqual(row["quality_status"], "fail")


class BASettingsTests(unittest.TestCase):
    def test_changed_sampling_damping_and_scaling_are_rejected(self):
        changes = {"pair_sample_cap": 8, "pair_sample_rescale": False, "message_damping": 0.5,
                   "initial_lambda": 1e-3, "pair_factor_scale": 0.3, "krylov_start_outer": 2,
                   "unreduced_unary": True, "no_pose_scaling": True}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            for key, value in changes.items():
                with self.subTest(key=key):
                    runner.write_json(path, dict(native_ba(), **{key: value}))
                    with self.assertRaises(ValueError):
                        runner.read_metrics("ba", path, 1)


if __name__ == "__main__":
    unittest.main()
