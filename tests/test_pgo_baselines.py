"""Mocked replay tests and tiny known-answer scoring fixtures; no solver runs."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pgo_baselines", ROOT / "baselines/pgo/run.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
try:
    scorer = runner.load_scorer()
except ImportError:
    scorer = None


def native(method):
    cfg = dict(num_outer=20, huber_delta=5)
    result = dict(configuration=cfg, timing=dict(solve_sec=2.0, total_sec=3.0,
                  setup_excluding_parse_sec=0.5, solver_wall_sec=2.5),
                  final_objective=dict(raw=50.0, huber=37.5), outer_iterations=[{} for _ in range(20)])
    if method == "cholmod1":
        cfg.update(threads=1, symbolic_reuse=True, block_ordering=True, diagonal_jitter=1e-10, early_outer_exit=False)
    elif method == "pcg16":
        cfg.update(requested_threads=16, observed_openmp_team=16, g2o_openmp=True, pcg_kernel_parallel=False,
                   pcg_tolerance=1e-6, pcg_max_iterations=-1, pcg_absolute_tolerance=False, diagonal_jitter=0)
        result.update(returned_iterations=20, iterations=[{} for _ in range(20)],
                      native_objective=dict(final_raw=50.0, final_huber=37.5))
    else:
        cfg.update(threads=16, preconditioner="two_level_additive_schwarz", schwarz_threads=16,
                   schwarz_subdomains=16, schwarz_overlap_layers=1, schwarz_local_shift=1e-10,
                   pcg_max_iterations=-1, pcg_tolerance=1e-8, diagonal_jitter=1e-10,
                   line_search=True, paper_gradient_stopping=True,
                   gradient_absolute_tolerance=1e-8, gradient_relative_tolerance=1e-6)
        result["outer_iterations"] = [{} for _ in range(4)]
    return result


class PolicyTests(unittest.TestCase):
    def command(self, method, name="FR079", space="SE2"):
        return runner.solver_command(method, name, space, "solver", "input", Path("output"))

    def test_cholmod_exact_budget(self):
        self.assertEqual(self.command("cholmod1")[5:], ["--num-outer", "20", "--huber-delta", "5"])

    def test_pcg_historical_serial_policy(self):
        self.assertEqual(self.command("pcg16")[5:],
                         "--dimension 2 --threads 16 --num-outer 20 --stopping historical-relative".split())
        self.assertIn("serial", runner.POLICIES["pcg16"]["label"])

    def test_amm_preserves_original_all_nine_budgets(self):
        expected = dict(FR079=100, FRH=300, M3500=300, ParkingGarage=300, Sphere=300,
                        Cubicle=300, Grid=20, Globe10k=300, Globe100k=100)
        self.assertEqual(runner.POLICIES["amm16"]["iterations"], expected)
        for name, count in expected.items():
            command = self.command("amm16", name)
            self.assertEqual(command[command.index("--iters") + 1], str(count))
            self.assertEqual(command[command.index("--loss_reg") + 1], "25")
            self.assertEqual(command[command.index("--local_max_accepted") + 1], "1")

    def test_cycle_no_robust_loss_added(self):
        self.assertEqual(self.command("cycle16")[1:],
                         "-i input -e edges.txt -v poses.txt -s summary.txt -n 20 -t 0".split())

    def test_schwarz_original_early_stopping(self):
        command = self.command("schwarz16")
        self.assertIn("--line-search", command)
        self.assertIn("--paper-gradient-stopping", command)
        self.assertEqual(command[command.index("--pcg-tolerance") + 1], "1e-08")
        self.assertEqual(command[command.index("--schwarz-overlap-layers") + 1], "1")

    def test_no_dataset_dispatch_except_archived_amm_budgets(self):
        for method in ("cholmod1", "pcg16", "cycle16", "schwarz16"):
            self.assertEqual(self.command(method, "FR079"), self.command(method, "Cubicle"))

    def test_intermixed_dataset_selector(self):
        args = runner.parse_args(["pcg16", "--binary-root", "bin", "--output-root", "out", "FR079"])
        self.assertEqual(args.datasets, ["FR079"])
        self.assertEqual(args.affinity_mask, "0xffff")

    def test_invalid_numerical_flags_and_timeout(self):
        for extra in (["--threads", "1"], ["--iters", "2"], ["--timeout", "0"], ["--timeout", "nan"],
                      ["--repeats", "0"], ["--library-root", "other"], ["--dry"]):
            with self.subTest(extra=extra), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.parse_args(["pcg16", "--binary-root", "bin", "--output-root", "out"] + extra)

    def test_environment_sanitized_and_fixed(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "99", "HGBP_FOO": "1", "G2O_TEST": "2",
                                         "MKL_NUM_THREADS": "80", "LD_PRELOAD": "injected", "KEEP": "yes"}, clear=True):
            env, record = runner.environment_for(runner.POLICIES["amm16"], Path("libs"))
        self.assertEqual(env["OMP_NUM_THREADS"], "16")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_PLACES"], "cores")
        self.assertEqual(env["KEEP"], "yes")
        for key in ("HGBP_FOO", "G2O_TEST", "LD_PRELOAD"):
            self.assertNotIn(key, env)
            self.assertIn(key, record["removed"])

    def test_linux_timeout_memory_and_safe_quoting(self):
        cmd = runner.linux_command(["/path with space/solver", "$(touch nope)"], "0xffff", 2000)
        self.assertEqual(cmd[:2], ["bash", "-c"])
        self.assertIn("ulimit -v 25165824", cmd[2])
        self.assertIn("--kill-after=10s 2000s /usr/bin/taskset 0xffff", cmd[2])
        self.assertEqual(shlex.split(cmd[2])[-2:], ["/path with space/solver", "$(touch nope)"])
        self.assertNotIn("taskset", runner.linux_command(["solver"], None, 30)[2])

    def test_configs_are_portable_lf(self):
        for name in ("policies.json", "provenance.json", "scoring_provenance.json"):
            content = (runner.HERE / name).read_bytes()
            self.assertNotIn(b"\r", content)
            self.assertNotRegex(content.decode(), r"[A-Z]:[/\\]")


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "FR079.g2o").write_text("canonical fixture\n")
        self.binary = self.root / "bin"
        self.binary.mkdir()
        self.manifest = self.root / "manifest.json"
        runner.core.write_json(self.manifest, {"datasets": {
            name: dict(suite="pgo", space="SE2", path="FR079.g2o", sha256=runner.core.sha256(self.data / "FR079.g2o"))
            for name in ("FR079", "Globe100k")}})
        for policy in runner.POLICIES.values():
            for binary in policy["executables"].values():
                suffix = ".exe" if os.name == "nt" and policy["platform"] == "native" else ""
                (self.binary / (binary + suffix)).write_bytes(b"mock binary, never execute")
        self.patch = mock.patch.object(runner, "MANIFEST_PATH", self.manifest)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def args(self, method="cholmod1", *extra):
        return runner.parse_args([method, "FR079", "--binary-root", str(self.binary),
                                  "--data-root", str(self.data), "--output-root", str(self.root / "out")] + list(extra))

    def metric(self, method, data):
        runner.core.write_json(self.run_dir / "result.json", data)
        return runner.read_metrics(method, self.run_dir, "FR079")

    def test_native_timing_scopes_and_early_stop(self):
        self.assertEqual(self.metric("cholmod1", native("cholmod1"))["native_wall_sec"], 2.5)
        self.assertEqual(self.metric("pcg16", native("pcg16"))["native_wall_sec"], 2.5)
        metrics = self.metric("schwarz16", native("schwarz16"))
        self.assertEqual((metrics["native_wall_sec"], metrics["iterations"]), (3.0, 4))

    def test_native_policy_and_all_history_finite(self):
        for key, value in (("pcg_kernel_parallel", True), ("observed_openmp_team", 1)):
            data = native("pcg16")
            data["configuration"][key] = value
            with self.assertRaises(ValueError):
                self.metric("pcg16", data)
        data = native("cholmod1")
        data["outer_iterations"][0]["raw_cost"] = float("nan")
        with mock.patch.object(runner.core, "load_json", return_value=data), self.assertRaises(ValueError):
            runner.read_metrics("cholmod1", self.run_dir, "FR079")

    def test_full_outer_budget_not_silently_shortened(self):
        data = native("cholmod1")
        data["outer_iterations"].pop()
        with self.assertRaises(ValueError):
            self.metric("cholmod1", data)

    def test_amm_native_wall_not_ideal_divided_time(self):
        text = "OMP_NUM_THREADS = '16'\n" + "0: iteration\n" * 100
        text += "measured algorithm wall: 3\nmeasured setup wall: 1\nmeasured optimization wall: 2\nfinal objective: 70\n"
        (self.run_dir / "stdout.txt").write_text(text)
        metrics = runner.read_metrics("amm16", self.run_dir, "FR079")
        self.assertEqual((metrics["native_wall_sec"], metrics["iterations"]), (3, 100))
        self.assertEqual(metrics["native_chordal_objective"], 70)
        (self.run_dir / "stdout.txt").write_text(text.replace("0: iteration\n", "", 1))
        with self.assertRaises(ValueError):
            runner.read_metrics("amm16", self.run_dir, "FR079")

    def test_cycle_scope_separation(self):
        (self.run_dir / "stdout.txt").write_text("OMP_NUM_THREADS = '16'\n")
        (self.run_dir / "summary.txt").write_text(
            "time_overall ( cycle_basis: 1 sec, ordering: 2 sec, solve: 3 sec ) sum: 6 sec\n"
            "time_process ( load data: 1 optimization: 7 dump data: 1 calculate vertices: 2 )\n"
            "n_iterations: 20\n")
        metrics = runner.read_metrics("cycle16", self.run_dir, "FR079")
        self.assertEqual((metrics["native_wall_sec"], metrics["optimization_and_vertices_sec"]), (6, 9))

    def test_gnu_process_time_is_not_launcher_time(self):
        (self.run_dir / "process_wall.txt").write_text("process_wall_sec 1.25\nmax_rss_kib 400\nexit_code 0\n")
        metrics = runner.launcher_metrics(self.run_dir, 2.0, True)
        self.assertEqual((metrics["launcher_wall_sec"], metrics["process_wall_sec"]), (2.0, 1.25))

    def test_dry_run_never_launches_or_sets_affinity(self):
        args = self.args("pcg16", "--dry-run")
        args.data_root = None
        with mock.patch.dict(os.environ, {"HGBP_PGO_DATA_ROOT": str(self.data)}), \
             mock.patch.object(runner.subprocess, "run") as execute, \
             mock.patch.object(runner.core, "process_affinity") as affinity, \
             mock.patch.object(runner, "load_scorer") as load, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(args), 0)
        execute.assert_not_called()
        affinity.assert_not_called()
        load.assert_not_called()
        record = runner.core.load_json(self.root / "out/runs.json")[0]
        self.assertFalse(record["runtime"]["archived_bundle_match"])
        self.assertEqual(record["status"], "planned")

    def test_hash_mismatch_rejected_before_output_created(self):
        (self.data / "FR079.g2o").write_text("changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            runner.run(self.args("cholmod1", "--dry-run"))
        self.assertFalse((self.root / "out").exists())

    def test_runtime_strict_rejects_unverified_binary(self):
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            runner.run(self.args("cholmod1", "--dry-run", "--require-archived-runtime"))

    def test_output_root_must_be_new_and_selector_known(self):
        args = self.args("cholmod1", "--dry-run")
        args.datasets = ["NotCanonical"]
        with self.assertRaisesRegex(ValueError, "formal-nine"):
            runner.run(args)
        args.datasets = ["FR079"]
        args.output_root.mkdir()
        with self.assertRaisesRegex(ValueError, "new directory"):
            runner.run(args)

    def test_historical_cells_are_not_fabricated_failures(self):
        args = self.args("cycle16", "--dry-run")
        args.datasets = ["Globe100k"]
        with contextlib.redirect_stdout(io.StringIO()):
            runner.run(args)
        record = runner.core.load_json(self.root / "out/runs.json")[0]
        self.assertEqual(record["status"], "historically_unmeasured")
        self.assertNotIn("metrics", record)

    def test_subprocess_timeout_is_recorded(self):
        with mock.patch.object(runner, "load_scorer"), \
             mock.patch.object(runner.core, "process_affinity", return_value=contextlib.nullcontext()), \
             mock.patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired("mock", 2)) as execute, \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.args("pcg16", "--timeout", "2")), 2)
        self.assertEqual(execute.call_args.kwargs["timeout"], 2)
        self.assertEqual(runner.core.load_json(self.root / "out/runs.json")[0]["status"], "timeout")

    def test_success_requires_independent_rescore(self):
        def completed(command, **kwargs):
            runner.core.write_json(Path(kwargs["cwd"]) / "result.json", native("cholmod1"))
            return SimpleNamespace(returncode=0)
        scores = {"scores": {"raw_objective": 50.0, "robust_objective": 37.5}}
        with mock.patch.object(runner, "load_scorer", return_value=SimpleNamespace(score_run=lambda *a: scores)), \
             mock.patch.object(runner.core, "process_affinity", return_value=contextlib.nullcontext()), \
             mock.patch.object(runner.subprocess, "run", side_effect=completed), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.run(self.args()), 0)
        record = runner.core.load_json(self.root / "out/runs.json")[0]
        self.assertEqual(record["status"], "ok")
        self.assertIn("independently rescored", record["shared_cost_status"])


@unittest.skipIf(scorer is None, "independent scoring requires NumPy and SciPy")
class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "input.g2o"
        self.graph.write_text("VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 0 0 0\nEDGE_SE2 0 1 0 0 0 1 0 0 1 0 1\n")
        self.sha = scorer.sha256(self.graph)

    def test_se2_known_half_ssr_and_huber(self):
        (self.root / "result.g2o").write_text("VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 10 0 0\n")
        result = scorer.score_run("cholmod1", self.graph, "SE2", self.root, self.sha)
        self.assertEqual(result["scores"]["raw_objective"], 50)
        self.assertEqual(result["scores"]["robust_objective"], 37.5)
        self.assertEqual(result["scores"]["anchor_cost_excluded"], 0)

    def test_se3_full_information_cross_terms(self):
        np = scorer.np
        poses = np.array([[0, 0, 0, 0, 0, 0, 1], [1, 2, 0, 0, 0, 0, 1]], dtype=float)
        info = np.eye(6)[None, :, :]
        info[0, 0, 1] = info[0, 1, 0] = 0.5
        graph = (poses.copy(), np.array([[0, 1]]), np.array([[0, 0, 0, 0, 0, 0, 1]]), info)
        result = scorer.se3.score_se3(graph, poses)
        self.assertAlmostEqual(result["raw_objective"], 3.5)
        self.assertAlmostEqual(result["robust_objective"], 3.5)

    def test_amm_saved_matrix_is_transposed_rotation(self):
        np = scorer.np
        rotations = scorer.conversion.Rotation.from_rotvec([[.2, .3, -.1], [.4, -.2, .1]]).as_matrix()
        translation = np.array([[1, 2, 3], [4, 5, 6]])
        path = self.root / "estimates_huber.txt"
        np.savetxt(path, np.vstack((translation, rotations.transpose(0, 2, 1).reshape(6, 3))))
        poses = scorer.conversion.amm_poses(path, np.zeros((2, 7)), False)
        np.testing.assert_allclose(poses[:, :3], translation)
        np.testing.assert_allclose(scorer.conversion.Rotation.from_quat(poses[:, 3:]).as_matrix(), rotations, atol=1e-14)

    def test_cycle_bfs_scored_not_official_poses(self):
        (self.root / "edges.txt").write_text("Edge 0 1 10 0 0\n")
        (self.root / "poses.txt").write_text("deliberately unused official vertices")
        result = scorer.score_run("cycle16", self.graph, "SE2", self.root, self.sha)
        self.assertEqual(result["scores"]["raw_objective"], 50)
        self.assertTrue((self.root / "reconstructed.g2o").is_file())

    def test_nonfinite_and_missing_poses_rejected(self):
        for text in ("VERTEX_SE2 0 0 0 0\n", "VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 nan 0 0\n"):
            (self.root / "result.g2o").write_text(text)
            with self.assertRaises(ValueError):
                scorer.score_run("cholmod1", self.graph, "SE2", self.root, self.sha)

    def test_rescore_hash_guard(self):
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            scorer.score_run("cholmod1", self.graph, "SE2", self.root, "0" * 64)


if __name__ == "__main__":
    unittest.main()
