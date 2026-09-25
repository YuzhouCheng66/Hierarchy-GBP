"""No native executable is launched by these tests."""

from contextlib import nullcontext, redirect_stdout, redirect_stderr
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ba_baselines", ROOT / "baselines/ba/run.py")
BA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BA)


def log_for(profile, iterations=2):
    return {"_static": {
        "problem_info": {"num_cameras": 2, "num_landmarks": 3, "num_observations": 4},
        "solver": {"num_threads_given": profile["threads"],
                   "num_threads_used": profile["threads"],
                   "solver_type": profile["solver_type"], "termination_type": 1,
                   "total_time_in_seconds": 0.25, "message": "budget reached"}},
        "iteration": list(range(iterations + 1)),
        "cost": [8.0] + [2.0] * iterations,
        "residual_block_mean": [1.5] + [0.5] * iterations}


class BaselinesTest(unittest.TestCase):
    def setUp(self):
        # Windows platform metadata can itself launch subprocesses. Isolate it
        # before any test intercepts subprocess.run for the native solver.
        for name, value in (("platform", "mock-platform"), ("processor", "mock-cpu")):
            metadata = patch.object(BA.platform, name, return_value=value)
            metadata.start()
            self.addCleanup(metadata.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "bin"
        self.runtime = self.root / "runtime"
        self.data = self.root / "data"
        for directory in (self.binary, self.runtime, self.data):
            directory.mkdir()
        for profile in BA.PROFILES["profiles"].values():
            (self.binary / (profile["binary"] + (".exe" if os.name == "nt" else ""))).write_bytes(b"mock executable")
        (self.runtime / "mock.dll").write_bytes(b"mock runtime")
        self.input = self.data / "tiny.txt"
        self.input.write_text("2 3 4\n", encoding="utf-8")
        self.dataset = {"suite": "ba", "path": "tiny.txt", "cameras": 2,
                        "points": 3, "observations": 4, "sha256": BA.sha256(self.input)}
        self.manifest = self.root / "manifest.json"
        self.write_manifest()
        self.profile = BA.PROFILES["profiles"]["Ceres-Sparse-1"]
        self.argv = ["--binary-root", str(self.binary), "--runtime-dir", str(self.runtime),
                     "--data-root", str(self.data), "--manifest", str(self.manifest),
                     "--datasets", "Tiny", "--profiles", "Ceres-Sparse-1",
                     "--output", str(self.root / "output"), "--affinity-mask", "none"]

    def write_manifest(self):
        self.manifest.write_text(json.dumps({"datasets": {"Tiny": self.dataset}}), encoding="utf-8")

    def prepare(self, extra=()):
        return BA.prepare(BA.parser().parse_args([*self.argv, *extra]))

    def test_measured_command_every_profile(self):
        for profile in BA.PROFILES["profiles"].values():
            cmd = BA.command("exe", "run", "input", profile)
            self.assertEqual(cmd, ["exe", "-C", "run", "--input", "input",
                "--max-num-iterations", "20", "--num-threads", str(profile["threads"]),
                "--function-tolerance", "0", "--gradient-tolerance", "0",
                "--parameter-tolerance", "0", "--no-save-output", "--verbosity-level", "0"])
            other = BA.command("exe", "run", "different-dataset", profile)
            self.assertEqual(cmd[:4] + cmd[5:], other[:4] + other[5:])
        self.assertEqual(len(BA.PROFILES["profiles"]), 5)

    def test_no_arbitrary_solver_flags(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            BA.parser().parse_args([*self.argv, "--power-order", "100"])

    def test_metric_conversion_and_early_stop(self):
        result = BA.metrics(log_for(self.profile), self.dataset, self.profile)
        self.assertEqual((result["cost"], result["rmse"], result["mre"]), (4, 1, 0.5))
        self.assertEqual(result["initial_cost"], 16)
        self.assertEqual(result["actual_iterations"], 2)
        self.assertEqual(result["seconds"], 0.25)

    def test_bad_logs_rejected(self):
        base = log_for(self.profile)
        mutations = [
            lambda x: x["_static"]["solver"].update(num_threads_used=16),
            lambda x: x["_static"]["solver"].update(solver_type="bal_qr"),
            lambda x: x["_static"]["solver"].update(termination_type=2),
            lambda x: x["_static"]["problem_info"].update(num_observations=5),
            lambda x: x["_static"]["problem_info"].update(num_landmarks=5),
            lambda x: x["_static"]["solver"].update(total_time_in_seconds=-1),
            lambda x: x["cost"].__setitem__(-1, float("nan")),
            lambda x: x["cost"].__setitem__(-1, float("inf")),
            lambda x: x["cost"].__setitem__(-1, -1),
            lambda x: x["residual_block_mean"].__setitem__(-1, 2),
            lambda x: x["iteration"].__setitem__(-1, 22),
            lambda x: x["iteration"].clear(),
            lambda x: x["cost"].pop(),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                value = copy.deepcopy(base)
                mutate(value)
                with self.assertRaises(ValueError):
                    BA.metrics(value, self.dataset, self.profile)

    def test_environment(self):
        with patch.dict(os.environ, {"GBP_DEBUG": "1", "hgbp_debug": "2", "OMP_NUM_THREADS": "99"}):
            env = BA.environment(16, [self.runtime])
        self.assertFalse(any(k.upper().startswith(("GBP_", "HGBP_")) for k in env))
        self.assertEqual(env["OMP_NUM_THREADS"], "16")
        self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "16")
        self.assertTrue(env["PATH"].startswith(str(self.runtime)))

    def test_dry_run_no_writes_no_process(self):
        before = set(self.root.rglob("*"))
        stdout = io.StringIO()
        with patch.object(BA.subprocess, "run") as run, patch.object(BA, "affinity") as affinity:
            with redirect_stdout(stdout):
                self.assertEqual(BA.main([*self.argv, "--dry-run"]), 0)
            run.assert_not_called()
            affinity.assert_not_called()
        self.assertEqual(before, set(self.root.rglob("*")))
        plan = json.loads(stdout.getvalue())
        self.assertTrue(plan["dry_run"])
        self.assertFalse(plan["affinity_applied"])
        self.assertEqual(len(plan["runtime_libraries_sha256"]), 1)
        self.assertFalse(plan["binaries"]["Ceres-Sparse-1"]["matches_measured_executable"])

    def test_strict_binary_hash(self):
        with self.assertRaisesRegex(ValueError, "archived measured"):
            self.prepare(["--require-measured-binaries"])

    def test_input_hash_and_shape(self):
        self.input.write_text("2 3 5\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.prepare()
        self.dataset["sha256"] = BA.sha256(self.input)
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "BAL header"):
            self.prepare()

    def test_manifest_cannot_escape_data_root(self):
        outside = self.root / "outside.txt"
        outside.write_text("2 3 4\n", encoding="utf-8")
        self.dataset["path"] = "../outside.txt"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.prepare()

    def test_existing_output_rejected_even_dry_run(self):
        (self.root / "output").mkdir()
        (self.root / "output/rootba_config.toml").write_text("[solver]\npower_order=99\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.prepare(["--dry-run"])

    def test_invalid_budgets(self):
        for extra in (["--timeout", "nan"], ["--timeout", "0"], ["--repeats", "0"],
                      ["--affinity-mask", "0"], ["--datasets", "Tiny", "Tiny"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.prepare(extra)

    def fake_solver(self, cmd, **kwargs):
        folder = Path(kwargs["cwd"])
        self.assertFalse((folder / "rootba_config.toml").exists())
        self.assertEqual(cmd[2], str(folder))
        (folder / "ba_log.json").write_text(json.dumps(log_for(self.profile)), encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    def test_mock_end_to_end_and_early_termination(self):
        with patch.object(BA.subprocess, "run", side_effect=self.fake_solver) as run:
            with patch.object(BA, "affinity", return_value=nullcontext()):
                self.assertEqual(BA.main([*self.argv, "--repeats", "2"]), 0)
        self.assertEqual(run.call_count, 2)
        rows = json.loads((self.root / "output/runs.json").read_text())
        self.assertEqual([r["actual_iterations"] for r in rows], [2, 2])
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        self.assertTrue(all(r["process_seconds"] >= 0 for r in rows))

    def test_failures_retained_and_nonzero_exit(self):
        cases = [subprocess.TimeoutExpired("mock", 1), OSError("missing DLL"),
                 subprocess.CompletedProcess([], 7), subprocess.CompletedProcess([], 0)]
        with patch.object(BA.subprocess, "run", side_effect=cases):
            with patch.object(BA, "affinity", return_value=nullcontext()):
                self.assertEqual(BA.main([*self.argv, "--repeats", "4"]), 1)
        rows = json.loads((self.root / "output/runs.json").read_text())
        self.assertEqual([r["status"] for r in rows], ["timeout", "failed", "failed", "validation_failed"])
        report = json.loads((self.root / "output/summary.json").read_text())[0]
        self.assertEqual((report["failed"], report["successful"], report["status"]), (4, 0, "incomplete"))

    def test_initial_objective_consistency(self):
        calls = []

        def solver(cmd, **kwargs):
            result = self.fake_solver(cmd, **kwargs)
            if calls:
                value = log_for(self.profile)
                value["cost"][0] = 16
                (Path(kwargs["cwd"]) / "ba_log.json").write_text(json.dumps(value))
            calls.append(cmd)
            return result

        with patch.object(BA.subprocess, "run", side_effect=solver):
            with patch.object(BA, "affinity", return_value=nullcontext()):
                self.assertEqual(BA.main([*self.argv, "--repeats", "2"]), 1)
        rows = json.loads((self.root / "output/runs.json").read_text())
        self.assertEqual(rows[-1]["error"], "inconsistent initial objective")

    def test_evidence_covers_only_measured_profiles(self):
        rows = BA.EVIDENCE["rows"]
        self.assertEqual(len(rows), 55)
        self.assertEqual({r["method"] for r in rows}, set(BA.PROFILES["profiles"]))
        self.assertTrue(all(r["provenance"]["repeats"] == 1 for r in rows if r["status"] == "ok"))
        failures = [(r["dataset"], r["method"], r["status"]) for r in rows if r["status"] != "ok"]
        self.assertEqual(failures, [("Final4585", "Ceres-Sparse-1", "timeout")])


if __name__ == "__main__":
    unittest.main()
