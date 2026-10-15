import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from compare_resident import summarize


class ResidentReportTests(unittest.TestCase):
    def test_retired_schema_cli_redirects_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            record = directory/'runs.json'
            record.write_bytes(b'preserve original record')
            process = subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('compare.py')), tmp],
                                     cwd=tmp, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn('compare_resident.py', process.stderr)
            self.assertEqual(record.read_bytes(), b'preserve original record')
            self.assertFalse((directory/'summary.json').exists())

    def setUp(self):
        base = dict(dataset="test", repeat=1, shadow=False, sanitizer="", exit_code=0,
                    process_ms=3.0, cuda_context_ms=0.1, cost=4.0, lower=None,
                    sha256="binary", dataset_sha256="input", reference_commit="commit")
        self.records = [dict(base, variant="reference", solver_ms=2., json="cpu.json"),
                        dict(base, variant="resident0", solver_ms=.5, json="gpu.json")]
        self.shadow = [dict(base, variant="resident0", repeat=0, shadow=True,
                            lower=dict(calls=60, fallback_batches=0, shadow_max_relative=0.,
                                       shadow_mean_max_relative=0.))]
        keys = ("outer_total_sec", "basis_build_sec", "coarse_lambda_sec", "sweeps_sec",
                "coarse_eta_sec", "coarse_solve_sec", "prolong_inject_sec", "factor_relinearize_sec",
                "objective_eval_sec", "gbp_sweeps_executed", "full_precision_sweeps", "eta_only_sweeps", "coarse_solves")
        self.history = dict(mg_history=[dict.fromkeys(keys, .001)])

    def evaluate(self, passed=True, compare_values=None, directories=None, extra_shadow=None):
        def reader(path):
            if Path(path) == Path("timing/runs.json"):
                return self.records
            if Path(path) == Path("shadow/runs.json"):
                return self.shadow
            if Path(path) == Path("extra/runs.json"):
                return extra_shadow
            return self.history
        with patch("compare_resident.read", side_effect=reader), \
             patch("compare_resident.compare", side_effect=compare_values,
                   return_value=dict(passed=passed, max_outer_cost_relative_error=0.,
                                     max_outer_pose_relative_error=0.)):
            return summarize(Path("timing"), directories or Path("shadow"))["rows"][0]

    def test_matching_gates_accept(self):
        self.assertTrue(self.evaluate()["accepted_2x"])

    def test_multiple_shadow_directories_keep_all_gates(self):
        paths = [Path("shadow"), Path("extra")]
        extra = copy.deepcopy(self.shadow)
        self.assertTrue(self.evaluate(directories=paths, extra_shadow=extra)["certified"])
        extra[0]["sha256"] = "different binary"
        self.assertFalse(self.evaluate(directories=paths, extra_shadow=extra)["certified"])
        extra[0]["sha256"] = "binary"
        extra[0]["lower"]["shadow_max_relative"] = 1e-3
        self.assertFalse(self.evaluate(directories=paths, extra_shadow=extra)["certified"])

    def test_error_summary_includes_every_repeat(self):
        self.records += [dict(r, repeat=2) for r in self.records]
        comparisons = [dict(passed=True, max_outer_cost_relative_error=1e-9,
                            max_outer_pose_relative_error=8e-9),
                       dict(passed=True, max_outer_cost_relative_error=7e-9,
                            max_outer_pose_relative_error=2e-9)]
        row = self.evaluate(compare_values=comparisons)
        self.assertEqual(row["all_repeat_max_cost_error"], 7e-9)
        self.assertEqual(row["all_repeat_max_pose_error"], 8e-9)

    def test_final_cost_or_old_gpu_equality_cannot_override_cpu_gate(self):
        self.assertFalse(self.evaluate(False)["accepted_2x"])

    def test_shadow_required_and_binary_must_match(self):
        self.shadow[0]["sha256"] = "other"
        self.assertFalse(self.evaluate()["certified"])
        self.shadow = []
        self.assertFalse(self.evaluate()["certified"])

    def test_failed_shadow_is_not_certified(self):
        self.shadow[0]["exit_code"] = 1
        self.shadow[0]["lower"] = None
        self.assertFalse(self.evaluate()["certified"])

    def test_mean_and_message_errors_cannot_hide_behind_exit_zero(self):
        for field in ("shadow_max_relative", "shadow_mean_max_relative"):
            for value in (1e-6, 3e-6, float("nan"), float("inf"), -1.):
                with self.subTest(field=field, value=value):
                    self.shadow[0]["lower"][field] = value
                    self.assertFalse(self.evaluate()["certified"])
            del self.shadow[0]["lower"][field]
            self.assertFalse(self.evaluate()["certified"])
            self.shadow[0]["lower"][field] = 0.

    def test_fallback_cannot_be_certified(self):
        self.shadow[0]["lower"]["fallback_batches"] = 1
        self.assertFalse(self.evaluate()["certified"])

    def test_profiling_samples_excluded(self):
        profile = copy.deepcopy(self.records[1])
        profile.update(profile_clock=True, solver_ms=1000.)
        self.records.append(profile)
        self.assertEqual(self.evaluate()["candidate_ms"], .5)

    def test_uninstrumented_device_stages_are_not_zero_time_claims(self):
        self.records[1]["lower"] = dict(wall_ms=.4, kernel_ms=0., upload_ms=0., download_ms=0.)
        self.records[1]["lower_stage_timing_available"] = False
        row = self.evaluate()
        self.assertFalse(row["lower_stage_timing_available"])
        self.assertEqual(row["lower_medians_ms"]["wall_ms"], .4)
        self.assertIsNone(row["lower_medians_ms"]["kernel_ms"])
        self.assertIsNone(row["lower_medians_ms"]["upload_ms"])
        self.assertIsNone(row["lower_medians_ms"]["download_ms"])

    def test_duplicate_and_unpaired_rejected(self):
        self.records.append(copy.deepcopy(self.records[1]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.evaluate()
        self.records.pop()
        self.records[1]["repeat"] = 2
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            self.evaluate()

    def test_invalid_time_and_provenance_rejected(self):
        self.records[1]["solver_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid timing"):
            self.evaluate()
        self.records[1]["solver_ms"] = .5
        self.records[1]["dataset_sha256"] = "different input"
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.evaluate()


if __name__ == "__main__":
    unittest.main()
