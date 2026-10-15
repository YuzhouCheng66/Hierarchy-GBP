"""Shared tools, CLI entry points and release integrity without native solvers."""
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks import common
from benchmarks.datasets import package_benchmarks as release


class CommonTests(unittest.TestCase):
    def test_hash_and_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'record.json'
            common.write_json(path, {'cost': 1.25}, newline='\n')
            self.assertNotIn(b'\r', path.read_bytes())
            self.assertEqual(common.sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(common.load_json(path), {'cost': 1.25})
            with self.assertRaises(ValueError):
                common.write_json(path, {'cost': float('nan')})

    def test_affinity_restores_after_failure(self):
        fake = SimpleNamespace(name='posix', sched_getaffinity=Mock(return_value={0, 1, 2}),
                               sched_setaffinity=Mock())
        with patch.object(common, 'os', fake):
            with self.assertRaisesRegex(RuntimeError, 'child failed'):
                with common.process_affinity('0x3'):
                    raise RuntimeError('child failed')
        self.assertEqual(fake.sched_setaffinity.call_args_list,
                         [unittest.mock.call(0, {0, 1}), unittest.mock.call(0, {0, 1, 2})])

    def test_affinity_opt_out_and_invalid_masks(self):
        fake = SimpleNamespace(name='posix', sched_getaffinity=Mock(return_value={0, 1}),
                               sched_setaffinity=Mock())
        with patch.object(common, 'os', fake):
            with common.process_affinity(None):
                pass
            fake.sched_getaffinity.assert_not_called()
            for mask in (0, -1, 2**64, True, 1.5, '0x4'):
                with self.subTest(mask=mask), self.assertRaises(ValueError):
                    with common.process_affinity(mask):
                        self.fail('invalid mask was applied')
            fake.sched_setaffinity.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Windows process affinity')
    def test_native_windows_affinity_and_restore(self):
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]

        def current():
            active, system = ctypes.c_size_t(), ctypes.c_size_t()
            self.assertTrue(kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(),
                                                         ctypes.byref(active), ctypes.byref(system)))
            return active.value

        original = current()
        selected = original & -original
        with self.assertRaises(RuntimeError):
            with common.process_affinity(selected):
                self.assertEqual(current(), selected)
                raise RuntimeError('restore on exception')
        self.assertEqual(current(), original)

    def test_public_help_from_unrelated_working_directory(self):
        entries = ['benchmarks/download.py', 'benchmarks/run.py',
                   'benchmarks/datasets/package_benchmarks.py',
                   'benchmarks/baselines/pgo/run.py', 'benchmarks/baselines/ba/run.py',
                   'code/variants/pure_gbp/run.py', 'code/variants/pure_gbp/archive.py']
        with tempfile.TemporaryDirectory() as cwd:
            for entry in entries:
                with self.subTest(entry=entry):
                    result = subprocess.run([sys.executable, '-B', str(ROOT/entry), '--help'],
                                            cwd=cwd, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.pgo = self.root/'tiny.g2o'
        self.pgo.write_text('VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 0 0 0\n'
                            'EDGE_SE2 0 1 0 0 0 1 0 0 1 0 1\n', encoding='ascii')
        self.ba = self.root/'tiny.txt'
        self.ba.write_text('1 1 1\n0 0 0 0\n' + '0\n'*12, encoding='ascii')
        self.pgo_row = dict(suite='pgo', space='SE2', path=self.pgo.name, nodes=2, factors=1,
                            sha256=common.sha256(self.pgo))
        self.ba_row = dict(suite='ba', path=self.ba.name, cameras=1, points=1, observations=1,
                           sha256=common.sha256(self.ba))

    def test_structure_checks(self):
        self.assertTrue(release.validate_pgo(self.pgo, self.pgo_row)['valid_factor_endpoints'])
        self.assertTrue(release.validate_bal(self.ba, self.ba_row)['valid_observation_indices'])
        for kind in ('duplicate', 'endpoint', 'nonfinite', 'count', 'record'):
            text = self.pgo.read_text()
            value = dict(self.pgo_row)
            if kind == 'duplicate':
                text += 'VERTEX_SE2 1 0 0 0\n'
            elif kind == 'endpoint':
                text = text.replace('EDGE_SE2 0 1', 'EDGE_SE2 0 2')
            elif kind == 'nonfinite':
                text = text.replace('VERTEX_SE2 0 0', 'VERTEX_SE2 0 nan')
            elif kind == 'count':
                value['factors'] = 2
            else:
                text += 'UNKNOWN 1\n'
            bad = self.root/(kind+'.g2o')
            bad.write_text(text, encoding='ascii')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                release.validate_pgo(bad, value)

    def test_bal_bad_values_and_truncation(self):
        for text in ('1 1 2\n', '1 1 1\n1 0 0 0\n', '1 1 1\n0 0 nan 0\n',
                     '1 1 1\n0 0 0 0\n', self.ba.read_text()+'0\n'):
            self.ba.write_text(text, encoding='ascii')
            with self.subTest(text=text), self.assertRaises(ValueError):
                release.validate_bal(self.ba, self.ba_row)

    def test_validation_survives_python_optimization(self):
        self.pgo.write_text('VERTEX_SE2 0 0 0 0\n', encoding='ascii')
        code = ('from pathlib import Path; from benchmarks.datasets.package_benchmarks import validate_pgo; '
                'validate_pgo(Path(' + repr(str(self.pgo)) + '), ' + repr(self.pgo_row) + ')')
        result = subprocess.run([sys.executable, '-O', '-B', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('node/factor count mismatch', result.stderr)

    def test_release_roundtrip_and_deterministic_archives(self):
        catalog = {'subsets': {'pgo': {'all': ['TinyPGO']}, 'ba': {'main10': ['TinyBA']}},
                   'datasets': {'TinyPGO': self.pgo_row, 'TinyBA': self.ba_row}}
        # Only the catalog load is intercepted; compression, manifests and hashes are real.
        with patch.object(release.json, 'loads', return_value=catalog), redirect_stdout(io.StringIO()):
            release.package(self.root, self.root, self.root/'first')
            release.package(self.root, self.root, self.root/'second')
        for folder in ('first', 'second'):
            manifest = common.load_json(self.root/folder/'manifest.json')
            self.assertEqual(manifest['subsets'], {'pgo': ['TinyPGO'], 'ba': ['TinyBA']})
            self.assertNotIn(b'\r', (self.root/folder/'manifest.json').read_bytes())
            for name, row in manifest['datasets'].items():
                archive = self.root/folder/row['archive']
                self.assertEqual(common.sha256(archive), row['archive_sha256'])
                raw = gzip.decompress(archive.read_bytes())
                self.assertEqual(hashlib.sha256(raw).hexdigest(), row['sha256'])
                self.assertEqual(raw, (self.root/row['path']).read_bytes())
                self.assertEqual(archive.read_bytes(), (self.root/'first'/row['archive']).read_bytes())

    def test_pure_archive_refuses_overwrite_and_records_support(self):
        runs = self.root/'runs'
        for variant in ('main', 'pure'):
            directory = runs/('Tiny_'+variant)
            directory.mkdir(parents=True)
            record = dict(dataset='Tiny', variant=variant, returncode=0, threads=16,
                          binary_sha256='fixture', native_sec=1, process_sec=2, cost=3,
                          coarse_sec=0.1, sweeps=10)
            common.write_json(directory/'record.json', record)
            common.write_json(directory/'result.json', {'fixture': True})
        output = self.root/'evidence'
        command = [sys.executable, '-B', str(ROOT/'code/variants/pure_gbp/archive.py'),
                   str(runs), '--output', str(output)]
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        source = common.load_json(output/'source_sha256.json')
        for path in ('benchmarks/run.py', 'benchmarks/common.py', 'code/build.ps1', 'code/cmake/msvc.ps1'):
            self.assertEqual(source[path], common.sha256(ROOT/path))
        before = {p.name: p.read_bytes() for p in output.iterdir()}
        result = subprocess.run(command, cwd=self.root, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})


if __name__ == '__main__':
    unittest.main()
