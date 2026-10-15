"""Build-script preflight tests; no compiler, downloads or installed deps needed."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
POWERSHELL = shutil.which('pwsh') or shutil.which('powershell')


def ps_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


@unittest.skipUnless(POWERSHELL, 'PowerShell unavailable')
class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        code = self.root/'code'
        (code/'cmake').mkdir(parents=True)
        shutil.copyfile(ROOT/'code/build.ps1', code/'build.ps1')
        # Isolate file checks from the local VS installation. Builds must not run.
        (code/'cmake/msvc.ps1').write_text(
            "function Initialize-HgbpMsvc { return 'test-cl.exe' }\n", encoding='ascii')
        self.deps = self.root/'deps'
        self.runtime = self.root/'runtime'
        files = ['external/eigen/Eigen/Core',
                 'vcpkg_installed/x64-windows/include/suitesparse/cholmod.h']
        files += ['vcpkg_installed/x64-windows/lib/'+name+'.lib' for name in
                  ('cholmod', 'suitesparseconfig', 'amd', 'colamd', 'camd', 'ccolamd', 'lapack', 'openblas')]
        for name in files:
            path = self.deps/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture only')
        (self.deps/'external_baselines/rootba').mkdir(parents=True)
        self.runtime.mkdir()
        (self.runtime/'openblas.dll').write_bytes(b'fixture only')

    def invoke(self, extra=''):
        command = ("function cmake { throw 'must not configure or build' }; "
                   "function ninja { throw 'must not build' }; & " + ps_literal(self.root/'code/build.ps1') +
                   ' -CheckOnly -DependencyRoot ' + ps_literal(self.deps) +
                   ' -PGORuntimeDir ' + ps_literal(self.runtime) + ' ' + extra)
        return subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command],
                              cwd=self.root, capture_output=True, text=True, timeout=30)

    def test_check_only_is_read_only(self):
        before = set(self.root.rglob('*'))
        result = self.invoke('-PGOOnly')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['Compiler'], 'test-cl.exe')
        self.assertIsNone(report['RootBA'])
        self.assertEqual(Path(report['BuildDirectory']), self.root/'build')
        self.assertEqual(set(self.root.rglob('*')), before)

    def test_missing_ba_dependencies_reported_before_build(self):
        result = self.invoke()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('rootba.lib', result.stderr)
        self.assertIn('tbb12.dll', result.stderr)
        self.assertFalse((self.root/'build').exists())

    def test_missing_pgo_dependency_and_invalid_flags(self):
        (self.runtime/'openblas.dll').unlink()
        result = self.invoke('-PGOOnly')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('openblas.dll', result.stderr)
        self.assertFalse((self.root/'build').exists())
        for extra, message in (('-PGOOnly -Jobs 0', 'Jobs'), ('-PureGBP', 'PGOOnly')):
            with self.subTest(extra=extra):
                result = self.invoke(extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)

    def test_all_powershell_sources_parse(self):
        files = sorted([ROOT/'code/build.ps1', *ROOT.glob('code/cmake/*.ps1'),
                        *ROOT.glob('code/variants/*/*.ps1')])
        for path in files:
            with self.subTest(path=path.relative_to(ROOT)):
                command = ("$tokens=$null; $errors=$null; $null="
                           '[System.Management.Automation.Language.Parser]::ParseFile(' +
                           ps_literal(path) + ',[ref]$tokens,[ref]$errors); '
                           'if ($errors.Count) { throw ($errors | Out-String) }')
                result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-Command', command],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
