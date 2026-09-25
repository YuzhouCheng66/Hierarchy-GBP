import gzip
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('downloader', ROOT/'scripts/download_datasets.py')
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw = b'VERTEX_SE2 0 0 0 0\n'
        self.archive = self.root/'input.gz'
        self.archive.write_bytes(gzip.compress(self.raw, mtime=0))
        self.expected = dict(suite='pgo', path='sample.g2o', sha256=hashlib.sha256(self.raw).hexdigest())
        self.row = dict(self.expected, archive='input.gz', archive_bytes=self.archive.stat().st_size,
                        archive_sha256=downloader.digest(self.archive), bytes=len(self.raw))

    def obtain(self, name, target):
        shutil.copyfile(self.root/name, target)

    def install(self):
        return downloader.install_one('sample', self.row, self.expected, self.root/'out', self.obtain)

    def test_install_and_cached_verification(self):
        self.assertEqual(self.install(), 'downloaded and verified')
        self.assertEqual((self.root/'out/pgo/sample.g2o').read_bytes(), self.raw)
        self.assertEqual(self.install(), 'already verified')

    def test_reject_changed_canonical_hash(self):
        self.row['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'solver catalog'):
            self.install()

    def test_reject_changed_archive(self):
        self.archive.write_bytes(b'broken')
        with self.assertRaisesRegex(ValueError, 'compressed file'):
            self.install()
        self.assertFalse((self.root/'out/pgo/sample.g2o').exists())

    def test_reject_excess_uncompressed_size(self):
        self.row['bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.install()

    def test_existing_file_never_overwritten(self):
        self.install()
        p = self.root/'out/pgo/sample.g2o'
        p.write_bytes(b'user data')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            self.install()
        self.assertEqual(p.read_bytes(), b'user data')

    def test_unsafe_paths(self):
        for path in ['../x', '/x', 'C:/x', 'a\\b', 'a/../b', 'a//b']:
            with self.subTest(path=path), self.assertRaises(ValueError):
                downloader.relative_path(path)

    def test_symlink_escape(self):
        outside = self.root/'outside'
        outside.mkdir()
        (self.root/'out').mkdir()
        try:
            (self.root/'out/pgo').symlink_to(outside, target_is_directory=True)
        except OSError:
            self.skipTest('symlink privilege unavailable')
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.install()

    def test_url_requires_fixed_commit(self):
        with self.assertRaises(ValueError):
            downloader.url_for('author/data', 'main', 'manifest.json')
        self.assertIn('/resolve/'+'a'*40+'/', downloader.url_for('author/data', 'a'*40, 'manifest.json'))


if __name__ == '__main__':
    unittest.main()
