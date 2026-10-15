"""Check public entry points and links after repository layout changes."""
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("layout_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayoutTests(unittest.TestCase):
    def test_benchmark_entry_points_share_repository_root(self):
        self.assertEqual(load("benchmarks/run.py").PROJECT_ROOT, ROOT)
        self.assertEqual(load("benchmarks/run.py").CONFIG_ROOT, ROOT / "benchmarks/configs")
        self.assertEqual(load("benchmarks/download.py").ROOT, ROOT)

    def test_pure_variant_uses_shared_runner_and_inputs(self):
        variant = load("code/variants/pure_gbp/run.py")
        self.assertEqual(variant.ROOT, ROOT)
        self.assertEqual(variant.official.CONFIG_ROOT, ROOT / "benchmarks/configs")
        self.assertEqual(load("code/variants/pure_gbp/archive.py").ROOT, ROOT)

    def test_public_document_links_resolve(self):
        documents = [ROOT / "README.md", *ROOT.glob("code/docs/*.md"),
                     *ROOT.glob("code/variants/*/README.md"),
                     *ROOT.glob("benchmarks/**/README.md")]
        for source in documents:
            for target in re.findall(r"\]\(([^)]+)\)", source.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                with self.subTest(document=source.relative_to(ROOT), target=target):
                    self.assertTrue((source.parent / target.split("#", 1)[0]).exists())


if __name__ == "__main__":
    unittest.main()
