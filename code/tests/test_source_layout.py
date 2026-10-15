"""Private source fragments must have one explicit, existing owner."""
from collections import Counter
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "code/src"


class SourceLayoutTests(unittest.TestCase):
    def test_top_level_contains_only_solver_entries(self):
        self.assertEqual({p.name for p in SRC.iterdir() if p.is_file()},
                         {"se2_solver.cpp", "se3_solver.cpp", "ba_solver.cpp"})

    def test_private_fragments_have_exactly_one_owner(self):
        references = Counter()
        for owner in SRC.rglob("*"):
            if owner.suffix not in (".cpp", ".h", ".inc"):
                continue
            for relative in re.findall(r'^#include "([^"]+\.inc)"',
                                       owner.read_text(encoding="utf-8"), re.MULTILINE):
                path = (owner.parent / relative).resolve()
                self.assertTrue(path.is_file(), f"Missing fragment: {owner}: {relative}")
                self.assertEqual(path.parent, SRC / "detail")
                references[path] += 1
        self.assertEqual(set(references), {p.resolve() for p in SRC.rglob("*.inc")})
        self.assertTrue(all(count == 1 for count in references.values()))

    def test_outer_iteration_routines_remain_in_entry_files(self):
        for name, routine in (("se2", "runSyntheticSE2Experiment"),
                              ("se3", "runSyntheticSE3Experiment"),
                              ("ba", "runRootHGBP")):
            with self.subTest(solver=name):
                source = (SRC / f"{name}_solver.cpp").read_text(encoding="utf-8")
                self.assertIn(routine + "(", source)
                self.assertRegex(source, r"for \(int outer = 1;")


if __name__ == "__main__":
    unittest.main()
