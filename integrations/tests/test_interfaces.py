import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def load(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Interfaces(unittest.TestCase):
    def test_g2o_preserves_edges_and_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.g2o';output=Path(folder)/'result.g2o'
            edge='EDGE_SE2 10 20 1 0 0 1 0 0 1 0 1\n'
            source.write_text('VERTEX_SE2 20 2 0 0\nVERTEX_SE2 10 0 0 0\n'+edge+'FIX 10\n')
            load('run').write_g2o(source,output,[[1,2,.3],[4,5,.6]],'SE2')
            self.assertIn('VERTEX_SE2 10 1 2 ',output.read_text())
            self.assertIn(edge+'FIX 10\n',output.read_text())
            with self.assertRaises(FileExistsError): load('run').write_g2o(source,output,[[1,2,3],[4,5,6]],'SE2')

    def test_colmap_coordinates_and_shared_intrinsics(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            (folder/'cameras.txt').write_text('1 SIMPLE_PINHOLE 640 480 500 320 240\n')
            (folder/'images.txt').write_text('1 1 0 0 0 0 0 0 1 im.png\n330 260 4\n2 1 0 0 0 0 0 0 1 im2.png\n\n')
            (folder/'points3D.txt').write_text('4 1 2 10 0 0 0 0 1 0\n')
            converter=load('colmap_to_bal')
            with self.assertRaises(ValueError): converter.convert(folder,folder/'a.txt',False)
            converter.convert(folder,folder/'a.txt',True)
            lines=(folder/'a.txt').read_text().splitlines()
            self.assertEqual(lines[0],'2 1 1')
            self.assertEqual(lines[1],'0 0 10 -20')
            self.assertAlmostEqual(float(lines[2]),3.141592653589793)


if __name__=='__main__': unittest.main()
