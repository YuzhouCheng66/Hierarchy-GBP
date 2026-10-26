"""Validate published replay dimensions, clocks, metrics and site assets."""
import json
import math
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]


def main():
    for name in ['ba','pgo']:
        meta=json.loads((ROOT/f'data/{name}.json').read_text())
        n=meta['points']+meta['cameras']
        for method in ['hgbp','baseline']:
            m=meta[method]; data=(ROOT/'data'/m['binary']).read_bytes()
            assert len(data)==len(m['frames'])*n*3*4
            assert all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',data))
            assert len(m['times'])==len(m['costs'])==21
            assert m['times'][0]==0 and abs(m['times'][-1]-m['total'])<1e-8
            assert all(a<=b for a,b in zip(m['times'],m['times'][1:]))
            assert m['frames'][0]==0 and m['frames'][-1]==20
            assert all(math.isfinite(c) and c>=0 for c in m['costs'])
    tables=json.loads((ROOT/'data/benchmarks.json').read_text())
    assert len(tables['ba']['rows'])==10 and len(tables['pgo']['rows'])==9
    for family in ['ba','pgo']:
        assert all(len(row['cells'])==7 for row in tables[family]['rows'])
    for file in ['app.js','styles.css','mechanism.js','replay.js','favicon.svg']:
        assert (ROOT/file).is_file()
    print('Validated: 19 benchmark rows, four real geometry captures, complete outer clocks and assets.')


if __name__=='__main__': main()
