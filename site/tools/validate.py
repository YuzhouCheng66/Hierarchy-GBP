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
        if 'view' in meta:
            view=json.loads((ROOT/'data'/meta['view']).read_text())
            B=view['basis']
            assert len(B)==3 and all(len(row)==3 for row in B)
            for i in range(3):
                for j in range(3):
                    assert abs(sum(B[i][k]*B[j][k] for k in range(3))-(i==j))<1e-10
            determinant=sum(B[0][i]*(B[1][(i+1)%3]*B[2][(i+2)%3]-B[1][(i+2)%3]*B[2][(i+1)%3]) for i in range(3))
            assert abs(determinant-1)<1e-10, 'Display must not mirror geometry'
            assert all(a<b for a,b in zip(view['focus_min'],view['focus_max']))
            assert 0<=view['fit_quantile']<.5 and 0<=view['vertical_fit_quantile']<.5
            assert -B[1][2]>.99, 'Venice tower must point upward'
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
