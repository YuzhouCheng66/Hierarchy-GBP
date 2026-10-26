"""Import the paper's displayed numerical tables from a pinned README revision."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
REVISION = 'ff6bed40b'


def main():
    text = subprocess.check_output(['git', 'show', f'{REVISION}:README.md'], cwd=ROOT, text=True)
    result = {'source': f'Paper Tables 1 and 2, arXiv:2610.06978v1; README {REVISION}; displayed precision retained.'}
    for family, heading, methods in [
        ('ba', '**Bundle adjustment**', ['Ceres Sparse-1', 'H-GBP-1', 'Ceres Sparse-16', 'Ceres Iterative-16', 'RootBA-QR-16', 'PowerBA-16', 'H-GBP-16']),
        ('pgo', '**Pose-graph optimization**', ['g2o-Cholesky-1', 'H-GBP-1', 'g2o-PCG-16', 'Cycle-PGO-16', 'AMM-PGO-16', 'Schwarz-PGO-16', 'H-GBP-16'])]:
        body = text.split(heading, 1)[1].split('\n\n', 2)[1]
        rows = []
        for line in body.splitlines():
            parts = [s.strip() for s in line.strip().strip('|').split('|')]
            if not parts or parts[0] in ['Dataset', '---']:
                continue
            if not line.startswith('|'):
                continue
            dataset = parts.pop(0)
            space = parts.pop(0) if family == 'pgo' else None
            cells = []
            for cell in parts:
                time, cost = cell.split('/')
                def number(value):
                    try:
                        return float(value.replace(',', ''))
                    except ValueError:
                        return None
                cells.append(dict(time=number(time), cost=number(cost), time_text=time, cost_text=cost))
            assert len(cells) == len(methods)
            rows.append(dict(dataset=dataset, space=space, cells=cells))
        assert len(rows) == (10 if family == 'ba' else 9), rows
        result[family] = dict(methods=methods, rows=rows)
    (ROOT/'site/data/benchmarks.json').write_text(json.dumps(result, indent=2)+'\n', newline='\n')


if __name__ == '__main__':
    main()
