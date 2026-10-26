"""Export real PGO keyframes; archived continuous runs supply all replay times.

The CHOLMOD executable is unmodified. Its first four GN steps are checkpointed
through 17-digit g2o files. These are visualization captures, not new timings.
No unrecorded baseline geometry is invented between the retained keyframes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.common import NUMERICAL_PREFIXES, process_affinity


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def graph(path):
    vertices, loops = [], []
    with Path(path).open() as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == 'VERTEX_SE3:QUAT':
                vertices.append((int(fields[1]), *map(float, fields[2:5])))
            elif fields[0] == 'EDGE_SE3:QUAT' and abs(int(fields[1])-int(fields[2])) != 1:
                loops.append((int(fields[1]), int(fields[2])))
    vertices.sort()
    return vertices, loops


def export(args):
    args.work.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    catalog = read(ROOT/'benchmarks/configs/datasets.json')['datasets']['Globe100k']
    assert hashlib.file_digest(args.input.open('rb'), 'sha256').hexdigest() == catalog['sha256']
    initial, loops = graph(args.input)
    selected = set(np.linspace(0, len(initial)-1, 8000, dtype=int))
    ids = [v[0] for v in initial]
    by_id = {v: i for i, v in enumerate(ids)}
    loop_sample = [loops[i] for i in np.linspace(0, len(loops)-1, 700, dtype=int)]
    for a, b in loop_sample:
        selected.update((by_id[a], by_id[b]))
    selected = sorted(selected)
    draw_index = {ids[i]: j for j, i in enumerate(selected)}
    edges = [[draw_index[a], draw_index[b]] for a, b in loop_sample]
    native = read(args.baseline_record)
    previous = args.input
    states = [np.array([initial[i][1:] for i in selected], dtype='<f4')]
    indices = [0]
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(NUMERICAL_PREFIXES)}
    env.update(OMP_NUM_THREADS='1', OMP_DYNAMIC='FALSE', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for outer in range(1, 5):
        output = args.work/f'outer_{outer}.json'
        command = [str(args.binary), '--input', str(previous), '--output-json', str(output),
                   '--num-outer', '1', '--huber-delta', '5']
        print(f'Capturing g2o Globe100k outer {outer}/4', flush=True)
        with process_affinity('0xffff'), (args.work/f'outer_{outer}.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=600,
                           creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        record = read(output)
        cost = record['final_objective']['raw']
        reference = native['outer_iterations'][outer-1]['raw_cost']
        error = abs(cost-reference)/reference
        assert error < 1e-7, (outer, error)
        previous = output.with_suffix('.g2o')
        poses, _ = graph(previous)
        assert [v[0] for v in poses] == ids
        states.append(np.array([poses[i][1:] for i in selected], dtype='<f4'))
        indices.append(outer)
        print(f'  outer={outer}, cost={cost:.9f}, archived-cost relative error={error:.3g}', flush=True)
    final, _ = graph(args.final_graph)
    assert [v[0] for v in final] == ids
    states.append(np.array([final[i][1:] for i in selected], dtype='<f4'))
    indices.append(20)
    np.stack(states).tofile(args.output/'pgo-g2o.bin')
    with zipfile.ZipFile(args.hgbp_archive) as archive:
        result = json.loads(archive.read('after/Globe100k_t16_r1.json'))
        poses = json.loads(archive.read('after/Globe100k_t16_r1_poses.json'))
    frames = poses['mg_pose_history']
    assert len(frames) == 21 and len(frames[0]) == len(initial)
    np.asarray([[frame[i][:3] for i in selected] for frame in frames], dtype='<f4').tofile(args.output/'pgo-hgbp.bin')
    # Use the paper's accepted native history, not the checkpoint-capture runtime.
    hgbp = read(args.hgbp_record)
    h_rows = hgbp['mg_history']
    def cumulative(values, total):
        values = np.asarray(values, dtype=float)
        return [0.0, *(np.cumsum(values) * total/values.sum()).tolist()]
    h_total = 42.71
    d_total = 1288.89
    metadata = dict(dataset='Globe100k', nodes=catalog['nodes'], factors=catalog['factors'],
        draw_nodes=len(selected), cameras=0, points=len(selected), loop_edges=edges,
        path_nodes=list(range(len(selected))),
        input_sha256=catalog['sha256'],
        hgbp=dict(binary='pgo-hgbp.bin', frames=list(range(21)),
                  times=cumulative([r['outer_total_sec'] for r in h_rows[1:]], h_total),
                  costs=[r['nonlinear_objective'] for r in h_rows], total=h_total),
        baseline=dict(name='g2o-Cholesky', threads=1, binary='pgo-g2o.bin', frames=indices,
                      times=cumulative([r['timing']['total_sec'] for r in native['outer_iterations']], d_total),
                      costs=[native['initial_objective']['raw'], *[r['raw_cost'] for r in native['outer_iterations']]], total=d_total),
        provenance=dict(timing='Archived native outer-time fractions calibrated to the paper Table 1 total; not capture timing.',
            geometry='Recorded H-GBP outer states; g2o keyframes 0,1,2,3,4,20. First four steps checkpointed with unchanged GN executable; final state from archived continuous run.',
            interpolation='Display-only interpolation between retained geometry keyframes. Cost readouts use completed outer iterations.',
            sampling='Uniform vertex-index sampling plus endpoints of 700 uniformly sampled loop closures; no residual-based selection.',
            paper='https://arxiv.org/abs/2610.06978v1'))
    (args.output/'pgo.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('PGO export complete', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('input', 'binary', 'baseline-record', 'final-graph', 'hgbp-archive', 'hgbp-record', 'work', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    export(p.parse_args())
