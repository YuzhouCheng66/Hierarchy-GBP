"""Capture Grid H-GBP and verify g2o keyframes against its archived continuous run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.common import process_affinity
from benchmarks.run import solver_command, environment_for
from replay_pgo import graph


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['input', 'hgbp', 'g2o', 'reference', 'final-graph', 'work', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--reuse-captures', action='store_true')
    parser.add_argument('--fresh-reference', action='store_true', help='Reference is a new continuous run captured in this session')
    args = parser.parse_args()
    if not args.reuse_captures:
        args.work.mkdir(parents=True, exist_ok=False)
    catalog = read(ROOT / 'benchmarks/configs/datasets.json')['datasets']['Grid']
    assert hashlib.file_digest(args.input.open('rb'), 'sha256').hexdigest() == catalog['sha256']
    config = read(ROOT / 'benchmarks/configs/pgo.json')
    env, _ = environment_for(config['profiles']['SE3']['environment'], 16)
    command = solver_command('pgo', config, 'SE3', args.hgbp.resolve(), args.input.resolve(), args.work / 'hgbp.json', 16, '0xffff')
    if not args.reuse_captures:
        with process_affinity('0xffff'), (args.work / 'hgbp.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=500)
        if args.fresh_reference:
            env, _ = environment_for({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, 1)
            command = [str(args.g2o.resolve()), '--input', str(args.input.resolve()), '--output-json', str(args.reference.resolve()), '--num-outer', '20', '--huber-delta', '5']
            with process_affinity('0xffff'), (args.work / 'g2o-continuous.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1900)
    native = read(args.reference)
    vertices, loops = graph(args.input)
    ids = [v[0] for v in vertices]
    states = [np.asarray([v[1:] for v in vertices], dtype='<f4')]
    previous = args.input
    env, _ = environment_for({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}, 1)
    for outer in range(1, 5):
        output = args.work / f'frame-{outer}.json'
        command = [str(args.g2o.resolve()), '--input', str(previous), '--output-json', str(output), '--num-outer', '1', '--huber-delta', '5']
        if not args.reuse_captures:
            with process_affinity('0xffff'), (args.work / f'frame-{outer}.log').open('w') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=400)
        cost = read(output)['final_objective']['raw']
        expected = native['outer_iterations'][outer - 1]['raw_cost']
        assert abs(cost / expected - 1) < 1e-7, (outer, cost, expected)
        previous = output.with_suffix('.g2o')
        rows, _ = graph(previous)
        assert [v[0] for v in rows] == ids
        states.append(np.asarray([v[1:] for v in rows], dtype='<f4'))
        print(f'Grid g2o keyframe {outer}: verified cost {cost:.8f}', flush=True)
    final, _ = graph(args.final_graph)
    assert [v[0] for v in final] == ids
    states.append(np.asarray([v[1:] for v in final], dtype='<f4'))
    np.stack(states).tofile(args.output / 'pgo-grid-baseline.bin')
    h, poses = read(args.work / 'hgbp.json'), read(args.work / 'hgbp_poses.json')
    np.asarray(poses['mg_pose_history'], dtype='<f4')[:, :, :3].copy().tofile(args.output / 'pgo-grid-hgbp.bin')
    ht = [0., *np.cumsum([r['outer_total_sec'] for r in h['mg_history'][1:]]).tolist()]
    dt = [0., *np.cumsum([r['timing']['total_sec'] for r in native['outer_iterations']]).tolist()]
    lookup = {v: i for i, v in enumerate(ids)}
    selected = [loops[i] for i in np.linspace(0, len(loops) - 1, 600, dtype=int)]
    meta = dict(dataset='Grid', nodes=len(ids), factors=catalog['factors'], points=len(ids), cameras=0,
        loop_edges=[[lookup[i], lookup[j]] for i, j in selected], input_sha256=catalog['sha256'],
        hgbp=dict(binary='pgo-grid-hgbp.bin', frames=list(range(21)), times=ht, costs=[r['nonlinear_objective'] for r in h['mg_history']], total=ht[-1]),
        baseline=dict(name='g2o-Cholesky', threads=1, binary='pgo-grid-baseline.bin', frames=[0, 1, 2, 3, 4, 20], times=dt,
            costs=[native['initial_objective']['raw'], *[r['raw_cost'] for r in native['outer_iterations']]], total=dt[-1]),
        provenance=dict(geometry='H-GBP states captured at every outer. g2o keyframes 0,1,2,3,4,20: checkpointed first steps verified against the continuous run; final state from that run.',
            timing=('Native per-outer wall times from new sequential H-GBP and continuous g2o 20-outer runs. These visualization clocks are separate from the paper benchmark.' if args.fresh_reference else 'H-GBP native outer clocks from the new capture; g2o native outer clocks from the archived continuous 20-outer run. Executions are from different sessions, not a new paired timing benchmark.'),
            interpolation='Display-only interpolation between retained geometry keyframes; costs use completed outer iterations.',
            sampling='All 8,000 input vertices and 600 uniformly sampled loop closures. The input and measurements are unchanged.', paper='https://arxiv.org/abs/2610.06978v1'))
    (args.output / 'pgo-grid.json').write_text(json.dumps(meta, indent=2) + '\n', newline='\n')
    print(f'Grid replay complete: {ht[-1]:.3f} / {dt[-1]:.3f} s', flush=True)


if __name__ == '__main__':
    main()
