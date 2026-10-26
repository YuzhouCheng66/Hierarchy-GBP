"""Standard BAL/g2o file interface to the unchanged, shared-policy H-GBP solvers."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.run import solver_command, environment_for
from benchmarks.common import process_affinity, sha256


def write_g2o(source, destination, poses, space):
    """Replace only vertices. Edge measurements, information and FIX stay intact."""
    tag = 'VERTEX_SE2' if space == 'SE2' else 'VERTEX_SE3:QUAT'
    lines = source.read_text().splitlines(keepends=True)
    ids = sorted(int(line.split()[1]) for line in lines if line.split() and line.split()[0] == tag)
    if len(ids) != len(set(ids)) or len(ids) != len(poses):
        raise ValueError('Vertex IDs and returned poses do not match')
    lookup = dict(zip(ids, poses))
    expected = 3 if space == 'SE2' else 7
    for values in poses:
        if len(values) != expected or not all(math.isfinite(v) for v in values):
            raise ValueError('Invalid returned pose')
    with destination.open('x', newline='') as out:
        for line in lines:
            parts = line.split()
            if parts and parts[0] == tag:
                out.write(f'{tag} {parts[1]} ' + ' '.join(format(v, '.17g') for v in lookup[int(parts[1])]) + '\n')
            else:
                out.write(line)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('format', choices=['bal', 'g2o'])
    p.add_argument('input', type=Path)
    p.add_argument('--space', choices=['SE2', 'SE3'])
    p.add_argument('--threads', type=int, choices=[1, 16], default=16)
    p.add_argument('--build-dir', type=Path, default=ROOT/'build')
    p.add_argument('--ba-adapter', type=Path, default=ROOT/'build-integrations/ba_file.exe')
    p.add_argument('--output-dir', type=Path, required=True, help='New directory; never overwritten')
    p.add_argument('--export', action='store_true', help='Also write optimized BAL/g2o model')
    p.add_argument('--no-affinity', action='store_true')
    p.add_argument('--timeout', type=float, default=2000)
    a = p.parse_args()
    a.input = a.input.resolve()
    if not a.input.is_file(): p.error('Input does not exist')
    if a.format == 'g2o' and not a.space: p.error('g2o requires --space SE2 or SE3')
    if a.format == 'bal' and a.space: p.error('--space is only for g2o')
    if not math.isfinite(a.timeout) or a.timeout <= 0: p.error('Timeout must be positive')
    suite = 'ba' if a.format == 'bal' else 'pgo'
    config = json.loads((ROOT/f'benchmarks/configs/{suite}.json').read_text())
    suffix = '.exe' if os.name == 'nt' else ''
    binary = (a.ba_adapter if a.export else a.build_dir/f'ba/ba_solver{suffix}') if suite == 'ba' else a.build_dir/f'pgo/{a.space.lower()}_benchmark{suffix}'
    binary = binary.resolve()
    if not binary.is_file(): p.error(f'Build the required executable first: {binary}')
    a.output_dir = a.output_dir.resolve(); a.output_dir.mkdir(parents=True, exist_ok=False)
    native = a.output_dir/'metrics.json'
    cmd = solver_command(suite, config, a.space, binary, a.input, native, a.threads, None if a.no_affinity else '0xffff')
    template = config['environment'] if suite == 'ba' else config['profiles'][a.space]['environment']
    env, _ = environment_for(template, a.threads)
    if suite == 'ba' and a.export: env['HGBP_MODEL_OUTPUT'] = str(a.output_dir/'optimized.txt')
    (a.output_dir/'run.json').write_text(json.dumps(dict(command=cmd, input_sha256=sha256(a.input),
        binary_sha256=sha256(binary), policy=config, format=a.format), indent=2))
    with process_affinity(None if a.no_affinity else '0xffff'), (a.output_dir/'stdout.txt').open('w') as log:
        subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=a.timeout, check=True)
    if suite == 'pgo' and a.export:
        pose_record = json.loads(native.with_name('metrics_poses.json').read_text())
        write_g2o(a.input, a.output_dir/'optimized.g2o', pose_record['mg_pose_history'][-1], a.space)
    print(f'Metrics and reproducible command: {a.output_dir}')


if __name__ == '__main__':
    main()
