"""Capture accepted BA geometry without modifying numerical solver policies."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.run import solver_command, environment_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method', choices=['hgbp', 'powerba'])
    p.add_argument('--binary', required=True, type=Path)
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--model-output', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT/'benchmarks/configs/ba.json').read_text())
    env, _ = environment_for(config['environment'], 16)
    env['HGBP_SCENE_PREFIX'] = str(args.output.resolve()/'scene')
    if args.model_output:
        env['HGBP_MODEL_OUTPUT'] = str(args.model_output.resolve())
    if args.method == 'hgbp':
        cmd = solver_command('ba', config, None, args.binary.resolve(), args.input.resolve(),
                             args.output.resolve()/'native.json', 16, None)
    else:
        cmd = [str(args.binary.resolve()), '-C', str(args.output.resolve()), '--input', str(args.input.resolve()),
               '--max-num-iterations', '20', '--num-threads', '16', '--function-tolerance', '0',
               '--gradient-tolerance', '0', '--parameter-tolerance', '0', '--no-save-output', '--verbosity-level', '0']
    (args.output/'command.json').write_text(json.dumps(cmd, indent=2))
    with (args.output/'stdout.txt').open('w') as out:
        subprocess.run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT, check=True)
    print(f'{args.method}: geometry capture complete: {args.output}')


if __name__ == '__main__':
    main()
