"""Pack real BA captures, or preserve a previously published PGO capture."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def upright_basis(path, points):
    with path.open() as stream:
        cameras, _, observations = map(int, stream.readline().split())
        for _ in range(observations):
            next(stream)
        parameters = np.array([float(next(stream)) for _ in range(cameras * 9)]).reshape(cameras, 9)
    ups = []
    for angle in parameters[:, :3]:
        theta = np.linalg.norm(angle)
        if theta < 1e-12:
            rotation = np.eye(3)
        else:
            x, y, z = angle / theta
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * skew @ skew
        ups.append(rotation[1])
    up = np.mean(ups, axis=0)
    if np.linalg.norm(up) < .05:
        _, vectors = np.linalg.eigh(np.asarray(ups).T @ np.asarray(ups))
        up = vectors[:, -1] * (1 if vectors[:, -1] @ ups[0] >= 0 else -1)
    up /= np.linalg.norm(up)
    centered = points - np.median(points, axis=0)
    inside = np.all(np.abs(centered) < np.quantile(np.abs(centered), .98, axis=0), axis=1)
    flat = centered[inside] - np.outer(centered[inside] @ up, up)
    _, vectors = np.linalg.eigh(flat.T @ flat)
    right = vectors[:, -1]
    right -= up * (right @ up)
    right /= np.linalg.norm(right)
    basis = np.column_stack((right, up, np.cross(right, up)))
    assert np.allclose(basis.T @ basis, np.eye(3)) and np.linalg.det(basis) > .999999
    return basis


def pack_ba(args):
    source = ROOT / 'benchmarks/configs/datasets.json'
    catalog = read(source)['datasets'][args.dataset]
    assert hashlib.file_digest(args.input.open('rb'), 'sha256').hexdigest() == catalog['sha256']
    captures = []
    for method in ['hgbp', 'powerba']:
        folder = args.work / method
        meta = read(folder / 'scene.json')
        assert meta['frames'] == list(range(21))
        n = meta['points'] + meta['cameras']
        xyz = np.fromfile(folder / 'scene.bin', dtype='<f4').reshape(21, n, 3).astype(float)
        assert np.isfinite(xyz).all()
        captures.append((meta, xyz))
    (hm, h), (pm, power) = captures
    assert hm == pm
    a = h[0] - np.median(h[0], axis=0)
    b = power[0] - np.median(power[0], axis=0)
    norms = np.linalg.norm(b, axis=1)
    valid = (norms > 1e-4) & (norms < np.quantile(norms, .98))
    scale = float(np.sum(a[valid] * b[valid]) / np.sum(b[valid] ** 2))
    offset = np.median(h[0] - power[0] * scale, axis=0)
    power = power * scale + offset
    error = float(np.max(np.linalg.norm(h[0] - power[0], axis=1) / np.maximum(1, np.linalg.norm(h[0], axis=1))))
    assert error < 2e-6, error
    center = np.median(h[-1], axis=0)
    basis = upright_basis(args.input, h[-1, :hm['points']])
    slug = 'ba-' + args.dataset.lower()
    final_display = (h[-1, :hm['points']] - center) @ basis
    view = dict(description='Upright overview focused on the central reconstruction. All sampled geometry remains in the replay; distant outliers may lie outside the viewport.',
        basis=np.eye(3).tolist(), focus_min=np.quantile(final_display, .05, axis=0).tolist(),
        focus_max=np.quantile(final_display, .95, axis=0).tolist(), yaw=.4, pitch=.75,
        fit_quantile=.01, vertical_fit_quantile=.01, horizontal_center=.5)
    (args.output / f'{slug}-view.json').write_text(json.dumps(view, indent=2) + '\n', newline='\n')
    for name, xyz in [('hgbp', h), ('powerba', power)]:
        ((xyz - center) @ basis).astype('<f4').tofile(args.output / f'{slug}-{name}.bin')
    hc, pc = read(args.work / 'hgbp/native.json'), read(args.work / 'powerba/ba_log.json')
    costs = []
    for cost, success in zip(pc['cost'], pc['step_is_successful']):
        costs.append(2 * cost if success or not costs else costs[-1])
    assert len(costs) == 21
    assert abs(costs[-1] / (2 * pc['cost'][-1]) - 1) < 1e-8 or not pc['step_is_successful'][-1]
    ht = [0., *np.cumsum(hc['outer_sec']).tolist()]
    pt = [0., *(np.asarray(pc['cumulative_time'][1:]) - pc['cumulative_time'][0]).tolist()]
    assert len(ht) == len(pt) == 21 and all(a <= b for a, b in zip(pt, pt[1:]))
    meta = dict(dataset=args.dataset, view=f'{slug}-view.json', points=hm['points'], cameras=hm['cameras'],
        full_points=catalog['points'], observations=catalog['observations'], input_sha256=catalog['sha256'],
        hgbp=dict(binary=f'{slug}-hgbp.bin', frames=list(range(21)), times=ht, costs=hc['costs'], total=ht[-1]),
        baseline=dict(name='PowerBA', threads=16, binary=f'{slug}-powerba.bin', frames=list(range(21)), times=pt, costs=costs, total=pt[-1]),
        provenance=dict(geometry='Actual accepted camera and landmark states after each of 20 outer iterations. Rejected PowerBA trials retain the preceding accepted state.',
            timing='Native iteration clocks from the sequential visualization captures. They exclude input parsing and are separate from the paper benchmark; capture I/O is outside the H-GBP outer timer.',
            interpolation='Display-only linear interpolation between recorded states; costs always use completed outer iterations.',
            sampling=f'{hm["points"]:,} uniformly sampled landmarks and all {hm["cameras"]:,} camera centers. One fixed initial-gauge scale/translation and a common camera-up display rotation, shared by both methods and all frames. No per-frame alignment.',
            initial_geometry_max_difference=error, baseline_display_scale=scale, baseline_display_translation=offset.tolist(), display_rotation=basis.tolist(), paper='https://arxiv.org/abs/2610.06978v1'))
    (args.output / f'{slug}.json').write_text(json.dumps(meta, indent=2) + '\n', newline='\n')
    print(f'{args.dataset}: {ht[-1]:.3f} / {pt[-1]:.3f} s; costs {hc["final_cost"]:.6f} / {costs[-1]:.6f}; initial geometry error {error:.3g}')


def restore_pgo(args):
    slug = 'pgo-globe100k'
    revision = '3e9ba9bf0'
    def blob(filename):
        return subprocess.check_output(['git', 'show', f'{revision}:site/data/{filename}'], cwd=ROOT)
    meta = json.loads(blob('pgo.json'))
    assert meta['dataset'] == 'Globe100k'
    for key in ['hgbp', 'baseline']:
        filename = f'{slug}-{key}.bin'
        (args.output / filename).write_bytes(blob(meta[key]['binary']))
        meta[key]['binary'] = filename
    meta['provenance']['geometry'] += f' Preserved unchanged from published capture commit {revision}.'
    (args.output / f'{slug}.json').write_text(json.dumps(meta, indent=2) + '\n', newline='\n')
    print('Preserved verified Globe100k keyframes, native cost histories and archived timing provenance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['ba', 'restore-pgo'])
    parser.add_argument('--dataset')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'site/data')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.kind == 'ba':
        assert args.dataset and args.input and args.work
        pack_ba(args)
    else:
        restore_pgo(args)
