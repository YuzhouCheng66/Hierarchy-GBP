"""Build byte-preserving, hash-verified benchmark archives for a dataset mirror."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8', newline='\n')


def validate_pgo(path, expected):
    is3 = expected['space'] == 'SE3'
    vtag, etag = ('VERTEX_SE3:QUAT', 'EDGE_SE3:QUAT') if is3 else ('VERTEX_SE2', 'EDGE_SE2')
    vertices, edges = set(), []
    with path.open(encoding='ascii') as f:
        for line_number, line in enumerate(f, 1):
            fields = line.split()
            if not fields or fields[0].startswith('#'):
                continue
            kind = fields[0]
            if kind == vtag:
                assert len(fields) == (9 if is3 else 5), (path, line_number)
                node = int(fields[1])
                assert node not in vertices, (path, 'duplicate vertex', node)
                vertices.add(node)
                values = [float(x) for x in fields[2:]]
                assert all(math.isfinite(x) for x in values)
                if is3:
                    assert abs(sum(x*x for x in values[3:])-1) < 1e-4
            elif kind == etag:
                assert len(fields) == (31 if is3 else 12), (path, line_number)
                edges.append((int(fields[1]), int(fields[2])))
                assert all(math.isfinite(float(x)) for x in fields[3:])
            else:
                raise ValueError(f'Unsupported g2o record {kind}: {path}:{line_number}')
    assert len(vertices) == expected['nodes'] and len(edges) == expected['factors']
    assert all(i in vertices and j in vertices for i, j in edges)
    return {'nodes': len(vertices), 'factors': len(edges), 'finite_values': True,
            'valid_factor_endpoints': True}


def validate_bal(path, expected):
    with path.open(encoding='ascii') as f:
        cameras, points, observations = map(int, f.readline().split())
        assert (cameras, points, observations) == tuple(expected[k] for k in
                                                     ('cameras', 'points', 'observations'))
        for index in range(observations):
            fields = f.readline().split()
            assert len(fields) == 4, (path, 'observation', index)
            assert 0 <= int(fields[0]) < cameras and 0 <= int(fields[1]) < points
            assert all(math.isfinite(float(x)) for x in fields[2:])
        count = 0
        for line in f:
            for token in line.split():
                assert math.isfinite(float(token)), path
                count += 1
        assert count == 9*cameras + 3*points, (path, count)
    return {'cameras': cameras, 'points': points, 'observations': observations,
            'finite_values': True, 'valid_observation_indices': True}


def package(pgo_root, ba_root, output):
    catalog = json.loads((ROOT/'configs/datasets.json').read_text(encoding='utf-8'))
    names = catalog['subsets']['pgo']['all'] + catalog['subsets']['ba']['main10']
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema_version': 1, 'format': 'individual gzip files; exact canonical bytes',
                'datasets': {}, 'subsets': {'pgo': names[:9], 'ba': names[9:]}}
    for name in names:
        row = dict(catalog['datasets'][name])
        source = (pgo_root if row['suite'] == 'pgo' else ba_root)/row['path']
        if digest(source) != row['sha256']:
            raise ValueError(f'Canonical input hash mismatch: {name}')
        checks = validate_pgo(source, row) if row['suite'] == 'pgo' else validate_bal(source, row)
        archive = row['suite'] + '-' + row['path'].replace('/', '-') + '.gz'
        with source.open('rb') as src, (output/archive).open('wb') as raw:
            # No local filename or timestamp is embedded in the distributed gzip.
            with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0, compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 4*1024*1024)
        with gzip.open(output/archive, 'rb') as f:
            h = hashlib.sha256()
            for block in iter(lambda: f.read(4*1024*1024), b''):
                h.update(block)
            assert h.hexdigest() == row['sha256'], name
        row.update(archive=archive, archive_sha256=digest(output/archive),
                   archive_bytes=(output/archive).stat().st_size,
                   bytes=source.stat().st_size, validation=checks)
        manifest['datasets'][name] = row
        print(name, row['bytes'], row['archive_bytes'], 'verified', flush=True)
    write_json(output/'manifest.json', manifest)
    with (output/'metadata.csv').open('w', encoding='utf-8', newline='') as f:
        columns = ['dataset', 'suite', 'space', 'nodes', 'factors', 'cameras', 'points',
                   'observations', 'bytes', 'archive_bytes', 'sha256', 'archive', 'preprocessing']
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for name, row in manifest['datasets'].items():
            writer.writerow(dict(dataset=name, **{key: row.get(key, '') for key in columns[1:]}))
    for src, dest in [('DATASET_CARD.md', 'README.md'), ('DATA_NOTICES.md', 'NOTICE.md')]:
        shutil.copyfile(Path(__file__).with_name(src), output/dest)
    print('TOTAL', sum(r['archive_bytes'] for r in manifest['datasets'].values()), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--pgo-root', required=True, type=Path)
    p.add_argument('--ba-root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='New directory; never overwrite an existing release')
    a = p.parse_args()
    package(a.pgo_root, a.ba_root, a.output)


if __name__ == '__main__':
    main()
