"""Fetch exact H-GBP benchmark inputs from Hugging Face and verify both hashes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def relative_path(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')) or '\\' in value or ':' in value:
        raise ValueError(f'Unsafe relative path: {value}')
    return p


def url_for(repo_id, revision, name):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo_id):
        raise ValueError('Expected a Hugging Face namespace/repository identifier')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Use a fixed 40-character Hugging Face commit, not a moving branch')
    relative_path(name)
    return f'https://huggingface.co/datasets/{repo_id}/resolve/{revision}/' + urllib.parse.quote(name, safe='/')


def fetch(url, target):
    request = urllib.request.Request(url, headers={'User-Agent': 'HGBP-benchmark-downloader/1'})
    with urllib.request.urlopen(request, timeout=120) as src, target.open('wb') as dst:
        shutil.copyfileobj(src, dst, 4*1024*1024)


def install_one(name, row, expected, root, obtain):
    for key in ('suite', 'path', 'sha256'):
        if row.get(key) != expected[key]:
            raise ValueError(f'{name}: manifest does not match the solver catalog ({key})')
    relative_path(row['archive'])
    relative_path(row['path'])
    target = root/row['suite']/row['path']
    resolved_root = root.resolve()
    if not target.resolve().is_relative_to(resolved_root):
        raise ValueError('Dataset path escapes the output root')
    if target.exists():
        if digest(target) != row['sha256']:
            raise ValueError(f'{target}: existing file has the wrong hash; refusing to overwrite')
        return 'already verified'
    target.parent.mkdir(parents=True, exist_ok=True)
    # Temporary files stay on the target filesystem; only validated bytes become visible.
    with tempfile.TemporaryDirectory(prefix='.hgbp-download-', dir=target.parent) as tmp:
        archive, unpacked = Path(tmp)/'input.gz', Path(tmp)/'input'
        obtain(row['archive'], archive)
        if archive.stat().st_size != row['archive_bytes'] or digest(archive) != row['archive_sha256']:
            raise ValueError(f'{name}: compressed file checksum mismatch')
        limit, written = row['bytes'], 0
        with gzip.open(archive, 'rb') as src, unpacked.open('wb') as dst:
            for block in iter(lambda: src.read(4*1024*1024), b''):
                written += len(block)
                if written > limit:
                    raise ValueError(f'{name}: decompressed file exceeds the declared size')
                dst.write(block)
        if written != limit or digest(unpacked) != row['sha256']:
            raise ValueError(f'{name}: canonical input checksum mismatch')
        # Do not replace a file another downloader/user created while this one ran.
        os.link(unpacked, target)
    return 'downloaded and verified'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--suite', choices=['pgo', 'ba', 'all'], default='all')
    p.add_argument('--datasets', nargs='+', help='canonical dataset names, instead of the whole suite')
    p.add_argument('--output-root', type=Path, default=ROOT/'data')
    p.add_argument('--source-dir', type=Path, help='verify/install a local release without network access')
    p.add_argument('--verify-only', action='store_true', help='check existing uncompressed files without downloading')
    a = p.parse_args(argv)
    catalog = json.loads((ROOT/'configs/datasets.json').read_text(encoding='utf-8'))
    names = (catalog['subsets']['pgo']['all'] if a.suite != 'ba' else []) + (
        catalog['subsets']['ba']['main10'] if a.suite != 'pgo' else [])
    if a.datasets:
        unknown = set(a.datasets)-set(names)
        if unknown:
            p.error('Not in the selected release/suite: ' + ', '.join(sorted(unknown)))
        names = list(dict.fromkeys(a.datasets))
    if a.verify_only:
        for name in names:
            row = catalog['datasets'][name]
            path = a.output_root/row['suite']/row['path']
            if not path.is_file() or digest(path) != row['sha256']:
                raise ValueError(f'{name}: missing file or checksum mismatch: {path}')
            print(name, 'verified', flush=True)
        return
    if a.source_dir:
        def obtain(name, target):
            relative_path(name)
            shutil.copyfile(a.source_dir/name, target)
    else:
        release = json.loads((ROOT/'configs/data_release.json').read_text(encoding='utf-8'))
        if not release.get('revision'):
            p.error('The dataset release is not published yet; use --source-dir for a local package')
        def obtain(name, target):
            fetch(url_for(release['repo_id'], release['revision'], name), target)
    with tempfile.TemporaryDirectory(prefix='hgbp-manifest-') as tmp:
        manifest_path = Path(tmp)/'manifest.json'
        obtain('manifest.json', manifest_path)
        if not a.source_dir and digest(manifest_path) != release['manifest_sha256']:
            raise ValueError('Published manifest checksum mismatch')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported dataset manifest version')
    for name in names:
        status = install_one(name, manifest['datasets'][name], catalog['datasets'][name], a.output_root, obtain)
        print(name, status, flush=True)


if __name__ == '__main__':
    main()
