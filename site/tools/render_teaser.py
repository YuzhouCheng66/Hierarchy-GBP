"""Encode the live project animation as a white-background README GIF.

Requires Pillow, NumPy, Node.js and @napi-rs/canvas (local install or NODE_PATH).
The geometry, typography and timing come directly from site/mechanism.js.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, help='Optional contact sheet at GitHub README width')
    parser.add_argument('--work-root', type=Path, help='Parent for automatically removed frame files')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='hgbp-teaser-', dir=args.work_root) as folder:
        folder = Path(folder)
        subprocess.run(['node', str(ROOT / 'site/tools/render_teaser.mjs'), str(folder)], check=True)
        meta = json.loads((folder / 'manifest.json').read_text())
        images = [Image.open(folder / f'{i:03}.png').convert('RGB') for i in range(meta['frames'])]

    # One palette for the entire loop avoids temporal color flicker.
    sample = Image.new('RGB', (478, 158 * 12), 'white')
    for row, index in enumerate(np.linspace(0, len(images) - 1, 12, dtype=int)):
        sample.paste(images[index].resize((478, 158), Image.Resampling.LANCZOS), (0, row * 158))
    anchors = ['#ffffff', '#528acb', '#d39448', '#70a483', '#aa8bba', '#205fc4',
               '#34464c', '#68797c', '#96766e', '#81959c', '#7e929c']
    adaptive = sample.quantize(colors=256 - len(anchors), dither=Image.Dither.NONE)
    palette = Image.new('P', (1, 1))
    palette.putpalette([channel for color in anchors for channel in bytes.fromhex(color[1:])]
                       + adaptive.getpalette()[:(256 - len(anchors)) * 3])
    palette.load()
    quantized = []
    for im in images:
        result = im.quantize(palette=palette, dither=Image.Dither.NONE)
        white = Image.fromarray(np.all(np.asarray(im) == 255, axis=2).astype('uint8') * 255)
        result.paste(0, mask=white)
        quantized.append(result)
    boundaries = np.rint(np.arange(len(images) + 1) * meta['loop_ms'] / (10 * len(images))).astype(int) * 10
    durations = np.diff(boundaries).tolist()
    path = ROOT / 'code/docs/assets/teaser.gif'
    quantized[0].save(path, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=False, disposal=1, background=0)

    previews = []
    with Image.open(path) as gif:
        assert gif.n_frames == meta['frames'] and gif.size == (meta['width'], meta['height'])
        total = 0
        for i in range(gif.n_frames):
            gif.seek(i); total += gif.info['duration']
            assert 'transparency' not in gif.info
            rgb = gif.convert('RGB'); w, h = rgb.size
            for box in [(0, 0, w, 12), (0, h - 12, w, h), (0, 0, 12, h), (w - 12, 0, w, h)]:
                assert rgb.crop(box).getextrema() == ((255, 255),) * 3, 'White margin must remain clean'
            if i in [0, 54, 99, 144, 179]:
                previews.append(rgb.resize((830, round(h * 830 / w)), Image.Resampling.LANCZOS))
        assert abs(total - meta['loop_ms']) <= 10
    if args.preview:
        preview = Image.new('RGB', (830, sum(im.height for im in previews)), 'white')
        y = 0
        for im in previews:
            preview.paste(im, (0, y)); y += im.height
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        preview.save(args.preview)
    print(f'{path}: {path.stat().st_size / 1024**2:.2f} MiB; {total / 1000:.2f} s/loop')
    print('Verified: live project renderer, same font/timing, 180 opaque frames, white edges.')


if __name__ == '__main__':
    main()
