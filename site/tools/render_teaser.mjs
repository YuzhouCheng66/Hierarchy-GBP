// Use the live project's renderer, not a second illustration implementation.
import {createRequire} from 'node:module';
import {mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {join} from 'node:path';
import {drawMechanism, MECHANISM_LOOP_MS} from '../mechanism.js';

const require = createRequire(import.meta.url);
const {createCanvas, GlobalFonts} = require('@napi-rs/canvas');
const output = process.argv[2];
if (!output) throw Error('Usage: node site/tools/render_teaser.mjs <frame-directory>');
const font = fileURLToPath(new URL('fonts/DM-Sans.ttf', import.meta.url));
if (!GlobalFonts.registerFromPath(font, 'DM Sans')) throw Error('Cannot load the project font');

const frames = 180, padding = 12, width = 1440, height = 476;
const canvas = createCanvas(width, height), ctx = canvas.getContext('2d');
await mkdir(output, {recursive: true});
for (let i = 0; i < frames; ++i) {
  ctx.resetTransform(); ctx.clearRect(0, 0, width, height);
  ctx.save(); ctx.translate(padding, padding);
  drawMechanism(ctx, width - padding * 2, height - padding * 2, i / frames);
  ctx.restore();
  // Composite all translucent strokes onto white, including antialiased edges.
  ctx.globalCompositeOperation = 'destination-over';
  ctx.fillStyle = '#ffffff'; ctx.fillRect(0, 0, width, height);
  ctx.globalCompositeOperation = 'source-over';
  await writeFile(join(output, `${String(i).padStart(3, '0')}.png`), canvas.toBuffer('image/png'));
}
await writeFile(join(output, 'manifest.json'), JSON.stringify({frames, width, height, loop_ms: MECHANISM_LOOP_MS}));
console.log(`Rendered ${frames} frames from site/mechanism.js with the project font on white.`);
