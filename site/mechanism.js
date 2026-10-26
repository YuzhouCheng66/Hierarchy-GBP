const palette = ['#528acb', '#d39448', '#70a483', '#aa8bba'];
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };
export const MECHANISM_LOOP_MS = 11000 / 1.5;

export function coarsePoint(cx, cy, u, v, distortion) {
  return [
    cx + 25 * (u + distortion * (.24 * v + .1 * Math.sin(Math.PI * v) + .12 * Math.sin(Math.PI * u))),
    cy + 25 * (v + distortion * (.2 * u + .3 * Math.cos(Math.PI * u / 2) - .12 * Math.sin(Math.PI * v)))
  ];
}

export function drawMechanism(ctx, width, height, phase) {
  ctx.clearRect(0, 0, width, height);
  const w = 940, h = 300;
  const scale = Math.min(width / w, height / h);
  ctx.save();
  ctx.translate((width - w * scale) / 2, (height - h * scale) / 2);
  ctx.scale(scale, scale);
  const hi = 1 - .94 * ease((phase - .04) / .25);
  const lo = 1 - .93 * ease((phase - .7) / .23);
  const point = (x, y, cx, high, low) => {
    const z = low * Math.sin(2 * x + .4) * Math.cos(1.7 * y) * 31
      + high * Math.sin(13 * x + 8 * y) * Math.cos(10 * y - 3 * x) * 9;
    return [cx + x * 108 + y * 66, 185 + y * 37 - x * 17 - z];
  };
  const line = (a, b, color, weight = 1) => {
    ctx.strokeStyle = color; ctx.lineWidth = weight;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  };
  const dot = (p, r, color) => {
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(...p, r, 0, 2 * Math.PI); ctx.fill();
  };
  function surface(cx, high, low, active) {
    for (let direction = 0; direction < 2; ++direction) {
      for (let i = 0; i <= 18; ++i) {
        ctx.beginPath(); ctx.strokeStyle = '#9ba6aa65'; ctx.lineWidth = .55;
        for (let j = 0; j <= 65; ++j) {
          const a = -1 + i / 9, b = -1 + j / 32.5;
          const p = point(direction ? a : b, direction ? b : a, cx, high, low);
          if (!j) ctx.moveTo(...p); else ctx.lineTo(...p);
        }
        ctx.stroke();
      }
    }
    const nodes = [];
    for (let y = 0; y < 5; ++y) for (let x = 0; x < 7; ++x) {
      const i = y * 7 + x, a = -.9 + x * .3, b = -.85 + y * .425;
      nodes.push(point(a, b, cx, high, low));
      if (x) line(nodes[i - 1], nodes[i], '#6d777866', .9);
      if (y) line(nodes[i - 7], nodes[i], '#6d777866', .9);
    }
    for (let i = 0; i < nodes.length; ++i) {
      const group = (i % 7 >= 3 ? 1 : 0) + (i >= 17 ? 2 : 0);
      dot(nodes[i], 3.6, palette[group]);
      ctx.strokeStyle = '#fff'; ctx.lineWidth = .65; ctx.stroke();
      if (active && i < nodes.length - 1 && i % 7 < 6) {
        const t = (phase * 8 + i * .113) % 1;
        const a = nodes[i], b = nodes[i + 1];
        dot([a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t], 2, '#205fc4');
      }
    }
  }
  surface(190, hi, 1, phase < .36);
  surface(750, hi * .3, lo, false);
  ctx.font = '500 24px "DM Sans", sans-serif'; ctx.textAlign = 'center';
  ctx.fillStyle = '#34464c';
  ctx.fillText('Fine graph', 190, 279);
  ctx.fillText('Recovered fine graph', 750, 279);

  function arrow(a, control, b, color, start, end) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(...a); ctx.quadraticCurveTo(...control, ...b); ctx.stroke();
    const angle = Math.atan2(b[1] - control[1], b[0] - control[0]);
    line(b, [b[0] - 6 * Math.cos(angle - .45), b[1] - 6 * Math.sin(angle - .45)], color, 1.4);
    line(b, [b[0] - 6 * Math.cos(angle + .45), b[1] - 6 * Math.sin(angle + .45)], color, 1.4);
    if (phase >= start && phase <= end) {
      const t = (phase - start) / (end - start), u = 1 - t;
      dot([u * u * a[0] + 2 * u * t * control[0] + t * t * b[0], u * u * a[1] + 2 * u * t * control[1] + t * t * b[1]], 4, '#205fc4');
    }
  }
  arrow([296, 121], [333, 47], [355, 56], '#81959c', .25, .43);
  arrow([425, 54], [467, 43], [506, 54], '#81959c', .43, .7);
  arrow([584, 58], [623, 48], [650, 120], '#81959c', .7, 1);
  // The lower branch contrasts slow, unaccelerated GBP; it is not a V-cycle stage.
  ctx.setLineDash([4, 5]);
  arrow([366, 209], [470, 225], [574, 209], '#bd8b80', 2, 3);
  ctx.setLineDash([]);
  ctx.save(); ctx.translate(460 + 14 * phase, 201);
  ctx.strokeStyle = '#96766e'; ctx.lineWidth = 1.2; ctx.beginPath();
  for (let t = 0; t <= 4 * Math.PI; t += .13) {
    const r = .3 + .48 * t, x = r * Math.cos(t), y = r * Math.sin(t);
    if (!t) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke(); ctx.beginPath(); ctx.moveTo(-8, 7); ctx.lineTo(12, 7); ctx.quadraticCurveTo(19, 7, 18, -1);
  ctx.moveTo(18, -1); ctx.lineTo(16, -5); ctx.moveTo(18, -1); ctx.lineTo(21, -5); ctx.stroke(); ctx.restore();
  const nodes = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  [390, 545].forEach((cx, n) => {
    const distortion = n ? 1 - ease((phase - .43) / .27) : 1;
    const point = (u, v) => coarsePoint(cx, 63, u, v, distortion);
    const curve = (a, b, color, width) => {
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      for (let j = 0; j <= 32; ++j) {
        const t = j / 32, p = point(a[0] * (1 - t) + b[0] * t, a[1] * (1 - t) + b[1] * t);
        if (!j) ctx.moveTo(...p); else ctx.lineTo(...p);
      }
      ctx.stroke();
    };
    for (let j = 0; j <= 6; ++j) {
      const v = -1 + j / 3;
      curve([-1, v], [1, v], '#9eafb54d', .6);
      curve([v, -1], [v, 1], '#9eafb54d', .6);
    }
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].forEach(([a, b]) => curve(nodes[a], nodes[b], '#7e929c', 1.3));
    const p = nodes.map(([u, v]) => point(u, v));
    p.forEach((q, i) => dot(q, 5.5, palette[i]));
  });
  ctx.font = '22.5px "DM Sans", sans-serif'; ctx.fillStyle = '#68797c';
  ctx.fillText('2  Abstraction', 270, 28);
  ctx.fillText('3  Coarse correction', 467, 20);
  ctx.fillText('4  Recovery', 670, 28);
  ctx.fillStyle = '#205fc4'; ctx.fillText('1  GBP smoothing', 145, 100);
  ctx.fillStyle = '#96766e'; ctx.font = '21px "DM Sans", sans-serif';
  ctx.fillText('Pure GBP:', 470, 238);
  ctx.fillText('slow low-frequency correction', 470, 262);
  ctx.restore();
}

export function initMechanism() {
  const canvas = document.querySelector('#mechanism'), button = document.querySelector('#mechanism-toggle');
  const ctx = canvas.getContext('2d'), reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let paused = reduced, visible = true, elapsed = reduced ? MECHANISM_LOOP_MS * .85 : 0, last = 0, width = 0, height = 0;
  const labels = ['01  GBP smooths high-frequency error', '02  Abstraction represents low-frequency modes', '03  Correction reduces low-frequency error', '04  Recovery transfers the correction to the fine graph'];
  const label = document.querySelector('#mechanism-phase');
  const render = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const phase = (elapsed / MECHANISM_LOOP_MS) % 1;
    drawMechanism(ctx, width, height, phase);
    label.textContent = labels[phase < .29 ? 0 : phase < .43 ? 1 : phase < .7 ? 2 : 3];
  };
  new ResizeObserver(entries => {
    ({width, height} = entries[0].contentRect);
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr); render();
  }).observe(canvas);
  new IntersectionObserver(entries => {visible = entries[0].isIntersecting; last = 0;}).observe(canvas);
  const update = () => {button.textContent = paused ? 'Play animation' : 'Pause animation'; button.setAttribute('aria-pressed', String(paused));};
  button.addEventListener('click', () => {paused = !paused; last = 0; update();}); update();
  function frame(now) {
    if (!paused && visible && !document.hidden) {if (last) elapsed += Math.min(now - last, 60); render();}
    last = now; requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
