const palette = ['#528acb', '#d39448', '#70a483', '#aa8bba'];
const clamp = x => Math.max(0, Math.min(1, x));
const ease = x => { x = clamp(x); return x * x * (3 - 2 * x); };

export function drawMechanism(ctx, width, height, phase) {
  ctx.clearRect(0, 0, width, height);
  const w = 1120, h = 300;
  const scale = Math.min(width / w, height / h);
  ctx.save();
  ctx.translate((width - w * scale) / 2, (height - h * scale) / 2);
  ctx.scale(scale, scale);
  const hi = 1 - .94 * ease((phase - .04) / .25);
  const lo = 1 - .93 * ease((phase - .43) / .32);
  const point = (x, y, cx, high, low) => {
    const z = low * Math.sin(2 * x + .4) * Math.cos(1.7 * y) * 31
      + high * Math.sin(13 * x + 8 * y) * Math.cos(10 * y - 3 * x) * 9;
    return [cx + x * 108 + y * 66, 190 + y * 37 - x * 17 - z];
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
  surface(212, hi, 1, phase < .36);
  surface(908, hi * .3, lo, phase > .72);
  ctx.font = '500 13px "DM Sans", sans-serif'; ctx.textAlign = 'center';
  ctx.fillStyle = '#34464c';
  ctx.fillText('Fine graph', 212, 279);
  ctx.fillText('Recovered fine graph', 908, 279);

  function arrow(a, control, b, color, offset) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.4;
    ctx.beginPath(); ctx.moveTo(...a); ctx.quadraticCurveTo(...control, ...b); ctx.stroke();
    const angle = Math.atan2(b[1] - control[1], b[0] - control[0]);
    line(b, [b[0] - 6 * Math.cos(angle - .45), b[1] - 6 * Math.sin(angle - .45)], color, 1.4);
    line(b, [b[0] - 6 * Math.cos(angle + .45), b[1] - 6 * Math.sin(angle + .45)], color, 1.4);
    const t = (phase * 3 + offset) % 1, u = 1 - t;
    dot([u * u * a[0] + 2 * u * t * control[0] + t * t * b[0], u * u * a[1] + 2 * u * t * control[1] + t * t * b[1]], 3.3, '#205fc4');
  }
  arrow([320, 126], [362, 42], [440, 61], '#81959c', 0);
  arrow([510, 59], [560, 46], [610, 59], '#81959c', .33);
  arrow([688, 63], [771, 42], [806, 125], '#81959c', .66);
  arrow([382, 204], [560, 235], [735, 204], '#adb7b7', .5);
  const nodes = [[-26, -15], [27, -13], [28, 18], [-25, 16]];
  [475, 649].forEach((cx, n) => {
    const correction = n ? lo : 1;
    const p = nodes.map(([x, y]) => [cx + x, 67 + y * correction]);
    [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].forEach(([a, b]) => line(p[a], p[b], '#8d9ba4', 1.2));
    p.forEach((q, i) => dot(q, 5.5, palette[i]));
  });
  ctx.font = '11px "DM Sans", sans-serif'; ctx.fillStyle = '#68797c';
  ctx.fillText('Abstraction', 355, 57);
  ctx.fillText('Coarse correction', 562, 25);
  ctx.fillText('Recovery', 763, 57);
  ctx.fillText('Local GBP refinement', 560, 250);
  ctx.restore();
}

export function initMechanism() {
  const canvas = document.querySelector('#mechanism'), button = document.querySelector('#mechanism-toggle');
  const ctx = canvas.getContext('2d'), reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let paused = reduced, visible = true, elapsed = 0, last = 0, width = 0, height = 0;
  const labels = ['01  Local GBP smooths high-frequency error', '02  Abstraction transfers coordinated modes', '03  Coarse correction removes low-frequency error', '04  Recovery brings global correction to the fine graph'];
  const label = document.querySelector('#mechanism-phase');
  const render = () => {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    drawMechanism(ctx, width, height, (elapsed / 11000) % 1);
    label.textContent = labels[Math.min(3, Math.floor((elapsed / 11000) % 1 * 4))];
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
