const fmt = (v, digits = 2) => v.toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

async function fetchJSON(url) {
  const response = await fetch(url); if (!response.ok) throw Error(`${url}: ${response.status}`);
  return response.json();
}

export async function initReplay(root) {
  const kind = root.dataset.scene, status = root.querySelector('[data-status]');
  try {
    const meta = await fetchJSON(`data/${kind}.json`);
    const n = meta.points + meta.cameras;
    const methods = {};
    for (const name of ['hgbp', 'baseline']) {
      const config = meta[name], response = await fetch(`data/${config.binary}`);
      if (!response.ok) throw Error(`Missing geometry: ${response.status}`);
      const data = new Float32Array(await response.arrayBuffer());
      if (data.length !== config.frames.length * n * 3 || !data.every(Number.isFinite)) throw Error('Invalid geometry capture');
      const canvas = root.querySelector(`[data-view="${name}"]`);
      methods[name] = {...config, data, canvas, ctx: canvas.getContext('2d'), projected: new Float32Array(n * 2)};
    }
    const first = methods.hgbp.data, last = first.subarray(first.length - n * 3);
    // One common display frame for both solvers, fixed throughout the replay.
    const center = [0, 0, 0];
    for (let axis = 0; axis < 3; ++axis) {
      const values = Array.from({length: n}, (_, i) => last[3 * i + axis]).sort((a, b) => a - b);
      const lo = values[Math.floor(n * .005)], hi = values[Math.floor(n * .995)];
      center[axis] = (hi + lo) / 2;
    }
    let yaw = kind === 'ba' ? -.5 : .4, pitch = kind === 'ba' ? -.85 : .22;
    let zoom = 1, position = 0, running = false, visible = false, lastTick = 0, lastPaint = 0, smooth = true;
    const duration = kind === 'ba' ? 18000 : 23000;
    const T = Math.max(meta.hgbp.total, meta.baseline.total), k = kind === 'ba' ? .7 : 5;
    const timeAt = u => k * Math.expm1(u * Math.log1p(T / k));
    if (kind === 'pgo') document.querySelector('[data-pgo-speed]').textContent = `${fmt(meta.baseline.total / meta.hgbp.total, 1)}\u00d7`;
    const logs = [...meta.hgbp.costs, ...meta.baseline.costs].map(Math.log10);
    const lowCost = Math.min(...logs) - .06, highCost = Math.max(...logs) + .06;
    const short = value => value >= 1e6 ? `${Number((value / 1e6).toPrecision(2))}M` : value >= 1e3 ? `${Number((value / 1e3).toPrecision(2))}k` : String(Number(value.toPrecision(2)));
    function drawCurve(name, time, outer) {
      const m = methods[name], canvas = root.querySelector(`[data-curve="${name}"]`), ctx = canvas.getContext('2d');
      const W = canvas.clientWidth, H = canvas.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
      if (!W || !H) return;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);}
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const left = 44, right = W - 10, top = 27, bottom = H - 30;
      const x = t => left + Math.log1p(t / k) / Math.log1p(T / k) * (right - left);
      const y = cost => bottom - (Math.log10(cost) - lowCost) / (highCost - lowCost) * (bottom - top);
      ctx.font = '11px "DM Sans", sans-serif'; ctx.lineWidth = .65; ctx.textAlign = 'right';
      for (const p of [0, .5, 1]) {
        const yy = top + p * (bottom - top), cost = 10 ** (highCost - p * (highCost - lowCost));
        ctx.strokeStyle = '#e1e5e4'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
        ctx.fillStyle = '#68777b'; ctx.fillText(short(cost), left - 6, yy + 4);
      }
      ctx.textAlign = 'center';
      for (const u of [0, .5, 1]) {
        ctx.fillText(fmt(timeAt(u), u ? 1 : 0), left + u * (right - left), bottom + 14);
      }
      ctx.fillStyle = '#40565d'; ctx.font = '12px "DM Sans", sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('Cost (log)', 7, 16); ctx.textAlign = 'right'; ctx.fillText('Time [s]', right, H - 2);
      ctx.strokeStyle = '#7c919766'; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(x(time), top); ctx.lineTo(x(time), bottom); ctx.stroke(); ctx.setLineDash([]);
      ctx.strokeStyle = name === 'hgbp' ? '#205fc4' : '#b96c42'; ctx.fillStyle = ctx.strokeStyle; ctx.lineWidth = 1.8;
      ctx.beginPath();
      for (let i = 0; i <= outer; ++i) {if (!i) ctx.moveTo(x(m.times[i]), y(m.costs[i])); else ctx.lineTo(x(m.times[i]), y(m.costs[i]));}
      ctx.stroke();
      for (let i = 0; i <= outer; ++i) {ctx.beginPath(); ctx.arc(x(m.times[i]), y(m.costs[i]), 2.1, 0, 2 * Math.PI); ctx.fill();}
    }
    let fit = null;
    function fitView(W, H) {
      const key = `${W},${H},${yaw},${pitch}`;
      if (fit?.key === key) return fit;
      const xs = [], ys = [], sy = Math.sin(yaw), cy = Math.cos(yaw), sp = Math.sin(pitch), cp = Math.cos(pitch);
      for (const frame of [first.subarray(0, n * 3), last]) {
        for (let i = 0; i < n; ++i) {
          const x = frame[i * 3] - center[0], y = frame[i * 3 + 1] - center[1], z = frame[i * 3 + 2] - center[2];
          xs.push(cy * x + sy * z); ys.push(cp * y - sp * (-sy * x + cy * z));
        }
      }
      xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
      const low = Math.floor(xs.length * .005), high = Math.floor(xs.length * .995);
      fit = {key, x: (xs[low] + xs[high]) / 2, y: (ys[low] + ys[high]) / 2,
        scale: .89 * Math.min(W / (xs[high] - xs[low]), H / (ys[high] - ys[low]))};
      return fit;
    }
    const play = root.querySelector('[data-play]'), scrub = root.querySelector('[data-scrub]');
    const clock = root.querySelector('[data-clock]');
    const updateButton = () => {play.textContent = running ? 'Pause replay' : 'Play replay'; play.setAttribute('aria-label', play.textContent);};
    for (const u of [0, .25, .5, .75, 1]) {
      const tick = document.createElement('span'); tick.textContent = `${fmt(timeAt(u), u ? 1 : 0)} s`;
      root.querySelector('[data-ticks]').append(tick);
    }
    const paragraphs = [
      `Geometry: ${meta.provenance.geometry}`,
      `Timing: ${meta.provenance.timing}`,
      `Display: ${meta.provenance.sampling} A single fixed viewing frame is shared by both methods. The central 99% coordinate range sets the viewport; extreme outliers may lie outside it. Drag either view to rotate both; use the mouse wheel to zoom.`,
      `Motion: ${meta.provenance.interpolation} Interpolated motion is for presentation only; cost values always use the last completed outer iteration. Switch to recorded states to remove interpolation. Both solvers use the same nonlinear clock, t = k(exp(u log(1+T/k)) − 1). This is not real-time playback.`
    ];
    paragraphs.forEach(text => {const p = document.createElement('p'); p.textContent = text; root.querySelector('[data-provenance]').append(p);});

    function renderMethod(name, time) {
      const m = methods[name], canvas = m.canvas, ctx = m.ctx;
      const W = canvas.clientWidth, H = canvas.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
      if (!W || !H) return;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);}
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      let outer = 0;
      while (outer + 1 < m.times.length && m.times[outer + 1] <= time) ++outer;
      let a = 0;
      while (a + 1 < m.frames.length && m.times[m.frames[a + 1]] <= time) ++a;
      const b = Math.min(a + 1, m.frames.length - 1);
      const t0 = m.times[m.frames[a]], t1 = m.times[m.frames[b]];
      const t = smooth && t1 > t0 ? clamp((time - t0) / (t1 - t0), 0, 1) : 0;
      const stride = n * 3, A = a * stride, B = b * stride;
      const sy = Math.sin(yaw), cy = Math.cos(yaw), sp = Math.sin(pitch), cp = Math.cos(pitch);
      const view = fitView(W, H), scale = view.scale * zoom;
      for (let i = 0; i < n; ++i) {
        const j = i * 3;
        const x = (1 - t) * m.data[A + j] + t * m.data[B + j] - center[0];
        const y = (1 - t) * m.data[A + j + 1] + t * m.data[B + j + 1] - center[1];
        const z = (1 - t) * m.data[A + j + 2] + t * m.data[B + j + 2] - center[2];
        const xx = cy * x + sy * z, zz = -sy * x + cy * z;
        m.projected[i * 2] = W / 2 + (xx - view.x) * scale;
        m.projected[i * 2 + 1] = H / 2 - (cp * y - sp * zz - view.y) * scale;
      }
      const p = m.projected;
      if (kind === 'pgo') {
        ctx.strokeStyle = '#202d34a0'; ctx.lineWidth = .4;
        ctx.beginPath();
        for (let i = 0; i < meta.points; ++i) {
          const j = 2 * i;
          if (!i) ctx.moveTo(p[j], p[j + 1]); else ctx.lineTo(p[j], p[j + 1]);
        }
        ctx.stroke();
        ctx.strokeStyle = '#c33f4152'; ctx.lineWidth = .5; ctx.beginPath();
        meta.loop_edges.forEach(([a, b]) => {ctx.moveTo(p[2 * a], p[2 * a + 1]); ctx.lineTo(p[2 * b], p[2 * b + 1]);});
        ctx.stroke();
      }
      ctx.fillStyle = kind === 'pgo' ? '#172d3c88' : '#354d5d70';
      const size = W < 300 ? .8 : 1.15;
      ctx.beginPath();
      for (let i = 0; i < meta.points; ++i) ctx.rect(p[2 * i], p[2 * i + 1], size, size);
      ctx.fill();
      if (meta.cameras) {
        ctx.fillStyle = name === 'hgbp' ? '#205fc4c0' : '#b96c42c0'; ctx.beginPath();
        for (let i = meta.points; i < n; ++i) ctx.rect(p[2 * i] - 1, p[2 * i + 1] - 1, 2.2, 2.2);
        ctx.fill();
      }
      const badge = root.querySelector(`[data-step="${name}"]`);
      badge.textContent = outer >= 20 ? '20 / 20 · complete' : `${String(outer).padStart(2, '0')} / 20 outer`;
      badge.classList.toggle('complete', outer >= 20);
      root.querySelector(`[data-cost="${name}"]`).textContent = fmt(m.costs[Math.min(outer, m.costs.length - 1)], 1);
      root.querySelector(`[data-finish="${name}"]`).textContent = `${fmt(m.total)} s ${outer >= 20 ? '· finished' : 'total'}`;
      drawCurve(name, time, outer);
    }
    function render() {
      const time = timeAt(position); scrub.value = Math.round(position * 1000); clock.textContent = `${fmt(time)} s`;
      renderMethod('hgbp', time); renderMethod('baseline', time);
    }
    play.disabled = false; status.hidden = true;
    play.addEventListener('click', () => {if (position >= .9999) position = 0; running = !running; lastTick = 0; updateButton(); render();});
    scrub.addEventListener('input', () => {position = Number(scrub.value) / 1000; running = false; updateButton(); render();});
    root.querySelector('[data-reset]').addEventListener('click', () => {position = 0; running = false; updateButton(); render();});
    root.querySelector('[data-motion]').addEventListener('click', event => {smooth = !smooth; event.target.textContent = smooth ? 'Show recorded states only' : 'Enable smooth display motion'; render();});
    for (const m of Object.values(methods)) {
      let drag = null;
      m.canvas.addEventListener('pointerdown', e => {drag = [e.clientX, e.clientY]; m.canvas.setPointerCapture(e.pointerId);});
      m.canvas.addEventListener('pointermove', e => {
        if (!drag) return; yaw += (e.clientX - drag[0]) * .008; pitch = clamp(pitch + (e.clientY - drag[1]) * .008, -1.55, 1.55);
        drag = [e.clientX, e.clientY]; render();
      });
      m.canvas.addEventListener('pointerup', () => {drag = null;});
      m.canvas.addEventListener('pointercancel', () => {drag = null;});
      m.canvas.addEventListener('wheel', e => {e.preventDefault(); zoom = clamp(zoom * Math.exp(-e.deltaY * .001), .25, 4); render();}, {passive: false});
    }
    new ResizeObserver(render).observe(root);
    new IntersectionObserver(entries => {visible = entries[0].isIntersecting; lastTick = 0;}).observe(root);
    // Start at the original input; users choose when to run either comparison.
    render(); updateButton();
    function frame(now) {
      if (running && visible && !document.hidden) {
        if (lastTick) position = Math.min(1, position + Math.min(now - lastTick, 80) / duration);
        if (now - lastPaint >= 33 || position >= 1) {render(); lastPaint = now;}
        if (position >= 1) {running = false; updateButton();}
      }
      lastTick = now; requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    root.dataset.ready = 'true';
  } catch (error) {
    status.hidden = false; status.textContent = `Replay unavailable: ${error.message}. The complete benchmark results remain available below.`;
    console.error(error);
  }
}
