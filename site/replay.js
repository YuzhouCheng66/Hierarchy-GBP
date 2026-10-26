const fmt = (v, digits = 2) => v.toLocaleString('en-US', {maximumFractionDigits: digits, minimumFractionDigits: digits});
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

async function fetchJSON(url) {
  const response = await fetch(url); if (!response.ok) throw Error(`${url}: ${response.status}`);
  return response.json();
}

export async function initReplay(root, metadataPath) {
  const kind = root.dataset.scene, status = root.querySelector('[data-status]');
  const events = new AbortController();
  const listen = (element, type, callback, options = {}) => element.addEventListener(type, callback, {...options, signal: events.signal});
  let disposed = false, frameRequest;
  const observers = [];
  const dispose = () => {disposed = true; events.abort(); cancelAnimationFrame(frameRequest); observers.forEach(observer => observer.disconnect());};
  root.dataset.ready = 'false';
  root.querySelector('[data-play]').disabled = true;
  root.querySelector('[data-ticks]').replaceChildren();
  root.querySelector('[data-provenance]').replaceChildren();
  root.querySelector('[data-motion]').textContent = 'Show recorded states only';
  status.hidden = false; status.textContent = 'Loading recorded states...';
  try {
    const meta = await fetchJSON(`${metadataPath ?? `data/${kind}.json`}?v=20261026-7`);
    const display = meta.view ? await fetchJSON(`data/${meta.view}?v=20261026-7`) : null;
    const n = meta.points + meta.cameras;
    const methods = {};
    for (const name of ['hgbp', 'baseline']) {
      const config = meta[name], response = await fetch(`data/${config.binary}?v=20261026-2`);
      if (!response.ok) throw Error(`Missing geometry: ${response.status}`);
      const data = new Float32Array(await response.arrayBuffer());
      if (data.length !== config.frames.length * n * 3 || !data.every(Number.isFinite)) throw Error('Invalid geometry capture');
      const canvas = root.querySelector(`[data-view="${name}"]`);
      methods[name] = {...config, data, canvas, ctx: canvas.getContext('2d'), projected: new Float32Array(n * 2)};
    }
    const originalLast = methods.hgbp.data.subarray(methods.hgbp.data.length - n * 3);
    const fitIndices = [];
    for (let i = 0; i < (display ? meta.points : n); ++i) {
      if (!display || [0, 1, 2].every(axis => originalLast[3 * i + axis] >= display.focus_min[axis] && originalLast[3 * i + axis] <= display.focus_max[axis])) fitIndices.push(i);
    }
    if (!fitIndices.length) throw Error('Empty display focal region');
    if (display) {
      // One proper rigid rotation, shared across methods and all outer states.
      const B = display.basis;
      for (const m of Object.values(methods)) for (let i = 0; i < m.data.length; i += 3) {
        const x = m.data[i], y = m.data[i + 1], z = m.data[i + 2];
        m.data[i] = B[0][0] * x + B[0][1] * y + B[0][2] * z;
        m.data[i + 1] = B[1][0] * x + B[1][1] * y + B[1][2] * z;
        m.data[i + 2] = B[2][0] * x + B[2][1] * y + B[2][2] * z;
      }
    }
    const first = methods.hgbp.data, last = first.subarray(first.length - n * 3);
    // One common display frame for both solvers, fixed throughout the replay.
    const center = [0, 0, 0];
    for (let axis = 0; axis < 3; ++axis) {
      const values = Array.from({length: n}, (_, i) => last[3 * i + axis]).sort((a, b) => a - b);
      const lo = values[Math.floor(n * .005)], hi = values[Math.floor(n * .995)];
      center[axis] = (hi + lo) / 2;
    }
    let yaw = display?.yaw ?? .4, pitch = display?.pitch ?? .22;
    let zoom = 1, position = 0, running = false, visible = false, lastTick = 0, lastPaint = 0, smooth = true;
    const duration = kind === 'ba' ? 18000 : 23000;
    const T = Math.max(meta.hgbp.total, meta.baseline.total), k = kind === 'ba' ? .7 : 5;
    const timeAt = u => u <= 0 ? 0 : u >= 1 ? T : k * Math.expm1(u * Math.log1p(T / k));
    const section = root.closest?.('section');
    if (section) {
      section.querySelector('[data-scene-speed]').textContent = `${fmt(meta.baseline.total / meta.hgbp.total, 2)}\u00d7`;
      section.querySelector('[data-dataset-summary]').textContent = kind === 'ba'
        ? `${fmt(meta.cameras, 0)} cameras · ${fmt(meta.full_points, 0)} points · ${fmt(meta.observations, 0)} observations`
        : `${fmt(meta.nodes, 0)} poses · ${fmt(meta.factors, 0)} factors${meta.initialization?.startsWith('Integrated') ? ' · odometry initialization' : ''}`;
      section.querySelector('[data-scene-caption]').textContent = kind === 'ba'
        ? `${meta.dataset}: all ${fmt(meta.cameras, 0)} camera centers and ${fmt(meta.points, 0)} sampled landmarks. Both cost curves use identical axes; each marker is a completed outer iteration.`
        : `${meta.dataset}: black odometry, red sampled loop closures.${meta.points < meta.nodes ? ` ${fmt(meta.points, 0)} of ${fmt(meta.nodes, 0)} poses shown.` : ''} Geometry is recorded, with display interpolation between keyframes. Timing provenance is provided below the replay.`;
      section.querySelector('[data-comparison-note]').innerHTML = kind === 'ba' ? 'vs. PowerBA<br>replay clocks · 16 threads' : 'vs. g2o-Cholesky<br>replay clocks · 20 outer';
    }
    const logs = [...meta.hgbp.costs, ...meta.baseline.costs].map(Math.log10);
    const lowCost = Math.min(...logs) - .06, highCost = Math.max(...logs) + .06;
    const short = value => value >= 1e6 ? `${Number((value / 1e6).toPrecision(2))}M` : value >= 1e3 ? `${Number((value / 1e3).toPrecision(2))}k` : String(Number(value.toPrecision(2)));
    function drawCurve(name, time, outer) {
      const m = methods[name], canvas = root.querySelector(`[data-curve="${name}"]`), ctx = canvas.getContext('2d');
      const W = canvas.clientWidth, H = canvas.clientHeight, dpr = Math.min(devicePixelRatio || 1, 2);
      if (!W || !H) return;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);}
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
      const left = 49, right = W - 12, top = 30, bottom = H - 34;
      const x = t => left + Math.log1p(t / k) / Math.log1p(T / k) * (right - left);
      const y = cost => bottom - (Math.log10(cost) - lowCost) / (highCost - lowCost) * (bottom - top);
      ctx.font = '12px "DM Sans", sans-serif'; ctx.lineWidth = .65; ctx.textAlign = 'right';
      for (const p of [0, .5, 1]) {
        const yy = top + p * (bottom - top), cost = 10 ** (highCost - p * (highCost - lowCost));
        ctx.strokeStyle = '#e1e5e4'; ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(right, yy); ctx.stroke();
        ctx.fillStyle = '#68777b'; ctx.fillText(short(cost), left - 6, yy + 4);
      }
      ctx.textAlign = 'center';
      for (const u of [0, .5, 1]) {
        ctx.fillText(fmt(timeAt(u), u ? 1 : 0), left + u * (right - left), bottom + 16);
      }
      ctx.fillStyle = '#40565d'; ctx.font = '14px "DM Sans", sans-serif'; ctx.textAlign = 'left';
      ctx.fillText('Cost (log)', 7, 18); ctx.textAlign = 'right'; ctx.fillText('Time [s]', right, H - 2);
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
        for (const i of fitIndices) {
          const x = frame[i * 3] - center[0], y = frame[i * 3 + 1] - center[1], z = frame[i * 3 + 2] - center[2];
          xs.push(cy * x + sy * z); ys.push(cp * y - sp * (-sy * x + cy * z));
        }
      }
      xs.sort((a, b) => a - b); ys.sort((a, b) => a - b);
      const quantile = display?.fit_quantile ?? .005;
      const bounds = (values, q) => [values[Math.floor(values.length * q)], values[Math.min(values.length - 1, Math.floor(values.length * (1 - q)))]];
      const [x0, x1] = bounds(xs, quantile);
      // Keep the complete tower height; robust horizontal fitting rejects distant outliers.
      const [y0, y1] = bounds(ys, display?.vertical_fit_quantile ?? quantile);
      fit = {key, x: (x0 + x1) / 2, y: (y0 + y1) / 2,
        scale: .89 * Math.min(W / (x1 - x0), H / (y1 - y0))};
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
      `Display: ${meta.provenance.sampling} ${display?.description ?? 'The central 99% coordinate range sets the viewport; extreme outliers may lie outside it.'} A single fixed viewing frame is shared by both methods. Drag either view to rotate both; use the mouse wheel to zoom.`,
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
        m.projected[i * 2] = W * (display?.horizontal_center ?? .5) + (xx - view.x) * scale;
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
      ctx.fillStyle = kind === 'pgo' ? '#172d3c88' : '#354d5d99';
      const size = W < 300 ? .8 : 1.15;
      ctx.beginPath();
      for (let i = 0; i < meta.points; ++i) ctx.rect(p[2 * i], p[2 * i + 1], size, size);
      ctx.fill();
      if (meta.cameras) {
        ctx.fillStyle = name === 'hgbp' ? '#205fc4c0' : '#b96c42c0'; ctx.beginPath();
        for (let i = meta.points; i < n; ++i) ctx.rect(p[2 * i] - .7, p[2 * i + 1] - .7, 1.4, 1.4);
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
    listen(play, 'click', () => {if (position >= .9999) position = 0; running = !running; lastTick = 0; updateButton(); render();});
    listen(scrub, 'input', () => {position = Number(scrub.value) / 1000; running = false; updateButton(); render();});
    listen(root.querySelector('[data-reset]'), 'click', () => {position = 0; running = false; updateButton(); render();});
    listen(root.querySelector('[data-motion]'), 'click', event => {smooth = !smooth; event.target.textContent = smooth ? 'Show recorded states only' : 'Enable smooth display motion'; render();});
    for (const m of Object.values(methods)) {
      let drag = null;
      listen(m.canvas, 'pointerdown', e => {drag = [e.clientX, e.clientY]; m.canvas.setPointerCapture(e.pointerId);});
      listen(m.canvas, 'pointermove', e => {
        if (!drag) return; yaw += (e.clientX - drag[0]) * .008; pitch = clamp(pitch + (e.clientY - drag[1]) * .008, -1.55, 1.55);
        drag = [e.clientX, e.clientY]; render();
      });
      listen(m.canvas, 'pointerup', () => {drag = null;});
      listen(m.canvas, 'pointercancel', () => {drag = null;});
      listen(m.canvas, 'wheel', e => {e.preventDefault(); zoom = clamp(zoom * Math.exp(-e.deltaY * .001), .25, 4); render();}, {passive: false});
    }
    const resize = new ResizeObserver(render);
    const visibility = new IntersectionObserver(entries => {visible = entries[0].isIntersecting; lastTick = 0;});
    observers.push(resize, visibility); resize.observe(root); visibility.observe(root);
    // Start at the original input; users choose when to run either comparison.
    render(); updateButton();
    function frame(now) {
      if (disposed) return;
      if (running && visible && !document.hidden) {
        if (lastTick) position = Math.min(1, position + Math.min(now - lastTick, 80) / duration);
        if (now - lastPaint >= 33 || position >= 1) {render(); lastPaint = now;}
        if (position >= 1) {running = false; updateButton();}
      }
      lastTick = now; frameRequest = requestAnimationFrame(frame);
    }
    frameRequest = requestAnimationFrame(frame);
    root.dataset.ready = 'true';
    return {meta, dispose};
  } catch (error) {
    status.hidden = false; status.textContent = `Replay unavailable: ${error.message}. The complete benchmark results remain available below.`;
    console.error(error);
    dispose();
    return {meta: null, dispose};
  }
}
