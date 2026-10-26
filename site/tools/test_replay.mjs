// Logic/render smoke tests with a minimal DOM and recording canvas, no browser dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {initReplay} from '../replay.js';
const rootURL = new URL('../', import.meta.url);
let nextFrames = [], draws = 0;
globalThis.devicePixelRatio = 1;
globalThis.requestAnimationFrame = callback => nextFrames.push(callback);
globalThis.cancelAnimationFrame = () => {};
globalThis.ResizeObserver = class { constructor(callback) {this.callback = callback;} observe() {this.callback();} disconnect() {} };
globalThis.IntersectionObserver = class { constructor(callback) {this.callback = callback;} observe() {this.callback([{isIntersecting:true}]);} disconnect() {} };
globalThis.fetch = async url => {
  const data = await readFile(new URL(url, rootURL));
  return {ok:true, json:async () => JSON.parse(data), arrayBuffer:async () => data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength)};
};
class Element {
  constructor(width = 520, height = 440) {
    this.listeners = {}; this.dataset = {}; this.clientWidth = width; this.clientHeight = height;
    this.classList = {toggle() {}}; this.textContent = ''; this.value = 0;
    this.context = new Proxy({}, {get(target, key) {
      return target[key] ?? ((...args) => {
        draws++;
        for (const value of args) if (typeof value === 'number') assert(Number.isFinite(value), `${String(key)} received a nonfinite value`);
      });
    }});
  }
  getContext() {return this.context;}
  append() {}
  replaceChildren() {}
  setAttribute() {}
  addEventListener(type, callback, options = {}) {
    this.listeners[type] = callback;
    options.signal?.addEventListener('abort', () => {if (this.listeners[type] === callback) delete this.listeners[type];}, {once:true});
  }
  emit(type) {this.listeners[type]?.({target:this});}
}
const gallery = JSON.parse(await readFile(new URL('data/gallery.json', rootURL)));
for (const [kind, entries] of Object.entries(gallery)) for (const entry of entries) {
  nextFrames = [];
  const root = new Element(), elements = new Map(); root.dataset.scene = kind;
  const header = new Map();
  root.closest = () => ({querySelector(selector) {
    if (!header.has(selector)) header.set(selector, new Element());
    return header.get(selector);
  }});
  root.querySelector = selector => {
    if (!elements.has(selector)) elements.set(selector, selector.startsWith('[data-curve') ? new Element(228, 146) : new Element());
    return elements.get(selector);
  };
  globalThis.document = {hidden:false, createElement:() => new Element(), querySelector:() => new Element()};
  const controller = await initReplay(root, entry.metadata);
  assert.equal(root.dataset.ready, 'true');
  const meta = JSON.parse(await readFile(new URL(entry.metadata, rootURL)));
  const scale = (meta.baseline.total / meta.hgbp.total).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:2});
  assert.equal(header.get('[data-scene-speed]').textContent, `${scale}\u00d7`);
  assert(header.get('[data-scene-caption]').textContent.startsWith(`${entry.label}:`));
  const scrub = root.querySelector('[data-scrub]');
  for (const value of [0, 100, 500, 1000]) {scrub.value = value; scrub.emit('input');}
  for (const method of ['hgbp', 'baseline']) {
    assert.equal(root.querySelector(`[data-step="${method}"]`).textContent, '20 / 20 · complete');
    const value = Number(root.querySelector(`[data-cost="${method}"]`).textContent.replaceAll(',', ''));
    assert(Math.abs(value - meta[method].costs.at(-1)) <= .051);
  }
  root.querySelector('[data-reset]').emit('click');
  assert.equal(root.querySelector('[data-clock]').textContent, '0.00 s');
  root.querySelector('[data-play]').emit('click');
  for (let now = 1000; now < 2000; now += 50) {const current = nextFrames; nextFrames = []; current.forEach(callback => callback(now));}
  assert.notEqual(root.querySelector('[data-clock]').textContent, '0.00 s');
  root.querySelector('[data-motion]').emit('click');
  assert.equal(root.querySelector('[data-motion]').textContent, 'Enable smooth display motion');
  controller.dispose();
  assert.equal(Object.keys(root.querySelector('[data-play]').listeners).length, 0, 'Switching must detach the old controls');
  console.log(`${entry.label}: play, restart, scrub, both final costs, display toggle, disposal and finite drawing verified`);
}
assert(draws > 1000);
