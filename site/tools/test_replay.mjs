// Logic/render smoke tests with a minimal DOM and recording canvas, no browser dependency.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {initReplay} from '../replay.js';
const rootURL = new URL('../', import.meta.url);
let nextFrames = [], draws = 0;
globalThis.devicePixelRatio = 1;
globalThis.requestAnimationFrame = callback => nextFrames.push(callback);
globalThis.ResizeObserver = class { constructor(callback) {this.callback = callback;} observe() {this.callback();} };
globalThis.IntersectionObserver = class { constructor(callback) {this.callback = callback;} observe() {this.callback([{isIntersecting:true}]);} };
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
  setAttribute() {}
  addEventListener(type, callback) {this.listeners[type] = callback;}
  emit(type) {this.listeners[type]?.({target:this});}
}
for (const kind of ['ba', 'pgo']) {
  nextFrames = [];
  const root = new Element(), elements = new Map(); root.dataset.scene = kind;
  root.querySelector = selector => {
    if (!elements.has(selector)) elements.set(selector, selector.startsWith('[data-curve') ? new Element(228, 146) : new Element());
    return elements.get(selector);
  };
  globalThis.document = {hidden:false, createElement:() => new Element(), querySelector:() => new Element()};
  await initReplay(root);
  assert.equal(root.dataset.ready, 'true');
  const meta = JSON.parse(await readFile(new URL(`data/${kind}.json`, rootURL)));
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
  console.log(`${kind}: play, restart, scrub, both final costs, display toggle and finite drawing verified`);
}
assert(draws > 1000);
