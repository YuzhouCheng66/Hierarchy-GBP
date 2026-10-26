import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {initGallery} from '../gallery.js';

const entries = JSON.parse(await readFile(new URL('../data/gallery.json', import.meta.url)));
globalThis.fetch = async () => ({ok: true, json: async () => entries});
globalThis.document = {createElement: () => ({})};
for (const kind of ['ba', 'pgo']) {
  const listeners = new Map(), calls = [], disposed = [];
  const select = {
    options: [], disabled: true,
    replaceChildren() {this.options = [];},
    append(option) {this.options.push(option);},
    addEventListener(type, callback) {listeners.set(type, callback);},
    removeEventListener(type) {listeners.delete(type);}
  };
  const requested = entries[kind].at(-1);
  globalThis.location = {href: `https://localhost/?${kind}=${requested.id}`};
  const root = {dataset: {scene: kind}, closest: () => ({querySelector: () => select})};
  const gallery = await initGallery(root, async (_, metadata) => {
    assert(select.disabled, 'Disable selection while downloading a capture');
    calls.push(metadata);
    return {dispose: () => disposed.push(metadata)};
  });
  assert.equal(calls.at(-1), requested.metadata, 'A shared scene URL must open its requested dataset');
  assert.equal(select.options.length, entries[kind].length);
  for (const entry of entries[kind]) {
    select.value = entry.id;
    await listeners.get('change')();
    assert.equal(calls.at(-1), entry.metadata);
    assert.equal(select.disabled, false);
  }
  gallery.dispose();
  assert.equal(disposed.length, calls.length, 'Every scene must be disposed exactly once');
  assert.equal(listeners.size, 0);
  console.log(`${kind}: all dataset selections, shared scene URLs, loading state and cleanup verified`);
}
