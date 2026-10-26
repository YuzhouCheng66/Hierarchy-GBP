import {initReplay} from './replay.js?v=20261026-8';

export async function initGallery(root, loadReplay = initReplay) {
  const kind = root.dataset.scene, section = root.closest('section');
  const select = section.querySelector('[data-scene-select]');
  const response = await fetch('data/gallery.json?v=20261026-7');
  if (!response.ok) throw Error('Scene catalog unavailable');
  const entries = (await response.json())[kind];
  select.replaceChildren();
  for (const entry of entries) {
    const option = document.createElement('option');
    option.value = entry.id; option.textContent = entry.label; select.append(option);
  }
  const requested = new URL(globalThis.location?.href ?? 'https://localhost/').searchParams.get(kind);
  select.value = entries.some(entry => entry.id === requested) ? requested : entries[0].id;
  let active = null;
  async function change() {
    select.disabled = true;
    active?.dispose();
    try {
      const entry = entries.find(entry => entry.id === select.value);
      active = await loadReplay(root, entry.metadata);
    } finally {
      select.disabled = false;
    }
  }
  select.addEventListener('change', change);
  await change();
  return {dispose() {active?.dispose(); select.removeEventListener('change', change);}};
}
