import {initMechanism} from './mechanism.js?v=20261026-4';
import {initReplay} from './replay.js?v=20261026-2';

initMechanism();
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {observer.unobserve(entry.target); initReplay(entry.target);}
}, {rootMargin: '600px'});
document.querySelectorAll('[data-scene]').forEach(root => observer.observe(root));

async function tables() {
  const response = await fetch('data/benchmarks.json');
  if (!response.ok) throw Error('Benchmark data unavailable');
  const data = await response.json(), target = document.querySelector('#benchmark-table');
  function show(family) {
    const table = document.createElement('table'), head = table.createTHead().insertRow();
    const dataset = document.createElement('th'); dataset.textContent = 'Dataset'; dataset.scope = 'col'; head.append(dataset);
    for (const name of data[family].methods) {
      const th = document.createElement('th'), split = name.lastIndexOf('-'); th.scope = 'col';
      th.append(document.createTextNode(name.slice(0, split)), document.createElement('br'), document.createTextNode(`${name.slice(split + 1)} thread${name.slice(split + 1) === '1' ? '' : 's'}`));
      if (name.startsWith('H-GBP')) th.className = 'ours'; head.append(th);
    }
    const body = table.createTBody();
    for (const row of data[family].rows) {
      const tr = body.insertRow(), label = tr.insertCell(); label.textContent = row.dataset;
      if (row.space) {const span = document.createElement('span'); span.className = 'space-tag'; span.textContent = row.space; label.append(span);}
      const fastest = Math.min(...row.cells.map(cell => cell.time ?? Infinity));
      row.cells.forEach((cell, i) => {
        const td = tr.insertCell(), cost = document.createElement('span');
        td.append(document.createTextNode(cell.time_text)); cost.className = 'cell-cost'; cost.textContent = cell.cost_text; td.append(cost);
        td.classList.toggle('ours', data[family].methods[i].startsWith('H-GBP')); td.classList.toggle('best', cell.time === fastest);
      });
    }
    target.replaceChildren(table); target.setAttribute('aria-labelledby', `${family}-tab`);
    document.querySelectorAll('[data-table]').forEach(button => button.setAttribute('aria-selected', String(button.dataset.table === family)));
  }
  document.querySelectorAll('[data-table]').forEach(button => button.addEventListener('click', () => show(button.dataset.table)));
  show('ba');
}
tables().catch(error => {document.querySelector('#benchmark-table').textContent = error.message; console.error(error);});

document.querySelector('#copy-citation').addEventListener('click', async event => {
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    event.target.textContent = 'Copied'; setTimeout(() => {event.target.textContent = 'Copy BibTeX';}, 2000);
  } catch {
    const selection = getSelection(), range = document.createRange(); range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges(); selection.addRange(range); event.target.textContent = 'Select and copy';
  }
});
