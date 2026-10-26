import assert from 'node:assert/strict';
import {coarsePoint, drawMechanism, MECHANISM_LOOP_MS} from '../mechanism.js';

assert.equal(MECHANISM_LOOP_MS * 1.5, 11000, 'The illustration must play at 1.5x its original speed');

const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
assert.deepEqual(corners.map(([u, v]) => coarsePoint(0, 0, u, v, 0)),
  [[-25, -25], [25, -25], [25, 25], [-25, 25]]);
const a = coarsePoint(0, 0, -1, -1, 1), b = coarsePoint(0, 0, 1, -1, 1);
const midpoint = coarsePoint(0, 0, 0, -1, 1);
assert(Math.abs(midpoint[1] - (a[1] + b[1]) / 2) > 5, 'Uncorrected edges must be curved');

let arcs = [];
const ctx = new Proxy({}, {get(target, key) {
  if (key === 'arc') return (...values) => arcs.push(values);
  return target[key] ?? ((...args) => {for (const value of args) if (typeof value === 'number') assert(Number.isFinite(value));});
}});
for (const phase of [0, .25, .5, .7, .85, .99]) {
  arcs = []; drawMechanism(ctx, 1120, 300, phase);
  const coarseNodes = arcs.filter(([, , radius]) => radius === 5.5);
  assert.equal(coarseNodes.length, 8);
  if (phase >= .7) {
    assert.deepEqual(coarseNodes.slice(4).map(([x, y]) => [x, y]),
      [[520, 38], [570, 38], [570, 88], [520, 88]], 'Corrected graph must remain a square, not collapse to a line');
  }
}
console.log('Mechanism: warped curves straighten into a nondegenerate square; animation phases verified.');
