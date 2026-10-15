"""Check segment bookkeeping against the original per-sweep precision policy."""
import copy
import math
import random
import unittest


def state():
    return dict(sweep=0, frozen=False, stable=0, checks=0, freezes=0, thaws=0,
                first=-1, last=0, interval=1, next=1, maps=False, full=0, eta=0,
                residual=0.)


def check(s, residual):
    s['residual'] = residual
    s['last'] = s['sweep']
    s['checks'] += 1
    if math.isfinite(residual) and residual <= 1e-8:
        s['stable'] += 1
        if not s['frozen'] and s['stable'] >= 3:
            s['frozen'] = True
            s['freezes'] += 1
            if s['first'] < 0:
                s['first'] = s['sweep']
    else:
        if s['frozen']:
            s['thaws'] += 1
        s['frozen'] = False
        s['stable'] = 0
    s['interval'] = 25 if s['frozen'] else 5 if s['stable'] > 0 else min(2*s['interval'], 25)
    s['next'] = s['sweep'] + s['interval']


def original(s, count, residuals):
    trace = []
    for _ in range(count):
        test = s['sweep'] + 1 >= s['next']
        eta = s['frozen'] and not test
        trace.append((s['sweep'] + 1, eta, test, not eta or not s['maps']))
        s['sweep'] += 1
        s['eta' if eta else 'full'] += 1
        s['maps'] = eta
        if test:
            check(s, residuals[s['sweep']])
    return trace


def segmented(s, count, residuals):
    trace = []
    remaining = count
    while remaining:
        ordinary = min(remaining, max(0, s['next']-s['sweep']-1))
        eta, ready = s['frozen'], s['maps']
        for step in range(ordinary):
            trace.append((s['sweep'] + step + 1, eta, False,
                          not eta or (step == 0 and not ready)))
        if ordinary:
            s['sweep'] += ordinary
            s['eta' if eta else 'full'] += ordinary
            s['maps'] = eta
        remaining -= ordinary
        if remaining:
            trace.append((s['sweep'] + 1, False, True, True))
            s['sweep'] += 1
            s['full'] += 1
            s['maps'] = False
            check(s, residuals[s['sweep']])
            remaining -= 1
    return trace


class SegmentScheduleTests(unittest.TestCase):
    def test_random_batched_freeze_thaw_policies(self):
        rng = random.Random(70623)
        for trial in range(500):
            a = state()
            b = copy.deepcopy(a)
            residuals = [rng.choice((0., 1e-9, 1e-8, 2e-8, math.inf)) for _ in range(1601)]
            for _ in range(12):
                count = rng.choice((1, 2, 5, 24, 25, 26, 99, 100, 101))
                self.assertEqual(original(a, count, residuals), segmented(b, count, residuals), trial)
                self.assertEqual(a, b, trial)

    def test_all_full_and_mostly_frozen(self):
        for residual in (0., 1e-3):
            a, b = state(), state()
            residuals = [residual]*6001
            for _ in range(60):
                self.assertEqual(original(a, 100, residuals), segmented(b, 100, residuals))
                self.assertEqual(a, b)
            self.assertEqual(a['full'] + a['eta'], 6000)

    def test_zero_batch_is_no_op(self):
        a = state()
        before = copy.deepcopy(a)
        self.assertEqual(segmented(a, 0, []), [])
        self.assertEqual(a, before)


if __name__ == '__main__':
    unittest.main()
