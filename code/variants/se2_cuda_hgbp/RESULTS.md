# Timing breakdown and limits

All times below are ms. They refer to the retained checkpoint, not an
incoming-layout trial and not the old unequal-budget 5x prototype.

## Unprofiled phase medians, 20 outers

| Phase | FR079 | FRH | M3500 |
|---|---:|---:|---:|
| Complete solver | 66.02 | 71.41 | 143.54 |
| Lower host interval, including overlapped basis | 50.62 | 49.43 | 84.58 |
| CPU basis, overlapping the previous row | 10.28 | 13.11 | 34.65 |
| Coarse matrix assembly | 1.95 | 2.38 | 8.93 |
| Coarse RHS/restrict | 0.72 | 1.04 | 1.84 |
| Coarse factorization/solve | 1.55 | 1.55 | 13.69 |
| Prolong/inject | 0.79 | 0.99 | 2.24 |
| Factor relinearization | 3.95 | 7.03 | 12.97 |
| Objective | 0.80 | 1.43 | 3.22 |

These are overlapping timers and separate medians, NOT additive exclusive
wall-time shares. The lower host interval includes submission, packing,
synchronization and basis overlap. Unlisted setup/control work also remains.
In JSON, `phase_medians_ms` has millisecond values despite the historical
field names ending in `_sec`.

## Separate Nsight Systems diagnostic

| Device activity | FR079 | FRH | M3500 |
|---|---:|---:|---:|
| Lower kernel intervals | 35.56 | 31.18 | 50.23 |
| All kernel intervals | 36.04 | 31.65 | 50.79 |
| Copy-engine intervals | 1.00 | 1.41 | 3.14 |

Do not subtract these profiled values from the unprofiled medians to manufacture
an exact host overhead. CUDA API waits overlap device execution. Trace-wide
cudaFree calls also include initialization/teardown; they are not all inside
the solver timer. Nsight Compute's latest attempt failed with
`ERR_NVGPUCTRPERM`; no current-kernel hardware-counter claim is made.

## Complete-process target

The process totals include context initialization, program/dependency loading,
input parsing, complete solving, all requested per-outer JSON output, resource
release and exit. Context is already inside these totals. The current retained
implementation does NOT achieve 2x versus CPU16 on this boundary.

Future non-lower optimization must measure these boundaries separately, preserve
the exact hierarchy and policy, and retain full trajectory output on both sides.
Common parser/serialization optimizations must also be offered to CPU16, with
the original released CPU comparison reported separately. Removing output only
on GPU, hiding initialization in a persistent service, or substituting a faster
but different basis is not acceptable evidence for the process-wall target.
