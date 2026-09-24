# Unified policies

`configs/pgo.json` defines one SE2 policy and one SE3 policy. Both use loopy
GBP and a reduced-basis coarse correction, adaptive precision checks, and
20 outer iterations. The shipped driver has no per-dataset numerical
overrides, fixed dataset-dependent freeze thresholds, or fine direct polish.

`configs/ba.json` defines one policy for every BA input. The GBP message graph
uses the top four normalized-covisibility neighbors per camera, their symmetric
union, and a maximum-weight spanning backbone within each connected component.
Pair blocks use at most 16 deterministically sampled landmark contributions,
with rescaling. These approximations affect the GBP preconditioner, not the
data objective: all observations remain in the exact landmark-implicit Schur
operator and reprojection evaluation. Connected aggregates supply the additive
coarse correction; flexible CG controls the linear solve.

BA full-precision updates stop at relative precision change `1e-6` or the
uniform cap of 32 sweeps, after which cached eta maps are used. Hitting the cap
is not proof of variance convergence; the output records `variance_defect`.
This retains the tested algorithm rather than silently changing its stopping
rule during release cleanup.

Reference timings are historical diagnostics, not cross-machine acceptance
criteria. Numerical references are separate for 1 and 16 threads. Alamo is a
seen development supplement, not evidence of a fresh held-out OOD evaluation;
its 1-thread reference predates the September 24 dataflow optimization.
