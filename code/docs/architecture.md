# Reading the solvers

`src/` has three entry files. Each keeps the nonlinear iteration and its
fine/coarse correction flow visible; command-line PGO frontends are in `apps/`.

| Problem | Entry | Main routine |
| --- | --- | --- |
| SE2 PGO | [se2_solver.cpp](../src/se2_solver.cpp) | `runSyntheticSE2Experiment` |
| SE3 PGO | [se3_solver.cpp](../src/se3_solver.cpp) | `runSyntheticSE3Experiment` |
| BA | [ba_solver.cpp](../src/ba_solver.cpp) | `runRootHGBP` |

## Private implementation

`src/detail/` contains model/linearization, GBP kernels, basis/coarse assembly
and input/output support. Files are prefixed by problem family. The existing
generic factor/variable graph remains shared by the two PGO solvers.

The `.inc` files are private implementation fragments, not separate backends.
They are included once by their owning source/header. Keeping the original
translation units preserves the kernel inlining boundaries, floating-point
compiler flags and definition order. Editing a fragment rebuilds its owner.

The long entry files were split without changing the expanded implementation.
The initial pruning removed only unreferenced private helpers and impossible
control-flow branches. Supported library APIs, sparse/dense fallback solves,
PSD/jitter safeguards and diagnostic outputs were retained. A path not visited
by the benchmark suite is not, by itself, considered dead code.

## Optional coarse GBP

[pure_gbp](../variants/pure_gbp/README.md) shares the fine-level kernels and
replaces the coarse solver at build time. It remains separate from the default
coarse-direct configuration. Policies are in `benchmarks/configs/` at the
repository root; dataset names do not select alternative solver code.
