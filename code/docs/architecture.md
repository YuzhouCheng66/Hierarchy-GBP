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

## Reproduction tools

These entry points are separate from solver code and need no Python package
installation. Run them from the repository root, or use their absolute paths.

| Task | Entry | Responsibility |
| --- | --- | --- |
| Configure/build | [build.ps1](../build.ps1) | Explicit dependency paths; `-CheckOnly` for preflight |
| Download/verify inputs | [download.py](../../benchmarks/download.py) | Pinned release, hashes, no input overwrite |
| Run H-GBP | [run.py](../../benchmarks/run.py) | Shared policies, sequential runs, metrics and quality checks |
| Compare baselines | [baselines](../../benchmarks/baselines/README.md) | Separate external implementations and runtimes |
| Numerical/runner checks | [tests](../tests/) | Small regression tests, not benchmark configurations |
| Generate/publish data | [datasets](../../benchmarks/datasets/README.md) | Maintainer tools; not needed to run downloaded inputs |

`benchmarks/common.py` supplies only small stdlib utilities (hashes, JSON,
argument validation and affinity). It does not select solver policies or import
the main runner. Independent PGO scoring remains in the baseline package and
requires its own NumPy/SciPy dependencies. Dataset generation requires NumPy;
downloading, packaging and runner planning do not.

Use `python -m unittest discover -s code/tests -p "test_*.py"` for tooling
checks and `ctest --test-dir build --output-on-failure` after building for
compiled numerical tests. Test coverage does not certify an unbuilt external
baseline, BA dependency stack or optional CUDA backend.

Historical validation JSON, archived runs, licenses and source/runtime locks
are retained as evidence, not executable configuration. Paths embedded in
those records describe the original measurement environment, not required
locations on a new machine.
