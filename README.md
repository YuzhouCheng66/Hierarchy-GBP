# Hierarchy-GBP

[**Paper (arXiv)**](https://arxiv.org/abs/2610.06978)

![Hierarchy-GBP: abstraction, coarse correction, recovery, and GBP refinement.](code/docs/assets/teaser.png)

High-performance CPU solvers for SE2/SE3 pose-graph optimization and bundle
adjustment, built around **hierarchical Gaussian belief propagation (H-GBP)**.

H-GBP combines **loopy Gaussian message passing** with **coarse-scale correction**
to couple local refinement and coordinated global updates. PGO uses a
reduced-basis hierarchy; BA applies GBP and connected coarse corrections to
the landmark-reduced camera system within a flexible conjugate-gradient solve.

Optimized fixed-size kernels, packed message storage and cached updates support
efficient single-threaded and parallel execution. See
[algorithm and policy details](code/docs/policies.md).

## Layout

```text
code/
  src/        Three solver entry points; private helpers in detail/
  apps/       PGO command-line frontends
  include/    C++ interfaces and kernels
  tests/      Numerical and runner regression tests
  variants/   Optional all-GBP coarse and SE2 CUDA implementations
  docs/       Policies, dependencies and project assets
benchmarks/
  configs/    Shared solver policies and pinned inputs
  baselines/  PGO and BA reproduction adapters
  datasets/   Synthetic generators and release tools
```

Build with `code/build.ps1`; download and run benchmarks with
`benchmarks/download.py` and `benchmarks/run.py`.
Start reading from [the solver entry points](code/docs/architecture.md).

Policies are shared across datasets within SE2, SE3 and BA respectively.
Dataset names select input metadata, not numerical parameter overrides.
Older research code remains in Git history and the `legacy` branch.

## Build

The validated environment is Windows x64, Visual Studio 2022, CMake/Ninja,
Python 3.10+, Eigen, SuiteSparse and OpenBLAS. BA also needs the compiled
RootBA dependency. See [dependencies](code/docs/dependencies.md) for the required
layout and runtime separation; dependencies and input data are not vendored.

```powershell
.\code\build.ps1 -DependencyRoot C:\deps -PGORuntimeDir C:\deps\pgo-runtime
ctest --test-dir build --output-on-failure
```

Use `-PGOOnly` to build without RootBA. Binaries are placed in `build/pgo`
and `build/ba`, preventing their different BLAS runtimes from being mixed.

## Benchmark inputs

The complete benchmark release is hosted on
[Hugging Face: HGBP-Benchmarks](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks):
**9 pose graphs** (3 SE2 and 6 SE3) and **10 bundle-adjustment problems**.
Download the exact inputs used by H-GBP and the baseline adapters:

```powershell
python benchmarks/download.py
```

The downloader verifies SHA-256 checksums and restores `data/pgo` and `data/ba`.
The release revision and manifest hash are pinned in
[`benchmarks/configs/data_release.json`](benchmarks/configs/data_release.json); the dataset card
records input provenance and preparation.
Use `--suite pgo`, `--suite ba` or `--datasets Sphere Ladybug49` for a subset.
For existing data, set `HGBP_PGO_DATA_ROOT` and `HGBP_BA_DATA_ROOT` instead.

## Run

```powershell
python benchmarks/run.py pgo --threads 16 --output-root results/reproduction/pgo
python benchmarks/run.py ba --threads 16 --output-root results/reproduction/ba
```

Use `--threads 1` for the serial path, `--compare-threads` for sequential
1/16-thread runs, or append dataset names to run a subset. Run `--help` for
custom inputs and runtime locations. No dataset-specific tuning is required.

Outputs record effective commands, input/binary hashes, native solver wall
time and process wall time separately. PGO reports its common quadratic
cost; BA reports squared reprojection cost, RMSE and mean reprojection error.
Numerical checks and timing diagnostics are separate: historical timings
are not portable performance guarantees.

See [baseline reproduction](benchmarks/baselines/README.md) for comparison methods.
Input definitions and sources are documented in the dataset card; the
[Globe generators](benchmarks/datasets/README.md) reproduce the synthetic inputs.

## GBP at every level

The optional [`pure_gbp`](code/variants/pure_gbp/README.md) variant retains
hierarchical abstraction and recovery while using GBP for fine smoothing and
both coarse levels, **without a coarse direct or Krylov solve**. Same-machine
16-thread, outer20 tests on all nine PGO inputs take **1.45-2.51x** the default
runtime, with a maximum final-cost increase of
**0.451%**. The faster coarse-direct route remains the default.
The independent folder contains the shared-policy configuration, build/run
instructions, numerical tests and [paired evidence](code/variants/pure_gbp/evidence/summary.json).
