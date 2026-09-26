# Hierarchy-GBP

High-performance CPU solvers for SE2/SE3 pose-graph optimization and bundle
adjustment, built around **hierarchical Gaussian belief propagation (H-GBP)**.

H-GBP combines **loopy Gaussian message passing** with **coarse-scale correction**
to couple local refinement and coordinated global updates. PGO uses a
reduced-basis hierarchy; BA applies GBP and connected coarse corrections to
the landmark-reduced camera system within a flexible conjugate-gradient solve.

Optimized fixed-size kernels, packed message storage and cached updates support
efficient single-threaded and parallel execution. See
[algorithm and policy details](docs/policies.md).

## Layout

- `src/se2_solver.cpp`, `src/se3_solver.cpp`, `src/ba_solver.cpp`: solvers.
- `include/hgbp/`: PGO public interfaces; `include/internal/`: implementation support.
- `configs/`: one policy per problem family, input hashes and validation references.
- `scripts/run_benchmarks.py`: reproducible runs, including previously unseen inputs.
- `tests/`: numerical kernels and runner checks.
- `scripts/datasets/`: reproducible synthetic Globe generators.
- `experiments/se2_cuda_hybrid_hgbp_5x/`: archived first-generation CUDA/CPU
  hybrid H-GBP prototype and its carefully scoped historical 5x result.

Policies are shared across datasets within SE2, SE3 and BA respectively.
Dataset names select input metadata, not numerical parameter overrides.
Older research code remains in Git history and the `legacy` branch.

## Build

The validated environment is Windows x64, Visual Studio 2022, CMake/Ninja,
Python 3.10+, Eigen, SuiteSparse and OpenBLAS. BA also needs the compiled
RootBA dependency. See [dependencies](docs/dependencies.md) for the required
layout and runtime separation; dependencies and input data are not vendored.

```powershell
.\scripts\build.ps1 -DependencyRoot C:\deps -PGORuntimeDir C:\deps\pgo-runtime
ctest --test-dir build --output-on-failure
```

Use `-PGOOnly` to build without RootBA. Binaries are placed in `build/pgo`
and `build/ba`, preventing their different BLAS runtimes from being mixed.

## Run

Download the benchmark inputs from
[Hugging Face](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks):

```powershell
python scripts/download_datasets.py
```

The downloader verifies SHA-256 checksums and restores `data/pgo` and `data/ba`.
Use `--suite pgo`, `--suite ba` or `--datasets Sphere Ladybug49` for a subset.
For existing data, set `HGBP_PGO_DATA_ROOT` and `HGBP_BA_DATA_ROOT` instead.

```powershell
python scripts/run_benchmarks.py pgo --threads 16 --output-root results/reproduction/pgo
python scripts/run_benchmarks.py ba --threads 16 --output-root results/reproduction/ba
```

Use `--threads 1` for the serial path, `--compare-threads` for sequential
1/16-thread runs, or append dataset names to run a subset. Run `--help` for
custom inputs and runtime locations. No dataset-specific tuning is required.

Outputs record effective commands, input/binary hashes, native solver wall
time and process wall time separately. PGO reports its common quadratic
cost; BA reports squared reprojection cost, RMSE and mean reprojection error.
Numerical checks and timing diagnostics are separate: historical timings
are not portable performance guarantees.

Baseline adapters and reproduction commands are in [baselines/](baselines/README.md).
Input definitions and sources are documented in the dataset card; the
[Globe generators](scripts/datasets/README.md) reproduce the synthetic inputs.
