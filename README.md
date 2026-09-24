# Hierarchy-GBP

CPU implementations of hierarchical Gaussian belief propagation for SE2/SE3
pose-graph optimization and bundle adjustment.

PGO combines loopy GBP smoothing with a reduced-basis coarse correction.
BA uses landmark elimination, GBP camera messages and a connected coarse
correction inside a flexible conjugate-gradient linear controller.
PGO checks precision convergence adaptively. BA switches to cached eta-only
updates at its precision tolerance or 32-sweep cap. Neither uses fine-level
direct polish. See [policy details](docs/policies.md).

## Layout

- `src/se2_solver.cpp`, `src/se3_solver.cpp`, `src/ba_solver.cpp`: solvers.
- `include/hgbp/`: PGO public interfaces; `include/internal/`: implementation support.
- `configs/`: one policy per problem family, input hashes and validation references.
- `scripts/run_benchmarks.py`: reproducible runs, including previously unseen inputs.
- `tests/`: numerical kernels and runner checks.
- `scripts/datasets/`: reproducible synthetic Globe generators.

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

Set `HGBP_PGO_DATA_ROOT` and `HGBP_BA_DATA_ROOT` to your canonical input
directories. Input SHA-256 checks guard against accidental changes to
initialization or preprocessing.

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

Cubicle's reference graph includes an offline PSD floor. The synthetic
Globe graphs are project-generated, not the original paper's dataset files;
see the [generation instructions](scripts/datasets/README.md).
