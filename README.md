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
  src/        SE2, SE3 and BA solvers
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

Results from [Tables 1 and 2 of the paper](https://arxiv.org/abs/2610.06978v1).
Each cell shows **runtime [s]** above final cost; parentheses give thread counts.
`OOM`: out of memory; `>2000`: time limit; `--`: unavailable.

**Pose-graph optimization**

| Method (threads) | FR079<br><sub>SE2</sub> | FRH<br><sub>SE2</sub> | M3500<br><sub>SE2</sub> |
| --- | ---: | ---: | ---: |
| g2o-Cholesky (1) | 0.02<br><sub>18.79</sub> | 0.04<br><sub>9.66e-5</sub> | 0.13<br><sub>68.96</sub> |
| **H-GBP (1)** | 0.29<br><sub>18.79</sub> | 0.35<br><sub>9.66e-5</sub> | 0.90<br><sub>68.96</sub> |
| g2o-PCG (16) | 1.09<br><sub>18.79</sub> | 0.41<br><sub>9.66e-5</sub> | 1.87<br><sub>68.96</sub> |
| Cycle-PGO (16) | 0.04<br><sub>18.79</sub> | 0.09<br><sub>9.66e-5</sub> | 0.23<br><sub>68.96</sub> |
| AMM-PGO (16) | 0.17<br><sub>129.13</sub> | 0.38<br><sub>1.02e-4</sub> | 0.91<br><sub>70.45</sub> |
| Schwarz-PGO (16) | 0.14<br><sub>18.79</sub> | 0.13<br><sub>9.66e-5</sub> | 1.47<br><sub>68.96</sub> |
| **H-GBP (16)** | 0.11<br><sub>18.79</sub> | 0.12<br><sub>9.66e-5</sub> | 0.27<br><sub>68.96</sub> |

| Method (threads) | ParkingGarage<br><sub>SE3</sub> | Sphere<br><sub>SE3</sub> | Cubicle<br><sub>SE3</sub> |
| --- | ---: | ---: | ---: |
| g2o-Cholesky (1) | 0.38<br><sub>0.63</sub> | 2.02<br><sub>675.70</sub> | 4.22<br><sub>1,378.53</sub> |
| **H-GBP (1)** | 7.56<br><sub>0.63</sub> | 4.76<br><sub>675.70</sub> | 22.63<br><sub>1,378.53</sub> |
| g2o-PCG (16) | 19.72<br><sub>0.63</sub> | 1.35<br><sub>675.70</sub> | 93.53<br><sub>1,378.53</sub> |
| Cycle-PGO (16) | 0.85<br><sub>0.63</sub> | 1.08<br><sub>675.70</sub> | 4.17<br><sub>1,609.14</sub> |
| AMM-PGO (16) | 1.37<br><sub>0.65</sub> | 1.08<br><sub>695.09</sub> | 7.67<br><sub>233,482.27</sub> |
| Schwarz-PGO (16) | 49.78<br><sub>0.63</sub> | 2.72<br><sub>675.70</sub> | 6.21<br><sub>1,378.53</sub> |
| **H-GBP (16)** | 1.77<br><sub>0.63</sub> | 1.18<br><sub>675.70</sub> | 4.86<br><sub>1,378.53</sub> |

| Method (threads) | Grid<br><sub>SE3</sub> | Globe10k<br><sub>SE3</sub> | Globe100k<br><sub>SE3</sub> |
| --- | ---: | ---: | ---: |
| g2o-Cholesky (1) | 493.90<br><sub>42,179.98</sub> | 93.45<br><sub>3,039.93</sub> | 1,288.89<br><sub>20,090.74</sub> |
| **H-GBP (1)** | 16.16<br><sub>42,179.98</sub> | 20.44<br><sub>3,039.93</sub> | 177.88<br><sub>20,090.74</sub> |
| g2o-PCG (16) | 4.38<br><sub>42,179.98</sub> | 7.04<br><sub>3,039.93</sub> | 662.81<br><sub>20,090.74</sub> |
| Cycle-PGO (16) | 621.95<br><sub>42,179.98</sub> | 42.04<br><sub>3,039.93</sub> | OOM<br><sub>--</sub> |
| AMM-PGO (16) | 7.94<br><sub>42,192.23</sub> | 12.78<br><sub>3,040.00</sub> | 132.07<br><sub>20,112.14</sub> |
| Schwarz-PGO (16) | 44.55<br><sub>42,179.98</sub> | 13.33<br><sub>3,039.93</sub> | >2000<br><sub>--</sub> |
| **H-GBP (16)** | 3.90<br><sub>42,179.98</sub> | 5.34<br><sub>3,039.93</sub> | 42.71<br><sub>20,090.74</sub> |

**Bundle adjustment** (cost: sum of squared reprojection residuals)

| Method (threads) | Ladybug49 | Venice89 | Final93 |
| --- | ---: | ---: | ---: |
| Ceres Sparse (1) | 0.98<br><sub>26,690</sub> | 19.22<br><sub>567,243</sub> | 7.39<br><sub>291,090</sub> |
| **H-GBP (1)** | 0.18<br><sub>26,705</sub> | 2.68<br><sub>576,781</sub> | 1.59<br><sub>291,092</sub> |
| Ceres Sparse (16) | 0.33<br><sub>26,690</sub> | 5.91<br><sub>567,243</sub> | 1.69<br><sub>291,090</sub> |
| Ceres Iterative (16) | 0.54<br><sub>26,691</sub> | 5.03<br><sub>567,055</sub> | 4.29<br><sub>291,090</sub> |
| RootBA-QR (16) | 0.63<br><sub>26,691</sub> | 6.22<br><sub>572,780</sub> | 6.84<br><sub>291,090</sub> |
| PowerBA (16) | 0.23<br><sub>27,299</sub> | 1.91<br><sub>580,416</sub> | 1.79<br><sub>291,159</sub> |
| **H-GBP (16)** | 0.07<br><sub>26,705</sub> | 0.59<br><sub>576,781</sub> | 0.38<br><sub>291,092</sub> |

| Method (threads) | Trafalgar257 | Dubrovnik356 | Final394 |
| --- | ---: | ---: | ---: |
| Ceres Sparse (1) | 7.21<br><sub>193,577</sub> | 50.15<br><sub>987,859</sub> | 38.57<br><sub>596,506</sub> |
| **H-GBP (1)** | 1.31<br><sub>197,378</sub> | 6.83<br><sub>1,021,102</sub> | 2.87<br><sub>598,296</sub> |
| Ceres Sparse (16) | 2.38<br><sub>193,577</sub> | 13.96<br><sub>987,859</sub> | 8.62<br><sub>596,506</sub> |
| Ceres Iterative (16) | 14.41<br><sub>193,671</sub> | 49.54<br><sub>987,859</sub> | 8.01<br><sub>596,536</sub> |
| RootBA-QR (16) | 22.72<br><sub>193,398</sub> | 119.17<br><sub>987,866</sub> | 29.59<br><sub>596,578</sub> |
| PowerBA (16) | 1.11<br><sub>205,427</sub> | 4.14<br><sub>1,059,446</sub> | 2.50<br><sub>611,101</sub> |
| **H-GBP (16)** | 0.45<br><sub>197,378</sub> | 1.63<br><sub>1,021,102</sub> | 0.83<br><sub>598,296</sub> |

| Method (threads) | Ladybug1723 | Venice1778 |
| --- | ---: | ---: |
| Ceres Sparse (1) | 60.56<br><sub>778,095</sub> | 330.93<br><sub>3,277,387</sub> |
| **H-GBP (1)** | 4.79<br><sub>759,600</sub> | 25.63<br><sub>3,363,149</sub> |
| Ceres Sparse (16) | 19.05<br><sub>778,095</sub> | 87.54<br><sub>3,277,387</sub> |
| Ceres Iterative (16) | 5.51<br><sub>770,444</sub> | 35.27<br><sub>6,932,878</sub> |
| RootBA-QR (16) | 9.18<br><sub>763,070</sub> | 85.45<br><sub>3,369,978</sub> |
| PowerBA (16) | 2.72<br><sub>797,704</sub> | 20.32<br><sub>3,391,089</sub> |
| **H-GBP (16)** | 1.22<br><sub>759,599</sub> | 5.89<br><sub>3,363,149</sub> |

| Method (threads) | Final3068 | Final4585 |
| --- | ---: | ---: |
| Ceres Sparse (1) | 586.17<br><sub>3,348,655</sub> | >2000<br><sub>--</sub> |
| **H-GBP (1)** | 12.40<br><sub>3,240,217</sub> | 48.55<br><sub>10,331,432</sub> |
| Ceres Sparse (16) | 142.78<br><sub>3,348,655</sub> | 526.06<br><sub>10,783,988</sub> |
| Ceres Iterative (16) | 13.17<br><sub>3,452,711</sub> | 72.54<br><sub>13,021,324</sub> |
| RootBA-QR (16) | 102.23<br><sub>3,254,156</sub> | 1,278.84<br><sub>10,158,450</sub> |
| PowerBA (16) | 6.15<br><sub>3,520,615</sub> | 33.72<br><sub>10,441,192</sub> |
| **H-GBP (16)** | 3.03<br><sub>3,240,217</sub> | 11.20<br><sub>10,331,432</sub> |

## GBP at every level

The optional [`pure_gbp`](code/variants/pure_gbp/README.md) variant retains
hierarchical abstraction and recovery while using GBP for fine smoothing and
both coarse levels, **without a coarse direct or Krylov solve**. Same-machine
16-thread, outer20 tests on all nine PGO inputs take **1.45-2.51x** the default
runtime, with a maximum final-cost increase of
**0.451%**. The faster coarse-direct route remains the default.
The independent folder contains the shared-policy configuration, build/run
instructions, numerical tests and [paired evidence](code/variants/pure_gbp/evidence/summary.json).
