# Hierarchy-GBP

This repository contains the cleaned, verified H-GBP implementations used for
the SE2 pose-graph, SE3 pose-graph, and bundle-adjustment experiments.
Experimental Aitken/Anderson solvers and superseded execution paths are not
included.

The former research workspace is preserved unchanged on the `legacy` branch.
The `main` branch contains only the benchmarked solver paths and their
reproduction metadata.

## Layout

- `src/se2_solver.cpp`, `include/hgbp/se2_solver.h`: SE2 solver.
- `src/se3_solver.cpp`, `include/hgbp/se3_solver.h`: SE3 solver.
- `src/ba_solver.cpp`, `src/ba_internal.h`: BA solver and its private support.
- `configs/pgo.json`: all formal SE2/SE3 parameters and reference results.
- `configs/ba.json`: all formal BA parameters and reference results.
- `configs/datasets.json`: input paths, dimensions, preprocessing, and SHA-256.
- `scripts/run_benchmarks.py`: the only formal reproduction driver.

Dataset-specific choices live in JSON rather than filename checks or
dataset-specific branches in C++.

## Dependencies

The reference build is Windows x64 with Visual Studio 2022, CMake 3.20 or
newer, Ninja, and Python 3.10 or newer. The measured binaries use:

- Eigen 3.4.0.
- SuiteSparse Config 7.8.3, CHOLMOD 5.3.0, and OpenBLAS 0.3.29 from vcpkg.
- RootBA commit `d3900037fc4bc98328e5d8d26f1c4eed922f88cc` and its runtime environment.
- MSVC OpenMP and AVX2 Release compilation.

Dependencies and datasets are deliberately not vendored. A dependency root
has the layout used below:

```text
<dependency-root>/
  external/eigen/
  external_baselines/rootba/
  vcpkg_installed/x64-windows/
```

The RootBA checkout must already provide `build-win/src/rootba/Release` and
`conda_env`. Individual paths can instead be supplied through
`HGBP_EIGEN_DIR`, `HGBP_VCPKG_INSTALLED_DIR`, and `HGBP_ROOTBA_DIR`.

## Build

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build.ps1 `
  -Config Release -Jobs 12 -DependencyRoot C:\path\to\dependencies
```

Alternatively set `HGBP_DEPENDENCY_ROOT` before invoking the script. Use
`-BuildDirectory build-clean` to create an independent validation build.

The build produces:

- `build/se2_benchmark.exe`
- `build/se3_benchmark.exe`
- `build/ba_solver.exe`

## Reproduce

Run all nine pose-graph datasets with either formal thread count:

```powershell
python .\scripts\run_benchmarks.py pgo --threads 1 --repeats 1 `
  --output-root .\results\formal_pgo_1thread
python .\scripts\run_benchmarks.py pgo --threads 16 --repeats 1 `
  --output-root .\results\formal_pgo_16thread
```

Run both thread modes, record the H-GBP-1/H-GBP-16 timing ratio, and verify
that neither path regresses beyond the configured wall-time tolerance:

```powershell
python .\scripts\run_benchmarks.py pgo --compare-threads --repeats 3 `
  --output-root .\results\formal_pgo_scaling
```

The paired runner writes `thread_scaling_summary.json` and CSV. The historical
ratio remains a visible diagnostic; the hard gate is per-thread performance
plus cross-thread numerical consistency, so a genuine speedup in either path
is not reported as a failure. A configured cooldown separates the two modes
to reduce thermal bias. A dataset subset can be supplied for a quick check.
Formal PGO accepts only `1` or `16` for `--threads`; obsolete solver-specific
thread flags are intentionally rejected.

Run all twelve configured BA datasets with either formal thread count:

```powershell
python .\scripts\run_benchmarks.py ba --threads 1 --repeats 1 `
  --output-root .\results\formal_ba_1thread
python .\scripts\run_benchmarks.py ba --threads 16 --repeats 1 `
  --output-root .\results\formal_ba_16thread
python .\scripts\run_benchmarks.py ba --compare-threads --repeats 1 `
  --output-root .\results\formal_ba_scaling
```

A subset can be named after the suite, for example:

```powershell
python .\scripts\run_benchmarks.py pgo Sphere Grid Globe100k
python .\scripts\run_benchmarks.py ba Ladybug49 Final394
```

The runner verifies every input SHA-256 before execution, records the resolved
configuration and command line, and checks the final cost (plus BA RMSE)
against the formal reference. BA additionally checks mean reprojection error
(MRE), and `--threads 1` caps build, GBP, TBB, OpenMP, and BLAS execution to
one worker. Use `HGBP_PGO_DATA_ROOT`, `HGBP_BA_DATA_ROOT`,
and `HGBP_ROOTBA_RUNTIME` to relocate data or the RootBA runtime.
Without overrides, the runner looks under `data/pgo`, `data/ba`, and
`third_party/rootba/conda_env` relative to the repository root.

Wall time depends on power mode and machine load. Numerical acceptance is the
authoritative reproduction check.

The publication comparison in `results/pgo_comparison_with_hgbp1.tex`
normalizes both H-GBP timing columns to the earlier table machine. The exact
ratio calculation is recorded in `results/pgo_time_normalization_20260727.json`;
the reference times below remain direct measurements on the current machine.
Because each dataset uses the same calibration factor for both H-GBP columns,
normalization preserves the H-GBP-1/H-GBP-16 ratio exactly. The
`--compare-threads` report preserves that provenance while allowing a newer
build to improve either thread path.

The BA publication table follows the same rule. Its source is
`results/ba_comparison_with_hgbp1.tex`, and the exact H-GBP-1 normalization is
recorded in `results/ba_time_normalization_20260727.json`. Both Ceres columns,
RootBA-QR, and PowerBA were run with 16 requested and 16 reported worker
threads; the table headers state this explicitly.

## Formal PGO Results

Each result is `time [s] / cost`. Times are medians of seven runs for SE2,
three runs for the smaller SE3 datasets, and three runs for SE3 1-thread
Globe100k. The large SE3 16-thread values are single full runs.

| Dataset | Nodes / factors | H-GBP-1 | H-GBP-16 |
|---|---:|---:|---:|
| FR079 | 989 / 1,217 | 0.785703 / 18.836390220 | 0.182861 / 18.837328430 |
| FRH | 1,316 / 2,820 | 0.269175 / 9.65709797e-5 | 0.126607 / 9.65709797e-5 |
| M3500 | 3,500 / 5,453 | 0.706199 / 68.988146786 | 0.276352 / 68.988146786 |
| ParkingGarage | 1,661 / 6,275 | 6.048004 / 0.634259907 | 1.756354 / 0.634259908 |
| Sphere | 2,500 / 4,949 | 2.869410 / 675.700963087 | 1.001534 / 675.700963087 |
| Cubicle | 5,750 / 16,869 | 19.167832 / 1384.276197764 | 4.772861 / 1384.276197764 |
| Grid | 8,000 / 22,236 | 15.075971 / 42236.335716683 | 4.333509 / 42236.335716683 |
| Globe10k | 10,000 / 20,899 | 16.927147 / 3040.438162889 | 7.075695 / 3040.438162889 |
| Globe100k | 99,856 / 200,395 | 161.445846 / 20170.111526341 | 53.403826 / 20170.111526341 |

The shared SE3 configuration is outer `20`, cycles `5`, pre-sweeps `50`,
group size `20`, reduced rank `12`, threads `16`, and Huber delta `5`.
Per-dataset basis rebuild, fixed-lambda, and final-polish choices are recorded
in `configs/pgo.json`. Cubicle uses the offline PSD-floor-1 input identified by
the hash in `configs/datasets.json`. Both PGO thread modes use process affinity
mask `0xFFFF` so the hybrid CPU does not migrate timed work onto E-cores.
Cached SE3 coarse matrices use 16 deterministic accumulation shards in both
thread modes: one thread executes the shards serially, while 16 threads execute
the same shards in parallel and reduce them in the same order. This keeps
Cubicle, Grid, Globe10k, and Globe100k objectives identical across thread
counts. SE2 jitter and fixed-eta thresholds are explicit command-line
parameters recorded in every run JSON rather than hidden runner environment
overrides.

## Formal BA Results

Each result is `time [s] / cost / reprojection RMSE [px]`. These are current
machine references; the publication table uses the normalization described
above.

| Dataset | H-GBP-1 | H-GBP-16 |
|---|---:|---:|
| Ladybug49 | 0.207022 / 26942.349049 / 0.919837 | 0.100867 / 26942.349048 / 0.919837 |
| Venice89 | 2.041507 / 580007.057819 / 1.015013 | 0.535544 / 580007.057816 / 1.015013 |
| Final93 | 1.329496 / 291102.494527 / 1.006331 | 0.327726 / 291102.494527 / 1.006331 |
| Ladybug138 | 0.517570 / 122678.814212 / 1.199835 | 0.166671 / 122678.814212 / 1.199835 |
| Trafalgar257 | 1.082858 / 201643.285957 / 0.944764 | 0.321548 / 201643.285956 / 0.944764 |
| Dubrovnik356 | 8.417697 / 1040079.485323 / 0.910259 | 1.283900 / 1040079.485325 / 0.910259 |
| Final394 | 3.849099 / 603152.714526 / 1.062373 | 0.764902 / 603152.704097 / 1.062373 |
| Ladybug1723 | 3.075111 / 771223.261789 / 1.065971 | 0.752244 / 771223.751280 / 1.065971 |
| Venice1778 | 37.817810 / 3390248.696771 / 0.823278 | 4.824911 / 3390248.696770 / 0.823278 |
| Final1936 | 30.913472 / 9306268.679998 / 1.336021 | 6.541774 / 9306268.679996 / 1.336021 |
| Final3068 | 10.763374 / 3496404.580924 / 1.454011 | 1.555132 / 3496404.583177 / 1.454011 |
| Final4585 | 51.243364 / 10447354.889746 / 1.070000 | 6.137337 / 10447354.889733 / 1.070000 |

All BA runs use 20 outer iterations. Grouping, retained pair sampling,
initial damping, pose scaling, and unary reduction are explicitly listed in
`configs/ba.json`.
