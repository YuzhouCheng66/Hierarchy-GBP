# Measured BA Baselines

Reproduction packaging, not new solver implementations. Python 3.9+ and external
native binaries/libraries are required; nothing is downloaded or built by this
runner. Run commands below from the release root. Do not run alongside another
benchmark. The runner is sequential within one invocation, not a machine-wide lock.

## Archived Binaries First

Set `BA_ARCHIVE` to your supplied `paper_full_retest_20260923` archive and
`ROOTBA_RUNTIME` to the matching runtime DLL directory (historically the RootBA
environment's `Library/bin`). These binaries/runtime are not in the HF data package.
Use the release downloader for the pinned dataset revision:

```powershell
python -B scripts/download_datasets.py --datasets Ladybug49
python -B baselines/ba/run.py --binary-root "$env:BA_ARCHIVE/ba_16/binaries" --runtime-dir "$env:ROOTBA_RUNTIME" --data-root data/ba --datasets Ladybug49 --profiles Ceres-Sparse-1 --output results/ba-smoke-sparse1 --require-measured-binaries --dry-run
```

After other benchmarks finish, remove **only** `--dry-run` for the small max20
Ceres Sparse1 smoke. The output directory must not already exist. A native
Ladybug49 smoke has passed for all five profiles; see validation scope below.
To smoke the four 16-thread profiles sequentially:

```powershell
python -B baselines/ba/run.py --binary-root "$env:BA_ARCHIVE/ba_16/binaries" --runtime-dir "$env:ROOTBA_RUNTIME" --data-root data/ba --datasets Ladybug49 --profiles Ceres-Sparse-16 Ceres-Iterative-16 RootBA-QR-16 PowerBA-16 --output results/ba-smoke-16 --require-measured-binaries
```

The default inherited CPU mask is the measured Windows mask `0xffff` (first 16
logical processors), including for Sparse1. Use another explicit mask or `none`
only when required by the host, and report that difference. Thread budgets do not
prove simultaneous utilization. Repeat `--runtime-dir` for additional required
library directories in search order. `--timeout` defaults to the measured 2000
process-wall seconds; `--repeats` defaults to one. No solver/numeric override flags
are exposed. Set all desired datasets explicitly; the main table has ten, while
OneDSfMAlamo is supplementary and not included in the public all19 data package.

[HGBP-Benchmarks](https://huggingface.co/datasets/yuzhoucheng66/HGBP-Benchmarks/tree/e57f8d72b4dd8f02d9d17d578696aa552b4e10af)
is pinned at `e57f8d72b4dd8f02d9d17d578696aa552b4e10af`. The common manifest binds
each input by SHA256 and shape; `--data-root` is the restored **BA subdirectory**,
not its parent. This runner never downloads, normalizes, or modifies input files.

## Fixed Profiles And Measurement

`profiles.json` is the single input-independent command template. Its
`inherited_defaults` section documents the pinned library, not extra argv or a
TOML file to install. Every run starts in a fresh directory **without**
`rootba_config.toml`, exactly as the measured stdout confirms. Supplying a different
library version can change implicit defaults even if its executable has the right
name. `--require-measured-binaries` checks archived executable hashes; rebuilt
binaries must omit that flag and are explicitly labelled unmatched, not certified.

| Profile | Archived entry point and selection |
| --- | --- |
| Ceres-Sparse-1 / -16 | `bal_ceres_sparse_schur`, Ceres `SPARSE_SCHUR` |
| Ceres-Iterative-16 | `bal_ceres_iterative_schur`, `ITERATIVE_SCHUR` + `SCHUR_JACOBI` |
| RootBA-QR-16 | `bal_qr`, official `SQUARE_ROOT` solver |
| PowerBA-16 | `bal_power_sc`, official `POWER_SCHUR_COMPLEMENT`, power order 10 |

All use double, native BAL normalization (scale 100), no perturbation/filtering,
nonrobust squared error over all observations, max20 outer iterations, and zero
function/gradient/parameter tolerances. Early termination and failed LM trials
remain native; max20 does not promise 20 accepted steps or equivalent work. Linear
solver defaults include max500 and eta0.1; Power's separate order is 10. QR uses
Householder, staged execution and reduction algorithm 1. Ceres' camera manifold,
point-first ordering, residuals and callbacks are RootBA's implementation, not the
stock Ceres bundle-adjuster example. There is no per-input parameter dispatch.

`runs.json` retains every failure/timeout and each native solver summary. A nonzero
exit indicates incomplete/invalid results. `summary.json` reports successful-only
medians plus failure counts; it never converts a failed group into a complete one.
Raw `ba_log.json` and stdout remain in each run directory. Native histories, shape,
threads, termination, finite metrics, max20 and initial-objective consistency are
checked. No historical cost tolerance has been invented as a pass criterion.

SSR = twice native half-SSR; RMSE = sqrt(SSR / number of **2D observations**);
MRE = mean Euclidean 2D residual norm. These are native log metrics, **not independent
rescoring of saved final geometry**. Native time is `_static.solver.total_time_in_seconds`:
RootBA/Power include internal setup and optimization; Ceres reports
`ceres::Solve`'s internal preprocessing/minimizer/postprocessing but excludes the
wrapper's earlier camera/Problem construction. Both exclude loading and BAL
normalization. Process wall is recorded separately and includes those operations.
Do not describe these as identical end-to-end timing boundaries.

`evidence.json` retains all 55 table cells: 54 successful one-repeat measurements
and the Ceres Sparse1 Final4585 2000-second timeout. The September 24 accepted table
reused these September 23 baseline runs; only H-GBP was replaced by later accepted
dataflow runs. One sample is not a timing distribution. Archive-relative paths are
provenance labels; raw archives are not shipped here. `runtime-measured.json` lists
archived runtime-directory DLL hashes, not a complete loaded-module trace. New runs
hash explicit runtime directories and adjacent DLLs, but OS/PATH fallback libraries,
compiler ABI, CPU, power policy and runtime provenance still affect reproducibility.

## Source And Build

No dependencies are vendored. `src/bal_ceres.cpp`, `src/bal_qr.cpp`, and
`src/bal_power_sc.cpp` are the actual archived entry points, unchanged except line
endings. Ceres selection is compile-time to avoid the measured MSVC enum-parser
problem. The independent CMake project links an existing matching static Release
RootBA build; it does not build the production H-GBP solver or alter any algorithm.

Pins: [RootBA](https://github.com/NikolausDemmel/rootba/tree/d3900037fc4bc98328e5d8d26f1c4eed922f88cc)
`d3900037fc4bc98328e5d8d26f1c4eed922f88cc`, including its recursive submodules;
[Ceres 2.2.0](https://github.com/ceres-solver/ceres-solver/releases/tag/2.2.0)
`85331393dc0dff09f6fb9903ab0c4bfa3e134b01`; Eigen **3.4.0**. The measured Ceres
install overrides RootBA's older Ceres submodule. Do not substitute the unrelated
Eigen version in a general-purpose runtime prefix.

The measured source was **not pristine upstream**. `patches/rootba-measured.patch`
is the archived tracked diff, including build portability changes, wrapper target
definitions and dormant H-GBP Schur-access helpers. Those helpers are preserved as
evidence, not newly introduced algorithms: Power's existing call leaves the added
scaling argument false and still scales separately. `source-lock.json` fingerprints
all 148 archived RootBA source/build files after LF normalization. Current source
matched those 148 entries at audit. Untracked anchor/helper sources are included
under `src/`; a tracked diff alone would miss them.

For a **new external checkout**, use the pinned commit, apply the archival patch
excluding its two non-applicable dirty-submodule marker hunks, then the recovered
submodule patches. These submodule diffs were recovered from the current local
checkout; the old zip did not archive their full contents. Fresh rebuild parity
therefore still requires verification. `$Package` below is this directory and
`$RootBA` is your external checkout, neither a release-vendored dependency.

```powershell
git clone --recurse-submodules https://github.com/NikolausDemmel/rootba.git $RootBA
git -C $RootBA checkout d3900037fc4bc98328e5d8d26f1c4eed922f88cc
git -C $RootBA submodule update --init --recursive
git -C $RootBA apply --exclude=external/basalt-headers --exclude=external/visit_struct "$Package/patches/rootba-measured.patch"
git -C "$RootBA/external/basalt-headers" apply "$Package/patches/basalt-msvc.patch"
git -C "$RootBA/external/visit_struct" apply "$Package/patches/visit-struct-msvc.patch"
Copy-Item "$Package/src/bal_power_sc.cpp" "$RootBA/src/app/bal_power_sc.cpp"
Copy-Item "$Package/src/cg_anchor.cpp" "$RootBA/src/rootba/cg/anchor.cpp"
Copy-Item "$Package/src/sc_anchor.cpp" "$RootBA/src/rootba/sc/anchor.cpp"
Copy-Item "$RootBA/external/clipp/clipp/include" "$RootBA/external/clipp/include/clipp" -Recurse
Copy-Item "$RootBA/external/nameof/nameof/include" "$RootBA/external/nameof/include/nameof" -Recurse
```

The last two commands replace the Windows symlinks removed by the measured patch.
Supply external Eigen3.4, SuiteSparse, TBB, glog/gflags, fmt, Abseil and Pangolin
packages. Pangolin is required at configure time by upstream even for CLI targets.
The archived cache selected static libraries, Visual Studio 16 2019 x64, Release,
double ON, float OFF, static landmark blocks ON, QR/Ceres ON, tests OFF. RootBA
resolved Abseil from a **different** installed prefix; the archive is not a complete
dependency lock. All packages must use compatible compiler/CRT/ABI settings.

Recipe templates, **not build-validated during packaging** (run only after the
benchmark queue is idle). `$Deps`, `$EigenPrefix`, `$AbseilPrefix` and `$CeresPrefix`
are user-supplied installations, `$CeresSource` is the pinned Ceres checkout:

```powershell
cmake -S $CeresSource -B "$CeresSource/build" -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="$CeresPrefix" -DCMAKE_PREFIX_PATH="$EigenPrefix;$Deps;$AbseilPrefix" -DBUILD_SHARED_LIBS=OFF -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DUSE_CUDA=OFF -DLAPACK=OFF -DSUITESPARSE=ON
cmake --build "$CeresSource/build" --config Release --target install --parallel 1
cmake -S $RootBA -B "$RootBA/build" -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH="$CeresPrefix;$EigenPrefix;$Deps;$AbseilPrefix" -DCeres_DIR="$CeresPrefix/lib/cmake/Ceres" -DEigen3_DIR="$EigenPrefix/share/eigen3/cmake" -DBUILD_SHARED_LIBS=OFF -DROOTBA_BUILD_CERES=ON -DROOTBA_BUILD_QR=ON -DROOTBA_INSTANTIATIONS_DOUBLE=ON -DROOTBA_INSTANTIATIONS_FLOAT=OFF -DROOTBA_INSTANTIATIONS_STATIC_LMB=ON -DROOTBA_ENABLE_TESTING=OFF
cmake --build "$RootBA/build" --config Release --target rootba_cli rootba_ceres rootba_solver --parallel 1
cmake -S baselines/ba -B build-ba-baselines -G "Visual Studio 16 2019" -A x64 -DROOTBA_SOURCE="$RootBA" -DROOTBA_BUILD="$RootBA/build" -DCMAKE_PREFIX_PATH="$CeresPrefix;$EigenPrefix;$Deps;$AbseilPrefix"
cmake --build build-ba-baselines --config Release --parallel 1
```

Use `build-ba-baselines/Release` as `--binary-root` for rebuilt wrappers, without
`--require-measured-binaries`. This recipe is not a bitwise-build promise; neither
fresh builds nor Linux/macOS numerical parity have been validated here.

## Tests And Licenses

```powershell
python -B -m unittest discover -s tests -p test_ba_baselines.py -v
```

Unit tests use tiny fake inputs, mocked processes and temporary outputs only.
Separately, on 2026-09-25 the primary release task completed a native Ladybug49
smoke with all five profiles, one run each, using `--require-measured-binaries`.
All five executable hashes matched the archived measurements, the input SHA256
matched the manifest, affinity `0xffff` was applied, and all five results were
`ok`. Final SSR values agreed with the historical rows to displayed precision
(Sparse1: 26690.469599; PowerBA16: 27299.285226), not necessarily bitwise.
Evidence labels: `archives/dataset_release_20260925/baselines_ba_smoke/`
`protocol.json`, `runs.json`, and `summary.json` (not bundled here).
This validates the packaged runner with measured binaries on one small input;
it is not a full-dataset regression, fresh-build validation, or timing study.
Solver builds and wrapper sources were unchanged.

RootBA
wrappers/patches and Basalt are BSD-3-Clause; visit_struct is Boost Software License
1.0. Notices are retained in `licenses/`. Ceres is BSD-3-Clause and stays external.
Other external dependencies retain their own licenses (including any SuiteSparse
component-specific obligations); this package grants no right to redistribute a
whole runtime or dataset. No third-party binaries, data, or dependency trees are
shipped. The primary release must still choose/document its own packaging-code
license and arrange archival binary/runtime distribution if desired.
