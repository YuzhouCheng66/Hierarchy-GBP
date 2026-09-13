# OpenVINS live end-to-end experiment

This branch integrates the existing covariance-root Gaussian backend into actual
OpenVINS state ownership. It consumes EuRoC images and IMU, runs tracking and
initialization, updates the real nominal state, and derives subsequent Jacobians
from its own trajectory. See [results/REPORT.md](results/REPORT.md) for measured
complete-sequence accuracy and timings.

Measured on one CPU core across all six full-sequence configurations: **1.042–1.185x
complete-process speedup** versus official OpenVINS, with **1.066–1.313x camera+IMU
API compute speedup**. All frozen accuracy checks pass. The largest candidate
position difference from official is 2.28 micrometres; the largest ATE RMSE
difference is 2.18e-7 m. These results do not support a 3x complete-system claim.

Validation includes 54 timed processes plus 18 warmups, 12 full-sequence audit
processes, 206,875 audited feature gates, and independent native tests. Six
ill-conditioned gates required the documented high-precision audit. The maximum
relative covariance audit discrepancy is 1.79e-11. The public results contain
all trials, per-frame timings, representative generated trajectories, build logs,
source/library hashes and the earlier failed audit attempts.

The experiment is deliberately labeled **root_messages**, not hierarchical GBP.
It does not establish that hierarchy causes a speedup. The original repository's
hierarchical pose-graph/bundle-adjustment implementation is unchanged.

| Method | State covariance | Visual update | Role |
|---|---|---|---|
| original | Official OpenVINS covariance | Unmodified official algorithm | Primary baseline |
| root_messages | Persistent upper covariance root with clone aliases | Shared Gaussian responses, certified gates, exact posterior | Candidate |
| enhanced_ekf | Dense covariance on the same unique coordinates | Same shared responses and certificate | Our optimized supplementary control |

The official source is pinned to [OpenVINS 6948812](https://github.com/rpng/open_vins/tree/69488123ed9362dd44b6f28e7f4680abbff1442b).
It is built from a separate clean checkout, with an external folder input/output
driver. Only execution controls are forced to one thread. The integrated checkout
has seven explicit source-interface patches; before/after hashes are recorded.

## Validated scope

Linux x86-64, single pinned CPU; stereo KLT; EuRoC V1_01_easy, V1_02_medium,
V1_03_difficult; clone limits 11 and 21; fixed calibration; GLOBAL_3D MSCKF;
static initialization. Online calibration, dynamic initialization, SLAM landmarks,
ZUPT and ArUco are rejected by the integration. GPU, multithreaded speedups, other
datasets and default configurations with different features are not established.
Ground truth is loaded only by offline scoring, never by the VIO executable.

## Build

Required: Git, CMake >= 3.19, Ninja, a C++17 compiler, Python 3, Eigen3, OpenCV4,
Boost, Ceres, glog/gflags and Intel MKL with the LP64 sequential libraries. The
measured build uses GCC 13.3, Eigen 3.4, OpenCV 4.6, Boost 1.83, Ceres 2.2
and a Core Ultra 5 245KF. No dependency
binaries or datasets are included. Install the OpenVINS dependencies through your
system or a separate prefix, and set the directory containing libmkl_intel_lp64,
libmkl_sequential and libmkl_core. The benchmark uses the original Eigen kernels
for official OpenVINS; it does not inject MKL into official estimator algebra.

From this directory:

```bash
python3 scripts/build.py \
  --work-dir "$PWD/local-runs/build" \
  --dependency-prefix /usr \
  --mkl-lib-dir /path/to/mkl/lib/intel64
python3 scripts/test_backend.py \
  --work-dir "$PWD/local-runs/build" \
  --output "$PWD/local-runs/unit"
```

An optional `--reference-repo /path/to/open_vins` avoids downloading a second Git
repository. The script creates isolated detached worktrees; it never edits that
reference checkout. Without this argument it clones upstream. `--jobs` controls
compilation only. Existing recognized patches can be updated; unrelated source
edits in the patched files are refused. Failed build logs and older manifests are preserved. The
binary is optimized for the build CPU (`-march=native`); rebuild on a different
machine rather than copying it.

## Run and reproduce

Arrange each unmodified EuRoC download as `$EUROC_ROOT/V1_01_easy/mav0/...`, etc.
Each output directory must be new, so a retry does not overwrite evidence.

```bash
python3 scripts/run_suite.py \
  --work-dir "$PWD/local-runs/build" --data-dir "$EUROC_ROOT" \
  --output "$PWD/local-runs/smoke" --phase smoke \
  --sequences V1_01_easy --clones 11
python3 scripts/run_suite.py \
  --work-dir "$PWD/local-runs/build" --data-dir "$EUROC_ROOT" \
  --output "$PWD/local-runs/audit" --phase audit
python3 scripts/run_suite.py \
  --work-dir "$PWD/local-runs/build" --data-dir "$EUROC_ROOT" \
  --output "$PWD/local-runs/formal" --phase formal
python3 scripts/analyze_suite.py \
  --suite "$PWD/local-runs/formal" --audit "$PWD/local-runs/audit" \
  --output "$PWD/local-runs/analysis"
```

Use an available isolated core via `--cpu N` if CPU 0 is unsuitable. Do not run
other heavy jobs during formal timing. The formal suite performs one warmup and
three independent timed processes for every method/cell, in a fixed shuffled
order. Thread controls, runtime library paths, source/binary/configuration hashes,
input CSV hashes, run order, logs, generated trajectories and scores are recorded.
The estimator runs faster than real time; sensor timestamps retain their original
meaning. No dataset frames are artificially downsampled for speed.

The executable also accepts `CONFIG DATA OUTPUT_PREFIX [DURATION_SECONDS]`;
negative duration selects the full sequence. In the integrated binary select
`OV_GAUSSIAN_BACKEND=root_messages` or `enhanced_ekf`, with the library environment
from its build manifest. `OV_GAUSSIAN_AUDIT=1` enables expensive validation; it
must be zero for timing. The clean binary accepts only `original`.

## Review guide

- [PROTOCOL.md](PROTOCOL.md): admission rules frozen before the first live build.
- [THEORY.md](THEORY.md): exact posterior, clone aliases, shared gates and retirement.
- [AUDIT_REFINEMENT.md](AUDIT_REFINEMENT.md): documented correction of an unstable numerical oracle, before timing.
- [scripts/patch_openvins.py](scripts/patch_openvins.py): all OpenVINS interface changes.
- [patches/openvins-6948812.patch](patches/openvins-6948812.patch): exact applied diff for review.
- [src/Backend.cpp](src/Backend.cpp): persistent production state and Gaussian operations.
- [src/Audit.cpp](src/Audit.cpp): optional high-precision reference, never used in production mode.
- [tests/test_backend.cpp](tests/test_backend.cpp): dense Joseph/Schur references, clone singularities, retirement and gate tests.
- [results/REPORT.md](results/REPORT.md): actual results and claim boundaries.

Primary end-to-end time is outer process wall time, including startup, file input,
PNG decode, tracking, initialization, state updates and output. Replay-loop wall
and camera+IMU API compute times are separate supplementary measures. A previous
backend replay speedup above 3x does not imply a 3x complete-system speedup.

See [NOTICE.md](NOTICE.md) and [LICENSE](LICENSE) for attribution and licensing.
