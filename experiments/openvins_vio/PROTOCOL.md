# Frozen end-to-end protocol

Written before the first integrated executable was built or run (2026-09-13).

Scope: EuRoC V1_01_easy, V1_02_medium, V1_03_difficult; clone limits 11 and 21;
complete sequences. Stereo KLT, fixed calibration, GLOBAL_3D MSCKF, static
initialization, no SLAM landmarks, ZUPT or ArUco. All three methods receive the
same raw camera and IMU records and configuration. Ground truth is used only by
the offline scorer. Each method generates its own nominal states and subsequent
Jacobians; no replayed residuals, covariances, accept/reject labels or reset shifts.

Methods: clean pinned official OpenVINS; root_messages; enhanced_ekf (our optimized
supplementary control, not official). No additional algorithm is introduced to
make a timing result pass. Integration fixes are allowed and must be documented.

Correctness first: deterministic dense-reference unit test and untimed live audits
of both candidate representations in all six cells. Audits compare the persistent
physical covariance and actual tangent increment against an independent dense
Joseph calculation, and every feature gate against a dense innovation solve.
Matrix/vector Frobenius tolerance: 1e-10 + 1e-6 * reference norm; minimum covariance
eigenvalue >= -1e-10. Audits must be disabled during formal timing.

Trajectory admission, per cell: successful initialization; equal processed and
initialized frame counts and estimate timestamps; finite states; yaw/translation
aligned ATE RMSE and 1-second translational RPE RMSE no worse than official by
max(1 mm, 5% of official). Rotational RPE RMSE no worse by max(0.01 degree, 5%).
Report exact trajectory discrepancies too; this is an engineering noninferiority
budget, not a statistical or universal equivalence theorem. All failures remain
in the report and must not be silently dropped from the six-cell scope.

Timing: one warmup per method/cell, followed by three independent process trials
per method/cell in a reproducibly shuffled order (seed 20260913). One pinned CPU,
one OpenCV/Eigen/BLAS thread, sequential MKL for both custom representations;
identical compiler optimization and original Eigen implementation for official.
No concurrent heavy jobs. Dataset pages are warm after warmup. Record hardware,
affinity, software, source and executable hashes, run order, return codes and logs.

Report all three trial values and medians. Primary system timing is outer process
wall time, including startup, loading, PNG decode, tracking, initialization,
propagation, update and output. Also report replay-loop wall time and camera+IMU
API compute time separately. Never substitute backend-only timing for end-to-end
speedup. Do not claim a 3x system speedup unless these measurements support it.
Three trials characterize this machine/run only, not population confidence.

The root candidate is an exact covariance-root Gaussian elimination implementation
with shared feature responses, norm-certified gates and right-Givens retirement.
This experiment does not establish an advantage caused by hierarchical GBP.
