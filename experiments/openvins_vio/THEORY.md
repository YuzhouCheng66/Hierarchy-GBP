# Live Gaussian backend

## What is being tested

This is a nonhierarchical Gaussian elimination implementation of the existing
MSCKF posterior. It is not a tree of independently iterated GBP messages. IMU
motion has a temporal chain, but multi-view visual constraints and the correlated
prior couple retained poses. An exact two-pass chain argument alone does not
establish a speedup for this full problem. The present experiment tests the
previously benchmarked root implementation after genuine live integration.

## Unique coordinates and deterministic clones

Let n denote velocity and the two IMU biases (9 coordinates), and p all unique
pose errors. Store an upper covariance root

\[
P_u=VV^T,\qquad
V=\begin{bmatrix}D&E\\0&U\end{bmatrix},\qquad
\delta x_{physical}=A\delta x_u.
\]

A is a row-selection/alias map, not a numerical inverse. Creating a stochastic
clone of the current pose appends its existing rows to A. This represents the
singular physical covariance exactly without jitter. Propagation creates a new
current pose, leaving old clones correlated. The physical covariance is AP_uA^T;
small covariance queries multiply only the selected root rows.

For IMU transition x' = F_n n + F_c c + w, substitute n = D z_n + E z_p and
c = U_c z_p. The new IMU response to old poses is F_n E + F_c U_c; its independent
noise covariance is Q + (F_n D)(F_n D)^T. A 15-dimensional reverse Cholesky of this
noise plus the old pose root yields the exact expanded upper root. Navigation
coordinates are permuted back to [v,bg,ba,current pose,newest...oldest clones].

## Shared visual responses

Upstream tracking, feature cleaning, triangulation, FEJ and Jacobian evaluation
run on the method's own current nominal state. Whiten a feature by its pixel
standard deviation and eliminate its 3D nuisance position by Householder QR.
The reduced model is r = H_p delta p + epsilon, epsilon ~ N(0,I). Form Z = H_p U
once. Its innovation covariance and gate statistic are

\[
S=I+ZZ^T,\qquad \chi^2=r^T S^{-1}r\leq\|r\|^2.
\]

A conservative floating-point upper bound on the squared norm certifies small
residual acceptance. The implementation uses FMA followed by outward nextafter
rounding for each accumulation, requires nearest rounding and gradual underflow,
and rejects nonfinite inputs. If certification fails, it evaluates the usual
innovation statistic. No heuristic rejection or relaxed threshold is used.
The certificate applies to the computed reduced FP64 model; the live audit also
checks its decision and statistic against a dense calculation of that model.

Stack accepted responses and residuals. With G=I+Z^T Z=L L^T and B=[E;U],

\[
\delta x_u=B G^{-1}Z^T r,\qquad
V^+=\begin{bmatrix}D & B L^{-T}\end{bmatrix}.
\]

Only pose response coordinates need the Gram solve. This is an exact Gaussian
posterior at the current linearization, not an approximation or a new estimator.
OpenVINS Type::update injects A delta x_u into the actual orientation, position,
velocity and bias values; the next frame uses those values. The persistent root
owns subsequent covariance propagation, cloning, gates, updates and retirement.
The old private covariance is emptied to expose missed integration paths.

## Retirement

Deleting a physical alias only removes a row of A if another alias still needs
the corresponding unique coordinate. Deleting the oldest unique pose removes
an upper-root suffix. For retained root rows [V11 V12], right Givens rotations
produce Vnew with Vnew Vnew^T = V11 V11^T + V12 V12^T. This preserves the marginal
without forming the full covariance. An interior unique removal uses an exact
RQ factorization of retained rows. Both paths have independent dense tests.

## Supplementary enhanced EKF

The control stores P_u, factors its pose block P_pp=L_p L_p^T, and reuses the same
feature elimination, responses and norm certificate. Its transport is
[P_np L_p^{-T}; L_p], with conditional inertial covariance
P_nn - P_np P_pp^{-1} P_pn. It applies the same Gaussian update and reconstructs
its covariance. This isolates the advantage of retaining a root from much of the
shared kernel engineering. Gains against unmodified upstream alone cannot be
attributed entirely to a root representation or to hierarchy.

## Audit and limits

Untimed audit mode maintains an independent dense physical covariance through
all live transitions, clone operations and marginalizations. It computes dense
innovation gates and Joseph updates using that run's actual factors, and compares
the increment before nominal injection. It does not replace the production root
with its reference result. The unit test additionally checks general nuisance
elimination against a dense Schur projector without using the backend QR basis.
Ill-conditioned gates that disagree in FP64 are checked in 113-bit arithmetic;
both independently represented models and the production decision must agree.
See AUDIT_REFINEMENT.md for the preserved failure and high-precision diagnosis.

Numerically equivalent factorizations need not produce bit-identical nonlinear
trajectories: tiny changes can affect later triangulation or gating. Full-sequence
ATE/RPE and timestamp coverage are therefore evaluated separately. Hierarchical
message scheduling, GPU execution, online calibration, SLAM landmarks, dynamic
initialization, ArUco and ZUPT are outside this integration's validated scope.
