#include "internal/se2_residual.h"
#include "internal/scoped_worker_affinity.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <stdexcept>

#include <omp.h>

namespace slam {

namespace {

using SteadyClock = std::chrono::steady_clock;

#if defined(_MSC_VER)
#define PACKED_RESIDUAL_FORCEINLINE __forceinline
#define PACKED_RESIDUAL_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define PACKED_RESIDUAL_FORCEINLINE inline __attribute__((always_inline))
#define PACKED_RESIDUAL_RESTRICT __restrict__
#else
#define PACKED_RESIDUAL_FORCEINLINE inline
#define PACKED_RESIDUAL_RESTRICT
#endif

double wrapAngle(double a) {
    return std::atan2(std::sin(a), std::cos(a));
}

Eigen::Matrix2d rot2(double theta) {
    const double c = std::cos(theta);
    const double s = std::sin(theta);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    return R;
}

Eigen::Vector3d se2Compose(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    out.head<2>() = a.head<2>() + rot2(a(2)) * b.head<2>();
    out(2) = wrapAngle(a(2) + b(2));
    return out;
}

Eigen::Vector3d se2Inverse(const Eigen::Vector3d& a) {
    const Eigen::Matrix2d RT = rot2(a(2)).transpose();
    Eigen::Vector3d out = Eigen::Vector3d::Zero();
    out.head<2>() = -(RT * a.head<2>());
    out(2) = wrapAngle(-a(2));
    return out;
}

Eigen::Vector3d se2Between(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    return se2Compose(se2Inverse(a), b);
}

Eigen::Vector3d se2Exp(const Eigen::Vector3d& xi) {
    const double vx = xi(0);
    const double vy = xi(1);
    const double w = xi(2);
    if (std::abs(w) < 1e-12) {
        return Eigen::Vector3d(vx, vy, 0.0);
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    Eigen::Matrix2d V;
    V << a, -b,
         b,  a;
    const Eigen::Vector2d t = V * Eigen::Vector2d(vx, vy);
    return Eigen::Vector3d(t(0), t(1), wrapAngle(w));
}

Eigen::Vector3d se2Log(const Eigen::Vector3d& pose) {
    const double tx = pose(0);
    const double ty = pose(1);
    const double w = pose(2);
    if (std::abs(w) < 1e-12) {
        return Eigen::Vector3d(tx, ty, 0.0);
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double denom = a * a + b * b;
    Eigen::Matrix2d V_inv;
    V_inv <<  a, b,
             -b, a;
    V_inv /= denom;
    const Eigen::Vector2d v = V_inv * Eigen::Vector2d(tx, ty);
    return Eigen::Vector3d(v(0), v(1), wrapAngle(w));
}

Eigen::Vector3d se2Plus(const Eigen::Vector3d& base_pose, const Eigen::Vector3d& delta) {
    return se2Compose(base_pose, se2Exp(delta));
}

Eigen::Matrix3d jacobianExpSE2(const Eigen::Vector3d& xi) {
    const double vx = xi(0);
    const double vy = xi(1);
    const double w = xi(2);

    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    if (std::abs(w) < 1e-8) {
        J.setIdentity();
        J(0, 2) = -0.5 * vy;
        J(1, 2) = 0.5 * vx;
        return J;
    }

    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
    const double db = (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);

    J(0, 0) = a;
    J(0, 1) = -b;
    J(1, 0) = b;
    J(1, 1) = a;
    J(0, 2) = da * vx - db * vy;
    J(1, 2) = db * vx + da * vy;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianPlusSE2(const Eigen::Vector3d& base_pose, const Eigen::Vector3d& delta) {
    const double c = std::cos(base_pose(2));
    const double s = std::sin(base_pose(2));
    Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
    G << c, -s, 0.0,
         s,  c, 0.0,
         0.0, 0.0, 1.0;
    return G * jacobianExpSE2(delta);
}

Eigen::Matrix<double, 3, 6> jacobianBetweenAbsolute(const Eigen::Vector3d& xi, const Eigen::Vector3d& xj) {
    const double thi = xi(2);
    const double c = std::cos(thi);
    const double s = std::sin(thi);
    Eigen::Matrix2d RT;
    RT <<  c, s,
          -s, c;

    const Eigen::Vector2d dp = xj.head<2>() - xi.head<2>();
    const Eigen::Vector2d r = RT * dp;
    const Eigen::Vector2d dr_dthi(r(1), -r(0));

    Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
    J.block<2, 2>(0, 0) = -RT;
    J.block<2, 1>(0, 2) = dr_dthi;
    J.block<2, 2>(0, 3) = RT;
    J(2, 2) = -1.0;
    J(2, 5) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianComposeInvConstant(const Eigen::Vector3d& z) {
    const double c = std::cos(z(2));
    const double s = std::sin(z(2));
    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    J(0, 0) = c;
    J(0, 1) = s;
    J(1, 0) = -s;
    J(1, 1) = c;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix3d jacobianLogSE2(const Eigen::Vector3d& pose) {
    const double tx = pose(0);
    const double ty = pose(1);
    const double w = pose(2);

    Eigen::Matrix3d J = Eigen::Matrix3d::Zero();
    if (std::abs(w) < 1e-8) {
        J.setIdentity();
        J(0, 2) = 0.5 * ty;
        J(1, 2) = -0.5 * tx;
        return J;
    }

    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double den = a * a + b * b;
    const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
    const double db = (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);
    const double dden = 2.0 * (a * da + b * db);

    const double c = a / den;
    const double d = b / den;
    const double dc = (da * den - a * dden) / (den * den);
    const double dd = (db * den - b * dden) / (den * den);

    J(0, 0) = c;
    J(0, 1) = d;
    J(1, 0) = -d;
    J(1, 1) = c;
    J(0, 2) = dc * tx + dd * ty;
    J(1, 2) = -dd * tx + dc * ty;
    J(2, 2) = 1.0;
    return J;
}

Eigen::Matrix<double, 3, 6> analyticEdgeResidualJacobian(
    const Eigen::Vector3d& base_i,
    const Eigen::Vector3d& base_j,
    const Eigen::Vector3d& z
) {
    const Eigen::Vector3d xi = se2Plus(base_i, Eigen::Vector3d::Zero());
    const Eigen::Vector3d xj = se2Plus(base_j, Eigen::Vector3d::Zero());
    const Eigen::Vector3d pred = se2Between(xi, xj);
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(z), pred);

    const Eigen::Matrix<double, 3, 6> J_between_abs = jacobianBetweenAbsolute(xi, xj);
    const Eigen::Matrix3d J_compose = jacobianComposeInvConstant(z);
    const Eigen::Matrix3d J_log = jacobianLogSE2(err_pose);

    Eigen::Matrix<double, 6, 6> J_plus = Eigen::Matrix<double, 6, 6>::Zero();
    J_plus.block<3, 3>(0, 0) = jacobianPlusSE2(base_i, Eigen::Vector3d::Zero());
    J_plus.block<3, 3>(3, 3) = jacobianPlusSE2(base_j, Eigen::Vector3d::Zero());

    return J_log * J_compose * J_between_abs * J_plus;
}

Eigen::Matrix3d analyticAnchorResidualJacobian(
    const Eigen::Vector3d& base_anchor,
    const Eigen::Vector3d& anchor_pose
) {
    const Eigen::Vector3d xi = se2Plus(base_anchor, Eigen::Vector3d::Zero());
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(anchor_pose), xi);
    return jacobianLogSE2(err_pose) * jacobianComposeInvConstant(anchor_pose) * jacobianPlusSE2(base_anchor, Eigen::Vector3d::Zero());
}

PACKED_RESIDUAL_FORCEINLINE double* vec3Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

PACKED_RESIDUAL_FORCEINLINE const double* vec3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

PACKED_RESIDUAL_FORCEINLINE double* sym6Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

PACKED_RESIDUAL_FORCEINLINE const double* sym6Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

inline double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

inline int effectiveThreadCount(int requested_threads) noexcept {
    return std::max(1, (requested_threads > 0) ? requested_threads : omp_get_max_threads());
}

class LightweightSpinBarrier {
public:
    explicit LightweightSpinBarrier(
        int thread_count,
        int spin_limit = 8192,
        bool allow_yield = true
    ) noexcept
        : thread_count_(thread_count),
          spin_limit_(spin_limit),
          allow_yield_(allow_yield),
          remaining_(thread_count),
          generation_(0) {}

    template <class CompletionFn>
    void arrive_and_wait(CompletionFn&& completion) noexcept {
        const int generation = generation_.load(std::memory_order_acquire);
        if (remaining_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            completion();
            remaining_.store(thread_count_, std::memory_order_release);
            generation_.fetch_add(1, std::memory_order_acq_rel);
            return;
        }

        int spin_count = 0;
        while (generation_.load(std::memory_order_acquire) == generation) {
            if (spin_count < spin_limit_) {
                _mm_pause();
                ++spin_count;
            } else if (allow_yield_) {
                std::this_thread::yield();
            } else {
                _mm_pause();
            }
        }
    }

private:
    int thread_count_;
    int spin_limit_;
    bool allow_yield_;
    std::atomic<int> remaining_;
    std::atomic<int> generation_;
};

std::vector<std::pair<int, int>> buildBalancedVariableRanges(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int thread_count
) {
    std::vector<std::pair<int, int>> ranges(static_cast<size_t>(thread_count), {0, 0});
    if (thread_count <= 1 || workspace.num_vars <= 0) {
        if (!ranges.empty()) {
            ranges[0] = {0, workspace.num_vars};
        }
        return ranges;
    }

    std::vector<int64_t> prefix(static_cast<size_t>(workspace.num_vars) + 1, 0);
    for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
        const int unary_count = workspace.unary_offsets[var_idx + 1] - workspace.unary_offsets[var_idx];
        const int binary_count = workspace.binary_offsets[var_idx + 1] - workspace.binary_offsets[var_idx];
        prefix[static_cast<size_t>(var_idx) + 1] =
            prefix[static_cast<size_t>(var_idx)] + static_cast<int64_t>(1 + unary_count + binary_count);
    }

    const int64_t total_work = prefix.back();
    int start = 0;
    for (int tid = 0; tid < thread_count; ++tid) {
        if (tid == thread_count - 1) {
            ranges[static_cast<size_t>(tid)] = {start, workspace.num_vars};
            break;
        }
        const int64_t target = (total_work * static_cast<int64_t>(tid + 1)) / thread_count;
        int end = static_cast<int>(
            std::lower_bound(prefix.begin() + start + 1, prefix.end(), target) - prefix.begin()
        );
        end = std::max(end, start + 1);
        end = std::min(end, workspace.num_vars);
        ranges[static_cast<size_t>(tid)] = {start, end};
        start = end;
    }
    return ranges;
}

std::vector<std::pair<int, int>> buildBalancedIndexRanges(
    int count,
    int thread_count
) {
    std::vector<std::pair<int, int>> ranges(static_cast<size_t>(thread_count), {0, 0});
    if (thread_count <= 1 || count <= 0) {
        if (!ranges.empty()) {
            ranges[0] = {0, count};
        }
        return ranges;
    }
    for (int tid = 0; tid < thread_count; ++tid) {
        const int begin = (count * tid) / thread_count;
        const int end = (count * (tid + 1)) / thread_count;
        ranges[static_cast<size_t>(tid)] = {begin, end};
    }
    return ranges;
}

bool disablePackedOmpRangeSweeps() noexcept {
    static const bool disabled = []() {
        const char* value = std::getenv("GBP_DISABLE_PACKED_OMP_RANGE_SWEEPS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return disabled;
}

bool enablePackedSweepUnaryRefresh() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_SWEEP_UNARY_REFRESH");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool enablePackedSweepUnaryBarrier() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_SWEEP_UNARY_BARRIER");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool enablePackedOmpSpinPhaseBarriers() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_OMP_SPIN_PHASE_BARRIERS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool enablePackedOmpAggressiveSpinPhaseBarriers() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_OMP_AGGRESSIVE_SPIN_PHASE_BARRIERS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

}  // namespace

PACKED_RESIDUAL_FORCEINLINE bool computeBinaryFactorAtNoDampingNoThrow(
    const SyntheticSE2PackedResidualBinaryFactor& data,
    const double* belief0_eta,
    const double* belief1_eta,
    const double* belief0_lam6,
    const double* belief1_lam6,
    const double* old0_eta,
    const double* old1_eta,
    const double* old0_lam6,
    const double* old1_lam6,
    int target,
    double* out_eta,
    double* out_lam6
) noexcept;

PACKED_RESIDUAL_FORCEINLINE bool computeBinaryFactorAtNoDampingNoThrow(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE void updateBelief3DLocalNoMu(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept;

int fixedEtaAfterSweep() noexcept;

PACKED_RESIDUAL_FORCEINLINE double* fixedEtaMap0Ptr(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE double* fixedEtaMap1Ptr(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE const double* fixedEtaMap0Ptr(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE const double* fixedEtaMap1Ptr(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE bool buildFixedEtaMapsForFactorSE2Buffered(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    const double* belief_lam6_base,
    const double* msg_lam6_base
) noexcept;

PACKED_RESIDUAL_FORCEINLINE bool computeFixedEtaFactorSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    int* map_builds
);

PACKED_RESIDUAL_FORCEINLINE void fixedEtaUpdate3(
    const double* PACKED_RESIDUAL_RESTRICT factor_eta_target,
    const double* PACKED_RESIDUAL_RESTRICT factor_eta_other,
    const double* PACKED_RESIDUAL_RESTRICT eta_map,
    const double* PACKED_RESIDUAL_RESTRICT belief_other_eta,
    const double* PACKED_RESIDUAL_RESTRICT old_other_eta,
    double* PACKED_RESIDUAL_RESTRICT out_eta
) noexcept;

PACKED_RESIDUAL_FORCEINLINE void updateBelief3DLocalEtaOnlyNoMu(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept;

double robustWeightForResidual(
    const Eigen::Vector3d& err,
    const Eigen::Matrix3d& information,
    const RobustLossConfig& config
) {
    if (!(config.huber_delta > 0.0)) {
        return 1.0;
    }
    const double quad = (err.transpose() * information * err)(0, 0);
    const double mahal_norm = std::sqrt(std::max(0.0, quad));
    if (!(mahal_norm > config.huber_delta) || !(mahal_norm > 0.0)) {
        return 1.0;
    }
    return config.huber_delta / mahal_norm;
}

struct Cholesky3x3 {
    double l00;
    double l10;
    double l20;
    double l11;
    double l21;
    double l22;
};

constexpr double kJitter = 1e-10;

double getenvDouble(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const double parsed = std::strtod(value, &end);
    if (end == value) {
        return fallback;
    }
    return parsed;
}

int getenvInt(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value) {
        return fallback;
    }
    return static_cast<int>(parsed);
}

double initializePackedResidualJitterValue() {
    const double scale = getenvDouble("GBP_FASTJITTER_JITTER_SCALE", 1.0);
    const double abs_override = getenvDouble("GBP_FASTJITTER_ABS_JITTER", -1.0);
    if (abs_override > 0.0) {
        return abs_override;
    }
    return kJitter * scale;
}

PACKED_RESIDUAL_FORCEINLINE double packedResidualJitterValue() noexcept {
    static const double value = initializePackedResidualJitterValue();
    return value;
}

int fixedEtaAfterSweep() noexcept {
    static const int value = getenvInt("GBP_SE2_FIXED_ETA_AFTER_SWEEP", -1);
    return value;
}

bool fixedEtaSerialDeltaBeliefEnabled() noexcept {
    static const bool value = getenvInt("GBP_SE2_FIXED_ETA_SERIAL_DELTA", 1) != 0;
    return value;
}

bool fixedEtaParallelDeltaBeliefEnabled() noexcept {
    static const bool value = getenvInt("GBP_SE2_FIXED_ETA_PARALLEL_DELTA", 0) != 0;
    return value;
}

bool fixedEtaVerifyFullEnabled() noexcept {
    static const bool value = getenvInt("GBP_SE2_FIXED_ETA_VERIFY_FULL", 0) != 0;
    return value;
}

PACKED_RESIDUAL_FORCEINLINE bool factorizeSpd3x3Lower(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    Cholesky3x3& chol
) noexcept {
    if (!(a00 > 0.0)) {
        return false;
    }
    chol.l00 = std::sqrt(a00);

    chol.l10 = a10 / chol.l00;
    chol.l20 = a20 / chol.l00;

    const double d11 = a11 - chol.l10 * chol.l10;
    if (!(d11 > 0.0)) {
        return false;
    }
    chol.l11 = std::sqrt(d11);

    chol.l21 = (a21 - chol.l20 * chol.l10) / chol.l11;

    const double d22 = a22 - chol.l20 * chol.l20 - chol.l21 * chol.l21;
    if (!(d22 > 0.0)) {
        return false;
    }
    chol.l22 = std::sqrt(d22);
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void solveSpd3x3(const Cholesky3x3& chol, const double* b, double* x) noexcept {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.l10 * y0) / chol.l11;
    const double y2 = (b[2] - chol.l20 * y0 - chol.l21 * y1) / chol.l22;

    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.l21 * x[2]) / chol.l11;
    x[0] = (y0 - chol.l10 * x[1] - chol.l20 * x[2]) / chol.l00;
}

PACKED_RESIDUAL_FORCEINLINE bool solveGeneral3x3LowerSym(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    const double* b,
    double* x
) noexcept {
    Eigen::Matrix3d A;
    A << a00, a10, a20,
         a10, a11, a21,
         a20, a21, a22;
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) {
        return false;
    }
    const Eigen::Vector3d rhs(b[0], b[1], b[2]);
    const Eigen::Vector3d sol = lu.solve(rhs);
    x[0] = sol[0];
    x[1] = sol[1];
    x[2] = sol[2];
    return true;
}

PACKED_RESIDUAL_FORCEINLINE bool schurMessage3x3NoDampingGeneralPackedSym(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* PACKED_RESIDUAL_RESTRICT loo6,
    double b00,
    double b10,
    double b20,
    double b01,
    double b11,
    double b21,
    double b02,
    double b12,
    double b22,
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double* PACKED_RESIDUAL_RESTRICT out_eta,
    double* PACKED_RESIDUAL_RESTRICT out_lam6
) noexcept {
    Eigen::Matrix3d A;
    A << a00, a10, a20,
         a10, a11, a21,
         a20, a21, a22;
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) {
        return false;
    }

    Eigen::Matrix3d B;
    B << b00, b01, b02,
         b10, b11, b12,
         b20, b21, b22;
    const Eigen::Vector3d eno(eno0, eno1, eno2);
    const Eigen::Vector3d eo(eo0, eo1, eo2);
    const Eigen::Vector3d solved_eta = lu.solve(eno);
    const Eigen::Matrix3d solved_lam = lu.solve(B.transpose());
    const Eigen::Matrix3d loo = (Eigen::Matrix3d() <<
        loo6[0], loo6[1], loo6[2],
        loo6[1], loo6[3], loo6[4],
        loo6[2], loo6[4], loo6[5]).finished();
    const Eigen::Matrix3d schur = loo - B * solved_lam;
    const Eigen::Vector3d out_eta_vec = eo - B * solved_eta;

    out_eta[0] = out_eta_vec[0];
    out_eta[1] = out_eta_vec[1];
    out_eta[2] = out_eta_vec[2];
    out_lam6[0] = schur(0, 0);
    out_lam6[1] = schur(1, 0);
    out_lam6[2] = schur(2, 0);
    out_lam6[3] = schur(1, 1);
    out_lam6[4] = schur(2, 1);
    out_lam6[5] = schur(2, 2);
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void beliefEtaFromLamMu(const double* lam6, const double* mu, double* eta) noexcept {
    eta[0] = lam6[0] * mu[0] + lam6[1] * mu[1] + lam6[2] * mu[2];
    eta[1] = lam6[1] * mu[0] + lam6[3] * mu[1] + lam6[4] * mu[2];
    eta[2] = lam6[2] * mu[0] + lam6[4] * mu[1] + lam6[5] * mu[2];
}

PACKED_RESIDUAL_FORCEINLINE double* unaryMsgEtaPtr(SyntheticSE2PackedResidualWorkspace& workspace, int idx) noexcept {
    return workspace.unary_msg_eta.data() + static_cast<size_t>(idx) * 3;
}

PACKED_RESIDUAL_FORCEINLINE const double* unaryMsgEtaPtr(const SyntheticSE2PackedResidualWorkspace& workspace, int idx) noexcept {
    return workspace.unary_msg_eta.data() + static_cast<size_t>(idx) * 3;
}

PACKED_RESIDUAL_FORCEINLINE double* unaryMsgLam6Ptr(SyntheticSE2PackedResidualWorkspace& workspace, int idx) noexcept {
    return workspace.unary_msg_lam6.data() + static_cast<size_t>(idx) * 6;
}

PACKED_RESIDUAL_FORCEINLINE const double* unaryMsgLam6Ptr(const SyntheticSE2PackedResidualWorkspace& workspace, int idx) noexcept {
    return workspace.unary_msg_lam6.data() + static_cast<size_t>(idx) * 6;
}

PACKED_RESIDUAL_FORCEINLINE double* binaryMsgEtaPtr(SyntheticSE2PackedResidualWorkspace& workspace, int slot_id) noexcept {
    return workspace.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

PACKED_RESIDUAL_FORCEINLINE const double* binaryMsgEtaPtr(const SyntheticSE2PackedResidualWorkspace& workspace, int slot_id) noexcept {
    return workspace.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

PACKED_RESIDUAL_FORCEINLINE double* binaryMsgLam6Ptr(SyntheticSE2PackedResidualWorkspace& workspace, int slot_id) noexcept {
    return workspace.binary_msg_lam6.data() + static_cast<size_t>(slot_id) * 6;
}

PACKED_RESIDUAL_FORCEINLINE const double* binaryMsgLam6Ptr(const SyntheticSE2PackedResidualWorkspace& workspace, int slot_id) noexcept {
    return workspace.binary_msg_lam6.data() + static_cast<size_t>(slot_id) * 6;
}

PACKED_RESIDUAL_FORCEINLINE bool schurMessage3x3NoDampingAdjugatePackedSym(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* PACKED_RESIDUAL_RESTRICT loo6,
    double b00,
    double b10,
    double b20,
    double b01,
    double b11,
    double b21,
    double b02,
    double b12,
    double b22,
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double* PACKED_RESIDUAL_RESTRICT out_eta,
    double* PACKED_RESIDUAL_RESTRICT out_lam6
) noexcept {
    static const bool use_cholesky = getenvInt("HGBP_SE2_CHOLESKY_SCHUR",0) != 0;
    if (use_cholesky) {
        Cholesky3x3 chol;
        if (!factorizeSpd3x3Lower(a00,a10,a20,a11,a21,a22,chol)) return false;
        const double rhs[3]={eno0,eno1,eno2};
        const double rows[3][3]={{b00,b01,b02},{b10,b11,b12},{b20,b21,b22}};
        double solved[3], inverse_rows[3][3];
        solveSpd3x3(chol,rhs,solved);
        for (int j=0;j<3;++j) solveSpd3x3(chol,rows[j],inverse_rows[j]);
        out_eta[0]=eo0-(b00*solved[0]+b01*solved[1]+b02*solved[2]);
        out_eta[1]=eo1-(b10*solved[0]+b11*solved[1]+b12*solved[2]);
        out_eta[2]=eo2-(b20*solved[0]+b21*solved[1]+b22*solved[2]);
        int index=0;
        for (int col=0;col<3;++col) for(int row=col;row<3;++row) {
            out_lam6[index]=loo6[index]-(rows[row][0]*inverse_rows[col][0]+
                rows[row][1]*inverse_rows[col][1]+rows[row][2]*inverse_rows[col][2]);
            ++index;
        }
        return true;
    }
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (!(det > 0.0)) {
        return false;
    }
    const double inv_det = 1.0 / det;

    const double t00 = b00 * cof00 + b01 * cof10 + b02 * cof20;
    const double t01 = b00 * cof10 + b01 * cof11 + b02 * cof21;
    const double t02 = b00 * cof20 + b01 * cof21 + b02 * cof22;
    const double t10 = b10 * cof00 + b11 * cof10 + b12 * cof20;
    const double t11 = b10 * cof10 + b11 * cof11 + b12 * cof21;
    const double t12 = b10 * cof20 + b11 * cof21 + b12 * cof22;
    const double t20 = b20 * cof00 + b21 * cof10 + b22 * cof20;
    const double t21 = b20 * cof10 + b21 * cof11 + b22 * cof21;
    const double t22 = b20 * cof20 + b21 * cof21 + b22 * cof22;

    out_eta[0] = eo0 - inv_det * (t00 * eno0 + t01 * eno1 + t02 * eno2);
    out_eta[1] = eo1 - inv_det * (t10 * eno0 + t11 * eno1 + t12 * eno2);
    out_eta[2] = eo2 - inv_det * (t20 * eno0 + t21 * eno1 + t22 * eno2);

    out_lam6[0] = loo6[0] - inv_det * (t00 * b00 + t01 * b01 + t02 * b02);
    out_lam6[1] = loo6[1] - inv_det * (t10 * b00 + t11 * b01 + t12 * b02);
    out_lam6[2] = loo6[2] - inv_det * (t20 * b00 + t21 * b01 + t22 * b02);
    out_lam6[3] = loo6[3] - inv_det * (t10 * b10 + t11 * b11 + t12 * b12);
    out_lam6[4] = loo6[4] - inv_det * (t20 * b10 + t21 * b11 + t22 * b12);
    out_lam6[5] = loo6[5] - inv_det * (t20 * b20 + t21 * b21 + t22 * b22);
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void computeUnaryFactorAtNoDamping(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[idx];
    std::memcpy(unaryMsgEtaPtr(workspace, idx), factor.eta, 3 * sizeof(double));
    std::memcpy(unaryMsgLam6Ptr(workspace, idx), factor.lam6, 6 * sizeof(double));
}

PACKED_RESIDUAL_FORCEINLINE void relaxEta3(double* value, const double* old, double alpha) noexcept {
    if (alpha == 1.0) return;
    for (int j = 0; j < 3; ++j) value[j] = old[j] + alpha * (value[j] - old[j]);
}

PACKED_RESIDUAL_FORCEINLINE bool computeBinaryFactorAtNoDampingNoThrow(
    const SyntheticSE2PackedResidualBinaryFactor& data,
    const double* belief0_eta,
    const double* belief1_eta,
    const double* belief0_lam6,
    const double* belief1_lam6,
    const double* old0_eta,
    const double* old1_eta,
    const double* old0_lam6,
    const double* old1_lam6,
    int target,
    double* out_eta,
    double* out_lam6
) noexcept {
    const double jitter = packedResidualJitterValue();
    const double eno0_0 = data.eta1[0] + (belief1_eta[0] - old1_eta[0]);
    const double eno0_1 = data.eta1[1] + (belief1_eta[1] - old1_eta[1]);
    const double eno0_2 = data.eta1[2] + (belief1_eta[2] - old1_eta[2]);
    const double a0_00 = data.diag1_lam6[0] + (belief1_lam6[0] - old1_lam6[0]) + jitter;
    const double a0_10 = data.diag1_lam6[1] + (belief1_lam6[1] - old1_lam6[1]);
    const double a0_20 = data.diag1_lam6[2] + (belief1_lam6[2] - old1_lam6[2]);
    const double a0_11 = data.diag1_lam6[3] + (belief1_lam6[3] - old1_lam6[3]) + jitter;
    const double a0_21 = data.diag1_lam6[4] + (belief1_lam6[4] - old1_lam6[4]);
    const double a0_22 = data.diag1_lam6[5] + (belief1_lam6[5] - old1_lam6[5]) + jitter;

    const double eno1_0 = data.eta0[0] + (belief0_eta[0] - old0_eta[0]);
    const double eno1_1 = data.eta0[1] + (belief0_eta[1] - old0_eta[1]);
    const double eno1_2 = data.eta0[2] + (belief0_eta[2] - old0_eta[2]);
    const double a1_00 = data.diag0_lam6[0] + (belief0_lam6[0] - old0_lam6[0]) + jitter;
    const double a1_10 = data.diag0_lam6[1] + (belief0_lam6[1] - old0_lam6[1]);
    const double a1_20 = data.diag0_lam6[2] + (belief0_lam6[2] - old0_lam6[2]);
    const double a1_11 = data.diag0_lam6[3] + (belief0_lam6[3] - old0_lam6[3]) + jitter;
    const double a1_21 = data.diag0_lam6[4] + (belief0_lam6[4] - old0_lam6[4]);
    const double a1_22 = data.diag0_lam6[5] + (belief0_lam6[5] - old0_lam6[5]) + jitter;

    const double* cross = data.cross01_lam9;
    const double b00 = cross[0];
    const double b10 = cross[1];
    const double b20 = cross[2];
    const double b01 = cross[3];
    const double b11 = cross[4];
    const double b21 = cross[5];
    const double b02 = cross[6];
    const double b12 = cross[7];
    const double b22 = cross[8];

    if (target == 0) {
        if (!schurMessage3x3NoDampingAdjugatePackedSym(
                data.eta0[0], data.eta0[1], data.eta0[2],
                eno0_0, eno0_1, eno0_2,
                data.diag0_lam6,
                b00, b10, b20, b01, b11, b21, b02, b12, b22,
                a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
                out_eta, out_lam6
            )) {
            if (!schurMessage3x3NoDampingGeneralPackedSym(
                    data.eta0[0], data.eta0[1], data.eta0[2],
                    eno0_0, eno0_1, eno0_2,
                    data.diag0_lam6,
                    b00, b10, b20, b01, b11, b21, b02, b12, b22,
                    a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
                    out_eta, out_lam6
                )) {
                return false;
            }
        }
        return true;
    }

    if (!schurMessage3x3NoDampingAdjugatePackedSym(
            data.eta1[0], data.eta1[1], data.eta1[2],
            eno1_0, eno1_1, eno1_2,
            data.diag1_lam6,
            b00, b01, b02, b10, b11, b12, b20, b21, b22,
            a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
            out_eta, out_lam6
        )) {
        if (!schurMessage3x3NoDampingGeneralPackedSym(
                data.eta1[0], data.eta1[1], data.eta1[2],
                eno1_0, eno1_1, eno1_2,
                data.diag1_lam6,
                b00, b01, b02, b10, b11, b12, b20, b21, b22,
                a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
                out_eta, out_lam6
            )) {
            return false;
        }
    }
    return true;
}

PACKED_RESIDUAL_FORCEINLINE bool computeBinaryFactorAtNoDampingNoThrow(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    const double jitter = packedResidualJitterValue();
    const SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[idx];
    const int var0_id = data.var0_id;
    const int var1_id = data.var1_id;
    const double* eta0 = data.eta0;
    const double* eta1 = data.eta1;
    const double* diag0 = data.diag0_lam6;
    const double* diag1 = data.diag1_lam6;
    const double* cross = data.cross01_lam9;
    const double* belief0_eta = vec3Ptr(workspace.belief_eta, var0_id);
    const double* belief1_eta = vec3Ptr(workspace.belief_eta, var1_id);
    const double* belief0_lam6 = sym6Ptr(workspace.belief_lam6, var0_id);
    const double* belief1_lam6 = sym6Ptr(workspace.belief_lam6, var1_id);

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;

    const double* old0_eta = binaryMsgEtaPtr(workspace, slot0);
    const double* old1_eta = binaryMsgEtaPtr(workspace, slot1);
    const double* old0_lam6 = binaryMsgLam6Ptr(workspace, slot0);
    const double* old1_lam6 = binaryMsgLam6Ptr(workspace, slot1);
    double* out0_eta = binaryMsgEtaPtr(workspace, slot0);
    double* out1_eta = binaryMsgEtaPtr(workspace, slot1);
    double* out0_lam6 = binaryMsgLam6Ptr(workspace, slot0);
    double* out1_lam6 = binaryMsgLam6Ptr(workspace, slot1);
    const double old0_e0 = old0_eta[0];
    const double old0_e1 = old0_eta[1];
    const double old0_e2 = old0_eta[2];
    const double old1_e0 = old1_eta[0];
    const double old1_e1 = old1_eta[1];
    const double old1_e2 = old1_eta[2];
    const double old0_l0 = old0_lam6[0];
    const double old0_l1 = old0_lam6[1];
    const double old0_l2 = old0_lam6[2];
    const double old0_l3 = old0_lam6[3];
    const double old0_l4 = old0_lam6[4];
    const double old0_l5 = old0_lam6[5];
    const double old1_l0 = old1_lam6[0];
    const double old1_l1 = old1_lam6[1];
    const double old1_l2 = old1_lam6[2];
    const double old1_l3 = old1_lam6[3];
    const double old1_l4 = old1_lam6[4];
    const double old1_l5 = old1_lam6[5];

    const double eno0_0 = eta1[0] + (belief1_eta[0] - old1_e0);
    const double eno0_1 = eta1[1] + (belief1_eta[1] - old1_e1);
    const double eno0_2 = eta1[2] + (belief1_eta[2] - old1_e2);
    const double a0_00 = diag1[0] + (belief1_lam6[0] - old1_l0) + jitter;
    const double a0_10 = diag1[1] + (belief1_lam6[1] - old1_l1);
    const double a0_20 = diag1[2] + (belief1_lam6[2] - old1_l2);
    const double a0_11 = diag1[3] + (belief1_lam6[3] - old1_l3) + jitter;
    const double a0_21 = diag1[4] + (belief1_lam6[4] - old1_l4);
    const double a0_22 = diag1[5] + (belief1_lam6[5] - old1_l5) + jitter;

    const double eno1_0 = eta0[0] + (belief0_eta[0] - old0_e0);
    const double eno1_1 = eta0[1] + (belief0_eta[1] - old0_e1);
    const double eno1_2 = eta0[2] + (belief0_eta[2] - old0_e2);
    const double a1_00 = diag0[0] + (belief0_lam6[0] - old0_l0) + jitter;
    const double a1_10 = diag0[1] + (belief0_lam6[1] - old0_l1);
    const double a1_20 = diag0[2] + (belief0_lam6[2] - old0_l2);
    const double a1_11 = diag0[3] + (belief0_lam6[3] - old0_l3) + jitter;
    const double a1_21 = diag0[4] + (belief0_lam6[4] - old0_l4);
    const double a1_22 = diag0[5] + (belief0_lam6[5] - old0_l5) + jitter;

    const double b00 = cross[0];
    const double b10 = cross[1];
    const double b20 = cross[2];
    const double b01 = cross[3];
    const double b11 = cross[4];
    const double b21 = cross[5];
    const double b02 = cross[6];
    const double b12 = cross[7];
    const double b22 = cross[8];

    if (!schurMessage3x3NoDampingAdjugatePackedSym(
            eta0[0], eta0[1], eta0[2],
            eno0_0, eno0_1, eno0_2,
            diag0,
            b00, b10, b20, b01, b11, b21, b02, b12, b22,
            a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
            out0_eta, out0_lam6
        )) {
        if (!schurMessage3x3NoDampingGeneralPackedSym(
                eta0[0], eta0[1], eta0[2],
                eno0_0, eno0_1, eno0_2,
                diag0,
                b00, b10, b20, b01, b11, b21, b02, b12, b22,
                a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
                out0_eta, out0_lam6
            )) {
            return false;
        }
    }

    if (!schurMessage3x3NoDampingAdjugatePackedSym(
            eta1[0], eta1[1], eta1[2],
            eno1_0, eno1_1, eno1_2,
            diag1,
            b00, b01, b02, b10, b11, b12, b20, b21, b22,
            a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
            out1_eta, out1_lam6
        )) {
        if (!schurMessage3x3NoDampingGeneralPackedSym(
                eta1[0], eta1[1], eta1[2],
                eno1_0, eno1_1, eno1_2,
                diag1,
                b00, b01, b02, b10, b11, b12, b20, b21, b22,
                a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
                out1_eta, out1_lam6
            )) {
            return false;
        }
    }

    const double previous0[3] = {old0_e0, old0_e1, old0_e2};
    const double previous1[3] = {old1_e0, old1_e1, old1_e2};
    relaxEta3(out0_eta, previous0, workspace.eta_relaxation);
    relaxEta3(out1_eta, previous1, workspace.eta_relaxation);
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void computeBinaryFactorAtNoDamping(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) {
    if (!computeBinaryFactorAtNoDampingNoThrow(workspace, idx)) {
        throw std::runtime_error("Linear Schur solve failed in SyntheticSE2PackedResidual");
    }
}

PACKED_RESIDUAL_FORCEINLINE double* fixedEtaMap0Ptr(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    return workspace.fixed_eta_map0_lam9.data() + static_cast<size_t>(idx) * 9;
}

PACKED_RESIDUAL_FORCEINLINE double* fixedEtaMap1Ptr(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    return workspace.fixed_eta_map1_lam9.data() + static_cast<size_t>(idx) * 9;
}

PACKED_RESIDUAL_FORCEINLINE const double* fixedEtaMap0Ptr(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    return workspace.fixed_eta_map0_lam9.data() + static_cast<size_t>(idx) * 9;
}

PACKED_RESIDUAL_FORCEINLINE const double* fixedEtaMap1Ptr(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    return workspace.fixed_eta_map1_lam9.data() + static_cast<size_t>(idx) * 9;
}

PACKED_RESIDUAL_FORCEINLINE bool inverseSym3PackedLower(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double* PACKED_RESIDUAL_RESTRICT inv
) noexcept {
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (!(det > 0.0)) {
        return false;
    }
    const double inv_det = 1.0 / det;
    inv[0] = cof00 * inv_det;
    inv[1] = cof10 * inv_det;
    inv[2] = cof20 * inv_det;
    inv[3] = cof10 * inv_det;
    inv[4] = cof11 * inv_det;
    inv[5] = cof21 * inv_det;
    inv[6] = cof20 * inv_det;
    inv[7] = cof21 * inv_det;
    inv[8] = cof22 * inv_det;
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void rowMajorMatMul3(
    const double b00,
    const double b01,
    const double b02,
    const double b10,
    const double b11,
    const double b12,
    const double b20,
    const double b21,
    const double b22,
    const double* PACKED_RESIDUAL_RESTRICT inv,
    double* PACKED_RESIDUAL_RESTRICT map
) noexcept {
    map[0] = b00 * inv[0] + b01 * inv[3] + b02 * inv[6];
    map[1] = b00 * inv[1] + b01 * inv[4] + b02 * inv[7];
    map[2] = b00 * inv[2] + b01 * inv[5] + b02 * inv[8];
    map[3] = b10 * inv[0] + b11 * inv[3] + b12 * inv[6];
    map[4] = b10 * inv[1] + b11 * inv[4] + b12 * inv[7];
    map[5] = b10 * inv[2] + b11 * inv[5] + b12 * inv[8];
    map[6] = b20 * inv[0] + b21 * inv[3] + b22 * inv[6];
    map[7] = b20 * inv[1] + b21 * inv[4] + b22 * inv[7];
    map[8] = b20 * inv[2] + b21 * inv[5] + b22 * inv[8];
}

PACKED_RESIDUAL_FORCEINLINE bool buildFixedEtaMapsForFactorSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept;

PACKED_RESIDUAL_FORCEINLINE bool buildFixedEtaMapsForFactorSE2Buffered(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    const double* belief_lam6_base,
    const double* msg_lam6_base
) noexcept {
    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[idx];
    const int var0_id = data.var0_id;
    const int var1_id = data.var1_id;
    const double* diag0 = data.diag0_lam6;
    const double* diag1 = data.diag1_lam6;
    const double* c = data.cross01_lam9;
    const double* belief0_lam6 = belief_lam6_base + static_cast<size_t>(var0_id) * 6;
    const double* belief1_lam6 = belief_lam6_base + static_cast<size_t>(var1_id) * 6;
    const double* old0_lam6 = msg_lam6_base + static_cast<size_t>(slot0) * 6;
    const double* old1_lam6 = msg_lam6_base + static_cast<size_t>(slot1) * 6;
    const double jitter = packedResidualJitterValue();

    double inv0[9];
    double inv1[9];
    if (!inverseSym3PackedLower(
            diag1[0] + (belief1_lam6[0] - old1_lam6[0]) + jitter,
            diag1[1] + (belief1_lam6[1] - old1_lam6[1]),
            diag1[2] + (belief1_lam6[2] - old1_lam6[2]),
            diag1[3] + (belief1_lam6[3] - old1_lam6[3]) + jitter,
            diag1[4] + (belief1_lam6[4] - old1_lam6[4]),
            diag1[5] + (belief1_lam6[5] - old1_lam6[5]) + jitter,
            inv0
        )) {
        return false;
    }
    if (!inverseSym3PackedLower(
            diag0[0] + (belief0_lam6[0] - old0_lam6[0]) + jitter,
            diag0[1] + (belief0_lam6[1] - old0_lam6[1]),
            diag0[2] + (belief0_lam6[2] - old0_lam6[2]),
            diag0[3] + (belief0_lam6[3] - old0_lam6[3]) + jitter,
            diag0[4] + (belief0_lam6[4] - old0_lam6[4]),
            diag0[5] + (belief0_lam6[5] - old0_lam6[5]) + jitter,
            inv1
        )) {
        return false;
    }

    const double b00 = c[0];
    const double b10 = c[1];
    const double b20 = c[2];
    const double b01 = c[3];
    const double b11 = c[4];
    const double b21 = c[5];
    const double b02 = c[6];
    const double b12 = c[7];
    const double b22 = c[8];
    rowMajorMatMul3(
        b00, b01, b02,
        b10, b11, b12,
        b20, b21, b22,
        inv0,
        fixedEtaMap0Ptr(workspace, idx)
    );
    rowMajorMatMul3(
        b00, b10, b20,
        b01, b11, b21,
        b02, b12, b22,
        inv1,
        fixedEtaMap1Ptr(workspace, idx)
    );
    workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] = 1;
    return true;
}

PACKED_RESIDUAL_FORCEINLINE bool buildFixedEtaMapsForFactorSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx
) noexcept {
    return buildFixedEtaMapsForFactorSE2Buffered(
        workspace,
        idx,
        workspace.belief_lam6.data(),
        workspace.binary_msg_lam6.data()
    );
}

PACKED_RESIDUAL_FORCEINLINE void fixedEtaUpdate3(
    const double* PACKED_RESIDUAL_RESTRICT factor_eta_target,
    const double* PACKED_RESIDUAL_RESTRICT factor_eta_other,
    const double* PACKED_RESIDUAL_RESTRICT eta_map,
    const double* PACKED_RESIDUAL_RESTRICT belief_other_eta,
    const double* PACKED_RESIDUAL_RESTRICT old_other_eta,
    double* PACKED_RESIDUAL_RESTRICT out_eta
) noexcept {
    const double x0 = factor_eta_other[0] + belief_other_eta[0] - old_other_eta[0];
    const double x1 = factor_eta_other[1] + belief_other_eta[1] - old_other_eta[1];
    const double x2 = factor_eta_other[2] + belief_other_eta[2] - old_other_eta[2];
    out_eta[0] = factor_eta_target[0] - (eta_map[0] * x0 + eta_map[1] * x1 + eta_map[2] * x2);
    out_eta[1] = factor_eta_target[1] - (eta_map[3] * x0 + eta_map[4] * x1 + eta_map[5] * x2);
    out_eta[2] = factor_eta_target[2] - (eta_map[6] * x0 + eta_map[7] * x1 + eta_map[8] * x2);
}

PACKED_RESIDUAL_FORCEINLINE bool computeFixedEtaFactorSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    int* map_builds
) {
    if (workspace.fixed_lam_initialized[static_cast<size_t>(idx)] == 0) {
        return false;
    }
    if (workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] == 0) {
        if (!buildFixedEtaMapsForFactorSE2(workspace, idx)) {
            return false;
        }
        if (map_builds != nullptr) {
            ++(*map_builds);
        }
    }

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[idx];
    const int var0_id = data.var0_id;
    const int var1_id = data.var1_id;
    const double* eta0 = data.eta0;
    const double* eta1 = data.eta1;
    double* msg0 = binaryMsgEtaPtr(workspace, slot0);
    double* msg1 = binaryMsgEtaPtr(workspace, slot1);
    const double old0[3] = {msg0[0], msg0[1], msg0[2]};
    const double old1[3] = {msg1[0], msg1[1], msg1[2]};
    double out0[3];
    double out1[3];
    fixedEtaUpdate3(
        eta0,
        eta1,
        fixedEtaMap0Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, var1_id),
        old1,
        out0
    );
    fixedEtaUpdate3(
        eta1,
        eta0,
        fixedEtaMap1Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, var0_id),
        old0,
        out1
    );
    if (fixedEtaVerifyFullEnabled()) {
        double full0_eta[3];
        double full1_eta[3];
        double full0_lam6[6];
        double full1_lam6[6];
        const bool ok0 = computeBinaryFactorAtNoDampingNoThrow(
            data,
            vec3Ptr(workspace.belief_eta, var0_id),
            vec3Ptr(workspace.belief_eta, var1_id),
            sym6Ptr(workspace.belief_lam6, var0_id),
            sym6Ptr(workspace.belief_lam6, var1_id),
            old0,
            old1,
            binaryMsgLam6Ptr(workspace, slot0),
            binaryMsgLam6Ptr(workspace, slot1),
            0,
            full0_eta,
            full0_lam6
        );
        const bool ok1 = computeBinaryFactorAtNoDampingNoThrow(
            data,
            vec3Ptr(workspace.belief_eta, var0_id),
            vec3Ptr(workspace.belief_eta, var1_id),
            sym6Ptr(workspace.belief_lam6, var0_id),
            sym6Ptr(workspace.belief_lam6, var1_id),
            old0,
            old1,
            binaryMsgLam6Ptr(workspace, slot0),
            binaryMsgLam6Ptr(workspace, slot1),
            1,
            full1_eta,
            full1_lam6
        );
        if (!ok0 || !ok1 ||
            std::abs(full0_eta[0] - out0[0]) > 1e-7 ||
            std::abs(full0_eta[1] - out0[1]) > 1e-7 ||
            std::abs(full0_eta[2] - out0[2]) > 1e-7 ||
            std::abs(full1_eta[0] - out1[0]) > 1e-7 ||
            std::abs(full1_eta[1] - out1[1]) > 1e-7 ||
            std::abs(full1_eta[2] - out1[2]) > 1e-7) {
            return false;
        }
    }
    relaxEta3(out0, old0, workspace.eta_relaxation);
    relaxEta3(out1, old1, workspace.eta_relaxation);
    for (int i = 0; i < 3; ++i) {
        msg0[i] = out0[i];
        msg1[i] = out1[i];
    }
    return true;
}

PACKED_RESIDUAL_FORCEINLINE bool computeFixedEtaFactorSerialDeltaSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    int* map_builds
) {
    if (workspace.fixed_lam_initialized[static_cast<size_t>(idx)] == 0) {
        return false;
    }
    if (workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] == 0) {
        if (!buildFixedEtaMapsForFactorSE2(workspace, idx)) {
            return false;
        }
        if (map_builds != nullptr) {
            ++(*map_builds);
        }
    }

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[idx];
    const int var0_id = data.var0_id;
    const int var1_id = data.var1_id;
    const double* eta0 = data.eta0;
    const double* eta1 = data.eta1;
    double* msg0 = binaryMsgEtaPtr(workspace, slot0);
    double* msg1 = binaryMsgEtaPtr(workspace, slot1);
    const double old0[3] = {msg0[0], msg0[1], msg0[2]};
    const double old1[3] = {msg1[0], msg1[1], msg1[2]};
    double out0[3];
    double out1[3];
    fixedEtaUpdate3(
        eta0,
        eta1,
        fixedEtaMap0Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, var1_id),
        old1,
        out0
    );
    fixedEtaUpdate3(
        eta1,
        eta0,
        fixedEtaMap1Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, var0_id),
        old0,
        out1
    );

    relaxEta3(out0, old0, workspace.eta_relaxation);
    relaxEta3(out1, old1, workspace.eta_relaxation);
    double* delta0 = vec3Ptr(workspace.serial_belief_delta_eta, var0_id);
    double* delta1 = vec3Ptr(workspace.serial_belief_delta_eta, var1_id);
    for (int i = 0; i < 3; ++i) {
        delta0[i] += out0[i] - old0[i];
        delta1[i] += out1[i] - old1[i];
        msg0[i] = out0[i];
        msg1[i] = out1[i];
    }
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void atomicAddDelta3(
    double* PACKED_RESIDUAL_RESTRICT dst,
    double d0,
    double d1,
    double d2
) noexcept {
    #pragma omp atomic
    dst[0] += d0;
    #pragma omp atomic
    dst[1] += d1;
    #pragma omp atomic
    dst[2] += d2;
}

PACKED_RESIDUAL_FORCEINLINE bool computeFixedEtaFactorParallelDeltaSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int idx,
    int* map_builds
) {
    if (workspace.fixed_lam_initialized[static_cast<size_t>(idx)] == 0) {
        return false;
    }
    if (workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] == 0) {
        if (!buildFixedEtaMapsForFactorSE2(workspace, idx)) {
            return false;
        }
        if (map_builds != nullptr) {
            ++(*map_builds);
        }
    }

    const SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[idx];
    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    double* msg0 = binaryMsgEtaPtr(workspace, slot0);
    double* msg1 = binaryMsgEtaPtr(workspace, slot1);
    const double old0[3] = {msg0[0], msg0[1], msg0[2]};
    const double old1[3] = {msg1[0], msg1[1], msg1[2]};
    double out0[3];
    double out1[3];
    fixedEtaUpdate3(
        data.eta0,
        data.eta1,
        fixedEtaMap0Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, data.var1_id),
        old1,
        out0
    );
    fixedEtaUpdate3(
        data.eta1,
        data.eta0,
        fixedEtaMap1Ptr(workspace, idx),
        vec3Ptr(workspace.belief_eta, data.var0_id),
        old0,
        out1
    );

    relaxEta3(out0, old0, workspace.eta_relaxation);
    relaxEta3(out1, old1, workspace.eta_relaxation);
    atomicAddDelta3(
        vec3Ptr(workspace.serial_belief_delta_eta, data.var0_id),
        out0[0] - old0[0],
        out0[1] - old0[1],
        out0[2] - old0[2]
    );
    atomicAddDelta3(
        vec3Ptr(workspace.serial_belief_delta_eta, data.var1_id),
        out1[0] - old1[0],
        out1[1] - old1[1],
        out1[2] - old1[2]
    );
    msg0[0] = out0[0];
    msg0[1] = out0[1];
    msg0[2] = out0[2];
    msg1[0] = out1[0];
    msg1[1] = out1[1];
    msg1[2] = out1[2];
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void applyParallelBeliefEtaDeltaAndResetSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept {
    double* belief_eta = vec3Ptr(workspace.belief_eta, var_idx);
    double* delta_eta = vec3Ptr(workspace.serial_belief_delta_eta, var_idx);
    belief_eta[0] += delta_eta[0];
    belief_eta[1] += delta_eta[1];
    belief_eta[2] += delta_eta[2];
    delta_eta[0] = 0.0;
    delta_eta[1] = 0.0;
    delta_eta[2] = 0.0;
    workspace.mu_valid[var_idx] = 0;
}

PACKED_RESIDUAL_FORCEINLINE void updateBelief3DLocalNoMu(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept {
    const double* prior_eta = vec3Ptr(workspace.prior_eta, var_idx);
    const double* prior_lam6 = sym6Ptr(workspace.prior_lam6, var_idx);

    double eta0 = prior_eta[0];
    double eta1 = prior_eta[1];
    double eta2 = prior_eta[2];
    double lam00 = prior_lam6[0];
    double lam10 = prior_lam6[1];
    double lam20 = prior_lam6[2];
    double lam11 = prior_lam6[3];
    double lam21 = prior_lam6[4];
    double lam22 = prior_lam6[5];

    const int unary_begin = workspace.unary_offsets[var_idx];
    const int unary_end = workspace.unary_offsets[var_idx + 1];
    for (int i = unary_begin; i < unary_end; ++i) {
        const int unary_id = workspace.unary_ids[i];
        const double* msg_eta = unaryMsgEtaPtr(workspace, unary_id);
        const double* msg_lam6 = unaryMsgLam6Ptr(workspace, unary_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
        lam00 += msg_lam6[0];
        lam10 += msg_lam6[1];
        lam20 += msg_lam6[2];
        lam11 += msg_lam6[3];
        lam21 += msg_lam6[4];
        lam22 += msg_lam6[5];
    }

    const int binary_begin = workspace.binary_offsets[var_idx];
    const int binary_end = workspace.binary_offsets[var_idx + 1];
    for (int i = binary_begin; i < binary_end; ++i) {
        const int slot_id = workspace.binary_slot_ids[i];
        const double* msg_eta = binaryMsgEtaPtr(workspace, slot_id);
        const double* msg_lam6 = binaryMsgLam6Ptr(workspace, slot_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
        lam00 += msg_lam6[0];
        lam10 += msg_lam6[1];
        lam20 += msg_lam6[2];
        lam11 += msg_lam6[3];
        lam21 += msg_lam6[4];
        lam22 += msg_lam6[5];
    }

    double* belief_eta = vec3Ptr(workspace.belief_eta, var_idx);
    double* belief_lam6 = sym6Ptr(workspace.belief_lam6, var_idx);
    belief_eta[0] = eta0;
    belief_eta[1] = eta1;
    belief_eta[2] = eta2;
    belief_lam6[0] = lam00;
    belief_lam6[1] = lam10;
    belief_lam6[2] = lam20;
    belief_lam6[3] = lam11;
    belief_lam6[4] = lam21;
    belief_lam6[5] = lam22;
    workspace.mu_valid[var_idx] = 0;
}

PACKED_RESIDUAL_FORCEINLINE void updateBelief3DLocalEtaOnlyNoMu(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept {
    const double* prior_eta = vec3Ptr(workspace.prior_eta, var_idx);
    double eta0 = prior_eta[0];
    double eta1 = prior_eta[1];
    double eta2 = prior_eta[2];

    const int unary_begin = workspace.unary_offsets[var_idx];
    const int unary_end = workspace.unary_offsets[var_idx + 1];
    for (int i = unary_begin; i < unary_end; ++i) {
        const int unary_id = workspace.unary_ids[i];
        const double* msg_eta = unaryMsgEtaPtr(workspace, unary_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
    }

    const int binary_begin = workspace.binary_offsets[var_idx];
    const int binary_end = workspace.binary_offsets[var_idx + 1];
    for (int i = binary_begin; i < binary_end; ++i) {
        const int slot_id = workspace.binary_slot_ids[i];
        const double* msg_eta = binaryMsgEtaPtr(workspace, slot_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
    }

    double* belief_eta = vec3Ptr(workspace.belief_eta, var_idx);
    belief_eta[0] = eta0;
    belief_eta[1] = eta1;
    belief_eta[2] = eta2;
    workspace.mu_valid[var_idx] = 0;
}

PACKED_RESIDUAL_FORCEINLINE void applySerialBeliefEtaDeltaSE2(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int var_idx
) noexcept {
    double* belief_eta = vec3Ptr(workspace.belief_eta, var_idx);
    double* delta_eta = vec3Ptr(workspace.serial_belief_delta_eta, var_idx);
    belief_eta[0] += delta_eta[0];
    belief_eta[1] += delta_eta[1];
    belief_eta[2] += delta_eta[2];
    workspace.mu_valid[var_idx] = 0;
}

PACKED_RESIDUAL_FORCEINLINE bool refreshMuAtNoThrow(SyntheticSE2PackedResidualWorkspace& workspace, int var_idx) noexcept {
    if (workspace.mu_valid[var_idx]) {
        return true;
    }
    const double* belief_eta = vec3Ptr(workspace.belief_eta, var_idx);
    const double* belief_lam6 = sym6Ptr(workspace.belief_lam6, var_idx);
    Cholesky3x3 chol;
    if (!factorizeSpd3x3Lower(
            belief_lam6[0], belief_lam6[1], belief_lam6[2],
            belief_lam6[3], belief_lam6[4], belief_lam6[5],
            chol
        )) {
        if (!solveGeneral3x3LowerSym(
                belief_lam6[0], belief_lam6[1], belief_lam6[2],
                belief_lam6[3], belief_lam6[4], belief_lam6[5],
                belief_eta,
                vec3Ptr(workspace.mu, var_idx)
            )) {
            return false;
        }
        workspace.mu_valid[var_idx] = 1;
        return true;
    }
    solveSpd3x3(chol, belief_eta, vec3Ptr(workspace.mu, var_idx));
    workspace.mu_valid[var_idx] = 1;
    return true;
}

PACKED_RESIDUAL_FORCEINLINE void refreshMuAt(SyntheticSE2PackedResidualWorkspace& workspace, int var_idx) {
    if (!refreshMuAtNoThrow(workspace, var_idx)) {
        throw std::runtime_error("Linear solve failed in SyntheticSE2PackedResidual refreshMuAt");
    }
}

inline void packSymLam3x3(const Eigen::Matrix3d& src, double* dst6) noexcept {
    dst6[0] = src(0, 0);
    dst6[1] = src(1, 0);
    dst6[2] = src(2, 0);
    dst6[3] = src(1, 1);
    dst6[4] = src(2, 1);
    dst6[5] = src(2, 2);
}

inline void copy3x3BlockPackedSym(const Eigen::Matrix<double, 6, 6>& src, int row0, int col0, double* dst6) noexcept {
    dst6[0] = src(row0 + 0, col0 + 0);
    dst6[1] = src(row0 + 1, col0 + 0);
    dst6[2] = src(row0 + 2, col0 + 0);
    dst6[3] = src(row0 + 1, col0 + 1);
    dst6[4] = src(row0 + 2, col0 + 1);
    dst6[5] = src(row0 + 2, col0 + 2);
}

inline void copy3x3BlockColMajor(const Eigen::Matrix<double, 6, 6>& src, int row0, int col0, double* dst9) noexcept {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            dst9[c * 3 + r] = src(row0 + r, col0 + c);
        }
    }
}

SyntheticSE2PackedResidualWorkspace buildSyntheticSE2PackedResidualWorkspace(
    const SyntheticSE2Problem& problem,
    double tiny_prior
) {
    SyntheticSE2PackedResidualWorkspace workspace;
    workspace.lift_correction_to_messages=getenvInt("HGBP_SE2_MESSAGE_LIFT",0)!=0;
    workspace.message_initialization=getenvInt("HGBP_SE2_MESSAGE_INITIALIZATION",0);
    if (const char* alpha = std::getenv("HGBP_SE2_ETA_RELAXATION")) {
        workspace.eta_relaxation = std::stod(alpha);
        if (!(workspace.eta_relaxation > 0.0 && workspace.eta_relaxation <= 1.0))
            throw std::runtime_error("SE2 eta relaxation must be in (0, 1]");
    }
    workspace.num_vars = static_cast<int>(problem.gt_poses.size());
    workspace.tiny_prior = tiny_prior;

    const int n = workspace.num_vars;
    workspace.prior_eta.resize(static_cast<size_t>(n) * 3, 0.0);
    workspace.prior_lam6.resize(static_cast<size_t>(n) * 6, 0.0);
    workspace.belief_eta.resize(static_cast<size_t>(n) * 3, 0.0);
    workspace.belief_lam6.resize(static_cast<size_t>(n) * 6, 0.0);
    workspace.belief_eta_alt.resize(static_cast<size_t>(n) * 3, 0.0);
    workspace.belief_lam6_alt.resize(static_cast<size_t>(n) * 6, 0.0);
    workspace.mu.resize(static_cast<size_t>(n) * 3, 0.0);
    workspace.mu_valid.assign(n, 0);
    for (int i = 0; i < n; ++i) {
        double* lam6 = sym6Ptr(workspace.prior_lam6, i);
        lam6[0] = tiny_prior;
        lam6[1] = 0.0;
        lam6[2] = 0.0;
        lam6[3] = tiny_prior;
        lam6[4] = 0.0;
        lam6[5] = tiny_prior;
        std::memcpy(sym6Ptr(workspace.belief_lam6, i), lam6, 6 * sizeof(double));
    }

    workspace.unary_factors.resize(1);
    workspace.unary_factors[0].var_id = 0;
    workspace.unary_msg_eta.resize(3, 0.0);
    workspace.unary_msg_lam6.resize(6, 0.0);

    workspace.binary_factors.resize(problem.edges.size());
    workspace.adaptive_precision=getenvInt("HGBP_SE2_ADAPTIVE_PRECISION",0)!=0;
    workspace.binary_msg_eta.resize(problem.edges.size() * 2 * 3, 0.0);
    workspace.binary_msg_lam6.resize(problem.edges.size() * 2 * 6, 0.0);
    workspace.binary_msg_eta_alt.resize(problem.edges.size() * 2 * 3, 0.0);
    workspace.binary_msg_lam6_alt.resize(problem.edges.size() * 2 * 6, 0.0);
    workspace.fixed_eta_map0_lam9.resize(problem.edges.size() * 9, 0.0);
    workspace.fixed_eta_map1_lam9.resize(problem.edges.size() * 9, 0.0);
    workspace.fixed_lam_initialized.assign(problem.edges.size(), 0);
    workspace.fixed_eta_map_valid.assign(problem.edges.size(), 0);
    workspace.serial_belief_delta_eta.resize(static_cast<size_t>(n) * 3, 0.0);

    std::vector<int> unary_degree(n, 0);
    std::vector<int> binary_degree(n, 0);
    unary_degree[0] = 1;
    for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
        const SyntheticSE2Edge& edge = problem.edges[edge_index];
        workspace.binary_factors[edge_index].var0_id = edge.i;
        workspace.binary_factors[edge_index].var1_id = edge.j;
        ++binary_degree[edge.i];
        ++binary_degree[edge.j];
    }

    workspace.unary_offsets.resize(n + 1, 0);
    workspace.binary_offsets.resize(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        workspace.unary_offsets[i + 1] = workspace.unary_offsets[i] + unary_degree[i];
        workspace.binary_offsets[i + 1] = workspace.binary_offsets[i] + binary_degree[i];
    }
    workspace.unary_ids.resize(workspace.unary_offsets.back(), 0);
    workspace.binary_slot_ids.resize(workspace.binary_offsets.back(), 0);

    std::vector<int> unary_cursor = workspace.unary_offsets;
    std::vector<int> binary_cursor = workspace.binary_offsets;
    workspace.unary_ids[unary_cursor[0]++] = 0;
    for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
        const SyntheticSE2Edge& edge = problem.edges[edge_index];
        workspace.binary_slot_ids[binary_cursor[edge.i]++] = 2 * edge_index + 0;
        workspace.binary_slot_ids[binary_cursor[edge.j]++] = 2 * edge_index + 1;
    }

    return workspace;
}

void relinearizeSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    const RobustLossConfig& robust_loss_config,
    int num_threads
) {
    if (workspace.num_vars != static_cast<int>(problem.gt_poses.size()) ||
        workspace.binary_factors.size() != problem.edges.size()) {
        throw std::runtime_error("Packed residual workspace topology does not match problem");
    }

    const int n = workspace.num_vars;
    const int thread_count = effectiveThreadCount(num_threads);
    workspace.jacobi_ready = false;
    std::fill(workspace.prior_eta.begin(), workspace.prior_eta.end(), 0.0);
    const bool do_parallel = thread_count > 1 && (n >= 128 || problem.edges.size() >= 128);
    if (do_parallel) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            std::memset(vec3Ptr(workspace.belief_eta, i), 0, 3 * sizeof(double));
            std::memcpy(sym6Ptr(workspace.belief_lam6, i), sym6Ptr(workspace.prior_lam6, i), 6 * sizeof(double));
            std::memset(vec3Ptr(workspace.belief_eta_alt, i), 0, 3 * sizeof(double));
            std::memcpy(sym6Ptr(workspace.belief_lam6_alt, i), sym6Ptr(workspace.prior_lam6, i), 6 * sizeof(double));
            std::memset(vec3Ptr(workspace.mu, i), 0, 3 * sizeof(double));
            workspace.mu_valid[i] = 0;
        }
    } else {
        for (int i = 0; i < n; ++i) {
            std::memset(vec3Ptr(workspace.belief_eta, i), 0, 3 * sizeof(double));
            std::memcpy(sym6Ptr(workspace.belief_lam6, i), sym6Ptr(workspace.prior_lam6, i), 6 * sizeof(double));
            std::memset(vec3Ptr(workspace.belief_eta_alt, i), 0, 3 * sizeof(double));
            std::memcpy(sym6Ptr(workspace.belief_lam6_alt, i), sym6Ptr(workspace.prior_lam6, i), 6 * sizeof(double));
            std::memset(vec3Ptr(workspace.mu, i), 0, 3 * sizeof(double));
            workspace.mu_valid[i] = 0;
        }
    }
    std::fill(workspace.unary_msg_eta.begin(), workspace.unary_msg_eta.end(), 0.0);
    std::fill(workspace.unary_msg_lam6.begin(), workspace.unary_msg_lam6.end(), 0.0);
    std::fill(workspace.binary_msg_eta.begin(), workspace.binary_msg_eta.end(), 0.0);
    std::fill(workspace.binary_msg_lam6.begin(), workspace.binary_msg_lam6.end(), 0.0);
    std::fill(workspace.binary_msg_eta_alt.begin(), workspace.binary_msg_eta_alt.end(), 0.0);
    std::fill(workspace.binary_msg_lam6_alt.begin(), workspace.binary_msg_lam6_alt.end(), 0.0);
    std::fill(workspace.fixed_lam_initialized.begin(), workspace.fixed_lam_initialized.end(), 0);
    std::fill(workspace.fixed_eta_map_valid.begin(), workspace.fixed_eta_map_valid.end(), 0);
    workspace.fixed_eta_maps_all_valid = 0;
    std::fill(workspace.serial_belief_delta_eta.begin(), workspace.serial_belief_delta_eta.end(), 0.0);
    workspace.sweeps_since_relinearize = 0;
    workspace.precision_frozen=false;
    workspace.precision_stable_checks=0;
    workspace.precision_checks=workspace.precision_freezes=workspace.precision_thaws=0;
    workspace.first_precision_freeze_sweep=-1;
    workspace.last_precision_check_sweep=0;
    workspace.precision_check_interval=SE2PrecisionPolicy::check_period;
    workspace.next_precision_check_sweep=SE2PrecisionPolicy::check_period;
    workspace.last_precision_residual=0;
    workspace.precision_scales_ready=false;
    workspace.full_precision_sweeps=workspace.eta_only_sweeps=0;
    workspace.belief_eta_message_consistent = 0;

    {
        const Eigen::Vector3d& base_anchor = base_poses.at(0);
        const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(problem.anchor_pose), base_anchor));
        const Eigen::Matrix3d J = analyticAnchorResidualJacobian(base_anchor, problem.anchor_pose);
        const Eigen::Matrix3d lam = J.transpose() * problem.anchor_information * J;
        const Eigen::Vector3d eta = -J.transpose() * problem.anchor_information * err0;
        SyntheticSE2PackedResidualUnaryFactor& anchor = workspace.unary_factors[0];
        std::memcpy(anchor.eta, eta.data(), 3 * sizeof(double));
        packSymLam3x3(lam, anchor.lam6);
        std::memcpy(unaryMsgEtaPtr(workspace, 0), anchor.eta, 3 * sizeof(double));
        std::memcpy(unaryMsgLam6Ptr(workspace, 0), anchor.lam6, 6 * sizeof(double));
    }

    if (do_parallel) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;

            SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[edge_index];
            data.eta0[0] = eta(0);
            data.eta0[1] = eta(1);
            data.eta0[2] = eta(2);
            data.eta1[0] = eta(3);
            data.eta1[1] = eta(4);
            data.eta1[2] = eta(5);
            copy3x3BlockPackedSym(lam, 0, 0, data.diag0_lam6);
            copy3x3BlockPackedSym(lam, 3, 3, data.diag1_lam6);
            copy3x3BlockColMajor(lam, 0, 3, data.cross01_lam9);
        }
    } else {
        for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;

            SyntheticSE2PackedResidualBinaryFactor& data = workspace.binary_factors[edge_index];
            data.eta0[0] = eta(0);
            data.eta0[1] = eta(1);
            data.eta0[2] = eta(2);
            data.eta1[0] = eta(3);
            data.eta1[1] = eta(4);
            data.eta1[2] = eta(5);
            copy3x3BlockPackedSym(lam, 0, 0, data.diag0_lam6);
            copy3x3BlockPackedSym(lam, 3, 3, data.diag1_lam6);
            copy3x3BlockColMajor(lam, 0, 3, data.cross01_lam9);
        }
    }
}

// Keep diagnostic temporaries out of the original fast-math Schur kernel.
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
bool computeBinaryFactorWithPrecisionCheckSE2(SyntheticSE2PackedResidualWorkspace& w,int idx) noexcept {
    double previous0[6],previous1[6];
    std::memcpy(previous0,binaryMsgLam6Ptr(w,2*idx),6*sizeof(double));
    std::memcpy(previous1,binaryMsgLam6Ptr(w,2*idx+1),6*sizeof(double));
    if(!computeBinaryFactorAtNoDampingNoThrow(w,idx)) return false;
    const double* scales=w.precision_scales.data()+12*idx;
    w.precision_residuals[idx]=std::max(
        scaledSE2PrecisionResidualSquared(binaryMsgLam6Ptr(w,2*idx),previous0,scales),
        scaledSE2PrecisionResidualSquared(binaryMsgLam6Ptr(w,2*idx+1),previous1,scales+6));
    return true;
}

bool checkSE2PrecisionThisSweep(const SyntheticSE2PackedResidualWorkspace& w) noexcept {
    return w.adaptive_precision && w.sweeps_since_relinearize+1>=w.next_precision_check_sweep;
}

void completeSE2PrecisionSweep(SyntheticSE2PackedResidualWorkspace& w,bool eta_only,bool check) noexcept {
    if(eta_only) ++w.eta_only_sweeps; else ++w.full_precision_sweeps;
    if(!check) return;
    double residual=0;
    for(double value:w.precision_residuals) residual=std::max(residual,value);
    residual=std::sqrt(residual);
    w.last_precision_residual=residual;
    w.last_precision_check_sweep=w.sweeps_since_relinearize;
    ++w.precision_checks;
    if(std::isfinite(residual) && residual<=SE2PrecisionPolicy::tolerance) {
        ++w.precision_stable_checks;
        if(!w.precision_frozen && w.precision_stable_checks>=SE2PrecisionPolicy::stable_checks) {
            w.precision_frozen=true;
            ++w.precision_freezes;
            if(w.first_precision_freeze_sweep<0) w.first_precision_freeze_sweep=w.sweeps_since_relinearize;
        }
    } else {
        if(w.precision_frozen) ++w.precision_thaws;
        w.precision_frozen=false;
        w.precision_stable_checks=0;
    }
    // Failed checks only delay testing, never skip full GBP. A successful check
    // returns to the dense schedule, and freezing still requires three passes.
    w.precision_check_interval=w.precision_frozen ? SE2PrecisionPolicy::frozen_check_period
        : w.precision_stable_checks>0 ? SE2PrecisionPolicy::check_period
        : std::min(2*w.precision_check_interval,SE2PrecisionPolicy::max_check_period);
    w.next_precision_check_sweep=w.sweeps_since_relinearize+w.precision_check_interval;
}

void blockJacobiSE2PackedIterations(SyntheticSE2PackedResidualWorkspace& w, int sweeps, int threads) {
    if(sweeps<=0) return;
    const int n=w.num_vars, count=effectiveThreadCount(threads);
    if(!w.jacobi_ready) {
        w.jacobi_diagonal.resize(6*n); w.jacobi_inverse.resize(9*n); w.jacobi_rhs.resize(3*n);
        w.jacobi_x.resize(3*n); w.jacobi_alt.resize(3*n);
        for(int v=0;v<n;++v) {
            double* diag=w.jacobi_diagonal.data()+6*v;
            double* rhs=w.jacobi_rhs.data()+3*v;
            std::memcpy(diag,w.prior_lam6.data()+6*v,6*sizeof(double));
            std::memcpy(rhs,w.prior_eta.data()+3*v,3*sizeof(double));
            for(int p=w.unary_offsets[v];p<w.unary_offsets[v+1];++p) {
                const auto& f=w.unary_factors[w.unary_ids[p]];
                for(int j=0;j<6;++j) diag[j]+=f.lam6[j];
                for(int j=0;j<3;++j) rhs[j]+=f.eta[j];
            }
            for(int p=w.binary_offsets[v];p<w.binary_offsets[v+1];++p) {
                const int slot=w.binary_slot_ids[p]; const auto& f=w.binary_factors[slot/2];
                const double* a=slot%2?f.diag1_lam6:f.diag0_lam6;
                const double* b=slot%2?f.eta1:f.eta0;
                for(int j=0;j<6;++j) diag[j]+=a[j];
                for(int j=0;j<3;++j) rhs[j]+=b[j];
            }
            Cholesky3x3 chol;
            if(!factorizeSpd3x3Lower(diag[0],diag[1],diag[2],diag[3],diag[4],diag[5],chol))
                throw std::runtime_error("Jacobi reference diagonal is not SPD");
            for(int j=0;j<3;++j) {
                double unit[3]={0,0,0}; unit[j]=1;
                solveSpd3x3(chol,unit,w.jacobi_inverse.data()+9*v+3*j);
            }
        }
        w.jacobi_ready=true;
    }
    for(int v=0;v<n;++v) {
        refreshMuAt(w,v);
        std::memcpy(w.jacobi_x.data()+3*v,w.mu.data()+3*v,3*sizeof(double));
    }
    // H <= 2*blockdiag(H) for PSD binary factors. A common omega=2/3
    // is stable, without dataset-specific spectral estimates or clipping.
    #pragma omp parallel num_threads(count) if(count>1 && n>=128)
    {
        for(int sweep=0;sweep<sweeps;++sweep) {
            const double* x=(sweep%2?w.jacobi_alt:w.jacobi_x).data();
            double* y=(sweep%2?w.jacobi_x:w.jacobi_alt).data();
            #pragma omp for schedule(static)
            for(int v=0;v<n;++v) {
                double product[3];
                beliefEtaFromLamMu(w.jacobi_diagonal.data()+6*v,x+3*v,product);
                for(int p=w.binary_offsets[v];p<w.binary_offsets[v+1];++p) {
                    const int slot=w.binary_slot_ids[p]; const auto& f=w.binary_factors[slot/2];
                    const double* other=x+3*(slot%2?f.var0_id:f.var1_id);
                    for(int r=0;r<3;++r) for(int c=0;c<3;++c)
                        product[r]+=f.cross01_lam9[slot%2?c+3*r:r+3*c]*other[c];
                }
                double residual[3];
                for(int j=0;j<3;++j) residual[j]=w.jacobi_rhs[3*v+j]-product[j];
                const double* inv=w.jacobi_inverse.data()+9*v;
                for(int r=0;r<3;++r)
                    y[3*v+r]=x[3*v+r]+(2./3.)*(inv[r]*residual[0]+inv[r+3]*residual[1]+inv[r+6]*residual[2]);
            }
        }
    }
    const auto& x=sweeps%2?w.jacobi_alt:w.jacobi_x;
    for(int v=0;v<n;++v) {
        std::memcpy(w.mu.data()+3*v,x.data()+3*v,3*sizeof(double));
        beliefEtaFromLamMu(w.belief_lam6.data()+6*v,x.data()+3*v,w.belief_eta.data()+3*v);
        w.mu_valid[v]=1;
    }
    w.belief_eta_message_consistent=0;
}

void initializeSE2PackedMessagesFromFactors(SyntheticSE2PackedResidualWorkspace& w, int mode, int threads) {
    if (mode<0 || mode>2) throw std::runtime_error("Unknown SE2 message initialization");
    if (mode==0) return;
    const int count=effectiveThreadCount(threads);
    #pragma omp parallel for schedule(static) num_threads(count) if(count>1 && w.num_vars>=128)
    for(int f=0;f<static_cast<int>(w.binary_factors.size());++f) {
        const auto& data=w.binary_factors[f];
        std::memcpy(binaryMsgEtaPtr(w,2*f),data.eta0,3*sizeof(double));
        std::memcpy(binaryMsgEtaPtr(w,2*f+1),data.eta1,3*sizeof(double));
        if(mode==2) {
            std::memcpy(binaryMsgLam6Ptr(w,2*f),data.diag0_lam6,6*sizeof(double));
            std::memcpy(binaryMsgLam6Ptr(w,2*f+1),data.diag1_lam6,6*sizeof(double));
        }
    }
    #pragma omp parallel for schedule(static) num_threads(count) if(count>1 && w.num_vars>=128)
    for(int v=0;v<w.num_vars;++v) updateBelief3DLocalNoMu(w,v);
    w.belief_eta_message_consistent=1;
}

void shiftSE2PackedCorrectionReference(SyntheticSE2PackedResidualWorkspace& w,
                                      const Eigen::VectorXd& delta) {
    if (delta.size()!=3*w.num_vars) throw std::runtime_error("SE2 correction reference dimension mismatch");
    // eta_f <- eta_f - H_f * delta preserves the original factorization of the
    // residual RHS. Moving all of it to near-zero unary priors has different
    // finite-iteration BP behavior despite the same assembled linear system.
    double product[3];
    for(int i=0;i<w.num_vars;++i) {
        beliefEtaFromLamMu(sym6Ptr(w.prior_lam6,i),delta.data()+3*i,product);
        for(int d=0;d<3;++d) w.prior_eta[3*i+d]-=product[d];
    }
    for(size_t i=0;i<w.unary_factors.size();++i) {
        auto& f=w.unary_factors[i];
        beliefEtaFromLamMu(f.lam6,delta.data()+3*f.var_id,product);
        for(int d=0;d<3;++d) {
            f.eta[d]-=product[d];
            w.unary_msg_eta[3*i+d]=f.eta[d];
        }
    }
    for(auto& f:w.binary_factors) {
        const double* d0=delta.data()+3*f.var0_id;
        const double* d1=delta.data()+3*f.var1_id;
        beliefEtaFromLamMu(f.diag0_lam6,d0,product);
        for(int r=0;r<3;++r) {
            double cross=0;
            for(int c=0;c<3;++c) cross+=f.cross01_lam9[r+3*c]*d1[c];
            f.eta0[r]-=product[r]+cross;
        }
        beliefEtaFromLamMu(f.diag1_lam6,d1,product);
        for(int r=0;r<3;++r) {
            double cross=0;
            for(int c=0;c<3;++c) cross+=f.cross01_lam9[c+3*r]*d0[c];
            f.eta1[r]-=product[r]+cross;
        }
    }
    w.belief_eta=w.prior_eta;
    for(const auto& f:w.unary_factors)
        for(int d=0;d<3;++d) w.belief_eta[3*f.var_id+d]+=f.eta[d];
    w.belief_eta_alt=w.belief_eta;
    std::fill(w.binary_msg_eta.begin(),w.binary_msg_eta.end(),0.0);
    std::fill(w.binary_msg_eta_alt.begin(),w.binary_msg_eta_alt.end(),0.0);
    std::fill(w.mu_valid.begin(),w.mu_valid.end(),0);
    w.belief_eta_message_consistent=1;
}

void fineResidualSyntheticSE2PackedInto(
    const SyntheticSE2PackedResidualWorkspace& w,
    const Eigen::VectorXd& x, Eigen::VectorXd& residual, int threads
) {
    if (x.size() != 3*w.num_vars) throw std::runtime_error("SE2 residual dimension mismatch");
    residual.resize(x.size());
    auto row = [&](int i) {
        const double* local = x.data()+3*i;
        double product[3];
        beliefEtaFromLamMu(sym6Ptr(w.prior_lam6,i),local,product);
        double* out = residual.data()+3*i;
        for (int d=0;d<3;++d) out[d] = w.prior_eta[3*i+d]-product[d];
        for (int p=w.unary_offsets[i];p<w.unary_offsets[i+1];++p) {
            const auto& f = w.unary_factors[w.unary_ids[p]];
            beliefEtaFromLamMu(f.lam6,local,product);
            for (int d=0;d<3;++d) out[d] += f.eta[d]-product[d];
        }
        for (int p=w.binary_offsets[i];p<w.binary_offsets[i+1];++p) {
            const int slot = w.binary_slot_ids[p];
            const auto& f = w.binary_factors[slot/2];
            const bool second = (slot%2)!=0;
            const double* other = x.data()+3*(second?f.var0_id:f.var1_id);
            const double* eta = second?f.eta1:f.eta0;
            beliefEtaFromLamMu(second?f.diag1_lam6:f.diag0_lam6,local,product);
            for (int r=0;r<3;++r) {
                double cross=0.0;
                for (int c=0;c<3;++c) cross += f.cross01_lam9[second?c+3*r:r+3*c]*other[c];
                out[r] += eta[r]-product[r]-cross;
            }
        }
    };
    if (threads>1 && w.num_vars>=128) {
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (int i=0;i<w.num_vars;++i) row(i);
    } else {
        for (int i=0;i<w.num_vars;++i) row(i);
    }
}

void synchronousIterationSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum,
    int num_threads
) {
    synchronousIterationsSyntheticSE2PackedResidualWorkspace(
        workspace,
        1,
        factor_pass_sec_accum,
        variable_pass_sec_accum,
        num_threads
    );
}

void synchronousIterationsSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int num_sweeps,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum,
    int num_threads
) {
    if (num_sweeps <= 0) {
        return;
    }

    const int fixed_eta_after = fixedEtaAfterSweep();
    const bool adaptive=workspace.adaptive_precision;
    const bool fixed_eta_enabled = adaptive || fixed_eta_after >= 0;
    if(adaptive && !workspace.precision_scales_ready) {
        workspace.precision_residuals.resize(workspace.binary_factors.size());
        workspace.precision_scales.resize(12*workspace.binary_factors.size());
        for(size_t idx=0;idx<workspace.binary_factors.size();++idx) {
            const auto& factor=workspace.binary_factors[idx];
            buildSE2PrecisionScales(factor.diag0_lam6,workspace.precision_scales.data()+12*idx);
            buildSE2PrecisionScales(factor.diag1_lam6,workspace.precision_scales.data()+12*idx+6);
        }
        workspace.precision_scales_ready=true;
    }
    if (!fixed_eta_enabled) {
        workspace.fixedeta_full_lambda_sweeps += static_cast<std::uint64_t>(num_sweeps);
        workspace.full_precision_sweeps+=num_sweeps;
    }

    const int thread_count = effectiveThreadCount(num_threads);
    const bool do_parallel = thread_count > 1 &&
        (workspace.binary_factors.size() >= 128 || workspace.num_vars >= 128);
    const bool refresh_unary = enablePackedSweepUnaryRefresh();
    const bool unary_barrier = refresh_unary || enablePackedSweepUnaryBarrier();
    if (!do_parallel) {
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = workspace.sweeps_since_relinearize;
            const bool check_precision=checkSE2PrecisionThisSweep(workspace);
            const bool use_fixed_lam = adaptive || (fixed_eta_enabled && global_sweep >= fixed_eta_after);
            const bool use_eta_only = adaptive ? workspace.precision_frozen && !check_precision
                : fixed_eta_enabled && global_sweep > fixed_eta_after;
            const bool use_serial_delta =
                use_eta_only &&
                fixedEtaSerialDeltaBeliefEnabled() &&
                workspace.belief_eta_message_consistent != 0 &&
                !workspace.serial_belief_delta_eta.empty();
            int fixed_eta_map_builds = 0;
            if (!use_eta_only) {
                workspace.fixed_eta_maps_all_valid = 0;
            }
            if (use_serial_delta) {
                std::fill(
                    workspace.serial_belief_delta_eta.begin(),
                    workspace.serial_belief_delta_eta.end(),
                    0.0
                );
            }
            if (refresh_unary) {
                for (int idx = 0; idx < static_cast<int>(workspace.unary_factors.size()); ++idx) {
                    const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[idx];
                    std::memcpy(workspace.unary_msg_eta.data() + static_cast<size_t>(idx) * 3, factor.eta, 3 * sizeof(double));
                    std::memcpy(workspace.unary_msg_lam6.data() + static_cast<size_t>(idx) * 6, factor.lam6, 6 * sizeof(double));
                }
            }
            const auto factor_t0 = SteadyClock::now();
            // The mode is constant for the whole sweep. Keep separate straight
            // loops so the serial kernel does not dispatch on every factor.
            if (use_serial_delta) {
                for (int idx = 0; idx < static_cast<int>(workspace.binary_factors.size()); ++idx) {
                    if (!computeFixedEtaFactorSerialDeltaSE2(workspace, idx, &fixed_eta_map_builds)) {
                        throw std::runtime_error("SE2 fixed eta update failed");
                    }
                }
            } else if (use_eta_only) {
                for (int idx = 0; idx < static_cast<int>(workspace.binary_factors.size()); ++idx) {
                    if (!computeFixedEtaFactorSE2(workspace, idx, &fixed_eta_map_builds)) {
                        throw std::runtime_error("SE2 fixed eta update failed");
                    }
                }
            } else {
                if (check_precision) {
                    for (int idx = 0; idx < static_cast<int>(workspace.binary_factors.size()); ++idx) {
                        if(!computeBinaryFactorWithPrecisionCheckSE2(workspace,idx))
                            throw std::runtime_error("SE2 precision check Schur update failed");
                    }
                } else {
                    for (int idx = 0; idx < static_cast<int>(workspace.binary_factors.size()); ++idx)
                        computeBinaryFactorAtNoDamping(workspace, idx);
                }
                if (fixed_eta_enabled) {
                    std::fill(workspace.fixed_lam_initialized.begin(),workspace.fixed_lam_initialized.end(),use_fixed_lam ? 1 : 0);
                    std::fill(workspace.fixed_eta_map_valid.begin(),workspace.fixed_eta_map_valid.end(),0);
                }
            }
            const auto factor_t1 = SteadyClock::now();

            const auto var_t0 = SteadyClock::now();
            if (use_serial_delta) {
                for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx)
                    applySerialBeliefEtaDeltaSE2(workspace, var_idx);
            } else if (use_eta_only) {
                for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx)
                    updateBelief3DLocalEtaOnlyNoMu(workspace, var_idx);
            } else {
                for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx)
                    updateBelief3DLocalNoMu(workspace, var_idx);
            }
            const auto var_t1 = SteadyClock::now();

            factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
            if (fixed_eta_enabled) {
                ++workspace.sweeps_since_relinearize;
                workspace.belief_eta_message_consistent = 1;
                workspace.fixedeta_eta_map_builds += static_cast<std::uint64_t>(fixed_eta_map_builds);
                if (use_eta_only) {
                    ++workspace.fixedeta_eta_only_sweeps;
                    workspace.fixed_eta_maps_all_valid = 1;
                    if (use_serial_delta) {
                        ++workspace.fixedeta_serial_delta_sweeps;
                    }
                } else if (use_fixed_lam && !adaptive) {
                    ++workspace.fixedeta_lambda_init_sweeps;
                } else {
                    ++workspace.fixedeta_full_lambda_sweeps;
                }
                completeSE2PrecisionSweep(workspace,use_eta_only,check_precision);
            }
        }
        return;
    }

    int fail_idx = -1;
    double factor_pass_batch_sec = 0.0;
    double variable_pass_batch_sec = 0.0;
    double factor_t0 = 0.0;
    double var_t0 = 0.0;
    LightweightSpinBarrier start_sweep_barrier(thread_count);
    const bool use_manual_ranges = !disablePackedOmpRangeSweeps();
    const bool use_spin_phase_barriers =
        use_manual_ranges && enablePackedOmpSpinPhaseBarriers();
    const bool use_aggressive_spin_phase_barriers =
        use_spin_phase_barriers && enablePackedOmpAggressiveSpinPhaseBarriers();
    LightweightSpinBarrier phase_barrier(
        thread_count,
        use_aggressive_spin_phase_barriers ? std::numeric_limits<int>::max() : 8192,
        !use_aggressive_spin_phase_barriers
    );
    const auto factor_ranges = use_manual_ranges
        ? buildBalancedIndexRanges(
            static_cast<int>(workspace.binary_factors.size()),
            thread_count
        )
        : std::vector<std::pair<int, int>>{};
    const auto variable_ranges = use_manual_ranges
        ? buildBalancedVariableRanges(workspace, thread_count)
        : std::vector<std::pair<int, int>>{};

    std::vector<int> fixed_eta_map_builds_by_thread(static_cast<size_t>(thread_count), 0);
    // Serial delta accumulation leaves the last increment in this scratch array.
    // A later parallel call must not apply that increment a second time.
    if (fixed_eta_enabled && fixedEtaParallelDeltaBeliefEnabled())
        std::fill(workspace.serial_belief_delta_eta.begin(), workspace.serial_belief_delta_eta.end(), 0.0);

    const auto sweep_cpu_mask = ScopedWorkerAffinity::availableMask(thread_count);
    #pragma omp parallel num_threads(thread_count) shared(fail_idx, factor_t0, var_t0, factor_pass_batch_sec, variable_pass_batch_sec)
    {
        const int tid = omp_get_thread_num();
        ScopedWorkerAffinity affinity(sweep_cpu_mask, tid);
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = fixed_eta_enabled ? workspace.sweeps_since_relinearize : 0;
            const bool check_precision=checkSE2PrecisionThisSweep(workspace);
            const bool use_fixed_lam_parallel = adaptive || (fixed_eta_enabled && global_sweep >= fixed_eta_after);
            const bool use_eta_only_parallel = adaptive ? workspace.precision_frozen && !check_precision
                : fixed_eta_enabled && global_sweep > fixed_eta_after;
            const bool use_parallel_delta =
                use_eta_only_parallel &&
                fixedEtaParallelDeltaBeliefEnabled() &&
                workspace.belief_eta_message_consistent != 0;
            int local_map_builds = 0;
            if (refresh_unary) {
                #pragma omp master
                {
                    for (int idx = 0; idx < static_cast<int>(workspace.unary_factors.size()); ++idx) {
                        const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[idx];
                        std::memcpy(workspace.unary_msg_eta.data() + static_cast<size_t>(idx) * 3, factor.eta, 3 * sizeof(double));
                        std::memcpy(workspace.unary_msg_lam6.data() + static_cast<size_t>(idx) * 6, factor.lam6, 6 * sizeof(double));
                    }
                }
            }
            if (unary_barrier) {
                start_sweep_barrier.arrive_and_wait([&]() noexcept {
                    factor_t0 = omp_get_wtime();
                });
            } else {
                #pragma omp master
                {
                    factor_t0 = omp_get_wtime();
                }
            }

            if (use_manual_ranges) {
                const auto [factor_begin, factor_end] = factor_ranges[static_cast<size_t>(tid)];
                for (int idx = factor_begin; idx < factor_end; ++idx) {
                    if (fail_idx >= 0) {
                        break;
                    }
                    if (use_eta_only_parallel) {
                        const bool ok = use_parallel_delta
                            ? computeFixedEtaFactorParallelDeltaSE2(workspace, idx, &local_map_builds)
                            : computeFixedEtaFactorSE2(workspace, idx, &local_map_builds);
                        if (!ok) {
                            #pragma omp critical
                            {
                                if (fail_idx < 0) {
                                    fail_idx = idx;
                                }
                            }
                        }
                        continue;
                    }
                    if (!(check_precision?computeBinaryFactorWithPrecisionCheckSE2(workspace,idx):
                          computeBinaryFactorAtNoDampingNoThrow(workspace,idx))) {
                        #pragma omp critical
                        {
                            if (fail_idx < 0) {
                                fail_idx = idx;
                            }
                        }
                    }
                    if (fixed_eta_enabled) {
                        workspace.fixed_lam_initialized[static_cast<size_t>(idx)] =
                            use_fixed_lam_parallel ? 1 : 0;
                        workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] = 0;
                    }
                }
                fixed_eta_map_builds_by_thread[static_cast<size_t>(tid)] = local_map_builds;
            } else {
                #pragma omp for schedule(static) nowait
                for (int idx = 0; idx < static_cast<int>(workspace.binary_factors.size()); ++idx) {
                    if (fail_idx >= 0) {
                        continue;
                    }
                    if (use_eta_only_parallel) {
                        const bool ok = use_parallel_delta
                            ? computeFixedEtaFactorParallelDeltaSE2(workspace, idx, &local_map_builds)
                            : computeFixedEtaFactorSE2(workspace, idx, &local_map_builds);
                        if (!ok) {
                            #pragma omp critical
                            {
                                if (fail_idx < 0) {
                                    fail_idx = idx;
                                }
                            }
                        }
                        continue;
                    }
                    if (!(check_precision?computeBinaryFactorWithPrecisionCheckSE2(workspace,idx):
                          computeBinaryFactorAtNoDampingNoThrow(workspace,idx))) {
                        #pragma omp critical
                        {
                            if (fail_idx < 0) {
                                fail_idx = idx;
                            }
                        }
                    }
                    if (fixed_eta_enabled) {
                        workspace.fixed_lam_initialized[static_cast<size_t>(idx)] =
                            use_fixed_lam_parallel ? 1 : 0;
                        workspace.fixed_eta_map_valid[static_cast<size_t>(idx)] = 0;
                    }
                }
                fixed_eta_map_builds_by_thread[static_cast<size_t>(tid)] = local_map_builds;
            }

            const auto publish_sweep_state = [&]() noexcept {
                if(!fixed_eta_enabled) return;
                int builds=0;
                for(int count:fixed_eta_map_builds_by_thread) builds+=count;
                ++workspace.sweeps_since_relinearize;
                workspace.belief_eta_message_consistent=1;
                workspace.fixedeta_eta_map_builds+=static_cast<std::uint64_t>(builds);
                workspace.fixed_eta_maps_all_valid=use_eta_only_parallel?1:0;
                if(use_eta_only_parallel) {
                    ++workspace.fixedeta_eta_only_sweeps;
                    if(use_parallel_delta) ++workspace.fixedeta_serial_delta_sweeps;
                } else if(use_fixed_lam_parallel && !adaptive) {
                    ++workspace.fixedeta_lambda_init_sweeps;
                } else {
                    ++workspace.fixedeta_full_lambda_sweeps;
                }
                completeSE2PrecisionSweep(workspace,use_eta_only_parallel,check_precision);
            };
            // Precision checks depend only on completed factor writes. Publish
            // before the variable barrier so it also orders next-sweep metadata.
            if (use_spin_phase_barriers) {
                phase_barrier.arrive_and_wait([&]() noexcept {
                    const double now = omp_get_wtime();
                    factor_pass_batch_sec += now - factor_t0;
                    var_t0 = now;
                    publish_sweep_state();
                });
            } else {
                #pragma omp barrier
                #pragma omp master
                {
                    factor_pass_batch_sec += omp_get_wtime() - factor_t0;
                    var_t0 = omp_get_wtime();
                    publish_sweep_state();
                }
            }

            if (use_manual_ranges) {
                const auto [var_begin, var_end] = variable_ranges[static_cast<size_t>(tid)];
                for (int var_idx = var_begin; var_idx < var_end; ++var_idx) {
                    if (fail_idx >= 0) {
                        break;
                    }
                    if (use_eta_only_parallel) {
                        if (use_parallel_delta) {
                            applyParallelBeliefEtaDeltaAndResetSE2(workspace, var_idx);
                        } else {
                            updateBelief3DLocalEtaOnlyNoMu(workspace, var_idx);
                        }
                    } else {
                        updateBelief3DLocalNoMu(workspace, var_idx);
                    }
                }
            } else {
                #pragma omp for schedule(static) nowait
                for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
                    if (fail_idx >= 0) {
                        continue;
                    }
                    if (use_eta_only_parallel) {
                        if (use_parallel_delta) {
                            applyParallelBeliefEtaDeltaAndResetSE2(workspace, var_idx);
                        } else {
                            updateBelief3DLocalEtaOnlyNoMu(workspace, var_idx);
                        }
                    } else {
                        updateBelief3DLocalNoMu(workspace, var_idx);
                    }
                }
            }

            if (use_spin_phase_barriers) {
                phase_barrier.arrive_and_wait([&]() noexcept {
                    variable_pass_batch_sec += omp_get_wtime() - var_t0;
                });
            } else {
                #pragma omp barrier
                #pragma omp master
                {
                    variable_pass_batch_sec += omp_get_wtime() - var_t0;
                }
            }

            if (fail_idx >= 0) {
                break;
            }
        }
    }

    if (fail_idx >= 0) {
        throw std::runtime_error("Linear Schur solve failed in SyntheticSE2PackedResidual");
    }

    factor_pass_sec_accum += factor_pass_batch_sec;
    variable_pass_sec_accum += variable_pass_batch_sec;
}

void stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
    SyntheticSE2PackedResidualWorkspace& workspace,
    Eigen::VectorXd& out,
    int num_threads
) {
    const int total_dim = workspace.num_vars * 3;
    if (out.size() != total_dim) {
        out.resize(total_dim);
    }
    double* out_data = out.data();
    const int thread_count = effectiveThreadCount(num_threads);
    const bool do_parallel = thread_count > 1 && workspace.num_vars >= 128;
    if (do_parallel) {
        int fail_var = -1;
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
            if (fail_var >= 0) {
                continue;
            }
            if (!refreshMuAtNoThrow(workspace, var_idx)) {
                #pragma omp critical
                {
                    if (fail_var < 0) {
                        fail_var = var_idx;
                    }
                }
                continue;
            }
            const double* mu = vec3Ptr(workspace.mu, var_idx);
            out_data[3 * var_idx + 0] = mu[0];
            out_data[3 * var_idx + 1] = mu[1];
            out_data[3 * var_idx + 2] = mu[2];
        }
        if (fail_var >= 0) {
            throw std::runtime_error("Linear solve failed in SyntheticSE2PackedResidual refreshMuAt");
        }
    } else {
        for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
            refreshMuAt(workspace, var_idx);
            const double* mu = vec3Ptr(workspace.mu, var_idx);
            out_data[3 * var_idx + 0] = mu[0];
            out_data[3 * var_idx + 1] = mu[1];
            out_data[3 * var_idx + 2] = mu[2];
        }
    }
}

Eigen::VectorXd stackedMeanVectorSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int num_threads
) {
    Eigen::VectorXd out;
    stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(workspace, out, num_threads);
    return out;
}

void injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    const Eigen::VectorXd& delta,
    int num_threads
) {
    const int thread_count = effectiveThreadCount(num_threads);
    const double* delta_data = delta.data();
    const bool do_parallel = thread_count > 1 && workspace.num_vars >= 128;
    if (do_parallel) {
        int fail_var = -1;
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
            if (fail_var >= 0) {
                continue;
            }
            if (!refreshMuAtNoThrow(workspace, var_idx)) {
                #pragma omp critical
                {
                    if (fail_var < 0) {
                        fail_var = var_idx;
                    }
                }
                continue;
            }
            double* mu = vec3Ptr(workspace.mu, var_idx);
            const double* lam6 = sym6Ptr(workspace.belief_lam6, var_idx);
            double* eta = vec3Ptr(workspace.belief_eta, var_idx);
            mu[0] += delta_data[3 * var_idx + 0];
            mu[1] += delta_data[3 * var_idx + 1];
            mu[2] += delta_data[3 * var_idx + 2];
            beliefEtaFromLamMu(lam6, mu, eta);
            workspace.mu_valid[var_idx] = 1;
        }
        if (fail_var >= 0) {
            throw std::runtime_error("Linear solve failed in SyntheticSE2PackedResidual refreshMuAt");
        }
    } else {
        for (int var_idx = 0; var_idx < workspace.num_vars; ++var_idx) {
            refreshMuAt(workspace, var_idx);
            double* mu = vec3Ptr(workspace.mu, var_idx);
            const double* lam6 = sym6Ptr(workspace.belief_lam6, var_idx);
            double* eta = vec3Ptr(workspace.belief_eta, var_idx);
            mu[0] += delta_data[3 * var_idx + 0];
            mu[1] += delta_data[3 * var_idx + 1];
            mu[2] += delta_data[3 * var_idx + 2];
            beliefEtaFromLamMu(lam6, mu, eta);
            workspace.mu_valid[var_idx] = 1;
        }
    }
    workspace.belief_eta_message_consistent = 0;
    if (workspace.lift_correction_to_messages) {
        // Minimum Euclidean change in incoming natural parameters subject to
        // sum(messages) + prior == the corrected belief. No factor/RHS changes.
        #pragma omp parallel for schedule(static) num_threads(thread_count) if(do_parallel)
        for (int v=0;v<workspace.num_vars;++v) {
            const int begin=workspace.binary_offsets[v],end=workspace.binary_offsets[v+1];
            if (end==begin) continue;
            double sum[3];
            std::memcpy(sum,vec3Ptr(workspace.prior_eta,v),3*sizeof(double));
            for(int p=workspace.unary_offsets[v];p<workspace.unary_offsets[v+1];++p) {
                const double* eta=unaryMsgEtaPtr(workspace,workspace.unary_ids[p]);
                for(int j=0;j<3;++j) sum[j]+=eta[j];
            }
            for(int p=begin;p<end;++p) {
                const double* eta=binaryMsgEtaPtr(workspace,workspace.binary_slot_ids[p]);
                for(int j=0;j<3;++j) sum[j]+=eta[j];
            }
            const double* target=vec3Ptr(workspace.belief_eta,v);
            const double step[3]={(target[0]-sum[0])/(end-begin),
                (target[1]-sum[1])/(end-begin),(target[2]-sum[2])/(end-begin)};
            for(int p=begin;p<end;++p) {
                double* eta=binaryMsgEtaPtr(workspace,workspace.binary_slot_ids[p]);
                for(int j=0;j<3;++j) eta[j]+=step[j];
            }
        }
        // Conservative: the next ordinary belief pass certifies consistency,
        // including isolated variables, before the delta-belief path is used.
    }
}

}  // namespace slam
