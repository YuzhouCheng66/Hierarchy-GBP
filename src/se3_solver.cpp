#include "internal/se3_solver_impl.h"
#include "internal/partial_symmetric_eigen.h"
#include "internal/se3_residual.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <Eigen/CholmodSupport>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SparseCholesky>
#include <omp.h>

#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace slam {

double nonlinearObjectiveForDelta(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta
);

namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat6x12 = Eigen::Matrix<double, 6, 12>;
using SteadyClock = std::chrono::steady_clock;
using Vec6Vector = std::vector<Vec6, Eigen::aligned_allocator<Vec6>>;
constexpr double kPi = 3.14159265358979323846;

struct SE3BasisData {
    std::vector<std::vector<int>> groups;
    std::vector<std::vector<int>> full_indices_per_group;
    std::vector<Eigen::MatrixXd> local_bases;
    std::vector<int> coarse_offsets;
    std::vector<int> group_global_offsets;
    std::vector<int> group_dims;
    std::vector<int> var_to_group;
    std::vector<int> var_to_local_offset;
    std::vector<int> var_global_offset;
    int total_dim = 0;
    int coarse_dim = 0;
    double block_build_sec = 0.0;
    double eigensolver_sec = 0.0;
    double copyout_sec = 0.0;
    double min_eigenvalue = std::numeric_limits<double>::infinity();
    double max_eigenvalue = -std::numeric_limits<double>::infinity();
    int negative_group_count = 0;
    int nonpositive_group_count = 0;
    int partial_attempt_count = 0;
    int partial_converged_count = 0;
    int full_eigensolver_count = 0;
};

struct SparseCholeskyFactor {
    Eigen::SparseMatrix<double> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ldlt_solver;
    bool analyzed = false;
    int rows = 0;
    int cols = 0;
    int nonzeros = 0;
    std::vector<int> outer_index;
    std::vector<int> inner_index;

    SparseCholeskyFactor() = default;
    SparseCholeskyFactor(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor& operator=(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor(SparseCholeskyFactor&&) noexcept = default;
    SparseCholeskyFactor& operator=(SparseCholeskyFactor&&) noexcept = default;
};

struct SE3SmootherSweepStats {
    double residual_before_norm = 0.0;
    double residual_after_norm = 0.0;
    double translation_update_norm_sum = 0.0;
    double rotation_update_norm_sum = 0.0;
    int local_rejects = 0;
};

struct SyntheticSE3ResidualRelinearizeStats {
    double factor_relinearize_sec = 0.0;
    double message_transport_sec = 0.0;
    double reset_state_sec = 0.0;
    double total_sec = 0.0;
};

struct SyntheticSE3ResidualGraphWorkspace {
    gbp::FactorGraph graph;
    std::vector<gbp::VariableNode*> vars;
    std::vector<gbp::Factor*> edge_factors;
    gbp::Factor* anchor_factor = nullptr;
    double tiny_prior = 1e-12;
    SE3PoseVector linearization_poses;
    bool has_linearization_poses = false;
};

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

inline int effectiveThreadCount(int requested_threads) noexcept {
    return std::max(1, (requested_threads > 0) ? requested_threads : omp_get_max_threads());
}

class ScopedSE3SingleThreadRuntime {
public:
    explicit ScopedSE3SingleThreadRuntime(bool enabled)
        : enabled_(enabled)
    {
        if (!enabled_) {
            return;
        }
        old_eigen_threads_ = Eigen::nbThreads();
        old_omp_dynamic_ = omp_get_dynamic();
        old_omp_max_threads_ = omp_get_max_threads();
        Eigen::setNbThreads(1);
        omp_set_dynamic(0);
        omp_set_num_threads(1);
    }

    ScopedSE3SingleThreadRuntime(const ScopedSE3SingleThreadRuntime&) = delete;
    ScopedSE3SingleThreadRuntime& operator=(const ScopedSE3SingleThreadRuntime&) = delete;

    ~ScopedSE3SingleThreadRuntime() {
        if (!enabled_) {
            return;
        }
        Eigen::setNbThreads(old_eigen_threads_);
        omp_set_dynamic(old_omp_dynamic_);
        omp_set_num_threads(old_omp_max_threads_);
    }

private:
    bool enabled_ = false;
    int old_eigen_threads_ = 0;
    int old_omp_dynamic_ = 0;
    int old_omp_max_threads_ = 1;
};

bool se3EnvFlagEnabled(const char* name) {
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0' && value[0] != '0';
}

bool se3EnvFlagEnabledOrDefault(const char* name, bool default_value) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return default_value;
    }
    return value[0] != '0';
}

bool se3ProfileSyncTimingEnabled() {
    return se3EnvFlagEnabledOrDefault("GBP_SE3_PROFILE_SYNC_TIMING", true);
}

bool se3SingleThreadRuntimeGuardEnabled() {
    return se3EnvFlagEnabledOrDefault("GBP_SE3_SINGLE_THREAD_RUNTIME_GUARD", true);
}

bool se3PartialBasisEigenEnabled() {
    const char* value = std::getenv("GBP_SE3_PARTIAL_BASIS_EIGEN");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return value[0] != '0';
}

bool se3PartialBasisAcceptUnconverged() {
    return se3EnvFlagEnabled("GBP_SE3_PARTIAL_BASIS_ACCEPT_UNCONVERGED");
}

int se3EnvIntOrDefault(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    try {
        return std::stoi(std::string(value));
    } catch (...) {
        return fallback;
    }
}

double se3EnvDoubleOrDefault(const char* name, double fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    try {
        return std::stod(std::string(value));
    } catch (...) {
        return fallback;
    }
}

int se3PostCoarseMessageSweeps() {
    return std::max(0, se3EnvIntOrDefault("GBP_SE3_POST_COARSE_MESSAGE_SWEEPS", 0));
}

int se3FinalDirectPolishSteps() {
    return std::max(0, se3EnvIntOrDefault("GBP_SE3_FINAL_DIRECT_POLISH_STEPS", 1));
}

int se3FixedLambdaAfterSweeps() {
    return se3EnvIntOrDefault("GBP_SE3_FIXED_LAMBDA_AFTER_SWEEPS", -1);
}

int se3FixedLambdaAfterSweepsForOuter(int outer_index) {
    const int base = se3FixedLambdaAfterSweeps();
    const int late_start = se3EnvIntOrDefault("GBP_SE3_FIXED_LAMBDA_LATE_START_OUTER", -1);
    if (late_start >= 0 && outer_index >= late_start) {
        return se3EnvIntOrDefault("GBP_SE3_FIXED_LAMBDA_AFTER_SWEEPS_LATE", base);
    }
    return base;
}

bool se3FixedLambdaHotEtaPassEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_FIXED_LAMBDA_HOT_ETA_PASS");
}

bool se3FixedLambdaHotEtaBatchEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_FIXED_LAMBDA_HOT_ETA_BATCH");
}

double se3FactorMessageMaxRelativeUpdate() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_MAX_REL_UPDATE");
        if (raw == nullptr || raw[0] == '\0') {
            return 0.0;
        }
        try {
            return std::max(0.0, std::stod(std::string(raw)));
        } catch (...) {
            return 0.0;
        }
    }();
    return value;
}

bool se3FixedLambdaInverseCacheForOuter(int outer_index) {
    const int default_start_outer = se3EnvIntOrDefault("GBP_SE3_FIXED_LAMBDA_LATE_START_OUTER", 0);
    const int start_outer = se3EnvIntOrDefault(
        "GBP_SE3_FIXED_LAMBDA_6D_INV_CACHE_START_OUTER",
        default_start_outer);
    return start_outer >= 0 && outer_index >= start_outer;
}

bool se3ReuseCoarsePatternEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_REUSE_COARSE_PATTERN");
}

bool se3CachedCoarseLambdaEnabled() {
    const char* value = std::getenv("GBP_SE3_CACHED_COARSE_LAMBDA");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return value[0] != '0';
}

int se3CoarseTouchedTlsMinFactors() {
    static const int value = std::max(
        0,
        se3EnvIntOrDefault("GBP_SE3_COARSE_TOUCHED_TLS_MIN_FACTORS", 10000)
    );
    return value;
}

bool se3CachedCoarseInplaceSymmetrizeEnabled() {
    const char* value = std::getenv("GBP_SE3_CACHED_COARSE_INPLACE_SYMMETRIZE");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return value[0] != '0';
}

bool se3CoarseRidgeInplaceEnabled() {
    const char* value = std::getenv("GBP_SE3_COARSE_RIDGE_INPLACE");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return value[0] != '0';
}

int se3CachedCoarseLambdaMinGroups() {
    return std::max(1, se3EnvIntOrDefault("GBP_SE3_CACHED_COARSE_LAMBDA_MIN_GROUPS", 200));
}

constexpr int kSE3CoarseAssemblyShards = 16;

int se3CoarseNumericRebuildPeriod() {
    return std::max(
        1,
        se3EnvIntOrDefault("GBP_SE3_COARSE_NUMERIC_REBUILD_PERIOD", 1)
    );
}

int se3CoarseReusePcgIterations() {
    return std::max(
        1,
        se3EnvIntOrDefault("GBP_SE3_COARSE_REUSE_PCG_ITERS", 3)
    );
}

int se3PartialBasisLargeGroupMinGroups() {
    return std::max(1, se3EnvIntOrDefault("GBP_SE3_PARTIAL_BASIS_LARGE_GROUP_MIN_GROUPS", 350));
}

double se3PartialBasisLargeGroupAcceptResidualTol() {
    return se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_LARGE_GROUP_ACCEPT_RESIDUAL_TOL",
        1e-3
    );
}

double se3PartialBasisSmallGroupAcceptResidualTol() {
    return se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_SMALL_GROUP_ACCEPT_RESIDUAL_TOL",
        1e-4
    );
}

double se3MessageDamping() {
    return std::clamp(se3EnvDoubleOrDefault("GBP_SE3_MESSAGE_DAMPING", 0.0), 0.0, 0.99);
}

int se3BasisRebuildPeriod() {
    return std::max(1, se3EnvIntOrDefault("GBP_SE3_BASIS_REBUILD_PERIOD", 1));
}

int se3BasisRebuildWarmupOuters() {
    return std::max(0, se3EnvIntOrDefault("GBP_SE3_BASIS_REBUILD_WARMUP_OUTERS", 0));
}

double jsonNumber(double value) {
    if (std::isfinite(value)) {
        return value;
    }
    return 0.0;
}

double huberWeightFromMahalanobisNorm(double mahal_norm, const RobustLossConfig& config) {
    if (!(config.huber_delta > 0.0) || !(mahal_norm > config.huber_delta)) {
        return 1.0;
    }
    if (!(mahal_norm > 0.0)) {
        return 1.0;
    }
    return config.huber_delta / mahal_norm;
}

double robustWeightForResidual(
    const Vec6& err,
    const Mat6& information,
    const RobustLossConfig& config
) {
    if (!(config.huber_delta > 0.0)) {
        return 1.0;
    }
    const double quad = (err.transpose() * information * err)(0, 0);
    const double mahal_norm = std::sqrt(std::max(0.0, quad));
    return huberWeightFromMahalanobisNorm(mahal_norm, config);
}

SE3Pose normalizedPose(const SE3Pose& pose) {
    SE3Pose out = pose;
    out.q.normalize();
    if (out.q.w() < 0.0) {
        out.q.coeffs() *= -1.0;
    }
    return out;
}

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m = Eigen::Matrix3d::Zero();
    m(0, 1) = -v.z();
    m(0, 2) = v.y();
    m(1, 0) = v.z();
    m(1, 2) = -v.x();
    m(2, 0) = -v.y();
    m(2, 1) = v.x();
    return m;
}

Mat6 adjointInverseSE3(const SE3Pose& pose) {
    Mat6 A = Mat6::Zero();
    const Eigen::Matrix3d Rt = pose.q.toRotationMatrix().transpose();
    A.topLeftCorner<3, 3>() = Rt;
    A.topRightCorner<3, 3>() = -Rt * skew(pose.t);
    A.bottomRightCorner<3, 3>() = Rt;
    return A;
}

Eigen::Matrix3d so3Exp(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    const Eigen::Matrix3d W = skew(w);
    const Eigen::Matrix3d W2 = W * W;

    double a = 1.0;
    double b = 0.5;
    if (theta > 1e-12) {
        a = std::sin(theta) / theta;
        b = (1.0 - std::cos(theta)) / (theta * theta);
    } else {
        const double theta2 = theta * theta;
        a = 1.0 - theta2 / 6.0;
        b = 0.5 - theta2 / 24.0;
    }

    return Eigen::Matrix3d::Identity() + a * W + b * W2;
}

Eigen::Vector3d so3Log(const Eigen::Matrix3d& R) {
    const double cos_theta = std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0);
    const double theta = std::acos(cos_theta);
    const Eigen::Vector3d vee(
        R(2, 1) - R(1, 2),
        R(0, 2) - R(2, 0),
        R(1, 0) - R(0, 1)
    );

    if (theta < 1e-8) {
        return 0.5 * vee;
    }

    if (kPi - theta < 1e-6) {
        Eigen::AngleAxisd aa(R);
        return aa.axis() * aa.angle();
    }

    return (0.5 * theta / std::sin(theta)) * vee;
}

Eigen::Matrix3d leftJacobianSO3(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    const Eigen::Matrix3d W = skew(w);
    const Eigen::Matrix3d W2 = W * W;

    double a = 0.5;
    double b = 1.0 / 6.0;
    if (theta > 1e-12) {
        a = (1.0 - std::cos(theta)) / (theta * theta);
        b = (theta - std::sin(theta)) / (theta * theta * theta);
    } else {
        const double theta2 = theta * theta;
        a = 0.5 - theta2 / 24.0;
        b = 1.0 / 6.0 - theta2 / 120.0;
    }

    return Eigen::Matrix3d::Identity() + a * W + b * W2;
}

Eigen::Matrix3d leftJacobianInverseSO3(const Eigen::Vector3d& w) {
    const double theta = w.norm();
    const Eigen::Matrix3d W = skew(w);
    const Eigen::Matrix3d W2 = W * W;

    if (theta < 1e-8) {
        return Eigen::Matrix3d::Identity() - 0.5 * W + (1.0 / 12.0) * W2;
    }

    const double half_theta = 0.5 * theta;
    const double cot_half = 1.0 / std::tan(half_theta);
    const double coeff = 1.0 / (theta * theta) - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta));
    return Eigen::Matrix3d::Identity() - 0.5 * W + coeff * W2;
}

Eigen::Matrix3d rightJacobianInverseSO3(const Eigen::Vector3d& w) {
    const double theta2 = w.squaredNorm();
    if (theta2 <= std::numeric_limits<double>::epsilon()) {
        return Eigen::Matrix3d::Identity();
    }

    const double theta = std::sqrt(theta2);
    const Eigen::Matrix3d W = skew(w);
    return Eigen::Matrix3d::Identity() + 0.5 * W +
        (1.0 / theta2 - (1.0 + std::cos(theta)) / (2.0 * theta * std::sin(theta))) * (W * W);
}

SE3Pose se3Compose(const SE3Pose& a, const SE3Pose& b) {
    SE3Pose out;
    out.q = a.q * b.q;
    out.q.normalize();
    out.t = a.t + a.q.toRotationMatrix() * b.t;
    if (out.q.w() < 0.0) {
        out.q.coeffs() *= -1.0;
    }
    return out;
}

SE3Pose se3Inverse(const SE3Pose& pose) {
    SE3Pose out;
    out.q = pose.q.conjugate();
    out.q.normalize();
    const Eigen::Matrix3d RT = out.q.toRotationMatrix();
    out.t = -(RT * pose.t);
    if (out.q.w() < 0.0) {
        out.q.coeffs() *= -1.0;
    }
    return out;
}

SE3Pose se3Between(const SE3Pose& a, const SE3Pose& b) {
    return se3Compose(se3Inverse(a), b);
}

Mat6 se3Adjoint(const SE3Pose& pose) {
    const Eigen::Matrix3d R = pose.q.normalized().toRotationMatrix();
    Mat6 Ad = Mat6::Zero();
    Ad.topLeftCorner<3, 3>() = R;
    Ad.topRightCorner<3, 3>() = skew(pose.t) * R;
    Ad.bottomRightCorner<3, 3>() = R;
    return Ad;
}

Eigen::Matrix3d computeQforExpmapDerivative(const Vec6& xi, double near_zero_threshold = 1e-5) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    const Eigen::Matrix3d V = skew(rho);
    const Eigen::Matrix3d W = skew(phi);
    const Eigen::Matrix3d WVW = W * V * W;

    const double angle = phi.norm();
    if (std::abs(angle) > near_zero_threshold) {
        const double s = std::sin(angle);
        const double c = std::cos(angle);
        const double a2 = angle * angle;
        const double a3 = a2 * angle;
        const double a4 = a3 * angle;
        const double a5 = a4 * angle;
        return -0.5 * V
            + (angle - s) / a3 * (W * V + V * W - WVW)
            + (1.0 - a2 / 2.0 - c) / a4 * (W * W * V + V * W * W - 3.0 * WVW)
            - 0.5 * ((1.0 - a2 / 2.0 - c) / a4 - 3.0 * (angle - s - a3 / 6.0) / a5) *
                (WVW * W + W * WVW);
    }

    return -0.5 * V
        + (1.0 / 6.0) * (W * V + V * W - WVW)
        - (1.0 / 24.0) * (W * W * V + V * W * W - 3.0 * WVW)
        + (1.0 / 120.0) * (WVW * W + W * WVW);
}

SE3Pose se3Exp(const Vec6& xi) {
    const Eigen::Vector3d rho = xi.head<3>();
    const Eigen::Vector3d phi = xi.tail<3>();
    const Eigen::Matrix3d R = so3Exp(phi);
    const Eigen::Matrix3d V = leftJacobianSO3(phi);

    SE3Pose out;
    out.t = V * rho;
    out.q = Eigen::Quaterniond(R);
    out.q.normalize();
    if (out.q.w() < 0.0) {
        out.q.coeffs() *= -1.0;
    }
    return out;
}

Vec6 se3Log(const SE3Pose& pose) {
    const Eigen::Matrix3d R = pose.q.normalized().toRotationMatrix();
    const Eigen::Vector3d phi = so3Log(R);
    const Eigen::Matrix3d V_inv = leftJacobianInverseSO3(phi);

    Vec6 xi = Vec6::Zero();
    xi.head<3>() = V_inv * pose.t;
    xi.tail<3>() = phi;
    return xi;
}

Mat6 se3LogmapDerivative(const SE3Pose& pose) {
    const Vec6 xi = se3Log(pose);
    const Eigen::Vector3d phi = xi.tail<3>();
    const Eigen::Matrix3d Jr_inv = rightJacobianInverseSO3(phi);
    const Eigen::Matrix3d Q = computeQforExpmapDerivative(xi);
    const Eigen::Matrix3d Q2 = -Jr_inv * Q * Jr_inv;

    Mat6 J = Mat6::Zero();
    J.topLeftCorner<3, 3>() = Jr_inv;
    J.topRightCorner<3, 3>() = Q2;
    J.bottomRightCorner<3, 3>() = Jr_inv;
    return J;
}

template <typename Derived>
bool allFiniteEigen(const Eigen::MatrixBase<Derived>& value) {
    return value.array().isFinite().all();
}

SE3Pose se3Plus(const SE3Pose& base_pose, const Vec6& delta) {
    return se3Compose(base_pose, se3Exp(delta));
}

Vec6 edgeResidual(
    const SE3Pose& base_i,
    const SE3Pose& base_j,
    const SE3Pose& measurement,
    const Vec6& ei,
    const Vec6& ej
) {
    const SE3Pose xi = se3Plus(base_i, ei);
    const SE3Pose xj = se3Plus(base_j, ej);
    const SE3Pose pred = se3Between(xi, xj);
    const SE3Pose err_pose = se3Compose(se3Inverse(measurement), pred);
    return se3Log(err_pose);
}

Vec6 anchorResidual(
    const SE3Pose& base_pose,
    const SE3Pose& anchor_pose,
    const Vec6& e
) {
    const SE3Pose xi = se3Plus(base_pose, e);
    const SE3Pose err_pose = se3Compose(se3Inverse(anchor_pose), xi);
    return se3Log(err_pose);
}

Mat6x12 analyticEdgeResidualJacobian(
    const SE3Pose& base_i,
    const SE3Pose& base_j,
    const SE3Pose& measurement
) {
    const SE3Pose pred = se3Between(base_i, base_j);
    const SE3Pose err_pose = se3Compose(se3Inverse(measurement), pred);
    const Mat6 Dlog = se3LogmapDerivative(err_pose);
    const Mat6 Dbetween_i = -se3Adjoint(se3Inverse(pred));

    Mat6x12 J = Mat6x12::Zero();
    J.leftCols<6>().noalias() = Dlog * Dbetween_i;
    J.rightCols<6>() = Dlog;
    return J;
}

Mat6 analyticAnchorResidualJacobian(
    const SE3Pose& base_pose,
    const SE3Pose& anchor_pose
) {
    const SE3Pose err_pose = se3Compose(se3Inverse(anchor_pose), base_pose);
    return se3LogmapDerivative(err_pose);
}

Mat6 info21ToMatrix(const std::array<double, 21>& vals) {
    Mat6 info = Mat6::Zero();
    int idx = 0;
    for (int r = 0; r < 6; ++r) {
        for (int c = r; c < 6; ++c) {
            info(r, c) = vals[idx];
            info(c, r) = vals[idx];
            ++idx;
        }
    }
    return info;
}

Mat6 symmetrizedInformationMatrix(const Mat6& raw_info) {
    return 0.5 * (raw_info + Mat6(raw_info.transpose()));
}

std::vector<std::vector<int>> orderedGroups(int n_vars, int group_size) {
    if (group_size <= 0) {
        throw std::runtime_error("group_size must be positive");
    }
    std::vector<std::vector<int>> groups;
    int start = 0;
    while (start < n_vars) {
        const int end = std::min(n_vars, start + group_size);
        std::vector<int> group;
        group.reserve(end - start);
        for (int id = start; id < end; ++id) {
            group.push_back(id);
        }
        groups.push_back(std::move(group));
        start = end;
    }
    return groups;
}

Eigen::SparseMatrix<double> symmetrizeSparse(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> sym = 0.5 * (A + Eigen::SparseMatrix<double>(A.transpose()));
    sym.makeCompressed();
    return sym;
}

struct JointInfSparsePatternCache {
    struct DenseBlockPlan {
        int row_offset = 0;
        int col_offset = 0;
        int rows = 0;
        int cols = 0;
        int factor_row = 0;
        int factor_col = 0;
        std::vector<int> value_indices;
    };

    struct EtaSegmentPlan {
        int global_offset = 0;
        int factor_offset = 0;
        int dofs = 0;
    };

    struct FactorPlan {
        int factor_index = -1;
        std::vector<EtaSegmentPlan> eta_segments;
        std::vector<DenseBlockPlan> blocks;
    };

    struct SymValuePair {
        int first = -1;
        int second = -1;
    };

    int total_dim = 0;
    int num_vars = 0;
    int num_factors = 0;
    std::vector<int> var_ix;
    Eigen::SparseMatrix<double> lam;
    std::unordered_map<long long, int> coeff_to_value_index;
    std::vector<DenseBlockPlan> prior_blocks;
    std::vector<FactorPlan> factor_plans;
    std::vector<SymValuePair> sym_value_pairs;
    bool valid = false;
};

long long sparseCoeffKey(int row, int col) {
    return (static_cast<long long>(row) << 32) ^
        static_cast<unsigned int>(col);
}

JointInfSparsePatternCache::DenseBlockPlan makeDenseBlockPlan(
    const JointInfSparsePatternCache& cache,
    int row_offset,
    int col_offset,
    int rows,
    int cols,
    int factor_row = 0,
    int factor_col = 0
) {
    JointInfSparsePatternCache::DenseBlockPlan plan;
    plan.row_offset = row_offset;
    plan.col_offset = col_offset;
    plan.rows = rows;
    plan.cols = cols;
    plan.factor_row = factor_row;
    plan.factor_col = factor_col;
    plan.value_indices.reserve(static_cast<size_t>(rows * cols));
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            const auto it = cache.coeff_to_value_index.find(
                sparseCoeffKey(row_offset + r, col_offset + c)
            );
            if (it == cache.coeff_to_value_index.end()) {
                throw std::runtime_error("cached joint plan missing sparse coefficient");
            }
            plan.value_indices.push_back(it->second);
        }
    }
    return plan;
}

template <typename DenseLike>
void addDenseBlockByPlan(
    double* values,
    const JointInfSparsePatternCache::DenseBlockPlan& plan,
    const DenseLike& block
) {
    size_t idx = 0;
    for (int c = 0; c < plan.cols; ++c) {
        for (int r = 0; r < plan.rows; ++r) {
            values[plan.value_indices[idx++]] +=
                block(plan.factor_row + r, plan.factor_col + c);
        }
    }
}

void buildJointInfSparsePatternCache(
    const gbp::FactorGraph& graph,
    JointInfSparsePatternCache& cache
) {
    cache = JointInfSparsePatternCache{};
    int max_id = -1;
    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        cache.total_dim += v->dofs;
        max_id = std::max(max_id, v->variableID);
    }
    cache.num_vars = static_cast<int>(graph.var_nodes.size());
    cache.num_factors = static_cast<int>(graph.factors.size());
    cache.var_ix.assign(max_id + 1, -1);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(cache.total_dim) * 10);

    int offset = 0;
    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        const int id = v->variableID;
        const int m = v->dofs;
        cache.var_ix[id] = offset;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                trips.emplace_back(offset + r, offset + c, 1.0);
            }
        }
        offset += m;
    }

    for (const auto& fptr : graph.factors) {
        const gbp::Factor& f = *fptr;
        if (!f.active) {
            continue;
        }
        const int k = static_cast<int>(f.adj_var_nodes.size());
        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da = va->dofs;
            const int oa = (ida >= 0 && ida < static_cast<int>(cache.var_ix.size()))
                ? cache.var_ix[ida]
                : -1;
            if (oa < 0) {
                throw std::runtime_error("cached joint pattern: missing ida");
            }
            for (int r = 0; r < da; ++r) {
                for (int c = 0; c < da; ++c) {
                    trips.emplace_back(oa + r, oa + c, 1.0);
                }
            }

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                (void)other_factor_ix;
                const gbp::VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db = vb->dofs;
                const int ob = (idb >= 0 && idb < static_cast<int>(cache.var_ix.size()))
                    ? cache.var_ix[idb]
                    : -1;
                if (ob < 0) {
                    throw std::runtime_error("cached joint pattern: missing idb");
                }
                if (idb > ida) {
                    for (int r = 0; r < da; ++r) {
                        for (int c = 0; c < db; ++c) {
                            trips.emplace_back(oa + r, ob + c, 1.0);
                            trips.emplace_back(ob + c, oa + r, 1.0);
                        }
                    }
                }
                other_factor_ix += db;
            }
            factor_ix += da;
        }
    }

    cache.lam.resize(cache.total_dim, cache.total_dim);
    cache.lam.setFromTriplets(trips.begin(), trips.end());
    cache.lam.makeCompressed();
    cache.coeff_to_value_index.reserve(static_cast<size_t>(cache.lam.nonZeros()) * 2);
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            cache.coeff_to_value_index.emplace(sparseCoeffKey(row, col), p);
        }
    }
    cache.sym_value_pairs.reserve(static_cast<size_t>(cache.lam.nonZeros() / 2));
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            if (row >= col) {
                continue;
            }
            const auto transpose_it = cache.coeff_to_value_index.find(sparseCoeffKey(col, row));
            if (transpose_it != cache.coeff_to_value_index.end()) {
                cache.sym_value_pairs.push_back(
                    JointInfSparsePatternCache::SymValuePair{p, transpose_it->second}
                );
            }
        }
    }

    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        const int offset = cache.var_ix[v->variableID];
        cache.prior_blocks.push_back(
            makeDenseBlockPlan(cache, offset, offset, v->dofs, v->dofs)
        );
    }

    cache.factor_plans.reserve(graph.factors.size());
    for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
        const gbp::Factor& f = *graph.factors[static_cast<size_t>(fi)];
        JointInfSparsePatternCache::FactorPlan factor_plan;
        factor_plan.factor_index = fi;
        const int k = static_cast<int>(f.adj_var_nodes.size());
        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da = va->dofs;
            const int oa = cache.var_ix[ida];
            factor_plan.eta_segments.push_back(
                JointInfSparsePatternCache::EtaSegmentPlan{oa, factor_ix, da}
            );
            factor_plan.blocks.push_back(
                makeDenseBlockPlan(cache, oa, oa, da, da, factor_ix, factor_ix)
            );

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                const gbp::VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db = vb->dofs;
                const int ob = cache.var_ix[idb];
                if (idb > ida) {
                    factor_plan.blocks.push_back(
                        makeDenseBlockPlan(cache, oa, ob, da, db, factor_ix, other_factor_ix)
                    );
                    factor_plan.blocks.push_back(
                        makeDenseBlockPlan(cache, ob, oa, db, da, other_factor_ix, factor_ix)
                    );
                }
                other_factor_ix += db;
            }
            factor_ix += da;
        }
        cache.factor_plans.push_back(std::move(factor_plan));
    }
    cache.valid = true;
}

void fillJointDistributionInfSparseCached(
    const gbp::FactorGraph& graph,
    JointInfSparsePatternCache& cache,
    Eigen::VectorXd& eta_out
) {
    if (!cache.valid ||
        cache.num_vars != static_cast<int>(graph.var_nodes.size()) ||
        cache.num_factors != static_cast<int>(graph.factors.size())) {
        buildJointInfSparsePatternCache(graph, cache);
    }

    std::fill(
        cache.lam.valuePtr(),
        cache.lam.valuePtr() + cache.lam.nonZeros(),
        0.0
    );

    eta_out.setZero(cache.total_dim);

    double* values = cache.lam.valuePtr();
    size_t prior_plan_index = 0;
    for (int vi = 0; vi < static_cast<int>(graph.var_nodes.size()); ++vi) {
        const auto& v = graph.var_nodes[static_cast<size_t>(vi)];
        if (!v) {
            continue;
        }
        const int id = v->variableID;
        const int m = v->dofs;
        const int offset = cache.var_ix[id];
        eta_out.segment(offset, m) += v->prior.eta();
        addDenseBlockByPlan(values, cache.prior_blocks[prior_plan_index++], v->prior.lam());
    }

    for (const JointInfSparsePatternCache::FactorPlan& plan : cache.factor_plans) {
        if (plan.factor_index < 0 ||
            plan.factor_index >= static_cast<int>(graph.factors.size())) {
            continue;
        }
        const gbp::Factor& f = *graph.factors[static_cast<size_t>(plan.factor_index)];
        if (!f.active) {
            continue;
        }
        const auto f_eta = f.factor.eta();
        const auto f_lam = f.factor.lam();

        for (const JointInfSparsePatternCache::EtaSegmentPlan& eta_plan : plan.eta_segments) {
            eta_out.segment(eta_plan.global_offset, eta_plan.dofs) +=
                f_eta.segment(eta_plan.factor_offset, eta_plan.dofs);
        }
        for (const JointInfSparsePatternCache::DenseBlockPlan& block_plan : plan.blocks) {
            addDenseBlockByPlan(values, block_plan, f_lam);
        }
    }

    for (const JointInfSparsePatternCache::SymValuePair& pair : cache.sym_value_pairs) {
        const double avg = 0.5 * (values[pair.first] + values[pair.second]);
        values[pair.first] = avg;
        values[pair.second] = avg;
    }
}

Eigen::VectorXd solveSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    const Eigen::VectorXd& eta,
    double ridge
) {
    if (lam.rows() == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    Eigen::SparseMatrix<double> A = symmetrizeSparse(lam);
    if (ridge != 0.0) {
        for (int i = 0; i < A.rows(); ++i) {
            A.coeffRef(i, i) += ridge;
        }
    }
    A.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }

    const Eigen::VectorXd x = solver.solve(eta);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
    return x;
}

void factorizeSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    double ridge,
    SparseCholeskyFactor& factor,
    bool reuse_pattern = false,
    bool input_already_symmetric = false
) {
    const Eigen::SparseMatrix<double>* A_ptr = &lam;
    if (!input_already_symmetric || ridge != 0.0) {
        factor.A = input_already_symmetric ? lam : symmetrizeSparse(lam);
        if (ridge != 0.0) {
            for (int i = 0; i < factor.A.rows(); ++i) {
                factor.A.coeffRef(i, i) += ridge;
            }
        }
        factor.A.makeCompressed();
        A_ptr = &factor.A;
    }
    const Eigen::SparseMatrix<double>& A = *A_ptr;

    if (!reuse_pattern) {
        factor.ldlt_solver.compute(A);
        if (factor.ldlt_solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky factorization failed");
        }
        return;
    }

    const int outer_size = static_cast<int>(A.outerSize()) + 1;
    const int inner_size = static_cast<int>(A.nonZeros());
    const bool can_reuse =
        reuse_pattern &&
        factor.analyzed &&
        factor.rows == A.rows() &&
        factor.cols == A.cols() &&
        factor.nonzeros == A.nonZeros() &&
        static_cast<int>(factor.outer_index.size()) == outer_size &&
        static_cast<int>(factor.inner_index.size()) == inner_size &&
        std::equal(
            factor.outer_index.begin(),
            factor.outer_index.end(),
            A.outerIndexPtr()
        ) &&
        std::equal(
            factor.inner_index.begin(),
            factor.inner_index.end(),
            A.innerIndexPtr()
    );

    if (!can_reuse) {
        factor.ldlt_solver.analyzePattern(A);
        if (factor.ldlt_solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky analyzePattern failed");
        }
        factor.analyzed = true;
        factor.rows = static_cast<int>(A.rows());
        factor.cols = static_cast<int>(A.cols());
        factor.nonzeros = static_cast<int>(A.nonZeros());
        factor.outer_index.assign(A.outerIndexPtr(), A.outerIndexPtr() + outer_size);
        factor.inner_index.assign(A.innerIndexPtr(), A.innerIndexPtr() + inner_size);
    }

    factor.ldlt_solver.factorize(A);
    if (factor.ldlt_solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }
}

void solveWithSparseCholeskyInto(
    const SparseCholeskyFactor& factor,
    const Eigen::VectorXd& eta,
    Eigen::VectorXd& x
) {
    x = factor.ldlt_solver.solve(eta);
    if (factor.ldlt_solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
}

void solveWithReusedCoarsePcgInto(
    const Eigen::SparseMatrix<double>& current_lam,
    const SparseCholeskyFactor& stale_factor,
    const Eigen::VectorXd& eta,
    int max_iterations,
    Eigen::VectorXd& x,
    Eigen::VectorXd& residual,
    Eigen::VectorXd& preconditioned_residual,
    Eigen::VectorXd& direction,
    Eigen::VectorXd& matrix_direction
) {
    const Eigen::Index n = eta.size();
    x.setZero(n);
    residual = eta;
    solveWithSparseCholeskyInto(stale_factor, residual, preconditioned_residual);
    direction = preconditioned_residual;

    double residual_preconditioned_dot = residual.dot(preconditioned_residual);
    if (!(residual_preconditioned_dot > 0.0) ||
        !std::isfinite(residual_preconditioned_dot)) {
        x = preconditioned_residual;
        return;
    }

    const double target_residual_sq =
        1e-12 * std::max(eta.squaredNorm(), 1e-30);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        matrix_direction.noalias() = current_lam * direction;
        const double curvature = direction.dot(matrix_direction);
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            if (iteration == 0) {
                x = preconditioned_residual;
            }
            return;
        }

        const double alpha = residual_preconditioned_dot / curvature;
        if (!std::isfinite(alpha)) {
            if (iteration == 0) {
                x = preconditioned_residual;
            }
            return;
        }
        x.noalias() += alpha * direction;
        residual.noalias() -= alpha * matrix_direction;
        if (residual.squaredNorm() <= target_residual_sq) {
            return;
        }

        solveWithSparseCholeskyInto(
            stale_factor,
            residual,
            preconditioned_residual
        );
        const double next_dot = residual.dot(preconditioned_residual);
        if (!(next_dot > 0.0) || !std::isfinite(next_dot)) {
            return;
        }
        const double beta = next_dot / residual_preconditioned_dot;
        direction = preconditioned_residual + beta * direction;
        residual_preconditioned_dot = next_dot;
    }
}

Eigen::MatrixXd densePrincipalSubmatrix(
    const Eigen::SparseMatrix<double>& matrix,
    const std::vector<int>& rows
) {
    const int n = static_cast<int>(rows.size());
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(n, n);
    for (int c = 0; c < n; ++c) {
        for (int r = 0; r < n; ++r) {
            block(r, c) = matrix.coeff(rows[r], rows[c]);
        }
    }
    block.template triangularView<Eigen::StrictlyLower>() =
        block.transpose().template triangularView<Eigen::StrictlyLower>();
    return block;
}

Eigen::VectorXd stackedMeanVector(gbp::FactorGraph& graph) {
    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }

    Eigen::VectorXd out = Eigen::VectorXd::Zero(total_dim);
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        vup->refreshMu();
        out.segment(offset, vup->dofs) = vup->mu;
        offset += vup->dofs;
    }
    return out;
}

void accumulateSE3DeltaComponentNorms(
    const Eigen::VectorXd& delta,
    double& translation_norm_sum,
    double& rotation_norm_sum
) {
    translation_norm_sum = 0.0;
    rotation_norm_sum = 0.0;
    const int n = static_cast<int>(delta.size() / 6);
    for (int i = 0; i < n; ++i) {
        translation_norm_sum += delta.segment<3>(6 * i).norm();
        rotation_norm_sum += delta.segment<3>(6 * i + 3).norm();
    }
}

int chooseActiveFixedLambdaStart(int hard_fixed_start) {
    return hard_fixed_start;
}
void recordSE3FixedLambdaSweepProfile(
    gbp::FactorGraph& graph,
    int hard_fixed_start,
    int active_fixed_start,
    bool use_fixed_lam,
    bool use_eta_only_variable_update
) {
    if (!graph.profile_sync_timing) {
        return;
    }
    ++graph.sync_total_sweeps_accum;
    if (hard_fixed_start >= 0 &&
        (graph.sync_fixed_lam_hard_start_sweep < 0 ||
         hard_fixed_start < graph.sync_fixed_lam_hard_start_sweep)) {
        graph.sync_fixed_lam_hard_start_sweep = hard_fixed_start;
    }
    if (active_fixed_start >= 0 &&
        (graph.sync_fixed_lam_effective_start_sweep < 0 ||
         active_fixed_start < graph.sync_fixed_lam_effective_start_sweep)) {
        graph.sync_fixed_lam_effective_start_sweep = active_fixed_start;
    }
    if (use_fixed_lam) {
        ++graph.sync_fixed_lam_factor_sweeps_accum;
    }
    if (use_eta_only_variable_update) {
        ++graph.sync_eta_only_variable_sweeps_accum;
    }
}

bool se3PackedSoASweepsEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_SWEEPS");
}

bool se3PackedSoAFullCopyEachCallEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_FULL_COPY_EACH_CALL");
}

bool se3PackedSoADeferGraphCopyEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_DEFER_GRAPH_COPY");
}

bool se3ImplicitFineOperatorEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_IMPLICIT_FINE_OPERATOR");
}

bool se3ImplicitFineOperatorCompareEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_IMPLICIT_FINE_OPERATOR_COMPARE");
}

int se3ImplicitFineOperatorThreadCount(int fallback_threads) {
    const int configured =
        se3EnvIntOrDefault("GBP_SE3_IMPLICIT_FINE_OPERATOR_THREADS", 0);
    return configured > 0 ? effectiveThreadCount(configured) : fallback_threads;
}

struct SE3PackedSoASweepRuntime {
    std::unique_ptr<SyntheticSE3PackedSoAWorkspace> workspace;
    const SyntheticSE3Problem* problem = nullptr;
    bool valid = false;
};

std::unordered_map<gbp::FactorGraph*, SE3PackedSoASweepRuntime>& se3PackedSoAStates() {
    static std::unordered_map<gbp::FactorGraph*, SE3PackedSoASweepRuntime> states;
    return states;
}

SyntheticSE3PackedSoAWorkspace* preparePackedSoAWorkspace(
    gbp::FactorGraph& graph,
    const SyntheticSE3Problem& problem
) {
    if (!se3PackedSoASweepsEnabled()) {
        return nullptr;
    }
    auto& state = se3PackedSoAStates()[&graph];
    if (state.workspace == nullptr || state.problem != &problem) {
        state.workspace = std::make_unique<SyntheticSE3PackedSoAWorkspace>(
            buildSyntheticSE3PackedSoAWorkspace(problem, 1e-12)
        );
        state.problem = &problem;
        state.valid = false;
    }
    if (!state.valid) {
        relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(*state.workspace, graph);
        state.valid = true;
    }
    return state.workspace.get();
}

bool tryStackedMeanVectorPackedSoA(
    gbp::FactorGraph& graph,
    int num_threads,
    Eigen::VectorXd& out
) {
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it == se3PackedSoAStates().end() ||
        !packed_state_it->second.valid ||
        packed_state_it->second.workspace == nullptr) {
        return false;
    }
    stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(
        *packed_state_it->second.workspace,
        out,
        num_threads
    );
    return true;
}

bool tryApplyMeanDeltaPackedSoA(
    gbp::FactorGraph& graph,
    const Eigen::VectorXd& delta,
    int num_threads
) {
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it == se3PackedSoAStates().end() ||
        !packed_state_it->second.valid ||
        packed_state_it->second.workspace == nullptr) {
        return false;
    }
    applyMeanDeltaSyntheticSE3PackedSoAWorkspace(
        *packed_state_it->second.workspace,
        delta,
        num_threads
    );
    return true;
}

bool tryManualSynchronousIterationsPackedSoA(
    gbp::FactorGraph& graph,
    int num_sweeps,
    int num_threads,
    int sweep_offset,
    int outer_index,
    int fixed_lam_after,
    const SyntheticSE3Problem* problem
) {
    if (!se3PackedSoASweepsEnabled() || num_sweeps <= 0) {
        return false;
    }
    auto& states = se3PackedSoAStates();
    auto& state = states[&graph];
    if (problem != nullptr &&
        (state.workspace == nullptr || state.problem != problem)) {
        state.workspace = std::make_unique<SyntheticSE3PackedSoAWorkspace>(
            buildSyntheticSE3PackedSoAWorkspace(*problem, 1e-12)
        );
        state.problem = problem;
        state.valid = false;
    }
    if (state.workspace == nullptr) {
        return false;
    }

    SyntheticSE3PackedSoAWorkspace& packed = *state.workspace;
    if (!state.valid) {
        relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(packed, graph);
        state.valid = true;
    }
    packed.sweeps_since_relinearize = std::max(0, sweep_offset);
    if (fixed_lam_after >= 0 && sweep_offset > fixed_lam_after) {
        std::fill(
            packed.fixed_lam_initialized.begin(),
            packed.fixed_lam_initialized.end(),
            static_cast<std::uint8_t>(1)
        );
    }

    const int active_fixed_lam_start =
        chooseActiveFixedLambdaStart(fixed_lam_after);
    if (graph.profile_sync_timing) {
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = sweep_offset + sweep;
            const bool use_fixed_lam =
                active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
            const bool use_eta_only =
                active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
            recordSE3FixedLambdaSweepProfile(
                graph,
                fixed_lam_after,
                active_fixed_lam_start,
                use_fixed_lam,
                use_eta_only
            );
        }
    }

    SyntheticSE3PackedSoAStats stats;
    SyntheticSE3PackedSoAStats* stats_ptr = graph.profile_sync_timing ? &stats : nullptr;
    synchronousIterationsSyntheticSE3PackedSoAWorkspace(
        packed,
        num_sweeps,
        num_threads,
        fixed_lam_after,
        graph.eta_damping,
        se3FactorMessageMaxRelativeUpdate(),
        stats_ptr
    );
    if (!se3PackedSoADeferGraphCopyEnabled()) {
        if (se3PackedSoAFullCopyEachCallEnabled()) {
            copySyntheticSE3PackedSoAToGraph(packed, graph, num_threads);
        } else {
            const bool all_sweeps_eta_only =
                fixed_lam_after >= 0 && sweep_offset > fixed_lam_after;
            copySyntheticSE3PackedSoABeliefsToGraph(
                packed,
                graph,
                num_threads,
                !all_sweeps_eta_only
            );
        }
    }
    state.valid = true;
    if (graph.profile_sync_timing) {
        graph.sync_factor_pass_sec_accum += stats.factor_pass_sec;
        graph.sync_variable_pass_sec_accum += stats.variable_pass_sec;
    }
    return true;
}

inline double se3SqNorm6Raw(const double* x) noexcept {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] +
           x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
}

inline void se3MatVec6ColMajorRaw(const double* matrix, const double* x, double* out) noexcept {
    out[0] = matrix[0] * x[0] + matrix[6] * x[1] + matrix[12] * x[2] +
             matrix[18] * x[3] + matrix[24] * x[4] + matrix[30] * x[5];
    out[1] = matrix[1] * x[0] + matrix[7] * x[1] + matrix[13] * x[2] +
             matrix[19] * x[3] + matrix[25] * x[4] + matrix[31] * x[5];
    out[2] = matrix[2] * x[0] + matrix[8] * x[1] + matrix[14] * x[2] +
             matrix[20] * x[3] + matrix[26] * x[4] + matrix[32] * x[5];
    out[3] = matrix[3] * x[0] + matrix[9] * x[1] + matrix[15] * x[2] +
             matrix[21] * x[3] + matrix[27] * x[4] + matrix[33] * x[5];
    out[4] = matrix[4] * x[0] + matrix[10] * x[1] + matrix[16] * x[2] +
             matrix[22] * x[3] + matrix[28] * x[4] + matrix[34] * x[5];
    out[5] = matrix[5] * x[0] + matrix[11] * x[1] + matrix[17] * x[2] +
             matrix[23] * x[3] + matrix[29] * x[4] + matrix[35] * x[5];
}

inline void se3StabilizeEta6Raw(double* out_eta, const double* old_eta, const double* ref_eta) noexcept {
    const double max_rel_update = se3FactorMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }
    const double old_norm = std::sqrt(se3SqNorm6Raw(old_eta));
    const double ref_norm = std::sqrt(se3SqNorm6Raw(ref_eta));
    const double eta_ref = std::max(1.0, std::max(old_norm, ref_norm));
    double diff_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double diff = out_eta[i] - old_eta[i];
        diff_sq += diff * diff;
    }
    const double rel_update = std::sqrt(diff_sq) / eta_ref;
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }
    const double step = max_rel_update / std::max(rel_update, 1e-300);
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
}

inline void se3FixedLamEtaHotUpdate(
    const double* factor_eta,
    int target_offset,
    int other_offset,
    const double* eta_map,
    const double* belief_other_eta,
    const double* old_other_eta,
    const double* old_target_eta,
    double eta_damping,
    double* out_eta
) noexcept {
    double eno[6];
    for (int i = 0; i < 6; ++i) {
        eno[i] = factor_eta[other_offset + i] + belief_other_eta[i] - old_other_eta[i];
    }

    double schur_eta[6];
    se3MatVec6ColMajorRaw(eta_map, eno, schur_eta);

    if (eta_damping == 0.0) {
        for (int i = 0; i < 6; ++i) {
            out_eta[i] = factor_eta[target_offset + i] - schur_eta[i];
        }
    } else {
        const double keep_new = 1.0 - eta_damping;
        for (int i = 0; i < 6; ++i) {
            const double proposed = factor_eta[target_offset + i] - schur_eta[i];
            out_eta[i] = keep_new * proposed + eta_damping * old_target_eta[i];
        }
    }
    se3StabilizeEta6Raw(out_eta, old_target_eta, factor_eta + target_offset);
}

bool rebuildSE3FixedLamEtaHotEntries(gbp::FactorGraph& graph) {
    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    entries.clear();
    entries.reserve(graph.factors.size());
    for (const auto& factor_ptr : graph.factors) {
        if (!factor_ptr || !factor_ptr->active) {
            continue;
        }
        gbp::FixedLamEta6HotEntry entry;
        if (factor_ptr->tryExportFixedLamEta6HotEntry(entry)) {
            entries.push_back(entry);
            continue;
        }
        if (factor_ptr->isUnaryFactor()) {
            continue;
        }
        graph.se3_fixed_lam_eta_hot_valid = false;
        entries.clear();
        return false;
    }
    graph.se3_fixed_lam_eta_hot_valid = true;
    return true;
}

bool runSE3FixedLamEtaHotFactorPass(
    gbp::FactorGraph& graph,
    int thread_count,
    double eta_damping
) {
    if (!graph.se3_fixed_lam_eta_hot_valid && !rebuildSE3FixedLamEtaHotEntries(graph)) {
        return false;
    }

    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    if (entries.empty()) {
        return false;
    }

    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
            const gbp::FixedLamEta6HotEntry& e = entries[static_cast<size_t>(i)];
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                e.msg0_eta, eta_damping, e.msg0_eta);
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                e.msg1_eta, eta_damping, e.msg1_eta);
        }
    } else {
        for (const gbp::FixedLamEta6HotEntry& e : entries) {
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                e.msg0_eta, eta_damping, e.msg0_eta);
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                e.msg1_eta, eta_damping, e.msg1_eta);
        }
    }
    ++graph.sync_fixed_lam_hot_eta_sweeps_accum;
    graph.sync_fixed_lam_hot_eta_entries = static_cast<int>(entries.size());
    return true;
}

bool runSE3FixedLamEtaHotSweepBatch(
    gbp::FactorGraph& graph,
    int thread_count,
    double eta_damping,
    int num_sweeps,
    double& factor_sec_accum,
    double& variable_sec_accum
) {
    if (num_sweeps <= 0) {
        return true;
    }
    if (!graph.se3_fixed_lam_eta_hot_valid && !rebuildSE3FixedLamEtaHotEntries(graph)) {
        return false;
    }

    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    if (entries.empty()) {
        return false;
    }

    const int num_vars = static_cast<int>(graph.var_nodes.size());
    if (thread_count > 1) {
        SteadyClock::time_point factor_t0;
        SteadyClock::time_point factor_t1;
        SteadyClock::time_point var_t0;
        SteadyClock::time_point var_t1;

        #pragma omp parallel num_threads(thread_count)
        {
            for (int sweep = 0; sweep < num_sweeps; ++sweep) {
                #pragma omp single
                {
                    factor_t0 = SteadyClock::now();
                }

                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
                    const gbp::FixedLamEta6HotEntry& e = entries[static_cast<size_t>(i)];
                    se3FixedLamEtaHotUpdate(
                        e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                        e.msg0_eta, eta_damping, e.msg0_eta);
                    se3FixedLamEtaHotUpdate(
                        e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                        e.msg1_eta, eta_damping, e.msg1_eta);
                }

                #pragma omp single
                {
                    factor_t1 = SteadyClock::now();
                    factor_sec_accum += elapsedSeconds(factor_t0, factor_t1);
                    var_t0 = factor_t1;
                }

                const bool update_mu = (sweep + 1 == num_sweeps);
                #pragma omp for schedule(static)
                for (int i = 0; i < num_vars; ++i) {
                    auto& var_ptr = graph.var_nodes[static_cast<size_t>(i)];
                    if (var_ptr && var_ptr->active) {
                        if (update_mu) {
                            var_ptr->updateBeliefEtaOnly();
                        } else {
                            var_ptr->updateBeliefEtaOnlyNoMu();
                        }
                    }
                }

                #pragma omp single
                {
                    var_t1 = SteadyClock::now();
                    variable_sec_accum += elapsedSeconds(var_t0, var_t1);
                }
            }
        }
    } else {
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const auto factor_t0 = SteadyClock::now();
            for (const gbp::FixedLamEta6HotEntry& e : entries) {
                se3FixedLamEtaHotUpdate(
                    e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                    e.msg0_eta, eta_damping, e.msg0_eta);
                se3FixedLamEtaHotUpdate(
                    e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                    e.msg1_eta, eta_damping, e.msg1_eta);
            }
            const auto factor_t1 = SteadyClock::now();
            const bool update_mu = (sweep + 1 == num_sweeps);
            for (auto& var_ptr : graph.var_nodes) {
                if (var_ptr && var_ptr->active) {
                    if (update_mu) {
                        var_ptr->updateBeliefEtaOnly();
                    } else {
                        var_ptr->updateBeliefEtaOnlyNoMu();
                    }
                }
            }
            const auto var_t1 = SteadyClock::now();
            factor_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            variable_sec_accum += elapsedSeconds(factor_t1, var_t1);
        }
    }

    graph.sync_fixed_lam_hot_eta_sweeps_accum += num_sweeps;
    graph.sync_fixed_lam_hot_eta_entries = static_cast<int>(entries.size());
    return true;
}

void manualSynchronousIterations(
    gbp::FactorGraph& graph,
    int num_sweeps,
    int num_threads,
    int sweep_offset = 0,
    int outer_index = 0,
    const SyntheticSE3Problem* packed_problem = nullptr
) {
    if (num_sweeps <= 0) {
        return;
    }
    const int thread_count = effectiveThreadCount(num_threads);
    const int fixed_lam_after = se3FixedLambdaAfterSweepsForOuter(outer_index);
    const bool allow_fixed_lam_inverse_cache = se3FixedLambdaInverseCacheForOuter(outer_index);
    if (tryManualSynchronousIterationsPackedSoA(
            graph,
            num_sweeps,
            thread_count,
            sweep_offset,
            outer_index,
            fixed_lam_after,
            packed_problem)) {
        return;
    }
    if (thread_count > 1) {
        graph.setSyncNumThreads(thread_count);
        graph.setSyncSchedule(gbp::FactorGraph::SyncScheduleKind::Static, 0);
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = sweep_offset + sweep;
            const int active_fixed_lam_start = chooseActiveFixedLambdaStart(fixed_lam_after);
            const bool use_fixed_lam =
                active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
            const bool use_eta_only_variable_update =
                active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
            const bool defer_eta_only_mu =
                use_eta_only_variable_update && (sweep + 1 < num_sweeps);
            const bool defer_full_mu =
                !use_eta_only_variable_update && (sweep + 1 < num_sweeps);
            recordSE3FixedLambdaSweepProfile(
                graph,
                fixed_lam_after,
                active_fixed_lam_start,
                use_fixed_lam,
                use_eta_only_variable_update
            );
            if (se3FixedLambdaHotEtaBatchEnabled() &&
                se3FixedLambdaHotEtaPassEnabled() &&
                use_eta_only_variable_update &&
                allow_fixed_lam_inverse_cache) {
                const int remaining_sweeps = num_sweeps - sweep;
                double batch_factor_sec = 0.0;
                double batch_variable_sec = 0.0;
                if (runSE3FixedLamEtaHotSweepBatch(
                        graph,
                        thread_count,
                        graph.eta_damping,
                        remaining_sweeps,
                        batch_factor_sec,
                        batch_variable_sec)) {
                    if (graph.profile_sync_timing) {
                        graph.sync_factor_pass_sec_accum += batch_factor_sec;
                        graph.sync_variable_pass_sec_accum += batch_variable_sec;
                        for (int extra = 1; extra < remaining_sweeps; ++extra) {
                            recordSE3FixedLambdaSweepProfile(
                                graph,
                                fixed_lam_after,
                                active_fixed_lam_start,
                                true,
                                true
                            );
                        }
                    }
                    sweep += remaining_sweeps - 1;
                    continue;
                }
            }
            std::exception_ptr first_exception = nullptr;
            int failed_index = -1;
            bool failed_in_factor_pass = false;
            SteadyClock::time_point factor_t0;
            SteadyClock::time_point factor_t1;
            SteadyClock::time_point var_t0;
            SteadyClock::time_point var_t1;

            factor_t0 = SteadyClock::now();
            const bool used_hot_eta_pass =
                se3FixedLambdaHotEtaPassEnabled() &&
                use_eta_only_variable_update &&
                allow_fixed_lam_inverse_cache &&
                runSE3FixedLamEtaHotFactorPass(graph, thread_count, graph.eta_damping);
            if (!used_hot_eta_pass) {
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int i = 0; i < static_cast<int>(graph.factors.size()); ++i) {
                    try {
                        auto& factor_ptr = graph.factors[static_cast<size_t>(i)];
                        if (factor_ptr && factor_ptr->active) {
                            if (use_fixed_lam) {
                                factor_ptr->computeMessagesFixedLam(
                                    graph.eta_damping,
                                    allow_fixed_lam_inverse_cache);
                            } else {
                                factor_ptr->computeMessages(graph.eta_damping);
                            }
                        }
                    } catch (...) {
                        #pragma omp critical(se3_gbp_sweep_exception)
                        {
                            if (!first_exception) {
                                first_exception = std::current_exception();
                                failed_index = i;
                                failed_in_factor_pass = true;
                            }
                        }
                    }
                }
            }
            factor_t1 = SteadyClock::now();

            if (!first_exception) {
                var_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int i = 0; i < static_cast<int>(graph.var_nodes.size()); ++i) {
                    try {
                        auto& var_ptr = graph.var_nodes[static_cast<size_t>(i)];
                        if (var_ptr && var_ptr->active) {
                            if (defer_eta_only_mu) {
                                var_ptr->updateBeliefEtaOnlyNoMu();
                            } else if (use_eta_only_variable_update) {
                                var_ptr->updateBeliefEtaOnly();
                            } else if (defer_full_mu) {
                                var_ptr->updateBeliefNoMu();
                            } else {
                                var_ptr->updateBelief();
                            }
                        }
                    } catch (...) {
                        #pragma omp critical(se3_gbp_sweep_exception)
                        {
                            if (!first_exception) {
                                first_exception = std::current_exception();
                                failed_index = i;
                                failed_in_factor_pass = false;
                            }
                        }
                    }
                }
                var_t1 = SteadyClock::now();
            } else {
                var_t0 = factor_t1;
                var_t1 = factor_t1;
            }
            if (first_exception) {
                try {
                    std::rethrow_exception(first_exception);
                } catch (const std::exception& e) {
                    if (failed_in_factor_pass) {
                        throw std::runtime_error(
                            "SE3 GBP parallel factor pass failed at sweep " +
                            std::to_string(sweep) + ", factor " +
                            std::to_string(failed_index) + ": " + e.what()
                        );
                    }
                    throw std::runtime_error(
                        "SE3 GBP parallel variable pass failed at sweep " +
                        std::to_string(sweep) + ", variable " +
                        std::to_string(failed_index) + ": " + e.what());
                } catch (...) {
                    if (failed_in_factor_pass) {
                        throw std::runtime_error(
                            "SE3 GBP parallel factor pass failed at sweep " +
                            std::to_string(sweep) + ", factor " +
                            std::to_string(failed_index)
                        );
                    }
                    throw std::runtime_error(
                        "SE3 GBP parallel variable pass failed at sweep " +
                        std::to_string(sweep) + ", variable " +
                        std::to_string(failed_index));
                }
            }
            if (graph.profile_sync_timing) {
                graph.sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
                graph.sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
            }
        }
        return;
    }
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        const int global_sweep = sweep_offset + sweep;
        const int active_fixed_lam_start = chooseActiveFixedLambdaStart(fixed_lam_after);
        const bool use_fixed_lam =
            active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
        const bool use_eta_only_variable_update =
            active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
        const bool defer_eta_only_mu =
            use_eta_only_variable_update && (sweep + 1 < num_sweeps);
        const bool defer_full_mu =
            !use_eta_only_variable_update && (sweep + 1 < num_sweeps);
        recordSE3FixedLambdaSweepProfile(
            graph,
            fixed_lam_after,
            active_fixed_lam_start,
            use_fixed_lam,
            use_eta_only_variable_update
        );
        const auto factor_t0 = SteadyClock::now();
        const bool used_hot_eta_pass =
            se3FixedLambdaHotEtaPassEnabled() &&
            use_eta_only_variable_update &&
            allow_fixed_lam_inverse_cache &&
            runSE3FixedLamEtaHotFactorPass(graph, thread_count, graph.eta_damping);
        if (!used_hot_eta_pass) {
            for (const auto& factor_ptr : graph.factors) {
                if (factor_ptr && factor_ptr->active) {
                    if (use_fixed_lam) {
                        factor_ptr->computeMessagesFixedLam(graph.eta_damping, allow_fixed_lam_inverse_cache);
                    } else {
                        factor_ptr->computeMessages(graph.eta_damping);
                    }
                }
            }
        }
        const auto factor_t1 = SteadyClock::now();
        const auto var_t0 = SteadyClock::now();
        for (const auto& var_ptr : graph.var_nodes) {
            if (var_ptr && var_ptr->active) {
                if (defer_eta_only_mu) {
                    var_ptr->updateBeliefEtaOnlyNoMu();
                } else if (use_eta_only_variable_update) {
                    var_ptr->updateBeliefEtaOnly();
                } else if (defer_full_mu) {
                    var_ptr->updateBeliefNoMu();
                } else {
                    var_ptr->updateBelief();
                }
            }
        }
        const auto var_t1 = SteadyClock::now();
        if (graph.profile_sync_timing) {
            graph.sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            graph.sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
        }
    }
}

void injectCorrectionKeepMessages(gbp::FactorGraph& graph, const Eigen::VectorXd& delta) {
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        gbp::VariableNode& var = *vup;
        var.refreshMu();
        var.mu.noalias() += delta.segment(offset, var.dofs);
        var.belief.setEta(var.belief.lam() * var.mu);
        var.markMuCurrent();
        offset += var.dofs;
    }
    if (offset != delta.size()) {
        throw std::runtime_error("injectCorrectionKeepMessages: delta vector size does not match graph dofs");
    }
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it != se3PackedSoAStates().end() &&
        packed_state_it->second.valid &&
        packed_state_it->second.workspace != nullptr) {
        SyntheticSE3PackedSoAWorkspace& packed = *packed_state_it->second.workspace;
        if (packed.num_vars == static_cast<int>(graph.var_nodes.size())) {
            for (int i = 0; i < packed.num_vars; ++i) {
                const gbp::VariableNode& var = *graph.var_nodes[static_cast<size_t>(i)];
                std::memcpy(
                    packed.belief_eta.data() + static_cast<size_t>(i) * 6,
                    var.belief.etaData(),
                    6 * sizeof(double)
                );
                packed.mu_valid[static_cast<size_t>(i)] = 0;
            }
        } else {
            packed_state_it->second.valid = false;
        }
    }
}

Eigen::MatrixXd buildGroupMessageConditionedInformation(
    const gbp::FactorGraph& graph,
    const std::vector<int>& group,
    const std::vector<int>& var_to_group,
    const std::vector<int>& var_to_local_offset,
    std::vector<int>& factor_stamp,
    int stamp_value
) {
    int block_dim = 0;
    for (int var_id : group) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        block_dim += var->dofs;
    }

    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(block_dim, block_dim);
    for (int var_id : group) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        const int local_offset = var_to_local_offset[var_id];
        info.block(local_offset, local_offset, var->dofs, var->dofs) += var->prior.lam();

        for (const auto& aref : var->adj_factors) {
            const gbp::Factor* factor = aref.factor;
            if (!factor || !factor->active) {
                continue;
            }

            bool factor_inside = true;
            for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                if (var_to_group[adj_var->variableID] != var_to_group[var_id]) {
                    factor_inside = false;
                    break;
                }
            }

            if (factor_inside) {
                const int fid = factor->factorID;
                if (fid >= 0 && fid < static_cast<int>(factor_stamp.size()) && factor_stamp[fid] == stamp_value) {
                    continue;
                }
                if (fid >= 0 && fid < static_cast<int>(factor_stamp.size())) {
                    factor_stamp[fid] = stamp_value;
                }

                std::vector<int> local_factor_offsets(factor->adj_var_nodes.size(), 0);
                int factor_offset = 0;
                for (int a = 0; a < static_cast<int>(factor->adj_var_nodes.size()); ++a) {
                    local_factor_offsets[a] = factor_offset;
                    factor_offset += factor->adj_var_nodes[a]->dofs;
                }

                for (int a = 0; a < static_cast<int>(factor->adj_var_nodes.size()); ++a) {
                    const gbp::VariableNode* va = factor->adj_var_nodes[a];
                    const int row_off = var_to_local_offset[va->variableID];
                    const int da = va->dofs;
                    const int factor_row = local_factor_offsets[a];
                    for (int b = 0; b < static_cast<int>(factor->adj_var_nodes.size()); ++b) {
                        const gbp::VariableNode* vb = factor->adj_var_nodes[b];
                        const int col_off = var_to_local_offset[vb->variableID];
                        const int db = vb->dofs;
                        const int factor_col = local_factor_offsets[b];
                        info.block(row_off, col_off, da, db) +=
                            factor->factor.lam().block(factor_row, factor_col, da, db);
                    }
                }
            } else {
                info.block(local_offset, local_offset, var->dofs, var->dofs) +=
                    factor->messages[aref.local_idx].lam();
            }
        }
    }

    return 0.5 * (info + info.transpose());
}

SE3BasisData buildMessageConditionedBasis(
    const gbp::FactorGraph& graph,
    const SE3PoseVector* pose_reference,
    int group_size,
    int r_reduced,
    int num_threads,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases = nullptr
) {
    SE3BasisData basis;
    basis.groups = orderedGroups(static_cast<int>(graph.var_nodes.size()), group_size);
    basis.var_to_group.assign(graph.var_nodes.size(), -1);
    basis.var_to_local_offset.assign(graph.var_nodes.size(), -1);
    basis.var_global_offset.assign(graph.var_nodes.size(), -1);
    basis.full_indices_per_group.resize(basis.groups.size());
    basis.local_bases.resize(basis.groups.size());
    basis.group_global_offsets.assign(basis.groups.size(), -1);
    basis.group_dims.assign(basis.groups.size(), 0);

    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    basis.total_dim = total_dim;

    int global_offset = 0;
    for (int var_id = 0; var_id < static_cast<int>(graph.var_nodes.size()); ++var_id) {
        const auto& vup = graph.var_nodes[var_id];
        if (!vup) {
            continue;
        }
        basis.var_global_offset[var_id] = global_offset;
        global_offset += vup->dofs;
    }

    basis.coarse_offsets.push_back(0);
    int coarse_offset = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        std::vector<int> full_indices;
        int local_offset = 0;
        int block_dim = 0;
        for (int var_id : basis.groups[g]) {
            const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
            basis.var_to_group[var_id] = g;
            basis.var_to_local_offset[var_id] = local_offset;
            const int base = basis.var_global_offset[var_id];
            for (int d = 0; d < var->dofs; ++d) {
                full_indices.push_back(base + d);
            }
            local_offset += var->dofs;
            block_dim += var->dofs;
        }
        basis.group_dims[g] = block_dim;
        bool contiguous_group = !full_indices.empty();
        for (int i = 0; i < static_cast<int>(full_indices.size()); ++i) {
            if (full_indices[static_cast<size_t>(i)] != full_indices.front() + i) {
                contiguous_group = false;
                break;
            }
        }
        if (contiguous_group) {
            basis.group_global_offsets[g] = full_indices.front();
        }
        basis.full_indices_per_group[g] = std::move(full_indices);
        const int r_local = std::min(std::max(1, r_reduced), block_dim);
        coarse_offset += r_local;
        basis.coarse_offsets.push_back(coarse_offset);
    }
    basis.coarse_dim = coarse_offset;

    struct BasisThreadStats {
        double block_build_sec = 0.0;
        double eigensolver_sec = 0.0;
        double copyout_sec = 0.0;
        double min_eigenvalue = std::numeric_limits<double>::infinity();
        double max_eigenvalue = -std::numeric_limits<double>::infinity();
        int negative_group_count = 0;
        int nonpositive_group_count = 0;
        int partial_attempt_count = 0;
        int partial_converged_count = 0;
        int full_eigensolver_count = 0;
    };

    PartialSymmetricEigenOptions partial_opts{};
    partial_opts.max_iters = std::max(
        1,
        se3EnvIntOrDefault("GBP_SE3_PARTIAL_BASIS_MAX_ITERS", partial_opts.max_iters)
    );
    partial_opts.residual_check_period = std::max(
        1,
        se3EnvIntOrDefault(
            "GBP_SE3_PARTIAL_BASIS_RESIDUAL_CHECK_PERIOD",
            partial_opts.residual_check_period
        )
    );
    partial_opts.residual_tol = se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_RESIDUAL_TOL",
        partial_opts.residual_tol
    );
    const bool allow_partial_eigensolver = se3PartialBasisEigenEnabled();
    const bool accept_unconverged_partial = se3PartialBasisAcceptUnconverged();
    double accept_partial_residual_tol = se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_ACCEPT_RESIDUAL_TOL",
        -1.0
    );
    if (accept_partial_residual_tol < 0.0 &&
        static_cast<int>(basis.groups.size()) >= se3PartialBasisLargeGroupMinGroups()) {
        accept_partial_residual_tol = se3PartialBasisLargeGroupAcceptResidualTol();
    }
    if (accept_partial_residual_tol < 0.0) {
        accept_partial_residual_tol = se3PartialBasisSmallGroupAcceptResidualTol();
    }

    auto process_group = [&](
        int g,
        std::vector<int>& factor_stamp,
        int& stamp_value,
        BasisThreadStats& stats,
        PartialSymmetricEigenWorkspace& partial_ws
    ) {
        const auto block_t0 = SteadyClock::now();
        Eigen::MatrixXd block = buildGroupMessageConditionedInformation(
            graph,
            basis.groups[g],
            basis.var_to_group,
            basis.var_to_local_offset,
            factor_stamp,
            ++stamp_value
        );
        const auto block_t1 = SteadyClock::now();
        stats.block_build_sec += elapsedSeconds(block_t0, block_t1);

        const int block_dim = static_cast<int>(block.rows());
        const int r_local = basis.coarse_offsets[g + 1] - basis.coarse_offsets[g];
        Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
        if (r_local < block_dim) {
            Eigen::MatrixXd spectral_candidates(block_dim, 0);
            bool built_candidates = false;

            const Eigen::MatrixXd* warm_basis =
                    (allow_partial_eigensolver &&
                     warm_start_local_bases != nullptr &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() > 0)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                const bool try_partial =
                    allow_partial_eigensolver &&
                    r_local < block_dim &&
                    block_dim >= std::max(24, 2 * r_local);

                Eigen::VectorXd evals;
                Eigen::MatrixXd evecs;
                bool eig_ok = false;

                if (try_partial) {
                    ++stats.partial_attempt_count;
                    const auto eig_t0 = SteadyClock::now();
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
                    const auto eig_t1 = SteadyClock::now();
                    stats.eigensolver_sec += elapsedSeconds(eig_t0, eig_t1);
                    if (partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local &&
                        partial.eigenvalues.size() == r_local &&
                        (partial.converged ||
                         accept_unconverged_partial ||
                         (accept_partial_residual_tol >= 0.0 &&
                          partial.max_relative_residual <= accept_partial_residual_tol))) {
                        evals = partial.eigenvalues;
                        evecs = partial.eigenvectors;
                        eig_ok = true;
                        if (partial.converged) {
                            ++stats.partial_converged_count;
                        }
                    }
                }

                if (!eig_ok) {
                    ++stats.full_eigensolver_count;
                    const auto eig_t0 = SteadyClock::now();
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(block);
                    const auto eig_t1 = SteadyClock::now();
                    stats.eigensolver_sec += elapsedSeconds(eig_t0, eig_t1);
                    if (eig.info() == Eigen::Success) {
                        evals = eig.eigenvalues();
                        evecs = eig.eigenvectors();
                        eig_ok = true;
                    }
                }

                if (eig_ok) {
                    const double min_eval = evals(0);
                    const double max_eval = evals(evals.size() - 1);
                    stats.min_eigenvalue = std::min(stats.min_eigenvalue, min_eval);
                    stats.max_eigenvalue = std::max(stats.max_eigenvalue, max_eval);
                    if (min_eval < -1e-10) {
                        ++stats.negative_group_count;
                    }
                    if (min_eval <= 1e-12) {
                        ++stats.nonpositive_group_count;
                    }

                    spectral_candidates.resize(block_dim, evals.size());
                    int spec_cols = 0;
                    for (int i = 0; i < evals.size(); ++i) {
                        spectral_candidates.col(spec_cols++) = evecs.col(i);
                    }
                    spectral_candidates.conservativeResize(block_dim, spec_cols);
                    built_candidates = true;
                }

            if (built_candidates) {
                if (spectral_candidates.cols() >= r_local) {
                    local_basis = spectral_candidates.leftCols(r_local);
                } else if (spectral_candidates.cols() > 0) {
                    local_basis.setZero(block_dim, r_local);
                    local_basis.leftCols(spectral_candidates.cols()) = spectral_candidates;
                }
            }
        }
        const auto copy_t0 = SteadyClock::now();
        basis.local_bases[g] = std::move(local_basis);
        const auto copy_t1 = SteadyClock::now();
        stats.copyout_sec += elapsedSeconds(copy_t0, copy_t1);
    };

    const int thread_count = effectiveThreadCount(num_threads);
    const bool do_parallel =
        thread_count > 1 && static_cast<int>(basis.groups.size()) >= std::max(8, thread_count * 2);

    if (!do_parallel) {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        BasisThreadStats stats;
        PartialSymmetricEigenWorkspace partial_ws;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            process_group(g, factor_stamp, stamp_value, stats, partial_ws);
        }
        basis.block_build_sec = stats.block_build_sec;
        basis.eigensolver_sec = stats.eigensolver_sec;
        basis.copyout_sec = stats.copyout_sec;
        basis.min_eigenvalue = stats.min_eigenvalue;
        basis.max_eigenvalue = stats.max_eigenvalue;
        basis.negative_group_count = stats.negative_group_count;
        basis.nonpositive_group_count = stats.nonpositive_group_count;
        basis.partial_attempt_count = stats.partial_attempt_count;
        basis.partial_converged_count = stats.partial_converged_count;
        basis.full_eigensolver_count = stats.full_eigensolver_count;
        return basis;
    }

    std::vector<std::vector<int>> thread_factor_stamps(
        static_cast<size_t>(thread_count),
        std::vector<int>(graph.factors.size(), -1)
    );
    std::vector<int> thread_stamp_values(static_cast<size_t>(thread_count), 0);
    std::vector<BasisThreadStats> thread_stats(static_cast<size_t>(thread_count));
    std::vector<PartialSymmetricEigenWorkspace> thread_partial_ws(static_cast<size_t>(thread_count));

    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const int tid = omp_get_thread_num();
        process_group(
            g,
            thread_factor_stamps[tid],
            thread_stamp_values[tid],
            thread_stats[tid],
            thread_partial_ws[tid]
        );
    }

    for (const BasisThreadStats& stats : thread_stats) {
        basis.block_build_sec += stats.block_build_sec;
        basis.eigensolver_sec += stats.eigensolver_sec;
        basis.copyout_sec += stats.copyout_sec;
        basis.min_eigenvalue = std::min(basis.min_eigenvalue, stats.min_eigenvalue);
        basis.max_eigenvalue = std::max(basis.max_eigenvalue, stats.max_eigenvalue);
        basis.negative_group_count += stats.negative_group_count;
        basis.nonpositive_group_count += stats.nonpositive_group_count;
        basis.partial_attempt_count += stats.partial_attempt_count;
        basis.partial_converged_count += stats.partial_converged_count;
        basis.full_eigensolver_count += stats.full_eigensolver_count;
    }

    return basis;
}

void restrictToCoarseInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse
);

void prolongToFineInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
);

void restrictToCoarseInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse
) {
    if (coarse.size() != basis.coarse_dim) {
        coarse.resize(basis.coarse_dim);
    }
    coarse.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const int off = basis.coarse_offsets[g];
        const int dim = basis.local_bases[g].cols();
        const int group_dim =
            (g < static_cast<int>(basis.group_dims.size()))
                ? basis.group_dims[static_cast<size_t>(g)]
                : static_cast<int>(basis.full_indices_per_group[g].size());
        const int fine_off =
            (g < static_cast<int>(basis.group_global_offsets.size()))
                ? basis.group_global_offsets[static_cast<size_t>(g)]
                : -1;
        if (fine_off >= 0 && fine_off + group_dim <= fine_vec.size()) {
            coarse.segment(off, dim).noalias() =
                basis.local_bases[g].transpose() * fine_vec.segment(fine_off, group_dim);
        } else {
            const std::vector<int>& rows = basis.full_indices_per_group[g];
            Eigen::VectorXd fine_local(rows.size());
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine_local(i) = fine_vec(rows[i]);
            }
            coarse.segment(off, dim).noalias() =
                basis.local_bases[g].transpose() * fine_local;
        }
    }
}

void prolongToFineInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
) {
    if (fine.size() != basis.total_dim) {
        fine.resize(basis.total_dim);
    }
    fine.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const int off = basis.coarse_offsets[g];
        const int dim = basis.local_bases[g].cols();
        const int group_dim =
            (g < static_cast<int>(basis.group_dims.size()))
                ? basis.group_dims[static_cast<size_t>(g)]
                : static_cast<int>(basis.full_indices_per_group[g].size());
        const int fine_off =
            (g < static_cast<int>(basis.group_global_offsets.size()))
                ? basis.group_global_offsets[static_cast<size_t>(g)]
                : -1;
        if (fine_off >= 0 && fine_off + group_dim <= fine.size()) {
            fine.segment(fine_off, group_dim).noalias() =
                basis.local_bases[g] * coarse_vec.segment(off, dim);
        } else {
            const std::vector<int>& rows = basis.full_indices_per_group[g];
            const Eigen::VectorXd fine_local =
                basis.local_bases[g] * coarse_vec.segment(off, dim);
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine(rows[i]) = fine_local(i);
            }
        }
    }
}

struct CoarseLambdaAssemblyCache {
    struct BlockMeta {
        int gi = 0;
        int gj = 0;
        int row_offset = 0;
        int col_offset = 0;
        int rows = 0;
        int cols = 0;
        std::vector<int> value_indices;
    };

    struct PriorPlan {
        int variable_index = -1;
        int block_index = -1;
        int group = -1;
        int local_offset = 0;
        int dofs = 0;
    };

    struct FactorBlockPlan {
        int block_index = -1;
        int ga = -1;
        int gb = -1;
        int la = 0;
        int lb = 0;
        int da = 0;
        int db = 0;
        int factor_row = 0;
        int factor_col = 0;
    };

    struct FactorPlan {
        int factor_index = -1;
        std::vector<FactorBlockPlan> blocks;
    };

    struct SymBlockPair {
        int first = -1;
        int second = -1;
    };

    int num_groups = 0;
    int coarse_dim = 0;
    int num_vars = 0;
    int num_factors = 0;
    std::vector<BlockMeta> blocks;
    std::vector<PriorPlan> prior_plans;
    std::vector<FactorPlan> factor_plans;
    std::vector<int> diagonal_block_indices;
    std::vector<SymBlockPair> sym_block_pairs;
    std::vector<Eigen::MatrixXd> block_values;
    std::vector<std::vector<Eigen::MatrixXd>> tls_block_values;
    std::vector<std::vector<int>> tls_touched_block_indices;
    std::vector<std::vector<unsigned char>> tls_block_touched;
    int tls_thread_count = 0;
    Eigen::SparseMatrix<double> lam;
    Eigen::SparseMatrix<double> sym_lam;
    std::unordered_map<long long, int> coeff_to_value_index;
    bool valid = false;
};

long long coarseBlockPairKey(int gi, int gj) {
    return (static_cast<long long>(gi) << 32) ^ static_cast<unsigned int>(gj);
}

void buildCoarseLambdaAssemblyCache(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    CoarseLambdaAssemblyCache& cache
) {
    cache = CoarseLambdaAssemblyCache{};
    cache.num_groups = static_cast<int>(basis.local_bases.size());
    cache.coarse_dim = basis.coarse_dim;
    cache.num_vars = static_cast<int>(graph.var_nodes.size());
    cache.num_factors = static_cast<int>(graph.factors.size());

    std::unordered_map<long long, int> block_index_by_pair;
    block_index_by_pair.reserve(graph.factors.size() * 4 + basis.local_bases.size());

    auto ensure_block = [&](int gi, int gj) -> int {
        const long long key = coarseBlockPairKey(gi, gj);
        const auto it = block_index_by_pair.find(key);
        if (it != block_index_by_pair.end()) {
            return it->second;
        }
        const int index = static_cast<int>(cache.blocks.size());
        CoarseLambdaAssemblyCache::BlockMeta meta;
        meta.gi = gi;
        meta.gj = gj;
        meta.row_offset = basis.coarse_offsets[gi];
        meta.col_offset = basis.coarse_offsets[gj];
        meta.rows = basis.coarse_offsets[gi + 1] - basis.coarse_offsets[gi];
        meta.cols = basis.coarse_offsets[gj + 1] - basis.coarse_offsets[gj];
        cache.blocks.push_back(meta);
        block_index_by_pair.emplace(key, index);
        return index;
    };

    cache.prior_plans.reserve(graph.var_nodes.size());
    for (int vi = 0; vi < static_cast<int>(graph.var_nodes.size()); ++vi) {
        const auto& v = graph.var_nodes[static_cast<size_t>(vi)];
        if (!v) {
            continue;
        }
        const int g = basis.var_to_group[v->variableID];
        if (g < 0) {
            continue;
        }
        cache.prior_plans.push_back(CoarseLambdaAssemblyCache::PriorPlan{
            vi,
            ensure_block(g, g),
            g,
            basis.var_to_local_offset[v->variableID],
            v->dofs
        });
    }

    cache.factor_plans.reserve(graph.factors.size());
    for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
        const gbp::Factor* factor = graph.factors[static_cast<size_t>(fi)].get();
        if (factor == nullptr) {
            continue;
        }
        CoarseLambdaAssemblyCache::FactorPlan factor_plan;
        factor_plan.factor_index = fi;
        const int k = static_cast<int>(factor->adj_var_nodes.size());
        std::vector<int> local_factor_offsets(static_cast<size_t>(k), 0);
        int factor_offset = 0;
        for (int a = 0; a < k; ++a) {
            local_factor_offsets[static_cast<size_t>(a)] = factor_offset;
            factor_offset += factor->adj_var_nodes[static_cast<size_t>(a)]->dofs;
        }

        factor_plan.blocks.reserve(static_cast<size_t>(k * k));
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = factor->adj_var_nodes[static_cast<size_t>(a)];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const int off_a = local_factor_offsets[static_cast<size_t>(a)];
            for (int b = 0; b < k; ++b) {
                const gbp::VariableNode* vb = factor->adj_var_nodes[static_cast<size_t>(b)];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const int off_b = local_factor_offsets[static_cast<size_t>(b)];
                factor_plan.blocks.push_back(CoarseLambdaAssemblyCache::FactorBlockPlan{
                    ensure_block(ga, gb),
                    ga,
                    gb,
                    la,
                    lb,
                    da,
                    db,
                    off_a,
                    off_b
                });
            }
        }
        cache.factor_plans.push_back(std::move(factor_plan));
    }

    cache.diagonal_block_indices.reserve(cache.blocks.size());
    cache.sym_block_pairs.reserve(cache.blocks.size() / 2);
    for (int bi = 0; bi < static_cast<int>(cache.blocks.size()); ++bi) {
        const CoarseLambdaAssemblyCache::BlockMeta& meta = cache.blocks[static_cast<size_t>(bi)];
        if (meta.gi == meta.gj) {
            cache.diagonal_block_indices.push_back(bi);
        } else if (meta.gi < meta.gj) {
            const auto transpose_it =
                block_index_by_pair.find(coarseBlockPairKey(meta.gj, meta.gi));
            if (transpose_it != block_index_by_pair.end()) {
                cache.sym_block_pairs.push_back(
                    CoarseLambdaAssemblyCache::SymBlockPair{bi, transpose_it->second}
                );
            }
        }
    }

    size_t triplet_count = 0;
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        triplet_count += static_cast<size_t>(meta.rows) * static_cast<size_t>(meta.cols);
    }
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(triplet_count);
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                trips.emplace_back(meta.row_offset + r, meta.col_offset + c, 1.0);
            }
        }
    }
    cache.lam.resize(cache.coarse_dim, cache.coarse_dim);
    cache.lam.setFromTriplets(trips.begin(), trips.end());
    cache.lam.makeCompressed();

    cache.coeff_to_value_index.reserve(static_cast<size_t>(cache.lam.nonZeros()) * 2);
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            cache.coeff_to_value_index.emplace(sparseCoeffKey(row, col), p);
        }
    }
    for (CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        meta.value_indices.clear();
        meta.value_indices.reserve(static_cast<size_t>(meta.rows) * static_cast<size_t>(meta.cols));
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                const auto it = cache.coeff_to_value_index.find(
                    sparseCoeffKey(meta.row_offset + r, meta.col_offset + c)
                );
                if (it == cache.coeff_to_value_index.end()) {
                    throw std::runtime_error("cached coarse lambda pattern missing sparse coefficient");
                }
                meta.value_indices.push_back(it->second);
            }
        }
    }

    cache.block_values.clear();
    cache.block_values.reserve(cache.blocks.size());
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        cache.block_values.emplace_back(Eigen::MatrixXd::Zero(meta.rows, meta.cols));
    }

    cache.valid = true;
}

bool coarseLambdaAssemblyCacheMatches(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache& cache
) {
    return cache.valid &&
        cache.num_groups == static_cast<int>(basis.local_bases.size()) &&
        cache.coarse_dim == basis.coarse_dim &&
        cache.num_vars == static_cast<int>(graph.var_nodes.size()) &&
        cache.num_factors == static_cast<int>(graph.factors.size());
}

std::vector<Eigen::MatrixXd> makeZeroCoarseBlocks(
    const CoarseLambdaAssemblyCache& cache
) {
    std::vector<Eigen::MatrixXd> blocks;
    blocks.reserve(cache.blocks.size());
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        blocks.emplace_back(Eigen::MatrixXd::Zero(meta.rows, meta.cols));
    }
    return blocks;
}

void zeroCoarseBlockValues(std::vector<Eigen::MatrixXd>& blocks) {
    for (Eigen::MatrixXd& block : blocks) {
        block.setZero();
    }
}

void ensureCoarseTlsBlockValues(
    const CoarseLambdaAssemblyCache& cache,
    int thread_count,
    std::vector<std::vector<Eigen::MatrixXd>>& tls_blocks,
    std::vector<std::vector<int>>& tls_touched_indices,
    std::vector<std::vector<unsigned char>>& tls_touched_flags,
    int& tls_thread_count
) {
    if (tls_thread_count == thread_count &&
        static_cast<int>(tls_blocks.size()) == thread_count &&
        static_cast<int>(tls_touched_indices.size()) == thread_count &&
        static_cast<int>(tls_touched_flags.size()) == thread_count) {
        bool valid = true;
        for (int tid = 0; tid < thread_count; ++tid) {
            std::vector<Eigen::MatrixXd>& local_blocks = tls_blocks[static_cast<size_t>(tid)];
            if (local_blocks.size() != cache.block_values.size()) {
                valid = false;
                break;
            }
            if (tls_touched_flags[static_cast<size_t>(tid)].size() != cache.block_values.size()) {
                valid = false;
                break;
            }
        }
        if (valid) {
            for (int tid = 0; tid < thread_count; ++tid) {
                std::vector<Eigen::MatrixXd>& local_blocks = tls_blocks[static_cast<size_t>(tid)];
                std::vector<int>& touched = tls_touched_indices[static_cast<size_t>(tid)];
                std::vector<unsigned char>& flags = tls_touched_flags[static_cast<size_t>(tid)];
                for (int block_index : touched) {
                    local_blocks[static_cast<size_t>(block_index)].setZero();
                    flags[static_cast<size_t>(block_index)] = 0;
                }
                touched.clear();
            }
            return;
        }
    }

    tls_blocks.clear();
    tls_touched_indices.clear();
    tls_touched_flags.clear();
    tls_blocks.reserve(static_cast<size_t>(thread_count));
    tls_touched_indices.resize(static_cast<size_t>(thread_count));
    tls_touched_flags.reserve(static_cast<size_t>(thread_count));
    for (int tid = 0; tid < thread_count; ++tid) {
        tls_blocks.push_back(makeZeroCoarseBlocks(cache));
        tls_touched_flags.emplace_back(cache.block_values.size(), 0);
    }
    tls_thread_count = thread_count;
}

void addPriorPlanToCoarseBlocks(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache::PriorPlan& plan,
    std::vector<Eigen::MatrixXd>& blocks
) {
    const gbp::VariableNode* var =
        graph.var_nodes[static_cast<size_t>(plan.variable_index)].get();
    if (var == nullptr) {
        return;
    }
    const auto B = basis.local_bases[plan.group].middleRows(plan.local_offset, plan.dofs);
    blocks[static_cast<size_t>(plan.block_index)].noalias() +=
        B.transpose() * var->prior.lam() * B;
}

void addFactorPlanToCoarseBlocks(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache::FactorPlan& plan,
    std::vector<Eigen::MatrixXd>& blocks,
    std::vector<int>* touched_indices = nullptr,
    std::vector<unsigned char>* touched_flags = nullptr
) {
    if (plan.factor_index < 0 ||
        plan.factor_index >= static_cast<int>(graph.factors.size())) {
        return;
    }
    const gbp::Factor* factor =
        graph.factors[static_cast<size_t>(plan.factor_index)].get();
    if (factor == nullptr || !factor->active) {
        return;
    }
    const auto factor_lam = factor->factor.lam();
    for (const CoarseLambdaAssemblyCache::FactorBlockPlan& block_plan : plan.blocks) {
        if (touched_indices != nullptr && touched_flags != nullptr) {
            const size_t block_index = static_cast<size_t>(block_plan.block_index);
            if ((*touched_flags)[block_index] == 0) {
                (*touched_flags)[block_index] = 1;
                touched_indices->push_back(block_plan.block_index);
            }
        }
        const auto Ba =
            basis.local_bases[block_plan.ga].middleRows(block_plan.la, block_plan.da);
        const auto Bb =
            basis.local_bases[block_plan.gb].middleRows(block_plan.lb, block_plan.db);
        blocks[static_cast<size_t>(block_plan.block_index)].noalias() +=
            Ba.transpose() *
            factor_lam.block(
                block_plan.factor_row,
                block_plan.factor_col,
                block_plan.da,
                block_plan.db
            ) *
            Bb;
    }
}

const Eigen::SparseMatrix<double>& assembleCoarseLambdaCachedInPlace(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    CoarseLambdaAssemblyCache& cache,
    int num_threads
) {
    if (!coarseLambdaAssemblyCacheMatches(graph, basis, cache)) {
        buildCoarseLambdaAssemblyCache(graph, basis, cache);
    }

    zeroCoarseBlockValues(cache.block_values);
    std::fill(
        cache.lam.valuePtr(),
        cache.lam.valuePtr() + cache.lam.nonZeros(),
        0.0
    );

    for (const CoarseLambdaAssemblyCache::PriorPlan& plan : cache.prior_plans) {
        addPriorPlanToCoarseBlocks(graph, basis, plan, cache.block_values);
    }

    if (cache.factor_plans.size() > 64) {
        const int thread_count = effectiveThreadCount(num_threads);
        const int shard_count = std::min(
            kSE3CoarseAssemblyShards,
            static_cast<int>(cache.factor_plans.size())
        );
        const bool use_touched_tls =
            static_cast<int>(cache.factor_plans.size()) >= se3CoarseTouchedTlsMinFactors();
        ensureCoarseTlsBlockValues(
            cache,
            shard_count,
            cache.tls_block_values,
            cache.tls_touched_block_indices,
            cache.tls_block_touched,
            cache.tls_thread_count
        );
        if (!use_touched_tls) {
            for (std::vector<Eigen::MatrixXd>& local_blocks : cache.tls_block_values) {
                zeroCoarseBlockValues(local_blocks);
            }
        }

        auto process_shard = [&](int shard) {
            const int plan_count = static_cast<int>(cache.factor_plans.size());
            const int plans_per_shard = plan_count / shard_count;
            const int remainder = plan_count % shard_count;
            const int begin =
                shard * plans_per_shard + std::min(shard, remainder);
            const int end =
                begin + plans_per_shard + (shard < remainder ? 1 : 0);
            for (int pi = begin; pi < end; ++pi) {
                if (use_touched_tls) {
                    std::vector<int>& touched_indices =
                        cache.tls_touched_block_indices[static_cast<size_t>(shard)];
                    std::vector<unsigned char>& touched_flags =
                        cache.tls_block_touched[static_cast<size_t>(shard)];
                    addFactorPlanToCoarseBlocks(
                        graph,
                        basis,
                        cache.factor_plans[static_cast<size_t>(pi)],
                        cache.tls_block_values[static_cast<size_t>(shard)],
                        &touched_indices,
                        &touched_flags
                    );
                } else {
                    addFactorPlanToCoarseBlocks(
                        graph,
                        basis,
                        cache.factor_plans[static_cast<size_t>(pi)],
                        cache.tls_block_values[static_cast<size_t>(shard)]
                    );
                }
            }
        };

        if (thread_count > 1) {
            #pragma omp parallel for schedule(static) num_threads(thread_count)
            for (int shard = 0; shard < shard_count; ++shard) {
                process_shard(shard);
            }
        } else {
            for (int shard = 0; shard < shard_count; ++shard) {
                process_shard(shard);
            }
        }

        if (use_touched_tls) {
            for (int shard = 0; shard < shard_count; ++shard) {
                const std::vector<Eigen::MatrixXd>& local_blocks =
                    cache.tls_block_values[static_cast<size_t>(shard)];
                const std::vector<int>& touched_indices =
                    cache.tls_touched_block_indices[static_cast<size_t>(shard)];
                for (int bi : touched_indices) {
                    cache.block_values[static_cast<size_t>(bi)] += local_blocks[static_cast<size_t>(bi)];
                }
            }
        } else {
            for (int shard = 0; shard < shard_count; ++shard) {
                const std::vector<Eigen::MatrixXd>& local_blocks =
                    cache.tls_block_values[static_cast<size_t>(shard)];
                for (int bi = 0; bi < static_cast<int>(cache.block_values.size()); ++bi) {
                    cache.block_values[static_cast<size_t>(bi)] += local_blocks[static_cast<size_t>(bi)];
                }
            }
        }
    } else {
        for (const CoarseLambdaAssemblyCache::FactorPlan& plan : cache.factor_plans) {
            addFactorPlanToCoarseBlocks(graph, basis, plan, cache.block_values);
        }
    }

    const bool inplace_sym = se3CachedCoarseInplaceSymmetrizeEnabled();
    if (inplace_sym) {
        for (int block_index : cache.diagonal_block_indices) {
            Eigen::MatrixXd& block = cache.block_values[static_cast<size_t>(block_index)];
            block = 0.5 * (block + Eigen::MatrixXd(block.transpose()));
        }
        for (const CoarseLambdaAssemblyCache::SymBlockPair& pair : cache.sym_block_pairs) {
            Eigen::MatrixXd avg =
                0.5 * (
                    cache.block_values[static_cast<size_t>(pair.first)] +
                    Eigen::MatrixXd(cache.block_values[static_cast<size_t>(pair.second)].transpose())
                );
            cache.block_values[static_cast<size_t>(pair.first)] = avg;
            cache.block_values[static_cast<size_t>(pair.second)] = avg.transpose();
        }
    }

    double* values = cache.lam.valuePtr();
    for (int bi = 0; bi < static_cast<int>(cache.blocks.size()); ++bi) {
        const CoarseLambdaAssemblyCache::BlockMeta& meta = cache.blocks[static_cast<size_t>(bi)];
        const Eigen::MatrixXd& block = cache.block_values[static_cast<size_t>(bi)];
        size_t value_ix = 0;
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                values[meta.value_indices[value_ix++]] = block(r, c);
            }
        }
    }

    if (inplace_sym) {
        return cache.lam;
    }
    cache.sym_lam = symmetrizeSparse(cache.lam);
    return cache.sym_lam;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirect(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    bool use_parallel,
    int num_threads
) {
    const int num_groups = static_cast<int>(basis.local_bases.size());
    const int num_block_pairs = num_groups * num_groups;
    std::vector<Eigen::MatrixXd> blocks(num_block_pairs);
    std::vector<char> active(num_block_pairs, 0);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int g = basis.var_to_group[var.variableID];
        const int l = basis.var_to_local_offset[var.variableID];
        const auto B = basis.local_bases[g].middleRows(l, var.dofs);
        const int block_index = g * num_groups + g;
        if (!active[block_index]) {
            blocks[block_index] = Eigen::MatrixXd::Zero(B.cols(), B.cols());
            active[block_index] = 1;
        }
        blocks[block_index].noalias() += B.transpose() * var.prior.lam() * B;
    }

    auto accumulate_factor = [&](const gbp::Factor& factor, std::vector<Eigen::MatrixXd>& target_blocks, std::vector<char>& target_active) {
        if (!factor.active) {
            return;
        }
        std::vector<int> local_factor_offsets(factor.adj_var_nodes.size(), 0);
        int factor_offset = 0;
        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            local_factor_offsets[a] = factor_offset;
            factor_offset += factor.adj_var_nodes[a]->dofs;
        }

        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const auto Ba = basis.local_bases[ga].middleRows(la, da);
            const int off_a = local_factor_offsets[a];

            for (int b = 0; b < static_cast<int>(factor.adj_var_nodes.size()); ++b) {
                const gbp::VariableNode* vb = factor.adj_var_nodes[b];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const auto Bb = basis.local_bases[gb].middleRows(lb, db);
                const int off_b = local_factor_offsets[b];
                const int block_index = ga * num_groups + gb;

                if (!target_active[block_index]) {
                    target_blocks[block_index] = Eigen::MatrixXd::Zero(Ba.cols(), Bb.cols());
                    target_active[block_index] = 1;
                }

                target_blocks[block_index].noalias() +=
                    Ba.transpose() *
                    factor.factor.lam().block(off_a, off_b, da, db) *
                    Bb;
            }
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = effectiveThreadCount(num_threads);
        std::vector<std::vector<Eigen::MatrixXd>> tls_blocks(static_cast<size_t>(thread_count));
        std::vector<std::vector<char>> tls_active(static_cast<size_t>(thread_count));
        for (int tid = 0; tid < thread_count; ++tid) {
            tls_blocks[tid].resize(num_block_pairs);
            tls_active[tid].assign(num_block_pairs, 0);
        }

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, tls_blocks[tid], tls_active[tid]);
        }

        for (int tid = 0; tid < thread_count; ++tid) {
            for (int block_index = 0; block_index < num_block_pairs; ++block_index) {
                if (!tls_active[tid][block_index]) {
                    continue;
                }
                if (!active[block_index]) {
                    blocks[block_index] = std::move(tls_blocks[tid][block_index]);
                    active[block_index] = 1;
                } else {
                    blocks[block_index] += tls_blocks[tid][block_index];
                }
            }
        }
    } else {
        for (const auto& fup : graph.factors) {
            const gbp::Factor* factor = fup.get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, blocks, active);
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    for (int gi = 0; gi < num_groups; ++gi) {
        const int row_off = basis.coarse_offsets[gi];
        for (int gj = 0; gj < num_groups; ++gj) {
            const int block_index = gi * num_groups + gj;
            if (!active[block_index]) {
                continue;
            }
            const int col_off = basis.coarse_offsets[gj];
            const Eigen::MatrixXd& block = blocks[block_index];
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    const double value = block(r, c);
                    if (value != 0.0) {
                        trips.emplace_back(row_off + r, col_off + c, value);
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> coarse_lam(basis.coarse_dim, basis.coarse_dim);
    coarse_lam.setFromTriplets(trips.begin(), trips.end());
    coarse_lam.makeCompressed();
    return symmetrizeSparse(coarse_lam);
}

std::vector<Eigen::VectorXd> emptyMeasurementBlocks(const Eigen::VectorXd&) {
    return {};
}

std::vector<Eigen::MatrixXd> emptyJacobianBlocks(const Eigen::VectorXd&) {
    return {};
}

SyntheticSE3ResidualGraphWorkspace buildSyntheticSE3ResidualGraphWorkspace(
    const SyntheticSE3Problem& problem,
    double tiny_prior
) {
    SyntheticSE3ResidualGraphWorkspace workspace;
    workspace.tiny_prior = tiny_prior;
    workspace.graph.nonlinear_factors = false;
    workspace.graph.eta_damping = se3MessageDamping();

    const int n = static_cast<int>(problem.gt_poses.size());
    workspace.graph.var_nodes.reserve(n);
    workspace.graph.var_residual.reserve(n);
    workspace.graph.factors.reserve(problem.edges.size() + 1);
    workspace.vars.resize(n, nullptr);
    workspace.edge_factors.resize(problem.edges.size(), nullptr);

    std::vector<int> variable_degree(static_cast<size_t>(n), 0);
    for (const SyntheticSE3Edge& edge : problem.edges) {
        ++variable_degree[edge.i];
        ++variable_degree[edge.j];
    }
    if (!variable_degree.empty()) {
        ++variable_degree[0];
    }

    const Mat6 weak_lam = tiny_prior * Mat6::Identity();
    for (int i = 0; i < n; ++i) {
        gbp::VariableNode* var = workspace.graph.addVariable(i, 6);
        var->GT = Eigen::VectorXd::Zero(6);
        var->adj_factors.reserve(variable_degree[static_cast<size_t>(i)]);
        var->adj_factors_raw.reserve(variable_degree[static_cast<size_t>(i)]);
        var->prior.setLam(weak_lam);
        var->prior.setEta(Eigen::VectorXd::Zero(6));
        var->belief.setLam(weak_lam);
        var->belief.setEta(Eigen::VectorXd::Zero(6));
        var->mu = Eigen::VectorXd::Zero(6);
        var->markMuCurrent();
        workspace.vars[i] = var;
    }

    int fid = 0;
    for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
        const SyntheticSE3Edge& edge = problem.edges[edge_index];
        gbp::VariableNode* vi = workspace.vars.at(edge.i);
        gbp::VariableNode* vj = workspace.vars.at(edge.j);
        gbp::Factor* factor = workspace.graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vi, vj},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        workspace.graph.connect(factor, vi, 0);
        workspace.graph.connect(factor, vj, 1);
        workspace.edge_factors[edge_index] = factor;
    }

    if (!workspace.vars.empty()) {
        gbp::VariableNode* v0 = workspace.vars.at(0);
        workspace.anchor_factor = workspace.graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{v0},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        workspace.graph.connect(workspace.anchor_factor, v0, 0);
    }

    return workspace;
}

void resetSyntheticSE3ResidualState(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    int num_threads
) {
    const int n = static_cast<int>(workspace.vars.size());
    const bool use_parallel = (num_threads != 1) && (n > 64);

    auto reset_var = [&](int i) {
        gbp::VariableNode* var = workspace.vars[i];
        if (!var) {
            return;
        }
        const int d = var->dofs;
        std::memset(var->belief.etaData(), 0, static_cast<size_t>(d) * sizeof(double));
        std::memcpy(var->belief.lamData(), var->prior.lamData(), static_cast<size_t>(d * d) * sizeof(double));
        var->mu.setZero();
        var->markMuCurrent();
    };

    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            reset_var(i);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            reset_var(i);
        }
    }
}

void clearGaussian(utils::NdimGaussian& gaussian) {
    const int d = gaussian.dim();
    if (d <= 0) {
        return;
    }
    std::memset(gaussian.etaData(), 0, static_cast<size_t>(d) * sizeof(double));
    std::memset(gaussian.lamData(), 0, static_cast<size_t>(d * d) * sizeof(double));
}

bool transportGaussianBetweenSE3Linearizations(
    utils::NdimGaussian& gaussian,
    const Vec6& old_to_new_delta
) {
    if (gaussian.dim() != 6) {
        return false;
    }

    Eigen::Map<const Mat6> old_lam(gaussian.lamData());
    Eigen::Map<const Vec6> old_eta(gaussian.etaData());
    if (old_lam.cwiseAbs().maxCoeff() < 1e-24 && old_eta.cwiseAbs().maxCoeff() < 1e-18) {
        clearGaussian(gaussian);
        return true;
    }
    if (!allFiniteEigen(old_lam) ||
        !allFiniteEigen(old_eta) ||
        !allFiniteEigen(old_to_new_delta)) {
        return false;
    }

    Vec6 new_eta = old_eta - old_lam * old_to_new_delta;
    if (!allFiniteEigen(new_eta)) {
        return false;
    }
    gaussian.etaRef() = new_eta;
    return true;
}

void transportSyntheticSE3FactorMessages(
    gbp::Factor* factor,
    const Vec6Vector& old_to_new_deltas
) {
    if (factor == nullptr) {
        return;
    }
    for (size_t k = 0; k < factor->messages.size(); ++k) {
        const gbp::VariableNode* var = factor->adj_var_nodes.at(k);
        const int variable_id = (var != nullptr) ? var->variableID : -1;
        if (variable_id < 0 ||
            variable_id >= static_cast<int>(old_to_new_deltas.size()) ||
            !transportGaussianBetweenSE3Linearizations(
                factor->messages[k],
                old_to_new_deltas[static_cast<size_t>(variable_id)]
            )) {
            clearGaussian(factor->messages[k]);
        }
    }
}

void transportSyntheticSE3ResidualMessages(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    const SE3PoseVector& old_base_poses,
    const SE3PoseVector& new_base_poses,
    int num_threads
) {
    const int m = static_cast<int>(workspace.edge_factors.size());
    const bool use_parallel = (num_threads != 1) && (m > 64);
    const int n = static_cast<int>(old_base_poses.size());
    Vec6Vector old_to_new_deltas(static_cast<size_t>(n));
    const bool delta_parallel =
        (num_threads != 1) &&
        (n > 64) &&
        old_base_poses.size() == new_base_poses.size();
    if (delta_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            old_to_new_deltas[static_cast<size_t>(i)] =
                se3Log(se3Compose(
                    se3Inverse(old_base_poses[static_cast<size_t>(i)]),
                    new_base_poses[static_cast<size_t>(i)]
                ));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            old_to_new_deltas[static_cast<size_t>(i)] =
                se3Log(se3Compose(
                    se3Inverse(old_base_poses[static_cast<size_t>(i)]),
                    new_base_poses[static_cast<size_t>(i)]
                ));
        }
    }
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < m; ++edge_index) {
            transportSyntheticSE3FactorMessages(
                workspace.edge_factors[static_cast<size_t>(edge_index)],
                old_to_new_deltas
            );
        }
    } else {
        for (int edge_index = 0; edge_index < m; ++edge_index) {
            transportSyntheticSE3FactorMessages(
                workspace.edge_factors[static_cast<size_t>(edge_index)],
                old_to_new_deltas
            );
        }
    }

    if (workspace.anchor_factor != nullptr) {
        transportSyntheticSE3FactorMessages(
            workspace.anchor_factor,
            old_to_new_deltas
        );
    }
}

void refreshSyntheticSE3ResidualBeliefs(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    int num_threads
) {
    const int n = static_cast<int>(workspace.vars.size());
    const bool use_parallel = (num_threads != 1) && (n > 64);
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            if (workspace.vars[i] != nullptr) {
                workspace.vars[i]->updateBelief();
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            if (workspace.vars[i] != nullptr) {
                workspace.vars[i]->updateBelief();
            }
        }
    }
}

void relinearizeSyntheticSE3ResidualGraph(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    int num_threads,
    int outer_index,
    SyntheticSE3ResidualRelinearizeStats* stats = nullptr,
    const RobustLossConfig& robust_loss_config = {}
) {
    const auto total_t0 = SteadyClock::now();
    workspace.graph.se3_fixed_lam_eta_hot_valid = false;
    workspace.graph.se3_fixed_lam_eta_hot_entries.clear();
    const bool use_parallel = (num_threads != 1) && (problem.edges.size() > 64);
    const bool transport_messages =
        outer_index >= 12 &&
        workspace.has_linearization_poses &&
        workspace.linearization_poses.size() == base_poses.size();
    auto packed_state_it = se3PackedSoAStates().find(&workspace.graph);
    if (packed_state_it != se3PackedSoAStates().end()) {
        if (transport_messages &&
            packed_state_it->second.valid &&
            packed_state_it->second.workspace != nullptr) {
            copySyntheticSE3PackedSoAToGraph(
                *packed_state_it->second.workspace,
                workspace.graph,
                num_threads
            );
        }
        packed_state_it->second.valid = false;
    }

    const auto transport_t0 = SteadyClock::now();
    if (transport_messages) {
        transportSyntheticSE3ResidualMessages(
            workspace,
            workspace.linearization_poses,
            base_poses,
            num_threads
        );
    }
    const auto transport_t1 = SteadyClock::now();

    const auto factor_t0 = SteadyClock::now();
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE3Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[static_cast<size_t>(edge_index)];
            const SE3Pose& base_i = base_poses.at(edge.i);
            const SE3Pose& base_j = base_poses.at(edge.j);
            const Vec6 r0 = edgeResidual(base_i, base_j, edge.measurement, Vec6::Zero(), Vec6::Zero());
            const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
            const Mat6 weighted_information = robust_weight * edge.information;
            const Mat6x12 J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 12, 12> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 12, 1> eta = -J.transpose() * weighted_information * r0;
            factor->setLinearFactorInfo(eta, lam);
            if (!transport_messages) {
                factor->clearMessages();
            }
        }
    } else {
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE3Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[static_cast<size_t>(edge_index)];
            const SE3Pose& base_i = base_poses.at(edge.i);
            const SE3Pose& base_j = base_poses.at(edge.j);
            const Vec6 r0 = edgeResidual(base_i, base_j, edge.measurement, Vec6::Zero(), Vec6::Zero());
            const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
            const Mat6 weighted_information = robust_weight * edge.information;
            const Mat6x12 J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 12, 12> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 12, 1> eta = -J.transpose() * weighted_information * r0;
            factor->setLinearFactorInfo(eta, lam);
            if (!transport_messages) {
                factor->clearMessages();
            }
        }
    }

    if (workspace.anchor_factor != nullptr && !base_poses.empty()) {
        const Vec6 r0 = anchorResidual(base_poses.at(0), problem.anchor_pose, Vec6::Zero());
        const Mat6 J = analyticAnchorResidualJacobian(base_poses.at(0), problem.anchor_pose);
        const Mat6 lam = J.transpose() * problem.anchor_information * J;
        const Vec6 eta = -J.transpose() * problem.anchor_information * r0;
        workspace.anchor_factor->setLinearFactorInfo(eta, lam);
        if (!transport_messages) {
            workspace.anchor_factor->clearMessages();
        }
    }
    const auto factor_t1 = SteadyClock::now();

    const auto reset_t0 = SteadyClock::now();
    if (transport_messages) {
        refreshSyntheticSE3ResidualBeliefs(workspace, num_threads);
    } else {
        resetSyntheticSE3ResidualState(workspace, num_threads);
    }
    const auto reset_t1 = SteadyClock::now();

    workspace.linearization_poses = base_poses;
    workspace.has_linearization_poses = true;

    if (stats) {
        stats->factor_relinearize_sec = elapsedSeconds(factor_t0, factor_t1);
        stats->message_transport_sec = elapsedSeconds(transport_t0, transport_t1);
        stats->reset_state_sec = elapsedSeconds(reset_t0, reset_t1);
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
}

void writePoseJson(std::ostream& out, const SE3Pose& pose) {
    const SE3Pose p = normalizedPose(pose);
    out << "[" << jsonNumber(p.t.x())
        << ", " << jsonNumber(p.t.y())
        << ", " << jsonNumber(p.t.z())
        << ", " << jsonNumber(p.q.x())
        << ", " << jsonNumber(p.q.y())
        << ", " << jsonNumber(p.q.z())
        << ", " << jsonNumber(p.q.w())
        << "]";
}

void writePoseVectorJson(std::ostream& out, const SE3PoseVector& poses, int indent) {
    const std::string pad(indent, ' ');
    out << "[\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        out << pad;
        writePoseJson(out, poses[i]);
        out << (i + 1 == poses.size() ? "\n" : ",\n");
    }
    out << std::string(std::max(0, indent - 2), ' ') << "]";
}

void writePoseHistoryJsonArray(std::ostream& out, const SE3PoseHistory& history, int indent) {
    const std::string pad(indent, ' ');
    out << "[\n";
    for (size_t i = 0; i < history.size(); ++i) {
        out << pad;
        writePoseVectorJson(out, history[i], indent + 2);
        out << (i + 1 == history.size() ? "\n" : ",\n");
    }
    out << std::string(std::max(0, indent - 2), ' ') << "]";
}

}  // namespace

SyntheticSE3Problem loadSyntheticSE3Problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open SE3 problem file: " + path);
    }

    std::unordered_map<int, SE3Pose> raw_vertices;
    struct RawEdge {
        int vi = -1;
        int vj = -1;
        SE3Pose measurement;
        Mat6 information = Mat6::Zero();
    };
    std::vector<RawEdge> raw_edges;

    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::istringstream iss(line);
        std::string tag;
        if (!(iss >> tag)) {
            continue;
        }

        if (tag == "VERTEX_SE3:QUAT") {
            int vid = -1;
            SE3Pose pose;
            if (!(iss >> vid
                >> pose.t.x() >> pose.t.y() >> pose.t.z()
                >> pose.q.x() >> pose.q.y() >> pose.q.z() >> pose.q.w())) {
                throw std::runtime_error("Malformed VERTEX_SE3:QUAT line in " + path + ":" + std::to_string(lineno));
            }
            pose = normalizedPose(pose);
            raw_vertices[vid] = pose;
        } else if (tag == "EDGE_SE3:QUAT") {
            RawEdge edge;
            std::array<double, 21> info_vals{};
            if (!(iss >> edge.vi >> edge.vj
                >> edge.measurement.t.x() >> edge.measurement.t.y() >> edge.measurement.t.z()
                >> edge.measurement.q.x() >> edge.measurement.q.y() >> edge.measurement.q.z() >> edge.measurement.q.w()
                >> info_vals[0] >> info_vals[1] >> info_vals[2] >> info_vals[3] >> info_vals[4] >> info_vals[5]
                >> info_vals[6] >> info_vals[7] >> info_vals[8] >> info_vals[9] >> info_vals[10]
                >> info_vals[11] >> info_vals[12] >> info_vals[13] >> info_vals[14]
                >> info_vals[15] >> info_vals[16] >> info_vals[17]
                >> info_vals[18] >> info_vals[19]
                >> info_vals[20])) {
                throw std::runtime_error("Malformed EDGE_SE3:QUAT line in " + path + ":" + std::to_string(lineno));
            }
            edge.measurement = normalizedPose(edge.measurement);
            edge.information = symmetrizedInformationMatrix(info21ToMatrix(info_vals));
            raw_edges.push_back(edge);
        } else if (tag == "VERTEX_SE2" || tag == "EDGE_SE2") {
            throw std::runtime_error("Expected an SE3 g2o file but found SE2 tag in " + path);
        } else {
            throw std::runtime_error("Unsupported tag in g2o file " + path + ":" + std::to_string(lineno) + " tag=" + tag);
        }
    }

    if (raw_vertices.empty()) {
        throw std::runtime_error("No VERTEX_SE3:QUAT entries found in " + path);
    }

    std::vector<int> original_ids;
    original_ids.reserve(raw_vertices.size());
    for (const auto& kv : raw_vertices) {
        original_ids.push_back(kv.first);
    }
    std::sort(original_ids.begin(), original_ids.end());

    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(original_ids.size());
    for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
        id_to_idx[original_ids[idx]] = idx;
    }

    SyntheticSE3Problem problem;
    problem.gt_poses.resize(original_ids.size());
    problem.init_poses.resize(original_ids.size());
    for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
        const SE3Pose pose = raw_vertices.at(original_ids[idx]);
        problem.gt_poses[idx] = pose;
        problem.init_poses[idx] = pose;
    }

    problem.anchor_pose = problem.init_poses.front();
    problem.anchor_information = symmetrizedInformationMatrix(1e8 * Mat6::Identity());

    problem.edges.reserve(raw_edges.size());
    for (const RawEdge& raw_edge : raw_edges) {
        const auto it_i = id_to_idx.find(raw_edge.vi);
        const auto it_j = id_to_idx.find(raw_edge.vj);
        if (it_i == id_to_idx.end() || it_j == id_to_idx.end()) {
            throw std::runtime_error("g2o edge references unknown vertex id in " + path);
        }
        SyntheticSE3Edge edge;
        edge.i = it_i->second;
        edge.j = it_j->second;
        edge.measurement = raw_edge.measurement;
        edge.information = raw_edge.information;
        edge.kind = (std::abs(raw_edge.vi - raw_edge.vj) == 1) ? "odometry" : "loop";
        problem.edges.push_back(edge);
    }

    return problem;
}

double nonlinearObjective(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& poses
) {
    double total = 0.0;
    for (const SyntheticSE3Edge& edge : problem.edges) {
        const SE3Pose pred = se3Between(poses.at(edge.i), poses.at(edge.j));
        const Vec6 err = se3Log(se3Compose(se3Inverse(edge.measurement), pred));
        total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
    }
    const Vec6 anchor_err = se3Log(se3Compose(se3Inverse(problem.anchor_pose), poses.at(0)));
    total += 0.5 * (anchor_err.transpose() * problem.anchor_information * anchor_err)(0, 0);
    return total;
}

SE3PoseVector applyPoseDeltas(
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    if (delta_vec.size() != static_cast<int>(base_poses.size()) * 6) {
        throw std::runtime_error("SE3 delta vector size does not match pose count");
    }

    SE3PoseVector out(base_poses.size());
    for (size_t i = 0; i < base_poses.size(); ++i) {
        const Vec6 delta = delta_vec.segment<6>(static_cast<int>(6 * i));
        out[i] = se3Plus(base_poses[i], delta);
    }
    return out;
}

double nonlinearObjectiveForDelta(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    return nonlinearObjective(problem, applyPoseDeltas(base_poses, delta_vec));
}

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    double tiny_prior,
    const RobustLossConfig& robust_loss_config
) {
    gbp::FactorGraph graph;
    graph.eta_damping = se3MessageDamping();
    std::vector<gbp::VariableNode*> vars(base_poses.size(), nullptr);
    const Mat6 weak_lam = tiny_prior * Mat6::Identity();

    for (int i = 0; i < static_cast<int>(base_poses.size()); ++i) {
        gbp::VariableNode* var = graph.addVariable(i, 6);
        var->prior.setLam(weak_lam);
        var->prior.setEta(Eigen::VectorXd::Zero(6));
        var->belief.setLam(weak_lam);
        var->belief.setEta(Eigen::VectorXd::Zero(6));
        var->mu = Eigen::VectorXd::Zero(6);
        var->markMuCurrent();
        vars[i] = var;
    }

    int fid = 0;
    for (const SyntheticSE3Edge& edge : problem.edges) {
        const Vec6 r0 = edgeResidual(
            base_poses.at(edge.i),
            base_poses.at(edge.j),
            edge.measurement,
            Vec6::Zero(),
            Vec6::Zero()
        );
        const Mat6x12 J = analyticEdgeResidualJacobian(
            base_poses.at(edge.i),
            base_poses.at(edge.j),
            edge.measurement
        );
        const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
        const Mat6 weighted_information = robust_weight * edge.information;

        auto meas_fn = [J](const Eigen::VectorXd& x) {
            return std::vector<Eigen::VectorXd>{J * x};
        };
        auto jac_fn = [J](const Eigen::VectorXd&) {
            return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd(J)};
        };

        gbp::Factor* factor = graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vars[edge.i], vars[edge.j]},
            std::vector<Eigen::VectorXd>{-r0},
            std::vector<Eigen::MatrixXd>{weighted_information},
            meas_fn,
            jac_fn
        );
        factor->computeFactor(Eigen::VectorXd::Zero(12), true);
        graph.connect(factor, vars[edge.i], 0);
        graph.connect(factor, vars[edge.j], 1);
    }

    if (!vars.empty()) {
        const Vec6 r0 = anchorResidual(base_poses.at(0), problem.anchor_pose, Vec6::Zero());
        const Mat6 J = analyticAnchorResidualJacobian(base_poses.at(0), problem.anchor_pose);
        auto meas_fn = [J](const Eigen::VectorXd& x) {
            return std::vector<Eigen::VectorXd>{J * x};
        };
        auto jac_fn = [J](const Eigen::VectorXd&) {
            return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd(J)};
        };
        gbp::Factor* factor = graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vars[0]},
            std::vector<Eigen::VectorXd>{-r0},
            std::vector<Eigen::MatrixXd>{problem.anchor_information},
            meas_fn,
            jac_fn
        );
        factor->computeFactor(Eigen::VectorXd::Zero(6), true);
        graph.connect(factor, vars[0], 0);
    }

    return graph;
}

SyntheticSE3ExperimentResults runSyntheticSE3Experiment(
    const SyntheticSE3Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config,
    int sync_num_threads,
    bool direct_enabled
) {
    SyntheticSE3ExperimentResults results;
    const int thread_count = effectiveThreadCount(sync_num_threads);
    const ScopedSE3SingleThreadRuntime single_thread_runtime_guard(
        thread_count == 1 && se3SingleThreadRuntimeGuardEnabled()
    );
    results.num_poses = static_cast<int>(problem.init_poses.size());
    results.num_edges = static_cast<int>(problem.edges.size());
    results.initial_objective = nonlinearObjective(problem, problem.init_poses);
    results.initial_poses = problem.init_poses;

    SE3PoseVector direct_poses = problem.init_poses;
    SE3PoseVector mg_poses = problem.init_poses;
    results.direct_history.push_back(SyntheticSE3OuterDirectRow{0, results.initial_objective});
    results.mg_history.push_back(SyntheticSE3OuterMGRow{0, results.initial_objective});
    results.direct_pose_history.push_back(direct_poses);
    results.mg_pose_history.push_back(mg_poses);

    std::unique_ptr<SyntheticSE3ResidualGraphWorkspace> mg_workspace;
    mg_workspace = std::make_unique<SyntheticSE3ResidualGraphWorkspace>(
        buildSyntheticSE3ResidualGraphWorkspace(problem, 1e-12)
    );
    const int basis_rebuild_period = se3BasisRebuildPeriod();
    const int basis_rebuild_warmup_outers = se3BasisRebuildWarmupOuters();
    SE3BasisData cached_basis;
    bool cached_basis_valid = false;
    std::vector<Eigen::MatrixXd> warm_start_basis_local_bases;
    SparseCholeskyFactor coarse_factor_cache;
    bool coarse_factor_cache_valid = false;
    Eigen::SparseMatrix<double> coarse_lam_direct_cache;
    JointInfSparsePatternCache mg_joint_cache;
    CoarseLambdaAssemblyCache mg_coarse_lambda_cache;
    const bool reuse_coarse_pattern = se3ReuseCoarsePatternEnabled();
    const int coarse_numeric_rebuild_period = se3CoarseNumericRebuildPeriod();
    const int coarse_reuse_pcg_iterations = se3CoarseReusePcgIterations();

    for (int outer = 1; outer <= num_outer; ++outer) {
        if (direct_enabled) {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            gbp::FactorGraph graph = buildLinearizedResidualGraph(problem, direct_poses, 1e-12, robust_loss_config);
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = graph.jointDistributionInfSparse();
            const Eigen::SparseMatrix<double> lam = symmetrizeSparse(J.lam);
            const auto joint_t1 = SteadyClock::now();

            const auto solve_t0 = SteadyClock::now();
            const Eigen::VectorXd e_star = solveSparseCholesky(lam, J.eta, 1e-10);
            const auto solve_t1 = SteadyClock::now();

            const auto apply_t0 = SteadyClock::now();
            direct_poses = applyPoseDeltas(direct_poses, e_star);
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = nonlinearObjective(problem, direct_poses);
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            results.direct_history.push_back(
                SyntheticSE3OuterDirectRow{
                    outer,
                    nonlinear_obj,
                    e_star.norm(),
                    (J.eta - lam * e_star).norm(),
                    elapsedSeconds(total_t0, total_t1),
                    elapsedSeconds(build_t0, build_t1),
                    elapsedSeconds(joint_t0, joint_t1),
                    elapsedSeconds(solve_t0, solve_t1),
                    elapsedSeconds(apply_t0, apply_t1),
                    elapsedSeconds(obj_t0, obj_t1),
                }
            );
            results.direct_pose_history.push_back(direct_poses);
        }

        {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            std::unique_ptr<gbp::FactorGraph> graph_owned;
            gbp::FactorGraph* graph_ptr = nullptr;
            SyntheticSE3ResidualRelinearizeStats relin_stats;
            if (mg_workspace) {
                relinearizeSyntheticSE3ResidualGraph(
                    *mg_workspace,
                    problem,
                    mg_poses,
                    thread_count,
                    outer,
                    &relin_stats,
                    robust_loss_config
                );
                graph_ptr = &mg_workspace->graph;
            } else {
                graph_owned = std::make_unique<gbp::FactorGraph>(
                    buildLinearizedResidualGraph(problem, mg_poses, 1e-12, robust_loss_config)
                );
                graph_ptr = graph_owned.get();
            }
            gbp::FactorGraph& graph = *graph_ptr;
            graph.setSyncNumThreads(thread_count);
            graph.setProfileSyncTiming(se3ProfileSyncTimingEnabled());
            graph.resetSyncTiming();
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            bool use_implicit_fine_operator = se3ImplicitFineOperatorEnabled();
            SyntheticSE3PackedSoAWorkspace* implicit_fine_workspace =
                use_implicit_fine_operator
                    ? preparePackedSoAWorkspace(graph, problem)
                    : nullptr;
            use_implicit_fine_operator = implicit_fine_workspace != nullptr;
            const int implicit_fine_threads =
                se3ImplicitFineOperatorThreadCount(thread_count);
            gbp::FactorGraph::JointInfResult J;
            Eigen::SparseMatrix<double> lam_storage;
            const Eigen::SparseMatrix<double>* lam_ptr = nullptr;
            if (use_implicit_fine_operator) {
                assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(
                    *implicit_fine_workspace,
                    J.eta,
                    implicit_fine_threads
                );
                J.total_dim = static_cast<int>(J.eta.size());
                lam_storage.resize(J.total_dim, J.total_dim);
                lam_ptr = &lam_storage;
            } else {
                fillJointDistributionInfSparseCached(graph, mg_joint_cache, J.eta);
                J.total_dim = mg_joint_cache.total_dim;
                lam_ptr = &mg_joint_cache.lam;
            }
            const Eigen::SparseMatrix<double>& lam = *lam_ptr;
            if (use_implicit_fine_operator &&
                se3ImplicitFineOperatorCompareEnabled()) {
                Eigen::VectorXd reference_eta;
                fillJointDistributionInfSparseCached(
                    graph,
                    mg_joint_cache,
                    reference_eta
                );
                Eigen::VectorXd probe =
                    Eigen::VectorXd::LinSpaced(J.total_dim, -0.5, 0.5);
                Eigen::VectorXd implicit_product;
                multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
                    *implicit_fine_workspace,
                    probe,
                    implicit_product,
                    implicit_fine_threads
                );
                const Eigen::VectorXd reference_product =
                    mg_joint_cache.lam * probe;
                const double eta_relative_error =
                    (J.eta - reference_eta).norm() /
                    std::max(reference_eta.norm(), 1e-15);
                const double product_relative_error =
                    (implicit_product - reference_product).norm() /
                    std::max(reference_product.norm(), 1e-15);
                std::cerr
                    << "[se3 implicit fine compare] outer=" << outer
                    << " eta_rel=" << eta_relative_error
                    << " product_rel=" << product_relative_error
                    << "\n";
            }
            auto assemble_fine_residual = [&](
                const Eigen::VectorXd& x,
                Eigen::VectorXd& residual
            ) {
                if (use_implicit_fine_operator) {
                    multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
                        *implicit_fine_workspace,
                        x,
                        residual,
                        implicit_fine_threads
                    );
                    residual = J.eta - residual;
                } else {
                    residual.noalias() = J.eta - lam * x;
                }
            };
            const auto joint_t1 = SteadyClock::now();

            const auto basis_t0 = SteadyClock::now();
            const bool basis_within_warmup =
                basis_rebuild_warmup_outers > 0 && outer <= basis_rebuild_warmup_outers;
            const int post_warmup_outer = outer - basis_rebuild_warmup_outers;
            const bool rebuild_basis =
                !cached_basis_valid ||
                basis_rebuild_period <= 1 ||
                basis_within_warmup ||
                (((post_warmup_outer - 1) % basis_rebuild_period) == 0);
            if (rebuild_basis) {
                cached_basis = buildMessageConditionedBasis(
                    graph,
                    &mg_poses,
                    group_size,
                    r_reduced,
                    thread_count,
                    warm_start_basis_local_bases.empty() ? nullptr : &warm_start_basis_local_bases
                );
                cached_basis_valid = true;
                warm_start_basis_local_bases = cached_basis.local_bases;
            }
            const auto basis_t1 = SteadyClock::now();
            const SE3BasisData& basis = cached_basis;

            const auto coarse_lam_t0 = SteadyClock::now();
            const bool use_cached_coarse_lambda =
                se3CachedCoarseLambdaEnabled() &&
                static_cast<int>(basis.local_bases.size()) >= se3CachedCoarseLambdaMinGroups();
            const bool rebuild_coarse_numeric =
                !coarse_factor_cache_valid ||
                rebuild_basis ||
                coarse_numeric_rebuild_period <= 1 ||
                basis_within_warmup ||
                (((post_warmup_outer - 1) % coarse_numeric_rebuild_period) == 0);
            const Eigen::SparseMatrix<double>* coarse_lam_ptr = nullptr;
            if (use_cached_coarse_lambda) {
                coarse_lam_ptr =
                    &assembleCoarseLambdaCachedInPlace(
                        graph,
                        basis,
                        mg_coarse_lambda_cache,
                        thread_count
                    );
            } else {
                coarse_lam_direct_cache =
                    assembleCoarseLambdaDirect(
                        graph,
                        basis,
                        thread_count > 1,
                        thread_count
                    );
                coarse_lam_ptr = &coarse_lam_direct_cache;
            }
            Eigen::SparseMatrix<double>& cached_coarse_lam_ref = mg_coarse_lambda_cache.lam;
            bool coarse_ridge_added_inplace = false;
            if (use_cached_coarse_lambda &&
                se3CoarseRidgeInplaceEnabled() &&
                coarse_lam_ptr == &cached_coarse_lam_ref) {
                for (int i = 0; i < cached_coarse_lam_ref.rows(); ++i) {
                    cached_coarse_lam_ref.coeffRef(i, i) += 1e-8;
                }
                cached_coarse_lam_ref.makeCompressed();
                coarse_ridge_added_inplace = true;
            }
            const Eigen::SparseMatrix<double>& coarse_lam = *coarse_lam_ptr;
            const auto coarse_lam_t1 = SteadyClock::now();

            SparseCholeskyFactor coarse_factor_local;
            const bool persist_coarse_factor =
                reuse_coarse_pattern || coarse_numeric_rebuild_period > 1;
            SparseCholeskyFactor* coarse_factor_ptr =
                persist_coarse_factor ? &coarse_factor_cache : &coarse_factor_local;
            const auto coarse_factor_t0 = SteadyClock::now();
            if (rebuild_coarse_numeric) {
                factorizeSparseCholesky(
                    coarse_lam,
                    coarse_ridge_added_inplace ? 0.0 : 1e-8,
                    *coarse_factor_ptr,
                    reuse_coarse_pattern && coarse_factor_cache_valid,
                    true
                );
                coarse_factor_cache_valid = persist_coarse_factor;
            }
            const auto coarse_factor_t1 = SteadyClock::now();
            const double coarse_factorize_sec = elapsedSeconds(coarse_factor_t0, coarse_factor_t1);

            double sweeps_sec = 0.0;
            double coarse_eta_sec = 0.0;
            double coarse_solve_sec = coarse_factorize_sec;
            double coarse_backsolve_sec = 0.0;
            double prolong_inject_sec = 0.0;
            double first_cycle_residual_before_sweeps = 0.0;
            double first_cycle_residual_after_sweeps = 0.0;
            double first_cycle_residual_after_coarse = 0.0;
            double last_cycle_residual_before_sweeps = 0.0;
            double last_cycle_residual_after_sweeps = 0.0;
            double last_cycle_residual_after_coarse = 0.0;
            double smoother_translation_update_norm_sum = 0.0;
            double smoother_rotation_update_norm_sum = 0.0;
            int smoother_local_rejects_total = 0;
            Eigen::VectorXd e_now = stackedMeanVector(graph);
            Eigen::VectorXd residual_before_sweeps(basis.total_dim);
            Eigen::VectorXd residual_after_sweeps(basis.total_dim);
            Eigen::VectorXd residual_after_coarse(basis.total_dim);
            Eigen::VectorXd coarse_eta(basis.coarse_dim);
            Eigen::VectorXd delta_z(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_residual(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_preconditioned_residual(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_direction(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_matrix_direction(basis.coarse_dim);
            Eigen::VectorXd delta_fine(basis.total_dim);
            int executed_inner_cycles = 0;
            bool have_previous_residual_after_coarse = false;
            double previous_residual_after_coarse_norm = 0.0;
            for (int cyc = 0; cyc < inner_cycles; ++cyc) {
                double residual_before_sweeps_norm = 0.0;
                if (have_previous_residual_after_coarse) {
                    residual_before_sweeps_norm = previous_residual_after_coarse_norm;
                } else {
                    assemble_fine_residual(e_now, residual_before_sweeps);
                    residual_before_sweeps_norm = residual_before_sweeps.norm();
                }
                const auto sweeps_t0 = SteadyClock::now();
                SE3SmootherSweepStats smoother_stats;
                smoother_stats.residual_before_norm = residual_before_sweeps_norm;
                bool residual_after_sweeps_is_current = false;
                const Eigen::VectorXd e_before_gbp_sweeps = e_now;

                const bool defer_packed_graph_copy =
                    se3PackedSoADeferGraphCopyEnabled() &&
                    se3PackedSoASweepsEnabled();
                manualSynchronousIterations(
                    graph,
                    pre_sweeps,
                    thread_count,
                    cyc * pre_sweeps,
                    outer,
                    &problem
                );
                if (!(defer_packed_graph_copy &&
                      tryStackedMeanVectorPackedSoA(graph, thread_count, e_now))) {
                    e_now = stackedMeanVector(graph);
                }
                const Eigen::VectorXd gbp_delta = e_now - e_before_gbp_sweeps;
                accumulateSE3DeltaComponentNorms(
                    gbp_delta,
                    smoother_stats.translation_update_norm_sum,
                    smoother_stats.rotation_update_norm_sum
                );
                assemble_fine_residual(e_now, residual_after_sweeps);
                smoother_stats.residual_after_norm = residual_after_sweeps.norm();
                residual_after_sweeps_is_current = true;
                const auto sweeps_t1 = SteadyClock::now();
                sweeps_sec += elapsedSeconds(sweeps_t0, sweeps_t1);
                smoother_translation_update_norm_sum += smoother_stats.translation_update_norm_sum;
                smoother_rotation_update_norm_sum += smoother_stats.rotation_update_norm_sum;
                smoother_local_rejects_total += smoother_stats.local_rejects;

                if (!residual_after_sweeps_is_current) {
                    assemble_fine_residual(e_now, residual_after_sweeps);
                    smoother_stats.residual_after_norm = residual_after_sweeps.norm();
                }
                const double residual_after_sweeps_norm = smoother_stats.residual_after_norm;
                if (cyc == 0) {
                    first_cycle_residual_before_sweeps = residual_before_sweeps_norm;
                    first_cycle_residual_after_sweeps = residual_after_sweeps_norm;
                }
                last_cycle_residual_before_sweeps = residual_before_sweeps_norm;
                last_cycle_residual_after_sweeps = residual_after_sweeps_norm;
                const auto coarse_eta_t0 = SteadyClock::now();
                restrictToCoarseInto(basis, residual_after_sweeps, coarse_eta);
                const auto coarse_eta_t1 = SteadyClock::now();
                coarse_eta_sec += elapsedSeconds(coarse_eta_t0, coarse_eta_t1);

                const auto coarse_solve_t0 = SteadyClock::now();
                if (rebuild_coarse_numeric) {
                    solveWithSparseCholeskyInto(
                        *coarse_factor_ptr,
                        coarse_eta,
                        delta_z
                    );
                } else {
                    solveWithReusedCoarsePcgInto(
                        coarse_lam,
                        *coarse_factor_ptr,
                        coarse_eta,
                        coarse_reuse_pcg_iterations,
                        delta_z,
                        coarse_pcg_residual,
                        coarse_pcg_preconditioned_residual,
                        coarse_pcg_direction,
                        coarse_pcg_matrix_direction
                    );
                }
                const auto coarse_solve_t1 = SteadyClock::now();
                const double backsolve_elapsed = elapsedSeconds(coarse_solve_t0, coarse_solve_t1);
                coarse_solve_sec += backsolve_elapsed;
                coarse_backsolve_sec += backsolve_elapsed;

                const auto inject_t0 = SteadyClock::now();
                prolongToFineInto(basis, delta_z, delta_fine);
                e_now.noalias() += delta_fine;
                const Eigen::VectorXd* coarse_delta_ptr = &delta_fine;
                const bool defer_packed_graph_copy_after_coarse =
                    se3PackedSoADeferGraphCopyEnabled() &&
                    se3PackedSoASweepsEnabled();
                const bool injected_packed =
                    defer_packed_graph_copy_after_coarse &&
                    tryApplyMeanDeltaPackedSoA(graph, *coarse_delta_ptr, thread_count);
                if (!injected_packed) {
                    injectCorrectionKeepMessages(graph, *coarse_delta_ptr);
                }
                manualSynchronousIterations(
                    graph,
                    se3PostCoarseMessageSweeps(),
                    thread_count
                );
                if (!(defer_packed_graph_copy_after_coarse &&
                      tryStackedMeanVectorPackedSoA(graph, thread_count, e_now))) {
                    e_now = stackedMeanVector(graph);
                }
                assemble_fine_residual(e_now, residual_after_coarse);
                const double residual_after_coarse_norm = residual_after_coarse.norm();
                previous_residual_after_coarse_norm = residual_after_coarse_norm;
                have_previous_residual_after_coarse = true;
                if (cyc == 0) {
                    first_cycle_residual_after_coarse = residual_after_coarse_norm;
                }
                last_cycle_residual_after_coarse = residual_after_coarse_norm;
                const auto inject_t1 = SteadyClock::now();
                prolong_inject_sec += elapsedSeconds(inject_t0, inject_t1);
                ++executed_inner_cycles;
            }

            const Eigen::VectorXd e_hat = e_now;
            const double raw_e_hat_norm = e_hat.norm();
            const auto apply_t0 = SteadyClock::now();
            const double base_obj = results.mg_history.empty()
                ? nonlinearObjective(problem, mg_poses)
                : results.mg_history.back().nonlinear_objective;
            double step_scale = 1.0;
            int line_search_rejects = 0;
            double trial_obj = base_obj;
            SE3PoseVector trial_poses = mg_poses;
            while (step_scale >= (1.0 / 1024.0)) {
                trial_poses = applyPoseDeltas(mg_poses, step_scale * e_hat);
                trial_obj = nonlinearObjective(problem, trial_poses);
                if (std::isfinite(trial_obj) && trial_obj <= base_obj) {
                    break;
                }
                step_scale *= 0.5;
                ++line_search_rejects;
            }
            if (step_scale < (1.0 / 1024.0)) {
                trial_poses = mg_poses;
                trial_obj = base_obj;
                step_scale = 0.0;
            }
            mg_poses = trial_poses;
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = trial_obj;
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            const int coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
            const double coarse_lambda_density =
                basis.coarse_dim > 0
                ? static_cast<double>(coarse_lambda_nnz) /
                    (static_cast<double>(basis.coarse_dim) * static_cast<double>(basis.coarse_dim))
                : 0.0;
            double linear_residual_approx_norm =
                previous_residual_after_coarse_norm;
            if (!(step_scale == 1.0 && have_previous_residual_after_coarse)) {
                assemble_fine_residual(
                    step_scale * e_hat,
                    residual_after_coarse
                );
                linear_residual_approx_norm = residual_after_coarse.norm();
            }

            results.mg_history.push_back(
                SyntheticSE3OuterMGRow{
                    outer,
                    nonlinear_obj,
                    raw_e_hat_norm,
                    (step_scale * e_hat).norm(),
                    first_cycle_residual_before_sweeps,
                    first_cycle_residual_after_sweeps,
                    first_cycle_residual_after_coarse,
                    last_cycle_residual_before_sweeps,
                    last_cycle_residual_after_sweeps,
                    last_cycle_residual_after_coarse,
                    linear_residual_approx_norm,
                    step_scale,
                    line_search_rejects,
                    executed_inner_cycles,
                    smoother_translation_update_norm_sum,
                    smoother_rotation_update_norm_sum,
                    smoother_local_rejects_total,
                    graph.sync_total_sweeps_accum,
                    graph.sync_fixed_lam_hard_start_sweep,
                    graph.sync_fixed_lam_effective_start_sweep,
                    graph.sync_fixed_lam_factor_sweeps_accum,
                    graph.sync_eta_only_variable_sweeps_accum,
                    graph.sync_fixed_lam_hot_eta_sweeps_accum,
                    graph.sync_fixed_lam_hot_eta_entries,
                    static_cast<int>(basis.groups.size()),
                    basis.coarse_dim,
                    coarse_lambda_nnz,
                    coarse_lambda_density,
                    elapsedSeconds(total_t0, total_t1),
                    elapsedSeconds(build_t0, build_t1),
                    rebuild_basis ? elapsedSeconds(basis_t0, basis_t1) : 0.0,
                    rebuild_basis ? basis.block_build_sec : 0.0,
                    rebuild_basis ? basis.eigensolver_sec : 0.0,
                    rebuild_basis ? basis.copyout_sec : 0.0,
                    std::isfinite(basis.min_eigenvalue) ? basis.min_eigenvalue : 0.0,
                    std::isfinite(basis.max_eigenvalue) ? basis.max_eigenvalue : 0.0,
                    basis.negative_group_count,
                    basis.nonpositive_group_count,
                    basis.partial_attempt_count,
                    basis.partial_converged_count,
                    basis.full_eigensolver_count,
                    elapsedSeconds(coarse_lam_t0, coarse_lam_t1),
                    elapsedSeconds(joint_t0, joint_t1),
                    sweeps_sec,
                    graph.sync_factor_pass_sec_accum,
                    graph.sync_variable_pass_sec_accum,
                    coarse_eta_sec,
                    coarse_solve_sec,
                    coarse_factorize_sec,
                    coarse_backsolve_sec,
                    prolong_inject_sec,
                    elapsedSeconds(apply_t0, apply_t1),
                    elapsedSeconds(obj_t0, obj_t1),
                    relin_stats.message_transport_sec,
                    relin_stats.factor_relinearize_sec,
                    relin_stats.reset_state_sec,
                    relin_stats.total_sec,
                }
            );
            results.mg_pose_history.push_back(mg_poses);
        }
    }

    const int final_polish_steps = se3FinalDirectPolishSteps();
    if (final_polish_steps > 0 && !results.mg_history.empty()) {
        for (int polish = 0; polish < final_polish_steps; ++polish) {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            gbp::FactorGraph graph =
                buildLinearizedResidualGraph(problem, mg_poses, 1e-12, robust_loss_config);
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = graph.jointDistributionInfSparse();
            const Eigen::SparseMatrix<double> lam = symmetrizeSparse(J.lam);
            const auto joint_t1 = SteadyClock::now();

            const auto solve_t0 = SteadyClock::now();
            const Eigen::VectorXd polish_step = solveSparseCholesky(lam, J.eta, 1e-10);
            const auto solve_t1 = SteadyClock::now();

            const auto apply_t0 = SteadyClock::now();
            const double base_obj = nonlinearObjective(problem, mg_poses);
            double step_scale = 1.0;
            int line_search_rejects = 0;
            double trial_obj = base_obj;
            SE3PoseVector trial_poses = mg_poses;
            while (step_scale >= (1.0 / 1024.0)) {
                trial_poses = applyPoseDeltas(mg_poses, step_scale * polish_step);
                trial_obj = nonlinearObjective(problem, trial_poses);
                if (std::isfinite(trial_obj) && trial_obj <= base_obj) {
                    break;
                }
                step_scale *= 0.5;
                ++line_search_rejects;
            }
            if (step_scale < (1.0 / 1024.0)) {
                trial_poses = mg_poses;
                trial_obj = base_obj;
                step_scale = 0.0;
            }
            mg_poses = trial_poses;
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = trial_obj;
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            SyntheticSE3OuterMGRow& row = results.mg_history.back();
            row.nonlinear_objective = nonlinear_obj;
            row.raw_e_hat_norm = polish_step.norm();
            row.e_hat_norm = (step_scale * polish_step).norm();
            row.linear_residual_approx = (J.eta - lam * (step_scale * polish_step)).norm();
            row.line_search_step_scale = step_scale;
            row.line_search_rejects += line_search_rejects;
            row.outer_total_sec += elapsedSeconds(total_t0, total_t1);
            row.build_graph_sec += elapsedSeconds(build_t0, build_t1);
            row.exact_joint_assembly_sec += elapsedSeconds(joint_t0, joint_t1);
            row.exact_solve_sec += elapsedSeconds(solve_t0, solve_t1);
            row.apply_step_sec += elapsedSeconds(apply_t0, apply_t1);
            row.objective_eval_sec += elapsedSeconds(obj_t0, obj_t1);
        }
        if (!results.mg_pose_history.empty()) {
            results.mg_pose_history.back() = mg_poses;
        } else {
            results.mg_pose_history.push_back(mg_poses);
        }
    }

    return results;
}

void writeSyntheticSE3ExperimentResultsJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config,
    int sync_num_threads,
    bool direct_enabled
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output JSON path: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"num_outer\": " << num_outer << ",\n";
    out << "    \"inner_cycles\": " << inner_cycles << ",\n";
    out << "    \"pre_sweeps\": " << pre_sweeps << ",\n";
    out << "    \"group_size\": " << group_size << ",\n";
    out << "    \"r_reduced\": " << r_reduced << ",\n";
    out << "    \"basis_rebuild_period\": " << se3BasisRebuildPeriod() << ",\n";
    out << "    \"basis_rebuild_warmup_outers\": " << se3BasisRebuildWarmupOuters() << ",\n";
    out << "    \"basis_source\": \"message_conditioned_information_generic\",\n";
    out << "    \"partial_basis_accept_residual_tol\": "
        << jsonNumber(se3EnvDoubleOrDefault("GBP_SE3_PARTIAL_BASIS_ACCEPT_RESIDUAL_TOL", -1.0))
        << ",\n";
    out << "    \"partial_basis_large_group_min_groups\": "
        << se3PartialBasisLargeGroupMinGroups() << ",\n";
    out << "    \"partial_basis_large_group_accept_residual_tol\": "
        << jsonNumber(se3PartialBasisLargeGroupAcceptResidualTol()) << ",\n";
    out << "    \"partial_basis_small_group_accept_residual_tol\": "
        << jsonNumber(se3PartialBasisSmallGroupAcceptResidualTol()) << ",\n";
    out << "    \"partial_basis_residual_check_period\": "
        << std::max(
               1,
               se3EnvIntOrDefault("GBP_SE3_PARTIAL_BASIS_RESIDUAL_CHECK_PERIOD", 1)
           )
        << ",\n";
    out << "    \"sync_num_threads\": " << effectiveThreadCount(sync_num_threads) << ",\n";
    out << "    \"enable_singlecore_fastsync\": false,\n";
    out << "    \"single_thread_runtime_guard\": "
        << (se3SingleThreadRuntimeGuardEnabled() ? "true" : "false") << ",\n";
    out << "    \"direct_enabled\": " << (direct_enabled ? "true" : "false") << ",\n";
    out << "    \"robust_huber_delta\": " << jsonNumber(robust_loss_config.huber_delta) << ",\n";
    out << "    \"transport_policy\": \"fixed_lambda_eta_shift\",\n";
    out << "    \"reuse_coarse_pattern\": "
        << (se3ReuseCoarsePatternEnabled() ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_lambda\": "
        << (se3CachedCoarseLambdaEnabled() ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_inplace_symmetrize\": "
        << (se3CachedCoarseInplaceSymmetrizeEnabled() ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_lambda_min_groups\": "
        << se3CachedCoarseLambdaMinGroups() << ",\n";
    out << "    \"coarse_assembly_shards\": "
        << kSE3CoarseAssemblyShards << ",\n";
    out << "    \"coarse_numeric_rebuild_period\": "
        << se3CoarseNumericRebuildPeriod() << ",\n";
    out << "    \"coarse_reuse_pcg_iterations\": "
        << se3CoarseReusePcgIterations() << ",\n";
    out << "    \"implicit_fine_operator\": "
        << (se3ImplicitFineOperatorEnabled() ? "true" : "false") << ",\n";
    out << "    \"implicit_fine_operator_threads\": "
        << se3EnvIntOrDefault("GBP_SE3_IMPLICIT_FINE_OPERATOR_THREADS", 0)
        << ",\n";
    out << "    \"final_direct_polish_steps\": "
        << se3FinalDirectPolishSteps() << "\n";
    out << "  },\n";
    out << "  \"problem\": {\n";
    out << "    \"num_poses\": " << results.num_poses << ",\n";
    out << "    \"num_edges\": " << results.num_edges << "\n";
    out << "  },\n";
    out << "  \"initial_objective\": " << jsonNumber(results.initial_objective) << ",\n";
    out << "  \"direct_history\": [\n";
    for (size_t i = 0; i < results.direct_history.size(); ++i) {
        const SyntheticSE3OuterDirectRow& row = results.direct_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"linear_step_norm\": " << jsonNumber(row.linear_step_norm)
            << ", \"linear_residual_norm\": " << jsonNumber(row.linear_residual_norm)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"joint_assembly_sec\": " << jsonNumber(row.joint_assembly_sec)
            << ", \"exact_solve_sec\": " << jsonNumber(row.exact_solve_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << "}";
        out << (i + 1 == results.direct_history.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"mg_history\": [\n";
    for (size_t i = 0; i < results.mg_history.size(); ++i) {
        const SyntheticSE3OuterMGRow& row = results.mg_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"raw_e_hat_norm\": " << jsonNumber(row.raw_e_hat_norm)
            << ", \"e_hat_norm\": " << jsonNumber(row.e_hat_norm)
            << ", \"first_cycle_residual_before_sweeps\": "
            << jsonNumber(row.first_cycle_residual_before_sweeps)
            << ", \"first_cycle_residual_after_sweeps\": "
            << jsonNumber(row.first_cycle_residual_after_sweeps)
            << ", \"first_cycle_residual_after_coarse\": "
            << jsonNumber(row.first_cycle_residual_after_coarse)
            << ", \"last_cycle_residual_before_sweeps\": "
            << jsonNumber(row.last_cycle_residual_before_sweeps)
            << ", \"last_cycle_residual_after_sweeps\": "
            << jsonNumber(row.last_cycle_residual_after_sweeps)
            << ", \"last_cycle_residual_after_coarse\": "
            << jsonNumber(row.last_cycle_residual_after_coarse)
            << ", \"linear_residual_approx\": " << jsonNumber(row.linear_residual_approx)
            << ", \"line_search_step_scale\": " << jsonNumber(row.line_search_step_scale)
            << ", \"line_search_rejects\": " << row.line_search_rejects
            << ", \"executed_inner_cycles\": " << row.executed_inner_cycles
            << ", \"smoother_translation_update_norm_sum\": "
            << jsonNumber(row.smoother_translation_update_norm_sum)
            << ", \"smoother_rotation_update_norm_sum\": "
            << jsonNumber(row.smoother_rotation_update_norm_sum)
            << ", \"smoother_local_rejects_total\": " << row.smoother_local_rejects_total
            << ", \"smoother_total_sweeps\": " << row.smoother_total_sweeps
            << ", \"fixed_lambda_hard_start_sweep\": " << row.fixed_lambda_hard_start_sweep
            << ", \"fixed_lambda_effective_start_sweep\": "
            << row.fixed_lambda_effective_start_sweep
            << ", \"fixed_lambda_factor_sweeps\": " << row.fixed_lambda_factor_sweeps
            << ", \"eta_only_variable_sweeps\": " << row.eta_only_variable_sweeps
            << ", \"fixed_lambda_hot_eta_sweeps\": " << row.fixed_lambda_hot_eta_sweeps
            << ", \"fixed_lambda_hot_eta_entries\": " << row.fixed_lambda_hot_eta_entries
            << ", \"num_groups\": " << row.num_groups
            << ", \"coarse_dim\": " << row.coarse_dim
            << ", \"coarse_lambda_nnz\": " << row.coarse_lambda_nnz
            << ", \"coarse_lambda_density\": " << jsonNumber(row.coarse_lambda_density)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"basis_build_sec\": " << jsonNumber(row.basis_build_sec)
            << ", \"basis_block_build_sec\": " << jsonNumber(row.basis_block_build_sec)
            << ", \"basis_eigensolver_sec\": " << jsonNumber(row.basis_eigensolver_sec)
            << ", \"basis_copyout_sec\": " << jsonNumber(row.basis_copyout_sec)
            << ", \"basis_min_eigenvalue\": " << jsonNumber(row.basis_min_eigenvalue)
            << ", \"basis_max_eigenvalue\": " << jsonNumber(row.basis_max_eigenvalue)
            << ", \"basis_negative_group_count\": " << row.basis_negative_group_count
            << ", \"basis_nonpositive_group_count\": " << row.basis_nonpositive_group_count
            << ", \"basis_partial_attempt_count\": " << row.basis_partial_attempt_count
            << ", \"basis_partial_converged_count\": " << row.basis_partial_converged_count
            << ", \"basis_full_eigensolver_count\": " << row.basis_full_eigensolver_count
            << ", \"coarse_lambda_sec\": " << jsonNumber(row.coarse_lambda_sec)
            << ", \"exact_joint_assembly_sec\": " << jsonNumber(row.exact_joint_assembly_sec)
            << ", \"sweeps_sec\": " << jsonNumber(row.sweeps_sec)
            << ", \"factor_pass_sec\": " << jsonNumber(row.factor_pass_sec)
            << ", \"variable_pass_sec\": " << jsonNumber(row.variable_pass_sec)
            << ", \"coarse_eta_sec\": " << jsonNumber(row.coarse_eta_sec)
            << ", \"coarse_solve_sec\": " << jsonNumber(row.coarse_solve_sec)
            << ", \"coarse_factorize_sec\": " << jsonNumber(row.coarse_factorize_sec)
            << ", \"coarse_backsolve_sec\": " << jsonNumber(row.coarse_backsolve_sec)
            << ", \"prolong_inject_sec\": " << jsonNumber(row.prolong_inject_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << ", \"relinearize_transport_sec\": "
            << jsonNumber(row.relinearize_transport_sec)
            << ", \"relinearize_factor_sec\": "
            << jsonNumber(row.relinearize_factor_sec)
            << ", \"relinearize_reset_sec\": "
            << jsonNumber(row.relinearize_reset_sec)
            << ", \"relinearize_total_sec\": "
            << jsonNumber(row.relinearize_total_sec)
            << "}";
        out << (i + 1 == results.mg_history.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void writeSyntheticSE3ExperimentPoseHistoryJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open pose-history JSON path: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"num_poses\": " << results.num_poses << ",\n";
    out << "  \"num_edges\": " << results.num_edges << ",\n";
    out << "  \"initial_poses\": ";
    writePoseVectorJson(out, results.initial_poses, 4);
    out << ",\n";
    out << "  \"direct_pose_history\": ";
    writePoseHistoryJsonArray(out, results.direct_pose_history, 4);
    out << ",\n";
    out << "  \"mg_pose_history\": ";
    writePoseHistoryJsonArray(out, results.mg_pose_history, 4);
    out << "\n";
    out << "}\n";
}

}  // namespace slam
