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
    int exact_reuse_count = 0;
    int partial_work_aborts = 0;
    int partial_iterations = 0;
    double exact_cache_sec = 0.0;
    std::vector<unsigned char> exact_reused;
};

struct SparseCholeskyFactor {
    Eigen::SparseMatrix<double> A;
    CoarseCholesky ldlt_solver;
    bool analyzed = false;
    int rows = 0;
    int cols = 0;
    int nonzeros = 0;
    std::vector<int> outer_index;
    std::vector<int> inner_index;

    SparseCholeskyFactor()
        : ldlt_solver([] {
            const char* value = std::getenv("HGBP_SE3_COARSE_CHOLMOD");
            return value && std::string(value) == "1";
        }()) {}
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
