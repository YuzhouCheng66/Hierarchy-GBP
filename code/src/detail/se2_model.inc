using PoseVec = std::vector<Eigen::Vector3d>;
using SteadyClock = std::chrono::steady_clock;

bool getenvEnabled(const char* name) {
    const char* value = std::getenv(name);
    if (!value) {
        return false;
    }
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "False") != 0;
}

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

std::string outerExecutionPathName(
    bool use_packed_residual_solver,
    bool use_fastsync_local_packed,
    bool use_fastsync_local,
    bool use_singlecore_fastsync_soa
) {
    if (use_packed_residual_solver) {
        return "packed_residual/default";
    }
    if (use_fastsync_local_packed) {
        return "fastsync_local_packed";
    }
    if (use_fastsync_local) {
        return "fastsync_local";
    }
    if (use_singlecore_fastsync_soa) {
        return "fastsync_soa";
    }
    return "generic_sync";
}

double maxFactorLamAsymmetry(const gbp::FactorGraph& graph) {
    double max_asym = 0.0;
    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor || !factor->active) {
            continue;
        }
        const int dim = factor->factor.dim();
        const double* lam = factor->factor.lamData();
        if (!lam || dim <= 1) {
            continue;
        }
        for (int c = 0; c < dim; ++c) {
            for (int r = c + 1; r < dim; ++r) {
                const double diff = std::abs(lam[c * dim + r] - lam[r * dim + c]);
                if (diff > max_asym) {
                    max_asym = diff;
                }
            }
        }
    }
    return max_asym;
}

Eigen::Matrix3d info6ToMatrix(const std::array<double, 6>& vals) {
    Eigen::Matrix3d info = Eigen::Matrix3d::Zero();
    info(0, 0) = vals[0];
    info(0, 1) = vals[1];
    info(0, 2) = vals[2];
    info(1, 0) = vals[1];
    info(1, 1) = vals[3];
    info(1, 2) = vals[4];
    info(2, 0) = vals[2];
    info(2, 1) = vals[4];
    info(2, 2) = vals[5];
    return info;
}

const char* syncScheduleName(gbp::FactorGraph::SyncScheduleKind kind) {
    switch (kind) {
    case gbp::FactorGraph::SyncScheduleKind::Dynamic:
        return "dynamic";
    case gbp::FactorGraph::SyncScheduleKind::Guided:
        return "guided";
    case gbp::FactorGraph::SyncScheduleKind::Static:
    default:
        return "static";
    }
}

const char* basisEigensolverName(BasisEigensolverKind kind) {
    switch (kind) {
    case BasisEigensolverKind::Full:
        return "full";
    case BasisEigensolverKind::Partial:
    default:
        return "partial";
    }
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
    const Eigen::Vector3d& err,
    const Eigen::Matrix3d& information,
    const RobustLossConfig& config
) {
    if (!(config.huber_delta > 0.0)) {
        return 1.0;
    }
    const double quad = (err.transpose() * information * err)(0, 0);
    const double mahal_norm = std::sqrt(std::max(0.0, quad));
    return huberWeightFromMahalanobisNorm(mahal_norm, config);
}

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
    const Eigen::Vector3d& z,
    const Eigen::Vector3d& ei,
    const Eigen::Vector3d& ej
) {
    const Eigen::Vector3d xi = se2Plus(base_i, ei);
    const Eigen::Vector3d xj = se2Plus(base_j, ej);
    const Eigen::Vector3d pred = se2Between(xi, xj);
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(z), pred);

    const Eigen::Matrix<double, 3, 6> J_between_abs = jacobianBetweenAbsolute(xi, xj);
    const Eigen::Matrix3d J_compose = jacobianComposeInvConstant(z);
    const Eigen::Matrix3d J_log = jacobianLogSE2(err_pose);

    Eigen::Matrix<double, 6, 6> J_plus = Eigen::Matrix<double, 6, 6>::Zero();
    J_plus.block<3, 3>(0, 0) = jacobianPlusSE2(base_i, ei);
    J_plus.block<3, 3>(3, 3) = jacobianPlusSE2(base_j, ej);

    return J_log * J_compose * J_between_abs * J_plus;
}

Eigen::Matrix3d analyticAnchorResidualJacobian(
    const Eigen::Vector3d& base_anchor,
    const Eigen::Vector3d& anchor_pose,
    const Eigen::Vector3d& ei
) {
    const Eigen::Vector3d xi = se2Plus(base_anchor, ei);
    const Eigen::Vector3d err_pose = se2Compose(se2Inverse(anchor_pose), xi);
    return jacobianLogSE2(err_pose) * jacobianComposeInvConstant(anchor_pose) * jacobianPlusSE2(base_anchor, ei);
}
