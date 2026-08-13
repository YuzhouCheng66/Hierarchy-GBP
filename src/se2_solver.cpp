#include "internal/se2_solver_impl.h"
#include "internal/se2_sync.h"
#include "internal/se2_group_sync.h"
#include "internal/se2_residual.h"
#include "internal/se2_basis.h"
#include "internal/se2_coarse.h"
#include "internal/se2_dense_sync.h"
#include "internal/se2_basis_data.h"
#include "internal/partial_symmetric_eigen.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <omp.h>

#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace slam {

namespace {

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

void injectCorrectionKeepMessages(gbp::FactorGraph& graph, const Eigen::VectorXd& delta) {
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        gbp::VariableNode& var = *vup;
        var.refreshMu();
        var.mu += delta.segment(offset, var.dofs);
        var.belief.setEta(var.belief.lam() * var.mu);
        var.markMuCurrent();
        offset += var.dofs;
    }
}

Eigen::SparseMatrix<double> symmetrizeSparse(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> sym = 0.5 * (A + Eigen::SparseMatrix<double>(A.transpose()));
    sym.makeCompressed();
    return sym;
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

struct SparseCholeskyFactor {
    Eigen::SparseMatrix<double> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    bool analyzed = false;
    int rows = 0;
    int cols = 0;
    int nonzeros = 0;

    SparseCholeskyFactor() = default;
    SparseCholeskyFactor(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor& operator=(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor(SparseCholeskyFactor&&) noexcept = default;
    SparseCholeskyFactor& operator=(SparseCholeskyFactor&&) noexcept = default;
};


void factorizeSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    double ridge,
    SparseCholeskyFactor& factor,
    bool reuse_pattern = false,
    bool assume_symmetric = false
) {
    factor.A = assume_symmetric ? lam : symmetrizeSparse(lam);
    if (ridge != 0.0) {
        for (int i = 0; i < factor.A.rows(); ++i) {
            factor.A.coeffRef(i, i) += ridge;
        }
    }
    factor.A.makeCompressed();

    const bool can_reuse =
        reuse_pattern &&
        factor.analyzed &&
        factor.rows == factor.A.rows() &&
        factor.cols == factor.A.cols() &&
        factor.nonzeros == factor.A.nonZeros();

    if (!can_reuse) {
        factor.solver.analyzePattern(factor.A);
        if (factor.solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky analyzePattern failed");
        }
        factor.analyzed = true;
        factor.rows = static_cast<int>(factor.A.rows());
        factor.cols = static_cast<int>(factor.A.cols());
        factor.nonzeros = static_cast<int>(factor.A.nonZeros());
    }

    factor.solver.factorize(factor.A);
    if (factor.solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }
}

void solveWithSparseCholeskyInto(
    const SparseCholeskyFactor& factor,
    const Eigen::VectorXd& eta,
    Eigen::VectorXd& out
) {
    out = factor.solver.solve(eta);
    if (factor.solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
}

std::vector<std::vector<int>> orderedGroups(int n_vars, int group_size) {
    if (group_size <= 0) {
        throw std::runtime_error("group_size must be positive");
    }
    std::vector<std::vector<int>> groups;
    int start = 0;
    while (start + 2 * group_size <= n_vars) {
        std::vector<int> group;
        group.reserve(group_size);
        for (int id = start; id < start + group_size; ++id) {
            group.push_back(id);
        }
        groups.push_back(std::move(group));
        start += group_size;
    }
    if (start < n_vars) {
        std::vector<int> tail;
        tail.reserve(n_vars - start);
        for (int id = start; id < n_vars; ++id) {
            tail.push_back(id);
        }
        groups.push_back(std::move(tail));
    }
    return groups;
}

Eigen::MatrixXd buildGroupMessageConditionedInformationOrdered3D(
    const gbp::FactorGraph& graph,
    int start_var_id,
    int group_len,
    std::vector<int>& factor_stamp,
    int stamp_value
) {
    const int total_dim = 3 * group_len;
    const int end_var_id = start_var_id + group_len;
    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(total_dim, total_dim);

    auto add_block3 = [&](int row_off, int col_off, const double* block) {
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
                info(row_off + r, col_off + c) += block[3 * c + r];
            }
        }
    };

    auto add_factor_block3 = [&](int row_off, int col_off, const double* factor_lam, int factor_dim, int factor_row, int factor_col) {
        for (int c = 0; c < 3; ++c) {
            const double* src = factor_lam + (factor_col + c) * factor_dim + factor_row;
            info(row_off + 0, col_off + c) += src[0];
            info(row_off + 1, col_off + c) += src[1];
            info(row_off + 2, col_off + c) += src[2];
        }
    };

    for (int local_idx = 0; local_idx < group_len; ++local_idx) {
        const int var_id = start_var_id + local_idx;
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        const int off = 3 * local_idx;
        add_block3(off, off, var->prior.lamData());

        for (const auto& aref : var->adj_factors) {
            const gbp::Factor* factor = aref.factor;
            if (!factor || !factor->active) {
                continue;
            }

            bool factor_inside = true;
            for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                const int adj_id = adj_var->variableID;
                if (adj_id < start_var_id || adj_id >= end_var_id) {
                    factor_inside = false;
                    break;
                }
            }

            if (factor_inside) {
                const int fid = factor->factorID;
                if (factor_stamp[fid] == stamp_value) {
                    continue;
                }
                factor_stamp[fid] = stamp_value;

                const double* factor_lam = factor->factor.lamData();
                const int arity = static_cast<int>(factor->adj_var_nodes.size());
                for (int a = 0; a < arity; ++a) {
                    const int row_off = 3 * (factor->adj_var_nodes[a]->variableID - start_var_id);
                    const int factor_row = 3 * a;
                    for (int b = 0; b < arity; ++b) {
                        const int col_off = 3 * (factor->adj_var_nodes[b]->variableID - start_var_id);
                        const int factor_col = 3 * b;
                        add_factor_block3(row_off, col_off, factor_lam, factor->factor.dim(), factor_row, factor_col);
                    }
                }
            } else {
                add_block3(off, off, factor->messages[aref.local_idx].lamData());
            }
        }
    }

    return 0.5 * (info + info.transpose());
}

void buildGroupMessageConditionedInformationOrdered3DFastPlan(
    const BasisData::GroupFastPlan& plan,
    int total_dim,
    Eigen::MatrixXd& info
) {
    info.setZero(total_dim, total_dim);

    auto add_block3 = [&](int row_off, int col_off, const double* block) {
        for (int c = 0; c < 3; ++c) {
            info(row_off + 0, col_off + c) += block[3 * c + 0];
            info(row_off + 1, col_off + c) += block[3 * c + 1];
            info(row_off + 2, col_off + c) += block[3 * c + 2];
        }
    };

    auto add_factor_block3 = [&](int row_off, int col_off, const double* src, int dim, int row0, int col0) {
        for (int c = 0; c < 3; ++c) {
            const double* src_col = src + (col0 + c) * dim + row0;
            info(row_off + 0, col_off + c) += src_col[0];
            info(row_off + 1, col_off + c) += src_col[1];
            info(row_off + 2, col_off + c) += src_col[2];
        }
    };

    for (size_t i = 0; i < plan.prior_lam_ptrs.size(); ++i) {
        add_block3(plan.prior_row_offs[i], plan.prior_row_offs[i], plan.prior_lam_ptrs[i]);
    }
    for (const BasisData::GroupBoundary3Plan& msg : plan.boundary_msgs) {
        add_block3(msg.row_off, msg.row_off, msg.factor->messages[msg.local_idx].lamData());
    }
    for (const BasisData::GroupUnary3Plan& unary : plan.interior_unary) {
        add_block3(unary.row_off, unary.row_off, unary.factor->factor.lamData());
    }
    for (const BasisData::GroupBinary3Plan& binary : plan.interior_binary) {
        const double* lam = binary.factor->factor.lamData();
        add_factor_block3(binary.row0_off, binary.row0_off, lam, 6, 0, 0);
        add_factor_block3(binary.row0_off, binary.row1_off, lam, 6, 0, 3);
        add_factor_block3(binary.row1_off, binary.row0_off, lam, 6, 3, 0);
        add_factor_block3(binary.row1_off, binary.row1_off, lam, 6, 3, 3);
    }

    for (int c = 0; c < total_dim; ++c) {
        for (int r = 0; r < c; ++r) {
            const double sym = 0.5 * (info(r, c) + info(c, r));
            info(r, c) = sym;
            info(c, r) = sym;
        }
    }
}

BasisData buildMessageConditionedBasis(
    const gbp::FactorGraph& graph,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    BasisData basis;
    basis.groups = orderedGroups(static_cast<int>(graph.var_nodes.size()), group_size);

    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    basis.total_dim = total_dim;

    basis.coarse_offsets.push_back(0);
    basis.var_to_group.assign(graph.var_nodes.size(), -1);
    basis.var_to_local_offset.assign(graph.var_nodes.size(), -1);
    basis.ordered_3d_fast = true;
    basis.factors_arity12_only_3d = true;
    basis.var_r_local.assign(graph.var_nodes.size(), 0);
    basis.var_coarse_offset.assign(graph.var_nodes.size(), 0);
    basis.var_basis_offset.assign(graph.var_nodes.size(), -1);
    basis.full_indices_per_group.resize(basis.groups.size());
    basis.local_bases.resize(basis.groups.size());
    basis.group_fast_plans.resize(basis.groups.size());
    std::vector<int> r_locals(basis.groups.size(), 0);
    int reduced_offset = 0;

    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        std::vector<int> full_indices;
        full_indices.reserve(group.size() * 3);
        int local_scalar_offset = 0;
        for (int var_id : group) {
            const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
            const int base = 3 * var_id;
            for (int d = 0; d < var->dofs; ++d) {
                full_indices.push_back(base + d);
            }
            basis.var_to_group[var_id] = g;
            basis.var_to_local_offset[var_id] = local_scalar_offset;
            if (var->dofs != 3) {
                basis.ordered_3d_fast = false;
            }
            local_scalar_offset += var->dofs;
        }
        const int block_dim = static_cast<int>(full_indices.size());
        const int r_local = std::min(r_reduced, block_dim);
        basis.full_indices_per_group[g] = std::move(full_indices);
        r_locals[g] = r_local;
        reduced_offset += r_local;
        basis.coarse_offsets.push_back(reduced_offset);
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor) {
            continue;
        }
        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity < 1 || arity > 2) {
            basis.factors_arity12_only_3d = false;
            break;
        }
        for (const gbp::VariableNode* var : factor->adj_var_nodes) {
            if (!var || var->dofs != 3) {
                basis.factors_arity12_only_3d = false;
                break;
            }
        }
        if (!basis.factors_arity12_only_3d) {
            break;
        }
    }

    if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            BasisData::GroupFastPlan& plan = basis.group_fast_plans[g];
            const std::vector<int>& group = basis.groups[g];
            const int start_var_id = group.front();
            const int end_var_id = start_var_id + static_cast<int>(group.size());
            ++stamp_value;

            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int var_id = group[local_idx];
                const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
                const int row_off = 3 * local_idx;
                plan.prior_lam_ptrs.push_back(var->prior.lamData());
                plan.prior_row_offs.push_back(row_off);

                for (const auto& aref : var->adj_factors) {
                    const gbp::Factor* factor = aref.factor;
                    if (!factor) {
                        continue;
                    }

                    bool factor_inside = true;
                    for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                        const int adj_id = adj_var->variableID;
                        if (adj_id < start_var_id || adj_id >= end_var_id) {
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
                        const int arity = static_cast<int>(factor->adj_var_nodes.size());
                        if (arity == 1) {
                            plan.interior_unary.push_back({factor, row_off});
                        } else if (arity == 2) {
                            const int id0 = factor->adj_var_nodes[0]->variableID;
                            const int id1 = factor->adj_var_nodes[1]->variableID;
                            const int row0 = 3 * (id0 - start_var_id);
                            const int row1 = 3 * (id1 - start_var_id);
                            plan.interior_binary.push_back({factor, row0, row1});
                        }
                    } else {
                        plan.boundary_msgs.push_back({factor, aref.local_idx, row_off});
                    }
                }
            }
        }
    }

    size_t total_basis_scalars = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        total_basis_scalars += static_cast<size_t>(basis.groups[g].size()) * 3 * r_locals[g];
    }
    basis.var_basis_blocks.reserve(total_basis_scalars);

    const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
    const bool do_parallel = use_parallel && thread_count > 1 && basis.groups.size() >= 8;
    const bool use_partial_eigensolver = basis_build_config.eigensolver == BasisEigensolverKind::Partial;
    const PartialSymmetricEigenOptions partial_opts{
        basis_build_config.partial_oversampling,
        basis_build_config.partial_max_iters,
        basis_build_config.partial_residual_tol,
        basis_build_config.partial_ridge,
        basis_build_config.partial_residual_check_period,
    };
    if (do_parallel) {
        std::vector<std::vector<int>> tls_factor_stamp(
            thread_count,
            std::vector<int>(graph.factors.size(), -1)
        );
        std::vector<int> tls_stamp_value(thread_count, 0);
        std::vector<PartialSymmetricEigenWorkspace> tls_partial_ws(thread_count);

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const int tid = omp_get_thread_num();
            const std::vector<int>& group = basis.groups[g];
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            Eigen::MatrixXd block;
            if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
                buildGroupMessageConditionedInformationOrdered3DFastPlan(
                    basis.group_fast_plans[g],
                    block_dim,
                    block
                );
            } else {
                block = buildGroupMessageConditionedInformationOrdered3D(
                    graph,
                    group.front(),
                    static_cast<int>(group.size()),
                    tls_factor_stamp[tid],
                    ++tls_stamp_value[tid]
                );
            }
            const int r_local = r_locals[g];

            Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
            if (r_local < block_dim) {
                const Eigen::MatrixXd* warm_basis =
                    (use_partial_eigensolver &&
                     basis_build_config.enable_warm_start &&
                     warm_start_local_bases &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() == r_local)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                bool used_partial = false;
                if (use_partial_eigensolver) {
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, tls_partial_ws[tid], partial_opts);
                    #pragma omp atomic
                    basis.partial_attempted += 1;
                    #pragma omp atomic
                    basis.partial_total_iters += partial.iterations;
                    if (partial.converged &&
                        partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local) {
                        local_basis = partial.eigenvectors;
                        used_partial = true;
                        #pragma omp atomic
                        basis.partial_converged += 1;
                    } else {
                        #pragma omp atomic
                        basis.partial_fallback += 1;
                    }
                }
                if (!used_partial) {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(block);
                    if (es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose conditioned information block");
                    }
                    const Eigen::MatrixXd evecs = es.eigenvectors();
                    local_basis.resize(block_dim, r_local);
                    for (int col = 0; col < r_local; ++col) {
                        local_basis.col(col) = evecs.col(col);
                    }
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    } else {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        PartialSymmetricEigenWorkspace partial_ws;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            Eigen::MatrixXd block;
            if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
                buildGroupMessageConditionedInformationOrdered3DFastPlan(
                    basis.group_fast_plans[g],
                    block_dim,
                    block
                );
            } else {
                block = buildGroupMessageConditionedInformationOrdered3D(
                    graph,
                    group.front(),
                    static_cast<int>(group.size()),
                    factor_stamp,
                    ++stamp_value
                );
            }
            const int r_local = r_locals[g];

            Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
            if (r_local < block_dim) {
                const Eigen::MatrixXd* warm_basis =
                    (use_partial_eigensolver &&
                     basis_build_config.enable_warm_start &&
                     warm_start_local_bases &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() == r_local)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                bool used_partial = false;
                if (use_partial_eigensolver) {
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
                    basis.partial_attempted += 1;
                    basis.partial_total_iters += partial.iterations;
                    if (partial.converged &&
                        partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local) {
                        local_basis = partial.eigenvectors;
                        used_partial = true;
                        basis.partial_converged += 1;
                    } else {
                        basis.partial_fallback += 1;
                    }
                }
                if (!used_partial) {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(block);
                    if (es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose conditioned information block");
                    }
                    const Eigen::MatrixXd evecs = es.eigenvectors();
                    local_basis.resize(block_dim, r_local);
                    for (int col = 0; col < r_local; ++col) {
                        local_basis.col(col) = evecs.col(col);
                    }
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    }

    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        const Eigen::MatrixXd& local_basis = basis.local_bases[g];
        const int r_local = local_basis.cols();
        for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
            const int var_id = group[local_var];
            basis.var_r_local[var_id] = r_local;
            basis.var_coarse_offset[var_id] = basis.coarse_offsets[g];
            basis.var_basis_offset[var_id] = static_cast<int>(basis.var_basis_blocks.size());
            basis.var_basis_blocks.resize(basis.var_basis_blocks.size() + 3 * r_local);
            double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
            const int row0 = 3 * local_var;
            for (int c = 0; c < r_local; ++c) {
                dst[3 * c + 0] = local_basis(row0 + 0, c);
                dst[3 * c + 1] = local_basis(row0 + 1, c);
                dst[3 * c + 2] = local_basis(row0 + 2, c);
            }
        }
    }

    basis.coarse_dim = reduced_offset;
    return basis;
}

bool rowsAreContiguous(const std::vector<int>& rows) {
    if (rows.empty()) {
        return false;
    }
    const int first = rows.front();
    for (int i = 1; i < static_cast<int>(rows.size()); ++i) {
        if (rows[static_cast<size_t>(i)] != first + i) {
            return false;
        }
    }
    return true;
}

void restrictToCoarseInto(
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse
) {
    if (coarse.size() != basis.coarse_dim) {
        coarse.resize(basis.coarse_dim);
    }
    coarse.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const std::vector<int>& rows = basis.full_indices_per_group[g];
        const int coarse_offset = basis.coarse_offsets[g];
        const int coarse_dim = basis.local_bases[g].cols();
        if (rowsAreContiguous(rows)) {
            coarse.segment(coarse_offset, coarse_dim).noalias() =
                basis.local_bases[g].transpose() * fine_vec.segment(rows.front(), rows.size());
        } else {
            Eigen::VectorXd fine_local(rows.size());
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine_local(i) = fine_vec(rows[i]);
            }
            coarse.segment(coarse_offset, coarse_dim).noalias() =
                basis.local_bases[g].transpose() * fine_local;
        }
    }
}

void prolongToFineInto(
    const BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
) {
    if (fine.size() != basis.total_dim) {
        fine.resize(basis.total_dim);
    }
    fine.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const std::vector<int>& rows = basis.full_indices_per_group[g];
        const int coarse_offset = basis.coarse_offsets[g];
        const int coarse_dim = basis.local_bases[g].cols();
        if (rowsAreContiguous(rows)) {
            fine.segment(rows.front(), rows.size()).noalias() =
                basis.local_bases[g] * coarse_vec.segment(coarse_offset, coarse_dim);
        } else {
            const Eigen::VectorXd fine_local =
                basis.local_bases[g] * coarse_vec.segment(coarse_offset, coarse_dim);
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine(rows[i]) = fine_local(i);
            }
        }
    }
}

inline void accumBasisTransposeTimesVec3(
    const double* basis_block,
    int r_local,
    const double* vec3,
    double* coarse_segment
) {
    for (int c = 0; c < r_local; ++c) {
        const double* b = basis_block + 3 * c;
        coarse_segment[c] += b[0] * vec3[0] + b[1] * vec3[1] + b[2] * vec3[2];
    }
}

inline void accumBasisTransposeLamBasis3x3(
    const double* basis_a,
    int r_a,
    const double* lam3,
    const double* basis_b,
    int r_b,
    double* dense,
    int dense_cols,
    int row_offset,
    int col_offset
) {
    for (int cb = 0; cb < r_b; ++cb) {
        const double* bb = basis_b + 3 * cb;
        const double lb0 = lam3[0] * bb[0] + lam3[3] * bb[1] + lam3[6] * bb[2];
        const double lb1 = lam3[1] * bb[0] + lam3[4] * bb[1] + lam3[7] * bb[2];
        const double lb2 = lam3[2] * bb[0] + lam3[5] * bb[1] + lam3[8] * bb[2];
        for (int ca = 0; ca < r_a; ++ca) {
            const double* ba = basis_a + 3 * ca;
            dense[(col_offset + cb) * dense_cols + (row_offset + ca)] +=
                ba[0] * lb0 + ba[1] * lb1 + ba[2] * lb2;
        }
    }
}

inline void packFactorBlock3x3(
    const double* src,
    int dim,
    int row0,
    int col0,
    double* dst
) {
    for (int c = 0; c < 3; ++c) {
        const double* src_col = src + (col0 + c) * dim + row0;
        dst[3 * c + 0] = src_col[0];
        dst[3 * c + 1] = src_col[1];
        dst[3 * c + 2] = src_col[2];
    }
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectOrdered3D(
    const gbp::FactorGraph& graph,
    const BasisData& basis
) {
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(basis.coarse_dim, basis.coarse_dim);
    double* dense_data = dense.data();

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        if (var.dofs != 3) {
            continue;
        }
        const int var_id = var.variableID;
        const int r_local = basis.var_r_local[var_id];
        const int coarse_offset = basis.var_coarse_offset[var_id];
        const double* basis_block = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3(
            basis_block,
            r_local,
            var.prior.lamData(),
            basis_block,
            r_local,
            dense_data,
            basis.coarse_dim,
            coarse_offset,
            coarse_offset
        );
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor || !factor->active) {
            continue;
        }

        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity == 1 && factor->adj_var_nodes[0]->dofs == 3) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const int id0 = v0->variableID;
            const int r0 = basis.var_r_local[id0];
            const int off0 = basis.var_coarse_offset[id0];
            const double* B0 = varBasis3xRPtr(basis, id0);
            accumBasisTransposeLamBasis3x3(
                B0, r0, factor->factor.lamData(), B0, r0, dense_data, basis.coarse_dim, off0, off0
            );
            continue;
        }

        if (arity == 2 &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor->adj_var_nodes[1];
            const int id0 = v0->variableID;
            const int id1 = v1->variableID;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* lam = factor->factor.lamData();
            double block00[9];
            double block01[9];
            double block10[9];
            double block11[9];
            packFactorBlock3x3(lam, 6, 0, 0, block00);
            packFactorBlock3x3(lam, 6, 0, 3, block01);
            packFactorBlock3x3(lam, 6, 3, 0, block10);
            packFactorBlock3x3(lam, 6, 3, 3, block11);

            accumBasisTransposeLamBasis3x3(B0, r0, block00, B0, r0, dense_data, basis.coarse_dim, off0, off0);
            accumBasisTransposeLamBasis3x3(B0, r0, block01, B1, r1, dense_data, basis.coarse_dim, off0, off1);
            accumBasisTransposeLamBasis3x3(B1, r1, block10, B0, r0, dense_data, basis.coarse_dim, off1, off0);
            accumBasisTransposeLamBasis3x3(B1, r1, block11, B1, r1, dense_data, basis.coarse_dim, off1, off1);
            continue;
        }
    }

    Eigen::SparseMatrix<double> coarse = dense.sparseView(0.0, 0.0);
    coarse.makeCompressed();
    return symmetrizeSparse(coarse);
}

Eigen::VectorXd assembleCoarseResidualDirectOrdered3D(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec
) {
    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);
    double* coarse_data = coarse.data();
    const double* fine_data = fine_vec.data();

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        if (var.dofs != 3) {
            continue;
        }
        const int var_id = var.variableID;
        const int r_local = basis.var_r_local[var_id];
        const int coarse_offset = basis.var_coarse_offset[var_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        const double* x = fine_data + 3 * var_id;
        const double* eta = var.prior.etaData();
        const double* lam = var.prior.lamData();
        double residual[3] = {
            eta[0] - (lam[0] * x[0] + lam[3] * x[1] + lam[6] * x[2]),
            eta[1] - (lam[1] * x[0] + lam[4] * x[1] + lam[7] * x[2]),
            eta[2] - (lam[2] * x[0] + lam[5] * x[1] + lam[8] * x[2]),
        };
        accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + coarse_offset);
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor || !factor->active) {
            continue;
        }

        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity == 1 && factor->adj_var_nodes[0]->dofs == 3) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const int id0 = v0->variableID;
            const int r0 = basis.var_r_local[id0];
            const int off0 = basis.var_coarse_offset[id0];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* x0 = fine_data + 3 * id0;
            const double* eta = factor->factor.etaData();
            const double* lam = factor->factor.lamData();
            double residual[3] = {
                eta[0] - (lam[0] * x0[0] + lam[3] * x0[1] + lam[6] * x0[2]),
                eta[1] - (lam[1] * x0[0] + lam[4] * x0[1] + lam[7] * x0[2]),
                eta[2] - (lam[2] * x0[0] + lam[5] * x0[1] + lam[8] * x0[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, residual, coarse_data + off0);
            continue;
        }

        if (arity == 2 &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor->adj_var_nodes[1];
            const int id0 = v0->variableID;
            const int id1 = v1->variableID;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* x0 = fine_data + 3 * id0;
            const double* x1 = fine_data + 3 * id1;
            const double* eta = factor->factor.etaData();
            const double* lam = factor->factor.lamData();
            double r0_vec[3] = {
                eta[0] - (lam[0] * x0[0] + lam[6] * x0[1] + lam[12] * x0[2] + lam[18] * x1[0] + lam[24] * x1[1] + lam[30] * x1[2]),
                eta[1] - (lam[1] * x0[0] + lam[7] * x0[1] + lam[13] * x0[2] + lam[19] * x1[0] + lam[25] * x1[1] + lam[31] * x1[2]),
                eta[2] - (lam[2] * x0[0] + lam[8] * x0[1] + lam[14] * x0[2] + lam[20] * x1[0] + lam[26] * x1[1] + lam[32] * x1[2]),
            };
            double r1_vec[3] = {
                eta[3] - (lam[3] * x0[0] + lam[9] * x0[1] + lam[15] * x0[2] + lam[21] * x1[0] + lam[27] * x1[1] + lam[33] * x1[2]),
                eta[4] - (lam[4] * x0[0] + lam[10] * x0[1] + lam[16] * x0[2] + lam[22] * x1[0] + lam[28] * x1[1] + lam[34] * x1[2]),
                eta[5] - (lam[5] * x0[0] + lam[11] * x0[1] + lam[17] * x0[2] + lam[23] * x1[0] + lam[29] * x1[1] + lam[35] * x1[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
            accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
            continue;
        }
    }

    return coarse;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    bool use_parallel,
    int num_threads
) {
    if (num_threads <= 1 && basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        return assembleCoarseLambdaDirectOrdered3D(graph, basis);
    }

    const int num_groups = static_cast<int>(basis.local_bases.size());
    const int num_block_pairs = num_groups * num_groups;

    std::vector<Eigen::MatrixXd> blocks(num_block_pairs);
    std::vector<char> active(num_block_pairs, 0);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const auto basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const int block_index = group * num_groups + group;
        if (!active[block_index]) {
            blocks[block_index] = Eigen::MatrixXd::Zero(basis_block.cols(), basis_block.cols());
            active[block_index] = 1;
        }
        blocks[block_index].noalias() += basis_block.transpose() * var.prior.lam() * basis_block;
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
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<std::vector<Eigen::MatrixXd>> tls_blocks(thread_count);
        std::vector<std::vector<char>> tls_active(thread_count);
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
            if (!fup) {
                continue;
            }
            accumulate_factor(*fup, blocks, active);
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    for (int gi = 0; gi < num_groups; ++gi) {
        const int row_offset = basis.coarse_offsets[gi];
        for (int gj = 0; gj < num_groups; ++gj) {
            const int block_index = gi * num_groups + gj;
            if (!active[block_index]) {
                continue;
            }
            const int col_offset = basis.coarse_offsets[gj];
            const Eigen::MatrixXd& block = blocks[block_index];
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    const double value = block(r, c);
                    if (value != 0.0) {
                        trips.emplace_back(row_offset + r, col_offset + c, value);
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

Eigen::VectorXd assembleCoarseResidualDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    bool use_parallel,
    int num_threads
) {
    if (num_threads <= 1 && basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        return assembleCoarseResidualDirectOrdered3D(graph, basis, fine_vec);
    }

    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const auto basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const auto x_local = fine_vec.segment(3 * var.variableID, var.dofs);
        coarse.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
            basis_block.transpose() * (var.prior.eta() - var.prior.lam() * x_local);
    }

    auto accumulate_factor_residual = [&](const gbp::Factor& factor, Eigen::VectorXd& target) {
        if (!factor.active) {
            return;
        }

        const int arity = static_cast<int>(factor.adj_var_nodes.size());
        if (arity == 1) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const int g0 = basis.var_to_group[v0->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int d0 = v0->dofs;
            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0;
            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            return;
        }

        if (arity == 2) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor.adj_var_nodes[1];
            const int d0 = v0->dofs;
            const int d1 = v1->dofs;

            const int g0 = basis.var_to_group[v0->variableID];
            const int g1 = basis.var_to_group[v1->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int l1 = basis.var_to_local_offset[v1->variableID];

            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::MatrixXd B1 = basis.local_bases[g1].middleRows(l1, d1);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd x1 = fine_vec.segment(3 * v1->variableID, d1);

            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0 -
                factor.factor.lam().block(0, d0, d0, d1) * x1;
            const Eigen::VectorXd r1 =
                factor.factor.eta().segment(d0, d1) -
                factor.factor.lam().block(d0, 0, d1, d0) * x0 -
                factor.factor.lam().block(d0, d0, d1, d1) * x1;

            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            target.segment(basis.coarse_offsets[g1], B1.cols()).noalias() += B1.transpose() * r1;
            return;
        }

        std::vector<int> local_factor_offsets(arity, 0);
        int factor_dim = 0;
        for (int a = 0; a < arity; ++a) {
            local_factor_offsets[a] = factor_dim;
            factor_dim += factor.adj_var_nodes[a]->dofs;
        }

        Eigen::VectorXd factor_x = Eigen::VectorXd::Zero(factor_dim);
        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            factor_x.segment(local_factor_offsets[a], va->dofs) = fine_vec.segment(3 * va->variableID, va->dofs);
        }
        const Eigen::VectorXd factor_residual = factor.factor.eta() - factor.factor.lam() * factor_x;

        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int group = basis.var_to_group[va->variableID];
            const int local_offset = basis.var_to_local_offset[va->variableID];
            const int dofs = va->dofs;
            const Eigen::MatrixXd basis_block = basis.local_bases[group].middleRows(local_offset, dofs);
            target.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
                basis_block.transpose() * factor_residual.segment(local_factor_offsets[a], dofs);
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<Eigen::VectorXd> tls(thread_count, Eigen::VectorXd::Zero(basis.coarse_dim));

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor_residual(*factor, tls[tid]);
        }

        for (const Eigen::VectorXd& local : tls) {
            coarse += local;
        }
    } else {
        for (const auto& fup : graph.factors) {
            if (!fup) {
                continue;
            }
            accumulate_factor_residual(*fup, coarse);
        }
    }

    return coarse;
}

std::string jsonNumber(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

void writePoseVectorJson(std::ofstream& out, const std::vector<Eigen::Vector3d>& poses, int indent_spaces) {
    const std::string indent(indent_spaces, ' ');
    out << "[\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const Eigen::Vector3d& p = poses[i];
        out << indent << "  [" << jsonNumber(p(0)) << ", " << jsonNumber(p(1)) << ", " << jsonNumber(p(2)) << "]";
        out << (i + 1 == poses.size() ? "\n" : ",\n");
    }
    out << indent << "]";
}

void writePoseHistoryJsonArray(std::ofstream& out, const std::vector<std::vector<Eigen::Vector3d>>& history, int indent_spaces) {
    const std::string indent(indent_spaces, ' ');
    out << "[\n";
    for (size_t i = 0; i < history.size(); ++i) {
        out << indent << "  ";
        writePoseVectorJson(out, history[i], indent_spaces + 2);
        out << (i + 1 == history.size() ? "\n" : ",\n");
    }
    out << indent << "]";
}

}  // namespace

SyntheticSE2Problem loadSyntheticSE2Problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open synthetic SE2 problem file: " + path);
    }

    std::string token;
    in >> token;
    if (token == "SYNTHETIC_SE2_PROBLEM") {
        int n = 0;
        in >> token >> n;
        if (token != "N") {
            throw std::runtime_error("Expected N section in " + path);
        }

        SyntheticSE2Problem problem;
        problem.gt_poses.resize(n, Eigen::Vector3d::Zero());
        problem.init_poses.resize(n, Eigen::Vector3d::Zero());

        in >> token;
        if (token != "POSES") {
            throw std::runtime_error("Expected POSES section in " + path);
        }
        for (int idx = 0; idx < n; ++idx) {
            int pose_id = -1;
            in >> pose_id
               >> problem.gt_poses[idx](0) >> problem.gt_poses[idx](1) >> problem.gt_poses[idx](2)
               >> problem.init_poses[idx](0) >> problem.init_poses[idx](1) >> problem.init_poses[idx](2);
            if (pose_id != idx) {
                throw std::runtime_error("Pose ids must be contiguous in synthetic problem file");
            }
        }

        in >> token;
        if (token != "ANCHOR") {
            throw std::runtime_error("Expected ANCHOR section in " + path);
        }
        in >> problem.anchor_pose(0) >> problem.anchor_pose(1) >> problem.anchor_pose(2);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                in >> problem.anchor_information(r, c);
            }
        }

        int num_edges = 0;
        in >> token >> num_edges;
        if (token != "EDGES") {
            throw std::runtime_error("Expected EDGES section in " + path);
        }
        problem.edges.reserve(num_edges);
        for (int e = 0; e < num_edges; ++e) {
            SyntheticSE2Edge edge;
            in >> edge.i >> edge.j >> edge.kind
               >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    in >> edge.information(r, c);
                }
            }
            problem.edges.push_back(edge);
        }

        return problem;
    }

    if (token == "VERTEX_SE2" || token == "EDGE_SE2") {
        in.close();
        std::ifstream gin(path);
        if (!gin) {
            throw std::runtime_error("Failed to reopen g2o problem file: " + path);
        }

        std::unordered_map<int, Eigen::Vector3d> raw_vertices;
        struct RawEdge {
            int vi = -1;
            int vj = -1;
            Eigen::Vector3d measurement = Eigen::Vector3d::Zero();
            Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
        };
        std::vector<RawEdge> raw_edges;

        std::string line;
        int lineno = 0;
        while (std::getline(gin, line)) {
            ++lineno;
            std::istringstream iss(line);
            std::string tag;
            if (!(iss >> tag)) {
                continue;
            }
            if (tag == "VERTEX_SE2") {
                int vid = -1;
                Eigen::Vector3d pose = Eigen::Vector3d::Zero();
                if (!(iss >> vid >> pose(0) >> pose(1) >> pose(2))) {
                    throw std::runtime_error("Malformed VERTEX_SE2 line in " + path + ":" + std::to_string(lineno));
                }
                raw_vertices[vid] = pose;
            } else if (tag == "EDGE_SE2") {
                RawEdge edge;
                std::array<double, 6> info_vals{};
                if (!(iss >> edge.vi >> edge.vj
                    >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2)
                    >> info_vals[0] >> info_vals[1] >> info_vals[2]
                    >> info_vals[3] >> info_vals[4] >> info_vals[5])) {
                    throw std::runtime_error("Malformed EDGE_SE2 line in " + path + ":" + std::to_string(lineno));
                }
                edge.information = info6ToMatrix(info_vals);
                raw_edges.push_back(edge);
            } else {
                throw std::runtime_error("Unsupported tag in g2o file " + path + ":" + std::to_string(lineno) + " tag=" + tag);
            }
        }

        if (raw_vertices.empty()) {
            throw std::runtime_error("No VERTEX_SE2 entries found in g2o file: " + path);
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

        SyntheticSE2Problem problem;
        problem.gt_poses.resize(original_ids.size(), Eigen::Vector3d::Zero());
        problem.init_poses.resize(original_ids.size(), Eigen::Vector3d::Zero());
        for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
            const Eigen::Vector3d pose = raw_vertices.at(original_ids[idx]);
            problem.gt_poses[idx] = pose;
            problem.init_poses[idx] = pose;
        }
        problem.anchor_pose = problem.init_poses.front();
        problem.anchor_information = 1e8 * Eigen::Matrix3d::Identity();

        problem.edges.reserve(raw_edges.size());
        for (const RawEdge& raw_edge : raw_edges) {
            const auto it_i = id_to_idx.find(raw_edge.vi);
            const auto it_j = id_to_idx.find(raw_edge.vj);
            if (it_i == id_to_idx.end() || it_j == id_to_idx.end()) {
                throw std::runtime_error("g2o edge references unknown vertex id in " + path);
            }
            SyntheticSE2Edge edge;
            edge.i = it_i->second;
            edge.j = it_j->second;
            edge.measurement = raw_edge.measurement;
            edge.information = raw_edge.information;
            edge.kind = (std::abs(raw_edge.vi - raw_edge.vj) == 1) ? "odometry" : "loop";
            problem.edges.push_back(edge);
        }

        return problem;
    }

    throw std::runtime_error("Unexpected problem header in " + path + ": " + token);
}

double nonlinearObjective(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& poses,
    int num_threads
) {
    double total = 0.0;
    const int thread_count = std::max(1, num_threads <= 0 ? omp_get_max_threads() : num_threads);
    const bool use_parallel = thread_count > 1 && static_cast<int>(problem.edges.size()) >= 256;
    if (!use_parallel) {
        for (const SyntheticSE2Edge& edge : problem.edges) {
            const Eigen::Vector3d pred = se2Between(poses.at(edge.i), poses.at(edge.j));
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(edge.measurement), pred));
            total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
        }
    } else {
        #pragma omp parallel for schedule(static) reduction(+:total) num_threads(thread_count)
        for (int edge_idx = 0; edge_idx < static_cast<int>(problem.edges.size()); ++edge_idx) {
            const SyntheticSE2Edge& edge = problem.edges[edge_idx];
            const Eigen::Vector3d pred = se2Between(poses.at(edge.i), poses.at(edge.j));
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(edge.measurement), pred));
            total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
        }
    }
    const Eigen::Vector3d anchor_err = se2Log(se2Compose(se2Inverse(problem.anchor_pose), poses.at(0)));
    total += 0.5 * (anchor_err.transpose() * problem.anchor_information * anchor_err)(0, 0);
    return total;
}

std::vector<Eigen::Vector3d> applyPoseDeltas(
    const std::vector<Eigen::Vector3d>& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    std::vector<Eigen::Vector3d> out(base_poses.size(), Eigen::Vector3d::Zero());
    for (int i = 0; i < static_cast<int>(base_poses.size()); ++i) {
        out[i] = se2Plus(base_poses[i], delta_vec.segment<3>(3 * i));
    }
    return out;
}

struct SyntheticResidualWorkspaceBuildStats {
    double variable_alloc_sec = 0.0;
    double factor_alloc_sec = 0.0;
    double connect_sec = 0.0;
    double total_sec = 0.0;
};

struct SyntheticResidualRelinearizeStats {
    double factor_relinearize_sec = 0.0;
    double reset_state_sec = 0.0;
    double total_sec = 0.0;
};

struct SyntheticResidualGraphWorkspace {
    gbp::FactorGraph graph;
    std::vector<gbp::VariableNode*> vars;
    std::vector<gbp::Factor*> edge_factors;
    gbp::Factor* anchor_factor = nullptr;
    double tiny_prior = 1e-12;
};

std::vector<Eigen::VectorXd> emptyMeasurementBlocks(const Eigen::VectorXd&) {
    return {};
}

std::vector<Eigen::MatrixXd> emptyJacobianBlocks(const Eigen::VectorXd&) {
    return {};
}

SyntheticResidualGraphWorkspace buildSyntheticResidualGraphWorkspace(
    const SyntheticSE2Problem& problem,
    double tiny_prior,
    SyntheticResidualWorkspaceBuildStats* stats = nullptr
) {
    const auto total_t0 = SteadyClock::now();

    SyntheticResidualGraphWorkspace workspace;
    workspace.tiny_prior = tiny_prior;
    workspace.graph.nonlinear_factors = false;
    workspace.graph.eta_damping = 0.0;

    const int n = static_cast<int>(problem.gt_poses.size());
    workspace.graph.var_nodes.reserve(n);
    workspace.graph.var_residual.reserve(n);
    workspace.graph.factors.reserve(problem.edges.size() + 1);
    workspace.vars.resize(n, nullptr);
    workspace.edge_factors.resize(problem.edges.size(), nullptr);

    std::vector<int> variable_degree(n, 0);
    for (const SyntheticSE2Edge& edge : problem.edges) {
        ++variable_degree[edge.i];
        ++variable_degree[edge.j];
    }
    if (!variable_degree.empty()) {
        ++variable_degree[0];
    }

    const auto vars_t0 = SteadyClock::now();
    for (int i = 0; i < n; ++i) {
        gbp::VariableNode* var = workspace.graph.addVariable(i, 3);
        var->GT = problem.gt_poses[i];
        var->adj_factors.reserve(variable_degree[i]);
        var->adj_factors_raw.reserve(variable_degree[i]);
        var->prior.setLam(tiny_prior * Eigen::Matrix3d::Identity());
        var->prior.setEta(Eigen::Vector3d::Zero());
        var->belief.setLam(tiny_prior * Eigen::Matrix3d::Identity());
        var->belief.setEta(Eigen::Vector3d::Zero());
        var->mu = Eigen::Vector3d::Zero();
        var->markMuCurrent();
        workspace.vars[i] = var;
    }
    const auto vars_t1 = SteadyClock::now();

    int factor_id = 0;
    double factor_alloc_sec = 0.0;
    double connect_sec = 0.0;

    for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
        const SyntheticSE2Edge& edge = problem.edges[edge_index];
        gbp::VariableNode* vi = workspace.vars.at(edge.i);
        gbp::VariableNode* vj = workspace.vars.at(edge.j);

        const auto factor_t0 = SteadyClock::now();
        gbp::Factor* factor = workspace.graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{vi, vj},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        const auto factor_t1 = SteadyClock::now();
        factor_alloc_sec += elapsedSeconds(factor_t0, factor_t1);

        const auto connect_t0 = SteadyClock::now();
        workspace.graph.connect(factor, vi, 0);
        workspace.graph.connect(factor, vj, 1);
        const auto connect_t1 = SteadyClock::now();
        connect_sec += elapsedSeconds(connect_t0, connect_t1);

        workspace.edge_factors[edge_index] = factor;
    }

    {
        gbp::VariableNode* v0 = workspace.vars.at(0);
        const auto factor_t0 = SteadyClock::now();
        workspace.anchor_factor = workspace.graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{v0},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        const auto factor_t1 = SteadyClock::now();
        factor_alloc_sec += elapsedSeconds(factor_t0, factor_t1);

        const auto connect_t0 = SteadyClock::now();
        workspace.graph.connect(workspace.anchor_factor, v0, 0);
        const auto connect_t1 = SteadyClock::now();
        connect_sec += elapsedSeconds(connect_t0, connect_t1);
    }

    if (stats) {
        stats->variable_alloc_sec = elapsedSeconds(vars_t0, vars_t1);
        stats->factor_alloc_sec = factor_alloc_sec;
        stats->connect_sec = connect_sec;
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
    return workspace;
}

void resetSyntheticResidualState(
    SyntheticResidualGraphWorkspace& workspace,
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
        if (d == 2) {
            var->mu2.setZero();
        }
        var->mu.setZero();
        var->markMuCurrent();
    };

    if (use_parallel) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
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

void relinearizeSyntheticResidualGraph(
    SyntheticResidualGraphWorkspace& workspace,
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    int num_threads,
    SyntheticResidualRelinearizeStats* stats = nullptr,
    const RobustLossConfig& robust_loss_config = {}
) {
    const auto total_t0 = SteadyClock::now();
    const bool use_parallel = (num_threads != 1) && (problem.edges.size() > 64);

    const auto factor_t0 = SteadyClock::now();
    if (use_parallel) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J =
                analyticEdgeResidualJacobian(base_i, base_j, edge.measurement, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;
            factor->setLinearFactorInfo(eta, lam);
            factor->clearMessages();
        }
    } else {
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J =
                analyticEdgeResidualJacobian(base_i, base_j, edge.measurement, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;
            factor->setLinearFactorInfo(eta, lam);
            factor->clearMessages();
        }
    }

    {
        const Eigen::Vector3d& base_anchor = base_poses.at(0);
        const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(problem.anchor_pose), base_anchor));
        const Eigen::Matrix3d J =
            analyticAnchorResidualJacobian(base_anchor, problem.anchor_pose, Eigen::Vector3d::Zero());
        const Eigen::Matrix3d lam = J.transpose() * problem.anchor_information * J;
        const Eigen::Vector3d eta = -J.transpose() * problem.anchor_information * err0;
        workspace.anchor_factor->setLinearFactorInfo(eta, lam);
        workspace.anchor_factor->clearMessages();
    }
    const auto factor_t1 = SteadyClock::now();

    const auto reset_t0 = SteadyClock::now();
    resetSyntheticResidualState(workspace, num_threads);
    const auto reset_t1 = SteadyClock::now();

    if (stats) {
        stats->factor_relinearize_sec = elapsedSeconds(factor_t0, factor_t1);
        stats->reset_state_sec = elapsedSeconds(reset_t0, reset_t1);
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
}

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    double tiny_prior,
    const RobustLossConfig& robust_loss_config
) {
    SyntheticResidualGraphWorkspace workspace = buildSyntheticResidualGraphWorkspace(problem, tiny_prior);
    relinearizeSyntheticResidualGraph(workspace, problem, base_poses, 1, nullptr, robust_loss_config);
    return std::move(workspace.graph);
}

ExperimentResults runSyntheticSE2Experiment(
    const SyntheticSE2Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    int sync_num_threads,
    gbp::FactorGraph::SyncScheduleKind sync_schedule_kind,
    int sync_schedule_chunk,
    bool profile_sync_thread_utilization,
    bool enable_singlecore_fastsync,
    double coarse_scale,
    const BasisBuildConfig& basis_build_config,
    const RobustLossConfig& robust_loss_config,
    int final_coarse_polish_passes
) {
    ExperimentResults results;
    const int upper_polish_passes = std::max(0, final_coarse_polish_passes);
    results.num_poses = static_cast<int>(problem.gt_poses.size());
    results.num_edges = static_cast<int>(problem.edges.size());
    results.initial_objective = nonlinearObjective(problem, problem.init_poses, 1);
    results.initial_poses = problem.init_poses;

    SyntheticResidualWorkspaceBuildStats mg_build_stats;
    SyntheticResidualGraphWorkspace mg_workspace =
        buildSyntheticResidualGraphWorkspace(problem, 1e-12, &mg_build_stats);
    mg_workspace.graph.setSyncNumThreads(sync_num_threads);
    mg_workspace.graph.setSyncUpdateMeans(false);
    mg_workspace.graph.setSyncSchedule(sync_schedule_kind, sync_schedule_chunk);
    mg_workspace.graph.setProfileSyncThreadUtilization(profile_sync_thread_utilization);
    results.residual_workspace_build_sec = mg_build_stats.total_sec;
    results.residual_workspace_variable_alloc_sec = mg_build_stats.variable_alloc_sec;
    results.residual_workspace_factor_alloc_sec = mg_build_stats.factor_alloc_sec;
    results.residual_workspace_connect_sec = mg_build_stats.connect_sec;
    SyntheticSE2FastSyncLocalPackedLayout fastsync_local_packed_layout =
        buildSyntheticSE2FastSyncLocalPackedLayout(mg_workspace.graph);
    SyntheticSE2PackedBasisTopology fastsync_local_packed_basis_topology;
    SyntheticSE2PackedCoarseAssemblyWorkspace fastsync_local_packed_coarse_workspace;
    bool fastsync_local_packed_coarse_workspace_valid = false;
    SparseCholeskyFactor fastsync_local_packed_coarse_factor;
    bool fastsync_local_packed_coarse_factor_valid = false;
    const bool fastsync_local_packed_basis_topology_valid = fastsync_local_packed_layout.supported;
    if (fastsync_local_packed_basis_topology_valid) {
        fastsync_local_packed_basis_topology =
            buildMessageConditionedBasisPackedTopology(fastsync_local_packed_layout, group_size);
    }
    SyntheticSE2FastSyncLocalLayout fastsync_local_layout =
        buildSyntheticSE2FastSyncLocalLayout(mg_workspace.graph);
    SyntheticSE2FastSyncSoALayout fastsync_layout =
        buildSyntheticSE2FastSyncSoALayout(mg_workspace.graph);
    SyntheticSE2PackedResidualWorkspace packed_residual_workspace;
    SyntheticSE2PackedBasisTopology packed_residual_basis_topology;
    SyntheticSE2PackedCoarseAssemblyWorkspace packed_residual_coarse_workspace;
    bool packed_residual_coarse_workspace_valid = false;
    SparseCholeskyFactor packed_residual_coarse_factor;
    bool packed_residual_coarse_factor_valid = false;
    const bool packed_residual_topology_valid = basis_build_config.enable_packed_residual_solver;
    if (packed_residual_topology_valid) {
        packed_residual_workspace = buildSyntheticSE2PackedResidualWorkspace(problem, 1e-12);
        packed_residual_basis_topology =
            buildMessageConditionedBasisPackedTopology(packed_residual_workspace, group_size);
        results.packed_schur_adjugate_fallback_count = packed_residual_workspace.schur_adjugate_fallback_count;
        results.packed_schur_general_failure_count = packed_residual_workspace.schur_general_failure_count;
        results.packed_kernel_prep_cycles = packed_residual_workspace.kernel_prep_cycles;
        results.packed_kernel_schur0_cycles = packed_residual_workspace.kernel_schur0_cycles;
        results.packed_kernel_schur1_cycles = packed_residual_workspace.kernel_schur1_cycles;
        results.packed_kernel_update_belief_cycles = packed_residual_workspace.kernel_update_belief_cycles;
        results.packed_fixedeta_full_lambda_sweeps = packed_residual_workspace.fixedeta_full_lambda_sweeps;
        results.packed_fixedeta_lambda_init_sweeps = packed_residual_workspace.fixedeta_lambda_init_sweeps;
        results.packed_fixedeta_eta_only_sweeps = packed_residual_workspace.fixedeta_eta_only_sweeps;
        results.packed_fixedeta_eta_map_builds = packed_residual_workspace.fixedeta_eta_map_builds;
        results.packed_fixedeta_serial_delta_sweeps = packed_residual_workspace.fixedeta_serial_delta_sweeps;
    }
    relinearizeSyntheticResidualGraph(mg_workspace, problem, problem.init_poses, sync_num_threads, nullptr, robust_loss_config);
    const bool force_unsafe_fastsync = getenvEnabled("GBP_FORCE_UNSAFE_FASTSYNC");
    const bool force_packed_residual_solver = getenvEnabled("GBP_FORCE_PACKED_RESIDUAL");
    const bool force_fastsync_local_packed = getenvEnabled("GBP_FORCE_FASTSYNC_LOCAL_PACKED");
    const bool force_fastsync_local = getenvEnabled("GBP_FORCE_FASTSYNC_LOCAL");
    const bool prefer_nonsym_safe_fastsync_global =
        !force_unsafe_fastsync && maxFactorLamAsymmetry(mg_workspace.graph) > 1e-7;

    PoseVec direct_poses = problem.init_poses;
    results.direct_history.push_back(OuterDirectRow{0, nonlinearObjective(problem, direct_poses, 1), 0.0, 0.0});
    results.direct_pose_history.push_back(direct_poses);

    PoseVec mg_poses = problem.init_poses;
    std::vector<Eigen::MatrixXd> previous_basis_local_bases;
    BasisData cached_basis;
    bool cached_basis_valid = false;
    {
        OuterMGRow row0;
        row0.outer = 0;
        row0.execution_path = "initial";
        row0.nonlinear_objective = nonlinearObjective(problem, mg_poses, 1);
        results.mg_history.push_back(std::move(row0));
    }
    results.mg_pose_history.push_back(mg_poses);
    for (int outer = 1; outer <= num_outer; ++outer) {
        const auto outer_t0 = SteadyClock::now();

        gbp::FactorGraph& residual_graph = mg_workspace.graph;
        const bool use_packed_residual_solver =
            (force_packed_residual_solver ||
             (basis_build_config.enable_packed_residual_solver &&
              enable_singlecore_fastsync &&
              residual_graph.eta_damping == 0.0 &&
              packed_residual_topology_valid &&
              !prefer_nonsym_safe_fastsync_global &&
              !profile_sync_thread_utilization));

        const auto build_t0 = SteadyClock::now();
        SyntheticResidualRelinearizeStats relin_stats;
        if (!use_packed_residual_solver) {
            relinearizeSyntheticResidualGraph(mg_workspace, problem, mg_poses, sync_num_threads, &relin_stats, robust_loss_config);
        }
        if (use_packed_residual_solver) {
            const auto packed_relin_t0 = SteadyClock::now();
            relinearizeSyntheticSE2PackedResidualWorkspace(
                packed_residual_workspace,
                problem,
                mg_poses,
                robust_loss_config,
                sync_num_threads
            );
            const auto packed_relin_t1 = SteadyClock::now();
            relin_stats.factor_relinearize_sec = elapsedSeconds(packed_relin_t0, packed_relin_t1);
            relin_stats.reset_state_sec = 0.0;
        }
        const double factor_lam_max_asym = use_packed_residual_solver ? 0.0 : maxFactorLamAsymmetry(residual_graph);
        const bool prefer_nonsym_safe_fastsync =
            prefer_nonsym_safe_fastsync_global ||
            (!use_packed_residual_solver && factor_lam_max_asym > 1e-7);
        const bool use_fastsync_local_packed =
            !use_packed_residual_solver &&
            (force_fastsync_local_packed ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              residual_graph.eta_damping == 0.0 &&
              fastsync_local_packed_layout.supported &&
              !prefer_nonsym_safe_fastsync &&
              !profile_sync_thread_utilization));
        const bool use_fastsync_local =
            !use_packed_residual_solver &&
            !use_fastsync_local_packed &&
            (force_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              residual_graph.eta_damping == 0.0 &&
              fastsync_local_layout.supported &&
              !prefer_nonsym_safe_fastsync &&
              !profile_sync_thread_utilization));
        const bool use_packed_non_sweeps =
            !use_packed_residual_solver &&
            use_fastsync_local_packed &&
            basis_build_config.enable_packed_builder;
        if (use_packed_residual_solver) {
            // Packed residual mode keeps the linearized state in the packed workspace.
        } else if (use_fastsync_local_packed) {
            refreshSyntheticSE2FastSyncLocalPackedLayout(fastsync_local_packed_layout);
        } else if (use_fastsync_local) {
            refreshSyntheticSE2FastSyncLocalLayout(fastsync_local_layout);
        } else if (enable_singlecore_fastsync && sync_num_threads == 1 && fastsync_layout.all_dofs3) {
            refreshSyntheticSE2FastSyncSoALayout(fastsync_layout);
        }
        const bool use_singlecore_fastsync_soa =
            !use_packed_residual_solver &&
            !use_fastsync_local_packed &&
            !use_fastsync_local &&
            enable_singlecore_fastsync &&
            sync_num_threads == 1 &&
            fastsync_layout.all_dofs3 &&
            !profile_sync_thread_utilization;
        const std::string execution_path = outerExecutionPathName(
            use_packed_residual_solver,
            use_fastsync_local_packed,
            use_fastsync_local,
            use_singlecore_fastsync_soa
        );
        residual_graph.setProfileSyncTiming(false);
        residual_graph.resetSyncTiming();
        const auto build_t1 = SteadyClock::now();
        const int packed_basis_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_basis_thread_override > 0)
            ? basis_build_config.packed_basis_thread_override
            : sync_num_threads;
        const int packed_coarse_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_coarse_thread_override > 0)
            ? basis_build_config.packed_coarse_thread_override
            : packed_basis_threads;
        const int packed_sweep_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_sweep_thread_override > 0)
            ? basis_build_config.packed_sweep_thread_override
            : sync_num_threads;

        const bool basis_within_warmup =
            basis_build_config.basis_rebuild_warmup_outers > 0 &&
            outer <= basis_build_config.basis_rebuild_warmup_outers;
        const int post_warmup_outer = outer - basis_build_config.basis_rebuild_warmup_outers;
        const bool rebuild_basis =
            !cached_basis_valid ||
            basis_build_config.basis_rebuild_period <= 1 ||
            basis_within_warmup ||
            (((post_warmup_outer - 1) % basis_build_config.basis_rebuild_period) == 0);
        const auto basis_t0 = SteadyClock::now();
        if (rebuild_basis) {
            const std::vector<Eigen::MatrixXd>* warm_basis_ptr =
                (basis_build_config.enable_warm_start && !previous_basis_local_bases.empty())
                ? &previous_basis_local_bases
                : nullptr;
            cached_basis = use_packed_residual_solver
                ? buildMessageConditionedBasisPacked(
                    packed_residual_workspace,
                    packed_residual_basis_topology,
                    r_reduced,
                    packed_basis_threads != 1,
                    packed_basis_threads,
                    basis_build_config,
                    warm_basis_ptr
                )
                : use_packed_non_sweeps
                ? buildMessageConditionedBasisPacked(
                    fastsync_local_packed_layout,
                    fastsync_local_packed_basis_topology,
                    r_reduced,
                    false,
                    1,
                    basis_build_config,
                    warm_basis_ptr
                )
                : buildMessageConditionedBasis(
                    residual_graph,
                    group_size,
                    r_reduced,
                    sync_num_threads != 1,
                    sync_num_threads,
                    basis_build_config,
                    warm_basis_ptr
                );
            cached_basis_valid = true;
        }
        const auto basis_t1 = SteadyClock::now();
        const BasisData& basis = cached_basis;
        if (basis_build_config.enable_warm_start) {
            if (rebuild_basis) {
                previous_basis_local_bases = basis.local_bases;
            }
        } else {
            previous_basis_local_bases.clear();
        }
        const double basis_build_sec = rebuild_basis ? elapsedSeconds(basis_t0, basis_t1) : 0.0;
        const double basis_block_build_sec = rebuild_basis ? basis.block_build_sec : 0.0;
        const double basis_eigensolver_sec = rebuild_basis ? basis.eigensolver_sec : 0.0;
        const double basis_copyout_sec = rebuild_basis ? basis.copyout_sec : 0.0;
        const int basis_partial_attempted = rebuild_basis ? basis.partial_attempted : 0;
        const int basis_partial_converged = rebuild_basis ? basis.partial_converged : 0;
        const int basis_partial_fallback = rebuild_basis ? basis.partial_fallback : 0;
        const int basis_partial_total_iters = rebuild_basis ? basis.partial_total_iters : 0;

        Eigen::SparseMatrix<double> coarse_lam;
        int coarse_lambda_nnz = 0;
        double coarse_lambda_density = 0.0;
        const auto coarse_lam_t0 = SteadyClock::now();
        if (use_packed_residual_solver) {
            if (!packed_residual_coarse_workspace_valid ||
                packed_residual_coarse_workspace.coarse_dim != basis.coarse_dim) {
                packed_residual_coarse_workspace =
                    buildSyntheticSE2PackedCoarseAssemblyWorkspace(
                        packed_residual_workspace,
                        basis
                    );
                packed_residual_coarse_workspace_valid = true;
                packed_residual_coarse_factor_valid = false;
            }
            coarse_lam = assembleCoarseLambdaDirectPackedOrdered3D(
                packed_residual_workspace,
                basis,
                packed_residual_coarse_workspace,
                packed_coarse_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        } else if (use_packed_non_sweeps) {
            if (!fastsync_local_packed_coarse_workspace_valid ||
                fastsync_local_packed_coarse_workspace.coarse_dim != basis.coarse_dim) {
                fastsync_local_packed_coarse_workspace =
                    buildSyntheticSE2PackedCoarseAssemblyWorkspace(
                        fastsync_local_packed_layout,
                        basis
                    );
                fastsync_local_packed_coarse_workspace_valid = true;
                fastsync_local_packed_coarse_factor_valid = false;
            }
            coarse_lam = assembleCoarseLambdaDirectPackedOrdered3D(
                fastsync_local_packed_layout,
                basis,
                fastsync_local_packed_coarse_workspace,
                sync_num_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        } else {
            coarse_lam = assembleCoarseLambdaDirect(
                residual_graph,
                basis,
                true,
                sync_num_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        }
        const auto coarse_lam_t1 = SteadyClock::now();
        if (basis.coarse_dim > 0) {
            const double denom = static_cast<double>(basis.coarse_dim) * static_cast<double>(basis.coarse_dim);
            coarse_lambda_density = coarse_lambda_nnz / denom;
        }

        double sweeps_sec = 0.0;
        double fastsync_factor_pass_sec = 0.0;
        double fastsync_variable_pass_sec = 0.0;
        double coarse_eta_sec = 0.0;
        double coarse_solve_sec = 0.0;
        double prolong_inject_sec = 0.0;
        const auto coarse_factor_t0 = SteadyClock::now();
        SparseCholeskyFactor generic_coarse_factor;
        SparseCholeskyFactor* coarse_factor = &generic_coarse_factor;
        bool reuse_coarse_pattern = false;
        if (use_packed_residual_solver) {
            coarse_factor = &packed_residual_coarse_factor;
            reuse_coarse_pattern = packed_residual_coarse_factor_valid;
        } else if (use_packed_non_sweeps) {
            coarse_factor = &fastsync_local_packed_coarse_factor;
            reuse_coarse_pattern = fastsync_local_packed_coarse_factor_valid;
        }
        factorizeSparseCholesky(
            coarse_lam,
            1e-10,
            *coarse_factor,
            reuse_coarse_pattern,
            use_packed_residual_solver || use_packed_non_sweeps
        );
        if (use_packed_residual_solver) {
            packed_residual_coarse_factor_valid = true;
        } else if (use_packed_non_sweeps) {
            fastsync_local_packed_coarse_factor_valid = true;
        }
        const auto coarse_factor_t1 = SteadyClock::now();
        coarse_solve_sec += elapsedSeconds(coarse_factor_t0, coarse_factor_t1);
        Eigen::VectorXd coarse_eta;
        Eigen::VectorXd delta_z;
        Eigen::VectorXd delta_e;
        Eigen::VectorXd e_now;
        Eigen::VectorXd e_hat;
        auto apply_upper_correction = [&]() {
            if (use_packed_residual_solver) {
                stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                    packed_residual_workspace,
                    e_now,
                    packed_sweep_threads
                );
            } else if (use_fastsync_local_packed) {
                e_now = stackedMeanVectorFastSyncLocalPacked(fastsync_local_packed_layout);
            } else if (use_fastsync_local) {
                e_now = stackedMeanVectorFastSyncLocal(fastsync_local_layout);
            } else {
                e_now = stackedMeanVector(residual_graph);
            }
            const auto coarse_eta_t0 = SteadyClock::now();
            if (use_packed_residual_solver) {
                assembleCoarseResidualDirectPackedOrdered3DInto(
                    packed_residual_workspace,
                    basis,
                    e_now,
                    coarse_eta,
                    packed_coarse_threads
                );
            } else if (use_packed_non_sweeps) {
                assembleCoarseResidualDirectPackedOrdered3DInto(
                    fastsync_local_packed_layout,
                    basis,
                    e_now,
                    coarse_eta,
                    sync_num_threads
                );
            } else {
                coarse_eta = assembleCoarseResidualDirect(residual_graph, basis, e_now, true, sync_num_threads);
            }
            const auto coarse_eta_t1 = SteadyClock::now();
            coarse_eta_sec += elapsedSeconds(coarse_eta_t0, coarse_eta_t1);

            const auto coarse_solve_t0 = SteadyClock::now();
            solveWithSparseCholeskyInto(*coarse_factor, coarse_eta, delta_z);
            const auto coarse_solve_t1 = SteadyClock::now();
            coarse_solve_sec += elapsedSeconds(coarse_solve_t0, coarse_solve_t1);

            const auto inject_t0 = SteadyClock::now();
            prolongToFineInto(basis, delta_z, delta_e);
            if (coarse_scale != 1.0) {
                delta_e *= coarse_scale;
            }
            if (use_packed_residual_solver) {
                injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(
                    packed_residual_workspace,
                    delta_e,
                    packed_sweep_threads
                );
            } else if (use_fastsync_local_packed) {
                injectCorrectionKeepMessagesFastSyncLocalPacked(fastsync_local_packed_layout, delta_e);
            } else if (use_fastsync_local) {
                injectCorrectionKeepMessagesFastSyncLocal(fastsync_local_layout, delta_e);
            } else {
                injectCorrectionKeepMessages(residual_graph, delta_e);
            }
            const auto inject_t1 = SteadyClock::now();
            prolong_inject_sec += elapsedSeconds(inject_t0, inject_t1);
        };

        for (int cyc = 0; cyc < inner_cycles; ++cyc) {
            const auto sweeps_t0 = SteadyClock::now();
            if (use_packed_residual_solver) {
                synchronousIterationsSyntheticSE2PackedResidualWorkspace(
                    packed_residual_workspace,
                    pre_sweeps,
                    fastsync_factor_pass_sec,
                    fastsync_variable_pass_sec,
                    packed_sweep_threads
                );
            } else {
                for (int sweep = 0; sweep < pre_sweeps; ++sweep) {
                    if (use_fastsync_local_packed) {
                    synchronousIterationFastSyncLocalPacked(
                        fastsync_local_packed_layout,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else if (use_fastsync_local) {
                        synchronousIterationFastSyncLocal(
                        fastsync_local_layout,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else if (enable_singlecore_fastsync &&
                        sync_num_threads == 1 &&
                        fastsync_layout.all_dofs3 &&
                        !profile_sync_thread_utilization) {
                        synchronousIterationFastSyncSoA(
                        fastsync_layout,
                        residual_graph.eta_damping,
                        false,
                        1,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else {
                        residual_graph.synchronousIteration();
                    }
                }
            }
            const auto sweeps_t1 = SteadyClock::now();
            sweeps_sec += elapsedSeconds(sweeps_t0, sweeps_t1);

            apply_upper_correction();
        }
        if (outer == num_outer) {
            for (int polish = 0; polish < upper_polish_passes; ++polish) {
                apply_upper_correction();
            }
        }

        if (use_packed_residual_solver) {
            stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                packed_residual_workspace,
                e_hat,
                packed_sweep_threads
            );
        } else if (use_fastsync_local_packed) {
            e_hat = stackedMeanVectorFastSyncLocalPacked(fastsync_local_packed_layout);
        } else if (use_fastsync_local) {
            e_hat = stackedMeanVectorFastSyncLocal(fastsync_local_layout);
        } else {
            e_hat = stackedMeanVector(residual_graph);
        }
        const double recorded_factor_pass_sec =
            (use_packed_residual_solver || use_fastsync_local_packed || use_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              fastsync_layout.all_dofs3 &&
              !profile_sync_thread_utilization))
            ? fastsync_factor_pass_sec
            : residual_graph.sync_factor_pass_sec_accum;
        const double recorded_variable_pass_sec =
            (use_packed_residual_solver || use_fastsync_local_packed || use_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              fastsync_layout.all_dofs3 &&
              !profile_sync_thread_utilization))
            ? fastsync_variable_pass_sec
            : residual_graph.sync_variable_pass_sec_accum;
        const auto apply_t0 = SteadyClock::now();
        mg_poses = applyPoseDeltas(mg_poses, e_hat);
        const auto apply_t1 = SteadyClock::now();
        const auto obj_t0 = SteadyClock::now();
        const double nonlinear_obj = nonlinearObjective(problem, mg_poses, sync_num_threads);
        const auto obj_t1 = SteadyClock::now();
        const auto outer_t1 = SteadyClock::now();
        results.mg_history.push_back(
            OuterMGRow{
                outer,
                execution_path,
                nonlinear_obj,
                e_hat.norm(),
                factor_lam_max_asym,
                prefer_nonsym_safe_fastsync,
                static_cast<int>(basis.groups.size()),
                basis.coarse_dim,
                coarse_lambda_nnz,
                coarse_lambda_density,
                false,
                elapsedSeconds(outer_t0, outer_t1),
                elapsedSeconds(build_t0, build_t1),
                relin_stats.factor_relinearize_sec,
                relin_stats.reset_state_sec,
                basis_build_sec,
                basis_block_build_sec,
                basis_eigensolver_sec,
                basis_copyout_sec,
                elapsedSeconds(coarse_lam_t0, coarse_lam_t1),
                sweeps_sec,
                recorded_factor_pass_sec,
                recorded_variable_pass_sec,
                residual_graph.sync_factor_work_sec_accum,
                residual_graph.sync_factor_wait_sec_accum,
                residual_graph.sync_variable_work_sec_accum,
                residual_graph.sync_variable_wait_sec_accum,
                residual_graph.sync_profiled_threads,
                coarse_eta_sec,
                coarse_solve_sec,
                prolong_inject_sec,
                elapsedSeconds(apply_t0, apply_t1),
                elapsedSeconds(obj_t0, obj_t1),
                basis_partial_attempted,
                basis_partial_converged,
                basis_partial_fallback,
                basis_partial_total_iters,
            }
        );
        results.mg_pose_history.push_back(mg_poses);
    }

    if (packed_residual_topology_valid) {
        results.packed_schur_adjugate_fallback_count = packed_residual_workspace.schur_adjugate_fallback_count;
        results.packed_schur_general_failure_count = packed_residual_workspace.schur_general_failure_count;
        results.packed_kernel_prep_cycles = packed_residual_workspace.kernel_prep_cycles;
        results.packed_kernel_schur0_cycles = packed_residual_workspace.kernel_schur0_cycles;
        results.packed_kernel_schur1_cycles = packed_residual_workspace.kernel_schur1_cycles;
        results.packed_kernel_update_belief_cycles = packed_residual_workspace.kernel_update_belief_cycles;
        results.packed_fixedeta_full_lambda_sweeps = packed_residual_workspace.fixedeta_full_lambda_sweeps;
        results.packed_fixedeta_lambda_init_sweeps = packed_residual_workspace.fixedeta_lambda_init_sweeps;
        results.packed_fixedeta_eta_only_sweeps = packed_residual_workspace.fixedeta_eta_only_sweeps;
        results.packed_fixedeta_eta_map_builds = packed_residual_workspace.fixedeta_eta_map_builds;
        results.packed_fixedeta_serial_delta_sweeps = packed_residual_workspace.fixedeta_serial_delta_sweeps;
    }

    return results;
}

void writeExperimentResultsJson(
    const ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    int sync_num_threads,
    gbp::FactorGraph::SyncScheduleKind sync_schedule_kind,
    int sync_schedule_chunk,
    bool profile_sync_thread_utilization,
    bool enable_singlecore_fastsync,
    const BasisBuildConfig& basis_build_config,
    const RobustLossConfig& robust_loss_config,
    int final_coarse_polish_passes,
    double packed_jitter,
    int fixed_eta_after_sweeps
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output JSON path: " + path);
    }

    out << "{\n";
    out << "  \"config\": {\n";
    out << "    \"num_outer\": " << num_outer << ",\n";
    out << "    \"inner_cycles\": " << inner_cycles << ",\n";
    out << "    \"pre_sweeps\": " << pre_sweeps << ",\n";
    out << "    \"group_size\": " << group_size << ",\n";
    out << "    \"r_reduced\": " << r_reduced << ",\n";
    out << "    \"sync_num_threads\": " << sync_num_threads << ",\n";
    out << "    \"sync_schedule\": \"" << syncScheduleName(sync_schedule_kind) << "\",\n";
    out << "    \"sync_schedule_chunk\": " << sync_schedule_chunk << ",\n";
    out << "    \"profile_sync_thread_utilization\": " << (profile_sync_thread_utilization ? "true" : "false") << ",\n";
    out << "    \"enable_singlecore_fastsync\": " << (enable_singlecore_fastsync ? "true" : "false") << ",\n";
    out << "    \"final_coarse_polish_passes\": " << std::max(0, final_coarse_polish_passes) << ",\n";
    out << "    \"basis_eigensolver\": \"" << basisEigensolverName(basis_build_config.eigensolver) << "\",\n";
    out << "    \"basis_warm_start\": " << (basis_build_config.enable_warm_start ? "true" : "false") << ",\n";
    out << "    \"basis_packed_builder\": " << (basis_build_config.enable_packed_builder ? "true" : "false") << ",\n";
    out << "    \"basis_packed_residual_solver\": " << (basis_build_config.enable_packed_residual_solver ? "true" : "false") << ",\n";
    out << "    \"packed_basis_thread_override\": " << basis_build_config.packed_basis_thread_override << ",\n";
    out << "    \"packed_coarse_thread_override\": " << basis_build_config.packed_coarse_thread_override << ",\n";
    out << "    \"packed_sweep_thread_override\": " << basis_build_config.packed_sweep_thread_override << ",\n";
    out << "    \"basis_rebuild_period\": " << basis_build_config.basis_rebuild_period << ",\n";
    out << "    \"basis_rebuild_warmup_outers\": " << basis_build_config.basis_rebuild_warmup_outers << ",\n";
    out << "    \"partial_oversampling\": " << basis_build_config.partial_oversampling << ",\n";
    out << "    \"partial_max_iters\": " << basis_build_config.partial_max_iters << ",\n";
    out << "    \"partial_residual_check_period\": " << basis_build_config.partial_residual_check_period << ",\n";
    out << "    \"partial_residual_tol\": " << jsonNumber(basis_build_config.partial_residual_tol) << ",\n";
    out << "    \"partial_ridge\": " << jsonNumber(basis_build_config.partial_ridge) << ",\n";
    out << "    \"robust_huber_delta\": " << jsonNumber(robust_loss_config.huber_delta) << ",\n";
    out << "    \"packed_jitter\": " << jsonNumber(packed_jitter) << ",\n";
    out << "    \"fixed_eta_after_sweeps\": " << fixed_eta_after_sweeps << ",\n";
    out << "    \"basis_source\": \"message_conditioned_information\"\n";
    out << "  },\n";
    out << "  \"problem\": {\n";
    out << "    \"num_poses\": " << results.num_poses << ",\n";
    out << "    \"num_edges\": " << results.num_edges << "\n";
    out << "  },\n";
    out << "  \"initial_objective\": " << jsonNumber(results.initial_objective) << ",\n";
    out << "  \"residual_workspace_build_sec\": " << jsonNumber(results.residual_workspace_build_sec) << ",\n";
    out << "  \"residual_workspace_variable_alloc_sec\": " << jsonNumber(results.residual_workspace_variable_alloc_sec) << ",\n";
    out << "  \"residual_workspace_factor_alloc_sec\": " << jsonNumber(results.residual_workspace_factor_alloc_sec) << ",\n";
    out << "  \"residual_workspace_connect_sec\": " << jsonNumber(results.residual_workspace_connect_sec) << ",\n";
    out << "  \"packed_schur_adjugate_fallback_count\": " << results.packed_schur_adjugate_fallback_count << ",\n";
    out << "  \"packed_schur_general_failure_count\": " << results.packed_schur_general_failure_count << ",\n";
    out << "  \"packed_kernel_prep_cycles\": " << results.packed_kernel_prep_cycles << ",\n";
    out << "  \"packed_kernel_schur0_cycles\": " << results.packed_kernel_schur0_cycles << ",\n";
    out << "  \"packed_kernel_schur1_cycles\": " << results.packed_kernel_schur1_cycles << ",\n";
    out << "  \"packed_kernel_update_belief_cycles\": " << results.packed_kernel_update_belief_cycles << ",\n";
    out << "  \"packed_fixedeta_full_lambda_sweeps\": " << results.packed_fixedeta_full_lambda_sweeps << ",\n";
    out << "  \"packed_fixedeta_lambda_init_sweeps\": " << results.packed_fixedeta_lambda_init_sweeps << ",\n";
    out << "  \"packed_fixedeta_eta_only_sweeps\": " << results.packed_fixedeta_eta_only_sweeps << ",\n";
    out << "  \"packed_fixedeta_eta_map_builds\": " << results.packed_fixedeta_eta_map_builds << ",\n";
    out << "  \"packed_fixedeta_serial_delta_sweeps\": " << results.packed_fixedeta_serial_delta_sweeps << ",\n";

    out << "  \"direct_history\": [\n";
    for (size_t i = 0; i < results.direct_history.size(); ++i) {
        const OuterDirectRow& row = results.direct_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"linear_step_norm\": " << jsonNumber(row.linear_step_norm)
            << ", \"linear_residual_norm\": " << jsonNumber(row.linear_residual_norm)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"factor_relinearize_sec\": " << jsonNumber(row.factor_relinearize_sec)
            << ", \"reset_state_sec\": " << jsonNumber(row.reset_state_sec)
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
        const OuterMGRow& row = results.mg_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"execution_path\": \"" << row.execution_path << "\""
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"e_hat_norm\": " << jsonNumber(row.e_hat_norm)
            << ", \"factor_lam_max_asym\": " << jsonNumber(row.factor_lam_max_asym)
            << ", \"prefer_nonsym_safe_fastsync\": " << (row.prefer_nonsym_safe_fastsync ? "true" : "false")
            << ", \"num_groups\": " << row.num_groups
            << ", \"coarse_dim\": " << row.coarse_dim
            << ", \"coarse_lambda_nnz\": " << row.coarse_lambda_nnz
            << ", \"coarse_lambda_density\": " << jsonNumber(row.coarse_lambda_density)
            << ", \"coarse_lambda_dense_factor\": " << (row.coarse_lambda_dense_factor ? "true" : "false")
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"factor_relinearize_sec\": " << jsonNumber(row.factor_relinearize_sec)
            << ", \"reset_state_sec\": " << jsonNumber(row.reset_state_sec)
            << ", \"basis_build_sec\": " << jsonNumber(row.basis_build_sec)
            << ", \"basis_block_build_sec\": " << jsonNumber(row.basis_block_build_sec)
            << ", \"basis_eigensolver_sec\": " << jsonNumber(row.basis_eigensolver_sec)
            << ", \"basis_copyout_sec\": " << jsonNumber(row.basis_copyout_sec)
            << ", \"coarse_lambda_sec\": " << jsonNumber(row.coarse_lambda_sec)
            << ", \"sweeps_sec\": " << jsonNumber(row.sweeps_sec)
            << ", \"factor_pass_sec\": " << jsonNumber(row.factor_pass_sec)
            << ", \"variable_pass_sec\": " << jsonNumber(row.variable_pass_sec)
            << ", \"factor_work_sec\": " << jsonNumber(row.factor_work_sec)
            << ", \"factor_wait_sec\": " << jsonNumber(row.factor_wait_sec)
            << ", \"variable_work_sec\": " << jsonNumber(row.variable_work_sec)
            << ", \"variable_wait_sec\": " << jsonNumber(row.variable_wait_sec)
            << ", \"sync_profiled_threads\": " << row.sync_profiled_threads
            << ", \"coarse_eta_sec\": " << jsonNumber(row.coarse_eta_sec)
            << ", \"coarse_solve_sec\": " << jsonNumber(row.coarse_solve_sec)
            << ", \"prolong_inject_sec\": " << jsonNumber(row.prolong_inject_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << ", \"basis_partial_attempted\": " << row.basis_partial_attempted
            << ", \"basis_partial_converged\": " << row.basis_partial_converged
            << ", \"basis_partial_fallback\": " << row.basis_partial_fallback
            << ", \"basis_partial_total_iters\": " << row.basis_partial_total_iters
            << "}";
        out << (i + 1 == results.mg_history.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void writeExperimentPoseHistoryJson(
    const ExperimentResults& results,
    const std::string& path
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open pose-history JSON path: " + path);
    }
    out << "{\n";
    out << "  \"num_poses\": " << results.num_poses << ",\n";
    out << "  \"num_edges\": " << results.num_edges << ",\n";
    out << "  \"initial_poses\": ";
    writePoseVectorJson(out, results.initial_poses, 2);
    out << ",\n";
    out << "  \"direct_pose_history\": ";
    writePoseHistoryJsonArray(out, results.direct_pose_history, 2);
    out << ",\n";
    out << "  \"mg_pose_history\": ";
    writePoseHistoryJsonArray(out, results.mg_pose_history, 2);
    out << "\n";
    out << "}\n";
}

}  // namespace slam
