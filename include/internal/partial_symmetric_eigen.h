#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace slam {

struct PartialSymmetricEigenOptions {
    int oversampling = 4;
    int max_iters = 6;
    double residual_tol = 1e-5;
    double ridge = 1e-10;
    int residual_check_period = 1;
};

struct PartialSymmetricEigenWorkspace {
    Eigen::MatrixXd shifted;
    Eigen::MatrixXd Q;
    Eigen::MatrixXd Z;
    Eigen::MatrixXd T;
    Eigen::MatrixXd AQ;
    Eigen::MatrixXd thin_identity;
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> small_es;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr;
};

struct PartialSymmetricEigenResult {
    Eigen::MatrixXd eigenvectors;
    Eigen::VectorXd eigenvalues;
    bool converged = false;
    int iterations = 0;
    double max_relative_residual = std::numeric_limits<double>::infinity();
};

inline double meanAbsDiagonal(const Eigen::Ref<const Eigen::MatrixXd>& a) {
    const int n = static_cast<int>(a.rows());
    if (n <= 0) {
        return 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::abs(a(i, i));
    }
    return sum / static_cast<double>(n);
}

inline void orthonormalizeTallSkinny(
    Eigen::MatrixXd& mat,
    PartialSymmetricEigenWorkspace& ws
) {
    ws.qr.compute(mat);
    const int rows = mat.rows();
    const int cols = mat.cols();
    if (ws.thin_identity.rows() != rows || ws.thin_identity.cols() != cols) {
        ws.thin_identity.resize(rows, cols);
    }
    ws.thin_identity.setZero();
    ws.thin_identity.diagonal().setOnes();
    mat.noalias() = ws.qr.householderQ() * ws.thin_identity;
}

inline void seedInitialSubspace(
    Eigen::MatrixXd& q,
    const Eigen::MatrixXd* warm_start,
    int used_cols
) {
    q.setZero();
    int filled = 0;
    if (warm_start && warm_start->rows() == q.rows() && warm_start->cols() > 0) {
        const int copy_cols = std::min({
            static_cast<int>(warm_start->cols()),
            static_cast<int>(q.cols()),
            used_cols
        });
        if (copy_cols > 0) {
            q.leftCols(copy_cols) = warm_start->leftCols(copy_cols);
            filled = copy_cols;
        }
    }

    int basis_col = 0;
    while (filled < q.cols()) {
        q.col(filled).setZero();
        q(basis_col % q.rows(), filled) = 1.0;
        ++filled;
        ++basis_col;
    }
}

inline PartialSymmetricEigenResult computeSmallestEigenpairsPartial(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    int k,
    const Eigen::MatrixXd* warm_start,
    PartialSymmetricEigenWorkspace& ws,
    const PartialSymmetricEigenOptions& options = {}
) {
    PartialSymmetricEigenResult result;
    const int n = a.rows();
    if (n == 0 || k <= 0) {
        result.eigenvectors = Eigen::MatrixXd::Zero(n, 0);
        result.eigenvalues = Eigen::VectorXd::Zero(0);
        result.converged = true;
        return result;
    }
    if (k >= n) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(a);
        result.eigenvectors = es.eigenvectors();
        result.eigenvalues = es.eigenvalues();
        result.converged = (es.info() == Eigen::Success);
        result.iterations = 1;
        return result;
    }

    const int p = std::min(n, std::max(k, k + std::max(options.oversampling, 0)));
    ws.shifted = a;
    const double ridge_scale = std::max(1.0, meanAbsDiagonal(a));
    const double ridge = options.ridge * ridge_scale;
    for (int i = 0; i < n; ++i) {
        ws.shifted(i, i) += ridge;
    }
    ws.ldlt.compute(ws.shifted);
    if (ws.ldlt.info() != Eigen::Success) {
        return result;
    }

    ws.Q.resize(n, p);
    seedInitialSubspace(ws.Q, warm_start, k);
    orthonormalizeTallSkinny(ws.Q, ws);

    result.eigenvectors.resize(n, k);
    result.eigenvalues.resize(k);

    const int residual_check_period = std::max(1, options.residual_check_period);
    for (int iter = 0; iter < options.max_iters; ++iter) {
        ws.Z = ws.ldlt.solve(ws.Q);
        if (ws.ldlt.info() != Eigen::Success) {
            return result;
        }
        orthonormalizeTallSkinny(ws.Z, ws);

        ws.T.noalias() = ws.Z.transpose() * a * ws.Z;
        ws.small_es.compute(ws.T);
        if (ws.small_es.info() != Eigen::Success) {
            return result;
        }

        ws.Q.noalias() = ws.Z * ws.small_es.eigenvectors();
        result.eigenvalues = ws.small_es.eigenvalues().head(k);
        result.eigenvectors = ws.Q.leftCols(k);
        result.iterations = iter + 1;

        const bool check_residual =
            ((iter + 1) == options.max_iters) ||
            (((iter + 1) % residual_check_period) == 0);
        if (!check_residual) {
            continue;
        }

        ws.AQ.noalias() = a * result.eigenvectors;
        double max_rel_resid = 0.0;
        for (int col = 0; col < k; ++col) {
            const double eval = result.eigenvalues[col];
            double resid_sq = 0.0;
            for (int row = 0; row < n; ++row) {
                const double diff = ws.AQ(row, col) - eval * result.eigenvectors(row, col);
                resid_sq += diff * diff;
            }
            const double resid = std::sqrt(resid_sq) / std::max(1.0, std::abs(eval));
            max_rel_resid = std::max(max_rel_resid, resid);
        }
        result.max_relative_residual = max_rel_resid;
        if (max_rel_resid <= options.residual_tol) {
            result.converged = true;
            break;
        }
    }

    return result;
}

}  // namespace slam
