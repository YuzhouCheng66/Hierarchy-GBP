#pragma once
#include "internal/certified_coarse.h"
#include <Eigen/CholmodSupport>
#include <Eigen/SparseCholesky>
#include <memory>
#include <stdexcept>

namespace slam {
using CoarseSparse = Eigen::SparseMatrix<double>;

class CholmodCoarseFactor : public Eigen::CholmodDecomposition<CoarseSparse> {
public:
    CholmodCoarseFactor() {
        cholmod().nmethods = 1;
        cholmod().method[0].ordering = CHOLMOD_AMD;
        setMode(Eigen::CholmodAuto);
    }
    bool available() const { return m_cholmodFactor != nullptr; }
    bool supernodal() const { return available() && m_cholmodFactor->is_super; }
    CoarseWorkEstimate workEstimate(Eigen::Index matrix_nonzeros) const {
        if (!available()) throw std::runtime_error("Missing coarse symbolic factor");
        CoarseWorkEstimate result;
        const auto* counts = static_cast<const int*>(m_cholmodFactor->ColCount);
        double nonzeros = 0;
        for (size_t j = 0; j < m_cholmodFactor->n; ++j) {
            const double count = counts[j];
            result.factor_work += count * count;
            nonzeros += count;
        }
        // Eigen's simplicial matrixL() excludes the diagonal; CHOLMOD ColCount
        // includes it. Keep the same structural-work convention across backends.
        const double n = static_cast<double>(m_cholmodFactor->n);
        result.solve_work = 4. * (nonzeros - n) + n;
        result.matvec_work = 2. * matrix_nonzeros + 12. * n;
        return result;
    }
};

// This backend is only used for the projected coarse system, never fine H.
class CoarseCholesky {
public:
    explicit CoarseCholesky(bool use_cholmod = false) {
        if (use_cholmod) cholmod_ = std::make_unique<CholmodCoarseFactor>();
    }
    void analyzePattern(const CoarseSparse& a) {
        if (a.rows() <= 0 || a.rows() != a.cols() || !a.isCompressed())
            throw std::runtime_error("Invalid coarse sparse matrix");
        if (cholmod_) {
            cholmod_->analyzePattern(a);
            info_ = cholmod_->available() ? cholmod_->info() : Eigen::NumericalIssue;
        } else {
            eigen_.analyzePattern(a);
            info_ = eigen_.info();
        }
        rows_ = a.rows();
    }
    void factorize(const CoarseSparse& a) {
        if (info_ != Eigen::Success || a.rows() != rows_)
            throw std::runtime_error("Missing or incompatible coarse analysis");
        if (cholmod_) {
            cholmod_->factorize(a);
            info_ = cholmod_->info();
        } else {
            eigen_.factorize(a);
            info_ = eigen_.info();
        }
    }
    void compute(const CoarseSparse& a) {
        analyzePattern(a);
        if (info_ == Eigen::Success) factorize(a);
    }
    void solveInto(const Eigen::VectorXd& rhs, Eigen::VectorXd& x) const {
        if (info_ != Eigen::Success || rhs.size() != rows_)
            throw std::runtime_error("Invalid coarse solve");
        if (cholmod_) x = cholmod_->solve(rhs);
        else x = eigen_.solve(rhs);
    }
    Eigen::ComputationInfo info() const { return info_; }
    Eigen::Index rows() const { return rows_; }
    bool supernodal() const { return cholmod_ && cholmod_->supernodal(); }
    CoarseWorkEstimate workEstimate(Eigen::Index matrix_nonzeros) const {
        if (cholmod_) return cholmod_->workEstimate(matrix_nonzeros);
        return estimateCoarseWork(eigen_.matrixL().nestedExpression(), matrix_nonzeros);
    }
private:
    Eigen::SimplicialLDLT<CoarseSparse> eigen_;
    std::unique_ptr<CholmodCoarseFactor> cholmod_;
    Eigen::ComputationInfo info_ = Eigen::InvalidInput;
    Eigen::Index rows_ = 0;
};
} // namespace slam
