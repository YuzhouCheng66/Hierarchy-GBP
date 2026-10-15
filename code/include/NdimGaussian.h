// NdimGaussian.h
#pragma once

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <cstddef>
#include <stdexcept>

namespace utils {

// Gaussian in information form (eta, lam).
// SOO fast-path:
//   - dim==2/3: stores eta/lam in fixed-size vectors/matrices (no heap alloc)
//   - dim!=2/3: stores eta/lam in VectorXd/MatrixXd
//
// Exposes a uniform API via Eigen::Map views (zero-copy).
// For hot paths, prefer the raw pointer accessors etaData()/lamData().
class NdimGaussian {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using Scalar = double;
    using Vector = Eigen::VectorXd;
    using Matrix = Eigen::MatrixXd;

    using VectorMapConst = Eigen::Map<const Eigen::VectorXd>;
    using MatrixMapConst = Eigen::Map<const Eigen::MatrixXd>;
    using VectorMap      = Eigen::Map<Eigen::VectorXd>;
    using MatrixMap      = Eigen::Map<Eigen::MatrixXd>;

    NdimGaussian();
    explicit NdimGaussian(int dimensionality);
    NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam);

    EIGEN_STRONG_INLINE int dim() const noexcept { return dim_; }

    // Zero-copy views for API-level access; hot paths use raw pointers.
    VectorMapConst eta() const;
    MatrixMapConst lam() const;

    // Writable views for API-level access.
    // NOTE: eta changes do NOT invalidate LLT cache (LLT depends only on lam)
    VectorMap etaRef();
    // lam changes DO invalidate LLT cache
    MatrixMap lamRef();

    // ==============================
    // Raw pointer access (hot path)
    // ==============================
    // Zero-cost accessors (no Map construction, no cache flags touched).
    // Return nullptr iff dim()==0.
    EIGEN_STRONG_INLINE const double* etaData() const noexcept {
        if (dim_ == 2) return eta2_.data();
        if (dim_ == 3) return eta3_.data();
        return eta_.data();
    }
    EIGEN_STRONG_INLINE double* etaData() noexcept {
        return const_cast<double*>(static_cast<const NdimGaussian*>(this)->etaData());
    }

    EIGEN_STRONG_INLINE const double* lamData() const noexcept {
        if (dim_ == 2) return lam2_.data();
        if (dim_ == 3) return lam3_.data();
        return lam_.data();
    }
    EIGEN_STRONG_INLINE double* lamData() noexcept {
        return const_cast<double*>(static_cast<const NdimGaussian*>(this)->lamData());
    }

    void setEta(const Vector& eta); // does NOT invalidate LLT cache
    void setLam(const Matrix& lam); // invalidates LLT cache

    void resizeLikeDim(int dimensionality); // invalidates LLT cache

    // Returns dense mu and Sigma (VectorXd/MatrixXd) for simplicity.
    Vector mu() const;
    void solveMuInto(double* out) const;
    void solveEtaInto(const double* eta, double* out) const;
    Matrix Sigma() const;

private:
    int dim_ = 0;

    // dim==2 storage
    Eigen::Vector2d eta2_ = Eigen::Vector2d::Zero();
    Eigen::Matrix2d lam2_ = Eigen::Matrix2d::Zero();

    // dim==3 storage
    Eigen::Vector3d eta3_ = Eigen::Vector3d::Zero();
    Eigen::Matrix3d lam3_ = Eigen::Matrix3d::Zero();

    // dim!=2 storage
    Vector eta_;
    Matrix lam_;

    // Factorization cache for solving (depends ONLY on lam)
    mutable bool cache_valid_ = false;
    mutable int  cached_dim_  = -1;

    mutable Eigen::LLT<Eigen::Matrix2d, Eigen::Upper> llt2_;
    mutable Eigen::LLT<Eigen::Matrix3d, Eigen::Upper> llt3_;
    mutable Eigen::LLT<Matrix, Eigen::Upper>          lltx_;

    void ensureFactorized() const;
};

} // namespace utils
