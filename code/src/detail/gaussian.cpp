// NdimGaussian.cpp
#include "NdimGaussian.h"

#include <cassert>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

namespace utils {

namespace {

extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info, std::size_t uplo_len);
void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info, std::size_t uplo_len);
}

using DpotrfFn = void (*)(const char* uplo, const int* n, double* a, const int* lda, int* info, std::size_t uplo_len);
using DpotrsFn = void (*)(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info, std::size_t uplo_len);

struct LapackOverride {
    DpotrfFn dpotrf = nullptr;
    DpotrsFn dpotrs = nullptr;
    bool enabled = false;
};

const LapackOverride& lapackOverride() {
    static const LapackOverride override_state = []() {
        LapackOverride state;
#if defined(_WIN32)
        const char* dll_path = std::getenv("GBP_LAPACK_DLL_OVERRIDE");
        if (dll_path == nullptr || dll_path[0] == '\0') {
            return state;
        }
        HMODULE module = LoadLibraryA(dll_path);
        if (module == nullptr) {
            return state;
        }
        auto dpotrf = reinterpret_cast<DpotrfFn>(GetProcAddress(module, "dpotrf_"));
        auto dpotrs = reinterpret_cast<DpotrsFn>(GetProcAddress(module, "dpotrs_"));
        if (dpotrf == nullptr || dpotrs == nullptr) {
            return state;
        }
        state.dpotrf = dpotrf;
        state.dpotrs = dpotrs;
        state.enabled = true;
#endif
        return state;
    }();
    return override_state;
}

bool lapackSolveUpperSpd3x3Vec(const double* a_upper, const double* rhs, double* out) {
    double a[9] = {
        a_upper[0], a_upper[1], a_upper[2],
        a_upper[3], a_upper[4], a_upper[5],
        a_upper[6], a_upper[7], a_upper[8]
    };
    a[1] = a[3];
    a[2] = a[6];
    a[5] = a[7];

    std::memcpy(out, rhs, 3 * sizeof(double));
    const char uplo = 'U';
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
    const int nrhs = 1;
    int info = 0;
    const LapackOverride& override_state = lapackOverride();
    if (override_state.enabled) {
        override_state.dpotrf(&uplo, &n, a, &lda, &info, 1);
    } else {
        dpotrf_(&uplo, &n, a, &lda, &info, 1);
    }
    if (info != 0) {
        return false;
    }
    if (override_state.enabled) {
        override_state.dpotrs(&uplo, &n, &nrhs, a, &lda, out, &ldb, &info, 1);
    } else {
        dpotrs_(&uplo, &n, &nrhs, a, &lda, out, &ldb, &info, 1);
    }
    return info == 0;
}

}  // namespace

NdimGaussian::NdimGaussian() = default;

NdimGaussian::NdimGaussian(int dimensionality) {
    resizeLikeDim(dimensionality);
}

NdimGaussian::NdimGaussian(int dimensionality, const Vector& eta, const Matrix& lam) {
    resizeLikeDim(dimensionality);
    setEta(eta);
    setLam(lam);
}

NdimGaussian::VectorMapConst NdimGaussian::eta() const {
    if (dim_ == 2) return VectorMapConst(eta2_.data(), 2);
    if (dim_ == 3) return VectorMapConst(eta3_.data(), 3);
    return VectorMapConst(eta_.data(), dim_);
}

NdimGaussian::MatrixMapConst NdimGaussian::lam() const {
    if (dim_ == 2) return MatrixMapConst(lam2_.data(), 2, 2);
    if (dim_ == 3) return MatrixMapConst(lam3_.data(), 3, 3);
    return MatrixMapConst(lam_.data(), dim_, dim_);
}

NdimGaussian::VectorMap NdimGaussian::etaRef() {
    // LLT cache depends only on lam, so DO NOT invalidate here.
    if (dim_ == 2) return VectorMap(eta2_.data(), 2);
    if (dim_ == 3) return VectorMap(eta3_.data(), 3);
    return VectorMap(eta_.data(), dim_);
}

NdimGaussian::MatrixMap NdimGaussian::lamRef() {
    // Any change to lam invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;

    if (dim_ == 2) return MatrixMap(lam2_.data(), 2, 2);
    if (dim_ == 3) return MatrixMap(lam3_.data(), 3, 3);
    return MatrixMap(lam_.data(), dim_, dim_);
}

void NdimGaussian::setEta(const Vector& eta) {
    if (dim_ == 0) resizeLikeDim((int)eta.size());
    assert((int)eta.size() == dim_);

    if (dim_ == 2) {
        eta2_ = eta.head<2>();
    } else if (dim_ == 3) {
        eta3_ = eta.head<3>();
    } else {
        eta_ = eta;
    }
    // DO NOT invalidate LLT cache (depends only on lam).
}

void NdimGaussian::setLam(const Matrix& lam) {
    if (dim_ == 0) resizeLikeDim((int)lam.rows());
    assert((int)lam.rows() == dim_ && (int)lam.cols() == dim_);

    if (dim_ == 2) {
        lam2_ = lam.topLeftCorner<2, 2>();
    } else if (dim_ == 3) {
        lam3_ = lam.topLeftCorner<3, 3>();
    } else {
        lam_ = lam;
    }

    // Changing lam invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;
}

void NdimGaussian::resizeLikeDim(int dimensionality) {
    if (dimensionality < 0) {
        throw std::runtime_error("NdimGaussian::resizeLikeDim: negative dim");
    }
    if (dim_ == dimensionality) return;

    dim_ = dimensionality;

    if (dim_ == 2) {
        // Fixed-size storage only; keep dynamic buffers empty to avoid heap.
        eta2_.setZero();
        lam2_.setZero();
        eta3_.setZero();
        lam3_.setZero();
        eta_.resize(0);
        lam_.resize(0, 0);
    } else if (dim_ == 3) {
        eta2_.setZero();
        lam2_.setZero();
        eta3_.setZero();
        lam3_.setZero();
        eta_.resize(0);
        lam_.resize(0, 0);
    } else {
        eta2_.setZero();
        lam2_.setZero();
        eta3_.setZero();
        lam3_.setZero();
        eta_.setZero(dim_);
        lam_.setZero(dim_, dim_);
    }

    // Dim change invalidates LLT cache.
    cache_valid_ = false;
    cached_dim_  = -1;
}

NdimGaussian::Vector NdimGaussian::mu() const {
    if (dim_ == 2) {
        Eigen::Vector2d mu2;
        try {
            ensureFactorized();
            mu2 = llt2_.solve(eta2_);
        } catch (const std::runtime_error&) {
            mu2 = lam2_.fullPivLu().solve(eta2_);
        }
        Vector out(2);
        out = mu2;
        return out;
    }
    if (dim_ == 3) {
        if (lapackOverride().enabled) {
            Vector out(3);
            if (lapackSolveUpperSpd3x3Vec(lam3_.data(), eta3_.data(), out.data())) {
                return out;
            }
        }
        Eigen::Vector3d mu3;
        try {
            ensureFactorized();
            mu3 = llt3_.solve(eta3_);
        } catch (const std::runtime_error&) {
            mu3 = lam3_.fullPivLu().solve(eta3_);
        }
        Vector out(3);
        out = mu3;
        return out;
    }
    try {
        ensureFactorized();
        return lltx_.solve(eta_);
    } catch (const std::runtime_error&) {
        return lam_.fullPivLu().solve(eta_);
    }
}

void NdimGaussian::solveMuInto(double* out) const {
    if (dim_ == 0) return;
    if (out == nullptr) {
        throw std::runtime_error("NdimGaussian::solveMuInto: null output");
    }

    if (dim_ == 2) {
        Eigen::Map<Eigen::Vector2d> out2(out);
        try {
            ensureFactorized();
            out2.noalias() = llt2_.solve(eta2_);
        } catch (const std::runtime_error&) {
            out2.noalias() = lam2_.fullPivLu().solve(eta2_);
        }
        return;
    }

    if (dim_ == 3) {
        if (lapackOverride().enabled && lapackSolveUpperSpd3x3Vec(lam3_.data(), eta3_.data(), out)) {
            return;
        }
        Eigen::Map<Eigen::Vector3d> out3(out);
        try {
            ensureFactorized();
            out3.noalias() = llt3_.solve(eta3_);
        } catch (const std::runtime_error&) {
            out3.noalias() = lam3_.fullPivLu().solve(eta3_);
        }
        return;
    }

    Eigen::Map<Vector> outx(out, dim_);
    try {
        ensureFactorized();
        outx.noalias() = lltx_.solve(eta_);
    } catch (const std::runtime_error&) {
        outx.noalias() = lam_.fullPivLu().solve(eta_);
    }
}

void NdimGaussian::solveEtaInto(const double* rhs, double* out) const {
    if (dim_ == 0) return;
    if (rhs == nullptr || out == nullptr) {
        throw std::runtime_error("NdimGaussian::solveEtaInto: null pointer");
    }

    if (dim_ == 2) {
        const Eigen::Map<const Eigen::Vector2d> rhs2(rhs);
        Eigen::Map<Eigen::Vector2d> out2(out);
        try {
            ensureFactorized();
            out2.noalias() = llt2_.solve(rhs2);
        } catch (const std::runtime_error&) {
            out2.noalias() = lam2_.fullPivLu().solve(rhs2);
        }
        return;
    }

    if (dim_ == 3) {
        const Eigen::Map<const Eigen::Vector3d> rhs3(rhs);
        Eigen::Map<Eigen::Vector3d> out3(out);
        try {
            ensureFactorized();
            out3.noalias() = llt3_.solve(rhs3);
        } catch (const std::runtime_error&) {
            out3.noalias() = lam3_.fullPivLu().solve(rhs3);
        }
        return;
    }

    const Eigen::Map<const Vector> rhsx(rhs, dim_);
    Eigen::Map<Vector> outx(out, dim_);
    try {
        ensureFactorized();
        outx.noalias() = lltx_.solve(rhsx);
    } catch (const std::runtime_error&) {
        outx.noalias() = lam_.fullPivLu().solve(rhsx);
    }
}

NdimGaussian::Matrix NdimGaussian::Sigma() const {
    ensureFactorized();

    if (dim_ == 2) {
        const Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
        const Eigen::Matrix2d S = llt2_.solve(I);
        Matrix out(2, 2);
        out = S;
        return out;
    }
    if (dim_ == 3) {
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d S = llt3_.solve(I);
        Matrix out(3, 3);
        out = S;
        return out;
    }

    const Matrix I = Matrix::Identity(dim_, dim_);
    return lltx_.solve(I);
}

void NdimGaussian::ensureFactorized() const {
    if (dim_ == 0) {
        throw std::runtime_error("NdimGaussian::ensureFactorized: dim==0");
    }

    // Cache hit
    if (cache_valid_ && cached_dim_ == dim_) return;

    if (dim_ == 2) {
        Eigen::Matrix2d lam_sym = lam2_;
        lam_sym(1, 0) = lam_sym(0, 1);
        llt2_.compute(lam_sym);
        if (llt2_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (dim==2, not SPD?)");
        }
    } else if (dim_ == 3) {
        Eigen::Matrix3d lam_sym = lam3_;
        lam_sym(1, 0) = lam_sym(0, 1);
        lam_sym(2, 0) = lam_sym(0, 2);
        lam_sym(2, 1) = lam_sym(1, 2);
        llt3_.compute(lam_sym);
        if (llt3_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (dim==3, not SPD?)");
        }
    } else {
        Matrix lam_sym = lam_;
        lam_sym.template triangularView<Eigen::StrictlyLower>() =
            lam_sym.transpose().template triangularView<Eigen::StrictlyLower>();
        lltx_.compute(lam_sym);
        if (lltx_.info() != Eigen::Success) {
            throw std::runtime_error("NdimGaussian::ensureFactorized: LLT failed (dynamic dim, not SPD?)");
        }
    }

    cache_valid_ = true;
    cached_dim_  = dim_;
}

} // namespace utils
