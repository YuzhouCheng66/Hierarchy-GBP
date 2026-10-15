#include "gbp/VariableNode.h"
#include "gbp/Factor.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif


namespace gbp {

namespace {

extern "C" {
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info, std::size_t uplo_len);
void dpotrs_(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info, std::size_t uplo_len);
}

using DpotrfFn = void (*)(const char* uplo, const int* n, double* a, const int* lda, int* info, std::size_t uplo_len);
using DpotrsFn = void (*)(const char* uplo, const int* n, const int* nrhs, const double* a, const int* lda, double* b, const int* ldb, int* info, std::size_t uplo_len);

#if defined(_MSC_VER)
#define GBP_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define GBP_FORCEINLINE inline __attribute__((always_inline))
#else
#define GBP_FORCEINLINE inline
#endif

bool disable3DFastPath() {
    static const bool disabled = []() {
        const char* value = std::getenv("GBP_DISABLE_3D_FASTPATH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

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

struct Cholesky3x3 {
    double l00;
    double u01;
    double u02;
    double l11;
    double u12;
    double l22;
};

GBP_FORCEINLINE bool factorizeSpd3x3(const double* a, Cholesky3x3& chol) noexcept {
    const double a00 = a[0];
    const double a01 = a[3];
    const double a02 = a[6];
    const double a11 = a[4];
    const double a12 = a[7];
    const double a22 = a[8];

    if (!(a00 > 0.0)) return false;
    chol.l00 = std::sqrt(a00);

    chol.u01 = a01 / chol.l00;
    chol.u02 = a02 / chol.l00;

    const double d11 = a11 - chol.u01 * chol.u01;
    if (!(d11 > 0.0)) return false;
    chol.l11 = std::sqrt(d11);

    chol.u12 = (a12 - chol.u01 * chol.u02) / chol.l11;

    const double d22 = a22 - chol.u02 * chol.u02 - chol.u12 * chol.u12;
    if (!(d22 > 0.0)) return false;
    chol.l22 = std::sqrt(d22);
    return true;
}

GBP_FORCEINLINE void solveSpd3x3(const Cholesky3x3& chol, const double* b, double* x) noexcept {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.u01 * y0) / chol.l11;
    const double y2 = (b[2] - chol.u02 * y0 - chol.u12 * y1) / chol.l22;

    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.u12 * x[2]) / chol.l11;
    x[0] = (y0 - chol.u01 * x[1] - chol.u02 * x[2]) / chol.l00;
}

GBP_FORCEINLINE bool solveGeneral3x3UpperSym(
    double a00,
    double a01,
    double a02,
    double a11,
    double a12,
    double a22,
    const double* b,
    double* x
) noexcept {
    Eigen::Matrix3d A;
    A << a00, a01, a02,
         a01, a11, a12,
         a02, a12, a22;
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

GBP_FORCEINLINE bool lapackSolveUpperSpd3x3Vec(
    const double* a_upper,
    const double* rhs,
    double* out
) noexcept {
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

#undef GBP_FORCEINLINE

}  // namespace

void VariableNode::updateBeliefImpl_(bool update_mu) {
    if (!active) return;
    if (dofs <= 0) return;

    ensureCache_();

    if (dofs == 2) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;

        Vec2 eta2 = prior.eta().head<2>();
        Mat2 lam2 = prior.lam().topLeftCorner<2, 2>();

        for (const auto& aref : adj_factors) {
            const Factor* f = aref.factor;
            const int k = aref.local_idx;
            assert(f != nullptr);
            assert(k >= 0 && k < (int)f->messages.size());
            eta2.noalias() += f->messages[k].eta().head<2>();
            lam2.noalias() += f->messages[k].lam().topLeftCorner<2, 2>();
        }

        belief.etaRef().head<2>() = eta2;
        belief.lamRef().topLeftCorner<2, 2>() = lam2;
        if (update_mu) {
            mu2 = belief.mu().head<2>();
            mu = mu2;
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
        return;
    }

    if (dofs == 3 && !disable3DFastPath()) {
        double* belief_eta = belief.etaData();
        double* belief_lam = belief.lamData();
        const double* prior_eta = prior.etaData();
        const double* prior_lam = prior.lamData();

        double eta0 = prior_eta[0];
        double eta1 = prior_eta[1];
        double eta2 = prior_eta[2];
        double lam00 = prior_lam[0];
        double lam10 = prior_lam[1];
        double lam20 = prior_lam[2];
        double lam01 = prior_lam[3];
        double lam11 = prior_lam[4];
        double lam21 = prior_lam[5];
        double lam02 = prior_lam[6];
        double lam12 = prior_lam[7];
        double lam22 = prior_lam[8];

        for (const auto& aref : adj_factors) {
            assert(aref.msg_eta_slot != nullptr);
            assert(aref.msg_lam_slot != nullptr);
            const double* msg_eta = *aref.msg_eta_slot;
            const double* msg_lam = *aref.msg_lam_slot;
            eta0 += msg_eta[0];
            eta1 += msg_eta[1];
            eta2 += msg_eta[2];
            lam00 += msg_lam[0];
            lam10 += msg_lam[1];
            lam20 += msg_lam[2];
            lam01 += msg_lam[3];
            lam11 += msg_lam[4];
            lam21 += msg_lam[5];
            lam02 += msg_lam[6];
            lam12 += msg_lam[7];
            lam22 += msg_lam[8];
        }

        belief_eta[0] = eta0;
        belief_eta[1] = eta1;
        belief_eta[2] = eta2;
        belief_lam[0] = lam00;
        belief_lam[1] = lam10;
        belief_lam[2] = lam20;
        belief_lam[3] = lam01;
        belief_lam[4] = lam11;
        belief_lam[5] = lam21;
        belief_lam[6] = lam02;
        belief_lam[7] = lam12;
        belief_lam[8] = lam22;

        if (update_mu) {
            if (lapackSolveUpperSpd3x3Vec(belief_lam, belief_eta, mu.data())) {
                markMuCurrent();
                return;
            }
            Cholesky3x3 chol;
            if (!factorizeSpd3x3(belief_lam, chol)) {
                if (!solveGeneral3x3UpperSym(
                        lam00, lam01, lam02,
                        lam11, lam12, lam22,
                        belief_eta,
                        mu.data()
                    )) {
                    throw std::runtime_error("Linear solve failed in VariableNode::updateBelief (3D fast path)");
                }
                markMuCurrent();
                return;
            }
            solveSpd3x3(chol, belief_eta, mu.data());
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
        return;
    }

    if (dofs == 3) {
        double* belief_eta = belief.etaData();
        double* belief_lam = belief.lamData();
        const double* prior_eta = prior.etaData();
        const double* prior_lam = prior.lamData();

        long double eta0 = prior_eta[0];
        long double eta1 = prior_eta[1];
        long double eta2 = prior_eta[2];
        long double lam00 = prior_lam[0];
        long double lam10 = prior_lam[1];
        long double lam20 = prior_lam[2];
        long double lam01 = prior_lam[3];
        long double lam11 = prior_lam[4];
        long double lam21 = prior_lam[5];
        long double lam02 = prior_lam[6];
        long double lam12 = prior_lam[7];
        long double lam22 = prior_lam[8];

        for (const auto& aref : adj_factors) {
            const Factor* f = aref.factor;
            const int k = aref.local_idx;
            assert(f != nullptr);
            assert(k >= 0 && k < (int)f->messages.size());
            const double* msg_eta = f->messages[k].etaData();
            const double* msg_lam = f->messages[k].lamData();
            eta0 += msg_eta[0];
            eta1 += msg_eta[1];
            eta2 += msg_eta[2];
            lam00 += msg_lam[0];
            lam10 += msg_lam[1];
            lam20 += msg_lam[2];
            lam01 += msg_lam[3];
            lam11 += msg_lam[4];
            lam21 += msg_lam[5];
            lam02 += msg_lam[6];
            lam12 += msg_lam[7];
            lam22 += msg_lam[8];
        }

        belief_eta[0] = static_cast<double>(eta0);
        belief_eta[1] = static_cast<double>(eta1);
        belief_eta[2] = static_cast<double>(eta2);
        belief_lam[0] = static_cast<double>(lam00);
        belief_lam[1] = static_cast<double>(lam10);
        belief_lam[2] = static_cast<double>(lam20);
        belief_lam[3] = static_cast<double>(lam01);
        belief_lam[4] = static_cast<double>(lam11);
        belief_lam[5] = static_cast<double>(lam21);
        belief_lam[6] = static_cast<double>(lam02);
        belief_lam[7] = static_cast<double>(lam12);
        belief_lam[8] = static_cast<double>(lam22);

        if (update_mu) {
            mu = belief.mu();
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
        return;
    }

    if (dofs == 6) {
        double* belief_eta = belief.etaData();
        auto belief_lam_ref = belief.lamRef();
        double* belief_lam = belief_lam_ref.data();
        const double* prior_eta = prior.etaData();
        const double* prior_lam = prior.lamData();

        double eta[6];
        double lam[36];
        std::memcpy(eta, prior_eta, 6 * sizeof(double));
        std::memcpy(lam, prior_lam, 36 * sizeof(double));

        for (const auto& aref : adj_factors) {
            assert(aref.factor != nullptr);
            assert(aref.local_idx >= 0);
            const double* msg_eta = nullptr;
            const double* msg_lam = nullptr;
            if (aref.msg_eta_slot != nullptr) {
                msg_eta = *aref.msg_eta_slot;
            }
            if (aref.msg_lam_slot != nullptr) {
                msg_lam = *aref.msg_lam_slot;
            }
            if (msg_eta == nullptr || msg_lam == nullptr) {
                assert(aref.local_idx < (int)aref.factor->messages.size());
                msg_eta = aref.factor->messages[aref.local_idx].etaData();
                msg_lam = aref.factor->messages[aref.local_idx].lamData();
            }
            for (int i = 0; i < 6; ++i) {
                eta[i] += msg_eta[i];
            }
            for (int i = 0; i < 36; ++i) {
                lam[i] += msg_lam[i];
            }
        }

        std::memcpy(belief_eta, eta, 6 * sizeof(double));
        std::memcpy(belief_lam, lam, 36 * sizeof(double));
        if (update_mu) {
            belief.solveMuInto(mu.data());
            markMuCurrent();
        } else {
            mu_valid_ = false;
        }
        return;
    }

    eta_acc_.noalias() = prior.eta();
    lam_acc_.noalias() = prior.lam();

    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());
        eta_acc_.noalias() += f->messages[k].eta();
        lam_acc_.noalias() += f->messages[k].lam();
    }

    belief.etaRef().noalias() = eta_acc_;
    belief.lamRef().noalias() = lam_acc_;
    if (update_mu) {
        mu = belief.mu();
        markMuCurrent();
    } else {
        mu_valid_ = false;
    }
}

VariableNode::VariableNode(int id_, int dofs_)
    : id(id_),
      variableID(id_),
      dofs(dofs_),
      dim(dofs_),
      active(true),
      prior(dofs_),
      belief(dofs_),
      GT(Eigen::VectorXd::Zero(dofs_)),
      mu2(Eigen::Vector2d::Zero()),
      mu(Eigen::VectorXd::Zero(dofs_)),
      eta_acc_(Eigen::VectorXd::Zero(dofs_)),
      lam_acc_(Eigen::MatrixXd::Zero(dofs_, dofs_)),
      lam_work_(Eigen::MatrixXd::Zero(dofs_, dofs_))
{
    // Nothing else
}

VariableNode::VariableNode()
    : id(-1),
      variableID(-1),
      dofs(0),
      dim(0),
      active(true),
      prior(0),
      belief(0),
      GT(),
      mu2(Eigen::Vector2d::Zero()),
      mu(Eigen::VectorXd::Zero(0)),
      eta_acc_(),
      lam_acc_(),
      lam_work_()
{
    // Nothing else
}

void VariableNode::updateBelief() {
    updateBeliefImpl_(true);
}

void VariableNode::updateBeliefNoMu() {
    updateBeliefImpl_(false);
}

void VariableNode::updateBeliefEtaOnly() {
    if (!active) return;
    if (dofs <= 0) return;

    if (dofs == 6) {
        double* belief_eta = belief.etaData();
        const double* prior_eta = prior.etaData();
        double eta0 = prior_eta[0];
        double eta1 = prior_eta[1];
        double eta2 = prior_eta[2];
        double eta3 = prior_eta[3];
        double eta4 = prior_eta[4];
        double eta5 = prior_eta[5];

        for (const auto& aref : adj_factors) {
            assert(aref.factor != nullptr);
            assert(aref.local_idx >= 0);
            const double* msg_eta = nullptr;
            if (aref.msg_eta_slot != nullptr) {
                msg_eta = *aref.msg_eta_slot;
            }
            if (msg_eta == nullptr) {
                assert(aref.local_idx < (int)aref.factor->messages.size());
                msg_eta = aref.factor->messages[aref.local_idx].etaData();
            }
            eta0 += msg_eta[0];
            eta1 += msg_eta[1];
            eta2 += msg_eta[2];
            eta3 += msg_eta[3];
            eta4 += msg_eta[4];
            eta5 += msg_eta[5];
        }

        belief_eta[0] = eta0;
        belief_eta[1] = eta1;
        belief_eta[2] = eta2;
        belief_eta[3] = eta3;
        belief_eta[4] = eta4;
        belief_eta[5] = eta5;
        belief.solveMuInto(mu.data());
        markMuCurrent();
        return;
    }

    ensureCache_();
    eta_acc_.noalias() = prior.eta();
    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());
        eta_acc_.noalias() += f->messages[k].eta();
    }
    belief.etaRef().noalias() = eta_acc_;
    mu = belief.mu();
    markMuCurrent();
}

void VariableNode::updateBeliefEtaOnlyNoMu() {
    if (!active) return;
    if (dofs <= 0) return;

    if (dofs == 6) {
        double* belief_eta = belief.etaData();
        const double* prior_eta = prior.etaData();
        double eta0 = prior_eta[0];
        double eta1 = prior_eta[1];
        double eta2 = prior_eta[2];
        double eta3 = prior_eta[3];
        double eta4 = prior_eta[4];
        double eta5 = prior_eta[5];

        for (const auto& aref : adj_factors) {
            assert(aref.factor != nullptr);
            assert(aref.local_idx >= 0);
            const double* msg_eta = nullptr;
            if (aref.msg_eta_slot != nullptr) {
                msg_eta = *aref.msg_eta_slot;
            }
            if (msg_eta == nullptr) {
                assert(aref.local_idx < (int)aref.factor->messages.size());
                msg_eta = aref.factor->messages[aref.local_idx].etaData();
            }
            eta0 += msg_eta[0];
            eta1 += msg_eta[1];
            eta2 += msg_eta[2];
            eta3 += msg_eta[3];
            eta4 += msg_eta[4];
            eta5 += msg_eta[5];
        }

        belief_eta[0] = eta0;
        belief_eta[1] = eta1;
        belief_eta[2] = eta2;
        belief_eta[3] = eta3;
        belief_eta[4] = eta4;
        belief_eta[5] = eta5;
        invalidateMu();
        return;
    }

    ensureCache_();
    eta_acc_.noalias() = prior.eta();
    for (const auto& aref : adj_factors) {
        const Factor* f = aref.factor;
        const int k = aref.local_idx;
        assert(f != nullptr);
        assert(k >= 0 && k < (int)f->messages.size());
        eta_acc_.noalias() += f->messages[k].eta();
    }
    belief.etaRef().noalias() = eta_acc_;
    invalidateMu();
}

void VariableNode::refreshMu() {
    if (mu_valid_) {
        return;
    }

    if (dofs == 2) {
        using Vec2 = Eigen::Matrix<double, 2, 1>;
        using Mat2 = Eigen::Matrix<double, 2, 2>;
        const Eigen::Map<const Vec2> eta2(belief.etaData());
        Mat2 lam2 = Eigen::Map<const Mat2>(belief.lamData());
        lam2(1, 0) = lam2(0, 1);
        Eigen::LLT<Mat2, Eigen::Upper> llt;
        llt.compute(lam2);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("LLT failed in VariableNode::refreshMu (2D)");
        }
        mu2 = llt.solve(eta2);
        mu = mu2;
        mu_valid_ = true;
        return;
    }

    if (dofs == 3 && !disable3DFastPath()) {
        if (lapackSolveUpperSpd3x3Vec(belief.lamData(), belief.etaData(), mu.data())) {
            mu_valid_ = true;
            return;
        }
        Cholesky3x3 chol;
        if (!factorizeSpd3x3(belief.lamData(), chol)) {
            const double* lam = belief.lamData();
            if (!solveGeneral3x3UpperSym(
                    lam[0], lam[3], lam[6],
                    lam[4], lam[7], lam[8],
                    belief.etaData(),
                    mu.data()
                )) {
                throw std::runtime_error("Linear solve failed in VariableNode::refreshMu (3D)");
            }
            mu_valid_ = true;
            return;
        }
        solveSpd3x3(chol, belief.etaData(), mu.data());
        mu_valid_ = true;
        return;
    }

    mu = belief.mu();
    mu_valid_ = true;
}

} // namespace gbp
