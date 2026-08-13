#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>   // std::move
#include <cstring>   // std::memcpy
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
constexpr double kRawSchurJitter = 1e-10;

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

bool enableHighPrecisionSchur3x3() {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_HIGHPREC_SCHUR_3X3");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

bool enableGenericRidgeRetry() {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_FACTOR_MSG_RIDGE_RETRY");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return value[0] != '0';
    }();
    return enabled;
}

double genericRidgeRetryStart() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_RIDGE_START");
        if (raw == nullptr || raw[0] == '\0') {
            return 1e-9;
        }
        try {
            return std::stod(std::string(raw));
        } catch (...) {
            return 1e-9;
        }
    }();
    return value;
}

double genericRidgeRetryMultiplier() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_RIDGE_MULT");
        if (raw == nullptr || raw[0] == '\0') {
            return 10.0;
        }
        try {
            return std::stod(std::string(raw));
        } catch (...) {
            return 10.0;
        }
    }();
    return value;
}

int genericRidgeRetrySteps() {
    static const int value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_RIDGE_STEPS");
        if (raw == nullptr || raw[0] == '\0') {
            return 12;
        }
        try {
            return std::stoi(std::string(raw));
        } catch (...) {
            return 12;
        }
    }();
    return value;
}

bool enableGenericLdltFallback() {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_FACTOR_MSG_LDLT_FALLBACK");
        if (value == nullptr || value[0] == '\0') {
            return true;
        }
        return value[0] != '0';
    }();
    return enabled;
}

double genericRelativeJitterFloor() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_RELATIVE_JITTER");
        if (raw == nullptr || raw[0] == '\0') {
            return 1e-12;
        }
        try {
            return (std::max)(0.0, std::stod(std::string(raw)));
        } catch (...) {
            return 1e-12;
        }
    }();
    return value;
}

double genericMaxRelativeJitter() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_MAX_RELATIVE_JITTER");
        if (raw == nullptr || raw[0] == '\0') {
            return 1e-8;
        }
        try {
            return (std::max)(0.0, std::stod(std::string(raw)));
        } catch (...) {
            return 1e-8;
        }
    }();
    return value;
}

double genericMessageMaxRelativeUpdate() {
    static const double value = []() {
        const char* raw = std::getenv("GBP_FACTOR_MSG_MAX_REL_UPDATE");
        if (raw == nullptr || raw[0] == '\0') {
            return 0.0;
        }
        try {
            return (std::max)(0.0, std::stod(std::string(raw)));
        } catch (...) {
            return 0.0;
        }
    }();
    return value;
}

int fixedLam6DKernelMode() {
    static const int value = []() {
        const char* raw = std::getenv("GBP_SE3_FIXED_LAMBDA_6D_KERNEL");
        if (raw == nullptr || raw[0] == '\0') {
            return 1;  // 0=old generic, 1=fixed 6x6 LLT, 2=raw maps with dynamic LLT
        }
        try {
            return (std::max)(0, std::stoi(std::string(raw)));
        } catch (...) {
            return 1;
        }
    }();
    return value;
}

bool fixedLam6DInplaceEta() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FIXED_LAMBDA_6D_INPLACE_ETA");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

bool full6DKernelEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FULL_6D_KERNEL");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

bool full6DBatchedSolveEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FULL_6D_BATCHED_SOLVE");
        if (raw == nullptr || raw[0] == '\0') {
            return false;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

bool full6DRawCholeskyEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FULL_6D_RAW_CHOLESKY");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

bool fixedLam6DInverseCacheEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FIXED_LAMBDA_6D_INV_CACHE");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

bool fixedLam6DEtaMapCacheEnabled() {
    static const bool enabled = []() {
        const char* raw = std::getenv("GBP_SE3_FIXED_LAMBDA_6D_ETA_MAP_CACHE");
        if (raw == nullptr || raw[0] == '\0') {
            return true;
        }
        return raw[0] != '0';
    }();
    return enabled;
}

template <typename Derived, typename LltType>
bool factorizeGenericSchurCavityWithJitter(
    Eigen::MatrixBase<Derived>& cavity_base,
    LltType& llt,
    double& applied_jitter
) {
    Derived& cavity = cavity_base.derived();
    applied_jitter = 0.0;
    llt.compute(cavity);
    if (llt.info() == Eigen::Success) {
        return true;
    }
    if (!enableGenericRidgeRetry()) {
        return false;
    }

    const double scale = (std::max)(1.0, cavity.cwiseAbs().maxCoeff());
    const double floor = (std::max)(genericRidgeRetryStart(), genericRelativeJitterFloor() * scale);
    const double max_total_jitter = (std::max)(floor, genericMaxRelativeJitter() * scale);

    double ridge = floor;
    const double ridge_mult = (std::max)(1.1, genericRidgeRetryMultiplier());
    const int ridge_steps = (std::max)(1, genericRidgeRetrySteps());
    for (int step = 0; step < ridge_steps; ++step) {
        if (applied_jitter + ridge > max_total_jitter) {
            break;
        }
        cavity.diagonal().array() += ridge;
        applied_jitter += ridge;
        llt.compute(cavity);
        if (llt.info() == Eigen::Success) {
            return true;
        }
        ridge *= ridge_mult;
    }
    return false;
}

template <
    typename OutLamDerived,
    typename OutEtaDerived,
    typename OldLamDerived,
    typename OldEtaDerived,
    typename RefLamDerived,
    typename RefEtaDerived
>
void stabilizeGenericMessageUpdate(
    Eigen::MatrixBase<OutLamDerived>& out_lam_base,
    Eigen::MatrixBase<OutEtaDerived>& out_eta_base,
    const Eigen::MatrixBase<OldLamDerived>& old_lam,
    const Eigen::MatrixBase<OldEtaDerived>& old_eta,
    const Eigen::MatrixBase<RefLamDerived>& ref_lam,
    const Eigen::MatrixBase<RefEtaDerived>& ref_eta
) {
    OutLamDerived& out_lam = out_lam_base.derived();
    OutEtaDerived& out_eta = out_eta_base.derived();

    // The Schur complement should be symmetric; force tiny numeric asymmetry
    // out before variable beliefs accumulate it over many sweeps.
    typename OutLamDerived::PlainObject out_lam_t = out_lam.transpose();
    out_lam = 0.5 * (out_lam + out_lam_t);

    const double max_rel_update = genericMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }

    const double lam_ref = (std::max)(1.0, (std::max)(old_lam.norm(), ref_lam.norm()));
    const double eta_ref = (std::max)(1.0, (std::max)(old_eta.norm(), ref_eta.norm()));
    const double rel_lam = (out_lam - old_lam).norm() / lam_ref;
    const double rel_eta = (out_eta - old_eta).norm() / eta_ref;
    const double rel_update = std::sqrt(rel_lam * rel_lam + rel_eta * rel_eta);
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }

    const double step = max_rel_update / (std::max)(rel_update, 1e-300);
    out_lam = old_lam + step * (out_lam - old_lam);
    out_eta = old_eta + step * (out_eta - old_eta);
}

template <
    typename OutEtaDerived,
    typename OldEtaDerived,
    typename RefEtaDerived
>
void stabilizeGenericEtaUpdate(
    Eigen::MatrixBase<OutEtaDerived>& out_eta_base,
    const Eigen::MatrixBase<OldEtaDerived>& old_eta,
    const Eigen::MatrixBase<RefEtaDerived>& ref_eta
) {
    OutEtaDerived& out_eta = out_eta_base.derived();
    const double max_rel_update = genericMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }

    const double eta_ref = (std::max)(1.0, (std::max)(old_eta.norm(), ref_eta.norm()));
    const double rel_update = (out_eta - old_eta).norm() / eta_ref;
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }

    const double step = max_rel_update / (std::max)(rel_update, 1e-300);
    out_eta = old_eta + step * (out_eta - old_eta);
}

GBP_FORCEINLINE double sqNorm6Raw(const double* x) noexcept {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] +
           x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
}

GBP_FORCEINLINE double maxAbs36Raw(const double* x) noexcept {
    double out = 0.0;
    for (int i = 0; i < 36; ++i) {
        out = (std::max)(out, std::abs(x[i]));
    }
    return out;
}

GBP_FORCEINLINE void stabilizeEta6Raw(
    double* out_eta,
    const double* old_eta,
    const double* ref_eta
) noexcept {
    const double max_rel_update = genericMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }

    const double old_norm = std::sqrt(sqNorm6Raw(old_eta));
    const double ref_norm = std::sqrt(sqNorm6Raw(ref_eta));
    const double eta_ref = (std::max)(1.0, (std::max)(old_norm, ref_norm));
    double diff_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double diff = out_eta[i] - old_eta[i];
        diff_sq += diff * diff;
    }
    const double rel_update = std::sqrt(diff_sq) / eta_ref;
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }

    const double step = max_rel_update / (std::max)(rel_update, 1e-300);
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
}

GBP_FORCEINLINE void stabilizeMessage6Raw(
    double* out_lam,
    double* out_eta,
    const double* old_lam,
    const double* old_eta,
    const double* ref_lam,
    int ref_lam_stride,
    const double* ref_eta
) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = col + 1; row < 6; ++row) {
            const int upper_idx = row * 6 + col;
            const int lower_idx = col * 6 + row;
            const double sym = 0.5 * (out_lam[upper_idx] + out_lam[lower_idx]);
            out_lam[upper_idx] = sym;
            out_lam[lower_idx] = sym;
        }
    }

    const double max_rel_update = genericMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }

    double old_lam_sq = 0.0;
    double ref_lam_sq = 0.0;
    double diff_lam_sq = 0.0;
    for (int col = 0; col < 6; ++col) {
        const int out_base = 6 * col;
        const int ref_base = ref_lam_stride * col;
        for (int row = 0; row < 6; ++row) {
            const double old_v = old_lam[out_base + row];
            const double ref_v = ref_lam[ref_base + row];
            const double diff = out_lam[out_base + row] - old_v;
            old_lam_sq += old_v * old_v;
            ref_lam_sq += ref_v * ref_v;
            diff_lam_sq += diff * diff;
        }
    }

    double old_eta_sq = 0.0;
    double ref_eta_sq = 0.0;
    double diff_eta_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double old_v = old_eta[i];
        const double ref_v = ref_eta[i];
        const double diff = out_eta[i] - old_v;
        old_eta_sq += old_v * old_v;
        ref_eta_sq += ref_v * ref_v;
        diff_eta_sq += diff * diff;
    }

    const double lam_ref = (std::max)(1.0, (std::max)(std::sqrt(old_lam_sq), std::sqrt(ref_lam_sq)));
    const double eta_ref = (std::max)(1.0, (std::max)(std::sqrt(old_eta_sq), std::sqrt(ref_eta_sq)));
    const double rel_lam = std::sqrt(diff_lam_sq) / lam_ref;
    const double rel_eta = std::sqrt(diff_eta_sq) / eta_ref;
    const double rel_update = std::sqrt(rel_lam * rel_lam + rel_eta * rel_eta);
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }

    const double step = max_rel_update / (std::max)(rel_update, 1e-300);
    for (int i = 0; i < 36; ++i) {
        out_lam[i] = old_lam[i] + step * (out_lam[i] - old_lam[i]);
    }
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
}

GBP_FORCEINLINE bool factorizeSpd6LowerRaw(const double* a, double* l) noexcept {
    const double l00 = std::sqrt(a[0]);
    if (!(l00 > 0.0) || !std::isfinite(l00)) return false;
    const double l10 = a[1] / l00;
    const double l20 = a[2] / l00;
    const double l30 = a[3] / l00;
    const double l40 = a[4] / l00;
    const double l50 = a[5] / l00;

    const double d1 = a[7] - l10 * l10;
    if (!(d1 > 0.0) || !std::isfinite(d1)) return false;
    const double l11 = std::sqrt(d1);
    const double l21 = (a[8] - l20 * l10) / l11;
    const double l31 = (a[9] - l30 * l10) / l11;
    const double l41 = (a[10] - l40 * l10) / l11;
    const double l51 = (a[11] - l50 * l10) / l11;

    const double d2 = a[14] - l20 * l20 - l21 * l21;
    if (!(d2 > 0.0) || !std::isfinite(d2)) return false;
    const double l22 = std::sqrt(d2);
    const double l32 = (a[15] - l30 * l20 - l31 * l21) / l22;
    const double l42 = (a[16] - l40 * l20 - l41 * l21) / l22;
    const double l52 = (a[17] - l50 * l20 - l51 * l21) / l22;

    const double d3 = a[21] - l30 * l30 - l31 * l31 - l32 * l32;
    if (!(d3 > 0.0) || !std::isfinite(d3)) return false;
    const double l33 = std::sqrt(d3);
    const double l43 = (a[22] - l40 * l30 - l41 * l31 - l42 * l32) / l33;
    const double l53 = (a[23] - l50 * l30 - l51 * l31 - l52 * l32) / l33;

    const double d4 = a[28] - l40 * l40 - l41 * l41 - l42 * l42 - l43 * l43;
    if (!(d4 > 0.0) || !std::isfinite(d4)) return false;
    const double l44 = std::sqrt(d4);
    const double l54 = (a[29] - l50 * l40 - l51 * l41 - l52 * l42 - l53 * l43) / l44;

    const double d5 =
        a[35] - l50 * l50 - l51 * l51 - l52 * l52 - l53 * l53 - l54 * l54;
    if (!(d5 > 0.0) || !std::isfinite(d5)) return false;
    const double l55 = std::sqrt(d5);

    l[0] = l00;
    l[1] = l10;  l[7] = l11;
    l[2] = l20;  l[8] = l21;  l[14] = l22;
    l[3] = l30;  l[9] = l31;  l[15] = l32;  l[21] = l33;
    l[4] = l40;  l[10] = l41; l[16] = l42; l[22] = l43; l[28] = l44;
    l[5] = l50;  l[11] = l51; l[17] = l52; l[23] = l53; l[29] = l54; l[35] = l55;
    return true;
}

GBP_FORCEINLINE bool factorizeSpd6LowerWithJitterRaw(double* cavity, double* l) noexcept {
    if (factorizeSpd6LowerRaw(cavity, l)) {
        return true;
    }
    if (!enableGenericRidgeRetry()) {
        return false;
    }

    const double scale = (std::max)(1.0, maxAbs36Raw(cavity));
    const double floor = (std::max)(genericRidgeRetryStart(), genericRelativeJitterFloor() * scale);
    const double max_total_jitter = (std::max)(floor, genericMaxRelativeJitter() * scale);

    double applied_jitter = 0.0;
    double ridge = floor;
    const double ridge_mult = (std::max)(1.1, genericRidgeRetryMultiplier());
    const int ridge_steps = (std::max)(1, genericRidgeRetrySteps());
    for (int step = 0; step < ridge_steps; ++step) {
        if (applied_jitter + ridge > max_total_jitter) {
            break;
        }
        for (int i = 0; i < 6; ++i) {
            cavity[6 * i + i] += ridge;
        }
        applied_jitter += ridge;
        if (factorizeSpd6LowerRaw(cavity, l)) {
            return true;
        }
        ridge *= ridge_mult;
    }
    return false;
}

GBP_FORCEINLINE void solveSpd6LowerOneRaw(
    const double* l,
    const double* rhs,
    double* out
) noexcept {
    const double y0 = rhs[0] / l[0];
    const double y1 = (rhs[1] - l[1] * y0) / l[7];
    const double y2 = (rhs[2] - l[2] * y0 - l[8] * y1) / l[14];
    const double y3 = (rhs[3] - l[3] * y0 - l[9] * y1 - l[15] * y2) / l[21];
    const double y4 =
        (rhs[4] - l[4] * y0 - l[10] * y1 - l[16] * y2 - l[22] * y3) / l[28];
    const double y5 =
        (rhs[5] - l[5] * y0 - l[11] * y1 - l[17] * y2 - l[23] * y3 - l[29] * y4) /
        l[35];

    const double x5 = y5 / l[35];
    const double x4 = (y4 - l[29] * x5) / l[28];
    const double x3 = (y3 - l[22] * x4 - l[23] * x5) / l[21];
    const double x2 = (y2 - l[16] * x4 - l[17] * x5 - l[15] * x3) / l[14];
    const double x1 =
        (y1 - l[10] * x4 - l[11] * x5 - l[9] * x3 - l[8] * x2) / l[7];
    const double x0 =
        (y0 - l[4] * x4 - l[5] * x5 - l[3] * x3 - l[2] * x2 - l[1] * x1) /
        l[0];

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = x3;
    out[4] = x4;
    out[5] = x5;
}

GBP_FORCEINLINE void solveSpd6LowerRaw(
    const double* l,
    const double* rhs,
    int nrhs,
    double* out
) noexcept {
    for (int col = 0; col < nrhs; ++col) {
        solveSpd6LowerOneRaw(l, rhs + 6 * col, out + 6 * col);
    }
}

GBP_FORCEINLINE bool schurMessage6Raw(
    const double* factor_eta,
    const double* factor_lam,
    int target_offset,
    int other_offset,
    const double* belief_other_eta,
    const double* belief_other_lam,
    const double* old_target_eta,
    const double* old_target_lam,
    const double* old_other_eta,
    const double* old_other_lam,
    double eta_damping,
    double* out_eta,
    double* out_lam
) noexcept {
    double cavity[36];
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            cavity[6 * col + row] =
                factor_lam[12 * (other_offset + col) + other_offset + row] +
                belief_other_lam[6 * col + row] -
                old_other_lam[6 * col + row];
        }
    }
    for (int col = 0; col < 6; ++col) {
        for (int row = col + 1; row < 6; ++row) {
            cavity[6 * col + row] = cavity[6 * row + col];
        }
    }
    for (int i = 0; i < 6; ++i) {
        cavity[6 * i + i] += kRawSchurJitter;
    }

    double l[36];
    if (!factorizeSpd6LowerWithJitterRaw(cavity, l)) {
        return false;
    }

    double rhs[42];
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            rhs[6 * col + row] =
                factor_lam[12 * (target_offset + col) + other_offset + row];
        }
    }
    for (int row = 0; row < 6; ++row) {
        rhs[36 + row] =
            factor_eta[other_offset + row] +
            belief_other_eta[row] -
            old_other_eta[row];
    }

    double solved[42];
    solveSpd6LowerRaw(l, rhs, 7, solved);

    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            double projected = 0.0;
            for (int k = 0; k < 6; ++k) {
                projected +=
                    factor_lam[12 * (other_offset + k) + target_offset + row] *
                    solved[6 * col + k];
            }
            out_lam[6 * col + row] =
                factor_lam[12 * (target_offset + col) + target_offset + row] -
                projected;
        }
    }
    for (int row = 0; row < 6; ++row) {
        double projected = 0.0;
        for (int k = 0; k < 6; ++k) {
            projected +=
                factor_lam[12 * (other_offset + k) + target_offset + row] *
                solved[36 + k];
        }
        out_eta[row] = factor_eta[target_offset + row] - projected;
    }

    if (eta_damping != 0.0) {
        const double keep_new = 1.0 - eta_damping;
        for (int i = 0; i < 36; ++i) {
            out_lam[i] = keep_new * out_lam[i] + eta_damping * old_target_lam[i];
        }
        for (int i = 0; i < 6; ++i) {
            out_eta[i] = keep_new * out_eta[i] + eta_damping * old_target_eta[i];
        }
    }

    stabilizeMessage6Raw(
        out_lam,
        out_eta,
        old_target_lam,
        old_target_eta,
        factor_lam + 12 * target_offset + target_offset,
        12,
        factor_eta + target_offset);
    return true;
}

GBP_FORCEINLINE void matVec6ColMajorRaw(
    const double* matrix,
    int leading_dim,
    const double* x,
    double* out
) noexcept {
    out[0] = matrix[0] * x[0] + matrix[leading_dim] * x[1] +
             matrix[2 * leading_dim] * x[2] + matrix[3 * leading_dim] * x[3] +
             matrix[4 * leading_dim] * x[4] + matrix[5 * leading_dim] * x[5];
    out[1] = matrix[1] * x[0] + matrix[leading_dim + 1] * x[1] +
             matrix[2 * leading_dim + 1] * x[2] + matrix[3 * leading_dim + 1] * x[3] +
             matrix[4 * leading_dim + 1] * x[4] + matrix[5 * leading_dim + 1] * x[5];
    out[2] = matrix[2] * x[0] + matrix[leading_dim + 2] * x[1] +
             matrix[2 * leading_dim + 2] * x[2] + matrix[3 * leading_dim + 2] * x[3] +
             matrix[4 * leading_dim + 2] * x[4] + matrix[5 * leading_dim + 2] * x[5];
    out[3] = matrix[3] * x[0] + matrix[leading_dim + 3] * x[1] +
             matrix[2 * leading_dim + 3] * x[2] + matrix[3 * leading_dim + 3] * x[3] +
             matrix[4 * leading_dim + 3] * x[4] + matrix[5 * leading_dim + 3] * x[5];
    out[4] = matrix[4] * x[0] + matrix[leading_dim + 4] * x[1] +
             matrix[2 * leading_dim + 4] * x[2] + matrix[3 * leading_dim + 4] * x[3] +
             matrix[4 * leading_dim + 4] * x[4] + matrix[5 * leading_dim + 4] * x[5];
    out[5] = matrix[5] * x[0] + matrix[leading_dim + 5] * x[1] +
             matrix[2 * leading_dim + 5] * x[2] + matrix[3 * leading_dim + 5] * x[3] +
             matrix[4 * leading_dim + 5] * x[4] + matrix[5 * leading_dim + 5] * x[5];
}

GBP_FORCEINLINE void fixedLambdaEta6FromCachedInverseRaw(
    const double* factor_eta,
    const double* factor_lam12,
    int target_offset,
    int other_offset,
    const double* inverse6,
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

    double y[6];
    matVec6ColMajorRaw(inverse6, 6, eno, y);

    double schur_eta[6];
    const double* lono = factor_lam12 + other_offset * 12 + target_offset;
    matVec6ColMajorRaw(lono, 12, y, schur_eta);

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
    stabilizeEta6Raw(out_eta, old_target_eta, factor_eta + target_offset);
}

GBP_FORCEINLINE void fixedLambdaEta6FromCachedEtaMapRaw(
    const double* factor_eta,
    int target_offset,
    int other_offset,
    const double* eta_map6,
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
    matVec6ColMajorRaw(eta_map6, 6, eno, schur_eta);

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
    stabilizeEta6Raw(out_eta, old_target_eta, factor_eta + target_offset);
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

GBP_FORCEINLINE bool factorizeSpd3x3Upper(
    double a00,
    double a01,
    double a02,
    double a11,
    double a12,
    double a22,
    Cholesky3x3& chol
) noexcept {
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

GBP_FORCEINLINE void schurMessage3x3(
    const Cholesky3x3& chol,
    const double* eo,
    const double* eno,
    const double* loo,
    const double* lono,
    const double* lnoo,
    const double* old_eta,
    const double* old_lam,
    double a,
    bool no_damping,
    double* out_eta,
    double* out_lam
) noexcept {
    double Y[9];
    double y[3];
    solveSpd3x3(chol, eno, y);
    solveSpd3x3(chol, lnoo + 0, Y + 0);
    solveSpd3x3(chol, lnoo + 3, Y + 3);
    solveSpd3x3(chol, lnoo + 6, Y + 6);

    const double proj_eta0 = lono[0] * y[0] + lono[3] * y[1] + lono[6] * y[2];
    const double proj_eta1 = lono[1] * y[0] + lono[4] * y[1] + lono[7] * y[2];
    const double proj_eta2 = lono[2] * y[0] + lono[5] * y[1] + lono[8] * y[2];

    if (no_damping) {
        out_eta[0] = eo[0] - proj_eta0;
        out_eta[1] = eo[1] - proj_eta1;
        out_eta[2] = eo[2] - proj_eta2;
    } else {
        const double s = 1.0 - a;
        out_eta[0] = s * (eo[0] - proj_eta0) + a * old_eta[0];
        out_eta[1] = s * (eo[1] - proj_eta1) + a * old_eta[1];
        out_eta[2] = s * (eo[2] - proj_eta2) + a * old_eta[2];
    }

    for (int col = 0; col < 3; ++col) {
        const double proj0 = lono[0] * Y[3 * col + 0] + lono[3] * Y[3 * col + 1] + lono[6] * Y[3 * col + 2];
        const double proj1 = lono[1] * Y[3 * col + 0] + lono[4] * Y[3 * col + 1] + lono[7] * Y[3 * col + 2];
        const double proj2 = lono[2] * Y[3 * col + 0] + lono[5] * Y[3 * col + 1] + lono[8] * Y[3 * col + 2];

        const int base = 3 * col;
        if (no_damping) {
            out_lam[base + 0] = loo[base + 0] - proj0;
            out_lam[base + 1] = loo[base + 1] - proj1;
            out_lam[base + 2] = loo[base + 2] - proj2;
        } else {
            const double s = 1.0 - a;
            out_lam[base + 0] = s * (loo[base + 0] - proj0) + a * old_lam[base + 0];
            out_lam[base + 1] = s * (loo[base + 1] - proj1) + a * old_lam[base + 1];
            out_lam[base + 2] = s * (loo[base + 2] - proj2) + a * old_lam[base + 2];
        }
    }
}

GBP_FORCEINLINE bool solveGeneral3x3(
    const Eigen::Matrix3d& A,
    const Eigen::Matrix3d& B,
    const Eigen::Vector3d& rhs,
    Eigen::Matrix3d& X,
    Eigen::Vector3d& x
) noexcept {
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) {
        return false;
    }
    X = lu.solve(B);
    x = lu.solve(rhs);
    return true;
}

GBP_FORCEINLINE bool lapackSolveUpperSpd3x3Rhs(
    const double* a_upper,
    const double* rhs,
    int nrhs,
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

    std::memcpy(out, rhs, static_cast<size_t>(3 * nrhs) * sizeof(double));
    const char uplo = 'U';
    const int n = 3;
    const int lda = 3;
    const int ldb = 3;
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

GBP_FORCEINLINE bool highPrecisionSolveUpperSpd3x3Rhs(
    const double* a_upper,
    const double* rhs,
    int nrhs,
    double* out
) noexcept {
    long double aug[3][7];
    if (nrhs < 1 || nrhs > 4) {
        return false;
    }
    aug[0][0] = static_cast<long double>(a_upper[0]);
    aug[0][1] = static_cast<long double>(a_upper[3]);
    aug[0][2] = static_cast<long double>(a_upper[6]);
    aug[1][0] = static_cast<long double>(a_upper[3]);
    aug[1][1] = static_cast<long double>(a_upper[4]);
    aug[1][2] = static_cast<long double>(a_upper[7]);
    aug[2][0] = static_cast<long double>(a_upper[6]);
    aug[2][1] = static_cast<long double>(a_upper[7]);
    aug[2][2] = static_cast<long double>(a_upper[8]);
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < nrhs; ++col) {
            aug[row][3 + col] = static_cast<long double>(rhs[3 * col + row]);
        }
    }

    for (int k = 0; k < 3; ++k) {
        int pivot = k;
        long double pivot_abs = std::fabsl(aug[k][k]);
        for (int row = k + 1; row < 3; ++row) {
            const long double cand = std::fabsl(aug[row][k]);
            if (cand > pivot_abs) {
                pivot = row;
                pivot_abs = cand;
            }
        }
        if (!(pivot_abs > 0.0L)) {
            return false;
        }
        if (pivot != k) {
            for (int col = k; col < 3 + nrhs; ++col) {
                std::swap(aug[k][col], aug[pivot][col]);
            }
        }
        const long double diag = aug[k][k];
        for (int row = k + 1; row < 3; ++row) {
            const long double factor = aug[row][k] / diag;
            aug[row][k] = 0.0L;
            for (int col = k + 1; col < 3 + nrhs; ++col) {
                aug[row][col] -= factor * aug[k][col];
            }
        }
    }

    for (int col = 0; col < nrhs; ++col) {
        long double x[3];
        for (int row = 2; row >= 0; --row) {
            long double sum = aug[row][3 + col];
            for (int j = row + 1; j < 3; ++j) {
                sum -= aug[row][j] * x[j];
            }
            if (!(std::fabsl(aug[row][row]) > 0.0L)) {
                return false;
            }
            x[row] = sum / aug[row][row];
        }
        for (int row = 0; row < 3; ++row) {
            out[3 * col + row] = static_cast<double>(x[row]);
        }
    }
    return true;
}

GBP_FORCEINLINE bool highPrecisionSchurMessage3x3(
    const double* eo,
    const double* eno,
    const double* loo,
    const double* lono,
    const double* lnoo,
    const double* lnono_upper,
    const double* old_eta,
    const double* old_lam,
    double a,
    bool no_damping,
    double* out_eta,
    double* out_lam
) noexcept {
    double rhs[12];
    std::memcpy(rhs + 0, lnoo + 0, 3 * sizeof(double));
    std::memcpy(rhs + 3, lnoo + 3, 3 * sizeof(double));
    std::memcpy(rhs + 6, lnoo + 6, 3 * sizeof(double));
    rhs[9] = eno[0];
    rhs[10] = eno[1];
    rhs[11] = eno[2];

    double solved[12];
    if (!highPrecisionSolveUpperSpd3x3Rhs(lnono_upper, rhs, 4, solved)) {
        return false;
    }

    const double* cols[3] = {solved + 0, solved + 3, solved + 6};
    const double* y = solved + 9;
    long double proj_eta[3] = {0.0L, 0.0L, 0.0L};
    for (int row = 0; row < 3; ++row) {
        proj_eta[row] =
            static_cast<long double>(lono[3 * 0 + row]) * static_cast<long double>(y[0]) +
            static_cast<long double>(lono[3 * 1 + row]) * static_cast<long double>(y[1]) +
            static_cast<long double>(lono[3 * 2 + row]) * static_cast<long double>(y[2]);
    }

    if (no_damping) {
        for (int row = 0; row < 3; ++row) {
            out_eta[row] = static_cast<double>(static_cast<long double>(eo[row]) - proj_eta[row]);
        }
    } else {
        const long double s = static_cast<long double>(1.0 - a);
        const long double aa = static_cast<long double>(a);
        for (int row = 0; row < 3; ++row) {
            out_eta[row] = static_cast<double>(s * (static_cast<long double>(eo[row]) - proj_eta[row]) +
                                               aa * static_cast<long double>(old_eta[row]));
        }
    }

    for (int col = 0; col < 3; ++col) {
        const double* yc = cols[col];
        long double proj[3];
        for (int row = 0; row < 3; ++row) {
            proj[row] =
                static_cast<long double>(lono[3 * 0 + row]) * static_cast<long double>(yc[0]) +
                static_cast<long double>(lono[3 * 1 + row]) * static_cast<long double>(yc[1]) +
                static_cast<long double>(lono[3 * 2 + row]) * static_cast<long double>(yc[2]);
        }
        const int base = 3 * col;
        if (no_damping) {
            for (int row = 0; row < 3; ++row) {
                out_lam[base + row] = static_cast<double>(static_cast<long double>(loo[base + row]) - proj[row]);
            }
        } else {
            const long double s = static_cast<long double>(1.0 - a);
            const long double aa = static_cast<long double>(a);
            for (int row = 0; row < 3; ++row) {
                out_lam[base + row] = static_cast<double>(s * (static_cast<long double>(loo[base + row]) - proj[row]) +
                                                          aa * static_cast<long double>(old_lam[base + row]));
            }
        }
    }
    return true;
}

GBP_FORCEINLINE bool lapackSchurMessage3x3(
    const double* eo,
    const double* eno,
    const double* loo,
    const double* lono,
    const double* lnoo,
    const double* lnono_upper,
    const double* old_eta,
    const double* old_lam,
    double a,
    bool no_damping,
    double* out_eta,
    double* out_lam
) noexcept {
    if (enableHighPrecisionSchur3x3()) {
        return highPrecisionSchurMessage3x3(
            eo, eno, loo, lono, lnoo, lnono_upper, old_eta, old_lam, a, no_damping, out_eta, out_lam
        );
    }
    double rhs[12];
    std::memcpy(rhs + 0, lnoo + 0, 3 * sizeof(double));
    std::memcpy(rhs + 3, lnoo + 3, 3 * sizeof(double));
    std::memcpy(rhs + 6, lnoo + 6, 3 * sizeof(double));
    rhs[9] = eno[0];
    rhs[10] = eno[1];
    rhs[11] = eno[2];

    double solved[12];
    if (!lapackSolveUpperSpd3x3Rhs(lnono_upper, rhs, 4, solved)) {
        return false;
    }

    const double* Y0 = solved + 0;
    const double* Y1 = solved + 3;
    const double* Y2 = solved + 6;
    const double* y = solved + 9;

    const double proj_eta0 = lono[0] * y[0] + lono[3] * y[1] + lono[6] * y[2];
    const double proj_eta1 = lono[1] * y[0] + lono[4] * y[1] + lono[7] * y[2];
    const double proj_eta2 = lono[2] * y[0] + lono[5] * y[1] + lono[8] * y[2];

    if (no_damping) {
        out_eta[0] = eo[0] - proj_eta0;
        out_eta[1] = eo[1] - proj_eta1;
        out_eta[2] = eo[2] - proj_eta2;
    } else {
        const double s = 1.0 - a;
        out_eta[0] = s * (eo[0] - proj_eta0) + a * old_eta[0];
        out_eta[1] = s * (eo[1] - proj_eta1) + a * old_eta[1];
        out_eta[2] = s * (eo[2] - proj_eta2) + a * old_eta[2];
    }

    const double* cols[3] = {Y0, Y1, Y2};
    for (int col = 0; col < 3; ++col) {
        const double* yc = cols[col];
        const double proj0 = lono[0] * yc[0] + lono[3] * yc[1] + lono[6] * yc[2];
        const double proj1 = lono[1] * yc[0] + lono[4] * yc[1] + lono[7] * yc[2];
        const double proj2 = lono[2] * yc[0] + lono[5] * yc[1] + lono[8] * yc[2];
        const int base = 3 * col;
        if (no_damping) {
            out_lam[base + 0] = loo[base + 0] - proj0;
            out_lam[base + 1] = loo[base + 1] - proj1;
            out_lam[base + 2] = loo[base + 2] - proj2;
        } else {
            const double s = 1.0 - a;
            out_lam[base + 0] = s * (loo[base + 0] - proj0) + a * old_lam[base + 0];
            out_lam[base + 1] = s * (loo[base + 1] - proj1) + a * old_lam[base + 1];
            out_lam[base + 2] = s * (loo[base + 2] - proj2) + a * old_lam[base + 2];
        }
    }
    return true;
}

#undef GBP_FORCEINLINE

}  // namespace

// ==============================
// ctor + workspace init
// ==============================

Factor::Factor(
    int id_,
    const std::vector<VariableNode*>& vars,
    const std::vector<Eigen::VectorXd>& z,
    const std::vector<Eigen::MatrixXd>& lambda,
    std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas,
    std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)>  jac
)
    : factorID(id_),
      active(true),
      adj_var_nodes(vars),
      measurement(z),
      measurement_lambda(lambda),
      meas_fn(std::move(meas)),
      jac_fn(std::move(jac)),
      factor(0) // resized below
{
    assert(adj_var_nodes.size() == 1 || adj_var_nodes.size() == 2);

    // Cache IDs and compute dofs
    adj_vIDs.reserve(adj_var_nodes.size());

    int total_dofs = 0;
    if (adj_var_nodes.size() == 1) {
        is_unary_  = true;
        is_binary_ = false;

        auto* v0 = adj_var_nodes[0];
        assert(v0 != nullptr);

        adj_vIDs.push_back(v0->variableID);

        d0_ = v0->dofs;
        d1_ = 0;
        D_  = d0_;
        total_dofs = D_;
    } else {
        is_unary_  = false;
        is_binary_ = true;

        auto* v0 = adj_var_nodes[0];
        auto* v1 = adj_var_nodes[1];
        assert(v0 != nullptr && v1 != nullptr);

        adj_vIDs.push_back(v0->variableID);
        adj_vIDs.push_back(v1->variableID);

        d0_ = v0->dofs;
        d1_ = v1->dofs;
        D_  = d0_ + d1_;
        total_dofs = D_;
    }

    // Allocate factor gaussian + linpoint
    factor   = utils::NdimGaussian(total_dofs);
    linpoint = Eigen::VectorXd::Zero(total_dofs);

    // Allocate messages (fixed per-variable dofs) [C: ping-pong buffers]
    messages.reserve(adj_var_nodes.size());
    messages_next.reserve(adj_var_nodes.size());
    for (auto* v : adj_var_nodes) {
        assert(v != nullptr);
        messages.emplace_back(v->dofs);
        messages_next.emplace_back(v->dofs);
    }

    // Sanity: Python assumes same length
    if (measurement.size() != measurement_lambda.size()) {
        throw std::runtime_error("Factor ctor: measurement and measurement_lambda size mismatch");
    }

    // Pre-allocate workspace (only meaningful for binary; unary is trivial)
    initWorkspace_();
    refreshHotPathPointers_();
}

void Factor::refreshHotPathPointers_() noexcept {
    for (int i = 0; i < 2; ++i) {
        belief_eta_ptr_[i] = nullptr;
        belief_lam_ptr_[i] = nullptr;
        msg_eta_ptr_[i] = nullptr;
        msg_lam_ptr_[i] = nullptr;
        msg_next_eta_ptr_[i] = nullptr;
        msg_next_lam_ptr_[i] = nullptr;
    }

    const int n = static_cast<int>(adj_var_nodes.size());
    for (int i = 0; i < n; ++i) {
        belief_eta_ptr_[i] = adj_var_nodes[i]->belief.etaData();
        belief_lam_ptr_[i] = adj_var_nodes[i]->belief.lamData();
        msg_eta_ptr_[i] = messages[i].etaData();
        msg_lam_ptr_[i] = messages[i].lamData();
        msg_next_eta_ptr_[i] = messages_next[i].etaData();
        msg_next_lam_ptr_[i] = messages_next[i].lamData();
    }
}

void Factor::swapMessageBuffers_() noexcept {
    messages.swap(messages_next);
    for (int i = 0; i < 2; ++i) {
        std::swap(msg_eta_ptr_[i], msg_next_eta_ptr_[i]);
        std::swap(msg_lam_ptr_[i], msg_next_lam_ptr_[i]);
    }
}

void Factor::initWorkspace_() {
    // Unary factor: computeMessages is just copy factor into messages[0]
    if (is_unary_) return;

    // Binary: allocate all buffers at maximum needed sizes
    const int max_d = (d0_ > d1_) ? d0_ : d1_;

    // factor scratch
    eta_f_.resize(D_);
    lam_f_.resize(D_, D_);

    // Schur scratch
    lnono_.resize(max_d, max_d);
    Y_.resize(max_d, max_d);
    y_.resize(max_d);

    // tmp for msg computations
    tmpLam_.resize(max_d, max_d);
    tmpEta_.resize(max_d);

    // message outputs write directly into messages_next
}

// ==============================
// factor computation
// ==============================
void Factor::invalidateJacobianCache() {
    jcache_valid_ = false;
    lamcache_set_ = false;
    J_cache_.clear();
    JO_cache_.clear();
    lambda_cache_.resize(0, 0);
}

void Factor::computeFactor(const Eigen::VectorXd& linpoint_in, bool update_self) {
    if (&linpoint_in != &linpoint) {
        linpoint = linpoint_in;
    }

    auto pred = meas_fn(linpoint);
    const int D = (int)linpoint.size();

    if (!jcache_valid_) {
        auto J = jac_fn(linpoint);

        if (J.size() != measurement.size() ||
            J.size() != measurement_lambda.size() ||
            J.size() != pred.size()) {
            throw std::runtime_error("computeFactor: block list size mismatch among J, measurement, measurement_lambda, pred");
        }

        J_cache_ = std::move(J);
        JO_cache_.resize(J_cache_.size());

        lambda_cache_.resize(D, D);
        lambda_cache_.setZero();

        for (size_t i = 0; i < J_cache_.size(); ++i) {
            const Eigen::MatrixXd& Ji = J_cache_[i];
            const Eigen::MatrixXd& Oi = measurement_lambda[i];

            JO_cache_[i].resize(Ji.cols(), Oi.cols());
            JO_cache_[i].noalias() = Ji.transpose() * Oi;

            lambda_cache_.noalias() += JO_cache_[i] * Ji;
        }

        jcache_valid_ = true;
        lamcache_set_ = false;
    } else {
        if (J_cache_.size() != measurement.size() ||
            J_cache_.size() != measurement_lambda.size() ||
            J_cache_.size() != pred.size()) {
            throw std::runtime_error("computeFactor(cached): block list size mismatch among cached J, measurement, measurement_lambda, pred");
        }
        if (lambda_cache_.rows() != D || lambda_cache_.cols() != D) {
            jcache_valid_ = false;
            computeFactor(linpoint_in, update_self);
            return;
        }
    }

    eta_f_.resize(D);
    eta_f_.setZero();

    for (size_t i = 0; i < J_cache_.size(); ++i) {
        const Eigen::MatrixXd& Ji = J_cache_[i];
        const Eigen::VectorXd& zi = measurement[i];
        const Eigen::VectorXd& hi = pred[i];

        const int m = (int)zi.size();

        if (ri_cf_.size() < m) ri_cf_.resize(m);
        auto ri = ri_cf_.head(m);
        ri.noalias() = Ji * linpoint;
        ri += zi;
        ri -= hi;

        eta_f_.noalias() += JO_cache_[i] * ri;
    }

    if (update_self) {
        if (!lamcache_set_) {
            factor.setLam(lambda_cache_);
            lamcache_set_ = true;
        }
        factor.setEta(eta_f_);
    }

    if (d0_ == 3 && d1_ == 3 && D_ == 6 && !disable3DFastPath()) {
        eta0_3_ = eta_f_.template segment<3>(0);
        eta1_3_ = eta_f_.template segment<3>(3);

        loo0_3_ = lambda_cache_.template block<3, 3>(0, 0);
        lono0_3_ = lambda_cache_.template block<3, 3>(0, 3);
        lnoo0_3_ = lambda_cache_.template block<3, 3>(3, 0);
        lnono0_3_ = lambda_cache_.template block<3, 3>(3, 3);

        loo1_3_ = lambda_cache_.template block<3, 3>(3, 3);
        lono1_3_ = lambda_cache_.template block<3, 3>(3, 0);
        lnoo1_3_ = lambda_cache_.template block<3, 3>(0, 3);
        lnono1_3_ = lambda_cache_.template block<3, 3>(0, 0);
    }
}

void Factor::setLinearFactorInfo(const Eigen::VectorXd& eta, const Eigen::MatrixXd& lam) {
    if (eta.size() != D_ || lam.rows() != D_ || lam.cols() != D_) {
        throw std::runtime_error("setLinearFactorInfo: eta/lambda dimension mismatch");
    }

    factor.setLam(lam);
    factor.setEta(eta);
    eta_f_ = eta;
    lambda_cache_ = lam;
    lamcache_set_ = true;
    jcache_valid_ = false;
    fixed_lam_valid_ = false;
    fixed_lam_target_valid0_ = false;
    fixed_lam_target_valid1_ = false;
    fixed_lam_target_ldlt0_ = false;
    fixed_lam_target_ldlt1_ = false;
    llt_valid0_ = false;
    llt_valid1_ = false;

    if (d0_ == 3 && d1_ == 3 && D_ == 6) {
        eta0_3_ = eta.template segment<3>(0);
        eta1_3_ = eta.template segment<3>(3);

        loo0_3_ = lam.template block<3, 3>(0, 0);
        lono0_3_ = lam.template block<3, 3>(0, 3);
        lnoo0_3_ = lam.template block<3, 3>(3, 0);
        lnono0_3_ = lam.template block<3, 3>(3, 3);

        loo1_3_ = lam.template block<3, 3>(3, 3);
        lono1_3_ = lam.template block<3, 3>(3, 0);
        lnoo1_3_ = lam.template block<3, 3>(0, 3);
        lnono1_3_ = lam.template block<3, 3>(0, 0);
    }
}

void Factor::clearMessages() {
    fixed_lam_valid_ = false;
    fixed_lam_target_valid0_ = false;
    fixed_lam_target_valid1_ = false;
    fixed_lam_target_ldlt0_ = false;
    fixed_lam_target_ldlt1_ = false;
    for (auto* buf : {&messages, &messages_next}) {
        for (auto& msg : *buf) {
            const int d = msg.dim();
            if (d <= 0) {
                continue;
            }
            std::memset(msg.etaData(), 0, static_cast<size_t>(d) * sizeof(double));
            std::memset(msg.lamData(), 0, static_cast<size_t>(d * d) * sizeof(double));
        }
    }
}

bool Factor::tryExportFixedLamEta6HotEntry(FixedLamEta6HotEntry& out) const noexcept {
    if (!active || !is_binary_ || d0_ != 6 || d1_ != 6 || D_ != 12) {
        return false;
    }
    if (!fixed_lam_valid_ || !fixed_lam_target_valid0_ || !fixed_lam_target_valid1_) {
        return false;
    }
    if (fixedLam6DKernelMode() == 0 ||
        !fixedLam6DInverseCacheEnabled() ||
        !fixedLam6DEtaMapCacheEnabled() ||
        !fixedLam6DInplaceEta()) {
        return false;
    }
    if (belief_eta_ptr_[0] == nullptr || belief_eta_ptr_[1] == nullptr ||
        msg_eta_ptr_[0] == nullptr || msg_eta_ptr_[1] == nullptr) {
        return false;
    }

    out.factor_eta = factor.etaData();
    out.eta_map0 = fixed_lam_eta_map6_0_.data();
    out.eta_map1 = fixed_lam_eta_map6_1_.data();
    out.belief0_eta = belief_eta_ptr_[0];
    out.belief1_eta = belief_eta_ptr_[1];
    out.msg0_eta = msg_eta_ptr_[0];
    out.msg1_eta = msg_eta_ptr_[1];
    return out.factor_eta != nullptr && out.eta_map0 != nullptr && out.eta_map1 != nullptr;
}

// ==============================
// computeMessages (no resize path)
// ==============================

void Factor::computeMessages(double eta_damping) {
    if (!active) return;

    // Unary: trivial
    if (is_unary_) {
        auto& outMsg = messages_next[0];
        outMsg.etaRef().noalias() = factor.eta();
        outMsg.lamRef().noalias() = factor.lam();
        swapMessageBuffers_();
        return;
    }

    const double a = eta_damping;
    const bool no_damping = (eta_damping == 0.0);

    // ------------------------------------------------------------
    // Fast path: 2D-2D binary factor (fixed-size Eigen kernels)
    // ------------------------------------------------------------
    if (d0_ == 2 && d1_ == 2 && D_ == 4) {
        using Vec2 = Eigen::Vector2d;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix2d;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        // Use raw pointers (avoid Map-return overhead in hot path).
        const auto* eta_ptr_f = factor.etaData();
        const auto* lam_ptr_f = factor.lamData();
        const Eigen::Map<const Vec4> eta_f0(eta_ptr_f);
        const Eigen::Map<const Mat4> lam_f0(lam_ptr_f);

        const Eigen::Map<const Vec2> old0_eta(messages[0].etaData());
        const Eigen::Map<const Vec2> old1_eta(messages[1].etaData());
        const Eigen::Map<const Mat2> old0_lam(messages[0].lamData());
        const Eigen::Map<const Mat2> old1_lam(messages[1].lamData());

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.etaData());
        const Eigen::Map<const Vec2> b1_eta(b1.etaData());
        const Eigen::Map<const Mat2> b0_lam(b0.lamData());
        const Eigen::Map<const Mat2> b1_lam(b1.lamData());

        const double s = 1.0 - a;

        // ---------------------
        // target = 0 (to v0, eliminate v1)
        // ---------------------
        {
            const auto eo = eta_f0.template segment<2>(0);
            Vec2 eno = eta_f0.template segment<2>(2);
            eno.noalias() += (b1_eta - old1_eta);

            // --- lam blocks ---
            const auto loo  = lam_f0.template block<2, 2>(0, 0);
            const auto lono = lam_f0.template block<2, 2>(0, 2);
            const auto lnoo = lam_f0.template block<2, 2>(2, 0);

            // lnono: 你要在其上加 (b1_lam-old1_lam) 和 jitter，必须是 owning Mat2（显式拷贝）
            Mat2 lnono = lam_f0.template block<2, 2>(2, 2);
            lnono.noalias() += (b1_lam - old1_lam);
            lnono(1, 0) = lnono(0, 1);
            lnono.diagonal().array() += kJitter;

            llt0_.compute(lnono);
            if (llt0_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (2D fast path, target=0)");
            }
            llt_valid0_ = true;

            const Mat2 Y = llt0_.solve(lnoo);
            const Vec2 y = llt0_.solve(eno);

            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old0_lam;
                outEta2.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            // Aggressive hot path: raw pointer write-back (NO Map construction, NO lamRef invalidation)
            // NOTE: messages' lam is never factorized (no mu()/Sigma() calls), so we do not maintain LLT cache here.
            std::memcpy(outMsg.etaData(), outEta2.data(), 2 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam2.data(), 4 * sizeof(double));
        }

        // ---------------------
        // target = 1 (to v1, eliminate v0)
        // ---------------------
        {
            const auto eo   = eta_f0.template segment<2>(2);
            Vec2 eno = eta_f0.template segment<2>(0);
            eno.noalias() += (b0_eta - old0_eta);

            const auto loo   = lam_f0.template block<2, 2>(2, 2);
            const auto lono  = lam_f0.template block<2, 2>(2, 0);
            const auto lnoo  = lam_f0.template block<2, 2>(0, 2);
            Mat2 lnono       = lam_f0.template block<2, 2>(0, 0);
            lnono.noalias() += (b0_lam - old0_lam);
            lnono(1, 0) = lnono(0, 1);
            lnono.diagonal().array() += kJitter;

            llt1_.compute(lnono);
            if (llt1_.info() != Eigen::Success) {
                throw std::runtime_error("LLT failed in Factor::computeMessages (2D fast path, target=1)");
            }
            llt_valid1_ = true;

            const Mat2 Y = llt1_.solve(lnoo);
            const Vec2 y = llt1_.solve(eno);

            Mat2 outLam2 = loo - lono * Y;
            Vec2 outEta2 = eo  - lono * y;

            if (a != 0.0) {
                outLam2 *= s;
                outEta2 *= s;
                outLam2.noalias() += a * old1_lam;
                outEta2.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            std::memcpy(outMsg.etaData(), outEta2.data(), 2 * sizeof(double));
            std::memcpy(outMsg.lamData(), outLam2.data(), 4 * sizeof(double));
        }

        swapMessageBuffers_();
        return;
    }

    // ------------------------------------------------------------
    // Fast path: 3D-3D binary factor (synthetic SE2 hot path)
    // ------------------------------------------------------------
    if (d0_ == 3 && d1_ == 3 && D_ == 6) {
        const double* old0_eta = msg_eta_ptr_[0];
        const double* old1_eta = msg_eta_ptr_[1];
        const double* old0_lam = msg_lam_ptr_[0];
        const double* old1_lam = msg_lam_ptr_[1];

        const double* b0_eta = belief_eta_ptr_[0];
        const double* b1_eta = belief_eta_ptr_[1];
        const double* b0_lam = belief_lam_ptr_[0];
        const double* b1_lam = belief_lam_ptr_[1];

        {
            const double* eo = eta0_3_.data();
            const double* eno_base = eta1_3_.data();
            const double* loo = loo0_3_.data();
            const double* lono = lono0_3_.data();
            const double* lnoo = lnoo0_3_.data();
            const double* lnono_base = lnono0_3_.data();

            double eno[3] = {
                eno_base[0] + (b1_eta[0] - old1_eta[0]),
                eno_base[1] + (b1_eta[1] - old1_eta[1]),
                eno_base[2] + (b1_eta[2] - old1_eta[2]),
            };
            double lnono_upper[9];
            std::memcpy(lnono_upper, lnono_base, 9 * sizeof(double));
            lnono_upper[0] += (b1_lam[0] - old1_lam[0]) + kJitter;
            lnono_upper[3] += (b1_lam[3] - old1_lam[3]);
            lnono_upper[6] += (b1_lam[6] - old1_lam[6]);
            lnono_upper[4] += (b1_lam[4] - old1_lam[4]) + kJitter;
            lnono_upper[7] += (b1_lam[7] - old1_lam[7]);
            lnono_upper[8] += (b1_lam[8] - old1_lam[8]) + kJitter;
            double* out_eta = msg_next_eta_ptr_[0];
            double* out_lam = msg_next_lam_ptr_[0];
            if (lapackSchurMessage3x3( eo, eno, loo, lono, lnoo, lnono_upper, old0_eta, old0_lam, a, no_damping, out_eta, out_lam)) {
                goto target0_done;
            }
            Cholesky3x3 chol;
            if (!factorizeSpd3x3Upper(
                    lnono_upper[0], lnono_upper[3], lnono_upper[6],
                    lnono_upper[4], lnono_upper[7], lnono_upper[8],
                    chol
                )) {
                Eigen::Matrix3d A = lnono0_3_;
                A(0, 0) += (b1_lam[0] - old1_lam[0]) + kJitter;
                A(0, 1) += (b1_lam[3] - old1_lam[3]);
                A(0, 2) += (b1_lam[6] - old1_lam[6]);
                A(1, 1) += (b1_lam[4] - old1_lam[4]) + kJitter;
                A(1, 2) += (b1_lam[7] - old1_lam[7]);
                A(2, 2) += (b1_lam[8] - old1_lam[8]) + kJitter;
                A(1, 0) = A(0, 1);
                A(2, 0) = A(0, 2);
                A(2, 1) = A(1, 2);
                const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
                Eigen::Matrix3d solved_lam;
                Eigen::Vector3d solved_eta;
                if (!solveGeneral3x3(A, lnoo0_3_, eno_vec, solved_lam, solved_eta)) {
                    throw std::runtime_error("Linear solve failed in Factor::computeMessages (3D fast path, target=0)");
                }
                const Eigen::Vector3d out_eta_vec = eta0_3_ - lono0_3_ * solved_eta;
                const Eigen::Matrix3d out_lam_mat = loo0_3_ - lono0_3_ * solved_lam;
                double* out_eta = msg_next_eta_ptr_[0];
                double* out_lam = msg_next_lam_ptr_[0];
                if (no_damping) {
                    std::memcpy(out_eta, out_eta_vec.data(), 3 * sizeof(double));
                    std::memcpy(out_lam, out_lam_mat.data(), 9 * sizeof(double));
                } else {
                    const double s = 1.0 - a;
                    for (int k = 0; k < 3; ++k) {
                        out_eta[k] = s * out_eta_vec[k] + a * old0_eta[k];
                    }
                    for (int k = 0; k < 9; ++k) {
                        out_lam[k] = s * out_lam_mat.data()[k] + a * old0_lam[k];
                    }
                }
                goto target0_done;
            }

            schurMessage3x3(chol, eo, eno, loo, lono, lnoo, old0_eta, old0_lam, a, no_damping, out_eta, out_lam);
        target0_done:;
        }

        {
            const double* eo = eta1_3_.data();
            const double* eno_base = eta0_3_.data();
            const double* loo = loo1_3_.data();
            const double* lono = lono1_3_.data();
            const double* lnoo = lnoo1_3_.data();
            const double* lnono_base = lnono1_3_.data();

            double eno[3] = {
                eno_base[0] + (b0_eta[0] - old0_eta[0]),
                eno_base[1] + (b0_eta[1] - old0_eta[1]),
                eno_base[2] + (b0_eta[2] - old0_eta[2]),
            };
            double lnono_upper[9];
            std::memcpy(lnono_upper, lnono_base, 9 * sizeof(double));
            lnono_upper[0] += (b0_lam[0] - old0_lam[0]) + kJitter;
            lnono_upper[3] += (b0_lam[3] - old0_lam[3]);
            lnono_upper[6] += (b0_lam[6] - old0_lam[6]);
            lnono_upper[4] += (b0_lam[4] - old0_lam[4]) + kJitter;
            lnono_upper[7] += (b0_lam[7] - old0_lam[7]);
            lnono_upper[8] += (b0_lam[8] - old0_lam[8]) + kJitter;
            double* out_eta = msg_next_eta_ptr_[1];
            double* out_lam = msg_next_lam_ptr_[1];
            if (lapackSchurMessage3x3(eo, eno, loo, lono, lnoo, lnono_upper, old1_eta, old1_lam, a, no_damping, out_eta, out_lam)) {
                goto target1_done;
            }
            Cholesky3x3 chol;
            if (!factorizeSpd3x3Upper(
                    lnono_upper[0], lnono_upper[3], lnono_upper[6],
                    lnono_upper[4], lnono_upper[7], lnono_upper[8],
                    chol
                )) {
                Eigen::Matrix3d A = lnono1_3_;
                A(0, 0) += (b0_lam[0] - old0_lam[0]) + kJitter;
                A(0, 1) += (b0_lam[3] - old0_lam[3]);
                A(0, 2) += (b0_lam[6] - old0_lam[6]);
                A(1, 1) += (b0_lam[4] - old0_lam[4]) + kJitter;
                A(1, 2) += (b0_lam[7] - old0_lam[7]);
                A(2, 2) += (b0_lam[8] - old0_lam[8]) + kJitter;
                A(1, 0) = A(0, 1);
                A(2, 0) = A(0, 2);
                A(2, 1) = A(1, 2);
                const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
                Eigen::Matrix3d solved_lam;
                Eigen::Vector3d solved_eta;
                if (!solveGeneral3x3(A, lnoo1_3_, eno_vec, solved_lam, solved_eta)) {
                    throw std::runtime_error("Linear solve failed in Factor::computeMessages (3D fast path, target=1)");
                }
                const Eigen::Vector3d out_eta_vec = eta1_3_ - lono1_3_ * solved_eta;
                const Eigen::Matrix3d out_lam_mat = loo1_3_ - lono1_3_ * solved_lam;
                double* out_eta = msg_next_eta_ptr_[1];
                double* out_lam = msg_next_lam_ptr_[1];
                if (no_damping) {
                    std::memcpy(out_eta, out_eta_vec.data(), 3 * sizeof(double));
                    std::memcpy(out_lam, out_lam_mat.data(), 9 * sizeof(double));
                } else {
                    const double s = 1.0 - a;
                    for (int k = 0; k < 3; ++k) {
                        out_eta[k] = s * out_eta_vec[k] + a * old1_eta[k];
                    }
                    for (int k = 0; k < 9; ++k) {
                        out_lam[k] = s * out_lam_mat.data()[k] + a * old1_lam[k];
                    }
                }
                goto target1_done;
            }

            schurMessage3x3(chol, eo, eno, loo, lono, lnoo, old1_eta, old1_lam, a, no_damping, out_eta, out_lam);
        target1_done:;
        }

        swapMessageBuffers_();
        return;
    }

    // ------------------------------------------------------------
    // Fast path: 6D-6D binary factor (SE3 full lambda/eta update)
    // ------------------------------------------------------------
    if (is_binary_ && d0_ == 6 && d1_ == 6 && D_ == 12 && full6DKernelEnabled()) {
        using Vec6 = Eigen::Matrix<double, 6, 1>;
        using Vec12 = Eigen::Matrix<double, 12, 1>;
        using Mat6 = Eigen::Matrix<double, 6, 6>;
        using Mat12 = Eigen::Matrix<double, 12, 12>;
        using Mat6x7 = Eigen::Matrix<double, 6, 7>;

        const Eigen::Map<const Vec12> factor_eta(factor.etaData());
        const Eigen::Map<const Mat12> factor_lam(factor.lamData());
        const Eigen::Map<const Vec6> old0_eta(messages[0].etaData());
        const Eigen::Map<const Vec6> old1_eta(messages[1].etaData());
        const Eigen::Map<const Mat6> old0_lam(messages[0].lamData());
        const Eigen::Map<const Mat6> old1_lam(messages[1].lamData());

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec6> b0_eta(b0.etaData());
        const Eigen::Map<const Vec6> b1_eta(b1.etaData());
        const Eigen::Map<const Mat6> b0_lam(b0.lamData());
        const Eigen::Map<const Mat6> b1_lam(b1.lamData());

        const double s = 1.0 - a;
        const bool use_raw_cholesky = full6DRawCholeskyEnabled();

        // target=0: eliminate variable 1.
        {
            utils::NdimGaussian& out_msg = messages_next[0];
            const bool raw_done =
                use_raw_cholesky &&
                schurMessage6Raw(
                    factor.etaData(),
                    factor.lamData(),
                    0,
                    6,
                    b1_eta.data(),
                    b1_lam.data(),
                    old0_eta.data(),
                    old0_lam.data(),
                    old1_eta.data(),
                    old1_lam.data(),
                    a,
                    out_msg.etaData(),
                    out_msg.lamData());

            if (!raw_done) {
                Vec6 eno = factor_eta.template segment<6>(6);
                eno.noalias() += (b1_eta - old1_eta);

                Mat6 cavity = factor_lam.template block<6, 6>(6, 6);
                cavity.noalias() += b1_lam;
                cavity.noalias() -= old1_lam;
                cavity.template triangularView<Eigen::StrictlyLower>() =
                    cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                cavity.diagonal().array() += kJitter;

                double applied_jitter = 0.0;
                bool solved = factorizeGenericSchurCavityWithJitter(
                    cavity, fixed_lam_llt6_0_, applied_jitter);
                bool used_ldlt = false;
                if (!solved && enableGenericLdltFallback()) {
                    fixed_lam_ldlt6_0_.compute(cavity);
                    if (fixed_lam_ldlt6_0_.info() == Eigen::Success) {
                        solved = true;
                        used_ldlt = true;
                    }
                }

                if (!solved) {
                    std::memcpy(out_msg.etaData(), old0_eta.data(), 6 * sizeof(double));
                    std::memcpy(out_msg.lamData(), old0_lam.data(), 36 * sizeof(double));
                } else {
                    Mat6 Y;
                    Vec6 y;
                    if (full6DBatchedSolveEnabled()) {
                        Mat6x7 rhs;
                        rhs.template leftCols<6>().noalias() =
                            factor_lam.template block<6, 6>(6, 0);
                        rhs.col(6).noalias() = eno;
                        Mat6x7 solved_rhs;
                        if (used_ldlt) {
                            solved_rhs.noalias() = fixed_lam_ldlt6_0_.solve(rhs);
                        } else {
                            solved_rhs.noalias() = fixed_lam_llt6_0_.solve(rhs);
                        }
                        Y.noalias() = solved_rhs.template leftCols<6>();
                        y.noalias() = solved_rhs.col(6);
                    } else {
                        if (used_ldlt) {
                            Y.noalias() = fixed_lam_ldlt6_0_.solve(factor_lam.template block<6, 6>(6, 0));
                            y.noalias() = fixed_lam_ldlt6_0_.solve(eno);
                        } else {
                            Y.noalias() = fixed_lam_llt6_0_.solve(factor_lam.template block<6, 6>(6, 0));
                            y.noalias() = fixed_lam_llt6_0_.solve(eno);
                        }
                    }

                    Mat6 out_lam = factor_lam.template block<6, 6>(0, 0);
                    out_lam.noalias() -= factor_lam.template block<6, 6>(0, 6) * Y;
                    Vec6 out_eta = factor_eta.template segment<6>(0);
                    out_eta.noalias() -= factor_lam.template block<6, 6>(0, 6) * y;

                    if (a != 0.0) {
                        out_lam *= s;
                        out_eta *= s;
                        out_lam.noalias() += a * old0_lam;
                        out_eta.noalias() += a * old0_eta;
                    }

                    stabilizeGenericMessageUpdate(
                        out_lam,
                        out_eta,
                        old0_lam,
                        old0_eta,
                        factor_lam.template block<6, 6>(0, 0),
                        factor_eta.template segment<6>(0));
                    std::memcpy(out_msg.etaData(), out_eta.data(), 6 * sizeof(double));
                    std::memcpy(out_msg.lamData(), out_lam.data(), 36 * sizeof(double));
                }
            }
        }

        // target=1: eliminate variable 0.
        {
            utils::NdimGaussian& out_msg = messages_next[1];
            const bool raw_done =
                use_raw_cholesky &&
                schurMessage6Raw(
                    factor.etaData(),
                    factor.lamData(),
                    6,
                    0,
                    b0_eta.data(),
                    b0_lam.data(),
                    old1_eta.data(),
                    old1_lam.data(),
                    old0_eta.data(),
                    old0_lam.data(),
                    a,
                    out_msg.etaData(),
                    out_msg.lamData());

            if (!raw_done) {
                Vec6 eno = factor_eta.template segment<6>(0);
                eno.noalias() += (b0_eta - old0_eta);

                Mat6 cavity = factor_lam.template block<6, 6>(0, 0);
                cavity.noalias() += b0_lam;
                cavity.noalias() -= old0_lam;
                cavity.template triangularView<Eigen::StrictlyLower>() =
                    cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                cavity.diagonal().array() += kJitter;

                double applied_jitter = 0.0;
                bool solved = factorizeGenericSchurCavityWithJitter(
                    cavity, fixed_lam_llt6_1_, applied_jitter);
                bool used_ldlt = false;
                if (!solved && enableGenericLdltFallback()) {
                    fixed_lam_ldlt6_1_.compute(cavity);
                    if (fixed_lam_ldlt6_1_.info() == Eigen::Success) {
                        solved = true;
                        used_ldlt = true;
                    }
                }

                if (!solved) {
                    std::memcpy(out_msg.etaData(), old1_eta.data(), 6 * sizeof(double));
                    std::memcpy(out_msg.lamData(), old1_lam.data(), 36 * sizeof(double));
                } else {
                    Mat6 Y;
                    Vec6 y;
                    if (full6DBatchedSolveEnabled()) {
                        Mat6x7 rhs;
                        rhs.template leftCols<6>().noalias() =
                            factor_lam.template block<6, 6>(0, 6);
                        rhs.col(6).noalias() = eno;
                        Mat6x7 solved_rhs;
                        if (used_ldlt) {
                            solved_rhs.noalias() = fixed_lam_ldlt6_1_.solve(rhs);
                        } else {
                            solved_rhs.noalias() = fixed_lam_llt6_1_.solve(rhs);
                        }
                        Y.noalias() = solved_rhs.template leftCols<6>();
                        y.noalias() = solved_rhs.col(6);
                    } else {
                        if (used_ldlt) {
                            Y.noalias() = fixed_lam_ldlt6_1_.solve(factor_lam.template block<6, 6>(0, 6));
                            y.noalias() = fixed_lam_ldlt6_1_.solve(eno);
                        } else {
                            Y.noalias() = fixed_lam_llt6_1_.solve(factor_lam.template block<6, 6>(0, 6));
                            y.noalias() = fixed_lam_llt6_1_.solve(eno);
                        }
                    }

                    Mat6 out_lam = factor_lam.template block<6, 6>(6, 6);
                    out_lam.noalias() -= factor_lam.template block<6, 6>(6, 0) * Y;
                    Vec6 out_eta = factor_eta.template segment<6>(6);
                    out_eta.noalias() -= factor_lam.template block<6, 6>(6, 0) * y;

                    if (a != 0.0) {
                        out_lam *= s;
                        out_eta *= s;
                        out_lam.noalias() += a * old1_lam;
                        out_eta.noalias() += a * old1_eta;
                    }

                    stabilizeGenericMessageUpdate(
                        out_lam,
                        out_eta,
                        old1_lam,
                        old1_eta,
                        factor_lam.template block<6, 6>(6, 6),
                        factor_eta.template segment<6>(6));
                    std::memcpy(out_msg.etaData(), out_eta.data(), 6 * sizeof(double));
                    std::memcpy(out_msg.lamData(), out_lam.data(), 36 * sizeof(double));
                }
            }
        }

        swapMessageBuffers_();
        return;
    }

    // Old message views (Map) are only needed by the generic / 2D paths.
    const auto old_eta0 = messages[0].eta();
    const auto old_lam0 = messages[0].lam();
    const auto old_eta1 = messages[1].eta();
    const auto old_lam1 = messages[1].lam();

    // Generic path
    for (int target = 0; target < 2; ++target) {
        // 1) eta_f_, lam_f_ = factor + belief_correction (no resize)
        eta_f_.noalias() = factor.eta();
        lam_f_.noalias() = factor.lam();

        if (target == 0) {
            const auto& b1 = adj_var_nodes[1]->belief;
            eta_f_.segment(d0_, d1_).noalias() += (b1.eta() - old_eta1);
            lam_f_.block(d0_, d0_, d1_, d1_).noalias() += (b1.lam() - old_lam1);
        } else {
            const auto& b0 = adj_var_nodes[0]->belief;
            eta_f_.segment(0, d0_).noalias() += (b0.eta() - old_eta0);
            lam_f_.block(0, 0, d0_, d0_).noalias() += (b0.lam() - old_lam0);
        }

        const int d_o  = (target == 0) ? d0_ : d1_;
        const int d_no = (target == 0) ? d1_ : d0_;

        auto eo  = (target == 0) ? eta_f_.segment(0,   d0_) : eta_f_.segment(d0_, d1_);
        auto eno = (target == 0) ? eta_f_.segment(d0_, d1_) : eta_f_.segment(0,   d0_);

        auto loo_view   = (target == 0) ? lam_f_.block(0,   0,   d0_, d0_) : lam_f_.block(d0_, d0_, d1_, d1_);
        auto lono_view  = (target == 0) ? lam_f_.block(0,   d0_, d0_, d1_) : lam_f_.block(d0_, 0,   d1_, d0_);
        auto lnoo_view  = (target == 0) ? lam_f_.block(d0_, 0,   d1_, d0_) : lam_f_.block(0,   d0_, d0_, d1_);
        auto lnono_view = (target == 0) ? lam_f_.block(d0_, d0_, d1_, d1_) : lam_f_.block(0,   0,   d0_, d0_);

        auto lnono = lnono_.topLeftCorner(d_no, d_no);
        lnono.noalias() = lnono_view;
        lnono.template triangularView<Eigen::StrictlyLower>() =
            lnono.transpose().template triangularView<Eigen::StrictlyLower>();
        lnono.diagonal().array() += kJitter;

        utils::NdimGaussian& outMsg = messages_next[target];
        auto outLam = outMsg.lamRef();
        auto outEta = outMsg.etaRef();
        if (d_o == 3 && d_no == 3) {
            double rhs[12];
            for (int col = 0; col < 3; ++col) {
                rhs[3 * col + 0] = lnoo_view(0, col);
                rhs[3 * col + 1] = lnoo_view(1, col);
                rhs[3 * col + 2] = lnoo_view(2, col);
            }
            rhs[9] = eno[0];
            rhs[10] = eno[1];
            rhs[11] = eno[2];
            double solved[12];
            if (!lapackSolveUpperSpd3x3Rhs(lnono.data(), rhs, 4, solved)) {
                throw std::runtime_error("LAPACK solve failed in Factor::computeMessages (generic 3D)");
            }
            Eigen::Map<const Eigen::Matrix<double, 3, 3>> Y3(solved);
            Eigen::Map<const Eigen::Vector3d> y3(solved + 9);
            outLam.noalias() = loo_view;
            outLam.noalias() -= (lono_view * Y3);
            outEta.noalias() = eo;
            outEta.noalias() -= (lono_view * y3);
        } else {
            auto Y = Y_.topLeftCorner(d_no, d_o);
            auto y = y_.head(d_no);
            double applied_jitter = 0.0;
            bool used_ldlt = false;
            Eigen::LDLT<Eigen::MatrixXd> ldlt;

            bool solved = factorizeGenericSchurCavityWithJitter(lnono, llt_, applied_jitter);

            if (solved) {
                Y.noalias() = llt_.solve(lnoo_view);
                y.noalias() = llt_.solve(eno);
            } else if (enableGenericLdltFallback()) {
                ldlt.compute(lnono);
                if (ldlt.info() == Eigen::Success) {
                    Y.noalias() = ldlt.solve(lnoo_view);
                    y.noalias() = ldlt.solve(eno);
                    solved = true;
                    used_ldlt = true;
                }
            }

            if (!solved) {
                if (target == 0) {
                    outLam.noalias() = old_lam0;
                    outEta.noalias() = old_eta0;
                } else {
                    outLam.noalias() = old_lam1;
                    outEta.noalias() = old_eta1;
                }
                continue;
            }

            outLam.noalias() = loo_view;
            outLam.noalias() -= (lono_view * Y);

            outEta.noalias() = eo;
            outEta.noalias() -= (lono_view * y);
            if (used_ldlt) {
                Eigen::MatrixXd outLamT = outLam.transpose();
                outLam = 0.5 * (outLam + outLamT);
            }
        }

        if (a != 0.0) {
            const double ss = 1.0 - a;
            outLam *= ss;
            outEta *= ss;

            if (target == 0) {
                outLam.noalias() += a * old_lam0;
                outEta.noalias() += a * old_eta0;
            } else {
                outLam.noalias() += a * old_lam1;
                outEta.noalias() += a * old_eta1;
            }
        }

        if (target == 0) {
            stabilizeGenericMessageUpdate(outLam, outEta, old_lam0, old_eta0, loo_view, eo);
        } else {
            stabilizeGenericMessageUpdate(outLam, outEta, old_lam1, old_eta1, loo_view, eo);
        }
    }

    swapMessageBuffers_();
}

void Factor::computeMessagesFixedLam(double eta_damping, bool allow_inverse_cache) {
    // 1) one-time init: do full update once
    if (!fixed_lam_valid_) {
        computeMessages(eta_damping);

        // Sync lambdas into ping-pong buffers to avoid stale lambdas after swap
        if (!messages.empty() && messages_next.size() == messages.size()) {
            for (size_t k = 0; k < messages.size(); ++k) {
                messages_next[k].lamRef().noalias() = messages[k].lam();
            }
        }

        fixed_lam_valid_ = true;
        fixed_lam_target_valid0_ = false;
        fixed_lam_target_valid1_ = false;
        fixed_lam_target_ldlt0_ = false;
        fixed_lam_target_ldlt1_ = false;
        return;
    }

    if (!active) return;

    // Unary
    if (is_unary_) {
        auto& outMsg = messages_next[0];
        outMsg.etaRef().noalias() = factor.eta();
        outMsg.lamRef().noalias() = factor.lam();
        swapMessageBuffers_();
        return;
    }

    // Old messages (avoid Map-return overhead in the hot path)
    const auto* old_eta0_ptr = messages[0].etaData();
    const auto* old_eta1_ptr = messages[1].etaData();
    const auto* old_lam0_ptr = messages[0].lamData();
    const auto* old_lam1_ptr = messages[1].lamData();

    const double a = eta_damping;
    const int fixed_lam_6d_mode = fixedLam6DKernelMode();

    if (is_binary_ && d0_ == 6 && d1_ == 6 && D_ == 12 && fixed_lam_6d_mode != 0) {
        using Vec6 = Eigen::Matrix<double, 6, 1>;
        using Vec12 = Eigen::Matrix<double, 12, 1>;
        using Mat6 = Eigen::Matrix<double, 6, 6>;
        using Mat12 = Eigen::Matrix<double, 12, 12>;
        const bool use_dynamic_llt = (fixed_lam_6d_mode == 2);
        const bool use_inverse_cache =
            allow_inverse_cache && fixedLam6DInverseCacheEnabled() && !use_dynamic_llt;
        const bool use_eta_map_cache =
            use_inverse_cache && fixedLam6DEtaMapCacheEnabled();
        const bool use_inplace_eta = fixedLam6DInplaceEta();

        if (use_inverse_cache &&
            use_inplace_eta &&
            fixed_lam_target_valid0_ &&
            fixed_lam_target_valid1_) {
            if (use_eta_map_cache) {
                fixedLambdaEta6FromCachedEtaMapRaw(
                    factor.etaData(),
                    0,
                    6,
                    fixed_lam_eta_map6_0_.data(),
                    belief_eta_ptr_[1],
                    old_eta1_ptr,
                    old_eta0_ptr,
                    a,
                    messages[0].etaData());
                fixedLambdaEta6FromCachedEtaMapRaw(
                    factor.etaData(),
                    6,
                    0,
                    fixed_lam_eta_map6_1_.data(),
                    belief_eta_ptr_[0],
                    old_eta0_ptr,
                    old_eta1_ptr,
                    a,
                    messages[1].etaData());
            } else {
                fixedLambdaEta6FromCachedInverseRaw(
                    factor.etaData(),
                    factor.lamData(),
                    0,
                    6,
                    fixed_lam_inv6_0_.data(),
                    belief_eta_ptr_[1],
                    old_eta1_ptr,
                    old_eta0_ptr,
                    a,
                    messages[0].etaData());
                fixedLambdaEta6FromCachedInverseRaw(
                    factor.etaData(),
                    factor.lamData(),
                    6,
                    0,
                    fixed_lam_inv6_1_.data(),
                    belief_eta_ptr_[0],
                    old_eta0_ptr,
                    old_eta1_ptr,
                    a,
                    messages[1].etaData());
            }
            return;
        }

        const Eigen::Map<const Vec12> factor_eta(factor.etaData());
        const Eigen::Map<const Mat12> factor_lam(factor.lamData());
        const Vec6 old0_eta = Eigen::Map<const Vec6>(old_eta0_ptr);
        const Vec6 old1_eta = Eigen::Map<const Vec6>(old_eta1_ptr);
        const Eigen::Map<const Mat6> old0_lam(old_lam0_ptr);
        const Eigen::Map<const Mat6> old1_lam(old_lam1_ptr);

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec6> b0_eta(b0.etaData());
        const Eigen::Map<const Vec6> b1_eta(b1.etaData());
        const Eigen::Map<const Mat6> b0_lam(b0.lamData());
        const Eigen::Map<const Mat6> b1_lam(b1.lamData());

        // target=0: eliminate variable 1.
        {
            bool solved = true;
            if (!fixed_lam_target_valid0_) {
                double applied_jitter = 0.0;
                fixed_lam_target_ldlt0_ = false;
                if (use_dynamic_llt) {
                    auto cavity = lnono_.topLeftCorner(6, 6);
                    cavity.noalias() = factor_lam.template block<6, 6>(6, 6);
                    cavity.noalias() += b1_lam;
                    cavity.noalias() -= old1_lam;
                    cavity.template triangularView<Eigen::StrictlyLower>() =
                        cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                    cavity.diagonal().array() += kJitter;

                    solved = factorizeGenericSchurCavityWithJitter(
                        cavity, fixed_lam_llt0_, applied_jitter);
                    if (!solved && enableGenericLdltFallback()) {
                        fixed_lam_ldlt0_.compute(cavity);
                        if (fixed_lam_ldlt0_.info() == Eigen::Success) {
                            solved = true;
                            fixed_lam_target_ldlt0_ = true;
                        }
                    }
                } else {
                    Mat6 cavity = factor_lam.template block<6, 6>(6, 6);
                    cavity.noalias() += b1_lam;
                    cavity.noalias() -= old1_lam;
                    cavity.template triangularView<Eigen::StrictlyLower>() =
                        cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                    cavity.diagonal().array() += kJitter;

                    solved = factorizeGenericSchurCavityWithJitter(
                        cavity, fixed_lam_llt6_0_, applied_jitter);
                    if (!solved && enableGenericLdltFallback()) {
                        fixed_lam_ldlt6_0_.compute(cavity);
                        if (fixed_lam_ldlt6_0_.info() == Eigen::Success) {
                            solved = true;
                            fixed_lam_target_ldlt0_ = true;
                        }
                    }
                    if (solved && use_inverse_cache) {
                        const Mat6 I = Mat6::Identity();
                        if (fixed_lam_target_ldlt0_) {
                            fixed_lam_inv6_0_.noalias() = fixed_lam_ldlt6_0_.solve(I);
                        } else {
                            fixed_lam_inv6_0_.noalias() = fixed_lam_llt6_0_.solve(I);
                        }
                        if (use_eta_map_cache) {
                            fixed_lam_eta_map6_0_.noalias() =
                                factor_lam.template block<6, 6>(0, 6) * fixed_lam_inv6_0_;
                        }
                    }
                }
                fixed_lam_target_valid0_ = solved;
            }

            utils::NdimGaussian& out_msg = use_inplace_eta ? messages[0] : messages_next[0];
            if (!fixed_lam_target_valid0_) {
                std::memcpy(out_msg.etaData(), old_eta0_ptr, 6 * sizeof(double));
            } else if (use_inverse_cache) {
                if (use_eta_map_cache) {
                    fixedLambdaEta6FromCachedEtaMapRaw(
                        factor_eta.data(),
                        0,
                        6,
                        fixed_lam_eta_map6_0_.data(),
                        b1_eta.data(),
                        old1_eta.data(),
                        old0_eta.data(),
                        a,
                        out_msg.etaData());
                } else {
                    fixedLambdaEta6FromCachedInverseRaw(
                        factor_eta.data(),
                        factor_lam.data(),
                        0,
                        6,
                        fixed_lam_inv6_0_.data(),
                        b1_eta.data(),
                        old1_eta.data(),
                        old0_eta.data(),
                        a,
                        out_msg.etaData());
                }
            } else {
                Vec6 eno = factor_eta.template segment<6>(6);
                eno.noalias() += (b1_eta - old1_eta);
                Vec6 y;
                if (use_dynamic_llt) {
                    if (fixed_lam_target_ldlt0_) {
                        y.noalias() = fixed_lam_ldlt0_.solve(eno);
                    } else {
                        y.noalias() = fixed_lam_llt0_.solve(eno);
                    }
                } else {
                    if (fixed_lam_target_ldlt0_) {
                        y.noalias() = fixed_lam_ldlt6_0_.solve(eno);
                    } else {
                        y.noalias() = fixed_lam_llt6_0_.solve(eno);
                    }
                }
                Eigen::Map<Vec6> out_eta(out_msg.etaData());
                out_eta.noalias() = factor_eta.template segment<6>(0);
                out_eta.noalias() -= factor_lam.template block<6, 6>(0, 6) * y;
                if (a != 0.0) {
                    out_eta *= (1.0 - a);
                    out_eta.noalias() += a * old0_eta;
                }
                stabilizeGenericEtaUpdate(
                    out_eta, old0_eta, factor_eta.template segment<6>(0));
            }
        }

        // target=1: eliminate variable 0.
        {
            bool solved = true;
            if (!fixed_lam_target_valid1_) {
                double applied_jitter = 0.0;
                fixed_lam_target_ldlt1_ = false;
                if (use_dynamic_llt) {
                    auto cavity = lnono_.topLeftCorner(6, 6);
                    cavity.noalias() = factor_lam.template block<6, 6>(0, 0);
                    cavity.noalias() += b0_lam;
                    cavity.noalias() -= old0_lam;
                    cavity.template triangularView<Eigen::StrictlyLower>() =
                        cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                    cavity.diagonal().array() += kJitter;

                    solved = factorizeGenericSchurCavityWithJitter(
                        cavity, fixed_lam_llt1_, applied_jitter);
                    if (!solved && enableGenericLdltFallback()) {
                        fixed_lam_ldlt1_.compute(cavity);
                        if (fixed_lam_ldlt1_.info() == Eigen::Success) {
                            solved = true;
                            fixed_lam_target_ldlt1_ = true;
                        }
                    }
                } else {
                    Mat6 cavity = factor_lam.template block<6, 6>(0, 0);
                    cavity.noalias() += b0_lam;
                    cavity.noalias() -= old0_lam;
                    cavity.template triangularView<Eigen::StrictlyLower>() =
                        cavity.transpose().template triangularView<Eigen::StrictlyLower>();
                    cavity.diagonal().array() += kJitter;

                    solved = factorizeGenericSchurCavityWithJitter(
                        cavity, fixed_lam_llt6_1_, applied_jitter);
                    if (!solved && enableGenericLdltFallback()) {
                        fixed_lam_ldlt6_1_.compute(cavity);
                        if (fixed_lam_ldlt6_1_.info() == Eigen::Success) {
                            solved = true;
                            fixed_lam_target_ldlt1_ = true;
                        }
                    }
                    if (solved && use_inverse_cache) {
                        const Mat6 I = Mat6::Identity();
                        if (fixed_lam_target_ldlt1_) {
                            fixed_lam_inv6_1_.noalias() = fixed_lam_ldlt6_1_.solve(I);
                        } else {
                            fixed_lam_inv6_1_.noalias() = fixed_lam_llt6_1_.solve(I);
                        }
                        if (use_eta_map_cache) {
                            fixed_lam_eta_map6_1_.noalias() =
                                factor_lam.template block<6, 6>(6, 0) * fixed_lam_inv6_1_;
                        }
                    }
                }
                fixed_lam_target_valid1_ = solved;
            }

            utils::NdimGaussian& out_msg = use_inplace_eta ? messages[1] : messages_next[1];
            if (!fixed_lam_target_valid1_) {
                std::memcpy(out_msg.etaData(), old_eta1_ptr, 6 * sizeof(double));
            } else if (use_inverse_cache) {
                if (use_eta_map_cache) {
                    fixedLambdaEta6FromCachedEtaMapRaw(
                        factor_eta.data(),
                        6,
                        0,
                        fixed_lam_eta_map6_1_.data(),
                        b0_eta.data(),
                        old0_eta.data(),
                        old1_eta.data(),
                        a,
                        out_msg.etaData());
                } else {
                    fixedLambdaEta6FromCachedInverseRaw(
                        factor_eta.data(),
                        factor_lam.data(),
                        6,
                        0,
                        fixed_lam_inv6_1_.data(),
                        b0_eta.data(),
                        old0_eta.data(),
                        old1_eta.data(),
                        a,
                        out_msg.etaData());
                }
            } else {
                Vec6 eno = factor_eta.template segment<6>(0);
                eno.noalias() += (b0_eta - old0_eta);
                Vec6 y;
                if (use_dynamic_llt) {
                    if (fixed_lam_target_ldlt1_) {
                        y.noalias() = fixed_lam_ldlt1_.solve(eno);
                    } else {
                        y.noalias() = fixed_lam_llt1_.solve(eno);
                    }
                } else {
                    if (fixed_lam_target_ldlt1_) {
                        y.noalias() = fixed_lam_ldlt6_1_.solve(eno);
                    } else {
                        y.noalias() = fixed_lam_llt6_1_.solve(eno);
                    }
                }
                Eigen::Map<Vec6> out_eta(out_msg.etaData());
                out_eta.noalias() = factor_eta.template segment<6>(6);
                out_eta.noalias() -= factor_lam.template block<6, 6>(6, 0) * y;
                if (a != 0.0) {
                    out_eta *= (1.0 - a);
                    out_eta.noalias() += a * old1_eta;
                }
                stabilizeGenericEtaUpdate(
                    out_eta, old1_eta, factor_eta.template segment<6>(6));
            }
        }

        if (!use_inplace_eta) {
            swapMessageBuffers_();
        }
        return;
    }

    if (!(d0_ == 2 && d1_ == 2 && D_ == 4)) {
        if (!is_binary_) {
            computeMessages(eta_damping);
            return;
        }

        const auto factor_eta = factor.eta();
        const auto factor_lam = factor.lam();
        const auto old_eta0 = messages[0].eta();
        const auto old_lam0 = messages[0].lam();
        const auto old_eta1 = messages[1].eta();
        const auto old_lam1 = messages[1].lam();

        for (int target = 0; target < 2; ++target) {
            const int off_o = (target == 0) ? 0 : d0_;
            const int off_no = (target == 0) ? d0_ : 0;
            const int d_o = (target == 0) ? d0_ : d1_;
            const int d_no = (target == 0) ? d1_ : d0_;

            const auto old_eta_target = (target == 0) ? old_eta0 : old_eta1;
            const auto old_lam_target = (target == 0) ? old_lam0 : old_lam1;
            const auto old_eta_other = (target == 0) ? old_eta1 : old_eta0;
            const auto old_lam_other = (target == 0) ? old_lam1 : old_lam0;
            const utils::NdimGaussian& other_belief =
                (target == 0) ? adj_var_nodes[1]->belief : adj_var_nodes[0]->belief;
            bool& cache_valid =
                (target == 0) ? fixed_lam_target_valid0_ : fixed_lam_target_valid1_;
            bool& cache_uses_ldlt =
                (target == 0) ? fixed_lam_target_ldlt0_ : fixed_lam_target_ldlt1_;
            auto& cache_llt = (target == 0) ? fixed_lam_llt0_ : fixed_lam_llt1_;
            auto& cache_ldlt = (target == 0) ? fixed_lam_ldlt0_ : fixed_lam_ldlt1_;

            auto eno = tmpEta_.head(d_no);
            eno.noalias() = factor_eta.segment(off_no, d_no);
            eno.noalias() += (other_belief.eta() - old_eta_other);

            auto y = y_.head(d_no);
            bool solved = true;
            if (!cache_valid) {
                auto lnono = lnono_.topLeftCorner(d_no, d_no);
                lnono.noalias() = factor_lam.block(off_no, off_no, d_no, d_no);
                lnono.noalias() += other_belief.lam();
                lnono.noalias() -= old_lam_other;
                lnono.template triangularView<Eigen::StrictlyLower>() =
                    lnono.transpose().template triangularView<Eigen::StrictlyLower>();
                lnono.diagonal().array() += kJitter;

                double applied_jitter = 0.0;
                solved = factorizeGenericSchurCavityWithJitter(lnono, cache_llt, applied_jitter);
                cache_uses_ldlt = false;
                if (!solved && enableGenericLdltFallback()) {
                    cache_ldlt.compute(lnono);
                    if (cache_ldlt.info() == Eigen::Success) {
                        solved = true;
                        cache_uses_ldlt = true;
                    }
                }
                cache_valid = solved;
            }

            if (cache_valid) {
                if (cache_uses_ldlt) {
                    y.noalias() = cache_ldlt.solve(eno);
                } else {
                    y.noalias() = cache_llt.solve(eno);
                }
            } else {
                solved = false;
            }

            utils::NdimGaussian& outMsg = messages_next[static_cast<size_t>(target)];
            auto outEta = outMsg.etaRef();
            if (!solved) {
                outEta.noalias() = old_eta_target;
                continue;
            }

            outEta.noalias() = factor_eta.segment(off_o, d_o);
            outEta.noalias() -= factor_lam.block(off_o, off_no, d_o, d_no) * y;
            if (a != 0.0) {
                outEta *= (1.0 - a);
                outEta.noalias() += a * old_eta_target;
            }
            stabilizeGenericEtaUpdate(outEta, old_eta_target, factor_eta.segment(off_o, d_o));
        }
        swapMessageBuffers_();
        return;
    }

    // Fixed-lam fast path: 2D-2D
    {
        using Vec2 = Eigen::Vector2d;
        using Vec4 = Eigen::Matrix<double, 4, 1>;
        using Mat2 = Eigen::Matrix2d;
        using Mat4 = Eigen::Matrix<double, 4, 4>;

        // Factor blocks via raw pointers
        const auto* eta_ptr_f = factor.etaData();
        const auto* lam_ptr_f = factor.lamData();
        const Eigen::Map<const Vec4> eta_f0(eta_ptr_f);
        const Eigen::Map<const Mat4> lam_f0(lam_ptr_f);

        // Old messages (maps over raw pointers)
        const Eigen::Map<const Vec2> old0_eta(old_eta0_ptr);
        const Eigen::Map<const Vec2> old1_eta(old_eta1_ptr);
        const Eigen::Map<const Mat2> old0_lam(old_lam0_ptr);
        const Eigen::Map<const Mat2> old1_lam(old_lam1_ptr);

        const auto& b0 = adj_var_nodes[0]->belief;
        const auto& b1 = adj_var_nodes[1]->belief;
        const Eigen::Map<const Vec2> b0_eta(b0.etaData());
        const Eigen::Map<const Vec2> b1_eta(b1.etaData());

        // target=0
        {
            const auto eo  = eta_f0.template segment<2>(0);
            //const Vec2 eno = eta_f0.template segment<2>(2) + (b1_eta - old1_eta);
            Vec2 eno = eta_f0.template segment<2>(2);
            eno.noalias() += (b1_eta - old1_eta);
            const auto lono = lam_f0.template block<2, 2>(0, 2);

            // llt0_ was factorized from l_no,no in computeFactor()
            const Vec2 y = llt0_.solve(eno);

            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old0_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[0];
            double* p = outMsg.etaData();
            p[0] = outEta2[0];
            p[1] = outEta2[1];
        }

        // target=1
        {
            const auto eo  = eta_f0.template segment<2>(2);
            Vec2 eno = eta_f0.template segment<2>(0);
            eno.noalias() += (b0_eta - old0_eta);
            const auto lono = lam_f0.template block<2, 2>(2, 0);

            const Vec2 y = llt1_.solve(eno);

            Vec2 outEta2 = eo - lono * y;

            if (a != 0.0) {
                outEta2 *= (1.0 - a);
                outEta2.noalias() += a * old1_eta;
            }

            utils::NdimGaussian& outMsg = messages_next[1];
            double* p = outMsg.etaData();
            p[0] = outEta2[0];
            p[1] = outEta2[1];
        }

        swapMessageBuffers_();
        return;
    }
}

} // namespace gbp
