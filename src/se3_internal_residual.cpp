#include "internal/se3_residual.h"
#include "internal/scoped_worker_affinity.h"
#include "internal/se3_schur_solve.h"
#include "internal/finite_double.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include <omp.h>

namespace slam {

namespace {

using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat6x7 = Eigen::Matrix<double, 6, 7>;
using Vec12 = Eigen::Matrix<double, 12, 1>;
using Mat12 = Eigen::Matrix<double, 12, 12>;
using AlignedDoubles = SyntheticSE3PackedSoAWorkspace::AlignedDoubles;
using SteadyClock = std::chrono::steady_clock;

#if defined(_MSC_VER)
#define SE3_PACKED_FORCEINLINE __forceinline
#define SE3_PACKED_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define SE3_PACKED_FORCEINLINE inline __attribute__((always_inline))
#define SE3_PACKED_RESTRICT __restrict__
#else
#define SE3_PACKED_FORCEINLINE inline
#define SE3_PACKED_RESTRICT
#endif

constexpr double kJitter = 1e-10;
constexpr double kSym21FroWeights[21] = {
    1.0,
    2.0, 1.0,
    2.0, 2.0, 1.0,
    2.0, 2.0, 2.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 1.0,
    2.0, 2.0, 2.0, 2.0, 2.0, 1.0
};

bool packedSoAHotFixedEtaKernelEnabled(int thread_count) {
    return thread_count == 1;
}

bool packedSoAL21CholeskyEnabled(int thread_count) {
    return thread_count == 1;
}

SE3_PACKED_FORCEINLINE double elapsedSeconds(
    const SteadyClock::time_point& start,
    const SteadyClock::time_point& end
) {
    return std::chrono::duration<double>(end - start).count();
}

SE3_PACKED_FORCEINLINE int effectiveThreadCount(int requested_threads) noexcept {
    return std::max(1, (requested_threads > 0) ? requested_threads : omp_get_max_threads());
}

SE3_PACKED_FORCEINLINE int sym21Index(int row, int col) noexcept {
    return (row <= col) ? (col * (col + 1)) / 2 + row : (row * (row + 1)) / 2 + col;
}

SE3_PACKED_FORCEINLINE double* vec6Ptr(AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

SE3_PACKED_FORCEINLINE const double* vec6Ptr(const AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

SE3_PACKED_FORCEINLINE double* sym21Ptr(AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 21;
}

SE3_PACKED_FORCEINLINE const double* sym21Ptr(const AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 21;
}

SE3_PACKED_FORCEINLINE double* mat36Ptr(AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 36;
}

SE3_PACKED_FORCEINLINE const double* mat36Ptr(const AlignedDoubles& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 36;
}

SE3_PACKED_FORCEINLINE double* binaryMsgEtaPtr(SyntheticSE3PackedSoAWorkspace& w, int slot) noexcept {
    return w.binary_msg_eta.data() + static_cast<size_t>(slot) * 6;
}

SE3_PACKED_FORCEINLINE const double* binaryMsgEtaPtr(const SyntheticSE3PackedSoAWorkspace& w, int slot) noexcept {
    return w.binary_msg_eta.data() + static_cast<size_t>(slot) * 6;
}

SE3_PACKED_FORCEINLINE double* binaryMsgLam21Ptr(SyntheticSE3PackedSoAWorkspace& w, int slot) noexcept {
    return w.binary_msg_lam21.data() + static_cast<size_t>(slot) * 21;
}

SE3_PACKED_FORCEINLINE const double* binaryMsgLam21Ptr(const SyntheticSE3PackedSoAWorkspace& w, int slot) noexcept {
    return w.binary_msg_lam21.data() + static_cast<size_t>(slot) * 21;
}

SE3_PACKED_FORCEINLINE void zero6(double* out) noexcept {
    for (int i = 0; i < 6; ++i) out[i] = 0.0;
}

SE3_PACKED_FORCEINLINE void copy6(const double* src, double* dst) noexcept {
    std::memcpy(dst, src, 6 * sizeof(double));
}

SE3_PACKED_FORCEINLINE void copy21(const double* src, double* dst) noexcept {
    std::memcpy(dst, src, 21 * sizeof(double));
}

void sym21FromFullColMajor(const double* full36, double* sym21) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row <= col; ++row) {
            const double a = full36[row + col * 6];
            const double b = full36[col + row * 6];
            sym21[sym21Index(row, col)] = 0.5 * (a + b);
        }
    }
}

void sym21FromUpperColMajor(const double* full36, double* sym21) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row <= col; ++row) {
            sym21[sym21Index(row, col)] = full36[row + col * 6];
        }
    }
}

void fullColMajorFromSym21(const double* sym21, double* full36) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            full36[row + col * 6] = sym21[sym21Index(row, col)];
        }
    }
}

void mat6FromSym21(const double* sym21, Mat6& out) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            out(row, col) = sym21[sym21Index(row, col)];
        }
    }
}

void sym21FromMat6(const Mat6& mat, double* sym21) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row <= col; ++row) {
            sym21[sym21Index(row, col)] = 0.5 * (mat(row, col) + mat(col, row));
        }
    }
}

void cavityFromSym21(
    const double* diag,
    const double* belief,
    const double* old_msg,
    Mat6& out
) noexcept {
    for (int col = 0; col < 6; ++col) {
        for (int row = 0; row < 6; ++row) {
            const int idx = sym21Index(row, col);
            out(row, col) = diag[idx] + belief[idx] - old_msg[idx];
        }
    }
    out.diagonal().array() += kJitter;
}

SE3_PACKED_FORCEINLINE double sqNorm6(const double* x) noexcept {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] +
           x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
}

SE3_PACKED_FORCEINLINE void sym21MatVec6Raw(
    const double* lam,
    const double* x,
    double* y
) noexcept {
    y[0] = lam[0] * x[0] + lam[1] * x[1] + lam[3] * x[2] +
           lam[6] * x[3] + lam[10] * x[4] + lam[15] * x[5];
    y[1] = lam[1] * x[0] + lam[2] * x[1] + lam[4] * x[2] +
           lam[7] * x[3] + lam[11] * x[4] + lam[16] * x[5];
    y[2] = lam[3] * x[0] + lam[4] * x[1] + lam[5] * x[2] +
           lam[8] * x[3] + lam[12] * x[4] + lam[17] * x[5];
    y[3] = lam[6] * x[0] + lam[7] * x[1] + lam[8] * x[2] +
           lam[9] * x[3] + lam[13] * x[4] + lam[18] * x[5];
    y[4] = lam[10] * x[0] + lam[11] * x[1] + lam[12] * x[2] +
           lam[13] * x[3] + lam[14] * x[4] + lam[19] * x[5];
    y[5] = lam[15] * x[0] + lam[16] * x[1] + lam[17] * x[2] +
           lam[18] * x[3] + lam[19] * x[4] + lam[20] * x[5];
}

double froNormSym21(const double* sym21) noexcept {
    double sum = 0.0;
    for (int i = 0; i < 21; ++i) {
        const double v = sym21[i];
        sum += kSym21FroWeights[i] * v * v;
    }
    return std::sqrt(sum);
}

SE3_PACKED_FORCEINLINE double froDiffSqSym21Ordered(
    const double* a,
    const double* b
) noexcept {
    double sum = 0.0;
    for (int i = 0; i < 21; ++i) {
        const double d = a[i] - b[i];
        sum += kSym21FroWeights[i] * d * d;
    }
    return sum;
}

SE3_PACKED_FORCEINLINE double cavityEntrySym21Raw(
    const double* diag,
    const double* belief,
    const double* old_msg,
    int idx
) noexcept {
    return diag[idx] + belief[idx] - old_msg[idx];
}

SE3_PACKED_FORCEINLINE void cavityFullColMajorFromSym21Raw(
    const double* diag,
    const double* belief,
    const double* old_msg,
    double* out
) noexcept {
    const double c00 = cavityEntrySym21Raw(diag, belief, old_msg, 0) + kJitter;
    const double c01 = cavityEntrySym21Raw(diag, belief, old_msg, 1);
    const double c11 = cavityEntrySym21Raw(diag, belief, old_msg, 2) + kJitter;
    const double c02 = cavityEntrySym21Raw(diag, belief, old_msg, 3);
    const double c12 = cavityEntrySym21Raw(diag, belief, old_msg, 4);
    const double c22 = cavityEntrySym21Raw(diag, belief, old_msg, 5) + kJitter;
    const double c03 = cavityEntrySym21Raw(diag, belief, old_msg, 6);
    const double c13 = cavityEntrySym21Raw(diag, belief, old_msg, 7);
    const double c23 = cavityEntrySym21Raw(diag, belief, old_msg, 8);
    const double c33 = cavityEntrySym21Raw(diag, belief, old_msg, 9) + kJitter;
    const double c04 = cavityEntrySym21Raw(diag, belief, old_msg, 10);
    const double c14 = cavityEntrySym21Raw(diag, belief, old_msg, 11);
    const double c24 = cavityEntrySym21Raw(diag, belief, old_msg, 12);
    const double c34 = cavityEntrySym21Raw(diag, belief, old_msg, 13);
    const double c44 = cavityEntrySym21Raw(diag, belief, old_msg, 14) + kJitter;
    const double c05 = cavityEntrySym21Raw(diag, belief, old_msg, 15);
    const double c15 = cavityEntrySym21Raw(diag, belief, old_msg, 16);
    const double c25 = cavityEntrySym21Raw(diag, belief, old_msg, 17);
    const double c35 = cavityEntrySym21Raw(diag, belief, old_msg, 18);
    const double c45 = cavityEntrySym21Raw(diag, belief, old_msg, 19);
    const double c55 = cavityEntrySym21Raw(diag, belief, old_msg, 20) + kJitter;

    out[0] = c00; out[1] = c01; out[2] = c02; out[3] = c03; out[4] = c04; out[5] = c05;
    out[6] = c01; out[7] = c11; out[8] = c12; out[9] = c13; out[10] = c14; out[11] = c15;
    out[12] = c02; out[13] = c12; out[14] = c22; out[15] = c23; out[16] = c24; out[17] = c25;
    out[18] = c03; out[19] = c13; out[20] = c23; out[21] = c33; out[22] = c34; out[23] = c35;
    out[24] = c04; out[25] = c14; out[26] = c24; out[27] = c34; out[28] = c44; out[29] = c45;
    out[30] = c05; out[31] = c15; out[32] = c25; out[33] = c35; out[34] = c45; out[35] = c55;
}

SE3_PACKED_FORCEINLINE void damp6(const double* old_eta, double damping, double* out_eta) noexcept {
    if (damping == 0.0) return;
    const double keep_new = 1.0 - damping;
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = keep_new * out_eta[i] + damping * old_eta[i];
    }
}

void damp21(const double* old_lam, double damping, double* out_lam) noexcept {
    if (damping == 0.0) return;
    const double keep_new = 1.0 - damping;
    for (int i = 0; i < 21; ++i) {
        out_lam[i] = keep_new * out_lam[i] + damping * old_lam[i];
    }
}

SE3_PACKED_FORCEINLINE void stabilizeEta6(
    const double* old_eta,
    const double* ref_eta,
    double max_rel_update,
    double* out_eta
) noexcept {
    if (!(max_rel_update > 0.0)) return;
    const double eta_ref = std::max(1.0, std::max(std::sqrt(sqNorm6(old_eta)), std::sqrt(sqNorm6(ref_eta))));
    double diff_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double d = out_eta[i] - old_eta[i];
        diff_sq += d * d;
    }
    const double max_scaled = max_rel_update * eta_ref;
    if (!(diff_sq > max_scaled * max_scaled)) return;
    const double rel = std::sqrt(diff_sq) / eta_ref;
    if (!finiteDouble(rel) || rel <= max_rel_update) return;
    const double step = max_rel_update / std::max(rel, 1e-300);
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
}

void stabilizeMessage6(
    const double* old_eta,
    const double* old_lam,
    const double* ref_eta,
    const double* ref_lam,
    double max_rel_update,
    double* out_eta,
    double* out_lam,
    const double* cached_reference_norms = nullptr
) noexcept {
    if (!(max_rel_update > 0.0)) return;
    const double ref_lam_norm = cached_reference_norms ? cached_reference_norms[0] : froNormSym21(ref_lam);
    const double ref_eta_norm = cached_reference_norms ? cached_reference_norms[1] : std::sqrt(sqNorm6(ref_eta));
    const double lam_ref = std::max(1.0, std::max(froNormSym21(old_lam), ref_lam_norm));
    const double eta_ref = std::max(1.0, std::max(std::sqrt(sqNorm6(old_eta)), ref_eta_norm));
    double eta_diff_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double d = out_eta[i] - old_eta[i];
        eta_diff_sq += d * d;
    }
    const double lam_diff_sq = froDiffSqSym21Ordered(out_lam, old_lam);
    const double rel_eta = std::sqrt(eta_diff_sq) / eta_ref;
    const double rel_lam = std::sqrt(lam_diff_sq) / lam_ref;
    const double rel = std::sqrt(rel_eta * rel_eta + rel_lam * rel_lam);
    if (!finiteDouble(rel) || rel <= max_rel_update) return;
    const double step = max_rel_update / std::max(rel, 1e-300);
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
    for (int i = 0; i < 21; ++i) {
        out_lam[i] = old_lam[i] + step * (out_lam[i] - old_lam[i]);
    }
}

Mat6 solveMatWithFallback(const Mat6& A, const Mat6& B) {
    Eigen::LLT<Mat6, Eigen::Upper> llt(A);
    if (llt.info() == Eigen::Success) return llt.solve(B);
    Eigen::LDLT<Mat6> ldlt(A);
    if (ldlt.info() == Eigen::Success) return ldlt.solve(B);
    return A.fullPivLu().solve(B);
}

Vec6 solveVecWithFallback(const Mat6& A, const Vec6& b) {
    Eigen::LLT<Mat6, Eigen::Upper> llt(A);
    if (llt.info() == Eigen::Success) return llt.solve(b);
    Eigen::LDLT<Mat6> ldlt(A);
    if (ldlt.info() == Eigen::Success) return ldlt.solve(b);
    return A.fullPivLu().solve(b);
}

Mat6x7 solveMat6x7WithFallback(const Mat6& A, const Mat6x7& B) {
    Eigen::LLT<Mat6, Eigen::Upper> llt(A);
    if (llt.info() == Eigen::Success) return llt.solve(B);
    Eigen::LDLT<Mat6> ldlt(A);
    if (ldlt.info() == Eigen::Success) return ldlt.solve(B);
    return A.fullPivLu().solve(B);
}

SE3_PACKED_FORCEINLINE bool factorizeSpd6LowerRaw(const double* a, double* l) noexcept {
    const double l00 = std::sqrt(a[0]);
    if (!(l00 > 0.0)) return false;
    const double l10 = a[1] / l00;
    const double l20 = a[2] / l00;
    const double l30 = a[3] / l00;
    const double l40 = a[4] / l00;
    const double l50 = a[5] / l00;

    const double d1 = a[7] - l10 * l10;
    if (!(d1 > 0.0)) return false;
    const double l11 = std::sqrt(d1);
    const double l21 = (a[8] - l20 * l10) / l11;
    const double l31 = (a[9] - l30 * l10) / l11;
    const double l41 = (a[10] - l40 * l10) / l11;
    const double l51 = (a[11] - l50 * l10) / l11;

    const double d2 = a[14] - l20 * l20 - l21 * l21;
    if (!(d2 > 0.0)) return false;
    const double l22 = std::sqrt(d2);
    const double l32 = (a[15] - l30 * l20 - l31 * l21) / l22;
    const double l42 = (a[16] - l40 * l20 - l41 * l21) / l22;
    const double l52 = (a[17] - l50 * l20 - l51 * l21) / l22;

    const double d3 = a[21] - l30 * l30 - l31 * l31 - l32 * l32;
    if (!(d3 > 0.0)) return false;
    const double l33 = std::sqrt(d3);
    const double l43 = (a[22] - l40 * l30 - l41 * l31 - l42 * l32) / l33;
    const double l53 = (a[23] - l50 * l30 - l51 * l31 - l52 * l32) / l33;

    const double d4 = a[28] - l40 * l40 - l41 * l41 - l42 * l42 - l43 * l43;
    if (!(d4 > 0.0)) return false;
    const double l44 = std::sqrt(d4);
    const double l54 = (a[29] - l50 * l40 - l51 * l41 - l52 * l42 - l53 * l43) / l44;

    const double d5 = a[35] - l50 * l50 - l51 * l51 - l52 * l52 - l53 * l53 - l54 * l54;
    if (!(d5 > 0.0)) return false;
    const double l55 = std::sqrt(d5);

    l[0] = l00;
    l[1] = l10;  l[7] = l11;
    l[2] = l20;  l[8] = l21;  l[14] = l22;
    l[3] = l30;  l[9] = l31;  l[15] = l32;  l[21] = l33;
    l[4] = l40;  l[10] = l41; l[16] = l42; l[22] = l43; l[28] = l44;
    l[5] = l50;  l[11] = l51; l[17] = l52; l[23] = l53; l[29] = l54; l[35] = l55;
    return true;
}

SE3_PACKED_FORCEINLINE bool factorizeSpd6LowerSym21TripleRaw(
    const double* diag,
    const double* belief,
    const double* old_msg,
    double* l
) noexcept {
    const double c00 = cavityEntrySym21Raw(diag, belief, old_msg, 0) + kJitter;
    const double c01 = cavityEntrySym21Raw(diag, belief, old_msg, 1);
    const double c11 = cavityEntrySym21Raw(diag, belief, old_msg, 2) + kJitter;
    const double c02 = cavityEntrySym21Raw(diag, belief, old_msg, 3);
    const double c12 = cavityEntrySym21Raw(diag, belief, old_msg, 4);
    const double c22 = cavityEntrySym21Raw(diag, belief, old_msg, 5) + kJitter;
    const double c03 = cavityEntrySym21Raw(diag, belief, old_msg, 6);
    const double c13 = cavityEntrySym21Raw(diag, belief, old_msg, 7);
    const double c23 = cavityEntrySym21Raw(diag, belief, old_msg, 8);
    const double c33 = cavityEntrySym21Raw(diag, belief, old_msg, 9) + kJitter;
    const double c04 = cavityEntrySym21Raw(diag, belief, old_msg, 10);
    const double c14 = cavityEntrySym21Raw(diag, belief, old_msg, 11);
    const double c24 = cavityEntrySym21Raw(diag, belief, old_msg, 12);
    const double c34 = cavityEntrySym21Raw(diag, belief, old_msg, 13);
    const double c44 = cavityEntrySym21Raw(diag, belief, old_msg, 14) + kJitter;
    const double c05 = cavityEntrySym21Raw(diag, belief, old_msg, 15);
    const double c15 = cavityEntrySym21Raw(diag, belief, old_msg, 16);
    const double c25 = cavityEntrySym21Raw(diag, belief, old_msg, 17);
    const double c35 = cavityEntrySym21Raw(diag, belief, old_msg, 18);
    const double c45 = cavityEntrySym21Raw(diag, belief, old_msg, 19);
    const double c55 = cavityEntrySym21Raw(diag, belief, old_msg, 20) + kJitter;

    const double l00 = std::sqrt(c00);
    if (!(l00 > 0.0)) return false;
    const double l10 = c01 / l00;
    const double l20 = c02 / l00;
    const double l30 = c03 / l00;
    const double l40 = c04 / l00;
    const double l50 = c05 / l00;

    const double d1 = c11 - l10 * l10;
    if (!(d1 > 0.0)) return false;
    const double l11 = std::sqrt(d1);
    const double l21 = (c12 - l20 * l10) / l11;
    const double l31 = (c13 - l30 * l10) / l11;
    const double l41 = (c14 - l40 * l10) / l11;
    const double l51 = (c15 - l50 * l10) / l11;

    const double d2 = c22 - l20 * l20 - l21 * l21;
    if (!(d2 > 0.0)) return false;
    const double l22 = std::sqrt(d2);
    const double l32 = (c23 - l30 * l20 - l31 * l21) / l22;
    const double l42 = (c24 - l40 * l20 - l41 * l21) / l22;
    const double l52 = (c25 - l50 * l20 - l51 * l21) / l22;

    const double d3 = c33 - l30 * l30 - l31 * l31 - l32 * l32;
    if (!(d3 > 0.0)) return false;
    const double l33 = std::sqrt(d3);
    const double l43 = (c34 - l40 * l30 - l41 * l31 - l42 * l32) / l33;
    const double l53 = (c35 - l50 * l30 - l51 * l31 - l52 * l32) / l33;

    const double d4 = c44 - l40 * l40 - l41 * l41 - l42 * l42 - l43 * l43;
    if (!(d4 > 0.0)) return false;
    const double l44 = std::sqrt(d4);
    const double l54 = (c45 - l50 * l40 - l51 * l41 - l52 * l42 - l53 * l43) / l44;

    const double d5 = c55 - l50 * l50 - l51 * l51 - l52 * l52 - l53 * l53 - l54 * l54;
    if (!(d5 > 0.0)) return false;
    const double l55 = std::sqrt(d5);

    l[0] = l00;
    l[1] = l10;  l[7] = l11;
    l[2] = l20;  l[8] = l21;  l[14] = l22;
    l[3] = l30;  l[9] = l31;  l[15] = l32;  l[21] = l33;
    l[4] = l40;  l[10] = l41; l[16] = l42; l[22] = l43; l[28] = l44;
    l[5] = l50;  l[11] = l51; l[17] = l52; l[23] = l53; l[29] = l54; l[35] = l55;
    return true;
}

SE3_PACKED_FORCEINLINE bool factorizeSpd6LowerSym21TriplePackedRaw(
    const double* diag,
    const double* belief,
    const double* old_msg,
    double* l
) noexcept {
    const double c00 = cavityEntrySym21Raw(diag, belief, old_msg, 0) + kJitter;
    const double c01 = cavityEntrySym21Raw(diag, belief, old_msg, 1);
    const double c11 = cavityEntrySym21Raw(diag, belief, old_msg, 2) + kJitter;
    const double c02 = cavityEntrySym21Raw(diag, belief, old_msg, 3);
    const double c12 = cavityEntrySym21Raw(diag, belief, old_msg, 4);
    const double c22 = cavityEntrySym21Raw(diag, belief, old_msg, 5) + kJitter;
    const double c03 = cavityEntrySym21Raw(diag, belief, old_msg, 6);
    const double c13 = cavityEntrySym21Raw(diag, belief, old_msg, 7);
    const double c23 = cavityEntrySym21Raw(diag, belief, old_msg, 8);
    const double c33 = cavityEntrySym21Raw(diag, belief, old_msg, 9) + kJitter;
    const double c04 = cavityEntrySym21Raw(diag, belief, old_msg, 10);
    const double c14 = cavityEntrySym21Raw(diag, belief, old_msg, 11);
    const double c24 = cavityEntrySym21Raw(diag, belief, old_msg, 12);
    const double c34 = cavityEntrySym21Raw(diag, belief, old_msg, 13);
    const double c44 = cavityEntrySym21Raw(diag, belief, old_msg, 14) + kJitter;
    const double c05 = cavityEntrySym21Raw(diag, belief, old_msg, 15);
    const double c15 = cavityEntrySym21Raw(diag, belief, old_msg, 16);
    const double c25 = cavityEntrySym21Raw(diag, belief, old_msg, 17);
    const double c35 = cavityEntrySym21Raw(diag, belief, old_msg, 18);
    const double c45 = cavityEntrySym21Raw(diag, belief, old_msg, 19);
    const double c55 = cavityEntrySym21Raw(diag, belief, old_msg, 20) + kJitter;

    const double l00 = std::sqrt(c00);
    if (!(l00 > 0.0)) return false;
    const double l10 = c01 / l00;
    const double l20 = c02 / l00;
    const double l30 = c03 / l00;
    const double l40 = c04 / l00;
    const double l50 = c05 / l00;

    const double d1 = c11 - l10 * l10;
    if (!(d1 > 0.0)) return false;
    const double l11 = std::sqrt(d1);
    const double l21 = (c12 - l20 * l10) / l11;
    const double l31 = (c13 - l30 * l10) / l11;
    const double l41 = (c14 - l40 * l10) / l11;
    const double l51 = (c15 - l50 * l10) / l11;

    const double d2 = c22 - l20 * l20 - l21 * l21;
    if (!(d2 > 0.0)) return false;
    const double l22 = std::sqrt(d2);
    const double l32 = (c23 - l30 * l20 - l31 * l21) / l22;
    const double l42 = (c24 - l40 * l20 - l41 * l21) / l22;
    const double l52 = (c25 - l50 * l20 - l51 * l21) / l22;

    const double d3 = c33 - l30 * l30 - l31 * l31 - l32 * l32;
    if (!(d3 > 0.0)) return false;
    const double l33 = std::sqrt(d3);
    const double l43 = (c34 - l40 * l30 - l41 * l31 - l42 * l32) / l33;
    const double l53 = (c35 - l50 * l30 - l51 * l31 - l52 * l32) / l33;

    const double d4 = c44 - l40 * l40 - l41 * l41 - l42 * l42 - l43 * l43;
    if (!(d4 > 0.0)) return false;
    const double l44 = std::sqrt(d4);
    const double l54 = (c45 - l50 * l40 - l51 * l41 - l52 * l42 - l53 * l43) / l44;

    const double d5 = c55 - l50 * l50 - l51 * l51 - l52 * l52 - l53 * l53 - l54 * l54;
    if (!(d5 > 0.0)) return false;
    const double l55 = std::sqrt(d5);

    l[0] = l00;
    l[1] = l10;  l[2] = l11;
    l[3] = l20;  l[4] = l21;  l[5] = l22;
    l[6] = l30;  l[7] = l31;  l[8] = l32;  l[9] = l33;
    l[10] = l40; l[11] = l41; l[12] = l42; l[13] = l43; l[14] = l44;
    l[15] = l50; l[16] = l51; l[17] = l52; l[18] = l53; l[19] = l54; l[20] = l55;
    return true;
}

SE3_PACKED_FORCEINLINE void solveSpd6LowerOneRaw(
    const double* l,
    const double* rhs,
    double* out
) noexcept {
    const double y0 = rhs[0] / l[0];
    const double y1 = (rhs[1] - l[1] * y0) / l[7];
    const double y2 = (rhs[2] - l[2] * y0 - l[8] * y1) / l[14];
    const double y3 = (rhs[3] - l[3] * y0 - l[9] * y1 - l[15] * y2) / l[21];
    const double y4 = (rhs[4] - l[4] * y0 - l[10] * y1 - l[16] * y2 - l[22] * y3) / l[28];
    const double y5 =
        (rhs[5] - l[5] * y0 - l[11] * y1 - l[17] * y2 - l[23] * y3 - l[29] * y4) / l[35];

    const double x5 = y5 / l[35];
    const double x4 = (y4 - l[29] * x5) / l[28];
    const double x3 = (y3 - l[22] * x4 - l[23] * x5) / l[21];
    const double x2 = (y2 - l[16] * x4 - l[17] * x5 - l[15] * x3) / l[14];
    const double x1 = (y1 - l[10] * x4 - l[11] * x5 - l[9] * x3 - l[8] * x2) / l[7];
    const double x0 =
        (y0 - l[4] * x4 - l[5] * x5 - l[3] * x3 - l[2] * x2 - l[1] * x1) / l[0];

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = x3;
    out[4] = x4;
    out[5] = x5;
}

SE3_PACKED_FORCEINLINE void solveSpd6LowerOnePackedRaw(
    const double* l,
    const double* rhs,
    double* out
) noexcept {
    const double y0 = rhs[0] / l[0];
    const double y1 = (rhs[1] - l[1] * y0) / l[2];
    const double y2 = (rhs[2] - l[3] * y0 - l[4] * y1) / l[5];
    const double y3 = (rhs[3] - l[6] * y0 - l[7] * y1 - l[8] * y2) / l[9];
    const double y4 = (rhs[4] - l[10] * y0 - l[11] * y1 - l[12] * y2 - l[13] * y3) / l[14];
    const double y5 =
        (rhs[5] - l[15] * y0 - l[16] * y1 - l[17] * y2 - l[18] * y3 - l[19] * y4) / l[20];

    const double x5 = y5 / l[20];
    const double x4 = (y4 - l[19] * x5) / l[14];
    const double x3 = (y3 - l[13] * x4 - l[18] * x5) / l[9];
    const double x2 = (y2 - l[12] * x4 - l[17] * x5 - l[8] * x3) / l[5];
    const double x1 = (y1 - l[11] * x4 - l[16] * x5 - l[7] * x3 - l[4] * x2) / l[2];
    const double x0 =
        (y0 - l[10] * x4 - l[15] * x5 - l[6] * x3 - l[3] * x2 - l[1] * x1) / l[0];

    out[0] = x0;
    out[1] = x1;
    out[2] = x2;
    out[3] = x3;
    out[4] = x4;
    out[5] = x5;
}

SE3_PACKED_FORCEINLINE void solveSpd6LowerRaw(
    const double* l,
    const double* rhs,
    int nrhs,
    double* out
) noexcept {
    for (int col = 0; col < nrhs; ++col) {
        solveSpd6LowerOneRaw(l, rhs + 6 * col, out + 6 * col);
    }
}

SE3_PACKED_FORCEINLINE void solveSpd6Lower7Raw(
    const double* l,
    const double* rhs,
    double* out
) noexcept {
    solveSpd6LowerOneRaw(l, rhs + 0, out + 0);
    solveSpd6LowerOneRaw(l, rhs + 6, out + 6);
    solveSpd6LowerOneRaw(l, rhs + 12, out + 12);
    solveSpd6LowerOneRaw(l, rhs + 18, out + 18);
    solveSpd6LowerOneRaw(l, rhs + 24, out + 24);
    solveSpd6LowerOneRaw(l, rhs + 30, out + 30);
    solveSpd6LowerOneRaw(l, rhs + 36, out + 36);
}

SE3_PACKED_FORCEINLINE void solveSpd6LowerCrossEtaRepeatedRaw(
    const double* l,
    const double* cross_rhs_col_major,
    const double* eta_rhs,
    double* out
) noexcept {
    solveSpd6CrossEtaBatched(l, cross_rhs_col_major, eta_rhs, out);
}

SE3_PACKED_FORCEINLINE void solveSpd6LowerCrossEtaPackedRepeatedRaw(
    const double* l,
    const double* cross_rhs_col_major,
    const double* eta_rhs,
    double* out
) noexcept {
    solveSpd6CrossEtaPackedBatched(l, cross_rhs_col_major, eta_rhs, out);
}

SE3_PACKED_FORCEINLINE double dotCrossSolvedRow6(
    const double* cross_col_major,
    const double* solved_col,
    int row
) noexcept {
    double projected = cross_col_major[row] * solved_col[0];
    projected += cross_col_major[row + 6] * solved_col[1];
    projected += cross_col_major[row + 12] * solved_col[2];
    projected += cross_col_major[row + 18] * solved_col[3];
    projected += cross_col_major[row + 24] * solved_col[4];
    projected += cross_col_major[row + 30] * solved_col[5];
    return projected;
}

SE3_PACKED_FORCEINLINE void projectSchurTargetRaw(
    const double* diag_target,
    const double* cross_target_other,
    const double* solved,
    double* out_lam,
    double* out_eta_projected
) noexcept {
    const double* s0 = solved;
    const double* s1 = solved + 6;
    const double* s2 = solved + 12;
    const double* s3 = solved + 18;
    const double* s4 = solved + 24;
    const double* s5 = solved + 30;
    const double* se = solved + 36;

    out_lam[0] = diag_target[0] - dotCrossSolvedRow6(cross_target_other, s0, 0);
    out_lam[1] = diag_target[1] - dotCrossSolvedRow6(cross_target_other, s1, 0);
    out_lam[2] = diag_target[2] - dotCrossSolvedRow6(cross_target_other, s1, 1);
    out_lam[3] = diag_target[3] - dotCrossSolvedRow6(cross_target_other, s2, 0);
    out_lam[4] = diag_target[4] - dotCrossSolvedRow6(cross_target_other, s2, 1);
    out_lam[5] = diag_target[5] - dotCrossSolvedRow6(cross_target_other, s2, 2);
    out_lam[6] = diag_target[6] - dotCrossSolvedRow6(cross_target_other, s3, 0);
    out_lam[7] = diag_target[7] - dotCrossSolvedRow6(cross_target_other, s3, 1);
    out_lam[8] = diag_target[8] - dotCrossSolvedRow6(cross_target_other, s3, 2);
    out_lam[9] = diag_target[9] - dotCrossSolvedRow6(cross_target_other, s3, 3);
    out_lam[10] = diag_target[10] - dotCrossSolvedRow6(cross_target_other, s4, 0);
    out_lam[11] = diag_target[11] - dotCrossSolvedRow6(cross_target_other, s4, 1);
    out_lam[12] = diag_target[12] - dotCrossSolvedRow6(cross_target_other, s4, 2);
    out_lam[13] = diag_target[13] - dotCrossSolvedRow6(cross_target_other, s4, 3);
    out_lam[14] = diag_target[14] - dotCrossSolvedRow6(cross_target_other, s4, 4);
    out_lam[15] = diag_target[15] - dotCrossSolvedRow6(cross_target_other, s5, 0);
    out_lam[16] = diag_target[16] - dotCrossSolvedRow6(cross_target_other, s5, 1);
    out_lam[17] = diag_target[17] - dotCrossSolvedRow6(cross_target_other, s5, 2);
    out_lam[18] = diag_target[18] - dotCrossSolvedRow6(cross_target_other, s5, 3);
    out_lam[19] = diag_target[19] - dotCrossSolvedRow6(cross_target_other, s5, 4);
    out_lam[20] = diag_target[20] - dotCrossSolvedRow6(cross_target_other, s5, 5);

    out_eta_projected[0] = dotCrossSolvedRow6(cross_target_other, se, 0);
    out_eta_projected[1] = dotCrossSolvedRow6(cross_target_other, se, 1);
    out_eta_projected[2] = dotCrossSolvedRow6(cross_target_other, se, 2);
    out_eta_projected[3] = dotCrossSolvedRow6(cross_target_other, se, 3);
    out_eta_projected[4] = dotCrossSolvedRow6(cross_target_other, se, 4);
    out_eta_projected[5] = dotCrossSolvedRow6(cross_target_other, se, 5);
}

double normalizedPrecisionResidual(
    const double* candidate, const double* old, const double* diagonal
) noexcept {
    // Diagonal normalization prevents rotation units from hiding translation drift.
    constexpr int diag_ids[6] = {0, 2, 5, 9, 14, 20};
    double root_diagonal[6];
    for (int i=0;i<6;++i)
        root_diagonal[i]=std::sqrt(std::max(std::abs(diagonal[diag_ids[i]]),1e-30));
    double delta2 = 0.0;
    double current2 = 0.0;
    double old2 = 0.0;
    for (int col = 0, k = 0; col < 6; ++col) {
        for (int row = 0; row <= col; ++row, ++k) {
            const double scale = root_diagonal[row] * root_diagonal[col];
            const double a = candidate[k] / scale;
            const double b = old[k] / scale;
            if (!finiteDouble(a) || !finiteDouble(b)) {
                return std::numeric_limits<double>::infinity();
            }
            const double weight = row == col ? 1.0 : 2.0;
            delta2 += weight * (a - b) * (a - b);
            current2 += weight * a * a;
            old2 += weight * b * b;
        }
    }
    // A zero-information tree message needs an absolute tolerance in these
    // normalized units, not a relative test against cancellation roundoff.
    return std::sqrt(delta2 / std::max({current2, old2, 1.0}));
}

bool computeFullTargetRaw(
    const double* eta_target,
    const double* eta_other,
    const double* diag_target,
    const double* diag_other,
    const double* cross_target_other,
    const double* cross_other_target,
    const double* belief_other_eta,
    const double* belief_other_lam,
    const double* old_target_eta,
    const double* old_target_lam,
    const double* old_other_eta,
    const double* old_other_lam,
    double eta_damping,
    double max_rel_update,
    bool use_l21_cholesky,
    double* out_eta,
    double* out_lam,
    double* precision_residual = nullptr,
    const double* cached_reference_norms = nullptr
) noexcept {
    double solved[42];
    if (use_l21_cholesky) {
        double l[21];
        if (!factorizeSpd6LowerSym21TriplePackedRaw(diag_other, belief_other_lam, old_other_lam, l)) {
            return false;
        }
        double eta_rhs[6];
        for (int i = 0; i < 6; ++i) {
            eta_rhs[i] = eta_other[i] + belief_other_eta[i] - old_other_eta[i];
        }
        solveSpd6LowerCrossEtaPackedRepeatedRaw(l, cross_other_target, eta_rhs, solved);
    } else {
        double l[36];
        if (!factorizeSpd6LowerSym21TripleRaw(diag_other, belief_other_lam, old_other_lam, l)) {
            return false;
        }
        double eta_rhs[6];
        for (int i = 0; i < 6; ++i) {
            eta_rhs[i] = eta_other[i] + belief_other_eta[i] - old_other_eta[i];
        }
        solveSpd6LowerCrossEtaRepeatedRaw(l, cross_other_target, eta_rhs, solved);
    }

    double eta_projected[6];
    projectSchurTargetRaw(diag_target, cross_target_other, solved, out_lam, eta_projected);
    if (precision_residual != nullptr) {
        *precision_residual = normalizedPrecisionResidual(out_lam, old_target_lam, diag_target);
    }
    for (int row = 0; row < 6; ++row) {
        out_eta[row] = eta_target[row] - eta_projected[row];
    }

    damp6(old_target_eta, eta_damping, out_eta);
    damp21(old_target_lam, eta_damping, out_lam);
    stabilizeMessage6(
        old_target_eta,
        old_target_lam,
        eta_target,
        diag_target,
        max_rel_update,
        out_eta,
        out_lam,
        cached_reference_norms
    );
    return true;
}

SE3_PACKED_FORCEINLINE void writeMat6RowMajorRaw(
    const Mat6& mat,
    double* out
) noexcept {
    for (int row = 0; row < 6; ++row) {
        const int base = 6 * row;
        out[base + 0] = mat(row, 0);
        out[base + 1] = mat(row, 1);
        out[base + 2] = mat(row, 2);
        out[base + 3] = mat(row, 3);
        out[base + 4] = mat(row, 4);
        out[base + 5] = mat(row, 5);
    }
}

double computeFullLambdaFactor(
    SyntheticSE3PackedSoAWorkspace& w,
    int idx,
    bool mark_fixed_initialized,
    double eta_damping,
    double max_rel_update,
    bool use_l21_cholesky,
    bool measure_precision = false
) {
    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const int v0 = w.binary_var0_id[static_cast<size_t>(idx)];
    const int v1 = w.binary_var1_id[static_cast<size_t>(idx)];

    const double* eta0 = vec6Ptr(w.binary_eta0, idx);
    const double* eta1 = vec6Ptr(w.binary_eta1, idx);
    const double* diag0 = sym21Ptr(w.binary_diag0_lam21, idx);
    const double* diag1 = sym21Ptr(w.binary_diag1_lam21, idx);
    const double* cross01_ptr = mat36Ptr(w.binary_cross01_lam36, idx);
    const double* cross10_ptr = mat36Ptr(w.binary_cross10_lam36, idx);
    const Eigen::Map<const Mat6> cross01(cross01_ptr);
    const Eigen::Map<const Mat6> cross10(cross10_ptr);

    const double* b0_eta = vec6Ptr(w.belief_eta, v0);
    const double* b1_eta = vec6Ptr(w.belief_eta, v1);
    const double* b0_lam = sym21Ptr(w.belief_lam21, v0);
    const double* b1_lam = sym21Ptr(w.belief_lam21, v1);
    const double* old0_eta = binaryMsgEtaPtr(w, slot0);
    const double* old1_eta = binaryMsgEtaPtr(w, slot1);
    const double* old0_lam = binaryMsgLam21Ptr(w, slot0);
    const double* old1_lam = binaryMsgLam21Ptr(w, slot1);

    double raw0_eta[6];
    double raw1_eta[6];
    double raw0_lam[21];
    double raw1_lam[21];
    double precision0 = std::numeric_limits<double>::infinity();
    double precision1 = std::numeric_limits<double>::infinity();
    const double* norms = nullptr;
    if (use_l21_cholesky && w.binary_reference_norms_valid)
        norms = w.binary_reference_norms.data() + 4*static_cast<size_t>(idx);
    const bool raw0_ok = computeFullTargetRaw(
        eta0,
        eta1,
        diag0,
        diag1,
        cross01_ptr,
        cross10_ptr,
        b1_eta,
        b1_lam,
        old0_eta,
        old0_lam,
        old1_eta,
        old1_lam,
        eta_damping,
        max_rel_update,
        use_l21_cholesky,
        raw0_eta,
        raw0_lam,
        measure_precision ? &precision0 : nullptr,
        norms
    );
    const bool raw1_ok = computeFullTargetRaw(
        eta1,
        eta0,
        diag1,
        diag0,
        cross10_ptr,
        cross01_ptr,
        b0_eta,
        b0_lam,
        old1_eta,
        old1_lam,
        old0_eta,
        old0_lam,
        eta_damping,
        max_rel_update,
        use_l21_cholesky,
        raw1_eta,
        raw1_lam,
        measure_precision ? &precision1 : nullptr,
        norms ? norms+2 : nullptr
    );
    if (raw0_ok && raw1_ok) {
        copy6(raw0_eta, binaryMsgEtaPtr(w, slot0));
        copy6(raw1_eta, binaryMsgEtaPtr(w, slot1));
        copy21(raw0_lam, binaryMsgLam21Ptr(w, slot0));
        copy21(raw1_lam, binaryMsgLam21Ptr(w, slot1));
        w.fixed_lam_initialized[static_cast<size_t>(idx)] = mark_fixed_initialized ? 1 : 0;
        w.fixed_eta_map_valid[static_cast<size_t>(idx)] = 0;
        return measure_precision ? std::max(precision0, precision1) : 0.0;
    }

    Vec6 eno0;
    Vec6 eno1;
    for (int i = 0; i < 6; ++i) {
        eno0[i] = eta1[i] + b1_eta[i] - old1_eta[i];
        eno1[i] = eta0[i] + b0_eta[i] - old0_eta[i];
    }

    Mat6 a0;
    Mat6 a1;
    cavityFromSym21(diag1, b1_lam, old1_lam, a0);
    cavityFromSym21(diag0, b0_lam, old0_lam, a1);

    Mat6x7 rhs0;
    rhs0.template leftCols<6>() = cross10;
    rhs0.col(6) = eno0;
    const Mat6x7 solved0 = solveMat6x7WithFallback(a0, rhs0);
    const Mat6 Y0 = solved0.template leftCols<6>();
    const Vec6 y0 = solved0.col(6);

    Mat6x7 rhs1;
    rhs1.template leftCols<6>() = cross01;
    rhs1.col(6) = eno1;
    const Mat6x7 solved1 = solveMat6x7WithFallback(a1, rhs1);
    const Mat6 Y1 = solved1.template leftCols<6>();
    const Vec6 y1 = solved1.col(6);

    Vec6 out0_eta = Eigen::Map<const Vec6>(eta0) - cross01 * y0;
    Vec6 out1_eta = Eigen::Map<const Vec6>(eta1) - cross10 * y1;
    Mat6 out0_lam_mat;
    Mat6 out1_lam_mat;
    mat6FromSym21(diag0, out0_lam_mat);
    out0_lam_mat.noalias() -= cross01 * Y0;
    mat6FromSym21(diag1, out1_lam_mat);
    out1_lam_mat.noalias() -= cross10 * Y1;

    double out0_lam[21];
    double out1_lam[21];
    sym21FromMat6(out0_lam_mat, out0_lam);
    sym21FromMat6(out1_lam_mat, out1_lam);
    damp6(old0_eta, eta_damping, out0_eta.data());
    damp6(old1_eta, eta_damping, out1_eta.data());
    damp21(old0_lam, eta_damping, out0_lam);
    damp21(old1_lam, eta_damping, out1_lam);
    stabilizeMessage6(old0_eta, old0_lam, eta0, diag0, max_rel_update, out0_eta.data(), out0_lam);
    stabilizeMessage6(old1_eta, old1_lam, eta1, diag1, max_rel_update, out1_eta.data(), out1_lam);

    copy6(out0_eta.data(), binaryMsgEtaPtr(w, slot0));
    copy6(out1_eta.data(), binaryMsgEtaPtr(w, slot1));
    copy21(out0_lam, binaryMsgLam21Ptr(w, slot0));
    copy21(out1_lam, binaryMsgLam21Ptr(w, slot1));

    w.fixed_lam_initialized[static_cast<size_t>(idx)] = mark_fixed_initialized ? 1 : 0;
    w.fixed_eta_map_valid[static_cast<size_t>(idx)] = 0;
    // A fallback solve must not certify precision convergence.
    return std::numeric_limits<double>::infinity();
}

bool buildFixedEtaMapsForFactor(SyntheticSE3PackedSoAWorkspace& w, int idx) {
    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const int v0 = w.binary_var0_id[static_cast<size_t>(idx)];
    const int v1 = w.binary_var1_id[static_cast<size_t>(idx)];
    const double* diag0 = sym21Ptr(w.binary_diag0_lam21, idx);
    const double* diag1 = sym21Ptr(w.binary_diag1_lam21, idx);
    const double* b0_lam = sym21Ptr(w.belief_lam21, v0);
    const double* b1_lam = sym21Ptr(w.belief_lam21, v1);
    const double* old0_lam = binaryMsgLam21Ptr(w, slot0);
    const double* old1_lam = binaryMsgLam21Ptr(w, slot1);
    const Eigen::Map<const Mat6> cross01(mat36Ptr(w.binary_cross01_lam36, idx));
    const Eigen::Map<const Mat6> cross10(mat36Ptr(w.binary_cross10_lam36, idx));

    Mat6 a0;
    Mat6 a1;
    cavityFromSym21(diag1, b1_lam, old1_lam, a0);
    cavityFromSym21(diag0, b0_lam, old0_lam, a1);
    const Mat6 inv0 = solveMatWithFallback(a0, Mat6::Identity());
    const Mat6 inv1 = solveMatWithFallback(a1, Mat6::Identity());
    const Mat6 map0 = cross01 * inv0;
    const Mat6 map1 = cross10 * inv1;
    if (!map0.allFinite() || !map1.allFinite()) return false;
    writeMat6RowMajorRaw(map0, mat36Ptr(w.fixed_eta_map0_lam36, idx));
    writeMat6RowMajorRaw(map1, mat36Ptr(w.fixed_eta_map1_lam36, idx));
    w.fixed_eta_map_valid[static_cast<size_t>(idx)] = 1;
    return true;
}

void fixedEtaUpdateRowMajorRaw(
    const double* factor_eta_target,
    const double* factor_eta_other,
    const double* eta_map_row_major,
    const double* belief_other_eta,
    const double* old_other_eta,
    const double* old_target_eta,
    double eta_damping,
    double max_rel_update,
    double* out_eta
) noexcept {
    const double x0 = factor_eta_other[0] + belief_other_eta[0] - old_other_eta[0];
    const double x1 = factor_eta_other[1] + belief_other_eta[1] - old_other_eta[1];
    const double x2 = factor_eta_other[2] + belief_other_eta[2] - old_other_eta[2];
    const double x3 = factor_eta_other[3] + belief_other_eta[3] - old_other_eta[3];
    const double x4 = factor_eta_other[4] + belief_other_eta[4] - old_other_eta[4];
    const double x5 = factor_eta_other[5] + belief_other_eta[5] - old_other_eta[5];

    for (int row = 0; row < 6; ++row) {
        const double* m = eta_map_row_major + 6 * row;
        out_eta[row] = factor_eta_target[row] -
            (m[0] * x0 + m[1] * x1 + m[2] * x2 +
             m[3] * x3 + m[4] * x4 + m[5] * x5);
    }
    damp6(old_target_eta, eta_damping, out_eta);
    stabilizeEta6(old_target_eta, factor_eta_target, max_rel_update, out_eta);
}

void computeFixedEtaFactor(
    SyntheticSE3PackedSoAWorkspace& w,
    int idx,
    double eta_damping,
    double max_rel_update,
    int* map_builds
) {
    if (w.fixed_lam_initialized[static_cast<size_t>(idx)] == 0) {
        throw std::runtime_error("SE3 packed SoA fixed-eta update requested before fixed lambda initialization");
    }
    if (w.fixed_eta_map_valid[static_cast<size_t>(idx)] == 0) {
        if (!buildFixedEtaMapsForFactor(w, idx)) {
            throw std::runtime_error("SE3 packed SoA fixed eta-map build failed");
        }
        if (map_builds != nullptr) {
            ++(*map_builds);
        }
    }

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const int v0 = w.binary_var0_id[static_cast<size_t>(idx)];
    const int v1 = w.binary_var1_id[static_cast<size_t>(idx)];
    const double* eta0 = vec6Ptr(w.binary_eta0, idx);
    const double* eta1 = vec6Ptr(w.binary_eta1, idx);
    double* msg0 = binaryMsgEtaPtr(w, slot0);
    double* msg1 = binaryMsgEtaPtr(w, slot1);
    const double* b0_eta = vec6Ptr(w.belief_eta, v0);
    const double* b1_eta = vec6Ptr(w.belief_eta, v1);
    const double* map0 = mat36Ptr(w.fixed_eta_map0_lam36, idx);
    const double* map1 = mat36Ptr(w.fixed_eta_map1_lam36, idx);

    double old0[6];
    double old1[6];
    copy6(msg0, old0);
    copy6(msg1, old1);
    fixedEtaUpdateRowMajorRaw(eta0, eta1, map0, b1_eta, old1, old0, eta_damping, max_rel_update, msg0);
    fixedEtaUpdateRowMajorRaw(eta1, eta0, map1, b0_eta, old0, old1, eta_damping, max_rel_update, msg1);
}

SE3_PACKED_FORCEINLINE void computeFixedEtaFactorHotSync(
    SyntheticSE3PackedSoAWorkspace& w,
    int idx,
    double eta_damping,
    double max_rel_update
) noexcept {
    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const int v0 = w.binary_var0_id[static_cast<size_t>(idx)];
    const int v1 = w.binary_var1_id[static_cast<size_t>(idx)];
    const double* eta0 = vec6Ptr(w.binary_eta0, idx);
    const double* eta1 = vec6Ptr(w.binary_eta1, idx);
    double* msg0 = binaryMsgEtaPtr(w, slot0);
    double* msg1 = binaryMsgEtaPtr(w, slot1);
    const double* b0_eta = vec6Ptr(w.belief_eta, v0);
    const double* b1_eta = vec6Ptr(w.belief_eta, v1);
    const double* map0 = mat36Ptr(w.fixed_eta_map0_lam36, idx);
    const double* map1 = mat36Ptr(w.fixed_eta_map1_lam36, idx);

    double old0[6];
    double old1[6];
    copy6(msg0, old0);
    copy6(msg1, old1);
    fixedEtaUpdateRowMajorRaw(eta0, eta1, map0, b1_eta, old1, old0, eta_damping, max_rel_update, msg0);
    fixedEtaUpdateRowMajorRaw(eta1, eta0, map1, b0_eta, old0, old1, eta_damping, max_rel_update, msg1);
}

void updateBeliefAt(
    SyntheticSE3PackedSoAWorkspace& w,
    int var_idx,
    bool eta_only,
    bool update_mu
) {
    double* belief_eta = vec6Ptr(w.belief_eta, var_idx);
    const double* prior_eta = vec6Ptr(w.prior_eta, var_idx);
    double eta[6];
    copy6(prior_eta, eta);

    const int unary_begin = w.unary_offsets[static_cast<size_t>(var_idx)];
    const int unary_end = w.unary_offsets[static_cast<size_t>(var_idx + 1)];
    for (int k = unary_begin; k < unary_end; ++k) {
        const double* msg = vec6Ptr(w.unary_msg_eta, w.unary_ids[static_cast<size_t>(k)]);
        for (int i = 0; i < 6; ++i) eta[i] += msg[i];
    }
    const int binary_begin = w.binary_offsets[static_cast<size_t>(var_idx)];
    const int binary_end = w.binary_offsets[static_cast<size_t>(var_idx + 1)];
    for (int k = binary_begin; k < binary_end; ++k) {
        const double* msg = binaryMsgEtaPtr(w, w.binary_slot_ids[static_cast<size_t>(k)]);
        for (int i = 0; i < 6; ++i) eta[i] += msg[i];
    }
    copy6(eta, belief_eta);

    if (!eta_only) {
        double* belief_lam = sym21Ptr(w.belief_lam21, var_idx);
        const double* prior_lam = sym21Ptr(w.prior_lam21, var_idx);
        copy21(prior_lam, belief_lam);
        for (int k = unary_begin; k < unary_end; ++k) {
            const double* msg = sym21Ptr(w.unary_msg_lam21, w.unary_ids[static_cast<size_t>(k)]);
            for (int i = 0; i < 21; ++i) belief_lam[i] += msg[i];
        }
        for (int k = binary_begin; k < binary_end; ++k) {
            const double* msg = binaryMsgLam21Ptr(w, w.binary_slot_ids[static_cast<size_t>(k)]);
            for (int i = 0; i < 21; ++i) belief_lam[i] += msg[i];
        }
    }

    if (update_mu) {
        Mat6 lam;
        mat6FromSym21(sym21Ptr(w.belief_lam21, var_idx), lam);
        const Vec6 mu = solveVecWithFallback(lam, Eigen::Map<const Vec6>(belief_eta));
        if (!mu.allFinite()) {
            throw std::runtime_error("SE3 packed SoA belief solve failed");
        }
        copy6(mu.data(), vec6Ptr(w.mu, var_idx));
        w.mu_valid[static_cast<size_t>(var_idx)] = 1;
    } else {
        w.mu_valid[static_cast<size_t>(var_idx)] = 0;
    }
}

bool refreshMuAt(SyntheticSE3PackedSoAWorkspace& w, int var_idx) {
    if (w.mu_valid[static_cast<size_t>(var_idx)] != 0) return true;
    Mat6 lam;
    mat6FromSym21(sym21Ptr(w.belief_lam21, var_idx), lam);
    const Vec6 mu = solveVecWithFallback(lam, Eigen::Map<const Vec6>(vec6Ptr(w.belief_eta, var_idx)));
    if (!mu.allFinite()) return false;
    copy6(mu.data(), vec6Ptr(w.mu, var_idx));
    w.mu_valid[static_cast<size_t>(var_idx)] = 1;
    return true;
}

void completePrecisionSweep(SyntheticSE3PackedSoAWorkspace& w, bool eta_only, bool check, int global_sweep) {
    ++w.sweeps_since_relinearize;
    if(eta_only) ++w.eta_only_sweeps;
    else ++w.full_precision_sweeps;
    if(!check) return;
    double residual=0;
    for(double value:w.precision_residuals) residual=std::max(residual,value);
    ++w.precision_checks;
    w.last_precision_check_sweep=global_sweep;
    w.last_precision_residual=residual;
    if(finiteDouble(residual) && residual<=SE3PrecisionPolicy::tolerance) {
        ++w.precision_stable_checks;
        if(!w.precision_frozen && w.precision_stable_checks>=SE3PrecisionPolicy::stable_checks) {
            w.precision_frozen=true;
            ++w.precision_freezes;
            if(w.first_precision_freeze_sweep<0) w.first_precision_freeze_sweep=w.sweeps_since_relinearize;
        }
    } else {
        if(w.precision_frozen) ++w.precision_thaws;
        w.precision_frozen=false;
        w.precision_stable_checks=0;
    }
}

void persistentPackedSweeps(SyntheticSE3PackedSoAWorkspace& w, int sweeps, int threads,
                           int fixed_start, double damping, double update_limit) {
    const auto allowed_cpus = ScopedWorkerAffinity::availableMask(threads);
    #pragma omp parallel num_threads(threads)
    {
        ScopedWorkerAffinity affinity(allowed_cpus,omp_get_thread_num());
        if(omp_get_thread_num()==0) ScopedWorkerAffinity::observeTeam(omp_get_num_threads());
        for(int sweep=0;sweep<sweeps;++sweep) {
            const int global=w.sweeps_since_relinearize;
            const bool check=w.adaptive_precision && (w.precision_frozen
                ? global-w.last_precision_check_sweep>=SE3PrecisionPolicy::frozen_check_period
                : (global+1)%SE3PrecisionPolicy::check_period==0);
            const bool fixed=w.adaptive_precision || (fixed_start>=0 && global>=fixed_start);
            const bool eta_only=w.adaptive_precision ? w.precision_frozen && !check
                : fixed_start>=0 && global>fixed_start;
            if(eta_only && w.fixed_eta_maps_all_valid) {
                #pragma omp for schedule(static)
                for(int f=0;f<w.num_binary_factors;++f)
                    computeFixedEtaFactorHotSync(w,f,damping,update_limit);
            } else if(eta_only) {
                #pragma omp for schedule(static)
                for(int f=0;f<w.num_binary_factors;++f)
                    computeFixedEtaFactor(w,f,damping,update_limit,nullptr);
            } else {
                #pragma omp for schedule(static)
                for(int f=0;f<w.num_binary_factors;++f) {
                    const double residual=computeFullLambdaFactor(w,f,fixed,damping,update_limit,false,check);
                    if(check) w.precision_residuals[f]=residual;
                }
            }
            // The factor barrier makes all certificates available. This metadata
            // update can overlap the variable pass: that pass only uses the local
            // eta_only flag. Its ending barrier publishes metadata for next sweep.
            #pragma omp single nowait
            {
                w.fixed_eta_maps_all_valid=eta_only?1:0;
                completePrecisionSweep(w,eta_only,check,global);
            }
            #pragma omp for schedule(static)
            for(int v=0;v<w.num_vars;++v) updateBeliefAt(w,v,eta_only,sweep+1==sweeps);
        }
    }
}

#undef SE3_PACKED_FORCEINLINE

}  // namespace

SyntheticSE3PackedSoAWorkspace buildSyntheticSE3PackedSoAWorkspace(
    const SyntheticSE3Problem& problem,
    double tiny_prior
) {
    SyntheticSE3PackedSoAWorkspace w;
    w.num_vars = static_cast<int>(problem.init_poses.size());
    w.num_binary_factors = static_cast<int>(problem.edges.size());
    w.num_unary_factors = w.num_vars > 0 ? 1 : 0;
    w.tiny_prior = tiny_prior;

    w.prior_eta.assign(static_cast<size_t>(w.num_vars) * 6, 0.0);
    w.prior_lam21.assign(static_cast<size_t>(w.num_vars) * 21, 0.0);
    w.belief_eta.assign(static_cast<size_t>(w.num_vars) * 6, 0.0);
    w.belief_lam21.assign(static_cast<size_t>(w.num_vars) * 21, 0.0);
    w.mu.assign(static_cast<size_t>(w.num_vars) * 6, 0.0);
    w.mu_valid.assign(static_cast<size_t>(w.num_vars), 0);
    for (int i = 0; i < w.num_vars; ++i) {
        double* prior_lam = sym21Ptr(w.prior_lam21, i);
        for (int d = 0; d < 6; ++d) {
            prior_lam[sym21Index(d, d)] = tiny_prior;
        }
        copy21(prior_lam, sym21Ptr(w.belief_lam21, i));
    }

    w.unary_var_id.assign(static_cast<size_t>(w.num_unary_factors), 0);
    w.unary_eta.assign(static_cast<size_t>(w.num_unary_factors) * 6, 0.0);
    w.unary_lam21.assign(static_cast<size_t>(w.num_unary_factors) * 21, 0.0);
    w.unary_msg_eta.assign(static_cast<size_t>(w.num_unary_factors) * 6, 0.0);
    w.unary_msg_lam21.assign(static_cast<size_t>(w.num_unary_factors) * 21, 0.0);

    w.binary_var0_id.resize(static_cast<size_t>(w.num_binary_factors));
    w.binary_var1_id.resize(static_cast<size_t>(w.num_binary_factors));
    for (int i = 0; i < w.num_binary_factors; ++i) {
        const SyntheticSE3Edge& edge = problem.edges[static_cast<size_t>(i)];
        w.binary_var0_id[static_cast<size_t>(i)] = edge.i;
        w.binary_var1_id[static_cast<size_t>(i)] = edge.j;
    }
    w.binary_eta0.assign(static_cast<size_t>(w.num_binary_factors) * 6, 0.0);
    w.binary_eta1.assign(static_cast<size_t>(w.num_binary_factors) * 6, 0.0);
    w.binary_diag0_lam21.assign(static_cast<size_t>(w.num_binary_factors) * 21, 0.0);
    w.binary_diag1_lam21.assign(static_cast<size_t>(w.num_binary_factors) * 21, 0.0);
    w.binary_cross01_lam36.assign(static_cast<size_t>(w.num_binary_factors) * 36, 0.0);
    w.binary_cross10_lam36.assign(static_cast<size_t>(w.num_binary_factors) * 36, 0.0);
    w.binary_msg_eta.assign(static_cast<size_t>(w.num_binary_factors) * 2 * 6, 0.0);
    w.binary_msg_lam21.assign(static_cast<size_t>(w.num_binary_factors) * 2 * 21, 0.0);
    w.fixed_eta_map0_lam36.assign(static_cast<size_t>(w.num_binary_factors) * 36, 0.0);
    w.fixed_eta_map1_lam36.assign(static_cast<size_t>(w.num_binary_factors) * 36, 0.0);
    w.fixed_lam_initialized.assign(static_cast<size_t>(w.num_binary_factors), 0);
    w.fixed_eta_map_valid.assign(static_cast<size_t>(w.num_binary_factors), 0);
    w.fixed_eta_maps_all_valid = 0;
    std::vector<int> unary_degree(static_cast<size_t>(w.num_vars), 0);
    std::vector<int> binary_degree(static_cast<size_t>(w.num_vars), 0);
    if (w.num_unary_factors > 0) unary_degree[0] = 1;
    for (const SyntheticSE3Edge& edge : problem.edges) {
        ++binary_degree[static_cast<size_t>(edge.i)];
        ++binary_degree[static_cast<size_t>(edge.j)];
    }
    w.unary_offsets.assign(static_cast<size_t>(w.num_vars) + 1, 0);
    w.binary_offsets.assign(static_cast<size_t>(w.num_vars) + 1, 0);
    for (int i = 0; i < w.num_vars; ++i) {
        w.unary_offsets[static_cast<size_t>(i + 1)] =
            w.unary_offsets[static_cast<size_t>(i)] + unary_degree[static_cast<size_t>(i)];
        w.binary_offsets[static_cast<size_t>(i + 1)] =
            w.binary_offsets[static_cast<size_t>(i)] + binary_degree[static_cast<size_t>(i)];
    }
    w.unary_ids.assign(static_cast<size_t>(w.unary_offsets.back()), 0);
    w.binary_slot_ids.assign(static_cast<size_t>(w.binary_offsets.back()), 0);
    std::vector<int> unary_cursor = w.unary_offsets;
    std::vector<int> binary_cursor = w.binary_offsets;
    if (w.num_unary_factors > 0) {
        w.unary_ids[static_cast<size_t>(unary_cursor[0]++)] = 0;
    }
    for (int edge_index = 0; edge_index < w.num_binary_factors; ++edge_index) {
        const int v0 = w.binary_var0_id[static_cast<size_t>(edge_index)];
        const int v1 = w.binary_var1_id[static_cast<size_t>(edge_index)];
        w.binary_slot_ids[static_cast<size_t>(binary_cursor[static_cast<size_t>(v0)]++)] = 2 * edge_index;
        w.binary_slot_ids[static_cast<size_t>(binary_cursor[static_cast<size_t>(v1)]++)] = 2 * edge_index + 1;
    }

    return w;
}

void relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(
    SyntheticSE3PackedSoAWorkspace& w,
    const gbp::FactorGraph& graph
) {
    if (w.num_vars != static_cast<int>(graph.var_nodes.size())) {
        throw std::runtime_error("SE3 packed SoA topology mismatch: variable count");
    }
    if (static_cast<int>(graph.factors.size()) != w.num_binary_factors + w.num_unary_factors) {
        throw std::runtime_error("SE3 packed SoA topology mismatch: factor count");
    }
    w.sweeps_since_relinearize = 0;
    w.binary_reference_norms_valid = false;
    w.jacobi_ready=false;
    w.defect_ready=false;
    w.defect_mean_sweeps=0;
    w.defect_map_builds=0;
    w.defect_clamps=0;
    w.defect_build_sec=0;
    w.defect_inverse_residual=0;
    w.jacobi_sweeps=0;
    w.cycle_message_rebuilds=0;
    w.precision_frozen = false;
    w.precision_stable_checks = 0;
    w.precision_checks = 0;
    w.precision_freezes = 0;
    w.precision_thaws = 0;
    w.first_precision_freeze_sweep = -1;
    w.last_precision_check_sweep = -1;
    w.last_precision_residual = 0.0;
    w.full_precision_sweeps = 0;
    w.eta_only_sweeps = 0;
    for (int i = 0; i < w.num_vars; ++i) {
        const gbp::VariableNode& var = *graph.var_nodes[static_cast<size_t>(i)];
        copy6(var.prior.etaData(), vec6Ptr(w.prior_eta, i));
        sym21FromUpperColMajor(var.prior.lamData(), sym21Ptr(w.prior_lam21, i));
        copy6(var.belief.etaData(), vec6Ptr(w.belief_eta, i));
        sym21FromUpperColMajor(var.belief.lamData(), sym21Ptr(w.belief_lam21, i));
        zero6(vec6Ptr(w.mu, i));
        w.mu_valid[static_cast<size_t>(i)] = 0;
    }
    std::fill(w.fixed_lam_initialized.begin(), w.fixed_lam_initialized.end(), 0);
    std::fill(w.fixed_eta_map_valid.begin(), w.fixed_eta_map_valid.end(), 0);
    w.fixed_eta_maps_all_valid = 0;
    for (int edge_index = 0; edge_index < w.num_binary_factors; ++edge_index) {
        const gbp::Factor& factor = *graph.factors[static_cast<size_t>(edge_index)];
        if (factor.factor.dim() != 12) {
            throw std::runtime_error("SE3 packed SoA expected 12D binary factor");
        }
        const double* eta = factor.factor.etaData();
        const double* lam = factor.factor.lamData();
        copy6(eta, vec6Ptr(w.binary_eta0, edge_index));
        copy6(eta + 6, vec6Ptr(w.binary_eta1, edge_index));

        double full6[36];
        for (int col = 0; col < 6; ++col) {
            for (int row = 0; row < 6; ++row) {
                full6[row + col * 6] = lam[row + col * 12];
            }
        }
        sym21FromUpperColMajor(full6, sym21Ptr(w.binary_diag0_lam21, edge_index));
        for (int col = 0; col < 6; ++col) {
            for (int row = 0; row < 6; ++row) {
                full6[row + col * 6] = lam[(row + 6) + (col + 6) * 12];
            }
        }
        sym21FromUpperColMajor(full6, sym21Ptr(w.binary_diag1_lam21, edge_index));
        double* cross = mat36Ptr(w.binary_cross01_lam36, edge_index);
        for (int col = 0; col < 6; ++col) {
            for (int row = 0; row < 6; ++row) {
                cross[row + col * 6] = lam[row + (col + 6) * 12];
            }
        }
        double* cross10 = mat36Ptr(w.binary_cross10_lam36, edge_index);
        for (int col = 0; col < 6; ++col) {
            for (int row = 0; row < 6; ++row) {
                cross10[row + col * 6] = lam[(row + 6) + col * 12];
            }
        }
        for (int local = 0; local < 2; ++local) {
            const int slot = 2 * edge_index + local;
            if (local < static_cast<int>(factor.messages.size())) {
                const utils::NdimGaussian& msg = factor.messages[static_cast<size_t>(local)];
                copy6(msg.etaData(), binaryMsgEtaPtr(w, slot));
                sym21FromUpperColMajor(msg.lamData(), binaryMsgLam21Ptr(w, slot));
            } else {
                zero6(binaryMsgEtaPtr(w, slot));
                std::fill(
                    binaryMsgLam21Ptr(w, slot),
                    binaryMsgLam21Ptr(w, slot) + 21,
                    0.0
                );
            }
        }
    }

    if (w.num_unary_factors > 0) {
        const gbp::Factor& factor = *graph.factors.back();
        if (factor.factor.dim() != 6) {
            throw std::runtime_error("SE3 packed SoA expected 6D unary factor");
        }
        copy6(factor.factor.etaData(), vec6Ptr(w.unary_eta, 0));
        sym21FromUpperColMajor(factor.factor.lamData(), sym21Ptr(w.unary_lam21, 0));
        // Unary messages are exactly the unary factor. The object graph may
        // have zeroed message buffers immediately after reset, before the
        // first factor pass, so importing factor.messages here would drop the
        // anchor until another object sweep runs.
        copy6(vec6Ptr(w.unary_eta, 0), vec6Ptr(w.unary_msg_eta, 0));
        copy21(sym21Ptr(w.unary_lam21, 0), sym21Ptr(w.unary_msg_lam21, 0));
    }
}

void blockJacobiSE3PackedIterations(SyntheticSE3PackedSoAWorkspace& w, int sweeps, int num_threads) {
    if(sweeps<=0) return;
    const int n=w.num_vars, threads=effectiveThreadCount(num_threads);
    if(!w.jacobi_ready) {
        w.jacobi_diagonal.resize(21*n); w.jacobi_inverse.resize(36*n); w.jacobi_rhs.resize(6*n);
        w.jacobi_x.resize(6*n); w.jacobi_alt.resize(6*n);
        #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
        for(int v=0;v<n;++v) {
            double* diag=sym21Ptr(w.jacobi_diagonal,v);
            double* rhs=vec6Ptr(w.jacobi_rhs,v);
            copy21(sym21Ptr(w.prior_lam21,v),diag); copy6(vec6Ptr(w.prior_eta,v),rhs);
            for(int p=w.unary_offsets[v];p<w.unary_offsets[v+1];++p) {
                const int f=w.unary_ids[p];
                const double* a=sym21Ptr(w.unary_lam21,f),*b=vec6Ptr(w.unary_eta,f);
                for(int j=0;j<21;++j) diag[j]+=a[j];
                for(int j=0;j<6;++j) rhs[j]+=b[j];
            }
            for(int p=w.binary_offsets[v];p<w.binary_offsets[v+1];++p) {
                const int slot=w.binary_slot_ids[p],f=slot/2;
                const double* a=sym21Ptr(slot%2?w.binary_diag1_lam21:w.binary_diag0_lam21,f);
                const double* b=vec6Ptr(slot%2?w.binary_eta1:w.binary_eta0,f);
                for(int j=0;j<21;++j) diag[j]+=a[j];
                for(int j=0;j<6;++j) rhs[j]+=b[j];
            }
            Mat6 a; mat6FromSym21(diag,a);
            const Mat6 inv=solveMatWithFallback(a,Mat6::Identity());
            std::copy(inv.data(),inv.data()+36,mat36Ptr(w.jacobi_inverse,v));
        }
        w.jacobi_ready=true;
    }
    #pragma omp parallel num_threads(threads) if(threads>1)
    {
        #pragma omp for schedule(static)
        for(int v=0;v<n;++v) {
            if(!refreshMuAt(w,v)) throw std::runtime_error("Jacobi initialization has nonfinite mean");
            copy6(vec6Ptr(w.mu,v),vec6Ptr(w.jacobi_x,v));
        }
        // PSD unary/binary factors imply H <= 2*blockdiag(H). Shared omega=2/3.
        for(int sweep=0;sweep<sweeps;++sweep) {
            const double* x=(sweep%2?w.jacobi_alt:w.jacobi_x).data();
            double* y=(sweep%2?w.jacobi_x:w.jacobi_alt).data();
            #pragma omp for schedule(static)
            for(int v=0;v<n;++v) {
                double product[6]; sym21MatVec6Raw(sym21Ptr(w.jacobi_diagonal,v),x+6*v,product);
                for(int p=w.binary_offsets[v];p<w.binary_offsets[v+1];++p) {
                    const int slot=w.binary_slot_ids[p], f=slot/2;
                    const double* z=x+6*(slot%2?w.binary_var0_id[f]:w.binary_var1_id[f]);
                    const double* a=mat36Ptr(slot%2?w.binary_cross10_lam36:w.binary_cross01_lam36,f);
                    for(int r=0;r<6;++r) product[r]+=a[r]*z[0]+a[r+6]*z[1]+a[r+12]*z[2]+a[r+18]*z[3]+a[r+24]*z[4]+a[r+30]*z[5];
                }
                double r[6]; for(int j=0;j<6;++j) r[j]=w.jacobi_rhs[6*v+j]-product[j];
                const double* a=mat36Ptr(w.jacobi_inverse,v);
                for(int j=0;j<6;++j)
                    y[6*v+j]=x[6*v+j]+(2./3.)*(a[j]*r[0]+a[j+6]*r[1]+a[j+12]*r[2]+a[j+18]*r[3]+a[j+24]*r[4]+a[j+30]*r[5]);
            }
        }
        const auto& x=sweeps%2?w.jacobi_alt:w.jacobi_x;
        #pragma omp for schedule(static)
        for(int v=0;v<n;++v) {
            copy6(x.data()+6*v,vec6Ptr(w.mu,v));
            sym21MatVec6Raw(sym21Ptr(w.belief_lam21,v),x.data()+6*v,vec6Ptr(w.belief_eta,v));
            w.mu_valid[v]=1;
        }
    }
    w.jacobi_sweeps+=sweeps;
}

void rebuildSE3PackedMessagesAtMean(SyntheticSE3PackedSoAWorkspace& w,
    const Eigen::VectorXd& mean, const Eigen::VectorXd& residual, int num_threads) {
    if(mean.size()!=6*w.num_vars || residual.size()!=mean.size())
        throw std::runtime_error("SE3 message reconstruction dimension mismatch");
    const int threads=effectiveThreadCount(num_threads);
    // m_fi = Lambda_fi*x_i + (b_fi - H_fi*x) - r_i/degree(i).
    // The final term makes sum(m_fi)+unary+prior == belief_Lambda*x.
    // At r=0 this is the Gaussian BP eta fixed point when Lambda is fixed.
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int f=0;f<w.num_binary_factors;++f) {
        for(int side=0;side<2;++side) {
            const int slot=2*f+side;
            const int i=side?w.binary_var1_id[f]:w.binary_var0_id[f];
            const int j=side?w.binary_var0_id[f]:w.binary_var1_id[f];
            const double* x=mean.data()+6*i,*y=mean.data()+6*j;
            double lam_x[6],diag_x[6];
            sym21MatVec6Raw(binaryMsgLam21Ptr(w,slot),x,lam_x);
            sym21MatVec6Raw(sym21Ptr(side?w.binary_diag1_lam21:w.binary_diag0_lam21,f),x,diag_x);
            const double* cross=mat36Ptr(side?w.binary_cross10_lam36:w.binary_cross01_lam36,f);
            const double* eta=vec6Ptr(side?w.binary_eta1:w.binary_eta0,f);
            double* message=binaryMsgEtaPtr(w,slot);
            const double degree=static_cast<double>(w.binary_offsets[i+1]-w.binary_offsets[i]);
            for(int d=0;d<6;++d)
                message[d]=lam_x[d]+eta[d]-diag_x[d]-
                    (cross[d]*y[0]+cross[d+6]*y[1]+cross[d+12]*y[2]+cross[d+18]*y[3]+cross[d+24]*y[4]+cross[d+30]*y[5])-
                    residual[6*i+d]/degree;
        }
    }
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int i=0;i<w.num_vars;++i) {
        updateBeliefAt(w,i,false,false);
        // For isolated variables there is no FV state to initialize.
        if(w.binary_offsets[i+1]>w.binary_offsets[i]) {
            copy6(mean.data()+6*i,vec6Ptr(w.mu,i));
            w.mu_valid[i]=1;
        }
    }
    ++w.cycle_message_rebuilds;
}

void recomputeSE3PackedBeliefs(SyntheticSE3PackedSoAWorkspace& w,int num_threads) {
    const int threads=effectiveThreadCount(num_threads);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int i=0;i<w.num_vars;++i) updateBeliefAt(w,i,false,false);
}

double measureSE3PackedEtaMismatch(const SyntheticSE3PackedSoAWorkspace& w) {
    double difference2=0, target2=0;
    for(int v=0;v<w.num_vars;++v) {
        double sum[6]; copy6(vec6Ptr(w.prior_eta,v),sum);
        for(int p=w.unary_offsets[v];p<w.unary_offsets[v+1];++p) {
            const double* message=vec6Ptr(w.unary_msg_eta,w.unary_ids[p]);
            for(int d=0;d<6;++d) sum[d]+=message[d];
        }
        for(int p=w.binary_offsets[v];p<w.binary_offsets[v+1];++p) {
            const double* message=binaryMsgEtaPtr(w,w.binary_slot_ids[p]);
            for(int d=0;d<6;++d) sum[d]+=message[d];
        }
        const double* eta=vec6Ptr(w.belief_eta,v);
        for(int d=0;d<6;++d) {
            const double error=eta[d]-sum[d];
            difference2+=error*error; target2+=eta[d]*eta[d];
        }
    }
    return std::sqrt(difference2/std::max(target2,1e-300));
}

void liftSE3PackedMeanCorrection(SyntheticSE3PackedSoAWorkspace& w,
    const Eigen::VectorXd& delta, bool precision_weighted, int num_threads) {
    if(delta.size()!=6*w.num_vars || !delta.allFinite())
        throw std::runtime_error("SE3 eta lift requires a finite mean correction");
    const int threads=effectiveThreadCount(num_threads);
    auto lift_variable=[&](int v) {
        const int begin=w.binary_offsets[v],end=w.binary_offsets[v+1];
        double target[6],base[6],sum[6];
        copy6(vec6Ptr(w.belief_eta,v),target);
        copy6(vec6Ptr(w.prior_eta,v),base);
        for(int p=w.unary_offsets[v];p<w.unary_offsets[v+1];++p) {
            const double* message=vec6Ptr(w.unary_msg_eta,w.unary_ids[p]);
            for(int d=0;d<6;++d) base[d]+=message[d];
        }
        copy6(base,sum);
        for(int p=begin;p<end;++p) {
            const int slot=w.binary_slot_ids[p];
            double* message=binaryMsgEtaPtr(w,slot);
            if(precision_weighted) {
                double shifted[6];
                sym21MatVec6Raw(binaryMsgLam21Ptr(w,slot),delta.data()+6*v,shifted);
                for(int d=0;d<6;++d) message[d]+=shifted[d];
            }
            for(int d=0;d<6;++d) sum[d]+=message[d];
        }
        if(begin==end) {
            double difference=0,scale=0;
            for(int d=0;d<6;++d) { difference+=std::abs(target[d]-sum[d]); scale+=std::abs(target[d])+std::abs(sum[d]); }
            return difference<=128*std::numeric_limits<double>::epsilon()*std::max(scale,1e-300);
        }
        double correction[6];
        for(int d=0;d<6;++d) correction[d]=(target[d]-sum[d])/double(end-begin);
        copy6(base,sum);
        for(int p=begin;p<end;++p) {
            double* message=binaryMsgEtaPtr(w,w.binary_slot_ids[p]);
            for(int d=0;d<6;++d) {
                message[d]+=correction[d];
                sum[d]+=message[d];
            }
        }
        // Use the actual message sum. Cached target mu is unchanged up to the
        // usual roundoff of belief_eta=belief_lambda*mu, not a different solve.
        copy6(sum,vec6Ptr(w.belief_eta,v));
        return true;
    };
    int failed=0;
    if(threads>1) {
        #pragma omp parallel for schedule(static) num_threads(threads) reduction(+:failed)
        for(int v=0;v<w.num_vars;++v) if(!lift_variable(v)) ++failed;
    } else for(int v=0;v<w.num_vars;++v) if(!lift_variable(v)) ++failed;
    if(failed) throw std::runtime_error("Cannot lift a nonzero correction through an isolated variable");
}

void initializeBalancedSE3PackedMessages(SyntheticSE3PackedSoAWorkspace& w, int num_threads) {
    if(w.sweeps_since_relinearize!=0)
        throw std::runtime_error("Balanced initialization requires a fresh SE3 linearization");
    const int threads=effectiveThreadCount(num_threads);
    // Each eta starts at its factor gradient. Their sum is the assembled b,
    // so b=0 is stationary even when individual factor gradients are nonzero.
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int f=0;f<w.num_binary_factors;++f) {
        copy6(vec6Ptr(w.binary_eta0,f),binaryMsgEtaPtr(w,2*f));
        copy6(vec6Ptr(w.binary_eta1,f),binaryMsgEtaPtr(w,2*f+1));
        copy21(sym21Ptr(w.binary_diag0_lam21,f),binaryMsgLam21Ptr(w,2*f));
        copy21(sym21Ptr(w.binary_diag1_lam21,f),binaryMsgLam21Ptr(w,2*f+1));
    }
    for(int f=0;f<w.num_unary_factors;++f) {
        copy6(vec6Ptr(w.unary_eta,f),vec6Ptr(w.unary_msg_eta,f));
        copy21(sym21Ptr(w.unary_lam21,f),sym21Ptr(w.unary_msg_lam21,f));
    }
    std::fill(w.fixed_lam_initialized.begin(),w.fixed_lam_initialized.end(),0);
    std::fill(w.fixed_eta_map_valid.begin(),w.fixed_eta_map_valid.end(),0);
    w.fixed_eta_maps_all_valid=0;
    w.precision_frozen=false;
    w.precision_stable_checks=0;
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int v=0;v<w.num_vars;++v) updateBeliefAt(w,v,false,false);
}

void assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(
    const SyntheticSE3PackedSoAWorkspace& w,
    Eigen::VectorXd& eta,
    int num_threads
) {
    eta.resize(static_cast<Eigen::Index>(w.num_vars) * 6);
    const int thread_count = effectiveThreadCount(num_threads);
    auto assemble_var = [&](int var_idx) {
        double* out = eta.data() + static_cast<size_t>(var_idx) * 6;
        copy6(vec6Ptr(w.prior_eta, var_idx), out);

        const int unary_begin = w.unary_offsets[static_cast<size_t>(var_idx)];
        const int unary_end = w.unary_offsets[static_cast<size_t>(var_idx + 1)];
        for (int position = unary_begin; position < unary_end; ++position) {
            const int unary_id = w.unary_ids[static_cast<size_t>(position)];
            const double* factor_eta = vec6Ptr(w.unary_eta, unary_id);
            for (int d = 0; d < 6; ++d) {
                out[d] += factor_eta[d];
            }
        }

        const int binary_begin = w.binary_offsets[static_cast<size_t>(var_idx)];
        const int binary_end = w.binary_offsets[static_cast<size_t>(var_idx + 1)];
        for (int position = binary_begin; position < binary_end; ++position) {
            const int slot = w.binary_slot_ids[static_cast<size_t>(position)];
            const int edge_index = slot >> 1;
            const double* factor_eta = (slot & 1) == 0
                ? vec6Ptr(w.binary_eta0, edge_index)
                : vec6Ptr(w.binary_eta1, edge_index);
            for (int d = 0; d < 6; ++d) {
                out[d] += factor_eta[d];
            }
        }
    };

    if (thread_count > 1 && w.num_vars > 64) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            assemble_var(var_idx);
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            assemble_var(var_idx);
        }
    }
}

void multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
    const SyntheticSE3PackedSoAWorkspace& w,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& out,
    int num_threads
) {
    if (x.size() != static_cast<Eigen::Index>(w.num_vars) * 6) {
        throw std::runtime_error("SE3 packed joint multiply size mismatch");
    }
    out.resize(x.size());
    const int thread_count = effectiveThreadCount(num_threads);
    auto multiply_var = [&](int var_idx) {
        const double* x_local = x.data() + static_cast<size_t>(var_idx) * 6;
        double* y = out.data() + static_cast<size_t>(var_idx) * 6;
        sym21MatVec6Raw(sym21Ptr(w.prior_lam21, var_idx), x_local, y);

        double contribution[6];
        const int unary_begin = w.unary_offsets[static_cast<size_t>(var_idx)];
        const int unary_end = w.unary_offsets[static_cast<size_t>(var_idx + 1)];
        for (int position = unary_begin; position < unary_end; ++position) {
            const int unary_id = w.unary_ids[static_cast<size_t>(position)];
            sym21MatVec6Raw(
                sym21Ptr(w.unary_lam21, unary_id),
                x_local,
                contribution
            );
            for (int d = 0; d < 6; ++d) {
                y[d] += contribution[d];
            }
        }

        const int binary_begin = w.binary_offsets[static_cast<size_t>(var_idx)];
        const int binary_end = w.binary_offsets[static_cast<size_t>(var_idx + 1)];
        for (int position = binary_begin; position < binary_end; ++position) {
            const int slot = w.binary_slot_ids[static_cast<size_t>(position)];
            const int edge_index = slot >> 1;
            const bool local_is_second = (slot & 1) != 0;
            const int other_var = local_is_second
                ? w.binary_var0_id[static_cast<size_t>(edge_index)]
                : w.binary_var1_id[static_cast<size_t>(edge_index)];
            const double* x_other =
                x.data() + static_cast<size_t>(other_var) * 6;
            const double* diagonal = local_is_second
                ? sym21Ptr(w.binary_diag1_lam21, edge_index)
                : sym21Ptr(w.binary_diag0_lam21, edge_index);
            const double* cross = local_is_second
                ? mat36Ptr(w.binary_cross10_lam36, edge_index)
                : mat36Ptr(w.binary_cross01_lam36, edge_index);

            sym21MatVec6Raw(diagonal, x_local, contribution);
            for (int d = 0; d < 6; ++d) {
                y[d] += contribution[d];
            }
            for (int row = 0; row < 6; ++row) {
                double value = 0.0;
                for (int col = 0; col < 6; ++col) {
                    value += cross[row + col * 6] * x_other[col];
                }
                y[row] += value;
            }
        }
    };

    if (thread_count > 1 && w.num_vars > 64) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            multiply_var(var_idx);
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            multiply_var(var_idx);
        }
    }
}

void synchronousIterationsSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& w,
    int num_sweeps,
    int num_threads,
    int fixed_lam_after_sweep,
    double eta_damping,
    double max_rel_update,
    SyntheticSE3PackedSoAStats* stats
) {
    if (num_sweeps <= 0) return;
    const int thread_count = effectiveThreadCount(num_threads);
    const bool collect_stats = stats != nullptr;
    if (thread_count == 1 && !w.binary_reference_norms_valid) {
        w.binary_reference_norms.resize(static_cast<size_t>(w.num_binary_factors)*4);
        for (int f=0; f<w.num_binary_factors; ++f) {
            double* norm=w.binary_reference_norms.data()+4*static_cast<size_t>(f);
            norm[0]=froNormSym21(sym21Ptr(w.binary_diag0_lam21,f));
            norm[1]=std::sqrt(sqNorm6(vec6Ptr(w.binary_eta0,f)));
            norm[2]=froNormSym21(sym21Ptr(w.binary_diag1_lam21,f));
            norm[3]=std::sqrt(sqNorm6(vec6Ptr(w.binary_eta1,f)));
        }
        w.binary_reference_norms_valid=true;
    }
    const bool use_l21_cholesky = packedSoAL21CholeskyEnabled(thread_count);
    const bool use_hot_fixed_eta_kernel =
        packedSoAHotFixedEtaKernelEnabled(thread_count);
    SyntheticSE3PackedSoAStats local_stats;
    if (w.adaptive_precision) w.precision_residuals.resize(w.num_binary_factors);
    if(w.persistent_sweeps && thread_count>1 && !collect_stats) {
        persistentPackedSweeps(w,num_sweeps,thread_count,fixed_lam_after_sweep,eta_damping,max_rel_update);
        return;
    }

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        const int global_sweep = w.sweeps_since_relinearize;
        const bool check_precision = w.adaptive_precision && (
            w.precision_frozen
                ? global_sweep - w.last_precision_check_sweep >= SE3PrecisionPolicy::frozen_check_period
                : (global_sweep + 1) % SE3PrecisionPolicy::check_period == 0);
        const bool use_fixed_lam =
            w.adaptive_precision ||
            (fixed_lam_after_sweep >= 0 && global_sweep >= fixed_lam_after_sweep);
        const bool use_eta_only =
            w.adaptive_precision ? (w.precision_frozen && !check_precision) :
            (fixed_lam_after_sweep >= 0 && global_sweep > fixed_lam_after_sweep);
        const bool use_hot_eta =
            use_eta_only &&
            use_hot_fixed_eta_kernel &&
            w.fixed_eta_maps_all_valid != 0;
        const bool update_mu = (sweep + 1 == num_sweeps);
        int map_builds = 0;
        if (!use_eta_only) {
            w.fixed_eta_maps_all_valid = 0;
        }
        SteadyClock::time_point factor_t0;
        SteadyClock::time_point factor_t1;
        SteadyClock::time_point var_t0;
        SteadyClock::time_point var_t1;
        if (collect_stats) {
            factor_t0 = SteadyClock::now();
        }
        if (thread_count > 1) {
            if (use_hot_eta) {
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    computeFixedEtaFactorHotSync(w, idx, eta_damping, max_rel_update);
                }
            } else if (use_eta_only) {
                #pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:map_builds)
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    computeFixedEtaFactor(w, idx, eta_damping, max_rel_update, &map_builds);
                }
            } else {
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    const double local_residual = computeFullLambdaFactor(
                        w,
                        idx,
                        use_fixed_lam,
                        eta_damping,
                        max_rel_update,
                        use_l21_cholesky,
                        check_precision
                    );
                    if (check_precision) w.precision_residuals[idx] = local_residual;
                }
            }
        } else {
            if (use_hot_eta) {
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    computeFixedEtaFactorHotSync(w, idx, eta_damping, max_rel_update);
                }
            } else if (use_eta_only) {
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    computeFixedEtaFactor(w, idx, eta_damping, max_rel_update, &map_builds);
                }
            } else {
                for (int idx = 0; idx < w.num_binary_factors; ++idx) {
                    const double local_residual = computeFullLambdaFactor(
                        w,
                        idx,
                        use_fixed_lam,
                        eta_damping,
                        max_rel_update,
                        use_l21_cholesky,
                        check_precision
                    );
                    if (check_precision) w.precision_residuals[idx] = local_residual;
                }
            }
        }
        if (use_eta_only) {
            w.fixed_eta_maps_all_valid = 1;
        }
        if (collect_stats) {
            factor_t1 = SteadyClock::now();
            var_t0 = SteadyClock::now();
        }

        if (thread_count > 1) {
            #pragma omp parallel for schedule(static) num_threads(thread_count)
            for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
                updateBeliefAt(w, var_idx, use_eta_only, update_mu);
            }
        } else {
            for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
                updateBeliefAt(w, var_idx, use_eta_only, update_mu);
            }
        }
        if (collect_stats) {
            var_t1 = SteadyClock::now();
        }

        completePrecisionSweep(w,use_eta_only,check_precision,global_sweep);
        if (collect_stats) {
            ++local_stats.sweeps;
            if (use_eta_only) {
                ++local_stats.fixed_eta_sweeps;
            } else if (use_fixed_lam) {
                ++local_stats.fixed_lambda_init_sweeps;
            } else {
                ++local_stats.full_lambda_sweeps;
            }
            local_stats.fixed_eta_map_builds += map_builds;
            local_stats.factor_pass_sec += elapsedSeconds(factor_t0, factor_t1);
            local_stats.variable_pass_sec += elapsedSeconds(var_t0, var_t1);
        }
    }

    if (stats != nullptr) {
        stats->sweeps += local_stats.sweeps;
        stats->full_lambda_sweeps += local_stats.full_lambda_sweeps;
        stats->fixed_lambda_init_sweeps += local_stats.fixed_lambda_init_sweeps;
        stats->fixed_eta_sweeps += local_stats.fixed_eta_sweeps;
        stats->fixed_eta_map_builds += local_stats.fixed_eta_map_builds;
        stats->factor_pass_sec += local_stats.factor_pass_sec;
        stats->variable_pass_sec += local_stats.variable_pass_sec;
    }
}

Eigen::VectorXd stackedMeanVectorSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& w,
    int num_threads
) {
    Eigen::VectorXd out;
    stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(w, out, num_threads);
    return out;
}

void stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(
    SyntheticSE3PackedSoAWorkspace& w,
    Eigen::VectorXd& out,
    int num_threads
) {
    out.resize(w.num_vars * 6);
    const int thread_count = effectiveThreadCount(num_threads);
    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            if (!refreshMuAt(w, var_idx)) {
                throw std::runtime_error("SE3 packed SoA stacked mean solve failed");
            }
            std::memcpy(out.data() + static_cast<size_t>(var_idx) * 6, vec6Ptr(w.mu, var_idx), 6 * sizeof(double));
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            if (!refreshMuAt(w, var_idx)) {
                throw std::runtime_error("SE3 packed SoA stacked mean solve failed");
            }
            std::memcpy(out.data() + static_cast<size_t>(var_idx) * 6, vec6Ptr(w.mu, var_idx), 6 * sizeof(double));
        }
    }
}

void applyMeanDeltaSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& w,
    const Eigen::VectorXd& delta,
    int num_threads
) {
    if (delta.size() != w.num_vars * 6) {
        throw std::runtime_error("SE3 packed SoA delta size mismatch");
    }
    const int thread_count = effectiveThreadCount(num_threads);
    auto apply_var = [&](int var_idx) {
        if (!refreshMuAt(w, var_idx)) {
            throw std::runtime_error("SE3 packed SoA delta apply mean solve failed");
        }
        double* mu = vec6Ptr(w.mu, var_idx);
        const double* d = delta.data() + static_cast<size_t>(var_idx) * 6;
        for (int i = 0; i < 6; ++i) {
            mu[i] += d[i];
        }
        const double* lam = sym21Ptr(w.belief_lam21, var_idx);
        double* eta = vec6Ptr(w.belief_eta, var_idx);
        sym21MatVec6Raw(lam, mu, eta);
        w.mu_valid[static_cast<size_t>(var_idx)] = 1;
    };
    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            apply_var(var_idx);
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            apply_var(var_idx);
        }
    }
}

void copySyntheticSE3PackedSoABeliefsToGraph(
    const SyntheticSE3PackedSoAWorkspace& w,
    gbp::FactorGraph& graph,
    int num_threads,
    bool copy_lambda
) {
    if (w.num_vars != static_cast<int>(graph.var_nodes.size())) {
        throw std::runtime_error("SE3 packed SoA copyBeliefsToGraph topology mismatch");
    }
    const int thread_count = effectiveThreadCount(num_threads);
    auto copy_var = [&](int var_idx) {
        gbp::VariableNode& var = *graph.var_nodes[static_cast<size_t>(var_idx)];
        std::memcpy(var.belief.etaData(), vec6Ptr(w.belief_eta, var_idx), 6 * sizeof(double));
        if (copy_lambda) {
            auto lam_ref = var.belief.lamRef();
            fullColMajorFromSym21(sym21Ptr(w.belief_lam21, var_idx), lam_ref.data());
        }
        if (w.mu_valid[static_cast<size_t>(var_idx)] != 0) {
            std::memcpy(var.mu.data(), vec6Ptr(w.mu, var_idx), 6 * sizeof(double));
            var.markMuCurrent();
        } else {
            var.invalidateMu();
        }
    };
    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            copy_var(var_idx);
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            copy_var(var_idx);
        }
    }
}

void copySyntheticSE3PackedSoAToGraph(
    const SyntheticSE3PackedSoAWorkspace& w,
    gbp::FactorGraph& graph,
    int num_threads
) {
    if (w.num_vars != static_cast<int>(graph.var_nodes.size()) ||
        static_cast<int>(graph.factors.size()) != w.num_binary_factors + w.num_unary_factors) {
        throw std::runtime_error("SE3 packed SoA copyToGraph topology mismatch");
    }
    const int thread_count = effectiveThreadCount(num_threads);
    auto copy_factor = [&](int edge_index) {
        gbp::Factor& factor = *graph.factors[static_cast<size_t>(edge_index)];
        for (int local = 0; local < 2; ++local) {
            const int slot = 2 * edge_index + local;
            utils::NdimGaussian& msg = factor.messages[static_cast<size_t>(local)];
            std::memcpy(msg.etaData(), binaryMsgEtaPtr(w, slot), 6 * sizeof(double));
            fullColMajorFromSym21(binaryMsgLam21Ptr(w, slot), msg.lamData());
            if (factor.messages_next.size() == factor.messages.size()) {
                utils::NdimGaussian& next = factor.messages_next[static_cast<size_t>(local)];
                std::memcpy(next.etaData(), msg.etaData(), 6 * sizeof(double));
                std::memcpy(next.lamData(), msg.lamData(), 36 * sizeof(double));
            }
        }
    };
    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < w.num_binary_factors; ++edge_index) {
            copy_factor(edge_index);
        }
    } else {
        for (int edge_index = 0; edge_index < w.num_binary_factors; ++edge_index) {
            copy_factor(edge_index);
        }
    }

    if (w.num_unary_factors > 0) {
        gbp::Factor& factor = *graph.factors.back();
        utils::NdimGaussian& msg = factor.messages[0];
        std::memcpy(msg.etaData(), vec6Ptr(w.unary_msg_eta, 0), 6 * sizeof(double));
        fullColMajorFromSym21(sym21Ptr(w.unary_msg_lam21, 0), msg.lamData());
        if (factor.messages_next.size() == factor.messages.size()) {
            utils::NdimGaussian& next = factor.messages_next[0];
            std::memcpy(next.etaData(), msg.etaData(), 6 * sizeof(double));
            std::memcpy(next.lamData(), msg.lamData(), 36 * sizeof(double));
        }
    }

    auto copy_var = [&](int var_idx) {
        gbp::VariableNode& var = *graph.var_nodes[static_cast<size_t>(var_idx)];
        std::memcpy(var.belief.etaData(), vec6Ptr(w.belief_eta, var_idx), 6 * sizeof(double));
        auto lam_ref = var.belief.lamRef();
        fullColMajorFromSym21(sym21Ptr(w.belief_lam21, var_idx), lam_ref.data());
        if (w.mu_valid[static_cast<size_t>(var_idx)] != 0) {
            std::memcpy(var.mu.data(), vec6Ptr(w.mu, var_idx), 6 * sizeof(double));
            var.markMuCurrent();
        } else {
            var.invalidateMu();
        }
    };
    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            copy_var(var_idx);
        }
    } else {
        for (int var_idx = 0; var_idx < w.num_vars; ++var_idx) {
            copy_var(var_idx);
        }
    }
}

}  // namespace slam
