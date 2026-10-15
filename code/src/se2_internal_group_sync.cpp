#include "internal/se2_group_sync.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace slam {

namespace {

using SteadyClock = std::chrono::steady_clock;

#if defined(_MSC_VER)
#define FASTLOCAL_PACKED_FORCEINLINE __forceinline
#define FASTLOCAL_PACKED_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define FASTLOCAL_PACKED_FORCEINLINE inline __attribute__((always_inline))
#define FASTLOCAL_PACKED_RESTRICT __restrict__
#else
#define FASTLOCAL_PACKED_FORCEINLINE inline
#define FASTLOCAL_PACKED_RESTRICT
#endif

constexpr double kJitter = 1e-10;

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

struct Cholesky3x3 {
    double l00;
    double l10;
    double l20;
    double l11;
    double l21;
    double l22;
};

FASTLOCAL_PACKED_FORCEINLINE bool factorizeSpd3x3Lower(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    Cholesky3x3& chol
) noexcept {
    if (!(a00 > 0.0)) {
        return false;
    }
    chol.l00 = std::sqrt(a00);

    chol.l10 = a10 / chol.l00;
    chol.l20 = a20 / chol.l00;

    const double d11 = a11 - chol.l10 * chol.l10;
    if (!(d11 > 0.0)) {
        return false;
    }
    chol.l11 = std::sqrt(d11);

    chol.l21 = (a21 - chol.l20 * chol.l10) / chol.l11;

    const double d22 = a22 - chol.l20 * chol.l20 - chol.l21 * chol.l21;
    if (!(d22 > 0.0)) {
        return false;
    }
    chol.l22 = std::sqrt(d22);
    return true;
}

FASTLOCAL_PACKED_FORCEINLINE void solveSpd3x3(const Cholesky3x3& chol, const double* b, double* x) noexcept {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.l10 * y0) / chol.l11;
    const double y2 = (b[2] - chol.l20 * y0 - chol.l21 * y1) / chol.l22;

    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.l21 * x[2]) / chol.l11;
    x[0] = (y0 - chol.l10 * x[1] - chol.l20 * x[2]) / chol.l00;
}

FASTLOCAL_PACKED_FORCEINLINE bool solveGeneral3x3LowerSym(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    const double* b,
    double* x
) noexcept {
    Eigen::Matrix3d A;
    A << a00, a10, a20,
         a10, a11, a21,
         a20, a21, a22;
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

FASTLOCAL_PACKED_FORCEINLINE bool schurMessage3x3NoDampingGeneralPackedSym(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* loo6,
    double b00,
    double b10,
    double b20,
    double b01,
    double b11,
    double b21,
    double b02,
    double b12,
    double b22,
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double* out_eta,
    double* out_lam6
) noexcept {
    Eigen::Matrix3d A;
    A << a00, a10, a20,
         a10, a11, a21,
         a20, a21, a22;
    Eigen::FullPivLU<Eigen::Matrix3d> lu(A);
    if (!lu.isInvertible()) {
        return false;
    }

    Eigen::Matrix3d B;
    B << b00, b01, b02,
         b10, b11, b12,
         b20, b21, b22;
    const Eigen::Vector3d eno(eno0, eno1, eno2);
    const Eigen::Vector3d eo(eo0, eo1, eo2);
    const Eigen::Vector3d solved_eta = lu.solve(eno);
    const Eigen::Matrix3d solved_lam = lu.solve(B.transpose());
    const Eigen::Matrix3d loo = (Eigen::Matrix3d() <<
        loo6[0], loo6[1], loo6[2],
        loo6[1], loo6[3], loo6[4],
        loo6[2], loo6[4], loo6[5]).finished();
    const Eigen::Matrix3d schur = loo - B * solved_lam;
    const Eigen::Vector3d out_eta_vec = eo - B * solved_eta;

    out_eta[0] = out_eta_vec[0];
    out_eta[1] = out_eta_vec[1];
    out_eta[2] = out_eta_vec[2];
    out_lam6[0] = schur(0, 0);
    out_lam6[1] = schur(1, 0);
    out_lam6[2] = schur(2, 0);
    out_lam6[3] = schur(1, 1);
    out_lam6[4] = schur(2, 1);
    out_lam6[5] = schur(2, 2);
    return true;
}

FASTLOCAL_PACKED_FORCEINLINE bool invertSpd3x3Sym(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double& s00,
    double& s10,
    double& s20,
    double& s11,
    double& s21,
    double& s22
) noexcept {
    const double c00 = a11 * a22 - a21 * a21;
    const double c10 = a20 * a21 - a10 * a22;
    const double c20 = a10 * a21 - a20 * a11;
    const double c11 = a00 * a22 - a20 * a20;
    const double c21 = a10 * a20 - a00 * a21;
    const double c22 = a00 * a11 - a10 * a10;
    const double det = a00 * c00 + a10 * c10 + a20 * c20;
    if (!(det > 0.0)) {
        return false;
    }
    const double inv_det = 1.0 / det;
    s00 = c00 * inv_det;
    s10 = c10 * inv_det;
    s20 = c20 * inv_det;
    s11 = c11 * inv_det;
    s21 = c21 * inv_det;
    s22 = c22 * inv_det;
    return true;
}

FASTLOCAL_PACKED_FORCEINLINE bool schurMessage3x3NoDampingAdjugatePackedSym(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* FASTLOCAL_PACKED_RESTRICT loo6,
    double b00,
    double b10,
    double b20,
    double b01,
    double b11,
    double b21,
    double b02,
    double b12,
    double b22,
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    double* FASTLOCAL_PACKED_RESTRICT out_eta,
    double* FASTLOCAL_PACKED_RESTRICT out_lam6
) noexcept {
    const double loo0 = loo6[0];
    const double loo1 = loo6[1];
    const double loo2 = loo6[2];
    const double loo3 = loo6[3];
    const double loo4 = loo6[4];
    const double loo5 = loo6[5];

    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (!(det > 0.0)) {
        return false;
    }
    const double inv_det = 1.0 / det;

    const double t00 = b00 * cof00 + b01 * cof10 + b02 * cof20;
    const double t01 = b00 * cof10 + b01 * cof11 + b02 * cof21;
    const double t02 = b00 * cof20 + b01 * cof21 + b02 * cof22;
    const double t10 = b10 * cof00 + b11 * cof10 + b12 * cof20;
    const double t11 = b10 * cof10 + b11 * cof11 + b12 * cof21;
    const double t12 = b10 * cof20 + b11 * cof21 + b12 * cof22;
    const double t20 = b20 * cof00 + b21 * cof10 + b22 * cof20;
    const double t21 = b20 * cof10 + b21 * cof11 + b22 * cof21;
    const double t22 = b20 * cof20 + b21 * cof21 + b22 * cof22;

    out_eta[0] = eo0 - inv_det * (t00 * eno0 + t01 * eno1 + t02 * eno2);
    out_eta[1] = eo1 - inv_det * (t10 * eno0 + t11 * eno1 + t12 * eno2);
    out_eta[2] = eo2 - inv_det * (t20 * eno0 + t21 * eno1 + t22 * eno2);

    out_lam6[0] = loo0 - inv_det * (t00 * b00 + t01 * b01 + t02 * b02);
    out_lam6[1] = loo1 - inv_det * (t10 * b00 + t11 * b01 + t12 * b02);
    out_lam6[2] = loo2 - inv_det * (t20 * b00 + t21 * b01 + t22 * b02);
    out_lam6[3] = loo3 - inv_det * (t10 * b10 + t11 * b11 + t12 * b12);
    out_lam6[4] = loo4 - inv_det * (t20 * b10 + t21 * b11 + t22 * b12);
    out_lam6[5] = loo5 - inv_det * (t20 * b20 + t21 * b21 + t22 * b22);
    return true;
}


FASTLOCAL_PACKED_FORCEINLINE double* vec3Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

FASTLOCAL_PACKED_FORCEINLINE const double* vec3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

FASTLOCAL_PACKED_FORCEINLINE double* sym6Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

FASTLOCAL_PACKED_FORCEINLINE const double* sym6Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

FASTLOCAL_PACKED_FORCEINLINE double* binaryMsgEtaPtr(SyntheticSE2FastSyncLocalPackedLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTLOCAL_PACKED_FORCEINLINE const double* binaryMsgEtaPtr(const SyntheticSE2FastSyncLocalPackedLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTLOCAL_PACKED_FORCEINLINE double* binaryMsgLam6Ptr(SyntheticSE2FastSyncLocalPackedLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_lam6.data() + static_cast<size_t>(slot_id) * 6;
}

FASTLOCAL_PACKED_FORCEINLINE const double* binaryMsgLam6Ptr(const SyntheticSE2FastSyncLocalPackedLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_lam6.data() + static_cast<size_t>(slot_id) * 6;
}

FASTLOCAL_PACKED_FORCEINLINE void packSymLam3x3(const double* src9, double* dst6) noexcept {
    dst6[0] = src9[0];
    dst6[1] = src9[1];
    dst6[2] = src9[2];
    dst6[3] = src9[4];
    dst6[4] = src9[5];
    dst6[5] = src9[8];
}

FASTLOCAL_PACKED_FORCEINLINE void copy3x3BlockPackedSym(const double* src6x6, int row0, int col0, double* dst6) noexcept {
    const int base = col0 * 6 + row0;
    dst6[0] = src6x6[base + 0];
    dst6[1] = src6x6[base + 1];
    dst6[2] = src6x6[base + 2];
    dst6[3] = src6x6[base + 7];
    dst6[4] = src6x6[base + 8];
    dst6[5] = src6x6[base + 14];
}

FASTLOCAL_PACKED_FORCEINLINE void copy3x3BlockColMajor(const double* src6x6, int row0, int col0, double* dst9) noexcept {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            dst9[c * 3 + r] = src6x6[(col0 + c) * 6 + (row0 + r)];
        }
    }
}

FASTLOCAL_PACKED_FORCEINLINE void schurMessage3x3NoDampingInversePackedSym(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* loo6,
    double b00,
    double b10,
    double b20,
    double b01,
    double b11,
    double b21,
    double b02,
    double b12,
    double b22,
    double s00,
    double s10,
    double s20,
    double s11,
    double s21,
    double s22,
    double* out_eta,
    double* out_lam6
) noexcept {
    const double c00 = b00 * s00 + b01 * s10 + b02 * s20;
    const double c01 = b00 * s10 + b01 * s11 + b02 * s21;
    const double c02 = b00 * s20 + b01 * s21 + b02 * s22;
    const double c10 = b10 * s00 + b11 * s10 + b12 * s20;
    const double c11 = b10 * s10 + b11 * s11 + b12 * s21;
    const double c12 = b10 * s20 + b11 * s21 + b12 * s22;
    const double c20 = b20 * s00 + b21 * s10 + b22 * s20;
    const double c21 = b20 * s10 + b21 * s11 + b22 * s21;
    const double c22 = b20 * s20 + b21 * s21 + b22 * s22;

    out_eta[0] = eo0 - (c00 * eno0 + c01 * eno1 + c02 * eno2);
    out_eta[1] = eo1 - (c10 * eno0 + c11 * eno1 + c12 * eno2);
    out_eta[2] = eo2 - (c20 * eno0 + c21 * eno1 + c22 * eno2);

    out_lam6[0] = loo6[0] - (c00 * b00 + c01 * b01 + c02 * b02);
    out_lam6[1] = loo6[1] - (c10 * b00 + c11 * b01 + c12 * b02);
    out_lam6[2] = loo6[2] - (c20 * b00 + c21 * b01 + c22 * b02);
    out_lam6[3] = loo6[3] - (c10 * b10 + c11 * b11 + c12 * b12);
    out_lam6[4] = loo6[4] - (c20 * b10 + c21 * b11 + c22 * b12);
    out_lam6[5] = loo6[5] - (c20 * b20 + c21 * b21 + c22 * b22);
}

FASTLOCAL_PACKED_FORCEINLINE void computeUnaryFactorAtNoDamping(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    int idx
) noexcept {
    gbp::Factor* factor = layout.unary_factors[idx];
    if (!factor || !factor->active) {
        return;
    }
    std::memcpy(vec3Ptr(layout.unary_msg_eta, idx), vec3Ptr(layout.unary_eta, idx), 3 * sizeof(double));
    std::memcpy(sym6Ptr(layout.unary_msg_lam6, idx), sym6Ptr(layout.unary_lam6, idx), 6 * sizeof(double));
}

FASTLOCAL_PACKED_FORCEINLINE void computeBinaryFactorAtNoDamping(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    int idx
) {
    gbp::Factor* factor = layout.binary_factors[idx];
    if (!factor || !factor->active) {
        return;
    }

    const SyntheticSE2PackedBinaryFactorData& data = layout.binary_data[idx];
    const double* belief0_eta = vec3Ptr(layout.belief_eta, data.var0_id);
    const double* belief1_eta = vec3Ptr(layout.belief_eta, data.var1_id);
    const double* belief0_lam6 = sym6Ptr(layout.belief_lam6, data.var0_id);
    const double* belief1_lam6 = sym6Ptr(layout.belief_lam6, data.var1_id);

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;

    const double* old0_eta = binaryMsgEtaPtr(layout, slot0);
    const double* old1_eta = binaryMsgEtaPtr(layout, slot1);
    const double* old0_lam6 = binaryMsgLam6Ptr(layout, slot0);
    const double* old1_lam6 = binaryMsgLam6Ptr(layout, slot1);
    double* out0_eta = binaryMsgEtaPtr(layout, slot0);
    double* out1_eta = binaryMsgEtaPtr(layout, slot1);
    double* out0_lam6 = binaryMsgLam6Ptr(layout, slot0);
    double* out1_lam6 = binaryMsgLam6Ptr(layout, slot1);

    const double eno0_0 = data.eta1[0] + (belief1_eta[0] - old1_eta[0]);
    const double eno0_1 = data.eta1[1] + (belief1_eta[1] - old1_eta[1]);
    const double eno0_2 = data.eta1[2] + (belief1_eta[2] - old1_eta[2]);
    const double a0_00 = data.diag1_lam6[0] + (belief1_lam6[0] - old1_lam6[0]) + kJitter;
    const double a0_10 = data.diag1_lam6[1] + (belief1_lam6[1] - old1_lam6[1]);
    const double a0_20 = data.diag1_lam6[2] + (belief1_lam6[2] - old1_lam6[2]);
    const double a0_11 = data.diag1_lam6[3] + (belief1_lam6[3] - old1_lam6[3]) + kJitter;
    const double a0_21 = data.diag1_lam6[4] + (belief1_lam6[4] - old1_lam6[4]);
    const double a0_22 = data.diag1_lam6[5] + (belief1_lam6[5] - old1_lam6[5]) + kJitter;

    const double eno1_0 = data.eta0[0] + (belief0_eta[0] - old0_eta[0]);
    const double eno1_1 = data.eta0[1] + (belief0_eta[1] - old0_eta[1]);
    const double eno1_2 = data.eta0[2] + (belief0_eta[2] - old0_eta[2]);
    const double a1_00 = data.diag0_lam6[0] + (belief0_lam6[0] - old0_lam6[0]) + kJitter;
    const double a1_10 = data.diag0_lam6[1] + (belief0_lam6[1] - old0_lam6[1]);
    const double a1_20 = data.diag0_lam6[2] + (belief0_lam6[2] - old0_lam6[2]);
    const double a1_11 = data.diag0_lam6[3] + (belief0_lam6[3] - old0_lam6[3]) + kJitter;
    const double a1_21 = data.diag0_lam6[4] + (belief0_lam6[4] - old0_lam6[4]);
    const double a1_22 = data.diag0_lam6[5] + (belief0_lam6[5] - old0_lam6[5]) + kJitter;

    const double* cross = data.cross01_lam9;
    const double b00 = cross[0];
    const double b10 = cross[1];
    const double b20 = cross[2];
    const double b01 = cross[3];
    const double b11 = cross[4];
    const double b21 = cross[5];
    const double b02 = cross[6];
    const double b12 = cross[7];
    const double b22 = cross[8];

    if (!schurMessage3x3NoDampingAdjugatePackedSym(
            data.eta0[0], data.eta0[1], data.eta0[2],
            eno0_0, eno0_1, eno0_2,
            data.diag0_lam6,
            b00, b10, b20, b01, b11, b21, b02, b12, b22,
            a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
            out0_eta, out0_lam6
        )) {
        if (!schurMessage3x3NoDampingGeneralPackedSym(
                data.eta0[0], data.eta0[1], data.eta0[2],
                eno0_0, eno0_1, eno0_2,
                data.diag0_lam6,
                b00, b10, b20, b01, b11, b21, b02, b12, b22,
                a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
                out0_eta, out0_lam6
            )) {
            throw std::runtime_error("Linear Schur solve failed in SyntheticSE2FastSyncLocalPacked factor target=0");
        }
    }

    if (!schurMessage3x3NoDampingAdjugatePackedSym(
            data.eta1[0], data.eta1[1], data.eta1[2],
            eno1_0, eno1_1, eno1_2,
            data.diag1_lam6,
            b00, b01, b02, b10, b11, b12, b20, b21, b22,
            a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
            out1_eta, out1_lam6
        )) {
        if (!schurMessage3x3NoDampingGeneralPackedSym(
                data.eta1[0], data.eta1[1], data.eta1[2],
                eno1_0, eno1_1, eno1_2,
                data.diag1_lam6,
                b00, b01, b02, b10, b11, b12, b20, b21, b22,
                a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
                out1_eta, out1_lam6
            )) {
            throw std::runtime_error("Linear Schur solve failed in SyntheticSE2FastSyncLocalPacked factor target=1");
        }
    }
}

FASTLOCAL_PACKED_FORCEINLINE void updateBelief3DLocalNoMu(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    int var_idx
) noexcept {
    const double* prior_eta = vec3Ptr(layout.prior_eta, var_idx);
    const double* prior_lam6 = sym6Ptr(layout.prior_lam6, var_idx);

    double eta0 = prior_eta[0];
    double eta1 = prior_eta[1];
    double eta2 = prior_eta[2];
    double lam00 = prior_lam6[0];
    double lam10 = prior_lam6[1];
    double lam20 = prior_lam6[2];
    double lam11 = prior_lam6[3];
    double lam21 = prior_lam6[4];
    double lam22 = prior_lam6[5];

    const int unary_begin = layout.unary_offsets[var_idx];
    const int unary_end = layout.unary_offsets[var_idx + 1];
    for (int i = unary_begin; i < unary_end; ++i) {
        const int unary_id = layout.unary_ids[i];
        const double* msg_eta = vec3Ptr(layout.unary_msg_eta, unary_id);
        const double* msg_lam6 = sym6Ptr(layout.unary_msg_lam6, unary_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
        lam00 += msg_lam6[0];
        lam10 += msg_lam6[1];
        lam20 += msg_lam6[2];
        lam11 += msg_lam6[3];
        lam21 += msg_lam6[4];
        lam22 += msg_lam6[5];
    }

    const int binary_begin = layout.binary_offsets[var_idx];
    const int binary_end = layout.binary_offsets[var_idx + 1];
    for (int i = binary_begin; i < binary_end; ++i) {
        const int slot_id = layout.binary_slot_ids[i];
        const double* msg_eta = binaryMsgEtaPtr(layout, slot_id);
        const double* msg_lam6 = binaryMsgLam6Ptr(layout, slot_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
        lam00 += msg_lam6[0];
        lam10 += msg_lam6[1];
        lam20 += msg_lam6[2];
        lam11 += msg_lam6[3];
        lam21 += msg_lam6[4];
        lam22 += msg_lam6[5];
    }

    double* belief_eta = vec3Ptr(layout.belief_eta, var_idx);
    double* belief_lam6 = sym6Ptr(layout.belief_lam6, var_idx);
    belief_eta[0] = eta0;
    belief_eta[1] = eta1;
    belief_eta[2] = eta2;
    belief_lam6[0] = lam00;
    belief_lam6[1] = lam10;
    belief_lam6[2] = lam20;
    belief_lam6[3] = lam11;
    belief_lam6[4] = lam21;
    belief_lam6[5] = lam22;
    layout.mu_valid[var_idx] = 0;
}

FASTLOCAL_PACKED_FORCEINLINE void refreshMuAt(SyntheticSE2FastSyncLocalPackedLayout& layout, int var_idx) {
    if (layout.mu_valid[var_idx]) {
        return;
    }
    const double* belief_eta = vec3Ptr(layout.belief_eta, var_idx);
    const double* belief_lam6 = sym6Ptr(layout.belief_lam6, var_idx);
    Cholesky3x3 chol;
    if (!factorizeSpd3x3Lower(
            belief_lam6[0], belief_lam6[1], belief_lam6[2],
            belief_lam6[3], belief_lam6[4], belief_lam6[5],
            chol
        )) {
        if (!solveGeneral3x3LowerSym(
                belief_lam6[0], belief_lam6[1], belief_lam6[2],
                belief_lam6[3], belief_lam6[4], belief_lam6[5],
                belief_eta,
                vec3Ptr(layout.mu, var_idx)
            )) {
            throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncLocalPacked refreshMuAt");
        }
        layout.mu_valid[var_idx] = 1;
        return;
    }
    solveSpd3x3(chol, belief_eta, vec3Ptr(layout.mu, var_idx));
    layout.mu_valid[var_idx] = 1;
}

FASTLOCAL_PACKED_FORCEINLINE void beliefEtaFromLamMu(const double* lam6, const double* mu, double* eta) noexcept {
    eta[0] = lam6[0] * mu[0] + lam6[1] * mu[1] + lam6[2] * mu[2];
    eta[1] = lam6[1] * mu[0] + lam6[3] * mu[1] + lam6[4] * mu[2];
    eta[2] = lam6[2] * mu[0] + lam6[4] * mu[1] + lam6[5] * mu[2];
}

}  // namespace

SyntheticSE2FastSyncLocalPackedLayout buildSyntheticSE2FastSyncLocalPackedLayout(gbp::FactorGraph& graph) {
    SyntheticSE2FastSyncLocalPackedLayout layout;

    std::unordered_map<gbp::VariableNode*, int> var_to_idx;
    var_to_idx.reserve(graph.var_nodes.size());
    for (int i = 0; i < static_cast<int>(graph.var_nodes.size()); ++i) {
        gbp::VariableNode* var = graph.var_nodes[i].get();
        layout.variables.push_back(var);
        if (!var || var->dofs != 3) {
            layout.supported = false;
            return layout;
        }
        var_to_idx.emplace(var, i);
    }

    std::unordered_map<gbp::Factor*, int> unary_factor_to_idx;
    std::unordered_map<gbp::Factor*, int> binary_factor_to_idx;
    unary_factor_to_idx.reserve(graph.factors.size());
    binary_factor_to_idx.reserve(graph.factors.size());

    for (auto& fup : graph.factors) {
        gbp::Factor* factor = fup.get();
        if (!factor) {
            continue;
        }
        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity == 1 &&
            factor->adj_var_nodes[0] != nullptr &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->factor.dim() == 3 &&
            factor->messages.size() == 1) {
            const int idx = static_cast<int>(layout.unary_factors.size());
            layout.unary_factors.push_back(factor);
            layout.unary_var_ids.push_back(var_to_idx.at(factor->adj_var_nodes[0]));
            unary_factor_to_idx.emplace(factor, idx);
            continue;
        }
        if (arity == 2 &&
            factor->adj_var_nodes[0] != nullptr &&
            factor->adj_var_nodes[1] != nullptr &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6 &&
            factor->messages.size() == 2) {
            const int idx = static_cast<int>(layout.binary_factors.size());
            layout.binary_factors.push_back(factor);
            layout.binary_data.push_back(SyntheticSE2PackedBinaryFactorData{});
            layout.binary_data.back().var0_id = var_to_idx.at(factor->adj_var_nodes[0]);
            layout.binary_data.back().var1_id = var_to_idx.at(factor->adj_var_nodes[1]);
            binary_factor_to_idx.emplace(factor, idx);
            continue;
        }
        layout.supported = false;
        return layout;
    }

    layout.supported = true;

    const int num_vars = static_cast<int>(layout.variables.size());
    layout.prior_eta.resize(static_cast<size_t>(num_vars) * 3, 0.0);
    layout.prior_lam6.resize(static_cast<size_t>(num_vars) * 6, 0.0);
    layout.belief_eta.resize(static_cast<size_t>(num_vars) * 3, 0.0);
    layout.belief_lam6.resize(static_cast<size_t>(num_vars) * 6, 0.0);
    layout.mu.resize(static_cast<size_t>(num_vars) * 3, 0.0);
    layout.mu_valid.resize(num_vars, 0);

    const int num_unary = static_cast<int>(layout.unary_factors.size());
    layout.unary_eta.resize(static_cast<size_t>(num_unary) * 3, 0.0);
    layout.unary_lam6.resize(static_cast<size_t>(num_unary) * 6, 0.0);
    layout.unary_msg_eta.resize(static_cast<size_t>(num_unary) * 3, 0.0);
    layout.unary_msg_lam6.resize(static_cast<size_t>(num_unary) * 6, 0.0);

    const int num_binary = static_cast<int>(layout.binary_factors.size());
    layout.binary_msg_eta.resize(static_cast<size_t>(num_binary) * 2 * 3, 0.0);
    layout.binary_msg_lam6.resize(static_cast<size_t>(num_binary) * 2 * 6, 0.0);

    layout.unary_offsets.reserve(num_vars + 1);
    layout.binary_offsets.reserve(num_vars + 1);
    int unary_offset = 0;
    int binary_offset = 0;
    for (gbp::VariableNode* var : layout.variables) {
        layout.unary_offsets.push_back(unary_offset);
        layout.binary_offsets.push_back(binary_offset);
        for (const auto& aref : var->adj_factors) {
            auto unary_it = unary_factor_to_idx.find(aref.factor);
            if (unary_it != unary_factor_to_idx.end()) {
                layout.unary_ids.push_back(unary_it->second);
                ++unary_offset;
                continue;
            }
            auto binary_it = binary_factor_to_idx.find(aref.factor);
            if (binary_it != binary_factor_to_idx.end()) {
                layout.binary_slot_ids.push_back(2 * binary_it->second + aref.local_idx);
                ++binary_offset;
                continue;
            }
            layout.supported = false;
            return layout;
        }
    }
    layout.unary_offsets.push_back(unary_offset);
    layout.binary_offsets.push_back(binary_offset);

    refreshSyntheticSE2FastSyncLocalPackedLayout(layout);
    return layout;
}

void refreshSyntheticSE2FastSyncLocalPackedLayout(SyntheticSE2FastSyncLocalPackedLayout& layout) {
    const int num_vars = static_cast<int>(layout.variables.size());
    for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
        gbp::VariableNode* var = layout.variables[var_idx];
        packSymLam3x3(var->prior.lamData(), sym6Ptr(layout.prior_lam6, var_idx));
        std::memcpy(vec3Ptr(layout.prior_eta, var_idx), var->prior.etaData(), 3 * sizeof(double));
        std::memcpy(vec3Ptr(layout.belief_eta, var_idx), vec3Ptr(layout.prior_eta, var_idx), 3 * sizeof(double));
        std::memcpy(sym6Ptr(layout.belief_lam6, var_idx), sym6Ptr(layout.prior_lam6, var_idx), 6 * sizeof(double));
        std::memset(vec3Ptr(layout.mu, var_idx), 0, 3 * sizeof(double));
        layout.mu_valid[var_idx] = 0;
    }

    std::fill(layout.unary_msg_eta.begin(), layout.unary_msg_eta.end(), 0.0);
    std::fill(layout.unary_msg_lam6.begin(), layout.unary_msg_lam6.end(), 0.0);
    std::fill(layout.binary_msg_eta.begin(), layout.binary_msg_eta.end(), 0.0);
    std::fill(layout.binary_msg_lam6.begin(), layout.binary_msg_lam6.end(), 0.0);

    for (int idx = 0; idx < static_cast<int>(layout.unary_factors.size()); ++idx) {
        gbp::Factor* factor = layout.unary_factors[idx];
        std::memcpy(vec3Ptr(layout.unary_eta, idx), factor->factor.etaData(), 3 * sizeof(double));
        packSymLam3x3(factor->factor.lamData(), sym6Ptr(layout.unary_lam6, idx));
    }

    for (int idx = 0; idx < static_cast<int>(layout.binary_factors.size()); ++idx) {
        gbp::Factor* factor = layout.binary_factors[idx];
        const double* eta6 = factor->factor.etaData();
        const double* lam6x6 = factor->factor.lamData();
        SyntheticSE2PackedBinaryFactorData& data = layout.binary_data[idx];
        data.eta0[0] = eta6[0];
        data.eta0[1] = eta6[1];
        data.eta0[2] = eta6[2];
        data.eta1[0] = eta6[3];
        data.eta1[1] = eta6[4];
        data.eta1[2] = eta6[5];
        copy3x3BlockPackedSym(lam6x6, 0, 0, data.diag0_lam6);
        copy3x3BlockPackedSym(lam6x6, 3, 3, data.diag1_lam6);
        copy3x3BlockColMajor(lam6x6, 0, 3, data.cross01_lam9);
    }
}

void synchronousIterationFastSyncLocalPacked(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum
) {
    const auto factor_t0 = SteadyClock::now();
    for (int idx = 0; idx < static_cast<int>(layout.unary_factors.size()); ++idx) {
        computeUnaryFactorAtNoDamping(layout, idx);
    }
    for (int idx = 0; idx < static_cast<int>(layout.binary_factors.size()); ++idx) {
        computeBinaryFactorAtNoDamping(layout, idx);
    }
    const auto factor_t1 = SteadyClock::now();

    const auto var_t0 = SteadyClock::now();
    for (int var_idx = 0; var_idx < static_cast<int>(layout.variables.size()); ++var_idx) {
        updateBelief3DLocalNoMu(layout, var_idx);
    }
    const auto var_t1 = SteadyClock::now();

    factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
    variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
}

Eigen::VectorXd stackedMeanVectorFastSyncLocalPacked(SyntheticSE2FastSyncLocalPackedLayout& layout) {
    const int num_vars = static_cast<int>(layout.variables.size());
    Eigen::VectorXd out = Eigen::VectorXd::Zero(num_vars * 3);
    double* out_data = out.data();
    for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
        refreshMuAt(layout, var_idx);
        const double* mu = vec3Ptr(layout.mu, var_idx);
        out_data[3 * var_idx + 0] = mu[0];
        out_data[3 * var_idx + 1] = mu[1];
        out_data[3 * var_idx + 2] = mu[2];
    }
    return out;
}

void injectCorrectionKeepMessagesFastSyncLocalPacked(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    const Eigen::VectorXd& delta
) {
    const int num_vars = static_cast<int>(layout.variables.size());
    const double* delta_data = delta.data();
    for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
        refreshMuAt(layout, var_idx);
        double* mu = vec3Ptr(layout.mu, var_idx);
        const double* lam6 = sym6Ptr(layout.belief_lam6, var_idx);
        double* eta = vec3Ptr(layout.belief_eta, var_idx);
        mu[0] += delta_data[3 * var_idx + 0];
        mu[1] += delta_data[3 * var_idx + 1];
        mu[2] += delta_data[3 * var_idx + 2];
        beliefEtaFromLamMu(lam6, mu, eta);
        layout.mu_valid[var_idx] = 1;
    }
}

}  // namespace slam
