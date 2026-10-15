#include "internal/se2_sync.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

namespace slam {

namespace {

using SteadyClock = std::chrono::steady_clock;

#if defined(_MSC_VER)
#define FASTLOCAL_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FASTLOCAL_FORCEINLINE inline __attribute__((always_inline))
#else
#define FASTLOCAL_FORCEINLINE inline
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

FASTLOCAL_FORCEINLINE bool factorizeSpd3x3Lower(
    double a00,
    double a10,
    double a20,
    double a11,
    double a21,
    double a22,
    Cholesky3x3& chol
) noexcept {
    if (!(a00 > 0.0)) return false;
    chol.l00 = std::sqrt(a00);

    chol.l10 = a10 / chol.l00;
    chol.l20 = a20 / chol.l00;

    const double d11 = a11 - chol.l10 * chol.l10;
    if (!(d11 > 0.0)) return false;
    chol.l11 = std::sqrt(d11);

    chol.l21 = (a21 - chol.l20 * chol.l10) / chol.l11;

    const double d22 = a22 - chol.l20 * chol.l20 - chol.l21 * chol.l21;
    if (!(d22 > 0.0)) return false;
    chol.l22 = std::sqrt(d22);
    return true;
}

FASTLOCAL_FORCEINLINE void solveSpd3x3(const Cholesky3x3& chol, const double* b, double* x) noexcept {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.l10 * y0) / chol.l11;
    const double y2 = (b[2] - chol.l20 * y0 - chol.l21 * y1) / chol.l22;

    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.l21 * x[2]) / chol.l11;
    x[0] = (y0 - chol.l10 * x[1] - chol.l20 * x[2]) / chol.l00;
}

FASTLOCAL_FORCEINLINE bool solveGeneral3x3LowerSym(
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

FASTLOCAL_FORCEINLINE bool invertSpd3x3Sym(
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

FASTLOCAL_FORCEINLINE double* vec3Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

FASTLOCAL_FORCEINLINE const double* vec3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

FASTLOCAL_FORCEINLINE double* sym6Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

FASTLOCAL_FORCEINLINE const double* sym6Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

FASTLOCAL_FORCEINLINE double* mat9Ptr(std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 9;
}

FASTLOCAL_FORCEINLINE const double* mat9Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 9;
}

FASTLOCAL_FORCEINLINE double* binaryMsgEtaPtr(SyntheticSE2FastSyncLocalLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTLOCAL_FORCEINLINE const double* binaryMsgEtaPtr(const SyntheticSE2FastSyncLocalLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTLOCAL_FORCEINLINE double* binaryMsgLamPtr(SyntheticSE2FastSyncLocalLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_lam.data() + static_cast<size_t>(slot_id) * 9;
}

FASTLOCAL_FORCEINLINE const double* binaryMsgLamPtr(const SyntheticSE2FastSyncLocalLayout& layout, int slot_id) noexcept {
    return layout.binary_msg_lam.data() + static_cast<size_t>(slot_id) * 9;
}

FASTLOCAL_FORCEINLINE void packSymLam3x3(const double* src9, double* dst6) noexcept {
    dst6[0] = src9[0];
    dst6[1] = src9[1];
    dst6[2] = src9[2];
    dst6[3] = src9[4];
    dst6[4] = src9[5];
    dst6[5] = src9[8];
}

FASTLOCAL_FORCEINLINE void copy3x3BlockPackedSym(const double* src6x6, int row0, int col0, double* dst6) noexcept {
    const int base = col0 * 6 + row0;
    dst6[0] = src6x6[base + 0];
    dst6[1] = src6x6[base + 1];
    dst6[2] = src6x6[base + 2];
    dst6[3] = src6x6[base + 7];
    dst6[4] = src6x6[base + 8];
    dst6[5] = src6x6[base + 14];
}

FASTLOCAL_FORCEINLINE void copy3x3BlockColMajor(const double* src6x6, int row0, int col0, double* dst9) noexcept {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            dst9[c * 3 + r] = src6x6[(col0 + c) * 6 + (row0 + r)];
        }
    }
}

FASTLOCAL_FORCEINLINE void schurMessage3x3NoDampingInverse(
    double eo0,
    double eo1,
    double eo2,
    double eno0,
    double eno1,
    double eno2,
    const double* loo,
    const double* lono,
    double s00,
    double s10,
    double s20,
    double s11,
    double s21,
    double s22,
    double* out_eta,
    double* out_lam
) noexcept {
    const double b00 = lono[0];
    const double b10 = lono[1];
    const double b20 = lono[2];
    const double b01 = lono[3];
    const double b11 = lono[4];
    const double b21 = lono[5];
    const double b02 = lono[6];
    const double b12 = lono[7];
    const double b22 = lono[8];

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

    out_lam[0] = loo[0] - (c00 * b00 + c01 * b01 + c02 * b02);
    out_lam[1] = loo[1] - (c10 * b00 + c11 * b01 + c12 * b02);
    out_lam[2] = loo[2] - (c20 * b00 + c21 * b01 + c22 * b02);
    out_lam[3] = out_lam[1];
    out_lam[4] = loo[4] - (c10 * b10 + c11 * b11 + c12 * b12);
    out_lam[5] = loo[5] - (c20 * b10 + c21 * b11 + c22 * b12);
    out_lam[6] = out_lam[2];
    out_lam[7] = out_lam[5];
    out_lam[8] = loo[8] - (c20 * b20 + c21 * b21 + c22 * b22);
}

FASTLOCAL_FORCEINLINE void computeUnaryFactorAtNoDamping(
    SyntheticSE2FastSyncLocalLayout& layout,
    int idx
) noexcept {
    gbp::Factor* factor = layout.unary_factors[idx];
    if (!factor || !factor->active) {
        return;
    }
    std::memcpy(layout.unary_msg_eta_ptrs[idx], vec3Ptr(layout.unary_eta, idx), 3 * sizeof(double));
    std::memcpy(layout.unary_msg_lam6_ptrs[idx], sym6Ptr(layout.unary_lam6, idx), 6 * sizeof(double));
}

FASTLOCAL_FORCEINLINE void computeBinaryFactorAtNoDamping(
    SyntheticSE2FastSyncLocalLayout& layout,
    int idx
) {
    gbp::Factor* factor = layout.binary_factors[idx];
    if (!factor || !factor->active) {
        return;
    }

    const int var0 = layout.binary_var0_ids[idx];
    const int var1 = layout.binary_var1_ids[idx];

    const double* belief0_eta = vec3Ptr(layout.belief_eta, var0);
    const double* belief1_eta = vec3Ptr(layout.belief_eta, var1);
    const double* belief0_lam6 = sym6Ptr(layout.belief_lam6, var0);
    const double* belief1_lam6 = sym6Ptr(layout.belief_lam6, var1);

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const double* old0_eta = binaryMsgEtaPtr(layout, slot0);
    const double* old1_eta = binaryMsgEtaPtr(layout, slot1);
    const double* old0_lam = binaryMsgLamPtr(layout, slot0);
    const double* old1_lam = binaryMsgLamPtr(layout, slot1);
    const double old0_eta0 = old0_eta[0];
    const double old0_eta1 = old0_eta[1];
    const double old0_eta2 = old0_eta[2];
    const double old1_eta0 = old1_eta[0];
    const double old1_eta1 = old1_eta[1];
    const double old1_eta2 = old1_eta[2];
    const double old0_lam0 = old0_lam[0];
    const double old0_lam1 = old0_lam[1];
    const double old0_lam2 = old0_lam[2];
    const double old0_lam4 = old0_lam[4];
    const double old0_lam5 = old0_lam[5];
    const double old0_lam8 = old0_lam[8];
    const double old1_lam0 = old1_lam[0];
    const double old1_lam1 = old1_lam[1];
    const double old1_lam2 = old1_lam[2];
    const double old1_lam4 = old1_lam[4];
    const double old1_lam5 = old1_lam[5];
    const double old1_lam8 = old1_lam[8];

    const double* eta0 = vec3Ptr(layout.eta0, idx);
    const double* eta1 = vec3Ptr(layout.eta1, idx);
    const double* block00 = mat9Ptr(layout.block00, idx);
    const double* block03 = mat9Ptr(layout.block03, idx);
    const double* block30 = mat9Ptr(layout.block30, idx);
    const double* block33 = mat9Ptr(layout.block33, idx);
    double* out0_eta = binaryMsgEtaPtr(layout, slot0);
    double* out0_lam = binaryMsgLamPtr(layout, slot0);
    double* out1_eta = binaryMsgEtaPtr(layout, slot1);
    double* out1_lam = binaryMsgLamPtr(layout, slot1);

    const double eno0_0 = eta1[0] + (belief1_eta[0] - old1_eta0);
    const double eno0_1 = eta1[1] + (belief1_eta[1] - old1_eta1);
    const double eno0_2 = eta1[2] + (belief1_eta[2] - old1_eta2);
    const double a0_00 = block33[0] + (belief1_lam6[0] - old1_lam0) + kJitter;
    const double a0_10 = block33[1] + (belief1_lam6[1] - old1_lam1);
    const double a0_20 = block33[2] + (belief1_lam6[2] - old1_lam2);
    const double a0_11 = block33[4] + (belief1_lam6[3] - old1_lam4) + kJitter;
    const double a0_21 = block33[5] + (belief1_lam6[4] - old1_lam5);
    const double a0_22 = block33[8] + (belief1_lam6[5] - old1_lam8) + kJitter;

    const double eno1_0 = eta0[0] + (belief0_eta[0] - old0_eta0);
    const double eno1_1 = eta0[1] + (belief0_eta[1] - old0_eta1);
    const double eno1_2 = eta0[2] + (belief0_eta[2] - old0_eta2);
    const double a1_00 = block00[0] + (belief0_lam6[0] - old0_lam0) + kJitter;
    const double a1_10 = block00[1] + (belief0_lam6[1] - old0_lam1);
    const double a1_20 = block00[2] + (belief0_lam6[2] - old0_lam2);
    const double a1_11 = block00[4] + (belief0_lam6[3] - old0_lam4) + kJitter;
    const double a1_21 = block00[5] + (belief0_lam6[4] - old0_lam5);
    const double a1_22 = block00[8] + (belief0_lam6[5] - old0_lam8) + kJitter;

    double s0_00, s0_10, s0_20, s0_11, s0_21, s0_22;
    if (!invertSpd3x3Sym(
            a0_00, a0_10, a0_20, a0_11, a0_21, a0_22,
            s0_00, s0_10, s0_20, s0_11, s0_21, s0_22
        )) {
        throw std::runtime_error("Inverse failed in SyntheticSE2FastSyncLocal factor target=0");
    }

    double s1_00, s1_10, s1_20, s1_11, s1_21, s1_22;
    if (!invertSpd3x3Sym(
            a1_00, a1_10, a1_20, a1_11, a1_21, a1_22,
            s1_00, s1_10, s1_20, s1_11, s1_21, s1_22
        )) {
        throw std::runtime_error("Inverse failed in SyntheticSE2FastSyncLocal factor target=1");
    }

    schurMessage3x3NoDampingInverse(
        eta0[0], eta0[1], eta0[2],
        eno0_0, eno0_1, eno0_2,
        block00, block03,
        s0_00, s0_10, s0_20, s0_11, s0_21, s0_22,
        out0_eta, out0_lam
    );

    schurMessage3x3NoDampingInverse(
        eta1[0], eta1[1], eta1[2],
        eno1_0, eno1_1, eno1_2,
        block33, block30,
        s1_00, s1_10, s1_20, s1_11, s1_21, s1_22,
        out1_eta, out1_lam
    );
}

FASTLOCAL_FORCEINLINE void updateBelief3DLocalNoMu(
    SyntheticSE2FastSyncLocalLayout& layout,
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

    const int bin_begin = layout.binary_offsets[var_idx];
    const int bin_end = layout.binary_offsets[var_idx + 1];
    for (int i = bin_begin; i < bin_end; ++i) {
        const int slot_id = layout.binary_slot_ids[i];
        const double* msg_eta = binaryMsgEtaPtr(layout, slot_id);
        const double* msg_lam = binaryMsgLamPtr(layout, slot_id);
        eta0 += msg_eta[0];
        eta1 += msg_eta[1];
        eta2 += msg_eta[2];
        lam00 += msg_lam[0];
        lam10 += msg_lam[1];
        lam20 += msg_lam[2];
        lam11 += msg_lam[4];
        lam21 += msg_lam[5];
        lam22 += msg_lam[8];
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

FASTLOCAL_FORCEINLINE void refreshMuAt(SyntheticSE2FastSyncLocalLayout& layout, int var_idx) {
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
            throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncLocal refreshMuAt");
        }
        layout.mu_valid[var_idx] = 1;
        return;
    }
    solveSpd3x3(chol, belief_eta, vec3Ptr(layout.mu, var_idx));
    layout.mu_valid[var_idx] = 1;
}

FASTLOCAL_FORCEINLINE void beliefEtaFromLamMu(const double* lam6, const double* mu, double* eta) noexcept {
    eta[0] = lam6[0] * mu[0] + lam6[1] * mu[1] + lam6[2] * mu[2];
    eta[1] = lam6[1] * mu[0] + lam6[3] * mu[1] + lam6[4] * mu[2];
    eta[2] = lam6[2] * mu[0] + lam6[4] * mu[1] + lam6[5] * mu[2];
}

}  // namespace

SyntheticSE2FastSyncLocalLayout buildSyntheticSE2FastSyncLocalLayout(gbp::FactorGraph& graph) {
    SyntheticSE2FastSyncLocalLayout layout;

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
            layout.binary_var0_ids.push_back(var_to_idx.at(factor->adj_var_nodes[0]));
            layout.binary_var1_ids.push_back(var_to_idx.at(factor->adj_var_nodes[1]));
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
    layout.prior_eta_ptrs.resize(num_vars, nullptr);
    layout.prior_lam6_ptrs.resize(num_vars, nullptr);
    layout.belief_eta_ptrs.resize(num_vars, nullptr);
    layout.belief_lam6_ptrs.resize(num_vars, nullptr);
    layout.mu_ptrs.resize(num_vars, nullptr);
    for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
        layout.prior_eta_ptrs[var_idx] = vec3Ptr(layout.prior_eta, var_idx);
        layout.prior_lam6_ptrs[var_idx] = sym6Ptr(layout.prior_lam6, var_idx);
        layout.belief_eta_ptrs[var_idx] = vec3Ptr(layout.belief_eta, var_idx);
        layout.belief_lam6_ptrs[var_idx] = sym6Ptr(layout.belief_lam6, var_idx);
        layout.mu_ptrs[var_idx] = vec3Ptr(layout.mu, var_idx);
    }

    const int num_unary = static_cast<int>(layout.unary_factors.size());
    layout.unary_eta.resize(static_cast<size_t>(num_unary) * 3, 0.0);
    layout.unary_lam6.resize(static_cast<size_t>(num_unary) * 6, 0.0);
    layout.unary_msg_eta.resize(static_cast<size_t>(num_unary) * 3, 0.0);
    layout.unary_msg_lam6.resize(static_cast<size_t>(num_unary) * 6, 0.0);
    layout.unary_msg_eta_ptrs.resize(num_unary, nullptr);
    layout.unary_msg_lam6_ptrs.resize(num_unary, nullptr);
    for (int idx = 0; idx < num_unary; ++idx) {
        layout.unary_msg_eta_ptrs[idx] = vec3Ptr(layout.unary_msg_eta, idx);
        layout.unary_msg_lam6_ptrs[idx] = sym6Ptr(layout.unary_msg_lam6, idx);
    }

    const int num_binary = static_cast<int>(layout.binary_factors.size());
    layout.eta0.resize(static_cast<size_t>(num_binary) * 3, 0.0);
    layout.eta1.resize(static_cast<size_t>(num_binary) * 3, 0.0);
    layout.block00.resize(static_cast<size_t>(num_binary) * 9, 0.0);
    layout.block03.resize(static_cast<size_t>(num_binary) * 9, 0.0);
    layout.block30.resize(static_cast<size_t>(num_binary) * 9, 0.0);
    layout.block33.resize(static_cast<size_t>(num_binary) * 9, 0.0);
    layout.binary_msg_eta.resize(static_cast<size_t>(num_binary) * 2 * 3, 0.0);
    layout.binary_msg_lam.resize(static_cast<size_t>(num_binary) * 2 * 9, 0.0);
    layout.binary_belief0_eta_ptrs.resize(num_binary, nullptr);
    layout.binary_belief1_eta_ptrs.resize(num_binary, nullptr);
    layout.binary_belief0_lam6_ptrs.resize(num_binary, nullptr);
    layout.binary_belief1_lam6_ptrs.resize(num_binary, nullptr);
    for (int idx = 0; idx < num_binary; ++idx) {
        layout.binary_belief0_eta_ptrs[idx] = layout.belief_eta_ptrs[layout.binary_var0_ids[idx]];
        layout.binary_belief1_eta_ptrs[idx] = layout.belief_eta_ptrs[layout.binary_var1_ids[idx]];
        layout.binary_belief0_lam6_ptrs[idx] = layout.belief_lam6_ptrs[layout.binary_var0_ids[idx]];
        layout.binary_belief1_lam6_ptrs[idx] = layout.belief_lam6_ptrs[layout.binary_var1_ids[idx]];
    }

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

    refreshSyntheticSE2FastSyncLocalLayout(layout);
    return layout;
}

void refreshSyntheticSE2FastSyncLocalLayout(SyntheticSE2FastSyncLocalLayout& layout) {
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
    std::fill(layout.binary_msg_lam.begin(), layout.binary_msg_lam.end(), 0.0);

    for (int idx = 0; idx < static_cast<int>(layout.unary_factors.size()); ++idx) {
        gbp::Factor* factor = layout.unary_factors[idx];
        std::memcpy(vec3Ptr(layout.unary_eta, idx), factor->factor.etaData(), 3 * sizeof(double));
        packSymLam3x3(factor->factor.lamData(), sym6Ptr(layout.unary_lam6, idx));
    }

    for (int idx = 0; idx < static_cast<int>(layout.binary_factors.size()); ++idx) {
        gbp::Factor* factor = layout.binary_factors[idx];
        const double* eta6 = factor->factor.etaData();
        const double* lam6x6 = factor->factor.lamData();
        double* eta0 = vec3Ptr(layout.eta0, idx);
        double* eta1 = vec3Ptr(layout.eta1, idx);
        eta0[0] = eta6[0];
        eta0[1] = eta6[1];
        eta0[2] = eta6[2];
        eta1[0] = eta6[3];
        eta1[1] = eta6[4];
        eta1[2] = eta6[5];
        copy3x3BlockColMajor(lam6x6, 0, 0, mat9Ptr(layout.block00, idx));
        copy3x3BlockColMajor(lam6x6, 0, 3, mat9Ptr(layout.block03, idx));
        copy3x3BlockColMajor(lam6x6, 3, 0, mat9Ptr(layout.block30, idx));
        copy3x3BlockColMajor(lam6x6, 3, 3, mat9Ptr(layout.block33, idx));
    }
}

void synchronousIterationFastSyncLocal(
    SyntheticSE2FastSyncLocalLayout& layout,
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

Eigen::VectorXd stackedMeanVectorFastSyncLocal(SyntheticSE2FastSyncLocalLayout& layout) {
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

void injectCorrectionKeepMessagesFastSyncLocal(
    SyntheticSE2FastSyncLocalLayout& layout,
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
