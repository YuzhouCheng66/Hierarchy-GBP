#include "internal/se2_dense_sync.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include <omp.h>

namespace slam {

namespace {

using SteadyClock = std::chrono::steady_clock;

#if defined(_MSC_VER)
#define FASTSYNC_FORCEINLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define FASTSYNC_FORCEINLINE inline __attribute__((always_inline))
#else
#define FASTSYNC_FORCEINLINE inline
#endif

double elapsedSeconds(const SteadyClock::time_point& start, const SteadyClock::time_point& end) {
    return std::chrono::duration<double>(end - start).count();
}

struct Cholesky3x3 {
    double l00;
    double u01;
    double u02;
    double l11;
    double u12;
    double l22;
};

constexpr double kJitter = 1e-10;

inline int effectiveThreads(int num_threads) {
    return num_threads > 0 ? num_threads : omp_get_max_threads();
}

void buildContiguousWeightedOffsets(
    int n_items,
    const std::vector<double>& rates,
    std::vector<int>& offsets
) {
    const int thread_count = static_cast<int>(rates.size());
    offsets.assign(thread_count + 1, 0);
    if (thread_count == 0 || n_items <= 0) {
        return;
    }

    double total_rate = 0.0;
    for (double rate : rates) {
        total_rate += std::max(rate, 1e-6);
    }
    if (!(total_rate > 0.0)) {
        total_rate = static_cast<double>(thread_count);
    }

    double prefix = 0.0;
    offsets[0] = 0;
    for (int tid = 0; tid < thread_count - 1; ++tid) {
        prefix += std::max(rates[tid], 1e-6);
        int next = static_cast<int>(std::llround(prefix / total_rate * n_items));
        next = std::max(next, offsets[tid]);
        next = std::min(next, n_items);
        offsets[tid + 1] = next;
    }
    offsets[thread_count] = n_items;
    for (int tid = 1; tid <= thread_count; ++tid) {
        offsets[tid] = std::max(offsets[tid], offsets[tid - 1]);
    }
}

void updateThroughputEma(
    const std::vector<int>& offsets,
    const std::vector<double>& work_times,
    std::vector<double>& rates
) {
    const int thread_count = static_cast<int>(rates.size());
    const double alpha = 0.6;
    for (int tid = 0; tid < thread_count; ++tid) {
        const int count = offsets[tid + 1] - offsets[tid];
        const double work = work_times[tid];
        if (count <= 0 || !(work > 0.0)) {
            continue;
        }
        const double inst_rate = static_cast<double>(count) / work;
        rates[tid] = alpha * rates[tid] + (1.0 - alpha) * inst_rate;
    }
}

FASTSYNC_FORCEINLINE bool factorizeSpd3x3Upper(
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

FASTSYNC_FORCEINLINE void solveSpd3x3(const Cholesky3x3& chol, const double* b, double* x) noexcept {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.u01 * y0) / chol.l11;
    const double y2 = (b[2] - chol.u02 * y0 - chol.u12 * y1) / chol.l22;

    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.u12 * x[2]) / chol.l11;
    x[0] = (y0 - chol.u01 * x[1] - chol.u02 * x[2]) / chol.l00;
}

FASTSYNC_FORCEINLINE bool solveGeneral3x3UpperSym(
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

FASTSYNC_FORCEINLINE bool solveGeneral3x3(
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

FASTSYNC_FORCEINLINE void schurMessage3x3(
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
    double y[3];
    solveSpd3x3(chol, eno, y);

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
        double y_col[3];
        solveSpd3x3(chol, lnoo + 3 * col, y_col);
        const double proj0 = lono[0] * y_col[0] + lono[3] * y_col[1] + lono[6] * y_col[2];
        const double proj1 = lono[1] * y_col[0] + lono[4] * y_col[1] + lono[7] * y_col[2];
        const double proj2 = lono[2] * y_col[0] + lono[5] * y_col[1] + lono[8] * y_col[2];
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

void copy3x3BlockColMajor(const double* src6x6, int row0, int col0, double* dst3x3) noexcept {
    for (int c = 0; c < 3; ++c) {
        for (int r = 0; r < 3; ++r) {
            dst3x3[c * 3 + r] = src6x6[(col0 + c) * 6 + (row0 + r)];
        }
    }
}

FASTSYNC_FORCEINLINE double* msgEtaPtr(SyntheticSE2FastSyncSoALayout& layout, int slot_id) noexcept {
    return layout.msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTSYNC_FORCEINLINE const double* msgEtaPtr(const SyntheticSE2FastSyncSoALayout& layout, int slot_id) noexcept {
    return layout.msg_eta.data() + static_cast<size_t>(slot_id) * 3;
}

FASTSYNC_FORCEINLINE double* msgLamPtr(SyntheticSE2FastSyncSoALayout& layout, int slot_id) noexcept {
    return layout.msg_lam.data() + static_cast<size_t>(slot_id) * 9;
}

FASTSYNC_FORCEINLINE const double* msgLamPtr(const SyntheticSE2FastSyncSoALayout& layout, int slot_id) noexcept {
    return layout.msg_lam.data() + static_cast<size_t>(slot_id) * 9;
}

FASTSYNC_FORCEINLINE const double* vec3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

FASTSYNC_FORCEINLINE const double* mat3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 9;
}

void refreshBinaryFactorAt(SyntheticSE2FastSyncSoALayout& layout, int idx) {
    gbp::Factor* factor = layout.binary_factors[idx];
    const double* eta6 = factor->factor.etaData();
    const double* lam6 = factor->factor.lamData();

    double* eta0 = layout.eta0.data() + static_cast<size_t>(idx) * 3;
    double* eta1 = layout.eta1.data() + static_cast<size_t>(idx) * 3;
    eta0[0] = eta6[0];
    eta0[1] = eta6[1];
    eta0[2] = eta6[2];
    eta1[0] = eta6[3];
    eta1[1] = eta6[4];
    eta1[2] = eta6[5];

    copy3x3BlockColMajor(lam6, 0, 0, layout.block00.data() + static_cast<size_t>(idx) * 9);
    copy3x3BlockColMajor(lam6, 0, 3, layout.block03.data() + static_cast<size_t>(idx) * 9);
    copy3x3BlockColMajor(lam6, 3, 0, layout.block30.data() + static_cast<size_t>(idx) * 9);
    copy3x3BlockColMajor(lam6, 3, 3, layout.block33.data() + static_cast<size_t>(idx) * 9);
}

void computeBinary3FactorAt(
    SyntheticSE2FastSyncSoALayout& layout,
    int idx,
    double eta_damping
) {
    gbp::Factor* factor = layout.binary_factors[idx];
    if (!factor->active) {
        return;
    }

    const bool no_damping = (eta_damping == 0.0);
    const double* belief0_eta = layout.belief0_eta_ptrs[idx];
    const double* belief1_eta = layout.belief1_eta_ptrs[idx];
    const double* belief0_lam = layout.belief0_lam_ptrs[idx];
    const double* belief1_lam = layout.belief1_lam_ptrs[idx];

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const double* old0_eta = msgEtaPtr(layout, slot0);
    const double* old1_eta = msgEtaPtr(layout, slot1);
    const double* old0_lam = msgLamPtr(layout, slot0);
    const double* old1_lam = msgLamPtr(layout, slot1);

    const double* eta0 = vec3Ptr(layout.eta0, idx);
    const double* eta1 = vec3Ptr(layout.eta1, idx);
    const double* block00 = mat3Ptr(layout.block00, idx);
    const double* block03 = mat3Ptr(layout.block03, idx);
    const double* block30 = mat3Ptr(layout.block30, idx);
    const double* block33 = mat3Ptr(layout.block33, idx);

    double out0_eta[3];
    double out0_lam[9];
    {
        double eno[3] = {
            eta1[0] + (belief1_eta[0] - old1_eta[0]),
            eta1[1] + (belief1_eta[1] - old1_eta[1]),
            eta1[2] + (belief1_eta[2] - old1_eta[2]),
        };
        Cholesky3x3 chol;
        if (!factorizeSpd3x3Upper(
                block33[0] + (belief1_lam[0] - old1_lam[0]) + kJitter,
                block33[3] + (belief1_lam[3] - old1_lam[3]),
                block33[6] + (belief1_lam[6] - old1_lam[6]),
                block33[4] + (belief1_lam[4] - old1_lam[4]) + kJitter,
                block33[7] + (belief1_lam[7] - old1_lam[7]),
                block33[8] + (belief1_lam[8] - old1_lam[8]) + kJitter,
                chol
            )) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            A << block33[0] + (belief1_lam[0] - old1_lam[0]) + kJitter,
                 block33[3] + (belief1_lam[3] - old1_lam[3]),
                 block33[6] + (belief1_lam[6] - old1_lam[6]),
                 block33[3] + (belief1_lam[3] - old1_lam[3]),
                 block33[4] + (belief1_lam[4] - old1_lam[4]) + kJitter,
                 block33[7] + (belief1_lam[7] - old1_lam[7]),
                 block33[6] + (belief1_lam[6] - old1_lam[6]),
                 block33[7] + (belief1_lam[7] - old1_lam[7]),
                 block33[8] + (belief1_lam[8] - old1_lam[8]) + kJitter;
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lnoo_mat(block30);
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lono_mat(block03);
            const Eigen::Map<const Eigen::Matrix<double, 3, 1>> eo_vec(eta0);
            const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
            Eigen::Matrix3d solved_lam;
            Eigen::Vector3d solved_eta;
            if (!solveGeneral3x3(A, lnoo_mat, eno_vec, solved_lam, solved_eta)) {
                throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncSoA factor target=0");
            }
            const Eigen::Vector3d out_eta_vec = eo_vec - lono_mat * solved_eta;
            const Eigen::Matrix3d out_lam_mat = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(block00) - lono_mat * solved_lam;
            if (no_damping) {
                std::memcpy(out0_eta, out_eta_vec.data(), 3 * sizeof(double));
                std::memcpy(out0_lam, out_lam_mat.data(), 9 * sizeof(double));
            } else {
                const double s = 1.0 - eta_damping;
                for (int k = 0; k < 3; ++k) {
                    out0_eta[k] = s * out_eta_vec[k] + eta_damping * old0_eta[k];
                }
                for (int k = 0; k < 9; ++k) {
                    out0_lam[k] = s * out_lam_mat.data()[k] + eta_damping * old0_lam[k];
                }
            }
            goto target0_done;
        }
        schurMessage3x3(chol, eta0, eno, block00, block03, block30, old0_eta, old0_lam, eta_damping, no_damping, out0_eta, out0_lam);
    target0_done:;
    }

    double out1_eta[3];
    double out1_lam[9];
    {
        double eno[3] = {
            eta0[0] + (belief0_eta[0] - old0_eta[0]),
            eta0[1] + (belief0_eta[1] - old0_eta[1]),
            eta0[2] + (belief0_eta[2] - old0_eta[2]),
        };
        Cholesky3x3 chol;
        if (!factorizeSpd3x3Upper(
                block00[0] + (belief0_lam[0] - old0_lam[0]) + kJitter,
                block00[3] + (belief0_lam[3] - old0_lam[3]),
                block00[6] + (belief0_lam[6] - old0_lam[6]),
                block00[4] + (belief0_lam[4] - old0_lam[4]) + kJitter,
                block00[7] + (belief0_lam[7] - old0_lam[7]),
                block00[8] + (belief0_lam[8] - old0_lam[8]) + kJitter,
                chol
            )) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            A << block00[0] + (belief0_lam[0] - old0_lam[0]) + kJitter,
                 block00[3] + (belief0_lam[3] - old0_lam[3]),
                 block00[6] + (belief0_lam[6] - old0_lam[6]),
                 block00[3] + (belief0_lam[3] - old0_lam[3]),
                 block00[4] + (belief0_lam[4] - old0_lam[4]) + kJitter,
                 block00[7] + (belief0_lam[7] - old0_lam[7]),
                 block00[6] + (belief0_lam[6] - old0_lam[6]),
                 block00[7] + (belief0_lam[7] - old0_lam[7]),
                 block00[8] + (belief0_lam[8] - old0_lam[8]) + kJitter;
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lnoo_mat(block03);
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lono_mat(block30);
            const Eigen::Map<const Eigen::Matrix<double, 3, 1>> eo_vec(eta1);
            const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
            Eigen::Matrix3d solved_lam;
            Eigen::Vector3d solved_eta;
            if (!solveGeneral3x3(A, lnoo_mat, eno_vec, solved_lam, solved_eta)) {
                throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncSoA factor target=1");
            }
            const Eigen::Vector3d out_eta_vec = eo_vec - lono_mat * solved_eta;
            const Eigen::Matrix3d out_lam_mat = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(block33) - lono_mat * solved_lam;
            if (no_damping) {
                std::memcpy(out1_eta, out_eta_vec.data(), 3 * sizeof(double));
                std::memcpy(out1_lam, out_lam_mat.data(), 9 * sizeof(double));
            } else {
                const double s = 1.0 - eta_damping;
                for (int k = 0; k < 3; ++k) {
                    out1_eta[k] = s * out_eta_vec[k] + eta_damping * old1_eta[k];
                }
                for (int k = 0; k < 9; ++k) {
                    out1_lam[k] = s * out_lam_mat.data()[k] + eta_damping * old1_lam[k];
                }
            }
            goto target1_done;
        }
        schurMessage3x3(chol, eta1, eno, block33, block30, block03, old1_eta, old1_lam, eta_damping, no_damping, out1_eta, out1_lam);
    target1_done:;
    }

    std::memcpy(msgEtaPtr(layout, slot0), out0_eta, 3 * sizeof(double));
    std::memcpy(msgLamPtr(layout, slot0), out0_lam, 9 * sizeof(double));
    std::memcpy(msgEtaPtr(layout, slot1), out1_eta, 3 * sizeof(double));
    std::memcpy(msgLamPtr(layout, slot1), out1_lam, 9 * sizeof(double));
}

FASTSYNC_FORCEINLINE void computeBinary3FactorAtNoDamping(
    SyntheticSE2FastSyncSoALayout& layout,
    int idx
) {
    gbp::Factor* factor = layout.binary_factors[idx];
    if (!factor->active) {
        return;
    }

    const double* belief0_eta = layout.belief0_eta_ptrs[idx];
    const double* belief1_eta = layout.belief1_eta_ptrs[idx];
    const double* belief0_lam = layout.belief0_lam_ptrs[idx];
    const double* belief1_lam = layout.belief1_lam_ptrs[idx];

    const int slot0 = 2 * idx;
    const int slot1 = slot0 + 1;
    const double* old0_eta = msgEtaPtr(layout, slot0);
    const double* old1_eta = msgEtaPtr(layout, slot1);
    const double* old0_lam = msgLamPtr(layout, slot0);
    const double* old1_lam = msgLamPtr(layout, slot1);

    const double* eta0 = vec3Ptr(layout.eta0, idx);
    const double* eta1 = vec3Ptr(layout.eta1, idx);
    const double* block00 = mat3Ptr(layout.block00, idx);
    const double* block03 = mat3Ptr(layout.block03, idx);
    const double* block30 = mat3Ptr(layout.block30, idx);
    const double* block33 = mat3Ptr(layout.block33, idx);

    double out0_eta[3];
    double out0_lam[9];
    {
        double eno[3] = {
            eta1[0] + (belief1_eta[0] - old1_eta[0]),
            eta1[1] + (belief1_eta[1] - old1_eta[1]),
            eta1[2] + (belief1_eta[2] - old1_eta[2]),
        };
        Cholesky3x3 chol;
        if (!factorizeSpd3x3Upper(
                block33[0] + (belief1_lam[0] - old1_lam[0]) + kJitter,
                block33[3] + (belief1_lam[3] - old1_lam[3]),
                block33[6] + (belief1_lam[6] - old1_lam[6]),
                block33[4] + (belief1_lam[4] - old1_lam[4]) + kJitter,
                block33[7] + (belief1_lam[7] - old1_lam[7]),
                block33[8] + (belief1_lam[8] - old1_lam[8]) + kJitter,
                chol
            )) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            A << block33[0] + (belief1_lam[0] - old1_lam[0]) + kJitter,
                 block33[3] + (belief1_lam[3] - old1_lam[3]),
                 block33[6] + (belief1_lam[6] - old1_lam[6]),
                 block33[3] + (belief1_lam[3] - old1_lam[3]),
                 block33[4] + (belief1_lam[4] - old1_lam[4]) + kJitter,
                 block33[7] + (belief1_lam[7] - old1_lam[7]),
                 block33[6] + (belief1_lam[6] - old1_lam[6]),
                 block33[7] + (belief1_lam[7] - old1_lam[7]),
                 block33[8] + (belief1_lam[8] - old1_lam[8]) + kJitter;
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lnoo_mat(block30);
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lono_mat(block03);
            const Eigen::Map<const Eigen::Matrix<double, 3, 1>> eo_vec(eta0);
            const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
            Eigen::Matrix3d solved_lam;
            Eigen::Vector3d solved_eta;
            if (!solveGeneral3x3(A, lnoo_mat, eno_vec, solved_lam, solved_eta)) {
                throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncSoA factor target=0");
            }
            const Eigen::Vector3d out_eta_vec = eo_vec - lono_mat * solved_eta;
            const Eigen::Matrix3d out_lam_mat = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(block00) - lono_mat * solved_lam;
            std::memcpy(out0_eta, out_eta_vec.data(), 3 * sizeof(double));
            std::memcpy(out0_lam, out_lam_mat.data(), 9 * sizeof(double));
            goto target0_done_nodamp;
        }
        schurMessage3x3(chol, eta0, eno, block00, block03, block30, old0_eta, old0_lam, 0.0, true, out0_eta, out0_lam);
    target0_done_nodamp:;
    }

    double out1_eta[3];
    double out1_lam[9];
    {
        double eno[3] = {
            eta0[0] + (belief0_eta[0] - old0_eta[0]),
            eta0[1] + (belief0_eta[1] - old0_eta[1]),
            eta0[2] + (belief0_eta[2] - old0_eta[2]),
        };
        Cholesky3x3 chol;
        if (!factorizeSpd3x3Upper(
                block00[0] + (belief0_lam[0] - old0_lam[0]) + kJitter,
                block00[3] + (belief0_lam[3] - old0_lam[3]),
                block00[6] + (belief0_lam[6] - old0_lam[6]),
                block00[4] + (belief0_lam[4] - old0_lam[4]) + kJitter,
                block00[7] + (belief0_lam[7] - old0_lam[7]),
                block00[8] + (belief0_lam[8] - old0_lam[8]) + kJitter,
                chol
            )) {
            Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
            A << block00[0] + (belief0_lam[0] - old0_lam[0]) + kJitter,
                 block00[3] + (belief0_lam[3] - old0_lam[3]),
                 block00[6] + (belief0_lam[6] - old0_lam[6]),
                 block00[3] + (belief0_lam[3] - old0_lam[3]),
                 block00[4] + (belief0_lam[4] - old0_lam[4]) + kJitter,
                 block00[7] + (belief0_lam[7] - old0_lam[7]),
                 block00[6] + (belief0_lam[6] - old0_lam[6]),
                 block00[7] + (belief0_lam[7] - old0_lam[7]),
                 block00[8] + (belief0_lam[8] - old0_lam[8]) + kJitter;
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lnoo_mat(block03);
            const Eigen::Map<const Eigen::Matrix<double, 3, 3>> lono_mat(block30);
            const Eigen::Map<const Eigen::Matrix<double, 3, 1>> eo_vec(eta1);
            const Eigen::Vector3d eno_vec(eno[0], eno[1], eno[2]);
            Eigen::Matrix3d solved_lam;
            Eigen::Vector3d solved_eta;
            if (!solveGeneral3x3(A, lnoo_mat, eno_vec, solved_lam, solved_eta)) {
                throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncSoA factor target=1");
            }
            const Eigen::Vector3d out_eta_vec = eo_vec - lono_mat * solved_eta;
            const Eigen::Matrix3d out_lam_mat = Eigen::Map<const Eigen::Matrix<double, 3, 3>>(block33) - lono_mat * solved_lam;
            std::memcpy(out1_eta, out_eta_vec.data(), 3 * sizeof(double));
            std::memcpy(out1_lam, out_lam_mat.data(), 9 * sizeof(double));
            goto target1_done_nodamp;
        }
        schurMessage3x3(chol, eta1, eno, block33, block30, block03, old1_eta, old1_lam, 0.0, true, out1_eta, out1_lam);
    target1_done_nodamp:;
    }

    std::memcpy(msgEtaPtr(layout, slot0), out0_eta, 3 * sizeof(double));
    std::memcpy(msgLamPtr(layout, slot0), out0_lam, 9 * sizeof(double));
    std::memcpy(msgEtaPtr(layout, slot1), out1_eta, 3 * sizeof(double));
    std::memcpy(msgLamPtr(layout, slot1), out1_lam, 9 * sizeof(double));
}

void updateBelief3DAt(
    SyntheticSE2FastSyncSoALayout& layout,
    int var_idx,
    bool update_mu
) {
    double* belief_eta = layout.belief_eta_ptrs[var_idx];
    double* belief_lam = layout.belief_lam_ptrs[var_idx];
    const double* prior_eta = layout.prior_eta_ptrs[var_idx];
    const double* prior_lam = layout.prior_lam_ptrs[var_idx];

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

    const int bin_begin = layout.binary_offsets[var_idx];
    const int bin_end = layout.binary_offsets[var_idx + 1];
    for (int i = bin_begin; i < bin_end; ++i) {
        const int slot_id = layout.binary_slot_ids[i];
        const double* msg_eta = msgEtaPtr(layout, slot_id);
        const double* msg_lam = msgLamPtr(layout, slot_id);
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

    const int fb_begin = layout.fallback_offsets[var_idx];
    const int fb_end = layout.fallback_offsets[var_idx + 1];
    for (int i = fb_begin; i < fb_end; ++i) {
        const double* msg_eta = *layout.fallback_eta_slots[i];
        const double* msg_lam = *layout.fallback_lam_slots[i];
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

    if (!update_mu) {
        layout.variables[var_idx]->invalidateMu();
        return;
    }

    Cholesky3x3 chol;
    if (!factorizeSpd3x3Upper(lam00, lam01, lam02, lam11, lam12, lam22, chol)) {
        if (!solveGeneral3x3UpperSym(lam00, lam01, lam02, lam11, lam12, lam22, belief_eta, layout.mu_ptrs[var_idx])) {
            throw std::runtime_error("Linear solve failed in SyntheticSE2FastSyncSoA variable update");
        }
        layout.variables[var_idx]->markMuCurrent();
        return;
    }
    double b[3] = {eta0, eta1, eta2};
    solveSpd3x3(chol, b, layout.mu_ptrs[var_idx]);
    layout.variables[var_idx]->markMuCurrent();
}

FASTSYNC_FORCEINLINE void updateBelief3DAtNoMu(
    SyntheticSE2FastSyncSoALayout& layout,
    int var_idx
) {
    double* belief_eta = layout.belief_eta_ptrs[var_idx];
    double* belief_lam = layout.belief_lam_ptrs[var_idx];
    const double* prior_eta = layout.prior_eta_ptrs[var_idx];
    const double* prior_lam = layout.prior_lam_ptrs[var_idx];

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

    const int bin_begin = layout.binary_offsets[var_idx];
    const int bin_end = layout.binary_offsets[var_idx + 1];
    for (int i = bin_begin; i < bin_end; ++i) {
        const int slot_id = layout.binary_slot_ids[i];
        const double* msg_eta = msgEtaPtr(layout, slot_id);
        const double* msg_lam = msgLamPtr(layout, slot_id);
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

    const int fb_begin = layout.fallback_offsets[var_idx];
    const int fb_end = layout.fallback_offsets[var_idx + 1];
    for (int i = fb_begin; i < fb_end; ++i) {
        const double* msg_eta = *layout.fallback_eta_slots[i];
        const double* msg_lam = *layout.fallback_lam_slots[i];
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

    layout.variables[var_idx]->invalidateMu();
}

}  // namespace

SyntheticSE2FastSyncSoALayout buildSyntheticSE2FastSyncSoALayout(gbp::FactorGraph& graph) {
    SyntheticSE2FastSyncSoALayout layout;
    layout.all_dofs3 = true;

    std::unordered_map<gbp::Factor*, int> binary_factor_to_idx;
    binary_factor_to_idx.reserve(graph.factors.size());

    for (auto& fup : graph.factors) {
        gbp::Factor* factor = fup.get();
        if (!factor) {
            continue;
        }
        const bool binary3 =
            factor->adj_var_nodes.size() == 2 &&
            factor->adj_var_nodes[0] != nullptr &&
            factor->adj_var_nodes[1] != nullptr &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6 &&
            factor->messages.size() == 2;

        if (!binary3) {
            layout.fallback_factors.push_back(factor);
            continue;
        }

        const int idx = static_cast<int>(layout.binary_factors.size());
        layout.binary_factors.push_back(factor);
        binary_factor_to_idx.emplace(factor, idx);
        layout.belief0_eta_ptrs.push_back(factor->adj_var_nodes[0]->belief.etaData());
        layout.belief1_eta_ptrs.push_back(factor->adj_var_nodes[1]->belief.etaData());
        layout.belief0_lam_ptrs.push_back(factor->adj_var_nodes[0]->belief.lamData());
        layout.belief1_lam_ptrs.push_back(factor->adj_var_nodes[1]->belief.lamData());
    }

    const size_t num_binary = layout.binary_factors.size();
    layout.eta0.resize(num_binary * 3, 0.0);
    layout.eta1.resize(num_binary * 3, 0.0);
    layout.block00.resize(num_binary * 9, 0.0);
    layout.block03.resize(num_binary * 9, 0.0);
    layout.block30.resize(num_binary * 9, 0.0);
    layout.block33.resize(num_binary * 9, 0.0);
    layout.msg_eta.resize(num_binary * 2 * 3, 0.0);
    layout.msg_lam.resize(num_binary * 2 * 9, 0.0);

    layout.variables.reserve(graph.var_nodes.size());
    layout.prior_eta_ptrs.reserve(graph.var_nodes.size());
    layout.prior_lam_ptrs.reserve(graph.var_nodes.size());
    layout.belief_eta_ptrs.reserve(graph.var_nodes.size());
    layout.belief_lam_ptrs.reserve(graph.var_nodes.size());
    layout.mu_ptrs.reserve(graph.var_nodes.size());
    layout.binary_offsets.reserve(graph.var_nodes.size() + 1);
    layout.fallback_offsets.reserve(graph.var_nodes.size() + 1);

    int binary_offset = 0;
    int fallback_offset = 0;
    for (auto& vup : graph.var_nodes) {
        gbp::VariableNode* var = vup.get();
        layout.binary_offsets.push_back(binary_offset);
        layout.fallback_offsets.push_back(fallback_offset);
        layout.variables.push_back(var);
        layout.prior_eta_ptrs.push_back(var ? var->prior.etaData() : nullptr);
        layout.prior_lam_ptrs.push_back(var ? var->prior.lamData() : nullptr);
        layout.belief_eta_ptrs.push_back(var ? var->belief.etaData() : nullptr);
        layout.belief_lam_ptrs.push_back(var ? var->belief.lamData() : nullptr);
        layout.mu_ptrs.push_back(var ? var->mu.data() : nullptr);
        if (!var) {
            continue;
        }
        if (var->dofs != 3) {
            layout.all_dofs3 = false;
        }

        for (const auto& aref : var->adj_factors) {
            auto it = binary_factor_to_idx.find(aref.factor);
            if (it != binary_factor_to_idx.end()) {
                layout.binary_slot_ids.push_back(2 * it->second + aref.local_idx);
                ++binary_offset;
            } else {
                layout.fallback_eta_slots.push_back(aref.msg_eta_slot);
                layout.fallback_lam_slots.push_back(aref.msg_lam_slot);
                ++fallback_offset;
            }
        }
    }
    layout.binary_offsets.push_back(binary_offset);
    layout.fallback_offsets.push_back(fallback_offset);

    refreshSyntheticSE2FastSyncSoALayout(layout);
    return layout;
}

void refreshSyntheticSE2FastSyncSoALayout(SyntheticSE2FastSyncSoALayout& layout) {
    std::fill(layout.msg_eta.begin(), layout.msg_eta.end(), 0.0);
    std::fill(layout.msg_lam.begin(), layout.msg_lam.end(), 0.0);
    for (int idx = 0; idx < static_cast<int>(layout.binary_factors.size()); ++idx) {
        refreshBinaryFactorAt(layout, idx);
    }
}

void synchronousIterationFastSyncSoA(
    SyntheticSE2FastSyncSoALayout& layout,
    double eta_damping,
    bool update_mu,
    int num_threads,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum
) {
    const int threads = effectiveThreads(num_threads);
    const int num_binary = static_cast<int>(layout.binary_factors.size());
    const int num_vars = static_cast<int>(layout.variables.size());
    const bool no_damping = (eta_damping == 0.0);
    const bool no_mu = !update_mu;

    if (threads <= 1 || (num_binary <= 1 && num_vars <= 1)) {
        const auto factor_t0 = SteadyClock::now();
        if (no_damping) {
            for (int idx = 0; idx < num_binary; ++idx) {
                computeBinary3FactorAtNoDamping(layout, idx);
            }
        } else {
            for (int idx = 0; idx < num_binary; ++idx) {
                computeBinary3FactorAt(layout, idx, eta_damping);
            }
        }
        for (gbp::Factor* factor : layout.fallback_factors) {
            if (!factor || !factor->active) {
                continue;
            }
            factor->computeMessages(eta_damping);
        }
        const auto factor_t1 = SteadyClock::now();

        const auto var_t0 = SteadyClock::now();
        if (no_mu) {
            for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
                if (layout.variables[var_idx]) {
                    updateBelief3DAtNoMu(layout, var_idx);
                }
            }
        } else {
            for (int var_idx = 0; var_idx < num_vars; ++var_idx) {
                if (layout.variables[var_idx]) {
                    updateBelief3DAt(layout, var_idx, update_mu);
                }
            }
        }
        const auto var_t1 = SteadyClock::now();

        factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
        variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
        return;
    }

    SteadyClock::time_point factor_t0;
    SteadyClock::time_point factor_t1;
    SteadyClock::time_point var_t0;
    SteadyClock::time_point var_t1;

    if (static_cast<int>(layout.factor_thread_rates.size()) != threads) {
        layout.factor_thread_rates.assign(threads, 1.0);
        layout.variable_thread_rates.assign(threads, 1.0);
        layout.factor_thread_work_last.assign(threads, 0.0);
        layout.variable_thread_work_last.assign(threads, 0.0);
    }
    buildContiguousWeightedOffsets(num_binary, layout.factor_thread_rates, layout.factor_thread_offsets);
    buildContiguousWeightedOffsets(num_vars, layout.variable_thread_rates, layout.variable_thread_offsets);

#pragma omp parallel num_threads(threads)
    {
        const int tid = omp_get_thread_num();
        const int factor_begin = layout.factor_thread_offsets[tid];
        const int factor_end = layout.factor_thread_offsets[tid + 1];

#pragma omp single
        factor_t0 = SteadyClock::now();

        const double factor_work_t0 = omp_get_wtime();
        if (no_damping) {
            for (int idx = factor_begin; idx < factor_end; ++idx) {
                computeBinary3FactorAtNoDamping(layout, idx);
            }
        } else {
            for (int idx = factor_begin; idx < factor_end; ++idx) {
                computeBinary3FactorAt(layout, idx, eta_damping);
            }
        }
        const double factor_work_t1 = omp_get_wtime();
        layout.factor_thread_work_last[tid] = factor_work_t1 - factor_work_t0;

#pragma omp barrier
#pragma omp single
        {
            for (gbp::Factor* factor : layout.fallback_factors) {
                if (!factor || !factor->active) {
                    continue;
                }
                factor->computeMessages(eta_damping);
            }
            factor_t1 = SteadyClock::now();
            var_t0 = factor_t1;
        }

        const int var_begin = layout.variable_thread_offsets[tid];
        const int var_end = layout.variable_thread_offsets[tid + 1];
        const double var_work_t0 = omp_get_wtime();
        if (no_mu) {
            for (int var_idx = var_begin; var_idx < var_end; ++var_idx) {
                if (layout.variables[var_idx]) {
                    updateBelief3DAtNoMu(layout, var_idx);
                }
            }
        } else {
            for (int var_idx = var_begin; var_idx < var_end; ++var_idx) {
                if (layout.variables[var_idx]) {
                    updateBelief3DAt(layout, var_idx, update_mu);
                }
            }
        }
        const double var_work_t1 = omp_get_wtime();
        layout.variable_thread_work_last[tid] = var_work_t1 - var_work_t0;

#pragma omp barrier
#pragma omp single
        var_t1 = SteadyClock::now();
    }

    updateThroughputEma(layout.factor_thread_offsets, layout.factor_thread_work_last, layout.factor_thread_rates);
    updateThroughputEma(layout.variable_thread_offsets, layout.variable_thread_work_last, layout.variable_thread_rates);

    factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
    variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
}

}  // namespace slam
