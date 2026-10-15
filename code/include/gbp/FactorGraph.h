#pragma once
#include <vector>
#include <deque>
#include <memory>
#include <functional>
#include <atomic>
#include <utility>
#include <cstdint>
#include <limits>
#include <thread>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "gbp/VariableNode.h"
#include "gbp/Factor.h"

// High-throughput approximate max-priority scheduler (MultiQueue)
#include "gbp/MultiQueue.h"

namespace gbp {
class FactorGraph {
public:
    enum class SyncScheduleKind {
        Static = 0,
        Dynamic = 1,
        Guided = 2,
    };

    FactorGraph() = default;

    // 禁用拷贝
    FactorGraph(const FactorGraph&) = delete;
    FactorGraph& operator=(const FactorGraph&) = delete;

    // 启用移动（关键：因为你有自定义析构函数）
    FactorGraph(FactorGraph&&) noexcept = default;
    FactorGraph& operator=(FactorGraph&&) noexcept = default;

    bool nonlinear_factors = false; // 目前你的phase: false/unused
    double eta_damping = 0.0;
    int sync_num_threads = 0; // 0 => OpenMP/runtime default
    bool sync_update_means = true;
    SyncScheduleKind sync_schedule_kind = SyncScheduleKind::Static;
    int sync_schedule_chunk = 0; // 0 => runtime default chunk
    bool sync_weighted_static_partition = true;
    bool profile_sync_timing = false;
    bool profile_sync_thread_utilization = false;
    double sync_factor_pass_sec_accum = 0.0;
    double sync_variable_pass_sec_accum = 0.0;
    double sync_factor_work_sec_accum = 0.0;
    double sync_factor_wait_sec_accum = 0.0;
    double sync_variable_work_sec_accum = 0.0;
    double sync_variable_wait_sec_accum = 0.0;
    int sync_profiled_threads = 0;
    int sync_total_sweeps_accum = 0;
    int sync_fixed_lam_hard_start_sweep = -1;
    int sync_fixed_lam_effective_start_sweep = -1;
    int sync_fixed_lam_factor_sweeps_accum = 0;
    int sync_eta_only_variable_sweeps_accum = 0;
    int sync_fixed_lam_hot_eta_sweeps_accum = 0;
    int sync_fixed_lam_hot_eta_entries = 0;

    std::vector<std::unique_ptr<VariableNode>> var_nodes;
    std::vector<std::unique_ptr<Factor>> factors;
    std::vector<FixedLamEta6HotEntry> se3_fixed_lam_eta_hot_entries;
    bool se3_fixed_lam_eta_hot_valid = false;

    // residual scheduling on variables (by id)
    double residual_eps = 1e-6; // kept for compatibility, but NOT used for early-break (Python parity)
    // Scalar mirror used by compatibility paths; the scheduler uses the atomics below.
    std::vector<double> var_residual;

    // Parallel-safe scheduler state.
    // - var_residual_a: current best-known residual (monotone increasing until explicitly reset)
    // - var_ver:        bumps whenever residual is updated/reset; used for lazy invalidation
    // NOTE: std::atomic<T> is neither copyable nor movable. On MSVC, using
    // std::vector<std::atomic<T>> with resize/reallocation can fail to compile
    // because vector growth attempts to move elements. Use deque instead.
    std::deque<std::atomic<double>>   var_residual_a;
    std::deque<std::atomic<uint32_t>> var_ver;

    // --- Max-heap on residual (positive) with Python-equivalent tie-break on varID ---
    // Python heap element: (-r, varID, var) in a MIN-heap
    // Equivalent pop order: r desc, varID asc
    struct HeapEntry {
        double r;   // residual (>=0)
        int varID;
        uint32_t ver = 0;
        bool operator<(const HeapEntry& other) const {
            if (r != other.r) return r < other.r;  // larger r first
            return varID > other.varID;            // smaller varID first
        }
    };

    struct HeapEntryKey {
        inline double operator()(const HeapEntry& e) const noexcept { return e.r; }
    };

    // Approximate max-priority queue (MultiQueue). Constructed lazily.
    std::unique_ptr<mqfast::MultiQueueFast<HeapEntry, HeapEntryKey>> var_mq;
    bool var_mq_initialized = false;

    // ------------
    // Parallel safety (for residualIterationVarHeap OpenMP worker loop)
    // ------------
    std::vector<omp_lock_t> var_locks;
    std::vector<omp_lock_t> fac_locks;
    bool locks_initialized = false;

    ~FactorGraph();

    VariableNode* addVariable(int id, int dofs);

    // ===== Python-equivalent Factor construction =====
    // z:                list of measurement blocks
    // measurement_lambda: list of precision blocks
    // meas_fn(x):        list of predicted measurement blocks
    // jac_fn(x):         list of Jacobian blocks
    Factor* addFactor(
        int id,
        const std::vector<VariableNode*>& vars,
        const std::vector<Eigen::VectorXd>& z,
        const std::vector<Eigen::MatrixXd>& measurement_lambda,
        std::function<std::vector<Eigen::VectorXd>(const Eigen::VectorXd&)> meas_fn,
        std::function<std::vector<Eigen::MatrixXd>(const Eigen::VectorXd&)> jac_fn
    );

    void connect(Factor* f, VariableNode* v, int local_idx);

    void setSyncNumThreads(int n_threads) { sync_num_threads = n_threads; }
    void setSyncUpdateMeans(bool enabled) { sync_update_means = enabled; }
    void setSyncSchedule(SyncScheduleKind kind, int chunk = 0) {
        sync_schedule_kind = kind;
        sync_schedule_chunk = chunk;
    }
    void setSyncWeightedStaticPartition(bool enabled) { sync_weighted_static_partition = enabled; }
    void setProfileSyncTiming(bool enabled) { profile_sync_timing = enabled; }
    void setProfileSyncThreadUtilization(bool enabled) { profile_sync_thread_utilization = enabled; }
    void resetSyncTiming() {
        sync_factor_pass_sec_accum = 0.0;
        sync_variable_pass_sec_accum = 0.0;
        sync_factor_work_sec_accum = 0.0;
        sync_factor_wait_sec_accum = 0.0;
        sync_variable_work_sec_accum = 0.0;
        sync_variable_wait_sec_accum = 0.0;
        sync_profiled_threads = 0;
        sync_total_sweeps_accum = 0;
        sync_fixed_lam_hard_start_sweep = -1;
        sync_fixed_lam_effective_start_sweep = -1;
        sync_fixed_lam_factor_sweeps_accum = 0;
        sync_eta_only_variable_sweeps_accum = 0;
        sync_fixed_lam_hot_eta_sweeps_accum = 0;
        sync_fixed_lam_hot_eta_entries = 0;
    }

    // synchronous iteration (like Python synchronous_iteration)
    void synchronousIteration(bool robustify=false);
    void synchronousIterationFixedLam(bool robustify=false);

    // ------------------------------------------------------------
    // Nonlinear support (Gauss-Newton style)
    // ------------------------------------------------------------
    // Re-linearize all factors at the current variable means (mu).
    // This updates each factor's internal information form (eta/lam) and
    // resets its messages accordingly via Factor::computeFactor(..., true).
    void relinearizeAllFactors();

    // residual iteration (variable heap) (like residual_iteration_var_heap)
    void residualIterationVarHeap(int max_updates = -1);

    // ======================
    // Batch (joint) utilities
    // ======================
    struct JointInfResult {
        Eigen::VectorXd eta;
        Eigen::SparseMatrix<double> lam;
        std::vector<int> var_ix;  // varID -> offset
        int total_dim = 0;
    };

    JointInfResult jointDistributionInfSparse() const;
    Eigen::VectorXd jointMAPSparse(double diag_jitter = 1e-12) const;

private:
    static constexpr double kJitter = 1e-12;
    std::vector<int> sync_factor_offsets_;
    std::vector<int> sync_variable_offsets_;
    std::vector<double> sync_factor_rates_;
    std::vector<double> sync_variable_rates_;
    std::vector<double> sync_factor_work_last_;
    std::vector<double> sync_variable_work_last_;
    bool sync_weighted_offsets_ready_ = false;
};
} // namespace gbp
