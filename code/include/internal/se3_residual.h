#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "gbp/FactorGraph.h"
#include "internal/se3_solver_impl.h"

namespace slam {

struct SE3PrecisionPolicy {
    static constexpr double tolerance = 1e-8;
    static constexpr int check_period = 5;
    static constexpr int stable_checks = 3;
    static constexpr int frozen_check_period = 25;
};

struct SyntheticSE3PackedSoAStats {
    int sweeps = 0;
    int full_lambda_sweeps = 0;
    int fixed_lambda_init_sweeps = 0;
    int fixed_eta_sweeps = 0;
    int fixed_eta_map_builds = 0;
    double factor_pass_sec = 0.0;
    double variable_pass_sec = 0.0;
};

struct SyntheticSE3PackedSoAWorkspace {
    using AlignedDoubles = std::vector<double, Eigen::aligned_allocator<double>>;

    int num_vars = 0;
    int num_binary_factors = 0;
    int num_unary_factors = 0;
    double tiny_prior = 1e-12;
    int sweeps_since_relinearize = 0;
    bool adaptive_precision = false;
    bool persistent_sweeps = false;
    bool precision_frozen = false;
    int precision_stable_checks = 0;
    int precision_checks = 0;
    int precision_freezes = 0;
    int precision_thaws = 0;
    int first_precision_freeze_sweep = -1;
    int last_precision_check_sweep = -1;
    double last_precision_residual = 0.0;
    int full_precision_sweeps = 0;
    int eta_only_sweeps = 0;
    AlignedDoubles precision_residuals;
    bool jacobi_ready = false;
    int jacobi_sweeps = 0;
    int cycle_message_rebuilds = 0;
    AlignedDoubles jacobi_diagonal, jacobi_inverse, jacobi_rhs, jacobi_x, jacobi_alt;
    bool defect_ready = false;
    int defect_mean_sweeps = 0;
    int defect_map_builds = 0;
    int defect_clamps = 0;
    double defect_build_sec = 0;
    double defect_inverse_residual = 0;
    AlignedDoubles defect_belief_inverse, defect_map0, defect_map1;

    AlignedDoubles prior_eta;       // num_vars * 6
    AlignedDoubles prior_lam21;     // num_vars * 21, symmetric upper packed
    AlignedDoubles belief_eta;      // num_vars * 6
    AlignedDoubles belief_lam21;    // num_vars * 21
    AlignedDoubles mu;              // num_vars * 6
    std::vector<std::uint8_t> mu_valid;

    std::vector<int> unary_var_id;  // num_unary_factors
    AlignedDoubles unary_eta;       // num_unary_factors * 6
    AlignedDoubles unary_lam21;     // num_unary_factors * 21
    AlignedDoubles unary_msg_eta;   // num_unary_factors * 6
    AlignedDoubles unary_msg_lam21; // num_unary_factors * 21

    std::vector<int> binary_var0_id; // num_binary_factors
    std::vector<int> binary_var1_id; // num_binary_factors
    AlignedDoubles binary_eta0;      // num_binary_factors * 6
    AlignedDoubles binary_eta1;      // num_binary_factors * 6
    AlignedDoubles binary_diag0_lam21;
    AlignedDoubles binary_diag1_lam21;
    // Serial-only, immutable factor reference norms for one linearization.
    AlignedDoubles binary_reference_norms;
    bool binary_reference_norms_valid = false;
    AlignedDoubles binary_cross01_lam36; // full 6x6 col-major
    AlignedDoubles binary_cross10_lam36; // full 6x6 col-major

    AlignedDoubles binary_msg_eta;      // 2 * num_binary_factors * 6
    AlignedDoubles binary_msg_lam21;    // 2 * num_binary_factors * 21
    AlignedDoubles fixed_eta_map0_lam36; // num_binary_factors * 36, cross01 * inv(cavity1)
    AlignedDoubles fixed_eta_map1_lam36; // num_binary_factors * 36, cross10 * inv(cavity0)
    std::vector<std::uint8_t> fixed_lam_initialized;
    std::vector<std::uint8_t> fixed_eta_map_valid;
    std::uint8_t fixed_eta_maps_all_valid = 0;

    std::vector<int> unary_offsets;
    std::vector<int> unary_ids;
    std::vector<int> binary_offsets;
    std::vector<int> binary_slot_ids;
};

SyntheticSE3PackedSoAWorkspace buildSyntheticSE3PackedSoAWorkspace(
    const SyntheticSE3Problem& problem,
    double tiny_prior = 1e-12
);

void relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(
    SyntheticSE3PackedSoAWorkspace& workspace,
    const gbp::FactorGraph& graph
);

// Call after basis construction, once per fresh linearization.
void initializeBalancedSE3PackedMessages(SyntheticSE3PackedSoAWorkspace& workspace, int num_threads = 1);
void recomputeSE3PackedBeliefs(SyntheticSE3PackedSoAWorkspace& workspace,int num_threads);
// Lift an already accepted mean correction into existing FV eta. Canonical
// factors and all precisions are unchanged; no new GBP iteration is performed.
void liftSE3PackedMeanCorrection(SyntheticSE3PackedSoAWorkspace& workspace,
    const Eigen::VectorXd& delta, bool precision_weighted, int num_threads);
double measureSE3PackedEtaMismatch(const SyntheticSE3PackedSoAWorkspace& workspace);
void blockJacobiSE3PackedIterations(SyntheticSE3PackedSoAWorkspace& workspace, int sweeps, int num_threads);
// Experimental mean solver. Fixed diagonal FV precisions are NOT exact marginals.
void prepareSE3PrecisionDefect(SyntheticSE3PackedSoAWorkspace& workspace, int num_threads);
void precisionDefectSE3Iterations(SyntheticSE3PackedSoAWorkspace& workspace,
    int sweeps, int num_threads, double damping, double update_limit);
// Reconstruct FV eta consistently with a coarse-corrected mean and true b-Hx.
// Leaves canonical factors and every precision unchanged.
void rebuildSE3PackedMessagesAtMean(SyntheticSE3PackedSoAWorkspace& workspace,
    const Eigen::VectorXd& mean, const Eigen::VectorXd& residual, int num_threads);

void assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(
    const SyntheticSE3PackedSoAWorkspace& workspace,
    Eigen::VectorXd& eta,
    int num_threads = 1
);

void multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
    const SyntheticSE3PackedSoAWorkspace& workspace,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& out,
    int num_threads = 1
);

void synchronousIterationsSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& workspace,
    int num_sweeps,
    int num_threads,
    int fixed_lam_after_sweep,
    double eta_damping,
    double max_rel_update,
    SyntheticSE3PackedSoAStats* stats = nullptr
);

Eigen::VectorXd stackedMeanVectorSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& workspace,
    int num_threads = 1
);

void stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(
    SyntheticSE3PackedSoAWorkspace& workspace,
    Eigen::VectorXd& out,
    int num_threads = 1
);

void applyMeanDeltaSyntheticSE3PackedSoAWorkspace(
    SyntheticSE3PackedSoAWorkspace& workspace,
    const Eigen::VectorXd& delta,
    int num_threads = 1
);

void copySyntheticSE3PackedSoABeliefsToGraph(
    const SyntheticSE3PackedSoAWorkspace& workspace,
    gbp::FactorGraph& graph,
    int num_threads = 1,
    bool copy_lambda = true
);

void copySyntheticSE3PackedSoAToGraph(
    const SyntheticSE3PackedSoAWorkspace& workspace,
    gbp::FactorGraph& graph,
    int num_threads = 1
);

}  // namespace slam
