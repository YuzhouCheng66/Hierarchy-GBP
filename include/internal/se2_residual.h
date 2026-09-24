#pragma once

#include <cstdint>
#include <vector>

#include "internal/se2_solver_impl.h"
#include "internal/se2_precision.h"

namespace slam {

struct SyntheticSE2PackedResidualWorkspace;
void fineResidualSyntheticSE2PackedInto(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const Eigen::VectorXd& x, Eigen::VectorXd& residual, int threads);
void shiftSE2PackedCorrectionReference(SyntheticSE2PackedResidualWorkspace& workspace,
                                      const Eigen::VectorXd& delta_reference);
void initializeSE2PackedMessagesFromFactors(SyntheticSE2PackedResidualWorkspace& workspace, int mode, int threads);

// Synthetic-only packed residual workspace intended to become the non-sweeps
// backbone for 1-core MG. This keeps residual factors, messages, beliefs, and
// adjacency in contiguous arrays so basis/coarse assembly can eventually bypass
// FactorGraph/Factor object traversal entirely.

struct SyntheticSE2PackedResidualUnaryFactor {
    int var_id = -1;
    double eta[3] = {};
    double lam6[6] = {};
};

struct SyntheticSE2PackedResidualBinaryFactor {
    int var0_id = -1;
    int var1_id = -1;
    double eta0[3] = {};
    double eta1[3] = {};
    double diag0_lam6[6] = {};
    double diag1_lam6[6] = {};
    double cross01_lam9[9] = {};
};

struct SyntheticSE2PackedResidualWorkspace {
    int num_vars = 0;
    double tiny_prior = 1e-12;
    double eta_relaxation = 1.0;
    bool lift_correction_to_messages = false;
    int message_initialization = 0;
    bool jacobi_ready = false;
    std::vector<double> jacobi_diagonal, jacobi_inverse, jacobi_rhs, jacobi_x, jacobi_alt;
    bool kernel_profile_enabled = false;
    bool kernel_split_profile_enabled = false;
    std::uint64_t schur_adjugate_fallback_count = 0;
    std::uint64_t schur_general_failure_count = 0;
    std::uint64_t kernel_prep_cycles = 0;
    std::uint64_t kernel_schur0_cycles = 0;
    std::uint64_t kernel_schur1_cycles = 0;
    std::uint64_t kernel_update_belief_cycles = 0;
    std::uint64_t fixedeta_full_lambda_sweeps = 0;
    std::uint64_t fixedeta_lambda_init_sweeps = 0;
    std::uint64_t fixedeta_eta_only_sweeps = 0;
    std::uint64_t fixedeta_eta_map_builds = 0;
    std::uint64_t fixedeta_serial_delta_sweeps = 0;
    int sweeps_since_relinearize = 0;
    bool adaptive_precision = false;
    bool precision_frozen = false;
    int precision_stable_checks = 0;
    int precision_checks = 0;
    int precision_freezes = 0;
    int precision_thaws = 0;
    int first_precision_freeze_sweep = -1;
    int last_precision_check_sweep = 0;
    int precision_check_interval = SE2PrecisionPolicy::check_period;
    int next_precision_check_sweep = SE2PrecisionPolicy::check_period;
    int full_precision_sweeps = 0;
    int eta_only_sweeps = 0;
    double last_precision_residual = 0.0;
    bool precision_scales_ready = false;
    std::vector<double> precision_scales;
    std::vector<double> precision_residuals;

    std::vector<double> prior_eta;
    std::vector<double> prior_lam6;
    std::vector<double> base_belief_eta;
    std::vector<double> base_belief_lam6;
    std::vector<double> belief_eta;
    std::vector<double> belief_lam6;
    std::vector<double> belief_eta_alt;
    std::vector<double> belief_lam6_alt;
    std::vector<double> mu;
    std::vector<uint8_t> mu_valid;

    std::vector<SyntheticSE2PackedResidualUnaryFactor> unary_factors;
    std::vector<SyntheticSE2PackedResidualBinaryFactor> binary_factors;

    std::vector<double> unary_msg_eta;
    std::vector<double> unary_msg_lam6;
    std::vector<double> binary_msg_eta;
    std::vector<double> binary_msg_lam6;
    std::vector<double> binary_msg_eta_alt;
    std::vector<double> binary_msg_lam6_alt;
    std::vector<double> fixed_eta_map0_lam9;
    std::vector<double> fixed_eta_map1_lam9;
    std::vector<uint8_t> fixed_lam_initialized;
    std::vector<uint8_t> fixed_eta_map_valid;
    uint8_t fixed_eta_maps_all_valid = 0;
    std::vector<double> serial_belief_delta_eta;
    uint8_t belief_eta_message_consistent = 0;
    std::vector<int> unary_offsets;
    std::vector<int> unary_ids;
    std::vector<int> binary_offsets;
    std::vector<int> binary_slot_ids;

};

SyntheticSE2PackedResidualWorkspace buildSyntheticSE2PackedResidualWorkspace(
    const SyntheticSE2Problem& problem,
    double tiny_prior = 1e-12
);

void relinearizeSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    const RobustLossConfig& robust_loss_config = {},
    int num_threads = 1
);

void synchronousIterationSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum,
    int num_threads = 1
);

void synchronousIterationsSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int num_sweeps,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum,
    int num_threads = 1
);

void blockJacobiSE2PackedIterations(SyntheticSE2PackedResidualWorkspace& workspace, int sweeps, int threads);

Eigen::VectorXd stackedMeanVectorSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    int num_threads = 1
);

void stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
    SyntheticSE2PackedResidualWorkspace& workspace,
    Eigen::VectorXd& out,
    int num_threads = 1
);

void injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& workspace,
    const Eigen::VectorXd& delta,
    int num_threads = 1
);

}  // namespace slam
