#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "gbp/FactorGraph.h"
#include "internal/se3_solver_impl.h"

namespace slam {

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
