#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>

#include "gbp/FactorGraph.h"
#include "internal/se2_basis_data.h"

namespace slam {

enum class BasisEigensolverKind {
    Full,
    Partial,
};

struct BasisBuildConfig {
    BasisEigensolverKind eigensolver = BasisEigensolverKind::Partial;
    bool enable_warm_start = true;
    bool enable_packed_builder = false;
    bool enable_packed_residual_solver = true;
    int basis_rebuild_period = 1;
    int basis_rebuild_warmup_outers = 0;
    int packed_basis_thread_override = 0;
    int packed_coarse_thread_override = 0;
    int packed_sweep_thread_override = 0;
    int partial_oversampling = 4;
    int partial_max_iters = 6;
    int partial_residual_check_period = 1;
    double partial_residual_tol = 1e-5;
    double partial_ridge = 1e-10;
};

struct SyntheticSE2Edge {
    int i = -1;
    int j = -1;
    Eigen::Vector3d measurement = Eigen::Vector3d::Zero();
    Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
    std::string kind;
};

struct SyntheticSE2Problem {
    std::vector<Eigen::Vector3d> gt_poses;
    std::vector<Eigen::Vector3d> init_poses;
    std::vector<SyntheticSE2Edge> edges;
    Eigen::Vector3d anchor_pose = Eigen::Vector3d::Zero();
    Eigen::Matrix3d anchor_information = Eigen::Matrix3d::Zero();
};

struct RobustLossConfig {
    double huber_delta = 0.0;
};

struct OuterDirectRow {
    int outer = 0;
    double nonlinear_objective = 0.0;
    double linear_step_norm = 0.0;
    double linear_residual_norm = 0.0;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double factor_relinearize_sec = 0.0;
    double reset_state_sec = 0.0;
    double joint_assembly_sec = 0.0;
    double exact_solve_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
};

struct OuterMGRow {
    int outer = 0;
    std::string execution_path;
    double nonlinear_objective = 0.0;
    double e_hat_norm = 0.0;
    double factor_lam_max_asym = 0.0;
    bool prefer_nonsym_safe_fastsync = false;
    int num_groups = 0;
    int coarse_dim = 0;
    int coarse_lambda_nnz = 0;
    double coarse_lambda_density = 0.0;
    bool coarse_lambda_dense_factor = false;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double factor_relinearize_sec = 0.0;
    double reset_state_sec = 0.0;
    double basis_build_sec = 0.0;
    double basis_block_build_sec = 0.0;
    double basis_eigensolver_sec = 0.0;
    double basis_copyout_sec = 0.0;
    double coarse_lambda_sec = 0.0;
    double sweeps_sec = 0.0;
    double factor_pass_sec = 0.0;
    double variable_pass_sec = 0.0;
    double factor_work_sec = 0.0;
    double factor_wait_sec = 0.0;
    double variable_work_sec = 0.0;
    double variable_wait_sec = 0.0;
    int sync_profiled_threads = 0;
    double coarse_eta_sec = 0.0;
    double coarse_solve_sec = 0.0;
    double prolong_inject_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
    int basis_partial_attempted = 0;
    int basis_partial_converged = 0;
    int basis_partial_fallback = 0;
    int basis_partial_total_iters = 0;
};

struct ExperimentResults {
    int num_poses = 0;
    int num_edges = 0;
    double initial_objective = 0.0;
    double residual_workspace_build_sec = 0.0;
    double residual_workspace_variable_alloc_sec = 0.0;
    double residual_workspace_factor_alloc_sec = 0.0;
    double residual_workspace_connect_sec = 0.0;
    std::uint64_t packed_schur_adjugate_fallback_count = 0;
    std::uint64_t packed_schur_general_failure_count = 0;
    std::uint64_t packed_kernel_prep_cycles = 0;
    std::uint64_t packed_kernel_schur0_cycles = 0;
    std::uint64_t packed_kernel_schur1_cycles = 0;
    std::uint64_t packed_kernel_update_belief_cycles = 0;
    std::uint64_t packed_fixedeta_full_lambda_sweeps = 0;
    std::uint64_t packed_fixedeta_lambda_init_sweeps = 0;
    std::uint64_t packed_fixedeta_eta_only_sweeps = 0;
    std::uint64_t packed_fixedeta_eta_map_builds = 0;
    std::uint64_t packed_fixedeta_serial_delta_sweeps = 0;
    std::vector<OuterDirectRow> direct_history;
    std::vector<OuterMGRow> mg_history;
    std::vector<Eigen::Vector3d> initial_poses;
    std::vector<std::vector<Eigen::Vector3d>> direct_pose_history;
    std::vector<std::vector<Eigen::Vector3d>> mg_pose_history;
};

SyntheticSE2Problem loadSyntheticSE2Problem(const std::string& path);

double nonlinearObjective(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& poses,
    int num_threads = 1
);

std::vector<Eigen::Vector3d> applyPoseDeltas(
    const std::vector<Eigen::Vector3d>& base_poses,
    const Eigen::VectorXd& delta_vec
);

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    double tiny_prior = 1e-12,
    const RobustLossConfig& robust_loss_config = {}
);

ExperimentResults runSyntheticSE2Experiment(
    const SyntheticSE2Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    int sync_num_threads = 0,
    gbp::FactorGraph::SyncScheduleKind sync_schedule_kind = gbp::FactorGraph::SyncScheduleKind::Static,
    int sync_schedule_chunk = 0,
    bool profile_sync_thread_utilization = false,
    bool enable_singlecore_fastsync = true,
    double coarse_scale = 1.0,
    const BasisBuildConfig& basis_build_config = {},
    const RobustLossConfig& robust_loss_config = {},
    int final_coarse_polish_passes = 0
);

void writeExperimentResultsJson(
    const ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    int sync_num_threads = 0,
    gbp::FactorGraph::SyncScheduleKind sync_schedule_kind = gbp::FactorGraph::SyncScheduleKind::Static,
    int sync_schedule_chunk = 0,
    bool profile_sync_thread_utilization = false,
    bool enable_singlecore_fastsync = true,
    const BasisBuildConfig& basis_build_config = {},
    const RobustLossConfig& robust_loss_config = {},
    int final_coarse_polish_passes = 0,
    double packed_jitter = 1e-10,
    int fixed_eta_after_sweeps = -1
);

void writeExperimentPoseHistoryJson(
    const ExperimentResults& results,
    const std::string& path
);

}  // namespace slam
