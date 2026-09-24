#pragma once

#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "gbp/FactorGraph.h"

namespace slam {

struct SE3Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
};

using SE3PoseVector = std::vector<SE3Pose, Eigen::aligned_allocator<SE3Pose>>;
using SE3PoseHistory = std::vector<SE3PoseVector>;

struct SyntheticSE3Edge {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int i = -1;
    int j = -1;
    SE3Pose measurement;
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix<double, 6, 6>::Zero();
    std::string kind;
};

using SyntheticSE3EdgeVector =
    std::vector<SyntheticSE3Edge, Eigen::aligned_allocator<SyntheticSE3Edge>>;

struct SyntheticSE3Problem {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3PoseVector gt_poses;
    SE3PoseVector init_poses;
    SyntheticSE3EdgeVector edges;
    SE3Pose anchor_pose;
    Eigen::Matrix<double, 6, 6> anchor_information = Eigen::Matrix<double, 6, 6>::Zero();
};

struct RobustLossConfig {
    double huber_delta = 0.0;
};

struct SyntheticSE3OuterDirectRow {
    int outer = 0;
    double nonlinear_objective = 0.0;
    double linear_step_norm = 0.0;
    double linear_residual_norm = 0.0;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double joint_assembly_sec = 0.0;
    double exact_solve_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
};

struct SE3CycleAuditRow {
    int cycle=0;
    double initial_residual=0, fine_residual=0, coarse_residual=0, final_residual=0;
    double fine_gain=0, coarse_gain=0, total_gain=0;
    double fine_energy=0, coarse_energy=0, cross_energy=0;
    double residual_defect_absolute=0, residual_defect_relative=0;
    double eta_defect_before=0, eta_defect_after=0;
};

struct SyntheticSE3OuterMGRow {
    int outer = 0;
    double nonlinear_objective = 0.0;
    double raw_e_hat_norm = 0.0;
    double e_hat_norm = 0.0;
    double first_cycle_residual_before_sweeps = 0.0;
    double first_cycle_residual_after_sweeps = 0.0;
    double first_cycle_residual_after_coarse = 0.0;
    double last_cycle_residual_before_sweeps = 0.0;
    double last_cycle_residual_after_sweeps = 0.0;
    double last_cycle_residual_after_coarse = 0.0;
    double linear_residual_approx = 0.0;
    double line_search_step_scale = 0.0;
    int line_search_rejects = 0;
    int executed_inner_cycles = 0;
    double smoother_translation_update_norm_sum = 0.0;
    double smoother_rotation_update_norm_sum = 0.0;
    int smoother_local_rejects_total = 0;
    int smoother_total_sweeps = 0;
    int fixed_lambda_hard_start_sweep = -1;
    int fixed_lambda_effective_start_sweep = -1;
    int fixed_lambda_factor_sweeps = 0;
    int eta_only_variable_sweeps = 0;
    int fixed_lambda_hot_eta_sweeps = 0;
    int fixed_lambda_hot_eta_entries = 0;
    int num_groups = 0;
    int coarse_dim = 0;
    int coarse_lambda_nnz = 0;
    double coarse_lambda_density = 0.0;
    double outer_total_sec = 0.0;
    double build_graph_sec = 0.0;
    double basis_build_sec = 0.0;
    double basis_block_build_sec = 0.0;
    double basis_eigensolver_sec = 0.0;
    double basis_copyout_sec = 0.0;
    double basis_min_eigenvalue = 0.0;
    double basis_max_eigenvalue = 0.0;
    int basis_negative_group_count = 0;
    int basis_nonpositive_group_count = 0;
    int basis_partial_attempt_count = 0;
    int basis_partial_converged_count = 0;
    int basis_full_eigensolver_count = 0;
    double coarse_lambda_sec = 0.0;
    double exact_joint_assembly_sec = 0.0;
    double exact_solve_sec = 0.0;
    double sweeps_sec = 0.0;
    double factor_pass_sec = 0.0;
    double variable_pass_sec = 0.0;
    double coarse_eta_sec = 0.0;
    double coarse_solve_sec = 0.0;
    double coarse_factorize_sec = 0.0;
    double coarse_backsolve_sec = 0.0;
    double prolong_inject_sec = 0.0;
    double apply_step_sec = 0.0;
    double objective_eval_sec = 0.0;
    double relinearize_transport_sec = 0.0;
    double relinearize_factor_sec = 0.0;
    double relinearize_reset_sec = 0.0;
    double relinearize_total_sec = 0.0;
    int precision_checks = 0;
    int precision_freezes = 0;
    int precision_thaws = 0;
    int first_precision_freeze_sweep = -1;
    double precision_residual = 0.0;
    int full_precision_sweeps = 0;
    int eta_only_sweeps = 0;
    int jacobi_sweeps = 0;
    int coarse_solves = 0;
    int cycle_message_rebuilds = 0;
    int coarse_reuse_attempts = 0;
    int coarse_reuse_accepts = 0;
    int coarse_reuse_fallbacks = 0;
    double coarse_reuse_max_accepted_residual = 0;
    double coarse_reuse_work_ratio = 0;
    int coarse_reuse_iterations = 0;
    int coarse_reuse_max_iterations = 0;
    int basis_exact_reuse_count = 0;
    double basis_exact_cache_sec = 0;
    int basis_partial_work_aborts = 0;
    int basis_partial_iterations = 0;
    int coarse_reuse_preconditioner_calls = 0;
    int coarse_reuse_matvec_calls = 0;
    bool coarse_amortized_refresh = false;
    double coarse_extra_work_before = 0;
    double coarse_extra_work_after = 0;
    double coarse_factor_work_estimate = 0;
    int precision_warm_slots=0;
    int precision_warm_projected=0;
    int precision_warm_failed=0;
    double precision_warm_max_projection=0;
    double precision_warm_sec=0;
    bool coarse_cholmod_supernodal=false;
    double nonlinear_huber_objective=0;
    int basis_prepass_sweeps=0;
    double basis_prepass_sec=0;
    double outer_pose_update_sec=0;
    double outer_acceptance_sec=0;
    int objective_evaluations=0;
    int message_lifts=0;
    int residual_stop_checks=0;
    double residual_stop_sec=0, residual_stop_relative=0;
    bool residual_stopped=false;
    double message_lift_sec=0;
    std::vector<SE3CycleAuditRow> cycle_audit;
    int defect_mean_sweeps=0, defect_map_builds=0, defect_clamps=0;
    double defect_build_sec=0, defect_inverse_residual=0;
};

struct SyntheticSE3ExperimentResults {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int num_poses = 0;
    int num_edges = 0;
    double solver_wall_sec = 0.0;
    double initial_objective = 0.0;
    std::vector<SyntheticSE3OuterDirectRow> direct_history;
    std::vector<SyntheticSE3OuterMGRow> mg_history;
    SE3PoseVector initial_poses;
    SE3PoseHistory direct_pose_history;
    SE3PoseHistory mg_pose_history;
};

SyntheticSE3Problem loadSyntheticSE3Problem(const std::string& path);

double nonlinearObjective(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& poses
);

SE3PoseVector applyPoseDeltas(
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta_vec
);

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    double tiny_prior = 1e-12,
    const RobustLossConfig& robust_loss_config = {}
);

SyntheticSE3ExperimentResults runSyntheticSE3Experiment(
    const SyntheticSE3Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config = {},
    int sync_num_threads = 1,
    bool direct_enabled = true
);

void writeSyntheticSE3ExperimentResultsJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config = {},
    int sync_num_threads = 1,
    bool direct_enabled = true
);

void writeSyntheticSE3ExperimentPoseHistoryJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path
);

}  // namespace slam
