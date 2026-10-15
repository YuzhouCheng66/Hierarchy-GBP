void writeExperimentResultsJson(
    const ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    int sync_num_threads,
    gbp::FactorGraph::SyncScheduleKind sync_schedule_kind,
    int sync_schedule_chunk,
    bool profile_sync_thread_utilization,
    bool enable_singlecore_fastsync,
    const BasisBuildConfig& basis_build_config,
    const RobustLossConfig& robust_loss_config,
    int final_coarse_polish_passes,
    double packed_jitter,
    int fixed_eta_after_sweeps
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output JSON path: " + path);
    }

    out << "{\n";
    out << "  \"worker_affinity\": \"scoped_ascending_allowed_cpus\",\n";
    out << "  \"worker_affinity_failures\": " << ScopedWorkerAffinity::failures.load() << ",\n";
    out << "  \"solver_wall_sec\": " << jsonNumber(results.solver_wall_sec) << ",\n";
    out << "  \"hgbp_solver_wall_sec\": " << jsonNumber(results.hgbp_solver_wall_sec) << ",\n";
    out << "  \"direct_solver_wall_sec\": " << jsonNumber(results.direct_solver_wall_sec) << ",\n";
    out << "  \"config\": {\n";
    out << "    \"num_outer\": " << num_outer << ",\n";
    out << "    \"adaptive_precision\": " << (getenvEnabled("HGBP_SE2_ADAPTIVE_PRECISION")?"true":"false") << ",\n";
    out << "    \"precision_tolerance\": " << SE2PrecisionPolicy::tolerance << ",\n";
    out << "    \"precision_check_period\": " << SE2PrecisionPolicy::check_period << ",\n";
    out << "    \"precision_check_backoff\": true,\n";
    out << "    \"precision_max_check_period\": " << SE2PrecisionPolicy::max_check_period << ",\n";
    out << "    \"precision_stable_checks\": " << SE2PrecisionPolicy::stable_checks << ",\n";
    out << "    \"frozen_precision_check_period\": " << SE2PrecisionPolicy::frozen_check_period << ",\n";
    out << "    \"direct_reference_enabled\": " << (results.direct_history.size()>1?"true":"false") << ",\n";
    out << "    \"direct_reference_only\": " << (results.mg_history.empty() && results.direct_history.size()>1?"true":"false") << ",\n";
    out << "    \"cycle_energy_krylov\": " << (getenvEnabled("HGBP_CYCLE_ENERGY")?"true":"false") << ",\n";
    out << "    \"residual_cycles\": " << (getenvEnabled("HGBP_RESIDUAL_CYCLES")?"true":"false") << ",\n";
    out << "    \"diagnostic_only\": " << (getenvEnabled("HGBP_LINEAR_AUDIT")?"true":"false") << ",\n";
    out << "    \"eta_relaxation\": " << (std::getenv("HGBP_SE2_ETA_RELAXATION") ? std::getenv("HGBP_SE2_ETA_RELAXATION") : "1") << ",\n";
    out << "    \"cholesky_schur\": " << (getenvEnabled("HGBP_SE2_CHOLESKY_SCHUR")?"true":"false") << ",\n";
    out << "    \"message_lift\": " << (getenvEnabled("HGBP_SE2_MESSAGE_LIFT")?"true":"false") << ",\n";
    out << "    \"message_initialization\": " << (std::getenv("HGBP_SE2_MESSAGE_INITIALIZATION")?std::getenv("HGBP_SE2_MESSAGE_INITIALIZATION"):"0") << ",\n";
    out << "    \"smoother\": \"" << (std::getenv("HGBP_SE2_SMOOTHER")?std::getenv("HGBP_SE2_SMOOTHER"):"gbp") << "\",\n";
    out << "    \"skip_coarse_quality_ablation\": " << (getenvEnabled("HGBP_SE2_SKIP_COARSE")?"true":"false") << ",\n";
    out << "    \"inner_cycles\": " << inner_cycles << ",\n";
    out << "    \"pre_sweeps\": " << pre_sweeps << ",\n";
    out << "    \"group_size\": " << group_size << ",\n";
    out << "    \"r_reduced\": " << r_reduced << ",\n";
    out << "    \"sync_num_threads\": " << sync_num_threads << ",\n";
    out << "    \"sync_schedule\": \"" << syncScheduleName(sync_schedule_kind) << "\",\n";
    out << "    \"sync_schedule_chunk\": " << sync_schedule_chunk << ",\n";
    out << "    \"profile_sync_thread_utilization\": " << (profile_sync_thread_utilization ? "true" : "false") << ",\n";
    out << "    \"enable_singlecore_fastsync\": " << (enable_singlecore_fastsync ? "true" : "false") << ",\n";
    out << "    \"final_coarse_polish_passes\": " << std::max(0, final_coarse_polish_passes) << ",\n";
    out << "    \"basis_eigensolver\": \"" << basisEigensolverName(basis_build_config.eigensolver) << "\",\n";
    out << "    \"basis_warm_start\": " << (basis_build_config.enable_warm_start ? "true" : "false") << ",\n";
    out << "    \"basis_packed_builder\": " << (basis_build_config.enable_packed_builder ? "true" : "false") << ",\n";
    out << "    \"basis_packed_residual_solver\": " << (basis_build_config.enable_packed_residual_solver ? "true" : "false") << ",\n";
    out << "    \"packed_basis_thread_override\": " << basis_build_config.packed_basis_thread_override << ",\n";
    out << "    \"packed_coarse_thread_override\": " << basis_build_config.packed_coarse_thread_override << ",\n";
    out << "    \"packed_sweep_thread_override\": " << basis_build_config.packed_sweep_thread_override << ",\n";
    out << "    \"basis_rebuild_period\": " << basis_build_config.basis_rebuild_period << ",\n";
    out << "    \"basis_rebuild_warmup_outers\": " << basis_build_config.basis_rebuild_warmup_outers << ",\n";
    out << "    \"partial_oversampling\": " << basis_build_config.partial_oversampling << ",\n";
    out << "    \"partial_max_iters\": " << basis_build_config.partial_max_iters << ",\n";
    out << "    \"partial_residual_check_period\": " << basis_build_config.partial_residual_check_period << ",\n";
    out << "    \"partial_residual_tol\": " << jsonNumber(basis_build_config.partial_residual_tol) << ",\n";
    out << "    \"partial_ridge\": " << jsonNumber(basis_build_config.partial_ridge) << ",\n";
    out << "    \"robust_huber_delta\": " << jsonNumber(robust_loss_config.huber_delta) << ",\n";
    out << "    \"packed_jitter\": " << jsonNumber(packed_jitter) << ",\n";
    out << "    \"fixed_eta_after_sweeps\": " << fixed_eta_after_sweeps << ",\n";
    out << "    \"basis_source\": \"message_conditioned_information\"\n";
    out << "  },\n";
    out << "  \"problem\": {\n";
    out << "    \"num_poses\": " << results.num_poses << ",\n";
    out << "    \"num_edges\": " << results.num_edges << "\n";
    out << "  },\n";
#ifdef HGBP_PURE_GBP
    out << "  \"coarse_linear_backend\": \"pure_gbp\",\n";
#else
    out << "  \"coarse_linear_backend\": \"cholesky\",\n";
#endif
    out << "  \"initial_objective\": " << jsonNumber(results.initial_objective) << ",\n";
    out << "  \"residual_workspace_build_sec\": " << jsonNumber(results.residual_workspace_build_sec) << ",\n";
    out << "  \"residual_workspace_variable_alloc_sec\": " << jsonNumber(results.residual_workspace_variable_alloc_sec) << ",\n";
    out << "  \"residual_workspace_factor_alloc_sec\": " << jsonNumber(results.residual_workspace_factor_alloc_sec) << ",\n";
    out << "  \"residual_workspace_connect_sec\": " << jsonNumber(results.residual_workspace_connect_sec) << ",\n";
    out << "  \"packed_schur_adjugate_fallback_count\": " << results.packed_schur_adjugate_fallback_count << ",\n";
    out << "  \"packed_schur_general_failure_count\": " << results.packed_schur_general_failure_count << ",\n";
    out << "  \"packed_kernel_prep_cycles\": " << results.packed_kernel_prep_cycles << ",\n";
    out << "  \"packed_kernel_schur0_cycles\": " << results.packed_kernel_schur0_cycles << ",\n";
    out << "  \"packed_kernel_schur1_cycles\": " << results.packed_kernel_schur1_cycles << ",\n";
    out << "  \"packed_kernel_update_belief_cycles\": " << results.packed_kernel_update_belief_cycles << ",\n";
    out << "  \"packed_fixedeta_full_lambda_sweeps\": " << results.packed_fixedeta_full_lambda_sweeps << ",\n";
    out << "  \"packed_fixedeta_lambda_init_sweeps\": " << results.packed_fixedeta_lambda_init_sweeps << ",\n";
    out << "  \"packed_fixedeta_eta_only_sweeps\": " << results.packed_fixedeta_eta_only_sweeps << ",\n";
    out << "  \"packed_fixedeta_eta_map_builds\": " << results.packed_fixedeta_eta_map_builds << ",\n";
    out << "  \"packed_fixedeta_serial_delta_sweeps\": " << results.packed_fixedeta_serial_delta_sweeps << ",\n";

    out << "  \"direct_history\": [\n";
    for (size_t i = 0; i < results.direct_history.size(); ++i) {
        const OuterDirectRow& row = results.direct_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"linear_step_norm\": " << jsonNumber(row.linear_step_norm)
            << ", \"linear_residual_norm\": " << jsonNumber(row.linear_residual_norm)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"factor_relinearize_sec\": " << jsonNumber(row.factor_relinearize_sec)
            << ", \"reset_state_sec\": " << jsonNumber(row.reset_state_sec)
            << ", \"joint_assembly_sec\": " << jsonNumber(row.joint_assembly_sec)
            << ", \"exact_solve_sec\": " << jsonNumber(row.exact_solve_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << "}";
        out << (i + 1 == results.direct_history.size() ? "\n" : ",\n");
    }
    out << "  ],\n";

    out << "  \"mg_history\": [\n";
    for (size_t i = 0; i < results.mg_history.size(); ++i) {
        const OuterMGRow& row = results.mg_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"execution_path\": \"" << row.execution_path << "\""
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"e_hat_norm\": " << jsonNumber(row.e_hat_norm)
            << ", \"factor_lam_max_asym\": " << jsonNumber(row.factor_lam_max_asym)
            << ", \"prefer_nonsym_safe_fastsync\": " << (row.prefer_nonsym_safe_fastsync ? "true" : "false")
            << ", \"num_groups\": " << row.num_groups
            << ", \"coarse_dim\": " << row.coarse_dim
            << ", \"coarse_lambda_nnz\": " << row.coarse_lambda_nnz
            << ", \"coarse_lambda_density\": " << jsonNumber(row.coarse_lambda_density)
            << ", \"coarse_lambda_dense_factor\": " << (row.coarse_lambda_dense_factor ? "true" : "false")
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"factor_relinearize_sec\": " << jsonNumber(row.factor_relinearize_sec)
            << ", \"reset_state_sec\": " << jsonNumber(row.reset_state_sec)
            << ", \"basis_build_sec\": " << jsonNumber(row.basis_build_sec)
            << ", \"basis_block_build_sec\": " << jsonNumber(row.basis_block_build_sec)
            << ", \"basis_eigensolver_sec\": " << jsonNumber(row.basis_eigensolver_sec)
            << ", \"basis_copyout_sec\": " << jsonNumber(row.basis_copyout_sec)
            << ", \"coarse_lambda_sec\": " << jsonNumber(row.coarse_lambda_sec)
            << ", \"sweeps_sec\": " << jsonNumber(row.sweeps_sec)
            << ", \"factor_pass_sec\": " << jsonNumber(row.factor_pass_sec)
            << ", \"variable_pass_sec\": " << jsonNumber(row.variable_pass_sec)
            << ", \"factor_work_sec\": " << jsonNumber(row.factor_work_sec)
            << ", \"factor_wait_sec\": " << jsonNumber(row.factor_wait_sec)
            << ", \"variable_work_sec\": " << jsonNumber(row.variable_work_sec)
            << ", \"variable_wait_sec\": " << jsonNumber(row.variable_wait_sec)
            << ", \"sync_profiled_threads\": " << row.sync_profiled_threads
            << ", \"coarse_eta_sec\": " << jsonNumber(row.coarse_eta_sec)
            << ", \"coarse_solve_sec\": " << jsonNumber(row.coarse_solve_sec)
            << ", \"prolong_inject_sec\": " << jsonNumber(row.prolong_inject_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << ", \"basis_partial_attempted\": " << row.basis_partial_attempted
            << ", \"basis_partial_converged\": " << row.basis_partial_converged
            << ", \"basis_partial_fallback\": " << row.basis_partial_fallback
            << ", \"basis_partial_total_iters\": " << row.basis_partial_total_iters
            << ", \"gbp_sweeps_executed\": " << row.gbp_sweeps_executed
            << ", \"jacobi_sweeps_executed\": " << row.jacobi_sweeps_executed
            << ", \"coarse_solves\": " << row.coarse_solves
            << ", \"full_precision_sweeps\": " << row.full_precision_sweeps
            << ", \"eta_only_sweeps\": " << row.eta_only_sweeps
            << ", \"precision_checks\": " << row.precision_checks
            << ", \"precision_freezes\": " << row.precision_freezes
            << ", \"precision_thaws\": " << row.precision_thaws
            << ", \"first_precision_freeze_sweep\": " << row.first_precision_freeze_sweep
            << ", \"precision_residual\": " << jsonNumber(row.precision_residual)
            << "}";
        out << (i + 1 == results.mg_history.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void writeExperimentPoseHistoryJson(
    const ExperimentResults& results,
    const std::string& path
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open pose-history JSON path: " + path);
    }
    out << "{\n";
    out << "  \"num_poses\": " << results.num_poses << ",\n";
    out << "  \"num_edges\": " << results.num_edges << ",\n";
    out << "  \"initial_poses\": ";
    writePoseVectorJson(out, results.initial_poses, 2);
    out << ",\n";
    out << "  \"direct_pose_history\": ";
    writePoseHistoryJsonArray(out, results.direct_pose_history, 2);
    out << ",\n";
    out << "  \"mg_pose_history\": ";
    writePoseHistoryJsonArray(out, results.mg_pose_history, 2);
    out << "\n";
    out << "}\n";
}
