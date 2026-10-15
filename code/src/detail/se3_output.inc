void writeSyntheticSE3ExperimentResultsJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config,
    int sync_num_threads,
    bool direct_enabled
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open output JSON path: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"worker_affinity\": \"scoped_ascending_allowed_cpus\",\n";
    out << "  \"worker_affinity_failures\": " << ScopedWorkerAffinity::failures.load() << ",\n";
    out << "  \"max_observed_sweep_threads\": " << ScopedWorkerAffinity::max_team.load() << ",\n";
    out << "  \"basis_backend\": \"exact_band_selected_else_original_partial_eigen\",\n";
    out << "  \"basis_eigenvalue_scope\": \"selected_for_partial_and_band_full_for_eigen\",\n";
    out << "  \"basis_band_calls\": " << BandedSymmetricEigenWorkspace::calls.load() << ",\n";
    out << "  \"basis_band_failures\": " << BandedSymmetricEigenWorkspace::failures.load() << ",\n";
    out << "  \"solver_wall_sec\": " << jsonNumber(results.solver_wall_sec) << ",\n";
    out << "  \"config\": {\n";
    out << "    \"outer_line_search_objective\": \""
        << (se3EnvFlagEnabled("HGBP_SE3_ROBUST_LINE_SEARCH")?"huber":"raw") << "\",\n";
    out << "    \"buffered_objective\": " << (se3EnvFlagEnabled("HGBP_SE3_BUFFERED_OBJECTIVE")?"true":"false") << ",\n";
    out << "    \"basis_prepass\": \"" << (std::getenv("HGBP_SE3_BASIS_PREPASS")?
        std::getenv("HGBP_SE3_BASIS_PREPASS"):"none") << "\",\n";
    out << "    \"num_outer\": " << num_outer << ",\n";
    out << "    \"adaptive_precision\": "
        << (se3EnvFlagEnabled("HGBP_SE3_ADAPTIVE_PRECISION") ? "true" : "false") << ",\n";
    out << "    \"cycle_energy_krylov\": "
        << (se3EnvFlagEnabled("HGBP_CYCLE_ENERGY") && !se3EnvFlagEnabled("HGBP_CYCLE_ENERGY_RESTART") ? "true" : "false") << ",\n";
    out << "    \"cycle_line_search\": " << (se3EnvFlagEnabled("HGBP_CYCLE_ENERGY_RESTART")?"true":"false") << ",\n";
    out << "    \"persistent_sweeps\": " << (se3EnvFlagEnabled("HGBP_SE3_PERSISTENT_SWEEPS")?"true":"false") << ",\n";
    out << "    \"smoother\": \"" << (std::getenv("HGBP_SE3_SMOOTHER")?std::getenv("HGBP_SE3_SMOOTHER"):"gbp") << "\",\n";
    out << "    \"cycle_message_rebuild\": " << (se3EnvFlagEnabled("HGBP_SE3_CYCLE_MESSAGE_REBUILD")?"true":"false") << ",\n";
    out << "    \"automatic_coarse\": " << (se3EnvFlagEnabled("HGBP_SE3_AUTOMATIC_COARSE")?"true":"false") << ",\n";
    out << "    \"coarse_linear_backend\": \""
#ifdef HGBP_PURE_GBP
        << "pure_gbp"
#else
        << (se3EnvFlagEnabled("HGBP_SE3_COARSE_CHOLMOD") ? "cholmod_auto" : "eigen_simplicial_ldlt")
#endif
        << "\",\n";
    out << "    \"boundary_basis\": \"" << (std::getenv("HGBP_SE3_BOUNDARY_BASIS")?
        std::getenv("HGBP_SE3_BOUNDARY_BASIS"):"none") << "\",\n";
    out << "    \"precision_initialization\": \"" << (std::getenv("HGBP_SE3_PRECISION_INITIALIZATION")?std::getenv("HGBP_SE3_PRECISION_INITIALIZATION"):"cold") << "\",\n";
    out << "    \"diagnostic_precision_audit\": " << ((std::getenv("HGBP_SE3_PRECISION_AUDIT") && std::getenv("HGBP_SE3_PRECISION_AUDIT")[0])?"true":"false") << ",\n";
    out << "    \"skip_coarse_quality_ablation\": " << (se3EnvFlagEnabled("HGBP_SE3_SKIP_COARSE")?"true":"false") << ",\n";
    out << "    \"precision_tolerance\": " << SE3PrecisionPolicy::tolerance << ",\n";
    out << "    \"precision_check_period\": " << SE3PrecisionPolicy::check_period << ",\n";
    out << "    \"precision_stable_checks\": " << SE3PrecisionPolicy::stable_checks << ",\n";
    out << "    \"frozen_precision_check_period\": " << SE3PrecisionPolicy::frozen_check_period << ",\n";
    out << "    \"inner_cycles\": " << inner_cycles << ",\n";
    out << "    \"pre_sweeps\": " << pre_sweeps << ",\n";
    out << "    \"group_size\": " << group_size << ",\n";
    out << "    \"r_reduced\": " << r_reduced << ",\n";
    out << "    \"basis_rebuild_period\": " << se3BasisRebuildPeriod() << ",\n";
    out << "    \"basis_rebuild_warmup_outers\": " << se3BasisRebuildWarmupOuters() << ",\n";
    out << "    \"basis_source\": \"message_conditioned_information_generic\",\n";
    out << "    \"partial_basis_accept_residual_tol\": "
        << jsonNumber(se3EnvDoubleOrDefault("GBP_SE3_PARTIAL_BASIS_ACCEPT_RESIDUAL_TOL", -1.0))
        << ",\n";
    out << "    \"partial_basis_large_group_min_groups\": "
        << se3PartialBasisLargeGroupMinGroups() << ",\n";
    out << "    \"partial_basis_large_group_accept_residual_tol\": "
        << jsonNumber(se3PartialBasisLargeGroupAcceptResidualTol()) << ",\n";
    out << "    \"partial_basis_small_group_accept_residual_tol\": "
        << jsonNumber(se3PartialBasisSmallGroupAcceptResidualTol()) << ",\n";
    out << "    \"partial_basis_residual_check_period\": "
        << std::max(
               1,
               se3EnvIntOrDefault("GBP_SE3_PARTIAL_BASIS_RESIDUAL_CHECK_PERIOD", 1)
           )
        << ",\n";
    out << "    \"sync_num_threads\": " << effectiveThreadCount(sync_num_threads) << ",\n";
    out << "    \"enable_singlecore_fastsync\": false,\n";
    out << "    \"single_thread_runtime_guard\": "
        << (se3SingleThreadRuntimeGuardEnabled() ? "true" : "false") << ",\n";
    out << "    \"direct_enabled\": " << (direct_enabled ? "true" : "false") << ",\n";
    out << "    \"robust_huber_delta\": " << jsonNumber(robust_loss_config.huber_delta) << ",\n";
    const std::string saved_precision_initialization=std::getenv("HGBP_SE3_PRECISION_INITIALIZATION")?
        std::getenv("HGBP_SE3_PRECISION_INITIALIZATION"):"cold";
    out << "    \"transport_policy\": \""
        << (saved_precision_initialization!="cold"?saved_precision_initialization:
            (se3EnvFlagEnabled("HGBP_SE3_RESET_TRANSPORT")?"reset":"legacy_outer12_eta_shift")) << "\",\n";
    out << "    \"legacy_outer12_transport_enabled\": "
        << (se3EnvFlagEnabled("HGBP_SE3_RESET_TRANSPORT")?"false":"true") << ",\n";
    out << "    \"message_initialization\": \""
        << (se3EnvFlagEnabled("HGBP_SE3_BALANCED_INITIALIZATION")?"balanced_after_basis":"legacy_zero_or_transport") << "\",\n";
    out << "    \"reuse_coarse_pattern\": "
        << (se3ReuseCoarsePatternEnabled() ? "true" : "false") << ",\n";
    out << "    \"exact_basis_cache\": "
        << (se3EnvFlagEnabled("HGBP_SE3_EXACT_BASIS_CACHE") ? "true" : "false") << ",\n";
    out << "    \"eta_lift\": \"" << (std::getenv("HGBP_SE3_ETA_LIFT")?std::getenv("HGBP_SE3_ETA_LIFT"):"none") << "\",\n";
    out << "    \"cycle_audit_enabled\": " << (se3EnvFlagEnabled("HGBP_SE3_CYCLE_AUDIT")?"true":"false") << ",\n";
    out << "    \"residual_stop_enabled\": " << (se3EnvFlagEnabled("HGBP_SE3_RESIDUAL_STOP")?"true":"false") << ",\n";
    out << "    \"residual_stop_tolerance\": 0.1,\n";
    out << "    \"precision_defect\": \"" << (std::getenv("HGBP_SE3_PRECISION_DEFECT")?std::getenv("HGBP_SE3_PRECISION_DEFECT"):"none") << "\",\n";
    out << "    \"amortized_coarse\": "
        << (se3EnvFlagEnabled("HGBP_SE3_AMORTIZED_COARSE") ? "true" : "false") << ",\n";
    out << "    \"predict_partial_budget\": "
        << (se3EnvFlagEnabled("HGBP_SE3_PREDICT_PARTIAL_BUDGET") ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_lambda\": "
        << (se3CachedCoarseLambdaEnabled() ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_inplace_symmetrize\": "
        << (se3CachedCoarseInplaceSymmetrizeEnabled() ? "true" : "false") << ",\n";
    out << "    \"cached_coarse_lambda_min_groups\": "
        << se3CachedCoarseLambdaMinGroups() << ",\n";
    out << "    \"coarse_assembly_shards\": "
        << kSE3CoarseAssemblyShards << ",\n";
    out << "    \"coarse_numeric_rebuild_period\": "
        << se3CoarseNumericRebuildPeriod() << ",\n";
    out << "    \"coarse_reuse_pcg_iterations\": "
        << se3CoarseReusePcgIterations() << ",\n";
    out << "    \"implicit_fine_operator\": "
        << (se3ImplicitFineOperatorEnabled() ? "true" : "false") << ",\n";
    out << "    \"implicit_fine_operator_threads\": "
        << se3EnvIntOrDefault("GBP_SE3_IMPLICIT_FINE_OPERATOR_THREADS", 0)
        << ",\n";
    out << "    \"final_direct_polish_steps\": "
        << se3FinalDirectPolishSteps() << "\n";
    out << "  },\n";
    out << "  \"problem\": {\n";
    out << "    \"num_poses\": " << results.num_poses << ",\n";
    out << "    \"num_edges\": " << results.num_edges << "\n";
    out << "  },\n";
    out << "  \"initial_objective\": " << jsonNumber(results.initial_objective) << ",\n";
    out << "  \"direct_history\": [\n";
    for (size_t i = 0; i < results.direct_history.size(); ++i) {
        const SyntheticSE3OuterDirectRow& row = results.direct_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"linear_step_norm\": " << jsonNumber(row.linear_step_norm)
            << ", \"linear_residual_norm\": " << jsonNumber(row.linear_residual_norm)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
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
        const SyntheticSE3OuterMGRow& row = results.mg_history[i];
        out << "    {\"outer\": " << row.outer
            << ", \"nonlinear_objective\": " << jsonNumber(row.nonlinear_objective)
            << ", \"raw_e_hat_norm\": " << jsonNumber(row.raw_e_hat_norm)
            << ", \"nonlinear_huber_objective\": ";
        if(se3EnvFlagEnabled("HGBP_SE3_ROBUST_LINE_SEARCH")) out << jsonNumber(row.nonlinear_huber_objective);
        else out << "null";
        out << ", \"basis_prepass_sweeps\": " << row.basis_prepass_sweeps
            << ", \"outer_pose_update_sec\": " << jsonNumber(row.outer_pose_update_sec)
            << ", \"outer_acceptance_sec\": " << jsonNumber(row.outer_acceptance_sec)
            << ", \"objective_evaluations\": " << row.objective_evaluations
            << ", \"basis_prepass_sec\": " << jsonNumber(row.basis_prepass_sec)
            << ", \"precision_checks\": " << row.precision_checks
            << ", \"precision_freezes\": " << row.precision_freezes
            << ", \"precision_thaws\": " << row.precision_thaws
            << ", \"first_precision_freeze_sweep\": " << row.first_precision_freeze_sweep
            << ", \"precision_residual\": " << jsonNumber(row.precision_residual)
            << ", \"full_precision_sweeps\": " << row.full_precision_sweeps
            << ", \"eta_only_sweeps\": " << row.eta_only_sweeps
            << ", \"e_hat_norm\": " << jsonNumber(row.e_hat_norm)
            << ", \"first_cycle_residual_before_sweeps\": "
            << jsonNumber(row.first_cycle_residual_before_sweeps)
            << ", \"first_cycle_residual_after_sweeps\": "
            << jsonNumber(row.first_cycle_residual_after_sweeps)
            << ", \"first_cycle_residual_after_coarse\": "
            << jsonNumber(row.first_cycle_residual_after_coarse)
            << ", \"last_cycle_residual_before_sweeps\": "
            << jsonNumber(row.last_cycle_residual_before_sweeps)
            << ", \"last_cycle_residual_after_sweeps\": "
            << jsonNumber(row.last_cycle_residual_after_sweeps)
            << ", \"last_cycle_residual_after_coarse\": "
            << jsonNumber(row.last_cycle_residual_after_coarse)
            << ", \"linear_residual_approx\": " << jsonNumber(row.linear_residual_approx)
            << ", \"line_search_step_scale\": " << jsonNumber(row.line_search_step_scale)
            << ", \"line_search_rejects\": " << row.line_search_rejects
            << ", \"executed_inner_cycles\": " << row.executed_inner_cycles
            << ", \"smoother_translation_update_norm_sum\": "
            << jsonNumber(row.smoother_translation_update_norm_sum)
            << ", \"smoother_rotation_update_norm_sum\": "
            << jsonNumber(row.smoother_rotation_update_norm_sum)
            << ", \"smoother_local_rejects_total\": " << row.smoother_local_rejects_total
            << ", \"smoother_total_sweeps\": " << row.smoother_total_sweeps
            << ", \"jacobi_sweeps\": " << row.jacobi_sweeps
            << ", \"coarse_solves\": " << row.coarse_solves
            << ", \"residual_stop_checks\": " << row.residual_stop_checks
            << ", \"residual_stop_sec\": " << row.residual_stop_sec
            << ", \"residual_stop_relative\": " << row.residual_stop_relative
            << ", \"residual_stopped\": " << (row.residual_stopped?"true":"false")
            << ", \"cycle_message_rebuilds\": " << row.cycle_message_rebuilds
            << ", \"message_lifts\": " << row.message_lifts
            << ", \"defect_mean_sweeps\": " << row.defect_mean_sweeps
            << ", \"defect_map_builds\": " << row.defect_map_builds
            << ", \"defect_clamps\": " << row.defect_clamps
            << ", \"defect_build_sec\": " << jsonNumber(row.defect_build_sec)
            << ", \"defect_inverse_residual\": " << jsonNumber(row.defect_inverse_residual)
            << ", \"message_lift_sec\": " << jsonNumber(row.message_lift_sec)
            << ", \"coarse_reuse_attempts\": " << row.coarse_reuse_attempts
            << ", \"coarse_reuse_accepts\": " << row.coarse_reuse_accepts
            << ", \"coarse_reuse_fallbacks\": " << row.coarse_reuse_fallbacks
            << ", \"coarse_reuse_max_accepted_residual\": " << jsonNumber(row.coarse_reuse_max_accepted_residual)
            << ", \"coarse_reuse_work_ratio\": " << jsonNumber(row.coarse_reuse_work_ratio)
            << ", \"coarse_reuse_iterations\": " << row.coarse_reuse_iterations
            << ", \"coarse_reuse_max_iterations\": " << row.coarse_reuse_max_iterations
            << ", \"basis_exact_reuse_count\": " << row.basis_exact_reuse_count
            << ", \"basis_exact_cache_sec\": " << jsonNumber(row.basis_exact_cache_sec)
            << ", \"basis_partial_work_aborts\": " << row.basis_partial_work_aborts
            << ", \"basis_partial_iterations\": " << row.basis_partial_iterations
            << ", \"coarse_reuse_preconditioner_calls\": " << row.coarse_reuse_preconditioner_calls
            << ", \"coarse_reuse_matvec_calls\": " << row.coarse_reuse_matvec_calls
            << ", \"coarse_amortized_refresh\": " << (row.coarse_amortized_refresh ? "true" : "false")
            << ", \"coarse_cholmod_supernodal\": " << (row.coarse_cholmod_supernodal ? "true" : "false")
            << ", \"coarse_extra_work_before\": " << jsonNumber(row.coarse_extra_work_before)
            << ", \"coarse_extra_work_after\": " << jsonNumber(row.coarse_extra_work_after)
            << ", \"coarse_factor_work_estimate\": " << jsonNumber(row.coarse_factor_work_estimate)
            << ", \"precision_warm_slots\": " << row.precision_warm_slots
            << ", \"precision_warm_projected\": " << row.precision_warm_projected
            << ", \"precision_warm_failed\": " << row.precision_warm_failed
            << ", \"precision_warm_max_projection\": " << jsonNumber(row.precision_warm_max_projection)
            << ", \"precision_warm_sec\": " << jsonNumber(row.precision_warm_sec)
            << ", \"fixed_lambda_hard_start_sweep\": " << row.fixed_lambda_hard_start_sweep
            << ", \"fixed_lambda_effective_start_sweep\": "
            << row.fixed_lambda_effective_start_sweep
            << ", \"fixed_lambda_factor_sweeps\": " << row.fixed_lambda_factor_sweeps
            << ", \"eta_only_variable_sweeps\": " << row.eta_only_variable_sweeps
            << ", \"fixed_lambda_hot_eta_sweeps\": " << row.fixed_lambda_hot_eta_sweeps
            << ", \"fixed_lambda_hot_eta_entries\": " << row.fixed_lambda_hot_eta_entries
            << ", \"num_groups\": " << row.num_groups
            << ", \"coarse_dim\": " << row.coarse_dim
            << ", \"coarse_lambda_nnz\": " << row.coarse_lambda_nnz
            << ", \"coarse_lambda_density\": " << jsonNumber(row.coarse_lambda_density)
            << ", \"outer_total_sec\": " << jsonNumber(row.outer_total_sec)
            << ", \"build_graph_sec\": " << jsonNumber(row.build_graph_sec)
            << ", \"basis_build_sec\": " << jsonNumber(row.basis_build_sec)
            << ", \"basis_block_build_sec\": " << jsonNumber(row.basis_block_build_sec)
            << ", \"basis_eigensolver_sec\": " << jsonNumber(row.basis_eigensolver_sec)
            << ", \"basis_copyout_sec\": " << jsonNumber(row.basis_copyout_sec)
            << ", \"basis_min_eigenvalue\": " << jsonNumber(row.basis_min_eigenvalue)
            << ", \"basis_max_eigenvalue\": " << jsonNumber(row.basis_max_eigenvalue)
            << ", \"basis_negative_group_count\": " << row.basis_negative_group_count
            << ", \"basis_nonpositive_group_count\": " << row.basis_nonpositive_group_count
            << ", \"basis_partial_attempt_count\": " << row.basis_partial_attempt_count
            << ", \"basis_partial_converged_count\": " << row.basis_partial_converged_count
            << ", \"basis_full_eigensolver_count\": " << row.basis_full_eigensolver_count
            << ", \"coarse_lambda_sec\": " << jsonNumber(row.coarse_lambda_sec)
            << ", \"exact_joint_assembly_sec\": " << jsonNumber(row.exact_joint_assembly_sec)
            << ", \"sweeps_sec\": " << jsonNumber(row.sweeps_sec)
            << ", \"factor_pass_sec\": " << jsonNumber(row.factor_pass_sec)
            << ", \"variable_pass_sec\": " << jsonNumber(row.variable_pass_sec)
            << ", \"coarse_eta_sec\": " << jsonNumber(row.coarse_eta_sec)
            << ", \"coarse_solve_sec\": " << jsonNumber(row.coarse_solve_sec)
            << ", \"coarse_factorize_sec\": " << jsonNumber(row.coarse_factorize_sec)
            << ", \"coarse_backsolve_sec\": " << jsonNumber(row.coarse_backsolve_sec)
            << ", \"prolong_inject_sec\": " << jsonNumber(row.prolong_inject_sec)
            << ", \"apply_step_sec\": " << jsonNumber(row.apply_step_sec)
            << ", \"objective_eval_sec\": " << jsonNumber(row.objective_eval_sec)
            << ", \"relinearize_transport_sec\": "
            << jsonNumber(row.relinearize_transport_sec)
            << ", \"relinearize_factor_sec\": "
            << jsonNumber(row.relinearize_factor_sec)
            << ", \"relinearize_reset_sec\": "
            << jsonNumber(row.relinearize_reset_sec)
            << ", \"relinearize_total_sec\": "
            << jsonNumber(row.relinearize_total_sec)
            << ", \"cycle_audit\": [";
        for(size_t j=0;j<row.cycle_audit.size();++j) {
            const auto& a=row.cycle_audit[j];
            if(j) out<<",";
            out<<"{\"cycle\":"<<a.cycle
                <<",\"initial_residual\":"<<jsonNumber(a.initial_residual)
                <<",\"fine_residual\":"<<jsonNumber(a.fine_residual)
                <<",\"coarse_residual\":"<<jsonNumber(a.coarse_residual)
                <<",\"final_residual\":"<<jsonNumber(a.final_residual)
                <<",\"fine_gain\":"<<jsonNumber(a.fine_gain)
                <<",\"coarse_gain\":"<<jsonNumber(a.coarse_gain)
                <<",\"total_gain\":"<<jsonNumber(a.total_gain)
                <<",\"fine_energy\":"<<jsonNumber(a.fine_energy)
                <<",\"coarse_energy\":"<<jsonNumber(a.coarse_energy)
                <<",\"cross_energy\":"<<jsonNumber(a.cross_energy)
                <<",\"residual_defect_absolute\":"<<jsonNumber(a.residual_defect_absolute)
                <<",\"residual_defect_relative\":"<<jsonNumber(a.residual_defect_relative)
                <<",\"eta_defect_before\":"<<jsonNumber(a.eta_defect_before)
                <<",\"eta_defect_after\":"<<jsonNumber(a.eta_defect_after)<<"}";
        }
        out<<"]}";
        out << (i + 1 == results.mg_history.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

void writeSyntheticSE3ExperimentPoseHistoryJson(
    const SyntheticSE3ExperimentResults& results,
    const std::string& path
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open pose-history JSON path: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"num_poses\": " << results.num_poses << ",\n";
    out << "  \"num_edges\": " << results.num_edges << ",\n";
    out << "  \"initial_poses\": ";
    writePoseVectorJson(out, results.initial_poses, 4);
    out << ",\n";
    out << "  \"direct_pose_history\": ";
    writePoseHistoryJsonArray(out, results.direct_pose_history, 4);
    out << ",\n";
    out << "  \"mg_pose_history\": ";
    writePoseHistoryJsonArray(out, results.mg_pose_history, 4);
    out << "\n";
    out << "}\n";
}
