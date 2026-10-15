void writeRootHGBPJson(
    const std::string& path,
    const BALProblem& problem,
    const Args& args,
    const HGBPConfig& lab,
    const RootHGBPResult& result
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "failed to open RootBA H-GBP JSON output: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"solver\": \"hgbp_ba\",\n";
    out << "  \"coarse_operator\": \"" << lab.coarse_operator << "\",\n";
    out << "  \"aggregation\": \"" << lab.aggregation << "\",\n";
    out << "  \"disconnected_groups\": " << result.disconnected_groups << ",\n";
    out << "  \"linear_controller\": \"" << lab.linear_controller << "\",\n";
    out << "  \"gbp_model\": \"" << lab.gbp_model << "\",\n";
    out << "  \"diagnostic_only\": " << "false" << ",\n";
    out << "  \"variance_rel_tol\": " << lab.variance_rel_tol << ",\n";
    out << "  \"configuration_policy\": \"normalized_covisibility_uniform_v1\",\n";
    out << "  \"graph_neighbors\": " << lab.graph_neighbors << ",\n";
    out << "  \"coarse_groups_budget\": " << lab.coarse_groups << ",\n";
    out << "  \"linear_rel_tol\": " << lab.linear_rel_tol << ",\n";
    out << "  \"num_cameras\": " << problem.num_cameras << ",\n";
    out << "  \"num_points\": " << problem.num_points << ",\n";
    out << "  \"num_observations\": " << problem.num_observations << ",\n";
    out << "  \"outer\": " << args.outer << ",\n";
    out << "  \"threads\": " << requestedGBPThreads(args) << ",\n";
    out << "  \"build_threads\": " << args.build_threads << ",\n";
    out << "  \"group_size\": " << (args.coarse_scale > 0.0 ?
        std::max(2, (problem.num_cameras + lab.coarse_groups - 1) / lab.coarse_groups) : 0) << ",\n";
    out << "  \"mg_cycles\": " << args.mg_cycles << ",\n";
    out << "  \"pre_sweeps\": " << args.pre_sweeps << ",\n";
    out << "  \"gbp_full_sweeps\": " << args.gbp_full_sweeps << ",\n";
    out << "  \"message_damping\": "
        << args.message_damping << ",\n";
    out << "  \"initial_lambda\": " << lab.initial_lambda << ",\n";
    out << "  \"pair_factor_scale\": "
        << lab.pair_factor_scale << ",\n";
    out << "  \"pair_backbone_min_coverage\": "
        << 0.9 << ",\n";
    out << "  \"krylov_start_outer\": "
        << lab.krylov_start_outer << ",\n";
    out << "  \"fine_smoother\": \""
        << "gbp"
        << "\",\n";
    out << "  \"min_pair_observations\": "
        << 1 << ",\n";
    out << "  \"pair_sample_cap\": "
        << lab.pair_sample_cap << ",\n";
    out << "  \"pair_sample_rescale\": "
        << "true"
        << ",\n";
    out << "  \"unreduced_unary\": "
        << "false"
        << ",\n";
    out << "  \"no_pose_scaling\": "
        << "false"
        << ",\n";
    out << "  \"preprocessor_sec\": " << result.preprocessor_sec << ",\n";
    out << "  \"preprocessor_linearization_sec\": "
        << result.preprocessor_linearization_sec << ",\n";
    out << "  \"preprocessor_topology_sec\": "
        << result.preprocessor_topology_sec << ",\n";
    out << "  \"total_sec\": " << result.total_sec << ",\n";
    out << "  \"final_cost\": " << result.final_cost << ",\n";
    out << "  \"final_reprojection_rmse_px\": "
        << result.final_rmse_px << ",\n";
    out << "  \"final_are_px\": " << result.final_are_px << ",\n";
    out << "  \"retained_camera_pairs\": "
        << result.retained_camera_pairs << ",\n";
    out << "  \"retained_pair_references\": "
        << result.retained_pair_references << ",\n";
    out << "  \"retained_observations\": "
        << result.retained_observations << ",\n";
    out << "  \"cached_cross_blocks\": "
        << result.cached_cross_blocks << ",\n";
    out << "  \"cross_cache_bytes\": "
        << static_cast<unsigned long long>(result.cached_cross_blocks) * sizeof(Mat93) << ",\n";
    out << "  \"strong_camera_pairs\": "
        << result.strong_camera_pairs << ",\n";
    out << "  \"strong_active_cameras\": "
        << result.strong_active_cameras << ",\n";
    out << "  \"pair_graph_active_cameras\": "
        << result.pair_graph_active_cameras << ",\n";
    out << "  \"pair_graph_components\": "
        << result.pair_graph_components << ",\n";
    out << "  \"pair_graph_largest_component\": "
        << result.pair_graph_largest_component << ",\n";
    out << "  \"pair_spanning_backbone_used\": "
        << (result.pair_spanning_backbone_used ? "true" : "false")
        << ",\n";
    out << "  \"grouping_cut_ratio\": "
        << result.grouping_cut_ratio << ",\n";
    out << "  \"costs\": ";
    writeDoubleArray(out, result.costs);
    out << ",\n  \"outer_sec\": ";
    writeDoubleArray(out, result.outer_sec);
    out << ",\n  \"linearize_sec\": ";
    writeDoubleArray(out, result.linearize_sec);
    out << ",\n  \"message_build_sec\": ";
    writeDoubleArray(out, result.message_build_sec);
    out << ",\n  \"build_setup_sec\": ";
    writeDoubleArray(out, result.build_setup_sec);
    out << ",\n  \"build_point_sec\": ";
    writeDoubleArray(out, result.build_point_sec);
    out << ",\n  \"build_rhs_reduce_sec\": ";
    writeDoubleArray(out, result.build_rhs_reduce_sec);
    out << ",\n  \"build_unary_reduce_sec\": ";
    writeDoubleArray(out, result.build_unary_reduce_sec);
    out << ",\n  \"build_pair_sec\": ";
    writeDoubleArray(out, result.build_pair_sec);
    out << ",\n  \"solve_sec\": ";
    writeDoubleArray(out, result.solve_sec);
    out << ",\n  \"backup_sec\": ";
    writeDoubleArray(out, result.backup_sec);
    out << ",\n  \"back_substitute_sec\": ";
    writeDoubleArray(out, result.back_substitute_sec);
    out << ",\n  \"metric_sec\": ";
    writeDoubleArray(out, result.metric_sec);
    out << ",\n  \"gbp_full_sec\": ";
    writeDoubleArray(out, result.gbp_full_sec);
    out << ",\n  \"gbp_fixed_map_sec\": ";
    writeDoubleArray(out, result.gbp_fixed_map_sec);
    out << ",\n  \"gbp_eta_sec\": ";
    writeDoubleArray(out, result.gbp_eta_sec);
    out << ",\n  \"gbp_mean_sec\": ";
    writeDoubleArray(out, result.gbp_mean_sec);
    out << ",\n  \"gbp_coarse_residual_sec\": ";
    writeDoubleArray(out, result.gbp_coarse_residual_sec);
    out << ",\n  \"gbp_exact_residual_sec\": ";
    writeDoubleArray(out, result.gbp_exact_residual_sec);
    out << ",\n  \"gbp_coarse_solve_sec\": ";
    writeDoubleArray(out, result.gbp_coarse_solve_sec);
    out << ",\n  \"accepted_alpha\": ";
    writeDoubleArray(out, result.accepted_alpha);
    out << ",\n  \"trial_evaluations\": ";
    writeIntArray(out, result.trial_evaluations);
    out << ",\n  \"linear_cycles\": ";
    writeIntArray(out, result.linear_cycles);
    out << ",\n  \"accepted_search_directions\": ";
    writeIntArray(out, result.accepted_search_directions);
    out << ",\n  \"true_schur_products\": ";
    writeIntArray(out, result.true_schur_products);
    out << ",\n  \"variance_defect\": ";
    writeDoubleArray(out, result.variance_defect);
    out << ",\n  \"lambda\": ";
    writeDoubleArray(out, result.lambda);
    out << ",\n  \"step_quality\": ";
    writeDoubleArray(out, result.step_quality);
    out << ",\n  \"linear_residual_norm\": ";
    writeDoubleArray(out, result.linear_residual);
    out << ",\n  \"groups\": ";
    writeIntArray(out, result.groups);
    out << ",\n  \"full_sweeps\": ";
    writeIntArray(out, result.full_sweeps);
    out << ",\n  \"eta_sweeps\": ";
    writeIntArray(out, result.eta_sweeps);
    out << "\n}\n";
}
