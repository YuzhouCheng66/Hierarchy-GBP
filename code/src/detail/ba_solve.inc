struct RootHGBPResult {
    double total_sec = 0.0;
    double preprocessor_sec = 0.0;
    double preprocessor_linearization_sec = 0.0;
    double preprocessor_topology_sec = 0.0;
    double final_cost = 0.0;
    double final_rmse_px = 0.0;
    double final_are_px = 0.0;
    int retained_camera_pairs = 0;
    long long retained_pair_references = 0;
    int retained_observations = 0;
    int cached_cross_blocks = 0;
    int strong_camera_pairs = 0;
    int strong_active_cameras = 0;
    int pair_graph_active_cameras = 0;
    int pair_graph_components = 0;
    int pair_graph_largest_component = 0;
    bool pair_spanning_backbone_used = false;
    double grouping_cut_ratio = 0.0;
    int disconnected_groups = 0;
    std::vector<double> costs;
    std::vector<double> outer_sec;
    std::vector<double> linearize_sec;
    std::vector<double> message_build_sec;
    std::vector<double> build_setup_sec;
    std::vector<double> build_point_sec;
    std::vector<double> build_rhs_reduce_sec;
    std::vector<double> build_unary_reduce_sec;
    std::vector<double> build_pair_sec;
    std::vector<double> solve_sec;
    std::vector<double> backup_sec;
    std::vector<double> back_substitute_sec;
    std::vector<double> metric_sec;
    std::vector<double> gbp_full_sec;
    std::vector<double> gbp_fixed_map_sec;
    std::vector<double> gbp_eta_sec;
    std::vector<double> gbp_mean_sec;
    std::vector<double> gbp_coarse_residual_sec;
    std::vector<double> gbp_exact_residual_sec;
    std::vector<double> gbp_coarse_solve_sec;
    std::vector<double> accepted_alpha;
    std::vector<double> lambda;
    std::vector<double> step_quality;
    std::vector<double> linear_residual;
    std::vector<int> groups;
    std::vector<int> full_sweeps;
    std::vector<int> eta_sweeps;
    std::vector<int> trial_evaluations;
    std::vector<int> linear_cycles;
    std::vector<int> accepted_search_directions, true_schur_products;
    std::vector<double> variance_defect;
};

void schurGBPFirstZeroMessageSweep(
    SchurGBPWorkspace& ws,
    double damping,
    int threads
) {
    const double omega =
        std::min(1.0, std::max(0.0, damping));
    const int edge_count =
        static_cast<int>(ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(
        threads, edge_count, ws.n);

#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads) if(parallel_threads > 1)
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int eidx = 0; eidx < edge_count; ++eidx) {
            const SchurGBPEdge& edge =
                ws.edges[static_cast<size_t>(eidx)];
            {
                Mat9 solved_cross = Mat9::Zero();
                Vec9 solved_eta = Vec9::Zero();
                Mat9 denominator = ws.unary_lam[
                    static_cast<size_t>(edge.j)];
                denominator.noalias() += edge.factor_lam_j;
                solveRegularizedSpd9(
                    denominator,
                    edge.aji,
                    ws.unary_eta[static_cast<size_t>(edge.j)],
                    solved_cross,
                    solved_eta);
                Mat9 lam = Mat9::Zero();
                lam.noalias() = -(edge.aij * solved_cross);
                lam.noalias() += edge.factor_lam_i;
                lam = 0.5 * (lam + lam.transpose());
                Vec9 eta = Vec9::Zero();
                mat9VecRawInto(
                    edge.aij,
                    solved_eta,
                    eta);
                ws.next_msg_lam[
                    static_cast<size_t>(edge.msg_to_i)] =
                    omega * lam;
                ws.next_msg_eta[
                    static_cast<size_t>(edge.msg_to_i)] =
                    -omega * eta;
            }
            {
                Mat9 solved_cross = Mat9::Zero();
                Vec9 solved_eta = Vec9::Zero();
                Mat9 denominator = ws.unary_lam[
                    static_cast<size_t>(edge.i)];
                denominator.noalias() += edge.factor_lam_i;
                solveRegularizedSpd9(
                    denominator,
                    edge.aij,
                    ws.unary_eta[static_cast<size_t>(edge.i)],
                    solved_cross,
                    solved_eta);
                Mat9 lam = Mat9::Zero();
                lam.noalias() = -(edge.aji * solved_cross);
                lam.noalias() += edge.factor_lam_j;
                lam = 0.5 * (lam + lam.transpose());
                Vec9 eta = Vec9::Zero();
                mat9VecRawInto(
                    edge.aji,
                    solved_eta,
                    eta);
                ws.next_msg_lam[
                    static_cast<size_t>(edge.msg_to_j)] =
                    omega * lam;
                ws.next_msg_eta[
                    static_cast<size_t>(edge.msg_to_j)] =
                    -omega * eta;
            }
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < ws.n; ++i) {
            Mat9 lam =
                ws.unary_lam[static_cast<size_t>(i)];
            Vec9 eta =
                ws.unary_eta[static_cast<size_t>(i)];
            const int begin =
                ws.incoming_offsets[static_cast<size_t>(i)];
            const int end =
                ws.incoming_offsets[
                    static_cast<size_t>(i + 1)];
            for (int pos = begin; pos < end; ++pos) {
                const int msg =
                    ws.incoming_msg_ids[
                        static_cast<size_t>(pos)];
                lam.noalias() += ws.next_msg_lam[
                    static_cast<size_t>(msg)];
                eta.noalias() += ws.next_msg_eta[
                    static_cast<size_t>(msg)];
            }
            ws.next_belief_lam[static_cast<size_t>(i)] =
                lam;
            ws.next_belief_eta[static_cast<size_t>(i)] =
                eta;
        }
    }

    ws.msg_lam.swap(ws.next_msg_lam);
    ws.msg_eta.swap(ws.next_msg_eta);
    ws.belief_lam.swap(ws.next_belief_lam);
    ws.belief_eta.swap(ws.next_belief_eta);
    ws.belief_lam_inv_valid = false;
}

Eigen::VectorXd solveFastHGBPWithFixedGroups(
    FastHGBPSystem& sys,
    PackedRootLandmarks& exact_landmarks,
    const Eigen::VectorXd& pose_scaling,
    double exact_damping,
    const Args& args,
    const std::vector<std::vector<int>>& groups,
    FastHGBPCoarseWorkspace& coarse_workspace,
    double linear_rel_tol,
    double variance_rel_tol,
    PackedGBPStats& stats
) {
    stats = PackedGBPStats{};
    SchurGBPWorkspace& ws = sys.ws;
    stats.edges = static_cast<int>(ws.edges.size());
    const int gbp_threads = effectiveGBPThreads(
        requestedGBPThreads(args), stats.edges, ws.n);
    const int eta_threads = effectiveGBPThreads(
        requestedGBPThreads(args), stats.edges, ws.n);
    stats.threads = gbp_threads;
    stats.reused_workspace = true;

    Eigen::VectorXd x =
        Eigen::VectorXd::Zero(9 * sys.free_cameras);
    Eigen::VectorXd rhs;
    fastHGBPRhsVectorInto(sys, rhs);
    Eigen::VectorXd residual = rhs;
    const double target_residual2 = linear_rel_tol * linear_rel_tol * rhs.squaredNorm();
    if (!groups.empty()) {
        const auto coarse_t0 = Clock::now();
        buildFastHGBPCoarseWorkspace(coarse_workspace, sys, groups);
        stats.coarse_solve_sec += elapsed(coarse_t0, Clock::now());
    }

    for (int sweep = 0; sweep < args.gbp_full_sweeps; ++sweep) {
        const auto sweep_t0 = Clock::now();
        if (sweep == 0) {
            schurGBPFirstZeroMessageSweep(
                ws, args.message_damping, gbp_threads);
        } else {
            schurGBPSweep(
                ws, args.message_damping, gbp_threads);
        }
        stats.full_sec += elapsed(sweep_t0, Clock::now());
        ++stats.full_sweeps;
        if (variance_rel_tol > 0.0) {
            double change2 = 0.0, norm2 = 0.0;
            for (size_t slot = 0; slot < ws.msg_lam.size(); ++slot) {
                change2 += (ws.msg_lam[slot] - ws.next_msg_lam[slot]).squaredNorm();
                norm2 += ws.msg_lam[slot].squaredNorm();
            }
            stats.variance_defect = std::sqrt(change2 / std::max(1e-300, norm2));
            if (stats.variance_defect <= variance_rel_tol) break;
        }
    }
    const auto map_t0 = Clock::now();
    buildSchurGBPFixedEtaMaps(ws, gbp_threads);
    buildSchurGBPBeliefLambdaInverses(ws, gbp_threads);
    stats.fixed_map_sec += elapsed(map_t0, Clock::now());

    Eigen::VectorXd cycle_delta;
    Eigen::VectorXd cycle_matvec;
    Eigen::VectorXd cycle_residual;
    Eigen::VectorXd coarse_delta;
    Eigen::VectorXd coarse_matvec;
    Eigen::VectorXd exact_cycle_matvec;
    std::vector<Eigen::VectorXd> fcg_directions;
    std::vector<Eigen::VectorXd> fcg_products;
    std::vector<double> fcg_curvature;
    fcg_directions.reserve(static_cast<size_t>(args.mg_cycles));
    fcg_products.reserve(static_cast<size_t>(args.mg_cycles));
    fcg_curvature.reserve(static_cast<size_t>(args.mg_cycles));
    for (int cycle = 0; cycle < args.mg_cycles; ++cycle) {
        resetSchurGBPEtaMessages(ws, residual, gbp_threads);
        const auto eta_t0 = Clock::now();
        schurGBPFixedEtaSweeps(
            ws, args.message_damping, args.pre_sweeps, eta_threads);
        stats.eta_sec += elapsed(eta_t0, Clock::now());
        stats.eta_sweeps += args.pre_sweeps;

        const auto mean_t0 = Clock::now();
        schurGBPMeanInto(ws, gbp_threads, cycle_delta);
        stats.mean_sec += elapsed(mean_t0, Clock::now());

        // Both additive corrections use the same residual. Keep the accepted
        // arithmetic and timer boundaries, including the zero model products.
        const auto residual_t0 = Clock::now();
        cycle_matvec.setZero(residual.size());
        cycle_residual = residual;
        cycle_residual.noalias() -= cycle_matvec;
        stats.coarse_residual_sec += elapsed(residual_t0, Clock::now());

        const auto coarse_t0 = Clock::now();
        solveFastHGBPCoarseCorrectionInto(coarse_workspace, cycle_residual, coarse_delta);
        stats.coarse_solve_sec += elapsed(coarse_t0, Clock::now());
        cycle_delta.noalias() += coarse_delta;

        const auto coarse_mv_t0 = Clock::now();
        coarse_matvec.setZero(residual.size());
        stats.coarse_residual_sec += elapsed(coarse_mv_t0, Clock::now());
        cycle_matvec.noalias() += coarse_matvec;

        const auto exact_t0 = Clock::now();
        packedRootExactSchurMultiplyInto(
            exact_landmarks, pose_scaling, exact_damping,
            cycle_delta, requestedGBPThreads(args), exact_cycle_matvec);
        ++stats.true_schur_products;
        const Eigen::VectorXd raw_direction = cycle_delta;
        const Eigen::VectorXd raw_product = exact_cycle_matvec;
        for (size_t previous = 0; previous < fcg_directions.size(); ++previous) {
            const double beta = -fcg_directions[previous].dot(exact_cycle_matvec) /
                fcg_curvature[previous];
            cycle_delta.noalias() += beta * fcg_directions[previous];
            exact_cycle_matvec.noalias() += beta * fcg_products[previous];
        }
        double denominator = cycle_delta.dot(exact_cycle_matvec);
        double numerator = residual.dot(cycle_delta);
        if (!(std::isfinite(numerator) &&
              std::isfinite(denominator) &&
              denominator > 0.0)) {
            fcg_directions.clear();
            fcg_products.clear();
            fcg_curvature.clear();
            cycle_delta = raw_direction;
            exact_cycle_matvec = raw_product;
            denominator = cycle_delta.dot(exact_cycle_matvec);
            numerator = residual.dot(cycle_delta);
        }
        const double alpha = std::isfinite(numerator) &&
                                 std::isfinite(denominator) &&
                                 denominator > 0.0
            ? numerator / denominator
            : 0.0;
        if (alpha != 0.0 && std::isfinite(denominator)) {
            fcg_directions.push_back(cycle_delta);
            fcg_products.push_back(exact_cycle_matvec);
            fcg_curvature.push_back(denominator);
        }
        residual.noalias() -= alpha * exact_cycle_matvec;
        if (alpha != 0.0) ++stats.accepted_search_directions;
        stats.exact_residual_sec += elapsed(exact_t0, Clock::now());
        x.noalias() += alpha * cycle_delta;
        ++stats.cycles;
        if (linear_rel_tol > 0.0 &&
            residual.squaredNorm() <= target_residual2) {
            break;
        }
    }
    stats.relative_linear_residual =
        residual.norm() / std::max(1e-300, rhs.norm());
    stats.model_decrease = 0.5 * x.dot(rhs + residual);
    return x;
}

BAStepModel backSubstitutePackedRoot(
    RootProblem& problem,
    const PackedRootLandmarks& packed,
    const Eigen::VectorXd& pose_increment,
    double landmark_damping,
    Vec3List& point_increment
) {
    point_increment.resize(packed.Hll.size());
    auto body = [&](const tbb::blocked_range<size_t>& range,
                    BAStepModel model) {
        for (size_t p = range.begin(); p != range.end(); ++p) {
            Vec3 tmp = Vec3::Zero();
            double pose_model = 0.0;
            const int begin = packed.point_offset[p];
            const int end = packed.point_offset[p + 1];
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera_id = packed.camera_id[e];
                const Vec2 projected_increment =
                    packed.Jc[e] *
                    pose_increment.segment<9>(9 * camera_id);
                tmp.noalias() +=
                    packed.Jl[e].transpose() *
                    (packed.residual[e] + projected_increment);
                pose_model += projected_increment.dot(
                    0.5 * projected_increment +
                    packed.residual[e]);
                model.linear_decrease -=
                    projected_increment.dot(packed.residual[e]);
            }

            Vec3 landmark_increment = -solvePoint3(packed.Hll_factor[p], tmp);
            const double landmark_model =
                0.5 * landmark_increment.dot(tmp) -
                0.5 * landmark_damping *
                    landmark_increment.squaredNorm();
            model.full_decrease -=
                pose_model + landmark_model;
            model.linear_decrease -= landmark_increment.dot(packed.bl[p]);
            landmark_increment.array() *=
                packed.Jl_scale[p].array();
            point_increment[p] = landmark_increment;
            problem.landmarks()[p].p_w += landmark_increment;
        }
        return model;
    };
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(
            0, packed.Hll.size(), 128),
        BAStepModel{},
        body,
        [](BAStepModel a, const BAStepModel& b) {
            a.full_decrease += b.full_decrease;
            a.linear_decrease += b.linear_decrease;
            return a;
        });
}

void applyRootCameraIncrementDirect(
    RootProblem& problem,
    const Eigen::VectorXd& increment
) {
    for (int c = 0; c < problem.num_cameras(); ++c) {
        auto& camera = problem.cameras()[static_cast<size_t>(c)];
        camera.apply_inc_pose(increment.segment<6>(9 * c));
        camera.apply_inc_intrinsics(increment.segment<3>(9 * c + 6));
    }
}
