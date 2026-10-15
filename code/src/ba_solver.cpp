#include "detail/ba_internal.h"
#include <queue>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba/bal/bal_bundle_adjustment_helper.hpp"

namespace {

#include "detail/ba_topology.inc"

#include "detail/ba_landmarks.inc"

#include "detail/ba_build.inc"

#include "detail/ba_solve.inc"

RootHGBPResult runRootHGBP(
    BALProblem& input_problem,
    const Args& args,
    const HGBPConfig& lab
) {
    RootHGBPResult result;
    const auto total_t0 = Clock::now();
    RootProblem problem = makeRootProblem(input_problem);
    const int threads = std::max(1, requestedGBPThreads(args));
    tbb::global_control thread_limit(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(threads));

    const double pose_scaling_epsilon =
        Sophus::Constants<double>::epsilonSqrt();

    const auto preprocessor_t0 = Clock::now();
    const auto preprocessor_linearization_t1 = Clock::now();
    const RootHGBPTopology topology =
        buildRootHGBPTopology(
            input_problem,
            problem,
            lab.graph_neighbors,
            threads);
    const PackedBlockSchurPattern& pattern = topology.pattern;
    const RootHGBPBuildPlan& build_plan = topology.build_plan;
    const RootCrossCachePlan cross_cache =
        makeRootCrossCachePlan(pattern, build_plan, lab.pair_sample_cap);
    result.cached_cross_blocks = cross_cache.slot_count;
    result.retained_camera_pairs =
        (pattern.row_ptr.back() - pattern.free_cameras) / 2;
    result.retained_pair_references =
        pattern.slot_pair_row_ptr.empty()
            ? 0
            : pattern.slot_pair_row_ptr.back();
    result.retained_observations =
        build_plan.retained_edge_count;
    result.strong_camera_pairs = topology.strong_camera_pairs;
    result.strong_active_cameras = topology.strong_active_cameras;
    result.pair_graph_active_cameras =
        topology.selected_active_cameras;
    result.pair_graph_components = topology.selected_components;
    result.pair_graph_largest_component =
        topology.selected_largest_component;
    result.pair_spanning_backbone_used =
        topology.spanning_backbone_used;
    PackedRootLandmarks packed_landmarks =
        makePackedRootLandmarks(problem, build_plan);
    refreshPackedRootCameraCache(
        problem, packed_landmarks);
    FastHGBPSystem fast_system;
    FastHGBPCoarseWorkspace coarse_workspace;
    const auto preprocessor_t1 = Clock::now();
    result.preprocessor_sec =
        elapsed(preprocessor_t0, preprocessor_t1);
    result.preprocessor_linearization_sec =
        elapsed(preprocessor_t0, preprocessor_linearization_t1);
    result.preprocessor_topology_sec =
        elapsed(preprocessor_linearization_t1, preprocessor_t1);

    RootMetrics metrics =
        computePackedRootMetrics(
            problem, packed_landmarks, false);
    if (result.costs.empty()) {
        result.costs.push_back(metrics.cost2);
    }
    double lambda = lab.initial_lambda;
    double lambda_vee = 2.0;
    bool need_linearization = true;
    Eigen::VectorXd pose_scaling;
    Eigen::VectorXd root_b;
    std::vector<std::vector<int>> fixed_groups;
    double fixed_cut_ratio = 0.0;
    Vec3List point_increment;

    for (int outer = 1; outer <= args.outer; ++outer) {
        const auto outer_t0 = Clock::now();
        double linearize_time = 0.0;
        const bool fuse_linearize_build = need_linearization;
        const auto build_t0 = Clock::now();
        RootBuildBreakdown build_breakdown;
        buildPackedHGBPSystemInto(
            fast_system,
            pattern,
            build_plan,
            problem,
            packed_landmarks,
            root_b,
            lambda,
            args.build_threads,
            lab.pair_sample_cap,
            true,
            lab.pair_factor_scale,
            fuse_linearize_build,
            pose_scaling_epsilon,
            pose_scaling,
            build_breakdown,
            cross_cache);
        normalizeBAGBPCanonicalPairs(fast_system, args.build_threads);
        refreshPSDPairPreconditioner(
            fast_system, args.build_threads);
        const auto build_t1 = Clock::now();

        int group_count = 0;
        refreshPackedRootCameraCache(problem, packed_landmarks);
        PackedGBPStats stats;
        const auto solve_t0 = Clock::now();
        if (fixed_groups.empty()) {
            const int target = std::max(2, (fast_system.free_cameras + lab.coarse_groups - 1) / lab.coarse_groups);
            fixed_groups = buildConnectedBAGroups(fast_system, target);
            result.disconnected_groups = baDisconnectedGroups(fast_system, fixed_groups);
            fixed_cut_ratio =
                schurCutRatio(fast_system, fixed_groups);
        }
        group_count = static_cast<int>(fixed_groups.size());
        Eigen::VectorXd increment =
            solveFastHGBPWithFixedGroups(
                fast_system,
                packed_landmarks,
                pose_scaling,
                lambda,
                args,
                fixed_groups,
                coarse_workspace,
                lab.linear_rel_tol,
                lab.variance_rel_tol,
                stats);
        const double linear_residual =
            stats.relative_linear_residual;
        const auto solve_t1 = Clock::now();

        const auto backup_t0 = Clock::now();
        problem.backup();
        const auto backup_t1 = Clock::now();
        const auto back_substitute_t0 = Clock::now();
        Eigen::VectorXd physical_increment =
            increment.cwiseProduct(pose_scaling);
        const BAStepModel step_model = backSubstitutePackedRoot(
            problem,
            packed_landmarks,
            physical_increment,
            lambda,
            point_increment);
        applyRootCameraIncrementDirect(
            problem, physical_increment);
        packed_landmarks.camera_cache_valid = false;
        const auto back_substitute_t1 = Clock::now();

        const auto metric_t0 = Clock::now();
        refreshPackedRootCameraCache(
            problem, packed_landmarks);
        RootMetrics trial_metrics =
            computePackedRootMetrics(
                problem, packed_landmarks, false);
        double alpha = 1.0;
        double quality = 0.0;
        int trial_evaluations = 1;
        bool accepted = false;
        for (int trial = 0; trial <= 8; ++trial) {
            const double predicted_decrease = step_model.decrease(alpha);
            const double actual_decrease =
                0.5 * (metrics.cost2 - trial_metrics.cost2);
            quality = actual_decrease / predicted_decrease;
            accepted = std::isfinite(quality) &&
                step_model.linear_decrease > 0.0 &&
                predicted_decrease > 0.0 &&
                actual_decrease > 0.0 &&
                actual_decrease >= 1e-4 * alpha * step_model.linear_decrease;
            if (accepted || trial == 8 ||
                !std::isfinite(step_model.linear_decrease) ||
                step_model.linear_decrease <= 0.0) {
                break;
            }
            // Backtrack the same full BA direction, not a newly eliminated point step.
            alpha *= 0.5;
            problem.restore();
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, point_increment.size(), 128),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t p = range.begin(); p != range.end(); ++p) {
                        problem.landmarks()[p].p_w += alpha * point_increment[p];
                    }
                });
            applyRootCameraIncrementDirect(problem, alpha * physical_increment);
            packed_landmarks.camera_cache_valid = false;
            refreshPackedRootCameraCache(problem, packed_landmarks);
            trial_metrics = computePackedRootMetrics(problem, packed_landmarks, false);
            ++trial_evaluations;
        }
        const auto metric_t1 = Clock::now();

        if (accepted) {
            metrics = trial_metrics;
            double damping_multiplier = std::max(
                1.0 / 3.0,
                1.0 - std::pow(2.0 * quality - 1.0, 3.0));
            // A backtracked step is evidence that the full-step trust region was too large.
            // Good agreement only at a tiny alpha must not decrease damping again.
            if (alpha < 1.0) {
                damping_multiplier = std::max(damping_multiplier, 1.0 / alpha);
            }
            lambda *= damping_multiplier;
            lambda = std::max(1e-16, lambda);
            lambda_vee = 2.0;
            need_linearization = true;
        } else {
            problem.restore();
            packed_landmarks.camera_cache_valid = false;
            lambda *= lambda_vee;
            lambda_vee *= 2.0;
            need_linearization = false;
        }

        const auto outer_t1 = Clock::now();
        result.costs.push_back(metrics.cost2);
        result.outer_sec.push_back(
            elapsed(outer_t0, outer_t1));
        result.linearize_sec.push_back(linearize_time);
        result.message_build_sec.push_back(
            elapsed(build_t0, build_t1));
        result.build_setup_sec.push_back(
            build_breakdown.setup_sec);
        result.build_point_sec.push_back(
            build_breakdown.point_sec);
        result.build_rhs_reduce_sec.push_back(
            build_breakdown.rhs_reduce_sec);
        result.build_unary_reduce_sec.push_back(
            build_breakdown.unary_reduce_sec);
        result.build_pair_sec.push_back(
            build_breakdown.pair_sec);
        result.solve_sec.push_back(
            elapsed(solve_t0, solve_t1));
        result.backup_sec.push_back(
            elapsed(backup_t0, backup_t1));
        result.back_substitute_sec.push_back(
            elapsed(back_substitute_t0, back_substitute_t1));
        result.metric_sec.push_back(
            elapsed(metric_t0, metric_t1));
        result.gbp_full_sec.push_back(stats.full_sec);
        result.gbp_fixed_map_sec.push_back(
            stats.fixed_map_sec);
        result.gbp_eta_sec.push_back(stats.eta_sec);
        result.gbp_mean_sec.push_back(stats.mean_sec);
        result.gbp_coarse_residual_sec.push_back(
            stats.coarse_residual_sec);
        result.gbp_exact_residual_sec.push_back(
            stats.exact_residual_sec);
        result.gbp_coarse_solve_sec.push_back(
            stats.coarse_solve_sec);
        result.accepted_alpha.push_back(
            accepted ? alpha : 0.0);
        result.trial_evaluations.push_back(trial_evaluations);
        result.linear_cycles.push_back(stats.cycles);
        result.accepted_search_directions.push_back(stats.accepted_search_directions);
        result.true_schur_products.push_back(stats.true_schur_products);
        result.variance_defect.push_back(stats.variance_defect);
        result.lambda.push_back(lambda);
        result.step_quality.push_back(quality);
        result.linear_residual.push_back(linear_residual);
        result.groups.push_back(group_count);
        result.full_sweeps.push_back(stats.full_sweeps);
        result.eta_sweeps.push_back(stats.eta_sweeps);

        std::cout << std::setprecision(17)
                  << "[H-GBP] outer=" << outer
                  << " cost=" << metrics.cost2
                  << " accepted=" << (accepted ? 1 : 0)
                  << " quality=" << quality
                  << " lambda=" << lambda
                  << " lin_rel=" << linear_residual
                  << " linearize=" << linearize_time
                  << " message_build="
                  << result.message_build_sec.back()
                  << " solve=" << result.solve_sec.back()
                  << " backup=" << result.backup_sec.back()
                  << " backsub=" << result.back_substitute_sec.back()
                  << " metric=" << result.metric_sec.back()
                  << " groups=" << group_count
                  << " cut=" << fixed_cut_ratio
                  << " full=" << stats.full_sweeps
                  << " eta=" << stats.eta_sweeps
                  << "\n";
    }

    refreshPackedRootCameraCache(
        problem, packed_landmarks);
    const RootMetrics final_metrics =
        computePackedRootMetrics(
            problem, packed_landmarks, true);
    metrics.rmse_px = final_metrics.rmse_px;
    metrics.are_px = final_metrics.are_px;
    result.grouping_cut_ratio = fixed_cut_ratio;
    result.total_sec = elapsed(total_t0, Clock::now());
    result.final_cost = metrics.cost2;
    result.final_rmse_px = metrics.rmse_px;
    result.final_are_px = metrics.are_px;
    return result;
}

#include "detail/ba_output.inc"

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> filtered_storage;
        std::vector<char*> filtered_argv;
        const HGBPConfig lab =
            parseHGBPConfig(argc, argv, filtered_storage, filtered_argv);
        Args args = parseArgs(
            static_cast<int>(filtered_argv.size()), filtered_argv.data());
        configureOpenMPWaitPolicy(args);
        BALProblem problem = readBAL(args.problem_file);
        std::cout << "[BAL H-GBP] cameras=" << problem.num_cameras
                  << " points=" << problem.num_points
                  << " observations=" << problem.num_observations << "\n";
        if (args.normalize_bal) {
            normalizeBALProblem(problem);
        }
        const RootHGBPResult result = runRootHGBP(problem, args, lab);
        writeRootHGBPJson(args.out_json, problem, args, lab, result);
        std::cout << std::setprecision(17)
                  << "[H-GBP summary] total=" << result.total_sec
                  << " preprocessor=" << result.preprocessor_sec
                  << " cost=" << result.final_cost
                  << " rmse=" << result.final_rmse_px
                  << " are=" << result.final_are_px
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
