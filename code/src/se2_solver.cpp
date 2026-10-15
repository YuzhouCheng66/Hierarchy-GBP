#include "internal/se2_solver_impl.h"
#ifdef HGBP_PURE_GBP
#include "pure_gbp/projection.h"
#endif
#include "internal/scoped_worker_affinity.h"
#include "internal/se2_sync.h"
#include "internal/se2_group_sync.h"
#include "internal/se2_residual.h"
#include "internal/se2_basis.h"
#include "internal/se2_coarse.h"
#include "internal/se2_dense_sync.h"
#include "internal/se2_basis_data.h"
#include "internal/partial_symmetric_eigen.h"
#include "internal/cycle_energy.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Eigenvalues>
#include <Eigen/SparseCholesky>
#include <omp.h>

#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace slam {

namespace {

#include "detail/se2_model.inc"

#include "detail/se2_basis.inc"

#include "detail/se2_coarse.inc"

std::string jsonNumber(double value) {
    if (!std::isfinite(value)) {
        return "null";
    }
    std::ostringstream oss;
    oss << std::setprecision(17) << value;
    return oss.str();
}

void writePoseVectorJson(std::ofstream& out, const std::vector<Eigen::Vector3d>& poses, int indent_spaces) {
    const std::string indent(indent_spaces, ' ');
    out << "[\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        const Eigen::Vector3d& p = poses[i];
        out << indent << "  [" << jsonNumber(p(0)) << ", " << jsonNumber(p(1)) << ", " << jsonNumber(p(2)) << "]";
        out << (i + 1 == poses.size() ? "\n" : ",\n");
    }
    out << indent << "]";
}

void writePoseHistoryJsonArray(std::ofstream& out, const std::vector<std::vector<Eigen::Vector3d>>& history, int indent_spaces) {
    const std::string indent(indent_spaces, ' ');
    out << "[\n";
    for (size_t i = 0; i < history.size(); ++i) {
        out << indent << "  ";
        writePoseVectorJson(out, history[i], indent_spaces + 2);
        out << (i + 1 == history.size() ? "\n" : ",\n");
    }
    out << indent << "]";
}

}  // namespace

#include "detail/se2_problem.inc"

#include "detail/se2_linearization.inc"

ExperimentResults runSyntheticSE2Experiment(
    const SyntheticSE2Problem& problem,
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
    double coarse_scale,
    const BasisBuildConfig& basis_build_config,
    const RobustLossConfig& robust_loss_config,
    int final_coarse_polish_passes
) {
    ExperimentResults results;
    const auto solver_wall_start = SteadyClock::now();
    std::ofstream linear_audit;
    if (const char* path = std::getenv("HGBP_LINEAR_AUDIT"); path && *path) {
        linear_audit.open(path);
        if (!linear_audit) throw std::runtime_error("Cannot open SE2 linear audit");
        linear_audit << std::setprecision(17);
    }
    const int upper_polish_passes = std::max(0, final_coarse_polish_passes);
    results.num_poses = static_cast<int>(problem.gt_poses.size());
    results.num_edges = static_cast<int>(problem.edges.size());
    results.initial_objective = nonlinearObjective(problem, problem.init_poses, 1);
    results.initial_poses = problem.init_poses;

    SyntheticResidualWorkspaceBuildStats mg_build_stats;
    SyntheticResidualGraphWorkspace mg_workspace =
        buildSyntheticResidualGraphWorkspace(problem, 1e-12, &mg_build_stats);
    mg_workspace.graph.setSyncNumThreads(sync_num_threads);
    mg_workspace.graph.setSyncUpdateMeans(false);
    mg_workspace.graph.setSyncSchedule(sync_schedule_kind, sync_schedule_chunk);
    mg_workspace.graph.setProfileSyncThreadUtilization(profile_sync_thread_utilization);
    results.residual_workspace_build_sec = mg_build_stats.total_sec;
    results.residual_workspace_variable_alloc_sec = mg_build_stats.variable_alloc_sec;
    results.residual_workspace_factor_alloc_sec = mg_build_stats.factor_alloc_sec;
    results.residual_workspace_connect_sec = mg_build_stats.connect_sec;
    SyntheticSE2FastSyncLocalPackedLayout fastsync_local_packed_layout =
        buildSyntheticSE2FastSyncLocalPackedLayout(mg_workspace.graph);
    SyntheticSE2PackedBasisTopology fastsync_local_packed_basis_topology;
    SyntheticSE2PackedCoarseAssemblyWorkspace fastsync_local_packed_coarse_workspace;
    bool fastsync_local_packed_coarse_workspace_valid = false;
    SparseCholeskyFactor fastsync_local_packed_coarse_factor;
    bool fastsync_local_packed_coarse_factor_valid = false;
    const bool fastsync_local_packed_basis_topology_valid = fastsync_local_packed_layout.supported;
    if (fastsync_local_packed_basis_topology_valid) {
        fastsync_local_packed_basis_topology =
            buildMessageConditionedBasisPackedTopology(fastsync_local_packed_layout, group_size);
    }
    SyntheticSE2FastSyncLocalLayout fastsync_local_layout =
        buildSyntheticSE2FastSyncLocalLayout(mg_workspace.graph);
    SyntheticSE2FastSyncSoALayout fastsync_layout =
        buildSyntheticSE2FastSyncSoALayout(mg_workspace.graph);
    SyntheticSE2PackedResidualWorkspace packed_residual_workspace;
    SyntheticSE2PackedBasisTopology packed_residual_basis_topology;
    SyntheticSE2PackedCoarseAssemblyWorkspace packed_residual_coarse_workspace;
    bool packed_residual_coarse_workspace_valid = false;
    SparseCholeskyFactor packed_residual_coarse_factor;
    bool packed_residual_coarse_factor_valid = false;
    const bool packed_residual_topology_valid = basis_build_config.enable_packed_residual_solver;
    if (packed_residual_topology_valid) {
        packed_residual_workspace = buildSyntheticSE2PackedResidualWorkspace(problem, 1e-12);
        packed_residual_basis_topology =
            buildMessageConditionedBasisPackedTopology(packed_residual_workspace, group_size);
        results.packed_schur_adjugate_fallback_count = packed_residual_workspace.schur_adjugate_fallback_count;
        results.packed_schur_general_failure_count = packed_residual_workspace.schur_general_failure_count;
        results.packed_kernel_prep_cycles = packed_residual_workspace.kernel_prep_cycles;
        results.packed_kernel_schur0_cycles = packed_residual_workspace.kernel_schur0_cycles;
        results.packed_kernel_schur1_cycles = packed_residual_workspace.kernel_schur1_cycles;
        results.packed_kernel_update_belief_cycles = packed_residual_workspace.kernel_update_belief_cycles;
        results.packed_fixedeta_full_lambda_sweeps = packed_residual_workspace.fixedeta_full_lambda_sweeps;
        results.packed_fixedeta_lambda_init_sweeps = packed_residual_workspace.fixedeta_lambda_init_sweeps;
        results.packed_fixedeta_eta_only_sweeps = packed_residual_workspace.fixedeta_eta_only_sweeps;
        results.packed_fixedeta_eta_map_builds = packed_residual_workspace.fixedeta_eta_map_builds;
        results.packed_fixedeta_serial_delta_sweeps = packed_residual_workspace.fixedeta_serial_delta_sweeps;
    }
    relinearizeSyntheticResidualGraph(mg_workspace, problem, problem.init_poses, sync_num_threads, nullptr, robust_loss_config);
    const bool force_unsafe_fastsync = getenvEnabled("GBP_FORCE_UNSAFE_FASTSYNC");
    const bool force_packed_residual_solver = getenvEnabled("GBP_FORCE_PACKED_RESIDUAL");
    const bool force_fastsync_local_packed = getenvEnabled("GBP_FORCE_FASTSYNC_LOCAL_PACKED");
    const bool force_fastsync_local = getenvEnabled("GBP_FORCE_FASTSYNC_LOCAL");
    const bool prefer_nonsym_safe_fastsync_global =
        !force_unsafe_fastsync && maxFactorLamAsymmetry(mg_workspace.graph) > 1e-7;

    PoseVec direct_poses = problem.init_poses;
    results.direct_history.push_back(OuterDirectRow{0, nonlinearObjective(problem, direct_poses, 1), 0.0, 0.0});
    results.direct_pose_history.push_back(direct_poses);

    PoseVec mg_poses = problem.init_poses;
    std::vector<Eigen::MatrixXd> previous_basis_local_bases;
    BasisData cached_basis;
    bool cached_basis_valid = false;
    {
        OuterMGRow row0;
        row0.outer = 0;
        row0.execution_path = "initial";
        row0.nonlinear_objective = nonlinearObjective(problem, mg_poses, 1);
        results.mg_history.push_back(std::move(row0));
    }
    results.mg_pose_history.push_back(mg_poses);
    for (int outer = 1; outer <= num_outer; ++outer) {
        const auto outer_t0 = SteadyClock::now();

        gbp::FactorGraph& residual_graph = mg_workspace.graph;
        const bool use_packed_residual_solver =
            (force_packed_residual_solver ||
             (basis_build_config.enable_packed_residual_solver &&
              enable_singlecore_fastsync &&
              residual_graph.eta_damping == 0.0 &&
              packed_residual_topology_valid &&
              !prefer_nonsym_safe_fastsync_global &&
              !profile_sync_thread_utilization));

        const auto build_t0 = SteadyClock::now();
        SyntheticResidualRelinearizeStats relin_stats;
        if (!use_packed_residual_solver) {
            relinearizeSyntheticResidualGraph(mg_workspace, problem, mg_poses, sync_num_threads, &relin_stats, robust_loss_config);
        }
        if (use_packed_residual_solver) {
            const auto packed_relin_t0 = SteadyClock::now();
            relinearizeSyntheticSE2PackedResidualWorkspace(
                packed_residual_workspace,
                problem,
                mg_poses,
                robust_loss_config,
                sync_num_threads
            );
            const auto packed_relin_t1 = SteadyClock::now();
            relin_stats.factor_relinearize_sec = elapsedSeconds(packed_relin_t0, packed_relin_t1);
            relin_stats.reset_state_sec = 0.0;
        }
        const double factor_lam_max_asym = use_packed_residual_solver ? 0.0 : maxFactorLamAsymmetry(residual_graph);
        const bool prefer_nonsym_safe_fastsync =
            prefer_nonsym_safe_fastsync_global ||
            (!use_packed_residual_solver && factor_lam_max_asym > 1e-7);
        const bool use_fastsync_local_packed =
            !use_packed_residual_solver &&
            (force_fastsync_local_packed ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              residual_graph.eta_damping == 0.0 &&
              fastsync_local_packed_layout.supported &&
              !prefer_nonsym_safe_fastsync &&
              !profile_sync_thread_utilization));
        const bool use_fastsync_local =
            !use_packed_residual_solver &&
            !use_fastsync_local_packed &&
            (force_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              residual_graph.eta_damping == 0.0 &&
              fastsync_local_layout.supported &&
              !prefer_nonsym_safe_fastsync &&
              !profile_sync_thread_utilization));
        const bool use_packed_non_sweeps =
            !use_packed_residual_solver &&
            use_fastsync_local_packed &&
            basis_build_config.enable_packed_builder;
        if (use_packed_residual_solver) {
            // Packed residual mode keeps the linearized state in the packed workspace.
        } else if (use_fastsync_local_packed) {
            refreshSyntheticSE2FastSyncLocalPackedLayout(fastsync_local_packed_layout);
        } else if (use_fastsync_local) {
            refreshSyntheticSE2FastSyncLocalLayout(fastsync_local_layout);
        } else if (enable_singlecore_fastsync && sync_num_threads == 1 && fastsync_layout.all_dofs3) {
            refreshSyntheticSE2FastSyncSoALayout(fastsync_layout);
        }
        const bool use_singlecore_fastsync_soa =
            !use_packed_residual_solver &&
            !use_fastsync_local_packed &&
            !use_fastsync_local &&
            enable_singlecore_fastsync &&
            sync_num_threads == 1 &&
            fastsync_layout.all_dofs3 &&
            !profile_sync_thread_utilization;
        const std::string execution_path = outerExecutionPathName(
            use_packed_residual_solver,
            use_fastsync_local_packed,
            use_fastsync_local,
            use_singlecore_fastsync_soa
        );
        residual_graph.setProfileSyncTiming(false);
        residual_graph.resetSyncTiming();
        const auto build_t1 = SteadyClock::now();
        const int packed_basis_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_basis_thread_override > 0)
            ? basis_build_config.packed_basis_thread_override
            : sync_num_threads;
        const int packed_coarse_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_coarse_thread_override > 0)
            ? basis_build_config.packed_coarse_thread_override
            : packed_basis_threads;
        const int packed_sweep_threads =
            (use_packed_residual_solver &&
             basis_build_config.packed_sweep_thread_override > 0)
            ? basis_build_config.packed_sweep_thread_override
            : sync_num_threads;

        const bool basis_within_warmup =
            basis_build_config.basis_rebuild_warmup_outers > 0 &&
            outer <= basis_build_config.basis_rebuild_warmup_outers;
        const int post_warmup_outer = outer - basis_build_config.basis_rebuild_warmup_outers;
        const bool rebuild_basis =
            !cached_basis_valid ||
            basis_build_config.basis_rebuild_period <= 1 ||
            basis_within_warmup ||
            (((post_warmup_outer - 1) % basis_build_config.basis_rebuild_period) == 0);
        const auto basis_t0 = SteadyClock::now();
        if (rebuild_basis) {
            const std::vector<Eigen::MatrixXd>* warm_basis_ptr =
                (basis_build_config.enable_warm_start && !previous_basis_local_bases.empty())
                ? &previous_basis_local_bases
                : nullptr;
            cached_basis = use_packed_residual_solver
                ? buildMessageConditionedBasisPacked(
                    packed_residual_workspace,
                    packed_residual_basis_topology,
                    r_reduced,
                    packed_basis_threads != 1,
                    packed_basis_threads,
                    basis_build_config,
                    warm_basis_ptr
                )
                : use_packed_non_sweeps
                ? buildMessageConditionedBasisPacked(
                    fastsync_local_packed_layout,
                    fastsync_local_packed_basis_topology,
                    r_reduced,
                    false,
                    1,
                    basis_build_config,
                    warm_basis_ptr
                )
                : buildMessageConditionedBasis(
                    residual_graph,
                    group_size,
                    r_reduced,
                    sync_num_threads != 1,
                    sync_num_threads,
                    basis_build_config,
                    warm_basis_ptr
                );
            cached_basis_valid = true;
        }
        const auto basis_t1 = SteadyClock::now();
        const BasisData& basis = cached_basis;
        if (basis_build_config.enable_warm_start) {
            if (rebuild_basis) {
                previous_basis_local_bases = basis.local_bases;
            }
        } else {
            previous_basis_local_bases.clear();
        }
        const double basis_build_sec = rebuild_basis ? elapsedSeconds(basis_t0, basis_t1) : 0.0;
        const double basis_block_build_sec = rebuild_basis ? basis.block_build_sec : 0.0;
        const double basis_eigensolver_sec = rebuild_basis ? basis.eigensolver_sec : 0.0;
        const double basis_copyout_sec = rebuild_basis ? basis.copyout_sec : 0.0;
        const int basis_partial_attempted = rebuild_basis ? basis.partial_attempted : 0;
        const int basis_partial_converged = rebuild_basis ? basis.partial_converged : 0;
        const int basis_partial_fallback = rebuild_basis ? basis.partial_fallback : 0;
        const int basis_partial_total_iters = rebuild_basis ? basis.partial_total_iters : 0;

        Eigen::SparseMatrix<double> coarse_lam;
        int coarse_lambda_nnz = 0;
        double coarse_lambda_density = 0.0;
        const auto coarse_lam_t0 = SteadyClock::now();
        if (use_packed_residual_solver) {
            if (!packed_residual_coarse_workspace_valid ||
                packed_residual_coarse_workspace.coarse_dim != basis.coarse_dim) {
                packed_residual_coarse_workspace =
                    buildSyntheticSE2PackedCoarseAssemblyWorkspace(
                        packed_residual_workspace,
                        basis
                    );
                packed_residual_coarse_workspace_valid = true;
                packed_residual_coarse_factor_valid = false;
            }
            coarse_lam = assembleCoarseLambdaDirectPackedOrdered3D(
                packed_residual_workspace,
                basis,
                packed_residual_coarse_workspace,
                packed_coarse_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        } else if (use_packed_non_sweeps) {
            if (!fastsync_local_packed_coarse_workspace_valid ||
                fastsync_local_packed_coarse_workspace.coarse_dim != basis.coarse_dim) {
                fastsync_local_packed_coarse_workspace =
                    buildSyntheticSE2PackedCoarseAssemblyWorkspace(
                        fastsync_local_packed_layout,
                        basis
                    );
                fastsync_local_packed_coarse_workspace_valid = true;
                fastsync_local_packed_coarse_factor_valid = false;
            }
            coarse_lam = assembleCoarseLambdaDirectPackedOrdered3D(
                fastsync_local_packed_layout,
                basis,
                fastsync_local_packed_coarse_workspace,
                sync_num_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        } else {
            coarse_lam = assembleCoarseLambdaDirect(
                residual_graph,
                basis,
                true,
                sync_num_threads
            );
            coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
        }
        const auto coarse_lam_t1 = SteadyClock::now();
        if (basis.coarse_dim > 0) {
            const double denom = static_cast<double>(basis.coarse_dim) * static_cast<double>(basis.coarse_dim);
            coarse_lambda_density = coarse_lambda_nnz / denom;
        }

        double sweeps_sec = 0.0;
        double fastsync_factor_pass_sec = 0.0;
        double fastsync_variable_pass_sec = 0.0;
        double coarse_eta_sec = 0.0;
        double coarse_solve_sec = 0.0;
        double prolong_inject_sec = 0.0;
        const auto coarse_factor_t0 = SteadyClock::now();
        SparseCholeskyFactor generic_coarse_factor;
        SparseCholeskyFactor* coarse_factor = &generic_coarse_factor;
        bool reuse_coarse_pattern = false;
        if (use_packed_residual_solver) {
            coarse_factor = &packed_residual_coarse_factor;
            reuse_coarse_pattern = packed_residual_coarse_factor_valid;
        } else if (use_packed_non_sweeps) {
            coarse_factor = &fastsync_local_packed_coarse_factor;
            reuse_coarse_pattern = fastsync_local_packed_coarse_factor_valid;
        }
#ifdef HGBP_PURE_GBP
        pure_gbp::begin(coarse_lam,1e-10,4,sync_num_threads);
#else
        factorizeSparseCholesky(
            coarse_lam,
            1e-10,
            *coarse_factor,
            reuse_coarse_pattern,
            use_packed_residual_solver || use_packed_non_sweeps
        );
#endif
        if (use_packed_residual_solver) {
            packed_residual_coarse_factor_valid = true;
        } else if (use_packed_non_sweeps) {
            fastsync_local_packed_coarse_factor_valid = true;
        }
        const auto coarse_factor_t1 = SteadyClock::now();
        coarse_solve_sec += elapsedSeconds(coarse_factor_t0, coarse_factor_t1);
        Eigen::VectorXd coarse_eta;
        Eigen::VectorXd delta_z;
        Eigen::VectorXd delta_e;
        Eigen::VectorXd e_now;
        Eigen::VectorXd e_hat;
        const bool cycle_energy_enabled = getenvEnabled("HGBP_CYCLE_ENERGY");
        const bool residual_cycles = getenvEnabled("HGBP_RESIDUAL_CYCLES");
        if (residual_cycles && !cycle_energy_enabled)
            throw std::runtime_error("residual cycle control requires cycle energy");
        if (cycle_energy_enabled && !use_packed_residual_solver)
            throw std::runtime_error("SE2 cycle energy requires the validated packed path");
        CycleEnergyHistory cycle_energy(inner_cycles);
        Eigen::VectorXd cycle_start, cycle_residual, trial_residual;
        Eigen::VectorXd accumulated, accumulated_residual;
        Eigen::VectorXd correction_reference;
        Eigen::SparseMatrix<double> audit_H;
        Eigen::VectorXd audit_b, audit_exact;
        Eigen::MatrixXd audit_P;
        int gbp_sweeps_executed=0,jacobi_sweeps_executed=0,coarse_solves=0;
        if (linear_audit.is_open()) {
            // Shadow reference only. It is never used to update the MG iterate.
            auto audit_graph = buildLinearizedResidualGraph(problem, mg_poses, 1e-12, robust_loss_config);
            auto joint = audit_graph.jointDistributionInfSparse();
            audit_H = joint.lam;
            audit_b = joint.eta;
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> exact;
            exact.compute(audit_H);
            if (exact.info() != Eigen::Success) throw std::runtime_error("SE2 diagnostic factorization failed");
            audit_exact = exact.solve(audit_b);
            audit_P.setZero(audit_b.size(), basis.coarse_dim);
            for (int v = 0; v < packed_residual_workspace.num_vars; ++v) {
                audit_P.block(3*v, basis.var_coarse_offset[v], 3, basis.var_r_local[v]) =
                    Eigen::Map<const Eigen::Matrix<double,3,Eigen::Dynamic>>(
                        varBasis3xRPtr(basis,v), 3, basis.var_r_local[v]);
            }
            const double direct_trial = nonlinearObjective(problem, applyPoseDeltas(mg_poses, audit_exact), 1);
            linear_audit << "{\"outer\":" << outer << ",\"stage\":\"reference\",\"exact_trial_raw_cost\":" << direct_trial
                << ",\"optimal_linear_decrease\":" << 0.5*audit_b.dot(audit_exact)
                << ",\"exact_step_norm\":" << audit_exact.norm()
                << ",\"coarse_operator_error\":" << (Eigen::MatrixXd(coarse_lam)-audit_P.transpose()*audit_H*audit_P).norm()
                << "}\n";
        }
        auto audit_state = [&](const char* stage, int cycle) {
            if (!linear_audit.is_open()) return;
            Eigen::VectorXd value;
            stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(packed_residual_workspace, value, 1);
            if (residual_cycles) value += correction_reference;
            const Eigen::VectorXd residual = audit_b-audit_H*value;
            linear_audit << "{\"outer\":" << outer << ",\"cycle\":" << cycle << ",\"stage\":\"" << stage
                << "\",\"relative_residual\":" << residual.norm()/std::max(1e-300,audit_b.norm())
                << ",\"projected_residual\":" << (audit_P.transpose()*residual).norm()
                << ",\"linear_decrease\":" << 0.5*value.dot(audit_b+residual)
                << ",\"step_error\":" << (value-audit_exact).norm()/std::max(1e-300,audit_exact.norm()) << "}\n";
        };
        if (residual_cycles) {
            accumulated=Eigen::VectorXd::Zero(3*packed_residual_workspace.num_vars);
            correction_reference=accumulated;
            fineResidualSyntheticSE2PackedInto(packed_residual_workspace,
                accumulated,accumulated_residual,packed_sweep_threads);
        }
        auto apply_upper_correction = [&]() {
            if(getenvEnabled("HGBP_SE2_SKIP_COARSE")) return;
            if (use_packed_residual_solver) {
                stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                    packed_residual_workspace,
                    e_now,
                    packed_sweep_threads
                );
            } else if (use_fastsync_local_packed) {
                e_now = stackedMeanVectorFastSyncLocalPacked(fastsync_local_packed_layout);
            } else if (use_fastsync_local) {
                e_now = stackedMeanVectorFastSyncLocal(fastsync_local_layout);
            } else {
                e_now = stackedMeanVector(residual_graph);
            }
            const auto coarse_eta_t0 = SteadyClock::now();
            if (use_packed_residual_solver) {
                assembleCoarseResidualDirectPackedOrdered3DInto(
                    packed_residual_workspace,
                    basis,
                    e_now,
                    coarse_eta,
                    packed_coarse_threads
                );
            } else if (use_packed_non_sweeps) {
                assembleCoarseResidualDirectPackedOrdered3DInto(
                    fastsync_local_packed_layout,
                    basis,
                    e_now,
                    coarse_eta,
                    sync_num_threads
                );
            } else {
                coarse_eta = assembleCoarseResidualDirect(residual_graph, basis, e_now, true, sync_num_threads);
            }
            const auto coarse_eta_t1 = SteadyClock::now();
            coarse_eta_sec += elapsedSeconds(coarse_eta_t0, coarse_eta_t1);

            const auto coarse_solve_t0 = SteadyClock::now();
#ifdef HGBP_PURE_GBP
            if(!use_packed_residual_solver)throw std::runtime_error("pure GBP requires the packed SE2 fine path");
            if(pure_gbp::needsProjectedInitialization())
                pure_gbp::initializeProjected(pure_gbp::projectSE2(packed_residual_workspace,basis,packed_coarse_threads));
            pure_gbp::solve(coarse_eta,delta_z);
#else
            solveWithSparseCholeskyInto(*coarse_factor, coarse_eta, delta_z);
#endif
            ++coarse_solves;
            const auto coarse_solve_t1 = SteadyClock::now();
            coarse_solve_sec += elapsedSeconds(coarse_solve_t0, coarse_solve_t1);

            const auto inject_t0 = SteadyClock::now();
            prolongToFineInto(basis, delta_z, delta_e);
            if (coarse_scale != 1.0) {
                delta_e *= coarse_scale;
            }
            if (use_packed_residual_solver) {
                injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(
                    packed_residual_workspace,
                    delta_e,
                    packed_sweep_threads
                );
            } else if (use_fastsync_local_packed) {
                injectCorrectionKeepMessagesFastSyncLocalPacked(fastsync_local_packed_layout, delta_e);
            } else if (use_fastsync_local) {
                injectCorrectionKeepMessagesFastSyncLocal(fastsync_local_layout, delta_e);
            } else {
                injectCorrectionKeepMessages(residual_graph, delta_e);
            }
            const auto inject_t1 = SteadyClock::now();
            prolong_inject_sec += elapsedSeconds(inject_t0, inject_t1);
        };

        if (use_packed_residual_solver)
            initializeSE2PackedMessagesFromFactors(packed_residual_workspace,
                packed_residual_workspace.message_initialization,packed_sweep_threads);
        for (int cyc = 0; cyc < inner_cycles; ++cyc) {
            audit_state("before_sweeps",cyc);
            const auto sweeps_t0 = SteadyClock::now();
            if (cycle_energy_enabled) {
                if (residual_cycles) {
                    cycle_start=accumulated;
                    cycle_residual=accumulated_residual;
                    shiftSE2PackedCorrectionReference(packed_residual_workspace,accumulated-correction_reference);
                    correction_reference=accumulated;
                } else {
                    stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                        packed_residual_workspace, cycle_start, packed_sweep_threads);
                    fineResidualSyntheticSE2PackedInto(
                        packed_residual_workspace, cycle_start, cycle_residual, packed_sweep_threads);
                }
            }
            if (use_packed_residual_solver) {
                const std::string smoother=std::getenv("HGBP_SE2_SMOOTHER")?std::getenv("HGBP_SE2_SMOOTHER"):"gbp";
                if(smoother=="jacobi") {
                    blockJacobiSE2PackedIterations(packed_residual_workspace,pre_sweeps,packed_sweep_threads);
                    jacobi_sweeps_executed+=pre_sweeps;
                } else if(smoother=="gbp") {
                synchronousIterationsSyntheticSE2PackedResidualWorkspace(
                    packed_residual_workspace,
                    pre_sweeps,
                    fastsync_factor_pass_sec,
                    fastsync_variable_pass_sec,
                    packed_sweep_threads
                );
                gbp_sweeps_executed+=pre_sweeps;
                }
            } else {
                for (int sweep = 0; sweep < pre_sweeps; ++sweep) {
                    if (use_fastsync_local_packed) {
                    synchronousIterationFastSyncLocalPacked(
                        fastsync_local_packed_layout,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else if (use_fastsync_local) {
                        synchronousIterationFastSyncLocal(
                        fastsync_local_layout,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else if (enable_singlecore_fastsync &&
                        sync_num_threads == 1 &&
                        fastsync_layout.all_dofs3 &&
                        !profile_sync_thread_utilization) {
                        synchronousIterationFastSyncSoA(
                        fastsync_layout,
                        residual_graph.eta_damping,
                        false,
                        1,
                        fastsync_factor_pass_sec,
                        fastsync_variable_pass_sec
                    );
                    } else {
                        residual_graph.synchronousIteration();
                    }
                }
            }
            const auto sweeps_t1 = SteadyClock::now();
            sweeps_sec += elapsedSeconds(sweeps_t0, sweeps_t1);

            audit_state("after_sweeps",cyc);
            apply_upper_correction();
            audit_state("after_coarse",cyc);
            if (cycle_energy_enabled) {
                const auto energy_t0 = SteadyClock::now();
                stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                    packed_residual_workspace, e_now, packed_sweep_threads);
                fineResidualSyntheticSE2PackedInto(
                    packed_residual_workspace, e_now, trial_residual, packed_sweep_threads);
                if (residual_cycles) e_now+=cycle_start;
                const Eigen::VectorXd unaccelerated = e_now;
                cycle_energy.accept(cycle_start, cycle_residual, e_now, trial_residual);
                if (residual_cycles) {
                    accumulated=e_now;
                    accumulated_residual=trial_residual;
                } else {
                    delta_e = e_now-unaccelerated;
                    injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(
                        packed_residual_workspace, delta_e, packed_sweep_threads);
                }
                prolong_inject_sec += elapsedSeconds(energy_t0,SteadyClock::now());
                audit_state("after_energy",cyc);
            }
        }
        if (outer == num_outer) {
            for (int polish = 0; polish < upper_polish_passes; ++polish) {
                apply_upper_correction();
            }
        }

        if (use_packed_residual_solver) {
            stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(
                packed_residual_workspace,
                e_hat,
                packed_sweep_threads
            );
        } else if (use_fastsync_local_packed) {
            e_hat = stackedMeanVectorFastSyncLocalPacked(fastsync_local_packed_layout);
        } else if (use_fastsync_local) {
            e_hat = stackedMeanVectorFastSyncLocal(fastsync_local_layout);
        } else {
            e_hat = stackedMeanVector(residual_graph);
        }
        if (residual_cycles) e_hat=accumulated;
        const double recorded_factor_pass_sec =
            (use_packed_residual_solver || use_fastsync_local_packed || use_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              fastsync_layout.all_dofs3 &&
              !profile_sync_thread_utilization))
            ? fastsync_factor_pass_sec
            : residual_graph.sync_factor_pass_sec_accum;
        const double recorded_variable_pass_sec =
            (use_packed_residual_solver || use_fastsync_local_packed || use_fastsync_local ||
             (enable_singlecore_fastsync &&
              sync_num_threads == 1 &&
              fastsync_layout.all_dofs3 &&
              !profile_sync_thread_utilization))
            ? fastsync_variable_pass_sec
            : residual_graph.sync_variable_pass_sec_accum;
        const auto apply_t0 = SteadyClock::now();
        mg_poses = applyPoseDeltas(mg_poses, e_hat);
        const auto apply_t1 = SteadyClock::now();
        const auto obj_t0 = SteadyClock::now();
        const double nonlinear_obj = nonlinearObjective(problem, mg_poses, sync_num_threads);
        const auto obj_t1 = SteadyClock::now();
        const auto outer_t1 = SteadyClock::now();
        results.mg_history.push_back(
            OuterMGRow{
                outer,
                execution_path,
                nonlinear_obj,
                e_hat.norm(),
                factor_lam_max_asym,
                prefer_nonsym_safe_fastsync,
                static_cast<int>(basis.groups.size()),
                basis.coarse_dim,
                coarse_lambda_nnz,
                coarse_lambda_density,
                false,
                elapsedSeconds(outer_t0, outer_t1),
                elapsedSeconds(build_t0, build_t1),
                relin_stats.factor_relinearize_sec,
                relin_stats.reset_state_sec,
                basis_build_sec,
                basis_block_build_sec,
                basis_eigensolver_sec,
                basis_copyout_sec,
                elapsedSeconds(coarse_lam_t0, coarse_lam_t1),
                sweeps_sec,
                recorded_factor_pass_sec,
                recorded_variable_pass_sec,
                residual_graph.sync_factor_work_sec_accum,
                residual_graph.sync_factor_wait_sec_accum,
                residual_graph.sync_variable_work_sec_accum,
                residual_graph.sync_variable_wait_sec_accum,
                residual_graph.sync_profiled_threads,
                coarse_eta_sec,
                coarse_solve_sec,
                prolong_inject_sec,
                elapsedSeconds(apply_t0, apply_t1),
                elapsedSeconds(obj_t0, obj_t1),
                basis_partial_attempted,
                basis_partial_converged,
                basis_partial_fallback,
                basis_partial_total_iters,
                gbp_sweeps_executed,jacobi_sweeps_executed,coarse_solves,
                packed_residual_workspace.full_precision_sweeps,
                packed_residual_workspace.eta_only_sweeps,
                packed_residual_workspace.precision_checks,
                packed_residual_workspace.precision_freezes,
                packed_residual_workspace.precision_thaws,
                packed_residual_workspace.first_precision_freeze_sweep,
                packed_residual_workspace.last_precision_residual,
            }
        );
        results.mg_pose_history.push_back(mg_poses);
    }

    if (packed_residual_topology_valid) {
        results.packed_schur_adjugate_fallback_count = packed_residual_workspace.schur_adjugate_fallback_count;
        results.packed_schur_general_failure_count = packed_residual_workspace.schur_general_failure_count;
        results.packed_kernel_prep_cycles = packed_residual_workspace.kernel_prep_cycles;
        results.packed_kernel_schur0_cycles = packed_residual_workspace.kernel_schur0_cycles;
        results.packed_kernel_schur1_cycles = packed_residual_workspace.kernel_schur1_cycles;
        results.packed_kernel_update_belief_cycles = packed_residual_workspace.kernel_update_belief_cycles;
        results.packed_fixedeta_full_lambda_sweeps = packed_residual_workspace.fixedeta_full_lambda_sweeps;
        results.packed_fixedeta_lambda_init_sweeps = packed_residual_workspace.fixedeta_lambda_init_sweeps;
        results.packed_fixedeta_eta_only_sweeps = packed_residual_workspace.fixedeta_eta_only_sweeps;
        results.packed_fixedeta_eta_map_builds = packed_residual_workspace.fixedeta_eta_map_builds;
        results.packed_fixedeta_serial_delta_sweeps = packed_residual_workspace.fixedeta_serial_delta_sweeps;
    }

    results.solver_wall_sec = elapsedSeconds(solver_wall_start,SteadyClock::now());
    return results;
}

#include "detail/se2_output.inc"

}  // namespace slam
