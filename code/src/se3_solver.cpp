#include "internal/se3_solver_impl.h"
#ifdef HGBP_PURE_GBP
#include "pure_gbp/projection.h"
#endif
#include "internal/scoped_worker_affinity.h"
#include "internal/partial_symmetric_eigen.h"
#include "internal/banded_symmetric_eigen.h"
#include "internal/se3_residual.h"
#include "internal/cycle_energy.h"
#include "internal/certified_coarse.h"
#include "internal/coarse_cholesky.h"
#include "internal/basis_alignment.h"
#include "internal/se3_precision_warm.h"
#include "internal/exact_basis_cache.h"
#include "internal/se3_boundary_precision.h"
#include "internal/se3_rigid_basis.h"
#include "internal/se3_objective.h"
#include "internal/se3_residual_certificate.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include <Eigen/CholmodSupport>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SparseCholesky>
#include <omp.h>

#include "gbp/Factor.h"
#include "gbp/VariableNode.h"

namespace slam {

double nonlinearObjectiveForDelta(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta
);

namespace {

#include "detail/se3_model.inc"

#include "detail/se3_coarse.inc"


#include "detail/se3_sweeps.inc"

#include "detail/se3_basis.inc"

#include "detail/se3_assembly.inc"

#include "detail/se3_linearization.inc"

void writePoseJson(std::ostream& out, const SE3Pose& pose) {
    const SE3Pose p = normalizedPose(pose);
    out << "[" << jsonNumber(p.t.x())
        << ", " << jsonNumber(p.t.y())
        << ", " << jsonNumber(p.t.z())
        << ", " << jsonNumber(p.q.x())
        << ", " << jsonNumber(p.q.y())
        << ", " << jsonNumber(p.q.z())
        << ", " << jsonNumber(p.q.w())
        << "]";
}

void writePoseVectorJson(std::ostream& out, const SE3PoseVector& poses, int indent) {
    const std::string pad(indent, ' ');
    out << "[\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        out << pad;
        writePoseJson(out, poses[i]);
        out << (i + 1 == poses.size() ? "\n" : ",\n");
    }
    out << std::string(std::max(0, indent - 2), ' ') << "]";
}

void writePoseHistoryJsonArray(std::ostream& out, const SE3PoseHistory& history, int indent) {
    const std::string pad(indent, ' ');
    out << "[\n";
    for (size_t i = 0; i < history.size(); ++i) {
        out << pad;
        writePoseVectorJson(out, history[i], indent + 2);
        out << (i + 1 == history.size() ? "\n" : ",\n");
    }
    out << std::string(std::max(0, indent - 2), ' ') << "]";
}

}  // namespace

#include "detail/se3_problem.inc"

SyntheticSE3ExperimentResults runSyntheticSE3Experiment(
    const SyntheticSE3Problem& problem,
    int num_outer,
    int inner_cycles,
    int pre_sweeps,
    int group_size,
    int r_reduced,
    const RobustLossConfig& robust_loss_config,
    int sync_num_threads,
    bool direct_enabled
) {
    SyntheticSE3ExperimentResults results;
    const auto solver_wall_start = SteadyClock::now();
    const int thread_count = effectiveThreadCount(sync_num_threads);
    const ScopedSE3SingleThreadRuntime single_thread_runtime_guard(
        thread_count == 1 && se3SingleThreadRuntimeGuardEnabled()
    );
    results.num_poses = static_cast<int>(problem.init_poses.size());
    results.num_edges = static_cast<int>(problem.edges.size());
    results.initial_objective = nonlinearObjective(problem, problem.init_poses);
    results.initial_poses = problem.init_poses;

    SE3PoseVector direct_poses = problem.init_poses;
    SE3PoseVector mg_poses = problem.init_poses;
    results.direct_history.push_back(SyntheticSE3OuterDirectRow{0, results.initial_objective});
    results.mg_history.push_back(SyntheticSE3OuterMGRow{0, results.initial_objective});
    const bool robust_line_search=se3EnvFlagEnabled("HGBP_SE3_ROBUST_LINE_SEARCH");
    const bool buffered_objective=se3EnvFlagEnabled("HGBP_SE3_BUFFERED_OBJECTIVE");
    if(buffered_objective && !robust_line_search)
        throw std::runtime_error("Buffered objective experiment requires consistent Huber acceptance");
    SE3ObjectiveWorkspace objective_workspace;
    SE3PoseVector buffered_trial_poses;
    if(robust_line_search) results.mg_history.back().nonlinear_huber_objective=
        evaluateSE3Objectives(problem,mg_poses,robust_loss_config).huber;
    results.direct_pose_history.push_back(direct_poses);
    results.mg_pose_history.push_back(mg_poses);

    std::unique_ptr<SyntheticSE3ResidualGraphWorkspace> mg_workspace;
    mg_workspace = std::make_unique<SyntheticSE3ResidualGraphWorkspace>(
        buildSyntheticSE3ResidualGraphWorkspace(problem, 1e-12)
    );
    const int basis_rebuild_period = se3BasisRebuildPeriod();
    const int basis_rebuild_warmup_outers = se3BasisRebuildWarmupOuters();
    SE3BasisData cached_basis;
    bool cached_basis_valid = false;
    std::vector<Eigen::MatrixXd> warm_start_basis_local_bases;
    std::vector<ExactBasisCacheEntry> exact_basis_cache;
    const bool use_exact_basis_cache = se3EnvFlagEnabled("HGBP_SE3_EXACT_BASIS_CACHE");
    const std::string eta_lift=std::getenv("HGBP_SE3_ETA_LIFT")?std::getenv("HGBP_SE3_ETA_LIFT"):"none";
    if(eta_lift!="none" && eta_lift!="degree" && eta_lift!="precision")
        throw std::runtime_error("Unknown SE3 eta lift policy");
    const bool cycle_audit_enabled=se3EnvFlagEnabled("HGBP_SE3_CYCLE_AUDIT");
    const bool residual_stop_enabled=se3EnvFlagEnabled("HGBP_SE3_RESIDUAL_STOP");
    const std::string precision_defect=std::getenv("HGBP_SE3_PRECISION_DEFECT")?std::getenv("HGBP_SE3_PRECISION_DEFECT"):"none";
    if(precision_defect!="none" && precision_defect!="diagonal")
        throw std::runtime_error("Unknown precision-defect experiment");
    if(precision_defect!="none" && (eta_lift!="precision" ||
       !std::getenv("HGBP_SE3_PRECISION_INITIALIZATION") ||
       std::string(std::getenv("HGBP_SE3_PRECISION_INITIALIZATION"))!="cold" ||
       (std::getenv("HGBP_SE3_SMOOTHER") && std::string(std::getenv("HGBP_SE3_SMOOTHER"))!="gbp")))
        throw std::runtime_error("Precision-defect requires cold diagonal messages, GBP path and precision eta lift");
    SparseCholeskyFactor coarse_factor_cache;
    bool coarse_factor_cache_valid = false;
    Eigen::SparseMatrix<double> coarse_lam_direct_cache;
    JointInfSparsePatternCache mg_joint_cache;
    CoarseLambdaAssemblyCache mg_coarse_lambda_cache;
    const bool reuse_coarse_pattern = se3ReuseCoarsePatternEnabled();
    const int coarse_numeric_rebuild_period = se3CoarseNumericRebuildPeriod();
    const int coarse_reuse_pcg_iterations = se3CoarseReusePcgIterations();
    const bool automatic_coarse=se3EnvFlagEnabled("HGBP_SE3_AUTOMATIC_COARSE");
    const bool amortized_coarse=se3EnvFlagEnabled("HGBP_SE3_AMORTIZED_COARSE");
    double coarse_extra_work = 0.0;
    const std::string precision_initialization=std::getenv("HGBP_SE3_PRECISION_INITIALIZATION")?
        std::getenv("HGBP_SE3_PRECISION_INITIALIZATION"):"cold";
    const std::string boundary_basis=std::getenv("HGBP_SE3_BOUNDARY_BASIS")?
        std::getenv("HGBP_SE3_BOUNDARY_BASIS"):"none";
    const std::string basis_prepass=std::getenv("HGBP_SE3_BASIS_PREPASS")?
        std::getenv("HGBP_SE3_BASIS_PREPASS"):"none";
    if(basis_prepass!="none" && basis_prepass!="control" && basis_prepass!="boundary" && basis_prepass!="rigid")
        throw std::runtime_error("Unknown SE3 presweep basis policy");
    if(basis_prepass!="none" && (boundary_basis!="none" ||
        !se3EnvFlagEnabled("HGBP_SE3_BALANCED_INITIALIZATION") ||
        !se3PackedSoADeferGraphCopyEnabled() || !se3PackedSoASweepsEnabled()))
        throw std::runtime_error("Presweep basis requires isolated balanced/deferred packed path");
    if (boundary_basis!="none" && boundary_basis!="early_control" && boundary_basis!="warm_boundary" &&
        boundary_basis!="rigid_control" && boundary_basis!="warm_rigid" &&
        boundary_basis!="rigid_unanchored_control" && boundary_basis!="warm_rigid_unanchored" &&
        boundary_basis!="transport_boundary" && boundary_basis!="transport_rigid_control" &&
        boundary_basis!="transport_rigid")
        throw std::runtime_error("Unknown SE3 boundary basis experiment");
    if (boundary_basis!="none" && !se3EnvFlagEnabled("HGBP_SE3_BALANCED_INITIALIZATION"))
        throw std::runtime_error("SE3 boundary experiment requires balanced packed initialization");
    SE3PrecisionWarmState precision_warm;
    const char* precision_audit_path=std::getenv("HGBP_SE3_PRECISION_AUDIT");

    for (int outer = 1; outer <= num_outer; ++outer) {
        if (direct_enabled) {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            gbp::FactorGraph graph = buildLinearizedResidualGraph(problem, direct_poses, 1e-12, robust_loss_config);
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = graph.jointDistributionInfSparse();
            const Eigen::SparseMatrix<double> lam = symmetrizeSparse(J.lam);
            const auto joint_t1 = SteadyClock::now();

            const auto solve_t0 = SteadyClock::now();
            const Eigen::VectorXd e_star = solveSparseCholesky(lam, J.eta, 1e-10);
            const auto solve_t1 = SteadyClock::now();

            const auto apply_t0 = SteadyClock::now();
            direct_poses = applyPoseDeltas(direct_poses, e_star);
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = nonlinearObjective(problem, direct_poses);
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            results.direct_history.push_back(
                SyntheticSE3OuterDirectRow{
                    outer,
                    nonlinear_obj,
                    e_star.norm(),
                    (J.eta - lam * e_star).norm(),
                    elapsedSeconds(total_t0, total_t1),
                    elapsedSeconds(build_t0, build_t1),
                    elapsedSeconds(joint_t0, joint_t1),
                    elapsedSeconds(solve_t0, solve_t1),
                    elapsedSeconds(apply_t0, apply_t1),
                    elapsedSeconds(obj_t0, obj_t1),
                }
            );
            results.direct_pose_history.push_back(direct_poses);
        }

        {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            SyntheticSE3ResidualRelinearizeStats relin_stats;
            relinearizeSyntheticSE3ResidualGraph(
                *mg_workspace,
                problem,
                mg_poses,
                thread_count,
                outer,
                &relin_stats,
                robust_loss_config
            );
            gbp::FactorGraph& graph = mg_workspace->graph;
            graph.setSyncNumThreads(thread_count);
            graph.setProfileSyncTiming(se3ProfileSyncTimingEnabled());
            graph.resetSyncTiming();
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            bool use_implicit_fine_operator = se3ImplicitFineOperatorEnabled();
            SyntheticSE3PackedSoAWorkspace* implicit_fine_workspace =
                use_implicit_fine_operator
                    ? preparePackedSoAWorkspace(graph, problem)
                    : nullptr;
            use_implicit_fine_operator = implicit_fine_workspace != nullptr;
            const int implicit_fine_threads =
                se3ImplicitFineOperatorThreadCount(thread_count);
            gbp::FactorGraph::JointInfResult J;
            Eigen::SparseMatrix<double> lam_storage;
            const Eigen::SparseMatrix<double>* lam_ptr = nullptr;
            if (use_implicit_fine_operator) {
                assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(
                    *implicit_fine_workspace,
                    J.eta,
                    implicit_fine_threads
                );
                J.total_dim = static_cast<int>(J.eta.size());
                lam_storage.resize(J.total_dim, J.total_dim);
                lam_ptr = &lam_storage;
            } else {
                fillJointDistributionInfSparseCached(graph, mg_joint_cache, J.eta);
                J.total_dim = mg_joint_cache.total_dim;
                lam_ptr = &mg_joint_cache.lam;
            }
            const Eigen::SparseMatrix<double>& lam = *lam_ptr;
            if (use_implicit_fine_operator &&
                se3ImplicitFineOperatorCompareEnabled()) {
                Eigen::VectorXd reference_eta;
                fillJointDistributionInfSparseCached(
                    graph,
                    mg_joint_cache,
                    reference_eta
                );
                Eigen::VectorXd probe =
                    Eigen::VectorXd::LinSpaced(J.total_dim, -0.5, 0.5);
                Eigen::VectorXd implicit_product;
                multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
                    *implicit_fine_workspace,
                    probe,
                    implicit_product,
                    implicit_fine_threads
                );
                const Eigen::VectorXd reference_product =
                    mg_joint_cache.lam * probe;
                const double eta_relative_error =
                    (J.eta - reference_eta).norm() /
                    std::max(reference_eta.norm(), 1e-15);
                const double product_relative_error =
                    (implicit_product - reference_product).norm() /
                    std::max(reference_product.norm(), 1e-15);
                std::cerr
                    << "[se3 implicit fine compare] outer=" << outer
                    << " eta_rel=" << eta_relative_error
                    << " product_rel=" << product_relative_error
                    << "\n";
            }
            auto assemble_fine_residual = [&](
                const Eigen::VectorXd& x,
                Eigen::VectorXd& residual
            ) {
                if (use_implicit_fine_operator) {
                    multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(
                        *implicit_fine_workspace,
                        x,
                        residual,
                        implicit_fine_threads
                    );
                    residual = J.eta - residual;
                } else {
                    residual.noalias() = J.eta - lam * x;
                }
            };
            const auto joint_t1 = SteadyClock::now();

            Eigen::VectorXd e_now = stackedMeanVector(graph);
            SE3PrecisionWarmStats precision_warm_stats;
            double precision_warm_sec=0;
            auto initialize_packed_messages = [&]() {
                if (!se3EnvFlagEnabled("HGBP_SE3_BALANCED_INITIALIZATION")) return;
                if(!implicit_fine_workspace) throw std::runtime_error("Balanced SE3 requires packed workspace");
                initializeBalancedSE3PackedMessages(*implicit_fine_workspace,thread_count);
                if(precision_audit_path && precision_audit_path[0] && outer==2)
                    auditSE3PrecisionWarm(precision_audit_path,graph,*implicit_fine_workspace,precision_warm,mg_poses,thread_count);
                if(precision_initialization!="cold" && !precision_warm.poses.empty()) {
                    const auto warm_t0=SteadyClock::now();
                    precision_warm_stats=initializeWarmSE3Precision(*implicit_fine_workspace,precision_warm,
                        mg_poses,precision_initialization=="warm-bounded",thread_count,
                        precision_initialization=="warm-transport" || precision_initialization=="warm-transport-balanced");
                    precision_warm_sec=elapsedSeconds(warm_t0,SteadyClock::now());
                }
                stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(*implicit_fine_workspace,e_now,thread_count);
            };
            const std::string smoother=std::getenv("HGBP_SE3_SMOOTHER")?std::getenv("HGBP_SE3_SMOOTHER"):"gbp";
            if (boundary_basis!="none" || basis_prepass!="none") initialize_packed_messages();
            double basis_prepass_sec=0;
            if(basis_prepass!="none") {
                if(!implicit_fine_workspace || smoother=="none")
                    throw std::runtime_error("Presweep basis needs a packed smoother");
                if(basis_prepass!="control" && smoother!="gbp")
                    throw std::runtime_error("Message-conditioned basis requires actual GBP precision");
                const auto prepass_t0=SteadyClock::now();
                if(smoother=="jacobi")
                    blockJacobiSE3PackedIterations(*implicit_fine_workspace,pre_sweeps,thread_count);
                else manualSynchronousIterations(graph,pre_sweeps,thread_count,0,outer,&problem);
                basis_prepass_sec=elapsedSeconds(prepass_t0,SteadyClock::now());
                // e_now deliberately retains the pre-sweep mean: the first cycle's
                // residual and energy line search must use the unchanged origin.
            }

            const auto basis_t0 = SteadyClock::now();
            const bool basis_within_warmup =
                basis_rebuild_warmup_outers > 0 && outer <= basis_rebuild_warmup_outers;
            const int post_warmup_outer = outer - basis_rebuild_warmup_outers;
            const bool rebuild_basis =
                !cached_basis_valid ||
                basis_rebuild_period <= 1 ||
                basis_within_warmup ||
                (((post_warmup_outer - 1) % basis_rebuild_period) == 0);
            if (rebuild_basis) {
                const bool transported=precision_warm_stats.slots>0;
                const bool use_boundary=boundary_basis=="warm_boundary" || boundary_basis=="warm_rigid" ||
                    boundary_basis=="warm_rigid_unanchored" ||
                    (transported && (boundary_basis=="transport_boundary" || boundary_basis=="transport_rigid")) ||
                    basis_prepass=="boundary" || basis_prepass=="rigid";
                const bool keep_rigid=boundary_basis=="rigid_control" || boundary_basis=="warm_rigid" ||
                    boundary_basis=="rigid_unanchored_control" || boundary_basis=="warm_rigid_unanchored" ||
                    (transported && (boundary_basis=="transport_rigid_control" || boundary_basis=="transport_rigid")) ||
                    basis_prepass=="rigid";
                const bool exclude_anchor=boundary_basis=="rigid_unanchored_control" || boundary_basis=="warm_rigid_unanchored" ||
                    boundary_basis=="transport_rigid_control" || boundary_basis=="transport_rigid" || basis_prepass=="rigid";
                SE3BasisData fresh_basis = buildMessageConditionedBasis(
                    graph,
                    &mg_poses,
                    group_size,
                    r_reduced,
                    thread_count,
                    warm_start_basis_local_bases.empty() ? nullptr : &warm_start_basis_local_bases,
                    use_exact_basis_cache ? &exact_basis_cache : nullptr,
                    use_boundary ? implicit_fine_workspace : nullptr,
                    keep_rigid,
                    exclude_anchor
                );
                // Keep eigensolver warm starts independent of the optional
                // coarse-coordinate rotation, so alignment cannot change its seeds.
                warm_start_basis_local_bases = fresh_basis.local_bases;
                if(automatic_coarse && cached_basis.local_bases.size()==fresh_basis.local_bases.size()) {
                    #pragma omp parallel for schedule(static) num_threads(thread_count) if(thread_count>1)
                    for(int g=0;g<static_cast<int>(fresh_basis.local_bases.size());++g) {
                        if (fresh_basis.exact_reused[g] &&
                            cached_basis.local_bases[g].rows() == fresh_basis.local_bases[g].rows() &&
                            cached_basis.local_bases[g].cols() == fresh_basis.local_bases[g].cols())
                            fresh_basis.local_bases[g] = cached_basis.local_bases[g];
                        else alignBasisCoordinates(fresh_basis.local_bases[g],cached_basis.local_bases[g]);
                    }
                }
                cached_basis = std::move(fresh_basis);
                cached_basis_valid = true;
            }
            const auto basis_t1 = SteadyClock::now();
            const SE3BasisData& basis = cached_basis;

            const auto coarse_lam_t0 = SteadyClock::now();
            const bool use_cached_coarse_lambda =
                se3CachedCoarseLambdaEnabled() &&
                static_cast<int>(basis.local_bases.size()) >= se3CachedCoarseLambdaMinGroups();
            bool rebuild_coarse_numeric = automatic_coarse
                ? (!coarse_factor_cache_valid || coarse_factor_cache.ldlt_solver.rows()!=basis.coarse_dim)
                : (
                !coarse_factor_cache_valid ||
                rebuild_basis ||
                coarse_numeric_rebuild_period <= 1 ||
                basis_within_warmup ||
                (((post_warmup_outer - 1) % coarse_numeric_rebuild_period) == 0));
            const Eigen::SparseMatrix<double>* coarse_lam_ptr = nullptr;
            if (use_cached_coarse_lambda) {
                coarse_lam_ptr =
                    &assembleCoarseLambdaCachedInPlace(
                        graph,
                        basis,
                        mg_coarse_lambda_cache,
                        thread_count
                    );
            } else {
                coarse_lam_direct_cache =
                    assembleCoarseLambdaDirect(
                        graph,
                        basis,
                        thread_count > 1,
                        thread_count
                    );
                coarse_lam_ptr = &coarse_lam_direct_cache;
            }
            Eigen::SparseMatrix<double>& cached_coarse_lam_ref = mg_coarse_lambda_cache.lam;
            bool coarse_ridge_added_inplace = false;
            if (use_cached_coarse_lambda &&
                se3CoarseRidgeInplaceEnabled() &&
                coarse_lam_ptr == &cached_coarse_lam_ref) {
                for (int i = 0; i < cached_coarse_lam_ref.rows(); ++i) {
                    cached_coarse_lam_ref.coeffRef(i, i) += 1e-8;
                }
                cached_coarse_lam_ref.makeCompressed();
                coarse_ridge_added_inplace = true;
            }
            const Eigen::SparseMatrix<double>& coarse_lam = *coarse_lam_ptr;
            double coarse_reuse_work_ratio=0;
            CoarseWorkEstimate coarse_work;
            bool coarse_amortized_refresh = false;
            const double coarse_extra_work_before = coarse_extra_work;
            if(automatic_coarse && !rebuild_coarse_numeric) {
                coarse_work=coarse_factor_cache.ldlt_solver.workEstimate(coarse_lam.nonZeros());
                coarse_reuse_work_ratio=coarse_work.reuseRatio(inner_cycles);
                if(coarse_reuse_work_ratio>=1.) rebuild_coarse_numeric=true;
                if(amortized_coarse && coarse_extra_work >= coarse_work.factor_work) {
                    rebuild_coarse_numeric=true;
                    coarse_amortized_refresh=true;
                }
            }
            const auto coarse_lam_t1 = SteadyClock::now();

            SparseCholeskyFactor coarse_factor_local;
            const bool persist_coarse_factor =
                automatic_coarse || reuse_coarse_pattern || coarse_numeric_rebuild_period > 1;
            SparseCholeskyFactor* coarse_factor_ptr =
                persist_coarse_factor ? &coarse_factor_cache : &coarse_factor_local;
            const auto coarse_factor_t0 = SteadyClock::now();
#ifdef HGBP_PURE_GBP
            pure_gbp::begin(coarse_lam,coarse_ridge_added_inplace?0.0:1e-8,12,thread_count);
#else
            if (rebuild_coarse_numeric) {
                factorizeSparseCholesky(
                    coarse_lam,
                    coarse_ridge_added_inplace ? 0.0 : 1e-8,
                    *coarse_factor_ptr,
                    reuse_coarse_pattern && coarse_factor_cache_valid,
                    true
                );
                coarse_factor_cache_valid = persist_coarse_factor;
                coarse_extra_work = 0.0;
            }
#endif
            const auto coarse_factor_t1 = SteadyClock::now();
            double coarse_factorize_sec = elapsedSeconds(coarse_factor_t0, coarse_factor_t1);
            bool coarse_factor_current=rebuild_coarse_numeric;
            int coarse_reuse_attempts=0,coarse_reuse_accepts=0,coarse_reuse_fallbacks=0;
            int coarse_reuse_iterations=0,coarse_reuse_max_iterations=0;
            int coarse_reuse_preconditioner_calls=0,coarse_reuse_matvec_calls=0;
            double coarse_reuse_max_accepted_residual=0;
            double sweeps_sec = basis_prepass_sec;
            double coarse_eta_sec = 0.0;
            double coarse_solve_sec = coarse_factorize_sec;
            double coarse_backsolve_sec = 0.0;
            double prolong_inject_sec = 0.0;
            double first_cycle_residual_before_sweeps = 0.0;
            double first_cycle_residual_after_sweeps = 0.0;
            double first_cycle_residual_after_coarse = 0.0;
            double last_cycle_residual_before_sweeps = 0.0;
            double last_cycle_residual_after_sweeps = 0.0;
            double last_cycle_residual_after_coarse = 0.0;
            double smoother_translation_update_norm_sum = 0.0;
            double smoother_rotation_update_norm_sum = 0.0;
            int smoother_local_rejects_total = 0;
            if (boundary_basis=="none" && basis_prepass=="none") initialize_packed_messages();
            Eigen::VectorXd residual_before_sweeps(basis.total_dim);
            Eigen::VectorXd residual_after_sweeps(basis.total_dim);
            Eigen::VectorXd residual_after_coarse(basis.total_dim);
            Eigen::VectorXd coarse_eta(basis.coarse_dim);
            Eigen::VectorXd delta_z(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_residual(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_preconditioned_residual(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_direction(basis.coarse_dim);
            Eigen::VectorXd coarse_pcg_matrix_direction(basis.coarse_dim);
            Eigen::VectorXd delta_fine(basis.total_dim);
            int executed_inner_cycles = 0;
            int coarse_solves=0;
            const bool cycle_energy_enabled = se3EnvFlagEnabled("HGBP_CYCLE_ENERGY");
            if(cycle_audit_enabled && !implicit_fine_workspace)
                throw std::runtime_error("Cycle audit requires the packed fine workspace");
            CycleEnergyHistory cycle_energy(inner_cycles);
            std::vector<SE3CycleAuditRow> cycle_audit;
            int message_lifts=0;
            double message_lift_sec=0;
            SE3ResidualCertificate residual_certificate;
            int residual_stop_checks=0;
            double residual_stop_sec=0, residual_stop_relative=0;
            bool residual_stopped=false;
            if(residual_stop_enabled) {
                if(!implicit_fine_workspace) throw std::runtime_error("Residual stopping requires packed canonical operator");
                const auto t0=SteadyClock::now();
                residual_certificate.initialize(*implicit_fine_workspace,thread_count);
                residual_stop_sec+=elapsedSeconds(t0,SteadyClock::now());
            }
            bool have_previous_residual_after_coarse = false;
            double previous_residual_after_coarse_norm = 0.0;
            for (int cyc = 0; cyc < inner_cycles; ++cyc) {
                SE3CycleAuditRow audit;
                audit.cycle=cyc+1;
                double residual_before_sweeps_norm = 0.0;
                if (have_previous_residual_after_coarse) {
                    residual_before_sweeps_norm = previous_residual_after_coarse_norm;
                    if (cycle_energy_enabled || cycle_audit_enabled) residual_before_sweeps = residual_after_coarse;
                } else {
                    assemble_fine_residual(e_now, residual_before_sweeps);
                    residual_before_sweeps_norm = residual_before_sweeps.norm();
                }
                const auto sweeps_t0 = SteadyClock::now();
                SE3SmootherSweepStats smoother_stats;
                smoother_stats.residual_before_norm = residual_before_sweeps_norm;
                const Eigen::VectorXd e_before_gbp_sweeps = e_now;

                const bool defer_packed_graph_copy =
                    se3PackedSoADeferGraphCopyEnabled() &&
                    se3PackedSoASweepsEnabled();
                if(cyc==0 && basis_prepass!="none") {
                    // Already executed, and included in the packed sweep counters.
                } else if(smoother=="jacobi") {
                    if(!implicit_fine_workspace) throw std::runtime_error("Jacobi control requires packed workspace");
                    blockJacobiSE3PackedIterations(*implicit_fine_workspace,pre_sweeps,thread_count);
                } else if(smoother=="gbp") {
                manualSynchronousIterations(
                    graph,
                    pre_sweeps,
                    thread_count,
                    cyc * pre_sweeps,
                    outer,
                    &problem
                );
                }
                if (!(defer_packed_graph_copy &&
                      tryStackedMeanVectorPackedSoA(graph, thread_count, e_now))) {
                    e_now = stackedMeanVector(graph);
                }
                const Eigen::VectorXd gbp_delta = e_now - e_before_gbp_sweeps;
                accumulateSE3DeltaComponentNorms(
                    gbp_delta,
                    smoother_stats.translation_update_norm_sum,
                    smoother_stats.rotation_update_norm_sum
                );
                assemble_fine_residual(e_now, residual_after_sweeps);
                smoother_stats.residual_after_norm = residual_after_sweeps.norm();
                const auto sweeps_t1 = SteadyClock::now();
                sweeps_sec += elapsedSeconds(sweeps_t0, sweeps_t1);
                smoother_translation_update_norm_sum += smoother_stats.translation_update_norm_sum;
                smoother_rotation_update_norm_sum += smoother_stats.rotation_update_norm_sum;
                smoother_local_rejects_total += smoother_stats.local_rejects;

                const double residual_after_sweeps_norm = smoother_stats.residual_after_norm;
                if (cyc == 0) {
                    first_cycle_residual_before_sweeps = residual_before_sweeps_norm;
                    first_cycle_residual_after_sweeps = residual_after_sweeps_norm;
                }
                last_cycle_residual_before_sweeps = residual_before_sweeps_norm;
                last_cycle_residual_after_sweeps = residual_after_sweeps_norm;
                const auto coarse_eta_t0 = SteadyClock::now();
                if(cycle_audit_enabled) {
                    audit.initial_residual=residual_before_sweeps_norm;
                    audit.fine_residual=residual_after_sweeps_norm;
                    audit.fine_gain=0.5*gbp_delta.dot(residual_before_sweeps+residual_after_sweeps);
                    audit.fine_energy=gbp_delta.dot(residual_before_sweeps-residual_after_sweeps);
                }
                restrictToCoarseInto(basis, residual_after_sweeps, coarse_eta);
                const auto coarse_eta_t1 = SteadyClock::now();
                coarse_eta_sec += elapsedSeconds(coarse_eta_t0, coarse_eta_t1);

                const auto coarse_solve_t0 = SteadyClock::now();
                double fallback_factor_sec=0;
                if(se3EnvFlagEnabled("HGBP_SE3_SKIP_COARSE")) {
                    delta_z.setZero();
                } else {
                ++coarse_solves;
#ifdef HGBP_PURE_GBP
                if(pure_gbp::needsProjectedInitialization())
                    pure_gbp::initializeProjected(pure_gbp::projectSE3(graph,basis,thread_count));
                pure_gbp::solve(coarse_eta,delta_z);
#else
                bool certified_reused=false;
                if(automatic_coarse && !coarse_factor_current) {
                    ++coarse_reuse_attempts;
                    const auto certificate=certifiedCoarsePcg(coarse_lam,coarse_ridge_added_inplace?0.0:1e-8,
                        coarse_eta,[&](const Eigen::VectorXd& rhs,Eigen::VectorXd& x) {
                            solveWithSparseCholeskyInto(*coarse_factor_ptr,rhs,x);
                        },delta_z,coarse_pcg_residual,coarse_pcg_preconditioned_residual,
                        coarse_pcg_direction,coarse_pcg_matrix_direction);
                    certified_reused=certificate.accepted;
                    coarse_reuse_iterations += certificate.iterations;
                    coarse_reuse_max_iterations = std::max(coarse_reuse_max_iterations, certificate.iterations);
                    coarse_reuse_preconditioner_calls += certificate.preconditioner_calls;
                    coarse_reuse_matvec_calls += certificate.matvec_calls;
                    if(certified_reused) {
                        if(amortized_coarse) coarse_extra_work += coarse_work.extraWork(certificate);
                        ++coarse_reuse_accepts;
                        coarse_reuse_max_accepted_residual=std::max(coarse_reuse_max_accepted_residual,certificate.relative_residual);
                    } else {
                        ++coarse_reuse_fallbacks;
                        const auto t0=SteadyClock::now();
                        factorizeSparseCholesky(coarse_lam,coarse_ridge_added_inplace?0.0:1e-8,
                            *coarse_factor_ptr,reuse_coarse_pattern,true);
                        fallback_factor_sec=elapsedSeconds(t0,SteadyClock::now());
                        coarse_factorize_sec+=fallback_factor_sec;
                        coarse_factor_current=true;
                        coarse_extra_work=0.0;
                    }
                }
                if(certified_reused) {
                    // delta_z was accepted using the current coarse equation.
                } else if (coarse_factor_current) {
                    solveWithSparseCholeskyInto(
                        *coarse_factor_ptr,
                        coarse_eta,
                        delta_z
                    );
                } else {
                    solveWithReusedCoarsePcgInto(
                        coarse_lam,
                        *coarse_factor_ptr,
                        coarse_eta,
                        coarse_reuse_pcg_iterations,
                        delta_z,
                        coarse_pcg_residual,
                        coarse_pcg_preconditioned_residual,
                        coarse_pcg_direction,
                        coarse_pcg_matrix_direction
                    );
                }
#endif
                }
                const auto coarse_solve_t1 = SteadyClock::now();
                const double backsolve_elapsed = elapsedSeconds(coarse_solve_t0, coarse_solve_t1);
                coarse_solve_sec += backsolve_elapsed;
                coarse_backsolve_sec += backsolve_elapsed-fallback_factor_sec;

                const auto inject_t0 = SteadyClock::now();
                prolongToFineInto(basis, delta_z, delta_fine);
                e_now.noalias() += delta_fine;
                const Eigen::VectorXd* coarse_delta_ptr = &delta_fine;
                const bool defer_packed_graph_copy_after_coarse =
                    se3PackedSoADeferGraphCopyEnabled() &&
                    se3PackedSoASweepsEnabled();
                const bool injected_packed =
                    defer_packed_graph_copy_after_coarse &&
                    tryApplyMeanDeltaPackedSoA(graph, *coarse_delta_ptr, thread_count);
                if (!injected_packed) {
                    injectCorrectionKeepMessages(graph, *coarse_delta_ptr);
                }
                manualSynchronousIterations(
                    graph,
                    se3PostCoarseMessageSweeps(),
                    thread_count
                );
                if (!(defer_packed_graph_copy_after_coarse &&
                      tryStackedMeanVectorPackedSoA(graph, thread_count, e_now))) {
                    e_now = stackedMeanVector(graph);
                }
                assemble_fine_residual(e_now, residual_after_coarse);
                if(cycle_audit_enabled) {
                    const Eigen::VectorXd actual_coarse_delta=e_now-e_before_gbp_sweeps-gbp_delta;
                    audit.coarse_residual=residual_after_coarse.norm();
                    audit.coarse_gain=0.5*actual_coarse_delta.dot(residual_after_sweeps+residual_after_coarse);
                    audit.coarse_energy=actual_coarse_delta.dot(residual_after_sweeps-residual_after_coarse);
                    audit.cross_energy=gbp_delta.dot(residual_after_sweeps-residual_after_coarse);
                }
                if (cycle_energy_enabled) {
                    const Eigen::VectorXd unaccelerated = e_now;
                    if(se3EnvFlagEnabled("HGBP_CYCLE_ENERGY_RESTART")) cycle_energy.clear();
                    cycle_energy.accept(e_before_gbp_sweeps, residual_before_sweeps,
                                        e_now, residual_after_coarse);
                    delta_fine = e_now - unaccelerated;
                    if (!(defer_packed_graph_copy_after_coarse &&
                          tryApplyMeanDeltaPackedSoA(graph, delta_fine, thread_count))) {
                        injectCorrectionKeepMessages(graph, delta_fine);
                    }
                }
                if(cycle_audit_enabled) {
                    audit.eta_defect_before=measureSE3PackedEtaMismatch(*implicit_fine_workspace);
                }
                if(eta_lift!="none") {
                    if(!implicit_fine_workspace || smoother!="gbp" || se3EnvFlagEnabled("HGBP_SE3_CYCLE_MESSAGE_REBUILD"))
                        throw std::runtime_error("Eta lift requires GBP and excludes message reconstruction");
                    const auto lift_t0=SteadyClock::now();
                    delta_fine=e_now-e_before_gbp_sweeps-gbp_delta;
                    liftSE3PackedMeanCorrection(*implicit_fine_workspace,delta_fine,eta_lift=="precision",thread_count);
                    ++message_lifts;
                    message_lift_sec+=elapsedSeconds(lift_t0,SteadyClock::now());
                }
                if(cycle_audit_enabled) {
                    audit.eta_defect_after=measureSE3PackedEtaMismatch(*implicit_fine_workspace);
                    audit.final_residual=residual_after_coarse.norm();
                    audit.total_gain=0.5*(e_now-e_before_gbp_sweeps).dot(residual_before_sweeps+residual_after_coarse);
                    Eigen::VectorXd true_residual;
                    assemble_fine_residual(e_now,true_residual);
                    audit.residual_defect_absolute=(true_residual-residual_after_coarse).norm();
                    audit.residual_defect_relative=audit.residual_defect_absolute/
                        std::max({true_residual.norm(),residual_before_sweeps_norm,1e-300});
                    cycle_audit.push_back(audit);
                }
                if(cyc+1<inner_cycles && se3EnvFlagEnabled("HGBP_SE3_CYCLE_MESSAGE_REBUILD")) {
                    if(!implicit_fine_workspace) throw std::runtime_error("Message rebuild requires packed workspace");
                    rebuildSE3PackedMessagesAtMean(*implicit_fine_workspace,e_now,residual_after_coarse,thread_count);
                }
                const double residual_after_coarse_norm = residual_after_coarse.norm();
                previous_residual_after_coarse_norm = residual_after_coarse_norm;
                have_previous_residual_after_coarse = true;
                if (cyc == 0) {
                    first_cycle_residual_after_coarse = residual_after_coarse_norm;
                }
                last_cycle_residual_after_coarse = residual_after_coarse_norm;
                const auto inject_t1 = SteadyClock::now();
                prolong_inject_sec += elapsedSeconds(inject_t0, inject_t1);
                ++executed_inner_cycles;
                if(residual_stop_enabled) {
                    const auto t0=SteadyClock::now();
                    residual_stop_relative=residual_certificate.relativeResidual(residual_after_coarse);
                    ++residual_stop_checks;
                    residual_stopped=SE3ResidualCertificate::accepts(residual_stop_relative) && cyc+1<inner_cycles;
                    residual_stop_sec+=elapsedSeconds(t0,SteadyClock::now());
                    if(residual_stopped) break;
                }
            }

            const Eigen::VectorXd e_hat = e_now;
            const double raw_e_hat_norm = e_hat.norm();
            const auto apply_t0 = SteadyClock::now();
            if((precision_initialization!="cold" || (precision_audit_path && precision_audit_path[0])) && implicit_fine_workspace) {
                const auto warm_t0=SteadyClock::now();
                precision_warm.capture(*implicit_fine_workspace,mg_poses,
                    precision_initialization=="warm-transport-balanced"?&e_hat:nullptr,thread_count);
                precision_warm_sec+=elapsedSeconds(warm_t0,SteadyClock::now());
            }
            const double base_raw_obj=results.mg_history.back().nonlinear_objective;
            const double base_obj=robust_line_search
                ? results.mg_history.back().nonlinear_huber_objective : base_raw_obj;
            double step_scale = 1.0;
            int line_search_rejects = 0;
            double trial_obj = base_obj;
            double trial_raw_obj=base_raw_obj;
            SE3PoseVector trial_poses = mg_poses;
            double outer_pose_update_sec=0,outer_acceptance_sec=0;
            int objective_evaluations=0;
            while (step_scale >= (1.0 / 1024.0)) {
                const auto update_t0=SteadyClock::now();
                if(buffered_objective) {
                    applySE3PoseDeltasInto(mg_poses,e_hat,step_scale,buffered_trial_poses,thread_count);
                    trial_poses.swap(buffered_trial_poses);
                } else trial_poses = applyPoseDeltas(mg_poses, step_scale * e_hat);
                const auto acceptance_t0=SteadyClock::now();
                outer_pose_update_sec+=elapsedSeconds(update_t0,acceptance_t0);
                ++objective_evaluations;
                if(robust_line_search) {
                    const auto objective=buffered_objective
                        ? evaluateSE3ObjectivesBuffered(problem,trial_poses,robust_loss_config,objective_workspace,thread_count)
                        : evaluateSE3Objectives(problem,trial_poses,robust_loss_config);
                    trial_obj=objective.huber;
                    trial_raw_obj=objective.raw;
                } else trial_obj = nonlinearObjective(problem, trial_poses);
                outer_acceptance_sec+=elapsedSeconds(acceptance_t0,SteadyClock::now());
                if (std::isfinite(trial_obj) && trial_obj <= base_obj) {
                    break;
                }
                step_scale *= 0.5;
                ++line_search_rejects;
            }
            if (step_scale < (1.0 / 1024.0)) {
                trial_poses = mg_poses;
                trial_obj = base_obj;
                trial_raw_obj=base_raw_obj;
                step_scale = 0.0;
            }
            mg_poses = trial_poses;
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = robust_line_search ? trial_raw_obj : trial_obj;
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            const int coarse_lambda_nnz = static_cast<int>(coarse_lam.nonZeros());
            const double coarse_lambda_density =
                basis.coarse_dim > 0
                ? static_cast<double>(coarse_lambda_nnz) /
                    (static_cast<double>(basis.coarse_dim) * static_cast<double>(basis.coarse_dim))
                : 0.0;
            double linear_residual_approx_norm =
                previous_residual_after_coarse_norm;
            if (!(step_scale == 1.0 && have_previous_residual_after_coarse)) {
                assemble_fine_residual(
                    step_scale * e_hat,
                    residual_after_coarse
                );
                linear_residual_approx_norm = residual_after_coarse.norm();
            }

            results.mg_history.push_back(
                SyntheticSE3OuterMGRow{
                    outer,
                    nonlinear_obj,
                    raw_e_hat_norm,
                    (step_scale * e_hat).norm(),
                    first_cycle_residual_before_sweeps,
                    first_cycle_residual_after_sweeps,
                    first_cycle_residual_after_coarse,
                    last_cycle_residual_before_sweeps,
                    last_cycle_residual_after_sweeps,
                    last_cycle_residual_after_coarse,
                    linear_residual_approx_norm,
                    step_scale,
                    line_search_rejects,
                    executed_inner_cycles,
                    smoother_translation_update_norm_sum,
                    smoother_rotation_update_norm_sum,
                    smoother_local_rejects_total,
                    graph.sync_total_sweeps_accum,
                    graph.sync_fixed_lam_hard_start_sweep,
                    graph.sync_fixed_lam_effective_start_sweep,
                    graph.sync_fixed_lam_factor_sweeps_accum,
                    graph.sync_eta_only_variable_sweeps_accum,
                    graph.sync_fixed_lam_hot_eta_sweeps_accum,
                    graph.sync_fixed_lam_hot_eta_entries,
                    static_cast<int>(basis.groups.size()),
                    basis.coarse_dim,
                    coarse_lambda_nnz,
                    coarse_lambda_density,
                    elapsedSeconds(total_t0, total_t1),
                    elapsedSeconds(build_t0, build_t1),
                    rebuild_basis ? elapsedSeconds(basis_t0, basis_t1) : 0.0,
                    rebuild_basis ? basis.block_build_sec : 0.0,
                    rebuild_basis ? basis.eigensolver_sec : 0.0,
                    rebuild_basis ? basis.copyout_sec : 0.0,
                    std::isfinite(basis.min_eigenvalue) ? basis.min_eigenvalue : 0.0,
                    std::isfinite(basis.max_eigenvalue) ? basis.max_eigenvalue : 0.0,
                    basis.negative_group_count,
                    basis.nonpositive_group_count,
                    basis.partial_attempt_count,
                    basis.partial_converged_count,
                    basis.full_eigensolver_count,
                    elapsedSeconds(coarse_lam_t0, coarse_lam_t1),
                    elapsedSeconds(joint_t0, joint_t1),
                    0.0, // No fine-level direct solve in an MG outer iteration.
                    sweeps_sec,
                    graph.sync_factor_pass_sec_accum,
                    graph.sync_variable_pass_sec_accum,
                    coarse_eta_sec,
                    coarse_solve_sec,
                    coarse_factorize_sec,
                    coarse_backsolve_sec,
                    prolong_inject_sec,
                    elapsedSeconds(apply_t0, apply_t1),
                    elapsedSeconds(obj_t0, obj_t1),
                    relin_stats.message_transport_sec,
                    relin_stats.factor_relinearize_sec,
                    relin_stats.reset_state_sec,
                    relin_stats.total_sec,
                }
            );
            if (implicit_fine_workspace != nullptr) {
                auto& row = results.mg_history.back();
                row.nonlinear_huber_objective=robust_line_search ? trial_obj : 0;
                row.basis_prepass_sweeps=basis_prepass!="none" ? pre_sweeps : 0;
                row.basis_prepass_sec=basis_prepass_sec;
                row.outer_pose_update_sec=outer_pose_update_sec;
                row.outer_acceptance_sec=outer_acceptance_sec;
                row.objective_evaluations=objective_evaluations;
                const auto& packed = *implicit_fine_workspace;
                row.precision_checks = packed.precision_checks;
                row.precision_freezes = packed.precision_freezes;
                row.precision_thaws = packed.precision_thaws;
                row.first_precision_freeze_sweep = packed.first_precision_freeze_sweep;
                row.precision_residual = packed.last_precision_residual;
                row.full_precision_sweeps = packed.full_precision_sweeps;
                row.eta_only_sweeps = packed.eta_only_sweeps;
                row.smoother_total_sweeps = packed.full_precision_sweeps + packed.eta_only_sweeps + packed.defect_mean_sweeps;
                row.defect_mean_sweeps=packed.defect_mean_sweeps;
                row.defect_map_builds=packed.defect_map_builds;
                row.defect_clamps=packed.defect_clamps;
                row.defect_build_sec=packed.defect_build_sec;
                row.defect_inverse_residual=packed.defect_inverse_residual;
                row.jacobi_sweeps=packed.jacobi_sweeps;
                row.coarse_solves=coarse_solves;
                row.cycle_message_rebuilds=packed.cycle_message_rebuilds;
                row.coarse_reuse_attempts=coarse_reuse_attempts;
                row.coarse_reuse_accepts=coarse_reuse_accepts;
                row.coarse_reuse_fallbacks=coarse_reuse_fallbacks;
                row.coarse_reuse_max_accepted_residual=coarse_reuse_max_accepted_residual;
                row.coarse_reuse_work_ratio=coarse_reuse_work_ratio;
                row.coarse_reuse_iterations=coarse_reuse_iterations;
                row.coarse_reuse_max_iterations=coarse_reuse_max_iterations;
                row.basis_exact_reuse_count=rebuild_basis ? basis.exact_reuse_count : 0;
                row.basis_exact_cache_sec=rebuild_basis ? basis.exact_cache_sec : 0.0;
                row.message_lifts=message_lifts;
                row.message_lift_sec=message_lift_sec;
                row.residual_stop_checks=residual_stop_checks;
                row.residual_stop_sec=residual_stop_sec;
                row.residual_stop_relative=std::isfinite(residual_stop_relative)?residual_stop_relative:-1;
                row.residual_stopped=residual_stopped;
                row.cycle_audit=std::move(cycle_audit);
                row.basis_partial_work_aborts=rebuild_basis ? basis.partial_work_aborts : 0;
                row.basis_partial_iterations=rebuild_basis ? basis.partial_iterations : 0;
                row.coarse_reuse_preconditioner_calls=coarse_reuse_preconditioner_calls;
                row.coarse_reuse_matvec_calls=coarse_reuse_matvec_calls;
                row.coarse_amortized_refresh=coarse_amortized_refresh;
                row.coarse_cholmod_supernodal=coarse_factor_ptr->ldlt_solver.supernodal();
                row.coarse_extra_work_before=coarse_extra_work_before;
                row.coarse_extra_work_after=coarse_extra_work;
                row.coarse_factor_work_estimate=coarse_work.factor_work;
                row.precision_warm_slots=precision_warm_stats.slots;
                row.precision_warm_projected=precision_warm_stats.projected;
                row.precision_warm_failed=precision_warm_stats.failed;
                row.precision_warm_max_projection=precision_warm_stats.max_projection_relative;
                row.precision_warm_sec=precision_warm_sec;
            }
            results.mg_pose_history.push_back(mg_poses);
        }
    }

    const int final_polish_steps = se3FinalDirectPolishSteps();
    if (final_polish_steps > 0 && !results.mg_history.empty()) {
        for (int polish = 0; polish < final_polish_steps; ++polish) {
            const auto total_t0 = SteadyClock::now();
            const auto build_t0 = SteadyClock::now();
            gbp::FactorGraph graph =
                buildLinearizedResidualGraph(problem, mg_poses, 1e-12, robust_loss_config);
            const auto build_t1 = SteadyClock::now();

            const auto joint_t0 = SteadyClock::now();
            gbp::FactorGraph::JointInfResult J = graph.jointDistributionInfSparse();
            const Eigen::SparseMatrix<double> lam = symmetrizeSparse(J.lam);
            const auto joint_t1 = SteadyClock::now();

            const auto solve_t0 = SteadyClock::now();
            const Eigen::VectorXd polish_step = solveSparseCholesky(lam, J.eta, 1e-10);
            const auto solve_t1 = SteadyClock::now();

            const auto apply_t0 = SteadyClock::now();
            const double base_obj = nonlinearObjective(problem, mg_poses);
            double step_scale = 1.0;
            int line_search_rejects = 0;
            double trial_obj = base_obj;
            SE3PoseVector trial_poses = mg_poses;
            while (step_scale >= (1.0 / 1024.0)) {
                trial_poses = applyPoseDeltas(mg_poses, step_scale * polish_step);
                trial_obj = nonlinearObjective(problem, trial_poses);
                if (std::isfinite(trial_obj) && trial_obj <= base_obj) {
                    break;
                }
                step_scale *= 0.5;
                ++line_search_rejects;
            }
            if (step_scale < (1.0 / 1024.0)) {
                trial_poses = mg_poses;
                trial_obj = base_obj;
                step_scale = 0.0;
            }
            mg_poses = trial_poses;
            const auto apply_t1 = SteadyClock::now();

            const auto obj_t0 = SteadyClock::now();
            const double nonlinear_obj = trial_obj;
            const auto obj_t1 = SteadyClock::now();
            const auto total_t1 = SteadyClock::now();

            SyntheticSE3OuterMGRow& row = results.mg_history.back();
            row.nonlinear_objective = nonlinear_obj;
            row.raw_e_hat_norm = polish_step.norm();
            row.e_hat_norm = (step_scale * polish_step).norm();
            row.linear_residual_approx = (J.eta - lam * (step_scale * polish_step)).norm();
            row.line_search_step_scale = step_scale;
            row.line_search_rejects += line_search_rejects;
            row.outer_total_sec += elapsedSeconds(total_t0, total_t1);
            row.build_graph_sec += elapsedSeconds(build_t0, build_t1);
            row.exact_joint_assembly_sec += elapsedSeconds(joint_t0, joint_t1);
            row.exact_solve_sec += elapsedSeconds(solve_t0, solve_t1);
            row.apply_step_sec += elapsedSeconds(apply_t0, apply_t1);
            row.objective_eval_sec += elapsedSeconds(obj_t0, obj_t1);
        }
        if (!results.mg_pose_history.empty()) {
            results.mg_pose_history.back() = mg_poses;
        } else {
            results.mg_pose_history.push_back(mg_poses);
        }
    }

    results.solver_wall_sec = elapsedSeconds(solver_wall_start, SteadyClock::now());
    return results;
}

#include "detail/se3_output.inc"

}  // namespace slam
