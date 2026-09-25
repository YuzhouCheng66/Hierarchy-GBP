#include "pgo_common.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include <omp.h>
#ifdef DIRECT_SE2
#include "se2_geometry.h"
using Geometry = reviewer_pcg::SE2Traits;
#else
#include "se3_geometry.h"
using Geometry = reviewer_pcg::SE3Traits;
#endif

namespace {
using namespace reviewer_pcg;
struct Record {
    int outer = 0;
    double raw = 0, huber = 0, step = 0, relative_residual = 0;
    double assembly = 0, solve = 0, residual = 0, update = 0, score = 0, total = 0;
    double symbolic = 0, numeric_and_backsolve = 0;
    int factor_nnz = 0;
};

int execute(int argc, char** argv) {
    // Same frontend/analytic geometry as the PCG audit. Only the linear solver
    // changes; the official g2o CHOLMOD class and DLL remain unmodified.
    bool block_ordering = true, fresh_symbolic = false;
    std::vector<char*> arguments{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--scalar-ordering") block_ordering = false;
        else if (option == "--fresh-symbolic") fresh_symbolic = true;
        else arguments.push_back(argv[i]);
    }
    const Args args = parseArgs(static_cast<int>(arguments.size()), arguments.data(), Geometry::geometryName());
    if (args.line_search || args.paper_gradient_stopping)
        throw std::runtime_error("This matched baseline requires all outers, without line search");
    Eigen::setNbThreads(1);
    omp_set_dynamic(0);
    omp_set_num_threads(1);
    constexpr int D = Geometry::kDimension;
    using Block = Eigen::Matrix<double, D, D>;
    const auto overall_begin = Clock::now();
    const auto input_begin = Clock::now();
    const auto problem = Geometry::load(args.input);
    const double input_seconds = elapsedSeconds(input_begin, Clock::now());
    const auto setup_begin = Clock::now();
    auto poses = problem.poses;
    BlockSystem<D> system(static_cast<int>(poses.size()), problem.fixed_index, problem.edges);
    const Objectives initial = evaluateObjectives<Geometry>(problem, poses, args.huber_delta);
    g2o::LinearSolverCholmod<Block> solver;
    solver.setBlockOrdering(block_ordering);
    solver.setWriteDebug(false);
    solver.init();
    const double setup_seconds = elapsedSeconds(setup_begin, Clock::now());
    std::vector<Record> records;
    records.reserve(args.num_outer);
    const auto solver_begin = Clock::now();
    for (int outer = 1; outer <= args.num_outer; ++outer) {
        Record row;
        row.outer = outer;
        const auto begin = Clock::now();
        auto t = Clock::now();
        assembleSystem<Geometry>(problem, poses, args.huber_delta, args.diagonal_jitter, system);
        row.assembly = elapsedSeconds(t, Clock::now());
        t = Clock::now();
        Eigen::VectorXd rhs = system.rhs();
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(rhs.size());
        if (fresh_symbolic) solver.init();
        g2o::G2OBatchStatistics stats{};
        g2o::G2OBatchStatistics::setGlobalStats(&stats);
        const bool ok = solver.solve(system.hessian(), delta.data(), rhs.data());
        g2o::G2OBatchStatistics::setGlobalStats(nullptr);
        row.solve = elapsedSeconds(t, Clock::now());
        row.symbolic = stats.timeSymbolicDecomposition;
        row.numeric_and_backsolve = stats.timeNumericDecomposition;
        row.factor_nnz = stats.choleskyNNZ;
        if (!ok || !delta.allFinite()) throw std::runtime_error("CHOLMOD failed or returned nonfinite step");
        t = Clock::now();
        const Eigen::VectorXd residual = system.rhs() - system.multiply(delta);
        row.relative_residual = residual.norm() / std::max(system.rhs().norm(), 1e-300);
        row.step = delta.norm();
        row.residual = elapsedSeconds(t, Clock::now());
        t = Clock::now();
        for (int pose = 0; pose < static_cast<int>(poses.size()); ++pose) {
            const int block = system.activeBlock(pose);
            if (block >= 0) poses[pose] = Geometry::plus(poses[pose], delta.template segment<D>(D * block));
        }
        row.update = elapsedSeconds(t, Clock::now());
        t = Clock::now();
        const Objectives objective = evaluateObjectives<Geometry>(problem, poses, args.huber_delta);
        row.raw = objective.raw;
        row.huber = objective.huber;
        row.score = elapsedSeconds(t, Clock::now());
        row.total = elapsedSeconds(begin, Clock::now());
        records.push_back(row);
        if (!args.quiet) std::cout << "outer=" << outer << " time=" << row.total
            << " cost=" << std::setprecision(17) << row.raw << " factor_nnz=" << row.factor_nnz << std::endl;
    }
    const double solve_seconds = elapsedSeconds(solver_begin, Clock::now());
    const double total_seconds = elapsedSeconds(overall_begin, Clock::now());
    ensureParentDirectory(args.output_json);
    ensureParentDirectory(args.output_poses);
    Geometry::writeG2O(problem, poses, args.output_poses);
    std::ofstream out(args.output_json);
    if (!out) throw std::runtime_error("Cannot open output JSON");
    out << std::setprecision(17);
    const auto final = records.back();
    out << "{\n  \"method\": \"official g2o LinearSolverCholmod, matched analytic Lie-log frontend\",\n"
        << "  \"configuration\": {\"threads\": 1, \"blas_threads_from_environment\": true, "
        << "\"early_outer_exit\": false, \"num_outer\": " << args.num_outer
        << ", \"huber_delta\": " << args.huber_delta << ", \"diagonal_jitter\": " << args.diagonal_jitter
        << ", \"block_ordering\": " << (block_ordering ? "true" : "false")
        << ", \"symbolic_reuse\": " << (fresh_symbolic ? "false" : "true")
        << ", \"cholmod_mode\": \"AUTO (official wrapper default, not forced supernodal)\", "
        << "\"gauge\": \"first FIX or smallest vertex ID hard-fixed\"},\n"
        << "  \"initial_objective\": {\"raw\": " << initial.raw << ", \"huber\": " << initial.huber << "},\n"
        << "  \"final_objective\": {\"raw\": " << final.raw << ", \"huber\": " << final.huber << "},\n"
        << "  \"timing\": {\"parse_sec\": " << input_seconds << ", \"setup_excluding_parse_sec\": " << setup_seconds
        << ", \"solve_sec\": " << solve_seconds << ", \"total_sec\": " << total_seconds << "},\n"
        << "  \"outer_iterations\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        out << "    {\"outer\": " << r.outer << ", \"raw_cost\": " << r.raw << ", \"huber_cost\": " << r.huber
            << ", \"step_norm\": " << r.step << ", \"relative_residual\": " << r.relative_residual
            << ", \"factor_nnz\": " << r.factor_nnz << ", \"timing\": {\"assembly_sec\": " << r.assembly
            << ", \"linear_solve_sec\": " << r.solve << ", \"symbolic_sec\": " << r.symbolic
            << ", \"numeric_and_backsolve_sec\": " << r.numeric_and_backsolve
            << ", \"residual_check_sec\": " << r.residual << ", \"update_sec\": " << r.update
            << ", \"objective_sec\": " << r.score << ", \"total_sec\": " << r.total << "}}"
            << (i + 1 == records.size() ? "\n" : ",\n");
    }
    out << "  ]\n}\n";
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return execute(argc, argv); }
    catch (const std::exception& error) {
        g2o::G2OBatchStatistics::setGlobalStats(nullptr);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
