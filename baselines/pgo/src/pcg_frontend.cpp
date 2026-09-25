#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <omp.h>

#include "g2o/config.h"
#include "g2o/core/batch_stats.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/core/optimization_algorithm_factory.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "se2_geometry.h"
#include "se3_geometry.h"

G2O_USE_OPTIMIZATION_LIBRARY(pcg);

namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::string number(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

template<class Traits>
class PoseVertex final : public g2o::BaseVertex<Traits::kDimension, typename Traits::Pose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void setToOriginImpl() override {
        if constexpr (Traits::kDimension == 3) this->_estimate.setZero();
        else this->_estimate = typename Traits::Pose{};
    }
    void oplusImpl(const double* update) override {
        this->_estimate = Traits::plus(this->_estimate, Eigen::Map<const typename Traits::Vector>(update));
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

template<class Traits>
class PoseEdge final : public g2o::BaseBinaryEdge<Traits::kDimension, typename Traits::Pose,
                                               PoseVertex<Traits>, PoseVertex<Traits>> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    void computeError() override {
        const auto& a = static_cast<const PoseVertex<Traits>*>(this->_vertices[0])->estimate();
        const auto& b = static_cast<const PoseVertex<Traits>*>(this->_vertices[1])->estimate();
        this->_error = Traits::residual(a, b, this->_measurement);
    }
    void linearizeOplus() override {
        const auto& a = static_cast<const PoseVertex<Traits>*>(this->_vertices[0])->estimate();
        const auto& b = static_cast<const PoseVertex<Traits>*>(this->_vertices[1])->estimate();
        typename Traits::JacobianPair jacobian;
        Traits::linearize(a, b, this->_measurement, this->_error, jacobian);
        this->_jacobianOplusXi = jacobian.template leftCols<Traits::kDimension>();
        this->_jacobianOplusXj = jacobian.template rightCols<Traits::kDimension>();
    }
    bool read(std::istream&) override { return false; }
    bool write(std::ostream&) const override { return false; }
};

template<class Traits>
int execute(const std::string& input, const std::string& output, int threads, int outers,
            bool historical_relative) {
#ifndef G2O_OPENMP
    throw std::runtime_error("This baseline requires the official OpenMP build");
#endif
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    g2o::OptimizableGraph::initMultiThreading();
    int observed_team = 0;
#pragma omp parallel
    {
#pragma omp single
        observed_team = omp_get_num_threads();
    }
    if (observed_team != threads) throw std::runtime_error("OpenMP team differs from request");

    const auto process_begin = Clock::now();
    g2o::SparseOptimizer optimizer;
    auto begin = Clock::now();
    const auto problem = Traits::load(input);
    const double parse_sec = seconds(begin);
    begin = Clock::now();
    const char* solver_name = Traits::kDimension == 3 ? "gn_pcg3_2" : "gn_pcg6_3";
    g2o::OptimizationAlgorithmProperty property;
    g2o::OptimizationAlgorithm* algorithm = nullptr;
    if (historical_relative) {
        constexpr int D = Traits::kDimension;
        using BlockSolver = g2o::BlockSolverPL<D, D == 3 ? 2 : 3>;
        auto linear = std::make_unique<g2o::LinearSolverPCG<typename BlockSolver::PoseMatrixType>>();
        // Exact stopping settings from the retained July matched-PCG baseline.
        linear->setTolerance(1e-6);
        linear->setMaxIterations(-1);
        linear->setAbsoluteTolerance(false);
        algorithm = new g2o::OptimizationAlgorithmGaussNewton(
            std::make_unique<BlockSolver>(std::move(linear)));
    } else {
        algorithm = g2o::OptimizationAlgorithmFactory::instance()->construct(solver_name, property);
    }
    if (!algorithm) throw std::runtime_error("Official PCG solver registration failed");
    optimizer.setAlgorithm(algorithm);
    optimizer.setVerbose(false);
    optimizer.setComputeBatchStatistics(true);
    if (problem.poses.empty()) throw std::runtime_error("Empty pose graph");
    std::vector<PoseVertex<Traits>*> vertices;
    vertices.reserve(problem.poses.size());
    for (size_t i = 0; i < problem.poses.size(); ++i) {
        auto* vertex = new PoseVertex<Traits>;
        vertex->setId(problem.original_ids[i]);
        vertex->setEstimate(problem.poses[i]);
        vertex->setFixed(static_cast<int>(i) == problem.fixed_index);
        if (!optimizer.addVertex(vertex)) throw std::runtime_error("Duplicate vertex");
        vertices.push_back(vertex);
    }
    for (size_t i = 0; i < problem.edges.size(); ++i) {
        const auto& original = problem.edges[i];
        auto* edge = new PoseEdge<Traits>;
        edge->setId(static_cast<int>(i));
        edge->setVertex(0, vertices[original.i]);
        edge->setVertex(1, vertices[original.j]);
        edge->setMeasurement(original.measurement);
        edge->setInformation(original.information);
        auto* kernel = new g2o::RobustKernelHuber;
        kernel->setDelta(5.);
        edge->setRobustKernel(kernel);
        if (!optimizer.addEdge(edge)) throw std::runtime_error("Invalid edge");
    }
    if (!optimizer.initializeOptimization()) throw std::runtime_error("Optimization initialization failed");
    optimizer.computeActiveErrors();
    const double initial_raw = .5 * optimizer.activeChi2();
    const double initial_huber = .5 * optimizer.activeRobustChi2();
    const double setup_sec = seconds(begin);

    begin = Clock::now();
    const int returned = optimizer.optimize(outers);
    optimizer.computeActiveErrors();
    const double final_raw = .5 * optimizer.activeChi2();
    const double final_huber = .5 * optimizer.activeRobustChi2();
    const double solve_sec = seconds(begin);
    const double total_before_output_sec = seconds(process_begin);
    const auto& stats = optimizer.batchStatistics();
    auto poses = problem.poses;
    for (size_t i = 0; i < vertices.size(); ++i) poses[i] = vertices[i]->estimate();
    Traits::writeG2O(problem, poses, output + ".g2o");
    std::ofstream out(output);
    if (!out) throw std::runtime_error("Cannot write result JSON");
    out << std::setprecision(17)
        << "{\n\"method\":\"official g2o OpenMP optimizer / PCG, matched analytic Lie-log frontend\",\n"
        << "\"configuration\":{\"requested_threads\":" << threads
        << ",\"observed_openmp_team\":" << observed_team
        << ",\"g2o_openmp\":true,\"pcg_kernel_parallel\":false,\"num_outer\":" << outers
        << ",\"huber_delta\":5,\"fixed_vertices\":1,\"diagonal_jitter\":0"
        << ",\"solver\":\"" << solver_name
        << "\",\"pcg_tolerance\":1e-6,\"pcg_max_iterations\":-1,\"pcg_absolute_tolerance\":"
        << (historical_relative ? "false" : "true") << "},\n"
        << "\"problem\":{\"vertices\":" << optimizer.vertices().size()
        << ",\"edges\":" << optimizer.edges().size() << "},\n"
        << "\"returned_iterations\":" << returned
        << ",\"native_objective\":{\"initial_raw\":" << number(initial_raw)
        << ",\"initial_huber\":" << number(initial_huber)
        << ",\"final_raw\":" << number(final_raw)
        << ",\"final_huber\":" << number(final_huber) << "},\n"
        << "\"timing\":{\"parse_sec\":" << parse_sec
        << ",\"setup_excluding_parse_sec\":" << setup_sec
        << ",\"solve_sec\":" << solve_sec << ",\"solver_wall_sec\":" << setup_sec + solve_sec
        << ",\"total_before_output_sec\":" << total_before_output_sec << "},\n\"iterations\":[";
    const int count = std::max(0, std::min(returned, static_cast<int>(stats.size())));
    for (int i = 0; i < count; ++i) {
        const auto& s = stats[i];
        if (i) out << ',';
        out << "{\"iteration\":" << i + 1 << ",\"iteration_sec\":" << number(s.timeIteration)
            << ",\"quadratic_form_sec\":" << number(s.timeQuadraticForm)
            << ",\"linear_solution_sec\":" << number(s.timeLinearSolution)
            << ",\"linear_iterations\":" << s.iterationsLinearSolver << '}';
    }
    out << "]\n}\n";
    std::cout << "solver_wall_sec=" << setup_sec + solve_sec << " returned=" << returned
              << " native_half_chi2=" << number(final_raw) << " openmp_team=" << observed_team << '\n';
    return returned == outers && std::isfinite(final_raw) && std::isfinite(final_huber) ? 0 : 2;
}

int run(int argc, char** argv) {
    std::string input, output;
    std::string stopping = "default";
    int dimension = 0, threads = 16, outers = 20;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (i + 1 == argc) throw std::runtime_error("Missing argument value");
        const std::string value = argv[++i];
        if (key == "--input") input = value;
        else if (key == "--output-json") output = value;
        else if (key == "--dimension") dimension = std::stoi(value);
        else if (key == "--threads") threads = std::stoi(value);
        else if (key == "--num-outer") outers = std::stoi(value);
        else if (key == "--stopping") stopping = value;
        else throw std::runtime_error("Unknown argument: " + key);
    }
    if (input.empty() || output.empty() || (dimension != 2 && dimension != 3)
        || threads < 1 || outers < 1 || (stopping != "default" && stopping != "historical-relative"))
        throw std::runtime_error("Invalid benchmark arguments");
    return dimension == 2 ? execute<reviewer_pcg::SE2Traits>(input, output, threads, outers, stopping=="historical-relative")
                          : execute<reviewer_pcg::SE3Traits>(input, output, threads, outers, stopping=="historical-relative");
}
}

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
