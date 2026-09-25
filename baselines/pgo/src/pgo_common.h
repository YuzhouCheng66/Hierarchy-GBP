#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCore>

#include "g2o/core/batch_stats.h"
#include "g2o/core/sparse_block_matrix.h"
#include "g2o/stuff/misc.h"
#include "g2o/solvers/pcg/linear_solver_pcg.h"
#include "two_level_schwarz.h"

namespace reviewer_pcg {

using Clock = std::chrono::steady_clock;

inline double elapsedSeconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double>(end - begin).count();
}

struct Args {
    std::string input;
    std::string output_json;
    std::string output_poses;
    int num_outer = 20;
    int pcg_max_iterations = 1000;
    double pcg_tolerance = 1e-6;
    std::string pcg_preconditioner = "block_jacobi";
    double ic_initial_shift = 1e-6;
    int schwarz_subdomains = 16;
    int schwarz_overlap_layers = 1;
    int schwarz_threads = 16;
    double schwarz_local_shift = 1e-10;
    bool schwarz_enable_coarse = true;
    bool schwarz_include_rotation_modes = true;
    bool paper_gradient_stopping = false;
    double gradient_absolute_tolerance = 1e-8;
    double gradient_relative_tolerance = 1e-6;
    double huber_delta = 5.0;
    double diagonal_jitter = 1e-10;
    double outer_step_tolerance = 1e-8;
    double outer_objective_tolerance = 1e-9;
    int line_search_max_trials = 12;
    double line_search_shrink = 0.5;
    bool line_search = false;
    bool quiet = false;
};

inline int parseInt(const std::string& text, const char* name) {
    try {
        size_t consumed = 0;
        const int value = std::stoi(text, &consumed);
        if (consumed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid integer for ") + name + ": " + text);
    }
}

inline double parseDouble(const std::string& text, const char* name) {
    try {
        size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != text.size() || !std::isfinite(value)) {
            throw std::invalid_argument("invalid floating-point value");
        }
        return value;
    } catch (const std::exception&) {
        throw std::runtime_error(std::string("invalid number for ") + name + ": " + text);
    }
}

inline std::string defaultPosePath(const std::string& json_path) {
    std::filesystem::path path(json_path);
    path.replace_extension(".g2o");
    return path.string();
}

inline Args parseArgs(int argc, char** argv, const std::string& geometry_name) {
    Args args;
    args.output_json = "g2o_block_pcg_" + geometry_name + ".json";
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto requireValue = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            return argv[++i];
        };

        if (option == "--input" || option == "--problem-file") {
            args.input = requireValue(option.c_str());
        } else if (option == "--output-json" || option == "--out-json") {
            args.output_json = requireValue(option.c_str());
        } else if (option == "--output-poses" || option == "--out-poses") {
            args.output_poses = requireValue(option.c_str());
        } else if (option == "--num-outer") {
            args.num_outer = parseInt(requireValue("--num-outer"), "--num-outer");
        } else if (option == "--pcg-max-iterations") {
            args.pcg_max_iterations =
                parseInt(requireValue("--pcg-max-iterations"), "--pcg-max-iterations");
        } else if (option == "--pcg-tolerance") {
            args.pcg_tolerance =
                parseDouble(requireValue("--pcg-tolerance"), "--pcg-tolerance");
        } else if (option == "--pcg-preconditioner") {
            args.pcg_preconditioner = requireValue("--pcg-preconditioner");
        } else if (option == "--ic-initial-shift") {
            args.ic_initial_shift =
                parseDouble(requireValue("--ic-initial-shift"), "--ic-initial-shift");
        } else if (option == "--schwarz-subdomains") {
            args.schwarz_subdomains = parseInt(
                requireValue("--schwarz-subdomains"),
                "--schwarz-subdomains"
            );
        } else if (option == "--schwarz-overlap-layers") {
            args.schwarz_overlap_layers = parseInt(
                requireValue("--schwarz-overlap-layers"),
                "--schwarz-overlap-layers"
            );
        } else if (option == "--schwarz-threads") {
            args.schwarz_threads = parseInt(
                requireValue("--schwarz-threads"),
                "--schwarz-threads"
            );
        } else if (option == "--schwarz-local-shift") {
            args.schwarz_local_shift = parseDouble(
                requireValue("--schwarz-local-shift"),
                "--schwarz-local-shift"
            );
        } else if (option == "--schwarz-one-level") {
            args.schwarz_enable_coarse = false;
        } else if (option == "--schwarz-no-rotation-mode") {
            args.schwarz_include_rotation_modes = false;
        } else if (option == "--paper-gradient-stopping") {
            args.paper_gradient_stopping = true;
        } else if (option == "--gradient-absolute-tolerance") {
            args.gradient_absolute_tolerance = parseDouble(
                requireValue("--gradient-absolute-tolerance"),
                "--gradient-absolute-tolerance"
            );
        } else if (option == "--gradient-relative-tolerance") {
            args.gradient_relative_tolerance = parseDouble(
                requireValue("--gradient-relative-tolerance"),
                "--gradient-relative-tolerance"
            );
        } else if (option == "--huber-delta") {
            args.huber_delta = parseDouble(requireValue("--huber-delta"), "--huber-delta");
        } else if (option == "--diagonal-jitter") {
            args.diagonal_jitter =
                parseDouble(requireValue("--diagonal-jitter"), "--diagonal-jitter");
        } else if (option == "--outer-step-tolerance") {
            args.outer_step_tolerance =
                parseDouble(requireValue("--outer-step-tolerance"), "--outer-step-tolerance");
        } else if (option == "--outer-objective-tolerance") {
            args.outer_objective_tolerance = parseDouble(
                requireValue("--outer-objective-tolerance"),
                "--outer-objective-tolerance"
            );
        } else if (option == "--line-search") {
            args.line_search = true;
        } else if (option == "--line-search-max-trials") {
            args.line_search_max_trials = parseInt(
                requireValue("--line-search-max-trials"),
                "--line-search-max-trials"
            );
        } else if (option == "--line-search-shrink") {
            args.line_search_shrink = parseDouble(
                requireValue("--line-search-shrink"),
                "--line-search-shrink"
            );
        } else if (option == "--quiet") {
            args.quiet = true;
        } else if (option == "--help" || option == "-h") {
            std::cout
                << "Usage: g2o_block_pcg_" << geometry_name
                << " --input graph.g2o [--output-json result.json]"
                   " [--output-poses final.g2o] [--num-outer 20]"
                   " [--huber-delta 5] [--pcg-max-iterations 1000|-1]"
                   " [--pcg-tolerance 1e-6] [--diagonal-jitter 1e-10]"
                   " [--pcg-preconditioner block_jacobi|incomplete_cholesky|"
                   "two_level_schwarz]"
                   " [--ic-initial-shift 1e-6]"
                   " [--schwarz-subdomains 16] [--schwarz-overlap-layers 1]"
                   " [--schwarz-threads 16] [--schwarz-local-shift 1e-10]"
                   " [--schwarz-one-level] [--schwarz-no-rotation-mode]"
                   " [--paper-gradient-stopping]"
                   " [--gradient-absolute-tolerance 1e-8]"
                   " [--gradient-relative-tolerance 1e-6]"
                   " [--line-search] [--line-search-max-trials 12]"
                   " [--line-search-shrink 0.5]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + option);
        }
    }

    if (args.input.empty()) {
        throw std::runtime_error("--input is required");
    }
    if (args.output_poses.empty()) {
        args.output_poses = defaultPosePath(args.output_json);
    }
    if (args.num_outer <= 0 ||
        (args.pcg_max_iterations == 0 || args.pcg_max_iterations < -1) ||
        args.line_search_max_trials <= 0) {
        throw std::runtime_error(
            "iteration counts must be positive; PCG max iterations may be -1 "
            "to use the g2o default (number of scalar rows)"
        );
    }
    if (!(args.pcg_tolerance > 0.0) || !(args.huber_delta > 0.0) ||
        !(args.gradient_absolute_tolerance > 0.0) ||
        !(args.gradient_relative_tolerance > 0.0) ||
        args.diagonal_jitter < 0.0 || args.ic_initial_shift < 0.0 ||
        args.schwarz_subdomains <= 0 || args.schwarz_overlap_layers < 0 ||
        args.schwarz_threads <= 0 || args.schwarz_local_shift < 0.0 ||
        !(args.line_search_shrink > 0.0) ||
        !(args.line_search_shrink < 1.0)) {
        throw std::runtime_error(
            "PCG tolerance and Huber delta must be positive; jitter must be "
            "nonnegative; line-search shrink must be in (0, 1)"
        );
    }
    if (args.pcg_preconditioner != "block_jacobi" &&
        args.pcg_preconditioner != "incomplete_cholesky" &&
        args.pcg_preconditioner != "two_level_schwarz") {
        throw std::runtime_error(
            "--pcg-preconditioner must be block_jacobi, incomplete_cholesky, "
            "or two_level_schwarz"
        );
    }
    return args;
}

inline std::string jsonEscape(const std::string& text) {
    std::ostringstream out;
    for (const char c : text) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default: out << c; break;
        }
    }
    return out.str();
}

inline void ensureParentDirectory(const std::string& path) {
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

struct Objectives {
    double raw = 0.0;
    double huber = 0.0;
};

inline double huberRho(double squared_mahalanobis, double delta) {
    const double s = std::max(0.0, squared_mahalanobis);
    const double delta2 = delta * delta;
    if (s <= delta2) {
        return s;
    }
    return 2.0 * delta * std::sqrt(s) - delta2;
}

inline double huberIrlsWeight(double squared_mahalanobis, double delta) {
    const double s = std::max(0.0, squared_mahalanobis);
    const double delta2 = delta * delta;
    if (s <= delta2 || s == 0.0) {
        return 1.0;
    }
    return delta / std::sqrt(s);
}

struct OuterRecord {
    int outer = 0;
    int linear_iterations = 0;
    int linear_max_iterations = 0;
    bool linear_converged = false;
    bool true_residual_converged = false;
    bool outer_converged = false;
    double raw_objective = 0.0;
    double huber_objective = 0.0;
    double relative_huber_change = 0.0;
    double proposed_step_norm = 0.0;
    double step_norm = 0.0;
    double step_scale = 1.0;
    int line_search_trials = 0;
    bool step_accepted = true;
    double rhs_norm = 0.0;
    double residual_norm = 0.0;
    double relative_residual_norm = 0.0;
    double initial_preconditioned_residual = 0.0;
    double final_preconditioned_residual = 0.0;
    double preconditioned_residual_ratio = 0.0;
    double assembly_sec = 0.0;
    double preconditioner_sec = 0.0;
    double pcg_sec = 0.0;
    double residual_check_sec = 0.0;
    double update_sec = 0.0;
    double objective_sec = 0.0;
    double total_sec = 0.0;
};

template <int D>
class BlockSystem {
public:
    using Block = Eigen::Matrix<double, D, D, Eigen::ColMajor>;
    using Matrix = g2o::SparseBlockMatrix<Block>;

    template <typename EdgeContainer>
    BlockSystem(int pose_count, int fixed_pose, const EdgeContainer& edges)
        : active_block_by_pose_(static_cast<size_t>(pose_count), -1) {
        if (pose_count < 2 || fixed_pose < 0 || fixed_pose >= pose_count) {
            throw std::runtime_error("pose graph needs at least two poses and one valid fixed pose");
        }

        int active_count = 0;
        for (int pose = 0; pose < pose_count; ++pose) {
            if (pose != fixed_pose) {
                active_block_by_pose_[static_cast<size_t>(pose)] = active_count++;
            }
        }
        block_indices_.resize(static_cast<size_t>(active_count));
        for (int block = 0; block < active_count; ++block) {
            block_indices_[static_cast<size_t>(block)] = D * (block + 1);
        }
        hessian_ = std::make_unique<Matrix>(
            block_indices_.data(),
            block_indices_.data(),
            active_count,
            active_count,
            true
        );
        rhs_ = Eigen::VectorXd::Zero(D * active_count);

        for (int block = 0; block < active_count; ++block) {
            hessian_->block(block, block, true)->setZero();
        }
        for (const auto& edge : edges) {
            const int bi = activeBlock(edge.i);
            const int bj = activeBlock(edge.j);
            if (bi >= 0 && bj >= 0 && bi != bj) {
                const int row = std::min(bi, bj);
                const int col = std::max(bi, bj);
                hessian_->block(row, col, true)->setZero();
            }
        }
    }

    int activeBlock(int pose) const {
        return active_block_by_pose_.at(static_cast<size_t>(pose));
    }

    const std::vector<int>& activeBlockByPose() const {
        return active_block_by_pose_;
    }

    Matrix& hessian() {
        return *hessian_;
    }

    const Matrix& hessian() const {
        return *hessian_;
    }

    Eigen::VectorXd& rhs() {
        return rhs_;
    }

    void reset(double diagonal_jitter) {
        for (auto& column : hessian_->blockCols()) {
            for (auto& entry : column) {
                entry.second->setZero();
            }
        }
        rhs_.setZero();
        if (diagonal_jitter > 0.0) {
            for (int block = 0; block < static_cast<int>(block_indices_.size()); ++block) {
                hessian_->block(block, block, false)->diagonal().array() += diagonal_jitter;
            }
        }
    }

    void addDiagonal(int block, const Block& value) {
        if (block >= 0) {
            *hessian_->block(block, block, false) += value;
        }
    }

    void addOffDiagonal(int row_block, int col_block, const Block& value) {
        if (row_block < 0 || col_block < 0 || row_block == col_block) {
            return;
        }
        if (row_block < col_block) {
            *hessian_->block(row_block, col_block, false) += value;
        } else {
            *hessian_->block(col_block, row_block, false) += value.transpose();
        }
    }

    void addRhs(int block, const Eigen::Matrix<double, D, 1>& value) {
        if (block >= 0) {
            rhs_.template segment<D>(D * block) += value;
        }
    }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(x.size());
        double* output = result.data();
        hessian_->multiplySymmetricUpperTriangle(output, x.data());
        return result;
    }

    Eigen::SparseMatrix<double> scalarSparseMatrix() const {
        using Triplet = Eigen::Triplet<double>;
        std::vector<Triplet> triplets;
        size_t block_count = 0;
        for (const auto& column : hessian_->blockCols()) {
            block_count += column.size();
        }
        triplets.reserve(block_count * D * D * 2);

        for (int column = 0;
             column < static_cast<int>(hessian_->blockCols().size());
             ++column) {
            for (const auto& entry : hessian_->blockCols()[static_cast<size_t>(column)]) {
                const int row = entry.first;
                const Block& block = *entry.second;
                for (int block_row = 0; block_row < D; ++block_row) {
                    for (int block_column = 0; block_column < D; ++block_column) {
                        const double value = block(block_row, block_column);
                        triplets.emplace_back(
                            D * row + block_row,
                            D * column + block_column,
                            value
                        );
                        if (row != column) {
                            triplets.emplace_back(
                                D * column + block_column,
                                D * row + block_row,
                                value
                            );
                        }
                    }
                }
            }
        }

        Eigen::SparseMatrix<double> matrix(rhs_.size(), rhs_.size());
        matrix.setFromTriplets(triplets.begin(), triplets.end());
        matrix.makeCompressed();
        return matrix;
    }

    double preconditionedResidualEnergy(const Eigen::VectorXd& residual) const {
        double energy = 0.0;
        for (int block = 0; block < static_cast<int>(block_indices_.size()); ++block) {
            const Block* diagonal = hessian_->block(block, block);
            const Eigen::Matrix<double, D, 1> r = residual.template segment<D>(D * block);
            Eigen::LDLT<Block> ldlt(0.5 * (*diagonal + diagonal->transpose()));
            if (ldlt.info() != Eigen::Success) {
                return std::numeric_limits<double>::infinity();
            }
            const Eigen::Matrix<double, D, 1> z = ldlt.solve(r);
            if (ldlt.info() != Eigen::Success || !z.allFinite()) {
                return std::numeric_limits<double>::infinity();
            }
            energy += r.dot(z);
        }
        return std::max(0.0, energy);
    }

private:
    std::vector<int> active_block_by_pose_;
    std::vector<int> block_indices_;
    std::unique_ptr<Matrix> hessian_;
    Eigen::VectorXd rhs_;
};

template <typename Traits>
Objectives evaluateObjectives(
    const typename Traits::Problem& problem,
    const typename Traits::PoseVector& poses,
    double huber_delta
) {
    Objectives result;
    for (const auto& edge : problem.edges) {
        const typename Traits::Vector residual = Traits::residual(
            poses.at(static_cast<size_t>(edge.i)),
            poses.at(static_cast<size_t>(edge.j)),
            edge.measurement
        );
        const double squared = std::max(
            0.0,
            (residual.transpose() * edge.information * residual)(0, 0)
        );
        result.raw += 0.5 * squared;
        result.huber += 0.5 * huberRho(squared, huber_delta);
    }
    return result;
}

template <typename Traits>
void assembleSystem(
    const typename Traits::Problem& problem,
    const typename Traits::PoseVector& poses,
    double huber_delta,
    double diagonal_jitter,
    BlockSystem<Traits::kDimension>& system
) {
    constexpr int D = Traits::kDimension;
    using Block = Eigen::Matrix<double, D, D>;
    using Vector = Eigen::Matrix<double, D, 1>;
    using JacobianPair = Eigen::Matrix<double, D, 2 * D>;

    system.reset(diagonal_jitter);
    for (const auto& edge : problem.edges) {
        Vector residual;
        JacobianPair jacobian;
        Traits::linearize(
            poses.at(static_cast<size_t>(edge.i)),
            poses.at(static_cast<size_t>(edge.j)),
            edge.measurement,
            residual,
            jacobian
        );
        const double squared = std::max(
            0.0,
            (residual.transpose() * edge.information * residual)(0, 0)
        );
        const double weight = huberIrlsWeight(squared, huber_delta);
        const Block weighted_information = weight * edge.information;
        const Block ji = jacobian.template leftCols<D>();
        const Block jj = jacobian.template rightCols<D>();
        const int bi = system.activeBlock(edge.i);
        const int bj = system.activeBlock(edge.j);

        if (bi >= 0) {
            system.addDiagonal(bi, ji.transpose() * weighted_information * ji);
            system.addRhs(bi, -ji.transpose() * weighted_information * residual);
        }
        if (bj >= 0) {
            system.addDiagonal(bj, jj.transpose() * weighted_information * jj);
            system.addRhs(bj, -jj.transpose() * weighted_information * residual);
        }
        if (bi >= 0 && bj >= 0) {
            system.addOffDiagonal(bi, bj, ji.transpose() * weighted_information * jj);
        }
    }
}

template <typename Traits>
void writeJson(
    const Args& args,
    const typename Traits::Problem& problem,
    const typename Traits::PoseVector& poses,
    const TwoLevelSchwarz<Traits::kDimension>* schwarz,
    const Objectives& initial_objective,
    const Objectives& final_objective,
    const std::vector<OuterRecord>& records,
    double setup_sec,
    double solve_sec,
    double total_sec,
    bool paper_gradient_stopped,
    int gradient_stop_check_outer,
    double initial_gradient_norm,
    double stopping_gradient_norm
) {
    ensureParentDirectory(args.output_json);
    std::ofstream out(args.output_json);
    if (!out) {
        throw std::runtime_error("failed to open JSON output: " + args.output_json);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"method\": \""
        << (args.pcg_preconditioner == "block_jacobi"
                ? "g2o_block_jacobi_pcg"
                : (args.pcg_preconditioner == "incomplete_cholesky"
                       ? "eigen_incomplete_cholesky_pcg"
                       : "two_level_additive_schwarz_pcg"))
        << "\",\n";
    out << "  \"geometry\": \"" << Traits::geometryName() << "\",\n";
    out << "  \"input\": \"" << jsonEscape(args.input) << "\",\n";
    out << "  \"output_poses\": \"" << jsonEscape(args.output_poses) << "\",\n";
    out << "  \"num_poses\": " << problem.poses.size() << ",\n";
    out << "  \"num_edges\": " << problem.edges.size() << ",\n";
    out << "  \"fixed_vertex_id\": "
        << problem.original_ids.at(static_cast<size_t>(problem.fixed_index)) << ",\n";
    out << "  \"configuration\": {\n";
    out << "    \"num_outer\": " << args.num_outer << ",\n";
    out << "    \"huber_delta\": " << args.huber_delta << ",\n";
    out << "    \"pcg_max_iterations\": " << args.pcg_max_iterations << ",\n";
    out << "    \"pcg_max_iterations_semantics\": \""
        << (args.pcg_max_iterations < 0
                ? "g2o default: number of scalar rows"
                : "fixed cap")
        << "\",\n";
    out << "    \"pcg_tolerance\": " << args.pcg_tolerance << ",\n";
    out << "    \"pcg_tolerance_semantics\": \""
        << (args.pcg_preconditioner == "block_jacobi" ||
                    args.pcg_preconditioner == "two_level_schwarz"
                ? (args.pcg_preconditioner == "two_level_schwarz"
                       ? "relative preconditioned residual norm: "
                         "sqrt(rMz / initial_rMz) <= tolerance"
                       : "g2o preconditioned residual energy: "
                         "rMz <= tolerance * initial_rMz")
                : "Eigen relative residual 2-norm")
        << "\",\n";
    out << "    \"preconditioner\": \""
        << (args.pcg_preconditioner == "block_jacobi"
                ? "block_jacobi_" + std::to_string(Traits::kDimension) + "x" +
                      std::to_string(Traits::kDimension)
                : (args.pcg_preconditioner == "incomplete_cholesky"
                       ? "incomplete_cholesky"
                       : "two_level_additive_schwarz"))
        << "\",\n";
    out << "    \"ic_initial_shift\": " << args.ic_initial_shift << ",\n";
    out << "    \"schwarz_subdomains\": " << args.schwarz_subdomains << ",\n";
    out << "    \"schwarz_overlap_layers\": "
        << args.schwarz_overlap_layers << ",\n";
    out << "    \"schwarz_threads\": " << args.schwarz_threads << ",\n";
    out << "    \"schwarz_local_shift\": " << args.schwarz_local_shift << ",\n";
    out << "    \"schwarz_coarse_space\": \""
        << (args.schwarz_enable_coarse
                ? "GDSW interface coordinate modes with discrete harmonic extension"
                : "disabled (one-level additive Schwarz)")
        << "\",\n";
    out << "    \"schwarz_rotation_modes\": "
        << (args.schwarz_include_rotation_modes ? "true" : "false")
        << ",\n";
    if (schwarz != nullptr) {
        out << "    \"schwarz_actual_subdomains\": "
            << schwarz->subdomains() << ",\n";
        out << "    \"schwarz_interface_blocks\": "
            << schwarz->interfaceBlocks() << ",\n";
        out << "    \"schwarz_coarse_dimension\": "
            << schwarz->coarseDimension() << ",\n";
        out << "    \"schwarz_basis_nonzeros\": "
            << schwarz->basisNonZeros() << ",\n";
        out << "    \"schwarz_total_local_blocks\": "
            << schwarz->totalLocalBlocks() << ",\n";
    }
    out << "    \"threads\": "
        << (args.pcg_preconditioner == "two_level_schwarz"
                ? args.schwarz_threads
                : 1)
        << ",\n";
    out << "    \"threading_note\": "
        << (args.pcg_preconditioner == "two_level_schwarz"
                ? "\"independent Schwarz local factorizations and solves use OpenMP\""
                : "\"official g2o LinearSolverPCG path is single-threaded\"")
        << ",\n";
    out << "    \"diagonal_jitter\": " << args.diagonal_jitter << ",\n";
    out << "    \"line_search\": " << (args.line_search ? "true" : "false") << ",\n";
    out << "    \"line_search_max_trials\": " << args.line_search_max_trials << ",\n";
    out << "    \"line_search_shrink\": " << args.line_search_shrink << ",\n";
    out << "    \"outer_step_tolerance\": " << args.outer_step_tolerance << ",\n";
    out << "    \"outer_objective_tolerance\": " << args.outer_objective_tolerance << ",\n";
    out << "    \"early_outer_exit\": "
        << (args.paper_gradient_stopping ? "true" : "false") << ",\n";
    out << "    \"paper_gradient_stopping\": "
        << (args.paper_gradient_stopping ? "true" : "false") << ",\n";
    out << "    \"gradient_absolute_tolerance\": "
        << args.gradient_absolute_tolerance << ",\n";
    out << "    \"gradient_relative_tolerance\": "
        << args.gradient_relative_tolerance << ",\n";
    out << "    \"linearization\": \"analytic_right_perturbation\",\n";
    out << "    \"objective_convention\": "
           "\"0.5 * sum chi2; Huber is 0.5 * sum rho_delta(chi2)\"\n";
    out << "  },\n";
    out << "  \"timing\": {\n";
    out << "    \"setup_sec\": " << setup_sec << ",\n";
    out << "    \"solve_sec\": " << solve_sec << ",\n";
    out << "    \"total_sec\": " << total_sec << ",\n";
    out << "    \"scope\": \"setup=parse+topology+initial objective; "
           "solve=all outer iterations; output serialization excluded\"\n";
    out << "  },\n";
    out << "  \"initial_objective\": {\"raw\": " << initial_objective.raw
        << ", \"huber\": " << initial_objective.huber << "},\n";
    out << "  \"final_objective\": {\"raw\": " << final_objective.raw
        << ", \"huber\": " << final_objective.huber << "},\n";
    out << "  \"outer_stopping\": {\n";
    out << "    \"paper_gradient_stopped\": "
        << (paper_gradient_stopped ? "true" : "false") << ",\n";
    out << "    \"stop_check_outer\": " << gradient_stop_check_outer << ",\n";
    out << "    \"initial_gradient_norm\": " << initial_gradient_norm << ",\n";
    out << "    \"stopping_gradient_norm\": "
        << stopping_gradient_norm << "\n";
    out << "  },\n";
    out << "  \"outer_iterations\": [\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const OuterRecord& row = records[i];
        out << "    {\n";
        out << "      \"outer\": " << row.outer << ",\n";
        out << "      \"raw_objective\": " << row.raw_objective << ",\n";
        out << "      \"huber_objective\": " << row.huber_objective << ",\n";
        out << "      \"relative_huber_change\": " << row.relative_huber_change << ",\n";
        out << "      \"proposed_step_norm\": " << row.proposed_step_norm << ",\n";
        out << "      \"step_norm\": " << row.step_norm << ",\n";
        out << "      \"step_scale\": " << row.step_scale << ",\n";
        out << "      \"line_search_trials\": " << row.line_search_trials << ",\n";
        out << "      \"step_accepted\": "
            << (row.step_accepted ? "true" : "false") << ",\n";
        out << "      \"linear_iterations\": " << row.linear_iterations << ",\n";
        out << "      \"linear_max_iterations\": "
            << row.linear_max_iterations << ",\n";
        out << "      \"linear_converged\": "
            << (row.linear_converged ? "true" : "false") << ",\n";
        out << "      \"true_residual_converged\": "
            << (row.true_residual_converged ? "true" : "false") << ",\n";
        out << "      \"outer_converged\": "
            << (row.outer_converged ? "true" : "false") << ",\n";
        out << "      \"rhs_norm\": " << row.rhs_norm << ",\n";
        out << "      \"residual_norm\": " << row.residual_norm << ",\n";
        out << "      \"relative_residual_norm\": " << row.relative_residual_norm << ",\n";
        out << "      \"initial_preconditioned_residual\": "
            << row.initial_preconditioned_residual << ",\n";
        out << "      \"final_preconditioned_residual\": "
            << row.final_preconditioned_residual << ",\n";
        out << "      \"preconditioned_residual_ratio\": "
            << row.preconditioned_residual_ratio << ",\n";
        out << "      \"timing\": {\"assembly_sec\": " << row.assembly_sec
            << ", \"preconditioner_sec\": " << row.preconditioner_sec
            << ", \"pcg_sec\": " << row.pcg_sec
            << ", \"residual_check_sec\": " << row.residual_check_sec
            << ", \"update_sec\": " << row.update_sec
            << ", \"objective_sec\": " << row.objective_sec
            << ", \"total_sec\": " << row.total_sec << "}\n";
        out << "    }" << (i + 1 == records.size() ? "\n" : ",\n");
    }
    out << "  ],\n";
    out << "  \"final_poses\": [\n";
    for (size_t i = 0; i < poses.size(); ++i) {
        out << "    ";
        Traits::writePoseJson(out, problem.original_ids[i], poses[i]);
        out << (i + 1 == poses.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
}

template <typename Traits>
int run(int argc, char** argv) {
    constexpr int D = Traits::kDimension;
    using Block = Eigen::Matrix<double, D, D, Eigen::ColMajor>;

    const Args args = parseArgs(argc, argv, Traits::geometryName());
    const auto setup_begin = Clock::now();
    typename Traits::Problem problem = Traits::load(args.input);
    typename Traits::PoseVector poses = problem.poses;
    BlockSystem<D> system(
        static_cast<int>(poses.size()),
        problem.fixed_index,
        problem.edges
    );
    std::unique_ptr<TwoLevelSchwarz<D>> schwarz;
    if (args.pcg_preconditioner == "two_level_schwarz") {
        schwarz = std::make_unique<TwoLevelSchwarz<D>>(
            problem,
            system,
            args.schwarz_subdomains,
            args.schwarz_overlap_layers,
            args.schwarz_threads,
            args.schwarz_local_shift,
            args.schwarz_enable_coarse,
            args.schwarz_include_rotation_modes
        );
    }
    const Objectives initial_objective =
        evaluateObjectives<Traits>(problem, poses, args.huber_delta);
    const auto setup_end = Clock::now();
    const double setup_sec = elapsedSeconds(setup_begin, setup_end);

    std::vector<OuterRecord> records;
    records.reserve(static_cast<size_t>(args.num_outer));
    Objectives previous_objective = initial_objective;
    double initial_gradient_norm = 0.0;
    double stopping_gradient_norm = 0.0;
    int gradient_stop_check_outer = 0;
    bool paper_gradient_stopped = false;

    const auto solve_begin = Clock::now();
    for (int outer = 1; outer <= args.num_outer; ++outer) {
        OuterRecord row;
        row.outer = outer;
        const auto outer_begin = Clock::now();

        const auto assembly_begin = Clock::now();
        assembleSystem<Traits>(
            problem,
            poses,
            args.huber_delta,
            args.diagonal_jitter,
            system
        );
        const auto assembly_end = Clock::now();
        row.assembly_sec = elapsedSeconds(assembly_begin, assembly_end);

        const Eigen::VectorXd rhs = system.rhs();
        row.rhs_norm = rhs.norm();
        if (outer == 1) {
            initial_gradient_norm = row.rhs_norm;
        }
        stopping_gradient_norm = row.rhs_norm;
        gradient_stop_check_outer = outer;
        if (
            args.paper_gradient_stopping &&
            outer > 1 &&
            (
                row.rhs_norm <= args.gradient_absolute_tolerance ||
                row.rhs_norm <=
                    args.gradient_relative_tolerance *
                    initial_gradient_norm
            )
        ) {
            paper_gradient_stopped = true;
            break;
        }
        Eigen::VectorXd schwarz_initial_z;
        if (schwarz) {
            const auto preconditioner_begin = Clock::now();
            if (!schwarz->factorize(system)) {
                throw std::runtime_error(
                    "two-level Schwarz numerical factorization failed"
                );
            }
            if (!schwarz->apply(rhs, schwarz_initial_z)) {
                throw std::runtime_error(
                    "two-level Schwarz initial preconditioner application failed"
                );
            }
            const auto preconditioner_end = Clock::now();
            row.preconditioner_sec =
                elapsedSeconds(preconditioner_begin, preconditioner_end);
            row.initial_preconditioned_residual =
                std::max(0.0, rhs.dot(schwarz_initial_z));
        } else {
            row.initial_preconditioned_residual =
                args.pcg_preconditioner == "block_jacobi"
                    ? system.preconditionedResidualEnergy(rhs)
                    : rhs.squaredNorm();
        }

        Eigen::VectorXd delta = Eigen::VectorXd::Zero(rhs.size());
        row.linear_max_iterations =
            args.pcg_max_iterations < 0
                ? static_cast<int>(rhs.size())
                : args.pcg_max_iterations;
        const auto pcg_begin = Clock::now();
        bool solve_ok = false;
        if (args.pcg_preconditioner == "block_jacobi") {
            Eigen::VectorXd mutable_rhs = rhs;
            g2o::LinearSolverPCG<Block> solver;
            solver.setMaxIterations(args.pcg_max_iterations);
            solver.setTolerance(args.pcg_tolerance);
            solver.setAbsoluteTolerance(false);
            solver.init();
            g2o::G2OBatchStatistics stats;
            stats.iterationsLinearSolver = -1;
            g2o::G2OBatchStatistics::setGlobalStats(&stats);
            solve_ok = solver.solve(system.hessian(), delta.data(), mutable_rhs.data());
            g2o::G2OBatchStatistics::setGlobalStats(nullptr);
            row.linear_iterations = std::max(0, stats.iterationsLinearSolver);
            row.linear_converged =
                row.linear_iterations < row.linear_max_iterations;
        } else if (args.pcg_preconditioner == "incomplete_cholesky") {
            const Eigen::SparseMatrix<double> matrix = system.scalarSparseMatrix();
            Eigen::ConjugateGradient<
                Eigen::SparseMatrix<double>,
                Eigen::Lower | Eigen::Upper,
                Eigen::IncompleteCholesky<double>
            > solver;
            solver.setMaxIterations(row.linear_max_iterations);
            solver.setTolerance(args.pcg_tolerance);
            solver.preconditioner().setInitialShift(args.ic_initial_shift);
            solver.compute(matrix);
            if (solver.info() == Eigen::Success) {
                delta = solver.solve(rhs);
            }
            row.linear_iterations = solver.iterations();
            row.linear_converged = solver.info() == Eigen::Success;
            solve_ok =
                solver.info() != Eigen::NumericalIssue && delta.allFinite();
        } else {
            Eigen::VectorXd residual = rhs;
            Eigen::VectorXd z = schwarz_initial_z;
            Eigen::VectorXd direction = z;
            Eigen::VectorXd product(rhs.size());
            double energy = residual.dot(z);
            const double initial_energy = energy;
            const double target =
                args.pcg_tolerance * args.pcg_tolerance *
                std::max(0.0, initial_energy);
            solve_ok =
                std::isfinite(energy) && energy >= 0.0 &&
                direction.allFinite();
            int iteration = 0;
            while (solve_ok && iteration < row.linear_max_iterations &&
                   energy > target) {
                product = system.multiply(direction);
                const double denominator = direction.dot(product);
                if (!std::isfinite(denominator) ||
                    denominator <=
                        std::numeric_limits<double>::epsilon() *
                            std::max(
                                std::numeric_limits<double>::min(),
                                direction.squaredNorm()
                            )) {
                    solve_ok = false;
                    break;
                }
                const double alpha = energy / denominator;
                delta.noalias() += alpha * direction;
                residual.noalias() -= alpha * product;
                Eigen::VectorXd next_z;
                if (!schwarz->apply(residual, next_z)) {
                    solve_ok = false;
                    break;
                }
                const double next_energy = residual.dot(next_z);
                if (!std::isfinite(next_energy) ||
                    next_energy < -1e-12 * std::max(1.0, initial_energy)) {
                    solve_ok = false;
                    break;
                }
                ++iteration;
                if (next_energy <= target) {
                    energy = std::max(0.0, next_energy);
                    z = std::move(next_z);
                    break;
                }
                const double beta = next_energy / energy;
                direction = next_z + beta * direction;
                z = std::move(next_z);
                energy = next_energy;
            }
            row.linear_iterations = iteration;
            row.linear_converged =
                solve_ok && energy <= target;
            solve_ok = solve_ok && delta.allFinite();
        }
        const auto pcg_end = Clock::now();
        row.pcg_sec = elapsedSeconds(pcg_begin, pcg_end);
        if (!solve_ok || !delta.allFinite()) {
            throw std::runtime_error("g2o block PCG failed or produced a non-finite step");
        }

        row.proposed_step_norm = delta.norm();

        const auto residual_begin = Clock::now();
        const Eigen::VectorXd true_residual = rhs - system.multiply(delta);
        row.residual_norm = true_residual.norm();
        row.relative_residual_norm =
            row.rhs_norm > 0.0 ? row.residual_norm / row.rhs_norm : row.residual_norm;
        row.final_preconditioned_residual =
            args.pcg_preconditioner == "block_jacobi"
                ? system.preconditionedResidualEnergy(true_residual)
                : true_residual.squaredNorm();
        if (schwarz) {
            Eigen::VectorXd final_z;
            if (!schwarz->apply(true_residual, final_z)) {
                throw std::runtime_error(
                    "two-level Schwarz final preconditioner application failed"
                );
            }
            row.final_preconditioned_residual =
                std::max(0.0, true_residual.dot(final_z));
        }
        row.preconditioned_residual_ratio =
            row.initial_preconditioned_residual > 0.0
                ? row.final_preconditioned_residual / row.initial_preconditioned_residual
                : row.final_preconditioned_residual;
        row.true_residual_converged =
            row.initial_preconditioned_residual == 0.0 ||
            (args.pcg_preconditioner == "block_jacobi" || schwarz
                 ? row.final_preconditioned_residual <=
                       (schwarz
                            ? args.pcg_tolerance * args.pcg_tolerance
                            : args.pcg_tolerance) *
                       row.initial_preconditioned_residual *
                           (1.0 + 1e-8)
                 : row.relative_residual_norm <= args.pcg_tolerance * (1.0 + 1e-8));
        const auto residual_end = Clock::now();
        row.residual_check_sec = elapsedSeconds(residual_begin, residual_end);

        Objectives objective;
        if (args.line_search) {
            double scale = 1.0;
            bool accepted = false;
            for (int trial = 1; trial <= args.line_search_max_trials; ++trial) {
                typename Traits::PoseVector candidate = poses;
                const auto update_begin = Clock::now();
                for (size_t pose = 0; pose < candidate.size(); ++pose) {
                    const int block = system.activeBlock(static_cast<int>(pose));
                    if (block >= 0) {
                        candidate[pose] = Traits::plus(
                            candidate[pose],
                            scale * delta.template segment<D>(D * block)
                        );
                    }
                }
                const auto update_end = Clock::now();
                row.update_sec += elapsedSeconds(update_begin, update_end);

                const auto objective_begin = Clock::now();
                const Objectives candidate_objective =
                    evaluateObjectives<Traits>(problem, candidate, args.huber_delta);
                const auto objective_end = Clock::now();
                row.objective_sec += elapsedSeconds(objective_begin, objective_end);
                row.line_search_trials = trial;
                if (candidate_objective.huber <=
                    previous_objective.huber +
                        1e-12 * std::max(1.0, std::abs(previous_objective.huber))) {
                    poses = std::move(candidate);
                    objective = candidate_objective;
                    row.step_scale = scale;
                    accepted = true;
                    break;
                }
                scale *= args.line_search_shrink;
            }
            row.step_accepted = accepted;
            if (!accepted) {
                row.step_scale = 0.0;
                objective = previous_objective;
            }
        } else {
            const auto update_begin = Clock::now();
            for (size_t pose = 0; pose < poses.size(); ++pose) {
                const int block = system.activeBlock(static_cast<int>(pose));
                if (block >= 0) {
                    poses[pose] = Traits::plus(
                        poses[pose],
                        delta.template segment<D>(D * block)
                    );
                }
            }
            const auto update_end = Clock::now();
            row.update_sec = elapsedSeconds(update_begin, update_end);

            const auto objective_begin = Clock::now();
            objective = evaluateObjectives<Traits>(problem, poses, args.huber_delta);
            const auto objective_end = Clock::now();
            row.objective_sec = elapsedSeconds(objective_begin, objective_end);
            row.line_search_trials = 1;
        }
        row.step_norm = row.step_scale * row.proposed_step_norm;
        row.raw_objective = objective.raw;
        row.huber_objective = objective.huber;
        row.relative_huber_change =
            std::abs(previous_objective.huber - objective.huber) /
            std::max(1.0, std::abs(previous_objective.huber));
        row.outer_converged =
            row.step_norm <= args.outer_step_tolerance ||
            row.relative_huber_change <= args.outer_objective_tolerance;
        previous_objective = objective;

        const auto outer_end = Clock::now();
        row.total_sec = elapsedSeconds(outer_begin, outer_end);
        records.push_back(row);
        if (!args.quiet) {
            std::cout << "[" << Traits::geometryName() << "] outer=" << outer
                      << " pcg_iter=" << row.linear_iterations
                      << " pcg_converged=" << row.linear_converged
                      << " true_residual_converged=" << row.true_residual_converged
                      << " rel_residual=" << std::setprecision(6)
                      << row.relative_residual_norm
                      << " raw=" << std::setprecision(17) << row.raw_objective
                      << " huber=" << row.huber_objective
                      << " sec=" << row.total_sec << "\n";
        }
    }
    const auto solve_end = Clock::now();
    const double solve_sec = elapsedSeconds(solve_begin, solve_end);
    const double total_sec = setup_sec + solve_sec;
    const Objectives final_objective =
        records.empty()
            ? initial_objective
            : Objectives{records.back().raw_objective, records.back().huber_objective};

    ensureParentDirectory(args.output_poses);
    Traits::writeG2O(problem, poses, args.output_poses);
    writeJson<Traits>(
        args,
        problem,
        poses,
        schwarz.get(),
        initial_objective,
        final_objective,
        records,
        setup_sec,
        solve_sec,
        total_sec,
        paper_gradient_stopped,
        gradient_stop_check_outer,
        initial_gradient_norm,
        stopping_gradient_norm
    );
    std::cout << std::setprecision(17)
              << "summary geometry=" << Traits::geometryName()
              << " poses=" << poses.size()
              << " edges=" << problem.edges.size()
              << " setup_sec=" << setup_sec
              << " solve_sec=" << solve_sec
              << " total_sec=" << total_sec
              << " raw=" << final_objective.raw
              << " huber=" << final_objective.huber
              << " json=" << args.output_json
              << " poses_out=" << args.output_poses << "\n";
    return 0;
}

}  // namespace reviewer_pcg
