#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace reviewer_pcg {

// Two-level additive Schwarz with a GDSW-type coarse space. The trajectory is
// decomposed into contiguous, nonoverlapping odometry-edge ranges. Adjacent
// closed ranges share one interface pose, and overlap is grown using odometry
// edges only. Each interface coordinate mode is extended into neighboring
// interiors by solving A_II phi_I = -A_I_Gamma phi_Gamma.
template <int D>
class TwoLevelSchwarz {
public:
    using DenseBlock = Eigen::Matrix<double, D, D, Eigen::ColMajor>;
    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
    using SparseFactor = Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower>;

    template <typename Problem, typename System>
    TwoLevelSchwarz(
        const Problem& problem,
        const System& system,
        int requested_subdomains,
        int overlap_layers,
        int threads,
        double local_shift,
        bool enable_coarse = true,
        bool include_rotation_modes = true
    )
        : threads_(std::max(1, threads)),
          local_shift_(std::max(0.0, local_shift)),
          enable_coarse_(enable_coarse),
          include_rotation_modes_(include_rotation_modes) {
        const std::vector<int>& active_by_pose = system.activeBlockByPose();
        pose_count_ = static_cast<int>(active_by_pose.size());
        active_blocks_ = 0;
        for (const int block : active_by_pose) {
            active_blocks_ = std::max(active_blocks_, block + 1);
        }
        if (pose_count_ < 2 || active_blocks_ <= 0) {
            throw std::runtime_error(
                "Schwarz preconditioner needs at least two poses"
            );
        }

        const int odometry_edge_count = pose_count_ - 1;
        subdomains_ = std::clamp(
            requested_subdomains,
            1,
            odometry_edge_count
        );
        mode_components_ = coordinateModes();

        std::vector<std::vector<int>> odometry_graph(
            static_cast<size_t>(active_blocks_)
        );
        for (const auto& edge : problem.edges) {
            const int bi = active_by_pose.at(static_cast<size_t>(edge.i));
            const int bj = active_by_pose.at(static_cast<size_t>(edge.j));
            if (bi < 0 || bj < 0 || bi == bj) {
                continue;
            }
            if (std::abs(edge.i - edge.j) == 1) {
                odometry_graph[static_cast<size_t>(bi)].push_back(bj);
                odometry_graph[static_cast<size_t>(bj)].push_back(bi);
            }
        }
        sortUnique(odometry_graph);

        interface_index_by_block_.assign(
            static_cast<size_t>(active_blocks_),
            -1
        );
        for (int domain_index = 1;
             domain_index < subdomains_;
             ++domain_index) {
            const int boundary_pose =
                (domain_index * odometry_edge_count) / subdomains_;
            const int boundary_block =
                active_by_pose.at(static_cast<size_t>(boundary_pose));
            if (boundary_block < 0) {
                continue;
            }
            if (interface_index_by_block_[static_cast<size_t>(boundary_block)] < 0) {
                interface_index_by_block_[static_cast<size_t>(boundary_block)] =
                    static_cast<int>(interface_blocks_.size());
                interface_blocks_.push_back(boundary_block);
            }
        }

        domains_.resize(static_cast<size_t>(subdomains_));
        for (int domain_index = 0;
             domain_index < subdomains_;
             ++domain_index) {
            Domain& domain = domains_[static_cast<size_t>(domain_index)];
            const int first_edge =
                (domain_index * odometry_edge_count) / subdomains_;
            const int edge_end =
                ((domain_index + 1) * odometry_edge_count) / subdomains_;
            for (int pose = first_edge; pose <= edge_end; ++pose) {
                const int block =
                    active_by_pose.at(static_cast<size_t>(pose));
                if (block >= 0) {
                    domain.core_blocks.push_back(block);
                }
            }
            sortUnique(domain.core_blocks);

            for (const int block : domain.core_blocks) {
                if (interface_index_by_block_[static_cast<size_t>(block)] < 0) {
                    domain.interior_blocks.push_back(block);
                } else {
                    appendInterfaceModes(domain, block);
                }
            }

            domain.overlap_blocks = domain.core_blocks;
            growOverlap(
                domain.overlap_blocks,
                odometry_graph,
                overlap_layers
            );
            sortUnique(domain.overlap_blocks);
            buildPattern(system, domain.overlap_blocks, domain.overlap);
            if (!domain.interior_blocks.empty()) {
                buildPattern(system, domain.interior_blocks, domain.interior);
            }
        }

        coarse_dimension_ =
            static_cast<int>(interface_blocks_.size()) *
            static_cast<int>(mode_components_.size());
        if (enable_coarse_ && coarse_dimension_ == 0) {
            enable_coarse_ = false;
        }
        if (enable_coarse_) {
            coarse_rhs_.resize(coarse_dimension_);
            coarse_solution_.resize(coarse_dimension_);
        }
    }

    template <typename System>
    bool factorize(const System& system) {
        int failures = 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static) reduction(+ : failures)
#endif
        for (int index = 0; index < subdomains_; ++index) {
            Domain& domain = domains_[static_cast<size_t>(index)];
            if (!refreshAndFactor(domain.overlap)) {
                ++failures;
                continue;
            }
            if (!enable_coarse_ || domain.interior_blocks.empty()) {
                continue;
            }
            if (!refreshAndFactor(domain.interior)) {
                ++failures;
                continue;
            }
            buildHarmonicExtension(system, domain);
            if (!domain.harmonic_solution.allFinite()) {
                ++failures;
            }
        }
        if (failures != 0) {
            return false;
        }
        if (!enable_coarse_) {
            basis_nonzeros_ = 0;
            return true;
        }

        using Triplet = Eigen::Triplet<double, int>;
        std::vector<Triplet> basis_triplets;
        size_t reserve = interface_blocks_.size() *
            mode_components_.size();
        for (const Domain& domain : domains_) {
            reserve += static_cast<size_t>(D) *
                domain.interior_blocks.size() *
                domain.modes.size();
        }
        basis_triplets.reserve(reserve);

        for (size_t interface = 0;
             interface < interface_blocks_.size();
             ++interface) {
            const int block = interface_blocks_[interface];
            for (size_t mode = 0; mode < mode_components_.size(); ++mode) {
                basis_triplets.emplace_back(
                    D * block + mode_components_[mode],
                    coarseColumn(static_cast<int>(interface), mode),
                    1.0
                );
            }
        }
        for (const Domain& domain : domains_) {
            for (int local_block = 0;
                 local_block <
                     static_cast<int>(domain.interior_blocks.size());
                 ++local_block) {
                const int global_block =
                    domain.interior_blocks[static_cast<size_t>(local_block)];
                for (int component = 0; component < D; ++component) {
                    const int row = D * global_block + component;
                    for (int local_mode = 0;
                         local_mode <
                             static_cast<int>(domain.modes.size());
                         ++local_mode) {
                        basis_triplets.emplace_back(
                            row,
                            domain.modes[static_cast<size_t>(local_mode)]
                                .coarse_column,
                            domain.harmonic_solution(
                                D * local_block + component,
                                local_mode
                            )
                        );
                    }
                }
            }
        }

        basis_.resize(D * active_blocks_, coarse_dimension_);
        basis_.setFromTriplets(
            basis_triplets.begin(),
            basis_triplets.end()
        );
        basis_.makeCompressed();
        basis_nonzeros_ = static_cast<size_t>(basis_.nonZeros());

        const SparseMatrix fine_matrix = system.scalarSparseMatrix();
        SparseMatrix coarse =
            basis_.transpose() * fine_matrix * basis_;
        SparseMatrix coarse_transpose = coarse.transpose();
        coarse = 0.5 * (coarse + coarse_transpose);
        coarse.makeCompressed();
        if (local_shift_ > 0.0) {
            for (int index = 0; index < coarse_dimension_; ++index) {
                coarse.coeffRef(index, index) += local_shift_;
            }
            coarse.makeCompressed();
        }
        coarse_matrix_ = std::move(coarse);
        coarse_factor_.compute(coarse_matrix_);
        return coarse_factor_.info() == Eigen::Success;
    }

    bool apply(const Eigen::VectorXd& residual, Eigen::VectorXd& result) {
        if (residual.size() != D * active_blocks_) {
            return false;
        }

#ifdef _OPENMP
#pragma omp parallel for num_threads(threads_) schedule(static)
#endif
        for (int index = 0; index < subdomains_; ++index) {
            Domain& domain = domains_[static_cast<size_t>(index)];
            gather(
                residual,
                domain.overlap_blocks,
                domain.overlap.rhs
            );
            domain.overlap.solution =
                domain.overlap.factor->solve(domain.overlap.rhs);
        }

        result.setZero(residual.size());
        for (const Domain& domain : domains_) {
            if (domain.overlap.factor->info() != Eigen::Success ||
                !domain.overlap.solution.allFinite()) {
                return false;
            }
            scatterAdd(
                domain.overlap.solution,
                domain.overlap_blocks,
                result
            );
        }

        if (enable_coarse_) {
            coarse_rhs_.noalias() = basis_.transpose() * residual;
            coarse_solution_ = coarse_factor_.solve(coarse_rhs_);
            if (coarse_factor_.info() != Eigen::Success ||
                !coarse_solution_.allFinite()) {
                return false;
            }
            result.noalias() += basis_ * coarse_solution_;
        }
        return result.allFinite();
    }

    int subdomains() const {
        return subdomains_;
    }

    int threads() const {
        return threads_;
    }

    int coarseDimension() const {
        return enable_coarse_ ? coarse_dimension_ : 0;
    }

    int interfaceBlocks() const {
        return static_cast<int>(interface_blocks_.size());
    }

    size_t basisNonZeros() const {
        return basis_nonzeros_;
    }

    size_t totalLocalBlocks() const {
        size_t count = 0;
        for (const Domain& domain : domains_) {
            count += domain.overlap_blocks.size();
        }
        return count;
    }

private:
    struct MatrixWorkspace {
        SparseMatrix matrix;
        std::unique_ptr<SparseFactor> factor;
        std::vector<const double*> sources;
        std::vector<int> diagonal_value_indices;
        Eigen::VectorXd rhs;
        Eigen::VectorXd solution;
    };

    struct InterfaceMode {
        int interface_block = -1;
        int component = -1;
        int coarse_column = -1;
    };

    struct Domain {
        std::vector<int> core_blocks;
        std::vector<int> interior_blocks;
        std::vector<int> overlap_blocks;
        std::vector<InterfaceMode> modes;
        MatrixWorkspace overlap;
        MatrixWorkspace interior;
        Eigen::MatrixXd harmonic_rhs;
        Eigen::MatrixXd harmonic_solution;
    };

    std::vector<int> coordinateModes() const {
        const int translation_dimension = D == 3 ? 2 : 3;
        std::vector<int> result;
        const int count =
            include_rotation_modes_ ? D : translation_dimension;
        result.reserve(static_cast<size_t>(count));
        for (int component = 0; component < count; ++component) {
            result.push_back(component);
        }
        return result;
    }

    int coarseColumn(int interface_index, size_t local_mode) const {
        return interface_index *
            static_cast<int>(mode_components_.size()) +
            static_cast<int>(local_mode);
    }

    void appendInterfaceModes(Domain& domain, int block) const {
        const int interface_index =
            interface_index_by_block_.at(static_cast<size_t>(block));
        if (interface_index < 0) {
            return;
        }
        for (size_t mode = 0; mode < mode_components_.size(); ++mode) {
            domain.modes.push_back(
                InterfaceMode{
                    block,
                    mode_components_[mode],
                    coarseColumn(interface_index, mode)
                }
            );
        }
    }

    static void sortUnique(std::vector<int>& values) {
        std::sort(values.begin(), values.end());
        values.erase(
            std::unique(values.begin(), values.end()),
            values.end()
        );
    }

    static void sortUnique(std::vector<std::vector<int>>& graph) {
        for (std::vector<int>& neighbors : graph) {
            sortUnique(neighbors);
        }
    }

    static void growOverlap(
        std::vector<int>& blocks,
        const std::vector<std::vector<int>>& graph,
        int layers
    ) {
        std::vector<unsigned char> included(graph.size(), 0);
        std::vector<int> frontier = blocks;
        for (const int block : blocks) {
            included[static_cast<size_t>(block)] = 1;
        }
        for (int layer = 0; layer < layers; ++layer) {
            std::vector<int> next;
            for (const int block : frontier) {
                for (const int neighbor : graph[static_cast<size_t>(block)]) {
                    if (!included[static_cast<size_t>(neighbor)]) {
                        included[static_cast<size_t>(neighbor)] = 1;
                        next.push_back(neighbor);
                        blocks.push_back(neighbor);
                    }
                }
            }
            frontier = std::move(next);
            if (frontier.empty()) {
                break;
            }
        }
    }

    template <typename System>
    static void buildPattern(
        const System& system,
        const std::vector<int>& blocks,
        MatrixWorkspace& workspace
    ) {
        using Triplet = Eigen::Triplet<double, int>;
        int active_blocks = 0;
        for (const int block : system.activeBlockByPose()) {
            active_blocks = std::max(active_blocks, block + 1);
        }
        std::vector<int> local_by_global(
            static_cast<size_t>(active_blocks),
            -1
        );
        for (int local = 0;
             local < static_cast<int>(blocks.size());
             ++local) {
            local_by_global[static_cast<size_t>(
                blocks[static_cast<size_t>(local)]
            )] = local;
        }

        std::vector<Triplet> triplets;
        const auto& hessian = system.hessian();
        for (int column = 0;
             column < static_cast<int>(hessian.blockCols().size());
             ++column) {
            const int local_column =
                local_by_global[static_cast<size_t>(column)];
            if (local_column < 0) {
                continue;
            }
            for (const auto& entry :
                 hessian.blockCols()[static_cast<size_t>(column)]) {
                const int row = entry.first;
                const int local_row =
                    local_by_global[static_cast<size_t>(row)];
                if (local_row < 0) {
                    continue;
                }
                if (row == column) {
                    for (int component_column = 0;
                         component_column < D;
                         ++component_column) {
                        for (int component_row = component_column;
                             component_row < D;
                             ++component_row) {
                            triplets.emplace_back(
                                D * local_row + component_row,
                                D * local_column + component_column,
                                1.0
                            );
                        }
                    }
                } else {
                    for (int component_column = 0;
                         component_column < D;
                         ++component_column) {
                        for (int component_row = 0;
                             component_row < D;
                             ++component_row) {
                            triplets.emplace_back(
                                D * local_column + component_column,
                                D * local_row + component_row,
                                1.0
                            );
                        }
                    }
                }
            }
        }

        const int dimension = D * static_cast<int>(blocks.size());
        workspace.matrix.resize(dimension, dimension);
        workspace.matrix.setFromTriplets(
            triplets.begin(),
            triplets.end()
        );
        workspace.matrix.makeCompressed();
        workspace.sources.resize(
            static_cast<size_t>(workspace.matrix.nonZeros())
        );

        int value_index = 0;
        for (int local_column = 0;
             local_column < workspace.matrix.outerSize();
             ++local_column) {
            for (typename SparseMatrix::InnerIterator entry(
                     workspace.matrix,
                     local_column
                 );
                 entry;
                 ++entry, ++value_index) {
                const int local_row = entry.row();
                const int row_block =
                    blocks[static_cast<size_t>(local_row / D)];
                const int column_block =
                    blocks[static_cast<size_t>(local_column / D)];
                const int row_component = local_row % D;
                const int column_component = local_column % D;
                const DenseBlock* source_block = nullptr;
                const double* source = nullptr;
                if (row_block == column_block) {
                    source_block = hessian.block(row_block, column_block);
                    source = source_block->data() +
                        row_component + D * column_component;
                    if (row_component == column_component) {
                        workspace.diagonal_value_indices.push_back(
                            value_index
                        );
                    }
                } else {
                    const int upper_row =
                        std::min(row_block, column_block);
                    const int upper_column =
                        std::max(row_block, column_block);
                    source_block =
                        hessian.block(upper_row, upper_column);
                    if (row_block > column_block) {
                        source = source_block->data() +
                            column_component + D * row_component;
                    } else {
                        source = source_block->data() +
                            row_component + D * column_component;
                    }
                }
                if (source_block == nullptr || source == nullptr) {
                    throw std::runtime_error(
                        "failed to map a Schwarz matrix entry"
                    );
                }
                workspace.sources[static_cast<size_t>(value_index)] =
                    source;
            }
        }
        workspace.factor = std::make_unique<SparseFactor>();
        workspace.factor->analyzePattern(workspace.matrix);
        if (workspace.factor->info() != Eigen::Success) {
            throw std::runtime_error(
                "Schwarz symbolic factorization failed"
            );
        }
        workspace.rhs.resize(dimension);
        workspace.solution.resize(dimension);
    }

    bool refreshAndFactor(MatrixWorkspace& workspace) const {
        double* values = workspace.matrix.valuePtr();
        for (size_t entry = 0;
             entry < workspace.sources.size();
             ++entry) {
            values[entry] = *workspace.sources[entry];
        }
        if (local_shift_ > 0.0) {
            for (const int diagonal_index :
                 workspace.diagonal_value_indices) {
                values[diagonal_index] += local_shift_;
            }
        }
        workspace.factor->factorize(workspace.matrix);
        return workspace.factor->info() == Eigen::Success;
    }

    template <typename System>
    static DenseBlock orientedBlock(
        const System& system,
        int row,
        int column
    ) {
        const DenseBlock* block = system.hessian().block(
            std::min(row, column),
            std::max(row, column)
        );
        if (block == nullptr) {
            return DenseBlock::Zero();
        }
        return row <= column ? *block : block->transpose().eval();
    }

    template <typename System>
    static void buildHarmonicExtension(
        const System& system,
        Domain& domain
    ) {
        const int rows =
            D * static_cast<int>(domain.interior_blocks.size());
        const int columns = static_cast<int>(domain.modes.size());
        domain.harmonic_rhs.setZero(rows, columns);
        for (int local_block = 0;
             local_block <
                 static_cast<int>(domain.interior_blocks.size());
             ++local_block) {
            const int interior_block =
                domain.interior_blocks[static_cast<size_t>(local_block)];
            for (int local_mode = 0;
                 local_mode < columns;
                 ++local_mode) {
                const InterfaceMode& mode =
                    domain.modes[static_cast<size_t>(local_mode)];
                const DenseBlock coupling = orientedBlock(
                    system,
                    interior_block,
                    mode.interface_block
                );
                domain.harmonic_rhs.template block<D, 1>(
                    D * local_block,
                    local_mode
                ) = -coupling.col(mode.component);
            }
        }
        domain.harmonic_solution =
            domain.interior.factor->solve(domain.harmonic_rhs);
    }

    static void gather(
        const Eigen::VectorXd& global,
        const std::vector<int>& blocks,
        Eigen::VectorXd& local
    ) {
        for (int index = 0;
             index < static_cast<int>(blocks.size());
             ++index) {
            local.template segment<D>(D * index) =
                global.template segment<D>(
                    D * blocks[static_cast<size_t>(index)]
                );
        }
    }

    static void scatterAdd(
        const Eigen::VectorXd& local,
        const std::vector<int>& blocks,
        Eigen::VectorXd& global
    ) {
        for (int index = 0;
             index < static_cast<int>(blocks.size());
             ++index) {
            global.template segment<D>(
                D * blocks[static_cast<size_t>(index)]
            ) += local.template segment<D>(D * index);
        }
    }

    int pose_count_ = 0;
    int active_blocks_ = 0;
    int subdomains_ = 0;
    int threads_ = 1;
    double local_shift_ = 0.0;
    bool enable_coarse_ = true;
    bool include_rotation_modes_ = true;
    int coarse_dimension_ = 0;
    size_t basis_nonzeros_ = 0;
    std::vector<int> mode_components_;
    std::vector<int> interface_blocks_;
    std::vector<int> interface_index_by_block_;
    std::vector<Domain> domains_;
    SparseMatrix basis_;
    SparseMatrix coarse_matrix_;
    SparseFactor coarse_factor_;
    Eigen::VectorXd coarse_rhs_;
    Eigen::VectorXd coarse_solution_;
};

}  // namespace reviewer_pcg
