#pragma once

#include <Eigen/SparseCore>

#include "internal/se2_basis_data.h"
#include "internal/se2_group_sync.h"
#include "internal/se2_residual.h"

namespace slam {

struct SyntheticSE2PackedCoarseBinaryBlockSlots {
    int idx00 = -1;
    int idx01 = -1;
    int idx10 = -1;
    int idx11 = -1;
};

struct SyntheticSE2PackedCoarseActiveBlock {
    int row_offset = 0;
    int col_offset = 0;
    int rows = 0;
    int cols = 0;
    size_t storage_offset = 0;
    Eigen::MatrixXd values;
};

struct SyntheticSE2PackedCoarseAssemblyWorkspace {
    int coarse_dim = 0;
    int num_groups = 0;
    std::vector<int> pair_to_block_index;
    std::vector<int> var_diag_block_slots;
    std::vector<int> unary_diag_block_slots;
    std::vector<SyntheticSE2PackedCoarseBinaryBlockSlots> binary_block_slots;
    std::vector<SyntheticSE2PackedCoarseActiveBlock> active_blocks;
    size_t triplet_capacity = 0;
    size_t block_storage_size = 0;
    int block_buffer_thread_count = 0;
    std::vector<std::vector<double>> thread_local_block_buffers;
};

SyntheticSE2PackedCoarseAssemblyWorkspace buildSyntheticSE2PackedCoarseAssemblyWorkspace(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis
);

SyntheticSE2PackedCoarseAssemblyWorkspace buildSyntheticSE2PackedCoarseAssemblyWorkspace(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis
);

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectPackedOrdered3D(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    SyntheticSE2PackedCoarseAssemblyWorkspace& assembly_workspace,
    int num_threads = 1
);

Eigen::VectorXd assembleCoarseResidualDirectPackedOrdered3D(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    int num_threads = 1
);

void assembleCoarseResidualDirectPackedOrdered3DInto(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse,
    int num_threads = 1
);

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectPackedOrdered3D(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    SyntheticSE2PackedCoarseAssemblyWorkspace& assembly_workspace,
    int num_threads = 1
);

Eigen::VectorXd assembleCoarseResidualDirectPackedOrdered3D(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    int num_threads = 1
);

void assembleCoarseResidualDirectPackedOrdered3DInto(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse,
    int num_threads = 1
);

}  // namespace slam
