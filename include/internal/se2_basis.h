#pragma once

#include <vector>

#include "internal/se2_basis_data.h"
#include "internal/se2_group_sync.h"
#include "internal/se2_solver_impl.h"
#include "internal/se2_residual.h"

namespace slam {

struct SyntheticSE2PackedBasisTopologyBoundaryPlan {
    bool is_unary = false;
    int row_off = 0;
    const double* lam6_ptr = nullptr;
};

struct SyntheticSE2PackedBasisTopologyUnaryPlan {
    int row_off = 0;
    const double* lam6_ptr = nullptr;
};

struct SyntheticSE2PackedBasisTopologyBinaryPlan {
    int row0_off = 0;
    int row1_off = 0;
    const double* diag0_lam6_ptr = nullptr;
    const double* diag1_lam6_ptr = nullptr;
    const double* cross01_lam9_ptr = nullptr;
};

struct SyntheticSE2PackedBasisTopologyGroupPlan {
    std::vector<const double*> prior_lam6_ptrs;
    std::vector<int> prior_row_offs;
    std::vector<SyntheticSE2PackedBasisTopologyUnaryPlan> interior_unary;
    std::vector<SyntheticSE2PackedBasisTopologyBinaryPlan> interior_binary;
    std::vector<SyntheticSE2PackedBasisTopologyBoundaryPlan> boundary_msgs;
};

struct SyntheticSE2PackedBasisTopology {
    std::vector<std::vector<int>> groups;
    std::vector<std::vector<int>> full_indices_per_group;
    std::vector<SyntheticSE2PackedBasisTopologyGroupPlan> group_plans;
    std::vector<int> var_to_group;
    std::vector<int> var_to_local_offset;
    int total_dim = 0;
    bool ordered_3d_fast = true;
    bool factors_arity12_only_3d = true;
};

SyntheticSE2PackedBasisTopology buildMessageConditionedBasisPackedTopology(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    int group_size
);

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const SyntheticSE2PackedBasisTopology& topology,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
);

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
);

SyntheticSE2PackedBasisTopology buildMessageConditionedBasisPackedTopology(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int group_size
);

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const SyntheticSE2PackedBasisTopology& topology,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
);

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
);

}  // namespace slam
