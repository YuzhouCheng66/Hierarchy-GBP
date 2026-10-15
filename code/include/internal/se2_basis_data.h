#pragma once

#include <vector>

#include <Eigen/Dense>

namespace gbp {
class Factor;
}

namespace slam {

struct BasisData {
    struct GroupUnary3Plan {
        const gbp::Factor* factor = nullptr;
        int row_off = 0;
    };

    struct GroupBinary3Plan {
        const gbp::Factor* factor = nullptr;
        int row0_off = 0;
        int row1_off = 0;
    };

    struct GroupBoundary3Plan {
        const gbp::Factor* factor = nullptr;
        int local_idx = 0;
        int row_off = 0;
    };

    struct GroupFastPlan {
        std::vector<const double*> prior_lam_ptrs;
        std::vector<int> prior_row_offs;
        std::vector<GroupUnary3Plan> interior_unary;
        std::vector<GroupBinary3Plan> interior_binary;
        std::vector<GroupBoundary3Plan> boundary_msgs;
    };

    std::vector<std::vector<int>> groups;
    std::vector<std::vector<int>> full_indices_per_group;
    std::vector<Eigen::MatrixXd> local_bases;
    std::vector<GroupFastPlan> group_fast_plans;
    std::vector<int> coarse_offsets;
    std::vector<int> var_to_group;
    std::vector<int> var_to_local_offset;
    bool ordered_3d_fast = false;
    bool factors_arity12_only_3d = false;
    std::vector<int> var_r_local;
    std::vector<int> var_coarse_offset;
    std::vector<int> var_basis_offset;
    std::vector<double> var_basis_blocks;
    int total_dim = 0;
    int coarse_dim = 0;
    int partial_attempted = 0;
    int partial_converged = 0;
    int partial_fallback = 0;
    int partial_total_iters = 0;
    double block_build_sec = 0.0;
    double eigensolver_sec = 0.0;
    double copyout_sec = 0.0;
};

inline const double* varBasis3xRPtr(const BasisData& basis, int var_id) {
    const int offset = basis.var_basis_offset[var_id];
    return offset >= 0 ? basis.var_basis_blocks.data() + offset : nullptr;
}

}  // namespace slam
