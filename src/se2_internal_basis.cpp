#include "internal/se2_basis.h"

#include <algorithm>
#include <stdexcept>

#include <omp.h>

#include "internal/partial_symmetric_eigen.h"

namespace slam {

namespace {

std::vector<std::vector<int>> orderedGroupsPacked(int n_vars, int group_size) {
    if (group_size <= 0) {
        throw std::runtime_error("group_size must be positive");
    }
    std::vector<std::vector<int>> groups;
    int start = 0;
    while (start + 2 * group_size <= n_vars) {
        std::vector<int> group;
        group.reserve(group_size);
        for (int id = start; id < start + group_size; ++id) {
            group.push_back(id);
        }
        groups.push_back(std::move(group));
        start += group_size;
    }
    if (start < n_vars) {
        std::vector<int> tail;
        tail.reserve(n_vars - start);
        for (int id = start; id < n_vars; ++id) {
            tail.push_back(id);
        }
        groups.push_back(std::move(tail));
    }
    return groups;
}

inline const double* sym6Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

inline const double* unaryMsgLam6Ptr(const SyntheticSE2PackedResidualWorkspace& workspace, int idx) noexcept {
    return workspace.unary_msg_lam6.data() + static_cast<size_t>(idx) * 6;
}

inline const double* binaryMsgLam6Ptr(const SyntheticSE2PackedResidualWorkspace& workspace, int slot_id) noexcept {
    return workspace.binary_msg_lam6.data() + static_cast<size_t>(slot_id) * 6;
}

inline void addSymBlock3(Eigen::MatrixXd& info, int off, const double* sym6) {
    info(off + 0, off + 0) += sym6[0];
    info(off + 1, off + 0) += sym6[1];
    info(off + 2, off + 0) += sym6[2];
    info(off + 0, off + 1) += sym6[1];
    info(off + 1, off + 1) += sym6[3];
    info(off + 2, off + 1) += sym6[4];
    info(off + 0, off + 2) += sym6[2];
    info(off + 1, off + 2) += sym6[4];
    info(off + 2, off + 2) += sym6[5];
}

inline void addCrossBlock3AndTranspose(
    Eigen::MatrixXd& info,
    int row0_off,
    int row1_off,
    const double* cross01
) {
    const double x00 = cross01[0];
    const double x10 = cross01[1];
    const double x20 = cross01[2];
    const double x01 = cross01[3];
    const double x11 = cross01[4];
    const double x21 = cross01[5];
    const double x02 = cross01[6];
    const double x12 = cross01[7];
    const double x22 = cross01[8];

    info(row0_off + 0, row1_off + 0) += x00;
    info(row0_off + 1, row1_off + 0) += x10;
    info(row0_off + 2, row1_off + 0) += x20;
    info(row0_off + 0, row1_off + 1) += x01;
    info(row0_off + 1, row1_off + 1) += x11;
    info(row0_off + 2, row1_off + 1) += x21;
    info(row0_off + 0, row1_off + 2) += x02;
    info(row0_off + 1, row1_off + 2) += x12;
    info(row0_off + 2, row1_off + 2) += x22;

    info(row1_off + 0, row0_off + 0) += x00;
    info(row1_off + 1, row0_off + 0) += x01;
    info(row1_off + 2, row0_off + 0) += x02;
    info(row1_off + 0, row0_off + 1) += x10;
    info(row1_off + 1, row0_off + 1) += x11;
    info(row1_off + 2, row0_off + 1) += x12;
    info(row1_off + 0, row0_off + 2) += x20;
    info(row1_off + 1, row0_off + 2) += x21;
    info(row1_off + 2, row0_off + 2) += x22;
}

void buildGroupMessageConditionedInformationPacked(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const SyntheticSE2PackedBasisTopologyGroupPlan& plan,
    int total_dim,
    Eigen::MatrixXd& info
) {
    info.setZero(total_dim, total_dim);

    for (size_t i = 0; i < plan.prior_lam6_ptrs.size(); ++i) {
        addSymBlock3(info, plan.prior_row_offs[i], plan.prior_lam6_ptrs[i]);
    }
    for (const SyntheticSE2PackedBasisTopologyBoundaryPlan& msg : plan.boundary_msgs) {
        addSymBlock3(info, msg.row_off, msg.lam6_ptr);
    }
    for (const SyntheticSE2PackedBasisTopologyUnaryPlan& unary : plan.interior_unary) {
        addSymBlock3(info, unary.row_off, unary.lam6_ptr);
    }
    for (const SyntheticSE2PackedBasisTopologyBinaryPlan& binary : plan.interior_binary) {
        addSymBlock3(info, binary.row0_off, binary.diag0_lam6_ptr);
        addCrossBlock3AndTranspose(info, binary.row0_off, binary.row1_off, binary.cross01_lam9_ptr);
        addSymBlock3(info, binary.row1_off, binary.diag1_lam6_ptr);
    }
}

void buildGroupMessageConditionedInformationPacked(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const SyntheticSE2PackedBasisTopologyGroupPlan& plan,
    int total_dim,
    Eigen::MatrixXd& info
) {
    (void)workspace;
    info.setZero(total_dim, total_dim);

    for (size_t i = 0; i < plan.prior_lam6_ptrs.size(); ++i) {
        addSymBlock3(info, plan.prior_row_offs[i], plan.prior_lam6_ptrs[i]);
    }
    for (const SyntheticSE2PackedBasisTopologyBoundaryPlan& msg : plan.boundary_msgs) {
        addSymBlock3(info, msg.row_off, msg.lam6_ptr);
    }
    for (const SyntheticSE2PackedBasisTopologyUnaryPlan& unary : plan.interior_unary) {
        addSymBlock3(info, unary.row_off, unary.lam6_ptr);
    }
    for (const SyntheticSE2PackedBasisTopologyBinaryPlan& binary : plan.interior_binary) {
        addSymBlock3(info, binary.row0_off, binary.diag0_lam6_ptr);
        addCrossBlock3AndTranspose(info, binary.row0_off, binary.row1_off, binary.cross01_lam9_ptr);
        addSymBlock3(info, binary.row1_off, binary.diag1_lam6_ptr);
    }
}

}  // namespace

SyntheticSE2PackedBasisTopology buildMessageConditionedBasisPackedTopology(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    int group_size
) {
    SyntheticSE2PackedBasisTopology topology;
    const int n_vars = static_cast<int>(layout.variables.size());
    topology.groups = orderedGroupsPacked(n_vars, group_size);
    topology.total_dim = 3 * n_vars;
    topology.var_to_group.assign(n_vars, -1);
    topology.var_to_local_offset.assign(n_vars, -1);
    topology.full_indices_per_group.resize(topology.groups.size());
    topology.group_plans.resize(topology.groups.size());

    for (int g = 0; g < static_cast<int>(topology.groups.size()); ++g) {
        const std::vector<int>& group = topology.groups[g];
        std::vector<int> full_indices;
        full_indices.reserve(group.size() * 3);
        int local_scalar_offset = 0;
        for (int var_id : group) {
            const int base = 3 * var_id;
            full_indices.push_back(base + 0);
            full_indices.push_back(base + 1);
            full_indices.push_back(base + 2);
            topology.var_to_group[var_id] = g;
            topology.var_to_local_offset[var_id] = local_scalar_offset;
            local_scalar_offset += 3;
        }
        topology.full_indices_per_group[g] = std::move(full_indices);

        SyntheticSE2PackedBasisTopologyGroupPlan& plan = topology.group_plans[g];
        const int start_var_id = group.front();
        const int end_var_id = start_var_id + static_cast<int>(group.size());
        std::vector<int> binary_stamp(layout.binary_factors.size(), 0);
        std::vector<int> unary_stamp(layout.unary_factors.size(), 0);
        const int stamp_value = g + 1;

        for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
            const int var_id = group[local_idx];
            const int row_off = 3 * local_idx;
            plan.prior_lam6_ptrs.push_back(sym6Ptr(layout.prior_lam6, var_id));
            plan.prior_row_offs.push_back(row_off);

            const int unary_begin = layout.unary_offsets[var_id];
            const int unary_end = layout.unary_offsets[var_id + 1];
            for (int u = unary_begin; u < unary_end; ++u) {
                const int unary_id = layout.unary_ids[u];
                const int unary_var = layout.unary_var_ids[unary_id];
                if (unary_var >= start_var_id && unary_var < end_var_id) {
                    if (unary_stamp[unary_id] != stamp_value) {
                        unary_stamp[unary_id] = stamp_value;
                        plan.interior_unary.push_back({3 * (unary_var - start_var_id), sym6Ptr(layout.unary_lam6, unary_id)});
                    }
                } else {
                    plan.boundary_msgs.push_back({true, row_off, sym6Ptr(layout.unary_msg_lam6, unary_id)});
                }
            }

            const int binary_begin = layout.binary_offsets[var_id];
            const int binary_end = layout.binary_offsets[var_id + 1];
            for (int b = binary_begin; b < binary_end; ++b) {
                const int slot_id = layout.binary_slot_ids[b];
                const int binary_id = slot_id / 2;
                const SyntheticSE2PackedBinaryFactorData& factor = layout.binary_data[binary_id];
                const int other_id = (slot_id % 2 == 0) ? factor.var1_id : factor.var0_id;
                if (other_id >= start_var_id && other_id < end_var_id) {
                    if (binary_stamp[binary_id] != stamp_value) {
                        binary_stamp[binary_id] = stamp_value;
                        plan.interior_binary.push_back({
                            3 * (factor.var0_id - start_var_id),
                            3 * (factor.var1_id - start_var_id),
                            factor.diag0_lam6,
                            factor.diag1_lam6,
                            factor.cross01_lam9,
                        });
                    }
                } else {
                    plan.boundary_msgs.push_back({false, row_off, sym6Ptr(layout.binary_msg_lam6, slot_id)});
                }
            }
        }
    }

    return topology;
}

SyntheticSE2PackedBasisTopology buildMessageConditionedBasisPackedTopology(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int group_size
) {
    SyntheticSE2PackedBasisTopology topology;
    const int n_vars = workspace.num_vars;
    topology.groups = orderedGroupsPacked(n_vars, group_size);
    topology.total_dim = 3 * n_vars;
    topology.var_to_group.assign(n_vars, -1);
    topology.var_to_local_offset.assign(n_vars, -1);
    topology.full_indices_per_group.resize(topology.groups.size());
    topology.group_plans.resize(topology.groups.size());

    for (int g = 0; g < static_cast<int>(topology.groups.size()); ++g) {
        const std::vector<int>& group = topology.groups[g];
        std::vector<int> full_indices;
        full_indices.reserve(group.size() * 3);
        int local_scalar_offset = 0;
        for (int var_id : group) {
            const int base = 3 * var_id;
            full_indices.push_back(base + 0);
            full_indices.push_back(base + 1);
            full_indices.push_back(base + 2);
            topology.var_to_group[var_id] = g;
            topology.var_to_local_offset[var_id] = local_scalar_offset;
            local_scalar_offset += 3;
        }
        topology.full_indices_per_group[g] = std::move(full_indices);

        SyntheticSE2PackedBasisTopologyGroupPlan& plan = topology.group_plans[g];
        const int start_var_id = group.front();
        const int end_var_id = start_var_id + static_cast<int>(group.size());
        std::vector<int> binary_stamp(workspace.binary_factors.size(), 0);
        std::vector<int> unary_stamp(workspace.unary_factors.size(), 0);
        const int stamp_value = g + 1;

        for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
            const int var_id = group[local_idx];
            const int row_off = 3 * local_idx;
            plan.prior_lam6_ptrs.push_back(sym6Ptr(workspace.prior_lam6, var_id));
            plan.prior_row_offs.push_back(row_off);

            const int unary_begin = workspace.unary_offsets[var_id];
            const int unary_end = workspace.unary_offsets[var_id + 1];
            for (int u = unary_begin; u < unary_end; ++u) {
                const int unary_id = workspace.unary_ids[u];
                const int unary_var = workspace.unary_factors[unary_id].var_id;
                if (unary_var >= start_var_id && unary_var < end_var_id) {
                    if (unary_stamp[unary_id] != stamp_value) {
                        unary_stamp[unary_id] = stamp_value;
                        plan.interior_unary.push_back({
                            3 * (unary_var - start_var_id),
                            workspace.unary_factors[unary_id].lam6,
                        });
                    }
                } else {
                    plan.boundary_msgs.push_back({true, row_off, unaryMsgLam6Ptr(workspace, unary_id)});
                }
            }

            const int binary_begin = workspace.binary_offsets[var_id];
            const int binary_end = workspace.binary_offsets[var_id + 1];
            for (int b = binary_begin; b < binary_end; ++b) {
                const int slot_id = workspace.binary_slot_ids[b];
                const int binary_id = slot_id / 2;
                const SyntheticSE2PackedResidualBinaryFactor& factor = workspace.binary_factors[binary_id];
                const int other_id = (slot_id % 2 == 0) ? factor.var1_id : factor.var0_id;
                if (other_id >= start_var_id && other_id < end_var_id) {
                    if (binary_stamp[binary_id] != stamp_value) {
                        binary_stamp[binary_id] = stamp_value;
                        plan.interior_binary.push_back({
                            3 * (factor.var0_id - start_var_id),
                            3 * (factor.var1_id - start_var_id),
                            factor.diag0_lam6,
                            factor.diag1_lam6,
                            factor.cross01_lam9,
                        });
                    }
                } else {
                    plan.boundary_msgs.push_back({false, row_off, binaryMsgLam6Ptr(workspace, slot_id)});
                }
            }
        }
    }

    return topology;
}

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const SyntheticSE2PackedBasisTopology& topology,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    BasisData basis;
    basis.groups = topology.groups;
    basis.full_indices_per_group = topology.full_indices_per_group;
    basis.total_dim = topology.total_dim;
    basis.var_to_group = topology.var_to_group;
    basis.var_to_local_offset = topology.var_to_local_offset;
    basis.ordered_3d_fast = topology.ordered_3d_fast;
    basis.factors_arity12_only_3d = topology.factors_arity12_only_3d;
    basis.var_r_local.assign(layout.variables.size(), 0);
    basis.var_coarse_offset.assign(layout.variables.size(), 0);
    basis.var_basis_offset.assign(layout.variables.size(), -1);
    basis.local_bases.resize(basis.groups.size());

    std::vector<int> r_locals(basis.groups.size(), 0);
    int reduced_offset = 0;
    basis.coarse_offsets.clear();
    basis.coarse_offsets.push_back(0);
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
        const int r_local = std::min(r_reduced, block_dim);
        r_locals[g] = r_local;
        reduced_offset += r_local;
        basis.coarse_offsets.push_back(reduced_offset);
    }

    size_t total_basis_scalars = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        total_basis_scalars += static_cast<size_t>(basis.groups[g].size()) * 3 * r_locals[g];
    }
    basis.var_basis_blocks.reserve(total_basis_scalars);

    const bool use_partial_eigensolver = basis_build_config.eigensolver == BasisEigensolverKind::Partial;
    const PartialSymmetricEigenOptions partial_opts{
        basis_build_config.partial_oversampling,
        basis_build_config.partial_max_iters,
        basis_build_config.partial_residual_tol,
        basis_build_config.partial_ridge,
        basis_build_config.partial_residual_check_period,
    };

    const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
    const bool do_parallel = use_parallel && thread_count > 1 && basis.groups.size() >= 8;
    double block_build_sec = 0.0;
    double eigensolver_sec = 0.0;

    if (do_parallel) {
        std::vector<PartialSymmetricEigenWorkspace> tls_partial_ws(thread_count);
        std::vector<Eigen::MatrixXd> tls_blocks(thread_count);
        std::vector<Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>> tls_full_es(thread_count);
        std::vector<double> tls_block_build_sec(thread_count, 0.0);
        std::vector<double> tls_eigensolver_sec(thread_count, 0.0);
        std::vector<int> tls_partial_attempted(thread_count, 0);
        std::vector<int> tls_partial_converged(thread_count, 0);
        std::vector<int> tls_partial_fallback(thread_count, 0);
        std::vector<int> tls_partial_total_iters(thread_count, 0);
        auto build_group = [&](int tid, int begin, int end) {
            for (int g = begin; g < end; ++g) {
                const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
                const int r_local = r_locals[g];
                Eigen::MatrixXd& block = tls_blocks[tid];
                const double block_t0 = omp_get_wtime();
                buildGroupMessageConditionedInformationPacked(layout, topology.group_plans[g], block_dim, block);
                tls_block_build_sec[tid] += omp_get_wtime() - block_t0;

                Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
                if (r_local < block_dim) {
                    const Eigen::MatrixXd* warm_basis =
                        (use_partial_eigensolver &&
                         basis_build_config.enable_warm_start &&
                         warm_start_local_bases &&
                         g < static_cast<int>(warm_start_local_bases->size()) &&
                         (*warm_start_local_bases)[g].rows() == block_dim &&
                         (*warm_start_local_bases)[g].cols() == r_local)
                        ? &(*warm_start_local_bases)[g]
                        : nullptr;

                    bool used_partial = false;
                    if (use_partial_eigensolver) {
                        const double eig_t0 = omp_get_wtime();
                        const PartialSymmetricEigenResult partial =
                            computeSmallestEigenpairsPartial(block, r_local, warm_basis, tls_partial_ws[tid], partial_opts);
                        tls_eigensolver_sec[tid] += omp_get_wtime() - eig_t0;
                        tls_partial_attempted[tid] += 1;
                        tls_partial_total_iters[tid] += partial.iterations;
                        if (partial.converged &&
                            partial.eigenvectors.rows() == block_dim &&
                            partial.eigenvectors.cols() == r_local) {
                            local_basis = partial.eigenvectors;
                            used_partial = true;
                            tls_partial_converged[tid] += 1;
                        } else {
                            tls_partial_fallback[tid] += 1;
                        }
                    }
                    if (!used_partial) {
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& es = tls_full_es[tid];
                        const double eig_t0 = omp_get_wtime();
                        es.compute(block);
                        tls_eigensolver_sec[tid] += omp_get_wtime() - eig_t0;
                        if (es.info() != Eigen::Success) {
                            throw std::runtime_error("Failed to eigendecompose packed conditioned information block");
                        }
                        local_basis = es.eigenvectors().leftCols(r_local);
                    }
                }
                basis.local_bases[g] = std::move(local_basis);
            }
        };
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            build_group(omp_get_thread_num(), g, g + 1);
        }
        for (int tid = 0; tid < thread_count; ++tid) {
            basis.partial_attempted += tls_partial_attempted[tid];
            basis.partial_converged += tls_partial_converged[tid];
            basis.partial_fallback += tls_partial_fallback[tid];
            basis.partial_total_iters += tls_partial_total_iters[tid];
        }
        block_build_sec = *std::max_element(tls_block_build_sec.begin(), tls_block_build_sec.end());
        eigensolver_sec = *std::max_element(tls_eigensolver_sec.begin(), tls_eigensolver_sec.end());
    } else {
        PartialSymmetricEigenWorkspace partial_ws;
        Eigen::MatrixXd block;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full_es;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            const int r_local = r_locals[g];
            const double block_t0 = omp_get_wtime();
            buildGroupMessageConditionedInformationPacked(layout, topology.group_plans[g], block_dim, block);
            block_build_sec += omp_get_wtime() - block_t0;

            Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
            if (r_local < block_dim) {
                const Eigen::MatrixXd* warm_basis =
                    (use_partial_eigensolver &&
                     basis_build_config.enable_warm_start &&
                     warm_start_local_bases &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() == r_local)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                bool used_partial = false;
                if (use_partial_eigensolver) {
                    const double eig_t0 = omp_get_wtime();
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
                    eigensolver_sec += omp_get_wtime() - eig_t0;
                    basis.partial_attempted += 1;
                    basis.partial_total_iters += partial.iterations;
                    if (partial.converged &&
                        partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local) {
                        local_basis = partial.eigenvectors;
                        used_partial = true;
                        basis.partial_converged += 1;
                    } else {
                        basis.partial_fallback += 1;
                    }
                }
                if (!used_partial) {
                    const double eig_t0 = omp_get_wtime();
                    full_es.compute(block);
                    eigensolver_sec += omp_get_wtime() - eig_t0;
                    if (full_es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose packed conditioned information block");
                    }
                    local_basis = full_es.eigenvectors().leftCols(r_local);
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    }

    size_t basis_scalar_offset = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        const int r_local = basis.local_bases[g].cols();
        const int coarse_offset = basis.coarse_offsets[g];
        for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
            const int var_id = group[local_var];
            basis.var_r_local[var_id] = r_local;
            basis.var_coarse_offset[var_id] = coarse_offset;
            basis.var_basis_offset[var_id] = static_cast<int>(basis_scalar_offset);
            basis_scalar_offset += static_cast<size_t>(3 * r_local);
        }
    }
    basis.var_basis_blocks.resize(basis_scalar_offset);

    const double copyout_t0 = omp_get_wtime();
    if (do_parallel) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const Eigen::MatrixXd& local_basis = basis.local_bases[g];
            const int r_local = local_basis.cols();
            for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
                const int var_id = group[local_var];
                double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
                const int row0 = 3 * local_var;
                for (int c = 0; c < r_local; ++c) {
                    dst[3 * c + 0] = local_basis(row0 + 0, c);
                    dst[3 * c + 1] = local_basis(row0 + 1, c);
                    dst[3 * c + 2] = local_basis(row0 + 2, c);
                }
            }
        }
    } else {
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const Eigen::MatrixXd& local_basis = basis.local_bases[g];
            const int r_local = local_basis.cols();
            for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
                const int var_id = group[local_var];
                double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
                const int row0 = 3 * local_var;
                for (int c = 0; c < r_local; ++c) {
                    dst[3 * c + 0] = local_basis(row0 + 0, c);
                    dst[3 * c + 1] = local_basis(row0 + 1, c);
                    dst[3 * c + 2] = local_basis(row0 + 2, c);
                }
            }
        }
    }
    basis.copyout_sec = omp_get_wtime() - copyout_t0;

    basis.coarse_dim = reduced_offset;
    basis.block_build_sec = block_build_sec;
    basis.eigensolver_sec = eigensolver_sec;
    return basis;
}

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    const SyntheticSE2PackedBasisTopology topology =
        buildMessageConditionedBasisPackedTopology(layout, group_size);
    return buildMessageConditionedBasisPacked(
        layout,
        topology,
        r_reduced,
        use_parallel,
        num_threads,
        basis_build_config,
        warm_start_local_bases
    );
}

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const SyntheticSE2PackedBasisTopology& topology,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    BasisData basis;
    basis.groups = topology.groups;
    basis.full_indices_per_group = topology.full_indices_per_group;
    basis.total_dim = topology.total_dim;
    basis.var_to_group = topology.var_to_group;
    basis.var_to_local_offset = topology.var_to_local_offset;
    basis.ordered_3d_fast = topology.ordered_3d_fast;
    basis.factors_arity12_only_3d = topology.factors_arity12_only_3d;
    basis.var_r_local.assign(workspace.num_vars, 0);
    basis.var_coarse_offset.assign(workspace.num_vars, 0);
    basis.var_basis_offset.assign(workspace.num_vars, -1);
    basis.local_bases.resize(basis.groups.size());

    std::vector<int> r_locals(basis.groups.size(), 0);
    int reduced_offset = 0;
    basis.coarse_offsets.clear();
    basis.coarse_offsets.push_back(0);
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
        const int r_local = std::min(r_reduced, block_dim);
        r_locals[g] = r_local;
        reduced_offset += r_local;
        basis.coarse_offsets.push_back(reduced_offset);
    }

    size_t total_basis_scalars = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        total_basis_scalars += static_cast<size_t>(basis.groups[g].size()) * 3 * r_locals[g];
    }
    basis.var_basis_blocks.reserve(total_basis_scalars);

    const bool use_partial_eigensolver = basis_build_config.eigensolver == BasisEigensolverKind::Partial;
    const PartialSymmetricEigenOptions partial_opts{
        basis_build_config.partial_oversampling,
        basis_build_config.partial_max_iters,
        basis_build_config.partial_residual_tol,
        basis_build_config.partial_ridge,
        basis_build_config.partial_residual_check_period,
    };

    const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
    const bool do_parallel = use_parallel && thread_count > 1 && basis.groups.size() >= 8;
    double block_build_sec = 0.0;
    double eigensolver_sec = 0.0;

    if (do_parallel) {
        std::vector<PartialSymmetricEigenWorkspace> tls_partial_ws(thread_count);
        std::vector<Eigen::MatrixXd> tls_blocks(thread_count);
        std::vector<Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>> tls_full_es(thread_count);
        std::vector<double> tls_block_build_sec(thread_count, 0.0);
        std::vector<double> tls_eigensolver_sec(thread_count, 0.0);
        std::vector<int> tls_partial_attempted(thread_count, 0);
        std::vector<int> tls_partial_converged(thread_count, 0);
        std::vector<int> tls_partial_fallback(thread_count, 0);
        std::vector<int> tls_partial_total_iters(thread_count, 0);
        auto build_group = [&](int tid, int begin, int end) {
            for (int g = begin; g < end; ++g) {
                const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
                const int r_local = r_locals[g];
                Eigen::MatrixXd& block = tls_blocks[tid];
                const double block_t0 = omp_get_wtime();
                buildGroupMessageConditionedInformationPacked(workspace, topology.group_plans[g], block_dim, block);
                tls_block_build_sec[tid] += omp_get_wtime() - block_t0;

                Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
                if (r_local < block_dim) {
                    const Eigen::MatrixXd* warm_basis =
                        (use_partial_eigensolver &&
                         basis_build_config.enable_warm_start &&
                         warm_start_local_bases &&
                         g < static_cast<int>(warm_start_local_bases->size()) &&
                         (*warm_start_local_bases)[g].rows() == block_dim &&
                         (*warm_start_local_bases)[g].cols() == r_local)
                        ? &(*warm_start_local_bases)[g]
                        : nullptr;
                    bool used_partial = false;
                    if (use_partial_eigensolver) {
                        const double eig_t0 = omp_get_wtime();
                        const PartialSymmetricEigenResult partial =
                            computeSmallestEigenpairsPartial(block, r_local, warm_basis, tls_partial_ws[tid], partial_opts);
                        tls_eigensolver_sec[tid] += omp_get_wtime() - eig_t0;
                        tls_partial_attempted[tid] += 1;
                        tls_partial_total_iters[tid] += partial.iterations;
                        if (partial.converged &&
                            partial.eigenvectors.rows() == block_dim &&
                            partial.eigenvectors.cols() == r_local) {
                            local_basis = partial.eigenvectors;
                            used_partial = true;
                            tls_partial_converged[tid] += 1;
                        } else {
                            tls_partial_fallback[tid] += 1;
                        }
                    }
                    if (!used_partial) {
                        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& es = tls_full_es[tid];
                        const double eig_t0 = omp_get_wtime();
                        es.compute(block);
                        tls_eigensolver_sec[tid] += omp_get_wtime() - eig_t0;
                        if (es.info() != Eigen::Success) {
                            throw std::runtime_error("Failed to eigendecompose packed residual conditioned information block");
                        }
                        local_basis = es.eigenvectors().leftCols(r_local);
                    }
                }
                basis.local_bases[g] = std::move(local_basis);
            }
        };
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            build_group(omp_get_thread_num(), g, g + 1);
        }
        for (int tid = 0; tid < thread_count; ++tid) {
            basis.partial_attempted += tls_partial_attempted[tid];
            basis.partial_converged += tls_partial_converged[tid];
            basis.partial_fallback += tls_partial_fallback[tid];
            basis.partial_total_iters += tls_partial_total_iters[tid];
        }
        block_build_sec = *std::max_element(tls_block_build_sec.begin(), tls_block_build_sec.end());
        eigensolver_sec = *std::max_element(tls_eigensolver_sec.begin(), tls_eigensolver_sec.end());
    } else {
        PartialSymmetricEigenWorkspace partial_ws;
        Eigen::MatrixXd block;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full_es;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            const int r_local = r_locals[g];
            const double block_t0 = omp_get_wtime();
            buildGroupMessageConditionedInformationPacked(workspace, topology.group_plans[g], block_dim, block);
            block_build_sec += omp_get_wtime() - block_t0;

            Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
            if (r_local < block_dim) {
                const Eigen::MatrixXd* warm_basis =
                    (use_partial_eigensolver &&
                     basis_build_config.enable_warm_start &&
                     warm_start_local_bases &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() == r_local)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                bool used_partial = false;
                if (use_partial_eigensolver) {
                    const double eig_t0 = omp_get_wtime();
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
                    eigensolver_sec += omp_get_wtime() - eig_t0;
                    basis.partial_attempted += 1;
                    basis.partial_total_iters += partial.iterations;
                    if (partial.converged &&
                        partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local) {
                        local_basis = partial.eigenvectors;
                        used_partial = true;
                        basis.partial_converged += 1;
                    } else {
                        basis.partial_fallback += 1;
                    }
                }
                if (!used_partial) {
                    const double eig_t0 = omp_get_wtime();
                    full_es.compute(block);
                    eigensolver_sec += omp_get_wtime() - eig_t0;
                    if (full_es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose packed residual conditioned information block");
                    }
                    local_basis = full_es.eigenvectors().leftCols(r_local);
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    }

    size_t basis_scalar_offset = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        const int r_local = basis.local_bases[g].cols();
        const int coarse_offset = basis.coarse_offsets[g];
        for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
            const int var_id = group[local_var];
            basis.var_r_local[var_id] = r_local;
            basis.var_coarse_offset[var_id] = coarse_offset;
            basis.var_basis_offset[var_id] = static_cast<int>(basis_scalar_offset);
            basis_scalar_offset += static_cast<size_t>(3 * r_local);
        }
    }
    basis.var_basis_blocks.resize(basis_scalar_offset);

    const double copyout_t0 = omp_get_wtime();
    if (do_parallel) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const Eigen::MatrixXd& local_basis = basis.local_bases[g];
            const int r_local = local_basis.cols();
            for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
                const int var_id = group[local_var];
                double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
                const int row0 = 3 * local_var;
                for (int c = 0; c < r_local; ++c) {
                    dst[3 * c + 0] = local_basis(row0 + 0, c);
                    dst[3 * c + 1] = local_basis(row0 + 1, c);
                    dst[3 * c + 2] = local_basis(row0 + 2, c);
                }
            }
        }
    } else {
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const Eigen::MatrixXd& local_basis = basis.local_bases[g];
            const int r_local = local_basis.cols();
            for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
                const int var_id = group[local_var];
                double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
                const int row0 = 3 * local_var;
                for (int c = 0; c < r_local; ++c) {
                    dst[3 * c + 0] = local_basis(row0 + 0, c);
                    dst[3 * c + 1] = local_basis(row0 + 1, c);
                    dst[3 * c + 2] = local_basis(row0 + 2, c);
                }
            }
        }
    }
    basis.copyout_sec = omp_get_wtime() - copyout_t0;

    basis.coarse_dim = reduced_offset;
    basis.block_build_sec = block_build_sec;
    basis.eigensolver_sec = eigensolver_sec;
    return basis;
}

BasisData buildMessageConditionedBasisPacked(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    const SyntheticSE2PackedBasisTopology topology =
        buildMessageConditionedBasisPackedTopology(workspace, group_size);
    return buildMessageConditionedBasisPacked(
        workspace,
        topology,
        r_reduced,
        use_parallel,
        num_threads,
        basis_build_config,
        warm_start_local_bases
    );
}

}  // namespace slam
