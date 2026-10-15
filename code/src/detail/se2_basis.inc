Eigen::VectorXd stackedMeanVector(gbp::FactorGraph& graph) {
    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    Eigen::VectorXd out = Eigen::VectorXd::Zero(total_dim);
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        vup->refreshMu();
        out.segment(offset, vup->dofs) = vup->mu;
        offset += vup->dofs;
    }
    return out;
}

void injectCorrectionKeepMessages(gbp::FactorGraph& graph, const Eigen::VectorXd& delta) {
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        gbp::VariableNode& var = *vup;
        var.refreshMu();
        var.mu += delta.segment(offset, var.dofs);
        var.belief.setEta(var.belief.lam() * var.mu);
        var.markMuCurrent();
        offset += var.dofs;
    }
}

Eigen::SparseMatrix<double> symmetrizeSparse(const Eigen::SparseMatrix<double>& A) {
    Eigen::SparseMatrix<double> sym = 0.5 * (A + Eigen::SparseMatrix<double>(A.transpose()));
    sym.makeCompressed();
    return sym;
}

Eigen::VectorXd solveSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    const Eigen::VectorXd& eta,
    double ridge
) {
    if (lam.rows() == 0) {
        return Eigen::VectorXd::Zero(0);
    }

    Eigen::SparseMatrix<double> A = symmetrizeSparse(lam);
    if (ridge != 0.0) {
        for (int i = 0; i < A.rows(); ++i) {
            A.coeffRef(i, i) += ridge;
        }
    }
    A.makeCompressed();

    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }

    const Eigen::VectorXd x = solver.solve(eta);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
    return x;
}

struct SparseCholeskyFactor {
    Eigen::SparseMatrix<double> A;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    bool analyzed = false;
    int rows = 0;
    int cols = 0;
    int nonzeros = 0;

    SparseCholeskyFactor() = default;
    SparseCholeskyFactor(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor& operator=(const SparseCholeskyFactor&) = delete;
    SparseCholeskyFactor(SparseCholeskyFactor&&) noexcept = default;
    SparseCholeskyFactor& operator=(SparseCholeskyFactor&&) noexcept = default;
};


void factorizeSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    double ridge,
    SparseCholeskyFactor& factor,
    bool reuse_pattern = false,
    bool assume_symmetric = false
) {
    factor.A = assume_symmetric ? lam : symmetrizeSparse(lam);
    if (ridge != 0.0) {
        for (int i = 0; i < factor.A.rows(); ++i) {
            factor.A.coeffRef(i, i) += ridge;
        }
    }
    factor.A.makeCompressed();

    const bool can_reuse =
        reuse_pattern &&
        factor.analyzed &&
        factor.rows == factor.A.rows() &&
        factor.cols == factor.A.cols() &&
        factor.nonzeros == factor.A.nonZeros();

    if (!can_reuse) {
        factor.solver.analyzePattern(factor.A);
        if (factor.solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky analyzePattern failed");
        }
        factor.analyzed = true;
        factor.rows = static_cast<int>(factor.A.rows());
        factor.cols = static_cast<int>(factor.A.cols());
        factor.nonzeros = static_cast<int>(factor.A.nonZeros());
    }

    factor.solver.factorize(factor.A);
    if (factor.solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }
}

void solveWithSparseCholeskyInto(
    const SparseCholeskyFactor& factor,
    const Eigen::VectorXd& eta,
    Eigen::VectorXd& out
) {
    out = factor.solver.solve(eta);
    if (factor.solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
}

std::vector<std::vector<int>> orderedGroups(int n_vars, int group_size) {
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

Eigen::MatrixXd buildGroupMessageConditionedInformationOrdered3D(
    const gbp::FactorGraph& graph,
    int start_var_id,
    int group_len,
    std::vector<int>& factor_stamp,
    int stamp_value
) {
    const int total_dim = 3 * group_len;
    const int end_var_id = start_var_id + group_len;
    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(total_dim, total_dim);

    auto add_block3 = [&](int row_off, int col_off, const double* block) {
        for (int c = 0; c < 3; ++c) {
            for (int r = 0; r < 3; ++r) {
                info(row_off + r, col_off + c) += block[3 * c + r];
            }
        }
    };

    auto add_factor_block3 = [&](int row_off, int col_off, const double* factor_lam, int factor_dim, int factor_row, int factor_col) {
        for (int c = 0; c < 3; ++c) {
            const double* src = factor_lam + (factor_col + c) * factor_dim + factor_row;
            info(row_off + 0, col_off + c) += src[0];
            info(row_off + 1, col_off + c) += src[1];
            info(row_off + 2, col_off + c) += src[2];
        }
    };

    for (int local_idx = 0; local_idx < group_len; ++local_idx) {
        const int var_id = start_var_id + local_idx;
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        const int off = 3 * local_idx;
        add_block3(off, off, var->prior.lamData());

        for (const auto& aref : var->adj_factors) {
            const gbp::Factor* factor = aref.factor;
            if (!factor || !factor->active) {
                continue;
            }

            bool factor_inside = true;
            for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                const int adj_id = adj_var->variableID;
                if (adj_id < start_var_id || adj_id >= end_var_id) {
                    factor_inside = false;
                    break;
                }
            }

            if (factor_inside) {
                const int fid = factor->factorID;
                if (factor_stamp[fid] == stamp_value) {
                    continue;
                }
                factor_stamp[fid] = stamp_value;

                const double* factor_lam = factor->factor.lamData();
                const int arity = static_cast<int>(factor->adj_var_nodes.size());
                for (int a = 0; a < arity; ++a) {
                    const int row_off = 3 * (factor->adj_var_nodes[a]->variableID - start_var_id);
                    const int factor_row = 3 * a;
                    for (int b = 0; b < arity; ++b) {
                        const int col_off = 3 * (factor->adj_var_nodes[b]->variableID - start_var_id);
                        const int factor_col = 3 * b;
                        add_factor_block3(row_off, col_off, factor_lam, factor->factor.dim(), factor_row, factor_col);
                    }
                }
            } else {
                add_block3(off, off, factor->messages[aref.local_idx].lamData());
            }
        }
    }

    return 0.5 * (info + info.transpose());
}

void buildGroupMessageConditionedInformationOrdered3DFastPlan(
    const BasisData::GroupFastPlan& plan,
    int total_dim,
    Eigen::MatrixXd& info
) {
    info.setZero(total_dim, total_dim);

    auto add_block3 = [&](int row_off, int col_off, const double* block) {
        for (int c = 0; c < 3; ++c) {
            info(row_off + 0, col_off + c) += block[3 * c + 0];
            info(row_off + 1, col_off + c) += block[3 * c + 1];
            info(row_off + 2, col_off + c) += block[3 * c + 2];
        }
    };

    auto add_factor_block3 = [&](int row_off, int col_off, const double* src, int dim, int row0, int col0) {
        for (int c = 0; c < 3; ++c) {
            const double* src_col = src + (col0 + c) * dim + row0;
            info(row_off + 0, col_off + c) += src_col[0];
            info(row_off + 1, col_off + c) += src_col[1];
            info(row_off + 2, col_off + c) += src_col[2];
        }
    };

    for (size_t i = 0; i < plan.prior_lam_ptrs.size(); ++i) {
        add_block3(plan.prior_row_offs[i], plan.prior_row_offs[i], plan.prior_lam_ptrs[i]);
    }
    for (const BasisData::GroupBoundary3Plan& msg : plan.boundary_msgs) {
        add_block3(msg.row_off, msg.row_off, msg.factor->messages[msg.local_idx].lamData());
    }
    for (const BasisData::GroupUnary3Plan& unary : plan.interior_unary) {
        add_block3(unary.row_off, unary.row_off, unary.factor->factor.lamData());
    }
    for (const BasisData::GroupBinary3Plan& binary : plan.interior_binary) {
        const double* lam = binary.factor->factor.lamData();
        add_factor_block3(binary.row0_off, binary.row0_off, lam, 6, 0, 0);
        add_factor_block3(binary.row0_off, binary.row1_off, lam, 6, 0, 3);
        add_factor_block3(binary.row1_off, binary.row0_off, lam, 6, 3, 0);
        add_factor_block3(binary.row1_off, binary.row1_off, lam, 6, 3, 3);
    }

    for (int c = 0; c < total_dim; ++c) {
        for (int r = 0; r < c; ++r) {
            const double sym = 0.5 * (info(r, c) + info(c, r));
            info(r, c) = sym;
            info(c, r) = sym;
        }
    }
}

BasisData buildMessageConditionedBasis(
    const gbp::FactorGraph& graph,
    int group_size,
    int r_reduced,
    bool use_parallel,
    int num_threads,
    const BasisBuildConfig& basis_build_config,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases
) {
    BasisData basis;
    basis.groups = orderedGroups(static_cast<int>(graph.var_nodes.size()), group_size);

    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    basis.total_dim = total_dim;

    basis.coarse_offsets.push_back(0);
    basis.var_to_group.assign(graph.var_nodes.size(), -1);
    basis.var_to_local_offset.assign(graph.var_nodes.size(), -1);
    basis.ordered_3d_fast = true;
    basis.factors_arity12_only_3d = true;
    basis.var_r_local.assign(graph.var_nodes.size(), 0);
    basis.var_coarse_offset.assign(graph.var_nodes.size(), 0);
    basis.var_basis_offset.assign(graph.var_nodes.size(), -1);
    basis.full_indices_per_group.resize(basis.groups.size());
    basis.local_bases.resize(basis.groups.size());
    basis.group_fast_plans.resize(basis.groups.size());
    std::vector<int> r_locals(basis.groups.size(), 0);
    int reduced_offset = 0;

    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        std::vector<int> full_indices;
        full_indices.reserve(group.size() * 3);
        int local_scalar_offset = 0;
        for (int var_id : group) {
            const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
            const int base = 3 * var_id;
            for (int d = 0; d < var->dofs; ++d) {
                full_indices.push_back(base + d);
            }
            basis.var_to_group[var_id] = g;
            basis.var_to_local_offset[var_id] = local_scalar_offset;
            if (var->dofs != 3) {
                basis.ordered_3d_fast = false;
            }
            local_scalar_offset += var->dofs;
        }
        const int block_dim = static_cast<int>(full_indices.size());
        const int r_local = std::min(r_reduced, block_dim);
        basis.full_indices_per_group[g] = std::move(full_indices);
        r_locals[g] = r_local;
        reduced_offset += r_local;
        basis.coarse_offsets.push_back(reduced_offset);
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor) {
            continue;
        }
        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity < 1 || arity > 2) {
            basis.factors_arity12_only_3d = false;
            break;
        }
        for (const gbp::VariableNode* var : factor->adj_var_nodes) {
            if (!var || var->dofs != 3) {
                basis.factors_arity12_only_3d = false;
                break;
            }
        }
        if (!basis.factors_arity12_only_3d) {
            break;
        }
    }

    if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            BasisData::GroupFastPlan& plan = basis.group_fast_plans[g];
            const std::vector<int>& group = basis.groups[g];
            const int start_var_id = group.front();
            const int end_var_id = start_var_id + static_cast<int>(group.size());
            ++stamp_value;

            for (int local_idx = 0; local_idx < static_cast<int>(group.size()); ++local_idx) {
                const int var_id = group[local_idx];
                const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
                const int row_off = 3 * local_idx;
                plan.prior_lam_ptrs.push_back(var->prior.lamData());
                plan.prior_row_offs.push_back(row_off);

                for (const auto& aref : var->adj_factors) {
                    const gbp::Factor* factor = aref.factor;
                    if (!factor) {
                        continue;
                    }

                    bool factor_inside = true;
                    for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                        const int adj_id = adj_var->variableID;
                        if (adj_id < start_var_id || adj_id >= end_var_id) {
                            factor_inside = false;
                            break;
                        }
                    }

                    if (factor_inside) {
                        const int fid = factor->factorID;
                        if (fid >= 0 && fid < static_cast<int>(factor_stamp.size()) && factor_stamp[fid] == stamp_value) {
                            continue;
                        }
                        if (fid >= 0 && fid < static_cast<int>(factor_stamp.size())) {
                            factor_stamp[fid] = stamp_value;
                        }
                        const int arity = static_cast<int>(factor->adj_var_nodes.size());
                        if (arity == 1) {
                            plan.interior_unary.push_back({factor, row_off});
                        } else if (arity == 2) {
                            const int id0 = factor->adj_var_nodes[0]->variableID;
                            const int id1 = factor->adj_var_nodes[1]->variableID;
                            const int row0 = 3 * (id0 - start_var_id);
                            const int row1 = 3 * (id1 - start_var_id);
                            plan.interior_binary.push_back({factor, row0, row1});
                        }
                    } else {
                        plan.boundary_msgs.push_back({factor, aref.local_idx, row_off});
                    }
                }
            }
        }
    }

    size_t total_basis_scalars = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        total_basis_scalars += static_cast<size_t>(basis.groups[g].size()) * 3 * r_locals[g];
    }
    basis.var_basis_blocks.reserve(total_basis_scalars);

    const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
    const bool do_parallel = use_parallel && thread_count > 1 && basis.groups.size() >= 8;
    const bool use_partial_eigensolver = basis_build_config.eigensolver == BasisEigensolverKind::Partial;
    const PartialSymmetricEigenOptions partial_opts{
        basis_build_config.partial_oversampling,
        basis_build_config.partial_max_iters,
        basis_build_config.partial_residual_tol,
        basis_build_config.partial_ridge,
        basis_build_config.partial_residual_check_period,
    };
    if (do_parallel) {
        std::vector<std::vector<int>> tls_factor_stamp(
            thread_count,
            std::vector<int>(graph.factors.size(), -1)
        );
        std::vector<int> tls_stamp_value(thread_count, 0);
        std::vector<PartialSymmetricEigenWorkspace> tls_partial_ws(thread_count);

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const int tid = omp_get_thread_num();
            const std::vector<int>& group = basis.groups[g];
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            Eigen::MatrixXd block;
            if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
                buildGroupMessageConditionedInformationOrdered3DFastPlan(
                    basis.group_fast_plans[g],
                    block_dim,
                    block
                );
            } else {
                block = buildGroupMessageConditionedInformationOrdered3D(
                    graph,
                    group.front(),
                    static_cast<int>(group.size()),
                    tls_factor_stamp[tid],
                    ++tls_stamp_value[tid]
                );
            }
            const int r_local = r_locals[g];

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
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, tls_partial_ws[tid], partial_opts);
                    #pragma omp atomic
                    basis.partial_attempted += 1;
                    #pragma omp atomic
                    basis.partial_total_iters += partial.iterations;
                    if (partial.converged &&
                        partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local) {
                        local_basis = partial.eigenvectors;
                        used_partial = true;
                        #pragma omp atomic
                        basis.partial_converged += 1;
                    } else {
                        #pragma omp atomic
                        basis.partial_fallback += 1;
                    }
                }
                if (!used_partial) {
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(block);
                    if (es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose conditioned information block");
                    }
                    const Eigen::MatrixXd evecs = es.eigenvectors();
                    local_basis.resize(block_dim, r_local);
                    for (int col = 0; col < r_local; ++col) {
                        local_basis.col(col) = evecs.col(col);
                    }
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    } else {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        PartialSymmetricEigenWorkspace partial_ws;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            const std::vector<int>& group = basis.groups[g];
            const int block_dim = static_cast<int>(basis.full_indices_per_group[g].size());
            Eigen::MatrixXd block;
            if (basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
                buildGroupMessageConditionedInformationOrdered3DFastPlan(
                    basis.group_fast_plans[g],
                    block_dim,
                    block
                );
            } else {
                block = buildGroupMessageConditionedInformationOrdered3D(
                    graph,
                    group.front(),
                    static_cast<int>(group.size()),
                    factor_stamp,
                    ++stamp_value
                );
            }
            const int r_local = r_locals[g];

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
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
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
                    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(block);
                    if (es.info() != Eigen::Success) {
                        throw std::runtime_error("Failed to eigendecompose conditioned information block");
                    }
                    const Eigen::MatrixXd evecs = es.eigenvectors();
                    local_basis.resize(block_dim, r_local);
                    for (int col = 0; col < r_local; ++col) {
                        local_basis.col(col) = evecs.col(col);
                    }
                }
            }
            basis.local_bases[g] = std::move(local_basis);
        }
    }

    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const std::vector<int>& group = basis.groups[g];
        const Eigen::MatrixXd& local_basis = basis.local_bases[g];
        const int r_local = local_basis.cols();
        for (int local_var = 0; local_var < static_cast<int>(group.size()); ++local_var) {
            const int var_id = group[local_var];
            basis.var_r_local[var_id] = r_local;
            basis.var_coarse_offset[var_id] = basis.coarse_offsets[g];
            basis.var_basis_offset[var_id] = static_cast<int>(basis.var_basis_blocks.size());
            basis.var_basis_blocks.resize(basis.var_basis_blocks.size() + 3 * r_local);
            double* dst = basis.var_basis_blocks.data() + basis.var_basis_offset[var_id];
            const int row0 = 3 * local_var;
            for (int c = 0; c < r_local; ++c) {
                dst[3 * c + 0] = local_basis(row0 + 0, c);
                dst[3 * c + 1] = local_basis(row0 + 1, c);
                dst[3 * c + 2] = local_basis(row0 + 2, c);
            }
        }
    }

    basis.coarse_dim = reduced_offset;
    return basis;
}
