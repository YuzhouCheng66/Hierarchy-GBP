struct JointInfSparsePatternCache {
    struct DenseBlockPlan {
        int row_offset = 0;
        int col_offset = 0;
        int rows = 0;
        int cols = 0;
        int factor_row = 0;
        int factor_col = 0;
        std::vector<int> value_indices;
    };

    struct EtaSegmentPlan {
        int global_offset = 0;
        int factor_offset = 0;
        int dofs = 0;
    };

    struct FactorPlan {
        int factor_index = -1;
        std::vector<EtaSegmentPlan> eta_segments;
        std::vector<DenseBlockPlan> blocks;
    };

    struct SymValuePair {
        int first = -1;
        int second = -1;
    };

    int total_dim = 0;
    int num_vars = 0;
    int num_factors = 0;
    std::vector<int> var_ix;
    Eigen::SparseMatrix<double> lam;
    std::unordered_map<long long, int> coeff_to_value_index;
    std::vector<DenseBlockPlan> prior_blocks;
    std::vector<FactorPlan> factor_plans;
    std::vector<SymValuePair> sym_value_pairs;
    bool valid = false;
};

long long sparseCoeffKey(int row, int col) {
    return (static_cast<long long>(row) << 32) ^
        static_cast<unsigned int>(col);
}

JointInfSparsePatternCache::DenseBlockPlan makeDenseBlockPlan(
    const JointInfSparsePatternCache& cache,
    int row_offset,
    int col_offset,
    int rows,
    int cols,
    int factor_row = 0,
    int factor_col = 0
) {
    JointInfSparsePatternCache::DenseBlockPlan plan;
    plan.row_offset = row_offset;
    plan.col_offset = col_offset;
    plan.rows = rows;
    plan.cols = cols;
    plan.factor_row = factor_row;
    plan.factor_col = factor_col;
    plan.value_indices.reserve(static_cast<size_t>(rows * cols));
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            const auto it = cache.coeff_to_value_index.find(
                sparseCoeffKey(row_offset + r, col_offset + c)
            );
            if (it == cache.coeff_to_value_index.end()) {
                throw std::runtime_error("cached joint plan missing sparse coefficient");
            }
            plan.value_indices.push_back(it->second);
        }
    }
    return plan;
}

template <typename DenseLike>
void addDenseBlockByPlan(
    double* values,
    const JointInfSparsePatternCache::DenseBlockPlan& plan,
    const DenseLike& block
) {
    size_t idx = 0;
    for (int c = 0; c < plan.cols; ++c) {
        for (int r = 0; r < plan.rows; ++r) {
            values[plan.value_indices[idx++]] +=
                block(plan.factor_row + r, plan.factor_col + c);
        }
    }
}

void buildJointInfSparsePatternCache(
    const gbp::FactorGraph& graph,
    JointInfSparsePatternCache& cache
) {
    cache = JointInfSparsePatternCache{};
    int max_id = -1;
    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        cache.total_dim += v->dofs;
        max_id = std::max(max_id, v->variableID);
    }
    cache.num_vars = static_cast<int>(graph.var_nodes.size());
    cache.num_factors = static_cast<int>(graph.factors.size());
    cache.var_ix.assign(max_id + 1, -1);

    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(static_cast<size_t>(cache.total_dim) * 10);

    int offset = 0;
    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        const int id = v->variableID;
        const int m = v->dofs;
        cache.var_ix[id] = offset;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < m; ++c) {
                trips.emplace_back(offset + r, offset + c, 1.0);
            }
        }
        offset += m;
    }

    for (const auto& fptr : graph.factors) {
        const gbp::Factor& f = *fptr;
        if (!f.active) {
            continue;
        }
        const int k = static_cast<int>(f.adj_var_nodes.size());
        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da = va->dofs;
            const int oa = (ida >= 0 && ida < static_cast<int>(cache.var_ix.size()))
                ? cache.var_ix[ida]
                : -1;
            if (oa < 0) {
                throw std::runtime_error("cached joint pattern: missing ida");
            }
            for (int r = 0; r < da; ++r) {
                for (int c = 0; c < da; ++c) {
                    trips.emplace_back(oa + r, oa + c, 1.0);
                }
            }

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                (void)other_factor_ix;
                const gbp::VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db = vb->dofs;
                const int ob = (idb >= 0 && idb < static_cast<int>(cache.var_ix.size()))
                    ? cache.var_ix[idb]
                    : -1;
                if (ob < 0) {
                    throw std::runtime_error("cached joint pattern: missing idb");
                }
                if (idb > ida) {
                    for (int r = 0; r < da; ++r) {
                        for (int c = 0; c < db; ++c) {
                            trips.emplace_back(oa + r, ob + c, 1.0);
                            trips.emplace_back(ob + c, oa + r, 1.0);
                        }
                    }
                }
                other_factor_ix += db;
            }
            factor_ix += da;
        }
    }

    cache.lam.resize(cache.total_dim, cache.total_dim);
    cache.lam.setFromTriplets(trips.begin(), trips.end());
    cache.lam.makeCompressed();
    cache.coeff_to_value_index.reserve(static_cast<size_t>(cache.lam.nonZeros()) * 2);
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            cache.coeff_to_value_index.emplace(sparseCoeffKey(row, col), p);
        }
    }
    cache.sym_value_pairs.reserve(static_cast<size_t>(cache.lam.nonZeros() / 2));
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            if (row >= col) {
                continue;
            }
            const auto transpose_it = cache.coeff_to_value_index.find(sparseCoeffKey(col, row));
            if (transpose_it != cache.coeff_to_value_index.end()) {
                cache.sym_value_pairs.push_back(
                    JointInfSparsePatternCache::SymValuePair{p, transpose_it->second}
                );
            }
        }
    }

    for (const auto& v : graph.var_nodes) {
        if (!v) {
            continue;
        }
        const int offset = cache.var_ix[v->variableID];
        cache.prior_blocks.push_back(
            makeDenseBlockPlan(cache, offset, offset, v->dofs, v->dofs)
        );
    }

    cache.factor_plans.reserve(graph.factors.size());
    for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
        const gbp::Factor& f = *graph.factors[static_cast<size_t>(fi)];
        JointInfSparsePatternCache::FactorPlan factor_plan;
        factor_plan.factor_index = fi;
        const int k = static_cast<int>(f.adj_var_nodes.size());
        int factor_ix = 0;
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = f.adj_var_nodes[a];
            const int ida = va->variableID;
            const int da = va->dofs;
            const int oa = cache.var_ix[ida];
            factor_plan.eta_segments.push_back(
                JointInfSparsePatternCache::EtaSegmentPlan{oa, factor_ix, da}
            );
            factor_plan.blocks.push_back(
                makeDenseBlockPlan(cache, oa, oa, da, da, factor_ix, factor_ix)
            );

            int other_factor_ix = 0;
            for (int b = 0; b < k; ++b) {
                const gbp::VariableNode* vb = f.adj_var_nodes[b];
                const int idb = vb->variableID;
                const int db = vb->dofs;
                const int ob = cache.var_ix[idb];
                if (idb > ida) {
                    factor_plan.blocks.push_back(
                        makeDenseBlockPlan(cache, oa, ob, da, db, factor_ix, other_factor_ix)
                    );
                    factor_plan.blocks.push_back(
                        makeDenseBlockPlan(cache, ob, oa, db, da, other_factor_ix, factor_ix)
                    );
                }
                other_factor_ix += db;
            }
            factor_ix += da;
        }
        cache.factor_plans.push_back(std::move(factor_plan));
    }
    cache.valid = true;
}

void fillJointDistributionInfSparseCached(
    const gbp::FactorGraph& graph,
    JointInfSparsePatternCache& cache,
    Eigen::VectorXd& eta_out
) {
    if (!cache.valid ||
        cache.num_vars != static_cast<int>(graph.var_nodes.size()) ||
        cache.num_factors != static_cast<int>(graph.factors.size())) {
        buildJointInfSparsePatternCache(graph, cache);
    }

    std::fill(
        cache.lam.valuePtr(),
        cache.lam.valuePtr() + cache.lam.nonZeros(),
        0.0
    );

    eta_out.setZero(cache.total_dim);

    double* values = cache.lam.valuePtr();
    size_t prior_plan_index = 0;
    for (int vi = 0; vi < static_cast<int>(graph.var_nodes.size()); ++vi) {
        const auto& v = graph.var_nodes[static_cast<size_t>(vi)];
        if (!v) {
            continue;
        }
        const int id = v->variableID;
        const int m = v->dofs;
        const int offset = cache.var_ix[id];
        eta_out.segment(offset, m) += v->prior.eta();
        addDenseBlockByPlan(values, cache.prior_blocks[prior_plan_index++], v->prior.lam());
    }

    for (const JointInfSparsePatternCache::FactorPlan& plan : cache.factor_plans) {
        if (plan.factor_index < 0 ||
            plan.factor_index >= static_cast<int>(graph.factors.size())) {
            continue;
        }
        const gbp::Factor& f = *graph.factors[static_cast<size_t>(plan.factor_index)];
        if (!f.active) {
            continue;
        }
        const auto f_eta = f.factor.eta();
        const auto f_lam = f.factor.lam();

        for (const JointInfSparsePatternCache::EtaSegmentPlan& eta_plan : plan.eta_segments) {
            eta_out.segment(eta_plan.global_offset, eta_plan.dofs) +=
                f_eta.segment(eta_plan.factor_offset, eta_plan.dofs);
        }
        for (const JointInfSparsePatternCache::DenseBlockPlan& block_plan : plan.blocks) {
            addDenseBlockByPlan(values, block_plan, f_lam);
        }
    }

    for (const JointInfSparsePatternCache::SymValuePair& pair : cache.sym_value_pairs) {
        const double avg = 0.5 * (values[pair.first] + values[pair.second]);
        values[pair.first] = avg;
        values[pair.second] = avg;
    }
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

void factorizeSparseCholesky(
    const Eigen::SparseMatrix<double>& lam,
    double ridge,
    SparseCholeskyFactor& factor,
    bool reuse_pattern = false,
    bool input_already_symmetric = false
) {
    const Eigen::SparseMatrix<double>* A_ptr = &lam;
    if (!input_already_symmetric || ridge != 0.0) {
        factor.A = input_already_symmetric ? lam : symmetrizeSparse(lam);
        if (ridge != 0.0) {
            for (int i = 0; i < factor.A.rows(); ++i) {
                factor.A.coeffRef(i, i) += ridge;
            }
        }
        factor.A.makeCompressed();
        A_ptr = &factor.A;
    }
    const Eigen::SparseMatrix<double>& A = *A_ptr;

    if (!reuse_pattern) {
        factor.ldlt_solver.compute(A);
        if (factor.ldlt_solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky factorization failed");
        }
        return;
    }

    const int outer_size = static_cast<int>(A.outerSize()) + 1;
    const int inner_size = static_cast<int>(A.nonZeros());
    const bool can_reuse =
        reuse_pattern &&
        factor.analyzed &&
        factor.rows == A.rows() &&
        factor.cols == A.cols() &&
        factor.nonzeros == A.nonZeros() &&
        static_cast<int>(factor.outer_index.size()) == outer_size &&
        static_cast<int>(factor.inner_index.size()) == inner_size &&
        std::equal(
            factor.outer_index.begin(),
            factor.outer_index.end(),
            A.outerIndexPtr()
        ) &&
        std::equal(
            factor.inner_index.begin(),
            factor.inner_index.end(),
            A.innerIndexPtr()
    );

    if (!can_reuse) {
        factor.ldlt_solver.analyzePattern(A);
        if (factor.ldlt_solver.info() != Eigen::Success) {
            throw std::runtime_error("Sparse Cholesky analyzePattern failed");
        }
        factor.analyzed = true;
        factor.rows = static_cast<int>(A.rows());
        factor.cols = static_cast<int>(A.cols());
        factor.nonzeros = static_cast<int>(A.nonZeros());
        factor.outer_index.assign(A.outerIndexPtr(), A.outerIndexPtr() + outer_size);
        factor.inner_index.assign(A.innerIndexPtr(), A.innerIndexPtr() + inner_size);
    }

    factor.ldlt_solver.factorize(A);
    if (factor.ldlt_solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky factorization failed");
    }
}

void solveWithSparseCholeskyInto(
    const SparseCholeskyFactor& factor,
    const Eigen::VectorXd& eta,
    Eigen::VectorXd& x
) {
    factor.ldlt_solver.solveInto(eta, x);
    if (factor.ldlt_solver.info() != Eigen::Success) {
        throw std::runtime_error("Sparse Cholesky solve failed");
    }
}

void solveWithReusedCoarsePcgInto(
    const Eigen::SparseMatrix<double>& current_lam,
    const SparseCholeskyFactor& stale_factor,
    const Eigen::VectorXd& eta,
    int max_iterations,
    Eigen::VectorXd& x,
    Eigen::VectorXd& residual,
    Eigen::VectorXd& preconditioned_residual,
    Eigen::VectorXd& direction,
    Eigen::VectorXd& matrix_direction
) {
    const Eigen::Index n = eta.size();
    x.setZero(n);
    residual = eta;
    solveWithSparseCholeskyInto(stale_factor, residual, preconditioned_residual);
    direction = preconditioned_residual;

    double residual_preconditioned_dot = residual.dot(preconditioned_residual);
    if (!(residual_preconditioned_dot > 0.0) ||
        !std::isfinite(residual_preconditioned_dot)) {
        x = preconditioned_residual;
        return;
    }

    const double target_residual_sq =
        1e-12 * std::max(eta.squaredNorm(), 1e-30);
    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        matrix_direction.noalias() = current_lam * direction;
        const double curvature = direction.dot(matrix_direction);
        if (!(curvature > 0.0) || !std::isfinite(curvature)) {
            if (iteration == 0) {
                x = preconditioned_residual;
            }
            return;
        }

        const double alpha = residual_preconditioned_dot / curvature;
        if (!std::isfinite(alpha)) {
            if (iteration == 0) {
                x = preconditioned_residual;
            }
            return;
        }
        x.noalias() += alpha * direction;
        residual.noalias() -= alpha * matrix_direction;
        if (residual.squaredNorm() <= target_residual_sq) {
            return;
        }

        solveWithSparseCholeskyInto(
            stale_factor,
            residual,
            preconditioned_residual
        );
        const double next_dot = residual.dot(preconditioned_residual);
        if (!(next_dot > 0.0) || !std::isfinite(next_dot)) {
            return;
        }
        const double beta = next_dot / residual_preconditioned_dot;
        direction = preconditioned_residual + beta * direction;
        residual_preconditioned_dot = next_dot;
    }
}
