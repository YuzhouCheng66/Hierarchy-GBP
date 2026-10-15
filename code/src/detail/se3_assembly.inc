struct CoarseLambdaAssemblyCache {
    struct BlockMeta {
        int gi = 0;
        int gj = 0;
        int row_offset = 0;
        int col_offset = 0;
        int rows = 0;
        int cols = 0;
        std::vector<int> value_indices;
    };

    struct PriorPlan {
        int variable_index = -1;
        int block_index = -1;
        int group = -1;
        int local_offset = 0;
        int dofs = 0;
    };

    struct FactorBlockPlan {
        int block_index = -1;
        int ga = -1;
        int gb = -1;
        int la = 0;
        int lb = 0;
        int da = 0;
        int db = 0;
        int factor_row = 0;
        int factor_col = 0;
    };

    struct FactorPlan {
        int factor_index = -1;
        std::vector<FactorBlockPlan> blocks;
    };

    struct SymBlockPair {
        int first = -1;
        int second = -1;
    };

    int num_groups = 0;
    int coarse_dim = 0;
    int num_vars = 0;
    int num_factors = 0;
    std::vector<BlockMeta> blocks;
    std::vector<PriorPlan> prior_plans;
    std::vector<FactorPlan> factor_plans;
    std::vector<int> diagonal_block_indices;
    std::vector<SymBlockPair> sym_block_pairs;
    std::vector<Eigen::MatrixXd> block_values;
    std::vector<std::vector<Eigen::MatrixXd>> tls_block_values;
    std::vector<std::vector<int>> tls_touched_block_indices;
    std::vector<std::vector<unsigned char>> tls_block_touched;
    int tls_thread_count = 0;
    Eigen::SparseMatrix<double> lam;
    Eigen::SparseMatrix<double> sym_lam;
    std::unordered_map<long long, int> coeff_to_value_index;
    bool valid = false;
};

long long coarseBlockPairKey(int gi, int gj) {
    return (static_cast<long long>(gi) << 32) ^ static_cast<unsigned int>(gj);
}

void buildCoarseLambdaAssemblyCache(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    CoarseLambdaAssemblyCache& cache
) {
    cache = CoarseLambdaAssemblyCache{};
    cache.num_groups = static_cast<int>(basis.local_bases.size());
    cache.coarse_dim = basis.coarse_dim;
    cache.num_vars = static_cast<int>(graph.var_nodes.size());
    cache.num_factors = static_cast<int>(graph.factors.size());

    std::unordered_map<long long, int> block_index_by_pair;
    block_index_by_pair.reserve(graph.factors.size() * 4 + basis.local_bases.size());

    auto ensure_block = [&](int gi, int gj) -> int {
        const long long key = coarseBlockPairKey(gi, gj);
        const auto it = block_index_by_pair.find(key);
        if (it != block_index_by_pair.end()) {
            return it->second;
        }
        const int index = static_cast<int>(cache.blocks.size());
        CoarseLambdaAssemblyCache::BlockMeta meta;
        meta.gi = gi;
        meta.gj = gj;
        meta.row_offset = basis.coarse_offsets[gi];
        meta.col_offset = basis.coarse_offsets[gj];
        meta.rows = basis.coarse_offsets[gi + 1] - basis.coarse_offsets[gi];
        meta.cols = basis.coarse_offsets[gj + 1] - basis.coarse_offsets[gj];
        cache.blocks.push_back(meta);
        block_index_by_pair.emplace(key, index);
        return index;
    };

    cache.prior_plans.reserve(graph.var_nodes.size());
    for (int vi = 0; vi < static_cast<int>(graph.var_nodes.size()); ++vi) {
        const auto& v = graph.var_nodes[static_cast<size_t>(vi)];
        if (!v) {
            continue;
        }
        const int g = basis.var_to_group[v->variableID];
        if (g < 0) {
            continue;
        }
        cache.prior_plans.push_back(CoarseLambdaAssemblyCache::PriorPlan{
            vi,
            ensure_block(g, g),
            g,
            basis.var_to_local_offset[v->variableID],
            v->dofs
        });
    }

    cache.factor_plans.reserve(graph.factors.size());
    for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
        const gbp::Factor* factor = graph.factors[static_cast<size_t>(fi)].get();
        if (factor == nullptr) {
            continue;
        }
        CoarseLambdaAssemblyCache::FactorPlan factor_plan;
        factor_plan.factor_index = fi;
        const int k = static_cast<int>(factor->adj_var_nodes.size());
        std::vector<int> local_factor_offsets(static_cast<size_t>(k), 0);
        int factor_offset = 0;
        for (int a = 0; a < k; ++a) {
            local_factor_offsets[static_cast<size_t>(a)] = factor_offset;
            factor_offset += factor->adj_var_nodes[static_cast<size_t>(a)]->dofs;
        }

        factor_plan.blocks.reserve(static_cast<size_t>(k * k));
        for (int a = 0; a < k; ++a) {
            const gbp::VariableNode* va = factor->adj_var_nodes[static_cast<size_t>(a)];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const int off_a = local_factor_offsets[static_cast<size_t>(a)];
            for (int b = 0; b < k; ++b) {
                const gbp::VariableNode* vb = factor->adj_var_nodes[static_cast<size_t>(b)];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const int off_b = local_factor_offsets[static_cast<size_t>(b)];
                factor_plan.blocks.push_back(CoarseLambdaAssemblyCache::FactorBlockPlan{
                    ensure_block(ga, gb),
                    ga,
                    gb,
                    la,
                    lb,
                    da,
                    db,
                    off_a,
                    off_b
                });
            }
        }
        cache.factor_plans.push_back(std::move(factor_plan));
    }

    cache.diagonal_block_indices.reserve(cache.blocks.size());
    cache.sym_block_pairs.reserve(cache.blocks.size() / 2);
    for (int bi = 0; bi < static_cast<int>(cache.blocks.size()); ++bi) {
        const CoarseLambdaAssemblyCache::BlockMeta& meta = cache.blocks[static_cast<size_t>(bi)];
        if (meta.gi == meta.gj) {
            cache.diagonal_block_indices.push_back(bi);
        } else if (meta.gi < meta.gj) {
            const auto transpose_it =
                block_index_by_pair.find(coarseBlockPairKey(meta.gj, meta.gi));
            if (transpose_it != block_index_by_pair.end()) {
                cache.sym_block_pairs.push_back(
                    CoarseLambdaAssemblyCache::SymBlockPair{bi, transpose_it->second}
                );
            }
        }
    }

    size_t triplet_count = 0;
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        triplet_count += static_cast<size_t>(meta.rows) * static_cast<size_t>(meta.cols);
    }
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(triplet_count);
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                trips.emplace_back(meta.row_offset + r, meta.col_offset + c, 1.0);
            }
        }
    }
    cache.lam.resize(cache.coarse_dim, cache.coarse_dim);
    cache.lam.setFromTriplets(trips.begin(), trips.end());
    cache.lam.makeCompressed();

    cache.coeff_to_value_index.reserve(static_cast<size_t>(cache.lam.nonZeros()) * 2);
    for (int col = 0; col < cache.lam.outerSize(); ++col) {
        for (int p = cache.lam.outerIndexPtr()[col]; p < cache.lam.outerIndexPtr()[col + 1]; ++p) {
            const int row = cache.lam.innerIndexPtr()[p];
            cache.coeff_to_value_index.emplace(sparseCoeffKey(row, col), p);
        }
    }
    for (CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        meta.value_indices.clear();
        meta.value_indices.reserve(static_cast<size_t>(meta.rows) * static_cast<size_t>(meta.cols));
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                const auto it = cache.coeff_to_value_index.find(
                    sparseCoeffKey(meta.row_offset + r, meta.col_offset + c)
                );
                if (it == cache.coeff_to_value_index.end()) {
                    throw std::runtime_error("cached coarse lambda pattern missing sparse coefficient");
                }
                meta.value_indices.push_back(it->second);
            }
        }
    }

    cache.block_values.clear();
    cache.block_values.reserve(cache.blocks.size());
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        cache.block_values.emplace_back(Eigen::MatrixXd::Zero(meta.rows, meta.cols));
    }

    cache.valid = true;
}

bool coarseLambdaAssemblyCacheMatches(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache& cache
) {
    return cache.valid &&
        cache.num_groups == static_cast<int>(basis.local_bases.size()) &&
        cache.coarse_dim == basis.coarse_dim &&
        cache.num_vars == static_cast<int>(graph.var_nodes.size()) &&
        cache.num_factors == static_cast<int>(graph.factors.size());
}

std::vector<Eigen::MatrixXd> makeZeroCoarseBlocks(
    const CoarseLambdaAssemblyCache& cache
) {
    std::vector<Eigen::MatrixXd> blocks;
    blocks.reserve(cache.blocks.size());
    for (const CoarseLambdaAssemblyCache::BlockMeta& meta : cache.blocks) {
        blocks.emplace_back(Eigen::MatrixXd::Zero(meta.rows, meta.cols));
    }
    return blocks;
}

void zeroCoarseBlockValues(std::vector<Eigen::MatrixXd>& blocks) {
    for (Eigen::MatrixXd& block : blocks) {
        block.setZero();
    }
}

void ensureCoarseTlsBlockValues(
    const CoarseLambdaAssemblyCache& cache,
    int thread_count,
    std::vector<std::vector<Eigen::MatrixXd>>& tls_blocks,
    std::vector<std::vector<int>>& tls_touched_indices,
    std::vector<std::vector<unsigned char>>& tls_touched_flags,
    int& tls_thread_count
) {
    if (tls_thread_count == thread_count &&
        static_cast<int>(tls_blocks.size()) == thread_count &&
        static_cast<int>(tls_touched_indices.size()) == thread_count &&
        static_cast<int>(tls_touched_flags.size()) == thread_count) {
        bool valid = true;
        for (int tid = 0; tid < thread_count; ++tid) {
            std::vector<Eigen::MatrixXd>& local_blocks = tls_blocks[static_cast<size_t>(tid)];
            if (local_blocks.size() != cache.block_values.size()) {
                valid = false;
                break;
            }
            if (tls_touched_flags[static_cast<size_t>(tid)].size() != cache.block_values.size()) {
                valid = false;
                break;
            }
        }
        if (valid) {
            for (int tid = 0; tid < thread_count; ++tid) {
                std::vector<Eigen::MatrixXd>& local_blocks = tls_blocks[static_cast<size_t>(tid)];
                std::vector<int>& touched = tls_touched_indices[static_cast<size_t>(tid)];
                std::vector<unsigned char>& flags = tls_touched_flags[static_cast<size_t>(tid)];
                for (int block_index : touched) {
                    local_blocks[static_cast<size_t>(block_index)].setZero();
                    flags[static_cast<size_t>(block_index)] = 0;
                }
                touched.clear();
            }
            return;
        }
    }

    tls_blocks.clear();
    tls_touched_indices.clear();
    tls_touched_flags.clear();
    tls_blocks.reserve(static_cast<size_t>(thread_count));
    tls_touched_indices.resize(static_cast<size_t>(thread_count));
    tls_touched_flags.reserve(static_cast<size_t>(thread_count));
    for (int tid = 0; tid < thread_count; ++tid) {
        tls_blocks.push_back(makeZeroCoarseBlocks(cache));
        tls_touched_flags.emplace_back(cache.block_values.size(), 0);
    }
    tls_thread_count = thread_count;
}

void addPriorPlanToCoarseBlocks(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache::PriorPlan& plan,
    std::vector<Eigen::MatrixXd>& blocks
) {
    const gbp::VariableNode* var =
        graph.var_nodes[static_cast<size_t>(plan.variable_index)].get();
    if (var == nullptr) {
        return;
    }
    const auto B = basis.local_bases[plan.group].middleRows(plan.local_offset, plan.dofs);
    blocks[static_cast<size_t>(plan.block_index)].noalias() +=
        B.transpose() * var->prior.lam() * B;
}

void addFactorPlanToCoarseBlocks(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    const CoarseLambdaAssemblyCache::FactorPlan& plan,
    std::vector<Eigen::MatrixXd>& blocks,
    std::vector<int>* touched_indices = nullptr,
    std::vector<unsigned char>* touched_flags = nullptr
) {
    if (plan.factor_index < 0 ||
        plan.factor_index >= static_cast<int>(graph.factors.size())) {
        return;
    }
    const gbp::Factor* factor =
        graph.factors[static_cast<size_t>(plan.factor_index)].get();
    if (factor == nullptr || !factor->active) {
        return;
    }
    const auto factor_lam = factor->factor.lam();
    for (const CoarseLambdaAssemblyCache::FactorBlockPlan& block_plan : plan.blocks) {
        if (touched_indices != nullptr && touched_flags != nullptr) {
            const size_t block_index = static_cast<size_t>(block_plan.block_index);
            if ((*touched_flags)[block_index] == 0) {
                (*touched_flags)[block_index] = 1;
                touched_indices->push_back(block_plan.block_index);
            }
        }
        const auto Ba =
            basis.local_bases[block_plan.ga].middleRows(block_plan.la, block_plan.da);
        const auto Bb =
            basis.local_bases[block_plan.gb].middleRows(block_plan.lb, block_plan.db);
        blocks[static_cast<size_t>(block_plan.block_index)].noalias() +=
            Ba.transpose() *
            factor_lam.block(
                block_plan.factor_row,
                block_plan.factor_col,
                block_plan.da,
                block_plan.db
            ) *
            Bb;
    }
}

const Eigen::SparseMatrix<double>& assembleCoarseLambdaCachedInPlace(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    CoarseLambdaAssemblyCache& cache,
    int num_threads
) {
    if (!coarseLambdaAssemblyCacheMatches(graph, basis, cache)) {
        buildCoarseLambdaAssemblyCache(graph, basis, cache);
    }

    zeroCoarseBlockValues(cache.block_values);
    std::fill(
        cache.lam.valuePtr(),
        cache.lam.valuePtr() + cache.lam.nonZeros(),
        0.0
    );

    for (const CoarseLambdaAssemblyCache::PriorPlan& plan : cache.prior_plans) {
        addPriorPlanToCoarseBlocks(graph, basis, plan, cache.block_values);
    }

    if (cache.factor_plans.size() > 64) {
        const int thread_count = effectiveThreadCount(num_threads);
        const int shard_count = std::min(
            kSE3CoarseAssemblyShards,
            static_cast<int>(cache.factor_plans.size())
        );
        const bool use_touched_tls =
            static_cast<int>(cache.factor_plans.size()) >= se3CoarseTouchedTlsMinFactors();
        ensureCoarseTlsBlockValues(
            cache,
            shard_count,
            cache.tls_block_values,
            cache.tls_touched_block_indices,
            cache.tls_block_touched,
            cache.tls_thread_count
        );
        if (!use_touched_tls) {
            for (std::vector<Eigen::MatrixXd>& local_blocks : cache.tls_block_values) {
                zeroCoarseBlockValues(local_blocks);
            }
        }

        auto process_shard = [&](int shard) {
            const int plan_count = static_cast<int>(cache.factor_plans.size());
            const int plans_per_shard = plan_count / shard_count;
            const int remainder = plan_count % shard_count;
            const int begin =
                shard * plans_per_shard + std::min(shard, remainder);
            const int end =
                begin + plans_per_shard + (shard < remainder ? 1 : 0);
            for (int pi = begin; pi < end; ++pi) {
                if (use_touched_tls) {
                    std::vector<int>& touched_indices =
                        cache.tls_touched_block_indices[static_cast<size_t>(shard)];
                    std::vector<unsigned char>& touched_flags =
                        cache.tls_block_touched[static_cast<size_t>(shard)];
                    addFactorPlanToCoarseBlocks(
                        graph,
                        basis,
                        cache.factor_plans[static_cast<size_t>(pi)],
                        cache.tls_block_values[static_cast<size_t>(shard)],
                        &touched_indices,
                        &touched_flags
                    );
                } else {
                    addFactorPlanToCoarseBlocks(
                        graph,
                        basis,
                        cache.factor_plans[static_cast<size_t>(pi)],
                        cache.tls_block_values[static_cast<size_t>(shard)]
                    );
                }
            }
        };

        if (thread_count > 1) {
            #pragma omp parallel for schedule(static) num_threads(thread_count)
            for (int shard = 0; shard < shard_count; ++shard) {
                process_shard(shard);
            }
        } else {
            for (int shard = 0; shard < shard_count; ++shard) {
                process_shard(shard);
            }
        }

        if (use_touched_tls) {
            for (int shard = 0; shard < shard_count; ++shard) {
                const std::vector<Eigen::MatrixXd>& local_blocks =
                    cache.tls_block_values[static_cast<size_t>(shard)];
                const std::vector<int>& touched_indices =
                    cache.tls_touched_block_indices[static_cast<size_t>(shard)];
                for (int bi : touched_indices) {
                    cache.block_values[static_cast<size_t>(bi)] += local_blocks[static_cast<size_t>(bi)];
                }
            }
        } else {
            for (int shard = 0; shard < shard_count; ++shard) {
                const std::vector<Eigen::MatrixXd>& local_blocks =
                    cache.tls_block_values[static_cast<size_t>(shard)];
                for (int bi = 0; bi < static_cast<int>(cache.block_values.size()); ++bi) {
                    cache.block_values[static_cast<size_t>(bi)] += local_blocks[static_cast<size_t>(bi)];
                }
            }
        }
    } else {
        for (const CoarseLambdaAssemblyCache::FactorPlan& plan : cache.factor_plans) {
            addFactorPlanToCoarseBlocks(graph, basis, plan, cache.block_values);
        }
    }

    const bool inplace_sym = se3CachedCoarseInplaceSymmetrizeEnabled();
    if (inplace_sym) {
        for (int block_index : cache.diagonal_block_indices) {
            Eigen::MatrixXd& block = cache.block_values[static_cast<size_t>(block_index)];
            block = 0.5 * (block + Eigen::MatrixXd(block.transpose()));
        }
        for (const CoarseLambdaAssemblyCache::SymBlockPair& pair : cache.sym_block_pairs) {
            Eigen::MatrixXd avg =
                0.5 * (
                    cache.block_values[static_cast<size_t>(pair.first)] +
                    Eigen::MatrixXd(cache.block_values[static_cast<size_t>(pair.second)].transpose())
                );
            cache.block_values[static_cast<size_t>(pair.first)] = avg;
            cache.block_values[static_cast<size_t>(pair.second)] = avg.transpose();
        }
    }

    double* values = cache.lam.valuePtr();
    for (int bi = 0; bi < static_cast<int>(cache.blocks.size()); ++bi) {
        const CoarseLambdaAssemblyCache::BlockMeta& meta = cache.blocks[static_cast<size_t>(bi)];
        const Eigen::MatrixXd& block = cache.block_values[static_cast<size_t>(bi)];
        size_t value_ix = 0;
        for (int c = 0; c < meta.cols; ++c) {
            for (int r = 0; r < meta.rows; ++r) {
                values[meta.value_indices[value_ix++]] = block(r, c);
            }
        }
    }

    if (inplace_sym) {
        return cache.lam;
    }
    cache.sym_lam = symmetrizeSparse(cache.lam);
    return cache.sym_lam;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirect(
    const gbp::FactorGraph& graph,
    const SE3BasisData& basis,
    bool use_parallel,
    int num_threads
) {
    const int num_groups = static_cast<int>(basis.local_bases.size());
    const int num_block_pairs = num_groups * num_groups;
    std::vector<Eigen::MatrixXd> blocks(num_block_pairs);
    std::vector<char> active(num_block_pairs, 0);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int g = basis.var_to_group[var.variableID];
        const int l = basis.var_to_local_offset[var.variableID];
        const auto B = basis.local_bases[g].middleRows(l, var.dofs);
        const int block_index = g * num_groups + g;
        if (!active[block_index]) {
            blocks[block_index] = Eigen::MatrixXd::Zero(B.cols(), B.cols());
            active[block_index] = 1;
        }
        blocks[block_index].noalias() += B.transpose() * var.prior.lam() * B;
    }

    auto accumulate_factor = [&](const gbp::Factor& factor, std::vector<Eigen::MatrixXd>& target_blocks, std::vector<char>& target_active) {
        if (!factor.active) {
            return;
        }
        std::vector<int> local_factor_offsets(factor.adj_var_nodes.size(), 0);
        int factor_offset = 0;
        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            local_factor_offsets[a] = factor_offset;
            factor_offset += factor.adj_var_nodes[a]->dofs;
        }

        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const auto Ba = basis.local_bases[ga].middleRows(la, da);
            const int off_a = local_factor_offsets[a];

            for (int b = 0; b < static_cast<int>(factor.adj_var_nodes.size()); ++b) {
                const gbp::VariableNode* vb = factor.adj_var_nodes[b];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const auto Bb = basis.local_bases[gb].middleRows(lb, db);
                const int off_b = local_factor_offsets[b];
                const int block_index = ga * num_groups + gb;

                if (!target_active[block_index]) {
                    target_blocks[block_index] = Eigen::MatrixXd::Zero(Ba.cols(), Bb.cols());
                    target_active[block_index] = 1;
                }

                target_blocks[block_index].noalias() +=
                    Ba.transpose() *
                    factor.factor.lam().block(off_a, off_b, da, db) *
                    Bb;
            }
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = effectiveThreadCount(num_threads);
        std::vector<std::vector<Eigen::MatrixXd>> tls_blocks(static_cast<size_t>(thread_count));
        std::vector<std::vector<char>> tls_active(static_cast<size_t>(thread_count));
        for (int tid = 0; tid < thread_count; ++tid) {
            tls_blocks[tid].resize(num_block_pairs);
            tls_active[tid].assign(num_block_pairs, 0);
        }

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, tls_blocks[tid], tls_active[tid]);
        }

        for (int tid = 0; tid < thread_count; ++tid) {
            for (int block_index = 0; block_index < num_block_pairs; ++block_index) {
                if (!tls_active[tid][block_index]) {
                    continue;
                }
                if (!active[block_index]) {
                    blocks[block_index] = std::move(tls_blocks[tid][block_index]);
                    active[block_index] = 1;
                } else {
                    blocks[block_index] += tls_blocks[tid][block_index];
                }
            }
        }
    } else {
        for (const auto& fup : graph.factors) {
            const gbp::Factor* factor = fup.get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, blocks, active);
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    for (int gi = 0; gi < num_groups; ++gi) {
        const int row_off = basis.coarse_offsets[gi];
        for (int gj = 0; gj < num_groups; ++gj) {
            const int block_index = gi * num_groups + gj;
            if (!active[block_index]) {
                continue;
            }
            const int col_off = basis.coarse_offsets[gj];
            const Eigen::MatrixXd& block = blocks[block_index];
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    const double value = block(r, c);
                    if (value != 0.0) {
                        trips.emplace_back(row_off + r, col_off + c, value);
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> coarse_lam(basis.coarse_dim, basis.coarse_dim);
    coarse_lam.setFromTriplets(trips.begin(), trips.end());
    coarse_lam.makeCompressed();
    return symmetrizeSparse(coarse_lam);
}
