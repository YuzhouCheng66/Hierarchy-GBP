Eigen::MatrixXd buildGroupMessageConditionedInformation(
    const gbp::FactorGraph& graph,
    const std::vector<int>& group,
    const std::vector<int>& var_to_group,
    const std::vector<int>& var_to_local_offset,
    std::vector<int>& factor_stamp,
    int stamp_value,
    const SyntheticSE3PackedSoAWorkspace* boundary_messages
) {
    int block_dim = 0;
    for (int var_id : group) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        block_dim += var->dofs;
    }

    Eigen::MatrixXd info = Eigen::MatrixXd::Zero(block_dim, block_dim);
    for (int var_id : group) {
        const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
        const int local_offset = var_to_local_offset[var_id];
        info.block(local_offset, local_offset, var->dofs, var->dofs) += var->prior.lam();

        for (const auto& aref : var->adj_factors) {
            const gbp::Factor* factor = aref.factor;
            if (!factor || !factor->active) {
                continue;
            }

            bool factor_inside = true;
            for (const gbp::VariableNode* adj_var : factor->adj_var_nodes) {
                if (var_to_group[adj_var->variableID] != var_to_group[var_id]) {
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

                std::vector<int> local_factor_offsets(factor->adj_var_nodes.size(), 0);
                int factor_offset = 0;
                for (int a = 0; a < static_cast<int>(factor->adj_var_nodes.size()); ++a) {
                    local_factor_offsets[a] = factor_offset;
                    factor_offset += factor->adj_var_nodes[a]->dofs;
                }

                for (int a = 0; a < static_cast<int>(factor->adj_var_nodes.size()); ++a) {
                    const gbp::VariableNode* va = factor->adj_var_nodes[a];
                    const int row_off = var_to_local_offset[va->variableID];
                    const int da = va->dofs;
                    const int factor_row = local_factor_offsets[a];
                    for (int b = 0; b < static_cast<int>(factor->adj_var_nodes.size()); ++b) {
                        const gbp::VariableNode* vb = factor->adj_var_nodes[b];
                        const int col_off = var_to_local_offset[vb->variableID];
                        const int db = vb->dofs;
                        const int factor_col = local_factor_offsets[b];
                        info.block(row_off, col_off, da, db) +=
                            factor->factor.lam().block(factor_row, factor_col, da, db);
                    }
                }
            } else {
                if (boundary_messages) {
                    info.block<6,6>(local_offset, local_offset) += se3PackedBoundaryPrecision(
                        *boundary_messages, factor->factorID, aref.local_idx);
                } else {
                    info.block(local_offset, local_offset, var->dofs, var->dofs) +=
                        factor->messages[aref.local_idx].lam();
                }
            }
        }
    }

    return 0.5 * (info + info.transpose());
}

SE3BasisData buildMessageConditionedBasis(
    const gbp::FactorGraph& graph,
    const SE3PoseVector* pose_reference,
    int group_size,
    int r_reduced,
    int num_threads,
    const std::vector<Eigen::MatrixXd>* warm_start_local_bases = nullptr,
    std::vector<ExactBasisCacheEntry>* exact_cache = nullptr,
    const SyntheticSE3PackedSoAWorkspace* boundary_messages = nullptr,
    bool preserve_rigid = false,
    bool exclude_anchored_rigid = false
) {
    if (boundary_messages) validateSE3BoundarySlots(graph, *boundary_messages);
    if (preserve_rigid && (!pose_reference || r_reduced<6 || exact_cache))
        throw std::runtime_error("Rigid basis experiment requires poses, rank>=6, no exact cache");
    SE3BasisData basis;
    basis.groups = orderedGroups(static_cast<int>(graph.var_nodes.size()), group_size);
    basis.var_to_group.assign(graph.var_nodes.size(), -1);
    basis.var_to_local_offset.assign(graph.var_nodes.size(), -1);
    basis.var_global_offset.assign(graph.var_nodes.size(), -1);
    basis.full_indices_per_group.resize(basis.groups.size());
    basis.local_bases.resize(basis.groups.size());
    basis.exact_reused.assign(basis.groups.size(), 0);
    if (exact_cache && exact_cache->size() != basis.groups.size()) {
        exact_cache->clear();
        exact_cache->resize(basis.groups.size());
    }
    basis.group_global_offsets.assign(basis.groups.size(), -1);
    basis.group_dims.assign(basis.groups.size(), 0);

    int total_dim = 0;
    for (const auto& vup : graph.var_nodes) {
        if (vup) {
            total_dim += vup->dofs;
        }
    }
    basis.total_dim = total_dim;

    int global_offset = 0;
    for (int var_id = 0; var_id < static_cast<int>(graph.var_nodes.size()); ++var_id) {
        const auto& vup = graph.var_nodes[var_id];
        if (!vup) {
            continue;
        }
        basis.var_global_offset[var_id] = global_offset;
        global_offset += vup->dofs;
    }

    basis.coarse_offsets.push_back(0);
    int coarse_offset = 0;
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        std::vector<int> full_indices;
        int local_offset = 0;
        int block_dim = 0;
        for (int var_id : basis.groups[g]) {
            const gbp::VariableNode* var = graph.var_nodes.at(var_id).get();
            basis.var_to_group[var_id] = g;
            basis.var_to_local_offset[var_id] = local_offset;
            const int base = basis.var_global_offset[var_id];
            for (int d = 0; d < var->dofs; ++d) {
                full_indices.push_back(base + d);
            }
            local_offset += var->dofs;
            block_dim += var->dofs;
        }
        basis.group_dims[g] = block_dim;
        bool contiguous_group = !full_indices.empty();
        for (int i = 0; i < static_cast<int>(full_indices.size()); ++i) {
            if (full_indices[static_cast<size_t>(i)] != full_indices.front() + i) {
                contiguous_group = false;
                break;
            }
        }
        if (contiguous_group) {
            basis.group_global_offsets[g] = full_indices.front();
        }
        basis.full_indices_per_group[g] = std::move(full_indices);
        const int r_local = std::min(std::max(1, r_reduced), block_dim);
        coarse_offset += r_local;
        basis.coarse_offsets.push_back(coarse_offset);
    }
    basis.coarse_dim = coarse_offset;

    struct BasisThreadStats {
        double block_build_sec = 0.0;
        double eigensolver_sec = 0.0;
        double copyout_sec = 0.0;
        double min_eigenvalue = std::numeric_limits<double>::infinity();
        double max_eigenvalue = -std::numeric_limits<double>::infinity();
        int negative_group_count = 0;
        int nonpositive_group_count = 0;
        int partial_attempt_count = 0;
        int partial_converged_count = 0;
        int full_eigensolver_count = 0;
        int exact_reuse_count = 0;
        int partial_work_aborts = 0;
        int partial_iterations = 0;
        double exact_cache_sec = 0.0;
    };

    PartialSymmetricEigenOptions partial_opts{};
    partial_opts.predict_failed_budget = se3EnvFlagEnabled("HGBP_SE3_PREDICT_PARTIAL_BUDGET");
    partial_opts.max_iters = std::max(
        1,
        se3EnvIntOrDefault("GBP_SE3_PARTIAL_BASIS_MAX_ITERS", partial_opts.max_iters)
    );
    partial_opts.residual_check_period = std::max(
        1,
        se3EnvIntOrDefault(
            "GBP_SE3_PARTIAL_BASIS_RESIDUAL_CHECK_PERIOD",
            partial_opts.residual_check_period
        )
    );
    partial_opts.residual_tol = se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_RESIDUAL_TOL",
        partial_opts.residual_tol
    );
    const bool allow_partial_eigensolver = se3PartialBasisEigenEnabled();
    const bool accept_unconverged_partial = se3PartialBasisAcceptUnconverged();
    double accept_partial_residual_tol = se3EnvDoubleOrDefault(
        "GBP_SE3_PARTIAL_BASIS_ACCEPT_RESIDUAL_TOL",
        -1.0
    );
    if (accept_partial_residual_tol < 0.0 &&
        static_cast<int>(basis.groups.size()) >= se3PartialBasisLargeGroupMinGroups()) {
        accept_partial_residual_tol = se3PartialBasisLargeGroupAcceptResidualTol();
    }
    if (accept_partial_residual_tol < 0.0) {
        accept_partial_residual_tol = se3PartialBasisSmallGroupAcceptResidualTol();
    }

    auto process_group = [&](
        int g,
        std::vector<int>& factor_stamp,
        int& stamp_value,
        BasisThreadStats& stats,
        PartialSymmetricEigenWorkspace& partial_ws,
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>& full_eig,
        BandedSymmetricEigenWorkspace& band_eig
    ) {
        const auto block_t0 = SteadyClock::now();
        Eigen::MatrixXd block = buildGroupMessageConditionedInformation(
            graph,
            basis.groups[g],
            basis.var_to_group,
            basis.var_to_local_offset,
            factor_stamp,
            ++stamp_value,
            boundary_messages
        );
        const auto block_t1 = SteadyClock::now();
        stats.block_build_sec += elapsedSeconds(block_t0, block_t1);

        const int block_dim = static_cast<int>(block.rows());
        const int r_local = basis.coarse_offsets[g + 1] - basis.coarse_offsets[g];
        if (exact_cache && r_local < block_dim) {
            const auto cache_t0 = SteadyClock::now();
            const auto& entry = (*exact_cache)[g];
            const bool reuse = entry.matches(block, r_local);
            if (reuse) {
                basis.local_bases[g] = entry.basis;
                basis.exact_reused[g] = 1;
                ++stats.exact_reuse_count;
                stats.min_eigenvalue = std::min(stats.min_eigenvalue, entry.min_eigenvalue);
                stats.max_eigenvalue = std::max(stats.max_eigenvalue, entry.max_eigenvalue);
                if (entry.min_eigenvalue < -1e-10) ++stats.negative_group_count;
                if (entry.min_eigenvalue <= 1e-12) ++stats.nonpositive_group_count;
            }
            stats.exact_cache_sec += elapsedSeconds(cache_t0, SteadyClock::now());
            if (reuse) return;
        }
        Eigen::MatrixXd local_basis = Eigen::MatrixXd::Identity(block_dim, r_local);
        bool cacheable = false;
        double cache_min = 0.0, cache_max = 0.0;
        bool absolute_factor=false;
        if (preserve_rigid && exclude_anchored_rigid) {
            for(int id:basis.groups[g]) for(const auto& adjacent:graph.var_nodes[id]->adj_factors)
                if(adjacent.factor && adjacent.factor->active && adjacent.factor->adj_var_nodes.size()==1)
                    absolute_factor=true;
        }
        if (preserve_rigid && !absolute_factor && r_local < block_dim) {
            const auto eig_t0=SteadyClock::now();
            local_basis=se3RigidPreservingBasis(block,
                se3GroupGaugeGenerators(*pose_reference,basis.groups[g]),r_local);
            stats.eigensolver_sec+=elapsedSeconds(eig_t0,SteadyClock::now());
            ++stats.full_eigensolver_count;
        } else if (r_local < block_dim) {
            const Eigen::MatrixXd* warm_basis =
                    (allow_partial_eigensolver &&
                     warm_start_local_bases != nullptr &&
                     g < static_cast<int>(warm_start_local_bases->size()) &&
                     (*warm_start_local_bases)[g].rows() == block_dim &&
                     (*warm_start_local_bases)[g].cols() > 0)
                    ? &(*warm_start_local_bases)[g]
                    : nullptr;
                const bool try_partial =
                    allow_partial_eigensolver &&
                    r_local < block_dim &&
                    block_dim >= std::max(24, 2 * r_local);

                Eigen::VectorXd evals;
                Eigen::MatrixXd evecs;
                bool eig_ok = false;

                // Exact-zero bandwidth selects the backend, never a scene name.
                // This preserves the matrix and rank; wider blocks keep the old path.
                const auto band_t0 = SteadyClock::now();
                if (band_eig.compute(block, r_local,
                    effectiveThreadCount(num_threads) == 1 ? 2 : 0)) {
                    ++stats.full_eigensolver_count;
                    evals = band_eig.values;
                    evecs = band_eig.vectors;
                    eig_ok = true;
                }
                stats.eigensolver_sec += elapsedSeconds(band_t0, SteadyClock::now());
                if (try_partial && !eig_ok) {
                    ++stats.partial_attempt_count;
                    const auto eig_t0 = SteadyClock::now();
                    const PartialSymmetricEigenResult partial =
                        computeSmallestEigenpairsPartial(block, r_local, warm_basis, partial_ws, partial_opts);
                    const auto eig_t1 = SteadyClock::now();
                    stats.eigensolver_sec += elapsedSeconds(eig_t0, eig_t1);
                    stats.partial_iterations += partial.iterations;
                    stats.partial_work_aborts += partial.aborted_for_work ? 1 : 0;
                    if (!partial.aborted_for_work && partial.eigenvectors.rows() == block_dim &&
                        partial.eigenvectors.cols() == r_local &&
                        partial.eigenvalues.size() == r_local &&
                        (partial.converged ||
                         accept_unconverged_partial ||
                         (accept_partial_residual_tol >= 0.0 &&
                          partial.max_relative_residual <= accept_partial_residual_tol))) {
                        evals = partial.eigenvalues;
                        evecs = partial.eigenvectors;
                        eig_ok = true;
                        if (partial.converged) {
                            ++stats.partial_converged_count;
                        }
                    }
                }

                if (!eig_ok) {
                    ++stats.full_eigensolver_count;
                    const auto eig_t0 = SteadyClock::now();
                    full_eig.compute(block);
                    const auto eig_t1 = SteadyClock::now();
                    stats.eigensolver_sec += elapsedSeconds(eig_t0, eig_t1);
                    if (full_eig.info() == Eigen::Success) {
                        evals = full_eig.eigenvalues();
                        evecs = full_eig.eigenvectors().leftCols(r_local);
                        eig_ok = true;
                    }
                }

                if (eig_ok) {
                    const double min_eval = evals(0);
                    const double max_eval = evals(evals.size() - 1);
                    cacheable = true;
                    cache_min = min_eval;
                    cache_max = max_eval;
                    stats.min_eigenvalue = std::min(stats.min_eigenvalue, min_eval);
                    stats.max_eigenvalue = std::max(stats.max_eigenvalue, max_eval);
                    if (min_eval < -1e-10) {
                        ++stats.negative_group_count;
                    }
                    if (min_eval <= 1e-12) {
                        ++stats.nonpositive_group_count;
                    }

                    local_basis = std::move(evecs);
                }
        }
        if (exact_cache && r_local < block_dim) {
            const auto cache_t0 = SteadyClock::now();
            if (cacheable) (*exact_cache)[g].store(block, local_basis, cache_min, cache_max);
            else (*exact_cache)[g].valid = false;
            stats.exact_cache_sec += elapsedSeconds(cache_t0, SteadyClock::now());
        }
        const auto copy_t0 = SteadyClock::now();
        basis.local_bases[g] = std::move(local_basis);
        const auto copy_t1 = SteadyClock::now();
        stats.copyout_sec += elapsedSeconds(copy_t0, copy_t1);
    };

    const int thread_count = effectiveThreadCount(num_threads);
    const bool do_parallel =
        thread_count > 1 && static_cast<int>(basis.groups.size()) >= std::max(8, thread_count * 2);

    if (!do_parallel) {
        std::vector<int> factor_stamp(graph.factors.size(), -1);
        int stamp_value = 0;
        BasisThreadStats stats;
        PartialSymmetricEigenWorkspace partial_ws;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full_eig;
        BandedSymmetricEigenWorkspace band_eig;
        for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
            process_group(g, factor_stamp, stamp_value, stats, partial_ws, full_eig, band_eig);
        }
        basis.block_build_sec = stats.block_build_sec;
        basis.eigensolver_sec = stats.eigensolver_sec;
        basis.copyout_sec = stats.copyout_sec;
        basis.min_eigenvalue = stats.min_eigenvalue;
        basis.max_eigenvalue = stats.max_eigenvalue;
        basis.negative_group_count = stats.negative_group_count;
        basis.nonpositive_group_count = stats.nonpositive_group_count;
        basis.partial_attempt_count = stats.partial_attempt_count;
        basis.partial_converged_count = stats.partial_converged_count;
        basis.full_eigensolver_count = stats.full_eigensolver_count;
        basis.exact_reuse_count = stats.exact_reuse_count;
        basis.partial_work_aborts = stats.partial_work_aborts;
        basis.partial_iterations = stats.partial_iterations;
        basis.exact_cache_sec = stats.exact_cache_sec;
        return basis;
    }

    std::vector<std::vector<int>> thread_factor_stamps(
        static_cast<size_t>(thread_count),
        std::vector<int>(graph.factors.size(), -1)
    );
    std::vector<int> thread_stamp_values(static_cast<size_t>(thread_count), 0);
    std::vector<BasisThreadStats> thread_stats(static_cast<size_t>(thread_count));
    std::vector<PartialSymmetricEigenWorkspace> thread_partial_ws(static_cast<size_t>(thread_count));
    std::vector<Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>> thread_full_eig(
        static_cast<size_t>(thread_count));
    std::vector<BandedSymmetricEigenWorkspace> thread_band_eig(static_cast<size_t>(thread_count));

    #pragma omp parallel for schedule(static) num_threads(thread_count)
    for (int g = 0; g < static_cast<int>(basis.groups.size()); ++g) {
        const int tid = omp_get_thread_num();
        process_group(
            g,
            thread_factor_stamps[tid],
            thread_stamp_values[tid],
            thread_stats[tid],
            thread_partial_ws[tid],
            thread_full_eig[tid],
            thread_band_eig[tid]
        );
    }

    for (const BasisThreadStats& stats : thread_stats) {
        basis.block_build_sec += stats.block_build_sec;
        basis.eigensolver_sec += stats.eigensolver_sec;
        basis.copyout_sec += stats.copyout_sec;
        basis.min_eigenvalue = std::min(basis.min_eigenvalue, stats.min_eigenvalue);
        basis.max_eigenvalue = std::max(basis.max_eigenvalue, stats.max_eigenvalue);
        basis.negative_group_count += stats.negative_group_count;
        basis.nonpositive_group_count += stats.nonpositive_group_count;
        basis.partial_attempt_count += stats.partial_attempt_count;
        basis.partial_converged_count += stats.partial_converged_count;
        basis.full_eigensolver_count += stats.full_eigensolver_count;
        basis.exact_reuse_count += stats.exact_reuse_count;
        basis.partial_work_aborts += stats.partial_work_aborts;
        basis.partial_iterations += stats.partial_iterations;
        basis.exact_cache_sec += stats.exact_cache_sec;
    }

    return basis;
}

void restrictToCoarseInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse
);

void prolongToFineInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
);

void restrictToCoarseInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse
) {
    if (coarse.size() != basis.coarse_dim) {
        coarse.resize(basis.coarse_dim);
    }
    coarse.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const int off = basis.coarse_offsets[g];
        const int dim = basis.local_bases[g].cols();
        const int group_dim =
            (g < static_cast<int>(basis.group_dims.size()))
                ? basis.group_dims[static_cast<size_t>(g)]
                : static_cast<int>(basis.full_indices_per_group[g].size());
        const int fine_off =
            (g < static_cast<int>(basis.group_global_offsets.size()))
                ? basis.group_global_offsets[static_cast<size_t>(g)]
                : -1;
        if (fine_off >= 0 && fine_off + group_dim <= fine_vec.size()) {
            coarse.segment(off, dim).noalias() =
                basis.local_bases[g].transpose() * fine_vec.segment(fine_off, group_dim);
        } else {
            const std::vector<int>& rows = basis.full_indices_per_group[g];
            Eigen::VectorXd fine_local(rows.size());
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine_local(i) = fine_vec(rows[i]);
            }
            coarse.segment(off, dim).noalias() =
                basis.local_bases[g].transpose() * fine_local;
        }
    }
}

void prolongToFineInto(
    const SE3BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
) {
    if (fine.size() != basis.total_dim) {
        fine.resize(basis.total_dim);
    }
    fine.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const int off = basis.coarse_offsets[g];
        const int dim = basis.local_bases[g].cols();
        const int group_dim =
            (g < static_cast<int>(basis.group_dims.size()))
                ? basis.group_dims[static_cast<size_t>(g)]
                : static_cast<int>(basis.full_indices_per_group[g].size());
        const int fine_off =
            (g < static_cast<int>(basis.group_global_offsets.size()))
                ? basis.group_global_offsets[static_cast<size_t>(g)]
                : -1;
        if (fine_off >= 0 && fine_off + group_dim <= fine.size()) {
            fine.segment(fine_off, group_dim).noalias() =
                basis.local_bases[g] * coarse_vec.segment(off, dim);
        } else {
            const std::vector<int>& rows = basis.full_indices_per_group[g];
            const Eigen::VectorXd fine_local =
                basis.local_bases[g] * coarse_vec.segment(off, dim);
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine(rows[i]) = fine_local(i);
            }
        }
    }
}
