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

void accumulateSE3DeltaComponentNorms(
    const Eigen::VectorXd& delta,
    double& translation_norm_sum,
    double& rotation_norm_sum
) {
    translation_norm_sum = 0.0;
    rotation_norm_sum = 0.0;
    const int n = static_cast<int>(delta.size() / 6);
    for (int i = 0; i < n; ++i) {
        translation_norm_sum += delta.segment<3>(6 * i).norm();
        rotation_norm_sum += delta.segment<3>(6 * i + 3).norm();
    }
}

int chooseActiveFixedLambdaStart(int hard_fixed_start) {
    return hard_fixed_start;
}
void recordSE3FixedLambdaSweepProfile(
    gbp::FactorGraph& graph,
    int hard_fixed_start,
    int active_fixed_start,
    bool use_fixed_lam,
    bool use_eta_only_variable_update
) {
    if (!graph.profile_sync_timing) {
        return;
    }
    ++graph.sync_total_sweeps_accum;
    if (hard_fixed_start >= 0 &&
        (graph.sync_fixed_lam_hard_start_sweep < 0 ||
         hard_fixed_start < graph.sync_fixed_lam_hard_start_sweep)) {
        graph.sync_fixed_lam_hard_start_sweep = hard_fixed_start;
    }
    if (active_fixed_start >= 0 &&
        (graph.sync_fixed_lam_effective_start_sweep < 0 ||
         active_fixed_start < graph.sync_fixed_lam_effective_start_sweep)) {
        graph.sync_fixed_lam_effective_start_sweep = active_fixed_start;
    }
    if (use_fixed_lam) {
        ++graph.sync_fixed_lam_factor_sweeps_accum;
    }
    if (use_eta_only_variable_update) {
        ++graph.sync_eta_only_variable_sweeps_accum;
    }
}

bool se3PackedSoASweepsEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_SWEEPS");
}

bool se3PackedSoAFullCopyEachCallEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_FULL_COPY_EACH_CALL");
}

bool se3PackedSoADeferGraphCopyEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_PACKED_SOA_DEFER_GRAPH_COPY");
}

bool se3ImplicitFineOperatorEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_IMPLICIT_FINE_OPERATOR");
}

bool se3ImplicitFineOperatorCompareEnabled() {
    return se3EnvFlagEnabled("GBP_SE3_IMPLICIT_FINE_OPERATOR_COMPARE");
}

int se3ImplicitFineOperatorThreadCount(int fallback_threads) {
    const int configured =
        se3EnvIntOrDefault("GBP_SE3_IMPLICIT_FINE_OPERATOR_THREADS", 0);
    return configured > 0 ? effectiveThreadCount(configured) : fallback_threads;
}

struct SE3PackedSoASweepRuntime {
    std::unique_ptr<SyntheticSE3PackedSoAWorkspace> workspace;
    const SyntheticSE3Problem* problem = nullptr;
    bool valid = false;
};

std::unordered_map<gbp::FactorGraph*, SE3PackedSoASweepRuntime>& se3PackedSoAStates() {
    static std::unordered_map<gbp::FactorGraph*, SE3PackedSoASweepRuntime> states;
    return states;
}

SyntheticSE3PackedSoAWorkspace* preparePackedSoAWorkspace(
    gbp::FactorGraph& graph,
    const SyntheticSE3Problem& problem
) {
    if (!se3PackedSoASweepsEnabled()) {
        return nullptr;
    }
    auto& state = se3PackedSoAStates()[&graph];
    if (state.workspace == nullptr || state.problem != &problem) {
        state.workspace = std::make_unique<SyntheticSE3PackedSoAWorkspace>(
            buildSyntheticSE3PackedSoAWorkspace(problem, 1e-12)
        );
        state.problem = &problem;
        state.valid = false;
    }
    if (!state.valid) {
        relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(*state.workspace, graph);
        state.valid = true;
    }
    return state.workspace.get();
}

bool tryStackedMeanVectorPackedSoA(
    gbp::FactorGraph& graph,
    int num_threads,
    Eigen::VectorXd& out
) {
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it == se3PackedSoAStates().end() ||
        !packed_state_it->second.valid ||
        packed_state_it->second.workspace == nullptr) {
        return false;
    }
    stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(
        *packed_state_it->second.workspace,
        out,
        num_threads
    );
    return true;
}

bool tryApplyMeanDeltaPackedSoA(
    gbp::FactorGraph& graph,
    const Eigen::VectorXd& delta,
    int num_threads
) {
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it == se3PackedSoAStates().end() ||
        !packed_state_it->second.valid ||
        packed_state_it->second.workspace == nullptr) {
        return false;
    }
    applyMeanDeltaSyntheticSE3PackedSoAWorkspace(
        *packed_state_it->second.workspace,
        delta,
        num_threads
    );
    return true;
}

bool tryManualSynchronousIterationsPackedSoA(
    gbp::FactorGraph& graph,
    int num_sweeps,
    int num_threads,
    int sweep_offset,
    int outer_index,
    int fixed_lam_after,
    const SyntheticSE3Problem* problem
) {
    if (!se3PackedSoASweepsEnabled() || num_sweeps <= 0) {
        return false;
    }
    auto& states = se3PackedSoAStates();
    auto& state = states[&graph];
    if (problem != nullptr &&
        (state.workspace == nullptr || state.problem != problem)) {
        state.workspace = std::make_unique<SyntheticSE3PackedSoAWorkspace>(
            buildSyntheticSE3PackedSoAWorkspace(*problem, 1e-12)
        );
        state.problem = problem;
        state.valid = false;
    }
    if (state.workspace == nullptr) {
        return false;
    }

    SyntheticSE3PackedSoAWorkspace& packed = *state.workspace;
    packed.adaptive_precision = se3EnvFlagEnabled("HGBP_SE3_ADAPTIVE_PRECISION");
    packed.persistent_sweeps=se3EnvFlagEnabled("HGBP_SE3_PERSISTENT_SWEEPS");
    if (!state.valid) {
        relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(packed, graph);
        state.valid = true;
    }
    packed.sweeps_since_relinearize = std::max(0, sweep_offset);
    if (fixed_lam_after >= 0 && sweep_offset > fixed_lam_after) {
        std::fill(
            packed.fixed_lam_initialized.begin(),
            packed.fixed_lam_initialized.end(),
            static_cast<std::uint8_t>(1)
        );
    }

    const int active_fixed_lam_start =
        chooseActiveFixedLambdaStart(fixed_lam_after);
    if (graph.profile_sync_timing) {
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = sweep_offset + sweep;
            const bool use_fixed_lam =
                active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
            const bool use_eta_only =
                active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
            recordSE3FixedLambdaSweepProfile(
                graph,
                fixed_lam_after,
                active_fixed_lam_start,
                use_fixed_lam,
                use_eta_only
            );
        }
    }

    SyntheticSE3PackedSoAStats stats;
    SyntheticSE3PackedSoAStats* stats_ptr = graph.profile_sync_timing ? &stats : nullptr;
    if(std::getenv("HGBP_SE3_PRECISION_DEFECT") && std::string(std::getenv("HGBP_SE3_PRECISION_DEFECT"))=="diagonal") {
        precisionDefectSE3Iterations(packed,num_sweeps,num_threads,graph.eta_damping,se3FactorMessageMaxRelativeUpdate());
    } else {
    synchronousIterationsSyntheticSE3PackedSoAWorkspace(
        packed,
        num_sweeps,
        num_threads,
        fixed_lam_after,
        graph.eta_damping,
        se3FactorMessageMaxRelativeUpdate(),
        stats_ptr
    );
    }
    if (!se3PackedSoADeferGraphCopyEnabled()) {
        if (se3PackedSoAFullCopyEachCallEnabled()) {
            copySyntheticSE3PackedSoAToGraph(packed, graph, num_threads);
        } else {
            const bool all_sweeps_eta_only =
                fixed_lam_after >= 0 && sweep_offset > fixed_lam_after;
            copySyntheticSE3PackedSoABeliefsToGraph(
                packed,
                graph,
                num_threads,
                !all_sweeps_eta_only
            );
        }
    }
    state.valid = true;
    if (graph.profile_sync_timing) {
        graph.sync_factor_pass_sec_accum += stats.factor_pass_sec;
        graph.sync_variable_pass_sec_accum += stats.variable_pass_sec;
    }
    return true;
}

inline double se3SqNorm6Raw(const double* x) noexcept {
    return x[0] * x[0] + x[1] * x[1] + x[2] * x[2] +
           x[3] * x[3] + x[4] * x[4] + x[5] * x[5];
}

inline void se3MatVec6ColMajorRaw(const double* matrix, const double* x, double* out) noexcept {
    out[0] = matrix[0] * x[0] + matrix[6] * x[1] + matrix[12] * x[2] +
             matrix[18] * x[3] + matrix[24] * x[4] + matrix[30] * x[5];
    out[1] = matrix[1] * x[0] + matrix[7] * x[1] + matrix[13] * x[2] +
             matrix[19] * x[3] + matrix[25] * x[4] + matrix[31] * x[5];
    out[2] = matrix[2] * x[0] + matrix[8] * x[1] + matrix[14] * x[2] +
             matrix[20] * x[3] + matrix[26] * x[4] + matrix[32] * x[5];
    out[3] = matrix[3] * x[0] + matrix[9] * x[1] + matrix[15] * x[2] +
             matrix[21] * x[3] + matrix[27] * x[4] + matrix[33] * x[5];
    out[4] = matrix[4] * x[0] + matrix[10] * x[1] + matrix[16] * x[2] +
             matrix[22] * x[3] + matrix[28] * x[4] + matrix[34] * x[5];
    out[5] = matrix[5] * x[0] + matrix[11] * x[1] + matrix[17] * x[2] +
             matrix[23] * x[3] + matrix[29] * x[4] + matrix[35] * x[5];
}

inline void se3StabilizeEta6Raw(double* out_eta, const double* old_eta, const double* ref_eta) noexcept {
    const double max_rel_update = se3FactorMessageMaxRelativeUpdate();
    if (!(max_rel_update > 0.0)) {
        return;
    }
    const double old_norm = std::sqrt(se3SqNorm6Raw(old_eta));
    const double ref_norm = std::sqrt(se3SqNorm6Raw(ref_eta));
    const double eta_ref = std::max(1.0, std::max(old_norm, ref_norm));
    double diff_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
        const double diff = out_eta[i] - old_eta[i];
        diff_sq += diff * diff;
    }
    const double rel_update = std::sqrt(diff_sq) / eta_ref;
    if (!std::isfinite(rel_update) || rel_update <= max_rel_update) {
        return;
    }
    const double step = max_rel_update / std::max(rel_update, 1e-300);
    for (int i = 0; i < 6; ++i) {
        out_eta[i] = old_eta[i] + step * (out_eta[i] - old_eta[i]);
    }
}

inline void se3FixedLamEtaHotUpdate(
    const double* factor_eta,
    int target_offset,
    int other_offset,
    const double* eta_map,
    const double* belief_other_eta,
    const double* old_other_eta,
    const double* old_target_eta,
    double eta_damping,
    double* out_eta
) noexcept {
    double eno[6];
    for (int i = 0; i < 6; ++i) {
        eno[i] = factor_eta[other_offset + i] + belief_other_eta[i] - old_other_eta[i];
    }

    double schur_eta[6];
    se3MatVec6ColMajorRaw(eta_map, eno, schur_eta);

    if (eta_damping == 0.0) {
        for (int i = 0; i < 6; ++i) {
            out_eta[i] = factor_eta[target_offset + i] - schur_eta[i];
        }
    } else {
        const double keep_new = 1.0 - eta_damping;
        for (int i = 0; i < 6; ++i) {
            const double proposed = factor_eta[target_offset + i] - schur_eta[i];
            out_eta[i] = keep_new * proposed + eta_damping * old_target_eta[i];
        }
    }
    se3StabilizeEta6Raw(out_eta, old_target_eta, factor_eta + target_offset);
}

bool rebuildSE3FixedLamEtaHotEntries(gbp::FactorGraph& graph) {
    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    entries.clear();
    entries.reserve(graph.factors.size());
    for (const auto& factor_ptr : graph.factors) {
        if (!factor_ptr || !factor_ptr->active) {
            continue;
        }
        gbp::FixedLamEta6HotEntry entry;
        if (factor_ptr->tryExportFixedLamEta6HotEntry(entry)) {
            entries.push_back(entry);
            continue;
        }
        if (factor_ptr->isUnaryFactor()) {
            continue;
        }
        graph.se3_fixed_lam_eta_hot_valid = false;
        entries.clear();
        return false;
    }
    graph.se3_fixed_lam_eta_hot_valid = true;
    return true;
}

bool runSE3FixedLamEtaHotFactorPass(
    gbp::FactorGraph& graph,
    int thread_count,
    double eta_damping
) {
    if (!graph.se3_fixed_lam_eta_hot_valid && !rebuildSE3FixedLamEtaHotEntries(graph)) {
        return false;
    }

    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    if (entries.empty()) {
        return false;
    }

    if (thread_count > 1) {
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
            const gbp::FixedLamEta6HotEntry& e = entries[static_cast<size_t>(i)];
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                e.msg0_eta, eta_damping, e.msg0_eta);
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                e.msg1_eta, eta_damping, e.msg1_eta);
        }
    } else {
        for (const gbp::FixedLamEta6HotEntry& e : entries) {
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                e.msg0_eta, eta_damping, e.msg0_eta);
            se3FixedLamEtaHotUpdate(
                e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                e.msg1_eta, eta_damping, e.msg1_eta);
        }
    }
    ++graph.sync_fixed_lam_hot_eta_sweeps_accum;
    graph.sync_fixed_lam_hot_eta_entries = static_cast<int>(entries.size());
    return true;
}

bool runSE3FixedLamEtaHotSweepBatch(
    gbp::FactorGraph& graph,
    int thread_count,
    double eta_damping,
    int num_sweeps,
    double& factor_sec_accum,
    double& variable_sec_accum
) {
    if (num_sweeps <= 0) {
        return true;
    }
    if (!graph.se3_fixed_lam_eta_hot_valid && !rebuildSE3FixedLamEtaHotEntries(graph)) {
        return false;
    }

    auto& entries = graph.se3_fixed_lam_eta_hot_entries;
    if (entries.empty()) {
        return false;
    }

    const int num_vars = static_cast<int>(graph.var_nodes.size());
    if (thread_count > 1) {
        SteadyClock::time_point factor_t0;
        SteadyClock::time_point factor_t1;
        SteadyClock::time_point var_t0;
        SteadyClock::time_point var_t1;

        #pragma omp parallel num_threads(thread_count)
        {
            for (int sweep = 0; sweep < num_sweeps; ++sweep) {
                #pragma omp single
                {
                    factor_t0 = SteadyClock::now();
                }

                #pragma omp for schedule(static)
                for (int i = 0; i < static_cast<int>(entries.size()); ++i) {
                    const gbp::FixedLamEta6HotEntry& e = entries[static_cast<size_t>(i)];
                    se3FixedLamEtaHotUpdate(
                        e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                        e.msg0_eta, eta_damping, e.msg0_eta);
                    se3FixedLamEtaHotUpdate(
                        e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                        e.msg1_eta, eta_damping, e.msg1_eta);
                }

                #pragma omp single
                {
                    factor_t1 = SteadyClock::now();
                    factor_sec_accum += elapsedSeconds(factor_t0, factor_t1);
                    var_t0 = factor_t1;
                }

                const bool update_mu = (sweep + 1 == num_sweeps);
                #pragma omp for schedule(static)
                for (int i = 0; i < num_vars; ++i) {
                    auto& var_ptr = graph.var_nodes[static_cast<size_t>(i)];
                    if (var_ptr && var_ptr->active) {
                        if (update_mu) {
                            var_ptr->updateBeliefEtaOnly();
                        } else {
                            var_ptr->updateBeliefEtaOnlyNoMu();
                        }
                    }
                }

                #pragma omp single
                {
                    var_t1 = SteadyClock::now();
                    variable_sec_accum += elapsedSeconds(var_t0, var_t1);
                }
            }
        }
    } else {
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const auto factor_t0 = SteadyClock::now();
            for (const gbp::FixedLamEta6HotEntry& e : entries) {
                se3FixedLamEtaHotUpdate(
                    e.factor_eta, 0, 6, e.eta_map0, e.belief1_eta, e.msg1_eta,
                    e.msg0_eta, eta_damping, e.msg0_eta);
                se3FixedLamEtaHotUpdate(
                    e.factor_eta, 6, 0, e.eta_map1, e.belief0_eta, e.msg0_eta,
                    e.msg1_eta, eta_damping, e.msg1_eta);
            }
            const auto factor_t1 = SteadyClock::now();
            const bool update_mu = (sweep + 1 == num_sweeps);
            for (auto& var_ptr : graph.var_nodes) {
                if (var_ptr && var_ptr->active) {
                    if (update_mu) {
                        var_ptr->updateBeliefEtaOnly();
                    } else {
                        var_ptr->updateBeliefEtaOnlyNoMu();
                    }
                }
            }
            const auto var_t1 = SteadyClock::now();
            factor_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            variable_sec_accum += elapsedSeconds(factor_t1, var_t1);
        }
    }

    graph.sync_fixed_lam_hot_eta_sweeps_accum += num_sweeps;
    graph.sync_fixed_lam_hot_eta_entries = static_cast<int>(entries.size());
    return true;
}

void manualSynchronousIterations(
    gbp::FactorGraph& graph,
    int num_sweeps,
    int num_threads,
    int sweep_offset = 0,
    int outer_index = 0,
    const SyntheticSE3Problem* packed_problem = nullptr
) {
    if (num_sweeps <= 0) {
        return;
    }
    const int thread_count = effectiveThreadCount(num_threads);
    const int fixed_lam_after = se3FixedLambdaAfterSweepsForOuter(outer_index);
    const bool allow_fixed_lam_inverse_cache = se3FixedLambdaInverseCacheForOuter(outer_index);
    if (tryManualSynchronousIterationsPackedSoA(
            graph,
            num_sweeps,
            thread_count,
            sweep_offset,
            outer_index,
            fixed_lam_after,
            packed_problem)) {
        return;
    }
    if (thread_count > 1) {
        graph.setSyncNumThreads(thread_count);
        graph.setSyncSchedule(gbp::FactorGraph::SyncScheduleKind::Static, 0);
        for (int sweep = 0; sweep < num_sweeps; ++sweep) {
            const int global_sweep = sweep_offset + sweep;
            const int active_fixed_lam_start = chooseActiveFixedLambdaStart(fixed_lam_after);
            const bool use_fixed_lam =
                active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
            const bool use_eta_only_variable_update =
                active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
            const bool defer_eta_only_mu =
                use_eta_only_variable_update && (sweep + 1 < num_sweeps);
            const bool defer_full_mu =
                !use_eta_only_variable_update && (sweep + 1 < num_sweeps);
            recordSE3FixedLambdaSweepProfile(
                graph,
                fixed_lam_after,
                active_fixed_lam_start,
                use_fixed_lam,
                use_eta_only_variable_update
            );
            if (se3FixedLambdaHotEtaBatchEnabled() &&
                se3FixedLambdaHotEtaPassEnabled() &&
                use_eta_only_variable_update &&
                allow_fixed_lam_inverse_cache) {
                const int remaining_sweeps = num_sweeps - sweep;
                double batch_factor_sec = 0.0;
                double batch_variable_sec = 0.0;
                if (runSE3FixedLamEtaHotSweepBatch(
                        graph,
                        thread_count,
                        graph.eta_damping,
                        remaining_sweeps,
                        batch_factor_sec,
                        batch_variable_sec)) {
                    if (graph.profile_sync_timing) {
                        graph.sync_factor_pass_sec_accum += batch_factor_sec;
                        graph.sync_variable_pass_sec_accum += batch_variable_sec;
                        for (int extra = 1; extra < remaining_sweeps; ++extra) {
                            recordSE3FixedLambdaSweepProfile(
                                graph,
                                fixed_lam_after,
                                active_fixed_lam_start,
                                true,
                                true
                            );
                        }
                    }
                    sweep += remaining_sweeps - 1;
                    continue;
                }
            }
            std::exception_ptr first_exception = nullptr;
            int failed_index = -1;
            bool failed_in_factor_pass = false;
            SteadyClock::time_point factor_t0;
            SteadyClock::time_point factor_t1;
            SteadyClock::time_point var_t0;
            SteadyClock::time_point var_t1;

            factor_t0 = SteadyClock::now();
            const bool used_hot_eta_pass =
                se3FixedLambdaHotEtaPassEnabled() &&
                use_eta_only_variable_update &&
                allow_fixed_lam_inverse_cache &&
                runSE3FixedLamEtaHotFactorPass(graph, thread_count, graph.eta_damping);
            if (!used_hot_eta_pass) {
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int i = 0; i < static_cast<int>(graph.factors.size()); ++i) {
                    try {
                        auto& factor_ptr = graph.factors[static_cast<size_t>(i)];
                        if (factor_ptr && factor_ptr->active) {
                            if (use_fixed_lam) {
                                factor_ptr->computeMessagesFixedLam(
                                    graph.eta_damping,
                                    allow_fixed_lam_inverse_cache);
                            } else {
                                factor_ptr->computeMessages(graph.eta_damping);
                            }
                        }
                    } catch (...) {
                        #pragma omp critical(se3_gbp_sweep_exception)
                        {
                            if (!first_exception) {
                                first_exception = std::current_exception();
                                failed_index = i;
                                failed_in_factor_pass = true;
                            }
                        }
                    }
                }
            }
            factor_t1 = SteadyClock::now();

            if (!first_exception) {
                var_t0 = SteadyClock::now();
                #pragma omp parallel for schedule(static) num_threads(thread_count)
                for (int i = 0; i < static_cast<int>(graph.var_nodes.size()); ++i) {
                    try {
                        auto& var_ptr = graph.var_nodes[static_cast<size_t>(i)];
                        if (var_ptr && var_ptr->active) {
                            if (defer_eta_only_mu) {
                                var_ptr->updateBeliefEtaOnlyNoMu();
                            } else if (use_eta_only_variable_update) {
                                var_ptr->updateBeliefEtaOnly();
                            } else if (defer_full_mu) {
                                var_ptr->updateBeliefNoMu();
                            } else {
                                var_ptr->updateBelief();
                            }
                        }
                    } catch (...) {
                        #pragma omp critical(se3_gbp_sweep_exception)
                        {
                            if (!first_exception) {
                                first_exception = std::current_exception();
                                failed_index = i;
                                failed_in_factor_pass = false;
                            }
                        }
                    }
                }
                var_t1 = SteadyClock::now();
            } else {
                var_t0 = factor_t1;
                var_t1 = factor_t1;
            }
            if (first_exception) {
                try {
                    std::rethrow_exception(first_exception);
                } catch (const std::exception& e) {
                    if (failed_in_factor_pass) {
                        throw std::runtime_error(
                            "SE3 GBP parallel factor pass failed at sweep " +
                            std::to_string(sweep) + ", factor " +
                            std::to_string(failed_index) + ": " + e.what()
                        );
                    }
                    throw std::runtime_error(
                        "SE3 GBP parallel variable pass failed at sweep " +
                        std::to_string(sweep) + ", variable " +
                        std::to_string(failed_index) + ": " + e.what());
                } catch (...) {
                    if (failed_in_factor_pass) {
                        throw std::runtime_error(
                            "SE3 GBP parallel factor pass failed at sweep " +
                            std::to_string(sweep) + ", factor " +
                            std::to_string(failed_index)
                        );
                    }
                    throw std::runtime_error(
                        "SE3 GBP parallel variable pass failed at sweep " +
                        std::to_string(sweep) + ", variable " +
                        std::to_string(failed_index));
                }
            }
            if (graph.profile_sync_timing) {
                graph.sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
                graph.sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
            }
        }
        return;
    }
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        const int global_sweep = sweep_offset + sweep;
        const int active_fixed_lam_start = chooseActiveFixedLambdaStart(fixed_lam_after);
        const bool use_fixed_lam =
            active_fixed_lam_start >= 0 && global_sweep >= active_fixed_lam_start;
        const bool use_eta_only_variable_update =
            active_fixed_lam_start >= 0 && global_sweep > active_fixed_lam_start;
        const bool defer_eta_only_mu =
            use_eta_only_variable_update && (sweep + 1 < num_sweeps);
        const bool defer_full_mu =
            !use_eta_only_variable_update && (sweep + 1 < num_sweeps);
        recordSE3FixedLambdaSweepProfile(
            graph,
            fixed_lam_after,
            active_fixed_lam_start,
            use_fixed_lam,
            use_eta_only_variable_update
        );
        const auto factor_t0 = SteadyClock::now();
        const bool used_hot_eta_pass =
            se3FixedLambdaHotEtaPassEnabled() &&
            use_eta_only_variable_update &&
            allow_fixed_lam_inverse_cache &&
            runSE3FixedLamEtaHotFactorPass(graph, thread_count, graph.eta_damping);
        if (!used_hot_eta_pass) {
            for (const auto& factor_ptr : graph.factors) {
                if (factor_ptr && factor_ptr->active) {
                    if (use_fixed_lam) {
                        factor_ptr->computeMessagesFixedLam(graph.eta_damping, allow_fixed_lam_inverse_cache);
                    } else {
                        factor_ptr->computeMessages(graph.eta_damping);
                    }
                }
            }
        }
        const auto factor_t1 = SteadyClock::now();
        const auto var_t0 = SteadyClock::now();
        for (const auto& var_ptr : graph.var_nodes) {
            if (var_ptr && var_ptr->active) {
                if (defer_eta_only_mu) {
                    var_ptr->updateBeliefEtaOnlyNoMu();
                } else if (use_eta_only_variable_update) {
                    var_ptr->updateBeliefEtaOnly();
                } else if (defer_full_mu) {
                    var_ptr->updateBeliefNoMu();
                } else {
                    var_ptr->updateBelief();
                }
            }
        }
        const auto var_t1 = SteadyClock::now();
        if (graph.profile_sync_timing) {
            graph.sync_factor_pass_sec_accum += elapsedSeconds(factor_t0, factor_t1);
            graph.sync_variable_pass_sec_accum += elapsedSeconds(var_t0, var_t1);
        }
    }
}

void injectCorrectionKeepMessages(gbp::FactorGraph& graph, const Eigen::VectorXd& delta) {
    int offset = 0;
    for (auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        gbp::VariableNode& var = *vup;
        var.refreshMu();
        var.mu.noalias() += delta.segment(offset, var.dofs);
        var.belief.setEta(var.belief.lam() * var.mu);
        var.markMuCurrent();
        offset += var.dofs;
    }
    if (offset != delta.size()) {
        throw std::runtime_error("injectCorrectionKeepMessages: delta vector size does not match graph dofs");
    }
    auto packed_state_it = se3PackedSoAStates().find(&graph);
    if (packed_state_it != se3PackedSoAStates().end() &&
        packed_state_it->second.valid &&
        packed_state_it->second.workspace != nullptr) {
        SyntheticSE3PackedSoAWorkspace& packed = *packed_state_it->second.workspace;
        if (packed.num_vars == static_cast<int>(graph.var_nodes.size())) {
            for (int i = 0; i < packed.num_vars; ++i) {
                const gbp::VariableNode& var = *graph.var_nodes[static_cast<size_t>(i)];
                std::memcpy(
                    packed.belief_eta.data() + static_cast<size_t>(i) * 6,
                    var.belief.etaData(),
                    6 * sizeof(double)
                );
                packed.mu_valid[static_cast<size_t>(i)] = 0;
            }
        } else {
            packed_state_it->second.valid = false;
        }
    }
}
