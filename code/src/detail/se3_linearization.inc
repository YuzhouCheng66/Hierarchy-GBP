std::vector<Eigen::VectorXd> emptyMeasurementBlocks(const Eigen::VectorXd&) {
    return {};
}

std::vector<Eigen::MatrixXd> emptyJacobianBlocks(const Eigen::VectorXd&) {
    return {};
}

SyntheticSE3ResidualGraphWorkspace buildSyntheticSE3ResidualGraphWorkspace(
    const SyntheticSE3Problem& problem,
    double tiny_prior
) {
    SyntheticSE3ResidualGraphWorkspace workspace;
    workspace.tiny_prior = tiny_prior;
    workspace.graph.nonlinear_factors = false;
    workspace.graph.eta_damping = se3MessageDamping();

    const int n = static_cast<int>(problem.gt_poses.size());
    workspace.graph.var_nodes.reserve(n);
    workspace.graph.var_residual.reserve(n);
    workspace.graph.factors.reserve(problem.edges.size() + 1);
    workspace.vars.resize(n, nullptr);
    workspace.edge_factors.resize(problem.edges.size(), nullptr);

    std::vector<int> variable_degree(static_cast<size_t>(n), 0);
    for (const SyntheticSE3Edge& edge : problem.edges) {
        ++variable_degree[edge.i];
        ++variable_degree[edge.j];
    }
    if (!variable_degree.empty()) {
        ++variable_degree[0];
    }

    const Mat6 weak_lam = tiny_prior * Mat6::Identity();
    for (int i = 0; i < n; ++i) {
        gbp::VariableNode* var = workspace.graph.addVariable(i, 6);
        var->GT = Eigen::VectorXd::Zero(6);
        var->adj_factors.reserve(variable_degree[static_cast<size_t>(i)]);
        var->adj_factors_raw.reserve(variable_degree[static_cast<size_t>(i)]);
        var->prior.setLam(weak_lam);
        var->prior.setEta(Eigen::VectorXd::Zero(6));
        var->belief.setLam(weak_lam);
        var->belief.setEta(Eigen::VectorXd::Zero(6));
        var->mu = Eigen::VectorXd::Zero(6);
        var->markMuCurrent();
        workspace.vars[i] = var;
    }

    int fid = 0;
    for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
        const SyntheticSE3Edge& edge = problem.edges[edge_index];
        gbp::VariableNode* vi = workspace.vars.at(edge.i);
        gbp::VariableNode* vj = workspace.vars.at(edge.j);
        gbp::Factor* factor = workspace.graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vi, vj},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        workspace.graph.connect(factor, vi, 0);
        workspace.graph.connect(factor, vj, 1);
        workspace.edge_factors[edge_index] = factor;
    }

    if (!workspace.vars.empty()) {
        gbp::VariableNode* v0 = workspace.vars.at(0);
        workspace.anchor_factor = workspace.graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{v0},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        workspace.graph.connect(workspace.anchor_factor, v0, 0);
    }

    return workspace;
}

void resetSyntheticSE3ResidualState(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    int num_threads
) {
    const int n = static_cast<int>(workspace.vars.size());
    const bool use_parallel = (num_threads != 1) && (n > 64);

    auto reset_var = [&](int i) {
        gbp::VariableNode* var = workspace.vars[i];
        if (!var) {
            return;
        }
        const int d = var->dofs;
        std::memset(var->belief.etaData(), 0, static_cast<size_t>(d) * sizeof(double));
        std::memcpy(var->belief.lamData(), var->prior.lamData(), static_cast<size_t>(d * d) * sizeof(double));
        var->mu.setZero();
        var->markMuCurrent();
    };

    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            reset_var(i);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            reset_var(i);
        }
    }
}

void clearGaussian(utils::NdimGaussian& gaussian) {
    const int d = gaussian.dim();
    if (d <= 0) {
        return;
    }
    std::memset(gaussian.etaData(), 0, static_cast<size_t>(d) * sizeof(double));
    std::memset(gaussian.lamData(), 0, static_cast<size_t>(d * d) * sizeof(double));
}

bool transportGaussianBetweenSE3Linearizations(
    utils::NdimGaussian& gaussian,
    const Vec6& old_to_new_delta
) {
    if (gaussian.dim() != 6) {
        return false;
    }

    Eigen::Map<const Mat6> old_lam(gaussian.lamData());
    Eigen::Map<const Vec6> old_eta(gaussian.etaData());
    if (old_lam.cwiseAbs().maxCoeff() < 1e-24 && old_eta.cwiseAbs().maxCoeff() < 1e-18) {
        clearGaussian(gaussian);
        return true;
    }
    if (!allFiniteEigen(old_lam) ||
        !allFiniteEigen(old_eta) ||
        !allFiniteEigen(old_to_new_delta)) {
        return false;
    }

    Vec6 new_eta = old_eta - old_lam * old_to_new_delta;
    if (!allFiniteEigen(new_eta)) {
        return false;
    }
    gaussian.etaRef() = new_eta;
    return true;
}

void transportSyntheticSE3FactorMessages(
    gbp::Factor* factor,
    const Vec6Vector& old_to_new_deltas
) {
    if (factor == nullptr) {
        return;
    }
    for (size_t k = 0; k < factor->messages.size(); ++k) {
        const gbp::VariableNode* var = factor->adj_var_nodes.at(k);
        const int variable_id = (var != nullptr) ? var->variableID : -1;
        if (variable_id < 0 ||
            variable_id >= static_cast<int>(old_to_new_deltas.size()) ||
            !transportGaussianBetweenSE3Linearizations(
                factor->messages[k],
                old_to_new_deltas[static_cast<size_t>(variable_id)]
            )) {
            clearGaussian(factor->messages[k]);
        }
    }
}

void transportSyntheticSE3ResidualMessages(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    const SE3PoseVector& old_base_poses,
    const SE3PoseVector& new_base_poses,
    int num_threads
) {
    const int m = static_cast<int>(workspace.edge_factors.size());
    const bool use_parallel = (num_threads != 1) && (m > 64);
    const int n = static_cast<int>(old_base_poses.size());
    Vec6Vector old_to_new_deltas(static_cast<size_t>(n));
    const bool delta_parallel =
        (num_threads != 1) &&
        (n > 64) &&
        old_base_poses.size() == new_base_poses.size();
    if (delta_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            old_to_new_deltas[static_cast<size_t>(i)] =
                se3Log(se3Compose(
                    se3Inverse(old_base_poses[static_cast<size_t>(i)]),
                    new_base_poses[static_cast<size_t>(i)]
                ));
        }
    } else {
        for (int i = 0; i < n; ++i) {
            old_to_new_deltas[static_cast<size_t>(i)] =
                se3Log(se3Compose(
                    se3Inverse(old_base_poses[static_cast<size_t>(i)]),
                    new_base_poses[static_cast<size_t>(i)]
                ));
        }
    }
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < m; ++edge_index) {
            transportSyntheticSE3FactorMessages(
                workspace.edge_factors[static_cast<size_t>(edge_index)],
                old_to_new_deltas
            );
        }
    } else {
        for (int edge_index = 0; edge_index < m; ++edge_index) {
            transportSyntheticSE3FactorMessages(
                workspace.edge_factors[static_cast<size_t>(edge_index)],
                old_to_new_deltas
            );
        }
    }

    if (workspace.anchor_factor != nullptr) {
        transportSyntheticSE3FactorMessages(
            workspace.anchor_factor,
            old_to_new_deltas
        );
    }
}

void refreshSyntheticSE3ResidualBeliefs(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    int num_threads
) {
    const int n = static_cast<int>(workspace.vars.size());
    const bool use_parallel = (num_threads != 1) && (n > 64);
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int i = 0; i < n; ++i) {
            if (workspace.vars[i] != nullptr) {
                workspace.vars[i]->updateBelief();
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            if (workspace.vars[i] != nullptr) {
                workspace.vars[i]->updateBelief();
            }
        }
    }
}

void relinearizeSyntheticSE3ResidualGraph(
    SyntheticSE3ResidualGraphWorkspace& workspace,
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    int num_threads,
    int outer_index,
    SyntheticSE3ResidualRelinearizeStats* stats = nullptr,
    const RobustLossConfig& robust_loss_config = {}
) {
    const auto total_t0 = SteadyClock::now();
    workspace.graph.se3_fixed_lam_eta_hot_valid = false;
    workspace.graph.se3_fixed_lam_eta_hot_entries.clear();
    const bool use_parallel = (num_threads != 1) && (problem.edges.size() > 64);
    const bool transport_messages =
        !se3EnvFlagEnabled("HGBP_SE3_RESET_TRANSPORT") &&
        outer_index >= 12 &&
        workspace.has_linearization_poses &&
        workspace.linearization_poses.size() == base_poses.size();
    auto packed_state_it = se3PackedSoAStates().find(&workspace.graph);
    if (packed_state_it != se3PackedSoAStates().end()) {
        if (transport_messages &&
            packed_state_it->second.valid &&
            packed_state_it->second.workspace != nullptr) {
            copySyntheticSE3PackedSoAToGraph(
                *packed_state_it->second.workspace,
                workspace.graph,
                num_threads
            );
        }
        packed_state_it->second.valid = false;
    }

    const auto transport_t0 = SteadyClock::now();
    if (transport_messages) {
        transportSyntheticSE3ResidualMessages(
            workspace,
            workspace.linearization_poses,
            base_poses,
            num_threads
        );
    }
    const auto transport_t1 = SteadyClock::now();

    const auto factor_t0 = SteadyClock::now();
    if (use_parallel) {
        const int thread_count = effectiveThreadCount(num_threads);
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE3Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[static_cast<size_t>(edge_index)];
            const SE3Pose& base_i = base_poses.at(edge.i);
            const SE3Pose& base_j = base_poses.at(edge.j);
            const Vec6 r0 = edgeResidual(base_i, base_j, edge.measurement, Vec6::Zero(), Vec6::Zero());
            const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
            const Mat6 weighted_information = robust_weight * edge.information;
            const Mat6x12 J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 12, 12> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 12, 1> eta = -J.transpose() * weighted_information * r0;
            factor->setLinearFactorInfo(eta, lam);
            if (!transport_messages) {
                factor->clearMessages();
            }
        }
    } else {
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE3Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[static_cast<size_t>(edge_index)];
            const SE3Pose& base_i = base_poses.at(edge.i);
            const SE3Pose& base_j = base_poses.at(edge.j);
            const Vec6 r0 = edgeResidual(base_i, base_j, edge.measurement, Vec6::Zero(), Vec6::Zero());
            const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
            const Mat6 weighted_information = robust_weight * edge.information;
            const Mat6x12 J = analyticEdgeResidualJacobian(base_i, base_j, edge.measurement);
            const Eigen::Matrix<double, 12, 12> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 12, 1> eta = -J.transpose() * weighted_information * r0;
            factor->setLinearFactorInfo(eta, lam);
            if (!transport_messages) {
                factor->clearMessages();
            }
        }
    }

    if (workspace.anchor_factor != nullptr && !base_poses.empty()) {
        const Vec6 r0 = anchorResidual(base_poses.at(0), problem.anchor_pose, Vec6::Zero());
        const Mat6 J = analyticAnchorResidualJacobian(base_poses.at(0), problem.anchor_pose);
        const Mat6 lam = J.transpose() * problem.anchor_information * J;
        const Vec6 eta = -J.transpose() * problem.anchor_information * r0;
        workspace.anchor_factor->setLinearFactorInfo(eta, lam);
        if (!transport_messages) {
            workspace.anchor_factor->clearMessages();
        }
    }
    const auto factor_t1 = SteadyClock::now();

    const auto reset_t0 = SteadyClock::now();
    if (transport_messages) {
        refreshSyntheticSE3ResidualBeliefs(workspace, num_threads);
    } else {
        resetSyntheticSE3ResidualState(workspace, num_threads);
    }
    const auto reset_t1 = SteadyClock::now();

    workspace.linearization_poses = base_poses;
    workspace.has_linearization_poses = true;

    if (stats) {
        stats->factor_relinearize_sec = elapsedSeconds(factor_t0, factor_t1);
        stats->message_transport_sec = elapsedSeconds(transport_t0, transport_t1);
        stats->reset_state_sec = elapsedSeconds(reset_t0, reset_t1);
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
}
