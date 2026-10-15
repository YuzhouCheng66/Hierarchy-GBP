struct SyntheticResidualWorkspaceBuildStats {
    double variable_alloc_sec = 0.0;
    double factor_alloc_sec = 0.0;
    double connect_sec = 0.0;
    double total_sec = 0.0;
};

struct SyntheticResidualRelinearizeStats {
    double factor_relinearize_sec = 0.0;
    double reset_state_sec = 0.0;
    double total_sec = 0.0;
};

struct SyntheticResidualGraphWorkspace {
    gbp::FactorGraph graph;
    std::vector<gbp::VariableNode*> vars;
    std::vector<gbp::Factor*> edge_factors;
    gbp::Factor* anchor_factor = nullptr;
    double tiny_prior = 1e-12;
};

std::vector<Eigen::VectorXd> emptyMeasurementBlocks(const Eigen::VectorXd&) {
    return {};
}

std::vector<Eigen::MatrixXd> emptyJacobianBlocks(const Eigen::VectorXd&) {
    return {};
}

SyntheticResidualGraphWorkspace buildSyntheticResidualGraphWorkspace(
    const SyntheticSE2Problem& problem,
    double tiny_prior,
    SyntheticResidualWorkspaceBuildStats* stats = nullptr
) {
    const auto total_t0 = SteadyClock::now();

    SyntheticResidualGraphWorkspace workspace;
    workspace.tiny_prior = tiny_prior;
    workspace.graph.nonlinear_factors = false;
    workspace.graph.eta_damping = 0.0;

    const int n = static_cast<int>(problem.gt_poses.size());
    workspace.graph.var_nodes.reserve(n);
    workspace.graph.var_residual.reserve(n);
    workspace.graph.factors.reserve(problem.edges.size() + 1);
    workspace.vars.resize(n, nullptr);
    workspace.edge_factors.resize(problem.edges.size(), nullptr);

    std::vector<int> variable_degree(n, 0);
    for (const SyntheticSE2Edge& edge : problem.edges) {
        ++variable_degree[edge.i];
        ++variable_degree[edge.j];
    }
    if (!variable_degree.empty()) {
        ++variable_degree[0];
    }

    const auto vars_t0 = SteadyClock::now();
    for (int i = 0; i < n; ++i) {
        gbp::VariableNode* var = workspace.graph.addVariable(i, 3);
        var->GT = problem.gt_poses[i];
        var->adj_factors.reserve(variable_degree[i]);
        var->adj_factors_raw.reserve(variable_degree[i]);
        var->prior.setLam(tiny_prior * Eigen::Matrix3d::Identity());
        var->prior.setEta(Eigen::Vector3d::Zero());
        var->belief.setLam(tiny_prior * Eigen::Matrix3d::Identity());
        var->belief.setEta(Eigen::Vector3d::Zero());
        var->mu = Eigen::Vector3d::Zero();
        var->markMuCurrent();
        workspace.vars[i] = var;
    }
    const auto vars_t1 = SteadyClock::now();

    int factor_id = 0;
    double factor_alloc_sec = 0.0;
    double connect_sec = 0.0;

    for (size_t edge_index = 0; edge_index < problem.edges.size(); ++edge_index) {
        const SyntheticSE2Edge& edge = problem.edges[edge_index];
        gbp::VariableNode* vi = workspace.vars.at(edge.i);
        gbp::VariableNode* vj = workspace.vars.at(edge.j);

        const auto factor_t0 = SteadyClock::now();
        gbp::Factor* factor = workspace.graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{vi, vj},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        const auto factor_t1 = SteadyClock::now();
        factor_alloc_sec += elapsedSeconds(factor_t0, factor_t1);

        const auto connect_t0 = SteadyClock::now();
        workspace.graph.connect(factor, vi, 0);
        workspace.graph.connect(factor, vj, 1);
        const auto connect_t1 = SteadyClock::now();
        connect_sec += elapsedSeconds(connect_t0, connect_t1);

        workspace.edge_factors[edge_index] = factor;
    }

    {
        gbp::VariableNode* v0 = workspace.vars.at(0);
        const auto factor_t0 = SteadyClock::now();
        workspace.anchor_factor = workspace.graph.addFactor(
            factor_id++,
            std::vector<gbp::VariableNode*>{v0},
            {},
            {},
            emptyMeasurementBlocks,
            emptyJacobianBlocks
        );
        const auto factor_t1 = SteadyClock::now();
        factor_alloc_sec += elapsedSeconds(factor_t0, factor_t1);

        const auto connect_t0 = SteadyClock::now();
        workspace.graph.connect(workspace.anchor_factor, v0, 0);
        const auto connect_t1 = SteadyClock::now();
        connect_sec += elapsedSeconds(connect_t0, connect_t1);
    }

    if (stats) {
        stats->variable_alloc_sec = elapsedSeconds(vars_t0, vars_t1);
        stats->factor_alloc_sec = factor_alloc_sec;
        stats->connect_sec = connect_sec;
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
    return workspace;
}

void resetSyntheticResidualState(
    SyntheticResidualGraphWorkspace& workspace,
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
        if (d == 2) {
            var->mu2.setZero();
        }
        var->mu.setZero();
        var->markMuCurrent();
    };

    if (use_parallel) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
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

void relinearizeSyntheticResidualGraph(
    SyntheticResidualGraphWorkspace& workspace,
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    int num_threads,
    SyntheticResidualRelinearizeStats* stats = nullptr,
    const RobustLossConfig& robust_loss_config = {}
) {
    const auto total_t0 = SteadyClock::now();
    const bool use_parallel = (num_threads != 1) && (problem.edges.size() > 64);

    const auto factor_t0 = SteadyClock::now();
    if (use_parallel) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J =
                analyticEdgeResidualJacobian(base_i, base_j, edge.measurement, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;
            factor->setLinearFactorInfo(eta, lam);
            factor->clearMessages();
        }
    } else {
        for (int edge_index = 0; edge_index < static_cast<int>(problem.edges.size()); ++edge_index) {
            const SyntheticSE2Edge& edge = problem.edges[edge_index];
            gbp::Factor* factor = workspace.edge_factors[edge_index];
            const Eigen::Vector3d& base_i = base_poses.at(edge.i);
            const Eigen::Vector3d& base_j = base_poses.at(edge.j);
            const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(edge.measurement), se2Between(base_i, base_j)));
            const double robust_weight = robustWeightForResidual(err0, edge.information, robust_loss_config);
            const Eigen::Matrix3d weighted_information = robust_weight * edge.information;
            const Eigen::Matrix<double, 3, 6> J =
                analyticEdgeResidualJacobian(base_i, base_j, edge.measurement, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
            const Eigen::Matrix<double, 6, 6> lam = J.transpose() * weighted_information * J;
            const Eigen::Matrix<double, 6, 1> eta = -J.transpose() * weighted_information * err0;
            factor->setLinearFactorInfo(eta, lam);
            factor->clearMessages();
        }
    }

    {
        const Eigen::Vector3d& base_anchor = base_poses.at(0);
        const Eigen::Vector3d err0 = se2Log(se2Compose(se2Inverse(problem.anchor_pose), base_anchor));
        const Eigen::Matrix3d J =
            analyticAnchorResidualJacobian(base_anchor, problem.anchor_pose, Eigen::Vector3d::Zero());
        const Eigen::Matrix3d lam = J.transpose() * problem.anchor_information * J;
        const Eigen::Vector3d eta = -J.transpose() * problem.anchor_information * err0;
        workspace.anchor_factor->setLinearFactorInfo(eta, lam);
        workspace.anchor_factor->clearMessages();
    }
    const auto factor_t1 = SteadyClock::now();

    const auto reset_t0 = SteadyClock::now();
    resetSyntheticResidualState(workspace, num_threads);
    const auto reset_t1 = SteadyClock::now();

    if (stats) {
        stats->factor_relinearize_sec = elapsedSeconds(factor_t0, factor_t1);
        stats->reset_state_sec = elapsedSeconds(reset_t0, reset_t1);
        stats->total_sec = elapsedSeconds(total_t0, SteadyClock::now());
    }
}

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& base_poses,
    double tiny_prior,
    const RobustLossConfig& robust_loss_config
) {
    SyntheticResidualGraphWorkspace workspace = buildSyntheticResidualGraphWorkspace(problem, tiny_prior);
    relinearizeSyntheticResidualGraph(workspace, problem, base_poses, 1, nullptr, robust_loss_config);
    return std::move(workspace.graph);
}

ExperimentResults runSyntheticSE2DirectReference(
    const SyntheticSE2Problem& problem, int num_outer,
    const RobustLossConfig& robust_loss_config
) {
    if(num_outer<1) throw std::runtime_error("Direct reference requires positive outer count");
    ExperimentResults results;
    const auto solver_begin=SteadyClock::now();
    auto workspace=buildSyntheticResidualGraphWorkspace(problem,1e-12,nullptr);
    PoseVec poses=problem.init_poses;
    results.direct_history.push_back(OuterDirectRow{0,nonlinearObjective(problem,poses,1),0.,0.});
    results.direct_pose_history.push_back(poses);
    results.num_poses=static_cast<int>(problem.init_poses.size());
    results.num_edges=static_cast<int>(problem.edges.size());
    results.initial_objective=results.direct_history.front().nonlinear_objective;
    results.initial_poses=problem.init_poses;
    // Same serial reference loop as the original strict-compare implementation.
    // Its state, matrix and poses are completely separate from H-GBP.
    for(int outer=1;outer<=num_outer;++outer) {
        const auto begin=SteadyClock::now();
        SyntheticResidualRelinearizeStats relin;
        relinearizeSyntheticResidualGraph(workspace,problem,poses,1,&relin,robust_loss_config);
        const auto built=SteadyClock::now();
        const auto joint=workspace.graph.jointDistributionInfSparse();
        const auto lam=symmetrizeSparse(joint.lam);
        const auto assembled=SteadyClock::now();
        const Eigen::VectorXd step=solveSparseCholesky(lam,joint.eta,0.);
        const auto solved=SteadyClock::now();
        poses=applyPoseDeltas(poses,step);
        const auto applied=SteadyClock::now();
        const double cost=nonlinearObjective(problem,poses,1);
        const auto scored=SteadyClock::now();
        results.direct_history.push_back(OuterDirectRow{
            outer,cost,step.norm(),(joint.eta-lam*step).norm(),
            elapsedSeconds(begin,scored),elapsedSeconds(begin,built),
            relin.factor_relinearize_sec,relin.reset_state_sec,
            elapsedSeconds(built,assembled),elapsedSeconds(assembled,solved),
            elapsedSeconds(solved,applied),elapsedSeconds(applied,scored)});
        results.direct_pose_history.push_back(poses);
    }
    results.solver_wall_sec=elapsedSeconds(solver_begin,SteadyClock::now());
    results.direct_solver_wall_sec=results.solver_wall_sec;
    return results;
}
