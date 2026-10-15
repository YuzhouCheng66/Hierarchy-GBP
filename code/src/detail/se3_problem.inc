SE3ChartTransition se3ChartTransition(const SE3Pose& previous,const SE3Pose& current) {
    const SE3Pose relative=se3Between(previous,current);
    const Vec6 d=se3Log(relative);
    if(d.tail<3>().norm()<1e-3) {
        Mat6 ad=Mat6::Zero();
        ad.topLeftCorner<3,3>()=skew(d.tail<3>());
        ad.bottomRightCorner<3,3>()=ad.topLeftCorner<3,3>();
        ad.topRightCorner<3,3>()=skew(d.head<3>());
        const Mat6 a2=ad*ad,a4=a2*a2,a6=a4*a2;
        return {d,Mat6::Identity()+.5*ad+a2/12.-a4/720.+a6/30240.};
    }
    return {d,se3LogmapDerivative(relative)};
}

SyntheticSE3Problem loadSyntheticSE3Problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open SE3 problem file: " + path);
    }

    std::unordered_map<int, SE3Pose> raw_vertices;
    struct RawEdge {
        int vi = -1;
        int vj = -1;
        SE3Pose measurement;
        Mat6 information = Mat6::Zero();
    };
    std::vector<RawEdge> raw_edges;

    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        std::istringstream iss(line);
        std::string tag;
        if (!(iss >> tag)) {
            continue;
        }

        if (tag == "VERTEX_SE3:QUAT") {
            int vid = -1;
            SE3Pose pose;
            if (!(iss >> vid
                >> pose.t.x() >> pose.t.y() >> pose.t.z()
                >> pose.q.x() >> pose.q.y() >> pose.q.z() >> pose.q.w())) {
                throw std::runtime_error("Malformed VERTEX_SE3:QUAT line in " + path + ":" + std::to_string(lineno));
            }
            pose = normalizedPose(pose);
            raw_vertices[vid] = pose;
        } else if (tag == "EDGE_SE3:QUAT") {
            RawEdge edge;
            std::array<double, 21> info_vals{};
            if (!(iss >> edge.vi >> edge.vj
                >> edge.measurement.t.x() >> edge.measurement.t.y() >> edge.measurement.t.z()
                >> edge.measurement.q.x() >> edge.measurement.q.y() >> edge.measurement.q.z() >> edge.measurement.q.w()
                >> info_vals[0] >> info_vals[1] >> info_vals[2] >> info_vals[3] >> info_vals[4] >> info_vals[5]
                >> info_vals[6] >> info_vals[7] >> info_vals[8] >> info_vals[9] >> info_vals[10]
                >> info_vals[11] >> info_vals[12] >> info_vals[13] >> info_vals[14]
                >> info_vals[15] >> info_vals[16] >> info_vals[17]
                >> info_vals[18] >> info_vals[19]
                >> info_vals[20])) {
                throw std::runtime_error("Malformed EDGE_SE3:QUAT line in " + path + ":" + std::to_string(lineno));
            }
            edge.measurement = normalizedPose(edge.measurement);
            edge.information = symmetrizedInformationMatrix(info21ToMatrix(info_vals));
            raw_edges.push_back(edge);
        } else if (tag == "VERTEX_SE2" || tag == "EDGE_SE2") {
            throw std::runtime_error("Expected an SE3 g2o file but found SE2 tag in " + path);
        } else {
            throw std::runtime_error("Unsupported tag in g2o file " + path + ":" + std::to_string(lineno) + " tag=" + tag);
        }
    }

    if (raw_vertices.empty()) {
        throw std::runtime_error("No VERTEX_SE3:QUAT entries found in " + path);
    }

    std::vector<int> original_ids;
    original_ids.reserve(raw_vertices.size());
    for (const auto& kv : raw_vertices) {
        original_ids.push_back(kv.first);
    }
    std::sort(original_ids.begin(), original_ids.end());

    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(original_ids.size());
    for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
        id_to_idx[original_ids[idx]] = idx;
    }

    SyntheticSE3Problem problem;
    problem.gt_poses.resize(original_ids.size());
    problem.init_poses.resize(original_ids.size());
    for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
        const SE3Pose pose = raw_vertices.at(original_ids[idx]);
        problem.gt_poses[idx] = pose;
        problem.init_poses[idx] = pose;
    }

    problem.anchor_pose = problem.init_poses.front();
    problem.anchor_information = symmetrizedInformationMatrix(1e8 * Mat6::Identity());

    problem.edges.reserve(raw_edges.size());
    for (const RawEdge& raw_edge : raw_edges) {
        const auto it_i = id_to_idx.find(raw_edge.vi);
        const auto it_j = id_to_idx.find(raw_edge.vj);
        if (it_i == id_to_idx.end() || it_j == id_to_idx.end()) {
            throw std::runtime_error("g2o edge references unknown vertex id in " + path);
        }
        SyntheticSE3Edge edge;
        edge.i = it_i->second;
        edge.j = it_j->second;
        edge.measurement = raw_edge.measurement;
        edge.information = raw_edge.information;
        edge.kind = (std::abs(raw_edge.vi - raw_edge.vj) == 1) ? "odometry" : "loop";
        problem.edges.push_back(edge);
    }

    return problem;
}

SE3ObjectiveValues evaluateSE3ObjectivesBuffered(const SyntheticSE3Problem& problem,
    const SE3PoseVector& poses,const RobustLossConfig& config,SE3ObjectiveWorkspace& workspace,int threads) {
    workspace.edge_values.resize(problem.edges.size());
    const int thread_count=effectiveThreadCount(threads);
    const double delta=config.huber_delta;
    #pragma omp parallel for schedule(static) num_threads(thread_count) if(thread_count>1)
    for(int i=0;i<static_cast<int>(problem.edges.size());++i) {
        const auto& edge=problem.edges[i];
        const SE3Pose pred=se3Between(poses[edge.i],poses[edge.j]);
        const Vec6 err=se3Log(se3Compose(se3Inverse(edge.measurement),pred));
        const double quadratic=(err.transpose()*edge.information*err)(0,0);
        workspace.edge_values[i].raw=.5*quadratic;
        workspace.edge_values[i].huber=(delta>0 && quadratic>delta*delta)
            ? delta*(std::sqrt(quadratic)-.5*delta) : .5*quadratic;
    }
    SE3ObjectiveValues values;
    // Preserve serial edge summation order; no thread-dependent reduction tree.
    for(const auto& edge:workspace.edge_values) {
        values.raw+=edge.raw;
        values.huber+=edge.huber;
    }
    const Vec6 anchor=se3Log(se3Compose(se3Inverse(problem.anchor_pose),poses.at(0)));
    const double prior=.5*(anchor.transpose()*problem.anchor_information*anchor)(0,0);
    values.raw+=prior; values.huber+=prior;
    return values;
}

void applySE3PoseDeltasInto(const SE3PoseVector& base,const Eigen::VectorXd& delta,
                          double scale,SE3PoseVector& output,int threads) {
    if(delta.size()!=6*static_cast<int>(base.size()) || &base==&output)
        throw std::runtime_error("Invalid buffered SE3 pose update");
    output.resize(base.size());
    const int thread_count=effectiveThreadCount(threads);
    #pragma omp parallel for schedule(static) num_threads(thread_count) if(thread_count>1)
    for(int i=0;i<static_cast<int>(base.size());++i) {
        const Vec6 step=scale*delta.segment<6>(6*i);
        output[i]=se3Plus(base[i],step);
    }
}

SE3ObjectiveValues evaluateSE3Objectives(const SyntheticSE3Problem& problem,
                                       const SE3PoseVector& poses,
                                       const RobustLossConfig& config) {
    SE3ObjectiveValues values;
    const double delta=config.huber_delta;
    for(const auto& edge:problem.edges) {
        const SE3Pose pred=se3Between(poses.at(edge.i),poses.at(edge.j));
        const Vec6 err=se3Log(se3Compose(se3Inverse(edge.measurement),pred));
        const double quadratic=(err.transpose()*edge.information*err)(0,0);
        values.raw+=0.5*quadratic;
        values.huber+=(delta>0 && quadratic>delta*delta)
            ? delta*(std::sqrt(quadratic)-0.5*delta) : 0.5*quadratic;
    }
    const Vec6 anchor=se3Log(se3Compose(se3Inverse(problem.anchor_pose),poses.at(0)));
    const double prior=0.5*(anchor.transpose()*problem.anchor_information*anchor)(0,0);
    values.raw+=prior;
    values.huber+=prior;
    return values;
}

double nonlinearObjective(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& poses
) {
    double total = 0.0;
    for (const SyntheticSE3Edge& edge : problem.edges) {
        const SE3Pose pred = se3Between(poses.at(edge.i), poses.at(edge.j));
        const Vec6 err = se3Log(se3Compose(se3Inverse(edge.measurement), pred));
        total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
    }
    const Vec6 anchor_err = se3Log(se3Compose(se3Inverse(problem.anchor_pose), poses.at(0)));
    total += 0.5 * (anchor_err.transpose() * problem.anchor_information * anchor_err)(0, 0);
    return total;
}

SE3PoseVector applyPoseDeltas(
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    if (delta_vec.size() != static_cast<int>(base_poses.size()) * 6) {
        throw std::runtime_error("SE3 delta vector size does not match pose count");
    }

    SE3PoseVector out(base_poses.size());
    for (size_t i = 0; i < base_poses.size(); ++i) {
        const Vec6 delta = delta_vec.segment<6>(static_cast<int>(6 * i));
        out[i] = se3Plus(base_poses[i], delta);
    }
    return out;
}

double nonlinearObjectiveForDelta(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    return nonlinearObjective(problem, applyPoseDeltas(base_poses, delta_vec));
}

gbp::FactorGraph buildLinearizedResidualGraph(
    const SyntheticSE3Problem& problem,
    const SE3PoseVector& base_poses,
    double tiny_prior,
    const RobustLossConfig& robust_loss_config
) {
    gbp::FactorGraph graph;
    graph.eta_damping = se3MessageDamping();
    std::vector<gbp::VariableNode*> vars(base_poses.size(), nullptr);
    const Mat6 weak_lam = tiny_prior * Mat6::Identity();

    for (int i = 0; i < static_cast<int>(base_poses.size()); ++i) {
        gbp::VariableNode* var = graph.addVariable(i, 6);
        var->prior.setLam(weak_lam);
        var->prior.setEta(Eigen::VectorXd::Zero(6));
        var->belief.setLam(weak_lam);
        var->belief.setEta(Eigen::VectorXd::Zero(6));
        var->mu = Eigen::VectorXd::Zero(6);
        var->markMuCurrent();
        vars[i] = var;
    }

    int fid = 0;
    for (const SyntheticSE3Edge& edge : problem.edges) {
        const Vec6 r0 = edgeResidual(
            base_poses.at(edge.i),
            base_poses.at(edge.j),
            edge.measurement,
            Vec6::Zero(),
            Vec6::Zero()
        );
        const Mat6x12 J = analyticEdgeResidualJacobian(
            base_poses.at(edge.i),
            base_poses.at(edge.j),
            edge.measurement
        );
        const double robust_weight = robustWeightForResidual(r0, edge.information, robust_loss_config);
        const Mat6 weighted_information = robust_weight * edge.information;

        auto meas_fn = [J](const Eigen::VectorXd& x) {
            return std::vector<Eigen::VectorXd>{J * x};
        };
        auto jac_fn = [J](const Eigen::VectorXd&) {
            return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd(J)};
        };

        gbp::Factor* factor = graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vars[edge.i], vars[edge.j]},
            std::vector<Eigen::VectorXd>{-r0},
            std::vector<Eigen::MatrixXd>{weighted_information},
            meas_fn,
            jac_fn
        );
        factor->computeFactor(Eigen::VectorXd::Zero(12), true);
        graph.connect(factor, vars[edge.i], 0);
        graph.connect(factor, vars[edge.j], 1);
    }

    if (!vars.empty()) {
        const Vec6 r0 = anchorResidual(base_poses.at(0), problem.anchor_pose, Vec6::Zero());
        const Mat6 J = analyticAnchorResidualJacobian(base_poses.at(0), problem.anchor_pose);
        auto meas_fn = [J](const Eigen::VectorXd& x) {
            return std::vector<Eigen::VectorXd>{J * x};
        };
        auto jac_fn = [J](const Eigen::VectorXd&) {
            return std::vector<Eigen::MatrixXd>{Eigen::MatrixXd(J)};
        };
        gbp::Factor* factor = graph.addFactor(
            fid++,
            std::vector<gbp::VariableNode*>{vars[0]},
            std::vector<Eigen::VectorXd>{-r0},
            std::vector<Eigen::MatrixXd>{problem.anchor_information},
            meas_fn,
            jac_fn
        );
        factor->computeFactor(Eigen::VectorXd::Zero(6), true);
        graph.connect(factor, vars[0], 0);
    }

    return graph;
}
