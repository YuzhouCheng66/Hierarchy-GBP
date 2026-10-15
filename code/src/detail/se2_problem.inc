SyntheticSE2Problem loadSyntheticSE2Problem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open synthetic SE2 problem file: " + path);
    }

    std::string token;
    in >> token;
    if (token == "SYNTHETIC_SE2_PROBLEM") {
        int n = 0;
        in >> token >> n;
        if (token != "N") {
            throw std::runtime_error("Expected N section in " + path);
        }

        SyntheticSE2Problem problem;
        problem.gt_poses.resize(n, Eigen::Vector3d::Zero());
        problem.init_poses.resize(n, Eigen::Vector3d::Zero());

        in >> token;
        if (token != "POSES") {
            throw std::runtime_error("Expected POSES section in " + path);
        }
        for (int idx = 0; idx < n; ++idx) {
            int pose_id = -1;
            in >> pose_id
               >> problem.gt_poses[idx](0) >> problem.gt_poses[idx](1) >> problem.gt_poses[idx](2)
               >> problem.init_poses[idx](0) >> problem.init_poses[idx](1) >> problem.init_poses[idx](2);
            if (pose_id != idx) {
                throw std::runtime_error("Pose ids must be contiguous in synthetic problem file");
            }
        }

        in >> token;
        if (token != "ANCHOR") {
            throw std::runtime_error("Expected ANCHOR section in " + path);
        }
        in >> problem.anchor_pose(0) >> problem.anchor_pose(1) >> problem.anchor_pose(2);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                in >> problem.anchor_information(r, c);
            }
        }

        int num_edges = 0;
        in >> token >> num_edges;
        if (token != "EDGES") {
            throw std::runtime_error("Expected EDGES section in " + path);
        }
        problem.edges.reserve(num_edges);
        for (int e = 0; e < num_edges; ++e) {
            SyntheticSE2Edge edge;
            in >> edge.i >> edge.j >> edge.kind
               >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    in >> edge.information(r, c);
                }
            }
            problem.edges.push_back(edge);
        }

        return problem;
    }

    if (token == "VERTEX_SE2" || token == "EDGE_SE2") {
        in.close();
        std::ifstream gin(path);
        if (!gin) {
            throw std::runtime_error("Failed to reopen g2o problem file: " + path);
        }

        std::unordered_map<int, Eigen::Vector3d> raw_vertices;
        struct RawEdge {
            int vi = -1;
            int vj = -1;
            Eigen::Vector3d measurement = Eigen::Vector3d::Zero();
            Eigen::Matrix3d information = Eigen::Matrix3d::Zero();
        };
        std::vector<RawEdge> raw_edges;

        std::string line;
        int lineno = 0;
        while (std::getline(gin, line)) {
            ++lineno;
            std::istringstream iss(line);
            std::string tag;
            if (!(iss >> tag)) {
                continue;
            }
            if (tag == "VERTEX_SE2") {
                int vid = -1;
                Eigen::Vector3d pose = Eigen::Vector3d::Zero();
                if (!(iss >> vid >> pose(0) >> pose(1) >> pose(2))) {
                    throw std::runtime_error("Malformed VERTEX_SE2 line in " + path + ":" + std::to_string(lineno));
                }
                raw_vertices[vid] = pose;
            } else if (tag == "EDGE_SE2") {
                RawEdge edge;
                std::array<double, 6> info_vals{};
                if (!(iss >> edge.vi >> edge.vj
                    >> edge.measurement(0) >> edge.measurement(1) >> edge.measurement(2)
                    >> info_vals[0] >> info_vals[1] >> info_vals[2]
                    >> info_vals[3] >> info_vals[4] >> info_vals[5])) {
                    throw std::runtime_error("Malformed EDGE_SE2 line in " + path + ":" + std::to_string(lineno));
                }
                edge.information = info6ToMatrix(info_vals);
                raw_edges.push_back(edge);
            } else {
                throw std::runtime_error("Unsupported tag in g2o file " + path + ":" + std::to_string(lineno) + " tag=" + tag);
            }
        }

        if (raw_vertices.empty()) {
            throw std::runtime_error("No VERTEX_SE2 entries found in g2o file: " + path);
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

        SyntheticSE2Problem problem;
        problem.gt_poses.resize(original_ids.size(), Eigen::Vector3d::Zero());
        problem.init_poses.resize(original_ids.size(), Eigen::Vector3d::Zero());
        for (int idx = 0; idx < static_cast<int>(original_ids.size()); ++idx) {
            const Eigen::Vector3d pose = raw_vertices.at(original_ids[idx]);
            problem.gt_poses[idx] = pose;
            problem.init_poses[idx] = pose;
        }
        problem.anchor_pose = problem.init_poses.front();
        problem.anchor_information = 1e8 * Eigen::Matrix3d::Identity();

        problem.edges.reserve(raw_edges.size());
        for (const RawEdge& raw_edge : raw_edges) {
            const auto it_i = id_to_idx.find(raw_edge.vi);
            const auto it_j = id_to_idx.find(raw_edge.vj);
            if (it_i == id_to_idx.end() || it_j == id_to_idx.end()) {
                throw std::runtime_error("g2o edge references unknown vertex id in " + path);
            }
            SyntheticSE2Edge edge;
            edge.i = it_i->second;
            edge.j = it_j->second;
            edge.measurement = raw_edge.measurement;
            edge.information = raw_edge.information;
            edge.kind = (std::abs(raw_edge.vi - raw_edge.vj) == 1) ? "odometry" : "loop";
            problem.edges.push_back(edge);
        }

        return problem;
    }

    throw std::runtime_error("Unexpected problem header in " + path + ": " + token);
}

double nonlinearObjective(
    const SyntheticSE2Problem& problem,
    const std::vector<Eigen::Vector3d>& poses,
    int num_threads
) {
    double total = 0.0;
    const int thread_count = std::max(1, num_threads <= 0 ? omp_get_max_threads() : num_threads);
    const bool use_parallel = thread_count > 1 && static_cast<int>(problem.edges.size()) >= 256;
    if (!use_parallel) {
        for (const SyntheticSE2Edge& edge : problem.edges) {
            const Eigen::Vector3d pred = se2Between(poses.at(edge.i), poses.at(edge.j));
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(edge.measurement), pred));
            total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
        }
    } else {
        #pragma omp parallel for schedule(static) reduction(+:total) num_threads(thread_count)
        for (int edge_idx = 0; edge_idx < static_cast<int>(problem.edges.size()); ++edge_idx) {
            const SyntheticSE2Edge& edge = problem.edges[edge_idx];
            const Eigen::Vector3d pred = se2Between(poses.at(edge.i), poses.at(edge.j));
            const Eigen::Vector3d err = se2Log(se2Compose(se2Inverse(edge.measurement), pred));
            total += 0.5 * (err.transpose() * edge.information * err)(0, 0);
        }
    }
    const Eigen::Vector3d anchor_err = se2Log(se2Compose(se2Inverse(problem.anchor_pose), poses.at(0)));
    total += 0.5 * (anchor_err.transpose() * problem.anchor_information * anchor_err)(0, 0);
    return total;
}

std::vector<Eigen::Vector3d> applyPoseDeltas(
    const std::vector<Eigen::Vector3d>& base_poses,
    const Eigen::VectorXd& delta_vec
) {
    std::vector<Eigen::Vector3d> out(base_poses.size(), Eigen::Vector3d::Zero());
    for (int i = 0; i < static_cast<int>(base_poses.size()); ++i) {
        out[i] = se2Plus(base_poses[i], delta_vec.segment<3>(3 * i));
    }
    return out;
}
