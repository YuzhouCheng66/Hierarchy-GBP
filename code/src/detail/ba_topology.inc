#include "ba_aggregation.h"

struct HGBPConfig {
    std::string aggregation = "connected";
    std::string coarse_operator = "additive";
    int graph_neighbors = 4;
    int coarse_groups = 24;
    double linear_rel_tol = 0.1;
    double variance_rel_tol = 1e-6;
    std::string linear_controller = "fcg";
    std::string gbp_model = "normalized";
    int pair_sample_cap = 16;
    double initial_lambda = 1e-4;
    double pair_factor_scale = 1.0;
    int krylov_start_outer = 1;
};

HGBPConfig parseHGBPConfig(
    int argc,
    char** argv,
    std::vector<std::string>& filtered_storage,
    std::vector<char*>& filtered_argv
) {
    HGBPConfig config;
    filtered_storage.reserve(static_cast<size_t>(argc));
    filtered_storage.emplace_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--aggregation" && i + 1 < argc) {
            config.aggregation = argv[++i];
            if (config.aggregation != "connected") {
                throw std::runtime_error("--aggregation must be connected");
            }
        } else if (arg == "--coarse-operator" && i + 1 < argc) {
            config.coarse_operator = argv[++i];
            if (config.coarse_operator != "additive") {
                throw std::runtime_error("--coarse-operator must be additive");
            }
        } else if (arg == "--graph-neighbors" && i + 1 < argc) {
            parseInt(argv[++i], config.graph_neighbors);
        } else if (arg == "--coarse-groups" && i + 1 < argc) {
            parseInt(argv[++i], config.coarse_groups);
        } else if (arg == "--linear-rel-tol" && i + 1 < argc) {
            parseDouble(argv[++i], config.linear_rel_tol);
        } else if (arg == "--variance-rel-tol" && i + 1 < argc) {
            parseDouble(argv[++i], config.variance_rel_tol);
        } else if (arg == "--linear-controller" && i + 1 < argc) {
            config.linear_controller = argv[++i];
            if (config.linear_controller != "fcg") {
                throw std::runtime_error("--linear-controller must be fcg");
            }
        } else if (arg == "--gbp-model" && i + 1 < argc) {
            config.gbp_model = argv[++i];
            if (config.gbp_model != "normalized") {
                throw std::runtime_error("--gbp-model must be normalized");
            }
        } else if (arg == "--initial-lambda" && i + 1 < argc) {
            parseDouble(argv[++i], config.initial_lambda);
        } else if (arg == "--pair-sample-cap" && i + 1 < argc) {
            parseInt(argv[++i], config.pair_sample_cap);
        } else if (arg == "--pair-factor-scale" && i + 1 < argc) {
            parseDouble(argv[++i], config.pair_factor_scale);
            if (config.pair_factor_scale != 1.0) {
                throw std::runtime_error("--pair-factor-scale must be 1");
            }
        } else if (arg == "--krylov-start-outer" && i + 1 < argc) {
            parseInt(argv[++i], config.krylov_start_outer);
            if (config.krylov_start_outer != 1) {
                throw std::runtime_error("--krylov-start-outer must be 1");
            }
        } else if (arg == "--fine-smoother" && i + 1 < argc) {
            if (std::string(argv[++i]) != "gbp") {
                throw std::runtime_error("--fine-smoother must be gbp");
            }
        } else {
            if (arg == "--help" || arg == "-h") {
                std::cout
                    << "Uniform BA: additive + FCG + normalized GBP + connected aggregates\n"
                    << "  --aggregation connected --coarse-operator additive\n"
                    << "  --linear-controller fcg --gbp-model normalized --fine-smoother gbp\n"
                    << "  --graph-neighbors <4> --coarse-groups <24> --pair-sample-cap <16>\n"
                    << "  --linear-rel-tol <0.1> --variance-rel-tol <1e-6>\n"
                    << "  --initial-lambda <1e-4> --pair-factor-scale 1 --krylov-start-outer 1\n";
            }
            filtered_storage.push_back(arg);
        }
    }
    if (config.graph_neighbors < 1 || config.coarse_groups < 1 ||
        config.pair_sample_cap < 1) {
        throw std::runtime_error("graph-neighbors, coarse-groups and pair-sample-cap must be positive");
    }
    if (config.linear_rel_tol < 0.0 || config.linear_rel_tol >= 1.0) {
        throw std::runtime_error("linear-rel-tol must be in [0,1)");
    }
    if (config.variance_rel_tol <= 0.0 || config.variance_rel_tol >= 1.0) {
        throw std::runtime_error("variance-rel-tol must be in (0,1)");
    }
    if (config.initial_lambda < 1e-16) {
        throw std::runtime_error("initial-lambda must be at least 1e-16");
    }
    filtered_argv.reserve(filtered_storage.size());
    for (std::string& arg : filtered_storage) {
        filtered_argv.push_back(arg.data());
    }
    return config;
}

using RootProblem = rootba::BalProblem<double>;

RootProblem makeRootProblem(const BALProblem& problem) {
    RootProblem root_problem;
    root_problem.set_quiet(true);
    root_problem.cameras().resize(static_cast<size_t>(problem.num_cameras));
    root_problem.landmarks().resize(static_cast<size_t>(problem.num_points));

    const RootProblem::SO3 axis_inversion(Vec3(1.0, -1.0, -1.0).asDiagonal());
    for (int c = 0; c < problem.num_cameras; ++c) {
        const Vec9& source = problem.cameras[static_cast<size_t>(c)];
        auto& target = root_problem.cameras()[static_cast<size_t>(c)];
        target.T_c_w.so3() =
            axis_inversion * RootProblem::SO3::exp(source.template head<3>());
        target.T_c_w.translation() =
            axis_inversion * source.template segment<3>(3);
        target.intrinsics = RootProblem::CameraModel(source.template tail<3>());
    }

    for (int p = 0; p < problem.num_points; ++p) {
        root_problem.landmarks()[static_cast<size_t>(p)].p_w =
            problem.points[static_cast<size_t>(p)];
    }
    for (const Observation& source : problem.observations) {
        auto& target =
            root_problem.landmarks()[static_cast<size_t>(source.point)]
                .obs[static_cast<size_t>(source.camera)];
        target.pos = Vec2(source.xy.x(), -source.xy.y());
    }
    return root_problem;
}

struct RootMetrics {
    double cost2 = 0.0;
    double rmse_px = 0.0;
    double are_px = 0.0;
};

struct RootHGBPBuildPlan {
    std::vector<int> block_offset;
    std::vector<int> free_camera;
    std::vector<int> global_edge;
    std::vector<unsigned char> edge_used;
    std::vector<int> retained_edge_slot;
    int retained_edge_count = 0;
};

struct RootHGBPTopology {
    PackedBlockSchurPattern pattern;
    RootHGBPBuildPlan build_plan;
    int strong_camera_pairs = 0;
    int strong_active_cameras = 0;
    int selected_active_cameras = 0;
    int selected_components = 0;
    int selected_largest_component = 0;
    bool spanning_backbone_used = false;
};

#include "ba_cross_cache.h"

void compactRootPairEdges(
    PackedBlockSchurPattern& pattern,
    RootHGBPBuildPlan& plan
) {
    plan.retained_edge_slot.assign(plan.edge_used.size(), -1);
    int retained_edge_count = 0;
    for (size_t edge = 0; edge < plan.edge_used.size(); ++edge) {
        if (plan.edge_used[edge] != 0) {
            plan.retained_edge_slot[edge] = retained_edge_count++;
        }
    }
    plan.retained_edge_count = retained_edge_count;
    for (int& edge : pattern.slot_pair_edge_a_global) {
        edge = plan.retained_edge_slot[static_cast<size_t>(edge)];
    }
    for (int& edge : pattern.slot_pair_edge_b_global) {
        edge = plan.retained_edge_slot[static_cast<size_t>(edge)];
    }
}

RootHGBPTopology buildRootHGBPTopology(
    const BALProblem& problem,
    const RootProblem& root_problem,
    int graph_neighbors,
    int requested_threads
) {
    const auto t0 = Clock::now();
    RootHGBPTopology topology;
    PackedBlockSchurPattern& pattern = topology.pattern;
    RootHGBPBuildPlan& plan = topology.build_plan;
    const auto& landmarks = root_problem.landmarks();
    const int n = problem.num_cameras;
    const size_t point_count = landmarks.size();

    pattern.total_cameras = n;
    pattern.free_cameras = n;
    pattern.fix_first_camera = false;
    pattern.camera_to_free.resize(static_cast<size_t>(n));
    pattern.free_to_camera.resize(static_cast<size_t>(n));
    for (int c = 0; c < n; ++c) {
        pattern.camera_to_free[static_cast<size_t>(c)] = c;
        pattern.free_to_camera[static_cast<size_t>(c)] = c;
    }

    plan.block_offset.resize(point_count + 1, 0);
    pattern.point_edge_offset.resize(point_count + 1, 0);
    for (size_t p = 0; p < point_count; ++p) {
        const int next =
            plan.block_offset[p] +
            static_cast<int>(landmarks[p].obs.size());
        plan.block_offset[p + 1] = next;
        pattern.point_edge_offset[p + 1] = next;
    }
    pattern.total_point_edges = plan.block_offset.back();
    plan.free_camera.resize(
        static_cast<size_t>(pattern.total_point_edges));
    plan.global_edge.resize(
        static_cast<size_t>(pattern.total_point_edges));
    plan.edge_used.assign(
        static_cast<size_t>(pattern.total_point_edges), 0);

    const size_t dense_size =
        static_cast<size_t>(n) * static_cast<size_t>(n);
    std::vector<int> pair_count(dense_size, 0);
    const int topology_threads = effectiveBuildThreads(
        requested_threads,
        static_cast<int>(point_count),
        n);
    int invalid_camera = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(static) reduction(|:invalid_camera)
#endif
    for (int p_index = 0;
         p_index < static_cast<int>(point_count);
         ++p_index) {
        const size_t p = static_cast<size_t>(p_index);
        const int edge_base = plan.block_offset[p];
        int edge_cursor = edge_base;
        for (const auto& [camera_index, observation] :
             landmarks[p].obs) {
            (void)observation;
            const int camera = static_cast<int>(camera_index);
            if (camera < 0 || camera >= n) {
                invalid_camera = 1;
                continue;
            }
            const int edge = edge_cursor++;
            plan.free_camera[static_cast<size_t>(edge)] = camera;
            plan.global_edge[static_cast<size_t>(edge)] = edge;
        }
    }
    if (invalid_camera != 0) {
        throw std::runtime_error(
            "RootBA camera index is outside H-GBP topology");
    }
    std::vector<int> camera_observations(static_cast<size_t>(n), 0);
    for (int camera : plan.free_camera) {
        ++camera_observations[static_cast<size_t>(camera)];
    }
    std::vector<int> camera_point_offsets(static_cast<size_t>(n + 1), 0);
    std::partial_sum(camera_observations.begin(), camera_observations.end(),
        camera_point_offsets.begin() + 1);
    std::vector<int> camera_points(plan.free_camera.size());
    std::vector<int> cursor = camera_point_offsets;
    for (size_t p = 0; p < point_count; ++p) {
        for (int e = plan.block_offset[p]; e < plan.block_offset[p + 1]; ++e) {
            const int camera = plan.free_camera[static_cast<size_t>(e)];
            camera_points[static_cast<size_t>(cursor[camera]++)] = static_cast<int>(p);
        }
    }
    // Each row has one owner: no atomics, thread-local dense matrices, or merge pass.
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(dynamic, 1)
#endif
    for (int camera = 0; camera < n; ++camera) {
        int* row = pair_count.data() + static_cast<size_t>(camera) * n;
        for (int pos = camera_point_offsets[camera];
             pos < camera_point_offsets[camera + 1]; ++pos) {
            const int p = camera_points[static_cast<size_t>(pos)];
            for (int e = plan.block_offset[p]; e < plan.block_offset[p + 1]; ++e) {
                ++row[plan.free_camera[static_cast<size_t>(e)]];
            }
        }
        row[camera] = 1;
    }

    // Dimensionless covisibility avoids thresholds tied to a dataset's track density.
    auto pair_weight = [&](int row, int col) {
        const int count = pair_count[static_cast<size_t>(row) * n + col];
        const double denominator = std::sqrt(
            static_cast<double>(std::max(1, camera_observations[row])) *
            static_cast<double>(std::max(1, camera_observations[col])));
        return static_cast<double>(count) / denominator;
    };
    std::vector<unsigned char> selected_pair(dense_size, 0);
    for (int row = 0; row < n; ++row) {
        std::vector<std::pair<double, int>> neighbors;
        for (int col = 0; col < n; ++col) {
            if (row != col && pair_count[static_cast<size_t>(row) * n + col] > 0) {
                neighbors.emplace_back(pair_weight(row, col), col);
            }
        }
        const int keep = std::min(graph_neighbors, static_cast<int>(neighbors.size()));
        std::partial_sort(neighbors.begin(), neighbors.begin() + keep, neighbors.end(),
            [](const auto& a, const auto& b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
        selected_pair[static_cast<size_t>(row) * n + row] = 1;
        for (int k = 0; k < keep; ++k) {
            const int col = neighbors[static_cast<size_t>(k)].second;
            selected_pair[static_cast<size_t>(row) * n + col] = 1;
            selected_pair[static_cast<size_t>(col) * n + row] = 1;
        }
    }
    std::vector<unsigned char> strong_active(static_cast<size_t>(n), 0);
    for (int row = 0; row < n; ++row) {
        const size_t base =
            static_cast<size_t>(row) * static_cast<size_t>(n);
        for (int col = row + 1; col < n; ++col) {
            if (selected_pair[base + static_cast<size_t>(col)]) {
                ++topology.strong_camera_pairs;
                strong_active[static_cast<size_t>(row)] = 1;
                strong_active[static_cast<size_t>(col)] = 1;
            }
        }
    }
    topology.strong_active_cameras = static_cast<int>(std::count(
        strong_active.begin(), strong_active.end(),
        static_cast<unsigned char>(1)));
    topology.spanning_backbone_used = true;
    {
        // Preserve a strongest path through every covisibility component
        // while adding only O(num_cameras) pair factors.
        std::vector<double> best_weight(static_cast<size_t>(n), 0.0);
        std::vector<int> parent(static_cast<size_t>(n), -1);
        std::vector<unsigned char> in_forest(static_cast<size_t>(n), 0);
        int inserted = 0;
        while (inserted < n) {
            int vertex = -1;
            double weight = 0.0;
            for (int camera = 0; camera < n; ++camera) {
                if (!in_forest[static_cast<size_t>(camera)] &&
                    best_weight[static_cast<size_t>(camera)] > weight) {
                    vertex = camera;
                    weight = best_weight[static_cast<size_t>(camera)];
                }
            }
            if (vertex < 0) {
                for (int camera = 0; camera < n; ++camera) {
                    if (!in_forest[static_cast<size_t>(camera)]) {
                        vertex = camera;
                        break;
                    }
                }
            }
            if (vertex < 0) {
                break;
            }
            in_forest[static_cast<size_t>(vertex)] = 1;
            ++inserted;
            const int predecessor = parent[static_cast<size_t>(vertex)];
            if (predecessor >= 0) {
                selected_pair[
                    static_cast<size_t>(vertex) * static_cast<size_t>(n) +
                    static_cast<size_t>(predecessor)] = 1;
                selected_pair[
                    static_cast<size_t>(predecessor) * static_cast<size_t>(n) +
                    static_cast<size_t>(vertex)] = 1;
            }
            for (int neighbor = 0; neighbor < n; ++neighbor) {
                if (in_forest[static_cast<size_t>(neighbor)]) {
                    continue;
                }
                const double candidate = pair_weight(vertex, neighbor);
                if (candidate > best_weight[static_cast<size_t>(neighbor)]) {
                    best_weight[static_cast<size_t>(neighbor)] = candidate;
                    parent[static_cast<size_t>(neighbor)] = vertex;
                }
            }
        }
    }

    auto pair_is_selected = [&](int row, int col) {
        if (row == col) {
            return true;
        }
        const size_t dense_index =
            static_cast<size_t>(row) * static_cast<size_t>(n) +
            static_cast<size_t>(col);
        return selected_pair[dense_index] != 0;
    };

    std::vector<int> component_parent(static_cast<size_t>(n));
    std::vector<int> component_size(static_cast<size_t>(n), 1);
    std::vector<int> selected_degree(static_cast<size_t>(n), 0);
    std::iota(component_parent.begin(), component_parent.end(), 0);
    auto find_component = [&](int vertex) {
        int root = vertex;
        while (component_parent[static_cast<size_t>(root)] != root) {
            root = component_parent[static_cast<size_t>(root)];
        }
        while (component_parent[static_cast<size_t>(vertex)] != vertex) {
            const int next = component_parent[static_cast<size_t>(vertex)];
            component_parent[static_cast<size_t>(vertex)] = root;
            vertex = next;
        }
        return root;
    };
    for (int row = 0; row < n; ++row) {
        for (int col = row + 1; col < n; ++col) {
            if (!pair_is_selected(row, col)) {
                continue;
            }
            ++selected_degree[static_cast<size_t>(row)];
            ++selected_degree[static_cast<size_t>(col)];
            int root_row = find_component(row);
            int root_col = find_component(col);
            if (root_row == root_col) {
                continue;
            }
            if (component_size[static_cast<size_t>(root_row)] <
                component_size[static_cast<size_t>(root_col)]) {
                std::swap(root_row, root_col);
            }
            component_parent[static_cast<size_t>(root_col)] = root_row;
            component_size[static_cast<size_t>(root_row)] +=
                component_size[static_cast<size_t>(root_col)];
        }
    }
    std::vector<unsigned char> counted_component(static_cast<size_t>(n), 0);
    for (int camera = 0; camera < n; ++camera) {
        if (selected_degree[static_cast<size_t>(camera)] == 0) {
            continue;
        }
        ++topology.selected_active_cameras;
        const int root = find_component(camera);
        if (counted_component[static_cast<size_t>(root)] == 0) {
            counted_component[static_cast<size_t>(root)] = 1;
            ++topology.selected_components;
            topology.selected_largest_component = std::max(
                topology.selected_largest_component,
                component_size[static_cast<size_t>(root)]);
        }
    }

    pattern.row_ptr.resize(static_cast<size_t>(n) + 1, 0);
    pattern.diag_slot.resize(static_cast<size_t>(n), -1);
    for (int row = 0; row < n; ++row) {
        int count = 0;
        for (int col = 0; col < n; ++col) {
            count += pair_is_selected(row, col);
        }
        pattern.row_ptr[static_cast<size_t>(row) + 1] =
            pattern.row_ptr[static_cast<size_t>(row)] + count;
    }
    pattern.col_idx.resize(
        static_cast<size_t>(pattern.row_ptr.back()));
    std::vector<int> dense_slot(dense_size, -1);
    pattern.slot_of.reserve(
        static_cast<size_t>(pattern.row_ptr.back()) * 2);
    for (int row = 0; row < n; ++row) {
        int slot = pattern.row_ptr[static_cast<size_t>(row)];
        const size_t base =
            static_cast<size_t>(row) * static_cast<size_t>(n);
        for (int col = 0; col < n; ++col) {
            if (!pair_is_selected(row, col)) {
                continue;
            }
            pattern.col_idx[static_cast<size_t>(slot)] = col;
            dense_slot[base + static_cast<size_t>(col)] = slot;
            pattern.slot_of.emplace(blockKey(row, col), slot);
            if (row == col) {
                pattern.diag_slot[static_cast<size_t>(row)] = slot;
            }
            ++slot;
        }
    }

    pattern.slot_pair_row_ptr.assign(
        pattern.col_idx.size() + 1, 0);
    std::vector<std::vector<int>> slot_count_by_thread(
        static_cast<size_t>(topology_threads),
        std::vector<int>(pattern.col_idx.size(), 0));
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(static)
#endif
    for (int p_index = 0;
         p_index < static_cast<int>(point_count);
         ++p_index) {
        const size_t p = static_cast<size_t>(p_index);
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<int>& local_slot_count =
            slot_count_by_thread[static_cast<size_t>(tid)];
        const int edge_begin = plan.block_offset[p];
        const int edge_end = plan.block_offset[p + 1];
        for (int a = edge_begin; a < edge_end; ++a) {
            for (int b = a + 1; b < edge_end; ++b) {
                const int ca =
                    plan.free_camera[static_cast<size_t>(a)];
                const int cb =
                    plan.free_camera[static_cast<size_t>(b)];
                if (ca == cb) {
                    continue;
                }
                const int lo = std::min(ca, cb);
                const int hi = std::max(ca, cb);
                const int slot = dense_slot[
                    static_cast<size_t>(lo) *
                        static_cast<size_t>(n) +
                    static_cast<size_t>(hi)];
                if (slot < 0) {
                    continue;
                }
                ++local_slot_count[static_cast<size_t>(slot)];
            }
        }
    }
    for (size_t slot = 0;
         slot < pattern.col_idx.size();
         ++slot) {
        int count = 0;
        for (const std::vector<int>& local :
             slot_count_by_thread) {
            count += local[slot];
        }
        pattern.slot_pair_row_ptr[slot + 1] = count;
    }
    for (size_t i = 1;
         i < pattern.slot_pair_row_ptr.size();
         ++i) {
        pattern.slot_pair_row_ptr[i] +=
            pattern.slot_pair_row_ptr[i - 1];
    }

    const int ref_count = pattern.slot_pair_row_ptr.back();
    pattern.slot_pair_edge_a_global.resize(
        static_cast<size_t>(ref_count));
    pattern.slot_pair_edge_b_global.resize(
        static_cast<size_t>(ref_count));
    pattern.slot_pair_transpose.assign(
        static_cast<size_t>(ref_count), 0);
    std::vector<std::vector<int>> cursor_by_thread(
        static_cast<size_t>(topology_threads),
        std::vector<int>(pattern.col_idx.size(), 0));
    for (size_t slot = 0;
         slot < pattern.col_idx.size();
         ++slot) {
        int cursor = pattern.slot_pair_row_ptr[slot];
        for (int tid = 0; tid < topology_threads; ++tid) {
            cursor_by_thread[static_cast<size_t>(tid)][slot] =
                cursor;
            cursor += slot_count_by_thread[
                static_cast<size_t>(tid)][slot];
        }
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(static)
#endif
    for (int p_index = 0;
         p_index < static_cast<int>(point_count);
         ++p_index) {
        const size_t p = static_cast<size_t>(p_index);
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        std::vector<int>& local_cursor =
            cursor_by_thread[static_cast<size_t>(tid)];
        const int edge_base = plan.block_offset[p];
        const int edge_end = plan.block_offset[p + 1];
        for (int a = edge_base; a < edge_end; ++a) {
            for (int b = a + 1; b < edge_end; ++b) {
                const int ca =
                    plan.free_camera[static_cast<size_t>(a)];
                const int cb =
                    plan.free_camera[static_cast<size_t>(b)];
                if (ca == cb) {
                    continue;
                }
                const bool forward = ca < cb;
                const int lo = forward ? ca : cb;
                const int hi = forward ? cb : ca;
                const int slot = dense_slot[
                    static_cast<size_t>(lo) *
                        static_cast<size_t>(n) +
                    static_cast<size_t>(hi)];
                if (slot < 0) {
                    continue;
                }
                const int dst =
                    local_cursor[static_cast<size_t>(slot)]++;
                pattern.slot_pair_edge_a_global[
                    static_cast<size_t>(dst)] =
                    forward ? a : b;
                pattern.slot_pair_edge_b_global[
                    static_cast<size_t>(dst)] =
                    forward ? b : a;
            }
        }
    }
    for (int ref = 0; ref < ref_count; ++ref) {
        plan.edge_used[static_cast<size_t>(
            pattern.slot_pair_edge_a_global[
                static_cast<size_t>(ref)])] = 1;
        plan.edge_used[static_cast<size_t>(
            pattern.slot_pair_edge_b_global[
                static_cast<size_t>(ref)])] = 1;
    }
    compactRootPairEdges(pattern, plan);
    pattern.build_sec = elapsed(t0, Clock::now());
    return topology;
}

void resetRootFastHGBPWorkspaceValues(
    FastHGBPSystem& sys,
    double damping
) {
    SchurGBPWorkspace& ws = sys.ws;
    ws.belief_lam_inv_valid = false;
    for (int i = 0; i < ws.n; ++i) {
        Mat9& unary = ws.unary_lam[static_cast<size_t>(i)];
        unary.setZero();
        unary.diagonal().array() = damping;
    }

    zeroPlainEigenStorage(ws.msg_lam);
    zeroPlainEigenStorage(ws.msg_eta);
    if (sys.rhs_blocks.size() !=
        static_cast<size_t>(sys.free_cameras)) {
        sys.rhs_blocks.resize(
            static_cast<size_t>(sys.free_cameras));
    }
}
