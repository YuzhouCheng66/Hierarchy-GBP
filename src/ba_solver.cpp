#include "ba_internal.h"

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

#include "rootba/bal/bal_bundle_adjustment_helper.hpp"

namespace {

struct HGBPConfig {
    int full_lambda_outers = 1;
    int min_pair_observations = 1;
    int pair_sample_cap = 0;
    bool pair_sample_rescale = true;
    bool unreduced_unary = false;
    bool no_pose_scaling = false;
    double initial_lambda = 1e-4;
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
        if (arg == "--initial-lambda" && i + 1 < argc) {
            parseDouble(argv[++i], config.initial_lambda);
        } else if (arg == "--exact-refinement-policy" &&
                   i + 1 < argc) {
            const std::string policy = argv[++i];
            if (policy != "cost-first") {
                throw std::runtime_error(
                    "--exact-refinement-policy must be cost-first");
            }
        } else if (arg == "--full-lambda-outers" &&
                   i + 1 < argc) {
            parseInt(argv[++i], config.full_lambda_outers);
        } else if (arg == "--min-pair-observations" &&
                   i + 1 < argc) {
            parseInt(argv[++i], config.min_pair_observations);
        } else if (arg == "--pair-sample-cap" && i + 1 < argc) {
            parseInt(argv[++i], config.pair_sample_cap);
        } else if (arg == "--pair-sample-no-rescale") {
            config.pair_sample_rescale = false;
        } else if (arg == "--unreduced-unary") {
            config.unreduced_unary = true;
        } else if (arg == "--no-pose-scaling") {
            config.no_pose_scaling = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ba_solver --problem-file <BAL.txt> "
                   "[--out-json <path>]\n"
                << "  --outer N --mg-cycles C --pre-sweeps K "
                   "--gbp-full-sweeps K\n"
                << "  --group-size G --build-threads T "
                   "--gbp-threads T --message-damping X\n"
                << "  --initial-lambda X --full-lambda-outers N\n"
                << "  --min-pair-observations N --pair-sample-cap N\n"
                << "  [--pair-sample-no-rescale] [--unreduced-unary] "
                   "[--no-pose-scaling]\n";
            std::exit(0);
        } else {
            filtered_storage.push_back(arg);
        }
    }
    config.full_lambda_outers =
        std::max(0, config.full_lambda_outers);
    config.min_pair_observations =
        std::max(1, config.min_pair_observations);
    config.pair_sample_cap =
        std::max(0, config.pair_sample_cap);
    config.initial_lambda =
        std::max(1e-16, config.initial_lambda);
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
};

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
    int min_pair_observations,
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
    std::vector<std::vector<int>> pair_count_by_thread(
        static_cast<size_t>(topology_threads),
        std::vector<int>(dense_size, 0));
    int invalid_camera = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(static) reduction(|:invalid_camera)
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
        std::vector<int>& local_pair_count =
            pair_count_by_thread[static_cast<size_t>(tid)];
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
        const int edge_end = plan.block_offset[p + 1];
        for (int a = edge_base; a < edge_end; ++a) {
            for (int b = a + 1; b < edge_end; ++b) {
                const size_t ca = static_cast<size_t>(
                    plan.free_camera[static_cast<size_t>(a)]);
                const size_t cb = static_cast<size_t>(
                    plan.free_camera[static_cast<size_t>(b)]);
                if (ca == cb) {
                    continue;
                }
                ++local_pair_count[
                    ca * static_cast<size_t>(n) + cb];
                ++local_pair_count[
                    cb * static_cast<size_t>(n) + ca];
            }
        }
    }
    if (invalid_camera != 0) {
        throw std::runtime_error(
            "RootBA camera index is outside H-GBP topology");
    }
#if defined(_OPENMP)
#pragma omp parallel for num_threads(topology_threads) schedule(static)
#endif
    for (int dense_index = 0;
         dense_index < static_cast<int>(dense_size);
         ++dense_index) {
        int count = 0;
        for (const std::vector<int>& local :
             pair_count_by_thread) {
            count += local[static_cast<size_t>(dense_index)];
        }
        pair_count[static_cast<size_t>(dense_index)] = count;
    }
    for (int c = 0; c < n; ++c) {
        pair_count[static_cast<size_t>(c) *
                       static_cast<size_t>(n) +
                   static_cast<size_t>(c)] = 1;
    }
    pair_count_by_thread.clear();

    pattern.row_ptr.resize(static_cast<size_t>(n) + 1, 0);
    pattern.diag_slot.resize(static_cast<size_t>(n), -1);
    for (int row = 0; row < n; ++row) {
        int count = 0;
        const size_t base =
            static_cast<size_t>(row) * static_cast<size_t>(n);
        for (int col = 0; col < n; ++col) {
            count +=
                row == col ||
                pair_count[base + static_cast<size_t>(col)] >=
                    min_pair_observations;
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
            if (row != col &&
                pair_count[base + static_cast<size_t>(col)] <
                    min_pair_observations) {
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
    double damping,
    bool reset_messages
) {
    SchurGBPWorkspace& ws = sys.ws;
    ws.belief_lam_inv_valid = false;
    for (int i = 0; i < ws.n; ++i) {
        Mat9& unary = ws.unary_lam[static_cast<size_t>(i)];
        unary.setZero();
        unary.diagonal().array() = damping;
    }

    // Eta-only first sweeps overwrite eta and do not read lambda messages.
    if (reset_messages) {
        zeroPlainEigenStorage(ws.msg_lam);
        zeroPlainEigenStorage(ws.msg_eta);
    }
    if (sys.rhs_blocks.size() !=
        static_cast<size_t>(sys.free_cameras)) {
        sys.rhs_blocks.resize(
            static_cast<size_t>(sys.free_cameras));
    }
}

struct RootBuildBreakdown {
    double setup_sec = 0.0;
    double point_sec = 0.0;
    double rhs_reduce_sec = 0.0;
    double unary_reduce_sec = 0.0;
    double pair_sec = 0.0;
};

using Mat29List =
    std::vector<Mat29, Eigen::aligned_allocator<Mat29>>;
using Mat23List =
    std::vector<Mat23, Eigen::aligned_allocator<Mat23>>;
using Vec2List =
    std::vector<Vec2, Eigen::aligned_allocator<Vec2>>;

struct PackedRootLandmarks {
    std::vector<int> point_offset;
    std::vector<int> camera_id;
    Vec2List measurement;
    Mat3List camera_rotation;
    Vec3List camera_translation;
    Vec3List camera_intrinsics;
    Mat29List Jc;
    Mat23List Jl;
    Vec2List residual;
    Mat3List Hll;
    Mat3List Hll_inv;
    Vec3List bl;
    Vec3List Jl_scale;
    std::vector<Eigen::VectorXd> pose_diag2_by_thread;
    std::vector<Eigen::VectorXd> thread_rhs;
    std::vector<Eigen::VectorXd> thread_matvec;
    Eigen::VectorXd schur_scaled_x;
    Eigen::VectorXd pcg_rhs;
    Eigen::VectorXd pcg_product;
    Eigen::VectorXd pcg_residual;
    Eigen::VectorXd pcg_preconditioned;
    Eigen::VectorXd pcg_direction;
    Mat9List pcg_fallback_inverse;
    std::vector<int> numerical_failures;
    bool camera_cache_valid = false;
};

PackedRootLandmarks makePackedRootLandmarks(
    const RootProblem& problem,
    const RootHGBPBuildPlan& build_plan
) {
    PackedRootLandmarks packed;
    packed.point_offset = build_plan.block_offset;
    packed.camera_id = build_plan.free_camera;
    const size_t observation_count = packed.camera_id.size();
    const size_t point_count = problem.landmarks().size();
    const size_t camera_count = problem.cameras().size();
    packed.measurement.resize(observation_count);
    packed.camera_rotation.resize(camera_count);
    packed.camera_translation.resize(camera_count);
    packed.camera_intrinsics.resize(camera_count);
    packed.Jc.resize(observation_count);
    packed.Jl.resize(observation_count);
    packed.residual.resize(observation_count);
    packed.Hll.resize(point_count);
    packed.Hll_inv.resize(point_count);
    packed.bl.resize(point_count);
    packed.Jl_scale.resize(point_count);

    int invalid_layout = 0;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) reduction(|:invalid_layout)
#endif
    for (int point_index = 0;
         point_index < static_cast<int>(point_count);
         ++point_index) {
        const size_t p = static_cast<size_t>(point_index);
        const auto& observations = problem.landmarks()[p].obs;
        const int begin = packed.point_offset[p];
        const int end = packed.point_offset[p + 1];
        if (end - begin != static_cast<int>(observations.size())) {
            invalid_layout = 1;
            continue;
        }
        int edge = begin;
        for (const auto& [camera, observation] : observations) {
            if (packed.camera_id[static_cast<size_t>(edge)] !=
                static_cast<int>(camera)) {
                invalid_layout = 1;
            }
            packed.measurement[static_cast<size_t>(edge)] =
                observation.pos;
            ++edge;
        }
    }
    if (invalid_layout != 0) {
        throw std::runtime_error(
            "packed landmark observation order does not match topology");
    }
    return packed;
}

void refreshPackedRootCameraCache(
    const RootProblem& problem,
    PackedRootLandmarks& packed
) {
    if (packed.camera_cache_valid) {
        return;
    }
    const size_t camera_count = problem.cameras().size();
    if (packed.camera_rotation.size() != camera_count ||
        packed.camera_translation.size() != camera_count ||
        packed.camera_intrinsics.size() != camera_count) {
        throw std::runtime_error(
            "packed camera cache does not match the problem");
    }
    for (size_t camera_id = 0;
         camera_id < camera_count;
         ++camera_id) {
        const auto& camera = problem.cameras()[camera_id];
        packed.camera_rotation[camera_id] =
            camera.T_c_w.rotationMatrix();
        packed.camera_translation[camera_id] =
            camera.T_c_w.translation();
        packed.camera_intrinsics[camera_id] =
            camera.intrinsics.getParam();
    }
    packed.camera_cache_valid = true;
}

GBP_FORCE_INLINE void linearizePackedBalObservation(
    const Vec2& measurement,
    const Vec3& point_w,
    const Mat3& rotation,
    const Vec3& translation,
    const Vec3& intrinsics,
    Vec2& residual,
    Mat29& Jcamera,
    Mat23& Jlandmark
) {
    const double px =
        rotation(0, 0) * point_w.x() +
        rotation(0, 1) * point_w.y() +
        rotation(0, 2) * point_w.z() +
        translation.x();
    const double py =
        rotation(1, 0) * point_w.x() +
        rotation(1, 1) * point_w.y() +
        rotation(1, 2) * point_w.z() +
        translation.y();
    const double pz =
        rotation(2, 0) * point_w.x() +
        rotation(2, 1) * point_w.y() +
        rotation(2, 2) * point_w.z() +
        translation.z();

    const double f = intrinsics.x();
    const double k1 = intrinsics.y();
    const double k2 = intrinsics.z();
    const double inv_z = 1.0 / pz;
    const double mx = px * inv_z;
    const double my = py * inv_z;
    const double mx2 = mx * mx;
    const double my2 = my * my;
    const double radius2 = mx2 + my2;
    const double radius4 = radius2 * radius2;
    const double radial = 1.0 + k1 * radius2 + k2 * radius4;
    residual.x() = f * mx * radial - measurement.x();
    residual.y() = f * my * radial - measurement.y();

    const double radial_derivative =
        k1 + 2.0 * k2 * radius2;
    const double common_xy =
        2.0 * f * mx * my * radial_derivative * inv_z;
    const double common_z =
        radial + 2.0 * radial_derivative * radius2;
    const double a00 =
        f * (radial + 2.0 * mx2 * radial_derivative) * inv_z;
    const double a01 = common_xy;
    const double a02 = -f * mx * common_z * inv_z;
    const double a10 = common_xy;
    const double a11 =
        f * (radial + 2.0 * my2 * radial_derivative) * inv_z;
    const double a12 = -f * my * common_z * inv_z;

    Jcamera(0, 0) = a00;
    Jcamera(0, 1) = a01;
    Jcamera(0, 2) = a02;
    Jcamera(1, 0) = a10;
    Jcamera(1, 1) = a11;
    Jcamera(1, 2) = a12;
    Jcamera(0, 3) = -pz * a01 + py * a02;
    Jcamera(1, 3) = -pz * a11 + py * a12;
    Jcamera(0, 4) = pz * a00 - px * a02;
    Jcamera(1, 4) = pz * a10 - px * a12;
    Jcamera(0, 5) = -py * a00 + px * a01;
    Jcamera(1, 5) = -py * a10 + px * a11;
    Jcamera(0, 6) = mx * radial;
    Jcamera(1, 6) = my * radial;
    Jcamera(0, 7) = f * mx * radius2;
    Jcamera(1, 7) = f * my * radius2;
    Jcamera(0, 8) = f * mx * radius4;
    Jcamera(1, 8) = f * my * radius4;

    for (int col = 0; col < 3; ++col) {
        Jlandmark(0, col) =
            a00 * rotation(0, col) +
            a01 * rotation(1, col) +
            a02 * rotation(2, col);
        Jlandmark(1, col) =
            a10 * rotation(0, col) +
            a11 * rotation(1, col) +
            a12 * rotation(2, col);
    }
}

RootMetrics computePackedRootMetrics(
    const RootProblem& problem,
    const PackedRootLandmarks& packed,
    bool include_residual_norm
) {
    struct Accumulator {
        double cost2 = 0.0;
        double residual_norm_sum = 0.0;
        long long observation_count = 0;
    };
    const auto body = [&](const tbb::blocked_range<int>& range,
                          Accumulator local) {
        for (int point_index = range.begin();
             point_index != range.end();
             ++point_index) {
            const size_t p = static_cast<size_t>(point_index);
            const int begin = packed.point_offset[p];
            const int end = packed.point_offset[p + 1];
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera_id = packed.camera_id[e];
                const Vec3& point = problem.landmarks()[p].p_w;
                const Mat3& rotation =
                    packed.camera_rotation[
                        static_cast<size_t>(camera_id)];
                const Vec3& translation =
                    packed.camera_translation[
                        static_cast<size_t>(camera_id)];
                const Vec3& intrinsics =
                    packed.camera_intrinsics[
                        static_cast<size_t>(camera_id)];
                const double px =
                    rotation(0, 0) * point.x() +
                    rotation(0, 1) * point.y() +
                    rotation(0, 2) * point.z() +
                    translation.x();
                const double py =
                    rotation(1, 0) * point.x() +
                    rotation(1, 1) * point.y() +
                    rotation(1, 2) * point.z() +
                    translation.y();
                const double pz =
                    rotation(2, 0) * point.x() +
                    rotation(2, 1) * point.y() +
                    rotation(2, 2) * point.z() +
                    translation.z();
                const double mx = px / pz;
                const double my = py / pz;
                const double radius2 = mx * mx + my * my;
                const double radial =
                    1.0 + intrinsics.y() * radius2 +
                    intrinsics.z() * radius2 * radius2;
                const double residual_x =
                    intrinsics.x() * mx * radial -
                    packed.measurement[e].x();
                const double residual_y =
                    intrinsics.x() * my * radial -
                    packed.measurement[e].y();
                const double residual2 =
                    residual_x * residual_x +
                    residual_y * residual_y;
                local.cost2 += residual2;
                if (include_residual_norm) {
                    local.residual_norm_sum +=
                        std::sqrt(residual2);
                }
                ++local.observation_count;
            }
        }
        return local;
    };
    const auto join = [](const Accumulator& a,
                         const Accumulator& b) {
        return Accumulator{
            a.cost2 + b.cost2,
            a.residual_norm_sum + b.residual_norm_sum,
            a.observation_count + b.observation_count};
    };
    const Accumulator total = tbb::parallel_reduce(
        tbb::blocked_range<int>(
            0, problem.num_landmarks(), 64),
        Accumulator{},
        body,
        join);
    const double count =
        static_cast<double>(
            std::max<long long>(1, total.observation_count));
    return {
        total.cost2,
        std::sqrt(std::max(0.0, total.cost2) / count),
        include_residual_norm
            ? total.residual_norm_sum / count
            : 0.0
    };
}

void packedRootExactSchurMultiplyInto(
    PackedRootLandmarks& packed,
    const Eigen::VectorXd& pose_scaling,
    double damping,
    const Eigen::VectorXd& x,
    int requested_threads,
    Eigen::VectorXd& y
) {
    const int camera_count =
        static_cast<int>(packed.camera_rotation.size());
    const int point_count =
        static_cast<int>(packed.Hll_inv.size());
    const Eigen::Index dimension =
        static_cast<Eigen::Index>(9 * camera_count);
    if (x.size() != dimension ||
        pose_scaling.size() != dimension) {
        throw std::runtime_error(
            "packed exact Schur dimensions do not match");
    }
    const int threads = effectiveBuildThreads(
        requested_threads, point_count, camera_count);
    packed.thread_matvec.resize(static_cast<size_t>(threads));
    for (Eigen::VectorXd& local : packed.thread_matvec) {
        if (local.size() != dimension) {
            local.resize(dimension);
        }
        local.setZero();
    }
    if (packed.schur_scaled_x.size() != dimension) {
        packed.schur_scaled_x.resize(dimension);
    }
    packed.schur_scaled_x.array() =
        pose_scaling.array() * x.array();
    const double* GBP_RESTRICT scaled_x_data =
        packed.schur_scaled_x.data();

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads)
#endif
    {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        Eigen::VectorXd& local =
            packed.thread_matvec[static_cast<size_t>(tid)];
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 512)
#endif
        for (int point_index = 0;
             point_index < point_count;
             ++point_index) {
            const size_t p = static_cast<size_t>(point_index);
            const int begin = packed.point_offset[p];
            const int end = packed.point_offset[p + 1];
            Vec3 point_projection = Vec3::Zero();
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera = packed.camera_id[e];
                const Eigen::Index offset =
                    static_cast<Eigen::Index>(9 * camera);
                Vec2 projected = Vec2::Zero();
                const double* GBP_RESTRICT camera_x =
                    scaled_x_data + offset;
                for (int col = 0; col < 9; ++col) {
                    projected.x() +=
                        packed.Jc[e](0, col) * camera_x[col];
                    projected.y() +=
                        packed.Jc[e](1, col) * camera_x[col];
                }
                const Mat23& jl = packed.Jl[e];
                point_projection.x() +=
                    jl(0, 0) * projected.x() +
                    jl(1, 0) * projected.y();
                point_projection.y() +=
                    jl(0, 1) * projected.x() +
                    jl(1, 1) * projected.y();
                point_projection.z() +=
                    jl(0, 2) * projected.x() +
                    jl(1, 2) * projected.y();
            }
            const Vec3 eliminated_projection =
                packed.Hll_inv[p] * point_projection;
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera = packed.camera_id[e];
                const Eigen::Index offset =
                    static_cast<Eigen::Index>(9 * camera);
                Vec2 projected = Vec2::Zero();
                const double* GBP_RESTRICT camera_x =
                    scaled_x_data + offset;
                for (int col = 0; col < 9; ++col) {
                    projected.x() +=
                        packed.Jc[e](0, col) * camera_x[col];
                    projected.y() +=
                        packed.Jc[e](1, col) * camera_x[col];
                }
                const Mat23& jl = packed.Jl[e];
                const double reduced_x =
                    projected.x() -
                    (jl(0, 0) * eliminated_projection.x() +
                     jl(0, 1) * eliminated_projection.y() +
                     jl(0, 2) * eliminated_projection.z());
                const double reduced_y =
                    projected.y() -
                    (jl(1, 0) * eliminated_projection.x() +
                     jl(1, 1) * eliminated_projection.y() +
                     jl(1, 2) * eliminated_projection.z());
                for (int col = 0; col < 9; ++col) {
                    local[offset + col] +=
                        packed.Jc[e](0, col) * reduced_x +
                        packed.Jc[e](1, col) * reduced_y;
                }
            }
        }
    }

    y.setZero(dimension);
    for (const Eigen::VectorXd& local : packed.thread_matvec) {
        y.noalias() += local;
    }
    y.array() *= pose_scaling.array();
    y.noalias() += damping * x;
}

void applyPackedRootBlockPreconditioner(
    const Mat9List& inverse_blocks,
    const Eigen::VectorXd& rhs,
    int requested_threads,
    Eigen::VectorXd& solution
) {
    const int block_count =
        static_cast<int>(inverse_blocks.size());
    solution.resize(rhs.size());
    const int threads = effectiveGBPThreads(
        requested_threads, block_count, block_count);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
#endif
    for (int block = 0; block < block_count; ++block) {
        const double* source =
            rhs.data() + 9 * static_cast<size_t>(block);
        double* target =
            solution.data() + 9 * static_cast<size_t>(block);
        const double* inverse =
            inverse_blocks[static_cast<size_t>(block)].data();
        for (int row = 0; row < 9; ++row) {
            double value = 0.0;
            for (int col = 0; col < 9; ++col) {
                value += inverse[row + 9 * col] * source[col];
            }
            target[row] = value;
        }
    }
}

int refinePackedRootSchurPCG(
    FastHGBPSystem& sys,
    PackedRootLandmarks& packed,
    const Eigen::VectorXd& pose_scaling,
    double damping,
    int requested_steps,
    int requested_threads,
    Eigen::VectorXd& x,
    double& relative_residual
) {
    relative_residual =
        std::numeric_limits<double>::quiet_NaN();
    if (requested_steps <= 0) {
        return 0;
    }

    Eigen::VectorXd& rhs = packed.pcg_rhs;
    Eigen::VectorXd& product = packed.pcg_product;
    Eigen::VectorXd& residual = packed.pcg_residual;
    Eigen::VectorXd& preconditioned =
        packed.pcg_preconditioned;
    Eigen::VectorXd& direction = packed.pcg_direction;
    fastHGBPRhsVectorInto(sys, rhs);
    packedRootExactSchurMultiplyInto(
        packed,
        pose_scaling,
        damping,
        x,
        requested_threads,
        product);
    residual = rhs - product;
    const double rhs_norm = std::max(1.0, rhs.norm());

    const int block_count = sys.free_cameras;
    Mat9List& fallback_inverse_blocks =
        packed.pcg_fallback_inverse;
    const Mat9List* inverse_blocks =
        &sys.ws.belief_lam_inv;
    if (!sys.ws.belief_lam_inv_valid) {
        fallback_inverse_blocks.assign(
            static_cast<size_t>(block_count), Mat9::Zero());
        const Mat9 identity = Mat9::Identity();
        const Vec9 zero = Vec9::Zero();
        const int threads = effectiveGBPThreads(
            requested_threads, block_count, block_count);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
#endif
        for (int block = 0; block < block_count; ++block) {
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(
                sys.ws.unary_lam[static_cast<size_t>(block)],
                identity,
                zero,
                fallback_inverse_blocks[
                    static_cast<size_t>(block)],
                unused);
        }
        inverse_blocks = &fallback_inverse_blocks;
    }

    applyPackedRootBlockPreconditioner(
        *inverse_blocks,
        residual,
        requested_threads,
        preconditioned);
    direction = preconditioned;
    double residual_preconditioned =
        residual.dot(preconditioned);
    int completed_steps = 0;
    for (int step = 0; step < requested_steps; ++step) {
        if (!std::isfinite(residual_preconditioned) ||
            residual_preconditioned <= 0.0) {
            break;
        }
        packedRootExactSchurMultiplyInto(
            packed,
            pose_scaling,
            damping,
            direction,
            requested_threads,
            product);
        const double denominator = direction.dot(product);
        if (!std::isfinite(denominator) ||
            denominator <= 0.0) {
            break;
        }
        const double alpha =
            residual_preconditioned / denominator;
        x.noalias() += alpha * direction;
        residual.noalias() -= alpha * product;
        ++completed_steps;
        relative_residual = residual.norm() / rhs_norm;
        if (relative_residual <= 1e-10) {
            break;
        }
        applyPackedRootBlockPreconditioner(
            *inverse_blocks,
            residual,
            requested_threads,
            preconditioned);
        const double next_residual_preconditioned =
            residual.dot(preconditioned);
        if (!std::isfinite(next_residual_preconditioned) ||
            next_residual_preconditioned <= 0.0) {
            break;
        }
        const double beta =
            next_residual_preconditioned /
            residual_preconditioned;
        direction =
            preconditioned + beta * direction;
        residual_preconditioned =
            next_residual_preconditioned;
    }
    if (completed_steps == 0) {
        relative_residual = residual.norm() / rhs_norm;
    }
    return completed_steps;
}

bool linearizePackedRootLandmark(
    RootProblem& problem,
    PackedRootLandmarks& packed,
    size_t point_index,
    double scaling_epsilon,
    Eigen::VectorXd* local_pose_diagonal
) {
    const int begin = packed.point_offset[point_index];
    const int end = packed.point_offset[point_index + 1];
    Vec3 landmark_column_norm2 = Vec3::Zero();
    double camera_jacobian_norm2 = 0.0;
    for (int edge = begin; edge < end; ++edge) {
        const size_t e = static_cast<size_t>(edge);
        const int camera_id = packed.camera_id[e];
        Mat23& Jlandmark = packed.Jl[e];
        Vec2& residual = packed.residual[e];
        Mat29& Jcamera = packed.Jc[e];
        linearizePackedBalObservation(
            packed.measurement[e],
            problem.landmarks()[point_index].p_w,
            packed.camera_rotation[
                static_cast<size_t>(camera_id)],
            packed.camera_translation[
                static_cast<size_t>(camera_id)],
            packed.camera_intrinsics[
                static_cast<size_t>(camera_id)],
            residual,
            Jcamera,
            Jlandmark);
        if (local_pose_diagonal != nullptr) {
            const Eigen::Matrix<double, 1, 9> column_norm2 =
                Jcamera.colwise().squaredNorm();
            local_pose_diagonal->segment<9>(9 * camera_id) +=
                column_norm2.transpose();
            camera_jacobian_norm2 += column_norm2.sum();
        } else {
            camera_jacobian_norm2 += Jcamera.squaredNorm();
        }
        landmark_column_norm2 +=
            Jlandmark.colwise().squaredNorm().transpose();
    }

    Vec3& landmark_scale = packed.Jl_scale[point_index];
    landmark_scale =
        (scaling_epsilon +
         landmark_column_norm2.array().sqrt())
            .inverse();
    Mat3 Hll = Mat3::Zero();
    Vec3 bl = Vec3::Zero();
    for (int edge = begin; edge < end; ++edge) {
        const size_t e = static_cast<size_t>(edge);
        Mat23& Jlandmark = packed.Jl[e];
        Jlandmark *= landmark_scale.asDiagonal();
        Hll.noalias() += Jlandmark.transpose() * Jlandmark;
        bl.noalias() += Jlandmark.transpose() * packed.residual[e];
    }
    packed.Hll[point_index] = Hll;
    packed.bl[point_index] = bl;
    return std::isfinite(camera_jacobian_norm2) &&
        landmark_scale.array().isFinite().all() &&
        Hll.array().isFinite().all() &&
        bl.array().isFinite().all();
}

GBP_FORCE_INLINE void accumulateRootPairBlock(
    const PackedBlockSchurPattern& pattern,
    const Mat93List& edge_EBinv,
    const Mat93List& edge_E,
    int slot,
    int sample_cap,
    bool sample_rescale,
    Mat9& block
) {
    block.setZero();
    const int begin =
        pattern.slot_pair_row_ptr[static_cast<size_t>(slot)];
    const int end =
        pattern.slot_pair_row_ptr[static_cast<size_t>(slot) + 1];
    const int ref_count = end - begin;
    const int sample_count =
        sample_cap > 0
            ? std::min(ref_count, sample_cap)
            : ref_count;
    const bool sampled = sample_count < ref_count;
    const double sample_weight =
        sampled && sample_rescale
            ? static_cast<double>(ref_count) /
                  static_cast<double>(sample_count)
            : 1.0;
    double* GBP_RESTRICT out = block.data();
    for (int sample = 0; sample < sample_count; ++sample) {
        const int ref = sampled
            ? begin +
                  static_cast<int>(
                      (static_cast<long long>(2 * sample + 1) *
                       ref_count) /
                      (2LL * sample_count))
            : begin + sample;
        const int edge_a =
            pattern.slot_pair_edge_a_global[
                static_cast<size_t>(ref)];
        const int edge_b =
            pattern.slot_pair_edge_b_global[
                static_cast<size_t>(ref)];
        const double* GBP_RESTRICT a =
            edge_EBinv[static_cast<size_t>(edge_a)].data();
        const double* GBP_RESTRICT b =
            edge_E[static_cast<size_t>(edge_b)].data();
        if (pattern.slot_pair_transpose[static_cast<size_t>(ref)] == 0) {
            for (int col = 0; col < 9; ++col) {
                const double b0 = b[col];
                const double b1 = b[9 + col];
                const double b2 = b[18 + col];
                double* GBP_RESTRICT dst =
                    out + 9 * static_cast<size_t>(col);
                for (int row = 0; row < 9; ++row) {
                    const double value =
                        a[row] * b0 +
                        a[9 + row] * b1 +
                        a[18 + row] * b2;
                    dst[row] -= sample_weight * value;
                }
            }
        } else {
            for (int col = 0; col < 9; ++col) {
                const double a0 = a[col];
                const double a1 = a[9 + col];
                const double a2 = a[18 + col];
                double* GBP_RESTRICT dst =
                    out + 9 * static_cast<size_t>(col);
                for (int row = 0; row < 9; ++row) {
                    const double value =
                        b[row] * a0 +
                        b[9 + row] * a1 +
                        b[18 + row] * a2;
                    dst[row] -= sample_weight * value;
                }
            }
        }
    }
}

GBP_FORCE_INLINE void accumulateRootUnaryBlock(
    Mat9& unary,
    const Mat29& Jc,
    const Mat93* E,
    const Mat93* EBinv,
    double reduction_weight
) {
    double* GBP_RESTRICT dst = unary.data();
    const double* GBP_RESTRICT jc = Jc.data();
    const double* GBP_RESTRICT e =
        E != nullptr ? E->data() : nullptr;
    const double* GBP_RESTRICT ebinv =
        EBinv != nullptr ? EBinv->data() : nullptr;
    for (int col = 0; col < 9; ++col) {
        const double jc0 = jc[2 * col];
        const double jc1 = jc[2 * col + 1];
        for (int row = 0; row <= col; ++row) {
            double value =
                jc[2 * row] * jc0 +
                jc[2 * row + 1] * jc1;
            if (e != nullptr) {
                value -= reduction_weight *
                    (ebinv[row] * e[col] +
                     ebinv[9 + row] * e[9 + col] +
                     ebinv[18 + row] * e[18 + col]);
            }
            dst[row + 9 * col] += value;
            if (row != col) {
                dst[col + 9 * row] += value;
            }
        }
    }
}

GBP_FORCE_INLINE Mat3 invertDampedSymmetric3x3(
    const Mat3& matrix,
    double damping
) noexcept {
    const double a00 = matrix(0, 0) + damping;
    const double a10 = matrix(1, 0);
    const double a20 = matrix(2, 0);
    const double a11 = matrix(1, 1) + damping;
    const double a21 = matrix(2, 1);
    const double a22 = matrix(2, 2) + damping;
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double determinant =
        a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (!(determinant > 0.0) || !std::isfinite(determinant)) {
        Mat3 fallback = matrix;
        fallback.diagonal().array() += damping;
        return fallback.inverse().eval();
    }
    const double inverse_determinant = 1.0 / determinant;
    Mat3 inverse;
    inverse(0, 0) = cof00 * inverse_determinant;
    inverse(1, 0) = cof10 * inverse_determinant;
    inverse(2, 0) = cof20 * inverse_determinant;
    inverse(0, 1) = inverse(1, 0);
    inverse(1, 1) = cof11 * inverse_determinant;
    inverse(2, 1) = cof21 * inverse_determinant;
    inverse(0, 2) = inverse(2, 0);
    inverse(1, 2) = inverse(2, 1);
    inverse(2, 2) = cof22 * inverse_determinant;
    return inverse;
}

void buildPackedHGBPSystemInto(
    FastHGBPSystem& sys,
    const PackedBlockSchurPattern& pattern,
    const RootHGBPBuildPlan& build_plan,
    RootProblem& problem,
    PackedRootLandmarks& landmarks,
    Eigen::VectorXd& root_b,
    double damping,
    int requested_threads,
    int pair_sample_cap,
    bool pair_sample_rescale,
    bool unreduced_unary,
    bool reset_messages,
    bool fused_linearize_build,
    bool scale_pose_columns,
    double pose_scaling_epsilon,
    Eigen::VectorXd& pose_scaling,
    RootBuildBreakdown& breakdown
) {
    const auto t0 = Clock::now();
    initializeFastHGBPSystemStructure(sys, pattern);
    resetRootFastHGBPWorkspaceValues(
        sys, damping, reset_messages);
    sys.linearized_cost = 0.0;

    sys.edge_E.resize(
        static_cast<size_t>(build_plan.retained_edge_count));
    sys.edge_EBinv.resize(
        static_cast<size_t>(build_plan.retained_edge_count));

    const size_t point_count = landmarks.Hll.size();
    if (build_plan.block_offset.size() != point_count + 1) {
        throw std::runtime_error(
            "RootBA build plan does not match landmark blocks");
    }

    const int parallel_threads = effectiveBuildThreads(
        requested_threads,
        static_cast<int>(sys.ws.edges.size()) * 2 + sys.free_cameras,
        sys.free_cameras);
    sys.build_threads_actual = parallel_threads;
    sys.build_accum.resize(static_cast<size_t>(parallel_threads));
    for (FastHGBPBuildThreadAccum& accum : sys.build_accum) {
        if (accum.diag_blocks.size() !=
            static_cast<size_t>(sys.free_cameras)) {
            accum.diag_blocks.resize(
                static_cast<size_t>(sys.free_cameras));
        }
    }
    landmarks.thread_rhs.resize(
        static_cast<size_t>(parallel_threads));
    const Eigen::Index dimension =
        static_cast<Eigen::Index>(9 * problem.num_cameras());
    for (Eigen::VectorXd& rhs : landmarks.thread_rhs) {
        if (rhs.size() != dimension) {
            rhs.resize(dimension);
        }
    }
    if (fused_linearize_build) {
        const Eigen::Index pose_dimension =
            static_cast<Eigen::Index>(
                9 * problem.num_cameras());
        landmarks.pose_diag2_by_thread.resize(
            scale_pose_columns
                ? static_cast<size_t>(parallel_threads)
                : 0);
        landmarks.numerical_failures.assign(
            static_cast<size_t>(parallel_threads), 0);
        for (Eigen::VectorXd& local :
             landmarks.pose_diag2_by_thread) {
            if (local.size() != pose_dimension) {
                local.resize(pose_dimension);
            }
        }
        refreshPackedRootCameraCache(problem, landmarks);
    }
    // These buffers are independent by worker. Clear them concurrently rather
    // than streaming all thread-local storage through the caller thread.
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static) if(parallel_threads > 1 && sys.free_cameras >= 512)
#endif
    for (int tid = 0; tid < parallel_threads; ++tid) {
        zeroPlainEigenStorage(
            sys.build_accum[static_cast<size_t>(tid)].diag_blocks);
        landmarks.thread_rhs[static_cast<size_t>(tid)].setZero();
        if (fused_linearize_build && scale_pose_columns) {
            landmarks
                .pose_diag2_by_thread[static_cast<size_t>(tid)]
                .setZero();
        }
    }
    const auto setup_t1 = Clock::now();
    breakdown.setup_sec =
        elapsed(t0, setup_t1);

    const auto point_t0 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads)
#endif
    {
        int tid = 0;
#if defined(_OPENMP)
        tid = omp_get_thread_num();
#endif
        Mat9List& local_diag =
            sys.build_accum[static_cast<size_t>(tid)].diag_blocks;
#if defined(_OPENMP)
#pragma omp for schedule(dynamic, 512)
#endif
        for (int p_index = 0;
             p_index < static_cast<int>(point_count);
             ++p_index) {
            const size_t p = static_cast<size_t>(p_index);
            if (fused_linearize_build &&
                !linearizePackedRootLandmark(
                    problem,
                    landmarks,
                    p,
                    pose_scaling_epsilon,
                    scale_pose_columns
                        ? &landmarks.pose_diag2_by_thread[
                              static_cast<size_t>(tid)]
                        : nullptr)) {
                ++landmarks.numerical_failures[
                    static_cast<size_t>(tid)];
                continue;
            }

            const Mat3 Binv = invertDampedSymmetric3x3(
                landmarks.Hll[p], damping);
            landmarks.Hll_inv[p] = Binv;
            const Vec3 Hll_inv_bl =
                Binv * landmarks.bl[p];
            const int plan_begin = build_plan.block_offset[p];
            const int plan_end = build_plan.block_offset[p + 1];
            for (int plan_pos = plan_begin;
                 plan_pos < plan_end;
                 ++plan_pos) {
                const size_t edge =
                    static_cast<size_t>(plan_pos);
                const int fc = build_plan.free_camera[edge];
                if (fc < 0) {
                    continue;
                }
                const int global_edge =
                    build_plan.global_edge[edge];
                const Mat29& Jc = landmarks.Jc[edge];
                const Mat23& Jp = landmarks.Jl[edge];
                const Vec2 reduced_residual =
                    landmarks.residual[edge] -
                    Jp * Hll_inv_bl;
                double* GBP_RESTRICT rhs =
                    landmarks
                        .thread_rhs[static_cast<size_t>(tid)]
                        .data() +
                    9 * static_cast<size_t>(fc);
                for (int k = 0; k < 9; ++k) {
                    rhs[k] +=
                        Jc(0, k) * reduced_residual.x() +
                        Jc(1, k) * reduced_residual.y();
                }
                const int retained_edge =
                    build_plan.retained_edge_slot[
                        static_cast<size_t>(global_edge)];
                const bool pair_uses_edge = retained_edge >= 0;
                const bool reduce_unary = !unreduced_unary;
                const bool need_cross =
                    pair_uses_edge || reduce_unary;
                if (need_cross) {
                    const Mat93 E = Jc.transpose() * Jp;
                    const Mat93 EBinv = E * Binv;
                    if (pair_uses_edge) {
                        sys.edge_E[
                            static_cast<size_t>(retained_edge)] =
                            E;
                        sys.edge_EBinv[
                            static_cast<size_t>(retained_edge)] =
                            EBinv;
                    }
                    accumulateRootUnaryBlock(
                        local_diag[static_cast<size_t>(fc)],
                        Jc,
                        reduce_unary ? &E : nullptr,
                        reduce_unary ? &EBinv : nullptr,
                        1.0);
                } else {
                    accumulateRootUnaryBlock(
                        local_diag[static_cast<size_t>(fc)],
                        Jc,
                        nullptr,
                        nullptr,
                        1.0);
                }
            }
        }
    }
    const auto point_t1 = Clock::now();
    breakdown.point_sec =
        elapsed(point_t0, point_t1);

    if (fused_linearize_build) {
        if (scale_pose_columns) {
            Eigen::VectorXd pose_diagonal =
                Eigen::VectorXd::Zero(
                    static_cast<Eigen::Index>(
                        9 * problem.num_cameras()));
            for (const Eigen::VectorXd& local :
                 landmarks.pose_diag2_by_thread) {
                pose_diagonal.noalias() += local;
            }
            pose_scaling =
                (pose_scaling_epsilon +
                 pose_diagonal.array().sqrt())
                    .inverse();
        } else {
            pose_scaling.setOnes(
                static_cast<Eigen::Index>(
                    9 * problem.num_cameras()));
        }
        const bool valid_linearization = std::all_of(
            landmarks.numerical_failures.begin(),
            landmarks.numerical_failures.end(),
            [](int count) { return count == 0; });
        if (!valid_linearization) {
            throw std::runtime_error(
                "RootBA H-GBP found an invalid fused linearization");
        }
    }

    const auto rhs_t0 = Clock::now();
    root_b.setZero(
        static_cast<Eigen::Index>(
            9 * problem.num_cameras()));
    for (const Eigen::VectorXd& local :
         landmarks.thread_rhs) {
        root_b.noalias() += local;
    }
    const auto rhs_t1 = Clock::now();
    breakdown.rhs_reduce_sec =
        elapsed(rhs_t0, rhs_t1);

    double matrix_norm_sq = 0.0;
    double rhs_norm_sq = 0.0;
    const auto unary_t0 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static) reduction(+:matrix_norm_sq,rhs_norm_sq) if(parallel_threads > 1)
#endif
    for (int fc = 0; fc < sys.free_cameras; ++fc) {
        Mat9& unary =
            sys.ws.unary_lam[static_cast<size_t>(fc)];
        Mat9 accumulated_unary = Mat9::Zero();
        for (const FastHGBPBuildThreadAccum& accum :
             sys.build_accum) {
            accumulated_unary.noalias() +=
                accum.diag_blocks[static_cast<size_t>(fc)];
        }
        Vec9 eta = -root_b.segment<9>(9 * fc);
        const auto scale =
            pose_scaling.segment<9>(9 * fc);
        for (int col = 0; col < 9; ++col) {
            for (int row = 0; row < 9; ++row) {
                unary(row, col) +=
                    accumulated_unary(row, col) *
                    scale[row] * scale[col];
            }
            eta[col] *= scale[col];
        }
        sys.rhs_blocks[static_cast<size_t>(fc)] = eta;
        sys.ws.unary_eta[static_cast<size_t>(fc)] = eta;
        sys.ws.belief_lam[static_cast<size_t>(fc)] = unary;
        sys.ws.belief_eta[static_cast<size_t>(fc)] = eta;
        sys.ws.next_belief_lam[static_cast<size_t>(fc)] =
            unary;
        sys.ws.next_belief_eta[static_cast<size_t>(fc)] =
            eta;
        matrix_norm_sq += unary.squaredNorm();
        rhs_norm_sq += eta.squaredNorm();
    }
    const auto unary_t1 = Clock::now();
    breakdown.unary_reduce_sec =
        elapsed(unary_t0, unary_t1);

    const auto pair_t0 = Clock::now();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static) reduction(+:matrix_norm_sq) if(parallel_threads > 1)
#endif
    for (int eidx = 0;
         eidx < static_cast<int>(sys.ws.edges.size());
         ++eidx) {
        SchurGBPEdge& edge =
            sys.ws.edges[static_cast<size_t>(eidx)];
        accumulateRootPairBlock(
            pattern,
            sys.edge_EBinv,
            sys.edge_E,
            edge.slot_ij,
            pair_sample_cap,
            pair_sample_rescale,
            edge.aij);
        const auto scale_i =
            pose_scaling.segment<9>(9 * edge.i);
        const auto scale_j =
            pose_scaling.segment<9>(9 * edge.j);
        for (int col = 0; col < 9; ++col) {
            for (int row = 0; row < 9; ++row) {
                edge.aij(row, col) *=
                    scale_i[row] * scale_j[col];
            }
        }
        edge.aji = edge.aij.transpose();
        const int pos_i =
            sys.edge_row_pos_i[static_cast<size_t>(eidx)];
        const int pos_j =
            sys.edge_row_pos_j[static_cast<size_t>(eidx)];
        sys.row_blocks[static_cast<size_t>(pos_i)] =
            edge.aij;
        sys.row_blocks[static_cast<size_t>(pos_j)] =
            edge.aji;
        matrix_norm_sq +=
            edge.aij.squaredNorm() +
            edge.aji.squaredNorm();
    }
    const auto pair_t1 = Clock::now();
    breakdown.pair_sec =
        elapsed(pair_t0, pair_t1);

    sys.matrix_fingerprint = std::sqrt(matrix_norm_sq);
    sys.rhs_fingerprint = std::sqrt(rhs_norm_sq);
    sys.build_sec = elapsed(t0, Clock::now());
}

struct RootHGBPResult {
    double total_sec = 0.0;
    double preprocessor_sec = 0.0;
    double preprocessor_linearization_sec = 0.0;
    double preprocessor_topology_sec = 0.0;
    double final_cost = 0.0;
    double final_rmse_px = 0.0;
    double final_are_px = 0.0;
    int retained_camera_pairs = 0;
    long long retained_pair_references = 0;
    int retained_observations = 0;
    double grouping_cut_ratio = 0.0;
    std::vector<double> costs;
    std::vector<double> outer_sec;
    std::vector<double> linearize_sec;
    std::vector<double> message_build_sec;
    std::vector<double> build_setup_sec;
    std::vector<double> build_point_sec;
    std::vector<double> build_rhs_reduce_sec;
    std::vector<double> build_unary_reduce_sec;
    std::vector<double> build_pair_sec;
    std::vector<double> solve_sec;
    std::vector<double> exact_refinement_sec;
    std::vector<double> backup_sec;
    std::vector<double> back_substitute_sec;
    std::vector<double> metric_sec;
    std::vector<double> gbp_full_sec;
    std::vector<double> gbp_fixed_map_sec;
    std::vector<double> gbp_eta_sec;
    std::vector<double> gbp_mean_sec;
    std::vector<double> gbp_coarse_residual_sec;
    std::vector<double> gbp_coarse_solve_sec;
    std::vector<double> accepted_alpha;
    std::vector<double> lambda;
    std::vector<double> step_quality;
    std::vector<double> linear_residual;
    std::vector<int> groups;
    std::vector<int> full_sweeps;
    std::vector<int> eta_sweeps;
    std::vector<int> exact_refinement_steps;
};

void schurGBPFirstZeroMessageSweep(
    SchurGBPWorkspace& ws,
    double damping,
    int threads
) {
    const double omega =
        std::min(1.0, std::max(0.0, damping));
    const Mat9 identity = Mat9::Identity();
    const Vec9 zero = Vec9::Zero();
    const int edge_count =
        static_cast<int>(ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(
        threads, edge_count, ws.n);

#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads) if(parallel_threads > 1)
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < ws.n; ++i) {
            Mat9 inverse = Mat9::Zero();
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(
                ws.unary_lam[static_cast<size_t>(i)],
                identity,
                zero,
                inverse,
                unused);
            ws.belief_lam_inv[static_cast<size_t>(i)] =
                inverse;
            mat9VecRawInto(
                inverse,
                ws.unary_eta[static_cast<size_t>(i)],
                ws.next_belief_eta[static_cast<size_t>(i)]);
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int eidx = 0; eidx < edge_count; ++eidx) {
            const SchurGBPEdge& edge =
                ws.edges[static_cast<size_t>(eidx)];
            {
                Mat9 solved_cross = Mat9::Zero();
                solved_cross.noalias() =
                    ws.belief_lam_inv[
                        static_cast<size_t>(edge.j)] *
                    edge.aji;
                Mat9 lam = Mat9::Zero();
                lam.noalias() = -(edge.aij * solved_cross);
                lam = 0.5 * (lam + lam.transpose());
                Vec9 eta = Vec9::Zero();
                mat9VecRawInto(
                    edge.aij,
                    ws.next_belief_eta[
                        static_cast<size_t>(edge.j)],
                    eta);
                ws.next_msg_lam[
                    static_cast<size_t>(edge.msg_to_i)] =
                    omega * lam;
                ws.next_msg_eta[
                    static_cast<size_t>(edge.msg_to_i)] =
                    -omega * eta;
            }
            {
                Mat9 solved_cross = Mat9::Zero();
                solved_cross.noalias() =
                    ws.belief_lam_inv[
                        static_cast<size_t>(edge.i)] *
                    edge.aij;
                Mat9 lam = Mat9::Zero();
                lam.noalias() = -(edge.aji * solved_cross);
                lam = 0.5 * (lam + lam.transpose());
                Vec9 eta = Vec9::Zero();
                mat9VecRawInto(
                    edge.aji,
                    ws.next_belief_eta[
                        static_cast<size_t>(edge.i)],
                    eta);
                ws.next_msg_lam[
                    static_cast<size_t>(edge.msg_to_j)] =
                    omega * lam;
                ws.next_msg_eta[
                    static_cast<size_t>(edge.msg_to_j)] =
                    -omega * eta;
            }
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < ws.n; ++i) {
            Mat9 lam =
                ws.unary_lam[static_cast<size_t>(i)];
            Vec9 eta =
                ws.unary_eta[static_cast<size_t>(i)];
            const int begin =
                ws.incoming_offsets[static_cast<size_t>(i)];
            const int end =
                ws.incoming_offsets[
                    static_cast<size_t>(i + 1)];
            for (int pos = begin; pos < end; ++pos) {
                const int msg =
                    ws.incoming_msg_ids[
                        static_cast<size_t>(pos)];
                lam.noalias() += ws.next_msg_lam[
                    static_cast<size_t>(msg)];
                eta.noalias() += ws.next_msg_eta[
                    static_cast<size_t>(msg)];
            }
            ws.next_belief_lam[static_cast<size_t>(i)] =
                lam;
            ws.next_belief_eta[static_cast<size_t>(i)] =
                eta;
        }
    }

    ws.msg_lam.swap(ws.next_msg_lam);
    ws.msg_eta.swap(ws.next_msg_eta);
    ws.belief_lam.swap(ws.next_belief_lam);
    ws.belief_eta.swap(ws.next_belief_eta);
    ws.belief_lam_inv_valid = false;
}

void schurGBPFirstZeroEtaSweep(
    SchurGBPWorkspace& ws,
    double damping,
    int threads
) {
    const double omega =
        std::min(1.0, std::max(0.0, damping));
    const Mat9 identity = Mat9::Identity();
    const Vec9 zero = Vec9::Zero();
    const int edge_count =
        static_cast<int>(ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(
        threads, edge_count, ws.n);

#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads) if(parallel_threads > 1)
#endif
    {
#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < ws.n; ++i) {
            Mat9 inverse = Mat9::Zero();
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(
                ws.unary_lam[static_cast<size_t>(i)],
                identity,
                zero,
                inverse,
                unused);
            ws.belief_lam_inv[static_cast<size_t>(i)] =
                inverse;
            mat9VecRawInto(
                inverse,
                ws.unary_eta[static_cast<size_t>(i)],
                ws.next_belief_eta[static_cast<size_t>(i)]);
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int eidx = 0; eidx < edge_count; ++eidx) {
            const SchurGBPEdge& edge =
                ws.edges[static_cast<size_t>(eidx)];
            Vec9 eta = Vec9::Zero();
            mat9VecRawInto(
                edge.aij,
                ws.next_belief_eta[
                    static_cast<size_t>(edge.j)],
                eta);
            ws.next_msg_eta[
                static_cast<size_t>(edge.msg_to_i)] =
                -omega * eta;
            mat9VecRawInto(
                edge.aji,
                ws.next_belief_eta[
                    static_cast<size_t>(edge.i)],
                eta);
            ws.next_msg_eta[
                static_cast<size_t>(edge.msg_to_j)] =
                -omega * eta;
        }

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
        for (int i = 0; i < ws.n; ++i) {
            Vec9 eta =
                ws.unary_eta[static_cast<size_t>(i)];
            const int begin =
                ws.incoming_offsets[static_cast<size_t>(i)];
            const int end =
                ws.incoming_offsets[
                    static_cast<size_t>(i + 1)];
            for (int pos = begin; pos < end; ++pos) {
                const int msg =
                    ws.incoming_msg_ids[
                        static_cast<size_t>(pos)];
                eta.noalias() += ws.next_msg_eta[
                    static_cast<size_t>(msg)];
            }
            ws.next_belief_eta[static_cast<size_t>(i)] =
                eta;
        }
    }

    ws.msg_eta.swap(ws.next_msg_eta);
    ws.belief_eta.swap(ws.next_belief_eta);
    ws.belief_lam_inv_valid = true;
}

Eigen::VectorXd solveFastHGBPWithFixedGroups(
    FastHGBPSystem& sys,
    const Args& args,
    const std::vector<std::vector<int>>& groups,
    FastHGBPCoarseWorkspace& coarse_workspace,
    bool first_sweep_eta_only,
    PackedGBPStats& stats
) {
    stats = PackedGBPStats{};
    SchurGBPWorkspace& ws = sys.ws;
    stats.edges = static_cast<int>(ws.edges.size());
    const int gbp_threads = effectiveGBPThreads(
        requestedGBPThreads(args), stats.edges, ws.n);
    const int eta_threads = effectiveGBPThreads(
        requestedGBPThreads(args), stats.edges, ws.n);
    stats.threads = gbp_threads;
    stats.reused_workspace = true;

    Eigen::VectorXd x =
        Eigen::VectorXd::Zero(9 * sys.free_cameras);
    Eigen::VectorXd rhs;
    fastHGBPRhsVectorInto(sys, rhs);
    Eigen::VectorXd residual = rhs;
    if (args.coarse_scale > 0.0 && !groups.empty()) {
        const auto coarse_t0 = Clock::now();
        buildFastHGBPCoarseWorkspace(
            coarse_workspace, sys, groups);
        stats.coarse_solve_sec +=
            elapsed(coarse_t0, Clock::now());
    }

    int remaining_full_sweeps = args.gbp_full_sweeps;
    bool fixed_maps_ready = false;
    Eigen::VectorXd cycle_delta;
    Eigen::VectorXd cycle_matvec;
    Eigen::VectorXd cycle_residual;
    Eigen::VectorXd coarse_delta;
    Eigen::VectorXd coarse_matvec;
    for (int cycle = 0; cycle < args.mg_cycles; ++cycle) {
        if (cycle != 0) {
            resetSchurGBPEtaMessages(
                ws, residual, gbp_threads);
        }
        const int full_this_cycle = std::min(
            args.pre_sweeps, remaining_full_sweeps);
        if (full_this_cycle > 0) {
            for (int sweep = 0;
                 sweep < full_this_cycle;
                 ++sweep) {
                const bool use_eta_only =
                    first_sweep_eta_only &&
                    cycle == 0 &&
                    sweep == 0;
                const auto sweep_t0 = Clock::now();
                if (use_eta_only) {
                    schurGBPFirstZeroEtaSweep(
                        ws,
                        args.message_damping,
                        gbp_threads);
                    stats.eta_sec +=
                        elapsed(sweep_t0, Clock::now());
                    ++stats.eta_sweeps;
                } else if (cycle == 0 && sweep == 0) {
                    schurGBPFirstZeroMessageSweep(
                        ws,
                        args.message_damping,
                        gbp_threads);
                    stats.full_sec +=
                        elapsed(sweep_t0, Clock::now());
                    ++stats.full_sweeps;
                } else {
                    schurGBPSweep(
                        ws,
                        args.message_damping,
                        gbp_threads);
                    stats.full_sec +=
                        elapsed(sweep_t0, Clock::now());
                    ++stats.full_sweeps;
                }
            }
            remaining_full_sweeps -= full_this_cycle;
        }
        if (full_this_cycle < args.pre_sweeps) {
            if (!fixed_maps_ready) {
                const auto map_t0 = Clock::now();
                buildSchurGBPFixedEtaMaps(
                    ws, gbp_threads);
                buildSchurGBPBeliefLambdaInverses(
                    ws, gbp_threads);
                stats.fixed_map_sec +=
                    elapsed(map_t0, Clock::now());
                fixed_maps_ready = true;
            }
            const auto eta_t0 = Clock::now();
            schurGBPFixedEtaSweeps(
                ws,
                args.message_damping,
                args.pre_sweeps - full_this_cycle,
                eta_threads);
            stats.eta_sec +=
                elapsed(eta_t0, Clock::now());
            stats.eta_sweeps +=
                args.pre_sweeps - full_this_cycle;
        }

        const auto mean_t0 = Clock::now();
        schurGBPMeanInto(ws, gbp_threads, cycle_delta);
        stats.mean_sec += elapsed(mean_t0, Clock::now());

        const auto residual_t0 = Clock::now();
        fastHGBPMultiplyInto(
            sys, cycle_delta, cycle_matvec, gbp_threads);
        cycle_residual = residual;
        cycle_residual.noalias() -= cycle_matvec;
        stats.coarse_residual_sec +=
            elapsed(residual_t0, Clock::now());

        if (args.coarse_scale > 0.0) {
            const auto coarse_t0 = Clock::now();
            solveFastHGBPCoarseCorrectionInto(
                coarse_workspace,
                cycle_residual,
                coarse_delta);
            if (args.coarse_scale != 1.0) {
                coarse_delta *= args.coarse_scale;
            }
            stats.coarse_solve_sec +=
                elapsed(coarse_t0, Clock::now());
            cycle_delta.noalias() += coarse_delta;

            const auto coarse_mv_t0 = Clock::now();
            fastCoarseAPMultiplyInto(
                coarse_workspace,
                coarse_matvec,
                gbp_threads,
                args.coarse_scale);
            stats.coarse_residual_sec +=
                elapsed(coarse_mv_t0, Clock::now());
            cycle_matvec.noalias() += coarse_matvec;
        }

        constexpr double alpha = 1.0;
        x.noalias() += alpha * cycle_delta;
        residual.noalias() -= alpha * cycle_matvec;
    }
    return x;
}

double backSubstitutePackedRoot(
    RootProblem& problem,
    const PackedRootLandmarks& packed,
    const Eigen::VectorXd& pose_increment,
    double landmark_damping
) {
    auto body = [&](const tbb::blocked_range<size_t>& range,
                    double model_decrease) {
        for (size_t p = range.begin(); p != range.end(); ++p) {
            Vec3 tmp = Vec3::Zero();
            double pose_model = 0.0;
            const int begin = packed.point_offset[p];
            const int end = packed.point_offset[p + 1];
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera_id = packed.camera_id[e];
                const Vec2 projected_increment =
                    packed.Jc[e] *
                    pose_increment.segment<9>(9 * camera_id);
                tmp.noalias() +=
                    packed.Jl[e].transpose() *
                    (packed.residual[e] + projected_increment);
                pose_model += projected_increment.dot(
                    0.5 * projected_increment +
                    packed.residual[e]);
            }

            Vec3 landmark_increment =
                -packed.Hll_inv[p] * tmp;
            const double landmark_model =
                0.5 * landmark_increment.dot(tmp) -
                0.5 * landmark_damping *
                    landmark_increment.squaredNorm();
            model_decrease -=
                pose_model + landmark_model;
            landmark_increment.array() *=
                packed.Jl_scale[p].array();
            problem.landmarks()[p].p_w += landmark_increment;
        }
        return model_decrease;
    };
    return tbb::parallel_reduce(
        tbb::blocked_range<size_t>(
            0, packed.Hll.size(), 128),
        0.0,
        body,
        std::plus<double>());
}

void applyRootCameraIncrementDirect(
    RootProblem& problem,
    const Eigen::VectorXd& increment
) {
    for (int c = 0; c < problem.num_cameras(); ++c) {
        auto& camera = problem.cameras()[static_cast<size_t>(c)];
        camera.apply_inc_pose(increment.segment<6>(9 * c));
        camera.apply_inc_intrinsics(increment.segment<3>(9 * c + 6));
    }
}

RootHGBPResult runRootHGBP(
    BALProblem& input_problem,
    const Args& args,
    const HGBPConfig& lab
) {
    RootHGBPResult result;
    const auto total_t0 = Clock::now();
    RootProblem problem = makeRootProblem(input_problem);
    const int threads = std::max(1, requestedGBPThreads(args));
    tbb::global_control thread_limit(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(threads));

    const double pose_scaling_epsilon =
        Sophus::Constants<double>::epsilonSqrt();

    const auto preprocessor_t0 = Clock::now();
    const auto preprocessor_linearization_t1 = Clock::now();
    const RootHGBPTopology topology =
        buildRootHGBPTopology(
            input_problem,
            problem,
            lab.min_pair_observations,
            threads);
    const PackedBlockSchurPattern& pattern = topology.pattern;
    const RootHGBPBuildPlan& build_plan = topology.build_plan;
    result.retained_camera_pairs =
        (pattern.row_ptr.back() - pattern.free_cameras) / 2;
    result.retained_pair_references =
        pattern.slot_pair_row_ptr.empty()
            ? 0
            : pattern.slot_pair_row_ptr.back();
    result.retained_observations =
        build_plan.retained_edge_count;
    PackedRootLandmarks packed_landmarks =
        makePackedRootLandmarks(problem, build_plan);
    refreshPackedRootCameraCache(
        problem, packed_landmarks);
    FastHGBPSystem fast_system;
    FastHGBPCoarseWorkspace coarse_workspace;
    const auto preprocessor_t1 = Clock::now();
    result.preprocessor_sec =
        elapsed(preprocessor_t0, preprocessor_t1);
    result.preprocessor_linearization_sec =
        elapsed(preprocessor_t0, preprocessor_linearization_t1);
    result.preprocessor_topology_sec =
        elapsed(preprocessor_linearization_t1, preprocessor_t1);

    RootMetrics metrics =
        computePackedRootMetrics(
            problem, packed_landmarks, false);
    if (result.costs.empty()) {
        result.costs.push_back(metrics.cost2);
    }
    double lambda = lab.initial_lambda;
    double lambda_vee = 2.0;
    bool need_linearization = true;
    Eigen::VectorXd pose_scaling;
    Eigen::VectorXd root_b;
    std::vector<std::vector<int>> fixed_groups;
    double fixed_cut_ratio = 0.0;

    for (int outer = 1; outer <= args.outer; ++outer) {
        const int hgbp_outer = outer;
        const auto outer_t0 = Clock::now();
        double linearize_time = 0.0;
        const bool fuse_linearize_build = need_linearization;
        const auto build_t0 = Clock::now();
        RootBuildBreakdown build_breakdown;
        buildPackedHGBPSystemInto(
            fast_system,
            pattern,
            build_plan,
            problem,
            packed_landmarks,
            root_b,
            lambda,
            args.build_threads,
            lab.pair_sample_cap,
            lab.pair_sample_rescale,
            lab.unreduced_unary,
            hgbp_outer <= lab.full_lambda_outers,
            fuse_linearize_build,
            !lab.no_pose_scaling,
            pose_scaling_epsilon,
            pose_scaling,
            build_breakdown);
        const auto build_t1 = Clock::now();

        int group_count = 0;
        PackedGBPStats stats;
        const auto solve_t0 = Clock::now();
        if (fixed_groups.empty()) {
            fixed_groups = buildCovisGroups(
                fast_system,
                args.group_size);
            fixed_cut_ratio =
                schurCutRatio(fast_system, fixed_groups);
        }
        group_count = static_cast<int>(fixed_groups.size());
        Eigen::VectorXd increment =
            solveFastHGBPWithFixedGroups(
                fast_system,
                args,
            fixed_groups,
            coarse_workspace,
            hgbp_outer > lab.full_lambda_outers,
            stats);
        double linear_residual =
            std::numeric_limits<double>::quiet_NaN();
        int exact_refinement_steps = 0;
        const auto exact_refinement_t0 = Clock::now();
        const bool small_problem =
            fast_system.free_cameras < 64;
        const bool run_exact_refinement =
            small_problem ||
            (outer >= 11 && fixed_cut_ratio >= 0.05);
        if (run_exact_refinement) {
            exact_refinement_steps =
                refinePackedRootSchurPCG(
                    fast_system,
                    packed_landmarks,
                    pose_scaling,
                    lambda,
                    3,
                    small_problem
                        ? std::min(6, threads)
                        : threads,
                    increment,
                    linear_residual);
        }
        const auto exact_refinement_t1 = Clock::now();
        const auto solve_t1 = Clock::now();

        const auto backup_t0 = Clock::now();
        problem.backup();
        const auto backup_t1 = Clock::now();
        double model_decrease = 0.0;
        const auto back_substitute_t0 = Clock::now();
        Eigen::VectorXd physical_increment =
            increment.cwiseProduct(pose_scaling);
        model_decrease = backSubstitutePackedRoot(
            problem,
            packed_landmarks,
            physical_increment,
            lambda);
        applyRootCameraIncrementDirect(
            problem, physical_increment);
        packed_landmarks.camera_cache_valid = false;
        const auto back_substitute_t1 = Clock::now();

        const auto metric_t0 = Clock::now();
        refreshPackedRootCameraCache(
            problem, packed_landmarks);
        const RootMetrics trial_metrics =
            computePackedRootMetrics(
                problem, packed_landmarks, false);
        const auto metric_t1 = Clock::now();
        const double actual_decrease =
            0.5 * (metrics.cost2 - trial_metrics.cost2);
        const double quality =
            actual_decrease / model_decrease;
        const bool accepted =
            std::isfinite(model_decrease) &&
            std::isfinite(quality) &&
            model_decrease > 0.0 &&
            quality > 0.0;

        if (accepted) {
            metrics = trial_metrics;
            lambda *= std::max(
                1.0 / 3.0,
                1.0 - std::pow(2.0 * quality - 1.0, 3.0));
            lambda = std::max(1e-16, lambda);
            lambda_vee = 2.0;
            need_linearization = true;
        } else {
            problem.restore();
            packed_landmarks.camera_cache_valid = false;
            lambda *= lambda_vee;
            lambda_vee *= 2.0;
            need_linearization = false;
        }

        const auto outer_t1 = Clock::now();
        result.costs.push_back(metrics.cost2);
        result.outer_sec.push_back(
            elapsed(outer_t0, outer_t1));
        result.linearize_sec.push_back(linearize_time);
        result.message_build_sec.push_back(
            elapsed(build_t0, build_t1));
        result.build_setup_sec.push_back(
            build_breakdown.setup_sec);
        result.build_point_sec.push_back(
            build_breakdown.point_sec);
        result.build_rhs_reduce_sec.push_back(
            build_breakdown.rhs_reduce_sec);
        result.build_unary_reduce_sec.push_back(
            build_breakdown.unary_reduce_sec);
        result.build_pair_sec.push_back(
            build_breakdown.pair_sec);
        result.solve_sec.push_back(
            elapsed(solve_t0, solve_t1));
        result.exact_refinement_sec.push_back(
            elapsed(
                exact_refinement_t0,
                exact_refinement_t1));
        result.backup_sec.push_back(
            elapsed(backup_t0, backup_t1));
        result.back_substitute_sec.push_back(
            elapsed(back_substitute_t0, back_substitute_t1));
        result.metric_sec.push_back(
            elapsed(metric_t0, metric_t1));
        result.gbp_full_sec.push_back(stats.full_sec);
        result.gbp_fixed_map_sec.push_back(
            stats.fixed_map_sec);
        result.gbp_eta_sec.push_back(stats.eta_sec);
        result.gbp_mean_sec.push_back(stats.mean_sec);
        result.gbp_coarse_residual_sec.push_back(
            stats.coarse_residual_sec);
        result.gbp_coarse_solve_sec.push_back(
            stats.coarse_solve_sec);
        result.accepted_alpha.push_back(
            accepted ? 1.0 : 0.0);
        result.lambda.push_back(lambda);
        result.step_quality.push_back(quality);
        result.linear_residual.push_back(linear_residual);
        result.groups.push_back(group_count);
        result.full_sweeps.push_back(stats.full_sweeps);
        result.eta_sweeps.push_back(stats.eta_sweeps);
        result.exact_refinement_steps.push_back(
            exact_refinement_steps);

        std::cout << std::setprecision(17)
                  << "[H-GBP] outer=" << outer
                  << " cost=" << metrics.cost2
                  << " accepted=" << (accepted ? 1 : 0)
                  << " quality=" << quality
                  << " lambda=" << lambda
                  << " lin_rel=" << linear_residual
                  << " linearize=" << linearize_time
                  << " message_build="
                  << result.message_build_sec.back()
                  << " solve=" << result.solve_sec.back()
                  << " backup=" << result.backup_sec.back()
                  << " backsub=" << result.back_substitute_sec.back()
                  << " metric=" << result.metric_sec.back()
                  << " groups=" << group_count
                  << " cut=" << fixed_cut_ratio
                  << " full=" << stats.full_sweeps
                  << " eta=" << stats.eta_sweeps
                  << " exact_refine=" << exact_refinement_steps
                  << "\n";
    }

    refreshPackedRootCameraCache(
        problem, packed_landmarks);
    const RootMetrics final_metrics =
        computePackedRootMetrics(
            problem, packed_landmarks, true);
    metrics.rmse_px = final_metrics.rmse_px;
    metrics.are_px = final_metrics.are_px;
    result.grouping_cut_ratio = fixed_cut_ratio;
    result.total_sec = elapsed(total_t0, Clock::now());
    result.final_cost = metrics.cost2;
    result.final_rmse_px = metrics.rmse_px;
    result.final_are_px = metrics.are_px;
    return result;
}

void writeRootHGBPJson(
    const std::string& path,
    const BALProblem& problem,
    const Args& args,
    const HGBPConfig& lab,
    const RootHGBPResult& result
) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "failed to open RootBA H-GBP JSON output: " + path);
    }
    out << std::setprecision(17);
    out << "{\n";
    out << "  \"solver\": \"hgbp_ba\",\n";
    out << "  \"num_cameras\": " << problem.num_cameras << ",\n";
    out << "  \"num_points\": " << problem.num_points << ",\n";
    out << "  \"num_observations\": " << problem.num_observations << ",\n";
    out << "  \"outer\": " << args.outer << ",\n";
    out << "  \"threads\": " << requestedGBPThreads(args) << ",\n";
    out << "  \"build_threads\": " << args.build_threads << ",\n";
    out << "  \"group_size\": " << args.group_size << ",\n";
    out << "  \"mg_cycles\": " << args.mg_cycles << ",\n";
    out << "  \"pre_sweeps\": " << args.pre_sweeps << ",\n";
    out << "  \"gbp_full_sweeps\": " << args.gbp_full_sweeps << ",\n";
    out << "  \"message_damping\": "
        << args.message_damping << ",\n";
    out << "  \"initial_lambda\": " << lab.initial_lambda << ",\n";
    out << "  \"exact_refinement_policy\": \"cost-first\",\n";
    out << "  \"full_lambda_outers\": "
        << lab.full_lambda_outers << ",\n";
    out << "  \"min_pair_observations\": "
        << lab.min_pair_observations << ",\n";
    out << "  \"pair_sample_cap\": "
        << lab.pair_sample_cap << ",\n";
    out << "  \"pair_sample_rescale\": "
        << (lab.pair_sample_rescale ? "true" : "false")
        << ",\n";
    out << "  \"unreduced_unary\": "
        << (lab.unreduced_unary ? "true" : "false")
        << ",\n";
    out << "  \"no_pose_scaling\": "
        << (lab.no_pose_scaling ? "true" : "false")
        << ",\n";
    out << "  \"preprocessor_sec\": " << result.preprocessor_sec << ",\n";
    out << "  \"preprocessor_linearization_sec\": "
        << result.preprocessor_linearization_sec << ",\n";
    out << "  \"preprocessor_topology_sec\": "
        << result.preprocessor_topology_sec << ",\n";
    out << "  \"total_sec\": " << result.total_sec << ",\n";
    out << "  \"final_cost\": " << result.final_cost << ",\n";
    out << "  \"final_reprojection_rmse_px\": "
        << result.final_rmse_px << ",\n";
    out << "  \"final_are_px\": " << result.final_are_px << ",\n";
    out << "  \"retained_camera_pairs\": "
        << result.retained_camera_pairs << ",\n";
    out << "  \"retained_pair_references\": "
        << result.retained_pair_references << ",\n";
    out << "  \"retained_observations\": "
        << result.retained_observations << ",\n";
    out << "  \"grouping_cut_ratio\": "
        << result.grouping_cut_ratio << ",\n";
    out << "  \"costs\": ";
    writeDoubleArray(out, result.costs);
    out << ",\n  \"outer_sec\": ";
    writeDoubleArray(out, result.outer_sec);
    out << ",\n  \"linearize_sec\": ";
    writeDoubleArray(out, result.linearize_sec);
    out << ",\n  \"message_build_sec\": ";
    writeDoubleArray(out, result.message_build_sec);
    out << ",\n  \"build_setup_sec\": ";
    writeDoubleArray(out, result.build_setup_sec);
    out << ",\n  \"build_point_sec\": ";
    writeDoubleArray(out, result.build_point_sec);
    out << ",\n  \"build_rhs_reduce_sec\": ";
    writeDoubleArray(out, result.build_rhs_reduce_sec);
    out << ",\n  \"build_unary_reduce_sec\": ";
    writeDoubleArray(out, result.build_unary_reduce_sec);
    out << ",\n  \"build_pair_sec\": ";
    writeDoubleArray(out, result.build_pair_sec);
    out << ",\n  \"solve_sec\": ";
    writeDoubleArray(out, result.solve_sec);
    out << ",\n  \"exact_refinement_sec\": ";
    writeDoubleArray(out, result.exact_refinement_sec);
    out << ",\n  \"backup_sec\": ";
    writeDoubleArray(out, result.backup_sec);
    out << ",\n  \"back_substitute_sec\": ";
    writeDoubleArray(out, result.back_substitute_sec);
    out << ",\n  \"metric_sec\": ";
    writeDoubleArray(out, result.metric_sec);
    out << ",\n  \"gbp_full_sec\": ";
    writeDoubleArray(out, result.gbp_full_sec);
    out << ",\n  \"gbp_fixed_map_sec\": ";
    writeDoubleArray(out, result.gbp_fixed_map_sec);
    out << ",\n  \"gbp_eta_sec\": ";
    writeDoubleArray(out, result.gbp_eta_sec);
    out << ",\n  \"gbp_mean_sec\": ";
    writeDoubleArray(out, result.gbp_mean_sec);
    out << ",\n  \"gbp_coarse_residual_sec\": ";
    writeDoubleArray(out, result.gbp_coarse_residual_sec);
    out << ",\n  \"gbp_coarse_solve_sec\": ";
    writeDoubleArray(out, result.gbp_coarse_solve_sec);
    out << ",\n  \"accepted_alpha\": ";
    writeDoubleArray(out, result.accepted_alpha);
    out << ",\n  \"lambda\": ";
    writeDoubleArray(out, result.lambda);
    out << ",\n  \"step_quality\": ";
    writeDoubleArray(out, result.step_quality);
    out << ",\n  \"linear_residual_norm\": ";
    writeDoubleArray(out, result.linear_residual);
    out << ",\n  \"groups\": ";
    writeIntArray(out, result.groups);
    out << ",\n  \"full_sweeps\": ";
    writeIntArray(out, result.full_sweeps);
    out << ",\n  \"eta_sweeps\": ";
    writeIntArray(out, result.eta_sweeps);
    out << ",\n  \"exact_refinement_steps\": ";
    writeIntArray(out, result.exact_refinement_steps);
    out << "\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::vector<std::string> filtered_storage;
        std::vector<char*> filtered_argv;
        const HGBPConfig lab =
            parseHGBPConfig(argc, argv, filtered_storage, filtered_argv);
        Args args = parseArgs(
            static_cast<int>(filtered_argv.size()), filtered_argv.data());
        configureOpenMPWaitPolicy(args);
        BALProblem problem = readBAL(args.problem_file);
        std::cout << "[BAL H-GBP] cameras=" << problem.num_cameras
                  << " points=" << problem.num_points
                  << " observations=" << problem.num_observations << "\n";
        if (args.normalize_bal) {
            normalizeBALProblem(problem);
        }
        const RootHGBPResult result = runRootHGBP(problem, args, lab);
        writeRootHGBPJson(args.out_json, problem, args, lab, result);
        std::cout << std::setprecision(17)
                  << "[H-GBP summary] total=" << result.total_sec
                  << " preprocessor=" << result.preprocessor_sec
                  << " cost=" << result.final_cost
                  << " rmse=" << result.final_rmse_px
                  << " are=" << result.final_are_px
                  << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
