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

#include "ba_point_schur.h"

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
    Mat3List Hll_factor;
    Vec3List bl;
    Vec3List Jl_scale;
    std::vector<Eigen::VectorXd> pose_diag2_by_thread;
    std::vector<Eigen::VectorXd> thread_rhs;
    std::vector<Eigen::VectorXd> thread_matvec;
    Eigen::VectorXd schur_scaled_x;
    std::vector<int> numerical_failures;
    bool camera_cache_valid = false;
    int schur_max_point_degree = -1;
    std::vector<Vec2List> schur_projected_by_thread;
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
    packed.Hll_factor.resize(point_count);
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
        static_cast<int>(packed.Hll_factor.size());
    const Eigen::Index dimension =
        static_cast<Eigen::Index>(9 * camera_count);
    if (x.size() != dimension ||
        pose_scaling.size() != dimension) {
        throw std::runtime_error(
            "packed exact Schur dimensions do not match");
    }
    const int threads = effectiveBuildThreads(
        requested_threads, point_count, camera_count);
    if (packed.schur_max_point_degree < 0) {
        packed.schur_max_point_degree = 0;
        for (int p = 0; p < point_count; ++p) {
            packed.schur_max_point_degree = std::max(packed.schur_max_point_degree,
                packed.point_offset[p + 1] - packed.point_offset[p]);
        }
    }
    packed.schur_projected_by_thread.resize(static_cast<size_t>(threads));
    for (Vec2List& scratch : packed.schur_projected_by_thread) {
        scratch.resize(static_cast<size_t>(packed.schur_max_point_degree));
    }
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
        Vec2* GBP_RESTRICT projections =
            packed.schur_projected_by_thread[static_cast<size_t>(tid)].data();
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
                // Store once after accumulating in registers; reuse in the transpose pass.
                projections[edge - begin] = projected;
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
            const Vec3 eliminated_projection = solvePoint3(packed.Hll_factor[p], point_projection);
            for (int edge = begin; edge < end; ++edge) {
                const size_t e = static_cast<size_t>(edge);
                const int camera = packed.camera_id[e];
                const Eigen::Index offset =
                    static_cast<Eigen::Index>(9 * camera);
                const Vec2& projected = projections[edge - begin];
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
