// Frozen reference from the 2026-09-22 selected source, function name only changed.
void packedRootExactSchurMultiplyIntoReference(
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
            const Vec3 eliminated_projection = solvePoint3(packed.Hll_factor[p], point_projection);
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
