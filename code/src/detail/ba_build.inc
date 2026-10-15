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

template <bool BuildDiagonal = true>
GBP_FORCE_INLINE void accumulateRootPairBlock(
    const PackedBlockSchurPattern& pattern,
    const Mat93List& edge_EBinv,
    const Mat93List& edge_E,
    int slot,
    int sample_cap,
    bool sample_rescale,
    Mat9& block,
    Mat9& factor_lam_i,
    Mat9& factor_lam_j,
    const std::vector<int>* cached_slots = nullptr
) {
    block.setZero();
    factor_lam_i.setZero();
    factor_lam_j.setZero();
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
        const int retained_a =
            pattern.slot_pair_edge_a_global[
                static_cast<size_t>(ref)];
        const int retained_b =
            pattern.slot_pair_edge_b_global[
                static_cast<size_t>(ref)];
        const int edge_a = cached_slots ? (*cached_slots)[retained_a] : retained_a;
        const int edge_b = cached_slots ? (*cached_slots)[retained_b] : retained_b;
        const double* GBP_RESTRICT a =
            edge_EBinv[static_cast<size_t>(edge_a)].data();
        const double* GBP_RESTRICT b =
            edge_E[static_cast<size_t>(edge_b)].data();
        const bool transpose =
            pattern.slot_pair_transpose[static_cast<size_t>(ref)] != 0;
        if (!transpose) {
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
        if constexpr (BuildDiagonal) {
            const int edge_i = transpose ? edge_b : edge_a;
            const int edge_j = transpose ? edge_a : edge_b;
            factor_lam_i.noalias() += sample_weight *
                edge_EBinv[static_cast<size_t>(edge_i)] *
                edge_E[static_cast<size_t>(edge_i)].transpose();
            factor_lam_j.noalias() += sample_weight *
                edge_EBinv[static_cast<size_t>(edge_j)] *
                edge_E[static_cast<size_t>(edge_j)].transpose();
        }
    }
    if constexpr (BuildDiagonal) {
        factor_lam_i =
            0.5 * (factor_lam_i + factor_lam_i.transpose()).eval();
        factor_lam_j =
            0.5 * (factor_lam_j + factor_lam_j.transpose()).eval();
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
    double pair_factor_scale,
    bool fused_linearize_build,
    double pose_scaling_epsilon,
    Eigen::VectorXd& pose_scaling,
    RootBuildBreakdown& breakdown,
    const RootCrossCachePlan& cross_cache
) {
    const auto t0 = Clock::now();
    initializeFastHGBPSystemStructure(sys, pattern);
    resetRootFastHGBPWorkspaceValues(
        sys, damping);
    sys.linearized_cost = 0.0;

    if (cross_cache.sample_cap != pair_sample_cap) {
        throw std::runtime_error("Cross cache does not match the unchanged pair sampler");
    }
    const std::vector<int>& cross_slots = cross_cache.edge_slot;
    sys.edge_E.resize(static_cast<size_t>(cross_cache.slot_count));

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
        landmarks.pose_diag2_by_thread.resize(static_cast<size_t>(parallel_threads));
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
        if (fused_linearize_build) {
            landmarks
                .pose_diag2_by_thread[static_cast<size_t>(tid)]
                .setZero();
        }
    }
    const auto setup_t1 = Clock::now();
    breakdown.setup_sec =
        elapsed(t0, setup_t1);

    const auto point_t0 = Clock::now();
    int point_factorization_failed = 0;
#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads) reduction(|:point_factorization_failed)
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
                    &landmarks.pose_diag2_by_thread[static_cast<size_t>(tid)])) {
                ++landmarks.numerical_failures[
                    static_cast<size_t>(tid)];
                continue;
            }

            Mat3& L = landmarks.Hll_factor[p];
            if (!factorDampedPoint3(landmarks.Hll[p], damping, L)) {
                point_factorization_failed = 1;
                continue;
            }
            const Vec3 Hll_inv_bl = solvePoint3(L, landmarks.bl[p]);
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
                    cross_slots[
                        static_cast<size_t>(global_edge)];
                const bool pair_uses_edge = retained_edge >= 0;
                Mat93 E = Jc.transpose() * Jp;
                whitenPointCross(E, L);
                if (pair_uses_edge) {
                    sys.edge_E[static_cast<size_t>(retained_edge)] = E;
                }
                accumulateRootUnaryBlock(
                    local_diag[static_cast<size_t>(fc)], Jc, &E, &E, 1.0);
            }
        }
    }
    const auto point_t1 = Clock::now();
    if (point_factorization_failed) throw std::runtime_error("Damped landmark block is not SPD");
    breakdown.point_sec =
        elapsed(point_t0, point_t1);

    if (fused_linearize_build) {
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
        edge.factor_scale = pair_factor_scale;
        // Normalization discards pair diagonals; keep only sampled cross blocks.
        accumulateRootPairBlock<false>(pattern, sys.edge_E, sys.edge_E, edge.slot_ij,
            pair_sample_cap, pair_sample_rescale, edge.aij,
            edge.factor_lam_i, edge.factor_lam_j, &cross_cache.retained_slot);
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
        edge.aij *= pair_factor_scale;
        edge.aji *= pair_factor_scale;
    }
    const auto pair_t1 = Clock::now();
    breakdown.pair_sec =
        elapsed(pair_t0, pair_t1);

    sys.matrix_fingerprint = std::sqrt(matrix_norm_sq);
    sys.rhs_fingerprint = std::sqrt(rhs_norm_sq);
    sys.build_sec = elapsed(t0, Clock::now());
}

void refreshPSDPairPreconditioner(
    FastHGBPSystem& sys,
    int requested_threads
) {
    SchurGBPWorkspace& ws = sys.ws;
    ws.preconditioner_lam.resize(static_cast<size_t>(ws.n));
    const int threads = effectiveGBPThreads(
        requested_threads, static_cast<int>(ws.edges.size()), ws.n);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) if(threads > 1)
#endif
    for (int camera = 0; camera < ws.n; ++camera) {
        Mat9 loaded = ws.unary_lam[static_cast<size_t>(camera)];
        const int begin =
            sys.row_edge_offsets[static_cast<size_t>(camera)];
        const int end =
            sys.row_edge_offsets[static_cast<size_t>(camera + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(
                sys.row_edge_ids[static_cast<size_t>(pos)])];
            loaded.noalias() += camera == edge.i
                ? edge.factor_lam_i
                : edge.factor_lam_j;
        }
        ws.preconditioner_lam[static_cast<size_t>(camera)] = loaded;
    }
}

#include "ba_gbp_model.h"
