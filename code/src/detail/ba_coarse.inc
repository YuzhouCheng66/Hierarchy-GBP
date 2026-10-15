void buildFastHGBPCoarseWorkspace(
    FastHGBPCoarseWorkspace& coarse,
    const FastHGBPSystem& sys,
    const std::vector<std::vector<int>>& groups
) {
    const int G = static_cast<int>(groups.size());
    const int coarse_dim = 9 * G;
    const int fine_dim = 9 * sys.free_cameras;
    coarse.free_cameras = sys.free_cameras;
    coarse.groups = G;
    coarse.gid.assign(static_cast<size_t>(sys.free_cameras), -1);
    coarse.scale.assign(static_cast<size_t>(G), 1.0);
    if (coarse.Ac.rows() != coarse_dim || coarse.Ac.cols() != coarse_dim) {
        coarse.Ac.resize(coarse_dim, coarse_dim);
    }
    coarse.Ac.setZero();
    if (coarse.bc.size() != coarse_dim) coarse.bc.resize(coarse_dim);
    if (coarse.z.size() != coarse_dim) coarse.z.resize(coarse_dim);
    if (coarse.correction.size() != fine_dim) coarse.correction.resize(fine_dim);
    coarse.bc.setZero();
    coarse.z.setZero();
    coarse.correction.setZero();

    for (int g = 0; g < G; ++g) {
        coarse.scale[static_cast<size_t>(g)] =
            1.0 / std::sqrt(static_cast<double>(std::max<size_t>(1, groups[static_cast<size_t>(g)].size())));
        for (int i : groups[static_cast<size_t>(g)]) {
            coarse.gid[static_cast<size_t>(i)] = g;
        }
    }

    coarse.ap_row_ptr.assign(static_cast<size_t>(sys.free_cameras) + 1, 0);
    coarse.ap_group_idx.clear();
    coarse.ap_blocks.clear();
    std::vector<Mat9> row_group_accum(static_cast<size_t>(G), Mat9::Zero());
    std::vector<int> touched_groups;
    std::vector<char> touched(static_cast<size_t>(G), 0);
    for (int i = 0; i < sys.free_cameras; ++i) {
        const int gi = coarse.gid[static_cast<size_t>(i)];
        coarse.ap_row_ptr[static_cast<size_t>(i)] = static_cast<int>(coarse.ap_group_idx.size());

        auto touch_group = [&](int g) {
            if (!touched[static_cast<size_t>(g)]) {
                touched[static_cast<size_t>(g)] = 1;
                touched_groups.push_back(g);
                row_group_accum[static_cast<size_t>(g)].setZero();
            }
        };

        touch_group(gi);
        row_group_accum[static_cast<size_t>(gi)].noalias() +=
            coarse.scale[static_cast<size_t>(gi)] *
            schurGBPPreconditionerLambda(sys.ws, i);

        const int begin = sys.row_block_offsets[static_cast<size_t>(i)];
        const int end = sys.row_block_offsets[static_cast<size_t>(i + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const int gj = coarse.gid[static_cast<size_t>(sys.row_block_cols[static_cast<size_t>(pos)])];
            touch_group(gj);
            row_group_accum[static_cast<size_t>(gj)].noalias() +=
                coarse.scale[static_cast<size_t>(gj)] *
                schurGBPPreconditionerEdgeScale(sys, pos) *
                sys.row_blocks[static_cast<size_t>(pos)];
        }

        std::sort(touched_groups.begin(), touched_groups.end());
        for (int gj : touched_groups) {
            const Mat9& ap_block = row_group_accum[static_cast<size_t>(gj)];
            coarse.ap_group_idx.push_back(gj);
            coarse.ap_blocks.push_back(ap_block);
            coarse.Ac.block<9, 9>(9 * gi, 9 * gj).noalias() +=
                coarse.scale[static_cast<size_t>(gi)] * ap_block;
            touched[static_cast<size_t>(gj)] = 0;
        }
        touched_groups.clear();
    }
    coarse.ap_row_ptr[static_cast<size_t>(sys.free_cameras)] =
        static_cast<int>(coarse.ap_group_idx.size());

    coarse.Ac = 0.5 * (coarse.Ac + coarse.Ac.transpose());
    coarse.ac_scale = std::max(1.0, coarse.Ac.cwiseAbs().maxCoeff());
    coarse.Ac.diagonal().array() += 1e-12 * coarse.ac_scale;
    coarse.ldlt.compute(coarse.Ac);
    if (coarse.ldlt.info() != Eigen::Success) {
        coarse.Ac.diagonal().array() += 1e-8 * coarse.ac_scale;
        coarse.ldlt.compute(coarse.Ac);
    }
    if (coarse.ldlt.info() != Eigen::Success) {
        throw std::runtime_error("fast HGBP coarse LDLT factorization failed");
    }
}

void solveFastHGBPCoarseCorrectionInto(
    FastHGBPCoarseWorkspace& coarse,
    const Eigen::VectorXd& residual,
    Eigen::VectorXd& correction
) {
    coarse.bc.setZero();
    const double* residual_data = residual.data();
    double* bc_data = coarse.bc.data();
    for (int i = 0; i < coarse.free_cameras; ++i) {
        const int g = coarse.gid[static_cast<size_t>(i)];
        const double s = coarse.scale[static_cast<size_t>(g)];
        const double* ri = residual_data + 9 * static_cast<size_t>(i);
        double* bg = bc_data + 9 * static_cast<size_t>(g);
        for (int k = 0; k < 9; ++k) {
            bg[k] += s * ri[k];
        }
    }
    coarse.z = coarse.bc;
    coarse.ldlt.solveInPlace(coarse.z);
    if (coarse.ldlt.info() != Eigen::Success || !coarse.z.allFinite()) {
        Eigen::MatrixXd reg = coarse.Ac;
        reg.diagonal().array() += 1e-8 * coarse.ac_scale;
        coarse.ldlt.compute(reg);
        coarse.z = coarse.bc;
        coarse.ldlt.solveInPlace(coarse.z);
    }
    if (correction.size() != 9 * coarse.free_cameras) {
        correction.resize(9 * coarse.free_cameras);
    }
    const double* z_data = coarse.z.data();
    double* correction_data = correction.data();
    for (int i = 0; i < coarse.free_cameras; ++i) {
        const int g = coarse.gid[static_cast<size_t>(i)];
        const double s = coarse.scale[static_cast<size_t>(g)];
        const double* zg = z_data + 9 * static_cast<size_t>(g);
        double* ci = correction_data + 9 * static_cast<size_t>(i);
        for (int k = 0; k < 9; ++k) {
            ci[k] = s * zg[k];
        }
    }
}

void fastCoarseAPMultiplyInto(
    const FastHGBPCoarseWorkspace& coarse,
    Eigen::VectorXd& y,
    int threads,
    double alpha = 1.0
) {
    const int dim = 9 * coarse.free_cameras;
    if (y.size() != dim) {
        y.resize(dim);
    }
    const int parallel_threads = effectiveGBPThreads(
        threads,
        static_cast<int>(coarse.ap_blocks.size()),
        coarse.free_cameras);
    const double* GBP_RESTRICT z_data = coarse.z.data();
    double* GBP_RESTRICT y_data = y.data();
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static) if(parallel_threads > 1)
#endif
    for (int r = 0; r < coarse.free_cameras; ++r) {
        double yr[9] = {};
        for (int pos = coarse.ap_row_ptr[static_cast<size_t>(r)];
             pos < coarse.ap_row_ptr[static_cast<size_t>(r + 1)];
             ++pos) {
            mat9VecRawAddInto(
                coarse.ap_blocks[static_cast<size_t>(pos)],
                z_data + 9 * static_cast<size_t>(coarse.ap_group_idx[static_cast<size_t>(pos)]),
                yr);
        }
        double* GBP_RESTRICT out = y_data + 9 * static_cast<size_t>(r);
        for (int k = 0; k < 9; ++k) {
            out[k] = alpha * yr[k];
        }
    }
}
