#pragma once

// Form a block-walk-summable canonical Gaussian model without duplicating
// positive landmark terms on the unary diagonal. No dataset thresholds enter.
void normalizeBAGBPCanonicalPairs(FastHGBPSystem& sys, int requested_threads) {
    SchurGBPWorkspace& ws = sys.ws;
    const int threads = effectiveGBPThreads(requested_threads,
        static_cast<int>(ws.edges.size()), ws.n);
    Mat9List inverse_lower(static_cast<size_t>(ws.n));
    int failure = 0;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:failure) if(threads > 1)
#endif
    for (int i = 0; i < ws.n; ++i) {
        Mat9& diagonal = ws.unary_lam[i];
        Eigen::LLT<Mat9> chol(diagonal);
        if (chol.info() != Eigen::Success) {
            diagonal.diagonal().array() += 100.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, diagonal.cwiseAbs().maxCoeff());
            chol.compute(diagonal);
        }
        if (chol.info() != Eigen::Success) {
            failure = 1;
        } else {
            inverse_lower[i] = chol.matrixL().solve(Mat9::Identity());
        }
    }
    if (failure) throw std::runtime_error("Canonical GBP diagonal is not SPD");
    std::vector<double> edge_norm(ws.edges.size());
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|:failure) if(threads > 1)
#endif
    for (int e = 0; e < static_cast<int>(ws.edges.size()); ++e) {
        const auto& edge = ws.edges[e];
        const Mat9& raw = sys.row_blocks[sys.edge_row_pos_i[e]];
        const Mat9 whitened = inverse_lower[edge.i] * raw * inverse_lower[edge.j].transpose();
        edge_norm[e] = whitened.norm();
    }
    std::vector<double> degree(static_cast<size_t>(ws.n), 1.0);
    for (size_t e = 0; e < ws.edges.size(); ++e) {
        if (!std::isfinite(edge_norm[e])) throw std::runtime_error("Nonfinite canonical pair norm");
        degree[ws.edges[e].i] += edge_norm[e];
        degree[ws.edges[e].j] += edge_norm[e];
    }
    for (size_t e = 0; e < ws.edges.size(); ++e) {
        auto& edge = ws.edges[e];
        edge.factor_scale = 1.0 / std::sqrt(degree[edge.i] * degree[edge.j]);
        edge.aij = edge.factor_scale * sys.row_blocks[sys.edge_row_pos_i[e]];
        edge.aji = edge.aij.transpose();
        edge.factor_lam_i.setZero();
        edge.factor_lam_j.setZero();
    }
    ws.belief_lam = ws.unary_lam;
    ws.next_belief_lam = ws.unary_lam;
}
