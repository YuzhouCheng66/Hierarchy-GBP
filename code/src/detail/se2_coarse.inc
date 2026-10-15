bool rowsAreContiguous(const std::vector<int>& rows) {
    if (rows.empty()) {
        return false;
    }
    const int first = rows.front();
    for (int i = 1; i < static_cast<int>(rows.size()); ++i) {
        if (rows[static_cast<size_t>(i)] != first + i) {
            return false;
        }
    }
    return true;
}


void prolongToFineInto(
    const BasisData& basis,
    const Eigen::VectorXd& coarse_vec,
    Eigen::VectorXd& fine
) {
    if (fine.size() != basis.total_dim) {
        fine.resize(basis.total_dim);
    }
    fine.setZero();
    for (int g = 0; g < static_cast<int>(basis.local_bases.size()); ++g) {
        const std::vector<int>& rows = basis.full_indices_per_group[g];
        const int coarse_offset = basis.coarse_offsets[g];
        const int coarse_dim = basis.local_bases[g].cols();
        if (rowsAreContiguous(rows)) {
            fine.segment(rows.front(), rows.size()).noalias() =
                basis.local_bases[g] * coarse_vec.segment(coarse_offset, coarse_dim);
        } else {
            const Eigen::VectorXd fine_local =
                basis.local_bases[g] * coarse_vec.segment(coarse_offset, coarse_dim);
            for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
                fine(rows[i]) = fine_local(i);
            }
        }
    }
}

inline void accumBasisTransposeTimesVec3(
    const double* basis_block,
    int r_local,
    const double* vec3,
    double* coarse_segment
) {
    for (int c = 0; c < r_local; ++c) {
        const double* b = basis_block + 3 * c;
        coarse_segment[c] += b[0] * vec3[0] + b[1] * vec3[1] + b[2] * vec3[2];
    }
}

inline void accumBasisTransposeLamBasis3x3(
    const double* basis_a,
    int r_a,
    const double* lam3,
    const double* basis_b,
    int r_b,
    double* dense,
    int dense_cols,
    int row_offset,
    int col_offset
) {
    for (int cb = 0; cb < r_b; ++cb) {
        const double* bb = basis_b + 3 * cb;
        const double lb0 = lam3[0] * bb[0] + lam3[3] * bb[1] + lam3[6] * bb[2];
        const double lb1 = lam3[1] * bb[0] + lam3[4] * bb[1] + lam3[7] * bb[2];
        const double lb2 = lam3[2] * bb[0] + lam3[5] * bb[1] + lam3[8] * bb[2];
        for (int ca = 0; ca < r_a; ++ca) {
            const double* ba = basis_a + 3 * ca;
            dense[(col_offset + cb) * dense_cols + (row_offset + ca)] +=
                ba[0] * lb0 + ba[1] * lb1 + ba[2] * lb2;
        }
    }
}

inline void packFactorBlock3x3(
    const double* src,
    int dim,
    int row0,
    int col0,
    double* dst
) {
    for (int c = 0; c < 3; ++c) {
        const double* src_col = src + (col0 + c) * dim + row0;
        dst[3 * c + 0] = src_col[0];
        dst[3 * c + 1] = src_col[1];
        dst[3 * c + 2] = src_col[2];
    }
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectOrdered3D(
    const gbp::FactorGraph& graph,
    const BasisData& basis
) {
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(basis.coarse_dim, basis.coarse_dim);
    double* dense_data = dense.data();

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        if (var.dofs != 3) {
            continue;
        }
        const int var_id = var.variableID;
        const int r_local = basis.var_r_local[var_id];
        const int coarse_offset = basis.var_coarse_offset[var_id];
        const double* basis_block = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3(
            basis_block,
            r_local,
            var.prior.lamData(),
            basis_block,
            r_local,
            dense_data,
            basis.coarse_dim,
            coarse_offset,
            coarse_offset
        );
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor || !factor->active) {
            continue;
        }

        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity == 1 && factor->adj_var_nodes[0]->dofs == 3) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const int id0 = v0->variableID;
            const int r0 = basis.var_r_local[id0];
            const int off0 = basis.var_coarse_offset[id0];
            const double* B0 = varBasis3xRPtr(basis, id0);
            accumBasisTransposeLamBasis3x3(
                B0, r0, factor->factor.lamData(), B0, r0, dense_data, basis.coarse_dim, off0, off0
            );
            continue;
        }

        if (arity == 2 &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor->adj_var_nodes[1];
            const int id0 = v0->variableID;
            const int id1 = v1->variableID;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* lam = factor->factor.lamData();
            double block00[9];
            double block01[9];
            double block10[9];
            double block11[9];
            packFactorBlock3x3(lam, 6, 0, 0, block00);
            packFactorBlock3x3(lam, 6, 0, 3, block01);
            packFactorBlock3x3(lam, 6, 3, 0, block10);
            packFactorBlock3x3(lam, 6, 3, 3, block11);

            accumBasisTransposeLamBasis3x3(B0, r0, block00, B0, r0, dense_data, basis.coarse_dim, off0, off0);
            accumBasisTransposeLamBasis3x3(B0, r0, block01, B1, r1, dense_data, basis.coarse_dim, off0, off1);
            accumBasisTransposeLamBasis3x3(B1, r1, block10, B0, r0, dense_data, basis.coarse_dim, off1, off0);
            accumBasisTransposeLamBasis3x3(B1, r1, block11, B1, r1, dense_data, basis.coarse_dim, off1, off1);
            continue;
        }
    }

    Eigen::SparseMatrix<double> coarse = dense.sparseView(0.0, 0.0);
    coarse.makeCompressed();
    return symmetrizeSparse(coarse);
}

Eigen::VectorXd assembleCoarseResidualDirectOrdered3D(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec
) {
    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);
    double* coarse_data = coarse.data();
    const double* fine_data = fine_vec.data();

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        if (var.dofs != 3) {
            continue;
        }
        const int var_id = var.variableID;
        const int r_local = basis.var_r_local[var_id];
        const int coarse_offset = basis.var_coarse_offset[var_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        const double* x = fine_data + 3 * var_id;
        const double* eta = var.prior.etaData();
        const double* lam = var.prior.lamData();
        double residual[3] = {
            eta[0] - (lam[0] * x[0] + lam[3] * x[1] + lam[6] * x[2]),
            eta[1] - (lam[1] * x[0] + lam[4] * x[1] + lam[7] * x[2]),
            eta[2] - (lam[2] * x[0] + lam[5] * x[1] + lam[8] * x[2]),
        };
        accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + coarse_offset);
    }

    for (const auto& fup : graph.factors) {
        const gbp::Factor* factor = fup.get();
        if (!factor || !factor->active) {
            continue;
        }

        const int arity = static_cast<int>(factor->adj_var_nodes.size());
        if (arity == 1 && factor->adj_var_nodes[0]->dofs == 3) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const int id0 = v0->variableID;
            const int r0 = basis.var_r_local[id0];
            const int off0 = basis.var_coarse_offset[id0];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* x0 = fine_data + 3 * id0;
            const double* eta = factor->factor.etaData();
            const double* lam = factor->factor.lamData();
            double residual[3] = {
                eta[0] - (lam[0] * x0[0] + lam[3] * x0[1] + lam[6] * x0[2]),
                eta[1] - (lam[1] * x0[0] + lam[4] * x0[1] + lam[7] * x0[2]),
                eta[2] - (lam[2] * x0[0] + lam[5] * x0[1] + lam[8] * x0[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, residual, coarse_data + off0);
            continue;
        }

        if (arity == 2 &&
            factor->adj_var_nodes[0]->dofs == 3 &&
            factor->adj_var_nodes[1]->dofs == 3 &&
            factor->factor.dim() == 6) {
            const gbp::VariableNode* v0 = factor->adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor->adj_var_nodes[1];
            const int id0 = v0->variableID;
            const int id1 = v1->variableID;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* x0 = fine_data + 3 * id0;
            const double* x1 = fine_data + 3 * id1;
            const double* eta = factor->factor.etaData();
            const double* lam = factor->factor.lamData();
            double r0_vec[3] = {
                eta[0] - (lam[0] * x0[0] + lam[6] * x0[1] + lam[12] * x0[2] + lam[18] * x1[0] + lam[24] * x1[1] + lam[30] * x1[2]),
                eta[1] - (lam[1] * x0[0] + lam[7] * x0[1] + lam[13] * x0[2] + lam[19] * x1[0] + lam[25] * x1[1] + lam[31] * x1[2]),
                eta[2] - (lam[2] * x0[0] + lam[8] * x0[1] + lam[14] * x0[2] + lam[20] * x1[0] + lam[26] * x1[1] + lam[32] * x1[2]),
            };
            double r1_vec[3] = {
                eta[3] - (lam[3] * x0[0] + lam[9] * x0[1] + lam[15] * x0[2] + lam[21] * x1[0] + lam[27] * x1[1] + lam[33] * x1[2]),
                eta[4] - (lam[4] * x0[0] + lam[10] * x0[1] + lam[16] * x0[2] + lam[22] * x1[0] + lam[28] * x1[1] + lam[34] * x1[2]),
                eta[5] - (lam[5] * x0[0] + lam[11] * x0[1] + lam[17] * x0[2] + lam[23] * x1[0] + lam[29] * x1[1] + lam[35] * x1[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
            accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
            continue;
        }
    }

    return coarse;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    bool use_parallel,
    int num_threads
) {
    if (num_threads <= 1 && basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        return assembleCoarseLambdaDirectOrdered3D(graph, basis);
    }

    const int num_groups = static_cast<int>(basis.local_bases.size());
    const int num_block_pairs = num_groups * num_groups;

    std::vector<Eigen::MatrixXd> blocks(num_block_pairs);
    std::vector<char> active(num_block_pairs, 0);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const auto basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const int block_index = group * num_groups + group;
        if (!active[block_index]) {
            blocks[block_index] = Eigen::MatrixXd::Zero(basis_block.cols(), basis_block.cols());
            active[block_index] = 1;
        }
        blocks[block_index].noalias() += basis_block.transpose() * var.prior.lam() * basis_block;
    }

    auto accumulate_factor = [&](const gbp::Factor& factor, std::vector<Eigen::MatrixXd>& target_blocks, std::vector<char>& target_active) {
        if (!factor.active) {
            return;
        }

        std::vector<int> local_factor_offsets(factor.adj_var_nodes.size(), 0);
        int factor_offset = 0;
        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            local_factor_offsets[a] = factor_offset;
            factor_offset += factor.adj_var_nodes[a]->dofs;
        }

        for (int a = 0; a < static_cast<int>(factor.adj_var_nodes.size()); ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int ga = basis.var_to_group[va->variableID];
            const int la = basis.var_to_local_offset[va->variableID];
            const int da = va->dofs;
            const auto Ba = basis.local_bases[ga].middleRows(la, da);
            const int off_a = local_factor_offsets[a];

            for (int b = 0; b < static_cast<int>(factor.adj_var_nodes.size()); ++b) {
                const gbp::VariableNode* vb = factor.adj_var_nodes[b];
                const int gb = basis.var_to_group[vb->variableID];
                const int lb = basis.var_to_local_offset[vb->variableID];
                const int db = vb->dofs;
                const auto Bb = basis.local_bases[gb].middleRows(lb, db);
                const int off_b = local_factor_offsets[b];
                const int block_index = ga * num_groups + gb;

                if (!target_active[block_index]) {
                    target_blocks[block_index] = Eigen::MatrixXd::Zero(Ba.cols(), Bb.cols());
                    target_active[block_index] = 1;
                }

                target_blocks[block_index].noalias() +=
                    Ba.transpose() *
                    factor.factor.lam().block(off_a, off_b, da, db) *
                    Bb;
            }
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<std::vector<Eigen::MatrixXd>> tls_blocks(thread_count);
        std::vector<std::vector<char>> tls_active(thread_count);
        for (int tid = 0; tid < thread_count; ++tid) {
            tls_blocks[tid].resize(num_block_pairs);
            tls_active[tid].assign(num_block_pairs, 0);
        }

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor(*factor, tls_blocks[tid], tls_active[tid]);
        }

        for (int tid = 0; tid < thread_count; ++tid) {
            for (int block_index = 0; block_index < num_block_pairs; ++block_index) {
                if (!tls_active[tid][block_index]) {
                    continue;
                }
                if (!active[block_index]) {
                    blocks[block_index] = std::move(tls_blocks[tid][block_index]);
                    active[block_index] = 1;
                } else {
                    blocks[block_index] += tls_blocks[tid][block_index];
                }
            }
        }
    } else {
        for (const auto& fup : graph.factors) {
            if (!fup) {
                continue;
            }
            accumulate_factor(*fup, blocks, active);
        }
    }

    std::vector<Eigen::Triplet<double>> trips;
    for (int gi = 0; gi < num_groups; ++gi) {
        const int row_offset = basis.coarse_offsets[gi];
        for (int gj = 0; gj < num_groups; ++gj) {
            const int block_index = gi * num_groups + gj;
            if (!active[block_index]) {
                continue;
            }
            const int col_offset = basis.coarse_offsets[gj];
            const Eigen::MatrixXd& block = blocks[block_index];
            for (int r = 0; r < block.rows(); ++r) {
                for (int c = 0; c < block.cols(); ++c) {
                    const double value = block(r, c);
                    if (value != 0.0) {
                        trips.emplace_back(row_offset + r, col_offset + c, value);
                    }
                }
            }
        }
    }

    Eigen::SparseMatrix<double> coarse_lam(basis.coarse_dim, basis.coarse_dim);
    coarse_lam.setFromTriplets(trips.begin(), trips.end());
    coarse_lam.makeCompressed();
    return symmetrizeSparse(coarse_lam);
}

Eigen::VectorXd assembleCoarseResidualDirect(
    const gbp::FactorGraph& graph,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    bool use_parallel,
    int num_threads
) {
    if (num_threads <= 1 && basis.ordered_3d_fast && basis.factors_arity12_only_3d) {
        return assembleCoarseResidualDirectOrdered3D(graph, basis, fine_vec);
    }

    Eigen::VectorXd coarse = Eigen::VectorXd::Zero(basis.coarse_dim);

    for (const auto& vup : graph.var_nodes) {
        if (!vup) {
            continue;
        }
        const gbp::VariableNode& var = *vup;
        const int group = basis.var_to_group[var.variableID];
        if (group < 0) {
            continue;
        }
        const int local_offset = basis.var_to_local_offset[var.variableID];
        const auto basis_block = basis.local_bases[group].middleRows(local_offset, var.dofs);
        const auto x_local = fine_vec.segment(3 * var.variableID, var.dofs);
        coarse.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
            basis_block.transpose() * (var.prior.eta() - var.prior.lam() * x_local);
    }

    auto accumulate_factor_residual = [&](const gbp::Factor& factor, Eigen::VectorXd& target) {
        if (!factor.active) {
            return;
        }

        const int arity = static_cast<int>(factor.adj_var_nodes.size());
        if (arity == 1) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const int g0 = basis.var_to_group[v0->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int d0 = v0->dofs;
            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0;
            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            return;
        }

        if (arity == 2) {
            const gbp::VariableNode* v0 = factor.adj_var_nodes[0];
            const gbp::VariableNode* v1 = factor.adj_var_nodes[1];
            const int d0 = v0->dofs;
            const int d1 = v1->dofs;

            const int g0 = basis.var_to_group[v0->variableID];
            const int g1 = basis.var_to_group[v1->variableID];
            const int l0 = basis.var_to_local_offset[v0->variableID];
            const int l1 = basis.var_to_local_offset[v1->variableID];

            const Eigen::MatrixXd B0 = basis.local_bases[g0].middleRows(l0, d0);
            const Eigen::MatrixXd B1 = basis.local_bases[g1].middleRows(l1, d1);
            const Eigen::VectorXd x0 = fine_vec.segment(3 * v0->variableID, d0);
            const Eigen::VectorXd x1 = fine_vec.segment(3 * v1->variableID, d1);

            const Eigen::VectorXd r0 =
                factor.factor.eta().segment(0, d0) -
                factor.factor.lam().block(0, 0, d0, d0) * x0 -
                factor.factor.lam().block(0, d0, d0, d1) * x1;
            const Eigen::VectorXd r1 =
                factor.factor.eta().segment(d0, d1) -
                factor.factor.lam().block(d0, 0, d1, d0) * x0 -
                factor.factor.lam().block(d0, d0, d1, d1) * x1;

            target.segment(basis.coarse_offsets[g0], B0.cols()).noalias() += B0.transpose() * r0;
            target.segment(basis.coarse_offsets[g1], B1.cols()).noalias() += B1.transpose() * r1;
            return;
        }

        std::vector<int> local_factor_offsets(arity, 0);
        int factor_dim = 0;
        for (int a = 0; a < arity; ++a) {
            local_factor_offsets[a] = factor_dim;
            factor_dim += factor.adj_var_nodes[a]->dofs;
        }

        Eigen::VectorXd factor_x = Eigen::VectorXd::Zero(factor_dim);
        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            factor_x.segment(local_factor_offsets[a], va->dofs) = fine_vec.segment(3 * va->variableID, va->dofs);
        }
        const Eigen::VectorXd factor_residual = factor.factor.eta() - factor.factor.lam() * factor_x;

        for (int a = 0; a < arity; ++a) {
            const gbp::VariableNode* va = factor.adj_var_nodes[a];
            const int group = basis.var_to_group[va->variableID];
            const int local_offset = basis.var_to_local_offset[va->variableID];
            const int dofs = va->dofs;
            const Eigen::MatrixXd basis_block = basis.local_bases[group].middleRows(local_offset, dofs);
            target.segment(basis.coarse_offsets[group], basis.local_bases[group].cols()).noalias() +=
                basis_block.transpose() * factor_residual.segment(local_factor_offsets[a], dofs);
        }
    };

    if (use_parallel && graph.factors.size() > 64) {
        const int thread_count = std::max(1, (num_threads > 0) ? num_threads : omp_get_max_threads());
        std::vector<Eigen::VectorXd> tls(thread_count, Eigen::VectorXd::Zero(basis.coarse_dim));

        #pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int fi = 0; fi < static_cast<int>(graph.factors.size()); ++fi) {
            const int tid = omp_get_thread_num();
            const gbp::Factor* factor = graph.factors[fi].get();
            if (!factor) {
                continue;
            }
            accumulate_factor_residual(*factor, tls[tid]);
        }

        for (const Eigen::VectorXd& local : tls) {
            coarse += local;
        }
    } else {
        for (const auto& fup : graph.factors) {
            if (!fup) {
                continue;
            }
            accumulate_factor_residual(*fup, coarse);
        }
    }

    return coarse;
}
