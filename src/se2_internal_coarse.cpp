#include "internal/se2_coarse.h"

#include <Eigen/Dense>
#include <cstdlib>
#include <omp.h>

namespace slam {

namespace {

inline const double* sym6Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 6;
}

inline const double* vec3Ptr(const std::vector<double>& buf, int idx) noexcept {
    return buf.data() + static_cast<size_t>(idx) * 3;
}

inline void accumBasisTransposeLamBasis3x3Sym6(
    const double* basis_a,
    int r_a,
    const double* lam6,
    const double* basis_b,
    int r_b,
    double* dense,
    int dense_cols,
    int row_offset,
    int col_offset
) {
    const double l00 = lam6[0];
    const double l10 = lam6[1];
    const double l20 = lam6[2];
    const double l11 = lam6[3];
    const double l21 = lam6[4];
    const double l22 = lam6[5];
    for (int cb = 0; cb < r_b; ++cb) {
        const double* bb = basis_b + 3 * cb;
        const double lb0 = l00 * bb[0] + l10 * bb[1] + l20 * bb[2];
        const double lb1 = l10 * bb[0] + l11 * bb[1] + l21 * bb[2];
        const double lb2 = l20 * bb[0] + l21 * bb[1] + l22 * bb[2];
        for (int ca = 0; ca < r_a; ++ca) {
            const double* ba = basis_a + 3 * ca;
            dense[(col_offset + cb) * dense_cols + (row_offset + ca)] +=
                ba[0] * lb0 + ba[1] * lb1 + ba[2] * lb2;
        }
    }
}

inline void accumBasisTransposeLamBasis3x3Cross(
    const double* basis_a,
    int r_a,
    const double* cross01,
    bool transpose_cross,
    const double* basis_b,
    int r_b,
    double* dense,
    int dense_cols,
    int row_offset,
    int col_offset
) {
    const double x00 = cross01[0];
    const double x10 = cross01[1];
    const double x20 = cross01[2];
    const double x01 = cross01[3];
    const double x11 = cross01[4];
    const double x21 = cross01[5];
    const double x02 = cross01[6];
    const double x12 = cross01[7];
    const double x22 = cross01[8];
    for (int cb = 0; cb < r_b; ++cb) {
        const double* bb = basis_b + 3 * cb;
        double lb0, lb1, lb2;
        if (!transpose_cross) {
            lb0 = x00 * bb[0] + x01 * bb[1] + x02 * bb[2];
            lb1 = x10 * bb[0] + x11 * bb[1] + x12 * bb[2];
            lb2 = x20 * bb[0] + x21 * bb[1] + x22 * bb[2];
        } else {
            lb0 = x00 * bb[0] + x10 * bb[1] + x20 * bb[2];
            lb1 = x01 * bb[0] + x11 * bb[1] + x21 * bb[2];
            lb2 = x02 * bb[0] + x12 * bb[1] + x22 * bb[2];
        }
        for (int ca = 0; ca < r_a; ++ca) {
            const double* ba = basis_a + 3 * ca;
            dense[(col_offset + cb) * dense_cols + (row_offset + ca)] +=
                ba[0] * lb0 + ba[1] * lb1 + ba[2] * lb2;
        }
    }
}

template <typename Scalar>
inline void accumBasisTransposeTimesVec3(
    const double* basis_block,
    int r_local,
    const double* vec3,
    Scalar* coarse_segment
) {
    for (int c = 0; c < r_local; ++c) {
        const double* b = basis_block + 3 * c;
        coarse_segment[c] +=
            static_cast<Scalar>(b[0]) * static_cast<Scalar>(vec3[0]) +
            static_cast<Scalar>(b[1]) * static_cast<Scalar>(vec3[1]) +
            static_cast<Scalar>(b[2]) * static_cast<Scalar>(vec3[2]);
    }
}

inline int effectiveThreadCountPackedCoarse(int num_threads) noexcept {
    if (num_threads <= 0) {
        return std::max(1, omp_get_max_threads());
    }
    return std::max(1, num_threads);
}

std::vector<std::pair<int, int>> buildBalancedIndexRangesPackedCoarse(
    int count,
    int thread_count
) {
    std::vector<std::pair<int, int>> ranges(static_cast<size_t>(thread_count), {0, 0});
    if (thread_count <= 1 || count <= 0) {
        if (!ranges.empty()) {
            ranges[0] = {0, count};
        }
        return ranges;
    }
    for (int tid = 0; tid < thread_count; ++tid) {
        const int begin = (count * tid) / thread_count;
        const int end = (count * (tid + 1)) / thread_count;
        ranges[static_cast<size_t>(tid)] = {begin, end};
    }
    return ranges;
}

int ensureActiveBlock(
    SyntheticSE2PackedCoarseAssemblyWorkspace& workspace,
    int gi,
    int gj,
    int rows,
    int cols,
    int row_offset,
    int col_offset
) {
    const int pair_index = gi * workspace.num_groups + gj;
    int block_index = workspace.pair_to_block_index[pair_index];
    if (block_index >= 0) {
        return block_index;
    }

    block_index = static_cast<int>(workspace.active_blocks.size());
    workspace.pair_to_block_index[pair_index] = block_index;
    SyntheticSE2PackedCoarseActiveBlock block;
    block.row_offset = row_offset;
    block.col_offset = col_offset;
    block.rows = rows;
    block.cols = cols;
    block.storage_offset = workspace.block_storage_size;
    block.values = Eigen::MatrixXd::Zero(rows, cols);
    workspace.block_storage_size += static_cast<size_t>(rows) * static_cast<size_t>(cols);
    workspace.triplet_capacity += static_cast<size_t>(rows) * static_cast<size_t>(cols);
    workspace.active_blocks.push_back(std::move(block));
    return block_index;
}

bool disablePackedOmpRangeCoarse() noexcept {
    static const bool disabled = []() {
        const char* value = std::getenv("GBP_DISABLE_PACKED_OMP_RANGE_COARSE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return disabled;
}

bool disablePackedOmpRangeCoarseLambda() noexcept {
    static const bool disabled = []() {
        const char* value = std::getenv("GBP_DISABLE_PACKED_OMP_RANGE_COARSE_LAMBDA");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return disabled;
}

bool disablePackedOmpRangeCoarseResidual() noexcept {
    static const bool disabled = []() {
        const char* value = std::getenv("GBP_DISABLE_PACKED_OMP_RANGE_COARSE_ETA");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return disabled;
}

bool enablePackedStableCoarseResidualReduce() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_COARSE_ETA_STABLE_REDUCE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool enablePackedLongDoubleCoarseResidualLocals() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_COARSE_ETA_LONG_DOUBLE_LOCALS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

bool enablePackedStableCoarseLambdaReduce() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("GBP_ENABLE_PACKED_COARSE_LAMBDA_STABLE_REDUCE");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

void reduceActiveBlockStable(
    SyntheticSE2PackedCoarseActiveBlock& block,
    const std::vector<std::vector<double>>& thread_local_block_buffers,
    int thread_count
) {
    const int rows = block.rows;
    const int cols = block.cols;
    const int elems = rows * cols;
    double* dest_ptr = block.values.data();
    for (int idx = 0; idx < elems; ++idx) {
        long double acc = 0.0L;
        for (int worker = 0; worker < thread_count; ++worker) {
            const double* src_ptr =
                thread_local_block_buffers[worker].data() + block.storage_offset;
            acc += static_cast<long double>(src_ptr[idx]);
        }
        dest_ptr[idx] = static_cast<double>(acc);
    }
}

void reduceCoarseLocals(
    const std::vector<Eigen::VectorXd>& coarse_locals,
    Eigen::VectorXd& coarse
) {
    if (!enablePackedStableCoarseResidualReduce()) {
        for (const Eigen::VectorXd& local : coarse_locals) {
            coarse += local;
        }
        return;
    }

    const int dim = static_cast<int>(coarse.size());
    const int thread_count = static_cast<int>(coarse_locals.size());
    for (int i = 0; i < dim; ++i) {
        long double acc = 0.0L;
        for (int tid = 0; tid < thread_count; ++tid) {
            acc += static_cast<long double>(coarse_locals[tid][i]);
        }
        coarse[i] += static_cast<double>(acc);
    }
}

std::vector<Eigen::VectorXd>& reusableCoarseResidualLocals(int thread_count, int coarse_dim) {
    static thread_local std::vector<Eigen::VectorXd> locals;
    if (static_cast<int>(locals.size()) != thread_count) {
        locals.resize(static_cast<size_t>(thread_count));
    }
    for (Eigen::VectorXd& local : locals) {
        if (local.size() != coarse_dim) {
            local.resize(coarse_dim);
        }
        local.setZero();
    }
    return locals;
}

void ensureThreadLocalBlockBuffers(
    SyntheticSE2PackedCoarseAssemblyWorkspace& workspace,
    int thread_count
) {
    if (workspace.block_buffer_thread_count == thread_count &&
        static_cast<int>(workspace.thread_local_block_buffers.size()) == thread_count) {
        bool all_match = true;
        for (const std::vector<double>& buffer : workspace.thread_local_block_buffers) {
            if (buffer.size() != workspace.block_storage_size) {
                all_match = false;
                break;
            }
        }
        if (all_match) {
            return;
        }
    }

    workspace.block_buffer_thread_count = thread_count;
    workspace.thread_local_block_buffers.assign(
        thread_count,
        std::vector<double>(workspace.block_storage_size, 0.0)
    );
}

inline double* blockBufferPtr(
    std::vector<double>& buffer,
    const SyntheticSE2PackedCoarseActiveBlock& block
) noexcept {
    return buffer.data() + block.storage_offset;
}

template <typename BinaryFactor>
void initializeCoarseAssemblyWorkspaceCommon(
    SyntheticSE2PackedCoarseAssemblyWorkspace& workspace,
    const BasisData& basis,
    int num_vars,
    const std::vector<int>& unary_var_ids,
    const std::vector<BinaryFactor>& binary_factors
) {
    workspace = SyntheticSE2PackedCoarseAssemblyWorkspace{};
    workspace.coarse_dim = basis.coarse_dim;
    workspace.num_groups = static_cast<int>(basis.coarse_offsets.size()) - 1;
    workspace.pair_to_block_index.assign(
        workspace.num_groups * workspace.num_groups,
        -1
    );
    workspace.var_diag_block_slots.resize(num_vars, -1);
    workspace.unary_diag_block_slots.resize(unary_var_ids.size(), -1);
    workspace.binary_block_slots.resize(binary_factors.size());

    for (int var_id = 0; var_id < num_vars; ++var_id) {
        const int g = basis.var_to_group[var_id];
        const int r_local = basis.var_r_local[var_id];
        workspace.var_diag_block_slots[var_id] = ensureActiveBlock(
            workspace,
            g,
            g,
            r_local,
            r_local,
            basis.coarse_offsets[g],
            basis.coarse_offsets[g]
        );
    }

    for (int unary_id = 0; unary_id < static_cast<int>(unary_var_ids.size()); ++unary_id) {
        const int var_id = unary_var_ids[unary_id];
        workspace.unary_diag_block_slots[unary_id] = workspace.var_diag_block_slots[var_id];
    }

    for (int factor_idx = 0; factor_idx < static_cast<int>(binary_factors.size()); ++factor_idx) {
        const BinaryFactor& factor = binary_factors[factor_idx];
        const int id0 = factor.var0_id;
        const int id1 = factor.var1_id;
        const int g0 = basis.var_to_group[id0];
        const int g1 = basis.var_to_group[id1];
        const int r0 = basis.var_r_local[id0];
        const int r1 = basis.var_r_local[id1];
        SyntheticSE2PackedCoarseBinaryBlockSlots slots;
        slots.idx00 = ensureActiveBlock(
            workspace,
            g0,
            g0,
            r0,
            r0,
            basis.coarse_offsets[g0],
            basis.coarse_offsets[g0]
        );
        slots.idx01 = ensureActiveBlock(
            workspace,
            g0,
            g1,
            r0,
            r1,
            basis.coarse_offsets[g0],
            basis.coarse_offsets[g1]
        );
        slots.idx10 = ensureActiveBlock(
            workspace,
            g1,
            g0,
            r1,
            r0,
            basis.coarse_offsets[g1],
            basis.coarse_offsets[g0]
        );
        slots.idx11 = ensureActiveBlock(
            workspace,
            g1,
            g1,
            r1,
            r1,
            basis.coarse_offsets[g1],
            basis.coarse_offsets[g1]
        );
        workspace.binary_block_slots[factor_idx] = slots;
    }
}

Eigen::SparseMatrix<double> sparseFromActiveBlocks(
    SyntheticSE2PackedCoarseAssemblyWorkspace& workspace
) {
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(workspace.triplet_capacity);
    for (const SyntheticSE2PackedCoarseActiveBlock& block : workspace.active_blocks) {
        for (int r = 0; r < block.rows; ++r) {
            for (int c = 0; c < block.cols; ++c) {
                const double value = block.values(r, c);
                if (value != 0.0) {
                    trips.emplace_back(block.row_offset + r, block.col_offset + c, value);
                }
            }
        }
    }

    Eigen::SparseMatrix<double> coarse_lam(workspace.coarse_dim, workspace.coarse_dim);
    coarse_lam.setFromTriplets(trips.begin(), trips.end());
    coarse_lam.makeCompressed();
    return 0.5 * (coarse_lam + Eigen::SparseMatrix<double>(coarse_lam.transpose()));
}

template <typename VarFn, typename UnaryFn, typename BinaryFn>
void assembleCoarseLambdaWithOpenMpRanges(
    SyntheticSE2PackedCoarseAssemblyWorkspace& assembly_workspace,
    int thread_count,
    int num_vars,
    int num_unaries,
    int num_binaries,
    const VarFn& process_var,
    const UnaryFn& process_unary,
    const BinaryFn& process_binary
) {
    ensureThreadLocalBlockBuffers(assembly_workspace, thread_count);
    const auto var_ranges = buildBalancedIndexRangesPackedCoarse(num_vars, thread_count);
    const auto unary_ranges = buildBalancedIndexRangesPackedCoarse(num_unaries, thread_count);
    const auto binary_ranges = buildBalancedIndexRangesPackedCoarse(num_binaries, thread_count);
    const auto block_ranges = buildBalancedIndexRangesPackedCoarse(
        static_cast<int>(assembly_workspace.active_blocks.size()),
        thread_count
    );

    #pragma omp parallel num_threads(thread_count)
    {
        const int tid = omp_get_thread_num();
        std::vector<double>& buffer = assembly_workspace.thread_local_block_buffers[tid];
        std::fill(buffer.begin(), buffer.end(), 0.0);

        const auto [var_begin, var_end] = var_ranges[static_cast<size_t>(tid)];
        for (int idx = var_begin; idx < var_end; ++idx) {
            process_var(buffer, idx);
        }

        const auto [unary_begin, unary_end] = unary_ranges[static_cast<size_t>(tid)];
        for (int idx = unary_begin; idx < unary_end; ++idx) {
            process_unary(buffer, idx);
        }

        const auto [binary_begin, binary_end] = binary_ranges[static_cast<size_t>(tid)];
        for (int idx = binary_begin; idx < binary_end; ++idx) {
            process_binary(buffer, idx);
        }

        #pragma omp barrier

        const auto [block_begin, block_end] = block_ranges[static_cast<size_t>(tid)];
        for (int block_idx = block_begin; block_idx < block_end; ++block_idx) {
            SyntheticSE2PackedCoarseActiveBlock& block = assembly_workspace.active_blocks[block_idx];
            if (enablePackedStableCoarseLambdaReduce()) {
                reduceActiveBlockStable(
                    block,
                    assembly_workspace.thread_local_block_buffers,
                    thread_count
                );
            } else {
                Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> dest(
                    block.values.data(),
                    block.rows,
                    block.cols
                );
                dest.setZero();
                for (int worker = 0; worker < thread_count; ++worker) {
                    const double* src_ptr = blockBufferPtr(
                        assembly_workspace.thread_local_block_buffers[worker],
                        block
                    );
                    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> src(
                        src_ptr,
                        block.rows,
                        block.cols
                    );
                    dest += src;
                }
            }
        }
    }
}

}  // namespace

SyntheticSE2PackedCoarseAssemblyWorkspace buildSyntheticSE2PackedCoarseAssemblyWorkspace(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis
) {
    SyntheticSE2PackedCoarseAssemblyWorkspace workspace;
    initializeCoarseAssemblyWorkspaceCommon(
        workspace,
        basis,
        static_cast<int>(layout.variables.size()),
        layout.unary_var_ids,
        layout.binary_data
    );
    return workspace;
}

SyntheticSE2PackedCoarseAssemblyWorkspace buildSyntheticSE2PackedCoarseAssemblyWorkspace(
    const SyntheticSE2PackedResidualWorkspace& workspace_data,
    const BasisData& basis
) {
    SyntheticSE2PackedCoarseAssemblyWorkspace workspace;
    std::vector<int> unary_var_ids;
    unary_var_ids.reserve(workspace_data.unary_factors.size());
    for (const SyntheticSE2PackedResidualUnaryFactor& factor : workspace_data.unary_factors) {
        unary_var_ids.push_back(factor.var_id);
    }
    initializeCoarseAssemblyWorkspaceCommon(
        workspace,
        basis,
        workspace_data.num_vars,
        unary_var_ids,
        workspace_data.binary_factors
    );
    return workspace;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectPackedOrdered3D(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    SyntheticSE2PackedCoarseAssemblyWorkspace& assembly_workspace,
    int num_threads
) {
    for (SyntheticSE2PackedCoarseActiveBlock& block : assembly_workspace.active_blocks) {
        block.values.setZero();
    }

    const int thread_count = effectiveThreadCountPackedCoarse(num_threads);
    if (!disablePackedOmpRangeCoarse() &&
        !disablePackedOmpRangeCoarseLambda() &&
        thread_count > 1 &&
        static_cast<int>(layout.binary_data.size()) >= 128) {
        assembleCoarseLambdaWithOpenMpRanges(
            assembly_workspace,
            thread_count,
            static_cast<int>(layout.variables.size()),
            static_cast<int>(layout.unary_var_ids.size()),
            static_cast<int>(layout.binary_data.size()),
            [&](std::vector<double>& buffer, int var_id) {
                const int r_local = basis.var_r_local[var_id];
                const int block_index = assembly_workspace.var_diag_block_slots[var_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const SyntheticSE2PackedCoarseActiveBlock& block = assembly_workspace.active_blocks[block_index];
                accumBasisTransposeLamBasis3x3Sym6(
                    B, r_local, sym6Ptr(layout.prior_lam6, var_id), B, r_local,
                    blockBufferPtr(buffer, block), r_local, 0, 0
                );
            },
            [&](std::vector<double>& buffer, int unary_id) {
                const int var_id = layout.unary_var_ids[unary_id];
                const int r_local = basis.var_r_local[var_id];
                const int block_index = assembly_workspace.unary_diag_block_slots[unary_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const SyntheticSE2PackedCoarseActiveBlock& block = assembly_workspace.active_blocks[block_index];
                accumBasisTransposeLamBasis3x3Sym6(
                    B, r_local, sym6Ptr(layout.unary_lam6, unary_id), B, r_local,
                    blockBufferPtr(buffer, block), r_local, 0, 0
                );
            },
            [&](std::vector<double>& buffer, int factor_idx) {
                const SyntheticSE2PackedBinaryFactorData& factor = layout.binary_data[factor_idx];
                const int id0 = factor.var0_id;
                const int id1 = factor.var1_id;
                const int r0 = basis.var_r_local[id0];
                const int r1 = basis.var_r_local[id1];
                const double* B0 = varBasis3xRPtr(basis, id0);
                const double* B1 = varBasis3xRPtr(basis, id1);
                const SyntheticSE2PackedCoarseBinaryBlockSlots& slots =
                    assembly_workspace.binary_block_slots[factor_idx];

                accumBasisTransposeLamBasis3x3Sym6(
                    B0, r0, factor.diag0_lam6, B0, r0,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx00]), r0, 0, 0
                );
                accumBasisTransposeLamBasis3x3Cross(
                    B0, r0, factor.cross01_lam9, false, B1, r1,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx01]), r0, 0, 0
                );
                accumBasisTransposeLamBasis3x3Cross(
                    B1, r1, factor.cross01_lam9, true, B0, r0,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx10]), r1, 0, 0
                );
                accumBasisTransposeLamBasis3x3Sym6(
                    B1, r1, factor.diag1_lam6, B1, r1,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx11]), r1, 0, 0
                );
            }
        );
        return sparseFromActiveBlocks(assembly_workspace);
    }

    for (int var_id = 0; var_id < static_cast<int>(layout.variables.size()); ++var_id) {
        const int r_local = basis.var_r_local[var_id];
        const int block_index = assembly_workspace.var_diag_block_slots[var_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3Sym6(
            B, r_local, sym6Ptr(layout.prior_lam6, var_id), B, r_local,
            assembly_workspace.active_blocks[block_index].values.data(), r_local, 0, 0
        );
    }

    for (int unary_id = 0; unary_id < static_cast<int>(layout.unary_var_ids.size()); ++unary_id) {
        const int var_id = layout.unary_var_ids[unary_id];
        const int r_local = basis.var_r_local[var_id];
        const int block_index = assembly_workspace.unary_diag_block_slots[unary_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3Sym6(
            B, r_local, sym6Ptr(layout.unary_lam6, unary_id), B, r_local,
            assembly_workspace.active_blocks[block_index].values.data(), r_local, 0, 0
        );
    }

    for (int factor_idx = 0; factor_idx < static_cast<int>(layout.binary_data.size()); ++factor_idx) {
        const SyntheticSE2PackedBinaryFactorData& factor = layout.binary_data[factor_idx];
        const int id0 = factor.var0_id;
        const int id1 = factor.var1_id;
        const int r0 = basis.var_r_local[id0];
        const int r1 = basis.var_r_local[id1];
        const double* B0 = varBasis3xRPtr(basis, id0);
        const double* B1 = varBasis3xRPtr(basis, id1);
        const SyntheticSE2PackedCoarseBinaryBlockSlots& slots =
            assembly_workspace.binary_block_slots[factor_idx];

        accumBasisTransposeLamBasis3x3Sym6(
            B0, r0, factor.diag0_lam6, B0, r0,
            assembly_workspace.active_blocks[slots.idx00].values.data(), r0, 0, 0
        );
        accumBasisTransposeLamBasis3x3Cross(
            B0, r0, factor.cross01_lam9, false, B1, r1,
            assembly_workspace.active_blocks[slots.idx01].values.data(), r0, 0, 0
        );
        accumBasisTransposeLamBasis3x3Cross(
            B1, r1, factor.cross01_lam9, true, B0, r0,
            assembly_workspace.active_blocks[slots.idx10].values.data(), r1, 0, 0
        );
        accumBasisTransposeLamBasis3x3Sym6(
            B1, r1, factor.diag1_lam6, B1, r1,
            assembly_workspace.active_blocks[slots.idx11].values.data(), r1, 0, 0
        );
    }

    return sparseFromActiveBlocks(assembly_workspace);
}

void assembleCoarseResidualDirectPackedOrdered3DInto(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse,
    int num_threads
) {
    if (coarse.size() != basis.coarse_dim) {
        coarse.resize(basis.coarse_dim);
    }
    coarse.setZero();
    const int thread_count = effectiveThreadCountPackedCoarse(num_threads);
    const double* fine_data = fine_vec.data();
    const bool use_parallel = thread_count > 1 &&
        (static_cast<int>(layout.binary_data.size()) >= 128 || static_cast<int>(layout.variables.size()) >= 128);
    if (!use_parallel) {
        double* coarse_data = coarse.data();
        for (int var_id = 0; var_id < static_cast<int>(layout.variables.size()); ++var_id) {
            const int r_local = basis.var_r_local[var_id];
            const int off = basis.var_coarse_offset[var_id];
            const double* B = varBasis3xRPtr(basis, var_id);
            const double* x = fine_data + 3 * var_id;
            const double* eta = vec3Ptr(layout.prior_eta, var_id);
            const double* lam = sym6Ptr(layout.prior_lam6, var_id);
            double residual[3] = {
                eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
            };
            accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
        }

        for (int unary_id = 0; unary_id < static_cast<int>(layout.unary_var_ids.size()); ++unary_id) {
            const int var_id = layout.unary_var_ids[unary_id];
            const int r_local = basis.var_r_local[var_id];
            const int off = basis.var_coarse_offset[var_id];
            const double* B = varBasis3xRPtr(basis, var_id);
            const double* x = fine_data + 3 * var_id;
            const double* eta = vec3Ptr(layout.unary_eta, unary_id);
            const double* lam = sym6Ptr(layout.unary_lam6, unary_id);
            double residual[3] = {
                eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
            };
            accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
        }

        for (const SyntheticSE2PackedBinaryFactorData& factor : layout.binary_data) {
            const int id0 = factor.var0_id;
            const int id1 = factor.var1_id;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* x0 = fine_data + 3 * id0;
            const double* x1 = fine_data + 3 * id1;

            const double x00 = factor.cross01_lam9[0];
            const double x10 = factor.cross01_lam9[1];
            const double x20 = factor.cross01_lam9[2];
            const double x01 = factor.cross01_lam9[3];
            const double x11 = factor.cross01_lam9[4];
            const double x21 = factor.cross01_lam9[5];
            const double x02 = factor.cross01_lam9[6];
            const double x12 = factor.cross01_lam9[7];
            const double x22 = factor.cross01_lam9[8];

            double r0_vec[3] = {
                factor.eta0[0] - (factor.diag0_lam6[0] * x0[0] + factor.diag0_lam6[1] * x0[1] + factor.diag0_lam6[2] * x0[2]
                                + x00 * x1[0] + x01 * x1[1] + x02 * x1[2]),
                factor.eta0[1] - (factor.diag0_lam6[1] * x0[0] + factor.diag0_lam6[3] * x0[1] + factor.diag0_lam6[4] * x0[2]
                                + x10 * x1[0] + x11 * x1[1] + x12 * x1[2]),
                factor.eta0[2] - (factor.diag0_lam6[2] * x0[0] + factor.diag0_lam6[4] * x0[1] + factor.diag0_lam6[5] * x0[2]
                                + x20 * x1[0] + x21 * x1[1] + x22 * x1[2]),
            };
            double r1_vec[3] = {
                factor.eta1[0] - (x00 * x0[0] + x10 * x0[1] + x20 * x0[2]
                                + factor.diag1_lam6[0] * x1[0] + factor.diag1_lam6[1] * x1[1] + factor.diag1_lam6[2] * x1[2]),
                factor.eta1[1] - (x01 * x0[0] + x11 * x0[1] + x21 * x0[2]
                                + factor.diag1_lam6[1] * x1[0] + factor.diag1_lam6[3] * x1[1] + factor.diag1_lam6[4] * x1[2]),
                factor.eta1[2] - (x02 * x0[0] + x12 * x0[1] + x22 * x0[2]
                                + factor.diag1_lam6[2] * x1[0] + factor.diag1_lam6[4] * x1[1] + factor.diag1_lam6[5] * x1[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
            accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
        }
    } else {
        std::vector<Eigen::VectorXd>& coarse_locals =
            reusableCoarseResidualLocals(thread_count, basis.coarse_dim);
        if (disablePackedOmpRangeCoarse() || disablePackedOmpRangeCoarseResidual()) {
            #pragma omp parallel num_threads(thread_count)
            {
                double* coarse_data = coarse_locals[omp_get_thread_num()].data();

                #pragma omp for schedule(static) nowait
                for (int var_id = 0; var_id < static_cast<int>(layout.variables.size()); ++var_id) {
                    const int r_local = basis.var_r_local[var_id];
                    const int off = basis.var_coarse_offset[var_id];
                    const double* B = varBasis3xRPtr(basis, var_id);
                    const double* x = fine_data + 3 * var_id;
                    const double* eta = vec3Ptr(layout.prior_eta, var_id);
                    const double* lam = sym6Ptr(layout.prior_lam6, var_id);
                    double residual[3] = {
                        eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                        eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                        eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                    };
                    accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
                }

                #pragma omp for schedule(static) nowait
                for (int unary_id = 0; unary_id < static_cast<int>(layout.unary_var_ids.size()); ++unary_id) {
                    const int var_id = layout.unary_var_ids[unary_id];
                    const int r_local = basis.var_r_local[var_id];
                    const int off = basis.var_coarse_offset[var_id];
                    const double* B = varBasis3xRPtr(basis, var_id);
                    const double* x = fine_data + 3 * var_id;
                    const double* eta = vec3Ptr(layout.unary_eta, unary_id);
                    const double* lam = sym6Ptr(layout.unary_lam6, unary_id);
                    double residual[3] = {
                        eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                        eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                        eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                    };
                    accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
                }

                #pragma omp for schedule(static)
                for (int factor_idx = 0; factor_idx < static_cast<int>(layout.binary_data.size()); ++factor_idx) {
                    const SyntheticSE2PackedBinaryFactorData& factor = layout.binary_data[factor_idx];
                    const int id0 = factor.var0_id;
                    const int id1 = factor.var1_id;
                    const int r0 = basis.var_r_local[id0];
                    const int r1 = basis.var_r_local[id1];
                    const int off0 = basis.var_coarse_offset[id0];
                    const int off1 = basis.var_coarse_offset[id1];
                    const double* B0 = varBasis3xRPtr(basis, id0);
                    const double* B1 = varBasis3xRPtr(basis, id1);
                    const double* x0 = fine_data + 3 * id0;
                    const double* x1 = fine_data + 3 * id1;

                    const double x00 = factor.cross01_lam9[0];
                    const double x10 = factor.cross01_lam9[1];
                    const double x20 = factor.cross01_lam9[2];
                    const double x01 = factor.cross01_lam9[3];
                    const double x11 = factor.cross01_lam9[4];
                    const double x21 = factor.cross01_lam9[5];
                    const double x02 = factor.cross01_lam9[6];
                    const double x12 = factor.cross01_lam9[7];
                    const double x22 = factor.cross01_lam9[8];

                    double r0_vec[3] = {
                        factor.eta0[0] - (factor.diag0_lam6[0] * x0[0] + factor.diag0_lam6[1] * x0[1] + factor.diag0_lam6[2] * x0[2]
                                        + x00 * x1[0] + x01 * x1[1] + x02 * x1[2]),
                        factor.eta0[1] - (factor.diag0_lam6[1] * x0[0] + factor.diag0_lam6[3] * x0[1] + factor.diag0_lam6[4] * x0[2]
                                        + x10 * x1[0] + x11 * x1[1] + x12 * x1[2]),
                        factor.eta0[2] - (factor.diag0_lam6[2] * x0[0] + factor.diag0_lam6[4] * x0[1] + factor.diag0_lam6[5] * x0[2]
                                        + x20 * x1[0] + x21 * x1[1] + x22 * x1[2]),
                    };
                    double r1_vec[3] = {
                        factor.eta1[0] - (x00 * x0[0] + x10 * x0[1] + x20 * x0[2]
                                        + factor.diag1_lam6[0] * x1[0] + factor.diag1_lam6[1] * x1[1] + factor.diag1_lam6[2] * x1[2]),
                        factor.eta1[1] - (x01 * x0[0] + x11 * x0[1] + x21 * x0[2]
                                        + factor.diag1_lam6[1] * x1[0] + factor.diag1_lam6[3] * x1[1] + factor.diag1_lam6[4] * x1[2]),
                        factor.eta1[2] - (x02 * x0[0] + x12 * x0[1] + x22 * x0[2]
                                        + factor.diag1_lam6[2] * x1[0] + factor.diag1_lam6[4] * x1[1] + factor.diag1_lam6[5] * x1[2]),
                    };
                    accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
                    accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
                }
            }
        } else {
            const auto var_ranges =
                buildBalancedIndexRangesPackedCoarse(static_cast<int>(layout.variables.size()), thread_count);
            const auto unary_ranges =
                buildBalancedIndexRangesPackedCoarse(static_cast<int>(layout.unary_var_ids.size()), thread_count);
            const auto binary_ranges =
                buildBalancedIndexRangesPackedCoarse(static_cast<int>(layout.binary_data.size()), thread_count);
            #pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                double* coarse_data = coarse_locals[tid].data();

                const auto [var_begin, var_end] = var_ranges[static_cast<size_t>(tid)];
                for (int var_id = var_begin; var_id < var_end; ++var_id) {
                    const int r_local = basis.var_r_local[var_id];
                    const int off = basis.var_coarse_offset[var_id];
                    const double* B = varBasis3xRPtr(basis, var_id);
                    const double* x = fine_data + 3 * var_id;
                    const double* eta = vec3Ptr(layout.prior_eta, var_id);
                    const double* lam = sym6Ptr(layout.prior_lam6, var_id);
                    double residual[3] = {
                        eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                        eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                        eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                    };
                    accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
                }

                const auto [unary_begin, unary_end] = unary_ranges[static_cast<size_t>(tid)];
                for (int unary_id = unary_begin; unary_id < unary_end; ++unary_id) {
                    const int var_id = layout.unary_var_ids[unary_id];
                    const int r_local = basis.var_r_local[var_id];
                    const int off = basis.var_coarse_offset[var_id];
                    const double* B = varBasis3xRPtr(basis, var_id);
                    const double* x = fine_data + 3 * var_id;
                    const double* eta = vec3Ptr(layout.unary_eta, unary_id);
                    const double* lam = sym6Ptr(layout.unary_lam6, unary_id);
                    double residual[3] = {
                        eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                        eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                        eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                    };
                    accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
                }

                const auto [binary_begin, binary_end] = binary_ranges[static_cast<size_t>(tid)];
                for (int factor_idx = binary_begin; factor_idx < binary_end; ++factor_idx) {
                    const SyntheticSE2PackedBinaryFactorData& factor = layout.binary_data[factor_idx];
                    const int id0 = factor.var0_id;
                    const int id1 = factor.var1_id;
                    const int r0 = basis.var_r_local[id0];
                    const int r1 = basis.var_r_local[id1];
                    const int off0 = basis.var_coarse_offset[id0];
                    const int off1 = basis.var_coarse_offset[id1];
                    const double* B0 = varBasis3xRPtr(basis, id0);
                    const double* B1 = varBasis3xRPtr(basis, id1);
                    const double* x0 = fine_data + 3 * id0;
                    const double* x1 = fine_data + 3 * id1;

                    const double x00 = factor.cross01_lam9[0];
                    const double x10 = factor.cross01_lam9[1];
                    const double x20 = factor.cross01_lam9[2];
                    const double x01 = factor.cross01_lam9[3];
                    const double x11 = factor.cross01_lam9[4];
                    const double x21 = factor.cross01_lam9[5];
                    const double x02 = factor.cross01_lam9[6];
                    const double x12 = factor.cross01_lam9[7];
                    const double x22 = factor.cross01_lam9[8];

                    double r0_vec[3] = {
                        factor.eta0[0] - (factor.diag0_lam6[0] * x0[0] + factor.diag0_lam6[1] * x0[1] + factor.diag0_lam6[2] * x0[2]
                                        + x00 * x1[0] + x01 * x1[1] + x02 * x1[2]),
                        factor.eta0[1] - (factor.diag0_lam6[1] * x0[0] + factor.diag0_lam6[3] * x0[1] + factor.diag0_lam6[4] * x0[2]
                                        + x10 * x1[0] + x11 * x1[1] + x12 * x1[2]),
                        factor.eta0[2] - (factor.diag0_lam6[2] * x0[0] + factor.diag0_lam6[4] * x0[1] + factor.diag0_lam6[5] * x0[2]
                                        + x20 * x1[0] + x21 * x1[1] + x22 * x1[2]),
                    };
                    double r1_vec[3] = {
                        factor.eta1[0] - (x00 * x0[0] + x10 * x0[1] + x20 * x0[2]
                                        + factor.diag1_lam6[0] * x1[0] + factor.diag1_lam6[1] * x1[1] + factor.diag1_lam6[2] * x1[2]),
                        factor.eta1[1] - (x01 * x0[0] + x11 * x0[1] + x21 * x0[2]
                                        + factor.diag1_lam6[1] * x1[0] + factor.diag1_lam6[3] * x1[1] + factor.diag1_lam6[4] * x1[2]),
                        factor.eta1[2] - (x02 * x0[0] + x12 * x0[1] + x22 * x0[2]
                                        + factor.diag1_lam6[2] * x1[0] + factor.diag1_lam6[4] * x1[1] + factor.diag1_lam6[5] * x1[2]),
                    };
                    accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
                    accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
                }
            }
        }
        reduceCoarseLocals(coarse_locals, coarse);
    }

    return;
}

Eigen::VectorXd assembleCoarseResidualDirectPackedOrdered3D(
    const SyntheticSE2FastSyncLocalPackedLayout& layout,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    int num_threads
) {
    Eigen::VectorXd coarse;
    assembleCoarseResidualDirectPackedOrdered3DInto(
        layout,
        basis,
        fine_vec,
        coarse,
        num_threads
    );
    return coarse;
}

Eigen::SparseMatrix<double> assembleCoarseLambdaDirectPackedOrdered3D(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    SyntheticSE2PackedCoarseAssemblyWorkspace& assembly_workspace,
    int num_threads
) {
    for (SyntheticSE2PackedCoarseActiveBlock& block : assembly_workspace.active_blocks) {
        block.values.setZero();
    }

    const int thread_count = effectiveThreadCountPackedCoarse(num_threads);
    if (!disablePackedOmpRangeCoarse() &&
        !disablePackedOmpRangeCoarseLambda() &&
        thread_count > 1 &&
        static_cast<int>(workspace.binary_factors.size()) >= 128) {
        assembleCoarseLambdaWithOpenMpRanges(
            assembly_workspace,
            thread_count,
            workspace.num_vars,
            static_cast<int>(workspace.unary_factors.size()),
            static_cast<int>(workspace.binary_factors.size()),
            [&](std::vector<double>& buffer, int var_id) {
                const int r_local = basis.var_r_local[var_id];
                const int block_index = assembly_workspace.var_diag_block_slots[var_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const SyntheticSE2PackedCoarseActiveBlock& block = assembly_workspace.active_blocks[block_index];
                accumBasisTransposeLamBasis3x3Sym6(
                    B, r_local, sym6Ptr(workspace.prior_lam6, var_id), B, r_local,
                    blockBufferPtr(buffer, block), r_local, 0, 0
                );
            },
            [&](std::vector<double>& buffer, int unary_id) {
                const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[unary_id];
                const int var_id = factor.var_id;
                const int r_local = basis.var_r_local[var_id];
                const int block_index = assembly_workspace.unary_diag_block_slots[unary_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const SyntheticSE2PackedCoarseActiveBlock& block = assembly_workspace.active_blocks[block_index];
                accumBasisTransposeLamBasis3x3Sym6(
                    B, r_local, factor.lam6, B, r_local,
                    blockBufferPtr(buffer, block), r_local, 0, 0
                );
            },
            [&](std::vector<double>& buffer, int factor_idx) {
                const SyntheticSE2PackedResidualBinaryFactor& factor = workspace.binary_factors[factor_idx];
                const int id0 = factor.var0_id;
                const int id1 = factor.var1_id;
                const int r0 = basis.var_r_local[id0];
                const int r1 = basis.var_r_local[id1];
                const double* B0 = varBasis3xRPtr(basis, id0);
                const double* B1 = varBasis3xRPtr(basis, id1);
                const SyntheticSE2PackedCoarseBinaryBlockSlots& slots =
                    assembly_workspace.binary_block_slots[factor_idx];

                accumBasisTransposeLamBasis3x3Sym6(
                    B0, r0, factor.diag0_lam6, B0, r0,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx00]), r0, 0, 0
                );
                accumBasisTransposeLamBasis3x3Cross(
                    B0, r0, factor.cross01_lam9, false, B1, r1,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx01]), r0, 0, 0
                );
                accumBasisTransposeLamBasis3x3Cross(
                    B1, r1, factor.cross01_lam9, true, B0, r0,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx10]), r1, 0, 0
                );
                accumBasisTransposeLamBasis3x3Sym6(
                    B1, r1, factor.diag1_lam6, B1, r1,
                    blockBufferPtr(buffer, assembly_workspace.active_blocks[slots.idx11]), r1, 0, 0
                );
            }
        );
        return sparseFromActiveBlocks(assembly_workspace);
    }

    for (int var_id = 0; var_id < workspace.num_vars; ++var_id) {
        const int r_local = basis.var_r_local[var_id];
        const int block_index = assembly_workspace.var_diag_block_slots[var_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3Sym6(
            B, r_local, sym6Ptr(workspace.prior_lam6, var_id), B, r_local,
            assembly_workspace.active_blocks[block_index].values.data(), r_local, 0, 0
        );
    }

    for (int unary_id = 0; unary_id < static_cast<int>(workspace.unary_factors.size()); ++unary_id) {
        const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[unary_id];
        const int var_id = factor.var_id;
        const int r_local = basis.var_r_local[var_id];
        const int block_index = assembly_workspace.unary_diag_block_slots[unary_id];
        const double* B = varBasis3xRPtr(basis, var_id);
        accumBasisTransposeLamBasis3x3Sym6(
            B, r_local, factor.lam6, B, r_local,
            assembly_workspace.active_blocks[block_index].values.data(), r_local, 0, 0
        );
    }

    for (int factor_idx = 0; factor_idx < static_cast<int>(workspace.binary_factors.size()); ++factor_idx) {
        const SyntheticSE2PackedResidualBinaryFactor& factor = workspace.binary_factors[factor_idx];
        const int id0 = factor.var0_id;
        const int id1 = factor.var1_id;
        const int r0 = basis.var_r_local[id0];
        const int r1 = basis.var_r_local[id1];
        const double* B0 = varBasis3xRPtr(basis, id0);
        const double* B1 = varBasis3xRPtr(basis, id1);
        const SyntheticSE2PackedCoarseBinaryBlockSlots& slots =
            assembly_workspace.binary_block_slots[factor_idx];

        accumBasisTransposeLamBasis3x3Sym6(
            B0, r0, factor.diag0_lam6, B0, r0,
            assembly_workspace.active_blocks[slots.idx00].values.data(), r0, 0, 0
        );
        accumBasisTransposeLamBasis3x3Cross(
            B0, r0, factor.cross01_lam9, false, B1, r1,
            assembly_workspace.active_blocks[slots.idx01].values.data(), r0, 0, 0
        );
        accumBasisTransposeLamBasis3x3Cross(
            B1, r1, factor.cross01_lam9, true, B0, r0,
            assembly_workspace.active_blocks[slots.idx10].values.data(), r1, 0, 0
        );
        accumBasisTransposeLamBasis3x3Sym6(
            B1, r1, factor.diag1_lam6, B1, r1,
            assembly_workspace.active_blocks[slots.idx11].values.data(), r1, 0, 0
        );
    }

    return sparseFromActiveBlocks(assembly_workspace);
}

void assembleCoarseResidualDirectPackedOrdered3DInto(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    Eigen::VectorXd& coarse,
    int num_threads
) {
    if (coarse.size() != basis.coarse_dim) {
        coarse.resize(basis.coarse_dim);
    }
    coarse.setZero();
    const int thread_count = effectiveThreadCountPackedCoarse(num_threads);
    const double* fine_data = fine_vec.data();
    const bool use_parallel = thread_count > 1 &&
        (static_cast<int>(workspace.binary_factors.size()) >= 128 || workspace.num_vars >= 128);
    if (!use_parallel) {
        double* coarse_data = coarse.data();
        for (int var_id = 0; var_id < workspace.num_vars; ++var_id) {
            const int r_local = basis.var_r_local[var_id];
            const int off = basis.var_coarse_offset[var_id];
            const double* B = varBasis3xRPtr(basis, var_id);
            const double* x = fine_data + 3 * var_id;
            const double* eta = vec3Ptr(workspace.prior_eta, var_id);
            const double* lam = sym6Ptr(workspace.prior_lam6, var_id);
            double residual[3] = {
                eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
            };
            accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
        }

        for (const SyntheticSE2PackedResidualUnaryFactor& factor : workspace.unary_factors) {
            const int var_id = factor.var_id;
            const int r_local = basis.var_r_local[var_id];
            const int off = basis.var_coarse_offset[var_id];
            const double* B = varBasis3xRPtr(basis, var_id);
            const double* x = fine_data + 3 * var_id;
            const double* lam = factor.lam6;
            double residual[3] = {
                factor.eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                factor.eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                factor.eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
            };
            accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
        }

        for (const SyntheticSE2PackedResidualBinaryFactor& factor : workspace.binary_factors) {
            const int id0 = factor.var0_id;
            const int id1 = factor.var1_id;
            const int r0 = basis.var_r_local[id0];
            const int r1 = basis.var_r_local[id1];
            const int off0 = basis.var_coarse_offset[id0];
            const int off1 = basis.var_coarse_offset[id1];
            const double* B0 = varBasis3xRPtr(basis, id0);
            const double* B1 = varBasis3xRPtr(basis, id1);
            const double* x0 = fine_data + 3 * id0;
            const double* x1 = fine_data + 3 * id1;

            const double x00 = factor.cross01_lam9[0];
            const double x10 = factor.cross01_lam9[1];
            const double x20 = factor.cross01_lam9[2];
            const double x01 = factor.cross01_lam9[3];
            const double x11 = factor.cross01_lam9[4];
            const double x21 = factor.cross01_lam9[5];
            const double x02 = factor.cross01_lam9[6];
            const double x12 = factor.cross01_lam9[7];
            const double x22 = factor.cross01_lam9[8];

            double r0_vec[3] = {
                factor.eta0[0] - (factor.diag0_lam6[0] * x0[0] + factor.diag0_lam6[1] * x0[1] + factor.diag0_lam6[2] * x0[2]
                                + x00 * x1[0] + x01 * x1[1] + x02 * x1[2]),
                factor.eta0[1] - (factor.diag0_lam6[1] * x0[0] + factor.diag0_lam6[3] * x0[1] + factor.diag0_lam6[4] * x0[2]
                                + x10 * x1[0] + x11 * x1[1] + x12 * x1[2]),
                factor.eta0[2] - (factor.diag0_lam6[2] * x0[0] + factor.diag0_lam6[4] * x0[1] + factor.diag0_lam6[5] * x0[2]
                                + x20 * x1[0] + x21 * x1[1] + x22 * x1[2]),
            };
            double r1_vec[3] = {
                factor.eta1[0] - (x00 * x0[0] + x10 * x0[1] + x20 * x0[2]
                                + factor.diag1_lam6[0] * x1[0] + factor.diag1_lam6[1] * x1[1] + factor.diag1_lam6[2] * x1[2]),
                factor.eta1[1] - (x01 * x0[0] + x11 * x0[1] + x21 * x0[2]
                                + factor.diag1_lam6[1] * x1[0] + factor.diag1_lam6[3] * x1[1] + factor.diag1_lam6[4] * x1[2]),
                factor.eta1[2] - (x02 * x0[0] + x12 * x0[1] + x22 * x0[2]
                                + factor.diag1_lam6[2] * x1[0] + factor.diag1_lam6[4] * x1[1] + factor.diag1_lam6[5] * x1[2]),
            };
            accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
            accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
        }
    } else {
        auto accumulate_var_range = [&](auto* coarse_data, int begin, int end) {
            for (int var_id = begin; var_id < end; ++var_id) {
                const int r_local = basis.var_r_local[var_id];
                const int off = basis.var_coarse_offset[var_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const double* x = fine_data + 3 * var_id;
                const double* eta = vec3Ptr(workspace.prior_eta, var_id);
                const double* lam = sym6Ptr(workspace.prior_lam6, var_id);
                double residual[3] = {
                    eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                    eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                    eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                };
                accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
            }
        };
        auto accumulate_unary_range = [&](auto* coarse_data, int begin, int end) {
            for (int unary_id = begin; unary_id < end; ++unary_id) {
                const SyntheticSE2PackedResidualUnaryFactor& factor = workspace.unary_factors[unary_id];
                const int var_id = factor.var_id;
                const int r_local = basis.var_r_local[var_id];
                const int off = basis.var_coarse_offset[var_id];
                const double* B = varBasis3xRPtr(basis, var_id);
                const double* x = fine_data + 3 * var_id;
                const double* lam = factor.lam6;
                double residual[3] = {
                    factor.eta[0] - (lam[0] * x[0] + lam[1] * x[1] + lam[2] * x[2]),
                    factor.eta[1] - (lam[1] * x[0] + lam[3] * x[1] + lam[4] * x[2]),
                    factor.eta[2] - (lam[2] * x[0] + lam[4] * x[1] + lam[5] * x[2]),
                };
                accumBasisTransposeTimesVec3(B, r_local, residual, coarse_data + off);
            }
        };
        auto accumulate_binary_range = [&](auto* coarse_data, int begin, int end) {
            for (int factor_idx = begin; factor_idx < end; ++factor_idx) {
                const SyntheticSE2PackedResidualBinaryFactor& factor = workspace.binary_factors[factor_idx];
                const int id0 = factor.var0_id;
                const int id1 = factor.var1_id;
                const int r0 = basis.var_r_local[id0];
                const int r1 = basis.var_r_local[id1];
                const int off0 = basis.var_coarse_offset[id0];
                const int off1 = basis.var_coarse_offset[id1];
                const double* B0 = varBasis3xRPtr(basis, id0);
                const double* B1 = varBasis3xRPtr(basis, id1);
                const double* x0 = fine_data + 3 * id0;
                const double* x1 = fine_data + 3 * id1;

                const double x00 = factor.cross01_lam9[0];
                const double x10 = factor.cross01_lam9[1];
                const double x20 = factor.cross01_lam9[2];
                const double x01 = factor.cross01_lam9[3];
                const double x11 = factor.cross01_lam9[4];
                const double x21 = factor.cross01_lam9[5];
                const double x02 = factor.cross01_lam9[6];
                const double x12 = factor.cross01_lam9[7];
                const double x22 = factor.cross01_lam9[8];

                double r0_vec[3] = {
                    factor.eta0[0] - (factor.diag0_lam6[0] * x0[0] + factor.diag0_lam6[1] * x0[1] + factor.diag0_lam6[2] * x0[2]
                                    + x00 * x1[0] + x01 * x1[1] + x02 * x1[2]),
                    factor.eta0[1] - (factor.diag0_lam6[1] * x0[0] + factor.diag0_lam6[3] * x0[1] + factor.diag0_lam6[4] * x0[2]
                                    + x10 * x1[0] + x11 * x1[1] + x12 * x1[2]),
                    factor.eta0[2] - (factor.diag0_lam6[2] * x0[0] + factor.diag0_lam6[4] * x0[1] + factor.diag0_lam6[5] * x0[2]
                                    + x20 * x1[0] + x21 * x1[1] + x22 * x1[2]),
                };
                double r1_vec[3] = {
                    factor.eta1[0] - (x00 * x0[0] + x10 * x0[1] + x20 * x0[2]
                                    + factor.diag1_lam6[0] * x1[0] + factor.diag1_lam6[1] * x1[1] + factor.diag1_lam6[2] * x1[2]),
                    factor.eta1[1] - (x01 * x0[0] + x11 * x0[1] + x21 * x0[2]
                                    + factor.diag1_lam6[1] * x1[0] + factor.diag1_lam6[3] * x1[1] + factor.diag1_lam6[4] * x1[2]),
                    factor.eta1[2] - (x02 * x0[0] + x12 * x0[1] + x22 * x0[2]
                                    + factor.diag1_lam6[2] * x1[0] + factor.diag1_lam6[4] * x1[1] + factor.diag1_lam6[5] * x1[2]),
                };
                accumBasisTransposeTimesVec3(B0, r0, r0_vec, coarse_data + off0);
                accumBasisTransposeTimesVec3(B1, r1, r1_vec, coarse_data + off1);
            }
        };

        if (enablePackedLongDoubleCoarseResidualLocals()) {
            std::vector<std::vector<long double>> coarse_locals(
                thread_count,
                std::vector<long double>(static_cast<size_t>(basis.coarse_dim), 0.0L)
            );
            const auto var_ranges = buildBalancedIndexRangesPackedCoarse(workspace.num_vars, thread_count);
            const auto unary_ranges = buildBalancedIndexRangesPackedCoarse(
                static_cast<int>(workspace.unary_factors.size()), thread_count
            );
            const auto binary_ranges = buildBalancedIndexRangesPackedCoarse(
                static_cast<int>(workspace.binary_factors.size()), thread_count
            );
            #pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                long double* coarse_data = coarse_locals[tid].data();

                const auto [var_begin, var_end] = var_ranges[static_cast<size_t>(tid)];
                accumulate_var_range(coarse_data, var_begin, var_end);

                const auto [unary_begin, unary_end] = unary_ranges[static_cast<size_t>(tid)];
                accumulate_unary_range(coarse_data, unary_begin, unary_end);

                const auto [binary_begin, binary_end] = binary_ranges[static_cast<size_t>(tid)];
                accumulate_binary_range(coarse_data, binary_begin, binary_end);
            }
            for (int i = 0; i < basis.coarse_dim; ++i) {
                long double acc = 0.0L;
                for (int tid = 0; tid < thread_count; ++tid) {
                    acc += coarse_locals[tid][static_cast<size_t>(i)];
                }
                coarse[i] += static_cast<double>(acc);
            }
            return;
        }

        std::vector<Eigen::VectorXd>& coarse_locals =
            reusableCoarseResidualLocals(thread_count, basis.coarse_dim);
        if (disablePackedOmpRangeCoarse() || disablePackedOmpRangeCoarseResidual()) {
            #pragma omp parallel num_threads(thread_count)
            {
                double* coarse_data = coarse_locals[omp_get_thread_num()].data();

                #pragma omp for schedule(static) nowait
                for (int var_id = 0; var_id < workspace.num_vars; ++var_id) {
                    accumulate_var_range(coarse_data, var_id, var_id + 1);
                }

                #pragma omp for schedule(static) nowait
                for (int unary_id = 0; unary_id < static_cast<int>(workspace.unary_factors.size()); ++unary_id) {
                    accumulate_unary_range(coarse_data, unary_id, unary_id + 1);
                }

                #pragma omp for schedule(static)
                for (int factor_idx = 0; factor_idx < static_cast<int>(workspace.binary_factors.size()); ++factor_idx) {
                    accumulate_binary_range(coarse_data, factor_idx, factor_idx + 1);
                }
            }
        } else {
            const auto var_ranges = buildBalancedIndexRangesPackedCoarse(workspace.num_vars, thread_count);
            const auto unary_ranges = buildBalancedIndexRangesPackedCoarse(
                static_cast<int>(workspace.unary_factors.size()), thread_count
            );
            const auto binary_ranges = buildBalancedIndexRangesPackedCoarse(
                static_cast<int>(workspace.binary_factors.size()), thread_count
            );
            #pragma omp parallel num_threads(thread_count)
            {
                const int tid = omp_get_thread_num();
                double* coarse_data = coarse_locals[tid].data();

                const auto [var_begin, var_end] = var_ranges[static_cast<size_t>(tid)];
                accumulate_var_range(coarse_data, var_begin, var_end);

                const auto [unary_begin, unary_end] = unary_ranges[static_cast<size_t>(tid)];
                accumulate_unary_range(coarse_data, unary_begin, unary_end);

                const auto [binary_begin, binary_end] = binary_ranges[static_cast<size_t>(tid)];
                accumulate_binary_range(coarse_data, binary_begin, binary_end);
            }
        }
        reduceCoarseLocals(coarse_locals, coarse);
    }

    return;
}

Eigen::VectorXd assembleCoarseResidualDirectPackedOrdered3D(
    const SyntheticSE2PackedResidualWorkspace& workspace,
    const BasisData& basis,
    const Eigen::VectorXd& fine_vec,
    int num_threads
) {
    Eigen::VectorXd coarse;
    assembleCoarseResidualDirectPackedOrdered3DInto(
        workspace,
        basis,
        fine_vec,
        coarse,
        num_threads
    );
    return coarse;
}

}  // namespace slam
