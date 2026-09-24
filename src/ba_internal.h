#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {

#if defined(_MSC_VER)
#define GBP_FORCE_INLINE __forceinline
#define GBP_RESTRICT __restrict
#else
#define GBP_FORCE_INLINE inline __attribute__((always_inline))
#define GBP_RESTRICT __restrict__
#endif

using Clock = std::chrono::steady_clock;
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec9 = Eigen::Matrix<double, 9, 1>;
using Mat23 = Eigen::Matrix<double, 2, 3>;
using Mat29 = Eigen::Matrix<double, 2, 9>;
using Mat3 = Eigen::Matrix3d;
using Mat9 = Eigen::Matrix<double, 9, 9>;
using Mat93 = Eigen::Matrix<double, 9, 3>;
using Vec3List = std::vector<Vec3, Eigen::aligned_allocator<Vec3>>;
using Mat3List = std::vector<Mat3, Eigen::aligned_allocator<Mat3>>;
using Mat93List = std::vector<Mat93, Eigen::aligned_allocator<Mat93>>;
using Vec9List = std::vector<Vec9, Eigen::aligned_allocator<Vec9>>;
using Mat9List = std::vector<Mat9, Eigen::aligned_allocator<Mat9>>;

struct BAStepModel {
    double full_decrease = 0.0;
    double linear_decrease = 0.0;

    double decrease(double alpha) const {
        return alpha * linear_decrease - alpha * alpha *
            (linear_decrease - full_decrease);
    }
};

struct Observation {
    int camera = -1;
    int point = -1;
    Vec2 xy = Vec2::Zero();
};

struct BALProblem {
    int num_cameras = 0;
    int num_points = 0;
    int num_observations = 0;
    std::vector<Observation> observations;
    std::vector<Vec9> cameras;
    std::vector<Vec3> points;
    std::vector<std::vector<int>> point_observations;
};

struct Args {
    std::string problem_file;
    std::string out_json = "ba_result.json";
    int outer = 20;
    int mg_cycles = 5;
    int pre_sweeps = 3;
    int gbp_full_sweeps = 32;
    int build_threads = 16;
    int gbp_threads = 16;
    double message_damping = 1.0;
    double coarse_scale = 1.0;
    bool normalize_bal = true;
};

struct PackedGBPStats {
    int accepted_search_directions = 0;
    int true_schur_products = 0;
    double model_decrease = 0.0;
    double variance_defect = std::numeric_limits<double>::quiet_NaN();
    int cycles = 0;
    double group_sec = 0.0;
    double workspace_sec = 0.0;
    double full_sec = 0.0;
    double fixed_map_sec = 0.0;
    double eta_sec = 0.0;
    double mean_sec = 0.0;
    double coarse_residual_sec = 0.0;
    double exact_residual_sec = 0.0;
    double coarse_solve_sec = 0.0;
    double relative_linear_residual =
        std::numeric_limits<double>::quiet_NaN();
    int full_sweeps = 0;
    int eta_sweeps = 0;
    int edges = 0;
    int threads = 1;
    bool reused_workspace = false;
};

struct PackedBlockSchurPattern;

struct PackedBlockSchurPattern {
    int free_cameras = 0;
    int total_cameras = 0;
    bool fix_first_camera = true;
    std::vector<int> camera_to_free;
    std::vector<int> free_to_camera;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<int> diag_slot;
    std::vector<std::vector<int>> point_pair_slots;
    std::vector<std::vector<int>> point_free_cameras;
    std::vector<int> point_edge_offset;
    int total_point_edges = 0;
    std::vector<int> slot_pair_row_ptr;
    std::vector<int> slot_pair_point;
    std::vector<int> slot_pair_edge_a;
    std::vector<int> slot_pair_edge_b;
    std::vector<int> slot_pair_edge_a_global;
    std::vector<int> slot_pair_edge_b_global;
    std::vector<unsigned char> slot_pair_transpose;
    std::vector<int> obs_free_camera;
    std::vector<int> obs_point_edge_index;
    std::unordered_map<unsigned long long, int> slot_of;
    double build_sec = 0.0;
};

double elapsed(const Clock::time_point& a, const Clock::time_point& b) {
    return std::chrono::duration<double>(b - a).count();
}

template <typename T, typename Alloc>
GBP_FORCE_INLINE void zeroPlainEigenStorage(std::vector<T, Alloc>& values) {
    if (!values.empty()) {
        std::memset(values.data(), 0, values.size() * sizeof(T));
    }
}

int effectiveBuildThreads(int requested, int block_nnz, int free_cameras) {
    int threads = std::max(1, requested);
#if defined(_OPENMP)
    threads = std::min(threads, std::max(1, omp_get_max_threads()));
#else
    threads = 1;
#endif
    if (threads <= 1) {
        return 1;
    }
    // The packed BA builder uses per-thread diagonal/RHS accumulators plus a
    // slot-owned Schur assembly, so 16 workers no longer replicate all blocks.
    constexpr int kMaxPackedSchurBuildThreads = 16;
    threads = std::min(threads, kMaxPackedSchurBuildThreads);
    const size_t per_thread_bytes =
        static_cast<size_t>(std::max(1, free_cameras)) * (sizeof(Mat9) + sizeof(Vec9));
    constexpr size_t kMaxThreadLocalBytes = static_cast<size_t>(1536) * 1024 * 1024;
    if (per_thread_bytes > 0) {
        const int memory_limited_threads =
            std::max(1, static_cast<int>(kMaxThreadLocalBytes / per_thread_bytes));
        threads = std::min(threads, memory_limited_threads);
    }
    return std::max(1, threads);
}

int effectiveGBPThreads(int requested, int edges, int variables) {
    int threads = std::max(1, requested);
#if defined(_OPENMP)
    threads = std::min(threads, std::max(1, omp_get_max_threads()));
#else
    threads = 1;
#endif
    if (threads <= 1) {
        return 1;
    }
    const int work_items = std::max(edges, variables);
    if (work_items < 512) {
        return 1;
    }
    return std::max(1, std::min(threads, std::max(1, work_items / 256)));
}

int requestedGBPThreads(const Args& args) {
    return args.gbp_threads > 0 ? args.gbp_threads : args.build_threads;
}

void configureOpenMPWaitPolicy(const Args& args) {
    if (args.build_threads <= 1) {
        return;
    }
#if defined(_OPENMP)
#if defined(_WIN32)
    if (std::getenv("OMP_WAIT_POLICY") == nullptr) {
        _putenv_s("OMP_WAIT_POLICY", "PASSIVE");
    }
    if (std::getenv("KMP_BLOCKTIME") == nullptr) {
        _putenv_s("KMP_BLOCKTIME", "0");
    }
#else
    if (std::getenv("OMP_WAIT_POLICY") == nullptr) {
        setenv("OMP_WAIT_POLICY", "PASSIVE", 0);
    }
    if (std::getenv("KMP_BLOCKTIME") == nullptr) {
        setenv("KMP_BLOCKTIME", "0", 0);
    }
#endif
#else
    (void)args;
#endif
}

unsigned long long blockKey(int r, int c) {
    return (static_cast<unsigned long long>(static_cast<unsigned int>(r)) << 32) |
           static_cast<unsigned int>(c);
}

void parseInt(const char* raw, int& out) {
    size_t consumed = 0;
    const std::string value(raw);
    const int parsed = std::stoi(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("Invalid integer: " + value);
    out = parsed;
}

void parseDouble(const char* raw, double& out) {
    size_t consumed = 0;
    const std::string value(raw);
    const double parsed = std::stod(value, &consumed);
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("Invalid finite number: " + value);
    }
    out = parsed;
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--problem-file" && i + 1 < argc) {
            args.problem_file = argv[++i];
        } else if (arg == "--out-json" && i + 1 < argc) {
            args.out_json = argv[++i];
        } else if (arg == "--outer" && i + 1 < argc) {
            parseInt(argv[++i], args.outer);
        } else if (arg == "--mg-cycles" && i + 1 < argc) {
            parseInt(argv[++i], args.mg_cycles);
        } else if (arg == "--pre-sweeps" && i + 1 < argc) {
            parseInt(argv[++i], args.pre_sweeps);
        } else if (arg == "--gbp-full-sweeps" && i + 1 < argc) {
            parseInt(argv[++i], args.gbp_full_sweeps);
        } else if (arg == "--build-threads" && i + 1 < argc) {
            parseInt(argv[++i], args.build_threads);
        } else if (arg == "--gbp-threads" && i + 1 < argc) {
            parseInt(argv[++i], args.gbp_threads);
        } else if (arg == "--message-damping" && i + 1 < argc) {
            parseDouble(argv[++i], args.message_damping);
        } else if (arg == "--coarse-scale" && i + 1 < argc) {
            parseDouble(argv[++i], args.coarse_scale);
        } else if (arg == "--normalize-bal") {
            args.normalize_bal = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ba_solver --problem-file <BAL.txt> [options]\n"
                << "  --out-json <result.json>\n"
                << "  --outer <20> --mg-cycles <5> --pre-sweeps <3>\n"
                << "  --gbp-full-sweeps <32> --normalize-bal\n"
                << "  --build-threads <16> --gbp-threads <16>\n"
                << "  --message-damping <1.0> --coarse-scale 1\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (args.problem_file.empty()) {
        throw std::runtime_error("--problem-file is required");
    }
    if (args.outer < 1 || args.mg_cycles < 1 || args.pre_sweeps < 1 ||
        args.gbp_full_sweeps < 1 || args.build_threads < 1 || args.gbp_threads < 1) {
        throw std::runtime_error("Iteration and thread budgets must be positive");
    }
    if (args.message_damping <= 0.0 || args.message_damping > 1.0) {
        throw std::runtime_error("message-damping must be in (0,1]");
    }
    if (args.coarse_scale != 1.0) {
        throw std::runtime_error("--coarse-scale must be 1 for additive H-GBP");
    }
    return args;
}

BALProblem readBAL(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to open BAL file: " + path);
    }

    BALProblem problem;
    in >> problem.num_cameras >> problem.num_points >> problem.num_observations;
    if (!in || problem.num_cameras <= 0 || problem.num_points <= 0 || problem.num_observations <= 0) {
        throw std::runtime_error("invalid BAL header: " + path);
    }

    problem.observations.resize(problem.num_observations);
    problem.point_observations.assign(problem.num_points, {});
    for (int i = 0; i < problem.num_observations; ++i) {
        Observation obs;
        in >> obs.camera >> obs.point >> obs.xy.x() >> obs.xy.y();
        if (!in || obs.camera < 0 || obs.camera >= problem.num_cameras ||
            obs.point < 0 || obs.point >= problem.num_points) {
            throw std::runtime_error("invalid BAL observation row");
        }
        problem.observations[i] = obs;
        problem.point_observations[obs.point].push_back(i);
    }

    problem.cameras.resize(problem.num_cameras);
    for (int c = 0; c < problem.num_cameras; ++c) {
        for (int k = 0; k < 9; ++k) {
            in >> problem.cameras[c][k];
        }
    }

    problem.points.resize(problem.num_points);
    for (int p = 0; p < problem.num_points; ++p) {
        in >> problem.points[p].x() >> problem.points[p].y() >> problem.points[p].z();
    }
    if (!in) {
        throw std::runtime_error("truncated BAL file: " + path);
    }
    return problem;
}

Eigen::Matrix3d skew(const Vec3& w) {
    Eigen::Matrix3d W;
    W << 0.0, -w.z(), w.y(),
         w.z(), 0.0, -w.x(),
        -w.y(), w.x(), 0.0;
    return W;
}

Mat3 rodriguesMatrix(const Vec3& w) {
    const double theta2 = w.squaredNorm();
    const Mat3 W = skew(w);
    if (theta2 < 1e-12) {
        return Mat3::Identity() + W + 0.5 * W * W;
    }
    const double theta = std::sqrt(theta2);
    const double A = std::sin(theta) / theta;
    const double B = (1.0 - std::cos(theta)) / theta2;
    return Mat3::Identity() + A * W + B * W * W;
}

double medianOf(std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + mid, values.end());
    double med = values[mid];
    if ((values.size() & 1u) == 0u) {
        std::nth_element(values.begin(), values.begin() + mid - 1, values.end());
        med = 0.5 * (med + values[mid - 1]);
    }
    return med;
}

void normalizeBALProblem(BALProblem& problem) {
    std::vector<double> xs(problem.points.size());
    std::vector<double> ys(problem.points.size());
    std::vector<double> zs(problem.points.size());
    for (size_t i = 0; i < problem.points.size(); ++i) {
        xs[i] = problem.points[i].x();
        ys[i] = problem.points[i].y();
        zs[i] = problem.points[i].z();
    }
    const Vec3 center(medianOf(xs), medianOf(ys), medianOf(zs));

    std::vector<double> distances(problem.points.size());
    for (size_t i = 0; i < problem.points.size(); ++i) {
        distances[i] = (problem.points[i] - center).norm();
    }
    const double mad = medianOf(distances);
    if (!(mad > 0.0) || !std::isfinite(mad)) {
        return;
    }
    const double scale = 100.0 / mad;

    for (Vec3& point : problem.points) {
        point = scale * (point - center);
    }
    for (Vec9& camera : problem.cameras) {
        const Mat3 R = rodriguesMatrix(camera.segment<3>(0));
        const Vec3 t = camera.segment<3>(3);
        const Vec3 camera_center = -R.transpose() * t;
        const Vec3 normalized_center = scale * (camera_center - center);
        camera.segment<3>(3) = -R * normalized_center;
    }
}

GBP_FORCE_INLINE void mat9VecRawAddInto(
    const Mat9& A,
    const double* GBP_RESTRICT x,
    double* GBP_RESTRICT y
) {
    const double* GBP_RESTRICT a = A.data();
    for (int r = 0; r < 9; ++r) {
        double sum = 0.0;
        sum += a[0 * 9 + r] * x[0];
        sum += a[1 * 9 + r] * x[1];
        sum += a[2 * 9 + r] * x[2];
        sum += a[3 * 9 + r] * x[3];
        sum += a[4 * 9 + r] * x[4];
        sum += a[5 * 9 + r] * x[5];
        sum += a[6 * 9 + r] * x[6];
        sum += a[7 * 9 + r] * x[7];
        sum += a[8 * 9 + r] * x[8];
        y[r] += sum;
    }
}


struct SchurGBPEdge {
    int i = -1;
    int j = -1;
    int msg_to_i = -1;
    int msg_to_j = -1;
    int slot_ij = -1;
    int slot_ji = -1;
    Mat9 aij = Mat9::Zero();
    Mat9 aji = Mat9::Zero();
    Mat9 factor_lam_i = Mat9::Zero();
    Mat9 factor_lam_j = Mat9::Zero();
    double factor_scale = 1.0;
};

struct SchurGBPWorkspace {
    int n = 0;
    Mat9List unary_lam;
    Mat9List preconditioner_lam;
    Vec9List unary_eta;
    Mat9List belief_lam;
    Vec9List belief_eta;
    Mat9List next_belief_lam;
    Vec9List next_belief_eta;
    Mat9List msg_lam;
    Vec9List msg_eta;
    Mat9List next_msg_lam;
    Vec9List next_msg_eta;
    Mat9List fixed_eta_map;
    Mat9List belief_lam_inv;
    bool belief_lam_inv_valid = false;
    std::vector<SchurGBPEdge> edges;
    std::vector<std::vector<int>> incoming;
    std::vector<int> incoming_offsets;
    std::vector<int> incoming_msg_ids;
    std::vector<int> fixed_eta_pair_i;
    std::vector<int> fixed_eta_pair_j;
    std::vector<int> fixed_eta_pair_msg_to_i;
    std::vector<int> fixed_eta_pair_msg_to_j;
    int fixed_eta_pair_edge_count = 0;
};

GBP_FORCE_INLINE const Mat9& schurGBPPreconditionerLambda(
    const SchurGBPWorkspace& ws,
    int variable
) {
    return ws.preconditioner_lam[static_cast<size_t>(variable)];
}

struct FastHGBPBuildThreadAccum {
    Mat9List diag_blocks;
    Mat9List row_blocks;
    Vec9List rhs_blocks;
    double linearized_cost = 0.0;
};

struct FastHGBPSystem {
    const PackedBlockSchurPattern* pattern = nullptr;
    int free_cameras = 0;
    int total_cameras = 0;
    bool fix_first_camera = true;
    SchurGBPWorkspace ws;
    Mat93List edge_E;
    Vec9List rhs_blocks;
    std::vector<int> row_edge_offsets;
    std::vector<int> row_edge_ids;
    std::vector<int> row_block_offsets;
    std::vector<int> row_block_cols;
    std::vector<int> row_block_edge_ids;
    std::vector<int> edge_row_pos_i;
    std::vector<int> edge_row_pos_j;
    std::vector<int> slot_to_row_block_pos;
    std::vector<unsigned char> row_block_forward;
    Mat9List row_blocks;
    double linearized_cost = 0.0;
    double build_sec = 0.0;
    double matrix_fingerprint = 0.0;
    double rhs_fingerprint = 0.0;
    int build_threads_actual = 1;
    std::vector<FastHGBPBuildThreadAccum> build_accum;
};

GBP_FORCE_INLINE double schurGBPPreconditionerEdgeScale(
    const FastHGBPSystem& sys,
    int row_block_position
) {
    const int edge = sys.row_block_edge_ids[
        static_cast<size_t>(row_block_position)];
    return sys.ws.edges[static_cast<size_t>(edge)].factor_scale;
}

struct FastHGBPCoarseWorkspace {
    int free_cameras = 0;
    int groups = 0;
    std::vector<int> gid;
    std::vector<double> scale;
    std::vector<int> ap_row_ptr;
    std::vector<int> ap_group_idx;
    Mat9List ap_blocks;
    Eigen::MatrixXd Ac;
    Eigen::LDLT<Eigen::MatrixXd> ldlt;
    Eigen::VectorXd bc;
    Eigen::VectorXd z;
    Eigen::VectorXd correction;
    double ac_scale = 1.0;
};

void finalizeSchurGBPIncomingCSR(SchurGBPWorkspace& ws) {
    ws.incoming_offsets.assign(static_cast<size_t>(ws.n) + 1, 0);
    int total = 0;
    for (int i = 0; i < ws.n; ++i) {
        total += static_cast<int>(ws.incoming[static_cast<size_t>(i)].size());
        ws.incoming_offsets[static_cast<size_t>(i + 1)] = total;
    }
    ws.incoming_msg_ids.assign(static_cast<size_t>(total), 0);
    for (int i = 0; i < ws.n; ++i) {
        const int begin = ws.incoming_offsets[static_cast<size_t>(i)];
        const auto& ids = ws.incoming[static_cast<size_t>(i)];
        for (int k = 0; k < static_cast<int>(ids.size()); ++k) {
            ws.incoming_msg_ids[static_cast<size_t>(begin + k)] = ids[static_cast<size_t>(k)];
        }
    }
}

GBP_FORCE_INLINE void dampedMat9VecDiffRawInto(
    const Mat9& A,
    const Vec9& belief,
    const Vec9& cavity_msg,
    const Vec9& old_msg,
    double keep,
    double omega,
    Vec9& out
) {
    const double* a = A.data();
    const double* bv = belief.data();
    const double* cv = cavity_msg.data();
    const double* ov = old_msg.data();
    double* y = out.data();
    const double d0 = bv[0] - cv[0];
    const double d1 = bv[1] - cv[1];
    const double d2 = bv[2] - cv[2];
    const double d3 = bv[3] - cv[3];
    const double d4 = bv[4] - cv[4];
    const double d5 = bv[5] - cv[5];
    const double d6 = bv[6] - cv[6];
    const double d7 = bv[7] - cv[7];
    const double d8 = bv[8] - cv[8];
    for (int r = 0; r < 9; ++r) {
        double sum = 0.0;
        sum += a[0 * 9 + r] * d0;
        sum += a[1 * 9 + r] * d1;
        sum += a[2 * 9 + r] * d2;
        sum += a[3 * 9 + r] * d3;
        sum += a[4 * 9 + r] * d4;
        sum += a[5 * 9 + r] * d5;
        sum += a[6 * 9 + r] * d6;
        sum += a[7 * 9 + r] * d7;
        sum += a[8 * 9 + r] * d8;
        y[r] = keep * ov[r] + omega * sum;
    }
}

GBP_FORCE_INLINE void mat9VecRawInto(const Mat9& A, const Vec9& x, Vec9& out) {
    const double* a = A.data();
    const double* xv = x.data();
    double* y = out.data();
    for (int r = 0; r < 9; ++r) {
        double sum = 0.0;
        sum += a[0 * 9 + r] * xv[0];
        sum += a[1 * 9 + r] * xv[1];
        sum += a[2 * 9 + r] * xv[2];
        sum += a[3 * 9 + r] * xv[3];
        sum += a[4 * 9 + r] * xv[4];
        sum += a[5 * 9 + r] * xv[5];
        sum += a[6 * 9 + r] * xv[6];
        sum += a[7 * 9 + r] * xv[7];
        sum += a[8 * 9 + r] * xv[8];
        y[r] = sum;
    }
}

GBP_FORCE_INLINE double diagonalAbsMax9(const Mat9& A) {
    double m = 1.0;
    for (int i = 0; i < 9; ++i) {
        m = std::max(m, std::abs(A(i, i)));
    }
    return m;
}

GBP_FORCE_INLINE bool solveSpd9Mat9VecCholeskyRaw(
    const Mat9& A,
    const Mat9& rhs_mat,
    const Vec9& rhs_vec,
    Mat9& out_mat,
    Vec9& out_vec
) {
    double L[81] = {};
    const double* A_data = A.data();
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A_data[j * 9 + i];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * 9 + k] * L[j * 9 + k];
            }
            if (i == j) {
                if (!(sum > 0.0) || !std::isfinite(sum)) {
                    return false;
                }
                L[i * 9 + j] = std::sqrt(sum);
            } else {
                L[i * 9 + j] = sum / L[j * 9 + j];
            }
        }
    }

    double X[90];
    const double* rhs_mat_data = rhs_mat.data();
    const double* rhs_vec_data = rhs_vec.data();
    for (int i = 0; i < 9; ++i) {
        for (int c = 0; c < 9; ++c) {
            X[i * 10 + c] = rhs_mat_data[c * 9 + i];
        }
        X[i * 10 + 9] = rhs_vec_data[i];
    }
    for (int i = 0; i < 9; ++i) {
        const double inv_diag = 1.0 / L[i * 9 + i];
        for (int c = 0; c < 10; ++c) {
            double sum = X[i * 10 + c];
            for (int k = 0; k < i; ++k) {
                sum -= L[i * 9 + k] * X[k * 10 + c];
            }
            X[i * 10 + c] = sum * inv_diag;
        }
    }
    for (int i = 8; i >= 0; --i) {
        const double inv_diag = 1.0 / L[i * 9 + i];
        for (int c = 0; c < 10; ++c) {
            double sum = X[i * 10 + c];
            for (int k = i + 1; k < 9; ++k) {
                sum -= L[k * 9 + i] * X[k * 10 + c];
            }
            X[i * 10 + c] = sum * inv_diag;
        }
    }
    double* out_mat_data = out_mat.data();
    double* out_vec_data = out_vec.data();
    for (int i = 0; i < 9; ++i) {
        for (int c = 0; c < 9; ++c) {
            out_mat_data[c * 9 + i] = X[i * 10 + c];
        }
        out_vec_data[i] = X[i * 10 + 9];
    }
    return true;
}

bool solveRegularizedSpd9(
    const Mat9& input,
    const Mat9& rhs_mat,
    const Vec9& rhs_vec,
    Mat9& out_mat,
    Vec9& out_vec
) {
    Mat9 A = 0.5 * (input + input.transpose());
    const double base = diagonalAbsMax9(A);
    double jitter = 1e-12 * base;
    for (int attempt = 0; attempt < 8; ++attempt) {
        Mat9 test = A;
        test.diagonal().array() += jitter;
        if (solveSpd9Mat9VecCholeskyRaw(test, rhs_mat, rhs_vec, out_mat, out_vec)) {
            return true;
        }
        jitter *= 10.0;
    }
    A.diagonal().array() += jitter;
    out_mat.noalias() = A.ldlt().solve(rhs_mat);
    out_vec.noalias() = A.ldlt().solve(rhs_vec);
    return out_mat.allFinite() && out_vec.allFinite();
}

bool solveRegularizedSpd9Vec(const Mat9& input, const Vec9& rhs_vec, Vec9& out_vec) {
    Mat9 dummy_rhs = Mat9::Zero();
    Mat9 dummy_out = Mat9::Zero();
    return solveRegularizedSpd9(input, dummy_rhs, rhs_vec, dummy_out, out_vec);
}



void initializeFastHGBPSystemStructure(FastHGBPSystem& sys, const PackedBlockSchurPattern& pattern) {
    const bool structure_ok =
        sys.pattern == &pattern &&
        sys.ws.n == pattern.free_cameras &&
        sys.ws.unary_lam.size() == static_cast<size_t>(pattern.free_cameras) &&
        sys.row_edge_offsets.size() == static_cast<size_t>(pattern.free_cameras + 1) &&
        sys.row_block_offsets.size() == static_cast<size_t>(pattern.free_cameras + 1) &&
        sys.row_blocks.size() == sys.row_block_cols.size();
    sys.pattern = &pattern;
    sys.free_cameras = pattern.free_cameras;
    sys.total_cameras = pattern.total_cameras;
    sys.fix_first_camera = pattern.fix_first_camera;
    if (structure_ok) {
        return;
    }

    SchurGBPWorkspace& ws = sys.ws;
    ws = SchurGBPWorkspace{};
    ws.n = pattern.free_cameras;
    ws.unary_lam.assign(ws.n, Mat9::Zero());
    ws.preconditioner_lam.assign(ws.n, Mat9::Zero());
    ws.unary_eta.assign(ws.n, Vec9::Zero());
    ws.belief_lam.assign(ws.n, Mat9::Zero());
    ws.belief_eta.assign(ws.n, Vec9::Zero());
    ws.next_belief_lam.assign(ws.n, Mat9::Zero());
    ws.next_belief_eta.assign(ws.n, Vec9::Zero());
    ws.belief_lam_inv.assign(ws.n, Mat9::Zero());
    ws.belief_lam_inv_valid = false;
    ws.incoming.assign(ws.n, {});
    for (int i = 0; i < ws.n; ++i) {
        for (int slot = pattern.row_ptr[static_cast<size_t>(i)];
             slot < pattern.row_ptr[static_cast<size_t>(i + 1)];
             ++slot) {
            const int j = pattern.col_idx[static_cast<size_t>(slot)];
            if (j <= i) {
                continue;
            }
            SchurGBPEdge edge;
            edge.i = i;
            edge.j = j;
            edge.slot_ij = slot;
            const auto back_it = pattern.slot_of.find(blockKey(j, i));
            if (back_it != pattern.slot_of.end()) {
                edge.slot_ji = back_it->second;
            }
            edge.msg_to_i = static_cast<int>(ws.msg_lam.size());
            ws.msg_lam.push_back(Mat9::Zero());
            ws.msg_eta.push_back(Vec9::Zero());
            ws.next_msg_lam.push_back(Mat9::Zero());
            ws.next_msg_eta.push_back(Vec9::Zero());
            ws.fixed_eta_map.push_back(Mat9::Zero());
            edge.msg_to_j = static_cast<int>(ws.msg_lam.size());
            ws.msg_lam.push_back(Mat9::Zero());
            ws.msg_eta.push_back(Vec9::Zero());
            ws.next_msg_lam.push_back(Mat9::Zero());
            ws.next_msg_eta.push_back(Vec9::Zero());
            ws.fixed_eta_map.push_back(Mat9::Zero());
            ws.incoming[static_cast<size_t>(i)].push_back(edge.msg_to_i);
            ws.incoming[static_cast<size_t>(j)].push_back(edge.msg_to_j);
            ws.edges.push_back(edge);
        }
    }
    finalizeSchurGBPIncomingCSR(ws);

    sys.row_edge_offsets.assign(static_cast<size_t>(ws.n) + 1, 0);
    for (const SchurGBPEdge& edge : ws.edges) {
        ++sys.row_edge_offsets[static_cast<size_t>(edge.i) + 1];
        ++sys.row_edge_offsets[static_cast<size_t>(edge.j) + 1];
    }
    for (int i = 0; i < ws.n; ++i) {
        sys.row_edge_offsets[static_cast<size_t>(i + 1)] +=
            sys.row_edge_offsets[static_cast<size_t>(i)];
    }
    sys.row_edge_ids.assign(static_cast<size_t>(sys.row_edge_offsets.back()), 0);
    std::vector<int> cursor = sys.row_edge_offsets;
    for (int eidx = 0; eidx < static_cast<int>(ws.edges.size()); ++eidx) {
        const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(eidx)];
        sys.row_edge_ids[static_cast<size_t>(cursor[static_cast<size_t>(edge.i)]++)] = eidx;
        sys.row_edge_ids[static_cast<size_t>(cursor[static_cast<size_t>(edge.j)]++)] = eidx;
    }

    sys.row_block_offsets.assign(static_cast<size_t>(ws.n) + 1, 0);
    for (const SchurGBPEdge& edge : ws.edges) {
        ++sys.row_block_offsets[static_cast<size_t>(edge.i) + 1];
        ++sys.row_block_offsets[static_cast<size_t>(edge.j) + 1];
    }
    for (int i = 0; i < ws.n; ++i) {
        sys.row_block_offsets[static_cast<size_t>(i + 1)] +=
            sys.row_block_offsets[static_cast<size_t>(i)];
    }
    const int row_block_nnz = sys.row_block_offsets.back();
    sys.row_block_cols.assign(static_cast<size_t>(row_block_nnz), 0);
    sys.row_block_edge_ids.assign(static_cast<size_t>(row_block_nnz), 0);
    sys.edge_row_pos_i.assign(ws.edges.size(), -1);
    sys.edge_row_pos_j.assign(ws.edges.size(), -1);
    sys.slot_to_row_block_pos.assign(pattern.col_idx.size(), -1);
    sys.row_block_forward.assign(static_cast<size_t>(row_block_nnz), 0);
    sys.row_blocks.assign(static_cast<size_t>(row_block_nnz), Mat9::Zero());
    cursor = sys.row_block_offsets;
    for (int eidx = 0; eidx < static_cast<int>(ws.edges.size()); ++eidx) {
        const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(eidx)];
        int pos = cursor[static_cast<size_t>(edge.i)]++;
        sys.row_block_cols[static_cast<size_t>(pos)] = edge.j;
        sys.row_block_edge_ids[static_cast<size_t>(pos)] = eidx;
        sys.edge_row_pos_i[static_cast<size_t>(eidx)] = pos;
        sys.slot_to_row_block_pos[static_cast<size_t>(edge.slot_ij)] = pos;
        sys.row_block_forward[static_cast<size_t>(pos)] = 1;
        pos = cursor[static_cast<size_t>(edge.j)]++;
        sys.row_block_cols[static_cast<size_t>(pos)] = edge.i;
        sys.row_block_edge_ids[static_cast<size_t>(pos)] = eidx;
        sys.edge_row_pos_j[static_cast<size_t>(eidx)] = pos;
        if (edge.slot_ji >= 0) {
            sys.slot_to_row_block_pos[static_cast<size_t>(edge.slot_ji)] = pos;
        }
        sys.row_block_forward[static_cast<size_t>(pos)] = 0;
    }
}

void resetSchurGBPEtaMessages(SchurGBPWorkspace& ws, const Eigen::VectorXd& unary_eta, int threads = 1) {
    const double* src = unary_eta.data();
    const int parallel_threads = effectiveGBPThreads(threads, ws.n, ws.n);
    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static)
        for (int i = 0; i < ws.n; ++i) {
            const double* eta_src = src + 9 * static_cast<size_t>(i);
            double* unary_dst = ws.unary_eta[static_cast<size_t>(i)].data();
            double* belief_dst = ws.belief_eta[static_cast<size_t>(i)].data();
            double* next_dst = ws.next_belief_eta[static_cast<size_t>(i)].data();
            for (int k = 0; k < 9; ++k) {
                const double v = eta_src[k];
                unary_dst[k] = v;
                belief_dst[k] = v;
                next_dst[k] = v;
            }
        }
        zeroPlainEigenStorage(ws.msg_eta);
        zeroPlainEigenStorage(ws.next_msg_eta);
        return;
#endif
    }
    for (int i = 0; i < ws.n; ++i) {
        const double* eta_src = src + 9 * static_cast<size_t>(i);
        double* unary_dst = ws.unary_eta[static_cast<size_t>(i)].data();
        double* belief_dst = ws.belief_eta[static_cast<size_t>(i)].data();
        double* next_dst = ws.next_belief_eta[static_cast<size_t>(i)].data();
        for (int k = 0; k < 9; ++k) {
            const double v = eta_src[k];
            unary_dst[k] = v;
            belief_dst[k] = v;
            next_dst[k] = v;
        }
    }
    zeroPlainEigenStorage(ws.msg_eta);
    zeroPlainEigenStorage(ws.next_msg_eta);
}

void fastHGBPRhsVectorInto(const FastHGBPSystem& sys, Eigen::VectorXd& rhs) {
    const int dim = 9 * sys.free_cameras;
    if (rhs.size() != dim) {
        rhs.resize(dim);
    }
    for (int i = 0; i < sys.free_cameras; ++i) {
        rhs.segment<9>(9 * i) = sys.rhs_blocks[static_cast<size_t>(i)];
    }
}

void fastHGBPMultiplyInto(
    const FastHGBPSystem& sys,
    const Eigen::VectorXd& x,
    Eigen::VectorXd& y,
    int threads
) {
    const int dim = 9 * sys.free_cameras;
    if (y.size() != dim) {
        y.resize(dim);
    }
    const int edge_count = static_cast<int>(sys.ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(threads, edge_count, sys.free_cameras);
    const double* GBP_RESTRICT xd = x.data();
    double* GBP_RESTRICT yd = y.data();
    auto multiply_row = [&](int r) {
        double yr[9] = {};
        mat9VecRawAddInto(
            schurGBPPreconditionerLambda(sys.ws, r),
            xd + 9 * static_cast<size_t>(r),
            yr);
        const int begin = sys.row_block_offsets[static_cast<size_t>(r)];
        const int end = sys.row_block_offsets[static_cast<size_t>(r + 1)];
        for (int pos = begin; pos < end; ++pos) {
            const Mat9 block = schurGBPPreconditionerEdgeScale(sys, pos) *
                sys.row_blocks[static_cast<size_t>(pos)];
            mat9VecRawAddInto(
                block,
                xd + 9 * static_cast<size_t>(sys.row_block_cols[static_cast<size_t>(pos)]),
                yr);
        }
        double* GBP_RESTRICT out = yd + 9 * static_cast<size_t>(r);
        for (int k = 0; k < 9; ++k) {
            out[k] = yr[k];
        }
    };
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static) if(parallel_threads > 1)
#endif
    for (int r = 0; r < sys.free_cameras; ++r) {
        multiply_row(r);
    }
}

double schurCutRatio(const FastHGBPSystem& sys, const std::vector<std::vector<int>>& groups) {
    std::vector<int> gid(sys.free_cameras, -1);
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
        for (int v : groups[static_cast<size_t>(g)]) {
            gid[static_cast<size_t>(v)] = g;
        }
    }
    double cut = 0.0;
    double total = 0.0;
    for (const SchurGBPEdge& edge : sys.ws.edges) {
        const double e = edge.aij.norm() + edge.aji.norm();
        if (e == 0.0) continue;
        total += e;
        if (gid[static_cast<size_t>(edge.i)] != gid[static_cast<size_t>(edge.j)]) {
            cut += e;
        }
    }
    return total > 0.0 ? cut / total : 0.0;
}


void schurGBPSweep(SchurGBPWorkspace& ws, double damping, int threads = 1) {
    const double omega = std::min(1.0, std::max(0.0, damping));
    const double keep = 1.0 - omega;
    const int edge_count = static_cast<int>(ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(threads, edge_count, ws.n);
    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads)
        {
#pragma omp for schedule(static)
            for (int eidx = 0; eidx < edge_count; ++eidx) {
                const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(eidx)];
                {
                    Mat9 cavity_lam = ws.belief_lam[edge.j] - ws.msg_lam[edge.msg_to_j];
                    cavity_lam.noalias() += edge.factor_lam_j;
                    const Vec9 cavity_eta = ws.belief_eta[edge.j] - ws.msg_eta[edge.msg_to_j];
                    Mat9 solved_cross = Mat9::Zero();
                    Vec9 solved_eta = Vec9::Zero();
                    solveRegularizedSpd9(cavity_lam, edge.aji, cavity_eta, solved_cross, solved_eta);
                    Mat9 lam = -(edge.aij * solved_cross);
                    lam.noalias() += edge.factor_lam_i;
                    lam = 0.5 * (lam + lam.transpose());
                    const Vec9 eta = -(edge.aij * solved_eta);
                    ws.next_msg_lam[edge.msg_to_i] = keep * ws.msg_lam[edge.msg_to_i] + omega * lam;
                    ws.next_msg_eta[edge.msg_to_i] = keep * ws.msg_eta[edge.msg_to_i] + omega * eta;
                }
                {
                    Mat9 cavity_lam = ws.belief_lam[edge.i] - ws.msg_lam[edge.msg_to_i];
                    cavity_lam.noalias() += edge.factor_lam_i;
                    const Vec9 cavity_eta = ws.belief_eta[edge.i] - ws.msg_eta[edge.msg_to_i];
                    Mat9 solved_cross = Mat9::Zero();
                    Vec9 solved_eta = Vec9::Zero();
                    solveRegularizedSpd9(cavity_lam, edge.aij, cavity_eta, solved_cross, solved_eta);
                    Mat9 lam = -(edge.aji * solved_cross);
                    lam.noalias() += edge.factor_lam_j;
                    lam = 0.5 * (lam + lam.transpose());
                    const Vec9 eta = -(edge.aji * solved_eta);
                    ws.next_msg_lam[edge.msg_to_j] = keep * ws.msg_lam[edge.msg_to_j] + omega * lam;
                    ws.next_msg_eta[edge.msg_to_j] = keep * ws.msg_eta[edge.msg_to_j] + omega * eta;
                }
            }

#pragma omp for schedule(static)
            for (int i = 0; i < ws.n; ++i) {
                Mat9 lam = ws.unary_lam[static_cast<size_t>(i)];
                Vec9 eta = ws.unary_eta[static_cast<size_t>(i)];
                const int msg_begin = ws.incoming_offsets[static_cast<size_t>(i)];
                const int msg_end = ws.incoming_offsets[static_cast<size_t>(i + 1)];
                for (int msg_pos = msg_begin; msg_pos < msg_end; ++msg_pos) {
                    const int msg_id = ws.incoming_msg_ids[static_cast<size_t>(msg_pos)];
                    lam.noalias() += ws.next_msg_lam[static_cast<size_t>(msg_id)];
                    eta.noalias() += ws.next_msg_eta[static_cast<size_t>(msg_id)];
                }
                ws.next_belief_lam[static_cast<size_t>(i)] = lam;
                ws.next_belief_eta[static_cast<size_t>(i)] = eta;
            }
        }
        ws.msg_lam.swap(ws.next_msg_lam);
        ws.msg_eta.swap(ws.next_msg_eta);
        ws.belief_lam.swap(ws.next_belief_lam);
        ws.belief_eta.swap(ws.next_belief_eta);
        ws.belief_lam_inv_valid = false;
        return;
#endif
    }

    for (int i = 0; i < ws.n; ++i) {
        ws.next_belief_lam[i] = ws.unary_lam[i];
        ws.next_belief_eta[i] = ws.unary_eta[i];
    }
    for (const SchurGBPEdge& edge : ws.edges) {
        {
            Mat9 cavity_lam = ws.belief_lam[edge.j] - ws.msg_lam[edge.msg_to_j];
            cavity_lam.noalias() += edge.factor_lam_j;
            const Vec9 cavity_eta = ws.belief_eta[edge.j] - ws.msg_eta[edge.msg_to_j];
            Mat9 solved_cross = Mat9::Zero();
            Vec9 solved_eta = Vec9::Zero();
            solveRegularizedSpd9(cavity_lam, edge.aji, cavity_eta, solved_cross, solved_eta);
            Mat9 lam = -(edge.aij * solved_cross);
            lam.noalias() += edge.factor_lam_i;
            lam = 0.5 * (lam + lam.transpose());
            const Vec9 eta = -(edge.aij * solved_eta);
            const Mat9 damped_lam = keep * ws.msg_lam[edge.msg_to_i] + omega * lam;
            const Vec9 damped_eta = keep * ws.msg_eta[edge.msg_to_i] + omega * eta;
            ws.next_msg_lam[edge.msg_to_i] = damped_lam;
            ws.next_msg_eta[edge.msg_to_i] = damped_eta;
            ws.next_belief_lam[edge.i].noalias() += damped_lam;
            ws.next_belief_eta[edge.i].noalias() += damped_eta;
        }
        {
            Mat9 cavity_lam = ws.belief_lam[edge.i] - ws.msg_lam[edge.msg_to_i];
            cavity_lam.noalias() += edge.factor_lam_i;
            const Vec9 cavity_eta = ws.belief_eta[edge.i] - ws.msg_eta[edge.msg_to_i];
            Mat9 solved_cross = Mat9::Zero();
            Vec9 solved_eta = Vec9::Zero();
            solveRegularizedSpd9(cavity_lam, edge.aij, cavity_eta, solved_cross, solved_eta);
            Mat9 lam = -(edge.aji * solved_cross);
            lam.noalias() += edge.factor_lam_j;
            lam = 0.5 * (lam + lam.transpose());
            const Vec9 eta = -(edge.aji * solved_eta);
            const Mat9 damped_lam = keep * ws.msg_lam[edge.msg_to_j] + omega * lam;
            const Vec9 damped_eta = keep * ws.msg_eta[edge.msg_to_j] + omega * eta;
            ws.next_msg_lam[edge.msg_to_j] = damped_lam;
            ws.next_msg_eta[edge.msg_to_j] = damped_eta;
            ws.next_belief_lam[edge.j].noalias() += damped_lam;
            ws.next_belief_eta[edge.j].noalias() += damped_eta;
        }
    }

    ws.msg_lam.swap(ws.next_msg_lam);
    ws.msg_eta.swap(ws.next_msg_eta);
    ws.belief_lam.swap(ws.next_belief_lam);
    ws.belief_eta.swap(ws.next_belief_eta);
    ws.belief_lam_inv_valid = false;
}

void buildSchurGBPFixedEtaMaps(SchurGBPWorkspace& ws, int threads = 1) {
    const Mat9 I = Mat9::Identity();
    const Vec9 zero = Vec9::Zero();
    const int edge_count = static_cast<int>(ws.edges.size());
    const int parallel_threads = effectiveGBPThreads(threads, edge_count, ws.n);
    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static)
        for (int eidx = 0; eidx < edge_count; ++eidx) {
            const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(eidx)];
            {
                Mat9 cavity_lam = ws.belief_lam[edge.j] - ws.msg_lam[edge.msg_to_j];
                cavity_lam.noalias() += edge.factor_lam_j;
                Mat9 inv_cavity = Mat9::Zero();
                Vec9 unused = Vec9::Zero();
                solveRegularizedSpd9(cavity_lam, I, zero, inv_cavity, unused);
                ws.fixed_eta_map[edge.msg_to_i].noalias() = -(edge.aij * inv_cavity);
            }
            {
                Mat9 cavity_lam = ws.belief_lam[edge.i] - ws.msg_lam[edge.msg_to_i];
                cavity_lam.noalias() += edge.factor_lam_i;
                Mat9 inv_cavity = Mat9::Zero();
                Vec9 unused = Vec9::Zero();
                solveRegularizedSpd9(cavity_lam, I, zero, inv_cavity, unused);
                ws.fixed_eta_map[edge.msg_to_j].noalias() = -(edge.aji * inv_cavity);
            }
        }
        return;
#endif
    }

    for (const SchurGBPEdge& edge : ws.edges) {
        {
            Mat9 cavity_lam = ws.belief_lam[edge.j] - ws.msg_lam[edge.msg_to_j];
            cavity_lam.noalias() += edge.factor_lam_j;
            Mat9 inv_cavity = Mat9::Zero();
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(cavity_lam, I, zero, inv_cavity, unused);
            ws.fixed_eta_map[edge.msg_to_i].noalias() = -(edge.aij * inv_cavity);
        }
        {
            Mat9 cavity_lam = ws.belief_lam[edge.i] - ws.msg_lam[edge.msg_to_i];
            cavity_lam.noalias() += edge.factor_lam_i;
            Mat9 inv_cavity = Mat9::Zero();
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(cavity_lam, I, zero, inv_cavity, unused);
            ws.fixed_eta_map[edge.msg_to_j].noalias() = -(edge.aji * inv_cavity);
        }
    }
}

void buildSchurGBPBeliefLambdaInverses(SchurGBPWorkspace& ws, int threads = 1) {
    const Mat9 I = Mat9::Identity();
    const Vec9 zero = Vec9::Zero();
    const int parallel_threads = effectiveGBPThreads(threads, ws.n, ws.n);
    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static)
        for (int i = 0; i < ws.n; ++i) {
            Mat9 inv_lam = Mat9::Zero();
            Vec9 unused = Vec9::Zero();
            solveRegularizedSpd9(ws.belief_lam[static_cast<size_t>(i)], I, zero, inv_lam, unused);
            ws.belief_lam_inv[static_cast<size_t>(i)] = inv_lam;
        }
        ws.belief_lam_inv_valid = true;
        return;
#endif
    }
    for (int i = 0; i < ws.n; ++i) {
        Mat9 inv_lam = Mat9::Zero();
        Vec9 unused = Vec9::Zero();
        solveRegularizedSpd9(ws.belief_lam[static_cast<size_t>(i)], I, zero, inv_lam, unused);
        ws.belief_lam_inv[static_cast<size_t>(i)] = inv_lam;
    }
    ws.belief_lam_inv_valid = true;
}

void prepareSchurGBPFixedEtaPairs(SchurGBPWorkspace& ws) {
    const int edge_count = static_cast<int>(ws.edges.size());
    if (ws.fixed_eta_pair_edge_count == edge_count &&
        ws.fixed_eta_pair_i.size() == static_cast<size_t>(edge_count)) {
        return;
    }
    ws.fixed_eta_pair_edge_count = edge_count;
    ws.fixed_eta_pair_i.resize(static_cast<size_t>(edge_count));
    ws.fixed_eta_pair_j.resize(static_cast<size_t>(edge_count));
    ws.fixed_eta_pair_msg_to_i.resize(static_cast<size_t>(edge_count));
    ws.fixed_eta_pair_msg_to_j.resize(static_cast<size_t>(edge_count));
    for (int eidx = 0; eidx < edge_count; ++eidx) {
        const SchurGBPEdge& edge = ws.edges[static_cast<size_t>(eidx)];
        ws.fixed_eta_pair_i[static_cast<size_t>(eidx)] = edge.i;
        ws.fixed_eta_pair_j[static_cast<size_t>(eidx)] = edge.j;
        ws.fixed_eta_pair_msg_to_i[static_cast<size_t>(eidx)] = edge.msg_to_i;
        ws.fixed_eta_pair_msg_to_j[static_cast<size_t>(eidx)] = edge.msg_to_j;
    }
}

void schurGBPFixedEtaSweepsPairSlot(
    SchurGBPWorkspace& ws,
    double damping,
    int sweep_count,
    int threads
) {
    if (sweep_count <= 0) {
        return;
    }
    prepareSchurGBPFixedEtaPairs(ws);
    const int edge_count = ws.fixed_eta_pair_edge_count;
    const int parallel_threads = effectiveGBPThreads(threads, edge_count, ws.n);
    const double omega = std::min(1.0, std::max(0.0, damping));
    const double keep = 1.0 - omega;
    const int* pair_i = ws.fixed_eta_pair_i.data();
    const int* pair_j = ws.fixed_eta_pair_j.data();
    const int* pair_msg_i = ws.fixed_eta_pair_msg_to_i.data();
    const int* pair_msg_j = ws.fixed_eta_pair_msg_to_j.data();

    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel num_threads(parallel_threads)
        {
            for (int sweep = 0; sweep < sweep_count; ++sweep) {
#pragma omp for schedule(static)
                for (int eidx = 0; eidx < edge_count; ++eidx) {
                    const int i = pair_i[eidx];
                    const int j = pair_j[eidx];
                    const int msg_i = pair_msg_i[eidx];
                    const int msg_j = pair_msg_j[eidx];
                    dampedMat9VecDiffRawInto(ws.fixed_eta_map[static_cast<size_t>(msg_i)],
                                             ws.belief_eta[static_cast<size_t>(j)],
                                             ws.msg_eta[static_cast<size_t>(msg_j)],
                                             ws.msg_eta[static_cast<size_t>(msg_i)],
                                             keep,
                                             omega,
                                             ws.next_msg_eta[static_cast<size_t>(msg_i)]);
                    dampedMat9VecDiffRawInto(ws.fixed_eta_map[static_cast<size_t>(msg_j)],
                                             ws.belief_eta[static_cast<size_t>(i)],
                                             ws.msg_eta[static_cast<size_t>(msg_i)],
                                             ws.msg_eta[static_cast<size_t>(msg_j)],
                                             keep,
                                             omega,
                                             ws.next_msg_eta[static_cast<size_t>(msg_j)]);
                }

#pragma omp for schedule(static)
                for (int i = 0; i < ws.n; ++i) {
                    Vec9 eta = ws.unary_eta[static_cast<size_t>(i)];
                    const int msg_begin = ws.incoming_offsets[static_cast<size_t>(i)];
                    const int msg_end = ws.incoming_offsets[static_cast<size_t>(i + 1)];
                    for (int msg_pos = msg_begin; msg_pos < msg_end; ++msg_pos) {
                        const int msg_id = ws.incoming_msg_ids[static_cast<size_t>(msg_pos)];
                        eta.noalias() += ws.next_msg_eta[static_cast<size_t>(msg_id)];
                    }
                    ws.next_belief_eta[static_cast<size_t>(i)] = eta;
                }

#pragma omp single
                {
                    ws.msg_eta.swap(ws.next_msg_eta);
                    ws.belief_eta.swap(ws.next_belief_eta);
                }
            }
        }
        return;
#else
        (void)parallel_threads;
#endif
    }

    for (int sweep = 0; sweep < sweep_count; ++sweep) {
        for (int i = 0; i < ws.n; ++i) {
            ws.next_belief_eta[static_cast<size_t>(i)] = ws.unary_eta[static_cast<size_t>(i)];
        }
        for (int eidx = 0; eidx < edge_count; ++eidx) {
            const int i = pair_i[eidx];
            const int j = pair_j[eidx];
            const int msg_i = pair_msg_i[eidx];
            const int msg_j = pair_msg_j[eidx];
            dampedMat9VecDiffRawInto(ws.fixed_eta_map[static_cast<size_t>(msg_i)],
                                     ws.belief_eta[static_cast<size_t>(j)],
                                     ws.msg_eta[static_cast<size_t>(msg_j)],
                                     ws.msg_eta[static_cast<size_t>(msg_i)],
                                     keep,
                                     omega,
                                     ws.next_msg_eta[static_cast<size_t>(msg_i)]);
            ws.next_belief_eta[static_cast<size_t>(i)].noalias() +=
                ws.next_msg_eta[static_cast<size_t>(msg_i)];
            dampedMat9VecDiffRawInto(ws.fixed_eta_map[static_cast<size_t>(msg_j)],
                                     ws.belief_eta[static_cast<size_t>(i)],
                                     ws.msg_eta[static_cast<size_t>(msg_i)],
                                     ws.msg_eta[static_cast<size_t>(msg_j)],
                                     keep,
                                     omega,
                                     ws.next_msg_eta[static_cast<size_t>(msg_j)]);
            ws.next_belief_eta[static_cast<size_t>(j)].noalias() +=
                ws.next_msg_eta[static_cast<size_t>(msg_j)];
        }
        ws.msg_eta.swap(ws.next_msg_eta);
        ws.belief_eta.swap(ws.next_belief_eta);
    }
}




void schurGBPFixedEtaSweeps(
    SchurGBPWorkspace& ws,
    double damping,
    int sweep_count,
    int threads = 1
) {
    schurGBPFixedEtaSweepsPairSlot(
        ws, damping, sweep_count, threads);
}

void schurGBPMeanInto(const SchurGBPWorkspace& ws, int threads, Eigen::VectorXd& x) {
    if (x.size() != 9 * ws.n) {
        x.resize(9 * ws.n);
    }
    const int parallel_threads = effectiveGBPThreads(threads, ws.n, ws.n);
    if (parallel_threads > 1) {
#if defined(_OPENMP)
#pragma omp parallel for num_threads(parallel_threads) schedule(static)
        for (int i = 0; i < ws.n; ++i) {
            Vec9 xi = Vec9::Zero();
            if (ws.belief_lam_inv_valid) {
                mat9VecRawInto(ws.belief_lam_inv[static_cast<size_t>(i)],
                               ws.belief_eta[static_cast<size_t>(i)],
                               xi);
            } else {
                solveRegularizedSpd9Vec(ws.belief_lam[static_cast<size_t>(i)],
                                        ws.belief_eta[static_cast<size_t>(i)],
                                        xi);
            }
            double* dst = x.data() + 9 * static_cast<size_t>(i);
            for (int k = 0; k < 9; ++k) {
                dst[k] = xi[k];
            }
        }
        return;
#endif
    }

    for (int i = 0; i < ws.n; ++i) {
        Vec9 xi = Vec9::Zero();
        if (ws.belief_lam_inv_valid) {
            mat9VecRawInto(ws.belief_lam_inv[static_cast<size_t>(i)],
                           ws.belief_eta[static_cast<size_t>(i)],
                           xi);
        } else {
            solveRegularizedSpd9Vec(ws.belief_lam[i], ws.belief_eta[i], xi);
        }
        x.segment<9>(9 * i) = xi;
    }
}

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


std::string jsonNumber(double value) {
    if (!std::isfinite(value)) return "null";
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

void writeDoubleArray(std::ostream& out, const std::vector<double>& xs) {
    out << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out << ", ";
        out << jsonNumber(xs[i]);
    }
    out << "]";
}

void writeIntArray(std::ostream& out, const std::vector<int>& xs) {
    out << "[";
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) out << ", ";
        out << xs[i];
    }
    out << "]";
}


}  // namespace
