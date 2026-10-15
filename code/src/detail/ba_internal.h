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


#include "ba_messages.inc"

#include "ba_coarse.inc"


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
