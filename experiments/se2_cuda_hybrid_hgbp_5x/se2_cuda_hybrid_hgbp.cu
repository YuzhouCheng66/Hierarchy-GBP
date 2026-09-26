#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cusolverDn.h>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <cholmod.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {
namespace cg = cooperative_groups;

constexpr float kDefaultJitter = 1e-6f;
constexpr double kGbpJitter = 1e-7;

struct Pose2 {
    double x = 0.0;
    double y = 0.0;
    double theta = 0.0;
};

struct Edge2 {
    int i = -1;
    int j = -1;
    std::array<double, 3> measurement{0.0, 0.0, 0.0};
    std::array<double, 9> information{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    std::string kind;
};

struct Problem {
    std::vector<Pose2> init_poses;
    std::vector<Edge2> edges;
    Pose2 anchor_pose;
    std::array<double, 9> anchor_information{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
};

struct DirectedBlock {
    int to = -1;
    std::array<float, 9> block{};
};

struct LinearSystem {
    int n = 0;
    std::vector<float> diag;  // n row-major 3x3 blocks
    std::vector<float> rhs;   // n 3-vectors
    std::vector<std::vector<DirectedBlock>> neighbors;
};

struct BucketHost {
    int cap = 0;
    int rows = 0;
    std::vector<int> nodes;
    std::vector<int> cols;
    std::vector<float> blocks;
};

struct BucketDevice {
    int cap = 0;
    int rows = 0;
    int* nodes = nullptr;
    int* cols = nullptr;
    float* blocks = nullptr;
};

struct FlatRowsHost {
    int rows = 0;
    int total_slots = 0;
    std::vector<int> nodes;
    std::vector<int> caps;
    std::vector<int> offsets;
    std::vector<int> cols;
    std::vector<float> block_soa;  // coefficient-major: block_soa[k * total_slots + slot]
};

struct FlatRowsDevice {
    int rows = 0;
    int total_slots = 0;
    int* nodes = nullptr;
    int* caps = nullptr;
    int* offsets = nullptr;
    int* cols = nullptr;
    float* block_soa = nullptr;
};

struct GbpGraph {
    int n = 0;
    int m = 0;
    std::vector<int> factor_i;
    std::vector<int> factor_j;
    std::vector<double> hii;
    std::vector<double> hij;
    std::vector<double> hji;
    std::vector<double> hjj;
    std::vector<double> eta_i;
    std::vector<double> eta_j;
    std::vector<double> unary_lam;
    std::vector<double> unary_eta;
    std::vector<std::vector<int>> incoming_slots;
};

struct IncomingFlatHost {
    int rows = 0;
    int total_slots = 0;
    std::vector<int> nodes;
    std::vector<int> node_rows;
    std::vector<int> caps;
    std::vector<int> offsets;
    std::vector<int> slots;
};

struct GbpDevice {
    int n = 0;
    int m = 0;
    int num_slots = 0;
    int incoming_total_slots = 0;
    int* factor_i = nullptr;
    int* factor_j = nullptr;
    int* incoming_nodes = nullptr;
    int* incoming_node_rows = nullptr;
    int* incoming_caps = nullptr;
    int* incoming_offsets = nullptr;
    int* incoming_slots = nullptr;
    double* hii = nullptr;
    double* hij = nullptr;
    double* hji = nullptr;
    double* hjj = nullptr;
    double* eta_i = nullptr;
    double* eta_j = nullptr;
    double* unary_lam = nullptr;
    double* unary_eta = nullptr;
    double* msg_lam_a = nullptr;
    double* msg_lam_b = nullptr;
    double* msg_eta_a = nullptr;
    double* msg_eta_b = nullptr;
    double* belief_lam_a = nullptr;
    double* belief_lam_b = nullptr;
    double* belief_eta_a = nullptr;
    double* belief_eta_b = nullptr;
    double* fixed_eta_map = nullptr;
};

struct ProblemDevice {
    int n = 0;
    int m = 0;
    double* poses = nullptr;        // row-major n x (x,y,theta)
    double* measurements = nullptr; // row-major m x 3
    double* information = nullptr;  // row-major m x 9
    double* anchor_pose = nullptr;  // 3
    double* anchor_info = nullptr;  // 9
};

struct CoarseBlockPatternHost {
    int groups = 0;
    int coarse_dim = 0;
    int block_dim = 3;
    int block_count = 0;
    std::vector<int> block_rows;
    std::vector<int> block_cols;
    std::vector<int> factor_slot_ii;
    std::vector<int> factor_slot_ij;
    std::vector<int> factor_slot_ji;
    std::vector<int> factor_slot_jj;
};

struct CoarseBlockPatternDevice {
    int groups = 0;
    int coarse_dim = 0;
    int block_dim = 3;
    int block_count = 0;
    int* block_rows = nullptr;
    int* block_cols = nullptr;
    int* factor_slot_ii = nullptr;
    int* factor_slot_ij = nullptr;
    int* factor_slot_ji = nullptr;
    int* factor_slot_jj = nullptr;
};

struct SvdBasisHost {
    int groups = 0;
    int r = 0;
    int coarse_dim = 0;
    std::vector<int> group_start;
    std::vector<int> group_len;
    std::vector<int> var_group;
    std::vector<double> var_basis;  // per variable, column-major 3xr block: basis[3*c + row]
};

struct SvdBasisDevice {
    int groups = 0;
    int r = 0;
    int coarse_dim = 0;
    int* var_group = nullptr;
    double* var_basis = nullptr;
};

struct Args {
    std::string problem_file;
    int synthetic_chain = 0;
    int num_outer = 0;
    int sweeps = 100;
    int warmup = 5;
    int repeat = 1;
    int cpu_threads = 16;
    double omega = 1.0;
    double huber_delta = 5.0;
    bool verify = false;
    std::string kernel = "gbp_persistent";
    std::string incoming_layout = "exact";
    std::string schur_kernel = "inverse";
    std::string coop_block_policy = "work_cap";
    int fixed_lambda_start = -1;
    bool hybrid_hgbp = false;
    bool hybrid_sparse_coarse = true;
    bool hybrid_svd_basis = false;
    bool hybrid_rigid_basis = false;
    bool hybrid_gpu_svd_basis = false;
    bool gpu_svd_warm_rr_after_first = false;
    bool gpu_svd_partial_from_start = false;
    bool gpu_svd_persistent_full_basis = false;
    int group_size = 20;
    int r_reduced = 4;
    int basis_rebuild_period = 1;
    int hybrid_cycles = 1;
    double coarse_scale = 1.0;
    double gpu_svd_tol = 1e-8;
    int gpu_svd_max_sweeps = 64;
    int graph_split_threads = 32;
    int persistent_threads = 32;
    bool telemetry_repeats = false;
    bool hybrid_profile_phases = false;
    std::string trajectory_dump;
};

[[noreturn]] void usage(const char* exe) {
    std::cerr
        << "Usage:\n"
        << "  " << exe << " --problem-file file.g2o|file.problem [--sweeps 200] [--verify]\n"
        << "  " << exe << " --synthetic-chain N [--sweeps 200] [--verify]\n"
        << "\nOptions:\n"
        << "  --huber-delta VALUE   Huber delta for linearization, 0 disables robust weighting (default 5)\n"
        << "  --num-outer N        Run N nonlinear hybrid outers instead of one linearized graph benchmark\n"
        << "  --omega VALUE         Jacobi relaxation (default 1)\n"
        << "  --warmup N            Untimed GPU warmup sweeps (default 5)\n"
        << "  --repeat N            Timed GPU repetitions (default 1)\n"
        << "  --cpu-threads N       CPU reference threads when --verify is set (default 16)\n"
        << "  --kernel NAME         gbp_persistent or gbp_graph_split (default gbp_persistent)\n"
        << "  --incoming-layout L   bucket or exact incoming slot rows (default exact)\n"
        << "  --schur-kernel K      inverse or cholesky full-lambda Schur micro-kernel (default inverse)\n"
        << "  --coop-block-policy P capacity or work_cap cooperative GBP launch blocks (default work_cap)\n"
        << "  --fixed-lambda-start N  CPU-style fixed-eta-after sweep N; eta-only starts at N+1; -1 disables (default -1)\n"
        << "  --fixed-eta-after N      Alias for --fixed-lambda-start\n"
        << "  --hybrid-hgbp        Run prototype GPU lower + GPU identity-group coarse + CPU CHOLMOD upper\n"
        << "  --hybrid-dense-coarse  Use old dense coarse matrix transfer instead of sparse block transfer\n"
        << "  --hybrid-svd-basis   Use CPU-built local SVD/eigen basis instead of identity group basis\n"
        << "  --hybrid-gpu-svd-basis  Use cuSOLVER batched local SVD/eigen basis on GPU\n"
        << "  --gpu-svd-warm-rr-after-first  After outer 1, update GPU SVD basis by warm Rayleigh-Ritz\n"
        << "  --gpu-svd-partial-from-start  Use GPU partial inverse/RR basis from outer 1 instead of full first SVD\n"
        << "  --gpu-svd-persistent-full-basis  Opt-in full 60x60 GPU basis with persistent workspace/handles\n"
        << "  --hybrid-rigid-basis Use analytic SE2 rigid/low-frequency group basis instead of eigensolver basis\n"
        << "  --group-size N       Hybrid coarse identity group size (default 20)\n"
        << "  --r-reduced N        Hybrid SVD basis columns per group (default 4)\n"
        << "  --basis-rebuild-period N  SVD basis rebuild period for nonlinear outer; 1=current default, 0=build once\n"
        << "  --hybrid-cycles N    Number of hybrid linear cycles (default 1; sweeps should be even)\n"
        << "  --coarse-scale X     Scale upper correction before GPU apply (default 1)\n"
        << "  --gpu-svd-tol X      cuSOLVER Jacobi tolerance for GPU SVD basis (default 1e-8)\n"
        << "  --gpu-svd-max-sweeps N  cuSOLVER Jacobi max sweeps for GPU SVD basis (default 64)\n"
        << "  --graph-split-threads N  Threads/block for --kernel gbp_graph_split (default 32)\n"
        << "  --persistent-threads N  Threads/block for --kernel gbp_persistent (default 32)\n"
        << "  --telemetry-repeats  Print nvidia-smi clock/power/P-state after each timed repeat\n"
        << "  --hybrid-profile-phases  Print detailed full-HGBP phase timers and closure diagnostics\n"
        << "  --trajectory-dump FILE  Correctness-only binary trajectory dump; disabled in timed runs\n";
    std::exit(2);
}

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int a = 1; a < argc; ++a) {
        const std::string key = argv[a];
        auto need_value = [&](const std::string& option) -> const char* {
            if (a + 1 >= argc) {
                throw std::runtime_error("Missing value for " + option);
            }
            return argv[++a];
        };
        if (key == "--problem-file") {
            args.problem_file = need_value(key);
        } else if (key == "--synthetic-chain") {
            args.synthetic_chain = std::stoi(need_value(key));
        } else if (key == "--sweeps") {
            args.sweeps = std::stoi(need_value(key));
        } else if (key == "--num-outer") {
            args.num_outer = std::stoi(need_value(key));
        } else if (key == "--warmup") {
            args.warmup = std::stoi(need_value(key));
        } else if (key == "--repeat") {
            args.repeat = std::max(1, std::stoi(need_value(key)));
        } else if (key == "--cpu-threads") {
            args.cpu_threads = std::max(1, std::stoi(need_value(key)));
        } else if (key == "--omega") {
            args.omega = std::stod(need_value(key));
        } else if (key == "--huber-delta") {
            args.huber_delta = std::stod(need_value(key));
        } else if (key == "--kernel") {
            args.kernel = need_value(key);
        } else if (key == "--incoming-layout") {
            args.incoming_layout = need_value(key);
        } else if (key == "--schur-kernel") {
            args.schur_kernel = need_value(key);
        } else if (key == "--coop-block-policy") {
            args.coop_block_policy = need_value(key);
        } else if (key == "--fixed-lambda-start" || key == "--fixed-eta-after") {
            args.fixed_lambda_start = std::stoi(need_value(key));
        } else if (key == "--hybrid-hgbp") {
            args.hybrid_hgbp = true;
        } else if (key == "--hybrid-dense-coarse") {
            args.hybrid_sparse_coarse = false;
        } else if (key == "--hybrid-svd-basis") {
            args.hybrid_svd_basis = true;
        } else if (key == "--hybrid-gpu-svd-basis") {
            args.hybrid_svd_basis = true;
            args.hybrid_gpu_svd_basis = true;
        } else if (key == "--gpu-svd-warm-rr-after-first") {
            args.gpu_svd_warm_rr_after_first = true;
        } else if (key == "--gpu-svd-partial-from-start") {
            args.gpu_svd_partial_from_start = true;
        } else if (key == "--gpu-svd-persistent-full-basis") {
            args.gpu_svd_persistent_full_basis = true;
        } else if (key == "--hybrid-rigid-basis") {
            args.hybrid_svd_basis = true;
            args.hybrid_rigid_basis = true;
        } else if (key == "--group-size") {
            args.group_size = std::stoi(need_value(key));
        } else if (key == "--r-reduced") {
            args.r_reduced = std::stoi(need_value(key));
        } else if (key == "--basis-rebuild-period") {
            args.basis_rebuild_period = std::stoi(need_value(key));
        } else if (key == "--hybrid-cycles") {
            args.hybrid_cycles = std::stoi(need_value(key));
        } else if (key == "--coarse-scale") {
            args.coarse_scale = std::stod(need_value(key));
        } else if (key == "--gpu-svd-tol") {
            args.gpu_svd_tol = std::stod(need_value(key));
        } else if (key == "--gpu-svd-max-sweeps") {
            args.gpu_svd_max_sweeps = std::stoi(need_value(key));
        } else if (key == "--graph-split-threads") {
            args.graph_split_threads = std::stoi(need_value(key));
        } else if (key == "--persistent-threads") {
            args.persistent_threads = std::stoi(need_value(key));
        } else if (key == "--telemetry-repeats") {
            args.telemetry_repeats = true;
        } else if (key == "--hybrid-profile-phases") {
            args.hybrid_profile_phases = true;
        } else if (key == "--trajectory-dump") {
            args.trajectory_dump = need_value(key);
        } else if (key == "--verify") {
            args.verify = true;
        } else if (key == "--help" || key == "-h") {
            usage(argv[0]);
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    if (args.problem_file.empty() && args.synthetic_chain <= 0) {
        usage(argv[0]);
    }
    if (!args.problem_file.empty() && args.synthetic_chain > 0) {
        throw std::runtime_error("Use either --problem-file or --synthetic-chain, not both");
    }
    if (args.sweeps < 0 || args.warmup < 0) {
        throw std::runtime_error("Sweep counts must be non-negative");
    }
    if (args.num_outer < 0) {
        throw std::runtime_error("--num-outer must be non-negative");
    }
    if (args.kernel != "gbp_persistent" && args.kernel != "gbp_graph_split" && args.kernel != "jacobi_bucket") {
        throw std::runtime_error("--kernel must be gbp_persistent or gbp_graph_split");
    }
    if (args.incoming_layout != "bucket" && args.incoming_layout != "exact") {
        throw std::runtime_error("--incoming-layout must be bucket or exact");
    }
    if (args.schur_kernel != "inverse" && args.schur_kernel != "cholesky") {
        throw std::runtime_error("--schur-kernel must be inverse or cholesky");
    }
    if (args.coop_block_policy != "capacity" && args.coop_block_policy != "work_cap") {
        throw std::runtime_error("--coop-block-policy must be capacity or work_cap");
    }
    if (args.group_size <= 0) {
        throw std::runtime_error("--group-size must be positive");
    }
    if (args.r_reduced <= 0) {
        throw std::runtime_error("--r-reduced must be positive");
    }
    if (args.hybrid_svd_basis && !args.hybrid_sparse_coarse) {
        throw std::runtime_error("--hybrid-svd-basis requires sparse block coarse mode");
    }
    if (args.hybrid_cycles <= 0) {
        throw std::runtime_error("--hybrid-cycles must be positive");
    }
    if (args.basis_rebuild_period < 0) {
        throw std::runtime_error("--basis-rebuild-period must be non-negative");
    }
    if (!(args.gpu_svd_tol > 0.0) || args.gpu_svd_max_sweeps <= 0) {
        throw std::runtime_error("GPU SVD tolerance/max sweeps must be positive");
    }
    if (args.graph_split_threads <= 0) {
        throw std::runtime_error("--graph-split-threads must be positive");
    }
    if (args.persistent_threads <= 0) {
        throw std::runtime_error("--persistent-threads must be positive");
    }
    return args;
}

void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

enum class TrajectoryTag : uint32_t {
    ConfigText = 1,
    InitialPoses = 2,
    OuterBasis = 3,
    OuterBasisValues = 4,
    CycleCoarseDelta = 5,
    CycleProlongedDelta = 6,
    CycleFineState = 7,
    OuterPoses = 8,
    FinalMsgLamA = 9,
    FinalMsgLamB = 10,
    FinalMsgEtaA = 11,
    FinalMsgEtaB = 12,
    FinalBeliefLamA = 13,
    FinalBeliefLamB = 14,
    FinalBeliefEtaA = 15,
    FinalBeliefEtaB = 16,
    FinalObjective = 17
};

enum class TrajectoryDtype : uint32_t {
    Bytes = 1,
    Float64 = 2
};

class TrajectoryDumpWriter {
public:
    explicit TrajectoryDumpWriter(const std::string& path) {
        if (path.empty()) {
            return;
        }
        out_.open(path, std::ios::binary);
        if (!out_) {
            throw std::runtime_error("failed to open trajectory dump: " + path);
        }
        char magic[16] = {};
        const char text[] = "HGBPSE2TRJ001";
        std::memcpy(magic, text, sizeof(text) - 1);
        writeRaw(magic, sizeof(magic));
        const uint32_t version = 1;
        const uint32_t endian = 0x01020304u;
        writeRaw(&version, sizeof(version));
        writeRaw(&endian, sizeof(endian));
    }

    bool enabled() const {
        return out_.is_open();
    }

    void writeText(TrajectoryTag tag, int outer, int cycle, const std::string& text) {
        writeRecordHeader(tag, outer, cycle, TrajectoryDtype::Bytes, text.size(), text.size());
        if (!text.empty()) {
            writeRaw(text.data(), text.size());
        }
    }

    void writeDoubles(TrajectoryTag tag, int outer, int cycle, const std::vector<double>& values) {
        const uint64_t bytes = static_cast<uint64_t>(values.size() * sizeof(double));
        writeRecordHeader(tag, outer, cycle, TrajectoryDtype::Float64, values.size(), bytes);
        if (!values.empty()) {
            writeRaw(values.data(), static_cast<size_t>(bytes));
        }
    }

private:
    std::ofstream out_;

    void writeRecordHeader(
        TrajectoryTag tag,
        int outer,
        int cycle,
        TrajectoryDtype dtype,
        size_t count,
        size_t bytes
    ) {
        const uint32_t tag_u = static_cast<uint32_t>(tag);
        const int32_t outer_i = static_cast<int32_t>(outer);
        const int32_t cycle_i = static_cast<int32_t>(cycle);
        const uint32_t dtype_u = static_cast<uint32_t>(dtype);
        const uint64_t count_u = static_cast<uint64_t>(count);
        const uint64_t bytes_u = static_cast<uint64_t>(bytes);
        writeRaw(&tag_u, sizeof(tag_u));
        writeRaw(&outer_i, sizeof(outer_i));
        writeRaw(&cycle_i, sizeof(cycle_i));
        writeRaw(&dtype_u, sizeof(dtype_u));
        writeRaw(&count_u, sizeof(count_u));
        writeRaw(&bytes_u, sizeof(bytes_u));
    }

    void writeRaw(const void* data, size_t bytes) {
        out_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out_) {
            throw std::runtime_error("failed to write trajectory dump");
        }
    }
};

std::vector<double> downloadDeviceDoubles(const double* ptr, size_t count, const char* label) {
    std::vector<double> values(count);
    if (count > 0) {
        cudaCheck(cudaMemcpy(values.data(), ptr, count * sizeof(double), cudaMemcpyDeviceToHost), label);
    }
    return values;
}

std::vector<double> makePoseVector(const Problem& problem) {
    std::vector<double> poses(problem.init_poses.size() * 3);
    for (size_t i = 0; i < problem.init_poses.size(); ++i) {
        poses[3 * i + 0] = problem.init_poses[i].x;
        poses[3 * i + 1] = problem.init_poses[i].y;
        poses[3 * i + 2] = problem.init_poses[i].theta;
    }
    return poses;
}

std::string trimCopy(const std::string& s) {
    size_t first = 0;
    while (first < s.size() && std::isspace(static_cast<unsigned char>(s[first]))) {
        ++first;
    }
    size_t last = s.size();
    while (last > first && std::isspace(static_cast<unsigned char>(s[last - 1]))) {
        --last;
    }
    return s.substr(first, last - first);
}

std::vector<std::string> splitCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        fields.push_back(trimCopy(field));
    }
    return fields;
}

struct GpuTelemetry {
    bool available = false;
    std::string sm_clock_mhz = "na";
    std::string pstate = "na";
    std::string power_w = "na";
    std::string temp_c = "na";
    std::string raw = "na";
};

GpuTelemetry queryGpuTelemetry() {
    GpuTelemetry out;
#if defined(_WIN32)
    const char* cmd =
        "nvidia-smi --query-gpu=clocks.sm,pstate,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits 2>NUL";
    FILE* pipe = _popen(cmd, "r");
#else
    const char* cmd =
        "nvidia-smi --query-gpu=clocks.sm,pstate,power.draw,temperature.gpu "
        "--format=csv,noheader,nounits 2>/dev/null";
    FILE* pipe = popen(cmd, "r");
#endif
    if (pipe == nullptr) {
        return out;
    }
    char buffer[512] = {};
    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        out.raw = trimCopy(buffer);
        const std::vector<std::string> fields = splitCsvLine(out.raw);
        if (fields.size() >= 4) {
            out.sm_clock_mhz = fields[0];
            out.pstate = fields[1];
            out.power_w = fields[2];
            out.temp_c = fields[3];
            out.available = true;
        }
    }
#if defined(_WIN32)
    _pclose(pipe);
#else
    pclose(pipe);
#endif
    return out;
}

void cusolverCheck(cusolverStatus_t status, const char* what) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuSOLVER status " + std::to_string(static_cast<int>(status)));
    }
}

void cublasCheck(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) + ": cuBLAS status " + std::to_string(static_cast<int>(status)));
    }
}

double wrapAngle(double a) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    while (a > pi) {
        a -= 2.0 * pi;
    }
    while (a < -pi) {
        a += 2.0 * pi;
    }
    return a;
}

std::array<double, 9> eye3(double scale) {
    return {scale, 0.0, 0.0,
            0.0, scale, 0.0,
            0.0, 0.0, scale};
}

std::array<double, 9> info6ToMatrix(const std::array<double, 6>& v) {
    return {v[0], v[1], v[2],
            v[1], v[3], v[4],
            v[2], v[4], v[5]};
}

std::array<double, 9> se3Info21ProjectXYYaw(const std::array<double, 21>& v) {
    // g2o SE3:QUAT stores the upper triangular 6x6 information matrix.
    // This projection is only for a topology/scale stress test of the SE2 GPU kernel.
    return {v[0], v[1], v[5],
            v[1], v[6], v[10],
            v[5], v[10], v[20]};
}

double yawFromQuat(double qx, double qy, double qz, double qw) {
    const double siny_cosp = 2.0 * (qw * qz + qx * qy);
    const double cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz);
    return wrapAngle(std::atan2(siny_cosp, cosy_cosp));
}

void mat33Mul(const double* a, const double* b, double* out);

Pose2 se2Compose(const Pose2& a, const Pose2& b) {
    const double c = std::cos(a.theta);
    const double s = std::sin(a.theta);
    return {
        a.x + c * b.x - s * b.y,
        a.y + s * b.x + c * b.y,
        wrapAngle(a.theta + b.theta)
    };
}

Pose2 se2Inverse(const Pose2& a) {
    const double c = std::cos(a.theta);
    const double s = std::sin(a.theta);
    return {
        -(c * a.x + s * a.y),
        -(-s * a.x + c * a.y),
        wrapAngle(-a.theta)
    };
}

Pose2 se2Between(const Pose2& a, const Pose2& b) {
    return se2Compose(se2Inverse(a), b);
}

std::array<double, 3> se2Log(const Pose2& pose) {
    const double tx = pose.x;
    const double ty = pose.y;
    const double w = pose.theta;
    if (std::abs(w) < 1e-12) {
        return {tx, ty, 0.0};
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double denom = a * a + b * b;
    const double vx = (a * tx + b * ty) / denom;
    const double vy = (-b * tx + a * ty) / denom;
    return {vx, vy, wrapAngle(w)};
}

void jacobianLogSE2(const Pose2& pose, double* J) {
    const double tx = pose.x;
    const double ty = pose.y;
    const double w = pose.theta;
    std::fill(J, J + 9, 0.0);
    if (std::abs(w) < 1e-8) {
        J[0] = 1.0;
        J[4] = 1.0;
        J[8] = 1.0;
        J[2] = 0.5 * ty;
        J[5] = -0.5 * tx;
        return;
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    const double den = a * a + b * b;
    const double da = (w * std::cos(w) - std::sin(w)) / (w * w);
    const double db = (w * std::sin(w) - (1.0 - std::cos(w))) / (w * w);
    const double dden = 2.0 * (a * da + b * db);
    const double c = a / den;
    const double d = b / den;
    const double dc = (da * den - a * dden) / (den * den);
    const double dd = (db * den - b * dden) / (den * den);
    J[0] = c;
    J[1] = d;
    J[2] = dc * tx + dd * ty;
    J[3] = -d;
    J[4] = c;
    J[5] = -dd * tx + dc * ty;
    J[8] = 1.0;
}

void jacobianComposeInvConstant(const Pose2& z, double* J) {
    const double c = std::cos(z.theta);
    const double s = std::sin(z.theta);
    std::fill(J, J + 9, 0.0);
    J[0] = c;
    J[1] = s;
    J[3] = -s;
    J[4] = c;
    J[8] = 1.0;
}

void jacobianBetweenAbsolute(const Pose2& xi, const Pose2& xj, double* J) {
    const double c = std::cos(xi.theta);
    const double s = std::sin(xi.theta);
    const double dx = xj.x - xi.x;
    const double dy = xj.y - xi.y;
    const double rx = c * dx + s * dy;
    const double ry = -s * dx + c * dy;
    std::fill(J, J + 18, 0.0);
    J[0 * 6 + 0] = -c;
    J[0 * 6 + 1] = -s;
    J[0 * 6 + 2] = ry;
    J[0 * 6 + 3] = c;
    J[0 * 6 + 4] = s;
    J[1 * 6 + 0] = s;
    J[1 * 6 + 1] = -c;
    J[1 * 6 + 2] = -rx;
    J[1 * 6 + 3] = -s;
    J[1 * 6 + 4] = c;
    J[2 * 6 + 2] = -1.0;
    J[2 * 6 + 5] = 1.0;
}

void jacobianPlusSE2Zero(const Pose2& base, double* G) {
    const double c = std::cos(base.theta);
    const double s = std::sin(base.theta);
    std::fill(G, G + 9, 0.0);
    G[0] = c;
    G[1] = -s;
    G[3] = s;
    G[4] = c;
    G[8] = 1.0;
}

void mat33Mul36(const double* A, const double* B, double* C) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 6; ++c) {
            C[6 * r + c] =
                A[3 * r + 0] * B[c] +
                A[3 * r + 1] * B[6 + c] +
                A[3 * r + 2] * B[12 + c];
        }
    }
}

void rightMultiplyBlockDiagPlus(const double* A, const double* Gi, const double* Gj, double* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[6 * r + c] =
                A[6 * r + 0] * Gi[c] +
                A[6 * r + 1] * Gi[3 + c] +
                A[6 * r + 2] * Gi[6 + c];
            out[6 * r + 3 + c] =
                A[6 * r + 3] * Gj[c] +
                A[6 * r + 4] * Gj[3 + c] +
                A[6 * r + 5] * Gj[6 + c];
        }
    }
}

void analyticEdgeResidualJacobian(const Pose2& base_i, const Pose2& base_j, const std::array<double, 3>& z_arr, double* err, double* J) {
    const Pose2 z{z_arr[0], z_arr[1], z_arr[2]};
    const Pose2 pred = se2Between(base_i, base_j);
    const Pose2 err_pose = se2Compose(se2Inverse(z), pred);
    const auto e = se2Log(err_pose);
    err[0] = e[0];
    err[1] = e[1];
    err[2] = e[2];

    double J_between[18];
    double J_compose[9];
    double J_log[9];
    double tmp0[18];
    double tmp1[18];
    double Gi[9];
    double Gj[9];
    jacobianBetweenAbsolute(base_i, base_j, J_between);
    jacobianComposeInvConstant(z, J_compose);
    jacobianLogSE2(err_pose, J_log);
    jacobianPlusSE2Zero(base_i, Gi);
    jacobianPlusSE2Zero(base_j, Gj);
    mat33Mul36(J_compose, J_between, tmp0);
    mat33Mul36(J_log, tmp0, tmp1);
    rightMultiplyBlockDiagPlus(tmp1, Gi, Gj, J);
}

void analyticAnchorResidualJacobian(const Pose2& base_anchor, const Pose2& anchor_pose, double* err, double* J) {
    const Pose2 err_pose = se2Compose(se2Inverse(anchor_pose), base_anchor);
    const auto e = se2Log(err_pose);
    err[0] = e[0];
    err[1] = e[1];
    err[2] = e[2];
    double J_log[9];
    double J_compose[9];
    double G[9];
    double tmp[9];
    jacobianLogSE2(err_pose, J_log);
    jacobianComposeInvConstant(anchor_pose, J_compose);
    jacobianPlusSE2Zero(base_anchor, G);
    mat33Mul(J_log, J_compose, tmp);
    mat33Mul(tmp, G, J);
}

Problem makeSyntheticChain(int n) {
    if (n < 2) {
        throw std::runtime_error("--synthetic-chain must be at least 2");
    }
    Problem p;
    p.init_poses.resize(n);
    for (int i = 0; i < n; ++i) {
        p.init_poses[i].x = static_cast<double>(i) + 0.05 * std::sin(0.013 * i);
        p.init_poses[i].y = 0.02 * std::sin(0.031 * i);
        p.init_poses[i].theta = 0.01 * std::sin(0.017 * i);
    }
    p.anchor_pose = p.init_poses.front();
    p.anchor_information = eye3(1e6);
    p.edges.reserve(static_cast<size_t>(n - 1 + n / 50));
    const auto odom_info = eye3(100.0);
    const auto loop_info = eye3(25.0);
    for (int i = 0; i + 1 < n; ++i) {
        Edge2 e;
        e.i = i;
        e.j = i + 1;
        e.measurement = {1.0, 0.0, 0.0};
        e.information = odom_info;
        e.kind = "odometry";
        p.edges.push_back(e);
    }
    for (int i = 0; i + 50 < n; i += 50) {
        Edge2 e;
        e.i = i;
        e.j = i + 50;
        e.measurement = {50.0, 0.0, 0.0};
        e.information = loop_info;
        e.kind = "loop";
        p.edges.push_back(e);
    }
    return p;
}

Problem loadProblem(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open problem file: " + path);
    }

    std::string token;
    in >> token;
    if (!in) {
        throw std::runtime_error("Empty problem file: " + path);
    }

    if (token == "SYNTHETIC_SE2_PROBLEM") {
        int n = 0;
        in >> token >> n;
        if (token != "N") {
            throw std::runtime_error("Expected N section in " + path);
        }
        Problem p;
        p.init_poses.resize(n);

        in >> token;
        if (token != "POSES") {
            throw std::runtime_error("Expected POSES section in " + path);
        }
        for (int idx = 0; idx < n; ++idx) {
            int pose_id = -1;
            Pose2 gt;
            in >> pose_id >> gt.x >> gt.y >> gt.theta
               >> p.init_poses[idx].x >> p.init_poses[idx].y >> p.init_poses[idx].theta;
            if (pose_id != idx) {
                throw std::runtime_error("Pose ids must be contiguous in " + path);
            }
        }

        in >> token;
        if (token != "ANCHOR") {
            throw std::runtime_error("Expected ANCHOR section in " + path);
        }
        in >> p.anchor_pose.x >> p.anchor_pose.y >> p.anchor_pose.theta;
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                in >> p.anchor_information[3 * r + c];
            }
        }

        int m = 0;
        in >> token >> m;
        if (token != "EDGES") {
            throw std::runtime_error("Expected EDGES section in " + path);
        }
        p.edges.reserve(m);
        for (int e = 0; e < m; ++e) {
            Edge2 edge;
            in >> edge.i >> edge.j >> edge.kind
               >> edge.measurement[0] >> edge.measurement[1] >> edge.measurement[2];
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    in >> edge.information[3 * r + c];
                }
            }
            p.edges.push_back(edge);
        }
        return p;
    }

    if (token != "VERTEX_SE2" && token != "EDGE_SE2" &&
        token != "VERTEX_SE3:QUAT" && token != "EDGE_SE3:QUAT") {
        throw std::runtime_error("Unsupported problem header in " + path + ": " + token);
    }

    in.close();
    std::ifstream gin(path);
    if (!gin) {
        throw std::runtime_error("Failed to reopen g2o file: " + path);
    }

    std::unordered_map<int, Pose2> raw_vertices;
    struct RawEdge {
        int vi = -1;
        int vj = -1;
        std::array<double, 3> measurement{};
        std::array<double, 9> information{};
    };
    std::vector<RawEdge> raw_edges;

    std::string line;
    int lineno = 0;
    while (std::getline(gin, line)) {
        ++lineno;
        std::istringstream iss(line);
        std::string tag;
        if (!(iss >> tag) || tag.empty() || tag[0] == '#') {
            continue;
        }
        if (tag == "VERTEX_SE2") {
            int id = -1;
            Pose2 pose;
            if (!(iss >> id >> pose.x >> pose.y >> pose.theta)) {
                throw std::runtime_error("Malformed VERTEX_SE2 at line " + std::to_string(lineno));
            }
            raw_vertices[id] = pose;
        } else if (tag == "EDGE_SE2") {
            RawEdge edge;
            std::array<double, 6> info{};
            if (!(iss >> edge.vi >> edge.vj
                      >> edge.measurement[0] >> edge.measurement[1] >> edge.measurement[2]
                      >> info[0] >> info[1] >> info[2] >> info[3] >> info[4] >> info[5])) {
                throw std::runtime_error("Malformed EDGE_SE2 at line " + std::to_string(lineno));
            }
            edge.information = info6ToMatrix(info);
            raw_edges.push_back(edge);
        } else if (tag == "VERTEX_SE3:QUAT") {
            int id = -1;
            Pose2 pose;
            double z = 0.0;
            double qx = 0.0;
            double qy = 0.0;
            double qz = 0.0;
            double qw = 1.0;
            if (!(iss >> id >> pose.x >> pose.y >> z >> qx >> qy >> qz >> qw)) {
                throw std::runtime_error("Malformed VERTEX_SE3:QUAT at line " + std::to_string(lineno));
            }
            pose.theta = yawFromQuat(qx, qy, qz, qw);
            raw_vertices[id] = pose;
        } else if (tag == "EDGE_SE3:QUAT") {
            RawEdge edge;
            double dz = 0.0;
            double qx = 0.0;
            double qy = 0.0;
            double qz = 0.0;
            double qw = 1.0;
            std::array<double, 21> info{};
            if (!(iss >> edge.vi >> edge.vj
                      >> edge.measurement[0] >> edge.measurement[1] >> dz
                      >> qx >> qy >> qz >> qw)) {
                throw std::runtime_error("Malformed EDGE_SE3:QUAT at line " + std::to_string(lineno));
            }
            for (double& x : info) {
                if (!(iss >> x)) {
                    throw std::runtime_error("Malformed EDGE_SE3:QUAT information at line " + std::to_string(lineno));
                }
            }
            edge.measurement[2] = yawFromQuat(qx, qy, qz, qw);
            edge.information = se3Info21ProjectXYYaw(info);
            raw_edges.push_back(edge);
        }
    }
    if (raw_vertices.empty()) {
        throw std::runtime_error("No VERTEX_SE2 entries in " + path);
    }

    std::vector<int> ids;
    ids.reserve(raw_vertices.size());
    for (const auto& kv : raw_vertices) {
        ids.push_back(kv.first);
    }
    std::sort(ids.begin(), ids.end());

    std::unordered_map<int, int> id_to_idx;
    id_to_idx.reserve(ids.size());
    for (int idx = 0; idx < static_cast<int>(ids.size()); ++idx) {
        id_to_idx[ids[idx]] = idx;
    }

    Problem p;
    p.init_poses.resize(ids.size());
    for (int idx = 0; idx < static_cast<int>(ids.size()); ++idx) {
        p.init_poses[idx] = raw_vertices.at(ids[idx]);
    }
    p.anchor_pose = p.init_poses.front();
    p.anchor_information = eye3(1e8);
    p.edges.reserve(raw_edges.size());
    for (const RawEdge& raw : raw_edges) {
        auto it_i = id_to_idx.find(raw.vi);
        auto it_j = id_to_idx.find(raw.vj);
        if (it_i == id_to_idx.end() || it_j == id_to_idx.end()) {
            throw std::runtime_error("EDGE_SE2 references unknown vertex id");
        }
        Edge2 e;
        e.i = it_i->second;
        e.j = it_j->second;
        e.measurement = raw.measurement;
        e.information = raw.information;
        e.kind = (std::abs(raw.vi - raw.vj) == 1) ? "odometry" : "loop";
        p.edges.push_back(e);
    }
    return p;
}

void mat33Mul(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += a[3 * r + k] * b[3 * k + c];
            }
            out[3 * r + c] = s;
        }
    }
}

void mat33TransMul(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) {
                s += a[3 * k + r] * b[3 * k + c];
            }
            out[3 * r + c] = s;
        }
    }
}

void mat33Vec(const double* a, const double* x, double* out) {
    for (int r = 0; r < 3; ++r) {
        out[r] = a[3 * r + 0] * x[0] + a[3 * r + 1] * x[1] + a[3 * r + 2] * x[2];
    }
}

void addMat(std::vector<float>& dst, int block, const double* src, double scale) {
    float* d = dst.data() + 9 * block;
    for (int k = 0; k < 9; ++k) {
        d[k] += static_cast<float>(scale * src[k]);
    }
}

void addVec(std::vector<float>& dst, int row, const double* src, double scale) {
    float* d = dst.data() + 3 * row;
    for (int k = 0; k < 3; ++k) {
        d[k] += static_cast<float>(scale * src[k]);
    }
}

std::array<float, 9> toFloatBlock(const double* src, double scale) {
    std::array<float, 9> out{};
    for (int k = 0; k < 9; ++k) {
        out[k] = static_cast<float>(scale * src[k]);
    }
    return out;
}

LinearSystem buildLinearSystem(const Problem& p, double huber_delta) {
    (void)p;
    (void)huber_delta;
    return {};
}

void appendMat(std::vector<double>& dst, const double* src, double scale) {
    const size_t old = dst.size();
    dst.resize(old + 9);
    for (int k = 0; k < 9; ++k) {
        dst[old + k] = scale * src[k];
    }
}

void appendVec(std::vector<double>& dst, const double* src, double scale) {
    const size_t old = dst.size();
    dst.resize(old + 3);
    for (int k = 0; k < 3; ++k) {
        dst[old + k] = scale * src[k];
    }
}

void addMat(std::vector<double>& dst, int block, const double* src, double scale) {
    double* d = dst.data() + 9 * block;
    for (int k = 0; k < 9; ++k) {
        d[k] += scale * src[k];
    }
}

void addVec(std::vector<double>& dst, int row, const double* src, double scale) {
    double* d = dst.data() + 3 * row;
    for (int k = 0; k < 3; ++k) {
        d[k] += scale * src[k];
    }
}

GbpGraph buildGbpGraph(const Problem& p, double huber_delta) {
    const int n = static_cast<int>(p.init_poses.size());
    GbpGraph graph;
    graph.n = n;
    graph.m = static_cast<int>(p.edges.size());
    graph.factor_i.reserve(graph.m);
    graph.factor_j.reserve(graph.m);
    graph.hii.reserve(static_cast<size_t>(graph.m) * 9);
    graph.hij.reserve(static_cast<size_t>(graph.m) * 9);
    graph.hji.reserve(static_cast<size_t>(graph.m) * 9);
    graph.hjj.reserve(static_cast<size_t>(graph.m) * 9);
    graph.eta_i.reserve(static_cast<size_t>(graph.m) * 3);
    graph.eta_j.reserve(static_cast<size_t>(graph.m) * 3);
    graph.unary_lam.assign(static_cast<size_t>(n) * 9, 0.0f);
    graph.unary_eta.assign(static_cast<size_t>(n) * 3, 0.0f);
    graph.incoming_slots.assign(n, {});

    for (int edge_index = 0; edge_index < static_cast<int>(p.edges.size()); ++edge_index) {
        const Edge2& e = p.edges[edge_index];
        if (e.i < 0 || e.i >= n || e.j < 0 || e.j >= n) {
            throw std::runtime_error("Edge references invalid pose index");
        }
        const Pose2& xi = p.init_poses[e.i];
        const Pose2& xj = p.init_poses[e.j];
        double r[3];
        double J[18];
        analyticEdgeResidualJacobian(xi, xj, e.measurement, r, J);
        double omega[9];
        for (int k = 0; k < 9; ++k) {
            omega[k] = e.information[k];
        }

        double omega_r[3];
        mat33Vec(omega, r, omega_r);
        const double chi = std::max(0.0, r[0] * omega_r[0] + r[1] * omega_r[1] + r[2] * omega_r[2]);
        double robust_weight = 1.0;
        if (huber_delta > 0.0) {
            const double norm = std::sqrt(chi);
            if (norm > huber_delta && norm > 0.0) {
                robust_weight = huber_delta / norm;
            }
        }
        double weighted_omega[9];
        for (int k = 0; k < 9; ++k) {
            weighted_omega[k] = robust_weight * omega[k];
        }
        double weighted_omega_r[3];
        mat33Vec(weighted_omega, r, weighted_omega_r);
        double OJ[18];
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 6; ++col) {
                OJ[6 * row + col] =
                    weighted_omega[3 * row + 0] * J[col] +
                    weighted_omega[3 * row + 1] * J[6 + col] +
                    weighted_omega[3 * row + 2] * J[12 + col];
            }
        }
        double lam[36];
        double eta[6];
        for (int col = 0; col < 6; ++col) {
            eta[col] = 0.0;
            for (int row = 0; row < 3; ++row) {
                eta[col] -= J[6 * row + col] * weighted_omega_r[row];
            }
            for (int col2 = 0; col2 < 6; ++col2) {
                double s = 0.0;
                for (int row = 0; row < 3; ++row) {
                    s += J[6 * row + col] * OJ[6 * row + col2];
                }
                lam[6 * col + col2] = s;
            }
        }
        double Hii[9], Hij[9], Hji[9], Hjj[9];
        for (int rr = 0; rr < 3; ++rr) {
            for (int cc = 0; cc < 3; ++cc) {
                Hii[3 * rr + cc] = lam[6 * rr + cc];
                Hij[3 * rr + cc] = lam[6 * rr + (3 + cc)];
                Hji[3 * rr + cc] = lam[6 * (3 + rr) + cc];
                Hjj[3 * rr + cc] = lam[6 * (3 + rr) + (3 + cc)];
            }
        }
        double bi[3] = {eta[0], eta[1], eta[2]};
        double bj[3] = {eta[3], eta[4], eta[5]};

        graph.factor_i.push_back(e.i);
        graph.factor_j.push_back(e.j);
        appendMat(graph.hii, Hii, 1.0);
        appendMat(graph.hij, Hij, 1.0);
        appendMat(graph.hji, Hji, 1.0);
        appendMat(graph.hjj, Hjj, 1.0);
        appendVec(graph.eta_i, bi, 1.0);
        appendVec(graph.eta_j, bj, 1.0);

        graph.incoming_slots[e.i].push_back(2 * edge_index);
        graph.incoming_slots[e.j].push_back(2 * edge_index + 1);
    }

    double anchor_r[3];
    double anchor_J[9];
    analyticAnchorResidualJacobian(p.init_poses.front(), p.anchor_pose, anchor_r, anchor_J);
    double anchor_OJ[9];
    mat33Mul(p.anchor_information.data(), anchor_J, anchor_OJ);
    double anchor_lam[9];
    mat33TransMul(anchor_J, anchor_OJ, anchor_lam);
    double anchor_info_r[3];
    mat33Vec(p.anchor_information.data(), anchor_r, anchor_info_r);
    double anchor_eta[3] = {0.0, 0.0, 0.0};
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 3; ++row) {
            anchor_eta[col] -= anchor_J[3 * row + col] * anchor_info_r[row];
        }
    }
    addMat(graph.unary_lam, 0, anchor_lam, 1.0);
    addVec(graph.unary_eta, 0, anchor_eta, 1.0);
    return graph;
}

int bucketCapacity(int degree) {
    int cap = 1;
    while (cap < degree) {
        cap <<= 1;
    }
    return cap;
}

std::vector<BucketHost> buildBuckets(const LinearSystem& sys) {
    std::map<int, std::vector<int>> by_cap;
    for (int node = 0; node < sys.n; ++node) {
        const int degree = static_cast<int>(sys.neighbors[node].size());
        by_cap[bucketCapacity(degree)].push_back(node);
    }

    std::vector<BucketHost> buckets;
    buckets.reserve(by_cap.size());
    for (const auto& kv : by_cap) {
        BucketHost b;
        b.cap = kv.first;
        b.rows = static_cast<int>(kv.second.size());
        b.nodes = kv.second;
        b.cols.assign(static_cast<size_t>(b.rows) * b.cap, -1);
        b.blocks.assign(static_cast<size_t>(b.rows) * b.cap * 9, 0.0f);
        for (int row = 0; row < b.rows; ++row) {
            const int node = b.nodes[row];
            const auto& nbrs = sys.neighbors[node];
            for (int s = 0; s < static_cast<int>(nbrs.size()); ++s) {
                b.cols[static_cast<size_t>(row) * b.cap + s] = nbrs[s].to;
                std::copy(nbrs[s].block.begin(), nbrs[s].block.end(),
                          b.blocks.begin() + (static_cast<size_t>(row) * b.cap + s) * 9);
            }
        }
        buckets.push_back(std::move(b));
    }
    return buckets;
}

FlatRowsHost buildFlatRows(const std::vector<BucketHost>& buckets) {
    FlatRowsHost flat;
    for (const BucketHost& b : buckets) {
        flat.rows += b.rows;
        flat.total_slots += b.rows * b.cap;
    }
    flat.nodes.reserve(flat.rows);
    flat.caps.reserve(flat.rows);
    flat.offsets.reserve(flat.rows);
    flat.cols.assign(flat.total_slots, -1);
    flat.block_soa.assign(static_cast<size_t>(flat.total_slots) * 9, 0.0f);

    int row_base = 0;
    int slot_base = 0;
    for (const BucketHost& b : buckets) {
        for (int row = 0; row < b.rows; ++row) {
            flat.nodes.push_back(b.nodes[row]);
            flat.caps.push_back(b.cap);
            flat.offsets.push_back(slot_base + row * b.cap);
            for (int slot = 0; slot < b.cap; ++slot) {
                const int src_slot = row * b.cap + slot;
                const int dst_slot = slot_base + src_slot;
                flat.cols[dst_slot] = b.cols[src_slot];
                for (int k = 0; k < 9; ++k) {
                    flat.block_soa[static_cast<size_t>(k) * flat.total_slots + dst_slot] =
                        b.blocks[static_cast<size_t>(src_slot) * 9 + k];
                }
            }
        }
        row_base += b.rows;
        slot_base += b.rows * b.cap;
    }
    (void)row_base;
    return flat;
}

IncomingFlatHost buildIncomingFlat(const GbpGraph& graph, const std::string& layout) {
    std::map<int, std::vector<int>> by_cap;
    for (int node = 0; node < graph.n; ++node) {
        const int degree = static_cast<int>(graph.incoming_slots[node].size());
        const int cap = (layout == "exact") ? degree : bucketCapacity(degree);
        by_cap[cap].push_back(node);
    }

    IncomingFlatHost flat;
    flat.rows = graph.n;
    flat.nodes.reserve(graph.n);
    flat.node_rows.assign(graph.n, -1);
    flat.caps.reserve(graph.n);
    flat.offsets.reserve(graph.n);
    for (const auto& kv : by_cap) {
        flat.total_slots += static_cast<int>(kv.second.size()) * kv.first;
    }
    flat.slots.assign(flat.total_slots, -1);

    int slot_base = 0;
    for (const auto& kv : by_cap) {
            const int cap = kv.first;
            for (int local_row = 0; local_row < static_cast<int>(kv.second.size()); ++local_row) {
                const int node = kv.second[local_row];
            flat.nodes.push_back(node);
            flat.node_rows[node] = static_cast<int>(flat.nodes.size()) - 1;
            flat.caps.push_back(cap);
            flat.offsets.push_back(slot_base + local_row * cap);
            const auto& incoming = graph.incoming_slots[node];
            for (int s = 0; s < static_cast<int>(incoming.size()); ++s) {
                flat.slots[slot_base + local_row * cap + s] = incoming[s];
            }
        }
        slot_base += static_cast<int>(kv.second.size()) * cap;
    }
    return flat;
}

CoarseBlockPatternHost buildIdentityCoarseBlockPattern(const GbpGraph& graph, int group_size) {
    if (group_size <= 0) {
        throw std::runtime_error("group_size must be positive");
    }
    CoarseBlockPatternHost pattern;
    pattern.groups = (graph.n + group_size - 1) / group_size;
    pattern.block_dim = 3;
    pattern.coarse_dim = 3 * pattern.groups;
    pattern.block_rows.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    pattern.block_cols.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    pattern.factor_slot_ii.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_ij.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_ji.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_jj.resize(static_cast<size_t>(graph.m));

    std::unordered_map<unsigned long long, int> slot_for_pair;
    slot_for_pair.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    auto keyFor = [](int row, int col) -> unsigned long long {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(row)) << 32) |
               static_cast<unsigned int>(col);
    };
    auto addPair = [&](int row, int col) -> int {
        const unsigned long long key = keyFor(row, col);
        auto it = slot_for_pair.find(key);
        if (it != slot_for_pair.end()) {
            return it->second;
        }
        const int slot = static_cast<int>(pattern.block_rows.size());
        slot_for_pair.emplace(key, slot);
        pattern.block_rows.push_back(row);
        pattern.block_cols.push_back(col);
        return slot;
    };

    for (int g = 0; g < pattern.groups; ++g) {
        addPair(g, g);
    }
    for (int e = 0; e < graph.m; ++e) {
        const int gi = graph.factor_i[static_cast<size_t>(e)] / group_size;
        const int gj = graph.factor_j[static_cast<size_t>(e)] / group_size;
        pattern.factor_slot_ii[static_cast<size_t>(e)] = addPair(gi, gi);
        pattern.factor_slot_ij[static_cast<size_t>(e)] = addPair(gi, gj);
        pattern.factor_slot_ji[static_cast<size_t>(e)] = addPair(gj, gi);
        pattern.factor_slot_jj[static_cast<size_t>(e)] = addPair(gj, gj);
    }
    pattern.block_count = static_cast<int>(pattern.block_rows.size());
    return pattern;
}

SvdBasisHost buildHostSvdBasisFirstPass(const GbpGraph& graph, int group_size, int r_reduced) {
    if (group_size <= 0 || r_reduced <= 0) {
        throw std::runtime_error("invalid SVD basis group/r");
    }
    SvdBasisHost basis;
    basis.r = r_reduced;
    basis.var_group.assign(static_cast<size_t>(graph.n), -1);

    int start = 0;
    while (start + 2 * group_size <= graph.n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(group_size);
        start += group_size;
    }
    if (start < graph.n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(graph.n - start);
    }
    basis.groups = static_cast<int>(basis.group_start.size());
    basis.coarse_dim = basis.groups * basis.r;
    basis.var_basis.assign(static_cast<size_t>(graph.n) * 3 * basis.r, 0.0);

    for (int g = 0; g < basis.groups; ++g) {
        const int g_start = basis.group_start[static_cast<size_t>(g)];
        const int g_len = basis.group_len[static_cast<size_t>(g)];
        if (3 * g_len < basis.r) {
            throw std::runtime_error("SVD basis group is smaller than r_reduced");
        }
        for (int local = 0; local < g_len; ++local) {
            basis.var_group[static_cast<size_t>(g_start + local)] = g;
        }
    }

    std::vector<std::vector<int>> interior_edges(static_cast<size_t>(basis.groups));
    for (int e = 0; e < graph.m; ++e) {
        const int i = graph.factor_i[static_cast<size_t>(e)];
        const int j = graph.factor_j[static_cast<size_t>(e)];
        const int gi = basis.var_group[static_cast<size_t>(i)];
        const int gj = basis.var_group[static_cast<size_t>(j)];
        if (gi == gj && gi >= 0) {
            interior_edges[static_cast<size_t>(gi)].push_back(e);
        }
    }

    #if defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int g = 0; g < basis.groups; ++g) {
        const int g_start = basis.group_start[static_cast<size_t>(g)];
        const int g_len = basis.group_len[static_cast<size_t>(g)];
        const int dim = 3 * g_len;
        Eigen::MatrixXd info = Eigen::MatrixXd::Zero(dim, dim);
        auto addBlock = [&](int row_off, int col_off, const double* block) {
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    info(row_off + r, col_off + c) += block[3 * r + c];
                }
            }
        };

        for (int local = 0; local < g_len; ++local) {
            const int var = g_start + local;
            addBlock(3 * local, 3 * local, graph.unary_lam.data() + static_cast<size_t>(var) * 9);
        }
        for (int e : interior_edges[static_cast<size_t>(g)]) {
            const int i = graph.factor_i[static_cast<size_t>(e)];
            const int j = graph.factor_j[static_cast<size_t>(e)];
            const int li = i - g_start;
            const int lj = j - g_start;
            addBlock(3 * li, 3 * li, graph.hii.data() + static_cast<size_t>(e) * 9);
            addBlock(3 * li, 3 * lj, graph.hij.data() + static_cast<size_t>(e) * 9);
            addBlock(3 * lj, 3 * li, graph.hji.data() + static_cast<size_t>(e) * 9);
            addBlock(3 * lj, 3 * lj, graph.hjj.data() + static_cast<size_t>(e) * 9);
        }
        info = 0.5 * (info + info.transpose());
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(info);
        if (es.info() != Eigen::Success) {
            throw std::runtime_error("SVD basis eigensolver failed");
        }
        const Eigen::MatrixXd local_basis = es.eigenvectors().leftCols(basis.r);
        for (int local = 0; local < g_len; ++local) {
            const int var = g_start + local;
            double* dst = basis.var_basis.data() + static_cast<size_t>(var) * 3 * basis.r;
            const int row0 = 3 * local;
            for (int c = 0; c < basis.r; ++c) {
                dst[3 * c + 0] = local_basis(row0 + 0, c);
                dst[3 * c + 1] = local_basis(row0 + 1, c);
                dst[3 * c + 2] = local_basis(row0 + 2, c);
            }
        }
    }

    return basis;
}

SvdBasisHost buildHostRigidBasisFromPoses(const Problem& problem, int group_size, int r_reduced) {
    if (group_size <= 0 || r_reduced <= 0) {
        throw std::runtime_error("invalid rigid basis group/r");
    }
    const int n = static_cast<int>(problem.init_poses.size());
    SvdBasisHost basis;
    basis.r = r_reduced;
    basis.var_group.assign(static_cast<size_t>(n), -1);
    int start = 0;
    while (start + 2 * group_size <= n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(group_size);
        start += group_size;
    }
    if (start < n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(n - start);
    }
    basis.groups = static_cast<int>(basis.group_start.size());
    basis.coarse_dim = basis.groups * basis.r;
    basis.var_basis.assign(static_cast<size_t>(n) * 3 * basis.r, 0.0);
    for (int g = 0; g < basis.groups; ++g) {
        const int g_start = basis.group_start[static_cast<size_t>(g)];
        const int g_len = basis.group_len[static_cast<size_t>(g)];
        if (3 * g_len < basis.r) {
            throw std::runtime_error("rigid basis group is smaller than r_reduced");
        }
        for (int local = 0; local < g_len; ++local) {
            basis.var_group[static_cast<size_t>(g_start + local)] = g;
        }

        double cx = 0.0;
        double cy = 0.0;
        for (int local = 0; local < g_len; ++local) {
            const Pose2& p = problem.init_poses[static_cast<size_t>(g_start + local)];
            cx += p.x;
            cy += p.y;
        }
        cx /= static_cast<double>(g_len);
        cy /= static_cast<double>(g_len);

        double tx = 1.0;
        double ty = 0.0;
        if (g_len > 1) {
            const Pose2& p0 = problem.init_poses[static_cast<size_t>(g_start)];
            const Pose2& p1 = problem.init_poses[static_cast<size_t>(g_start + g_len - 1)];
            tx = p1.x - p0.x;
            ty = p1.y - p0.y;
            const double tn = std::hypot(tx, ty);
            if (tn > 1e-12) {
                tx /= tn;
                ty /= tn;
            } else {
                tx = std::cos(p0.theta);
                ty = std::sin(p0.theta);
            }
        }

        std::vector<double> cols(static_cast<size_t>(basis.r) * 3 * g_len, 0.0);
        auto colPtr = [&](int c) -> double* {
            return cols.data() + static_cast<size_t>(c) * 3 * g_len;
        };
        auto setLocal = [&](double* col, int local, double wx, double wy, double wt) {
            const Pose2& p = problem.init_poses[static_cast<size_t>(g_start + local)];
            const double c = std::cos(p.theta);
            const double s = std::sin(p.theta);
            col[3 * local + 0] = c * wx + s * wy;
            col[3 * local + 1] = -s * wx + c * wy;
            col[3 * local + 2] = wt;
        };

        std::vector<std::vector<double>> candidates;
        candidates.reserve(static_cast<size_t>(std::max(8, basis.r + 4)));
        auto addCandidate = [&](auto fill_fn) {
            std::vector<double> v(static_cast<size_t>(3 * g_len), 0.0);
            for (int local = 0; local < g_len; ++local) {
                fill_fn(v.data(), local);
            }
            candidates.push_back(std::move(v));
        };

        addCandidate([&](double* col, int local) { setLocal(col, local, 1.0, 0.0, 0.0); });
        addCandidate([&](double* col, int local) { setLocal(col, local, 0.0, 1.0, 0.0); });
        addCandidate([&](double* col, int local) {
            const Pose2& p = problem.init_poses[static_cast<size_t>(g_start + local)];
            setLocal(col, local, -(p.y - cy), p.x - cx, 1.0);
        });
        addCandidate([&](double* col, int local) {
            const double denom = std::max(1, g_len - 1);
            const double tau = (static_cast<double>(local) / static_cast<double>(denom)) - 0.5;
            setLocal(col, local, tau * tx, tau * ty, 0.0);
        });
        addCandidate([&](double* col, int local) {
            const double denom = std::max(1, g_len - 1);
            const double tau = (static_cast<double>(local) / static_cast<double>(denom)) - 0.5;
            setLocal(col, local, -tau * ty, tau * tx, 0.0);
        });
        addCandidate([&](double* col, int local) {
            const double denom = std::max(1, g_len - 1);
            const double tau = (static_cast<double>(local) / static_cast<double>(denom)) - 0.5;
            setLocal(col, local, 0.0, 0.0, tau);
        });

        int accepted = 0;
        auto tryAccept = [&](std::vector<double> v) {
            for (int c = 0; c < accepted; ++c) {
                const double* q = colPtr(c);
                double dot = 0.0;
                for (int k = 0; k < 3 * g_len; ++k) {
                    dot += v[static_cast<size_t>(k)] * q[k];
                }
                for (int k = 0; k < 3 * g_len; ++k) {
                    v[static_cast<size_t>(k)] -= dot * q[k];
                }
            }
            double nrm = 0.0;
            for (double x : v) {
                nrm += x * x;
            }
            nrm = std::sqrt(nrm);
            if (nrm <= 1e-12) {
                return false;
            }
            double* dst = colPtr(accepted);
            for (int k = 0; k < 3 * g_len; ++k) {
                dst[k] = v[static_cast<size_t>(k)] / nrm;
            }
            ++accepted;
            return true;
        };
        for (const std::vector<double>& cand : candidates) {
            if (accepted >= basis.r) {
                break;
            }
            tryAccept(cand);
        }
        for (int k = 0; accepted < basis.r && k < 3 * g_len; ++k) {
            std::vector<double> unit(static_cast<size_t>(3 * g_len), 0.0);
            unit[static_cast<size_t>(k)] = 1.0;
            tryAccept(std::move(unit));
        }
        if (accepted != basis.r) {
            throw std::runtime_error("rigid basis Gram-Schmidt failed");
        }

        for (int local = 0; local < g_len; ++local) {
            const int var = g_start + local;
            double* dst = basis.var_basis.data() + static_cast<size_t>(var) * 3 * basis.r;
            for (int c = 0; c < basis.r; ++c) {
                const double* src = colPtr(c);
                dst[3 * c + 0] = src[3 * local + 0];
                dst[3 * c + 1] = src[3 * local + 1];
                dst[3 * c + 2] = src[3 * local + 2];
            }
        }
    }
    return basis;
}

CoarseBlockPatternHost buildSvdCoarseBlockPattern(const GbpGraph& graph, const SvdBasisHost& basis) {
    CoarseBlockPatternHost pattern;
    pattern.groups = basis.groups;
    pattern.block_dim = basis.r;
    pattern.coarse_dim = basis.coarse_dim;
    pattern.block_rows.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    pattern.block_cols.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    pattern.factor_slot_ii.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_ij.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_ji.resize(static_cast<size_t>(graph.m));
    pattern.factor_slot_jj.resize(static_cast<size_t>(graph.m));

    std::unordered_map<unsigned long long, int> slot_for_pair;
    slot_for_pair.reserve(static_cast<size_t>(pattern.groups) + 2 * graph.m);
    auto keyFor = [](int row, int col) -> unsigned long long {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(row)) << 32) |
               static_cast<unsigned int>(col);
    };
    auto addPair = [&](int row, int col) -> int {
        const unsigned long long key = keyFor(row, col);
        auto it = slot_for_pair.find(key);
        if (it != slot_for_pair.end()) {
            return it->second;
        }
        const int slot = static_cast<int>(pattern.block_rows.size());
        slot_for_pair.emplace(key, slot);
        pattern.block_rows.push_back(row);
        pattern.block_cols.push_back(col);
        return slot;
    };

    for (int g = 0; g < pattern.groups; ++g) {
        addPair(g, g);
    }
    for (int e = 0; e < graph.m; ++e) {
        const int gi = basis.var_group[static_cast<size_t>(graph.factor_i[static_cast<size_t>(e)])];
        const int gj = basis.var_group[static_cast<size_t>(graph.factor_j[static_cast<size_t>(e)])];
        pattern.factor_slot_ii[static_cast<size_t>(e)] = addPair(gi, gi);
        pattern.factor_slot_ij[static_cast<size_t>(e)] = addPair(gi, gj);
        pattern.factor_slot_ji[static_cast<size_t>(e)] = addPair(gj, gi);
        pattern.factor_slot_jj[static_cast<size_t>(e)] = addPair(gj, gj);
    }
    pattern.block_count = static_cast<int>(pattern.block_rows.size());
    return pattern;
}

void printIncomingStats(const GbpGraph& graph, const IncomingFlatHost& flat) {
    int max_degree = 0;
    std::uint64_t actual_slots = 0;
    std::map<int, int> rows_by_cap;
    for (int node = 0; node < graph.n; ++node) {
        const int degree = static_cast<int>(graph.incoming_slots[node].size());
        max_degree = std::max(max_degree, degree);
        actual_slots += static_cast<std::uint64_t>(degree);
    }
    for (int cap : flat.caps) {
        rows_by_cap[cap] += 1;
    }
    const double waste = flat.total_slots > 0
        ? 100.0 * static_cast<double>(flat.total_slots - actual_slots) / static_cast<double>(flat.total_slots)
        : 0.0;
    std::cout << "incoming_bucket_stats max_degree=" << max_degree
              << " buckets=" << rows_by_cap.size()
              << " actual_slots=" << actual_slots
              << " padded_slots=" << flat.total_slots
              << " padding_waste=" << std::fixed << std::setprecision(2) << waste << "%\n";
    for (const auto& kv : rows_by_cap) {
        std::cout << "  cap=" << std::setw(3) << kv.first << " rows=" << kv.second << "\n";
    }
}

void printBucketStats(const LinearSystem& sys, const std::vector<BucketHost>& buckets) {
    int max_degree = 0;
    std::uint64_t actual_slots = 0;
    std::uint64_t padded_slots = 0;
    for (const auto& nbrs : sys.neighbors) {
        max_degree = std::max(max_degree, static_cast<int>(nbrs.size()));
        actual_slots += nbrs.size();
    }
    for (const BucketHost& b : buckets) {
        padded_slots += static_cast<std::uint64_t>(b.rows) * b.cap;
    }
    const double waste = padded_slots > 0
        ? 100.0 * static_cast<double>(padded_slots - actual_slots) / static_cast<double>(padded_slots)
        : 0.0;
    std::cout << "bucket_stats max_degree=" << max_degree
              << " buckets=" << buckets.size()
              << " actual_slots=" << actual_slots
              << " padded_slots=" << padded_slots
              << " padding_waste=" << std::fixed << std::setprecision(2) << waste << "%\n";
    for (const BucketHost& b : buckets) {
        std::cout << "  cap=" << std::setw(3) << b.cap << " rows=" << b.rows << "\n";
    }
}

__device__ void solve3SpdDevice(const float* a, const float* b, float* x) {
    float jitter = kDefaultJitter;
    float l00 = 0.0f, l10 = 0.0f, l11 = 0.0f, l20 = 0.0f, l21 = 0.0f, l22 = 0.0f;
    bool ok = false;
    #pragma unroll
    for (int attempt = 0; attempt < 12; ++attempt) {
        const float a00 = a[0] + jitter;
        const float a11 = a[4] + jitter;
        const float a22 = a[8] + jitter;
        if (a00 > 0.0f) {
            l00 = sqrtf(a00);
            l10 = a[3] / l00;
            const float d11 = a11 - l10 * l10;
            if (d11 > 0.0f) {
                l11 = sqrtf(d11);
                l20 = a[6] / l00;
                l21 = (a[7] - l20 * l10) / l11;
                const float d22 = a22 - l20 * l20 - l21 * l21;
                if (d22 > 0.0f) {
                    l22 = sqrtf(d22);
                    ok = true;
                    break;
                }
            }
        }
        jitter *= 10.0f;
    }
    if (!ok) {
        const float d0 = fmaxf(fabsf(a[0]), kDefaultJitter);
        const float d1 = fmaxf(fabsf(a[4]), kDefaultJitter);
        const float d2 = fmaxf(fabsf(a[8]), kDefaultJitter);
        x[0] = b[0] / d0;
        x[1] = b[1] / d1;
        x[2] = b[2] / d2;
        return;
    }

    const float y0 = b[0] / l00;
    const float y1 = (b[1] - l10 * y0) / l11;
    const float y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22;
    x[1] = (y1 - l21 * x[2]) / l11;
    x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
}

__device__ void solve3SpdMatDevice(const float* a, const float* b, float* x) {
    #pragma unroll
    for (int col = 0; col < 3; ++col) {
        float rhs[3] = {b[col], b[3 + col], b[6 + col]};
        float sol[3];
        solve3SpdDevice(a, rhs, sol);
        x[col] = sol[0];
        x[3 + col] = sol[1];
        x[6 + col] = sol[2];
    }
}

__device__ void mat33MulDevice(const float* a, const float* b, float* out) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            out[3 * r + c] =
                a[3 * r + 0] * b[c] +
                a[3 * r + 1] * b[3 + c] +
                a[3 * r + 2] * b[6 + c];
        }
    }
}

__device__ void mat33VecDevice(const float* a, const float* x, float* out) {
    out[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2];
    out[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2];
    out[2] = a[6] * x[0] + a[7] * x[1] + a[8] * x[2];
}

__device__ void solve3SpdDevice(const double* a, const double* b, double* x) {
    double jitter = kGbpJitter;
    double l00 = 0.0, l10 = 0.0, l11 = 0.0, l20 = 0.0, l21 = 0.0, l22 = 0.0;
    bool ok = false;
    #pragma unroll
    for (int attempt = 0; attempt < 8; ++attempt) {
        const double a00 = a[0] + jitter;
        const double a11 = a[4] + jitter;
        const double a22 = a[8] + jitter;
        if (a00 > 0.0) {
            l00 = sqrt(a00);
            l10 = a[3] / l00;
            const double d11 = a11 - l10 * l10;
            if (d11 > 0.0) {
                l11 = sqrt(d11);
                l20 = a[6] / l00;
                l21 = (a[7] - l20 * l10) / l11;
                const double d22 = a22 - l20 * l20 - l21 * l21;
                if (d22 > 0.0) {
                    l22 = sqrt(d22);
                    ok = true;
                    break;
                }
            }
        }
        jitter *= 10.0;
    }
    if (!ok) {
        const double d0 = fmax(fabs(a[0]), kGbpJitter);
        const double d1 = fmax(fabs(a[4]), kGbpJitter);
        const double d2 = fmax(fabs(a[8]), kGbpJitter);
        x[0] = b[0] / d0;
        x[1] = b[1] / d1;
        x[2] = b[2] / d2;
        return;
    }
    const double y0 = b[0] / l00;
    const double y1 = (b[1] - l10 * y0) / l11;
    const double y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22;
    x[1] = (y1 - l21 * x[2]) / l11;
    x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
}

__device__ void solve3SpdMatDevice(const double* a, const double* b, double* x) {
    #pragma unroll
    for (int col = 0; col < 3; ++col) {
        double rhs[3] = {b[col], b[3 + col], b[6 + col]};
        double sol[3];
        solve3SpdDevice(a, rhs, sol);
        x[col] = sol[0];
        x[3 + col] = sol[1];
        x[6 + col] = sol[2];
    }
}

__device__ void mat33MulDevice(const double* a, const double* b, double* out) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            out[3 * r + c] =
                a[3 * r + 0] * b[c] +
                a[3 * r + 1] * b[3 + c] +
                a[3 * r + 2] * b[6 + c];
        }
    }
}

__device__ void mat33VecDevice(const double* a, const double* x, double* out) {
    out[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2];
    out[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2];
    out[2] = a[6] * x[0] + a[7] * x[1] + a[8] * x[2];
}

struct Pose2Dev {
    double x;
    double y;
    double theta;
};

__device__ __forceinline__ double wrapAngleDevice(double a) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    while (a > pi) {
        a -= 2.0 * pi;
    }
    while (a < -pi) {
        a += 2.0 * pi;
    }
    return a;
}

__device__ __forceinline__ Pose2Dev poseFromArrayDevice(const double* poses, int idx) {
    return {poses[3 * idx + 0], poses[3 * idx + 1], poses[3 * idx + 2]};
}

__device__ __forceinline__ Pose2Dev se2ComposeDevice(const Pose2Dev& a, const Pose2Dev& b) {
    const double c = cos(a.theta);
    const double s = sin(a.theta);
    return {
        a.x + c * b.x - s * b.y,
        a.y + s * b.x + c * b.y,
        wrapAngleDevice(a.theta + b.theta)
    };
}

__device__ __forceinline__ Pose2Dev se2InverseDevice(const Pose2Dev& a) {
    const double c = cos(a.theta);
    const double s = sin(a.theta);
    return {
        -(c * a.x + s * a.y),
        -(-s * a.x + c * a.y),
        wrapAngleDevice(-a.theta)
    };
}

__device__ __forceinline__ Pose2Dev se2BetweenDevice(const Pose2Dev& a, const Pose2Dev& b) {
    return se2ComposeDevice(se2InverseDevice(a), b);
}

__device__ __forceinline__ void se2LogDevice(const Pose2Dev& pose, double* out) {
    const double tx = pose.x;
    const double ty = pose.y;
    const double w = pose.theta;
    if (fabs(w) < 1e-12) {
        out[0] = tx;
        out[1] = ty;
        out[2] = 0.0;
        return;
    }
    const double a = sin(w) / w;
    const double b = (1.0 - cos(w)) / w;
    const double denom = a * a + b * b;
    out[0] = (a * tx + b * ty) / denom;
    out[1] = (-b * tx + a * ty) / denom;
    out[2] = wrapAngleDevice(w);
}

__device__ void jacobianLogSE2Device(const Pose2Dev& pose, double* J) {
    const double tx = pose.x;
    const double ty = pose.y;
    const double w = pose.theta;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        J[k] = 0.0;
    }
    if (fabs(w) < 1e-8) {
        J[0] = 1.0;
        J[4] = 1.0;
        J[8] = 1.0;
        J[2] = 0.5 * ty;
        J[5] = -0.5 * tx;
        return;
    }
    const double a = sin(w) / w;
    const double b = (1.0 - cos(w)) / w;
    const double den = a * a + b * b;
    const double da = (w * cos(w) - sin(w)) / (w * w);
    const double db = (w * sin(w) - (1.0 - cos(w))) / (w * w);
    const double dden = 2.0 * (a * da + b * db);
    const double cc = a / den;
    const double d = b / den;
    const double dc = (da * den - a * dden) / (den * den);
    const double dd = (db * den - b * dden) / (den * den);
    J[0] = cc;
    J[1] = d;
    J[2] = dc * tx + dd * ty;
    J[3] = -d;
    J[4] = cc;
    J[5] = -dd * tx + dc * ty;
    J[8] = 1.0;
}

__device__ void jacobianComposeInvConstantDevice(const Pose2Dev& z, double* J) {
    const double c = cos(z.theta);
    const double s = sin(z.theta);
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        J[k] = 0.0;
    }
    J[0] = c;
    J[1] = s;
    J[3] = -s;
    J[4] = c;
    J[8] = 1.0;
}

__device__ void jacobianBetweenAbsoluteDevice(const Pose2Dev& xi, const Pose2Dev& xj, double* J) {
    const double c = cos(xi.theta);
    const double s = sin(xi.theta);
    const double dx = xj.x - xi.x;
    const double dy = xj.y - xi.y;
    const double rx = c * dx + s * dy;
    const double ry = -s * dx + c * dy;
    #pragma unroll
    for (int k = 0; k < 18; ++k) {
        J[k] = 0.0;
    }
    J[0 * 6 + 0] = -c;
    J[0 * 6 + 1] = -s;
    J[0 * 6 + 2] = ry;
    J[0 * 6 + 3] = c;
    J[0 * 6 + 4] = s;
    J[1 * 6 + 0] = s;
    J[1 * 6 + 1] = -c;
    J[1 * 6 + 2] = -rx;
    J[1 * 6 + 3] = -s;
    J[1 * 6 + 4] = c;
    J[2 * 6 + 2] = -1.0;
    J[2 * 6 + 5] = 1.0;
}

__device__ void jacobianPlusSE2ZeroDevice(const Pose2Dev& base, double* G) {
    const double c = cos(base.theta);
    const double s = sin(base.theta);
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        G[k] = 0.0;
    }
    G[0] = c;
    G[1] = -s;
    G[3] = s;
    G[4] = c;
    G[8] = 1.0;
}

__device__ void mat33Mul36Device(const double* A, const double* B, double* C) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 6; ++c) {
            C[6 * r + c] =
                A[3 * r + 0] * B[c] +
                A[3 * r + 1] * B[6 + c] +
                A[3 * r + 2] * B[12 + c];
        }
    }
}

__device__ void rightMultiplyBlockDiagPlusDevice(const double* A, const double* Gi, const double* Gj, double* out) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            out[6 * r + c] =
                A[6 * r + 0] * Gi[c] +
                A[6 * r + 1] * Gi[3 + c] +
                A[6 * r + 2] * Gi[6 + c];
            out[6 * r + 3 + c] =
                A[6 * r + 3] * Gj[c] +
                A[6 * r + 4] * Gj[3 + c] +
                A[6 * r + 5] * Gj[6 + c];
        }
    }
}

__device__ void analyticEdgeResidualJacobianDevice(
    const Pose2Dev& base_i,
    const Pose2Dev& base_j,
    const double* z_arr,
    double* err,
    double* J
) {
    const Pose2Dev z{z_arr[0], z_arr[1], z_arr[2]};
    const Pose2Dev pred = se2BetweenDevice(base_i, base_j);
    const Pose2Dev err_pose = se2ComposeDevice(se2InverseDevice(z), pred);
    se2LogDevice(err_pose, err);
    double J_between[18];
    double J_compose[9];
    double J_log[9];
    double tmp0[18];
    double tmp1[18];
    double Gi[9];
    double Gj[9];
    jacobianBetweenAbsoluteDevice(base_i, base_j, J_between);
    jacobianComposeInvConstantDevice(z, J_compose);
    jacobianLogSE2Device(err_pose, J_log);
    jacobianPlusSE2ZeroDevice(base_i, Gi);
    jacobianPlusSE2ZeroDevice(base_j, Gj);
    mat33Mul36Device(J_compose, J_between, tmp0);
    mat33Mul36Device(J_log, tmp0, tmp1);
    rightMultiplyBlockDiagPlusDevice(tmp1, Gi, Gj, J);
}

__device__ void analyticAnchorResidualJacobianDevice(
    const Pose2Dev& base_anchor,
    const Pose2Dev& anchor_pose,
    double* err,
    double* J
) {
    const Pose2Dev err_pose = se2ComposeDevice(se2InverseDevice(anchor_pose), base_anchor);
    se2LogDevice(err_pose, err);
    double J_log[9];
    double J_compose[9];
    double G[9];
    double tmp[9];
    jacobianLogSE2Device(err_pose, J_log);
    jacobianComposeInvConstantDevice(anchor_pose, J_compose);
    jacobianPlusSE2ZeroDevice(base_anchor, G);
    mat33MulDevice(J_log, J_compose, tmp);
    mat33MulDevice(tmp, G, J);
}

struct Chol3Device {
    double l00;
    double l10;
    double l20;
    double l11;
    double l21;
    double l22;
};

__device__ bool factorize3LowerNoJitterDevice(const double* a, Chol3Device& chol) {
    const double a00 = a[0];
    const double a10 = a[3];
    const double a20 = a[6];
    const double a11 = a[4];
    const double a21 = a[7];
    const double a22 = a[8];
    if (!(a00 > 0.0)) {
        return false;
    }
    chol.l00 = sqrt(a00);
    chol.l10 = a10 / chol.l00;
    chol.l20 = a20 / chol.l00;
    const double d11 = a11 - chol.l10 * chol.l10;
    if (!(d11 > 0.0)) {
        return false;
    }
    chol.l11 = sqrt(d11);
    chol.l21 = (a21 - chol.l20 * chol.l10) / chol.l11;
    const double d22 = a22 - chol.l20 * chol.l20 - chol.l21 * chol.l21;
    if (!(d22 > 0.0)) {
        return false;
    }
    chol.l22 = sqrt(d22);
    return true;
}

__device__ void solve3LowerDevice(const Chol3Device& chol, const double* b, double* x) {
    const double y0 = b[0] / chol.l00;
    const double y1 = (b[1] - chol.l10 * y0) / chol.l11;
    const double y2 = (b[2] - chol.l20 * y0 - chol.l21 * y1) / chol.l22;
    x[2] = y2 / chol.l22;
    x[1] = (y1 - chol.l21 * x[2]) / chol.l11;
    x[0] = (y0 - chol.l10 * x[1] - chol.l20 * x[2]) / chol.l00;
}

__device__ void solve3LowerMatDevice(const Chol3Device& chol, const double* b, double* x) {
    #pragma unroll
    for (int col = 0; col < 3; ++col) {
        const double rhs[3] = {b[col], b[3 + col], b[6 + col]};
        double sol[3];
        solve3LowerDevice(chol, rhs, sol);
        x[col] = sol[0];
        x[3 + col] = sol[1];
        x[6 + col] = sol[2];
    }
}

__device__ void computeGbpMessageDevice(
    const double* self_lam,
    const double* cross,
    const double* cross_t,
    const double* other_lam,
    const double* self_eta,
    const double* other_eta,
    const double* cavity_lam,
    const double* cavity_eta,
    const double* old_msg_lam6,
    const double* old_msg_eta,
    double damping,
    int schur_mode,
    double* out_msg_lam6,
    double* out_msg_eta
) {
    double cond[9];
    double cond_eta[3];
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        cond[k] = other_lam[k] + cavity_lam[k];
    }
    cond[0] += kGbpJitter;
    cond[4] += kGbpJitter;
    cond[8] += kGbpJitter;
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        cond_eta[k] = other_eta[k] + cavity_eta[k];
    }

    const double a00 = cond[0];
    const double a10 = cond[3];
    const double a20 = cond[6];
    const double a11 = cond[4];
    const double a21 = cond[7];
    const double a22 = cond[8];
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;

    double schur[9];
    double eta_schur[3];
    if (schur_mode == 0 && det > 0.0) {
        const double inv_det = 1.0 / det;
        const double inv[9] = {
            cof00 * inv_det, cof10 * inv_det, cof20 * inv_det,
            cof10 * inv_det, cof11 * inv_det, cof21 * inv_det,
            cof20 * inv_det, cof21 * inv_det, cof22 * inv_det
        };
        double map[9];
        mat33MulDevice(cross, inv, map);
        double solved_eta[3];
        mat33VecDevice(inv, cond_eta, solved_eta);
        mat33VecDevice(cross, solved_eta, eta_schur);

        const double schur00 = map[0] * cross_t[0] + map[1] * cross_t[3] + map[2] * cross_t[6];
        const double schur01 = map[0] * cross_t[1] + map[1] * cross_t[4] + map[2] * cross_t[7];
        const double schur02 = map[0] * cross_t[2] + map[1] * cross_t[5] + map[2] * cross_t[8];
        const double schur11 = map[3] * cross_t[1] + map[4] * cross_t[4] + map[5] * cross_t[7];
        const double schur12 = map[3] * cross_t[2] + map[4] * cross_t[5] + map[5] * cross_t[8];
        const double schur22 = map[6] * cross_t[2] + map[7] * cross_t[5] + map[8] * cross_t[8];
        out_msg_lam6[0] = (1.0 - damping) * old_msg_lam6[0] + damping * (self_lam[0] - schur00);
        out_msg_lam6[1] = (1.0 - damping) * old_msg_lam6[1] + damping * (self_lam[1] - schur01);
        out_msg_lam6[2] = (1.0 - damping) * old_msg_lam6[2] + damping * (self_lam[2] - schur02);
        out_msg_lam6[3] = (1.0 - damping) * old_msg_lam6[3] + damping * (self_lam[4] - schur11);
        out_msg_lam6[4] = (1.0 - damping) * old_msg_lam6[4] + damping * (self_lam[5] - schur12);
        out_msg_lam6[5] = (1.0 - damping) * old_msg_lam6[5] + damping * (self_lam[8] - schur22);

        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            const double fresh = self_eta[k] - eta_schur[k];
            out_msg_eta[k] = (1.0 - damping) * old_msg_eta[k] + damping * fresh;
        }
        return;
    } else {
        Chol3Device chol{};
        if (factorize3LowerNoJitterDevice(cond, chol)) {
            double solved_cross_t[9];
            solve3LowerMatDevice(chol, cross_t, solved_cross_t);
            mat33MulDevice(cross, solved_cross_t, schur);
            double solved_eta[3];
            solve3LowerDevice(chol, cond_eta, solved_eta);
            mat33VecDevice(cross, solved_eta, eta_schur);
        } else {
            double solved_cross_t[9];
            solve3SpdMatDevice(cond, cross_t, solved_cross_t);
            mat33MulDevice(cross, solved_cross_t, schur);
            double solved_eta[3];
            solve3SpdDevice(cond, cond_eta, solved_eta);
            mat33VecDevice(cross, solved_eta, eta_schur);
        }
    }

    const double fresh00 = self_lam[0] - schur[0];
    const double fresh01 = 0.5 * ((self_lam[1] - schur[1]) + (self_lam[3] - schur[3]));
    const double fresh02 = 0.5 * ((self_lam[2] - schur[2]) + (self_lam[6] - schur[6]));
    const double fresh11 = self_lam[4] - schur[4];
    const double fresh12 = 0.5 * ((self_lam[5] - schur[5]) + (self_lam[7] - schur[7]));
    const double fresh22 = self_lam[8] - schur[8];
    out_msg_lam6[0] = (1.0 - damping) * old_msg_lam6[0] + damping * fresh00;
    out_msg_lam6[1] = (1.0 - damping) * old_msg_lam6[1] + damping * fresh01;
    out_msg_lam6[2] = (1.0 - damping) * old_msg_lam6[2] + damping * fresh02;
    out_msg_lam6[3] = (1.0 - damping) * old_msg_lam6[3] + damping * fresh11;
    out_msg_lam6[4] = (1.0 - damping) * old_msg_lam6[4] + damping * fresh12;
    out_msg_lam6[5] = (1.0 - damping) * old_msg_lam6[5] + damping * fresh22;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const double fresh = self_eta[k] - eta_schur[k];
        out_msg_eta[k] = (1.0 - damping) * old_msg_eta[k] + damping * fresh;
    }
}

__device__ void buildEtaMapDevice(
    const double* cross,
    const double* other_lam,
    const double* cavity_lam,
    int schur_mode,
    double* eta_map
) {
    double cond[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        cond[k] = other_lam[k] + cavity_lam[k];
    }
    cond[0] += kGbpJitter;
    cond[4] += kGbpJitter;
    cond[8] += kGbpJitter;

    const double a00 = cond[0];
    const double a10 = cond[3];
    const double a20 = cond[6];
    const double a11 = cond[4];
    const double a21 = cond[7];
    const double a22 = cond[8];
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (schur_mode == 0 && det > 0.0) {
        const double inv_det = 1.0 / det;
        const double inv[9] = {
            cof00 * inv_det, cof10 * inv_det, cof20 * inv_det,
            cof10 * inv_det, cof11 * inv_det, cof21 * inv_det,
            cof20 * inv_det, cof21 * inv_det, cof22 * inv_det
        };
        mat33MulDevice(cross, inv, eta_map);
        return;
    }

    Chol3Device chol{};
    if (factorize3LowerNoJitterDevice(cond, chol)) {
        const double cross_t[9] = {
            cross[0], cross[3], cross[6],
            cross[1], cross[4], cross[7],
            cross[2], cross[5], cross[8]
        };
        double solved[9];
        solve3LowerMatDevice(chol, cross_t, solved);
        // solved = C^{-1} cross^T, so eta_map = cross C^{-1} = solved^T.
        eta_map[0] = solved[0];
        eta_map[1] = solved[3];
        eta_map[2] = solved[6];
        eta_map[3] = solved[1];
        eta_map[4] = solved[4];
        eta_map[5] = solved[7];
        eta_map[6] = solved[2];
        eta_map[7] = solved[5];
        eta_map[8] = solved[8];
        return;
    }

    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        eta_map[k] = 0.0;
    }
}

__device__ void computeEtaOnlyMessageDevice(
    const double* self_eta,
    const double* other_eta,
    const double* cavity_eta,
    const double* eta_map,
    const double* old_msg_eta,
    double* out_msg_eta
) {
    (void)old_msg_eta;
    const double eno0 = other_eta[0] + cavity_eta[0];
    const double eno1 = other_eta[1] + cavity_eta[1];
    const double eno2 = other_eta[2] + cavity_eta[2];
    const double proj0 = eta_map[0] * eno0 + eta_map[1] * eno1 + eta_map[2] * eno2;
    const double proj1 = eta_map[3] * eno0 + eta_map[4] * eno1 + eta_map[5] * eno2;
    const double proj2 = eta_map[6] * eno0 + eta_map[7] * eno1 + eta_map[8] * eno2;
    out_msg_eta[0] = self_eta[0] - proj0;
    out_msg_eta[1] = self_eta[1] - proj1;
    out_msg_eta[2] = self_eta[2] - proj2;
}

__device__ __forceinline__ bool computeGbpMessagePackedInverseFromFullDevice(
    const double* __restrict__ self_lam,
    const double* __restrict__ cross_ij,
    bool cross_transpose,
    const double* __restrict__ other_lam,
    const double* __restrict__ self_eta,
    const double* __restrict__ other_eta,
    double cavity00,
    double cavity01,
    double cavity02,
    double cavity11,
    double cavity12,
    double cavity22,
    const double* __restrict__ cavity_eta,
    const double* __restrict__ old_msg_lam6,
    const double* __restrict__ old_msg_eta,
    double damping,
    double* __restrict__ out_msg_lam6,
    double* __restrict__ out_msg_eta
) {
    const double b00 = cross_ij[0];
    const double b01 = cross_transpose ? cross_ij[3] : cross_ij[1];
    const double b02 = cross_transpose ? cross_ij[6] : cross_ij[2];
    const double b10 = cross_transpose ? cross_ij[1] : cross_ij[3];
    const double b11 = cross_ij[4];
    const double b12 = cross_transpose ? cross_ij[7] : cross_ij[5];
    const double b20 = cross_transpose ? cross_ij[2] : cross_ij[6];
    const double b21 = cross_transpose ? cross_ij[5] : cross_ij[7];
    const double b22 = cross_ij[8];

    const double a00 = other_lam[0] + cavity00 + kGbpJitter;
    const double a10 = other_lam[1] + cavity01;
    const double a20 = other_lam[2] + cavity02;
    const double a11 = other_lam[4] + cavity11 + kGbpJitter;
    const double a21 = other_lam[5] + cavity12;
    const double a22 = other_lam[8] + cavity22 + kGbpJitter;

    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (!(det > 0.0)) {
        return false;
    }

    const double inv_det = 1.0 / det;
    const double s00 = cof00 * inv_det;
    const double s10 = cof10 * inv_det;
    const double s20 = cof20 * inv_det;
    const double s11 = cof11 * inv_det;
    const double s21 = cof21 * inv_det;
    const double s22 = cof22 * inv_det;

    const double eno0 = other_eta[0] + cavity_eta[0];
    const double eno1 = other_eta[1] + cavity_eta[1];
    const double eno2 = other_eta[2] + cavity_eta[2];
    const double y0 = s00 * eno0 + s10 * eno1 + s20 * eno2;
    const double y1 = s10 * eno0 + s11 * eno1 + s21 * eno2;
    const double y2 = s20 * eno0 + s21 * eno1 + s22 * eno2;

    const double z00 = b00 * s00 + b01 * s10 + b02 * s20;
    const double z01 = b00 * s10 + b01 * s11 + b02 * s21;
    const double z02 = b00 * s20 + b01 * s21 + b02 * s22;
    const double z10 = b10 * s00 + b11 * s10 + b12 * s20;
    const double z11 = b10 * s10 + b11 * s11 + b12 * s21;
    const double z12 = b10 * s20 + b11 * s21 + b12 * s22;
    const double z20 = b20 * s00 + b21 * s10 + b22 * s20;
    const double z21 = b20 * s10 + b21 * s11 + b22 * s21;
    const double z22 = b20 * s20 + b21 * s21 + b22 * s22;

    const double schur00 = z00 * b00 + z01 * b01 + z02 * b02;
    const double schur01 = z00 * b10 + z01 * b11 + z02 * b12;
    const double schur02 = z00 * b20 + z01 * b21 + z02 * b22;
    const double schur11 = z10 * b10 + z11 * b11 + z12 * b12;
    const double schur12 = z10 * b20 + z11 * b21 + z12 * b22;
    const double schur22 = z20 * b20 + z21 * b21 + z22 * b22;

    out_msg_lam6[0] = (1.0 - damping) * old_msg_lam6[0] + damping * (self_lam[0] - schur00);
    out_msg_lam6[1] = (1.0 - damping) * old_msg_lam6[1] + damping * (self_lam[1] - schur01);
    out_msg_lam6[2] = (1.0 - damping) * old_msg_lam6[2] + damping * (self_lam[2] - schur02);
    out_msg_lam6[3] = (1.0 - damping) * old_msg_lam6[3] + damping * (self_lam[4] - schur11);
    out_msg_lam6[4] = (1.0 - damping) * old_msg_lam6[4] + damping * (self_lam[5] - schur12);
    out_msg_lam6[5] = (1.0 - damping) * old_msg_lam6[5] + damping * (self_lam[8] - schur22);

    const double eta_proj0 = b00 * y0 + b01 * y1 + b02 * y2;
    const double eta_proj1 = b10 * y0 + b11 * y1 + b12 * y2;
    const double eta_proj2 = b20 * y0 + b21 * y1 + b22 * y2;
    out_msg_eta[0] = (1.0 - damping) * old_msg_eta[0] + damping * (self_eta[0] - eta_proj0);
    out_msg_eta[1] = (1.0 - damping) * old_msg_eta[1] + damping * (self_eta[1] - eta_proj1);
    out_msg_eta[2] = (1.0 - damping) * old_msg_eta[2] + damping * (self_eta[2] - eta_proj2);
    return true;
}

__device__ __forceinline__ void assembleCavityFromIncomingDevice(
    int node,
    int exclude_slot,
    const int* __restrict__ incoming_node_rows,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    const double* __restrict__ msg_lam,
    const double* __restrict__ msg_eta,
    double* __restrict__ cavity_lam,
    double* __restrict__ cavity_eta
) {
    double lam0 = unary_lam[6 * node + 0];
    double lam1 = unary_lam[6 * node + 1];
    double lam2 = unary_lam[6 * node + 2];
    double lam3 = unary_lam[6 * node + 3];
    double lam4 = unary_lam[6 * node + 4];
    double lam5 = unary_lam[6 * node + 5];
    double eta0 = unary_eta[3 * node + 0];
    double eta1 = unary_eta[3 * node + 1];
    double eta2 = unary_eta[3 * node + 2];

    const int row = incoming_node_rows[node];
    const int cap = incoming_caps[row];
    const int offset = incoming_offsets[row];
    for (int s = 0; s < cap; ++s) {
        const int slot = incoming_slots[offset + s];
        if (slot < 0) {
            continue;
        }
        lam0 += msg_lam[6 * slot + 0];
        lam1 += msg_lam[6 * slot + 1];
        lam2 += msg_lam[6 * slot + 2];
        lam3 += msg_lam[6 * slot + 3];
        lam4 += msg_lam[6 * slot + 4];
        lam5 += msg_lam[6 * slot + 5];
        eta0 += msg_eta[3 * slot + 0];
        eta1 += msg_eta[3 * slot + 1];
        eta2 += msg_eta[3 * slot + 2];
    }

    lam0 -= msg_lam[6 * exclude_slot + 0];
    lam1 -= msg_lam[6 * exclude_slot + 1];
    lam2 -= msg_lam[6 * exclude_slot + 2];
    lam3 -= msg_lam[6 * exclude_slot + 3];
    lam4 -= msg_lam[6 * exclude_slot + 4];
    lam5 -= msg_lam[6 * exclude_slot + 5];
    eta0 -= msg_eta[3 * exclude_slot + 0];
    eta1 -= msg_eta[3 * exclude_slot + 1];
    eta2 -= msg_eta[3 * exclude_slot + 2];

    cavity_lam[0] = lam0;
    cavity_lam[1] = lam1;
    cavity_lam[2] = lam2;
    cavity_lam[3] = lam1;
    cavity_lam[4] = lam3;
    cavity_lam[5] = lam4;
    cavity_lam[6] = lam2;
    cavity_lam[7] = lam4;
    cavity_lam[8] = lam5;
    cavity_eta[0] = eta0;
    cavity_eta[1] = eta1;
    cavity_eta[2] = eta2;
}

#if 0
__device__ __forceinline__ void computeGbpMessagePackedInverseDevice(
    const double* self_lam6,
    const double* cross_ij,
    bool cross_transpose,
    const double* other_lam6,
    const double* self_eta,
    const double* other_eta,
    const double* cavity_lam6,
    const double* cavity_eta,
    const double* old_msg_lam6,
    const double* old_msg_eta,
    double damping,
    double* out_msg_lam6,
    double* out_msg_eta
) {
    const double b00 = cross_ij[0];
    const double b01 = cross_transpose ? cross_ij[3] : cross_ij[1];
    const double b02 = cross_transpose ? cross_ij[6] : cross_ij[2];
    const double b10 = cross_transpose ? cross_ij[1] : cross_ij[3];
    const double b11 = cross_ij[4];
    const double b12 = cross_transpose ? cross_ij[7] : cross_ij[5];
    const double b20 = cross_transpose ? cross_ij[2] : cross_ij[6];
    const double b21 = cross_transpose ? cross_ij[5] : cross_ij[7];
    const double b22 = cross_ij[8];

    const double a00 = other_lam6[0] + cavity_lam6[0] + kGbpJitter;
    const double a10 = other_lam6[1] + cavity_lam6[1];
    const double a20 = other_lam6[2] + cavity_lam6[2];
    const double a11 = other_lam6[3] + cavity_lam6[3] + kGbpJitter;
    const double a21 = other_lam6[4] + cavity_lam6[4];
    const double a22 = other_lam6[5] + cavity_lam6[5] + kGbpJitter;

    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double inv_det = 1.0 / (a00 * cof00 + a10 * cof10 + a20 * cof20);
    const double s00 = cof00 * inv_det;
    const double s10 = cof10 * inv_det;
    const double s20 = cof20 * inv_det;
    const double s11 = cof11 * inv_det;
    const double s21 = cof21 * inv_det;
    const double s22 = cof22 * inv_det;

    const double eno0 = other_eta[0] + cavity_eta[0];
    const double eno1 = other_eta[1] + cavity_eta[1];
    const double eno2 = other_eta[2] + cavity_eta[2];
    const double y0 = s00 * eno0 + s10 * eno1 + s20 * eno2;
    const double y1 = s10 * eno0 + s11 * eno1 + s21 * eno2;
    const double y2 = s20 * eno0 + s21 * eno1 + s22 * eno2;

    const double z00 = b00 * s00 + b01 * s10 + b02 * s20;
    const double z01 = b00 * s10 + b01 * s11 + b02 * s21;
    const double z02 = b00 * s20 + b01 * s21 + b02 * s22;
    const double z10 = b10 * s00 + b11 * s10 + b12 * s20;
    const double z11 = b10 * s10 + b11 * s11 + b12 * s21;
    const double z12 = b10 * s20 + b11 * s21 + b12 * s22;
    const double z20 = b20 * s00 + b21 * s10 + b22 * s20;
    const double z21 = b20 * s10 + b21 * s11 + b22 * s21;
    const double z22 = b20 * s20 + b21 * s21 + b22 * s22;

    const double schur00 = z00 * b00 + z01 * b01 + z02 * b02;
    const double schur01 = z00 * b10 + z01 * b11 + z02 * b12;
    const double schur02 = z00 * b20 + z01 * b21 + z02 * b22;
    const double schur11 = z10 * b10 + z11 * b11 + z12 * b12;
    const double schur12 = z10 * b20 + z11 * b21 + z12 * b22;
    const double schur22 = z20 * b20 + z21 * b21 + z22 * b22;

    out_msg_lam6[0] = (1.0 - damping) * old_msg_lam6[0] + damping * (self_lam6[0] - schur00);
    out_msg_lam6[1] = (1.0 - damping) * old_msg_lam6[1] + damping * (self_lam6[1] - schur01);
    out_msg_lam6[2] = (1.0 - damping) * old_msg_lam6[2] + damping * (self_lam6[2] - schur02);
    out_msg_lam6[3] = (1.0 - damping) * old_msg_lam6[3] + damping * (self_lam6[3] - schur11);
    out_msg_lam6[4] = (1.0 - damping) * old_msg_lam6[4] + damping * (self_lam6[4] - schur12);
    out_msg_lam6[5] = (1.0 - damping) * old_msg_lam6[5] + damping * (self_lam6[5] - schur22);

    const double eta_proj0 = b00 * y0 + b01 * y1 + b02 * y2;
    const double eta_proj1 = b10 * y0 + b11 * y1 + b12 * y2;
    const double eta_proj2 = b20 * y0 + b21 * y1 + b22 * y2;
    out_msg_eta[0] = (1.0 - damping) * old_msg_eta[0] + damping * (self_eta[0] - eta_proj0);
    out_msg_eta[1] = (1.0 - damping) * old_msg_eta[1] + damping * (self_eta[1] - eta_proj1);
    out_msg_eta[2] = (1.0 - damping) * old_msg_eta[2] + damping * (self_eta[2] - eta_proj2);
}

__device__ __noinline__ void computeGbpMessagePackedDevice(
    const double* self_lam6,
    const double* cross_ij,
    bool cross_transpose,
    const double* other_lam6,
    const double* self_eta,
    const double* other_eta,
    const double* cavity_lam6,
    const double* cavity_eta,
    const double* old_msg_lam6,
    const double* old_msg_eta,
    double damping,
    int schur_mode,
    double* out_msg_lam6,
    double* out_msg_eta
) {
    const double b00 = cross_ij[0];
    const double b01 = cross_transpose ? cross_ij[3] : cross_ij[1];
    const double b02 = cross_transpose ? cross_ij[6] : cross_ij[2];
    const double b10 = cross_transpose ? cross_ij[1] : cross_ij[3];
    const double b11 = cross_ij[4];
    const double b12 = cross_transpose ? cross_ij[7] : cross_ij[5];
    const double b20 = cross_transpose ? cross_ij[2] : cross_ij[6];
    const double b21 = cross_transpose ? cross_ij[5] : cross_ij[7];
    const double b22 = cross_ij[8];

    const double a00 = other_lam6[0] + cavity_lam6[0] + kGbpJitter;
    const double a10 = other_lam6[1] + cavity_lam6[1];
    const double a20 = other_lam6[2] + cavity_lam6[2];
    const double a11 = other_lam6[3] + cavity_lam6[3] + kGbpJitter;
    const double a21 = other_lam6[4] + cavity_lam6[4];
    const double a22 = other_lam6[5] + cavity_lam6[5] + kGbpJitter;
    const double eno0 = other_eta[0] + cavity_eta[0];
    const double eno1 = other_eta[1] + cavity_eta[1];
    const double eno2 = other_eta[2] + cavity_eta[2];

    double y0 = 0.0;
    double y1 = 0.0;
    double y2 = 0.0;
    double z00 = 0.0;
    double z10 = 0.0;
    double z20 = 0.0;
    double z01 = 0.0;
    double z11 = 0.0;
    double z21 = 0.0;
    double z02 = 0.0;
    double z12 = 0.0;
    double z22 = 0.0;

    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (schur_mode == 0 && det > 0.0) {
        const double inv_det = 1.0 / det;
        const double s00 = cof00 * inv_det;
        const double s10 = cof10 * inv_det;
        const double s20 = cof20 * inv_det;
        const double s11 = cof11 * inv_det;
        const double s21 = cof21 * inv_det;
        const double s22 = cof22 * inv_det;
        y0 = s00 * eno0 + s10 * eno1 + s20 * eno2;
        y1 = s10 * eno0 + s11 * eno1 + s21 * eno2;
        y2 = s20 * eno0 + s21 * eno1 + s22 * eno2;
        z00 = b00 * s00 + b01 * s10 + b02 * s20;
        z01 = b00 * s10 + b01 * s11 + b02 * s21;
        z02 = b00 * s20 + b01 * s21 + b02 * s22;
        z10 = b10 * s00 + b11 * s10 + b12 * s20;
        z11 = b10 * s10 + b11 * s11 + b12 * s21;
        z12 = b10 * s20 + b11 * s21 + b12 * s22;
        z20 = b20 * s00 + b21 * s10 + b22 * s20;
        z21 = b20 * s10 + b21 * s11 + b22 * s21;
        z22 = b20 * s20 + b21 * s21 + b22 * s22;
    } else {
        const double cond[9] = {
            a00, a10, a20,
            a10, a11, a21,
            a20, a21, a22
        };
        Chol3Device chol{};
        if (factorize3LowerNoJitterDevice(cond, chol)) {
            const double rhs_eta[3] = {eno0, eno1, eno2};
            double sol[3];
            solve3LowerDevice(chol, rhs_eta, sol);
            y0 = sol[0];
            y1 = sol[1];
            y2 = sol[2];
            const double rhs0[3] = {b00, b01, b02};
            const double rhs1[3] = {b10, b11, b12};
            const double rhs2[3] = {b20, b21, b22};
            solve3LowerDevice(chol, rhs0, sol);
            z00 = sol[0];
            z01 = sol[1];
            z02 = sol[2];
            solve3LowerDevice(chol, rhs1, sol);
            z10 = sol[0];
            z11 = sol[1];
            z12 = sol[2];
            solve3LowerDevice(chol, rhs2, sol);
            z20 = sol[0];
            z21 = sol[1];
            z22 = sol[2];
        } else {
            const double d0 = fmax(fabs(a00), kGbpJitter);
            const double d1 = fmax(fabs(a11), kGbpJitter);
            const double d2 = fmax(fabs(a22), kGbpJitter);
            y0 = eno0 / d0;
            y1 = eno1 / d1;
            y2 = eno2 / d2;
            z00 = b00 / d0;
            z01 = b01 / d1;
            z02 = b02 / d2;
            z10 = b10 / d0;
            z11 = b11 / d1;
            z12 = b12 / d2;
            z20 = b20 / d0;
            z21 = b21 / d1;
            z22 = b22 / d2;
        }
    }

    const double schur00 = z00 * b00 + z01 * b01 + z02 * b02;
    const double schur01 = z00 * b10 + z01 * b11 + z02 * b12;
    const double schur02 = z00 * b20 + z01 * b21 + z02 * b22;
    const double schur11 = z10 * b10 + z11 * b11 + z12 * b12;
    const double schur12 = z10 * b20 + z11 * b21 + z12 * b22;
    const double schur22 = z20 * b20 + z21 * b21 + z22 * b22;
    out_msg_lam6[0] = (1.0 - damping) * old_msg_lam6[0] + damping * (self_lam6[0] - schur00);
    out_msg_lam6[1] = (1.0 - damping) * old_msg_lam6[1] + damping * (self_lam6[1] - schur01);
    out_msg_lam6[2] = (1.0 - damping) * old_msg_lam6[2] + damping * (self_lam6[2] - schur02);
    out_msg_lam6[3] = (1.0 - damping) * old_msg_lam6[3] + damping * (self_lam6[3] - schur11);
    out_msg_lam6[4] = (1.0 - damping) * old_msg_lam6[4] + damping * (self_lam6[4] - schur12);
    out_msg_lam6[5] = (1.0 - damping) * old_msg_lam6[5] + damping * (self_lam6[5] - schur22);

    const double eta_proj0 = b00 * y0 + b01 * y1 + b02 * y2;
    const double eta_proj1 = b10 * y0 + b11 * y1 + b12 * y2;
    const double eta_proj2 = b20 * y0 + b21 * y1 + b22 * y2;
    out_msg_eta[0] = (1.0 - damping) * old_msg_eta[0] + damping * (self_eta[0] - eta_proj0);
    out_msg_eta[1] = (1.0 - damping) * old_msg_eta[1] + damping * (self_eta[1] - eta_proj1);
    out_msg_eta[2] = (1.0 - damping) * old_msg_eta[2] + damping * (self_eta[2] - eta_proj2);
}

#endif

__global__ void gbpPersistentKernel(
    int n,
    int m,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double* __restrict__ fixed_eta_map,
    double damping,
    int schur_mode,
    int fixed_lambda_start
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_lam_new = even ? belief_lam_b : belief_lam_a;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        const bool eta_only = fixed_lambda_start >= 0 && sweep > fixed_lambda_start;
        const bool first_eta_only = eta_only && sweep == fixed_lambda_start + 1;

        for (int e = tid; e < m; e += stride) {
            const int i = factor_i[e];
            const int j = factor_j[e];
            const int slot_i = 2 * e;
            const int slot_j = slot_i + 1;

            double cavity_eta_j[3];
            double cavity_eta_i[3];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                cavity_eta_j[k] = belief_eta_old[3 * j + k] - msg_eta_old[3 * slot_j + k];
                cavity_eta_i[k] = belief_eta_old[3 * i + k] - msg_eta_old[3 * slot_i + k];
            }

            if (eta_only) {
                if (first_eta_only) {
                    double cavity_lam_j[9];
                    const double cj00 = belief_lam_old[6 * j + 0] - msg_lam_old[6 * slot_j + 0];
                    const double cj01 = belief_lam_old[6 * j + 1] - msg_lam_old[6 * slot_j + 1];
                    const double cj02 = belief_lam_old[6 * j + 2] - msg_lam_old[6 * slot_j + 2];
                    const double cj11 = belief_lam_old[6 * j + 3] - msg_lam_old[6 * slot_j + 3];
                    const double cj12 = belief_lam_old[6 * j + 4] - msg_lam_old[6 * slot_j + 4];
                    const double cj22 = belief_lam_old[6 * j + 5] - msg_lam_old[6 * slot_j + 5];
                    cavity_lam_j[0] = cj00;
                    cavity_lam_j[1] = cj01;
                    cavity_lam_j[2] = cj02;
                    cavity_lam_j[3] = cj01;
                    cavity_lam_j[4] = cj11;
                    cavity_lam_j[5] = cj12;
                    cavity_lam_j[6] = cj02;
                    cavity_lam_j[7] = cj12;
                    cavity_lam_j[8] = cj22;
                    buildEtaMapDevice(
                        hij + 9 * e, hjj + 9 * e, cavity_lam_j, schur_mode,
                        fixed_eta_map + 9 * slot_i);

                    double cavity_lam_i[9];
                    const double ci00 = belief_lam_old[6 * i + 0] - msg_lam_old[6 * slot_i + 0];
                    const double ci01 = belief_lam_old[6 * i + 1] - msg_lam_old[6 * slot_i + 1];
                    const double ci02 = belief_lam_old[6 * i + 2] - msg_lam_old[6 * slot_i + 2];
                    const double ci11 = belief_lam_old[6 * i + 3] - msg_lam_old[6 * slot_i + 3];
                    const double ci12 = belief_lam_old[6 * i + 4] - msg_lam_old[6 * slot_i + 4];
                    const double ci22 = belief_lam_old[6 * i + 5] - msg_lam_old[6 * slot_i + 5];
                    cavity_lam_i[0] = ci00;
                    cavity_lam_i[1] = ci01;
                    cavity_lam_i[2] = ci02;
                    cavity_lam_i[3] = ci01;
                    cavity_lam_i[4] = ci11;
                    cavity_lam_i[5] = ci12;
                    cavity_lam_i[6] = ci02;
                    cavity_lam_i[7] = ci12;
                    cavity_lam_i[8] = ci22;
                    buildEtaMapDevice(
                        hji + 9 * e, hii + 9 * e, cavity_lam_i, schur_mode,
                        fixed_eta_map + 9 * slot_j);
                }
                computeEtaOnlyMessageDevice(
                    eta_i + 3 * e, eta_j + 3 * e,
                    cavity_eta_j,
                    fixed_eta_map + 9 * slot_i,
                    msg_eta_old + 3 * slot_i,
                    msg_eta_new + 3 * slot_i);
                computeEtaOnlyMessageDevice(
                    eta_j + 3 * e, eta_i + 3 * e,
                    cavity_eta_i,
                    fixed_eta_map + 9 * slot_j,
                    msg_eta_old + 3 * slot_j,
                    msg_eta_new + 3 * slot_j);
            } else {
                double cavity_lam_j[9];
                double cavity_lam_i[9];
                const double cj00 = belief_lam_old[6 * j + 0] - msg_lam_old[6 * slot_j + 0];
                const double cj01 = belief_lam_old[6 * j + 1] - msg_lam_old[6 * slot_j + 1];
                const double cj02 = belief_lam_old[6 * j + 2] - msg_lam_old[6 * slot_j + 2];
                const double cj11 = belief_lam_old[6 * j + 3] - msg_lam_old[6 * slot_j + 3];
                const double cj12 = belief_lam_old[6 * j + 4] - msg_lam_old[6 * slot_j + 4];
                const double cj22 = belief_lam_old[6 * j + 5] - msg_lam_old[6 * slot_j + 5];
                cavity_lam_j[0] = cj00;
                cavity_lam_j[1] = cj01;
                cavity_lam_j[2] = cj02;
                cavity_lam_j[3] = cj01;
                cavity_lam_j[4] = cj11;
                cavity_lam_j[5] = cj12;
                cavity_lam_j[6] = cj02;
                cavity_lam_j[7] = cj12;
                cavity_lam_j[8] = cj22;
                const double ci00 = belief_lam_old[6 * i + 0] - msg_lam_old[6 * slot_i + 0];
                const double ci01 = belief_lam_old[6 * i + 1] - msg_lam_old[6 * slot_i + 1];
                const double ci02 = belief_lam_old[6 * i + 2] - msg_lam_old[6 * slot_i + 2];
                const double ci11 = belief_lam_old[6 * i + 3] - msg_lam_old[6 * slot_i + 3];
                const double ci12 = belief_lam_old[6 * i + 4] - msg_lam_old[6 * slot_i + 4];
                const double ci22 = belief_lam_old[6 * i + 5] - msg_lam_old[6 * slot_i + 5];
                cavity_lam_i[0] = ci00;
                cavity_lam_i[1] = ci01;
                cavity_lam_i[2] = ci02;
                cavity_lam_i[3] = ci01;
                cavity_lam_i[4] = ci11;
                cavity_lam_i[5] = ci12;
                cavity_lam_i[6] = ci02;
                cavity_lam_i[7] = ci12;
                cavity_lam_i[8] = ci22;
                computeGbpMessageDevice(
                    hii + 9 * e, hij + 9 * e, hji + 9 * e, hjj + 9 * e,
                    eta_i + 3 * e, eta_j + 3 * e,
                    cavity_lam_j, cavity_eta_j,
                    msg_lam_old + 6 * slot_i, msg_eta_old + 3 * slot_i,
                    damping,
                    schur_mode,
                    msg_lam_new + 6 * slot_i, msg_eta_new + 3 * slot_i);

                computeGbpMessageDevice(
                    hjj + 9 * e, hji + 9 * e, hij + 9 * e, hii + 9 * e,
                    eta_j + 3 * e, eta_i + 3 * e,
                    cavity_lam_i, cavity_eta_i,
                    msg_lam_old + 6 * slot_j, msg_eta_old + 3 * slot_j,
                    damping,
                    schur_mode,
                    msg_lam_new + 6 * slot_j, msg_eta_new + 3 * slot_j);
            }
        }

        grid.sync();

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta[3];
            double lam6[6];
            if (eta_only && first_eta_only) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    lam6[k] = belief_lam_old[6 * node + k];
                }
            } else if (!eta_only) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    lam6[k] = unary_lam[6 * node + k];
                }
            }
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                eta[k] = unary_eta[3 * node + k];
            }
            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming_slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                if (!eta_only) {
                    lam6[0] += msg_lam_new[6 * slot + 0];
                    lam6[1] += msg_lam_new[6 * slot + 1];
                    lam6[2] += msg_lam_new[6 * slot + 2];
                    lam6[3] += msg_lam_new[6 * slot + 3];
                    lam6[4] += msg_lam_new[6 * slot + 4];
                    lam6[5] += msg_lam_new[6 * slot + 5];
                }
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    eta[k] += msg_eta_new[3 * slot + k];
                }
            }
            if (!eta_only || first_eta_only) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    belief_lam_new[6 * node + k] = lam6[k];
                }
            }
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                belief_eta_new[3 * node + k] = eta[k];
            }
        }

        if (sweep + 1 < sweeps) {
            grid.sync();
        }
    }
}

__global__ void gbpSplitMessageKernel(
    int m,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const double* __restrict__ msg_lam_old,
    const double* __restrict__ msg_eta_old,
    const double* __restrict__ belief_lam_old,
    const double* __restrict__ belief_eta_old,
    double* __restrict__ msg_lam_new,
    double* __restrict__ msg_eta_new,
    double damping,
    int schur_mode
) {
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int e = tid; e < m; e += stride) {
        const int i = factor_i[e];
        const int j = factor_j[e];
        const int slot_i = 2 * e;
        const int slot_j = slot_i + 1;

        double cavity_eta_j[3];
        double cavity_eta_i[3];
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            cavity_eta_j[k] = belief_eta_old[3 * j + k] - msg_eta_old[3 * slot_j + k];
            cavity_eta_i[k] = belief_eta_old[3 * i + k] - msg_eta_old[3 * slot_i + k];
        }

        double cavity_lam_j[9];
        double cavity_lam_i[9];
        const double cj00 = belief_lam_old[6 * j + 0] - msg_lam_old[6 * slot_j + 0];
        const double cj01 = belief_lam_old[6 * j + 1] - msg_lam_old[6 * slot_j + 1];
        const double cj02 = belief_lam_old[6 * j + 2] - msg_lam_old[6 * slot_j + 2];
        const double cj11 = belief_lam_old[6 * j + 3] - msg_lam_old[6 * slot_j + 3];
        const double cj12 = belief_lam_old[6 * j + 4] - msg_lam_old[6 * slot_j + 4];
        const double cj22 = belief_lam_old[6 * j + 5] - msg_lam_old[6 * slot_j + 5];
        cavity_lam_j[0] = cj00;
        cavity_lam_j[1] = cj01;
        cavity_lam_j[2] = cj02;
        cavity_lam_j[3] = cj01;
        cavity_lam_j[4] = cj11;
        cavity_lam_j[5] = cj12;
        cavity_lam_j[6] = cj02;
        cavity_lam_j[7] = cj12;
        cavity_lam_j[8] = cj22;
        const double ci00 = belief_lam_old[6 * i + 0] - msg_lam_old[6 * slot_i + 0];
        const double ci01 = belief_lam_old[6 * i + 1] - msg_lam_old[6 * slot_i + 1];
        const double ci02 = belief_lam_old[6 * i + 2] - msg_lam_old[6 * slot_i + 2];
        const double ci11 = belief_lam_old[6 * i + 3] - msg_lam_old[6 * slot_i + 3];
        const double ci12 = belief_lam_old[6 * i + 4] - msg_lam_old[6 * slot_i + 4];
        const double ci22 = belief_lam_old[6 * i + 5] - msg_lam_old[6 * slot_i + 5];
        cavity_lam_i[0] = ci00;
        cavity_lam_i[1] = ci01;
        cavity_lam_i[2] = ci02;
        cavity_lam_i[3] = ci01;
        cavity_lam_i[4] = ci11;
        cavity_lam_i[5] = ci12;
        cavity_lam_i[6] = ci02;
        cavity_lam_i[7] = ci12;
        cavity_lam_i[8] = ci22;

        computeGbpMessageDevice(
            hii + 9 * e, hij + 9 * e, hji + 9 * e, hjj + 9 * e,
            eta_i + 3 * e, eta_j + 3 * e,
            cavity_lam_j, cavity_eta_j,
            msg_lam_old + 6 * slot_i, msg_eta_old + 3 * slot_i,
            damping,
            schur_mode,
            msg_lam_new + 6 * slot_i, msg_eta_new + 3 * slot_i);

        computeGbpMessageDevice(
            hjj + 9 * e, hji + 9 * e, hij + 9 * e, hii + 9 * e,
            eta_j + 3 * e, eta_i + 3 * e,
            cavity_lam_i, cavity_eta_i,
            msg_lam_old + 6 * slot_j, msg_eta_old + 3 * slot_j,
            damping,
            schur_mode,
            msg_lam_new + 6 * slot_j, msg_eta_new + 3 * slot_j);
    }
}

__global__ void gbpSplitBeliefKernel(
    int n,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    const double* __restrict__ msg_lam_new,
    const double* __restrict__ msg_eta_new,
    double* __restrict__ belief_lam_new,
    double* __restrict__ belief_eta_new
) {
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int row = tid; row < n; row += stride) {
        const int node = incoming_nodes[row];
        double lam6[6];
        double eta[3];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            lam6[k] = unary_lam[6 * node + k];
        }
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            eta[k] = unary_eta[3 * node + k];
        }
        const int cap = incoming_caps[row];
        const int offset = incoming_offsets[row];
        for (int s = 0; s < cap; ++s) {
            const int slot = incoming_slots[offset + s];
            if (slot < 0) {
                continue;
            }
            lam6[0] += msg_lam_new[6 * slot + 0];
            lam6[1] += msg_lam_new[6 * slot + 1];
            lam6[2] += msg_lam_new[6 * slot + 2];
            lam6[3] += msg_lam_new[6 * slot + 3];
            lam6[4] += msg_lam_new[6 * slot + 4];
            lam6[5] += msg_lam_new[6 * slot + 5];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                eta[k] += msg_eta_new[3 * slot + k];
            }
        }
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            belief_lam_new[6 * node + k] = lam6[k];
        }
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            belief_eta_new[3 * node + k] = eta[k];
        }
    }
}

__global__ void gbpOnTheFlyCavityPersistentKernel(
    int n,
    int m,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_node_rows,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double damping,
    int schur_mode
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;

        for (int e = tid; e < m; e += stride) {
            const int i = factor_i[e];
            const int j = factor_j[e];
            const int slot_i = 2 * e;
            const int slot_j = slot_i + 1;

            double cavity_lam_j[9];
            double cavity_lam_i[9];
            double cavity_eta_j[3];
            double cavity_eta_i[3];
            assembleCavityFromIncomingDevice(
                j, slot_j,
                incoming_node_rows, incoming_caps, incoming_offsets, incoming_slots,
                unary_lam, unary_eta, msg_lam_old, msg_eta_old,
                cavity_lam_j, cavity_eta_j);
            assembleCavityFromIncomingDevice(
                i, slot_i,
                incoming_node_rows, incoming_caps, incoming_offsets, incoming_slots,
                unary_lam, unary_eta, msg_lam_old, msg_eta_old,
                cavity_lam_i, cavity_eta_i);

            computeGbpMessageDevice(
                hii + 9 * e, hij + 9 * e, hji + 9 * e, hjj + 9 * e,
                eta_i + 3 * e, eta_j + 3 * e,
                cavity_lam_j, cavity_eta_j,
                msg_lam_old + 6 * slot_i, msg_eta_old + 3 * slot_i,
                damping,
                schur_mode,
                msg_lam_new + 6 * slot_i, msg_eta_new + 3 * slot_i);

            computeGbpMessageDevice(
                hjj + 9 * e, hji + 9 * e, hij + 9 * e, hii + 9 * e,
                eta_j + 3 * e, eta_i + 3 * e,
                cavity_lam_i, cavity_eta_i,
                msg_lam_old + 6 * slot_j, msg_eta_old + 3 * slot_j,
                damping,
                schur_mode,
                msg_lam_new + 6 * slot_j, msg_eta_new + 3 * slot_j);
        }

        grid.sync();
    }

    const bool even_sweeps = (sweeps & 1) == 0;
    const double* msg_lam_final = even_sweeps ? msg_lam_a : msg_lam_b;
    const double* msg_eta_final = even_sweeps ? msg_eta_a : msg_eta_b;
    double* belief_lam_final = even_sweeps ? belief_lam_a : belief_lam_b;
    double* belief_eta_final = even_sweeps ? belief_eta_a : belief_eta_b;
    for (int row = tid; row < n; row += stride) {
        const int node = incoming_nodes[row];
        double eta0 = unary_eta[3 * node + 0];
        double eta1 = unary_eta[3 * node + 1];
        double eta2 = unary_eta[3 * node + 2];
        double lam0 = unary_lam[6 * node + 0];
        double lam1 = unary_lam[6 * node + 1];
        double lam2 = unary_lam[6 * node + 2];
        double lam3 = unary_lam[6 * node + 3];
        double lam4 = unary_lam[6 * node + 4];
        double lam5 = unary_lam[6 * node + 5];
        const int cap = incoming_caps[row];
        const int offset = incoming_offsets[row];
        for (int s = 0; s < cap; ++s) {
            const int slot = incoming_slots[offset + s];
            if (slot < 0) {
                continue;
            }
            lam0 += msg_lam_final[6 * slot + 0];
            lam1 += msg_lam_final[6 * slot + 1];
            lam2 += msg_lam_final[6 * slot + 2];
            lam3 += msg_lam_final[6 * slot + 3];
            lam4 += msg_lam_final[6 * slot + 4];
            lam5 += msg_lam_final[6 * slot + 5];
            eta0 += msg_eta_final[3 * slot + 0];
            eta1 += msg_eta_final[3 * slot + 1];
            eta2 += msg_eta_final[3 * slot + 2];
        }
        belief_lam_final[6 * node + 0] = lam0;
        belief_lam_final[6 * node + 1] = lam1;
        belief_lam_final[6 * node + 2] = lam2;
        belief_lam_final[6 * node + 3] = lam3;
        belief_lam_final[6 * node + 4] = lam4;
        belief_lam_final[6 * node + 5] = lam5;
        belief_eta_final[3 * node + 0] = eta0;
        belief_eta_final[3 * node + 1] = eta1;
        belief_eta_final[3 * node + 2] = eta2;
    }
}

__global__ void gbpFullPersistentKernel(
    int n,
    int m,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double damping,
    int schur_mode
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_lam_new = even ? belief_lam_b : belief_lam_a;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        for (int e = tid; e < m; e += stride) {
            const int i = factor_i[e];
            const int j = factor_j[e];
            const int slot_i = 2 * e;
            const int slot_j = slot_i + 1;

            double cavity_eta_j[3];
            double cavity_eta_i[3];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                cavity_eta_j[k] = belief_eta_old[3 * j + k] - msg_eta_old[3 * slot_j + k];
                cavity_eta_i[k] = belief_eta_old[3 * i + k] - msg_eta_old[3 * slot_i + k];
            }

            double cavity_lam_j[9];
            double cavity_lam_i[9];
            const double cj00 = belief_lam_old[6 * j + 0] - msg_lam_old[6 * slot_j + 0];
            const double cj01 = belief_lam_old[6 * j + 1] - msg_lam_old[6 * slot_j + 1];
            const double cj02 = belief_lam_old[6 * j + 2] - msg_lam_old[6 * slot_j + 2];
            const double cj11 = belief_lam_old[6 * j + 3] - msg_lam_old[6 * slot_j + 3];
            const double cj12 = belief_lam_old[6 * j + 4] - msg_lam_old[6 * slot_j + 4];
            const double cj22 = belief_lam_old[6 * j + 5] - msg_lam_old[6 * slot_j + 5];
            cavity_lam_j[0] = cj00;
            cavity_lam_j[1] = cj01;
            cavity_lam_j[2] = cj02;
            cavity_lam_j[3] = cj01;
            cavity_lam_j[4] = cj11;
            cavity_lam_j[5] = cj12;
            cavity_lam_j[6] = cj02;
            cavity_lam_j[7] = cj12;
            cavity_lam_j[8] = cj22;
            const double ci00 = belief_lam_old[6 * i + 0] - msg_lam_old[6 * slot_i + 0];
            const double ci01 = belief_lam_old[6 * i + 1] - msg_lam_old[6 * slot_i + 1];
            const double ci02 = belief_lam_old[6 * i + 2] - msg_lam_old[6 * slot_i + 2];
            const double ci11 = belief_lam_old[6 * i + 3] - msg_lam_old[6 * slot_i + 3];
            const double ci12 = belief_lam_old[6 * i + 4] - msg_lam_old[6 * slot_i + 4];
            const double ci22 = belief_lam_old[6 * i + 5] - msg_lam_old[6 * slot_i + 5];
            cavity_lam_i[0] = ci00;
            cavity_lam_i[1] = ci01;
            cavity_lam_i[2] = ci02;
            cavity_lam_i[3] = ci01;
            cavity_lam_i[4] = ci11;
            cavity_lam_i[5] = ci12;
            cavity_lam_i[6] = ci02;
            cavity_lam_i[7] = ci12;
            cavity_lam_i[8] = ci22;

            (void)computeGbpMessagePackedInverseFromFullDevice(
                hii + 9 * e, hij + 9 * e, false, hjj + 9 * e,
                eta_i + 3 * e, eta_j + 3 * e,
                cj00, cj01, cj02, cj11, cj12, cj22,
                cavity_eta_j,
                msg_lam_old + 6 * slot_i, msg_eta_old + 3 * slot_i,
                damping,
                msg_lam_new + 6 * slot_i, msg_eta_new + 3 * slot_i);

            (void)computeGbpMessagePackedInverseFromFullDevice(
                hjj + 9 * e, hij + 9 * e, true, hii + 9 * e,
                eta_j + 3 * e, eta_i + 3 * e,
                ci00, ci01, ci02, ci11, ci12, ci22,
                cavity_eta_i,
                msg_lam_old + 6 * slot_j, msg_eta_old + 3 * slot_j,
                damping,
                msg_lam_new + 6 * slot_j, msg_eta_new + 3 * slot_j);
        }

        grid.sync();

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta0 = unary_eta[3 * node + 0];
            double eta1 = unary_eta[3 * node + 1];
            double eta2 = unary_eta[3 * node + 2];
            double lam0 = unary_lam[6 * node + 0];
            double lam1 = unary_lam[6 * node + 1];
            double lam2 = unary_lam[6 * node + 2];
            double lam3 = unary_lam[6 * node + 3];
            double lam4 = unary_lam[6 * node + 4];
            double lam5 = unary_lam[6 * node + 5];
            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming_slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                lam0 += msg_lam_new[6 * slot + 0];
                lam1 += msg_lam_new[6 * slot + 1];
                lam2 += msg_lam_new[6 * slot + 2];
                lam3 += msg_lam_new[6 * slot + 3];
                lam4 += msg_lam_new[6 * slot + 4];
                lam5 += msg_lam_new[6 * slot + 5];
                eta0 += msg_eta_new[3 * slot + 0];
                eta1 += msg_eta_new[3 * slot + 1];
                eta2 += msg_eta_new[3 * slot + 2];
            }
            belief_lam_new[6 * node + 0] = lam0;
            belief_lam_new[6 * node + 1] = lam1;
            belief_lam_new[6 * node + 2] = lam2;
            belief_lam_new[6 * node + 3] = lam3;
            belief_lam_new[6 * node + 4] = lam4;
            belief_lam_new[6 * node + 5] = lam5;
            belief_eta_new[3 * node + 0] = eta0;
            belief_eta_new[3 * node + 1] = eta1;
            belief_eta_new[3 * node + 2] = eta2;
        }

        grid.sync();
    }
}

__global__ void gbpDirectedPersistentKernel(
    int n,
    int m,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double* __restrict__ fixed_eta_map,
    double damping,
    int schur_mode,
    int fixed_lambda_start
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;
    const int directed_count = 2 * m;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_lam_new = even ? belief_lam_b : belief_lam_a;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        const bool eta_only = fixed_lambda_start >= 0 && sweep > fixed_lambda_start;
        const bool first_eta_only = eta_only && sweep == fixed_lambda_start + 1;

        for (int slot = tid; slot < directed_count; slot += stride) {
            const int e = slot >> 1;
            const bool target_is_i = (slot & 1) == 0;
            const int target = target_is_i ? factor_i[e] : factor_j[e];
            const int source = target_is_i ? factor_j[e] : factor_i[e];
            const int source_slot = target_is_i ? (slot + 1) : (slot - 1);
            (void)target;

            const double* self_lam = target_is_i ? (hii + 9 * e) : (hjj + 9 * e);
            const double* cross_lam = target_is_i ? (hij + 9 * e) : (hji + 9 * e);
            const double* cross_lam_t = target_is_i ? (hji + 9 * e) : (hij + 9 * e);
            const double* other_lam = target_is_i ? (hjj + 9 * e) : (hii + 9 * e);
            const double* self_eta = target_is_i ? (eta_i + 3 * e) : (eta_j + 3 * e);
            const double* other_eta = target_is_i ? (eta_j + 3 * e) : (eta_i + 3 * e);

            double cavity_eta[3] = {
                belief_eta_old[3 * source + 0] - msg_eta_old[3 * source_slot + 0],
                belief_eta_old[3 * source + 1] - msg_eta_old[3 * source_slot + 1],
                belief_eta_old[3 * source + 2] - msg_eta_old[3 * source_slot + 2]
            };

            if (eta_only) {
                if (first_eta_only) {
                    const double c00 = belief_lam_old[6 * source + 0] - msg_lam_old[6 * source_slot + 0];
                    const double c01 = belief_lam_old[6 * source + 1] - msg_lam_old[6 * source_slot + 1];
                    const double c02 = belief_lam_old[6 * source + 2] - msg_lam_old[6 * source_slot + 2];
                    const double c11 = belief_lam_old[6 * source + 3] - msg_lam_old[6 * source_slot + 3];
                    const double c12 = belief_lam_old[6 * source + 4] - msg_lam_old[6 * source_slot + 4];
                    const double c22 = belief_lam_old[6 * source + 5] - msg_lam_old[6 * source_slot + 5];
                    double cavity_lam[9] = {
                        c00, c01, c02,
                        c01, c11, c12,
                        c02, c12, c22
                    };
                    buildEtaMapDevice(
                        cross_lam,
                        other_lam,
                        cavity_lam,
                        schur_mode,
                        fixed_eta_map + 9 * slot);
                }
                computeEtaOnlyMessageDevice(
                    self_eta,
                    other_eta,
                    cavity_eta,
                    fixed_eta_map + 9 * slot,
                    msg_eta_old + 3 * slot,
                    msg_eta_new + 3 * slot);
            } else {
                const double c00 = belief_lam_old[6 * source + 0] - msg_lam_old[6 * source_slot + 0];
                const double c01 = belief_lam_old[6 * source + 1] - msg_lam_old[6 * source_slot + 1];
                const double c02 = belief_lam_old[6 * source + 2] - msg_lam_old[6 * source_slot + 2];
                const double c11 = belief_lam_old[6 * source + 3] - msg_lam_old[6 * source_slot + 3];
                const double c12 = belief_lam_old[6 * source + 4] - msg_lam_old[6 * source_slot + 4];
                const double c22 = belief_lam_old[6 * source + 5] - msg_lam_old[6 * source_slot + 5];
                double cavity_lam[9] = {
                    c00, c01, c02,
                    c01, c11, c12,
                    c02, c12, c22
                };
                computeGbpMessageDevice(
                    self_lam,
                    cross_lam,
                    cross_lam_t,
                    other_lam,
                    self_eta,
                    other_eta,
                    cavity_lam,
                    cavity_eta,
                    msg_lam_old + 6 * slot,
                    msg_eta_old + 3 * slot,
                    damping,
                    schur_mode,
                    msg_lam_new + 6 * slot,
                    msg_eta_new + 3 * slot);
            }
        }

        grid.sync();

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta[3];
            double lam6[6];
            if (eta_only && first_eta_only) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    lam6[k] = belief_lam_old[6 * node + k];
                }
            } else if (!eta_only) {
                #pragma unroll
                for (int k = 0; k < 6; ++k) {
                    lam6[k] = unary_lam[6 * node + k];
                }
            }
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                eta[k] = unary_eta[3 * node + k];
            }
            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming_slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                if (!eta_only) {
                    lam6[0] += msg_lam_new[6 * slot + 0];
                    lam6[1] += msg_lam_new[6 * slot + 1];
                    lam6[2] += msg_lam_new[6 * slot + 2];
                    lam6[3] += msg_lam_new[6 * slot + 3];
                    lam6[4] += msg_lam_new[6 * slot + 4];
                    lam6[5] += msg_lam_new[6 * slot + 5];
                }
                eta[0] += msg_eta_new[3 * slot + 0];
                eta[1] += msg_eta_new[3 * slot + 1];
                eta[2] += msg_eta_new[3 * slot + 2];
            }
            if (!eta_only || first_eta_only) {
                belief_lam_new[6 * node + 0] = lam6[0];
                belief_lam_new[6 * node + 1] = lam6[1];
                belief_lam_new[6 * node + 2] = lam6[2];
                belief_lam_new[6 * node + 3] = lam6[3];
                belief_lam_new[6 * node + 4] = lam6[4];
                belief_lam_new[6 * node + 5] = lam6[5];
            }
            belief_eta_new[3 * node + 0] = eta[0];
            belief_eta_new[3 * node + 1] = eta[1];
            belief_eta_new[3 * node + 2] = eta[2];
        }

        grid.sync();
    }
}

__global__ void gbpWarpDirectedPersistentKernel(
    int n,
    int m,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double damping
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;
    const int lane = threadIdx.x & 31;
    const int warp = tid >> 5;
    const int warp_stride = stride >> 5;
    const unsigned mask = 0xffffffffu;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_lam_new = even ? belief_lam_b : belief_lam_a;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        for (int slot = warp; slot < 2 * m; slot += warp_stride) {
            const int e = slot >> 1;
            const bool to_i = (slot & 1) == 0;
            const int i = factor_i[e];
            const int j = factor_j[e];
            const int self = to_i ? i : j;
            const int other = to_i ? j : i;
            const int other_slot = slot ^ 1;

            const double* self_lam = to_i ? (hii + 9 * e) : (hjj + 9 * e);
            const double* other_lam = to_i ? (hjj + 9 * e) : (hii + 9 * e);
            const double* self_eta = to_i ? (eta_i + 3 * e) : (eta_j + 3 * e);
            const double* other_eta = to_i ? (eta_j + 3 * e) : (eta_i + 3 * e);
            const double* cross = hij + 9 * e;

            const double b00 = cross[0];
            const double b01 = to_i ? cross[1] : cross[3];
            const double b02 = to_i ? cross[2] : cross[6];
            const double b10 = to_i ? cross[3] : cross[1];
            const double b11 = cross[4];
            const double b12 = to_i ? cross[5] : cross[7];
            const double b20 = to_i ? cross[6] : cross[2];
            const double b21 = to_i ? cross[7] : cross[5];
            const double b22 = cross[8];

            const double c00 = belief_lam_old[6 * other + 0] - msg_lam_old[6 * other_slot + 0];
            const double c01 = belief_lam_old[6 * other + 1] - msg_lam_old[6 * other_slot + 1];
            const double c02 = belief_lam_old[6 * other + 2] - msg_lam_old[6 * other_slot + 2];
            const double c11 = belief_lam_old[6 * other + 3] - msg_lam_old[6 * other_slot + 3];
            const double c12 = belief_lam_old[6 * other + 4] - msg_lam_old[6 * other_slot + 4];
            const double c22 = belief_lam_old[6 * other + 5] - msg_lam_old[6 * other_slot + 5];
            const double a00 = other_lam[0] + c00 + kGbpJitter;
            const double a10 = other_lam[1] + c01;
            const double a20 = other_lam[2] + c02;
            const double a11 = other_lam[4] + c11 + kGbpJitter;
            const double a21 = other_lam[5] + c12;
            const double a22 = other_lam[8] + c22 + kGbpJitter;

            double cof = 0.0;
            if (lane == 0) cof = a11 * a22 - a21 * a21;
            if (lane == 1) cof = a20 * a21 - a10 * a22;
            if (lane == 2) cof = a10 * a21 - a20 * a11;
            if (lane == 3) cof = a00 * a22 - a20 * a20;
            if (lane == 4) cof = a10 * a20 - a00 * a21;
            if (lane == 5) cof = a00 * a11 - a10 * a10;
            const double cof00 = __shfl_sync(mask, cof, 0);
            const double cof10 = __shfl_sync(mask, cof, 1);
            const double cof20 = __shfl_sync(mask, cof, 2);
            const double cof11 = __shfl_sync(mask, cof, 3);
            const double cof21 = __shfl_sync(mask, cof, 4);
            const double cof22 = __shfl_sync(mask, cof, 5);
            const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
            const double inv_det = 1.0 / det;
            const double s00 = cof00 * inv_det;
            const double s10 = cof10 * inv_det;
            const double s20 = cof20 * inv_det;
            const double s11 = cof11 * inv_det;
            const double s21 = cof21 * inv_det;
            const double s22 = cof22 * inv_det;

            if (lane < 6) {
                double ra0 = b00, ra1 = b01, ra2 = b02;
                double rb0 = b00, rb1 = b01, rb2 = b02;
                int self_idx = 0;
                if (lane == 1) {
                    rb0 = b10; rb1 = b11; rb2 = b12; self_idx = 1;
                } else if (lane == 2) {
                    rb0 = b20; rb1 = b21; rb2 = b22; self_idx = 2;
                } else if (lane == 3) {
                    ra0 = b10; ra1 = b11; ra2 = b12;
                    rb0 = b10; rb1 = b11; rb2 = b12; self_idx = 4;
                } else if (lane == 4) {
                    ra0 = b10; ra1 = b11; ra2 = b12;
                    rb0 = b20; rb1 = b21; rb2 = b22; self_idx = 5;
                } else if (lane == 5) {
                    ra0 = b20; ra1 = b21; ra2 = b22;
                    rb0 = b20; rb1 = b21; rb2 = b22; self_idx = 8;
                }
                const double sb0 = s00 * rb0 + s10 * rb1 + s20 * rb2;
                const double sb1 = s10 * rb0 + s11 * rb1 + s21 * rb2;
                const double sb2 = s20 * rb0 + s21 * rb1 + s22 * rb2;
                const double schur = ra0 * sb0 + ra1 * sb1 + ra2 * sb2;
                msg_lam_new[6 * slot + lane] =
                    (1.0 - damping) * msg_lam_old[6 * slot + lane] +
                    damping * (self_lam[self_idx] - schur);
            } else if (lane < 9) {
                const int row = lane - 6;
                const double eno0 = other_eta[0] + belief_eta_old[3 * other + 0] - msg_eta_old[3 * other_slot + 0];
                const double eno1 = other_eta[1] + belief_eta_old[3 * other + 1] - msg_eta_old[3 * other_slot + 1];
                const double eno2 = other_eta[2] + belief_eta_old[3 * other + 2] - msg_eta_old[3 * other_slot + 2];
                const double y0 = s00 * eno0 + s10 * eno1 + s20 * eno2;
                const double y1 = s10 * eno0 + s11 * eno1 + s21 * eno2;
                const double y2 = s20 * eno0 + s21 * eno1 + s22 * eno2;
                double br0 = b00, br1 = b01, br2 = b02;
                if (row == 1) {
                    br0 = b10; br1 = b11; br2 = b12;
                } else if (row == 2) {
                    br0 = b20; br1 = b21; br2 = b22;
                }
                const double proj = br0 * y0 + br1 * y1 + br2 * y2;
                msg_eta_new[3 * slot + row] =
                    (1.0 - damping) * msg_eta_old[3 * slot + row] +
                    damping * (self_eta[row] - proj);
            }
        }

        grid.sync();

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta0 = unary_eta[3 * node + 0];
            double eta1 = unary_eta[3 * node + 1];
            double eta2 = unary_eta[3 * node + 2];
            double lam0 = unary_lam[6 * node + 0];
            double lam1 = unary_lam[6 * node + 1];
            double lam2 = unary_lam[6 * node + 2];
            double lam3 = unary_lam[6 * node + 3];
            double lam4 = unary_lam[6 * node + 4];
            double lam5 = unary_lam[6 * node + 5];
            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int msg_slot = incoming_slots[offset + s];
                if (msg_slot < 0) {
                    continue;
                }
                lam0 += msg_lam_new[6 * msg_slot + 0];
                lam1 += msg_lam_new[6 * msg_slot + 1];
                lam2 += msg_lam_new[6 * msg_slot + 2];
                lam3 += msg_lam_new[6 * msg_slot + 3];
                lam4 += msg_lam_new[6 * msg_slot + 4];
                lam5 += msg_lam_new[6 * msg_slot + 5];
                eta0 += msg_eta_new[3 * msg_slot + 0];
                eta1 += msg_eta_new[3 * msg_slot + 1];
                eta2 += msg_eta_new[3 * msg_slot + 2];
            }
            belief_lam_new[6 * node + 0] = lam0;
            belief_lam_new[6 * node + 1] = lam1;
            belief_lam_new[6 * node + 2] = lam2;
            belief_lam_new[6 * node + 3] = lam3;
            belief_lam_new[6 * node + 4] = lam4;
            belief_lam_new[6 * node + 5] = lam5;
            belief_eta_new[3 * node + 0] = eta0;
            belief_eta_new[3 * node + 1] = eta1;
            belief_eta_new[3 * node + 2] = eta2;
        }

        grid.sync();
    }
}

__global__ void gbpTargetNodePersistentKernel(
    int n,
    int sweeps,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_lam,
    const double* __restrict__ unary_eta,
    double* __restrict__ msg_lam_a,
    double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double* __restrict__ fixed_eta_map,
    double damping,
    int schur_mode,
    int fixed_lambda_start
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(blockDim.x) * gridDim.x;

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_lam_new = even ? msg_lam_b : msg_lam_a;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_lam_new = even ? belief_lam_b : belief_lam_a;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        const bool eta_only = fixed_lambda_start >= 0 && sweep > fixed_lambda_start;
        const bool first_eta_only = eta_only && sweep == fixed_lambda_start + 1;

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta_acc[3] = {
                unary_eta[3 * node + 0],
                unary_eta[3 * node + 1],
                unary_eta[3 * node + 2]
            };
            double lam_acc[6];
            if (eta_only && first_eta_only) {
                lam_acc[0] = belief_lam_old[6 * node + 0];
                lam_acc[1] = belief_lam_old[6 * node + 1];
                lam_acc[2] = belief_lam_old[6 * node + 2];
                lam_acc[3] = belief_lam_old[6 * node + 3];
                lam_acc[4] = belief_lam_old[6 * node + 4];
                lam_acc[5] = belief_lam_old[6 * node + 5];
            } else if (!eta_only) {
                lam_acc[0] = unary_lam[6 * node + 0];
                lam_acc[1] = unary_lam[6 * node + 1];
                lam_acc[2] = unary_lam[6 * node + 2];
                lam_acc[3] = unary_lam[6 * node + 3];
                lam_acc[4] = unary_lam[6 * node + 4];
                lam_acc[5] = unary_lam[6 * node + 5];
            }

            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming_slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                const int e = slot >> 1;
                const bool target_is_i = (slot & 1) == 0;
                const int source = target_is_i ? factor_j[e] : factor_i[e];
                const int source_slot = target_is_i ? (slot + 1) : (slot - 1);
                const double* self_lam = target_is_i ? (hii + 9 * e) : (hjj + 9 * e);
                const double* cross_lam = target_is_i ? (hij + 9 * e) : (hji + 9 * e);
                const double* cross_lam_t = target_is_i ? (hji + 9 * e) : (hij + 9 * e);
                const double* other_lam = target_is_i ? (hjj + 9 * e) : (hii + 9 * e);
                const double* self_eta = target_is_i ? (eta_i + 3 * e) : (eta_j + 3 * e);
                const double* other_eta = target_is_i ? (eta_j + 3 * e) : (eta_i + 3 * e);

                double cavity_eta[3] = {
                    belief_eta_old[3 * source + 0] - msg_eta_old[3 * source_slot + 0],
                    belief_eta_old[3 * source + 1] - msg_eta_old[3 * source_slot + 1],
                    belief_eta_old[3 * source + 2] - msg_eta_old[3 * source_slot + 2]
                };

                if (eta_only) {
                    if (first_eta_only) {
                        const double c00 = belief_lam_old[6 * source + 0] - msg_lam_old[6 * source_slot + 0];
                        const double c01 = belief_lam_old[6 * source + 1] - msg_lam_old[6 * source_slot + 1];
                        const double c02 = belief_lam_old[6 * source + 2] - msg_lam_old[6 * source_slot + 2];
                        const double c11 = belief_lam_old[6 * source + 3] - msg_lam_old[6 * source_slot + 3];
                        const double c12 = belief_lam_old[6 * source + 4] - msg_lam_old[6 * source_slot + 4];
                        const double c22 = belief_lam_old[6 * source + 5] - msg_lam_old[6 * source_slot + 5];
                        double cavity_lam[9] = {
                            c00, c01, c02,
                            c01, c11, c12,
                            c02, c12, c22
                        };
                        buildEtaMapDevice(cross_lam, other_lam, cavity_lam, schur_mode, fixed_eta_map + 9 * slot);
                    }
                    computeEtaOnlyMessageDevice(
                        self_eta,
                        other_eta,
                        cavity_eta,
                        fixed_eta_map + 9 * slot,
                        msg_eta_old + 3 * slot,
                        msg_eta_new + 3 * slot);
                } else {
                    const double c00 = belief_lam_old[6 * source + 0] - msg_lam_old[6 * source_slot + 0];
                    const double c01 = belief_lam_old[6 * source + 1] - msg_lam_old[6 * source_slot + 1];
                    const double c02 = belief_lam_old[6 * source + 2] - msg_lam_old[6 * source_slot + 2];
                    const double c11 = belief_lam_old[6 * source + 3] - msg_lam_old[6 * source_slot + 3];
                    const double c12 = belief_lam_old[6 * source + 4] - msg_lam_old[6 * source_slot + 4];
                    const double c22 = belief_lam_old[6 * source + 5] - msg_lam_old[6 * source_slot + 5];
                    double cavity_lam[9] = {
                        c00, c01, c02,
                        c01, c11, c12,
                        c02, c12, c22
                    };
                    computeGbpMessageDevice(
                        self_lam,
                        cross_lam,
                        cross_lam_t,
                        other_lam,
                        self_eta,
                        other_eta,
                        cavity_lam,
                        cavity_eta,
                        msg_lam_old + 6 * slot,
                        msg_eta_old + 3 * slot,
                        damping,
                        schur_mode,
                        msg_lam_new + 6 * slot,
                        msg_eta_new + 3 * slot);
                }

                if (!eta_only) {
                    lam_acc[0] += msg_lam_new[6 * slot + 0];
                    lam_acc[1] += msg_lam_new[6 * slot + 1];
                    lam_acc[2] += msg_lam_new[6 * slot + 2];
                    lam_acc[3] += msg_lam_new[6 * slot + 3];
                    lam_acc[4] += msg_lam_new[6 * slot + 4];
                    lam_acc[5] += msg_lam_new[6 * slot + 5];
                }
                eta_acc[0] += msg_eta_new[3 * slot + 0];
                eta_acc[1] += msg_eta_new[3 * slot + 1];
                eta_acc[2] += msg_eta_new[3 * slot + 2];
            }

            if (!eta_only || first_eta_only) {
                belief_lam_new[6 * node + 0] = lam_acc[0];
                belief_lam_new[6 * node + 1] = lam_acc[1];
                belief_lam_new[6 * node + 2] = lam_acc[2];
                belief_lam_new[6 * node + 3] = lam_acc[3];
                belief_lam_new[6 * node + 4] = lam_acc[4];
                belief_lam_new[6 * node + 5] = lam_acc[5];
            }
            belief_eta_new[3 * node + 0] = eta_acc[0];
            belief_eta_new[3 * node + 1] = eta_acc[1];
            belief_eta_new[3 * node + 2] = eta_acc[2];
        }

        grid.sync();
    }
}

__global__ void gbpEtaOnlyPersistentKernel(
    int n,
    int m,
    int eta_sweeps,
    int start_sweep,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const int* __restrict__ incoming_nodes,
    const int* __restrict__ incoming_caps,
    const int* __restrict__ incoming_offsets,
    const int* __restrict__ incoming_slots,
    const double* __restrict__ unary_eta,
    const double* __restrict__ msg_lam_a,
    const double* __restrict__ msg_lam_b,
    double* __restrict__ msg_eta_a,
    double* __restrict__ msg_eta_b,
    double* __restrict__ belief_lam_a,
    double* __restrict__ belief_lam_b,
    double* __restrict__ belief_eta_a,
    double* __restrict__ belief_eta_b,
    double* __restrict__ fixed_eta_map,
    int schur_mode
) {
    cg::grid_group grid = cg::this_grid();
    const int tid = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int stride = static_cast<int>(gridDim.x) * blockDim.x;
    const bool even_start = (start_sweep & 1) == 0;
    const double* msg_lam_current = even_start ? msg_lam_a : msg_lam_b;
    const double* belief_lam_current = even_start ? belief_lam_a : belief_lam_b;
    double* belief_lam_other = even_start ? belief_lam_b : belief_lam_a;

    for (int e = tid; e < m; e += stride) {
        const int i = factor_i[e];
        const int j = factor_j[e];
        const int slot_i = 2 * e;
        const int slot_j = slot_i + 1;

        double cavity_lam_j[9];
        const double cj00 = belief_lam_current[6 * j + 0] - msg_lam_current[6 * slot_j + 0];
        const double cj01 = belief_lam_current[6 * j + 1] - msg_lam_current[6 * slot_j + 1];
        const double cj02 = belief_lam_current[6 * j + 2] - msg_lam_current[6 * slot_j + 2];
        const double cj11 = belief_lam_current[6 * j + 3] - msg_lam_current[6 * slot_j + 3];
        const double cj12 = belief_lam_current[6 * j + 4] - msg_lam_current[6 * slot_j + 4];
        const double cj22 = belief_lam_current[6 * j + 5] - msg_lam_current[6 * slot_j + 5];
        cavity_lam_j[0] = cj00;
        cavity_lam_j[1] = cj01;
        cavity_lam_j[2] = cj02;
        cavity_lam_j[3] = cj01;
        cavity_lam_j[4] = cj11;
        cavity_lam_j[5] = cj12;
        cavity_lam_j[6] = cj02;
        cavity_lam_j[7] = cj12;
        cavity_lam_j[8] = cj22;
        buildEtaMapDevice(
            hij + 9 * e, hjj + 9 * e, cavity_lam_j, schur_mode,
            fixed_eta_map + 9 * slot_i);

        double cavity_lam_i[9];
        const double ci00 = belief_lam_current[6 * i + 0] - msg_lam_current[6 * slot_i + 0];
        const double ci01 = belief_lam_current[6 * i + 1] - msg_lam_current[6 * slot_i + 1];
        const double ci02 = belief_lam_current[6 * i + 2] - msg_lam_current[6 * slot_i + 2];
        const double ci11 = belief_lam_current[6 * i + 3] - msg_lam_current[6 * slot_i + 3];
        const double ci12 = belief_lam_current[6 * i + 4] - msg_lam_current[6 * slot_i + 4];
        const double ci22 = belief_lam_current[6 * i + 5] - msg_lam_current[6 * slot_i + 5];
        cavity_lam_i[0] = ci00;
        cavity_lam_i[1] = ci01;
        cavity_lam_i[2] = ci02;
        cavity_lam_i[3] = ci01;
        cavity_lam_i[4] = ci11;
        cavity_lam_i[5] = ci12;
        cavity_lam_i[6] = ci02;
        cavity_lam_i[7] = ci12;
        cavity_lam_i[8] = ci22;
        buildEtaMapDevice(
            hji + 9 * e, hii + 9 * e, cavity_lam_i, schur_mode,
            fixed_eta_map + 9 * slot_j);
    }

    for (int row = tid; row < n; row += stride) {
        const int node = incoming_nodes[row];
        #pragma unroll
        for (int k = 0; k < 6; ++k) {
            belief_lam_other[6 * node + k] = belief_lam_current[6 * node + k];
        }
    }

    grid.sync();

    for (int local_sweep = 0; local_sweep < eta_sweeps; ++local_sweep) {
        const int sweep = start_sweep + local_sweep;
        const bool even = (sweep & 1) == 0;
        const double* msg_eta_old = even ? msg_eta_a : msg_eta_b;
        double* msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const double* belief_eta_old = even ? belief_eta_a : belief_eta_b;
        double* belief_eta_new = even ? belief_eta_b : belief_eta_a;

        for (int e = tid; e < m; e += stride) {
            const int i = factor_i[e];
            const int j = factor_j[e];
            const int slot_i = 2 * e;
            const int slot_j = slot_i + 1;
            const double cavity_eta_j[3] = {
                belief_eta_old[3 * j + 0] - msg_eta_old[3 * slot_j + 0],
                belief_eta_old[3 * j + 1] - msg_eta_old[3 * slot_j + 1],
                belief_eta_old[3 * j + 2] - msg_eta_old[3 * slot_j + 2]
            };
            const double cavity_eta_i[3] = {
                belief_eta_old[3 * i + 0] - msg_eta_old[3 * slot_i + 0],
                belief_eta_old[3 * i + 1] - msg_eta_old[3 * slot_i + 1],
                belief_eta_old[3 * i + 2] - msg_eta_old[3 * slot_i + 2]
            };
            computeEtaOnlyMessageDevice(
                eta_i + 3 * e, eta_j + 3 * e,
                cavity_eta_j,
                fixed_eta_map + 9 * slot_i,
                msg_eta_old + 3 * slot_i,
                msg_eta_new + 3 * slot_i);
            computeEtaOnlyMessageDevice(
                eta_j + 3 * e, eta_i + 3 * e,
                cavity_eta_i,
                fixed_eta_map + 9 * slot_j,
                msg_eta_old + 3 * slot_j,
                msg_eta_new + 3 * slot_j);
        }

        grid.sync();

        for (int row = tid; row < n; row += stride) {
            const int node = incoming_nodes[row];
            double eta0 = unary_eta[3 * node + 0];
            double eta1 = unary_eta[3 * node + 1];
            double eta2 = unary_eta[3 * node + 2];
            const int cap = incoming_caps[row];
            const int offset = incoming_offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming_slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                eta0 += msg_eta_new[3 * slot + 0];
                eta1 += msg_eta_new[3 * slot + 1];
                eta2 += msg_eta_new[3 * slot + 2];
            }
            belief_eta_new[3 * node + 0] = eta0;
            belief_eta_new[3 * node + 1] = eta1;
            belief_eta_new[3 * node + 2] = eta2;
        }

        grid.sync();
    }
}

__device__ __forceinline__ void atomicAddDenseBlock3(
    double* __restrict__ A,
    int dim,
    int row0,
    int col0,
    const double* __restrict__ block
) {
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            atomicAdd(A + static_cast<size_t>(row0 + r) * dim + (col0 + c), block[3 * r + c]);
        }
    }
}

__device__ __forceinline__ void atomicAddCoarseBlock3(
    double* __restrict__ block_values,
    int slot,
    const double* __restrict__ block
) {
    double* dst = block_values + static_cast<size_t>(slot) * 9;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        atomicAdd(dst + k, block[k]);
    }
}

__device__ __forceinline__ void atomicAddVec3(double* __restrict__ b, int off, double v0, double v1, double v2) {
    atomicAdd(b + off + 0, v0);
    atomicAdd(b + off + 1, v1);
    atomicAdd(b + off + 2, v2);
}

__device__ __forceinline__ void lam6VecDevice(const double* __restrict__ lam6, const double* __restrict__ x, double* out) {
    out[0] = lam6[0] * x[0] + lam6[1] * x[1] + lam6[2] * x[2];
    out[1] = lam6[1] * x[0] + lam6[3] * x[1] + lam6[4] * x[2];
    out[2] = lam6[2] * x[0] + lam6[4] * x[1] + lam6[5] * x[2];
}

__global__ void computeBeliefMeanKernel(
    int n,
    const double* __restrict__ belief_lam6,
    const double* __restrict__ belief_eta,
    double* __restrict__ fine_x
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const double* lam6 = belief_lam6 + static_cast<size_t>(node) * 6;
    const double full[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    solve3SpdDevice(full, belief_eta + static_cast<size_t>(node) * 3, fine_x + static_cast<size_t>(node) * 3);
}

__global__ void assembleCoarseUnaryIdentityKernel(
    int n,
    int group_size,
    int coarse_dim,
    const double* __restrict__ unary_lam6,
    const double* __restrict__ unary_eta,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_A,
    double* __restrict__ coarse_b
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int off = 3 * g;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    const double block[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    atomicAddDenseBlock3(coarse_A, coarse_dim, off, off, block);

    double hx[3];
    lam6VecDevice(lam6, fine_x + static_cast<size_t>(node) * 3, hx);
    atomicAddVec3(
        coarse_b,
        off,
        unary_eta[3 * node + 0] - hx[0],
        unary_eta[3 * node + 1] - hx[1],
        unary_eta[3 * node + 2] - hx[2]);
}

__global__ void assembleCoarseUnaryIdentityBlockKernel(
    int n,
    int group_size,
    const double* __restrict__ unary_lam6,
    const double* __restrict__ unary_eta,
    const double* __restrict__ fine_x,
    double* __restrict__ block_values,
    double* __restrict__ coarse_b
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int off = 3 * g;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    const double block[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    atomicAddCoarseBlock3(block_values, g, block);

    double hx[3];
    lam6VecDevice(lam6, fine_x + static_cast<size_t>(node) * 3, hx);
    atomicAddVec3(
        coarse_b,
        off,
        unary_eta[3 * node + 0] - hx[0],
        unary_eta[3 * node + 1] - hx[1],
        unary_eta[3 * node + 2] - hx[2]);
}

__global__ void assembleCoarseUnaryLambdaBlockKernel(
    int n,
    int group_size,
    const double* __restrict__ unary_lam6,
    double* __restrict__ block_values
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    const double block[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    atomicAddCoarseBlock3(block_values, g, block);
}

__global__ void assembleCoarseUnaryResidualIdentityKernel(
    int n,
    int group_size,
    const double* __restrict__ unary_lam6,
    const double* __restrict__ unary_eta,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_b
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int off = 3 * g;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    double hx[3];
    lam6VecDevice(lam6, fine_x + static_cast<size_t>(node) * 3, hx);
    atomicAddVec3(
        coarse_b,
        off,
        unary_eta[3 * node + 0] - hx[0],
        unary_eta[3 * node + 1] - hx[1],
        unary_eta[3 * node + 2] - hx[2]);
}

__global__ void assembleCoarseBinaryIdentityKernel(
    int m,
    int group_size,
    int coarse_dim,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_A,
    double* __restrict__ coarse_b
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const int gi = i / group_size;
    const int gj = j / group_size;
    const int oi = 3 * gi;
    const int oj = 3 * gj;
    const double* Hii = hii + static_cast<size_t>(e) * 9;
    const double* Hij = hij + static_cast<size_t>(e) * 9;
    const double* Hji = hji + static_cast<size_t>(e) * 9;
    const double* Hjj = hjj + static_cast<size_t>(e) * 9;
    atomicAddDenseBlock3(coarse_A, coarse_dim, oi, oi, Hii);
    atomicAddDenseBlock3(coarse_A, coarse_dim, oi, oj, Hij);
    atomicAddDenseBlock3(coarse_A, coarse_dim, oj, oi, Hji);
    atomicAddDenseBlock3(coarse_A, coarse_dim, oj, oj, Hjj);

    const double* xi = fine_x + static_cast<size_t>(i) * 3;
    const double* xj = fine_x + static_cast<size_t>(j) * 3;
    double Hii_xi[3];
    double Hij_xj[3];
    double Hji_xi[3];
    double Hjj_xj[3];
    mat33VecDevice(Hii, xi, Hii_xi);
    mat33VecDevice(Hij, xj, Hij_xj);
    mat33VecDevice(Hji, xi, Hji_xi);
    mat33VecDevice(Hjj, xj, Hjj_xj);
    atomicAddVec3(
        coarse_b,
        oi,
        eta_i[3 * e + 0] - Hii_xi[0] - Hij_xj[0],
        eta_i[3 * e + 1] - Hii_xi[1] - Hij_xj[1],
        eta_i[3 * e + 2] - Hii_xi[2] - Hij_xj[2]);
    atomicAddVec3(
        coarse_b,
        oj,
        eta_j[3 * e + 0] - Hji_xi[0] - Hjj_xj[0],
        eta_j[3 * e + 1] - Hji_xi[1] - Hjj_xj[1],
        eta_j[3 * e + 2] - Hji_xi[2] - Hjj_xj[2]);
}

__global__ void assembleCoarseBinaryLambdaBlockKernel(
    int m,
    const int* __restrict__ factor_slot_ii,
    const int* __restrict__ factor_slot_ij,
    const int* __restrict__ factor_slot_ji,
    const int* __restrict__ factor_slot_jj,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    double* __restrict__ block_values
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    atomicAddCoarseBlock3(block_values, factor_slot_ii[e], hii + static_cast<size_t>(e) * 9);
    atomicAddCoarseBlock3(block_values, factor_slot_ij[e], hij + static_cast<size_t>(e) * 9);
    atomicAddCoarseBlock3(block_values, factor_slot_ji[e], hji + static_cast<size_t>(e) * 9);
    atomicAddCoarseBlock3(block_values, factor_slot_jj[e], hjj + static_cast<size_t>(e) * 9);
}

__global__ void assembleCoarseBinaryResidualIdentityKernel(
    int m,
    int group_size,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_b
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const int gi = i / group_size;
    const int gj = j / group_size;
    const int oi = 3 * gi;
    const int oj = 3 * gj;
    const double* Hii = hii + static_cast<size_t>(e) * 9;
    const double* Hij = hij + static_cast<size_t>(e) * 9;
    const double* Hji = hji + static_cast<size_t>(e) * 9;
    const double* Hjj = hjj + static_cast<size_t>(e) * 9;
    const double* xi = fine_x + static_cast<size_t>(i) * 3;
    const double* xj = fine_x + static_cast<size_t>(j) * 3;
    double Hii_xi[3];
    double Hij_xj[3];
    double Hji_xi[3];
    double Hjj_xj[3];
    mat33VecDevice(Hii, xi, Hii_xi);
    mat33VecDevice(Hij, xj, Hij_xj);
    mat33VecDevice(Hji, xi, Hji_xi);
    mat33VecDevice(Hjj, xj, Hjj_xj);
    atomicAddVec3(
        coarse_b,
        oi,
        eta_i[3 * e + 0] - Hii_xi[0] - Hij_xj[0],
        eta_i[3 * e + 1] - Hii_xi[1] - Hij_xj[1],
        eta_i[3 * e + 2] - Hii_xi[2] - Hij_xj[2]);
    atomicAddVec3(
        coarse_b,
        oj,
        eta_j[3 * e + 0] - Hji_xi[0] - Hjj_xj[0],
        eta_j[3 * e + 1] - Hji_xi[1] - Hjj_xj[1],
        eta_j[3 * e + 2] - Hji_xi[2] - Hjj_xj[2]);
}

__device__ __forceinline__ double basisLamBasisValue(
    const double* __restrict__ Ba,
    int ca,
    const double* __restrict__ H,
    const double* __restrict__ Bb,
    int cb
) {
    const double* a = Ba + 3 * ca;
    const double* b = Bb + 3 * cb;
    const double hb0 = H[0] * b[0] + H[1] * b[1] + H[2] * b[2];
    const double hb1 = H[3] * b[0] + H[4] * b[1] + H[5] * b[2];
    const double hb2 = H[6] * b[0] + H[7] * b[1] + H[8] * b[2];
    return a[0] * hb0 + a[1] * hb1 + a[2] * hb2;
}

__device__ __forceinline__ void atomicAddBasisLamBasisBlock(
    double* __restrict__ block_values,
    int slot,
    int r,
    const double* __restrict__ Ba,
    const double* __restrict__ H,
    const double* __restrict__ Bb
) {
    double* dst = block_values + static_cast<size_t>(slot) * r * r;
    for (int ca = 0; ca < r; ++ca) {
        for (int cb = 0; cb < r; ++cb) {
            atomicAdd(dst + ca * r + cb, basisLamBasisValue(Ba, ca, H, Bb, cb));
        }
    }
}

__device__ __forceinline__ void atomicAddBasisResidual(
    double* __restrict__ coarse_b,
    int group,
    int r,
    const double* __restrict__ B,
    const double* __restrict__ residual
) {
    double* dst = coarse_b + static_cast<size_t>(group) * r;
    for (int c = 0; c < r; ++c) {
        const double* b = B + 3 * c;
        atomicAdd(dst + c, b[0] * residual[0] + b[1] * residual[1] + b[2] * residual[2]);
    }
}

__global__ void assembleCoarseUnaryLambdaSvdKernel(
    int n,
    int r,
    const int* __restrict__ var_group,
    const double* __restrict__ var_basis,
    const double* __restrict__ unary_lam6,
    double* __restrict__ block_values
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = var_group[node];
    const double* B = var_basis + static_cast<size_t>(node) * 3 * r;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    const double H[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    atomicAddBasisLamBasisBlock(block_values, g, r, B, H, B);
}

__global__ void assembleCoarseBinaryLambdaSvdKernel(
    int m,
    int r,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const int* __restrict__ factor_slot_ii,
    const int* __restrict__ factor_slot_ij,
    const int* __restrict__ factor_slot_ji,
    const int* __restrict__ factor_slot_jj,
    const double* __restrict__ var_basis,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    double* __restrict__ block_values
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const double* Bi = var_basis + static_cast<size_t>(i) * 3 * r;
    const double* Bj = var_basis + static_cast<size_t>(j) * 3 * r;
    atomicAddBasisLamBasisBlock(block_values, factor_slot_ii[e], r, Bi, hii + static_cast<size_t>(e) * 9, Bi);
    atomicAddBasisLamBasisBlock(block_values, factor_slot_ij[e], r, Bi, hij + static_cast<size_t>(e) * 9, Bj);
    atomicAddBasisLamBasisBlock(block_values, factor_slot_ji[e], r, Bj, hji + static_cast<size_t>(e) * 9, Bi);
    atomicAddBasisLamBasisBlock(block_values, factor_slot_jj[e], r, Bj, hjj + static_cast<size_t>(e) * 9, Bj);
}

__global__ void assembleCoarseUnaryResidualSvdKernel(
    int n,
    int r,
    const int* __restrict__ var_group,
    const double* __restrict__ var_basis,
    const double* __restrict__ unary_lam6,
    const double* __restrict__ unary_eta,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_b
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = var_group[node];
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    double hx[3];
    lam6VecDevice(lam6, fine_x + static_cast<size_t>(node) * 3, hx);
    const double residual[3] = {
        unary_eta[3 * node + 0] - hx[0],
        unary_eta[3 * node + 1] - hx[1],
        unary_eta[3 * node + 2] - hx[2]
    };
    atomicAddBasisResidual(coarse_b, g, r, var_basis + static_cast<size_t>(node) * 3 * r, residual);
}

__global__ void assembleCoarseBinaryResidualSvdKernel(
    int m,
    int r,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const int* __restrict__ var_group,
    const double* __restrict__ var_basis,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const double* __restrict__ fine_x,
    double* __restrict__ coarse_b
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const double* Hii = hii + static_cast<size_t>(e) * 9;
    const double* Hij = hij + static_cast<size_t>(e) * 9;
    const double* Hji = hji + static_cast<size_t>(e) * 9;
    const double* Hjj = hjj + static_cast<size_t>(e) * 9;
    const double* xi = fine_x + static_cast<size_t>(i) * 3;
    const double* xj = fine_x + static_cast<size_t>(j) * 3;
    double Hii_xi[3];
    double Hij_xj[3];
    double Hji_xi[3];
    double Hjj_xj[3];
    mat33VecDevice(Hii, xi, Hii_xi);
    mat33VecDevice(Hij, xj, Hij_xj);
    mat33VecDevice(Hji, xi, Hji_xi);
    mat33VecDevice(Hjj, xj, Hjj_xj);
    const double ri[3] = {
        eta_i[3 * e + 0] - Hii_xi[0] - Hij_xj[0],
        eta_i[3 * e + 1] - Hii_xi[1] - Hij_xj[1],
        eta_i[3 * e + 2] - Hii_xi[2] - Hij_xj[2]
    };
    const double rj[3] = {
        eta_j[3 * e + 0] - Hji_xi[0] - Hjj_xj[0],
        eta_j[3 * e + 1] - Hji_xi[1] - Hjj_xj[1],
        eta_j[3 * e + 2] - Hji_xi[2] - Hjj_xj[2]
    };
    atomicAddBasisResidual(coarse_b, var_group[i], r, var_basis + static_cast<size_t>(i) * 3 * r, ri);
    atomicAddBasisResidual(coarse_b, var_group[j], r, var_basis + static_cast<size_t>(j) * 3 * r, rj);
}

__global__ void applyCoarseDeltaSvdKernel(
    int n,
    int r,
    double coarse_scale,
    const int* __restrict__ var_group,
    const double* __restrict__ var_basis,
    const double* __restrict__ coarse_delta,
    const double* __restrict__ belief_lam6,
    double* __restrict__ belief_eta,
    double* __restrict__ fine_x
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = var_group[node];
    const double* B = var_basis + static_cast<size_t>(node) * 3 * r;
    const double* dz = coarse_delta + static_cast<size_t>(g) * r;
    double dx[3] = {0.0, 0.0, 0.0};
    for (int c = 0; c < r; ++c) {
        const double z = coarse_scale * dz[c];
        dx[0] += B[3 * c + 0] * z;
        dx[1] += B[3 * c + 1] * z;
        dx[2] += B[3 * c + 2] * z;
    }
    double* x = fine_x + static_cast<size_t>(node) * 3;
    x[0] += dx[0];
    x[1] += dx[1];
    x[2] += dx[2];
    const double* lam6 = belief_lam6 + static_cast<size_t>(node) * 6;
    double* eta = belief_eta + static_cast<size_t>(node) * 3;
    lam6VecDevice(lam6, x, eta);
}

__device__ __forceinline__ void atomicAddGroupDenseBlockColMajor(
    double* __restrict__ mats,
    int group,
    int dim,
    int row_off,
    int col_off,
    const double* __restrict__ block
) {
    double* A = mats + static_cast<size_t>(group) * dim * dim;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            atomicAdd(A + static_cast<size_t>(col_off + c) * dim + (row_off + r), block[3 * r + c]);
        }
    }
}

__global__ void assembleGpuSvdUnaryMatrixKernel(
    int n,
    int group_size,
    int dim,
    const double* __restrict__ unary_lam6,
    double* __restrict__ group_mats
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int local = node - g * group_size;
    const int off = 3 * local;
    const double* lam6 = unary_lam6 + static_cast<size_t>(node) * 6;
    const double H[9] = {
        lam6[0], lam6[1], lam6[2],
        lam6[1], lam6[3], lam6[4],
        lam6[2], lam6[4], lam6[5]
    };
    atomicAddGroupDenseBlockColMajor(group_mats, g, dim, off, off, H);
}

__global__ void assembleGpuSvdBinaryMatrixKernel(
    int m,
    int group_size,
    int dim,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    double* __restrict__ group_mats
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const int gi = i / group_size;
    const int gj = j / group_size;
    if (gi != gj) {
        return;
    }
    const int li = i - gi * group_size;
    const int lj = j - gi * group_size;
    atomicAddGroupDenseBlockColMajor(group_mats, gi, dim, 3 * li, 3 * li, hii + static_cast<size_t>(e) * 9);
    atomicAddGroupDenseBlockColMajor(group_mats, gi, dim, 3 * li, 3 * lj, hij + static_cast<size_t>(e) * 9);
    atomicAddGroupDenseBlockColMajor(group_mats, gi, dim, 3 * lj, 3 * li, hji + static_cast<size_t>(e) * 9);
    atomicAddGroupDenseBlockColMajor(group_mats, gi, dim, 3 * lj, 3 * lj, hjj + static_cast<size_t>(e) * 9);
}

__global__ void extractGpuSvdBasisKernel(
    int n,
    int group_size,
    int dim,
    int r,
    const double* __restrict__ eigvecs,
    double* __restrict__ var_basis
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int local = node - g * group_size;
    const int row0 = 3 * local;
    const double* A = eigvecs + static_cast<size_t>(g) * dim * dim;
    double* dst = var_basis + static_cast<size_t>(node) * 3 * r;
    for (int c = 0; c < r; ++c) {
        dst[3 * c + 0] = A[static_cast<size_t>(c) * dim + row0 + 0];
        dst[3 * c + 1] = A[static_cast<size_t>(c) * dim + row0 + 1];
        dst[3 * c + 2] = A[static_cast<size_t>(c) * dim + row0 + 2];
    }
}

__global__ void buildWarmRayleighSubspaceKernel(
    int groups,
    int group_size,
    int dim,
    int r,
    int p,
    const double* __restrict__ warm_var_basis,
    double* __restrict__ q
) {
    const int g = static_cast<int>(blockIdx.x);
    if (g >= groups || threadIdx.x != 0) {
        return;
    }
    double* Q = q + static_cast<size_t>(g) * dim * p;
    for (int c = 0; c < p; ++c) {
        for (int row = 0; row < dim; ++row) {
            Q[static_cast<size_t>(c) * dim + row] = 0.0;
        }
    }
    for (int c = 0; c < r; ++c) {
        for (int local = 0; local < group_size; ++local) {
            const int var = g * group_size + local;
            const double* src = warm_var_basis + static_cast<size_t>(var) * 3 * r + 3 * c;
            Q[static_cast<size_t>(c) * dim + 3 * local + 0] = src[0];
            Q[static_cast<size_t>(c) * dim + 3 * local + 1] = src[1];
            Q[static_cast<size_t>(c) * dim + 3 * local + 2] = src[2];
        }
    }
    for (int c = r; c < p; ++c) {
        const int row = c - r;
        if (row < dim) {
            Q[static_cast<size_t>(c) * dim + row] = 1.0;
        }
    }

    for (int c = 0; c < p; ++c) {
        for (int prev = 0; prev < c; ++prev) {
            double dot = 0.0;
            for (int row = 0; row < dim; ++row) {
                dot += Q[static_cast<size_t>(prev) * dim + row] *
                       Q[static_cast<size_t>(c) * dim + row];
            }
            for (int row = 0; row < dim; ++row) {
                Q[static_cast<size_t>(c) * dim + row] -=
                    dot * Q[static_cast<size_t>(prev) * dim + row];
            }
        }
        double norm_sq = 0.0;
        for (int row = 0; row < dim; ++row) {
            const double v = Q[static_cast<size_t>(c) * dim + row];
            norm_sq += v * v;
        }
        if (norm_sq < 1e-24) {
            for (int row = 0; row < dim; ++row) {
                Q[static_cast<size_t>(c) * dim + row] = (row == c) ? 1.0 : 0.0;
            }
            for (int prev = 0; prev < c; ++prev) {
                double dot = 0.0;
                for (int row = 0; row < dim; ++row) {
                    dot += Q[static_cast<size_t>(prev) * dim + row] *
                           Q[static_cast<size_t>(c) * dim + row];
                }
                for (int row = 0; row < dim; ++row) {
                    Q[static_cast<size_t>(c) * dim + row] -=
                        dot * Q[static_cast<size_t>(prev) * dim + row];
                }
            }
            norm_sq = 0.0;
            for (int row = 0; row < dim; ++row) {
                const double v = Q[static_cast<size_t>(c) * dim + row];
                norm_sq += v * v;
            }
        }
        const double inv_norm = rsqrt(norm_sq);
        for (int row = 0; row < dim; ++row) {
            Q[static_cast<size_t>(c) * dim + row] *= inv_norm;
        }
    }
}

__global__ void buildInitialRayleighSubspaceKernel(
    int groups,
    int dim,
    int p,
    double* __restrict__ q
) {
    const int g = static_cast<int>(blockIdx.x);
    if (g >= groups || threadIdx.x != 0) {
        return;
    }
    double* Q = q + static_cast<size_t>(g) * dim * p;
    for (int c = 0; c < p; ++c) {
        for (int row = 0; row < dim; ++row) {
            Q[static_cast<size_t>(c) * dim + row] = (row == c) ? 1.0 : 0.0;
        }
    }
}

__global__ void computeWarmRayleighMatrixKernel(
    int groups,
    int dim,
    int p,
    const double* __restrict__ group_mats,
    const double* __restrict__ q,
    double* __restrict__ small_mats
) {
    const int g = static_cast<int>(blockIdx.x);
    const int idx = static_cast<int>(threadIdx.x);
    if (g >= groups || idx >= p * p) {
        return;
    }
    const int a = idx % p;
    const int b = idx / p;
    const double* A = group_mats + static_cast<size_t>(g) * dim * dim;
    const double* Q = q + static_cast<size_t>(g) * dim * p;
    double value = 0.0;
    for (int col = 0; col < dim; ++col) {
        const double qb = Q[static_cast<size_t>(b) * dim + col];
        double aq = 0.0;
        for (int row = 0; row < dim; ++row) {
            aq += A[static_cast<size_t>(col) * dim + row] *
                  Q[static_cast<size_t>(a) * dim + row];
        }
        value += aq * qb;
    }
    small_mats[static_cast<size_t>(g) * p * p + static_cast<size_t>(b) * p + a] = value;
}

__global__ void prepareWarmInverseIterationKernel(
    int groups,
    int dim,
    int p,
    const double* __restrict__ group_mats,
    double* __restrict__ factor_mats,
    double** __restrict__ factor_ptrs,
    double** __restrict__ rhs_ptrs,
    double* __restrict__ q
) {
    const int g = static_cast<int>(blockIdx.x);
    if (g >= groups || threadIdx.x != 0) {
        return;
    }
    const double* A = group_mats + static_cast<size_t>(g) * dim * dim;
    double* F = factor_mats + static_cast<size_t>(g) * dim * dim;
    factor_ptrs[g] = F;
    rhs_ptrs[g] = q + static_cast<size_t>(g) * dim * p;
    double diag_abs_sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        diag_abs_sum += fabs(A[static_cast<size_t>(i) * dim + i]);
    }
    const double ridge = 1e-10 * fmax(1.0, diag_abs_sum / static_cast<double>(dim));
    for (int col = 0; col < dim; ++col) {
        for (int row = 0; row < dim; ++row) {
            double v = A[static_cast<size_t>(col) * dim + row];
            if (row == col) {
                v += ridge;
            }
            F[static_cast<size_t>(col) * dim + row] = v;
        }
    }
}

__global__ void orthonormalizeGroupColumnsKernel(
    int groups,
    int dim,
    int p,
    double* __restrict__ q
) {
    const int g = static_cast<int>(blockIdx.x);
    if (g >= groups || threadIdx.x != 0) {
        return;
    }
    double* Q = q + static_cast<size_t>(g) * dim * p;
    for (int c = 0; c < p; ++c) {
        for (int prev = 0; prev < c; ++prev) {
            double dot = 0.0;
            for (int row = 0; row < dim; ++row) {
                dot += Q[static_cast<size_t>(prev) * dim + row] *
                       Q[static_cast<size_t>(c) * dim + row];
            }
            for (int row = 0; row < dim; ++row) {
                Q[static_cast<size_t>(c) * dim + row] -=
                    dot * Q[static_cast<size_t>(prev) * dim + row];
            }
        }
        double norm_sq = 0.0;
        for (int row = 0; row < dim; ++row) {
            const double v = Q[static_cast<size_t>(c) * dim + row];
            norm_sq += v * v;
        }
        const double inv_norm = rsqrt(fmax(norm_sq, 1e-24));
        for (int row = 0; row < dim; ++row) {
            Q[static_cast<size_t>(c) * dim + row] *= inv_norm;
        }
    }
}

__global__ void extractWarmRayleighBasisKernel(
    int n,
    int group_size,
    int dim,
    int r,
    int p,
    const double* __restrict__ q,
    const double* __restrict__ small_eigvecs,
    double* __restrict__ var_basis
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    const int local = node - g * group_size;
    const int row0 = 3 * local;
    const double* Q = q + static_cast<size_t>(g) * dim * p;
    const double* U = small_eigvecs + static_cast<size_t>(g) * p * p;
    double* dst = var_basis + static_cast<size_t>(node) * 3 * r;
    for (int c = 0; c < r; ++c) {
        double out0 = 0.0;
        double out1 = 0.0;
        double out2 = 0.0;
        for (int a = 0; a < p; ++a) {
            const double u = U[static_cast<size_t>(c) * p + a];
            out0 += Q[static_cast<size_t>(a) * dim + row0 + 0] * u;
            out1 += Q[static_cast<size_t>(a) * dim + row0 + 1] * u;
            out2 += Q[static_cast<size_t>(a) * dim + row0 + 2] * u;
        }
        dst[3 * c + 0] = out0;
        dst[3 * c + 1] = out1;
        dst[3 * c + 2] = out2;
    }
}

__device__ __forceinline__ Pose2Dev se2ExpPoseDevice(const double* xi) {
    const double vx = xi[0];
    const double vy = xi[1];
    const double w = xi[2];
    if (fabs(w) < 1e-12) {
        return {vx, vy, 0.0};
    }
    const double a = sin(w) / w;
    const double b = (1.0 - cos(w)) / w;
    return {
        a * vx - b * vy,
        b * vx + a * vy,
        wrapAngleDevice(w)
    };
}

__global__ void zeroUnarySe2Kernel(int n, double* __restrict__ unary_lam6, double* __restrict__ unary_eta) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    #pragma unroll
    for (int k = 0; k < 6; ++k) {
        unary_lam6[6 * node + k] = 0.0;
    }
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        unary_eta[3 * node + k] = 0.0;
    }
}

__global__ void linearizeSe2FactorsKernel(
    int m,
    const double* __restrict__ poses,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const double* __restrict__ measurements,
    const double* __restrict__ information,
    double huber_delta,
    double* __restrict__ hii,
    double* __restrict__ hij,
    double* __restrict__ hji,
    double* __restrict__ hjj,
    double* __restrict__ eta_i,
    double* __restrict__ eta_j
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const Pose2Dev xi = poseFromArrayDevice(poses, i);
    const Pose2Dev xj = poseFromArrayDevice(poses, j);
    double r[3];
    double J[18];
    analyticEdgeResidualJacobianDevice(xi, xj, measurements + static_cast<size_t>(e) * 3, r, J);
    const double* omega = information + static_cast<size_t>(e) * 9;
    double omega_r[3];
    mat33VecDevice(omega, r, omega_r);
    const double chi = fmax(0.0, r[0] * omega_r[0] + r[1] * omega_r[1] + r[2] * omega_r[2]);
    double robust_weight = 1.0;
    if (huber_delta > 0.0) {
        const double norm = sqrt(chi);
        if (norm > huber_delta && norm > 0.0) {
            robust_weight = huber_delta / norm;
        }
    }
    double weighted_omega[9];
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        weighted_omega[k] = robust_weight * omega[k];
    }
    double weighted_omega_r[3];
    mat33VecDevice(weighted_omega, r, weighted_omega_r);
    double OJ[18];
    #pragma unroll
    for (int row = 0; row < 3; ++row) {
        #pragma unroll
        for (int col = 0; col < 6; ++col) {
            OJ[6 * row + col] =
                weighted_omega[3 * row + 0] * J[col] +
                weighted_omega[3 * row + 1] * J[6 + col] +
                weighted_omega[3 * row + 2] * J[12 + col];
        }
    }
    double lam[36];
    double eta[6];
    #pragma unroll
    for (int col = 0; col < 6; ++col) {
        double ev = 0.0;
        #pragma unroll
        for (int row = 0; row < 3; ++row) {
            ev -= J[6 * row + col] * weighted_omega_r[row];
        }
        eta[col] = ev;
        #pragma unroll
        for (int col2 = 0; col2 < 6; ++col2) {
            double s = 0.0;
            #pragma unroll
            for (int row = 0; row < 3; ++row) {
                s += J[6 * row + col] * OJ[6 * row + col2];
            }
            lam[6 * col + col2] = s;
        }
    }
    double* Hii = hii + static_cast<size_t>(e) * 9;
    double* Hij = hij + static_cast<size_t>(e) * 9;
    double* Hji = hji + static_cast<size_t>(e) * 9;
    double* Hjj = hjj + static_cast<size_t>(e) * 9;
    #pragma unroll
    for (int rr = 0; rr < 3; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < 3; ++cc) {
            Hii[3 * rr + cc] = lam[6 * rr + cc];
            Hij[3 * rr + cc] = lam[6 * rr + (3 + cc)];
            Hji[3 * rr + cc] = lam[6 * (3 + rr) + cc];
            Hjj[3 * rr + cc] = lam[6 * (3 + rr) + (3 + cc)];
        }
    }
    eta_i[3 * e + 0] = eta[0];
    eta_i[3 * e + 1] = eta[1];
    eta_i[3 * e + 2] = eta[2];
    eta_j[3 * e + 0] = eta[3];
    eta_j[3 * e + 1] = eta[4];
    eta_j[3 * e + 2] = eta[5];
}

__global__ void linearizeSe2AnchorKernel(
    const double* __restrict__ poses,
    const double* __restrict__ anchor_pose_arr,
    const double* __restrict__ anchor_info,
    double* __restrict__ unary_lam6,
    double* __restrict__ unary_eta
) {
    double r[3];
    double J[9];
    const Pose2Dev base = poseFromArrayDevice(poses, 0);
    const Pose2Dev anchor{anchor_pose_arr[0], anchor_pose_arr[1], anchor_pose_arr[2]};
    analyticAnchorResidualJacobianDevice(base, anchor, r, J);
    double OJ[9];
    mat33MulDevice(anchor_info, J, OJ);
    double lam[9];
    #pragma unroll
    for (int rr = 0; rr < 3; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < 3; ++cc) {
            lam[3 * rr + cc] =
                J[0 * 3 + rr] * OJ[0 * 3 + cc] +
                J[1 * 3 + rr] * OJ[1 * 3 + cc] +
                J[2 * 3 + rr] * OJ[2 * 3 + cc];
        }
    }
    double info_r[3];
    mat33VecDevice(anchor_info, r, info_r);
    double eta[3] = {0.0, 0.0, 0.0};
    #pragma unroll
    for (int col = 0; col < 3; ++col) {
        #pragma unroll
        for (int row = 0; row < 3; ++row) {
            eta[col] -= J[3 * row + col] * info_r[row];
        }
    }
    unary_lam6[0] = lam[0];
    unary_lam6[1] = 0.5 * (lam[1] + lam[3]);
    unary_lam6[2] = 0.5 * (lam[2] + lam[6]);
    unary_lam6[3] = lam[4];
    unary_lam6[4] = 0.5 * (lam[5] + lam[7]);
    unary_lam6[5] = lam[8];
    unary_eta[0] = eta[0];
    unary_eta[1] = eta[1];
    unary_eta[2] = eta[2];
}

__global__ void applyPoseDeltasSe2Kernel(int n, const double* __restrict__ delta, double* __restrict__ poses) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const Pose2Dev base = poseFromArrayDevice(poses, node);
    const Pose2Dev inc = se2ExpPoseDevice(delta + static_cast<size_t>(node) * 3);
    const Pose2Dev out = se2ComposeDevice(base, inc);
    poses[3 * node + 0] = out.x;
    poses[3 * node + 1] = out.y;
    poses[3 * node + 2] = out.theta;
}

__global__ void assembleCoarseBinaryIdentityBlockKernel(
    int m,
    int group_size,
    const int* __restrict__ factor_i,
    const int* __restrict__ factor_j,
    const int* __restrict__ factor_slot_ii,
    const int* __restrict__ factor_slot_ij,
    const int* __restrict__ factor_slot_ji,
    const int* __restrict__ factor_slot_jj,
    const double* __restrict__ hii,
    const double* __restrict__ hij,
    const double* __restrict__ hji,
    const double* __restrict__ hjj,
    const double* __restrict__ eta_i,
    const double* __restrict__ eta_j,
    const double* __restrict__ fine_x,
    double* __restrict__ block_values,
    double* __restrict__ coarse_b
) {
    const int e = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (e >= m) {
        return;
    }
    const int i = factor_i[e];
    const int j = factor_j[e];
    const int gi = i / group_size;
    const int gj = j / group_size;
    const int oi = 3 * gi;
    const int oj = 3 * gj;
    const double* Hii = hii + static_cast<size_t>(e) * 9;
    const double* Hij = hij + static_cast<size_t>(e) * 9;
    const double* Hji = hji + static_cast<size_t>(e) * 9;
    const double* Hjj = hjj + static_cast<size_t>(e) * 9;
    atomicAddCoarseBlock3(block_values, factor_slot_ii[e], Hii);
    atomicAddCoarseBlock3(block_values, factor_slot_ij[e], Hij);
    atomicAddCoarseBlock3(block_values, factor_slot_ji[e], Hji);
    atomicAddCoarseBlock3(block_values, factor_slot_jj[e], Hjj);

    const double* xi = fine_x + static_cast<size_t>(i) * 3;
    const double* xj = fine_x + static_cast<size_t>(j) * 3;
    double Hii_xi[3];
    double Hij_xj[3];
    double Hji_xi[3];
    double Hjj_xj[3];
    mat33VecDevice(Hii, xi, Hii_xi);
    mat33VecDevice(Hij, xj, Hij_xj);
    mat33VecDevice(Hji, xi, Hji_xi);
    mat33VecDevice(Hjj, xj, Hjj_xj);
    atomicAddVec3(
        coarse_b,
        oi,
        eta_i[3 * e + 0] - Hii_xi[0] - Hij_xj[0],
        eta_i[3 * e + 1] - Hii_xi[1] - Hij_xj[1],
        eta_i[3 * e + 2] - Hii_xi[2] - Hij_xj[2]);
    atomicAddVec3(
        coarse_b,
        oj,
        eta_j[3 * e + 0] - Hji_xi[0] - Hjj_xj[0],
        eta_j[3 * e + 1] - Hji_xi[1] - Hjj_xj[1],
        eta_j[3 * e + 2] - Hji_xi[2] - Hjj_xj[2]);
}

__global__ void applyCoarseDeltaIdentityKernel(
    int n,
    int group_size,
    double coarse_scale,
    const double* __restrict__ coarse_delta,
    const double* __restrict__ belief_lam6,
    double* __restrict__ belief_eta,
    double* __restrict__ fine_x
) {
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) {
        return;
    }
    const int g = node / group_size;
    double* x = fine_x + static_cast<size_t>(node) * 3;
    x[0] += coarse_scale * coarse_delta[3 * g + 0];
    x[1] += coarse_scale * coarse_delta[3 * g + 1];
    x[2] += coarse_scale * coarse_delta[3 * g + 2];
    const double* lam6 = belief_lam6 + static_cast<size_t>(node) * 6;
    double* eta = belief_eta + static_cast<size_t>(node) * 3;
    lam6VecDevice(lam6, x, eta);
}

__global__ void bucketJacobiKernel(
    int rows,
    int cap,
    const int* __restrict__ nodes,
    const int* __restrict__ cols,
    const float* __restrict__ blocks,
    const float* __restrict__ diag,
    const float* __restrict__ rhs,
    const float* __restrict__ x,
    float* __restrict__ x_next,
    float omega
) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const int node = nodes[row];
    float accum[3] = {
        rhs[3 * node + 0],
        rhs[3 * node + 1],
        rhs[3 * node + 2]
    };

    for (int slot = 0; slot < cap; ++slot) {
        const int nb = cols[row * cap + slot];
        if (nb < 0) {
            continue;
        }
        const float* h = blocks + (static_cast<size_t>(row) * cap + slot) * 9;
        const float x0 = x[3 * nb + 0];
        const float x1 = x[3 * nb + 1];
        const float x2 = x[3 * nb + 2];
        accum[0] -= h[0] * x0 + h[1] * x1 + h[2] * x2;
        accum[1] -= h[3] * x0 + h[4] * x1 + h[5] * x2;
        accum[2] -= h[6] * x0 + h[7] * x1 + h[8] * x2;
    }

    float solved[3];
    solve3SpdDevice(diag + 9 * node, accum, solved);
    x_next[3 * node + 0] = (1.0f - omega) * x[3 * node + 0] + omega * solved[0];
    x_next[3 * node + 1] = (1.0f - omega) * x[3 * node + 1] + omega * solved[1];
    x_next[3 * node + 2] = (1.0f - omega) * x[3 * node + 2] + omega * solved[2];
}

void solve3SpdHost(const float* a, const float* b, float* x) {
    float jitter = kDefaultJitter;
    float l00 = 0.0f, l10 = 0.0f, l11 = 0.0f, l20 = 0.0f, l21 = 0.0f, l22 = 0.0f;
    bool ok = false;
    for (int attempt = 0; attempt < 12; ++attempt) {
        const float a00 = a[0] + jitter;
        const float a11 = a[4] + jitter;
        const float a22 = a[8] + jitter;
        if (a00 > 0.0f) {
            l00 = std::sqrt(a00);
            l10 = a[3] / l00;
            const float d11 = a11 - l10 * l10;
            if (d11 > 0.0f) {
                l11 = std::sqrt(d11);
                l20 = a[6] / l00;
                l21 = (a[7] - l20 * l10) / l11;
                const float d22 = a22 - l20 * l20 - l21 * l21;
                if (d22 > 0.0f) {
                    l22 = std::sqrt(d22);
                    ok = true;
                    break;
                }
            }
        }
        jitter *= 10.0f;
    }
    if (!ok) {
        const float d0 = std::max(std::abs(a[0]), kDefaultJitter);
        const float d1 = std::max(std::abs(a[4]), kDefaultJitter);
        const float d2 = std::max(std::abs(a[8]), kDefaultJitter);
        x[0] = b[0] / d0;
        x[1] = b[1] / d1;
        x[2] = b[2] / d2;
        return;
    }
    const float y0 = b[0] / l00;
    const float y1 = (b[1] - l10 * y0) / l11;
    const float y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22;
    x[1] = (y1 - l21 * x[2]) / l11;
    x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
}

void solve3SpdHost(const double* a, const double* b, double* x) {
    double jitter = kGbpJitter;
    double l00 = 0.0, l10 = 0.0, l11 = 0.0, l20 = 0.0, l21 = 0.0, l22 = 0.0;
    bool ok = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const double a00 = a[0] + jitter;
        const double a11 = a[4] + jitter;
        const double a22 = a[8] + jitter;
        if (a00 > 0.0) {
            l00 = std::sqrt(a00);
            l10 = a[3] / l00;
            const double d11 = a11 - l10 * l10;
            if (d11 > 0.0) {
                l11 = std::sqrt(d11);
                l20 = a[6] / l00;
                l21 = (a[7] - l20 * l10) / l11;
                const double d22 = a22 - l20 * l20 - l21 * l21;
                if (d22 > 0.0) {
                    l22 = std::sqrt(d22);
                    ok = true;
                    break;
                }
            }
        }
        jitter *= 10.0;
    }
    if (!ok) {
        const double d0 = std::max(std::abs(a[0]), kGbpJitter);
        const double d1 = std::max(std::abs(a[4]), kGbpJitter);
        const double d2 = std::max(std::abs(a[8]), kGbpJitter);
        x[0] = b[0] / d0;
        x[1] = b[1] / d1;
        x[2] = b[2] / d2;
        return;
    }
    const double y0 = b[0] / l00;
    const double y1 = (b[1] - l10 * y0) / l11;
    const double y2 = (b[2] - l20 * y0 - l21 * y1) / l22;
    x[2] = y2 / l22;
    x[1] = (y1 - l21 * x[2]) / l11;
    x[0] = (y0 - l10 * x[1] - l20 * x[2]) / l00;
}

void solve3SpdMatHost(const double* a, const double* b, double* x) {
    for (int col = 0; col < 3; ++col) {
        double rhs[3] = {b[col], b[3 + col], b[6 + col]};
        double sol[3];
        solve3SpdHost(a, rhs, sol);
        x[col] = sol[0];
        x[3 + col] = sol[1];
        x[6 + col] = sol[2];
    }
}

void mat33MulHost(const double* a, const double* b, double* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[3 * r + c] =
                a[3 * r + 0] * b[c] +
                a[3 * r + 1] * b[3 + c] +
                a[3 * r + 2] * b[6 + c];
        }
    }
}

void mat33VecHost(const double* a, const double* x, double* out) {
    out[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2];
    out[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2];
    out[2] = a[6] * x[0] + a[7] * x[1] + a[8] * x[2];
}

void computeGbpMessageHost(
    const double* self_lam,
    const double* cross,
    const double* cross_t,
    const double* other_lam,
    const double* self_eta,
    const double* other_eta,
    const double* cavity_lam,
    const double* cavity_eta,
    const double* old_msg_lam,
    const double* old_msg_eta,
    double damping,
    double* out_msg_lam,
    double* out_msg_eta
) {
    double cond[9];
    double cond_eta[3];
    for (int k = 0; k < 9; ++k) {
        cond[k] = other_lam[k] + cavity_lam[k];
    }
    cond[0] += kGbpJitter;
    cond[4] += kGbpJitter;
    cond[8] += kGbpJitter;
    for (int k = 0; k < 3; ++k) {
        cond_eta[k] = other_eta[k] + cavity_eta[k];
    }

    const double a00 = cond[0];
    const double a10 = cond[3];
    const double a20 = cond[6];
    const double a11 = cond[4];
    const double a21 = cond[7];
    const double a22 = cond[8];
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;

    double schur[9];
    double eta_schur[3];
    if (det > 0.0) {
        const double inv_det = 1.0 / det;
        const double inv[9] = {
            cof00 * inv_det, cof10 * inv_det, cof20 * inv_det,
            cof10 * inv_det, cof11 * inv_det, cof21 * inv_det,
            cof20 * inv_det, cof21 * inv_det, cof22 * inv_det
        };
        double map[9];
        mat33MulHost(cross, inv, map);
        mat33MulHost(map, cross_t, schur);
        double solved_eta[3];
        mat33VecHost(inv, cond_eta, solved_eta);
        mat33VecHost(cross, solved_eta, eta_schur);
    } else {
        double solved_cross_t[9];
        solve3SpdMatHost(cond, cross_t, solved_cross_t);
        mat33MulHost(cross, solved_cross_t, schur);
        double solved_eta[3];
        solve3SpdHost(cond, cond_eta, solved_eta);
        mat33VecHost(cross, solved_eta, eta_schur);
    }

    for (int k = 0; k < 9; ++k) {
        const double fresh = self_lam[k] - schur[k];
        out_msg_lam[k] = (1.0 - damping) * old_msg_lam[k] + damping * fresh;
    }
    const double s01 = 0.5 * (out_msg_lam[1] + out_msg_lam[3]);
    const double s02 = 0.5 * (out_msg_lam[2] + out_msg_lam[6]);
    const double s12 = 0.5 * (out_msg_lam[5] + out_msg_lam[7]);
    out_msg_lam[1] = s01;
    out_msg_lam[3] = s01;
    out_msg_lam[2] = s02;
    out_msg_lam[6] = s02;
    out_msg_lam[5] = s12;
    out_msg_lam[7] = s12;

    for (int k = 0; k < 3; ++k) {
        const double fresh = self_eta[k] - eta_schur[k];
        out_msg_eta[k] = (1.0 - damping) * old_msg_eta[k] + damping * fresh;
    }
}

void buildEtaMapHost(
    const double* cross,
    const double* other_lam,
    const double* cavity_lam,
    double* eta_map
) {
    double cond[9];
    for (int k = 0; k < 9; ++k) {
        cond[k] = other_lam[k] + cavity_lam[k];
    }
    cond[0] += kGbpJitter;
    cond[4] += kGbpJitter;
    cond[8] += kGbpJitter;

    const double a00 = cond[0];
    const double a10 = cond[3];
    const double a20 = cond[6];
    const double a11 = cond[4];
    const double a21 = cond[7];
    const double a22 = cond[8];
    const double cof00 = a11 * a22 - a21 * a21;
    const double cof10 = a20 * a21 - a10 * a22;
    const double cof20 = a10 * a21 - a20 * a11;
    const double cof11 = a00 * a22 - a20 * a20;
    const double cof21 = a10 * a20 - a00 * a21;
    const double cof22 = a00 * a11 - a10 * a10;
    const double det = a00 * cof00 + a10 * cof10 + a20 * cof20;
    if (det > 0.0) {
        const double inv_det = 1.0 / det;
        const double inv[9] = {
            cof00 * inv_det, cof10 * inv_det, cof20 * inv_det,
            cof10 * inv_det, cof11 * inv_det, cof21 * inv_det,
            cof20 * inv_det, cof21 * inv_det, cof22 * inv_det
        };
        mat33MulHost(cross, inv, eta_map);
        return;
    }

    const double cross_t[9] = {
        cross[0], cross[3], cross[6],
        cross[1], cross[4], cross[7],
        cross[2], cross[5], cross[8]
    };
    double solved_cross_t[9];
    solve3SpdMatHost(cond, cross_t, solved_cross_t);
    eta_map[0] = solved_cross_t[0];
    eta_map[1] = solved_cross_t[3];
    eta_map[2] = solved_cross_t[6];
    eta_map[3] = solved_cross_t[1];
    eta_map[4] = solved_cross_t[4];
    eta_map[5] = solved_cross_t[7];
    eta_map[6] = solved_cross_t[2];
    eta_map[7] = solved_cross_t[5];
    eta_map[8] = solved_cross_t[8];
}

void computeEtaOnlyMessageHost(
    const double* self_eta,
    const double* other_eta,
    const double* cavity_eta,
    const double* eta_map,
    const double* old_msg_eta,
    double* out_msg_eta
) {
    (void)old_msg_eta;
    const double eno0 = other_eta[0] + cavity_eta[0];
    const double eno1 = other_eta[1] + cavity_eta[1];
    const double eno2 = other_eta[2] + cavity_eta[2];
    const double proj0 = eta_map[0] * eno0 + eta_map[1] * eno1 + eta_map[2] * eno2;
    const double proj1 = eta_map[3] * eno0 + eta_map[4] * eno1 + eta_map[5] * eno2;
    const double proj2 = eta_map[6] * eno0 + eta_map[7] * eno1 + eta_map[8] * eno2;
    out_msg_eta[0] = self_eta[0] - proj0;
    out_msg_eta[1] = self_eta[1] - proj1;
    out_msg_eta[2] = self_eta[2] - proj2;
}

void cpuGbpSweeps(
    const GbpGraph& graph,
    const IncomingFlatHost& incoming,
    int sweeps,
    int fixed_lambda_start,
    double damping,
    int threads,
    std::vector<double>& out_belief_lam,
    std::vector<double>& out_belief_eta
) {
#if defined(_OPENMP)
    omp_set_num_threads(threads);
#else
    (void)threads;
#endif
    const int n = graph.n;
    const int m = graph.m;
    const int num_slots = 2 * m;
    std::vector<double> msg_lam_a(static_cast<size_t>(num_slots) * 9, 0.0);
    std::vector<double> msg_lam_b(static_cast<size_t>(num_slots) * 9, 0.0);
    std::vector<double> msg_eta_a(static_cast<size_t>(num_slots) * 3, 0.0);
    std::vector<double> msg_eta_b(static_cast<size_t>(num_slots) * 3, 0.0);
    std::vector<double> belief_lam_a = graph.unary_lam;
    std::vector<double> belief_lam_b(static_cast<size_t>(n) * 9, 0.0);
    std::vector<double> belief_eta_a = graph.unary_eta;
    std::vector<double> belief_eta_b(static_cast<size_t>(n) * 3, 0.0);
    std::vector<double> fixed_eta_map(static_cast<size_t>(num_slots) * 9, 0.0);

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const std::vector<double>& msg_lam_old = even ? msg_lam_a : msg_lam_b;
        const std::vector<double>& msg_eta_old = even ? msg_eta_a : msg_eta_b;
        std::vector<double>& msg_lam_new = even ? msg_lam_b : msg_lam_a;
        std::vector<double>& msg_eta_new = even ? msg_eta_b : msg_eta_a;
        const std::vector<double>& belief_lam_old = even ? belief_lam_a : belief_lam_b;
        const std::vector<double>& belief_eta_old = even ? belief_eta_a : belief_eta_b;
        std::vector<double>& belief_lam_new = even ? belief_lam_b : belief_lam_a;
        std::vector<double>& belief_eta_new = even ? belief_eta_b : belief_eta_a;
        const bool eta_only = fixed_lambda_start >= 0 && sweep > fixed_lambda_start;
        const bool first_eta_only = eta_only && sweep == fixed_lambda_start + 1;

        if (first_eta_only) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
            for (int e = 0; e < m; ++e) {
                const int i = graph.factor_i[e];
                const int j = graph.factor_j[e];
                const int slot_i = 2 * e;
                const int slot_j = slot_i + 1;

                double cavity_lam_j[9];
                double cavity_lam_i[9];
                for (int k = 0; k < 9; ++k) {
                    cavity_lam_j[k] = belief_lam_old[9 * j + k] - msg_lam_old[9 * slot_j + k];
                    cavity_lam_i[k] = belief_lam_old[9 * i + k] - msg_lam_old[9 * slot_i + k];
                }

                buildEtaMapHost(
                    graph.hij.data() + 9 * e, graph.hjj.data() + 9 * e, cavity_lam_j,
                    fixed_eta_map.data() + 9 * slot_i);
                buildEtaMapHost(
                    graph.hji.data() + 9 * e, graph.hii.data() + 9 * e, cavity_lam_i,
                    fixed_eta_map.data() + 9 * slot_j);
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int e = 0; e < m; ++e) {
            const int i = graph.factor_i[e];
            const int j = graph.factor_j[e];
            const int slot_i = 2 * e;
            const int slot_j = slot_i + 1;

            double cavity_eta_j[3];
            double cavity_eta_i[3];
            for (int k = 0; k < 3; ++k) {
                cavity_eta_j[k] = belief_eta_old[3 * j + k] - msg_eta_old[3 * slot_j + k];
                cavity_eta_i[k] = belief_eta_old[3 * i + k] - msg_eta_old[3 * slot_i + k];
            }

            if (eta_only) {
                computeEtaOnlyMessageHost(
                    graph.eta_i.data() + 3 * e, graph.eta_j.data() + 3 * e,
                    cavity_eta_j,
                    fixed_eta_map.data() + 9 * slot_i,
                    msg_eta_old.data() + 3 * slot_i,
                    msg_eta_new.data() + 3 * slot_i);

                computeEtaOnlyMessageHost(
                    graph.eta_j.data() + 3 * e, graph.eta_i.data() + 3 * e,
                    cavity_eta_i,
                    fixed_eta_map.data() + 9 * slot_j,
                    msg_eta_old.data() + 3 * slot_j,
                    msg_eta_new.data() + 3 * slot_j);
            } else {
                double cavity_lam_j[9];
                double cavity_lam_i[9];
                for (int k = 0; k < 9; ++k) {
                    cavity_lam_j[k] = belief_lam_old[9 * j + k] - msg_lam_old[9 * slot_j + k];
                    cavity_lam_i[k] = belief_lam_old[9 * i + k] - msg_lam_old[9 * slot_i + k];
                }
                computeGbpMessageHost(
                    graph.hii.data() + 9 * e, graph.hij.data() + 9 * e, graph.hji.data() + 9 * e, graph.hjj.data() + 9 * e,
                    graph.eta_i.data() + 3 * e, graph.eta_j.data() + 3 * e,
                    cavity_lam_j, cavity_eta_j,
                    msg_lam_old.data() + 9 * slot_i, msg_eta_old.data() + 3 * slot_i,
                    damping,
                    msg_lam_new.data() + 9 * slot_i, msg_eta_new.data() + 3 * slot_i);

                computeGbpMessageHost(
                    graph.hjj.data() + 9 * e, graph.hji.data() + 9 * e, graph.hij.data() + 9 * e, graph.hii.data() + 9 * e,
                    graph.eta_j.data() + 3 * e, graph.eta_i.data() + 3 * e,
                    cavity_lam_i, cavity_eta_i,
                    msg_lam_old.data() + 9 * slot_j, msg_eta_old.data() + 3 * slot_j,
                    damping,
                    msg_lam_new.data() + 9 * slot_j, msg_eta_new.data() + 3 * slot_j);
            }
        }

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
        for (int row = 0; row < incoming.rows; ++row) {
            const int node = incoming.nodes[row];
            double lam[9];
            double eta[3];
            if (eta_only && first_eta_only) {
                for (int k = 0; k < 9; ++k) {
                    lam[k] = belief_lam_old[9 * node + k];
                }
            } else if (!eta_only) {
                for (int k = 0; k < 9; ++k) {
                    lam[k] = graph.unary_lam[9 * node + k];
                }
            }
            for (int k = 0; k < 3; ++k) {
                eta[k] = graph.unary_eta[3 * node + k];
            }
            const int cap = incoming.caps[row];
            const int offset = incoming.offsets[row];
            for (int s = 0; s < cap; ++s) {
                const int slot = incoming.slots[offset + s];
                if (slot < 0) {
                    continue;
                }
                if (!eta_only) {
                    for (int k = 0; k < 9; ++k) {
                        lam[k] += msg_lam_new[9 * slot + k];
                    }
                }
                for (int k = 0; k < 3; ++k) {
                    eta[k] += msg_eta_new[3 * slot + k];
                }
            }
            if (!eta_only || first_eta_only) {
                for (int k = 0; k < 9; ++k) {
                    belief_lam_new[9 * node + k] = lam[k];
                }
            }
            for (int k = 0; k < 3; ++k) {
                belief_eta_new[3 * node + k] = eta[k];
            }
        }
    }

    if ((sweeps & 1) == 0) {
        out_belief_lam = std::move(belief_lam_a);
        out_belief_eta = std::move(belief_eta_a);
    } else {
        out_belief_lam = std::move(belief_lam_b);
        out_belief_eta = std::move(belief_eta_b);
    }
}

std::vector<double> beliefMeanVector(const std::vector<double>& belief_lam, const std::vector<double>& belief_eta, int n) {
    std::vector<double> mean(static_cast<size_t>(n) * 3, 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int node = 0; node < n; ++node) {
        solve3SpdHost(belief_lam.data() + 9 * node, belief_eta.data() + 3 * node, mean.data() + 3 * node);
    }
    return mean;
}

void cpuSweep(const LinearSystem& sys, std::vector<float>& x, std::vector<float>& x_next, float omega) {
    for (int node = 0; node < sys.n; ++node) {
        float accum[3] = {
            sys.rhs[3 * node + 0],
            sys.rhs[3 * node + 1],
            sys.rhs[3 * node + 2]
        };
        for (const DirectedBlock& nb : sys.neighbors[node]) {
            const float* h = nb.block.data();
            const float x0 = x[3 * nb.to + 0];
            const float x1 = x[3 * nb.to + 1];
            const float x2 = x[3 * nb.to + 2];
            accum[0] -= h[0] * x0 + h[1] * x1 + h[2] * x2;
            accum[1] -= h[3] * x0 + h[4] * x1 + h[5] * x2;
            accum[2] -= h[6] * x0 + h[7] * x1 + h[8] * x2;
        }
        float solved[3];
        solve3SpdHost(sys.diag.data() + 9 * node, accum, solved);
        x_next[3 * node + 0] = (1.0f - omega) * x[3 * node + 0] + omega * solved[0];
        x_next[3 * node + 1] = (1.0f - omega) * x[3 * node + 1] + omega * solved[1];
        x_next[3 * node + 2] = (1.0f - omega) * x[3 * node + 2] + omega * solved[2];
    }
    x.swap(x_next);
}

std::vector<BucketDevice> uploadBuckets(const std::vector<BucketHost>& host) {
    std::vector<BucketDevice> dev;
    dev.reserve(host.size());
    for (const BucketHost& h : host) {
        BucketDevice d;
        d.cap = h.cap;
        d.rows = h.rows;
        cudaCheck(cudaMalloc(&d.nodes, h.nodes.size() * sizeof(int)), "cudaMalloc bucket nodes");
        cudaCheck(cudaMalloc(&d.cols, h.cols.size() * sizeof(int)), "cudaMalloc bucket cols");
        cudaCheck(cudaMalloc(&d.blocks, h.blocks.size() * sizeof(float)), "cudaMalloc bucket blocks");
        cudaCheck(cudaMemcpy(d.nodes, h.nodes.data(), h.nodes.size() * sizeof(int), cudaMemcpyHostToDevice), "copy bucket nodes");
        cudaCheck(cudaMemcpy(d.cols, h.cols.data(), h.cols.size() * sizeof(int), cudaMemcpyHostToDevice), "copy bucket cols");
        cudaCheck(cudaMemcpy(d.blocks, h.blocks.data(), h.blocks.size() * sizeof(float), cudaMemcpyHostToDevice), "copy bucket blocks");
        dev.push_back(d);
    }
    return dev;
}

void freeBuckets(std::vector<BucketDevice>& dev) {
    for (BucketDevice& d : dev) {
        cudaFree(d.nodes);
        cudaFree(d.cols);
        cudaFree(d.blocks);
        d.nodes = nullptr;
        d.cols = nullptr;
        d.blocks = nullptr;
    }
}

template <typename T>
void uploadVector(T*& dst, const std::vector<T>& src, const char* name) {
    cudaCheck(cudaMalloc(&dst, src.size() * sizeof(T)), name);
    if (!src.empty()) {
        cudaCheck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), name);
    }
}

ProblemDevice uploadProblemDevice(const Problem& problem) {
    ProblemDevice dev;
    dev.n = static_cast<int>(problem.init_poses.size());
    dev.m = static_cast<int>(problem.edges.size());
    std::vector<double> poses(static_cast<size_t>(dev.n) * 3);
    for (int i = 0; i < dev.n; ++i) {
        const Pose2& p = problem.init_poses[static_cast<size_t>(i)];
        poses[static_cast<size_t>(i) * 3 + 0] = p.x;
        poses[static_cast<size_t>(i) * 3 + 1] = p.y;
        poses[static_cast<size_t>(i) * 3 + 2] = p.theta;
    }
    std::vector<double> measurements(static_cast<size_t>(dev.m) * 3);
    std::vector<double> information(static_cast<size_t>(dev.m) * 9);
    for (int e = 0; e < dev.m; ++e) {
        const Edge2& edge = problem.edges[static_cast<size_t>(e)];
        for (int k = 0; k < 3; ++k) {
            measurements[static_cast<size_t>(e) * 3 + k] = edge.measurement[static_cast<size_t>(k)];
        }
        for (int k = 0; k < 9; ++k) {
            information[static_cast<size_t>(e) * 9 + k] = edge.information[static_cast<size_t>(k)];
        }
    }
    std::vector<double> anchor_pose = {problem.anchor_pose.x, problem.anchor_pose.y, problem.anchor_pose.theta};
    std::vector<double> anchor_info(problem.anchor_information.begin(), problem.anchor_information.end());
    uploadVector(dev.poses, poses, "upload problem poses");
    uploadVector(dev.measurements, measurements, "upload problem measurements");
    uploadVector(dev.information, information, "upload problem information");
    uploadVector(dev.anchor_pose, anchor_pose, "upload problem anchor pose");
    uploadVector(dev.anchor_info, anchor_info, "upload problem anchor info");
    return dev;
}

void freeProblemDevice(ProblemDevice& dev) {
    cudaFree(dev.poses);
    cudaFree(dev.measurements);
    cudaFree(dev.information);
    cudaFree(dev.anchor_pose);
    cudaFree(dev.anchor_info);
    dev = {};
}

std::vector<double> packLam9ToLam6(const std::vector<double>& lam9, int n) {
    std::vector<double> lam6(static_cast<size_t>(n) * 6, 0.0);
    for (int i = 0; i < n; ++i) {
        const double* src = lam9.data() + static_cast<size_t>(i) * 9;
        double* dst = lam6.data() + static_cast<size_t>(i) * 6;
        dst[0] = src[0];
        dst[1] = 0.5 * (src[1] + src[3]);
        dst[2] = 0.5 * (src[2] + src[6]);
        dst[3] = src[4];
        dst[4] = 0.5 * (src[5] + src[7]);
        dst[5] = src[8];
    }
    return lam6;
}

GbpDevice uploadGbpDevice(const GbpGraph& graph, const IncomingFlatHost& incoming) {
    GbpDevice dev;
    dev.n = graph.n;
    dev.m = graph.m;
    dev.num_slots = 2 * graph.m;
    dev.incoming_total_slots = incoming.total_slots;
    uploadVector(dev.factor_i, graph.factor_i, "upload factor_i");
    uploadVector(dev.factor_j, graph.factor_j, "upload factor_j");
    uploadVector(dev.incoming_nodes, incoming.nodes, "upload incoming_nodes");
    uploadVector(dev.incoming_node_rows, incoming.node_rows, "upload incoming_node_rows");
    uploadVector(dev.incoming_caps, incoming.caps, "upload incoming_caps");
    uploadVector(dev.incoming_offsets, incoming.offsets, "upload incoming_offsets");
    uploadVector(dev.incoming_slots, incoming.slots, "upload incoming_slots");
    uploadVector(dev.hii, graph.hii, "upload hii");
    uploadVector(dev.hij, graph.hij, "upload hij");
    uploadVector(dev.hji, graph.hji, "upload hji");
    uploadVector(dev.hjj, graph.hjj, "upload hjj");
    uploadVector(dev.eta_i, graph.eta_i, "upload eta_i");
    uploadVector(dev.eta_j, graph.eta_j, "upload eta_j");
    const std::vector<double> unary_lam6 = packLam9ToLam6(graph.unary_lam, graph.n);
    uploadVector(dev.unary_lam, unary_lam6, "upload unary_lam6");
    uploadVector(dev.unary_eta, graph.unary_eta, "upload unary_eta");

    const size_t msg_lam_bytes = static_cast<size_t>(dev.num_slots) * 6 * sizeof(double);
    const size_t msg_eta_bytes = static_cast<size_t>(dev.num_slots) * 3 * sizeof(double);
    const size_t belief_lam_bytes = static_cast<size_t>(dev.n) * 6 * sizeof(double);
    const size_t belief_eta_bytes = static_cast<size_t>(dev.n) * 3 * sizeof(double);
    const size_t fixed_eta_map_bytes = static_cast<size_t>(dev.num_slots) * 9 * sizeof(double);
    cudaCheck(cudaMalloc(&dev.msg_lam_a, msg_lam_bytes), "malloc msg_lam_a");
    cudaCheck(cudaMalloc(&dev.msg_lam_b, msg_lam_bytes), "malloc msg_lam_b");
    cudaCheck(cudaMalloc(&dev.msg_eta_a, msg_eta_bytes), "malloc msg_eta_a");
    cudaCheck(cudaMalloc(&dev.msg_eta_b, msg_eta_bytes), "malloc msg_eta_b");
    cudaCheck(cudaMalloc(&dev.belief_lam_a, belief_lam_bytes), "malloc belief_lam_a");
    cudaCheck(cudaMalloc(&dev.belief_lam_b, belief_lam_bytes), "malloc belief_lam_b");
    cudaCheck(cudaMalloc(&dev.belief_eta_a, belief_eta_bytes), "malloc belief_eta_a");
    cudaCheck(cudaMalloc(&dev.belief_eta_b, belief_eta_bytes), "malloc belief_eta_b");
    cudaCheck(cudaMalloc(&dev.fixed_eta_map, fixed_eta_map_bytes), "malloc fixed_eta_map");
    return dev;
}

void updateGbpDeviceNumeric(const GbpDevice& dev, const GbpGraph& graph) {
    if (dev.n != graph.n || dev.m != graph.m) {
        throw std::runtime_error("updateGbpDeviceNumeric topology dimension mismatch");
    }
    cudaCheck(cudaMemcpy(dev.hii, graph.hii.data(), graph.hii.size() * sizeof(double), cudaMemcpyHostToDevice), "update hii");
    cudaCheck(cudaMemcpy(dev.hij, graph.hij.data(), graph.hij.size() * sizeof(double), cudaMemcpyHostToDevice), "update hij");
    cudaCheck(cudaMemcpy(dev.hji, graph.hji.data(), graph.hji.size() * sizeof(double), cudaMemcpyHostToDevice), "update hji");
    cudaCheck(cudaMemcpy(dev.hjj, graph.hjj.data(), graph.hjj.size() * sizeof(double), cudaMemcpyHostToDevice), "update hjj");
    cudaCheck(cudaMemcpy(dev.eta_i, graph.eta_i.data(), graph.eta_i.size() * sizeof(double), cudaMemcpyHostToDevice), "update eta_i");
    cudaCheck(cudaMemcpy(dev.eta_j, graph.eta_j.data(), graph.eta_j.size() * sizeof(double), cudaMemcpyHostToDevice), "update eta_j");
    const std::vector<double> unary_lam6 = packLam9ToLam6(graph.unary_lam, graph.n);
    cudaCheck(cudaMemcpy(dev.unary_lam, unary_lam6.data(), unary_lam6.size() * sizeof(double), cudaMemcpyHostToDevice), "update unary_lam6");
    cudaCheck(cudaMemcpy(dev.unary_eta, graph.unary_eta.data(), graph.unary_eta.size() * sizeof(double), cudaMemcpyHostToDevice), "update unary_eta");
}

CoarseBlockPatternDevice uploadCoarseBlockPatternDevice(const CoarseBlockPatternHost& host) {
    CoarseBlockPatternDevice dev;
    dev.groups = host.groups;
    dev.coarse_dim = host.coarse_dim;
    dev.block_dim = host.block_dim;
    dev.block_count = host.block_count;
    uploadVector(dev.block_rows, host.block_rows, "upload coarse block rows");
    uploadVector(dev.block_cols, host.block_cols, "upload coarse block cols");
    uploadVector(dev.factor_slot_ii, host.factor_slot_ii, "upload coarse factor slot ii");
    uploadVector(dev.factor_slot_ij, host.factor_slot_ij, "upload coarse factor slot ij");
    uploadVector(dev.factor_slot_ji, host.factor_slot_ji, "upload coarse factor slot ji");
    uploadVector(dev.factor_slot_jj, host.factor_slot_jj, "upload coarse factor slot jj");
    return dev;
}

void freeCoarseBlockPatternDevice(CoarseBlockPatternDevice& dev) {
    cudaFree(dev.block_rows);
    cudaFree(dev.block_cols);
    cudaFree(dev.factor_slot_ii);
    cudaFree(dev.factor_slot_ij);
    cudaFree(dev.factor_slot_ji);
    cudaFree(dev.factor_slot_jj);
    dev = {};
}

SvdBasisDevice uploadSvdBasisDevice(const SvdBasisHost& host) {
    SvdBasisDevice dev;
    dev.groups = host.groups;
    dev.r = host.r;
    dev.coarse_dim = host.coarse_dim;
    uploadVector(dev.var_group, host.var_group, "upload svd var_group");
    uploadVector(dev.var_basis, host.var_basis, "upload svd var_basis");
    return dev;
}

struct GpuSvdBasisBuildResult {
    SvdBasisHost host;
    SvdBasisDevice device;
    std::vector<double> selected_values;
    double assemble_ms = 0.0;
    double eig_ms = 0.0;
    double extract_ms = 0.0;
    double upload_ms = 0.0;
    bool device_reused = false;
};

struct GpuWarmRayleighWorkspace {
    int groups = 0;
    int dim = 0;
    int p = 0;
    double* d_group_mats = nullptr;
    double* d_factor_mats = nullptr;
    double* d_q = nullptr;
    double* d_small_mats = nullptr;
    double* d_evals = nullptr;
    double* d_work = nullptr;
    double** d_factor_ptrs = nullptr;
    double** d_rhs_ptrs = nullptr;
    int* d_info = nullptr;
    int lwork = 0;
    cusolverDnHandle_t solver = nullptr;
    cublasHandle_t blas = nullptr;
    syevjInfo_t params = nullptr;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void init(int groups_in, int dim_in, int p_in, double eig_tol, int eig_max_sweeps) {
        if (groups == groups_in && dim == dim_in && p == p_in && solver != nullptr) {
            cusolverCheck(cusolverDnXsyevjSetTolerance(params, eig_tol), "warm workspace set tol");
            cusolverCheck(cusolverDnXsyevjSetMaxSweeps(params, eig_max_sweeps), "warm workspace set sweeps");
            return;
        }
        destroy();
        groups = groups_in;
        dim = dim_in;
        p = p_in;
        const size_t matrix_values = static_cast<size_t>(groups) * dim * dim;
        const size_t q_values = static_cast<size_t>(groups) * dim * p;
        const size_t small_values = static_cast<size_t>(groups) * p * p;
        cudaCheck(cudaMalloc(&d_group_mats, matrix_values * sizeof(double)), "malloc warm workspace group mats");
        cudaCheck(cudaMalloc(&d_factor_mats, matrix_values * sizeof(double)), "malloc warm workspace factor mats");
        cudaCheck(cudaMalloc(&d_q, q_values * sizeof(double)), "malloc warm workspace q");
        cudaCheck(cudaMalloc(&d_small_mats, small_values * sizeof(double)), "malloc warm workspace small mats");
        cudaCheck(cudaMalloc(&d_evals, static_cast<size_t>(groups) * p * sizeof(double)), "malloc warm workspace evals");
        cudaCheck(cudaMalloc(&d_factor_ptrs, static_cast<size_t>(groups) * sizeof(double*)), "malloc warm workspace factor ptrs");
        cudaCheck(cudaMalloc(&d_rhs_ptrs, static_cast<size_t>(groups) * sizeof(double*)), "malloc warm workspace rhs ptrs");
        cudaCheck(cudaMalloc(&d_info, static_cast<size_t>(groups) * sizeof(int)), "malloc warm workspace info");
        cudaCheck(cudaEventCreate(&ev0), "create warm workspace event 0");
        cudaCheck(cudaEventCreate(&ev1), "create warm workspace event 1");
        cusolverCheck(cusolverDnCreate(&solver), "warm workspace cusolverDnCreate");
        cublasCheck(cublasCreate(&blas), "warm workspace cublasCreate");
        cusolverCheck(cusolverDnCreateSyevjInfo(&params), "warm workspace create syevj info");
        cusolverCheck(cusolverDnXsyevjSetTolerance(params, eig_tol), "warm workspace set tol");
        cusolverCheck(cusolverDnXsyevjSetMaxSweeps(params, eig_max_sweeps), "warm workspace set sweeps");
        cusolverCheck(cusolverDnXsyevjSetSortEig(params, 1), "warm workspace sort eig");
        cusolverCheck(cusolverDnDsyevjBatched_bufferSize(
                          solver,
                          CUSOLVER_EIG_MODE_VECTOR,
                          CUBLAS_FILL_MODE_LOWER,
                          p,
                          d_small_mats,
                          p,
                          d_evals,
                          &lwork,
                          params,
                          groups),
                      "warm workspace syevj bufferSize");
        cudaCheck(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)), "malloc warm workspace eig work");
    }

    template <typename Fn>
    double eventMs(Fn&& fn) {
        cudaCheck(cudaEventRecord(ev0), "record warm workspace start");
        fn();
        cudaCheck(cudaEventRecord(ev1), "record warm workspace stop");
        cudaCheck(cudaEventSynchronize(ev1), "sync warm workspace stop");
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed warm workspace");
        return static_cast<double>(ms);
    }

    void destroy() {
        if (params != nullptr) {
            cusolverDnDestroySyevjInfo(params);
            params = nullptr;
        }
        if (blas != nullptr) {
            cublasDestroy(blas);
            blas = nullptr;
        }
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
            solver = nullptr;
        }
        if (ev0 != nullptr) {
            cudaEventDestroy(ev0);
            ev0 = nullptr;
        }
        if (ev1 != nullptr) {
            cudaEventDestroy(ev1);
            ev1 = nullptr;
        }
        cudaFree(d_group_mats);
        cudaFree(d_factor_mats);
        cudaFree(d_q);
        cudaFree(d_small_mats);
        cudaFree(d_evals);
        cudaFree(d_work);
        cudaFree(d_factor_ptrs);
        cudaFree(d_rhs_ptrs);
        cudaFree(d_info);
        d_group_mats = nullptr;
        d_factor_mats = nullptr;
        d_q = nullptr;
        d_small_mats = nullptr;
        d_evals = nullptr;
        d_work = nullptr;
        d_factor_ptrs = nullptr;
        d_rhs_ptrs = nullptr;
        d_info = nullptr;
        lwork = 0;
        groups = 0;
        dim = 0;
        p = 0;
    }

    ~GpuWarmRayleighWorkspace() {
        destroy();
    }
};

struct GpuFullSvdWorkspace {
    int n = 0;
    int groups = 0;
    int dim = 0;
    int r = 0;
    SvdBasisDevice basis;
    double* d_group_mats = nullptr;
    double* d_evals = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    int lwork = 0;
    cusolverDnHandle_t solver = nullptr;
    syevjInfo_t params = nullptr;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void init(
        int n_in,
        int groups_in,
        int dim_in,
        int r_in,
        int coarse_dim_in,
        const std::vector<int>& var_group_host,
        double eig_tol,
        int eig_max_sweeps
    ) {
        if (n == n_in && groups == groups_in && dim == dim_in && r == r_in && solver != nullptr) {
            cusolverCheck(cusolverDnXsyevjSetTolerance(params, eig_tol), "full workspace set tol");
            cusolverCheck(cusolverDnXsyevjSetMaxSweeps(params, eig_max_sweeps), "full workspace set sweeps");
            return;
        }
        destroy();
        n = n_in;
        groups = groups_in;
        dim = dim_in;
        r = r_in;
        basis.groups = groups;
        basis.r = r;
        basis.coarse_dim = coarse_dim_in;
        uploadVector(basis.var_group, var_group_host, "upload full workspace var_group");
        cudaCheck(cudaMalloc(&basis.var_basis, static_cast<size_t>(n) * 3 * r * sizeof(double)),
                  "malloc full workspace var_basis");
        const size_t matrix_values = static_cast<size_t>(groups) * dim * dim;
        cudaCheck(cudaMalloc(&d_group_mats, matrix_values * sizeof(double)), "malloc full workspace group mats");
        cudaCheck(cudaMalloc(&d_evals, static_cast<size_t>(groups) * dim * sizeof(double)), "malloc full workspace evals");
        cudaCheck(cudaMalloc(&d_info, static_cast<size_t>(groups) * sizeof(int)), "malloc full workspace info");
        cudaCheck(cudaEventCreate(&ev0), "create full workspace event 0");
        cudaCheck(cudaEventCreate(&ev1), "create full workspace event 1");
        cusolverCheck(cusolverDnCreate(&solver), "full workspace cusolverDnCreate");
        cusolverCheck(cusolverDnCreateSyevjInfo(&params), "full workspace create syevj info");
        cusolverCheck(cusolverDnXsyevjSetTolerance(params, eig_tol), "full workspace set tol");
        cusolverCheck(cusolverDnXsyevjSetMaxSweeps(params, eig_max_sweeps), "full workspace set sweeps");
        cusolverCheck(cusolverDnXsyevjSetSortEig(params, 1), "full workspace sort eig");
        cusolverCheck(cusolverDnDsyevjBatched_bufferSize(
                          solver,
                          CUSOLVER_EIG_MODE_VECTOR,
                          CUBLAS_FILL_MODE_LOWER,
                          dim,
                          d_group_mats,
                          dim,
                          d_evals,
                          &lwork,
                          params,
                          groups),
                      "full workspace syevj bufferSize");
        cudaCheck(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)), "malloc full workspace eig work");
    }

    template <typename Fn>
    double eventMs(Fn&& fn) {
        cudaCheck(cudaEventRecord(ev0), "record full workspace start");
        fn();
        cudaCheck(cudaEventRecord(ev1), "record full workspace stop");
        cudaCheck(cudaEventSynchronize(ev1), "sync full workspace stop");
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed full workspace");
        return static_cast<double>(ms);
    }

    void destroy() {
        if (params != nullptr) {
            cusolverDnDestroySyevjInfo(params);
            params = nullptr;
        }
        if (solver != nullptr) {
            cusolverDnDestroy(solver);
            solver = nullptr;
        }
        if (ev0 != nullptr) {
            cudaEventDestroy(ev0);
            ev0 = nullptr;
        }
        if (ev1 != nullptr) {
            cudaEventDestroy(ev1);
            ev1 = nullptr;
        }
        cudaFree(basis.var_group);
        cudaFree(basis.var_basis);
        cudaFree(d_group_mats);
        cudaFree(d_evals);
        cudaFree(d_work);
        cudaFree(d_info);
        basis = {};
        d_group_mats = nullptr;
        d_evals = nullptr;
        d_work = nullptr;
        d_info = nullptr;
        lwork = 0;
        n = 0;
        groups = 0;
        dim = 0;
        r = 0;
    }

    ~GpuFullSvdWorkspace() {
        destroy();
    }
};

SvdBasisHost makeContiguousSvdBasisMetadata(int n, int group_size, int r_reduced, bool allocate_host_basis) {
    SvdBasisHost basis;
    basis.r = r_reduced;
    basis.var_group.assign(static_cast<size_t>(n), -1);
    int start = 0;
    while (start + 2 * group_size <= n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(group_size);
        start += group_size;
    }
    if (start < n) {
        basis.group_start.push_back(start);
        basis.group_len.push_back(n - start);
    }
    basis.groups = static_cast<int>(basis.group_start.size());
    basis.coarse_dim = basis.groups * basis.r;
    if (allocate_host_basis) {
        basis.var_basis.assign(static_cast<size_t>(n) * 3 * basis.r, 0.0);
    }
    for (int g = 0; g < basis.groups; ++g) {
        const int g_start = basis.group_start[static_cast<size_t>(g)];
        const int g_len = basis.group_len[static_cast<size_t>(g)];
        if (3 * g_len < basis.r) {
            throw std::runtime_error("SVD basis group is smaller than r_reduced");
        }
        for (int local = 0; local < g_len; ++local) {
            basis.var_group[static_cast<size_t>(g_start + local)] = g;
        }
    }
    return basis;
}

GpuSvdBasisBuildResult buildGpuSvdBasisFromDevice(
    const GbpGraph& graph,
    const GbpDevice& dev,
    int group_size,
    int r_reduced,
    double eig_tol,
    int eig_max_sweeps,
    bool capture_values = false,
    GpuFullSvdWorkspace* workspace = nullptr
) {
    if (group_size <= 0 || r_reduced <= 0) {
        throw std::runtime_error("invalid GPU SVD basis group/r");
    }
    if (graph.n != dev.n || graph.m != dev.m) {
        throw std::runtime_error("GPU SVD basis graph/device mismatch");
    }
    GpuSvdBasisBuildResult out;
    out.host = makeContiguousSvdBasisMetadata(graph.n, group_size, r_reduced, false);
    for (int len : out.host.group_len) {
        if (len != group_size) {
            throw std::runtime_error("GPU SVD basis currently requires all groups to have exactly group_size variables");
        }
    }
    const int groups = out.host.groups;
    const int dim = 3 * group_size;
    if (r_reduced > dim) {
        throw std::runtime_error("r_reduced exceeds GPU SVD group dimension");
    }

    const size_t matrix_values = static_cast<size_t>(groups) * dim * dim;
    double* d_group_mats = nullptr;
    double* d_evals = nullptr;
    double* d_work = nullptr;
    int* d_info = nullptr;
    int lwork = 0;
    cusolverDnHandle_t solver = nullptr;
    syevjInfo_t params = nullptr;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    bool owns_local_workspace = workspace == nullptr;
    if (workspace != nullptr) {
        const auto upload_t0 = std::chrono::steady_clock::now();
        workspace->init(
            graph.n,
            groups,
            dim,
            r_reduced,
            out.host.coarse_dim,
            out.host.var_group,
            eig_tol,
            eig_max_sweeps);
        cudaCheck(cudaDeviceSynchronize(), "sync full svd workspace init");
        const auto upload_t1 = std::chrono::steady_clock::now();
        out.upload_ms = std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count();
        out.device = workspace->basis;
        out.device_reused = true;
        d_group_mats = workspace->d_group_mats;
        d_evals = workspace->d_evals;
        d_work = workspace->d_work;
        d_info = workspace->d_info;
        lwork = workspace->lwork;
        solver = workspace->solver;
        params = workspace->params;
    } else {
        out.device.groups = groups;
        out.device.r = r_reduced;
        out.device.coarse_dim = out.host.coarse_dim;
        const auto upload_t0 = std::chrono::steady_clock::now();
        uploadVector(out.device.var_group, out.host.var_group, "upload gpu svd var_group");
        cudaCheck(cudaMalloc(&out.device.var_basis, static_cast<size_t>(graph.n) * 3 * r_reduced * sizeof(double)),
                  "malloc gpu svd var_basis");
        cudaCheck(cudaDeviceSynchronize(), "sync gpu svd metadata upload");
        const auto upload_t1 = std::chrono::steady_clock::now();
        out.upload_ms = std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count();
        cudaCheck(cudaMalloc(&d_group_mats, matrix_values * sizeof(double)), "malloc gpu svd group mats");
        cudaCheck(cudaMalloc(&d_evals, static_cast<size_t>(groups) * dim * sizeof(double)), "malloc gpu svd evals");
        cudaCheck(cudaMalloc(&d_info, static_cast<size_t>(groups) * sizeof(int)), "malloc gpu svd info");
        cudaCheck(cudaEventCreate(&ev0), "create gpu svd event 0");
        cudaCheck(cudaEventCreate(&ev1), "create gpu svd event 1");
        cusolverCheck(cusolverDnCreate(&solver), "cusolverDnCreate");
        cusolverCheck(cusolverDnCreateSyevjInfo(&params), "cusolverDnCreateSyevjInfo");
        cusolverCheck(cusolverDnXsyevjSetTolerance(params, eig_tol), "cusolverDnXsyevjSetTolerance");
        cusolverCheck(cusolverDnXsyevjSetMaxSweeps(params, eig_max_sweeps), "cusolverDnXsyevjSetMaxSweeps");
        cusolverCheck(cusolverDnXsyevjSetSortEig(params, 1), "cusolverDnXsyevjSetSortEig");
        cusolverCheck(cusolverDnDsyevjBatched_bufferSize(
                          solver,
                          CUSOLVER_EIG_MODE_VECTOR,
                          CUBLAS_FILL_MODE_LOWER,
                          dim,
                          d_group_mats,
                          dim,
                          d_evals,
                          &lwork,
                          params,
                          groups),
                      "cusolverDnDsyevjBatched_bufferSize");
        cudaCheck(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)), "malloc gpu svd work");
    }
    auto eventMs = [&](auto fn) {
        if (workspace != nullptr) {
            return workspace->eventMs(fn);
        }
        cudaCheck(cudaEventRecord(ev0), "record gpu svd start");
        fn();
        cudaCheck(cudaEventRecord(ev1), "record gpu svd stop");
        cudaCheck(cudaEventSynchronize(ev1), "sync gpu svd stop");
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed gpu svd");
        return static_cast<double>(ms);
    };

    constexpr int threads = 128;
    const int node_blocks = (graph.n + threads - 1) / threads;
    const int factor_blocks = (graph.m + threads - 1) / threads;
    out.assemble_ms = eventMs([&]() {
        cudaCheck(cudaMemset(d_group_mats, 0, matrix_values * sizeof(double)), "zero gpu svd group mats");
        assembleGpuSvdUnaryMatrixKernel<<<node_blocks, threads>>>(
            graph.n,
            group_size,
            dim,
            dev.unary_lam,
            d_group_mats);
        cudaCheck(cudaGetLastError(), "launch assembleGpuSvdUnaryMatrixKernel");
        assembleGpuSvdBinaryMatrixKernel<<<factor_blocks, threads>>>(
            graph.m,
            group_size,
            dim,
            dev.factor_i,
            dev.factor_j,
            dev.hii,
            dev.hij,
            dev.hji,
            dev.hjj,
            d_group_mats);
        cudaCheck(cudaGetLastError(), "launch assembleGpuSvdBinaryMatrixKernel");
    });

    out.eig_ms = eventMs([&]() {
        cusolverCheck(cusolverDnDsyevjBatched(
                          solver,
                          CUSOLVER_EIG_MODE_VECTOR,
                          CUBLAS_FILL_MODE_LOWER,
                          dim,
                          d_group_mats,
                          dim,
                          d_evals,
                          d_work,
                          lwork,
                          d_info,
                          params,
                          groups),
                      "cusolverDnDsyevjBatched");
    });
    std::vector<int> info(static_cast<size_t>(groups), 0);
    cudaCheck(cudaMemcpy(info.data(), d_info, info.size() * sizeof(int), cudaMemcpyDeviceToHost),
              "download gpu svd info");
    for (int g = 0; g < groups; ++g) {
        if (info[static_cast<size_t>(g)] != 0) {
            throw std::runtime_error("GPU SVD basis cuSOLVER failed for group " + std::to_string(g));
        }
    }
    if (capture_values) {
        std::vector<double> evals(static_cast<size_t>(groups) * dim);
        cudaCheck(cudaMemcpy(evals.data(), d_evals, evals.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "download gpu svd evals");
        out.selected_values.assign(static_cast<size_t>(groups) * r_reduced, 0.0);
        for (int g = 0; g < groups; ++g) {
            for (int c = 0; c < r_reduced; ++c) {
                out.selected_values[static_cast<size_t>(g) * r_reduced + c] =
                    evals[static_cast<size_t>(g) * dim + c];
            }
        }
    }

    out.extract_ms = eventMs([&]() {
        extractGpuSvdBasisKernel<<<node_blocks, threads>>>(
            graph.n,
            group_size,
            dim,
            r_reduced,
            d_group_mats,
            out.device.var_basis);
        cudaCheck(cudaGetLastError(), "launch extractGpuSvdBasisKernel");
    });

    if (owns_local_workspace) {
        cusolverDnDestroySyevjInfo(params);
        cusolverDnDestroy(solver);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaFree(d_group_mats);
        cudaFree(d_evals);
        cudaFree(d_work);
        cudaFree(d_info);
    }
    return out;
}

GpuSvdBasisBuildResult buildGpuWarmRayleighBasisFromDevice(
    const GbpGraph& graph,
    const GbpDevice& dev,
    int group_size,
    int r_reduced,
    const SvdBasisDevice& warm_basis,
    double eig_tol,
    int eig_max_sweeps,
    GpuWarmRayleighWorkspace* workspace = nullptr,
    int inverse_iters = 1,
    bool capture_values = false
) {
    if (group_size <= 0 || r_reduced <= 0) {
        throw std::runtime_error("invalid warm GPU SVD basis group/r");
    }
    if (graph.n != dev.n || graph.m != dev.m) {
        throw std::runtime_error("warm GPU SVD basis graph/device mismatch");
    }
    GpuSvdBasisBuildResult out;
    out.host = makeContiguousSvdBasisMetadata(graph.n, group_size, r_reduced, false);
    for (int len : out.host.group_len) {
        if (len != group_size) {
            throw std::runtime_error("warm GPU SVD basis currently requires exact group_size variables");
        }
    }
    const int groups = out.host.groups;
    const int dim = 3 * group_size;
    const int p = std::min(dim, r_reduced + 4);
    const bool use_warm_basis = warm_basis.var_basis != nullptr;
    if (use_warm_basis && (warm_basis.r != r_reduced || warm_basis.groups != groups)) {
        throw std::runtime_error("warm GPU SVD basis metadata mismatch");
    }
    inverse_iters = std::max(1, inverse_iters);

    out.device.groups = groups;
    out.device.r = r_reduced;
    out.device.coarse_dim = out.host.coarse_dim;
    if (use_warm_basis) {
        out.device.var_group = warm_basis.var_group;
        out.device.var_basis = warm_basis.var_basis;
        out.device_reused = true;
    } else {
        const auto upload_t0 = std::chrono::steady_clock::now();
        uploadVector(out.device.var_group, out.host.var_group, "upload warm gpu svd var_group");
        cudaCheck(cudaMalloc(&out.device.var_basis, static_cast<size_t>(graph.n) * 3 * r_reduced * sizeof(double)),
                  "malloc warm gpu svd var_basis");
        cudaCheck(cudaDeviceSynchronize(), "sync warm gpu svd metadata upload");
        const auto upload_t1 = std::chrono::steady_clock::now();
        out.upload_ms = std::chrono::duration<double, std::milli>(upload_t1 - upload_t0).count();
    }

    GpuWarmRayleighWorkspace local_workspace;
    GpuWarmRayleighWorkspace* ws = workspace != nullptr ? workspace : &local_workspace;
    ws->init(groups, dim, p, eig_tol, eig_max_sweeps);
    double* d_group_mats = ws->d_group_mats;
    double* d_factor_mats = ws->d_factor_mats;
    double* d_q = ws->d_q;
    double* d_small_mats = ws->d_small_mats;
    double* d_evals = ws->d_evals;
    double* d_work = ws->d_work;
    double** d_factor_ptrs = ws->d_factor_ptrs;
    double** d_rhs_ptrs = ws->d_rhs_ptrs;
    int* d_info = ws->d_info;
    cusolverDnHandle_t solver = ws->solver;
    cublasHandle_t blas = ws->blas;
    syevjInfo_t params = ws->params;
    int lwork = ws->lwork;
    const size_t matrix_values = static_cast<size_t>(groups) * dim * dim;

    constexpr int threads = 128;
    const int node_blocks = (graph.n + threads - 1) / threads;
    const int factor_blocks = (graph.m + threads - 1) / threads;
    out.assemble_ms = ws->eventMs([&]() {
        cudaCheck(cudaMemset(d_group_mats, 0, matrix_values * sizeof(double)), "zero warm gpu svd group mats");
        assembleGpuSvdUnaryMatrixKernel<<<node_blocks, threads>>>(
            graph.n,
            group_size,
            dim,
            dev.unary_lam,
            d_group_mats);
        cudaCheck(cudaGetLastError(), "launch warm assembleGpuSvdUnaryMatrixKernel");
        assembleGpuSvdBinaryMatrixKernel<<<factor_blocks, threads>>>(
            graph.m,
            group_size,
            dim,
            dev.factor_i,
            dev.factor_j,
            dev.hii,
            dev.hij,
            dev.hji,
            dev.hjj,
            d_group_mats);
        cudaCheck(cudaGetLastError(), "launch warm assembleGpuSvdBinaryMatrixKernel");
        if (use_warm_basis) {
            buildWarmRayleighSubspaceKernel<<<groups, 1>>>(
                groups,
                group_size,
                dim,
                r_reduced,
                p,
                warm_basis.var_basis,
                d_q);
            cudaCheck(cudaGetLastError(), "launch buildWarmRayleighSubspaceKernel");
        } else {
            buildInitialRayleighSubspaceKernel<<<groups, 1>>>(
                groups,
                dim,
                p,
                d_q);
            cudaCheck(cudaGetLastError(), "launch buildInitialRayleighSubspaceKernel");
        }
        prepareWarmInverseIterationKernel<<<groups, 1>>>(
            groups,
            dim,
            p,
            d_group_mats,
            d_factor_mats,
            d_factor_ptrs,
            d_rhs_ptrs,
            d_q);
        cudaCheck(cudaGetLastError(), "launch prepareWarmInverseIterationKernel");
    });

    out.assemble_ms += ws->eventMs([&]() {
        cusolverCheck(cusolverDnDpotrfBatched(
                          solver,
                          CUBLAS_FILL_MODE_LOWER,
                          dim,
                          d_factor_ptrs,
                          dim,
                          d_info,
                          groups),
                      "warm cusolverDnDpotrfBatched");
        const double alpha = 1.0;
        for (int iter = 0; iter < inverse_iters; ++iter) {
            cublasCheck(cublasDtrsmBatched(
                            blas,
                            CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_N,
                            CUBLAS_DIAG_NON_UNIT,
                            dim,
                            p,
                            &alpha,
                            reinterpret_cast<const double* const*>(d_factor_ptrs),
                            dim,
                            d_rhs_ptrs,
                            dim,
                            groups),
                        "warm cublasDtrsmBatched lower");
            cublasCheck(cublasDtrsmBatched(
                            blas,
                            CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T,
                            CUBLAS_DIAG_NON_UNIT,
                            dim,
                            p,
                            &alpha,
                            reinterpret_cast<const double* const*>(d_factor_ptrs),
                            dim,
                            d_rhs_ptrs,
                            dim,
                            groups),
                        "warm cublasDtrsmBatched upper");
            orthonormalizeGroupColumnsKernel<<<groups, 1>>>(
                groups,
                dim,
                p,
                d_q);
            cudaCheck(cudaGetLastError(), "launch orthonormalizeGroupColumnsKernel");
        }
        computeWarmRayleighMatrixKernel<<<groups, 64>>>(
            groups,
            dim,
            p,
            d_group_mats,
            d_q,
            d_small_mats);
        cudaCheck(cudaGetLastError(), "launch computeWarmRayleighMatrixKernel");
    });

    out.eig_ms = ws->eventMs([&]() {
        cusolverCheck(cusolverDnDsyevjBatched(
                          solver,
                          CUSOLVER_EIG_MODE_VECTOR,
                          CUBLAS_FILL_MODE_LOWER,
                          p,
                          d_small_mats,
                          p,
                          d_evals,
                          d_work,
                          lwork,
                          d_info,
                          params,
                          groups),
                      "warm cusolverDnDsyevjBatched");
    });
    std::vector<int> info(static_cast<size_t>(groups), 0);
    cudaCheck(cudaMemcpy(info.data(), d_info, info.size() * sizeof(int), cudaMemcpyDeviceToHost),
              "download warm gpu svd info");
    for (int g = 0; g < groups; ++g) {
        if (info[static_cast<size_t>(g)] != 0) {
            throw std::runtime_error("warm GPU SVD basis cuSOLVER failed for group " + std::to_string(g));
        }
    }
    if (capture_values) {
        std::vector<double> evals(static_cast<size_t>(groups) * p);
        cudaCheck(cudaMemcpy(evals.data(), d_evals, evals.size() * sizeof(double), cudaMemcpyDeviceToHost),
                  "download warm gpu svd evals");
        out.selected_values.assign(static_cast<size_t>(groups) * r_reduced, 0.0);
        for (int g = 0; g < groups; ++g) {
            for (int c = 0; c < r_reduced; ++c) {
                out.selected_values[static_cast<size_t>(g) * r_reduced + c] =
                    evals[static_cast<size_t>(g) * p + c];
            }
        }
    }

    out.extract_ms = ws->eventMs([&]() {
        extractWarmRayleighBasisKernel<<<node_blocks, threads>>>(
            graph.n,
            group_size,
            dim,
            r_reduced,
            p,
            d_q,
            d_small_mats,
            out.device.var_basis);
        cudaCheck(cudaGetLastError(), "launch extractWarmRayleighBasisKernel");
    });

    return out;
}

void freeSvdBasisDevice(SvdBasisDevice& dev) {
    cudaFree(dev.var_group);
    cudaFree(dev.var_basis);
    dev = {};
}

void resetGbpDevice(const GbpDevice& dev) {
    const size_t msg_lam_bytes = static_cast<size_t>(dev.num_slots) * 6 * sizeof(double);
    const size_t msg_eta_bytes = static_cast<size_t>(dev.num_slots) * 3 * sizeof(double);
    const size_t belief_lam_bytes = static_cast<size_t>(dev.n) * 6 * sizeof(double);
    const size_t belief_eta_bytes = static_cast<size_t>(dev.n) * 3 * sizeof(double);
    cudaCheck(cudaMemset(dev.msg_lam_a, 0, msg_lam_bytes), "zero msg_lam_a");
    cudaCheck(cudaMemset(dev.msg_lam_b, 0, msg_lam_bytes), "zero msg_lam_b");
    cudaCheck(cudaMemset(dev.msg_eta_a, 0, msg_eta_bytes), "zero msg_eta_a");
    cudaCheck(cudaMemset(dev.msg_eta_b, 0, msg_eta_bytes), "zero msg_eta_b");
    cudaCheck(cudaMemcpy(dev.belief_lam_a, dev.unary_lam, belief_lam_bytes, cudaMemcpyDeviceToDevice), "init belief_lam_a");
    cudaCheck(cudaMemcpy(dev.belief_eta_a, dev.unary_eta, belief_eta_bytes, cudaMemcpyDeviceToDevice), "init belief_eta_a");
    cudaCheck(cudaMemset(dev.belief_lam_b, 0, belief_lam_bytes), "zero belief_lam_b");
    cudaCheck(cudaMemset(dev.belief_eta_b, 0, belief_eta_bytes), "zero belief_eta_b");
}

void freeGbpDevice(GbpDevice& dev) {
    cudaFree(dev.factor_i);
    cudaFree(dev.factor_j);
    cudaFree(dev.incoming_nodes);
    cudaFree(dev.incoming_node_rows);
    cudaFree(dev.incoming_caps);
    cudaFree(dev.incoming_offsets);
    cudaFree(dev.incoming_slots);
    cudaFree(dev.hii);
    cudaFree(dev.hij);
    cudaFree(dev.hji);
    cudaFree(dev.hjj);
    cudaFree(dev.eta_i);
    cudaFree(dev.eta_j);
    cudaFree(dev.unary_lam);
    cudaFree(dev.unary_eta);
    cudaFree(dev.msg_lam_a);
    cudaFree(dev.msg_lam_b);
    cudaFree(dev.msg_eta_a);
    cudaFree(dev.msg_eta_b);
    cudaFree(dev.belief_lam_a);
    cudaFree(dev.belief_lam_b);
    cudaFree(dev.belief_eta_a);
    cudaFree(dev.belief_eta_b);
    cudaFree(dev.fixed_eta_map);
    dev = {};
}

struct GbpCoopLaunchConfig {
    int threads = 32;
    int sms = 0;
    int active_blocks_per_sm = 0;
    int capacity_blocks = 0;
    int work_blocks = 0;
    int launched_blocks = 0;
    double idle_thread_fraction_factor = 0.0;
    double idle_thread_fraction_variable = 0.0;
};

int ceilDivInt(int a, int b) {
    return (a + b - 1) / b;
}

int cooperativeBlockCapacityForGbp(int threads, int* out_active_blocks_per_sm = nullptr, int* out_sms = nullptr) {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_persistent");
    }
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    if (out_active_blocks_per_sm != nullptr) {
        *out_active_blocks_per_sm = active_blocks_per_sm;
    }
    if (out_sms != nullptr) {
        *out_sms = prop.multiProcessorCount;
    }
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeBlockCount() {
    constexpr int threads = 32;
    return cooperativeBlockCapacityForGbp(threads);
}

GbpCoopLaunchConfig computeGbpCoopLaunchConfig(
    int n,
    int m,
    int threads,
    const std::string& policy
) {
    GbpCoopLaunchConfig cfg;
    cfg.threads = threads;
    cfg.capacity_blocks = cooperativeBlockCapacityForGbp(
        threads, &cfg.active_blocks_per_sm, &cfg.sms);
    cfg.work_blocks = std::max(1, ceilDivInt(std::max(n, m), threads));
    if (policy == "work_cap") {
        cfg.launched_blocks = std::min(cfg.capacity_blocks, cfg.work_blocks);
    } else {
        cfg.launched_blocks = cfg.capacity_blocks;
    }

    const double launched_threads =
        static_cast<double>(cfg.launched_blocks) * static_cast<double>(threads);
    const auto idle_fraction = [&](int work_items) {
        if (launched_threads <= 0.0) {
            return 0.0;
        }
        return std::max(0.0, launched_threads - static_cast<double>(work_items)) / launched_threads;
    };
    cfg.idle_thread_fraction_factor = idle_fraction(m);
    cfg.idle_thread_fraction_variable = idle_fraction(n);
    return cfg;
}

void printGbpCoopLaunchConfig(int n, int m, int threads, const std::string& policy) {
    const GbpCoopLaunchConfig cfg = computeGbpCoopLaunchConfig(n, m, threads, policy);
    std::cout << std::fixed << std::setprecision(3)
              << "coop_launch policy=" << policy
              << " threads=" << cfg.threads
              << " sms=" << cfg.sms
              << " active_blocks_per_sm=" << cfg.active_blocks_per_sm
              << " capacity_blocks=" << cfg.capacity_blocks
              << " work_blocks=" << cfg.work_blocks
              << " launched_blocks=" << cfg.launched_blocks
              << " idle_thread_fraction_factor=" << cfg.idle_thread_fraction_factor
              << " idle_thread_fraction_variable=" << cfg.idle_thread_fraction_variable
              << "\n";
}

struct GbpGraphSplitExec {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaStream_t stream = nullptr;
    int sweeps = 0;
    int message_blocks = 0;
    int belief_blocks = 0;
    int message_threads = 32;
    int belief_threads = 32;
};

void destroyGbpGraphSplitExec(GbpGraphSplitExec& graph_exec) {
    if (graph_exec.exec != nullptr) {
        cudaGraphExecDestroy(graph_exec.exec);
        graph_exec.exec = nullptr;
    }
    if (graph_exec.graph != nullptr) {
        cudaGraphDestroy(graph_exec.graph);
        graph_exec.graph = nullptr;
    }
    if (graph_exec.stream != nullptr) {
        cudaStreamDestroy(graph_exec.stream);
        graph_exec.stream = nullptr;
    }
    graph_exec = {};
}

GbpGraphSplitExec createGbpGraphSplitExec(
    const GbpDevice& dev,
    int sweeps,
    double damping,
    int schur_mode,
    int threads
) {
    GbpGraphSplitExec graph_exec;
    graph_exec.sweeps = sweeps;
    graph_exec.message_threads = threads;
    graph_exec.belief_threads = threads;
    graph_exec.message_blocks = std::max(1, ceilDivInt(dev.m, graph_exec.message_threads));
    graph_exec.belief_blocks = std::max(1, ceilDivInt(dev.n, graph_exec.belief_threads));

    cudaCheck(cudaStreamCreateWithFlags(&graph_exec.stream, cudaStreamNonBlocking),
              "create graph split stream");
    cudaCheck(cudaStreamBeginCapture(graph_exec.stream, cudaStreamCaptureModeGlobal),
              "begin graph split capture");

    for (int sweep = 0; sweep < sweeps; ++sweep) {
        const bool even = (sweep & 1) == 0;
        const double* msg_lam_old = even ? dev.msg_lam_a : dev.msg_lam_b;
        const double* msg_eta_old = even ? dev.msg_eta_a : dev.msg_eta_b;
        double* msg_lam_new = even ? dev.msg_lam_b : dev.msg_lam_a;
        double* msg_eta_new = even ? dev.msg_eta_b : dev.msg_eta_a;
        const double* belief_lam_old = even ? dev.belief_lam_a : dev.belief_lam_b;
        const double* belief_eta_old = even ? dev.belief_eta_a : dev.belief_eta_b;
        double* belief_lam_new = even ? dev.belief_lam_b : dev.belief_lam_a;
        double* belief_eta_new = even ? dev.belief_eta_b : dev.belief_eta_a;

        gbpSplitMessageKernel<<<graph_exec.message_blocks, graph_exec.message_threads, 0, graph_exec.stream>>>(
            dev.m,
            dev.factor_i,
            dev.factor_j,
            dev.hii,
            dev.hij,
            dev.hji,
            dev.hjj,
            dev.eta_i,
            dev.eta_j,
            msg_lam_old,
            msg_eta_old,
            belief_lam_old,
            belief_eta_old,
            msg_lam_new,
            msg_eta_new,
            damping,
            schur_mode);

        gbpSplitBeliefKernel<<<graph_exec.belief_blocks, graph_exec.belief_threads, 0, graph_exec.stream>>>(
            dev.n,
            dev.incoming_nodes,
            dev.incoming_caps,
            dev.incoming_offsets,
            dev.incoming_slots,
            dev.unary_lam,
            dev.unary_eta,
            msg_lam_new,
            msg_eta_new,
            belief_lam_new,
            belief_eta_new);
    }

    cudaCheck(cudaStreamEndCapture(graph_exec.stream, &graph_exec.graph),
              "end graph split capture");
    cudaCheck(cudaGraphInstantiate(&graph_exec.exec, graph_exec.graph, nullptr, nullptr, 0),
              "instantiate graph split");
    return graph_exec;
}

void launchGbpGraphSplit(const GbpGraphSplitExec& graph_exec) {
    cudaCheck(cudaGraphLaunch(graph_exec.exec, graph_exec.stream), "launch graph split");
}

int cooperativeOnTheFlyBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_on_the_fly");
    }
    constexpr int threads = 32;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpOnTheFlyCavityPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor on-the-fly");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeFullBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_full_persistent");
    }
    constexpr int threads = 32;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpFullPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor full");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeEtaOnlyBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_persistent");
    }
    constexpr int threads = 128;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpEtaOnlyPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor eta-only");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeTargetNodeBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_target_node");
    }
    constexpr int threads = 128;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpTargetNodePersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor target-node");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeDirectedBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_directed");
    }
    constexpr int threads = 128;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpDirectedPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor directed");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

int cooperativeWarpDirectedBlockCount() {
    cudaDeviceProp prop{};
    int device = 0;
    cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    if (!prop.cooperativeLaunch) {
        throw std::runtime_error("This GPU does not support cooperativeLaunch, required by gbp_warp_directed");
    }
    constexpr int threads = 32;
    int active_blocks_per_sm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                  &active_blocks_per_sm, gbpWarpDirectedPersistentKernel, threads, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor warp directed");
    active_blocks_per_sm = std::max(1, active_blocks_per_sm);
    return std::max(1, active_blocks_per_sm * prop.multiProcessorCount);
}

void launchGbpPersistent(
    const GbpDevice& dev,
    int sweeps,
    double damping,
    int schur_mode,
    int fixed_lambda_start,
    int launch_blocks,
    int launch_threads
) {
    const int threads = std::max(1, launch_threads);
    int n = dev.n;
    int m = dev.m;
    int s = sweeps;
    double d = damping;
    int mode = schur_mode;
    int fixed_start = fixed_lambda_start;
    int blocks = std::max(1, launch_blocks);
    int* factor_i = dev.factor_i;
    int* factor_j = dev.factor_j;
    int* incoming_nodes = dev.incoming_nodes;
    int* incoming_node_rows = dev.incoming_node_rows;
    int* incoming_caps = dev.incoming_caps;
    int* incoming_offsets = dev.incoming_offsets;
    int* incoming_slots = dev.incoming_slots;
    double* hii = dev.hii;
    double* hij = dev.hij;
    double* hji = dev.hji;
    double* hjj = dev.hjj;
    double* eta_i = dev.eta_i;
    double* eta_j = dev.eta_j;
    double* unary_lam = dev.unary_lam;
    double* unary_eta = dev.unary_eta;
    double* msg_lam_a = dev.msg_lam_a;
    double* msg_lam_b = dev.msg_lam_b;
    double* msg_eta_a = dev.msg_eta_a;
    double* msg_eta_b = dev.msg_eta_b;
    double* belief_lam_a = dev.belief_lam_a;
    double* belief_lam_b = dev.belief_lam_b;
    double* belief_eta_a = dev.belief_eta_a;
    double* belief_eta_b = dev.belief_eta_b;
    double* fixed_eta_map = dev.fixed_eta_map;

    constexpr bool kUseOnTheFlyCavityKernel = false;
    if (kUseOnTheFlyCavityKernel && fixed_lambda_start < 0) {
        constexpr int otf_threads = 32;
        int otf_blocks = cooperativeOnTheFlyBlockCount();
        void* otf_params[] = {
            &n,
            &m,
            &s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_node_rows,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &d,
            &mode
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpOnTheFlyCavityPersistentKernel),
                      dim3(otf_blocks), dim3(otf_threads), otf_params),
                  "launch gbpOnTheFlyCavityPersistentKernel");
        return;
    }

    constexpr bool kUseWarpDirectedNoFixedKernel = false;
    if (kUseWarpDirectedNoFixedKernel && fixed_lambda_start < 0 && schur_mode == 0) {
        constexpr int warp_threads = 128;
        int warp_blocks = cooperativeWarpDirectedBlockCount();
        void* warp_params[] = {
            &n,
            &m,
            &s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &d
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpWarpDirectedPersistentKernel),
                      dim3(warp_blocks), dim3(warp_threads), warp_params),
                  "launch gbpWarpDirectedPersistentKernel");
        return;
    }

    constexpr bool kUseFullOnlyKernel = false;
    if (kUseFullOnlyKernel && fixed_lambda_start < 0) {
        int full_blocks = cooperativeFullBlockCount();
        void* full_params[] = {
            &n,
            &m,
            &s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &d,
            &mode
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpFullPersistentKernel),
                      dim3(full_blocks), dim3(threads), full_params),
                  "launch gbpFullPersistentKernel");
        return;
    }

    constexpr bool kUseDirectedKernel = false;
    if (kUseDirectedKernel) {
        int directed_blocks = cooperativeDirectedBlockCount();
        void* directed_params[] = {
            &n,
            &m,
            &s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &fixed_eta_map,
            &d,
            &mode,
            &fixed_start
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpDirectedPersistentKernel),
                      dim3(directed_blocks), dim3(threads), directed_params),
                  "launch gbpDirectedPersistentKernel");
        return;
    }

    constexpr bool kUseTargetNodeKernel = false;
    if (kUseTargetNodeKernel) {
        int blocks = cooperativeTargetNodeBlockCount();
        void* target_params[] = {
            &n,
            &s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &fixed_eta_map,
            &d,
            &mode,
            &fixed_start
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpTargetNodePersistentKernel),
                      dim3(blocks), dim3(threads), target_params),
                  "launch gbpTargetNodePersistentKernel");
        return;
    }

    const int full_sweeps_before_eta = (fixed_lambda_start >= 0)
        ? std::min(s, fixed_lambda_start + 1)
        : s;
    const int eta_sweeps = (fixed_lambda_start >= 0 && s > full_sweeps_before_eta)
        ? (s - full_sweeps_before_eta)
        : 0;
    constexpr bool kUseSplitEtaOnlyKernel = false;
    if (kUseSplitEtaOnlyKernel && eta_sweeps > 0) {
        int full_s = full_sweeps_before_eta;
        int no_fixed = -1;
        int full_blocks = cooperativeBlockCount();
        void* full_params[] = {
            &n,
            &m,
            &full_s,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_lam,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &fixed_eta_map,
            &d,
            &mode,
            &no_fixed
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpPersistentKernel),
                      dim3(full_blocks), dim3(threads), full_params),
                  "launch gbpPersistentKernel full phase");

        int eta_s = eta_sweeps;
        int start_sweep = full_sweeps_before_eta;
        int eta_blocks = cooperativeEtaOnlyBlockCount();
        void* eta_params[] = {
            &n,
            &m,
            &eta_s,
            &start_sweep,
            &factor_i,
            &factor_j,
            &hii,
            &hij,
            &hji,
            &hjj,
            &eta_i,
            &eta_j,
            &incoming_nodes,
            &incoming_caps,
            &incoming_offsets,
            &incoming_slots,
            &unary_eta,
            &msg_lam_a,
            &msg_lam_b,
            &msg_eta_a,
            &msg_eta_b,
            &belief_lam_a,
            &belief_lam_b,
            &belief_eta_a,
            &belief_eta_b,
            &fixed_eta_map,
            &mode
        };
        cudaCheck(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(gbpEtaOnlyPersistentKernel),
                      dim3(eta_blocks), dim3(threads), eta_params),
                  "launch gbpEtaOnlyPersistentKernel");
        return;
    }

    void* params[] = {
        &n,
        &m,
        &s,
        &factor_i,
        &factor_j,
        &hii,
        &hij,
        &hji,
        &hjj,
        &eta_i,
        &eta_j,
        &incoming_nodes,
        &incoming_caps,
        &incoming_offsets,
        &incoming_slots,
        &unary_lam,
        &unary_eta,
        &msg_lam_a,
        &msg_lam_b,
        &msg_eta_a,
        &msg_eta_b,
        &belief_lam_a,
        &belief_lam_b,
        &belief_eta_a,
        &belief_eta_b,
        &fixed_eta_map,
        &d,
        &mode,
        &fixed_start
    };
    cudaCheck(cudaLaunchCooperativeKernel(
                  reinterpret_cast<void*>(gbpPersistentKernel),
                  dim3(blocks), dim3(threads), params),
              "launch gbpPersistentKernel");
}

void downloadGbpBelief(const GbpDevice& dev, int sweeps, std::vector<double>& belief_lam, std::vector<double>& belief_eta) {
    belief_lam.resize(static_cast<size_t>(dev.n) * 9);
    belief_eta.resize(static_cast<size_t>(dev.n) * 3);
    const bool even_sweeps = (sweeps & 1) == 0;
    const double* lam_src = even_sweeps ? dev.belief_lam_a : dev.belief_lam_b;
    const double* eta_src = even_sweeps ? dev.belief_eta_a : dev.belief_eta_b;
    std::vector<double> belief_lam6(static_cast<size_t>(dev.n) * 6);
    cudaCheck(cudaMemcpy(belief_lam6.data(), lam_src, belief_lam6.size() * sizeof(double), cudaMemcpyDeviceToHost),
              "download belief_lam6");
    for (int node = 0; node < dev.n; ++node) {
        const double* src = belief_lam6.data() + static_cast<size_t>(node) * 6;
        double* dst = belief_lam.data() + static_cast<size_t>(node) * 9;
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[1];
        dst[4] = src[3];
        dst[5] = src[4];
        dst[6] = src[2];
        dst[7] = src[4];
        dst[8] = src[5];
    }
    cudaCheck(cudaMemcpy(belief_eta.data(), eta_src, belief_eta.size() * sizeof(double), cudaMemcpyDeviceToHost),
              "download belief_eta");
}

double linearizeSe2GraphOnGpu(const ProblemDevice& problem_dev, const GbpDevice& dev, double huber_delta) {
    if (problem_dev.n != dev.n || problem_dev.m != dev.m) {
        throw std::runtime_error("linearizeSe2GraphOnGpu problem/device mismatch");
    }
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    cudaCheck(cudaEventCreate(&ev0), "create gpu linearize event 0");
    cudaCheck(cudaEventCreate(&ev1), "create gpu linearize event 1");
    constexpr int threads = 256;
    const int node_blocks = (dev.n + threads - 1) / threads;
    const int factor_blocks = (dev.m + threads - 1) / threads;
    cudaCheck(cudaEventRecord(ev0), "record gpu linearize start");
    zeroUnarySe2Kernel<<<node_blocks, threads>>>(dev.n, dev.unary_lam, dev.unary_eta);
    cudaCheck(cudaGetLastError(), "launch zeroUnarySe2Kernel");
    linearizeSe2FactorsKernel<<<factor_blocks, threads>>>(
        dev.m,
        problem_dev.poses,
        dev.factor_i,
        dev.factor_j,
        problem_dev.measurements,
        problem_dev.information,
        huber_delta,
        dev.hii,
        dev.hij,
        dev.hji,
        dev.hjj,
        dev.eta_i,
        dev.eta_j);
    cudaCheck(cudaGetLastError(), "launch linearizeSe2FactorsKernel");
    linearizeSe2AnchorKernel<<<1, 1>>>(
        problem_dev.poses,
        problem_dev.anchor_pose,
        problem_dev.anchor_info,
        dev.unary_lam,
        dev.unary_eta);
    cudaCheck(cudaGetLastError(), "launch linearizeSe2AnchorKernel");
    cudaCheck(cudaEventRecord(ev1), "record gpu linearize stop");
    cudaCheck(cudaEventSynchronize(ev1), "sync gpu linearize stop");
    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed gpu linearize");
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return static_cast<double>(ms);
}

double applyPoseDeltasOnGpu(const ProblemDevice& problem_dev, const double* d_delta) {
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;
    cudaCheck(cudaEventCreate(&ev0), "create gpu apply pose event 0");
    cudaCheck(cudaEventCreate(&ev1), "create gpu apply pose event 1");
    constexpr int threads = 256;
    const int node_blocks = (problem_dev.n + threads - 1) / threads;
    cudaCheck(cudaEventRecord(ev0), "record gpu apply pose start");
    applyPoseDeltasSe2Kernel<<<node_blocks, threads>>>(problem_dev.n, d_delta, problem_dev.poses);
    cudaCheck(cudaGetLastError(), "launch applyPoseDeltasSe2Kernel");
    cudaCheck(cudaEventRecord(ev1), "record gpu apply pose stop");
    cudaCheck(cudaEventSynchronize(ev1), "sync gpu apply pose stop");
    float ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed gpu apply pose");
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    return static_cast<double>(ms);
}

void downloadProblemDevicePoses(const ProblemDevice& problem_dev, Problem& problem) {
    std::vector<double> poses(static_cast<size_t>(problem_dev.n) * 3);
    cudaCheck(cudaMemcpy(poses.data(), problem_dev.poses, poses.size() * sizeof(double), cudaMemcpyDeviceToHost),
              "download problem device poses");
    for (int i = 0; i < problem_dev.n; ++i) {
        Pose2& p = problem.init_poses[static_cast<size_t>(i)];
        p.x = poses[static_cast<size_t>(i) * 3 + 0];
        p.y = poses[static_cast<size_t>(i) * 3 + 1];
        p.theta = poses[static_cast<size_t>(i) * 3 + 2];
    }
}

std::vector<double> downloadProblemDevicePoseVector(const ProblemDevice& problem_dev) {
    return downloadDeviceDoubles(
        problem_dev.poses,
        static_cast<size_t>(problem_dev.n) * 3,
        "download problem device pose vector");
}

std::vector<double> prolongCoarseDeltaHost(
    int n,
    int r,
    double coarse_scale,
    const std::vector<int>& var_group,
    const std::vector<double>& var_basis,
    const std::vector<double>& coarse_delta
) {
    std::vector<double> fine(static_cast<size_t>(n) * 3, 0.0);
    for (int node = 0; node < n; ++node) {
        const int g = var_group[static_cast<size_t>(node)];
        const double* B = var_basis.data() + static_cast<size_t>(node) * 3 * r;
        const double* dz = coarse_delta.data() + static_cast<size_t>(g) * r;
        double* out = fine.data() + static_cast<size_t>(node) * 3;
        for (int c = 0; c < r; ++c) {
            const double z = coarse_scale * dz[c];
            out[0] += B[3 * c + 0] * z;
            out[1] += B[3 * c + 1] * z;
            out[2] += B[3 * c + 2] * z;
        }
    }
    return fine;
}

void dumpFinalGbpState(TrajectoryDumpWriter* dump, const GbpDevice& dev) {
    if (dump == nullptr || !dump->enabled()) {
        return;
    }
    dump->writeDoubles(
        TrajectoryTag::FinalMsgLamA,
        -1,
        -1,
        downloadDeviceDoubles(dev.msg_lam_a, static_cast<size_t>(dev.num_slots) * 6, "dump msg_lam_a"));
    dump->writeDoubles(
        TrajectoryTag::FinalMsgLamB,
        -1,
        -1,
        downloadDeviceDoubles(dev.msg_lam_b, static_cast<size_t>(dev.num_slots) * 6, "dump msg_lam_b"));
    dump->writeDoubles(
        TrajectoryTag::FinalMsgEtaA,
        -1,
        -1,
        downloadDeviceDoubles(dev.msg_eta_a, static_cast<size_t>(dev.num_slots) * 3, "dump msg_eta_a"));
    dump->writeDoubles(
        TrajectoryTag::FinalMsgEtaB,
        -1,
        -1,
        downloadDeviceDoubles(dev.msg_eta_b, static_cast<size_t>(dev.num_slots) * 3, "dump msg_eta_b"));
    dump->writeDoubles(
        TrajectoryTag::FinalBeliefLamA,
        -1,
        -1,
        downloadDeviceDoubles(dev.belief_lam_a, static_cast<size_t>(dev.n) * 6, "dump belief_lam_a"));
    dump->writeDoubles(
        TrajectoryTag::FinalBeliefLamB,
        -1,
        -1,
        downloadDeviceDoubles(dev.belief_lam_b, static_cast<size_t>(dev.n) * 6, "dump belief_lam_b"));
    dump->writeDoubles(
        TrajectoryTag::FinalBeliefEtaA,
        -1,
        -1,
        downloadDeviceDoubles(dev.belief_eta_a, static_cast<size_t>(dev.n) * 3, "dump belief_eta_a"));
    dump->writeDoubles(
        TrajectoryTag::FinalBeliefEtaB,
        -1,
        -1,
        downloadDeviceDoubles(dev.belief_eta_b, static_cast<size_t>(dev.n) * 3, "dump belief_eta_b"));
}

std::string lastWin32ErrorText(DWORD err) {
    char* buffer = nullptr;
    const DWORD n = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr,
        err,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<char*>(&buffer),
        0,
        nullptr);
    std::string out = (n > 0 && buffer != nullptr) ? std::string(buffer, n) : std::string("unknown error");
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    while (!out.empty() && (out.back() == '\r' || out.back() == '\n')) {
        out.pop_back();
    }
    return out;
}

std::string executableDirectory() {
    std::vector<char> path(MAX_PATH);
    DWORD n = GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size()));
    while (n == path.size()) {
        path.resize(path.size() * 2);
        n = GetModuleFileNameA(nullptr, path.data(), static_cast<DWORD>(path.size()));
    }
    if (n == 0) {
        return ".";
    }
    std::string s(path.data(), n);
    const size_t pos = s.find_last_of("\\/");
    return (pos == std::string::npos) ? "." : s.substr(0, pos);
}

std::string joinPath(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return b;
    }
    const char last = a.back();
    return (last == '\\' || last == '/') ? (a + b) : (a + "\\" + b);
}

bool fileExists(const std::string& path) {
    const DWORD attr = GetFileAttributesA(path.c_str());
    return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

struct DynamicCholmodApi {
    HMODULE module = nullptr;
    using start_t = int(__cdecl*)(cholmod_common*);
    using finish_t = int(__cdecl*)(cholmod_common*);
    using allocate_sparse_t = cholmod_sparse*(__cdecl*)(size_t, size_t, size_t, int, int, int, int, cholmod_common*);
    using free_sparse_t = int(__cdecl*)(cholmod_sparse**, cholmod_common*);
    using sort_t = int(__cdecl*)(cholmod_sparse*, cholmod_common*);
    using allocate_dense_t = cholmod_dense*(__cdecl*)(size_t, size_t, size_t, int, cholmod_common*);
    using free_dense_t = int(__cdecl*)(cholmod_dense**, cholmod_common*);
    using analyze_t = cholmod_factor*(__cdecl*)(cholmod_sparse*, cholmod_common*);
    using factorize_t = int(__cdecl*)(cholmod_sparse*, cholmod_factor*, cholmod_common*);
    using free_factor_t = int(__cdecl*)(cholmod_factor**, cholmod_common*);
    using solve_t = cholmod_dense*(__cdecl*)(int, cholmod_factor*, cholmod_dense*, cholmod_common*);

    start_t start = nullptr;
    finish_t finish = nullptr;
    allocate_sparse_t allocate_sparse = nullptr;
    free_sparse_t free_sparse = nullptr;
    sort_t sort = nullptr;
    allocate_dense_t allocate_dense = nullptr;
    free_dense_t free_dense = nullptr;
    analyze_t analyze = nullptr;
    factorize_t factorize = nullptr;
    free_factor_t free_factor = nullptr;
    solve_t solve = nullptr;
};

FARPROC loadRequiredProc(HMODULE module, const char* name) {
    FARPROC proc = GetProcAddress(module, name);
    if (proc == nullptr) {
        throw std::runtime_error(std::string("CHOLMOD missing export: ") + name);
    }
    return proc;
}

DynamicCholmodApi& dynamicCholmod() {
    static DynamicCholmodApi api;
    static bool initialized = false;
    if (initialized) {
        return api;
    }

    std::vector<std::string> dirs;
    if (const char* env = std::getenv("HGBP_CHOLMOD_BIN")) {
        dirs.emplace_back(env);
    }
    dirs.push_back("cpp\\vcpkg_installed\\x64-windows\\bin");
    const std::string exe_dir = executableDirectory();
    dirs.push_back(joinPath(exe_dir, "..\\..\\..\\vcpkg_installed\\x64-windows\\bin"));
    dirs.push_back(exe_dir);

    DWORD last_err = ERROR_MOD_NOT_FOUND;
    for (const std::string& dir : dirs) {
        const std::string dll = joinPath(dir, "cholmod.dll");
        if (!fileExists(dll)) {
            continue;
        }
        SetDllDirectoryA(dir.c_str());
        HMODULE module = LoadLibraryA(dll.c_str());
        if (module == nullptr) {
            last_err = GetLastError();
            continue;
        }
        api.module = module;
        api.start = reinterpret_cast<DynamicCholmodApi::start_t>(loadRequiredProc(module, "cholmod_start"));
        api.finish = reinterpret_cast<DynamicCholmodApi::finish_t>(loadRequiredProc(module, "cholmod_finish"));
        api.allocate_sparse = reinterpret_cast<DynamicCholmodApi::allocate_sparse_t>(loadRequiredProc(module, "cholmod_allocate_sparse"));
        api.free_sparse = reinterpret_cast<DynamicCholmodApi::free_sparse_t>(loadRequiredProc(module, "cholmod_free_sparse"));
        api.sort = reinterpret_cast<DynamicCholmodApi::sort_t>(loadRequiredProc(module, "cholmod_sort"));
        api.allocate_dense = reinterpret_cast<DynamicCholmodApi::allocate_dense_t>(loadRequiredProc(module, "cholmod_allocate_dense"));
        api.free_dense = reinterpret_cast<DynamicCholmodApi::free_dense_t>(loadRequiredProc(module, "cholmod_free_dense"));
        api.analyze = reinterpret_cast<DynamicCholmodApi::analyze_t>(loadRequiredProc(module, "cholmod_analyze"));
        api.factorize = reinterpret_cast<DynamicCholmodApi::factorize_t>(loadRequiredProc(module, "cholmod_factorize"));
        api.free_factor = reinterpret_cast<DynamicCholmodApi::free_factor_t>(loadRequiredProc(module, "cholmod_free_factor"));
        api.solve = reinterpret_cast<DynamicCholmodApi::solve_t>(loadRequiredProc(module, "cholmod_solve"));
        initialized = true;
        return api;
    }

    throw std::runtime_error("failed to load cholmod.dll dynamically: " + lastWin32ErrorText(last_err));
}

std::vector<double> solveDenseLowerWithDynamicCholmod(
    const std::vector<double>& dense_row_major,
    const std::vector<double>& rhs,
    int n,
    double diag_jitter
) {
    if (static_cast<int>(rhs.size()) != n ||
        static_cast<int>(dense_row_major.size()) != n * n) {
        throw std::runtime_error("CHOLMOD coarse solve dimension mismatch");
    }

    DynamicCholmodApi& api = dynamicCholmod();
    cholmod_common c;
    api.start(&c);
    c.supernodal = CHOLMOD_AUTO;

    SuiteSparse_long nnz_lower = 0;
    constexpr double kDropTol = 1e-14;
    for (int col = 0; col < n; ++col) {
        for (int row = col; row < n; ++row) {
            double v = 0.5 * (
                dense_row_major[static_cast<size_t>(row) * n + col] +
                dense_row_major[static_cast<size_t>(col) * n + row]);
            if (row == col) {
                v += diag_jitter;
            }
            if (row == col || std::abs(v) > kDropTol) {
                ++nnz_lower;
            }
        }
    }

    cholmod_sparse* A = api.allocate_sparse(
        static_cast<size_t>(n),
        static_cast<size_t>(n),
        static_cast<size_t>(nnz_lower),
        0,
        1,
        -1,
        CHOLMOD_REAL,
        &c);
    if (A == nullptr) {
        api.finish(&c);
        throw std::runtime_error("CHOLMOD allocate_sparse failed");
    }

    SuiteSparse_long pos = 0;
    if (c.itype == CHOLMOD_LONG) {
        auto* Ap = static_cast<SuiteSparse_long*>(A->p);
        auto* Ai = static_cast<SuiteSparse_long*>(A->i);
        auto* Ax = static_cast<double*>(A->x);
        Ap[0] = 0;
        for (int col = 0; col < n; ++col) {
            for (int row = col; row < n; ++row) {
                double v = 0.5 * (
                    dense_row_major[static_cast<size_t>(row) * n + col] +
                    dense_row_major[static_cast<size_t>(col) * n + row]);
                if (row == col) {
                    v += diag_jitter;
                }
                if (row == col || std::abs(v) > kDropTol) {
                    Ai[pos] = static_cast<SuiteSparse_long>(row);
                    Ax[pos] = v;
                    ++pos;
                }
            }
            Ap[col + 1] = pos;
        }
    } else if (c.itype == CHOLMOD_INT) {
        auto* Ap = static_cast<int*>(A->p);
        auto* Ai = static_cast<int*>(A->i);
        auto* Ax = static_cast<double*>(A->x);
        Ap[0] = 0;
        for (int col = 0; col < n; ++col) {
            for (int row = col; row < n; ++row) {
                double v = 0.5 * (
                    dense_row_major[static_cast<size_t>(row) * n + col] +
                    dense_row_major[static_cast<size_t>(col) * n + row]);
                if (row == col) {
                    v += diag_jitter;
                }
                if (row == col || std::abs(v) > kDropTol) {
                    Ai[static_cast<int>(pos)] = row;
                    Ax[static_cast<int>(pos)] = v;
                    ++pos;
                }
            }
            Ap[col + 1] = static_cast<int>(pos);
        }
    } else {
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD unsupported index type");
    }
    api.sort(A, &c);

    cholmod_dense* b = api.allocate_dense(
        static_cast<size_t>(n),
        1,
        static_cast<size_t>(n),
        CHOLMOD_REAL,
        &c);
    if (b == nullptr) {
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD allocate_dense failed");
    }
    std::memcpy(b->x, rhs.data(), static_cast<size_t>(n) * sizeof(double));

    cholmod_factor* L = api.analyze(A, &c);
    const int ok = api.factorize(A, L, &c);
    if (!ok || c.status != CHOLMOD_OK) {
        api.free_factor(&L, &c);
        api.free_dense(&b, &c);
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD coarse factorization failed");
    }

    cholmod_dense* x = api.solve(CHOLMOD_A, L, b, &c);
    if (x == nullptr || c.status != CHOLMOD_OK) {
        if (x != nullptr) {
            api.free_dense(&x, &c);
        }
        api.free_factor(&L, &c);
        api.free_dense(&b, &c);
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD coarse solve failed");
    }

    std::vector<double> out(static_cast<size_t>(n));
    std::memcpy(out.data(), x->x, static_cast<size_t>(n) * sizeof(double));

    api.free_dense(&x, &c);
    api.free_factor(&L, &c);
    api.free_dense(&b, &c);
    api.free_sparse(&A, &c);
    api.finish(&c);
    return out;
}

std::vector<double> solveBlockSparse3WithDynamicCholmod(
    const CoarseBlockPatternHost& pattern,
    const std::vector<double>& block_values,
    const std::vector<double>& rhs,
    double diag_jitter
) {
    const int n = pattern.coarse_dim;
    const int bd = pattern.block_dim;
    if (static_cast<int>(rhs.size()) != n ||
        static_cast<int>(block_values.size()) != pattern.block_count * bd * bd) {
        throw std::runtime_error("CHOLMOD block-sparse coarse solve dimension mismatch");
    }

    std::unordered_map<unsigned long long, double> lower_values;
    lower_values.reserve(static_cast<size_t>(pattern.block_count) * 9 + static_cast<size_t>(n));
    auto keyFor = [](int col, int row) -> unsigned long long {
        return (static_cast<unsigned long long>(static_cast<unsigned int>(col)) << 32) |
               static_cast<unsigned int>(row);
    };
    auto addLower = [&](int row, int col, double value) {
        if (std::abs(value) <= 1e-14) {
            return;
        }
        if (row == col) {
            lower_values[keyFor(col, row)] += value;
        } else if (row > col) {
            lower_values[keyFor(col, row)] += 0.5 * value;
        } else {
            lower_values[keyFor(row, col)] += 0.5 * value;
        }
    };

    for (int slot = 0; slot < pattern.block_count; ++slot) {
        const int gr = pattern.block_rows[static_cast<size_t>(slot)];
        const int gc = pattern.block_cols[static_cast<size_t>(slot)];
        const double* block = block_values.data() + static_cast<size_t>(slot) * bd * bd;
        for (int br = 0; br < bd; ++br) {
            for (int bc = 0; bc < bd; ++bc) {
                addLower(bd * gr + br, bd * gc + bc, block[bd * br + bc]);
            }
        }
    }
    for (int d = 0; d < n; ++d) {
        lower_values[keyFor(d, d)] += diag_jitter;
    }

    struct Entry {
        int col = 0;
        int row = 0;
        double value = 0.0;
    };
    std::vector<Entry> entries;
    entries.reserve(lower_values.size());
    for (const auto& kv : lower_values) {
        const int col = static_cast<int>(kv.first >> 32);
        const int row = static_cast<int>(kv.first & 0xffffffffu);
        if (row < col) {
            continue;
        }
        if (row == col || std::abs(kv.second) > 1e-14) {
            entries.push_back(Entry{col, row, kv.second});
        }
    }
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
        return (a.col < b.col) || (a.col == b.col && a.row < b.row);
    });

    DynamicCholmodApi& api = dynamicCholmod();
    cholmod_common c;
    api.start(&c);
    c.supernodal = CHOLMOD_AUTO;

    cholmod_sparse* A = api.allocate_sparse(
        static_cast<size_t>(n),
        static_cast<size_t>(n),
        entries.size(),
        0,
        1,
        -1,
        CHOLMOD_REAL,
        &c);
    if (A == nullptr) {
        api.finish(&c);
        throw std::runtime_error("CHOLMOD allocate_sparse failed for block coarse");
    }

    if (c.itype == CHOLMOD_LONG) {
        auto* Ap = static_cast<SuiteSparse_long*>(A->p);
        auto* Ai = static_cast<SuiteSparse_long*>(A->i);
        auto* Ax = static_cast<double*>(A->x);
        size_t pos = 0;
        Ap[0] = 0;
        for (int col = 0; col < n; ++col) {
            while (pos < entries.size() && entries[pos].col == col) {
                Ai[pos] = static_cast<SuiteSparse_long>(entries[pos].row);
                Ax[pos] = entries[pos].value;
                ++pos;
            }
            Ap[col + 1] = static_cast<SuiteSparse_long>(pos);
        }
    } else if (c.itype == CHOLMOD_INT) {
        auto* Ap = static_cast<int*>(A->p);
        auto* Ai = static_cast<int*>(A->i);
        auto* Ax = static_cast<double*>(A->x);
        size_t pos = 0;
        Ap[0] = 0;
        for (int col = 0; col < n; ++col) {
            while (pos < entries.size() && entries[pos].col == col) {
                Ai[pos] = entries[pos].row;
                Ax[pos] = entries[pos].value;
                ++pos;
            }
            Ap[col + 1] = static_cast<int>(pos);
        }
    } else {
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD unsupported index type for block coarse");
    }
    api.sort(A, &c);

    cholmod_dense* b = api.allocate_dense(
        static_cast<size_t>(n),
        1,
        static_cast<size_t>(n),
        CHOLMOD_REAL,
        &c);
    if (b == nullptr) {
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD allocate_dense failed for block coarse");
    }
    std::memcpy(b->x, rhs.data(), static_cast<size_t>(n) * sizeof(double));

    cholmod_factor* L = api.analyze(A, &c);
    const int ok = api.factorize(A, L, &c);
    if (!ok || c.status != CHOLMOD_OK) {
        api.free_factor(&L, &c);
        api.free_dense(&b, &c);
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD block coarse factorization failed");
    }

    cholmod_dense* x = api.solve(CHOLMOD_A, L, b, &c);
    if (x == nullptr || c.status != CHOLMOD_OK) {
        if (x != nullptr) {
            api.free_dense(&x, &c);
        }
        api.free_factor(&L, &c);
        api.free_dense(&b, &c);
        api.free_sparse(&A, &c);
        api.finish(&c);
        throw std::runtime_error("CHOLMOD block coarse solve failed");
    }

    std::vector<double> out(static_cast<size_t>(n));
    std::memcpy(out.data(), x->x, static_cast<size_t>(n) * sizeof(double));

    api.free_dense(&x, &c);
    api.free_factor(&L, &c);
    api.free_dense(&b, &c);
    api.free_sparse(&A, &c);
    api.finish(&c);
    return out;
}

class DynamicCholmodReusableFactor {
public:
    DynamicCholmodReusableFactor(
        const CoarseBlockPatternHost& pattern,
        double diag_jitter
    ) : api_(&dynamicCholmod()), n_(pattern.coarse_dim) {
        const int bd = pattern.block_dim;
        block_value_count_ = pattern.block_count * bd * bd;
        api_->start(&common_);
        started_ = true;
        common_.supernodal = CHOLMOD_AUTO;

        std::unordered_map<unsigned long long, int> lower_pos_by_key;
        lower_pos_by_key.reserve(static_cast<size_t>(pattern.block_count) * bd * bd + static_cast<size_t>(n_));
        auto keyFor = [](int col, int row) -> unsigned long long {
            return (static_cast<unsigned long long>(static_cast<unsigned int>(col)) << 32) |
                   static_cast<unsigned int>(row);
        };
        auto addKey = [&](int col, int row) {
            const unsigned long long key = keyFor(col, row);
            if (lower_pos_by_key.find(key) == lower_pos_by_key.end()) {
                const int pos = static_cast<int>(lower_pos_by_key.size());
                lower_pos_by_key.emplace(key, pos);
            }
        };
        auto addLowerPattern = [&](int row, int col) {
            if (row == col) {
                addKey(col, row);
            } else if (row > col) {
                addKey(col, row);
            } else {
                addKey(row, col);
            }
        };
        for (int slot = 0; slot < pattern.block_count; ++slot) {
            const int gr = pattern.block_rows[static_cast<size_t>(slot)];
            const int gc = pattern.block_cols[static_cast<size_t>(slot)];
            for (int br = 0; br < bd; ++br) {
                for (int bc = 0; bc < bd; ++bc) {
                    addLowerPattern(bd * gr + br, bd * gc + bc);
                }
            }
        }
        for (int d = 0; d < n_; ++d) {
            addKey(d, d);
        }

        struct Entry {
            int col = 0;
            int row = 0;
            unsigned long long key = 0;
        };
        std::vector<Entry> entries;
        entries.reserve(lower_pos_by_key.size());
        for (const auto& kv : lower_pos_by_key) {
            const int col = static_cast<int>(kv.first >> 32);
            const int row = static_cast<int>(kv.first & 0xffffffffu);
            if (row >= col) {
                entries.push_back(Entry{col, row, kv.first});
            }
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) {
            return (a.col < b.col) || (a.col == b.col && a.row < b.row);
        });
        lower_pos_by_key.clear();
        lower_pos_by_key.reserve(entries.size());
        for (int pos = 0; pos < static_cast<int>(entries.size()); ++pos) {
            lower_pos_by_key.emplace(entries[static_cast<size_t>(pos)].key, pos);
        }

        A_ = api_->allocate_sparse(
            static_cast<size_t>(n_),
            static_cast<size_t>(n_),
            entries.size(),
            0,
            1,
            -1,
            CHOLMOD_REAL,
            &common_);
        if (A_ == nullptr) {
            throw std::runtime_error("CHOLMOD reusable allocate_sparse failed");
        }
        if (common_.itype == CHOLMOD_LONG) {
            auto* Ap = static_cast<SuiteSparse_long*>(A_->p);
            auto* Ai = static_cast<SuiteSparse_long*>(A_->i);
            auto* Ax = static_cast<double*>(A_->x);
            size_t pos = 0;
            Ap[0] = 0;
            for (int col = 0; col < n_; ++col) {
                while (pos < entries.size() && entries[pos].col == col) {
                    Ai[pos] = static_cast<SuiteSparse_long>(entries[pos].row);
                    Ax[pos] = 0.0;
                    ++pos;
                }
                Ap[col + 1] = static_cast<SuiteSparse_long>(pos);
            }
        } else if (common_.itype == CHOLMOD_INT) {
            auto* Ap = static_cast<int*>(A_->p);
            auto* Ai = static_cast<int*>(A_->i);
            auto* Ax = static_cast<double*>(A_->x);
            size_t pos = 0;
            Ap[0] = 0;
            for (int col = 0; col < n_; ++col) {
                while (pos < entries.size() && entries[pos].col == col) {
                    Ai[pos] = entries[pos].row;
                    Ax[pos] = 0.0;
                    ++pos;
                }
                Ap[col + 1] = static_cast<int>(pos);
            }
        } else {
            api_->free_sparse(&A_, &common_);
            throw std::runtime_error("CHOLMOD reusable unsupported index type");
        }
        api_->sort(A_, &common_);

        contributions_.reserve(static_cast<size_t>(pattern.block_count) * bd * bd);
        auto addContribution = [&](int slot, int br, int bc, int row, int col) {
            const bool same = (row == col);
            const int lower_col = same ? col : std::min(row, col);
            const int lower_row = same ? row : std::max(row, col);
            const unsigned long long key = keyFor(lower_col, lower_row);
            auto it = lower_pos_by_key.find(key);
            if (it == lower_pos_by_key.end()) {
                throw std::runtime_error("CHOLMOD reusable contribution mapping failed");
            }
            contributions_.push_back(Contribution{
                it->second,
                slot * bd * bd + br * bd + bc,
                same ? 1.0 : 0.5
            });
        };
        for (int slot = 0; slot < pattern.block_count; ++slot) {
            const int gr = pattern.block_rows[static_cast<size_t>(slot)];
            const int gc = pattern.block_cols[static_cast<size_t>(slot)];
            for (int br = 0; br < bd; ++br) {
                for (int bc = 0; bc < bd; ++bc) {
                    addContribution(slot, br, bc, bd * gr + br, bd * gc + bc);
                }
            }
        }
        diag_positions_.resize(static_cast<size_t>(n_));
        for (int d = 0; d < n_; ++d) {
            auto it = lower_pos_by_key.find(keyFor(d, d));
            if (it == lower_pos_by_key.end()) {
                throw std::runtime_error("CHOLMOD reusable diagonal mapping failed");
            }
            diag_positions_[static_cast<size_t>(d)] = it->second;
        }
        diag_jitter_ = diag_jitter;
        factor_ = api_->analyze(A_, &common_);
        if (factor_ == nullptr || common_.status != CHOLMOD_OK) {
            throw std::runtime_error("CHOLMOD reusable analyze failed");
        }
        rhs_ = api_->allocate_dense(
            static_cast<size_t>(n_),
            1,
            static_cast<size_t>(n_),
            CHOLMOD_REAL,
            &common_);
        if (rhs_ == nullptr) {
            throw std::runtime_error("CHOLMOD reusable allocate_dense failed");
        }
    }

    DynamicCholmodReusableFactor(const DynamicCholmodReusableFactor&) = delete;
    DynamicCholmodReusableFactor& operator=(const DynamicCholmodReusableFactor&) = delete;

    ~DynamicCholmodReusableFactor() {
        if (rhs_ != nullptr) {
            api_->free_dense(&rhs_, &common_);
        }
        if (factor_ != nullptr) {
            api_->free_factor(&factor_, &common_);
        }
        if (A_ != nullptr) {
            api_->free_sparse(&A_, &common_);
        }
        if (started_) {
            api_->finish(&common_);
        }
    }

    void factorize(const std::vector<double>& block_values) {
        if (static_cast<int>(block_values.size()) != block_value_count_) {
            throw std::runtime_error("CHOLMOD reusable factorize dimension mismatch");
        }
        auto* Ax = static_cast<double*>(A_->x);
        std::fill(Ax, Ax + A_->nzmax, 0.0);
        for (const Contribution& c : contributions_) {
            Ax[c.ax_pos] += c.scale * block_values[static_cast<size_t>(c.value_pos)];
        }
        for (int pos : diag_positions_) {
            Ax[pos] += diag_jitter_;
        }
        common_.status = CHOLMOD_OK;
        const int ok = api_->factorize(A_, factor_, &common_);
        if (!ok || common_.status != CHOLMOD_OK) {
            throw std::runtime_error("CHOLMOD reusable numeric factorization failed");
        }
    }

    std::vector<double> solve(const std::vector<double>& rhs) {
        if (static_cast<int>(rhs.size()) != n_) {
            throw std::runtime_error("CHOLMOD reusable solve dimension mismatch");
        }
        std::memcpy(rhs_->x, rhs.data(), static_cast<size_t>(n_) * sizeof(double));
        common_.status = CHOLMOD_OK;
        cholmod_dense* x = api_->solve(CHOLMOD_A, factor_, rhs_, &common_);
        if (x == nullptr || common_.status != CHOLMOD_OK) {
            if (x != nullptr) {
                api_->free_dense(&x, &common_);
            }
            throw std::runtime_error("CHOLMOD reusable solve failed");
        }
        std::vector<double> out(static_cast<size_t>(n_));
        std::memcpy(out.data(), x->x, static_cast<size_t>(n_) * sizeof(double));
        api_->free_dense(&x, &common_);
        return out;
    }

private:
    struct Contribution {
        int ax_pos = 0;
        int value_pos = 0;
        double scale = 1.0;
    };
    DynamicCholmodApi* api_ = nullptr;
    cholmod_common common_{};
    cholmod_sparse* A_ = nullptr;
    cholmod_factor* factor_ = nullptr;
    cholmod_dense* rhs_ = nullptr;
    std::vector<Contribution> contributions_;
    std::vector<int> diag_positions_;
    int n_ = 0;
    int block_value_count_ = 0;
    double diag_jitter_ = 0.0;
    bool started_ = false;
};

std::vector<double> solveDenseLowerHostCholesky(
    const std::vector<double>& dense_row_major,
    const std::vector<double>& rhs,
    int n,
    double diag_jitter
) {
    if (static_cast<int>(rhs.size()) != n ||
        static_cast<int>(dense_row_major.size()) != n * n) {
        throw std::runtime_error("coarse dense Cholesky dimension mismatch");
    }
    std::vector<double> L(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.5 * (
                dense_row_major[static_cast<size_t>(i) * n + j] +
                dense_row_major[static_cast<size_t>(j) * n + i]);
            if (i == j) {
                sum += diag_jitter;
            }
            for (int k = 0; k < j; ++k) {
                sum -= L[static_cast<size_t>(i) * n + k] *
                       L[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (sum <= 0.0 || !std::isfinite(sum)) {
                    throw std::runtime_error("coarse dense Cholesky factorization failed");
                }
                L[static_cast<size_t>(i) * n + j] = std::sqrt(sum);
            } else {
                L[static_cast<size_t>(i) * n + j] =
                    sum / L[static_cast<size_t>(j) * n + j];
            }
        }
    }

    std::vector<double> y(static_cast<size_t>(n), 0.0);
    for (int i = 0; i < n; ++i) {
        double sum = rhs[static_cast<size_t>(i)];
        for (int k = 0; k < i; ++k) {
            sum -= L[static_cast<size_t>(i) * n + k] * y[static_cast<size_t>(k)];
        }
        y[static_cast<size_t>(i)] = sum / L[static_cast<size_t>(i) * n + i];
    }

    std::vector<double> x(static_cast<size_t>(n), 0.0);
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[static_cast<size_t>(i)];
        for (int k = i + 1; k < n; ++k) {
            sum -= L[static_cast<size_t>(k) * n + i] * x[static_cast<size_t>(k)];
        }
        x[static_cast<size_t>(i)] = sum / L[static_cast<size_t>(i) * n + i];
    }
    return x;
}

struct HybridPrototypeResult {
    int groups = 0;
    int coarse_dim = 0;
    int coarse_blocks = 0;
    bool sparse_coarse = false;
    double basis_build_ms = 0.0;
    double basis_upload_ms = 0.0;
    double basis_host_ms = 0.0;
    double gpu_svd_assemble_ms = 0.0;
    double gpu_svd_eig_ms = 0.0;
    double gpu_svd_extract_ms = 0.0;
    double pattern_build_ms = 0.0;
    double pattern_upload_ms = 0.0;
    double alloc_ms = 0.0;
    double reset_ms = 0.0;
    double cholmod_load_ms = 0.0;
    double core_ms = 0.0;
    double lower_ms = 0.0;
    double gpu_coarse_ms = 0.0;
    double gpu_coarse_lambda_ms = 0.0;
    double gpu_coarse_rhs_ms = 0.0;
    double d2h_ms = 0.0;
    double d2h_lambda_ms = 0.0;
    double d2h_rhs_ms = 0.0;
    double coarse_factor_ms = 0.0;
    double coarse_solve_ms = 0.0;
    double h2d_ms = 0.0;
    double apply_ms = 0.0;
    double fine_download_ms = 0.0;
    double cleanup_ms = 0.0;
    double total_ms = 0.0;
    double fine_norm = 0.0;
    double coarse_rhs_norm = 0.0;
    double coarse_delta_norm = 0.0;
};

struct HybridGpuWorkspace {
    int n = 0;
    int coarse_dim = 0;
    int block_count = 0;
    int block_dim = 0;
    bool sparse = true;
    double* d_fine_x = nullptr;
    double* d_coarse_A = nullptr;
    double* d_coarse_blocks = nullptr;
    double* d_coarse_b = nullptr;
    double* d_coarse_delta = nullptr;
    cudaEvent_t ev0 = nullptr;
    cudaEvent_t ev1 = nullptr;

    void init(int n_in, int coarse_dim_in, int block_count_in, int block_dim_in, bool sparse_in) {
        if (d_fine_x != nullptr &&
            n == n_in &&
            coarse_dim == coarse_dim_in &&
            block_count == block_count_in &&
            block_dim == block_dim_in &&
            sparse == sparse_in) {
            return;
        }
        destroy();
        n = n_in;
        coarse_dim = coarse_dim_in;
        block_count = block_count_in;
        block_dim = block_dim_in;
        sparse = sparse_in;
        const size_t fine_bytes = static_cast<size_t>(n) * 3 * sizeof(double);
        const size_t coarse_mat_bytes = static_cast<size_t>(coarse_dim) * coarse_dim * sizeof(double);
        const size_t coarse_vec_bytes = static_cast<size_t>(coarse_dim) * sizeof(double);
        const size_t coarse_block_bytes =
            static_cast<size_t>(block_count) * static_cast<size_t>(block_dim) *
            static_cast<size_t>(block_dim) * sizeof(double);
        cudaCheck(cudaMalloc(&d_fine_x, fine_bytes), "malloc hybrid workspace fine_x");
        if (sparse) {
            cudaCheck(cudaMalloc(&d_coarse_blocks, coarse_block_bytes), "malloc hybrid workspace coarse blocks");
        } else {
            cudaCheck(cudaMalloc(&d_coarse_A, coarse_mat_bytes), "malloc hybrid workspace coarse_A");
        }
        cudaCheck(cudaMalloc(&d_coarse_b, coarse_vec_bytes), "malloc hybrid workspace coarse_b");
        cudaCheck(cudaMalloc(&d_coarse_delta, coarse_vec_bytes), "malloc hybrid workspace coarse_delta");
        cudaCheck(cudaEventCreate(&ev0), "create hybrid workspace event 0");
        cudaCheck(cudaEventCreate(&ev1), "create hybrid workspace event 1");
    }

    template <typename Fn>
    double eventMs(Fn&& fn) {
        cudaCheck(cudaEventRecord(ev0), "record hybrid workspace start");
        fn();
        cudaCheck(cudaEventRecord(ev1), "record hybrid workspace stop");
        cudaCheck(cudaEventSynchronize(ev1), "sync hybrid workspace stop");
        float ms = 0.0f;
        cudaCheck(cudaEventElapsedTime(&ms, ev0, ev1), "elapsed hybrid workspace");
        return static_cast<double>(ms);
    }

    void destroy() {
        if (ev0 != nullptr) {
            cudaEventDestroy(ev0);
            ev0 = nullptr;
        }
        if (ev1 != nullptr) {
            cudaEventDestroy(ev1);
            ev1 = nullptr;
        }
        cudaFree(d_fine_x);
        cudaFree(d_coarse_A);
        cudaFree(d_coarse_blocks);
        cudaFree(d_coarse_b);
        cudaFree(d_coarse_delta);
        d_fine_x = nullptr;
        d_coarse_A = nullptr;
        d_coarse_blocks = nullptr;
        d_coarse_b = nullptr;
        d_coarse_delta = nullptr;
        n = 0;
        coarse_dim = 0;
        block_count = 0;
        block_dim = 0;
        sparse = true;
    }

    ~HybridGpuWorkspace() {
        destroy();
    }
};

double vectorNormHost(const std::vector<double>& v) {
    long double sum = 0.0L;
    for (double x : v) {
        sum += static_cast<long double>(x) * static_cast<long double>(x);
    }
    return std::sqrt(static_cast<double>(sum));
}

HybridPrototypeResult runHybridHgbpPrototype(
    const GbpGraph& graph,
    const GbpDevice& dev,
    const Args& args,
    int schur_mode,
    std::vector<double>* out_fine = nullptr,
    const SvdBasisHost* cached_svd_basis = nullptr,
    const SvdBasisDevice* cached_d_svd_basis = nullptr,
    const CoarseBlockPatternHost* cached_block_pattern = nullptr,
    const CoarseBlockPatternDevice* cached_d_block_pattern = nullptr,
    double* out_device_fine = nullptr,
    DynamicCholmodReusableFactor* cached_sparse_factor = nullptr,
    HybridGpuWorkspace* cached_runtime_workspace = nullptr,
    TrajectoryDumpWriter* trajectory_dump = nullptr,
    int trajectory_outer = -1,
    const std::vector<double>* cached_basis_values = nullptr
) {
    if ((args.sweeps & 1) != 0) {
        throw std::runtime_error("--hybrid-hgbp currently requires even --sweeps so GBP state returns to buffer A");
    }
    const auto total_t0 = std::chrono::steady_clock::now();
    const int n = dev.n;
    const int m = dev.m;
    const GbpCoopLaunchConfig gbp_launch =
        computeGbpCoopLaunchConfig(n, m, args.persistent_threads, args.coop_block_policy);
    HybridPrototypeResult result;
    SvdBasisHost svd_basis;
    SvdBasisDevice d_svd_basis;
    std::vector<double> basis_selected_values;
    int groups = (n + args.group_size - 1) / args.group_size;
    int coarse_dim = 3 * groups;
    CoarseBlockPatternHost block_pattern;
    CoarseBlockPatternDevice d_block_pattern;
    bool owns_svd_basis = false;
    bool owns_block_pattern = false;
    if (args.hybrid_sparse_coarse) {
        if (args.hybrid_svd_basis) {
            const bool have_basis_cache =
                cached_svd_basis != nullptr &&
                cached_d_svd_basis != nullptr;
            if (have_basis_cache) {
                svd_basis = *cached_svd_basis;
                d_svd_basis = *cached_d_svd_basis;
            } else {
                if (args.hybrid_rigid_basis) {
                    throw std::runtime_error("--hybrid-rigid-basis requires the nonlinear outer cache path");
                }
                if (args.hybrid_gpu_svd_basis) {
                    const auto basis_host_t0 = std::chrono::steady_clock::now();
                    GpuSvdBasisBuildResult gpu_basis =
                        buildGpuSvdBasisFromDevice(
                            graph,
                            dev,
                            args.group_size,
                            args.r_reduced,
                            args.gpu_svd_tol,
                            args.gpu_svd_max_sweeps,
                            trajectory_dump != nullptr && trajectory_dump->enabled());
                    const auto basis_host_t1 = std::chrono::steady_clock::now();
                    result.basis_host_ms +=
                        std::chrono::duration<double, std::milli>(basis_host_t1 - basis_host_t0).count();
                    result.basis_build_ms +=
                        gpu_basis.assemble_ms + gpu_basis.eig_ms + gpu_basis.extract_ms;
                    result.basis_upload_ms += gpu_basis.upload_ms;
                    result.gpu_svd_assemble_ms += gpu_basis.assemble_ms;
                    result.gpu_svd_eig_ms += gpu_basis.eig_ms;
                    result.gpu_svd_extract_ms += gpu_basis.extract_ms;
                    svd_basis = std::move(gpu_basis.host);
                    d_svd_basis = gpu_basis.device;
                    basis_selected_values = std::move(gpu_basis.selected_values);
                    gpu_basis.device = {};
                    owns_svd_basis = true;
                } else {
                    const auto basis_build_t0 = std::chrono::steady_clock::now();
                    svd_basis = buildHostSvdBasisFirstPass(graph, args.group_size, args.r_reduced);
                    const auto basis_build_t1 = std::chrono::steady_clock::now();
                    result.basis_build_ms +=
                        std::chrono::duration<double, std::milli>(basis_build_t1 - basis_build_t0).count();
                    result.basis_host_ms += result.basis_build_ms;

                    const auto basis_upload_t0 = std::chrono::steady_clock::now();
                    d_svd_basis = uploadSvdBasisDevice(svd_basis);
                    cudaCheck(cudaDeviceSynchronize(), "sync svd basis upload");
                    const auto basis_upload_t1 = std::chrono::steady_clock::now();
                    result.basis_upload_ms +=
                        std::chrono::duration<double, std::milli>(basis_upload_t1 - basis_upload_t0).count();
                    owns_svd_basis = true;
                }
            }
            const bool have_pattern_cache =
                cached_block_pattern != nullptr &&
                cached_d_block_pattern != nullptr;
            if (have_pattern_cache) {
                block_pattern = *cached_block_pattern;
                d_block_pattern = *cached_d_block_pattern;
            } else {
                const auto pattern_build_t0 = std::chrono::steady_clock::now();
                block_pattern = buildSvdCoarseBlockPattern(graph, svd_basis);
                const auto pattern_build_t1 = std::chrono::steady_clock::now();
                result.pattern_build_ms +=
                    std::chrono::duration<double, std::milli>(pattern_build_t1 - pattern_build_t0).count();
            }
            groups = svd_basis.groups;
            coarse_dim = svd_basis.coarse_dim;
        } else {
            const auto pattern_build_t0 = std::chrono::steady_clock::now();
            block_pattern = buildIdentityCoarseBlockPattern(graph, args.group_size);
            const auto pattern_build_t1 = std::chrono::steady_clock::now();
            result.pattern_build_ms +=
                std::chrono::duration<double, std::milli>(pattern_build_t1 - pattern_build_t0).count();
            groups = block_pattern.groups;
            coarse_dim = block_pattern.coarse_dim;
        }
        if (d_block_pattern.block_count == 0) {
            const auto pattern_upload_t0 = std::chrono::steady_clock::now();
            d_block_pattern = uploadCoarseBlockPatternDevice(block_pattern);
            cudaCheck(cudaDeviceSynchronize(), "sync coarse block pattern upload");
            const auto pattern_upload_t1 = std::chrono::steady_clock::now();
            result.pattern_upload_ms +=
                std::chrono::duration<double, std::milli>(pattern_upload_t1 - pattern_upload_t0).count();
            owns_block_pattern = true;
        }
    }
    std::vector<double> trajectory_basis;
    if (trajectory_dump != nullptr && trajectory_dump->enabled() && args.hybrid_svd_basis) {
        trajectory_basis = downloadDeviceDoubles(
            d_svd_basis.var_basis,
            static_cast<size_t>(n) * 3 * svd_basis.r,
            "dump svd basis");
        trajectory_dump->writeDoubles(TrajectoryTag::OuterBasis, trajectory_outer, -1, trajectory_basis);
        if (cached_basis_values != nullptr && !cached_basis_values->empty()) {
            trajectory_dump->writeDoubles(
                TrajectoryTag::OuterBasisValues,
                trajectory_outer,
                -1,
                *cached_basis_values);
        } else if (!basis_selected_values.empty()) {
            trajectory_dump->writeDoubles(
                TrajectoryTag::OuterBasisValues,
                trajectory_outer,
                -1,
                basis_selected_values);
        }
    }
    const size_t fine_bytes = static_cast<size_t>(n) * 3 * sizeof(double);
    const size_t coarse_mat_bytes = static_cast<size_t>(coarse_dim) * coarse_dim * sizeof(double);
    const size_t coarse_vec_bytes = static_cast<size_t>(coarse_dim) * sizeof(double);
    const size_t coarse_block_bytes =
        static_cast<size_t>(block_pattern.block_count) *
        static_cast<size_t>(block_pattern.block_dim) *
        static_cast<size_t>(block_pattern.block_dim) *
        sizeof(double);
    HybridGpuWorkspace local_runtime_workspace;
    HybridGpuWorkspace* runtime_workspace =
        cached_runtime_workspace != nullptr ? cached_runtime_workspace : &local_runtime_workspace;
    const auto alloc_t0 = std::chrono::steady_clock::now();
    runtime_workspace->init(
        n,
        coarse_dim,
        block_pattern.block_count,
        block_pattern.block_dim,
        args.hybrid_sparse_coarse);
    cudaCheck(cudaDeviceSynchronize(), "sync hybrid allocation");
    const auto alloc_t1 = std::chrono::steady_clock::now();
    result.alloc_ms += std::chrono::duration<double, std::milli>(alloc_t1 - alloc_t0).count();
    double* d_fine_x = runtime_workspace->d_fine_x;
    double* d_coarse_A = runtime_workspace->d_coarse_A;
    double* d_coarse_b = runtime_workspace->d_coarse_b;
    double* d_coarse_delta = runtime_workspace->d_coarse_delta;
    double* d_coarse_blocks = runtime_workspace->d_coarse_blocks;
    auto eventMs = [&](auto fn) {
        return runtime_workspace->eventMs(fn);
    };

    result.groups = groups;
    result.coarse_dim = coarse_dim;
    result.coarse_blocks = args.hybrid_sparse_coarse ? block_pattern.block_count : groups * groups;
    result.sparse_coarse = args.hybrid_sparse_coarse;
    const auto reset_t0 = std::chrono::steady_clock::now();
    resetGbpDevice(dev);
    cudaCheck(cudaDeviceSynchronize(), "sync hybrid reset");
    const auto reset_t1 = std::chrono::steady_clock::now();
    result.reset_ms += std::chrono::duration<double, std::milli>(reset_t1 - reset_t0).count();

    const auto cholmod_load_t0 = std::chrono::steady_clock::now();
    dynamicCholmod();
    const auto cholmod_load_t1 = std::chrono::steady_clock::now();
    result.cholmod_load_ms +=
        std::chrono::duration<double, std::milli>(cholmod_load_t1 - cholmod_load_t0).count();

    constexpr int threads = 256;
    const int node_blocks = (n + threads - 1) / threads;
    const int factor_blocks = (m + threads - 1) / threads;

    const auto core_t0 = std::chrono::steady_clock::now();
    std::unique_ptr<DynamicCholmodReusableFactor> owned_sparse_factor;
    DynamicCholmodReusableFactor* sparse_factor = cached_sparse_factor;
    if (args.hybrid_sparse_coarse) {
        result.gpu_coarse_lambda_ms += eventMs([&]() {
            cudaCheck(cudaMemset(d_coarse_blocks, 0, coarse_block_bytes), "zero hybrid coarse lambda blocks");
            if (args.hybrid_svd_basis) {
                assembleCoarseUnaryLambdaSvdKernel<<<node_blocks, threads>>>(
                    n,
                    svd_basis.r,
                    d_svd_basis.var_group,
                    d_svd_basis.var_basis,
                    dev.unary_lam,
                    d_coarse_blocks);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseUnaryLambdaSvdKernel");
                assembleCoarseBinaryLambdaSvdKernel<<<factor_blocks, threads>>>(
                    m,
                    svd_basis.r,
                    dev.factor_i,
                    dev.factor_j,
                    d_block_pattern.factor_slot_ii,
                    d_block_pattern.factor_slot_ij,
                    d_block_pattern.factor_slot_ji,
                    d_block_pattern.factor_slot_jj,
                    d_svd_basis.var_basis,
                    dev.hii,
                    dev.hij,
                    dev.hji,
                    dev.hjj,
                    d_coarse_blocks);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseBinaryLambdaSvdKernel");
            } else {
                assembleCoarseUnaryLambdaBlockKernel<<<node_blocks, threads>>>(
                    n,
                    args.group_size,
                    dev.unary_lam,
                    d_coarse_blocks);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseUnaryLambdaBlockKernel");
                assembleCoarseBinaryLambdaBlockKernel<<<factor_blocks, threads>>>(
                    m,
                    d_block_pattern.factor_slot_ii,
                    d_block_pattern.factor_slot_ij,
                    d_block_pattern.factor_slot_ji,
                    d_block_pattern.factor_slot_jj,
                    dev.hii,
                    dev.hij,
                    dev.hji,
                    dev.hjj,
                    d_coarse_blocks);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseBinaryLambdaBlockKernel");
            }
        });
        result.gpu_coarse_ms += result.gpu_coarse_lambda_ms;

        std::vector<double> coarse_blocks(
            static_cast<size_t>(block_pattern.block_count) *
            static_cast<size_t>(block_pattern.block_dim) *
            static_cast<size_t>(block_pattern.block_dim));
        const auto d2h_lam_t0 = std::chrono::steady_clock::now();
        cudaCheck(cudaMemcpy(coarse_blocks.data(), d_coarse_blocks, coarse_block_bytes, cudaMemcpyDeviceToHost),
                  "download hybrid coarse lambda block values");
        const auto d2h_lam_t1 = std::chrono::steady_clock::now();
        result.d2h_lambda_ms += std::chrono::duration<double, std::milli>(d2h_lam_t1 - d2h_lam_t0).count();
        result.d2h_ms += result.d2h_lambda_ms;

        const auto factor_t0 = std::chrono::steady_clock::now();
        if (sparse_factor == nullptr) {
            owned_sparse_factor = std::make_unique<DynamicCholmodReusableFactor>(block_pattern, 1e-9);
            sparse_factor = owned_sparse_factor.get();
        }
        sparse_factor->factorize(coarse_blocks);
        const auto factor_t1 = std::chrono::steady_clock::now();
        result.coarse_factor_ms += std::chrono::duration<double, std::milli>(factor_t1 - factor_t0).count();
    }

    for (int cyc = 0; cyc < args.hybrid_cycles; ++cyc) {
        result.lower_ms += eventMs([&]() {
            launchGbpPersistent(
                dev,
                args.sweeps,
                args.omega,
                schur_mode,
                args.fixed_lambda_start,
                gbp_launch.launched_blocks,
                args.persistent_threads);
        });

        if (args.hybrid_sparse_coarse) {
            result.gpu_coarse_rhs_ms += eventMs([&]() {
                computeBeliefMeanKernel<<<node_blocks, threads>>>(n, dev.belief_lam_a, dev.belief_eta_a, d_fine_x);
                cudaCheck(cudaGetLastError(), "launch computeBeliefMeanKernel");
                cudaCheck(cudaMemset(d_coarse_b, 0, coarse_vec_bytes), "zero hybrid coarse_b");
                if (args.hybrid_svd_basis) {
                    assembleCoarseUnaryResidualSvdKernel<<<node_blocks, threads>>>(
                        n,
                        svd_basis.r,
                        d_svd_basis.var_group,
                        d_svd_basis.var_basis,
                        dev.unary_lam,
                        dev.unary_eta,
                        d_fine_x,
                        d_coarse_b);
                    cudaCheck(cudaGetLastError(), "launch assembleCoarseUnaryResidualSvdKernel");
                    assembleCoarseBinaryResidualSvdKernel<<<factor_blocks, threads>>>(
                        m,
                        svd_basis.r,
                        dev.factor_i,
                        dev.factor_j,
                        d_svd_basis.var_group,
                        d_svd_basis.var_basis,
                        dev.hii,
                        dev.hij,
                        dev.hji,
                        dev.hjj,
                        dev.eta_i,
                        dev.eta_j,
                        d_fine_x,
                        d_coarse_b);
                    cudaCheck(cudaGetLastError(), "launch assembleCoarseBinaryResidualSvdKernel");
                } else {
                    assembleCoarseUnaryResidualIdentityKernel<<<node_blocks, threads>>>(
                        n,
                        args.group_size,
                        dev.unary_lam,
                        dev.unary_eta,
                        d_fine_x,
                        d_coarse_b);
                    cudaCheck(cudaGetLastError(), "launch assembleCoarseUnaryResidualIdentityKernel");
                    assembleCoarseBinaryResidualIdentityKernel<<<factor_blocks, threads>>>(
                        m,
                        args.group_size,
                        dev.factor_i,
                        dev.factor_j,
                        dev.hii,
                        dev.hij,
                        dev.hji,
                        dev.hjj,
                        dev.eta_i,
                        dev.eta_j,
                        d_fine_x,
                        d_coarse_b);
                    cudaCheck(cudaGetLastError(), "launch assembleCoarseBinaryResidualIdentityKernel");
                }
            });
            result.gpu_coarse_ms = result.gpu_coarse_lambda_ms + result.gpu_coarse_rhs_ms;

            std::vector<double> coarse_b(static_cast<size_t>(coarse_dim));
            const auto d2h_rhs_t0 = std::chrono::steady_clock::now();
            cudaCheck(cudaMemcpy(coarse_b.data(), d_coarse_b, coarse_vec_bytes, cudaMemcpyDeviceToHost), "download hybrid coarse_b");
            const auto d2h_rhs_t1 = std::chrono::steady_clock::now();
            result.d2h_rhs_ms += std::chrono::duration<double, std::milli>(d2h_rhs_t1 - d2h_rhs_t0).count();
            result.d2h_ms = result.d2h_lambda_ms + result.d2h_rhs_ms;
            result.coarse_rhs_norm = vectorNormHost(coarse_b);

            const auto solve_t0 = std::chrono::steady_clock::now();
            std::vector<double> coarse_delta = sparse_factor->solve(coarse_b);
            const auto solve_t1 = std::chrono::steady_clock::now();
            result.coarse_solve_ms += std::chrono::duration<double, std::milli>(solve_t1 - solve_t0).count();
            result.coarse_delta_norm = vectorNormHost(coarse_delta);
            if (trajectory_dump != nullptr && trajectory_dump->enabled()) {
                trajectory_dump->writeDoubles(
                    TrajectoryTag::CycleCoarseDelta,
                    trajectory_outer,
                    cyc + 1,
                    coarse_delta);
                if (args.hybrid_svd_basis && !trajectory_basis.empty()) {
                    trajectory_dump->writeDoubles(
                        TrajectoryTag::CycleProlongedDelta,
                        trajectory_outer,
                        cyc + 1,
                        prolongCoarseDeltaHost(
                            n,
                            svd_basis.r,
                            args.coarse_scale,
                            svd_basis.var_group,
                            trajectory_basis,
                            coarse_delta));
                }
            }

            const auto h2d_t0 = std::chrono::steady_clock::now();
            cudaCheck(cudaMemcpy(d_coarse_delta, coarse_delta.data(), coarse_vec_bytes, cudaMemcpyHostToDevice), "upload hybrid coarse_delta");
            const auto h2d_t1 = std::chrono::steady_clock::now();
            result.h2d_ms += std::chrono::duration<double, std::milli>(h2d_t1 - h2d_t0).count();
        } else {
            result.gpu_coarse_ms += eventMs([&]() {
                computeBeliefMeanKernel<<<node_blocks, threads>>>(n, dev.belief_lam_a, dev.belief_eta_a, d_fine_x);
                cudaCheck(cudaGetLastError(), "launch computeBeliefMeanKernel");
                cudaCheck(cudaMemset(d_coarse_b, 0, coarse_vec_bytes), "zero hybrid coarse_b");
                cudaCheck(cudaMemset(d_coarse_A, 0, coarse_mat_bytes), "zero hybrid coarse_A");
                assembleCoarseUnaryIdentityKernel<<<node_blocks, threads>>>(
                    n,
                    args.group_size,
                    coarse_dim,
                    dev.unary_lam,
                    dev.unary_eta,
                    d_fine_x,
                    d_coarse_A,
                    d_coarse_b);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseUnaryIdentityKernel");
                assembleCoarseBinaryIdentityKernel<<<factor_blocks, threads>>>(
                    m,
                    args.group_size,
                    coarse_dim,
                    dev.factor_i,
                    dev.factor_j,
                    dev.hii,
                    dev.hij,
                    dev.hji,
                    dev.hjj,
                    dev.eta_i,
                    dev.eta_j,
                    d_fine_x,
                    d_coarse_A,
                    d_coarse_b);
                cudaCheck(cudaGetLastError(), "launch assembleCoarseBinaryIdentityKernel");
            });

            std::vector<double> coarse_A(static_cast<size_t>(coarse_dim) * coarse_dim);
            std::vector<double> coarse_b(static_cast<size_t>(coarse_dim));
            const auto d2h_t0 = std::chrono::steady_clock::now();
            cudaCheck(cudaMemcpy(coarse_A.data(), d_coarse_A, coarse_mat_bytes, cudaMemcpyDeviceToHost), "download hybrid coarse_A");
            cudaCheck(cudaMemcpy(coarse_b.data(), d_coarse_b, coarse_vec_bytes, cudaMemcpyDeviceToHost), "download hybrid coarse_b");
            const auto d2h_t1 = std::chrono::steady_clock::now();
            result.d2h_ms += std::chrono::duration<double, std::milli>(d2h_t1 - d2h_t0).count();
            result.coarse_rhs_norm = vectorNormHost(coarse_b);

            const auto chol_t0 = std::chrono::steady_clock::now();
            std::vector<double> coarse_delta = solveDenseLowerWithDynamicCholmod(coarse_A, coarse_b, coarse_dim, 1e-9);
            const auto chol_t1 = std::chrono::steady_clock::now();
            result.coarse_solve_ms += std::chrono::duration<double, std::milli>(chol_t1 - chol_t0).count();
            result.coarse_delta_norm = vectorNormHost(coarse_delta);
            if (trajectory_dump != nullptr && trajectory_dump->enabled()) {
                trajectory_dump->writeDoubles(
                    TrajectoryTag::CycleCoarseDelta,
                    trajectory_outer,
                    cyc + 1,
                    coarse_delta);
            }

            const auto h2d_t0 = std::chrono::steady_clock::now();
            cudaCheck(cudaMemcpy(d_coarse_delta, coarse_delta.data(), coarse_vec_bytes, cudaMemcpyHostToDevice), "upload hybrid coarse_delta");
            const auto h2d_t1 = std::chrono::steady_clock::now();
            result.h2d_ms += std::chrono::duration<double, std::milli>(h2d_t1 - h2d_t0).count();
        }

        result.apply_ms += eventMs([&]() {
            if (args.hybrid_svd_basis) {
                applyCoarseDeltaSvdKernel<<<node_blocks, threads>>>(
                    n,
                    svd_basis.r,
                    args.coarse_scale,
                    d_svd_basis.var_group,
                    d_svd_basis.var_basis,
                    d_coarse_delta,
                    dev.belief_lam_a,
                    dev.belief_eta_a,
                    d_fine_x);
                cudaCheck(cudaGetLastError(), "launch applyCoarseDeltaSvdKernel");
            } else {
                applyCoarseDeltaIdentityKernel<<<node_blocks, threads>>>(
                    n,
                    args.group_size,
                    args.coarse_scale,
                    d_coarse_delta,
                    dev.belief_lam_a,
                    dev.belief_eta_a,
                    d_fine_x);
                cudaCheck(cudaGetLastError(), "launch applyCoarseDeltaIdentityKernel");
            }
        });
        if (trajectory_dump != nullptr && trajectory_dump->enabled()) {
            trajectory_dump->writeDoubles(
                TrajectoryTag::CycleFineState,
                trajectory_outer,
                cyc + 1,
                downloadDeviceDoubles(d_fine_x, static_cast<size_t>(n) * 3, "dump cycle fine state"));
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "hybrid final sync");
    const auto core_t1 = std::chrono::steady_clock::now();
    result.core_ms = std::chrono::duration<double, std::milli>(core_t1 - core_t0).count();

    if (out_device_fine != nullptr) {
        cudaCheck(cudaMemcpy(out_device_fine, d_fine_x, fine_bytes, cudaMemcpyDeviceToDevice),
                  "copy hybrid fine_x to outer device delta");
    }

    if (out_fine != nullptr) {
        std::vector<double> fine(static_cast<size_t>(n) * 3);
        const auto fine_download_t0 = std::chrono::steady_clock::now();
        cudaCheck(cudaMemcpy(fine.data(), d_fine_x, fine_bytes, cudaMemcpyDeviceToHost), "download hybrid fine_x");
        const auto fine_download_t1 = std::chrono::steady_clock::now();
        result.fine_download_ms +=
            std::chrono::duration<double, std::milli>(fine_download_t1 - fine_download_t0).count();
        result.fine_norm = vectorNormHost(fine);
        *out_fine = std::move(fine);
    }

    const auto cleanup_t0 = std::chrono::steady_clock::now();
    if (args.hybrid_sparse_coarse && owns_block_pattern) {
        freeCoarseBlockPatternDevice(d_block_pattern);
    }
    if (args.hybrid_svd_basis && owns_svd_basis) {
        freeSvdBasisDevice(d_svd_basis);
    }
    cudaCheck(cudaDeviceSynchronize(), "sync hybrid cleanup");
    const auto cleanup_t1 = std::chrono::steady_clock::now();
    result.cleanup_ms += std::chrono::duration<double, std::milli>(cleanup_t1 - cleanup_t0).count();
    const auto total_t1 = std::chrono::steady_clock::now();
    result.total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    return result;
}

void launchSweeps(
    const std::vector<BucketDevice>& buckets,
    int sweeps,
    float omega,
    const float* d_diag,
    const float* d_rhs,
    float*& d_x,
    float*& d_x_next
) {
    constexpr int threads = 256;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
        for (const BucketDevice& b : buckets) {
            const int blocks = (b.rows + threads - 1) / threads;
            if (blocks > 0) {
                bucketJacobiKernel<<<blocks, threads>>>(
                    b.rows, b.cap, b.nodes, b.cols, b.blocks,
                    d_diag, d_rhs, d_x, d_x_next, omega);
            }
        }
        std::swap(d_x, d_x_next);
    }
}

template <typename T>
double relativeError(const std::vector<T>& a, const std::vector<T>& b) {
    long double diff2 = 0.0;
    long double ref2 = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const long double d = static_cast<long double>(a[i]) - static_cast<long double>(b[i]);
        diff2 += d * d;
        ref2 += static_cast<long double>(b[i]) * static_cast<long double>(b[i]);
    }
    return std::sqrt(static_cast<double>(diff2 / std::max<long double>(ref2, 1e-30L)));
}

template <typename T>
double norm2(const std::vector<T>& x) {
    long double s = 0.0;
    for (float v : x) {
        s += static_cast<long double>(v) * v;
    }
    return std::sqrt(static_cast<double>(s));
}

Pose2 se2ExpPose(const double* xi) {
    const double vx = xi[0];
    const double vy = xi[1];
    const double w = xi[2];
    if (std::abs(w) < 1e-12) {
        return {vx, vy, 0.0};
    }
    const double a = std::sin(w) / w;
    const double b = (1.0 - std::cos(w)) / w;
    return {
        a * vx - b * vy,
        b * vx + a * vy,
        wrapAngle(w)
    };
}

Pose2 se2PlusPose(const Pose2& base, const double* delta) {
    return se2Compose(base, se2ExpPose(delta));
}

void applyPoseDeltasInPlace(Problem& problem, const std::vector<double>& delta) {
    if (delta.size() != static_cast<size_t>(problem.init_poses.size()) * 3) {
        throw std::runtime_error("pose delta size mismatch");
    }
    for (int i = 0; i < static_cast<int>(problem.init_poses.size()); ++i) {
        problem.init_poses[static_cast<size_t>(i)] =
            se2PlusPose(problem.init_poses[static_cast<size_t>(i)], delta.data() + static_cast<size_t>(i) * 3);
    }
}

double quadratic3(const std::array<double, 9>& info, const std::array<double, 3>& r) {
    double wr[3] = {
        info[0] * r[0] + info[1] * r[1] + info[2] * r[2],
        info[3] * r[0] + info[4] * r[1] + info[5] * r[2],
        info[6] * r[0] + info[7] * r[1] + info[8] * r[2]
    };
    return r[0] * wr[0] + r[1] * wr[1] + r[2] * wr[2];
}

double nonlinearObjectiveRaw(const Problem& problem) {
    double total = 0.0;
    for (const Edge2& edge : problem.edges) {
        const Pose2 pred = se2Between(problem.init_poses[static_cast<size_t>(edge.i)],
                                      problem.init_poses[static_cast<size_t>(edge.j)]);
        const Pose2 z{edge.measurement[0], edge.measurement[1], edge.measurement[2]};
        const std::array<double, 3> r = se2Log(se2Compose(se2Inverse(z), pred));
        total += 0.5 * quadratic3(edge.information, r);
    }
    const std::array<double, 3> ar =
        se2Log(se2Compose(se2Inverse(problem.anchor_pose), problem.init_poses.front()));
    total += 0.5 * quadratic3(problem.anchor_information, ar);
    return total;
}

void runHybridOuterPrototype(Problem problem, const Args& args, int schur_mode) {
    const auto total_t0 = std::chrono::steady_clock::now();
    double final_objective = nonlinearObjectiveRaw(problem);
    std::cout << std::fixed << std::setprecision(6)
              << "hybrid_outer_initial_objective=" << final_objective << "\n";
    double hybrid_ms_sum = 0.0;
    double outer_ms_sum = 0.0;
    double build_graph_ms_sum = 0.0;
    double incoming_ms_sum = 0.0;
    double upload_ms_sum = 0.0;
    double free_dev_ms_sum = 0.0;
    double apply_ms_sum = 0.0;
    double objective_ms_sum = 0.0;
    double cached_basis_build_ms_sum = 0.0;
    double cached_basis_upload_ms_sum = 0.0;
    double cached_pattern_build_ms_sum = 0.0;
    double cached_pattern_upload_ms_sum = 0.0;
    const bool use_outer_basis_cache =
        args.hybrid_rigid_basis ||
        (args.hybrid_svd_basis && !args.hybrid_gpu_svd_basis && args.basis_rebuild_period != 1);
    bool have_cached_basis = false;
    SvdBasisHost cached_svd_basis;
    SvdBasisDevice cached_d_svd_basis;
    CoarseBlockPatternHost cached_block_pattern;
    CoarseBlockPatternDevice cached_d_block_pattern;
    auto clearCachedBasis = [&]() {
        if (have_cached_basis) {
            freeSvdBasisDevice(cached_d_svd_basis);
            freeCoarseBlockPatternDevice(cached_d_block_pattern);
            have_cached_basis = false;
            cached_svd_basis = {};
            cached_block_pattern = {};
        }
    };
    for (int outer = 1; outer <= args.num_outer; ++outer) {
        const auto outer_t0 = std::chrono::steady_clock::now();
        const auto build_t0 = std::chrono::steady_clock::now();
        GbpGraph graph = buildGbpGraph(problem, args.huber_delta);
        const auto build_t1 = std::chrono::steady_clock::now();
        IncomingFlatHost incoming = buildIncomingFlat(graph, args.incoming_layout);
        const auto incoming_t1 = std::chrono::steady_clock::now();
        double cached_basis_build_ms = 0.0;
        double cached_basis_upload_ms = 0.0;
        double cached_pattern_build_ms = 0.0;
        double cached_pattern_upload_ms = 0.0;
        if (use_outer_basis_cache) {
            const bool periodic_rebuild =
                args.basis_rebuild_period > 1 &&
                ((outer - 1) % args.basis_rebuild_period == 0);
            if (!have_cached_basis || periodic_rebuild) {
                clearCachedBasis();
                const auto basis_build_t0 = std::chrono::steady_clock::now();
                cached_svd_basis = args.hybrid_rigid_basis
                    ? buildHostRigidBasisFromPoses(problem, args.group_size, args.r_reduced)
                    : buildHostSvdBasisFirstPass(graph, args.group_size, args.r_reduced);
                const auto basis_build_t1 = std::chrono::steady_clock::now();
                cached_basis_build_ms =
                    std::chrono::duration<double, std::milli>(basis_build_t1 - basis_build_t0).count();

                const auto basis_upload_t0 = std::chrono::steady_clock::now();
                cached_d_svd_basis = uploadSvdBasisDevice(cached_svd_basis);
                cudaCheck(cudaDeviceSynchronize(), "outer cached svd basis upload sync");
                const auto basis_upload_t1 = std::chrono::steady_clock::now();
                cached_basis_upload_ms =
                    std::chrono::duration<double, std::milli>(basis_upload_t1 - basis_upload_t0).count();

                const auto pattern_build_t0 = std::chrono::steady_clock::now();
                cached_block_pattern = buildSvdCoarseBlockPattern(graph, cached_svd_basis);
                const auto pattern_build_t1 = std::chrono::steady_clock::now();
                cached_pattern_build_ms =
                    std::chrono::duration<double, std::milli>(pattern_build_t1 - pattern_build_t0).count();

                const auto pattern_upload_t0 = std::chrono::steady_clock::now();
                cached_d_block_pattern = uploadCoarseBlockPatternDevice(cached_block_pattern);
                cudaCheck(cudaDeviceSynchronize(), "outer cached coarse block pattern upload sync");
                const auto pattern_upload_t1 = std::chrono::steady_clock::now();
                cached_pattern_upload_ms =
                    std::chrono::duration<double, std::milli>(pattern_upload_t1 - pattern_upload_t0).count();
                have_cached_basis = true;
            }
        }
        GbpDevice dev = uploadGbpDevice(graph, incoming);
        cudaCheck(cudaDeviceSynchronize(), "outer upload sync");
        const auto upload_t1 = std::chrono::steady_clock::now();
        std::vector<double> fine_delta;
        const HybridPrototypeResult h = use_outer_basis_cache
            ? runHybridHgbpPrototype(
                graph,
                dev,
                args,
                schur_mode,
                &fine_delta,
                &cached_svd_basis,
                &cached_d_svd_basis,
                &cached_block_pattern,
                &cached_d_block_pattern)
            : runHybridHgbpPrototype(graph, dev, args, schur_mode, &fine_delta);
        const auto free_dev_t0 = std::chrono::steady_clock::now();
        freeGbpDevice(dev);
        cudaCheck(cudaDeviceSynchronize(), "outer free dev sync");
        const auto free_dev_t1 = std::chrono::steady_clock::now();
        const auto apply_t0 = std::chrono::steady_clock::now();
        applyPoseDeltasInPlace(problem, fine_delta);
        const auto apply_t1 = std::chrono::steady_clock::now();
        const auto objective_t0 = std::chrono::steady_clock::now();
        final_objective = nonlinearObjectiveRaw(problem);
        const auto objective_t1 = std::chrono::steady_clock::now();
        const auto outer_t1 = std::chrono::steady_clock::now();
        const double outer_ms = std::chrono::duration<double, std::milli>(outer_t1 - outer_t0).count();
        const double build_graph_ms = std::chrono::duration<double, std::milli>(build_t1 - build_t0).count();
        const double incoming_ms = std::chrono::duration<double, std::milli>(incoming_t1 - build_t1).count();
        const double upload_ms = std::chrono::duration<double, std::milli>(upload_t1 - incoming_t1).count();
        const double free_dev_ms = std::chrono::duration<double, std::milli>(free_dev_t1 - free_dev_t0).count();
        const double apply_pose_ms = std::chrono::duration<double, std::milli>(apply_t1 - apply_t0).count();
        const double objective_ms = std::chrono::duration<double, std::milli>(objective_t1 - objective_t0).count();
        hybrid_ms_sum += h.total_ms;
        outer_ms_sum += outer_ms;
        build_graph_ms_sum += build_graph_ms;
        incoming_ms_sum += incoming_ms;
        upload_ms_sum += upload_ms;
        free_dev_ms_sum += free_dev_ms;
        apply_ms_sum += apply_pose_ms;
        objective_ms_sum += objective_ms;
        cached_basis_build_ms_sum += cached_basis_build_ms;
        cached_basis_upload_ms_sum += cached_basis_upload_ms;
        cached_pattern_build_ms_sum += cached_pattern_build_ms;
        cached_pattern_upload_ms_sum += cached_pattern_upload_ms;
        std::cout << std::fixed << std::setprecision(3)
                  << "hybrid_outer outer=" << outer
                  << " objective=" << std::setprecision(6) << final_objective
                  << std::setprecision(3)
                  << " outer_ms=" << outer_ms
                  << " hybrid_ms=" << h.total_ms
                  << " hybrid_core_ms=" << h.core_ms
                  << " lower_ms=" << h.lower_ms
                  << " gpu_coarse_ms=" << h.gpu_coarse_ms
                  << " cholmod_factor_ms=" << h.coarse_factor_ms
                  << " cholmod_solve_ms=" << h.coarse_solve_ms
                  << " e_norm=" << h.fine_norm
                  << "\n";
        std::cout << std::fixed << std::setprecision(3)
                  << "hybrid_outer_detail outer=" << outer
                  << " build_graph_ms=" << build_graph_ms
                  << " incoming_ms=" << incoming_ms
                  << " upload_ms=" << upload_ms
                  << " basis_build_ms=" << h.basis_build_ms
                  << " basis_upload_ms=" << h.basis_upload_ms
                  << " pattern_build_ms=" << h.pattern_build_ms
                  << " pattern_upload_ms=" << h.pattern_upload_ms
                  << " cached_basis_build_ms=" << cached_basis_build_ms
                  << " cached_basis_upload_ms=" << cached_basis_upload_ms
                  << " cached_pattern_build_ms=" << cached_pattern_build_ms
                  << " cached_pattern_upload_ms=" << cached_pattern_upload_ms
                  << " hybrid_alloc_ms=" << h.alloc_ms
                  << " hybrid_reset_ms=" << h.reset_ms
                  << " cholmod_load_ms=" << h.cholmod_load_ms
                  << " fine_download_ms=" << h.fine_download_ms
                  << " hybrid_cleanup_ms=" << h.cleanup_ms
                  << " free_dev_ms=" << free_dev_ms
                  << " apply_pose_ms=" << apply_pose_ms
                  << " objective_ms=" << objective_ms
                  << "\n";
    }
    const auto total_t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    std::cout << std::fixed << std::setprecision(6)
              << "hybrid_outer_final_objective=" << final_objective << "\n";
    clearCachedBasis();
    std::cout << std::fixed << std::setprecision(3)
              << "hybrid_outer_summary outers=" << args.num_outer
              << " total_ms=" << total_ms
              << " summed_outer_ms=" << outer_ms_sum
              << " summed_hybrid_ms=" << hybrid_ms_sum
              << " summed_build_graph_ms=" << build_graph_ms_sum
              << " summed_incoming_ms=" << incoming_ms_sum
              << " summed_upload_ms=" << upload_ms_sum
              << " summed_free_dev_ms=" << free_dev_ms_sum
              << " summed_apply_pose_ms=" << apply_ms_sum
              << " summed_objective_ms=" << objective_ms_sum
              << " summed_cached_basis_build_ms=" << cached_basis_build_ms_sum
              << " summed_cached_basis_upload_ms=" << cached_basis_upload_ms_sum
              << " summed_cached_pattern_build_ms=" << cached_pattern_build_ms_sum
              << " summed_cached_pattern_upload_ms=" << cached_pattern_upload_ms_sum
              << "\n";
}

void runHybridOuterPrototypeGpuLinearize(Problem problem, const Args& args, int schur_mode) {
    const auto setup_t0 = std::chrono::steady_clock::now();
    GbpGraph topology_graph = buildGbpGraph(problem, args.huber_delta);
    IncomingFlatHost incoming = buildIncomingFlat(topology_graph, args.incoming_layout);
    GbpDevice dev = uploadGbpDevice(topology_graph, incoming);
    ProblemDevice problem_dev = uploadProblemDevice(problem);
    SvdBasisHost cached_pattern_basis;
    CoarseBlockPatternHost cached_block_pattern;
    CoarseBlockPatternDevice cached_d_block_pattern;
    std::unique_ptr<DynamicCholmodReusableFactor> cached_sparse_factor;
    if (args.hybrid_sparse_coarse && args.hybrid_svd_basis) {
        cached_pattern_basis =
            makeContiguousSvdBasisMetadata(topology_graph.n, args.group_size, args.r_reduced, false);
        cached_block_pattern = buildSvdCoarseBlockPattern(topology_graph, cached_pattern_basis);
        cached_d_block_pattern = uploadCoarseBlockPatternDevice(cached_block_pattern);
        cached_sparse_factor =
            std::make_unique<DynamicCholmodReusableFactor>(cached_block_pattern, 1e-9);
    }
    double* d_outer_delta = nullptr;
    cudaCheck(cudaMalloc(&d_outer_delta, static_cast<size_t>(topology_graph.n) * 3 * sizeof(double)),
              "malloc outer device delta");
    cudaCheck(cudaDeviceSynchronize(), "gpu outer setup sync");
    const auto setup_t1 = std::chrono::steady_clock::now();

    const auto total_t0 = std::chrono::steady_clock::now();
    double final_objective = nonlinearObjectiveRaw(problem);
    TrajectoryDumpWriter trajectory_dump(args.trajectory_dump);
    TrajectoryDumpWriter* dump = trajectory_dump.enabled() ? &trajectory_dump : nullptr;
    if (dump != nullptr) {
        std::ostringstream cfg;
        cfg << "{"
            << "\"format_version\":1,"
            << "\"nodes\":" << topology_graph.n << ","
            << "\"edges\":" << topology_graph.m << ","
            << "\"num_slots\":" << dev.num_slots << ","
            << "\"groups\":" << ((topology_graph.n + args.group_size - 1) / args.group_size) << ","
            << "\"outer\":" << args.num_outer << ","
            << "\"cycles\":" << args.hybrid_cycles << ","
            << "\"sweeps\":" << args.sweeps << ","
            << "\"group_size\":" << args.group_size << ","
            << "\"r_reduced\":" << args.r_reduced << ","
            << "\"huber_delta\":" << args.huber_delta << ","
            << "\"fixed_lambda_start\":" << args.fixed_lambda_start << ","
            << "\"incoming_layout\":\"" << args.incoming_layout << "\","
            << "\"schur_kernel\":\"" << args.schur_kernel << "\","
            << "\"coop_block_policy\":\"" << args.coop_block_policy << "\","
            << "\"persistent_threads\":" << args.persistent_threads << ","
            << "\"gpu_svd_warm_rr_after_first\":" << (args.gpu_svd_warm_rr_after_first ? 1 : 0) << ","
            << "\"gpu_svd_partial_from_start\":" << (args.gpu_svd_partial_from_start ? 1 : 0) << ","
            << "\"gpu_svd_persistent_full_basis\":" << (args.gpu_svd_persistent_full_basis ? 1 : 0) << ","
            << "\"gpu_svd_tol\":" << args.gpu_svd_tol << ","
            << "\"gpu_svd_max_sweeps\":" << args.gpu_svd_max_sweeps
            << "}";
        dump->writeText(TrajectoryTag::ConfigText, 0, 0, cfg.str());
        dump->writeDoubles(TrajectoryTag::InitialPoses, 0, 0, makePoseVector(problem));
    }
    std::cout << std::fixed << std::setprecision(6)
              << "gpu_outer_initial_objective=" << final_objective << "\n";
    std::cout << std::fixed << std::setprecision(3)
              << "gpu_outer_setup_ms="
              << std::chrono::duration<double, std::milli>(setup_t1 - setup_t0).count()
              << "\n";

    double outer_ms_sum = 0.0;
    double linearize_ms_sum = 0.0;
    double hybrid_ms_sum = 0.0;
    double apply_gpu_ms_sum = 0.0;
    double download_pose_ms_sum = 0.0;
    double objective_ms_sum = 0.0;
    double basis_external_host_ms_sum = 0.0;
    double basis_host_ms_sum = 0.0;
    double outer_profile_known_ms_sum = 0.0;
    double outer_profile_gap_ms_sum = 0.0;
    double hybrid_component_sum_ms = 0.0;
    double hybrid_component_gap_ms_sum = 0.0;
    double hybrid_core_component_sum_ms = 0.0;
    double hybrid_core_gap_ms_sum = 0.0;
    double hybrid_alloc_ms_sum = 0.0;
    double hybrid_reset_ms_sum = 0.0;
    double hybrid_cholmod_load_ms_sum = 0.0;
    double hybrid_d2h_ms_sum = 0.0;
    double hybrid_h2d_ms_sum = 0.0;
    double hybrid_apply_ms_sum = 0.0;
    double hybrid_cleanup_ms_sum = 0.0;
    SvdBasisDevice warm_svd_basis;
    bool have_warm_svd_basis = false;
    GpuWarmRayleighWorkspace warm_rayleigh_workspace;
    GpuFullSvdWorkspace full_svd_workspace;
    HybridGpuWorkspace hybrid_runtime_workspace;
    for (int outer = 1; outer <= args.num_outer; ++outer) {
        const auto outer_t0 = std::chrono::steady_clock::now();
        const double linearize_ms = linearizeSe2GraphOnGpu(problem_dev, dev, args.huber_delta);

        GpuSvdBasisBuildResult outer_basis;
        bool use_external_gpu_basis = false;
        double basis_external_host_ms = 0.0;
        const auto basis_external_t0 = std::chrono::steady_clock::now();
        if (args.gpu_svd_warm_rr_after_first) {
            if (have_warm_svd_basis || args.gpu_svd_partial_from_start) {
                SvdBasisDevice empty_warm_basis;
                outer_basis = buildGpuWarmRayleighBasisFromDevice(
                    topology_graph,
                    dev,
                    args.group_size,
                    args.r_reduced,
                    have_warm_svd_basis ? warm_svd_basis : empty_warm_basis,
                    args.gpu_svd_tol,
                    args.gpu_svd_max_sweeps,
                    &warm_rayleigh_workspace,
                    have_warm_svd_basis ? 1 : 4,
                    dump != nullptr);
            } else {
                outer_basis = buildGpuSvdBasisFromDevice(
                    topology_graph,
                    dev,
                    args.group_size,
                    args.r_reduced,
                    args.gpu_svd_tol,
                    args.gpu_svd_max_sweeps,
                    dump != nullptr);
            }
            use_external_gpu_basis = true;
        } else if (args.gpu_svd_persistent_full_basis) {
            outer_basis = buildGpuSvdBasisFromDevice(
                topology_graph,
                dev,
                args.group_size,
                args.r_reduced,
                args.gpu_svd_tol,
                args.gpu_svd_max_sweeps,
                dump != nullptr,
                &full_svd_workspace);
            use_external_gpu_basis = true;
        }
        const auto basis_external_t1 = std::chrono::steady_clock::now();
        if (use_external_gpu_basis) {
            basis_external_host_ms =
                std::chrono::duration<double, std::milli>(basis_external_t1 - basis_external_t0).count();
        }

        HybridPrototypeResult h = runHybridHgbpPrototype(
            topology_graph,
            dev,
            args,
            schur_mode,
            nullptr,
            use_external_gpu_basis ? &outer_basis.host : nullptr,
            use_external_gpu_basis ? &outer_basis.device : nullptr,
            &cached_block_pattern,
            &cached_d_block_pattern,
            d_outer_delta,
            cached_sparse_factor.get(),
            &hybrid_runtime_workspace,
            dump,
            outer,
            use_external_gpu_basis ? &outer_basis.selected_values : nullptr);
        if (use_external_gpu_basis) {
            h.basis_host_ms += basis_external_host_ms;
            h.basis_build_ms += outer_basis.assemble_ms + outer_basis.eig_ms + outer_basis.extract_ms;
            h.basis_upload_ms += outer_basis.upload_ms;
            h.gpu_svd_assemble_ms += outer_basis.assemble_ms;
            h.gpu_svd_eig_ms += outer_basis.eig_ms;
            h.gpu_svd_extract_ms += outer_basis.extract_ms;
            h.total_ms += h.basis_host_ms;
            if (args.gpu_svd_warm_rr_after_first) {
                if (have_warm_svd_basis && !outer_basis.device_reused) {
                    freeSvdBasisDevice(warm_svd_basis);
                }
                warm_svd_basis = outer_basis.device;
                outer_basis.device = {};
                have_warm_svd_basis = true;
            } else if (!outer_basis.device_reused) {
                freeSvdBasisDevice(outer_basis.device);
                outer_basis.device = {};
            }
        }

        const double apply_gpu_ms = applyPoseDeltasOnGpu(problem_dev, d_outer_delta);
        if (dump != nullptr) {
            dump->writeDoubles(TrajectoryTag::OuterPoses, outer, -1, downloadProblemDevicePoseVector(problem_dev));
        }

        double download_pose_ms = 0.0;
        double objective_ms = 0.0;
        double printed_objective = std::numeric_limits<double>::quiet_NaN();
        if (outer == args.num_outer) {
            const auto download_pose_t0 = std::chrono::steady_clock::now();
            downloadProblemDevicePoses(problem_dev, problem);
            const auto download_pose_t1 = std::chrono::steady_clock::now();
            download_pose_ms =
                std::chrono::duration<double, std::milli>(download_pose_t1 - download_pose_t0).count();

            const auto objective_t0 = std::chrono::steady_clock::now();
            final_objective = nonlinearObjectiveRaw(problem);
            const auto objective_t1 = std::chrono::steady_clock::now();
            objective_ms =
                std::chrono::duration<double, std::milli>(objective_t1 - objective_t0).count();
            printed_objective = final_objective;
        }
        const auto outer_t1 = std::chrono::steady_clock::now();
        const double outer_ms =
            std::chrono::duration<double, std::milli>(outer_t1 - outer_t0).count();

        outer_ms_sum += outer_ms;
        linearize_ms_sum += linearize_ms;
        hybrid_ms_sum += h.total_ms;
        apply_gpu_ms_sum += apply_gpu_ms;
        download_pose_ms_sum += download_pose_ms;
        objective_ms_sum += objective_ms;
        basis_external_host_ms_sum += basis_external_host_ms;
        basis_host_ms_sum += h.basis_host_ms;

        const double basis_phase_host_ms =
            (h.basis_host_ms > 0.0 && args.hybrid_gpu_svd_basis)
                ? h.basis_host_ms
                : h.basis_build_ms + h.basis_upload_ms;
        const double hybrid_component_sum =
            basis_phase_host_ms +
            h.pattern_build_ms +
            h.pattern_upload_ms +
            h.alloc_ms +
            h.reset_ms +
            h.cholmod_load_ms +
            h.core_ms +
            h.fine_download_ms +
            h.cleanup_ms;
        const double hybrid_component_gap = h.total_ms - hybrid_component_sum;
        const double hybrid_core_component_sum =
            h.lower_ms +
            h.gpu_coarse_ms +
            h.d2h_ms +
            h.coarse_factor_ms +
            h.coarse_solve_ms +
            h.h2d_ms +
            h.apply_ms;
        const double hybrid_core_gap = h.core_ms - hybrid_core_component_sum;
        const double outer_known_profile =
            linearize_ms + h.total_ms + apply_gpu_ms + download_pose_ms + objective_ms;
        const double outer_profile_gap = outer_ms - outer_known_profile;
        outer_profile_known_ms_sum += outer_known_profile;
        outer_profile_gap_ms_sum += outer_profile_gap;
        hybrid_component_sum_ms += hybrid_component_sum;
        hybrid_component_gap_ms_sum += hybrid_component_gap;
        hybrid_core_component_sum_ms += hybrid_core_component_sum;
        hybrid_core_gap_ms_sum += hybrid_core_gap;
        hybrid_alloc_ms_sum += h.alloc_ms;
        hybrid_reset_ms_sum += h.reset_ms;
        hybrid_cholmod_load_ms_sum += h.cholmod_load_ms;
        hybrid_d2h_ms_sum += h.d2h_ms;
        hybrid_h2d_ms_sum += h.h2d_ms;
        hybrid_apply_ms_sum += h.apply_ms;
        hybrid_cleanup_ms_sum += h.cleanup_ms;

        std::cout << std::fixed << std::setprecision(3)
                  << "gpu_outer outer=" << outer
                  << " objective=" << std::setprecision(6) << printed_objective
                  << std::setprecision(3)
                  << " outer_ms=" << outer_ms
                  << " gpu_linearize_ms=" << linearize_ms
                  << " hybrid_ms=" << h.total_ms
                  << " hybrid_core_ms=" << h.core_ms
                  << " basis_build_ms=" << h.basis_build_ms
                  << " basis_upload_ms=" << h.basis_upload_ms
                  << " basis_host_ms=" << h.basis_host_ms
                  << " gpu_svd_assemble_ms=" << h.gpu_svd_assemble_ms
                  << " gpu_svd_eig_ms=" << h.gpu_svd_eig_ms
                  << " gpu_svd_extract_ms=" << h.gpu_svd_extract_ms
                  << " lower_ms=" << h.lower_ms
                  << " gpu_coarse_ms=" << h.gpu_coarse_ms
                  << " cholmod_factor_ms=" << h.coarse_factor_ms
                  << " cholmod_solve_ms=" << h.coarse_solve_ms
                  << " gpu_apply_pose_ms=" << apply_gpu_ms
                  << " download_pose_ms=" << download_pose_ms
                  << " objective_ms=" << objective_ms
                  << " e_norm=" << h.fine_norm
                  << "\n";
        if (args.hybrid_profile_phases) {
            const double basis_event_sum =
                outer_basis.assemble_ms + outer_basis.eig_ms + outer_basis.extract_ms + outer_basis.upload_ms;
            const double basis_external_gap =
                use_external_gpu_basis ? basis_external_host_ms - basis_event_sum : 0.0;
            std::cout << std::fixed << std::setprecision(3)
                      << "gpu_outer_profile outer=" << outer
                      << " basis_external_host_ms=" << basis_external_host_ms
                      << " basis_external_event_ms=" << (use_external_gpu_basis ? basis_event_sum : 0.0)
                      << " basis_external_gap_ms=" << basis_external_gap
                      << " basis_host_ms=" << h.basis_host_ms
                      << " basis_event_ms=" << h.basis_build_ms + h.basis_upload_ms
                      << " basis_host_minus_event_ms=" << h.basis_host_ms - (h.basis_build_ms + h.basis_upload_ms)
                      << " hybrid_alloc_ms=" << h.alloc_ms
                      << " hybrid_reset_ms=" << h.reset_ms
                      << " cholmod_load_ms=" << h.cholmod_load_ms
                      << " gpu_coarse_lambda_ms=" << h.gpu_coarse_lambda_ms
                      << " gpu_coarse_rhs_ms=" << h.gpu_coarse_rhs_ms
                      << " d2h_lambda_ms=" << h.d2h_lambda_ms
                      << " d2h_rhs_ms=" << h.d2h_rhs_ms
                      << " d2h_ms=" << h.d2h_ms
                      << " h2d_ms=" << h.h2d_ms
                      << " coarse_apply_ms=" << h.apply_ms
                      << " fine_download_ms=" << h.fine_download_ms
                      << " hybrid_cleanup_ms=" << h.cleanup_ms
                      << " hybrid_component_sum_ms=" << hybrid_component_sum
                      << " hybrid_component_gap_ms=" << hybrid_component_gap
                      << " hybrid_core_component_sum_ms=" << hybrid_core_component_sum
                      << " hybrid_core_gap_ms=" << hybrid_core_gap
                      << " outer_profile_known_ms=" << outer_known_profile
                      << " outer_profile_gap_ms=" << outer_profile_gap
                      << "\n";
        }
    }

    const auto total_t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(total_t1 - total_t0).count();
    std::cout << std::fixed << std::setprecision(6)
              << "gpu_outer_final_objective=" << final_objective << "\n";
    if (dump != nullptr) {
        dump->writeDoubles(TrajectoryTag::FinalObjective, args.num_outer, -1, std::vector<double>{final_objective});
        dumpFinalGbpState(dump, dev);
    }
    std::cout << std::fixed << std::setprecision(3)
              << "gpu_outer_summary outers=" << args.num_outer
              << " total_ms=" << total_ms
              << " summed_outer_ms=" << outer_ms_sum
              << " summed_gpu_linearize_ms=" << linearize_ms_sum
              << " summed_hybrid_ms=" << hybrid_ms_sum
              << " summed_gpu_apply_pose_ms=" << apply_gpu_ms_sum
              << " summed_download_pose_ms=" << download_pose_ms_sum
              << " summed_objective_ms=" << objective_ms_sum
              << "\n";
    if (args.hybrid_profile_phases) {
        const double total_closure_gap = total_ms - outer_profile_known_ms_sum;
        const double total_closure_rel = std::abs(total_closure_gap) / std::max(total_ms, 1e-12);
        const double outer_sum_closure_gap = outer_ms_sum - outer_profile_known_ms_sum;
        const double outer_sum_closure_rel = std::abs(outer_sum_closure_gap) / std::max(outer_ms_sum, 1e-12);
        std::cout << std::fixed << std::setprecision(3)
                  << "gpu_outer_profile_summary outers=" << args.num_outer
                  << " total_ms=" << total_ms
                  << " summed_outer_ms=" << outer_ms_sum
                  << " summed_profile_known_ms=" << outer_profile_known_ms_sum
                  << " summed_outer_profile_gap_ms=" << outer_profile_gap_ms_sum
                  << " total_closure_gap_ms=" << total_closure_gap
                  << " total_closure_rel=" << total_closure_rel
                  << " outer_sum_closure_gap_ms=" << outer_sum_closure_gap
                  << " outer_sum_closure_rel=" << outer_sum_closure_rel
                  << " summed_basis_external_host_ms=" << basis_external_host_ms_sum
                  << " summed_basis_host_ms=" << basis_host_ms_sum
                  << " summed_hybrid_component_ms=" << hybrid_component_sum_ms
                  << " summed_hybrid_component_gap_ms=" << hybrid_component_gap_ms_sum
                  << " summed_hybrid_core_component_ms=" << hybrid_core_component_sum_ms
                  << " summed_hybrid_core_gap_ms=" << hybrid_core_gap_ms_sum
                  << " summed_hybrid_alloc_ms=" << hybrid_alloc_ms_sum
                  << " summed_hybrid_reset_ms=" << hybrid_reset_ms_sum
                  << " summed_hybrid_cholmod_load_ms=" << hybrid_cholmod_load_ms_sum
                  << " summed_hybrid_d2h_ms=" << hybrid_d2h_ms_sum
                  << " summed_hybrid_h2d_ms=" << hybrid_h2d_ms_sum
                  << " summed_hybrid_coarse_apply_ms=" << hybrid_apply_ms_sum
                  << " summed_hybrid_cleanup_ms=" << hybrid_cleanup_ms_sum
                  << "\n";
    }

    cudaFree(d_outer_delta);
    if (have_warm_svd_basis) {
        freeSvdBasisDevice(warm_svd_basis);
    }
    if (cached_d_block_pattern.block_count != 0) {
        freeCoarseBlockPatternDevice(cached_d_block_pattern);
    }
    freeProblemDevice(problem_dev);
    freeGbpDevice(dev);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parseArgs(argc, argv);
        cudaCheck(cudaSetDevice(0), "cudaSetDevice");
        if (args.kernel == "jacobi_bucket") {
            throw std::runtime_error("jacobi_bucket is deprecated for this task; use the true GBP path: --kernel gbp_persistent");
        }
        if (args.hybrid_hgbp && args.kernel != "gbp_persistent") {
            throw std::runtime_error("--hybrid-hgbp currently requires --kernel gbp_persistent");
        }
        if (args.kernel == "gbp_graph_split" && args.fixed_lambda_start >= 0) {
            throw std::runtime_error("--kernel gbp_graph_split currently supports only --fixed-lambda-start -1");
        }

        auto t0 = std::chrono::steady_clock::now();
        const int schur_mode = (args.schur_kernel == "inverse") ? 0 : 1;
        Problem problem = args.problem_file.empty()
            ? makeSyntheticChain(args.synthetic_chain)
            : loadProblem(args.problem_file);
        auto t1 = std::chrono::steady_clock::now();

        if (args.hybrid_hgbp && args.num_outer > 0) {
            std::cout << "problem nodes=" << problem.init_poses.size()
                      << " edges=" << problem.edges.size()
                      << " num_outer=" << args.num_outer
                      << " inner_cycles=" << args.hybrid_cycles
                      << " sweeps=" << args.sweeps
                      << " basis_mode=" << (args.hybrid_gpu_svd_basis ? "gpu_svd" : (args.hybrid_rigid_basis ? "rigid" : (args.hybrid_svd_basis ? "svd" : "identity")))
                      << " r_reduced=" << (args.hybrid_svd_basis ? args.r_reduced : 3)
                      << " huber_delta=" << args.huber_delta
                      << " coop_block_policy=" << args.coop_block_policy
                      << " persistent_threads=" << args.persistent_threads
                      << " fixed_lambda_start=" << args.fixed_lambda_start
                      << " gpu_svd_warm_rr_after_first=" << (args.gpu_svd_warm_rr_after_first ? 1 : 0)
                      << " gpu_svd_partial_from_start=" << (args.gpu_svd_partial_from_start ? 1 : 0)
                      << " gpu_svd_persistent_full_basis=" << (args.gpu_svd_persistent_full_basis ? 1 : 0)
                      << " gpu_svd_tol=" << args.gpu_svd_tol
                      << " gpu_svd_max_sweeps=" << args.gpu_svd_max_sweeps
                      << " hybrid_profile_phases=" << (args.hybrid_profile_phases ? 1 : 0)
                      << " trajectory_dump_enabled=" << (!args.trajectory_dump.empty() ? 1 : 0)
                      << " load_ms=" << std::fixed << std::setprecision(3)
                      << std::chrono::duration<double, std::milli>(t1 - t0).count()
                      << "\n";
            printGbpCoopLaunchConfig(
                static_cast<int>(problem.init_poses.size()),
                static_cast<int>(problem.edges.size()),
                args.persistent_threads,
                args.coop_block_policy);
            if (args.hybrid_gpu_svd_basis) {
                runHybridOuterPrototypeGpuLinearize(problem, args, schur_mode);
            } else {
                runHybridOuterPrototype(problem, args, schur_mode);
            }
            return 0;
        }

        GbpGraph graph = buildGbpGraph(problem, args.huber_delta);
        auto t2 = std::chrono::steady_clock::now();

        IncomingFlatHost incoming = buildIncomingFlat(graph, args.incoming_layout);
        auto t3 = std::chrono::steady_clock::now();

        std::cout << "problem nodes=" << graph.n
                  << " edges=" << graph.m
                  << " sweeps=" << args.sweeps
                  << " gbp_damping=" << args.omega
                  << " kernel=" << args.kernel
                  << " incoming_layout=" << args.incoming_layout
                  << " schur_kernel=" << args.schur_kernel
                  << " coop_block_policy=" << args.coop_block_policy
                  << " persistent_threads=" << args.persistent_threads
                  << " fixed_lambda_start=" << args.fixed_lambda_start
                  << " huber_delta=" << args.huber_delta << "\n";
        printIncomingStats(graph, incoming);
        GbpCoopLaunchConfig gbp_launch;
        if (args.kernel == "gbp_persistent") {
            gbp_launch = computeGbpCoopLaunchConfig(
                graph.n, graph.m, args.persistent_threads, args.coop_block_policy);
            printGbpCoopLaunchConfig(graph.n, graph.m, args.persistent_threads, args.coop_block_policy);
        }

        GbpDevice dev = uploadGbpDevice(graph, incoming);
        cudaCheck(cudaDeviceSynchronize(), "post upload sync");
        auto t4 = std::chrono::steady_clock::now();

        if (args.hybrid_hgbp) {
            const auto h = runHybridHgbpPrototype(graph, dev, args, schur_mode);
            const auto ms = [](auto a, auto b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::cout << std::fixed << std::setprecision(3)
                      << "timing_ms load=" << ms(t0, t1)
                      << " linearize=" << ms(t1, t2)
                      << " incoming_bucketize=" << ms(t2, t3)
                      << " upload=" << ms(t3, t4)
                      << "\n";
            std::cout << std::fixed << std::setprecision(3)
                      << "hybrid_hgbp groups=" << h.groups
                      << " coarse_dim=" << h.coarse_dim
                      << " coarse_blocks=" << h.coarse_blocks
                      << " coarse_mode=" << (h.sparse_coarse ? "sparse_block" : "dense")
                      << " basis_mode=" << (args.hybrid_gpu_svd_basis ? "gpu_svd" : (args.hybrid_rigid_basis ? "rigid" : (args.hybrid_svd_basis ? "svd" : "identity")))
                      << " r_reduced=" << (args.hybrid_svd_basis ? args.r_reduced : 3)
                      << " cycles=" << args.hybrid_cycles
                      << " group_size=" << args.group_size
                      << " coarse_scale=" << args.coarse_scale
                      << " lower_sweeps_per_cycle=" << args.sweeps
                      << "\n";
            std::cout << std::fixed << std::setprecision(3)
                      << "hybrid_timing_ms lower_gbp=" << h.lower_ms
                      << " gpu_coarse_assemble=" << h.gpu_coarse_ms
                      << " d2h_coarse=" << h.d2h_ms
                      << " cpu_cholmod_factor=" << h.coarse_factor_ms
                      << " cpu_cholmod_solve=" << h.coarse_solve_ms
                      << " h2d_delta=" << h.h2d_ms
                      << " gpu_prolong_apply=" << h.apply_ms
                      << " core=" << h.core_ms
                      << " total=" << h.total_ms
                      << "\n";
            std::cout << std::fixed << std::setprecision(3)
                      << "hybrid_setup_ms basis_build=" << h.basis_build_ms
                      << " basis_upload=" << h.basis_upload_ms
                      << " pattern_build=" << h.pattern_build_ms
                      << " pattern_upload=" << h.pattern_upload_ms
                      << " alloc=" << h.alloc_ms
                      << " reset=" << h.reset_ms
                      << " cholmod_load=" << h.cholmod_load_ms
                      << " fine_download=" << h.fine_download_ms
                      << " cleanup=" << h.cleanup_ms
                      << "\n";
            std::cout << std::fixed << std::setprecision(3)
                      << "hybrid_detail_ms gpu_lambda=" << h.gpu_coarse_lambda_ms
                      << " gpu_rhs=" << h.gpu_coarse_rhs_ms
                      << " d2h_lambda=" << h.d2h_lambda_ms
                      << " d2h_rhs=" << h.d2h_rhs_ms
                      << "\n";
            std::cout << std::scientific << std::setprecision(6)
                      << "hybrid_norms fine_x=" << h.fine_norm
                      << " coarse_rhs=" << h.coarse_rhs_norm
                      << " coarse_delta=" << h.coarse_delta_norm
                      << "\n";
            freeGbpDevice(dev);
            return 0;
        }

        GbpGraphSplitExec graph_split_exec;
        bool have_graph_split_exec = false;
        double graph_split_setup_ms = 0.0;

        if (args.kernel == "gbp_graph_split" && args.warmup > 0) {
            GbpGraphSplitExec warm_graph =
                createGbpGraphSplitExec(
                    dev, args.warmup, args.omega, schur_mode, args.graph_split_threads);
            resetGbpDevice(dev);
            cudaCheck(cudaDeviceSynchronize(), "sync graph split warm reset");
            launchGbpGraphSplit(warm_graph);
            cudaCheck(cudaStreamSynchronize(warm_graph.stream), "sync graph split warmup");
            destroyGbpGraphSplitExec(warm_graph);
        } else if (args.kernel == "gbp_persistent" && args.warmup > 0) {
            resetGbpDevice(dev);
            launchGbpPersistent(
                dev,
                args.warmup,
                args.omega,
                schur_mode,
                args.fixed_lambda_start,
                gbp_launch.launched_blocks,
                args.persistent_threads);
            cudaCheck(cudaDeviceSynchronize(), "warmup sweeps");
        }

        if (args.kernel == "gbp_graph_split") {
            const auto graph_setup_t0 = std::chrono::steady_clock::now();
            graph_split_exec = createGbpGraphSplitExec(
                dev, args.sweeps, args.omega, schur_mode, args.graph_split_threads);
            const auto graph_setup_t1 = std::chrono::steady_clock::now();
            graph_split_setup_ms =
                std::chrono::duration<double, std::milli>(graph_setup_t1 - graph_setup_t0).count();
            have_graph_split_exec = true;
            std::cout << std::fixed << std::setprecision(3)
                      << "graph_split_setup message_threads=" << graph_split_exec.message_threads
                      << " message_blocks=" << graph_split_exec.message_blocks
                      << " belief_threads=" << graph_split_exec.belief_threads
                      << " belief_blocks=" << graph_split_exec.belief_blocks
                      << " graph_nodes=" << (2 * args.sweeps)
                      << " setup_ms=" << graph_split_setup_ms
                      << "\n";
        }

        cudaEvent_t ev_start = nullptr;
        cudaEvent_t ev_stop = nullptr;
        cudaCheck(cudaEventCreate(&ev_start), "create start event");
        cudaCheck(cudaEventCreate(&ev_stop), "create stop event");

        float best_ms = std::numeric_limits<float>::infinity();
        std::vector<float> repeat_ms;
        repeat_ms.reserve(static_cast<size_t>(args.repeat));
        for (int rep = 0; rep < args.repeat; ++rep) {
            resetGbpDevice(dev);
            if (args.kernel == "gbp_graph_split") {
                cudaCheck(cudaDeviceSynchronize(), "sync graph split timed reset");
                cudaCheck(cudaEventRecord(ev_start, graph_split_exec.stream), "record graph split start");
                launchGbpGraphSplit(graph_split_exec);
                cudaCheck(cudaEventRecord(ev_stop, graph_split_exec.stream), "record graph split stop");
            } else {
                cudaCheck(cudaEventRecord(ev_start), "record start");
                launchGbpPersistent(
                    dev,
                    args.sweeps,
                    args.omega,
                    schur_mode,
                    args.fixed_lambda_start,
                    gbp_launch.launched_blocks,
                    args.persistent_threads);
                cudaCheck(cudaEventRecord(ev_stop), "record stop");
            }
            cudaCheck(cudaEventSynchronize(ev_stop), "sync stop");
            float ms = 0.0f;
            cudaCheck(cudaEventElapsedTime(&ms, ev_start, ev_stop), "elapsed time");
            best_ms = std::min(best_ms, ms);
            repeat_ms.push_back(ms);
            if (args.telemetry_repeats) {
                const GpuTelemetry telemetry = queryGpuTelemetry();
                std::cout << std::fixed << std::setprecision(3)
                          << "repeat_telemetry rep=" << rep
                          << " gpu_ms=" << ms
                          << " sm_clock_mhz=" << telemetry.sm_clock_mhz
                          << " pstate=" << telemetry.pstate
                          << " power_w=" << telemetry.power_w
                          << " temp_c=" << telemetry.temp_c
                          << " available=" << (telemetry.available ? 1 : 0)
                          << "\n";
            }
        }
        std::vector<float> sorted_repeat_ms = repeat_ms;
        std::sort(sorted_repeat_ms.begin(), sorted_repeat_ms.end());
        auto percentile_ms = [&](double q) -> float {
            if (sorted_repeat_ms.empty()) {
                return 0.0f;
            }
            const double pos = q * static_cast<double>(sorted_repeat_ms.size() - 1);
            const size_t idx = static_cast<size_t>(std::llround(pos));
            return sorted_repeat_ms[std::min(idx, sorted_repeat_ms.size() - 1)];
        };
        const float p10_ms = percentile_ms(0.10);
        const float median_ms = percentile_ms(0.50);
        const float p90_ms = percentile_ms(0.90);

        std::vector<double> gpu_belief_lam;
        std::vector<double> gpu_belief_eta;
        downloadGbpBelief(dev, args.sweeps, gpu_belief_lam, gpu_belief_eta);
        std::vector<double> gpu_mean = beliefMeanVector(gpu_belief_lam, gpu_belief_eta, graph.n);

        const auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::cout << std::fixed << std::setprecision(3)
                  << "timing_ms load=" << ms(t0, t1)
                  << " linearize=" << ms(t1, t2)
                  << " incoming_bucketize=" << ms(t2, t3)
                  << " upload=" << ms(t3, t4)
                  << " gpu_gbp_sweeps_best=" << best_ms
                  << " gpu_gbp_sweeps_p10=" << p10_ms
                  << " gpu_gbp_sweeps_median=" << median_ms
                  << " gpu_gbp_sweeps_p90=" << p90_ms
                  << " repeat=" << args.repeat
                  << " per_sweep=" << (args.sweeps > 0 ? best_ms / args.sweeps : 0.0f)
                  << "\n";
        const int full_lambda_sweeps = (args.fixed_lambda_start >= 0)
            ? std::min(args.sweeps, args.fixed_lambda_start + 1)
            : args.sweeps;
        const int eta_only_sweeps = std::max(0, args.sweeps - full_lambda_sweeps);
        std::cout << "fixed_lambda_stats full_lambda_sweeps=" << full_lambda_sweeps
                  << " eta_only_sweeps=" << eta_only_sweeps
                  << " eta_map_builds=" << (eta_only_sweeps > 0 ? 1 : 0)
                  << "\n";
        const double gpu_seconds = static_cast<double>(best_ms) * 1e-3;
        const double directed_factor_messages = 2.0 * static_cast<double>(graph.m) * static_cast<double>(args.sweeps);
        const double padded_variable_slots = static_cast<double>(incoming.total_slots) * static_cast<double>(args.sweeps);
        const double factor_mmsg_s = gpu_seconds > 0.0 ? directed_factor_messages / gpu_seconds / 1.0e6 : 0.0;
        const double padded_mslot_s = gpu_seconds > 0.0 ? padded_variable_slots / gpu_seconds / 1.0e6 : 0.0;
        std::cout << std::fixed << std::setprecision(3)
                  << "throughput factor_msg_Ms=" << factor_mmsg_s
                  << " padded_incoming_slot_Ms=" << padded_mslot_s
                  << " directed_factor_messages=" << static_cast<long long>(directed_factor_messages)
                  << " padded_variable_slots=" << static_cast<long long>(padded_variable_slots)
                  << "\n";
        std::cout << std::scientific << std::setprecision(6)
                  << "gpu_mean_norm=" << norm2(gpu_mean) << "\n";

        if (args.verify) {
            std::vector<double> cpu_belief_lam;
            std::vector<double> cpu_belief_eta;
            auto c0 = std::chrono::steady_clock::now();
            cpuGbpSweeps(
                graph, incoming, args.sweeps, args.fixed_lambda_start, args.omega, args.cpu_threads,
                cpu_belief_lam, cpu_belief_eta);
            std::vector<double> cpu_mean = beliefMeanVector(cpu_belief_lam, cpu_belief_eta, graph.n);
            auto c1 = std::chrono::steady_clock::now();
            const double cpu_ms = ms(c0, c1);
            std::cout << std::fixed << std::setprecision(3)
                      << "cpu_gbp_reference_ms=" << cpu_ms
                      << " cpu_threads=" << args.cpu_threads
                      << " cpu_per_sweep=" << (args.sweeps > 0 ? cpu_ms / args.sweeps : 0.0)
                      << " gpu_vs_cpu_speedup=" << (best_ms > 0.0f ? cpu_ms / best_ms : 0.0)
                      << "\n";
            std::cout << std::scientific << std::setprecision(6)
                      << "gpu_vs_cpu_mean_rel=" << relativeError(gpu_mean, cpu_mean)
                      << " gpu_vs_cpu_belief_lam_rel=" << relativeError(gpu_belief_lam, cpu_belief_lam)
                      << " gpu_vs_cpu_belief_eta_rel=" << relativeError(gpu_belief_eta, cpu_belief_eta)
                      << " cpu_mean_norm=" << norm2(cpu_mean) << "\n";
        }

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        if (have_graph_split_exec) {
            destroyGbpGraphSplitExec(graph_split_exec);
        }
        freeGbpDevice(dev);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
