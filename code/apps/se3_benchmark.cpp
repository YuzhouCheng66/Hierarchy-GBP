#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include "hgbp/se3_solver.h"

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

int parseInt(const std::string& option, const char* text) {
    try {
        std::size_t consumed = 0;
        const long long value = std::stoll(text, &consumed, 0);
        if (consumed != std::string(text).size() ||
            value < (std::numeric_limits<int>::min)() ||
            value > (std::numeric_limits<int>::max)()) {
            throw std::invalid_argument("out of range");
        }
        return static_cast<int>(value);
    } catch (...) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
}

double parseDouble(const std::string& option, const char* text) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(text, &consumed);
        if (consumed != std::string(text).size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (...) {
        throw std::runtime_error("invalid value for " + option + ": " + text);
    }
}

void setEnv(const char* name, const std::string& value) {
#if defined(_WIN32)
    _putenv_s(name, value.c_str());
#else
    setenv(name, value.c_str(), 1);
#endif
}

#ifdef _WIN32
bool applyProcessAffinityMask(const std::string& text, DWORD_PTR& previous_process_mask) {
    if (text.empty()) {
        return false;
    }
    const unsigned long long mask_value = std::stoull(text, nullptr, 0);
    if (mask_value == 0ull) {
        throw std::runtime_error("invalid --process-affinity-mask value: 0");
    }
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &process_mask, &system_mask)) {
        throw std::runtime_error("GetProcessAffinityMask failed");
    }
    const DWORD_PTR requested_mask = static_cast<DWORD_PTR>(mask_value);
    const DWORD_PTR clamped_mask = requested_mask & system_mask;
    if (clamped_mask == 0) {
        throw std::runtime_error(
            "requested --process-affinity-mask does not overlap system affinity mask"
        );
    }
    previous_process_mask = process_mask;
    if (!SetProcessAffinityMask(GetCurrentProcess(), clamped_mask)) {
        throw std::runtime_error("SetProcessAffinityMask failed");
    }
    return true;
}
#endif

void configureUniformSE3Path(
    int fixed_lambda_after_sweeps,
    int final_direct_polish_steps,
    int basis_rebuild_period,
    int coarse_numeric_rebuild_period,
    int coarse_reuse_pcg_iters,
    int implicit_fine_operator_threads,
    int partial_basis_max_iters,
    bool partial_basis_accept_unconverged
) {
    setEnv("GBP_SE3_PACKED_SOA_SWEEPS", "1");
    setEnv("GBP_SE3_PACKED_SOA_DEFER_GRAPH_COPY", "1");
    setEnv("GBP_SE3_PACKED_SOA_FULL_COPY_EACH_CALL", "0");
    setEnv("GBP_SE3_PROFILE_SYNC_TIMING", "0");

    setEnv("GBP_SE3_MESSAGE_DAMPING", "0.3");
    setEnv("GBP_SE3_POST_COARSE_MESSAGE_SWEEPS", "0");
    setEnv("GBP_FACTOR_MSG_MAX_REL_UPDATE", "2");

    setEnv("GBP_SE3_REUSE_COARSE_PATTERN", "1");
    setEnv("GBP_SE3_CACHED_COARSE_LAMBDA", "1");
    setEnv("GBP_SE3_CACHED_COARSE_LAMBDA_MIN_GROUPS", "200");
    setEnv("GBP_SE3_CACHED_COARSE_INPLACE_SYMMETRIZE", "1");
    setEnv("GBP_SE3_COARSE_RIDGE_INPLACE", "1");
    setEnv(
        "GBP_SE3_COARSE_NUMERIC_REBUILD_PERIOD",
        std::to_string(coarse_numeric_rebuild_period)
    );
    setEnv(
        "GBP_SE3_COARSE_REUSE_PCG_ITERS",
        std::to_string(coarse_reuse_pcg_iters)
    );
    setEnv("GBP_SE3_IMPLICIT_FINE_OPERATOR", "1");
    setEnv(
        "GBP_SE3_IMPLICIT_FINE_OPERATOR_THREADS",
        std::to_string(implicit_fine_operator_threads)
    );
    setEnv("GBP_SE3_IMPLICIT_FINE_OPERATOR_COMPARE", "0");
    setEnv("GBP_SE3_FIXED_LAMBDA_LATE_START_OUTER", "-1");
    setEnv("GBP_SE3_FIXED_LAMBDA_AFTER_SWEEPS_LATE", "10");
    setEnv("GBP_SE3_FIXED_LAMBDA_AFTER_SWEEPS", std::to_string(fixed_lambda_after_sweeps));
    setEnv("GBP_SE3_FIXED_LAMBDA_6D_INV_CACHE_START_OUTER", "12");
    setEnv("GBP_SE3_PARTIAL_BASIS_EIGEN", "1");
    // Packed kernels retain their validated serial/parallel selection.
    setEnv("GBP_SE3_FINAL_DIRECT_POLISH_STEPS", std::to_string(final_direct_polish_steps));
    setEnv("GBP_SE3_PARTIAL_BASIS_RESIDUAL_CHECK_PERIOD", "1");
    setEnv("GBP_SE3_BASIS_REBUILD_PERIOD", std::to_string(basis_rebuild_period));
    setEnv("GBP_SE3_BASIS_REBUILD_WARMUP_OUTERS", "0");
    setEnv("GBP_SE3_PARTIAL_BASIS_ACCEPT_RESIDUAL_TOL", "-1");
    setEnv("GBP_SE3_PARTIAL_BASIS_LARGE_GROUP_MIN_GROUPS", "350");
    setEnv(
        "GBP_SE3_PARTIAL_BASIS_LARGE_GROUP_ACCEPT_RESIDUAL_TOL",
        "-1"
    );
    setEnv("GBP_SE3_PARTIAL_BASIS_SMALL_GROUP_ACCEPT_RESIDUAL_TOL", "-1");
    setEnv(
        "GBP_SE3_PARTIAL_BASIS_ACCEPT_UNCONVERGED",
        partial_basis_accept_unconverged ? "1" : "0"
    );
    setEnv(
        "GBP_SE3_PARTIAL_BASIS_MAX_ITERS",
        std::to_string(partial_basis_max_iters)
    );

    // Pin the released policy even when inherited experiment settings are present.
    setEnv("HGBP_SE3_ADAPTIVE_PRECISION", "1");
    setEnv("HGBP_SE3_BALANCED_INITIALIZATION", "1");
    setEnv("HGBP_SE3_RESET_TRANSPORT", "1");
    setEnv("HGBP_CYCLE_ENERGY", "1");
    setEnv("HGBP_CYCLE_ENERGY_RESTART", "1");
    setEnv("HGBP_SE3_PERSISTENT_SWEEPS", "1");
    setEnv("HGBP_SE3_SMOOTHER", "gbp");
    setEnv("HGBP_SE3_SKIP_COARSE", "0");
    setEnv("HGBP_SE3_CYCLE_MESSAGE_REBUILD", "0");
    setEnv("HGBP_SE3_AUTOMATIC_COARSE", "1");
    setEnv("HGBP_SE3_PRECISION_INITIALIZATION", "warm-transport-balanced");
    setEnv("HGBP_SE3_PRECISION_AUDIT", "");
    setEnv("HGBP_SE3_EXACT_BASIS_CACHE", "0");
    setEnv("HGBP_SE3_PREDICT_PARTIAL_BUDGET", "0");
    setEnv("HGBP_SE3_AMORTIZED_COARSE", "1");
    setEnv("HGBP_SE3_COARSE_CHOLMOD", "1");
    setEnv("HGBP_SE3_ROBUST_LINE_SEARCH", "1");
    setEnv("HGBP_SE3_BASIS_PREPASS", "none");
    setEnv("HGBP_SE3_BOUNDARY_BASIS", "none");
    setEnv("HGBP_SE3_BUFFERED_OBJECTIVE", "1");
    setEnv("HGBP_SE3_ETA_LIFT", "precision");
    setEnv("HGBP_SE3_CYCLE_AUDIT", "0");
    setEnv("HGBP_SE3_PRECISION_DEFECT", "none");
    setEnv("HGBP_SE3_RESIDUAL_STOP", "1");
}

template <typename Row>
double sumOuterSeconds(const std::vector<Row>& rows) {
    double total = 0.0;
    for (const Row& row : rows) {
        total += row.outer_total_sec;
    }
    return total;
}

void printUsage() {
    std::cout
        << "Usage: se3_benchmark --problem-file <path> [--out-json <path>]\n"
        << "  Defaults: outer=20, c=5, k=50, group=20, r=12, threads=16, huber=5,\n"
        << "            adaptive variance, diagonal messages, balanced transported precision,\n"
        << "            automatic coarse correction, cycle line search, no fine direct solve.\n"
        << "  Optional: --num-outer N --inner-cycles C --pre-sweeps K --group-size G\n"
        << "            --r-reduced R --threads T --huber-delta D\n"
        << "            --implicit-fine-operator-threads N (defaults to --threads)\n"
        << "            --process-affinity-mask MASK\n"
        << "            --write-poses\n"
        << "  Formal-command compatibility (only these values are supported):\n"
        << "            --basis-rebuild-period 1 --coarse-numeric-rebuild-period 1\n"
        << "            --coarse-reuse-pcg-iters 1 --fixed-lambda-after-sweeps -1\n"
        << "            --partial-basis-max-iters 6 --partial-basis-accept-unconverged 0\n"
        << "            --final-direct-polish-steps 0 --skip-direct\n"
        << "            --variance-policy adaptive --message-initialization diagonal\n"
        << "            --precision-initialization warm-transport-balanced\n"
        << "            --persistent-sweeps --automatic-coarse --cycle-line-search\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string problem_file;
        std::string out_json = "se3_benchmark_results.json";
        int num_outer = 20;
        int inner_cycles = 5;
        int pre_sweeps = 50;
        int group_size = 20;
        int r_reduced = 12;
        int sync_num_threads = 16;
        int fixed_lambda_after_sweeps = -1;
        int final_direct_polish_steps = 0;
        int basis_rebuild_period = 1;
        int coarse_numeric_rebuild_period = 1;
        int coarse_reuse_pcg_iters = 1;
        int implicit_fine_operator_threads = 16;
        bool implicit_fine_operator_threads_set = false;
        int partial_basis_max_iters = 6;
        int partial_basis_accept_unconverged = 0;
        std::string process_affinity_mask;
        const bool direct_enabled = false;
        bool write_poses = false;
        std::string variance_policy = "adaptive";
        std::string message_initialization = "diagonal";
        std::string precision_initialization = "warm-transport-balanced";
        hgbp::se3::RobustLossConfig robust_loss_config;
        robust_loss_config.huber_delta = 5.0;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--problem-file" && i + 1 < argc) {
                problem_file = argv[++i];
            } else if (arg == "--out-json" && i + 1 < argc) {
                out_json = argv[++i];
            } else if (arg == "--num-outer" && i + 1 < argc) {
                num_outer = parseInt(arg, argv[++i]);
            } else if (arg == "--inner-cycles" && i + 1 < argc) {
                inner_cycles = parseInt(arg, argv[++i]);
            } else if (arg == "--pre-sweeps" && i + 1 < argc) {
                pre_sweeps = parseInt(arg, argv[++i]);
            } else if (arg == "--group-size" && i + 1 < argc) {
                group_size = parseInt(arg, argv[++i]);
            } else if (arg == "--r-reduced" && i + 1 < argc) {
                r_reduced = parseInt(arg, argv[++i]);
            } else if (arg == "--threads" && i + 1 < argc) {
                sync_num_threads = parseInt(arg, argv[++i]);
            } else if (arg == "--fixed-lambda-after-sweeps" && i + 1 < argc) {
                fixed_lambda_after_sweeps = parseInt(arg, argv[++i]);
            } else if (arg == "--variance-policy" && i + 1 < argc) {
                variance_policy = argv[++i];
            } else if (arg == "--message-initialization" && i+1<argc) {
                message_initialization=argv[++i];
            } else if (arg == "--cycle-line-search" || arg == "--persistent-sweeps" ||
                       arg == "--automatic-coarse" || arg == "--skip-direct") {
                // Accepted formal commands explicitly restate the released defaults.
            } else if(arg=="--precision-initialization" && i+1<argc) {
                precision_initialization=argv[++i];
            } else if (arg == "--basis-rebuild-period" && i + 1 < argc) {
                basis_rebuild_period = parseInt(arg, argv[++i]);
            } else if (arg == "--coarse-numeric-rebuild-period" &&
                       i + 1 < argc) {
                coarse_numeric_rebuild_period = parseInt(arg, argv[++i]);
            } else if (arg == "--coarse-reuse-pcg-iters" && i + 1 < argc) {
                coarse_reuse_pcg_iters = parseInt(arg, argv[++i]);
            } else if (arg == "--implicit-fine-operator-threads" &&
                       i + 1 < argc) {
                implicit_fine_operator_threads = parseInt(arg, argv[++i]);
                implicit_fine_operator_threads_set = true;
            } else if (arg == "--partial-basis-max-iters" && i + 1 < argc) {
                partial_basis_max_iters = parseInt(arg, argv[++i]);
            } else if (arg == "--partial-basis-accept-unconverged" &&
                       i + 1 < argc) {
                partial_basis_accept_unconverged = parseInt(arg, argv[++i]);
            } else if (arg == "--final-direct-polish-steps" && i + 1 < argc) {
                final_direct_polish_steps = parseInt(arg, argv[++i]);
            } else if (arg == "--huber-delta" && i + 1 < argc) {
                robust_loss_config.huber_delta = parseDouble(arg, argv[++i]);
            } else if (arg == "--process-affinity-mask" && i + 1 < argc) {
                process_affinity_mask = argv[++i];
            } else if (arg == "--write-poses") {
                write_poses = true;
            } else if (arg == "--help" || arg == "-h") {
                printUsage();
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + arg);
            }
        }

        if (problem_file.empty()) {
            throw std::runtime_error("--problem-file is required");
        }
        if (!implicit_fine_operator_threads_set) {
            implicit_fine_operator_threads = sync_num_threads;
        }
        if (num_outer < 1 || inner_cycles < 1 || pre_sweeps < 0 ||
            group_size < 1 || r_reduced < 1 || sync_num_threads < 1 ||
            implicit_fine_operator_threads < 1 ||
            robust_loss_config.huber_delta < 0.0) {
            throw std::runtime_error("invalid benchmark configuration");
        }
        if (fixed_lambda_after_sweeps != -1 || final_direct_polish_steps != 0 ||
            basis_rebuild_period != 1 || coarse_numeric_rebuild_period != 1 ||
            coarse_reuse_pcg_iters != 1 || partial_basis_max_iters != 6 ||
            partial_basis_accept_unconverged != 0 || variance_policy != "adaptive" ||
            message_initialization != "diagonal" ||
            precision_initialization != "warm-transport-balanced") {
            throw std::runtime_error(
                "only the released adaptive SE3 policy is supported; see --help for compatibility values"
            );
        }

        configureUniformSE3Path(
            fixed_lambda_after_sweeps,
            final_direct_polish_steps,
            basis_rebuild_period,
            coarse_numeric_rebuild_period,
            coarse_reuse_pcg_iters,
            implicit_fine_operator_threads,
            partial_basis_max_iters,
            partial_basis_accept_unconverged != 0
        );

#ifdef _WIN32
        DWORD_PTR previous_affinity_mask = 0;
        const bool applied_affinity_mask =
            applyProcessAffinityMask(process_affinity_mask, previous_affinity_mask);
#endif

        const hgbp::se3::Problem problem = hgbp::se3::loadSyntheticSE3Problem(problem_file);
        const hgbp::se3::Results results = hgbp::se3::runSyntheticSE3Experiment(
            problem,
            num_outer,
            inner_cycles,
            pre_sweeps,
            group_size,
            r_reduced,
            robust_loss_config,
            sync_num_threads,
            direct_enabled
        );

        hgbp::se3::writeSyntheticSE3ExperimentResultsJson(
            results,
            out_json,
            num_outer,
            inner_cycles,
            pre_sweeps,
            group_size,
            r_reduced,
            robust_loss_config,
            sync_num_threads,
            direct_enabled
        );

        if (write_poses) {
            std::string pose_json = out_json;
            const std::size_t dot = pose_json.rfind('.');
            if (dot == std::string::npos) {
                pose_json += "_poses.json";
            } else {
                pose_json.insert(dot, "_poses");
            }
            hgbp::se3::writeSyntheticSE3ExperimentPoseHistoryJson(results, pose_json);
        }

        const double mg_total_sec = sumOuterSeconds(results.mg_history);
        const double mg_cost = results.mg_history.empty()
            ? 0.0
            : results.mg_history.back().nonlinear_objective;

        std::cout << "wrote " << out_json << "\n";
        std::cout << "nodes=" << results.num_poses
                  << " factors=" << results.num_edges << "\n";
        std::cout << "mg_total_sec=" << mg_total_sec
                  << " mg_cost=" << mg_cost << "\n";
#ifdef _WIN32
        if (!process_affinity_mask.empty()) {
            std::cout << "process_affinity_mask=" << process_affinity_mask << "\n";
            if (applied_affinity_mask && previous_affinity_mask != 0) {
                SetProcessAffinityMask(GetCurrentProcess(), previous_affinity_mask);
            }
        }
#endif
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
