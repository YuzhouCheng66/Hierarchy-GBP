#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "internal/se2_solver_impl.h"

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

std::string preciseDouble(double value) {
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
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
        throw std::runtime_error("requested --process-affinity-mask does not overlap system affinity mask");
    }
    previous_process_mask = process_mask;
    if (!SetProcessAffinityMask(GetCurrentProcess(), clamped_mask)) {
        throw std::runtime_error("SetProcessAffinityMask failed");
    }
    return true;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string problem_file;
        std::string out_json = "synthetic_se2_mg_svd_results.json";
        int num_outer = 20;
        int inner_cycles = 3;
        int pre_sweeps = 100;
        int group_size = 20;
        int r_reduced = 4;
        int sync_num_threads = 16;
        constexpr int sync_schedule_chunk = 0;
        constexpr bool profile_sync_thread_utilization = false;
        constexpr bool enable_singlecore_fastsync = true;
        double coarse_scale = 1.0;
        double packed_jitter = 1e-10;
        int fixed_eta_after_sweeps = -1;
        constexpr int final_coarse_polish_passes = 0;
        std::string process_affinity_mask;
        bool write_poses = false;
        slam::BasisBuildConfig basis_build_config;
        slam::RobustLossConfig robust_loss_config;
        robust_loss_config.huber_delta = 5.0;
        constexpr gbp::FactorGraph::SyncScheduleKind sync_schedule =
            gbp::FactorGraph::SyncScheduleKind::Static;

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
            } else if (arg == "--partial-residual-tol" && i + 1 < argc) {
                basis_build_config.partial_residual_tol = parseDouble(arg, argv[++i]);
            } else if (arg == "--basis-rebuild-period" && i + 1 < argc) {
                basis_build_config.basis_rebuild_period = parseInt(arg, argv[++i]);
            } else if (arg == "--basis-rebuild-warmup-outers" && i + 1 < argc) {
                basis_build_config.basis_rebuild_warmup_outers =
                    parseInt(arg, argv[++i]);
            } else if (arg == "--coarse-scale" && i + 1 < argc) {
                coarse_scale = parseDouble(arg, argv[++i]);
            } else if (arg == "--huber-delta" && i + 1 < argc) {
                robust_loss_config.huber_delta = parseDouble(arg, argv[++i]);
            } else if (arg == "--jitter" && i + 1 < argc) {
                packed_jitter = parseDouble(arg, argv[++i]);
            } else if (arg == "--fixed-eta-after-sweeps" && i + 1 < argc) {
                fixed_eta_after_sweeps = parseInt(arg, argv[++i]);
            } else if (arg == "--process-affinity-mask" && i + 1 < argc) {
                process_affinity_mask = argv[++i];
            } else if (arg == "--write-poses") {
                write_poses = true;
            } else if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: se2_benchmark --problem-file <path> [--out-json <path>] "
                             "[--num-outer N] [--inner-cycles C] [--pre-sweeps K] "
                             "[--group-size G] [--r-reduced R] [--threads T] "
                             "[--partial-residual-tol X] "
                             "[--basis-rebuild-period N] "
                             "[--basis-rebuild-warmup-outers N] "
                             "[--process-affinity-mask MASK] "
                             "[--coarse-scale X] "
                             "[--huber-delta X] "
                             "[--jitter X] "
                             "[--fixed-eta-after-sweeps K] "
                             "[--write-poses]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + arg);
            }
        }

        if (problem_file.empty()) {
            throw std::runtime_error("--problem-file is required");
        }

        if (num_outer < 1 || inner_cycles < 1 || pre_sweeps < 0 ||
            group_size < 1 || r_reduced < 1 || sync_num_threads < 1 ||
            basis_build_config.basis_rebuild_period < 1 ||
            basis_build_config.basis_rebuild_warmup_outers < 0 ||
            !(basis_build_config.partial_residual_tol > 0.0) ||
            !(coarse_scale > 0.0) || robust_loss_config.huber_delta < 0.0 ||
            !(packed_jitter > 0.0) || fixed_eta_after_sweeps < -1) {
            throw std::runtime_error("invalid benchmark configuration");
        }

        setEnv("GBP_FASTJITTER_ABS_JITTER", preciseDouble(packed_jitter));
        setEnv(
            "GBP_SE2_FIXED_ETA_AFTER_SWEEP",
            std::to_string(fixed_eta_after_sweeps)
        );
        setEnv("GBP_SE2_FIXED_ETA_SERIAL_DELTA", "1");
        setEnv("GBP_SE2_FIXED_ETA_PARALLEL_DELTA", "0");

#ifdef _WIN32
        DWORD_PTR previous_affinity_mask = 0;
        const bool applied_affinity_mask = applyProcessAffinityMask(process_affinity_mask, previous_affinity_mask);
#endif

        slam::SyntheticSE2Problem problem = slam::loadSyntheticSE2Problem(problem_file);
        slam::ExperimentResults results = slam::runSyntheticSE2Experiment(
            problem,
            num_outer,
            inner_cycles,
            pre_sweeps,
            group_size,
            r_reduced,
            sync_num_threads,
            sync_schedule,
            sync_schedule_chunk,
            profile_sync_thread_utilization,
            enable_singlecore_fastsync,
            coarse_scale,
            basis_build_config,
            robust_loss_config,
            final_coarse_polish_passes
        );
        slam::writeExperimentResultsJson(
            results,
            out_json,
            num_outer,
            inner_cycles,
            pre_sweeps,
            group_size,
            r_reduced,
            sync_num_threads,
            sync_schedule,
            sync_schedule_chunk,
            profile_sync_thread_utilization,
            enable_singlecore_fastsync,
            basis_build_config,
            robust_loss_config,
            final_coarse_polish_passes,
            packed_jitter,
            fixed_eta_after_sweeps
        );
        if (write_poses) {
            std::string pose_json = out_json;
            const std::size_t dot = pose_json.rfind('.');
            if (dot != std::string::npos) {
                pose_json.insert(dot, "_poses");
            } else {
                pose_json += "_poses.json";
            }
            slam::writeExperimentPoseHistoryJson(results, pose_json);
        }

        std::cout << "wrote " << out_json << "\n";
        if (robust_loss_config.huber_delta > 0.0) {
            std::cout << "robust_huber_delta=" << robust_loss_config.huber_delta << "\n";
        }
#ifdef _WIN32
        if (!process_affinity_mask.empty()) {
            std::cout << "process_affinity_mask=" << process_affinity_mask << "\n";
            if (applied_affinity_mask && previous_affinity_mask != 0) {
                SetProcessAffinityMask(GetCurrentProcess(), previous_affinity_mask);
            }
        }
#endif
        if (results.direct_history.size() > 1) {
            std::cout << "direct_final_objective=" << results.direct_history.back().nonlinear_objective << "\n";
        }
        if (!results.mg_history.empty()) {
            std::cout << "mg_final_objective=" << results.mg_history.back().nonlinear_objective << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
