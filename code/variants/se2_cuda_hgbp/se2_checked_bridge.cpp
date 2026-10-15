#include "se2_cuda.h"
#include "internal/se2_residual.h"
#include "internal/se2_precision.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace slam {
void cpuReferenceSE2Sweeps(SyntheticSE2PackedResidualWorkspace&, int, double&, double&, int);
namespace {
double relativeError(const std::vector<double>& a, const std::vector<double>& b) {
    double error=0, scale=0;
    for (size_t i=0;i<a.size();++i) {
        if(!std::isfinite(a[i]) || !std::isfinite(b[i])) return INFINITY;
        error=std::hypot(error,a[i]-b[i]); scale=std::hypot(scale,b[i]);
    }
    return error/std::max(1.,scale);
}
bool enabled(const char* key) {
    const char* value=std::getenv(key); return value && std::string(value)=="1";
}
struct Stats {
    int calls=0, fallback=0;
    double wall=0, upload=0, kernel=0, download=0, shadow_error=0, shadow_mean_error=0;
    ~Stats() {
        std::cout << std::setprecision(17) << "CUDA_LOWER_SUMMARY {\"calls\":" << calls
            << ",\"fallback_batches\":" << fallback << ",\"wall_ms\":" << wall
            << ",\"upload_ms\":" << upload << ",\"kernel_ms\":" << kernel
            << ",\"download_ms\":" << download << ",\"shadow_max_relative\":" << shadow_error
            << ",\"shadow_mean_max_relative\":" << shadow_mean_error << "}\n";
    }
};
}

void synchronousIterationsSyntheticSE2PackedResidualWorkspace(
    SyntheticSE2PackedResidualWorkspace& w,int sweeps,double& factor_time,double& variable_time,int threads) {
    const char* backend=std::getenv("HGBP_CUDA_BACKEND");
    if (backend && std::string(backend)=="cpu") {
        cpuReferenceSE2Sweeps(w,sweeps,factor_time,variable_time,threads); return;
    }
    if(sweeps>1 && enabled("HGBP_CUDA_SHADOW_SINGLE")) {
        for(int i=0;i<sweeps;++i)
            synchronousIterationsSyntheticSE2PackedResidualWorkspace(w,1,factor_time,variable_time,threads);
        return;
    }
    if(sweeps<=0) return;
    if(enabled("HGBP_CUDA_ALLOW_FALLBACK") || enabled("HGBP_RESIDUAL_CYCLES") ||
        enabled("HGBP_CYCLE_ENERGY") || w.lift_correction_to_messages)
        throw std::runtime_error("Resident experiment does not support replay fallback or message/RHS-changing correction policies");
    if(!w.adaptive_precision || threads!=16 || w.eta_relaxation!=1. || w.message_initialization!=2)
        throw std::runtime_error("Strict CUDA currently supports only the released SE2 16-thread policy");
    static Stats stats;
    ++stats.calls;
    SyntheticSE2PackedResidualWorkspace expected;
    if(enabled("HGBP_CUDA_SHADOW")) {
        expected=w; double f=0,v=0;
        cpuReferenceSE2Sweeps(expected,sweeps,f,v,threads);
    }
    const auto start=std::chrono::steady_clock::now();
    if(!w.precision_scales_ready) {
        w.precision_residuals.resize(w.binary_factors.size());
        w.precision_scales.resize(12*w.binary_factors.size());
        for(size_t f=0;f<w.binary_factors.size();++f) {
            buildSE2PrecisionScales(w.binary_factors[f].diag0_lam6,w.precision_scales.data()+12*f);
            buildSE2PrecisionScales(w.binary_factors[f].diag1_lam6,w.precision_scales.data()+12*f+6);
        }
        w.precision_scales_ready=true;
    }
    // Preserve prior/unary summation order. Only the binary incoming pass moves.
    std::vector<double> base_eta=w.prior_eta,base_lam=w.prior_lam6;
    for(int v=0;v<w.num_vars;++v) for(int j=w.unary_offsets[v];j<w.unary_offsets[v+1];++j) {
        const int u=w.unary_ids[j];
        for(int k=0;k<3;++k) base_eta[3*v+k]+=w.unary_msg_eta[3*u+k];
        for(int k=0;k<6;++k) base_lam[6*v+k]+=w.unary_msg_lam6[6*u+k];
    }
    static_assert(sizeof(SE2CudaFactor)==sizeof(SyntheticSE2PackedResidualBinaryFactor));
    static_assert(offsetof(SE2CudaFactor,cross01_lam9)==offsetof(SyntheticSE2PackedResidualBinaryFactor,cross01_lam9));
    SE2CudaView view{};
    view.n=w.num_vars; view.m=static_cast<int>(w.binary_factors.size());
    view.factors=reinterpret_cast<const SE2CudaFactor*>(w.binary_factors.data());
    view.offsets=w.binary_offsets.data(); view.slots=w.binary_slot_ids.data();
    view.base_eta=base_eta.data(); view.base_lam=base_lam.data();
    view.belief_eta=w.belief_eta.data(); view.belief_lam=w.belief_lam6.data();
    view.message_eta=w.binary_msg_eta.data(); view.message_lam=w.binary_msg_lam6.data();
    view.map0=w.fixed_eta_map0_lam9.data(); view.map1=w.fixed_eta_map1_lam9.data();
    view.map_valid=w.fixed_eta_map_valid.data(); view.scales=w.precision_scales.data();
    view.residuals=w.precision_residuals.data();
    view.state={w.sweeps_since_relinearize,w.precision_frozen,w.precision_stable_checks,
        w.precision_checks,w.precision_freezes,w.precision_thaws,w.first_precision_freeze_sweep,
        w.last_precision_check_sweep,w.precision_check_interval,w.next_precision_check_sweep,
        w.fixed_eta_maps_all_valid,-1,0,0,0,w.last_precision_residual};
    const char* jitter_text=std::getenv("GBP_FASTJITTER_ABS_JITTER");
    const auto timing=runSE2Cuda(view,sweeps,jitter_text?std::stod(jitter_text):1e-7);
    if(stats.calls==1) std::cout << "CUDA_LAUNCH blocks=" << timing.blocks << " threads=" << timing.threads
        << " registers=" << timing.registers << '\n';
    stats.upload+=timing.upload_ms; stats.kernel+=timing.kernel_ms; stats.download+=timing.download_ms;
    if(view.state.failed_slot>=0) {
        // The CUDA backend leaves host numerical state untouched on failure.
        // Replay the entire batch through the original LU-capable CPU kernel.
        ++stats.fallback;
        if(!enabled("HGBP_CUDA_ALLOW_FALLBACK"))
            throw std::runtime_error("CUDA non-SPD cavity at slot "+std::to_string(view.state.failed_slot));
        cpuReferenceSE2Sweeps(w,sweeps,factor_time,variable_time,threads);
    } else {
        const auto& s=view.state;
        w.sweeps_since_relinearize=s.sweep; w.precision_frozen=s.frozen!=0;
        w.precision_stable_checks=s.stable_checks; w.precision_checks=s.checks;
        w.precision_freezes=s.freezes; w.precision_thaws=s.thaws;
        w.first_precision_freeze_sweep=s.first_freeze; w.last_precision_check_sweep=s.last_check;
        w.precision_check_interval=s.interval; w.next_precision_check_sweep=s.next_check;
        w.fixed_eta_maps_all_valid=static_cast<std::uint8_t>(s.maps_valid);
        w.last_precision_residual=s.residual;
        w.full_precision_sweeps+=s.full; w.eta_only_sweeps+=s.eta;
        w.fixedeta_full_lambda_sweeps+=s.full; w.fixedeta_eta_only_sweeps+=s.eta;
        w.fixedeta_eta_map_builds+=s.builds;
        std::fill(w.mu_valid.begin(),w.mu_valid.end(),0);
        std::fill(w.fixed_lam_initialized.begin(),w.fixed_lam_initialized.end(),1);
        w.belief_eta_message_consistent=1;
        // Lower time includes transfers, packing, synchronization and allocation.
        // The combined executor does not pretend to split overlapping GPU phases.
        factor_time+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    }
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    stats.wall+=wall;
    if(enabled("HGBP_CUDA_SHADOW")) {
        const double error=std::max({relativeError(w.belief_eta,expected.belief_eta),
            relativeError(w.belief_lam6,expected.belief_lam6),relativeError(w.binary_msg_eta,expected.binary_msg_eta),
            relativeError(w.binary_msg_lam6,expected.binary_msg_lam6)});
        stats.shadow_error=std::max(stats.shadow_error,error);
        auto actual=w;
        Eigen::VectorXd gpu_mean,cpu_mean;
        stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(actual,gpu_mean,threads);
        stackedMeanVectorSyntheticSE2PackedResidualWorkspaceInto(expected,cpu_mean,threads);
        const double mean_error=(gpu_mean-cpu_mean).norm()/std::max(1.,cpu_mean.norm());
        stats.shadow_mean_error=std::max(stats.shadow_mean_error,mean_error);
        const bool policy=w.precision_checks==expected.precision_checks &&
            w.precision_freezes==expected.precision_freezes && w.precision_thaws==expected.precision_thaws &&
            w.first_precision_freeze_sweep==expected.first_precision_freeze_sweep &&
            w.full_precision_sweeps==expected.full_precision_sweeps && w.eta_only_sweeps==expected.eta_only_sweeps &&
            w.precision_stable_checks==expected.precision_stable_checks &&
            w.next_precision_check_sweep==expected.next_precision_check_sweep &&
            w.last_precision_check_sweep==expected.last_precision_check_sweep &&
            w.precision_check_interval==expected.precision_check_interval;
        std::cout << "CUDA_SHADOW " << stats.calls << " error=" << std::setprecision(17) << error
            << " mean_error=" << mean_error << " policy=" << policy << " full=" << view.state.full << " eta=" << view.state.eta
            << " gpu_state=" << w.precision_checks << ',' << w.precision_freezes << ',' << w.precision_thaws
            << ',' << w.first_precision_freeze_sweep << ',' << w.precision_stable_checks << ',' << w.next_precision_check_sweep
            << ',' << w.last_precision_residual
            << " cpu_state=" << expected.precision_checks << ',' << expected.precision_freezes << ',' << expected.precision_thaws
            << ',' << expected.first_precision_freeze_sweep << ',' << expected.precision_stable_checks << ',' << expected.next_precision_check_sweep
            << ',' << expected.last_precision_residual << '\n';
        if(!policy || !(error<1e-6) || !(mean_error<1e-6)) throw std::runtime_error("CPU/CUDA sweep equivalence gate failed");
    }
}
}
