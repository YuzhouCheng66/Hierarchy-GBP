#pragma once
#include <cstdint>

struct SE2CudaFactor {
    int var0_id, var1_id;
    double eta0[3], eta1[3], diag0_lam6[6], diag1_lam6[6], cross01_lam9[9];
};

struct SE2CudaState {
    int sweep, frozen, stable_checks, checks, freezes, thaws;
    int first_freeze, last_check, interval, next_check, maps_valid;
    int failed_slot;
    std::uint64_t full, eta, builds;
    double residual;
};

struct SE2CudaView {
    int n, m;
    const SE2CudaFactor* factors;
    const int* offsets;
    const int* slots;
    const double* base_eta;
    const double* base_lam;
    double* belief_eta;
    double* belief_lam;
    double* message_eta;
    double* message_lam;
    double* map0;
    double* map1;
    std::uint8_t* map_valid;
    const double* scales;
    double* residuals;
    SE2CudaState state;
};

struct SE2CudaTiming {
    double upload_ms, kernel_ms, download_ms;
    int blocks, threads, registers;
};

void initializeSE2Cuda();
SE2CudaTiming runSE2Cuda(SE2CudaView& view, int sweeps, double jitter);
