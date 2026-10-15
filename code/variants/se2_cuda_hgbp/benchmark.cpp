#include <chrono>
#include <cstdlib>
#include <iostream>
#include "se2_cuda.h"
#define main referenceBenchmarkMain
#include "../../apps/se2_benchmark.cpp"
#undef main

int main(int argc, char** argv) {
    try {
        const char* backend = std::getenv("HGBP_CUDA_BACKEND");
        if (!backend || std::string(backend) != "cpu") {
            const auto begin = std::chrono::steady_clock::now();
            initializeSE2Cuda();
            std::cout << "CUDA_CONTEXT_MS " << std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now()-begin).count() << '\n';
        }
        return referenceBenchmarkMain(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "CUDA benchmark failed: " << e.what() << '\n';
        return 2;
    }
}
