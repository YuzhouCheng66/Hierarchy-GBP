#include "se2_host_overlap.h"
#include <stdexcept>
#include <utility>
#ifdef HGBP_OVERLAP_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace {
thread_local std::function<void()> pending;
}

void setSE2HostOverlap(std::function<void()> work) {
    if(pending) throw std::runtime_error("Nested SE2 host overlap");
    pending=std::move(work);
}

void runSE2HostOverlap() {
    auto work=std::move(pending);
    pending={};
    if(work) {
#ifdef HGBP_OVERLAP_NVTX
        nvtxRangePushA("conditioned_basis_overlap");
        try { work(); } catch(...) { nvtxRangePop(); throw; }
        nvtxRangePop();
#else
        work();
#endif
    }
}

void clearSE2HostOverlap() { pending={}; }
