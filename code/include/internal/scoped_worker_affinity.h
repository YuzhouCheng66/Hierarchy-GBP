#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
#include <atomic>
#include <cstdint>

namespace slam {
class ScopedWorkerAffinity {
public:
    inline static std::atomic<int> failures{0}, max_team{0};
    static std::uintptr_t availableMask(int workers) {
#ifdef _WIN32
        DWORD_PTR allowed=0,system=0;
        if(!GetProcessAffinityMask(GetCurrentProcess(),&allowed,&system)) return 0;
        int count=0;
        for(auto m=allowed;m;m&=m-1) ++count;
        return count>=workers ? allowed : 0;
#else
        return 0;
#endif
    }
    static void observeTeam(int count) {
        int old=max_team.load(std::memory_order_relaxed);
        while(old<count && !max_team.compare_exchange_weak(old,count,std::memory_order_relaxed)) {}
    }
    ScopedWorkerAffinity(std::uintptr_t allowed,int worker) noexcept {
#ifdef _WIN32
        if(!allowed) return;
        for(int i=0;i<worker && allowed;++i) allowed&=allowed-1;
        if(!allowed) return;
        previous_=SetThreadAffinityMask(GetCurrentThread(),allowed & (~allowed+1));
        if(!previous_) failures.fetch_add(1,std::memory_order_relaxed);
#endif
    }
    ~ScopedWorkerAffinity() {
#ifdef _WIN32
        if(previous_ && !SetThreadAffinityMask(GetCurrentThread(),previous_))
            failures.fetch_add(1,std::memory_order_relaxed);
#endif
    }
    ScopedWorkerAffinity(const ScopedWorkerAffinity&)=delete;
    ScopedWorkerAffinity& operator=(const ScopedWorkerAffinity&)=delete;
private:
#ifdef _WIN32
    DWORD_PTR previous_=0;
#endif
};
} // namespace slam
