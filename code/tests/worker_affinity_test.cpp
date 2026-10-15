#include "internal/scoped_worker_affinity.h"
#include <omp.h>
#include <iostream>

int main() {
    int errors=0;
    for(int count : {1,2,4,8,16}) {
        const auto mask=slam::ScopedWorkerAffinity::availableMask(count);
        #pragma omp parallel num_threads(count) reduction(+:errors)
        {
            if(omp_get_thread_num()==0) slam::ScopedWorkerAffinity::observeTeam(omp_get_num_threads());
#ifdef _WIN32
            GROUP_AFFINITY before{},inside{},after{};
            GetThreadGroupAffinity(GetCurrentThread(),&before);
            {
                slam::ScopedWorkerAffinity pin(mask,omp_get_thread_num());
                GetThreadGroupAffinity(GetCurrentThread(),&inside);
                if(mask && ((inside.Mask & (inside.Mask-1)) || !(inside.Mask & mask))) ++errors;
            }
            GetThreadGroupAffinity(GetCurrentThread(),&after);
            if(before.Mask!=after.Mask || before.Group!=after.Group) ++errors;
#endif
        }
    }
    errors+=slam::ScopedWorkerAffinity::failures.load();
    if(slam::ScopedWorkerAffinity::max_team!=16) ++errors;
    std::cout << "affinity pin/restore and team audit errors=" << errors << '\n';
    return errors ? 1 : 0;
}
