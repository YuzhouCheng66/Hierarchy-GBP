#include "se2_cuda.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    try {
        _putenv_s("HGBP_RESIDENT_THREADS","32");
        _putenv_s("HGBP_RESIDENT_EXPORT_ALL","1");
        initializeSE2Cuda();
        constexpr int n=64,m=n/2;
        std::vector<SE2CudaFactor> factors(m);
        std::vector<int> offsets(n+1),slots(n);
        std::vector<double> base_eta(3*n,0.),base_lam(6*n,0.),belief_eta(3*n,0.),belief_lam(6*n,0.);
        std::vector<double> message_eta(6*m,77.),message_lam(12*m,77.),map0(9*m,77.),map1(9*m,77.);
        std::vector<double> scales(12*m,1.),residuals(m,77.);
        std::vector<std::uint8_t> valid(m,77);
        for(int i=0;i<n;++i) {
            offsets[i]=i; slots[i]=i;
            for(int k:{0,3,5}) { base_lam[6*i+k]=1.; belief_lam[6*i+k]=2.; }
        }
        offsets[n]=n;
        for(int f=0;f<m;++f) {
            factors[f].var0_id=2*f; factors[f].var1_id=2*f+1;
            for(int k:{0,3,5}) factors[f].diag0_lam6[k]=factors[f].diag1_lam6[k]=1.;
        }
        SE2CudaView v{};
        v.n=n; v.m=m; v.factors=factors.data(); v.offsets=offsets.data(); v.slots=slots.data();
        v.base_eta=base_eta.data(); v.base_lam=base_lam.data();
        v.belief_eta=belief_eta.data(); v.belief_lam=belief_lam.data();
        v.message_eta=message_eta.data(); v.message_lam=message_lam.data();
        v.map0=map0.data(); v.map1=map1.data(); v.map_valid=valid.data();
        v.scales=scales.data(); v.residuals=residuals.data();
        auto reset=[&](int next) {
            v.state={}; v.state.failed_slot=-1; v.state.first_freeze=-1;
            v.state.interval=1; v.state.next_check=next;
        };
        auto require=[](bool ok,const char* why) { if(!ok) throw std::runtime_error(why); };
        // Fail during an ordinary segment and during an immediate check.
        for(int next:{200,1}) {
            reset(next);
            for(int k:{0,3,5}) belief_lam[k]=-1.;
            const auto before=belief_lam;
            runSE2Cuda(v,100,1e-7);
            require(v.state.failed_slot>=0,"Expected a rejected non-SPD cavity");
            require(belief_lam==before,"Failure committed beliefs");
            require(std::all_of(message_lam.begin(),message_lam.end(),[](double x){return x==77.;}),
                "Failure committed messages");
        }
        reset(1);
        for(int k:{0,3,5}) belief_lam[k]=2.;
        runSE2Cuda(v,100,1e-7);
        require(v.state.failed_slot<0 && v.state.sweep==100,"Recovery batch failed");
        require(v.state.full+v.state.eta==100,"Recovery work accounting changed");
        require(v.state.freezes==1 && v.state.frozen,"Expected the trivial variance fixed point");
        for(int i=0;i<n;++i) for(int k=0;k<6;++k)
            require(message_lam[6*i+k]==((k==0||k==3||k==5)?1.:0.),"Wrong recovered message");
        v.state.full=v.state.eta=v.state.builds=0;
        runSE2Cuda(v,17,1e-7);
        require(v.state.failed_slot<0 && v.state.sweep==117,"Cycle replay failed");
        require(v.state.full+v.state.eta==17,"Cycle work accounting changed");
        std::cout<<"non-SPD rollback, recovery, and cycle replay passed\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
