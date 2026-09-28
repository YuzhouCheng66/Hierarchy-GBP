#include "se2_transfer_layout.h"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

void require(bool ok) { if(!ok) throw std::runtime_error("SE2 transfer layout test failed"); }
void checkRanges(size_t size,const std::vector<std::pair<size_t,size_t>>& ranges) {
    size_t last=0;
    for(auto range:ranges) {
        require(range.first%128==0 && range.first>=last);
        last=range.first+range.second;
        require(last<=size);
    }
}
int main() {
    for(int n:{1,2,31,32,33,989,1316,3500}) for(int m:{0,1,17,1217,2820,5453}) {
        SE2TransferLayout l(n,m);
        require(l.cycle_size<=l.factors && l.belief_eta+24*n<=l.cycle_size);
        checkRanges(l.input_size,{{l.state,sizeof(SE2CudaState)},{l.belief_eta,24*n},
            {l.factors,sizeof(SE2CudaFactor)*m},{l.base_eta,24*n},{l.base_lam,48*n},
            {l.belief_lam,48*n},{l.scales,96*m}});
        checkRanges(l.output_size,{{l.out_state,sizeof(SE2CudaState)},{l.out_eta,24*n},{l.out_lam,48*n}});
    }
    bool overflow=false;
    size_t cursor=std::numeric_limits<size_t>::max()-32;
    try { SE2TransferLayout::append(cursor,1,8); } catch(const std::overflow_error&) { overflow=true; }
    require(overflow);
    bool invalid=false;
    try { SE2TransferLayout l(0,2); } catch(const std::invalid_argument&) { invalid=true; }
    require(invalid);
    std::cout<<"48 packet layouts, overflow and invalid dimensions passed\n";
}
