#pragma once
#include "se2_cuda.h"
#include <cstddef>
#include <limits>
#include <stdexcept>

struct SE2TransferLayout {
    size_t input_size=0,output_size=0,cycle_size=0;
    size_t state=0,belief_eta=0,factors=0,base_eta=0,base_lam=0,belief_lam=0,scales=0;
    size_t out_state=0,out_eta=0,out_lam=0;
    static size_t append(size_t& cursor,size_t count,size_t element) {
        const auto limit=std::numeric_limits<size_t>::max();
        if(cursor>limit-127 || count>limit/element) throw std::overflow_error("SE2 transfer layout");
        cursor=(cursor+127)&~size_t(127);
        const auto start=cursor,bytes=count*element;
        if(bytes>limit-cursor) throw std::overflow_error("SE2 transfer layout");
        cursor+=bytes;
        return start;
    }
    SE2TransferLayout(int n,int m) {
        if(n<=0 || m<0) throw std::invalid_argument("SE2 transfer dimensions");
        const size_t nodes=n,edges=m;
        state=append(input_size,1,sizeof(SE2CudaState));
        belief_eta=append(input_size,3*nodes,sizeof(double));
        append(input_size,0,1); cycle_size=input_size;
        factors=append(input_size,edges,sizeof(SE2CudaFactor));
        base_eta=append(input_size,3*nodes,sizeof(double));
        base_lam=append(input_size,6*nodes,sizeof(double));
        belief_lam=append(input_size,6*nodes,sizeof(double));
        scales=append(input_size,12*edges,sizeof(double));
        out_state=append(output_size,1,sizeof(SE2CudaState));
        out_eta=append(output_size,3*nodes,sizeof(double));
        out_lam=append(output_size,6*nodes,sizeof(double));
    }
};
