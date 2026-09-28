struct SegmentGather {
    int var=0,begin=0,end=0,width=2,lane=0;
    unsigned mask=0;
    IncomingSlots slots{};
    __device__ SegmentGather(Device d,int id) {
        if(id>=d.s) return;
        var=d.owner[id]; width=d.width[id]; lane=id&(width-1);
        begin=d.offsets[var]; end=d.offsets[var+1];
        mask=(0xffffffffu>>(32-width))<<((threadIdx.x&31)&~(width-1));
        #pragma unroll
        for(int k=0;k<4;++k) slots.slot[k]=begin+k<end?d.slots[begin+k]:0;
    }
    __device__ __forceinline__ void get(Device d,bool initial,bool lambda,
        const double* eta,const double* lam,double* incoming) const {
        const double v0=gatherComponent(d,var,lane,begin,end,initial,lambda,eta,lam,slots);
        const double v1=gatherComponent(d,var,lane+width,begin,end,initial,lambda,eta,lam,slots);
        const double v2=gatherComponent(d,var,lane+2*width,begin,end,initial,lambda,eta,lam,slots);
        const double v3=gatherComponent(d,var,lane+3*width,begin,end,initial,lambda,eta,lam,slots);
        const double v4=gatherComponent(d,var,lane+4*width,begin,end,initial,lambda,eta,lam,slots);
        #pragma unroll
        for(int k=0;k<9;++k) {
            const double component=k<width?v0:k<2*width?v1:k<3*width?v2:k<4*width?v3:v4;
            incoming[k]=__shfl_sync(mask,component,k&(width-1),width);
        }
    }
};

template<bool Eta,bool Test> __device__ __forceinline__ double segmentMessage(
    Device d,int id,CachedMessage& cached,const SegmentGather& gather,
    int ei,int li,bool initial,bool lambda,double jitter) {
    double maximum=0;
    if(id<d.s) {
        double incoming[9];
        gather.get(d,initial,lambda,d.eta[ei],d.lam[li],incoming);
        if(d.source[id]>=0) {
            cachedMessage<true>(d,id,cached,Eta,Test,d.eta[ei],d.eta[ei^1],
                d.lam[li],d.lam[Eta?li:li^1],jitter,incoming);
            if constexpr(Test) maximum=d.residual[id];
        }
    }
    return maximum;
}

// Messages remain globally synchronous. Only bookkeeping is grouped between
// unchanged precision-check points; no stale halo or skipped node is allowed.
__global__ void sweepsSegmentKernel(Device d,int sweeps,double jitter) {
    auto grid=cooperative_groups::this_grid();
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    const int stride=blockDim.x*gridDim.x;
    __shared__ SE2CudaState control;
    if(threadIdx.x==0) control=*d.state;
    __syncthreads();
    const SegmentGather gather(d,id);
    CachedMessage cached;
    const bool active=id<d.s && d.source[id]>=0;
    if(active) {
        #pragma unroll
        for(int j=0;j<27;++j) cached.coef[j]=d.coef[j*d.s+id];
        #pragma unroll
        for(int j=0;j<9;++j) cached.map[j]=d.map[j*d.s+id];
        cached.valid=d.valid[id]!=0;
    }
    int ei=d.indices[0],li=d.indices[1],completed=0;
    while(completed<sweeps) {
        const int count=min(sweeps-completed,max(0,control.next_check-control.sweep-1));
        const bool eta=control.frozen!=0;
        const bool map_ready=control.maps_valid!=0;
        bool checked=false;
        if(eta) {
            for(int step=0;step<count;++step) {
                segmentMessage<true,false>(d,id,cached,gather,ei,li,completed+step==0,
                    step==0 && !map_ready,jitter);
                grid.sync();
                ei^=1;
            }
        } else {
            for(int step=0;step<count;++step) {
                segmentMessage<false,false>(d,id,cached,gather,ei,li,completed+step==0,true,jitter);
                grid.sync();
                ei^=1; li^=1;
            }
        }
        if(threadIdx.x==0 && count) {
            control.sweep+=count;
            if(eta) control.eta+=count; else control.full+=count;
            control.maps_valid=eta;
        }
        completed+=count;
        __syncthreads();
        if(completed<sweeps) {
            double maximum=segmentMessage<false,true>(d,id,cached,gather,ei,li,completed==0,true,jitter);
            for(int offset=16;offset>0;offset/=2)
                maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
            if(threadIdx.x==0) d.block_residual[blockIdx.x]=maximum;
            grid.sync();
            double residual=0;
#ifdef HGBP_SINGLE_CHECK
            if(blockIdx.x==0) {
#endif
            for(int p=threadIdx.x;p<gridDim.x;p+=blockDim.x)
                residual=fmax(residual,d.block_residual[p]);
            for(int offset=16;offset>0;offset/=2)
                residual=fmax(residual,__shfl_down_sync(0xffffffffu,residual,offset));
#ifdef HGBP_SINGLE_CHECK
                if(threadIdx.x==0) {
                    d.block_residual[gridDim.x]=sqrt(residual);
                    d.block_residual[gridDim.x+1]=d.state->failed_slot;
                }
            }
            // This replaces the end-of-check global barrier. These two slots
            // are not overwritten until all CTAs reach the next check barrier.
            grid.sync();
            residual=d.block_residual[gridDim.x];
#endif
            if(threadIdx.x==0) {
                auto& s=control;
                ++s.sweep; ++s.full; s.maps_valid=0;
#ifdef HGBP_SINGLE_CHECK
                s.residual=residual;
                s.failed_slot=static_cast<int>(d.block_residual[gridDim.x+1]);
#else
                s.residual=sqrt(residual); s.last_check=s.sweep; ++s.checks;
#endif
#ifdef HGBP_SINGLE_CHECK
                s.last_check=s.sweep; ++s.checks;
#endif
                if(isfinite(s.residual) && s.residual<=1e-8) {
                    ++s.stable_checks;
                    if(!s.frozen && s.stable_checks>=3) {
                        s.frozen=1; ++s.freezes; if(s.first_freeze<0) s.first_freeze=s.sweep;
                    }
                } else {
                    if(s.frozen) ++s.thaws;
                    s.frozen=0; s.stable_checks=0;
                }
                s.interval=s.frozen?25:s.stable_checks>0?5:min(2*s.interval,25);
                s.next_check=s.sweep+s.interval;
            }
            ei^=1; li^=1; ++completed;
            checked=true;
        }
        // A global boundary protects residual readers and the failure snapshot
        // before any partition can start writing the next segment's flags.
        if(threadIdx.x==0
#ifdef HGBP_SINGLE_CHECK
            && !checked
#endif
        ) control.failed_slot=d.state->failed_slot;
        __syncthreads();
        const bool failed=control.failed_slot>=0;
#ifndef HGBP_SINGLE_CHECK
        grid.sync();
#endif
        if(failed) break;
    }
    if(active) {
        #pragma unroll
        for(int j=0;j<9;++j) d.map[j*d.s+id]=cached.map[j];
        d.valid[id]=cached.valid;
    }
    if(control.failed_slot<0)
        for(int var=id;var<d.n;var+=stride)
            gatherOwnedBelief(d,d.belief,d.belief,d.eta[ei],d.lam[li],var,false);
    grid.sync();
    if(id==0) {
        control.builds=d.state->builds; control.failed_slot=d.state->failed_slot;
        *d.state=control; d.indices[0]=ei; d.indices[1]=li; d.indices[2]=0;
    }
}
