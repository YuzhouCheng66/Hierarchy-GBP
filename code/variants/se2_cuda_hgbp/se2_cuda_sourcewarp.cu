#include "se2_host_overlap.h"
#include "se2_cuda.h"
#ifdef HGBP_TOPOLOGY_PLAN
#include "se2_source_plan.h"
#endif
#ifdef HGBP_SOURCE_DEPENDENCY
#include "se2_dependency_plan.h"
#endif
#ifdef HGBP_DEVICE_CYCLE
#include "se2_cycle.h"
#endif
#ifdef HGBP_PACKED_TRANSPORT
#include "se2_transfer_layout.h"
#endif
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cooperative_groups.h>
#include <cuda/atomic>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <vector>

#if defined(HGBP_LOCAL_MESSAGES) && (!defined(HGBP_TOPOLOGY_PLAN) || defined(HGBP_PHASE_GRAPHS) || defined(HGBP_GRAPH_STEPS) || defined(HGBP_SCHUR_ROW_STREAM))
#error Local message storage requires the topology-planned persistent executor
#endif

namespace {
void check(cudaError_t error) {
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
template<class T> struct Buffer {
    T* p=nullptr; size_t capacity=0;
    T* pinned=nullptr; T* pending_dst=nullptr; size_t pending_size=0;
    ~Buffer() { if(p) cudaFree(p); if(pinned) cudaFreeHost(pinned); }
    void reserve(size_t n) {
        if(n<=capacity) return;
        if(p) check(cudaFree(p));
        if(pinned) { check(cudaFreeHost(pinned)); pinned=nullptr; }
        p=nullptr; capacity=0;
        check(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(T))); capacity=n;
    }
    void staging() {
        if(!pinned) check(cudaMallocHost(reinterpret_cast<void**>(&pinned),capacity*sizeof(T)));
    }
    void upload(const T* src,size_t n) {
        reserve(n); staging(); std::memcpy(pinned,src,n*sizeof(T));
        check(cudaMemcpyAsync(p,pinned,n*sizeof(T),cudaMemcpyHostToDevice));
    }
    void download(T* dst,size_t n) {
        staging(); pending_dst=dst; pending_size=n;
        check(cudaMemcpyAsync(pinned,p,n*sizeof(T),cudaMemcpyDeviceToHost));
    }
    void finish(bool commit=true) {
        if(pending_dst && commit) std::memcpy(pending_dst,pinned,pending_size*sizeof(T));
        pending_dst=nullptr; pending_size=0;
    }
};
struct Device {
    int n,m,s;
    int *neighbor,*offsets,*slots,*indices;
    double *coef,*base,*belief,*eta[2],*lam[2],*map,*scale,*residual,*block_residual;
    std::uint8_t* valid;
    SE2CudaState* state;
    double* belief_alt;
    int *source,*forward,*opposite,*order;
    unsigned long long* clocks;
    int *target,*pending;
    const int *owner,*width;
#ifdef HGBP_SOURCE_DEPENDENCY
    unsigned* arrival;
    const int *dependency_offsets,*dependency_blocks;
#endif
#ifdef HGBP_WARP_GRID
    unsigned *barrier_count,*barrier_epoch;
#endif
#ifdef HGBP_LOCAL_MESSAGES
    const std::uint8_t* external;
#endif
};

// Keep iteration-local addresses separate from the immutable kernel parameter.
struct LocalMessageView {
#ifdef HGBP_LOCAL_MESSAGES
#ifdef HGBP_SHARED_PTX
    unsigned local_eta_old,local_lam_old,local_eta_next,local_lam_next;
#else
    double *local_eta_old,*local_lam_old,*local_eta_next,*local_lam_next;
#endif
    bool publish_message;
#endif
};

#ifdef HGBP_SHARED_PTX
__device__ __forceinline__ double sharedMessageLoad(unsigned base,int index) {
    double value;
    const unsigned address=base+sizeof(double)*index;
    asm volatile("ld.shared.f64 %0, [%1];" : "=d"(value) : "r"(address) : "memory");
    return value;
}
__device__ __forceinline__ void sharedMessageStore(unsigned base,int index,double value) {
    const unsigned address=base+sizeof(double)*index;
    asm volatile("st.shared.f64 [%0], %1;" :: "r"(address),"d"(value) : "memory");
}
#endif

__device__ __forceinline__ void localSync() {
#ifdef HGBP_WARP_BLOCK
    __syncwarp();
#else
    __syncthreads();
#endif
}

#ifdef HGBP_SOURCE_DEPENDENCY
// Publish step t, then wait for producers of t and readers of t-1. Symmetric
// dependencies protect both double buffers without a global ordinary-step fence.
__device__ __forceinline__ void sourceDependencySync(Device d,unsigned epoch,bool global) {
    __syncthreads();
    if(threadIdx.x==0) {
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> flag(d.arrival[32*blockIdx.x]);
        flag.store(epoch,cuda::memory_order_release);
    }
    const int begin=global?0:d.dependency_offsets[blockIdx.x];
    const int end=global?gridDim.x:d.dependency_offsets[blockIdx.x+1];
    for(int p=begin+threadIdx.x;p<end;p+=blockDim.x) {
        const int b=global?p:d.dependency_blocks[p];
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> flag(d.arrival[32*b]);
        while(flag.load(cuda::memory_order_acquire)<epoch) {}
    }
    __syncthreads();
}
#endif

#ifdef HGBP_WARP_GRID
__device__ __forceinline__ void warpGridSync(Device d,unsigned epoch) {
    __syncwarp();
    if(threadIdx.x==0) {
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> count(*d.barrier_count);
        cuda::atomic_ref<unsigned,cuda::thread_scope_device> ready(*d.barrier_epoch);
        // The RMW chain acquires every warp's publication before releasing epoch.
        if(count.fetch_add(1,cuda::memory_order_acq_rel)==gridDim.x-1) {
            count.store(0,cuda::memory_order_relaxed);
            ready.store(epoch,cuda::memory_order_release);
        } else {
            while(ready.load(cuda::memory_order_acquire)<epoch) {}
        }
    }
    __syncwarp();
}
#endif

template<int Components> __device__ __forceinline__ int messageIndex(int component,int slot,int stride) {
#ifdef HGBP_MESSAGE_INTERLEAVED
    return Components*slot+component;
#else
    return component*stride+slot;
#endif
}

template<int Components> __device__ __forceinline__ double readOldMessage(
    Device d,const double* global,int component,int slot,LocalMessageView view={}) {
#ifdef HGBP_LOCAL_MESSAGES
    if((slot>>5)==blockIdx.x) {
#ifdef HGBP_SHARED_PTX
        return sharedMessageLoad(Components==3?view.local_eta_old:view.local_lam_old,32*component+(slot&31));
#else
        const double* local=Components==3?view.local_eta_old:view.local_lam_old;
        return local[32*component+(slot&31)];
#endif
    }
#endif
    return global[messageIndex<Components>(component,slot,d.s)];
}
template<int Components> __device__ __forceinline__ void writeNextMessage(
    Device d,double* global,int component,int slot,double value,LocalMessageView view={}) {
#ifdef HGBP_LOCAL_MESSAGES
#ifdef HGBP_SHARED_PTX
    sharedMessageStore(Components==3?view.local_eta_next:view.local_lam_next,32*component+(slot&31),value);
#else
    double* local=Components==3?view.local_eta_next:view.local_lam_next;
    local[32*component+(slot&31)]=value;
#endif
    if(view.publish_message)
#endif
        global[messageIndex<Components>(component,slot,d.s)]=value;
}

template<bool Enabled> __device__ __forceinline__ void clockMark(Device d,unsigned long long& tick,int stage) {
    if constexpr(Enabled) if(threadIdx.x==0) {
        const auto now=clock64(); d.clocks[blockIdx.x*5+stage]+=now-tick; tick=now;
    }
}

__global__ void pack(SE2CudaView raw,Device d) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<d.s) {
        const int original=d.source[id];
        if(original>=0) {
        const auto& f=raw.factors[original/2];
        const int side=original&1;
        d.neighbor[id]=side?f.var0_id:f.var1_id;
        d.target[id]=side?f.var1_id:f.var0_id;
        const double* out_lam=side?f.diag1_lam6:f.diag0_lam6;
        const double* in_lam=side?f.diag0_lam6:f.diag1_lam6;
        const double* out_eta=side?f.eta1:f.eta0;
        const double* in_eta=side?f.eta0:f.eta1;
        #pragma unroll
        for(int k=0;k<6;++k) {
            d.coef[k*d.s+id]=out_lam[k];
            d.coef[(6+k)*d.s+id]=in_lam[k];
            d.lam[0][messageIndex<6>(k,id,d.s)]=out_lam[k];
            d.scale[k*d.s+id]=raw.scales[6*original+k];
        }
        #pragma unroll
        for(int k=0;k<3;++k) {
            d.coef[(12+k)*d.s+id]=out_eta[k];
            d.coef[(15+k)*d.s+id]=in_eta[k];
            d.eta[0][messageIndex<3>(k,id,d.s)]=out_eta[k];
        }
        #pragma unroll
        for(int r=0;r<3;++r) for(int c=0;c<3;++c) {
            d.coef[(18+3*r+c)*d.s+id]=f.cross01_lam9[side?c+3*r:r+3*c];
            d.map[(3*r+c)*d.s+id]=0.;
        }
        d.valid[id]=0;
        d.residual[id]=0.;
        } else { d.valid[id]=0; d.residual[id]=0.; }
    }
    if(id<d.n) {
        #pragma unroll
        for(int k=0;k<3;++k) {
            d.base[k*d.n+id]=raw.base_eta[3*id+k];
            d.belief[k*d.n+id]=raw.belief_eta[3*id+k];
        }
        #pragma unroll
        for(int k=0;k<6;++k) {
            d.base[(3+k)*d.n+id]=raw.base_lam[6*id+k];
            d.belief[(3+k)*d.n+id]=raw.belief_lam[6*id+k];
        }
    }
}

__global__ void initializePending(Device d) {
    const int var=blockIdx.x*blockDim.x+threadIdx.x;
    if(var<d.n) d.pending[var]=d.offsets[var+1]-d.offsets[var];
}

__global__ void patchBeliefEta(SE2CudaView raw,Device d) {
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    double* belief=d.indices[2]?d.belief_alt:d.belief;
    if(id<d.n) for(int k=0;k<3;++k) belief[k*d.n+id]=raw.belief_eta[3*id+k];
}

// MSVC 19.44 /fp:fast contracts these dot products left-to-right.
// Explicit CUDA intrinsics prevent NVCC choosing a different unfused product.
__device__ __forceinline__ double dot3Cpu(double a,double x,double b,double y,double c,double z) {
    return __fma_rn(c,z,__fma_rn(b,y,__dmul_rn(a,x)));
}
__device__ __forceinline__ bool adjugate(const double a[6],double c[9],double& invdet,bool map_build) {
    c[0]=__fma_rn(a[3],a[5],-__dmul_rn(a[4],a[4]));
    c[1]=__fma_rn(a[2],a[4],-__dmul_rn(a[1],a[5]));
    c[2]=__fma_rn(a[1],a[4],-__dmul_rn(a[2],a[3]));
    c[4]=__fma_rn(a[0],a[5],-__dmul_rn(a[2],a[2]));
    c[5]=__fma_rn(a[1],a[2],-__dmul_rn(a[0],a[4]));
    c[8]=__fma_rn(a[0],a[3],-__dmul_rn(a[1],a[1]));
    c[3]=c[1]; c[6]=c[2]; c[7]=c[5];
    const double det=map_build
        ? __fma_rn(a[2],c[2],__fma_rn(a[0],c[0],__dmul_rn(a[1],c[1])))
        : dot3Cpu(a[0],c[0],a[1],c[1],a[2],c[2]);
    if(!(det>0.) || !isfinite(det)) return false;
    invdet=1./det; return true;
}

struct CachedMessage {
    double coef[27],map[9];
    bool valid;
};
#ifdef HGBP_SCHUR_ROW_STREAM
#include "se2_schur_rowstream.cuh"
#endif
template<bool Mapped=false> __device__ __forceinline__ void cachedMessage(Device d,int id,CachedMessage& cached,bool eta_only,bool precision_check,
    const double* old_eta,double* new_eta,const double* old_lam,double* new_lam,double jitter,const double* incoming=nullptr,LocalMessageView view={}) {
    const int other=Mapped?d.opposite[id]:(id^1), var=d.neighbor[id], stride=d.s;
    double x[3], b[9], t[9];
    #pragma unroll
    for(int k=0;k<3;++k) {
        const double factor_eta=cached.coef[15+k];
        const double belief=incoming?incoming[k]:d.belief[k*d.n+var],old=readOldMessage<3>(d,old_eta,k,other,view);
        x[k]=eta_only?(factor_eta+belief)-old:factor_eta+(belief-old);
    }
    const bool build_map=eta_only && !cached.valid;
#ifdef HGBP_SCHUR_ROW_STREAM
    static_assert(Mapped,"Row-stream helper requires the source-slot mapping");
    if(!eta_only) {
        fullMessageRowStream(d,id,cached,precision_check,x,new_eta,old_lam,new_lam,jitter,incoming);
        return;
    }
#endif
    if(!eta_only || build_map) {
        double a[6],c[9],invdet;
        #pragma unroll
        for(int k=0;k<6;++k) {
            a[k]=cached.coef[6+k]+((incoming?incoming[3+k]:d.belief[(3+k)*d.n+var])-readOldMessage<6>(d,old_lam,k,other,view));
            if(k==0 || k==3 || k==5) a[k]+=jitter;
        }
        if(!adjugate(a,c,invdet,eta_only)) {
            atomicCAS(&d.state->failed_slot,-1,id);
#ifdef HGBP_SEGMENT_EXECUTOR
            // Failed batches are never committed, but keep both banks defined
            // while all partitions finish their fixed-length segment safely.
            #pragma unroll
            for(int k=0;k<3;++k) new_eta[messageIndex<3>(k,id,stride)]=old_eta[messageIndex<3>(k,id,stride)];
            if(!eta_only) {
                #pragma unroll
                for(int k=0;k<6;++k) new_lam[messageIndex<6>(k,id,stride)]=old_lam[messageIndex<6>(k,id,stride)];
            }
            if(precision_check) d.residual[id]=CUDART_INF;
            cached.valid=0;
#endif
            return;
        }
        #pragma unroll
        for(int k=0;k<9;++k) {
            b[k]=cached.coef[18+k];
            if(eta_only) c[k]*=invdet;
        }
        #pragma unroll
        for(int r=0;r<3;++r) for(int k=0;k<3;++k)
            t[3*r+k]=dot3Cpu(b[3*r],c[k],b[3*r+1],c[3+k],b[3*r+2],c[6+k]);
        // The release CPU's noinline precision-check helper rounds the middle
        // product first for t10/t11 on side 1 (MSVC 19.44 disassembly).
        if(precision_check && ((Mapped?d.source[id]:id)&1)) {
            #pragma unroll
            for(int k=0;k<2;++k)
                t[3+k]=__fma_rn(b[5],c[6+k],__fma_rn(b[3],c[k],__dmul_rn(b[4],c[3+k])));
        }
        if(eta_only) {
            #pragma unroll
            for(int k=0;k<9;++k) cached.map[k]=t[k];
            cached.valid=1;
            if(!((Mapped?d.source[id]:id)&1)) atomicAdd(reinterpret_cast<unsigned long long*>(&d.state->builds),1ULL);
        } else {
            #pragma unroll
            for(int r=0;r<3;++r)
                writeNextMessage<3>(d,new_eta,r,id,__fma_rn(-invdet,dot3Cpu(t[3*r],x[0],t[3*r+1],x[1],t[3*r+2],x[2]),cached.coef[12+r]),view);
            constexpr int row[6]={0,1,2,1,2,2}, col[6]={0,0,0,1,1,2};
            double delta2=0,next2=0,old2=0;
            #pragma unroll
            for(int k=0;k<6;++k) {
                const int r=row[k],s=col[k];
                const double value=__fma_rn(-invdet,dot3Cpu(t[3*r],b[3*s],t[3*r+1],b[3*s+1],t[3*r+2],b[3*s+2]),cached.coef[k]);
                writeNextMessage<6>(d,new_lam,k,id,value,view);
                if(precision_check) {
                    const double scale=d.scale[k*stride+id];
                    const double aa=value*scale,bb=readOldMessage<6>(d,old_lam,k,id,view)*scale;
                    const double weight=r==s?1.:2.;
                    delta2+=weight*(aa-bb)*(aa-bb); next2+=weight*aa*aa; old2+=weight*bb*bb;
                }
            }
            if(precision_check)
                d.residual[id]=isfinite(delta2)&&isfinite(next2)&&isfinite(old2)?delta2/fmax(1.,fmax(next2,old2)):CUDART_INF;
            cached.valid=0;
            return;
        }
    }
    #pragma unroll
    for(int r=0;r<3;++r)
        writeNextMessage<3>(d,new_eta,r,id,__dsub_rn(cached.coef[12+r],dot3Cpu(
            cached.map[3*r],x[0],cached.map[3*r+1],x[1],cached.map[3*r+2],x[2])),view);
}

__device__ __forceinline__ void gatherOwnedBelief(Device d,const double* previous,double* next,
    const double* eta,const double* lam,int var,bool eta_only) {
    double value[9];
    #pragma unroll
    for(int k=0;k<9;++k) value[k]=d.base[k*d.n+var];
    for(int pos=d.offsets[var];pos<d.offsets[var+1];++pos) {
        const int slot=d.slots[pos];
        #pragma unroll
        for(int k=0;k<3;++k) value[k]+=eta[messageIndex<3>(k,slot,d.s)];
        if(!eta_only) {
            #pragma unroll
            for(int k=0;k<6;++k) value[3+k]+=lam[messageIndex<6>(k,slot,d.s)];
        }
    }
    #pragma unroll
    for(int k=0;k<3;++k) next[k*d.n+var]=value[k];
    #pragma unroll
    for(int k=3;k<9;++k) next[k*d.n+var]=eta_only?previous[k*d.n+var]:value[k];
}


struct IncomingSlots { int slot[4]; };

__device__ __forceinline__ double gatherComponent(Device d,int var,int component,int begin,int end,
    bool initial,bool need_lambda,const double* eta,const double* lam,IncomingSlots slots={},double cached_base=0.,LocalMessageView view={}) {
    if(component>=9 || (component>=3 && !need_lambda)) return 0.;
    if(initial) return d.belief[component*d.n+var];
    double value=
#ifdef HGBP_CACHED_BASE
        cached_base;
#else
        d.base[component*d.n+var];
#endif
    const double* input=component<3?eta:lam;
    const int plane=component<3?component:component-3;
    int tail=begin;
#ifdef HGBP_CACHED_INCOMING
    #pragma unroll
    for(int k=0;k<4;++k) if(begin+k<end) {
        const int slot=slots.slot[k];
        value+=component<3?readOldMessage<3>(d,input,plane,slot,view):readOldMessage<6>(d,input,plane,slot,view);
    }
    tail=begin+4;
#endif
    for(int pos=tail;pos<end;++pos) {
        const int slot=d.slots[pos];
        value+=component<3?readOldMessage<3>(d,input,plane,slot,view):readOldMessage<6>(d,input,plane,slot,view);
    }
    return value;
}

#ifdef HGBP_GATHER_SPECIALIZED
template<int Width> __device__ __forceinline__ void gatherTile(Device d,int var,int lane,int begin,int end,
    bool initial,bool need_lambda,const double* eta,const double* lam,unsigned mask,double incoming[9],LocalMessageView view={}) {
    // Constant tile widths shorten the live range of each component accumulator.
    #pragma unroll
    for(int base=0;base<9;base+=Width) {
        const double value=gatherComponent(d,var,base+lane,begin,end,initial,need_lambda,eta,lam,{},0.,view);
        #pragma unroll
        for(int k=0;k<9;++k) if(k/Width==base/Width)
            incoming[k]=__shfl_sync(mask,value,k%Width,Width);
    }
}
#endif

__global__ void sweepsGatherKernel(Device d,int sweeps,double jitter) {
    auto grid=cooperative_groups::this_grid();
    const int id=blockIdx.x*blockDim.x+threadIdx.x, stride=blockDim.x*gridDim.x;
    __shared__ SE2CudaState control;
#ifndef HGBP_WARP_BLOCK
    __shared__ double reduction[256];
#endif
    if(threadIdx.x==0) control=*d.state;
#ifndef HGBP_LOCAL_MESSAGES
    localSync();
#endif
    int ei=d.indices[0],li=d.indices[1];
    CachedMessage cached;
    const bool active=id<d.s && d.source[id]>=0;
    LocalMessageView view{};
#ifdef HGBP_LOCAL_MESSAGES
    __shared__ double local_eta[2][3*32],local_lam[2][6*32];
#ifdef HGBP_SHARED_PTX
    const unsigned shared_eta=static_cast<unsigned>(__cvta_generic_to_shared(local_eta));
    const unsigned shared_lam=static_cast<unsigned>(__cvta_generic_to_shared(local_lam));
#endif
    view.publish_message=active && d.external[id];
    if(active) {
#ifdef HGBP_SHARED_PTX
        #pragma unroll
        for(int k=0;k<3;++k) sharedMessageStore(shared_eta,(ei*3+k)*32+threadIdx.x,d.eta[ei][messageIndex<3>(k,id,d.s)]);
        #pragma unroll
        for(int k=0;k<6;++k) sharedMessageStore(shared_lam,(li*6+k)*32+threadIdx.x,d.lam[li][messageIndex<6>(k,id,d.s)]);
#else
        #pragma unroll
        for(int k=0;k<3;++k) local_eta[ei][k*32+threadIdx.x]=d.eta[ei][messageIndex<3>(k,id,d.s)];
        #pragma unroll
        for(int k=0;k<6;++k) local_lam[li][k*32+threadIdx.x]=d.lam[li][messageIndex<6>(k,id,d.s)];
#endif
    }
#endif
    int var=0,begin=0,end=0,width=4,lane=0;
    unsigned mask=0;
    if(id<d.s) {
        var=d.owner[id]; width=d.width[id]; lane=id&(width-1);
        begin=d.offsets[var]; end=d.offsets[var+1];
        mask=(0xffffffffu>>(32-width))<<((threadIdx.x&31)&~(width-1));
    }
    IncomingSlots incoming_slots{};
#ifdef HGBP_CACHED_BASE
    const double base0=id<d.s && lane<9?d.base[lane*d.n+var]:0.;
    const double base1=id<d.s && lane+width<9?d.base[(lane+width)*d.n+var]:0.;
    const double base2=id<d.s && lane+2*width<9?d.base[(lane+2*width)*d.n+var]:0.;
    const double base3=id<d.s && lane+3*width<9?d.base[(lane+3*width)*d.n+var]:0.;
    const double base4=id<d.s && lane+4*width<9?d.base[(lane+4*width)*d.n+var]:0.;
#else
    constexpr double base0=0.,base1=0.,base2=0.,base3=0.,base4=0.;
#endif
#ifdef HGBP_CACHED_INCOMING
    if(id<d.s) {
        #pragma unroll
        for(int k=0;k<4;++k) incoming_slots.slot[k]=begin+k<end?d.slots[begin+k]:0;
    }
#endif
    if(active) {
        #pragma unroll
        for(int j=0;j<27;++j) cached.coef[j]=d.coef[j*d.s+id];
        #pragma unroll
        for(int j=0;j<9;++j) cached.map[j]=d.map[j*d.s+id];
        cached.valid=d.valid[id]!=0;
    }
#ifdef HGBP_LOCAL_MESSAGES
    localSync();
#endif
    for(int iteration=0;iteration<sweeps;++iteration) {
        const bool test=control.sweep+1>=control.next_check;
        const bool eta_only=control.frozen && !test;
        const double* old_eta=d.eta[ei]; double* new_eta=d.eta[ei^1];
        const double* old_lam=d.lam[li]; double* new_lam=eta_only?d.lam[li]:d.lam[li^1];
#ifdef HGBP_LOCAL_MESSAGES
#ifdef HGBP_SHARED_PTX
        view.local_eta_old=shared_eta+ei*3*32*sizeof(double); view.local_eta_next=shared_eta+(ei^1)*3*32*sizeof(double);
        view.local_lam_old=shared_lam+li*6*32*sizeof(double); view.local_lam_next=shared_lam+(li^1)*6*32*sizeof(double);
#else
        view.local_eta_old=local_eta[ei]; view.local_eta_next=local_eta[ei^1];
        view.local_lam_old=local_lam[li]; view.local_lam_next=local_lam[li^1];
#endif
#endif
        double maximum=0;
        if(id<d.s) {
            const bool need_lambda=!eta_only || !control.maps_valid;
            double incoming[9];
#ifdef HGBP_GATHER_SPECIALIZED
            if(width==2) gatherTile<2>(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,mask,incoming,view);
            else if(width==4) gatherTile<4>(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,mask,incoming,view);
            else if(width==8) gatherTile<8>(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,mask,incoming,view);
            else if(width==16) gatherTile<16>(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,mask,incoming,view);
            else gatherTile<32>(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,mask,incoming,view);
#else
            const double v0=gatherComponent(d,var,lane,begin,end,iteration==0,need_lambda,old_eta,old_lam,incoming_slots,base0,view);
            const double v1=gatherComponent(d,var,lane+width,begin,end,iteration==0,need_lambda,old_eta,old_lam,incoming_slots,base1,view);
            const double v2=gatherComponent(d,var,lane+2*width,begin,end,iteration==0,need_lambda,old_eta,old_lam,incoming_slots,base2,view);
            const double v3=gatherComponent(d,var,lane+3*width,begin,end,iteration==0,need_lambda,old_eta,old_lam,incoming_slots,base3,view);
            const double v4=gatherComponent(d,var,lane+4*width,begin,end,iteration==0,need_lambda,old_eta,old_lam,incoming_slots,base4,view);
            #pragma unroll
            for(int k=0;k<9;++k) {
                const double component=k<width?v0:k<2*width?v1:k<3*width?v2:k<4*width?v3:v4;
                incoming[k]=__shfl_sync(mask,component,k&(width-1),width);
            }
#endif
            if(active) {
                cachedMessage<true>(d,id,cached,eta_only,test,old_eta,new_eta,old_lam,new_lam,jitter,incoming,view);
                if(test) maximum=d.residual[id];
            }
        }
        if(test) {
#ifdef HGBP_WARP_BLOCK
            for(int offset=16;offset>0;offset/=2)
                maximum=fmax(maximum,__shfl_down_sync(0xffffffffu,maximum,offset));
            if(threadIdx.x==0) d.block_residual[blockIdx.x]=maximum;
#else
            reduction[threadIdx.x]=maximum;
            __syncthreads();
            for(int size=blockDim.x/2;size>0;size/=2) {
                if(threadIdx.x<size) reduction[threadIdx.x]=fmax(reduction[threadIdx.x],reduction[threadIdx.x+size]);
                __syncthreads();
            }
            if(threadIdx.x==0) d.block_residual[blockIdx.x]=reduction[0];
#endif
        }
        // The sole cross-block dependency is publication of the next messages.
#ifdef HGBP_SOURCE_DEPENDENCY
        sourceDependencySync(d,iteration+1,test);
#elif defined(HGBP_WARP_GRID)
        warpGridSync(d,control.sweep+1);
#else
        grid.sync();
#endif
#ifndef HGBP_SOURCE_DEPENDENCY
        if(d.state->failed_slot>=0) break;
#endif
        double global_residual=0;
        if(test) {
            // Max is associative for these nonnegative, finite-or-inf norms.
            // Distribute the global precision check across the CTA instead of
            // serially loading every block's residual on each CTA leader.
            for(int j=threadIdx.x;j<gridDim.x;j+=blockDim.x)
                global_residual=fmax(global_residual,d.block_residual[j]);
            for(int offset=16;offset>0;offset/=2)
                global_residual=fmax(global_residual,__shfl_down_sync(0xffffffffu,global_residual,offset));
#ifndef HGBP_WARP_BLOCK
            if((threadIdx.x&31)==0) reduction[threadIdx.x/32]=global_residual;
            __syncthreads();
            if(threadIdx.x==0) {
                global_residual=0;
                for(int warp=0;warp<blockDim.x/32;++warp)
                    global_residual=fmax(global_residual,reduction[warp]);
            }
#endif
        }
        if(threadIdx.x==0) {
            auto& s=control;
            ++s.sweep; if(eta_only) ++s.eta; else ++s.full;
            s.maps_valid=eta_only;
            if(test) {
                s.residual=sqrt(global_residual); s.last_check=s.sweep; ++s.checks;
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
        }
        ei^=1; if(!eta_only) li^=1;
        localSync();
#ifdef HGBP_SOURCE_DEPENDENCY
        // Every CTA must finish reading the global precision residual before
        // any independent partition can publish a subsequent check's residual.
        if(test) grid.sync();
#endif
    }
#if defined(HGBP_SOURCE_DEPENDENCY) && !defined(HGBP_LOCAL_MESSAGES)
    // Failure is committed only at the batch boundary, never by stranding
    // neighboring partitions that still need this CTA's completion epoch.
    grid.sync();
#endif
    if(active) {
        #pragma unroll
        for(int j=0;j<9;++j) d.map[j*d.s+id]=cached.map[j];
        d.valid[id]=cached.valid;
    }
#ifdef HGBP_LOCAL_MESSAGES
    // Host/shadow export and final belief gather need the full current state,
    // including slots whose per-sweep publications stayed inside this CTA.
    if(active) {
#ifdef HGBP_SHARED_PTX
        #pragma unroll
        for(int k=0;k<3;++k) d.eta[ei][messageIndex<3>(k,id,d.s)]=sharedMessageLoad(shared_eta,(ei*3+k)*32+threadIdx.x);
        #pragma unroll
        for(int k=0;k<6;++k) d.lam[li][messageIndex<6>(k,id,d.s)]=sharedMessageLoad(shared_lam,(li*6+k)*32+threadIdx.x);
#else
        #pragma unroll
        for(int k=0;k<3;++k) d.eta[ei][messageIndex<3>(k,id,d.s)]=local_eta[ei][k*32+threadIdx.x];
        #pragma unroll
        for(int k=0;k<6;++k) d.lam[li][messageIndex<6>(k,id,d.s)]=local_lam[li][k*32+threadIdx.x];
#endif
    }
    grid.sync();
#endif
    if(d.state->failed_slot<0)
        for(int var=id;var<d.n;var+=stride)
            gatherOwnedBelief(d,d.belief,d.belief,d.eta[ei],d.lam[li],var,false);
    // Finish every reader of the failure state before publishing final control.
    grid.sync();
    if(id==0) {
        control.builds=d.state->builds; control.failed_slot=d.state->failed_slot;
        *d.state=control; d.indices[0]=ei; d.indices[1]=li; d.indices[2]=0;
    }
}

__global__ void unpack(SE2CudaView raw,Device d,bool all
#ifdef HGBP_PACKED_TRANSPORT
    ,SE2CudaState* out_state
#endif
) {
#ifdef HGBP_PACKED_TRANSPORT
    if(blockIdx.x==0 && threadIdx.x==0) *out_state=*d.state;
#endif
    if(d.state->failed_slot>=0) return;
    const int id=blockIdx.x*blockDim.x+threadIdx.x;
    if(all && id<d.s) {
        const int original=d.source[id];
        if(original>=0) {
        #pragma unroll
        for(int k=0;k<3;++k) raw.message_eta[3*original+k]=d.eta[d.indices[0]][messageIndex<3>(k,id,d.s)];
        #pragma unroll
        for(int k=0;k<6;++k) raw.message_lam[6*original+k]=d.lam[d.indices[1]][messageIndex<6>(k,id,d.s)];
        #pragma unroll
        for(int k=0;k<9;++k) ((original&1)?raw.map1:raw.map0)[9*(original/2)+k]=d.map[k*d.s+id];
        if(!(original&1)) raw.map_valid[original/2]=d.valid[id];
        }
    }
    if(id<d.n && raw.belief_eta) {
        const double* belief=d.indices[2]?d.belief_alt:d.belief;
        #pragma unroll
        for(int k=0;k<3;++k) raw.belief_eta[3*id+k]=belief[k*d.n+id];
        #pragma unroll
        for(int k=0;k<6;++k) raw.belief_lam[6*id+k]=belief[(3+k)*d.n+id];
    }
    if(all && id<d.m) raw.residuals[id]=fmax(d.residual[d.forward[2*id]],d.residual[d.forward[2*id+1]]);
}

#ifdef HGBP_SEGMENT_EXECUTOR
#include "se2_segment_kernel.cuh"
#endif
#ifdef HGBP_GRAPH_STEPS
#include "se2_graph_steps.cuh"
#endif
#ifdef HGBP_DEVICE_CYCLE
#include "se2_cycle_kernels.cuh"
#endif
#ifdef HGBP_TRANSPORT_GRAPH
#include "se2_transport_graph.cuh"
#endif

#ifdef HGBP_PHASE_GRAPHS
#include "se2_phase_kernels.cuh"
#endif

struct Engine {
#ifdef HGBP_TRANSPORT_GRAPH
    TransportGraphStorage transport;
#endif
#ifdef HGBP_TOPOLOGY_PLAN
    std::string topology_mode;
    Buffer<std::uint8_t> external;
#endif
#ifdef HGBP_PHASE_GRAPHS
    PhaseStorage phase;
#endif
#ifdef HGBP_SOURCE_DEPENDENCY
    Buffer<unsigned> arrival;
    Buffer<int> dependency_offsets,dependency_blocks;
    int dependency_threads=-1;
#endif
#ifdef HGBP_DEVICE_CYCLE
    CycleStorage cycle;
#endif
#ifdef HGBP_GRAPH_STEPS
    Buffer<SE2GraphControl> graph_controls;
    cudaGraph_t sweep_graph=nullptr;
    cudaGraphExec_t sweep_exec=nullptr;
    cudaStream_t capture_stream=nullptr;
    int graph_count=-1,graph_threads=-1,graph_blocks=-1;
    double graph_jitter=0;
#endif
#ifdef HGBP_PACKED_TRANSPORT
    Buffer<unsigned char> input_packet,output_packet;
    int cached_threads=-1,cached_device=-1,cached_active=0;
    cudaDeviceProp cached_properties{};
    cudaFuncAttributes cached_attributes{};
#endif
#ifdef HGBP_WARP_GRID
    Buffer<unsigned> barrier;
#endif
    Buffer<SE2CudaFactor> factors;
    Buffer<int> owners,widths;
    std::vector<int> h_owner,h_width;
    Buffer<int> offsets,slots,neighbors,indices,source,forward,opposite,order,target,pending;
    Buffer<double> base_eta,base_lam,belief_eta,belief_lam,message_eta,message_lam,map0,map1,scales,residuals;
    Buffer<std::uint8_t> valid,valid_soa;
    Buffer<double> coef,base,belief,belief_alt,eta0,eta1,lam0,lam1,map,scale,residual,block_residual;
    Buffer<SE2CudaState> state;
    Buffer<unsigned long long> clocks;
    cudaEvent_t events[4]{};
    bool initialized=false;
    int last_n=0,last_m=0;
    const int* last_offsets=nullptr;
    const int* last_slots=nullptr;
    int last_layout=-1,last_lanes=-1;
    std::vector<int> h_offsets,h_slots,h_source,h_forward,h_opposite,h_order;
    Engine() { for(auto& e:events) check(cudaEventCreate(&e)); }
    ~Engine() {
#ifdef HGBP_GRAPH_STEPS
        if(sweep_exec) cudaGraphExecDestroy(sweep_exec);
        if(sweep_graph) cudaGraphDestroy(sweep_graph);
        if(capture_stream) cudaStreamDestroy(capture_stream);
#endif
        for(auto e:events) cudaEventDestroy(e);
    }
    SE2CudaTiming run(SE2CudaView& v,int count,double jitter) {
        const int n=v.n,m=v.m,raw_slots=2*m;
        const int lanes=std::getenv("HGBP_RESIDENT_GROUP")?std::atoi(std::getenv("HGBP_RESIDENT_GROUP")):0;
        const bool last_writer=std::getenv("HGBP_RESIDENT_LAST_WRITER")!=nullptr;
        const int layout=std::getenv("HGBP_RESIDENT_LAYOUT")?std::atoi(std::getenv("HGBP_RESIDENT_LAYOUT")):0;
        if(layout && lanes==0) throw std::runtime_error("Owned layout requires an owned executor");
        if(last_writer && (layout || lanes)) throw std::runtime_error("Last-writer requires contiguous directed layout");
        const bool new_topology=!initialized || n!=last_n || m!=last_m ||
            last_offsets!=v.offsets || last_slots!=v.slots || layout!=last_layout || lanes!=last_lanes;
        const bool new_outer=new_topology || v.state.sweep==0;
#ifdef HGBP_DEVICE_CYCLE
        const bool fused_cycle=se2CycleResident() && std::getenv("HGBP_CYCLE_FUSED") && !new_outer && cycle.prepared;
        if(new_topology) cycle.invalidateGraph();
#endif
        if(new_outer && v.state.sweep!=0)
            throw std::runtime_error("Resident workspace must begin at a relinearization boundary");
        const char* shadow=std::getenv("HGBP_CUDA_SHADOW");
        const char* export_all=std::getenv("HGBP_RESIDENT_EXPORT_ALL");
        const bool all=(shadow && std::strcmp(shadow,"1")==0) ||
            (export_all && std::strcmp(export_all,"1")==0);
        SE2CudaView raw=v;
#ifndef HGBP_TRANSPORT_GRAPH
        check(cudaEventRecord(events[0]));
#endif
        if(new_topology) {
#ifdef HGBP_TOPOLOGY_PLAN
            const char* requested=std::getenv("HGBP_TOPOLOGY_MODE");
            topology_mode=requested?requested:"greedy";
            const auto plan=makeSE2SourcePlan(n,v.offsets,v.slots,topology_mode);
            h_order=plan.order; h_source=plan.source; h_owner=plan.owner;
            h_width=plan.width; h_forward=plan.forward;
            h_offsets=plan.offsets; h_slots=plan.slots; h_opposite=plan.opposite;
            external.upload(plan.external.data(),plan.external.size());
            std::cout<<"CUDA_TOPOLOGY mode="<<topology_mode<<" internal="<<plan.internal_messages
                <<" messages="<<raw_slots<<" blocks="<<(h_source.size()+31)/32<<'\n';
#else
            h_order.resize(n); std::iota(h_order.begin(),h_order.end(),0);
            auto tile_width=[&](int var) {
                int width=2;
                while(width<v.offsets[var+1]-v.offsets[var]) width*=2;
                if(width>32) throw std::runtime_error("Source tile requires degree <=32");
                return width;
            };
            std::stable_sort(h_order.begin(),h_order.end(),[&](int a,int b){return tile_width(a)>tile_width(b);});
            h_source.clear(); h_owner.clear(); h_width.clear(); h_forward.assign(raw_slots,-1);
            for(int var:h_order) {
                const int degree=v.offsets[var+1]-v.offsets[var],width=tile_width(var);
                if(h_source.size()%width) throw std::runtime_error("Source tile alignment");
                for(int lane=0;lane<width;++lane) {
                    const int original=lane<degree?(v.slots[v.offsets[var]+lane]^1):-1;
                    if(original>=0) h_forward[original]=static_cast<int>(h_source.size());
                    h_source.push_back(original); h_owner.push_back(var); h_width.push_back(width);
                }
            }
            if(std::find(h_forward.begin(),h_forward.end(),-1)!=h_forward.end())
                throw std::runtime_error("Source tile lost a directed message");
            h_offsets.assign(v.offsets,v.offsets+n+1); h_slots.resize(raw_slots);
            for(int i=0;i<raw_slots;++i) h_slots[i]=h_forward[v.slots[i]];
            h_opposite.resize(h_source.size());
            for(size_t i=0;i<h_source.size();++i) h_opposite[i]=h_source[i]<0?0:h_forward[h_source[i]^1];
#endif
            owners.upload(h_owner.data(),h_owner.size()); widths.upload(h_width.data(),h_width.size());
            offsets.upload(h_offsets.data(),h_offsets.size()); slots.upload(h_slots.data(),h_slots.size());
            source.upload(h_source.data(),h_source.size()); forward.upload(h_forward.data(),h_forward.size());
            opposite.upload(h_opposite.data(),h_opposite.size()); order.upload(h_order.data(),h_order.size());
            std::cout<<"CUDA_SLOT_LAYOUT original="<<raw_slots<<" allocated="<<h_source.size()
                <<" layout=source_warp min_width=2 max_width=32\n";
        }
        const int s=static_cast<int>(h_source.size());
        raw.offsets=offsets.p; raw.slots=slots.p;
#ifdef HGBP_PACKED_TRANSPORT
        const SE2TransferLayout packet(n,m);
        const bool grow_input=input_packet.capacity<packet.input_size;
        input_packet.reserve(packet.input_size); input_packet.staging();
        if(grow_input) std::memset(input_packet.pinned,0,input_packet.capacity);
        auto stage=[&](size_t offset,const void* src,size_t size) {
            std::memcpy(input_packet.pinned+offset,src,size);
        };
        stage(packet.state,&v.state,sizeof(v.state));
        bool resident_cycle=false;
#ifdef HGBP_DEVICE_CYCLE
        resident_cycle=se2CycleResident();
#endif
        if(new_outer || !resident_cycle) stage(packet.belief_eta,v.belief_eta,3*size_t(n)*sizeof(double));
        if(new_outer) {
            stage(packet.factors,v.factors,size_t(m)*sizeof(SE2CudaFactor));
            stage(packet.base_eta,v.base_eta,3*size_t(n)*sizeof(double));
            stage(packet.base_lam,v.base_lam,6*size_t(n)*sizeof(double));
            stage(packet.belief_lam,v.belief_lam,6*size_t(n)*sizeof(double));
            stage(packet.scales,v.scales,6*size_t(raw_slots)*sizeof(double));
        }
#ifdef HGBP_DEVICE_CYCLE
        if(!fused_cycle)
#endif
#ifndef HGBP_TRANSPORT_GRAPH
        check(cudaMemcpyAsync(input_packet.p,input_packet.pinned,
            new_outer?packet.input_size:resident_cycle?sizeof(SE2CudaState):packet.cycle_size,cudaMemcpyHostToDevice));
#endif
        auto* device_state=reinterpret_cast<SE2CudaState*>(input_packet.p+packet.state);
        raw.factors=reinterpret_cast<SE2CudaFactor*>(input_packet.p+packet.factors);
        raw.base_eta=reinterpret_cast<double*>(input_packet.p+packet.base_eta);
        raw.base_lam=reinterpret_cast<double*>(input_packet.p+packet.base_lam);
        raw.belief_eta=reinterpret_cast<double*>(input_packet.p+packet.belief_eta);
        raw.belief_lam=reinterpret_cast<double*>(input_packet.p+packet.belief_lam);
        raw.scales=reinterpret_cast<double*>(input_packet.p+packet.scales);
        if(all) {
            message_eta.reserve(3*raw_slots); message_lam.reserve(6*raw_slots);
            map0.reserve(9*m); map1.reserve(9*m); valid.reserve(m); residuals.reserve(m);
        }
        raw.message_eta=message_eta.p; raw.message_lam=message_lam.p;
        raw.map0=map0.p; raw.map1=map1.p; raw.map_valid=valid.p; raw.residuals=residuals.p;
        const bool grow_output=output_packet.capacity<packet.output_size;
        output_packet.reserve(packet.output_size); output_packet.staging();
        if(grow_output) check(cudaMemsetAsync(output_packet.p,0,output_packet.capacity));
#else
        if(new_outer) {
            factors.upload(v.factors,m);
            base_eta.upload(v.base_eta,3*n); base_lam.upload(v.base_lam,6*n);
            belief_lam.upload(v.belief_lam,6*n);
            // Mode-2 initialization is exactly the factor natural parameters.
            // Recreate these on device instead of uploading duplicate arrays.
            message_eta.reserve(3*raw_slots); message_lam.reserve(6*raw_slots);
            map0.reserve(9*m); map1.reserve(9*m); valid.reserve(m);
            scales.upload(v.scales,6*raw_slots); residuals.reserve(m);
        }
        raw.factors=factors.p; raw.base_eta=base_eta.p; raw.base_lam=base_lam.p;
        belief_eta.upload(v.belief_eta,3*n); raw.belief_eta=belief_eta.p;
        raw.belief_lam=belief_lam.p; raw.message_eta=message_eta.p; raw.message_lam=message_lam.p;
        raw.map0=map0.p; raw.map1=map1.p; raw.map_valid=valid.p;
        raw.scales=scales.p; raw.residuals=residuals.p;
        state.upload(&v.state,1);
        auto* device_state=state.p;
#endif
        neighbors.reserve(s); target.reserve(s); pending.reserve(n); indices.reserve(3); coef.reserve(27*s); base.reserve(9*n); belief.reserve(9*n); belief_alt.reserve(9*n);
        eta0.reserve(3*s); eta1.reserve(3*s); lam0.reserve(6*s); lam1.reserve(6*s);
        map.reserve(9*s); scale.reserve(6*s); residual.reserve(s); valid_soa.reserve(s);
        block_residual.reserve(
#ifdef HGBP_SINGLE_CHECK
            2+
#endif
#ifdef HGBP_GRAPH_STEPS
            2*
#endif
            ((std::max(16*n,s)+31)/32));
        const bool profile=std::getenv("HGBP_RESIDENT_CLOCK")!=nullptr;
        if(profile && (lanes || last_writer)) throw std::runtime_error("Stage clocks currently require two-pass control");
        if(profile) clocks.reserve(5*block_residual.capacity);
        Device d{n,m,s,neighbors.p,offsets.p,slots.p,indices.p,coef.p,base.p,belief.p,
            {eta0.p,eta1.p},{lam0.p,lam1.p},map.p,scale.p,residual.p,block_residual.p,valid_soa.p,device_state,belief_alt.p,
            source.p,forward.p,opposite.p,order.p,profile?clocks.p:nullptr,target.p,pending.p,owners.p,widths.p
#ifdef HGBP_WARP_GRID
            ,nullptr,nullptr
#endif
        };
#ifdef HGBP_LOCAL_MESSAGES
        d.external=external.p;
#endif
#ifdef HGBP_TRANSPORT_GRAPH
        if(profile || lanes || last_writer || layout)
            throw std::runtime_error("Transport graph requires the packed source-warp path");
        const auto timing=transport.run(d,raw,packet,input_packet,output_packet,count,jitter,new_outer,new_topology,all);
        std::memcpy(&v.state,output_packet.pinned+packet.out_state,sizeof(v.state));
        const bool commit=v.state.failed_slot<0;
        if(commit) {
            std::memcpy(v.belief_eta,output_packet.pinned+packet.out_eta,3*size_t(n)*sizeof(double));
            std::memcpy(v.belief_lam,output_packet.pinned+packet.out_lam,6*size_t(n)*sizeof(double));
        }
        if(all) {
            message_eta.download(v.message_eta,3*raw_slots); message_lam.download(v.message_lam,6*raw_slots);
            map0.download(v.map0,9*m); map1.download(v.map1,9*m); valid.download(v.map_valid,m);
            residuals.download(v.residuals,m);
            check(cudaStreamSynchronize(nullptr));
            message_eta.finish(commit); message_lam.finish(commit);
            map0.finish(commit); map1.finish(commit); valid.finish(commit); residuals.finish(commit);
        }
        initialized=commit; last_n=n; last_m=m;
        last_offsets=v.offsets; last_slots=v.slots; last_layout=layout; last_lanes=lanes;
        return timing;
#endif
#ifdef HGBP_DEVICE_CYCLE
        cycle.lower=d; cycle.raw=raw;
        if(new_outer) { cycle.prepared=false; cycle.injected=false; cycle.rhs_ready=false; }
#endif
#ifdef HGBP_WARP_GRID
        barrier.reserve(64);
        d.barrier_count=barrier.p; d.barrier_epoch=barrier.p+32;
        if(new_outer) check(cudaMemsetAsync(barrier.p,0,64*sizeof(unsigned)));
#endif
        const int pack_blocks=(std::max(n,s)+255)/256;
        if(new_outer) {
            check(cudaMemsetAsync(indices.p,0,3*sizeof(int)));
            pack<<<pack_blocks,256>>>(raw,d);
            if(last_writer) initializePending<<<(n+255)/256,256>>>(d);
        } else {
#ifdef HGBP_DEVICE_CYCLE
            if(resident_cycle) {
                if(!cycle.injected) throw std::runtime_error("Resident cycle missing upper injection");
                cycle.injected=false;
            } else
#endif
            patchBeliefEta<<<(n+255)/256,256>>>(raw,d);
        }
        check(cudaGetLastError());
        check(cudaEventRecord(events[1]));
        const int threads=std::getenv("HGBP_RESIDENT_THREADS")?std::atoi(std::getenv("HGBP_RESIDENT_THREADS")):128;
#ifdef HGBP_TOPOLOGY_PLAN
        if(threads!=32) throw std::runtime_error("Topology partitions require 32-thread CTAs");
        const char* requested=std::getenv("HGBP_TOPOLOGY_MODE");
        if(topology_mode!=(requested?requested:"greedy")) throw std::runtime_error("Cannot switch topology during a solve");
#endif
#ifdef HGBP_WARP_BLOCK
        if(threads!=32) throw std::runtime_error("Warp executor requires exactly 32 threads per CTA");
#endif
        if((threads!=32 && threads!=64 && threads!=128 && threads!=256) || ((lanes || last_writer) && threads!=128))
            throw std::runtime_error("Unsupported resident thread count");
        if(last_writer || lanes || layout || profile) throw std::runtime_error("Cached gather is a separate executor");
        const void* kernel=reinterpret_cast<const void*>(
#ifdef HGBP_PHASE_GRAPHS
            phaseFullKernel
#elif defined(HGBP_SEGMENT_EXECUTOR)
            sweepsSegmentKernel
#elif defined(HGBP_GRAPH_STEPS)
            sweepsStepKernel
#else
            sweepsGatherKernel
#endif
        );
#ifdef HGBP_SEGMENT_EXECUTOR
        if(threads!=32) throw std::runtime_error("Segment executor requires 32-thread CTAs");
#endif
        int active=0,device=0; cudaDeviceProp prop{}; cudaFuncAttributes attr{};
#ifdef HGBP_PACKED_TRANSPORT
        check(cudaGetDevice(&device));
        if(cached_device>=0 && cached_device!=device)
            throw std::runtime_error("SE2 persistent workspace cannot switch CUDA devices");
        if(cached_threads!=threads) {
            check(cudaGetDeviceProperties(&cached_properties,device));
            check(cudaFuncGetAttributes(&cached_attributes,kernel));
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cached_active,kernel,threads,0));
            cached_threads=threads; cached_device=device;
        }
        prop=cached_properties; attr=cached_attributes; active=cached_active;
#else
        check(cudaGetDevice(&device)); check(cudaGetDeviceProperties(&prop,device));
        check(cudaFuncGetAttributes(&attr,kernel));
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,threads,0));
#endif
        if(!prop.cooperativeLaunch || active==0) throw std::runtime_error("Cooperative CUDA launch unavailable");
        const int work=
#ifdef HGBP_PHASE_WIDE_FULL
            8*m;
#else
            lanes?n*lanes:std::max(n,s);
#endif
#ifdef HGBP_GRAPH_STEPS
        const int blocks=(work+threads-1)/threads;
        graph_controls.reserve(2);
        if(2*blocks>block_residual.capacity) throw std::runtime_error("Graph residual buffers too small");
        if(!sweep_exec || new_topology || graph_count!=count || graph_threads!=threads ||
           graph_blocks!=blocks || graph_jitter!=jitter) {
            if(sweep_exec) { check(cudaGraphExecDestroy(sweep_exec)); sweep_exec=nullptr; }
            if(sweep_graph) { check(cudaGraphDestroy(sweep_graph)); sweep_graph=nullptr; }
            if(!capture_stream) check(cudaStreamCreateWithFlags(&capture_stream,cudaStreamNonBlocking));
            check(cudaStreamBeginCapture(capture_stream,cudaStreamCaptureModeThreadLocal));
            for(int step=0;step<count;++step)
                sweepsStepKernel<<<blocks,threads,0,capture_stream>>>(d,graph_controls.p,step,blocks,jitter);
            finishStepBatch<<<(std::max(n,s)+255)/256,256,0,capture_stream>>>(d,graph_controls.p,count,blocks);
            check(cudaStreamEndCapture(capture_stream,&sweep_graph));
            check(cudaGraphInstantiate(&sweep_exec,sweep_graph,nullptr,nullptr,0));
            graph_count=count; graph_threads=threads; graph_blocks=blocks; graph_jitter=jitter;
            std::cout<<"CUDA_SWEEP_GRAPH nodes="<<count+1<<" sweeps="<<count<<" blocks="<<blocks
                <<" threads="<<threads<<'\n';
        }
        check(cudaGraphLaunch(sweep_exec,nullptr));
#else
        const int blocks=std::min(active*prop.multiProcessorCount,(work+threads-1)/threads);
        if(blocks*threads<work) throw std::runtime_error("Required cooperative work exceeds resident capacity");
#ifdef HGBP_SOURCE_DEPENDENCY
        if(new_topology || dependency_threads!=threads) {
            const auto plan=makeSE2DependencyPlan(h_owner,h_offsets.data(),h_slots.data(),n,threads);
            if(plan.offsets.size()!=size_t(blocks+1)) throw std::runtime_error("Source dependency shape mismatch");
            dependency_offsets.upload(plan.offsets.data(),plan.offsets.size());
            dependency_blocks.upload(plan.blocks.data(),plan.blocks.size());
            dependency_threads=threads;
            std::cout<<"CUDA_SOURCE_DEPENDENCY blocks="<<blocks<<" directed_dependencies="<<plan.blocks.size()
                <<" global_pairs="<<blocks*blocks<<'\n';
        }
        arrival.reserve(32*blocks);
        check(cudaMemsetAsync(arrival.p,0,32*blocks*sizeof(unsigned)));
        d.arrival=arrival.p; d.dependency_offsets=dependency_offsets.p; d.dependency_blocks=dependency_blocks.p;
#endif
#ifdef HGBP_DEVICE_CYCLE
        if(fused_cycle) cycle.launchGraph(d,count,jitter,blocks,threads,
            reinterpret_cast<SE2CudaState*>(output_packet.p+packet.out_state),
            reinterpret_cast<SE2CudaState*>(output_packet.pinned+packet.out_state));
        else {
#endif
#ifdef HGBP_PHASE_GRAPHS
        phase.launch(d,count,jitter,blocks,threads,prop.multiProcessorCount,new_topology);
#else
        void* args[]={&d,&count,&jitter};
        check(cudaLaunchCooperativeKernel(kernel,blocks,threads,args));
#endif
#ifdef HGBP_DEVICE_CYCLE
        }
#endif
#endif
        check(cudaEventRecord(events[2]));
#ifdef HGBP_PACKED_TRANSPORT
#ifdef HGBP_DEVICE_CYCLE
        if(!fused_cycle || all) {
#endif
        auto output_view=raw;
        output_view.belief_eta=reinterpret_cast<double*>(output_packet.p+packet.out_eta);
        output_view.belief_lam=reinterpret_cast<double*>(output_packet.p+packet.out_lam);
        if(resident_cycle && !all) output_view.belief_eta=nullptr;
        unpack<<<pack_blocks,256>>>(output_view,d,all,
            reinterpret_cast<SE2CudaState*>(output_packet.p+packet.out_state));
        check(cudaGetLastError());
        check(cudaMemcpyAsync(output_packet.pinned,output_packet.p,
            resident_cycle && !all?sizeof(SE2CudaState):packet.output_size,cudaMemcpyDeviceToHost));
#ifdef HGBP_DEVICE_CYCLE
        }
#endif
#else
        unpack<<<pack_blocks,256>>>(raw,d,all); check(cudaGetLastError());
        state.download(&v.state,1);
        belief_eta.download(v.belief_eta,3*n); belief_lam.download(v.belief_lam,6*n);
#endif
        if(all) {
            message_eta.download(v.message_eta,3*raw_slots); message_lam.download(v.message_lam,6*raw_slots);
            map0.download(v.map0,9*m); map1.download(v.map1,9*m);
            valid.download(v.map_valid,m); residuals.download(v.residuals,m);
        }
        check(cudaEventRecord(events[3]));
        // Pinned transfers may finish while the CPU constructs the basis;
        // numerical host arrays are committed only after the callback returns.
        const auto status=cudaEventQuery(events[3]);
        if(status!=cudaSuccess && status!=cudaErrorNotReady) check(status);
        runSE2HostOverlap();
        check(cudaEventSynchronize(events[3]));
#ifdef HGBP_PACKED_TRANSPORT
        std::memcpy(&v.state,output_packet.pinned+packet.out_state,sizeof(v.state));
        const bool success=v.state.failed_slot<0;
        if(success && (!resident_cycle || all)) {
            std::memcpy(v.belief_eta,output_packet.pinned+packet.out_eta,3*size_t(n)*sizeof(double));
            std::memcpy(v.belief_lam,output_packet.pinned+packet.out_lam,6*size_t(n)*sizeof(double));
        }
#else
        state.finish();
        const bool success=v.state.failed_slot<0;
        belief_eta.finish(success); belief_lam.finish(success);
#endif
        message_eta.finish(success); message_lam.finish(success);
        map0.finish(success); map1.finish(success); valid.finish(success); residuals.finish(success);
        float upload=0,kernel_ms=0,download=0;
        check(cudaEventElapsedTime(&upload,events[0],events[1]));
        check(cudaEventElapsedTime(&kernel_ms,events[1],events[2]));
        check(cudaEventElapsedTime(&download,events[2],events[3]));
        if(profile) {
            std::vector<unsigned long long> values(5*blocks);
            clocks.download(values.data(),values.size()); check(cudaStreamSynchronize(nullptr)); clocks.finish();
            std::cout<<"CUDA_STAGE_CLOCK sweep="<<v.state.sweep;
            for(int j=0;j<5;++j) {
                unsigned long long sum=0;
                for(int b=0;b<blocks;++b) sum+=values[5*b+j];
                std::cout<<" stage"<<j<<'='<<sum;
            }
            std::cout<<'\n';
        }
        initialized=v.state.failed_slot<0; last_n=n; last_m=m;
        last_offsets=v.offsets; last_slots=v.slots;
        last_layout=layout; last_lanes=lanes;
        return {upload,kernel_ms,download,blocks,threads,attr.numRegs};
    }
};
}

void initializeSE2Cuda() { check(cudaFree(nullptr)); }
namespace {
Engine& cycleEngine() {
    static Engine engine;
    return engine;
}
}
SE2CudaTiming runSE2Cuda(SE2CudaView& view,int sweeps,double jitter) {
    return cycleEngine().run(view,sweeps,jitter);
}
#ifdef HGBP_DEVICE_CYCLE
void se2CycleMean(double* mean) { cycleEngine().cycle.getMean(mean,true); }
void se2CycleDownloadMean(double* mean) { cycleEngine().cycle.getMean(mean,false); }
void se2CycleRestrict(const SE2CycleView& v,double* rhs,double* mean) { cycleEngine().cycle.restrict(v,rhs,mean); }
void se2CycleInject(const SE2CycleView& v,const double* delta,double scale,double* mean,double* eta) {
    cycleEngine().cycle.inject(v,delta,scale,mean,eta);
}
#endif
