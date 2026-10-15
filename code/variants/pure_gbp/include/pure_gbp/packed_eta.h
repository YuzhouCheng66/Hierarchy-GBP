#pragma once

namespace pure_gbp {
inline void GBP::buildPackedEta() {
    const auto start=workspace_timing_enabled?Clock::now():Clock::time_point{};
    ++workspace_metrics.pack_calls;
    if(!(eta_relaxation>0 && eta_relaxation<2))throw std::runtime_error("invalid packed eta relaxation");
    const bool reuse=workspace_reuse_enabled && frozen && packed_precision_ready;
    if(!reuse) {
        packed_stride=0;for(int i=0;i<blocks;++i)packed_stride=std::max(packed_stride,size(i));
        if(packed_stride>12)throw std::runtime_error("packed eta currently supports block dimensions <=12");
        auto resize=[&](std::vector<double>& values,std::size_t count) {
            if(values.capacity()<count)++workspace_metrics.pack_buffer_growths;
            values.resize(count);
        };
        const std::size_t s=std::size_t(packed_stride);
        resize(packed_map,edges.size()*s*s);resize(packed_eta,edges.size()*s);resize(packed_next,edges.size()*s);
        resize(packed_belief,std::size_t(n));resize(packed_inverse,std::size_t(blocks)*s*s);
        // Keep the legacy copy order when building precision (including gate=0).
        for(int p=0;p<int(edges.size());++p) {
            const auto& e=edges[p];
            for(int r=0;r<size(e.to);++r) {
                packed_eta[p*s+r]=e.eta[r];
                for(int c=0;c<size(e.from);++c)packed_map[(p*s+r)*s+c]=e.eta_map(r,c);
            }
        }
        for(int i=0;i<blocks;++i) for(int r=0;r<size(i);++r) {
            packed_belief[offsets[i]+r]=belief_eta[i][r];
            for(int c=0;c<size(i);++c)packed_inverse[(i*s+r)*s+c]=belief_inverse[i](r,c);
        }
        packed_precision_ready=workspace_reuse_enabled && frozen;
        ++workspace_metrics.precision_packs;
    } else {
        ++workspace_metrics.precision_reuses;
        // Eta may change with each RHS even when precision is frozen. Keep
        // importing it and exporting at return to preserve public Eigen state.
        const int s=packed_stride;
        for(int p=0;p<int(edges.size());++p) {
            const auto& e=edges[p];
            for(int r=0;r<size(e.to);++r)packed_eta[p*s+r]=e.eta[r];
        }
        for(int i=0;i<blocks;++i) for(int r=0;r<size(i);++r)
            packed_belief[offsets[i]+r]=belief_eta[i][r];
    }
    packed_ready=true;
    if(workspace_timing_enabled)workspace_metrics.pack_sec+=seconds(start);
}
template<int D> inline double packedDot(const double* a,const double* b) {
    double x=0;for(int k=0;k<D;++k)x+=a[k]*b[k];return x;
}
inline double packedDotDynamic(const double* a,const double* b,int count) {
    if(count==4)return packedDot<4>(a,b);
    if(count==12)return packedDot<12>(a,b);
    if(count==9)return packedDot<9>(a,b);
    if(count==6)return packedDot<6>(a,b);
    double x=0;for(int k=0;k<count;++k)x+=a[k]*b[k];return x;
}
inline void GBP::packedCompute(int p) {
    const auto& e=edges[p];const int s=packed_stride;
    double cavity[12];const double* rev=packed_eta.data()+e.reverse*s;
    const double* mean=packed_belief.data()+offsets[e.from];
    for(int k=0;k<size(e.from);++k)cavity[k]=mean[k]-rev[k];
    const double* map=packed_map.data()+p*s*s;double* out=packed_next.data()+p*s;
    if(eta_relaxation==1.) {
        for(int r=0;r<size(e.to);++r)out[r]=packedDotDynamic(map+r*s,cavity,size(e.from));
    } else {
        const double* old=packed_eta.data()+p*s;
        for(int r=0;r<size(e.to);++r)
            out[r]=(1.-eta_relaxation)*old[r]+eta_relaxation*packedDotDynamic(map+r*s,cavity,size(e.from));
    }
}
inline void GBP::packedCommit(int p) {
    const auto& e=edges[p];const int s=packed_stride;
    double* old=packed_eta.data()+p*s;const double* next=packed_next.data()+p*s;
    double* mean=packed_belief.data()+offsets[e.to];
    for(int r=0;r<size(e.to);++r) {mean[r]+=next[r]-old[r];old[r]=next[r];}
}
inline void GBP::packedMean(int i,Vec& z) {
    const double* m=packed_inverse.data()+i*packed_stride*packed_stride;
    const double* h=packed_belief.data()+offsets[i];double* out=z.data()+offsets[i];
    for(int r=0;r<size(i);++r)out[r]=packedDotDynamic(m+r*packed_stride,h,size(i));
}
inline void GBP::unpackEta() {
    const auto start=workspace_timing_enabled?Clock::now():Clock::time_point{};
    ++workspace_metrics.eta_unpacks;
    for(int p=0;p<int(edges.size());++p) for(int r=0;r<size(edges[p].to);++r)edges[p].eta[r]=packed_eta[p*packed_stride+r];
    for(int i=0;i<blocks;++i) for(int r=0;r<size(i);++r)belief_eta[i][r]=packed_belief[offsets[i]+r];
    if(workspace_timing_enabled)workspace_metrics.unpack_sec+=seconds(start);
}
}
