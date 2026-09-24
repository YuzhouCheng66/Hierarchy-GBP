#pragma once
#include <immintrin.h>

namespace slam {
// Exact double reciprocals, not approximate SIMD reciprocal instructions.
// All seven RHS reuse the six pivots. This changes rounding, not the solve.
template<int batch>
inline void solveSpd6ReciprocalLanes(const double* l,const double* inv,
        const double* cross,const double* eta,double* out) noexcept {
    const auto a=[l](int i) { int row=i%6,col=i/6; return l[row*(row+1)/2+col]; };
    const auto load=[cross,eta](int r) {
        return batch==0?_mm256_setr_pd(cross[r],cross[6+r],cross[12+r],cross[18+r]):
            _mm256_setr_pd(cross[24+r],cross[30+r],eta[r],0.0);
    };
    const auto sub=[](__m256d x,double b,__m256d y) {
        return _mm256_sub_pd(x,_mm256_mul_pd(_mm256_set1_pd(b),y));
    };
    const auto div=[inv](__m256d x,int r) {return _mm256_mul_pd(x,_mm256_set1_pd(inv[r]));};
    __m256d y[6],x[6];
    y[0]=div(load(0),0);
    y[1]=div(sub(load(1),a(1),y[0]),1);
    y[2]=div(sub(sub(load(2),a(2),y[0]),a(8),y[1]),2);
    y[3]=div(sub(sub(sub(load(3),a(3),y[0]),a(9),y[1]),a(15),y[2]),3);
    y[4]=div(sub(sub(sub(sub(load(4),a(4),y[0]),a(10),y[1]),a(16),y[2]),a(22),y[3]),4);
    y[5]=div(sub(sub(sub(sub(sub(load(5),a(5),y[0]),a(11),y[1]),a(17),y[2]),a(23),y[3]),a(29),y[4]),5);
    x[5]=div(y[5],5);
    x[4]=div(sub(y[4],a(29),x[5]),4);
    x[3]=div(sub(sub(y[3],a(22),x[4]),a(23),x[5]),3);
    x[2]=div(sub(sub(sub(y[2],a(16),x[4]),a(17),x[5]),a(15),x[3]),2);
    x[1]=div(sub(sub(sub(sub(y[1],a(10),x[4]),a(11),x[5]),a(9),x[3]),a(8),x[2]),1);
    x[0]=div(sub(sub(sub(sub(sub(y[0],a(4),x[4]),a(5),x[5]),a(3),x[3]),a(2),x[2]),a(1),x[1]),0);
    for(int r=0;r<6;++r) {
        alignas(32) double lanes[4]; _mm256_store_pd(lanes,x[r]);
        for(int j=0;j<(batch==0?4:3);++j) out[6*(4*batch+j)+r]=lanes[j];
    }
}
inline void solveSpd6CrossEtaPackedReciprocal(const double* l,const double* cross,
                                            const double* eta,double* out) noexcept {
    const double inv[6]={1/l[0],1/l[2],1/l[5],1/l[9],1/l[14],1/l[20]};
    solveSpd6ReciprocalLanes<0>(l,inv,cross,eta,out);
    solveSpd6ReciprocalLanes<1>(l,inv,cross,eta,out);
}
}
