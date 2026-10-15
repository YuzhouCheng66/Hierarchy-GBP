#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace slam {
struct SE2PrecisionPolicy {
    static constexpr double tolerance=1e-8;
    static constexpr int check_period=5;
    static constexpr int stable_checks=3;
    static constexpr int frozen_check_period=25;
    static constexpr int max_check_period=25;
};

inline bool finiteSE2PrecisionValue(double value) {
    static_assert(sizeof(double)==sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits,&value,sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000);
}

inline void buildSE2PrecisionScales(const double* diagonal,double* scales) {
    constexpr int row[6]={0,1,2,1,2,2},col[6]={0,0,0,1,1,2};
    const double inv[3]={1/std::sqrt(std::max(std::abs(diagonal[0]),1e-30)),
                         1/std::sqrt(std::max(std::abs(diagonal[3]),1e-30)),
                         1/std::sqrt(std::max(std::abs(diagonal[5]),1e-30))};
    for(int k=0;k<6;++k) scales[k]=inv[row[k]]*inv[col[k]];
}

inline double scaledSE2PrecisionResidualSquared(const double* next,const double* old,const double* scales) {
    constexpr double weights[6]={1.,2.,2.,1.,2.,1.};
    double delta2=0,next2=0,old2=0;
    for(int k=0;k<6;++k) {
        const double scale=scales[k];
        const double a=next[k]*scale,b=old[k]*scale;
        const double weight=weights[k];
        delta2+=weight*(a-b)*(a-b); next2+=weight*a*a; old2+=weight*b*b;
    }
    // Nonfinite elements or overflow propagate to these nonnegative sums.
    // Bit checks avoid per-element CRT classification calls under MSVC /fp:fast.
    if(!finiteSE2PrecisionValue(delta2) || !finiteSE2PrecisionValue(next2) || !finiteSE2PrecisionValue(old2))
        return std::numeric_limits<double>::infinity();
    // Absolute floor in dimensionless factor-diagonal coordinates handles
    // nearly zero tree messages without dividing by cancellation noise.
    return delta2/std::max({1.,next2,old2});
}

inline double normalizedSE2PrecisionResidual(const double* next,const double* old,const double* diagonal) {
    double scales[6];
    buildSE2PrecisionScales(diagonal,scales);
    return std::sqrt(scaledSE2PrecisionResidualSquared(next,old,scales));
}
}
