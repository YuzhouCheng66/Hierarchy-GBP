#include "internal/se3_schur_solve.h"
#include "internal/finite_double.h"
#include <Eigen/Core>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
void scalar(const double* l, const double* rhs, double* out) {
    const double y0 = rhs[0] / l[0];
    const double y1 = (rhs[1] - l[1]*y0) / l[7];
    const double y2 = (rhs[2] - l[2]*y0 - l[8]*y1) / l[14];
    const double y3 = (rhs[3] - l[3]*y0 - l[9]*y1 - l[15]*y2) / l[21];
    const double y4 = (rhs[4] - l[4]*y0 - l[10]*y1 - l[16]*y2 - l[22]*y3) / l[28];
    const double y5 = (rhs[5] - l[5]*y0 - l[11]*y1 - l[17]*y2 - l[23]*y3 - l[29]*y4) / l[35];
    const double x5 = y5 / l[35];
    const double x4 = (y4-l[29]*x5) / l[28];
    const double x3 = (y3-l[22]*x4-l[23]*x5) / l[21];
    const double x2 = (y2-l[16]*x4-l[17]*x5-l[15]*x3) / l[14];
    const double x1 = (y1-l[10]*x4-l[11]*x5-l[9]*x3-l[8]*x2) / l[7];
    const double x0 = (y0-l[4]*x4-l[5]*x5-l[3]*x3-l[2]*x2-l[1]*x1) / l[0];
    out[0]=x0; out[1]=x1; out[2]=x2; out[3]=x3; out[4]=x4; out[5]=x5;
}
}

int main() {
    using Mat6 = Eigen::Matrix<double,6,6>;
    using Mat67 = Eigen::Matrix<double,6,7>;
    std::mt19937_64 rng(721035);
    std::uniform_real_distribution<double> uniform(-1,1);
    for(int i=0;i<100000;++i) {
        const std::uint64_t bits=rng();
        double value;
        std::memcpy(&value,&bits,sizeof(value));
        if(slam::finiteDouble(value)!=std::isfinite(value))
            throw std::runtime_error("finite classification mismatch");
    }
    for(double value:{0.,-0.,std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::signaling_NaN()})
        if(slam::finiteDouble(value)!=std::isfinite(value)) throw std::runtime_error("finite edge-case mismatch");
    double worst=0;
    for (int trial=0;trial<12000;++trial) {
        Mat6 l=Mat6::Zero(); Mat67 rhs;
        for(int c=0;c<6;++c) {
            l(c,c)=std::pow(10.,uniform(rng)*3);
            for(int r=c+1;r<6;++r) l(r,c)=uniform(rng);
        }
        for(int i=0;i<rhs.size();++i) rhs.data()[i]=trial==0 ? 0. : uniform(rng);
        std::array<double,44> guarded; guarded.fill(1234567.);
        Mat67 ref;
        for(int c=0;c<7;++c) scalar(l.data(),rhs.col(c).data(),ref.col(c).data());
        slam::solveSpd6CrossEtaBatched(l.data(),rhs.data(),rhs.col(6).data(),guarded.data()+1);
        if(guarded.front()!=1234567. || guarded.back()!=1234567.) throw std::runtime_error("output overwrite");
        if(std::memcmp(ref.data(),guarded.data()+1,sizeof(double)*42)!=0) throw std::runtime_error("scalar/SIMD bits differ");
        double packed[21];
        for(int r=0;r<6;++r) for(int c=0;c<=r;++c) packed[r*(r+1)/2+c]=l(r,c);
        guarded.fill(1234567.);
        // The parallel division kernel must remain bitwise identical.
        // The serial reciprocal kernel has its own backward-error regression test.
        slam::solveSpd6CrossEtaLanes<0,true>(packed,rhs.data(),rhs.col(6).data(),guarded.data()+1);
        slam::solveSpd6CrossEtaLanes<1,true>(packed,rhs.data(),rhs.col(6).data(),guarded.data()+1);
        if(guarded.front()!=1234567. || guarded.back()!=1234567.) throw std::runtime_error("packed output overwrite");
        if(std::memcmp(ref.data(),guarded.data()+1,sizeof(double)*42)!=0) throw std::runtime_error("packed scalar/SIMD bits differ");
        const Eigen::Map<const Mat67> x(guarded.data()+1);
        const Mat6 a=l*l.transpose();
        const double error=(a*x-rhs).norm()/std::max(a.norm()*x.norm()+rhs.norm(),1e-300);
        worst=std::max(worst,error);
        if(!std::isfinite(error) || error>1e-12) throw std::runtime_error("large solve backward error");
    }
    std::cout << "12000 systems / 504000 coefficients bitwise identical; worst backward error=" << worst << '\n';
}
