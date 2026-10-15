#pragma once
#include "internal/se3_residual.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slam {
// A common, canonical-H criterion: no GBP belief precision enters this metric.
class SE3ResidualCertificate {
    using M6=Eigen::Matrix<double,6,6>;
    using V6=Eigen::Matrix<double,6,1>;
    std::vector<M6,Eigen::aligned_allocator<M6>> inverse_;
    mutable std::vector<double> parts_;
    double rhs_square_=0;
    int threads_=1;
    static M6 unpack(const double* p) {
        M6 a;
        for(int c=0;c<6;++c) for(int r=0;r<=c;++r) a(r,c)=a(c,r)=p[c*(c+1)/2+r];
        return a;
    }
    double squareNorm(const Eigen::VectorXd& r) const {
        if(r.size()!=6*static_cast<Eigen::Index>(inverse_.size()))
            throw std::runtime_error("Residual certificate dimension mismatch");
        #pragma omp parallel for schedule(static) num_threads(threads_) if(threads_>1)
        for(int i=0;i<static_cast<int>(inverse_.size());++i) {
            const V6 v=r.segment<6>(6*i);
            parts_[i]=v.dot(inverse_[i]*v);
        }
        double total=0;
        for(double part:parts_) {
            if(!std::isfinite(part) || part<0) return std::numeric_limits<double>::infinity();
            total+=part;
        }
        return total;
    }
public:
    void initialize(const SyntheticSE3PackedSoAWorkspace& w,int threads) {
        threads_=std::max(1,threads);
        inverse_.resize(w.num_vars); parts_.resize(w.num_vars);
        Eigen::VectorXd rhs(6*w.num_vars);
        int failures=0;
        #pragma omp parallel for schedule(static) num_threads(threads_) if(threads_>1) reduction(+:failures)
        for(int i=0;i<w.num_vars;++i) {
            M6 a=unpack(w.prior_lam21.data()+21*i);
            V6 b=Eigen::Map<const V6>(w.prior_eta.data()+6*i);
            for(int p=w.unary_offsets[i];p<w.unary_offsets[i+1];++p) {
                const int f=w.unary_ids[p];
                a+=unpack(w.unary_lam21.data()+21*f);
                b+=Eigen::Map<const V6>(w.unary_eta.data()+6*f);
            }
            for(int p=w.binary_offsets[i];p<w.binary_offsets[i+1];++p) {
                const int slot=w.binary_slot_ids[p],f=slot/2;
                a+=unpack((slot%2?w.binary_diag1_lam21:w.binary_diag0_lam21).data()+21*f);
                b+=Eigen::Map<const V6>((slot%2?w.binary_eta1:w.binary_eta0).data()+6*f);
            }
            if(!a.allFinite() || !b.allFinite() || (a.diagonal().array()<=0.).any()) { ++failures; continue; }
            const V6 s=a.diagonal().array().sqrt().inverse();
            const M6 scaled=s.asDiagonal()*a*s.asDiagonal();
            Eigen::LLT<M6> llt(scaled);
            if(llt.info()!=Eigen::Success) { ++failures; continue; }
            const M6 inverse=llt.solve(M6::Identity());
            if(!inverse.allFinite() || (scaled*inverse-M6::Identity()).cwiseAbs().maxCoeff()>1e-7) { ++failures; continue; }
            inverse_[i]=s.asDiagonal()*inverse*s.asDiagonal();
            rhs.segment<6>(6*i)=b;
        }
        if(failures) throw std::runtime_error("Residual certificate canonical block inverse failed");
        rhs_square_=squareNorm(rhs);
        if(!std::isfinite(rhs_square_)) throw std::runtime_error("Invalid residual certificate rhs norm");
    }
    double relativeResidual(const Eigen::VectorXd& residual) const {
        const double value=squareNorm(residual);
        if(rhs_square_==0) return value==0 ? 0 : std::numeric_limits<double>::infinity();
        return std::sqrt(value/rhs_square_);
    }
    static bool accepts(double relative) { return std::isfinite(relative) && relative>=0 && relative<=0.1; }
};
}
