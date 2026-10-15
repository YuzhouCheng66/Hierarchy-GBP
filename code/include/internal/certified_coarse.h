#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <Eigen/Core>

namespace slam {
struct CoarseReuseCertificate {
    bool accepted = false;
    int iterations = 0;
    int preconditioner_calls = 0;
    int matvec_calls = 0;
    double relative_residual = std::numeric_limits<double>::infinity();
};

struct CoarseWorkEstimate {
    double factor_work = 0;
    double solve_work = 0;
    double matvec_work = 0;

    double extraWork(const CoarseReuseCertificate& certificate) const {
        return std::max(0, certificate.preconditioner_calls - 1) * solve_work +
            certificate.matvec_calls * matvec_work;
    }
    double reuseRatio(int cycles) const {
        return 6. * cycles * (solve_work + matvec_work) /
            std::max(factor_work + cycles * solve_work, 1.);
    }
};

template<class SparseLower>
CoarseWorkEstimate estimateCoarseWork(const SparseLower& lower,Eigen::Index matrix_nonzeros) {
    CoarseWorkEstimate estimate;
    for(Eigen::Index j=0;j<lower.outerSize();++j) {
        const double count=1.+lower.outerIndexPtr()[j+1]-lower.outerIndexPtr()[j];
        estimate.factor_work+=count*count;
    }
    estimate.solve_work=4.*lower.nonZeros()+lower.rows();
    estimate.matvec_work=2.*matrix_nonzeros+12.*lower.rows();
    return estimate;
}

template<class SparseLower>
double coarseReuseWorkRatio(const SparseLower& lower,Eigen::Index matrix_nonzeros,int cycles) {
    // Conservative six-step cap versus one fresh factor plus c backsolves.
    return estimateCoarseWork(lower,matrix_nonzeros).reuseRatio(cycles);
}

// Only a coarse-space solve. An old factor is a preconditioner, never an
// unchecked replacement for the current matrix. Rejection requires refactoring.
template<class Matrix, class Solve>
CoarseReuseCertificate certifiedCoarsePcg(const Matrix& a, double ridge,
    const Eigen::VectorXd& b, Solve precondition,
    Eigen::VectorXd& x, Eigen::VectorXd& r, Eigen::VectorXd& z,
    Eigen::VectorXd& p, Eigen::VectorXd& ap) {
    constexpr int max_iterations=6;
    constexpr double relative_tolerance=1e-6;
    CoarseReuseCertificate result;
    x.setZero(b.size()); r=b;
    const double bnorm=b.norm();
    if(!std::isfinite(bnorm)) return result;
    if(bnorm==0) { result.accepted=true; result.relative_residual=0; return result; }
    precondition(r,z); ++result.preconditioner_calls; p=z;
    double rz=r.dot(z);
    for(int i=0;i<max_iterations;++i) {
        if(!(rz>0) || !std::isfinite(rz)) break;
        ap.noalias()=a*p;
        ++result.matvec_calls;
        if(ridge!=0) ap.noalias()+=ridge*p;
        const double curvature=p.dot(ap);
        if(!(curvature>0) || !std::isfinite(curvature)) break;
        const double alpha=rz/curvature;
        if(!std::isfinite(alpha)) break;
        x.noalias()+=alpha*p; r.noalias()-=alpha*ap;
        result.iterations=i+1;
        if(r.norm()<=relative_tolerance*bnorm) break;
        if(i+1==max_iterations) break;
        precondition(r,z); ++result.preconditioner_calls;
        const double next=r.dot(z);
        if(!(next>0) || !std::isfinite(next)) break;
        p=z+(next/rz)*p; rz=next;
    }
    // A true residual protects against recurrence drift and stale-model reuse.
    ap.noalias()=a*x;
    ++result.matvec_calls;
    if(ridge!=0) ap.noalias()+=ridge*x;
    r=b-ap;
    result.relative_residual=r.norm()/bnorm;
    result.accepted=std::isfinite(result.relative_residual) &&
        result.relative_residual<=relative_tolerance && x.allFinite();
    return result;
}
}
