#include "internal/se3_residual.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace slam {
namespace {
using W = SyntheticSE3PackedSoAWorkspace;
using M6 = Eigen::Matrix<double,6,6>;
using R6 = Eigen::Matrix<double,6,6,Eigen::RowMajor>;
using V6 = Eigen::Matrix<double,6,1>;
M6 unpack(const double* p) {
    M6 m;
    for(int c=0;c<6;++c) for(int r=0;r<=c;++r) m(r,c)=m(c,r)=p[c*(c+1)/2+r];
    return m;
}
bool inverse(const M6& a,M6& out,double& residual) {
    if(!a.allFinite() || (a.diagonal().array()<=0.).any()) return false;
    const V6 s=a.diagonal().array().sqrt().inverse();
    const M6 scaled=s.asDiagonal()*a*s.asDiagonal();
    Eigen::LLT<M6> llt(scaled);
    if(llt.info()!=Eigen::Success) return false;
    const M6 inv=llt.solve(M6::Identity());
    residual=(scaled*inv-M6::Identity()).cwiseAbs().maxCoeff();
    out=s.asDiagonal()*inv*s.asDiagonal();
    return out.allFinite() && residual<=1e-7;
}
void rowProduct(const double* a,const double* x,double* y) noexcept {
    for(int r=0;r<6;++r) {
        const double* p=a+6*r;
        y[r]=p[0]*x[0]+p[1]*x[1]+p[2]*x[2]+p[3]*x[3]+p[4]*x[4]+p[5]*x[5];
    }
}
void columnProduct(const double* a,const double* x,double* y) noexcept {
    for(int r=0;r<6;++r)
        y[r]=a[r]*x[0]+a[r+6]*x[1]+a[r+12]*x[2]+a[r+18]*x[3]+a[r+24]*x[4]+a[r+30]*x[5];
}
int finishMessage(const double* old,const double* reference,double* value,double damping,double limit) noexcept {
    double diff=0,old2=0,ref2=0;
    for(int d=0;d<6;++d) {
        value[d]=(1.-damping)*value[d]+damping*old[d];
        const double z=value[d]-old[d]; diff+=z*z; old2+=old[d]*old[d]; ref2+=reference[d]*reference[d];
    }
    if(limit<=0) return 0;
    const double scale=std::max(1.,std::max(std::sqrt(old2),std::sqrt(ref2)));
    if(diff<=limit*limit*scale*scale) return 0;
    const double relative=std::sqrt(diff)/scale;
    if(!std::isfinite(relative) || relative<=limit) return 0;
    const double step=limit/std::max(relative,1e-300);
    for(int d=0;d<6;++d) value[d]=old[d]+step*(value[d]-old[d]);
    return 1;
}
int factorPass(W& w,int f,double damping,double limit) noexcept {
    const int i=w.binary_var0_id[f],j=w.binary_var1_id[f];
    const double* hi=w.binary_eta0.data()+6*f,*hj=w.binary_eta1.data()+6*f;
    double cross0[6],cross1[6],target0[6],target1[6],defect0[6],defect1[6];
    double old0[6],old1[6],new0[6],new1[6];
    std::copy_n(w.binary_msg_eta.data()+12*f,6,old0);
    std::copy_n(w.binary_msg_eta.data()+12*f+6,6,old1);
    columnProduct(w.binary_cross01_lam36.data()+36*f,w.mu.data()+6*j,cross0);
    columnProduct(w.binary_cross10_lam36.data()+36*f,w.mu.data()+6*i,cross1);
    // L_fi=A_fi: L_fi*x_i+r_fi = h_fi-H_fij*x_j.
    // Read both old messages before either write; different factors own disjoint slots.
    for(int d=0;d<6;++d) {
        target0[d]=hi[d]-cross0[d]; target1[d]=hj[d]-cross1[d];
        defect0[d]=old0[d]-target0[d]; defect1[d]=old1[d]-target1[d];
    }
    rowProduct(w.defect_map0.data()+36*f,defect1,new0);
    rowProduct(w.defect_map1.data()+36*f,defect0,new1);
    for(int d=0;d<6;++d) { new0[d]+=target0[d]; new1[d]+=target1[d]; }
    const int clamps=finishMessage(old0,hi,new0,damping,limit)+finishMessage(old1,hj,new1,damping,limit);
    std::copy_n(new0,6,w.binary_msg_eta.data()+12*f);
    std::copy_n(new1,6,w.binary_msg_eta.data()+12*f+6);
    return clamps;
}
bool variablePass(W& w,int i) noexcept {
    double eta[6]; std::copy_n(w.prior_eta.data()+6*i,6,eta);
    for(int p=w.unary_offsets[i];p<w.unary_offsets[i+1];++p) {
        const double* m=w.unary_msg_eta.data()+6*w.unary_ids[p];
        for(int d=0;d<6;++d) eta[d]+=m[d];
    }
    for(int p=w.binary_offsets[i];p<w.binary_offsets[i+1];++p) {
        const double* m=w.binary_msg_eta.data()+6*w.binary_slot_ids[p];
        for(int d=0;d<6;++d) eta[d]+=m[d];
    }
    double mu[6]; rowProduct(w.defect_belief_inverse.data()+36*i,eta,mu);
    std::copy_n(eta,6,w.belief_eta.data()+6*i);
    std::copy_n(mu,6,w.mu.data()+6*i); w.mu_valid[i]=1;
    for(int d=0;d<6;++d) if(!std::isfinite(eta[d]) || !std::isfinite(mu[d])) return false;
    return true;
}
}

void prepareSE3PrecisionDefect(W& w,int num_threads) {
    if(w.defect_ready) return;
    const auto start=std::chrono::steady_clock::now();
    const int threads=std::max(1,num_threads),n=w.num_vars,nf=w.num_binary_factors;
    for(int f=0;f<nf;++f) for(int q=0;q<21;++q)
        if(w.binary_msg_lam21[42*f+q]!=w.binary_diag0_lam21[21*f+q] ||
           w.binary_msg_lam21[42*f+21+q]!=w.binary_diag1_lam21[21*f+q])
            throw std::runtime_error("Precision-defect diagonal mode requires exact factor-diagonal seeds");
    w.defect_belief_inverse.resize(36*n); w.defect_map0.resize(36*nf); w.defect_map1.resize(36*nf);
    W::AlignedDoubles cavity_inverse(36*n);
    int failures=0; double max_residual=0;
    std::vector<double> residuals(n,0.);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1) reduction(+:failures)
    for(int i=0;i<n;++i) {
        const M6 b=unpack(w.belief_lam21.data()+21*i);
        M6 inv,ci; double defect=0;
        if(!inverse(b,inv,defect)) { ++failures; continue; }
        residuals[i]=defect;
        Eigen::Map<R6>(w.defect_belief_inverse.data()+36*i)=inv;
        M6 c=b;
        c.diagonal().array()+=std::sqrt(std::numeric_limits<double>::epsilon())*b.diagonal().array();
        if(!inverse(c,ci,defect)) { ++failures; continue; }
        residuals[i]=std::max(residuals[i],defect);
        Eigen::Map<M6>(cavity_inverse.data()+36*i)=ci;
    }
    if(failures) throw std::runtime_error("Precision-defect belief/cavity inverse failed residual certificate");
    for(double value:residuals) max_residual=std::max(max_residual,value);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1) reduction(+:failures)
    for(int f=0;f<nf;++f) {
        const M6 a=Eigen::Map<const M6>(w.binary_cross01_lam36.data()+36*f)*
            Eigen::Map<const M6>(cavity_inverse.data()+36*w.binary_var1_id[f]);
        const M6 b=Eigen::Map<const M6>(w.binary_cross10_lam36.data()+36*f)*
            Eigen::Map<const M6>(cavity_inverse.data()+36*w.binary_var0_id[f]);
        if(!a.allFinite() || !b.allFinite()) { ++failures; continue; }
        Eigen::Map<R6>(w.defect_map0.data()+36*f)=a;
        Eigen::Map<R6>(w.defect_map1.data()+36*f)=b;
    }
    if(failures) throw std::runtime_error("Nonfinite precision-defect map");
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1) reduction(+:failures)
    for(int i=0;i<n;++i) if(!variablePass(w,i)) ++failures;
    if(failures) throw std::runtime_error("Nonfinite precision-defect initial mean");
    w.defect_ready=true; w.defect_map_builds+=2*nf;
    w.defect_inverse_residual=max_residual;
    w.defect_build_sec+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}

void precisionDefectSE3Iterations(W& w,int sweeps,int num_threads,double damping,double update_limit) {
    if(sweeps<=0) return;
    if(!(damping>=0 && damping<1)) throw std::runtime_error("Invalid precision-defect damping");
    prepareSE3PrecisionDefect(w,num_threads);
    const int threads=std::max(1,num_threads); int failures=0,clamps=0;
    if(threads>1) {
        #pragma omp parallel num_threads(threads) reduction(+:failures,clamps)
        {
            for(int sweep=0;sweep<sweeps;++sweep) {
                #pragma omp for schedule(static)
                for(int f=0;f<w.num_binary_factors;++f) clamps+=factorPass(w,f,damping,update_limit);
                #pragma omp for schedule(static)
                for(int i=0;i<w.num_vars;++i) if(!variablePass(w,i)) ++failures;
            }
        }
    } else {
        for(int sweep=0;sweep<sweeps;++sweep) {
            for(int f=0;f<w.num_binary_factors;++f) clamps+=factorPass(w,f,damping,update_limit);
            for(int i=0;i<w.num_vars;++i) if(!variablePass(w,i)) ++failures;
        }
    }
    w.defect_mean_sweeps+=sweeps; w.sweeps_since_relinearize+=sweeps; w.defect_clamps+=clamps;
    if(failures) throw std::runtime_error("Nonfinite precision-defect iteration");
}
}
