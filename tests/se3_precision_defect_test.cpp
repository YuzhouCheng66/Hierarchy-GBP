#include "internal/se3_residual.h"
#include "internal/se3_residual_certificate.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
using M=Eigen::Matrix<double,6,6>;
using R=Eigen::Matrix<double,6,6,Eigen::RowMajor>;
using V=Eigen::Matrix<double,6,1>;
using W=slam::SyntheticSE3PackedSoAWorkspace;
static void need(bool x,const char* s) { if(!x) throw std::runtime_error(s); }
static M unpack(const double* p) {
    M m; for(int c=0;c<6;++c) for(int r=0;r<=c;++r) m(r,c)=m(c,r)=p[c*(c+1)/2+r]; return m;
}
static W setup(int n,bool loop) {
    slam::SyntheticSE3Problem p; p.init_poses.resize(n);
    for(int i=0;i<n;++i) p.init_poses[i].t=Eigen::Vector3d(1.07*i,.03*i,-.02*i);
    p.anchor_pose=p.init_poses[0]; p.anchor_information=1000.*M::Identity();
    M a=M::Identity(); for(int i=0;i<5;++i) a(i,i+1)=.17;
    auto edge=[&](int i,int j) {
        slam::SyntheticSE3Edge e; e.i=i; e.j=j; e.measurement.t=Eigen::Vector3d(j-i,0,0);
        e.information=10.*a.transpose()*a; p.edges.push_back(e);
    };
    for(int i=0;i<n-1;++i) edge(i,i+1);
    if(loop && n>2) edge(0,n-1);
    auto graph=slam::buildLinearizedResidualGraph(p,p.init_poses);
    auto w=slam::buildSyntheticSE3PackedSoAWorkspace(p);
    slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(w,graph);
    slam::initializeBalancedSE3PackedMessages(w,1);
    return w;
}
static void canonical(const W& a,const W& b) {
    need(a.binary_eta0==b.binary_eta0 && a.binary_eta1==b.binary_eta1 &&
         a.binary_diag0_lam21==b.binary_diag0_lam21 && a.binary_diag1_lam21==b.binary_diag1_lam21 &&
         a.binary_cross01_lam36==b.binary_cross01_lam36 && a.binary_cross10_lam36==b.binary_cross10_lam36 &&
         a.prior_eta==b.prior_eta && a.prior_lam21==b.prior_lam21 && a.unary_eta==b.unary_eta &&
         a.unary_lam21==b.unary_lam21 && a.unary_msg_eta==b.unary_msg_eta &&
         a.unary_msg_lam21==b.unary_msg_lam21 && a.belief_lam21==b.belief_lam21 &&
         a.binary_msg_lam21==b.binary_msg_lam21,"canonical/precision modified");
}
static Eigen::VectorXd direct(const W& w,Eigen::MatrixXd& h,Eigen::VectorXd& b) {
    int n=w.num_vars; h=Eigen::MatrixXd::Zero(6*n,6*n); b=Eigen::VectorXd::Zero(6*n);
    for(int i=0;i<n;++i) {
        h.block<6,6>(6*i,6*i)=unpack(w.prior_lam21.data()+21*i);
        b.segment<6>(6*i)=Eigen::Map<const V>(w.prior_eta.data()+6*i);
    }
    for(int f=0;f<w.num_unary_factors;++f) {
        int i=w.unary_var_id[f]; h.block<6,6>(6*i,6*i)+=unpack(w.unary_lam21.data()+21*f);
        b.segment<6>(6*i)+=Eigen::Map<const V>(w.unary_eta.data()+6*f);
    }
    for(int f=0;f<w.num_binary_factors;++f) {
        int i=w.binary_var0_id[f],j=w.binary_var1_id[f];
        h.block<6,6>(6*i,6*i)+=unpack(w.binary_diag0_lam21.data()+21*f);
        h.block<6,6>(6*j,6*j)+=unpack(w.binary_diag1_lam21.data()+21*f);
        h.block<6,6>(6*i,6*j)+=Eigen::Map<const M>(w.binary_cross01_lam36.data()+36*f);
        h.block<6,6>(6*j,6*i)+=Eigen::Map<const M>(w.binary_cross10_lam36.data()+36*f);
        b.segment<6>(6*i)+=Eigen::Map<const V>(w.binary_eta0.data()+6*f);
        b.segment<6>(6*j)+=Eigen::Map<const V>(w.binary_eta1.data()+6*f);
    }
    return h.ldlt().solve(b);
}
int main() {
    try {
        for(int n:{2,5,8}) for(bool loop:{false,true}) {
            auto original=setup(n,loop),w=original;
            slam::prepareSE3PrecisionDefect(w,1);
            auto expected=w.binary_msg_eta;
            for(int f=0;f<w.num_binary_factors;++f) {
                const int i=w.binary_var0_id[f],j=w.binary_var1_id[f];
                const V a=Eigen::Map<const V>(w.binary_eta0.data()+6*f)-
                    Eigen::Map<const M>(w.binary_cross01_lam36.data()+36*f)*Eigen::Map<const V>(w.mu.data()+6*j);
                const V b=Eigen::Map<const V>(w.binary_eta1.data()+6*f)-
                    Eigen::Map<const M>(w.binary_cross10_lam36.data()+36*f)*Eigen::Map<const V>(w.mu.data()+6*i);
                const V old0=Eigen::Map<const V>(w.binary_msg_eta.data()+12*f),old1=Eigen::Map<const V>(w.binary_msg_eta.data()+12*f+6);
                Eigen::Map<V>(expected.data()+12*f)=.3*old0+.7*(a+Eigen::Map<const R>(w.defect_map0.data()+36*f)*(old1-b));
                Eigen::Map<V>(expected.data()+12*f+6)=.3*old1+.7*(b+Eigen::Map<const R>(w.defect_map1.data()+36*f)*(old0-a));
            }
            slam::precisionDefectSE3Iterations(w,1,1,.3,0.);
            need((Eigen::Map<const Eigen::VectorXd>(expected.data(),expected.size())-
                  Eigen::Map<const Eigen::VectorXd>(w.binary_msg_eta.data(),expected.size())).norm()<1e-10,"one sweep formula");
            canonical(original,w);
            Eigen::MatrixXd h; Eigen::VectorXd b; const auto exact=direct(w,h,b);
            slam::SE3ResidualCertificate certificate,parallel_certificate;
            certificate.initialize(w,1); parallel_certificate.initialize(w,16);
            const Eigen::VectorXd residual=Eigen::VectorXd::LinSpaced(b.size(),-.3,.8);
            double numerator=0,denominator=0;
            for(int i=0;i<n;++i) {
                const M diag=h.block<6,6>(6*i,6*i);
                const V r=residual.segment<6>(6*i),rhs=b.segment<6>(6*i);
                numerator+=r.dot(diag.ldlt().solve(r));
                denominator+=rhs.dot(diag.ldlt().solve(rhs));
            }
            const double measured=certificate.relativeResidual(residual);
            need(std::abs(measured-std::sqrt(numerator/denominator))<1e-10,"canonical block residual norm");
            need(measured==parallel_certificate.relativeResidual(residual),"residual metric thread dependent");
            need(std::abs(certificate.relativeResidual(b)-1.)<1e-12,"rhs reference is not canonical gradient");
            need(certificate.relativeResidual(.05*b)<.1 && certificate.relativeResidual(.2*b)>.1,"residual threshold");
            auto changed=w;
            for(auto& v:changed.belief_lam21) v*=10;
            for(auto& v:changed.binary_msg_lam21) v*=.1;
            slam::SE3ResidualCertificate changed_certificate; changed_certificate.initialize(changed,1);
            need(measured==changed_certificate.relativeResidual(residual),"criterion depends on GBP precision");
            need(!slam::SE3ResidualCertificate::accepts(std::numeric_limits<double>::infinity()) &&
                 !slam::SE3ResidualCertificate::accepts(-1.),"invalid residual accepted");
            canonical(original,w);
            for(int f=0;f<w.num_binary_factors;++f) {
                const int i=w.binary_var0_id[f],j=w.binary_var1_id[f];
                Eigen::Map<V>(w.binary_msg_eta.data()+12*f)=Eigen::Map<const V>(w.binary_eta0.data()+6*f)-
                    Eigen::Map<const M>(w.binary_cross01_lam36.data()+36*f)*exact.segment<6>(6*j);
                Eigen::Map<V>(w.binary_msg_eta.data()+12*f+6)=Eigen::Map<const V>(w.binary_eta1.data()+6*f)-
                    Eigen::Map<const M>(w.binary_cross10_lam36.data()+36*f)*exact.segment<6>(6*i);
            }
            slam::recomputeSE3PackedBeliefs(w,1);
            // Preparation recomputes mu with the exact unjittered belief inverse.
            w.defect_ready=false; slam::prepareSE3PrecisionDefect(w,1);
            for(double& q:w.defect_map0) q*=.83;
            for(double& q:w.defect_map1) q*=.83;
            slam::precisionDefectSE3Iterations(w,20,1,.3,2.);
            auto x=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(w,1);
            need((h*x-b).norm()/std::max(1.,b.norm())<1e-9,"inexact-map fixed point biased");
            canonical(original,w);
            w=original; slam::precisionDefectSE3Iterations(w,10000,1,.3,2.);
            x=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(w,1);
            need((h*x-b).norm()/std::max(1.,b.norm())<1e-7,"small-system iteration failed to converge");
            need(w.defect_mean_sweeps==10000 && w.full_precision_sweeps==0 && w.eta_only_sweeps==0,"dishonest work counters");
        }
        auto s=setup(140,true),p=s;
        slam::precisionDefectSE3Iterations(s,100,1,.3,2.);
        slam::precisionDefectSE3Iterations(p,37,16,.3,2.);
        slam::precisionDefectSE3Iterations(p,63,16,.3,2.);
        need(s.binary_msg_eta==p.binary_msg_eta && s.mu==p.mu && s.belief_eta==p.belief_eta,"parallel/split call differs");
        const auto before=s;
        Eigen::VectorXd delta=Eigen::VectorXd::LinSpaced(6*s.num_vars,-.01,.03);
        slam::applyMeanDeltaSyntheticSE3PackedSoAWorkspace(s,delta,1);
        slam::liftSE3PackedMeanCorrection(s,delta,true,1);
        slam::precisionDefectSE3Iterations(s,10,1,.3,2.);
        need(s.defect_map_builds==before.defect_map_builds,"coarse correction rebuilt unchanged maps");
        need(slam::measureSE3PackedEtaMismatch(s)==0.,"message sum not belief");
        canonical(before,s);
        auto wrong=setup(2,false); wrong.binary_msg_lam21[0]+=1.; bool rejected=false;
        try { slam::prepareSE3PrecisionDefect(wrong,1); } catch(const std::runtime_error&) { rejected=true; }
        need(rejected,"non-diagonal seed silently accepted");
        std::cout<<"precision-defect fixed means, inverse perturbation, invariants,1/16 ownership passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
