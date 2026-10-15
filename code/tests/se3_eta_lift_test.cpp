#include "internal/se3_residual.h"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using Mat6=Eigen::Matrix<double,6,6>;
using Vec6=Eigen::Matrix<double,6,1>;
using Workspace=slam::SyntheticSE3PackedSoAWorkspace;
static void require(bool b,const char* s) { if(!b) throw std::runtime_error(s); }
static Mat6 unpack(const double* p) {
    Mat6 a;
    for(int c=0;c<6;++c) for(int r=0;r<=c;++r) a(r,c)=a(c,r)=p[c*(c+1)/2+r];
    return a;
}
static slam::SyntheticSE3Problem problem(int n) {
    slam::SyntheticSE3Problem p; p.init_poses.resize(n);
    for(int i=0;i<n;++i) p.init_poses[i].t=Eigen::Vector3d(1.07*i,.03*i,-.02*i);
    p.anchor_pose=p.init_poses.front(); p.anchor_information=1000.*Mat6::Identity();
    for(int i=0;i<n-1;++i) {
        slam::SyntheticSE3Edge e; e.i=i; e.j=i+1; e.measurement.t=Eigen::Vector3d(1,0,0);
        e.information=10.*Mat6::Identity(); p.edges.push_back(e);
    }
    if(n>2) { auto e=p.edges.front(); e.j=n-1; e.measurement.t=Eigen::Vector3d(n-1,0,0); p.edges.push_back(e); }
    return p;
}
static Workspace initialized(int n,bool adaptive) {
    auto p=problem(n); auto graph=slam::buildLinearizedResidualGraph(p,p.init_poses);
    auto w=slam::buildSyntheticSE3PackedSoAWorkspace(p);
    slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(w,graph);
    slam::initializeBalancedSE3PackedMessages(w,1);
    w.adaptive_precision=adaptive; w.persistent_sweeps=true;
    slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(w,100,1,-1,.3,0);
    return w;
}
static void unchanged(const Workspace& a,const Workspace& b) {
    require(a.binary_eta0==b.binary_eta0 && a.binary_eta1==b.binary_eta1 &&
            a.binary_diag0_lam21==b.binary_diag0_lam21 && a.binary_diag1_lam21==b.binary_diag1_lam21 &&
            a.binary_cross01_lam36==b.binary_cross01_lam36 && a.binary_cross10_lam36==b.binary_cross10_lam36 &&
            a.prior_eta==b.prior_eta && a.prior_lam21==b.prior_lam21 &&
            a.unary_eta==b.unary_eta && a.unary_lam21==b.unary_lam21 &&
            a.unary_msg_eta==b.unary_msg_eta && a.unary_msg_lam21==b.unary_msg_lam21,
            "canonical/prior/unary changed");
    require(a.belief_lam21==b.belief_lam21 && a.binary_msg_lam21==b.binary_msg_lam21 &&
            a.fixed_eta_map0_lam36==b.fixed_eta_map0_lam36 && a.fixed_eta_map1_lam36==b.fixed_eta_map1_lam36 &&
            a.fixed_eta_map_valid==b.fixed_eta_map_valid && a.precision_frozen==b.precision_frozen &&
            a.precision_checks==b.precision_checks && a.sweeps_since_relinearize==b.sweeps_since_relinearize,
            "precision/caches/counters changed");
}
int main() {
    try {
        for(int n:{2,5,140}) for(bool adaptive:{false,true}) for(bool weighted:{false,true}) {
            auto original=initialized(n,adaptive);
            const Eigen::VectorXd initial=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(original,1);
            Eigen::VectorXd delta=Eigen::VectorXd::LinSpaced(6*n,-.03,.04);
            auto serial=original;
            slam::applyMeanDeltaSyntheticSE3PackedSoAWorkspace(serial,delta,1);
            require(slam::measureSE3PackedEtaMismatch(serial)>1e-4,"test must create a real mismatch");
            auto parallel=serial;
            const auto target=serial.belief_eta;
            auto expected=original.binary_msg_eta;
            for(int v=0;v<n;++v) {
                Vec6 sum=Eigen::Map<const Vec6>(original.belief_eta.data()+6*v);
                for(int k=original.binary_offsets[v];k<original.binary_offsets[v+1];++k) {
                    const int s=original.binary_slot_ids[k];
                    if(weighted) {
                        const Vec6 d=unpack(original.binary_msg_lam21.data()+21*s)*delta.segment<6>(6*v);
                        Eigen::Map<Vec6>(expected.data()+6*s)+=d; sum+=d;
                    }
                }
                const Vec6 correction=(Eigen::Map<const Vec6>(target.data()+6*v)-sum)/
                    double(original.binary_offsets[v+1]-original.binary_offsets[v]);
                for(int k=original.binary_offsets[v];k<original.binary_offsets[v+1];++k)
                    Eigen::Map<Vec6>(expected.data()+6*original.binary_slot_ids[k])+=correction;
            }
            slam::liftSE3PackedMeanCorrection(serial,delta,weighted,1);
            slam::liftSE3PackedMeanCorrection(parallel,delta,weighted,16);
            unchanged(original,serial);
            require(serial.binary_msg_eta==parallel.binary_msg_eta && serial.belief_eta==parallel.belief_eta,
                    "parallel incoming ownership changed result");
            require(slam::measureSE3PackedEtaMismatch(serial)==0.,"belief is not the actual message sum");
            require((Eigen::Map<const Eigen::VectorXd>(expected.data(),expected.size())-
                    Eigen::Map<const Eigen::VectorXd>(serial.binary_msg_eta.data(),serial.binary_msg_eta.size())).norm()<1e-9,
                    "lift differs from independent formula");
            Eigen::VectorXd solved(6*n);
            for(int v=0;v<n;++v) solved.segment<6>(6*v)=unpack(serial.belief_lam21.data()+21*v).ldlt().solve(
                Eigen::Map<const Vec6>(serial.belief_eta.data()+6*v));
            require((solved-initial-delta).norm()/std::max(1.,solved.norm())<1e-9,"target mean not preserved");
            const auto zero_state=serial;
            slam::liftSE3PackedMeanCorrection(serial,Eigen::VectorXd::Zero(6*n),weighted,1);
            require(serial.binary_msg_eta==zero_state.binary_msg_eta,"zero lift changed messages");
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(serial,10,1,-1,.3,0);
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(parallel,10,16,-1,.3,0);
            const auto xs=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(serial,1);
            const auto xp=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(parallel,16);
            require(xs.allFinite() && xp.allFinite() && (xs-xp).norm()/std::max(1.,xs.norm())<1e-7,"post-lift GBP mismatch");
        }
        auto isolated=initialized(1,false);
        slam::liftSE3PackedMeanCorrection(isolated,Eigen::VectorXd::Zero(6),false,1);
        const Eigen::VectorXd d=Eigen::VectorXd::Ones(6);
        slam::applyMeanDeltaSyntheticSE3PackedSoAWorkspace(isolated,d,1);
        bool rejected=false;
        try { slam::liftSE3PackedMeanCorrection(isolated,d,false,1); } catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"isolated variable cannot silently discard a correction");
        std::cout<<"eta lift invariants and1/16thread tests passed\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
