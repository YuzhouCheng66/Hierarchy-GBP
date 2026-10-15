#include "internal/se2_residual.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {
    if(!condition) throw std::runtime_error(message);
}
slam::SyntheticSE2Problem problem() {
    slam::SyntheticSE2Problem p;
    for(int i=0;i<140;++i) p.init_poses.emplace_back(i*1.01,.002*i,.0001*i);
    p.gt_poses=p.init_poses;
    p.anchor_pose=p.init_poses[0];
    p.anchor_information=Eigen::Matrix3d::Identity()*1000;
    for(int i=0;i<139;++i) {
        slam::SyntheticSE2Edge edge;
        edge.i=i; edge.j=i+1; edge.measurement=Eigen::Vector3d(1,0,0);
        edge.information=Eigen::Matrix3d::Identity()*10;
        p.edges.push_back(edge);
    }
    return p;
}
void sweeps(slam::SyntheticSE2PackedResidualWorkspace& w,int count,int threads) {
    double f=0,v=0;
    slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(w,count,f,v,threads);
}
double relative(const Eigen::VectorXd& a,const Eigen::VectorXd& b) {
    return (a-b).norm()/std::max(1.,b.norm());
}
}

int main() {
    try {
        {
            double diagonal[6]={4,0,0,9,0,16},previous[6]={1,.1,.2,2,.3,3};
            double next[6]={1.01,.12,.2,2.02,.29,2.99};
            const double reference=slam::normalizedSE2PrecisionResidual(next,previous,diagonal);
            constexpr int row[6]={0,1,2,1,2,2},col[6]={0,0,0,1,1,2};
            const double coordinate_scale[3]={1e-3,7,1e3};
            for(int i=0;i<6;++i) {
                const double scale=coordinate_scale[row[i]]*coordinate_scale[col[i]];
                diagonal[i]*=scale; previous[i]*=scale; next[i]*=scale;
            }
            require(std::abs(reference-slam::normalizedSE2PrecisionResidual(next,previous,diagonal))<1e-14,
                    "precision residual depends on coordinate units");
            next[2]=std::numeric_limits<double>::quiet_NaN();
            require(!std::isfinite(slam::normalizedSE2PrecisionResidual(next,previous,diagonal)),
                    "nonfinite precision accepted");
            next[2]=1e300;
            require(!std::isfinite(slam::normalizedSE2PrecisionResidual(next,previous,diagonal)),
                    "overflowed precision norm accepted");
            for(double invalid:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::quiet_NaN()}) {
                require(!slam::finiteSE2PrecisionValue(invalid),"nonfinite bit check failed");
                for(int k=0;k<6;++k) {
                    double current[6]={1,0,0,1,0,1},old[6]={1,0,0,1,0,1},diag[6]={1,0,0,1,0,1};
                    current[k]=invalid;
                    require(!std::isfinite(slam::normalizedSE2PrecisionResidual(current,old,diag)),
                            "nonfinite current element accepted");
                    current[k]=0; old[k]=invalid;
                    require(!std::isfinite(slam::normalizedSE2PrecisionResidual(current,old,diag)),
                            "nonfinite previous element accepted");
                }
            }
        }
        const auto p=problem();
        auto initial=slam::buildSyntheticSE2PackedResidualWorkspace(p,.1);
        slam::relinearizeSyntheticSE2PackedResidualWorkspace(initial,p,p.init_poses);
        slam::initializeSE2PackedMessagesFromFactors(initial,2,1);
        {
            auto pending=initial;
            pending.adaptive_precision=true;
            sweeps(pending,5,1);
            require(pending.precision_checks==1 && pending.precision_stable_checks==0,
                    "expected initial nonstationary precision");
            require(pending.precision_check_interval==10 && pending.next_precision_check_sweep==15,
                    "failed precision check did not back off");
            sweeps(pending,9,1);
            require(pending.precision_checks==1 && pending.full_precision_sweeps==14 && pending.eta_only_sweeps==0,
                    "check backoff skipped GBP work or checked prematurely");
        }
        auto full=initial;
        sweeps(full,300,1);
        const auto reference=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(full,1);
        Eigen::VectorXd first;
        for(int threads:{1,16}) for(bool split:{false,true}) {
            auto adaptive=initial;
            adaptive.adaptive_precision=true;
            if(split) { sweeps(adaptive,47,threads); sweeps(adaptive,53,threads); sweeps(adaptive,200,threads); }
            else sweeps(adaptive,300,threads);
            const auto x=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(adaptive,threads);
            std::cout << "threads=" << threads << " split=" << split
                      << " full=" << adaptive.full_precision_sweeps << " eta=" << adaptive.eta_only_sweeps
                      << " x_rel=" << relative(x,reference) << '\n';
            require(adaptive.precision_freezes>0 && adaptive.eta_only_sweeps>0,"precision never froze");
            require(adaptive.full_precision_sweeps+adaptive.eta_only_sweeps==300,"sweep count mismatch");
            require(adaptive.fixedeta_full_lambda_sweeps==static_cast<unsigned>(adaptive.full_precision_sweeps),
                    "full sweep counters disagree");
            require(adaptive.first_precision_freeze_sweep>=15 && adaptive.precision_checks>=3,"premature freeze");
            require(relative(x,reference)<1e-6,"adaptive changed converged Gaussian mean");
            if(first.size()==0) first=x;
            require(relative(x,first)<1e-10,"executor or call splitting changed adaptive result");
            const int full_before=adaptive.full_precision_sweeps;
            const int checks_before=adaptive.precision_checks;
            for(auto& f:adaptive.binary_factors) { f.diag0_lam6[0]+=1; f.diag1_lam6[0]+=1; }
            const int until_check=slam::SE2PrecisionPolicy::frozen_check_period-
                (adaptive.sweeps_since_relinearize-adaptive.last_precision_check_sweep);
            sweeps(adaptive,until_check,threads);
            require(adaptive.precision_thaws>0 && !adaptive.precision_frozen,"changed canonical precision did not thaw");
            require(adaptive.precision_checks==checks_before+1 && adaptive.full_precision_sweeps>full_before,
                    "frozen recheck was not a real full sweep");
            slam::relinearizeSyntheticSE2PackedResidualWorkspace(adaptive,p,p.init_poses);
            require(!adaptive.precision_frozen && adaptive.precision_stable_checks==0 &&
                    adaptive.precision_checks==0 && adaptive.sweeps_since_relinearize==0,
                    "relinearization retained precision certificate");
            require(adaptive.fixed_eta_maps_all_valid==0,"relinearization retained eta maps");
            require(!adaptive.precision_scales_ready,"relinearization retained normalization scales");
            require(adaptive.next_precision_check_sweep==slam::SE2PrecisionPolicy::check_period,
                    "relinearization retained check backoff");
        }
        std::cout << "PASS adaptive SE2 precision\n";
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
