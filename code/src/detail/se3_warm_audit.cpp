#include "internal/se3_precision_warm.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <Eigen/SparseCholesky>

namespace slam {
void auditSE3PrecisionWarm(const std::string& path,const gbp::FactorGraph& graph,
    const SyntheticSE3PackedSoAWorkspace& initial,const SE3PrecisionWarmState& previous,
    const SE3PoseVector& current,int threads) {
    const auto joint=graph.jointDistributionInfSparse();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> direct(joint.lam);
    if(direct.info()!=Eigen::Success) throw std::runtime_error("Warm audit reference factorization failed");
    const Eigen::VectorXd exact=direct.solve(joint.eta);
    const double rhs_norm=std::max(joint.eta.norm(),1e-30);
    const double reference_residual=(joint.eta-joint.lam*exact).norm()/rhs_norm;
    if(!exact.allFinite() || reference_residual>1e-6) throw std::runtime_error("Warm audit reference residual too large");
    const double minimum=.5*exact.dot(joint.lam*exact)-joint.eta.dot(exact);
    std::ofstream out(path);
    if(!out) throw std::runtime_error("Cannot write warm audit");
    out << std::setprecision(17);
    for(const std::string mode:{"cold","warm-gradient","warm-bounded","warm-transport"}) {
        auto w=initial;
        w.adaptive_precision=true;
        w.persistent_sweeps=true;
        if(mode!="cold") initializeWarmSE3Precision(w,previous,current,mode=="warm-bounded",threads,mode=="warm-transport");
        for(int sweep=0;sweep<=250;sweep+=5) {
            if(sweep) synchronousIterationsSyntheticSE3PackedSoAWorkspace(w,5,threads,-1,.3,2.);
            const auto x=stackedMeanVectorSyntheticSE3PackedSoAWorkspace(w,threads);
            const Eigen::VectorXd product=joint.lam*x;
            const double error=(x-exact).norm()/std::max(exact.norm(),1e-30);
            const double residual=(joint.eta-product).norm()/rhs_norm;
            const double energy=.5*x.dot(product)-joint.eta.dot(x);
            out << "{\"variant\":\"" << mode << "\",\"sweep\":" << sweep
                << ",\"x_relative\":" << error << ",\"residual_relative\":" << residual
                << ",\"energy_excess\":" << energy-minimum << ",\"mean_norm\":" << x.norm()
                << ",\"reference_residual\":" << reference_residual
                << ",\"full_sweeps\":" << w.full_precision_sweeps << ",\"eta_sweeps\":" << w.eta_only_sweeps
                << ",\"precision_residual\":" << w.last_precision_residual << "}\n";
        }
    }
}
}
