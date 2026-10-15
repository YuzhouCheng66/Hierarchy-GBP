#include "internal/se2_residual.h"
#include <iostream>
#include <stdexcept>

int main() {
    try {
        slam::SyntheticSE2Problem p;
        for (int i=0;i<140;++i) p.init_poses.emplace_back(i*1.02,0.02*i,0.001*i);
        p.gt_poses=p.init_poses;
        p.anchor_pose=p.init_poses[0];
        p.anchor_information=Eigen::Matrix3d::Identity()*1000;
        for (int i=0;i<139;++i) {
            slam::SyntheticSE2Edge e;
            e.i=i; e.j=i+1; e.measurement=Eigen::Vector3d(1,0,0);
            e.information=Eigen::Matrix3d::Identity()*10;
            p.edges.push_back(e);
        }
        auto graph=slam::buildLinearizedResidualGraph(p,p.init_poses);
        const auto joint=graph.jointDistributionInfSparse();
        {
            const auto input_before=p.init_poses;
            slam::RobustLossConfig loss;
            loss.huber_delta=5;
            const auto direct=slam::runSyntheticSE2DirectReference(p,20,loss);
            if(direct.direct_history.size()!=21 || direct.direct_pose_history.size()!=21 ||
               !direct.mg_history.empty() || !direct.mg_pose_history.empty() ||
               direct.direct_history.back().nonlinear_objective>1e-16)
                throw std::runtime_error("standalone Direct reference skipped outers or used an MG iterate");
            const auto expected_first=slam::applyPoseDeltas(p.init_poses,Eigen::MatrixXd(joint.lam).ldlt().solve(joint.eta));
            for(size_t i=0;i<input_before.size();++i) {
                if((p.init_poses[i]-input_before[i]).norm()!=0 ||
                   (direct.direct_pose_history[1][i]-expected_first[i]).norm()>1e-7)
                    throw std::runtime_error("Direct reference changed input or differs from explicit solve");
            }
        }
        auto packed=slam::buildSyntheticSE2PackedResidualWorkspace(p);
        slam::relinearizeSyntheticSE2PackedResidualWorkspace(packed,p,p.init_poses);
        const Eigen::VectorXd x=Eigen::VectorXd::LinSpaced(420,-0.1,0.2);
        const Eigen::VectorXd expected=joint.eta-joint.lam*x;
        Eigen::VectorXd serial,parallel;
        slam::fineResidualSyntheticSE2PackedInto(packed,x,serial,1);
        slam::fineResidualSyntheticSE2PackedInto(packed,x,parallel,16);
        const double error=(serial-expected).cwiseAbs().maxCoeff();
        if(error>1e-9 || (parallel-serial).norm()>1e-12)
            throw std::runtime_error("packed residual does not match the true assembled system");
        const Eigen::VectorXd reference_shift=Eigen::VectorXd::LinSpaced(420,0.2,0.5);
        const auto old_precision=packed.binary_msg_lam6;
        slam::shiftSE2PackedCorrectionReference(packed,reference_shift);
        slam::fineResidualSyntheticSE2PackedInto(packed,x,serial,16);
        if ((serial-(joint.eta-joint.lam*(x+reference_shift))).cwiseAbs().maxCoeff()>1e-9 ||
            packed.binary_msg_lam6!=old_precision)
            throw std::runtime_error("correction RHS changed H or precision messages");
        slam::relinearizeSyntheticSE2PackedResidualWorkspace(packed,p,p.init_poses);
        slam::fineResidualSyntheticSE2PackedInto(packed,x,serial,1);
        if ((serial-expected).cwiseAbs().maxCoeff()>1e-9)
            throw std::runtime_error("relinearization failed to restore original RHS");
        double factor_sec=0, variable_sec=0;
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(packed,12,factor_sec,variable_sec,1);
        auto raw=packed, damped=packed, damped_parallel=packed;
        raw.eta_relaxation=1;
        damped.eta_relaxation=damped_parallel.eta_relaxation=0.5;
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(raw,1,factor_sec,variable_sec,1);
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(damped,1,factor_sec,variable_sec,1);
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(damped_parallel,1,factor_sec,variable_sec,16);
        if (raw.binary_msg_lam6!=damped.binary_msg_lam6)
            throw std::runtime_error("eta relaxation changed precision");
        // /fp:fast emits slightly different serial/parallel specializations;
        // require roundoff agreement, not cross-executor bit identity.
        for(size_t k=0;k<damped.binary_msg_lam6.size();++k) {
            const double a=damped.binary_msg_lam6[k],b=damped_parallel.binary_msg_lam6[k];
            if(std::abs(a-b)>32*std::numeric_limits<double>::epsilon()*std::max({1.,std::abs(a),std::abs(b)}))
                throw std::runtime_error("serial/parallel precision differs beyond roundoff");
        }
        for(size_t j=0;j<packed.binary_msg_eta.size();++j) {
            const double reference=packed.binary_msg_eta[j]+0.5*(raw.binary_msg_eta[j]-packed.binary_msg_eta[j]);
            if (std::abs(reference-damped.binary_msg_eta[j])>1e-12 ||
                std::abs(reference-damped_parallel.binary_msg_eta[j])>1e-12)
                throw std::runtime_error("eta relaxation is not synchronous convex mixing");
        }
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(damped,25,factor_sec,variable_sec,1);
        slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(damped_parallel,25,factor_sec,variable_sec,16);
        const auto mu1=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(damped,1);
        const auto mu16=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(damped_parallel,16);
        if ((mu1-mu16).norm()>1e-8*(1+mu1.norm()))
            throw std::runtime_error("damped sweep executors disagree");
        damped.lift_correction_to_messages=true;
        const auto precision_before=damped.binary_msg_lam6;
        slam::injectCorrectionKeepMessagesSyntheticSE2PackedResidualWorkspace(damped,x,16);
        const auto lifted_mu=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(damped,1);
        std::cout << "lift absolute error=" << (lifted_mu-mu1-x).norm()
                  << " reference norm=" << (mu1+x).norm() << '\n';
        if((lifted_mu-mu1-x).norm()>1e-12*(1+(mu1+x).norm()) || damped.binary_msg_lam6!=precision_before)
            throw std::runtime_error("message lift changed the requested mean or precision");
        for(int v=0;v<damped.num_vars;++v) {
            Eigen::Vector3d sum=Eigen::Map<const Eigen::Vector3d>(damped.prior_eta.data()+3*v);
            for(int p=damped.unary_offsets[v];p<damped.unary_offsets[v+1];++p)
                sum+=Eigen::Map<const Eigen::Vector3d>(damped.unary_msg_eta.data()+3*damped.unary_ids[p]);
            for(int p=damped.binary_offsets[v];p<damped.binary_offsets[v+1];++p)
                sum+=Eigen::Map<const Eigen::Vector3d>(damped.binary_msg_eta.data()+3*damped.binary_slot_ids[p]);
            if((sum-Eigen::Map<const Eigen::Vector3d>(damped.belief_eta.data()+3*v)).norm()>1e-9)
                throw std::runtime_error("coarse mean is inconsistent with lifted messages");
        }
        slam::fineResidualSyntheticSE2PackedInto(damped,x,serial,1);
        if((serial-expected).norm()>1e-9)
            throw std::runtime_error("message lift changed the target system");
        for(int initialization:{1,2}) {
            auto stationary=slam::buildSyntheticSE2PackedResidualWorkspace(p);
            slam::relinearizeSyntheticSE2PackedResidualWorkspace(stationary,p,p.init_poses);
            std::fill(stationary.prior_eta.begin(),stationary.prior_eta.end(),0);
            std::fill(stationary.unary_msg_eta.begin(),stationary.unary_msg_eta.end(),0);
            for(auto& f:stationary.unary_factors) for(double& e:f.eta) e=0;
            for(auto& f:stationary.binary_factors) for(int j=0;j<3;++j) {
                f.eta0[j]=j+1; f.eta1[j]=-j-1;
                stationary.prior_eta[3*f.var0_id+j]-=f.eta0[j];
                stationary.prior_eta[3*f.var1_id+j]-=f.eta1[j];
            }
            slam::initializeSE2PackedMessagesFromFactors(stationary,initialization,16);
            slam::synchronousIterationsSyntheticSE2PackedResidualWorkspace(stationary,30,factor_sec,variable_sec,16);
            const auto zero=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(stationary,16);
            if(zero.norm()>1e-12) throw std::runtime_error("balanced factor gradients disturbed a stationary iterate");
        }
        auto jacobi=slam::buildSyntheticSE2PackedResidualWorkspace(p);
        slam::relinearizeSyntheticSE2PackedResidualWorkspace(jacobi,p,p.init_poses);
        slam::initializeSE2PackedMessagesFromFactors(jacobi,2,1);
        Eigen::VectorXd reference=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(jacobi,1);
        const Eigen::MatrixXd dense=joint.lam;
        for(int step=0;step<5;++step) {
            const Eigen::VectorXd residual=joint.eta-joint.lam*reference;
            for(int v=0;v<jacobi.num_vars;++v)
                reference.segment<3>(3*v)+=(2./3.)*dense.block<3,3>(3*v,3*v).ldlt().solve(residual.segment<3>(3*v));
        }
        auto jacobi16=jacobi;
        slam::blockJacobiSE2PackedIterations(jacobi,5,1);
        slam::blockJacobiSE2PackedIterations(jacobi16,5,16);
        const auto j1=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(jacobi,1);
        const auto j16=slam::stackedMeanVectorSyntheticSE2PackedResidualWorkspace(jacobi16,16);
        if((j1-reference).norm()>1e-9 || (j1-j16).norm()>1e-12)
            throw std::runtime_error("persistent packed Jacobi differs from explicit reference");
        std::cout << "PASS max residual error=" << error << '\n';
        return 0;
    } catch(const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
