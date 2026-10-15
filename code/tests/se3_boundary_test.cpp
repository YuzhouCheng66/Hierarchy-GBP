#include "internal/se3_boundary_precision.h"
#include "internal/se3_precision_warm.h"
#include "internal/se3_rigid_basis.h"
#include <Eigen/Eigenvalues>
#include <iostream>

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
}

int main() {
    try {
        slam::SyntheticSE3Problem p;
        p.init_poses.resize(4);
        for (int i=0;i<4;++i) p.init_poses[i].t=Eigen::Vector3d(i*1.03,0.02*i,0);
        p.anchor_pose=p.init_poses[0];
        p.anchor_information=Eigen::Matrix<double,6,6>::Identity()*100;
        for (int i=0;i<3;++i) {
            slam::SyntheticSE3Edge edge;
            edge.i=i; edge.j=i+1; edge.measurement.t=Eigen::Vector3d(1,0,0);
            edge.information=Eigen::Matrix<double,6,6>::Identity()*10;
            p.edges.push_back(edge);
        }
        {
            auto rotated=p;
            for(int i=0;i<4;++i) rotated.init_poses[i].q=Eigen::Quaterniond(
                Eigen::AngleAxisd(.13*i,Eigen::Vector3d(1,2,3).normalized()));
            auto rigid_graph=slam::buildLinearizedResidualGraph(rotated,rotated.init_poses);
            const std::vector<int> group{0,1,2,3};
            const auto g=slam::se3GroupGaugeGenerators(rotated.init_poses,group);
            Eigen::MatrixXd k=Eigen::MatrixXd::Zero(24,24);
            for(int i=0;i<3;++i) {
                const auto h=rigid_graph.factors[i]->factor.lam();
                const Eigen::MatrixXd pair=g.middleRows(6*i,12);
                require((h*pair).norm()/std::max(1.,h.norm()*pair.norm())<1e-10,
                        "Group gauge is not a nullspace of analytic internal factors");
                k.block(6*i,6*i,12,12)+=h;
            }
            // Boundary precision need not annihilate gauge modes.
            k.diagonal()+=Eigen::VectorXd::LinSpaced(24,1,24);
            const auto basis=slam::se3RigidPreservingBasis(k,g,12);
            require((basis.transpose()*basis-Eigen::MatrixXd::Identity(12,12)).norm()<1e-11,
                    "Constrained basis not orthonormal");
            require((g-basis*(basis.transpose()*g)).norm()/g.norm()<1e-12,
                    "Constrained basis lost a rigid generator");
            const Eigen::MatrixXd v=basis.rightCols(6),qg=basis.leftCols(6);
            const Eigen::MatrixXd kv=k*v-qg*(qg.transpose()*k*v);
            require((kv-v*(v.transpose()*kv)).norm()/kv.norm()<1e-10,
                    "Constrained spectral stationarity failed");
            auto shifted=rotated.init_poses;
            for(auto& pose:shifted) pose.t+=Eigen::Vector3d(10,-30,7);
            require((slam::se3GroupGaugeGenerators(shifted,group)-g).norm()<1e-12,
                    "Centered rigid generator changed under coordinate translation");
        }
        auto graph=slam::buildLinearizedResidualGraph(p,p.init_poses);
        auto w=slam::buildSyntheticSE3PackedSoAWorkspace(p);
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(w,graph);
        slam::validateSE3BoundarySlots(graph,w);
        for (int e=0;e<3;++e) for(int side=0;side<2;++side) {
            Eigen::Matrix<double,6,6> a;
            for(int c=0;c<6;++c) for(int r=0;r<6;++r)
                a(r,c)=std::sin(1.+r+7*c+42*(2*e+side));
            const Eigen::Matrix<double,6,6> precision=a.transpose()*a;
            for(int c=0;c<6;++c) for(int r=0;r<=c;++r)
                w.binary_msg_lam21[21*(2*e+side)+c*(c+1)/2+r]=precision(r,c);
            require((slam::se3PackedBoundaryPrecision(w,e,side)-precision).norm()<1e-13,
                    "Sym21 slot/local-order unpack mismatch");
        }
        ++w.binary_var0_id[1];
        bool rejected=false;
        try { slam::validateSE3BoundarySlots(graph,w); }
        catch(const std::runtime_error&) { rejected=true; }
        require(rejected,"Wrong packed topology was accepted");
        --w.binary_var0_id[1];
        slam::initializeBalancedSE3PackedMessages(w,1);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(w,100,1,-1,0.3,2);
        slam::SE3PrecisionWarmState previous;
        previous.capture(w,p.init_poses);
        auto next=slam::applyPoseDeltas(p.init_poses,Eigen::VectorXd::LinSpaced(24,-.01,.02));
        auto next_graph=slam::buildLinearizedResidualGraph(p,next);
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(w,next_graph);
        slam::initializeBalancedSE3PackedMessages(w,1);
        Eigen::VectorXd b_before,ax_before,b_after,ax_after;
        const Eigen::VectorXd x=Eigen::VectorXd::LinSpaced(24,-.2,.3);
        slam::assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(w,b_before,1);
        slam::multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(w,x,ax_before,1);
        slam::initializeWarmSE3Precision(w,previous,next,false,1,true);
        slam::validateSE3BoundarySlots(next_graph,w);
        double precision_norm=0;
        for(int e=0;e<3;++e) for(int side=0;side<2;++side) {
            const auto lam=slam::se3PackedBoundaryPrecision(w,e,side);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> es(lam);
            require(es.eigenvalues().minCoeff()>-1e-8,"Transported boundary is not PSD");
            precision_norm+=lam.norm();
            require(next_graph.factors[e]->messages[side].lam().norm()==0,
                    "Graph boundary control no longer zero");
        }
        require(precision_norm>1,"Warm boundary unexpectedly empty");
        slam::assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(w,b_after,1);
        slam::multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(w,x,ax_after,1);
        require((b_before-b_after).norm()==0 && (ax_before-ax_after).norm()==0,
                "Boundary transport/read changed canonical system");
        std::cout << "boundary slots/PSD/canonical checks passed, norm=" << precision_norm << '\n';
        return 0;
    } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
