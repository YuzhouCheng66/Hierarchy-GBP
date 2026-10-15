#include "internal/se3_objective.h"
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace {
void require(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
}

int main() {
    try {
        slam::SyntheticSE3Problem p;
        p.init_poses.resize(2);
        p.anchor_pose=p.init_poses.front();
        p.anchor_information=Eigen::Matrix<double,6,6>::Identity()*100;
        slam::SyntheticSE3Edge e;
        e.i=0; e.j=1; e.information=Eigen::Matrix<double,6,6>::Identity();
        p.edges.push_back(e);
        slam::RobustLossConfig config; config.huber_delta=5;
        for(double distance:{0.,1.,5.,5.000001,10.}) {
            p.init_poses[1].t=Eigen::Vector3d(distance,0,0);
            const auto score=slam::evaluateSE3Objectives(p,p.init_poses,config);
            const double expected=distance<=5 ? .5*distance*distance : 5*(distance-2.5);
            require(std::abs(score.huber-expected)<1e-12,"Huber piecewise cost is incorrect");
            require(score.raw==slam::nonlinearObjective(p,p.init_poses),"Raw cost contract changed");
            slam::RobustLossConfig no_huber;
            const auto ordinary=slam::evaluateSE3Objectives(p,p.init_poses,no_huber);
            require(ordinary.raw==ordinary.huber,"Disabled Huber differs from quadratic");
        }
        p.init_poses[0].t=Eigen::Vector3d(20,0,0);
        p.init_poses[1].t=Eigen::Vector3d(30,0,0);
        const auto anchored=slam::evaluateSE3Objectives(p,p.init_poses,config);
        require(std::abs(anchored.huber-(20000+37.5))<1e-10,"Gauge prior was robustified");

        p.init_poses[0].t=Eigen::Vector3d(.02,-.03,.01);
        p.init_poses[1].t=Eigen::Vector3d(10,.3,-.2);
        p.init_poses[1].q=Eigen::Quaterniond(Eigen::AngleAxisd(.2,Eigen::Vector3d(1,2,3).normalized()));
        auto graph=slam::buildLinearizedResidualGraph(p,p.init_poses,1e-12,config);
        const auto joint=graph.jointDistributionInfSparse();
        Eigen::VectorXd gradient(12);
        for(int j=0;j<12;++j) {
            Eigen::VectorXd delta=Eigen::VectorXd::Zero(12); delta[j]=1e-6;
            const double plus=slam::evaluateSE3Objectives(p,slam::applyPoseDeltas(p.init_poses,delta),config).huber;
            const double minus=slam::evaluateSE3Objectives(p,slam::applyPoseDeltas(p.init_poses,-delta),config).huber;
            gradient[j]=(plus-minus)/2e-6;
        }
        const double error=(gradient+joint.eta).norm()/joint.eta.norm();
        require(error<1e-7,"Huber objective gradient disagrees with analytic weighted factors");
        for(int i=0;i<1024;++i) {
            auto edge=e;
            edge.information*=1+.07*i;
            edge.measurement.t=Eigen::Vector3d(std::sin(i),std::cos(i),.003*i);
            p.edges.push_back(edge);
        }
        slam::SE3ObjectiveWorkspace workspace;
        const auto serial=slam::evaluateSE3Objectives(p,p.init_poses,config);
        for(int threads:{1,16}) {
            for(int repeat=0;repeat<3;++repeat) {
                const auto buffered=slam::evaluateSE3ObjectivesBuffered(p,p.init_poses,config,workspace,threads);
                require(serial.raw==buffered.raw && serial.huber==buffered.huber,
                        "Buffered objective did not preserve serial summation");
            }
            const Eigen::VectorXd delta=Eigen::VectorXd::LinSpaced(12,-.2,.1);
            slam::SE3PoseVector poses;
            for(double scale:{1.,.5,.0009765625}) {
                const auto expected=slam::applyPoseDeltas(p.init_poses,scale*delta);
                slam::applySE3PoseDeltasInto(p.init_poses,delta,scale,poses,threads);
                for(int i=0;i<2;++i)
                    require((poses[i].t.array()==expected[i].t.array()).all() &&
                        (poses[i].q.coeffs().array()==expected[i].q.coeffs().array()).all(),
                        "Buffered pose update changed values");
            }
        }
        std::cout << "Huber edge/gauge/analytic-gradient tests passed; relative error=" << error << '\n';
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
