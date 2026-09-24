#pragma once

#include "internal/se3_solver_impl.h"
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <stdexcept>

namespace slam {

inline Eigen::MatrixXd se3GroupGaugeGenerators(const SE3PoseVector& poses,
                                              const std::vector<int>& group) {
    if (group.empty()) throw std::runtime_error("Empty rigid group");
    Eigen::Vector3d center=Eigen::Vector3d::Zero();
    for(int i:group) center+=poses.at(i).t;
    center/=static_cast<double>(group.size());
    Eigen::MatrixXd g=Eigen::MatrixXd::Zero(6*group.size(),6);
    for(int j=0;j<static_cast<int>(group.size());++j) {
        const auto& pose=poses.at(group[j]);
        const Eigen::Matrix3d rt=pose.q.normalized().toRotationMatrix().transpose();
        const Eigen::Vector3d t=pose.t-center;
        Eigen::Matrix3d skew;
        skew << 0,-t.z(),t.y(),t.z(),0,-t.x(),-t.y(),t.x(),0;
        g.block<3,3>(6*j,0)=rt;
        g.block<3,3>(6*j,3)=-rt*skew;
        g.block<3,3>(6*j+3,3)=rt;
    }
    return g;
}

// Exact constrained spectral construction for the diagnostic experiment.
// Qg spans the six group-rigid modes. The remaining r-6 columns minimize
// trace(V' K V) on Qg's orthogonal complement. No penalty/tuned weight is used.
inline Eigen::MatrixXd se3RigidPreservingBasis(const Eigen::MatrixXd& information,
                                              const Eigen::MatrixXd& generators,
                                              int rank) {
    const int n=static_cast<int>(information.rows());
    if(information.cols()!=n || generators.rows()!=n || generators.cols()!=6 || rank<6 || rank>n)
        throw std::runtime_error("Invalid constrained SE3 basis dimensions");
    if(rank==n) return Eigen::MatrixXd::Identity(n,n);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(generators);
    const Eigen::MatrixXd q=qr.householderQ()*Eigen::MatrixXd::Identity(n,n);
    Eigen::MatrixXd basis(n,rank);
    basis.leftCols(6)=q.leftCols(6);
    if(rank>6) {
        const auto complement=q.rightCols(n-6);
        Eigen::MatrixXd k=complement.transpose()*information*complement;
        k=(0.5*(k+k.transpose())).eval();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(k);
        if(eig.info()!=Eigen::Success) throw std::runtime_error("Constrained SE3 basis eigensolve failed");
        basis.rightCols(rank-6).noalias()=complement*eig.eigenvectors().leftCols(rank-6);
    }
    return basis;
}

} // namespace slam
