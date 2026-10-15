#pragma once
#include <Eigen/Core>
#include <Eigen/SVD>

namespace slam {
// Orthogonal Procrustes changes coordinates only, not the selected subspace.
inline void alignBasisCoordinates(Eigen::MatrixXd& basis,const Eigen::MatrixXd& reference) {
    if(basis.rows()!=reference.rows() || basis.cols()!=reference.cols() || basis.cols()==0) return;
    const Eigen::MatrixXd overlap=basis.transpose()*reference;
    Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::NoQRPreconditioner> svd(overlap,Eigen::ComputeFullU|Eigen::ComputeFullV);
    if(svd.info()!=Eigen::Success) return;
    basis=(basis*(svd.matrixU()*svd.matrixV().transpose())).eval();
}
}
