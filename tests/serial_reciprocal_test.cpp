#include "internal/se3_schur_solve.h"
#include <Eigen/Cholesky>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    std::mt19937_64 rng(924); std::normal_distribution<double> sample;
    double maxdiff=0,maxres=0;
    for(int i=0;i<20000;++i) {
        Eigen::Matrix<double,6,6> r;
        Eigen::Matrix<double,6,7> b;
        for(double& x:r.reshaped()) x=sample(rng);
        for(double& x:b.reshaped()) x=sample(rng);
        auto a=(r*r.transpose()).eval(); a.diagonal().array()+=std::pow(10.,-(i%12));
        Eigen::LLT<Eigen::Matrix<double,6,6>> llt(a);
        Eigen::Matrix<double,6,6> l=llt.matrixL();
        double p[21]; for(int row=0;row<6;++row) for(int col=0;col<=row;++col) p[row*(row+1)/2+col]=l(row,col);
        Eigen::Matrix<double,6,7> old,now;
        slam::solveSpd6CrossEtaLanes<0,true>(p,b.data(),b.data()+36,old.data());
        slam::solveSpd6CrossEtaLanes<1,true>(p,b.data(),b.data()+36,old.data());
        slam::solveSpd6CrossEtaPackedBatched(p,b.data(),b.data()+36,now.data());
        double diff=(now-old).norm()/std::max(1.,old.norm());
        double res=(a*now-b).norm()/(a.norm()*now.norm()+b.norm());
        maxdiff=std::max(maxdiff,diff); maxres=std::max(maxres,res);
        if(!now.allFinite() || diff>1e-12 || res>1e-13) throw std::runtime_error("reciprocal solve error");
    }
    std::cout<<"reciprocal cases=20000 max_relative_change="<<maxdiff<<" max_backward_error="<<maxres<<"\n";
}
