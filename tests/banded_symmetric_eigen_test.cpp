#include "internal/banded_symmetric_eigen.h"
#include <Eigen/Eigenvalues>
#include <iostream>
#include <random>
#include <stdexcept>

static void require(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}

int main() {
    slam::BandedSymmetricEigenWorkspace ws;
    std::mt19937_64 rng(9047);
    std::normal_distribution<double> normal;
    int cases = 0;
    for (int n : {1, 6, 24, 60, 120}) for (int kd : {0, 1, 5, 11, 29}) {
        if (kd > n/4) continue;
        for (double scale : {1e-8, 1.0, 1e8}) for (bool spd : {false, true}) for(int backend : {0,2}) {
            Eigen::MatrixXd a = Eigen::MatrixXd::Zero(n,n);
            for (int j=0; j<n; ++j) for(int i=j; i<std::min(n,j+kd+1); ++i)
                a(i,j)=a(j,i)=normal(rng);
            if (spd) a.diagonal().array() += 4*n;
            a *= scale;
            const Eigen::MatrixXd saved = a;
            const int r = std::min(12,n);
            require(ws.compute(a,r,backend), "band solve failed");
            require((a.array()==saved.array()).all(), "input changed");
            require(slam::BandedSymmetricEigenWorkspace::exactHalfBandwidth(a)==kd,
                    "incorrect exact bandwidth");
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> ref(a);
            const double norm = std::max(1.0,a.norm());
            require((ws.values-ref.eigenvalues().head(r)).norm()/norm < 1e-11, "eigenvalues");
            require((a*ws.vectors-ws.vectors*ws.values.asDiagonal()).norm()/norm < 1e-11, "residual");
            require((ws.vectors.transpose()*ws.vectors-Eigen::MatrixXd::Identity(r,r)).norm()<1e-10,
                    "orthogonality");
            // A random simple spectrum has a unique selected invariant subspace.
            const Eigen::MatrixXd v = ref.eigenvectors().leftCols(r);
            require((ws.vectors*ws.vectors.transpose()-v*v.transpose()).norm()<1e-7, "subspace");
            ++cases;
        }
    }
    Eigen::MatrixXd a=Eigen::MatrixXd::Identity(120,120);
    require(ws.compute(a,12), "repeated spectrum");
    require(ws.compute(a,12,2), "MRRR repeated spectrum");
    a(119,0)=a(0,119)=1e-250;
    require(slam::BandedSymmetricEigenWorkspace::exactHalfBandwidth(a)==119, "tiny entry discarded");
    require(!ws.compute(a,12), "dense matrix incorrectly used band path");
    a(0,0)=std::numeric_limits<double>::quiet_NaN();
    require(!ws.compute(a,12), "NaN accepted");
    require(slam::BandedSymmetricEigenWorkspace::failures==0, "LAPACK failure");
    for(int kind=0;kind<3;++kind) for(double scale:{1e-200,1.0,1e200}) for(int backend:{0,2}) {
        Eigen::MatrixXd base=Eigen::MatrixXd::Zero(120,120);
        for(int b=0;b<20;++b) {
            Eigen::Matrix<double,6,4> r;
            for(double& v:r.reshaped()) v=normal(rng);
            base.block<6,6>(6*b,6*b).noalias()=r*r.transpose();
        }
        if(kind==1) base.diagonal().array()+=1e-12;
        if(kind==2) base=Eigen::MatrixXd::Identity(120,120)+base*1e-12;
        a=base*scale;
        require(ws.compute(a,12,backend),"clustered/scaled spectrum failed");
        require((base*ws.vectors-ws.vectors*(ws.values/scale).asDiagonal()).norm()/std::max(1.,base.norm())<1e-11,
                "clustered/scaled residual");
        require((ws.vectors.transpose()*ws.vectors-Eigen::MatrixXd::Identity(12,12)).norm()<1e-10,
                "clustered/scaled orthogonality");
    }
    std::cout << "band eigen cases=" << cases << ", residual/subspace/orthogonality/input/guards passed\n";
    std::cout << "18 rank-deficient/clustered/extreme-scaling cases passed\n";
}
