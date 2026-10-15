#include "internal/coarse_cholesky.h"
#include <iostream>

static void require(bool test, const char* message) {
    if (!test) throw std::runtime_error(message);
}

int main() {
    try {
        for (int n : {3, 24, 120}) {
            Eigen::MatrixXd seed(n,n);
            for (int i=0;i<n;++i) for(int j=0;j<n;++j)
                seed(i,j)=std::sin(.1+i+1.7*j+.13*i*j);
            Eigen::MatrixXd dense=seed.transpose()*seed;
            dense.diagonal().array()+=.25;
            slam::CoarseSparse a=dense.sparseView(); a.makeCompressed();
            Eigen::VectorXd rhs=Eigen::VectorXd::LinSpaced(n,-1.,1.);
            slam::CoarseCholesky eigen(false), cholmod(true);
            for (int update=0;update<3;++update) {
                if(update==0) { eigen.compute(a); cholmod.compute(a); }
                else { eigen.factorize(a); cholmod.factorize(a); }
                require(eigen.info()==Eigen::Success && cholmod.info()==Eigen::Success,"Factorization failed");
                if(n==120) require(cholmod.supernodal(),"Dense120 AUTO did not exercise supernodal");
                Eigen::VectorXd x,y;
                eigen.solveInto(rhs,x); cholmod.solveInto(rhs,y);
                require((x-y).norm()<1e-10*std::max(1.,x.norm()),"Coarse backends disagree");
                require((a*y-rhs).norm()<1e-10*rhs.norm(),"True residual failed");
                auto work=cholmod.workEstimate(a.nonZeros());
                require(work.factor_work>0 && work.solve_work>0 && work.matvec_work>0,"Invalid work model");
                if(n==3) {
                    auto expected=eigen.workEstimate(a.nonZeros());
                    require(expected.factor_work==work.factor_work && expected.solve_work==work.solve_work,
                            "Diagonal counting convention differs");
                }
                for(int i=0;i<n;++i) a.coeffRef(i,i)+=.03;
            }
            // Certify an old coarse factor only as a preconditioner for new A.
            Eigen::VectorXd x,r,z,p,ap;
            auto certificate=slam::certifiedCoarsePcg(a,0.,rhs,
                [&](const Eigen::VectorXd& b,Eigen::VectorXd& out){cholmod.solveInto(b,out);},x,r,z,p,ap);
            require(certificate.accepted && certificate.relative_residual<=1e-6,"Certified reuse failed");
            slam::CoarseSparse diagonal(n,n); diagonal.setIdentity(); diagonal.makeCompressed();
            cholmod.compute(diagonal); cholmod.solveInto(rhs,x);
            require((x-rhs).norm()<1e-14,"Changed pattern reanalysis failed");
        }
        std::cout << "PASS coarse backend: full/reused solves, work model and changed pattern\n";
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
