#include "internal/partial_symmetric_eigen.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr int n=120, k=12;
        Eigen::MatrixXd seed(n,n);
        for(int i=0;i<n;++i) for(int j=0;j<n;++j)
            seed(i,j)=std::sin(0.17+1.1*i+2.3*j+0.71*i*j);
        const Eigen::MatrixXd q=seed.householderQr().householderQ();
        int aborted=0, converged=0;
        for(int problem=0;problem<4;++problem) {
            Eigen::VectorXd spectrum(n);
            for(int i=0;i<n;++i) {
                if(problem==0) spectrum[i]=100.+i;
                if(problem==1) spectrum[i]=std::pow(1.1,i);
                if(problem==2) spectrum[i]=1.+(i/6);
                if(problem==3) spectrum[i]=i<6 ? 1e-8 : 1.+i;
            }
            Eigen::MatrixXd a=q*spectrum.asDiagonal()*q.transpose();
            a=(0.5*(a+a.transpose())).eval();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> full(a);
            require(full.info()==Eigen::Success,"Reference full eigensolve failed");
            Eigen::MatrixXd warm=full.eigenvectors().leftCols(k);
            for(bool use_warm:{false,true}) {
                slam::PartialSymmetricEigenOptions options;
                slam::PartialSymmetricEigenWorkspace first_ws, repeated_ws, predicted_ws;
                const auto original=slam::computeSmallestEigenpairsPartial(a,k,use_warm?&warm:nullptr,first_ws,options);
                const auto repeated=slam::computeSmallestEigenpairsPartial(a,k,use_warm?&warm:nullptr,repeated_ws,options);
                require((original.eigenvectors-repeated.eigenvectors).norm()==0.,"Disabled heuristic changed iterates");
                options.predict_failed_budget=true;
                const auto predicted=slam::computeSmallestEigenpairsPartial(a,k,use_warm?&warm:nullptr,predicted_ws,options);
                if(predicted.aborted_for_work) {
                    ++aborted;
                    require(!predicted.converged && predicted.iterations>=2 && predicted.iterations<options.max_iters,
                            "Work abort must force full fallback before exhausting budget");
                }
                if(predicted.converged) {
                    ++converged;
                    require(!predicted.aborted_for_work && predicted.max_relative_residual<=options.residual_tol,
                            "Work heuristic weakened the acceptance tolerance");
                }
                const Eigen::MatrixXd p=predicted.converged?predicted.eigenvectors:full.eigenvectors().leftCols(k).eval();
                const Eigen::VectorXd values=predicted.converged?predicted.eigenvalues:full.eigenvalues().head(k).eval();
                require((p.transpose()*p-Eigen::MatrixXd::Identity(k,k)).norm()<1e-10,"Result is not orthonormal");
                for(int col=0;col<k;++col)
                    require((a*p.col(col)-values[col]*p.col(col)).norm()/std::max(1.,std::abs(values[col]))<1.001e-5,
                            "Accepted or full fallback eigenvector failed residual test");
            }
        }
        require(aborted>0 && converged>0,"Test did not exercise both work abort and convergence");
        std::cout << "PASS partial budget: abort=" << aborted << " converged=" << converged << '\n';
        return 0;
    } catch(const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
