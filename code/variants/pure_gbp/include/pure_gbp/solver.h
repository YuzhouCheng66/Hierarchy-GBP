#pragma once
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <map>
#include <stdexcept>
#include <utility>

namespace pure_gbp {
using Sparse=Eigen::SparseMatrix<double>;
struct PairData {
    Eigen::MatrixXd first,second;
    Eigen::VectorXd eta_first,eta_second;
};
using PairDiagonals=std::map<std::pair<int,int>,PairData>;

// One non-reentrant solve context per native benchmark process.
void begin(const Sparse& matrix,double ridge,int block_size,int threads);
bool needsProjectedInitialization();
void initializeProjected(const PairDiagonals& pairs);
void solve(const Eigen::VectorXd& rhs,Eigen::VectorXd& result);
}
