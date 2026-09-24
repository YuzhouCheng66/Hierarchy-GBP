#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    std::mt19937_64 rng(20260923);
    std::uniform_real_distribution<double> sample(-1.0, 1.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> cached;
    const int dimensions[] = {120, 120, 60, 60, 6, 1, 2, 30, 120};
    int cases = 0;
    for (int round = 0; round < 8; ++round) {
        for (int n : dimensions) {
            Eigen::MatrixXd a(n, n);
            for (int j = 0; j < n; ++j)
                for (int i = j; i < n; ++i)
                    a(i, j) = a(j, i) = sample(rng);
            a.diagonal().array() += (round % 2 ? n : 0);
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> fresh(a);
            cached.compute(a);
            if (fresh.info() != Eigen::Success || cached.info() != Eigen::Success ||
                std::memcmp(fresh.eigenvalues().data(), cached.eigenvalues().data(),
                            sizeof(double) * n) != 0 ||
                std::memcmp(fresh.eigenvectors().data(), cached.eigenvectors().data(),
                            sizeof(double) * n * n) != 0)
                throw std::runtime_error("Workspace reuse changed eigenpairs");
            const int r = std::min(12, n);
            Eigen::MatrixXd old_all = fresh.eigenvectors();
            Eigen::MatrixXd old_selected = old_all.leftCols(r);
            Eigen::MatrixXd selected = cached.eigenvectors().leftCols(r);
            if (std::memcmp(selected.data(), old_selected.data(), sizeof(double) * n * r))
                throw std::runtime_error("Selected-mode copy changed basis");
            ++cases;
        }
    }
    std::cout << cases << " basis workspace cases bitwise equal\n";
}
