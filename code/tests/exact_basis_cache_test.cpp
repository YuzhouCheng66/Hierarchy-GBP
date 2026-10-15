#include "internal/exact_basis_cache.h"
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool passed, const char* message) {
    if (!passed) throw std::runtime_error(message);
}

int main() {
    try {
        slam::ExactBasisCacheEntry cache;
        Eigen::MatrixXd a = Eigen::Vector3d(1., 2., 3.).asDiagonal();
        Eigen::MatrixXd p = Eigen::MatrixXd::Identity(3, 1);
        require(!cache.matches(a, 1), "Empty cache must miss");
        cache.store(a, p, 1., 1.);
        require(cache.matches(a, 1), "Identical eigenproblem must hit");
        require(!cache.matches(a, 2), "Rank changes must invalidate");
        Eigen::MatrixXd b = a;
        b(0, 0) = std::nextafter(1., 2.);
        require(!cache.matches(b, 1), "One ULP changes must invalidate");
        b = a;
        b(0, 1) = -0.;
        require(!cache.matches(b, 1), "Exact cache distinguishes signed zero");
        b = a;
        b(2, 2) = 0.5;
        require((b * p - p).norm() == 0., "Counterexample remains invariant");
        require(!cache.matches(b, 1), "Invariant but no longer lowest must miss");
        require(!cache.matches(Eigen::MatrixXd::Identity(4, 4), 1), "Size changes must miss");
        b = a;
        b(0, 0) = std::numeric_limits<double>::quiet_NaN();
        cache.store(b, p, 1., 1.);
        require(!cache.matches(b, 1), "NaN matrix must not enter cache");
        cache.store(a, p, 1., std::numeric_limits<double>::infinity());
        require(!cache.matches(a, 1), "Invalid eigenvalue must not enter cache");
        p(1, 0) = std::numeric_limits<double>::infinity();
        cache.store(a, p, 1., 1.);
        require(!cache.matches(a, 1), "Invalid basis must not enter cache");
        std::cout << "exact_basis_cache_test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
