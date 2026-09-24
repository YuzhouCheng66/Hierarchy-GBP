#pragma once

#include <cmath>
#include <cstring>
#include <Eigen/Core>

namespace slam {

// Cache the accepted eigenproblem, not just a small Ritz residual. The latter
// cannot certify that an old invariant subspace still contains the lowest modes.
struct ExactBasisCacheEntry {
    Eigen::MatrixXd matrix;
    Eigen::MatrixXd basis;
    double min_eigenvalue = 0.0;
    double max_eigenvalue = 0.0;
    bool valid = false;

    bool matches(const Eigen::MatrixXd& current, int rank) const {
        return valid && current.rows() == matrix.rows() &&
            current.cols() == matrix.cols() && basis.rows() == current.rows() &&
            basis.cols() == rank && current.size() > 0 &&
            std::memcmp(current.data(), matrix.data(),
                static_cast<size_t>(current.size()) * sizeof(double)) == 0;
    }

    void store(const Eigen::MatrixXd& current, const Eigen::MatrixXd& accepted,
               double minimum, double maximum) {
        valid = current.rows() == current.cols() && current.rows() > 0 &&
            accepted.rows() == current.rows() && accepted.cols() > 0 &&
            accepted.cols() < current.rows() && current.allFinite() &&
            accepted.allFinite() && std::isfinite(minimum) && std::isfinite(maximum);
        if (!valid) return;
        matrix = current;
        basis = accepted;
        min_eigenvalue = minimum;
        max_eigenvalue = maximum;
    }
};

} // namespace slam
