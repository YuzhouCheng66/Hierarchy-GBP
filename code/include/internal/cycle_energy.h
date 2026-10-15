#pragma once

#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Core>

namespace slam {

// Flexible H-conjugate recombination of complete multigrid-cycle directions.
// Hq is obtained from two true residuals; the helper never solves the fine system.
class CycleEnergyHistory {
public:
    explicit CycleEnergyHistory(int capacity) {
        directions_.reserve(capacity);
        products_.reserve(capacity);
        energies_.reserve(capacity);
    }

    bool accept(const Eigen::VectorXd& x0, const Eigen::VectorXd& r0,
                Eigen::VectorXd& x1, Eigen::VectorXd& r1) {
        Eigen::VectorXd q = x1 - x0;
        Eigen::VectorXd hq = r0 - r1;
        for (int pass = 0; pass < 2; ++pass) {
            for (size_t j = 0; j < directions_.size(); ++j) {
                const double beta = directions_[j].dot(hq) / energies_[j];
                q.noalias() -= beta * directions_[j];
                hq.noalias() -= beta * products_[j];
            }
        }
        const double energy = q.dot(hq);
        const double scale = q.norm() * hq.norm();
        if (!std::isfinite(energy) || !std::isfinite(scale) ||
            energy <= 64 * std::numeric_limits<double>::epsilon() * scale) {
            x1 = x0;
            r1 = r0;
            return false;
        }
        const double alpha = q.dot(r0) / energy;
        if (!std::isfinite(alpha)) {
            x1 = x0;
            r1 = r0;
            return false;
        }
        x1.noalias() = x0 + alpha * q;
        r1.noalias() = r0 - alpha * hq;
        directions_.push_back(std::move(q));
        products_.push_back(std::move(hq));
        energies_.push_back(energy);
        return true;
    }

    int size() const { return static_cast<int>(directions_.size()); }
    void clear() { directions_.clear(); products_.clear(); energies_.clear(); }

private:
    std::vector<Eigen::VectorXd> directions_;
    std::vector<Eigen::VectorXd> products_;
    std::vector<double> energies_;
};
}
