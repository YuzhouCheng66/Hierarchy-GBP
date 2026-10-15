#pragma once

#include <Eigen/Core>
#include <algorithm>
#include <atomic>
#include <limits>
#include <vector>

// LP64 LAPACKE middle-level API from the existing frozen OpenBLAS runtime.
extern "C" int LAPACKE_dsbevx_work(int, char, char, char, int, int,
    double*, int, double*, int, double, double, int, int, double,
    int*, double*, double*, int, double*, int*, int*);
extern "C" int LAPACKE_dsbtrd_work(int, char, char, int, int, double*, int,
    double*, double*, double*, int, double*);
extern "C" int LAPACKE_dstemr_work(int, char, char, int, double*, double*, double,
    double, int, int, int*, double*, double*, int, int, int*, int*, double*, int, int*, int);

namespace slam {

class BandedSymmetricEigenWorkspace {
public:
    inline static std::atomic<int> calls{0}, failures{0};

    static int exactHalfBandwidth(const Eigen::Ref<const Eigen::MatrixXd>& a) {
        int bandwidth = 0;
        for (int j = 0; j < a.cols(); ++j) {
            for (int i = static_cast<int>(a.rows()) - 1; i > j + bandwidth; --i) {
                // Exact zero only: no sparsification or numerical threshold.
                if (a(i,j) != 0.0) { bandwidth = i-j; break; }
            }
        }
        return bandwidth;
    }

    // Backend 0 is the scaled DSBEVX driver; backend 2 is MRRR with DSBEVX fallback.
    bool compute(const Eigen::Ref<const Eigen::MatrixXd>& a, int rank, int backend = 0) {
        if (a.rows() == 0 || a.rows() != a.cols() || rank < 1 || rank > a.rows()
            || a.rows() > std::numeric_limits<int>::max() || !a.allFinite()) return false;
        const int n = static_cast<int>(a.rows());
        const int kd = exactHalfBandwidth(a);
        if (kd > n/4) return false;
        calls.fetch_add(1, std::memory_order_relaxed);
        band_.setZero(kd+1, n);
        for (int j = 0; j < n; ++j)
            for (int i = j; i <= std::min(n-1, j+kd); ++i)
                band_(i-j,j) = a(i,j);
        q_.resize(n,n);
        vectors.resize(n,rank);
        spectrum_.resize(n);
        if (backend == 2) {
            // Keep the legacy scaled driver for extreme magnitudes. DSBTRD
            // itself does not supply the driver's overflow/underflow scaling.
            const double norm = a.cwiseAbs().maxCoeff();
            const double safe=std::sqrt(std::numeric_limits<double>::min()/std::numeric_limits<double>::epsilon());
            if (norm > 1.0/safe || (norm > 0.0 && norm < safe)) return compute(a,rank,0);
            diagonal_.resize(n); off_diagonal_.setZero(n);
            work_.resize(18*static_cast<size_t>(n));
            iwork_.resize(10*static_cast<size_t>(n));
            support_.resize(2*static_cast<size_t>(n));
            tridiagonal_vectors_.resize(n,rank);
            int info=LAPACKE_dsbtrd_work(102,'V','L',n,kd,band_.data(),kd+1,
                diagonal_.data(),off_diagonal_.data(),q_.data(),n,work_.data());
            int found=0,try_relative_accuracy=1;
            if (info==0) info=LAPACKE_dstemr_work(102,'V','I',n,diagonal_.data(),off_diagonal_.data(),
                0,0,1,rank,&found,spectrum_.data(),tridiagonal_vectors_.data(),n,rank,
                support_.data(),&try_relative_accuracy,work_.data(),static_cast<int>(work_.size()),
                iwork_.data(),static_cast<int>(iwork_.size()));
            if (info!=0 || found!=rank || !spectrum_.head(rank).allFinite() || !tridiagonal_vectors_.allFinite())
                return compute(a,rank,0);
            values=spectrum_.head(rank);
            vectors.noalias()=q_*tridiagonal_vectors_;
            if (!vectors.allFinite()) return compute(a,rank,0);
            return true;
        }
        work_.resize(7*static_cast<size_t>(n));
        iwork_.resize(5*static_cast<size_t>(n));
        ifail_.resize(n);
        int found = 0;
        const int info = LAPACKE_dsbevx_work(102, 'V', 'I', 'L', n, kd,
            band_.data(), kd+1, q_.data(), n, 0, 0, 1, rank,
            2*std::numeric_limits<double>::min(), &found, spectrum_.data(),
            vectors.data(), n, work_.data(), iwork_.data(), ifail_.data());
        if (info != 0 || found != rank || !vectors.allFinite() ||
            !spectrum_.head(rank).allFinite()) {
            failures.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        values = spectrum_.head(rank);
        return true;
    }

    Eigen::MatrixXd vectors;
    Eigen::VectorXd values;
private:
    Eigen::MatrixXd band_, q_, tridiagonal_vectors_;
    Eigen::VectorXd spectrum_, diagonal_, off_diagonal_;
    std::vector<double> work_;
    std::vector<int> iwork_, ifail_;
    std::vector<int> support_;
};

} // namespace slam
