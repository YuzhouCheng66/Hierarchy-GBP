#pragma once

// Keep a square-root factor, not a cofactor-form inverse. Near rank-deficient
// landmarks otherwise amplify cancellation in the reduced camera diagonal.
GBP_FORCE_INLINE bool factorDampedPoint3(const Mat3& B, double lambda, Mat3& L) noexcept {
    L.setZero();
    const double p0 = B(0, 0) + lambda;
    if (!(p0 > 0.0) || !std::isfinite(p0)) return false;
    L(0, 0) = std::sqrt(p0);
    L(1, 0) = B(1, 0) / L(0, 0);
    L(2, 0) = B(2, 0) / L(0, 0);
    const double p1 = std::fma(-L(1, 0), L(1, 0), B(1, 1) + lambda);
    if (!(p1 > 0.0) || !std::isfinite(p1)) return false;
    L(1, 1) = std::sqrt(p1);
    L(2, 1) = std::fma(-L(2, 0), L(1, 0), B(2, 1)) / L(1, 1);
    const double p2 = std::fma(-L(2, 1), L(2, 1),
        std::fma(-L(2, 0), L(2, 0), B(2, 2) + lambda));
    if (!(p2 > 0.0) || !std::isfinite(p2)) return false;
    L(2, 2) = std::sqrt(p2);
    return true;
}

GBP_FORCE_INLINE Vec3 solvePoint3(const Mat3& L, const Vec3& b) noexcept {
    const double y0 = b[0] / L(0, 0);
    const double y1 = (b[1] - L(1, 0) * y0) / L(1, 1);
    const double y2 = (b[2] - L(2, 0) * y0 - L(2, 1) * y1) / L(2, 2);
    Vec3 x;
    x[2] = y2 / L(2, 2);
    x[1] = (y1 - L(2, 1) * x[2]) / L(1, 1);
    x[0] = (y0 - L(1, 0) * x[1] - L(2, 0) * x[2]) / L(0, 0);
    return x;
}

template<int Rows>
GBP_FORCE_INLINE void whitenPointCross(Eigen::Matrix<double, Rows, 3>& E, const Mat3& L) noexcept {
    for (int r = 0; r < Rows; ++r) {
        E(r, 0) /= L(0, 0);
        E(r, 1) = (E(r, 1) - L(1, 0) * E(r, 0)) / L(1, 1);
        E(r, 2) = (E(r, 2) - L(2, 0) * E(r, 0) - L(2, 1) * E(r, 1)) / L(2, 2);
    }
}
