// SPDX-License-Identifier: GPL-3.0-or-later
#include "Audit.h"
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <stdexcept>
#include <vector>
namespace ov_gaussian {
namespace {
using Real = boost::multiprecision::cpp_bin_float_quad;
double quadratic(std::vector<Real> s, const Vector &r) {
  int m = r.size();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j <= i; ++j) {
      Real value = s[i * m + j];
      for (int k = 0; k < j; ++k)
        value -= s[i * m + k] * s[j * m + k];
      if (i == j) {
        if (!(value > 0))
          throw std::runtime_error(
              "High-precision audit innovation is not SPD");
        s[i * m + j] = sqrt(value);
      } else
        s[i * m + j] = value / s[j * m + j];
    }
  }
  std::vector<Real> y(m);
  Real chi = 0;
  for (int i = 0; i < m; ++i) {
    y[i] = Real(r(i));
    for (int j = 0; j < i; ++j)
      y[i] -= s[i * m + j] * y[j];
    y[i] /= s[i * m + i];
    chi += y[i] * y[i];
  }
  return chi.convert_to<double>();
}
} // namespace
AccurateChi accurate_chi(const Matrix &h, const Matrix &p, const Vector &r,
                         const Matrix &z) {
  int m = h.rows(), n = h.cols(), k = z.cols();
  std::vector<Real> hh(m * n), pp(n * n), zz(m * k), hp(m * n), dense(m * m),
      root(m * m);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      hh[i * n + j] = h(i, j);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      pp[i * n + j] = (Real(p(i, j)) + Real(p(j, i))) / 2;
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j)
      zz[i * k + j] = z(i, j);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      for (int l = 0; l < n; ++l)
        hp[i * n + j] += hh[i * n + l] * pp[l * n + j];
  for (int i = 0; i < m; ++i)
    for (int j = 0; j <= i; ++j) {
      Real d = (i == j ? 1 : 0), v = d;
      for (int l = 0; l < n; ++l)
        d += hp[i * n + l] * hh[j * n + l];
      for (int l = 0; l < k; ++l)
        v += zz[i * k + l] * zz[j * k + l];
      dense[i * m + j] = d;
      root[i * m + j] = v;
    }
  return {quadratic(std::move(dense), r), quadratic(std::move(root), r)};
}
} // namespace ov_gaussian
