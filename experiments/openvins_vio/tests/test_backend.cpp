// SPDX-License-Identifier: GPL-3.0-or-later
#include "ov_gaussian/Backend.h"
#include <iostream>
#include <stdexcept>
using namespace ov_gaussian;
void check(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
Matrix values(int n, int m, double scale) {
  Matrix a(n, m);
  for (int j = 0; j < m; ++j)
    for (int i = 0; i < n; ++i)
      a(i, j) = scale * std::sin(0.3 + i * 1.7 + j * 2.3);
  return a;
}
void close(const Matrix &a, const Matrix &b) {
  check((a - b).norm() <= 1e-10 + 1e-8 * b.norm(),
        "independent reference mismatch");
}
int main() {
  try {
    Backend root("root_messages", true), ekf("enhanced_ekf", true);
    Matrix a = values(15, 15, 0.03),
           p = Matrix::Identity(15, 15) * 0.1 + a * a.transpose();
    root.initialize(p);
    ekf.initialize(p);
    auto propagate = [&]() {
      Matrix phi = Matrix::Identity(15, 15) + values(15, 15, 0.002),
             q = Matrix::Identity(15, 15) * 0.0001;
      Matrix f = Matrix::Identity(p.rows(), p.cols());
      f.topLeftCorner(15, 15) = phi;
      p = (f * p * f.transpose()).eval();
      p.topLeftCorner(15, 15) += q;
      root.propagate(phi, q);
      ekf.propagate(phi, q);
      close(root.full_covariance(), p);
      close(ekf.full_covariance(), p);
    };
    auto clone = [&]() {
      int n = p.rows();
      Matrix next(n + 6, n + 6);
      next.topLeftCorner(n, n) = p;
      next.topRightCorner(n, 6) = p.leftCols(6);
      next.bottomLeftCorner(6, n) = p.topRows(6);
      next.bottomRightCorner(6, 6) = p.topLeftCorner(6, 6);
      p = next;
      root.clone_pose(0, 6);
      ekf.clone_pose(0, 6);
      close(root.full_covariance(), p);
    };
    auto remove = [&](int id) {
      int n = p.rows();
      Matrix next(n - 6, n - 6);
      for (int i = 0; i < n - 6; ++i)
        for (int j = 0; j < n - 6; ++j)
          next(i, j) = p(i + (i >= id ? 6 : 0), j + (j >= id ? 6 : 0));
      p = next;
      root.marginalize(id, 6);
      ekf.marginalize(id, 6);
      close(root.full_covariance(), p);
      close(ekf.full_covariance(), p);
    };
    propagate();
    propagate();
    clone();
    clone(); // Exact duplicate aliases, including deterministic singularity.
    propagate();
    clone();
    propagate();
    clone();
    propagate();
    remove(21); // Remove one duplicate; its shared unique state must remain.
    remove(
        21); // Remove an interior unique pose: exercises exact RQ retirement.
    remove(15); // Remove the oldest unique pose: exercises suffix right Givens.
    for (int step = 0; step < 12; ++step) {
      propagate();
      clone();
      int n = p.rows(), k = n - 15;
      Indices cols(k);
      for (int j = 0; j < k; ++j)
        cols[j] = 15 + j;
      // Full-rank nuisance occupies only the first three rows; its nullspace is
      // explicit.
      Matrix hf = Matrix::Zero(11, 3);
      hf.topRows(3) = Matrix::Identity(3, 3);
      Matrix hx = values(11, k, 0.4);
      Vector residual = values(11, 1, 0.01).col(0);
      Matrix h = Matrix::Zero(8, n);
      h.rightCols(k) = hx.bottomRows(8);
      Vector r = residual.tail(8);
      root.begin_visual();
      ekf.begin_visual();
      auto c = root.add_feature(hf, hx, residual, cols, 1, 100);
      auto e = ekf.add_feature(hf, hx, residual, cols, 1, 100);
      check(c.accepted && e.accepted && c.certified && e.certified,
            "small residual certificate");
      Vector outlier = Vector::Constant(11, 100);
      c = root.add_feature(hf, hx, outlier, cols, 1, 0.001);
      e = ekf.add_feature(hf, hx, outlier, cols, 1, 0.001);
      check(!c.accepted && !e.accepted && !c.certified && !e.certified,
            "outlier innovation solve");
      Matrix s = Matrix::Identity(8, 8) + h * p * h.transpose();
      Matrix gain = s.llt().solve(Matrix(h * p)).transpose();
      Vector expected = gain * r;
      close(root.finish_visual(), expected);
      close(ekf.finish_visual(), expected);
      Matrix j = Matrix::Identity(n, n) - gain * h;
      p = (j * p * j.transpose() + gain * gain.transpose()).eval();
      close(root.full_covariance(), p);
      close(ekf.full_covariance(), p);
      Indices subset = {3, 4, 5, 15, 16, 17};
      Matrix pm(6, 6);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
          pm(i, j) = p(subset[i], subset[j]);
      close(root.marginal(subset), pm);
      if (root.dimension() > 39)
        remove(15);
    }
    root.begin_visual();
    check(root.finish_visual().size() == 0, "empty update must be a no-op");
    check(root.statistics().general_removals > 0 &&
              root.statistics().suffix_removals > 0,
          "retirement branch coverage");
    check(root.statistics().audit_gates == 24 &&
              root.statistics().certified_accepts == 12 &&
              root.statistics().innovation_solves == 12,
          "gate coverage");
    // Independently eliminate a general nuisance Jacobian by a dense orthogonal
    // projector (normal-equation Schur complement), without QR or a chosen
    // basis.
    {
      int n = p.rows(), k = n - 15;
      Indices cols(k);
      for (int j = 0; j < k; ++j)
        cols[j] = 15 + j;
      Matrix hf = values(13, 3, 0.3), hx = values(13, k, 0.2);
      hf.topRows(3) += Matrix::Identity(3, 3);
      Vector residual = values(13, 1, 0.01).col(0);
      double variance = 0.7;
      Matrix projector =
          Matrix::Identity(13, 13) -
          hf * (hf.transpose() * hf).ldlt().solve(Matrix(hf.transpose()));
      Matrix h = Matrix::Zero(13, n);
      h.rightCols(k) = projector * hx / std::sqrt(variance);
      Vector r = projector * residual / std::sqrt(variance);
      Matrix s = Matrix::Identity(13, 13) + h * p * h.transpose();
      Matrix gain = s.llt().solve(Matrix(h * p)).transpose();
      for (Backend *backend : {&root, &ekf}) {
        backend->begin_visual();
        check(backend->add_feature(hf, hx, residual, cols, variance, 100)
                  .accepted,
              "general nuisance feature acceptance");
        close(backend->finish_visual(), gain * r);
      }
      Matrix j = Matrix::Identity(n, n) - gain * h;
      p = (j * p * j.transpose() + gain * gain.transpose()).eval();
      close(root.full_covariance(), p);
      close(ekf.full_covariance(), p);
    }
    std::cout << "PASS: dense Joseph, live propagation, singular aliases, "
                 "interior/suffix retirement, certified and rejected gates, "
                 "marginal queries\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
