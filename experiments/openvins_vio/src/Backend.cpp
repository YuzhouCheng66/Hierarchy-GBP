// SPDX-License-Identifier: GPL-3.0-or-later
// Existing covariance-root/shared-EKF algorithms adapted to live OpenVINS state
// ownership.
#include "ov_gaussian/Backend.h"
#include "Audit.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <xmmintrin.h>

extern "C" {
void dgeqrf_(const int *, const int *, double *, const int *, double *,
             double *, const int *, int *);
void dormqr_(const char *, const char *, const int *, const int *, const int *,
             const double *, const int *, const double *, double *, const int *,
             double *, const int *, int *);
void dpotrf_(const char *, const int *, double *, const int *, int *);
void dtrsm_(const char *, const char *, const char *, const char *, const int *,
            const int *, const double *, const double *, const int *, double *,
            const int *);
void dtrmm_(const char *, const char *, const char *, const char *, const int *,
            const int *, const double *, const double *, const int *, double *,
            const int *);
void dgemm_(const char *, const char *, const int *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            const int *, const double *, double *, const int *);
void dsyrk_(const char *, const char *, const int *, const int *,
            const double *, const double *, const int *, const double *,
            double *, const int *);
}
namespace ov_gaussian {
namespace {
const Indices nav = {6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5};
const Indices physical_imu = {9, 10, 11, 12, 13, 14, 0, 1, 2, 3, 4, 5, 6, 7, 8};
void require(bool ok, const char *reason) {
  if (!ok)
    throw std::runtime_error(std::string("Gaussian backend: ") + reason);
}
Matrix symmetric(const Matrix &a) { return a.selfadjointView<Eigen::Upper>(); }
Matrix select(const Matrix &a, const Indices &r, const Indices &c) {
  Matrix b(r.size(), c.size());
  for (int j = 0; j < int(c.size()); ++j)
    for (int i = 0; i < int(r.size()); ++i)
      b(i, j) = a(r[i], c[j]);
  return b;
}
Vector select_vector(const Vector &a, const Indices &r) {
  Vector b(r.size());
  for (int i = 0; i < int(r.size()); ++i)
    b(i) = a(r[i]);
  return b;
}
Matrix product(const Matrix &a, const Matrix &b, bool ta = false,
               bool tb = false) {
  int m = ta ? a.cols() : a.rows(), k = ta ? a.rows() : a.cols(),
      n = tb ? b.rows() : b.cols(), la = a.rows(), lb = b.rows();
  require(k == (tb ? b.cols() : b.rows()), "GEMM dimensions");
  Matrix c(m, n);
  double one = 1, zero = 0;
  char at = ta ? 'T' : 'N', bt = tb ? 'T' : 'N';
  dgemm_(&at, &bt, &m, &n, &k, &one, a.data(), &la, b.data(), &lb, &zero,
         c.data(), &m);
  return c;
}
void syrk(Matrix &p, const Matrix &a, double alpha, bool transpose) {
  int n = p.rows(), k = transpose ? a.rows() : a.cols(), la = a.rows();
  double one = 1;
  char up = 'U', tr = transpose ? 'T' : 'N';
  dsyrk_(&up, &tr, &n, &k, &alpha, a.data(), &la, &one, p.data(), &n);
  p = p.selfadjointView<Eigen::Upper>();
}
Matrix cholesky(const Matrix &a) {
  Matrix l = a;
  int n = l.rows(), info = 0;
  char u = 'L';
  dpotrf_(&u, &n, l.data(), &n, &info);
  require(info == 0, "non-positive Cholesky pivot");
  l.triangularView<Eigen::StrictlyUpper>().setZero();
  return l;
}
void solve(const Matrix &a, Matrix &b, bool upper) {
  int m = b.rows(), n = b.cols(), la = a.rows();
  char side = 'L', up = upper ? 'U' : 'L', tr = 'N', diag = 'N';
  double one = 1;
  dtrsm_(&side, &up, &tr, &diag, &m, &n, &one, a.data(), &la, b.data(), &m);
}
Matrix covariance_root(const Matrix &p) {
  int n = p.rows();
  Matrix reverse = p.colwise().reverse().rowwise().reverse(),
         l = cholesky(reverse), v = Matrix::Zero(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i)
      v(i, j) = l(n - 1 - i, n - 1 - j);
  return v;
}
void environment() {
  require(std::fegetround() == FE_TONEAREST &&
              (_mm_getcsr() & ((1 << 15) | (1 << 6))) == 0,
          "certified gate needs nearest rounding and gradual underflow");
}
double norm_upper(const Vector &r) {
  double upper = 0;
  for (int i = 0; i < r.size(); ++i)
    upper = std::nextafter(std::fma(r(i), r(i), upper),
                           std::numeric_limits<double>::infinity());
  return upper;
}
void compare(const Matrix &got, const Matrix &expected, double &max_abs,
             double &max_rel, const char *what) {
  require(got.rows() == expected.rows() && got.cols() == expected.cols(),
          "audit shape");
  double a = (got - expected).norm(), n = expected.norm();
  max_abs = std::max(max_abs, a);
  max_rel = std::max(max_rel, a / std::max(n, 1e-300));
  require(std::isfinite(a) && a <= 1e-10 + 1e-6 * n, what);
}
// Right Givens rotations absorb the discarded suffix while preserving V V^T.
Matrix suffix_root(const Matrix &v, int r) {
  int n = v.rows();
  Matrix a = v.topLeftCorner(r, r), b = v.topRightCorner(r, n - r);
  for (int k = 0; k < b.cols(); ++k) {
    double *extra = b.col(k).data();
    for (int j = r - 1; j >= 0; --j) {
      if (extra[j] == 0)
        continue;
      double t = std::hypot(a(j, j), extra[j]);
      require(t > 0 && std::isfinite(t), "suffix rotation");
      double c = a(j, j) / t, s = extra[j] / t;
      double *col = a.col(j).data();
      for (int i = 0; i < j; ++i) {
        double u = col[i], w = extra[i];
        col[i] = c * u + s * w;
        extra[i] = -s * u + c * w;
      }
      a(j, j) = t;
      extra[j] = 0;
    }
  }
  return a;
}
} // namespace
Backend::Backend(const std::string &mode, bool audit)
    : root_(mode == "root_messages"), audit_(audit), mode_(mode) {
  require(root_ || mode == "enhanced_ekf", "unknown mode");
}
std::string Backend::selected_mode() {
  const char *p = std::getenv("OV_GAUSSIAN_BACKEND");
  std::string m = p ? p : "original";
  require(m == "original" || m == "root_messages" || m == "enhanced_ekf",
          "unknown OV_GAUSSIAN_BACKEND");
  return m;
}
bool Backend::enabled() { return selected_mode() != "original"; }
bool Backend::audit_requested() {
  const char *p = std::getenv("OV_GAUSSIAN_AUDIT");
  return p && std::string(p) == "1";
}
void Backend::initialize(const Matrix &covariance) {
  require(!in_visual_ && covariance.rows() == 15 && covariance.cols() == 15 &&
              covariance.allFinite(),
          "initial covariance must contain only the live IMU");
  Matrix p = select(symmetric(covariance), nav, nav);
  representation_ = root_ ? covariance_root(p) : p;
  physical_ = physical_imu;
  ++stats_.initializations;
  if (audit_) {
    audit_covariance_ = symmetric(covariance);
    audit_covariance();
  }
}
Matrix Backend::covariance_unobserved() const {
  if (!root_)
    return select(representation_, physical_, physical_);
  Matrix rows(physical_.size(), representation_.cols());
  for (int i = 0; i < int(physical_.size()); ++i)
    rows.row(i) = representation_.row(physical_[i]);
  return product(rows, rows, false, true);
}
Matrix Backend::full_covariance() {
  ++stats_.full_queries;
  return covariance_unobserved();
}
Matrix Backend::marginal(const Indices &ids) {
  ++stats_.marginal_queries;
  Indices rows;
  for (int id : ids) {
    require(id >= 0 && id < dimension(), "marginal physical id");
    rows.push_back(physical_[id]);
  }
  if (!root_)
    return select(representation_, rows, rows);
  Matrix a(rows.size(), representation_.cols());
  for (int i = 0; i < int(rows.size()); ++i)
    a.row(i) = representation_.row(rows[i]);
  return product(a, a, false, true);
}
void Backend::audit_covariance() {
  if (!audit_)
    return;
  ++stats_.audit_events;
  Matrix p = covariance_unobserved();
  compare(p, audit_covariance_, stats_.audit_covariance_max_abs,
          stats_.audit_covariance_max_rel,
          "persistent covariance audit failed");
  Eigen::SelfAdjointEigenSolver<Matrix> e(p, Eigen::EigenvaluesOnly);
  require(e.info() == Eigen::Success && e.eigenvalues()(0) >= -1e-10,
          "covariance PSD audit failed");
}
void Backend::retain_unique(const Indices &keep) {
  int n = representation_.rows(), r = keep.size();
  require(r >= 15 && r <= n, "retained state dimension");
  if (r == n)
    return;
  bool prefix = true;
  for (int i = 0; i < r; ++i)
    prefix = prefix && keep[i] == i;
  if (!root_)
    representation_ = select(representation_, keep, keep);
  else if (prefix) {
    representation_ = suffix_root(representation_, r);
    ++stats_.suffix_removals;
  } else { // Exact RQ for an interior removal; not a dense-covariance fallback.
    Matrix a(n, r);
    for (int j = 0; j < r; ++j)
      a.col(j) = representation_.row(keep[r - 1 - j]).transpose();
    Eigen::HouseholderQR<Matrix> qr(a);
    Matrix rr = qr.matrixQR().topRows(r).triangularView<Eigen::Upper>(),
           next = Matrix::Zero(r, r);
    for (int j = 0; j < r; ++j)
      for (int i = 0; i <= j; ++i)
        next(i, j) = rr(r - 1 - j, r - 1 - i);
    representation_ = std::move(next);
    ++stats_.general_removals;
  }
}
void Backend::compact_unreferenced() {
  Indices keep, renumber(representation_.rows(), -1);
  for (int id : physical_) {
    require(id >= 0 && id < representation_.rows(), "physical alias mapping");
    renumber[id] = 0;
  }
  for (int i = 0; i < int(renumber.size()); ++i)
    if (renumber[i] == 0) {
      renumber[i] = keep.size();
      keep.push_back(i);
    }
  retain_unique(keep);
  for (int &id : physical_)
    id = renumber[id];
}
void Backend::propagate(const Matrix &phi, const Matrix &noise) {
  require(!in_visual_ && dimension() >= 15 && phi.rows() == 15 &&
              phi.cols() == 15 && noise.rows() == 15 && noise.cols() == 15 &&
              phi.allFinite() && noise.allFinite(),
          "IMU propagation contract");
  int n = representation_.rows(), p = n - 9;
  Matrix f = select(phi, nav, nav);
  if (root_) {
    Matrix inertial = phi.rightCols(9),
           tn = product(inertial, representation_.topLeftCorner(9, 9));
    Matrix trans = product(inertial, representation_.topRightCorner(9, p));
    trans += product(phi.leftCols(6),
                     representation_.bottomRightCorner(p, p).topRows(6));
    Matrix q = symmetric(noise);
    syrk(q, tn, 1, false);
    Matrix next = Matrix::Zero(n + 6, n + 6);
    next.topLeftCorner(15, 15) = covariance_root(select(q, nav, nav));
    for (int i = 0; i < 15; ++i)
      next.block(i, 15, 1, p) = trans.row(nav[i]);
    next.bottomRightCorner(p, p) = representation_.bottomRightCorner(p, p);
    representation_ = std::move(next);
  } else {
    Matrix next = Matrix::Zero(n + 6, n + 6);
    Matrix cross = product(f, representation_.topRightCorner(15, p));
    next.topLeftCorner(15, 15) =
        product(product(f, representation_.topLeftCorner(15, 15)), f, false,
                true) +
        select(symmetric(noise), nav, nav);
    next.topRightCorner(15, p) = cross;
    next.bottomLeftCorner(p, 15) = cross.transpose();
    next.bottomRightCorner(p, p) = representation_.bottomRightCorner(p, p);
    representation_ = symmetric(next);
  }
  for (int i = 15; i < dimension(); ++i) {
    require(physical_[i] >= 9, "clone contains inertial coordinates");
    physical_[i] += 6;
  }
  for (int i = 0; i < 15; ++i)
    physical_[i] = physical_imu[i];
  compact_unreferenced();
  ++stats_.propagations;
  if (audit_) {
    Matrix transition = Matrix::Identity(dimension(), dimension());
    transition.topLeftCorner(15, 15) = phi;
    audit_covariance_ =
        (transition * audit_covariance_ * transition.transpose()).eval();
    audit_covariance_.topLeftCorner(15, 15) += symmetric(noise);
    audit_covariance();
  }
}
void Backend::clone_pose(int id, int size) {
  require(!in_visual_ && id == 0 && size == 6,
          "only current IMU pose cloning is supported");
  Indices source(physical_.begin(), physical_.begin() + 6);
  physical_.insert(physical_.end(), source.begin(), source.end());
  ++stats_.clones;
  if (audit_) {
    int n = audit_covariance_.rows();
    Indices ids(n + 6);
    std::iota(ids.begin(), ids.begin() + n, 0);
    for (int i = 0; i < 6; ++i)
      ids[n + i] = i;
    audit_covariance_ = select(audit_covariance_, ids, ids);
    audit_covariance();
  }
}
void Backend::marginalize(int id, int size) {
  require(!in_visual_ && id >= 15 && size == 6 && id + size <= dimension(),
          "only a complete clone may be removed");
  int n = dimension();
  physical_.erase(physical_.begin() + id, physical_.begin() + id + size);
  compact_unreferenced();
  ++stats_.removals;
  if (audit_) {
    Indices ids;
    for (int i = 0; i < n; ++i)
      if (i < id || i >= id + size)
        ids.push_back(i);
    audit_covariance_ = select(audit_covariance_, ids, ids);
    audit_covariance();
  }
}
void Backend::begin_visual() {
  require(!in_visual_, "nested visual batch");
  in_visual_ = true;
  accepted_.clear();
  int p = representation_.rows() - 9;
  cache_ = root_ ? Matrix(representation_.bottomRightCorner(p, p))
                 : cholesky(representation_.bottomRightCorner(p, p));
  ++stats_.visual_attempts;
}
GateDecision Backend::add_feature(const Matrix &hf, const Matrix &hx,
                                  const Vector &r, const Indices &columns,
                                  double variance, double cutoff) {
  require(in_visual_ && hf.cols() == 3 && hf.rows() > 3 &&
              hx.rows() == hf.rows() && r.rows() == hf.rows() &&
              int(columns.size()) == hx.cols(),
          "raw feature shape");
  require(variance > 0 && std::isfinite(variance) && cutoff >= 0 &&
              std::isfinite(cutoff) && hf.allFinite() && hx.allFinite() &&
              r.allFinite(),
          "raw feature noise or finite values");
  int m = hf.rows(), k = hx.cols(), three = 3, n = k + 1, info = 0,
      lwork = 64 * std::max(3, n);
  double sigma = std::sqrt(variance);
  Matrix qr = hf / sigma, target(m, n);
  target.leftCols(k) = hx / sigma;
  target.col(k) = r / sigma;
  std::array<double, 3> tau;
  std::vector<double> work(lwork);
  dgeqrf_(&m, &three, qr.data(), &m, tau.data(), work.data(), &lwork, &info);
  require(info == 0, "feature GEQRF");
  for (int j = 0; j < 3; ++j)
    require(qr(j, j) != 0, "rank-deficient nuisance factor");
  char side = 'L', transpose = 'T';
  dormqr_(&side, &transpose, &m, &n, &three, qr.data(), &m, tau.data(),
          target.data(), &m, work.data(), &lwork, &info);
  require(info == 0, "feature ORMQR");
  Matrix h = target.block(3, 0, m - 3, k);
  Response q;
  q.residual = target.block(3, k, m - 3, 1);
  int poses = cache_.rows(), rows = h.rows();
  q.z = Matrix::Zero(rows, poses);
  for (int j = 0; j < k; ++j) {
    require(columns[j] >= 15 && columns[j] < dimension(),
            "visual columns must be live clone poses");
    int id = physical_[columns[j]];
    require(id >= 9, "visual inertial coordinate");
    q.z.col(id - 9) += h.col(j);
  }
  char right = 'R', triangle = root_ ? 'U' : 'L', normal = 'N', diag = 'N';
  double one = 1;
  dtrmm_(&right, &triangle, &normal, &diag, &rows, &poses, &one, cache_.data(),
         &poses, q.z.data(), &rows);
  environment();
  require(q.z.allFinite() && q.residual.allFinite(),
          "nonfinite whitened response");
  double upper = norm_upper(q.residual);
  GateDecision decision;
  decision.certified = upper <= cutoff;
  decision.chi2 = std::numeric_limits<double>::quiet_NaN();
  auto full_chi = [&]() {
    Matrix innovation = Matrix::Identity(rows, rows);
    syrk(innovation, q.z, 1, false);
    Matrix factor = cholesky(innovation), solved = q.residual;
    solve(factor, solved, false);
    environment();
    double value = solved.squaredNorm();
    require(std::isfinite(value), "nonfinite innovation statistic");
    return value;
  };
  if (decision.certified) {
    decision.accepted = true;
    ++stats_.certified_accepts;
  } else {
    decision.chi2 = full_chi();
    decision.accepted = !(decision.chi2 > cutoff);
    ++stats_.innovation_solves;
  }
  ++stats_.features;
  if (audit_) {
    Matrix marginal = select(audit_covariance_, columns, columns),
           innovation =
               Matrix::Identity(rows, rows) + h * marginal * h.transpose();
    double chi = q.residual.dot(innovation.llt().solve(q.residual));
    double own = decision.certified ? full_chi() : decision.chi2;
    stats_.audit_chi_max_rel =
        std::max(stats_.audit_chi_max_rel,
                 std::abs(own - chi) / std::max(std::abs(chi), 1e-300));
    if (!std::isfinite(chi) ||
        std::abs(own - chi) > 1e-10 + 1e-6 * std::abs(chi)) {
      ++stats_.audit_high_precision_gates;
      const char *path = std::getenv("OV_GAUSSIAN_AUDIT_DUMP");
      if (path && stats_.audit_high_precision_gates == 1) {
        std::ofstream dump(path);
        dump << std::setprecision(17) << own << ' ' << chi << ' ' << cutoff
             << '\n';
        for (const Matrix &matrix :
             {h, marginal, Matrix(q.residual), q.z, innovation}) {
          dump << matrix.rows() << ' ' << matrix.cols() << '\n'
               << matrix << '\n';
        }
      }
      auto accurate = accurate_chi(h, marginal, q.residual, q.z);
      double difference = std::abs(accurate.root - accurate.dense);
      stats_.audit_high_precision_model_max_rel =
          std::max(stats_.audit_high_precision_model_max_rel,
                   difference / std::max(std::abs(accurate.dense), 1e-300));
      stats_.audit_fp64_chi_vs_high_precision_max_rel =
          std::max(stats_.audit_fp64_chi_vs_high_precision_max_rel,
                   std::abs(own - accurate.root) /
                       std::max(std::abs(accurate.root), 1e-300));
      require(difference <= 1e-10 + 1e-6 * std::abs(accurate.dense),
              "high-precision independent innovation model mismatch");
      require(decision.accepted == !(accurate.root > cutoff),
              "production gate differs from high-precision root model");
      chi = accurate.dense;
    }
    require(decision.accepted == !(chi > cutoff),
            "live independent gate decision mismatch");
    ++stats_.audit_gates;
    q.h = h;
    q.columns = columns;
  }
  if (decision.accepted) {
    ++stats_.accepted;
    accepted_.push_back(std::move(q));
  }
  return decision;
}
Vector Backend::finish_visual() {
  require(in_visual_, "no visual batch");
  in_visual_ = false;
  if (accepted_.empty()) {
    cache_.resize(0, 0);
    return Vector();
  }
  int n = representation_.rows(), p = n - 9, rows = 0;
  for (const auto &q : accepted_)
    rows += q.z.rows();
  Matrix z(rows, p), residual(rows, 1);
  int offset = 0;
  for (const auto &q : accepted_) {
    z.middleRows(offset, q.z.rows()) = q.z;
    residual.middleRows(offset, q.residual.rows()) = q.residual;
    offset += q.z.rows();
  }
  Matrix gram = Matrix::Identity(p, p);
  syrk(gram, z, 1, true);
  Matrix gamma = product(z, residual, true, false), lower = cholesky(gram),
         upper = lower.transpose(), delta = gamma;
  solve(lower, delta, false);
  solve(upper, delta, true);
  Matrix transport, conditional;
  if (root_)
    transport = representation_.rightCols(p);
  else {
    Matrix top = representation_.topRightCorner(9, p);
    char right = 'R', low = 'L', trans = 'T', diag = 'N';
    double one = 1;
    int nine = 9;
    dtrsm_(&right, &low, &trans, &diag, &nine, &p, &one, cache_.data(), &p,
           top.data(), &nine);
    transport = Matrix::Zero(n, p);
    transport.topRows(9) = top;
    transport.bottomRows(p) = cache_;
    conditional = representation_.topLeftCorner(9, 9);
    syrk(conditional, top, -1, false);
  }
  Vector unique_delta = transport * delta.col(0);
  char right = 'R', up = 'U', normal = 'N', diag = 'N';
  double one = 1;
  dtrsm_(&right, &up, &normal, &diag, &n, &p, &one, upper.data(), &p,
         transport.data(), &n);
  if (root_)
    representation_.rightCols(p) = transport;
  else {
    Matrix next = Matrix::Zero(n, n);
    syrk(next, transport, 1, false);
    next.topLeftCorner(9, 9) += conditional;
    representation_ = std::move(next);
  }
  Vector result = select_vector(unique_delta, physical_);
  require(result.allFinite() && representation_.allFinite(),
          "nonfinite posterior");
  ++stats_.updates;
  if (audit_) {
    Matrix h = Matrix::Zero(rows, dimension());
    Vector rr(rows);
    offset = 0;
    for (const auto &q : accepted_) {
      for (int j = 0; j < q.h.cols(); ++j)
        h.block(offset, q.columns[j], q.h.rows(), 1) += q.h.col(j);
      rr.segment(offset, q.h.rows()) = q.residual;
      offset += q.h.rows();
    }
    if (h.rows() > h.cols()) {
      Eigen::HouseholderQR<Matrix> qr(h);
      rr = (qr.householderQ().adjoint() * rr).eval();
      int k = h.cols();
      h = Matrix(qr.matrixQR().topRows(k).triangularView<Eigen::Upper>());
      rr = Vector(rr.head(k));
    }
    Matrix innovation = Matrix::Identity(h.rows(), h.rows()) +
                        h * audit_covariance_ * h.transpose();
    Eigen::LLT<Matrix> llt(innovation);
    require(llt.info() == Eigen::Success, "independent Joseph innovation");
    Matrix gain = llt.solve(Matrix(h * audit_covariance_)).transpose();
    Vector expected = gain * rr;
    double ignored = 0;
    compare(result, expected, stats_.audit_delta_max_abs, ignored,
            "live tangent increment audit failed");
    Matrix a = Matrix::Identity(dimension(), dimension()) - gain * h;
    audit_covariance_ =
        (a * audit_covariance_ * a.transpose() + gain * gain.transpose())
            .eval();
    audit_covariance();
  }
  accepted_.clear();
  cache_.resize(0, 0);
  return result;
}
void Backend::write_statistics(const std::string &path) const {
  std::ofstream f(path);
  require(bool(f), "statistics output");
  f << std::setprecision(17) << "{\n  \"mode\": \"" << mode_
    << "\",\n  \"audit\": " << (audit_ ? "true" : "false");
#define OV_STAT(name) f << ",\n  \"" #name "\": " << stats_.name
  OV_STAT(initializations);
  OV_STAT(propagations);
  OV_STAT(clones);
  OV_STAT(removals);
  OV_STAT(visual_attempts);
  OV_STAT(features);
  OV_STAT(accepted);
  OV_STAT(updates);
  OV_STAT(certified_accepts);
  OV_STAT(innovation_solves);
  OV_STAT(marginal_queries);
  OV_STAT(full_queries);
  OV_STAT(suffix_removals);
  OV_STAT(general_removals);
  OV_STAT(audit_events);
  OV_STAT(audit_gates);
  OV_STAT(audit_covariance_max_abs);
  OV_STAT(audit_covariance_max_rel);
  OV_STAT(audit_delta_max_abs);
  OV_STAT(audit_chi_max_rel);
  OV_STAT(audit_high_precision_gates);
  OV_STAT(audit_high_precision_model_max_rel);
  OV_STAT(audit_fp64_chi_vs_high_precision_max_rel);
#undef OV_STAT
  f << ",\n  \"physical_dimension\": " << dimension()
    << ",\n  \"unique_dimension\": " << unique_dimension() << "\n}\n";
}
} // namespace ov_gaussian
