// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ov_gaussian {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Indices = std::vector<int>;

struct Statistics {
  std::uint64_t audit_high_precision_gates = 0;
  double audit_high_precision_model_max_rel = 0;
  double audit_fp64_chi_vs_high_precision_max_rel = 0;
  std::uint64_t initializations = 0, propagations = 0, clones = 0, removals = 0;
  std::uint64_t visual_attempts = 0, features = 0, accepted = 0, updates = 0;
  std::uint64_t certified_accepts = 0, innovation_solves = 0,
                marginal_queries = 0, full_queries = 0;
  std::uint64_t suffix_removals = 0, general_removals = 0, audit_events = 0,
                audit_gates = 0;
  double audit_covariance_max_abs = 0, audit_covariance_max_rel = 0,
         audit_delta_max_abs = 0;
  double audit_chi_max_rel = 0;
};
struct GateDecision {
  bool accepted = false, certified = false;
  double chi2 = 0;
};

// Persistent prior in unique coordinates: [v,bg,ba,current pose,newest ...
// oldest clone]. Clone aliases are represented exactly, without jitter.
// Physical ids remain OpenVINS ids.
class Backend {
public:
  explicit Backend(const std::string &mode, bool audit = false);
  static std::string selected_mode();
  static bool audit_requested();
  static bool enabled();
  int dimension() const { return static_cast<int>(physical_.size()); }
  int unique_dimension() const { return representation_.rows(); }
  bool auditing() const { return audit_; }
  const std::string &mode() const { return mode_; }
  const Statistics &statistics() const { return stats_; }
  const Indices &physical_map() const { return physical_; }

  void initialize(const Matrix &covariance);
  void propagate(const Matrix &phi, const Matrix &noise);
  void clone_pose(int physical_id, int size);
  void marginalize(int physical_id, int size);
  Matrix marginal(const Indices &physical_rows);
  Matrix full_covariance();

  void begin_visual();
  GateDecision add_feature(const Matrix &hf, const Matrix &hx,
                           const Vector &residual,
                           const Indices &physical_columns, double variance,
                           double cutoff);
  // Returns the physical tangent increment. Caller injects it into its own live
  // nominal states. There is no recorded nominal, recorded mean shift, or
  // oracle input in this API.
  Vector finish_visual();
  void write_statistics(const std::string &path) const;

private:
  struct Response {
    Matrix z, h;
    Vector residual;
    Indices columns;
  };
  bool root_, audit_, in_visual_ = false;
  std::string mode_;
  Matrix representation_, cache_, audit_covariance_;
  Indices physical_;
  std::vector<Response> accepted_;
  Statistics stats_;
  Matrix covariance_unobserved() const;
  void retain_unique(const Indices &keep);
  void compact_unreferenced();
  void audit_covariance();
};
} // namespace ov_gaussian
