// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "ov_gaussian/Backend.h"
namespace ov_gaussian {
// Audit only: 113-bit arithmetic, independent dense covariance and root models.
struct AccurateChi {
  double dense, root;
};
AccurateChi accurate_chi(const Matrix &h, const Matrix &p, const Vector &r,
                         const Matrix &z);
} // namespace ov_gaussian
