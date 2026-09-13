# Attribution and scope

This experiment directory is distributed under GPL-3.0-or-later; see LICENSE.
This statement applies to this directory, not to unrelated repository contents.

OpenVINS is by Patrick Geneva, Guoquan Huang, the rpng/open_vins contributors and
their respective copyright holders. The build scripts download/locate OpenVINS
at commit 69488123ed9362dd44b6f28e7f4680abbff1442b. The patch retains its copyright
headers, extracts its feature/Jacobian construction verbatim at build time, and
changes explicit state-management and MSCKF interfaces. The EuRoC configuration
files are derived from OpenVINS configurations. Credit the upstream project and
its authors in any publication using this experiment.

The live backend, test, external folder driver, scoring and benchmark scripts are
research additions for this repository. The enhanced EKF is our supplementary
implementation, not a contribution claimed to be part of official OpenVINS.

Eigen, OpenCV, Boost, Ceres, glog, gflags and Intel MKL are external dependencies
with their own licenses. No dependency binaries or EuRoC images/IMU/ground-truth
data are redistributed here. Obtain datasets from their original provider and
follow their terms. Benchmark outputs contain generated estimates and aggregate
error measurements, not raw ground-truth trajectories.
