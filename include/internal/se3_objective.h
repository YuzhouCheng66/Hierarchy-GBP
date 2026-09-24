#pragma once
#include "internal/se3_solver_impl.h"

namespace slam {
struct SE3ObjectiveValues {
    double raw=0;
    double huber=0;
};
struct SE3ObjectiveWorkspace {
    std::vector<SE3ObjectiveValues> edge_values;
};
// The gauge prior remains quadratic. Only measured edges are robustified.
SE3ObjectiveValues evaluateSE3Objectives(const SyntheticSE3Problem& problem,
                                       const SE3PoseVector& poses,
                                       const RobustLossConfig& config);
SE3ObjectiveValues evaluateSE3ObjectivesBuffered(const SyntheticSE3Problem& problem,
    const SE3PoseVector& poses,const RobustLossConfig& config,SE3ObjectiveWorkspace& workspace,int threads);
void applySE3PoseDeltasInto(const SE3PoseVector& base,const Eigen::VectorXd& delta,
                          double scale,SE3PoseVector& output,int threads);
}
