#pragma once

// Public SE3 entry point for the NeurIPS H-GBP code package.
// The implementation is in src/se3_solver.cpp and follows the verified
// benchmark path used during packaging.
#include "internal/se3_solver_impl.h"

namespace hgbp::se3 {

using Pose = slam::SE3Pose;
using PoseVector = slam::SE3PoseVector;
using Edge = slam::SyntheticSE3Edge;
using Problem = slam::SyntheticSE3Problem;
using RobustLossConfig = slam::RobustLossConfig;
using Results = slam::SyntheticSE3ExperimentResults;

using slam::loadSyntheticSE3Problem;
using slam::nonlinearObjective;
using slam::runSyntheticSE3Experiment;
using slam::writeSyntheticSE3ExperimentPoseHistoryJson;
using slam::writeSyntheticSE3ExperimentResultsJson;

}  // namespace hgbp::se3
