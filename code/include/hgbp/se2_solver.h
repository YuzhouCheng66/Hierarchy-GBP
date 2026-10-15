#pragma once

// Public SE2 entry point for the NeurIPS H-GBP code package.
// The implementation keeps the verified namespace used by the benchmark path.
#include "internal/se2_solver_impl.h"

namespace hgbp::se2 {

using Pose = Eigen::Vector3d;
using Edge = slam::SyntheticSE2Edge;
using Problem = slam::SyntheticSE2Problem;
using BasisBuildConfig = slam::BasisBuildConfig;
using RobustLossConfig = slam::RobustLossConfig;
using Results = slam::ExperimentResults;

using slam::loadSyntheticSE2Problem;
using slam::nonlinearObjective;
using slam::runSyntheticSE2Experiment;
using slam::writeExperimentPoseHistoryJson;
using slam::writeExperimentResultsJson;

}  // namespace hgbp::se2
