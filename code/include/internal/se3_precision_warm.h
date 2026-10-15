#pragma once
#include "internal/se3_residual.h"

namespace slam {
struct SE3ChartTransition {
    Eigen::Matrix<double,6,1> offset;
    Eigen::Matrix<double,6,6> jacobian;
};
SE3ChartTransition se3ChartTransition(const SE3Pose& previous,const SE3Pose& current);

struct SE3PrecisionWarmState {
    SE3PoseVector poses;
    SyntheticSE3PackedSoAWorkspace::AlignedDoubles precision;
    SyntheticSE3PackedSoAWorkspace::AlignedDoubles eta;
    void capture(const SyntheticSE3PackedSoAWorkspace& w,const SE3PoseVector& base,
        const Eigen::VectorXd* corrected_mean=nullptr,int threads=1);
};
struct SE3PrecisionWarmStats {
    int slots=0;
    int projected=0;
    int failed=0;
    double max_projection_relative=0;
};
SE3PrecisionWarmStats initializeWarmSE3Precision(SyntheticSE3PackedSoAWorkspace& w,
    const SE3PrecisionWarmState& previous,const SE3PoseVector& current,bool bounded_mean,int threads,bool transport_eta=false);
// Diagnostic only: the exact shadow solution never modifies the live iterate.
void auditSE3PrecisionWarm(const std::string& path,const gbp::FactorGraph& graph,
    const SyntheticSE3PackedSoAWorkspace& initial,const SE3PrecisionWarmState& previous,
    const SE3PoseVector& current,int threads);
}
