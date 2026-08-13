#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "gbp/FactorGraph.h"

namespace slam {

struct SyntheticSE2PackedBinaryFactorData {
    int var0_id = -1;
    int var1_id = -1;
    double eta0[3] = {};
    double eta1[3] = {};
    double diag0_lam6[6] = {};
    double diag1_lam6[6] = {};
    double cross01_lam9[9] = {};
};

struct SyntheticSE2FastSyncLocalPackedLayout {
    bool supported = false;

    std::vector<gbp::VariableNode*> variables;
    std::vector<gbp::Factor*> unary_factors;
    std::vector<gbp::Factor*> binary_factors;

    std::vector<int> unary_var_ids;
    std::vector<int> unary_offsets;
    std::vector<int> unary_ids;
    std::vector<int> binary_offsets;
    std::vector<int> binary_slot_ids;

    std::vector<double> prior_eta;
    std::vector<double> prior_lam6;
    std::vector<double> belief_eta;
    std::vector<double> belief_lam6;
    std::vector<double> mu;
    std::vector<uint8_t> mu_valid;

    std::vector<double> unary_eta;
    std::vector<double> unary_lam6;
    std::vector<double> unary_msg_eta;
    std::vector<double> unary_msg_lam6;

    std::vector<SyntheticSE2PackedBinaryFactorData> binary_data;
    std::vector<double> binary_msg_eta;
    std::vector<double> binary_msg_lam6;
};

SyntheticSE2FastSyncLocalPackedLayout buildSyntheticSE2FastSyncLocalPackedLayout(gbp::FactorGraph& graph);
void refreshSyntheticSE2FastSyncLocalPackedLayout(SyntheticSE2FastSyncLocalPackedLayout& layout);

void synchronousIterationFastSyncLocalPacked(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum
);

Eigen::VectorXd stackedMeanVectorFastSyncLocalPacked(SyntheticSE2FastSyncLocalPackedLayout& layout);
void injectCorrectionKeepMessagesFastSyncLocalPacked(
    SyntheticSE2FastSyncLocalPackedLayout& layout,
    const Eigen::VectorXd& delta
);

}  // namespace slam
