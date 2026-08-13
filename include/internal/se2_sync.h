#pragma once

#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "gbp/FactorGraph.h"

namespace slam {

struct SyntheticSE2FastSyncLocalLayout {
    bool supported = false;

    std::vector<gbp::VariableNode*> variables;

    std::vector<gbp::Factor*> unary_factors;
    std::vector<gbp::Factor*> binary_factors;

    std::vector<int> unary_var_ids;
    std::vector<int> binary_var0_ids;
    std::vector<int> binary_var1_ids;

    std::vector<int> unary_offsets;
    std::vector<int> unary_ids;
    std::vector<int> binary_offsets;
    std::vector<int> binary_slot_ids;
    std::vector<double*> unary_incoming_eta_slots;
    std::vector<double*> unary_incoming_lam6_slots;
    std::vector<double*> binary_incoming_eta_slots;
    std::vector<double*> binary_incoming_lam_slots;

    std::vector<double> prior_eta;
    std::vector<double> prior_lam6;
    std::vector<double> belief_eta;
    std::vector<double> belief_lam6;
    std::vector<double> mu;
    std::vector<uint8_t> mu_valid;
    std::vector<double*> prior_eta_ptrs;
    std::vector<double*> prior_lam6_ptrs;
    std::vector<double*> belief_eta_ptrs;
    std::vector<double*> belief_lam6_ptrs;
    std::vector<double*> mu_ptrs;

    std::vector<double> unary_eta;
    std::vector<double> unary_lam6;
    std::vector<double> unary_msg_eta;
    std::vector<double> unary_msg_lam6;
    std::vector<double*> unary_msg_eta_ptrs;
    std::vector<double*> unary_msg_lam6_ptrs;

    std::vector<double> eta0;
    std::vector<double> eta1;
    std::vector<double> block00;
    std::vector<double> block03;
    std::vector<double> block30;
    std::vector<double> block33;
    std::vector<double> binary_msg_eta;
    std::vector<double> binary_msg_lam;
    std::vector<double*> binary_belief0_eta_ptrs;
    std::vector<double*> binary_belief1_eta_ptrs;
    std::vector<double*> binary_belief0_lam6_ptrs;
    std::vector<double*> binary_belief1_lam6_ptrs;
};

SyntheticSE2FastSyncLocalLayout buildSyntheticSE2FastSyncLocalLayout(gbp::FactorGraph& graph);
void refreshSyntheticSE2FastSyncLocalLayout(SyntheticSE2FastSyncLocalLayout& layout);

void synchronousIterationFastSyncLocal(
    SyntheticSE2FastSyncLocalLayout& layout,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum
);

Eigen::VectorXd stackedMeanVectorFastSyncLocal(SyntheticSE2FastSyncLocalLayout& layout);
void injectCorrectionKeepMessagesFastSyncLocal(
    SyntheticSE2FastSyncLocalLayout& layout,
    const Eigen::VectorXd& delta
);

}  // namespace slam
