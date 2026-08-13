#pragma once

#include <vector>

#include "gbp/FactorGraph.h"

namespace slam {

struct SyntheticSE2FastSyncSoALayout {
    bool all_dofs3 = false;

    std::vector<gbp::Factor*> binary_factors;
    std::vector<gbp::Factor*> fallback_factors;

    std::vector<const double*> belief0_eta_ptrs;
    std::vector<const double*> belief1_eta_ptrs;
    std::vector<const double*> belief0_lam_ptrs;
    std::vector<const double*> belief1_lam_ptrs;

    std::vector<double> eta0;
    std::vector<double> eta1;
    std::vector<double> block00;
    std::vector<double> block03;
    std::vector<double> block30;
    std::vector<double> block33;

    std::vector<double> msg_eta;
    std::vector<double> msg_lam;

    std::vector<gbp::VariableNode*> variables;
    std::vector<const double*> prior_eta_ptrs;
    std::vector<const double*> prior_lam_ptrs;
    std::vector<double*> belief_eta_ptrs;
    std::vector<double*> belief_lam_ptrs;
    std::vector<double*> mu_ptrs;

    std::vector<int> binary_offsets;
    std::vector<int> binary_slot_ids;
    std::vector<int> fallback_offsets;
    std::vector<double* const*> fallback_eta_slots;
    std::vector<double* const*> fallback_lam_slots;

    std::vector<int> factor_thread_offsets;
    std::vector<int> variable_thread_offsets;
    std::vector<double> factor_thread_rates;
    std::vector<double> variable_thread_rates;
    std::vector<double> factor_thread_work_last;
    std::vector<double> variable_thread_work_last;
};

SyntheticSE2FastSyncSoALayout buildSyntheticSE2FastSyncSoALayout(gbp::FactorGraph& graph);
void refreshSyntheticSE2FastSyncSoALayout(SyntheticSE2FastSyncSoALayout& layout);

void synchronousIterationFastSyncSoA(
    SyntheticSE2FastSyncSoALayout& layout,
    double eta_damping,
    bool update_mu,
    int num_threads,
    double& factor_pass_sec_accum,
    double& variable_pass_sec_accum
);

}  // namespace slam
