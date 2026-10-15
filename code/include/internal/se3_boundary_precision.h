#pragma once

#include "internal/se3_residual.h"
#include "gbp/Factor.h"
#include "gbp/VariableNode.h"
#include <stdexcept>

namespace slam {

// Validate once outside the parallel basis loop. Packed binary edge e owns
// slots 2*e and 2*e+1 in the same local-variable order as the canonical factor.
inline void validateSE3BoundarySlots(const gbp::FactorGraph& graph,
                                    const SyntheticSE3PackedSoAWorkspace& w) {
    if (w.num_vars != static_cast<int>(graph.var_nodes.size()) ||
        w.num_binary_factors > static_cast<int>(graph.factors.size()) ||
        w.binary_msg_lam21.size() != static_cast<size_t>(42*w.num_binary_factors))
        throw std::runtime_error("SE3 boundary workspace size mismatch");
    for (int e=0; e<w.num_binary_factors; ++e) {
        const auto& f=*graph.factors[e];
        if (f.factorID!=e || f.adj_var_nodes.size()!=2 ||
            f.adj_var_nodes[0]->dofs!=6 || f.adj_var_nodes[1]->dofs!=6 ||
            f.adj_var_nodes[0]->variableID!=w.binary_var0_id[e] ||
            f.adj_var_nodes[1]->variableID!=w.binary_var1_id[e])
            throw std::runtime_error("SE3 boundary slot topology mismatch");
    }
}

inline Eigen::Matrix<double,6,6> se3PackedBoundaryPrecision(
    const SyntheticSE3PackedSoAWorkspace& w, int factor_id, int local_index) {
    Eigen::Matrix<double,6,6> result;
    const double* packed=w.binary_msg_lam21.data()+21*(2*factor_id+local_index);
    for (int c=0; c<6; ++c)
        for (int r=0; r<=c; ++r)
            result(r,c)=result(c,r)=packed[c*(c+1)/2+r];
    return result;
}

} // namespace slam
