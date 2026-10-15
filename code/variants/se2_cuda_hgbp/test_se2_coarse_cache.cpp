#include "internal/se2_coarse.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    using namespace slam;
    _putenv_s("HGBP_COARSE_SHADOW","1");
    SyntheticSE2PackedResidualWorkspace w;
    w.num_vars=2; w.prior_lam6.assign(12,0.); w.binary_factors.resize(1);
    w.binary_factors[0].var0_id=0; w.binary_factors[0].var1_id=1;
    BasisData basis;
    basis.var_to_group={0,1}; basis.var_r_local={4,4};
    basis.var_coarse_offset={0,4}; basis.var_basis_offset={0,12};
    basis.coarse_offsets={0,4,8}; basis.coarse_dim=8;
    basis.var_basis_blocks.resize(24);
    std::mt19937 random(17);
    auto value=[&]{return static_cast<double>(static_cast<int>(random()%17)-8);};
    for(double& x:basis.var_basis_blocks) x=value();
    auto workspace=buildSyntheticSE2PackedCoarseAssemblyWorkspace(w,basis);
    for(int iteration=0;iteration<100;++iteration) {
        for(double& x:w.prior_lam6) x=value();
        auto& f=w.binary_factors[0];
        for(double& x:f.diag0_lam6) x=value();
        for(double& x:f.diag1_lam6) x=value();
        for(double& x:f.cross01_lam9) x=iteration%3?value():0.;
        if(iteration%7==0) {
            std::fill(w.prior_lam6.begin(),w.prior_lam6.end(),0.);
            std::fill_n(f.diag0_lam6,6,0.); std::fill_n(f.diag1_lam6,6,0.);
            std::fill_n(f.cross01_lam9,9,0.);
        }
        if(iteration==50) {
            basis.var_to_group={0,0}; basis.var_coarse_offset={0,0};
            basis.coarse_offsets={0,4}; basis.coarse_dim=4;
            workspace=buildSyntheticSE2PackedCoarseAssemblyWorkspace(w,basis);
        }
        _putenv_s("HGBP_CACHED_COARSE_PATTERN","0");
        const auto reference=assembleCoarseLambdaDirectPackedOrdered3D(w,basis,workspace,1);
        _putenv_s("HGBP_CACHED_COARSE_PATTERN","1");
        const auto actual=assembleCoarseLambdaDirectPackedOrdered3D(w,basis,workspace,1);
        if(actual.rows()!=reference.rows() || actual.nonZeros()!=reference.nonZeros() ||
           !std::equal(actual.outerIndexPtr(),actual.outerIndexPtr()+actual.outerSize()+1,reference.outerIndexPtr()) ||
           !std::equal(actual.innerIndexPtr(),actual.innerIndexPtr()+actual.nonZeros(),reference.innerIndexPtr()) ||
           !std::equal(actual.valuePtr(),actual.valuePtr()+actual.nonZeros(),reference.valuePtr()))
            throw std::runtime_error("coarse cache structural/value regression");
    }
    std::cout<<"100 sparse pattern, zero transition and topology-change tests passed\n";
}
