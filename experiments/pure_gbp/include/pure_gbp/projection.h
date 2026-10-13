#pragma once
#include "solver.h"
#include <vector>
#include <omp.h>

namespace pure_gbp {
// Preserve each fine factor's diagonal budget and the original per-pair
// reduction order. The correction RHS is assigned to canonical unary terms.
template<int D> class PairProjection {
    struct Term {
        Eigen::Matrix<double,D,D> first,second;
        Eigen::Matrix<double,D,Eigen::Dynamic> bi,bj;
    };
    std::map<std::pair<int,int>,std::vector<Term>> terms;
public:
    void add(int i,int j,Term term) {
        if(i>j) {std::swap(i,j);term.first.swap(term.second);std::swap(term.bi,term.bj);}
        terms[{i,j}].push_back(std::move(term));
    }
    template<class A,class B,class BI,class BJ>
    void add(int i,int j,const A& a,const B& b,const BI& bi,const BJ& bj) {
        add(i,j,Term{a,b,bi,bj});
    }
    PairDiagonals finish(int threads) const {
        PairDiagonals pairs;
        struct Slot {PairData* out;const std::vector<Term>* input;};
        std::vector<Slot> slots;
        for(const auto& item:terms) {
            auto& out=pairs[item.first];
            out.eta_first=Eigen::VectorXd::Zero(item.second[0].bi.cols());
            out.eta_second=Eigen::VectorXd::Zero(item.second[0].bj.cols());
            slots.push_back({&out,&item.second});
        }
#pragma omp parallel for num_threads(threads) if(threads>1) schedule(static)
        for(int k=0;k<int(slots.size());++k) {
            auto& out=*slots[k].out;bool first=true;
            for(const auto& t:*slots[k].input) {
                Eigen::MatrixXd a=t.bi.transpose()*t.first*t.bi,b=t.bj.transpose()*t.second*t.bj;
                if(first) {out.first=std::move(a);out.second=std::move(b);first=false;}
                else {out.first+=a;out.second+=b;}
            }
        }
        return pairs;
    }
};

template<class Workspace,class Basis>
PairDiagonals projectSE2(const Workspace& ws,const Basis& basis,int threads) {
    PairProjection<3> projection;
    const auto unpack=[](const double* s) {Eigen::Matrix3d a;a<<s[0],s[1],s[2],s[1],s[3],s[4],s[2],s[4],s[5];return a;};
    for(const auto& f:ws.binary_factors) {
        const int i=f.var0_id,j=f.var1_id,gi=basis.var_to_group[i],gj=basis.var_to_group[j];
        if(gi==gj)continue;
        const Eigen::Map<const Eigen::Matrix<double,3,Eigen::Dynamic>> bi(
            basis.var_basis_blocks.data()+basis.var_basis_offset[i],3,basis.var_r_local[i]);
        const Eigen::Map<const Eigen::Matrix<double,3,Eigen::Dynamic>> bj(
            basis.var_basis_blocks.data()+basis.var_basis_offset[j],3,basis.var_r_local[j]);
        projection.add(gi,gj,unpack(f.diag0_lam6),unpack(f.diag1_lam6),bi,bj);
    }
    return projection.finish(threads);
}

template<class Graph,class Basis>
PairDiagonals projectSE3(const Graph& graph,const Basis& basis,int threads) {
    PairProjection<6> projection;
    for(const auto& fp:graph.factors) {
        if(!fp || !fp->active || fp->adj_var_nodes.size()!=2)continue;
        const auto a=fp->adj_var_nodes[0],b=fp->adj_var_nodes[1];
        const int i=a->variableID,j=b->variableID,gi=basis.var_to_group[i],gj=basis.var_to_group[j];
        if(gi==gj)continue;
        if(a->dofs!=6 || b->dofs!=6)throw std::runtime_error("pure GBP SE3 projection requires 6D variables");
        projection.add(gi,gj,fp->factor.lam().template topLeftCorner<6,6>(),
            fp->factor.lam().template bottomRightCorner<6,6>(),
            basis.local_bases[gi].middleRows(basis.var_to_local_offset[i],6),
            basis.local_bases[gj].middleRows(basis.var_to_local_offset[j],6));
    }
    return projection.finish(threads);
}
}
