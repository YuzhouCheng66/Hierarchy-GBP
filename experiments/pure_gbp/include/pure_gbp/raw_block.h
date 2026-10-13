#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <cmath>

// Include after GBP is defined and before the executor implementations.
// Opt-in proposal kernel only; this header does not install any dispatch.
namespace pure_gbp {
namespace raw_block {

// Contiguous column-major D*D doubles, with leading dimension D. Only the
// lower triangle is read/written. On failure the scratch factor is partial.
template<int D>
inline int choleskyLowerInPlace(double* lower) {
    static_assert(D==4 || D==6 || D==12,"unsupported raw GBP block dimension");
    for(int col=0;col<D;++col) {
        double squared=0.;
        for(int k=0;k<col;++k) {
            const double value=lower[col+D*k];
            squared+=value*value;
        }
        const double pivot=lower[col+D*col]-squared;
        // Match local Eigen LLT's test, including its NaN comparison behavior.
        // Do not introduce a pivot tolerance, jitter, or a stronger finite check.
        if(pivot<=0.) return 1;
        const double diagonal=std::sqrt(pivot);
        lower[col+D*col]=diagonal;
        for(int row=col+1;row<D;++row) {
            double product=0.;
            for(int k=0;k<col;++k) product+=lower[row+D*k]*lower[col+D*k];
            lower[row+D*col]=(lower[row+D*col]-product)/diagonal;
        }
    }
    return 0;
}

// Solve (L L^T) X = B in place for contiguous column-major D*RhsCols doubles.
// The factor must come from a successful choleskyLowerInPlace<D>; buffers must
// not overlap. No alignment requirement, wrapper, inverse, or additional check.
template<int D,int RhsCols=D>
inline void choleskySolveInPlace(const double* lower,double* rhs) {
    static_assert(D==4 || D==6 || D==12,"unsupported raw GBP block dimension");
    static_assert(RhsCols>0,"at least one right-hand side is required");
    for(int col=0;col<RhsCols;++col) {
        double* x=rhs+D*col;
        for(int row=0;row<D;++row) {
            double product=0.;
            for(int k=0;k<row;++k) product+=lower[row+D*k]*x[k];
            x[row]=(x[row]-product)/lower[row+D*row];
        }
        for(int row=D-1;row>=0;--row) {
            double product=0.;
            for(int k=row+1;k<D;++k) product+=lower[k+D*row]*x[k];
            x[row]=(x[row]-product)/lower[row+D*row];
        }
    }
}

} // namespace raw_block

// Same call and mutation contract as fixedBlockFullCompute<D>. The caller
// dispatches only non-frozen, equal-D endpoints and retains commit ordering.
// Return 1 on a failed factorization or nonfinite next_eta, otherwise 0.
// Reduction order differs from Eigen; this is not a bitwise-equivalent path.
template<int D>
inline int rawBlockFullCompute(GBP& workspace,int edge_index,double damping,double& change) {
    static_assert(D==4 || D==6 || D==12,"unsupported raw GBP block dimension");
    using BlockMat=Eigen::Matrix<double,D,D,Eigen::ColMajor>;
    using BlockVec=Eigen::Matrix<double,D,1>;

    auto& e=workspace.edges[edge_index];
    const auto& reverse=workspace.edges[e.reverse];
    assert(workspace.size(e.from)==D && workspace.size(e.to)==D);

    BlockMat lower=workspace.belief[e.from]-reverse.lam;
    if(raw_block::choleskyLowerInPlace<D>(lower.data())) return 1;

    const BlockMat cross=e.cross;
    BlockMat solved=cross;
    raw_block::choleskySolveInPlace<D,D>(lower.data(),solved.data());
    const BlockMat eta_map=-solved.transpose();
    e.eta_map=eta_map;

    // Keep the proposal algebra/evaluation boundaries from the fixed kernel.
    const BlockMat lam=e.lam;
    BlockMat next_lam=(1.-damping)*lam+damping*(eta_map*cross);
    next_lam=.5*(next_lam+next_lam.transpose()).eval();
    e.next_lam=next_lam;
    change=(next_lam-lam).norm()/std::max(1.,next_lam.norm());

    const BlockVec cavity_eta=workspace.belief_eta[e.from]-reverse.eta;
    const BlockVec eta=e.eta;
    const BlockVec next_eta=(1.-damping)*eta+damping*(eta_map*cavity_eta);
    e.next_eta=next_eta;
    return next_eta.allFinite()?0:1;
}


} // namespace pure_gbp
