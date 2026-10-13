#pragma once

#include "gbp.h"
#include "policy.h"
#include "internal/partial_symmetric_eigen.h"

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pure_gbp {

struct ThirdConfig {
    int group_size=16,rank=0,cycles=3,pre_sweeps=40,top_sweeps=400;
    int top_serial_limit=256;
    double top_eta_relaxation=1.;
    double tolerance=1e-8,precision_tolerance=1e-12;
    bool trace=false;
};

struct ThirdStats {
    Stats lower,top;
    int cycles=0,rejected=0;
    int extrapolated_steps=0;
    double maximum_energy_step=0;
    double build_sec=0,lower_sec=0,top_sec=0,transfer_sec=0,residual=1;
    double energy=0;
    double before_final=0,after_final=0;
    std::vector<double> after_pre,after_top,top_residual,top_tail;
    // Exclusive of lower_sec/top_sec/transfer_sec; preparation includes swaps,
    // scratch sizing and RHS setup.
    double prepare_sec=0,restore_sec=0,diagnostic_sec=0;
};

// Residual correction on the unchanged lower equation. The top equation is
// solved by GBP as well, never by a direct or Krylov fallback.
class ThirdLevel {
public:
    ThirdConfig config;
    Sparse prolongation,matrix;
    std::vector<int> group,local;
    std::vector<std::vector<int>> members;
    std::vector<Mat> basis;
    std::unique_ptr<GBP> top;
    std::map<std::pair<int,int>,PairData> pairs;
    int rank=0;
    int solves=0;
    double build_sec=0,orthogonality_error=0;
    // Disjoint constructor phases; build_sec also includes bookkeeping.
    double group_sec=0,basis_sec=0,prolongation_sec=0,galerkin_sec=0;
    double budget_projection_sec=0,top_prepare_sec=0;
    // Fallbacks include skipped small/full-rank groups. Reduce in group order.
    int partial_attempts=0,partial_successes=0,partial_fallbacks=0,partial_skipped=0;
    int partial_iterations=0,partial_spd_rejections=0,partial_lowest_rejections=0;
    double partial_max_residual=0,partial_max_accepted_residual=0;

    ThirdLevel(const GBP& lower,const ThirdConfig& settings):config(settings) {
        const auto start=Clock::now();
        if(settings.group_size<2 || settings.cycles<1 || settings.pre_sweeps<1 || settings.top_sweeps<1)
            throw std::runtime_error("invalid third-level budget");
        if(lower.initial_lam.size()!=lower.edges.size())
            throw std::runtime_error("third level requires original projected factor budgets");
        rank=settings.rank>0?settings.rank:std::min(6,lower.size(0));
        for(int i=0;i<lower.blocks;++i) if(lower.size(i)<rank)
            throw std::runtime_error("third-level rank exceeds a source block size");
        auto phase=Clock::now();
        group.assign(lower.blocks,-1);local.resize(lower.blocks);
        // Deterministic connected aggregates, strongest boundary coupling first.
        for(int seed=0;seed<lower.blocks;++seed) if(group[seed]<0) {
            const int g=int(members.size());members.emplace_back();
            std::map<int,double> frontier;
            auto take=[&](int i) {
                group[i]=g;members.back().push_back(i);frontier.erase(i);
                for(int p:lower.incoming[i]) {
                    int j=lower.edges[p].from;
                    if(group[j]<0) frontier[j]+=lower.edges[p].cross.squaredNorm();
                }
            };
            take(seed);
            while(int(members.back().size())<settings.group_size && !frontier.empty()) {
                auto best=frontier.begin();
                for(auto it=frontier.begin();it!=frontier.end();++it)
                    if(it->second>best->second) best=it;
                take(best->first);
            }
        }
        group_sec=seconds(phase);phase=Clock::now();
        basis.resize(members.size());std::vector<int> bad(members.size(),0);
        auto localMatrix=[&](int g,Mat& h) {
            int dimension=0;
            for(int i:members[g]) {local[i]=dimension;dimension+=lower.size(i);}
            h.setIdentity(dimension,dimension);
            for(int i:members[g]) for(int p:lower.incoming[i]) {
                const auto& e=lower.edges[p];const int j=e.from;
                if(group[j]==g) h.block(local[i],local[j],lower.size(i),lower.size(j))=e.cross.transpose();
                else h.block(local[i],local[i],lower.size(i),lower.size(i))+=lower.initial_lam[p];
            }
            h=.5*(h+h.transpose()).eval();
        };
        std::vector<PartialBasisOutcome> outcomes(members.size());
#pragma omp parallel num_threads(lower.threads) if(lower.threads>1)
        {
            Mat h;
            PartialBasisWorkspace partial_ws;
            Eigen::SelfAdjointEigenSolver<Mat> full_eig;
#pragma omp for schedule(dynamic,1)
            for(int g=0;g<int(members.size());++g) {
                localMatrix(g,h);
                auto& outcome=outcomes[g];
                outcome=tryPartialBasis(h,rank,partial_ws,basis[g]);
                if(outcome.accepted)continue;
                full_eig.compute(h);
                if(full_eig.info()!=Eigen::Success) {bad[g]=1;continue;}
                basis[g]=full_eig.eigenvectors().leftCols(rank);
            }
        }
        for(const auto& outcome:outcomes) {
            partial_attempts+=int(outcome.attempted);
            partial_successes+=int(outcome.accepted);
            partial_fallbacks+=int(!outcome.accepted);
            partial_skipped+=int(!outcome.attempted);
            partial_iterations+=outcome.iterations;
            partial_spd_rejections+=int(outcome.spd_rejected);
            partial_lowest_rejections+=int(outcome.lowest_rejected);
            partial_max_residual=std::max(partial_max_residual,outcome.residual);
            if(outcome.accepted)
                partial_max_accepted_residual=std::max(partial_max_accepted_residual,outcome.residual);
        }
        if(std::accumulate(bad.begin(),bad.end(),0)) throw std::runtime_error("third-level local eigensolver failed");
        basis_sec=seconds(phase);phase=Clock::now();
        prolongation.resize(lower.n,int(members.size())*rank);
        // Stream unique CSC entries in global row order, retaining explicit zeros.
        prolongation.reserve(Eigen::Index(lower.n)*rank);
        std::vector<int> ordered;
        for(int g=0;g<int(members.size());++g) {
            orthogonality_error=std::max(orthogonality_error,(basis[g].transpose()*basis[g]-Mat::Identity(rank,rank)).norm());
            ordered=members[g];std::sort(ordered.begin(),ordered.end());
            for(int c=0;c<rank;++c) {
                prolongation.startVec(g*rank+c);
                for(int i:ordered) for(int r=0;r<lower.size(i);++r)
                    prolongation.insertBack(lower.offsets[i]+r,g*rank+c)=basis[g](local[i]+r,c);
            }
        }
        prolongation.finalize();
        prolongation_sec=seconds(phase);phase=Clock::now();
        // Preserve the left-associated Galerkin product and symmetrization.
        matrix=prolongation.transpose()*lower.normalized*prolongation;
        matrix=(.5*(matrix+Sparse(matrix.transpose()))).eval();
        galerkin_sec=seconds(phase);phase=Clock::now();
        const Vec zero=Vec::Zero(rank);
        Mat projected_i,projected_j;
        for(int p=0;p<int(lower.edges.size());p+=2) {
            const auto& e=lower.edges[p];const int i=e.from,j=e.to,gi=group[i],gj=group[j];
            if(gi==gj) continue;
            const auto bi=basis[gi].middleRows(local[i],lower.size(i));
            const auto bj=basis[gj].middleRows(local[j],lower.size(j));
            projected_i.noalias()=-bi.transpose()*lower.initial_lam[e.reverse]*bi;
            projected_j.noalias()=-bj.transpose()*lower.initial_lam[p]*bj;
            const auto key=std::make_pair(std::min(gi,gj),std::max(gi,gj));
            const Mat& first=gi<gj?projected_i:projected_j;
            const Mat& second=gi<gj?projected_j:projected_i;
            auto found=pairs.lower_bound(key);
            if(found==pairs.end() || key<found->first)
                pairs.emplace_hint(found,key,PairData{first,second,zero,zero});
            else {found->second.first+=first;found->second.second+=second;}
        }
        budget_projection_sec=seconds(phase);phase=Clock::now();
        top=std::make_unique<GBP>(matrix,rank,lower.threads,false);
        top->useParallelTreeSchedule();top->packed_eta_enabled=true;top->stable_master=true;
        top->native_serial=top->n<=settings.top_serial_limit;
        top->eta_relaxation=settings.top_eta_relaxation;
        top->projectedInitialization(pairs);
        top_prepare_sec=seconds(phase);
        build_sec=seconds(start);
    }

    static void addStats(Stats& dst,const Stats& src) {
        dst.sweeps+=src.sweeps;dst.full_sweeps+=src.full_sweeps;dst.fixed_sweeps+=src.fixed_sweeps;
        dst.full_sec+=src.full_sec;dst.fixed_sec+=src.fixed_sec;
        dst.numerical_failure=dst.numerical_failure || src.numerical_failure;
        dst.spd=dst.spd && src.spd;dst.observed_threads=std::max(dst.observed_threads,src.observed_threads);
    }

    ThirdStats solve(GBP& lower,const Vec& q,Vec& z) {
        return solveWorkspace(lower,q,z);
    }

private:
    struct PartialBasisOutcome {
        bool attempted=false,accepted=false,spd_rejected=false,lowest_rejected=false;
        int iterations=0;
        double residual=0;
    };
    struct PartialBasisWorkspace {
        slam::PartialSymmetricEigenWorkspace eigen;
        Mat defect,gram,lifted;
        Eigen::LLT<Mat> lowest_check;
    };

    static PartialBasisOutcome tryPartialBasis(const Mat& h,int retained,PartialBasisWorkspace& ws,Mat& result) {
        PartialBasisOutcome out;
        const int n=int(h.rows());
        if(n<std::max(24,2*(retained+4)) || retained>=n || !h.allFinite())return out;
        const double scale=std::max(1.,h.norm());
        if(!std::isfinite(scale))return out;
        slam::PartialSymmetricEigenOptions options;
        options.oversampling=4;options.max_iters=12;options.residual_tol=1e-7;
        options.ridge=1e-10;options.residual_check_period=1;options.predict_failed_budget=true;
        out.attempted=true;
        // Re-seed every group; reused work arrays are never a numerical warm start.
        const auto partial=slam::computeSmallestEigenpairsPartial(h,retained,nullptr,ws.eigen,options);
        out.iterations=partial.iterations;
        if(std::isfinite(partial.max_relative_residual))out.residual=partial.max_relative_residual;
        // Inverse iteration needs an SPD shifted operator to order lowest modes.
        if(ws.eigen.ldlt.info()!=Eigen::Success || !ws.eigen.ldlt.vectorD().allFinite() ||
           !(ws.eigen.ldlt.vectorD().minCoeff()>0)) {
            out.spd_rejected=true;return out;
        }
        if(!partial.converged || partial.aborted_for_work ||
           partial.eigenvectors.rows()!=n || partial.eigenvectors.cols()!=retained ||
           partial.eigenvalues.size()!=retained || !partial.eigenvectors.allFinite() ||
           !partial.eigenvalues.allFinite())return out;
        const auto& u=partial.eigenvectors;const auto& values=partial.eigenvalues;
        for(int c=1;c<retained;++c)if(values[c]<values[c-1])return out;
        ws.gram.noalias()=u.transpose()*u;
        ws.gram.diagonal().array()-=1.;
        const double orthogonality=ws.gram.norm();
        if(!std::isfinite(orthogonality) || orthogonality>1e-10)return out;
        ws.defect.noalias()=h*u;
        for(int c=0;c<retained;++c)ws.defect.col(c).noalias()-=values[c]*u.col(c);
        if(!ws.defect.allFinite())return out;
        double checked_residual=0;
        for(int c=0;c<retained;++c) {
            const double residual=ws.defect.col(c).norm()/std::max(1.,std::abs(values[c]));
            if(!std::isfinite(residual))return out;
            checked_residual=std::max(checked_residual,residual);
        }
        out.residual=checked_residual;
        if(checked_residual>options.residual_tol)return out;

        // A small Ritz residual can miss a low invariant subspace. This positive
        // rank-k update certifies lowest-mode selection, with conservative margins
        // for residuals, roundoff and clusters straddling the cutoff.
        const double roundoff=128.*std::numeric_limits<double>::epsilon()*n*scale;
        const double cutoff=values[retained-1]+4.*(ws.defect.norm()+orthogonality*scale+roundoff);
        const double lift=2.*scale+std::abs(cutoff);
        if(!std::isfinite(cutoff) || !std::isfinite(lift))return out;
        ws.lifted=h;ws.lifted.diagonal().array()-=cutoff;
        ws.lifted.noalias()+=lift*u*u.transpose();
        if(!ws.lifted.allFinite())return out;
        ws.lowest_check.compute(ws.lifted);
        if(ws.lowest_check.info()!=Eigen::Success ||
           !(ws.lowest_check.matrixLLT().diagonal().array().square().minCoeff()>roundoff)) {
            out.lowest_rejected=true;return out;
        }
        result=u;out.accepted=true;
        return out;
    }

    struct SolveWorkspace {
        Vec residual,delta,product,coarse_rhs,top_rhs,coarse_delta,recovered;

        void prepare(const GBP& lower,const GBP& top) {
            residual.resize(lower.n);delta.resize(lower.n);product.resize(lower.n);
            coarse_rhs.resize(top.n);top_rhs.resize(top.n);
            coarse_delta.resize(top.n);recovered.resize(top.n);
        }
    } scratch;

    // Canonical GBP ignores initial_eta while initial_lam is empty. Hide precision
    // with an O(1) swap and restore it and warm_eta even if a solve throws.
    struct ScopedInitialMessages {
        GBP& lower;
        std::vector<Mat> precision;
        bool warm,active=true;

        explicit ScopedInitialMessages(GBP& source):lower(source),warm(source.warm_eta) {
            precision.swap(lower.initial_lam);
        }
        void restore() noexcept {
            if(!active)return;
            precision.swap(lower.initial_lam);
            lower.warm_eta=warm;active=false;
        }
        ~ScopedInitialMessages() {restore();}
    };

    ThirdStats solveWorkspace(GBP& lower,const Vec& q,Vec& z) {
        auto start=Clock::now();
        ThirdStats stats;z=Vec::Zero(lower.n);
        const double q_norm=q.norm();
        if(q_norm==0) {stats.residual=0;stats.prepare_sec=seconds(start);return stats;}
        if(lower.initial_eta.size()!=lower.edges.size())
            throw std::runtime_error("third level requires projected linear terms");
        // Scratch belongs to this hierarchy and is overwritten before every read.
        scratch.prepare(lower,*top);
        if(config.trace) {
            stats.after_pre.reserve(config.cycles);stats.after_top.reserve(config.cycles);
            stats.top_residual.reserve(config.cycles);stats.top_tail.reserve(config.cycles);
        }
        ScopedInitialMessages restore(lower);
        top->initial_lam.clear();
        Vec& r=scratch.residual;Vec& delta=scratch.delta;Vec& product=scratch.product;
        r=q;
        stats.prepare_sec=seconds(start);
        auto addCorrection=[&](const Vec& correction) {
            product.noalias()=lower.normalized*correction;
            double gain=correction.dot(r),curvature=correction.dot(product);
            if(!correction.allFinite() || !std::isfinite(gain) || !std::isfinite(curvature) || !(gain>0) || !(curvature>0)) {
                ++stats.rejected;return;
            }
            const double optimum=gain/curvature;
            if(!std::isfinite(optimum)) {++stats.rejected;return;}
            const double alpha=std::min(1.,optimum);
            stats.maximum_energy_step=std::max(stats.maximum_energy_step,alpha);
            stats.extrapolated_steps+=alpha>1.;
            z.noalias()+=alpha*correction;r.noalias()-=alpha*product;
        };
        for(int cycle=0;cycle<config.cycles;++cycle) {
            start=Clock::now();
            lower.warm_eta=false;
            const bool reuse=cycle>0 || solves>0;
            auto fine=lower.solve(r,delta,config.pre_sweeps,1.,config.tolerance,0,reuse,config.precision_tolerance);
            stats.lower_sec+=seconds(start);addStats(stats.lower,fine);
            if(!fine.spd)throw std::runtime_error("third-level lower GBP lost SPD");
            if(fine.numerical_failure)throw std::runtime_error("third-level lower GBP numerical failure");
            start=Clock::now();addCorrection(delta);
            if(config.trace)stats.after_pre.push_back(r.norm()/q_norm);
            scratch.coarse_rhs.noalias()=prolongation.transpose()*r;
            stats.transfer_sec+=seconds(start);
            start=Clock::now();
            for(int i=0;i<top->blocks;++i)
                scratch.top_rhs.segment(top->offsets[i],top->size(i))=
                    top->lower[i].triangularView<Eigen::Lower>().solve(scratch.coarse_rhs.segment(top->offsets[i],top->size(i)));
            auto upper=top->solve(scratch.top_rhs,scratch.coarse_delta,config.top_sweeps,1.,config.tolerance,0,reuse,config.precision_tolerance);
            if(config.trace) {stats.top_residual.push_back(upper.residual);stats.top_tail.push_back(upper.tail_ratio);}
            stats.top_sec+=seconds(start);addStats(stats.top,upper);
            if(!upper.spd)throw std::runtime_error("third-level top GBP lost SPD; no fallback");
            if(upper.numerical_failure)throw std::runtime_error("third-level top GBP numerical failure; no fallback");
            start=Clock::now();
            for(int i=0;i<top->blocks;++i)
                scratch.recovered.segment(top->offsets[i],top->size(i))=
                    top->lower[i].transpose().triangularView<Eigen::Upper>().solve(scratch.coarse_delta.segment(top->offsets[i],top->size(i)));
            delta.noalias()=prolongation*scratch.recovered;addCorrection(delta);
            ++stats.cycles;stats.residual=r.norm()/q_norm;
            if(config.trace)stats.after_top.push_back(stats.residual);
            stats.transfer_sec+=seconds(start);
            if(stats.residual<=config.tolerance)break;
        }
        stats.before_final=stats.residual;
        stats.after_final=stats.before_final;
        start=Clock::now();
        // Diagnostics use this call's actual Az, not the recursively updated r.
        product.noalias()=lower.normalized*z;
        stats.residual=(q-product).norm()/q_norm;
        stats.energy=.5*z.dot(product)-z.dot(q);
        stats.diagnostic_sec=seconds(start);
        start=Clock::now();restore.restore();stats.restore_sec=seconds(start);
        ++solves;
        return stats;
    }
};

}  // namespace pure_gbp
