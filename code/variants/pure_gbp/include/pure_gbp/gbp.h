#pragma once
#include "solver.h"
#include "eta_program.h"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <map>
#include <numeric>
#include <omp.h>

namespace pure_gbp {
using Mat=Eigen::MatrixXd;
using Vec=Eigen::VectorXd;
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point t) {return std::chrono::duration<double>(Clock::now()-t).count();}

struct Stats {
    int sweeps=0,full_sweeps=0,fixed_sweeps=0,failed_sweep=0,observed_threads=1;
    bool spd=true,converged=false,numerical_failure=false;
    double residual=1,precision_change=0,tail_ratio=0,full_sec=0,fixed_sec=0;
};
struct Edge {
    int from,to,reverse;
    Mat cross,lam,next_lam,eta_map;
    Vec eta,next_eta;
};
class SynchronousState;

// Exact block congruence: A = L N L^T, q = L^-1 b, x = L^-T z.
// Every off-diagonal block remains in N; the tree is only an update schedule.
class GBP {
public:
    Sparse raw,normalized;
    Eigen::SparseMatrix<double,Eigen::RowMajor> residual_rows;
    int n=0,blocks=0,threads=1;
    std::vector<int> offsets,sweep_order,tree_parent_edge;
    std::vector<Mat> lower,belief,belief_inverse,initial_lam;
    std::vector<Vec> rhs,belief_eta,initial_eta;
    std::vector<std::vector<int>> incoming,tree_children;
    std::vector<Edge> edges;
    struct ParallelTask {int kind,index;};
    struct ParallelBatch {int begin,end;bool serial;};
    std::vector<ParallelTask> parallel_tasks;
    std::vector<ParallelBatch> full_batches,eta_batches;
    std::shared_ptr<SynchronousState> synchronous_state;
    PackedEtaProgram eta_program;
    std::uint64_t schedule_generation=0;
    int parallel_levels=0,parallel_max_width=0;
    bool parallel_tree=false,native_serial=false,synchronous_packed=false;
    bool packed_eta_enabled=true,packed_ready=false,packed_precision_ready=false;
    bool stable_master=true,workspace_reuse_enabled=true,workspace_timing_enabled=false;
    bool frozen=false,warm_eta=false,tree_is_forest=false;
    int stable=0,check_period=5;
    double whitening_error=0,eta_relaxation=1;
    struct WorkspaceMetrics {
        double constructor_diag_sec=0,constructor_cross_sec=0;
        double constructor_whiten_edges_sec=0,constructor_sparse_sec=0,constructor_diagnostics_sec=0;
        // Prep includes entry-time packing; pack also includes mid-solve freezing.
        double solve_prep_sec=0,pack_sec=0,unpack_sec=0;
        std::size_t solve_calls=0,pack_calls=0,precision_packs=0,precision_reuses=0;
        std::size_t eta_unpacks=0,pack_buffer_growths=0,tree_scratch_skips=0;
        std::size_t cavity_cache_hits=0,cavity_cache_checks=0;
        std::size_t static_packs=0,static_reuses=0,belief_inverse_builds=0;
    } workspace_metrics;
    int packed_stride=0;
    std::vector<double> packed_map,packed_eta,packed_next,packed_belief,packed_inverse;
    void useParallelTreeSchedule(int full_min=4,int eta_min=16);
    Stats solveParallelTree(const Vec&,Vec&,int,double,double,double,double);
    Stats solveSerialTree(const Vec&,Vec&,int,double,double,double,double);
    Stats solveSynchronous(const Vec&,Vec&,int,double,double,double,double);
    void buildPackedEta();
    void packedCompute(int);
    void packedCommit(int);
    void packedMean(int,Vec&);
    void unpackEta();
    void invalidatePackedPrecision() {packed_precision_ready=false;packed_ready=false;}

    GBP(const Sparse& matrix, int block_size, int nt=1, bool diagnostics=true):raw(matrix),n(int(matrix.rows())),threads(nt) {
        auto phase_start=workspace_timing_enabled?Clock::now():Clock::time_point{};
        auto record_phase=[&](double& elapsed) {
            if(workspace_timing_enabled) {
                const auto now=Clock::now();
                elapsed=std::chrono::duration<double>(now-phase_start).count();phase_start=now;
            }
        };
        if(block_size<1) throw std::runtime_error("invalid block size");
        for(int i=0;i<n;i+=block_size) offsets.push_back(i);
        offsets.push_back(n); blocks=int(offsets.size())-1;
        lower.resize(blocks); belief.resize(blocks); belief_inverse.resize(blocks);
        rhs.resize(blocks); belief_eta.resize(blocks); incoming.resize(blocks);
        for(int i=0;i<blocks;++i) {
            int s=size(i), o=offsets[i];
            Mat diag=Mat(raw.block(o,o,s,s)); diag=0.5*(diag+diag.transpose()).eval();
            Eigen::LLT<Mat> llt(diag);
            if(llt.info()!=Eigen::Success) throw std::runtime_error("non-SPD coarse diagonal block");
            lower[i]=llt.matrixL(); belief[i]=Mat::Identity(s,s);
            belief_inverse[i]=Mat::Identity(s,s); rhs[i]=Vec::Zero(s); belief_eta[i]=rhs[i];
        }
        record_phase(workspace_metrics.constructor_diag_sec);
        std::map<std::pair<int,int>,Mat> cross;
        for(int k=0;k<raw.outerSize();++k) for(Sparse::InnerIterator it(raw,k);it;++it) {
            const int i=int(it.row())/block_size,j=int(it.col())/block_size;
            if(i>=j || it.value()==0) continue;
            auto key=std::make_pair(i,j);
            auto entry=cross.try_emplace(key);
            if(entry.second) entry.first->second=Mat::Zero(size(i),size(j));
            entry.first->second(int(it.row())-offsets[i],int(it.col())-offsets[j])=it.value();
        }
        record_phase(workspace_metrics.constructor_cross_sec);
        (void)diagnostics;
        std::vector<Eigen::Triplet<double>> ts;
        if(workspace_reuse_enabled) {
            edges.reserve(2*cross.size());
            std::size_t entries=std::size_t(n);
            for(const auto& kv:cross) entries+=2*std::size_t(kv.second.size());
            ts.reserve(entries);
        }
        for(int i=0;i<n;++i) ts.emplace_back(i,i,1.);
        for(auto& kv:cross) {
            int i=kv.first.first,j=kv.first.second;
            Mat t=lower[i].triangularView<Eigen::Lower>().solve(kv.second);
            Mat w=lower[j].triangularView<Eigen::Lower>().solve(t.transpose()).transpose();
            int p=int(edges.size());
            edges.push_back({i,j,p+1,w,Mat::Zero(size(j),size(j)),Mat::Zero(size(j),size(j)),Mat::Zero(size(j),size(i)),Vec::Zero(size(j)),Vec::Zero(size(j))});
            edges.push_back({j,i,p,w.transpose(),Mat::Zero(size(i),size(i)),Mat::Zero(size(i),size(i)),Mat::Zero(size(i),size(j)),Vec::Zero(size(i)),Vec::Zero(size(i))});
            incoming[j].push_back(p); incoming[i].push_back(p+1);
            for(int r=0;r<w.rows();++r) for(int c=0;c<w.cols();++c) {
                ts.emplace_back(offsets[i]+r,offsets[j]+c,w(r,c));
                ts.emplace_back(offsets[j]+c,offsets[i]+r,w(r,c));
            }
            whitening_error=std::max(whitening_error,(lower[i]*w*lower[j].transpose()-kv.second).norm()/std::max(1e-300,kv.second.norm()));
        }
        record_phase(workspace_metrics.constructor_whiten_edges_sec);
        normalized.resize(n,n); normalized.setFromTriplets(ts.begin(),ts.end());
        residual_rows=normalized;
        sweep_order.resize(edges.size());std::iota(sweep_order.begin(),sweep_order.end(),0);
        record_phase(workspace_metrics.constructor_sparse_sec);
        record_phase(workspace_metrics.constructor_diagnostics_sec);
    }
    void useTreeSchedule() {
        // A schedule, NOT graph pruning: every directed edge is updated once per sweep.
        // Non-tree edges first, then collect/distribute along a maximum spanning forest.
        std::vector<int> parent(blocks),rank(blocks),ids;
        std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int i) {while(parent[i]!=i) {parent[i]=parent[parent[i]];i=parent[i];}return i;};
        for(int p=0;p<int(edges.size());p+=2) ids.push_back(p);
        std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return edges[a].cross.squaredNorm()>edges[b].cross.squaredNorm();});
        std::vector<std::vector<int>> tree(blocks);
        sweep_order.clear();
        for(int p:ids) {
            int a=root(edges[p].from),b=root(edges[p].to);
            if(a==b) {sweep_order.push_back(p);sweep_order.push_back(edges[p].reverse);continue;}
            if(rank[a]<rank[b]) std::swap(a,b);
            parent[b]=a;if(rank[a]==rank[b]) ++rank[a];
            tree[edges[p].from].push_back(p);tree[edges[p].to].push_back(edges[p].reverse);
        }
        std::vector<char> seen(blocks,0);std::vector<int> forward;
        tree_is_forest=sweep_order.empty();
        for(int start=0;start<blocks;++start) if(!seen[start]) {
            std::vector<int> stack{start};seen[start]=1;
            while(!stack.empty()) {
                int u=stack.back();stack.pop_back();
                for(int p:tree[u]) if(!seen[edges[p].to]) {
                    seen[edges[p].to]=1;forward.push_back(p);stack.push_back(edges[p].to);
                }
            }
        }
        for(auto it=forward.rbegin();it!=forward.rend();++it) sweep_order.push_back(edges[*it].reverse);
        sweep_order.insert(sweep_order.end(),forward.begin(),forward.end());
        if(sweep_order.size()!=edges.size()) throw std::runtime_error("tree schedule omitted an edge");
        check_period=1;
    }
    void projectedInitialization(const PairDiagonals& pairs,bool reuse_diagonal=false) {
        const bool rebuild=!reuse_diagonal || initial_lam.size()!=edges.size();
        initial_lam.resize(edges.size());initial_eta.resize(edges.size());
        for(int p=0;p<int(edges.size());++p) {
            const auto& e=edges[p];
            const bool forward=e.from<e.to;
            const auto& pair=pairs.at({std::min(e.from,e.to),std::max(e.from,e.to)});
            const Mat& d=forward?pair.second:pair.first;
            const Vec& h=forward?pair.eta_second:pair.eta_first;
            if(rebuild) {
                Mat t=lower[e.to].triangularView<Eigen::Lower>().solve(d);
                initial_lam[p]=-lower[e.to].triangularView<Eigen::Lower>().solve(t.transpose()).transpose();
            }
            initial_eta[p]=-lower[e.to].triangularView<Eigen::Lower>().solve(h);
        }
    }
    int size(int i) const {return offsets[i+1]-offsets[i];}
    Vec whitenRhs(const Vec& b) const {
        Vec q(n);
        for(int i=0;i<blocks;++i) q.segment(offsets[i],size(i))=lower[i].triangularView<Eigen::Lower>().solve(b.segment(offsets[i],size(i)));
        return q;
    }
    Vec recover(const Vec& z) const {
        Vec x(n);
        for(int i=0;i<blocks;++i) x.segment(offsets[i],size(i))=lower[i].transpose().triangularView<Eigen::Upper>().solve(z.segment(offsets[i],size(i)));
        return x;
    }
    void reset(double shift=0) {
        frozen=false; stable=0;invalidatePackedPrecision();
        for(int p=0;p<int(edges.size());++p) {auto& e=edges[p];e.lam.setZero();if(!initial_lam.empty()) e.lam=initial_lam[p];e.eta.setZero();}
        for(int i=0;i<blocks;++i) {
            belief[i]=(1.+shift)*Mat::Identity(size(i),size(i));
            for(int p:incoming[i]) belief[i]+=edges[p].lam;
            belief_eta[i].setZero();
        }
    }
    Stats solve(const Vec& q, Vec& z, int max_sweeps, double damping, double tolerance,
                double shift=0, bool reuse_precision=false, double precision_tol=1e-12) {
        const auto prep_start=workspace_timing_enabled?Clock::now():Clock::time_point{};
        ++workspace_metrics.solve_calls;
        auto record_prep=[&]() {if(workspace_timing_enabled)workspace_metrics.solve_prep_sec+=seconds(prep_start);};
        if(!reuse_precision) reset(shift);
        // Full precision sweeps may overwrite both caches. RHS/origin changes do not.
        if(!frozen) packed_precision_ready=false;
        if(workspace_reuse_enabled) packed_ready=false;
        if((!reuse_precision || !warm_eta) && (!workspace_reuse_enabled || reuse_precision))
            for(auto& e:edges) e.eta.setZero();
        for(int i=0;i<blocks;++i) rhs[i]=belief_eta[i]=q.segment(offsets[i],size(i));
        if(reuse_precision && warm_eta) for(auto& e:edges) belief_eta[e.to]+=e.eta;
        if(!initial_lam.empty() && (!reuse_precision || !warm_eta)) {
            // Native zero-message initialization is (-F_jj, -h_fj) in this gauge.
            // The diagonal-budget eta is only a legacy control for precision-only captures.
            for(int p=0;p<int(edges.size());++p) {auto& e=edges[p];e.eta=initial_eta.empty()?Vec(e.lam*rhs[e.to]):initial_eta[p];belief_eta[e.to]+=e.eta;}
        }
        z=Vec::Zero(n); Stats stats;
        if(q.norm()==0) {stats.converged=true;stats.residual=0;record_prep();return stats;}
        if(!synchronous_packed && parallel_tree && frozen && packed_eta_enabled) buildPackedEta();
        record_prep();
        if(synchronous_packed) return solveSynchronous(q,z,max_sweeps,damping,tolerance,shift,precision_tol);
        if(native_serial) return solveSerialTree(q,z,max_sweeps,damping,tolerance,shift,precision_tol);
        return solveParallelTree(q,z,max_sweeps,damping,tolerance,shift,precision_tol);
    }
};
}
#include "raw_block.h"
#include "ordered.h"
#include "serial.h"
#include "packed_eta.h"
#include "synchronous.h"
