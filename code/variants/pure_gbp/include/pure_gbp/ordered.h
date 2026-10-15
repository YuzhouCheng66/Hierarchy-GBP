#pragma once

namespace pure_gbp {

inline void GBP::useParallelTreeSchedule(int full_min,int eta_min) {
    ++schedule_generation;eta_program.invalidate();
    useTreeSchedule();
    parallel_tree=true;
    tree_parent_edge.assign(blocks,-1);tree_children.assign(blocks,{});
    // Reconstruct the identical maximum spanning forest from the original order.
    std::vector<int> parent(blocks),rank(blocks),ids;
    std::iota(parent.begin(),parent.end(),0);
    auto root=[&](int i) {while(parent[i]!=i) {parent[i]=parent[parent[i]];i=parent[i];}return i;};
    for(int p=0;p<int(edges.size());p+=2) ids.push_back(p);
    std::stable_sort(ids.begin(),ids.end(),[&](int a,int b){return edges[a].cross.squaredNorm()>edges[b].cross.squaredNorm();});
    int off_tree=0;
    for(int p:ids) {
        int a=root(edges[p].from),b=root(edges[p].to);
        if(a==b) {off_tree+=2;continue;}
        if(rank[a]<rank[b]) std::swap(a,b);
        parent[b]=a;if(rank[a]==rank[b]) ++rank[a];
    }
    const int tree_edges=(int(edges.size())-off_tree)/2;
    for(int k=off_tree+tree_edges;k<int(sweep_order.size());++k) {
        int p=sweep_order[k];tree_parent_edge[edges[p].to]=p;
        tree_children[edges[p].from].push_back(p);
    }
    std::vector<std::vector<ParallelTask>> levels;
    std::vector<int> last_write(blocks,-1),last_read(blocks,-1);
    // Off-tree messages keep the serial read/write order. Shared reads are safe;
    // every write-after-read/write and read-after-write dependency gets a barrier.
    for(int k=0;k<off_tree;++k) {
        int p=sweep_order[k],i=edges[p].from,j=edges[p].to;
        int level=1+std::max(last_write[i],std::max(last_write[j],last_read[j]));
        if(int(levels.size())<=level) levels.resize(level+1);
        levels[level].push_back({0,p});
        last_read[i]=std::max(last_read[i],level);last_write[j]=level;
    }
    std::vector<int> height(blocks,0),depth(blocks,0);
    for(int k=off_tree+tree_edges;k<int(sweep_order.size());++k) {
        const auto& e=edges[sweep_order[k]];depth[e.to]=depth[e.from]+1;
    }
    for(int k=int(sweep_order.size())-1;k>=off_tree+tree_edges;--k) {
        const auto& e=edges[sweep_order[k]];height[e.from]=std::max(height[e.from],height[e.to]+1);
    }
    int base=int(levels.size());levels.resize(base+1+*std::max_element(height.begin(),height.end()));
    for(int i=0;i<blocks;++i) levels[base+height[i]].push_back({1,i});
    base=int(levels.size());levels.resize(base+1+*std::max_element(depth.begin(),depth.end()));
    for(int i=0;i<blocks;++i) levels[base+depth[i]].push_back({2,i});
    parallel_tasks.clear();full_batches.clear();eta_batches.clear();
    parallel_levels=int(levels.size());parallel_max_width=0;
    auto add_batch=[](std::vector<ParallelBatch>& batches,int begin,int end,bool serial) {
        if(serial && !batches.empty() && batches.back().serial) batches.back().end=end;
        else batches.push_back({begin,end,serial});
    };
    for(int level_id=0;level_id<int(levels.size());++level_id) {
        const auto& level=levels[level_id];
        if(level.empty()) continue;
        int begin=int(parallel_tasks.size());parallel_tasks.insert(parallel_tasks.end(),level.begin(),level.end());
        int end=int(parallel_tasks.size());parallel_max_width=std::max(parallel_max_width,end-begin);
        add_batch(full_batches,begin,end,end-begin<full_min);
        add_batch(eta_batches,begin,end,end-begin<eta_min);
    }
}

inline Stats GBP::solveParallelTree(const Vec& q,Vec& z,int max_sweeps,double damping,
                                    double tolerance,double shift,double precision_tol) {
    Stats stats;bool done=false;
    const auto executor_start=Clock::now();
    const bool started_fixed=frozen;
    bool timed_freeze=started_fixed;
    constexpr int period=8;
    int previous_check=0;
    bool eta_program_ready=frozen && eta_program.prepare(*this,schedule_generation);
    const int executor_threads=threads;
    const double denominator=std::max(1e-300,q.norm());
    std::vector<double> changes(edges.size(),0.);
    std::vector<int> failed_edge(edges.size(),0),failed_node(blocks,0);
    std::vector<double> residual_squared(n,0.);
#pragma omp parallel num_threads(executor_threads) if(executor_threads>1)
    {
        Mat cavity,solved;Vec cavity_eta;
        Eigen::LLT<Mat> llt;
#pragma omp single
        stats.observed_threads=omp_get_num_threads();
        for(int sweep=1;sweep<=max_sweeps;++sweep) {
            const bool fixed=frozen;
            const bool check=sweep%period==0 || sweep==max_sweeps;
            // Precision freezing requires three successive checks. If it can
            // occur now, prepare inverses before the frozen maps are packed.
            constexpr bool need_mean=true;
            auto compute=[&](int p) {
                if(fixed && packed_ready) {failed_edge[p]=0;packedCompute(p);return;}
                auto& e=edges[p];const auto& reverse=edges[e.reverse];
                failed_edge[p]=0;
                if(!fixed && size(e.from)==size(e.to)) {
                    switch(size(e.from)) {
                    case 4:failed_edge[p]=rawBlockFullCompute<4>(*this,p,damping,changes[p]);return;
                    case 6:failed_edge[p]=rawBlockFullCompute<6>(*this,p,damping,changes[p]);return;
                    case 12:failed_edge[p]=rawBlockFullCompute<12>(*this,p,damping,changes[p]);return;
                    }
                }
                if(!fixed) {
                    cavity=belief[e.from]-reverse.lam;llt.compute(cavity);
                    if(llt.info()!=Eigen::Success) {failed_edge[p]=1;return;}
                    solved=e.cross;llt.solveInPlace(solved);e.eta_map=-solved.transpose();
                    e.next_lam=(1.-damping)*e.lam+damping*(e.eta_map*e.cross);
                    e.next_lam=.5*(e.next_lam+e.next_lam.transpose()).eval();
                    changes[p]=(e.next_lam-e.lam).norm()/std::max(1.,e.next_lam.norm());
                }
                cavity_eta=belief_eta[e.from]-reverse.eta;
                double omega=fixed?eta_relaxation:damping;
                e.next_eta=(1.-omega)*e.eta+omega*(e.eta_map*cavity_eta);
                if(!e.next_eta.allFinite()) failed_edge[p]=1;
            };
            auto commit=[&](int p) {
                if(fixed && packed_ready) {packedCommit(p);return;}
                if(failed_edge[p]) return;
                auto& e=edges[p];
                if(!fixed) {belief[e.to]+=e.next_lam-e.lam;e.lam.swap(e.next_lam);}
                belief_eta[e.to]+=e.next_eta-e.eta;e.eta.swap(e.next_eta);
            };
            auto task=[&](int k) {
                if(fixed && eta_program_ready) {eta_program.task(*this,k);return;}
                const auto t=parallel_tasks[k];
                if(t.kind==0) {compute(t.index);commit(t.index);return;}
                int i=t.index,p=tree_parent_edge[i];
                if(t.kind==1) {
                    // A single node owner commits siblings in the original serial order.
                    for(auto it=tree_children[i].rbegin();it!=tree_children[i].rend();++it) commit(edges[*it].reverse);
                    if(p>=0) compute(edges[p].reverse);
                } else {
                    if(p>=0) commit(p);
                    for(int child:tree_children[i]) compute(child);
                }
            };
            const auto& batches=fixed?eta_batches:full_batches;
            for(const auto& batch:batches) {
                if(batch.serial || threads==1) {
#pragma omp master
                    for(int k=batch.begin;k<batch.end;++k)task(k);
#pragma omp barrier
                } else {
#pragma omp for schedule(static)
                    for(int k=batch.begin;k<batch.end;++k)task(k);
                }
            }
            if(!fixed || check) {
#pragma omp for schedule(static)
            for(int i=0;i<blocks;++i) {
                failed_node[i]=0;
                if(fixed && packed_ready) {packedMean(i,z);continue;}
                if(!fixed) {
                    llt.compute(belief[i]);
                    if(llt.info()!=Eigen::Success) {failed_node[i]=1;continue;}
                    if(!need_mean) {if(!belief_eta[i].allFinite())failed_node[i]=1;continue;}
                    belief_inverse[i].setIdentity();llt.solveInPlace(belief_inverse[i]);
                }
                z.segment(offsets[i],size(i)).noalias()=belief_inverse[i]*belief_eta[i];
            }
            }
            if(packed_eta_enabled && check) {
#pragma omp for schedule(static)
                for(int row=0;row<n;++row) {
                    double value=q[row]-shift*z[row];
                    for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(residual_rows,row);it;++it)value-=it.value()*z[it.col()];
                    residual_squared[row]=value*value;
                }
            }
#pragma omp single
            {
                stats.sweeps=sweep;
                if(fixed) ++stats.fixed_sweeps;else ++stats.full_sweeps;
                int bad=std::accumulate(failed_edge.begin(),failed_edge.end(),0)+std::accumulate(failed_node.begin(),failed_node.end(),0);
                if(bad || !z.allFinite()) {stats.spd=false;stats.failed_sweep=sweep;done=true;}
                if(!fixed && !done) {
                    double change=changes.empty()?0:*std::max_element(changes.begin(),changes.end());
                    stats.precision_change=change;
                    if(change<precision_tol) ++stable;else stable=0;
                    if(stable>=3 || (tree_is_forest && damping==1.)) frozen=true;
                    if(frozen && packed_eta_enabled && !packed_ready) buildPackedEta();
                    if(frozen)eta_program_ready=eta_program.prepare(*this,schedule_generation);
                    if(frozen && !timed_freeze) {stats.full_sec=seconds(executor_start);timed_freeze=true;}
                }
                if(!done && check) {
                    double previous=stats.residual;
                    stats.residual=packed_eta_enabled?std::sqrt(std::accumulate(residual_squared.begin(),residual_squared.end(),0.))/denominator:(q-normalized*z-shift*z).norm()/denominator;
                    stats.tail_ratio=std::pow(stats.residual/std::max(1e-300,previous),1./(sweep-previous_check));previous_check=sweep;
                    if(stats.residual<=tolerance) {stats.converged=true;done=true;}
                    if(!std::isfinite(stats.residual) || stats.residual>1e12) {stats.numerical_failure=true;done=true;}
                }
            }
            if(done) break;
        }
    }
    if(packed_ready)unpackEta();
    stats.residual=(q-normalized*z-shift*z).norm()/denominator;
    stats.numerical_failure=stats.numerical_failure || !std::isfinite(stats.residual) || stats.residual>1e12;
    stats.converged=stats.spd && !stats.numerical_failure && stats.residual<=tolerance;
    const double elapsed=seconds(executor_start);
    if(timed_freeze)stats.fixed_sec=elapsed-stats.full_sec;else stats.full_sec=elapsed;
    return stats;
}
}
