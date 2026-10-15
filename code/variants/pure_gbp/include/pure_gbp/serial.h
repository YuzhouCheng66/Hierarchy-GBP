#pragma once

namespace pure_gbp {

inline Stats GBP::solveSerialTree(const Vec& q,Vec& z,int max_sweeps,double damping,
                                  double tolerance,double shift,double precision_tol) {
    Stats stats;
    const auto executor_start=Clock::now();
    bool timed_freeze=frozen;
    constexpr int period=8;
    int previous_check=0;
    const double denominator=std::max(1e-300,q.norm());
    std::vector<double> changes(edges.size(),0.),residual_squared(n,0.);
    std::vector<int> failed_edge(edges.size(),0),failed_node(blocks,0);
    Mat cavity,solved;Vec cavity_eta;Eigen::LLT<Mat> llt;
    for(int sweep=1;sweep<=max_sweeps;++sweep) {
        const bool fixed=frozen;
        const bool check=sweep%period==0 || sweep==max_sweeps;
        constexpr bool need_mean=true;
        auto compute=[&](int p) {
            failed_edge[p]=0;
            if(fixed && packed_ready) {packedCompute(p);return;}
            auto& e=edges[p];const auto& reverse=edges[e.reverse];
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
            const double omega=fixed?eta_relaxation:damping;
            e.next_eta=(1.-omega)*e.eta+omega*(e.eta_map*cavity_eta);
            if(!e.next_eta.allFinite()) failed_edge[p]=1;
        };
        auto commit=[&](int p) {
            if(fixed && packed_ready) {packedCommit(p);return;}
            if(failed_edge[p])return;
            auto& e=edges[p];
            if(!fixed) {belief[e.to]+=e.next_lam-e.lam;e.lam.swap(e.next_lam);}
            belief_eta[e.to]+=e.next_eta-e.eta;e.eta.swap(e.next_eta);
        };
        const bool programmed=fixed && eta_program.prepare(*this,schedule_generation)
            && eta_program.sweep(*this,schedule_generation);
        if(programmed) {
            std::fill(failed_edge.begin(),failed_edge.end(),0);
        } else for(const auto t:parallel_tasks) {
            if(t.kind==0) {compute(t.index);commit(t.index);continue;}
            const int i=t.index,p=tree_parent_edge[i];
            if(t.kind==1) {
                for(auto it=tree_children[i].rbegin();it!=tree_children[i].rend();++it)commit(edges[*it].reverse);
                if(p>=0)compute(edges[p].reverse);
            } else {
                if(p>=0)commit(p);
                for(int child:tree_children[i])compute(child);
            }
        }
        if(!fixed || check) for(int i=0;i<blocks;++i) {
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
        stats.sweeps=sweep;
        if(fixed)++stats.fixed_sweeps;else ++stats.full_sweeps;
        if(std::accumulate(failed_edge.begin(),failed_edge.end(),0)+std::accumulate(failed_node.begin(),failed_node.end(),0) || !z.allFinite()) {
            stats.spd=false;stats.failed_sweep=sweep;break;
        }
        if(!fixed) {
            stats.precision_change=changes.empty()?0:*std::max_element(changes.begin(),changes.end());
            if(stats.precision_change<precision_tol)++stable;else stable=0;
            if(stable>=3 || (tree_is_forest && damping==1.))frozen=true;
            if(frozen && packed_eta_enabled && !packed_ready)buildPackedEta();
            if(frozen && !timed_freeze) {stats.full_sec=seconds(executor_start);timed_freeze=true;}
        }
        if(check) {
            const double previous=stats.residual;
            for(int row=0;row<n;++row) {
                double value=q[row]-shift*z[row];
                for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(residual_rows,row);it;++it)value-=it.value()*z[it.col()];
                residual_squared[row]=value*value;
            }
            stats.residual=std::sqrt(std::accumulate(residual_squared.begin(),residual_squared.end(),0.))/denominator;
            stats.tail_ratio=std::pow(stats.residual/std::max(1e-300,previous),1./(sweep-previous_check));previous_check=sweep;
            if(stats.residual<=tolerance) {stats.converged=true;break;}
            if(!std::isfinite(stats.residual) || stats.residual>1e12) {stats.numerical_failure=true;break;}
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
