#pragma once
#include <cstring>

namespace pure_gbp {

// All proposals read one immutable sweep version. Destination-owned gather
// updates beliefs; only the collective boundary swaps current/next messages.
class SynchronousState {
    using Buffer=std::vector<double,Eigen::aligned_allocator<double>>;
    struct Link {int from,to,reverse;};
    const GBP* owner=nullptr;
    int stride=0,matrix_stride=0,uniform=0;
    std::vector<Link> links;
    std::vector<int> dims,offsets,incoming_offsets,incoming_slots;
    Buffer cross,lam,next_lam,map,last_lower,eta,next_eta,belief,belief_eta,inverse,rhs;
    bool forward_only=false;
    std::vector<int> failed_edge,failed_node;
    std::vector<unsigned char> inverse_valid;
    std::vector<std::size_t> inverse_builds;
    std::vector<double> change,residual_squared;

    template<int D> using Matrix=Eigen::Matrix<double,D,D>;
    template<int D> using Vector=Eigen::Matrix<double,D,1>;
    template<int D> auto m(Buffer& a,int index,int rows,int cols) {
        return Eigen::Map<Matrix<D>,Eigen::Unaligned>(a.data()+std::size_t(index)*matrix_stride,rows,cols);
    }
    template<int D> auto v(Buffer& a,int index,int rows) {
        return Eigen::Map<Vector<D>,Eigen::Unaligned>(a.data()+std::size_t(index)*stride,rows);
    }
    template<int D> int propose(int p,double damping,double eta_damping,bool fixed,bool raw) {
        const auto e=links[p];const int a=dims[e.from],b=dims[e.to];
        auto transform=m<D>(map,p,b,a);
        Vector<D> cavity_eta=v<D>(belief_eta,e.from,a)-v<D>(eta,e.reverse,a);
        if(!fixed) {
            Matrix<D> cavity=m<D>(belief,e.from,a,a)-m<D>(lam,e.reverse,a,a);
            Matrix<D> solved=m<D>(cross,p,a,b);
            if constexpr(D==4 || D==6 || D==12) {
                if(raw) {
                    if(raw_block::choleskyLowerInPlace<D>(cavity.data()))return 1;
                    if(forward_only) {
                        cavity.template triangularView<Eigen::Lower>().solveInPlace(solved);
                        cavity.template triangularView<Eigen::Lower>().solveInPlace(cavity_eta);
                        m<D>(last_lower,p,a,a)=cavity;
                    } else raw_block::choleskySolveInPlace<D,D>(cavity.data(),solved.data());
                } else {
                    Eigen::LLT<Matrix<D>> llt(cavity);
                    if(llt.info()!=Eigen::Success)return 1;
                    if(forward_only) {
                        llt.matrixL().solveInPlace(solved);llt.matrixL().solveInPlace(cavity_eta);
                        m<D>(last_lower,p,a,a)=llt.matrixL();
                    } else llt.solveInPlace(solved);
                }
            } else {
                if(a==1 && !(cavity(0,0)>1e-14))return 1;
                Eigen::LLT<Matrix<D>> llt(cavity);
                if(llt.info()!=Eigen::Success)return 1;
                if(forward_only) {
                    llt.matrixL().solveInPlace(solved);llt.matrixL().solveInPlace(cavity_eta);
                    m<D>(last_lower,p,a,a)=llt.matrixL();
                } else llt.solveInPlace(solved);
            }
            transform=-solved.transpose();
            Matrix<D> proposal;
            if(forward_only)proposal=(1.-damping)*m<D>(lam,p,b,b)+damping*(transform*solved);
            else proposal=(1.-damping)*m<D>(lam,p,b,b)+damping*(transform*m<D>(cross,p,a,b));
            proposal=.5*(proposal+proposal.transpose()).eval();
            change[p]=(proposal-m<D>(lam,p,b,b)).norm()/std::max(1.,proposal.norm());
            m<D>(next_lam,p,b,b)=proposal;
        }
        const Vector<D> proposal=(1.-eta_damping)*v<D>(eta,p,b)+eta_damping*(transform*cavity_eta);
        v<D>(next_eta,p,b)=proposal;
        return proposal.allFinite()?0:1;
    }
    template<int D> void materializeMap(int p) {
        const int a=dims[links[p].from],b=dims[links[p].to];
        Matrix<D> solved=-m<D>(map,p,b,a).transpose();
        m<D>(last_lower,p,a,a).transpose().template triangularView<Eigen::Upper>().solveInPlace(solved);
        m<D>(map,p,b,a)=-solved.transpose();
    }
    template<int D> int gather(int i,Vec& z,bool fixed,bool mean,double shift) {
        const int a=dims[i];auto h=v<D>(belief_eta,i,a);
        h=v<D>(rhs,i,a);
        auto b=m<D>(belief,i,a,a);
        if(!fixed) {b.setIdentity();b*=1.+shift;}
        for(int k=incoming_offsets[i];k<incoming_offsets[i+1];++k) {
            const int p=incoming_slots[k];
            if(failed_edge[p])return 1;
            h+=v<D>(next_eta,p,a);
            if(!fixed)b+=m<D>(next_lam,p,a,a);
        }
        if(!h.allFinite())return 1;
        if(!fixed || (mean && !inverse_valid[i])) {
            Eigen::LLT<Matrix<D>> llt(b);
            if(llt.info()!=Eigen::Success)return 1;
            inverse_valid[i]=0;
            if(mean) {
                auto inv=m<D>(inverse,i,a,a);inv.setIdentity();llt.solveInPlace(inv);
                inverse_valid[i]=1;++inverse_builds[i];
            }
        }
        if(mean) {
            Eigen::Map<Vector<D>> out(z.data()+offsets[i],a);
            out.noalias()=m<D>(inverse,i,a,a)*h;
            if(!out.allFinite())return 1;
        }
        return 0;
    }
    void import(GBP& w) {
        forward_only=true;
        const bool reuse=owner==&w && w.frozen && w.packed_precision_ready
            && links.size()==w.edges.size() && offsets==w.offsets;
        if(!reuse) {
            stride=0;for(int i=0;i<w.blocks;++i)stride=std::max(stride,w.size(i));
            matrix_stride=stride*stride;uniform=stride;
            dims.resize(w.blocks);offsets=w.offsets;
            incoming_offsets.assign(w.blocks+1,0);incoming_slots.clear();
            for(int i=0;i<w.blocks;++i) {
                dims[i]=w.size(i);if(dims[i]!=stride)uniform=0;
                incoming_slots.insert(incoming_slots.end(),w.incoming[i].begin(),w.incoming[i].end());
                incoming_offsets[i+1]=int(incoming_slots.size());
            }
            const auto es=w.edges.size(),ns=std::size_t(w.blocks);
            cross.resize(es*matrix_stride);lam.resize(es*matrix_stride);next_lam.resize(es*matrix_stride);map.resize(es*matrix_stride);
            if(forward_only)last_lower.resize(es*matrix_stride);
            eta.resize(es*stride);next_eta.resize(es*stride);
            belief.resize(ns*matrix_stride);inverse.resize(ns*matrix_stride);
            belief_eta.resize(ns*stride);rhs.resize(ns*stride);
            links.resize(es);failed_edge.resize(es);failed_node.resize(ns);change.resize(es);residual_squared.resize(w.n);
            inverse_valid.resize(ns);inverse_builds.resize(ns);
            for(int p=0;p<int(es);++p) {
                const auto& e=w.edges[p];links[p]={e.from,e.to,e.reverse};
                m<Eigen::Dynamic>(cross,p,dims[e.from],dims[e.to])=e.cross;
            }
            ++w.workspace_metrics.static_packs;
        } else ++w.workspace_metrics.static_reuses;
        if(forward_only)last_lower.resize(links.size()*matrix_stride);
        if(!reuse) {
            for(int p=0;p<int(links.size());++p) {
                const auto& e=w.edges[p];
                m<Eigen::Dynamic>(lam,p,dims[e.to],dims[e.to])=e.lam;
                m<Eigen::Dynamic>(map,p,dims[e.to],dims[e.from])=e.eta_map;
            }
            for(int i=0;i<w.blocks;++i) {
                m<Eigen::Dynamic>(belief,i,dims[i],dims[i])=w.belief[i];
                m<Eigen::Dynamic>(inverse,i,dims[i],dims[i])=w.belief_inverse[i];
                inverse_valid[i]=1;
            }
            ++w.workspace_metrics.precision_packs;
        } else ++w.workspace_metrics.precision_reuses;
        for(int p=0;p<int(links.size());++p)v<Eigen::Dynamic>(eta,p,dims[links[p].to])=w.edges[p].eta;
        for(int i=0;i<w.blocks;++i) {
            v<Eigen::Dynamic>(rhs,i,dims[i])=w.rhs[i];
            v<Eigen::Dynamic>(belief_eta,i,dims[i])=w.belief_eta[i];
        }
        std::fill(failed_edge.begin(),failed_edge.end(),0);std::fill(failed_node.begin(),failed_node.end(),0);
        std::fill(inverse_builds.begin(),inverse_builds.end(),0);
        owner=&w;
    }
    void exportTo(GBP& w,bool precision) {
        for(int p=0;p<int(links.size());++p) {
            auto& e=w.edges[p];const int a=dims[e.from],b=dims[e.to];
            e.eta=v<Eigen::Dynamic>(eta,p,b);
            if(precision) {
                e.lam=m<Eigen::Dynamic>(lam,p,b,b);
                e.eta_map=m<Eigen::Dynamic>(map,p,b,a);
            }
        }
        for(int i=0;i<w.blocks;++i) {
            w.belief_eta[i]=v<Eigen::Dynamic>(belief_eta,i,dims[i]);
            if(precision) {
                w.belief[i]=m<Eigen::Dynamic>(belief,i,dims[i],dims[i]);
                w.belief_inverse[i]=m<Eigen::Dynamic>(inverse,i,dims[i],dims[i]);
            }
        }
        w.packed_precision_ready=w.frozen;w.packed_ready=false;
    }
    template<int D> Stats execute(GBP& w,const Vec& q,Vec& z,int budget,double damping,
                                  double tolerance,double shift,double precision_tol) {
        Stats stats;bool done=false;const auto start=Clock::now();
        const bool started_fixed=w.frozen;bool timed_freeze=started_fixed;
        constexpr bool raw=true;
        constexpr double eta_override=0.;
        constexpr int period=8;
        const double denominator=std::max(1e-300,q.norm());int previous_check=0;
        const auto residualRow=[&](int row) {
            double r=q[row]-shift*z[row];
            for(Eigen::SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(w.residual_rows,row);it;++it)r-=it.value()*z[it.col()];
            residual_squared[row]=r*r;
        };
        const auto advance=[&](int sweep,bool fixed,bool check) {
            stats.sweeps=sweep;
            const int bad_edge=std::accumulate(failed_edge.begin(),failed_edge.end(),0);
            const int bad=bad_edge+std::accumulate(failed_node.begin(),failed_node.end(),0);
            if(!bad_edge) {
                if(fixed)++stats.fixed_sweeps;else ++stats.full_sweeps;
                eta.swap(next_eta);if(!fixed)lam.swap(next_lam);
            }
            if(bad) {stats.spd=false;stats.failed_sweep=sweep;done=true;}
            if(!done) {
                if(!fixed) {
                    stats.precision_change=change.empty()?0.:*std::max_element(change.begin(),change.end());
                    if(stats.precision_change<precision_tol)++w.stable;else w.stable=0;
                    // A synchronous tree sweep is not a full tree traversal.
                    if(w.stable>=3)w.frozen=true;
                    if(w.frozen && !timed_freeze && !forward_only) {stats.full_sec=seconds(start);timed_freeze=true;}
                }
                if(check) {
                    const double previous=stats.residual;
                    stats.residual=std::sqrt(std::accumulate(residual_squared.begin(),residual_squared.end(),0.))/denominator;
                    stats.tail_ratio=std::pow(stats.residual/std::max(1e-300,previous),1./(sweep-previous_check));previous_check=sweep;
                    if(stats.residual<=tolerance){stats.converged=true;done=true;}
                    if(!std::isfinite(stats.residual) || stats.residual>1e12){stats.numerical_failure=true;done=true;}
                }
            }
        };
#pragma omp parallel num_threads(w.threads) if(w.threads>1)
        {
#pragma omp single
            stats.observed_threads=omp_get_num_threads();
            for(int sweep=1;sweep<=budget;++sweep) {
                const bool fixed=w.frozen,check=sweep%period==0 || sweep==budget;
                const double omega=eta_override>0?eta_override:(fixed?w.eta_relaxation:damping);
#pragma omp for schedule(static)
                for(int p=0;p<int(links.size());++p)failed_edge[p]=propose<D>(p,damping,omega,fixed,raw);
#pragma omp for schedule(static)
                for(int i=0;i<w.blocks;++i)failed_node[i]=gather<D>(i,z,fixed,
                    check || !fixed,shift);
                if(check) {
#pragma omp for schedule(static)
                    for(int row=0;row<w.n;++row)residualRow(row);
                }
#pragma omp single
                {
                    advance(sweep,fixed,check);
                }
                // Convert the last actual cavity's L/Y once, not from a later
                // belief. This preserves the fixed-eta map and public state.
                if(forward_only && !fixed && (w.frozen || done || sweep==budget)) {
#pragma omp for schedule(static)
                    for(int p=0;p<int(links.size());++p)if(!failed_edge[p])materializeMap<D>(p);
#pragma omp single
                    {
                        if(w.frozen && !timed_freeze) {stats.full_sec=seconds(start);timed_freeze=true;}
                    }
                }
                if(done)break;
            }
        }
        // Successful exits have a current inverse (last/check/freeze sweep).
        // Also keep the public state valid on an early numerical-failure exit.
        for(int i=0;i<w.blocks;++i)if(!inverse_valid[i]) {
            const int a=dims[i];Eigen::LLT<Matrix<D>> llt(m<D>(belief,i,a,a));
            if(llt.info()!=Eigen::Success){stats.spd=false;continue;}
            auto inv=m<D>(inverse,i,a,a);inv.setIdentity();llt.solveInPlace(inv);
            inverse_valid[i]=1;++inverse_builds[i];
        }
        w.workspace_metrics.belief_inverse_builds+=std::accumulate(inverse_builds.begin(),inverse_builds.end(),std::size_t(0));
        exportTo(w,!started_fixed);
        if(!stats.spd)w.invalidatePackedPrecision();
        stats.residual=(q-w.normalized*z-shift*z).norm()/denominator;
        stats.numerical_failure=stats.numerical_failure || !std::isfinite(stats.residual) || stats.residual>1e12;
        stats.converged=stats.spd && !stats.numerical_failure && stats.residual<=tolerance;
        const double elapsed=seconds(start);
        if(timed_freeze)stats.fixed_sec=elapsed-stats.full_sec;else stats.full_sec=elapsed;
        return stats;
    }
public:
    Stats solve(GBP& w,const Vec& q,Vec& z,int budget,double damping,double tolerance,double shift,double precision_tol) {
        import(w);
        switch(uniform) {
        case 4:return execute<4>(w,q,z,budget,damping,tolerance,shift,precision_tol);
        case 6:return execute<6>(w,q,z,budget,damping,tolerance,shift,precision_tol);
        case 12:return execute<12>(w,q,z,budget,damping,tolerance,shift,precision_tol);
        default:return execute<Eigen::Dynamic>(w,q,z,budget,damping,tolerance,shift,precision_tol);
        }
    }
};

inline Stats GBP::solveSynchronous(const Vec& q,Vec& z,int max_sweeps,double damping,double tolerance,double shift,double precision_tol) {
    if(!synchronous_state || !synchronous_state.unique())synchronous_state=std::make_shared<SynchronousState>();
    return synchronous_state->solve(*this,q,z,max_sweeps,damping,tolerance,shift,precision_tol);
}
}
