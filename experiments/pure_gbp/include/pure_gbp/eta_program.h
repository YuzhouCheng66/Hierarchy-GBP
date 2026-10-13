#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace pure_gbp {

// Numerical message pass only. The caller owns serial execution, cache epochs,
// failure bookkeeping, means, residual checks, stopping, and final unpacking.
class PackedEtaProgram {
public:
    enum class Kind : unsigned char { Compute, Commit, ComputeCommit };
    struct Operation {
        Kind kind;
        std::size_t map_offset, eta_offset, reverse_offset;
        std::size_t source_offset, destination_offset;
    };

    PackedEtaProgram()=default;
    PackedEtaProgram(const PackedEtaProgram&)=default;
    PackedEtaProgram& operator=(const PackedEtaProgram&)=default;
    PackedEtaProgram(PackedEtaProgram&& other) noexcept { *this=std::move(other); }
    PackedEtaProgram& operator=(PackedEtaProgram&& other) noexcept {
        if(this!=std::addressof(other)) {
            owner_=other.owner_;generation_=other.generation_;dimension_=other.dimension_;
            blocks_=other.blocks_;scalars_=other.scalars_;edges_=other.edges_;tasks_=other.tasks_;
            operations_=std::move(other.operations_);
            task_offsets_=std::move(other.task_offsets_);
            // A moved-from table must not pass matches() with an empty program.
            other.invalidate();
        }
        return *this;
    }

    void invalidate() noexcept {
        owner_=nullptr;
        generation_=0;
        dimension_=0;
        blocks_=scalars_=edges_=tasks_=0;
        operations_.clear();
        task_offsets_.clear();
    }
    bool ready() const noexcept { return owner_!=nullptr; }
    int dimension() const noexcept { return dimension_; }
    std::size_t edgeCount() const noexcept { return edges_; }
    const std::vector<Operation>& operations() const noexcept { return operations_; }

    // generation must change for ANY schedule/topology/layout change, including
    // same-size edits. Packed values and allocations may change without rebinding.
    template<class Workspace>
    bool prepare(const Workspace& w,std::uint64_t generation) {
        if(matches(w,generation)) return true;
        return build(w,generation);
    }

    // Returns false without changing workspace data. An unsuccessful rebuild
    // invalidates this program; allocation exceptions also leave it invalid.
    template<class Workspace>
    bool build(const Workspace& w,std::uint64_t generation) {
        invalidate();
        if(!eligible(w)) return false;
        const int d=w.packed_stride;
        if(d!=4 && d!=6 && d!=12) return false;
        if(w.blocks<=0 || w.n<=0) return false;
        const std::size_t blocks=static_cast<std::size_t>(w.blocks);
        const std::size_t count=w.edges.size(),stride=static_cast<std::size_t>(d);
        const std::size_t limit=std::numeric_limits<std::size_t>::max();
        if(blocks>limit/stride || count>limit/(stride*stride) ||
           count>static_cast<std::size_t>(std::numeric_limits<int>::max())) return false;
        if(static_cast<std::size_t>(w.n)!=blocks*stride ||
           w.offsets.size()!=blocks+1 || w.tree_parent_edge.size()!=blocks ||
           w.tree_children.size()!=blocks || w.offsets.front()!=0) return false;
        for(std::size_t i=0;i<blocks;++i) {
            if(w.offsets[i]<0 || w.offsets[i+1]<w.offsets[i] ||
               w.offsets[i+1]-w.offsets[i]!=d) return false;
        }
        if(w.offsets.back()!=w.n ||
           w.packed_map.size()!=count*stride*stride ||
           w.packed_eta.size()!=count*stride || w.packed_next.size()!=count*stride ||
           w.packed_belief.size()!=blocks*stride) return false;

        const auto validEdge=[&](int p) {
            return p>=0 && static_cast<std::size_t>(p)<count;
        };
        for(std::size_t p=0;p<count;++p) {
            const auto& e=w.edges[p];
            if(e.from<0 || e.from>=w.blocks || e.to<0 || e.to>=w.blocks ||
               e.from==e.to || !validEdge(e.reverse)) return false;
            const auto& reverse=w.edges[e.reverse];
            if(reverse.reverse!=static_cast<int>(p) ||
               reverse.from!=e.to || reverse.to!=e.from) return false;
        }

        std::vector<Operation> program;
        if(count>program.max_size()/2) return false;
        program.reserve(count*2);
        // 0 = not computed; 1 = live proposal; 2 = committed this sweep.
        std::vector<unsigned char> state(count,0);
        const auto append=[&](int p,Kind kind) {
            if(!validEdge(p)) return false;
            if(kind==Kind::Commit) {
                if(state[p]!=1) return false;
                state[p]=2;
            } else {
                if(state[p]!=0) return false;
                state[p]=kind==Kind::Compute ? 1 : 2;
            }
            const auto& e=w.edges[p];
            const std::size_t eta=static_cast<std::size_t>(p)*stride;
            program.push_back({kind,eta*stride,eta,
                static_cast<std::size_t>(e.reverse)*stride,
                static_cast<std::size_t>(w.offsets[e.from]),
                static_cast<std::size_t>(w.offsets[e.to])});
            return true;
        };
        for(const auto& task:w.parallel_tasks) {
            task_offsets_.push_back(program.size());
            if(task.kind==0) {
                if(!append(task.index,Kind::ComputeCommit)) return false;
                continue;
            }
            if((task.kind!=1 && task.kind!=2) || task.index<0 ||
               task.index>=w.blocks) return false;
            const int node=task.index,parent=w.tree_parent_edge[node];
            if(parent < -1 || (parent>=0 &&
               (!validEdge(parent) || w.edges[parent].to!=node))) return false;
            const auto& children=w.tree_children[node];
            for(int child:children)
                if(!validEdge(child) || w.edges[child].from!=node) return false;
            if(task.kind==1) {
                for(auto it=children.rbegin();it!=children.rend();++it)
                    if(!append(w.edges[*it].reverse,Kind::Commit)) return false;
                if(parent>=0 && !append(w.edges[parent].reverse,Kind::Compute)) return false;
            } else {
                if(parent>=0 && !append(parent,Kind::Commit)) return false;
                for(int child:children)
                    if(!append(child,Kind::Compute)) return false;
            }
        }
        for(unsigned char status:state) if(status!=2) return false;
        task_offsets_.push_back(program.size());
        operations_=std::move(program);
        dimension_=d;
        blocks_=blocks;
        scalars_=blocks*stride;
        edges_=count;
        tasks_=w.parallel_tasks.size();
        generation_=generation;
        owner_=std::addressof(w);
        return true;
    }

    // Exactly one original message sweep. No automatic rebuild in the hot path.
    // False means no workspace writes occurred and the caller can use its fallback.
    template<class Workspace>
    bool sweep(Workspace& w,std::uint64_t generation) const {
        if(!matches(w,generation)) return false;
        const Buffers buffers{w.packed_map.data(),w.packed_eta.data(),
                              w.packed_next.data(),w.packed_belief.data()};
        const double omega=w.eta_relaxation;
        switch(dimension_) {
        case 4: dispatchRelaxation<4>(buffers,omega);break;
        case 6: dispatchRelaxation<6>(buffers,omega);break;
        case 12:dispatchRelaxation<12>(buffers,omega);break;
        default:return false;
        }
        return true;
    }

    // Caller has already prepared this epoch and enforces the original task
    // dependencies. Disjoint tasks may run concurrently; no readiness metadata
    // or global belief reduction is modified here.
    template<class Workspace> void task(Workspace& w,int k) const {
        const Buffers b{w.packed_map.data(),w.packed_eta.data(),w.packed_next.data(),w.packed_belief.data()};
        const auto begin=task_offsets_[k],end=task_offsets_[k+1];
        switch(dimension_) {
        case 4:dispatchRange<4>(b,w.eta_relaxation,begin,end);break;
        case 6:dispatchRange<6>(b,w.eta_relaxation,begin,end);break;
        case 12:dispatchRange<12>(b,w.eta_relaxation,begin,end);break;
        }
    }

private:
    struct Buffers {
        const double* map;
        double* eta;
        double* next;
        double* belief;
    };
    const void* owner_=nullptr;
    std::uint64_t generation_=0;
    int dimension_=0;
    std::size_t blocks_=0,scalars_=0,edges_=0,tasks_=0;
    std::vector<Operation> operations_;
    std::vector<std::size_t> task_offsets_;

    template<class Workspace>
    static bool eligible(const Workspace& w) {
        return w.parallel_tree && w.packed_eta_enabled && w.frozen && w.packed_ready &&
            w.eta_relaxation>0. && w.eta_relaxation<2.;
    }
    template<class Workspace>
    bool matches(const Workspace& w,std::uint64_t generation) const {
        if(!ready() || owner_!=std::addressof(w) || generation_!=generation ||
           !eligible(w) || w.packed_stride!=dimension_ || w.blocks<=0 || w.n<=0 ||
           static_cast<std::size_t>(w.blocks)!=blocks_ ||
           static_cast<std::size_t>(w.n)!=scalars_ || w.edges.size()!=edges_ ||
           w.parallel_tasks.size()!=tasks_ || w.offsets.size()!=blocks_+1 ||
           w.tree_parent_edge.size()!=blocks_ || w.tree_children.size()!=blocks_ ||
           task_offsets_.size()!=tasks_+1) return false;
        const std::size_t stride=static_cast<std::size_t>(dimension_);
        return w.packed_map.size()==edges_*stride*stride &&
            w.packed_eta.size()==edges_*stride && w.packed_next.size()==edges_*stride &&
            w.packed_belief.size()==scalars_;
    }

    template<int D>
    static double dot(const double* a,const double* b) noexcept {
        double value=0;
        for(int k=0;k<D;++k) value+=a[k]*b[k];
        return value;
    }
    template<int D,bool UnitRelaxation>
    static void compute(const Operation& op,const Buffers& b,double omega) noexcept {
        double cavity[D];
        const double* reverse=b.eta+op.reverse_offset;
        const double* source=b.belief+op.source_offset;
        for(int k=0;k<D;++k) cavity[k]=source[k]-reverse[k];
        const double* map=b.map+op.map_offset;
        double* out=b.next+op.eta_offset;
        if constexpr(UnitRelaxation) {
            for(int r=0;r<D;++r) out[r]=dot<D>(map+r*D,cavity);
        } else {
            const double* old=b.eta+op.eta_offset;
            for(int r=0;r<D;++r)
                out[r]=(1.-omega)*old[r]+omega*dot<D>(map+r*D,cavity);
        }
    }
    template<int D>
    static void commit(const Operation& op,const Buffers& b) noexcept {
        double* old=b.eta+op.eta_offset;
        const double* next=b.next+op.eta_offset;
        double* destination=b.belief+op.destination_offset;
        for(int r=0;r<D;++r) {destination[r]+=next[r]-old[r];old[r]=next[r];}
    }
    template<int D,bool UnitRelaxation>
    void execute(const Buffers& buffers,double omega) const noexcept {
        executeRange<D,UnitRelaxation>(buffers,omega,0,operations_.size());
    }
    template<int D,bool UnitRelaxation>
    void executeRange(const Buffers& buffers,double omega,std::size_t begin,std::size_t end) const noexcept {
        for(std::size_t k=begin;k<end;++k) {
            const Operation& op=operations_[k];
            switch(op.kind) {
            case Kind::Compute:compute<D,UnitRelaxation>(op,buffers,omega);break;
            case Kind::Commit:commit<D>(op,buffers);break;
            case Kind::ComputeCommit:
                // Finish ALL proposal rows before committing any row, as before.
                compute<D,UnitRelaxation>(op,buffers,omega);
                commit<D>(op,buffers);
                break;
            }
        }
    }
    template<int D> void dispatchRange(const Buffers& b,double omega,std::size_t begin,std::size_t end) const {
        if(omega==1.)executeRange<D,true>(b,omega,begin,end);
        else executeRange<D,false>(b,omega,begin,end);
    }
    template<int D>
    void dispatchRelaxation(const Buffers& buffers,double omega) const noexcept {
        static_assert(D==4 || D==6 || D==12,"unsupported packed eta dimension");
        if(omega==1.) execute<D,true>(buffers,omega);
        else execute<D,false>(buffers,omega);
    }
};

} // namespace pure_gbp
