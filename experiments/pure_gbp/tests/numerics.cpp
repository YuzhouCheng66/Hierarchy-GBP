#include "pure_gbp/hierarchy.h"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition,const char* message) {if(!condition)throw std::runtime_error(message);}
pure_gbp::Sparse graph(int d,int blocks) {
    std::vector<Eigen::Triplet<double>> entries;
    for(int i=0;i<d*blocks;++i)entries.emplace_back(i,i,3.+.002*i);
    for(int i=0;i<blocks;++i)for(int step:{1,3})if(i+step<blocks)
        for(int r=0;r<d;++r) {
            const int a=d*i+r,b=d*(i+step)+r;
            entries.emplace_back(a,b,.03);entries.emplace_back(b,a,.03);
        }
    pure_gbp::Sparse a(d*blocks,d*blocks);a.setFromTriplets(entries.begin(),entries.end());return a;
}
pure_gbp::PairDiagonals budgets(const pure_gbp::GBP& w) {
    pure_gbp::PairDiagonals pairs;
    for(int p=0;p<int(w.edges.size());p+=2) {
        const auto& e=w.edges[p];
        pairs[{e.from,e.to}]={.1*pure_gbp::Mat::Identity(w.size(e.from),w.size(e.from)),
            .1*pure_gbp::Mat::Identity(w.size(e.to),w.size(e.to)),
            pure_gbp::Vec::Zero(w.size(e.from)),pure_gbp::Vec::Zero(w.size(e.to))};
    }
    return pairs;
}
}
int main() {
    int cases=0;
    for(int d:{4,6,12})for(int threads:{1,16})for(int mode:{0,1,2}) {
        const auto a=graph(d,8);
        pure_gbp::GBP w(a,d,threads,false);
        if(mode==0)w.synchronous_packed=true;
        else {w.useParallelTreeSchedule();w.native_serial=mode==1;}
        pure_gbp::Vec b(a.rows());for(int i=0;i<b.size();++i)b[i]=1.+.003*i;
        const auto q=w.whitenRhs(b);pure_gbp::Vec z;
        for(int pass=0;pass<2;++pass) {
            const auto stats=w.solve(q,z,200,1.,1e-10,0,pass>0,1e-12);
            require(stats.spd && !stats.numerical_failure,"GBP failure on SPD graph");
            require((b-a*w.recover(z)).norm()/b.norm()<1e-9,"GBP linear residual mismatch");
        }
        w.solve(pure_gbp::Vec::Zero(w.n),z,10,1.,1e-10);
        require(z.norm()==0,"Zero RHS has nonzero correction");++cases;
    }
    for(int d:{4,6,12})for(int threads:{1,16}) {
        const auto a=graph(d,24);pure_gbp::GBP w(a,d,threads,false);w.synchronous_packed=true;
        w.projectedInitialization(budgets(w));pure_gbp::ThirdConfig config;
        pure_gbp::ThirdLevel hierarchy(w,config);
        pure_gbp::Vec q=pure_gbp::Vec::Ones(w.n),z;
        for(int pass=0;pass<2;++pass) {
            const auto stats=hierarchy.solve(w,q,z);
            require(z.allFinite() && stats.residual<1e-6,"Three-level correction residual");
            require(stats.energy<0,"Three-level correction is not a descent step");
            require(w.initial_lam.size()==w.edges.size(),"Original factor budgets not restored");
        }
        ++cases;
    }
    {
        pure_gbp::Sparse invalid(4,4);invalid.setIdentity();invalid.coeffRef(0,0)=-1;
        bool rejected=false;
        try {pure_gbp::GBP w(invalid,4);}catch(const std::runtime_error&) {rejected=true;}
        require(rejected,"Non-SPD block was silently accepted");++cases;
    }
    {
        std::vector<Eigen::Triplet<double>> entries;
        for(int i=0;i<8;++i)entries.emplace_back(i,i,1.);
        for(int i=0;i<4;++i) {entries.emplace_back(i,i+4,2.);entries.emplace_back(i+4,i,2.);}
        pure_gbp::Sparse indefinite(8,8);indefinite.setFromTriplets(entries.begin(),entries.end());
        pure_gbp::GBP w(indefinite,4,16,false);w.synchronous_packed=true;pure_gbp::Vec z;
        const auto result=w.solve(pure_gbp::Vec::Ones(8),z,20,1.,1e-8);
        require(!result.spd,"Indefinite cavity did not report failure");++cases;
    }
    std::cout<<"PASS "<<cases<<" pure GBP executor/hierarchy cases\n";
}
