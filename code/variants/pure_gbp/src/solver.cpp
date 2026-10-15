#include "pure_gbp/hierarchy.h"
#include <fstream>
#include <iomanip>
#include <memory>

namespace pure_gbp {
namespace {
std::unique_ptr<GBP> workspace;
std::unique_ptr<ThirdLevel> hierarchy;
bool projected_initialized=false;
int matrix_generation=0,solve_count=0;

ThirdConfig policy() {
    ThirdConfig p;
    p.group_size=policyInteger("HGBP_PURE_GROUP_SIZE",p.group_size,2);
    p.cycles=policyInteger("HGBP_PURE_CYCLES",p.cycles);
    p.pre_sweeps=policyInteger("HGBP_PURE_PRE_SWEEPS",p.pre_sweeps);
    p.top_sweeps=policyInteger("HGBP_PURE_TOP_SWEEPS",p.top_sweeps);
    p.top_serial_limit=policyInteger("HGBP_PURE_TOP_SERIAL_LIMIT",p.top_serial_limit,0);
    p.tolerance=policyTolerance("HGBP_PURE_TOLERANCE",p.tolerance);
    p.precision_tolerance=policyTolerance("HGBP_PURE_PRECISION_TOLERANCE",p.precision_tolerance);
    p.trace=std::getenv("HGBP_PURE_TRACE")!=nullptr;
    return p;
}
}

void begin(const Sparse& input,double ridge,int block_size,int threads) {
    if(input.rows()!=input.cols() || !input.rows() || threads<1 || !std::isfinite(ridge) || ridge<0)
        throw std::runtime_error("invalid pure GBP coarse equation");
    hierarchy.reset();
    Sparse a=input;
    for(int i=0;i<a.rows();++i)a.coeffRef(i,i)+=ridge;
    workspace=std::make_unique<GBP>(a,block_size,threads,false);
    workspace->synchronous_packed=true;
    projected_initialized=false;solve_count=0;++matrix_generation;
}

bool needsProjectedInitialization() {return !projected_initialized;}

void initializeProjected(const PairDiagonals& pairs) {
    if(!workspace)throw std::runtime_error("pure GBP context not initialized");
    if(projected_initialized)return;
    workspace->projectedInitialization(pairs,true);
    projected_initialized=true;
}

void solve(const Eigen::VectorXd& rhs,Eigen::VectorXd& result) {
    if(!workspace || !projected_initialized || rhs.size()!=workspace->n)
        throw std::runtime_error("pure GBP requires current projected factor budgets");
    double setup_sec=0;
    if(!hierarchy) {
        hierarchy=std::make_unique<ThirdLevel>(*workspace,policy());
        setup_sec=hierarchy->build_sec;
    }
    const Vec q=workspace->whitenRhs(rhs);Vec z;
    const auto stats=hierarchy->solve(*workspace,q,z);
    result=workspace->recover(z);++solve_count;
    if(!result.allFinite())throw std::runtime_error("pure GBP produced a nonfinite correction; no fallback");
    if(const char* filename=std::getenv("HGBP_PURE_TRACE")) {
        std::ofstream out(filename,std::ios::app);
        if(!out)throw std::runtime_error("could not open pure GBP trace");
        out<<std::setprecision(16)<<"{\"matrix\":"<<matrix_generation<<",\"solve\":"<<solve_count
           <<",\"lower_dimension\":"<<workspace->n<<",\"top_dimension\":"<<hierarchy->top->n
           <<",\"lower_pairs\":"<<workspace->edges.size()/2<<",\"top_pairs\":"<<hierarchy->top->edges.size()/2
           <<",\"lower_sweeps\":"<<stats.lower.sweeps<<",\"lower_full\":"<<stats.lower.full_sweeps
           <<",\"top_sweeps\":"<<stats.top.sweeps<<",\"top_full\":"<<stats.top.full_sweeps
           <<",\"lower_threads\":"<<stats.lower.observed_threads<<",\"top_threads\":"<<stats.top.observed_threads
           <<",\"setup_sec\":"<<setup_sec<<",\"lower_sec\":"<<stats.lower_sec<<",\"top_sec\":"<<stats.top_sec
           <<",\"transfer_sec\":"<<stats.transfer_sec<<",\"relative_residual\":"<<stats.residual<<"}\n";
    }
}
}
