#include "internal/se3_precision_warm.h"
#include <algorithm>
#include <stdexcept>
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <omp.h>

namespace slam {
namespace {
using Mat6=Eigen::Matrix<double,6,6>;
Mat6 unpack(const double* packed) {
    Mat6 a;
    for(int c=0;c<6;++c) for(int r=0;r<=c;++r) a(r,c)=a(c,r)=packed[c*(c+1)/2+r];
    return a;
}
void pack(const Mat6& a,double* out) {
    for(int c=0;c<6;++c) for(int r=0;r<=c;++r) out[c*(c+1)/2+r]=a(r,c);
}
}

void SE3PrecisionWarmState::capture(const SyntheticSE3PackedSoAWorkspace& w,const SE3PoseVector& base,
    const Eigen::VectorXd* corrected_mean,int threads) {
    poses=base;
    precision=w.binary_msg_lam21;
    eta=w.binary_msg_eta;
    if(!corrected_mean) return;
    if(corrected_mean->size()!=6*w.num_vars) throw std::runtime_error("Warm capture mean dimension mismatch");
    Eigen::VectorXd correction(6*w.num_vars);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int i=0;i<w.num_vars;++i) {
        const Eigen::Map<const Eigen::Matrix<double,6,1>> prior(w.prior_eta.data()+6*i);
        correction.segment<6>(6*i)=unpack(w.belief_lam21.data()+21*i)*corrected_mean->segment<6>(6*i)-prior;
        for(int p=w.unary_offsets[i];p<w.unary_offsets[i+1];++p)
            correction.segment<6>(6*i)-=Eigen::Map<const Eigen::Matrix<double,6,1>>(w.unary_msg_eta.data()+6*w.unary_ids[p]);
        for(int p=w.binary_offsets[i];p<w.binary_offsets[i+1];++p)
            correction.segment<6>(6*i)-=Eigen::Map<const Eigen::Matrix<double,6,1>>(w.binary_msg_eta.data()+6*w.binary_slot_ids[p]);
    }
    // Minimum Euclidean eta lift with equal incoming-slot weights. Only the
    // saved warm guess is changed, not the live state or canonical factors.
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int slot=0;slot<2*w.num_binary_factors;++slot) {
        const int i=slot%2?w.binary_var1_id[slot/2]:w.binary_var0_id[slot/2];
        const double degree=w.binary_offsets[i+1]-w.binary_offsets[i];
        Eigen::Map<Eigen::Matrix<double,6,1>>(eta.data()+6*slot)+=correction.segment<6>(6*i)/degree;
    }
}

SE3PrecisionWarmStats initializeWarmSE3Precision(SyntheticSE3PackedSoAWorkspace& w,
    const SE3PrecisionWarmState& previous,const SE3PoseVector& current,bool bounded_mean,int threads,bool transport_eta) {
    if(previous.poses.size()!=current.size() || previous.precision.size()!=w.binary_msg_lam21.size() ||
       current.size()!=static_cast<size_t>(w.num_vars) || w.sweeps_since_relinearize!=0)
        throw std::runtime_error("Invalid SE3 precision warm-start state");
    if(transport_eta && (bounded_mean || previous.eta.size()!=w.binary_msg_eta.size()))
        throw std::runtime_error("Invalid SE3 eta transport state");
    Eigen::VectorXd mean,residual,eta;
    if(bounded_mean) {
        stackedMeanVectorSyntheticSE3PackedSoAWorkspaceInto(w,mean,threads);
        assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(w,eta,threads);
        multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(w,mean,residual,threads);
        residual=eta-residual;
    }
    std::vector<SE3ChartTransition,Eigen::aligned_allocator<SE3ChartTransition>> charts(current.size());
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1)
    for(int i=0;i<w.num_vars;++i) charts[i]=se3ChartTransition(previous.poses[i],current[i]);
    int projected=0,failed=0;
    std::vector<double> projections(2*w.num_binary_factors,0.);
    #pragma omp parallel for schedule(static) num_threads(threads) if(threads>1) reduction(+:projected,failed)
    for(int slot=0;slot<2*w.num_binary_factors;++slot) {
        const int f=slot/2,i=slot%2?w.binary_var1_id[f]:w.binary_var0_id[f];
        const Mat6 old=unpack(previous.precision.data()+21*slot);
        const Mat6& t=charts[i].jacobian;
        Mat6 next=t.transpose()*old*t;
        next=(.5*(next+next.transpose())).eval();
        if(!next.allFinite()) { ++failed; continue; }
        Eigen::Matrix<double,6,1> next_eta;
        if(transport_eta) {
            const Eigen::Map<const Eigen::Matrix<double,6,1>> old_eta(previous.eta.data()+6*slot);
            next_eta=t.transpose()*(old_eta-old*charts[i].offset);
            if(!next_eta.allFinite()) { ++failed; continue; }
        }
        Eigen::LLT<Mat6> positive(next);
        if(positive.info()!=Eigen::Success) {
            Eigen::SelfAdjointEigenSolver<Mat6> eig(next);
            if(eig.info()!=Eigen::Success || !eig.eigenvalues().allFinite()) { ++failed; continue; }
            if(eig.eigenvalues().minCoeff()<0) {
                const Mat6 psd=eig.eigenvectors()*eig.eigenvalues().cwiseMax(0.).asDiagonal()*eig.eigenvectors().transpose();
                projections[slot]=(psd-next).norm()/std::max(next.norm(),1e-30);
                next=psd; ++projected;
            }
        }
        pack(next,w.binary_msg_lam21.data()+21*slot);
        if(transport_eta) {
            Eigen::Map<Eigen::Matrix<double,6,1>>(w.binary_msg_eta.data()+6*slot)=next_eta;
        }
    }
    if(bounded_mean) rebuildSE3PackedMessagesAtMean(w,mean,residual,threads);
    else recomputeSE3PackedBeliefs(w,threads);
    // Never transfer a convergence certificate to a different linearized model.
    w.precision_frozen=false;
    w.precision_stable_checks=0;
    w.fixed_eta_maps_all_valid=0;
    std::fill(w.fixed_eta_map_valid.begin(),w.fixed_eta_map_valid.end(),0);
    std::fill(w.fixed_lam_initialized.begin(),w.fixed_lam_initialized.end(),0);
    const double max_projection=projections.empty()?0.:*std::max_element(projections.begin(),projections.end());
    return {2*w.num_binary_factors-failed,projected,failed,max_projection};
}
}
