#include "internal/se3_residual.h"
#include "internal/cycle_energy.h"
#include "internal/certified_coarse.h"
#include "internal/basis_alignment.h"
#include "internal/se3_precision_warm.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

slam::SyntheticSE3Problem problem(int n=5) {
    slam::SyntheticSE3Problem p;
    p.init_poses.resize(n);
    for (int i = 0; i < n; ++i) p.init_poses[i].t = Eigen::Vector3d(i * 1.07, i * 0.03, -i * 0.02);
    p.anchor_pose = p.init_poses.front();
    p.anchor_information = Eigen::Matrix<double, 6, 6>::Identity() * 1000.0;
    for (int i = 0; i < n-1; ++i) {
        slam::SyntheticSE3Edge edge;
        edge.i = i;
        edge.j = i + 1;
        edge.measurement.t = Eigen::Vector3d(1, 0, 0);
        edge.information = Eigen::Matrix<double, 6, 6>::Identity() * 10.0;
        p.edges.push_back(edge);
    }
    return p;
}

double relative(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a-b).norm()/std::max(b.norm(),1e-12);
}
}

int main() {
    try {
        {
            slam::SE3PoseVector origin(1);
            Eigen::VectorXd xi(6);
            const Eigen::Vector3d axis=Eigen::Vector3d(1,2,-3).normalized();
            double worst=0;
            for(double angle:{0.,1e-8,1e-5,1e-4,0.01,0.4,2.8}) {
                xi << .7,-1.1,.3,angle*axis.x(),angle*axis.y(),angle*axis.z();
                const auto next=slam::applyPoseDeltas(origin,xi);
                const auto chart=slam::se3ChartTransition(origin[0],next[0]);
                Eigen::Matrix<double,6,6> numeric;
                for(int j=0;j<6;++j) {
                    Eigen::VectorXd step=Eigen::VectorXd::Zero(6); step[j]=1e-6;
                    const auto plus=slam::applyPoseDeltas(next,step),minus=slam::applyPoseDeltas(next,-step);
                    numeric.col(j)=(slam::se3ChartTransition(origin[0],plus[0]).offset-
                        slam::se3ChartTransition(origin[0],minus[0]).offset)/(2e-6);
                }
                const double error=(numeric-chart.jacobian).norm()/chart.jacobian.norm();
                worst=std::max(worst,error);
                require(error<1e-6,"SE3 chart Jacobian disagrees with numerical chart derivative");
                const Eigen::Matrix<double,6,6> lam=Eigen::Matrix<double,6,6>::Identity()*3;
                const Eigen::Matrix<double,6,1> eta=Eigen::Matrix<double,6,1>::LinSpaced(1,6);
                const auto mapped_lam=(chart.jacobian.transpose()*lam*chart.jacobian).eval();
                const auto mapped_eta=(chart.jacobian.transpose()*(eta-lam*chart.offset)).eval();
                Eigen::Matrix<double,6,1> z=Eigen::Matrix<double,6,1>::LinSpaced(-.03,.02);
                const Eigen::Matrix<double,6,1> old=chart.offset+chart.jacobian*z;
                const double difference=.5*old.dot(lam*old)-eta.dot(old)-
                    (.5*chart.offset.dot(lam*chart.offset)-eta.dot(chart.offset));
                require(std::abs(difference-(.5*z.dot(mapped_lam*z)-mapped_eta.dot(z)))<1e-12,
                        "affine Gaussian chart transport energy identity");
            }
            std::cout << "chart_relative_error=" << worst << '\n';
        }
        {
            Eigen::MatrixXd reference=Eigen::MatrixXd::Identity(8,3);
            Eigen::Matrix3d rotation;
            rotation << 0,-1,0,1,0,0,0,0,-1;
            Eigen::MatrixXd current=reference*rotation;
            const Eigen::MatrixXd projector=current*current.transpose();
            slam::alignBasisCoordinates(current,reference);
            require((current-reference).norm()<1e-12,"Procrustes failed exact basis rotation");
            require((current*current.transpose()-projector).norm()<1e-12,"Procrustes changed subspace");
        }
        {
            Eigen::MatrixXd a=Eigen::MatrixXd::Zero(24,24);
            for(int i=0;i<24;++i) a(i,i)=1.0+i*i;
            const Eigen::VectorXd b=Eigen::VectorXd::Ones(24);
            Eigen::VectorXd x,r,z,p,ap;
            const auto good=slam::certifiedCoarsePcg(a,1e-8,b,
                [&](const Eigen::VectorXd& rhs,Eigen::VectorXd& out) { out=rhs.cwiseQuotient(a.diagonal()); },x,r,z,p,ap);
            require(good.accepted && good.iterations<=2,"close coarse preconditioner was rejected");
            require(good.preconditioner_calls==good.iterations && good.matvec_calls==good.iterations+1,
                    "accepted coarse work counters are incorrect");
            require((a*x+1e-8*x-b).norm()/b.norm()<=1e-6,"coarse certificate ignored true current residual");
            const auto bad=slam::certifiedCoarsePcg(a,0,b,
                [](const Eigen::VectorXd& rhs,Eigen::VectorXd& out) { out=rhs; },x,r,z,p,ap);
            require(!bad.accepted && bad.iterations==6,"stale coarse preconditioner did not request fallback");
            require(bad.preconditioner_calls==6 && bad.matvec_calls==7,
                    "iteration-cap path performed an unused triangular solve");
            const auto zero=slam::certifiedCoarsePcg(a,0,Eigen::VectorXd::Zero(24),
                [](const Eigen::VectorXd& rhs,Eigen::VectorXd& out) { out=rhs; },x,r,z,p,ap);
            require(zero.accepted && x.norm()==0,"zero coarse RHS not handled");
            require(zero.preconditioner_calls==0 && zero.matvec_calls==0,"zero RHS did unnecessary work");
            a=-a;
            const auto indefinite=slam::certifiedCoarsePcg(a,0,b,
                [](const Eigen::VectorXd& rhs,Eigen::VectorXd& out) { out=rhs; },x,r,z,p,ap);
            require(!indefinite.accepted,"negative coarse curvature accepted");
            const Eigen::SparseMatrix<double> diagonal=Eigen::MatrixXd::Identity(100,100).sparseView();
            require(slam::coarseReuseWorkRatio(diagonal,100,5)>1.,"diagonal factor should be rebuilt cheaply");
            Eigen::SparseMatrix<double> filled(500,500);
            for(int j=0;j<500;++j) for(int i=j;i<500;++i) filled.insert(i,j)=1.;
            filled.makeCompressed();
            require(slam::coarseReuseWorkRatio(filled,1500,5)<1.,"high fill should admit reuse trial");
            const auto work=slam::estimateCoarseWork(filled,1500);
            require(work.extraWork(zero)==0,"zero RHS created amortization debt");
            require(work.extraWork(bad)==5.*work.solve_work+7.*work.matvec_work,
                    "amortization must subtract one necessary baseline solve");
            double debt=0;
            int repeats=0;
            while(debt<work.factor_work && repeats<100000) { debt+=work.extraWork(bad); ++repeats; }
            require(repeats>1 && repeats<100000,"amortization should trigger from accumulated work");
        }
        Eigen::Matrix3d h;
        h << 4, 1, 0, 1, 3, 0.2, 0, 0.2, 2;
        Eigen::VectorXd rhs(3);
        rhs << 1, 2, 3;
        Eigen::VectorXd sol = Eigen::VectorXd::Zero(3);
        Eigen::VectorXd residual = rhs;
        slam::CycleEnergyHistory energy(3);
        for (int i=0; i<3; ++i) {
            const Eigen::VectorXd x0 = sol, r0 = residual;
            const double old_energy = 0.5 * sol.dot(h*sol) - rhs.dot(sol);
            sol += 0.1 * residual;
            residual = rhs - h*sol;
            require(energy.accept(x0,r0,sol,residual), "positive SPD cycle rejected");
            require((residual-(rhs-h*sol)).norm() < 1e-12, "incremental residual mismatch");
            require(0.5*sol.dot(h*sol)-rhs.dot(sol) <= old_energy+1e-12, "quadratic energy increased");
        }
        require(residual.norm() < 1e-10, "conjugate directions failed a three-dimensional SPD solve");
        const auto p = problem();
        auto graph = slam::buildLinearizedResidualGraph(p, p.init_poses);
        auto full = slam::buildSyntheticSE3PackedSoAWorkspace(p);
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(full, graph);
        auto adaptive = full;
        auto split = full;
        auto parallel = full;
        adaptive.adaptive_precision = split.adaptive_precision = parallel.adaptive_precision = true;
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(full, 250, 1, -1, 0.3, 0);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(adaptive, 250, 1, -1, 0.3, 0);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(parallel, 250, 16, -1, 0.3, 0);
        for (int cycle=0; cycle<5; ++cycle)
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(split, 50, 1, -1, 0.3, 0);
        const auto x = slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(full);
        const auto a = slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(adaptive);
        const auto b = slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(split);
        const auto c = slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(parallel);
        std::cout << "diagnostic residual=" << adaptive.last_precision_residual
                  << " checks=" << adaptive.precision_checks
                  << " means=" << relative(a,x) << '\n';
        require(adaptive.precision_freezes > 0, "tree did not reach a precision fixed point");
        require(adaptive.eta_only_sweeps > 0, "eta-only path was not exercised");
        require(adaptive.full_precision_sweeps + adaptive.eta_only_sweeps == 250, "sweep accounting");
        require(relative(a,x) < 1e-6, "adaptive/full mean mismatch");
        require(relative(a,b) < 1e-12, "cycle partition changed the state");
        require(relative(a,c) < 1e-6, "thread-count mean mismatch");
        require(adaptive.precision_checks >= 4, "freeze rechecks missing");

        // Invalidating one edge must not be hidden by averaging over the graph.
        adaptive.binary_diag0_lam21[0] += 1.0;
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(adaptive, 25, 1, -1, 0.3, 0);
        require(adaptive.precision_thaws > 0, "changed model did not thaw frozen precision");
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(adaptive, graph);
        require(!adaptive.precision_frozen && adaptive.precision_checks == 0, "relinearization kept a stale certificate");
        require(adaptive.fixed_eta_maps_all_valid == 0, "relinearization kept stale eta maps");
        auto balanced=slam::buildSyntheticSE3PackedSoAWorkspace(p);
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(balanced,graph);
        const auto joint=graph.jointDistributionInfSparse();
        slam::initializeBalancedSE3PackedMessages(balanced,1);
        Eigen::VectorXd eta;
        slam::assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(balanced,eta,1);
        require(relative(eta,joint.eta)<1e-12,"balanced initialization changed RHS");
        require(relative(Eigen::Map<const Eigen::VectorXd>(balanced.belief_eta.data(),eta.size()),eta)<1e-12,
                "balanced beliefs do not sum to true gradient");
        Eigen::VectorXd mean=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(balanced);
        const Eigen::MatrixXd dense=joint.lam;
        for(int threads:{1,16}) {
            auto jacobi=balanced;
            Eigen::MatrixXd inverse=Eigen::MatrixXd::Zero(dense.rows(),dense.cols());
            for(int i=0;i<jacobi.num_vars;++i)
                inverse.block<6,6>(6*i,6*i)=dense.block<6,6>(6*i,6*i).ldlt().solve(Eigen::Matrix<double,6,6>::Identity());
            Eigen::VectorXd reference=mean;
            for(int count:{1,2,7,50}) {
                for(int j=0;j<count;++j) reference+=(2./3.)*inverse*(eta-dense*reference);
                slam::blockJacobiSE3PackedIterations(jacobi,count,threads);
                require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(jacobi),reference)<1e-10,
                        "packed Jacobi disagrees with explicit matrix iteration");
            }
            require(jacobi.jacobi_sweeps==60 && jacobi.full_precision_sweeps==0,
                    "Jacobi control sweep accounting");
            const Eigen::VectorXd correction=Eigen::VectorXd::LinSpaced(mean.size(),-0.1,0.2);
            slam::applyMeanDeltaSyntheticSE3PackedSoAWorkspace(jacobi,correction,threads);
            reference+=correction;
            reference+=(2./3.)*inverse*(eta-dense*reference);
            slam::blockJacobiSE3PackedIterations(jacobi,1,threads);
            require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(jacobi),reference)<1e-10,
                    "packed Jacobi discarded injected correction");
            slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(jacobi,graph);
            require(!jacobi.jacobi_ready && jacobi.jacobi_sweeps==0,"stale Jacobi cache after relinearization");
        }
        for(int i=0;i<balanced.num_vars;++i)
            require((dense.block<6,6>(6*i,6*i)*mean.segment<6>(6*i)-eta.segment<6>(6*i)).norm()<1e-9,
                    "balanced initial mean is not diagonal solve");
        auto balanced_parallel=balanced, balanced_adaptive=balanced;
        balanced_adaptive.adaptive_precision=true;
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(balanced,250,1,-1,0.3,2);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(balanced_parallel,250,16,-1,0.3,2);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(balanced_adaptive,250,16,-1,0.3,2);
        const auto base=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(balanced);
        {
            slam::SE3PrecisionWarmState saved;
            saved.capture(balanced,p.init_poses);
            {
                auto mapped=balanced;
                slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(mapped,graph);
                slam::initializeBalancedSE3PackedMessages(mapped,1);
                slam::initializeWarmSE3Precision(mapped,saved,p.init_poses,false,1,true);
                require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(mapped),base)<1e-9,
                        "identity full Gaussian transport changed belief");
                const Eigen::VectorXd target=base+Eigen::VectorXd::LinSpaced(base.size(),-.1,.2);
                slam::SE3PrecisionWarmState lifted;
                lifted.capture(balanced,p.init_poses,&target,16);
                require(balanced.binary_msg_eta==saved.eta,"warm capture modified live messages");
                auto lifted_workspace=balanced;
                lifted_workspace.binary_msg_eta=lifted.eta;
                slam::recomputeSE3PackedBeliefs(lifted_workspace,16);
                require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(lifted_workspace),target)<1e-9,
                        "saved FV eta lift did not match corrected mean");
            }
            for(int threads:{1,16}) for(bool bounded:{false,true}) {
                auto warm=slam::buildSyntheticSE3PackedSoAWorkspace(p);
                slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(warm,graph);
                slam::initializeBalancedSE3PackedMessages(warm,threads);
                const auto cold_mean=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(warm);
                const auto stats=slam::initializeWarmSE3Precision(warm,saved,p.init_poses,bounded,threads);
                require(stats.slots==2*warm.num_binary_factors && stats.failed==0,"precision transport lost a valid slot");
                require(!warm.precision_frozen && warm.sweeps_since_relinearize==0 && warm.fixed_eta_maps_all_valid==0,
                        "warm precision reused a stale certificate");
                Eigen::VectorXd b_after,product;
                slam::assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(warm,b_after,threads);
                slam::multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(warm,cold_mean,product,threads);
                require((b_after-eta).norm()<1e-12 && (product-dense*cold_mean).norm()<1e-9,"warm precision changed H,b");
                if(bounded) require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(warm),cold_mean)<1e-12,
                                    "bounded warm start changed initial mean");
                slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(warm,250,threads,-1,0.3,2);
                require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(warm),base)<1e-5,
                        "warm precision changed converged Gaussian solution");
            }
        }
        for(int threads:{1,16}) {
            auto rebuilt=balanced;
            const Eigen::VectorXd target=Eigen::VectorXd::LinSpaced(eta.size(),-0.3,0.5);
            const Eigen::VectorXd residual=eta-dense*target;
            const auto old_precision=rebuilt.binary_msg_lam21;
            slam::rebuildSE3PackedMessagesAtMean(rebuilt,target,residual,threads);
            require(rebuilt.binary_msg_lam21==old_precision,"message reconstruction altered precision");
            for(int i=0;i<rebuilt.num_vars;++i) {
                Eigen::Matrix<double,6,6> b;
                for(int c=0;c<6;++c) for(int r=0;r<=c;++r)
                    b(r,c)=b(c,r)=rebuilt.belief_lam21[21*i+c*(c+1)/2+r];
                const Eigen::Map<const Eigen::Matrix<double,6,1>> sum(rebuilt.belief_eta.data()+6*i);
                require((sum-b*target.segment<6>(6*i)).norm()<1e-8,
                        "reconstructed message sum is inconsistent with corrected mean");
            }
            Eigen::VectorXd rhs_after,product_after;
            slam::assembleJointEtaSyntheticSE3PackedSoAWorkspaceInto(rebuilt,rhs_after,threads);
            slam::multiplyJointLambdaSyntheticSE3PackedSoAWorkspaceInto(rebuilt,target,product_after,threads);
            require((rhs_after-eta).norm()<1e-12 && (product_after-dense*target).norm()<1e-9,
                    "message reconstruction changed the canonical system");
            const Eigen::VectorXd exact=dense.ldlt().solve(eta);
            slam::rebuildSE3PackedMessagesAtMean(rebuilt,exact,eta-dense*exact,threads);
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(rebuilt,1,threads,-1,0,0);
            require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(rebuilt),exact)<1e-6,
                    "reconstruction disturbed the true stationary point");
        }
        require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(balanced_parallel),base)<1e-6,
                "balanced 1/16-thread mismatch");
        require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(balanced_adaptive),base)<1e-6,
                "balanced adaptive/full mismatch");
        auto stationary=slam::buildSyntheticSE3PackedSoAWorkspace(p);
        slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(stationary,graph);
        std::fill(stationary.prior_eta.begin(),stationary.prior_eta.end(),0);
        std::fill(stationary.unary_eta.begin(),stationary.unary_eta.end(),0);
        for(int f=0;f<stationary.num_binary_factors;++f) for(int j=0;j<6;++j) {
            stationary.binary_eta0[6*f+j]=j+1;
            stationary.binary_eta1[6*f+j]=-j-1;
            stationary.prior_eta[6*stationary.binary_var0_id[f]+j]-=j+1;
            stationary.prior_eta[6*stationary.binary_var1_id[f]+j]+=j+1;
        }
        slam::initializeBalancedSE3PackedMessages(stationary,16);
        slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(stationary,100,16,-1,0.3,2);
        require(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(stationary).norm()<1e-12,
                "nonzero balanced factor gradients disturbed a stationary iterate");
        for(int n:{5,140}) for(bool adaptive_policy:{false,true}) for(bool cyclic:{false,true}) {
            auto p2=problem(n);
            if(cyclic) for(int i=0;i<n-2;i+=7) {
                slam::SyntheticSE3Edge edge;
                edge.i=i; edge.j=std::min(i+11,n-1);
                edge.measurement.t=p2.init_poses[edge.j].t-p2.init_poses[edge.i].t;
                edge.information=Eigen::Matrix<double,6,6>::Identity()*4;
                p2.edges.push_back(edge);
            }
            const auto graph2=slam::buildLinearizedResidualGraph(p2,p2.init_poses);
            auto legacy=slam::buildSyntheticSE3PackedSoAWorkspace(p2);
            slam::relinearizeSyntheticSE3PackedSoAWorkspaceFromGraph(legacy,graph2);
            slam::initializeBalancedSE3PackedMessages(legacy,16);
            legacy.adaptive_precision=adaptive_policy;
            auto persistent=legacy, split_persistent=legacy;
            persistent.persistent_sweeps=split_persistent.persistent_sweeps=true;
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(legacy,250,16,-1,0.3,2);
            slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(persistent,250,16,-1,0.3,2);
            for(int i=0;i<5;++i) slam::synchronousIterationsSyntheticSE3PackedSoAWorkspace(split_persistent,50,16,-1,0.3,2);
            const auto reference=slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(legacy);
            require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(persistent),reference)<1e-10,
                    "persistent executor changed synchronous iterates");
            require(relative(slam::stackedMeanVectorSyntheticSE3PackedSoAWorkspace(split_persistent),reference)<1e-10,
                    "persistent executor changed split-cycle iterates");
            require(legacy.eta_only_sweeps==persistent.eta_only_sweeps && legacy.precision_checks==persistent.precision_checks,
                    "persistent executor changed precision scheduling");
            require(legacy.binary_msg_eta==persistent.binary_msg_eta &&
                    legacy.binary_msg_lam21==persistent.binary_msg_lam21 &&
                    legacy.binary_msg_eta==split_persistent.binary_msg_eta &&
                    legacy.binary_msg_lam21==split_persistent.binary_msg_lam21,
                    "persistent/split executor changed message bits");
        }
        std::cout << "PASS adaptive/full=" << relative(a,x)
                  << " threads=" << relative(a,c)
                  << " freeze_sweep=" << split.first_precision_freeze_sweep
                  << " eta_only=" << split.eta_only_sweeps << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
