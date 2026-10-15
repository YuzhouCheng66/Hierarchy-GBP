#include "../src/detail/ba_internal.h"

int main() {
    try {
        constexpr int n = 4096;
#if defined(_OPENMP)
        omp_set_dynamic(0);
        omp_set_num_threads(16);
#endif
        SchurGBPWorkspace initial;
        initial.n = n;
        initial.unary_lam.assign(n, 2.0 * Mat9::Identity());
        initial.preconditioner_lam = initial.unary_lam;
        initial.unary_eta.resize(n);
        initial.incoming.resize(n);
        Vec9List expected(n);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < 9; ++d) expected[i][d] = std::sin(0.13 * i + 0.2 * d);
            initial.unary_eta[i] = 2.0 * expected[i];
        }
        for (int i = 0; i < n; ++i) {
            SchurGBPEdge edge;
            edge.i = i;
            edge.j = (i + 1) % n;
            edge.msg_to_i = 2 * i;
            edge.msg_to_j = 2 * i + 1;
            edge.aij = -0.04 * Mat9::Identity();
            for (int d = 0; d < 8; ++d) edge.aij(d, d + 1) = 0.002;
            edge.aji = edge.aij.transpose();
            initial.unary_eta[edge.i] += edge.aij * expected[edge.j];
            initial.unary_eta[edge.j] += edge.aji * expected[edge.i];
            initial.incoming[edge.i].push_back(edge.msg_to_i);
            initial.incoming[edge.j].push_back(edge.msg_to_j);
            initial.edges.push_back(edge);
        }
        initial.belief_lam = initial.next_belief_lam = initial.unary_lam;
        initial.belief_eta = initial.next_belief_eta = initial.unary_eta;
        initial.msg_lam.assign(2 * n, Mat9::Zero());
        initial.next_msg_lam = initial.msg_lam;
        initial.msg_eta.assign(2 * n, Vec9::Zero());
        initial.next_msg_eta = initial.msg_eta;
        initial.fixed_eta_map.resize(2 * n);
        initial.belief_lam_inv.assign(n, Mat9::Zero());
        finalizeSchurGBPIncomingCSR(initial);
        if (effectiveGBPThreads(16, n, n) != 16)
            throw std::runtime_error("Fixture did not select the parallel GBP executor");
        auto serial = initial;
        auto parallel = initial;
        for (int sweep = 0; sweep < 30; ++sweep) {
            schurGBPSweep(serial, 1.0, 1);
            schurGBPSweep(parallel, 1.0, 16);
        }
        for (auto* ws : {&serial, &parallel}) {
            const auto precision = ws->msg_lam;
            const auto beliefs = ws->belief_lam;
            buildSchurGBPFixedEtaMaps(*ws, ws == &serial ? 1 : 16);
            buildSchurGBPBeliefLambdaInverses(*ws, ws == &serial ? 1 : 16);
            schurGBPFixedEtaSweeps(*ws, 1.0, 20, ws == &serial ? 1 : 16);
            for (int slot = 0; slot < 2 * n; ++slot)
                if ((ws->msg_lam[slot] - precision[slot]).norm() != 0.0)
                    throw std::runtime_error("Eta-only sweeps changed FV precision");
            for (int i = 0; i < n; ++i)
                if ((ws->belief_lam[i] - beliefs[i]).norm() != 0.0)
                    throw std::runtime_error("Eta-only sweeps changed belief precision");
        }
        Eigen::VectorXd x1, x16;
        schurGBPMeanInto(serial, 1, x1);
        schurGBPMeanInto(parallel, 16, x16);
        double exact_error = 0.0;
        for (int i = 0; i < n; ++i)
            exact_error = std::max(exact_error, (x1.segment<9>(9 * i) - expected[i]).norm());
        const double executor_error = (x1 - x16).cwiseAbs().maxCoeff();
        std::cout << "BA GBP exact_error=" << exact_error
                  << " serial_parallel_error=" << executor_error << '\n';
        if (exact_error > 1e-10 || executor_error > 1e-10)
            throw std::runtime_error("BA full/fixed-precision GBP failed known-solution test");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
