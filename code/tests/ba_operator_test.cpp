#include "../src/ba_internal.h"

int main() {
    const Eigen::Vector3d residual(2.0, -1.0, 0.5);
    const Eigen::Vector3d projected(-0.7, 0.4, -0.1);
    BAStepModel model;
    model.linear_decrease = -residual.dot(projected);
    model.full_decrease = model.linear_decrease - 0.5 * projected.squaredNorm();
    for (double alpha : {1.0, 0.5, 0.125, 0.0}) {
        const double direct = 0.5 * (residual.squaredNorm() -
            (residual + alpha * projected).squaredNorm());
        if (std::abs(direct - model.decrease(alpha)) > 1e-14) {
            return 2;
        }
    }
    FastHGBPSystem sys;
    sys.free_cameras = 3;
    sys.ws.n = 3;
    sys.ws.unary_lam.assign(3, 2.0 * Mat9::Identity());
    sys.ws.preconditioner_lam.assign(3, 3.0 * Mat9::Identity());
    sys.ws.edges.resize(2);
    sys.ws.edges[0].factor_scale = 0.3;
    sys.ws.edges[1].factor_scale = 0.7;
    sys.row_block_offsets = {0, 1, 3, 4};
    sys.row_block_cols = {1, 0, 2, 1};
    sys.row_block_edge_ids = {0, 0, 1, 1};
    sys.row_blocks.assign(4, -0.4 * Mat9::Identity());

    Eigen::MatrixXd M = Eigen::MatrixXd::Zero(27, 27);
    for (int i = 0; i < 3; ++i) {
        M.block<9, 9>(9 * i, 9 * i) = sys.ws.preconditioner_lam[i];
        for (int p = sys.row_block_offsets[i]; p < sys.row_block_offsets[i + 1]; ++p) {
            M.block<9, 9>(9 * i, 9 * sys.row_block_cols[p]) =
                schurGBPPreconditionerEdgeScale(sys, p) * sys.row_blocks[p];
        }
    }
    Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(27, -1.0, 2.0);
    Eigen::VectorXd actual;
    fastHGBPMultiplyInto(sys, x, actual, 1);
    const double matvec_error = (actual - M * x).norm();

    const std::vector<std::vector<int>> groups = {{0, 1}, {2}};
    FastHGBPCoarseWorkspace coarse;
    buildFastHGBPCoarseWorkspace(coarse, sys, groups);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(27, 18);
    for (int i = 0; i < 3; ++i) {
        const int g = coarse.gid[i];
        P.block<9, 9>(9 * i, 9 * g) = coarse.scale[g] * Mat9::Identity();
    }
    const Eigen::MatrixXd exact_coarse = P.transpose() * M * P;
    const double coarse_error = (coarse.Ac - exact_coarse).norm();
    coarse.z = Eigen::VectorXd::LinSpaced(18, -0.5, 1.0);
    fastCoarseAPMultiplyInto(coarse, actual, 1);
    const double ap_error = (actual - M * P * coarse.z).norm();
    std::cout << "matvec_error=" << matvec_error
              << " coarse_error=" << coarse_error
              << " AP_error=" << ap_error << '\n';
    if (matvec_error > 1e-12 || coarse_error > 1e-10 || ap_error > 1e-12) {
        return 1;
    }
    return 0;
}
