#define main ba_benchmark_main
#include "../src/ba_solver.cpp"
#undef main
#include "ba_dataflow_checks.h"
#include "ba_policy_checks.h"

int testCanonicalGBP() {
    PackedBlockSchurPattern pattern;
    pattern.free_cameras = pattern.total_cameras = 3;
    pattern.fix_first_camera = false;
    pattern.row_ptr = {0, 3, 6, 9};
    pattern.col_idx = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) pattern.slot_of[blockKey(i, j)] = 3 * i + j;
    }
    FastHGBPSystem sys;
    initializeFastHGBPSystemStructure(sys, pattern);
    for (int i = 0; i < 3; ++i) {
        Mat9 L = Mat9::Identity();
        for (int r = 0; r < 9; ++r) {
            L(r, r) = 1.0 + 0.1 * r + i;
            for (int c = 0; c < r; ++c) L(r, c) = 0.1 * std::sin(i + 2.0 * r + c);
        }
        sys.ws.unary_lam[i] = L * L.transpose();
    }
    for (size_t e = 0; e < sys.ws.edges.size(); ++e) {
        Mat9 cross;
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) cross(r, c) = 3.0 * std::cos(0.2 * (1 + e + r + 3 * c));
        }
        sys.row_blocks[sys.edge_row_pos_i[e]] = cross;
        sys.row_blocks[sys.edge_row_pos_j[e]] = cross.transpose();
    }
    normalizeBAGBPCanonicalPairs(sys, 1);
    refreshPSDPairPreconditioner(sys, 1);
    Eigen::MatrixXd M(27, 27);
    for (int i = 0; i < 27; ++i) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(27), product;
        unit[i] = 1.0;
        fastHGBPMultiplyInto(sys, unit, product, 1);
        M.col(i) = product;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(M);
    if (eig.eigenvalues().minCoeff() <= 0.0) return 4;
    const Eigen::VectorXd rhs = Eigen::VectorXd::LinSpaced(27, -1.0, 2.0);
    resetSchurGBPEtaMessages(sys.ws, rhs);
    for (int sweep = 0; sweep < 100; ++sweep) schurGBPSweep(sys.ws, 1.0);
    Eigen::VectorXd full_mean, fixed_mean;
    schurGBPMeanInto(sys.ws, 1, full_mean);
    buildSchurGBPFixedEtaMaps(sys.ws);
    buildSchurGBPBeliefLambdaInverses(sys.ws);
    resetSchurGBPEtaMessages(sys.ws, rhs);
    schurGBPFixedEtaSweeps(sys.ws, 1.0, 100, 1);
    schurGBPMeanInto(sys.ws, 1, fixed_mean);
    const Eigen::VectorXd direct = M.ldlt().solve(rhs);
    const double full_error = (full_mean - direct).norm() / direct.norm();
    const double fixed_error = (fixed_mean - direct).norm() / direct.norm();
    std::cout << "canonical_GBP_min_eigenvalue=" << eig.eigenvalues().minCoeff()
              << " full_mean_error=" << full_error << " fixed_mean_error=" << fixed_error << '\n';
    return full_error < 1e-9 && fixed_error < 1e-9 ? 0 : 5;
}

int testGaugeJacobian() {
    const Mat3 R = rodriguesMatrix(Vec3(0.2, -0.1, 0.3));
    const Vec3 t(0.1, -0.2, 1.2), X(0.5, 0.7, 3.0), center(-0.4, 0.1, 0.6);
    Vec2 residual;
    Mat29 Jc;
    Mat23 Jp;
    linearizePackedBalObservation(Vec2(20.0, 30.0), X, R, t,
        Vec3(400.0, -0.01, 0.001), residual, Jc, Jp);
    double max_error = 0.0;
    for (int mode = 0; mode < 7; ++mode) {
        Vec9 camera = Vec9::Zero();
        Vec3 point;
        if (mode < 3) {
            point = Vec3::Unit(mode);
            camera.head<3>() = -R * point;
        } else if (mode < 6) {
            const Vec3 axis = Vec3::Unit(mode - 3);
            point = axis.cross(X - center);
            camera.head<3>() = -(t + R * center).cross(R * axis);
            camera.segment<3>(3) = -R * axis;
        } else {
            point = X - center;
            camera.head<3>() = t + R * center;
        }
        max_error = std::max(max_error, (Jc * camera + Jp * point).norm());
    }
    std::cout << "geometric_gauge_Jacobian_error=" << max_error << '\n';
    return max_error < 1e-10 ? 0 : 6;
}

int testRetractionJacobian() {
    RootProblem problem;
    problem.cameras().resize(1);
    auto& camera = problem.cameras()[0];
    camera.T_c_w = RootProblem::SE3(RootProblem::SO3::exp(Vec3(0.2, -0.1, 0.3)), Vec3(0.1, -0.2, 1.2));
    camera.intrinsics = RootProblem::CameraModel(Vec3(400.0, -0.01, 0.001));
    const auto original = camera;
    const Vec3 point(0.5, 0.7, 3.0);
    const Vec2 measurement(20.0, 30.0);
    Mat29 Jc, scratch_c;
    Mat23 Jp, scratch_p;
    Vec2 residual;
    linearizePackedBalObservation(measurement, point, camera.T_c_w.rotationMatrix(),
        camera.T_c_w.translation(), camera.intrinsics.getParam(), residual, Jc, Jp);
    double max_error = 0.0;
    for (int k = 0; k < 12; ++k) {
        Vec2 numeric[2];
        const double h = k == 6 ? 1e-3 : 1e-6;
        for (int sign = 0; sign < 2; ++sign) {
            camera = original;
            Vec3 displaced = point;
            const double step = sign == 0 ? h : -h;
            if (k < 9) {
                Eigen::VectorXd delta = Eigen::VectorXd::Zero(9);
                delta[k] = step;
                applyRootCameraIncrementDirect(problem, delta);
            } else {
                displaced[k - 9] += step;
            }
            linearizePackedBalObservation(measurement, displaced, camera.T_c_w.rotationMatrix(),
                camera.T_c_w.translation(), camera.intrinsics.getParam(), numeric[sign], scratch_c, scratch_p);
        }
        const Vec2 analytic = k < 9 ? Vec2(Jc.col(k)) : Vec2(Jp.col(k - 9));
        max_error = std::max(max_error, ((numeric[0] - numeric[1]) / (2 * h) - analytic).norm() /
            std::max(1.0, analytic.norm()));
    }
    std::cout << "actual_retraction_Jacobian_relative_error=" << max_error << '\n';
    return max_error < 1e-7 ? 0 : 13;
}

int testConnectedAggregation() {
    FastHGBPSystem sys;
    sys.free_cameras = sys.ws.n = 16;
    const std::vector<int> chain = {0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5, 12, 6, 13};
    for (size_t k = 1; k < chain.size(); ++k) {
        SchurGBPEdge edge;
        edge.i = chain[k - 1];
        edge.j = chain[k];
        edge.aij = edge.aji = Mat9::Identity();
        sys.ws.edges.push_back(edge);
    }
    const auto connected = buildConnectedBAGroups(sys, 5);
    const int disconnected = baDisconnectedGroups(sys, connected);
    std::cout << "connected_disconnected=" << disconnected << '\n';
    if (disconnected != 0 || connected != buildConnectedBAGroups(sys, 5)) return 15;
    std::vector<int> seen(sys.free_cameras, 0);
    for (const auto& group : connected) {
        for (int camera : group) ++seen[camera];
    }
    if (std::any_of(seen.begin(), seen.end(), [](int count) { return count != 1; })) return 16;
    for (const auto& group : connected) if (group.size() > 10) return 16;
    return 0;
}

int main() {
    if (const int status = testDataflowSchur()) return status;
    if (const int status = testSharedCrossBlocks()) return status;
    if (const int status = testUniformPolicyArguments()) return status;
    if (const int status = testCanonicalGBP()) return status;
    if (const int status = testGaugeJacobian()) return status;
    if (const int status = testRetractionJacobian()) return status;
    if (const int status = testConnectedAggregation()) return status;
    Mat23 degenerate_J;
    for (int k = 0; k < 3; ++k) {
        degenerate_J(0, k) = std::cos(0.1 + 0.6 * k);
        degenerate_J(1, k) = std::sin(0.1 + 0.6 * k);
    }
    for (double lambda : {1e-6, 1e-8, 1e-9, 1e-10, 1e-12}) {
        const Mat3 B = degenerate_J.transpose() * degenerate_J;
        Mat3 L;
        if (!factorDampedPoint3(B, lambda, L)) return 10;
        Mat23 white = degenerate_J;
        whitenPointCross(white, L);
        const Eigen::Matrix2d reduced = Eigen::Matrix2d::Identity() - white * white.transpose();
        const Eigen::Matrix2d exact = lambda * (degenerate_J * degenerate_J.transpose() + lambda * Eigen::Matrix2d::Identity()).inverse();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eigen(reduced);
        const Eigen::Matrix2d old_reduced = Eigen::Matrix2d::Identity() - degenerate_J *
            (B + lambda * Mat3::Identity()).inverse() * degenerate_J.transpose();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> old_eigen((0.5 * (old_reduced + old_reduced.transpose())).eval());
        std::cout << "point_lambda=" << lambda << " sqrt_error=" << (reduced - exact).norm()
                  << " sqrt_min_eigenvalue=" << eigen.eigenvalues().minCoeff()
                  << " inverse_min_eigenvalue=" << old_eigen.eigenvalues().minCoeff() << '\n';
        if ((reduced - exact).norm() > 1e-13 || eigen.eigenvalues().minCoeff() <= 0.0) return 11;
    }
    RootProblem problem;
    problem.landmarks().resize(2);
    PackedRootLandmarks packed;
    packed.point_offset = {0, 2, 4};
    packed.camera_id = {0, 1, 0, 1};
    packed.Jc.resize(4);
    packed.Jl.resize(4);
    packed.residual.resize(4);
    packed.Hll.resize(2);
    packed.Hll_factor.resize(2);
    packed.bl.resize(2);
    packed.Jl_scale.assign(2, Vec3(0.2, 0.7, 1.3));
    const double damping = 0.13;
    for (int e = 0; e < 4; ++e) {
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 9; ++c) {
                packed.Jc[e](r, c) = std::sin(0.7 * (1 + e + 2 * r + 3 * c));
            }
            for (int c = 0; c < 3; ++c) {
                packed.Jl[e](r, c) = std::cos(0.4 * (1 + 2 * e + r + 3 * c));
            }
        }
        packed.residual[e] = Vec2(0.1 * e - 0.4, 0.3 * e + 0.6);
    }
    for (int p = 0; p < 2; ++p) {
        problem.landmarks()[p].p_w = Vec3::Zero();
        packed.Hll[p].setZero();
        packed.bl[p].setZero();
        for (int e = 2 * p; e < 2 * p + 2; ++e) {
            packed.Hll[p].noalias() += packed.Jl[e].transpose() * packed.Jl[e];
            packed.bl[p].noalias() += packed.Jl[e].transpose() * packed.residual[e];
        }
        if (!factorDampedPoint3(packed.Hll[p], damping, packed.Hll_factor[p])) return 10;
    }
    const Eigen::VectorXd camera_delta = Eigen::VectorXd::LinSpaced(18, -0.03, 0.05);
    Vec3List point_delta;
    const BAStepModel model = backSubstitutePackedRoot(
        problem, packed, camera_delta, damping, point_delta);
    double maximum_error = 0.0;
    for (double alpha : {0.0, 1e-3, 0.25, 0.5, 1.0}) {
        double direct = 0.0;
        for (int e = 0; e < 4; ++e) {
            const int p = e / 2;
            const Vec3 point_scaled = point_delta[p].cwiseQuotient(packed.Jl_scale[p]);
            const Vec2 j_delta = packed.Jc[e] * camera_delta.segment<9>(9 * packed.camera_id[e]) +
                packed.Jl[e] * point_scaled;
            direct += 0.5 * (packed.residual[e].squaredNorm() -
                (packed.residual[e] + alpha * j_delta).squaredNorm());
        }
        maximum_error = std::max(maximum_error, std::abs(direct - model.decrease(alpha)));
    }
    std::cout << "packed_backsub_scaled_model_error=" << maximum_error << '\n';
    if (maximum_error >= 1e-12) return 1;
    packed.camera_rotation = {Mat3::Identity(), rodriguesMatrix(Vec3(0.1, 0.2, 0.3))};
    packed.camera_translation = {Vec3(0.1, 0.2, 1.0), Vec3(-0.3, 0.1, 1.2)};
    const Eigen::VectorXd scaling = Eigen::VectorXd::LinSpaced(18, 0.5, 2.0);
    Eigen::MatrixXd S(18, 18);
    for (int c = 0; c < 18; ++c) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(18), y;
        e[c] = 1.0;
        packedRootExactSchurMultiplyInto(packed, scaling, damping, e, 1, y);
        S.col(c) = y;
    }
    const Eigen::VectorXd linear_rhs = Eigen::VectorXd::LinSpaced(18, -0.5, 0.9);
    PackedBlockSchurPattern linear_pattern;
    linear_pattern.free_cameras = linear_pattern.total_cameras = 2;
    linear_pattern.fix_first_camera = false;
    linear_pattern.row_ptr = {0, 2, 4};
    linear_pattern.col_idx = {0, 1, 0, 1};
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) linear_pattern.slot_of[blockKey(i, j)] = 2 * i + j;
    }
    FastHGBPSystem linear_system;
    initializeFastHGBPSystemStructure(linear_system, linear_pattern);
    linear_system.rhs_blocks.resize(2);
    for (int i = 0; i < 2; ++i) {
        linear_system.ws.unary_lam[i] = S.block<9, 9>(9 * i, 9 * i);
        linear_system.rhs_blocks[i] = linear_rhs.segment<9>(9 * i);
    }
    linear_system.row_blocks[linear_system.edge_row_pos_i[0]] = S.topRightCorner<9, 9>();
    linear_system.row_blocks[linear_system.edge_row_pos_j[0]] = S.bottomLeftCorner<9, 9>();
    normalizeBAGBPCanonicalPairs(linear_system, 1);
    refreshPSDPairPreconditioner(linear_system, 1);
    resetSchurGBPEtaMessages(linear_system.ws, linear_rhs);
    Eigen::MatrixXd M(18, 18);
    for (int k = 0; k < 18; ++k) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(18), product;
        unit[k] = 1.0;
        fastHGBPMultiplyInto(linear_system, unit, product, 1);
        M.col(k) = product;
    }
    Eigen::MatrixXd P(18, 9);
    P.topRows<9>() = Mat9::Identity() / std::sqrt(2.0);
    P.bottomRows<9>() = Mat9::Identity() / std::sqrt(2.0);
    Eigen::MatrixXd Ac = P.transpose() * M * P;
    Ac = (0.5 * (Ac + Ac.transpose())).eval();
    Ac.diagonal().array() += 1e-12 * std::max(1.0, Ac.cwiseAbs().maxCoeff());
    const Eigen::LDLT<Eigen::MatrixXd> coarse_factor(Ac);

    for (int threads : {1, 2, 16}) {
        for (int cycles : {1, 5, 18}) {
            FastHGBPSystem candidate = linear_system;
            FastHGBPSystem reference = linear_system;
            Args linear_args;
            linear_args.mg_cycles = cycles;
            linear_args.gbp_threads = linear_args.build_threads = threads;
            FastHGBPCoarseWorkspace coarse;
            PackedGBPStats stats;
            const double tolerance = cycles == 18 ? 1e-10 : 0.0;
            const Eigen::VectorXd actual = solveFastHGBPWithFixedGroups(candidate,
                packed, scaling, damping, linear_args, {{0, 1}}, coarse,
                tolerance, 1e-6, stats);

            // Independent dense S/P^T M P algebra checks the selected additive
            // correction and FCG update, not an alternative production solver.
            auto& ws = reference.ws;
            int full_sweeps = 0;
            double variance_defect = 0.0;
            for (int sweep = 0; sweep < linear_args.gbp_full_sweeps; ++sweep) {
                if (sweep == 0) schurGBPFirstZeroMessageSweep(ws, 1.0, 1);
                else schurGBPSweep(ws, 1.0, 1);
                ++full_sweeps;
                double change2 = 0.0, norm2 = 0.0;
                for (size_t slot = 0; slot < ws.msg_lam.size(); ++slot) {
                    change2 += (ws.msg_lam[slot] - ws.next_msg_lam[slot]).squaredNorm();
                    norm2 += ws.msg_lam[slot].squaredNorm();
                }
                variance_defect = std::sqrt(change2 / std::max(1e-300, norm2));
                if (variance_defect <= 1e-6) break;
            }
            buildSchurGBPFixedEtaMaps(ws, 1);
            buildSchurGBPBeliefLambdaInverses(ws, 1);
            Eigen::VectorXd expected = Eigen::VectorXd::Zero(18), residual = linear_rhs;
            std::vector<Eigen::VectorXd> directions, products;
            std::vector<double> curvature;
            int reference_cycles = 0;
            for (int cycle = 0; cycle < cycles; ++cycle) {
                resetSchurGBPEtaMessages(ws, residual, 1);
                schurGBPFixedEtaSweeps(ws, 1.0, 3, 1);
                Eigen::VectorXd direction;
                schurGBPMeanInto(ws, 1, direction);
                direction.noalias() += P * coarse_factor.solve(P.transpose() * residual);
                Eigen::VectorXd product = S * direction;
                for (size_t j = 0; j < directions.size(); ++j) {
                    const double beta = -directions[j].dot(product) / curvature[j];
                    direction.noalias() += beta * directions[j];
                    product.noalias() += beta * products[j];
                }
                const double denominator = direction.dot(product);
                if (!(denominator > 0.0) || !std::isfinite(denominator)) return 40;
                const double alpha = residual.dot(direction) / denominator;
                expected.noalias() += alpha * direction;
                residual.noalias() -= alpha * product;
                directions.push_back(direction);
                products.push_back(product);
                curvature.push_back(denominator);
                ++reference_cycles;
                if (tolerance > 0.0 && residual.norm() <= tolerance * linear_rhs.norm()) break;
            }
            const double error = (actual - expected).norm() / std::max(1.0, expected.norm());
            const double actual_residual = (S * actual - linear_rhs).norm() / linear_rhs.norm();
            const double energy_decrease = linear_rhs.dot(actual) - 0.5 * actual.dot(S * actual);
            std::cout << "uniform_fcg_threads=" << threads << " budget=" << cycles
                      << " reference_error=" << error << " residual=" << actual_residual << '\n';
            if (!actual.allFinite() || error > 1e-8 || energy_decrease <= 0.0 ||
                std::abs(stats.model_decrease - energy_decrease) > 1e-9 ||
                std::abs(stats.relative_linear_residual - actual_residual) > 1e-9 ||
                stats.full_sweeps != full_sweeps || stats.variance_defect != variance_defect ||
                stats.cycles != reference_cycles || stats.eta_sweeps != 3 * stats.cycles ||
                stats.true_schur_products != stats.cycles ||
                stats.accepted_search_directions != stats.cycles) return 41;
            if (cycles == 18 && actual_residual > 1e-9) return 42;
        }
    }
    FastHGBPSystem zero_system = linear_system;
    zeroPlainEigenStorage(zero_system.rhs_blocks);
    resetSchurGBPEtaMessages(zero_system.ws, Eigen::VectorXd::Zero(18));
    Args zero_args;
    zero_args.gbp_threads = zero_args.build_threads = 1;
    FastHGBPCoarseWorkspace zero_coarse;
    PackedGBPStats zero_stats;
    const Eigen::VectorXd zero_solution = solveFastHGBPWithFixedGroups(zero_system,
        packed, scaling, damping, zero_args, {{0, 1}}, zero_coarse, 0.1, 1e-6, zero_stats);
    if (!zero_solution.allFinite() || zero_solution.squaredNorm() != 0.0 ||
        zero_stats.accepted_search_directions != 0 || zero_stats.cycles != 1 ||
        zero_stats.relative_linear_residual != 0.0 || zero_stats.model_decrease != 0.0) return 43;
    return 0;
}
