#include "ba_schur_reference.h"

int testDataflowSchur() {
    int cases = 0;
    double max_reference_error = 0.0, max_algebra_error = 0.0;
    for (int cameras : {1, 19}) {
        PackedRootLandmarks packed;
        packed.camera_rotation.resize(cameras, Mat3::Identity());
        packed.point_offset.push_back(0);
        const int points = cameras == 1 ? 7 : 1031;
        for (int p = 0; p < points; ++p) {
            const int degree = p == points - 1 ? 513 : p % 17;
            for (int k = 0; k < degree; ++k) {
                const int e = static_cast<int>(packed.camera_id.size());
                packed.camera_id.push_back((p + 7 * k) % cameras);
                Mat29 jc;
                Mat23 jl;
                for (int r = 0; r < 2; ++r) {
                    for (int c = 0; c < 9; ++c)
                        jc(r, c) = std::sin(0.13 * (e + 3 * r + c)) / (1.0 + c);
                    for (int c = 0; c < 3; ++c)
                        jl(r, c) = std::cos(0.17 * (e + r + 5 * c));
                }
                packed.Jc.push_back(jc);
                packed.Jl.push_back(jl);
            }
            packed.point_offset.push_back(static_cast<int>(packed.camera_id.size()));
        }
        packed.Hll_factor.resize(points);
        Eigen::VectorXd scaling = Eigen::VectorXd::LinSpaced(9 * cameras, 0.2, 2.0);
        for (double damping : {1e-8, 0.13, 10.0}) {
            for (int p = 0; p < points; ++p) {
                Mat3 b = Mat3::Zero();
                for (int e = packed.point_offset[p]; e < packed.point_offset[p + 1]; ++e)
                    b.noalias() += packed.Jl[e].transpose() * packed.Jl[e];
                if (!factorDampedPoint3(b, damping, packed.Hll_factor[p])) return 30;
            }
            for (bool zero : {false, true}) {
                Eigen::VectorXd x = Eigen::VectorXd::LinSpaced(9 * cameras, -0.7, 1.1);
                if (zero) x.setZero();
                Eigen::VectorXd scaled = x.cwiseProduct(scaling);
                Eigen::VectorXd expected = Eigen::VectorXd::Zero(x.size());
                for (int p = 0; p < points; ++p) {
                    Vec3 projected = Vec3::Zero();
                    for (int e = packed.point_offset[p]; e < packed.point_offset[p + 1]; ++e)
                        projected.noalias() += packed.Jl[e].transpose() *
                            (packed.Jc[e] * scaled.segment<9>(9 * packed.camera_id[e]));
                    const Vec3 eliminated = solvePoint3(packed.Hll_factor[p], projected);
                    for (int e = packed.point_offset[p]; e < packed.point_offset[p + 1]; ++e)
                        expected.segment<9>(9 * packed.camera_id[e]).noalias() += packed.Jc[e].transpose() *
                            (packed.Jc[e] * scaled.segment<9>(9 * packed.camera_id[e]) - packed.Jl[e] * eliminated);
                }
                expected.array() *= scaling.array();
                expected.noalias() += damping * x;
                for (int threads : {1, 2, 16, 1}) {
                    Eigen::VectorXd actual, reference;
                    packedRootExactSchurMultiplyIntoReference(packed, scaling, damping, x, threads, reference);
                    packedRootExactSchurMultiplyInto(packed, scaling, damping, x, threads, actual);
                    const double error = (actual - reference).norm() / std::max(1.0, reference.norm());
                    const double algebra_error = (actual - expected).norm() / std::max(1.0, expected.norm());
                    max_reference_error = std::max(max_reference_error, error);
                    max_algebra_error = std::max(max_algebra_error, algebra_error);
                    if (!actual.allFinite() || error > 1e-12 || algebra_error > 1e-12) return 31;
                    // Serial execution preserves the exact reference arithmetic order.
                    if (threads == 1 && (actual - reference).cwiseAbs().maxCoeff() != 0.0) return 32;
                    ++cases;
                }
            }
        }
    }
    std::cout << "dataflow_schur_cases=" << cases << " reference_error=" << max_reference_error
              << " algebra_error=" << max_algebra_error << '\n';
    return 0;
}

int testSharedCrossBlocks() {
    PackedBlockSchurPattern pattern;
    pattern.slot_pair_row_ptr = {0, 31};
    Mat93List cross(62);
    for (int e = 0; e < 62; ++e) {
        for (int i = 0; i < 27; ++i) cross[e].data()[i] = std::sin(0.07 * (2 * e + i));
    }
    const Mat93List duplicate = cross;
    RootHGBPBuildPlan plan;
    plan.retained_edge_count = 62;
    for (int e = 0; e < 62; ++e) plan.retained_edge_slot.push_back(e);
    plan.retained_edge_slot.push_back(-1);
    for (int k = 0; k < 31; ++k) {
        pattern.slot_pair_edge_a_global.push_back(2 * k);
        pattern.slot_pair_edge_b_global.push_back(2 * k + 1);
        pattern.slot_pair_transpose.push_back(k % 2);
    }
    for (int cap : {0, 1, 16, 40}) {
        const RootCrossCachePlan cache = makeRootCrossCachePlan(pattern, plan, cap);
        Mat93List compact(static_cast<size_t>(cache.slot_count));
        if (cache.slot_count != 2 * (cap > 0 ? std::min(31, cap) : 31) ||
            cache.edge_slot.back() != -1) return 34;
        for (int e = 0; e < 62; ++e) {
            if (cache.edge_slot[e] != cache.retained_slot[e]) return 35;
            if (cache.edge_slot[e] >= 0) compact[cache.edge_slot[e]] = cross[e];
        }
        for (bool rescale : {false, true}) {
            Mat9 a, ai, aj, b, bi, bj;
            accumulateRootPairBlock(pattern, cross, duplicate, 0, cap, rescale, a, ai, aj);
            accumulateRootPairBlock(pattern, cross, cross, 0, cap, rescale, b, bi, bj);
            if ((a - b).norm() != 0.0 || (ai - bi).norm() != 0.0 || (aj - bj).norm() != 0.0) return 33;
            accumulateRootPairBlock(pattern, compact, compact, 0, cap, rescale, b, bi, bj,
                &cache.retained_slot);
            if ((a - b).norm() != 0.0 || (ai - bi).norm() != 0.0 || (aj - bj).norm() != 0.0) return 36;
            accumulateRootPairBlock<false>(pattern, compact, compact, 0, cap, rescale, b, bi, bj,
                &cache.retained_slot);
            if ((a - b).norm() != 0.0 || bi.norm() != 0.0 || bj.norm() != 0.0) return 37;
        }
    }
    std::cout << "shared_cross_cases=8 compact_cache_cases=8 cross_only_cases=8 error=0\n";
    return 0;
}
