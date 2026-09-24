#pragma once

#include <immintrin.h>
#include "se3_schur_reciprocal.h"

namespace slam {

// Four independent right-hand sides per AVX lane group. Keep the scalar
// kernel's subtraction order and divisions; do not use reciprocals or FMA.
template<int batch, bool packed = false>
inline void solveSpd6CrossEtaLanes(const double* l, const double* cross,
                                  const double* eta, double* out) noexcept {
        const auto a = [l](int index) {
            const int row = index % 6, col = index / 6;
            return l[packed ? row * (row + 1) / 2 + col : index];
        };
        const auto load = [cross, eta](int row) {
            return batch == 0
                ? _mm256_setr_pd(cross[row], cross[6+row], cross[12+row], cross[18+row])
                : _mm256_setr_pd(cross[24+row], cross[30+row], eta[row], 0.0);
        };
        const auto subtract = [](__m256d value, double a, __m256d x) {
            return _mm256_sub_pd(value, _mm256_mul_pd(_mm256_set1_pd(a), x));
        };
        __m256d y[6];
        y[0] = _mm256_div_pd(load(0), _mm256_set1_pd(a(0)));
        y[1] = _mm256_div_pd(subtract(load(1), a(1), y[0]), _mm256_set1_pd(a(7)));
        y[2] = _mm256_div_pd(subtract(subtract(load(2), a(2), y[0]), a(8), y[1]), _mm256_set1_pd(a(14)));
        y[3] = _mm256_div_pd(subtract(subtract(subtract(load(3), a(3), y[0]), a(9), y[1]), a(15), y[2]), _mm256_set1_pd(a(21)));
        y[4] = _mm256_div_pd(subtract(subtract(subtract(subtract(load(4), a(4), y[0]), a(10), y[1]), a(16), y[2]), a(22), y[3]), _mm256_set1_pd(a(28)));
        y[5] = _mm256_div_pd(subtract(subtract(subtract(subtract(subtract(load(5), a(5), y[0]), a(11), y[1]), a(17), y[2]), a(23), y[3]), a(29), y[4]), _mm256_set1_pd(a(35)));
        __m256d x[6];
        x[5] = _mm256_div_pd(y[5], _mm256_set1_pd(a(35)));
        x[4] = _mm256_div_pd(subtract(y[4], a(29), x[5]), _mm256_set1_pd(a(28)));
        x[3] = _mm256_div_pd(subtract(subtract(y[3], a(22), x[4]), a(23), x[5]), _mm256_set1_pd(a(21)));
        x[2] = _mm256_div_pd(subtract(subtract(subtract(y[2], a(16), x[4]), a(17), x[5]), a(15), x[3]), _mm256_set1_pd(a(14)));
        x[1] = _mm256_div_pd(subtract(subtract(subtract(subtract(y[1], a(10), x[4]), a(11), x[5]), a(9), x[3]), a(8), x[2]), _mm256_set1_pd(a(7)));
        x[0] = _mm256_div_pd(subtract(subtract(subtract(subtract(subtract(y[0], a(4), x[4]), a(5), x[5]), a(3), x[3]), a(2), x[2]), a(1), x[1]), _mm256_set1_pd(a(0)));
        for (int row = 0; row < 6; ++row) {
            alignas(32) double lanes[4];
            _mm256_store_pd(lanes, x[row]);
            for (int lane = 0; lane < (batch == 0 ? 4 : 3); ++lane)
                out[6*(4*batch+lane)+row] = lanes[lane];
        }
}

inline void solveSpd6CrossEtaBatched(const double* l, const double* cross,
                                    const double* eta, double* out) noexcept {
    solveSpd6CrossEtaLanes<0>(l,cross,eta,out);
    solveSpd6CrossEtaLanes<1>(l,cross,eta,out);
}

inline void solveSpd6CrossEtaPackedBatched(const double* l, const double* cross,
                                          const double* eta, double* out) noexcept {
    solveSpd6CrossEtaPackedReciprocal(l,cross,eta,out);
}

}  // namespace slam
