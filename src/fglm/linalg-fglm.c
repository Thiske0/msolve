/* This file is part of msolve.
 *
 * msolve is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 2 of the License, or
 * (at your option) any later version.
 *
 * msolve is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with msolve.  If not, see <https://www.gnu.org/licenses/>
 *
 * Authors:
 * Jérémy Berthomieu
 * Christian Eder
 * Vincent Neiger
 * Mohab Safey El Din */

#include <stdint.h>
#include <inttypes.h>
#include <flint/nmod.h>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

// for dot product, simultaneous dot products, and matrix-vector products,
// vectorized functions below accumulate 8 terms:
//  -> modulus <= DOT2_ACC8_MAX_MODULUS   (about 2**30.5, slightly less)
//  -> len <= DOT2_ACC8_MAX_LEN           (about 2**28.5)
// (see bottom of flint/src/nmod_vec/dot.c for more details)
#define DOT2_ACC8_MAX_MODULUS UWORD(1515531528)
#define DOT2_ACC8_MAX_LEN UWORD(380368697)

// parameters for splitting
#define __DOT_SPLIT_BITS 56
#define __DOT_SPLIT_MASK 72057594037927935UL // (1UL << __DOT_SPLIT_BITS) - 1

/*--------------------------------------*/
/* non-vectorized matrix vector product */
/*--------------------------------------*/
// FIXME what are the constraints (prime bitsize, length, ..) for this non-vectorized version?

static inline void non_avx_matrix_vector_product(uint32_t* vec_res, const uint32_t* mat,
                                                 const uint32_t* vec, const uint32_t ncols,
                                                 const uint32_t nrows, const uint32_t PRIME,
                                                 const uint32_t RED_32, const uint32_t RED_64,
                                                 md_t *st)
{
    uint32_t i, j;
    int64_t prod1, prod2, prod3, prod4;
    const int64_t modsquare = (int64_t)PRIME*PRIME;

    j = 0;
    if (nrows >= 4) {
        for (j = 0; j < nrows-3; j += 4) {
            i = 0;
            prod1 =  0;
            prod2 =  0;
            prod3 =  0;
            prod4 =  0;
            if (ncols >= 8) {
                while (i < ncols-7) {
                    prod1 -=  (int64_t)mat[j*ncols+i] * vec[i];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i] * vec[i];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i] * vec[i];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i] * vec[i];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+1] * vec[i+1];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+1] * vec[i+1];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+1] * vec[i+1];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+1] * vec[i+1];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+2] * vec[i+2];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+2] * vec[i+2];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+2] * vec[i+2];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+2] * vec[i+2];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+3] * vec[i+3];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+3] * vec[i+3];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+3] * vec[i+3];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+3] * vec[i+3];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+4] * vec[i+4];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+4] * vec[i+4];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+4] * vec[i+4];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+4] * vec[i+4];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+5] * vec[i+5];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+5] * vec[i+5];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+5] * vec[i+5];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+5] * vec[i+5];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+6] * vec[i+6];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+6] * vec[i+6];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+6] * vec[i+6];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+6] * vec[i+6];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    prod1 -=  (int64_t)mat[j*ncols+i+7] * vec[i+7];
                    prod2 -=  (int64_t)mat[(j+1)*ncols+i+7] * vec[i+7];
                    prod3 -=  (int64_t)mat[(j+2)*ncols+i+7] * vec[i+7];
                    prod4 -=  (int64_t)mat[(j+3)*ncols+i+7] * vec[i+7];
                    prod1 +=  ((prod1 >> 63)) & modsquare;
                    prod2 +=  ((prod2 >> 63)) & modsquare;
                    prod3 +=  ((prod3 >> 63)) & modsquare;
                    prod4 +=  ((prod4 >> 63)) & modsquare;
                    i     +=  8;
                }
            }
            while (i < ncols) {
                prod1 -=  (int64_t)mat[j*ncols+i] * vec[i];
                prod2 -=  (int64_t)mat[(j+1)*ncols+i] * vec[i];
                prod3 -=  (int64_t)mat[(j+2)*ncols+i] * vec[i];
                prod4 -=  (int64_t)mat[(j+3)*ncols+i] * vec[i];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod2 +=  ((prod2 >> 63)) & modsquare;
                prod3 +=  ((prod3 >> 63)) & modsquare;
                prod4 +=  ((prod4 >> 63)) & modsquare;
                i     +=  1;
            }
            /* ensure prod being positive */
            prod1 =   -prod1;
            prod1 +=  (prod1 >> 63) & modsquare;
            prod2 =   -prod2;
            prod2 +=  (prod2 >> 63) & modsquare;
            prod3 =   -prod3;
            prod3 +=  (prod3 >> 63) & modsquare;
            prod4 =   -prod4;
            prod4 +=  (prod4 >> 63) & modsquare;
            vec_res[j]    = (uint32_t)(prod1 % PRIME);
            vec_res[j+1]  = (uint32_t)(prod2 % PRIME);
            vec_res[j+2]  = (uint32_t)(prod3 % PRIME);
            vec_res[j+3]  = (uint32_t)(prod4 % PRIME);
        }
    }
    for (; j < nrows; ++j) {
        i = 0;
        prod1 =  0;
        if (ncols >= 8) {
            while (i < ncols-7) {
                prod1 -=  (int64_t)mat[j*ncols+i] * vec[i];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+1] * vec[i+1];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+2] * vec[i+2];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+3] * vec[i+3];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+4] * vec[i+4];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+5] * vec[i+5];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+6] * vec[i+6];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 -=  (int64_t)mat[j*ncols+i+7] * vec[i+7];
                prod1 +=  ((prod1 >> 63)) & modsquare;
                prod1 +=  ((prod1 >> 63)) & modsquare;
                i     +=  8;
            }
        }
        while (i < ncols) {
            prod1 -=  (int64_t)mat[j*ncols+i] * vec[i];
            prod1 +=  ((prod1 >> 63)) & modsquare;
            i     +=  1;
        }
        /* ensure prod being positive */
        prod1 =   -prod1;
        prod1 +=  (prod1 >> 63) & modsquare;
        vec_res[j]    = (uint32_t)(prod1 % PRIME);
    }
}

/*-----------------------------------------*/
/* vectorized (AVX2) matrix vector product */
/*-----------------------------------------*/

#ifdef HAVE_AVX2


float _nmod32_vec_dot_split_avx2(const float * vec1_alligned, const float * vec2, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    // accumulator
    __m256d acc_0 = _mm256_setzero_pd();
    __m256d acc_1 = _mm256_setzero_pd();
    __m256d acc_2 = _mm256_setzero_pd();
    __m256d acc_3 = _mm256_setzero_pd();

    int64_t i = 0;
    // process blocks of 4 floats at a time
    for (; i + 3 < len; i += 16)
    {   
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v1_23 = _mm256_load_ps(vec1_alligned + i + 8);
        __m256d v1_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 0));
        __m256d v1_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 1));
        __m256 v2_01 = _mm256_loadu_ps(vec2 + i);
        __m256d v2_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_01, 0));
        __m256d v2_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_01, 1));
        __m256 v2_23 = _mm256_loadu_ps(vec2 + i + 8);
        __m256d v2_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_23, 0));
        __m256d v2_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_23, 1));

        // acc += v1 * v2
        acc_0 = _mm256_fmadd_pd(v1_0, v2_0, acc_0);
        acc_1 = _mm256_fmadd_pd(v1_1, v2_1, acc_1);
        acc_2 = _mm256_fmadd_pd(v1_2, v2_2, acc_2);
        acc_3 = _mm256_fmadd_pd(v1_3, v2_3, acc_3);
    }
    for (; i < len; i+=8) {
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v2_01 = _mm256_loadu_ps(vec2 + i);
        __m256d v2_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_01, 0));
        __m256d v2_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_01, 1));
        acc_0 = _mm256_fmadd_pd(v1_0, v2_0, acc_0);
        acc_1 = _mm256_fmadd_pd(v1_1, v2_1, acc_1);
    }
    // combine acc_0, acc_1, acc_2, acc_3
    acc_0 = _mm256_add_pd(acc_0, acc_1);
    acc_2 = _mm256_add_pd(acc_2, acc_3);
    acc_0 = _mm256_add_pd(acc_0, acc_2);

    // horizontal sum of acc
    double tmp[4];
    _mm256_storeu_pd(tmp, acc_0);
    double res = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // remaining elements
    for (; i < len; i++)
        res += vec1_alligned[i] * (double) vec2[i];

    // modulo reduction
    res = fmod(res, (double)mod.n);

    return res;
}

void _nmod32_vec_dot2_split_avx2(float * res, const float * vec1_alligned, const float * vec2_0, const float * vec2_1, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    // accumulator
    __m256d acc_0_0 = _mm256_setzero_pd();
    __m256d acc_0_1 = _mm256_setzero_pd();
    __m256d acc_0_2 = _mm256_setzero_pd();
    __m256d acc_0_3 = _mm256_setzero_pd();
    __m256d acc_1_0 = _mm256_setzero_pd();
    __m256d acc_1_1 = _mm256_setzero_pd();
    __m256d acc_1_2 = _mm256_setzero_pd();
    __m256d acc_1_3 = _mm256_setzero_pd();

    int64_t i = 0;
    // process blocks of 4 floats at a time
    for (; i + 3 < len; i += 16)
    {
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v1_23 = _mm256_load_ps(vec1_alligned + i + 8);
        __m256d v1_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 0));
        __m256d v1_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 1));
        __m256 v2_0_01 = _mm256_loadu_ps(vec2_0 + i);
        __m256d v2_0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 0));
        __m256d v2_0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 1));
        __m256 v2_0_23 = _mm256_loadu_ps(vec2_0 + i + 8);
        __m256d v2_0_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_23, 0));
        __m256d v2_0_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_23, 1));
        __m256 v2_1_01 = _mm256_loadu_ps(vec2_1 + i);
        __m256d v2_1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 0));
        __m256d v2_1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 1));
        __m256 v2_1_23 = _mm256_loadu_ps(vec2_1 + i + 8);
        __m256d v2_1_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_23, 0));
        __m256d v2_1_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_23, 1));

        // acc += v1 * v2
        acc_0_0 = _mm256_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm256_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_0_2 = _mm256_fmadd_pd(v1_2, v2_0_2, acc_0_2);
        acc_0_3 = _mm256_fmadd_pd(v1_3, v2_0_3, acc_0_3);
        acc_1_0 = _mm256_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm256_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_1_2 = _mm256_fmadd_pd(v1_2, v2_1_2, acc_1_2);
        acc_1_3 = _mm256_fmadd_pd(v1_3, v2_1_3, acc_1_3);

    }
    for (; i < len; i+=8) {
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v2_0_01 = _mm256_loadu_ps(vec2_0 + i);
        __m256d v2_0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 0));
        __m256d v2_0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 1));
        __m256 v2_1_01 = _mm256_loadu_ps(vec2_1 + i);
        __m256d v2_1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 0));
        __m256d v2_1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 1));
        acc_0_0 = _mm256_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm256_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_1_0 = _mm256_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm256_fmadd_pd(v1_1, v2_1_1, acc_1_1);
    }
    // combine acc_0, acc_1, acc_2, acc_3
    acc_0_0 = _mm256_add_pd(acc_0_0, acc_0_1);
    acc_0_2 = _mm256_add_pd(acc_0_2, acc_0_3);
    acc_0_0 = _mm256_add_pd(acc_0_0, acc_0_2);
    acc_1_0 = _mm256_add_pd(acc_1_0, acc_1_1);
    acc_1_2 = _mm256_add_pd(acc_1_2, acc_1_3);
    acc_1_0 = _mm256_add_pd(acc_1_0, acc_1_2);

    // horizontal sum of acc
    double tmp[4];
    _mm256_storeu_pd(tmp, acc_0_0);
    double res_0 = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, acc_1_0);
    double res_1 = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    // remaining elements
    for (; i < len; i++) {
        res_0 += vec1_alligned[i] * (double) vec2_0[i];
        res_1 += vec1_alligned[i] * (double) vec2_1[i];
    }

    // modulo reduction
    res[0] = fmod(res_0, (double)mod.n);
    res[1] = fmod(res_1, (double)mod.n);
}

void _nmod32_vec_dot3_split_avx2(float * res, const float * vec1_alligned, const float * vec2_0, const float * vec2_1, const float * vec2_2, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    // accumulator
    __m256d acc_0_0 = _mm256_setzero_pd();
    __m256d acc_0_1 = _mm256_setzero_pd();
    __m256d acc_0_2 = _mm256_setzero_pd();
    __m256d acc_0_3 = _mm256_setzero_pd();
    __m256d acc_1_0 = _mm256_setzero_pd();
    __m256d acc_1_1 = _mm256_setzero_pd();
    __m256d acc_1_2 = _mm256_setzero_pd();
    __m256d acc_1_3 = _mm256_setzero_pd();
    __m256d acc_2_0 = _mm256_setzero_pd();
    __m256d acc_2_1 = _mm256_setzero_pd();
    __m256d acc_2_2 = _mm256_setzero_pd();
    __m256d acc_2_3 = _mm256_setzero_pd();

    int64_t i = 0;
    // process blocks of 8 floats at a time
    for (; i + 3 < len; i += 16)
    {
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v1_23 = _mm256_load_ps(vec1_alligned + i + 8);
        __m256d v1_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 0));
        __m256d v1_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_23, 1));
        __m256 v2_0_01 = _mm256_loadu_ps(vec2_0 + i);
        __m256d v2_0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 0));
        __m256d v2_0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 1));
        __m256 v2_0_23 = _mm256_loadu_ps(vec2_0 + i + 8);
        __m256d v2_0_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_23, 0));
        __m256d v2_0_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_23, 1));
        __m256 v2_1_01 = _mm256_loadu_ps(vec2_1 + i);
        __m256d v2_1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 0));
        __m256d v2_1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 1));
        __m256 v2_1_23 = _mm256_loadu_ps(vec2_1 + i + 8);
        __m256d v2_1_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_23, 0));
        __m256d v2_1_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_23, 1));
        __m256 v2_2_01 = _mm256_loadu_ps(vec2_2 + i);
        __m256d v2_2_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_01, 0));
        __m256d v2_2_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_01, 1));
        __m256 v2_2_23 = _mm256_loadu_ps(vec2_2 + i + 8);
        __m256d v2_2_2 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_23, 0));
        __m256d v2_2_3 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_23, 1));

        // acc += v1 * v2
        acc_0_0 = _mm256_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm256_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_0_2 = _mm256_fmadd_pd(v1_2, v2_0_2, acc_0_2);
        acc_0_3 = _mm256_fmadd_pd(v1_3, v2_0_3, acc_0_3);
        acc_1_0 = _mm256_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm256_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_1_2 = _mm256_fmadd_pd(v1_2, v2_1_2, acc_1_2);
        acc_1_3 = _mm256_fmadd_pd(v1_3, v2_1_3, acc_1_3);
        acc_2_0 = _mm256_fmadd_pd(v1_0, v2_2_0, acc_2_0);
        acc_2_1 = _mm256_fmadd_pd(v1_1, v2_2_1, acc_2_1);
        acc_2_2 = _mm256_fmadd_pd(v1_2, v2_2_2, acc_2_2);
        acc_2_3 = _mm256_fmadd_pd(v1_3, v2_2_3, acc_2_3);

    }
    for (; i < len; i+=8) {
        __m256 v1_01 = _mm256_load_ps(vec1_alligned + i);
        __m256d v1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 0));
        __m256d v1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v1_01, 1));
        __m256 v2_0_01 = _mm256_loadu_ps(vec2_0 + i);
        __m256d v2_0_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 0));
        __m256d v2_0_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_0_01, 1));
        __m256 v2_1_01 = _mm256_loadu_ps(vec2_1 + i);
        __m256d v2_1_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 0));
        __m256d v2_1_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_1_01, 1));
        __m256 v2_2_01 = _mm256_loadu_ps(vec2_2 + i);
        __m256d v2_2_0 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_01, 0));
        __m256d v2_2_1 = _mm256_cvtps_pd(_mm256_extractf128_ps(v2_2_01, 1));
        acc_0_0 = _mm256_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm256_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_1_0 = _mm256_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm256_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_2_0 = _mm256_fmadd_pd(v1_0, v2_2_0, acc_2_0);
        acc_2_1 = _mm256_fmadd_pd(v1_1, v2_2_1, acc_2_1);
    }
    // combine acc_0, acc_1, acc_2, acc_3
    acc_0_0 = _mm256_add_pd(acc_0_0, acc_0_1);
    acc_0_2 = _mm256_add_pd(acc_0_2, acc_0_3);
    acc_0_0 = _mm256_add_pd(acc_0_0, acc_0_2);
    acc_1_0 = _mm256_add_pd(acc_1_0, acc_1_1);
    acc_1_2 = _mm256_add_pd(acc_1_2, acc_1_3);
    acc_1_0 = _mm256_add_pd(acc_1_0, acc_1_2);
    acc_2_0 = _mm256_add_pd(acc_2_0, acc_2_1);
    acc_2_2 = _mm256_add_pd(acc_2_2, acc_2_3);
    acc_2_0 = _mm256_add_pd(acc_2_0, acc_2_2);

    // horizontal sum of acc
    double tmp[4];
    _mm256_storeu_pd(tmp, acc_0_0);
    double res_0 = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, acc_1_0);
    double res_1 = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    _mm256_storeu_pd(tmp, acc_2_0);
    double res_2 = tmp[0] + tmp[1] + tmp[2] + tmp[3];


    // remaining elements
    for (; i < len; i++) {
        res_0 += vec1_alligned[i] * (double) vec2_0[i];
        res_1 += vec1_alligned[i] * (double) vec2_1[i];
        res_2 += vec1_alligned[i] * (double) vec2_2[i];
    }

    // modulo reduction
    res[0] = fmod(res_0, (double)mod.n);
    res[1] = fmod(res_1, (double)mod.n);
    res[2] = fmod(res_2, (double)mod.n);

}

static inline void _avx2_matrix_vector_product(float * vec_res,
                                               const float * mat,
                                               const float * vec,
                                               const uint32_t * dst,
                                               const uint32_t ncols,
                                               const uint32_t nrows,
                                               const nmod_t mod,
                                               const uint64_t pow2_precomp,
                                               md_t *st)
{
    slong i = 0;

    if (nrows > 2)
    {
#pragma omp parallel for num_threads (st->nthrds) lastprivate(i)
        for (i=0; i < nrows-2; i+=3)
        {
            int64_t len = ncols - MIN(dst[i], MIN(dst[i+1], dst[i+2]));
            _nmod32_vec_dot3_split_avx2(vec_res+i, vec,
                                        mat + i*ncols,
                                        mat + (i+1)*ncols,
                                        mat + (i+2)*ncols,
                                        len, mod, pow2_precomp);
        }
    }

    if (nrows - i == 2)
    {
        int64_t len = ncols - MIN(dst[i], dst[i+1]);
        _nmod32_vec_dot2_split_avx2(vec_res+i,
                                    vec, mat + i*ncols, mat + (i+1)*ncols,
                                    len, mod, pow2_precomp);
    }
    else if (nrows - i == 1)
        vec_res[i] = _nmod32_vec_dot_split_avx2(vec, mat + i*ncols, ncols - dst[i], mod, pow2_precomp);
}
#endif

/*-------------------------------------------*/
/* vectorized (AVX512) matrix vector product */
/*-------------------------------------------*/

#ifdef HAVE_AVX512_F

// avx512 horizontal sum
FLINT_FORCE_INLINE uint64_t _mm512_hsum(__m512i a)
{
    return _mm512_reduce_add_epi64(a);
}

float _nmod32_vec_dot_split_avx512(const float * vec1_aligned, const float * vec2, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    __m512d acc_0 = _mm512_setzero_pd();
    __m512d acc_1 = _mm512_setzero_pd();
    __m512d acc_2 = _mm512_setzero_pd();
    __m512d acc_3 = _mm512_setzero_pd();

    int64_t i = 0;

    for (; i + 31 < len; i += 32)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));
        __m512 v1_23 = _mm512_load_ps(vec1_aligned + i + 16);
        __m512d v1_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 0));
        __m512d v1_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 1));

        __m512 v2_01 = _mm512_loadu_ps(vec2 + i);
        __m512d v2_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_01, 0));
        __m512d v2_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_01, 1));
        __m512 v2_23 = _mm512_loadu_ps(vec2 + i + 16);
        __m512d v2_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_23, 0));
        __m512d v2_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_23, 1));

        acc_0 = _mm512_fmadd_pd(v1_0, v2_0, acc_0);
        acc_1 = _mm512_fmadd_pd(v1_1, v2_1, acc_1);
        acc_2 = _mm512_fmadd_pd(v1_2, v2_2, acc_2);
        acc_3 = _mm512_fmadd_pd(v1_3, v2_3, acc_3);
    }

    for (; i + 15 < len; i += 16)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));

        __m512 v2_01 = _mm512_loadu_ps(vec2 + i);
        __m512d v2_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_01, 0));
        __m512d v2_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_01, 1));

        acc_0 = _mm512_fmadd_pd(v1_0, v2_0, acc_0);
        acc_1 = _mm512_fmadd_pd(v1_1, v2_1, acc_1);
    }

    acc_0 = _mm512_add_pd(acc_0, acc_1);
    acc_2 = _mm512_add_pd(acc_2, acc_3);
    acc_0 = _mm512_add_pd(acc_0, acc_2);

    double res = _mm512_reduce_add_pd(acc_0);

    for (; i < len; i++)
        res += vec1_aligned[i] * (double)vec2[i];

    res = fmod(res, (double)mod.n);

    return res;
}void _nmod32_vec_dot2_split_avx512(float * res, const float * vec1_aligned, const float * vec2_0, const float * vec2_1, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    __m512d acc_0_0 = _mm512_setzero_pd();
    __m512d acc_0_1 = _mm512_setzero_pd();
    __m512d acc_0_2 = _mm512_setzero_pd();
    __m512d acc_0_3 = _mm512_setzero_pd();
    __m512d acc_1_0 = _mm512_setzero_pd();
    __m512d acc_1_1 = _mm512_setzero_pd();
    __m512d acc_1_2 = _mm512_setzero_pd();
    __m512d acc_1_3 = _mm512_setzero_pd();

    int64_t i = 0;

    for (; i + 31 < len; i += 32)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));
        __m512 v1_23 = _mm512_load_ps(vec1_aligned + i + 16);
        __m512d v1_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 0));
        __m512d v1_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 1));

        __m512 v2_0_01 = _mm512_loadu_ps(vec2_0 + i);
        __m512d v2_0_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 0));
        __m512d v2_0_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 1));
        __m512 v2_0_23 = _mm512_loadu_ps(vec2_0 + i + 16);
        __m512d v2_0_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_23, 0));
        __m512d v2_0_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_23, 1));

        __m512 v2_1_01 = _mm512_loadu_ps(vec2_1 + i);
        __m512d v2_1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 0));
        __m512d v2_1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 1));
        __m512 v2_1_23 = _mm512_loadu_ps(vec2_1 + i + 16);
        __m512d v2_1_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_23, 0));
        __m512d v2_1_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_23, 1));

        acc_0_0 = _mm512_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm512_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_0_2 = _mm512_fmadd_pd(v1_2, v2_0_2, acc_0_2);
        acc_0_3 = _mm512_fmadd_pd(v1_3, v2_0_3, acc_0_3);
        acc_1_0 = _mm512_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm512_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_1_2 = _mm512_fmadd_pd(v1_2, v2_1_2, acc_1_2);
        acc_1_3 = _mm512_fmadd_pd(v1_3, v2_1_3, acc_1_3);
    }

    for (; i + 15 < len; i += 16)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));

        __m512 v2_0_01 = _mm512_loadu_ps(vec2_0 + i);
        __m512d v2_0_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 0));
        __m512d v2_0_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 1));

        __m512 v2_1_01 = _mm512_loadu_ps(vec2_1 + i);
        __m512d v2_1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 0));
        __m512d v2_1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 1));

        acc_0_0 = _mm512_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm512_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_1_0 = _mm512_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm512_fmadd_pd(v1_1, v2_1_1, acc_1_1);
    }

    acc_0_0 = _mm512_add_pd(acc_0_0, acc_0_1);
    acc_0_2 = _mm512_add_pd(acc_0_2, acc_0_3);
    acc_0_0 = _mm512_add_pd(acc_0_0, acc_0_2);
    acc_1_0 = _mm512_add_pd(acc_1_0, acc_1_1);
    acc_1_2 = _mm512_add_pd(acc_1_2, acc_1_3);
    acc_1_0 = _mm512_add_pd(acc_1_0, acc_1_2);

    double res_0 = _mm512_reduce_add_pd(acc_0_0);
    double res_1 = _mm512_reduce_add_pd(acc_1_0);

    for (; i < len; i++) {
        res_0 += vec1_aligned[i] * (double)vec2_0[i];
        res_1 += vec1_aligned[i] * (double)vec2_1[i];
    }

    res[0] = fmod(res_0, (double)mod.n);
    res[1] = fmod(res_1, (double)mod.n);
}

void _nmod32_vec_dot3_split_avx512(float * res, const float * vec1_aligned, const float * vec2_0, const float * vec2_1, const float * vec2_2, int64_t len,
                                    nmod_t mod, uint64_t pow2_precomp)
{
    __m512d acc_0_0 = _mm512_setzero_pd();
    __m512d acc_0_1 = _mm512_setzero_pd();
    __m512d acc_0_2 = _mm512_setzero_pd();
    __m512d acc_0_3 = _mm512_setzero_pd();
    __m512d acc_1_0 = _mm512_setzero_pd();
    __m512d acc_1_1 = _mm512_setzero_pd();
    __m512d acc_1_2 = _mm512_setzero_pd();
    __m512d acc_1_3 = _mm512_setzero_pd();
    __m512d acc_2_0 = _mm512_setzero_pd();
    __m512d acc_2_1 = _mm512_setzero_pd();
    __m512d acc_2_2 = _mm512_setzero_pd();
    __m512d acc_2_3 = _mm512_setzero_pd();

    int64_t i = 0;

    for (; i + 31 < len; i += 32)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));
        __m512 v1_23 = _mm512_load_ps(vec1_aligned + i + 16);
        __m512d v1_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 0));
        __m512d v1_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_23, 1));

        __m512 v2_0_01 = _mm512_loadu_ps(vec2_0 + i);
        __m512d v2_0_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 0));
        __m512d v2_0_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 1));
        __m512 v2_0_23 = _mm512_loadu_ps(vec2_0 + i + 16);
        __m512d v2_0_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_23, 0));
        __m512d v2_0_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_23, 1));

        __m512 v2_1_01 = _mm512_loadu_ps(vec2_1 + i);
        __m512d v2_1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 0));
        __m512d v2_1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 1));
        __m512 v2_1_23 = _mm512_loadu_ps(vec2_1 + i + 16);
        __m512d v2_1_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_23, 0));
        __m512d v2_1_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_23, 1));

        __m512 v2_2_01 = _mm512_loadu_ps(vec2_2 + i);
        __m512d v2_2_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_01, 0));
        __m512d v2_2_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_01, 1));
        __m512 v2_2_23 = _mm512_loadu_ps(vec2_2 + i + 16);
        __m512d v2_2_2 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_23, 0));
        __m512d v2_2_3 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_23, 1));

        acc_0_0 = _mm512_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm512_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_0_2 = _mm512_fmadd_pd(v1_2, v2_0_2, acc_0_2);
        acc_0_3 = _mm512_fmadd_pd(v1_3, v2_0_3, acc_0_3);
        acc_1_0 = _mm512_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm512_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_1_2 = _mm512_fmadd_pd(v1_2, v2_1_2, acc_1_2);
        acc_1_3 = _mm512_fmadd_pd(v1_3, v2_1_3, acc_1_3);
        acc_2_0 = _mm512_fmadd_pd(v1_0, v2_2_0, acc_2_0);
        acc_2_1 = _mm512_fmadd_pd(v1_1, v2_2_1, acc_2_1);
        acc_2_2 = _mm512_fmadd_pd(v1_2, v2_2_2, acc_2_2);
        acc_2_3 = _mm512_fmadd_pd(v1_3, v2_2_3, acc_2_3);
    }

    for (; i + 15 < len; i += 16)
    {
        __m512 v1_01 = _mm512_load_ps(vec1_aligned + i);
        __m512d v1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 0));
        __m512d v1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v1_01, 1));

        __m512 v2_0_01 = _mm512_loadu_ps(vec2_0 + i);
        __m512d v2_0_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 0));
        __m512d v2_0_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_0_01, 1));

        __m512 v2_1_01 = _mm512_loadu_ps(vec2_1 + i);
        __m512d v2_1_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 0));
        __m512d v2_1_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_1_01, 1));

        __m512 v2_2_01 = _mm512_loadu_ps(vec2_2 + i);
        __m512d v2_2_0 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_01, 0));
        __m512d v2_2_1 = _mm512_cvtps_pd(_mm512_extractf32x8_ps(v2_2_01, 1));

        acc_0_0 = _mm512_fmadd_pd(v1_0, v2_0_0, acc_0_0);
        acc_0_1 = _mm512_fmadd_pd(v1_1, v2_0_1, acc_0_1);
        acc_1_0 = _mm512_fmadd_pd(v1_0, v2_1_0, acc_1_0);
        acc_1_1 = _mm512_fmadd_pd(v1_1, v2_1_1, acc_1_1);
        acc_2_0 = _mm512_fmadd_pd(v1_0, v2_2_0, acc_2_0);
        acc_2_1 = _mm512_fmadd_pd(v1_1, v2_2_1, acc_2_1);
    }

    acc_0_0 = _mm512_add_pd(acc_0_0, acc_0_1);
    acc_0_2 = _mm512_add_pd(acc_0_2, acc_0_3);
    acc_0_0 = _mm512_add_pd(acc_0_0, acc_0_2);
    acc_1_0 = _mm512_add_pd(acc_1_0, acc_1_1);
    acc_1_2 = _mm512_add_pd(acc_1_2, acc_1_3);
    acc_1_0 = _mm512_add_pd(acc_1_0, acc_1_2);
    acc_2_0 = _mm512_add_pd(acc_2_0, acc_2_1);
    acc_2_2 = _mm512_add_pd(acc_2_2, acc_2_3);
    acc_2_0 = _mm512_add_pd(acc_2_0, acc_2_2);

    double res_0 = _mm512_reduce_add_pd(acc_0_0);
    double res_1 = _mm512_reduce_add_pd(acc_1_0);
    double res_2 = _mm512_reduce_add_pd(acc_2_0);

    for (; i < len; i++) {
        res_0 += vec1_aligned[i] * (double)vec2_0[i];
        res_1 += vec1_aligned[i] * (double)vec2_1[i];
        res_2 += vec1_aligned[i] * (double)vec2_2[i];
    }

    res[0] = fmod(res_0, (double)mod.n);
    res[1] = fmod(res_1, (double)mod.n);
    res[2] = fmod(res_2, (double)mod.n);
}

static inline void _avx512_matrix_vector_product(float * vec_res,
                                                 const float * mat,
                                                 const float * vec,
                                                 const uint32_t * dst,
                                                 const uint32_t ncols,
                                                 const uint32_t nrows,
                                                 const nmod_t mod,
                                                 const uint64_t pow2_precomp,
                                                 md_t *st)
{
    slong i = 0;

    if (nrows > 2)
    {
#pragma omp parallel for num_threads (st->nthrds) lastprivate(i)
        for (i = 0; i < nrows-2; i+=3)
        {
            int64_t len = ncols - MIN(dst[i], MIN(dst[i+1], dst[i+2]));
            _nmod32_vec_dot3_split_avx512(vec_res+i, vec,
                                          mat + i*ncols,
                                          mat + (i+1)*ncols,
                                          mat + (i+2)*ncols,
                                          len, mod, pow2_precomp);
        }
    }

    if (nrows - i == 2)
    {
        int64_t len = ncols - MIN(dst[i], dst[i+1]);
        _nmod32_vec_dot2_split_avx512(vec_res+i,
                                      vec, mat + i*ncols, mat + (i+1)*ncols,
                                      len, mod, pow2_precomp);
    }
    else if (nrows - i == 1)
        vec_res[i] = _nmod32_vec_dot_split_avx512(vec, mat + i*ncols, ncols - dst[i], mod, pow2_precomp);
}

#endif
