// src/kernels/cpu/dot_x4.cpp - see the header.
//
// The arithmetic is ggml-cpu's AVX2 branches (ggml-cpu/arch/x86/quants.c at the llama.cpp the engines pin,
// 3cf03257): vec_dot_q5_0_q8_0's nibbles-and-high-bits build with mul_sum_i8_pairs_float, and
// vec_dot_iq2_xs_q8_K's grid/signs/scales walk.  A block's i32 sums are exact, so a 512-bit register's two
// independent halves or any grouping of them gives the same integers; the float part is ggml's own sequence -
// the same per-block fmadd into an 8-lane acc, hsum_float_8's tree at the end - and it stays 256-bit.
#include "strata/kernels/cpu/dot_x4.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#include <cstring>

#if defined(__AVX2__)
namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }

// ---- ggml-cpu's AVX2 helpers, verbatim in what they compute (quants.c at the pin) ----

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

inline __m256i bytes_from_bits_32(const uint8_t* x) {
    uint32_t x32;
    memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202, 0x0101010101010101,
                                                0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

inline __m256i bytes_from_nibbles_32(const uint8_t* rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i*) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    // ggml's sum_i16_pairs_float: the madd's multiplier is set1_epi16(1) - every 16-bit lane a one - so the
    // i32 lane gets BOTH 16-bit halves; with set1_epi32(1) the high half would read as zero and drop a pair
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, _mm256_maddubs_epi16(ax, sy));
    return _mm256_cvtepi32_ps(summed_pairs);
}

// multiply int8_t, add results pairwise twice and return as float vector
inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return mul_sum_us8_pairs_float(ax, sy);
}

inline __m128i get_scale_shuffle(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
         2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
         4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
         6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
         8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10,10,10,10,10,10,10,10, 11,11,11,11,11,11,11,11,
        12,12,12,12,12,12,12,12, 13,13,13,13,13,13,13,13,
        14,14,14,14,14,14,14,14, 15,15,15,15,15,15,15,15
    };
    return _mm_loadu_si128((const __m128i*) k_shuffle + i);
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

// ---- GGML_TYPE_Q5_0: the activation's 32 bytes are loaded once for the four rows ----
void q5_0_x4(int n, float out[4], const void* w_base, size_t row_bytes, const void* a) {
    const int nb = n / QK8_0;
    const block_q8_0* y = (const block_q8_0*) a;
    const uint8_t* w0 = (const uint8_t*) w_base;
    __m256 acc[4];
    for (int r = 0; r < 4; ++r) acc[r] = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256i qy = _mm256_loadu_si256((const __m256i*) y[ib].qs);
        for (int r = 0; r < 4; ++r) {
            const block_q5_0* xr =
                (const block_q5_0*) (w0 + (size_t) r * row_bytes + (size_t) ib * sizeof(block_q5_0));
            const __m256 d = _mm256_set1_ps(h2f(xr->d) * h2f(y[ib].d));
            __m256i qx = bytes_from_nibbles_32(xr->qs);
            __m256i bxhi = bytes_from_bits_32(xr->qh);
            bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char) 0xF0));
            qx = _mm256_or_si256(qx, bxhi);
            acc[r] = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc[r]);
        }
    }
    for (int r = 0; r < 4; ++r) out[r] = hsum_float_8(acc[r]);
}

// ---- GGML_TYPE_IQ2_XS: the four rows' grids and signs differ; their q8 loads and this walk's shape are shared ----
void iq2_xs_x4(int n, float out[4], const void* w_base, size_t row_bytes, const void* a) {
    static const char block_sign_shuffle_mask_1[32] = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
        0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06, 0x06,
    };
    static const char block_sign_shuffle_mask_2[32] = {
        0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a, 0x0a,
        0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0c, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e, 0x0e,
    };
    static const uint8_t bit_selector_mask_bytes[32] = {
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
        0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80,
    };
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*) bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_2);
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*) k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m256i mone = _mm256_set1_epi8(1);

    const int nb = n / QK_K;
    const block_q8_K* y = (const block_q8_K*) a;
    const uint8_t* w0 = (const uint8_t*) w_base;
    __m256 accumf[4];
    for (int r = 0; r < 4; ++r) accumf[r] = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        const uint8_t* rows[4];
        float d[4];
        __m128i scales[4];
        for (int r = 0; r < 4; ++r) {
            const block_iq2_xs* xr =
                (const block_iq2_xs*) (w0 + (size_t) r * row_bytes + (size_t) i * sizeof(block_iq2_xs));
            rows[r] = (const uint8_t*) xr->qs;
            d[r] = h2f(xr->d) * y[i].d;
            uint64_t aux64;
            memcpy(&aux64, xr->scales, 8);
            __m128i stmp = _mm_set1_epi64x((long long) aux64);
            stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
            scales[r] = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);
        }
        const int8_t* q8 = y[i].qs;
        __m256i sumi1[4], sumi2[4];
        for (int r = 0; r < 4; ++r) sumi1[r] = sumi2[r] = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            // the activation's four 32-byte walks, once for the rows
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i*) q8); q8 += 32;
            for (int r = 0; r < 4; ++r) {
                const __m256i q2_data = _mm256_loadu_si256((const __m256i*) rows[r]);
                rows[r] += 32;
                __m256i aux_gindex = _mm256_and_si256(q2_data, m511);
                const uint16_t* gindex = (const uint16_t*) &aux_gindex;

                const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
                const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
                const __m256i partial_sign_bits_for_counting =
                    _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
                const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
                const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

                const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
                const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
                const __m256i full_signs_1 = MM256_SET_M128I(full_signs_l, full_signs_l);
                const __m256i full_signs_2 = MM256_SET_M128I(full_signs_h, full_signs_h);

                const __m256i q2_1 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[3]],
                                                       (long long) iq2xs_grid[gindex[2]],
                                                       (long long) iq2xs_grid[gindex[1]],
                                                       (long long) iq2xs_grid[gindex[0]]);
                const __m256i q2_2 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[7]],
                                                       (long long) iq2xs_grid[gindex[6]],
                                                       (long long) iq2xs_grid[gindex[5]],
                                                       (long long) iq2xs_grid[gindex[4]]);
                const __m256i q2_3 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[11]],
                                                       (long long) iq2xs_grid[gindex[10]],
                                                       (long long) iq2xs_grid[gindex[9]],
                                                       (long long) iq2xs_grid[gindex[8]]);
                const __m256i q2_4 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[15]],
                                                       (long long) iq2xs_grid[gindex[14]],
                                                       (long long) iq2xs_grid[gindex[13]],
                                                       (long long) iq2xs_grid[gindex[12]]);

                __m256i signs;
                signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_1);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_1 = _mm256_sign_epi8(q8_1, _mm256_or_si256(signs, mone));

                signs = _mm256_shuffle_epi8(full_signs_1, block_sign_shuffle_2);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_2 = _mm256_sign_epi8(q8_2, _mm256_or_si256(signs, mone));

                signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_1);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_3 = _mm256_sign_epi8(q8_3, _mm256_or_si256(signs, mone));

                signs = _mm256_shuffle_epi8(full_signs_2, block_sign_shuffle_2);
                signs = _mm256_cmpeq_epi8(_mm256_and_si256(signs, bit_selector_mask), bit_selector_mask);
                const __m256i q8s_4 = _mm256_sign_epi8(q8_4, _mm256_or_si256(signs, mone));

                const __m256i dot1 = _mm256_maddubs_epi16(q2_1, q8s_1);
                const __m256i dot2 = _mm256_maddubs_epi16(q2_2, q8s_2);
                const __m256i dot3 = _mm256_maddubs_epi16(q2_3, q8s_3);
                const __m256i dot4 = _mm256_maddubs_epi16(q2_4, q8s_4);

                const __m256i sc1 =
                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales[r], get_scale_shuffle(ib32 + 0)));
                const __m256i sc2 =
                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales[r], get_scale_shuffle(ib32 + 1)));
                const __m256i sc3 =
                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales[r], get_scale_shuffle(ib32 + 2)));
                const __m256i sc4 =
                    _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales[r], get_scale_shuffle(ib32 + 3)));

                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot1, sc1));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot2, sc2));
                sumi1[r] = _mm256_add_epi32(sumi1[r], _mm256_madd_epi16(dot3, sc3));
                sumi2[r] = _mm256_add_epi32(sumi2[r], _mm256_madd_epi16(dot4, sc4));
            }
        }
        for (int r = 0; r < 4; ++r)
            accumf[r] = _mm256_fmadd_ps(_mm256_set1_ps(d[r]),
                                        _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1[r], sumi2[r])), accumf[r]);
    }
    for (int r = 0; r < 4; ++r) out[r] = 0.125f * hsum_float_8(accumf[r]);
}

}  // namespace

DotX4Fn dot_x4_kernel(int type) {
    switch (type) {
        case 6: return q5_0_x4;      // GGML_TYPE_Q5_0
        case 17: return iq2_xs_x4;   // GGML_TYPE_IQ2_XS
        default: return nullptr;
    }
}

}  // namespace strata::kernels::cpu
#else
namespace strata::kernels::cpu {
DotX4Fn dot_x4_kernel(int) { return nullptr; }
}  // namespace strata::kernels::cpu
#endif
