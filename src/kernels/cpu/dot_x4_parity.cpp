// src/kernels/cpu/dot_x4_parity.cpp - the four-row kernels (dot_x4.cpp) bit for bit against the verbatim AVX2
// chains of ggml-cpu (quants.c at the llama.cpp the engines pin, 3cf03257) and, where the engines link ggml-cpu,
// against its own vec_dot.  No GPU, no model: random blocks, so every grid index and sign pattern occurs, and
// random activations quantized the way the engine quantizes them.
//
//     dot_x4_parity [--reps N]
//
// The verbatim copies are the oracle on purpose: the chains wrap pair sums in int16 (an IQ2_XS pair reaches
// 65280 with the grid bytes to 255), so a clean per-value integer sum is NOT the same arithmetic - only the
// instructions ggml runs are.  A value of the four-row kernel must equal the one-row vec_dot's for the same
// pair - memcmp, not a tolerance.
#include "strata/kernels/cpu/dot_x4.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

// STRATA_DOT_X4_NO_GGML: the verbatim oracle only (a checkout without ggml-cpu to link - the ggml-cpu
// comparison runs where the engines link it).
#ifndef STRATA_DOT_X4_NO_GGML
#include "ggml.h"
#include "ggml-cpu.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <vector>

namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint16_t to_half(float f) {
    return (uint16_t) _mm_cvtsi128_si32(_mm_cvtps_ph(_mm_set_ss(f), _MM_FROUND_TO_NEAREST_INT));
}
#define GGML_CPU_FP16_TO_FP32(x) h2f(x)

// ---- ggml-cpu/arch/x86/quants.c at the pin, verbatim ----

static inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;  memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                                0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32((int) x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *) rsi);
    const __m256i bytes = _mm256_insertf128_si256(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4), 1);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

static inline __m256 sum_i16_pairs_float(const __m256i x) {
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    return _mm256_cvtepi32_ps(summed_pairs);
}

static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return sum_i16_pairs_float(_mm256_maddubs_epi16(ax, sy));
}

// ggml_vec_dot_q5_0_q8_0's AVX2 branch, verbatim (the fp16-to-fp32 lookups aside: the hardware convert is the
// same value the table holds)
float ref_q5_0(const void* vx, const void* vy, int n) {
    const block_q5_0 * x = (const block_q5_0*) vx;
    const block_q8_0 * y = (const block_q8_0*) vy;
    const int nb = n / QK8_0;
    __m256 acc = _mm256_setzero_ps();
    for (int ib = 0; ib < nb; ++ib) {
        const __m256 d = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d));
        __m256i qx = bytes_from_nibbles_32(x[ib].qs);
        __m256i bxhi = bytes_from_bits_32(x[ib].qh);
        bxhi = _mm256_andnot_si256(bxhi, _mm256_set1_epi8((char) 0xF0));
        qx = _mm256_or_si256(qx, bxhi);
        const __m256i qy = _mm256_loadu_si256((const __m256i *) y[ib].qs);
        const __m256 q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }
    return hsum_float_8(acc);
}

static inline __m128i get_scale_shuffle(int i) {
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

// ggml_vec_dot_iq2_xs_q8_K's AVX2 branch, verbatim
float ref_iq2_xs(const void* vx, const void* vy, int n) {
    const block_iq2_xs * x = (const block_iq2_xs*) vx;
    const block_q8_K   * y = (const block_q8_K*) vy;
    const int nb = n / QK_K;

    const __m256i mone = _mm256_set1_epi8(1);
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
    const __m256i bit_selector_mask = _mm256_loadu_si256((const __m256i*) bit_selector_mask_bytes);
    const __m256i block_sign_shuffle_1 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_1);
    const __m256i block_sign_shuffle_2 = _mm256_loadu_si256((const __m256i*) block_sign_shuffle_mask_2);
    static const uint8_t k_bit_helper[32] = {
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
        0x00, 0x80, 0x80, 0x00, 0x80, 0x00, 0x00, 0x80, 0x80, 0x00, 0x00, 0x80, 0x00, 0x80, 0x80, 0x00,
    };
    const __m256i bit_helper = _mm256_loadu_si256((const __m256i*) k_bit_helper);
    const __m256i m511 = _mm256_set1_epi16(511);
    const __m128i m4 = _mm_set1_epi8(0xf);
    const __m128i m1 = _mm_set1_epi8(1);

    uint64_t aux64;
    __m256i aux_gindex;
    const uint16_t * gindex = (const uint16_t *) &aux_gindex;

    __m256 accumf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const float d = GGML_CPU_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint16_t * q2 = x[i].qs;
        const int8_t   * q8 = y[i].qs;
        memcpy(&aux64, x[i].scales, 8);
        __m128i stmp = _mm_set1_epi64x((long long) aux64);
        stmp = _mm_unpacklo_epi8(_mm_and_si128(stmp, m4), _mm_and_si128(_mm_srli_epi16(stmp, 4), m4));
        const __m128i scales = _mm_add_epi8(_mm_slli_epi16(stmp, 1), m1);

        __m256i sumi1 = _mm256_setzero_si256();
        __m256i sumi2 = _mm256_setzero_si256();
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 4) {
            const __m256i q2_data = _mm256_loadu_si256((const __m256i*) q2);  q2 += 16;
            aux_gindex = _mm256_and_si256(q2_data, m511);
            const __m256i partial_sign_bits = _mm256_srli_epi16(q2_data, 9);
            const __m256i partial_sign_bits_upper = _mm256_srli_epi16(q2_data, 13);
            const __m256i partial_sign_bits_for_counting = _mm256_xor_si256(partial_sign_bits, partial_sign_bits_upper);
            const __m256i odd_bits = _mm256_shuffle_epi8(bit_helper, partial_sign_bits_for_counting);
            const __m256i full_sign_bits = _mm256_or_si256(partial_sign_bits, odd_bits);

            const __m256i q8_1 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;
            const __m256i q8_4 = _mm256_loadu_si256((const __m256i *) q8); q8 += 32;

            const __m256i q2_1 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[ 3]], (long long) iq2xs_grid[gindex[ 2]],
                                                   (long long) iq2xs_grid[gindex[ 1]], (long long) iq2xs_grid[gindex[ 0]]);
            const __m256i q2_2 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[ 7]], (long long) iq2xs_grid[gindex[ 6]],
                                                   (long long) iq2xs_grid[gindex[ 5]], (long long) iq2xs_grid[gindex[ 4]]);
            const __m256i q2_3 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[11]], (long long) iq2xs_grid[gindex[10]],
                                                   (long long) iq2xs_grid[gindex[ 9]], (long long) iq2xs_grid[gindex[ 8]]);
            const __m256i q2_4 = _mm256_set_epi64x((long long) iq2xs_grid[gindex[15]], (long long) iq2xs_grid[gindex[14]],
                                                   (long long) iq2xs_grid[gindex[13]], (long long) iq2xs_grid[gindex[12]]);

            const __m128i full_signs_l = _mm256_castsi256_si128(full_sign_bits);
            const __m128i full_signs_h = _mm256_extractf128_si256(full_sign_bits, 1);
            const __m256i full_signs_1 = _mm256_insertf128_si256(_mm256_castsi128_si256(full_signs_l), full_signs_l, 1);
            const __m256i full_signs_2 = _mm256_insertf128_si256(_mm256_castsi128_si256(full_signs_h), full_signs_h, 1);

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

            const __m256i sc1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 0)));
            const __m256i sc2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 1)));
            const __m256i sc3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 2)));
            const __m256i sc4 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(ib32 + 3)));

            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot1, sc1));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot2, sc2));
            sumi1 = _mm256_add_epi32(sumi1, _mm256_madd_epi16(dot3, sc3));
            sumi2 = _mm256_add_epi32(sumi2, _mm256_madd_epi16(dot4, sc4));
        }
        accumf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_add_epi32(sumi1, sumi2)), accumf);
    }
    return 0.125f * hsum_float_8(accumf);
}

// ---- the random blocks ----

std::mt19937 rng(12345);

void gen_q5_0(std::vector<uint8_t>& w, size_t& row_bytes, std::vector<uint8_t>& a, int n, int rows) {
    row_bytes = (size_t) (n / QK8_0) * sizeof(block_q5_0);
    w.assign(row_bytes * (size_t) rows, 0);
    std::uniform_int_distribution<int> byte(0, 255), nib(0, 15);
    std::uniform_real_distribution<float> df(0.001f, 0.1f);
    for (int r = 0; r < rows; ++r) {
        block_q5_0* x = (block_q5_0*) (w.data() + (size_t) r * row_bytes);
        for (int i = 0; i < n / QK8_0; ++i) {
            x[i].d = to_half(df(rng));
            for (int v = 0; v < 4; ++v) x[i].qh[v] = (uint8_t) byte(rng);
            for (int v = 0; v < QK5_0 / 2; ++v) x[i].qs[v] = (uint8_t) ((nib(rng) << 4) | nib(rng));
        }
    }
    const size_t ab = sizeof(block_q8_0) * (size_t) (n / QK8_0);
    a.assign(ab, 0);
    auto* y = (block_q8_0*) a.data();
    for (int i = 0; i < n / QK8_0; ++i) {
        y[i].d = to_half(df(rng));
        for (int v = 0; v < QK8_0; ++v) y[i].qs[v] = (int8_t) byte(rng);
    }
}

void gen_iq2_xs(std::vector<uint8_t>& w, size_t& row_bytes, std::vector<uint8_t>& a, int n, int rows) {
    row_bytes = (size_t) (n / QK_K) * sizeof(block_iq2_xs);
    w.assign(row_bytes * (size_t) rows, 0);
    std::uniform_int_distribution<int> byte16(0, 65535), byte(0, 255), nib(0, 15);
    std::uniform_real_distribution<float> df(0.001f, 0.1f);
    for (int r = 0; r < rows; ++r) {
        block_iq2_xs* x = (block_iq2_xs*) (w.data() + (size_t) r * row_bytes);
        for (int i = 0; i < n / QK_K; ++i) {
            x[i].d = to_half(df(rng));
            for (int j = 0; j < QK_K / 8; ++j) x[i].qs[j] = (uint16_t) byte16(rng);
            for (int j = 0; j < QK_K / 32; ++j) x[i].scales[j] = (uint8_t) ((2 * nib(rng) + 1) | ((2 * nib(rng) + 1) << 4));
        }
    }
    const size_t ab = sizeof(block_q8_K) * (size_t) (n / QK_K);
    a.assign(ab, 0);
    auto* y = (block_q8_K*) a.data();
    for (int i = 0; i < n / QK_K; ++i) {
        y[i].d = df(rng);
        for (int v = 0; v < QK_K; ++v) y[i].qs[v] = (int8_t) byte(rng);
    }
}

int fails = 0;

void check(const char* what, int type, int n, int rows, const std::vector<uint8_t>& w, size_t row_bytes,
           const std::vector<uint8_t>& a) {
    const strata::kernels::cpu::DotX4Fn k = strata::kernels::cpu::dot_x4_kernel(type);
    if (!k) {
        std::printf("dot_x4_parity: no kernel for type %d on this cpu\n", type);
        return;
    }
    // one call from the first row: the kernel reads the four rows that follow it
    float o[4] = {};
    k(n, o, w.data(), row_bytes, a.data());
    for (int r = 0; r < rows; ++r) {
        const uint8_t* wr = w.data() + (size_t) r * row_bytes;
        const float ref = type == 6 ? ref_q5_0(wr, a.data(), n) : ref_iq2_xs(wr, a.data(), n);
        if (std::memcmp(&o[r], &ref, 4) != 0) {
            std::printf("dot_x4_parity: %s row %d: x4 %.9g != verbatim ggml %.9g\n", what, r, o[r], ref);
            ++fails;
        }
#ifndef STRATA_DOT_X4_NO_GGML
        float g = 0.f;
        const ggml_type_traits_cpu* t = ggml_get_type_traits_cpu((ggml_type) type);
        t->vec_dot(n, &g, 0, wr, 0, a.data(), 0, 1);
        if (std::memcmp(&o[r], &g, 4) != 0) {
            std::printf("dot_x4_parity: %s row %d: x4 %.9g != ggml-cpu %.9g\n", what, r, o[r], g);
            ++fails;
        }
#endif
    }
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 8;
    if (argc > 2 && !std::strcmp(argv[1], "--reps")) reps = std::atoi(argv[2]);
    std::printf("dot_x4_parity: %d reps of random q5_0 and iq2_xs rows\n", reps);
    for (int blocks : {1, 2, 3, 4, 8, 16, 32, 64, 128}) {
        std::vector<uint8_t> w, a;
        size_t row_bytes;
        int was = fails;
        gen_q5_0(w, row_bytes, a, blocks * QK8_0, 4);
        check("q5_0", 6, blocks * QK8_0, 4, w, row_bytes, a);
        std::printf("  q5_0 %3d blocks: %s\n", blocks, fails == was ? "ok" : "DIVERGES");
        was = fails;
        gen_iq2_xs(w, row_bytes, a, blocks * QK_K, 4);
        check("iq2_xs", 17, blocks * QK_K, 4, w, row_bytes, a);
        std::printf("  iq2_xs %3d blocks: %s\n", blocks, fails == was ? "ok" : "DIVERGES");
    }
    for (int rep = 0; rep < reps; ++rep) {
        std::vector<uint8_t> w, a;
        size_t row_bytes;
        gen_q5_0(w, row_bytes, a, 4096, 4);
        check("q5_0", 6, 4096, 4, w, row_bytes, a);
        gen_iq2_xs(w, row_bytes, a, 4096, 4);
        check("iq2_xs", 17, 4096, 4, w, row_bytes, a);
    }
    if (!fails) std::printf("dot_x4_parity: PASSED, the bits agree\n");
    return fails ? 1 : 0;
}
