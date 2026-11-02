// src/poly/expert_mma_test.cu - ExpertMma::mul against the same products on the host: the weights are random
// Q3_K blocks (their dequant is the port of include/strata/artifact/dequant.hpp), the reference sums the
// kernel's half-rounded operands in fp32; a full-precision reference runs besides it, so a rounding-mirror bug
// and a kernel bug tell themselves apart.  Ragged bounds (an empty expert among them) and a scrambled ids
// scatter included.  The verdict is on the error scaled by the reference's spread (the products' k sums are
// wide, so a plain relative error drowns in the near-zero sums).
//
// usage: poly_expert_mma_test   (exit 0: every product inside the tolerance)
#include "expert_mma.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

constexpr int kQ3KBytes = 110;

// dequantize_q3_K (the validated reference), one block of 256 values, full precision
void dequant_q3k(const uint8_t* block, float* out) {
    const uint8_t* hm = block;
    const uint8_t* q = block + 32;
    const float d_all = __half2float(*reinterpret_cast<const half*>(block + 108));
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, block + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            const float dl = d_all * (float) (scales[is++] - 32);
            for (int l = 0; l < 16; ++l) *out++ = dl * (float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
            const float dl2 = d_all * (float) (scales[is++] - 32);
            for (int l = 0; l < 16; ++l)
                *out++ = dl2 * (float) ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
            shift += 2;
            m = (uint8_t) (m << 1);
        }
        q += 32;
    }
}

// the operands the kernel's fragments carry: half(code) * half(d_all * (sc - 32)) per value
void dequant_q3k_fragments(const uint8_t* block, float* out) {
    const uint8_t* hm = block;
    const uint8_t* q = block + 32;
    const float d_all = __half2float(*reinterpret_cast<const half*>(block + 108));
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4] = {0, 0, 0, 0};
    std::memcpy(aux, block + 96, 12);
    const uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
    int is = 0;
    uint8_t m = 1;
    for (int n = 0; n < 256; n += 128) {
        int shift = 0;
        for (int j = 0; j < 4; ++j) {
            const float s0 = __half2float(__float2half(d_all * (float) (scales[is++] - 32)));
            for (int l = 0; l < 16; ++l)
                *out++ = __half2float(__float2half((float) ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)))) * s0;
            const float s1 = __half2float(__float2half(d_all * (float) (scales[is++] - 32)));
            for (int l = 0; l < 16; ++l)
                *out++ =
                    __half2float(__float2half((float) ((int8_t) ((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4)))) * s1;
            shift += 2;
            m = (uint8_t) (m << 1);
        }
        q += 32;
    }
}

// the kernel's quantize on the host: the same d and q, the same half rounding of the operands
void host_quantize(const float* x, int cols, int64_t total_rows, uint8_t* xq, float* operands) {
    const int nb = cols / 32;
    const int64_t q_row = (int64_t) nb * 36;
    for (int64_t row = 0; row < total_rows; ++row) {
        uint8_t* out = xq + (size_t) row * q_row;
        for (int b = 0; b < nb; ++b) {
            float amax = 0.f;
            for (int l = 0; l < 32; ++l) amax = fmaxf(amax, fabsf(x[row * cols + 32 * b + l]));
            const float d = amax > 0.f ? amax / 127.f : 1.f;
            *reinterpret_cast<half*>(out + 36 * b) = __float2half(d);
            for (int l = 0; l < 32; ++l) {
                int q = (int) std::round(x[row * cols + 32 * b + l] / d);
                q = std::max(-127, std::min(127, q));
                out[36 * b + 4 + l] = (int8_t) q;
                operands[row * cols + 32 * b + l] =
                    __half2float(__float2half((float) q)) * __half2float(__float2half(d));
            }
        }
    }
}

int check(int m, int w_rows, int w_cols, const std::vector<int>& bounds, bool scramble, std::mt19937& rng,
          bool dump) {
    const int total_rows = bounds.back();
    const int nb = w_cols / 256;
    const int row_bytes = nb * 110;
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_int_distribution<int> scale6(0, 63);

    // the weights: random Q3_K blocks, m experts
    std::vector<uint8_t> w((size_t) m * w_rows * row_bytes);
    for (uint8_t& v : w) v = (uint8_t) (rng() & 255);
    for (size_t e = 0; e < (size_t) m; ++e)
        for (int r = 0; r < w_rows; ++r) {
            uint8_t* blk = w.data() + (e * w_rows + r) * row_bytes;
            for (int b = 0; b < nb; ++b) {
                *reinterpret_cast<half*>(blk + 110 * b + 108) = __float2half(0.05f * nd(rng));
                for (int s = 0; s < 12; ++s) blk[110 * b + 96 + s] = (uint8_t) scale6(rng);
            }
        }
    std::vector<float> wop((size_t) m * w_rows * w_cols);
    for (size_t e = 0; e < (size_t) m; ++e)
        for (int r = 0; r < w_rows; ++r)
            for (int b = 0; b < nb; ++b)
                dequant_q3k_fragments(w.data() + (e * w_rows + r) * row_bytes + 110 * b,
                                      wop.data() + ((e * w_rows + r) * w_cols) + 256 * b);

    std::vector<float> x((size_t) total_rows * w_cols);
    for (float& v : x) v = nd(rng);
    std::vector<uint8_t> xq(ExpertMma::x8_bytes(total_rows, w_cols));
    std::vector<float> xop((size_t) total_rows * w_cols);
    host_quantize(x.data(), w_cols, total_rows, xq.data(), xop.data());

    std::vector<int32_t> ids;
    if (scramble) {
        ids.resize(total_rows);
        for (int i = 0; i < total_rows; ++i) ids[i] = i;
        std::shuffle(ids.begin(), ids.end(), rng);
    }

    // two references: the fragment-rounded operands (what the kernel sums), and the full-precision dequant
    const int64_t ld_dst = w_rows;
    std::vector<float> want((size_t) total_rows * w_rows, 12345.f), full(want.size(), 12345.f);
    std::vector<float> wfull(wop.size());
    if (dump)
        for (size_t e = 0; e < (size_t) m; ++e)
            for (int r = 0; r < w_rows; ++r)
                for (int b = 0; b < nb; ++b)
                    dequant_q3k(w.data() + (e * w_rows + r) * row_bytes + 110 * b,
                                wfull.data() + ((e * w_rows + r) * w_cols) + 256 * b);
    for (size_t e = 0; e < (size_t) m; ++e)
        for (int t = bounds[e]; t < bounds[e + 1]; ++t) {
            const int64_t drow = scramble ? ids[t] : t;
            for (int r = 0; r < w_rows; ++r) {
                float acc = 0.f, accf = 0.f;
                const float* wr = wop.data() + ((e * w_rows + r) * w_cols);
                const float* xf = xop.data() + (size_t) t * w_cols;
                for (int k = 0; k < w_cols; ++k) acc = std::fma(wr[k], xf[k], acc);
                if (dump)
                    for (int k = 0; k < w_cols; ++k)
                        accf = std::fma(wfull[((e * w_rows + r) * w_cols) + k], x[(size_t) t * w_cols + k], accf);
                want[(size_t) drow * w_rows + r] = acc;
                if (dump) full[(size_t) drow * w_rows + r] = accf;
            }
        }

    uint8_t* dw = nullptr, * dxq = nullptr;
    int32_t* dbounds = nullptr, * dids = nullptr;
    float* ddst = nullptr;
    cudaMalloc(&dw, w.size());
    cudaMalloc(&dxq, xq.size());
    cudaMalloc(&dbounds, bounds.size() * 4);
    if (scramble) cudaMalloc(&dids, ids.size() * 4);
    cudaMalloc(&ddst, want.size() * 4);
    cudaMemcpy(dw, w.data(), w.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dxq, xq.data(), xq.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dbounds, bounds.data(), bounds.size() * 4, cudaMemcpyHostToDevice);
    if (scramble) cudaMemcpy(dids, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice);
    ExpertMma::mul(dw, w_rows, w_cols, (size_t) w_rows * row_bytes, m, dxq, dbounds,
                   scramble ? dids : nullptr, total_rows, total_rows, ddst, ld_dst, 0);
    std::vector<float> got(want.size());
    cudaMemcpy(got.data(), ddst, got.size() * 4, cudaMemcpyDeviceToHost);
    cudaFree(dw), cudaFree(dxq), cudaFree(dbounds), cudaFree(dids), cudaFree(ddst);

    // the spread of the reference sums and the kernel's error against it
    double l2 = 0, worst_abs = 0;
    for (size_t e = 0; e < (size_t) m; ++e)
        for (int t = bounds[e]; t < bounds[e + 1]; ++t) {
            const int64_t drow = scramble ? ids[t] : t;
            for (int r = 0; r < w_rows; ++r) {
                const double ref = want[(size_t) drow * w_rows + r];
                l2 += ref * ref;
                worst_abs = fmax(worst_abs, std::fabs((double) got[(size_t) drow * w_rows + r] - ref));
            }
        }
    const double sigma = std::sqrt(l2 / std::max(1, (int) want.size()));
    const double tol = 3e-2 * std::sqrt((double) w_cols) * 2.0 + 1e-3;   // ~2 rms of one product, grown by sqrt(k)
    int bad = worst_abs > tol;
    std::printf("%s m %d w_rows %4d w_cols %4d rows %4d%s   sigma %8.2f  worst abs %10.4f  tol %8.3f\n",
                bad ? "FAIL" : "ok  ", m, w_rows, w_cols, total_rows, scramble ? " scrambled" : "           ",
                sigma, worst_abs, tol);
    if (dump) {
        std::printf("     first outputs of expert 0 (row, token: got ~ fragment-ref ~ full-ref):\n");
        for (int r = 0; r < 6; ++r)
            for (int t = 0; t < 3; ++t)
                std::printf("     (%2d,%2d)  %12.4f ~ %12.4f ~ %12.4f\n", r, t,
                            got[(size_t) t * w_rows + r], want[(size_t) t * w_rows + r],
                            full[(size_t) t * w_rows + r]);
    }
    return bad ? 1 : 0;
}

}  // namespace

int main() {
    if (!ExpertMma::fits(ExpertMma::kQ3K, 1024)) {
        std::printf("the card takes no expert mma (sm_80+ and Q3_K needed)\n");
        return 1;
    }
    std::mt19937 rng(7);
    int bad = 0;
    bad += check(1, 16, 128, {0, 32}, false, rng, true);      // one BK step, one Q3_K block: the whole dump
    bad += check(1, 160, 1024, {0, 100}, false, rng, false);
    bad += check(3, 160, 1024, {0, 37, 37, 100}, false, rng, false);    // an expert without rows
    bad += check(3, 160, 1024, {0, 37, 37, 100}, true, rng, false);     // the same, scrambled dst
    bad += check(8, 160, 1024, {0, 1, 5, 20, 21, 55, 56, 90, 100}, true, rng, false);
    bad += check(2, 256, 2048, {0, 64, 100}, false, rng, false);        // two 128-steps of k, two row tiles
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
