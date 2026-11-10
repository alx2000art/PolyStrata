// src/poly/expert_mma.cu - see the header.  The mma fragment convention is the validated one of
// src/kernels/cuda/qsa_prompt_attn.cu (idx_score_mma): a lane's A and B registers hold the same k values, the
// rows ride g8 = lane/4 and the columns of C are 2 * t4 (t4 = lane % 4), and one m16n8k16 instruction's
// registers cover k steps of four (d .. d + 3) with the (bb, hh) loop covering the rest of the 32 — the
// hardware's k labels differ from the registers' contents, but A and B carry the same contents per lane, so
// every product pairs as it should and every k of the step is summed once.

#include "expert_mma.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#if defined(__HIPCC__) || defined(__gfx906__)
// HIP/AMD: no mma.sync in this file.  The flag stays off (fits() is 0) and the calls never come; the host
// side still links.
#include <stdexcept>
#include <string>

namespace strata::poly {

bool ExpertMma::fits(int ggml_type, int64_t w_cols) {
    (void) ggml_type; (void) w_cols;
    return false;
}

size_t ExpertMma::x8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 32) * 36;
}

void ExpertMma::quantize(const float*, const int32_t*, void*, int64_t, int64_t, int64_t, void*) {
    throw std::runtime_error("expert mma: not built for this backend");
}

void ExpertMma::mul(const void*, int64_t, int64_t, size_t, int, const void*, const int32_t*, const int32_t*,
                    int64_t, int64_t, float*, int64_t, void*) {
    throw std::runtime_error("expert mma: not built for this backend");
}

}  // namespace strata::poly

#else
namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int BM = 64;    // weight rows a block takes, four m16 tiles
constexpr int BN = 32;    // xq rows (tokens) a block takes, four n8 tiles
constexpr int BK = 128;   // the k a step takes: half a Q3_K block, eight mma calls
constexpr int SR = BK + 8;   // a shared row's halfs, padded so the lanes' rows land in different banks

__shared__ __align__(16) half sA[BM][SR];   // the codes of BM weight rows (exact in a half)
__shared__ __align__(16) half sB[BN][SR];   // the codes of BN xq rows
__shared__ float sSC[BM][8];                // a row's eight 16-block scales of the half: d_all * (sc - 32)
__shared__ float sBD[BN][4];                // an xq row's per-32 scale, the four 32-blocks of the step

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void) c; (void) a; (void) b;
    __trap();   // fits() keeps this kernel off cards without the f16 m16n8k16
#endif
}

__device__ __forceinline__ uint32_t hmul2(uint32_t v, float s) {
    __half2 h = *reinterpret_cast<__half2*>(&v);
    h = __hmul2(h, __half2half2(__float2half(s)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ float warp_amax(float x) {
    x = fabsf(x);
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, fabsf(__shfl_xor_sync(0xffffffff, x, o, WARP)));
    return x;
}

// x -> a plain per-32 int8 with an fp16 scale, 36 bytes a block ({ half d; half pad; int8 qs[32] }).
// A warp a 32-value block; row = blockIdx.x; row i of xq is row ids[i] of x, or row i when ids is null.
__global__ __launch_bounds__(256) void expert_mma_quant_kernel(const float* __restrict__ x,
                                                               const int32_t* __restrict__ ids,
                                                               uint8_t* __restrict__ xq, int cols, int64_t ld,
                                                               int64_t q_row) {
    const int lane = threadIdx.x & (WARP - 1), wid = threadIdx.x >> 5;
    const int64_t row = blockIdx.x;
    const int nb = (cols + 31) / 32;
    const int64_t src = ids ? (int64_t) ids[row] : row;
    for (int b32 = wid; b32 < nb; b32 += blockDim.x / WARP) {
        const int k = 32 * b32 + lane;
        const float v = k < cols ? x[(size_t) src * ld + k] : 0.f;
        const float amax = warp_amax(v);
        const float d = amax > 0.f ? amax / 127.f : 1.f;
        int q = (int) roundf(v / d);
        q = max(-127, min(127, q));
        uint8_t* blk = xq + (size_t) row * q_row + (size_t) b32 * 36;
        if (lane == 0) *reinterpret_cast<half*>(blk) = __float2half(d);
        blk[4 + lane] = (int8_t) q;
    }
}

// One block: BM weight rows of one expert against BN xq rows, the whole k in steps of BK.
__global__ __launch_bounds__(256) void expert_mma_mul_kernel(const uint8_t* __restrict__ w, int w_rows, int w_cols,
                                                             size_t stride, const uint8_t* __restrict__ xq,
                                                             const int32_t* __restrict__ bounds,
                                                             const int32_t* __restrict__ ids, int64_t total_rows,
                                                             float* __restrict__ dst, int64_t ld_dst) {
    const int lane = threadIdx.x & (WARP - 1), wid = threadIdx.x >> 5;
    const int g8 = lane >> 2, t4 = lane & 3;
    const int e = blockIdx.z;
    const int be0 = bounds[e], be1 = bounds[e + 1];
    const int bm0 = blockIdx.x * BM, tok0 = blockIdx.y * BN;
    const int row_bytes = w_cols / 256 * 110, q_row = w_cols / 32 * 36;
    const int wm = wid & 3, wn = wid >> 2;   // m tile wm * 16, n tiles (2 * wn + {0, 1}) * 8

    float c[2][4];
#pragma unroll
    for (int ti = 0; ti < 2; ++ti)
#pragma unroll
        for (int j = 0; j < 4; ++j) c[ti][j] = 0.f;

    for (int k0 = 0; k0 < w_cols; k0 += BK) {
        // ---- sA: the codes of BM rows, this half (hb) of the Q3_K block; sSC: their eight 16-block scales
        {
            const int row = threadIdx.x >> 2, j = threadIdx.x & 3;
            const int row_g = bm0 + row;
            const int hb = (k0 >> 7) & 1;
            float sc0 = 0.f, sc1 = 0.f;
            if (row_g < w_rows) {
                const uint8_t* rb = w + (size_t) e * stride + (size_t) row_g * row_bytes + (size_t) (k0 / 256) * 110;
                const float d_all = __half2float(*reinterpret_cast<const half*>(rb + 108));
                // the scales' byte shuffle of dequantize_q3_K (kmask1 0x03030303, kmask2 0x0f0f0f0f)
                const uint8_t* sp = rb + 96;
                const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
                uint32_t a0 = sp[0] | sp[1] << 8 | sp[2] << 16 | sp[3] << 24;
                uint32_t a1 = sp[4] | sp[5] << 8 | sp[6] << 16 | sp[7] << 24;
                uint32_t a2 = sp[8] | sp[9] << 8 | sp[10] << 16 | sp[11] << 24;
                const uint32_t tmp = a2;
                a2 = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                uint32_t a3 = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                a0 = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
                a1 = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
                const uint32_t aux[4] = {a0, a1, a2, a3};
                const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
                // the thread's j covers k = j * 32 .. j * 32 + 31 of the half: subs 2j (p = 0) and 2j + 1 (p = 1);
                // the reference's is counter runs on across the halves, so the second half starts at 8
                sc0 = d_all * (float) (scales[8 * hb + 2 * j] - 32);
                sc1 = d_all * (float) (scales[8 * hb + 2 * j + 1] - 32);
                const int shift = 2 * j, hbit = j + 4 * hb;
                const uint8_t* qs = rb + 32 + 32 * hb;
                const uint8_t* hm = rb;
                half* sa = sA[row];
#pragma unroll
                for (int q = 0; q < 16; ++q) {
                    const int m0 = (qs[q] >> shift) & 3, m1 = (qs[q + 16] >> shift) & 3;
                    const int c0 = m0 - ((hm[q] >> hbit) & 1 ? 0 : 4);
                    const int c1 = m1 - ((hm[q + 16] >> hbit) & 1 ? 0 : 4);
                    sa[j * 32 + q] = __float2half((float) c0);
                    sa[j * 32 + 16 + q] = __float2half((float) c1);
                }
            } else {
#pragma unroll
                for (int q = 0; q < 32; ++q) sA[row][j * 32 + q] = __float2half(0.f);
            }
            // the window holds one half of the block, so its eight subs sit at 2j and 2j + 1 (an 8*hb offset
            // here ran past the row's eight slots and the second window read the neighbor's scales)
            sSC[row][2 * j] = sc0;
            sSC[row][2 * j + 1] = sc1;
        }
        // ---- sB: the codes of BN xq rows; sBD: their per-32 scales.  A thread fills one 16-value half of
        // one 32-block of its column: (t >> 3) the column, ((t >> 1) & 3) the block, (t & 1) the half.
        {
            const int col = threadIdx.x >> 3, b32 = (threadIdx.x >> 1) & 3, h16 = threadIdx.x & 1;
            const int kc = 32 * b32 + 16 * h16;
            const int xq_row = be0 + tok0 + col;
            half* sb = sB[col];
            float bd = 0.f;
            if (xq_row < be1 && xq_row < total_rows) {
                const uint8_t* blk = xq + (size_t) xq_row * q_row + (size_t) ((k0 >> 5) + b32) * 36;
                bd = __half2float(*reinterpret_cast<const half*>(blk));
                const char* qs = reinterpret_cast<const char*>(blk + 4);
#pragma unroll
                for (int q = 0; q < 16; ++q) sb[kc + q] = __float2half((float) qs[16 * h16 + q]);
            } else {
#pragma unroll
                for (int q = 0; q < 16; ++q) sb[kc + q] = __float2half(0.f);
            }
            if (h16 == 0) sBD[col][b32] = bd;
        }
        __syncthreads();

        // ---- the products: four 32-k steps of the window, two mma calls each
        const int arow = wm * 16;
#pragma unroll
        for (int bb = 0; bb < 4; ++bb) {
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int d = 32 * bb + 8 * t4 + 4 * hh;
                const int sub = 2 * bb + (t4 >= 2 ? 1 : 0);
                const float s0 = sSC[arow + g8][sub], s1 = sSC[arow + g8 + 8][sub];
                const uint2 lo = *reinterpret_cast<const uint2*>(&sA[arow + g8][d]);
                const uint2 hi = *reinterpret_cast<const uint2*>(&sA[arow + g8 + 8][d]);
                const uint32_t a[4] = {hmul2(lo.x, s0), hmul2(hi.x, s1), hmul2(lo.y, s0), hmul2(hi.y, s1)};
#pragma unroll
                for (int ti = 0; ti < 2; ++ti) {
                    const int col = (2 * wn + ti) * 8 + g8;
                    const float bd = sBD[col][bb];
                    const uint32_t b[2] = {hmul2(*reinterpret_cast<const uint32_t*>(&sB[col][d]), bd),
                                           hmul2(*reinterpret_cast<const uint32_t*>(&sB[col][d + 2]), bd)};
                    mma16816(c[ti], a, b);
                }
            }
        }
        __syncthreads();
    }

    // ---- the scatter: weight row wrow of the matrix, xq row be0 + tok0 + the tile's column
#pragma unroll
    for (int ti = 0; ti < 2; ++ti) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int wrow = bm0 + wm * 16 + g8 + 8 * (j >> 1);
            const int xq_row = be0 + tok0 + (2 * wn + ti) * 8 + 2 * t4 + (j & 1);
            if (wrow < w_rows && xq_row < be1 && xq_row < total_rows) {
                const int dst_row = ids ? ids[xq_row] : xq_row;
                dst[(size_t) dst_row * ld_dst + wrow] = c[ti][j];
            }
        }
    }
}

int device_cc() {
    int dev = 0, major = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
    return major;
}

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

bool ExpertMma::fits(int ggml_type, int64_t w_cols) {
    // a Q3_K block is 256 values: the matrix width goes by whole blocks
    return ggml_type == kQ3K && w_cols > 0 && w_cols % 256 == 0 && device_cc() >= 8;
}

size_t ExpertMma::x8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (cols / 32) * 36;
}

void ExpertMma::quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows,
                         void* stream) {
    if (rows <= 0) return;
    expert_mma_quant_kernel<<<(unsigned) rows, 256, 0, (cudaStream_t) stream>>>(
        x, ids, (uint8_t*) xq, (int) cols, ld, (int64_t) (cols / 32) * 36);
    ck("expert mma quantize");
}

void ExpertMma::mul(const void* w, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
                    const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                    int64_t ld_dst, void* stream) {
    if (n <= 0 || max_rows <= 0) return;
    if (w_cols % 256 != 0) throw std::runtime_error("expert mma: w_cols is not a whole number of Q3_K blocks");
    const dim3 grid((unsigned) ((w_rows + BM - 1) / BM), (unsigned) ((max_rows + BN - 1) / BN), (unsigned) n);
    expert_mma_mul_kernel<<<grid, 256, 0, (cudaStream_t) stream>>>(
        (const uint8_t*) w, (int) w_rows, (int) w_cols, stride, (const uint8_t*) xq, bounds, ids, total_rows, dst,
        ld_dst);
    ck("expert mma product");
}

}  // namespace strata::poly

#endif   // __HIPCC__ || __gfx906__
