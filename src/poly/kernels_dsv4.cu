// src/poly/kernels_dsv4.cu - DeepSeek V4's kernels (kernels.hpp).  The reference for every formula is llama.cpp's
// deepseek4 graph (src/models/deepseek4.cpp) and its compressed caches (src/llama-kv-cache-dsv4.cpp).  Where a kernel
// takes several tokens, blockIdx.y is the token unless it says otherwise.
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

constexpr int WARP = 32;
constexpr int SELECT_THREADS = 1024;
constexpr int ATTEND_THREADS = 256;
constexpr int SCORE_KEYS = 32;      // dsv4_idx_score: the keys a block takes
constexpr int IDX_DIM = 128;        // an indexer key's values
constexpr int IDX_TOKENS = 16;      // idx_score_mma: the tokens a block scores, one mma tile
constexpr int IDX_KEYS = 128;       // idx_score_mma: the compressed rows a block takes, 2 tiles of 8 a warp
constexpr int GROUP_HEADS = 16;     // attend_rows_kernel: the heads of a token a block takes, a warp each.  16,
                                    // twice the first 8: a staged key row serves twice the heads before it is
                                    // dropped, and the 66 KiB of shared memory a block takes at dim 512 still
                                    // fits what the card lends a block (a 48 KiB card takes the fallback kernel)
constexpr int GROUP_KEYS = WARP;    // the keys it holds in shared memory at a time, a lane each
constexpr int GROUP_ROWS = 32;      // dsv4_attend takes attend_rows_kernel from this many tokens a call
constexpr int GROUP_DIM = 512;      // the longest head it takes
constexpr int ATTEND_BATCH = 2;     // attend_batch_kernel: the tokens a block attends together, a warp keeping a
                                    // head of each; one staged key row serves both tokens before it is dropped
constexpr int BATCH_KEYS = GROUP_KEYS + ATTEND_BATCH - 1;   // a round of staging: the union of their raw windows,
                                                            // one key longer than a window when both are full
constexpr int PAIR_PROBS = 2 * WARP;   // a pair warp's probability slot: two keys a lane scores, its e[0] at
                                       // [lane] and e[1] at [WARP + lane] - 64 floats, not BATCH_KEYS, or the
                                       // lanes' second keys spill into the warp behind

cudaStream_t cs(void* s) { return (cudaStream_t) s; }
int grid(int n, int b) { return (n + b - 1) / b; }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Lets `kernel` take as much dynamic shared memory as the current card gives a block, once a card; the most it may
// then take, in bytes.
template <class K>
size_t allow_smem(K kernel, unsigned& done) {
    int dev = 0, most = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess || most <= 1024)
        throw std::runtime_error("dsv4: the card's shared memory a block is unknown");
    const size_t room = size_t(most) - 1024;   // the kernels' own static shared memory
    if (dev < 32 && !(done & (1u << dev))) {
        if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(room)) != cudaSuccess)
            throw std::runtime_error("dsv4: a kernel cannot be given the card's shared memory");
        done |= 1u << dev;
    }
    return room;
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffff, x, o, WARP);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int o = WARP / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, o, WARP));
    return x;
}

// One value per thread summed (or the largest of them) over the block; `part` has a float a warp.
__device__ __forceinline__ float block_sum(float x, float* part) {
    x = warp_sum(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < int(blockDim.x) / WARP; ++k) s += part[k];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_max(float x, float* part) {
    x = warp_max(x);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = x;
    __syncthreads();
    float m = part[0];
    for (int k = 1; k < int(blockDim.x) / WARP; ++k) m = fmaxf(m, part[k]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ float half_at(const uint16_t* p, size_t i) {
    return __half2float(__ushort_as_half(p[i]));
}

__device__ __forceinline__ float q8_at(const int8_t* codes, const uint16_t* scales, size_t i) {
    return float(codes[i]) * __half2float(__ushort_as_half(scales[i / 32]));
}

// ggml-cuda's rope_yarn() and rope_yarn_ramp(), with the pair's index i0 counted from the first turned value.
__device__ __forceinline__ void yarn(float theta_extrap, const Rope& r, int i0, bool back, float& c, float& s) {
    const float theta_interp = r.freq_scale * theta_extrap;
    float theta = theta_interp, mscale = r.attn_factor;
    if (r.ext_factor != 0.0f) {
        const float y = (i0 / 2 - r.corr0) / fmaxf(0.001f, r.corr1 - r.corr0);
        const float ramp_mix = (1.0f - fminf(1.0f, fmaxf(0.0f, y))) * r.ext_factor;
        theta = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / r.freq_scale);
    }
    c = cosf(theta) * mscale;
    s = sinf(theta) * mscale;
    if (back) s = -s;
}

// Token t's position: what the card holds at `at` for it when the launch runs (a recorded launch), else
// pos + t * step.
__device__ __forceinline__ long long position(const long long* at, long long pos, int step, int t) {
    return at ? at[2 * t] : pos + (long long) t * step;
}

// A block a head of a token; a thread a pair of values (dim <= 2 * blockDim.x).
__global__ void rope_kernel(const float* __restrict__ x, float* y, int dim, int off, int n_rot, Rope r,
                            long long pos, int step, bool norm, float eps, bool back, const long long* at) {
    __shared__ float part[32];
    const size_t base = (size_t(blockIdx.y) * gridDim.x + blockIdx.x) * dim;
    const int i0 = 2 * threadIdx.x;
    float x0 = 0.0f, x1 = 0.0f;
    if (i0 < dim) {
        x0 = x[base + i0];
        x1 = x[base + i0 + 1];
    }
    float scale = 1.0f;
    if (norm) scale = 1.0f / sqrtf(block_sum(x0 * x0 + x1 * x1, part) / float(dim) + eps);
    if (i0 >= dim) return;
    x0 *= scale;
    x1 *= scale;
    if (i0 >= off && i0 < off + n_rot) {
        const int iw = i0 - off;
        const float p = float(position(at, pos, step, blockIdx.y));
        float c, s;
        yarn(p * powf(r.theta_scale, iw / 2.0f), r, iw, back, c, s);
        const float a = x0, b = x1;
        x0 = a * c - b * s;
        x1 = a * s + b * c;
    }
    y[base + i0] = x0;
    y[base + i0 + 1] = x1;
}

// A thread a value of a token's row.
__global__ void state_kernel(const float* __restrict__ kv, const float* __restrict__ score,
                             const float* __restrict__ ape, float* __restrict__ ring_kv, float* __restrict__ ring_score,
                             int ring, int width, int ratio, long long pos, const long long* at) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width) return;
    const long long p = position(at, pos, 1, blockIdx.y);
    const size_t src = size_t(blockIdx.y) * width + i, dst = size_t(p % ring) * width + i;
    ring_kv[dst] = kv[src];
    ring_score[dst] = score[src] + ape[size_t(p % ratio) * width + i];
}

// A block a token, a thread a value: nothing unless the token ends a block of ratio positions, else the block's
// compressed row.  Its values come in the order llama.cpp concatenates them, with overlap the rows before the block
// first; the weights are the softmax's, then the sum of weight times value in that order, as ggml's soft_max, mul
// and sum_rows give it.  Then the row is RMS-normed with its weights, turned by the block's first position on its
// last n_rot values, and stored as fp16 at the block's index.
__global__ void squeeze_kernel(const float* __restrict__ ring_kv, const float* __restrict__ ring_score, int ring,
                               int dim, int ratio, bool overlap, const float* __restrict__ norm, float eps, Rope r,
                               int n_rot, long long pos, const long long* at, uint16_t* __restrict__ cache) {
    __shared__ float part[32], vals[1024];
    const long long p = position(at, pos, 1, blockIdx.x);
    if ((p + 1) % ratio != 0) return;
    const int d = threadIdx.x;
    const long long start = p + 1 - ratio;
    const int width = overlap ? 2 * dim : dim, n = overlap ? 2 * ratio : ratio;
    // the j-th value: a ring row and a column, or nothing for a row before the sequence
    auto src = [&](int j, size_t& idx) {
        long long row;
        int col = d;
        if (!overlap) {
            row = start + j;
        } else if (j < ratio) {
            row = start - ratio + j;
        } else {
            row = start + j - ratio;
            col = dim + d;
        }
        if (row < 0) return false;
        idx = size_t(row % ring) * width + col;
        return true;
    };
    float mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) mx = fmaxf(mx, ring_score[idx]);
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) sum += expf(ring_score[idx] - mx);
    }
    const float inv = 1.0f / sum;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
        size_t idx;
        if (src(j, idx)) acc += expf(ring_score[idx] - mx) * inv * ring_kv[idx];
    }
    const float scale = 1.0f / sqrtf(block_sum(acc * acc, part) / float(dim) + eps);
    vals[d] = acc * scale * norm[d];
    __syncthreads();
    const int off = dim - n_rot;
    float v = vals[d];
    if (d >= off) {
        const int iw = (d - off) & ~1;
        float c, s;
        yarn(float(start) * powf(r.theta_scale, iw / 2.0f), r, iw, false, c, s);
        const float a = vals[off + iw], b = vals[off + iw + 1];
        v = (d - off) & 1 ? a * s + b * c : a * c - b * s;
    }
    cache[size_t(start / ratio) * dim + d] = __half_as_ushort(__float2half(v));
}

// A block a token and SCORE_KEYS keys; a warp a key at a time, a lane two heads of it.  The queries are in shared
// memory a head a row, one value longer than a row so the lanes' rows lie in different banks.
__global__ void idx_score_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                 const uint16_t* __restrict__ keys, float* __restrict__ score, int score_stride,
                                 int n_head, int ratio, long long pos, float scale, const long long* at) {
    extern __shared__ float sq[];   // n_head * (IDX_DIM + 1) queries, n_head weights, SCORE_KEYS * IDX_DIM keys
    float* sw = sq + n_head * (IDX_DIM + 1);
    float* sk = sw + n_head;
    const int t = blockIdx.y;
    const long long visible = (position(at, pos, 1, t) + 1) / ratio;
    const long long k0 = (long long) blockIdx.x * SCORE_KEYS;
    if (k0 >= visible) return;
    const float* qt = q + size_t(t) * n_head * IDX_DIM;
    for (int i = threadIdx.x; i < n_head * IDX_DIM; i += blockDim.x)
        sq[(i / IDX_DIM) * (IDX_DIM + 1) + i % IDX_DIM] = qt[i];
    for (int h = threadIdx.x; h < n_head; h += blockDim.x) sw[h] = w[size_t(t) * n_head + h];
    const int nk = (int) min((long long) SCORE_KEYS, visible - k0);
    for (int i = threadIdx.x; i < nk * IDX_DIM; i += blockDim.x) sk[i] = half_at(keys, size_t(k0) * IDX_DIM + i);
    __syncthreads();
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    for (int k = wid; k < nk; k += warps) {
        const float* key = sk + k * IDX_DIM;
        float acc = 0.0f;
        for (int h = lane; h < n_head; h += WARP) {
            const float* qh = sq + h * (IDX_DIM + 1);
            float dot = 0.0f;
#pragma unroll 8
            for (int i = 0; i < IDX_DIM; ++i) dot += qh[i] * key[i];
            acc += fmaxf(dot, 0.0f) * sw[h];
        }
        acc = warp_sum(acc);
        if (lane == 0) score[size_t(t) * score_stride + k0 + k] = acc * scale;
    }
}

#if !defined(GGML_USE_HIP)

// mma.sync of fattn.cu: 16 rows by 8 columns of the two 8-half fragments, the sums in floats.
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// The indexer's scores on the tensor cores: a block scores IDX_TOKENS tokens against IDX_KEYS compressed rows, a
// warp 2 tiles of 8 rows, a lane one key of a tile.  The keys are read once a block and kept in the registers; per
// head the block stages the head's 16 rows of queries as halves in shared memory and runs each 128-value dot as 8
// mma steps, then folds the head's relu and weight into the running sums.  A row's arithmetic never depends on the
// other rows, and rows past a token's visible count are computed but never written, so a token's scores come out
// the same whatever tokens the launch has beside it.  The fragments are numbered as fdec_kernel's (fattn.cu): the
// queries a lane's 2 rows of 4 halves a step, the keys a lane's 8 halves a 32-value block, c[2 * r + j] the row
// g8 + 8r and column 2t + j.
__global__ void idx_score_mma_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                     const uint16_t* __restrict__ keys, float* __restrict__ score,
                                     int score_stride, int rows, int n_head, int ratio, long long pos, float scale,
                                     const long long* __restrict__ at) {
    constexpr int QS = IDX_DIM + 8;    // a row one value longer, so the lanes' rows land in different banks
    extern __shared__ __align__(16) char sm[];
    half* sq = reinterpret_cast<half*>(sm);   // 16 * QS queries, a head at a time
    float* sw = reinterpret_cast<float*>(sm + IDX_TOKENS * QS * sizeof(half));   // 16 * n_head weights
    const int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP, g8 = lane / 4, t4 = lane % 4;
    const int t0 = IDX_TOKENS * blockIdx.y;
    const long long kb = (long long) blockIdx.x * IDX_KEYS;
    const int in_tile = rows - t0 < IDX_TOKENS ? rows - t0 : IDX_TOKENS;
    const long long visible = (position(at, pos, 1, t0 + in_tile - 1) + 1) / ratio;
    if (kb >= visible) return;
    // the warp's 16 keys: 2 tiles of 8, a lane its key's 8 values of each 32-value block, the 4 blocks at once
    uint4 kx[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const long long key = kb + 16 * wid + 8 * nt + g8;
#pragma unroll
        for (int b = 0; b < 4; ++b)
            kx[nt][b] = key >= visible ? make_uint4(0u, 0u, 0u, 0u)
                      : *reinterpret_cast<const uint4*>(keys + (size_t) key * IDX_DIM + 32 * b + 8 * t4);
    }
    for (int i = threadIdx.x; i < IDX_TOKENS * n_head; i += blockDim.x) {
        const int row = i / n_head, h = i % n_head, tok = t0 + row;
        sw[i] = tok < rows ? w[(size_t) tok * n_head + h] : 0.f;
    }
    const int tok[2] = {t0 + g8, t0 + g8 + 8};   // the two rows this lane folds and writes
    const long long vis[2] = {tok[0] < rows ? (position(at, pos, 1, tok[0]) + 1) / ratio : 0,
                              tok[1] < rows ? (position(at, pos, 1, tok[1]) + 1) / ratio : 0};
    __syncthreads();
    float s[2][4];   // the relu'd, weighted dots summed over the heads
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
    for (int h = 0; h < n_head; ++h) {
        for (int i = threadIdx.x; i < IDX_TOKENS * IDX_DIM; i += blockDim.x) {
            const int row = i / IDX_DIM, d = i % IDX_DIM, tk = t0 + row;
            sq[row * QS + d] = tk < rows ? __float2half(q[((size_t) tk * n_head + h) * IDX_DIM + d])
                                         : __float2half(0.f);
        }
        __syncthreads();
        float c[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) c[nt][0] = c[nt][1] = c[nt][2] = c[nt][3] = 0.f;
#pragma unroll
        for (int b = 0; b < 4; ++b)
#pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int d = 32 * b + 8 * t4 + 4 * hh;   // the thread's 4 dimensions of the step
                const uint2 lo = *reinterpret_cast<const uint2*>(&sq[g8 * QS + d]);
                const uint2 hi = *reinterpret_cast<const uint2*>(&sq[(g8 + 8) * QS + d]);
                const uint32_t a[4] = {lo.x, hi.x, lo.y, hi.y};
#pragma unroll
                for (int nt = 0; nt < 2; ++nt) {
                    const uint32_t bk0 = hh ? kx[nt][b].z : kx[nt][b].x, bk1 = hh ? kx[nt][b].w : kx[nt][b].y;
                    mma16816(c[nt], a, bk0, bk1);
                }
            }
        const float wr0 = sw[g8 * n_head + h], wr1 = sw[(g8 + 8) * n_head + h];
#pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            s[nt][0] += fmaxf(c[nt][0], 0.f) * wr0;
            s[nt][1] += fmaxf(c[nt][1], 0.f) * wr0;
            s[nt][2] += fmaxf(c[nt][2], 0.f) * wr1;
            s[nt][3] += fmaxf(c[nt][3], 0.f) * wr1;
        }
        __syncthreads();   // sq is the next head's
    }
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int r = j >> 1;
            const long long key = kb + 16 * wid + 8 * nt + 2 * t4 + (j & 1);
            if (key < vis[r]) score[(size_t) tok[r] * score_stride + key] = s[nt][j] * scale;
        }
}

#endif   // !GGML_USE_HIP

__device__ __forceinline__ uint32_t order_key(float s) {
    const uint32_t b = __float_as_uint(s + 0.0f);
    return b & 0x80000000u ? ~b : b | 0x80000000u;
}

// x summed over the threads before this one, and over all of them in *all.  part has a slot a warp.
__device__ __forceinline__ int block_before(int x, int* part, int* all) {
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    int in = x;
#pragma unroll
    for (int o = 1; o < WARP; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffff, in, o, WARP);
        if (lane >= o) in += v;
    }
    if (lane == WARP - 1) part[wid] = in;
    __syncthreads();
    int before = 0, total = 0;
    for (int k = 0; k < warps; ++k) {
        if (k < wid) before += part[k];
        total += part[k];
    }
    __syncthreads();
    *all = total;
    return before + in - x;
}

// A block a token.  The n_top blocks with the highest scores by a radix search for the n_top-th score, eight bits a
// round, then the blocks above it and of those on it the lowest, written in the order of the blocks.  A token that
// sees n_top blocks or fewer takes them all.
__global__ void select_kernel(const float* __restrict__ score, int score_stride, int32_t* __restrict__ sel,
                              int sel_stride, int n_top, int ratio, long long pos, const long long* at) {
    __shared__ int hist[SELECT_THREADS / WARP][256], part[SELECT_THREADS / WARP], pick[2];
    const int n = int((position(at, pos, 1, blockIdx.x) + 1) / ratio);
    int32_t* out = sel + size_t(blockIdx.x) * sel_stride;
    if (n <= n_top) {
        for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = i;
        return;
    }
    const float* s = score + size_t(blockIdx.x) * score_stride;
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    uint32_t cut = 0, mask = 0;
    int want = n_top;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int d = lane; d < 256; d += WARP) hist[wid][d] = 0;
        __syncwarp();
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t k = order_key(s[i]);
            if ((k & mask) == cut) atomicAdd(&hist[wid][(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (threadIdx.x < 256) {
            int t = 0;
            for (int k = 0; k < SELECT_THREADS / WARP; ++k) t += hist[k][threadIdx.x];
            hist[0][threadIdx.x] = t;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int d = 255, above = 0;
            for (; d > 0 && above + hist[0][d] < want; --d) above += hist[0][d];
            pick[0] = d;
            pick[1] = above;
        }
        __syncthreads();
        cut |= uint32_t(pick[0]) << shift;
        mask |= 255u << shift;
        want -= pick[1];
        __syncthreads();
    }
    const int per = (n + int(blockDim.x) - 1) / int(blockDim.x);
    const int lo = threadIdx.x * per, hi = min(lo + per, n);
    int g = 0, e = 0;
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        g += k > cut;
        e += k == cut;
    }
    int g_all, e_all;
    const int g0 = block_before(g, part, &g_all);
    int e0 = block_before(e, part, &e_all);
    const int need = n_top - g_all;
    int o = g0 + min(e0, need);
    for (int i = lo; i < hi; ++i) {
        const uint32_t k = order_key(s[i]);
        if (k > cut || (k == cut && e0++ < need)) out[o++] = i;
    }
}

// A block a head of a token.  Its keys are the raw rows of its window, oldest first, then its compressed rows; a
// key is its value too.  Scores, the softmax with the head's sink, then the weighted sum of the keys.
__global__ void attend_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring, int window,
                              const uint16_t* __restrict__ comp, int ratio, const int32_t* __restrict__ sel,
                              int sel_stride, int top, const float* __restrict__ sinks, float* __restrict__ out,
                              int dim, int n_head, float scale, long long pos, const long long* at,
                              const int8_t* k_codes, const uint16_t* k_scales) {
    extern __shared__ float smem[];   // dim query values, then a score a key, then a key's row
    __shared__ float part[32];
    float* sq = smem;
    float* sc = smem + dim;
    const int h = blockIdx.x, t = blockIdx.y;
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int) min((long long) window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int) min(visible, (long long) top) : (int) visible;
    }
    const int n_key = n_raw + n_comp;
    int* row = (int*) (sc + n_key);
    const float* qh = q + (size_t(t) * n_head + h) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) sq[i] = qh[i];
    for (int k = threadIdx.x; k < n_key; k += blockDim.x)
        row[k] = k < n_raw ? int((first_raw + k) % ring) : sel ? sel[size_t(t) * sel_stride + k - n_raw] : k - n_raw;
    __syncthreads();
    const bool q8 = k_codes != nullptr;
    auto key_val = [&](int k, int i) -> float {
        const int r = row[k];
        if (k < n_raw) {
            return q8 ? q8_at(k_codes + (size_t) r * dim, k_scales + (size_t) r * (dim / 32), i)
                      : half_at(raw + (size_t) r * dim, i);
        } else {
            return half_at(comp + (size_t) r * dim, i);
        }
    };
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP, warps = blockDim.x / WARP;
    for (int k = wid; k < n_key; k += warps) {
        float dot = 0.0f;
        for (int i = lane; i < dim; i += WARP) dot += sq[i] * key_val(k, i);
        dot = warp_sum(dot);
        if (lane == 0) sc[k] = dot * scale;
    }
    __syncthreads();
    const float sink = sinks[h];
    float mx = sink;
    for (int k = threadIdx.x; k < n_key; k += blockDim.x) mx = fmaxf(mx, sc[k]);
    mx = block_max(mx, part);
    float sum = 0.0f;
    for (int k = threadIdx.x; k < n_key; k += blockDim.x) {
        const float e = expf(sc[k] - mx);
        sc[k] = e;
        sum += e;
    }
    sum = block_sum(sum, part) + expf(sink - mx);
    const float inv = 1.0f / sum;
    float* oh = out + (size_t(t) * n_head + h) * dim;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < n_key; ++k) acc += sc[k] * key_val(k, i);
        oh[i] = acc * inv;
    }
}

// attend_kernel for a batch of tokens.  The heads share their one key row, so a block takes GH heads of a token, a
// warp each, and reads the keys GROUP_KEYS at a time into shared memory once for all of them; the softmax runs
// along the keys, from the sink.  attend_kernel read every key twice for every head.  GH is what --group-heads
// names (16, twice the 8 the kernel first took; the launcher takes either).
template <int GH>
__global__ void attend_rows_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring,
                                   int window, const uint16_t* __restrict__ comp, int ratio,
                                   const int32_t* __restrict__ sel, int sel_stride, int top,
                                   const float* __restrict__ sinks, float* __restrict__ out, int dim, int n_head,
                                   float scale, long long pos, const long long* at,
                                   const int8_t* k_codes, const uint16_t* k_scales, int q8_fast) {
    // GH rows of query values; GROUP_KEYS rows of keys, a word a pair of halves and four words more a row: read 16
    // bytes at a time, the 8 lanes of a quarter warp then reach every bank once, along a row or down a column; a
    // probability a head and key
    extern __shared__ float smem[];
    __shared__ int row[GROUP_KEYS];
    const int pitch = dim / 2 + 4;
    float* sq = smem;
    uint32_t* sk = (uint32_t*) (sq + GH * dim);
    float* sp = (float*) (sk + GROUP_KEYS * pitch);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int h = blockIdx.x * GH + w, t = blockIdx.y;
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int) min((long long) window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int) min(visible, (long long) top) : (int) visible;
    }
    const int n_key = n_raw + n_comp;
    for (int i = threadIdx.x; i < GH * dim; i += blockDim.x) {
        const int hh = blockIdx.x * GH + i / dim;
        sq[i] = hh < n_head ? q[(size_t(t) * n_head + hh) * dim + i % dim] : 0.0f;
    }
    // this lane's part of the head's output: values 8 lane to 8 lane + 7 of every 256
    float o[GROUP_DIM / WARP] = {};
    const int chunks = dim / (8 * WARP);
    float m = h < n_head ? sinks[h] : 0.0f, l = 1.0f;
    const bool q8 = k_codes != nullptr;
    for (int k0 = 0; k0 < n_key; k0 += GROUP_KEYS) {
        const int nk = min(GROUP_KEYS, n_key - k0);
        __syncthreads();   // the keys before are read
        if (threadIdx.x < GROUP_KEYS) {
            const int k = k0 + threadIdx.x;
            row[threadIdx.x] = k >= k0 + nk ? 0
                               : k < n_raw  ? int((first_raw + k) % ring)
                               : sel        ? sel[size_t(t) * sel_stride + k - n_raw]
                                            : k - n_raw;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < GROUP_KEYS * (dim / 8); i += blockDim.x) {
            const int kk = i / (dim / 8), c = i % (dim / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (kk < nk) {
                const int r = row[kk];
                if (k0 + kk < n_raw && q8) {
                    if (q8_fast) {
                        // q8_0, dequantized into the shared row: the 8 values of a 16-byte word start at a
                        // multiple of 8 and never leave their one block of 32, so one scale serves them all; a
                        // half2 is packed a pair, and the 8 codes behind them are the L1's consecutive bytes
                        // (--q8-dequant 1)
                        const float s = __half2float(
                            __ushort_as_half(k_scales[(size_t) r * (dim / 32) + (c >> 2)]));
                        const int8_t* cd = k_codes + (size_t) r * dim + c * 8;
                        half2* dst = (half2*) &v;
                        dst[0] = __floats2half2_rn(float(cd[0]) * s, float(cd[1]) * s);
                        dst[1] = __floats2half2_rn(float(cd[2]) * s, float(cd[3]) * s);
                        dst[2] = __floats2half2_rn(float(cd[4]) * s, float(cd[5]) * s);
                        dst[3] = __floats2half2_rn(float(cd[6]) * s, float(cd[7]) * s);
                    } else {
                        // a value at a time, the row's scale fetched for each (--q8-dequant 0)
                        const int8_t* codes = k_codes + (size_t) r * dim;
                        const uint16_t* scales = k_scales + (size_t) r * (dim / 32);
                        half* dst = (half*) &v;
                        for (int j = 0; j < 8; ++j) {
                            int idx = c * 8 + j;
                            if (idx < dim) {
                                dst[j] = __float2half(float(codes[idx]) *
                                                      __half2float(__ushort_as_half(scales[idx / 32])));
                            }
                        }
                    }
                } else if (k0 + kk < n_raw) {
                    v = ((const uint4*) (raw + (size_t) r * dim))[c];
                } else {
                    v = ((const uint4*) (comp + (size_t) r * dim))[c];
                }
            }
            *(uint4*) &sk[kk * pitch + 4 * c] = v;
        }
        __syncthreads();
        float s = -INFINITY;
        if (lane < nk) {
            const uint4* kr = (const uint4*) (sk + lane * pitch);
            const float4* qh = (const float4*) (sq + w * dim);
            float d0 = 0.0f, d1 = 0.0f;
            for (int c = 0; c < dim / 8; ++c) {
                const uint4 kv = kr[c];
                const float4 qa = qh[2 * c], qb = qh[2 * c + 1];
                const float2 a = __half22float2(*(const __half2*) &kv.x), b = __half22float2(*(const __half2*) &kv.y);
                const float2 e = __half22float2(*(const __half2*) &kv.z), f = __half22float2(*(const __half2*) &kv.w);
                d0 += qa.x * a.x + qa.y * a.y + qa.z * b.x + qa.w * b.y;
                d1 += qb.x * e.x + qb.y * e.y + qb.z * f.x + qb.w * f.y;
            }
            s = (d0 + d1) * scale;
        }
        const float mn = fmaxf(m, warp_max(s));
        const float alpha = expf(m - mn), e = lane < nk ? expf(s - mn) : 0.0f;
        l = l * alpha + warp_sum(e);
        m = mn;
        sp[w * GROUP_KEYS + lane] = e;
        __syncwarp();
#pragma unroll
        for (int j = 0; j < GROUP_DIM / WARP; ++j) o[j] *= alpha;
        for (int k = 0; k < nk; ++k) {
            const float pk = sp[w * GROUP_KEYS + k];
            const uint4* kr = (const uint4*) (sk + k * pitch) + lane;
#pragma unroll
            for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)   // whole, so that o stays in registers
                if (j < chunks) {
                    const uint4 v = kr[WARP * j];
                    const float2 a = __half22float2(*(const __half2*) &v.x), b = __half22float2(*(const __half2*) &v.y);
                    const float2 c = __half22float2(*(const __half2*) &v.z), d = __half22float2(*(const __half2*) &v.w);
                    float* oj = o + 8 * j;
                    oj[0] += pk * a.x, oj[1] += pk * a.y, oj[2] += pk * b.x, oj[3] += pk * b.y;
                    oj[4] += pk * c.x, oj[5] += pk * c.y, oj[6] += pk * d.x, oj[7] += pk * d.y;
                }
        }
        __syncwarp();   // sp is written again for the next keys
    }
    if (h >= n_head) return;
    float* oh = out + (size_t(t) * n_head + h) * dim + 8 * lane;
    const float inv = 1.0f / l;
#pragma unroll
    for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)
        if (j < chunks) {
            const float* oj = o + 8 * j;
            *(float4*) &oh[8 * WARP * j] = make_float4(oj[0] * inv, oj[1] * inv, oj[2] * inv, oj[3] * inv);
            *(float4*) &oh[8 * WARP * j + 4] = make_float4(oj[4] * inv, oj[5] * inv, oj[6] * inv, oj[7] * inv);
        }
}

#if !defined(GGML_USE_HIP)

// Two helpers of fdec_kernel (fattn.cu): a q8_0 code pair as exact halves, and a half2 scaled in place.
__device__ __forceinline__ uint32_t i8pair(uint32_t w, uint32_t sel) {
    uint32_t x = __byte_perm(w, 0x64646464u, sel);
    half2 h = __hsub2(*reinterpret_cast<half2*>(&x), __float2half2_rn(1152.0f));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t mul2(uint32_t x, half2 s) {
    half2 h = __hmul2(*reinterpret_cast<half2*>(&x), s);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t word_of(const uint4& v, int i) {
    return i < 2 ? (i == 0 ? v.x : v.y) : (i == 2 ? v.z : v.w);
}

__device__ __forceinline__ uint4 widen2(uint2 v) { return make_uint4(v.x, v.y, 0u, 0u); }

// attend_rows_kernel on the tensor cores, fdec_kernel's mechanics over a prompt batch (fattn.cu): a block keeps
// GROUP_HEADS heads of one token - the mma's M tile, the heads share the keys, this attention is MQA - and a warp
// scores 8 of the 64 keys a phase takes: S = Q K^T, mma m16n8k16, the q8 codes exact as halves with each
// 32-value block's scale applied to its sum in floats, the fp16 rows as they lie.  The softmax goes per head
// through red[] across the warps, and O += P V with P as halves of the shared memory and V of the warp's own
// dimensions read straight from the caches, a raw q8 key's values unpacking by i8pair with the block's scale as
// mul2 on the way.  A key is read once a block, so the keys are not staged at all: the shared memory holds the
// queries (halves), the probabilities and the reductions, about 20 KiB against attend_rows_kernel's 67.
template <int DIM>
__global__ void __launch_bounds__(8 * WARP)
attend_rows_mma_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring, int window,
                       const uint16_t* __restrict__ comp, int ratio, const int32_t* __restrict__ sel,
                       int sel_stride, int top, const float* __restrict__ sinks, float* __restrict__ out,
                       int n_head, float scale, long long pos, const long long* at,
                       const int8_t* k_codes, const uint16_t* k_scales) {
    constexpr int QS = GROUP_DIM + 8, PS = 64 + 8, NW = 8, DW = DIM / NW;
    extern __shared__ float smem[];
    half* sq = reinterpret_cast<half*>(smem);                  // GROUP_HEADS * QS queries, halves
    half* sp = reinterpret_cast<half*>(sq + GROUP_HEADS * QS); // GROUP_HEADS * PS probabilities
    float* red = reinterpret_cast<float*>(sp + GROUP_HEADS * PS);   // 2 * NW * GROUP_HEADS: max, sum a phase
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP, g8 = lane / 4, t4 = lane % 4;
    const int h0 = blockIdx.x * GROUP_HEADS, t = blockIdx.y;
    const long long p = position(at, pos, 1, t);
    const int n_raw = (int) min((long long) window, p + 1);
    const long long first_raw = p + 1 - n_raw;
    int n_comp = 0;
    if (comp) {
        const long long visible = (p + 1) / ratio;
        n_comp = sel ? (int) min(visible, (long long) top) : (int) visible;
    }
    const int n_key = n_raw + n_comp;
    const bool q8 = k_codes != nullptr;
    for (int i = threadIdx.x; i < GROUP_HEADS * DIM; i += blockDim.x) {
        const int hh = i / DIM, d = i % DIM, h = h0 + hh;
        sq[hh * QS + d] = __float2half(h < n_head ? q[(size_t(t) * n_head + h) * DIM + d] : 0.f);
    }
    float o[NW][4] = {};                     // this warp's DW values; o[j][2r + c] is row g8 + 8r, c-th of the tile
    float m[2], l[2] = {1.f, 1.f};           // the two rows (heads) this lane keeps
    #pragma unroll
    for (int r = 0; r < 2; ++r) m[r] = h0 + g8 + 8 * r < n_head ? sinks[h0 + g8 + 8 * r] : 0.f;
    __syncthreads();
    for (int k0 = 0; k0 < n_key; k0 += 64) {
        const int nk = min(64, n_key - k0);
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        for (int b = 0; b < DIM / 32; ++b) {
            float c[2][4] = {};
            const int key = k0 + 8 * w + g8;   // this lane's key
            #pragma unroll
            for (int hh = 0; hh < 2; ++hh) {
                const int d = 32 * b + 8 * t4 + 4 * hh;
                const uint2 lo = *reinterpret_cast<const uint2*>(&sq[g8 * QS + d]);
                const uint2 hi = *reinterpret_cast<const uint2*>(&sq[(g8 + 8) * QS + d]);
                const uint32_t a[4] = {lo.x, hi.x, lo.y, hi.y};
                uint32_t b0 = 0, b1 = 0;
                if (key < nk) {
                    if (key < n_raw && q8) {
                        const uint2 w8 = *reinterpret_cast<const uint2*>(k_codes + (size_t) ((first_raw + key) % ring) * DIM + 32 * b + 8 * t4);
                        const uint32_t word = (hh ? w8.y : w8.x) ^ 0x80808080u;
                        b0 = i8pair(word, 0x4140); b1 = i8pair(word, 0x4342);
                    } else if (key < n_raw) {
                        const uint4 v = *reinterpret_cast<const uint4*>(raw + (size_t) ((first_raw + key) % ring) * DIM + 32 * b + 8 * t4);
                        b0 = hh ? v.z : v.x; b1 = hh ? v.w : v.y;
                    } else {
                        const int r = sel ? sel[size_t(t) * sel_stride + key - n_raw] : key - n_raw;
                        const uint4 v = *reinterpret_cast<const uint4*>(comp + (size_t) r * DIM + 32 * b + 8 * t4);
                        b0 = hh ? v.z : v.x; b1 = hh ? v.w : v.y;
                    }
                }
                mma16816(c[hh], a, b0, b1);
            }
            float sc[2] = {1.f, 1.f};   // the two columns' keys' scales of this 32-value block
            if (q8) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int key = k0 + 8 * w + 2 * t4 + j;
                    if (key < n_raw) sc[j] = __half2float(__ushort_as_half(k_scales[(size_t) ((first_raw + key) % ring) * (DIM / 32) + b]));
                }
            }
            // the C fragment is c[hh][0] = row g8, col 2t4; c[hh][1] = row g8, col 2t4+1; c[hh][2] = row g8+8,
            // col 2t4; c[hh][3] = row g8+8, col 2t4+1 - the row lives in the second index's high bit, hh is the
            // dimension half, so a head's score adds the two halves of the SAME (row, column)
            s[0] += (c[0][0] + c[1][0]) * sc[0];   // head g8,   key 2t4
            s[1] += (c[0][1] + c[1][1]) * sc[1];   // head g8,   key 2t4+1
            s[2] += (c[0][2] + c[1][2]) * sc[0];   // head g8+8, key 2t4
            s[3] += (c[0][3] + c[1][3]) * sc[1];   // head g8+8, key 2t4+1
        }
        // the phase's max per head over the eight warps, the probabilities, the rescale.  The masks are the lane's
        // two keys in the whole phase (8w + 2t4, +1), not the tile's column: a warp past nk scores no keys, and a
        // column-only mask would let exp(0 - top) of its zero scores into the sum
        const int kc = 8 * w + 2 * t4;
        float mx[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            float x = fmaxf(kc < nk ? s[2 * r] * scale : -INFINITY,
                            kc + 1 < nk ? s[2 * r + 1] * scale : -INFINITY);
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 1));
            mx[r] = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 2));
        }
        if (t4 == 0) {
            red[w * GROUP_HEADS + g8] = mx[0];
            red[w * GROUP_HEADS + g8 + 8] = mx[1];
        }
        __syncthreads();
        float alpha[2], top[2];
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g8 + 8 * r;
            float tp = m[r];
            for (int k = 0; k < NW; ++k) tp = fmaxf(tp, red[k * GROUP_HEADS + row]);
            top[r] = tp;
            alpha[r] = tp == -INFINITY ? 1.f : expf(m[r] - tp);
            m[r] = tp;
        }
        float sum[2] = {0.f, 0.f};
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g8 + 8 * r;
            const float x0 = kc < nk ? expf(s[2 * r] * scale - top[r]) : 0.f;
            const float x1 = kc + 1 < nk ? expf(s[2 * r + 1] * scale - top[r]) : 0.f;
            *reinterpret_cast<half2*>(&sp[row * PS + 8 * w + 2 * t4]) = __floats2half2_rn(x0, x1);
            sum[r] = x0 + x1;
            #pragma unroll
            for (int j = 0; j < NW; ++j) { o[j][2 * r] *= alpha[r]; o[j][2 * r + 1] *= alpha[r]; }
        }
        sum[0] += __shfl_xor_sync(0xffffffffu, sum[0], 1);
        sum[0] += __shfl_xor_sync(0xffffffffu, sum[0], 2);
        sum[1] += __shfl_xor_sync(0xffffffffu, sum[1], 1);
        sum[1] += __shfl_xor_sync(0xffffffffu, sum[1], 2);
        if (t4 == 0) {
            red[NW * GROUP_HEADS + w * GROUP_HEADS + g8] = sum[0];
            red[NW * GROUP_HEADS + w * GROUP_HEADS + g8 + 8] = sum[1];
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = g8 + 8 * r;
            float sum = 0.f;
            for (int k = 0; k < NW; ++k) sum += red[NW * GROUP_HEADS + k * GROUP_HEADS + row];
            l[r] = l[r] * alpha[r] + sum;
        }
        // O += P V: k-step kk is phase keys 16 kk.., n-tile j is dimension DW * w + 8 * g8 + j; the raw q8 keys'
        // values unpack by i8pair with their block's scale as mul2, the fp16 rows as they lie
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const half* pp = &sp[g8 * PS + 16 * kk + 2 * t4];
            const uint32_t a[4] = {*reinterpret_cast<const uint32_t*>(pp),
                                   *reinterpret_cast<const uint32_t*>(pp + 8 * PS),
                                   *reinterpret_cast<const uint32_t*>(pp + 8),
                                   *reinterpret_cast<const uint32_t*>(pp + 8 * PS + 8)};
            uint4 vx[4];
            uint16_t vsc[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int key = k0 + 16 * kk + 2 * t4 + (i & 1) + 8 * (i >> 1);
                vx[i] = make_uint4(0u, 0u, 0u, 0u); vsc[i] = 0x3c00;   // half 1
                if (key < nk) {
                    if (key < n_raw && q8) {
                        vx[i] = widen2(*reinterpret_cast<const uint2*>(k_codes + (size_t) ((first_raw + key) % ring) * DIM + DW * w + 8 * g8));
                        vsc[i] = k_scales[(size_t) ((first_raw + key) % ring) * (DIM / 32) + (DW * w + 8 * g8) / 32];
                    } else if (key < n_raw) {
                        vx[i] = *reinterpret_cast<const uint4*>(raw + (size_t) ((first_raw + key) % ring) * DIM + DW * w + 8 * g8);
                    } else {
                        const int r = sel ? sel[size_t(t) * sel_stride + key - n_raw] : key - n_raw;
                        vx[i] = *reinterpret_cast<const uint4*>(comp + (size_t) r * DIM + DW * w + 8 * g8);
                    }
                }
            }
            if (q8) {
                const half2 s01 = __halves2half2(__ushort_as_half(vsc[0]), __ushort_as_half(vsc[1]));
                const half2 s89 = __halves2half2(__ushort_as_half(vsc[2]), __ushort_as_half(vsc[3]));
                #pragma unroll
                for (int wd = 0; wd < 2; ++wd) {
                    const uint32_t x0 = (wd ? vx[0].y : vx[0].x) ^ 0x80808080u;
                    const uint32_t x1 = (wd ? vx[1].y : vx[1].x) ^ 0x80808080u;
                    const uint32_t x2 = (wd ? vx[2].y : vx[2].x) ^ 0x80808080u;
                    const uint32_t x3 = (wd ? vx[3].y : vx[3].x) ^ 0x80808080u;
                    const uint32_t lo01 = __byte_perm(x0, x1, 0x5140), hi01 = __byte_perm(x0, x1, 0x7362);
                    const uint32_t lo89 = __byte_perm(x2, x3, 0x5140), hi89 = __byte_perm(x2, x3, 0x7362);
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t selb = jj & 1 ? 0x4342 : 0x4140;
                        const uint32_t b0 = mul2(i8pair(jj < 2 ? lo01 : hi01, selb), s01);
                        const uint32_t b1 = mul2(i8pair(jj < 2 ? lo89 : hi89, selb), s89);
                        #pragma unroll
                        for (int j = 0; j < NW; ++j) mma16816(o[4 * wd + jj], a, b0, b1);
                    }
                }
                continue;
            }
            #pragma unroll
            for (int j = 0; j < NW; ++j) {
                const uint32_t selb = j & 1 ? 0x7632 : 0x5410;
                const uint32_t b0 = __byte_perm(word_of(vx[0], j / 2), word_of(vx[1], j / 2), selb);
                const uint32_t b1 = __byte_perm(word_of(vx[2], j / 2), word_of(vx[3], j / 2), selb);
                mma16816(o[j], a, b0, b1);
            }
        }
        __syncthreads();   // sp and red are the next phase's
    }
    if (h0 + g8 >= n_head) return;
    const float inv[2] = {1.f / l[0], 1.f / l[1]};
    #pragma unroll
    for (int r = 0; r < 2; ++r) {
        float4* dst = reinterpret_cast<float4*>(out + ((size_t) t * n_head + h0 + g8 + 8 * r) * DIM + DW * w + 16 * t4);
        dst[0] = make_float4(o[0][2 * r] * inv[r], o[1][2 * r] * inv[r], o[2][2 * r] * inv[r], o[3][2 * r] * inv[r]);
        dst[1] = make_float4(o[4][2 * r] * inv[r], o[5][2 * r] * inv[r], o[6][2 * r] * inv[r], o[7][2 * r] * inv[r]);
        dst[2] = make_float4(o[0][2 * r + 1] * inv[r], o[1][2 * r + 1] * inv[r], o[2][2 * r + 1] * inv[r], o[3][2 * r + 1] * inv[r]);
        dst[3] = make_float4(o[4][2 * r + 1] * inv[r], o[5][2 * r + 1] * inv[r], o[6][2 * r + 1] * inv[r], o[7][2 * r + 1] * inv[r]);
    }
}

#endif   // !GGML_USE_HIP

// attend_rows_kernel for pairs of tokens: a warp keeps a head of both, and the raw keys a block stages are the two
// windows' union - the second token's window is the first's moved one key on - so a prompt batch reads the keys about
// half as often, the very traffic its attention time is.  The queries the warps read straight from q, a warp one row,
// the lanes of it the same addresses: L1 serves them as one read.  A token sees the keys of its own window alone and
// scores each with attend_rows_kernel's dot; the softmax steps fall where the union's rounds put them, not where a
// token's own rounds would, so a sum differs from a token-a-block's in its last bits - the tests' 1e-3 takes both.
// The compressed rows each token's indexer picked it scores in rounds of its own, staged for that token alone.  The
// queries need no shared memory, so a block takes less of it than attend_rows_kernel's, not more.
__global__ void attend_batch_kernel(const float* __restrict__ q, const uint16_t* __restrict__ raw, int ring,
                                    int window, const uint16_t* __restrict__ comp, int ratio,
                                    const int32_t* __restrict__ sel, int sel_stride, int top,
                                    const float* __restrict__ sinks, float* __restrict__ out, int dim, int n_head,
                                    float scale, long long pos, int rows, const long long* at,
                                    const int8_t* k_codes, const uint16_t* k_scales, int q8_fast) {
    // BATCH_KEYS staged key rows, a word a pair of halves and four words more a row (attend_rows_kernel's layout);
    // a probability a warp's scored key, PAIR_PROBS a warp (two keys a lane), of a token
    extern __shared__ float smem[];
    __shared__ int row[BATCH_KEYS];
    const int pitch = dim / 2 + 4;
    uint32_t* sk = (uint32_t*) smem;
    float* sp = (float*) (sk + size_t(BATCH_KEYS) * pitch);
    const int lane = threadIdx.x % WARP, w = threadIdx.x / WARP;
    const int h = blockIdx.x * GROUP_HEADS + w;
    const int t0 = ATTEND_BATCH * blockIdx.y;
    const bool q8 = k_codes != nullptr;

    // the tokens, their places and their raw windows; the second of a pair may be past the batch's end
    long long p[ATTEND_BATCH];
    int n_raw[ATTEND_BATCH];
    for (int t = 0; t < ATTEND_BATCH; ++t) {
        const bool ok = t0 + t < rows;
        p[t] = ok ? position(at, pos, 1, t0 + t) : 0;
        n_raw[t] = ok ? (int) min((long long) window, p[t] + 1) : 0;
    }
    // the union runs from the earlier window's start to the later token's place; a token's own keys sit behind its
    // window's start, at from[t]
    long long first = p[0] + 1 - n_raw[0];
    int n_union = 0, from[ATTEND_BATCH];
    for (int t = 0; t < ATTEND_BATCH; ++t) {
        from[t] = (int) (p[t] + 1 - n_raw[t] - first);
        if (n_raw[t] > 0) n_union = max(n_union, from[t] + n_raw[t]);   // a token past the batch's end widens
                                                                         // the union by nothing
    }

    // this lane's part of each token's head's output: values 8 lane to 8 lane + 7 of every 256
    float o[ATTEND_BATCH][GROUP_DIM / WARP] = {};
    const int chunks = dim / (8 * WARP);
    float m[ATTEND_BATCH], l[ATTEND_BATCH];
    for (int t = 0; t < ATTEND_BATCH; ++t) {
        m[t] = h < n_head ? sinks[h] : 0.0f;
        l[t] = 1.0f;
    }

    for (int j0 = 0; j0 < n_union; j0 += BATCH_KEYS) {
        const int nk = min(BATCH_KEYS, n_union - j0);
        __syncthreads();   // the keys before are read
        if (threadIdx.x < BATCH_KEYS) {
            const int j = j0 + threadIdx.x;
            row[threadIdx.x] = j >= j0 + nk ? 0 : int((first + j) % ring);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < BATCH_KEYS * (dim / 8); i += blockDim.x) {
            const int kk = i / (dim / 8), c = i % (dim / 8);
            uint4 v = make_uint4(0, 0, 0, 0);
            if (kk < nk) {
                const int r = row[kk];
                if (q8) {
                    if (q8_fast) {
                        // q8_0, dequantized into the shared row: the 8 values of a 16-byte word start at a
                        // multiple of 8 and never leave their one block of 32, so one scale serves them all
                        // (--q8-dequant 1)
                        const float s = __half2float(__ushort_as_half(k_scales[(size_t) r * (dim / 32) + (c >> 2)]));
                        const int8_t* cd = k_codes + (size_t) r * dim + c * 8;
                        half2* dst = (half2*) &v;
                        dst[0] = __floats2half2_rn(float(cd[0]) * s, float(cd[1]) * s);
                        dst[1] = __floats2half2_rn(float(cd[2]) * s, float(cd[3]) * s);
                        dst[2] = __floats2half2_rn(float(cd[4]) * s, float(cd[5]) * s);
                        dst[3] = __floats2half2_rn(float(cd[6]) * s, float(cd[7]) * s);
                    } else {
                        // a value at a time, the row's scale fetched for each (--q8-dequant 0)
                        const int8_t* codes = k_codes + (size_t) r * dim;
                        const uint16_t* scales = k_scales + (size_t) r * (dim / 32);
                        half* dst = (half*) &v;
                        for (int j = 0; j < 8; ++j) {
                            int idx = c * 8 + j;
                            if (idx < dim) {
                                dst[j] = __float2half(float(codes[idx]) *
                                                      __half2float(__ushort_as_half(scales[idx / 32])));
                            }
                        }
                    }
                } else {
                    v = ((const uint4*) (raw + (size_t) r * dim))[c];
                }
            }
            *(uint4*) &sk[kk * pitch + 4 * c] = v;
        }
        __syncthreads();
        // a lane scores up to two of the round's keys - its own, and the one past the warp's - for one token at a
        // time; a key outside the token's window scores -inf, as one past the round does
        for (int t = 0; t < ATTEND_BATCH; ++t) {
            float s[2] = {-INFINITY, -INFINITY};
            if (n_raw[t] > 0) {
                const float* qh = q + (size_t(t0 + t) * n_head + h) * dim;
                for (int u = 0; u < 2; ++u) {
                    const int j = lane + u * WARP;      // this round's slot in the staging...
                    const int g = j0 + j;               // ...and the union key it holds
                    if (j < nk && g >= from[t] && g < from[t] + n_raw[t]) {
                        const uint4* kr = (const uint4*) (sk + j * pitch);
                        const float4* q4 = (const float4*) qh;
                        float d0 = 0.0f, d1 = 0.0f;
                        for (int c = 0; c < dim / 8; ++c) {
                            const uint4 kv = kr[c];
                            const float4 qa = q4[2 * c], qb = q4[2 * c + 1];
                            const float2 a = __half22float2(*(const __half2*) &kv.x),
                                         b = __half22float2(*(const __half2*) &kv.y);
                            const float2 e = __half22float2(*(const __half2*) &kv.z),
                                         f = __half22float2(*(const __half2*) &kv.w);
                            d0 += qa.x * a.x + qa.y * a.y + qa.z * b.x + qa.w * b.y;
                            d1 += qb.x * e.x + qb.y * e.y + qb.z * f.x + qb.w * f.y;
                        }
                        s[u] = (d0 + d1) * scale;
                    }
                }
            }
            const float mn = fmaxf(m[t], fmaxf(warp_max(s[0]), warp_max(s[1])));
            const float alpha = expf(m[t] - mn);
            const float e[2] = {s[0] == -INFINITY ? 0.0f : expf(s[0] - mn),
                                s[1] == -INFINITY ? 0.0f : expf(s[1] - mn)};
            l[t] = l[t] * alpha + warp_sum(e[0] + e[1]);
            m[t] = mn;
            float* sb = sp + (size_t(t) * GROUP_HEADS + w) * PAIR_PROBS;
            sb[lane] = e[0];
            sb[WARP + lane] = e[1];
            __syncwarp();
#pragma unroll
            for (int j = 0; j < GROUP_DIM / WARP; ++j) o[t][j] *= alpha;
            const int first_g = from[t] > j0 ? from[t] : j0;
            for (int g = first_g; g < j0 + nk && g < from[t] + n_raw[t]; ++g) {
                const int j = g - j0;                  // the key's slot in this round's staging
                const float pk = sb[j];
                const uint4* kr = (const uint4*) (sk + j * pitch) + lane;
#pragma unroll
                for (int u = 0; u < GROUP_DIM / (8 * WARP); ++u)   // whole, so that o stays in registers
                    if (u < chunks) {
                        const uint4 v = kr[WARP * u];
                        const float2 a = __half22float2(*(const __half2*) &v.x),
                                     b = __half22float2(*(const __half2*) &v.y);
                        const float2 c = __half22float2(*(const __half2*) &v.z),
                                     d = __half22float2(*(const __half2*) &v.w);
                        float* ou = o[t] + 8 * u;
                        ou[0] += pk * a.x, ou[1] += pk * a.y, ou[2] += pk * b.x, ou[3] += pk * b.y;
                        ou[4] += pk * c.x, ou[5] += pk * c.y, ou[6] += pk * d.x, ou[7] += pk * d.y;
                    }
            }
            __syncwarp();   // sb is written again for the next keys
        }
    }

    // the compressed rows a token's indexer picked: rounds of that token alone, staged for it alone
    for (int t = 0; t < ATTEND_BATCH; ++t) {
        if (t0 + t >= rows) break;
        const int32_t* mine = sel ? sel + size_t(t0 + t) * sel_stride : nullptr;
        int n_comp = 0;
        if (comp) {
            const long long visible = (p[t] + 1) / ratio;
            n_comp = mine ? (int) min(visible, (long long) top) : (int) visible;
        }
        for (int j0 = 0; j0 < n_comp; j0 += GROUP_KEYS) {
            const int nk = min(GROUP_KEYS, n_comp - j0);
            __syncthreads();
            if (threadIdx.x < GROUP_KEYS) {
                const int j = j0 + threadIdx.x;
                row[threadIdx.x] = j >= j0 + nk ? 0 : mine ? mine[j] : j;
            }
            __syncthreads();
            for (int i = threadIdx.x; i < GROUP_KEYS * (dim / 8); i += blockDim.x) {
                const int kk = i / (dim / 8), c = i % (dim / 8);
                uint4 v = make_uint4(0, 0, 0, 0);
                if (kk < nk) v = ((const uint4*) (comp + (size_t) row[kk] * dim))[c];
                *(uint4*) &sk[kk * pitch + 4 * c] = v;
            }
            __syncthreads();
            const float* qh = q + (size_t(t0 + t) * n_head + h) * dim;
            float s = -INFINITY;
            if (lane < nk) {
                const uint4* kr = (const uint4*) (sk + lane * pitch);
                const float4* q4 = (const float4*) qh;
                float d0 = 0.0f, d1 = 0.0f;
                for (int c = 0; c < dim / 8; ++c) {
                    const uint4 kv = kr[c];
                    const float4 qa = q4[2 * c], qb = q4[2 * c + 1];
                    const float2 a = __half22float2(*(const __half2*) &kv.x), b = __half22float2(*(const __half2*) &kv.y);
                    const float2 e = __half22float2(*(const __half2*) &kv.z), f = __half22float2(*(const __half2*) &kv.w);
                    d0 += qa.x * a.x + qa.y * a.y + qa.z * b.x + qa.w * b.y;
                    d1 += qb.x * e.x + qb.y * e.y + qb.z * f.x + qb.w * f.y;
                }
                s = (d0 + d1) * scale;
            }
            const float mn = fmaxf(m[t], warp_max(s));
            const float alpha = expf(m[t] - mn), e = lane < nk ? expf(s - mn) : 0.0f;
            l[t] = l[t] * alpha + warp_sum(e);
            m[t] = mn;
            float* sb = sp + (size_t(t) * GROUP_HEADS + w) * PAIR_PROBS;
            sb[lane] = e;
            __syncwarp();
#pragma unroll
            for (int j = 0; j < GROUP_DIM / WARP; ++j) o[t][j] *= alpha;
            for (int j = 0; j < nk; ++j) {
                const float pk = sb[j];
                const uint4* kr = (const uint4*) (sk + j * pitch) + lane;
#pragma unroll
                for (int u = 0; u < GROUP_DIM / (8 * WARP); ++u)
                    if (u < chunks) {
                        const uint4 v = kr[WARP * u];
                        const float2 a = __half22float2(*(const __half2*) &v.x),
                                     b = __half22float2(*(const __half2*) &v.y);
                        const float2 c = __half22float2(*(const __half2*) &v.z),
                                     d = __half22float2(*(const __half2*) &v.w);
                        float* ou = o[t] + 8 * u;
                        ou[0] += pk * a.x, ou[1] += pk * a.y, ou[2] += pk * b.x, ou[3] += pk * b.y;
                        ou[4] += pk * c.x, ou[5] += pk * c.y, ou[6] += pk * d.x, ou[7] += pk * d.y;
                    }
            }
            __syncwarp();
        }
    }

    if (h >= n_head) return;
    for (int t = 0; t < ATTEND_BATCH; ++t) {
        if (t0 + t >= rows) break;
        const float inv = 1.0f / l[t];
        float* oh = out + (size_t(t0 + t) * n_head + h) * dim + 8 * lane;
#pragma unroll
        for (int j = 0; j < GROUP_DIM / (8 * WARP); ++j)
            if (j < chunks) {
                const float* oj = o[t] + 8 * j;
                *(float4*) &oh[8 * WARP * j] = make_float4(oj[0] * inv, oj[1] * inv, oj[2] * inv, oj[3] * inv);
                *(float4*) &oh[8 * WARP * j + 4] = make_float4(oj[4] * inv, oj[5] * inv, oj[6] * inv, oj[7] * inv);
            }
    }
}

__global__ void permute_kernel(const float* __restrict__ x, float* __restrict__ y, int n, int groups, int rows,
                               bool back) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int g = blockIdx.y % groups, t = blockIdx.y / groups;
    const size_t a = (size_t(t) * groups + g) * n + i, b = (size_t(g) * rows + t) * n + i;
    if (back) y[a] = x[b];
    else y[b] = x[a];
}

}  // namespace

void dsv4_rope(const float* x, float* y, int dim, int heads, int off, int n_rot, const Rope& r, int64_t pos,
               int step, int rows, bool norm, float eps, bool back, void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % 2 || dim > 2048 || off % 2 || off + n_rot > dim)
        throw std::invalid_argument("dsv4_rope: the heads are not dim <= 2048 with the turned values inside");
    const int threads = std::max(WARP, (dim / 2 + WARP - 1) / WARP * WARP);
    rope_kernel<<<dim3(heads, rows), threads, 0, cs(stream)>>>(x, y, dim, off, n_rot, r, (long long) pos, step, norm,
                                                               eps, back, at);
    ck("dsv4_rope");
}

void dsv4_state(const float* kv, const float* score, const float* ape, float* ring_kv, float* ring_score, int ring,
                int width, int ratio, int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    state_kernel<<<dim3(grid(width, 256), rows), 256, 0, cs(stream)>>>(kv, score, ape, ring_kv, ring_score, ring, width,
                                                                       ratio, (long long) pos, at);
    ck("dsv4_state");
}

void dsv4_squeeze(const float* ring_kv, const float* ring_score, int ring, int dim, int ratio, bool overlap,
                  const float* norm, float eps, const Rope& r, int n_rot, int64_t pos, int rows, uint16_t* cache,
                  void* stream, const long long* at) {
    if (rows <= 0) return;
    if (dim % WARP || dim > 1024 || n_rot % 2 || n_rot > dim)
        throw std::invalid_argument("dsv4_squeeze: a row of 32 to 1024 values, in 32s");
    squeeze_kernel<<<rows, dim, 0, cs(stream)>>>(ring_kv, ring_score, ring, dim, ratio, overlap, norm, eps, r, n_rot,
                                                 (long long) pos, at, cache);
    ck("dsv4_squeeze");
}

void dsv4_idx_score(const float* q, const float* w, const uint16_t* keys, float* score, int score_stride, int dim,
                    int n_head, int ratio, int64_t pos, int rows, float scale, void* stream, const long long* at,
                    int room, int mma) {
    if (dim != IDX_DIM)
        throw std::invalid_argument("dsv4_idx_score: the indexer's keys are not 128 values");
    const long long visible = (at ? room : pos + rows) / ratio;   // the last token sees the most
    if (rows <= 0 || visible == 0) return;
#if !defined(GGML_USE_HIP)
    {   // the tensor cores, when the card has them: a block of 16 tokens against 128 compressed rows
        // (--idx-mma 0 keeps the scalar kernel)
        static unsigned done_mma = 0;
        int dev = 0, major = 0;
        if (mma != 0 && cudaGetDevice(&dev) == cudaSuccess &&
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) == cudaSuccess && major >= 8) {
            const size_t smem_mma =
                IDX_TOKENS * (IDX_DIM + 8) * sizeof(half) + IDX_TOKENS * n_head * sizeof(float);
            if (smem_mma <= allow_smem(idx_score_mma_kernel, done_mma)) {
                idx_score_mma_kernel<<<dim3(grid((int) visible, IDX_KEYS), grid(rows, IDX_TOKENS)), 256, smem_mma,
                                       cs(stream)>>>(q, w, keys, score, score_stride, rows, n_head, ratio,
                                                     (long long) pos, scale, at);
                ck("dsv4_idx_score");
                return;
            }
        }
    }
#endif
    const size_t smem = (size_t(n_head) * (IDX_DIM + 1) + n_head + SCORE_KEYS * IDX_DIM) * sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(idx_score_kernel, done))
        throw std::runtime_error("dsv4_idx_score: the indexer's queries do not fit the card's shared memory");
    idx_score_kernel<<<dim3(grid((int) visible, SCORE_KEYS), rows), 256, smem, cs(stream)>>>(
        q, w, keys, score, score_stride, n_head, ratio, (long long) pos, scale, at);
    ck("dsv4_idx_score");
}

void dsv4_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int ratio,
                 int64_t pos, int rows, void* stream, const long long* at) {
    if (rows <= 0) return;
    select_kernel<<<rows, SELECT_THREADS, 0, cs(stream)>>>(score, score_stride, sel, sel_stride, n_top, ratio,
                                                           (long long) pos, at);
    ck("dsv4_select");
}

void dsv4_attend(const float* q, const uint16_t* raw, int ring, int window, const uint16_t* comp, int ratio,
                 const int32_t* sel, int sel_stride, int top, const float* sinks, float* out, int dim, int n_head,
                 float scale, int64_t pos, int rows, void* stream, const long long* at, int room,
                 Kvq8 kq8, int batch, int group_heads, int q8_fast, int chunks) {
    if (rows <= 0) return;
    const bool q8 = kq8.codes != nullptr;
    // Pairs of tokens first: a block stages the union of the two's raw windows, about half the key traffic of a
    // token a block.  --attend-batch 0 takes a token a block.
    if (batch != 0 && rows >= GROUP_ROWS && dim % (8 * WARP) == 0 && dim <= GROUP_DIM) {
        const size_t smem = size_t(BATCH_KEYS) * (dim / 2 + 4) * sizeof(uint32_t) +
                            size_t(ATTEND_BATCH) * GROUP_HEADS * PAIR_PROBS * sizeof(float);
        static unsigned done = 0;
        if (smem <= allow_smem(attend_batch_kernel, done)) {
            // The whole batch, a block a pair of tokens on the grid's second axis, in launches of at most 65535
            // pairs (CUDA's grid.y limit, as the launches of one token a block below): a pair's tokens' rows of q,
            // sel, out and at sit behind the pairs before, and the caches every token reads as its own position
            // says: one set serves all the chunks.
            constexpr int kGridPairs = 65535;
            for (int first = 0; first < rows; first += ATTEND_BATCH * kGridPairs) {
                const int m = std::min(ATTEND_BATCH * kGridPairs, rows - first);
                attend_batch_kernel<<<dim3(grid(n_head, GROUP_HEADS), grid(m, ATTEND_BATCH)), GROUP_HEADS * WARP,
                                      smem, cs(stream)>>>(
                    q + (size_t) first * n_head * dim, raw, ring, window, comp, ratio,
                    sel ? sel + (size_t) first * sel_stride : nullptr, sel_stride, top, sinks,
                    out + (size_t) first * n_head * dim, dim, n_head, scale, (long long) (pos + first), m,
                    at ? at + 2 * (size_t) first : nullptr, q8 ? kq8.codes : nullptr, q8 ? kq8.scales : nullptr,
                    q8_fast);
                ck("dsv4_attend");
            }
            return;
        }
    }
    // the tensor cores: a block of GROUP_HEADS heads on the mma's M tile (this attention is MQA, the heads share
    // the keys), fdec_kernel's mechanics.  The kernels below stay for other head counts, dimensions and cards.
    if (rows >= GROUP_ROWS && dim == GROUP_DIM && n_head % GROUP_HEADS == 0) {
#if !defined(GGML_USE_HIP)
        static unsigned done = 0;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess ||
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess || major < 8)
            goto scalar;   // before the tensor cores: the kernel below stays
        {
        const size_t smem = (size_t(GROUP_HEADS) * (GROUP_DIM + 8) + GROUP_HEADS * (64 + 8)) * sizeof(half) +
                            2 * 8 * GROUP_HEADS * sizeof(float);
        if (smem <= allow_smem(attend_rows_mma_kernel<GROUP_DIM>, done)) {
            for (int first = 0; first < rows; first += 65535) {
                const int m = std::min(65535, rows - first);
                attend_rows_mma_kernel<GROUP_DIM><<<dim3(n_head / GROUP_HEADS, m), 8 * WARP, smem, cs(stream)>>>(
                    q + (size_t) first * n_head * dim, raw, ring, window, comp, ratio,
                    sel ? sel + (size_t) first * sel_stride : nullptr, sel_stride, top, sinks,
                    out + (size_t) first * n_head * dim, n_head, scale, (long long) (pos + first),
                    at ? at + 2 * (size_t) first : nullptr, q8 ? kq8.codes : nullptr, q8 ? kq8.scales : nullptr);
                ck("dsv4_attend");
            }
            return;
        }
        }
#endif
    }
scalar:
    // a batch: its shared memory does not grow with the keys.  A few tokens keep a block a head, which spreads a
    // token over more of the card.
    if (rows >= GROUP_ROWS && dim % (8 * WARP) == 0 && dim <= GROUP_DIM) {
        const int gh = group_heads == 8 ? 8 : GROUP_HEADS;   // --group-heads: 8 as the kernel first was
        const size_t smem = (size_t(gh) * dim + size_t(gh) * GROUP_KEYS) * sizeof(float) +
                            size_t(GROUP_KEYS) * (dim / 2 + 4) * sizeof(uint32_t);
        static unsigned done8 = 0, done16 = 0;   // allow_smem() asks the card once a kernel
        const auto launch = gh == 8 ? attend_rows_kernel<8> : attend_rows_kernel<16>;
        if (smem <= allow_smem(launch, gh == 8 ? done8 : done16)) {
            // The whole batch, a block a token on the grid's second axis, in launches of at most 65535 rows when
            // the chunks are on: no grid axis takes more blocks than that, so a --prompt-batch above it goes on in
            // the next launch, its tokens' rows of q, sel, out and at behind the chunk before.  The caches (raw,
            // comp, the q8_0 codes and scales) every token reads as its own position says: one set serves all the
            // chunks.  --attend-chunks 0 leaves the one launch of before, which a batch above 65535 rows fails.
            const int step = chunks != 0 ? 65535 : rows;   // CUDA's grid.y limit
            for (int first = 0; first < rows; first += step) {
                const int m = std::min(step, rows - first);
                launch<<<dim3(grid(n_head, gh), m), gh * WARP, smem, cs(stream)>>>(
                    q + (size_t) first * n_head * dim, raw, ring, window, comp, ratio,
                    sel ? sel + (size_t) first * sel_stride : nullptr, sel_stride, top, sinks,
                    out + (size_t) first * n_head * dim, dim, n_head, scale, (long long) (pos + first),
                    at ? at + 2 * (size_t) first : nullptr, q8 ? kq8.codes : nullptr, q8 ? kq8.scales : nullptr,
                    q8_fast);
                ck("dsv4_attend");
            }
            return;
        }
    }
    long long keys = window;
    if (comp) keys += sel ? top : (at ? room : pos + rows) / ratio;
    const size_t smem = (size_t(dim) + 2 * size_t(keys)) * sizeof(float);
    static unsigned done = 0;
    if (smem > allow_smem(attend_kernel, done))
        throw std::runtime_error("dsv4_attend: " + std::to_string(keys) + " keys a token do not fit the card's shared "
                                 "memory; lower --max-context");
    attend_kernel<<<dim3(n_head, rows), ATTEND_THREADS, smem, cs(stream)>>>(q, raw, ring, window, comp, ratio, sel,
                                                                            sel_stride, top, sinks, out, dim, n_head,
                                                                            scale, (long long) pos, at,
                                                                            q8 ? kq8.codes : nullptr, q8 ? kq8.scales : nullptr);
    ck("dsv4_attend");
}

void permute_groups(const float* x, float* y, int n, int groups, int rows, bool back, void* stream) {
    if (rows <= 0) return;
    permute_kernel<<<dim3(grid(n, 256), groups * rows), 256, 0, cs(stream)>>>(x, y, n, groups, rows, back);
    ck("permute_groups");
}

}  // namespace strata::poly
