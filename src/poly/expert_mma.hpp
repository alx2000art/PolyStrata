// src/poly/expert_mma.hpp - the experts' gate/up (and down, when it is the same type) products on the tensor
// cores, own kernels instead of llama.cpp's generic mmq: the weights stay Q3_K, the rows are quantized to a
// plain per-32 format of this file's own, and mma.sync f16 fragments carry the codes (an int8 code is exact in
// a half, so a fragment holds code * scale exactly enough; the reference for every Q3_K formula is
// include/strata/artifact/dequant.hpp, itself validated against ggml).
//
// Why: the generic mmq launches cover row tiles padded to the round's widest expert, and its Q3_K path is the
// dp4a-era scheduler; a tile of 64 weight rows against 32 tokens at a K step of 128 with the dequantized codes
// in shared memory fits a staging round's shapes (K 4096/1024, a few dozen experts, a few dozen rows each).
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.  Needs sm_80+ (f16 mma.sync).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

class ExpertMma {
public:
    static constexpr int kQ3K = 11;   // ggml's GGML_TYPE_Q3_K

    /// 1 when this build takes the products of `ggml_type` (Q3_K only) with matrices `w_cols` values wide.
    static bool fits(int ggml_type, int64_t w_cols);
    /// Bytes of `rows` rows of `cols` values as quantize() writes them.
    static size_t x8_bytes(int64_t rows, int64_t cols);

    /// Row i of xq is row ids[i] of x (ld floats a row), or row i when ids is null: a plain per-32 int8 block
    /// with an fp16 scale, 36 bytes a block ({ half d; half pad; int8 qs[32] }).
    static void quantize(const float* x, const int32_t* ids, void* xq, int64_t cols, int64_t ld, int64_t rows,
                         void* stream);

    /// One launch over n matrices of [w_rows, w_cols] of Q3_K, `stride` bytes apart from w: matrix e times the
    /// rows [bounds[e], bounds[e + 1]) of xq.  Row i's product goes to row ids[i] of dst, ld_dst floats a row
    /// (row i when ids is null).  bounds and ids are on the device; max_rows is the most rows one of the
    /// matrices has.  Throws std::runtime_error.
    static void mul(const void* w, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
                    const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
                    int64_t ld_dst, void* stream);
};

}  // namespace strata::poly
