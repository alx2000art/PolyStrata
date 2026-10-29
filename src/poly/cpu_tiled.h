// src/poly/cpu_tiled.h - see cpu_tiled.cpp.
// The vendored llama.cpp tiled K-quant CPU kernels (Q2_K/Q3_K GEMM with
// Q8_K activations, AVX512VNNI), behind a raw-pointer strata API.
#pragma once
#include <cstddef>

namespace strata::poly {

bool cpu_tiled_supported(int w_type);
size_t cpu_tiled_scratch_bytes(int n_threads);

bool cpu_tiled_mmul_one(const void* w, std::ptrdiff_t w_row_bytes, int w_type, int n_out, int n_in,
                        const void* act, std::ptrdiff_t act_row_bytes, int n_tok,
                        float* out, std::ptrdiff_t out_row_bytes,
                        int ith, int nth, char* scratch);

}  // namespace strata::poly
