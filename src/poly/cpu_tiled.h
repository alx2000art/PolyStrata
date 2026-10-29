// src/poly/cpu_tiled.h - see cpu_tiled.cpp.
// The vendored llama.cpp tiled K-quant CPU kernels (Q2_K/Q3_K GEMM with
// Q8_K activations, AVX512VNNI), behind a raw-pointer strata API.
#pragma once
#include <cstddef>

namespace strata::poly {

bool cpu_tiled_supported(int w_type);
size_t cpu_tiled_scratch_bytes(int n_threads);
int cpu_tiled_group();
void cpu_tiled_row_range(int n_out, int ith, int nth, int* lo, int* hi);

// One expert's weight panel [n_out x n_in] of w_type against Q8_K activations.
// act holds rows_total Q8_K rows (the call's tokens); rows points at n_tok
// (src1 row, batch 0) pairs — the run's tokens.  out is the calling thread's
// slice [slice_rows x n_tok] floats, row-major; the panel's rows r land at
// (r - row_lo) * n_tok + j, where [row_lo, row_hi) is cpu_tiled_row_range's
// partition for (ith, nth).  Returns false when not eligible (vec_dot fallback).
bool cpu_tiled_mmul_one(const void* w, std::ptrdiff_t w_row_bytes, int w_type, int n_out, int n_in,
                        const void* act, std::ptrdiff_t act_row_bytes, int rows_total,
                        const int32_t* rows, int n_tok,
                        float* out, int out_stride, int row_lo,
                        int ith, int nth, char* scratch_base);

}  // namespace strata::poly
