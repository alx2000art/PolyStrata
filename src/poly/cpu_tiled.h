// src/poly/cpu_tiled.h - see cpu_tiled.cpp.
// The vendored llama.cpp tiled K-quant CPU kernels (Q2_K/Q3_K GEMM with
// Q8_K activations, AVX512VNNI), behind a raw-pointer strata API.
#pragma once
#include <cstddef>

namespace strata::poly {

bool cpu_tiled_supported(int w_type);
size_t cpu_tiled_scratch_bytes(int n_threads);
size_t cpu_tiled_ws_slot_bytes();

// One expert's weight panel [n_out x n_in] of w_type against Q8_K activations.
// act holds rows_total Q8_K rows (the call's tokens); rows points at n_tok
// (src1 row, batch 0) pairs — the run's tokens.  out is the full panel
// [n_out x n_tok] floats, row-major: the kernel partitions the run's tokens
// between ith/nth itself and writes every n_out row of its slice.
// ws_slot: this thread's 64B-aligned tiled workspace (cpu_tiled_ws_slot_bytes).
// Returns false when not eligible (vec_dot fallback).
bool cpu_tiled_mmul_one(const void* w, std::ptrdiff_t w_row_bytes, int w_type, int n_out, int n_in,
                        const void* act, std::ptrdiff_t act_row_bytes, int rows_total,
                        const int32_t* rows, int n_tok,
                        float* out, int ith, int nth, char* ws_slot);

}  // namespace strata::poly
