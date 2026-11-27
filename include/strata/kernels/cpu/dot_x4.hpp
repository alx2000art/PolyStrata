// src/kernels/cpu/dot_x4.hpp - four weight rows against one activation row, ggml-cpu's own bits.
//
// The decode's expert products share their activation: the one-row vec_dots load it for every weight row, these
// kernels load it once for four.  Each row's value is the bits ggml-cpu's vec_dot of that type returns for the
// same pair - the integer sums are exact, so the per-block grouping is free, and the float part keeps ggml's
// AVX2 sequence: a block's i32 sums as floats, times the block's scale, added into an 8-lane acc block by
// block, the 8 lanes folded by hsum_float_8's tree at the end (dot_x4_parity checks it against ggml-cpu).
#ifndef STRATA_KERNELS_CPU_DOT_X4_HPP
#define STRATA_KERNELS_CPU_DOT_X4_HPP

#include <cstddef>
#include <cstdint>

namespace strata::kernels::cpu {

/// out[q] = dot(weight row q (w_base + q * row_bytes), activation a) for q < 4.  n is a whole number of the
/// weight type's blocks (and of the activation type's); row_bytes a whole number of them too.
using DotX4Fn = void (*)(int n, float out[4], const void* w_base, size_t row_bytes, const void* a);

/// The type's four-row kernel on this CPU, or null.  The expert types of DeepSeek V4 Flash's UD-IQ2_XXS:
/// GGML_TYPE_IQ2_XS (17, Q8_K activations) and GGML_TYPE_Q5_0 (6, Q8_0).  AVX2; null without it.
DotX4Fn dot_x4_kernel(int type);

}  // namespace strata::kernels::cpu

#endif
