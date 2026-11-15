// src/poly/eagle3.cu - the EAGLE-3 draft head's implementation.  The wiring is
// SafeAILab/EAGLE's eagle/model/cnets.py, LlamaDecoderLayeremb, read and verified:
//
//   residual = hidden_states                       (4096: the fc's output, or the recursion's)
//   h = hidden_norm(hidden);  e = input_layernorm(emb)
//   cat = [e | h]                                  (8192 - the attention's input)
//   q/k/v = cat x the projections;  the rope;  the cache rows;  the causal attention (GQA 64:8)
//   o = att x o_proj;  hidden = residual + o
//   h2 = post_attention_layernorm(hidden);  mlp = down(silu(gate(h2)) x up(h2))
//   hidden = hidden + mlp;  logits = lm_head(norm(hidden))   (the draft steps only)
//
// Every norm+quantize and residual+norm+quantize rides kernels::native_add_rms_norm_q8_1(), the swiglu
// is kernels::swiglu_clamp() (the limit makes it the plain Llama one), the attention is attend_batch() -
// the fork's generic causal attention (the Qwen prompt path), whose cache layout this head's linear
// fp16 cache matches.  No kernel of this file's own.  The norms write contiguous halves and the two
// fp32 halves are assembled into the concat with two 2D copies; the projections dispatch on the call's
// rows (linear(): a prompt batch dequantizes the BF16 matrix to F32 and goes through cuBLAS, a few
// tokens through gemv_bf16).
#include "eagle3.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace strata::poly {

namespace {
constexpr float kPi = 3.14159265358979323846f;

// deepseek4.cpp's make_rope, mirrored: the head's rope is the main model's yarn with the head's
// config's knobs (the corr dims in pair indices, the magnitude the yarn formula gives).
Rope make_rope(float base, int n_rot, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
               float beta_slow) {
    Rope r;
    r.theta_scale = powf(base, -2.0f / (float) n_rot);
    r.freq_scale = freq_scale;
    r.ext_factor = ext_factor;
    r.attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const auto corr = [&](float rot) {
        return (float) n_rot * logf((float) n_ctx_orig / (rot * 2 * kPi)) / (2 * logf(base));
    };
    r.corr0 = std::max(0.0f, floorf(corr(beta_fast)));
    r.corr1 = std::min((float) (n_rot - 1), ceilf(corr(beta_slow)));
    return r;
}
}

// The BF16 weights quantize to Q8_0 on the host: a block of 32 values, the scale the max's / 127
// in fp16, the values rounded - the GGML Q8_0 layout the mmvq kernels read.


void Eagle3::load(Shards& dg) {
    // the head's K of 12288 takes the Q8_0 multi kernel's two-rows path (the engine's own linears
    // are K=4096 and take the one-row one); STRATA_NO_MMVQ_ROWS2=1 keeps one row for large K - the
    // documented switch - and no-overwrite keeps the user's own setting
    setenv("STRATA_NO_MMVQ_ROWS2", "1", 0);
    auto kv = [&dg](const char* k) { return dg.get(k); };
    n_embd_ = int(kv("eagle3.embedding_length")->u);
    n_head_ = int(kv("eagle3.attention.head_count")->u);
    n_kv_ = int(kv("eagle3.attention.head_count_kv")->u);
    head_dim_ = int(kv("eagle3.attention.key_length")->u);
    n_ff_ = int(kv("eagle3.intermediate_size")->u);
    dvocab_ = int(kv("eagle3.draft_vocab_size")->u);
    eps_ = float(kv("eagle3.rms_norm_eps")->f);

    // the rope: the head's yarn scaling, the corr dims the main model's rope helpers work out
    const float base = float(kv("eagle3.rope.freq_base")->f);
    const float factor = float(kv("eagle3.rope.scaling.factor")->f);
    const int64_t orig = int64_t(kv("eagle3.rope.scaling.original_context_length")->u);
    const bool yarn = factor > 1.f;
    rope_ = make_rope(base, head_dim_, yarn ? 1.f / factor : 1.f, yarn ? 1.f : 0.f, orig, 32.f, 1.f);

    // the vocab mapping
    if (const MetaValue* t = kv("eagle3.d2t"); t && t->type == MetaType::ARRAY) {
        d2t_.reserve(t->items.size());
        for (const MetaValue& v : t->items) d2t_.push_back(int64_t(v.u));
    }
    if (d2t_.size() < (size_t) dvocab_) {   // the file has no d2t: the draft ids are the target ids
        std::fprintf(stderr,
                     "strata-poly: the eagle3 head has %zu d2t entries for a %d draft vocab - the draft ids "
                     "map to themselves\n",
                     d2t_.size(), dvocab_);
        d2t_.resize((size_t) dvocab_);
        for (int i = 0; i < dvocab_; ++i) d2t_[(size_t) i] = i;
    } else {
        std::fprintf(stderr, "strata-poly: the eagle3 d2t: %zu entries, first %lld %lld %lld %lld\n",
                     d2t_.size(), (long long) d2t_[0], (long long) d2t_[1], (long long) d2t_[2],
                     (long long) d2t_[3]);
    }

    // the weights: the file's own BF16 bytes on card, the projections read them through the engine's
    // BF16 paths (gemv_bf16, and bf16_to_f32 + cuBLAS for a prompt batch)
    const auto w = [&](const char* name) {
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        const size_t n_bytes = t->elements() * 2;
        void* dev = nullptr; ck(cudaMalloc(&dev, n_bytes + 256), name);
        ck(cudaMemcpy(dev, dg.data(*t), n_bytes, cudaMemcpyHostToDevice), "eagle weight");
        ck(cudaMemsetAsync((char*) dev + n_bytes, 0, 256, nullptr), "eagle weight tail");
        return dev;
    };
    fc_w_ = w("fc.weight");
    q_w_ = w("midlayer.self_attn.q_proj.weight");
    k_w_ = w("midlayer.self_attn.k_proj.weight");
    v_w_ = w("midlayer.self_attn.v_proj.weight");
    o_w_ = w("midlayer.self_attn.o_proj.weight");
    gate_w_ = w("midlayer.mlp.gate_proj.weight");
    up_w_ = w("midlayer.mlp.up_proj.weight");
    down_w_ = w("midlayer.mlp.down_proj.weight");
    lm_head_w_ = w("lm_head.weight");
    // the norms: BF16 in the file, F32 on the card (the rms_norm kernels read floats) - each pair
    // of bytes is the high half of the float
    const auto f = [&](const char* name) {
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        const size_t n_vals = t->elements();
        std::vector<float> w(n_vals);
        const uint8_t* src = dg.data(*t);
        for (size_t i = 0; i < n_vals; ++i) {
            uint16_t h;
            std::memcpy(&h, src + i * 2, 2);
            const uint32_t bits = uint32_t(h) << 16;
            std::memcpy(&w[i], &bits, 4);
        }
        float* dev = nullptr; ck(cudaMalloc(&dev, n_vals * sizeof(float)), name);
        ck(cudaMemcpy(dev, w.data(), n_vals * sizeof(float), cudaMemcpyHostToDevice), "eagle norm");
        return dev;
    };
    ln1_ = f("midlayer.input_layernorm.weight");
    ln2_ = f("midlayer.post_attention_layernorm.weight");
    hnorm_ = f("midlayer.hidden_norm.weight");
    out_norm_ = f("norm.weight");
    loaded_ = true;
    std::fprintf(stderr, "strata-poly: the EAGLE-3 drafter loaded (%d heads of %d, draft vocab %d)\n",
                 n_head_, head_dim_, dvocab_);
}

void Eagle3::begin(int rows, int64_t max_context, void* blas) {
    rows_ = rows;
    cache_cap_ = max_context;
    blas_ = blas;
    ck(cudaMalloc(&k_cache_, size_t(cache_cap_) * n_kv_ * head_dim_ * 2), "the eagle3 k cache");
    ck(cudaMalloc(&v_cache_, size_t(cache_cap_) * n_kv_ * head_dim_ * 2), "the eagle3 v cache");
    // the attention's scratch: the score blocks and the grouped queries/results the cuBLAS path
    // keeps there (attend_batch's doc)
    ck(cudaMalloc(&score_, kScoreFloats * sizeof(float)), "the eagle3 attention score");
    const size_t work_floats = size_t(rows_) * n_head_ * (head_dim_ + head_dim_ / 2 + 3) + 1;
    ck(cudaMalloc(&work_, work_floats * sizeof(float)), "the eagle3 attention work");
    ck(cudaMalloc(&hidden_, size_t(rows_) * n_embd_ * sizeof(float)), "the eagle3 hidden");
    ck(cudaMalloc(&emb_dev_, size_t(rows_) * n_embd_ * sizeof(float)), "the eagle3 emb");
    ck(cudaMalloc(&e_norm_, size_t(rows_) * n_embd_ * sizeof(float)), "the eagle3 e norm");
    ck(cudaMalloc(&f_norm_, size_t(rows_) * n_embd_ * sizeof(float)), "the eagle3 f norm");
    ck(cudaMalloc(&e_q8_, size_t(rows_) * kernels::native_q8_1_bytes(n_embd_)), "the eagle3 e q8_1");
    ck(cudaMalloc(&f_q8_, size_t(rows_) * kernels::native_q8_1_bytes(n_embd_)), "the eagle3 f q8_1");
    ck(cudaMalloc(&q_, size_t(rows_) * n_head_ * head_dim_ * sizeof(float)), "the eagle3 q");
    ck(cudaMalloc(&k_, size_t(rows_) * n_kv_ * head_dim_ * sizeof(float)), "the eagle3 k");
    ck(cudaMalloc(&v_, size_t(rows_) * n_kv_ * head_dim_ * sizeof(float)), "the eagle3 v");
    ck(cudaMalloc(&o_, size_t(rows_) * n_head_ * head_dim_ * sizeof(float)), "the eagle3 o");
    ck(cudaMalloc(&normed_, size_t(rows_) * n_embd_ * sizeof(float)), "the eagle3 normed");
    ck(cudaMalloc(&normed_q8_, size_t(rows_) * kernels::native_q8_1_bytes(n_embd_)), "the eagle3 normed q8_1");
    ck(cudaMalloc(&cat_f32_, size_t(rows_) * 2 * n_embd_ * sizeof(float)), "the eagle3 cat fp32");
    ck(cudaMalloc(&gate_out_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 gate");
    ck(cudaMalloc(&up_out_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 up");
    ck(cudaMalloc(&mlp_h_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 mlp");
    ck(cudaMalloc(&logits_, size_t(dvocab_) * sizeof(float)), "the eagle3 logits");
    // the prompt batches' F32 view of one BF16 matrix: the widest the head dequantizes is n_ff x n_embd
    ck(cudaMalloc(&wf_e_, size_t(n_ff_) * n_embd_ * sizeof(float)), "the eagle3 weight f32 view");
}

// One projection of n rows of BF16 weights: a prompt batch dequantizes the matrix to F32 once (the
// engine's own lin() does the same for its BF16 weights) and goes through cuBLAS; a few tokens
// through gemv_bf16.  The Q8_0 kernels NaN'd on this head's K of 12288.
void Eagle3::linear(const void* w, const float* x, int n_in, int n_out, float* y, int n, void* stream) {
    if (n >= kStageRows) {
        bf16_to_f32((const uint16_t*) w, wf_e_, size_t(n_out) * n_in, stream);
        gemm_f32(blas_, wf_e_, x, y, n_in, n_out, n_out, n);
        return;
    }
    gemv_bf16((const uint16_t*) w, x, y, n_in, n_out, n_out, n, stream);
}

void Eagle3::midlayer(float* emb_dev, int64_t pos, int n, void* stream) {
    const int hd = n_head_ * head_dim_;
    // the two norms, each also into its own Q8_1; the second's x is the residual in place (the add
    // is null, so hidden_ keeps itself) - the residual survives both launches
    kernels::native_add_rms_norm_q8_1(emb_dev, nullptr, ln1_, e_norm_, e_q8_, n_embd_, n, eps_, 256, stream);
    kernels::native_add_rms_norm_q8_1(hidden_, nullptr, hnorm_, f_norm_, f_q8_, n_embd_, n, eps_, 256, stream);
    // the halves assemble into [e | h] in fp32: the Q8_1 blocks of 32 lie inside one half or the
    // other, so quantizing the concat gives the halves' own quantizes to the bit
    ck(cudaMemcpy2DAsync(cat_f32_, 2 * n_embd_ * sizeof(float), e_norm_, n_embd_ * sizeof(float),
                         n_embd_ * sizeof(float), n, cudaMemcpyDeviceToDevice, cs(stream)), "eagle cat e");
    ck(cudaMemcpy2DAsync(cat_f32_ + n_embd_, 2 * n_embd_ * sizeof(float), f_norm_, n_embd_ * sizeof(float),
                         n_embd_ * sizeof(float), n, cudaMemcpyDeviceToDevice, cs(stream)), "eagle cat f");
    linear(q_w_, cat_f32_, 2 * n_embd_, hd, q_, n, stream);
    linear(k_w_, cat_f32_, 2 * n_embd_, n_kv_ * head_dim_, k_, n, stream);
    linear(v_w_, cat_f32_, 2 * n_embd_, n_kv_ * head_dim_, v_, n, stream);
    dsv4_rope(q_, q_, head_dim_, n_head_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    dsv4_rope(k_, k_, head_dim_, n_kv_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    // the cache rows: the fp32 k_/v_ converted to the cache's fp16 (a byte copy would store the
    // floats' low halves, which read back as wild fp16 values - inf among them, and the attention's
    // softmax turned those into nan)
    kernels::f32_to_f16_bulk(k_, k_cache_ + size_t(pos) * n_kv_ * head_dim_, size_t(n) * n_kv_ * head_dim_, stream);
    kernels::f32_to_f16_bulk(v_, v_cache_ + size_t(pos) * n_kv_ * head_dim_, size_t(n) * n_kv_ * head_dim_, stream);
    // the causal attention over the whole linear cache (the head's 8 heads a group misses the
    // dim-64 fast path's per == 4, so the cuBLAS path takes it)
    attend_batch(blas_, q_, k_cache_, v_cache_, pos, n, score_, kScoreFloats, work_, o_, head_dim_,
                 n_head_, n_kv_, 1.f / std::sqrt(float(head_dim_)), stream);
    if (dbg_att_ < 1) {   // the first inject's attention output: finite means the cache and the softmax hold
        ++dbg_att_;
        float h[4] = {};
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg syncA");
        ck(cudaMemcpy(h, o_, sizeof h, cudaMemcpyDeviceToHost), "eagle dbg att");
        std::fprintf(stderr, "[eagle] att->o (pos %lld): %.4f %.4f %.4f %.4f\n", (long long) pos, h[0], h[1], h[2],
                     h[3]);
    }
    // the residual + the norm (and its Q8_1, which nothing reads any more) in one launch
    kernels::native_add_rms_norm_q8_1(hidden_, o_, ln2_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
    linear(gate_w_, normed_, n_embd_, n_ff_, gate_out_, n, stream);
    linear(up_w_, normed_, n_embd_, n_ff_, up_out_, n, stream);
    // silu(gate) x up in fp32 (the limit makes swiglu_clamp's min/clamp no-ops) - the poly
    // kernels live in strata::poly itself, no namespace step
    swiglu_clamp(gate_out_, up_out_, mlp_h_, size_t(n) * n_ff_, 1e30f, stream);
    linear(down_w_, mlp_h_, n_ff_, n_embd_, o_, n, stream);
    // the residual + the final norm in one launch - the lm_head's input ready
    kernels::native_add_rms_norm_q8_1(hidden_, o_, out_norm_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
}

void Eagle3::inject(const float* taps_dev, float* emb_dev, int64_t pos, int n, void* stream) {
    // the taps' stack [rows x 3 x n_embd] through the fc, the output lands in hidden_ (the residual)
    linear(fc_w_, taps_dev, 3 * n_embd_, n_embd_, hidden_, n, stream);
    if (dbg_inj_ < 1) {   // the first inject only: the taps' range as the head sees it
        ++dbg_inj_;
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg sync");
        std::vector<float> th((size_t) n * 3 * n_embd_);
        ck(cudaMemcpy(th.data(), taps_dev, th.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dbg taps");
        double s = 0;
        float mn = th[0], mx = th[0];
        for (float x : th) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
        std::fprintf(stderr, "[eagle] inject n=%d pos=%lld taps mean=%.4f min=%.2f max=%.2f\n", n, (long long) pos,
                     s / th.size(), mn, mx);
    }
    midlayer(emb_dev, pos, n, stream);
    cache_pos_ = pos + n;
}

const float* Eagle3::draft_logits(int32_t token, const float* emb_host, void* stream) {
    ck(cudaMemcpyAsync(emb_dev_, emb_host, n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(stream)),
       "eagle draft emb");
    midlayer(emb_dev_, cache_pos_, 1, stream);
    cache_pos_ += 1;
    gemv_bf16((const uint16_t*) lm_head_w_, normed_, logits_, n_embd_, dvocab_, dvocab_, 1, stream);
    return logits_;
}

}  // namespace strata::poly
