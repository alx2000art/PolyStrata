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
// Every norm+quantize and residual+norm+quantize rides native_add_rms_norm_q8_1(), the swiglu
// rides native_swiglu_quantize_q8_1(), the attention is fdec_f16() (the GQA causal of the fork's
// own kernels) - no kernel of this file's own.  The norms write contiguous halves and the two
// Q8_1 halves are assembled into one strided input with two 2D copies.
#include "eagle3.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "fattn.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/rope_scaling.hpp"

#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {

namespace {
constexpr int kTypeQ8_0 = 6;   // the ggml type id the mmvq kernels take (this fork passes the numbers)
constexpr int ALIGN = 32;
}

// The BF16 weights quantize to Q8_0 on the host: a block of 32 values, the scale the max's / 127
// in fp16, the values rounded - the GGML Q8_0 layout the mmvq kernels read.
static void bf16_to_q8_0(const uint8_t* src, std::vector<uint8_t>& dst, size_t n_vals) {
    dst.resize(n_vals / 32 * 34);
    for (size_t b = 0; b + 32 <= n_vals; b += 32) {
        float mx = 0.f;
        float xs[32];
        for (int i = 0; i < 32; ++i) {
            uint16_t h;
            std::memcpy(&h, src + (b + i) * 2, 2);
            uint32_t bits = uint32_t(h) << 16;
            float x;
            std::memcpy(&x, &bits, 4);
            xs[i] = x;
            mx = std::max(mx, std::fabs(x));
        }
        float d = mx / 127.f;
        if (d == 0.f) d = 1e-30f;
        const _half2 dh = __floats2half2_rn(d, d);
        std::memcpy(dst.data() + b / 32 * 34, &dh, 2);
        for (int i = 0; i < 32; ++i) {
            const int q = int(xs[i] / d + (xs[i] > 0.f ? 0.5f : -0.5f));
            dst[b / 32 * 34 + 2 + i] = int8_t(std::clamp(q, -127, 127));
        }
    }
}

void Eagle3::load(Shards& dg) {
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
    float corr0 = 0.f, corr1 = 0.f;
    yarn_corr_dims(base, orig, 131072, 32.f, 1.f, corr0, corr1);
    const bool yarn = factor > 1.f && corr1 > corr0;
    rope_ = make_rope(base, head_dim_, yarn ? 1.f / factor : 1.f, yarn ? 1.f : 0.f, orig, 32.f, 1.f);
    rope_.corr0 = corr0;
    rope_.corr1 = corr1;

    // the vocab mapping
    if (const MetaValue* t = kv("eagle3.d2t"); t && t->type == MetaType::ARRAY) {
        d2t_.reserve(t->items.size());
        for (const MetaValue& v : t->items) d2t_.push_back(int64_t(v.u));
    }

    // the weights: BF16 mapped -> Q8_0 on card
    const auto w = [&](const char* name) {
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        const size_t n_vals = dg.bytes(*t) / 2;
        std::vector<uint8_t> q;
        bf16_to_q8_0(dg.data(*t), q, n_vals);
        void* dev = dmalloc(q.size(), name);
        ck(cudaMemcpy(dev, q.data(), q.size(), cudaMemcpyHostToDevice), "eagle weight");
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
    const auto f = [&](const char* name) {
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        float* dev = (float*) dmalloc(t->file_size(), name);
        ck(cudaMemcpy(dev, dg.data(*t), t->file_size(), cudaMemcpyHostToDevice), "eagle norm");
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

void Eagle3::begin(int rows, int64_t max_context) {
    rows_ = rows;
    cache_cap_ = max_context;
    k_cache_ = (uint16_t*) dmalloc(size_t(cache_cap_) * n_kv_ * head_dim_ * 2, "the eagle3 k cache");
    v_cache_ = (uint16_t*) dmalloc(size_t(cache_cap_) * n_kv_ * head_dim_ * 2, "the eagle3 v cache");
    scratch_floats_ = 64u << 20 / sizeof(float);
    scratch_ = (float*) dmalloc(scratch_floats_ * sizeof(float), "the eagle3 attention scratch");
    if (!fdec_fits(head_dim_, n_head_, n_kv_, int(cache_cap_), scratch_floats_))
        std::fprintf(stderr,
                     "strata-poly: the eagle3 attention scratch is short for %lld cache rows - the draft's "
                     "attention falls back shorter\n",
                     (long long) cache_cap_);
    hidden_ = (float*) dmalloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 hidden");
    emb_dev_ = (float*) dmalloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 emb");
    e_norm_ = (float*) dmalloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 e norm");
    f_norm_ = (float*) dmalloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 f norm");
    e_q8_ = dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 e q8_1");
    f_q8_ = dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 f q8_1");
    cat_q8_ = dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(2 * n_embd_), "the eagle3 cat q8_1");
    q_ = (float*) dmalloc(size_t(rows_) * n_head_ * head_dim_ * sizeof(float), "the eagle3 q");
    k_ = (float*) dmalloc(size_t(rows_) * n_kv_ * head_dim_ * sizeof(float), "the eagle3 k");
    v_ = (float*) dmalloc(size_t(rows_) * n_kv_ * head_dim_ * sizeof(float), "the eagle3 v");
    o_ = (float*) dmalloc(size_t(rows_) * n_head_ * head_dim_ * sizeof(float), "the eagle3 o");
    o_q8_ = dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(n_head_ * head_dim_), "the eagle3 o q8_1");
    normed_ = (float*) dmalloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 normed");
    normed_q8_ = dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 normed q8_1");
    gate_out_ = (float*) dmalloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 gate");
    up_out_ = (float*) dmalloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 up");
    mlp_h_ = (float*) dmalloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 mlp");
    logits_ = (float*) dmalloc(size_t(rows_) * dvocab_ * sizeof(float), "the eagle3 logits");
    taps_q8_ = (float*) dmalloc(size_t(rows_) * kernels::native_q8_1_bytes(3 * n_embd_), "the eagle3 taps q8_1");
}

void Eagle3::midlayer(int64_t pos, int n, void* stream) {
    const int hd = n_head_ * head_dim_;
    // the two norms, each quantized into its own Q8_1 half; the second's x is the residual in
    // place (the add is null, so hidden_ keeps itself) - the residual survives both launches
    native_add_rms_norm_q8_1(emb_dev_, nullptr, ln1_, e_norm_, e_q8_, n_embd_, n, eps_, 256, stream);
    native_add_rms_norm_q8_1(hidden_, nullptr, hnorm_, f_norm_, f_q8_, n_embd_, n, eps_, 256, stream);
    // the halves assemble into one strided Q8_1 input the q/k/v projections read
    ck(cudaMemcpy2DAsync(cat_q8_, kernels::native_q8_1_bytes(2 * n_embd_), e_q8_,
                         kernels::native_q8_1_bytes(n_embd_), kernels::native_q8_1_bytes(n_embd_), n,
                         cudaMemcpyDeviceToDevice, cs(stream)), "eagle cat e");
    ck(cudaMemcpy2DAsync((char*) cat_q8_ + kernels::native_q8_1_bytes(n_embd_),
                         kernels::native_q8_1_bytes(2 * n_embd_), f_q8_, kernels::native_q8_1_bytes(n_embd_),
                         kernels::native_q8_1_bytes(n_embd_), n, cudaMemcpyDeviceToDevice, cs(stream)),
       "eagle cat f");
    native_mmvq(kTypeQ8_0, q_w_, cat_q8_, q_, 2 * n_embd_, hd, n, stream);
    native_mmvq(kTypeQ8_0, k_w_, cat_q8_, k_, 2 * n_embd_, n_kv_ * head_dim_, n, stream);
    native_mmvq(kTypeQ8_0, v_w_, cat_q8_, v_, 2 * n_embd_, n_kv_ * head_dim_, n, stream);
    dsv4_rope(q_, q_, head_dim_, n_head_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    dsv4_rope(k_, k_, head_dim_, n_kv_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    ck(cudaMemcpyAsync(k_cache_ + size_t(pos) * n_kv_ * head_dim_, k_, size_t(n) * n_kv_ * head_dim_ * 2,
                       cudaMemcpyDeviceToDevice, cs(stream)), "eagle k store");
    ck(cudaMemcpyAsync(v_cache_ + size_t(pos) * n_kv_ * head_dim_, v_, size_t(n) * n_kv_ * head_dim_ * 2,
                       cudaMemcpyDeviceToDevice, cs(stream)), "eagle v store");
    if (!fdec_f16(q_, k_cache_, v_cache_, pos, n, nullptr, 0, o_, head_dim_, n_head_, n_kv_,
                  1.f / std::sqrt(float(head_dim_)), scratch_, scratch_floats_, stream))
        throw std::runtime_error("the eagle3 attention does not fit its scratch");
    // the residual + the norm + the quantize in one launch - the mlp's input ready
    native_add_rms_norm_q8_1(hidden_, o_, ln2_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
    native_mmvq(kTypeQ8_0, gate_w_, normed_q8_, gate_out_, n_embd_, n_ff_, n, stream);
    native_mmvq(kTypeQ8_0, up_w_, normed_q8_, up_out_, n_embd_, n_ff_, n, stream);
    native_swiglu_quantize_q8_1(gate_out_, up_out_, normed_q8_, n_ff_, n, stream);
    native_mmvq(kTypeQ8_0, down_w_, normed_q8_, o_, n_ff_, n_embd_, n, stream);
    // the residual + the final norm + the quantize in one launch - the lm_head's input ready
    native_add_rms_norm_q8_1(hidden_, o_, out_norm_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
}

void Eagle3::inject(const float* taps_dev, const float* emb_dev, int64_t pos, int n, void* stream) {
    // the taps' stack [rows x 3 x n_embd] through the fc, the output lands in hidden_ (the residual)
    native_quantize_q8_1(taps_dev, taps_q8_, 3 * n_embd_, n, stream);
    native_mmvq(kTypeQ8_0, fc_w_, taps_q8_, hidden_, 3 * n_embd_, n_embd_, n, stream);
    midlayer(emb_dev, pos, n, stream);
    cache_pos_ = pos + n;
}

const float* Eagle3::draft_logits(int32_t token, const float* emb_host, void* stream) {
    ck(cudaMemcpyAsync(emb_dev_, emb_host, n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(stream)),
       "eagle draft emb");
    midlayer(cache_pos_, 1, stream);
    cache_pos_ += 1;
    native_mmvq(kTypeQ8_0, lm_head_w_, normed_q8_, logits_, n_embd_, dvocab_, 1, stream);
    return logits_;
}

}  // namespace strata::poly
