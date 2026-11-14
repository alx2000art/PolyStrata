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
// rows (linear(): a prompt batch through Mmq, a few tokens through native_mmvq at 8 a launch).
#include "eagle3.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_fp16.h>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace strata::poly {

namespace {
constexpr int kTypeQ8_0 = 6;   // the ggml type id the mmvq kernels take (this fork passes the numbers)
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
        const __half2 dh = __floats2half2_rn(d, d);
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
    const bool yarn = factor > 1.f;
    rope_ = make_rope(base, head_dim_, yarn ? 1.f / factor : 1.f, yarn ? 1.f : 0.f, orig, 32.f, 1.f);

    // the vocab mapping
    if (const MetaValue* t = kv("eagle3.d2t"); t && t->type == MetaType::ARRAY) {
        d2t_.reserve(t->items.size());
        for (const MetaValue& v : t->items) d2t_.push_back(int64_t(v.u));
    }

    // the weights: BF16 mapped -> Q8_0 on card.  kTail zero bytes follow every matrix the Mmq
    // kernels may multiply: they read up to a step of 256 columns past a row's end.
    const auto w = [&](const char* name) {
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        const size_t n_vals = t->elements();
        std::vector<uint8_t> q;
        bf16_to_q8_0(dg.data(*t), q, n_vals);
        void* dev = nullptr; ck(cudaMalloc(&dev, q.size() + 1024), name);
        ck(cudaMemcpy(dev, q.data(), q.size(), cudaMemcpyHostToDevice), "eagle weight");
        ck(cudaMemsetAsync((char*) dev + q.size(), 0, 1024, nullptr), "eagle weight tail");
        return dev;
    };
    fc_w_ = w("fc.weight");
    fc_bytes_ = size_t(dg.find("fc.weight")->elements()) / 32 * 34;
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
    ck(cudaMalloc(&cat_q8_, size_t(rows_) * kernels::native_q8_1_bytes(2 * n_embd_)), "the eagle3 cat q8_1");
    ck(cudaMalloc(&gate_out_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 gate");
    ck(cudaMalloc(&up_out_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 up");
    ck(cudaMalloc(&mlp_h_, size_t(rows_) * n_ff_ * sizeof(float)), "the eagle3 mlp");
    ck(cudaMalloc(&mlp_q8_, size_t(rows_) * kernels::native_q8_1_bytes(n_ff_)), "the eagle3 mlp q8_1");
    ck(cudaMalloc(&logits_, size_t(rows_) * dvocab_ * sizeof(float)), "the eagle3 logits");
    ck(cudaMalloc(&taps_q8_, size_t(rows_) * kernels::native_q8_1_bytes(3 * n_embd_)), "the eagle3 taps q8_1");
    // the prompt injects' batched projections: llama.cpp's MMQ over the Q8_0 weights, if the card
    // has a tile for every shape the head multiplies
    const int hd = n_head_ * head_dim_;
    mq_fits_ = Mmq::fits(kTypeQ8_0, n_embd_, 3 * n_embd_) && Mmq::fits(kTypeQ8_0, hd, 2 * n_embd_) &&
               Mmq::fits(kTypeQ8_0, n_kv_ * head_dim_, 2 * n_embd_) && Mmq::fits(kTypeQ8_0, n_ff_, n_embd_) &&
               Mmq::fits(kTypeQ8_0, n_embd_, n_ff_);
    ck(cudaMalloc(&mqx_, Mmq::q8_bytes(rows_, n_ff_)), "the eagle3 mmq x");
    if (!mq_fits_)
        std::fprintf(stderr, "strata-poly: the eagle3 head has no MMQ tile for a shape - its prompt "
                             "injections fall back to the 8-row launches\n");
}

// One projection of n rows: a prompt batch (n >= kStageRows) through Mmq - llama.cpp's MMQ reads
// the Q8_0 weights once for all the rows; a few tokens quantize x into xq and go through
// native_mmvq, a launch taking 8 tokens at most, each coming out as it would alone.
void Eagle3::linear(const void* w, const float* x, void* xq, int n_in, int n_out, float* y, int n, void* stream) {
    if (n >= kStageRows && mq_fits_) {
        Mmq::quantize(x, nullptr, mqx_, kTypeQ8_0, n_in, n_in, n, stream);
        mq_.mul(w, kTypeQ8_0, n_out, n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, n_out, stream);
        return;
    }
    kernels::native_quantize_q8_1(x, xq, n_in, n, stream);
    const size_t token = kernels::native_q8_1_bytes(n_in);
    for (int at = 0; at < n; at += 8)
        kernels::native_mmvq(kTypeQ8_0, w, (const uint8_t*) xq + (size_t) at * token,
                             y + (size_t) at * n_out, n_in, n_out, std::min(8, n - at), stream);
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
    linear(q_w_, cat_f32_, cat_q8_, 2 * n_embd_, hd, q_, n, stream);
    linear(k_w_, cat_f32_, cat_q8_, 2 * n_embd_, n_kv_ * head_dim_, k_, n, stream);
    linear(v_w_, cat_f32_, cat_q8_, 2 * n_embd_, n_kv_ * head_dim_, v_, n, stream);
    dsv4_rope(q_, q_, head_dim_, n_head_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    dsv4_rope(k_, k_, head_dim_, n_kv_, 0, head_dim_, rope_, pos, 1, n, false, 0.f, false, stream, nullptr);
    ck(cudaMemcpyAsync(k_cache_ + size_t(pos) * n_kv_ * head_dim_, k_, size_t(n) * n_kv_ * head_dim_ * 2,
                       cudaMemcpyDeviceToDevice, cs(stream)), "eagle k store");
    ck(cudaMemcpyAsync(v_cache_ + size_t(pos) * n_kv_ * head_dim_, v_, size_t(n) * n_kv_ * head_dim_ * 2,
                       cudaMemcpyDeviceToDevice, cs(stream)), "eagle v store");
    // the causal attention over the whole linear cache (the head's 8 heads a group misses the
    // dim-64 fast path's per == 4, so the cuBLAS path takes it)
    attend_batch(blas_, q_, k_cache_, v_cache_, pos, n, score_, kScoreFloats, work_, o_, head_dim_,
                 n_head_, n_kv_, 1.f / std::sqrt(float(head_dim_)), stream);
    // the residual + the norm (+ the quantize the few-token path's projections read) in one launch
    kernels::native_add_rms_norm_q8_1(hidden_, o_, ln2_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
    linear(gate_w_, normed_, normed_q8_, n_embd_, n_ff_, gate_out_, n, stream);
    linear(up_w_, normed_, normed_q8_, n_embd_, n_ff_, up_out_, n, stream);
    // silu(gate) x up in fp32 (the limit makes swiglu_clamp's min/clamp no-ops) - the poly
    // kernels live in strata::poly itself, no namespace step
    swiglu_clamp(gate_out_, up_out_, mlp_h_, size_t(n) * n_ff_, 1e30f, stream);
    linear(down_w_, mlp_h_, mlp_q8_, n_ff_, n_embd_, o_, n, stream);
    // the residual + the final norm (+ the quantize) in one launch - the lm_head's input ready
    kernels::native_add_rms_norm_q8_1(hidden_, o_, out_norm_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
}

void Eagle3::inject(const float* taps_dev, float* emb_dev, int64_t pos, int n, void* stream) {
    // the taps' stack [rows x 3 x n_embd] through the fc, the output lands in hidden_ (the residual)
    linear(fc_w_, taps_dev, taps_q8_, 3 * n_embd_, n_embd_, hidden_, n, stream);
    if (dbg_inj_ < 2) {   // one-shot: sane taps in + sane fc out means the capture's layout, not the head
        ++dbg_inj_;
        std::vector<float> th(size_t(n) * 3 * n_embd_), hh(size_t(n) * n_embd_);
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg sync");
        ck(cudaMemcpy(th.data(), taps_dev, th.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dbg taps");
        ck(cudaMemcpy(hh.data(), hidden_, hh.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dbg hidden");
        const auto st = [](const std::vector<float>& v) {
            double s = 0; float mn = v[0], mx = v[0];
            for (float x : v) { s += x; mn = std::min(mn, x); mx = std::max(mx, x); }
            return std::string("mean=") + std::to_string(s / v.size()) + " min=" + std::to_string(mn) +
                   " max=" + std::to_string(mx) + " head: " + std::to_string(v[0]) + " " +
                   std::to_string(v[1]) + " " + std::to_string(v[2]);
        };
        std::fprintf(stderr, "[eagle] inject n=%d pos=%lld taps %s\n", n, (long long) pos, st(th).c_str());
        std::fprintf(stderr, "[eagle] inject fc->hidden %s\n", st(hh).c_str());
        // the bisect: the same fc through the few-token path (mmvq, 8 rows a launch) into o_ - sane
        // here with NaN from Mmq names the batched chain, NaN in both names the weights
        for (int at = 0; at < n; at += 8) {
            const int c = std::min(8, n - at);
            kernels::native_quantize_q8_1(taps_dev + size_t(at) * 3 * n_embd_, taps_q8_, 3 * n_embd_, c, stream);
            kernels::native_mmvq(kTypeQ8_0, fc_w_, taps_q8_, o_ + size_t(at) * n_embd_, 3 * n_embd_, n_embd_, c,
                                 stream);
        }
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg sync3");
        std::vector<float> oo(size_t(n) * n_embd_);
        ck(cudaMemcpy(oo.data(), o_, oo.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dbg o");
        std::fprintf(stderr, "[eagle] inject fc(mmvq) %s\n", st(oo).c_str());
        // the fc's own Q8_0 blocks (start / middle / last): the fp16 scale and the int8 spread - an
        // inf or nan scale here names the host quantizer, sane blocks name the activation side
        const size_t blocks = fc_bytes_ / 34;
        for (size_t which : {size_t(0), blocks / 2, blocks - 1}) {
            uint8_t b34[34];
            ck(cudaMemcpy(b34, (const char*) fc_w_ + which * 34, 34, cudaMemcpyDeviceToHost), "eagle dbg w");
            uint16_t h;
            std::memcpy(&h, b34, 2);
            const float d = __half2float(*(__half*) &h);
            int8_t mn = 127, mx = -128;
            for (int i = 2; i < 34; ++i) {
                mn = std::min(mn, (int8_t) b34[i]);
                mx = std::max(mx, (int8_t) b34[i]);
            }
            std::fprintf(stderr, "[eagle] fc block@%zu/%zu: d=%.6f q8 [%d..%d]\n", which, blocks, d, (int) mn,
                         (int) mx);
        }
        // the taps' first Q8_1 block as the native quantize wrote it: the scale and the original
        // values' sum (ds.y) - a sum past 65504 turned inf by the sink-channel outliers would name
        // the activation quantize
        uint8_t b36[36];
        ck(cudaMemcpy(b36, taps_q8_, 36, cudaMemcpyDeviceToHost), "eagle dbg x");
        __half2 ds;
        std::memcpy(&ds, b36, 4);
        std::fprintf(stderr, "[eagle] taps q8_1 blk0: d=%.5f dsum=%.1f\n", __half2float(__low2half(ds)),
                     __half2float(__high2half(ds)));
    }
    midlayer(emb_dev, pos, n, stream);
    cache_pos_ = pos + n;
}

const float* Eagle3::draft_logits(int32_t token, const float* emb_host, void* stream) {
    ck(cudaMemcpyAsync(emb_dev_, emb_host, n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(stream)),
       "eagle draft emb");
    midlayer(emb_dev_, cache_pos_, 1, stream);
    cache_pos_ += 1;
    kernels::native_mmvq(kTypeQ8_0, lm_head_w_, normed_q8_, logits_, n_embd_, dvocab_, 1, stream);
    if (dbg_logits_ < 2) {   // one-shot: the logits' scale tells garbage (taps/fc/midlayer) from a broken d2t
        ++dbg_logits_;
        std::vector<float> lg((size_t) dvocab_);
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg sync2");
        ck(cudaMemcpy(lg.data(), logits_, lg.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dbg logits");
        double s = 0; float mx = lg[0];
        for (float x : lg) { s += x; mx = std::max(mx, x); }
        std::fprintf(stderr, "[eagle] logits mean=%.5f max=%.3f\n", s / lg.size(), mx);
    }
    return logits_;
}

}  // namespace strata::poly
