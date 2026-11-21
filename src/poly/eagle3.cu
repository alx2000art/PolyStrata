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
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <thread>
#include <chrono>

namespace strata::poly {

namespace {
constexpr float kPi = 3.14159265358979323846f;
// the type id native_mmvq.hpp numbers: Q8_0 is 8 (6 is Q5_0 - passing it made the kernels read the
// head's 34-byte Q8_0 blocks as 22-byte Q5_0 ones: the old NaN / degenerate-logits "Q8_0 mystery")
constexpr int kTypeQ8_0 = 8;

// The card's free VRAM wobbles for a second or two while the previous engine's context drains (the
// server restarts this engine right after the last one), and the head's ~2.5 GB sit in the narrow
// margin beside the model: every allocation retries through the wobble before it gives up.
void* eagle_alloc(size_t bytes, const char* what) {
    void* dev = nullptr;
    cudaError_t e = cudaSuccess;
    for (int a = 0; a < 5; ++a) {
        e = cudaMalloc(&dev, bytes);
        if (e == cudaSuccess) return dev;
        std::fprintf(stderr, "strata-poly: the eagle3 head: %s: %s (retry %d/5)\n", what,
                     cudaGetErrorString(e), a + 1);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    ck(e, what);
    return dev;
}

// The BF16 weights quantize to Q8_0 on the host: a block of 32 values, the scale the max's / 127
// in fp16, the values rounded - the GGML Q8_0 layout the mmvq kernels read.
void bf16_to_q8_0(const uint8_t* src, std::vector<uint8_t>& dst, size_t n_vals) {
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

// DBG(eagle3) STRATA_EAGLE_DUMP_ATT: the attention dump's gate - teacher_step arms it, midlayer takes
// the dump right after attend_batch (o_ is overwritten by the down projection later in the call, so a
// dump taken at the end of the step holds the mlp's output, not the attention's)
static const char* g_att_path = std::getenv("STRATA_EAGLE_DUMP_ATT");
static long g_att_min_pos = g_att_path && std::getenv("STRATA_EAGLE_DUMP_ATT_POS")
                                ? std::atol(std::getenv("STRATA_EAGLE_DUMP_ATT_POS")) : 0;
static bool g_att_armed = false;
static bool g_att_done = false;

void Eagle3::load(Shards& dg) {
    // STRATA_EAGLE_CAT=he: the midlayer's input written [hidden | embedding] instead of [embedding | hidden]
    if (const char* v = std::getenv("STRATA_EAGLE_CAT")) cat_he_ = std::string(v) == "he";
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
    const auto beta = [&](const char* k, float dflt) {
        const MetaValue* b = kv(k);
        return b && b->f > 0.f ? float(b->f) : dflt;
    };
    rope_ = make_rope(base, head_dim_, yarn ? 1.f / factor : 1.f, yarn ? 1.f : 0.f, orig,
                      beta("eagle3.rope.scaling.beta_fast", 32.f), beta("eagle3.rope.scaling.beta_slow", 1.f));
    // The rope's magnitude: transformers' yarn (4.57) computes attention_factor as
    // get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim) - the head's config carries both
    // and they are equal (0.707), so the ratio is 1 and the rope's cos and sin carry no magnitude.  The
    // fork's yarn multiplies attn_factor by (1 + 0.1 ln s) inside the kernel, so the default attn_factor
    // (1 / (1 + 0.1 ln s)) gives the effective 1 the reference has.  The file's mscale alone would have
    // given 0.903 - the attention's logits 0.82 of what they should be.
    // DBG(eagle3) STRATA_EAGLE_ROPE_SCALE=<effective>: the magnitude the reference would have used
    if (const char* rs = std::getenv("STRATA_EAGLE_ROPE_SCALE"); rs && *rs) {
        const float boost = 1.f + 0.1f * std::log(1.f / rope_.freq_scale);
        rope_.attn_factor = float(std::atof(rs)) / boost;
    }
    std::fprintf(stderr, "strata-poly: the eagle3 rope: base %g, freq_scale %g, attn_factor %g, effective %g, "
                 "corr0 %g corr1 %g (n_rot %d)\n", base,
                 rope_.freq_scale, rope_.attn_factor, rope_.attn_factor * (1.f + 0.1f * std::log(1.f / rope_.freq_scale)),
                 (double) rope_.corr0, (double) rope_.corr1, head_dim_);

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
        void* dev = eagle_alloc(n_bytes + 256, name);
        ck(cudaMemcpy(dev, dg.data(*t), n_bytes, cudaMemcpyHostToDevice), "eagle weight");
        ck(cudaMemsetAsync((char*) dev + n_bytes, 0, 256, nullptr), "eagle weight tail");
        return dev;
    };
    fc_w_ = w("fc.weight");
    // The head's rope is transformers' Llama rope: rotate_half turns the dims (i, i + head_dim / 2)
    // together, the pair's angle being base^(-2i/head_dim).  The fork's dsv4_rope turns the NEIGHBOURS
    // (2i, 2i + 1) - so the head's q and k rows are permuted here and the neighbours become the pair
    // (i, i + 32).  The angle, the ramp's dim and the pairing then all match the reference.  The values
    // are not roped, so the o_proj keeps its order and the attention's output needs no undoing.
    // DBG(eagle3) STRATA_EAGLE_ROPE_NEOX=0: the rows as they lie (the neighbours' pairing)
    const bool neox = [] {
        const char* v = std::getenv("STRATA_EAGLE_ROPE_NEOX");
        return !(v && v[0] == '0');
    }();
    const auto w_rope = [&](const char* name, int n_out) {
        if (!neox) return w(name);
        const TensorInfo* t = dg.find(name);
        if (!t) throw std::runtime_error(std::string("the eagle3 head has no ") + name);
        const size_t n_in = t->elements() / size_t(n_out);
        const uint16_t* src = (const uint16_t*) dg.data(*t);
        std::vector<uint16_t> buf((size_t) n_out * n_in);
        const int hd = head_dim_;
        for (int h = 0; h < n_out / hd; ++h)
            for (int i = 0; i < hd / 2; ++i) {
                const uint16_t* a = src + ((size_t) h * hd + i) * n_in;
                const uint16_t* b = src + ((size_t) h * hd + i + hd / 2) * n_in;
                std::memcpy(&buf[((size_t) h * hd + 2 * i) * n_in], a, n_in * 2);
                std::memcpy(&buf[((size_t) h * hd + 2 * i + 1) * n_in], b, n_in * 2);
            }
        void* dev = eagle_alloc(buf.size() * 2 + 256, name);
        ck(cudaMemcpy(dev, buf.data(), buf.size() * 2, cudaMemcpyHostToDevice), "eagle weight");
        ck(cudaMemsetAsync((char*) dev + buf.size() * 2, 0, 256, nullptr), "eagle weight tail");
        return dev;
    };
    q_w_ = w_rope("midlayer.self_attn.q_proj.weight", n_head_ * head_dim_);
    k_w_ = w_rope("midlayer.self_attn.k_proj.weight", n_kv_ * head_dim_);
    v_w_ = w("midlayer.self_attn.v_proj.weight");
    o_w_ = w("midlayer.self_attn.o_proj.weight");
    gate_w_ = w("midlayer.mlp.gate_proj.weight");
    up_w_ = w("midlayer.mlp.up_proj.weight");
    down_w_ = w("midlayer.mlp.down_proj.weight");
    // the lm_head: Q8_0 by default (417 MB of reads a draft step; the type id is 8 now).  The TEACH
    // metric reads the same path and matched the BF16 build's numbers bit for bit when the id was fixed
    // (X 40.2/39.1 against 40.2/38.4).  STRATA_EAGLE_LM_Q8=0 keeps the BF16 copy (786 MB).
    lm_bf16_ = [] {
        const char* v = std::getenv("STRATA_EAGLE_LM_Q8");
        return v && v[0] == '0';
    }();
    if (lm_bf16_) {
        lm_head_w_ = w("lm_head.weight");
    } else {
        const TensorInfo* t = dg.find("lm_head.weight");
        if (!t) throw std::runtime_error("the eagle3 head has no lm_head.weight");
        std::vector<uint8_t> q;
        bf16_to_q8_0(dg.data(*t), q, t->elements());
        lm_q8_ = eagle_alloc(q.size() + 256, "lm_head.weight");
        ck(cudaMemcpy(lm_q8_, q.data(), q.size(), cudaMemcpyHostToDevice), "eagle lm_head");
        ck(cudaMemsetAsync((char*) lm_q8_ + q.size(), 0, 256, nullptr), "eagle lm_head tail");
    }
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
        float* dev = (float*) eagle_alloc(n_vals * sizeof(float), name);
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
    k_cache_ = (uint16_t*) eagle_alloc(size_t(cache_cap_) * n_kv_ * head_dim_ * 2, "the eagle3 k cache");
    v_cache_ = (uint16_t*) eagle_alloc(size_t(cache_cap_) * n_kv_ * head_dim_ * 2, "the eagle3 v cache");
    // the attention's scratch: the score blocks and the grouped queries/results the cuBLAS path
    // keeps there (attend_batch's doc)
    score_ = (float*) eagle_alloc(kScoreFloats * sizeof(float), "the eagle3 attention score");
    const size_t work_floats = size_t(rows_) * n_head_ * (head_dim_ + head_dim_ / 2 + 3) + 1;
    work_ = (float*) eagle_alloc(work_floats * sizeof(float), "the eagle3 attention work");
    hidden_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 hidden");
    emb_dev_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 emb");
    e_norm_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 e norm");
    f_norm_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 f norm");
    e_q8_ = eagle_alloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 e q8_1");
    f_q8_ = eagle_alloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 f q8_1");
    q_ = (float*) eagle_alloc(size_t(rows_) * n_head_ * head_dim_ * sizeof(float), "the eagle3 q");
    k_ = (float*) eagle_alloc(size_t(rows_) * n_kv_ * head_dim_ * sizeof(float), "the eagle3 k");
    v_ = (float*) eagle_alloc(size_t(rows_) * n_kv_ * head_dim_ * sizeof(float), "the eagle3 v");
    o_ = (float*) eagle_alloc(size_t(rows_) * n_head_ * head_dim_ * sizeof(float), "the eagle3 o");
    o_out_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 o proj");
    att_dump_ = (float*) eagle_alloc(size_t(n_head_) * head_dim_ * sizeof(float), "the eagle3 att dump");
    fc_state_ = (float*) eagle_alloc(size_t(n_embd_) * sizeof(float), "the eagle3 fc state");
    normed_ = (float*) eagle_alloc(size_t(rows_) * n_embd_ * sizeof(float), "the eagle3 normed");
    normed_q8_ = eagle_alloc(size_t(rows_) * kernels::native_q8_1_bytes(n_embd_), "the eagle3 normed q8_1");
    cat_f32_ = (float*) eagle_alloc(size_t(rows_) * 2 * n_embd_ * sizeof(float), "the eagle3 cat fp32");
    gate_out_ = (float*) eagle_alloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 gate");
    up_out_ = (float*) eagle_alloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 up");
    mlp_h_ = (float*) eagle_alloc(size_t(rows_) * n_ff_ * sizeof(float), "the eagle3 mlp");
    logits_ = (float*) eagle_alloc(size_t(dvocab_) * sizeof(float), "the eagle3 logits");
    // the prompt batches' F32 view of one BF16 matrix: the widest the head dequantizes is n_ff x n_embd
    wf_e_ = (float*) eagle_alloc(size_t(n_ff_) * n_embd_ * sizeof(float), "the eagle3 weight f32 view");
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
    // the halves assemble into [e | h] in fp32 (STRATA_EAGLE_CAT=he: [h | e], which the trained head may
    // expect instead - the order is the one thing about the concat that is not in the weights)
    const float* lo = cat_he_ ? f_norm_ : e_norm_;
    const float* hi = cat_he_ ? e_norm_ : f_norm_;
    ck(cudaMemcpy2DAsync(cat_f32_, 2 * n_embd_ * sizeof(float), lo, n_embd_ * sizeof(float),
                         n_embd_ * sizeof(float), n, cudaMemcpyDeviceToDevice, cs(stream)), "eagle cat lo");
    ck(cudaMemcpy2DAsync(cat_f32_ + n_embd_, 2 * n_embd_ * sizeof(float), hi, n_embd_ * sizeof(float),
                         n_embd_ * sizeof(float), n, cudaMemcpyDeviceToDevice, cs(stream)), "eagle cat hi");
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
    // DBG(eagle3) STRATA_EAGLE_DUMP_STAGE=<file>: one single-row step's stages, taken HERE because the
    // gate reuses q_ and the up reuses k_ further down.  The logits of the trainer and the engine
    // disagree at cos ~0.23 with the engine's own cache fed back, so the difference is either in the
    // K/V rows (data) or in q/k/v (weights/rope): e_norm/f_norm/q/k/v/o say which, stage by stage.
    static const char* g_stage_path = std::getenv("STRATA_EAGLE_DUMP_STAGE");
    // _POS: which position to take (the round dump names it); without it the first single-row step wins,
    // and that can be a read-tail inject rather than the draft this dump is for
    static const long long g_stage_pos = [] {
        const char* v = std::getenv("STRATA_EAGLE_DUMP_STAGE_POS");
        return v ? std::atoll(v) : -1LL;
    }();
    static bool g_stage_done = false;
    // the first single-row steps name themselves: a gate that never fires is usually a position the
    // caller did not expect (a read tail, an eval, or cache_pos after a rejected round)
    if (g_stage_path && n == 1) {
        static int staged = 0;
        if (staged < 8) {
            ++staged;
            std::fprintf(stderr, "[eagle] stage gate: n=1 pos=%lld want=%lld done=%d\n",
                         (long long) pos, (long long) g_stage_pos, (int) g_stage_done);
        }
    }
    if (g_stage_path && !g_stage_done && n == 1 && (g_stage_pos < 0 || (long long) pos == g_stage_pos)) {
        g_stage_done = true;
        ck(cudaStreamSynchronize(cs(stream)), "stage dump sync");
        if (FILE* f = std::fopen(g_stage_path, "wb")) {
            const int32_t hdr[3] = {n_embd_, hd, n_kv_ * head_dim_};   // e/f: n_embd, q/o: hd, k/v: kv*hd
            const int64_t p64 = (int64_t) pos;
            std::fwrite(hdr, sizeof hdr, 1, f);
            std::fwrite(&p64, sizeof p64, 1, f);
            struct Part { const float* src; size_t n; };
            // hidden and emb ride along: the stage file is then self-contained (the checker needs no
            // round dump and no guessed position), and e/f are recomputable from them
            const Part parts[8] = {{hidden_, (size_t) n_embd_}, {emb_dev_, (size_t) n_embd_},
                                   {e_norm_, (size_t) n_embd_}, {f_norm_, (size_t) n_embd_},
                                   {q_, (size_t) hd}, {k_, (size_t) n_kv_ * head_dim_},
                                   {v_, (size_t) n_kv_ * head_dim_}, {o_, (size_t) hd}};
            std::vector<float> buf;
            for (const Part& pt : parts) {
                buf.resize(pt.n);
                std::fwrite(&pt.n, sizeof(size_t), 1, f);
                // a device read per part: the stage is one row, so this is a few hundred KB in total
                ck(cudaMemcpy(buf.data(), pt.src, pt.n * sizeof(float), cudaMemcpyDeviceToHost), "stage dump");
                std::fwrite(buf.data(), sizeof(float), pt.n, f);
            }
            std::fclose(f);
            std::fprintf(stderr, "[eagle] stage dump: pos %lld -> %s (e,f,q,k,v,o)\n",
                         (long long) pos, g_stage_path);
        }
    }
    if (dump_att_ && att_dump_ && dump_row_ < (size_t) n) {
        dump_att_ = false;
        ck(cudaMemcpyAsync(att_dump_, o_ + dump_row_ * (size_t) n_head_ * head_dim_,
                           (size_t) n_head_ * head_dim_ * sizeof(float), cudaMemcpyDeviceToDevice, cs(stream)),
           "eagle att dump copy");
    }
    if (g_att_path && g_att_armed && !g_att_done && (long long) pos >= g_att_min_pos) {   // DBG(eagle3): o_ dies
        g_att_done = true;                            // at the down projection - the dump
                                                      // must read it here, while it still holds the attention
        ck(cudaStreamSynchronize(cs(stream)), "att dump sync");
        FILE* f = std::fopen(g_att_path, "wb");
        if (f) {
            const int rows = int(pos) + 1;
            std::fprintf(f, "%d %d %d %d %d\n", (int) pos, n_head_, n_kv_, head_dim_, rows);
            std::vector<float> ob((size_t) n_head_ * head_dim_);
            ck(cudaMemcpy(ob.data(), q_, ob.size() * sizeof(float), cudaMemcpyDeviceToHost), "att dump q");
            std::fwrite(ob.data(), sizeof(float), ob.size(), f);
            std::vector<uint16_t> kvh((size_t) rows * n_kv_ * head_dim_);
            ck(cudaMemcpy(kvh.data(), k_cache_, kvh.size() * 2, cudaMemcpyDeviceToHost), "att dump k");
            std::fwrite(kvh.data(), 2, kvh.size(), f);
            ck(cudaMemcpy(kvh.data(), v_cache_, kvh.size() * 2, cudaMemcpyDeviceToHost), "att dump v");
            std::fwrite(kvh.data(), 2, kvh.size(), f);
            ck(cudaMemcpy(ob.data(), o_, ob.size() * sizeof(float), cudaMemcpyDeviceToHost), "att dump o");
            std::fwrite(ob.data(), sizeof(float), ob.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[eagle] dumped the attention at pos %lld (rows %d)\n", (long long) pos, rows);
        }
    }
    if (dbg_att_ < 1) {   // DBG(eagle3) the first inject.s attention output: finite means the cache and the softmax hold
        ++dbg_att_;
        float h[4] = {};
        ck(cudaStreamSynchronize(cs(stream)), "eagle dbg syncA");
        ck(cudaMemcpy(h, o_, sizeof h, cudaMemcpyDeviceToHost), "eagle dbg att");
        std::fprintf(stderr, "[eagle] att->o (pos %lld): %.4f %.4f %.4f %.4f\n", (long long) pos, h[0], h[1], h[2],
                     h[3]);
    }
    // the attention's output projection (the reference's o_proj: the head outputs x o_proj) - without
    // it the raw head outputs went into the residual and the head's guess was noise
    linear(o_w_, o_, n_head_ * head_dim_, n_embd_, o_out_, n, stream);
    // the residual + the norm (and its Q8_1, which nothing reads any more) in one launch
    kernels::native_add_rms_norm_q8_1(hidden_, o_out_, ln2_, normed_, normed_q8_, n_embd_, n, eps_, 256, stream);
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
    if (dbg_inj_ < 1) {   // DBG(eagle3) the first inject only: the taps' range as the head sees it
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
    // DBG(eagle3) STRATA_EAGLE_DUMP=<file>: the first inject's taps and its fc output BEFORE the midlayer
    // touches hidden_ (after it, hidden_ is the residual's sum, not what fc produced).
    // DBG(eagle3) STRATA_EAGLE_DUMP_ROW=<i>: dump the batch's row i (STRATA_EAGLE_DUMP_ROW0=1 is row 0).
    // Row 0's attention attends only itself (the rope at pos 0 is the identity), so the trainer
    // reproduces it from (taps, emb) alone; a later row also checks the rope's phase and the cache.
    const char* dpath = std::getenv("STRATA_EAGLE_DUMP");
    const bool dumping = dpath && !dumped_ && n > 1;
    size_t drow = (size_t) (n - 1);
    if (const char* dr = std::getenv("STRATA_EAGLE_DUMP_ROW"); dr && *dr)
        drow = std::min((size_t) std::max(0, std::atoi(dr)), (size_t) (n - 1));
    else if (std::getenv("STRATA_EAGLE_DUMP_ROW0"))
        drow = 0;
    std::vector<float> fc_dump;
    if (dumping) {
        ck(cudaStreamSynchronize(cs(stream)), "eagle dump sync");
        fc_dump.assign((size_t) n_embd_, 0.f);
        ck(cudaMemcpy(fc_dump.data(), hidden_ + drow * n_embd_, fc_dump.size() * sizeof(float),
                      cudaMemcpyDeviceToHost), "eagle dump fc");
    }
    // the round's first step's state: fc(taps) as it is, before the midlayer adds the residual
    if (fc_state_)
        ck(cudaMemcpyAsync(fc_state_, hidden_ + size_t(n - 1) * n_embd_, size_t(n_embd_) * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(stream)),
           "the eagle fc state");
    dump_att_ = dumping;
    dump_row_ = drow;
    midlayer(emb_dev, pos, n, stream);
    if (dumping) {
        dumped_ = true;
        FILE* f = std::fopen(dpath, "wb");
        if (f) {
            // the stages in order: taps (3*n_embd), fc out, the embedding, ln1 out, hnorm out, the
            // final normed state, the lm_head's logits - a numpy forward can be run on each of them
            const size_t e = (size_t) n_embd_;
            std::vector<float> buf;
            const auto put = [&](const float* dev, size_t cnt, size_t row, size_t stride) {
                buf.assign(cnt, 0.f);
                ck(cudaMemcpy(buf.data(), dev + row * stride, cnt * sizeof(float), cudaMemcpyDeviceToHost),
                   "eagle dump");
                std::fwrite(buf.data(), sizeof(float), cnt, f);
            };
            put(taps_dev, 3 * e, drow, 3 * e);
            std::fwrite(fc_dump.data(), sizeof(float), fc_dump.size(), f);
            put(emb_dev, e, drow, e);
            put(e_norm_, e, drow, e);
            put(f_norm_, e, drow, e);
            put(normed_, e, drow, e);
            // the layer's two halves apart: hidden = fc + o_out + o(mlp).  o_out names the attention
            // (its direction and its scale), the mlp's output names the rest
            put(o_out_, e, drow, e);
            put(o_, e, drow, e);
            // and the attention's own inputs: q (after the rope), k, v of the row - a divergence here
            // names the projections/rope, a match names the softmax/mask
            put(q_, (size_t) n_head_ * head_dim_, drow, (size_t) n_head_ * head_dim_);
            put(k_, (size_t) n_kv_ * head_dim_, drow, (size_t) n_kv_ * head_dim_);
            put(v_, (size_t) n_kv_ * head_dim_, drow, (size_t) n_kv_ * head_dim_);
            put(att_dump_, e, 0, e);   // the attention's output (pre-o_proj) of the dumped row
            std::fclose(f);
            std::fprintf(stderr, "[eagle] dumped the stages to %s (pos %lld)\n", dpath, (long long) (pos + drow));
        }
    }
    // the draft's feature is the call's LAST row (the token the drafter starts from): move it to row 0,
    // where a one-row midlayer finds it.  Without this the draft read row 0 - the chunk's FIRST token,
    // thousands of positions back - and its guesses were noise.
    if (n > 1)
        ck(cudaMemcpyAsync(hidden_, hidden_ + size_t(n - 1) * n_embd_, size_t(n_embd_) * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(stream)),
           "the eagle's last hidden");
    cache_pos_ = pos + n;
}

// DBG(eagle3) STRATA_EAGLE_TEACH: the head on a KNOWN position - the taps of the row the engine already
// read and the embedding of the token that follows it; its guess is compared with the token after that.
const float* Eagle3::teacher_step(const float* taps_row, const float* emb_host, int64_t pos, void* stream) {
    linear(fc_w_, taps_row, 3 * n_embd_, n_embd_, hidden_, 1, stream);
    // DBG(eagle3) STRATA_EAGLE_DUMP_ATT=<file>: the first teacher step's attention, so a numpy softmax can
    // be run on the very q/k/v it used (pos, then q [n_head*dim], k and v [pos+1][n_kv*dim], then o)
    g_att_armed = g_att_path && !g_att_done;
    if (const char* ab = std::getenv("STRATA_EAGLE_ROUND_AB"); ab && *ab)
        std::fprintf(stderr, "[eagle] teacher at pos %lld (cache_pos %lld, cap %lld)\n", (long long) pos,
                     (long long) cache_pos_, (long long) cache_cap_);
    ck(cudaMemcpyAsync(emb_dev_, emb_host, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(stream)),
       "eagle teach emb");
    midlayer(emb_dev_, pos, 1, stream);
    if (lm_bf16_)
        gemv_bf16((const uint16_t*) lm_head_w_, normed_, logits_, n_embd_, dvocab_, dvocab_, 1, stream);
    else
        kernels::native_mmvq(kTypeQ8_0, lm_q8_, normed_q8_, logits_, n_embd_, dvocab_, 1, stream);
    return logits_;
}

void Eagle3::keep_state(int n, void* stream) {
    if (n < 1) return;
    ck(cudaMemcpyAsync(hidden_, hidden_ + size_t(n - 1) * n_embd_, size_t(n_embd_) * sizeof(float),
                       cudaMemcpyDeviceToDevice, cs(stream)),
       "the eagle's kept recursion state");
}

void Eagle3::fc_into(float* out, const float* taps_dev, void* stream) {
    linear(fc_w_, taps_dev, 3 * n_embd_, n_embd_, out, 1, stream);
}

void Eagle3::restore_fc(void* stream) {
    if (!fc_state_) return;
    ck(cudaMemcpyAsync(hidden_, fc_state_, size_t(n_embd_) * sizeof(float), cudaMemcpyDeviceToDevice, cs(stream)),
       "the eagle's fc state back");
}

// The drafter's own part of a park/restore: hidden_[0] (the chain's input at the save point) and fc_state_
// (the pure fc the round's first guess restores), then the cache rows - a row's k then v, fp16.  The engine's
// save/load state and rows append these to the target's, so Conversations carries the drafter with the
// sequence: a parked-and-resumed conversation keeps a drafter whose cache matches the restored rows.

size_t Eagle3::state_bytes() const {
    return 2 * (size_t) n_embd_ * sizeof(float);
}

void Eagle3::save_state(uint8_t* dst) {
    ck(cudaMemcpy(dst, hidden_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost), "eagle save state");
    if (fc_state_)
        ck(cudaMemcpy(dst + (size_t) n_embd_ * sizeof(float), fc_state_, (size_t) n_embd_ * sizeof(float),
                      cudaMemcpyDeviceToHost), "eagle save state fc");
    else
        std::memset(dst + (size_t) n_embd_ * sizeof(float), 0, (size_t) n_embd_ * sizeof(float));
}

void Eagle3::load_state(const uint8_t* src, int64_t pos) {
    ck(cudaMemcpy(hidden_, src, (size_t) n_embd_ * sizeof(float), cudaMemcpyHostToDevice), "eagle load state");
    if (fc_state_)
        ck(cudaMemcpy(fc_state_, src + (size_t) n_embd_ * sizeof(float), (size_t) n_embd_ * sizeof(float),
                      cudaMemcpyHostToDevice), "eagle load state fc");
    cache_pos_ = pos;                                // the rows [0, pos) are the sequence's again
}

size_t Eagle3::row_bytes() const {
    return 2 * (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
}

void Eagle3::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    const size_t part = (size_t) n * n_kv_ * head_dim_ * sizeof(uint16_t);
    ck(cudaMemcpy(dst, k_cache_ + (size_t) first * n_kv_ * head_dim_, part, cudaMemcpyDeviceToHost),
       "eagle save rows");
    ck(cudaMemcpy(dst + part, v_cache_ + (size_t) first * n_kv_ * head_dim_, part, cudaMemcpyDeviceToHost),
       "eagle save rows v");
}

void Eagle3::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t part = (size_t) count * n_kv_ * head_dim_ * sizeof(uint16_t);
    const size_t skip = (size_t) (first - from) * row;
    ck(cudaMemcpy(k_cache_ + (size_t) first * n_kv_ * head_dim_, src + skip, (size_t) n * row,
                  cudaMemcpyHostToDevice), "eagle load rows");
    ck(cudaMemcpy(v_cache_ + (size_t) first * n_kv_ * head_dim_, src + part + skip, (size_t) n * row,
                  cudaMemcpyHostToDevice), "eagle load rows v");
}

void Eagle3::stage_state(int slot, void* stream) {
    if (slot < 1) return;   // row 0's state is already where it is
    ck(cudaMemcpyAsync(hidden_ + size_t(slot) * n_embd_, hidden_, size_t(n_embd_) * sizeof(float),
                       cudaMemcpyDeviceToDevice, cs(stream)),
       "the eagle's staged row state");
}

const float* Eagle3::draft_logits(int32_t token, const float* emb_host, void* stream) {
    // DBG(eagle3): the draft step's own input - if these are the same for different tokens, the head never
    // sees the embedding and every guess is the same word
    if (dbg_logits_ < 3) {
        ++dbg_logits_;
        float hh[4] = {};
        ck(cudaMemcpy(hh, hidden_, sizeof hh, cudaMemcpyDeviceToHost), "eagle draft h");
        std::fprintf(stderr, "[eagle] draft step: token %d, emb %.4f %.4f %.4f %.4f, hidden %.4f %.4f %.4f\n",
                     (int) token, (double) emb_host[0], (double) emb_host[1], (double) emb_host[2],
                     (double) emb_host[3], (double) hh[0], (double) hh[1], (double) hh[2]);
    }
    ck(cudaMemcpyAsync(emb_dev_, emb_host, n_embd_ * sizeof(float), cudaMemcpyHostToDevice, cs(stream)),
       "eagle draft emb");
    midlayer(emb_dev_, cache_pos_, 1, stream);
    cache_pos_ += 1;
    if (lm_bf16_)
        gemv_bf16((const uint16_t*) lm_head_w_, normed_, logits_, n_embd_, dvocab_, dvocab_, 1, stream);
    else
        kernels::native_mmvq(kTypeQ8_0, lm_q8_, normed_q8_, logits_, n_embd_, dvocab_, 1, stream);
    // DBG(eagle3) once per load: a nan in the logits would silently turn every guess into draft id 0 -
    // named here, with the path that produced them
    static bool nf_checked = false;
    if (!nf_checked) {
        nf_checked = true;
        ck(cudaStreamSynchronize(cs(stream)), "eagle nf sync");
        std::vector<float> lg((size_t) dvocab_);
        ck(cudaMemcpy(lg.data(), logits_, lg.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle nf");
        size_t bad = 0;
        for (float x : lg)
            if (!std::isfinite(x)) ++bad;
        if (bad)
            std::fprintf(stderr, "[eagle] the draft's logits hold %zu nonfinite values of %d (the lm_head's %s path)\n",
                         bad, dvocab_, lm_bf16_ ? "bf16" : "q8_0");
    }
    // DBG(eagle3) the dump.s last block: this step.s logits, appended to what inject wrote
    if (const char* dp = std::getenv("STRATA_EAGLE_DUMP"); dp && dumped_ && !dumped_logits_) {
        dumped_logits_ = true;
        ck(cudaStreamSynchronize(cs(stream)), "eagle dump sync2");
        std::vector<float> lg((size_t) dvocab_);
        ck(cudaMemcpy(lg.data(), logits_, lg.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dump logits");
        FILE* f = std::fopen(dp, "ab");
        if (f) {
            // this step's own normed state first: the lm_head check must use the row the logits came from
            std::vector<float> nm((size_t) n_embd_);
            ck(cudaMemcpy(nm.data(), normed_, nm.size() * sizeof(float), cudaMemcpyDeviceToHost), "eagle dump nm");
            std::fwrite(nm.data(), sizeof(float), nm.size(), f);
            std::fwrite(lg.data(), sizeof(float), lg.size(), f);
            std::fclose(f);
            std::fprintf(stderr, "[eagle] dumped the draft step's logits too (%d)\n", dvocab_);
        }
    }
    return logits_;
}

}  // namespace strata::poly
