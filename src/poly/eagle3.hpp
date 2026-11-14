// src/poly/eagle3.hpp - the EAGLE-3 draft head (AQ-MedAI's DeepSeek-V4-Flash-0731-eagle3): one
// Llama midlayer over [the main model's embedding of the token, fc(the three tapped layers'
// hyper-meaned states)], its own lm_head over a reduced 96,000-token vocabulary, the d2t mapping
// back to the target's ids.  The wiring is SafeAILab/EAGLE's eagle/model/cnets.py,
// LlamaDecoderLayeremb, read and verified.  The weights are converted to Q8_0 at load and stay in
// VRAM (~0.8 GB); the KV is a linear fp16 cache of the target's positions, filled by inject() as
// the target reads them and rewritten at the draft positions by every step of draft_logits().
#pragma once

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <vector>

namespace strata::poly {

class Shards;

class Eagle3 {
public:
    void load(Shards& dg);                     // the geometry, the weights (Q8_0 on card), d2t
    void begin(int rows, int64_t max_context); // the cache and the scratch (rows = the engine's)
    // The call's rows: the taps (the device, rows x 3 x n_embd stacked) and the embeddings (the
    // device, rows x n_embd - the caller staged them) -> fc, the norms, the concat, the midlayer
    // -> the cache rows [pos, pos + n).  The last row's state stays as the draft's feature.
    void inject(const float* taps_dev, const float* emb_dev, int64_t pos, int n, void* stream);
    // One iterative step at cache_pos_: the main model's embedding of token (the host, n_embd
    // floats) rides in, the input [that embedding, the last hidden] -> the midlayer -> lm_head.
    // Returns the logits' device pointer; the caller does the likeliest and the d2t translation.
    const float* draft_logits(int32_t token, const float* emb_host, void* stream);
    int dvocab() const { return dvocab_; }
    int32_t d2t(int32_t draft_id) const { return d2t_[(size_t) draft_id]; }
    bool ready() const { return loaded_; }
    float* emb_dev() { return emb_dev_; }

private:
    void midlayer(float* emb_dev, int64_t pos, int n, void* stream);
    // the weights (Q8_0, the device) and the norms (F32)
    void* fc_w_ = nullptr, *q_w_ = nullptr, *k_w_ = nullptr, *v_w_ = nullptr, *o_w_ = nullptr;
    void* gate_w_ = nullptr, *up_w_ = nullptr, *down_w_ = nullptr, *lm_head_w_ = nullptr;
    float *ln1_ = nullptr, *ln2_ = nullptr, *hnorm_ = nullptr, *out_norm_ = nullptr;
    // the geometry
    int n_embd_ = 0, n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_ff_ = 0, dvocab_ = 0;
    float eps_ = 0;
    Rope rope_;
    // the KV: a linear fp16 cache, k and v of kv_heads * head_dim halves a row
    uint16_t *k_cache_ = nullptr, *v_cache_ = nullptr;
    int64_t cache_cap_ = 0, cache_pos_ = 0;
    int rows_ = 0;
    // the device scratch (rows a first dimension everywhere)
    float *hidden_ = nullptr;     // n_embd a row: the residual state, the last row is the draft's feature
    float *emb_dev_ = nullptr;    // n_embd a row: the embeddings (the host's ride in)
    float *e_norm_ = nullptr, *f_norm_ = nullptr;   // n_embd a row: the two norms' outputs
    void *e_q8_ = nullptr, *f_q8_ = nullptr;        // their Q8_1
    void *cat_q8_ = nullptr;      // 2 * n_embd a row: [e_q8 | f_q8] - the q/k/v's input
    float *q_ = nullptr;          // n_head * head_dim a row: the queries, then the gate's output
    float *k_ = nullptr;          // n_kv * head_dim a row: the keys, then the up's output
    float *v_ = nullptr;          // n_kv * head_dim a row: the values
    float *o_ = nullptr;          // n_head * head_dim a row: the attention's output, then the down's
    void* o_q8_ = nullptr;        // the attention output's Q8_1 (the residual norm's fused launch)
    float *normed_ = nullptr;     // n_embd a row: the normed state (the mlp's and the lm_head's input)
    void* normed_q8_ = nullptr;   // its Q8_1
    float *gate_out_ = nullptr, *up_out_ = nullptr;   // n_ff a row: the mlp's gate and up
    float *mlp_h_ = nullptr;      // n_ff a row: the swiglu's output
    float *logits_ = nullptr;     // dvocab a row: the lm_head's output (the draft steps)
    float *taps_q8_ = nullptr;    // 3 * n_embd a row: the taps' Q8_1 (the fc's input)
    float *scratch_ = nullptr;
    size_t scratch_floats_ = 0;
    std::vector<int64_t> d2t_;
    bool loaded_ = false;
};

}  // namespace strata::poly
