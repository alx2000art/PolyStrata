# DSpark port: the plan (feat/dspark)

The DSpark drafter (DeepSeek's speculative decoding for DeepSeek-V4-Flash, superior to naive
MTP; llama.cpp PR 25784 is the reference) ported to this fork's engine. The goal: the decode's
1.5-1.9x on the cards where the windows pay. The main model config carries the hook points:
`dspark_target_layer_ids: [40, 41, 42]`, `dspark_markov_rank: 256`.

## The algorithm (from PR 25784)

1. The target model exports **taps**: hidden states at layers 40, 41, 42, each collapsed over the
   hyper-connection multiplier dimension (`dsv4_hc_mean`, our 4 residual streams) to n_embd-wide
   vectors.
2. The three taps stack to `n_embd_enc = 3 * n_embd` (12288) and go through the drafter's
   `fc.weight` (12288, 4096 - it is in the drafter's GGUF) to a n_embd input.
3. The **drafter is a whole V4-arch MoE model** (~11 GB Q8_0; "the same model structure as
   DeepSeek-V4-Flash-DSpark") - not one block like the MTP head. It runs forward from the fc input
   and proposes up to `n_max` tokens; its own taps feed its own recursion (EAGLE3-style: the
   target's taps for the first step, the drafter's own for the rest).
4. **Markov and confidence heads** grade the proposals; `p_min` prunes unsure ones (llama.cpp
   users report 56.5% -> 69.8% acceptance from the gating alone).
5. Acceptance decays per position (measured: 0.767, 0.488, 0.264, 0.137, 0.051) - the sweet spot
   was `n_max` 1 to 3, not 5, on the cards where the verify is not free.

## What our engine has already

- `Engine::draft(token, i, prob)` (engine.cpp:4491): the per-token draft interface with the
  probability - main.cpp's guess loop consumes it (`engine.draft(from, n_d, &p)`), the windows,
  the policy and the verify all exist and are graph-replayed.
- The MTP block draft (`draft_step`, `block_`/`dg_`): one block reading the main model's state -
  its shape is the pattern; the DSpark drafter replaces its internals.
- The Engine class runs any V4-arch GGUF: the drafter is a **second Engine instance**.
- The row-input path (`images`/`input()`): a provided embedding per row - the fc output can ride
  it instead of a token's embedding.

## What the port adds

1. **The taps export in the main engine**: during decode, capture the post-layer hidden state at
   the layers 40/41/42 (the indices from the metadata), collapse with `hc_mean`, keep the last
   row's three taps on the host or device (48 KB a token - nothing).
2. **The drafter as a second Engine** (`--draft-model` reused): its options - the model file, the
   context (the draft depth + the prompt tail it replays), no expert pool (its experts go through
   the mapping/page cache), its own KV.
3. **The fc + input mode**: the drafter's fc consumes the stacked taps; its output rides the
   row-input path. Which tensors the drafter file carries (fc, its own embed, the heads) decides
   the exact wiring - **inspect the file first**.
4. **The KV sync**: after a window's verify accepts k tokens, the drafter replays them (its own
   eval at the aligned positions); the rejected tail's rows are dropped the way the main engine
   drops its own. The llama.cpp PR's trickiest part (`state_write`/`state_read`, the recurrent
   rollback) - expect this to be where the debugging goes.
5. **The proposal loop behind `Engine::draft`**: stage 0 = fc(target taps) -> the drafter's
   forward -> its likeliest + confidence; the recursion from its own taps; the Markov head's
   re-rank if the file carries it.
6. The knobs: `--spec-draft-n-max` (as `--draft-n-max`), `p_min`, the taps' layer indices from the
   metadata with a CLI override.

## The phases

- **0. The file inspection** (blocks the design's details): download
  `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` (10.9 GB, unsloth), dump its metadata and tensor
  names: the layer count, the fc/embed/head tensors' names and shapes, the tap indices in its own
  metadata, the quant mix (what is Q8_0, what MXFP4).
- **1. The taps export** in the main engine (self-contained, dumpable, testable by eye).
- **2. The drafter Engine skeleton**: the second instance loads, runs a token forward, the logits
  match llama.cpp's run of the same file on the same prompt (the parity check against
  `llama-cli`).
- **3. The draft loop** behind `draft()`: the single-token draft first (n_max=1), the acceptance
  measured against the no-draft baseline.
- **4. The KV sync + the multi-token recursion** (n_max up to 3), the acceptance per position.
- **5. The knobs and the tuning**: p_min, n_max, the taps' layers.

## The risks

- The VRAM: the drafter's dense part (its file's non-expert tensors) resident - expect 1.5-2.5 GB;
  the main model's expert cache shrinks by that (5.2 GiB -> ~3-3.7 GiB). The ledger against the
  acceptance is the whole point of phase 3.
- The IO: the drafter's MoE experts read through the page cache - one small model's worth, hot;
  but two models' expert reads now share the disk's channel with the main decode's.
- The KV sync's edge cases: the cancelled requests mid-window, the conversation cache interplay,
  the prompt-cache restores (the drafter's state must follow).
- The answer drift: like llama.cpp's, the guesses change the sampled path only through accepted
  tokens (the verify is exact) - the drift class of the existing guesses, not a new one.


## Phase 0 done: the drafter's anatomy (from the header, 2026-11-13)

The drafter GGUF (unsloth's `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf`, 10.9 GB) is the
**`dflash`** architecture - not deepseek4, the Engine class cannot load it:

- **3 blocks** of the DSv4 block structure (the same tensor names: attn_q_a/q_b, kv, hc_*,
  MoE with 256 experts, 6 used, expert_ffn 2048, exp_probs_b, the indexer in the metadata) -
  but `compress_ratios [0,0,0]`: no compressed rows, a plain sliding-window attention of 128 -
  far simpler than the main model's decode;
- the **enc**: `fc.weight` (3*n_embd -> n_embd, the PR's number confirmed) +
  `enc.output_norm.weight` - the stacked taps go through it into the blocks;
- the **heads**: `output_hc_base/fn/scale.weight` + `output_norm.weight` (the logits),
  `conf_proj.weight` (the confidence), `markov_w1/w2.weight` (the Markov re-rank);
- **no embed_tokens / output.weight of its own** - the vocabulary is shared with the main model
  (what the draft_ gate wants);
- `dflash.target_layers: [41, 42, 43]` (1-based) = the config's [40, 41, 42] - the main model's
  last three layers;
- the quant mix: the experts MXFP4 (9 tensors), the attention projections Q3_K_XS (6), the rest
  Q8_1/F32; no smaller quant exists (Q8_0 and BF16 only) and none is needed: the disk is the
  experts (already MXFP4), the VRAM-resident dense part is ~1.5-2 GB.

**The revised shape**: not a second Engine - a `DFlash` drafter class, a stripped DeepSeek4
twin: the 3 blocks reuse the kernels (the same tensor structure), no comp rows, no hash layers,
plus the enc and the heads; the drafter's KV is a simple ring of 128. The VRAM ledger improved:
the main model's expert cache gives up ~1.5-2 GB, not 2.5-3.


## The engine pattern found: Glm5Next's draft virtuals

The Engine base already virtualizes the draft block: `check_draft_block()`, `block_attend(pos, n)`,
`block_rows(pos, n)` - Glm5Next (the glm5-next arch, 45 main layers + the draft block as the 46th in
the same file) implements them, and `hc_mean` (kernels.hpp:48, the HC streams' mean per row - the
taps primitive) already serves its output path.  The DSpark difference: the drafter is a **separate
dflash file**, not an extra block in the main arch's naming - so the nextn-block approach (the
engine reading `blk.43.nextn.*` as its own layer) cannot consume it.

**The revised architecture**: a `DFlash` drafter class - not an Engine subclass, a small runner
holding: the dflash file (Shards), the 3 blocks' weights, the enc (fc + norm), the heads, its own KV
rings (3 layers, window 128, no compressed rows).  DeepSeek4's draft path constructs it when
`--draft-model`'s file is dflash (its `general.architecture` says so), and the draft sequence runs:
the taps (hc_mean of `r_` at the layers 40/41/42, per row) -> fc -> the 3 blocks -> the heads ->
(the token, the confidence).  The verify windows, the policy and `Engine::draft`'s interface stay as
they are.

The tap capture rides the layer loop: after layer_tail at the tap layers, `hc_mean(r_, tap, n_embd_,
rows, st_)` - inside the replayed lambda so the graphs bake it in; the taps buffer is 3 x rows x
n_embd, per-row, rewritten every call.


## The better drafter found: EAGLE-3 for DeepSeek-V4-Flash-0731 (2026-11-14)

**AQ-MedAI/DeepSeek-V4-Flash-0731-eagle3**: a 0.7B Llama-arch EAGLE-3 head trained on our exact
checkpoint (safetensors BF16, MIT; SGLang-only instructions, no GGUF).  The acceptance lengths:
**2.62-3.20 tokens a round** (HumanEval 3.16, Mtbench 2.86) - far above the MTP head's 0.74 and
the DSpark's ~1.7.  The mass profile is the dream for a 16 GB card: dense, no MoE, the whole
drafter ~1.4 GB BF16 - **resident in VRAM entirely**.

**The anatomy** (the safetensors header + config.json, read over HF Range):
- `fc.weight` [4096, 12288] - the SAME 3 x n_embd taps shape as the dspark's; the config says
  `eagle_aux_hidden_state_layer_ids: [1, 20, 39]` - the low/mid/high layers (EAGLE-3's spread,
  not the dspark's last-three);
- ONE `midlayer` - a standard Llama block: GQA attention (64 heads, 8 kv, head_dim 64), SwiGLU
  18432; the input is 8192-wide = [the target's embedding of the token, fc(the three taps)] - the
  classic EAGLE-3 concat;
- `lm_head.weight` [96000, 4096] - the drafter's own head over a REDUCED vocabulary (96000 of
  129280), with `d2t` (the draft-to-target ids) and `t2d` (the target-to-draft mask) mappings -
  the rare tokens are never proposed;
- no embed_tokens - the token embedding comes from the main model's token_embd (the shared-embed
  pattern our draft_ gate wants); `hidden_norm` for the features; the rope carries the main
  model's yarn (factor 16, original 65536).

**The flow**: the target's taps (hc_mean at layers 1/20/39 - the phase-1 capture with these
indices) -> fc -> concat with the token's embedding -> the midlayer -> the draft; the recursion
feeds the midlayer's own 4096 output as the next step's feature; the confidence = the sampled
token's probability (no separate head needed).  The drafter keeps its own KV over the accepted
tokens (a 0.7B layer - its replay is trivial); accept() is NOT a noop here, unlike DSpark.

**The port plan (revised - the DFlash sections above stay as the dspark notes)**:
- 0. the conversion: the safetensors -> GGUF (tools/gguf_writer.py; the tensors keep their HF
  names; the KVs carry the draft vocab, the tap layers, the rope, d2t/t2d as arrays);
- 1. the taps capture generalized: the indices from the draft file's KVs instead of the main
  model's metadata (the phase-1 code, one change);
- 2. the Eagle3 drafter class: the file load, the ONE Llama layer forward (the fork's window
  attention kernel reused for the drafter's own KV - the draft heads are forgiving of a window),
  the fc, the lm_head, the d2t translation;
- 3. the draft loop: the iterative n_max steps behind Engine::draft, the drafter's KV replay per
  verified window (0.7B - cheap);
- 4. the acceptance on the real workload - the convention check (hc_mean vs another collapse)
  shows itself instantly: the acceptance ~2.9 says the taps match SGLang's export, ~0.5 says no.

**The honest economics** (the same ledger as the dspark's): the window of 4-5 tokens costs
~4.5 x the per-token expert IO (the union amortizes to ~4x) + ~1.5x the dense; the round yields
~3.9 tokens - the ledger sits at ~1.0-1.1x on paper and lives or dies on the REAL acceptance of
the head against our workload and our tap convention.  The MTP head (one block, 0.74) stays the
cheap calibration experiment that measures the window overhead before this port's phases 2-3
run.
