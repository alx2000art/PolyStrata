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
