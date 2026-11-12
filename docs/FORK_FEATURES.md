# This fork: DeepSeek V4 Flash 0731, accelerated

This is a fork of [PolyStrata](https://github.com/VecSzn/PolyStrata) (itself built on
[Strata](https://github.com/Niko1221/Strata) by Niko1221) that adds a chain of experimental
optimizations for running [DeepSeek V4 Flash 0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)
faster on one NVIDIA card. Everything is in the `base` branch.

The chain is an 8-patch series plus a file-reads / read-heat / attend-batch layer and a small-GEMM
`hc_mix` kernel. On the test workstation it wrote at **247 tokens/s** with an 11k context, cold.
A token is about ¾ of a word.

## The patch chain

Each patch is a self-contained file in the repo root (`*.patch`). They apply on top of each other
in order; the later ones depend on the earlier ones.

1. **q8_0 KV cache** (`deepseek4_q8_0_kv_cache.patch`) — the attention cache for DeepSeek V4 is
   stored as q8_0 (8-bit) instead of fp16. It halves the KV cache's VRAM footprint, leaving the
   expert cache more room on the card.

2. **Fast q8_0 prefill dequant** (`prefill_q8_fast_dequant.patch`) — the packed dequantization of a
   staged key row in the attention kernel: one scale fetch per word and half2 packing cut the
   key-staging loop's instructions about in half. Reading the window's keys is what a prompt's
   attention time is.

3. **16 group heads** (`group_heads_16.patch`) — the attention kernel takes 16 heads per block
   instead of 8. One staged key row serves twice the heads before it is dropped, halving the key
   traffic of a prompt batch's attention. For DeepSeek V4 (64 heads) a token takes 4 blocks instead
   of 8.

4. **Multichunk attend rows** (`multichunk_attend_rows.patch`) — a prompt batch above 65535 rows
   (CUDA's grid.y limit) is attended in launches of that many rows on the host, its tokens' rows of
   q, sel, out and `at` behind the chunk before. Works for q8_0 and fp16 alike.

5. **Indexer scores on tensor cores** (`idx_score_mma.patch`) — the indexer's scores are computed
   with `mma.sync` tensor-core instructions where the card has them (compute capability 8 or
   newer), instead of the scalar kernel. A block scores 16 tokens against 128 compressed rows.

6. **Async overlap for prefill** (`async_overlap_fixed.patch`) — the chunks of a long prompt's read
   overlap: the next chunk's host work and uploads run beside the last layers of the chunk before,
   by an event, instead of the card draining between chunks. Controlled by `--read-overlap`.

7. **Conversation cache, sized from the free RAM** (`conversation_cache_mib_auto.patch`) —
   `--conversation-cache-mib auto` sizes the conversation cache from the RAM free once the model is
   loaded: 40% of what is left above the parking floor, at most 8192 MiB, in steps of 256. A fixed
   8 GiB on a PC whose RAM the model's file already fills pushed that file out of the file cache.

8. **Conversation cache config key** (`conversation_cache_config_key.patch`) — a `conversation_cache`
   key in the server config turns the conversation cache on: `"auto"` or `true` sizes it from the
   free RAM, a number means that many MiB. `--conversation-cache-mib` in the args wins over it.

## The base layer

On top of the patches, the `base` commit adds three engine features, each with a switch:

- **file-reads** (`--file-reads`) — routed experts are read from their file by position
  (`pread` / `ReadFile`) into the pinned buffers, not copied through the mapped file, whose missing
  pages arrive a fault at a time when RAM is tight for the model. Default on.
- **read-heat** (`--read-heat`) — what a prompt routes to warms the expert cache: the staging buffer
  holds what this prompt reads most, and after the read the cache takes in its busiest experts.
  Default on.
- **attend-batch** (`--attend-batch`) — DeepSeek V4's attention takes two tokens a block, their raw
  windows' union staged once. Default on.

And a final commit replaces `hc_pre`'s and `hc_head`'s mix with a small GEMM for batches (a block
of four rows computes all outputs), keeping the old kernel for decode.

## The switches

The chain adds these command-line switches to the engine:

| switch | what it does | default |
| --- | --- | --- |
| `--read-overlap N` | 1: the next chunk of a long prompt's read overlaps the last layers of the chunk before; 0: the card drains between chunks | 1 |
| `--file-reads N` | 1: routed experts read from their file by position into the pinned buffers; 0: copies through the mapping | 1 |
| `--read-heat N` | 1: what a prompt routes to warms the expert cache; 0: reads change nothing | 1 |
| `--attend-batch N` | 1: DeepSeek V4's attention takes two tokens a block; 0: a token a block | 1 |
| `--group-heads N` | the heads an attention block takes: 16 or 8 | 16 |
| `--q8-dequant N` | with `--kv q8_0`: 1 = packed dequant of a staged key row, 0 = a value at a time | 1 |
| `--attend-chunks N` | 1: a batch above 65535 rows is attended in launches of that many; 0: one launch | 1 |
| `--idx-mma N` | 1: the indexer's scores on the tensor cores where the card has them; 0: the scalar kernel | 1 |
| `--ram-experts-mib N` | a RAM pool of this many MiB keeping the most used experts outside VRAM whole in the process's own memory (an LRU of whole experts; the reads of a kept expert stop touching the page cache; the decode swaps read through it too - a miss settles into the pool and the next ask promotes out of RAM at memcpy speed, which holds the model's files only while nothing else wants the RAM). 0 = through the page cache, as before | 0 (off) |
| `--vram-pool` | claim the card's VRAM once at load (all that is free but for the reserve) and hand it to the growing maps from a pool of handles, so the expert cache's and attention caches' stages stop asking the driver each time. Needs VMM, growing attention caches and one card | off |
| `--resident-topk N` | a decode window computes only the N heaviest of each token's routed experts, wherever they live (the VRAM cache's included), the top-1 always kept, the kept weights renormalised the model's way; prefill stays exact. Changes the answers (Maya's Turbo's cap; DSv4 routes 6) | 0 (off) |
| `--prefetch-k N` | read ahead N of each token's likeliest experts of the next MoE layer, guessed on this layer's input while the card takes the next one's attention; the misses go into --ram-experts-mib's pool. Decode and verify windows only (16 tokens at most); needs --ram-experts-mib | 0 (off) |
| `--expert-mma N` | 1: the experts' Q3_K gate/up (and down, when it is Q3_K too) on this fork's own tensor-core kernel: the rows to a plain per-32 int8, the codes in f16 mma fragments; 0: llama.cpp's generic mmq. Needs sm_80+; other types stay on the generic mmq | 0 (off) |
| `--cpu-tiled N` | 1: the CPU experts' gate/up through the tiled K-quant kernels (register tiling, AVX-512/VNNI); 0: the per-row vec_dot path. Tiled covers Q2_K only, other weight types take vec_dot regardless | 1 |
| `--conversation-cache-mib` | `auto`: 40% of the RAM free once loaded, above the floor, at most 8192; a number: that many MiB | 0 (off) |

The q8_0 KV cache is turned on per model with the engine's existing `--kv q8_0` flag, which this
fork extends to DeepSeek V4 (it was qwen35moe-only before).

## Environment switches (the engine reads these from the config's env block)

| env switch | what it does | default |
| --- | --- | --- |
| `STRATA_POLY_STAGE_SPEC=<M>` | the staging's guess-reader: while the card takes layer j's batches, a thread settles layer j+1's whole non-cached expert set into --ram-experts-mib's pool (positioned reads). The next layer's staging then reads its experts out of RAM at memcpy speed instead of faulting them in. On whenever --ram-experts-mib is on; the value caps the candidate count per layer (256 covers a 256-expert layer) | on, 256 |
| `STRATA_POLY_SCORE_CONTEXT=<N>` | the indexer's score stride covers N/4+1 entries instead of max_context/4+1. A 26k-token prompt needs stride 6501, not 33193 — the score tensor's HBM traffic drops proportionally. Set to the expected prompt's token count | max_context |
| `STRATA_POLY_SCORE_MIB=<N>` | the indexer's score buffer budget in MiB: a bigger budget fits more tokens per launch, fewer launches, a steadier read | 128 |
| `STRATA_POLY_STAGE2_DEBUG=1` | prints each staging round's copy time and each layer's route/card/copies split to stderr | off |

The staging reads by position (pread) instead of memcpy through the mapping: a steady channel the
page cache's state does not bend. The mapper's whole-file warmer stays off when the staging's
guess-reader runs: the guess-reader's positioned reads warm the page cache exactly where the read
routes to, beside settling the pool — a whole-file pass beside it only fights it for the disk.

The indexer's score and select are pipelined across two streams when a prompt batch has more than
one score batch: score(batch k+1) on the indexer's stream runs beside select(batch k) on the main
stream — the score kernel is memory-bound, the select is compute-light, so they share the GPU.
Each select waits its batch's score via an event; the attend below waits all selects on the main
stream. Single-batch calls keep the old one-stream path.

The idx scorer's mma runs at the decode windows' shapes only (n ≤ 16 tokens, +17% decode): the
prompt batches' 6500-token calls take the scalar dp4a scorer they win with (−8% of the read).

## Verdicts of the live checks (DeepSeek V4 Flash 0731, one RTX 4080 SUPER, November 2026)

- **`--ram-experts-mib 6144` — kept.** Quality-neutral (a ten-prompt battery at temperature 0
  answers identically to the plain run; the pool's self-check passes 387/0, no torn slots), the
  decode sweep puts 6144 ahead of 2048, the long talks hit rate at 64.6 percent.
- **`--vram-pool` — kept.** Clean, the whole new configuration together runs at 350 to 640
  prefill tokens a second at 25k context and 27.8 decode tokens a second — the best of the
  readings so far.
- **`--expert-mma` — rejected.** The first kernel loses to the generic mmq at prefill (180 to
  280 tokens a second against 350 to 640 with it off) and its rounding on CUDA's stream-k
  changes the answers over a long generation (echo loops, a switch of language, lost context).
  The kernel stays in the tree as research, with its parity tests; a live return is only
  thinkable alongside a bit-exact mode.
- **`--resident-topk` — rejected on DSv4.** The router's weights are flat (the top-3 hold only
  65 to 67 percent of the token's expert weight), the cap dulls the model. The cells are out of
  the sweep.
- **STRATA_POLY_STAGED_TILES (env)** — off for live use: the measured decode gain of 9.3
  percent exists, but the same stream-k rounding class as --expert-mma.
- **`--prefetch-k`** — not ported: the upstream series rides on foreign refactors; the native
  `--ram-experts-mib` pool took over its part.

## How to use it

Build the engine on your PC, then run the server with the switches you want, e.g.:

```text
strata-poly --model deepseek-v4-flash-0731.gguf --kv q8_0 --read-overlap 1 --attend-batch 1
```

To compare versions, apply and revert each patch with `git apply` / `git apply -R`. The order of
the later kernels does not matter; the q8_0 and prefill patches must stay first.

## Version

`v0.1.45-poly.2`, on top of PolyStrata's 0.1.44/0.1.45 engine.

## Upstream

[PolyStrata](https://github.com/VecSzn/PolyStrata) · [Strata](https://github.com/Niko1221/Strata)
