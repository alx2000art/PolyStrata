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
| `--conversation-cache-mib` | `auto`: 40% of the RAM free once loaded, above the floor, at most 8192; a number: that many MiB | 0 (off) |

The q8_0 KV cache is turned on per model with the engine's existing `--kv q8_0` flag, which this
fork extends to DeepSeek V4 (it was qwen35moe-only before).

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
