#!/usr/bin/env python3
"""A/B harness for the PolyStrata engine patches (ab_strata.py's shape, turned on the patch switches).

The patches of this repo each have an engine key now (patch_switches.patch): --read-overlap,
--file-reads, --read-heat, --attend-batch, --group-heads, --q8-dequant, --attend-chunks, --idx-mma.
An engine restart applies them, so the workflow of `sweep` is:

  1) one capture with everything on (the plain config, the defaults)   -> all-on.json
  2) for each patch: the config with that one key turned off, a server
     restart, a capture                                               -> <patch>.off.json
  3) a summary: turning patch X off cost -Y% decode / -Z% prefill, next
     to the within-config noise floor (the same judge ab_strata.compare uses)

Each capture runs the prompt suite of ab_strata (short tool-result-like prompts, medium and long
prompts, greedy decoding) with --repeat passes.  Repeats that differ are a diagnostic, not a
failure: the engine drifts run to run (drafts, MoE, the CPU-share path).  The suite's long prompts
are what most of the patches touch - prefill - so compare carries a prefill column beside decode.

Two ways to restart the server (both reach the same captures):

  local    python3 ab_patches.py sweep --config my-config.json --out-dir bench
           The script writes config.<variant>.json beside the captures and starts
           `serve/server.py --config <it> --port <port>` itself, waiting for /health.
           The server ends its engine on the way out, so each variant starts clean.

  manual   python3 ab_patches.py sweep --config my-config.json --out-dir bench --manual
           For a server elsewhere: the script prints each variant's engine args and
           waits for /health while you restart yours with them (ab_strata's workflow).

`variants` only writes the configs and prints the args, nothing runs - to eyeball or to drive a
remote box by hand.  Only the standard library is used.

Usage:
  python3 ab_patches.py capture --out base.json --base-url http://127.0.0.1:8095/v1 --model my-model
  python3 ab_patches.py compare all-on.json read_heat_cache.off.json
  python3 ab_patches.py variants --config my-config.json --out-dir bench
  python3 ab_patches.py sweep --config my-config.json --out-dir bench --suite poly --repeat 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# --------------------------------------------------------------------------- the patches

# (name for the capture files and the summary, the key as the engine spells it, the arguments that
# turn the patch off, what a run with it off no longer has).  --kv q8_0 is not here: it picks the
# cache's format (and its answers), not a speed path; A/B it with the same tool by hand.
PATCHES: list[tuple[str, str, list[str], str]] = [
    ("async_overlap", "--read-overlap", ["--read-overlap", "0"],
     "the chunks of a long prompt's read no longer overlap"),
    ("expert_direct_reads", "--file-reads", ["--file-reads", "0"],
     "experts are copied through the mapped file, not read by position"),
    ("read_heat_cache", "--read-heat", ["--read-heat", "0"],
     "a read's routing warms neither the staging order nor the cache"),
    ("attend_batch_pairs", "--attend-batch", ["--attend-batch", "0"],
     "attention takes a token a block, not a pair"),
    ("group_heads_16", "--group-heads", ["--group-heads", "8"],
     "an attention block takes 8 heads, not 16"),
    ("prefill_q8_fast_dequant", "--q8-dequant", ["--q8-dequant", "0"],
     "with --kv q8_0: a staged key row dequantizes a value at a time"),
    ("multichunk_attend_rows", "--attend-chunks", ["--attend-chunks", "0"],
     "a batch above 65535 rows is one launch again (a fix; no effect below that)"),
    ("idx_score_mma", "--idx-mma", ["--idx-mma", "0"],
     "the indexer's scores take the scalar kernel"),
]

# Opt-in engine options the run config does not carry (their absence is the baseline): each variant ADDS the
# pair, so a positive number in the summary means turning it on sped that up - the patches above are the other
# way round (a negative number means turning them off slowed things down).
OPTINS: list[tuple[str, list[str], str]] = [
    ("lookup", ["--lookup", "8"],
     "guess from earlier repeats (the suffix drafter): decode-only, pays where an answer quotes its prompt"),
]

# --------------------------------------------------------------------------- suite (ab_strata's)

def _filler(words: int, kind: str) -> str:
    """Deterministic pseudo-text of roughly `words` words (stable token counts)."""
    if kind == "code":
        base = (
            "def compute_step(state, index, gain=0.125):",
            "    total = 0.0",
            "    for i, value in enumerate(state):",
            "        total += value * gain * (index % 7 + 1)",
            "        if total > 1024.0:",
            "            total -= 512.0",
            "    return total",
            "",
        )
    else:
        base = (
            "The survey station logged another quiet interval before the "
            "afternoon shift began recalibrating the eastern sensors.",
            "Wind readings stayed within the expected band while the crew "
            "compared them against last week's archived measurements.",
            "A brief note in the maintenance log mentioned a loose panel on "
            "the secondary mast, already scheduled for the next inspection.",
            "Meanwhile the database synchronized its hourly snapshot and the "
            "operators reviewed the anomaly flags queued since midnight.",
        )
    out: list[str] = []
    i = 0
    while len(out) < words:
        out.extend(base[i % len(base)].split())
        i += 1
    return " ".join(out[:words])


def build_suite(suite: str) -> list[dict[str, Any]]:
    # (id, prompt words, kind, max_tokens, what it probes)
    # (id, prompt words, kind, max_tokens, what it probes); max_tokens leaves
    # room for the thinking budget (~1024) plus the actual answer
    spec = [
        ("tool-350", 350, "text", 1536, "CPU-share path (<1024-token chunk)"),
        ("chat-512", 520, "text", 1536, "CPU-share path, upper edge"),
        ("story-2k", 1500, "text", 2048, "decode-heavy, experts mostly streamed"),
        ("code-2k", 1400, "code", 2048, "code decode (draft acceptance path)"),
    ]
    if suite in ("long", "poly"):
        spec += [
            ("doc-8k", 6000, "text", 2048, "long prompt, stager/page-cache path"),
            ("doc-16k", 12000, "text", 2048, "long prompt, file tier if model > RAM"),
        ]
    if suite == "poly":
        spec += [
            ("doc-30k", 23000, "text", 2048, "very long prompt: read overlap, attention batching"),
        ]
    prompts = []
    for pid, words, kind, max_tokens, probes in spec:
        body = _filler(words, kind)
        prompts.append(
            {
                "id": pid,
                "words": words,
                "max_tokens": max_tokens,
                "probes": probes,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Read the text below and continue it with at least "
                            "three paragraphs (300+ words). Stay in the same "
                            "style, do not summarize.\n\n" + body
                        ),
                    }
                ],
            }
        )
    return prompts


# --------------------------------------------------------------------------- capture (ab_strata's)

def _post_streaming(base_url: str, api_key: str, model: str, entry: dict[str, Any],
                    temperature: float, timeout: float, seed: int,
                    thinking_off: bool) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": entry["messages"],
        "max_tokens": entry["max_tokens"],
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed >= 0:
        body["seed"] = seed
    if thinking_off:
        # Several spellings; a server that understands none of them keeps
        # thinking and the run reports thinking_chars > 0.
        body["chat_template_kwargs"] = {"enable_thinking": False}
        body["reasoning_effort"] = "none"
        body["thinking"] = {"type": "disabled"}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    first_token_at: float | None = None
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish = ""
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or ""
            if piece:
                content_parts.append(piece)
            if reasoning:
                thinking_parts.append(reasoning)
            if (piece or reasoning) and first_token_at is None:
                first_token_at = time.perf_counter()
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    ended = time.perf_counter()
    content = "".join(content_parts)
    thinking = "".join(thinking_parts)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    if completion_tokens == 0:
        completion_tokens = max(1, (len(content) + len(thinking)) // 4)
    ttft = (first_token_at - started) if first_token_at else (ended - started)
    decode_s = max(ended - (first_token_at or started), 1e-6)
    decode_tps = (completion_tokens - 1) / decode_s if completion_tokens > 1 else 0.0
    prefill_tps = prompt_tokens / max(ttft, 1e-6) if prompt_tokens else 0.0
    return {
        "text": content,
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "thinking_chars": len(thinking),
        "thinking_sha256": hashlib.sha256(thinking.encode()).hexdigest(),
        "chars": len(content),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "ttft_s": round(ttft, 3),
        "total_s": round(ended - started, 3),
        "decode_tps": round(decode_tps, 2),
        "prefill_tps": round(prefill_tps, 1),
        "finish_reason": finish,
    }


def run_suite(base_url: str, api_key: str, model: str, suite: str, repeat: int, warmup: int,
              temperature: float, seed: int, thinking_off: bool, timeout: float) -> list[dict[str, Any]]:
    """The suite of `capture` as one function, so sweep can run it under each config."""
    for warm in range(warmup):
        print(f"  warmup {warm + 1}/{warmup} ... ", end="", flush=True)
        try:
            run = _post_streaming(
                base_url, api_key, model,
                {"messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 64},
                temperature, timeout, seed, thinking_off,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise SystemExit(f"  FAILED: {error}") from error
        print(f"{run['total_s']:.2f}s")
    results: list[dict[str, Any]] = []
    for entry in build_suite(suite):
        for rep in range(repeat):
            label = f"{entry['id']}#{rep}"
            print(f"  {label} ... ", end="", flush=True)
            try:
                run = _post_streaming(base_url, api_key, model, entry,
                                      temperature, timeout, seed, thinking_off)
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                raise SystemExit(f"  FAILED: {error}") from error
            run.update({"id": entry["id"], "repeat": rep, "probes": entry["probes"]})
            results.append(run)
            print(
                f"prefill {run['prefill_tps']:.0f} tok/s | decode {run['decode_tps']:.1f} tok/s | "
                f"TTFT {run['ttft_s']:.2f}s | {run['completion_tokens']} tok | "
                f"think {run['thinking_chars']} ch | {run['sha256'][:12]}"
            )
    return results


def _capture_doc(base_url: str, api_key: str, model: str, args: argparse.Namespace,
                 results: list[dict[str, Any]]) -> dict[str, Any]:
    doc = {
        "meta": {
            "base_url": base_url,
            "model": model,
            "suite": args.suite,
            "repeat": args.repeat,
            "temperature": args.temperature,
            "seed": args.seed,
            "thinking_off": args.no_thinking,
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "runs": results,
    }
    if results and any(r["thinking_chars"] > 0 for r in results) and args.no_thinking:
        print("note: thinking was still produced - the server ignored the disable request")
    # intra-config sanity: this engine shows run-to-run content drift even at
    # temperature 0 (MTP drafts, MoE atomics, the CPU-share prefill path), so
    # identity across repeats is a diagnostic, not a requirement
    by_id: dict[str, list[dict[str, Any]]] = {}
    for run in results:
        by_id.setdefault(run["id"], []).append(run)
    for pid, runs in by_id.items():
        if len({run["sha256"] for run in runs}) > 1:
            print(f"note: {pid} answers differ between repeats (known engine drift) - "
                  f"judge by speed only")
    return doc


def cmd_capture(args: argparse.Namespace) -> int:
    results = run_suite(args.base_url, args.api_key, args.model, args.suite, args.repeat,
                        args.warmup, args.temperature, args.seed, args.no_thinking, args.timeout)
    doc = _capture_doc(args.base_url, args.api_key, args.model, args, results)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)
    print(f"saved {len(results)} runs -> {args.out}")
    return 0


# --------------------------------------------------------------------------- compare (ab_strata's, plus prefill)

def _first_divergence(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit  # one is a prefix of the other


def _geo(values: list[float]) -> float:
    if not values:
        return 0.0
    out = 1.0
    for value in values:
        out *= value
    return out ** (1.0 / len(values))


def _noise(runs: dict[tuple[str, int], dict[str, Any]], field: str) -> float:
    """Per-id spread of `field` across repeats, as a share of the mean."""
    per_id: dict[str, list[float]] = {}
    for (_pid, _rep), run in runs.items():
        if run.get(field, 0) > 0:
            per_id.setdefault(_pid, []).append(run[field])
    spreads = []
    for values in per_id.values():
        if len(values) >= 2:
            mean = sum(values) / len(values)
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            spreads.append((var ** 0.5) / mean if mean else 0.0)
    return (sum(spreads) / len(spreads) * 100.0) if spreads else 0.0


def compare_pair(path_a: str, path_b: str, verbose: bool = True) -> dict[str, Any]:
    """The heart of ab_strata.compare, reusable by sweep: decode, prefill and TTFT deltas."""
    with open(path_a, encoding="utf-8") as handle:
        base = json.load(handle)
    with open(path_b, encoding="utf-8") as handle:
        variant = json.load(handle)
    runs_a = {(r["id"], r["repeat"]): r for r in base["runs"]}
    runs_b = {(r["id"], r["repeat"]): r for r in variant["runs"]}
    keys = sorted(set(runs_a) | set(runs_b))
    rows: list[dict[str, Any]] = []
    identical = changed = missing = 0
    decode_r: list[float] = []
    prefill_r: list[float] = []
    for key in keys:
        ra, rb = runs_a.get(key), runs_b.get(key)
        if ra is None or rb is None:
            missing += 1
            continue
        same = ra["sha256"] == rb["sha256"]
        identical += same
        changed += not same
        note = "" if same else f"first diff at char {_first_divergence(ra['text'], rb['text'])}"
        for field, store in (("decode_tps", decode_r), ("prefill_tps", prefill_r)):
            va, vb = ra[field], rb[field]
            if va > 0 and vb > 0:
                store.append(vb / va)
        rows.append({"id": f"{key[0]}#{key[1]}", "same": same,
                     "decode_a": ra["decode_tps"], "decode_b": rb["decode_tps"],
                     "prefill_a": ra["prefill_tps"], "prefill_b": rb["prefill_tps"],
                     "ttft_a": ra["ttft_s"], "ttft_b": rb["ttft_s"], "note": note})
        if verbose:
            d = ((rb["decode_tps"] / ra["decode_tps"]) - 1.0) * 100.0 if ra["decode_tps"] > 0 else 0.0
            p = ((rb["prefill_tps"] / ra["prefill_tps"]) - 1.0) * 100.0 if ra["prefill_tps"] > 0 else 0.0
            print(f"  {key[0]}#{key[1]:<12} decode {ra['decode_tps']:>6.1f}->{rb['decode_tps']:<6.1f} "
                  f"{d:+6.1f}%   prefill {ra['prefill_tps']:>7.0f}->{rb['prefill_tps']:<7.0f} {p:+6.1f}%   "
                  f"TTFT {ra['ttft_s']:.2f}s->{rb['ttft_s']:.2f}s  {note}")
    noise_a = max(_noise(runs_a, "decode_tps"), _noise(runs_a, "prefill_tps"))
    noise_b = max(_noise(runs_b, "decode_tps"), _noise(runs_b, "prefill_tps"))
    out = {
        "meta_a": base["meta"], "meta_b": variant["meta"],
        "decode_delta_pct": round((_geo(decode_r) - 1.0) * 100.0, 2) if decode_r else 0.0,
        "prefill_delta_pct": round((_geo(prefill_r) - 1.0) * 100.0, 2) if prefill_r else 0.0,
        "pairs": len(decode_r), "noise_pct": round(max(noise_a, noise_b), 2),
        "identical": identical, "changed": changed, "missing": missing,
        "rows": rows,
    }
    if verbose:
        print(f"  decode {_geo(decode_r) - 1.0:+.1%} | prefill {_geo(prefill_r) - 1.0:+.1%} "
              f"({len(decode_r)} pairs; noise +/-{out['noise_pct']:.1f}%; "
              f"outputs {identical} same / {changed} diff)")
    return out


def cmd_compare(args: argparse.Namespace) -> int:
    print(f"A = {args.a}")
    print(f"B = {args.b}")
    for field in ("suite", "repeat", "temperature", "seed", "model"):
        va, vb = None, None
        try:
            with open(args.a, encoding="utf-8") as handle:
                va = json.load(handle)["meta"].get(field)
            with open(args.b, encoding="utf-8") as handle:
                vb = json.load(handle)["meta"].get(field)
        except (OSError, ValueError, KeyError):
            pass
        if va != vb:
            print(f"META MISMATCH: {field}: {va!r} (A) vs {vb!r} (B) - not a valid A/B")
    out = compare_pair(args.a, args.b)
    if args.json:
        print(json.dumps(out, indent=1))
    return 0


# --------------------------------------------------------------------------- configs and server

def load_config(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        cfg = json.load(handle)
    for field in ("exe", "args"):
        if field not in cfg:
            raise SystemExit(f"{path}: no '{field}' - a config holds what serve/server.py takes "
                             f"(exe, args, cwd, tokenizer, model_name)")
    cfg.setdefault("cwd", None)
    cfg.setdefault("tokenizer", None)
    cfg.setdefault("model_name", None)
    return cfg


def variant_args(args: list[str], off: list[str] | None) -> list[str]:
    """The engine's args with the one key of `off` turned to its off value (None: the plain args).

    A key the args do not carry is at its default (on); turning it off adds it.  A key they carry
    is replaced, whatever the caller had it at."""
    if off is None:
        return list(args)
    key = off[0]
    out = list(args)
    at = out.index(key) if key in out else -1
    if at >= 0:
        del out[at:at + 2]   # the key and its value
    return out + off


def write_variant(out_dir: str, cfg: dict[str, Any], name: str, args: list[str]) -> str:
    path = os.path.join(out_dir, f"config.{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({**cfg, "args": args}, handle, ensure_ascii=False, indent=1)
    return path


def apply_variant(path: str, cfg: dict[str, Any], args: list[str]) -> str:
    """Write the variant's args into the live config `path`, keeping every other field of it (the
    server's port, sampling, api key, ...).  The first call of a sweep keeps a copy beside it and
    no later call overwrites that copy, so it always holds what was there before the sweep; restore
    it and remove it when the sweep is done.  Returns the copy's path."""
    live = dict(cfg)
    import shutil
    backup = path + ".bak-sweep"
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            try:
                live = json.load(handle)
            except (OSError, ValueError):
                pass
        if not os.path.isfile(backup):   # the first apply keeps it; the rest leave it alone
            shutil.copy2(path, backup)
    else:
        backup = "(none: the file was not there)"
    live["args"] = list(args)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(live, handle, ensure_ascii=False, indent=1)
    return backup


def _health(base_url: str, api_key: str) -> tuple[bool, bool]:
    """(the server answered, its engine is loaded) from /health; unreachable is (False, False)."""
    health = base_url.rstrip("/").rsplit("/v1", 1)[0] + "/health"
    try:
        request = urllib.request.Request(health, headers={"Authorization": f"Bearer {api_key}"})
        with urllib.request.urlopen(request, timeout=5) as response:
            if response.status != 200:
                return False, False
            body = json.loads(response.read().decode("utf-8", "replace"))
            return True, bool(body.get("loaded"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return False, False


def wait_restart(base_url: str, api_key: str, timeout_s: float) -> bool:
    """Wait a restart out, with nothing asked of whoever runs this: first the old server going away
    (the port closing, or its engine unloading - /health says which), then the new one up with its
    engine loaded.  A server that is already away skips the first wait.  False on a timeout."""
    reachable, loaded = _health(base_url, api_key)
    if reachable and loaded:
        print("  waiting for the restart: the server's port to close or its engine to unload ...",
              flush=True)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            reachable, loaded = _health(base_url, api_key)
            if not reachable or not loaded:
                break
            time.sleep(1.0)
        else:
            return False
    print("  waiting for the engine to load ...", flush=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        reachable, loaded = _health(base_url, api_key)
        if reachable and loaded:
            return True
        time.sleep(2.0)
    return False


def start_server(root: str, config_path: str, port: int, api_key: str,
                 log_path: str) -> subprocess.Popen:
    command = [sys.executable, os.path.join("serve", "server.py"),
               "--config", config_path, "--port", str(port), "--host", "127.0.0.1"]
    if api_key:
        command += ["--api-key", api_key]
    log = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=log,
                            start_new_session=(os.name != "nt"))
    return proc


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        proc.terminate()
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)   # the server ends its engine on the way out
    try:
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        proc.kill()


# --------------------------------------------------------------------------- sweep

def cmd_variants(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"base engine args:\n  {' '.join(cfg['args'])}\n")
    print("all-on (the plain config):")
    print(f"  {write_variant(args.out_dir, cfg, 'all-on', variant_args(cfg['args'], None))}")
    for name, _key, off, why in PATCHES:
        path = write_variant(args.out_dir, cfg, f"{name}.off", variant_args(cfg["args"], off))
        print(f"{name}.off ({why}):\n  {path}\n  engine args: {' '.join(variant_args(cfg['args'], off))}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    root = os.path.abspath(args.root)
    cfg = load_config(args.config)
    os.makedirs(args.out_dir, exist_ok=True)
    only = set(args.only.split(",")) if args.only else ({name for name, _k, _o, _w in PATCHES} |
                                                        {name for name, _o, _w in OPTINS})
    for skip in (args.skip.split(",") if args.skip else []):
        only.discard(skip)
    # (label, the args pair of the variant): patches turn a key off, opt-ins add one the config does not carry
    variants: list[tuple[str, list[str] | None]] = [("all-on", None)]
    variants += [(f"{name}.off", off) for name, _key, off, _why in PATCHES if name in only]
    variants += [(f"{name}.on", on) for name, on, _why in OPTINS if name in only]
    if args.with_all_off:
        # every key off at once: what the whole patch series is worth together
        args_list = cfg["args"]
        for _name, _key, off, _why in PATCHES:
            args_list = variant_args(args_list, off)
        variants.append(("all-off", args_list))
    captures: dict[str, str] = {}
    for name, off in variants:
        label = name
        path = os.path.join(args.out_dir, f"{label}.json")
        print(f"\n=== {label} " + "=" * max(0, 60 - len(label)))
        if not args.force and os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    if json.load(handle).get("runs"):
                        print(f"kept {path} (captured before; --force redoes it)")
                        captures[label] = path
                        continue
            except (OSError, ValueError):
                pass
        cfg_path = write_variant(args.out_dir, cfg, label, variant_args(cfg["args"], off))
        engine_args = variant_args(cfg["args"], off)
        if args.manual:
            print(f"engine args (config written to {cfg_path}):")
            print(f"  {' '.join(engine_args)}")
            if args.apply_config:
                backup = apply_variant(args.apply_config, cfg, engine_args)
                print(f"  written into {args.apply_config} (the sweep's first copy of it: {backup})")
            print("restart your server now - the sweep goes on by itself when the engine has loaded "
                  f"(each wait up to {args.boot_timeout:.0f}s)")
        else:
            print(f"starting serve/server.py on port {args.port} (log: {args.out_dir}/server.log) ...")
            proc = start_server(root, cfg_path, args.port, args.api_key,
                                os.path.join(args.out_dir, "server.log"))
        base_url = f"http://127.0.0.1:{args.port}/v1" if not args.manual else args.base_url
        try:
            if args.restart_command:
                # every variant starts the same way, the base one too: a benchmark's baseline gets no
                # head start from a server that has been up and warm for another config
                print(f"  restarting your server: {args.restart_command}")
                rc = subprocess.call(args.restart_command, shell=True)
                if rc != 0:
                    print(f"  the restart command said {rc}; waiting for the server anyway "
                          "(it may restart asynchronously)")
            if not wait_restart(base_url, args.api_key, args.boot_timeout):
                raise SystemExit(f"{label}: the server never became healthy at {base_url}")
            model = cfg.get("model_name") or args.model
            results = run_suite(base_url, args.api_key, model, args.suite, args.repeat,
                                args.warmup, args.temperature, args.seed, args.no_thinking,
                                args.timeout)
            doc = _capture_doc(base_url, args.api_key, model, args, results)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, ensure_ascii=False, indent=1)
            print(f"saved {len(results)} runs -> {path}")
            captures[label] = path
            base_path = captures.get("all-on")
            if base_path and label != "all-on":
                print(f"--- {label} against all-on:")
                compare_pair(base_path, path, verbose=True)
        finally:
            if not args.manual:
                print("stopping the server ...")
                stop_server(proc)
    # ---- the summary: each patch's cost of being off, against the noise floor
    base = captures.get("all-on")
    summary: list[dict[str, Any]] = []
    print("\n=== summary (a negative number: turning the patch off slowed that down) " + "=" * 10)
    header = f"{'patch':<26} {'decode':>9} {'prefill':>9} {'TTFT med':>14}  off means"
    print(header)
    print("-" * len(header))
    for name, _key, _off, why in PATCHES:
        label = f"{name}.off"
        if label not in captures:
            continue
        pair = compare_pair(base, captures[label], verbose=False)
        ttfts_a = [r["ttft_a"] for r in pair["rows"]]
        ttfts_b = [r["ttft_b"] for r in pair["rows"]]
        ttft_note = "-"
        if ttfts_a and ttfts_b:
            sa, sb = sorted(ttfts_a), sorted(ttfts_b)
            mid = len(sa) // 2
            med_a = sa[mid] if len(sa) % 2 else (sa[mid - 1] + sa[mid]) / 2
            mid = len(sb) // 2
            med_b = sb[mid] if len(sb) % 2 else (sb[mid - 1] + sb[mid]) / 2
            ttft_note = f"{med_a:.1f}s->{med_b:.1f}s"
        print(f"{name:<26} {pair['decode_delta_pct']:>+8.1f}% {pair['prefill_delta_pct']:>+8.1f}% "
              f"{ttft_note:>14}  {why}")
        summary.append({"patch": name, **{k: pair[k] for k in
                         ("decode_delta_pct", "prefill_delta_pct", "pairs", "noise_pct",
                          "identical", "changed")},
                        "ttft_note": ttft_note, "off_means": why})
    if any(f"{name}.on" in captures for name, _on, _why in OPTINS):
        print("\nopt-ins (a POSITIVE number: turning it on sped that up):")
        for name, _on, why in OPTINS:
            label = f"{name}.on"
            if label not in captures:
                continue
            pair = compare_pair(base, captures[label], verbose=False)
            ttfts_a = [r["ttft_a"] for r in pair["rows"]]
            ttfts_b = [r["ttft_b"] for r in pair["rows"]]
            ttft_note = "-"
            if ttfts_a and ttfts_b:
                sa, sb = sorted(ttfts_a), sorted(ttfts_b)
                mid = len(sa) // 2
                med_a = sa[mid] if len(sa) % 2 else (sa[mid - 1] + sa[mid]) / 2
                mid = len(sb) // 2
                med_b = sb[mid] if len(sb) % 2 else (sb[mid - 1] + sb[mid]) / 2
                ttft_note = f"{med_a:.1f}s->{med_b:.1f}s"
            print(f"{name:<26} {pair['decode_delta_pct']:>+8.1f}% {pair['prefill_delta_pct']:>+8.1f}% "
                  f"{ttft_note:>14}  {why}")
            summary.append({"optin": name, **{k: pair[k] for k in
                             ("decode_delta_pct", "prefill_delta_pct", "pairs", "noise_pct",
                              "identical", "changed")},
                            "ttft_note": ttft_note, "on_means": why})
    if "all-off" in captures:
        pair = compare_pair(base, captures["all-off"], verbose=False)
        print(f"{'(every patch off)':<26} {pair['decode_delta_pct']:>+8.1f}% "
              f"{pair['prefill_delta_pct']:>+8.1f}%")
        summary.append({"patch": "(every patch off)",
                        **{k: pair[k] for k in ("decode_delta_pct", "prefill_delta_pct", "pairs",
                                                "noise_pct", "identical", "changed")}})
    print("-" * len(header))
    noise = max((s["noise_pct"] for s in summary), default=0.0)
    print(f"a patch's number is outside the run-to-run noise (+/-{noise:.1f}%) before it means anything")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"patches": summary, "noise_pct": noise,
                   "base": captures.get("all-on")}, handle, indent=1)
    md = ["# patch sweep", "",
          f"* base (every patch on): `{captures.get('all-on')}`",
          f"* a negative number: turning the patch off slowed that down",
          f"* numbers within +/-{noise:.1f}% are run-to-run noise", "",
          "| patch | decode | prefill | TTFT med | turning it off means |",
          "|---|---:|---:|---|---|"]
    for s in summary:
        md.append(f"| {s['patch']} | {s['decode_delta_pct']:+.1f}% | {s['prefill_delta_pct']:+.1f}% "
                  f"| {s.get('ttft_note', '-')} | {s.get('off_means', '')} |")
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md) + "\n")
    print(f"wrote {os.path.join(args.out_dir, 'summary.json')} and {md_path}")
    return 0


def _restore_live_config(args: argparse.Namespace) -> None:
    """The end of a sweep, or its failure: the live config back to what it was, the sweep's copy
    of it removed; with a restart command, the server is brought back up on it."""
    path = args.apply_config
    backup = path + ".bak-sweep"
    if not os.path.isfile(backup):
        return
    import shutil
    shutil.copy2(backup, path)
    os.remove(backup)
    print(f"\nrestored your config {path} (the sweep's copy of it is removed)")
    if args.restart_command:
        print(f"restarting your server on it: {args.restart_command}")
        rc = subprocess.call(args.restart_command, shell=True)
        if rc != 0:
            print(f"the restart command said {rc}; restart your server by hand to run on your own "
                  "config again")
            return
        base_url = f"http://127.0.0.1:{args.port}/v1" if not args.manual else args.base_url
        if not wait_restart(base_url, args.api_key, args.boot_timeout):
            print("the server did not come back - restart it by hand")
    else:
        print("restart your server once more to run on your own config")


# --------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="run the prompt suite, save JSON")
    capture.add_argument("--out", required=True)
    capture.add_argument("--suite", choices=("quick", "long", "poly"), default="poly")
    capture.add_argument("--repeat", type=int, default=2)
    capture.add_argument("--warmup", type=int, default=1,
                         help="thrown-away requests before the suite (0 for a truly cold engine)")
    capture.add_argument("--base-url", default="http://127.0.0.1:8095/v1")
    capture.add_argument("--api-key", default="")
    capture.add_argument("--model", default="my-model")
    capture.add_argument("--temperature", type=float, default=0.0)
    capture.add_argument("--seed", type=int, default=42, help="-1 omits the seed from the request")
    capture.add_argument("--no-thinking", action="store_true",
                         help="ask the server to disable thinking (deterministic outputs; "
                              "default keeps thinking on, as in production)")
    capture.add_argument("--timeout", type=float, default=1200.0)
    capture.set_defaults(func=cmd_capture)

    compare = sub.add_parser("compare", help="diff two capture files")
    compare.add_argument("a")
    compare.add_argument("b")
    compare.add_argument("--json", action="store_true")
    compare.set_defaults(func=cmd_compare)

    variants = sub.add_parser("variants", help="write the config of each variant, run nothing")
    variants.add_argument("--config", required=True,
                          help="your engine config, as serve/server.py --config takes it "
                               "(exe, args, cwd, tokenizer, model_name)")
    variants.add_argument("--out-dir", default="bench_variants")
    variants.set_defaults(func=cmd_variants)

    sweep = sub.add_parser("sweep", help="capture all-on, then each patch off, and summarize")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--out-dir", default="bench_patches")
    sweep.add_argument("--root", default=".",
                       help="the PolyStrata checkout serve/server.py runs from")
    sweep.add_argument("--suite", choices=("quick", "long", "poly"), default="poly")
    sweep.add_argument("--repeat", type=int, default=2)
    sweep.add_argument("--warmup", type=int, default=1)
    sweep.add_argument("--only", help="comma-separated patch names (default: all)")
    sweep.add_argument("--skip", help="comma-separated patch names to leave out")
    sweep.add_argument("--with-all-off", action="store_true",
                       help="one more capture with every patch's key off: the series together")
    sweep.add_argument("--force", action="store_true",
                       help="capture a variant again even if its JSON is in the out-dir already")
    sweep.add_argument("--manual", action="store_true",
                       help="do not start serve/server.py: print each variant's engine args and "
                            "wait while you restart your own server")
    sweep.add_argument("--apply-config", metavar="PATH", default=None,
                       help="with --manual: also write each variant's args into this, your live "
                            "config file (the earlier copy is kept beside it), so a restart of "
                            "your usual service picks the variant up")
    sweep.add_argument("--restart-command", metavar="CMD", default=None,
                       help="a shell command that restarts your server, run after each variant is "
                            "written into --apply-config: the sweep is then fully automatic, no "
                            "Enter and no hand work.  E.g. 'systemctl --user restart polystrata'")
    sweep.add_argument("--base-url", default="http://127.0.0.1:8095/v1",
                       help="with --manual: where your server is")
    sweep.add_argument("--port", type=int, default=8137, help="the port of the server it starts")
    sweep.add_argument("--api-key", default="")
    sweep.add_argument("--model", default="my-model",
                       help="a fallback model name when the config has none")
    sweep.add_argument("--temperature", type=float, default=0.0)
    sweep.add_argument("--seed", type=int, default=42)
    sweep.add_argument("--no-thinking", action="store_true")
    sweep.add_argument("--timeout", type=float, default=1200.0)
    sweep.add_argument("--boot-timeout", type=float, default=3600.0,
                       help="seconds to wait for /health after a start (a big model loads slowly)")
    sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args()
    # a sweep ends with the live config back as it was even under a stray SIGTERM (a restart command
    # that kills too much, an admin): the default death would skip the finally below
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        return args.func(args)
    finally:
        # a sweep leaves your live config as it found it, however it ended
        if args.command == "sweep" and getattr(args, "apply_config", None):
            _restore_live_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
