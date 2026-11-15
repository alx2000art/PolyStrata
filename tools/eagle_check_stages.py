#!/usr/bin/env python3
# tools/eagle_check_stages.py - checks the drafter's every stage against the file, on the numbers the
# engine itself produced (STRATA_EAGLE_DUMP).  Each stage is recomputed here from the previous one, so a
# stage that does not match names the line where the engine and the formula part ways.
#
#   python3 tools/eagle_check_stages.py <head.safetensors> <dump.bin> <config.json>
import json
import struct
import sys

import numpy as np


def bf16(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def load_tensor(f, base, hdr, name):
    t = hdr[name]
    f.seek(base + t["data_offsets"][0])
    raw = f.read(t["data_offsets"][1] - t["data_offsets"][0])
    w = bf16(raw) if t["dtype"] == "BF16" else np.frombuffer(raw, dtype="<f4")
    return w.reshape(t["shape"]), t["shape"]


def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * w


def cmp(name, got, want):
    d = np.abs(got - want)
    den = np.linalg.norm(got) * np.linalg.norm(want)
    cos = float(got @ want / den) if den > 0 else 0.0
    print(f"{name:22s} max|d| {d.max():9.4f}  rel {d.max()/(np.abs(want).max()+1e-9):8.3e}  cos {cos:.6f}")
    return cos


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: eagle_check_stages.py <head.safetensors> <dump.bin> <config.json>")
    st, dump, cf = sys.argv[1:4]
    cfg = json.load(open(cf, encoding="utf-8"))
    n_embd = cfg["hidden_size"]
    eps = cfg["rms_norm_eps"]
    dvocab = cfg["draft_vocab_size"]

    f = open(st, "rb")
    n, = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    base = 8 + n

    d = np.fromfile(dump, dtype="<f4")
    off = 0
    def take(cnt):
        nonlocal off
        v = d[off:off + cnt].copy()
        off += cnt
        return v
    taps = take(3 * n_embd).reshape(3, n_embd)
    fc_out = take(n_embd)
    emb = take(n_embd)
    e_norm = take(n_embd)
    f_norm = take(n_embd)
    normed = take(n_embd)
    # the draft step appends its own normed row, then its logits - the lm_head check needs the pair
    normed_step = take(n_embd) if len(d) >= off + n_embd + dvocab else None
    logits = take(dvocab) if len(d) >= off + dvocab else None

    print(f"dump: {len(d)} floats; logits {'yes' if logits is not None else 'no'}\n")
    scores = []

    # 1. fc
    wq, _ = load_tensor(f, base, hdr, "fc.weight")
    scores.append(cmp("fc", wq @ np.concatenate(list(taps)), fc_out))

    # 2. ln1(emb) and hnorm(fc)
    ln1, _ = load_tensor(f, base, hdr, "midlayer.input_layernorm.weight")
    hn, _ = load_tensor(f, base, hdr, "midlayer.hidden_norm.weight")
    scores.append(cmp("ln1(emb)", rmsnorm(emb, ln1, eps), e_norm))
    scores.append(cmp("hnorm(fc)", rmsnorm(fc_out, hn, eps), f_norm))

    # 3. the lm_head on the draft step's own normed row
    if logits is not None and normed_step is not None:
        wl, _ = load_tensor(f, base, hdr, "lm_head.weight")
        lg = wl @ normed_step
        scores.append(cmp("lm_head(step)", lg, logits))
        got = int(np.argmax(lg))
        want = int(np.argmax(logits))
        print(f"  argmax: from the file {got}, the engine {want}  {'MATCH' if got == want else 'DIFFER'}")
        print(f"  step normed rms {np.sqrt((normed_step**2).mean()):.3f}")
    elif logits is not None:
        wl, _ = load_tensor(f, base, hdr, "lm_head.weight")
        scores.append(cmp("lm_head(normed)", wl @ normed, logits))
    if True:
        on, _ = load_tensor(f, base, hdr, "norm.weight")
        print(f"  norm.weight: mean {on.mean():.4f}, min {on.min():.4f}, max {on.max():.4f}")
        print(f"  normed stats: mean|.| {np.abs(normed).mean():.4f}, rms {np.sqrt((normed**2).mean()):.3f}")
        print(f"  e_norm rms {np.sqrt((e_norm**2).mean()):.3f}, f_norm rms {np.sqrt((f_norm**2).mean()):.3f}")

    print(f"\nlowest stage cos: {min(scores):.6f}")


if __name__ == "__main__":
    main()
