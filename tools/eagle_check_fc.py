#!/usr/bin/env python3
# tools/eagle_check_fc.py - checks the drafter's fc step against the file, on the very numbers the engine
# produced (STRATA_EAGLE_DUMP): fc.weight (BF16, [n_embd, 3*n_embd]) times the dumped taps against the
# dumped fc output.  It also tries every ordering of the three taps, so a wrong layer order shows itself.
#
#   python3 tools/eagle_check_fc.py ~/models/<head>.safetensors ~/eagle_dump.bin ~/models/<head>.config.json
import json
import struct
import sys

import numpy as np


def bf16_to_f32(raw):
    u = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
    return u.view(np.float32)


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: eagle_check_fc.py <head.safetensors> <dump.bin> <config.json>")
    st_path, dump_path, cfg_path = sys.argv[1:4]
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    n_embd = cfg["hidden_size"]

    f = open(st_path, "rb")
    n, = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    base = 8 + n
    fc = hdr["fc.weight"]
    f.seek(base + fc["data_offsets"][0])
    w = bf16_to_f32(f.read(fc["data_offsets"][1] - fc["data_offsets"][0]))
    out_dim, in_dim = fc["shape"]
    w = w.reshape(out_dim, in_dim)
    print(f"fc.weight: {out_dim} x {in_dim}, taps = {in_dim // n_embd}")

    d = np.fromfile(dump_path, dtype="<f4")
    taps = d[: 3 * n_embd].reshape(3, n_embd)
    ref = d[3 * n_embd : 3 * n_embd + n_embd]
    print(f"dump: taps {taps.shape}, engine fc out {ref.shape}")

    def score(perm, mode):
        t = [taps[p].astype(np.float32) for p in perm]
        if mode == "rms":      # each tap RMS-normed on its own (the model's states are ~unit scale)
            t = [v / (np.sqrt((v * v).mean()) + 1e-6) for v in t]
        elif mode == "rms-same":   # all three taps with one scale
            s = np.sqrt(np.mean(np.concatenate(t) ** 2)) + 1e-6
            t = [v / s for v in t]
        x = np.concatenate(t).astype(np.float32)
        got = (w @ x).astype(np.float32)           # fc: out = W (out x in) . x
        diff = np.abs(got - ref).max()
        denom = np.linalg.norm(got) * np.linalg.norm(ref)
        cos = float(got @ ref / denom) if denom > 0 else 0.0
        rel = float(diff / (np.abs(ref).max() + 1e-9))
        return diff, rel, cos

    print(f"taps stats: mean|.| {np.abs(taps).mean():.4f}, max|.| {np.abs(taps).max():.2f}, "
          f"rms {np.sqrt((taps ** 2).mean()):.3f}; ref mean|.| {np.abs(ref).mean():.4f}, "
          f"rms {np.sqrt((ref ** 2).mean()):.3f}")
    best = None
    for mode in ("raw", "rms", "rms-same"):
        for perm in ((0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)):
            diff, rel, cos = score(perm, mode)
            print(f"  {mode:8s} taps {perm}: max|diff| {diff:.4f}, rel {rel:.3e}, cos {cos:.6f}")
            if best is None or cos > best[2]:
                best = (mode, perm, cos)
    print(f"best: {best[0]} order {best[1]} (cos {best[2]:.6f})")
    if best[2] < 0.99:
        print("ВЕРДИКТ: fc не сходится - расходятся сами taps (не порядок и не масштаб)")
    else:
        print("ВЕРДИКТ: fc сходится - вход (taps) надо брать так, как в лучшей строке")


if __name__ == "__main__":
    sys.exit(main())
