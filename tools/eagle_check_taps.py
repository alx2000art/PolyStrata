#!/usr/bin/env python3
# tools/eagle_check_taps.py - which collapse of DeepSeek V4's streams the drafter's fc expects.  The engine
# dumps, per tap layer, that layer's row of streams (STRATA_EAGLE_DUMP_R: 3 blocks of kHc*n_embd) and its
# own fc input/output (STRATA_EAGLE_DUMP).  This recomputes fc for every plausible collapse, per slot and
# with one collapse for all three, and reports the cosine against the engine's own fc output.
#
#   python3 tools/eagle_check_taps.py <head.safetensors> <dump.bin> <dump_r.bin> <config.json>
import json
import struct
import sys
from itertools import product

import numpy as np


def bf16(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)


def main():
    if len(sys.argv) != 5:
        raise SystemExit("usage: eagle_check_taps.py <head.safetensors> <dump.bin> <dump_r.bin> <config.json>")
    st, dump, dumper, cf = sys.argv[1:5]
    cfg = json.load(open(cf, encoding="utf-8"))
    n_embd = cfg["hidden_size"]

    f = open(st, "rb")
    n, = struct.unpack("<Q", f.read(8))
    hdr = json.loads(f.read(n))
    base = 8 + n
    t = hdr["fc.weight"]
    f.seek(base + t["data_offsets"][0])
    w = bf16(f.read(t["data_offsets"][1] - t["data_offsets"][0])).reshape(t["shape"])

    d = np.fromfile(dump, dtype="<f4")
    eng_taps = d[0:3 * n_embd].reshape(3, n_embd)
    fc_out = d[3 * n_embd:4 * n_embd]
    r = np.fromfile(dumper, dtype="<f4")
    layers = r.size // (4 * n_embd)
    r = r.reshape(layers, 4, n_embd)
    print(f"streams dumped for {layers} tap layers; rms per stream of layer 0: "
          f"{np.round(np.sqrt((r[0] ** 2).mean(axis=1)), 3)}")

    def cos_of(x):
        got = w @ x
        den = np.linalg.norm(got) * np.linalg.norm(fc_out)
        return float(got @ fc_out / den) if den > 0 else 0.0

    print(f"engine taps rms per slot: {np.round(np.sqrt((eng_taps ** 2).mean(axis=1)), 3)}")

    modes = {"s0": lambda a: a[0], "s1": lambda a: a[1], "s2": lambda a: a[2], "s3": lambda a: a[3],
             "mean": lambda a: a.mean(axis=0), "sum": lambda a: a.sum(axis=0)}
    print("\none collapse for all three slots:")
    for name, fn in modes.items():
        x = np.concatenate([fn(r[min(i, layers - 1)]) for i in range(3)])
        print(f"  {name:5s} cos {cos_of(x):.6f}")

    print("\nper-slot collapse, all combinations:")
    best = None
    for combo in product(modes.items(), repeat=3):
        x = np.concatenate([combo[i][1](r[min(i, layers - 1)]) for i in range(3)])
        c = cos_of(x)
        if best is None or c > best[1]:
            best = ([n_ for n_, _ in combo], c)
    print(f"  best {best[0]} cos {best[1]:.6f}")
    print(f"\nengine's own taps cos {cos_of(np.concatenate(eng_taps)):.6f}")


if __name__ == "__main__":
    main()
