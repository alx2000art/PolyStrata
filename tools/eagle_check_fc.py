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

    def score(perm):
        x = np.concatenate([taps[p] for p in perm]).astype(np.float32)
        got = (w @ x).astype(np.float32)           # fc: out = W (out x in) . x
        diff = np.abs(got - ref).max()
        denom = np.linalg.norm(got) * np.linalg.norm(ref)
        cos = float(got @ ref / denom) if denom > 0 else 0.0
        rel = float(diff / (np.abs(ref).max() + 1e-9))
        return diff, rel, cos

    best = None
    for perm in ((0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)):
        diff, rel, cos = score(perm)
        tag = " <- tap order as the engine has it" if perm == (0, 1, 2) else ""
        print(f"  taps {perm}: max|diff| {diff:.4f}, rel {rel:.3e}, cos {cos:.6f}{tag}")
        if best is None or cos > best[1]:
            best = (perm, cos)
    print(f"best order {best[0]} (cos {best[1]:.6f})")
    if best[1] < 0.99:
        print("ВЕРДИКТ: fc не сходится ни при каком порядке taps - расходятся ВЕСА или сам taps (не порядок)")


if __name__ == "__main__":
    sys.exit(main())
