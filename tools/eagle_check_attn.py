#!/usr/bin/env python3
# tools/eagle_check_attn.py - the drafter's attention against a plain numpy softmax on the very q/k/v it used
# (STRATA_EAGLE_DUMP_ATT).  A cosine well below 1 names the attention as the broken link; 1.0 clears it.
#
#   python3 tools/eagle_check_attn.py <dump_att.bin>
import sys

import numpy as np


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: eagle_check_attn.py <dump_att.bin>")
    f = open(sys.argv[1], "rb")
    pos, nh, nkv, dim, rows = (int(x) for x in f.readline().decode().split())
    per = nh // nkv
    q = np.frombuffer(f.read(4 * nh * dim), dtype="<f4").reshape(nh, dim)
    k = np.frombuffer(f.read(2 * rows * nkv * dim), dtype="<f2").reshape(rows, nkv, dim).astype(np.float32)
    v = np.frombuffer(f.read(2 * rows * nkv * dim), dtype="<f2").reshape(rows, nkv, dim).astype(np.float32)
    o = np.frombuffer(f.read(4 * nh * dim), dtype="<f4").reshape(nh, dim)
    print(f"pos {pos}, heads {nh}, kv {nkv}, dim {dim}, keys {rows}")

    out = np.zeros((nh, dim), dtype=np.float32)
    scale = np.float32(1.0 / np.sqrt(dim))
    for h in range(nh):
        kvh = h // per
        s = (k[:, kvh, :] @ q[h]) * scale
        s = s - s.max()
        w = np.exp(s)
        w /= w.sum()
        out[h] = w @ v[:, kvh, :]

    d = np.abs(out - o)
    den = np.linalg.norm(out) * np.linalg.norm(o)
    cos = float((out * o).sum() / den) if den > 0 else 0.0
    print(f"numpy attention vs engine: max|d| {d.max():.5f}, cos {cos:.6f}")
    print("ВЕРДИКТ:", "внимание верно" if cos > 0.999 else "внимание РАСХОДИТСЯ")


if __name__ == "__main__":
    main()
