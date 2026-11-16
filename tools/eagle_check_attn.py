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

    def cos_of(hyp):
        # every hypothesis compares against the engine's own o, so a matching alt names the layout
        d = np.abs(hyp - o)
        den = np.linalg.norm(hyp) * np.linalg.norm(o)
        c = float((hyp * o).sum() / den) if den > 0 else 0.0
        return d.max(), c

    d, cos = cos_of(out)
    print(f"numpy attention vs engine: max|d| {d.max():.5f}, cos {cos:.6f}")
    verdict = "внимание верно" if cos > 0.999 else "внимание РАСХОДИТСЯ"
    print("ВЕРДИКТ:", verdict)
    if cos <= 0.999:   # the alternates, so one dump names the layout too
        alt = np.zeros_like(out)
        for h in range(nh):
            kvh = h % nkv
            s = (k[:, kvh, :] @ q[h]) * scale
            s = s - s.max()
            w = np.exp(s)
            w /= w.sum()
            alt[h] = w @ v[:, kvh, :]
        d2, cos2 = cos_of(alt)
        print(f"  alt: группы h%nkv  -> max|d| {d2:.5f}, cos {cos2:.6f}")
        nos = np.zeros_like(out)
        for h in range(nh):
            kvh = h // per
            s = k[:, kvh, :] @ q[h]
            s = s - s.max()
            w = np.exp(s)
            w /= w.sum()
            nos[h] = w @ v[:, kvh, :]
        d3, cos3 = cos_of(nos)
        print(f"  alt: без scale     -> max|d| {d3:.5f}, cos {cos3:.6f}")


if __name__ == "__main__":
    main()
