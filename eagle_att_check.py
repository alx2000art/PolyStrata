#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle_att_check.py - the engine's attend_batch vs numpy, on its own dumped q/k/v/o.

The dump (STRATA_EAGLE_DUMP_ATT=<file>, the first teacher step) holds, in fp32:
  pos i32, n_head i32, n_kv i32, head_dim i32, rows i32,
  q [n_head*head_dim], k [rows*n_kv*head_dim] fp16, v [rows*n_kv*head_dim] fp16, o [n_head*head_dim].

    python3 eagle_att_check.py ~/eagle_att.bin
"""
import struct
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "/home/a.makarov/eagle_att.bin"
data = open(path, "rb").read()
# the header is TEXT: "%d %d %d %d %d\n" (pos, n_head, n_kv, head_dim, rows), then the binary payload
hdr_end = data.index(b"\n") + 1
pos, n_head, n_kv, hd, rows = (int(x) for x in data[:hdr_end].split())
rest = data[hdr_end:]
q = np.frombuffer(rest[: n_head * hd * 4], dtype=np.float32).astype(np.float64)
off = n_head * hd * 4
k = np.frombuffer(rest[off: off + rows * n_kv * hd * 2], dtype=np.float16).astype(np.float64)
off += rows * n_kv * hd * 2
v = np.frombuffer(rest[off: off + rows * n_kv * hd * 2], dtype=np.float16).astype(np.float64)
off += rows * n_kv * hd * 2
o = np.frombuffer(rest[off: off + n_head * hd * 4], dtype=np.float32).astype(np.float64)
print(f"pos {pos}, голов {n_head}:{n_kv}, hd {hd}, строк {rows}")

per = n_head // n_kv
scale = 1.0 / np.sqrt(hd)

# q: [n_head, hd]; k/v: [rows, n_kv, hd]; o: [n_head, hd]
Q = q.reshape(n_head, hd)
K = k.reshape(rows, n_kv, hd)
V = v.reshape(rows, n_kv, hd)
O = o.reshape(n_head, hd)

# GQA: голова h -> группа h // per
scores = np.zeros((n_head, rows))
for h in range(n_head):
    g = h // per
    scores[h] = (Q[h] * K[:, g, :]).sum(-1) * scale
W = np.exp(scores - scores.max(-1, keepdims=True))
W /= W.sum(-1, keepdims=True)
mine = np.zeros((n_head, hd))
for h in range(n_head):
    g = h // per
    mine[h] = (W[h][:, None] * V[:, g, :]).sum(0)

cos = float((mine.flatten() @ O.flatten()) / max(np.linalg.norm(mine) * np.linalg.norm(O), 1e-12))
print(f"numpy softmax(qk)v vs движковый o: cos={cos:+.4f}  |numpy|={np.linalg.norm(mine):.4g}  "
      f"|движок|={np.linalg.norm(O):.4g}")

# и построчно: какая голова ближе/дальше
per_head = []
for h in range(n_head):
    c = float(mine[h] @ O[h] / max(np.linalg.norm(mine[h]) * np.linalg.norm(O[h]), 1e-12))
    per_head.append(c)
per_head = np.array(per_head)
print(f"по головам: cos min={per_head.min():+.3f} median={np.median(per_head):+.3f} max={per_head.max():+.3f}")
print("худшие головы:", np.argsort(per_head)[:6].tolist())
print("веса головы 0 (движок не пишет их - только наша проверка формы):",
      np.round(W[0], 3).tolist()[:8], "...")
