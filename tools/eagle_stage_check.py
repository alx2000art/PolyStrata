#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle_stage_check.py - сверка стадий шага драфта: движок против тренера.

    python3 tools/eagle_stage_check.py <stage.bin> --safetensors ... --config ... \
        [--round ~/eagle_round.bin] [--cpu]

Движок пишет (STRATA_EAGLE_DUMP_STAGE=<файл> STRATA_EAGLE_DUMP_STAGE_POS=<позиция>):
    hdr: i32 n_embd, i32 q_cols (n_head*hd), i32 kv_cols (n_kv*hd); i64 pos
    6 частей: e_norm, f_norm, q, k, v, o  (u64 длина + f32 значения)
q/k/v - ПОСЛЕ rope; o - выход внимания до o_proj.

Скрипт считает ту же цепочку своей математикой (на том же hidden/emb из round-дампа)
и печатает по стадии cos и max|diff|.  Так видно, ЧТО расходится:
  * e/f уже разные  -> нормы/eps/входы (hidden, emb)
  * q/k/v разные    -> веса проекций или rope
  * q/k/v совпали, o нет -> само внимание (K/V-строки, маска, порядок голов)
"""
import argparse
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from eagle3_ft_train import load_model, rmsnorm, rope_apply  # noqa: E402


def read_round(path):
    raw = open(path, "rb").read()
    ne, tok = struct.unpack("<ii", raw[:8])
    pos, = struct.unpack("<q", raw[8:16])
    hv = np.frombuffer(raw[16:16 + 4 * ne], dtype="<f4").copy()
    ev = np.frombuffer(raw[16 + 4 * ne:16 + 8 * ne], dtype="<f4").copy()
    return ne, tok, pos, hv, ev


def read_stage(path):
    raw = open(path, "rb").read()
    ne, qc, kvc = struct.unpack("<iii", raw[:12])
    pos, = struct.unpack("<q", raw[12:20])
    off, parts = 20, []
    for _ in range(6):
        n, = struct.unpack("<Q", raw[off:off + 8]); off += 8
        parts.append(np.frombuffer(raw[off:off + 4 * n], dtype="<f4").copy()); off += 4 * n
    return dict(ne=ne, qc=qc, kvc=kvc, pos=pos,
                e=parts[0], f=parts[1], q=parts[2], k=parts[3], v=parts[4], o=parts[5])


def cmp(name, a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {name:6s} ФОРМА {a.shape} против {b.shape} — не сравнить")
        return
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mxd = float(np.max(np.abs(a - b)))
    print(f"  {name:6s} cos={cos:+.6f}  maxdiff={mxd:.4g}  |движок|={np.linalg.norm(a):.4g}  |тренер|={np.linalg.norm(b):.4g}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--round", default=None, help="round-дамп (hidden/emb); по умолчанию stage-файл без .stage")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    head_cfg = {}
    if args.config:
        import json
        head_cfg = json.load(open(args.config, encoding="utf-8"))
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    st = read_stage(args.stage)
    rpath = args.round or args.stage.replace(".stage", "")
    ne, tok, pos, hv, ev = read_round(rpath)
    print(f"стадии: pos {st['pos']} (round {pos}), token {tok}, E {st['ne']}, "
          f"q {st['qc']}, kv {st['kvc']}")
    with torch.no_grad():
        h = torch.from_numpy(hv[None]).to(device)
        e = torch.from_numpy(ev[None]).to(device)
        e_n = rmsnorm(e, model.ln1.weight, model.eps)
        f_n = rmsnorm(h, model.hnorm.weight, model.eps)
        x = torch.cat((e_n, f_n), dim=-1)
        q = rope_apply(model.wq(x).view(1, model.n_head, model.hd), torch.tensor([pos], device=device), model.rope)
        k = rope_apply(model.wk(x).view(1, model.n_kv, model.hd), torch.tensor([pos], device=device), model.rope)
        v = model.wv(x).view(1, model.n_kv, model.hd)
    print("сверка (движок против тренера):")
    cmp("e", st["e"], e_n[0].cpu().numpy())
    cmp("f", st["f"], f_n[0].cpu().numpy())
    cmp("q", st["q"], q[0].reshape(-1).cpu().numpy())
    cmp("k", st["k"], k[0].reshape(-1).cpu().numpy())
    cmp("v", st["v"], v[0].reshape(-1).cpu().numpy())
    print("  o      (выход внимания — сравнивать вместе с --replay-ctx: нужен кэш)")
    print("\nчтение: e/f разные -> нормы/входы; q/k/v разные -> веса или rope; "
          "q/k/v совпали -> расходятся сами K/V-строки или маска.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
