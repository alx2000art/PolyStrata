#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle3_compare.py - the engine's head forward vs the trainer's, stage by stage.

The engine dumps the FIRST read inject's last row's stages (STRATA_EAGLE_DUMP=<file>):
  taps[3*E] fc[E] emb[E] ln1[E] hnorm[E] normed[E]  (float32)
and, once the first draft step runs, normed[E] logits[dvocab] of that step.
This script runs the trainer's Eagle3Head over the SAME row (the first block of the v2 taps dump,
last row of its first inject = the same row the engine dumped), and compares:
  fc == fc(taps)?  ln1 == ln1(emb)?  hnorm == hnorm(fc)?  normed == the full layer's output?
A mismatch names the stage that diverges (taps order, rope, attention, norms).

    python3 eagle3_compare.py --dump ~/eagle_taps/taps-XXXX.bin --engine-dump ~/eagle_dump.bin \
        --safetensors ... --config ...
"""
import argparse
import json
import os
import struct
import sys

import numpy as np


def read_engine_dump(path: str, n_embd: int) -> dict:
    data = np.fromfile(path, dtype=np.float32)
    e = n_embd
    names = ["taps", "fc", "emb", "ln1", "hnorm", "normed"]
    out, off = {}, 0
    widths = {"taps": 3 * e}
    for name in names:
        w = widths.get(name, e)
        if off + w > len(data):
            break
        out[name] = data[off:off + w]
        off += w
    out["_rest"] = data[off:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="v2 taps dump (taps-*.bin) with the read's rows")
    ap.add_argument("--engine-dump", required=True, help="STRATA_EAGLE_DUMP file of the same read")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--row", type=int, default=-1, help="row index inside block 0 (default: last of its first inject)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eagle3_ft_train import Eagle3Head, load_model, load_block, scan_blocks  # noqa: E402
    import torch

    head_cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    E = meta["n_embd"]

    blocks, _, _ = scan_blocks([args.dump])
    if not blocks:
        print("нет блоков в дампе", file=sys.stderr)
        return 1
    path, off, pos0, n_rows, tokens = blocks[0]
    print(f"блок 0: {n_rows} строк, pos0 {pos0}; движковый дамп — стадии его первого inject")
    row = args.row if args.row >= 0 else min(2047, n_rows - 1)
    pos0_b, tokens_b, embs_b, taps_b = load_block(path, off, min(row + 1, n_rows), E, meta["taps_n"])

    eng = read_engine_dump(args.engine_dump, E)
    print(f"стадии движка: {', '.join(k for k in eng if not k.startswith('_'))} "
          f"(+{eng['_rest'].size} float хвостом)")
    if "taps" not in eng or "fc" not in eng:
        print("в движковом дампе нет стадий - включи STRATA_EAGLE_DUMP", file=sys.stderr)
        return 1

    x_embs = torch.from_numpy(embs_b[:row + 1]).to(device)
    x_taps = torch.from_numpy(taps_b[:row + 1]).to(device)
    pos = torch.arange(pos0_b, pos0_b + row + 1, device=device)
    with torch.no_grad():
        h = model.fc(x_taps.reshape(row + 1, -1))
        e = torch.from_numpy(embs_b[:row + 1]).to(device)
        o = model._residual(torch.cat((model.ln1(e), model.hnorm(h)), dim=-1), pos)
        hidden = model._head_out(h, o)
        normed = model.out_norm(hidden)
        got = {
            "fc": h[row].float().cpu().numpy(),
            "emb": embs_b[row],
            "ln1": model.ln1(e[row:row + 1]).float().cpu().numpy()[0],
            "hnorm": model.hnorm(h[row:row + 1]).float().cpu().numpy()[0],
            "normed": normed[row].float().cpu().numpy(),
        }

    print(f"\nстрока {row} блока 0 (абс. поз. {pos0_b + row}):")
    for name in ("taps", "fc", "emb", "ln1", "hnorm", "normed"):
        if name not in eng:
            continue
        a = np.asarray(eng[name], dtype=np.float64)
        b = np.asarray(got.get(name, np.full_like(a, np.nan)), dtype=np.float64)
        if name == "taps":
            b = np.asarray(taps_b[row].reshape(-1), dtype=np.float64)
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        print(f"  {name:7s} cos={cos:+.4f}  |a-b|max={np.abs(a - b).max():.4g}  "
              f"|a|={np.linalg.norm(a):.3g} |b|={np.linalg.norm(b):.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
