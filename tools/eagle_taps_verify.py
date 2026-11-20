#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle_taps_verify.py — decode-дамп против read-дампа ТОГО ЖЕ текста.

Вопрос: taps, которые движок снимает на decode-строках, — те же, что при read?
(Их ложность не ловится ничем в движке: ROUND_AB=1 сравнивает fc(taps) с самим собой.)

Как:
  1. оба дампа читаются целиком (ETAP v2: pos0/rows/tokens/[embs|taps] bf16), перекрытия
     парных decode-блоков схлопываются по абсолютной позиции;
  2. последовательность decode-токенов выравнивается по вхождению в read-последовательность
     (первые PROBE токенов совпадают подряд) — тексты те же, токенайзер детерминирован;
  3. на выровненных строках считается cos для embs и каждого из taps (min/median/max +
     худшие строки).  cos ≈ 1 на последних строках (где контексты сходятся) = taps верны.

Плюс печатает команду полного офлайн-замера (eval-only на read-дампе с
STRATA_EVAL_POS = позиция начала ответа): leg1/leg3/chain в ПОЛНОМ контексте.

    # read-дамп того же текста: PROMPT + ответ diag-прогона
    cat ~/prompts/doc-2k.txt ~/PolyStrata/diag_run/diag-run.txt > /tmp/pa.txt
    EAGLE_DUMP_TAPS=~/eagle_taps_read ... eagle_dump_corpus.py --corpus /tmp/pa.txt --chars 80000
    python3 tools/eagle_taps_verify.py --decode <taps-decode-*.bin> --read <taps-read-*.bin>
"""
import argparse
import os
import struct

import numpy as np

MAGIC = 0x50415445  # "ETAP"


def load_dump(path):
    f = open(path, "rb")
    magic, ver, n_embd, taps_n, has_emb = struct.unpack("<IIIII", f.read(20))
    if magic != MAGIC:
        raise SystemExit(f"{path}: не ETAP-дамп (magic {magic:#x})")
    if ver != 2 or not has_emb:
        raise SystemExit(f"{path}: версия {ver}, has_emb {has_emb} — ожидался v2 с эмбеддингами")
    seen, poss, toks, embs, taps = set(), [], [], [], []
    while True:
        hdr = f.read(12)
        if len(hdr) < 12:
            break
        pos0, rows = struct.unpack("<QI", hdr)
        tk = np.frombuffer(f.read(4 * rows), dtype="<i4")
        raw = np.frombuffer(f.read(2 * rows * (taps_n + 1) * n_embd), dtype="<u2") \
            .reshape(rows, taps_n + 1, n_embd)
        e = (raw[:, 0, :].astype(np.uint32) << 16).view(np.float32)
        t = (raw[:, 1:, :].astype(np.uint32) << 16).view(np.float32)
        for i in range(rows):
            p = pos0 + i
            if p in seen:
                continue  # перекрытие парных блоков
            seen.add(p)
            poss.append(p)
            toks.append(int(tk[i]))
            embs.append(e[i])
            taps.append(t[i])
    o = np.argsort(poss)
    return (np.array(poss)[o], np.array(toks)[o], np.stack(embs)[o], np.stack(taps)[o], n_embd, taps_n)


def cos_rows(a, b):
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    return np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)


def align(dec_toks, rd_toks, min_run=8):
    """decode-дамп содержит и ОТКЛОНЁННЫЕ draft-строки (токены-выдумки), read — только текст:
    матчим строки по 3-окну токенов (в тексте почти уникальны, вставки-выдумки окон не дают).
    Возвращает список пар (decode_row, read_row)."""
    rmap = {}
    for k in range(len(rd_toks) - 2):
        rmap.setdefault((int(rd_toks[k]), int(rd_toks[k + 1]), int(rd_toks[k + 2])), []).append(k)
    pairs, used = [], set()
    for i2 in range(len(dec_toks) - 2):
        key = (int(dec_toks[i2]), int(dec_toks[i2 + 1]), int(dec_toks[i2 + 2]))
        for k in rmap.get(key, ()):
            if k not in used:
                pairs.append((i2, k))
                used.add(k)
                break
    return pairs if len(pairs) >= min_run else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode", required=True, help="taps-decode-*.bin из DUMP_DECODE на diag-прогоне")
    ap.add_argument("--read", required=True, help="read-дамп того же текста (prompt+ответ)")
    ap.add_argument("--probe", type=int, default=16)
    ap.add_argument("--by-pos", action="store_true",
                    help="сравнивать строки с РАВНЫМИ позициями (read и decode нумеруют абсолютно)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    dp, dt, de, dk, n_embd, taps_n = load_dump(args.decode)
    rp, rt, re_, rk, n_embd2, taps_n2 = load_dump(args.read)
    if (n_embd, taps_n) != (n_embd2, taps_n2):
        raise SystemExit(f"дампы разной формы: {n_embd},{taps_n} vs {n_embd2},{taps_n2}")
    print(f"decode: {len(dt)} строк, pos {dp[0]}..{dp[-1]}")
    print(f"read:   {len(rt)} строк, pos {rp[0]}..{rp[-1]}")

    if args.by_pos:
        # строки с РАВНЫМИ позициями (оба дампа нумеруют абсолютно): чистая сверка capture
        rm = {int(p): i for i, p in enumerate(rp)}
        pairs = [(i2, rm[int(p)]) for i2, p in enumerate(dp) if int(p) in rm]
        pairs = pairs[: args.limit] if args.limit else pairs
        print(f"по позициям: {len(pairs)} общих строк, pos {dp[pairs[0][0]]}..{dp[pairs[-1][0]]}" if pairs
              else "по позициям: общих строк нет")
    else:
        pairs = align(dt, rt, args.probe)
    if pairs is None:
        raise SystemExit(f"выравнивание не удалось (нет >= {args.probe} согласованных строк): "
                         f"decode[:8]={dt[:8].tolist()} read[:8]={rt[:8].tolist()}")
    di = np.array([p[0] for p in pairs]); ri = np.array([p[1] for p in pairs])
    print(f"выравнивание: {len(pairs)} согласованных строк, decode pos {dp[di[0]]}..{dp[di[-1]]} <-> "
          f"read pos {rp[ri[0]]}..{rp[ri[-1]]}")
    dt = dt[di]; dp = dp[di]; de = de[di]; dk = dk[di]
    rt = rt[ri]; rp = rp[ri]; re_ = re_[ri]; rk = rk[ri]
    start = 0
    n = len(pairs)

    ce = cos_rows(de[:n], re_[start:start + n])
    print(f"\n{'embs':>6}: cos min={ce.min():+.4f} median={np.median(ce):+.4f} max={ce.max():+.4f}")
    for k in range(taps_n):
        c = cos_rows(dk[:n, k, :], rk[start:start + n, k, :])
        print(f"tap {k:>2}: cos min={c.min():+.4f} median={np.median(c):+.4f} max={c.max():+.4f}")

    # хвост (контексты сходятся — там вердикт чище) и худшие строки
    tail = slice(max(0, n - max(16, n // 4)), n)
    for k in range(taps_n):
        c = cos_rows(dk[tail][:, k, :], rk[start:start + n][tail][:, k, :])
        print(f"tap {k:>2} (хвост {n - tail.start} строк): median cos={np.median(c):+.4f}")
    c_all = np.mean([cos_rows(dk[:n, k, :], rk[start:start + n][:, k, :]) for k in range(taps_n)], axis=0)
    worst = np.argsort(c_all)[:5]
    print("\nхудшие строки (по среднему cos taps):")
    for i in worst:
        print(f"  decode pos {dp[i]:>6} / read pos {rp[start + i]:>6}: cos={c_all[i]:+.3f} "
              f"tok={dt[i]}")

    print("\n=== полный офлайн-замер на read-дампе (полный контекст) ===")
    print(f"STRATA_EVAL_POS={rp[0]} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
          f"~/.gpu-bench/bin/python eagle3_ft_train.py --dump {args.read} "
          f"--safetensors <голова> --config <конфиг головы> --eval-only --max-rows 512")


if __name__ == "__main__":
    main()
