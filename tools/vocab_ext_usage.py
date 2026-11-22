#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vocab_ext_usage.py - what the extended vocab saves and which tail tokens real text uses.

    python3 tools/vocab_ext_usage.py \
        --pack ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer \
        --ext  ~/tokenizer_ru_pruned \
        --ext-pack ~/tokenizer_ru_pruned_pack \
        --text ~/corpus/corpus_clean.txt \
        --out  ~/vocab_ext_usage.json

--pack is the BASE pack (the oracle tokenizer: tools/strata_tokenizer.py).  The extended side is
either an HF dir (--ext) or a built pack (--ext-pack).  MEASURE WITH --ext-pack: the pack is what
the engine tokenizes with, and an HF pipeline assembled from a single pre_pattern does not match
the engine's PRE_SEQUENCES (it differs already on base tokens).  With both given, the tail is
checked structurally: every id >= base must carry the SAME token string in both - that is the
alignment the sidecar's row order depends on.

Output json: {new_id: occurrences} for every tail id the text uses - the input of
prune_vocab_ext.py.
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as ST  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="base pack dir (vocab.json/merges.txt/tokenizer.json)")
    ap.add_argument("--ext", help="extended HF tokenizer dir (tokenizer.json + vocab.json)")
    ap.add_argument("--ext-pack", help="extended PACK dir (the engine's format)")
    ap.add_argument("--text", action="append", required=True, help="text file(s), repeatable")
    ap.add_argument("--out", help="where to write {new_id: count} json")
    ap.add_argument("--max-chars", type=int, default=0, help="0 = the whole file")
    args = ap.parse_args()
    if not args.ext and not args.ext_pack:
        raise SystemExit("give --ext and/or --ext-pack")

    base = ST.Tokenizer.from_dir(args.pack)
    n_base = len(base.tokens)
    ext_hf = None
    if args.ext:
        from tokenizers import Tokenizer
        ext_hf = Tokenizer.from_file(os.path.join(args.ext, "tokenizer.json"))
        n_ext = len(json.load(open(os.path.join(args.ext, "vocab.json"), encoding="utf-8")))
    else:
        n_ext = 0
    ext_pk = ST.Tokenizer.from_dir(args.ext_pack) if args.ext_pack else None
    if ext_pk is not None:
        n_ext = len(ext_pk.tokens)
    print(f"[*] база: {n_base} токенов (пак), ext: {n_ext}")

    # структурная сверка: id хвоста <-> строка токена <-> строка сайдкара.
    # Последовательности id HF и пака сравнивать НЕЛЬЗЯ: HF-пайплайн, собранный по одному
    # pre_pattern, расходится с PRE_SEQUENCES движка уже на базовых токенах (так и вышло).
    if args.ext and ext_pk is not None:
        hv = json.load(open(os.path.join(args.ext, "vocab.json"), encoding="utf-8"))
        tail_hf = sorted((i, t) for t, i in hv.items() if i >= n_base)
        tail_pk = [(i, ext_pk.tokens[i]) for i in range(n_base, len(ext_pk.tokens))]
        if len(tail_hf) != len(tail_pk):
            raise SystemExit(f"tail sizes differ: HF {len(tail_hf)} vs pack {len(tail_pk)}")
        for (i, a), (j, b) in zip(tail_hf, tail_pk):
            if i != j or a != b:
                raise SystemExit(f"tail misaligned at id {i}: HF {a!r} vs pack {b!r}")
        print(f"[+] хвост 1:1: {len(tail_hf)} токенов, id и строки совпадают")

    used_all = collections.Counter()
    for path in args.text:
        text = open(path, encoding="utf-8", errors="replace").read()
        if args.max_chars:
            text = text[: args.max_chars]
        ib = base.encode(text)
        ie = ext_pk.encode(text) if ext_pk is not None else ext_hf.encode(text).ids
        used = collections.Counter(i for i in ie if i >= n_base)
        used_all.update(used)
        tot = sum(used.values())
        save = 100.0 * (1.0 - len(ie) / max(1, len(ib)))
        acc, marks = 0, {}
        for k, (_, c) in enumerate(used.most_common(), 1):
            acc += c
            for f in (0.90, 0.99, 0.999):
                if f not in marks and acc >= f * tot:
                    marks[f] = k
        print(f"[+] {path}: {len(text):,} символов, база {len(ib):,} -> ext {len(ie):,} "
              f"({save:.2f}% меньше)")
        print(f"    новых употреблений {tot:,} ({100.0 * tot / max(1, len(ie)):.1f}%), "
              f"различных {len(used):,}, покрытие 90/99/99.9%: {marks}")
    if args.out:
        json.dump({str(i): c for i, c in used_all.items()}, open(args.out, "w", encoding="utf-8"))
        print(f"[+] частоты: {args.out} ({len(used_all)} различных новых id)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
