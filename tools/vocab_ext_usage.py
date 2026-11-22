#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vocab_ext_usage.py - what the extended vocab saves and which tail tokens real text uses.

    python3 tools/vocab_ext_usage.py \
        --pack ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer \
        --ext  ~/tokenizer_ru \
        --text ~/corpus/corpus_clean.txt \
        --out  ~/vocab_ext_usage.json

The base count comes from the PACK's own tokenizer (tools/strata_tokenizer.py, the reference
implementation the engine was checked against), not from a rebuilt HF tokenizer: the pack is the
oracle, and its conventions are exactly what the engine tokenizes with.  The extended count comes
from the HF tokenizer.json the trainer wrote.

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
    ap.add_argument("--ext", required=True, help="extended tokenizer dir (tokenizer.json + vocab.json)")
    ap.add_argument("--text", action="append", required=True, help="text file(s), repeatable")
    ap.add_argument("--out", help="where to write {new_id: count} json")
    ap.add_argument("--max-chars", type=int, default=0, help="0 = the whole file")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    base = ST.Tokenizer.from_dir(args.pack)
    n_base = len(base.tokens)
    ext = Tokenizer.from_file(os.path.join(args.ext, "tokenizer.json"))
    vocab = json.load(open(os.path.join(args.ext, "vocab.json"), encoding="utf-8"))
    print(f"[*] база: {n_base} токенов (пак), ext: {len(vocab)}")

    used_all = collections.Counter()
    for path in args.text:
        text = open(path, encoding="utf-8", errors="replace").read()
        if args.max_chars:
            text = text[: args.max_chars]
        ib = base.encode(text)
        ie = ext.encode(text).ids
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
