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
either an HF dir (--ext) or a built pack (--ext-pack); with both, the first --check-chars are
encoded by each and the id sequences must be IDENTICAL - that is the end-to-end check that the
pack conversion kept the tail 1:1 with the sidecar's row order.

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
    ap.add_argument("--check-chars", type=int, default=200000,
                    help="how much text to compare id-for-id between --ext and --ext-pack")
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

    used_all = collections.Counter()
    for path in args.text:
        text = open(path, encoding="utf-8", errors="replace").read()
        if args.max_chars:
            text = text[: args.max_chars]
        ib = base.encode(text)
        if ext_pk is not None:
            ie = ext_pk.encode(text)
            if ext_hf is not None and args.check_chars > 0:
                head = text[: args.check_chars]
                ih = ext_hf.encode(head).ids
                ip = ext_pk.encode(head)
                if ih != ip:
                    k = next((j for j, (x, y) in enumerate(zip(ih, ip)) if x != y),
                             min(len(ih), len(ip)))
                    raise SystemExit(
                        f"ID MISMATCH at {k}: HF {ih[k:k+5]} vs pack {ip[k:k+5]} "
                        f"(lengths {len(ih)} vs {len(ip)}) - конвенция пака разъехалась")
                print(f"[+] id-сверка HF/пак на {len(head):,} символах: совпало")
        else:
            ie = ext_hf.encode(text).ids
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
