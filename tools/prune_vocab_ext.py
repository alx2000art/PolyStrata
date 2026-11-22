#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prune_vocab_ext.py - drop tail tokens the corpus never uses, keeping the BPE merge closure.

WHY.  The extended vocab (98,071 tail tokens) was trained on 2026-11-20, before the corpus
encoding fix, so its tail carries 3,471 tokens of U+FFFD runs, 2,766 base64 fragments and 30,587
ASCII tokens the base vocab already covers.  The clean part is 60,431 real Russian words (median
8 chars) and it is worth keeping - retraining on 8M clean chars would give far fewer long words.

A tail token can be dropped only if no KEPT token needs it as a merge piece; otherwise BPE stops
assembling that token.  The kept set is therefore the merge closure (restricted to the tail,
the base alphabet is always available) of the clean Cyrillic tokens the corpus uses.  The closure
admits CYRILLIC intermediates only: an ASCII piece in the tail (" 5" for " 5-й") would open a new
merge on pure-ASCII text and change English/code tokenization - and those must stay bit-identical,
so a Russian token that needs such a piece is dropped instead.

    python3 tools/prune_vocab_ext.py \
        --ext ~/tokenizer_ru --usage ~/vocab_ext_usage.json \
        --out ~/tokenizer_ru_pruned --min-count 1

Output: a HF tokenizer directory (tokenizer.json/vocab.json/merges.txt) in the --ext convention,
tail renumbered from BASE.  Base ids never move; merge order (rank) is preserved.
"""
import argparse
import collections
import json
import os
import sys

# GPT-2 byte-level alphabet: code point -> byte.
_BS = list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
_CS = _BS[:]
_n = 0
for _b in range(256):
    if _b not in _BS:
        _BS.append(_b)
        _CS.append(256 + _n)
        _n += 1
_DEC = dict(zip(_CS, _BS))
_CY = set(chr(c) for c in range(0x400, 0x500))


def raw_bytes(tok: str):
    try:
        return bytes(_DEC[ord(c)] for c in tok)
    except KeyError:
        return None


def cls(tok: str) -> str:
    b = raw_bytes(tok)
    if b is None:
        return "not_bytes"
    if b"\xef\xbf\xbd" in b:
        return "u_fffd"
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return "fragment"
    if any(c in _CY for c in s):
        return "ru"
    if all(ord(c) < 128 for c in s):
        return "ascii"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", required=True, help="extended tokenizer dir to prune")
    ap.add_argument("--usage", help="json from vocab_ext_usage.py; without it all clean tail tokens are seeds")
    ap.add_argument("--base-vocab", type=int, default=129280, help="tail starts at this id")
    ap.add_argument("--min-count", type=int, default=1, help="keep a used seed only from this count up")
    ap.add_argument("--keep-ascii", action="store_true", help="also seed ASCII tokens (debug)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    BASE = args.base_vocab

    vocab = json.load(open(os.path.join(args.ext, "vocab.json"), encoding="utf-8"))
    merges = []
    with open(os.path.join(args.ext, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))
    tail = {t for t, i in vocab.items() if i >= BASE}
    alphabet = {t for t, i in vocab.items() if i < BASE}
    if len(tail) + len(alphabet) != len(vocab):
        raise SystemExit("vocabulary is not a 0..N-1 prefix plus a tail")

    usage = {}
    if args.usage:
        usage = {int(k): v for k, v in json.load(open(args.usage, encoding="utf-8")).items()}
    seed = set()
    for t in tail:
        count = usage.get(vocab[t], 0) if args.usage else 1
        if count < args.min_count:
            continue
        k = cls(t)
        if k == "ru" or (args.keep_ascii and k == "ascii"):
            seed.add(t)

    dep = {}
    for a, b in merges:
        dep[a + b] = (a, b)

    keep = set(seed)
    stack = list(seed)
    while stack:
        t = stack.pop()
        ab = dep.get(t)
        if not ab:
            continue
        for p in ab:
            # Только кириллические токены: ASCII-промежуток (например, " 5" для " 5-й") в хвосте
            # открыл бы новый мердж на чистом ASCII и изменил бы токенизацию английского и кода.
            # Русский токен, которому такой промежуток нужен, отбросит fixpoint ниже.
            if p in tail and p not in keep and cls(p) == "ru":
                keep.add(p)
                stack.append(p)

    # A kept token must be buildable from the base alphabet and other kept tokens.
    changed = True
    while changed:
        changed = False
        for t in list(keep):
            ab = dep.get(t)
            if not ab or not all((p in alphabet) or (p in keep) for p in ab):
                keep.discard(t)
                changed = True

    tail_sorted = sorted((i, t) for t, i in vocab.items() if i >= BASE and t in keep)
    new_vocab = {t: i for t, i in vocab.items() if i < BASE}
    for k, (_, t) in enumerate(tail_sorted):
        new_vocab[t] = BASE + k
    new_merges = [(a, b) for a, b in merges
                  if a in new_vocab and b in new_vocab and (a + b) in new_vocab]

    tj = json.load(open(os.path.join(args.ext, "tokenizer.json"), encoding="utf-8"))
    tj["model"]["vocab"] = new_vocab
    orig = tj["model"].get("merges", [])
    was_str = bool(orig) and isinstance(orig[0], str)
    norm = []
    for m in orig:
        a, b = (m.split(" ", 1) if isinstance(m, str) else (m[0], m[1]))
        if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
            norm.append([a, b])
    tj["model"]["merges"] = [" ".join(m) for m in norm] if was_str else norm

    os.makedirs(args.out, exist_ok=True)
    json.dump(new_vocab, open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    with open(os.path.join(args.out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for a, b in new_merges:
            f.write(f"{a} {b}\n")
    json.dump(tj, open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    dropped = [t for t in tail if t not in keep]
    dc = collections.Counter(cls(t) for t in dropped)
    print(f"[*] база: {len(alphabet)}, хвост: {len(tail)}")
    print(f"[*] seed (чистые ru, count>={args.min_count}): {len(seed)}")
    print(f"[+] оставлено: {len(keep)} из {len(tail)}; выброшено {len(dropped)}: {dict(dc)}")
    print(f"[+] мерджей: {len(merges)} -> {len(new_merges)}; словарь: {len(new_vocab)}")
    top = sorted(((usage.get(vocab[t], 0), t) for t in dropped), reverse=True)
    print("[*] крупнейшие выброшенные:", [(cls(t), c) for c, t in top[:5]])
    print(f"[*] новый --n для сайдкара: {len(new_vocab) - BASE}")
    print(f"[+] готово: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
