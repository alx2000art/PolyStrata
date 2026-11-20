#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_pack_tokenizer.py — расширенный токенайзер в ФОРМАТЕ ПАКА (движковый).

Пак (vocab.json/merges.txt/token_type.json/tokenizer.json) пишется
strata_tokenizer.extract() из GGUF; формат воспроизводим.  Здесь: базовые токены
и их id НЕ двигаются (строки эмбеддингов в паре с ними), новые токены из
обученного BPE добавляются в хвост, их строки нормализуются в конвенцию пака
(Ġ↔▁ и байт-unicode совпадают у HF tokenizers и llama.cpp).

    python3 tools/make_pack_tokenizer.py \
        --base-dir ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer \
        --new ~/tokenizer_ru \
        --out ~/tokenizer_ru_pack
"""
import argparse
import json
import os


def load_pack(d):
    vocab = json.load(open(os.path.join(d, "vocab.json"), encoding="utf-8"))
    merges = []
    with open(os.path.join(d, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append(tuple(parts))
    types = json.load(open(os.path.join(d, "token_type.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(d, "token_type.json")) else None
    cfg = json.load(open(os.path.join(d, "tokenizer.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(d, "tokenizer.json")) else {}
    return vocab, merges, types, cfg


def convention(vocab):
    """Пак бывает в Ġ- или ▁-конвенции пробела; считаем по обоим."""
    g = sum("Ġ" in k for k in vocab)
    u = sum("▁" in k for k in vocab)
    return "g" if g >= u else "u"


def to_canon(s, conv):
    """Канон = строка с пробелом вместо маркера (для сравнения между конвенциями)."""
    return s.replace("Ġ", " ") if conv == "g" else s.replace("▁", " ")


def from_canon(s, conv):
    return s.replace(" ", "Ġ") if conv == "g" else s.replace(" ", "▁")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="каталог токенайзера пака")
    ap.add_argument("--new", required=True, help="обученный HF-токенайзер (vocab.json/merges.txt)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    b_vocab, b_merges, b_types, b_cfg = load_pack(args.base_dir)
    b_conv = convention(b_vocab)
    print(f"[*] пак: vocab {len(b_vocab)}, merges {len(b_merges)}, конвенция '{b_conv}', "
          f"types {'есть' if b_types else 'нет'}")

    n_vocab = json.load(open(os.path.join(args.new, "vocab.json"), encoding="utf-8"))
    n_conv = convention(n_vocab)
    n_merges = []
    with open(os.path.join(args.new, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                p = line.split(" ")
                if len(p) == 2:
                    n_merges.append(tuple(p))
    print(f"[*] новый: vocab {len(n_vocab)}, merges {len(n_merges)}, конвенция '{n_conv}'")

    # канонический набор базовых токенов: что уже есть (перенос сверху) и что ново
    base_canon = {to_canon(t, b_conv) for t in b_vocab}
    out_vocab = dict(b_vocab)                        # id базовых не двигаются
    added = 0
    for t, _id in sorted(n_vocab.items(), key=lambda kv: kv[1]):
        c = to_canon(t, n_conv)
        if c in base_canon:
            continue
        out_vocab[from_canon(c, b_conv)] = len(out_vocab)
        added += 1
    print(f"[*] добавлено новых токенов: {added} (итого {len(out_vocab)})")

    base_merges = {tuple(m) for m in b_merges}
    out_merges = list(b_merges)
    added_m = 0
    for m in n_merges:
        cm = (to_canon(m[0], n_conv), to_canon(m[1], n_conv))
        # переводим правило в конвенцию пака; нужно, чтобы все три части были в вокабе
        mm = (from_canon(cm[0], b_conv), from_canon(cm[1], b_conv))
        j = from_canon(cm[0] + cm[1], b_conv)
        if mm in base_merges or len(cm[0]) == 0 or len(cm[1]) == 0:
            continue
        if mm[0] in out_vocab and mm[1] in out_vocab and j in out_vocab:
            out_merges.append(mm)
            added_m += 1
    print(f"[*] добавлено новых мерджей: {added_m} (итого {len(out_merges)})")

    os.makedirs(args.out, exist_ok=True)
    json.dump(out_vocab, open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8"),
              ensure_ascii=False)
    with open(os.path.join(args.out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{a} {b}" for a, b in out_merges))
    if b_types:
        types = list(b_types) + [1] * added         # 1 = обычный токен (как у пака)
        json.dump(types, open(os.path.join(args.out, "token_type.json"), "w", encoding="utf-8"))
    cfg = dict(b_cfg)
    cfg["vocab_size"] = len(out_vocab)
    cfg["n_merges"] = len(out_merges)
    json.dump(cfg, open(os.path.join(args.out, "tokenizer.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    # остальные файлы пака (chat_template.jinja и пр.) копируем как есть
    for fn in os.listdir(args.base_dir):
        if fn in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json"):
            continue
        src = os.path.join(args.base_dir, fn)
        if os.path.isfile(src):
            open(os.path.join(args.out, fn), "wb").write(open(src, "rb").read())
    print(f"[+] пак-токенайзер: {args.out}")


if __name__ == "__main__":
    main()
