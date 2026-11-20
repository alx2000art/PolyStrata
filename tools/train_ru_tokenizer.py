#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_ru_tokenizer.py — расширение текущего токенайзера новыми русскими мерджами.

База: vocab.json + merges.txt каталога пака (сырые файлы BPE — читаются напрямую).
Новые мерджи учатся на НАШЕМ миксе текстов (проза + код + диалоги) стандартным
BpeTrainer; итог — HF-tokenizer.json + сырые vocab.json/merges.txt.  Экспорт в
движковый формат — отдельный шаг (нужны образцы tokenizer.json/token_type.json пака).

    python3 tools/train_ru_tokenizer.py \
        --base-dir ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer \
        --text ~/corpus/corpus_full.txt --text ~/eagle_gen/dialogs_corpus.txt \
        --add 8000 --out ~/tokenizer_ru

    # и сразу «было/стало» на том же тексте:
    python3 tools/tokenizer_bench.py --text <файл> --whole --current <base-dir> --extra ~/tokenizer_ru
"""
import argparse
import json
import os


def load_base(base_dir):
    vocab = json.load(open(os.path.join(base_dir, "vocab.json"), encoding="utf-8"))
    merges = []
    with open(os.path.join(base_dir, "merges.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(" ")
            if len(parts) == 2:
                merges.append(tuple(parts))
    try:
        spec = json.load(open(os.path.join(base_dir, "tokenizer.json"), encoding="utf-8"))
    except Exception:
        spec = {}
    return vocab, merges, spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, help="каталог пака с vocab.json/merges.txt/tokenizer.json")
    ap.add_argument("--text", action="append", required=True, help="файл(ы) микса (можно несколько раз)")
    ap.add_argument("--add", type=int, default=8000, help="сколько новых токенов добавить")
    ap.add_argument("--out", required=True, help="каталог результата (HF-токенайзер)")
    args = ap.parse_args()

    vocab, merges, spec = load_base(args.base_dir)
    pat = spec.get("pre_pattern")
    print(f"[*] база: vocab {len(vocab)}, merges {len(merges)}, pre_pattern: {'есть' if pat else 'НЕТ'}")

    texts = []
    for p in args.text:
        t = open(p, encoding="utf-8", errors="ignore").read()
        texts.append(t)
        print(f"[*] текст: {p}: {len(t):,} символов")
    total = sum(len(t) for t in texts)
    print(f"[*] микс: {total:,} символов; цель = {len(vocab) + args.add} токенов")

    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tk = Tokenizer(models.BPE(vocab=vocab, merges=merges))
    # пайплайн пака: split по собственному pre_pattern, затем byte-level
    if pat:
        tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(pat, behavior="isolated"),
                                                     pre_tokenizers.ByteLevel(add_prefix_space=False)])
    else:
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=len(vocab) + args.add,
        show_progress=True,
        special_tokens=["<unk>", "<s>", "</s>"],
    )
    tk.train_from_iterator(texts, trainer=trainer)
    os.makedirs(args.out, exist_ok=True)
    tk.save(os.path.join(args.out, "tokenizer.json"))

    vs = tk.get_vocab()
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vs, f, ensure_ascii=False)
    with open(os.path.join(args.out, "merges.txt"), "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for m in tk.model.get_merges():
            f.write(f"{m[0]} {m[1]}\n")
    print(f"[+] готово: {args.out} ({len(vs)} токенов, +{len(vs) - len(vocab)})")
    print("[далее] экспорт в движковый формат и TokenAdapt-синтез эмбеддингов новых токенов")


if __name__ == "__main__":
    main()
