#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tokenizer_bench.py — сколько токенов даёт токенайзер на русском тексте?

Замер до всякого TokenAdapt: текущий токенайзер против кандидатов (HF-имена или
локальные каталоги — качаются только файлы токенайзера, не модели).  Метрика —
токенов на 1000 символов реального текста: во сколько раз меньше токенов, во
столько раз дешевле контекст/кэш/префилл, и тем легче драфту.

    python3 tools/tokenizer_bench.py --text ~/corpus/corpus_full.txt \
        --current ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer

Кандидаты по умолчанию (можно перекрыть --extra "имя1 имя2"):
    Qwen/Qwen2.5-7B           — мультиязычный 151k, сильный русский
    ai-forever/ruGPT-3.5-13B  — русский BPE 58k
    ai-forever/ruT5-base      — русский SentencePiece
"""
import argparse
import random

CANDIDATES = [
    "deepseek-ai/DeepSeek-V3",   # клан текущего (вокаб 129280): числится "текущим", если каталог пака не читается
    "Qwen/Qwen2.5-7B",
    "ai-forever/ruGPT-3.5-13B",
]


def load(path):
    """HF-имя или каталог; каталог может быть и не HF-формата — пробуем raw tokenizers."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception as e:
        import os
        tj = os.path.join(path, "tokenizer.json")
        if os.path.exists(tj):
            from tokenizers import Tokenizer
            return Tokenizer.from_file(tj)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True, help="файл с русским текстом (реальный домен)")
    ap.add_argument("--current", required=True, help="текущий токенайзер (каталог/имя HF)")
    ap.add_argument("--extra", default="", help="кандидаты через пробел (HF-имена/каталоги)")
    ap.add_argument("--chunks", type=int, default=20, help="сколько кусков текста мерить")
    ap.add_argument("--chunk-chars", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    text = open(args.text, encoding="utf-8", errors="ignore").read()
    # чистим: без пустых строк и без обрывов — меряем на связном тексте
    text = "\n".join(s for s in text.splitlines() if s.strip())
    if len(text) < args.chunk_chars * 2:
        raise SystemExit(f"текст короткий: {len(text)} символов")
    rng = random.Random(args.seed)
    starts = [rng.randrange(0, len(text) - args.chunk_chars) for _ in range(args.chunks)]
    chunks = [text[s:s + args.chunk_chars] for s in starts]
    total_chars = sum(len(c) for c in chunks)
    print(f"текст: {args.text}: {len(text):,} символов, {args.chunks} кусков по {args.chunk_chars:,}")

    names = [("текущий", args.current)] + [("кандидат", n) for n in (args.extra.split() or CANDIDATES)]
    base_ratio = None
    print(f"\n{'токенайзер':<34}{'токенов':>9}{'tok/1000 симв':>15}{'байт/токен':>12}{'к текущему':>12}")
    for label, name in names:
        try:
            tk = load(name)
        except Exception as e:
            print(f"{name:<34}  НЕ ЗАГРУЗИЛСЯ: {str(e)[:90]}")
            continue
        toks = 0
        for c in chunks:
            r = tk.encode(c, add_special_tokens=False)
            toks += len(r.ids) if hasattr(r, "ids") else len(r)
        ratio = toks / total_chars * 1000.0
        if base_ratio is None:
            base_ratio = ratio
        vs = f"{ratio / base_ratio * 100 - 100:+.1f}%"
        print(f"{name:<34}{toks:>9,}{ratio:>15.1f}{total_chars / max(toks, 1):>12.1f}{vs:>12}")
        if label == "текущий":
            base_ratio = ratio
        elif base_ratio is None:
            base_ratio = ratio   # текущий не читается: база = первый успешный (DeepSeek-V3 = его клан)
    print("\n(tok/1000 симв: меньше = лучше; 'к текущему' — сколько % токенов даёт кандидат против текущего)")


if __name__ == "__main__":
    main()
