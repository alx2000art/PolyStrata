#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_prompt_list.py - список промптов для сбора дампов, по СЛОЯМ корпуса.

    python3 tools/make_prompt_list.py ~/corpus/corpus_clean.txt ~/eagle_prompts.list 400

Зачем: случайные 4000-символьные куски по всему файлу повторяют его пропорции, а книги
там 40% и они однородны - код и документы получают меньше состояний, чем встречаются в
работе.  Здесь слои берутся с заданными весами, каждый кусок режется по границам строк.

Формат: NUL-разделённый файл, его читает eagle_collect_decode.sh через PROMPT_LIST.
"""
import re
import sys

# веса слоёв: как в скиле dataset-prep - работа (документы, код) весит не меньше прозы
WEIGHTS = {"books": 0.35, "docs": 0.25, "docs_en": 0.12, "dialog": 0.10, "code": 0.10, "lor": 0.08}
CHUNK = 3500


def chunks(text, chunk=CHUNK):
    out, cur = [], ""
    for line in text.splitlines(True):
        if len(cur) + len(line) > chunk:
            if len(cur.strip()) > 200:
                out.append(cur)
            cur = ""
        cur += line
    if len(cur.strip()) > 200:
        out.append(cur)
    return out


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    corpus, out_path, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
    text = open(corpus, encoding="utf-8", errors="replace").read()
    parts = re.split(r"<<<DOC (.*?)>>>", text)
    docs = list(zip(parts[1::2], parts[2::2]))
    layer = {}
    for name, body in docs:
        layer.setdefault(name.split(":", 1)[0], []).append(body)

    # сколько кусков какому слою: вес x всего, но не больше, чем слой может дать
    picked, used = [], {}
    for tag, w in WEIGHTS.items():
        if tag not in layer:
            continue
        pool = chunks("".join(layer[tag]))
        want = max(1, int(round(total * w)))
        take = pool[:want] if len(pool) >= want else pool
        used[tag] = len(take)
        picked += take
    # добор до total из общего пула (если какой-то слой не дал своего)
    if len(picked) < total:
        extra = [c for tag in layer for c in chunks("".join(layer[tag])) if c not in picked]
        picked += extra[: total - len(picked)]
    picked = picked[:total]
    open(out_path, "w", encoding="utf-8").write(chr(0).join(picked))
    print(f"{out_path}: {len(picked)} промптов, {sum(map(len, picked)) / 1e6:.2f} млн символов")
    for tag, n in sorted(used.items(), key=lambda kv: -kv[1]):
        print(f"  {tag:9s} {n:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
