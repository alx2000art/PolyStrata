#!/usr/bin/env python3
"""Петли в сохранённых ответах сьюта: слов, доля уникальных и самый частый повтор окна из 12 слов.

Читает JSON-файлы ячеек bench_sweep (там лежат тексты ответов) и печатает по промптам строку:
нормальный текст даёт повтор окна 1, крутящийся — 3 и больше.  Заодно средний декод и префилл.

    python3 tools/loop_score.py lookup-ab-0/80-eta-iq256off.json lookup-ab-3/80-eta-iq256off.json
"""
import json
import re
import sys
from collections import Counter, defaultdict


def score(text):
    words = re.findall(r"[\w-]+", (text or "").lower())
    if not words:
        return 0, 0.0, 0
    win = 12
    rep = 0
    if len(words) >= win:
        rep = max(Counter(tuple(words[i:i + win]) for i in range(len(words) - win + 1)).values())
    return len(words), len(set(words)) / len(words), rep


def main(paths):
    if not paths:
        print(__doc__)
        return 2
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        runs = doc.get("runs") or []
        by = defaultdict(list)
        for r in runs:
            w, u, rep = score(r.get("text"))
            by[r.get("id") or "?"].append({"w": w, "u": u, "rep": rep,
                                           "d": r.get("decode_tps") or 0.0,
                                           "p": r.get("prefill_tps") or 0.0})
        print(f"\n=== {path} ===")
        print("| промпт | слов | uniq | повтор окна (худший) | петель | декод tok/s |")
        print("|---|---:|---:|---:|---:|---:|")
        total = 0
        for name, rows in by.items():
            worst = max(x["rep"] for x in rows)
            loops = sum(1 for x in rows if x["rep"] >= 3)
            total += loops
            print(f"| {name} | {int(sum(x['w'] for x in rows) / len(rows))} | "
                  f"{sum(x['u'] for x in rows) / len(rows):.2f} | {worst} | {loops} из {len(rows)} | "
                  f"{sum(x['d'] for x in rows) / len(rows):.1f} |")
        dec = sum(x["d"] for rows in by.values() for x in rows) / max(1, len(runs))
        pre = sum(x["p"] for rows in by.values() for x in rows) / max(1, len(runs))
        print(f"петель всего: {total} | средний декод {dec:.1f} tok/s, префилл {pre:.0f} tok/s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
