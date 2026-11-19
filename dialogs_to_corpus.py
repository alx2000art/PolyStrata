#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dialogs_to_corpus.py - dialogs.jsonl (лог прокси serve_dialog_logger) -> плоский текст
для read-дампа.  Делает три вещи:

1. склеивает цепочки: один диалог агентской сессии сохраняется прокси много раз с растущим
   префиксом - в корпус идёт только САМОЕ ДЛИННОЕ состояние каждой цепочки (по общему первому
   сообщению пользователя);
2. маскирует секреты (hf_*/sk-*/ghp_*/JWT/bearer/api_key=...) - после обучения модель не должна
   уметь отвечать настоящими ключами;
3. размечает роли [SYSTEM]/[USER]/[ASSISTANT]/[TOOL] - текст читается движком как обычный корпус.

    python3 dialogs_to_corpus.py ~/eagle_gen/dialogs.jsonl ~/eagle_gen/dialogs_corpus.txt
"""
import hashlib
import json
import re
import sys

SECRET = re.compile(
    r"(hf_[A-Za-z0-9]{6,}|hf_[A-Za-z0-9]{3,}\.\.\."
    r"|sk-[A-Za-z0-9_\-]{6,}"
    r"|ghp_[A-Za-z0-9]{6,}"
    r"|github_pat_[A-Za-z0-9_]{10,}"
    r"|xox[baprs]-[A-Za-z0-9\-]{10,}"
    r"|eyJ[A-Za-z0-9._\-]{20,}"
    r"|Bearer\s+[A-Za-z0-9._\-]{10,}"
    r"|(api[_-]?key|token|password|secret)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9._\-]{8,})",
    re.I)

TAG = {"system": "SYSTEM", "user": "USER", "assistant": "ASSISTANT", "tool": "TOOL"}


def scrub(s: str) -> str:
    return SECRET.sub(lambda m: m.group(0)[:5] + "***СКРЫТО***", s)


def content_of(m) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = " ".join(str(x.get("text") or x.get("content") or "") for x in c if isinstance(x, dict))
    return (c or "").strip()


def main(src: str, dst: str) -> None:
    recs = [json.loads(l) for l in open(src, encoding="utf-8")]
    # цепочка = общий хеш первого сообщения пользователя; храним самое полное состояние
    chains = {}
    order = []
    for r in recs:
        msgs = r.get("messages") or []
        if not msgs or not (r.get("answer") or "").strip():
            continue
        key = hashlib.md5(content_of(msgs[0]).encode("utf-8")).hexdigest()[:12]
        if key not in chains:
            order.append(key)
        chains[key] = r
    out = []
    for key in order:
        r = chains[key]
        buf = []
        for m in (r.get("messages") or []):
            c = content_of(m)
            if not c:
                continue
            buf.append(f"[{TAG.get(m['role'], m['role'].upper())}]\n{scrub(c)}")
        buf.append("[ASSISTANT]\n" + scrub(r.get("answer") or ""))
        out.append("\n\n".join(buf))
    text = "\n\n=== ДИАЛОГ ===\n\n".join(out)
    open(dst, "w", encoding="utf-8").write(text)
    print(f"записей {len(recs)} -> диалогов {len(out)}; текст {len(text)} симв -> {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
