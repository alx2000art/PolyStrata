#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ctx_needle.py - the long-context gate: does the model really attend across the window it reports?

A window that is *granted* is not a window that works: the engine can clamp, cap or mis-place rows and
every short prompt still answers.  This gate puts a unique service code at the very START of a prompt
that fills the requested window and asks for it at the far END, where only attention over the whole
span can find it.  A row passes only when the answer carries the code back.

    python3 tools/ctx_needle.py                     # the default rows
    ROWS="8192 32768" python3 tools/ctx_needle.py   # pick the windows
    TOKENIZER_DIR=... python3 tools/ctx_needle.py   # else the config's "tokenizer"

Exit status: 0 only if every row answers at its window.  A row that cannot be built (over the
engine's max_context) is a SKIP and counts as a FAILURE - a skip reads like a pass in a summary.

The filler is numbered line by line (never the same text twice): a repeated block lets a lazy path
copy one line instead of attending, and the needle would then "pass" without any long-range work.
"""
from __future__ import annotations

import json
import os
import random
import string
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("CONFIG") or Path.home() / "PolyStrata/strata-dsv4-ud-iq2_xxs.json")
ROWS = [int(x) for x in (os.environ.get("ROWS") or "8192 32768").split()]
MARGIN = int(os.environ.get("NEEDLE_MARGIN") or 64)     # tokens kept for the question and the answer
TIMEOUT = float(os.environ.get("NEEDLE_TIMEOUT") or 3600)
FILLER = ("Проверка длинного окна: набор предложений о модели, её кэше и декодировании, "
          "чтобы строка была длинной и не повторяла соседнюю. ")


def cfg_load(path: Path) -> dict:
    return json.load(open(path, encoding="utf-8-sig"))


def host_port(cfg: dict) -> tuple[str, int]:
    host = cfg.get("host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    return host, int(cfg.get("port") or 8080)


def tokenizer(cfg: dict):
    """The engine's own tokenizer, so the prompt is measured exactly as the server will read it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import strata_tokenizer as ST
    tp = Path(os.environ.get("TOKENIZER_DIR") or cfg["tokenizer"])
    vocab = json.loads((tp / "vocab.json").read_text())
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    return ST.Tokenizer(toks, (tp / "merges.txt").read_text().split("\n"),
                        json.loads((tp / "token_type.json").read_text()))


def build(tok, target: int, code: str) -> str:
    """A prompt of ~target tokens: the code line first, numbered filler, the question last."""
    head = f"СЛУЖЕБНЫЙ КОД: {code}\n\n"
    tail = ("Вопрос: какой служебный код был указан в самом начале этого документа? "
            "Ответь ровно кодом, без пояснений.")
    per = len(tok.encode(FILLER * 3)) / 3.0 or 1.0
    lines = max(0, int((target - len(tok.encode(head + tail)) - 8) / per))
    for _ in range(4):                                    # the template's own tokens move the count a little
        body = "".join(f"Строка {i:06d}. {FILLER}" for i in range(1, lines + 1))
        have = len(tok.encode(head + body + tail))
        if abs(have - target) <= 8:
            break
        lines = max(0, lines + int((target - have) / per))
    return head + body + tail


def ask(base: str, key: str, model: str, prompt: str, max_tokens: int) -> tuple[dict, float]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        out = json.load(r)
    return out, time.perf_counter() - t0


def main() -> int:
    cfg = cfg_load(CONFIG)
    host, port = host_port(cfg)
    base = f"http://{host}:{port}/v1"
    key = cfg.get("api_key") or os.environ.get("API_KEY") or ""
    model = cfg.get("model_name") or "my-model"
    max_ctx = 0
    args = [str(a) for a in (cfg.get("args") or [])]
    if "--max-context" in args:
        try:
            max_ctx = int(args[args.index("--max-context") + 1])
        except (IndexError, ValueError):
            max_ctx = 0
    tok = tokenizer(cfg)

    print(f"ctx_needle: {base} model={model} max_context={max_ctx or '?'} tokenizer={cfg.get('tokenizer')}")
    bad = 0
    for want in ROWS:
        room = max_ctx - MARGIN if max_ctx else want
        if want > room:
            print(f"SKIP  {want:>7}: over the engine's window ({room} usable)")
            bad += 1
            continue
        code = "ZEBRA-" + "".join(random.choice(string.hexdigits.upper()[:16]) for _ in range(6))
        prompt = build(tok, want - MARGIN, code)
        n_local = len(tok.encode(prompt))
        try:
            out, took = ask(base, key, model, prompt, 24)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"FAIL  {want:>7}: the request did not complete ({e})")
            bad += 1
            continue
        usage = out.get("usage") or {}
        seen = int(usage.get("prompt_tokens") or 0)
        answer = ""
        for ch in out.get("choices") or []:
            answer += (ch.get("message") or {}).get("content") or ""
        hit = code.lower() in answer.lower()
        prefill = seen / took if took > 0 and seen else 0.0
        verdict = "PASS" if (hit and seen >= want * 0.9) else "FAIL"
        if verdict == "FAIL":
            bad += 1
        print(f"{verdict}  {want:>7}: prompt={seen} (локально {n_local}) {took:.1f} c, "
              f"prefill {prefill:.0f} tok/s, код {'найден' if hit else 'НЕ найден'}: {answer.strip()[:60]!r}")
    print(f"ctx_needle: {len(ROWS) - bad}/{len(ROWS)} строк прошли")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())