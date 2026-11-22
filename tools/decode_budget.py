#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""decode_budget.py - the steady-state per-token budget, asserted.

A throughput number is a mean over a window that includes the prefill; a regression that only shows up
in the decode loop (an extra launch per layer, a dequant pass that came back) hides inside it.  This
measures the two requests that isolate it:

    t(1 token)  = prefill + one decode step
    t(N tokens) = prefill + N decode steps
    ms/token    = (t(N) - t(1)) / (N - 1)

and fails when ms/token is over the budget (or the prefill rate is under its floor).  The prompt is
kept short on purpose: the number describes the decode loop, not the prompt cache.

    python3 tools/decode_budget.py                        # defaults
    FLOOR_TPS=15 MS_TOKEN=60 python3 tools/decode_budget.py
    python3 tools/decode_budget.py --verbose              # also prints each request

Exit status: 0 only when both budgets hold.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("CONFIG") or Path.home() / "PolyStrata/strata-dsv4-ud-iq2_xxs.json")
N = int(os.environ.get("BUDGET_TOKENS") or 48)          # the long request's generated tokens
MS_TOKEN = float(os.environ.get("MS_TOKEN") or 60.0)    # the decode budget, ms/token
FLOOR_PREFILL = float(os.environ.get("FLOOR_PREFILL_TPS") or 5.0)   # tok/s on the short prompt
TIMEOUT = float(os.environ.get("BUDGET_TIMEOUT") or 1800)
VERBOSE = os.environ.get("BUDGET_VERBOSE") == "1"
PROMPT = (os.environ.get("BUDGET_PROMPT")
          or "Напиши короткий, но осмысленный ответ на русском языке о том, как устроен кэш "
             "внимания в больших языковых моделях и почему он ограничен по памяти.")


def ask(base: str, key: str, model: str, prompt: str, max_tokens: int) -> tuple[dict, float, int, int]:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        out = json.load(r)
    took = time.perf_counter() - t0
    seen = int((out.get("usage") or {}).get("prompt_tokens") or 0)
    made = int((out.get("usage") or {}).get("completion_tokens") or 0)
    return out, took, made, seen


def main() -> int:
    cfg = json.load(open(CONFIG, encoding="utf-8-sig"))
    host = cfg.get("host") or "127.0.0.1"
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base = f"http://{host}:{int(cfg.get('port') or 8080)}/v1"
    key = cfg.get("api_key") or os.environ.get("API_KEY") or ""
    model = cfg.get("model_name") or "my-model"

    print(f"decode_budget: {base} model={model} N={N} бюджет {MS_TOKEN:.0f} мс/токен, "
          f"prefill-пол {FLOOR_PREFILL:.0f} tok/s")
    rows = []
    for label, want in (("t1", 1), (f"t{N}", N)):
        try:
            out, took, made, seen = ask(base, key, model, PROMPT, want)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"FAIL  {label}: запрос не прошёл ({e})")
            return 1
        rows.append((label, took, made, seen))
        if VERBOSE:
            print(f"  {label}: {took:.2f} c, prompt={seen}, сгенерировано {made}")
    (l1, t1, m1, p1), (l2, t2, m2, p2) = rows
    made = max(m2 - m1, 1)
    ms = (t2 - t1) / made * 1000.0
    prefill = p1 / t1 if t1 > 0 and p1 else 0.0
    tps = 1000.0 / ms if ms > 0 else 0.0
    print(f"decode: {ms:.1f} мс/токен = {tps:.1f} tok/s (бюджет {MS_TOKEN:.0f} мс, "
          f"{1000 / MS_TOKEN:.1f} tok/s) | prefill {prefill:.1f} tok/s (пол {FLOOR_PREFILL:.0f})")
    bad = []
    if ms > MS_TOKEN:
        bad.append(f"decode {ms:.1f} мс/токен > бюджета {MS_TOKEN:.0f}")
    if prefill and prefill < FLOOR_PREFILL:
        bad.append(f"prefill {prefill:.1f} tok/s < пола {FLOOR_PREFILL:.0f}")
    if bad:
        print("FAIL  " + "; ".join(bad))
        return 1
    print("PASS  decode и prefill в бюджете")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())