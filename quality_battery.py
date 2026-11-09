#!/usr/bin/env python3
# quality_battery.py - an objective quality meter for "does the model feel dumber": ten prompts whose
# answers are checkable without taste - arithmetic, counting, logic, knowledge, code.  Run once with the
# setting under test ON, once with it OFF, and the two files line up for comparison (same prompts, same
# order, temperature 0, thinking off).  The tails of the output carry a self-check section with the
# expected answers where there is one.
#
# usage: quality_battery.py <label> [config]
#   label   what this run is: "pool6144", "plain", ...
#   config  the engine config JSON (default: strata-dsv4-ud-iq2_xxs.json) - takes host, port, api_key,
#           model_name from it, like the sweep does
import json
import os
import sys
import urllib.request

PROMPTS = [
    ("arith", "Вычисли: 17 * 23. Ответ дай одним числом."),
    ("count", "Сколько букв 'о' в слове 'молоко'? Ответ дай одним числом."),
    ("logic", "У Маши было 5 яблок, она отдала 2 и купила ещё 7. Сколько яблок у Маши? Ответ дай одним числом."),
    ("seq", "Продолжи последовательность: 2, 4, 8, 16, 32, ... Какое следующее число и по какому правилу?"),
    ("know", "Какая планета Солнечной системы самая большая? Ответь одним словом."),
    ("trap", "Что тяжелее: килограмм пуха или килограмм железа? Ответь одним предложением."),
    ("capit", "Столица Австралии? Ответь одним словом."),
    ("math", "Реши уравнение: 3x + 9 = 30. Чему равен x? Ответ дай одним числом."),
    ("code", "Напиши функцию на Python, которая переворачивает строку (только код)."),
    ("free", "Кратко объясни, почему небо днём голубое (2-3 предложения)."),
]

EXPECT = {
    "arith": "391",
    "count": "3",
    "logic": "10",
    "seq": "64",
    "know": "юпитер",
    "math": "7",
    "capit": "канберра",
}


def ask(base, key, model, prompt):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 600,
        "reasoning_budget_tokens": 0,   # the battery wants the answer, not the thinking
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(base + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode("utf-8"))
    text = d["choices"][0]["message"].get("content")
    if not text:
        text = "(no content: the reply's thinking ate the budget)"
    return text


def main():
    if len(sys.argv) < 2:
        print("usage: quality_battery.py <label> [config]")
        return 1
    label = sys.argv[1]
    cfg = sys.argv[2] if len(sys.argv) > 2 else "strata-dsv4-ud-iq2_xxs.json"
    c = json.load(open(cfg, encoding="utf-8-sig"))
    host = "127.0.0.1" if c.get("host") in ("0.0.0.0", "::") else c.get("host", "127.0.0.1")
    base = f"http://{host}:{c.get('port', 8080)}/v1"
    key, model = c.get("api_key", ""), c.get("model_name", "my-model")
    out = []
    for pid, prompt in PROMPTS:
        text = ask(base, key, model, prompt)
        out.append((pid, prompt, text))
        print(f"[{pid}] ok ({len(text)} chars)")
    path = f"quality_{label}.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# quality battery, label: {label}, model: {model}\n\n")
        for pid, prompt, text in out:
            f.write(f"===== {pid} =====\nQ: {prompt}\nA: {text}\n\n")
        f.write("===== SELF-CHECK (expected answers) =====\n")
        for pid, prompt, text in out:
            exp = EXPECT.get(pid)
            if exp:
                hit = exp.lower() in text.lower()
                f.write(f"{pid}: {'OK' if hit else 'WRONG'} (expected: {exp})\n")
    print(f"saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
