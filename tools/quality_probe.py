#!/usr/bin/env python3
"""Проверки качества и логики на уже запущенном сервере — минута вместо сорока.

Свип мерит скорость, детерминизм и совпадение с базой, но не «правильный ли ответ».  Здесь наоборот:
короткие однозначные задания с автоматической проверкой, всё на greedy (temperature 0), поэтому
результат воспроизводим, а не «в среднем по 40 минутам».

    python3 tools/quality_probe.py            # все проверки, ~1 минута
    python3 tools/quality_probe.py --long     # плюс длинный ответ с подсчётом петель

Смысл проверок:
  echo      — повторить строку дословно: ловит порчу/потерю токенов промпта (наши «неболюбие»);
  count     — посчитать буквы: мелкая арифметика над строкой, где квант обычно врёт;
  math      — арифметика;
  logic     — логический вывод, где верный ответ «нет»;
  fact      — столица (проверка, что модель вообще помнит факты);
  translate — перевод, проверка по ключевым словам;
  format    — ответ одним словом;
  code      — найти и исправить ошибку в функции;
  loop      — длинный ответ, повтор окна из 12 слов (петля: 3 и больше).

Код возврата — число провалов, так что годится и для «до/после» одной правки конфига.
"""
import http.client
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(os.environ.get("CFG") or ROOT / "strata-dsv4-ud-iq2_xxs.json")

def cfg_load():
    with open(CFG, encoding="utf-8-sig") as fh:
        return json.load(fh)

def has(text, *needles):
    low = text.lower().replace("ё", "е")
    return all(n.lower().replace("ё", "е") in low for n in needles)

def has_word(text, *words):
    low = text.lower().replace("ё", "е")
    return all(re.search(rf"(?<![а-яa-z]){re.escape(w.lower().replace('ё', 'е'))}(?![а-яa-z])", low) for w in words)

PROBES = [
    ("echo", "Повтори ровно эту строку, ничего не добавляя и не объясняя: «небо голубое, трава зелёная»",
     lambda t: has(t, "небо голубое", "трава зелёная"), 80),
    ("count", "Сколько букв «о» в слове «молоко»? Ответь одним числом.", lambda t: has_word(t, "3"), 60),
    ("math", "Сколько будет 17 * 23? Ответь одним числом.", lambda t: has_word(t, "391"), 60),
    ("logic", "Все коты — животные. Некоторые животные — чёрные. Следует ли из этого, что некоторые коты "
              "чёрные? Ответь «да» или «нет».", lambda t: has_word(t, "нет") and not has_word(t, "да"), 60),
    ("fact", "Столица Франции? Ответь одним словом.", lambda t: has_word(t, "париж", "paris"), 40),
    ("translate", "Переведи на английский: «Дом стоит у реки». Только перевод.",
     lambda t: has(t, "house") and has(t, "river"), 80),
    ("format", "Назови химический символ золота. Ответь одним словом, без пояснений.",
     lambda t: has_word(t, "au") or has(t, "au "), 40),
    ("code", "Вот функция на Python:\n```python\ndef moving_average(xs, k):\n    out = []\n    s = 0\n"
             "    for i, x in enumerate(xs):\n        s += x\n        if i >= k:\n            s -= xs[i - k]\n"
             "        out.append(s / k)\n    return out\n```\n"
             "На первых k элементах среднее неверное. Объясни причину и покажи исправленный код.",
     lambda t: has(t, "min(") or has(t, "i + 1") or has(t, "k - 1"), 500),
]

LONG = ("Продолжи рассказ не менее 600 слов, в том же стиле:\n\n"
        "Дождь начался ещё до рассвета, и к утру дорога превратилась в полосу глины. Иван Николаевич "
        "стоял на крыльце станции и смотрел, как поезд уходит без него.")


def host_of(cfg):
    """Сервер слушает поле host конфига (192.168.5.80), а не только localhost."""
    host = str(cfg.get("host") or "").strip()
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


def ask(cfg, prompt, max_tokens):
    port = int(cfg.get("port") or 1524)
    body = json.dumps({"model": cfg.get("model_name") or "x",
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens, "stream": False,
                       # без рассуждений: иначе на коротком max_tokens весь бюджет уходит в <think>,
                       # а content приходит пустым (так же их отключает ab_patches.py)
                       "chat_template_kwargs": {"enable_thinking": False},
                       "thinking": {"type": "disabled"}}).encode()
    try:
        conn = http.client.HTTPConnection(host_of(cfg), port, timeout=600)
        conn.request("POST", "/v1/chat/completions", body,
                     {"Authorization": "Bearer " + (cfg.get("api_key") or ""),
                      "Content-Type": "application/json"})
        raw = conn.getresponse().read().decode("utf-8", "replace")
    except OSError as error:
        return None, f"нет соединения с {host_of(cfg)}:{port} ({error})"
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, f"не JSON: {raw[:120]}"
    if doc.get("error"):
        return None, f"ошибка: {str(doc['error'])[:160]}"
    msg = (doc.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    # thinking был включён (сервер не послушал): ответ ищем и в рассуждении
    if not text.strip():
        text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return text, None


def loop_score(text):
    words = re.findall(r"[\w-]+", text.lower())
    if len(words) < 12:
        return len(words), 0
    rep = max(Counter(tuple(words[i:i + 12]) for i in range(len(words) - 11)).values())
    return len(words), rep


def main():
    cfg = cfg_load()
    long_run = "--long" in sys.argv
    fails = 0
    print(f"конфиг: {CFG}")
    args = [str(a) for a in cfg.get("args") or []]
    for key in ("--lookup", "--tail-skip", "--ctx-cap", "--prompt-cache", "--conversation-cache-slots"):
        if key in args[:-1]:
            print(f"  {key} {args[args.index(key) + 1]}", end="")
    print(f"  --no-graphs {'да' if '--no-graphs' in args else 'нет'}"
          f"  temp {((cfg.get('sampling') or {}).get('temperature'))}")
    print("\n| проверка | итог | ответ |")
    print("|---|---|---|")
    for name, prompt, check, cap in PROBES:
        text, err = ask(cfg, prompt, max(cap, 160))   # thinking, если сервер его не отключил, съедает бюджет
        if err:
            fails += 1
            print(f"| {name} | ОШИБКА | {err} |")
            continue
        if not text.strip():
            fails += 1
            print(f"| {name} | ПУСТО | ответа нет: весь бюджет ушёл в рассуждение? |")
            continue
        ok = check(text)
        if not ok:
            fails += 1
        tail = "" if ok else " "
        print(f"| {name} | {'ок' if ok else 'ПРОВАЛ'} |{tail}{text.strip()[:90].replace('|', '/').replace(chr(10), ' ')} |")
    if long_run:
        text, err = ask(cfg, LONG, 900)
        if err:
            fails += 1
            print(f"| loop | ОШИБКА | {err} |")
        else:
            words, rep = loop_score(text)
            ok = rep < 3
            if not ok:
                fails += 1
            print(f"| loop | {'ок' if ok else 'ПРОВАЛ'} ({words} слов) | повтор окна {rep}, начало: "
                  f"{text.strip()[:70].replace('|', '/').replace(chr(10), ' ')} |")
    print(f"\nпровалов: {fails}")
    return fails


if __name__ == "__main__":
    sys.exit(main())
