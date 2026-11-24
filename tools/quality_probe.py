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

def has_any(text, *words):
    """Любое из слов (альтернативы: «париж» или «paris»), а не все сразу."""
    low = text.lower().replace("ё", "е")
    return any(re.search(rf"(?<![а-яa-z]){re.escape(w.lower().replace('ё', 'е'))}(?![а-яa-z])", low) for w in words)

# name, prompt, check, cap, critical.  Критичные — целостность входа и логика: они и решают вердикт.
# Информационные — счёт, арифметика, факт, формат: на IQ2_XXS они врут штатно, их провал сам по себе
# не значит «движок сломался», но считать их полезно.
PROBES = [
    ("echo", "Повтори ровно эту строку, ничего не добавляя и не объясняя: «небо голубое, трава зелёная»",
     lambda t: has(t, "небо голубое", "трава зелёная"), 160, True),
    ("echo2", "Повтори дословно: «Голубое небо над рекой»", lambda t: has(t, "голубое небо над рекой"), 160, True),
    ("logic", "Все коты — животные. Некоторые животные — чёрные. Следует ли из этого, что некоторые коты "
              "чёрные? Ответь «да» или «нет».", lambda t: has_word(t, "нет") and not has_word(t, "да"), 200, True),
    ("translate", "Переведи на английский: «Дом стоит у реки». Только перевод.",
     lambda t: has(t, "house") and has(t, "river"), 200, True),
    ("code", "Вот функция на Python:\n```python\ndef moving_average(xs, k):\n    out = []\n    s = 0\n"
             "    for i, x in enumerate(xs):\n        s += x\n        if i >= k:\n            s -= xs[i - k]\n"
             "        out.append(s / k)\n    return out\n```\n"
             "На первых k элементах среднее неверное. Объясни причину и покажи исправленный код.",
     lambda t: has(t, "min(") or has(t, "i + 1") or has(t, "k - 1"), 700, True),
    ("count", "Сколько букв «о» в слове «молоко»? Ответь одним числом.", lambda t: has_any(t, "3"), 200, False),
    ("count-long", "Перечисли подряд все целые числа от 1 до 80 через запятую, без пояснений и без пропусков.",
     lambda t: has_any(t, "80") and has_any(t, "79") and has_any(t, "40"), 400, True),
    ("math", "Сколько будет 17 * 23? Ответь одним числом.", lambda t: has_any(t, "391"), 200, False),
    ("fact", "Столица Франции? Ответь одним словом.", lambda t: has_any(t, "париж", "paris"), 200, False),
    ("format", "Назови химический символ золота. Ответь одним словом, без пояснений.",
     lambda t: has_any(t, "au"), 160, False),
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
        return None, f"нет соединения с {host_of(cfg)}:{port} ({error})", ""
    try:
        doc = json.loads(raw)
    except ValueError:
        return None, f"не JSON: {raw[:120]}", ""
    if doc.get("error"):
        return None, f"ошибка: {str(doc['error'])[:160]}", ""
    choice = (doc.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    text = msg.get("content") or ""
    # thinking был включён (сервер не послушал): ответ ищем и в рассуждении
    if not text.strip():
        text = msg.get("reasoning_content") or msg.get("reasoning") or ""
    return text, None, choice.get("finish_reason") or ""


def loop_score(text):
    words = re.findall(r"[\w-]+", text.lower())
    if len(words) < 12:
        return len(words), 0
    rep = max(Counter(tuple(words[i:i + 12]) for i in range(len(words) - 11)).values())
    return len(words), rep


def main():
    cfg = cfg_load()
    long_run = "--long" in sys.argv
    crit = info = 0
    print(f"конфиг: {CFG}")
    args = [str(a) for a in cfg.get("args") or []]
    for key in ("--lookup", "--tail-skip", "--ctx-cap", "--prompt-cache", "--conversation-cache-slots"):
        if key in args[:-1]:
            print(f"  {key} {args[args.index(key) + 1]}", end="")
    print(f"  --no-graphs {'да' if '--no-graphs' in args else 'нет'}"
          f"  temp {((cfg.get('sampling') or {}).get('temperature'))}")
    print("\n| проверка | итог | ответ |")
    print("|---|---|---|")
    crit = info = 0
    for name, prompt, check, cap, critical in PROBES:
        text, err, finish = ask(cfg, prompt, max(cap, 160))   # thinking, если его не отключили, съедает бюджет
        if err:
            text, ok, why = "", False, err
        elif not text.strip():
            ok, why = False, "пустой ответ"
        else:
            ok, why = check(text), f"{text.strip()[:110].replace('|', '/').replace(chr(10), ' ')} [{finish}]"
        if not ok:
            if critical:
                crit += 1
            else:
                info += 1
        print(f"| {name}{'' if critical else ' (инф.)'} | {'ок' if ok else 'ПРОВАЛ'} | {why} |")
    if long_run:
        text, err, finish = ask(cfg, LONG, 900)
        if err:
            crit += 1
            print(f"| loop | ПРОВАЛ | {err} |")
        else:
            words, rep = loop_score(text)
            ok = rep < 3
            if not ok:
                crit += 1
            print(f"| loop | {'ок' if ok else 'ПРОВАЛ'} ({words} слов) | повтор окна {rep}, начало: "
                  f"{text.strip()[:70].replace('|', '/').replace(chr(10), ' ')} |")
    print(f"\nкритичных провалов: {crit} (echo/logic/translate/code/loop), информационных: {info}")
    return crit


if __name__ == "__main__":
    sys.exit(main())
