#!/usr/bin/env python3
"""Проверка дискового кэша разговоров (--conversation-store) на живом сервере, HTTP-сценарием.

Сценарий (каждый шаг — то, что делает живой клиент):
  1. Чат A: длинный документ (~6k токенов) + вопрос -> ответ A1 (холодный префилл).
  2. Чат B: другой документ + вопрос -> ответ B1.  При этой смене чата A должен уйти в файл.
  3. Снова A: тот же разговор + новый вопрос -> ответ A2.  Это смена чата обратно: сервер обязан
     поднять A из файла, и `reused` будет ~весь разговор A, а prompt_ms — доли холодного.
  4. Чат C: новый документ + ответ.  Пауза > idle: файл C должен появиться без всякой смены чата.
  5. Ещё один ход C (чтобы состояние стало новее файла) — на остановке сервера в свипе он должен
     лечь в файл ещё раз.

Проверки: файлы появились и они не пустые; в логе сервера есть строки `-> conv-…` (сейв) и
`back from conv-…` (restore); `reused` шага 3 ~ длина разговора A; prompt_ms шага 3 много меньше
холодного; после паузы файл C новее, чем был до неё.

    python3 tools/session_store_check.py --base http://127.0.0.1:8095 --key K \
        --store ~/PolyStrata/bench_sweep/store-check --log ~/PolyStrata/bench_sweep/server.log.last --idle 3
"""
import argparse
import glob
import http.client
import json
import os
import random
import re
import sys
import time
import urllib.parse

SAVE_LINE = re.compile(r"conversation store: (\d+) tokens, ([\d.]+) MiB -> (conv-[\w.-]+) in ([\d.]+) s")
BACK_LINE = re.compile(r"conversation store: (\d+) tokens, ([\d.]+) MiB back from (conv-[\w.-]+) in ([\d.]+) s")


class Client:
    def __init__(self, base, key, model="x"):
        u = urllib.parse.urlsplit(base if base.startswith("http") else "http://" + base)
        self.host, self.port = u.hostname or "127.0.0.1", u.port or 80
        self.key = key or ""
        self.model = model or "x"

    def _do(self, method, path, body=None, timeout=1800):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        headers = {"Authorization": "Bearer " + self.key}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        conn.request(method, path, data, headers)
        r = conn.getresponse()
        raw = r.read().decode("utf-8", "replace")
        try:
            return r.status, json.loads(raw)
        except ValueError:
            return r.status, {"raw": raw[:300]}

    def chat(self, messages, max_tokens=64):
        st, doc = self._do("POST", "/v1/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"}})
        if st != 200 or doc.get("error"):
            raise SystemExit(f"запрос не прошёл ({st}): {str(doc)[:300]}")
        msg = (doc.get("choices") or [{}])[0].get("message") or {}
        return (msg.get("content") or "").strip(), (doc.get("usage") or {})

    def last_request(self):
        st, doc = self._do("GET", "/metrics", timeout=120)
        reqs = (doc or {}).get("requests") or []
        return reqs[0] if reqs else {}


def long_text(seed, lines=700):
    """Длинный документ из разных строк (~6-7k токенов), детерминированный по seed."""
    rng = random.Random(seed)
    who = ["инженер", "архивист", "матрос", "ботаник", "часовщик", "переводчик", "капитан", "геолог"]
    what = ["журнал", "компас", "гербарий", "чертёж", "словарь", "маятник", "карту", "образец"]
    how = ["сверял", "прятал", "рисовал", "взвешивал", "чинил", "подписывал", "сравнивал", "упаковывал"]
    where = ["в подвале", "на чердаке", "у причала", "в лаборатории", "на складе", "в кабинете", "у маяка"]
    out = [f"Журнал станции, том {seed % 90 + 1}, записи по порядку."]
    for i in range(1, lines + 1):
        out.append(f"Запись {i}: {rng.choice(who)} {rng.choice(how)} {rng.choice(what)} {rng.choice(where)}; "
                   f"партия {i * 37 % 9973}, отметка {i % 24}:{i * 7 % 60:02d}.")
    return "\n".join(out)


def log_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8095")
    ap.add_argument("--key", default="")
    ap.add_argument("--model", default="x", help="model_name этого сервера (в теле запроса)")
    ap.add_argument("--store", required=True, help="каталог conversation_store (его файлы проверяются)")
    ap.add_argument("--log", required=True, help="лог сервера (строки conversation store)")
    ap.add_argument("--idle", type=float, default=3.0, help="conversation_store_idle_s этого прогона")
    a = ap.parse_args()
    cli = Client(a.base, a.key, a.model)
    fails = []

    def saves(text):
        return [m for m in SAVE_LINE.finditer(text)]

    def backs(text):
        return [m for m in BACK_LINE.finditer(text)]

    print(f"store-check: сервер {a.base}, каталог {a.store}, лог {a.log}", flush=True)

    # 1) чат A — холодный префилл
    doc_a = long_text(1)
    msgs_a = [{"role": "user", "content": doc_a + "\n\nВопрос: сколько записей в журнале? Ответь коротко."}]
    t0 = time.time()
    a1, use_a = cli.chat(msgs_a)
    r = cli.last_request()
    conv_a = int(use_a.get("prompt_tokens") or r.get("prompt_total") or 0) + int(use_a.get("completion_tokens") or 0)
    cold_ms = float(r.get("prompt_ms") or 0.0)
    print(f"1) A: {r.get('prompt_total')} токенов промпта, {cold_ms:.0f} мс, ответ за {time.time() - t0:.1f} с "
          f"(разговор A = {conv_a} токенов)", flush=True)
    before = log_text(a.log)

    # 2) чат B — при этой смене чата A уходит в файл
    doc_b = long_text(2)
    msgs_b = [{"role": "user", "content": doc_b + "\n\nВопрос: о чём запись 5? Ответь коротко."}]
    b1, _ = cli.chat(msgs_b)
    after_b = log_text(a.log)
    new_saves = saves(after_b)[len(saves(before)):]
    if new_saves:
        m = new_saves[-1]
        print(f"2) B: A сохранён — {m.group(1)} токенов, {m.group(2)} MiB, {m.group(3)}, {m.group(4)} с", flush=True)
    else:
        fails.append("смена чата A->B не сохранила A (нет строки '-> conv-' в логе)")

    # 3) назад в A: RESTORE и продолжение
    msgs_a2 = msgs_a + [{"role": "assistant", "content": a1},
                        {"role": "user", "content": "Вопрос: сколько записей с партией меньше ста?"}]
    a2, _ = cli.chat(msgs_a2)
    r = cli.last_request()
    reused = int(r.get("reused") or 0)
    warm_ms = float(r.get("prompt_ms") or 0.0)
    after_a2 = log_text(a.log)
    new_backs = backs(after_a2)[len(backs(after_b)):]
    print(f"3) назад в A: reused {reused} из {r.get('prompt_total')} (разговор A = {conv_a}), "
          f"prompt {warm_ms:.0f} мс против {cold_ms:.0f} мс холодного", flush=True)
    if new_backs:
        m = new_backs[-1]
        print(f"   RESTORE: {m.group(1)} токенов, {m.group(2)} MiB, {m.group(3)}, {m.group(4)} с", flush=True)
    else:
        fails.append("возврат в A не поднял файл (нет строки 'back from conv-' в логе)")
    if reused < conv_a - 8:
        fails.append(f"reused {reused} меньше разговора A ({conv_a}): файл не подставил состояние")
    if cold_ms > 0 and warm_ms > max(3000.0, 0.5 * cold_ms):
        fails.append(f"prompt после RESTORE {warm_ms:.0f} мс — не похоже на восстановление")

    # 4) чат C: сохранение по простою (без смены чата)
    doc_c = long_text(3)
    msgs_c = [{"role": "user", "content": doc_c + "\n\nВопрос: сколько записей в журнале? Ответь коротко."}]
    c1, _ = cli.chat(msgs_c)
    before_idle = log_text(a.log)
    print(f"4) C: ответ получен, жду {a.idle + 7:.0f} с простоя (idle {a.idle:g} с + тик 5 с) ...", flush=True)
    time.sleep(a.idle + 7.0)
    after_idle = log_text(a.log)
    idle_saves = saves(after_idle)[len(saves(before_idle)):]
    if idle_saves:
        m = idle_saves[-1]
        print(f"   простой сохранил C: {m.group(1)} токенов, {m.group(2)} MiB, {m.group(3)}, {m.group(4)} с", flush=True)
    else:
        fails.append("простой не сохранил живой чат (нет новой строки '-> conv-' после паузы)")

    # 5) ещё ход C: состояние новее файла, и на остановке сервера свип должен увидеть ещё один сейв
    msgs_c2 = msgs_c + [{"role": "assistant", "content": c1},
                        {"role": "user", "content": "Вопрос: о чём запись 9? Ответь коротко."}]
    cli.chat(msgs_c2)
    print("5) C продолжен на ход (на остановке сервера свип проверит сохранение при выходе)", flush=True)

    # файлы
    files = sorted(glob.glob(os.path.join(a.store, "conv-*.spcs")))
    total = sum(os.path.getsize(p) for p in files if os.path.isfile(p))
    print(f"файлы: {len(files)} шт, {total / 2**20:.1f} MiB всего", flush=True)
    for p in files:
        print(f"   {os.path.basename(p)}: {os.path.getsize(p) / 2**20:.1f} MiB", flush=True)
    if len(files) < 2:
        fails.append(f"в каталоге {len(files)} файлов: чаты A и C должны были сохраниться")

    print("ИТОГ:", "PASS" if not fails else "FAIL", flush=True)
    for why in fails:
        print("  -", why, flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
