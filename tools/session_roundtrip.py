#!/usr/bin/env python3
"""SAVE / RESTORE на живом сервере: разговор уезжает в файл и возвращается после перезапуска движка.

Что проверяется (и почему так):
  1. длинный разговор C2 (T + Q1 + A1 + Q2 + A2, ~15k токенов) уходит в файл (SAVE);
  2. тот же следующий ход Q3 спрашивается вживую — это контрольный ответ;
  3. сервер выгружает модель (/unload), загружает снова (/load, свежий движок) и делает RESTORE;
  4. Q3 спрашивается снова: reused должен быть ≈ |C2| (промпт не читается), а ответ — совпасть с
     контрольным (полностью или длинным префиксом: окна спекуляции после перезапуска решают иначе,
     чем прогретая политика, поэтому поздние токены могут разойтись и без порчи состояния);
  5. всё повторяется ещё раз — два RESTORE одного файла обязаны дать побитово один ответ.

Сервер должен быть запущен с --slot-save-path DIR (или slot_save_path в конфиге), иначе шаги
SAVE/RESTORE ответят 501.  /unload и /load требуют JSON и ключ, как и остальные запросы.

    python tools/session_roundtrip.py            # ~4-6 минут (две загрузки модели)
    CFG=другой.json python tools/session_roundtrip.py
"""
import http.client
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(os.environ.get("CFG") or ROOT / "strata-dsv4-ud-iq2_xxs.json")
NAME = os.environ.get("SESSION_NAME") or "roundtrip.spcs"


def cfg_load():
    with open(CFG, encoding="utf-8-sig") as fh:
        return json.load(fh)


def host_of(cfg):
    """Сервер слушает поле host конфига (192.168.5.80), а не только localhost."""
    host = str(cfg.get("host") or "").strip()
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


class Client:
    def __init__(self, cfg):
        self.cfg = cfg
        self.host = host_of(cfg)
        self.port = int(cfg.get("port") or 1524)
        self.key = cfg.get("api_key") or ""
        self.model = cfg.get("model_name") or "x"

    def _do(self, method, path, body=None, timeout=3600):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        headers = {"Authorization": "Bearer " + self.key}
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, path, data, headers)
            r = conn.getresponse()
            raw = r.read().decode("utf-8", "replace")
        except OSError as error:
            raise SystemExit(f"нет соединения с {self.host}:{self.port} ({error})")
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = {"raw": raw[:400]}
        return r.status, doc

    def chat(self, messages, max_tokens):
        t0 = time.time()
        st, doc = self._do("POST", "/v1/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "stream": False, "chat_template_kwargs": {"enable_thinking": False},
            "thinking": {"type": "disabled"}})
        if st != 200 or doc.get("error"):
            raise SystemExit(f"запрос не прошёл ({st}): {str(doc)[:400]}")
        msg = (doc.get("choices") or [{}])[0].get("message") or {}
        text = msg.get("content") or ""
        return text.strip(), time.time() - t0

    def metrics(self):
        st, doc = self._do("GET", "/metrics", timeout=120)
        return doc if st == 200 else {}

    def last_request(self):
        """Что сервер записал о последнем запросе: prompt_total, reused, prompt_ms, ..."""
        m = self.metrics()
        reqs = m.get("requests") or []
        return reqs[0] if reqs else {}

    def slot(self, action, name=NAME):
        t0 = time.time()
        st, doc = self._do("POST", f"/slots/0?action={action}", {"filename": name})
        return st, doc, time.time() - t0

    def unload(self):
        st, doc = self._do("POST", "/unload?wait=1", {})
        if st != 200:
            raise SystemExit(f"/unload отказал ({st}): {str(doc)[:300]}")

    def load(self):
        t0 = time.time()
        st, doc = self._do("POST", "/load", {})
        if st != 200:
            raise SystemExit(f"/load отказал ({st}): {str(doc)[:300]}")
        return time.time() - t0


def long_text(lines=1200):
    """Длинный документ из разных строк: ~10-15k токенов, без повторов, детерминированный."""
    rng = random.Random(20261126)
    subjects = ["инженер", "архивист", "матрос", "ботаник", "часовщик", "переводчик", "капитан", "геолог"]
    objects = ["журнал", "компас", "гербарий", "чертёж", "словарь", "маятник", "карту", "образец"]
    verbs = ["сверял", "прятал", "рисовал", "взвешивал", "чинил", "подписывал", "сравнивал", "упаковывал"]
    places = ["в подвале", "на чердаке", "у причала", "в лаборатории", "на складе", "в кабинете", "у маяка", "в поле"]
    out = ["Журнал наблюдений станции «Северная», том три, записи по порядку."]
    for i in range(1, lines + 1):
        out.append(f"Запись {i}: {rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} "
                   f"{rng.choice(places)}; номер партии {i * 37 % 9973}, отметка {i % 24}:{i * 7 % 60:02d}.")
    return "\n".join(out)


def prefix_len(a, b):
    n = 0
    for x, y in zip(a.split(), b.split()):
        if x != y:
            break
        n += 1
    return n


def main():
    cfg = cfg_load()
    c = Client(cfg)
    print(f"конфиг: {CFG}  сервер {c.host}:{c.port}  файл сессии: {NAME}")

    doc = long_text()
    q1 = "\n\nВопрос: сколько записей в журнале и о чём запись 7? Ответь коротко."
    msgs = [{"role": "user", "content": doc + q1}]
    a1, dt = c.chat(msgs, 96)
    r = c.last_request()
    print(f"1) C1: прочитано {r.get('prompt_total')} токенов, reused {r.get('reused')}, "
          f"prompt {r.get('prompt_ms')} мс, ответ {dt:.1f} с")
    print(f"   A1: {a1[:100].replace(chr(10), ' ')}")

    msgs += [{"role": "assistant", "content": a1},
             {"role": "user", "content": "Вопрос: сколько записей с номером партии меньше ста?"}]
    a2, dt = c.chat(msgs, 96)
    r = c.last_request()
    c2 = r.get("prompt_total") or 0
    print(f"2) C2: прочитано {r.get('prompt_total')} токенов, reused {r.get('reused')}, "
          f"prompt {r.get('prompt_ms')} мс, ответ {dt:.1f} с")
    print(f"   A2: {a2[:100].replace(chr(10), ' ')}")

    st, doc_save, dt = c.slot("save")
    if st == 501:
        raise SystemExit("SAVE закрыт: запустите сервер с --slot-save-path DIR (или slot_save_path в конфиге)")
    if st != 200:
        raise SystemExit(f"SAVE отказал ({st}): {str(doc_save)[:400]}")
    print(f"3) SAVE: {doc_save.get('n_saved')} токенов, {doc_save.get('n_written', 0) / 2**20:.1f} MiB, "
          f"{doc_save.get('timings', {}).get('save_ms', 0):.0f} мс")

    msgs += [{"role": "assistant", "content": a2},
             {"role": "user", "content": "Вопрос: перечисли номера партий из записей 3, 4 и 5."}]
    a_live, dt = c.chat(msgs, 96)
    r = c.last_request()
    print(f"4) Q3 вживую: reused {r.get('reused')} из {r.get('prompt_total')}, prompt {r.get('prompt_ms')} мс, "
          f"ответ {dt:.1f} с")
    print(f"   A_live: {a_live[:100].replace(chr(10), ' ')}")

    answers = []
    for run in (1, 2):
        c.unload()
        load_s = c.load()
        st, doc_r, dt = c.slot("restore")
        if st != 200:
            raise SystemExit(f"RESTORE отказал ({st}): {str(doc_r)[:400]}")
        print(f"5.{run}) перезапуск: загрузка {load_s:.1f} с, RESTORE {doc_r.get('n_restored')} токенов, "
              f"{doc_r.get('n_read', 0) / 2**20:.1f} MiB, "
              f"{doc_r.get('timings', {}).get('restore_ms', 0) / 1000:.1f} с")
        a_rest, dt = c.chat(msgs, 96)
        r = c.last_request()
        answers.append(a_rest)
        print(f"6.{run}) Q3 после RESTORE: reused {r.get('reused')} из {r.get('prompt_total')}, "
              f"prompt {r.get('prompt_ms')} мс, ответ {dt:.1f} с")
        print(f"   A_rest: {a_rest[:100].replace(chr(10), ' ')}")

    ok = True
    used = r.get("reused") or 0
    if used < c2 - 8:
        ok = False
        print(f"ПРОВАЛ: после RESTORE reused {used}, а разговор занимает {c2}: файл не подставил состояние")
    if answers[0] != answers[1]:
        ok = False
        print("ПРОВАЛ: два RESTORE одного файла дали разные ответы — состояние восстанавливается не побитово")
    same_live = answers[0] == a_live
    pre = prefix_len(answers[0], a_live)
    print(f"\nответы RESTORE против живого контроля: {'побитово равны' if same_live else f'общий префикс {pre} слов'}")
    if not same_live and pre < 20:
        ok = False
        print("ПРОВАЛ: восстановленный разговор отвечает иначе с первых слов — состояние не то")
    elif not same_live:
        print("(различие после 20+ слов: ожидаемо, если после перезапуска окна спекуляции решают иначе; "
              "на lossless-вход это не влияет)")
    print("ИТОГ:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
