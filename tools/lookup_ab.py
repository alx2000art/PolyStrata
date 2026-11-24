#!/usr/bin/env python3
"""lookup A/B: the same requests with the n-gram drafts on and off, and the answers scored for the loop.

Зачем: подозрение, что `--lookup 3` ломает длинные ответы.  Скрипт по очереди ставит в конфиг
`--lookup 0` и `--lookup 3`, перезапускает сервер, задаёт одни и те же два длинных запроса и считает
для каждого ответа: длину, долю уникальных слов и **самый частый повтор окна из 12 слов** — это и есть
подпись петли («абзац крутится»), которую глазами видно, а в цифрах нет.

    python3 tools/lookup_ab.py          # ~10 минут: 2 рестарта + 4 генерации
    N=5 MAXNEW=2000 python3 tools/lookup_ab.py

Конфиг возвращается в исходное состояние, сервер поднимается обратно.
"""
import http.client
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = Path(os.environ.get("CFG") or ROOT / "strata-dsv4-ud-iq2_xxs.json")
MAXNEW = int(os.environ.get("MAXNEW") or 1200)
N = int(os.environ.get("N") or 2)          # повторов на каждый запрос в каждом режиме

# Два запроса: длинное объяснение (обычный текст) и переписывание с сохранением структуры
# (много дословных повторов — там n-граммные догадки должны и работать лучше всего).
PROMPTS = [
    ("long-form", "Подробно объясни устройство газотурбинного двигателя: принцип работы, основные узлы, "
                  "ступени компрессора и турбины, ограничения по температуре. Не короче 900 слов."),
    ("rewrite", "Перепиши следующий текст, исправив орфографию и пунктуацию, сохранив все абзацы и порядок "
                "слов настолько, насколько возможно:\n\n"
                "Вчера мы долго шли по лесу и вышли к старому дому. Дом стоял на краю поля, окна были "
                "заколочены досками, а крыша просела. Мы обошли его кругом и нашли дверь с другой стороны. "
                "Дверь была приоткрыта. Внутри пахло сыростью и старой бумагой. На столе лежали письма, "
                "перевязанные лентой. Мы прочитали первое письмо и поняли, что дом принадлежал учителю, "
                "который уехал перед войной и так и не вернулся. Мы закрыли дверь и пошли обратно к дороге."),
]


def load_cfg():
    with open(CFG, encoding="utf-8-sig") as fh:
        return json.load(fh)


def save_cfg(cfg):
    with open(CFG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)


def port_of(cfg):
    return int(cfg.get("port") or 1524)


def key_of(cfg):
    return cfg.get("api_key") or ""


def stop_server(port):
    subprocess.run(["pkill", "-TERM", "-f", f"{ROOT}/serve/server.py"], check=False)
    for _ in range(90):
        try:   # the port still takes connections: the server is on its way out
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.connect()
            conn.close()
            time.sleep(1)
        except OSError:
            break
    subprocess.run(["pkill", "-KILL", "-f", f"{ROOT}/serve/server.py"], check=False)
    subprocess.run(["pkill", "-KILL", "-f", "strata-poly"], check=False)
    for _ in range(30):
        if not subprocess.run(["pgrep", "-f", "strata-poly"], capture_output=True).stdout:
            break
        time.sleep(1)
    time.sleep(2)


def start_server(cfg, log):
    py = ROOT / ".venv/bin/python"
    out = open(log, "w", encoding="utf-8")
    subprocess.Popen([str(py), str(ROOT / "serve/server.py"), "--engine", "strata",
                      "--config", str(CFG), "--port", str(port_of(cfg))],
                     cwd=ROOT, stdout=out, stderr=subprocess.STDOUT, start_new_session=True,
                     env={**os.environ})
    deadline = time.time() + 900
    while time.time() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port_of(cfg), timeout=5)
            conn.request("GET", "/health", headers={"Authorization": "Bearer " + key_of(cfg)})
            body = conn.getresponse().read().decode("utf-8", "replace")
            if '"loaded"' in body and "true" in body:
                return True
        except OSError:
            pass
        time.sleep(5)
    return False


def ask(cfg, prompt):
    body = json.dumps({
        "model": cfg.get("model_name") or "x",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float((cfg.get("sampling") or {}).get("temperature", 0.7)),
        "max_tokens": MAXNEW,
        "stream": False,
    }).encode()
    conn = http.client.HTTPConnection("127.0.0.1", port_of(cfg), timeout=1800)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", body,
                 {"Authorization": "Bearer " + key_of(cfg), "Content-Type": "application/json"})
    raw = conn.getresponse().read().decode("utf-8", "replace")
    dt = time.time() - t0
    try:
        resp = json.loads(raw)
    except ValueError:
        print(f"    !!! ответ не JSON: {raw[:200]}")
        return "", 0, dt, "bad-json"
    if resp.get("error"):
        print(f"    !!! ошибка сервера: {str(resp['error'])[:200]}")
        return "", 0, dt, "error"
    choice = (resp.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    toks = (resp.get("usage") or {}).get("completion_tokens") or 0
    return text, toks, dt, choice.get("finish_reason") or ""


def score(text):
    """Длина, доля уникальных слов и самый частый повтор окна из 12 слов — подпись петли."""
    words = re.findall(r"[\w-]+", text.lower())
    if not words:
        return {"words": 0, "uniq": 0.0, "rep": 0, "eos_text": False}
    win = 12
    if len(words) >= win:
        seen = Counter(tuple(words[i:i + win]) for i in range(len(words) - win + 1))
        rep = max(seen.values())
    else:
        rep = 0
    return {"words": len(words), "uniq": round(len(set(words)) / len(words), 2), "rep": rep,
            "eos_text": "<|endoftext|>" in text or "<|im_end|>" in text}


def set_lookup(cfg, value):
    args = [str(a) for a in cfg.get("args") or []]
    if "--lookup" in args[:-1]:
        args[args.index("--lookup") + 1] = str(value)
    else:
        args += ["--lookup", str(value)]
    cfg["args"] = args
    return cfg


def main():
    cfg0 = load_cfg()
    save_cfg(cfg0)   # нормализованный вид для бэкапа ниже
    backup = CFG.with_suffix(".json.lookup-ab-backup")
    backup.write_bytes(CFG.read_bytes())
    results = {}
    try:
        for mode in (0, 3):
            cfg = load_cfg()
            cfg = set_lookup(cfg, mode)
            save_cfg(cfg)
            print(f"\n=== lookup {mode}: перезапуск ===", flush=True)
            stop_server(port_of(cfg))
            if not start_server(cfg, ROOT / f".lookup-ab-{mode}.log"):
                print(f"!!! lookup {mode}: сервер не поднялся, смотри .lookup-ab-{mode}.log")
                return 2
            print(f"    сервер готов", flush=True)
            rows = []
            for name, prompt in PROMPTS:
                for i in range(N):
                    text, toks, dt, finish = ask(cfg, prompt)
                    s = score(text)
                    rows.append({"prompt": name, "run": i, **s, "tps": round(toks / dt, 1) if dt else 0,
                                 "finish": finish})
                    print(f"    {name}#{i}: {s['words']} слов, uniq {s['uniq']}, повтор окна {s['rep']}, "
                          f"{rows[-1]['tps']} tok/s, finish={finish}", flush=True)
            results[mode] = rows
    finally:
        stop_server(port_of(load_cfg()))
        CFG.write_bytes(backup.read_bytes())
        backup.unlink(missing_ok=True)
        cfg = load_cfg()
        start_server(cfg, ROOT / ".lookup-ab-restored.log")
        print("\nконфиг восстановлен, сервер поднят заново", flush=True)

    print("\n| lookup | запрос | слов | uniq | повтор окна | tok/s | finish |")
    print("|---|---|---:|---:|---:|---:|---|")
    for mode, rows in results.items():
        for r in rows:
            print(f"| {mode} | {r['prompt']}#{r['run']} | {r['words']} | {r['uniq']} | {r['rep']} | "
                  f"{r['tps']} | {r['finish']} |")
    loops = {m: sum(1 for r in rows if r["rep"] >= 3) for m, rows in results.items()}
    speeds = {m: (sum(r["tps"] for r in rows) / len(rows)) for m, rows in results.items()}
    print(f"\nпетли (окно повторено ≥3 раз): lookup 0 — {loops.get(0, 0)}, lookup 3 — {loops.get(3, 0)}")
    print(f"средний декод: lookup 0 — {speeds.get(0, 0):.1f} tok/s, lookup 3 — {speeds.get(3, 0):.1f} tok/s")
    if loops.get(3, 0) > loops.get(0, 0):
        print("вердикт: lookup ломает ответы — выключать")
    elif speeds.get(3, 0) > speeds.get(0, 0) * 1.03 and loops.get(3, 0) <= loops.get(0, 0):
        print("вердикт: lookup чище/быстрее — оставлять")
    else:
        print("вердикт: разницы нет (в пределах шума) — держать выключенным, он ничего не даёт")
    return 0


if __name__ == "__main__":
    sys.exit(main())
