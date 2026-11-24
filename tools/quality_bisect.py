#!/usr/bin/env python3
"""Бисект качества одной командой: правит конфиг, перезапускает сервер, гоняет пробы, сравнивает.

Каждый вариант — это одна правка прод-конфига (вернуть графы, вернуть --lookup 3, вернуть штрафы
частоты, включить разговорный кэш, ...).  Скрипт по очереди применяет вариант, поднимает сервер заново
(иначе движок остаётся со старыми аргументами - на этом мы дважды обожглись), прогоняет
tools/quality_probe.py и в конце печатает матрицу: какой вариант ломает критические проверки.

    ./tools/quality_bisect.py                          # base + все варианты, ~25 мин
    VARIANTS=base,graphs,lookup3 ./tools/quality_bisect.py   # только эти, ~10 мин

Конфиг в конце возвращается в исходный вид, сервер поднимается на нём.
"""
import http.client
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))
import quality_probe as qp   # noqa: E402  - те же пробы, что в одиночном запуске

CFG = Path(os.environ.get("CFG") or ROOT / "strata-dsv4-ud-iq2_xxs.json")
LONG = "--long" in sys.argv or os.environ.get("LONG", "1") == "1"


def host_of(cfg):
    host = str(cfg.get("host") or "").strip()
    return "127.0.0.1" if host in ("", "0.0.0.0", "::") else host


def load():
    with open(CFG, encoding="utf-8-sig") as fh:
        return json.load(fh)


def save(cfg):
    with open(CFG, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------- варианты правки конфига
def set_arg(cfg, key, value):
    args = [str(a) for a in cfg.get("args") or []]
    if key in args[:-1]:
        args[args.index(key) + 1] = str(value)
    else:
        args += [key, str(value)]
    cfg["args"] = args


def del_arg(cfg, key):
    cfg["args"] = [str(a) for a in cfg.get("args") or [] if str(a) != key]


def v_base(cfg):
    # никаких следов прошлых экспериментов
    env = dict(cfg.get("env") or {})
    env.pop("STRATA_GRAPH_PARTS", None)
    cfg["env"] = env


def v_graphs(cfg):
    del_arg(cfg, "--no-graphs")


def v_nographs(cfg):
    """Эталон: графы выключены (когда в конфиге они включены)."""
    args = [str(a) for a in cfg.get("args") or []]
    if "--no-graphs" not in args:
        args.append("--no-graphs")
    cfg["args"] = args


def graph_part(letter):
    """Графы включены, записывается только эта часть (h голова, t хвост, s shared, l логиты, c маршрутизатор)."""
    def apply(cfg):
        del_arg(cfg, "--no-graphs")
        env = dict(cfg.get("env") or {})
        env["STRATA_GRAPH_PARTS"] = letter
        cfg["env"] = env
    return apply


def v_lookup3(cfg):
    set_arg(cfg, "--lookup", 3)


def v_penalties(cfg):
    s = dict(cfg.get("sampling") or {})
    s["frequency_penalty"] = 0.1
    s["presence_penalty"] = 0.05
    cfg["sampling"] = s


def v_convcache(cfg):
    set_arg(cfg, "--conversation-cache-mib", 2048)
    set_arg(cfg, "--conversation-cache-slots", 4)


def v_promptcache(cfg):
    set_arg(cfg, "--prompt-cache", 6)


def v_tail0(cfg):
    set_arg(cfg, "--tail-skip", 0)


VARIANTS = {
    "base": v_base,
    "graphs": v_graphs,
    "nographs": v_nographs,
    "g_h": graph_part("h"),   # голова слоя со вниманием
    "g_t": graph_part("t"),   # хвост слоя
    "g_s": graph_part("s"),   # shared-эксперт
    "g_l": graph_part("l"),   # логиты
    "g_c": graph_part("c"),   # маршрутизатор карты
    "lookup3": v_lookup3,
    "penalties": v_penalties,
    "convcache": v_convcache,
    "promptcache": v_promptcache,
    "tail0": v_tail0,
}


# ---------------------------------------------------------------- сервер
def alive(cfg):
    try:
        c = http.client.HTTPConnection(host_of(cfg), int(cfg.get("port") or 1524), timeout=2)
        c.connect()
        c.close()
        return True
    except OSError:
        return False


def stop_server(cfg):
    subprocess.run(["pkill", "-TERM", "-f", f"{ROOT}/serve/server.py"], check=False)
    for _ in range(90):
        if not alive(cfg):
            break
        time.sleep(1)
    subprocess.run(["pkill", "-KILL", "-f", f"{ROOT}/serve/server.py"], check=False)
    subprocess.run(["pkill", "-KILL", "-f", "strata-poly"], check=False)
    for _ in range(30):
        if not subprocess.run(["pgrep", "-f", "strata-poly"], capture_output=True).stdout:
            break
        time.sleep(1)
    time.sleep(2)


def start_server(cfg, log_path):
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([str(ROOT / ".venv/bin/python"), str(ROOT / "serve/server.py"), "--engine", "strata",
                             "--config", str(CFG), "--port", str(int(cfg.get("port") or 1524))],
                            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ})
    t0 = time.time()
    shown = 0
    for _ in range(200):          # до 10 минут, но с отчётом каждые 15 секунд
        try:
            c = http.client.HTTPConnection(host_of(cfg), int(cfg.get("port") or 1524), timeout=5)
            c.request("GET", "/health", headers={"Authorization": "Bearer " + (cfg.get("api_key") or "")})
            body = c.getresponse().read().decode("utf-8", "replace")
            if '"loaded"' in body and "true" in body:
                print(f"    сервер готов ({time.time() - t0:.0f} с)", flush=True)
                return True
        except OSError:
            pass
        if proc.poll() is not None:   # сервер вышел: не ждём впустую
            print(f"!!! сервер вышел с кодом {proc.returncode}; хвост {log_path}:", flush=True)
            try:
                lines = Path(log_path).read_text(encoding="utf-8", errors="replace").strip().splitlines()
                for line in lines[-6:]:
                    print("      " + line, flush=True)
            except OSError:
                pass
            return False
        elapsed = time.time() - t0
        if elapsed - shown >= 15:
            shown = elapsed
            print(f"    жду сервер: {elapsed:.0f} с ...", flush=True)
        time.sleep(3)
    print("!!! сервер не поднялся за 10 минут", flush=True)
    return False


# ---------------------------------------------------------------- пробы
def run_probes():
    rows = []
    for name, prompt, check, cap, critical in qp.PROBES:
        text, err, finish = qp.ask(qp.cfg_load(), prompt, max(cap, 160))
        if err:
            rows.append({"name": name, "critical": critical, "ok": False, "why": err})
        elif not text.strip():
            rows.append({"name": name, "critical": critical, "ok": False, "why": "пустой ответ"})
        else:
            rows.append({"name": name, "critical": critical, "ok": check(text),
                         "why": f"{text.strip()[:70].replace(chr(10), ' ')} [{finish}]"})
        r = rows[-1]
        print(f"      {name}: {'ок' if r['ok'] else 'ПРОВАЛ'} — {r['why'][:70]}", flush=True)
    if LONG:
        text, err, finish = qp.ask(qp.cfg_load(), qp.LONG, 900)
        if err:
            rows.append({"name": "loop", "critical": True, "ok": False, "why": err})
        else:
            words, rep = qp.loop_score(text)
            rows.append({"name": "loop", "critical": True, "ok": rep < 3,
                         "why": f"{words} слов, повтор окна {rep}"})
        r = rows[-1]
        print(f"      loop: {'ок' if r['ok'] else 'ПРОВАЛ'} — {r['why'][:70]}", flush=True)
    return rows


def main():
    names = [n.strip() for n in
             os.environ.get("VARIANTS", "nographs,graphs,g_h,g_t,g_s,g_l").split(",") if n.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        print(f"неизвестные варианты: {unknown}; есть: {list(VARIANTS)}")
        return 2
    pristine = load()
    backup = CFG.with_suffix(".json.bisect-backup")
    backup.write_text(json.dumps(pristine, ensure_ascii=False, indent=1), encoding="utf-8")
    results = {}
    try:
        for name in names:
            cfg = json.loads(json.dumps(pristine))   # глубокая копия каждого варианта от исходного
            VARIANTS[name](cfg)
            save(cfg)
            print(f"\n=== вариант {name}: рестарт ===", flush=True)
            stop_server(cfg)
            print("    сервер остановлен, поднимаю ...", flush=True)
            if not start_server(cfg, ROOT / f".bisect-{name}.log"):
                print(f"!!! {name}: сервер не поднялся (.bisect-{name}.log)", flush=True)
                results[name] = None
                continue
            print("    пробы:", flush=True)
            rows = run_probes()
            results[name] = rows
            crit = sum(1 for r in rows if r["critical"] and not r["ok"])
            info = sum(1 for r in rows if not r["critical"] and not r["ok"])
            print(f"    {name}: критичных провалов {crit}, информационных {info}", flush=True)
    finally:
        stop_server(load())
        CFG.write_text(json.dumps(pristine, ensure_ascii=False, indent=1), encoding="utf-8")
        backup.unlink(missing_ok=True)
        start_server(load(), ROOT / ".bisect-restored.log")
        print("\nконфиг восстановлен, сервер поднят заново", flush=True)

    probe_names = [p[0] for p in qp.PROBES] + (["loop"] if LONG else [])
    print("\n| вариант | " + " | ".join(probe_names) + " | критичных | инф. |")
    print("|---|" + "---|" * len(probe_names) + "---|---|")
    base_crit = None
    for name in names:
        rows = results.get(name)
        if rows is None:
            print(f"| {name} | " + " | ".join("—" for _ in probe_names) + " | сервер не поднялся | |")
            continue
        by = {r["name"]: r for r in rows}
        cells = []
        for p in probe_names:
            r = by.get(p)
            cells.append("—" if r is None else ("ок" if r["ok"] else ("**ПРОВАЛ**" if r["critical"] else "провал")))
        crit = sum(1 for r in rows if r["critical"] and not r["ok"])
        info = sum(1 for r in rows if not r["critical"] and not r["ok"])
        if name == "base":
            base_crit = crit
        print(f"| {name} | " + " | ".join(cells) + f" | {crit} | {info} |")
    if base_crit is not None:
        worse = [n for n in names if results.get(n) and
                 sum(1 for r in results[n] if r["critical"] and not r["ok"]) > base_crit]
        if worse:
            print(f"\nвердикт: критические проверки ломает {', '.join(worse)} "
                  f"(на base было {base_crit}) - виновник среди них")
        else:
            print(f"\nвердикт: ни один вариант не ухудшил критические проверки относительно base "
                  f"({base_crit} провалов)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
