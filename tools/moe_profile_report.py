#!/usr/bin/env python3
"""moe_profile_report.py LOG RESP REPORT - the [moe] profile of one request.

The engine's stderr (STRATA_POLY_STAGE2_DEBUG) prints two kinds of lines:
  the answer's path, one a layer a token:
    [moe] il=3 n=1 route 2 ms (wait 1, host 1), cpu 7 ms, total 13 ms, cached 5, host_experts 3
  a prompt batch, one a layer:
    [moe] il=3 n=108 route 9 ms (wait 7, host 2), card 40 ms, copies 12 ms, rounds 3, cpu_rows 0
The first splits the layer into the router wait, the host's routing (and the lists), the CPU experts and the
whole call; the second is the read's batch, where the experts outside the cache come through the staging buffer.
"""
import json
import re
import sys

log_path, resp_path, report_path = sys.argv[1:4]
srv_path = sys.argv[4] if len(sys.argv) > 4 else ""
text = open(log_path, encoding="utf-8", errors="replace").read()
resp = json.load(open(resp_path, encoding="utf-8"))
u = resp.get("usage") or {}
gen = int(u.get("completion_tokens") or 0)
wall = float(resp.get("_wall_s") or 0.0)

stats = None
m = re.findall(r"prompt (\d+) tokens = (\d+) reused \+ \d+(?: of \d+)? read in ([\d.]+) ms \(([\d.]+) tok/s\), "
               r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\)", text)
if m:
    p, reused, read_ms, pp, g, gen_ms, tg = m[-1]
    stats = dict(prompt=int(p), reused=int(reused), read_ms=float(read_ms), pp=float(pp), gen=int(g),
                 gen_ms=float(gen_ms), tg=float(tg))
# The server's per-request line is the decode-only figure (the engine's log does not print one): the last one
# is the measured request, the warm-ups come before it.
done = None
if srv_path:
    try:
        srv = open(srv_path, encoding="utf-8", errors="replace").read()
    except OSError:
        srv = ""
    d = re.findall(r"done: (\d+) tokens in ([\d.]+) s \(([\d.]+) tok/s\)", srv)
    if d:
        n2, el, tps = d[-1]
        done = dict(tokens=int(n2), el=float(el), tps=float(tps))
if done:
    tokens = done["tokens"]
elif stats and stats["gen"] > 0:
    tokens = stats["gen"]
else:
    tokens = gen

RXD = re.compile(r"\[moe\] il=(\d+) n=(\d+) route (\d+) ms \(wait (\d+), host (\d+)\), cpu (\d+) ms, total (\d+) ms, "
                 r"cached (\d+), host_experts (\d+)")
RXP = re.compile(r"\[moe\] il=(\d+) (?:n=(\d+) )?route (\d+) ms(?: \(wait (\d+), host (\d+)\))?, card (\d+) ms, "
                 r"copies (\d+) ms, rounds (\d+), cpu_rows (\d+)")
dec, pre = [], []
for line in text.splitlines():
    mm = RXD.search(line)
    if mm:
        dec.append(dict(il=int(mm.group(1)), n=int(mm.group(2)), route=float(mm.group(3)), wait=float(mm.group(4)),
                        host=float(mm.group(5)), cpu=float(mm.group(6)), total=float(mm.group(7)),
                        cached=int(mm.group(8)), host_experts=int(mm.group(9))))
        continue
    mm = RXP.search(line)
    if mm:
        pre.append(dict(il=int(mm.group(1)), n=int(mm.group(2) or 0), route=float(mm.group(3)),
                        wait=float(mm.group(4) or 0), host=float(mm.group(5) or 0), card=float(mm.group(6)),
                        copies=float(mm.group(7)), rounds=int(mm.group(8)), cpu_rows=int(mm.group(9))))

out = []
if not dec and not pre:
    out.append("# [moe]-строк нет")
    out.append("")
    out.append("В сегменте лога движка [moe] нет (%d строк). Проверь: (1) сборка из ветки с этим отчётом; "
               "(2) в сегменте нет `debug env var(s) ignored` (нужен STRATA_EAGLE_DEBUG=1); "
               "(3) если оба в порядке - все эксперты были в VRAM (moe_card) и печатать было нечего."
               % len(text.splitlines()))
    rep = "\n".join(out) + "\n"
    open(report_path, "w", encoding="utf-8").write(rep)
    print(rep)
    sys.exit(1)

out.append("# Профиль [moe]: %d токенов декода" % tokens)
if done:
    out.append("- декод: %d токенов за %.0f s = **%.2f мс/токен** (%.1f tok/s); prompt %d токенов"
               % (done["tokens"], done["el"], 1000.0 / done["tps"], done["tps"], u.get("prompt_tokens") or 0))
elif stats:
    out.append("- движок: prompt %d (reused %d), чтение %.0f ms (%.1f tok/s), декод %d за %.0f ms = "
               "**%.2f ms/токен** (%.2f tok/s)"
               % (stats["prompt"], stats["reused"], stats["read_ms"], stats["pp"], stats["gen"], stats["gen_ms"],
                  1000.0 / stats["tg"], stats["tg"]))
else:
    out.append("- ни серверной строки `done:`, ни статистики движка нет; wall %.2f s, completion %d -> "
               "%.2f ms/токен (с префиллом!)" % (wall, gen, wall * 1000.0 / max(gen, 1)))
out.append("")

den = max(tokens, 1)
if dec:
    n_il = len({r["il"] for r in dec[-400:]})
    want = n_il * den
    tail = dec[-want:] if len(dec) >= want else dec
    full = len(tail) == want
    out.append("## Декод: %d строк, %d слоёв%s"
               % (len(dec), n_il, "" if full else " (неполно: %d из %d - суммы занижены)" % (len(tail), want)))

    def per(key):
        return sum(r[key] for r in tail) / den

    out.append("")
    out.append("| фаза (сумма по слоям) | мс/токен |")
    out.append("|---|---:|")
    out.append("| route, всего | %.1f |" % per("route"))
    out.append("| — ожидание роутера (wait) | %.1f |" % per("wait"))
    out.append("| — хост: маршрутизация и списки (host) | %.1f |" % per("host"))
    out.append("| CPU-эксперты (cpu, внутри total) | %.1f |" % per("cpu"))
    out.append("| **слои MoE, всего (total)** | %.1f |" % per("total"))
    if done or stats:
        tt = 1000.0 / (done["tps"] if done else stats["tg"])
        out.append("")
        out.append("**Остаток вне MoE**: %.1f мс/токен из %.1f (attention, dense, выход, хост вне moe)."
                   % (tt - per("total"), tt))
    nc = sum(r["cached"] for r in tail) / max(1, len(tail))
    nh = sum(r["host_experts"] for r in tail) / max(1, len(tail))
    out.append("")
    out.append("* экспертов на слой: в VRAM %.1f, на CPU %.1f (%.0f%% на CPU; tail-skip уже применён)"
               % (nc, nh, 100.0 * nh / max(1e-9, nc + nh)))

    def means(key):
        g = {}
        for r in tail:
            g.setdefault(r["il"], []).append(r[key])
        return sorted(((sum(v) / len(v), il) for il, v in g.items()), reverse=True)

    out.append("* дорогие слои по total: " + ", ".join("il %d %.1f ms" % (il, v) for v, il in means("total")[:5]))
    out.append("* дорогие слои по cpu: " + ", ".join("il %d %.1f ms" % (il, v) for v, il in means("cpu")[:5]))
    out.append("* на слой: total %.1f ms, cpu %.1f ms, wait+host %.1f ms"
               % (per("total") / max(1, n_il), per("cpu") / max(1, n_il), per("route") / max(1, n_il)))
else:
    out.append("## Декода в логе нет: бинарник старше аудита декода (в логе только префилл)" if pre else "")
    out.append("")

if pre:
    out.append("")
    out.append("## Префилл: %d строк (один или несколько батчей)" % len(pre))
    out.append("* card %.0f ms (copies %.0f ms), route %.0f ms (wait %.0f / host %.0f), rounds %.1f"
               % (sum(r["card"] for r in pre), sum(r["copies"] for r in pre), sum(r["route"] for r in pre),
                  sum(r["wait"] for r in pre), sum(r["host"] for r in pre),
                  sum(r["rounds"] for r in pre) / max(1, len(pre))))
    out.append("")

out.append("## Что смотреть")
out.append("* большой `wait` - синк роутера на критическом пути; `host` - маршрутизация и списки в moe().")
out.append("* `cpu` велик относительно `total` - CPU-эксперты (их чтение из page cache/диска) держат слой.")
out.append("* `total` много меньше мс/токен - остаток в attention/dense/выходе: следующий профиль вне MoE.")
rep = "\n".join(out) + "\n"
open(report_path, "w", encoding="utf-8").write(rep)
print(rep)
