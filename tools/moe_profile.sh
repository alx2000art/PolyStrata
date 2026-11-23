#!/usr/bin/env bash
# ============================================================================
# tools/moe_profile.sh - где уходит миллисекунда токена: один короткий запрос с
# движковым [moe]-профилем по слоям и сводка по фазам.
#
#   tools/moe_profile.sh [файл-промпта] [max_tokens]
#
# env: CONFIG (по умолчанию $STRATA_DIR/strata-dsv4-ud-iq2_xxs.json), PORT (из конфига),
#      API_KEY (из конфига api_key или окружения), MODEL, OUT_DIR, BOOT_TIMEOUT, PROMPT.
#
# Что делает: бэкапит конфиг, вписывает в его env STRATA_POLY_STAGE2_DEBUG=1 и
# STRATA_EAGLE_DEBUG=1 (движок вычищает первый без второго), перезапускает сервер,
# шлёт один запрос, ждёт, возвращает конфиг и перезапускает сервер на нём.
# Лог этого прогона: $OUT_DIR/server.log, отчёт: $OUT_DIR/report.md.
#
# Формат [moe] (ветка feat/moe-ms; старый бинарник печатает route без скобок - скрипт понимает оба):
#   [moe] il=3 route 2 ms (wait 1, host 1), card 14 ms, copies 0 ms, rounds 1, cpu_rows 8
#   [moe] il=3 cpu 3 ms / gpu 11 ms (both had to run in parallel)
#
# wait  - сколько слоя ждал копию роутера и его top-k (route_rows на карте);
# host  - сколько хостовый цикл строил списки экспертов, границы и таблицу;
# card  - GPU-часть слоя целиком (включая copies - фолт-копии стейджинга).
# ============================================================================
set -o pipefail

STRATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PY="$STRATA_DIR/.venv/bin/python"
[[ -x "$PY" ]] || PY=python3
BOOT_TIMEOUT="${BOOT_TIMEOUT:-3600}"

[[ -f "$CONFIG" ]] || { echo "нет конфига: $CONFIG" >&2; exit 1; }
[[ -f "$STRATA_DIR/serve/server.py" ]] || { echo "нет $STRATA_DIR/serve/server.py" >&2; exit 1; }

PORT="${PORT:-$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(int(c.get("port") or 1524))' "$CONFIG" 2>/dev/null || echo 1524)}"
CFG_KEY="$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(c.get("api_key") or "")' "$CONFIG" 2>/dev/null)"
API_KEY="${API_KEY:-$CFG_KEY}"
MODEL="${MODEL:-my-model}"

PROMPT_FILE="${1:-}"
MAX_TOKENS="${2:-${MAX_TOKENS:-64}}"
PROMPT="${PROMPT:-}"
if [[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]]; then PROMPT="$(cat "$PROMPT_FILE")"; fi
[[ -n "$PROMPT" ]] || PROMPT="Lighthouses along the Baltic were first lit under Peter the Great: a tower fire in Kronstadt in 1704 began a tradition whose core has never changed - a light source, a Fresnel lens, and a published table of characteristics that lets a captain tell one light from another. Many of those towers still run on nineteenth-century optics, with a solar panel where the keeper used to sleep. Tell me what a lighthouse keeper's day looked like in 1900, in two short paragraphs."

OUT_DIR="${OUT_DIR:-$STRATA_DIR/moe_profile-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_DIR" || exit 1
LOG="$OUT_DIR/server.log"
BODY="$OUT_DIR/request.json"
RESP="$OUT_DIR/response.json"
REPORT="$OUT_DIR/report.md"
cp "$CONFIG" "$OUT_DIR/config.bak" || { echo "не смог забэкапить конфиг" >&2; exit 1; }

stop_server() {
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  sleep 2
}
start_server() {
  : > "$LOG"
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" >> "$LOG" 2>&1 & echo $! > "$OUT_DIR/server.pid" )
  echo "  сервер запущен (pid $(cat "$OUT_DIR/server.pid")), лог: $LOG"
}
wait_ready() {
  local deadline=$((SECONDS + BOOT_TIMEOUT)) body
  while (( SECONDS < deadline )); do
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
    printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' && return 0
    sleep 5
  done
  return 1
}
on_int() {
  echo
  echo "прервано: возвращаю конфиг и останавливаю сервер"
  cp "$OUT_DIR/config.bak" "$CONFIG" 2>/dev/null || true
  stop_server
  exit 130
}
trap on_int INT TERM

"$PY" - "$CONFIG" <<'PYCFG'
import json, os, sys
p = sys.argv[1]
with open(p, encoding="utf-8-sig") as f:
    cfg = json.load(f)
env = cfg.get("env") or {}
env["STRATA_POLY_STAGE2_DEBUG"] = "1"
env["STRATA_EAGLE_DEBUG"] = "1"     # движок оставляет первый только при втором
cfg["env"] = env
tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, p)
PYCFG
echo "  конфиг: env STRATA_POLY_STAGE2_DEBUG=1, STRATA_EAGLE_DEBUG=1"

echo "  останавливаю сервер ..."
stop_server
start_server
echo "  жду готовности (до ${BOOT_TIMEOUT}s) ..."
if ! wait_ready; then
  echo "  !!! сервер не поднялся - смотри $LOG" >&2
  cp "$OUT_DIR/config.bak" "$CONFIG"
  exit 1
fi
echo "  модель загружена"

echo "  запрос: $MAX_TOKENS токенов ..."
"$PY" - "$PORT" "$API_KEY" "$MODEL" "$MAX_TOKENS" "$BODY" "$RESP" "$PROMPT" <<'PYREQ'
import json, sys, time, urllib.request
port, key, model, max_tokens, body_path, resp_path, prompt = sys.argv[1:8]
body = {
    "model": model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": int(max_tokens),
    "temperature": 0.0,
    "stream": False,
    "chat_template_kwargs": {"enable_thinking": False},
    "reasoning_effort": "none",
    "thinking": {"type": "disabled"},
}
with open(body_path, "w", encoding="utf-8") as f:
    json.dump(body, f, ensure_ascii=False, indent=1)
req = urllib.request.Request("http://127.0.0.1:%s/v1/chat/completions" % port,
                             data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
resp["_wall_s"] = time.perf_counter() - t0
with open(resp_path, "w", encoding="utf-8") as f:
    json.dump(resp, f, ensure_ascii=False, indent=1)
u = resp.get("usage") or {}
print("  ответ: completion %s, prompt %s, wall %.2f s" % (u.get("completion_tokens"), u.get("prompt_tokens"), resp["_wall_s"]))
PYREQ

sleep 3   # строка статистики движка печатается после ответа

echo "  возвращаю конфиг и перезапускаю сервер ..."
cp "$OUT_DIR/config.bak" "$CONFIG"
stop_server
start_server
wait_ready || echo "  (!) сервер не поднялся после профиля - смотри $LOG"
trap - INT TERM

"$PY" - "$LOG" "$RESP" "$REPORT" <<'PYAN'
import json, re, sys
log_path, resp_path, report_path = sys.argv[1:4]
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
    stats = dict(prompt=int(p), reused=int(reused), read_ms=float(read_ms), pp=float(pp),
                 gen=int(g), gen_ms=float(gen_ms), tg=float(tg))

rx = re.compile(r"\[moe\] il=(\d+) route (\d+) ms(?: \(wait (\d+), host (\d+)\))?, card (\d+) ms, copies (\d+) ms, "
                r"rounds (\d+), cpu_rows (\d+)")
rx2 = re.compile(r"\[moe\] il=(\d+) cpu (\d+) ms / gpu (\d+) ms")
rows, pairs = [], {}
for line in text.splitlines():
    mm = rx.search(line)
    if mm:
        rows.append(dict(il=int(mm.group(1)), route=float(mm.group(2)),
                         wait=float(mm.group(3)) if mm.group(3) else None,
                         host=float(mm.group(4)) if mm.group(4) else None,
                         card=float(mm.group(5)), copies=float(mm.group(6)),
                         rounds=int(mm.group(7)), cpu_rows=int(mm.group(8))))
        continue
    mm = rx2.search(line)
    if mm:
        pairs.setdefault(int(mm.group(1)), []).append((float(mm.group(2)), float(mm.group(3))))

out = []
if not rows:
    out.append("# [moe]-строк нет")
    out.append("")
    out.append("Движок не печатал [moe]. Проверь: (1) сборка из ветки с этим отчётом, "
               "(2) в логе нет строки `debug env var(s) ignored` (нужен STRATA_EAGLE_DEBUG=1).")
    print("\n".join(out))
    open(report_path, "w", encoding="utf-8").write("\n".join(out) + "\n")
    raise SystemExit(1)

tokens = stats["gen"] if stats and stats["gen"] > 0 else gen
n_il = len({r["il"] for r in rows[-400:]})
want = n_il * max(tokens, 1)
dec = rows[-want:] if want > 0 and len(rows) >= want else rows

out.append("# Профиль [moe]: %.0f токенов декода" % tokens)
if stats:
    out.append("- сервер: prompt %d (reused %d), чтение %.0f ms (%.1f tok/s), декод %d за %.0f ms = %.2f tok/s"
               " -> **%.2f ms/токен**" % (stats["prompt"], stats["reused"], stats["read_ms"], stats["pp"], stats["gen"],
                                          stats["gen_ms"], stats["tg"], 1000.0 / stats["tg"]))
else:
    out.append("- строки статистики движка не нашёл; wall %.2f s, completion %d -> %.2f ms/токен"
               % (wall, gen, wall * 1000.0 / max(gen, 1)))
out.append("- [moe]-строк: %d всего, %d взято как декод в %d слоях" % (len(rows), len(dec), n_il))
if len(dec) != want:
    out.append("- (!) декод-строк меньше ожидаемого (%d < %d) - суммы на токен занижены" % (len(dec), want))

denom = max(tokens, 1)
def per(name):
    return sum(r[name] for r in dec) / denom

have_split = any(r["wait"] is not None for r in dec)
route, card, copies = per("route"), per("card"), per("copies")
out.append("")
out.append("## На токен (сумма по слоям)")
out.append("| фаза | мс/токен |")
out.append("|---|---:|")
out.append("| route, всего | %.1f |" % route)
if have_split:
    out.append("| — ожидание роутера (wait) | %.1f |" % per("wait"))
    out.append("| — хост-цикл moe() (host) | %.1f |" % per("host"))
out.append("| card (GPU MoE, включает copies) | %.1f |" % card)
out.append("| copies (внутри card) | %.1f |" % copies)
if stats:
    total = 1000.0 / stats["tg"]
    out.append("")
    out.append("**Остаток вне MoE**: %.1f мс/токен из %.1f (dense, attention, выход, хост вне moe)." % (total - route - card, total))

def layer_means(key):
    g = {}
    for r in dec:
        g.setdefault(r["il"], []).append(r[key])
    return sorted(((sum(v) / len(v), il) for il, v in g.items()), reverse=True)

out.append("")
out.append("## Самые дорогие слои (среднее за токен)")
for title, key in (("card", "card"), ("copies", "copies")):
    top = layer_means(key)[:5]
    out.append("* %s: %s" % (title, ", ".join("il %d %.1f ms" % (il, v) for v, il in top)))

allp = [p for v in pairs.values() for p in v]
if allp:
    mean_cpu = sum(p[0] for p in allp) / len(allp)
    mean_gpu = sum(p[1] for p in allp) / len(allp)
    slower = sum(1 for p in allp if p[0] > p[1])
    out.append("")
    out.append("## CPU-лейн: %d слоёв, среднее cpu %.1f / gpu %.1f мс; cpu>gpu в %d из %d"
               % (len(pairs), mean_cpu, mean_gpu, slower, len(allp)))

out.append("")
out.append("## Что смотреть")
out.append("* большой `wait` — синки/роутер на критическом пути; `host` — хост-цикл moe() (списки/границы).")
out.append("* большой `card` при малых `copies` — GPU-ядра mmq; большой `copies` — IO и кэш экспертов.")
out.append("* большой остаток — dense/attention: следующий профиль уже вне [moe].")

rep = "\n".join(out) + "\n"
print("")
print(rep)
open(report_path, "w", encoding="utf-8").write(rep)
print("отчёт: %s" % report_path)
PYAN
