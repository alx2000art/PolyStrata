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
# Лог движка этого прогона: $OUT_DIR/engine.log.profile (stderr движка идёт в файл из
# поля "log" конфига, не в лог сервера), отчёт: $OUT_DIR/report.md.
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
HOST="${HOST:-$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(c.get("host") or "127.0.0.1")' "$CONFIG" 2>/dev/null || echo 127.0.0.1)}"
CFG_KEY="$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(c.get("api_key") or "")' "$CONFIG" 2>/dev/null)"
API_KEY="${API_KEY:-$CFG_KEY}"
MODEL="${MODEL:-my-model}"
# The [moe] lines and the per-request stats are the ENGINE's stderr, and the server sends it to the config's
# own log file (server.py: stderr=self.log, cfg["log"]), not to the server's stdout log.  Read that file, and
# cut this run's segment by its size taken before the restart (it is opened in append mode across restarts).
ELOG="${ELOG:-$("$PY" -c 'import json,sys;c=json.load(open(sys.argv[1],encoding="utf-8-sig"));print(c.get("log") or "")' "$CONFIG" 2>/dev/null)}"
[[ -n "$ELOG" ]] || ELOG="${CONFIG%.*}.log"

PROMPT_FILE="${1:-}"
MAX_TOKENS="${2:-${MAX_TOKENS:-96}}"
WARM="${WARM:-3}"                 # warm-up requests with the same prompt before the measured one
PROMPT_CHARS="${PROMPT_CHARS:-0}" # > 0: repeat the prompt's text up to this many characters (~3 per Russian token)
PROMPT="${PROMPT:-}"
if [[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]]; then PROMPT="$(cat "$PROMPT_FILE")"; fi
[[ -n "$PROMPT" ]] || PROMPT="Lighthouses along the Baltic were first lit under Peter the Great: a tower fire in Kronstadt in 1704 began a tradition whose core has never changed - a light source, a Fresnel lens, and a published table of characteristics that lets a captain tell one light from another. Many of those towers still run on nineteenth-century optics, with a solar panel where the keeper used to sleep. Tell me what a lighthouse keeper's day looked like in 1900, in two short paragraphs."
if (( PROMPT_CHARS > 0 )); then
  PROMPT=$("$PY" - "$PROMPT_CHARS" "$PROMPT" <<'PYPAD'
import sys
target, text = int(sys.argv[1]), sys.argv[2].rstrip()
if text and len(text) < target:
    n = (target - len(text)) // (len(text) + 2) + 1
    text = ("\n\n".join([text] * n) + "\n\n" + text)
print(text)
PYPAD
)
fi

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
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "http://$HOST:$PORT/health" 2>/dev/null || true)
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
LOG_OFF=$(stat -c%s "$ELOG" 2>/dev/null || echo 0)   # our engine's lines start here

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

# A request, saved only when a path is given.  The measured one goes last; the warm-ups use the SAME prompt,
# because the first request after a start pays the cold page cache and the expert cache's first fill, and a
# different prompt would warm another set of experts (the first profile ran at ~15 tok/s for that reason).
ask() {  # $1 = max_tokens, $2 = save path ("" = discard), $3 = prompt
  "$PY" - "$HOST" "$PORT" "$API_KEY" "$MODEL" "$1" "$2" "$3" "$BODY" <<'PYREQ'
import json, sys, time, urllib.request
host, port, key, model, max_tokens, save_path, prompt, body_path = sys.argv[1:9]
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
req = urllib.request.Request("http://%s:%s/v1/chat/completions" % (host, port),
                             data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)
resp["_wall_s"] = time.perf_counter() - t0
if save_path:
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(resp, f, ensure_ascii=False, indent=1)
u = resp.get("usage") or {}
print("  ответ: completion %s, prompt %s, wall %.2f s"
      % (u.get("completion_tokens"), u.get("prompt_tokens"), resp["_wall_s"]))
PYREQ
}

for w in $(seq 1 "$WARM"); do
  echo "  прогрев $w/$WARM (32 токена, тот же промпт) ..."
  ask 32 "" "$PROMPT"
  sleep 2
done

echo "  запрос: $MAX_TOKENS токенов ..."
ask "$MAX_TOKENS" "$RESP" "$PROMPT"

sleep 3   # строка статистики движка печатается после ответа
# The engine's log is append-only across restarts: keep only this run's segment, from the size taken before
# our start.  The final restart truncates the server's own log, so it cannot hold the [moe] lines at all.
if [[ -f "$ELOG" ]]; then
  tail -c +$((LOG_OFF + 1)) "$ELOG" > "$OUT_DIR/engine.log.profile" 2>/dev/null \
    || cp "$ELOG" "$OUT_DIR/engine.log.profile"
else
  : > "$OUT_DIR/engine.log.profile"
fi
echo "  лог движка этого прогона: $OUT_DIR/engine.log.profile ($(wc -l < "$OUT_DIR/engine.log.profile" 2>/dev/null || echo 0) строк)"

echo "  возвращаю конфиг и перезапускаю сервер ..."
cp "$OUT_DIR/config.bak" "$CONFIG"
stop_server
start_server
wait_ready || echo "  (!) сервер не поднялся после профиля - смотри $LOG"
trap - INT TERM

"$PY" "$STRATA_DIR/tools/moe_profile_report.py" "$OUT_DIR/engine.log.profile" "$RESP" "$REPORT"
