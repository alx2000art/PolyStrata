#!/usr/bin/env bash
# ============================================================================
# cold_test.sh - one cold-start measurement of the staging guess-reader, all
# by itself: stops the server, moves the engine's log away (so the run's own
# lines are the whole log), starts the server on a cold pool, runs dmon in
# the background, sends one prompt (fresh text please), then reports:
#
#   - the guess's settled counts of the run's first pass
#   - the dmon busy/idle split (the saw-tooth's idle is what the guess kills)
#   - the prompt read's tail tok/s and the nvme sectors read around the run
#
# usage: PROMPT_FILE=/path/to/prompt.txt [PROMPT_OFFSET=n] ./cold_test.sh
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/cold_test}"
MAX_TOKENS="${MAX_TOKENS:-384}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-1}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]] || die "укажи PROMPT_FILE=/путь/к/промпту.txt"
if [[ -z "${API_KEY:-}" ]]; then
    API_KEY="$(python3 -c "
import json,sys
try: print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('api_key') or '')
except Exception: print('')" "$CONFIG")"
fi
[[ -n "$API_KEY" ]] || die "нет API-ключа (export API_KEY=... или поле api_key конфига)"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv"

PORT="$(python3 -c "
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('port') or 8080)" "$CONFIG")"
MODEL="$(python3 -c "
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('model_name') or 'my-model')" "$CONFIG")"
HOST="$(python3 -c "
import json,sys
h=json.load(open(sys.argv[1], encoding='utf-8-sig')).get('host') or '127.0.0.1'
print('127.0.0.1' if h in ('0.0.0.0','::') else h)" "$CONFIG")"
LOG="$(python3 -c "
import json,sys
print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('log') or '')" "$CONFIG")"
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"

PROMPT="$(awk -v a="$PROMPT_OFFSET" -v b="$PROMPT_LIMIT" 'NR>=a && (b==0 || NR<a+b)' "$PROMPT_FILE")"
[[ -n "$PROMPT" ]] || die "PROMPT_FILE пуст"

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }
stop_server() {
    pkill -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    local deadline=$((SECONDS + STOP_WAIT))
    while (( SECONDS < deadline )); do port_open || return 0; sleep 1; done
    pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    pkill -KILL -f "engine-poly/strata-poly" 2>/dev/null || true
    sleep 2
}
start_server() {
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" >> "$LOG" 2>&1 & )
}
wait_ready() {
    local deadline=$((SECONDS + BOOT_TIMEOUT)) body
    log "  жду готовности (до ${BOOT_TIMEOUT}s) ..."
    while (( SECONDS < deadline )); do
        port_open || { sleep 5; continue; }
        body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
        if printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then
            log "  модель загружена (холодный пул)"
            return 0
        fi
        sleep 5
    done
    return 1
}
ask() {
    printf '%s' "$PROMPT" > "$1.prompt"
    python3 - "$BASE_URL" "$API_KEY" "$MODEL" "$1" "$MAX_TOKENS" <<'PYEOF'
import json, sys, time, urllib.request
base, key, model, out, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
prompt = open(out + ".prompt", encoding="utf-8").read()
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.9, "max_tokens": mt, "reasoning_budget_tokens": 0,
                   "stream": False}).encode("utf-8")
req = urllib.request.Request(base + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.loads(r.read().decode("utf-8"))
el = time.time() - t0
open(out, "w", encoding="utf-8").write((d["choices"][0]["message"].get("content") or "")
                                       + f"\n[elapsed {el:.0f} s]\n")
print(f"  elapsed {el:.0f} s")
PYEOF
}
sectors() { awk '$3 ~ /^nvme[0-9]+n[0-9]+$/{s+=$6} END{print s+0}' /proc/diskstats; }

mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
log "=== cold test: рестарт, чистый лог, dmon, один запрос ==="
stop_server
if [[ -n "$LOG" && -f "$LOG" ]]; then
    mv "$LOG" "$LOG.cold-$(date +%H%M%S)" || die "не отодвинуть лог"
fi
start_server
wait_ready || { restore=1; log "сервер не поднялся — проверь лог $LOG"; exit 1; }

nvidia-smi dmon -s pucm -d 1 -o TD > "$OUT_DIR/gpu_cold.log" 2>/dev/null &
DMON=$!
log "  dmon снят в фон (pid $DMON)"
b=$(sectors)
log "  запрос ..."
ask "$OUT_DIR/cold-run.txt"
sleep 2
a=$(sectors)
kill "$DMON" 2>/dev/null; wait "$DMON" 2>/dev/null

log ""
log "=== ОТЧЁТ ==="
log "1) спека (первые 8 задач прогона):"
grep "stage-spec" "$LOG" 2>/dev/null | head -8 | sed 's/^/   /'
log "2) чтение промпта (хвост):"
grep "reading the prompt" "$LOG" 2>/dev/null | tail -3 | sed 's/^/   /'
log "3) done-строка:"
grep "\[strata\] done:" "$LOG" 2>/dev/null | tail -1 | sed 's/^/   /'
log "4) dmon занятость (sm>=50 против sm<5):"
awk '$5>=50 {hot++} $5<5 {idle++} END{printf "   busy: %d s, idle: %d s\n", hot, idle}' "$OUT_DIR/gpu_cold.log"
log "5) SSD за прогон: $(( (a - b) * 512 / 1024 / 1024 )) MiB"
log ""
log "  вердикт: settled 200+ в п.1 и idle ~0 в п.4 - спека работает; elapsed"
log "  сравни с прежними ~151 s первого касания ( ответы: $OUT_DIR/cold-run.txt )."
