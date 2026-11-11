#!/usr/bin/env bash
# ============================================================================
# diag_run.sh - git pull, rebuild, one cold measured run, the whole report.
#   PROMPT_FILE=~/prompts/doc-30k.txt ./diag_run.sh
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/diag_run}"
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
[[ -n "$API_KEY" ]] || die "нет API-ключа"
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
[[ -n "$PROMPT" ]] || die "PROMPT_FILE пуст (проверь PROMPT_OFFSET - файл короткий?)"

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
    while (( SECONDS < deadline )); do
        port_open || { sleep 5; continue; }
        body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
        if printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then return 0; fi
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

log "=== 1/5: git pull ==="
git -C "$STRATA_DIR" stash push -m "diag auto" >/dev/null 2>&1 || true
git -C "$STRATA_DIR" pull --ff-only 2>&1 | tail -2

log "=== 2/5: пересборка strata-poly ==="
"$STRATA_DIR/.venv/bin/cmake" --build "$STRATA_DIR/build-poly" --target strata-poly -j 16 2>&1 | tail -2

log "=== 3/5: рестарт на холодном пуле, чистый лог ==="
stop_server
if [[ -n "$LOG" && -f "$LOG" ]]; then mv "$LOG" "$LOG.diag-$(date +%H%M%S)" || true; fi
start_server
wait_ready || die "сервер не поднялся — смотри $LOG"

nvidia-smi dmon -s pucm -d 1 -o TD > "$OUT_DIR/gpu_diag.log" 2>/dev/null &
DMON=$!
b=$(sectors)
log "=== 4/5: запрос ==="
ask "$OUT_DIR/diag-run.txt"
sleep 2
a=$(sectors)
kill "$DMON" 2>/dev/null; wait "$DMON" 2>/dev/null

log "=== 5/5: ОТЧЁТ ==="
log "-- [moe] слоёв 20+ (route / card / copies):"
grep "\[moe\]" "$LOG" 2>/dev/null | awk -F"il=" '{split($2,a,","); if (a[1]+0 >= 20) print}' | head -100 | sed 's/^/   /'
log "-- fault-копии:"
grep "copy took" "$LOG" 2>/dev/null | head -20 | sed 's/^/   /'
log "-- dmon занятость:"
awk '$5>=50 {hot++} $5<5 {idle++} END{printf "   busy: %d s, idle: %d s\n", hot, idle}' "$OUT_DIR/gpu_diag.log"
log "-- SSD за прогон: $(( (a - b) * 512 / 1024 / 1024 )) MiB"
log "-- хвост чтения:"
grep "reading the prompt" "$LOG" 2>/dev/null | tail -3 | sed 's/^/   /'
log "-- done:"
grep "\[strata\] done:" "$LOG" 2>/dev/null | tail -1 | sed 's/^/   /'
log "готово. ответ: $OUT_DIR/diag-run.txt"
