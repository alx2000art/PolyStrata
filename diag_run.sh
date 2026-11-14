#!/usr/bin/env bash
# ============================================================================
# diag_run.sh - one cold measured run with the whole report, bench_sweep's
# style: the server's log lands in $OUT_DIR/server.log.last (cleared before
# the start), dmon runs in the background, one prompt, and the report prints
# the [moe] route/card/copies lines, the fault copies, the dmon busy/idle
# split, the nvme sectors, the reading tail and the done line.
#
#   PROMPT_FILE=~/prompts/doc-30k.txt ./diag_run.sh
#   REBUILD=1 PROMPT_FILE=... ./diag_run.sh     # also ./setup.sh --rebuild-engine
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
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv — запустите ./setup.sh хотя бы раз"

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
PROMPT_CHARS="${#PROMPT}"
mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"

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
    : > "$OUT_DIR/server.log.last"
    ( cd "$STRATA_DIR" && STRATA_POLY_STAGE2_DEBUG=1 nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" \
        >> "$OUT_DIR/server.log.last" 2>&1 & echo $! > "$OUT_DIR/server.pid" )
    log "  сервер запущен (pid $(cat "$OUT_DIR/server.pid")), лог: $OUT_DIR/server.log.last"
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

log "=== 1/4: git pull ==="
SELF="$(basename "$0")"
BEFORE=$(git -C "$STRATA_DIR" hash-object "$STRATA_DIR/$SELF" 2>/dev/null || echo x)
git -C "$STRATA_DIR" stash push -m "diag auto" >/dev/null 2>&1 || true
git -C "$STRATA_DIR" pull --ff-only 2>&1 | tail -2
# the pull may have rewritten THIS script - and bash reads it lazily, so a rewrite mid-run executes a
# frankenstein of the two versions (the pull's own diff proves it happened once).  Restart fresh.
AFTER=$(git -C "$STRATA_DIR" hash-object "$STRATA_DIR/$SELF" 2>/dev/null || echo y)
if [[ "$BEFORE" != "$AFTER" ]]; then
    log "  $SELF обновился при pull — перезапускаю себя целиком"
    exec "$STRATA_DIR/$SELF" "$@"
fi

if [[ "${REBUILD:-0}" == 1 ]]; then
    log "=== 2/4: пересборка (setup.sh --rebuild-engine --no-start; лог: $OUT_DIR/build.log) ==="
    stop_server
    ( cd "$STRATA_DIR" && ./setup.sh --update --build --rebuild-engine --cuda 13 --no-start --yes </dev/null >> "$OUT_DIR/build.log" 2>&1 )
    rc=$?
    tail -5 "$OUT_DIR/build.log"
    (( rc != 0 )) && die "сборка кончилась с ошибкой $rc — смотри $OUT_DIR/build.log"
else
    log "=== 2/4: пересборка пропущена (REBUILD=1 чтобы собрать) ==="
fi

log "=== 3/4: рестарт на холодном пуле ==="
stop_server
if [[ -n "$LOG" && -f "$LOG" ]]; then mv "$LOG" "$LOG.diag-$(date +%H%M%S)" || true; fi
# the server spawns the engine with the config's env block, not the caller's: the [moe] lines need their
# switch inside the config.  The rest of the env stays as the config carries it: the stage spec's default
# (64) won its verdict, and SCORE_CONTEXT is a footgun - the chars/3 estimate undershoots Cyrillic, and a
# stride shorter than the prompt corrupts the scores.
python3 - "$CONFIG" <<'PYEOF'
import json, os, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
cfg.setdefault("env", {})["STRATA_POLY_STAGE2_DEBUG"] = "1"
tmp = sys.argv[1] + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, sys.argv[1])
PYEOF
start_server
wait_ready || die "сервер не поднялся — смотри $OUT_DIR/server.log.last"

nvidia-smi dmon -s pucm -d 1 -o TD > "$OUT_DIR/gpu_diag.log" 2>/dev/null &
DMON=$!
b=$(sectors)
log "=== 4/4: запрос ==="
ask "$OUT_DIR/diag-run.txt"
sleep 2
a=$(sectors)
kill "$DMON" 2>/dev/null; wait "$DMON" 2>/dev/null

ELOG="$LOG"   # the engine-side lines ([moe], [stage-spec]) go to the engine's own log, not the server's
log ""
log "=== ОТЧЁТ ==="
log "-- [moe] слоёв 20+ (route / card / copies):"
grep "\[moe\]" "$ELOG" 2>/dev/null | awk -F"il=" '{split($2,a,","); if (a[1]+0 >= 20) print}' | head -100 | sed 's/^/   /'
log "-- fault-копии:"
grep "copy took" "$ELOG" 2>/dev/null | head -20 | sed 's/^/   /'
log "-- dmon занятость:"
awk '$5>=50 {hot++} $5<5 {idle++} END{printf "   busy: %d s, idle: %d s\n", hot, idle}' "$OUT_DIR/gpu_diag.log"
log "-- SSD за прогон: $(( (a - b) * 512 / 1024 / 1024 )) MiB"
log "-- хвост чтения:"
grep "reading the prompt" "$OUT_DIR/server.log.last" 2>/dev/null | tail -3 | sed 's/^/   /'
log "-- done:"
grep "\[strata\] done:" "$OUT_DIR/server.log.last" 2>/dev/null | tail -1 | sed 's/^/   /'
log "-- промпт и ответ:"
log "   промпт: $PROMPT_CHARS символов; ответ: $OUT_DIR/diag-run.txt"
log "готово. ответ: $OUT_DIR/diag-run.txt, лог: $OUT_DIR/server.log.last"
