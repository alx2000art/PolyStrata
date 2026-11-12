#!/usr/bin/env bash
# ============================================================================
# loop606_repro.sh - hunting the '#606' one-token loop: cold boots of the
# server, then short decode-heavy requests one after another; a reply that
# repeats one token 256 times in a row is the loop (the server marks it with
# its "repeated one token" line).  The report keeps the pattern the sightings
# showed - which request of a boot loops, and whether the next request
# recovers without a restart - and saves every reply.
#
#   PROMPT_FILE=~/prompts/doc-30k.txt ./loop606_repro.sh
#   BOOTS=8 REQUESTS=4 ./loop606_repro.sh
#
# The prompt is the file's head (~532 tokens, the sightings' size); the
# request mirrors the ones that looped: temperature 0.9, thinking on (no
# reasoning budget), max_tokens 256.  ~4 minutes a boot.
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/repro606}"
BOOTS="${BOOTS:-5}"
REQUESTS="${REQUESTS:-4}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT_TOKENS="${PROMPT_TOKENS:-532}"
PROMPT_FILE="${PROMPT_FILE:-}"
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

PROMPT="$(head -c $(( PROMPT_TOKENS * 3 )) "$PROMPT_FILE")"
[[ -n "$PROMPT" ]] || die "PROMPT_FILE пуст"
PROMPT_CHARS="${#PROMPT}"
mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
log "петля #606: $BOOTS холодных стартов x $REQUESTS запросов, промпт $PROMPT_CHARS символов (~$PROMPT_TOKENS токенов)"

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
    : > "$OUT_DIR/server.log"
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" \
        >> "$OUT_DIR/server.log" 2>&1 & echo $! > "$OUT_DIR/server.pid" )
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
# one request; prints its elapsed.  The reply lands in $1 (and $1.prompt holds the prompt).
ask() {
    printf '%s' "$PROMPT" > "$1.prompt"
    python3 - "$BASE_URL" "$API_KEY" "$MODEL" "$1" "$MAX_TOKENS" <<'PYEOF'
import json, sys, time, urllib.request
base, key, model, out, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
prompt = open(out + ".prompt", encoding="utf-8").read()
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.9, "max_tokens": mt, "stream": False}).encode("utf-8")
req = urllib.request.Request(base + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.loads(r.read().decode("utf-8"))
el = time.time() - t0
text = d["choices"][0]["message"].get("content") or ""
open(out, "w", encoding="utf-8").write(text + f"\n[elapsed {el:.0f} s]\n")
print(f"{el:.0f}s")
PYEOF
}
repeats() { grep -c "repeated one token" "$OUT_DIR/server.log" 2>/dev/null || true; }

TOTAL_LOOPS=0
for boot in $(seq 1 "$BOOTS"); do
    log "=== старт $boot/$BOOTS ==="
    stop_server
    if [[ -n "$LOG" && -f "$LOG" ]]; then mv "$LOG" "$LOG.repro-$boot-$(date +%H%M%S)" 2>/dev/null || true; fi
    start_server
    if ! wait_ready; then log "  сервер не поднялся - старт пропущен"; continue; fi
    loops=0
    for req in $(seq 1 "$REQUESTS"); do
        before=$(repeats)
        out="$OUT_DIR/boot-$boot-req-$req"
        el=$(ask "$out") || el="?"
        sleep 1
        after=$(repeats)
        done_line="$(grep '\[strata\] done:' "$OUT_DIR/server.log" | tail -1 | sed 's/^.*done: /done /;s/, expert cache.*//')"
        if (( after > before )); then
            loops=$((loops + 1)); TOTAL_LOOPS=$((TOTAL_LOOPS + 1))
            log "  запрос $req (${el}s): ПЕТЛЯ!  [$done_line]"
            grep "repeated one token" "$OUT_DIR/server.log" | tail -1 | sed 's/^/      /'
            printf '      ответ: %s\n' "$(head -c 100 "$out" | tr '\n' ' ')"
        else
            log "  запрос $req (${el}s): ок  [$done_line]"
        fi
    done
    if (( loops > 0 )); then
        note="не проверены"
        (( loops < REQUESTS )) && note="восстановились сами"
        log "  старт $boot: $loops петель из $REQUESTS (последующие запросы $note)"
    fi
done
log ""
log "=== итог: $TOTAL_LOOPS петель за $BOOTS стартов x $REQUESTS запросов; ответы и логи в $OUT_DIR ==="
