#!/usr/bin/env bash
# ============================================================================
# eagle_taps_ab.sh - which tap representation the EAGLE-3 head wants, by the
# draft's own confidence p at its first step (a good wiring gives p well above
# a few thousandths).  The variants: STRATA_EAGLE_TAPS=mean (the layer's
# hyper-connection streams averaged), s0..s3 (one stream as it lies).
#
#   ./eagle_taps_ab.sh
#   PROMPT_LIMIT=40 MAX_TOKENS=16 ./eagle_taps_ab.sh
#
# A short prompt keeps each try to a minute: the p of the first draft step is
# what is compared, not the answer.
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PROMPT_FILE="${PROMPT_FILE:-$HOME/prompts/doc-30k.txt}"
PROMPT_LIMIT="${PROMPT_LIMIT:-40}"
MAX_TOKENS="${MAX_TOKENS:-32}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"
VARIANTS="${VARIANTS:-mean mix ffn attn s0}"
CATS="${CATS:-eh he}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }
[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -f "$PROMPT_FILE" ]] || die "нет промпта $PROMPT_FILE"

PORT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig')).get('port') or 8080)" "$CONFIG")"
MODEL="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig')).get('model_name') or 'my-model')" "$CONFIG")"
LOG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig')).get('log') or '')" "$CONFIG")"
API_KEY="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1],encoding='utf-8-sig')).get('api_key') or '')" "$CONFIG")"
BASE="http://192.168.5.80:$PORT/v1"
[[ -n "$API_KEY" ]] || die "нет api_key в конфиге"

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/192.168.5.80/$PORT" 2>/dev/null; }
stop_server() {
    pkill -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    local deadline=$((SECONDS + STOP_WAIT))
    while (( SECONDS < deadline )); do port_open || return 0; sleep 1; done
    pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    pkill -KILL -f "engine-poly/strata-poly" 2>/dev/null || true
    sleep 2
}
set_taps() {   # $1 = mean | s0..s3, $2 = eh | he
    python3 - "$CONFIG" "$1" "$2" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
e = cfg.setdefault("env", {})
if sys.argv[2] == "mean":
    e.pop("STRATA_EAGLE_TAPS", None)
else:
    e["STRATA_EAGLE_TAPS"] = sys.argv[2]
if sys.argv[3] == "he":
    e["STRATA_EAGLE_CAT"] = "he"
else:
    e.pop("STRATA_EAGLE_CAT", None)
tmp = sys.argv[1] + ".tmp"
json.dump(cfg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(tmp, sys.argv[1])
PY
}
ask() {   # prints the done line
    python3 - "$BASE" "$API_KEY" "$MODEL" "$PROMPT_FILE" "$PROMPT_LIMIT" "$MAX_TOKENS" <<'PY'
import json, sys, time, urllib.request
base, key, model, pf, limit, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
prompt = "".join(open(pf, encoding="utf-8").readlines()[:limit])
prompt = f"[taps probe {time.time_ns()}]\n" + prompt          # past the prompt cache
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": mt, "reasoning_budget_tokens": 0,
                   "stream": False}).encode()
req = urllib.request.Request(base + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=1800) as r:
        r.read()
except Exception as e:
    print(f"  запрос: {e}")
PY
}

log "=== A/B по представлению taps: $VARIANTS; порядок конката: $CATS ==="
log "промпт: $PROMPT_FILE (первые $PROMPT_LIMIT строк), max_tokens $MAX_TOKENS"
declare -A P
for v in $VARIANTS; do
  for c in $CATS; do
    label="$v/$c"
    log "--- вариант: $label ---"
    set_taps "$v" "$c"
    stop_server
    if [[ -n "$LOG" && -f "$LOG" ]]; then mv "$LOG" "$LOG.tapsab-${v}-${c}-$(date +%H%M%S)" || true; fi
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" >> "$STRATA_DIR/diag_run/taps_ab_${v}_${c}.log" 2>&1 & )
    ready=0
    for _ in $(seq 1 $((BOOT_TIMEOUT / 5))); do
        if port_open && curl -s -m 5 -H "Authorization: Bearer $API_KEY" "http://192.168.5.80:$PORT/health" \
             | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then ready=1; break; fi
        sleep 5
    done
    (( ready )) || { log "  сервер не поднялся"; continue; }
    ask >/dev/null
    p="$(grep -o 'draft 1/[0-9]*: p=[0-9.]*' "$LOG" 2>/dev/null | head -1 | grep -o '[0-9.]*$')"
    a="$(grep -o 'drafts [0-9]*/[0-9]*' "$LOG" 2>/dev/null | tail -1)"
    P[$label]="${p:-?}"
    log "  p первого драфта: ${p:-нет}   ($a)"
  done
done
stop_server
set_taps mean eh
log ""
log "=== ИТОГ (p первого драфта; больше - лучше) ==="
for v in $VARIANTS; do for c in $CATS; do log "  $v/$c: ${P[$v/$c]:-?}"; done; done
log "конфиг возвращён на mean/eh; сервер остановлен - подними обычным пуском"
