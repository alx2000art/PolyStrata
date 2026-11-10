#!/usr/bin/env bash
# ============================================================================
# pool_ab.sh - does a bigger --ram-experts-mib cut the prompt reads' disk
# traffic?  The SSD sectors read are counted around the same request twice:
# once on the config's own pool and once on the bigger one, each after a warm
# run of the same prompt (the page cache's state is levelled by the warm run).
#
# For every variant the script rewrites "--ram-experts-mib" in the config,
# restarts the server, runs the prompt once to warm, then once more with the
# disk counter around it, and prints both variants' deltas and done lines.
#
# usage: PROMPT_FILE=/path/to/fresh-text.txt ./pool_ab.sh
#   the prompt file: 25-30k tokens of text the model has NOT read yet (a new
#   chapter, a codebase dump...) - fresh, so the first touch shows at all;
#   PROMPT_OFFSET/PROMPT_LIMIT can point at a slice of it (lines).
#   API_KEY comes from the config's api_key field, as everywhere.
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/pool_ab}"
VARIANTS=(${VARIANTS:-6144 12288})
MAX_TOKENS="${MAX_TOKENS:-384}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-1}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"          # 0: the whole file
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]] || die "укажи PROMPT_FILE=/путь/к/свежему/тексту.txt (25-30k токенов)"
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
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"

# the prompt's text: the file's slice, as one string
PROMPT="$(awk -v a="$PROMPT_OFFSET" -v b="$PROMPT_LIMIT" 'NR>=a && (b==0 || NR<a+b)' "$PROMPT_FILE")"
[[ -n "$PROMPT" ]] || die "PROMPT_FILE пуст"
PROMPT_CHARS="${#PROMPT}"

BACKUP="$CONFIG.pool-ab-backup-$(date +%Y%m%d-%H%M%S)"
cp "$CONFIG" "$BACKUP" || die "не забэкапить конфиг"
restore_config() { cp "$BACKUP" "$CONFIG"; }
on_interrupt() {
    log "прервано: восстанавливаю конфиг и глушу сервер"
    restore_config
    pkill -f "$STRATA_DIR/serve/server.py" 2>/dev/null
    exit 130
}
trap on_interrupt INT TERM

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
        --engine strata --config "$CONFIG" --port "$PORT" >> "$OUT_DIR/server.log" 2>&1 & )
}
wait_ready() {
    local deadline=$((SECONDS + BOOT_TIMEOUT)) body
    log "  жду готовности (до ${BOOT_TIMEOUT}s) ..."
    while (( SECONDS < deadline )); do
        port_open || { sleep 5; continue; }
        body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
        if printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then
            log "  модель загружена"
            return 0
        fi
        sleep 5
    done
    return 1
}

# the pool size lives in the config's args list, right after "--ram-experts-mib"
set_pool_mib() {
    python3 - "$CONFIG" "$1" <<'PYEOF'
import json, os, sys
path, mib = sys.argv[1], sys.argv[2]
cfg = json.load(open(path, encoding="utf-8-sig"))
args = cfg["args"] if isinstance(cfg.get("args"), list) else cfg["args"].split()
i = args.index("--ram-experts-mib")
args[i + 1] = str(mib)
cfg["args"] = args
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
PYEOF
}

# one request of the prompt; the reply lands in $1, the wall time is printed
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
u = d.get("usage") or {}
text = d["choices"][0]["message"].get("content") or ""
open(out, "w", encoding="utf-8").write(text + f"\n[elapsed {el:.0f} s, prompt_tokens {u.get('prompt_tokens')}, completion {u.get('completion_tokens')}]\n")
print(f"  elapsed {el:.0f} s, prompt {u.get('prompt_tokens')} tok, completion {u.get('completion_tokens')} tok")
PYEOF
}

# the whole disk's sectors read (all nvme namespaces)
sectors() { awk '$3 ~ /^nvme[0-9]+n[0-9]+$/{s+=$6} END{print s+0}' /proc/diskstats; }

mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
printf '%s' "$PROMPT" > "$OUT_DIR/last.prompt"
log "=== pool A/B: промпт $PROMPT_CHARS символов из $PROMPT_FILE, max_tokens $MAX_TOKENS ==="
log "    варианты: ${VARIANTS[*]} (каждый: прогрев-запрос, затем замеренный повтор)"
log ""

declare -A DELTA DECLINE
for V in "${VARIANTS[@]}"; do
    log "=== пул $V MiB ==="
    set_pool_mib "$V"
    stop_server
    start_server
    if ! wait_ready; then
        log "  сервер не поднялся — пропуск"
        DELTA[$V]="-1"; DECLINE[$V]="СЕРВЕР НЕ ПОДНЯЛСЯ"
        continue
    fi
    log "  прогрев (тот же промпт, ответ в $OUT_DIR/warm-$V.txt) ..."
    ask "$OUT_DIR/warm-$V.txt" || true
    sleep 3
    local_b=$(sectors)
    log "  замеренный прогон ..."
    ask "$OUT_DIR/run-$V.txt" || true
    sleep 2
    local_a=$(sectors)
    d=$(( (local_a - local_b) * 512 / 1024 / 1024 ))
    DELTA[$V]=$d
    DECLINE[$V]="$(grep '\[strata\] done:' "$OUT_DIR/server.log" | tail -1)"
    log "  SSD за замеренный прогон: $d MiB"
    log "  done: ${DECLINE[$V]}"
    log ""
done

restore_config
log "=== ИТОГ ==="
for V in "${VARIANTS[@]}"; do
    log "  пул $V MiB: SSD ${DELTA[$V]:-?} MiB   (${DECLINE[$V]:-})"
done
log ""
log "  вердикт: у большего пула SSD-дельта заметно ниже и хвостовая tok/s выше"
log "  → поднять --ram-experts-mib; разница в шуме → сидеть на своём."
log ""
log "восстанавливаю конфиг и сервер на нём ..."
stop_server
start_server
wait_ready || log "  (!) сервер не поднялся — запусти ./setup.sh руками"
log "готово. ответы: $OUT_DIR/run-*.txt"
