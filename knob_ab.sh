#!/usr/bin/env bash
# ============================================================================
# knob_ab.sh - A/B of one numeric config flag, live: does the change help?
#
#   PROMPT_FILE=/path/to/prompt.txt ./knob_ab.sh --stage2-mib 512 256
#   PROMPT_FILE=/path/to/fresh.txt ./knob_ab.sh --ram-experts-mib 6144 12288
#
# For every value the script rewrites the flag in the config, restarts the
# server, runs the prompt once to warm, then runs it again with the meters
# around it: the nvme sectors read, the prompt reading's tail and average
# tok/s (the last and the span of the "reading the prompt" lines), and the
# server's done line.  The config restores and the server comes back on it.
#
# The prompt: the same file serves both values, so the comparison is honest.
# A prompt the model has already read works too (prompt-cache returns its
# prefix; the fresh tail's speeds are what the meters catch).
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/knob_ab}"
MAX_TOKENS="${MAX_TOKENS:-384}"
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT_OFFSET="${PROMPT_OFFSET:-1}"
PROMPT_LIMIT="${PROMPT_LIMIT:-0}"          # 0: the whole file
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

FLAG="${1:-}"
shift || true
VALUES=("$@")
[[ -n "$FLAG" && ${#VALUES[@]} -ge 2 ]] || die "usage: PROMPT_FILE=... ./knob_ab.sh --флаг значение1 значение2 [значение3 ...]"
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
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"

PROMPT="$(awk -v a="$PROMPT_OFFSET" -v b="$PROMPT_LIMIT" 'NR>=a && (b==0 || NR<a+b)' "$PROMPT_FILE")"
[[ -n "$PROMPT" ]] || die "PROMPT_FILE пуст"
PROMPT_CHARS="${#PROMPT}"

BACKUP="$CONFIG.knob-ab-backup-$(date +%Y%m%d-%H%M%S)"
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

# the flag's value lives in the config's args list, right after the flag itself
set_flag() {
    python3 - "$CONFIG" "$FLAG" "$1" <<'PYEOF'
import json, os, sys
path, flag, val = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(path, encoding="utf-8-sig"))
args = cfg["args"] if isinstance(cfg.get("args"), list) else cfg["args"].split()
i = args.index(flag)
args[i + 1] = str(val)
cfg["args"] = args
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
PYEOF
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
u = d.get("usage") or {}
text = d["choices"][0]["message"].get("content") or ""
open(out, "w", encoding="utf-8").write(text + f"\n[elapsed {el:.0f} s, prompt_tokens {u.get('prompt_tokens')}, completion {u.get('completion_tokens')}]\n")
print(f"  elapsed {el:.0f} s, prompt {u.get('prompt_tokens')} tok, completion {u.get('completion_tokens')} tok")
PYEOF
}

sectors() { awk '$3 ~ /^nvme[0-9]+n[0-9]+$/{s+=$6} END{print s+0}' /proc/diskstats; }

# one "reading the prompt:" line -> (tokens, s, tok/s); the numbers carry commas
reading_fields() {  # $1 = the line; prints "tokens s toks"
    printf '%s' "$1" | sed 's/.*reading the prompt: //' | tr -d ',' | awk '{print $1, $5, $9}'
}

mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
log "=== knob A/B: $FLAG ${VALUES[*]} ==="
log "    промпт: $PROMPT_CHARS символов из $PROMPT_FILE, max_tokens $MAX_TOKENS"
log "    каждый вариант: прогрев-запрос, затем замеренный повтор (SSD, tok/s, done)"
log ""

declare -A R_TAIL R_AVG R_SSD R_DONE
for V in "${VALUES[@]}"; do
    log "=== $FLAG $V ==="
    set_flag "$V"
    stop_server
    start_server
    if ! wait_ready; then
        log "  сервер не поднялся — пропуск"
        R_SSD[$V]="?"; R_TAIL[$V]="?"; R_AVG[$V]="?"; R_DONE[$V]="СЕРВЕР НЕ ПОДНЯЛСЯ"
        continue
    fi
    log "  прогрев (ответ в $OUT_DIR/warm-$V.txt) ..."
    ask "$OUT_DIR/warm-$V.txt" || true
    sleep 3
    mark=$(( $(wc -l < "$OUT_DIR/server.log") + 1 ))
    b=$(sectors)
    log "  замеренный прогон ..."
    ask "$OUT_DIR/run-$V.txt" || true
    sleep 2
    a=$(sectors)
    tail -n "+$mark" "$OUT_DIR/server.log" > "$OUT_DIR/segment-$V.log"
    first=$(grep '\[strata\] reading the prompt:' "$OUT_DIR/segment-$V.log" | head -1)
    last=$(grep '\[strata\] reading the prompt:' "$OUT_DIR/segment-$V.log" | tail -1)
    if [[ -n "$first" && -n "$last" ]]; then
        read t1 s1 _ <<< "$(reading_fields "$first")"
        read t2 s2 v2 <<< "$(reading_fields "$last")"
        dt=$(( s2 > s1 ? s2 - s1 : 1 ))
        R_AVG[$V]=$(( (t2 - t1) / dt ))
        R_TAIL[$V]=$v2
    else
        R_AVG[$V]="?"; R_TAIL[$V]="?"
    fi
    R_SSD[$V]=$(( (a - b) * 512 / 1024 / 1024 ))
    R_DONE[$V]="$(grep '\[strata\] done:' "$OUT_DIR/server.log" | tail -1)"
    log "  хвост чтения ${R_TAIL[$V]} tok/s, средняя ${R_AVG[$V]} tok/s, SSD ${R_SSD[$V]} MiB"
    log ""
done

restore_config
log "=== ИТОГ: $FLAG ==="
printf '%s\n' "  значение | хвост tok/s | средняя tok/s | SSD MiB | done"
for V in "${VALUES[@]}"; do
    printf '  %8s | %11s | %13s | %7s | %s\n' "$V" "${R_TAIL[$V]:-?}" "${R_AVG[$V]:-?}" "${R_SSD[$V]:-?}" "${R_DONE[$V]:-}"
done
log ""
log "  вердикт: сравни хвост/среднюю tok/s и SSD между значениями; decode смотри"
log "  в done-строках (tok/s там — decode всей генерации)."
log ""
log "восстанавливаю конфиг и сервер на нём ..."
stop_server
start_server
wait_ready || log "  (!) сервер не поднялся — запусти ./setup.sh руками"
log "готово. ответы: $OUT_DIR/run-*.txt"
