#!/usr/bin/env bash
# ============================================================================
# eagle_probe.sh - ОДИН прогон сверки головы EAGLE-3: движок против тренера.
#
#   REBUILD=1 ./eagle_probe.sh        # пересобрать движок (нужно после правок C++)
#   ./eagle_probe.sh                  # обычный прогон
#   ROW=500 PROMPT_FILE=~/prompts/doc-30k.txt ./eagle_probe.sh
#
# Что делает:
#  1) git pull (и перезапускает себя, если скрипт обновился);
#  2) прописывает в env конфига DUMP_TAPS, DUMP=<file>, DUMP_ROW=<ROW>, выключает кэши;
#  3) пересобирает движок (REBUILD=1), стартует сервер, дожидается health;
#  4) кормит промпт фидером (read-строки и стадии одной строки уезжают в файлы);
#  5) останавливает сервер, гоняет eagle3_compare.py на строке ROW;
#  6) печатает rope-строки движка и тренера и таблицу cos по стадиям.
#
# После прогона конфиг остаётся в дамп-режиме (кэши выключены) - для прод-работы
# верните кэши и уберите STRATA_EAGLE_DUMP* руками.
# ============================================================================
set -uo pipefail
PROBE_VER="v4 2026-11-18 (q/k/v stages)"; printf "eagle_probe %s
" "$PROBE_VER"

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PY="${PY:-$HOME/.gpu-bench/bin/python}"
PROMPT_FILE="${PROMPT_FILE:-$HOME/prompts/doc-2k.txt}"
ROW="${ROW:-200}"
REBUILD="${REBUILD:-0}"
SATEF="${SATEF:-$HOME/llama_server/models/eagle3-ru-ft3.safetensors}"
HCONFIG="${HCONFIG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
DUMPDIR="${DUMPDIR:-$HOME/eagle_taps}"
EDUMP="${EDUMP:-$HOME/eagle_dump_row.bin}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/diag_run}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -f "$PROMPT_FILE" ]] || die "нет промпта: PROMPT_FILE=$PROMPT_FILE"
[[ -f "$SATEF" ]] || die "нет головы: SATEF=$SATEF"
mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"

API_KEY="$(python3 -c "
import json,sys
try: print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('api_key') or '')
except Exception: print('')" "$CONFIG")"
[[ -n "$API_KEY" ]] || die "нет API-ключа в конфиге"
PORT="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('port') or 8080)" "$CONFIG")"
HOST="$(python3 -c "
import json,sys
h=json.load(open(sys.argv[1], encoding='utf-8-sig')).get('host') or '127.0.0.1'
print('127.0.0.1' if h in ('0.0.0.0','::') else h)" "$CONFIG")"
LOG="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8-sig')).get('log') or '')" "$CONFIG")"
HEALTH_URL="http://$HOST:$PORT/health"

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
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" >> "$OUT_DIR/server.log.last" 2>&1 & \
        echo $! > "$OUT_DIR/server.pid" )
    log "  сервер запущен (pid $(cat "$OUT_DIR/server.pid"))"
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

log "=== 1/5: git pull ==="
SELF="$(basename "$0")"
BEFORE=$(git -C "$STRATA_DIR" hash-object "$STRATA_DIR/$SELF" 2>/dev/null || echo x)
git -C "$STRATA_DIR" stash push -m "eagle_probe auto" >/dev/null 2>&1 || true
git -C "$STRATA_DIR" pull --ff-only 2>&1 | tail -2
AFTER=$(git -C "$STRATA_DIR" hash-object "$STRATA_DIR/$SELF" 2>/dev/null || echo y)
if [[ "$BEFORE" != "$AFTER" ]]; then
    log "  $SELF обновился при pull — перезапускаю себя целиком"
    exec "$STRATA_DIR/$SELF" "$@"
fi

log "=== 2/5: env дампа (DUMP_TAPS, DUMP_ROW=$ROW, кэши выкл.) ==="
python3 - "$CONFIG" "$DUMPDIR" "$EDUMP" "$ROW" <<'PYEOF'
import json, os, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
env = cfg.setdefault('env', {})
env['STRATA_EAGLE_DUMP_TAPS'] = sys.argv[2]
env['STRATA_EAGLE_DUMP'] = sys.argv[3]
env['STRATA_EAGLE_DUMP_ROW'] = sys.argv[4]
for k in ('STRATA_EAGLE_DUMP_ROW0', 'STRATA_EAGLE_TEACH', 'STRATA_EAGLE_TEACH_PERM',
          'STRATA_EAGLE_TOP5', 'STRATA_EAGLE_STATS'):
    env.pop(k, None)
args = [str(a) for a in (cfg.get('args') or [])]
for k, v in (('--conversation-cache-mib', '0'), ('--prompt-cache', '0')):
    i = args.index(k) if k in args else -1
    if i >= 0 and i + 1 < len(args):
        args[i + 1] = v
    else:
        args += [k, v]
cfg['args'] = args
tmp = sys.argv[1] + '.tmp'
with open(tmp, 'w', encoding='utf-8') as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, sys.argv[1])
print(f"  DUMP_TAPS={sys.argv[2]} DUMP={sys.argv[3]} ROW={sys.argv[4]}")
PYEOF

if [[ "$REBUILD" == 1 ]]; then
    log "=== 3/5: пересборка движка (лог: $OUT_DIR/build.log) ==="
    stop_server
    ( cd "$STRATA_DIR" && ./setup.sh --update --build --rebuild-engine --cuda 13 --no-start --yes </dev/null >> "$OUT_DIR/build.log" 2>&1 )
    rc=$?
    tail -5 "$OUT_DIR/build.log"
    (( rc != 0 )) && die "сборка кончилась с ошибкой $rc"
else
    log "=== 3/5: пересборка пропущена (REBUILD=1 чтобы собрать) ==="
fi

log "=== 4/5: сервер + чтение промпта ==="
stop_server
start_server
wait_ready || die "сервер не поднялся — смотри $OUT_DIR/server.log.last"
ELOG="$OUT_DIR/server.log.last"
[[ -n "$LOG" && -f "$LOG" ]] && ELOG="$LOG"
log "-- версия: $PROBE_VER"
log "-- rope движка:"
grep -h "eagle3 rope" "$ELOG" 2>/dev/null | tail -1 | sed 's/^/   /' || true

NEW0=$(ls -t "$DUMPDIR"/taps-*.bin 2>/dev/null | head -1 || true)
( cd "$STRATA_DIR" && "$PY" eagle_dump_corpus.py --corpus "$PROMPT_FILE" --chars 80000 ) || die "фидер не отработал"
NEW=$(ls -t "$DUMPDIR"/taps-*.bin 2>/dev/null | head -1 || true)
if [[ -z "$NEW" || "$NEW" == "${NEW0:-}" ]]; then die "новый taps-файл не появился — дамп не писался (env?)"; fi
log "  новый дамп: $NEW"
[[ -s "$EDUMP" ]] || die "движковый дамп $EDUMP не создан (STRATA_EAGLE_DUMP?)"

log "=== 5/5: сервер стоп, сверка строки $ROW ==="
stop_server
"$PY" "$STRATA_DIR/eagle3_compare.py" --dump "$NEW" --engine-dump "$EDUMP" \
    --safetensors "$SATEF" --config "$HCONFIG" --row "$ROW" > "$OUT_DIR/compare_row$ROW.log" 2>&1 || true
head -2 "$OUT_DIR/compare_row$ROW.log" | sed 's/^/   /'
grep -A14 "строка $ROW " "$OUT_DIR/compare_row$ROW.log" | sed 's/^/   /' || tail -20 "$OUT_DIR/compare_row$ROW.log"
grep -A10 "перебор attention" "$OUT_DIR/compare_row$ROW.log" | sed 's/^/   /' || true
grep "стадии движка" "$OUT_DIR/compare_row$ROW.log" | sed 's/^/   /' || true
log ""
log "готово: $OUT_DIR/compare_row$ROW.log (полный лог сверки)"
log "внимание: конфиг остался в дамп-режиме (DUMP_TAPS/DUMP, кэши выключены)."
