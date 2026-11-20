#!/usr/bin/env bash
# ============================================================================
# ft_dialogs.sh — диалоги → read-дамп → смесь → обучение драфтера, одной командой.
#
# Фазы (конфиг бэкапится и восстанавливается, каждая с проверками):
#   dump:   движок в дамп-режиме.  Урок, на котором обожглись: env движок читает
#           НА СТАРТЕ — правка конфига без рестарта ничего не даёт, и дамп
#           молча не пишется.  Поэтому: правка env → рестарт → проверка, что
#           блоки РЕАЛЬНО появляются (число файлов в каталоге до/после + свежий
#           хвост лога движка на "tap dump: block").
#   mix:    новые дамп-файлы диалогов ×OVERSAMPLE + корпусные дампы (1 из 3).
#   train:  обучение от базового ft (--chain-depth 3), nohup, первые шаги в лог.
#
# Запуск:
#   ./ft_dialogs.sh                     # все фазы
#   ./ft_dialogs.sh --from mix          # блоки уже надампаны
#   ./ft_dialogs.sh --from train        # смесь уже собрана
#   OVERSAMPLE=6 FT_BASE=.../ft7.safetensors FT_OUT=.../ft8.safetensors ./ft_dialogs.sh --from dump
#
# Результаты: DUMP_DIR (блоки), MIX_DIR (ссылки), FT_OUT (веса), FT_LOG (лог тренировки).
# ============================================================================
set -uo pipefail

# ------------------------- настройки ----------------------------------------
STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
DIALOGS="${DIALOGS:-$HOME/eagle_gen/dialogs.jsonl}"
CORPUS_TEXT="${CORPUS_TEXT:-}"          # если пусто — конвертер сделает сам из DIALOGS
DUMP_DIR="${DUMP_DIR:-$HOME/eagle_taps_gen}"
CORPUS_DIR="${CORPUS_DIR:-$HOME/eagle_taps}"
MIX_DIR="${MIX_DIR:-$HOME/eagle_taps_mix}"
OVERSAMPLE="${OVERSAMPLE:-4}"
CORPUS_SHARE="${CORPUS_SHARE:-3}"       # 1 из CORPUS_SHARE корпусных файлов в смесь
CHARS="${CHARS:-80000}"                 # символов на чанк фидера
FT_BASE="${FT_BASE:-$HOME/llama_server/models/eagle3-ru-ft6.safetensors}"
FT_OUT="${FT_OUT:-$HOME/llama_server/models/eagle3-ru-ft7.safetensors}"
FT_CONFIG="${FT_CONFIG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
FT_EPOCHS="${FT_EPOCHS:-1}"
FT_MAX_ROWS="${FT_MAX_ROWS:-512}"
FT_LR="${FT_LR:-7e-5}"
FT_LOG="${FT_LOG:-$HOME/ft7_train.log}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"
FORCE="${FORCE:-0}"
OUT_PID="$STRATA_DIR/ft_dialogs.server.pid"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

FROM="dump"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--from" ]]; then FROM="${2:-}"; shift 2; else shift; fi
done
case "$FROM" in dump|mix|train) ;; *) die "неизвестная --from: $FROM (dump|mix|train)";; esac

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv"
[[ -f "$DIALOGS" ]] || die "нет диалогов $DIALOGS"
[[ -f "$FT_BASE" ]] || die "нет базовых весов $FT_BASE"
[[ -f "$FT_CONFIG" ]] || die "нет конфига головы $FT_CONFIG"

API_KEY="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("api_key") or "")
PY
)"
[[ -n "$API_KEY" ]] || die "нет api_key в $CONFIG"
PORT="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("port") or 8080)
PY
)"
HOST="$(python3 - "$CONFIG" <<'PY'
import json,sys
h=json.load(open(sys.argv[1], encoding="utf-8-sig")).get("host") or "127.0.0.1"
print("127.0.0.1" if h in ("0.0.0.0","::") else h)
PY
)"
MODEL="$(python3 - "$CONFIG" <<'PY'
import json,sys
a=json.load(open(sys.argv[1], encoding="utf-8-sig")).get("args") or []
print(a[a.index("--model")+1] if "--model" in a else "")
PY
)"
ENG_LOG="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("log") or "")
PY
)"
[[ -n "$ENG_LOG" ]] || die "в конфиге нет 'log'"
HEALTH_URL="http://$HOST:$PORT/health"
mkdir -p "$DUMP_DIR" "$MIX_DIR" "$(dirname "$FT_OUT")"

log "=============================================================="
log "ft_dialogs: from=$FROM  дамп=$DUMP_DIR  смесь=$MIX_DIR  oversample=$OVERSAMPLE"
log "тренировка: $FT_OUT от $(basename "$FT_BASE") (epochs=$FT_EPOCHS, lr=$FT_LR)"
log "=============================================================="

# ------------------------- сервер -------------------------------------------
port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }
wait_port_closed() {
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do port_open || return 0; sleep 1; done
  return 1
}
stop_server() {
  log "  останавливаю сервер ..."
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  wait_port_closed && log "  порт закрыт" || log "  не остановился — добиваю"
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do pgrep -f "strata-poly" >/dev/null 2>&1 || break; sleep 1; done
  sleep 2
}
start_server() {  # $1 = файл лога
  : > "$1"
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" >> "$1" 2>&1 & echo $! > "$OUT_PID" )
  log "  сервер запущен (pid $(cat "$OUT_PID"))"
}
wait_ready() {
  local deadline=$((SECONDS + BOOT_TIMEOUT)) body
  log "  жду готовности (${BOOT_TIMEOUT}s) ..."
  while (( SECONDS < deadline )); do
    port_open || { sleep 5; continue; }
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
    printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' && { log "  модель загружена"; return 0; }
    sleep 5
  done
  die "сервер не готов за ${BOOT_TIMEOUT}s (лог: $1)"
}

# ------------------------- конфиг: бэкап и правки ---------------------------
BACKUP="$STRATA_DIR/ft_dialogs.config.backup.json"
cp "$CONFIG" "$BACKUP"
log "[*] конфиг забэкаплен: $BACKUP"
restore_config() { cp "$BACKUP" "$CONFIG"; }

# $1 = каталог дампа | "" — убрать ключ; попутно кэши 0 (дамп) или прод-дефолты
set_dump_env() {
  python3 - "$CONFIG" "$1" <<'PY'
import json, os, sys
p, d = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
env = cfg.get("env") or {}
for k in list(env):
    if k.startswith("STRATA_EAGLE_"):
        del env[k]
if d:
    env["STRATA_EAGLE_DUMP_TAPS"] = d
args = [str(a) for a in (cfg.get("args") or [])]
want_cache = "0" if d else None      # в дамп-режиме кэши выключены, в проде — дефолты движка
out, i = [], 0
while i < len(args):
    if args[i] in ("--prompt-cache", "--conversation-cache-mib") and i + 1 < len(args):
        if want_cache is None:
            i += 2; continue          # вернуть дефолты = убрать флаги
        args[i + 1] = want_cache
    out.append(args[i]); i += 1
if want_cache is not None:
    for k in ("--prompt-cache", "--conversation-cache-mib"):
        if k not in out:
            out += [k, want_cache]
cfg["args"] = out
cfg["env"] = env
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
print("  env: " + (" ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(пусто)"))
PY
}

dump_dir_files() { ls "$DUMP_DIR"/taps-*.bin 2>/dev/null | wc -l; }

# ============================================================================
# ФАЗА dump
# ============================================================================
if [[ "$FROM" == "dump" ]]; then
  log "[dump] конвертирую диалоги в текст ..."
  if [[ -z "$CORPUS_TEXT" ]]; then
    CORPUS_TEXT="$HOME/eagle_gen/dialogs_corpus.txt"
    ( cd "$STRATA_DIR" && python3 dialogs_to_corpus.py "$DIALOGS" "$CORPUS_TEXT" ) \
      || die "конвертер диалогов не отработал"
  fi
  [[ -s "$CORPUS_TEXT" ]] || die "пустой текст корпуса: $CORPUS_TEXT"
  log "[dump] текст: $(stat -c%s "$CORPUS_TEXT") байт"

  BEFORE=$(dump_dir_files)
  set_dump_env "$DUMP_DIR"
  stop_server
  start_server "$STRATA_DIR/ft_dialogs.engine.dump.log"
  wait_ready "$STRATA_DIR/ft_dialogs.engine.dump.log"
  MARK=$(stat -c%s "$ENG_LOG" 2>/dev/null || echo 0)

  log "[dump] фидер: $CHARS символов на чанк (смотри строки 'tap dump: block' в логе движка)"
  ( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$CORPUS_TEXT" --chars "$CHARS" ) \
    || die "фидер не отработал"

  AFTER=$(dump_dir_files)
  NEW=$((AFTER - BEFORE))
  log "[dump] файлов в $DUMP_DIR: $BEFORE -> $AFTER (новых $NEW)"
  tail -c +$((MARK + 1)) "$ENG_LOG" 2>/dev/null | grep -c "tap dump: block" || true
  TAPS=$(tail -c +$((MARK + 1)) "$ENG_LOG" 2>/dev/null | grep -c "tap dump: block" || true)
  if [[ "$NEW" -lt 1 || "$TAPS" -lt 1 ]]; then
    restore_config
    die "дамп не писал: новых файлов $NEW, строк 'tap dump' в логе $TAPS — движок не знал про каталог (рестарт после правки env?)"
  fi
  log "[dump] OK: $NEW новых блоков, $TAPS строк 'tap dump' в логе"

  set_dump_env ""
  restore_config
  stop_server
  log "[dump] конфиг восстановлен, сервер остановлен (GPU для тренировки)"
  FROM="mix"
fi

# ============================================================================
# ФАЗА mix
# ============================================================================
if [[ "$FROM" == "mix" || "$FROM" == "dump" ]]; then
  GEN_LIST=$(ls -t "$DUMP_DIR"/taps-*.bin 2>/dev/null | head -50)
  [[ -n "$GEN_LIST" ]] || die "в $DUMP_DIR нет дампов — сначала фаза dump"
  if [[ "$FORCE" == "1" ]]; then rm -rf "$MIX_DIR"; mkdir -p "$MIX_DIR"; fi
  mkdir -p "$MIX_DIR"
  rm -f "$MIX_DIR"/gen*-taps-*.bin 2>/dev/null || true
  GEN_N=0
  while read -r f; do
    b=$(basename "$f")
    for i in $(seq 1 "$OVERSAMPLE"); do ln -sf "$f" "$MIX_DIR/gen$i-$b"; GEN_N=$((GEN_N+1)); done
  done <<< "$GEN_LIST"
  CORPUS_N=0
  ls "$CORPUS_DIR"/taps-2026*.bin 2>/dev/null | grep -v decode | grep -v "$(basename "$DUMP_DIR")" \
    | awk -v s="$CORPUS_SHARE" 'NR%s==1' | while read -r f; do ln -sf "$f" "$MIX_DIR/"; done
  CORPUS_N=$(ls "$MIX_DIR" | grep -vc "^gen")
  TOTAL=$(ls "$MIX_DIR" | wc -l)
  log "[mix] диалоговых ссылок $GEN_N (×$OVERSAMPLE) + корпусных $CORPUS_N (1 из $CORPUS_SHARE) = $TOTAL"
  [[ "$TOTAL" -ge 10 ]] || die "смесь подозрительно мала: $TOTAL"
  FROM="train"
fi

# ============================================================================
# ФАЗА train
# ============================================================================
if true; then
  pgrep -f "strata-poly" >/dev/null 2>&1 && die "движок ещё работает — останови его перед тренировкой (GPU занят)"
  log "[train] запускаю обучение -> $FT_OUT"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup "$GPU_PY" "$STRATA_DIR/eagle3_ft_train.py" \
    --dump "$MIX_DIR" \
    --safetensors "$FT_BASE" \
    --config "$FT_CONFIG" \
    --out "$FT_OUT" \
    --epochs "$FT_EPOCHS" --max-rows "$FT_MAX_ROWS" --lr "$FT_LR" --opt8 --chain-depth 3 \
    > "$FT_LOG" 2>&1 &
  TRAIN_PID=$!
  log "[train] pid $TRAIN_PID, лог: $FT_LOG"
  sleep 120
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    tail -20 "$FT_LOG"
    die "тренировка упала на старте (лог: $FT_LOG)"
  fi
  tail -8 "$FT_LOG"
  log "[train] идёт. Ожидание конца: tail -3 $FT_LOG (строка 'сохранено: ...')"
  log "после сохранения: eval-only -> конверт -> установка -> EAGLE_STATS diag"
fi
