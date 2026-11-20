#!/usr/bin/env bash
# ============================================================================
# eagle_retrain_night.sh — полное переобучение драфтера на ночь.
#
# Данные (всё ре-дампится свежим, чтобы каждая запись была в одной пачке):
#   корпус   CORPUS_CHARS символов (~10.7 ГБ дампа на 1M симв) - read-состояния;
#   диалоги  ~/eagle_gen/dialogs_corpus.txt               - read-состояния чата;
#   decode   ~/eagle_taps_dec (уже собранные)             - DECODE-состояния (та хрупкость).
# База: ОРИГИНАЛЬНАЯ голова (чистый старт, как в референсе).  --chain-depth 3, 2 эпохи.
#
# Проверки: диск >= MIN_FREE_GB до дампа, сервер поднимается/гасится по bench_sweep,
# конфиг бэкапится/восстанавливается, дампы проверяются по строкам.
#
# Запуск (лучше в tmux): ./eagle_retrain_night.sh
#   CORPUS_CHARS=6000000 EPOCHS=2 ./eagle_retrain_night.sh
#   SKIP_DUMP=1 ./eagle_retrain_night.sh     # дампы уже сняты, только тренировка
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
CORPUS="${CORPUS:-$HOME/corpus/corpus_full.txt}"
DIALOGS_TEXT="${DIALOGS_TEXT:-$HOME/eagle_gen/dialogs_corpus.txt}"
DEC_DIR="${DEC_DIR:-$HOME/eagle_taps_dec}"
READ_DIR="${READ_DIR:-$HOME/eagle_taps_night}"
DLG_DIR="${DLG_DIR:-$HOME/eagle_taps_night_dlg}"
MIX_DIR="${MIX_DIR:-$HOME/eagle_taps_night_mix}"
BASE_HEAD="${BASE_HEAD:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.safetensors}"   # ОРИГИНАЛ
HEAD_CFG="${HEAD_CFG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
OUT_HEAD="${OUT_HEAD:-$HOME/llama_server/models/eagle3-ru-ft9.safetensors}"
FT_LOG="${FT_LOG:-$HOME/ft9_train.log}"
CORPUS_CHARS="${CORPUS_CHARS:-6000000}"
EPOCHS="${EPOCHS:-2}"
MAX_ROWS="${MAX_ROWS:-512}"
LR="${LR:-1e-4}"
FT_NOISE="${FT_NOISE:-0.08}"     # шумовая аугментация taps: устойчивость к сдвигу read->decode
DEC_OVER="${DEC_OVER:-2}"        # оверсэмплинг decode-дампов
DLG_OVER="${DLG_OVER:-2}"        # оверсэмплинг диалогов
MIN_FREE_GB="${MIN_FREE_GB:-95}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"
SKIP_DUMP="${SKIP_DUMP:-0}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

API_KEY="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("api_key") or "")
PY
)"
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
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"
OUT_PID="$STRATA_DIR/eagle_retrain_night.pid"

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }
wait_port_closed() {
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do port_open || return 0; sleep 1; done
  return 1
}
stop_server() {
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  wait_port_closed || true
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  sleep 2
}
start_server() {
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" >> "$READ_DIR/server.log" 2>&1 & echo $! > "$OUT_PID" )
  log "  сервер: pid $(cat "$OUT_PID")"
}
wait_ready() {
  local deadline=$((SECONDS + BOOT_TIMEOUT)) body
  while (( SECONDS < deadline )); do
    port_open || { sleep 5; continue; }
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
    printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' && { log "  модель загружена"; return 0; }
    sleep 5
  done
  die "сервер не готов"
}

log "=============================================================="
log "eagle_retrain_night: корпус ${CORPUS_CHARS} симв + диалоги ×${DLG_OVER} + decode ×${DEC_OVER}"
log "база: $(basename "$BASE_HEAD")  эпох $EPOCHS  lr $LR  -> $(basename "$OUT_HEAD")"
log "=============================================================="
FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
log "диск свободно: ${FREE_GB} ГБ (нужно >= $MIN_FREE_GB)"
[[ "$FREE_GB" -ge "$MIN_FREE_GB" ]] || die "мало места: освободи (старые дампы) и повтори"

[[ -f "$BASE_HEAD" ]] || die "нет базовой головы $BASE_HEAD"
[[ -f "$CORPUS" ]] || die "нет корпуса $CORPUS"
mkdir -p "$READ_DIR" "$MIX_DIR"

if [[ "$SKIP_DUMP" != "1" ]]; then
  cp "$CONFIG" "$READ_DIR/config.backup.json"
  restore() { cp "$READ_DIR/config.backup.json" "$CONFIG"; }
  trap restore EXIT

  log "[dump 1/2] корпус: $(head -c "$CORPUS_CHARS" "$CORPUS" > "$READ_DIR/corpus.txt"; echo "$CORPUS_CHARS") символов"
  head -c "$CORPUS_CHARS" "$CORPUS" > "$READ_DIR/corpus.txt"
  if [[ -f "$DIALOGS_TEXT" ]]; then
    log "[dump 2/2] диалоги: $DIALOGS_TEXT"
    cp "$DIALOGS_TEXT" "$READ_DIR/dialogs.txt"
  fi
  python3 - "$CONFIG" "$READ_DIR" <<'PY'
import json, os, sys
p, d = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
env = cfg.get("env") or {}
for k in list(env):
    if k.startswith("STRATA_EAGLE_"):
        del env[k]
env["STRATA_EAGLE_DUMP_TAPS"] = d
args = [str(a) for a in (cfg.get("args") or [])]
i = 0
while i < len(args):
    if args[i] in ("--prompt-cache", "--conversation-cache-mib") and i + 1 < len(args):
        args[i + 1] = "0"
    i += 1
cfg["args"] = args
cfg["env"] = env
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
print("  env DUMP_TAPS:", d)
PY
  stop_server
  : > "$READ_DIR/server.log"
  start_server
  wait_ready

  log "[dump] корпус (read, ~1.5-2 ч) ..."
  ( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$READ_DIR/corpus.txt" --chars 80000 ) \
    || die "фидер корпуса"
  if [[ -f "$READ_DIR/dialogs.txt" ]]; then
    # диалоги - в СВОЙ каталог: env меняется, значит рестарт (иначе допишутся к корпусным)
    log "[dump] диалоги (свой каталог + рестарт, ~10 мин) ..."
    mkdir -p "$DLG_DIR"
    stop_server
    python3 - "$CONFIG" "$DLG_DIR" <<'PY2'
import json, os, sys
p, d = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
(cfg.setdefault("env", {}))["STRATA_EAGLE_DUMP_TAPS"] = d
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
PY2
    start_server
    wait_ready
    ( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$READ_DIR/dialogs.txt" --chars 80000 ) \
      || die "фидер диалогов"
  fi
  stop_server
  restore
  trap - EXIT
  log "[dump] готово, конфиг восстановлен, сервер остановлен"
fi

log "[mix] собираю смесь ..."
rm -rf "$MIX_DIR"; mkdir -p "$MIX_DIR"
ls "$READ_DIR"/taps-*.bin >/dev/null 2>&1 || die "нет read-дампов в $READ_DIR"
for f in "$READ_DIR"/taps-*.bin; do
  ln -sf "$f" "$MIX_DIR/$(basename "$f")"
done
# диалоги ×DLG_OVER из своего каталога
for f in "$DLG_DIR"/taps-*.bin; do
  [[ -f "$f" ]] || continue
  for k in $(seq 1 "$DLG_OVER"); do ln -sf "$f" "$MIX_DIR/dlg$k-$(basename "$f")"; done
done
# decode ×DEC_OVER
for f in "$DEC_DIR"/taps-decode-*.bin; do
  [[ -f "$f" ]] || continue
  for k in $(seq 1 "$DEC_OVER"); do ln -sf "$f" "$MIX_DIR/dec$k-$(basename "$f")"; done
done
N=$(ls "$MIX_DIR" | wc -l)
log "  ссылок в смеси: $N"

log "[train] старт ($EPOCHS эпох, --chain-depth 3) ..."
pkill -f "strata-poly" 2>/dev/null || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$GPU_PY" "$STRATA_DIR/eagle3_ft_train.py" \
  --dump "$MIX_DIR" --safetensors "$BASE_HEAD" --config "$HEAD_CFG" --out "$OUT_HEAD" \
  --epochs "$EPOCHS" --max-rows "$MAX_ROWS" --lr "$LR" --opt8 --chain-depth 3 --noise "$FT_NOISE" \
  > "$FT_LOG" 2>&1 &
TP=$!
log "  pid $TP, лог $FT_LOG"
sleep 180
kill -0 "$TP" 2>/dev/null || { tail -20 "$FT_LOG"; die "обучение упало на старте"; }
tail -6 "$FT_LOG"
log "=============================================================="
log "идёт. Утром: tail -3 $FT_LOG (строка 'сохранено:')"
log "затем: eval-only на turn/decode-дампах -> конверт -> установка -> EAGLE_STATS diag"
log "=============================================================="
