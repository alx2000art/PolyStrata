#!/usr/bin/env bash
# ============================================================================
# eagle_ft_night.sh — ночное дообучение русской головы EAGLE-3 на ГОТОВОЙ смеси.
#
# Дампы НЕ снимаются: берётся ~/eagle_taps_night_mix (64 095 блоков, 3 049 874 строки,
# уже парные read+decode).  Старт — от ОРИГИНАЛЬНОЙ головы, с контекстом движка:
#   --ctx-win 1024   окно внимания = строки предыдущих блоков (как в движке)
#   --chain-depth 3  цепочка догадок
#   --noise 0.08     шумовая аугментация taps
#
# Почему такие числа: пробник ft16 (1 эпоха, ctx-win 1024, без chain) прошёл 3,05 млн
# строк за 16,9 мин (2 948 строк/с), loss 7,51 -> 5,99.  chain-depth 3 добавляет два
# форварда на шаг: ожидаемо 25-40 мин/эпоха, 4 эпохи = 2-2,5 ч.
#
# Запуск (лучше в tmux): ./eagle_ft_night.sh
#   EPOCHS=2 ./eagle_ft_night.sh
#   CONVERT=1 ./eagle_ft_night.sh      # после обучения сразу GGUF для движка
#   DETACH=1 ./eagle_ft_night.sh       # проверить старт и выйти, обучение останется
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
PY="${PY:-$HOME/.gpu-bench/bin/python}"
DUMP="${DUMP:-$HOME/eagle_taps_night_mix}"
BASE_HEAD="${BASE_HEAD:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.safetensors}"
HEAD_CFG="${HEAD_CFG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
OUT_HEAD="${OUT_HEAD:-$HOME/llama_server/models/eagle3-ru-ft17.safetensors}"
OUT_GGUF="${OUT_GGUF:-$HOME/llama_server/models/eagle3-ru-ft17.gguf}"
LOG="${LOG:-$HOME/ft17_night.log}"
EPOCHS="${EPOCHS:-4}"
MAX_ROWS="${MAX_ROWS:-512}"
LR="${LR:-5e-5}"
CTX_WIN="${CTX_WIN:-1024}"
CTX_CHUNK="${CTX_CHUNK:-256}"
CHAIN="${CHAIN:-3}"
NOISE="${NOISE:-0.08}"
CONVERT="${CONVERT:-0}"
DETACH="${DETACH:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

log "=============================================================="
log "eagle_ft_night: $(basename "$DUMP") -> $(basename "$OUT_HEAD")"
log "эпох $EPOCHS, chain $CHAIN, noise $NOISE, ctx-win $CTX_WIN/$CTX_CHUNK, lr $LR"
log "=============================================================="

[[ -f "$BASE_HEAD" ]] || die "нет базовой головы $BASE_HEAD"
[[ -f "$HEAD_CFG" ]] || die "нет конфига головы $HEAD_CFG"
[[ -d "$DUMP" ]] || die "нет дампа $DUMP"
[[ -x "$PY" ]] || die "нет интерпретатора $PY"

FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
log "диск свободно: ${FREE_GB} ГБ (нужно >= $MIN_FREE_GB)"
[[ "$FREE_GB" -ge "$MIN_FREE_GB" ]] || die "мало места: удали старые дампы и повтори"

# Обучение берёт VRAM, поэтому прод-сервер и его движок должны быть остановлены.
if pgrep -f "serve/server.py" >/dev/null || pgrep -f "strata-poly" >/dev/null; then
  log "[*] останавливаю прод-сервер и движок (нужна VRAM) ..."
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  sleep 3
fi
if pgrep -f "strata-poly" >/dev/null; then die "движок всё ещё жив"; fi
FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -dc '0-9')
log "VRAM свободно: ${FREE_VRAM} MiB"
[[ "${FREE_VRAM:-0}" -ge 4000 ]] || die "мало VRAM (${FREE_VRAM} MiB) — кто-то держит карту"

: > "$LOG"
log "[train] старт $(date -Is), лог $LOG"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$PY" -u "$STRATA_DIR/eagle3_ft_train.py" \
  --dump "$DUMP" --safetensors "$BASE_HEAD" --config "$HEAD_CFG" --out "$OUT_HEAD" \
  --epochs "$EPOCHS" --max-rows "$MAX_ROWS" --ctx-win "$CTX_WIN" --ctx-chunk "$CTX_CHUNK" \
  --chain-depth "$CHAIN" --noise "$NOISE" --lr "$LR" --opt8 --mask-draft0 \
  >> "$LOG" 2>&1 &
TP=$!
log "  pid $TP"

sleep 180
kill -0 "$TP" 2>/dev/null || { tail -30 "$LOG"; die "обучение упало на старте"; }
log "[train] жив через 3 минуты:"
tail -6 "$LOG"

if [[ "$DETACH" == "1" ]]; then
  log "=============================================================="
  log "идёт. Утром: tail -3 $LOG (строка 'сохранено:')"
  exit 0
fi

wait "$TP"
RC=$?
log "[train] завершено $(date -Is), код $RC"
tail -4 "$LOG"
[[ "$RC" -eq 0 ]] || die "обучение вернуло $RC"
[[ -f "$OUT_HEAD" ]] || die "нет выходной головы $OUT_HEAD"
ls -la "$OUT_HEAD"

if [[ "$CONVERT" == "1" ]]; then
  log "[convert] $OUT_HEAD -> $OUT_GGUF"
  "$PY" "$STRATA_DIR/tools/eagle3_convert.py" "$OUT_HEAD" "$HEAD_CFG" "$OUT_GGUF" || die "конвертер"
  ls -la "$OUT_GGUF"
fi

log "=============================================================="
log "Утром — сравнить три головы на одних дампах:"
log "  for H in DeepSeek-V4-Flash-0731-eagle3 eagle3-ru-ft16_probe $(basename "$OUT_HEAD" .safetensors); do"
log "    $PY $STRATA_DIR/eagle3_ft_train.py --eval-only --eval-ctx \\"
log "      --dump $DUMP \\"
log "      --safetensors $HOME/llama_server/models/\$H.safetensors \\"
log "      --config $HEAD_CFG 2>&1 | tail -6; done"
log "Затем — живой TEACH: движок с ${OUT_GGUF##*/} (--window 4) и домен-тест ru/en/code."
log "=============================================================="
