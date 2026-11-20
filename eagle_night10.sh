#!/usr/bin/env bash
# ============================================================================
# eagle_night10.sh — ft10 на НОРМАЛЬНЫХ данных, одной ночью.
#
# 1. Чистый текст из 88 PDF (PyMuPDF; склейка порванных строк в абзацы) - домен;
#    (старый corpus_full.txt на 46.7% мохибейк - не используется)
# 2. Стриминг русских датасетов: fineweb-2 rus_Cyrl + Wikipedia ru (срезы);
# 3. Корпус = PDF-текст + датасеты (+ диалоги отдельным дампом, decode-дампы как есть);
# 4. read-дамп корпуса+диалогов (дамп-режим, рестарты по правилам);
# 5. Обучение от оригинальной головы: --chain-depth 3, --noise 0.08, --mask-draft0, 2 эпохи.
#
# Запуск: PDF_DIR=~/docs ./eagle_night10.sh
#   SKIP_DATA=1 ./eagle_night10.sh    # данные уже подготовлены (~/eagle_ru/corpus.txt)
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PDF_DIR="${PDF_DIR:-}"                      # пусто: find по дому
WORK="${WORK:-$HOME/eagle_ru}"              # подготовленные тексты
READ_DIR="${READ_DIR:-$HOME/eagle_taps_n10}"
DLG_DIR="${DLG_DIR:-$HOME/eagle_taps_n10_dlg}"
DEC_DIR="${DEC_DIR:-$HOME/eagle_taps_dec}"
MIX_DIR="${MIX_DIR:-$HOME/eagle_taps_n10_mix}"
BASE_HEAD="${BASE_HEAD:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.safetensors}"
HEAD_CFG="${HEAD_CFG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
OUT_HEAD="${OUT_HEAD:-$HOME/llama_server/models/eagle3-ru-ft10.safetensors}"
FT_LOG="${FT_LOG:-$HOME/ft10_train.log}"
PDF_CHARS="${PDF_CHARS:-4000000}"
WIKI_CHARS="${WIKI_CHARS:-3000000}"
FWEB_CHARS="${FWEB_CHARS:-4000000}"
EPOCHS="${EPOCHS:-2}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"
SKIP_DATA="${SKIP_DATA:-0}"
MIN_FREE_GB="${MIN_FREE_GB:-110}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

FREE_GB=$(df --output=avail -BG /home | tail -1 | tr -dc '0-9')
log "=============================================================="
log "eagle_night10: PDF ${PDF_CHARS} + wiki ${WIKI_CHARS} + fineweb ${FWEB_CHARS} симв"
log "диск: ${FREE_GB} ГБ (нужно >= $MIN_FREE_GB)"
log "=============================================================="
[[ "$FREE_GB" -ge "$MIN_FREE_GB" ]] || die "мало места"
mkdir -p "$WORK" "$READ_DIR" "$MIX_DIR"

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
OUT_PID="$STRATA_DIR/eagle_night10.pid"

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
  log "  сервер pid $(cat "$OUT_PID")"
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

# ------------------------- 1. данные ----------------------------------------
if [[ "$SKIP_DATA" != "1" ]]; then
  log "[data] зависимости ..."
  "$STRATA_DIR/.venv/bin/pip" install -q pymupdf datasets 2>&1 | tail -1 || die "pip"

  log "[data] PDF -> текст (${PDF_CHARS} симв) ..."
  if [[ -z "$PDF_DIR" ]]; then
    PDF_DIR=$(find "$HOME" /mnt /data /srv -maxdepth 5 -type d -name "docs" 2>/dev/null | head -1)
  fi
  PDF_N=0
  [[ -n "$PDF_DIR" && -d "$PDF_DIR" ]] && PDF_N=$(find "$PDF_DIR" -iname "*.pdf" 2>/dev/null | wc -l)
  if [[ "$PDF_N" -eq 0 ]]; then
    # PDF на сервере нет - берём ЧИСТУЮ часть старого корпуса (U+FFFD-строки выкидываются)
    log "  PDF не найдены - чищу старый corpus_full.txt (выкидываю мохибейк)"
    python3 - "$HOME/corpus/corpus_full.txt" "$WORK/pdf.txt" "$PDF_CHARS" <<'PY'
import sys
src, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = open(src, encoding="utf-8", errors="ignore").read()
first = t.find("\ufffd")
clean = t[:first] if first > 0 else t          # до первого битого - гарантированно чисто
lines = [l for l in t[first:].splitlines() if "\ufffd" not in l] if first > 0 else []
text = (clean + "\n" + "\n".join(lines)) if first > 0 else clean
text = text[:limit]
bad = text.count("�")
open(out, "w", encoding="utf-8").write(text)
print(f"  очищено: {len(text)} символов, U+FFFD {bad} ({100*bad/max(len(text),1):.2f}%)")
PY
  else
    log "  PDF найдены: $PDF_N в $PDF_DIR"
    "$STRATA_DIR/.venv/bin/python" - "$PDF_DIR" "$WORK/pdf.txt" "$PDF_CHARS" <<'PY'
import sys
import fitz  # pymupdf

src, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
import glob, os
files = sorted(glob.glob(os.path.join(src, "**", "*.pdf"), recursive=True))
print(f"  PDF: {len(files)} файлов")
chunks, total = [], 0
for f in files:
    if total >= limit:
        break
    try:
        doc = fitz.open(f)
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        print(f"  пропуск {os.path.basename(f)}: {e}")
        continue
    paras, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        cur.append(s)
        if s.endswith((".", "!", "?", ":")):
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    block = "\n".join(paras)
    if len(block.strip()) < 200:
        continue
    chunks.append(block)
    total += len(block)
text = "\n\n".join(chunks)[:limit]
open(out, "w", encoding="utf-8").write(text)
bad = text.count("�")
print(f"  очищено: {len(text)} символов, U+FFFD {bad} ({100*bad/max(len(text),1):.2f}%)")
PY
  fi
  "$STRATA_DIR/.venv/bin/python" - "$PDF_DIR" "$WORK/pdf.txt" "$PDF_CHARS" <<'PY'
import sys
import fitz  # pymupdf

src, out, limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
import glob, os
files = sorted(glob.glob(os.path.join(src, "**", "*.pdf"), recursive=True))
print(f"  PDF: {len(files)} файлов")
chunks, total = [], 0
for f in files:
    if total >= limit:
        break
    try:
        doc = fitz.open(f)
        text = "\n".join(page.get_text("text") for page in doc)
    except Exception as e:
        print(f"  пропуск {os.path.basename(f)}: {e}")
        continue
    # склейка порванных строк: пустая строка или конец предложения закрывают абзац
    paras, cur = [], []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        cur.append(s)
        if s.endswith((".", "!", "?", ":")):
            paras.append(" ".join(cur)); cur = []
    if cur:
        paras.append(" ".join(cur))
    block = "\n".join(paras)
    if len(block.strip()) < 200:
        continue
    chunks.append(block)
    total += len(block)
text = "\n\n".join(chunks)[:limit]
open(out, "w", encoding="utf-8").write(text)
bad = text.count("\ufffd")
print(f"  очищено: {len(text)} символов, U+FFFD {bad} ({100*bad/max(len(text),1):.2f}%)")
PY

  log "[data] Wikipedia ru (${WIKI_CHARS} симв) ..."
  "$STRATA_DIR/.venv/bin/python" - "$WORK/wiki.txt" "$WIKI_CHARS" <<'PY'
import sys
from datasets import load_dataset

out, limit = sys.argv[1], int(sys.argv[2])
ds = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train", streaming=True)
buf, total = [], 0
for row in ds:
    t = (row.get("text") or "").strip()
    if len(t) < 300:
        continue
    buf.append(t)
    total += len(t)
    if total >= limit:
        break
text = "\n\n".join(buf)[:limit]
open(out, "w", encoding="utf-8").write(text)
print(f"  wiki: {len(text)} символов")
PY

  log "[data] fineweb-2 rus_Cyrl (${FWEB_CHARS} симв) ..."
  "$STRATA_DIR/.venv/bin/python" - "$WORK/fweb.txt" "$FWEB_CHARS" <<'PY'
import sys
from datasets import load_dataset

out, limit = sys.argv[1], int(sys.argv[2])
ds = load_dataset("HuggingFaceFW/fineweb-2", "rus_Cyrl", split="train", streaming=True)
buf, total = [], 0
for row in ds:
    t = (row.get("text") or "").strip()
    if len(t) < 300:
        continue
    buf.append(t)
    total += len(t)
    if total >= limit:
        break
text = "\n\n".join(buf)[:limit]
open(out, "w", encoding="utf-8").write(text)
print(f"  fineweb: {len(text)} символов")
PY

  log "[data] сборка корпуса ..."
  cat "$WORK/pdf.txt" "$WORK/wiki.txt" "$WORK/fweb.txt" > "$WORK/corpus.txt" 2>/dev/null || true
  [[ -s "$WORK/corpus.txt" ]] || die "корпус пуст"
  log "  corpus.txt: $(wc -c < "$WORK/corpus.txt") байт"
fi

CORPUS="$WORK/corpus.txt"
[[ -s "$CORPUS" ]] || die "нет $CORPUS"
DIALOGS_TEXT="${DIALOGS_TEXT:-$HOME/eagle_gen/dialogs_corpus.txt}"

# ------------------------- 2. дампы -----------------------------------------
cp "$CONFIG" "$WORK/config.backup.json"
restore() { cp "$WORK/config.backup.json" "$CONFIG"; }
trap restore EXIT
set_dump() {  # $1 = каталог
  python3 - "$CONFIG" "$1" <<'PY'
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
PY
}

log "[dump] корпус ($(wc -c < "$CORPUS") байт) ..."
set_dump "$READ_DIR"
stop_server
start_server
wait_ready
( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$CORPUS" --chars 80000 ) || die "фидер корпуса"

if [[ -f "$DIALOGS_TEXT" ]]; then
  log "[dump] диалоги ..."
  mkdir -p "$DLG_DIR"
  stop_server
  set_dump "$DLG_DIR"
  start_server
  wait_ready
  ( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$DIALOGS_TEXT" --chars 80000 ) || die "фидер диалогов"
fi
stop_server
restore
trap - EXIT
log "[dump] готово, конфиг восстановлен"

# ------------------------- 3. смесь и тренировка ----------------------------
log "[mix] ..."
rm -rf "$MIX_DIR"; mkdir -p "$MIX_DIR"
for f in "$READ_DIR"/taps-*.bin; do [[ -f "$f" ]] && ln -sf "$f" "$MIX_DIR/$(basename "$f")"; done
for f in "$DLG_DIR"/taps-*.bin; do [[ -f "$f" ]] || continue
  for k in 1 2; do ln -sf "$f" "$MIX_DIR/dlg$k-$(basename "$f")"; done; done
for f in "$DEC_DIR"/taps-decode-*.bin; do [[ -f "$f" ]] || continue
  for k in 1 2; do ln -sf "$f" "$MIX_DIR/dec$k-$(basename "$f")"; done; done
log "  ссылок: $(ls "$MIX_DIR" | wc -l)"

pkill -f "strata-poly" 2>/dev/null || true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup "$GPU_PY" "$STRATA_DIR/eagle3_ft_train.py" \
  --dump "$MIX_DIR" --safetensors "$BASE_HEAD" --config "$HEAD_CFG" --out "$OUT_HEAD" \
  --epochs "$EPOCHS" --max-rows 512 --lr 1e-4 --opt8 --chain-depth 3 \
  --noise 0.08 --mask-draft0 > "$FT_LOG" 2>&1 &
TP=$!
log "[train] pid $TP, лог $FT_LOG"
sleep 180
kill -0 "$TP" 2>/dev/null || { tail -20 "$FT_LOG"; die "упало на старте"; }
tail -8 "$FT_LOG"
log "=============================================================="
log "идёт. Утром: tail -3 $FT_LOG -> eval -> конверт -> установка (LM_Q8=0) -> diag"
log "=============================================================="
