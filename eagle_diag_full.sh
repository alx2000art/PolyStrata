#!/usr/bin/env bash
# ============================================================================
# eagle_diag_full.sh — полный пакет проверок драфтера за один запуск.
#
# Что делает (каждый шаг с проверкой, конфиг бэкапится и восстанавливается,
# сервер поднимается/гасится по образцу bench_sweep):
#   1. дамп-режим: env конфига STRATA_EAGLE_DUMP_TAPS (read) + STRATA_EAGLE_DUMP_DECODE
#      (decode), кэши выкл → РЕСТАРТ сервера (env читается на старте!);
#   2. ЖИВОЙ запрос (prompt, 384 токена, thinking off) — ответ сохраняется, из логов
#      берутся живые числа: "the drafter's first guess ... X of Y", "rounds: t1=.. pays=",
#      "drafts A/O accepted ... from drafts";
#   3. READ-дамп того же текста (prompt+ответ) фидером — полный контекст;
#   4. стоп сервера, конфиг восстанавливается;
#   5. СВЕРКА taps: decode-дамп против read-дампа (tools/eagle_taps_verify.py);
#   6. офлайн-замер на read-дампе в ПОЛНОМ контексте (leg1/leg3/chain) — eval-only
#      с STRATA_EVAL_POS на позиции начала ответа.
#
# Запуск:  ./eagle_diag_full.sh
# Перекрытия: PROMPT_FILE=~/prompts/doc-2k.txt MAX_TOKENS=384 HEAD=ft6 ./eagle_diag_full.sh
# Результаты: OUT_DIR (answer.txt, taps-verify.log, eval-read.log, engine.log)
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PROMPT_FILE="${PROMPT_FILE:-$HOME/prompts/doc-2k.txt}"
HEAD="${HEAD:-$HOME/llama_server/models/eagle3-ru-ft6.safetensors}"
HEAD_CFG="${HEAD_CFG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
MAX_TOKENS="${MAX_TOKENS:-384}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/eagle_diag_full}"
READ_DIR="${READ_DIR:-$HOME/eagle_taps_read}"
DEC_DIR="${DEC_DIR:-$HOME/eagle_taps_dec}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -f "$PROMPT_FILE" ]] || die "нет промпта $PROMPT_FILE"
[[ -f "$HEAD" ]] || die "нет головы $HEAD"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv"
mkdir -p "$OUT_DIR" "$READ_DIR" "$DEC_DIR"
rm -f "$READ_DIR"/taps-*.bin "$DEC_DIR"/taps-decode-*.bin 2>/dev/null || true

API_KEY="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("api_key") or "")
PY
)"
[[ -n "$API_KEY" ]] || die "нет api_key"
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
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"
OUT_PID="$OUT_DIR/server.pid"

log "=============================================================="
log "eagle_diag_full: prompt=$PROMPT_FILE  head=$(basename "$HEAD")  out=$OUT_DIR"
log "движок: $BASE_URL  его лог: $ENG_LOG"
log "=============================================================="

# ------------------------- сервер (стиль bench_sweep) ----------------------
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
start_server() {
  : > "$OUT_DIR/server.log.last"
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" >> "$OUT_DIR/server.log.last" 2>&1 & echo $! > "$OUT_PID" )
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
  die "сервер не готов (лог: $OUT_DIR/server.log.last)"
}

# ------------------------- конфиг: бэкап, дамп-режим, восстановление -------
BACKUP="$OUT_DIR/config.backup.json"
cp "$CONFIG" "$BACKUP"
restore_config() { cp "$BACKUP" "$CONFIG"; }
set_dump_env() {
  python3 - "$CONFIG" "$1" "$2" <<'PY'
import json, os, sys
p, rdir, ddir = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = json.load(open(p, encoding="utf-8-sig"))
env = cfg.get("env") or {}
for k in list(env):
    if k.startswith("STRATA_EAGLE_"):
        del env[k]
env["STRATA_EAGLE_DUMP_TAPS"] = rdir
env["STRATA_EAGLE_DUMP_DECODE"] = ddir
args = [str(a) for a in (cfg.get("args") or [])]
out, i = [], 0
while i < len(args):
    if args[i] in ("--prompt-cache", "--conversation-cache-mib") and i + 1 < len(args):
        args[i + 1] = "0"
    out.append(args[i]); i += 1
cfg["args"] = out
cfg["env"] = env
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
print("  env:", {k: v for k, v in env.items() if k.startswith("STRATA_EAGLE_")})
PY
}

ask() {  # $1 = файл ответа; жадный-ish запрос без thinking
  local raw="$1.raw"
  curl -s -m 900 -o "$raw" "$BASE_URL/chat/completions" -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(python3 - "$PROMPT_FILE" "$MAX_TOKENS" <<'PY'
import json, sys
txt = open(sys.argv[1], encoding="utf-8").read()
print(json.dumps({"model": "x", "messages": [{"role": "user", "content": txt}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0.0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "reasoning_effort": "none", "thinking": {"type": "disabled"}}, ensure_ascii=False))
PY
)"
  python3 - "$raw" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
c = d.get("choices", [{}])[0]
open(sys.argv[2], "w", encoding="utf-8").write((c.get("message") or {}).get("content") or "")
print(f"[finish {c.get('finish_reason')}]")
PY
}

# ============================================================================
log "[1] дамп-режим и рестарт"
set_dump_env "$READ_DIR" "$DEC_DIR"
stop_server
start_server
wait_ready
MARK=$(stat -c%s "$ENG_LOG" 2>/dev/null || echo 0)

log "[2] живой запрос (max_tokens=$MAX_TOKENS, thinking off)"
ask "$OUT_DIR/answer.txt" || die "запрос не удался"
log "  ответ: $(wc -c < "$OUT_DIR/answer.txt") байт"
LIVE=$(tail -c +$((MARK + 1)) "$ENG_LOG" 2>/dev/null | grep -E "first guess|rounds:|pays=" | tail -4)
log "  живые числа движка:"; printf '%s\n' "$LIVE" | sed 's/^/    /'
grep -E "from drafts" "$OUT_DIR/server.log.last" | tail -2 | sed 's/^/    /' || true

log "[3] read-дамп того же текста (prompt+ответ)"
cat "$PROMPT_FILE" "$OUT_DIR/answer.txt" > "$OUT_DIR/prompt_answer.txt"
( cd "$STRATA_DIR" && .venv/bin/python eagle_dump_corpus.py --corpus "$OUT_DIR/prompt_answer.txt" --chars 80000 ) \
  || die "фидер не отработал"
RD_N=$(ls "$READ_DIR"/taps-*.bin 2>/dev/null | wc -l)
DC_N=$(ls "$DEC_DIR"/taps-decode-*.bin 2>/dev/null | wc -l)
log "  read-дамп: $RD_N файл(ов), decode-дамп: $DC_N файл(ов)"
[[ "$RD_N" -ge 1 && "$DC_N" -ge 1 ]] || { restore_config; stop_server; die "дампы не записались"; }

log "[4] стоп сервера, конфиг восстанавливается"
restore_config
stop_server

log "[5] сверка taps: decode vs read"
DEC=$(ls -t "$DEC_DIR"/taps-decode-*.bin | head -1)
RD=$(ls -t "$READ_DIR"/taps-*.bin | head -1)
.venv/bin/python "$STRATA_DIR/tools/eagle_taps_verify.py" --decode "$DEC" --read "$RD" \
  > "$OUT_DIR/taps-verify.log" 2>&1
RC=$?
tail -30 "$OUT_DIR/taps-verify.log"
EVAL_POS=$(grep -oE "STRATA_EVAL_POS=[0-9]+" "$OUT_DIR/taps-verify.log" | head -1 | cut -d= -f2)

log "[6] офлайн-замер на read-дампе (полный контекст), STRATA_EVAL_POS=${EVAL_POS:-1088}"
STRATA_EVAL_POS="${EVAL_POS:-1088}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" "$STRATA_DIR/eagle3_ft_train.py" --dump "$RD" --safetensors "$HEAD" \
  --config "$HEAD_CFG" --eval-only --max-rows 512 > "$OUT_DIR/eval-read.log" 2>&1
grep -E "eval-only|склеенных" "$OUT_DIR/eval-read.log" | tail -2 | sed 's/^/    /'

log "=============================================================="
log "ИТОГ:"
log "  живые:   $(printf '%s' "$LIVE" | grep -oE "[0-9]+ of [0-9]+ \([0-9.]+%\)" | tail -1)  (first guess)"
log "  taps:    $(grep -E '^tap 0|^tap  0' "$OUT_DIR/taps-verify.log" | head -1 | sed 's/ *//;s/  */ /g')"
log "  офлайн:  $(grep 'eval-only (top-1' "$OUT_DIR/eval-read.log" | tail -1 | sed 's/.*: //' | cut -c1-160)"
log "  логи:    $OUT_DIR"
log "=============================================================="
exit $RC
