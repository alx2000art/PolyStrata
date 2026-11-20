#!/usr/bin/env bash
# ============================================================================
# eagle_turn_check.sh — решающий тест структуры: голова на РЕАЛЬНОМ assistant-ходе.
#
# Гипотеза: тренировочные дампы сняты фидером одним user-message (текст внутри
# сообщения), а live-декод идёт в структуре чата (user-промпт -> заголовок
# assistant -> ответ).  Скрипт читает ТОТ ЖЕ текст запросом с реальным ходом
# assistant (messages=[user, assistant]) и меряет офлайн-цепочку ТОЛЬКО на
# строках ответа (STRATA_EVAL_ROWS).
#
#   chain ~50-80% на строках ответа -> структура не виновата, баг в исполнении раунда
#   chain ~1-5%  на строках ответа -> КОРЕНЬ: структура; лечится дампами диалогов
#                                   как настоящих разговоров
#
# Запуск: ./eagle_turn_check.sh   (конфиг бэкапится/восстанавливается, сервер
#         поднимается/гасится по образцу bench_sweep)
# Результаты: OUT_DIR (answer.txt, dump, eval.log)
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PROMPT_FILE="${PROMPT_FILE:-$HOME/prompts/doc-2k.txt}"
ANSWER_FILE="${ANSWER_FILE:-$STRATA_DIR/eagle_diag_full/answer.txt}"
HEAD="${HEAD:-$HOME/llama_server/models/eagle3-ru-ft6.safetensors}"
HEAD_CFG="${HEAD_CFG:-$HOME/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.config.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/eagle_turn_check}"
TURN_DIR="${TURN_DIR:-$HOME/eagle_taps_turn}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -f "$PROMPT_FILE" ]] || die "нет промпта $PROMPT_FILE"
[[ -f "$ANSWER_FILE" ]] || die "нет ответа $ANSWER_FILE (прогони сначала ./eagle_diag_full.sh)"
[[ -f "$HEAD" ]] || die "нет головы $HEAD"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет .venv"
mkdir -p "$OUT_DIR" "$TURN_DIR"
rm -f "$TURN_DIR"/taps-*.bin 2>/dev/null || true

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
ENG_LOG="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("log") or "")
PY
)"
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"
OUT_PID="$OUT_DIR/server.pid"

log "=============================================================="
log "eagle_turn_check: turn-структура (user+assistant) -> chain на строках ответа"
log "промпт: $PROMPT_FILE  ответ: $ANSWER_FILE  out: $OUT_DIR"
log "=============================================================="

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
  log "  жду готовности ..."
  while (( SECONDS < deadline )); do
    port_open || { sleep 5; continue; }
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
    printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' && { log "  модель загружена"; return 0; }
    sleep 5
  done
  die "сервер не готов (лог: $OUT_DIR/server.log.last)"
}

BACKUP="$OUT_DIR/config.backup.json"
cp "$CONFIG" "$BACKUP"
restore_config() { cp "$BACKUP" "$CONFIG"; }
set_dump_env() {
  python3 - "$CONFIG" "$1" <<'PY'
import json, os, sys
p, rdir = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
env = cfg.get("env") or {}
for k in list(env):
    if k.startswith("STRATA_EAGLE_"):
        del env[k]
env["STRATA_EAGLE_DUMP_TAPS"] = rdir
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
print("  env:", {k: v for k, v in env.items() if k.startswith("STRATA_EAGLE_")})
PY
}

log "[1] дамп-режим и рестарт"
set_dump_env "$TURN_DIR"
stop_server
start_server
wait_ready

log "[2] запрос с РЕАЛЬНЫМ ходом assistant (messages=[user, assistant])"
python3 - "$PROMPT_FILE" "$ANSWER_FILE" > "$OUT_DIR/req.json" <<'PY'
import json, sys
prompt = open(sys.argv[1], encoding="utf-8").read()
ans = open(sys.argv[2], encoding="utf-8").read()
print(json.dumps({"model": "x",
                  "messages": [{"role": "user", "content": prompt},
                               {"role": "assistant", "content": ans}],
                  "max_tokens": 2, "temperature": 0.0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "reasoning_effort": "none", "thinking": {"type": "disabled"}},
                 ensure_ascii=False))
PY
MARK=$(stat -c%s "$ENG_LOG" 2>/dev/null || echo 0)
curl -s -m 900 -o "$OUT_DIR/resp.json" "$BASE_URL/chat/completions" \
  -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  -d "@$OUT_DIR/req.json" || die "запрос не прошёл"
grep -c "tap dump: block" <(tail -c +$((MARK + 1)) "$ENG_LOG" 2>/dev/null) 2>/dev/null | sed 's/^/  строк tap dump: /'
N=$(ls "$TURN_DIR"/taps-*.bin 2>/dev/null | wc -l)
log "  turn-дамп: $N файл(ов)"
[[ "$N" -ge 1 ]] || { restore_config; stop_server; die "дамп не записался"; }

log "[3] стоп, конфиг восстанавливается"
restore_config
stop_server

log "[4] ищу начало ответа в дампе (токенайзер движка, 3-окно)"
TD=$(ls -t "$TURN_DIR"/taps-*.bin | head -1)
.venv/bin/python - "$TD" "$ANSWER_FILE" "$TURN_DIR" <<'PY' > "$OUT_DIR/answer_range.txt"
import json, struct, sys
import numpy as np
sys.path.insert(0, "tools")
import strata_tokenizer as ST

td, ans_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
# токены дампа (все блоки, по порядку)
f = open(td, "rb")
magic, ver, n_embd, taps_n, has_emb = struct.unpack("<IIIII", f.read(20))
toks, pos0 = [], None
while True:
    hdr = f.read(12)
    if len(hdr) < 12: break
    p0, rows = struct.unpack("<QI", hdr)
    tk = np.frombuffer(f.read(4 * rows), dtype="<i4")
    f.seek(2 * rows * (taps_n + 1) * n_embd, 1)
    if pos0 is None: pos0 = p0
    toks.extend(int(x) for x in tk)
# токенайзер пака из конфига
import os
cfg = json.load(open(os.path.join("/home/a.makarov/PolyStrata", "strata-dsv4-ud-iq2_xxs.json"), encoding="utf-8-sig"))
tp = None
for a in (cfg.get("args") or []):
    pass
tokdir = cfg.get("tokenizer") or "/home/a.makarov/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer"
tk = ST.Tokenizer.from_dir(tokdir)
ans = tk.encode(open(ans_path, encoding="utf-8").read(), parse_special=False)
ans = [int(x) for x in ans][:12]
# 3-окно
rmap = {}
for k in range(len(toks) - 2):
    rmap.setdefault(tuple(toks[k:k+3]), []).append(k)
start = None
for k in rmap.get(tuple(ans[:3]), []):
    if toks[k:k+12] == ans:
        start = k
        break
if start is None:
    print("0:0")
    print("# начало ответа не найдено — eval по всему дампу", file=sys.stderr)
else:
    lo = pos0 + start
    hi = pos0 + len(toks)
    print(f"{lo}:{hi}")
    print(f"# ответ с pos {lo} (строка {start} из {len(toks)})", file=sys.stderr)
PY
RANGE=$(head -1 "$OUT_DIR/answer_range.txt")
grep '^#' "$OUT_DIR/answer_range.txt" | sed 's/^/  /'
log "  диапазон строк ответа: STRATA_EVAL_ROWS=$RANGE"

log "[5] офлайн-цепочка ТОЛЬКО на строках ответа (turn-структура, полный контекст)"
STRATA_EVAL_ROWS="$RANGE" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$GPU_PY" "$STRATA_DIR/eagle3_ft_train.py" --dump "$TD" --safetensors "$HEAD" \
  --config "$HEAD_CFG" --eval-only --max-rows 512 > "$OUT_DIR/eval.log" 2>&1
grep -E "eval-only" "$OUT_DIR/eval.log" | tail -1 | sed 's/^/  /'

log "=============================================================="
RES=$(grep -oE 'chain \(emb p\+1, h=p\) -> p\+2: [0-9]+ из [0-9]+ \([0-9.]+%\)' "$OUT_DIR/eval.log" | tail -1)
log "chain (turn-структура, строки ответа): ${RES:-НЕ ИЗМЕРЕНО}"
log "для сравнения: read-структура 80.5% · decode-дамп 46.2% · живой движок ~1%"
log "=============================================================="
