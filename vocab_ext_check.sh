#!/usr/bin/env bash
# ============================================================================
# vocab_ext_check.sh — полная проверка механизма --vocab-ext (Ubuntu).
#
# Что проверяет:
#   Сайдкар с ZERO-строками (новые id без расширенного токенайзера недостижимы)
#   обязан давать ОТВЕТЫ БИТ-В-БИТ идентичные прогону без сайдкара.
#   Это доказывает, что канал прозрачен, и на него можно класть настоящий
#   TokenAdapt-синтез (фаза 2).
#
# Методика:
#   A (with):    в args конфига подставляется --vocab-ext, рестарт, запрос
#                temperature=0 (жадный = детерминизм), ответ -> OUT/A.txt;
#                из лога берётся строка "vocab-ext: N rows over the main V".
#   B (without): пара --vocab-ext удаляется из args, рестарт, тот же запрос
#                -> OUT/B.txt.
#   verdict:     cmp A B; при расхождении — первый разъехавшийся байт.
#   В конце конфиг восстанавливается из бэкапа, сервер остаётся на исходных
#   настройках (сайдкар-пара убирается).
#
# Предпосылки:
#   - бинарник собран из ветки feat/vocab-ext (проверка: strings | grep vocab-ext);
#   - сайдкар строится сам (tools/make_vocab_sidecar.py --n $N) при отсутствии.
#
# Запуск (движок на время проверки останавливается/перезапускается дважды):
#   ./vocab_ext_check.sh
# Переопределения — переменными окружения:
#   N=8192 SIDECAR=~/models/ext.gguf PROMPT="..." MAX_TOKENS=300 ./vocab_ext_check.sh
#   FORCE=1 ./vocab_ext_check.sh      # перезаписать готовые результаты
# Результаты: OUT_DIR/A.txt B.txt engine-a.log line, OUT_DIR/verdict.txt
# ============================================================================
set -uo pipefail

# ------------------------- настройки (перекрываются env) --------------------
STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/vocab_ext_check}"
N="${N:-4096}"                       # строк в сайдкаре
SIDECAR="${SIDECAR:-$HOME/models/vocab_ext_test.gguf}"
PROMPT="${PROMPT:-Объясни, что такое MoE-модель: routing, эксперты, и почему это экономит вычисления. Два абзаца.}"
MAX_TOKENS="${MAX_TOKENS:-220}"
TEMPERATURE="${TEMPERATURE:-0.0}"    # жадный запрос = детерминизм, основа сравнения
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
LOG_MARK=0
STOP_WAIT="${STOP_WAIT:-90}"
FORCE="${FORCE:-0}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv — запустите ./setup.sh"
MARKERS=$(strings "$STRATA_DIR/engine-poly/strata-poly" 2>/dev/null | grep -c vocab-ext || true)
[[ "$MARKERS" -ge 1 ]] || die "бинарник без vocab-ext — соберите ветку feat/vocab-ext (rm engine.cpp.o, cmake --build, cp)"

mkdir -p "$OUT_DIR"
log "=============================================================="
log "vocab_ext_check: N=$N  sidecar=$SIDECAR  temperature=$TEMPERATURE"
log "конфиг: $CONFIG  результаты: $OUT_DIR"
log "=============================================================="

# ------------------------- ключ/хост/порт/модель из конфига -----------------
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
BASE_URL="http://$HOST:$PORT/v1"
HEALTH_URL="${BASE_URL%/v1}/health"
command -v curl >/dev/null || die "нужен curl"
ENG_LOG="$(python3 - "$CONFIG" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("log") or "")
PY
)"
[[ -n "$ENG_LOG" ]] || die "в конфиге нет 'log' — некуда grep'ать строки движка"
log "движок: $BASE_URL  модель: $MODEL  его лог: $ENG_LOG"

# ------------------------- сайдкар ------------------------------------------
if [[ ! -s "$SIDECAR" || "$FORCE" == "1" ]]; then
  log "[*] строю сайдкар ($N zero-строк) ..."
  ( cd "$STRATA_DIR" && python3 tools/make_vocab_sidecar.py \
      --base "$MODEL" --n "$N" --out "$SIDECAR" ) || die "писатель сайдкара не отработал"
else
  log "[*] сайдкар уже есть: $SIDECAR ($(stat -c%s "$SIDECAR") байт)"
fi

# ------------------------- конфиг: бэкап и правки args ----------------------
BACKUP="$OUT_DIR/config.backup.json"
cp "$CONFIG" "$BACKUP"
log "[*] конфиг забэкаплен: $BACKUP"
restore_config() {  # вернуть бэкап (состояние ДО проверки) — финальный шаг
  cp "$BACKUP" "$CONFIG"
}

set_ext() {  # $1 = путь сайдкара или "" (убрать пару)
  python3 - "$CONFIG" "$1" <<'PY'
import json, os, sys
p, val = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
args = [str(a) for a in (cfg.get("args") or [])]
i = args.index("--vocab-ext") if "--vocab-ext" in args else -1
if i >= 0:
    del args[i:i+2]
if val:
    args += ["--vocab-ext", val]
cfg["args"] = args
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
PY
}

# ------------------------- сервер: жизненный цикл (стиль bench_sweep) -------
port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }

wait_port_closed() {
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do
    port_open || return 0
    sleep 1
  done
  return 1
}

stop_server() {
  log "  останавливаю сервер ..."
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  wait_port_closed && log "  порт закрыт" || log "  не остановился за ${STOP_WAIT}s — добиваю"
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do
    pgrep -f "strata-poly" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 2
}

start_server() {  # $1 = куда писать лог движка
  : > "$1"
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" >> "$1" 2>&1 & echo $! > "$OUT_DIR/server.pid" )
  log "  сервер запущен (pid $(cat "$OUT_DIR/server.pid")), лог: $1"
}

wait_ready() {  # $1 = лог сервера; $2 = "want-ext" — ждём СВЕЖУЮ строку vocab-ext в логе движка
                # (лог движка аппендится между прогонами — ищем только после байт-отметки)
  local deadline=$((SECONDS + BOOT_TIMEOUT)) body want_ext="$2"
  if [[ -n "$want_ext" ]]; then
    LOG_MARK=$(stat -c%s "$ENG_LOG" 2>/dev/null || echo 0)
    log "  отметка лога движка: $LOG_MARK байт (vocab-ext ищу только после неё)"
  fi
  log "  жду готовности (${BOOT_TIMEOUT}s) ..."
  while (( SECONDS < deadline )); do
    port_open || { sleep 5; continue; }
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
    printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' || { sleep 5; continue; }
    if [[ -n "$want_ext" ]] && ! tail -c +$((LOG_MARK + 1)) "$ENG_LOG" 2>/dev/null \
        | grep -q "vocab-ext: .* rows over the main"; then
      sleep 2; continue
    fi
    log "  модель загружена"
    return 0
  done
  die "сервер не готов за ${BOOT_TIMEOUT}s (лог движка: $ENG_LOG, сервера: $1)"
}

ask() {  # $1 = файл ответа; жадный запрос, текст ответа без обёртки
  curl -s -m 600 "$BASE_URL/chat/completions" -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$(python3 - "$PROMPT" "$TEMPERATURE" "$MAX_TOKENS" "$MODEL" <<'PY'
import json, sys
print(json.dumps({"model": sys.argv[4], "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[3]), "temperature": float(sys.argv[2])}, ensure_ascii=False))
PY
)" | python3 -c "
import json, sys
d = json.load(sys.stdin)
c = d.get('choices', [{}])[0]
print((c.get('message') or c.get('delta') or {}).get('content', ''), end='')
"
}

# ------------------------- прогон A: с сайдкаром ----------------------------
log "[A] вариант С --vocab-ext ($N строк)"
set_ext "$SIDECAR"
stop_server
start_server "$OUT_DIR/engine-a.log"
wait_ready "$OUT_DIR/engine-a.log" want-ext
EXT_LINE=$(tail -c +$((LOG_MARK + 1)) "$ENG_LOG" 2>/dev/null \
  | grep "vocab-ext: .* rows over the main" | head -1)
log "  движок: $EXT_LINE"
[[ "$EXT_LINE" == *"$N rows over the main"* ]] || die "строка vocab-ext не сходится с N=$N: $EXT_LINE"
log "  запрос (temperature=$TEMPERATURE, max_tokens=$MAX_TOKENS) ..."
ask > "$OUT_DIR/A.txt" || die "запрос A не удался (лог: $OUT_DIR/engine-a.log)"
log "  ответ A: $(wc -c < "$OUT_DIR/A.txt") байт -> $OUT_DIR/A.txt"

# ------------------------- прогон B: без сайдкара ---------------------------
log "[B] вариант БЕЗ --vocab-ext"
set_ext ""
stop_server
start_server "$OUT_DIR/engine-b.log"
LOG_MARK=$(stat -c%s "$ENG_LOG" 2>/dev/null || echo 0)   # своя отметка: не видеть строку прогона A
wait_ready "$OUT_DIR/engine-b.log"
tail -c +$((LOG_MARK + 1)) "$ENG_LOG" 2>/dev/null | grep -q "vocab-ext:" \
  && die "в прогоне B движок неожиданно загрузил сайдкар (лог движка: $ENG_LOG)"
log "  запрос ..."
ask > "$OUT_DIR/B.txt" || die "запрос B не удался (лог: $OUT_DIR/engine-b.log)"
log "  ответ B: $(wc -c < "$OUT_DIR/B.txt") байт -> $OUT_DIR/B.txt"

# ------------------------- вердикт ------------------------------------------
log "=============================================================="
if cmp -s "$OUT_DIR/A.txt" "$OUT_DIR/B.txt"; then
  log "ВЕРДИКТ: ИДЕНТИЧНО ✓ — канал прозрачен, сайдкар невидим для декода"
  echo "PASS: ответы бит-в-бит идентичны ($EXT_LINE)" > "$OUT_DIR/verdict.txt"
  RC=0
else
  FIRST=$(cmp "$OUT_DIR/A.txt" "$OUT_DIR/B.txt" 2>&1 | head -1)
  log "ВЕРДИКТ: РАЗЛИЧИЕ ✗ — $FIRST"
  log "  |A|=$(wc -c < "$OUT_DIR/A.txt")  |B|=$(wc -c < "$OUT_DIR/B.txt")"
  diff "$OUT_DIR/A.txt" "$OUT_DIR/B.txt" | head -10 || true
  echo "FAIL: $FIRST" > "$OUT_DIR/verdict.txt"
  RC=1
fi
log "=============================================================="

# ------------------------- восстановление -----------------------------------
log "[*] восстанавливаю конфиг (без --vocab-ext) и сервер на исходных настройках"
restore_config
stop_server
start_server "$OUT_DIR/engine-restore.log"
wait_ready "$OUT_DIR/engine-restore.log"
log "готово. результаты: $OUT_DIR (A.txt B.txt verdict.txt engine-a.log)"
exit $RC
