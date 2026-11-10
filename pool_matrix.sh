#!/usr/bin/env bash
# ============================================================================
# pool_matrix.sh - бисекция путей RAM-пула на живом '!'-глюке.
#
# Глюк воспроизводится на ПЕРВОМ запросе после старта сервера при включённом
# --ram-experts-mib. Матрица гоняет 4 конфигурации гейтов (старт сервера +
# один короткий промпт на каждую) и печатает, какой путь виноват:
#
#   run1  оба пути on      (как сейчас)      — база воспроизведения
#   run2  STRATA_POLY_POOL_STAGE=0           — staging мимо пула
#   run3  STRATA_POLY_POOL_SWAP=0            — decode-свапы мимо пула
#   run4  оба =0                             — пул выделен, но не используется
#
# Итог: run4 чистый + 2/3 сломанные → конкретный путь; run4 сломанный → пул
# не при делах (сдвиг лейаута памяти — тогда ASAN/размер пула); всё чистое →
# триггер не в матрице (другой промпт/состояние).
#
# usage: API_KEY=... ./pool_matrix.sh [config]
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/pool_matrix}"
API_KEY="${API_KEY:-}"
PROMPT="${PROMPT:-Составь короткий план на день из 5 пунктов и кратко поясни каждый.}"
MAX_TOKENS="${MAX_TOKENS:-256}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига $CONFIG"
if [[ -z "$API_KEY" ]]; then
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

BACKUP="$CONFIG.pool-matrix-backup-$(date +%Y%m%d-%H%M%S)"
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
set_env() {  # $1 = "VAR=значение" или "VAR" (снести); прочие env не трогаются
  python3 - "$CONFIG" "$1" <<'PYEOF'
import json, os, sys
path, spec = sys.argv[1], sys.argv[2]
cfg = json.load(open(path, encoding="utf-8-sig"))
env = cfg.setdefault("env", {})
if "=" in spec:
    k, v = spec.split("=", 1)
    env[k] = v
else:
    env.pop(spec, None)
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
PYEOF
}
ask() {  # $1 = файл ответа; ответ модели одним запросом, thinking выключен
  python3 - "$BASE_URL" "$API_KEY" "$MODEL" "$1" "$PROMPT" "$MAX_TOKENS" <<'PYEOF'
import json, sys, urllib.request
base, key, model, out, prompt, mt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.9, "max_tokens": mt, "reasoning_budget_tokens": 0,
                   "stream": False}).encode("utf-8")
req = urllib.request.Request(base + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read().decode("utf-8"))
    text = d["choices"][0]["message"].get("content") or ""
    finish = d["choices"][0].get("finish_reason") or "?"
except Exception as e:
    text, finish = f"(ошибка запроса: {e})", "error"
open(out, "w", encoding="utf-8").write(text + f"\n[finish_reason: {finish}]\n")
print(text)
PYEOF
}
degenerate() {  # $1 = файл ответа: детект длинного повтора (символ или слово)
  python3 - "$1" <<'PYEOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8").read()
t = t.split("[finish_reason:")[0]
if re.search(r"(.)\1{63,}", t):
    print("yes (повтор символа)")
elif re.search(r"(\b\w+\b)([^\w]+\1\1\b){9,}", t, re.IGNORECASE):
    print("yes (повтор слова)")
else:
    print("no")
PYEOF
}

mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
log "=== матрица путей RAM-пула: промпт $MAX_TOKENS токенов × 4 конфигурации ==="
log "    промпт: $PROMPT"
log ""

declare -A RESULT
run_matrix() {
  local run="$1" env_spec="$2" desc="$3"
  log "=== $run: $desc ==="
  set_env "$env_spec"
  stop_server
  start_server
  if ! wait_ready; then
    log "  сервер не поднялся — пропуск"
    RESULT["$run"]="СЕРВЕР НЕ ПОДНЯЛСЯ"
    return
  fi
  local f="$OUT_DIR/$run.txt"
  ask "$f"
  local verdict
  verdict="$(degenerate "$f")"
  log "  детект: $verdict  (ответ: $f)"
  if [[ "$verdict" == no* ]]; then RESULT["$run"]="чисто"; else RESULT["$run"]="ГЛЮК ($verdict)"; fi
  log ""
}

run_matrix "run1-both-on"  ""                                        "оба пути on (как сейчас)"
run_matrix "run2-stage-off" "STRATA_POLY_POOL_STAGE=0"               "staging мимо пула, свапы через пул"
run_matrix "run3-swap-off"  "STRATA_POLY_POOL_SWAP=0"                "свапы мимо пула, staging через пул"
run_matrix "run4-both-off"  "STRATA_POLY_POOL_STAGE=0 STRATA_POLY_POOL_SWAP=0"  "пул выделен, но не используется"

restore_config
log "=== МАТРИЦА ==="
for r in run1-both-on run2-stage-off run3-swap-off run4-both-off; do
  log "  $r: ${RESULT[$r]:-не гонялся}"
done
log ""
log "разбор: run4 чистый + run2/3 сломанные → конкретный путь (см. какой сломан);"
log "        run4 сломанный → пул не при делах (сдвиг лейаута памяти);"
log "        всё чистое → триггер вне матрицы (другой промпт/длиннее запрос)."
log ""
log "восстанавливаю конфиг и сервер на нём ..."
stop_server
start_server
wait_ready || log "  (!) сервер не поднялся — запусти ./setup.sh руками"
log "готово. ответы: $OUT_DIR/run*.txt"
