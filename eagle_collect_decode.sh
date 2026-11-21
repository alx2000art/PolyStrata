#!/usr/bin/env bash
# ============================================================================
# eagle_collect_decode.sh — сбор decode-дампов: сервер тёплый, N запросов,
# каждый пишет taps-decode-*.bin (живые decode-состояния) в DEC_DIR.
#
# Нужно для ft8: голова хрупка к сдвигу read→decode (cos 0.90-0.99: taps decode-пути
# отличаются от read из-за батчевых численностей, а голова плоская - аргмакс
# переворачивается на 0.2-0.3 лога).  Единственное лечение - обучать на decode-состояниях.
#
# Запуск:  N=100 ./eagle_collect_decode.sh
#   PROMPT_FILE - затравка; при N>1 берутся куски корпуса (PROMPT_FILE + corpus_full)
# Результат: DEC_DIR/taps-decode-*.bin (~800 строк на запрос)
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
PROMPT_FILE="${PROMPT_FILE:-$HOME/prompts/doc-2k.txt}"
CORPUS="${CORPUS:-$HOME/corpus/corpus_full.txt}"
DEC_DIR="${DEC_DIR:-$HOME/eagle_taps_dec}"
N="${N:-100}"
MAX_TOKENS="${MAX_TOKENS:-384}"
# EXTRA_ARGS="--window 1 --lookup 0" - аргументы движка поверх конфига.  Снимать дампы
# для ОБУЧЕНИЯ нужно без спекуляции: при окне >1 соседние строки - это догадки
# головы, и метка "следующая строка" становится её собственным предложением, а не
# выбором модели (офлайн получается 60% на само-предсказании и 15% живьём).
EXTRA_ARGS="${EXTRA_ARGS:-}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-900}"
STOP_WAIT="${STOP_WAIT:-90}"
WORK="$STRATA_DIR/eagle_collect"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -f "$CONFIG" ]] || die "нет конфига"
[[ -f "$PROMPT_FILE" ]] || die "нет промпта $PROMPT_FILE"
mkdir -p "$DEC_DIR" "$WORK"
# Один сборщик на машину: второй экземпляр делает stop_server и поднимает свой сервер
# на том же порту - первый прогон после этого видит "запрос не прошёл" до конца и
# оставляет недособранный дамп, неотличимый от рабочего.  Лок держится на fd 9.
exec 9>"$WORK/collect.lock" || die "не открыть лок $WORK/collect.lock"
flock -n 9 || die "другой сбор уже идёт (лок $WORK/collect.lock) - дождись его конца"
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
BACKUP="$WORK/config.backup.json"
cp "$CONFIG" "$BACKUP"
restore_config() { cp "$BACKUP" "$CONFIG"; }

log "=============================================================="
log "eagle_collect_decode: N=$N  dump=$DEC_DIR  (prompt: $(basename "$PROMPT_FILE"))"
log "=============================================================="
python3 - "$CONFIG" "$DEC_DIR" <<'PY'
import json, os, sys
p, d = sys.argv[1], sys.argv[2]
cfg = json.load(open(p, encoding="utf-8-sig"))
env = cfg.get("env") or {}
for k in list(env):
    if k.startswith("STRATA_EAGLE_"):
        del env[k]
env["STRATA_EAGLE_DUMP_DECODE"] = d
args = [str(a) for a in (cfg.get("args") or [])]
i = 0
while i < len(args):
    if args[i] in ("--prompt-cache", "--conversation-cache-mib") and i + 1 < len(args):
        args[i + 1] = "0"
    i += 1
# --- измеряем ТОЛЬКО драфтера: n-граммный lookup выключен ---------------------
# lookup и голова делят ОДНО окно раунда, а арбитр DraftPolicy::choose
# (src/spec/draft_policy.cpp) отдаёт раунд тому, кто по замерам платит больше: при
# включённом lookup часть раундов уходит n-граммам, и числа "drafts N/M accepted"
# смешивают два механизма - непонятно, кто дал попадание.  Окно остаётся у головы.
# Переопределить: LOOKUP=3 ./script ...  (0 - выключен, 3 - дефолт движка)
_lk = os.environ.get("LOOKUP", "0")
_li = args.index("--lookup") if "--lookup" in args else -1
if _li >= 0 and _li + 1 < len(args) and not args[_li + 1].startswith("--"):
    args[_li + 1] = _lk
elif _li >= 0:
    args.insert(_li + 1, _lk)
else:
    args += ["--lookup", _lk]
# EXTRA_ARGS: пары "--ключ значение" поверх конфига (например --window 1 --lookup 0)
_extra = os.environ.get("EXTRA_ARGS", "").split()
# "@NAME=value" кладёт переменную в env движка (конфиг), а не в его CLI-аргументы -
# например @STRATA_EAGLE_TAPS_ONLY=1, чтобы снимать taps без догадок
for _tok in [t for t in _extra if t.startswith("@")]:
    _name, _, _val = _tok[1:].partition("=")
    if _name:
        env[_name] = _val
_extra = [t for t in _extra if not t.startswith("@")]
_k = 0
while _k + 1 < len(_extra) + 1 and _k < len(_extra):
    _key = _extra[_k]
    if _k + 1 < len(_extra) and not _extra[_k + 1].startswith("--"):
        _val = _extra[_k + 1]; _k += 2
    else:
        _val = None; _k += 1
    _idx = args.index(_key) if _key in args else -1
    if _idx >= 0 and _val is not None and _idx + 1 < len(args):
        args[_idx + 1] = _val
    elif _idx >= 0 and _val is not None:
        args.insert(_idx + 1, _val)
    elif _idx < 0:
        args += [_key] + ([_val] if _val is not None else [])
print("  args:", " ".join(args[-6:]) if args else "")
cfg["args"] = args
cfg["env"] = env
json.dump(cfg, open(p + ".tmp", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
os.replace(p + ".tmp", p)
print("  env DUMP_DECODE:", d)
PY
stop_server
: > "$WORK/server.log"
( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
    --engine strata --config "$CONFIG" --port "$PORT" >> "$WORK/server.log" 2>&1 & echo $! > "$WORK/server.pid" )
log "  сервер: pid $(cat "$WORK/server.pid"), жду загрузки ..."
deadline=$((SECONDS + BOOT_TIMEOUT)); body=""
while (( SECONDS < deadline )); do
  port_open || { sleep 5; continue; }
  body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
  printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' && break
  sleep 5
done
printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true' || { restore_config; stop_server; die "сервер не готов"; }
log "  модель загружена; собираю $N запросов (по $MAX_TOKENS токенов)"

python3 - "$PROMPT_FILE" "$CORPUS" "$N" "$WORK" <<'PY'
import os, random, sys
prompt_file, corpus, n, work = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompts = [open(prompt_file, encoding="utf-8").read()]
# РЕАЛЬНЫЕ запросы: твои диалоги (user-сообщения, chat-домен) + куски корпуса (doc-домен).
# Никакой синтетики.
dialogs = os.path.expanduser("~/eagle_gen/dialogs.jsonl")
nd = 0
if os.path.exists(dialogs):
    import json
    seen = set()
    with open(dialogs, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            for m in (rec.get("messages") or []):
                if m.get("role") == "user":
                    # content бывает СПИСКОМ кусков (мультимодальные сообщения): на строке
                    # сбор падал с "'list' object has no attribute 'strip'", prompts.txt не
                    # пересобирался и сбор шёл по старым промптам из прошлого запуска
                    c = m.get("content")
                    if isinstance(c, list):
                        c = " ".join(str(x.get("text") or "") for x in c if isinstance(x, dict))
                    if not isinstance(c, str):
                        c = ""
                    c = c.strip()
                    if 200 <= len(c) <= 4000 and c[:80] not in seen:
                        seen.add(c[:80])
                        prompts.append(c[:4000])
                        nd += 1
                    break   # один первый вопрос на диалог
    print(f"  из диалогов: {nd}")
nc = 0
if n > len(prompts):
    text = open(corpus, encoding="utf-8", errors="ignore").read()
    text = "\n".join(s for s in text.splitlines() if s.strip())
    rng = random.Random(42)
    while len(prompts) < n:
        st = rng.randrange(0, max(1, len(text) - 4000))
        prompts.append(text[st:st + 4000])
        nc += 1
print(f"  из корпуса: {nc}")
prompts = prompts[:n]
if len(prompts) < min(n, 10):
    raise SystemExit(f"промптов собралось {len(prompts)} при n={n} - проверь PROMPT_FILE/диалоги/корпус")
open(os.path.join(work, "prompts.txt"), "w", encoding="utf-8").write("\0".join(prompts))
print(f"  промптов: {len(prompts)}")
PY

BEFORE=$(ls "$DEC_DIR"/taps-decode-*.bin 2>/dev/null | wc -l)
FAILS=0
i=0
while IFS= read -r -d '' P; do
  i=$((i+1))
  python3 - "$P" "$MAX_TOKENS" > "$WORK/req.json" <<'PY'
import json, sys
print(json.dumps({"model": "x", "messages": [{"role": "user", "content": sys.argv[1]}],
                  "max_tokens": int(sys.argv[2]), "temperature": 0.0,
                  "chat_template_kwargs": {"enable_thinking": False},
                  "reasoning_effort": "none", "thinking": {"type": "disabled"}}, ensure_ascii=False))
PY
  if curl -s -m 600 -o /dev/null "$BASE_URL/chat/completions" -H "Authorization: Bearer $API_KEY" \
       -H "Content-Type: application/json" -d "@$WORK/req.json"; then
    FAILS=0
  else
    FAILS=$((FAILS + 1))
    log "  ! запрос $i не прошёл (подряд: $FAILS)"
    # Три подряд - движок лёг (OOM, падение, порт занят другим сбором).  Молча прогонять
    # остальные девяносто - это час времени и десяток строк дампа, которые выглядят как
    # рабочий прогон: падаем сразу и показываем хвост серверного лога.
    if (( FAILS >= 3 )); then
      log "  хвост лога сервера:"
      tail -20 "$WORK/server.log" | sed "s/^/    /"
      restore_config
      stop_server
      die "сервер не отвечает на 3 запроса подряд - сбор остановлен"
    fi
  fi
  if (( i % 10 == 0 )); then
    NOW=$(ls "$DEC_DIR"/taps-decode-*.bin 2>/dev/null | wc -l)
    log "  $i/$N, файлов дампа: $((NOW - BEFORE))"
  fi
done < "$WORK/prompts.txt"

AFTER=$(ls "$DEC_DIR"/taps-decode-*.bin 2>/dev/null | wc -l)
log "  готово: +(($AFTER - $BEFORE)) файлов, всего $AFTER"
restore_config
stop_server
log "[ok] конфиг восстановлен, сервер остановлен"
log "далее: ft8 по DEC_DIR (склейка блоков уже в тренере):"
log "  ln -sf ~/eagle_taps_dec/taps-decode-*.bin ~/eagle_taps_mix8/"
log "  ... + треть корпуса, --chain-depth 3"
