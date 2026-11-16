#!/usr/bin/env bash
# eagle_matrix.sh - runs the same tale prompt through the engine under flag combinations and
# classifies each answer's reasoning as on-topic or drifted, printing the matrix:
#
#   clean   window 1, prompt-cache 0, conversation-cache 0   - the control
#   park    window 1, prompt-cache 0, conversation-cache 1G  - the parking isolated
#   pcache  window 1, prompt-cache 6, conversation-cache 0   - the checkpoints isolated
#   window  window 4, prompt-cache 0, conversation-cache 0   - the drafter's window isolated
#   prod    window 4, prompt-cache 6, conversation-cache 1G  - the production combo
#
# 3 identical requests per case: the first after the engine's start, then repeats (the park/restore
# cycle's signature is the first clean and the repeats drifted).  The script restarts the server
# itself (kills the window's copy, starts its own over nohup) and restores the original config's
# args at the end.
#
#   ./eagle_matrix.sh                 # the whole matrix, ~20 minutes
#   CASES="clean window" ./eagle_matrix.sh
#   NREQ=2 ./eagle_matrix.sh          # 2 requests per case
set -u

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json"
OUT="$STRATA_DIR/diag_run"
KEY="${API_KEY:-llama_server_pflb1}"
NREQ="${NREQ:-3}"
CASES="${CASES:-clean park pcache window prod}"
MAXTOK="${MAXTOK:-250}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-240}"

# порт, адрес и имя модели — из конфига, как в bench_sweep.sh: 0.0.0.0 превращаем в 127.0.0.1
PORT="$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("port") or 8080)
except Exception:
    print(8080)
PY
)"
HOST="${HOST:-$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("host") or "127.0.0.1")
except Exception:
    print("127.0.0.1")
PY
)}"
[[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]] && HOST="127.0.0.1"
BASE_URL="${BASE_URL:-http://$HOST:$PORT/v1}"
HEALTH_URL="${BASE_URL%/v1}/health"

mkdir -p "$OUT"
TALE="$OUT/mx_prompt.txt"
cat > "$TALE" <<'EOF'
Почему небо голубое? Объясни кратко и по-русски.
EOF

# the config's args patched for a case: the three flags out, the case's values in
patch_args() {
python3 - "$CONFIG" "$1" <<'PYEOF'
import json, sys
cfg_path, case = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path, encoding="utf-8-sig"))
a = cfg["args"]
vals = {"clean":  (1, 0, 0), "park":   (1, 0, 1024), "pcache": (1, 6, 0),
        "window": (4, 0, 0), "prod":   (4, 6, 1024)}[case]
for flag in ("--window", "--prompt-cache", "--conversation-cache-mib"):
    while flag in a:
        i = a.index(flag)
        del a[i:i + 2]
a += ["--window", str(vals[0]), "--prompt-cache", str(vals[1]),
      "--conversation-cache-mib", str(vals[2])]
tmp = cfg_path + ".tmp"
json.dump(cfg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PYEOF
}

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }

stop_server() {
    pkill -TERM -f "serve/server.py" 2>/dev/null || true
    for _ in $(seq 1 20); do port_open || break; sleep 1; done
    pkill -KILL -f "serve/server.py" 2>/dev/null || true
    pkill -KILL -f "engine-poly/strata-poly" 2>/dev/null || true
    pkill -KILL -f "strata-poly" 2>/dev/null || true
    for _ in $(seq 1 20); do pgrep -f "strata-poly" >/dev/null 2>&1 || break; sleep 1; done
    sleep 2
}

start_server() {
    : > "$OUT/matrix_server.log"
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" \
        >> "$OUT/matrix_server.log" 2>&1 & echo $! > "$OUT/matrix_server.pid" )
    echo -n "   жду ready ($HEALTH_URL)"
    local deadline=$((SECONDS + BOOT_TIMEOUT)) code body
    while (( SECONDS < deadline )); do
        port_open || { echo -n "."; sleep 5; continue; }
        body=$(curl -s -m 5 -H "Authorization: Bearer $KEY" "$HEALTH_URL" 2>/dev/null || true)
        if printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then
            echo " модель загружена"
            return 0
        fi
        code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$HEALTH_URL" 2>/dev/null || true)
        if [[ "${code:-000}" == "404" ]]; then
            code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $KEY" \
                        "$BASE_URL/models" 2>/dev/null || true)
            if [[ "${code:-000}" == "200" ]]; then
                echo " API отвечает"
                return 0
            fi
        fi
        echo -n "."
        sleep 5
    done
    echo " ТАЙМАУТ — последние строки сервера:"
    tail -8 "$OUT/matrix_server.log" | sed 's/^/     /'
    return 1
}

classify() {
python3 - "$1" <<'PYEOF'
import json, sys
try:
    m = json.load(open(sys.argv[1]))["choices"][0]["message"]
except Exception as e:
    print(f"ОШИБКА ЗАПРОСА: {e}")
    raise SystemExit
r = ((m.get("reasoning_content") or "") + " " + (m.get("content") or "")).lower()
foreign = ["we need respond", "we need to respond", "we need to produce", "we need to answer",
           "<user_init_", "[tool begins]", "function_valuation", "plantviz", "let's parse",
           "let's analyze", "instructional article", "user message appears",
           "tools are always available", "html formatting", "css"]
topic = ["небо", "голуб", "рассеян", "свет"]
fh = sum(1 for k in foreign if k in r)
th = sum(1 for k in topic if k in r)
cyr = sum(1 for c in r if '\u0400' <= c <= '\u04ff')
lat = sum(1 for c in r if 'a' <= c <= 'z')
# the drift's own symptom: a long, rambling reasoning for a "кратко" request, usually in English
long_ramble = len(r) > 3500 or (lat > cyr and len(r) > 1200)
v = "ЧИСТО" if (fh == 0 and th >= 1 and cyr >= lat and len(r) <= 2500) else \
    ("БРЕД" if (fh or long_ramble) else "НЕЯСНО")
head = (m.get("reasoning_content") or "")[:110].replace("\n", " ")
print(f"{v}  чужие_маркеры={fh} тема={th} кириллица/латиница={cyr}/{lat} ризонинг={len(r)} симв. | {head}")
PYEOF
}

say() { echo "[$(date +%H:%M:%S)] $*"; }

ORIG=$(python3 -c "import json; print(json.dumps(json.load(open('$CONFIG', encoding='utf-8-sig'))['args']))")
say "оригинальные args сохранены"

for case in $CASES; do
    say "=== $case: пересборка конфига, рестарт сервера ==="
    patch_args "$case" || { say "patch упал"; continue; }
    stop_server
    start_server || continue
    for i in $(seq 1 "$NREQ"); do
        say "  $case запрос $i/$NREQ ..."
        python3 -c "
import json
print(json.dumps({'messages':[{'role':'user','content':open('$TALE').read()}],'temperature':0,'max_tokens':$MAXTOK}))" > "$OUT/mx_body.json"
        curl -sS -m 300 -o "$OUT/mx_${case}_${i}.json" "$BASE_URL/chat/completions" \
            -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
            -d @"$OUT/mx_body.json" || true
        echo -n "    $case[$i]: "
        classify "$OUT/mx_${case}_${i}.json"
    done
done

python3 - "$CONFIG" "$ORIG" <<'PYEOF'
import json, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
cfg["args"] = json.loads(sys.argv[2])
tmp = sys.argv[1] + ".tmp"
json.dump(cfg, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PYEOF
say "оригинальные args возвращены в конфиг"
say "ответы: $OUT/mx_*.json; полный вывод классификации выше. Логи серверов: $OUT/matrix_server.log"
