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
PORT="${PORT:-1524}"
URL="http://127.0.0.1:$PORT/v1/chat/completions"
KEY="${API_KEY:-llama_server_pflb1}"
NREQ="${NREQ:-3}"
CASES="${CASES:-clean park pcache window prod}"
MAXTOK="${MAXTOK:-250}"

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

stop_server() {
    pkill -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    for _ in $(seq 1 15); do
        pgrep -f "$STRATA_DIR/serve/server.py" >/dev/null || break
        sleep 1
    done
    pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
    pkill -KILL -f "engine-poly/strata-poly" 2>/dev/null || true
    sleep 2
}

start_server() {
    : > "$OUT/matrix_server.log"
    ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
        --engine strata --config "$CONFIG" --port "$PORT" \
        >> "$OUT/matrix_server.log" 2>&1 & echo $! > "$OUT/matrix_server.pid" )
    echo -n "   жду ready"
    for _ in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $KEY" \
                    "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)
        [[ "$code" == "200" ]] && { echo " ok"; return 0; }
        echo -n "."
        sleep 3
    done
    echo " ТАЙМАУТ — смотри $OUT/matrix_server.log"
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
v = "ЧИСТО" if (fh == 0 and th >= 2) else ("БРЕД" if fh else "НЕЯСНО")
print(f"{v}  чужие_маркеры={fh} тема={th} кириллица/латиница={cyr}/{lat} ризонинг={len(r)} симв.")
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
        curl -sS -m 300 -o "$OUT/mx_${case}_${i}.json" "$URL" \
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
