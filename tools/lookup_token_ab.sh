#!/usr/bin/env bash
# Детерминированный A/B окна: один вход, temperature 0, --lookup 0 против --lookup 3.
#
# Догадки lookup проверяются точным совпадением с выбором модели, поэтому текст ОБЯЗАН быть побитово
# тем же.  Первое расхождение в потоке токенов - позиция, где верификация окна приняла чужой токен
# (или окно испортило состояние).  Прямой запуск движка, без сервера; сервер на время останавливается.
#
#   ./tools/lookup_token_ab.sh                 # 24 повторяющихся id, 40 новых токенов
#   TOKENS=17,42,17,42,... MAXNEW=60 ./tools/lookup_token_ab.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
CFG="${CFG:-strata-dsv4-ud-iq2_xxs.json}"
BIN="${BIN:-engine-poly/strata-poly}"
# повторяющийся вход: n-граммной догонке нужно, чтобы история уже встречалась
TOKENS="${TOKENS:-1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4}"
MAXNEW="${MAXNEW:-40}"
PORT="$(python3 - "$CFG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("port") or 1524)
PY
)"

port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }
stop_server() {
  pkill -TERM -f "$ROOT/serve/server.py" 2>/dev/null || true
  for _ in $(seq 1 60); do port_open || break; sleep 1; done
  pkill -KILL -f "$ROOT/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  for _ in $(seq 1 30); do pgrep -f "strata-poly" >/dev/null || break; sleep 1; done
  sleep 2
}
start_server() {
  nohup "$ROOT/.venv/bin/python" "$ROOT/serve/server.py" --engine strata --config "$ROOT/$CFG" \
      --port "$PORT" > "/tmp/lookup-ab-server.log" 2>&1 &
  echo "(сервер поднимается, лог /tmp/lookup-ab-server.log)"
}
trap start_server EXIT

mapfile -d '' BASE < <(python3 - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8-sig"))
sys.stdout.write("\0".join(str(a) for a in (c.get("args") or []) if str(a) != "--no-graphs"))
PY
)

run() {  # $1 = подпись, $2 = значение --lookup
  local tag="$1" lookup="$2"
  local -a A=("${BASE[@]}")
  local i
  for i in "${!A[@]}"; do [[ "${A[$i]}" == "--lookup" ]] && A[$((i + 1))]="$lookup"; done
  "$BIN" "${A[@]}" --tokens "$TOKENS" --max-new "$MAXNEW" --temperature 0 --seed 1 \
      > "/tmp/lookup-$tag.out" 2> "/tmp/lookup-$tag.err" || {
    echo "!!! $tag: движок вышел с ошибкой, хвост:"; tail -4 "/tmp/lookup-$tag.err"; return 1; }
}

stop_server
echo "=== lookup 0 ==="
run l0 0
echo "=== lookup 3 ==="
run l3 3

python3 - <<'PY'
import re
import sys


def toks(path):
    """CLI печатает ответ одной строкой: "12,345,678,..." (не строки T - это серверный протокол)."""
    for line in open(path, encoding="utf-8", errors="replace"):
        line = line.strip()
        if line and re.fullmatch(r"-?\d+(,-?\d+)*", line):
            return [int(x) for x in line.split(",")]
    return []


def guessed(path):
    """Строка движка: '... N of M guessed tokens right' - окно вообще включалось или нет."""
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"(\d+) of (\d+) guessed tokens right", text)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


a, b = toks("/tmp/lookup-l0.out"), toks("/tmp/lookup-l3.out")
ga, gb = guessed("/tmp/lookup-l0.err"), guessed("/tmp/lookup-l3.err")
print(f"\nокно 0: {len(a)} токенов, догадок {ga[1]} (верных {ga[0]})")
print(f"окно 3: {len(b)} токенов, догадок {gb[1]} (верных {gb[0]})")
if not a and not b:
    print("!!! ни там ни там нет токенов - смотри /tmp/lookup-l0.err")
    sys.exit(1)
if gb[1] == 0:
    print("!!! с --lookup 3 не было ни одной догадки: тест ничего не проверил "
          "(нужен повторяющийся вход, попробуй TOKENS=... с длинным повтором)")
    sys.exit(2)
n = min(len(a), len(b))
first = next((i for i in range(n) if a[i] != b[i]), None)
if first is None and len(a) == len(b):
    print("вердикт: потоки совпадают символ в символ - окно чистое")
else:
    if first is None:
        first = n
    print(f"вердикт: РАСХОЖДЕНИЕ с токена {first}")
    lo = max(0, first - 4)
    print(f"  окно 0: {a[lo:first + 6]}")
    print(f"  окно 3: {b[lo:first + 6]}")
    print(f"  до этого места оба дали: {a[:first]}")
PY
