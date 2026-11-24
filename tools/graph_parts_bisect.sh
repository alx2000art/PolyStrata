#!/usr/bin/env bash
# Бисект графов по частям на прямом запуске движка: один и тот же короткий промпт, зонд логитов.
#
# Части (STRATA_GRAPH_PARTS): h — голова слоя со вниманием, t — хвост слоя, s — shared-эксперт,
# l — логиты.  (c — маршрутизатор карты — не пробуем: card_moe() требует all_cached(), у нас эксперты
# в RAM, так что эта ветка не запускается.)  Первые строки зонда — величины логитов на pos 6, 7, 8:
# расхождение 0.4-3 — это не округление (то было бы 1e-4), а другая ветка расчёта.
#
# Прямому движку нужна вся VRAM, поэтому сервер на время прогона останавливается и поднимается обратно
# (trap, в том числе на Ctrl+C).  Запуск: ./tools/graph_parts_bisect.sh   (~1-2 мин на прогон)
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
CFG="${CFG:-strata-dsv4-ud-iq2_xxs.json}"
BIN="${BIN:-engine-poly/strata-poly}"
TOKENS="${TOKENS:-1,2,3,4,5,6,7}"
MAXNEW="${MAXNEW:-8}"

[[ -x "$BIN" ]] || { echo "нет $BIN — соберите (REBUILD=1 ./setup.sh)"; exit 2; }
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
      --port "$PORT" > "/tmp/graph-bisect-server.log" 2>&1 &
  echo "(сервер поднимается, лог /tmp/graph-bisect-server.log)"
}
trap start_server EXIT

# аргументы конфига без --no-graphs: графами управляем сами
mapfile -d '' BASE < <(python3 - "$CFG" <<'PY'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8-sig"))
sys.stdout.write("\0".join(str(a) for a in (c.get("args") or []) if str(a) != "--no-graphs"))
PY
)

run() {  # $1 = подпись, $2 = части ("" = все, "none" = без графов)
  local tag="$1" parts="${2-}"
  local -a env=(STRATA_POLY_LOGITS_TOP=6) extra=()
  if [[ "$parts" == "none" ]]; then
    extra+=(--no-graphs)
  elif [[ -n "$parts" ]]; then
    env+=(STRATA_GRAPH_PARTS="$parts")
  fi
  echo "=== $tag ==="
  env "${env[@]}" "$BIN" "${BASE[@]}" "${extra[@]}" --tokens "$TOKENS" --max-new "$MAXNEW" \
      --temperature 0 > /dev/null 2> "/tmp/parts-$tag.log"
  grep -m 6 "\[logits\]" "/tmp/parts-$tag.log" | sed 's/ n=1//' || {
    echo "(строк зонда нет, хвост лога:)"
    tail -4 "/tmp/parts-$tag.log"
  }
}

stop_server
run base none
for p in h t s l; do run "$p" "$p"; done
run all ""
