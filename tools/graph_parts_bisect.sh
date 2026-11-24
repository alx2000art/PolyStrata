#!/usr/bin/env bash
# Бисект графов по частям на прямом запуске движка: один и тот же короткий промпт, зонд логитов.
#
# Части (STRATA_GRAPH_PARTS): h — голова слоя со вниманием, t — хвост слоя, s — shared-эксперт,
# c — маршрутизатор карты, l — логиты.  Первая строка зонда — pos, на которой расходится прямой запуск
# и граф: величина 0.4-3 логита — это не округление (то было бы 1e-4), а другая ветка расчёта.
#
# Запуск: ./tools/graph_parts_bisect.sh            (7 прогонов движка, ~1-2 мин каждый)
#         TOKENS=1,2,3,4,5,6,7 MAXNEW=8
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${CFG:-strata-dsv4-ud-iq2_xxs.json}"
BIN="${BIN:-engine-poly/strata-poly}"
TOKENS="${TOKENS:-1,2,3,4,5,6,7}"
MAXNEW="${MAXNEW:-8}"

[[ -x "$BIN" ]] || { echo "нет $BIN — соберите (REBUILD=1 ./setup.sh)"; exit 2; }
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
  env "${env[@]}" "$BIN" "${BASE[@]}" "${extra[@]}" --tokens "$TOKENS" --max-new "$MAXNEW" \
      --temperature 0 > /dev/null 2> "/tmp/parts-$tag.log"
  echo "--- $tag ---"
  grep -m 6 "\[logits\]" "/tmp/parts-$tag.log" | sed 's/ n=1//' || {
    echo "(строк зонда нет, хвост лога:)"
    tail -5 "/tmp/parts-$tag.log"
  }
}

run base none
for p in h t s c l; do run "$p" "$p"; done
run all ""
