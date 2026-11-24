#!/usr/bin/env bash
# Графы против прямых запусков: одни и те же логиты на одном и том же запросе.
#
# Зонд STRATA_POLY_LOGITS_TOP=<k> печатает top-5 логитов первым k вызовам, которым они нужны: шаг 0 — это
# последний токен промпта, то есть первый токен ответа.  Два прогона (ячейка 80 с --no-graphs и 94 с
# графами) на одном сьюте дают два списка; по ним видно, на каком шаге пути расходятся и на сколько:
# - top-1 тот же, значения отличаются на ~1e-4 и меньше — округление, ответы разойдутся только на ничьих;
# - top-1 другой при близких значениях — тоже ничья, но перевёрнутая;
# - top-1 другой при разнице в разы — это другой расчёт, и тогда бисект частями (ячейки 79/96/97/98/95).
#
# Движковый stderr идёт в поле log конфига, а не в server.log.last (тот — вывод сервера); если строк
# оттуда не нашлось, скрипт смотрит в server.log.last последней папки прогона.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG="${CFG:-strata-dsv4-ud-iq2_xxs.json}"
LOG="$(python3 - "$CFG" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("log") or "")
PY
)"
[[ -n "$LOG" ]] || { echo "в $CFG нет поля log — некуда смотреть"; exit 2; }
cp -f "$CFG" "$CFG.logit-probe-backup"
restore() { cp -f "$CFG.logit-probe-backup" "$CFG"; rm -f "$CFG.logit-probe-backup"; }
trap restore EXIT
python3 - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p, encoding="utf-8-sig"))
env = dict(c.get("env") or {})
env["STRATA_POLY_LOGITS_TOP"] = "3"
c["env"] = env
json.dump(c, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
PY

run() {  # $1 = ONLY, $2 = куда положить строки зонда
  local only="$1" out="$2"
  rm -f "$LOG"
  echo "=== ячейка $only ==="
  ONLY="$only" SUMMARY_BASE=80-eta-iq256off WARM_CHECK=0 SUITE=quick SUITE_WARM=quick \
    REPEAT=1 WARMUP=1 WARM_PASSES=1 ./bench_sweep.sh || true
  grep -m 8 "\[logits\]" "$LOG" > "$out" 2>/dev/null || true
  if [[ ! -s "$out" ]]; then
    local d
    d=$(ls -dt bench_sweep-* 2>/dev/null | head -1)
    [[ -n "$d" ]] && grep -m 8 "\[logits\]" "$d/server.log.last" > "$out" || true
  fi
  echo "$only: $(wc -l < "$out") строк зонда -> $out"
}

run 80 /tmp/logits-base.txt
run 94 /tmp/logits-graphs.txt
echo "=== 80 (без графов) ==="
cat /tmp/logits-base.txt
echo "=== 94 (графы) ==="
cat /tmp/logits-graphs.txt
echo "=== разница ==="
diff -u /tmp/logits-base.txt /tmp/logits-graphs.txt | head -40 || true
