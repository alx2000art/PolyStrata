#!/usr/bin/env bash
# lookup A/B поверх bench_sweep: сервером, конфигом и прогревом управляет свип (он это умеет),
# здесь только два его прогона с разным SWEEP_LOOKUP и подсчёт петель по сохранённым ответам.
#
#   ./tools/lookup_ab2.sh              # poly-сьют (7 промптов, включая doc-8k/16k/30k), ~35-40 мин
#   SUITE=quick ./tools/lookup_ab2.sh  # быстрый, 4 промпта
#
# Каждый прогон кладёт свои JSON в свою папку (lookup-ab-0, lookup-ab-3), поэтому ничего не
# перезаписывается; в конце печатаются две таблицы петель и скорости.
set -uo pipefail
cd "$(dirname "$0")/.."
SUITE_="${SUITE:-poly}"
BASE="80-eta-iq256off"

run_one() {  # $1 = lookup value
  local lookup="$1"
  echo "=============================================================="
  echo "=== SWEEP_LOOKUP=$lookup (сьют $SUITE_)"
  echo "=============================================================="
  SWEEP_LOOKUP="$lookup" ONLY=80 SUMMARY_BASE="$BASE" WARM_CHECK=0 SUITE="$SUITE_" SUITE_WARM=long \
    REPEAT=2 WARMUP=1 WARM_PASSES=1 ./bench_sweep.sh
  local d
  d=$(ls -dt bench_sweep bench_sweep-* 2>/dev/null | head -1)
  if [[ -z "$d" ]]; then
    echo "!!! не нашёл папку прогона"
    return 1
  fi
  rm -rf "lookup-ab-$lookup"
  mv "$d" "lookup-ab-$lookup"
  echo "  ответы: lookup-ab-$lookup/$BASE.json"
  [[ -f "lookup-ab-$lookup/$BASE.json" ]] || { echo "!!! ячейки $BASE нет — прогон не состоялся"; return 1; }
}

rm -rf lookup-ab-0 lookup-ab-3
run_one 0 || exit 1
run_one 3 || exit 1
python3 tools/loop_score.py "lookup-ab-0/$BASE.json" "lookup-ab-3/$BASE.json"
