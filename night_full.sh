#!/usr/bin/env bash
# night_full.sh - все три свита одной ночи, подряд.
#
#   1) decode  (окно, swaps, lookup, draft-vocab, kv-whole, pool-триада)
#   2) prefill (tail-skip 2/6/8, prompt-batch, stage-*, ram-pool, threads — KNOBS_PRESET=prefill SUITE=long)
#   3) poly    (expert-mma в чистоте и поверх набора — клетки 58/59)
#
# Каждый вложенный скрипт ведёт свой бэкап/рестор конфига и свои папки
# (run-id маркер уводит повторные прогоны в свежие папки).  Один упал —
# остальные всё равно идут.
#
# usage: API_KEY=... ./night_full.sh
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
LOG="$STRATA_DIR/night_full-$(date +%Y%m%d-%H%M%S).log"
failures=()

step() {  # $1 = имя, $2.. = окружение и сама команда
  local name="$1"; shift
  echo "" | tee -a "$LOG"
  echo "=== [$name] старт ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
  ( cd "$STRATA_DIR" && env "$@" ) >> "$LOG" 2>&1
  local rc=$?
  if (( rc != 0 )); then
    echo "=== [$name] НЕ УДАЛСЯ (exit $rc) — продолжаю ===" | tee -a "$LOG"
    failures+=("$name")
  else
    echo "=== [$name] готово ($(date +%H:%M:%S)) ===" | tee -a "$LOG"
  fi
}

if [[ -z "${API_KEY:-}" ]]; then
  echo "ОШИБКА: export API_KEY=... перед запуском" >&2
  exit 1
fi

# sweep_complete БАЗА ПРЕФИКС-RUN-ID: найден ли уже ЗАВЕРШЁННЫЙ забег этого рода
# (папка с run-id.txt, начинающимся на префикс, и summary.md) — тогда шаг пропускается.
sweep_complete() {
  local d
  for d in "$STRATA_DIR/$1" "$STRATA_DIR/$1"-*; do
    [[ -d "$d" ]] || continue
    [[ "$(cat "$d/run-id.txt" 2>/dev/null)" == "$2"* && -f "$d/summary.md" ]] && return 0
  done
  return 1
}

echo "=== ночная полная очередь: $(date) ===" | tee "$LOG"

if sweep_complete bench_sweep_decode "decode/"; then
  echo "=== [decode] уже снят полностью (забег с run-id decode/* и summary есть) — пропуск ===" | tee -a "$LOG"
else
  step decode API_KEY="$API_KEY" ./bench_sweep_decode.sh
fi

if sweep_complete bench_sweep_decode "long/prefill"; then
  echo "=== [prefill] уже снят полностью — пропуск ===" | tee -a "$LOG"
else
  step prefill API_KEY="$API_KEY" KNOBS_PRESET=prefill SUITE=long ./bench_sweep_decode.sh
fi

if sweep_complete bench_sweep "poly"; then
  echo "=== [poly] уже снят полностью — пропуск ===" | tee -a "$LOG"
else
  step poly API_KEY="$API_KEY" ./bench_sweep.sh
fi

echo "" | tee -a "$LOG"
echo "=== ночь завершена: $(date) ===" | tee -a "$LOG"
echo "сводки:" | tee -a "$LOG"
for d in "$STRATA_DIR"/bench_sweep_decode "$STRATA_DIR"/bench_sweep_decode-*/ "$STRATA_DIR"/bench_sweep "$STRATA_DIR"/bench_sweep-*/; do
  s="${d%/}/summary.md"
  [[ -f "$s" ]] && echo "  $s" | tee -a "$LOG"
done 2>/dev/null
if (( ${#failures[@]} > 0 )); then
  echo "ПАДЕНИЯ: ${failures[*]}" | tee -a "$LOG"
  echo "подробности в $LOG" | tee -a "$LOG"
else
  echo "все три прошли чисто" | tee -a "$LOG"
fi
