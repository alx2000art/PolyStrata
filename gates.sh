#!/usr/bin/env bash
# ============================================================================
# gates.sh - дисциплина гейтов: длинный контекст + бюджет декода, с кодом возврата.
#
#        ./gates.sh                       # оба гейта
#        ./gates.sh budget                # только бюджет мс/токен
#        ./gates.sh needle                # только needle на длинном окне
#
# Гейты работают против ЖИВОГО сервера (конфиг: host/port/api_key) - то есть их
# можно гонять и на проде, не поднимая ничего.  Ноль на выходе = всё в бюджете.
#
# Окружение (как у tools/*.py):
#   CONFIG=...            конфиг модели (по умолчанию ~/PolyStrata/strata-dsv4-ud-iq2_xxs.json)
#   ROWS="8192 32768"     окна needle-гейта (дорого: 128k на нашем префилле ~30-60 мин)
#   MS_TOKEN=60           бюджет декода, мс/токен (у нас ~40-50)
#   FLOOR_PREFILL_TPS=5   пол префилла, tok/s
#   BUDGET_TOKENS=48      сколько токенов генерировать в бюджете
# ============================================================================
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
[[ -x "$PY" ]] || PY=python3

which="${1:-all}"
rc=0

if [[ "$which" == "all" || "$which" == "budget" ]]; then
    echo "=== гейт 1/2: бюджет декода (steady-state мс/токен) ==================="
    "$PY" "$ROOT/tools/decode_budget.py" || rc=1
    echo
fi

if [[ "$which" == "all" || "$which" == "needle" ]]; then
    echo "=== гейт 2/2: needle на длинном окне ================================="
    "$PY" "$ROOT/tools/ctx_needle.py" || rc=1
    echo
fi

if (( rc == 0 )); then
    echo "гейты: PASS"
else
    echo "гейты: FAIL (см. выше)"
fi
exit $rc