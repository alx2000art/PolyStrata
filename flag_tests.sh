#!/usr/bin/env bash
# ============================================================================
# flag_tests.sh - the week's three flag A/Bs in one run, each with its own
# server restarts, then one summary of all three tables.
#
#   PROMPT_FILE=~/prompts/doc-30k.txt ./flag_tests.sh
#
#   1) --prefetch-k 0 4    : the decode windows' own guess-reader (pf_read)
#                            fills the pool for the next layer - the prompt's
#                            spec-reader's twin, which no run has used yet.
#   2) --stage-ahead 1024 4096 : prompt batches under 4096 tokens leave the
#                            CPU a share of the experts (27 idle cores at
#                            prefill now), the card takes the rest.
#   3) --ram-experts-mib 6144 8192 : the pool's untrodden middle point
#                            (12288 lost to the page cache, 6144 is today's).
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
PROMPT_FILE="${PROMPT_FILE:-}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/flag_tests}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -n "$PROMPT_FILE" && -f "$PROMPT_FILE" ]] || die "укажи PROMPT_FILE=/путь/к/промпту.txt"
[[ -x "$STRATA_DIR/knob_ab.sh" ]] || die "нет $STRATA_DIR/knob_ab.sh (ветка base)"
mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"

declare -A TABLE

run() {  # $1 = метка, $2 = флаг, $3 = значение A, $4 = значение B
    local label="$1"
    local flag="$2"
    local a="$3"
    local b="$4"
    local f="$OUT_DIR/$label.log"
    log ""
    log "=== A/B: $flag $a $b ($label) ==="
    PROMPT_FILE="$PROMPT_FILE" "$STRATA_DIR/knob_ab.sh" "$flag" "$a" "$b" 2>&1 | tee "$f" | tail -10
    TABLE["$label"]="$(sed -n '/=== ИТОГ/,$p' "$f" | head -8)"
}

run prefetch --prefetch-k 0 4
run ahead --stage-ahead 1024 4096
run pool8 --ram-experts-mib 6144 8192

log ""
log "=========== СВОДКА ТРЁХ ТЕСТОВ ==========="
for k in prefetch ahead pool8; do
    log "--- $k ---"
    log "${TABLE[$k]:-(таблица не собралась — смотри $OUT_DIR/$k.log)}"
done
log ""
log "читай: у каждого A/B своя done-строка на значение (там decode tok/s и"
log "expert cache %); SSD и средняя скорость — в таблице. Вердикты берём"
log "только по парам, где разница больше пары процентов."