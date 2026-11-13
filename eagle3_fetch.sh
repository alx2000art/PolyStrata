#!/usr/bin/env bash
# ============================================================================
# eagle3_fetch.sh - the EAGLE-3 draft head for DeepSeek-V4-Flash-0731
# (AQ-MedAI, 0.7B Llama-arch, ~1.4 GB BF16 - the drafter resident in VRAM)
# and, with DRAFT_MTP=1, the MTP-only file (philpax, 5.88 GB) - the cheap
# calibration drafter.  Both files verified by size, resumable.
#
#   ~/models/eagle3_fetch.sh                # the EAGLE-3 head
#   DRAFT_MTP=1 ~/models/eagle3_fetch.sh    # + the MTP head
#
# After: the GGUF conversion (the port's phase 0) wires it into
# --draft-model.  The config.json rides along - the port's converter reads
# the tap layers and the draft vocab from it.
# ============================================================================
set -uo pipefail

DEST="${DEST:-$HOME/models}"
EAGLE_URL="https://huggingface.co/AQ-MedAI/DeepSeek-V4-Flash-0731-eagle3/resolve/main"
MTP_URL="https://huggingface.co/philpax/DeepSeek-V4-Flash-MTP-Only-GGUF/resolve/main"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }
mkdir -p "$DEST" || die "не создать $DEST"

fetch() {  # url, out, want-bytes
    local url="$1" out="$2" want="$3" got
    if [[ -f "$out" ]] && [[ $(stat -c%s "$out") == "$want" ]]; then
        log "  есть: $out ($(numfmt --to=iec-i --suffix=B "$want" 2>/dev/null || echo "$want B"))"
        return 0
    fi
    log "  качаю $(basename "$out") ($want байт) ..."
    wget -c -q -O "$out.part" "$url" || curl -sL -C - -o "$out.part" "$url"
    got=$(stat -c%s "$out.part" 2>/dev/null || echo 0)
    [[ "$got" == "$want" ]] || die "$(basename "$out"): скачалось $got из $want - перезапусти, докачает"
    mv "$out.part" "$out"
    log "  готово: $out"
}

log "=== EAGLE-3 драфтер (AQ-MedAI, 0.7B, MIT) ==="
fetch "$EAGLE_URL/model.safetensors"  "$DEST/DeepSeek-V4-Flash-0731-eagle3.safetensors" 1458452216
fetch "$EAGLE_URL/config.json"        "$DEST/DeepSeek-V4-Flash-0731-eagle3.config.json" 1049

if [[ "${DRAFT_MTP:-0}" == 1 ]]; then
    log "=== MTP-only драфтер (philpax, 5.88 GB) — калибровочный ==="
    fetch "$MTP_URL/DeepSeek-V4-Flash-MTP-bf16.gguf" "$DEST/DeepSeek-V4-Flash-MTP-bf16.gguf" 5884978624
fi

log "=== готово: файлы в $DEST ==="
log "следующий шаг — конверсия в GGUF (фаза 0 порта) и --draft-model в конфиге"
