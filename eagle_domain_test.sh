#!/usr/bin/env bash
# ============================================================================
# eagle_domain_test.sh - приём драфтера по ДОМЕНАМ на одном стенде.
#
# Зачем: отличить три версии сразу -
#   (1) наша имплементация/раунд в движке плохи   -> низкий приём ВЕЗДЕ, включая
#       английский и код на ОРИГИНАЛЬНОЙ голове;
#   (2) стенд (IO-матрица) в принципе не даёт выигрыша -> приём есть, скорости нет;
#   (3) наш русский тюн хуже оригинала -> на en/zh/code приём высокий, на ru низкий.
#
# Домены: en (английская проза из доков), code (исходники движка), ru (книги),
#         zh (файл ~/eagle_domains/zh.txt - если нет, домен пропускается).
# Каждый домен = N разных промптов по ~3000 символов, MAX_TOKENS на ответ.
#
# Запуск:  N=12 MAX_TOKENS=256 ./eagle_domain_test.sh
# ============================================================================
set -uo pipefail

STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
CORPUS="${CORPUS:-$HOME/corpus/corpus_clean.txt}"
OUT="${OUT:-$HOME/eagle_domains}"
N="${N:-12}"
MAX_TOKENS="${MAX_TOKENS:-256}"
SRC="${SRC:-$STRATA_DIR/src/poly}"
GPU_PY="${GPU_PY:-$HOME/.gpu-bench/bin/python}"

log() { printf '%s\n' "$*"; }
die() { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

mkdir -p "$OUT"
[[ -f "$CORPUS" ]] || die "нет корпуса $CORPUS"

# ------------------------- 1. списки промптов ---------------------------------
log "[1/3] готовлю списки промптов по доменам ..."
python3 - "$CORPUS" "$OUT" "$SRC" "$N" <<'PY'
import glob, os, re, sys

corpus, out, src, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
CHUNK = 3000


def chunks_at_lines(text):
    out, cur = [], ""
    for line in text.splitlines(True):
        if len(cur) + len(line) > CHUNK:
            if cur.strip():
                out.append(cur[:CHUNK])
            cur = ""
        cur += line
    if cur.strip():
        out.append(cur[:CHUNK])
    return out


def write(name, parts):
    parts = [p for p in parts if len(p.strip()) > 200][:n]
    path = os.path.join(out, name + ".list")
    open(path, "w", encoding="utf-8").write(chr(0).join(parts))
    print(f"  {name}: {len(parts)} промптов, {sum(map(len, parts))} симв -> {path}")


text = open(corpus, encoding="utf-8", errors="replace").read()
parts = re.split(r"<<<DOC (.*?)>>>", text)
docs = list(zip(parts[1::2], parts[2::2]))

for dom, prefix in (("en", "docs_en:"), ("ru", "books:")):
    body = "".join(b for nm, b in docs if nm.startswith(prefix))
    write(dom, chunks_at_lines(body))

code = []
for pat in ("*.cu", "*.cpp", "*.hpp", "*.py"):
    for path in sorted(glob.glob(os.path.join(src, "**", pat), recursive=True))[:40]:
        try:
            code.append(open(path, encoding="utf-8", errors="replace").read())
        except OSError:
            pass
write("code", chunks_at_lines("".join(code)))

zh_path = os.path.join(out, "zh.txt")
if os.path.exists(zh_path) and os.path.getsize(zh_path) > 2000:
    write("zh", chunks_at_lines(open(zh_path, encoding="utf-8", errors="replace").read()))
else:
    print(f"  zh: нет {zh_path} — домен будет пропущен (см. подсказку в конце)")
PY

# ------------------------- 2. прогоны по доменам -------------------------------
log ""
log "[2/3] прогоны (домен = N промптов, --window 4, текущая голова в конфиге) ..."
SLOG="$STRATA_DIR/eagle_collect/server.log"
printf '%-6s %10s %10s %10s %8s\n' домен принято догадок tok/s раундов
for dom in en code ru zh; do
  list="$OUT/$dom.list"
  [[ -s "$list" ]] || { printf '%-6s %10s\n' "$dom" "(нет списка)"; continue; }
  PROMPT_LIST="$list" N="$N" MAX_TOKENS="$MAX_TOKENS" DEC_DIR="/tmp/dom-$dom" \
    "$STRATA_DIR/eagle_collect_decode.sh" > "$OUT/$dom.run.log" 2>&1 || true
  if [[ ! -f "$SLOG" ]]; then printf '%-6s %10s\n' "$dom" "(нет лога)"; continue; fi
  acc=$(grep -o "drafts [0-9]*/[0-9]* accepted" "$SLOG" | awk -F'[ /]' '{a+=$2; b+=$3} END {printf "%d/%d", a, b}')
  pct=$(grep -o "drafts [0-9]*/[0-9]* accepted" "$SLOG" | awk -F'[ /]' '{a+=$2; b+=$3} END {printf "%.1f%%", (b ? 100*a/b : 0)}')
  spd=$(grep -o "([0-9.]* tok/s)" "$SLOG" | tr -d '()' | sed 's/ tok.s//' | awk '{s+=$1; n++} END {printf "%.1f", (n ? s/n : 0)}')
  rnd=$(grep -c "draft 1/3" "$SLOG" || true)
  printf '%-6s %10s %10s %10s %8s\n' "$dom" "$acc" "$pct" "$spd" "$rnd"
  cp "$SLOG" "$OUT/$dom.server.log" 2>/dev/null || true
done

log ""
log "[3/3] готово. Логи: $OUT/<домен>.server.log, прогоны: $OUT/<домен>.run.log"
log "если zh не было: получить китайский текст можно тем же сервером -"
log '  python3 - <<PY  (одним запросом, затем положить ответ в ~/eagle_domains/zh.txt)'
log '  см. LOOKUP ниже: PROMPT_LIST=~/eagle_domains/zh.list'
