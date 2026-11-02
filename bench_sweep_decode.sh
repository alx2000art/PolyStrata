#!/usr/bin/env bash
# ============================================================================
# bench_sweep_decode.sh — A/B свип decode-настроек PolyStrata (Ubuntu).
#
# Отличие от bench_sweep.sh:
#   * эталон — не «все фичи выключены», а ПРОД-НАБОР (рабочий конфиг:
#     idx-mma, file-reads, read-heat, attend-chunks, group-heads 16,
#     q8-dequant, read-overlap, cpu-tiled; lookup/stage2/promt-cache выключены,
#     window 8, expert-swaps 64, stage-mib 512);
#   * поверх эталона крутится по ОДНОЙ decode-ручке за вариант: размер
#     спекулятивного окна, агрессивность кэша экспертов, lookup, kv-whole,
#     stage-mib, prompt-cache;
#   * сьют — decode: короткий промпт (~100-170 токенов), генерация 1500+
#     токенов, так что метрика варианта почти целиком decode (prefill в
#     таблице будет шумным — это нормально, сигнал читается по decode).
#
# Методика та же: 00-эталон дважды (пол шума), контроль дрейфа в середине
# (15) и в конце (99), между ними — только-ручка варианты.  Вариант
# детерминирован: перед записью спеки все «управляемые» ключи сносятся из
# args и расставляются заново (никакого накопления от варианта к варианту).
# --kv-whole — голый флаг без значения: спека "__FLAG__" добавляет его,
# эталон/другие варианты его сносят.
#
# Запуск (долго, ~5-6 часов — лучше в tmux/screen):
#   chmod +x bench_sweep_decode.sh
#   ./bench_sweep_decode.sh
# Переопределения — переменными окружения, как у bench_sweep.sh, напр.:
#   OUT_DIR=~/bench_decode2 FORCE=1 ./bench_sweep_decode.sh
#
# Результаты: OUT_DIR/<вариант>.json, OUT_DIR/summary.md, OUT_DIR/sweep.log
# Конфиг на время свипа переписывается, в конце восстанавливается из бэкапа.
# Никогда не правьте файлы скриптов, пока свип идёт (баннер печатает версию).
# ============================================================================
set -uo pipefail

# ------------------------- настройки (можно перекрыть через env) ------------
STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"
SWEEP_VERSION="${SWEEP_VERSION:-$(git -C "$STRATA_DIR" describe --always --dirty 2>/dev/null || echo not-git)}"
AB="${AB:-$STRATA_DIR/ab_patches.py}"
OUT_DIR="${OUT_DIR:-$STRATA_DIR/bench_sweep_decode}"
API_KEY="${API_KEY:-}"              # ключ: из окружения или поля api_key конфига
SUITE="${SUITE:-decode}"            # decode-сьют: короткий промпт, длинная генерация
REPEAT="${REPEAT:-2}"               # decode берётся максимумом по повторам (лотерея драфта)
WARMUP="${WARMUP:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
THINKING="${THINKING:-off}"
TIMEOUT_S="${TIMEOUT_S:-1800}"
BOOT_TIMEOUT="${BOOT_TIMEOUT:-3600}"
STOP_WAIT="${STOP_WAIT:-90}"
WARM_PASSES="${WARM_PASSES:-1}"     # один полный проход сьюта греет файловый кэш GGUF
FORCE="${FORCE:-0}"

# ------------------------- эталон (рабочий конфиг, 2026-11-02) ---------------
# Все ключи, которыми свип управляет, перечислены здесь явно — вариант
# определяется ТОЛЬКО своей спекой, наследования между вариантами нет.
# Формат спеки: пары "--ключ значение"; "__FLAG__" после ключа = голый флаг.
# Эталон = рабочий конфиг продакшена: read-heat/file-reads 0 (на живой
# нагрузке замедляют, вопреки свипу 11-01 на синтетике), attend-batch 0,
# stage2 512 включён.  --prompt-cache гасится для чистоты замера (продовое
# значение 6 вернёт финальный restore); threads/draft-vocab/no-graphs не
# заданы = дефолты движка.
MANAGED_KEYS="--read-overlap --file-reads --read-heat --attend-batch --attend-chunks
--group-heads --q8-dequant --idx-mma --cpu-tiled --lookup --stage2-mib --stage-mib
--prompt-cache --window --expert-swaps --kv-whole
--tail-skip --stage-ahead --prompt-batch --stage-read-mib --threads --draft-vocab --no-graphs
--ram-experts-mib --conversation-cache-mib"

BASELINE_SPEC="--read-overlap 1 --file-reads 0 --read-heat 0 --attend-batch 0
--attend-chunks 1 --group-heads 16 --q8-dequant 1 --idx-mma 1 --cpu-tiled 1
--lookup 0 --stage2-mib 512 --stage-mib 512 --tail-skip 4 --stage-ahead 1024
--prompt-batch 2048 --stage-read-mib 2048 --prompt-cache 0 --window 8
--expert-swaps 64"
# (без --kv-whole и --no-graphs: стрип в apply_variant гарантирует, что флагов нет)

# ------------------------- ручки ---------------------------------------------
# Формат: "имя|дельта-спеки" — поверх эталона меняется одна ручка.
# Набор выбирается KNOBS_PRESET: decode (дефолт, SUITE=decode) или prefill
# (SUITE=long: короткий прогрев и префилловые ручки).
KNOBS_DECODE=(
  "window-4|--window 4"
  "window-12|--window 12"
  "swaps-24|--expert-swaps 24"
  "swaps-128|--expert-swaps 128"
  "lookup-3|--lookup 3"
  "lookup-8|--lookup 8"
  "draft-vocab-32768|--draft-vocab 32768"
  "draft-vocab-65536|--draft-vocab 65536"
  "kv-whole|--kv-whole __FLAG__"
  "stage-mib-1024|--stage-mib 1024"
  "stage-mib-2048|--stage-mib 2048"
  "prompt-cache-6|--prompt-cache 6"
)
KNOBS_PREFILL=(
  "tail-skip-2|--tail-skip 2"
  "tail-skip-6|--tail-skip 6"
  "tail-skip-8|--tail-skip 8"
  "read-heat-1|--read-heat 1"        # перепроверка на синтетике вердикта живой нагрузки
  "file-reads-1|--file-reads 1"
  "stage-ahead-0|--stage-ahead 0"
  "stage-ahead-2048|--stage-ahead 2048"
  "prompt-batch-1024|--prompt-batch 1024"
  "prompt-batch-4096|--prompt-batch 4096"
  "prompt-batch-8192|--prompt-batch 8192"   # у Maya 8192/16384 там, где RAM держит экспертов: +29..47% prefill
  "stage-read-mib-0|--stage-read-mib 0"
  "stage-read-mib-4096|--stage-read-mib 4096"
  "threads-16|--threads 16"
  "no-graphs|--no-graphs __FLAG__"
  "stage2-off|--stage2-mib 0"
  "ram-pool-2048|--ram-experts-mib 2048"   # свой RAM-пул горячих экспертов (ram_pool.hpp)
  "ram-pool-6144|--ram-experts-mib 6144"
  "convcache-0|--conversation-cache-mib 0"     # освободить RAM под страницы экспертов
  "convcache-1024|--conversation-cache-mib 1024"
)
KNOBS_PRESET="${KNOBS_PRESET:-decode}"
case "$KNOBS_PRESET" in
  decode)  KNOBS=("${KNOBS_DECODE[@]}")  ;;
  prefill) KNOBS=("${KNOBS_PREFILL[@]}") ;;
  *) echo "KNOBS_PRESET: decode | prefill" >&2; exit 1 ;;
esac

# ------------------------- варианты -----------------------------------------
VARIANTS=()
VARIANTS+=("00-prod|$BASELINE_SPEC")    # ЭТАЛОН: прод-набор
VARIANTS+=("01-prod-b|$BASELINE_SPEC")  # эталон ещё раз — пол шума
i=30
for e in "${KNOBS[@]}"; do
  name="${e%%|*}"; delta="${e#*|}"
  VARIANTS+=("${i}-only-${name}|$BASELINE_SPEC $delta")
  i=$((i + 1))
done
VARIANTS+=("15-prod-m|$BASELINE_SPEC")  # контроль дрейфа в середине
VARIANTS+=("99-prod-c|$BASELINE_SPEC")  # контроль дрейфа в конце

# ------------------------- проверки -----------------------------------------
log()  { printf '%s\n' "$*"; }
die()  { printf 'ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ -d "$STRATA_DIR" ]] || die "нет каталога $STRATA_DIR"
[[ -f "$CONFIG" ]]     || die "нет конфига $CONFIG"
if [[ -z "$API_KEY" ]]; then   # ключ из поля api_key конфига (в git не хранится)
  API_KEY="$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("api_key") or "")
except Exception:
    print("")
PY
)"
fi
[[ -n "$API_KEY" ]] || die "нет API-ключа: export API_KEY=... или поле api_key в $CONFIG"
[[ -f "$AB" ]]         || die "нет $AB — обновите ab_patches.py на сервере (decode-сьют появился в нём)"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv — запустите ./setup.sh хотя бы раз"
command -v curl    >/dev/null || die "нужен curl"
command -v python3 >/dev/null || die "нужен python3"

PORT="$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("port") or 8080)
except Exception:
    print(8080)
PY
)"
MODEL="${MODEL:-$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("model_name") or "my-model")
except Exception:
    print("my-model")
PY
)}"
HOST="${HOST:-$(python3 - "$CONFIG" <<'PY'
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8-sig")).get("host") or "127.0.0.1")
except Exception:
    print("127.0.0.1")
PY
)}"
[[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]] && HOST="127.0.0.1"
BASE_URL="${BASE_URL:-http://$HOST:$PORT/v1}"
HEALTH_URL="${BASE_URL%/v1}/health"
BACKUP="$CONFIG.bench-backup-$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
exec > >(tee -a "$OUT_DIR/sweep.log") 2>&1

log "=============================================================="
log "PolyStrata DECODE sweep (эталон: прод-набор, поверх — по одной ручке)"
log "  конфиг:     $CONFIG  (порт $PORT, модель $MODEL)"
log "  версия:     $SWEEP_VERSION  (суффикс -dirty = файл правлен руками!)"
log "  base URL:   $BASE_URL"
log "  сьют:       $SUITE, repeat $REPEAT, warmup $WARMUP, thinking $THINKING"
log "  ручек:      ${#KNOBS[@]}, вариантов: ${#VARIANTS[@]} (примерно 20-30 мин на вариант)"
log "  результаты: $OUT_DIR"
log "=============================================================="

# ------------------------- работа с конфигом и сервером ---------------------
apply_variant() {  # $1 = spec: спека варианта (полное состояние управляемых ключей)
  # 1) снести из args ВСЕ управляемые ключи (значением считается токен,
  #    не начинающийся с "--" — так сносится и пара "ключ значение", и
  #    голый флаг вроде --kv-whole); 2) расставить спеку заново.
  python3 - "$CONFIG" "$1" "$MANAGED_KEYS" <<'PY'
import json, os, sys
path, spec = sys.argv[1], sys.argv[2]
managed = set(sys.argv[3].split())
cfg = json.load(open(path, encoding="utf-8-sig"))
args = [str(a) for a in (cfg.get("args") or [])]
kept = []
i = 0
while i < len(args):
    a = args[i]
    if a in managed:
        i += 1
        if i < len(args) and not args[i].startswith("--"):
            i += 1          # снести и значение
        continue
    kept.append(a)
    i += 1
toks = spec.split()
i = 0
while i < len(toks):
    k = toks[i]
    if i + 1 < len(toks) and toks[i + 1] == "__FLAG__":
        kept.append(k)          # голый флаг без значения (--kv-whole)
        i += 2
    elif i + 1 < len(toks):
        kept += [k, toks[i + 1]]
        i += 2
    else:
        kept.append(k)
        i += 1
# дедуп: спека варианта = эталон + дельта, ключи могут повторяться —
# оставляем ПОСЛЕДНЕЕ вхождение ключа (и его значение, если оно не ключ)
last = {}
for idx, a in enumerate(kept):
    if a.startswith("--"):
        last[a] = idx
final = []
i = 0
while i < len(kept):
    a = kept[i]
    if a.startswith("--"):
        if last.get(a) != i:
            i += 1
            if i < len(kept) and not kept[i].startswith("--"):
                i += 1          # снести и значение перекрытого ключа
            continue
        final.append(a)
        i += 1
        if i < len(kept) and not kept[i].startswith("--"):
            final.append(kept[i])
            i += 1
    else:
        final.append(a)
        i += 1
cfg["args"] = final
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
print("  args: " + " ".join(kept))
PY
}

port_open() { timeout 2 bash -c "exec 3<>/dev/tcp/$HOST/$PORT" 2>/dev/null; }

wait_port_closed() {
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do
    port_open || return 0
    sleep 1
  done
  return 1
}

stop_server() {
  log "  останавливаю сервер ..."
  pkill -TERM -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  if wait_port_closed; then
    log "  порт закрыт — сервер остановился"
  else
    log "  сервер не остановился за ${STOP_WAIT}s — добиваю"
  fi
  pkill -KILL -f "$STRATA_DIR/serve/server.py" 2>/dev/null || true
  pkill -KILL -f "$STRATA_DIR/engine-poly/strata-poly" 2>/dev/null || true
  pkill -KILL -f "strata-poly" 2>/dev/null || true
  local deadline=$((SECONDS + STOP_WAIT))
  while (( SECONDS < deadline )); do
    pgrep -f "strata-poly" >/dev/null 2>&1 || break
    sleep 1
  done
  sleep 2
}

start_server() {
  : > "$OUT_DIR/server.log.last"
  ( cd "$STRATA_DIR" && nohup "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/serve/server.py" \
      --engine strata --config "$CONFIG" --port "$PORT" \
      >> "$OUT_DIR/server.log.last" 2>&1 & echo $! > "$OUT_DIR/server.pid" )
  log "  сервер запущен (pid $(cat "$OUT_DIR/server.pid")), лог: $OUT_DIR/server.log.last"
}

wait_ready() {
  local deadline=$((SECONDS + BOOT_TIMEOUT)) code body
  log "  жду готовности сервера ($HEALTH_URL), до ${BOOT_TIMEOUT}s ..."
  while (( SECONDS < deadline )); do
    port_open || { sleep 5; continue; }
    body=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" "$HEALTH_URL" 2>/dev/null || true)
    if printf '%s' "$body" | grep -q '"loaded"[[:space:]]*:[[:space:]]*true'; then
      log "  модель загружена"
      return 0
    fi
    code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "$HEALTH_URL" 2>/dev/null || true)
    code=${code:-000}
    if [[ "$code" == "404" ]]; then
      code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 -H "Authorization: Bearer $API_KEY" "$BASE_URL/models" 2>/dev/null || true)
      code=${code:-000}
      [[ "$code" == "200" ]] && { log "  API отвечает (нет /health)"; return 0; }
    fi
    sleep 5
  done
  return 1
}

run_capture() {  # $1 = имя варианта
  local name="$1" extra=()
  [[ "$THINKING" == "off" ]] && extra+=(--no-thinking)
  python3 "$AB" capture \
    --out "$OUT_DIR/$name.json" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --suite "$SUITE" --repeat "$REPEAT" --warmup "$WARMUP" \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --timeout "$TIMEOUT_S" ${extra[@]+"${extra[@]}"}
}

variant_done() {  # $1 = имя: уже есть валидный захват?
  local f="$OUT_DIR/$1.json"
  [[ -f "$f" ]] || return 1
  python3 -c 'import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if d.get("runs") else 1)' "$f" 2>/dev/null
}

# ------------------------- главный цикл -------------------------------------
restore_config() { cp "$BACKUP" "$CONFIG"; }
on_interrupt() {
  echo
  log "Прервано: останавливаю сервер и восстанавливаю конфиг из $BACKUP"
  stop_server || true
  restore_config
  exit 130
}
trap on_interrupt INT TERM

cp "$CONFIG" "$BACKUP" || die "не смог забэкапить конфиг"
log "бэкап конфига: $BACKUP"
[[ "$FORCE" == "1" ]] && rm -f "$OUT_DIR"/*.json

failed=()
for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"
  spec="${entry#|}"
  [[ "$spec" == "$entry" ]] && spec=""
  log ""
  log "=== $name ${spec:+($spec)} ====================================================="
  if variant_done "$name" && [[ "$FORCE" != "1" ]]; then
    log "  уже есть $OUT_DIR/$name.json — пропускаю (FORCE=1, чтобы переделать)"
    continue
  fi
  apply_variant "$spec" || { log "  !!! не удалось записать вариант $name — пропускаю"; failed+=("$name"); continue; }
  stop_server
  start_server
  if ! wait_ready; then
    log "  !!! сервер не поднялся на варианте $name — пропускаю (лог: $OUT_DIR/server.log.last)"
    failed+=("$name")
    continue
  fi
  if (( WARM_PASSES > 0 )) && [[ ! -f "$OUT_DIR/.warmed" ]]; then
    for w in $(seq 1 "$WARM_PASSES"); do
      log "  прогрев $w/$WARM_PASSES: полный проход сьюта без записи результатов ..."
      run_capture ".warm-$w" >> "$OUT_DIR/warm.log" 2>&1 || log "  (прогрев не удался — продолжаю)"
      rm -f "$OUT_DIR/.warm-$w.json"
    done
    touch "$OUT_DIR/.warmed"
    log "  прогрев завершён (грел файловый кэш GGUF); перезапускаю сервер перед первым захватом"
    stop_server
    start_server
    wait_ready || log "  (!) сервер не поднялся после прогрева — захват, вероятно, упадёт"
  fi
  if run_capture "$name"; then
    log "  сохранено: $OUT_DIR/$name.json"
  else
    log "  !!! захват варианта $name упал — пропускаю"
    failed+=("$name")
  fi
done

# ------------------------- финал: конфиг назад, сервер вверх, сводка --------
log ""
log "восстанавливаю исходный конфиг и перезапускаю сервер на нём ..."
restore_config
stop_server
start_server
wait_ready || log "  (!) сервер не поднялся автоматически — смотрите $OUT_DIR/server.log.last и запустите ./setup.sh руками"

SUMMARY_BASE="${SUMMARY_BASE:-00-prod}"
SUMMARY_NOISE="${SUMMARY_NOISE:-01-prod-b}"
log ""
log "=== таблица прироста каждой ручки против прод-набора ($SUMMARY_BASE) ==="
python3 - "$OUT_DIR" "$SUMMARY_BASE" "$SUMMARY_NOISE" <<'PY'
import json, math, os, sys, datetime

out_dir, base_name, noise_name = sys.argv[1], sys.argv[2], sys.argv[3]

def load(name):
    p = os.path.join(out_dir, name + ".json")
    if not os.path.isfile(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
        return d if d.get("runs") else None
    except Exception:
        return None

def agg(path):
    """decode — максимум по повторам (лотерея драфта даёт вторую, вдвое
    медленную моду); prefill — повтор 0.  На decode-сьюте промпты короткие,
    так что prefill-колонка шумная — сигнал читается по decode."""
    d = json.load(open(path, encoding="utf-8"))
    pre, dec, tt = {}, {}, {}
    for r in d["runs"]:
        i = r["id"]
        if r.get("decode_tps", 0) > 0:
            dec[i] = max(dec.get(i, 0.0), r["decode_tps"])
        if r.get("repeat", 0) == 0:
            if r.get("prefill_tps", 0) > 0:
                pre[i] = r["prefill_tps"]
            tt[i] = r["ttft_s"]
    return pre, dec, tt

def geo(xs):
    xs = [x for x in xs if x and x > 0 and x == x]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else float("nan")

def med(xs):
    return sorted(xs)[len(xs) // 2] if xs else float("nan")

def delta(a_path, b_path):
    pa, da, ta = agg(a_path)
    pb, db, tb = agg(b_path)
    ids = sorted(set(pa) & set(pb) & set(da) & set(db))
    dr = [db[k] / da[k] for k in ids if da[k] > 0]
    pr = [pb[k] / pa[k] for k in ids if pa[k] > 0]
    return (len(dr), (geo(dr) - 1) * 100, (geo(pr) - 1) * 100,
            med([ta[k] for k in ids if k in ta]), med([tb[k] for k in ids if k in tb]))

def label(name):
    if "-only-" in name:
        return "ручка: " + name.split("-only-", 1)[1]
    if name.startswith("99-"):
        return "ПРОД-НАБОР В КОНЦЕ (контроль дрейфа)"
    if name.startswith("15-"):
        return "ПРОД-НАБОР В СЕРЕДИНЕ (контроль дрейфа)"
    return name

base = load(base_name)
if base is None:
    print(f"нет эталона {base_name}.json — сводка невозможна")
    sys.exit(0)

noise_d = noise_p = None
if noise_name and noise_name != base_name and load(noise_name) is not None:
    n = delta(os.path.join(out_dir, base_name + ".json"), os.path.join(out_dir, noise_name + ".json"))
    noise_d, noise_p = abs(n[1]), abs(n[2])
    print(f"пол шума (эталон, два прогона): decode ±{noise_d:.1f}%, prefill ±{noise_p:.1f}%")
floor = max(noise_d or 0.0, noise_p or 0.0, 1.0)

names = sorted(
    f[:-5] for f in os.listdir(out_dir)
    if f.endswith(".json") and f[:-5] != "summary" and load(f[:-5]) is not None
)
rows = []
for name in names:
    if name == base_name:
        continue
    note = " (повтор эталона — это и есть шум)" if name == noise_name else ""
    try:
        n, d, p, ta, tb = delta(os.path.join(out_dir, base_name + ".json"),
                                os.path.join(out_dir, name + ".json"))
    except Exception as e:
        print(f"{name}: не удалось сравнить ({e})")
        continue
    marks = []
    rec = "—"
    if noise_d is not None:
        dec_up = d > noise_d
        dec_dn = d < -noise_d
        pre_up, pre_dn = p > noise_p, p < -noise_p
        if dec_up: marks.append("decode ↑")
        if dec_dn: marks.append("decode ↓")
        if pre_up: marks.append("prefill ↑")
        if pre_dn: marks.append("prefill ↓")
        verdict = ("реальный эффект: " + ", ".join(marks)) if marks else "в пределах шума"
        if "-only-" in name:
            if dec_dn:
                rec = "❌ не внедрять: decode падает"
            elif dec_up:
                rec = "✅ внедрять"
            else:
                rec = "— нейтрально по decode"
    else:
        verdict = "в пределах шума (полоса шума не измерена)"
    rows.append((name, n, d, p, ta, tb, verdict, rec, note))

rows.sort(key=lambda r: -(r[2]))  # по приросту decode, сверху самые полезные
lines = []
lines.append(f"# Decode-ручки против прод-набора — {datetime.datetime.now():%Y-%m-%d %H:%M}")
lines.append("")
lines.append(f"* эталон: `{base_name}.json` — прод-набор (см. BASELINE_SPEC в bench_sweep_decode.sh)")
if noise_d is not None:
    lines.append(f"* пол шума: decode ±{noise_d:.1f}%, prefill ±{noise_p:.1f}% "
                 f"(эталон, два прогона: `{base_name}` против `{noise_name}`)")
lines.append("* сьют decode: промпты короткие, prefill-колонка шумная — решение по decode-колонке")
lines.append("* контроль дрейфа: 15-середина / 99-конец показывают, куда полз сам эталон за свип")
lines.append("")
lines.append("| вариант | пар | decode прирост | prefill прирост | TTFT мед. эталон→вариант | вывод | рекомендация |")
lines.append("|---|---:|---:|---:|---|---|---|")
for name, n, d, p, ta, tb, v, rec, note in rows:
    lines.append(f"| {label(name)}{note} | {n} | {d:+.1f}% | {p:+.1f}% | {ta:.2f}s → {tb:.2f}s | {v} | {rec} |")
lines.append("")

# детально по промптам: decode Δ каждого варианта против эталона
aggs = {nm: agg(os.path.join(out_dir, nm + ".json")) for nm in names}
lottery = []
for i in sorted({i for _p, dd, _t in aggs.values() for i in dd}):
    vals = [dd[i] for _p, dd, _t in aggs.values() if i in dd]
    if len(vals) >= 4 and max(vals) / min(vals) > 1.6:
        lottery.append(i)
if lottery:
    lines.append(f"(!) «лотерея драфта» (decode различается >1.6x между захватами): "
                 f"{', '.join(lottery)} — их дельты шумные; в расчёте берётся максимум decode по повторам")
    lines.append("")
lines.append("## decode Δ по промптам против эталона, %")
lines.append("")

def short(nm):
    return nm.split("-only-", 1)[1] if "-only-" in nm else label(nm)

cols = [nm for nm in names if nm != base_name]
_bd = aggs[base_name][1]
lines.append("| промпт | " + " | ".join(short(nm) for nm in cols) + " |")
lines.append("|---|" + "---:|" * len(cols))
for i in sorted(_bd):
    cells = []
    for nm in cols:
        dd = aggs[nm][1]
        cells.append(f"{(dd[i] / _bd[i] - 1) * 100:+.0f}%" if i in dd and _bd[i] > 0 else "-")
    lines.append(f"| {i} | " + " | ".join(cells) + " |")
md = os.path.join(out_dir, "summary.md")
with open(md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print(f"\nсводка: {md}")
PY

if (( ${#failed[@]} > 0 )); then
  log ""
  log "не удались варианты: ${failed[*]} (подробности в sweep.log)"
fi
log "готово. конфиг восстановлен из $BACKUP, сервер работает на нём."
