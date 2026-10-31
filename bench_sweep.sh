#!/usr/bin/env bash
# ============================================================================
# bench_sweep.sh — автоматическая серия A/B-бенчмарков PolyStrata (Ubuntu).
#
# Методика:
#   1. ЭТАЛОН (00-all-off): конфиг со ВСЕМИ фичами форка выключенными.
#      Прогоняется дважды (01-all-off-b) — разница прогонов = пол шума.
#   2. ВСЁ ВКЛЮЧЕНО (10-all-on): все фичи форка включены (ваши рабочие
#      настройки) — суммарный эффект набора.
#   3. Для каждой фичи (20+): эталон + включена ТОЛЬКО эта фича.
#
# Для каждого варианта скрипт сам переписывает "args" конфига, перезапускает
# сервер, ждёт загрузки модели (/health -> loaded:true) и снимает промпт-сьют
# ab_patches.py. В конце — таблица ПРИРОСТА каждой фичи против эталона
# (все-выкл) с вердиктом «реальный прирост / в пределах шума / замедляет».
# Никаких ручных сравнений не нужно.
#
# Почему повторы считаются так, а не «в лоб»: свип принудительно гасит
# --prompt-cache (см. apply_variant) — чекпоинты дают только разброс и отравляют
# ячейки частичными попаданиями, а повтор 1 остаётся вторым холодным замером.
# Поэтому prefill в сводке берётся из повтора 0, а decode — максимум по
# повторам: генерация «не в масть» драфтеру даёт вторую моду скорости (~x2
# медленнее) и без максимума ложно роняет вариант.
# Пол шума меряется дублем эталона на двух разных загрузках сервера.
#
# Запуск (долго, часы — лучше в tmux/screen):
#   chmod +x bench_sweep.sh
#   ./bench_sweep.sh
# Переопределение настроек — переменными окружения, напр.:
#   SUITE=poly OUT_DIR=~/bench2 ./bench_sweep.sh
#   FORCE=1 ./bench_sweep.sh        # перезаписать уже готовые прогоны
#
# Результаты: OUT_DIR/<вариант>.json, OUT_DIR/summary.md, OUT_DIR/sweep.log
# Конфиг strata-*.json на время свипа переписывается, в конце восстанавливается
# из бэкапа, сервер остаётся работать на исходных настройках.
# ============================================================================
set -uo pipefail

# ------------------------- настройки (можно перекрыть через env) ------------
STRATA_DIR="${STRATA_DIR:-$HOME/PolyStrata}"                  # каталог PolyStrata
CONFIG="${CONFIG:-$STRATA_DIR/strata-dsv4-ud-iq2_xxs.json}"   # live-конфиг модели
# версия = коммит, из которого запущен скрипт; "-dirty" = файл правлен руками
SWEEP_VERSION="${SWEEP_VERSION:-$(git -C "$STRATA_DIR" describe --always --dirty 2>/dev/null || echo not-git)}"
AB="${AB:-$STRATA_DIR/ab_patches.py}"                         # где лежит ab_patches.py
OUT_DIR="${OUT_DIR:-$STRATA_DIR/bench_sweep}"
API_KEY="${API_KEY:-}"              # ключ: из окружения или поля api_key конфига (блок проверок ниже)
SUITE="${SUITE:-poly}"              # poly = long + doc-30k: большой вход даёт чистый prefill-сигнал
REPEAT="${REPEAT:-2}"               # 2 повтора: prefill в сводке берётся из повтора 0 (холодного),
                                    # decode — максимум по повторам (защита от «лотереи драфта»)
WARMUP="${WARMUP:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-42}"
THINKING="${THINKING:-off}"         # off = --no-thinking (меньше разброса), on = как в проде
TIMEOUT_S="${TIMEOUT_S:-1800}"      # таймаут одного запроса, с
BOOT_TIMEOUT="${BOOT_TIMEOUT:-3600}"  # сколько ждать загрузки модели, с
STOP_WAIT="${STOP_WAIT:-90}"        # сколько ждать остановки старого сервера, с
WARM_PASSES="${WARM_PASSES:-1}"     # полных проходов сьюта без записи перед первым замером
                                    # (греет файловый кэш GGUF; одного хватает — остаточное
                                    # смещение первого варианта гасят контролы 15/99 и пары)
FORCE="${FORCE:-0}"

# ------------------------- фичи форка (docs/FORK_FEATURES.ru.md) ------------
# Формат: "имя|--ключ|значение-ВКЛ|значение-ВЫКЛ".
# значение-ВКЛ должно совпадать с тем, что стоит (или стояло бы по умолчанию)
# в вашем рабочем конфиге; значение-ВЫКЛ — документированное "off".
FEATURES=(
  "read-overlap|--read-overlap|1|0"
  "file-reads|--file-reads|1|0"
  "read-heat|--read-heat|1|0"
  "attend-batch|--attend-batch|1|0"
  "group-heads-16|--group-heads|16|8"
  "q8-dequant|--q8-dequant|1|0"
  "attend-chunks|--attend-chunks|1|0"
  "idx-mma|--idx-mma|1|0"
  "cpu-tiled|--cpu-tiled|1|0"
  "lookup|--lookup|8|0"
  "stage2|--stage2-mib|512|0"
)

split_feat() {  # "имя|--ключ|ВКЛ|ВЫКЛ" -> FEAT_NAME/FEAT_KEY/FEAT_ON/FEAT_OFF.
  # Разбор чистыми подстановками параметров: без read/IFS/herestring — на боксе
  # они отдали пустые поля (пустые спеки свипов 10-31 при рабочем массиве),
  # здесь разбор не зависит ни от окружения, ни от версии шелла.
  FEAT_NAME="${1%%|*}"
  local rest="${1#*|}"
  FEAT_KEY="${rest%%|*}"
  rest="${rest#*|}"
  FEAT_ON="${rest%%|*}"
  FEAT_OFF="${rest#*|}"
}

all_off_spec() {  # все фичи ВЫКЛ — эталон
  local out="" e
  for e in "${FEATURES[@]}"; do
    split_feat "$e"
    out+=" $FEAT_KEY $FEAT_OFF"
  done
  echo "${out# }"
}

all_on_spec() {  # все фичи ВКЛ (ваш рабочий набор)
  local out="" e
  for e in "${FEATURES[@]}"; do
    split_feat "$e"
    out+=" $FEAT_KEY $FEAT_ON"
  done
  echo "${out# }"
}

only_on_spec() {  # $1 = имя фичи: эталон + включена только она
  local want="$1" out="" e
  for e in "${FEATURES[@]}"; do
    split_feat "$e"
    if [[ "$FEAT_NAME" == "$want" ]]; then out+=" $FEAT_KEY $FEAT_ON"; else out+=" $FEAT_KEY $FEAT_OFF"; fi
  done
  echo "${out# }"
}

# ------------------------- варианты -----------------------------------------
VARIANTS=()
VARIANTS+=("00-all-off|$(all_off_spec)")     # ЭТАЛОН: всё выключено
VARIANTS+=("01-all-off-b|$(all_off_spec)")   # эталон ещё раз — пол шума
VARIANTS+=("10-all-on|$(all_on_spec)")       # всё включено (ваш конфиг)
i=20
for e in "${FEATURES[@]}"; do
  split_feat "$e"
  VARIANTS+=("${i}-only-${FEAT_NAME}|$(only_on_spec "$FEAT_NAME")")
  i=$((i + 1))
done
VARIANTS+=("15-all-off-m|$(all_off_spec)")   # контроль дрейфа в середине свипа
# --- дополнительные варианты (крутки) ---------------------------------------
# stage2 (второй staging-буфер). БАЗА флага живёт в FEATURES ("stage2"):
# эталон/all-on/only-ячейки получают его из спеки (0 или 512), крутки ниже
# перекрывают значение на месте (apply_variant правит существующий ключ, без
# дубликатов). Без этого строка конфига с --stage2-mib 512 протекла бы во все
# "выкл"-варианты (ночь 10-31: все 17 вариантов прошли с одним прод-конфигом).
# На бинарнике без stage2 крутки не поднимутся и будут пропущены.
VARIANTS+=("50-on-stage2-512|$(all_on_spec)")                     # один в один 10-all-on — мера шума
VARIANTS+=("51-on-stage2-256|$(all_on_spec) --stage2-mib 256")    # меньше отъёма кэша
VARIANTS+=("52-eta-stage2-512|$(all_off_spec) --stage2-mib 512")  # механизм в чистоте
# --- доля CPU-пула (поверх ВСЕГО НАБОРА: «фикс против auto» на проде).
# ВАЖНО: --cpu-tiled сам по себе булев (0/1) — "0.7" через atoi() = тихое
# ВЫКЛЮЧЕНИЕ tiled; долей работ рулит --cpu-share (auto = движок балансирует
# сам по таймингам). В проде ключа нет = auto, так что утечки в другие ячейки нет.
VARIANTS+=("55-on-share50|$(all_on_spec) --cpu-share 0.5")
VARIANTS+=("56-on-share70|$(all_on_spec) --cpu-share 0.7")
VARIANTS+=("57-on-share100|$(all_on_spec) --cpu-share 1.0")
# поверх ЭТАЛОНА — прочие крутки (раскомментируйте нужное):
# VARIANTS+=("40-eta-pb2048|$(all_off_spec) --prompt-batch 2048")
# VARIANTS+=("41-eta-tailskip0|$(all_off_spec) --tail-skip 0")
# VARIANTS+=("42-eta-nocache|$(all_off_spec) --prompt-cache 0")
# VARIANTS+=("43-eta-stageread4096|$(all_off_spec) --stage-read-mib 4096")
# поверх ВСЁ-ВКЛЮЧЕНО — эффект крутки на рабочем конфиге:
# VARIANTS+=("45-on-pb2048|$(all_on_spec) --prompt-batch 2048")
# VARIANTS+=("46-on-convcache0|$(all_on_spec) --conversation-cache-mib 0")
VARIANTS+=("99-all-off-c|$(all_off_spec)")   # контроль дрейфа (stage2 гасится самой спекой эталона)

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
[[ -f "$AB" ]]         || die "нет $AB — скопируйте ab_patches.py на сервер, напр.: scp ab_patches.py server:$AB"
[[ -x "$STRATA_DIR/.venv/bin/python" ]] || die "нет $STRATA_DIR/.venv — запустите ./setup.sh хотя бы раз"
command -v curl    >/dev/null || die "нужен curl"
command -v python3 >/dev/null || die "нужен python3"

# порт и имя модели — из конфига
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
# адрес и порт сервера — из конфига ("host", "port"); 0.0.0.0 превращаем в 127.0.0.1
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
log "PolyStrata bench sweep (эталон: все фичи выключены)"
log "  конфиг:     $CONFIG  (порт $PORT, модель $MODEL)"
log "  версия:     $SWEEP_VERSION  (суффикс -dirty = файл правлен руками!)"
log "  base URL:   $BASE_URL"
log "  сьют:       $SUITE, repeat $REPEAT, warmup $WARMUP, thinking $THINKING"
  log "  фич:        ${#FEATURES[@]}, вариантов: ${#VARIANTS[@]} (примерно 12-25 мин на вариант при suite poly)"
log "  результаты: $OUT_DIR"
log "=============================================================="

# ------------------------- работа с конфигом и сервером ---------------------
apply_variant() {  # $1 = spec пар "ключ значение ..."
  python3 - "$CONFIG" "$1" <<'PY'
import json, os, sys
path, spec = sys.argv[1], sys.argv[2].strip()
cfg = json.load(open(path, encoding="utf-8-sig"))
args = [str(a) for a in (cfg.get("args") or [])]
toks = spec.split()
i = 0
while i < len(toks):
    k, v = toks[i], (toks[i + 1] if i + 1 < len(toks) else "")
    idx = args.index(k) if k in args else -1
    if idx >= 0 and idx + 1 < len(args):
        args[idx + 1] = v
    else:
        args += [k, v]
    i += 2
# свип меряет скорость движка, а не поведение кэша: чекпоинты промптов дают
# только разброс (и отравляют ячейку при частичном попадании) — гасим во всех
# вариантах; продовое значение вернёт финальный restore_config.
# Переопределить: SWEEP_PROMPT_CACHE=6 ./bench_sweep.sh
force = {"--prompt-cache": os.environ.get("SWEEP_PROMPT_CACHE", "0")}
for k, v in force.items():
    idx = args.index(k) if k in args else -1
    if idx >= 0 and idx + 1 < len(args):
        args[idx + 1] = v
    else:
        args += [k, v]
cfg["args"] = args
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
print("  args: " + " ".join(args))
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
  # ждём, пока движок действительно уйдёт (иначе следующий старт может не найти VRAM)
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
      # нет /health — считаем готовым, когда отвечает сам API
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
  [[ "$spec" == "$entry" ]] && spec=""     # строки без "|" вообще
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

SUMMARY_BASE="${SUMMARY_BASE:-00-all-off}"
SUMMARY_NOISE="${SUMMARY_NOISE:-01-all-off-b}"
log ""
log "=== таблица прироста каждой фичи против эталона ($SUMMARY_BASE) ==="
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
    """prefill — только повтор 0 (повтор 1 попадает в prompt-cache); decode — максимум
    по повторам: генерация «не в масть» драфтеру даёт вторую моду скорости (~x2 медленнее,
    видели на chat-512: 70 или 34 tok/s) и без максимума ложно хоронит вариант."""
    d = json.load(open(path, encoding="utf-8"))
    pre, dec, tt = {}, {}, {}
    cached = []
    for r in d["runs"]:
        i = r["id"]
        if r.get("decode_tps", 0) > 0:
            dec[i] = max(dec.get(i, 0.0), r["decode_tps"])
        if r.get("repeat", 0) == 0:
            # предохранитель: повтор 0 с крошечным TTFT — попадание в prompt-cache
            # (бывает после прогревочных проходов); такой prefill в агрегат нельзя
            if r.get("prompt_tokens", 0) >= 256 and r.get("ttft_s", 99) < 1.0:
                cached.append(i)
                continue
            if r.get("prefill_tps", 0) > 0:
                pre[i] = r["prefill_tps"]
            tt[i] = r["ttft_s"]
    if cached:
        print(f"(warn) {os.path.basename(path)}: повтор 0 для {', '.join(sorted(set(cached)))} "
              f"похож на попадание в prompt-cache (TTFT < 1 c) — исключён из prefill")
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
        return "фича: " + name.split("-only-", 1)[1]
    if name.endswith("-all-on"):
        return "ВСЕ ФИЧИ ВМЕСТЕ"
    if name.startswith("99-"):
        return "ЭТАЛОН В КОНЦЕ (контроль дрейфа)"
    if name.startswith("15-"):
        return "ЭТАЛОН В СЕРЕДИНЕ (контроль дрейфа)"
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
guard = float(os.environ.get("DECODE_GUARD_PCT") or 0.0)  # жёсткий порог decode-регрессии, %

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
        dec_dn = d < -max(noise_d, guard)   # регрессия decode: за шум или за жёсткий порог
        pre_up, pre_dn = p > noise_p, p < -noise_p
        if dec_up: marks.append("decode ↑")
        if dec_dn: marks.append("decode ↓")
        if pre_up: marks.append("prefill ↑")
        if pre_dn: marks.append("prefill ↓")
        verdict = ("реальный эффект: " + ", ".join(marks)) if marks else "в пределах шума"
        if "-only-" in name:
            if dec_dn:
                rec = "❌ не внедрять: decode падает"
            elif dec_up or pre_up:
                rec = "✅ внедрять"
            else:
                rec = "— нейтрально"
    else:
        verdict = "в пределах шума (полоса шума не измерена)"
    rows.append((name, n, d, p, ta, tb, verdict, rec, note))

rows.sort(key=lambda r: -(r[2]))  # по приросту decode, сверху самые полезные
lines = []
lines.append(f"# Прирост каждой фичи против эталона (все фичи выключены) — {datetime.datetime.now():%Y-%m-%d %H:%M}")
lines.append("")
lines.append(f"* эталон: `{base_name}.json` — все фичи форка выключены")
if noise_d is not None:
    lines.append(f"* пол шума: decode ±{noise_d:.1f}%, prefill ±{noise_p:.1f}% "
                 f"(эталон, два прогона: `{base_name}` против `{noise_name}`)")
lines.append("* прирост = насколько вариант быстрее эталона; реален, только когда больше шума")
lines.append("* сумма приростов отдельных фич может не сходиться с «ВСЕ ФИЧИ ВМЕСТЕ» — фичи взаимодействуют")
if guard > 0:
    lines.append(f"* жёсткий порог decode-регрессии: −{guard:.1f}% (env DECODE_GUARD_PCT)")
lines.append("")
lines.append("| вариант | пар | decode прирост | prefill прирост | TTFT мед. эталон→вариант | вывод | рекомендация |")
lines.append("|---|---:|---:|---:|---|---|---|")
for name, n, d, p, ta, tb, v, rec, note in rows:
    lines.append(f"| {label(name)}{note} | {n} | {d:+.1f}% | {p:+.1f}% | {ta:.2f}s → {tb:.2f}s | {v} | {rec} |")
lines.append("")
lines.append("## Контроль decode (decode не должен падать после патчей)")
if noise_d is not None:
    thr = max(noise_d, guard)
    all_rows = [r for r in rows if r[0].endswith("-all-on")]
    if all_rows:
        d_all = all_rows[0][2]
        lines.append(f"* весь набор (ВСЕ ФИЧИ ВМЕСТЕ): decode {d_all:+.1f}% (порог регрессии −{thr:.1f}%) → "
                     + ("decode-безопасен" if d_all >= -thr else f"РЕГРЕССИЯ decode на {abs(d_all):.1f}%"))
    bad = [label(r[0]) for r in rows if "-only-" in r[0] and r[2] < -thr]
    lines.append("* фичи с реальным падением decode: " + (", ".join(bad) if bad else "нет"))

# детально по промптам: decode Δ каждого варианта против эталона
aggs = {nm: agg(os.path.join(out_dir, nm + ".json")) for nm in names}
lottery = []
for i in sorted({i for _p, dd, _t in aggs.values() for i in dd}):
    vals = [dd[i] for _p, dd, _t in aggs.values() if i in dd]
    if len(vals) >= 4 and max(vals) / min(vals) > 1.6:
        lottery.append(i)
if lottery:
    lines.append("")
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
