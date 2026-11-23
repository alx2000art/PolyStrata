#!/usr/bin/env bash
# ============================================================================
# bench_sweep.sh — автоматическая серия A/B-бенчмарков PolyStrata (Ubuntu).
#
# Методика:
#   1. ЭТАЛОН (00-all-off): конфиг со ВСЕМИ фичами форка выключенными.
#      Прогоняется дважды (01-all-off-b) — разница прогонов = пол шума.
#   2. ВСЁ ВКЛЮЧЕНО (10-all-on): рабочий конфиг 2026-11-02 (WORKING_SPEC:
#      read-heat/file-reads/attend-batch выключены по вердикту живой
#      нагрузки, stage2 512 включён) — суммарный эффект набора.
#   3. Для каждой фичи (20+): эталон + включена ТОЛЬКО эта фича.  Это
#      изоляция механизма против all-off — её вердикты НЕ равны «внедрять в
#      прод» (read-heat/file-reads выиграли на синтетике и проиграли в
#      живую); решение по фиче — дельта против РАБОЧЕГО конфига.
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
SUITE_WARM="${SUITE_WARM:-long}"    # сьют прогрева: long хватало, doc-30k на прогрев не нужен
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
ONLY="${ONLY:-}"                    # ONLY="77,78" - гнать только эти варианты (по номеру или подстроке имени);
                                    # c NOW=1 поднимает их в начало списка.  Пусто = все.
WARM_CHECK="${WARM_CHECK:-1}"       # префлайт: файл-режим warm_start — пин читается на старте, запрос с
                                    # совпадающим префиксом переиспользует его (reused >= 2048). 0 = пропустить;
                                    # WARM_CHECK_ONLY=1 — только проверка, без свипа (конфиг восстанавливается).
BUDGET_CHECK="${BUDGET_CHECK:-1}"   # гейт скорости после свипа: steady-state мс/токен и префилл на прод-конфиге
                                    # (tools/decode_budget.py; MS_TOKEN / FLOOR_PREFILL_TPS задают пол)
WARM_CHECK_ONLY="${WARM_CHECK_ONLY:-0}"
WARM_CHECK_PROMPT="${WARM_CHECK_PROMPT:-}"   # текстовый файл для пина (>= 2048 токенов); иначе синтетический

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

# Рабочий конфиг 2026-11-02 — эталон для «поверх всего набора» (10-all-on и
# крутки-on).  Отличия от таблицы выше: read-heat и file-reads на живой
# нагрузке ЗАМЕДЛЯЮТ (вердикт свипа 11-01 на синтетике не перенёсся) — в
# проде 0; attend-batch в проде 0; stage2 512 включён.  only-ячейки остались
# против all-off: это изоляция механизма, не рекомендация.
WORKING_SPEC="--read-overlap 1 --file-reads 0 --read-heat 0 --attend-batch 0
--group-heads 16 --q8-dequant 1 --attend-chunks 1 --idx-mma 1 --cpu-tiled 1
--lookup 0 --stage2-mib 512"

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
VARIANTS+=("10-all-on|$WORKING_SPEC")       # рабочий конфиг 2026-11-02 (эталон набора)
# Драфтер и расширенный вокаб: 10-all-on уже несёт драфтер (он в args конфига);
# 16 снимает его для прямого A/B скорости с/без; 17 — сайдкар расширенного
# вокаба (--vocab-ext), 18 — явный путь драфтера (если конфиг без него).
DRAFT_MODEL="${DRAFT_MODEL:-$HOME/models/DeepSeek-V4-Flash-0731-eagle3.gguf}"
VOCAB_EXT="${VOCAB_EXT:-$HOME/models/vocab_ext_test.gguf}"
VARIANTS+=("16-no-drafter|$WORKING_SPEC DEL:--draft-model")
[[ -f "$DRAFT_MODEL" ]] && VARIANTS+=("18-drafter|$WORKING_SPEC --draft-model $DRAFT_MODEL")
[[ -f "$VOCAB_EXT" ]] && VARIANTS+=("17-vocab-ext|$WORKING_SPEC --vocab-ext $VOCAB_EXT")
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
# --- stage2 (второй staging-буфер). Реабилитирован 2026-10-31: «гонка» T1 была
# предсуществующим дрейфом тёплого состояния (кэш → другой сплит CPU/GPU и
# границы раундов → другой float-порядок) — контроль показал тот же дрейф при
# --stage2-mib 0; семантическая валидация 6/6 чисто на обоих режимах. База
# флага в FEATURES (эталоны гасят его явно), крутки перекрывают на месте.
VARIANTS+=("50-on-stage2-512|$WORKING_SPEC")                     # один в один 10-all-on — мера шума
VARIANTS+=("51-on-stage2-256|$WORKING_SPEC --stage2-mib 256")    # меньше отъёма кэша
VARIANTS+=("52-eta-stage2-512|$(all_off_spec) --stage2-mib 512")  # механизм в чистоте
# --- доля CPU-пула (поверх ВСЕГО НАБОРА: «фикс против auto» на проде).
# ВАЖНО: --cpu-tiled сам по себе булев (0/1) — "0.7" через atoi() = тихое
# ВЫКЛЮЧЕНИЕ tiled; долей работ рулит --cpu-share (auto = движок балансирует
# сам по таймингам). В проде ключа нет = auto, так что утечки в другие ячейки нет.
# --- собственное mma-ядро Q3_K продуктов (--expert-mma, ветка feat/expert-mma).  На бинарнике без
# флага сервер не поднимется и вариант будет пропущен с записью в лог.
VARIANTS+=("58-eta-expertmma|$(all_off_spec) --expert-mma 1")   # механизм в чистоте против эталона
VARIANTS+=("59-on-expertmma|$WORKING_SPEC --expert-mma 1")     # поверх всего набора
VARIANTS+=("55-on-share50|$WORKING_SPEC --cpu-share 0.5")
VARIANTS+=("56-on-share70|$WORKING_SPEC --cpu-share 0.7")
VARIANTS+=("57-on-share100|$WORKING_SPEC --cpu-share 1.0")
# поверх ЭТАЛОНА — прочие крутки (раскомментируйте нужное):
# VARIANTS+=("40-eta-pb2048|$(all_off_spec) --prompt-batch 2048")
# VARIANTS+=("41-eta-tailskip0|$(all_off_spec) --tail-skip 0")
# VARIANTS+=("42-eta-nocache|$(all_off_spec) --prompt-cache 0")
# VARIANTS+=("43-eta-stageread4096|$(all_off_spec) --stage-read-mib 4096")
# поверх ВСЁ-ВКЛЮЧЕНО — эффект крутки на рабочем конфиге:
# VARIANTS+=("45-on-pb2048|$(all_on_spec) --prompt-batch 2048")
# VARIANTS+=("46-on-convcache0|$(all_on_spec) --conversation-cache-mib 0")
# --- новые опции 2026-11-17: env-ключи движка (ENV:NAME=value; ENV:NAME= снимает) и
# экономика драфтера (DEL:--draft-model).  SCORE_CONTEXT здесь меряется только по скорости и
# отсутствию падений — его порчу на >30k промптах свип не докажет (для этого diag_run на doc-30k
# с семантической проверкой).  FREEZE на 16 ГБ может упасть на росте KV — вариант запишет фейл.
VARIANTS+=("60-eta-scorectx|$(all_off_spec) ENV:STRATA_POLY_SCORE_CONTEXT=30720")
VARIANTS+=("61-on-scorectx|$WORKING_SPEC ENV:STRATA_POLY_SCORE_CONTEXT=30720")
VARIANTS+=("62-eta-freeze|$(all_off_spec) ENV:STRATA_FREEZE_EXPERTS=1")
VARIANTS+=("63-on-freeze|$WORKING_SPEC ENV:STRATA_FREEZE_EXPERTS=1")
VARIANTS+=("64-eta-nodraft|$(all_off_spec) DEL:--draft-model")
VARIANTS+=("65-on-nodraft|$WORKING_SPEC DEL:--draft-model")
VARIANTS+=("66-on-lmq8off|$WORKING_SPEC ENV:STRATA_EAGLE_LM_Q8=0")
VARIANTS+=("99-all-off-c|$(all_off_spec)")   # контроль дрейфа (stage2 гасится самой спекой эталона)
# --- открытые клетки ревью 2026-11-23: размеры стейджинга, guess-reader, swap-settle.
# Их не было в свипе 11-11 (там --ram-experts-mib 6144 не менялся, --stage-mib 512 во всех ячейках,
# --expert-swaps не задавался = дефолт 64, STRATA_POLY_STAGE_SPEC не использовался).
# Каждая клетка ниже задаёт --stage-mib и --expert-swaps ЯВНО: apply_variant не сбрасывает ключ,
# которого нет в спеке, поэтому значение предыдущей ячейки протекло бы в следующую (класс «stale args»).
VARIANTS+=("70-on-stagespec32|$WORKING_SPEC --stage-mib 512 ENV:STRATA_POLY_STAGE_SPEC=32")
VARIANTS+=("71-on-stagespec16|$WORKING_SPEC --stage-mib 512 ENV:STRATA_POLY_STAGE_SPEC=16")
VARIANTS+=("72-on-stagespec0|$WORKING_SPEC --stage-mib 512 ENV:STRATA_POLY_STAGE_SPEC=0")
VARIANTS+=("73-on-stage128|$WORKING_SPEC --stage-mib 128")
VARIANTS+=("74-on-stage2048|$WORKING_SPEC --stage-mib 2048")
VARIANTS+=("75-on-swaps0|$WORKING_SPEC --stage-mib 512 --expert-swaps 0")
VARIANTS+=("76-on-swaps16|$WORKING_SPEC --stage-mib 512 --expert-swaps 16")
# Без полной синхронизации swap-стрима на старте токена (STRATA_POLY_SWAP_SYNC=0): слоты остаются
# зарезервированы до прихода события, follow() берёт приземлившиеся - токен не ждёт хост.
VARIANTS+=("77-on-swapnosync|$WORKING_SPEC ENV:STRATA_POLY_SWAP_SYNC=0")
# Один граф на пару «хвост слоя + голова следующего» вместо двух (STRATA_POLY_MERGE=1): 43 запуска
# cudaGraphLaunch на токен вместо 86.  Проверять вместе с 78-79 (контроль) - ломание захвата видно сразу.
VARIANTS+=("78-on-merge|$WORKING_SPEC ENV:STRATA_POLY_MERGE=1")

# --- CPU lane: родные iq256-ядра (iq_avx2.cpp) в poly.  Пара 80/81 меряет эффект, 82 отделяет down-проекции
# (STRATA_POLY_IQ256_DOWN=0 оставляет их на ggml).  80 — та же рабочая спека, что 10-all-on: контроль дрейфа
# рядом с 81, иначе разница между 10-all-on и 81 была бы разницей во времени, а не в лейне.
# 83/84: AVX-512-ядра (iq_avx512.cpp) от двух токенов (83, правило ядра) и на одиночном токене (84).
VARIANTS+=("80-eta-iq256off|$WORKING_SPEC")
VARIANTS+=("81-on-iq256|$WORKING_SPEC ENV:STRATA_POLY_IQ256=1")
VARIANTS+=("82-on-iq256-gu|$WORKING_SPEC ENV:STRATA_POLY_IQ256=1 ENV:STRATA_POLY_IQ256_DOWN=0")
VARIANTS+=("83-on-iq256-512|$WORKING_SPEC ENV:STRATA_POLY_IQ256=1 ENV:STRATA_POLY_IQ512=1")
VARIANTS+=("84-on-iq512-min1|$WORKING_SPEC ENV:STRATA_POLY_IQ256=1 ENV:STRATA_POLY_IQ512=1 ENV:STRATA_POLY_IQ512_MIN=1")

# ONLY: оставить только выбранные варианты.  NOW=1 - те же, но первыми (контроли 00/01/15/99 и
# рабочая спека уходят в конец, чтобы быстрый прогон не тратил время на них).
if [[ -n "$ONLY" ]]; then
    keep=()
    for entry in "${VARIANTS[@]}"; do
        name="${entry%%|*}"
        hit=0
        for want in ${ONLY//,/ }; do
            if [[ "$name" == "$want" || "$name" == *"$want"* ]]; then hit=1; break; fi
        done
        (( hit )) && keep+=("$entry")
    done
    if [[ "${NOW:-0}" == "1" ]]; then
        # поднять only-ячейки вперёд, остальные (контроли, эталоны) - после них
        rest=()
        for entry in "${VARIANTS[@]}"; do
            name="${entry%%|*}"
            hit=0
            for want in ${ONLY//,/ }; do
                [[ "$name" == "$want" || "$name" == *"$want"* ]] && { hit=1; break; }
            done
            (( hit )) || rest+=("$entry")
        done
        keep=("${keep[@]}" "${rest[@]}")
    fi
    VARIANTS=("${keep[@]}")
    printf '%s\n' "ONLY=$ONLY: вариантов ${#VARIANTS[@]}"   # log() is defined below this point
fi

# SKIP_CONTROLS=1: без дублей эталона и контролей дрейфа (01-all-off-b, 15-all-off-m, 99-all-off-c).
# Для быстрой проверки клеток: пол шума тогда не измерен (дельты <5% не считаем), зато прогон короче на
# три ячейки.  Полный свип их сохраняет - на них держится «реальный эффект / в пределах шума».
if [[ "${SKIP_CONTROLS:-0}" == "1" ]]; then
    drop=()
    for entry in "${VARIANTS[@]}"; do
        name="${entry%%|*}"
        [[ "$name" == "01-all-off-b" || "$name" == "15-all-off-m" || "$name" == "99-all-off-c" ]] \
            && continue
        drop+=("$entry")
    done
    VARIANTS=("${drop[@]}")
    log "SKIP_CONTROLS=1: вариантов ${#VARIANTS[@]}"
fi

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
# Маркер run-id.txt хранит сьют того, что лежит в папке: чужой забег или завершённый целиком — свежая
# папка с меткой времени, свой недоигранный — добирается на месте.
run_id="$SUITE"
id_file="$OUT_DIR/run-id.txt"
old_id="$(cat "$id_file" 2>/dev/null || echo '')"
has_json="$(ls "$OUT_DIR"/*.json 2>/dev/null | head -1)"
if [[ -n "$has_json" && "$old_id" != "$run_id" ]]; then
  new_dir="$OUT_DIR-$(date +%Y%m%d-%H%M%S)"
  log "в $OUT_DIR лежит забег '$old_id' — свежий ('$run_id') пишется в $new_dir"
  OUT_DIR="$new_dir"
  id_file="$OUT_DIR/run-id.txt"
  mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
fi
printf '%s
' "$run_id" > "$id_file"
complete=0
if [[ -f "$OUT_DIR/summary.md" ]]; then
  complete=1
  for entry in "${VARIANTS[@]}"; do
    [[ -f "$OUT_DIR/${entry%%|*}.json" ]] || { complete=0; break; }
  done
fi
if (( complete )) && ((${#VARIANTS[@]} > 0)); then
  new_dir="$OUT_DIR-$(date +%Y%m%d-%H%M%S)"
  log "предыдущий прогон в $OUT_DIR завершён целиком — свежий пишется в $new_dir"
  OUT_DIR="$new_dir"
  mkdir -p "$OUT_DIR" || die "не создать $OUT_DIR"
  printf '%s
' "$run_id" > "$OUT_DIR/run-id.txt"
fi
exec > >(tee -a "$OUT_DIR/sweep.log") 2>&1

log "=============================================================="
log "PolyStrata bench sweep (эталон: все фичи выключены)"
log "  конфиг:     $CONFIG  (порт $PORT, модель $MODEL)"
log "  версия:     $SWEEP_VERSION  (суффикс -dirty = файл правлен руками!)"
log "  base URL:   $BASE_URL"
log "  сьют:       $SUITE (прогрев: $SUITE_WARM), repeat $REPEAT, warmup $WARMUP, thinking $THINKING"
  log "  фич:        ${#FEATURES[@]}, вариантов: ${#VARIANTS[@]} (примерно 12-25 мин на вариант при suite poly)"
log "  результаты: $OUT_DIR"
log "=============================================================="

# ------------------------- работа с конфигом и сервером ---------------------
apply_variant() {  # $1 = spec пар "ключ значение ...", plus ENV:KEY=VALUE / ENV:KEY= / DEL:--flag
  python3 - "$CONFIG" "$1" <<'PY'
import json, os, sys
path, spec = sys.argv[1], sys.argv[2].strip()
cfg = json.load(open(path, encoding="utf-8-sig"))
args = [str(a) for a in (cfg.get("args") or [])]
env = cfg.get("env") or {}
# Every variant starts from a clean env for the keys this sweep manages: keys left in the config
# (or by a previous variant) must not leak into the baselines - the diag runs' stale TEACH/FREEZE
# lesson.  The model's own env (PIN_EXPERTS, STAGED_TILES, ...) stays.
MANAGED = ("STRATA_POLY_SCORE_CONTEXT", "STRATA_FREEZE_EXPERTS", "STRATA_POLY_STAGE2_DEBUG", "STRATA_POLY_STAGE_SPEC",
           "STRATA_POLY_SWAP_SYNC", "STRATA_POLY_IQ256", "STRATA_POLY_IQ256_DOWN", "STRATA_POLY_IQ256_CHECK",
           "STRATA_POLY_IQ512", "STRATA_POLY_IQ512_MIN")
for k in list(env):
    if k in MANAGED or k.startswith("STRATA_EAGLE_"):
        del env[k]
toks = spec.split()
bad = []
i = 0
while i < len(toks):
    k = toks[i]
    if k.startswith("ENV:"):                 # ENV:NAME=value sets it; ENV:NAME= removes it
        name, _, value = k[4:].partition("=")
        if value:
            env[name] = value
            if env.get(name) != value:
                bad.append(k)
        else:
            env.pop(name, None)
            if name in env:
                bad.append(k)
        i += 1
        continue
    if k.startswith("DEL:"):                 # DEL:--flag removes the flag and its value from args
        name = k[4:]
        idx = args.index(name) if name in args else -1
        if idx >= 0:
            has_val = idx + 1 < len(args) and not args[idx + 1].startswith("--")
            del args[idx:(idx + 2 if has_val else idx + 1)]
        if name in args:
            bad.append(k)
        i += 1
        continue
    v = (toks[i + 1] if i + 1 < len(toks) else "")
    idx = args.index(k) if k in args else -1
    if idx >= 0 and idx + 1 < len(args):
        args[idx + 1] = v
    else:
        args += [k, v]
    j = args.index(k) if k in args else -1
    if j < 0 or j + 1 >= len(args) or args[j + 1] != v:
        bad.append(f"{k} {v}")
    i += 2
# свип меряет скорость движка, а не поведение кэша: чекпоинты промптов дают
# только разброс (и отравляют ячейку при частичном попадании) — гасим во всех
# вариантах; продовое значение вернёт финальный restore_config.
# Переопределить: SWEEP_PROMPT_CACHE=6 ./bench_sweep.sh
# lookup (n-граммные догадки) делит ОДНО окно раунда с драфтером, а арбитр
# DraftPolicy::choose (src/spec/draft_policy.cpp) отдаёт раунд тому, кто по замерам
# платит больше - при включённом lookup часть раундов уходит n-граммам, и приём
# драфтера в ячейке получается смешанным.  Для свипа гасим его во всех вариантах;
# продовое значение вернёт финальный restore_config.  Вернуть: SWEEP_LOOKUP=3
force = {"--prompt-cache": os.environ.get("SWEEP_PROMPT_CACHE", "0"),
         "--lookup": os.environ.get("SWEEP_LOOKUP", "0")}
for k, v in force.items():
    idx = args.index(k) if k in args else -1
    if idx >= 0 and idx + 1 < len(args):
        args[idx + 1] = v
    else:
        args += [k, v]
cfg["args"] = args
cfg["env"] = env
# Свип по умолчанию меряет бит-стабильный greedy: без STRATA_FREEZE_EXPERTS сплит
# экспертов VRAM/RAM пересобирается, и два одинаковых запроса дают разные ответы
# (sweep 2026-11-23: completion_tokens 48..2048 на одном и том же промпте).
# Ячейка может задать своё значение в спеке; SWEEP_FREEZE=0 выключает.
if os.environ.get("SWEEP_FREEZE", "1") == "1" and "STRATA_FREEZE_EXPERTS" not in env:
    env["STRATA_FREEZE_EXPERTS"] = "1"
# Ячейки, которые мерят отладочные ключи (EAGLE_*, SCORE_CONTEXT, ...), обязаны снять
# мастер-гейт скраба движка (c8be1792): без STRATA_EAGLE_DEBUG=1 движок вычищает эти
# переменные, и ячейка тихо меряет базу.  Прод скрабится как раньше.
SCRUBBED = {"STRATA_POLY_LOOP_WATCH", "STRATA_POLY_STAGE2_DEBUG", "STRATA_POLY_STAGE2_SERIAL",
            "STRATA_POLY_MMQ_DEBUG", "STRATA_POLY_MMQ_TYPES", "STRATA_POLY_ROUTE_DUMP",
            "STRATA_POLY_TOPK_STATS", "STRATA_POLY_POOL_SELFTEST", "STRATA_FATTN", "STRATA_FDEC",
            "STRATA_NO_MMVQ_ROWS2", "STRATA_POLY_SCORE_CONTEXT", "STRATA_POLY_SCORE_MIB",
            "STRATA_VMM_RELEASE_WITH_ADDRESSES", "STRATA_LOGP_DEBUG"}
if any(k.startswith("STRATA_EAGLE_") or k in SCRUBBED for k in env) and "STRATA_EAGLE_DEBUG" not in env:
    env["STRATA_EAGLE_DEBUG"] = "1"
tmp = path + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(tmp, path)
print("  args: " + " ".join(args))
print("  env:  " + (" ".join(f"{k}={v}" for k, v in sorted(env.items())) or "(пусто)"))
# Проверка, что спека действительно применилась (копится в bad по ходу разбора).
# Пустая спека, потерянный ключ или неверное значение раньше давали «16 ячеек
# с одним конфигом» (sweep 2026-11-23) — теперь это падение, а не тихий базовый прогон.
if bad:
    print("  !!! спека не применилась: " + " ".join(bad), file=sys.stderr)
    sys.exit(3)
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

run_capture() {  # $1 = имя варианта, $2 = сьют (не задан: $SUITE)
  local name="$1" suite="${2:-$SUITE}" extra=()
  [[ "$THINKING" == "off" ]] && extra+=(--no-thinking)
  python3 "$AB" capture \
    --out "$OUT_DIR/$name.json" \
    --base-url "$BASE_URL" \
    --api-key "$API_KEY" \
    --model "$MODEL" \
    --suite "$suite" --repeat "$REPEAT" --warmup "$WARMUP" \
    --temperature "$TEMPERATURE" --seed "$SEED" \
    --timeout "$TIMEOUT_S" ${extra[@]+"${extra[@]}"} || return 1
  # Отпечаток кванта/модели и проверка детерминизма: старые файлы другого кванта
  # больше не попадут в сводку, а нестабильный greedy виден сразу.
  python3 - "$OUT_DIR/$name.json" "$CONFIG" <<'PY'
import hashlib, json, os, sys
p, cfgp = sys.argv[1], sys.argv[2]
d = json.load(open(p, encoding="utf-8"))
c = json.load(open(cfgp, encoding="utf-8-sig"))
args = [str(x) for x in (c.get("args") or [])]
model = None
for i, a in enumerate(args):
    if a == "--model" and i + 1 < len(args):
        model = args[i + 1]
fp = {"model": model}
if model and os.path.exists(model):
    st = os.stat(model)
    fp.update(size=st.st_size, mtime=int(st.st_mtime))
d.setdefault("meta", {})["fingerprint"] = fp
per = {}
for r in d["runs"]:
    per.setdefault(r["id"], set()).add(
        (r.get("completion_tokens"), hashlib.sha256((r.get("text") or "").encode("utf-8")).hexdigest()))
det = all(len(v) == 1 for v in per.values() if len(v) > 1)
d["meta"]["deterministic"] = det
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"  детерминизм повторов: {'да' if det else 'НЕТ — ответы расходятся, дельта не считается'}")
PY
}

variant_done() {  # $1 = имя: уже есть валидный захват СВОЕГО сьюта И ЭТОГО кванта?
  local f="$OUT_DIR/$1.json"
  [[ -f "$f" ]] || return 1
  python3 - "$f" "$SUITE" "$CONFIG" <<'PY'
import json, os, sys
f, suite, cfgp = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = json.load(open(f, encoding="utf-8"))
except Exception:
    sys.exit(1)
if not d.get("runs") or d.get("meta", {}).get("suite") != suite:
    sys.exit(1)
c = json.load(open(cfgp, encoding="utf-8-sig"))
args = [str(x) for x in (c.get("args") or [])]
model = None
for i, a in enumerate(args):
    if a == "--model" and i + 1 < len(args):
        model = args[i + 1]
cur = {"model": model}
if model and os.path.exists(model):
    st = os.stat(model)
    cur.update(size=st.st_size, mtime=int(st.st_mtime))
sys.exit(0 if d.get("meta", {}).get("fingerprint") == cur else 1)
PY
}

# ------------------------- префлайт: warm_start файл-режим -------------------
# Пин-файл {"prompt": <текст>} кладётся в конфиг, сервер рестартует: в логе должна
# появиться строка завершённого тёплого чтения, а запрос, ПРОДОЛЖАЮЩИЙ пин новым ходом
# (первый ход повторён дословно: тот же user-текст, ответ ассистента, новый вопрос),
# должен переиспользовать >= 2048 токенов (строка «... tokens = N reused» печатает
# сервер после каждого запроса).  Дописывать вопрос в ТОТ ЖЕ user-текст нельзя: промпт
# расходится до единственного чекпойнта (конца хода), и переиспользовать нечего.
# Конфиг после проверки восстанавливается из бэкапа (restore_config), сервер
# поднимается на исходных настройках.
warm_check() {
  local pin="$OUT_DIR/warm-check-request.json" text reused verdict=0 okboot=0 reuse_count=0
  if [[ -n "$WARM_CHECK_PROMPT" && -f "$WARM_CHECK_PROMPT" ]]; then
    text="$(cat "$WARM_CHECK_PROMPT")"
  else
    text="$(printf 'Проверка тёплого старта движка страта. %.0s' $(seq 1 400))"   # ~15k симв >= 2048 токенов
  fi
  log "  warm-check: пин-файл $pin ($(printf '%s' "$text" | wc -c) символов)"
  python3 - "$pin" "$text" <<'PY'
import json, sys
json.dump({"prompt": sys.argv[2]}, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False)
PY
  python3 - "$CONFIG" "$pin" <<'PY'
import json, os, sys
cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
cfg["warm_start"] = sys.argv[2]
with open(sys.argv[1] + ".tmp", "w", encoding="utf-8") as f:
    json.dump(cfg, f, ensure_ascii=False, indent=1)
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
PY
  log "  warm-check: конфиг warm_start -> $pin; рестарт ..."
  stop_server
  start_server
  if ! wait_ready; then
    log "  !!! warm-check: сервер не поднялся с пин-файлом — ПРОВАЛ"
    restore_config; stop_server; start_server; wait_ready || true
    return 1
  fi
  # ждём завершения тёплого чтения (запрос во время чтения останавливает его —
  # тогда переиспользуется только прочитанная часть, и проверка ложно падает)
  local deadline=$((SECONDS + 300))
  while (( SECONDS < deadline )); do
    grep -q "warm start: read " "$OUT_DIR/server.log.last" && { okboot=1; break; }
    sleep 2
  done
  (( okboot )) || log "  warm-check: тёплое чтение не завершилось за 300 s — продолжаю по факту"
  reused=$(python3 - "$BASE_URL" "$API_KEY" "$MODEL" "$pin" <<'PY'
import json, sys, urllib.request
base, key, model, pin = sys.argv[1:5]
text = json.load(open(pin, encoding="utf-8"))["prompt"]
body = json.dumps({"model": model,
                   "messages": [{"role": "user", "content": text},
                                {"role": "assistant", "content": "Готово."},
                                {"role": "user", "content": "Ответь ровно одним словом: готово"}],
                   "max_tokens": 8, "temperature": 0, "stream": False}).encode()
req = urllib.request.Request(base + "/chat/completions", data=body,
                             headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
print("ok" if r.get("choices") else "no-answer")
PY
)
  sleep 2   # строка «... tokens = N reused» печатается сервером после ответа
  reuse_count=$(grep -oE "prompt [0-9]+ tokens = [0-9]+ reused" "$OUT_DIR/server.log.last" | tail -1 \
                | grep -oE "[0-9]+ reused" | grep -oE "^[0-9]+")
  reuse_count=${reuse_count:-0}
  local srv_line
  srv_line=$(grep -oE "prompt [0-9]+ tokens = [0-9]+ reused.*" "$OUT_DIR/server.log.last" | tail -1)
  if [[ "$okboot" == 1 && "$reused" == "ok" && "$reuse_count" -ge 2048 ]]; then
    log "  warm-check: ПРОЙДЕНА — пин прочитан на старте, запрос переиспользовал $reuse_count токенов"
  else
    verdict=1
    log "  !!! warm-check: ПРОВАЛ (прочитано: $okboot, ответ: $reused, reused: $reuse_count) — лог: $OUT_DIR/server.log.last"
    log "      сервер сказал: ${srv_line:-<строки reused нет>}"
  fi
  rm -f "$pin"
  log "  warm-check: восстанавливаю конфиг и сервер ..."
  restore_config
  stop_server
  start_server
  wait_ready || log "  (!) сервер не поднялся после warm-check — смотри $OUT_DIR/server.log.last"
  return $verdict
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

warm_check_rc=0
if [[ "$WARM_CHECK" == "1" || "$WARM_CHECK_ONLY" == "1" ]]; then
  if warm_check; then
    log "warm-check: файл-режим warm_start работает"
  else
    warm_check_rc=1
    failed+=("warm-check")
    log "warm-check: ПРОВАЛ — файл-режим warm_start сломан (свит продолжается)"
  fi
fi
if [[ "$WARM_CHECK_ONLY" == "1" ]]; then
  log "WARM_CHECK_ONLY: свип пропущен"
  exit $warm_check_rc
fi

for entry in "${VARIANTS[@]}"; do
  name="${entry%%|*}"
  spec="${entry#|}"
  [[ "$spec" == "$entry" ]] && spec=""     # строки без "|" вообще
  if [[ -z "$spec" ]]; then
    log "  !!! у ячейки '$name' пустая спека — таблица VARIANTS собрана неверно."
    log "  !!! Останавливаюсь: иначе разные имена меряют один и тот же конфиг (так было 2026-11-23)."
    failed+=("$name")
    exit 1
  fi
  log ""
  log "=== $name ${spec:+($spec)} ====================================================="
  if variant_done "$name" && [[ "$FORCE" != "1" ]]; then
    log "  уже есть $OUT_DIR/$name.json — пропускаю (FORCE=1, чтобы переделать)"
    continue
  fi
  if ! apply_variant "$spec"; then
    rc=$?
    log "  !!! вариант $name не применён (код $rc) — пропускаю"
    failed+=("$name")
    if [[ "$rc" == "3" ]]; then
      log "  !!! спека и конфиг разошлись — дальше мерить нельзя, останавливаюсь"
      exit 1
    fi
    continue
  fi
  stop_server
  start_server
  if ! wait_ready; then
    log "  !!! сервер не поднялся на варианте $name — пропускаю (лог: $OUT_DIR/server.log.last)"
    failed+=("$name")
    continue
  fi
  # что движок получил на самом деле: env из /proc, а не из конфига
  EPID=$(pgrep -f "strata-poly.*--serve" | head -1)
  if [[ -n "$EPID" && -r "/proc/$EPID/environ" ]]; then
    tr '\0' '\n' < "/proc/$EPID/environ" | grep -E "^STRATA_" | sort | sed 's/^/  engine env: /'
  fi
  if (( WARM_PASSES > 0 )) && [[ ! -f "$OUT_DIR/.warmed" ]]; then
    for w in $(seq 1 "$WARM_PASSES"); do
      log "  прогрев $w/$WARM_PASSES: проход сьюта $SUITE_WARM без записи результатов ..."
      run_capture ".warm-$w" "$SUITE_WARM" >> "$OUT_DIR/warm.log" 2>&1 || log "  (прогрев не удался — продолжаю)"
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

# Гейт скорости: бюджет декода на восстановленном (продовом) конфиге.  Свип сравнивает варианты
# между собой, поэтому общий регресс («всё стало медленнее на 30%») в его таблице не виден - все
# дельты остаются около нуля.  Этот гейт ставит жёсткий пол: steady-state мс/токен и префилл.
if [[ "$BUDGET_CHECK" == "1" ]]; then
    log ""
    log "=== гейт скорости: бюджет декода (прод-конфиг) ========================="
    if CONFIG="$CONFIG" "$STRATA_DIR/.venv/bin/python" "$STRATA_DIR/tools/decode_budget.py"; then
        log "бюджет декода: PASS"
    else
        log "  !!! бюджет декода: FAIL - decode/prefill вне бюджета (MS_TOKEN/FLOOR_PREFILL_TPS)"
        failed+=("decode-budget")
    fi
fi

log ""
log "=== таблица прироста каждой фичи против эталона ($SUMMARY_BASE) ==="
python3 - "$OUT_DIR" "$SUMMARY_BASE" "$SUMMARY_NOISE" "$CONFIG" <<'PY'
import json, math, os, sys, datetime

out_dir, base_name, noise_name, cfg_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

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

# Только ячейки ЭТОГО кванта: старые файлы (до отпечатка) в сравнение не идут.
cfg = json.load(open(cfg_path, encoding="utf-8-sig"))
args0 = [str(x) for x in (cfg.get("args") or [])]
model = None
for i, a in enumerate(args0):
    if a == "--model" and i + 1 < len(args0):
        model = args0[i + 1]
cur = {"model": model}
if model and os.path.exists(model):
    st = os.stat(model)
    cur.update(size=st.st_size, mtime=int(st.st_mtime))

def fp_of(name):
    d = load(name)
    return (d or {}).get("meta", {}).get("fingerprint")

all_names = sorted(
    f[:-5] for f in os.listdir(out_dir)
    if f.endswith(".json") and f[:-5] != "summary" and load(f[:-5]) is not None
)
skipped = [n for n in all_names if fp_of(n) != cur]
names = [n for n in all_names if fp_of(n) == cur]
if base_name not in names:
    print(f"эталон {base_name}.json — другого кванта или нет отпечатка: сводка невозможна.")
    print("Запусти свип заново: variant_done теперь требует совпадения отпечатка и переснимет старые ячейки.")
    sys.exit(0)
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
    det = (load(name) or {}).get("meta", {}).get("deterministic")
    if det is False:
        verdict = "НЕДЕТЕРМИНИРОВАН (повторы расходятся) — дельта не считается"
        rec = "—"
    rows.append((name, n, d, p, ta, tb, verdict, rec, note))

rows.sort(key=lambda r: -(r[2]))  # по приросту decode, сверху самые полезные
lines = []
lines.append(f"# Прирост каждой фичи против эталона (все фичи выключены) — {datetime.datetime.now():%Y-%m-%d %H:%M}")
lines.append("")
lines.append(f"* эталон: `{base_name}.json` — все фичи форка выключены")
if noise_d is not None:
    lines.append(f"* пол шума: decode ±{noise_d:.1f}%, prefill ±{noise_p:.1f}% "
                 f"(эталон, два прогона: `{base_name}` против `{noise_name}`)")
if skipped:
    lines.append("* НЕ сравнивались (другой квант или нет отпечатка): " + ", ".join(skipped))
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

# CPU lane: 80 — та же рабочая спека без iq256; 81/82 — с ним.  Здесь считается ПАРА, а не против эталона:
# иначе эффект лейна тонет в сумме всех фич и в дрейфе между временами прогонов.
if "80-eta-iq256off" in names:
    lines.append("")
    lines.append("## CPU lane iq256 (пара к `80-eta-iq256off`)")
    for nm, lbl in (("81-on-iq256", "оба (gate/up + down)"), ("82-on-iq256-gu", "только gate/up"),
                    ("83-on-iq256-512", "iq512 на группах 2+ токена"), ("84-on-iq512-min1", "iq512 и на 1 токене")):
        if nm in names:
            try:
                nn, dd, pp, ta, tb = delta(os.path.join(out_dir, "80-eta-iq256off.json"),
                                           os.path.join(out_dir, nm + ".json"))
                lines.append(f"* {lbl}: decode {dd:+.1f}%, prefill {pp:+.1f}% ({nn} пар, TTFT {ta:.2f} → {tb:.2f}s)")
            except Exception as e:
                lines.append(f"* {lbl}: не удалось сравнить ({e})")

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
