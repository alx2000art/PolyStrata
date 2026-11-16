# EAGLE-3 отладка: реестр и как убрать

Всё временное помечено в коде маркером **`DBG(eagle3)`** — убрать одним проходом:

```bash
grep -rn "DBG(eagle3)" src/        # что удалять
```

## Ключи окружения (в конфиге `"env"`)

| Ключ | Что делает | Чем заменить после отладки |
|---|---|---|
| `STRATA_EAGLE_STATS` | печатает точность top-1 первого драфта в конце ответа | оставить как диагностику либо удалить |
| `STRATA_EAGLE_ONE_DRAFT` | 1 драфт на раунд без весов политики (измерительный режим) | удалить |
| `STRATA_EAGLE_FORCE_GUESSES` | все драфты в окно, без учёта времени (медленно!) | удалить |
| `STRATA_EAGLE_DUMP` | пишет стадии головы (taps, fc, emb, ln1, hnorm, normed, logits) | удалить |
| `STRATA_EAGLE_DUMP_R` | пишет потоки трёх tap-слоёв | удалить |
| `STRATA_EAGLE_DUMP_ATT` | пишет q/k/v/o внимания первого teacher-шага (`tools/eagle_check_attn.py`); дамп снимается в `midlayer` сразу после `attend_batch` — дальше `o_` перезаписывает down-проекция | удалить |
| `STRATA_EAGLE_TEACH` | голова спрашивается на известных позициях (точность против реального следующего токена) | удалить |
| `STRATA_EAGLE_TEACH_PERM` | TEACH на всех 6 порядках трёх taps | удалить |
| `STRATA_EAGLE_ROPE_PLAIN` | отключает yarn-mscale головы (transformers применяет его только за 65536) | удалить (выбран plain) |
| `STRATA_EAGLE_CAT` | порядок полувходов конката: `eh` (по умолчанию) / `he` | удалить |
| `STRATA_EAGLE_TAPS` | выбор свёртки потоков: mean/mix/ffn/attn/s0..s3 | оставить победителя, остальное вырезать |
| `STRATA_EAGLE_TAP_SHIFT` | сдвиг номеров tap-слоёв | удалить (выбран 0) |

## Точки в коде

- `src/poly/eagle3.cu`
  - `dbg_inj_` — печать диапазона taps на первом инжекте
  - `dbg_att_` — печать выхода внимания на первом инжекте
  - `g_att_path` / `g_att_armed` / `g_att_done` — статические ворота дампа внимания (ставит `teacher_step`, снимает `midlayer`)
  - блок `STRATA_EAGLE_DUMP` в `inject()` (7 стадий в файл)
  - блок `STRATA_EAGLE_DUMP` в `draft_logits()` (дописывание logits)
  - `cat_he_` / `STRATA_EAGLE_CAT` — порядок полувходов (если выбран `eh`, удалить)
  - `teacher_step()` целиком (TEACH)
- `src/poly/eagle3.hpp` — поля `dbg_inj_`, `dbg_att_`, `dbg_logits_`, `dumped_`, `dumped_logits_`, `cat_he_`
- `src/poly/engine.cpp`
  - `eagle_dbg_` — печать `draft i/3: p=... draft_id -> target_id`
  - счётчик `probe: the draft equals its own input token`
  - `STRATA_EAGLE_TAP_SHIFT`, `STRATA_EAGLE_TAPS` (оставить только выбранный режим)
  - блок `STRATA_EAGLE_TEACH` в `eval()` (вся TEACH-логика + PERM)
  - синхронизации `draft after embed / draft_logits / likeliest` (их можно свернуть в одну)
- `src/poly/engine.hpp` — `eagle_dbg_`, `eagle_taps_`, `tap_row_`, `tap_row_slot_` (нужны только с отладкой taps)
- `src/poly/deepseek4.cpp` — `tap_state_mix` (оставить рабочий режим), `STRATA_EAGLE_DUMP_R`
- `src/poly/deepseek4.hpp` — `tap_rows_`
- `src/poly/main.cpp` — `eagle_stats`, `eagle_hit`, `eagle_tot`, печать `first guess`
- `src/poly/guesses.hpp` — `force()`, `one_draft()`

## Инструменты (оставить — они полезны и после порта)

- `tools/eagle_check_fc.py` — fc против файла
- `tools/eagle_check_stages.py` — каждая стадия против файла
- `tools/eagle_check_taps.py` — какая свёртка потоков нужна fc
- `tools/eagle_check_attn.py` — внимание против numpy-softmax (удалить вместе с DUMP_ATT)
- `tools/eagle3_convert.py` — конвертер головы (постоянный)
- `eagle_taps_ab.sh` — A/B представлений taps
- `diag_run.sh` — живой прогресс + отчёт
