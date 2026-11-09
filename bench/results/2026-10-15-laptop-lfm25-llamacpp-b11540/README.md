# LFM2.5-24B-A2B UD-Q4_K_XL on the 8 GB laptop: PolyStrata and llama.cpp b11540 (2026-10-15)

RTX 4070 Laptop (8 GB), i7-14700HX, 32 GB of RAM, Windows 10. `bench/bench_vs_llama.py run --machine <file>
--rounds 3`: each server in turn, a warm-up, then prompts of about 4,000 and 32,000 tokens (`prompt-4k.txt`,
`prompt-32k.txt`) with a 36,864 context, at temperature 0 and 0.7, and a summary of the text above asked for.
`summary.txt` is `bench_vs_llama.py summary` over the folder; `raw/` has every request.

| Config | Engine | Settings |
| --- | --- | --- |
| `polystrata-setup` | strata-poly of 165bb4da | the run config setup writes for this laptop (`strata-cfg-polystrata-setup.json`) |
| `polystrata-split` | 4f5947f5: attention in blocks of rows | the same config |
| `polystrata-flash` | d74c5e25: a prompt batch's attention in registers | the same config |
| `polystrata-batch` | d74c5e25 | the config with `--prompt-batch 2048`, as setup writes it from 36fdc62e on |
| `polystrata-last` | 4a644a8d | the same |
| `llama-tuned` | llama-server b11540 | `-ngl 99 --n-cpu-moe 25 -fa on -ctk q8_0 -ctv q8_0 -t 20 -ub 512`, the fastest of the settings tried |
| `llama-fit` | llama-server b11540 | `--fit on -fa on -ctk q8_0 -ctv q8_0` |

The PolyStrata configs ran the same file with the engine of each commit (`machine-laptop*.json`). With the last
one:

| | PolyStrata (4a644a8d) | llama.cpp, `llama-tuned` | llama.cpp, `llama-fit` |
| --- | ---: | ---: | ---: |
| Reads about 4,000 tokens | 2,193 to 2,215 tokens/s | 498 to 505 | 483 to 485 |
| Writes after them | 89 to 94 tokens/s | 57.7 to 58.0 | 55.1 to 55.3 |
| Reads about 32,000 tokens | 3,543 to 3,545 tokens/s | 528 to 537 | 505 to 516 |
| Writes after them | 71 to 73 tokens/s | 52.8 to 53.0 | 50.9 to 51.0 |
| First token after 32,000 | 8.8 s | 57.8 to 58.8 s | 60.2 to 61.5 s |
