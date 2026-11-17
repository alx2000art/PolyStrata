#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle_dump_corpus.py - the fine-tune's data: run a corpus through the engine with the taps dumped.

The engine writes one block a read when STRATA_EAGLE_DUMP_TAPS=<dir> is set (engine env, i.e. the
config's "env" block).  This script slices the corpus at line boundaries and sends each slice as one
request with max_tokens=1 - the read is all we need; the answer costs a token.  Every slice lands
as one block of {pos0, n, tokens[n], taps[n][3][n_embd] bf16} (~24.6 KB a row).

    python3 eagle_dump_corpus.py --setup        # put the env key into the config, then restart the server
    python3 eagle_dump_corpus.py --corpus ~/corpus/ru.txt --chars 80000

Notes:
- the dump directory must be given in the config's env before the server starts; --setup does it and
  prints what to do next;
- request a fresh conversation per slice (different text), so the engine re-reads it; the template's
  shared head is reused and its positions are not dumped - they carry no training value anyway;
- the last row of every block has no label: the trainer drops it.
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))


def config_path() -> str:
    return os.environ.get("CONFIG") or os.path.join(ROOT, "strata-dsv4-ud-iq2_xxs.json")


def load_cfg(path: str) -> dict:
    return json.load(open(path, encoding="utf-8-sig"))


def setup(path: str, dump_dir: str) -> None:
    cfg = load_cfg(path)
    cfg.setdefault("env", {})["STRATA_EAGLE_DUMP_TAPS"] = os.path.abspath(os.path.expanduser(dump_dir))
    # each dump chunk must be read from position 0 as its own conversation: a reused template prefix
    # leaves the block without the rows its taps attended, and the trainer's causal attention would
    # differ from the engine's at every row
    args = [str(a) for a in (cfg.get("args") or [])]
    for k, v in (("--conversation-cache-mib", "0"), ("--prompt-cache", "0")):
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
    print(f"env: STRATA_EAGLE_DUMP_TAPS={cfg['env']['STRATA_EAGLE_DUMP_TAPS']} -> {path}")
    print("args: кэши разговоров и промптов выключены (блоки дампа самодостаточны)")
    print("перезапустите сервер (./setup.sh) и запустите этот скрипт без --setup")
    print("после дампа верните кэши (продовое значение) обычной правкой конфига")


def chunks(lines, chars: int):
    part, size = [], 0
    for ln in lines:
        if part and size + len(ln) > chars:
            yield "".join(part)
            part, size = [], 0
        part.append(ln)
        size += len(ln)
    if part:
        yield "".join(part)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", help="файл корпуса (текст UTF-8)")
    ap.add_argument("--chars", type=int, default=80000, help="символов в чанке (~25k токенов)")
    ap.add_argument("--setup", action="store_true", help="прописать STRATA_EAGLE_DUMP_TAPS в env конфига")
    ap.add_argument("--dump-dir", default=None, help="каталог дампа (по умолчанию ~/eagle_taps)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    path = args.config or config_path()
    dump_dir = args.dump_dir or os.path.expanduser("~/eagle_taps")
    if args.setup:
        os.makedirs(os.path.expanduser(dump_dir), exist_ok=True)
        setup(path, dump_dir)
        return 0
    if not args.corpus:
        print("укажи --corpus FILE (или --setup)", file=sys.stderr)
        return 2
    cfg = load_cfg(path)
    key = cfg.get("api_key") or ""
    port = cfg.get("port") or 8080
    host = cfg.get("host") or "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    model = cfg.get("model_name") or "my-model"
    env = cfg.get("env") or {}
    if not env.get("STRATA_EAGLE_DUMP_TAPS"):
        print("внимание: в конфиге нет STRATA_EAGLE_DUMP_TAPS — дамп не пишется (см. --setup)", file=sys.stderr)
    text = open(os.path.expanduser(args.corpus), encoding="utf-8", errors="replace").read()
    parts = list(chunks(text.splitlines(keepends=True), args.chars))
    print(f"корпус {len(text)} символов, чанков {len(parts)} по ~{args.chars} символов")
    url = f"http://{host}:{port}/v1/chat/completions"
    t0 = time.time()
    done = 0
    for i, part in enumerate(parts, 1):
        body = json.dumps({
            "model": model, "max_tokens": 1, "temperature": 0.0,
            "messages": [{"role": "user", "content": part}],
        }).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        for attempt in (1, 2, 3):
            try:
                with urllib.request.urlopen(req, timeout=1800) as r:
                    json.loads(r.read())
                break
            except Exception as e:  # noqa: BLE001 - transient load/restart errors are retried
                if attempt == 3:
                    print(f"\nчанк {i} не прошёл: {e}", file=sys.stderr)
                    return 1
                time.sleep(5)
        done += len(part)
        el = time.time() - t0
        print(f"  {i}/{len(parts)}: {done} символов, {el:.0f} s, {done / max(el, 1e-9):.0f} симв/с", flush=True)
    files = sorted(glob.glob(os.path.join(os.path.expanduser(dump_dir), "taps-*.bin")))
    total = sum(os.path.getsize(f) for f in files)
    print(f"готово: {len(files)} файл(ов) дампа, {total / 1e9:.1f} ГБ в {os.path.expanduser(dump_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
