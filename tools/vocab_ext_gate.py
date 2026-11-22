#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vocab_ext_gate.py - the tokenizer-swap gate: teacher-forced NLL per CHARACTER, base vs extended.

    sweep --config CFG.json --pack-base DIR --pack-ext DIR --sidecar F --text FILE --out DIR
          [--chars 12000] [--max-delta 0.02] [--env K=V ...]
        one command: builds the ids for both packs, edits --vocab-ext in the config, starts the
        engine once per phase (like bench_sweep: the lifecycle is the script's job), scores, stops,
        restores the config, prints bits/char for both and their delta.

    ids --pack DIR --text FILE --out IDS.json [--max-chars N]
    run --config CFG.json --ids IDS.json --out TSV [--env K=V ...]
    nll --tsv TSV --ids IDS.json | --chars N
    cmp --pack DIR --ext-pack DIR --text FILE [--max-chars N]

WHY PER CHARACTER.  The two tokenizations split the same text into a different number of tokens, so
per-token perplexity is not comparable between them; -sum(logp)/chars is.

The engine (`strata-poly`) takes token ids, never text - tokenization is the client's job with
tools/strata_tokenizer.py - and its `LOGP <ids>` command feeds each id on its own with want_logits
and prints `LP <pos> <id> <logp>` for the next id.  The tokenizer switch is therefore: the client
encodes with the extended pack, and the engine gets --vocab-ext <sidecar> so those ids have rows.

The production server must be stopped: every phase starts its own engine on the same port.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# args of the OTHER program (strata generate) that a leftover config edit can leave behind:
# strata-poly knows none of them and dies with a usage dump (exit 2) before READY.
GEN_ONLY = ("--pack", "--short-read", "--tokens", "--tokens-file")


# -------------------------------------------------------------------------------- ids
def text_ids(pack, text, max_chars):
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_dir(pack)
    text = "".join(l for l in text.splitlines(True) if not l.startswith("<<<DOC "))
    if max_chars:
        text = text[:max_chars]
    return tok.encode(text), len(text)


def write_ids(path, ids, chars, pack, name):
    json.dump({"version": "gate", "seg_tokens": len(ids), "chars": chars, "pack": pack,
               "segments": [{"name": name, "ids": ids}]},
              open(path, "w", encoding="utf-8"))


def cmd_ids(a) -> int:
    text = open(a.text, encoding="utf-8", errors="replace").read()
    ids, chars = text_ids(a.pack, text, a.max_chars)
    write_ids(a.out, ids, chars, a.pack, os.path.basename(a.text))
    print(f"[+] {a.out}: {len(ids)} токенов на {chars} символов "
          f"({chars / max(1, len(ids)):.2f} симв/токен, {a.pack})")
    return 0


# -------------------------------------------------------------------------------- engine life
def load_cfg(path):
    return json.load(open(path, encoding="utf-8-sig"))


def save_cfg(path, cfg):
    json.dump(cfg, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def set_vocab_ext(cfg, sidecar):
    args = [str(x) for x in (cfg.get("args") or [])]
    i = 0
    while i < len(args):
        if args[i] == "--vocab-ext":
            del args[i:i + 2]
        else:
            i += 1
    if sidecar:
        args += ["--vocab-ext", sidecar]
    cfg["args"] = args


def engine_args(cfg):
    args = [str(x) for x in (cfg.get("args") or [])]
    args = [x for x in args if x != "--serve"]
    for name in GEN_ONLY:
        i = 0
        while i < len(args):
            if args[i] == name:
                del args[i:i + 2]
            else:
                i += 1
    return args + ["--serve"]


def start_engine(cfg, err_path, env_extra):
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(list(cfg["lib_dirs"]) + [env.get("LD_LIBRARY_PATH", "")])
    for kv in env_extra or []:
        k, _, v = kv.partition("=")
        env[k] = v
    args = engine_args(cfg)
    errf = open(err_path, "w")
    print(f"[*] {cfg['exe']} {' '.join(args)}", flush=True)
    p = subprocess.Popen([cfg["exe"], *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=errf, text=True, cwd=cfg.get("cwd"), env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            return p, errf
    errf.close()
    raise SystemExit(f"движок завершился до READY (код {p.wait()}); см. {err_path}. "
                     f"Порт занят? Останови прод-сервер.")


def score(p, ids, tsv, title):
    p.stdin.write("LOGP " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    done = None
    rows = 0
    t0 = time.time()
    with open(tsv, "w", encoding="utf-8") as fo:
        for line in p.stdout:
            if line.startswith("LP "):
                f = line.split()
                fo.write(f"{f[1]}\t{f[2]}\t{f[3]}\n")
                rows += 1
                if rows % 1000 == 0:
                    print(f"    {title} {rows}/{len(ids) - 1} ({time.time() - t0:.0f} s)", flush=True)
                continue
            if line.startswith(("LPDONE", "ERR")):
                done = line.strip()
                break
    if done is None or not done.startswith("LPDONE"):
        raise SystemExit(f"{title}: движок ответил {done!r}")
    print(f"[+] {title}: {rows} позиций за {time.time() - t0:.0f} s", flush=True)
    return rows


def stop_engine(p, errf):
    try:
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
    except (BrokenPipeError, ValueError):
        p.kill()
    finally:
        errf.close()


# -------------------------------------------------------------------------------- nll
def nll_of(tsv, ids_path):
    st = json.load(open(ids_path, encoding="utf-8"))
    chars = st["chars"]
    limit = len(st["segments"][0]["ids"])
    tot, n = 0.0, 0
    for line in open(tsv, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        try:
            pos, lp = int(f[0]), float(f[2])
        except ValueError:
            continue
        if pos >= limit or lp != lp:
            continue
        tot += lp
        n += 1
    if not n:
        raise SystemExit(f"{tsv}: ни одной оценки")
    nll = -tot
    return n, chars, nll, nll / chars / math.log(2)


def cmd_nll(a) -> int:
    chars = a.chars
    limit = None
    if a.ids:
        st = json.load(open(a.ids))
        chars = chars or st["chars"]
        limit = len(st["segments"][0]["ids"])
    if not chars:
        raise SystemExit("нужен --ids или --chars")
    tot, n = 0.0, 0
    for line in open(a.tsv, encoding="utf-8", errors="replace"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 3:
            continue
        try:
            pos, lp = int(f[0]), float(f[2])
        except ValueError:
            continue
        if limit is not None and pos >= limit:
            continue
        if lp != lp:
            continue
        tot += lp
        n += 1
    if not n:
        raise SystemExit(f"{a.tsv}: ни одной оценки")
    nll = -tot
    print(f"[+] {a.tsv}: позиций {n}, NLL {nll:.4f}, "
          f"бит/символ {nll / chars / math.log(2):.5f}, ppl/токен {math.exp(nll / n):.4f}")
    return 0


# -------------------------------------------------------------------------------- cmp
def cmd_cmp(a) -> int:
    """Pure-ASCII lines must tokenize IDENTICALLY: the tail holds Cyrillic tokens only."""
    import strata_tokenizer as ST
    base = ST.Tokenizer.from_dir(a.pack)
    ext = ST.Tokenizer.from_dir(a.ext_pack)
    text = open(a.text, encoding="utf-8", errors="replace").read()
    if a.max_chars:
        text = text[: a.max_chars]
    ascii_n = diff_n = nonascii_n = 0
    saved = 0
    first = None
    for no, line in enumerate(text.splitlines(True), 1):
        ib = base.encode(line)
        ie = ext.encode(line)
        saved += len(ib) - len(ie)
        if all(ord(c) < 128 for c in line):
            ascii_n += 1
            if ib != ie:
                diff_n += 1
                if first is None:
                    first = (no, line[:60].rstrip("\n"), len(ib), len(ie))
        else:
            nonascii_n += 1
    print(f"[+] ASCII-строк {ascii_n}: разошлись {diff_n}; не-ASCII строк {nonascii_n}; "
          f"экономия {saved} токенов на файле")
    if diff_n:
        print(f"[!] первая разошедшаяся ASCII-строка {first[0]}: {first[1]!r} "
              f"({first[2]} -> {first[3]} токенов)")
        return 1
    print("[+] английский и код: id совпадают побитово")
    return 0


# -------------------------------------------------------------------------------- run (one phase)
def cmd_run(a) -> int:
    cfg = load_cfg(a.config)
    st = json.load(open(a.ids, encoding="utf-8"))
    ids = st["segments"][0]["ids"]
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        os.unlink(out)
    p, errf = start_engine(cfg, out + ".stderr.txt", a.env)
    try:
        score(p, ids, out, "run")
    finally:
        stop_engine(p, errf)
    n = sum(1 for _ in open(out, encoding="utf-8", errors="replace")) if os.path.exists(out) else 0
    print(f"[+] {out}: {n} позиций (ожидалось {len(ids) - 1})")
    if n < (len(ids) - 1) // 2:
        raise SystemExit("слишком мало позиций")
    return 0


# -------------------------------------------------------------------------------- sweep (both phases)
def cmd_sweep(a) -> int:
    os.makedirs(a.out, exist_ok=True)
    raw = open(a.config, "rb").read()
    try:
        text = open(a.text, encoding="utf-8", errors="replace").read()
        ib, chars = text_ids(a.pack_base, text, a.chars)
        ie, chars2 = text_ids(a.pack_ext, text, a.chars)
        if chars != chars2:
            raise SystemExit("the two packs saw different text lengths")
        pb = os.path.join(a.out, "base.ids.json")
        pe = os.path.join(a.out, "ext.ids.json")
        write_ids(pb, ib, chars, a.pack_base, os.path.basename(a.text))
        write_ids(pe, ie, chars, a.pack_ext, os.path.basename(a.text))
        print(f"[*] {chars} символов: база {len(ib)} токенов ({chars / len(ib):.2f} симв/ток), "
              f"ext {len(ie)} токенов ({chars / len(ie):.2f} симв/ток)")

        # phase A: the base tokenizer, no sidecar
        cfg = load_cfg(a.config)
        set_vocab_ext(cfg, None)
        save_cfg(a.config, cfg)
        t0 = time.time()
        p, errf = start_engine(cfg, os.path.join(a.out, "base.stderr.txt"), a.env)
        try:
            score(p, ib, os.path.join(a.out, "base.tsv"), "base")
        finally:
            stop_engine(p, errf)
        print(f"[*] базовая фаза: {time.time() - t0:.0f} s")

        # phase B: the extended tokenizer and its sidecar rows
        cfg = load_cfg(a.config)
        set_vocab_ext(cfg, a.sidecar)
        save_cfg(a.config, cfg)
        t0 = time.time()
        p, errf = start_engine(cfg, os.path.join(a.out, "ext.stderr.txt"), a.env)
        try:
            score(p, ie, os.path.join(a.out, "ext.tsv"), "ext")
        finally:
            stop_engine(p, errf)
        print(f"[*] ext-фаза: {time.time() - t0:.0f} s")
    finally:
        open(a.config, "wb").write(raw)
        print("[*] конфиг восстановлен")

    nb, _, nllb, bpcb = nll_of(os.path.join(a.out, "base.tsv"), pb)
    ne, _, nlle, bpce = nll_of(os.path.join(a.out, "ext.tsv"), pe)
    d = bpce - bpcb
    print()
    print(f"База:      позиций {nb}, NLL {nllb:.2f}, бит/символ {bpcb:.5f}")
    print(f"Расширенный: позиций {ne}, NLL {nlle:.2f}, бит/символ {bpce:.5f}")
    print(f"Дельта:    {d:+.5f} бит/символ (порог {a.max_delta:+.2f})")
    if d > a.max_delta:
        print("[!] ГЕЙТ НЕ ПРОЙДЕН")
        return 1
    print("[+] гейт пройден")
    return 0


# -------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ids")
    p.add_argument("--pack", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-chars", type=int, default=12000)
    p.set_defaults(fn=cmd_ids)

    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--env", action="append")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("nll")
    p.add_argument("--tsv", required=True)
    p.add_argument("--ids")
    p.add_argument("--chars", type=int, default=0)
    p.set_defaults(fn=cmd_nll)

    p = sub.add_parser("cmp")
    p.add_argument("--pack", required=True)
    p.add_argument("--ext-pack", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--max-chars", type=int, default=0)
    p.set_defaults(fn=cmd_cmp)

    p = sub.add_parser("sweep")
    p.add_argument("--config", required=True)
    p.add_argument("--pack-base", required=True)
    p.add_argument("--pack-ext", required=True)
    p.add_argument("--sidecar", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--chars", type=int, default=12000)
    p.add_argument("--max-delta", type=float, default=0.02)
    p.add_argument("--env", action="append")
    p.set_defaults(fn=cmd_sweep)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
