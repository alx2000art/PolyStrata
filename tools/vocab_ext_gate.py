#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vocab_ext_gate.py - the tokenizer-swap gate: teacher-forced NLL per CHARACTER, base vs extended.

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
"""
import argparse
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def cmd_ids(a) -> int:
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_dir(a.pack)
    text = open(a.text, encoding="utf-8", errors="replace").read()
    text = "".join(l for l in text.splitlines(True) if not l.startswith("<<<DOC "))
    if a.max_chars:
        text = text[: a.max_chars]
    ids = tok.encode(text)
    json.dump({"version": "gate", "seg_tokens": len(ids), "chars": len(text), "pack": a.pack,
               "segments": [{"name": os.path.basename(a.text), "ids": ids}]},
              open(a.out, "w", encoding="utf-8"))
    print(f"[+] {a.out}: {len(ids)} токенов на {len(text)} символов "
          f"({len(text) / max(1, len(ids)):.2f} симв/токен, {a.pack})")
    return 0


def cmd_run(a) -> int:
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(a.ids))
    ids = st["segments"][0]["ids"]
    out = os.path.abspath(a.out)
    if os.path.exists(out):
        os.unlink(out)
    env = dict(os.environ)
    env.update(cfg.get("env") or {})
    if cfg.get("lib_dirs"):
        env["LD_LIBRARY_PATH"] = ":".join(list(cfg["lib_dirs"]) + [env.get("LD_LIBRARY_PATH", "")])
    for kv in a.env or []:
        k, _, v = kv.partition("=")
        env[k] = v
    args = [str(x) for x in (cfg.get("args") or [])]
    args = [x for x in args if x != "--serve"]
    args.append("--serve")
    err_path = out + ".stderr.txt"
    errf = open(err_path, "w")
    print(f"[*] {cfg['exe']} {' '.join(args)}", flush=True)
    p = subprocess.Popen([cfg["exe"], *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=errf, text=True, cwd=cfg.get("cwd"), env=env, bufsize=1)
    try:
        for line in p.stdout:
            if line.startswith("READY"):
                break
        else:
            raise SystemExit(f"движок завершился до READY (код {p.wait()}); см. {err_path}")
        p.stdin.write("LOGP " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        done = None
        rows = 0
        with open(out, "w", encoding="utf-8") as fo:
            for line in p.stdout:
                if line.startswith("LP "):
                    f = line.split()
                    fo.write(f"{f[1]}\t{f[2]}\t{f[3]}\n")
                    rows += 1
                    if rows % 500 == 0:
                        print(f"    {rows}/{len(ids) - 1}", flush=True)
                    continue
                if line.startswith(("LPDONE", "ERR")):
                    done = line.strip()
                    break
        if done is None or not done.startswith("LPDONE"):
            raise SystemExit(f"движок ответил {done!r}; см. {err_path}")
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
    finally:
        errf.close()
    n = sum(1 for _ in open(out, encoding="utf-8", errors="replace")) if os.path.exists(out) else 0
    print(f"[+] {out}: {n} позиций (ожидалось {len(ids)})")
    if n < len(ids) // 2:
        raise SystemExit("слишком мало позиций: промпт ушёл батчевым путём, --short-read не сработал")
    return 0


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
        if lp != lp:                                  # nan: the position was not scored
            continue
        tot += lp
        n += 1
    if not n:
        raise SystemExit(f"{a.tsv}: ни одной оценки")
    nll = -tot
    print(f"[+] {a.tsv}: позиций {n}, NLL {nll:.4f}, "
          f"бит/символ {nll / chars / math.log(2):.5f}, ppl/токен {math.exp(nll / n):.4f}")
    return 0


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

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
