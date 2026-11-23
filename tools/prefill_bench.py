#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prefill_bench.py - one long prompt through the engine's own GEN protocol, and its prompt timing.

    python3 tools/prefill_bench.py --config CFG --pack PACK --text FILE \
        [--max-chars 90000] [--max-new 1] [--out /tmp/prefill] [--env K=V ...]

Why not the server log: serve/server.py parses a DONE format this engine does not print, so no
`[strata] request ... prefill X tok/s` line appears.  Here the engine is started the way
vocab_ext_gate starts it, the prompt's ids go in as `GEN <max_new> <ids>`, and the engine's own
DONE line is read: `DONE <generated> <prompt> <prompt ms> <decode ms> ...` -> prompt tok/s.

Tokenization is the client's job (tools/strata_tokenizer.py), exactly as in production.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from vocab_ext_gate import load_cfg, start_engine, stop_engine, text_ids  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--max-chars", type=int, default=90000)
    ap.add_argument("--max-new", type=int, default=1)
    ap.add_argument("--out", default="/tmp/prefill")
    ap.add_argument("--env", action="append")
    args = ap.parse_args()

    text = open(args.text, encoding="utf-8", errors="replace").read()
    ids, chars = text_ids(args.pack, text, args.max_chars)
    print(f"[*] {chars} символов -> {len(ids)} токенов", flush=True)

    cfg = load_cfg(args.config)
    p, errf = start_engine(cfg, args.out + ".stderr.txt", args.env)
    done, gen = None, []
    try:
        p.stdin.write("GEN %d %s\n" % (args.max_new, ",".join(map(str, ids))))
        p.stdin.flush()
        for line in p.stdout:
            if line.startswith("T "):
                gen.append(int(line.split()[1]))
            if line.startswith(("DONE", "ERR")):
                done = line.strip()
                break
    finally:
        stop_engine(p, errf)

    print(f"[*] {done}")
    if done and done.startswith("DONE"):
        f = done.split()
        if len(f) >= 5:
            n, ms = int(f[2]), float(f[3])
            print(f"[+] prompt {n} токенов за {ms:.0f} ms = {n / ms * 1000:.1f} tok/s "
                  f"(decode {float(f[4]):.0f} ms, сгенерировано {int(f[1])})")
        print(f"[*] первые токены: {gen[:16]}")
        return 0
    print(f"[!] движок ответил {done!r}; см. {args.out}.stderr.txt", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
