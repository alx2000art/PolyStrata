#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""engine_arg.py <config.json> KEY VALUE [KEY VALUE ...] - правка CLI-аргументов движка в конфиге.

    python3 tools/engine_arg.py strata-dsv4-ud-iq2_xxs.json --lookup 0

Ключ, которого в конфиге нет, добавляется; существующий - перезаписывается на месте
(порядок аргументов не меняется).  Зачем отдельный файл: аргументы движка живут в
env-блоке конфига, а не в вызывающем скрипте, и семь разных диагностик правят конфиг
каждая своим inline-python - второй такой же inline-блок на каждую правку гарантированно
разъедется.
"""
import json
import os
import sys


def main() -> int:
    raw = sys.argv[2:]
    dels = [p for p in raw if p.startswith("del:")]
    pairs = [p for p in raw if not p.startswith("del:")]
    if len(sys.argv) < 3 or (not pairs and not dels) or len(pairs) % 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    path = sys.argv[1]
    cfg = json.load(open(path, encoding="utf-8-sig"))
    args = [str(a) for a in (cfg.get("args") or [])]
    # "del:--flag" убирает ключ вместе со значением (leftover экспериментов в конфиге)
    for k in dels:
        name = k[4:]
        idx = args.index(name) if name in args else -1
        if idx >= 0:
            has_val = idx + 1 < len(args) and not args[idx + 1].startswith("--")
            del args[idx:(idx + 2 if has_val else idx + 1)]
    for i in range(0, len(pairs), 2):
        key, val = pairs[i], pairs[i + 1]
        idx = args.index(key) if key in args else -1
        if idx >= 0 and idx + 1 < len(args) and not args[idx + 1].startswith("--"):
            args[idx + 1] = val
        elif idx >= 0:                      # flag without a value: give it one
            args.insert(idx + 1, val)
        else:
            args += [key, val]
    cfg["args"] = args
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    print("  args: " + " ".join(args))
    return 0


if __name__ == "__main__":
    sys.exit(main())
