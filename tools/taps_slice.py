#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""taps_slice.py - вырезать начало дампа v2 (ETAP) в отдельный корректный файл.

    python3 tools/taps_slice.py <in.bin> <out.bin> --blocks 2000
    python3 tools/taps_slice.py <in.bin> <out.bin> --mb 400

Зачем: дампы весят гигабайты (один файл на сессию), а быстрый замер на голове
нужен по срезу.  Файл, обрезанный `head -c`, ломается на последнем неполном блоке -
этот режет ТОЛЬКО по границам блоков и закрывает файл терминатором n=0, так что
лоадер (scan_blocks) читает срез как обычный дамп.

Формат: заголовок 20 байт (magic, ver, n_embd, taps_n, has_emb), далее блоки
{u64 pos0, u32 n, i32 tokens[n], bf16 payload n*(taps_n+1)*n_embd}, конец - n=0.
"""
import os
import struct
import sys

MAGIC = 0x50415445


def main() -> int:
    args = sys.argv[1:]
    if len(args) < 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, dst = args[0], args[1]
    blocks_cap, mb_cap = None, None
    i = 2
    while i < len(args):
        if args[i] == "--blocks":
            blocks_cap = int(args[i + 1]); i += 2
        elif args[i] == "--mb":
            mb_cap = float(args[i + 1]) * 1024 * 1024; i += 2
        else:
            print(f"неизвестный аргумент {args[i]}", file=sys.stderr); return 2

    with open(src, "rb") as f:
        head = f.read(20)
        magic, ver, n_embd, taps_n, has_emb = struct.unpack("<5I", head)
        if magic != MAGIC or ver != 2:
            print(f"{src}: не дамп v2 (magic={magic:#x} ver={ver})", file=sys.stderr)
            return 1
        row_bytes = 2 * (taps_n + 1) * n_embd
        out = bytearray(head)
        rows = taken = 0
        while True:
            if blocks_cap is not None and taken >= blocks_cap:
                break
            if mb_cap is not None and len(out) >= mb_cap:
                break
            off = f.tell()
            hdr = f.read(12)
            if len(hdr) < 12:
                break
            pos0, n = struct.unpack("<QI", hdr)
            if n == 0:
                break
            body = f.read(4 * n + n * row_bytes)
            if len(body) < 4 * n + n * row_bytes:
                print(f"  ! {src}: хвостовой блок неполный (позиция {off}) - обрываю на нём", file=sys.stderr)
                break
            out += hdr + body
            taken += 1
            rows += n
        out += struct.pack("<QI", 0, 0)          # терминатор: лоадер останавливается чисто
        with open(dst, "wb") as g:
            g.write(out)
    print(f"{dst}: блоков {taken}, строк {rows}, {len(out) / 1e6:.1f} МБ "
          f"(из {os.path.getsize(src) / 1e6:.0f} МБ, n_embd {n_embd}, taps {taps_n})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
