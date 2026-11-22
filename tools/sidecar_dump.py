#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sidecar_dump.py - what a vocab-ext sidecar really contains, row by row.

    python3 tools/sidecar_dump.py ~/models/vocab_ext_ru2.gguf

Prints the metadata, where the reader will put the data start (align 32 unless
general.alignment says otherwise), every tensor's dims/type/offset, and the dequantized
first and last row of each tensor.  The expected picture for a masked sidecar:
output_ext rows are a constant -30 (d = 30/127), token_embd_ext rows are small means.
"""
import json
import os
import struct
import sys

import numpy as np

VAL_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPES = {0: ("F32", 4, 1), 1: ("F16", 2, 1), 8: ("Q8_0", 34, 32), 14: ("Q6_K", 210, 256),
         12: ("Q4_K", 144, 256), 17: ("IQ2_XS", 74, 256), 16: ("IQ2_XXS", 66, 256)}


def main() -> int:
    path = sys.argv[1]
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        version, = struct.unpack("<I", f.read(4))
        nt, nkv = struct.unpack("<QQ", f.read(16))

        def rstr():
            n, = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t == 8:
                return rstr()
            if t == 9:
                et, = struct.unpack("<I", f.read(4))
                n, = struct.unpack("<Q", f.read(8))
                return [rval(et) for _ in range(n)]
            fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
                   10: "<Q", 11: "<q", 12: "<d"}[t]
            return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

        kv = {}
        for _ in range(nkv):
            k = rstr()
            t, = struct.unpack("<I", f.read(4))
            kv[k] = rval(t)
        infos = []
        for _ in range(nt):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            ty, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            infos.append((name, dims, ty, off))
        pos = f.tell()
        align = int(kv.get("general.alignment", 32) or 32)
        start = (pos + align - 1) // align * align
        print(f"{path}: v{version}, tensors {nt}, kv {nkv}, file {os.path.getsize(path)}")
        print(f"  kv: {json.dumps({k: v for k, v in kv.items()}, ensure_ascii=False, default=str)}")
        print(f"  header end {pos}, alignment {align}, data_start {start}")
        for name, dims, ty, off in infos:
            tn, b, e = TYPES.get(ty, (f"type{ty}", 0, 0))
            row = dims[0] // e * b if e else 0
            print(f"  {name}: dims={dims} {tn} offset={off} row={row} B")
            if e and row and off + row <= os.path.getsize(path) - start:
                f.seek(start + off)
                rows = []
                for r in (0, dims[1] - 1):
                    f.seek(start + off + r * row)
                    buf = f.read(row)
                    if tn == "Q8_0":
                        dt = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
                        a = np.frombuffer(buf, dtype=dt)
                        v = (a["q"].astype(np.float32) * a["d"].astype(np.float32)[:, None]).reshape(-1)
                    elif tn == "F16":
                        v = np.frombuffer(buf, "<f2").astype(np.float32)
                    elif tn == "F32":
                        v = np.frombuffer(buf, "<f4").astype(np.float32)
                    else:
                        rows.append(f"row {r}: (no dequant here)")
                        continue
                    rows.append(f"row {r}: min {v.min():.4f} max {v.max():.4f} mean {v.mean():.6f}")
                print("    " + "; ".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
