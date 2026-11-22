#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_vocab_sidecar.py - a sidecar GGUF with the extended vocab's extra rows.

The sidecar carries two tensors, token_embd_ext.weight and output_ext.weight, of
[embedding_length, N] each, in the SAME types as the main file's tables (the engine reads
the token rows through one dequant path and rides the output rows inside its VRAM matrix).
Phase 1 writes ZERO rows: they dequantize to zero vectors, the new ids are unreachable
without the extended tokenizer, and the model answers bit-identically - the mechanism test.
The real rows come from the TokenAdapt synthesis (phase 2, --rows-file F32 [N, n_embd]).

    python3 make_vocab_sidecar.py --base <main.gguf> --n 4096 --out vocab_ext.gguf
"""
import argparse
import struct
import sys

# ggml: (bytes per block, elements per block); types absent here fall back to the gguf package
BLK = {
    0: (4, 1),      # F32
    1: (2, 1),      # F16
    2: (18, 32),    # Q4_0
    3: (20, 32),    # Q4_1
    6: (22, 32),    # Q5_0
    7: (24, 32),    # Q5_1
    8: (34, 32),    # Q8_0
    9: (36, 32),    # Q8_1
    10: (84, 256),  # Q2_K
    11: (110, 256),  # Q3_K
    12: (144, 256),  # Q4_K
    13: (176, 256),  # Q5_K
    14: (210, 256),  # Q6_K
    30: (20, 32),   # IQ4_NL
}

VAL_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_header(path):
    f = open(path, "rb")
    if f.read(4) != b"GGUF":
        raise SystemExit(f"{path}: not a GGUF")
    version, = struct.unpack("<I", f.read(4))
    tensors, kvs = struct.unpack("<QQ", f.read(16))

    def rstr():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def skip_val(t):
        if t == 8:
            rstr()
        elif t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            for _ in range(n):
                skip_val(et)
        else:
            f.read(VAL_SIZES[t])

    for _ in range(kvs):
        rstr()
        t, = struct.unpack("<I", f.read(4))
        skip_val(t)
    out = {}
    for _ in range(tensors):
        name = rstr()
        nd, = struct.unpack("<I", f.read(4))
        dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
        ty, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        out[name] = (dims, ty)
    return out, version


def row_size(ty, n_embd):
    if ty in BLK:
        b, e = BLK[ty]
    else:
        try:
            from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
            _, blck, size = GGML_QUANT_SIZES[GGMLQuantizationType(ty)]
            b, e = size, blck
        except Exception:
            raise SystemExit(f"ggml type {ty}: no size known - add it to BLK")
    if n_embd % e:
        raise SystemExit(f"ggml type {ty}: embedding_length {n_embd} is not a multiple of {e}")
    return n_embd // e * b


def rows_from_file(path, n, n_embd, ty):
    """Read [n, n_embd] F32 rows and quantize them to the main table's type."""
    import numpy as np
    a = np.fromfile(path, dtype="<f4")
    if a.size != n * n_embd:
        raise SystemExit(f"{path}: {a.size} floats, expected {n * n_embd} ({n} x {n_embd})")
    a = a.reshape(n, n_embd).astype(np.float32)
    if ty == 8:                                     # Q8_0, ggml reference: d = amax/127
        return q8_0_bytes(a)
    if ty == 1:
        return a.astype("<f2").tobytes()
    if ty == 0:
        return a.astype("<f4").tobytes()
    raise SystemExit(f"ggml type {ty}: --rows-* supports only F32/F16/Q8_0")


def q8_0_bytes(a):
    import numpy as np
    n, ne = a.shape
    if ne % 32:
        raise SystemExit("Q8_0 rows need n_embd % 32 == 0")
    nb = ne // 32
    x = a.reshape(n, nb, 32)
    amax = np.abs(x).max(axis=2)
    d = (amax / 127.0).astype(np.float16)
    inv = np.where(amax > 0, 127.0 / np.maximum(amax, 1e-30), 0.0).astype(np.float32)
    q = np.clip(np.rint(x * inv[:, :, None]), -127, 127).astype(np.int8)
    out = np.zeros((n, nb, 34), dtype=np.uint8)
    out[:, :, 0:2] = d.view(np.uint8).reshape(n, nb, 2)
    out[:, :, 2:34] = q.view(np.uint8).reshape(n, nb, 32)
    return out.tobytes()


def fill_rows(n, n_embd, ty, value):
    """A constant row: the output side's mask (exp(-30) is zero in the softmax)."""
    import numpy as np
    if ty == 8:
        return q8_0_bytes(np.full((n, n_embd), value, dtype=np.float32))
    if ty == 1:
        return np.full((n, n_embd), value, dtype="<f2").tobytes()
    if ty == 0:
        return np.full((n, n_embd), value, dtype="<f4").tobytes()
    raise SystemExit(f"ggml type {ty}: --fill-out supports only F32/F16/Q8_0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the main model GGUF (for n_embd and the table types)")
    ap.add_argument("--n", type=int, required=True, help="how many rows the extended vocab adds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-vocab", type=int, default=0,
                    help="the main file's vocab (default: read from token_embd.weight's shape)")
    ap.add_argument("--rows-emb", help="F32 [n, n_embd] rows for token_embd_ext (default: zeros)")
    ap.add_argument("--rows-out", help="F32 [n, n_embd] rows for output_ext (default: --fill-out)")
    ap.add_argument("--fill-out", type=float, default=0.0,
                    help="constant for output_ext when --rows-out is absent (-30 masks the tokens)")
    ap.add_argument("--name", default="vocab-ext")
    args = ap.parse_args()

    hdr, version = read_header(args.base)
    te = hdr.get("token_embd.weight")
    ow = hdr.get("output.weight")
    if not te:
        raise SystemExit("the main GGUF has no token_embd.weight")
    dims, ty_tok = te
    n_embd = dims[0]
    base_vocab = args.base_vocab or dims[1]
    ty_out = ow[1] if ow else ty_tok
    untied = ow is not None
    print(f"[*] base: n_embd={n_embd}, vocab={base_vocab}, token_embd type={ty_tok}, "
          f"output {'type=' + str(ty_out) if untied else 'TIED (no output.weight)'}")

    rs_tok = row_size(ty_tok, n_embd)
    rs_out = row_size(ty_out, n_embd) if untied else rs_tok
    print(f"[*] row sizes: token {rs_tok} B, output {rs_out} B; sidecar {args.n} rows -> "
          f"{args.n * (rs_tok + rs_out) / 1e6:.1f} MB")

    kvs = [
        ("general.name", (8, args.name)),
        ("ext.base_vocab", (10, base_vocab)),
        ("ext.count", (10, args.n)),
    ]
    tensors = [
        ("token_embd_ext.weight", [n_embd, args.n], ty_tok, rs_tok * args.n),
        ("output_ext.weight", [n_embd, args.n], ty_out, rs_out * args.n),
    ]

    # --- the GGUF: kvs, tensor infos, data (rows per id, row k <-> token base_vocab+k) ---
    align = 32
    head = bytearray()
    head += b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs))

    def kv_str(s):
        return struct.pack("<Q", len(s)) + s.encode()

    def kv_val(t, v):
        if t == 8:
            return kv_str(v)
        return struct.pack("<Q", v)   # the ext.* keys are u64

    for key, (t, v) in kvs:
        head += kv_str(key) + struct.pack("<I", t) + kv_val(t, v)
    infos = bytearray()
    off = 0
    for name, dims, ty, sz in tensors:
        infos += struct.pack("<Q", len(name)) + name.encode()
        infos += struct.pack("<I", len(dims)) + struct.pack("<" + "Q" * len(dims), *dims)
        infos += struct.pack("<IQ", ty, off)
        off += sz
    start = (len(head) + len(infos) + align - 1) // align * align
    head += infos + b"\0" * (start - len(head))

    data_tok = rows_from_file(args.rows_emb, args.n, n_embd, ty_tok) if args.rows_emb \
        else b"\0" * (rs_tok * args.n)
    if args.rows_out:
        data_out = rows_from_file(args.rows_out, args.n, n_embd, ty_out)
    elif args.fill_out != 0.0:
        data_out = fill_rows(args.n, n_embd, ty_out, args.fill_out)
        print(f"[*] output_ext: постоянные строки {args.fill_out} (маска, вероятность ~0)")
    else:
        data_out = b"\0" * (rs_out * args.n)
    if len(data_tok) != rs_tok * args.n or len(data_out) != rs_out * args.n:
        raise SystemExit("row payload size mismatch")
    head += data_tok + data_out
    if len(head) != start + rs_tok * args.n + rs_out * args.n:
        raise SystemExit(f"file layout mismatch: {len(head)} vs {start + (rs_tok + rs_out) * args.n}")
    open(args.out, "wb").write(head)
    print(f"[+] sidecar: {args.out} ({len(head)/1e6:.1f} MB, data at {start})")


if __name__ == "__main__":
    main()
