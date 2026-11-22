#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""token_adapt_rows.py - synthesize the extended vocab's rows from their base pieces.

    python3 tools/token_adapt_rows.py \
        --pack ~/PolyStrata-data/packs/dsv4-ud-iq2_xxs/tokenizer \
        --ext  ~/tokenizer_ru_pruned \
        --base ~/models/ds/D:modelsdsv4-hf/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf \
        --out-emb ~/rows_emb.f32

For every tail token (id >= the pack's vocab size, in id order) the token's raw bytes are
segmented by the BASE pack's own merge ranks, the base token_embd rows of those pieces are
dequantized and averaged, and the F32 row is written in tail order (row k <-> token BASE+k).

Mean, not sum: the residual stream is RMS-normalized right after the embedding, so the scale
cancels and the average is the conservative direction mix.  The output side is NOT synthesized
here - it stays a constant mask (-30) so the model cannot emit untrained tokens.

The pack must be byte-mapped (the DeepSeek/llama.cpp convention: Ġ for space, Ð-style Cyrillic).
A raw-UTF-8 pack is rejected rather than silently mis-segmented.
"""
import argparse
import contextlib
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strata_tokenizer as ST  # noqa: E402

import numpy as np  # noqa: E402

B2U = ST.bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def gguf_header(path):
    """(tensor name -> (dims, type, offset), data_start) for one shard."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit(f"{path}: not a GGUF")
        struct.unpack("<I", f.read(4))
        nt, nkv = struct.unpack("<QQ", f.read(16))
        alignment = 32

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

        for _ in range(nkv):
            key = rstr()
            t, = struct.unpack("<I", f.read(4))
            v = rval(t)
            if key == "general.alignment":
                alignment = int(v)
        infos = {}
        for _ in range(nt):
            name = rstr()
            nd, = struct.unpack("<I", f.read(4))
            dims = struct.unpack("<" + "Q" * nd, f.read(8 * nd))
            ty, = struct.unpack("<I", f.read(4))
            off, = struct.unpack("<Q", f.read(8))
            infos[name] = (dims, ty, off)
        pos = f.tell()
    return infos, (pos + alignment - 1) // alignment * alignment


def dequant(buf, ty, n_embd):
    if ty == 8:                                     # Q8_0: [f16 d][32 int8]
        dt = np.dtype([("d", "<f2"), ("q", "i1", (32,))])
        a = np.frombuffer(buf, dtype=dt)
        return (a["q"].astype(np.float32) * a["d"].astype(np.float32)[:, None]).reshape(-1)
    if ty == 1:
        return np.frombuffer(buf, "<f2").astype(np.float32)
    if ty == 0:
        return np.frombuffer(buf, "<f4").astype(np.float32)
    raise SystemExit(f"ggml type {ty}: this tool reads Q8_0/F16/F32 only")


class Table:
    def __init__(self, path, data_start, ty, off, n_embd, row_bytes):
        self.f = open(path, "rb")
        self.base = data_start + off
        self.ty = ty
        self.row_bytes = row_bytes
        self.n_embd = n_embd
        self.cache = {}

    def row(self, i):
        v = self.cache.get(i)
        if v is not None:
            return v
        self.f.seek(self.base + i * self.row_bytes)
        buf = self.f.read(self.row_bytes)
        if len(buf) != self.row_bytes:
            raise SystemExit(f"short read at row {i}")
        v = dequant(buf, self.ty, self.n_embd)
        if len(self.cache) < 200000:
            self.cache[i] = v
        return v


def find_tables(base_path, rows):
    """Locate token_embd/output across the split shards and return {name: Table}."""
    d = os.path.dirname(os.path.abspath(base_path))
    stem = os.path.basename(base_path).split("-of-")[0]
    import glob
    shards = sorted(glob.glob(os.path.join(d, stem + "-of-*.gguf"))) or [base_path]
    found = {}
    for path in shards:
        infos, data_start = gguf_header(path)
        for name in rows:
            if name in infos and name not in found:
                dims, ty, off = infos[name]
                n_embd = dims[0]
                blk = {8: (34, 32), 1: (2, 1), 0: (4, 1)}[ty]
                if n_embd % blk[1]:
                    raise SystemExit(f"{name}: n_embd {n_embd} is not a multiple of {blk[1]}")
                found[name] = Table(path, data_start, ty, off, n_embd, n_embd // blk[1] * blk[0])
    missing = [r for r in rows if r not in found]
    if missing:
        raise SystemExit(f"tensors not found in {shards}: {missing}")
    return found


def token_bytes_of(tok):
    try:
        return bytes(U2B[ch] for ch in tok)
    except KeyError:
        raise SystemExit(f"token {tok!r} is not byte-mapped - this tool needs the byte-level pack")


def segment(tk, raw):
    """The base pack's own BPE over the token's bytes -> base token ids."""
    syms = [B2U[b] for b in raw]
    for s in syms:
        if s not in tk.ids:
            raise SystemExit(f"no base token for byte {ord(s):#04x}")
    while len(syms) > 1:
        best = None
        for k in range(len(syms) - 1):
            merged = syms[k] + syms[k + 1]
            if merged not in tk.ids:
                continue
            r = tk.ranks.get((syms[k], syms[k + 1]))
            if r is not None and (best is None or r < best[0]):
                best = (r, k)
        if best is None:
            break
        k = best[1]
        syms[k:k + 2] = [syms[k] + syms[k + 1]]
    return [tk.ids[s] for s in syms]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="base pack dir (the oracle tokenizer)")
    ap.add_argument("--ext", required=True, help="extended tokenizer dir (pruned)")
    ap.add_argument("--base", required=True, help="first shard of the main model")
    ap.add_argument("--out-emb", required=True, help="F32 [n, n_embd] rows for token_embd_ext")
    ap.add_argument("--out-out", help="optional F32 rows for output_ext (default: none)")
    args = ap.parse_args()

    tk = ST.Tokenizer.from_dir(args.pack)
    base = len(tk.tokens)
    vocab = json.load(open(os.path.join(args.ext, "vocab.json"), encoding="utf-8"))
    tail = sorted((i, t) for t, i in vocab.items() if i >= base)
    if not tail:
        raise SystemExit("the extended vocab has no tail")
    if tail[0][0] != base or tail[-1][0] != base + len(tail) - 1:
        raise SystemExit("the tail is not contiguous from the base size")
    if sum(1 for i in vocab.values() if i < base) != base:
        raise SystemExit("the base prefix is not intact")

    print(f"[*] база {base}, хвост {len(tail)}; пак: {len(tk.tokens)} токенов, {len(tk.ranks)} мерджей")
    tables = find_tables(args.base, ["token_embd.weight"] + (["output.weight"] if args.out_out else []))
    emb = tables["token_embd.weight"]
    out = tables.get("output.weight")
    n_embd = emb.n_embd
    print(f"[*] n_embd={n_embd}, token_embd {emb.ty}/{emb.row_bytes} B, "
          f"output {out.ty if out else '-'}")

    hist = {}
    samples = []
    n_out = open(args.out_out, "wb") if args.out_out else None
    total = 0
    with open(args.out_emb, "wb") as fe:
        for k, (_, tok) in enumerate(tail):
            raw = token_bytes_of(tok)
            pieces = segment(tk, raw)
            v = np.mean([emb.row(i) for i in pieces], axis=0)
            fe.write(v.astype("<f4").tobytes())
            if n_out is not None:
                vo = np.mean([out.row(i) for i in pieces], axis=0)
                n_out.write(vo.astype("<f4").tobytes())
            hist[len(pieces)] = hist.get(len(pieces), 0) + 1
            total += len(pieces)
            if len(samples) < 6:
                samples.append((raw.decode("utf-8", "replace").replace("\n", " "),
                                [tk.decode([i], errors="replace") for i in pieces]))
    if n_out is not None:
        n_out.close()

    size = len(tail) * n_embd * 4
    got = os.path.getsize(args.out_emb)
    if got != size:
        raise SystemExit(f"{args.out_emb}: {got} bytes, expected {size}")
    print(f"[+] кусков на токен: среднее {total / len(tail):.2f}, гистограмма {dict(sorted(hist.items()))}")
    for text, pieces in samples:
        print(f"    {text!r} -> {pieces}")
    print(f"[+] {args.out_emb}: {got/2**20:.1f} MiB для {len(tail)} строк")
    return 0


if __name__ == "__main__":
    sys.exit(main())
