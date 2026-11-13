#!/usr/bin/env python3
# tools/eagle3_convert.py - the EAGLE-3 draft head (AQ-MedAI's safetensors for
# DeepSeek-V4-Flash-0731) into a GGUF this fork's Eagle3 drafter class reads.
#
#   python3 tools/eagle3_convert.py <model.safetensors> <config.json> <out.gguf>
#
# The tensors keep their HF names (the class reads them as they lie); the BF16
# payloads copy byte for byte (GGML BF16 is the same layout); d2t/t2d become KVs
# (the arrays), as do the geometry, the rope and the tap layers from config.json.
import json
import struct
import sys

ALIGN = 32

def w_str(out, s):
    b = s.encode("utf-8")
    out.append(struct.pack("<Q", len(b)))
    out.append(b)

def w_kv_value(out, t, v):
    if t == 4:
        out.append(struct.pack("<I", v))
    elif t == 7:
        out.append(struct.pack("<B", 1 if v else 0))
    elif t == 10:
        out.append(struct.pack("<Q", v))
    elif t == 8:
        w_str(out, v)
    else:
        raise SystemExit(f"kv elem type {t} not written")

def w_kv(out, key, t, v):
    w_str(out, key)
    out.append(struct.pack("<I", t))
    if t == 4:
        out.append(struct.pack("<I", v))
    elif t == 6:
        out.append(struct.pack("<f", v))
    elif t == 7:
        out.append(struct.pack("<B", 1 if v else 0))
    elif t == 8:
        w_str(out, v)
    elif t == 9:
        at, items = v
        out.append(struct.pack("<I", at))
        out.append(struct.pack("<Q", len(items)))
        for x in items:
            w_kv_value(out, at, x)
    elif t == 10:
        out.append(struct.pack("<Q", v))
    else:
        raise SystemExit(f"kv type {t} not written")

def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: eagle3_convert.py <model.safetensors> <config.json> <out.gguf>")
    st_path, cfg_path, out_path = sys.argv[1:4]

    st = open(st_path, "rb")
    hlen, = struct.unpack("<Q", st.read(8))
    hdr = json.loads(st.read(hlen))
    base = 8 + hlen
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    eagle = cfg.get("eagle_config", {})
    taps = eagle["eagle_aux_hidden_state_layer_ids"]
    rs = cfg.get("rope_scaling", {})

    n_embd = cfg["hidden_size"]
    n_vocab = cfg["vocab_size"]
    dvocab = cfg["draft_vocab_size"]
    print(f"голова: {len(tensors)} тензоров, n_embd {n_embd}, draft vocab {dvocab}, taps {taps}")

    # the tensor bytes in file order (the safetensors offsets are non-decreasing)
    order = sorted(tensors.items(), key=lambda kv: kv[1]["data_offsets"][0])
    need = {"fc.weight", "lm_head.weight", "norm.weight", "midlayer.hidden_norm.weight"}
    names = {name for name, _ in order}
    missing = need - names
    if missing:
        raise SystemExit(f"в файле нет {sorted(missing)} - это не та голова?")

    # the vocab mappings ride as KVs, not tensors
    d2t_name = next((name for name, _ in order if "d2t" in name), None)
    t2d_name = next((name for name, _ in order if "t2d" in name), None)
    print(f"мэппинги вокаба: {d2t_name}, {t2d_name}")

    D = {"BF16": 30}  # this fork's GGML BF16 type
    blobs = []
    total = 0
    for name, ti in order:
        if name in (d2t_name, t2d_name):
            continue
        a, b = ti["data_offsets"]
        st.seek(base + a)
        dt = ti["dtype"]
        if dt not in D:
            raise SystemExit(f"dtype {dt} ({name}) не поддержан конвертером")
        blobs.append((name, ti["shape"], D[dt], st.read(b - a)))
        total += b - a
    print(f"тензоров к записи: {len(blobs)}, байт: {total}")

    # the KVs
    kvs = [
        ("general.architecture", 8, "eagle3"),
        ("general.name", 8, "DeepSeek-V4-Flash-0731-eagle3"),
        ("general.file_type", 4, 30),  # BF16
        ("eagle3.block_count", 4, 1),
        ("eagle3.embedding_length", 4, n_embd),
        ("eagle3.attention.head_count", 4, cfg["num_attention_heads"]),
        ("eagle3.attention.head_count_kv", 4, cfg["num_key_value_heads"]),
        ("eagle3.attention.key_length", 4, cfg["head_dim"]),
        ("eagle3.attention.value_length", 4, cfg["head_dim"]),
        ("eagle3.intermediate_size", 4, cfg["intermediate_size"]),
        ("eagle3.rms_norm_eps", 6, cfg["rms_norm_eps"]),
        ("eagle3.rope.freq_base", 6, cfg["rope_theta"]),
        ("eagle3.rope.scaling.type", 8, rs.get("type", "yarn")),
        ("eagle3.rope.scaling.factor", 6, rs.get("factor", 16.0)),
        ("eagle3.rope.scaling.original_context_length", 4, rs.get("original_max_position_embeddings", 65536)),
        ("eagle3.draft_vocab_size", 4, dvocab),
        ("eagle3.target_vocab_size", 4, n_vocab),
        ("eagle3.tap_layers", 9, (4, taps)),
    ]
    if d2t_name:
        d2t_ti = tensors[d2t_name]
        st.seek(base + d2t_ti["data_offsets"][0])
        raw = st.read(d2t_ti["data_offsets"][1] - d2t_ti["data_offsets"][0])
        kvs.append(("eagle3.d2t", 9, (10, list(struct.unpack(f"<{len(raw)//8}q", raw)))))
    if t2d_name:
        t2d_ti = tensors[t2d_name]
        st.seek(base + t2d_ti["data_offsets"][0])
        raw = st.read(t2d_ti["data_offsets"][1] - t2d_ti["data_offsets"][0])
        kvs.append(("eagle3.t2d", 9, (7, [x != 0 for x in raw])))

    # the tensor infos: the HF shape (out, in) reversed to the GGUF ne (the input dim fastest);
    # the offsets aligned, in the file order
    off = 0
    infos = []
    for name, shape, t, blob in blobs:
        ne = list(reversed(shape))
        while len(ne) > 1 and ne[-1] == 1:
            ne.pop()
        pad = (ALIGN - off % ALIGN) % ALIGN
        off += pad
        infos.append((name, ne, t, off, blob))
        off += len(blob)

    out = bytearray()
    out += b"GGUF"
    out += struct.pack("<I", 3)
    out += struct.pack("<Q", len(blobs))
    out += struct.pack("<Q", len(kvs))
    for k, t, v in kvs:
        w_kv(out, k, t, v)
    for name, ne, t, o, _ in infos:
        w_str(out, name)
        out += struct.pack("<I", len(ne))
        for d in ne:
            out += struct.pack("<Q", d)
        out += struct.pack("<I", t)
        out += struct.pack("<Q", o)
    out += b"\x00" * ((ALIGN - len(out) % ALIGN) % ALIGN)
    for _, _, _, _, blob in infos:
        out += blob

    with open(out_path, "wb") as f:
        f.write(out)
    print(f"написано: {out_path} ({len(out)} байт)")

    # the self-check: the header re-read
    chk = open(out_path, "rb")
    assert chk.read(4) == b"GGUF"
    v, = struct.unpack("<I", chk.read(4))
    nt, = struct.unpack("<Q", chk.read(8))
    nk, = struct.unpack("<Q", chk.read(8))
    ok = v == 3 and nt == len(blobs) and nk == len(kvs)
    print(f"проверка: version {v}, тензоров {nt}, kv {nk}: {'ок' if ok else 'НЕ СОШЛАСЬ'}")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
