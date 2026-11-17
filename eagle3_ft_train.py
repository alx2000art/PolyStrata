#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle3_ft_train.py - файн-тюн головы EAGLE-3 на дампе taps (v2) движка.

    python3 eagle3_ft_train.py --safetensors ~/llama_server/models/DeepSeek-V4-Flash-0731-eagle3.safetensors \
        --dump ~/eagle_taps --out eagle3-ft.safetensors \
        [--epochs 2 --lr 1e-4 --max-rows 1024 --coverage]

Что внутри:
- данные: блоки дампа v2 {pos0, n, tokens[n], [embs|taps] bf16}; пара (строка p) -> метка tokens[p+1];
- модель: ровно та, что в чекпоинте и движке (fc(12288->4096), один Llama-слой GQA 64:8 со
  смежными парами rope, YaRN-интерполяция с corr-границами, эффективный множитель 1.0),
  веса стартуют из safetensors чекпоинта;
- метки: токены вне draft-словаря (t2d=false) маскируются (-100); --coverage печатает долю
  покрываемых позиций - потолок, который голова в принципе может достигнуть;
- сохранение: те же имена тензоров в safetensors -> tools/eagle3_convert.py -> GGUF.

Данные остаются на месте: тренер читает блоки потоково, порядок блоков перемешивается по сиду.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import struct
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("нет torch: pip install torch --index-url https://download.pytorch.org/whl/cu128 "
          "(или просто pip install torch для CPU)", file=sys.stderr)
    raise

# ----------------------------- дамп v2 -------------------------------------

MAGIC = 0x50415445  # "ETAP"


def scan_blocks(paths):
    """Метаданные блоков без чтения payload: (path, offset, pos0, n, tokens).  Данные ленивые."""
    out, n_embd, taps_n = [], 0, 0
    for path in paths:
        with open(path, "rb") as f:
            magic, ver, n_embd, taps_n, has_emb = struct.unpack("<5I", f.read(20))
            if magic != MAGIC or ver != 2 or has_emb != 1:
                raise SystemExit(f"{path}: не дамп v2 (magic={magic:#x}, ver={ver})")
            while True:
                off = f.tell()
                head = f.read(12)
                if len(head) < 12:
                    break
                pos0, n = struct.unpack("<QI", head)
                if n == 0:
                    break
                tokens = np.frombuffer(f.read(4 * n), dtype="<i4").copy()
                f.seek(2 * n * (taps_n + 1) * n_embd, 1)
                out.append((path, off, pos0, n, tokens))
    return out, n_embd, taps_n


def load_block(path: str, off: int, n: int, n_embd: int, taps_n: int):
    """Один блок с диска: (pos0, tokens, embs f32[n, E], taps f32[n, K, E])."""
    with open(path, "rb") as f:
        f.seek(off)
        pos0 = struct.unpack("<QI", f.read(12))[0]
        tokens = np.frombuffer(f.read(4 * n), dtype="<i4")
        rows = np.frombuffer(f.read(2 * n * (taps_n + 1) * n_embd), dtype="<u2")             .reshape(n, taps_n + 1, n_embd)
    embs = _bf16_to_f32(rows[:, 0, :]).astype(np.float32)
    taps = _bf16_to_f32(rows[:, 1:, :]).astype(np.float32)
    return pos0, tokens, embs, taps


def _bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


def bf16(t) -> "torch.Tensor":
    return (t.view(torch.int32) + 0x7FFF + ((t.view(torch.int32) >> 16) & 1)).view(torch.int16) \
        if False else t


# ----------------------------- rope (порт rope_kernel) ----------------------

def yarn_phase(pos: "torch.Tensor", iw_half: "torch.Tensor", base: float, freq_scale: float,
               ext_factor: float, corr0: float, corr1: float, n_rot: int):
    """theta и mscale по yarn(): theta_interp/extrap и их смесь на corr-границах.

    Порт rope_kernel: theta_extrap = pos * base ** (-iw/ (n_rot/2)), iw = 2*i0_half;
    эффективный mscale = attn_factor * (1 + 0.1*ln(1/freq_scale)); у нашей головы
    attn_factor = 1 / (1 + 0.1*ln(1/freq_scale)), то есть произведение = 1.0 ('effective 1').
    """
    theta_scale = base ** (-2.0 / n_rot)
    theta_extrap = pos.float().unsqueeze(-1) * (base ** (-iw_half.float() * 2.0 / n_rot)).unsqueeze(0)
    theta_interp = freq_scale * theta_extrap
    y = (iw_half.float().unsqueeze(0) - corr0) / max(0.001, corr1 - corr0)
    ramp = (1.0 - torch.clamp(y, 0.0, 1.0)) * ext_factor
    theta = theta_interp * (1 - ramp) + theta_extrap * ramp
    mscale = 1.0 + 0.1 * math.log(1.0 / freq_scale)      # = 1.0 при our attn_factor (см. доку)
    return theta, mscale


def rope_apply(x: "torch.Tensor", pos: "torch.Tensor", r) -> "torch.Tensor":
    """x [T, H, D]: смежные пары (2i, 2i+1), все D повернуты (off=0, n_rot=D), off иного нет."""
    T, H, D = x.shape
    half = D // 2
    iw = torch.arange(half, device=x.device)
    theta, mscale = yarn_phase(pos, iw, r["base"], r["freq_scale"], r["ext_factor"], r["corr0"],
                               r["corr1"], D)
    c, s = torch.cos(theta) * mscale, torch.sin(theta) * mscale        # [T, half]
    a = x[:, :, 0::2]
    b = x[:, :, 1::2]
    c = c.unsqueeze(1)
    s = s.unsqueeze(1)
    x0 = a * c - b * s
    x1 = a * s + b * c
    out = torch.stack((x0, x1), dim=-1).reshape(T, H, D)
    return out


def rmsnorm(x: "torch.Tensor", w: "torch.Tensor", eps: float) -> "torch.Tensor":
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


# ----------------------------- модель ---------------------------------------

class Eagle3Head(nn.Module):
    """fc -> [ln1(emb) | hnorm(h)] -> один Llama-слой (GQA, смежные пары rope) -> out_norm -> lm_head."""

    def __init__(self, n_embd: int, taps_n: int, n_head: int, n_kv: int, head_dim: int, n_ff: int,
                 dvocab: int, eps: float, rope: dict):
        super().__init__()
        self.E = n_embd
        self.n_head, self.n_kv, self.hd = n_head, n_kv, head_dim
        self.eps = eps
        self.rope = rope
        self.fc = nn.Linear(taps_n * n_embd, n_embd, bias=False)
        self.ln1 = nn.RMSNorm(n_embd, eps)      # только вес: w в forward ниже
        self.hnorm = nn.RMSNorm(n_embd, eps)
        self.ln2 = nn.RMSNorm(n_embd, eps)
        self.out_norm = nn.RMSNorm(n_embd, eps)
        self.wq = nn.Linear(2 * n_embd, n_head * head_dim, bias=False)
        self.wk = nn.Linear(2 * n_embd, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(2 * n_embd, n_kv * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, n_embd, bias=False)
        self.gate = nn.Linear(n_embd, n_ff, bias=False)
        self.up = nn.Linear(n_embd, n_ff, bias=False)
        self.down = nn.Linear(n_ff, n_embd, bias=False)
        self.lm_head = nn.Linear(n_embd, dvocab, bias=False)

    def forward(self, embs, taps, pos):
        # embs [T, E], taps [T, K, E], pos [T] (абсолютные позиции строк блока)
        T = embs.shape[0]
        h = self.fc(taps.reshape(T, -1))
        x = torch.cat((rmsnorm(embs, self.ln1.weight, self.eps),
                       rmsnorm(h, self.hnorm.weight, self.eps)), dim=-1)          # [T, 2E]
        q = self.wq(x).view(T, self.n_head, self.hd)
        k = self.wk(x).view(T, self.n_kv, self.hd)
        v = self.wv(x).view(T, self.n_kv, self.hd)
        q = rope_apply(q, pos, self.rope)
        k = rope_apply(k, pos, self.rope)
        # GQA: голова h -> kvh = h // per (смежные группы, проверено check_attn);
        # scores каждой строки против ВСЕХ ключей [T_key]: [Tq, n_kv, per, Tk]
        per = self.n_head // self.n_kv
        qg = q.view(T, self.n_kv, per, self.hd)
        att = torch.einsum("tqph,skh->tqps", qg, k) / math.sqrt(self.hd)   # [Tq, n_kv, per, Tk]
        i = torch.arange(T, device=embs.device)
        causal = (i.view(-1, 1, 1, 1) >= i.view(1, 1, 1, -1))              # [Tq, 1, 1, Tk]
        w = torch.softmax(att.masked_fill(~causal, float("-inf")), dim=-1)
        o = torch.einsum("tqps,skh->tqph", w, v)                             # [Tq, n_kv, per, hd]
        o = o.reshape(T, self.n_head * self.hd)
        hidden = h + self.wo(o)
        mlp = self.down(F.silu(self.gate(rmsnorm(hidden, self.ln2.weight, self.eps))) * self.up(rmsnorm(hidden, self.ln2.weight, self.eps)))
        hidden = hidden + mlp
        normed = rmsnorm(hidden, self.out_norm.weight, self.eps)
        return self.lm_head(normed)


# ----------------------------- веса из safetensors --------------------------

ST_MAP = {
    "fc.weight": ("fc", "weight"),
    "midlayer.input_layernorm.weight": ("ln1", "weight"),
    "midlayer.hidden_norm.weight": ("hnorm", "weight"),
    "midlayer.self_attn.q_proj.weight": ("wq", "weight"),
    "midlayer.self_attn.k_proj.weight": ("wk", "weight"),
    "midlayer.self_attn.v_proj.weight": ("wv", "weight"),
    "midlayer.self_attn.o_proj.weight": ("wo", "weight"),
    "midlayer.post_attention_layernorm.weight": ("ln2", "weight"),
    "midlayer.mlp.gate_proj.weight": ("gate", "weight"),
    "midlayer.mlp.up_proj.weight": ("up", "weight"),
    "midlayer.mlp.down_proj.weight": ("down", "weight"),
    "norm.weight": ("out_norm", "weight"),
    "lm_head.weight": ("lm_head", "weight"),
}


def load_safetensors(path: str) -> dict:
    try:
        from safetensors.torch import load_file
        return load_file(path)
    except ImportError:
        # минимальный ридер: заголовок u32 len + json + сырые блоки
        import numpy as np
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]      # длина заголовка - u64 в safetensors
            hdr = json.loads(f.read(n))
        out = {}
        base = 8 + n
        dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
              "I64": torch.int64, "I32": torch.int32, "F64": torch.float64, "BOOL": torch.bool}
        with open(path, "rb") as f:
            for name, ti in hdr.items():
                if name == "__metadata__":
                    continue
                a, b = ti["data_offsets"]
                f.seek(base + a)
                raw = f.read(b - a)
                t = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())
                out[name] = t.view(dt[ti["dtype"]]).reshape(ti["shape"])
        return out


def load_model(st_path: str, head_cfg: dict, eps: float, device: str) -> tuple:
    sd = load_safetensors(st_path)
    fcw = sd["fc.weight"]
    n_embd = fcw.shape[0]                        # [E, 3E]
    taps_n = fcw.shape[1] // n_embd
    hd = int(head_cfg.get("head_dim", 64))
    wq = sd["midlayer.self_attn.q_proj.weight"]
    n_head = wq.shape[0] // hd
    n_kv = sd["midlayer.self_attn.k_proj.weight"].shape[0] // hd
    n_ff = sd["midlayer.mlp.gate_proj.weight"].shape[0]
    dvocab = sd["lm_head.weight"].shape[0]
    # rope: параметры головы (eagle config); corr-границы - формулой из make_rope
    base = head_cfg.get("rope_theta", 10000.0)
    rs = head_cfg.get("rope_scaling") or {}
    factor = rs.get("factor", 16.0)
    n_ctx_orig = rs.get("original_max_position_embeddings", 65536)
    ext = 1.0 if rs.get("rope_type", rs.get("type")) == "yarn" else 0.0
    n_rot = head_cfg.get("head_dim", 64)
    beta_fast, beta_slow = 32.0, 1.0
    corr = lambda rot: n_rot * math.log(n_ctx_orig / (rot * 2 * math.pi)) / (2 * math.log(base))  # noqa: E731
    rope = {"base": base, "freq_scale": 1.0 / factor, "ext_factor": ext,
            "corr0": math.floor(corr(beta_fast)), "corr1": math.ceil(corr(beta_slow))}
    m = Eagle3Head(n_embd, taps_n, n_head, n_kv, hd, n_ff, dvocab, eps, rope).to(device)
    with torch.no_grad():
        for st_name, (mod, par) in ST_MAP.items():
            w = sd[st_name]
            # ggml хранит [out, in]; nn.Linear.weight тоже [out, in] - порядок совпадает
            getattr(m, mod).weight.copy_(w.to(device).float())
    d2t = sd.get("d2t.weight", sd.get("d2t"))
    t2d = sd.get("t2d.weight", sd.get("t2d"))
    return m, dict(n_embd=n_embd, taps_n=taps_n, n_head=n_head, n_kv=n_kv, dvocab=dvocab,
                   d2t=d2t, t2d=t2d)


# ----------------------------- обучение -------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", default=None, help="config.json головы (rope/eps); иначе дефолты")
    ap.add_argument("--dump", required=True, help="каталог дампа v2 (taps-*.bin)")
    ap.add_argument("--out", default="eagle3-ft.safetensors")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-rows", type=int, default=1024, help="строк блока за шаг (хвост отбрасывается)")
    ap.add_argument("--coverage", action="store_true", help="только отчёт по покрытию d2t и выход")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    head_cfg = {}
    if args.config:
        head_cfg = json.load(open(args.config, encoding="utf-8"))
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    print(f"модель: E {meta['n_embd']}, голов {meta['n_head']}:{meta['n_kv']}, dvocab {meta['dvocab']}, "
          f"device {device}")

    # t2d: какие target-токены есть в draft-словаре (метки вне - маска)
    t2d = meta.get("t2d")
    t2d_t = None
    if t2d is not None:
        t2d_t = (torch.as_tensor(np.asarray(t2d).astype(np.int32)) != 0)
        print(f"t2d: {int(t2d_t.sum())} из {len(t2d_t)} target-токенов в draft-словаре")
    # обратная карта меток: target -> draft строится из d2t (target = draft + d2t[draft]) - иначе CE
    # сравнивает id целевого словаря с классами draft-словаря (разные пространства)
    tgt2draft = None
    d2t = meta.get("d2t")
    if d2t is not None:
        d2t_np = np.asarray(d2t).astype(np.int64)
        span = int((d2t_np + np.arange(len(d2t_np))).max()) + 1
        tgt2draft = np.full(span, -100, dtype=np.int64)
        for draft_i, off in enumerate(d2t_np):
            tgt = draft_i + int(off)
            if 0 <= tgt < span and tgt2draft[tgt] == -100:
                tgt2draft[tgt] = draft_i
        print(f"d2t: {int((tgt2draft >= 0).sum())} target-токенов имеют draft-id")

    if os.path.isfile(os.path.expanduser(args.dump)):
        files = [os.path.expanduser(args.dump)]        # задан сам файл дампа - берём его
    else:
        files = sorted(glob.glob(os.path.join(args.dump, "taps-*.bin")))
    blocks, _, _ = scan_blocks(files)
    if not blocks:
        print("нет блоков дампа", file=sys.stderr)
        return 1
    rows_total = sum(n for _, _, _, n, _ in blocks)
    print(f"блоков {len(blocks)}, строк {rows_total} (данные читаются с диска на каждом шаге)")

    if args.coverage:
        cov = tot = 0
        for _, _, _, n, tokens in blocks:
            if t2d_t is None:
                cov += len(tokens) - 1
                tot += len(tokens) - 1
                continue
            lab = np.asarray(tokens[1:])
            m = t2d_t[lab]
            cov += int(m.sum())
            tot += len(lab)
        print(f"покрытие d2t: {cov} из {tot} позиций ({100 * cov / max(tot, 1):.1f}%) — "
              f"это потолок top-1 головы на этом корпусе")
        return 0

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(42)
    steps = 0
    for epoch in range(args.epochs):
        order = list(range(len(blocks)))
        rng.shuffle(order)
        tot_loss, tot_rows = 0.0, 0
        for bi in order:
            path, off, pos0, n_rows, tokens_meta = blocks[bi]
            # блок целиком в CPU f32 (одна загрузка с диска), нарезается на шаги по max_rows строк:
            # память шага ограничена, а данные используются полностью.  Каждая нарезка - каузальная
            # последовательность со своими абсолютными позициями (срез учится на локальном контексте,
            # как и в штатном обучении EAGLE на последовательностях ограниченной длины)
            pos0_b, tokens_b, embs_b, taps_b = load_block(path, off, n_rows,
                                                          meta["n_embd"], meta["taps_n"])
            for start in range(0, n_rows - 1, args.max_rows):
                length = min(args.max_rows, n_rows - 1 - start)
                if length < 8:
                    break
                x_embs = torch.from_numpy(embs_b[start:start + length]).to(device)
                x_taps = torch.from_numpy(taps_b[start:start + length]).to(device)
                pos = torch.arange(pos0_b + start, pos0_b + start + length, device=device)
                # bf16-автокаст: веса в fp32, матмулы в bf16 - активации и логиты вдвое меньше
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits = model(x_embs, x_taps, pos)              # [n, dvocab]
                    if tgt2draft is not None:
                        lab_np = np.asarray(tokens_b[start + 1:start + length + 1])
                        lab_d = torch.from_numpy(tgt2draft[lab_np]).to(device)  # -100: вне словаря
                        keep = (lab_d >= 0)
                        lab_d = torch.where(keep, lab_d, torch.zeros_like(lab_d))
                        loss = F.cross_entropy(logits.float(), lab_d, ignore_index=-100,
                                               reduction="sum") / max(int(keep.sum()), 1)
                    else:
                        lab = torch.from_numpy(
                            np.asarray(tokens_b[start + 1:start + length + 1]).astype(np.int64)).to(device)
                        loss = F.cross_entropy(logits.float(), lab, reduction="mean")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                tot_loss += float(loss.detach()) * length
                tot_rows += length
            if steps % 20 == 0:
                print(f"  epoch {epoch + 1} step {steps}: loss {tot_loss / max(tot_rows, 1):.3f} "
                      f"({tot_rows} строк)", flush=True)
        print(f"epoch {epoch + 1}: средний loss {tot_loss / max(tot_rows, 1):.4f}")
    # сохранение: имена как в чекпоинте (eagle3_convert.py берёт их оттуда)
    from safetensors.torch import save_file
    inv = {v: k for k, v in ST_MAP.items()}
    out_sd = {}
    sd = load_safetensors(args.safetensors)
    for st_name, (mod, par) in ST_MAP.items():
        out_sd[st_name] = getattr(model, mod).__getattr__(par).detach().to(sd[st_name].dtype).cpu()
    for extra in ("d2t", "t2d"):
        for k in sd:
            if k.startswith(extra):
                out_sd[k] = sd[k]
    save_file(out_sd, args.out)
    print(f"сохранено: {args.out} — далее tools/eagle3_convert.py -> GGUF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
