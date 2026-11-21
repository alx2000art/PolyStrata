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
import time

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


def map_labels(tokens: "np.ndarray", tgt2draft):
    """target-токены -> draft-классы; вне d2t-таблицы (хвост словаря таргета) и вне словаря: -100.
    Без t2d/d2t метки - сами target-id (сравнение в CE пространства не путается: dvocab = словарь)."""
    if tgt2draft is None:
        return tokens.astype(np.int64)
    lab = np.full(tokens.shape, -100, dtype=np.int64)
    ok = (tokens >= 0) & (tokens < len(tgt2draft))
    lab[ok] = tgt2draft[tokens[ok]]
    return lab


def scan_blocks(paths):
    """Метаданные блоков без чтения payload: (path, offset, pos0, n, tokens).  Данные ленивые."""
    out, n_embd, taps_n = [], 0, 0
    for path in paths:
        with open(path, "rb") as f:
            magic, ver, n_embd, taps_n, has_emb = struct.unpack("<5I", f.read(20))
            if magic != MAGIC or ver != 2 or has_emb != 1:
                print(f"пропускаю не-дамп: {path} (magic={magic:#x}, ver={ver})", file=sys.stderr)
                continue   # в каталоге смеси бывают и посторонние .bin
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
    # движок: mscale = attn_factor * (1 + 0.1*ln(1/freq_scale)), где attn_factor при yarn =
    # 1/(1 + 0.1*ln(1/freq_scale)) - произведение РОВНО 1.0. Раньше здесь стоял только boost
    # (1.277 при factor 16): q/k офлайн вращались на 28% сильнее движка, скоры внимания были
    # острее в 1.63 раза - голова училась под чужую температуру (73% офлайн vs 35% в движке).
    mscale = 1.0
    return theta, mscale


def rope_apply(x: "torch.Tensor", pos: "torch.Tensor", r) -> "torch.Tensor":
    """x [T, H, D]: HF/transformers rotate_half - пара (i, i + D/2), как у референса (Llama rope).
    Движок снимает это, переставляя строки q_proj/k_proj при загрузке (eagle3.cu w_rope: интерливит
    (i, i+32) в соседние и крутит соседние пары).  Соседние пары на СЫРЫХ строках - другая операция:
    с ней обученная голова в движке видит другое q/k (cos 0.09) и проваливается (73% офлайн vs 32%)."""
    T, H, D = x.shape
    half = D // 2
    iw = torch.arange(half, device=x.device)
    theta, mscale = yarn_phase(pos, iw, r["base"], r["freq_scale"], r["ext_factor"], r["corr0"],
                               r["corr1"], D)
    c, s = torch.cos(theta) * mscale, torch.sin(theta) * mscale        # [T, half]
    a = x[:, :, :half]
    b = x[:, :, half:]
    c = c.unsqueeze(1)
    s = s.unsqueeze(1)
    x0 = a * c - b * s
    x1 = a * s + b * c
    return torch.cat((x0, x1), dim=-1)


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

    def _residual(self, x, pos):
        """Общая часть слоя: x [T, 2E] -> hidden [T, E] (residual до out_norm)."""
        T = x.shape[0]
        q = self.wq(x).view(T, self.n_head, self.hd)
        k = self.wk(x).view(T, self.n_kv, self.hd)
        v = self.wv(x).view(T, self.n_kv, self.hd)
        q = rope_apply(q, pos, self.rope)
        k = rope_apply(k, pos, self.rope)
        # GQA: голова h -> kvh = h // per (смежные группы, проверено check_attn)
        per = self.n_head // self.n_kv
        qg = q.view(T, self.n_kv, per, self.hd)
        att = torch.einsum("tqph,skh->tqps", qg, k) / math.sqrt(self.hd)   # [Tq, n_kv, per, Tk]
        i = torch.arange(T, device=x.device)
        causal = (i.view(-1, 1, 1, 1) >= i.view(1, 1, 1, -1))              # [Tq, 1, 1, Tk]
        w = torch.softmax(att.masked_fill(~causal, float("-inf")), dim=-1)
        o = torch.einsum("tqps,skh->tqph", w, v)                             # [Tq, n_kv, per, hd]
        o = o.reshape(T, self.n_head * self.hd)
        hidden = o                                                           # o уже в n_embd (wo ниже)
        return o

    def _head_out(self, h_in, o):
        hidden = h_in + self.wo(o)
        mlp = self.down(F.silu(self.gate(rmsnorm(hidden, self.ln2.weight, self.eps))) * self.up(rmsnorm(hidden, self.ln2.weight, self.eps)))
        hidden = hidden + mlp
        return hidden

    def forward(self, embs, taps, pos):
        # embs [T, E], taps [T, K, E], pos [T] (абсолютные позиции строк блока)
        T = embs.shape[0]
        h = self.fc(taps.reshape(T, -1))
        x = torch.cat((rmsnorm(embs, self.ln1.weight, self.eps),
                       rmsnorm(h, self.hnorm.weight, self.eps)), dim=-1)          # [T, 2E]
        o = self._residual(x, pos)
        hidden = self._head_out(h, o)
        normed = rmsnorm(hidden, self.out_norm.weight, self.eps)
        return self.lm_head(normed), hidden, x

    def chain_depth(self, K, V, h_prev, embs_cur, pos_cur, pos_stack, half_att=False):
        """Одна глубина цепочки В ДВИЖКОВОМ КОНТЕКСТЕ: строка j смотрит на позиционный стек K/V
        (в движке это кэш: строки инжекта = leg1-строки на своих позициях, затёртые ранними
        строками раунда) + на свою строку.  Референс держит ровно такой кэш сквозь глубины
        (traineagle3/cnets.py: cache_hidden = [[], []] до цикла, обновляется каждую глубину).
        K/V: [T, kv, hd] по позициям pos_stack; возвращает (logits, hidden, k_cur, v_cur)."""
        M = embs_cur.shape[0]
        x = torch.cat((rmsnorm(embs_cur, self.ln1.weight, self.eps),
                       rmsnorm(h_prev, self.hnorm.weight, self.eps)), dim=-1)
        q = rope_apply(self.wq(x).view(M, self.n_head, self.hd), pos_cur, self.rope)
        k = rope_apply(self.wk(x).view(M, self.n_kv, self.hd), pos_cur, self.rope)
        v = self.wv(x).view(M, self.n_kv, self.hd)
        per = self.n_head // self.n_kv
        if half_att:
            # движок приводит q к half и держит K/V в fp16-кэше (attn_group_kernel: __float2half),
            # а считает всё в fp32 - если паритет ловится только так, значит тренер обязан
            # повторять fp16-арифметику внимания, а не f32
            q = q.half().float()
            K = K.half().float()
            V = V.half().float()
        qg = q.view(M, self.n_kv, per, self.hd)
        ka = torch.cat((K, k), dim=0)                                            # [T+M, kv, hd]
        va = torch.cat((V, v), dim=0)
        att = torch.einsum("tqph,skh->tqps", qg, ka) / math.sqrt(self.hd)
        j = torch.arange(M, device=embs_cur.device)
        causal = (pos_stack.view(1, -1) < pos_cur.view(-1, 1))                    # [M, T]
        own = (j.view(-1, 1) == j.view(1, -1))                                    # [M, M]
        mask = torch.cat((causal, own), dim=1)                                    # [M, T+M]
        w = torch.softmax(att.masked_fill(~mask[:, None, None, :], float("-inf")), dim=-1)
        o = torch.einsum("tqps,skh->tqph", w, va).reshape(M, self.n_head * self.hd)
        hidden = self._head_out(h_prev, o)
        normed = rmsnorm(hidden, self.out_norm.weight, self.eps)
        return self.lm_head(normed), hidden, k, v

    def step_hidden(self, h_prev, embs, pos):
        """Шаг рекурсии decode-раунда: вход h - СОБСТВЕННОЕ скрытое состояние головы с прошлого шага
        (не fc(taps)): x = [ln1(emb) | hnorm(h_prev)], тот же слой, каузально по срезу."""
        x = torch.cat((rmsnorm(embs, self.ln1.weight, self.eps),
                       rmsnorm(h_prev, self.hnorm.weight, self.eps)), dim=-1)
        o = self._residual(x, pos)
        hidden = self._head_out(h_prev, o)
        normed = rmsnorm(hidden, self.out_norm.weight, self.eps)
        return self.lm_head(normed)

    def step_hidden_h(self, h_prev, embs, pos):
        """Тот же шаг цепочки, но возвращает и накопленное hidden - вход следующей глубины.
        Референс (traineagle3/cnets.py) гонит глубины 1..7 ровно так: hidden_states = layer_outputs[0]
        (pre-norm накопленное состояние) без detach, градиенты сквозь глубины."""
        x = torch.cat((rmsnorm(embs, self.ln1.weight, self.eps),
                       rmsnorm(h_prev, self.hnorm.weight, self.eps)), dim=-1)
        o = self._residual(x, pos)
        hidden = self._head_out(h_prev, o)
        normed = rmsnorm(hidden, self.out_norm.weight, self.eps)
        return self.lm_head(normed), hidden


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
    print(f"rope: base {rope['base']} freq_scale {rope['freq_scale']:.6g} ext {rope['ext_factor']} corr0 {rope['corr0']} corr1 {rope['corr1']} n_rot {n_rot}")
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
    ap.add_argument("--dump", help="каталог дампа v2 (taps-*.bin); не нужен для --replay-round")
    ap.add_argument("--out", default="eagle3-ft.safetensors")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt8", action="store_true",
                    help="8-битный Adam (bitsandbytes): состояния ~0.8 ГБ вместо ~6.3")
    ap.add_argument("--rec-weight", type=float, default=1.0,
                    help="вес ноги рекурсии (0 = как раньше, только leg1)")
    ap.add_argument("--mask-draft0", action="store_true",
                    help="маскировать метки draft-0 (target-1): decode-дампы кладут в метки "
                         "отклонённые draft-токены - голова выучивает аттрактор копии")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="шумовая аугментация taps: sigma = noise * rms(taps) на каждом шаге "
                         "(голова хрупка к сдвигу read->decode 3-4%%; 0 = выкл, рекомендовано 0.08)")
    ap.add_argument("--chain-depth", type=int, default=0,
                    help="глубокая цепочка как у референса: глубины 1..D без detach, веса 0.8^d "
                         "(0 = нога рекурсии как раньше; D>=1 заменяет её)")
    ap.add_argument("--max-rows", type=int, default=1024, help="строк блока за шаг (хвост отбрасывается)")
    ap.add_argument("--coverage", action="store_true", help="только отчёт по покрытию d2t и выход")
    ap.add_argument("--eval-only", action="store_true",
                    help="без обучения: top-1/рекурсия головы на дампах - как TEACH движка; "
                         "сверка форварда тренера с движком")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--replay-round", default=None,
                    help="дамп раунда mov'а (STRATA_EAGLE_DUMP_ROUND): прогнать голову на ЕГО состоянии")
    ap.add_argument("--replay-ctx", default=None,
                    help="дамп раунда С КЭШЕМ (STRATA_EAGLE_DUMP_ROUND + STRATA_EAGLE_DUMP_CTX_KV=1): "
                         "прогнать голову с ТЕМ ЖЕ вниманием по кэшу, что у движка, и рядом - без него")
    args = ap.parse_args()

    head_cfg = {}
    if args.config:
        head_cfg = json.load(open(args.config, encoding="utf-8"))
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if not args.replay_round and not args.replay_ctx and not args.dump:
        print("нужен --dump (или --replay-round для одного входа)", file=sys.stderr)
        return 2
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    print(f"модель: E {meta['n_embd']}, голов {meta['n_head']}:{meta['n_kv']}, dvocab {meta['dvocab']}, "
          f"device {device}")

    if args.replay_round:
        # Реплей ТОЧНОГО входа раунда: hidden_ + emb + pos, снятые движком (STRATA_EAGLE_DUMP_ROUND).
        # step_hidden(h, emb, pos) = тот же шаг цепочки, что draft_logits: если argmax совпадает с
        # движковым draft_id - head и входы у движка верны, и расхождение вне этого шага; если нет -
        # движок и тренер не согласны ЛИБО на этом входе (голова/квант), ЛИБО состояние - не то.
        import struct as _st
        raw = open(args.replay_round, "rb").read()
        ne, tok = _st.unpack("<ii", raw[:8])
        posr, = _st.unpack("<q", raw[8:16])
        hv = np.frombuffer(raw[16:16 + 4 * ne], dtype="<f4")
        ev = np.frombuffer(raw[16 + 4 * ne:16 + 8 * ne], dtype="<f4")
        print(f"round dump: token {tok} pos {posr}, |h| {np.linalg.norm(hv):.4g}")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lg = model.step_hidden(torch.from_numpy(hv[None]).to(device),
                                   torch.from_numpy(ev[None]).to(device),
                                   torch.tensor([posr], device=device))
        lgf = lg.float().cpu().numpy()[0]
        top = np.argsort(-lgf)[:5]
        print("офлайн (тот же вход):", " ".join(f"{int(i)}/{lgf[i]:.2f}" for i in top))
        print("(сравни с движковым 'draft 1/3: ... draft_id=...')")
        return 0

    if args.replay_ctx:
        # Реплей раунда С КОНТЕКСТОМ: движок снял не только (hidden, emb, pos), но и свой KV-кэш
        # [0, pos).  chain_depth с этим стеком считает ровно то, что считает midlayer в движке
        # (маска pos_stack < pos + своя строка), а рядом печатается ответ БЕЗ контекста
        # (step_hidden) - разница между ними и есть та функция, которой не хватало обучению.
        import struct as _st
        raw = open(args.replay_ctx, "rb").read()
        ne, tok = _st.unpack("<ii", raw[:8])
        posr, = _st.unpack("<q", raw[8:16])
        off = 16
        hv = np.frombuffer(raw[off:off + 4 * ne], dtype="<f4").copy(); off += 4 * ne
        ev = np.frombuffer(raw[off:off + 4 * ne], dtype="<f4").copy(); off += 4 * ne
        if off + 8 > len(raw):
            print("в дампе нет кэша: сними его с STRATA_EAGLE_DUMP_CTX_KV=1", file=sys.stderr)
            return 2
        rows, = _st.unpack("<q", raw[off:off + 8]); off += 8
        nkv, hd = int(meta["n_kv"]), int(model.hd)
        part = rows * nkv * hd
        K = np.frombuffer(raw[off:off + 2 * part], dtype="<f2").reshape(rows, nkv, hd).astype(np.float32)
        V = np.frombuffer(raw[off + 2 * part:off + 4 * part], dtype="<f2").reshape(rows, nkv, hd).astype(np.float32)
        # Кэш хранит k в порядке ДВИЖКА: w_rope переставляет строки wq/wk так, что dsv4_rope
        # поворачивает соседние пары (движок: индекс 2i <- i, 2i+1 <- i+32).  Тренер считает
        # в HF-порядке (половины), поэтому K надо развернуть обратно, иначе q·k считается между
        # разноупорядоченными векторами и даёт мусор (cos 0.23 при идеальных q/k/v).
        _h = hd // 2
        _idx = np.empty(hd, dtype=np.int64)
        _idx[:_h] = 2 * np.arange(_h)
        _idx[_h:] = 2 * np.arange(_h) + 1
        K = K[..., _idx]
        # v не поворачивается и не переставляется - он уже в HF-порядке
        Kf, Vf = K, V
        print(f"round dump: token {tok} pos {posr}, |h| {np.linalg.norm(hv):.4g}, "
              f"кэш {rows} строк ({(len(raw) - 16 - 8 * ne - 8) / 1e6:.2f} МБ)")
        # логиты того же шага (STRATA_EAGLE_DUMP_CTX_LOGITS): без них гейт сводится к argmax, а он
        # при p~0.01 шум - сравнивать надо распределения
        eng = None
        _lgpath = args.replay_ctx + ".logits"
        if os.path.exists(_lgpath):
            _raw = open(_lgpath, "rb").read()
            _n, = _st.unpack("<i", _raw[:4])
            eng = np.frombuffer(_raw[4:4 + 4 * _n], dtype="<f4")
            print(f"логиты движка: {_n}, его top-1 {int(np.argmax(eng))}")
        else:
            print(f"(нет {_lgpath}: сними шаг с STRATA_EAGLE_DUMP_CTX_LOGITS=<файл>)")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            lg_noctx = model.step_hidden(torch.from_numpy(hv[None]).to(device),
                                         torch.from_numpy(ev[None]).to(device),
                                         torch.tensor([posr], device=device))
            lg_ctx, _, _, _ = model.chain_depth(
                torch.from_numpy(Kf).to(device),
                torch.from_numpy(Vf).to(device),
                torch.from_numpy(hv[None]).to(device),
                torch.from_numpy(ev[None]).to(device),
                torch.tensor([posr], device=device),
                torch.arange(rows, device=device))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                lg_ctx_half, _, _, _ = model.chain_depth(
                    torch.from_numpy(Kf).to(device),
                    torch.from_numpy(Vf).to(device),
                    torch.from_numpy(hv[None]).to(device),
                    torch.from_numpy(ev[None]).to(device),
                    torch.tensor([posr], device=device),
                    torch.arange(rows, device=device), half_att=True)
        for name, lg in (("без кэша (step_hidden)", lg_noctx),
                         ("с кэшем движка (chain_depth)", lg_ctx),
                         ("с кэшем, внимание fp16 (как в движке)", lg_ctx_half)):
            lgf = lg.float().cpu().numpy()[0]
            top = np.argsort(-lgf)[:5]
            print(f"{name}:", " ".join(f"{int(i)}/{lgf[i]:.2f}" for i in top))
            if eng is not None and len(eng) == len(lgf):
                a = lgf.astype(np.float64)
                b = eng.astype(np.float64)
                cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                mxd = float(np.max(np.abs(a - b)))
                eid = int(np.argmax(eng))
                rank = int((a > a[eid]).sum())     # сколько классов тренера выше id движка
                print(f"   vs движок: cos={cos:+.5f} maxdiff={mxd:.2f} id_движка={eid} у_тренера_ранг={rank}")
        print("(сравни 'с кэшем' с движковым 'draft 1/3: ... draft_id=...': совпало -> тренер "
              "воспроизводит раунд, и обучению не хватает именно контекста)")
        return 0

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
        # *.bin, а не только "taps-*.bin": оверсэмплинг-ссылки называются <prefix>-taps-...,
        # и прежний glob их НЕ видел - ft7 молча обучался без диалоговых данных
        files = sorted(glob.glob(os.path.join(args.dump, "*.bin")))
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

    if args.eval_only:
        # та же метрика, что TEACH движка: (emb p, h p) -> p+1 и (emb p+1, h p) -> p+2.
        # top-1 считается по позициям, чья метка есть в draft-словаре (иначе позиция не измерима).
        # Нога 1 меряется на n_rows-1 строках, ноги 2/3 - на n_rows-2 (их метки на строку дальше):
        # decode-дамп приходит ПАРНЫМИ блоками по 2 строки - нога 1 там единственная измеримая.
        tot1 = hit1 = tot2 = hit2 = tot3 = hit3 = tot4 = hit4 = 0
        # decode-дамп приходит ПАРНЫМИ 2-строчными блоками ОДНОЙ последовательности: склеиваем
        # соседние блоки по непрерывности pos0 в длинные срезы - иначе внимание и цепочка видят
        # 2 ключа вместо живого контекста (нога 4 на 2 строках вообще не измерима)
        # блоки ПЕРЕКРЫВАЮТСЯ: окно [p, p+1], следующее [p+1, p+2] (decode-шаг 1 токен) -
        # допускаем перекрытие и при склейке пропускаем уже покрытые строки
        groups, cur = [], []
        for b in blocks:
            # та же логика, что в TRAIN: внутри файла + малое перекрытие 1-2 строки
            # (read-чанки тоже начинаются с pos 0 - широкое перекрытие слило бы их)
            same_file = bool(cur) and cur[-1][0] == b[0]
            small = cur and (cur[-1][2] + cur[-1][3] - 2 <= b[2] <= cur[-1][2] + cur[-1][3] + 1)
            if cur and not (same_file and small):
                groups.append(cur)
                cur = []
            cur.append(b)
        if cur:
            groups.append(cur)
        if len(groups) != len(blocks):
            print(f"eval-only: блоков {len(blocks)} -> склеенных срезов {len(groups)}")
        with torch.no_grad():
            for grp in groups:
                pos0_b = grp[0][2]
                expect = pos0_b
                toks, embs, taps = [], [], []
                for (p, o, p0, n, _t) in grp:
                    part = load_block(p, o, n, meta["n_embd"], meta["taps_n"])
                    skip = max(0, expect - p0)          # строки, уже покрытые предыдущими блоками
                    if skip >= n:
                        continue
                    toks.append(part[1][skip:])
                    embs.append(part[2][skip:])
                    taps.append(part[3][skip:])
                    expect = p0 + n
                tokens_b = np.concatenate(toks)
                embs_b = np.concatenate(embs)
                taps_b = np.concatenate(taps)
                n_rows = int(len(tokens_b))
                for start in range(0, max(n_rows - 1, 0), args.max_rows):
                    length = min(args.max_rows, n_rows - 1 - start)
                    if length < 1:
                        break
                    pos = torch.arange(pos0_b + start, pos0_b + start + length, device=device)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        # нога 1: (emb p, h p) -> p+1 - строка p несёт и embs[p], и taps[p]
                        logits, hid, x_att = model(torch.from_numpy(embs_b[start:start + length]).to(device),
                                            torch.from_numpy(taps_b[start:start + length]).to(device), pos)
                    lab1 = map_labels(np.asarray(tokens_b[start + 1:start + length + 1]), tgt2draft)
                    # STRATA_EVAL_ROWS=from:to - считать только строки в диапазоне (ответ в turn-дампе)
                    _er = os.environ.get("STRATA_EVAL_ROWS", "")
                    if _er:
                        _a, _, _b = _er.partition(":")
                        _lo, _hi = int(_a or 0), int(_b or (1 << 62))
                        _rp = pos0_b + start + np.arange(length)
                        lab1 = np.where((_rp >= _lo) & (_rp < _hi), lab1, -100)
                    pred = logits.argmax(-1).cpu().numpy()
                    if args.mask_draft0:
                        # метрика без "draft 0": иначе попадания по отравленным декодным рядам
                        # (label 0) завышают число и голова, всегда отвечающая 0, выглядит хорошей
                        lab1 = np.where(lab1 == 0, -100, lab1)
                    # DBG: состояние строк вокруг первого decode-раунда (STRATA_EVAL_POS, default 1088):
                    # сравнить с движковым "draft step: ... hidden ..." - подача hidden_ в раунд
                    dbg_pos = int(os.environ.get("STRATA_EVAL_POS", "1088"))
                    i0 = dbg_pos - pos0_b - start
                    if 0 <= i0 < length:
                        for jj in range(i0, min(length, i0 + 3)):
                            hv = [round(float(v), 4) for v in hid[jj][:3]]
                            lab = int(lab1[jj])
                            print(f"[dbg] pos {int(pos0_b + start + jj)}: hid {hv}, "
                                  f"pred draft {int(pred[jj])}, label draft {lab}")
                    ok = (lab1 >= 0) & (pred == lab1)
                    tot1 += int((lab1 >= 0).sum())
                    hit1 += int(ok.sum())
                    # нога 2: (emb p+1, h p) -> p+2 - эмбеддинг следующей строки при taps текущей
                    # (нужны строки p+1 и метка p+2: в парном decode-блоке из 2 строк её нет - пропуск)
                    if start + length + 1 < n_rows:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                            logits2, _, _ = model(torch.from_numpy(embs_b[start + 1:start + length + 1]).to(device),
                                            torch.from_numpy(taps_b[start:start + length]).to(device), pos)
                        lab2 = map_labels(np.asarray(tokens_b[start + 2:start + length + 2]), tgt2draft)
                        if _er:   # та же маска диапазона, что у leg1: строка пары = p+1
                            lab2 = np.where((pos0_b + start + np.arange(length) + 1 >= _lo) &
                                            (pos0_b + start + np.arange(length) + 1 < _hi), lab2, -100)
                        pred2 = logits2.argmax(-1).cpu().numpy()
                        if args.mask_draft0:
                            lab2 = np.where(lab2 == 0, -100, lab2)
                        ok2 = (lab2 >= 0) & (pred2 == lab2)
                        tot2 += int((lab2 >= 0).sum())
                        hit2 += int(ok2.sum())
                    # нога 3: математика ПЕРВОГО decode-шага движка: h = собственный residual головы
                    # на строке p (inject уже прогнал слой), позиция p+1, метка та же p+1.
                    # Если тут ~0% при 30%+ у ноги 1 - движок спрашивает голову не тем входом.
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                        logits3 = model.step_hidden(hid, torch.from_numpy(embs_b[start:start + length]).to(device),
                                                    pos + 1)
                    pred3 = logits3.argmax(-1).cpu().numpy()
                    ok3 = (lab1 >= 0) & (pred3 == lab1)
                    tot3 += int((lab1 >= 0).sum())
                    hit3 += int(ok3.sum())
                    # нога 4 - ПАРА ЦЕПОЧКИ, на которой едут decode-раунды: (emb x_{j+1},
                    # накопленное hidden головы с строки j) на позиции j+1 -> x_{j+2}.
                    # Внимание - ДВИЖКОВОГО контекста: позиционный стек leg1-ключей (как кэш
                    # инжекта) + своя строка - ровно то, что видит раунд в движке.
                    rows4 = length - 2
                    if rows4 >= 1:
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                            Kst4 = rope_apply(model.wk(x_att).view(length, model.n_kv, model.hd), pos, model.rope)
                            Vst4 = model.wv(x_att).view(length, model.n_kv, model.hd)
                            logits4, _, _, _ = model.chain_depth(
                                Kst4, Vst4, hid[:rows4],
                                torch.from_numpy(embs_b[start + 1:start + 1 + rows4]).to(device),
                                pos[1:1 + rows4], pos)
                        lab4 = map_labels(np.asarray(tokens_b[start + 2:start + 2 + rows4]), tgt2draft)
                        if _er:   # строка пары глубины 1 = j+1
                            lab4 = np.where((pos0_b + start + np.arange(rows4) + 1 >= _lo) &
                                            (pos0_b + start + np.arange(rows4) + 1 < _hi), lab4, -100)
                        pred4 = logits4.argmax(-1).cpu().numpy()
                        if args.mask_draft0:
                            lab4 = np.where(lab4 == 0, -100, lab4)
                        ok4 = (lab4 >= 0) & (pred4 == lab4)
                        tot4 += int((lab4 >= 0).sum())
                        hit4 += int(ok4.sum())
        print(f"eval-only (top-1 среди покрытых d2t): "
              f"(emb p, h p) -> p+1: {hit1} из {tot1} ({100 * hit1 / max(tot1, 1):.1f}%), "
              f"(emb p+1, h p) -> p+2: {hit2} из {tot2} ({100 * hit2 / max(tot2, 1):.1f}%), "
              f"decode step (h=residual p, pos p+1): {hit3} из {tot3} ({100 * hit3 / max(tot3, 1):.1f}%), "
              f"chain (emb p+1, h=p) -> p+2: {hit4} из {tot4} ({100 * hit4 / max(tot4, 1):.1f}%)")
        return 0

    if args.opt8:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
        print("оптимизатор: AdamW8bit")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    rng = random.Random(42)
    steps = 0
    # группы последовательностей - ЗАРАНЕЕ, в файловом порядке (в перемешанном нельзя:
    # последовательность разрезалась бы и переобрабатывалась)
    groups, cur_g = [], []
    for bi in range(len(blocks)):
        b2 = blocks[bi]
        # ТОЛЬКО внутри одного файла: read- и decode-дампы в смеси нумеруют позиции
        # абсолютно (0..1470 и 1088..1470) - склейка через границу файла смешала бы
        # чужие строки с одинаковыми позициями
        same_file = bool(cur_g) and blocks[cur_g[-1]][0] == b2[0]
        # склеиваем ТОЛЬКО decode-пары со смещением 1-2 строки.  Read-блоки одного файла
        # тоже начинаются с pos 0 (каждый чанк - своё чтение с нуля): широкое перекрытие
        # слило бы их, а дедуп выбросил бы почти весь корпус.
        small_overlap = cur_g and (blocks[cur_g[-1]][2] + blocks[cur_g[-1]][3] - 2 <= b2[2]
                                   <= blocks[cur_g[-1]][2] + blocks[cur_g[-1]][3] + 1)
        if cur_g and not (same_file and small_overlap):
            groups.append(cur_g)
            cur_g = []
        cur_g.append(bi)
    if cur_g:
        groups.append(cur_g)
    print(f"последовательностей: {len(groups)} (из {len(blocks)} блоков)")
    # штампы времени: без них потом нельзя сказать, сколько шли эпохи и когда стартовали
    t_run = time.time()
    print(f"старт {time.strftime('%Y-%m-%d %H:%M:%S')}: эпох {args.epochs}, lr {args.lr}, "
          f"chain-depth {args.chain_depth}, noise {args.noise}, mask-draft0 {bool(args.mask_draft0)}",
          flush=True)
    for epoch in range(args.epochs):
        order = list(range(len(groups)))
        rng.shuffle(order)
        tot_loss, tot_rows = 0.0, 0
        win_loss, win_rows = 0.0, 0
        t_ep = time.time()
        print(f"эпоха {epoch + 1}/{args.epochs} начата {time.strftime('%H:%M:%S')}", flush=True)
      # окно с последней печати: мгновенный лосс
        for gi in order:
            # ГРУППА = одна последовательность: decode-блоки идут парами с перекрытием в строку,
            # корпусные - целыми блоками.  Перекрытия схлопываются по абсолютной позиции (иначе
            # дублированные строки попадают в обучение с чужими метками).
            members = groups[gi]
            toks, embs_l, taps_l = [], [], []
            expect = blocks[members[0]][2]
            pos0_b = expect
            for mi in members:
                p2, o2, p02, n2, _t2 = blocks[mi]
                part = load_block(p2, o2, n2, meta["n_embd"], meta["taps_n"])
                skip = max(0, expect - p02)
                if skip >= n2:
                    continue
                toks.append(part[1][skip:])
                embs_l.append(part[2][skip:])
                taps_l.append(part[3][skip:])
                expect = p02 + n2
            if not toks:
                continue
            tokens_b = np.concatenate(toks)
            embs_b = np.concatenate(embs_l)
            taps_b = np.concatenate(taps_l)
            n_rows = int(len(tokens_b))
            if n_rows < 8:
                continue
            for start in range(0, n_rows - 1, args.max_rows):
                length = min(args.max_rows, n_rows - 1 - start)
                if length < 8:
                    break
                x_embs = torch.from_numpy(embs_b[start:start + length]).to(device)
                x_taps = torch.from_numpy(taps_b[start:start + length]).to(device)
                # шумовая аугментация: те же taps, сдвинутые как read->decode (cos 0.90-0.99).
                # Голова учится не заучивать конкретные состояния, а быть устойчивой к гулу.
                if args.noise > 0:
                    sig = float(x_taps.pow(2).mean().sqrt()) * args.noise
                    x_taps = x_taps + torch.randn_like(x_taps) * sig
                pos = torch.arange(pos0_b + start, pos0_b + start + length, device=device)
                # bf16-автокаст: веса в fp32, матмулы в bf16 - активации и логиты вдвое меньше
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    logits, hid, x_att = model(x_embs, x_taps, pos)  # [n, dvocab], [n, E], [n, 2E]
                    if tgt2draft is not None:
                        lab_d = torch.from_numpy(map_labels(np.asarray(tokens_b[start + 1:start + length + 1]),
                                                            tgt2draft)).to(device)  # -100: вне словаря
                        if args.mask_draft0:
                            lab_d = torch.where(lab_d == 0, torch.full_like(lab_d, -100), lab_d)
                        # -100 (вне draft-словаря и строки, снятые --mask-draft0) остаётся -100.
                        # Подстановка нулей ДО взятия маски превращала каждый такой ряд в цель
                        # "предскажи draft 0": четверть словаря таргета - хвост вне d2t, и голова
                        # уезжала в константу (logit 9.5 на id 0 в живом декоде).
                        # чанковый CE: fp32-временники логитов держим по куску строк, не целиком
                        csum, ckeep = 0.0, 0
                        for cs in range(0, length, 256):
                            ce = cs + 256
                            part = logits[cs:ce].float()
                            kp = (lab_d[cs:ce] >= 0)
                            csum = csum + F.cross_entropy(part, lab_d[cs:ce],
                                                          ignore_index=-100, reduction="sum")
                            ckeep += int(kp.sum())
                        loss = csum / max(ckeep, 1)
                    else:
                        lab = torch.from_numpy(
                            np.asarray(tokens_b[start + 1:start + length + 1]).astype(np.int64)).to(device)
                        loss = F.cross_entropy(logits.float(), lab, reduction="mean")
                    # нога рекурсии (так учит штатный EAGLE-3): decode-раунды после первого шага
                    # кормят голову её СОБСТВЕННЫМ скрытым состоянием с прошлого шага.  Без этой
                    # ноги файн-тюн перекашивает голову: leg1 растёт, рекурсия деградирует в
                    # вырожденный токен (TEACH: leg1 23%, recursion 1.4% у первого ft).
                    # При --chain-depth >= 1 её заменяет глубокая цепочка ниже (тот же шаг d=1,
                    # но без detach и с градиентами сквозь глубины).
                    if args.rec_weight > 0 and args.chain_depth == 0 and tgt2draft is not None:
                        hid_in = torch.cat((torch.zeros_like(hid[:1]), hid[:-1].detach()), dim=0)
                        logits2 = model.step_hidden(hid_in, x_embs, pos)
                        csum2, ckeep2 = 0.0, 0
                        for cs in range(1, length, 256):     # строка 0 без предшественника - мимо
                            ce = min(cs + 256, length)
                            part = logits2[cs:ce].float()
                            kp = (lab_d[cs:ce] >= 0)
                            csum2 = csum2 + F.cross_entropy(part, lab_d[cs:ce],
                                                            ignore_index=-100, reduction="sum")
                            ckeep2 += int(kp.sum())
                        loss = loss + args.rec_weight * (csum2 / max(ckeep2, 1))
                    # глубокая цепочка - рецепт референса (traineagle3/cnets.py + main.py):
                    # глубина d: строки j = (emb x_{j+d}, накопленное hidden глубины d-1 с ТОГО ЖЕ j)
                    # на позиции j+d, метка x_{j+d+1}; состояние подаётся БЕЗ detach - градиент течёт
                    # сквозь глубины, и голова учится не только потреблять, но и ПРОИЗВОДИТЬ
                    # состояния, пригодные для собственной цепочки (одношаговая detached-нога
                    # оставляла цепочку на 6.5-8% при leg1 41%).  Веса 0.8^d как в референсе.
                    # Контекст внимания - ДВИЖКОВЫЙ: позиционный стек K/V из leg1-строк (инжект),
                    # который ранние глубины перезаписывают покрытыми строками (референс держит
                    # cache_hidden сквозь глубины) - иначе строка цепочки в обучении видит не тот
                    # контекст, что в decode-раунде, и живьём проваливается в вырожденный токен.
                    if args.chain_depth > 0 and tgt2draft is not None:
                        Kst = rope_apply(model.wk(x_att).view(length, model.n_kv, model.hd), pos, model.rope)
                        Vst = model.wv(x_att).view(length, model.n_kv, model.hd)
                        h = hid
                        for d in range(1, args.chain_depth + 1):
                            rows = length - d - 1
                            if rows < 1:
                                break
                            logits_d, h_new, k_c, v_c = model.chain_depth(
                                Kst, Vst, h[:rows], x_embs[d:d + rows], pos[d:d + rows], pos)
                            # строки глубины d занимают позиции d..d+rows-1 - пишем их в стек
                            Kst = torch.cat((Kst[:d], k_c, Kst[d + rows:]), dim=0)
                            Vst = torch.cat((Vst[:d], v_c, Vst[d + rows:]), dim=0)
                            h = h_new
                            lab_d2 = torch.from_numpy(
                                map_labels(np.asarray(tokens_b[start + d + 1:start + d + 1 + rows]),
                                           tgt2draft)).to(device)
                            # та же маска, что у глубины 0: без неё цель "draft 0" учится на
                            # каждой строке вне словаря и на отравленных декодных рядах
                            if args.mask_draft0:
                                lab_d2 = torch.where(lab_d2 == 0, torch.full_like(lab_d2, -100), lab_d2)
                            csum_d, ckeep_d = 0.0, 0
                            for cs in range(0, rows, 256):
                                ce = min(cs + 256, rows)
                                part = logits_d[cs:ce].float()
                                kp = (lab_d2[cs:ce] >= 0)
                                csum_d = csum_d + F.cross_entropy(part, lab_d2[cs:ce],
                                                                  ignore_index=-100, reduction="sum")
                                ckeep_d += int(kp.sum())
                            loss = loss + (0.8 ** d) * (csum_d / max(ckeep_d, 1))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                steps += 1
                tot_loss += float(loss.detach()) * length
                tot_rows += length
                win_loss += float(loss.detach()) * length
                win_rows += length
            if steps % 20 == 0:
                # кумулятивное среднее по эпохе + мгновенное окно: смесь неоднородна (проза, код,
                # диалоги, декодные ряды), и по одному кумулятиву не видно, учится голова или нет
                _el = time.time() - t_ep
                print(f"  epoch {epoch + 1} step {steps}: loss {tot_loss / max(tot_rows, 1):.3f} "
                      f"(окно {win_loss / max(win_rows, 1):.3f}, {tot_rows} строк, "
                      f"{_el / 60:.1f} мин, {tot_rows / max(_el, 1e-9):.0f} строк/с)", flush=True)
                win_loss, win_rows = 0.0, 0
        _dt = time.time() - t_ep
        print(f"epoch {epoch + 1}: средний loss {tot_loss / max(tot_rows, 1):.4f} за "
              f"{_dt / 60:.1f} мин ({tot_rows / max(_dt, 1e-9):.0f} строк/с, до "
              f"{time.strftime('%H:%M:%S')})", flush=True)
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
