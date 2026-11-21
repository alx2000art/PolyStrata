#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle_stage_check.py - сверка стадий шага драфта: движок против тренера.

    python3 tools/eagle_stage_check.py <stage.bin> --safetensors ... --config ... \
        [--round ~/eagle_round.bin] [--cpu]

Движок пишет (STRATA_EAGLE_DUMP_STAGE=<файл> STRATA_EAGLE_DUMP_STAGE_POS=<позиция>):
    hdr: i32 n_embd, i32 q_cols (n_head*hd), i32 kv_cols (n_kv*hd); i64 pos
    6 частей: e_norm, f_norm, q, k, v, o  (u64 длина + f32 значения)
q/k/v - ПОСЛЕ rope; o - выход внимания до o_proj.

Скрипт считает ту же цепочку своей математикой (на том же hidden/emb из round-дампа)
и печатает по стадии cos и max|diff|.  Так видно, ЧТО расходится:
  * e/f уже разные  -> нормы/eps/входы (hidden, emb)
  * q/k/v разные    -> веса проекций или rope
  * q/k/v совпали, o нет -> само внимание (K/V-строки, маска, порядок голов)
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("tools", 1)[0])
from eagle3_ft_train import load_model, rmsnorm, rope_apply  # noqa: E402


def read_round(path):
    raw = open(path, "rb").read()
    ne, tok = struct.unpack("<ii", raw[:8])
    pos, = struct.unpack("<q", raw[8:16])
    hv = np.frombuffer(raw[16:16 + 4 * ne], dtype="<f4").copy()
    ev = np.frombuffer(raw[16 + 4 * ne:16 + 8 * ne], dtype="<f4").copy()
    return ne, tok, pos, hv, ev


def read_stage(path):
    raw = open(path, "rb").read()
    ne, qc, kvc = struct.unpack("<iii", raw[:12])
    pos, = struct.unpack("<q", raw[12:20])
    off, parts = 20, []
    for _ in range(8):
        n, = struct.unpack("<Q", raw[off:off + 8]); off += 8
        parts.append(np.frombuffer(raw[off:off + 4 * n], dtype="<f4").copy()); off += 4 * n
    # hidden и emb идут первыми: файл самодостаточен (round-дамп не нужен)
    return dict(ne=ne, qc=qc, kvc=kvc, pos=pos, hidden=parts[0], emb=parts[1],
                e=parts[2], f=parts[3], q=parts[4], k=parts[5], v=parts[6], o=parts[7])


def cmp(name, a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        print(f"  {name:6s} ФОРМА {a.shape} против {b.shape} — не сравнить")
        return
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    mxd = float(np.max(np.abs(a - b)))
    # норма та же, но вектор другой = перестановка компонент: rope поворачивает пары, а
    # поворот ортогонален (норму бережёт).  Мультимножество отличает перестановку от
    # настоящего расхождения - и тогда q·k (а значит и внимание) не страдают вовсе.
    perm = np.allclose(np.sort(a), np.sort(b), atol=1e-3) and cos < 0.99
    tail = "  <- ПЕРЕСТАНОВКА компонент (мультимножество совпало)" if perm else ""
    print(f"  {name:6s} cos={cos:+.6f}  maxdiff={mxd:.4g}  |движок|={np.linalg.norm(a):.4g}  "
          f"|тренер|={np.linalg.norm(b):.4g}{tail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--round", default=None, help="round-дамп (hidden/emb); по умолчанию stage-файл без .stage")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--kv-layout", choices=["row", "kv"], default="row",
                    help="раскладка кэша: row=[row][kv][hd] (attend_batch так и читает), kv=[kv][row][hd]")
    ap.add_argument("--kv-order", choices=["hf", "raw"], default="hf",
                    help="порядок K в кэше: hf - развернуть w_rope (по умолчанию), raw - как лежит")
    ap.add_argument("--scores", default=None, help="дамп скоров (по умолчанию <kv>.scores)")
    ap.add_argument("--weights", default=None, help="дамп весов внимания (по умолчанию <kv>.weights)")
    ap.add_argument("--kv", default=None, help="round-дамп с кэшем (по умолчанию <stage без .stage>)")
    args = ap.parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    head_cfg = {}
    if args.config:
        import json
        head_cfg = json.load(open(args.config, encoding="utf-8"))
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    st = read_stage(args.stage)
    # hidden/emb уже в файле стадий; --round нужен только старым файлам без них
    hv, ev = st["hidden"], st["emb"]
    pos = st["pos"]
    tok = -1
    if args.round:
        _ne, tok, _pos, hv_r, ev_r = read_round(args.round)
        if len(hv_r) == len(hv):
            hv, ev = hv_r, ev_r
    print(f"стадии: pos {st['pos']} (round {pos}), token {tok}, E {st['ne']}, "
          f"q {st['qc']}, kv {st['kvc']}")
    with torch.no_grad():
        h = torch.from_numpy(hv[None]).to(device)
        e = torch.from_numpy(ev[None]).to(device)
        e_n = rmsnorm(e, model.ln1.weight, model.eps)
        f_n = rmsnorm(h, model.hnorm.weight, model.eps)
        x = torch.cat((e_n, f_n), dim=-1)
        q = rope_apply(model.wq(x).view(1, model.n_head, model.hd), torch.tensor([pos], device=device), model.rope)
        k = rope_apply(model.wk(x).view(1, model.n_kv, model.hd), torch.tensor([pos], device=device), model.rope)
        v = model.wv(x).view(1, model.n_kv, model.hd)
    # движок держит q/k в порядке w_rope (2i <- i, 2i+1 <- i+32); тренер - в HF.  Разворачиваем
    # стадии движка в HF, и тогда cos должен стать 1.0 - иначе остаётся настоящее расхождение
    _h = model.hd // 2
    _idx = np.empty(model.hd, dtype=np.int64)
    _idx[:_h] = 2 * np.arange(_h)
    _idx[_h:] = 2 * np.arange(_h) + 1
    qe = st["q"].reshape(model.n_head, model.hd)[:, _idx].reshape(-1)
    ke = st["k"].reshape(model.n_kv, model.hd)[:, _idx].reshape(-1)
    print("сверка (движок против тренера), q/k движка приведены в HF-порядок:")
    cmp("e", st["e"], e_n[0].cpu().numpy())
    cmp("f", st["f"], f_n[0].cpu().numpy())
    cmp("q", qe, q[0].reshape(-1).cpu().numpy())
    cmp("k", ke, k[0].reshape(-1).cpu().numpy())
    cmp("v", st["v"], v[0].reshape(-1).cpu().numpy())
    # --- o и логиты: нужен кэш движка (STRATA_EAGLE_DUMP_CTX_KV=1 в том же прогоне) ---
    kvdump = args.kv or args.stage.replace(".stage", "")
    if os.path.exists(kvdump):
        raw = open(kvdump, "rb").read()
        ne2, tok2 = struct.unpack("<ii", raw[:8])
        pos2, = struct.unpack("<q", raw[8:16])
        off2 = 16 + 8 * ne2
        if off2 + 8 <= len(raw):
            rows, = struct.unpack("<q", raw[off2:off2 + 8]); off2 += 8
            part = rows * model.n_kv * model.hd
            if off2 + 4 * part <= len(raw) and ne2 == len(st["e"]):
                K = np.frombuffer(raw[off2:off2 + 2 * part], dtype="<f2").reshape(rows, model.n_kv, model.hd).astype(np.float32)
                V = np.frombuffer(raw[off2 + 2 * part:off2 + 4 * part], dtype="<f2").reshape(rows, model.n_kv, model.hd).astype(np.float32)
                # кэш в порядке движка - в HF, как и q/k стадий
                if args.kv_layout == "kv":      # [kv][row][hd] вместо [row][kv][hd]
                    K = K.reshape(model.n_kv, rows, model.hd).transpose(1, 0, 2)
                    V = V.reshape(model.n_kv, rows, model.hd).transpose(1, 0, 2)
                if args.kv_order == "hf":
                    K = K[..., _idx]      # кэш в порядке w_rope -> HF
                # o ДО _head_out: та же арифметика внимания, но результат наружу - иначе не видно,
                # расходятся ли сами K/V-строки (q/k/v уже совпали до 1e-4)
                with torch.no_grad():
                    qq = q[0].view(model.n_head, model.hd).float()
                    kk = k[0].view(model.n_kv, model.hd).float()
                    vv = v[0].view(model.n_kv, model.hd).float()
                    per = model.n_head // model.n_kv
                    Ka = torch.cat((torch.from_numpy(K).to(device), kk[None]), dim=0)
                    Va = torch.cat((torch.from_numpy(V).to(device), vv[None]), dim=0)
                    att = torch.einsum("qph,skh->qps", qq.view(model.n_kv, per, model.hd), Ka) / (model.hd ** 0.5)
                    w = torch.softmax(att, dim=-1)                     # [n_kv, per, keys]
                    oo = torch.einsum("qps,skh->qph", w, Va).reshape(model.n_head * model.hd)
                cmp("o", st["o"], oo.cpu().numpy())
                # веса внимания: у движка они лежат half-ом, строка на (группа, голова группы)
                wp = args.weights or kvdump + ".weights"
                if os.path.exists(wp):
                    rb = open(wp, "rb").read()
                    nh, nkv_w, nk = struct.unpack("<iii", rb[:12])
                    p_w, = struct.unpack("<q", rb[12:20])
                    if p_w != pos:
                        print(f"  w_att  ПРОПУСК: веса сняты с pos {p_w}, стадии с {pos} - "
                              f"нужен STRATA_EAGLE_DUMP_WEIGHTS_POS={pos}")
                        raise SystemExit(0)
                    eng_w = np.frombuffer(rb[20:20 + 2 * nh * nk], dtype="<f2").reshape(nh, nk).astype(np.float64)
                    tr_w = w.reshape(nh, nk).double().cpu().numpy()
                    rs = eng_w.sum(1, keepdims=True)
                    eng_n = eng_w / np.where(rs == 0, 1.0, rs)     # в скретче сырые exp: sum делится позже
                    cmp("w_att", eng_n.reshape(-1), tr_w.reshape(-1))
                    print(f"         суммы: движок {eng_w.sum(1).mean():.4f}±{eng_w.sum(1).std():.4f}, "
                          f"тренер {tr_w.sum(1).mean():.4f}±{tr_w.sum(1).std():.4f}")
                    dh = int(np.argmax(np.abs(eng_n - tr_w).sum(1)))
                    e3 = np.argsort(-eng_n[dh])[:3]
                    t3 = np.argsort(-tr_w[dh])[:3]
                    print(f"         голова {dh}: движок top3 " +
                          " ".join(f"{int(j)}/{eng_n[dh, j]:.4f}" for j in e3) +
                          " | тренер top3 " + " ".join(f"{int(j)}/{tr_w[dh, j]:.4f}" for j in t3))
                else:
                    print(f"  (нет {wp}: снимай с STRATA_EAGLE_DUMP_WEIGHTS)")
                # сырые скоры: до softmax и до half, поэтому сравнение прямое - самая чистая точка
                sp = args.scores or kvdump + ".scores"
                if os.path.exists(sp):
                    rb2 = open(sp, "rb").read()
                    nh2, nkv2, nk2 = struct.unpack("<iii", rb2[:12])
                    p2, = struct.unpack("<q", rb2[12:20])
                    if p2 != pos or nk2 != int(Ka.shape[0]):
                        print(f"  scores ПРОПУСК: файл от pos {p2} ({nk2} ключей), стадии от {pos} "
                              f"({Ka.shape[0]} ключей) - нужен STRATA_EAGLE_DUMP_SCORES_POS={pos}")
                        raise SystemExit(0)
                    e_sc = np.frombuffer(rb2[20:20 + 4 * nh2 * nk2], dtype="<f4").reshape(nh2, nk2)
                    with torch.no_grad():
                        t_sc = (torch.einsum("qph,skh->qps", qq.view(model.n_kv, per, model.hd),
                                             Ka) / (model.hd ** 0.5)).reshape(nh2, nk2).double().cpu().numpy()
                    cmp("scores", e_sc.reshape(-1), t_sc.reshape(-1))
                    # движок приводит q к half перед GEMM (attn_group_kernel: __float2half), K и так
                    # half.  Если с q, округлённым до half, cos подскакивает - это вся разница
                    with torch.no_grad():
                        qh = qq.half().float().view(model.n_kv, per, model.hd)
                        t_sc_h = (torch.einsum("qph,skh->qps", qh,
                                               Ka) / (model.hd ** 0.5)).reshape(nh2, nk2).double().cpu().numpy()
                    cmp("scores(q=half)", e_sc.reshape(-1), t_sc_h.reshape(-1))
                    dh2 = int(np.argmax(np.abs(e_sc - t_sc).sum(1)))
                    e3 = np.argsort(-e_sc[dh2])[:3]
                    t3 = np.argsort(-t_sc[dh2])[:3]
                    # по головам: cos по всем 64 головам сразу скрывает картину - часть голов может
                    # совпадать идеально, а часть быть мусором (тогда дело в раскладке q по kv-головам)
                    hc = []
                    for h in range(nh2):
                        ae, at = e_sc[h], t_sc[h]
                        hc.append(float(ae @ at / (np.linalg.norm(ae) * np.linalg.norm(at) + 1e-12)))
                    hc = np.array(hc)
                    good = int((hc > 0.99).sum())
                    print(f"         голов: {good} из {nh2} с cos>0.99; худшие " +
                          " ".join(f"{int(h)}/{hc[h]:.3f}" for h in np.argsort(hc)[:5]) +
                          f"; медиана {np.median(hc):.3f}")
                    # альтернативная раскладка: голова h -> kv h%n_kv (интерлив) вместо h//per
                    with torch.no_grad():
                        q_int = torch.stack([qq[h] for h in range(nh2)]).view(model.n_kv, per, model.hd)
                        q_alt = torch.stack([qq[p2_ * model.n_kv + kv2] for kv2 in range(model.n_kv)
                                             for p2_ in range(per)]).view(model.n_kv, per, model.hd)
                        t_alt = (torch.einsum("qph,skh->qps", q_alt, Ka) / (model.hd ** 0.5)
                                 ).reshape(nh2, nk2).double().cpu().numpy()
                    hc2 = []
                    for h in range(nh2):
                        ae, at = e_sc[h], t_alt[h]
                        hc2.append(float(ae @ at / (np.linalg.norm(ae) * np.linalg.norm(at) + 1e-12)))
                    print(f"         интерлив-раскладка: {int((np.array(hc2) > 0.99).sum())} из {nh2} с cos>0.99 "
                          f"(медиана {np.median(hc2):.3f})")
                    # сдвиг по ключам: если скор движка на ключе j равен скору тренера на j+s,
                    # значит строки кэша повёрнуты rope на другую позицию (off-by-one в inject)
                    for sh in (-2, -1, 1, 2):
                        a_e = e_sc[dh2, max(0, -sh):nk2 - max(0, sh)]
                        a_t = t_sc[dh2, max(0, sh):nk2 - max(0, -sh)]
                        if a_e.size > 8:
                            c = float(a_e @ a_t / (np.linalg.norm(a_e) * np.linalg.norm(a_t) + 1e-12))
                            print(f"         сдвиг {sh:+d}: cos={c:+.4f}")
                    print(f"         голова {dh2}: движок top3 " +
                          " ".join(f"{int(j)}/{e_sc[dh2, j]:.2f}" for j in e3) +
                          " | тренер top3 " + " ".join(f"{int(j)}/{t_sc[dh2, j]:.2f}" for j in t3))
                else:
                    print(f"  (нет {sp}: снимай с STRATA_EAGLE_DUMP_SCORES)")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                    lg, hid2, _, _ = model.chain_depth(
                        torch.from_numpy(K).to(device), torch.from_numpy(V).to(device),
                        torch.from_numpy(hv[None]).to(device), torch.from_numpy(ev[None]).to(device),
                        torch.tensor([pos], device=device), torch.arange(rows, device=device))
                print(f"  кэш: {rows} строк из {kvdump}")
                # o стадий движка - выход внимания до o_proj; у тренера он внутри chain_depth,
                # поэтому сравниваем следующий наблюдаемый уровень: логиты
                # логиты движок пишет рядом с ROUND-дампом (STRATA_EAGLE_DUMP_CTX_LOGITS=1
                # даёт ~/eagle_round.bin.logits), а не рядом со stage-файлом
                lp = kvdump + ".logits"
                if os.path.exists(lp):
                    rb = open(lp, "rb").read()
                    n2, = struct.unpack("<i", rb[:4])
                    engd = np.frombuffer(rb[4:4 + 4 * n2], dtype="<f4")
                    lgf = lg.float().cpu().numpy()[0]
                    a = lgf.astype(np.float64); b = engd.astype(np.float64)
                    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
                    eid = int(np.argmax(engd))
                    print(f"  логиты cos={cos:+.6f} maxdiff={np.max(np.abs(a - b)):.4g} "
                          f"id_движка={eid} тренер_top1={int(np.argmax(lgf))} "
                          f"ранг_id_движка={int((a > a[eid]).sum())}")
                else:
                    print(f"  (нет {lp} - снимай шаг с STRATA_EAGLE_DUMP_CTX_LOGITS)")
        else:
            print("  (в дампе нет кэша: снимай с STRATA_EAGLE_DUMP_CTX_KV=1)")
    else:
        print("  (нет KV-дампа рядом - пропускаю o/логиты)")
    print("\nчтение: e/f разные -> нормы/входы; q/k/v разные -> веса или rope; "
          "q/k/v совпали -> расходятся сами K/V-строки или маска.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
