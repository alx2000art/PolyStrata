#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eagle3_compare.py - the engine's head forward vs the trainer's, stage by stage.

The engine dumps the FIRST read inject's last row's stages (STRATA_EAGLE_DUMP=<file>):
  taps[3*E] fc[E] emb[E] ln1[E] hnorm[E] normed[E]  (float32)
and, once the first draft step runs, normed[E] logits[dvocab] of that step.
This script runs the trainer's Eagle3Head over the SAME row (the first block of the v2 taps dump,
last row of its first inject = the same row the engine dumped), and compares:
  fc == fc(taps)?  ln1 == ln1(emb)?  hnorm == hnorm(fc)?  normed == the full layer's output?
A mismatch names the stage that diverges (taps order, rope, attention, norms).

    python3 eagle3_compare.py --dump ~/eagle_taps/taps-XXXX.bin --engine-dump ~/eagle_dump.bin \
        --safetensors ... --config ...
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

CMP_VER = "v3 2026-11-18 (q/k/v stages)"


def read_engine_dump(path: str, n_embd: int, kv_width: int = 0) -> dict:
    data = np.fromfile(path, dtype=np.float32)
    e = n_embd
    names = ["taps", "fc", "emb", "ln1", "hnorm", "normed", "o_out", "o_mlp", "q", "k", "v", "o_att"]
    out, off = {}, 0
    widths = {"taps": 3 * e, "k": kv_width or None, "v": kv_width or None}
    for name in names:
        w = widths.get(name, e)
        if w is None:
            break
        if off + w > len(data):
            break
        out[name] = data[off:off + w]
        off += w
    out["_rest"] = data[off:]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="v2 taps dump (taps-*.bin) with the read's rows")
    ap.add_argument("--engine-dump", required=True, help="STRATA_EAGLE_DUMP file of the same read")
    ap.add_argument("--safetensors", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--row", type=int, default=-1, help="row index inside block 0 (default: last of its first inject)")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from eagle3_ft_train import Eagle3Head, load_model, load_block, scan_blocks  # noqa: E402
    import torch
    import torch.nn.functional as F

    head_cfg = json.load(open(args.config, encoding="utf-8")) if args.config else {}
    eps = float(head_cfg.get("rms_norm_eps", 1e-6))
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(args.safetensors, head_cfg, eps, device)
    E = meta["n_embd"]

    print(f"eagle3_compare {CMP_VER}")
    blocks, _, _ = scan_blocks([args.dump])
    if not blocks:
        print("нет блоков в дампе", file=sys.stderr)
        return 1
    path, off, pos0, n_rows, tokens = blocks[0]
    print(f"блок 0: {n_rows} строк, pos0 {pos0}; движковый дамп — стадии его первого inject")
    row = args.row if args.row >= 0 else min(2047, n_rows - 1)
    # грузим блок целиком (как в поиске): путь load_block(n=1) отдавал не ту строку - два способа
    # чтения одного файла расходились, и сравнение смотрело на чужие taps
    pos0_b, tokens_b, embs_b, taps_b = load_block(path, off, min(n_rows, max(row + 1, 8192)),
                                                  E, meta["taps_n"])
    if row >= embs_b.shape[0]:
        print(f"строка {row} вне загруженного среза {embs_b.shape[0]}", file=sys.stderr)
        return 1

    eng = read_engine_dump(args.engine_dump, E, model.n_kv * model.hd)
    print(f"стадии движка: {', '.join(k for k in eng if not k.startswith('_'))} "
          f"(+{eng['_rest'].size} float хвостом)")
    if "taps" not in eng or "fc" not in eng:
        print("в движковом дампе нет стадий - включи STRATA_EAGLE_DUMP", file=sys.stderr)
        return 1

    x_embs = torch.from_numpy(embs_b[:row + 1]).to(device)
    x_taps = torch.from_numpy(taps_b[:row + 1]).to(device)
    pos = torch.arange(pos0_b, pos0_b + row + 1, device=device)
    with torch.no_grad():
        h = model.fc(x_taps.reshape(row + 1, -1))
        e = torch.from_numpy(embs_b[:row + 1]).to(device)
        o = model._residual(torch.cat((model.ln1(e), model.hnorm(h)), dim=-1), pos)
        wo_out = model.wo(o)
        hidden_a = h + wo_out
        norm2 = model.ln2(hidden_a)
        mlp_out = model.down(F.silu(model.gate(norm2)) * model.up(norm2))
        hidden = hidden_a + mlp_out
        normed = model.out_norm(hidden)
        got = {
            "fc": h[row].float().cpu().numpy(),
            "emb": embs_b[row],
            "ln1": model.ln1(e[row:row + 1]).float().cpu().numpy()[0],
            "hnorm": model.hnorm(h[row:row + 1]).float().cpu().numpy()[0],
            "normed": normed[row].float().cpu().numpy(),
            "o_out": wo_out[row].float().cpu().numpy(),
            "o_mlp": mlp_out[row].float().cpu().numpy(),
        }
        from eagle3_ft_train import rope_apply as _ra2
        xin = torch.cat((model.ln1(e), model.hnorm(h)), dim=-1)
        Tq = xin.shape[0]
        q_t = _ra2(model.wq(xin).view(Tq, model.n_head, model.hd), pos, model.rope)
        k_t = _ra2(model.wk(xin).view(Tq, model.n_kv, model.hd), pos, model.rope)
        v_t = model.wv(xin).view(Tq, model.n_kv, model.hd)
        got["q"] = q_t[row].float().cpu().numpy().reshape(-1)
        got["k"] = k_t[row].float().cpu().numpy().reshape(-1)
        got["v"] = v_t[row].float().cpu().numpy().reshape(-1)
        got["o_att"] = o[row].float().cpu().numpy().reshape(-1)   # attention output, pre-o_proj

    print(f"\nстрока {row} блока 0 (абс. поз. {pos0_b + row}):")
    for name in ("taps", "fc", "emb", "ln1", "hnorm", "normed", "o_out", "o_mlp", "q", "k", "v", "o_att"):
        if name not in eng:
            continue
        a = np.asarray(eng[name], dtype=np.float64)
        b = np.asarray(got.get(name, np.full_like(a, np.nan)), dtype=np.float64)
        if name == "taps":
            b = np.asarray(taps_b[row].reshape(-1), dtype=np.float64)
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        print(f"  {name:7s} cos={cos:+.4f}  |a-b|max={np.abs(a - b).max():.4g}  "
              f"|a|={np.linalg.norm(a):.3g} |b|={np.linalg.norm(b):.3g}")

    # какую строку дампа движок снял на самом деле: ищем лучший cos по всем блокам
    print("\nпоиск строки движкового дампа по всем блокам:")
    E3 = 3 * E
    for name, key in (("emb", embs_b),):
        a = np.asarray(eng[name], dtype=np.float32)
        an = float(np.linalg.norm(a))
        best = []
        for bi, (path_b, off_b, pos0_b2, n_b, _tok) in enumerate(blocks):
            if bi > 2:
                break
            _p, _t, emb_b, tap_b = load_block(path_b, off_b, min(n_b, 8192), E, meta["taps_n"])
            flat = emb_b.reshape(-1, E)
            dots = flat @ a
            norms = np.linalg.norm(flat, axis=1) * max(an, 1e-12)
            cos = dots / np.maximum(norms, 1e-12)
            idx = np.argsort(-cos)[:3]
            for i in idx:
                best.append((float(cos[i]), bi, int(i), int(pos0_b2 + i)))
        best.sort(reverse=True)
        print(f"  {name}: " + ", ".join(f"cos={c:+.4f} блок {b} строка {r} (поз {p})" for c, b, r, p in best[:5]))

    # стадии движка через НАШУ математику - идентичность строки не нужна: если cos ~1 на всех
    # трёх, то fc/нормы/веса головы у движка и тренера - одни и те же, и расхождение сидит
    # в attention/позиции/подаче (то, что от строки зависит), а не в весах и конвенции
    # перебор вариантов attention: какая семантика даёт выход, совпадающий с движковым o_out.
    # чей вариант совпадёт - та геометрия у движка, и под неё надо учить/править тренер
    if "o_out" in eng:
        import math as _math
        with torch.no_grad():
            hh = model.fc(x_taps.reshape(row + 1, -1))
            xin = torch.cat((model.ln1(e), model.hnorm(hh)), dim=-1)
            T = xin.shape[0]
            n_head, n_kv, hd = model.n_head, model.n_kv, model.hd
            per = n_head // n_kv
            q = model.wq(xin).view(T, n_head, hd)
            k = model.wk(xin).view(T, n_kv, hd)
            v = model.wv(xin).view(T, n_kv, hd)

            from eagle3_ft_train import rope_apply as _ra
            q = _ra(q, pos, model.rope)
            k = _ra(k, pos, model.rope)
            eng_o = np.asarray(eng["o_out"], dtype=np.float64)
            i = torch.arange(T, device=xin.device)
            causal = i.view(-1, 1) >= i.view(1, -1)
            print("\nперебор attention-вариантов (cos с выходом движка o_out):")
            xsw = torch.cat((model.hnorm(hh), model.ln1(e)), dim=-1)   # [h | emb] вместо [emb | h]
            for xname, xin2 in (("cat=eh", xin), ("cat=he", xsw)):
                q2 = model.wq(xin2).view(T, n_head, hd)
                k2 = model.wk(xin2).view(T, n_kv, hd)
                v2 = model.wv(xin2).view(T, n_kv, hd)
                q2 = _ra(q2, pos, model.rope)
                k2 = _ra(k2, pos, model.rope)
                for gqa in ("adjacent", "interleaved"):
                    for s in (1.0, 0.5, 0.2, 0.1, 0.05):
                        if gqa == "adjacent":
                            qg = q2.view(T, n_kv, per, hd)
                        else:
                            qg = q2.view(T, per, n_kv, hd).permute(0, 2, 1, 3)
                        kg = k2
                        att = (torch.einsum("tqph,skh->tqps", qg, kg) / _math.sqrt(hd)) * s
                        w = torch.softmax(att.masked_fill(~causal[:, None, None, :], float("-inf")), dim=-1)
                        o = torch.einsum("tqps,skh->tqph", w, v2).reshape(T, n_head * hd)
                        oo = model.wo(o)[row].float().cpu().numpy().astype(np.float64)
                        c = float(eng_o @ oo / max(np.linalg.norm(eng_o) * np.linalg.norm(oo), 1e-12))
                        print(f"  {xname} {gqa:12s} mscale {s:.4f}: cos={c:+.4f}  |b|={np.linalg.norm(oo):.4g} "
                              f"(движок |a|={np.linalg.norm(eng_o):.4g})")

    # фактический угол rope движка по парам: из пре-rope q и пост-rope q угол восстанавливается
    # однозначно; сравнение с нашим theta по парам покажет закономерность расхождения
    if "q" in eng:
        import math as _m2
        with torch.no_grad():
            hh = model.fc(x_taps.reshape(row + 1, -1))
            xin = torch.cat((model.ln1(e), model.hnorm(hh)), dim=-1)
            pre = model.wq(xin)[row].float().cpu().numpy().reshape(model.n_head, model.hd)
            engq = np.asarray(eng["q"], dtype=np.float64).reshape(model.n_head, model.hd)
            a = pre[0, 0::2].astype(np.float64)
            b = pre[0, 1::2].astype(np.float64)
            x0 = engq[0, 0::2]
            x1 = engq[0, 1::2]
            r2 = np.maximum(a * a + b * b, 1e-30)
            cc = (a * x0 + b * x1) / r2
            ss = (a * x1 - b * x0) / r2
            theta_eng = np.arctan2(ss, cc)
            from eagle3_ft_train import yarn_phase as _yp
            iw = torch.arange(model.hd // 2, dtype=torch.float32)
            pos_t = torch.tensor([float(row)])
            th, _ms = _yp(pos_t, iw, model.rope["base"], model.rope["freq_scale"],
                          model.rope["ext_factor"], model.rope["corr0"], model.rope["corr1"], model.hd)
            theta_ours = th[0].numpy()
            print(f"\nrope-углы по парам (движок vs наш, голова 0, позиция {row}):")
            for i in (0, 4, 8, 12, 16, 19, 20, 21, 24, 28, 31):
                d = float(np.mod(theta_eng[i] - theta_ours[i] + np.pi, 2 * np.pi) - np.pi)
                print(f"  пара {i:2d}: движок {theta_eng[i]:+.4f}  наш {theta_ours[i]:+.4f}  "
                      f"diff {d:+.4f} (в периодах {d / (2 * np.pi):+.4f})")

    # attention из САМИХ стадий движка: q/k разворачиваем из интерлива к HF-порядку, прошлые
    # ключи/значения берём наши (в дампе только текущая строка), считаем softmax и сравниваем с o_att
    if all(k in eng for k in ("q", "k", "v", "o_att")):
        import math as _m3
        with torch.no_grad():
            hh = model.fc(x_taps.reshape(row + 1, -1))
            xin = torch.cat((model.ln1(e), model.hnorm(hh)), dim=-1)
            T = xin.shape[0]
            n_head, n_kv, hd = model.n_head, model.n_kv, model.hd
            per = n_head // n_kv
            engq = np.asarray(eng["q"], dtype=np.float64).reshape(n_head, hd)
            engk = np.asarray(eng["k"], dtype=np.float64).reshape(n_kv, hd)
            q_un = np.concatenate((engq[:, 0::2], engq[:, 1::2]), axis=1)
            k_un = np.concatenate((engk[:, 0::2], engk[:, 1::2]), axis=1)
            v_row = np.asarray(eng["v"], dtype=np.float64).reshape(n_kv, hd)
            qh = _ra2(model.wq(xin).view(T, n_head, hd), pos, model.rope)
            kh = _ra2(model.wk(xin).view(T, n_kv, hd), pos, model.rope)
            vh = model.wv(xin).view(T, n_kv, hd)
            qa = qh.clone(); qa[row] = torch.from_numpy(q_un).to(qh.dtype)
            ka = kh.clone(); ka[row] = torch.from_numpy(k_un).to(kh.dtype)
            va = vh.clone(); va[row] = torch.from_numpy(v_row).to(vh.dtype)
            i2 = torch.arange(T, device=xin.device)
            causal2 = i2.view(-1, 1) >= i2.view(1, -1)
            qg = qa.view(T, n_kv, per, hd)
            att = torch.einsum("tqph,skh->tqps", qg, ka) / _m3.sqrt(hd)
            w = torch.softmax(att.masked_fill(~causal2[:, None, None, :], float("-inf")), dim=-1)
            o2 = torch.einsum("tqps,skh->tqph", w, va).reshape(T, n_head * hd)[row]
            o2 = o2.float().cpu().numpy().astype(np.float64)
            a = np.asarray(eng["o_att"], dtype=np.float64)
            c = float(a @ o2 / max(np.linalg.norm(a) * np.linalg.norm(o2), 1e-12))
            print("")
            print(f"attention из стадий движка: cos(o_att, наш)={c:+.4f}  "
                  f"|движок|={np.linalg.norm(a):.3g} |наш|={np.linalg.norm(o2):.3g}")
    # самоконтроль дампа: у движка o_out = wo(o_att); если нет - o_att снят не с той строки
    if "o_att" in eng and "o_out" in eng:
        with torch.no_grad():
            a = torch.from_numpy(np.asarray(eng["o_att"], dtype=np.float32)[None]).to(device)
            chk = model.wo(a).float().cpu().numpy()[0].astype(np.float64)
        b = np.asarray(eng["o_out"], dtype=np.float64)
        c = float(chk @ b / max(np.linalg.norm(chk) * np.linalg.norm(b), 1e-12))
        print("")
        print(f"самоконтроль движка: cos(wo(o_att), o_out)={c:+.4f}  "
              f"|wo(o_att)|={np.linalg.norm(chk):.3g} |o_out|={np.linalg.norm(b):.3g}")

    # движок интерливит строки q/k при загрузке (w_rope: 2i<-i, 2i+1<-i+32): разворачиваем и сверяем
    if "q" in eng and "k" in eng:
        engq = np.asarray(eng["q"], dtype=np.float64).reshape(model.n_head, model.hd)
        engk = np.asarray(eng["k"], dtype=np.float64).reshape(model.n_kv, model.hd)
        q_un = np.concatenate((engq[:, 0::2], engq[:, 1::2]), axis=1).reshape(-1)
        k_un = np.concatenate((engk[:, 0::2], engk[:, 1::2]), axis=1).reshape(-1)
        for name, a in (("q", q_un), ("k", k_un)):
            b = np.asarray(got[name], dtype=np.float64)
            c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            print(f"  {name} (развёрнутая перестановка движка) cos={c:+.4f}")

    print("\nсверка стадий движка с нашей математикой (входы - сами стадии движка):")
    with torch.no_grad():
        t_eng = torch.from_numpy(np.asarray(eng["taps"], dtype=np.float32)[None]).to(device)
        fc_self = model.fc(t_eng).float().cpu().numpy()[0]
        e_eng = torch.from_numpy(np.asarray(eng["emb"], dtype=np.float32)[None]).to(device)
        ln1_self = model.ln1(e_eng).float().cpu().numpy()[0]
        fc_in = torch.from_numpy(np.asarray(eng["fc"], dtype=np.float32)[None]).to(device)
        hn_self = model.hnorm(fc_in).float().cpu().numpy()[0]
    for name, mine in (("fc", fc_self), ("ln1", ln1_self), ("hnorm", hn_self)):
        a = np.asarray(eng[name], dtype=np.float64)
        b = np.asarray(mine, dtype=np.float64)
        cos = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
        print(f"  {name:7s} cos={cos:+.4f}  |a-b|max={np.abs(a - b).max():.4g}  "
              f"|a|={np.linalg.norm(a):.3g} |b|={np.linalg.norm(b):.3g}")

    # taps по отдельности: подмена порядка слоёв видна как перестановка единиц и нулей
    print("\ntaps по отдельности (движок vs наша строка):")
    et = np.asarray(eng["taps"], dtype=np.float64).reshape(-1, E)
    for tp in range(et.shape[0]):
        for cand in range(et.shape[0]):
            a, b = et[tp], np.asarray(taps_b[row][cand], dtype=np.float64)
            c = float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
            if abs(c) > 0.5:
                print(f"  движковый tap{tp} ~ наш tap{cand}: cos={c:+.4f}")
        best = max((float(et[tp] @ np.asarray(taps_b[row][cand], dtype=np.float64)) /
                    max(np.linalg.norm(et[tp]) * np.linalg.norm(taps_b[row][cand]), 1e-12), cand)
                   for cand in range(et.shape[0]))
        if abs(best[0]) <= 0.5:
            print(f"  движковый tap{tp}: лучший cos={best[0]:+.4f} (tap{best[1]}) - не наша строка")
    # и поиск строки по taps
    print("\nпоиск строки движкового дампа по taps:")
    a = et.reshape(-1)
    an = float(np.linalg.norm(a))
    best = []
    for bi, (path_b, off_b, pos0_b2, n_b, _tok) in enumerate(blocks):
        _p, _t, _emb, tap_b = load_block(path_b, off_b, min(n_b, 8192), E, meta["taps_n"])
        flat = tap_b.reshape(-1, 3 * E)
        cos = (flat @ a) / np.maximum(np.linalg.norm(flat, axis=1) * max(an, 1e-12), 1e-12)
        idx = np.argsort(-cos)[:3]
        for i in idx:
            best.append((float(cos[i]), bi, int(i), int(pos0_b2 + i)))
    best.sort(reverse=True)
    print("  " + ", ".join(f"cos={c:+.4f} блок {b} строка {r} (поз {p})" for c, b, r, p in best[:5]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
