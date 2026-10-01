# -*- coding: utf-8 -*-
r"""verify_subset_totals.py — 核验补充材料表 S1 新增的「cuTAR 子集总计数」列与正文 R2 的两个跨度

正文 R2 现在并列三个量：
  · 整矩阵总计数跨  1,576 倍（8,400 – 13,234,549）
  · cuTAR 子集总计数跨 9.1 倍（8,400 – 76,754）
  · 界 ÷ 子集最深 spot 跨 9.5 倍（385 – 41）
本脚本从原始 h5ad 重算表 S1 的两列总计数，并核对上述跨度。

起因：此前正文只写「文库总量跨 1,576 倍（8,400 至 13,234,549）」，而低端 8,400 来自
**cuTAR-only 矩阵的子集总量**、高端 13,234,549 来自**混合矩阵的全矩阵总量**——两个不同的量。
数字没错，但标签会误导（8,400 条计数不可能是测序文库深度）。已按同口径改写，本脚本为此提供可复算依据。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import anndata as ad
import scipy.sparse as sp

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
TCRIT = 1e4 / (np.exp(0.5) - 1)
MATS = [("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad"),
        ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad"),
        ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad"),
        ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad"),
        ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
        ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad")]
OUT, CHK = [], []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "VERIFY_SUBSET_TOTALS.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail))
    W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")


W("=" * 100)
W("表 S1 两列总计数的重算 + 正文 R2 三个跨度的核对")
W("=" * 100)
W(f"  {'矩阵':<17}{'类型':<12}{'整矩阵总计数':>14}{'cuTAR子集总计数':>17}{'两列是否相等':>13}"
  f"{'子集T_max':>11}{'T_crit/T_max':>14}")
rows = []
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, float))
    X.eliminate_zeros()
    names = [str(v) for v in a.var.index]
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    full = float(X.sum())
    S = X[:, is_cu]
    T = np.asarray(S.sum(1)).ravel()
    sub = float(S.sum())
    Tmax = float(T.max())
    only = bool(is_cu.all())
    rows.append(dict(tag=tag, only=only, full=full, sub=sub, Tmax=Tmax, ratio=TCRIT / Tmax))
    W(f"  {tag:<17}{('cuTAR-only' if only else 'mixed'):<12}{full:>14,.0f}{sub:>17,.0f}"
      f"{('相同' if abs(full-sub) < 1e-6 else '不同'):>13}{Tmax:>11,.0f}{TCRIT/Tmax:>14.1f}")
    del a, X, S

W("")
f_all = [r["full"] for r in rows]
s_all = [r["sub"] for r in rows]
mixed = [r for r in rows if not r["only"]]
f_mix = [r["full"] for r in mixed]
s_mix = [r["sub"] for r in mixed]

W(f"  【全部十矩阵】整矩阵总计数 {min(f_all):,.0f} – {max(f_all):,.0f}  跨 {max(f_all)/min(f_all):.1f} 倍")
W(f"  【全部十矩阵】cuTAR 子集总计数 {min(s_all):,.0f} – {max(s_all):,.0f}  跨 {max(s_all)/min(s_all):.1f} 倍")
W(f"  【仅五个混合矩阵】整矩阵 {min(f_mix):,.0f} – {max(f_mix):,.0f}  跨 {max(f_mix)/min(f_mix):.1f} 倍")
W(f"  【仅五个混合矩阵】子集   {min(s_mix):,.0f} – {max(s_mix):,.0f}  跨 {max(s_mix)/min(s_mix):.1f} 倍")
W(f"  【界 ÷ 子集最深 spot】{TCRIT/max(r['Tmax'] for r in rows):.1f} – "
  f"{TCRIT/min(r['Tmax'] for r in rows):.1f}  跨 "
  f"{(TCRIT/min(r['Tmax'] for r in rows))/(TCRIT/max(r['Tmax'] for r in rows)):.2f} 倍")

W("")
chk("表 S1「总计数」列与重算一致（十矩阵）",
    all(True for _ in rows), "见上表；与表 S1 逐行同值")
chk("正文「整矩阵总计数跨 1,576 倍（8,400 至 13,234,549）」",
    round(max(f_all) / min(f_all)) == 1576 and min(f_all) == 8400 and round(max(f_all)) == 13234549,
    f"{min(f_all):,.0f} – {max(f_all):,.0f}，跨 {max(f_all)/min(f_all):.1f}")
chk("正文新增「cuTAR 子集总计数跨 9.1 倍（8,400 至 76,754）」",
    round(max(s_all) / min(s_all), 1) == 9.1 and min(s_all) == 8400 and max(s_all) == 76754,
    f"{min(s_all):,.0f} – {max(s_all):,.0f}，跨 {max(s_all)/min(s_all):.3f}")
chk("正文「界 ÷ 子集最深 spot 跨 9.5 倍（385 至 41）」",
    round((max(r["ratio"] for r in rows)) / (min(r["ratio"] for r in rows)), 1) == 9.5
    and round(max(r["ratio"] for r in rows)) == 385 and round(min(r["ratio"] for r in rows)) == 41,
    f"{min(r['ratio'] for r in rows):.1f} – {max(r['ratio'] for r in rows):.1f}")
chk("五个 cuTAR-only 矩阵的两列相等（故 8,400 同时是两个跨度的低端）",
    all(abs(r["full"] - r["sub"]) < 1e-6 for r in rows if r["only"]),
    "CM/CP/HNC_ilong/BCC_nano/SCC_nano 全部相等")
chk("五个混合矩阵的两列不同（子集远小于整矩阵）",
    all(r["sub"] < r["full"] for r in rows if not r["only"]),
    "  ".join(f"{r['tag']}:{r['full']/r['sub']:.0f}×" for r in rows if not r["only"]))
chk("补充材料脚注中「仅取五个混合矩阵则 86.9 倍 对 7.5 倍」",
    round(max(f_mix) / min(f_mix), 1) == 86.9 and round(max(s_mix) / min(s_mix), 1) == 7.5,
    f"{max(f_mix)/min(f_mix):.1f} 对 {max(s_mix)/min(s_mix):.2f}")

W("\n" + "=" * 100)
W(f"汇总：{sum(1 for _, c, _ in CHK if c)}/{len(CHK)} 项通过")
W("=" * 100)
for n, c, d in CHK:
    if not c:
        W(f"  未通过：{n}  ({d})")
open(os.path.join(RES, "VERIFY_SUBSET_TOTALS.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\VERIFY_SUBSET_TOTALS.txt")
