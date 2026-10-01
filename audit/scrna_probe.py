# -*- coding: utf-8 -*-
"""
scrna_probe.py — 核验资源内唯一的单细胞对象能否充当「检出性真值」锚点。
Melanoma_scRNA.h5ad: 33,524 cell x 34,853 gene，含 cell_types / customclassif / gene_cell_types 与 1,314 个 cuTAR。
输出：
  T1 每个 cuTAR 的真实表达细胞比例（跨模态可检出性真值）
  T2 cuTAR 检测率 vs 细胞类型数 / 是否细胞类型受限
  T3 与空间矩阵 cuTAR 检出率的对照（同一批转录本、不同平台）
  T4 var 名重复等结构缺陷
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import anndata as ad
import scipy.sparse as sp

ROOT = _cfg.WORKSPACE + r""
P = os.path.join(ROOT, r"data\SPanC-Lnc-pan\Melanoma_scRNA.h5ad")
OUT, W = [], lambda s="": (print(s, flush=True), OUT.append(s))

a = ad.read_h5ad(P, backed="r")
names = np.array([str(x) for x in a.var.index])
is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
idx_cu = np.where(is_cu)[0]
n_cell, n_gene = a.shape

W("=" * 100)
W("Melanoma_scRNA.h5ad — 单细胞对象核验")
W("=" * 100)
W(f"  {n_cell:,} cells x {n_gene:,} genes | cuTAR 特征 {len(idx_cu):,} ({100*is_cu.mean():.2f}%)")
W(f"  X = {type(a.X).__name__} {getattr(a.X,'dtype','')}")

# T4 结构缺陷
from collections import Counter
cc = Counter(names.tolist())
dups = {k: v for k, v in cc.items() if v > 1}
W(f"\n  T4 结构缺陷:")
W(f"    var.index 重复名 = {len(dups)} 个, 涉及 {sum(dups.values())} 行")
W(f"    重复的 cuTAR 名（前 10）= {[k for k in dups if k.upper().startswith('CUTAR')][:10]}")
W(f"    var 只有 1 列: {list(a.var.columns)}（无 gene symbol 之外的注释）")
W(f"    obs 列: {list(a.obs.columns)}")
W(f"    → 这是资源中唯一带细胞类型标签的对象；10 个空间对象全部无样本/区域注释")

# 细胞类型
ct_col = "cell_types" if "cell_types" in a.obs.columns else None
ct = a.obs[ct_col].astype(str).values if ct_col else None
cats = sorted(set(ct)) if ct is not None else []
W(f"\n  细胞类型（{ct_col}）{len(cats)} 类: {cats}")
cnt = {c: int((ct == c).sum()) for c in cats}
W(f"    各类型细胞数: {cnt}")
if "orig.ident" in a.obs.columns:
    oi = a.obs["orig.ident"].astype(str).values
    W(f"    样本（orig.ident）: {sorted(set(oi))}")

# T1 逐 cuTAR 真实检出
det = np.zeros(len(idx_cu), np.int64)
tot = np.zeros(len(idx_cu), np.float64)
ct_hit = {c: np.zeros(len(idx_cu), np.int64) for c in cats}
CH = 50
for s in range(0, len(idx_cu), CH):
    cols = idx_cu[s:s + CH]
    blk = a[:, cols].to_memory().X
    blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
    blk = blk.astype(np.float64)
    det[s:s + CH] = (blk > 0).sum(0)
    tot[s:s + CH] = blk.sum(0)
    for c in cats:
        m = (ct == c)
        ct_hit[c][s:s + CH] = (blk[m] > 0).sum(0)
    print(f"    chunk {s+len(cols)}/{len(idx_cu)}", flush=True)

pct = 100.0 * det / n_cell
W(f"\n  T1 1,314 个 cuTAR 在 33,524 个细胞中的真实检出:")
W(f"    检出细胞比例: min={pct.min():.3f}% 中位={np.median(pct):.3f}% max={pct.max():.2f}%")
W(f"    零检出 cuTAR 数 = {int((det==0).sum())} ({100*(det==0).mean():.1f}%)")
for thr in (1, 5, 10, 20, 50):
    W(f"    >= {thr:>2}% 细胞的 cuTAR = {int((pct>=thr).sum()):>5} ({100*(pct>=thr).mean():5.1f}%)")

# T2 细胞类型受限性
nz = det > 0
top_frac, n_ct_ge10 = np.zeros(len(idx_cu)), np.zeros(len(idx_cu), np.int64)
for i, c in enumerate(cats):
    f = np.divide(ct_hit[c], max(1, cnt[c]), out=np.zeros(len(idx_cu)), where=cnt[c] > 0)
    top_frac = np.maximum(top_frac, f)
    n_ct_ge10 += (f >= 0.10)
top_frac = 100.0 * top_frac
W(f"\n  T2 细胞类型受限性（设 11 类，均匀分布时每类占 9.1%）:")
W(f"    检测率 >=10% 覆盖的细胞类型数: 中位={np.median(n_ct_ge10[nz]):.0f} "
  f"max={n_ct_ge10.max()} | 仅 1 类 >=10% 的 cuTAR = {int((n_ct_ge10[nz]==1).sum())}")
W(f"    单一类型最高检出率: 中位={np.median(top_frac[nz]):.2f}%  max={top_frac.max():.2f}%")
best = np.argsort(-top_frac)[:10]
W(f"    {'cuTAR':<16}{'cells%':>8}{'best_ctype':>34}{'that_ctype%':>14}")
for i in best:
    c = cats[int(np.argmax([np.divide(ct_hit[k][i], max(1, cnt[k])) for k in cats]))]
    W(f"    {names[idx_cu[i]]:<16}{pct[i]:>8.2f}{c:>34}{top_frac[i]:>14.2f}")

# T3 与空间矩阵对照
sp_ref = {}
for tag, rel in [("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
                 ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                 ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
                 ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
                 ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]:
    b = ad.read_h5ad(os.path.join(ROOT, rel))
    bn = [str(x) for x in b.var.index]
    m = {n: i for i, n in enumerate(bn)}
    Xb = b.X.toarray() if sp.issparse(b.X) else np.asarray(b.X)
    d = {}
    for i, g in enumerate(names[idx_cu]):
        if g in m:
            d[g] = float((Xb[:, m[g]] > 0).mean())
    sp_ref[tag] = d
    W(f"    空间 {tag}: 与 scRNA 共有的 cuTAR 名 = {len(d):,} / {len(idx_cu):,}")
    del b, Xb

W(f"\n  T3 同一转录本、跨平台检出率对照（scRNA 细胞 vs 空间 spot）:")
W(f"    {'pair':<28}{'n_shared':>10}{'scRNA中位%':>13}{'空间中位%':>12}{'Spearman':>10}")
rows = []
for tag, d in sp_ref.items():
    if len(d) < 20:
        continue
    gs = list(d.keys())
    p_sc = np.array([pct[list(names[idx_cu]).index(g)] for g in gs])
    p_sp = np.array([100 * d[g] for g in gs])
    rho = float(np.corrcoef(np.argsort(np.argsort(p_sc)), np.argsort(np.argsort(p_sp)))[0, 1])
    W(f"    {'Melanoma_scRNA vs ' + tag:<28}{len(gs):>10,}{np.median(p_sc):>13.3f}{np.median(p_sp):>12.3f}{rho:>10.3f}")
    rows.append({"pair": tag, "n": len(gs), "med_sc": float(np.median(p_sc)),
                 "med_sp": float(np.median(p_sp)), "spearman": rho})

res = {"n_cell": int(n_cell), "n_gene": int(n_gene), "n_cuTAR": int(len(idx_cu)),
       "n_dup_names": len(dups),
       "pct_cells_median": float(np.median(pct)), "pct_cells_max": float(pct.max()),
       "n_zero_detection": int((det == 0).sum()),
       "n_cuTAR_ge10pct_cells": int((pct >= 10).sum()),
       "n_cuTAR_ge1pct_cells": int((pct >= 1).sum()),
       "cell_types": cats, "cross_platform": rows}
with open(os.path.join(ROOT, "results", "SCRNA_probe.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)

# 逐基因数组，供出图使用
np.savez_compressed(
    os.path.join(ROOT, "results", "SCRNA_pergene.npz"),
    cu_names=names[idx_cu], pct=pct, det=det, tot=tot,
    top_frac=top_frac, n_ct_ge10=n_ct_ge10,
    ct_matrix=np.vstack([np.divide(ct_hit[c], max(1, cnt[c])) * 100 for c in cats]),
    ct_names=np.array(cats), ct_counts=np.array([cnt[c] for c in cats]),
    sp_tags=np.array(list(sp_ref.keys())),
    sp_names=np.array([",".join(sp_ref[k].keys()) for k in sp_ref.keys()]),
    sp_pct=np.array([",".join(f"{100*v:.6f}" for v in sp_ref[k].values()) for k in sp_ref.keys()]),
)
with open(os.path.join(ROOT, "results", "SCRNA_probe.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
print("\n[written] results\\SCRNA_probe.txt/.json")
