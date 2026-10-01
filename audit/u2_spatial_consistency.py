# -*- coding: utf-8 -*-
"""
u2_spatial_consistency.py — U2：空间-空间检出率一致性检验
目的：把"跨模态不相关"里的两种解释分开：
  (i) 检出由平台/特征决定  ->  同一转录本在不同空间矩阵间的检出率应当**高度一致**（即便癌种不同）
  (ii) 检出反映样本生物学  ->  空间-空间一致性与空间-scRNA 一致性应当**同量级**

同时给出 D2 的初步数字：每条 cuTAR 达到 10% 检出所需的深度倍数。
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
OUT = []
W = lambda s="": (print(s, flush=True), OUT.append(s),
                  open(os.path.join(ROOT, "results", "U2_spatial_consistency.txt"),
                       "w", encoding="utf-8").write("\n".join(OUT)))

MATS = [
    ("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad", "PacBio cuTAR-only"),
    ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad", "PacBio cuTAR-only"),
    ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad", "ONT cuTAR-only"),
    ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad", "ONT cuTAR-only"),
    ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad", "ONT cuTAR-only"),
    ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad", "short-read mixed"),
    ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad", "short-read mixed"),
    ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad", "short-read mixed"),
    ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad", "short-read mixed"),
    ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad", "short-read mixed"),
]


def spearman(a, b):
    if len(a) < 5 or len(np.unique(a)) < 2 or len(np.unique(b)) < 2:
        return np.nan
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


D = {}
for tag, rel, grp in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    if sp.issparse(X):
        Xc = X.tocsc()
        fd = np.diff(Xc.indptr).astype(np.int64)
    else:
        fd = (np.asarray(X) > 0).sum(0).astype(np.int64)
    n = a.n_obs
    idx = np.where(is_cu)[0]
    det = fd / n
    D[tag] = {"grp": grp, "n": n,
              "names": np.array([names[i] for i in idx]),
              "det": det[idx]}
    W(f"  载入 {tag:<16} n={n:>6,}  cuTAR={len(idx):>6,}  最高检出率={100*det[idx].max():.2f}%")
    del a, X

W("\n" + "=" * 104)
W("U2  空间-空间 cuTAR 检出率一致性（共享特征上的 Spearman）")
W("=" * 104)
W(f"  {'pair':<36}{'n共享':>8}{'rho':>9}{'组别关系':>26}")
pairs = []
for i in range(len(MATS)):
    for j in range(i + 1, len(MATS)):
        ti, tj = MATS[i][0], MATS[j][0]
        di, dj = D[ti], D[tj]
        m = {g: k for k, g in enumerate(dj["names"])}
        common = [(k, m[g]) for k, g in enumerate(di["names"]) if g in m]
        if len(common) < 5:
            continue
        x = np.array([di["det"][k] for k, _ in common])
        y = np.array([dj["det"][k] for _, k in common])
        r = spearman(x, y)
        rel = "同癌种" if ti.split("_")[0] == tj.split("_")[0] else \
              ("同平台" if di["grp"] == dj["grp"] else "跨癌种跨平台")
        pairs.append(dict(a=ti, b=tj, n=len(common), rho=r, rel=rel))
        W(f"  {ti+' vs '+tj:<36}{len(common):>8,}{r:>9.3f}{rel:>26}")

W("\n" + "=" * 104)
W("★ 对照：空间-空间 vs 空间-单细胞")
W("=" * 104)
sc = json.load(open(os.path.join(ROOT, "results", "SCRNA_probe.json"), encoding="utf-8"))
scr = {d["pair"]: {"n": d["n"], "rho": d.get("spearman", d.get("rho"))} for d in sc["cross_platform"]}
try:
    rc5 = json.load(open(os.path.join(ROOT, "results", "RECHECK.json"), encoding="utf-8")).get("c5", [])
    for d in rc5:
        if d["pair"] in scr:
            scr[d["pair"]]["ci"] = d["ci"]
except Exception:
    pass
sp_sp = [p["rho"] for p in pairs if not np.isnan(p["rho"])]
sp_sc = [d["rho"] for d in scr.values() if d["rho"] is not None]
W(f"  空间-空间   ρ: n={len(sp_sp)}  中位 {np.median(sp_sp):.3f}  范围 {min(sp_sp):.3f} – {max(sp_sp):.3f}")
W(f"  空间-单细胞 ρ: n={len(sp_sc)}  中位 {np.median(sp_sc):.3f}  范围 {min(sp_sc):.3f} – {max(sp_sc):.3f}")
for k, v in scr.items():
    ci = v.get("ci")
    W(f"      Melanoma_scRNA vs {k:<16} n={v['n']:>4}  ρ={v['rho']:.3f}"
      + (f"  CI=[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""))
# 分层：同癌种 / 同平台跨癌种 / 跨癌种跨平台
for lab, key in [("同癌种", "同癌种"), ("同平台跨癌种", "同平台"), ("跨癌种跨平台", "跨癌种跨平台")]:
    v = [p["rho"] for p in pairs if p["rel"] == key and not np.isnan(p["rho"])]
    if v:
        W(f"      {lab:<14} n={len(v):>3}  中位 {np.median(v):.3f}  范围 {min(v):.3f} – {max(v):.3f}")
W("\n  判读：")
W("    若 空间-空间 ρ 显著高于 空间-单细胞 ρ -> 检出率是转录本在该平台上的**稳定属性**，")
W("       与样本（癌种）关系不大；空间检出不代表真实细胞表达频率。")
W("    若两者同量级 -> '跨模态不相关'可能只是样本间生物学差异，R4 的跨模态论据需要削弱。")

W("\n" + "=" * 104)
W("D2（初步）  每条 cuTAR 达到 10% 检出所需的深度倍数")
W("=" * 104)
W("  说明：按检出率与深度近似成正比（低检出区 Poisson 捕获）估算，**是粗估**，需在正文标注。")
W(f"  {'matrix':<18}{'最高检出率':>11}{'所需倍数':>11}{'总read':>14}")
D2 = []
for tag, rel, grp in MATS:
    d = D[tag]
    mx = float(d["det"].max())
    mult = 0.10 / mx if mx > 0 else np.inf
    tot = None
    try:
        a = ad.read_h5ad(os.path.join(ROOT, rel))
        Xa = a.X
        tot = float(Xa.sum())
        del a, Xa
    except Exception:
        pass
    W(f"  {tag:<18}{100*mx:>10.2f}%{mult:>10.1f}x{tot if tot is None else int(tot):>14,}")
    D2.append(dict(tag=tag, best_det=100 * mx, multiplier=mult, total_reads=tot))

json.dump({"pairs": pairs,
           "median_spatial_spatial": float(np.median(sp_sp)),
           "spatial_scRNA": scr, "d2": D2},
          open(os.path.join(ROOT, "results", "U2_spatial_consistency.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\U2_spatial_consistency.txt/.json")
