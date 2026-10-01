# -*- coding: utf-8 -*-
"""
atlas_quadrant_test.py — 检验"图普的 HH/LL 二分在稀有特征上是否等价于检出"

图普方法原文（Nature Methods 2026, Methods）：
  "spots showing (1) high expression of both the gene and the cuTAR -> HH; (2) spots with gene
   expression and no cuTAR expression -> LH; (3) cuTAR expression and lack of gene expression -> HL;
   (4) no expression of both -> LL; and (5) NS for spots with insignificant Moran's I"
判据（"high"/"no expression" 如何界定）**未在任何地方给出**。
其 Moran's I 公式用的是 standardised（z）值 —— 与"相对均值"的 LISA 象限分类一致。

因此两种最自然的读法：
  读法 A（固定截断）：value > t 为"high"
  读法 B（相对均值）：z > 0 即 value > mean 为"high"
本脚本检验：对稀有特征，读法 B 是否也塌缩为"检出"（mean < 最小非零值）。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, numpy as np, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import anndata as ad, scipy.sparse as sp

ROOT = _cfg.WORKSPACE + r""
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
OUT, RES = [], []
def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "ATLAS_quadrant_test.txt"), "w", encoding="utf-8").write("\n".join(OUT))

W("=" * 104)
W("检验：对稀有特征，'相对均值判定 high' 是否等价于 '检出'")
W("  若 mean < 最小非零值，则 value > mean  <=>  value > 0  <=>  检出")
W("=" * 104)
W(f"  {'矩阵':<17}{'cuTAR数':>8}{'mean<min非零 的比例':>21}{'min(最小非零/均值)':>21}{'最大检出率':>11}")
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    Xd = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    S = Xd[:, is_cu]
    T = S.sum(1)
    L = np.log1p(S / np.where(T == 0, 1.0, T)[:, None] * 1e4)      # CP10K + log1p
    nz = L > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        mn = np.where(nz.any(0), np.where(nz, L, np.inf).min(0), np.nan)   # 每个特征最小非零值
        mu = L.mean(0)                                                      # 每个特征均值（含零）
        ratio = mn / mu
    ok = np.isfinite(ratio)
    frac = float((mn[ok] > mu[ok]).mean())
    W(f"  {tag:<17}{int(is_cu.sum()):>8,}{100*frac:>20.1f}%{np.nanmin(ratio):>21.2f}"
      f"{100*(nz.sum(0).max()/a.n_obs):>10.2f}%")
    RES.append(dict(tag=tag, n_cu=int(is_cu.sum()), frac_mean_below_min=frac,
                    min_ratio=float(np.nanmin(ratio)),
                    best_det_pct=float(100 * nz.sum(0).max() / a.n_obs)))
    del a, X, Xd, S, L

W("\n判读：")
W("  'frac_mean_below_min' = 在一个矩阵的 cuTAR 中，'均值 < 最小非零值'成立的比例。")
W("  该值为 100% => 对每一条 cuTAR，'value > 均值' 与 '检出' 是同一件事。")
tot = np.mean([r["frac_mean_below_min"] for r in RES])
W(f"\n  10 个矩阵平均：{100*tot:.1f}% 的 cuTAR 满足'均值 < 最小非零值'")
W(f"  最小比值（最小非零值 / 均值）的全局最小 = {min(r['min_ratio'] for r in RES):.2f}  (>1 即为成立)")
json.dump(RES, open(os.path.join(ROOT, "results", "ATLAS_quadrant_test.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\ATLAS_quadrant_test.txt/.json")
