# -*- coding: utf-8 -*-
"""tie_rows.py — 每个矩阵中「第6与第7近邻距离精确并列」的 spot 数，以及邻接集合因此不同的行数"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np, anndata as ad
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
ROOT = _cfg.WORKSPACE + r""
K = 6
tot_t = tot_d = tot_n = 0
for tag in ["SCC", "BCC", "HNC", "KidneyCancer", "Melanoma"]:
    a = ad.read_h5ad(os.path.join(ROOT, "data", "SPanC-Lnc-pan", tag + ".h5ad"), backed="r")
    c = np.asarray(a.obsm["spatial"], float); n = len(c)
    D = cdist(c, c); np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    tie = int((np.abs(Ds[:, K] - Ds[:, K - 1]) < 1e-9).sum())
    nb_bf = np.argpartition(D, K - 1, axis=1)[:, :K]
    _, i2 = cKDTree(c).query(c, k=K + 1)
    diff = int((np.sort(nb_bf, 1) != np.sort(i2[:, 1:], 1)).any(1).sum())
    tot_t += tie; tot_d += diff; tot_n += n
    print(f"  {tag:<14} n={n:>5}  6/7 距离精确并列的 spot = {tie:>4} ({100*tie/n:>4.1f}%)"
          f"   邻接集合因此不同 = {diff:>3}")
    del a, D
print(f"  合计 {tot_n} spot；并列 {tot_t} ({100*tot_t/tot_n:.1f}%)；集合不同 {tot_d}")
