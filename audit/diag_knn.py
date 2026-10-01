# -*- coding: utf-8 -*-
"""diag_knn.py — 诊断两件事：(1) 暴力 kNN 与 cKDTree 为何不一致；(2) 坐标来源差异"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, anndata as ad, scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = _cfg.WORKSPACE + r""
K = 6

for tag in ["SCC", "Melanoma"]:
    a = ad.read_h5ad(os.path.join(ROOT, "data", "SPanC-Lnc-pan", tag + ".h5ad"))
    c = np.asarray(a.obsm["spatial"], float)
    n = len(c)
    print(f"\n{'='*90}\n[{tag}] n={n}")
    # 重复坐标？
    uniq = np.unique(c, axis=0)
    print(f"  唯一坐标数 {len(uniq)} / {n}  -> 重复坐标组 {n-len(uniq)}")
    if len(uniq) < n:
        _, cnt = np.unique(c, axis=0, return_counts=True)
        print(f"  重复次数分布：{dict(zip(*np.unique(cnt, return_counts=True)))}")

    D = cdist(c, c); np.fill_diagonal(D, np.inf)
    nb_bf = np.argpartition(D, K - 1, axis=1)[:, :K]
    _, i2 = cKDTree(c).query(c, k=K + 1)
    nb_tk = i2[:, 1:]
    diff = np.where((np.sort(nb_bf, 1) != np.sort(nb_tk, 1)).any(1))[0]
    print(f"  邻接集合不同的行数：{len(diff)} / {n}")
    for r in diff[:4]:
        ds = np.sort(D[r])[:K + 3]
        print(f"   行{r}: 第1..{K+3} 近邻距离 = " + " ".join(f"{v:.6f}" for v in ds))
        print(f"        第{K}近={ds[K-1]:.9f}  第{K+1}近={ds[K]:.9f}  差={ds[K]-ds[K-1]:.3e}")
        print(f"        暴力取 {sorted(nb_bf[r].tolist())}  KDTree取 {sorted(nb_tk[r].tolist())}")

    # 距离并列计数
    ties = 0
    for r in range(n):
        ds = np.sort(D[r])[:K + 1]
        if abs(ds[K] - ds[K - 1]) < 1e-9:
            ties += 1
    print(f"  第6与第7近邻距离并列(<1e-9)的行数：{ties}")

    # 坐标来源
    ic = a.obs["imagecol"].to_numpy(float); ir = a.obs["imagerow"].to_numpy(float)
    print(f"  obsm['spatial'] 前3行：{c[:3].tolist()}")
    print(f"  obs imagecol 前3：{ic[:3].tolist()}  imagerow 前3：{ir[:3].tolist()}")
    print(f"  spatial[:,0] 与 imagecol 的关系: 差的中位={np.median(c[:,0]-ic):.4f} "
          f"比值中位={np.median(c[:,0]/np.where(ic==0,np.nan,ic)):.6f}")
    print(f"  spatial[:,1] 与 imagerow 的关系: 差的中位={np.median(c[:,1]-ir):.4f} "
          f"比值中位={np.median(c[:,1]/np.where(ir==0,np.nan,ir)):.6f}")
    # 等价性：是否只是线性缩放/平移
    for j, (sp_, ob) in enumerate([(c[:, 0], ic), (c[:, 1], ir)]):
        A = np.c_[np.ones(n), ob]
        beta, *_ = np.linalg.lstsq(A, sp_, rcond=None)
        res = np.abs(A @ beta - sp_).max()
        print(f"  spatial[:,{j}] = {beta[0]:.5f} + {beta[1]:.6f}*obs  -> 最大残差 {res:.3e}"
              f"  {'（线性等价）' if res < 1e-3 else '（不等价）'}")
    del a
