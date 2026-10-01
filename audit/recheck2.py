# -*- coding: utf-8 -*-
"""
recheck2.py — 复核方案 A 中最关键、也最容易被审稿人击穿的一处对照：
C8/C9  "cuTAR 域恢复不了注释基因域（ARI 0.03-0.07）" 是不是纯粹的特征数效应？

问题：corrected cuTAR 域只用 18 个（SCC）/ 5 个（BCC）特征，而参照域用 1,500 个特征。
低维聚类之间 ARI 本来就可能很低，与 cuTAR 本身无关。必须做三组对照：

  T-cuTAR   : 实际过 10% 门槛的 cuTAR（18 / 5 个）
  T-random  : 随机抽同样数量的蛋白编码基因（100 次）
  T-matched : 按检出率逐一配对的蛋白编码基因（100 次）—— 把"cuTAR 身份"与"可检出性"分离

指标：与参照域（top-1500 蛋白编码，k=4）的 ARI；以及与深度十分位的 ARI。
k 取 2/4/6 以排除簇数选择的影响。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, gzip, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

ROOT = _cfg.WORKSPACE + r""
RNG = np.random.default_rng(7)
OUT = []
W = lambda s="": (print(s, flush=True), OUT.append(s))
NDRAW = 100


def lognorm(S):
    t = S.sum(1).copy()
    t[t == 0] = 1.0
    return np.log1p(S / t[:, None] * 1e4)


def cluster_of(M, k, ncomp=15):
    ncomp = int(min(ncomp, M.shape[1] - 1, M.shape[0] - 1))
    if ncomp < 2:
        return np.zeros(M.shape[0], int)
    Z = TruncatedSVD(n_components=ncomp, random_state=0).fit_transform(M)
    return KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3,
                  random_state=0).fit_predict(Z)


def rank_bins(v, k=10):
    o = np.argsort(v, kind="stable")
    b = np.empty(len(v), int)
    b[o] = (np.arange(len(v)) * k) // len(v)
    return b


gi = {}
with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"), "rt", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    it, isym = hdr.index("type_of_gene"), hdr.index("Symbol")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > max(it, isym):
            gi[p[isym].upper()] = p[it]


def run(tag, rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    is_pc = np.array([gi.get(n.upper(), "") == "protein-coding" for n in names])
    n = X.shape[0]
    fd = (X > 0).sum(0)
    det = fd / n
    sub_tot = X[:, is_cu].sum(1)

    W("\n" + "=" * 104)
    W(f"[{tag}]  {a.n_obs:,} spot | cuTAR {int(is_cu.sum()):,} | 蛋白编码 {int(is_pc.sum()):,}")
    keep = is_cu & (det >= 0.10)
    k_cu = int(keep.sum())
    W(f"  过 10% 检出门槛的 cuTAR = {k_cu} 个 -> 检出率 {np.round(np.sort(det[keep])*100, 2).tolist()}%")
    if k_cu < 2:
        W("  cuTAR 数过少，跳过")
        return None

    # 参照域：top-1500 蛋白编码
    pc_idx = np.where(is_pc)[0]
    hvg = pc_idx[np.argsort(-np.median(X[:, pc_idx], 0))[:1500]]
    s_ref = np.where((X[:, hvg] > 0).sum(1) > 0)[0]
    Lref = lognorm(X[s_ref][:, hvg])
    depth_bin = rank_bins(sub_tot, 10)

    # 配对池
    dcu = det[keep]
    pc_det = det[pc_idx]
    pool = []
    for d in dcu:
        c = pc_idx[(pc_det >= 0.80 * d) & (pc_det <= 1.25 * d)]
        if len(c) == 0:
            c = pc_idx[[np.argmin(np.abs(pc_det - d))]]
        pool.append(c)

    def eval_set(cols, k):
        m = (X[:, cols] > 0).sum(1) > 0
        s = np.where(m)[0]
        if len(s) < 20 or len(np.unique(s)) < 2:
            return np.nan, np.nan, 0
        lab = cluster_of(lognorm(X[s][:, cols]), k)
        common = np.intersect1d(s, s_ref)
        if len(common) < 20:
            return np.nan, np.nan, len(s)
        lab_ref = cluster_of(Lref[np.isin(s_ref, common)], k)
        la = np.asarray(lab)[np.isin(s, common)]
        r = float(ari(lab_ref, la))
        d10 = float(ari(rank_bins(sub_tot[s], 10)[np.isin(s, np.arange(len(sub_tot)))] if False
                        else rank_bins(sub_tot, 10)[s], lab))
        return r, d10, len(s)

    cu_cols = np.where(keep)[0]
    W(f"\n  {'k':>3}{'T-cuTAR ARI(vs ref)':>21}{'T-random mean[2.5,97.5]':>28}"
      f"{'T-matched mean[2.5,97.5]':>28}{'T-cuTAR ARI(vs depth)':>24}")
    rows = []
    for k in (2, 4, 6):
        r_cu, d_cu, nsp = eval_set(cu_cols, k)
        rnd, mat = [], []
        for _ in range(NDRAW):
            cols = RNG.choice(pc_idx, k_cu, replace=False)
            v, _, _ = eval_set(cols, k)
            if not np.isnan(v):
                rnd.append(v)
            cols2 = np.array([RNG.choice(p) for p in pool])
            v2, _, _ = eval_set(cols2, k)
            if not np.isnan(v2):
                mat.append(v2)
        rnd, mat = np.array(rnd), np.array(mat)
        f = lambda z: f"{z.mean():.3f} [{np.percentile(z,2.5):.3f}, {np.percentile(z,97.5):.3f}]" if len(z) else "n/a"
        W(f"  {k:>3}{r_cu:>21.4f}{f(rnd):>28}{f(mat):>28}{d_cu:>24.4f}")
        rows.append(dict(k=k, cu=r_cu, cu_depth=d_cu, n_spot=nsp,
                         random_mean=float(rnd.mean()) if len(rnd) else None,
                         random_lo=float(np.percentile(rnd, 2.5)) if len(rnd) else None,
                         random_hi=float(np.percentile(rnd, 97.5)) if len(rnd) else None,
                         matched_mean=float(mat.mean()) if len(mat) else None,
                         matched_lo=float(np.percentile(mat, 2.5)) if len(mat) else None,
                         matched_hi=float(np.percentile(mat, 97.5)) if len(mat) else None,
                         k_cu=k_cu))
    del X, a
    return dict(tag=tag, n_spot=int(n), k_cu=k_cu, rows=rows)


res = []
for tag, rel in [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                 ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
                 ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
                 ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad")]:
    r = run(tag, rel)
    if r:
        res.append(r)

W("\n" + "=" * 104)
W("判读规则")
W("=" * 104)
W("  若 T-cuTAR 的 ARI 落在 T-random / T-matched 的 95% 区间内 -> 低 ARI 只是特征数/可检出性效应，")
W("     不可声称'cuTAR 域恢复不了结构'是 cuTAR 特有现象。")
W("  若 T-cuTAR 显著低于两者 -> 才可主张 cuTAR 特有的失效。")
json.dump(res, open(os.path.join(ROOT, "results", "RECHECK2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
open(os.path.join(ROOT, "results", "RECHECK2.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\RECHECK2.txt/.json")
