# -*- coding: utf-8 -*-
"""
ladder_control.py — 把 recheck2 的对照扩展到稀释阶梯。
问题：corrected cuTAR 域在稀释下崩塌（ARI 0.01-0.08），而 top-1500 蛋白编码域稳健（0.40-0.88）。
      这差距是"cuTAR 特有"，还是仅仅"特征数少 + 检出率低"？

对照三组（同一切片、同一稀释、同 k）：
  A cuTAR  : 实际过 10% 门槛的 cuTAR
  B random : 随机抽同样数量的蛋白编码基因
  C matched: 按检出率配对的蛋白编码基因
每组算与全深度标签的 ARI，B/C 各重复 20 次取均值与 95% 区间。
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
RNG = np.random.default_rng(11)
OUT = []
W = lambda s="": (print(s, flush=True), OUT.append(s))
NDRAW = 20


def LN(S):
    t = S.sum(1).copy(); t[t == 0] = 1.0
    return np.log1p(S / t[:, None] * 1e4)


def clus(M, k):
    nc = int(min(15, M.shape[1] - 1, M.shape[0] - 1))
    if nc < 2:
        return np.zeros(M.shape[0], int)
    Z = TruncatedSVD(n_components=nc, random_state=0).fit_transform(M)
    return KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3, random_state=0).fit_predict(Z)


gi = {}
with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"), "rt", errors="replace") as f:
    h = f.readline().rstrip("\n").split("\t")
    it, isy = h.index("type_of_gene"), h.index("Symbol")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > max(it, isy):
            gi[p[isy].upper()] = p[it]


def run(tag, rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    is_pc = np.array([gi.get(n.upper(), "") == "protein-coding" for n in names])
    n = X.shape[0]
    det = (X > 0).sum(0) / n
    keep = is_cu & (det >= 0.10)
    cols_cu = np.where(keep)[0]
    if len(cols_cu) < 2:
        W(f"[{tag}] cuTAR 太少，跳过"); return None
    nc = len(cols_cu)

    pc_idx = np.where(is_pc)[0]
    hvg = pc_idx[np.argsort(-np.median(X[:, pc_idx], 0))[:1500]]
    s_ref = np.where((X[:, hvg] > 0).sum(1) > 0)[0]

    pool = []
    for d in det[cols_cu]:
        c = pc_idx[(det[pc_idx] >= 0.80 * d) & (det[pc_idx] <= 1.25 * d)]
        pool.append(c if len(c) else pc_idx[[np.argmin(np.abs(det[pc_idx] - d))]])

    base_mask = LN(X[:, cols_cu]) > 0.5
    K = 4

    def lab_full(cols):
        m = (X[:, cols] > 0).sum(1) > 0
        s = np.where(m)[0]
        return s, (clus(LN(X[s][:, cols]), K) if len(s) >= K + 2 else None)

    s_ref_lab = clus(LN(X[s_ref][:, hvg]), K)
    ref_by_spot = np.full(n, -1)
    ref_by_spot[s_ref] = s_ref_lab

    def lab_dil(cols, frac):
        Xd = RNG.binomial(np.round(X[:, cols]).astype(np.int64), frac).astype(np.float64)
        m = (Xd > 0).sum(1) > 0
        s = np.where(m)[0]
        if len(s) < K + 2:
            return None
        return clus(LN(Xd[s]), K), s

    W("\n" + "=" * 104)
    W(f"[{tag}] n={n:,} spot | cuTAR 过10% = {nc} | 参照 = top-1500 蛋白编码, k={K}")
    W(f"  {'frac':>6}{'A cuTAR':>10}{'B random(20x mean[lo,hi])':>30}{'C matched(20x mean[lo,hi])':>30}"
      f"{'掩码Jaccard':>13}")
    rows = []
    s0, l0 = lab_full(cols_cu)

    def ari_common(s_other, l_other):
        """与全深度结果在共同 spot 上比较 ARI。"""
        if l_other is None or l0 is None:
            return np.nan
        common = np.intersect1d(s0, s_other)
        if len(common) < 20:
            return np.nan
        return float(ari(np.asarray(l0)[np.isin(s0, common)],
                         np.asarray(l_other)[np.isin(s_other, common)]))

    for frac in (0.50, 0.25, 0.10):
        Xd_all = RNG.binomial(np.round(X).astype(np.int64), frac).astype(np.float64)
        mn = LN(Xd_all[:, cols_cu]) > 0.5
        jac = float((mn & base_mask).sum() / max(1, (mn | base_mask).sum()))
        A = lab_dil(cols_cu, frac)
        a = ari_common(A[1], A[0]) if A else np.nan
        B, C = [], []
        for _ in range(NDRAW):
            r = lab_dil(RNG.choice(pc_idx, nc, replace=False), frac)
            if r:
                v = ari_common(r[1], r[0])
                if not np.isnan(v):
                    B.append(v)
            r2 = lab_dil(np.array([RNG.choice(p) for p in pool]), frac)
            if r2:
                v2 = ari_common(r2[1], r2[0])
                if not np.isnan(v2):
                    C.append(v2)
        f = lambda z: (f"{np.mean(z):.3f} [{np.percentile(z,2.5):.3f}, {np.percentile(z,97.5):.3f}]"
                       if len(z) else "n/a")
        W(f"  {frac:>6.2f}{a:>10.4f}{f(B):>30}{f(C):>30}{jac:>13.4f}")
        rows.append(dict(tag=tag, frac=frac, cuTAR=a, random_mean=float(np.mean(B)) if B else None,
                         random_lo=float(np.percentile(B, 2.5)) if B else None,
                         random_hi=float(np.percentile(B, 97.5)) if B else None,
                         matched_mean=float(np.mean(C)) if C else None,
                         matched_lo=float(np.percentile(C, 2.5)) if C else None,
                         matched_hi=float(np.percentile(C, 97.5)) if C else None,
                         mask_jaccard=jac, n_cu=nc))
    # 全深度下三组与参照域的一致性
    W(f"\n  全深度下与参照域（top-1500 蛋白编码）的一致性:")
    def vs_ref(cols):
        s, l = lab_full(cols)
        if l is None:
            return np.nan
        ok = ref_by_spot[s] >= 0
        return float(ari(ref_by_spot[s][ok], l[ok])) if ok.sum() > 20 else np.nan
    vA = vs_ref(cols_cu)
    vB = [vs_ref(RNG.choice(pc_idx, nc, replace=False)) for _ in range(NDRAW)]
    vC = [vs_ref(np.array([RNG.choice(p) for p in pool])) for _ in range(NDRAW)]
    vB = [v for v in vB if not np.isnan(v)]; vC = [v for v in vC if not np.isnan(v)]
    W(f"    A cuTAR = {vA:.4f}")
    W(f"    B random = {np.mean(vB):.4f} [{np.percentile(vB,2.5):.4f}, {np.percentile(vB,97.5):.4f}]")
    W(f"    C matched = {np.mean(vC):.4f} [{np.percentile(vC,2.5):.4f}, {np.percentile(vC,97.5):.4f}]")
    del X, a
    return dict(tag=tag, variants=rows, n_cu=nc,
                vs_ref_cu=vA, vs_ref_random=[float(np.mean(vB)), float(np.percentile(vB, 2.5)), float(np.percentile(vB, 97.5))],
                vs_ref_matched=[float(np.mean(vC)), float(np.percentile(vC, 2.5)), float(np.percentile(vC, 97.5))])


res = []
for tag, rel in [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                 ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
                 ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad")]:
    r = run(tag, rel)
    if r:
        res.append(r)

W("\n" + "=" * 104)
W("判读")
W("=" * 104)
W("  若 A 落在 B/C 的 95% 区间内 -> '低维特征集在稀释下崩塌'是可检出性/维度效应，")
W("     不是 cuTAR 特有；论文必须按可检出性叙述，不能按 cuTAR 身份叙述。")
json.dump(res, open(os.path.join(ROOT, "results", "RECHECK3.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
open(os.path.join(ROOT, "results", "RECHECK3.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\RECHECK3.txt/.json")
