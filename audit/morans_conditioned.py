# -*- coding: utf-8 -*-
"""
morans_conditioned.py — 决定性实验：图谱的跨癌种 cuTAR–基因空间关联，是不是检出性驱动的？

框架复现：bivariate Moran's I（Wartenberg），空间权重 = spot 坐标 kNN（k=6）行标准化。
  I_xy = (W·Zx)^T · Zy / (||Zx|| · ||Zy||)

三个变体：
  A 标准：log1p(CP10K) 值           —— 图谱/领域惯用做法
  B 条件化：每个特征的值对 log(spot 深度) 做线性回归后取残差 —— 直接去掉深度混杂
  C 深度匹配：仅用深度高于中位的 spot

四项判决：
  J1  Spearman(I_A, 共检出诊断) —— 关联是否只是"两个特征都测得到"
  J2  A 与 B 的头部伙伴重叠率   —— 去掉深度后，图谱报出的伙伴还在不在
  J3  置换检验：A 显著而 B 不显著的配对数
  J4  匹配零模型：cuTAR 的最佳伙伴强度 vs 检出率匹配的"伪 cuTAR"
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = _cfg.WORKSPACE + r""
RNG = np.random.default_rng(20260923)
K = 6            # Visium 六边形，1 阶邻居
NPERM = 199
DET_MIN = 0.05   # 特征入选门槛：至少 5% spot 检出
TOPN = 5         # 每个 cuTAR 取前 N 个伙伴做置换检验
OUT, RES = [], []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "MORANS_conditioned.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def knn_w(coords, k=K):
    t = cKDTree(coords)
    d, idx = t.query(coords, k=k + 1)
    n = coords.shape[0]
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].ravel()
    w = sp.csr_matrix((np.ones(n * k), (rows, cols)), shape=(n, n))
    w = w.multiply(1.0 / np.asarray(w.sum(1)))       # 行标准化
    return w.tocsr()


def cp10k_log(X):
    t = X.sum(1).copy(); t[t == 0] = 1.0
    return np.log1p(X / t[:, None] * 1e4)


def resid_on_depth(M, dep):
    """对 log(dep) 线性回归取残差（含截距）。M: n×p"""
    d = np.log1p(dep)
    A = np.c_[np.ones(len(d)), d]
    beta, *_ = np.linalg.lstsq(A, M, rcond=None)
    return M - A @ beta


def biv_moran(Wm, Zx, Zy):
    """返回 p×q 的 bivariate Moran's I"""
    WZx = Wm @ Zx
    nx = np.linalg.norm(Zx, axis=0)
    ny = np.linalg.norm(Zy, axis=0)
    return (WZx.T @ Zy) / np.outer(nx, ny)


def run(tag, rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    n = X.shape[0]
    det = (X > 0).mean(0)
    coords = np.asarray(a.obsm["spatial"], dtype=float) if "spatial" in a.obsm else \
        np.c_[a.obs["imagecol"].to_numpy(float), a.obs["imagerow"].to_numpy(float)]

    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    if len(icu) < 3 or len(ipc) < 50:
        W(f"\n[{tag}] cuTAR={len(icu)} 非cuTAR={len(ipc)}，样本不足，跳过"); return None

    Wm = knn_w(coords)
    dep = X[:, is_cu].sum(1)                       # spot 的 cuTAR 子集深度（归一化口径，与判据一致）

    LA = cp10k_log(X[:, icu]); LB = cp10k_log(X[:, ipc])
    RA = cp10k_log(X[:, icu]); RB = cp10k_log(X[:, ipc])
    ZxA = LA - LA.mean(0); ZyA = LB - LB.mean(0)
    IA = biv_moran(Wm, ZxA, ZyA)

    # 变体 B：对 log 深度取残差
    Rx = resid_on_depth(LA, dep); Ry = resid_on_depth(LB, dep)
    ZxB = Rx - Rx.mean(0); ZyB = Ry - Ry.mean(0)
    IB = biv_moran(Wm, ZxB, ZyB)

    # 变体 C：仅深度高于中位的 spot
    hi = np.where(dep > np.median(dep))[0]
    WmC = knn_w(coords[hi])
    ZxC = LA[hi] - LA[hi].mean(0); ZyC = LB[hi] - LB[hi].mean(0)
    IC = biv_moran(WmC, ZxC, ZyC)

    W(f"\n{'='*104}")
    W(f"[{tag}]  {n:,} spot | cuTAR(≥{DET_MIN:.0%}检出) = {len(icu):,} | 非cuTAR(≥{DET_MIN:.0%}) = {len(ipc):,}"
      f" | 深度>中位 spot = {len(hi):,}")
    W(f"  配对总数 = {len(icu)*len(ipc):,}")

    # ---- J1 关联是否只是"都测得到"
    det_c = det[icu][:, None]; det_p = det[ipc][None, :]
    codet = np.zeros((len(icu), len(ipc)))
    Xcu_b = (X[:, icu] > 0).astype(np.float32); Xpc_b = (X[:, ipc] > 0).astype(np.float32)
    codet = (Xcu_b.T @ Xpc_b) / n
    jac = codet / (det_c + det_p - codet)
    def sp_corr(u, v):
        u = u.ravel(); v = v.ravel()
        return float(np.corrcoef(np.argsort(np.argsort(u)), np.argsort(np.argsort(v)))[0, 1])
    r_codet = sp_corr(IA, jac)
    r_mindet = sp_corr(IA, np.minimum(np.broadcast_to(det_c, IA.shape), np.broadcast_to(det_p, IA.shape)))
    r_prod = sp_corr(IA, np.broadcast_to(det_c, IA.shape) * np.broadcast_to(det_p, IA.shape))
    W(f"\n  J1  I_A 与检出性诊断的 Spearman:")
    W(f"        与 检出集 Jaccard      = {r_codet:.3f}")
    W(f"        与 min(两特征检出率)   = {r_mindet:.3f}")
    W(f"        与 检出率乘积          = {r_prod:.3f}")

    # ---- J2 头部伙伴重叠
    def topk_sets(I):
        return [set(np.argsort(-I[i])[:10].tolist()) for i in range(I.shape[0])]
    ta, tb = topk_sets(IA), topk_sets(IB)
    ov = np.array([len(x & y) / 10.0 for x, y in zip(ta, tb)])
    W(f"\n  J2  A 与 B（去深度）的前 10 伙伴重叠率: 均值 {ov.mean():.3f}  中位 {np.median(ov):.3f}")

    # ---- J3 置换检验（只对每个 cuTAR 的 A-前 TOPN 伙伴）
    def perm_p(xc, yv, B=NPERM):
        zx = xc - xc.mean(); nx = np.linalg.norm(zx)
        if nx == 0: return np.nan, np.nan
        Wzx = (Wm @ zx)
        perms = np.array([RNG.permutation(n) for _ in range(B)])
        Yp = yv[perms]                                  # B×n
        ny = np.linalg.norm(yv - yv.mean())
        obs = float(Wzx @ (yv - yv.mean()) / (nx * ny))
        null = (Yp - yv.mean()) @ Wzx / (nx * ny)
        return obs, float((np.abs(null) >= abs(obs)).mean())

    sigA = sigB = both = onlyA = 0
    for i in range(len(icu)):
        order = np.argsort(-IA[i])[:TOPN]
        for j in order:
            oA, pA = perm_p(LA[:, i], LB[:, j])
            oB, pB = perm_p(Rx[:, i], Ry[:, j])
            if pA < 0.05: sigA += 1
            if pB < 0.05: sigB += 1
            if pA < 0.05 and pB < 0.05: both += 1
            if pA < 0.05 and pB >= 0.05: onlyA += 1
    W(f"\n  J3  置换检验（每个 cuTAR 的 A-前 {TOPN} 伙伴，共 {len(icu)*TOPN} 对）:")
    W(f"        A 显著 (p<0.05) = {sigA}   B 显著 = {sigB}   两者皆显著 = {both}   仅 A 显著 = {onlyA}")

    # ---- J4 匹配零模型：cuTAR 的最佳伙伴强度 vs 检出率匹配的伪 cuTAR
    ipc_det = det[ipc]
    best_obs, best_null = [], []
    for i in range(len(icu)):
        best_obs.append(float(np.max(IA[i])))
        d = det[icu][i]
        pool = ipc[np.abs(ipc_det - d) <= 0.25 * d]
        if len(pool) < 5: pool = ipc[[np.argmin(np.abs(ipc_det - d))]]
        j = RNG.choice(pool)
        col = cp10k_log(X[:, [j]])
        z = col[:, 0] - col[:, 0].mean()
        nz = np.linalg.norm(z)
        if nz == 0:
            continue
        Wz = Wm @ z
        Irow = (Wz @ ZyA) / (nz * np.linalg.norm(ZyA, axis=0))
        # 排除自身所在列（伪 cuTAR 是真基因，会与自己相关）
        keep = np.ones(len(ipc), bool); keep[np.where(ipc == j)[0]] = False
        best_null.append(float(np.max(Irow[keep])))
    bo, bn = np.array(best_obs), np.array(best_null)
    W(f"\n  J4  最佳伙伴强度（A 口径）:")
    W(f"        真 cuTAR   : 均值 {bo.mean():.4f}  中位 {np.median(bo):.4f}")
    W(f"        匹配伪cuTAR: 均值 {bn.mean():.4f}  中位 {np.median(bn):.4f}   n={len(bn)}")
    W(f"        差值 = {bo.mean()-bn.mean():+.4f}")

    RES.append(dict(tag=tag, n_spot=int(n), n_cu=int(len(icu)), n_pc=int(len(ipc)),
                    r_jaccard=round(r_codet, 4), r_mindet=round(r_mindet, 4), r_prod=round(r_prod, 4),
                    top_overlap=round(float(ov.mean()), 4),
                    sig_A=sigA, sig_B=sigB, both=both, onlyA=onlyA,
                    best_cu=round(float(bo.mean()), 4), best_null=round(float(bn.mean()), 4)))
    del X, a
    return RES[-1]


def main():
    jobs = [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
            ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
            ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
            ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
            ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]
    for tag, rel in jobs:
        try:
            run(tag, rel)
        except Exception as e:
            import traceback
            W(f"\n[{tag}] FAILED: {type(e).__name__}: {e}"); W(traceback.format_exc())
    W("\n" + "="*104)
    W("机器可读汇总")
    W(json.dumps(RES, ensure_ascii=False, indent=1))
    json.dump(RES, open(os.path.join(ROOT, "results", "MORANS_conditioned.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[written] results\\MORANS_conditioned.txt/.json")


if __name__ == "__main__":
    main()
