# -*- coding: utf-8 -*-
"""
morans_stratified.py — 强化条件化检验（上一版 resid_on_depth 对零膨胀数据太弱）

上一版的变体 B 是"对 log(深度) 取残差"。对零膨胀的对数值，取残差后仍以 0 为主，
**并没有真正去掉检出信息**——所以"关联存活"可能是条件化太弱造成的假象。
本脚本改用**深度分层置换**：只在同一深度十等分位内打乱 y。
这样保持 y 的边际分布与深度结构，只破坏超出深度的关联。

判决：若 p_strat 显著的比例仍高 -> 关联确实超出深度；若塌到接近 0 -> 关联是检出性驱动的。
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
RNG = np.random.default_rng(4242)
K, NPERM, DET_MIN, TOPN = 6, 199, 0.05, 5
OUT, RES = [], []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "MORANS_stratified.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def knn_w(c, k=K):
    n = c.shape[0]
    _, idx = cKDTree(c).query(c, k=k + 1)
    w = sp.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), idx[:, 1:].ravel())), shape=(n, n))
    return w.multiply(1.0 / np.asarray(w.sum(1))).tocsr()


def cp10k_log(X):
    t = X.sum(1).copy(); t[t == 0] = 1.0
    return np.log1p(X / t[:, None] * 1e4)


def dec(v, k=10):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, k + 1)[1:-1]))


def strat_perms(lab, B, n):
    """在 lab 各层内打乱的置换索引矩阵 B×n"""
    groups = [np.where(lab == g)[0] for g in np.unique(lab)]
    P = np.tile(np.arange(n), (B, 1))
    for g in groups:
        if len(g) < 2:
            continue
        for b in range(B):
            P[b, g] = g[RNG.permutation(len(g))]
    return P


def run(tag, rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    n = X.shape[0]
    det = (X > 0).mean(0)
    coords = np.asarray(a.obsm["spatial"], float) if "spatial" in a.obsm else \
        np.c_[a.obs["imagecol"].to_numpy(float), a.obs["imagerow"].to_numpy(float)]
    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    if len(icu) < 3 or len(ipc) < 50:
        W(f"\n[{tag}] 样本不足，跳过"); return None

    Wm = knn_w(coords)
    dep = X[:, is_cu].sum(1)
    lab = dec(dep, 10)

    LA = cp10k_log(X[:, icu]); LB = cp10k_log(X[:, ipc])
    Zx = LA - LA.mean(0); Zy = LB - LB.mean(0)
    WZx = Wm @ Zx
    nx = np.linalg.norm(Zx, axis=0); ny = np.linalg.norm(Zy, axis=0)
    IA = (WZx.T @ Zy) / np.outer(nx, ny)

    # 深度分层置换的零模型（对 y 在层内打乱）
    P = strat_perms(lab, NPERM, n)
    # 对每一列 y 预生成置换后的 Zy 块：B×n×q 过大，改为逐列算
    W(f"\n{'='*104}")
    W(f"[{tag}]  n={n:,} | cuTAR(≥5%)={len(icu)} | 非cuTAR(≥5%)={len(ipc):,} | 深度层数={len(np.unique(lab))}")
    W(f"  {'cuTAR':<16}{'伙伴':<14}{'I_obs':>9}{'p_plain':>9}{'p_strat':>9}")

    sig_plain = sig_strat = tot = 0
    per_cu = []
    for i in range(len(icu)):
        order = np.argsort(-IA[i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean()
        ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = Wm @ cx
        k_i_p = k_i_s = 0
        for j in order:
            y = LB[:, j]; my = y.mean(); cy = y - my; ncy = np.linalg.norm(cy)
            if ncy == 0:
                continue
            obs = float(Wcx @ cy / (ncx * ncy))
            # plain permutation
            Yp = cy[RNG.permutation(n)[None, :].repeat(0, 0)] if False else None
            pl = np.array([cy[RNG.permutation(n)] for _ in range(NPERM)])
            null_p = (pl @ Wcx) / (ncx * ncy)
            p_plain = float((np.abs(null_p) >= abs(obs)).mean())
            # stratified permutation
            Ys = cy[P]                       # B×n
            null_s = (Ys @ Wcx) / (ncx * ncy)
            p_strat = float((np.abs(null_s) >= abs(obs)).mean())
            tot += 1
            if p_plain < 0.05: sig_plain += 1
            if p_strat < 0.05:
                sig_strat += 1
                k_i_s += 1
            if p_plain < 0.05: k_i_p += 1
            if i < 4:
                W(f"  {names[icu[i]][:14]:<16}{names[ipc[j]][:12]:<14}{obs:>9.4f}{p_plain:>9.3f}{p_strat:>9.3f}")
        per_cu.append((k_i_p, k_i_s))
    W(f"  ---- 合计 {tot} 对：plain 显著 {sig_plain} ({100*sig_plain/max(1,tot):.1f}%) | "
      f"**深度分层 显著 {sig_strat} ({100*sig_strat/max(1,tot):.1f}%)**")
    pc = np.array(per_cu)
    W(f"  每个 cuTAR 平均：plain {pc[:,0].mean():.2f}/5 显著 | 分层 {pc[:,1].mean():.2f}/5 显著")
    RES.append(dict(tag=tag, n_cu=int(len(icu)), n_pc=int(len(ipc)), pairs=int(tot),
                    sig_plain=int(sig_plain), sig_strat=int(sig_strat),
                    pct_plain=round(100*sig_plain/max(1,tot), 1),
                    pct_strat=round(100*sig_strat/max(1,tot), 1)))
    del X, a
    return RES[-1]


def main():
    for tag, rel in [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                     ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
                     ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
                     ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
                     ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]:
        try:
            run(tag, rel)
        except Exception as e:
            import traceback
            W(f"\n[{tag}] FAILED {type(e).__name__}: {e}"); W(traceback.format_exc())
    W("\n" + "="*104); W("汇总"); W(json.dumps(RES, ensure_ascii=False, indent=1))
    json.dump(RES, open(os.path.join(ROOT, "results", "MORANS_stratified.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[written] results\\MORANS_stratified.txt/.json")


if __name__ == "__main__":
    main()
