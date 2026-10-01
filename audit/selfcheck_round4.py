# -*- coding: utf-8 -*-
"""
selfcheck_round4.py — 对第四轮实验的对抗性自查

怀疑点1【方法论】: 我用的零模型是"置换 y"（plain）与"深度分层内置换 y"（stratified）。
   两者**都把 y 自身的空间自相关打掉了**。于是**任何两个各自空间成片的特征**都会显示"显著"关联，
   即使它们彼此独立。这正是我批评旧论文时指出的同一类错误——零模型没有条件化在正确的量上。
   正确做法：**环面平移（toroidal shift）**——保留 y 的全部空间结构，只在空间上整体位移。
   若关联在环面平移下消失，则"连续统计稳健"的结论不成立。

怀疑点2【算术】: 我把 A vs B 的 top-10 伙伴重叠的随机基线写成 k/N。
   正确的期望是 k²/N。**我低估了 10 倍**，"高于随机 30-400 倍"应为"约 2-38 倍"。

怀疑点3【概念】: "二值化丢失信息"本身是平凡的。真正非平凡的是"0.5 这个截断在这些深度下
   根本不是表达量阈值"。标题与摘要的重心应回到**恒等**，而非"阈值 vs 连续"的对比。
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
RNG = np.random.default_rng(99)
K, NSHIFT, DET_MIN, TOPN = 6, 199, 0.05, 5
OUT, RES = [], []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "SELFCHECK4.txt"), "w", encoding="utf-8").write("\n".join(OUT))


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


def torus_idx(coords, B):
    """B 个环面平移索引：把坐标整体位移后取最近邻 spot。保留 y 的空间结构。"""
    lo = coords.min(0); ext = coords.max(0) - lo
    t = cKDTree(coords)
    out = []
    for _ in range(B):
        d = RNG.uniform(-0.5, 0.5, 2) * ext          # 位移量
        tgt = lo + np.mod(coords + d - lo, ext)
        _, idx = t.query(tgt, k=1)
        out.append(idx)
    return np.array(out)


def strat_idx(lab, B, n):
    P = np.tile(np.arange(n), (B, 1))
    for g in np.unique(lab):
        gi = np.where(lab == g)[0]
        if len(gi) < 2:
            continue
        for b in range(B):
            P[b, gi] = gi[RNG.permutation(len(gi))]
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

    T = torus_idx(coords, NSHIFT)      # 环面平移索引
    P = strat_idx(lab, NSHIFT, n)      # 深度分层置换索引
    Q = np.array([RNG.permutation(n) for _ in range(NSHIFT)])   # 朴素置换索引

    # ---- 修正 top-10 重叠的随机基线：k^2/N（不是我原先写的 k/N）
    def topk(I):
        return [set(np.argsort(-I[i])[:10].tolist()) for i in range(I.shape[0])]
    W(f"\n{'='*104}")
    W(f"[{tag}]  n={n:,} | cuTAR={len(icu)} | 非cuTAR={len(ipc):,}")

    W(f"\n  【怀疑2】top-10 伙伴重叠的随机基线修正")
    W(f"    非cuTAR 特征数 N = {len(ipc):,}   正确基线 = 10²/N = {100/len(ipc):.4f}"
      f"   （我原先误写为 10/N = {10/len(ipc):.4f}，低估 10 倍）")

    W(f"\n  【怀疑1】三种零模型对比：plain / 深度分层 / **环面平移**")
    W(f"  {'cuTAR':<15}{'伙伴':<13}{'I_obs':>8}{'p_plain':>9}{'p_strat':>9}{'p_torus':>9}")
    sp_ = ss_ = st_ = tot = 0
    for i in range(len(icu)):
        order = np.argsort(-IA[i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = Wm @ cx
        for j in order:
            y = LB[:, j]; cy = y - y.mean(); ncy = np.linalg.norm(cy)
            if ncy == 0:
                continue
            obs = float(Wcx @ cy / (ncx * ncy))
            def pv(idxmat):
                Y = cy[idxmat]
                den = np.linalg.norm(Y, axis=1)
                nu = Y @ Wcx
                ok = den > 0
                null = np.zeros(len(Y))
                null[ok] = nu[ok] / (ncx * den[ok])
                return float((np.abs(null) >= abs(obs)).mean())
            pp, ps, pt = pv(Q), pv(P), pv(T)
            tot += 1
            if pp < 0.05: sp_ += 1
            if ps < 0.05: ss_ += 1
            if pt < 0.05: st_ += 1
            if i < 3:
                W(f"  {names[icu[i]][:13]:<15}{names[ipc[j]][:11]:<13}{obs:>8.4f}{pp:>9.3f}{ps:>9.3f}{pt:>9.3f}")
    W(f"  ---- 合计 {tot} 对")
    W(f"       plain 置换显著        : {sp_:>4} ({100*sp_/tot:5.1f}%)")
    W(f"       深度分层置换显著      : {ss_:>4} ({100*ss_/tot:5.1f}%)")
    W(f"       **环面平移显著**      : {st_:>4} ({100*st_/tot:5.1f}%)   <- 保留 y 空间自相关的严格零模型")

    RES.append(dict(tag=tag, pairs=tot, pct_plain=round(100*sp_/tot, 1),
                    pct_strat=round(100*ss_/tot, 1), pct_torus=round(100*st_/tot, 1),
                    n_pc=int(len(ipc)), chance_overlap=round(100/len(ipc), 4)))
    del X, a
    return RES[-1]


def main():
    W("=" * 104)
    W("第四轮实验对抗性自查")
    W("=" * 104)
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
    W("\n" + "=" * 104); W("汇总"); W(json.dumps(RES, ensure_ascii=False, indent=1))
    json.dump(RES, open(os.path.join(ROOT, "results", "SELFCHECK4.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n[written] results\\SELFCHECK4.txt/.json")


if __name__ == "__main__":
    main()
