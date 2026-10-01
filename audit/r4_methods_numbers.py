# -*- coding: utf-8 -*-
"""
r4_methods_numbers.py — 为改写 Methods 中「环面平移局限性」一段，产出**可复现**的数字

原稿那段的三个数（0.437 / 0.408±0.064 / 94%）与三组零分布 SD 范围都有问题：
  · 0.437 系列用的是**原始计数**，而全文其余部分用 log1p CP10K；换成后者，位移反而**抬高**自相关。
  · 766 / 85 是 30 次位移的**均值**，不是"一次典型位移"。
  · 三组 SD 范围在 78 个 SCC cuTAR 中**没有一个**能同时复现。
本脚本在统一口径（log1p CP10K、cKDTree 邻接、199 次位移 seed 99）下重算，供直接引用。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np, anndata as ad, scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
K, DET_MIN, TOPN, B = 6, 0.05, 5, 199
MATS = [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]
OUT, JS = [], {}


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "R4_METHODS_NUMBERS.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def lag(nb, Z, k=K, chunk=512):
    n = Z.shape[0]; rows = np.repeat(np.arange(n), k); flat = nb.ravel()
    out = np.zeros_like(Z)
    for s in range(0, Z.shape[1], chunk):
        np.add.at(out[:, s:s + chunk], rows, Z[:, s:s + chunk][flat] / k)
    return out


rng = np.random.default_rng(99)
W("=" * 104)
W("环面平移的简并程度（199 次位移，seed 99，与 SELFCHECK4 同源）")
W("=" * 104)
W(f"  {'矩阵':<15}{'spot':>7}{'唯一目标均值':>13}{'唯一中位':>10}{'唯一范围':>18}"
  f"{'最大重复中位':>13}{'最大重复max':>13}")
deg = {}
scc_data = None
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n = X.shape[0]
    Z = X.copy(); Z.data = (Z.data > 0).astype(float)
    det = np.asarray(Z.sum(0)).ravel() / n
    names = np.array([str(v) for v in a.var.index])
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    coords = np.asarray(a.obsm["spatial"], float)
    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    Xd = X[:, np.r_[icu, ipc]].toarray().astype(float)
    _, it = cKDTree(coords).query(coords, k=K + 1)
    nb = it[:, 1:]

    def cp(V):
        t = V.sum(1); t = np.where(t == 0, 1.0, t)
        return np.log1p(np.divide(V * 1e4, t[:, None]))

    LA = cp(Xd[:, :len(icu)]); LB = cp(Xd[:, len(icu):])
    Zx = LA - LA.mean(0); Zy = LB - LB.mean(0)
    IA = lag(nb, Zx).T @ Zy / np.outer(np.linalg.norm(Zx, axis=0), np.linalg.norm(Zy, axis=0))

    # 位移（与原件同 RNG 流序：位移 -> 分层置换 -> 朴素置换）
    shifts = np.array([rng.uniform(-0.5, 0.5, 2) for _ in range(B)])
    dep = np.asarray(X[:, is_cu].sum(1)).ravel()
    lab = np.digitize(dep, np.quantile(dep, np.linspace(0, 1, 11)[1:-1]))
    P = np.tile(np.arange(n), (B, 1))
    for g in np.unique(lab):
        gi = np.where(lab == g)[0]
        if len(gi) >= 2:
            for b in range(B):
                P[b, gi] = gi[rng.permutation(len(gi))]
    Q = np.array([rng.permutation(n) for _ in range(B)])
    lo = coords.min(0); ext = coords.max(0) - lo; t = cKDTree(coords)
    T = np.array([t.query(lo + np.mod(coords + shifts[s] * ext - lo, ext), k=1)[1]
                  for s in range(B)])
    u = np.array([len(np.unique(T[s])) for s in range(B)])
    m = np.array([np.bincount(T[s], minlength=n).max() for s in range(B)])
    deg[tag] = dict(n=n, uniq_mean=float(u.mean()), uniq_median=int(np.median(u)),
                    uniq_min=int(u.min()), uniq_max=int(u.max()),
                    rep_median=int(np.median(m)), rep_max=int(m.max()), pct_uniq=100 * u.mean() / n)
    W(f"  {tag:<15}{n:>7}{u.mean():>13.0f}{int(np.median(u)):>10}"
      f"{'['+str(u.min())+', '+str(u.max())+']':>18}{int(np.median(m)):>13}{m.max():>13}"
      f"   唯一占比 {100*u.mean()/n:.1f}%")

    # 零分布 SD：全部 cuTAR 的前 5 伙伴
    sd = {k: [] for k in ("torus", "plain", "strat")}
    for i in range(len(icu)):
        order = np.argsort(-IA[i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = lag(nb, cx[:, None])[:, 0]
        CY = LB[:, order] - LB[:, order].mean(0)
        for key, M in (("torus", T), ("plain", Q), ("strat", P)):
            Ys = CY[M]; den = np.linalg.norm(Ys, axis=1)
            sd[key].append(np.std(np.einsum("bnq,n->bq", Ys, Wcx) / (ncx * den), axis=0, ddof=1))
    stat = {k: dict(lo=float(np.min(v)), med=float(np.median(v)), hi=float(np.max(v)),
                    n=len(v)) for k, v in sd.items()}
    JS.setdefault("null_sd_all_pairs", {})[tag] = stat
    W(f"     零分布 SD（该矩阵全部 {len(sd['torus'])} 对）：")
    for k in ("torus", "plain", "strat"):
        W(f"       {k:<6} min {stat[k]['lo']:.4f}  中位 {stat[k]['med']:.4f}  max {stat[k]['hi']:.4f}")

    if tag == "SCC":
        # 单变量 Moran's I：统一用 log1p CP10K 口径（与全文一致）
        j = int(np.argmax(det[icu]))
        z = LA[:, j] - LA[:, j].mean()
        I0 = float(z @ lag(nb, z[:, None])[:, 0] / (z @ z))
        Is = np.array([float((z[T[s]] @ lag(nb, z[T[s]][:, None])[:, 0]) / (z[T[s]] @ z[T[s]]))
                       for s in range(B)])
        scc_data = dict(feature=str(names[icu[j]]), det=float(det[icu[j]]), I0=I0,
                        Is_mean=float(Is.mean()), Is_sd=float(Is.std(ddof=1)),
                        ratio=100 * Is.mean() / I0)
        W(f"\n  SCC 检出率最高的 cuTAR = {scc_data['feature']}（{100*scc_data['det']:.2f}% of spots）")
        W(f"    [统一 log1p CP10K 口径] 未平移 Moran's I = {I0:.4f}；199 次位移后 "
          f"{Is.mean():.4f} ± {Is.std(ddof=1):.4f}  -> 保留 {100*Is.mean()/I0:.1f}%")
        # 原始计数口径（原稿用的）
        xr = Xd[:, j]
        I0r = float((lambda v: v @ lag(nb, v[:, None])[:, 0] / (v @ v))(xr - xr.mean()))
        Isr = np.array([float((lambda v: v @ lag(nb, v[:, None])[:, 0] / (v @ v))(xr[T[s]] - xr[T[s]].mean()))
                        for s in range(B)])
        W(f"    [原稿的原始计数口径]   未平移 {I0r:.4f}；位移后 {Isr.mean():.4f} ± {Isr.std(ddof=1):.4f}"
          f"  -> 保留 {100*Isr.mean()/I0r:.1f}%")
        JS["scc_univariate"] = dict(log=dict(I0=I0, mean=float(Is.mean()), sd=float(Is.std(ddof=1)),
                                             retain=100 * Is.mean() / I0),
                                    raw=dict(I0=I0r, mean=float(Isr.mean()),
                                             sd=float(Isr.std(ddof=1)),
                                             retain=100 * Isr.mean() / I0r),
                                    feature=scc_data["feature"])
    del X, Z, a, Xd, LA, LB

W("\n" + "=" * 104)
W("供 Methods 引用的统一口径汇总")
W("=" * 104)
um = np.mean([deg[t]["uniq_mean"] for t in deg]); umd = np.median([deg[t]["uniq_median"] for t in deg])
W(f"  · 环面平移不是置换：五矩阵中位「唯一目标」占 spot 数 "
  f"{np.median([100*deg[t]['uniq_mean']/deg[t]['n'] for t in deg]):.0f}%；"
  f"SCC 199 次位移中位 {deg['SCC']['uniq_median']}/{deg['SCC']['n']}，范围 "
  f"{deg['SCC']['uniq_min']}–{deg['SCC']['uniq_max']}，最大重复指派最高 {deg['SCC']['rep_max']} 次")
allt = np.concatenate([np.array([JS['null_sd_all_pairs'][t]['torus']['lo'],
                                 JS['null_sd_all_pairs'][t]['torus']['hi']]) for t in deg])
W(f"  · 三种零模型零分布 SD（该矩阵全部配对的 min–max）：")
for k in ("torus", "plain", "strat"):
    lo = np.median([JS['null_sd_all_pairs'][t][k]['lo'] for t in deg])
    hi = np.median([JS['null_sd_all_pairs'][t][k]['hi'] for t in deg])
    W(f"      {k:<6} 中位矩阵 [{lo:.4f}, {hi:.4f}]")
JS["torus_degeneracy"] = deg
json.dump(JS, open(os.path.join(RES, "R4_METHODS_NUMBERS.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\R4_METHODS_NUMBERS.txt / .json")
