# -*- coding: utf-8 -*-
"""
diag_r4_followup.py — 追查 VERIFY_R4_INDEPENDENT 里 12 项未通过的**成因**

四件事：
 1. 环面平移的最近邻：暴力 argmin 与 cKDTree 差多少行？（并列与否）
 2. 决定性实验：把**近邻选择**换回原件的 cKDTree（其余仍用本脚本的独立实现：
    np.add.at 散射累加、双重求和定义、独立 RNG 复刻），看 Part B 的存活率是否与
    SELFCHECK4.json 精确相同。若相同，则本脚本的独立实现**正确**，
    此前的不一致只来自 k=6 近邻在并列处的取舍。
 3. 单变量 Moran's I：原件用的是**原始计数** x（不是 log1p CP10K），且只有 30 次位移、
    seed 7。按该口径重算，核对 Methods 的 0.437 / 0.408±0.064 / 766 / 85。
 4. 在 SCC 的 78 个 cuTAR 中搜索，看是否存在某个 cuTAR 的前 5 伙伴使三种零分布的 SD
    范围落在稿中声明的 0.070–0.076 / 0.016–0.017 / 0.009–0.012。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
K, DET_MIN, TOPN, B_SMALL, ALPHA = 6, 0.05, 5, 199, 0.05
MATS = [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]
MATS_ORDER = [t for t, _ in MATS]
OUT, JS = [], {}


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "DIAG_R4_FOLLOWUP.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def lag_edges(nb, Z, k=K, chunk=512):
    n = Z.shape[0]
    rows = np.repeat(np.arange(n), k); flat = nb.ravel()
    out = np.zeros_like(Z)
    for s in range(0, Z.shape[1], chunk):
        np.add.at(out[:, s:s + chunk], rows, Z[:, s:s + chunk][flat] / k)
    return out


def torus_kdtree(coords, shifts):
    lo = coords.min(0); ext = coords.max(0) - lo
    t = cKDTree(coords); n = len(coords); B = len(shifts)
    out = np.empty((B, n), np.int64)
    for s in range(B):
        _, j = t.query(lo + np.mod(coords + shifts[s] * ext - lo, ext), k=1)
        out[s] = j
    return out


def torus_brute(coords, shifts, chunk=400):
    lo = coords.min(0); ext = coords.max(0) - lo
    n = len(coords); B = len(shifts)
    out = np.empty((B, n), np.int64)
    for s in range(B):
        tgt = lo + np.mod(coords + shifts[s] * ext - lo, ext)
        b = np.empty(n, np.int64)
        for c in range(0, n, chunk):
            b[c:c + chunk] = cdist(tgt[c:c + chunk], coords).argmin(1)
        out[s] = b
    return out


def cp10k_log_dense(X):
    t = X.sum(1); t = np.where(t == 0, 1.0, t)
    return np.log1p(np.divide(X * 1e4, t[:, None]))


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# ---------------------------------------------------------------- 载入
D = {}
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n, G = X.shape
    Z = X.copy(); Z.data = (Z.data > 0).astype(np.float64)
    det = np.asarray(Z.sum(0)).ravel() / n
    names = np.array([str(v) for v in a.var.index])
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    coords = np.asarray(a.obsm["spatial"], float)
    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    cols = np.r_[icu, ipc]
    Xd = X[:, cols].toarray().astype(np.float64)
    _, it = cKDTree(coords).query(coords, k=K + 1)
    _Dc = cdist(coords, coords)
    np.fill_diagonal(_Dc, np.inf)
    d = dict(tag=tag, n=n, names=names, icu=icu, ipc=ipc, det=det, coords=coords,
             Xd=Xd, ncu=len(icu), npc=len(ipc), nb_tree=it[:, 1:],
             dep_all=np.asarray(X[:, is_cu].sum(1)).ravel(),   # 关键：全部 cuTAR 列之和（原件口径）
             nb_brute=np.argpartition(_Dc, K - 1, axis=1)[:, :K])
    del _Dc
    d["LA"] = cp10k_log_dense(Xd[:, :d["ncu"]])
    d["LB"] = cp10k_log_dense(Xd[:, d["ncu"]:])
    D[tag] = d
    del X, Z, a

# ---------------------------------------------------------------- 1. 环面平移 NN
W("=" * 104); W("追查 1：环面平移的最近邻，暴力 argmin 与 cKDTree 是否一致")
W("=" * 104)
rng = np.random.default_rng(99)
for tag, rel in MATS:
    d = D[tag]
    sh = np.array([rng.uniform(-0.5, 0.5, 2) for _ in range(B_SMALL)])
    d["shifts"] = sh
    # 其余矩阵继续消费 RNG 流，保证 SCC 之后的矩阵位移与原件同源
    dep = d["dep_all"]                       # 原件口径：X[:, is_cu].sum(1)，全部 cuTAR 列
    lab = np.digitize(dep, np.quantile(dep, np.linspace(0, 1, 11)[1:-1]))
    P = np.tile(np.arange(d["n"]), (B_SMALL, 1))
    for g in np.unique(lab):
        gi = np.where(lab == g)[0]
        if len(gi) >= 2:
            for b in range(B_SMALL):
                P[b, gi] = gi[rng.permutation(len(gi))]
    d["P"] = P
    Q = np.array([rng.permutation(d["n"]) for _ in range(B_SMALL)])
    d["Q"] = Q
    d["lab"] = lab
    d["T_brute"] = torus_brute(d["coords"], sh)
    d["T_tree"] = torus_kdtree(d["coords"], sh)
    nd = int((d["T_brute"] != d["T_tree"]).sum())
    rows = int((d["T_brute"] != d["T_tree"]).any(1).sum())
    W(f"  {tag:<13} 平移索引不一致的条目 {nd} / {d['T_brute'].size}"
      f"（{100*nd/d['T_brute'].size:.4f}%），涉及 {rows}/{B_SMALL} 次位移")
    JS.setdefault("torus_nn_mismatch", {})[tag] = dict(entries=nd, shifts=rows)

# ---------------------------------------------------------------- 2. 决定性实验
W("\n" + "=" * 104)
W("追查 2：仅把近邻选择换回原件的 cKDTree，其余保持独立实现，重算三种零模型存活率")
W("=" * 104)
json4 = {x["tag"]: x for x in json.load(open(os.path.join(RES, "SELFCHECK4.json"), encoding="utf-8"))}
res = {}
for tag, rel in MATS:
    d = D[tag]
    sym = d["nb_tree"]
    for nbname, nb, T in (("tree/tree", d["nb_tree"], d["T_tree"]),
                          ("brute/brute", d["nb_brute"], d["T_brute"])):
        IA = (lag_edges(nb, d["LA"] - d["LA"].mean(0)).T @ (d["LB"] - d["LB"].mean(0))) / np.outer(
            np.linalg.norm(d["LA"] - d["LA"].mean(0), axis=0),
            np.linalg.norm(d["LB"] - d["LB"].mean(0), axis=0))
        sig = dict(plain=0, strat=0, torus=0)
        for i in range(d["ncu"]):
            order = np.argsort(-IA[i])[:TOPN]
            cx = d["LA"][:, i] - d["LA"][:, i].mean(); ncx = np.linalg.norm(cx)
            if ncx == 0:
                continue
            Wcx = lag_edges(nb, cx[:, None])[:, 0]
            CY = d["LB"][:, order] - d["LB"][:, order].mean(0)
            obs = (Wcx @ CY) / (ncx * np.linalg.norm(CY, axis=0))
            for key, M in (("plain", d["Q"]), ("torus", T)):
                Ys = CY[M]; den = np.linalg.norm(Ys, axis=1)
                num = np.einsum("bnq,n->bq", Ys, Wcx)
                p = (np.abs(num / (ncx * den)) >= np.abs(obs)[None, :]).mean(0)
                sig[key] += int((p < ALPHA).sum())
        npair = d["ncu"] * TOPN
        res[(tag, nbname)] = dict(pairs=npair, torus=sig["torus"],
                                  pct_torus=round(100 * sig["torus"] / npair, 1),
                                  pct_plain=round(100 * sig["plain"] / npair, 1))
        mark = "== json" if (nbname == "tree/tree" and
                             res[(tag, nbname)]["pct_torus"] == json4[tag]["pct_torus"]) else ""
        W(f"  {tag:<13} {nbname:<12} 环面 {sig['torus']:>4}/{npair} "
          f"({res[(tag,nbname)]['pct_torus']:5.1f}%)   plain {res[(tag,nbname)]['pct_plain']:5.1f}%"
          f"   [json 环面 {json4[tag]['pct_torus']:5.1f}%] {mark}")
four = ["SCC", "HNC", "KidneyCancer", "BCC"]
for nbname in ("tree/tree", "brute/brute"):
    t4 = sum(res[(t, nbname)]["pairs"] for t in four)
    s4 = sum(res[(t, nbname)]["torus"] for t in four)
    lo, hi = wilson(s4, t4)
    W(f"  -> {nbname:<12} 四矩阵 {s4}/{t4} = {100*s4/t4:.2f}%  95%CI=[{100*lo:.2f}, {100*hi:.2f}]"
      f"   Melanoma {res[('Melanoma',nbname)]['torus']}/70 = "
      f"{res[('Melanoma',nbname)]['pct_torus']:.1f}%")
    JS.setdefault("partB_nb_variant", {})[nbname] = dict(
        four=dict(sig=s4, pairs=t4, pct=round(100 * s4 / t4, 2),
                  ci=[round(100 * lo, 2), round(100 * hi, 2)]),
        per_matrix={t: res[(t, nbname)]["pct_torus"] for t in MATS_ORDER})

# ---------------------------------------------------------------- 3. 单变量 Moran's I
W("\n" + "=" * 104)
W("追查 3：单变量 Moran's I —— 原件口径是**原始计数**、30 次位移、seed 7")
W("=" * 104)
scc = D["SCC"]
det_cu = scc["det"][scc["icu"]]
j = int(np.argmax(det_cu))
cu_name = str(scc["names"][scc["icu"][j]])
x_raw = scc["Xd"][:, j]                     # 原始计数
x_log = scc["LA"][:, j]
W(f"  SCC 检出率最高的 cuTAR = {cu_name}（{100*det_cu[j]:.2f}% of spots）")
lo = scc["coords"].min(0); ext = scc["coords"].max(0) - lo
t = cKDTree(scc["coords"]); n = scc["n"]


def moran_raw(v):
    v = v - v.mean()
    return float((v @ (lag_edges(scc["nb_tree"], v[:, None])[:, 0])) / (v @ v))


rng7 = np.random.default_rng(7)
I30, u30, m30 = [], [], []
for _ in range(30):
    T = rng7.uniform(-0.5, 0.5, 2)
    _, j2 = t.query(lo + np.mod(scc["coords"] + T * ext - lo, ext), k=1)
    u30.append(len(np.unique(j2))); m30.append(int(np.bincount(j2, minlength=n).max()))
    I30.append(moran_raw(x_raw[j2]))
u30 = np.array(u30); m30 = np.array(m30); I30 = np.array(I30)
I0_raw = moran_raw(x_raw)
W(f"  [原始计数, 30 次位移 seed 7] 未平移 Moran's I = {I0_raw:.4f}")
W(f"      平移后 {I30.mean():.4f} ± {I30.std():.4f}   唯一目标均值 {u30.mean():.0f}/{n}"
  f"   最大重复均值 {m30.mean():.0f}")
W(f"      -> 与 AUDIT2_diagnose.txt 的 0.4365 / 0.4078±0.0640 / 766 / 85 对照")
rng99 = np.random.default_rng(99)
I199, u199, m199 = [], [], []
for s in range(B_SMALL):
    _, j2 = t.query(lo + np.mod(scc["coords"] + scc["shifts"][s] * ext - lo, ext), k=1)
    u199.append(len(np.unique(j2))); m199.append(int(np.bincount(j2, minlength=n).max()))
    I199.append(moran_raw(x_raw[j2]))
u199 = np.array(u199); m199 = np.array(m199); I199 = np.array(I199)
W(f"  [原始计数, 199 次位移 seed 99] 未平移 {I0_raw:.4f}；平移后 {I199.mean():.4f} ± {I199.std():.4f}")
W(f"      唯一目标：均值 {u199.mean():.0f}  中位 {int(np.median(u199))}  范围 [{u199.min()}, {u199.max()}]")
W(f"      最大重复：均值 {m199.mean():.1f}  中位 {int(np.median(m199))}  范围 [{m199.min()}, {m199.max()}]")
W(f"      -> 稿中「94% 保留」实为 {100*I199.mean()/I0_raw:.1f}%（30 次口径 {100*I30.mean()/I0_raw:.1f}%）")
# log 值口径对照
Ilog = np.array([moran_raw(x_log[j2]) for j2 in
                 [t.query(lo + np.mod(scc["coords"] + scc["shifts"][s] * ext - lo, ext), k=1)[1]
                  for s in range(B_SMALL)]])
W(f"  [log1p CP10K, 199 次位移] 未平移 {moran_raw(x_log):.4f}；平移后 {Ilog.mean():.4f} ± {Ilog.std(ddof=1):.4f}"
  f"  -> 口径不同，结论方向都不同")
JS["univariate"] = dict(feature=cu_name, det=float(det_cu[j]),
                        raw_I0=float(I0_raw), raw_I30=[float(I30.mean()), float(I30.std())],
                        raw_I199=[float(I199.mean()), float(I199.std(ddof=1))],
                        uniq_mean199=float(u199.mean()), maxrep_mean199=float(m199.mean()),
                        log_I0=float(moran_raw(x_log)), log_I199=[float(Ilog.mean()), float(Ilog.std(ddof=1))])

# ---------------------------------------------------------------- 4. 搜索 SD 范围
W("\n" + "=" * 104)
W("追查 4：稿中三组零分布 SD 范围（0.070–0.076 / 0.016–0.017 / 0.009–0.012）能否在某个 cuTAR 上复现")
W("=" * 104)
LA, LB = scc["LA"], scc["LB"]
IA = (lag_edges(scc["nb_tree"], LA - LA.mean(0)).T @ (LB - LB.mean(0))) / np.outer(
    np.linalg.norm(LA - LA.mean(0), axis=0), np.linalg.norm(LB - LB.mean(0), axis=0))
P = scc["P"]
best = []
W(f"  {'cuTAR':<16}{'det%':>7}   torusSD                plainSD                stratSD")
for i in range(scc["ncu"]):
    order = np.argsort(-IA[i])[:TOPN]
    cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
    if ncx == 0:
        continue
    Wcx = lag_edges(scc["nb_tree"], cx[:, None])[:, 0]
    CY = LB[:, order] - LB[:, order].mean(0)
    sds = {}
    for key, M in (("torus", scc["T_tree"]), ("plain", scc["Q"]), ("strat", P)):
        Ys = CY[M]; den = np.linalg.norm(Ys, axis=1)
        sds[key] = np.std(np.einsum("bnq,n->bq", Ys, Wcx) / (ncx * den), axis=0, ddof=1)
    ok = (sds["torus"].min() >= 0.0695 and sds["torus"].max() <= 0.0765 and
          sds["plain"].min() >= 0.0155 and sds["plain"].max() <= 0.0175 and
          sds["strat"].min() >= 0.0085 and sds["strat"].max() <= 0.0125)
    if ok:
        best.append((str(scc["names"][scc["icu"][i]]), sds))
    if i < 6:
        W(f"  {str(scc['names'][scc['icu'][i]])[:15]:<16}{100*det_cu[i]:>7.2f}   "
          f"[{sds['torus'].min():.4f},{sds['torus'].max():.4f}]   "
          f"[{sds['plain'].min():.4f},{sds['plain'].max():.4f}]   "
          f"[{sds['strat'].min():.4f},{sds['strat'].max():.4f}]")
W(f"\n  满足稿中三组范围的 cuTAR 数 = {len(best)} / {scc['ncu']}"
  + (f"  例：{best[0][0]}" if best else ""))
# 逐个 cuTAR 的 torus SD 上界分布
tmax = []
for i in range(scc["ncu"]):
    order = np.argsort(-IA[i])[:TOPN]
    cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
    if ncx == 0:
        continue
    Wcx = lag_edges(scc["nb_tree"], cx[:, None])[:, 0]
    CY = LB[:, order] - LB[:, order].mean(0)
    Ys = CY[scc["T_tree"]]; den = np.linalg.norm(Ys, axis=1)
    tmax.append(np.std(np.einsum("bnq,n->bq", Ys, Wcx) / (ncx * den), axis=0, ddof=1).max())
tmax = np.array(tmax)
W(f"  全部 {len(tmax)} 个 cuTAR 的「前5伙伴 torus SD 中最大值」："
  f"最小 {tmax.min():.4f}  中位 {np.median(tmax):.4f}  最大 {tmax.max():.4f}")
W(f"  其中落在 0.070–0.076 的 cuTAR 数 = {int(((tmax>=0.0695)&(tmax<=0.0765)).sum())}")
JS["sd_search"] = dict(n_match=len(best), tmax=[float(tmax.min()), float(np.median(tmax)), float(tmax.max())],
                       n_tmax_in_range=int(((tmax >= 0.0695) & (tmax <= 0.0765)).sum()))

W("\n" + "=" * 104)
open(os.path.join(RES, "DIAG_R4_FOLLOWUP.txt"), "w", encoding="utf-8").write("\n".join(OUT))
json.dump(JS, open(os.path.join(RES, "DIAG_R4_FOLLOWUP.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print("\n[written] results\\DIAG_R4_FOLLOWUP.txt / .json")
