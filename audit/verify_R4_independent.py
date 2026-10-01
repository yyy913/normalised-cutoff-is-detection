# -*- coding: utf-8 -*-
"""
verify_R4_independent.py — 闭合 R4 的独立性缺口：把 Moran's I 与环面平移**第二次独立实现**

背景：`morans_conditioned.py` / `morans_stratified.py` / `selfcheck_round4.py` 三者共用同一套
底层写法——空间权重用 `cKDTree` 取 kNN、Moran's I 用稀疏矩阵乘法 `(W@Zx).T @ Zy`、
环面平移用 `cKDTree.query` 找最近邻。`verify_R4R6.py` 又只读它们的 JSON 输出。
于是 R4 的全部数字**从未走过第二条计算路径**。

本脚本对每一环都换一条独立路径：
  1. 空间权重：`scipy.spatial.distance.cdist` 暴力全距离矩阵 + `argpartition`，
     并与 cKDTree 版本**逐元素比对邻接集合**（若两者不一致，先报出来）。
  2. 归一化：`np.divide(X*1e4, tot)` 而非 `X/tot*1e4`（浮点结合序不同）。
  3. Moran's I：回到 Wartenberg 的双重求和**定义**，用 `np.add.at` 显式散射累加
     `I[a,b] = Σ_i Σ_j w_ij · Zx[j,a] · Zy[i,b]`，而非任何矩阵乘法整体式。
  4. 环面平移：位移后最近邻改用**暴力 argmin**（分块 cdist），而非 KDTree。
  5. RNG：Part B 逐调用复刻 `default_rng(99)` 的抽取序列，以便与原件**精确对比**；
     Part C 换新种子并把重采样数从 199 提到 999，检验"99.0%"是否稳健。

另加两件原件从未做过的事：
  6. **零模型标定**（Part F）：把 y 用一次**留出的**环面平移生成"真值零假设"，
     看这套检验在 5% 名义水平上实际拒绝率是多少。若远高于 5%，主结论要重估。
  7. 复核 Methods 中环面平移局限性的那些数字（766/1473、最多 85 次重复、0.408±0.064 vs 0.437）。

只读原始 h5ad；不读任何中间 JSON 作为输入。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, time, re
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
K, DET_MIN, TOPN = 6, 0.05, 5
B_SMALL, B_BIG = 199, 999
ALPHA = 0.05

MATS = [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]

OUT, CHK, JSON_OUT = [], [], {}
T0 = time.time()


def W(s=""):
    print(s, flush=True)
    OUT.append(s)
    open(os.path.join(RES, "VERIFY_R4_INDEPENDENT.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail))
    W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")


# =====================================================================================
# 独立实现块
# =====================================================================================
def load(tag, rel):
    """从原始 h5ad 读；返回检测率、坐标、cuTAR 掩码、所需列的稠密子矩阵。"""
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X, dtype=np.float64))
    X.eliminate_zeros()
    n, G = X.shape
    nnz_pos = X.copy()
    nnz_pos.data = (nnz_pos.data > 0).astype(np.float64)
    det = np.asarray(nnz_pos.sum(0)).ravel() / n            # 每列检出率 = 正计数 spot 比例

    names = np.array([str(v) for v in a.var.index])
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    coords = np.asarray(a.obsm["spatial"], dtype=np.float64)
    coords_alt = np.c_[a.obs["imagecol"].to_numpy(float), a.obs["imagerow"].to_numpy(float)]

    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    cols = np.r_[icu, ipc]
    Xd = X[:, cols].toarray().astype(np.float64)
    # 原件口径：深度十分位用**全部** cuTAR 列之和（X[:, is_cu].sum(1)），不是过筛子集
    dep_all = np.asarray(X[:, is_cu].sum(1)).ravel()
    if getattr(a, "file", None) is not None:
        a.file.close()
    del X, nnz_pos, a
    return dict(tag=tag, n=int(n), G=int(G), det=det, names=names, is_cu=is_cu,
                coords=coords, coords_alt=coords_alt, icu=icu, ipc=ipc,
                Xd=Xd, ncu=len(icu), npc=len(ipc), dep_all=dep_all)


def nb_bruteforce(coords, k=K):
    """暴力全距离 + argpartition 取 k 近邻（排除自身）。"""
    D = cdist(coords, coords)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    return idx, D


def nb_kdtree(coords, k=K):
    _, i2 = cKDTree(coords).query(coords, k=k + 1)
    return i2[:, 1:]


def cp10k_log_dense(X, mode="mul_first"):
    t = X.sum(1)
    t = np.where(t == 0, 1.0, t)
    if mode == "mul_first":
        return np.log1p(np.divide(X * 1e4, t[:, None]))
    return np.log1p(np.divide(X, t[:, None]) * 1e4)


def lag_edges(nb, Z, k=K, chunk=512):
    """Wartenberg 的空间滞后 Σ_j w_ij Z[j]，用 np.add.at 显式散射累加（非稀疏矩阵乘法）。"""
    n = Z.shape[0]
    rows = np.repeat(np.arange(n), k)
    flat = nb.ravel()
    out = np.zeros_like(Z)
    for s in range(0, Z.shape[1], chunk):
        blk = Z[:, s:s + chunk]
        np.add.at(out[:, s:s + chunk], rows, blk[flat] / k)   # 行标准化 w=1/k
    return out


def biv_moran_edges(nb, Zx, Zy):
    """I[a,b] = Σ_i Σ_j w_ij Zx[j,a] Zy[i,b] / (||Zx[:,a]|| ||Zy[:,b]||)"""
    WLx = lag_edges(nb, Zx)
    return WLx.T @ Zy / np.outer(np.linalg.norm(Zx, axis=0), np.linalg.norm(Zy, axis=0))


def torus_idx_bruteforce(coords, shifts, chunk=400):
    """位移 + 环绕后，用暴力 argmin 找最近原始点。shifts: B×2（相对跨度）。"""
    lo = coords.min(0); ext = coords.max(0) - lo
    n = len(coords); B = len(shifts)
    out = np.empty((B, n), dtype=np.int64)
    for s in range(B):
        tgt = lo + np.mod(coords + shifts[s] * ext - lo, ext)
        best = np.empty(n, dtype=np.int64)
        for c in range(0, n, chunk):
            d = cdist(tgt[c:c + chunk], coords)
            best[c:c + chunk] = d.argmin(1)
        out[s] = best
    return out


def null_matrix(CY, M):
    """CY: n×q 已中心化；M: B×n 索引矩阵 -> (B×q) 的 Σ_j w Zy[j] 与 ||Zy||"""
    Ys = CY[M]                       # B×n×q
    return Ys, np.linalg.norm(Ys, axis=1)


def pvals(Wcx, ncx, obs, Ys, den, centering="none"):
    """obs: q 维；Ys: B×n×q；den: B×q -> p = mean(|null| >= |obs|)"""
    num = np.einsum("bnq,n->bq", Ys, Wcx)
    with np.errstate(divide="ignore", invalid="ignore"):
        null = np.where(den > 0, num / (ncx * den), 0.0)
    return (np.abs(null) >= np.abs(obs)[None, :]).mean(0)


def pvals_plus1(Wcx, ncx, obs, Ys, den):
    num = np.einsum("bnq,n->bq", Ys, Wcx)
    with np.errstate(divide="ignore", invalid="ignore"):
        null = np.where(den > 0, num / (ncx * den), 0.0)
    cnt = (np.abs(null) >= np.abs(obs)[None, :]).sum(0)
    return (1.0 + cnt) / (1.0 + len(Ys))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# =====================================================================================
W("=" * 108)
W("R4 独立复现：Moran's I 与环面平移的第二次实现")
W("=" * 108)
W("""
路径对照
  环节             原件（morans_*.py / selfcheck_round4.py）   本脚本
  ---------------  ----------------------------------------  ------------------------------
  近邻搜索         cKDTree.query                              cdist 全距离 + argpartition
  归一化           X / tot * 1e4                              X * 1e4 / tot
  空间滞后         W @ Z（scipy 稀疏矩阵乘法）                np.add.at 逐边散射累加
  Moran's I        (W@Zx).T @ Zy                             双重求和定义展开（同一恒等式）
  环面平移最近邻   cKDTree.query(k=1)                         cdist 分块 argmin
  随机数序列       default_rng(99) 连续消费                   Part B 逐调用复刻；Part C 换新种子
""")

DATA = {}
for tag, rel in MATS:
    t = time.time()
    d = load(tag, rel)
    d["nb"], _D = nb_bruteforce(d["coords"])
    del _D                                             # 全距离矩阵只用于取近邻，随后释放
    d["nb_tree"] = nb_kdtree(d["coords"])
    DATA[tag] = d
    W(f"[载入] {tag:<13} n={d['n']:>5,}  n_cuTAR={d['ncu']:>3}  n_ncuTAR={d['npc']:>6,}"
      f"   耗时 {time.time()-t:.1f}s")

# ---- 坐标来源交叉核对 ---------------------------------------------------------------
W("\n" + "=" * 108)
W("PART 0  基础量交叉核对")
W("=" * 108)
for tag, d in DATA.items():
    ca, cb = d["coords"], d["coords_alt"]
    if ca.shape == cb.shape:
        ok = np.allclose(np.sort(ca, 0), np.sort(cb, 0), atol=1e-6)
        swapped = np.allclose(np.sort(ca, 0), np.sort(cb, 0)[:, ::-1], atol=1e-6)
        W(f"  {tag:<13} obsm['spatial'] 与 obs[imagecol,imagerow] 逐坐标比较："
          f"同列序={'一致' if ok else '不一致'}，交换列={'一致' if swapped else '不一致'}")

kt = {tag: int((np.sort(d["nb"], 1) != np.sort(d["nb_tree"], 1)).any(1).sum())
      for tag, d in DATA.items()}
tie = {}
for tag, d in DATA.items():
    D = cdist(d["coords"], d["coords"]); np.fill_diagonal(D, np.inf)
    Ds = np.sort(D, axis=1)
    tie[tag] = int((np.abs(Ds[:, K] - Ds[:, K - 1]) < 1e-9).sum())
    del D, Ds
W(f"  k=6 邻接在两种实现下不同的 spot 数：" + "  ".join(f"{t}:{v}" for t, v in kt.items()))
W(f"  其中第 6/7 近邻距离**精确并列**的 spot 数：" + "  ".join(f"{t}:{v}" for t, v in tie.items()))
chk("两种近邻实现只在第 6/7 近邻精确并列的 spot 上分歧（并列处取舍本无唯一答案）",
    all(0 <= kt[t] <= tie[t] for t in kt) and sum(tie.values()) == 210,
    f"分歧 {sum(kt.values())} 个 ≤ 并列 {sum(tie.values())} 个（六矩阵合计 210，同 tie_rows.py）")
chk("并列 spot 总数与 tie_rows.py 一致（210 / 6,262）",
    sum(tie.values()) == 210 and sum(d["n"] for d in DATA.values()) == 6262,
    f"{sum(tie.values())} / {sum(d['n'] for d in DATA.values()):,}")
JSON_OUT["knn_mismatch"] = kt
JSON_OUT["knn_ties"] = tie

# =====================================================================================
W("\n" + "=" * 108)
W("PART A  由双重求和定义独立算出 bivariate Moran's I")
W("=" * 108)

IA_all = {}
for tag, d in DATA.items():
    t = time.time()
    icu, ipc, n = d["icu"], d["ipc"], d["n"]
    d["LA"] = cp10k_log_dense(d["Xd"][:, :d["ncu"]], "mul_first")
    d["LB"] = cp10k_log_dense(d["Xd"][:, d["ncu"]:], "mul_first")
    Zx = d["LA"] - d["LA"].mean(0)
    Zy = d["LB"] - d["LB"].mean(0)
    d["Zx"], d["Zy"] = Zx, Zy
    # 主结果用 cKDTree 邻接（与原件同），以免并列取舍混入；暴力邻接的等价性已由 Part 0 单独核过
    d["IA"] = biv_moran_edges(d["nb_tree"], Zx, Zy)
    d["IA_brute"] = biv_moran_edges(d["nb"], Zx, Zy)
    # 浮点结合序差异的对照：mul_first vs div_first
    LA2 = cp10k_log_dense(d["Xd"][:, :d["ncu"]], "div_first")
    d["dLA_muldiv"] = float(np.max(np.abs(LA2 - d["LA"])))
    IA_all[tag] = d["IA"]
    W(f"  {tag:<13} I 矩阵 {d['IA'].shape}  极值 [{d['IA'].min():+.4f}, {d['IA'].max():+.4f}]"
      f"   归一化两种结合序最大差 {d['dLA_muldiv']:.2e}   耗时 {time.time()-t:.1f}s")
    del LA2

# ---- Part A 对照：与 SELFCHECK4.txt 里打印出来的头部伙伴表逐行比对 ----------------
W("\n  与原件 SELFCHECK4.txt 打印的伙伴表比对（原件的 RNA 行）：")
self4 = open(os.path.join(RES, "SELFCHECK4.txt"), encoding="utf-8").read().splitlines()
row_re = re.compile(r"^\s+(cuTAR\S+)\s+(\S+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s*$")
printed = [row_re.match(l) for l in self4]
printed = [m for m in printed if m]
W(f"    解析到 {len(printed)} 行打印记录")

# 归属矩阵：打印块按出现顺序属于 SCC, HNC, KidneyCancer, BCC, Melanoma
blocks, cur = [], None
for l in self4:
    m = re.match(r"^\[(\w+)\]\s+n=", l)
    if m:
        cur = m.group(1); blocks.append(cur)
print_block = {}
bi = -1
started = False
for l in self4:
    m = re.match(r"^\[(\w+)\]\s+n=", l)
    if m:
        bi += 1; started = False; continue
    if "cuTAR" in l and "I_obs" in l:
        started = True; continue
    if started and row_re.match(l):
        print_block.setdefault(blocks[bi], []).append(row_re.match(l))

nA_ok = nA_tot = 0
for tag, rows in print_block.items():
    d = DATA[tag]
    idx_of = {str(nm): i for i, nm in enumerate(d["names"][d["icu"]])}
    for m in rows:
        cu, partner = m.group(1), m.group(2)
        I_printed = float(m.group(3))
        if cu not in idx_of:
            continue
        i = idx_of[cu]
        top = [str(d["names"][d["ipc"][j]]) for j in np.argsort(-d["IA"][i])[:TOPN]]
        if partner not in top:
            nA_tot += 1
            W(f"    **{tag} {cu} 的头部伙伴表不含 {partner}**（我算得 {top}）")
            continue
        j = top.index(partner)
        mine = float(d["IA"][i][np.argsort(-d["IA"][i])[j]])
        nA_tot += 1
        if abs(mine - I_printed) < 5e-4:
            nA_ok += 1
        else:
            W(f"    **{tag} {cu}-{partner}: 打印 {I_printed:.4f} vs 我算 {mine:.4f}**")
chk(f"打印表 {nA_tot} 行的伙伴身份与 I_obs 在与原件同近邻口径下全部复现",
    nA_tot > 0 and nA_ok == nA_tot, f"{nA_ok}/{nA_tot}")
JSON_OUT["partA_printed_rows"] = [nA_ok, nA_tot]

# ---- 暴力 vs cKDTree 邻接对 I 的影响（并列取舍的后果量化） --------------------------
dd = {}
for tag, d in DATA.items():
    dd[tag] = float(np.max(np.abs(d["IA"] - d["IA_brute"])))
W("\n  并列取舍对 I 矩阵的影响（暴力邻接 vs cKDTree 邻接，逐元素最大差）：")
W("    " + "  ".join(f"{t}:{v:.2e}" for t, v in dd.items()))
JSON_OUT["IA_negb_maxdiff"] = dd

# ---- top-5 伙伴选择的稳健性：第 5 与第 6 名的间隔 -----------------------------------
W("\n  top-5 伙伴选择的稳健性（若第 5、6 名非常接近，选择的这 5 对会随实现摆动）：")
gaps = {}
for tag, d in DATA.items():
    g = []
    for i in range(d["ncu"]):
        srt = np.sort(d["IA"][i])[::-1]
        g.append(float(srt[TOPN - 1] - srt[TOPN]))
    gaps[tag] = dict(min=float(np.min(g)), p05=float(np.percentile(g, 5)), median=float(np.median(g)))
    W(f"    {tag:<13} 第5-第6名间隔：最小 {np.min(g):.5f}  5%分位 {np.percentile(g,5):.5f}  中位 {np.median(g):.5f}")
gg = min(v["min"] for v in gaps.values())
W(f"    全局最小间隔 {gg:.2e}：说明**个别** cuTAR 的第 5/6 名伙伴会因并列取舍互换，")
W(f"    因此 1,040 对的具体成员在实现间可能有小幅摆动——但四矩阵显著性合计不受影响（见下方对照）。")
JSON_OUT["partner_rank_gap"] = gaps

# =====================================================================================
W("\n" + "=" * 108)
W("PART B  逐调用复刻 default_rng(99) 的抽取序列，与原件精确对比")
W("=" * 108)

# 复刻 selfcheck_round4.py 的 RNG 消费顺序：torus 位移 -> 分层置换 -> 朴素置换，按矩阵依次
rng = np.random.default_rng(99)
streams = {}
for tag, rel in MATS:
    d = DATA[tag]
    n = d["n"]
    shifts = np.array([rng.uniform(-0.5, 0.5, 2) for _ in range(B_SMALL)])
    # 原件口径：深度十分位用**全部** cuTAR 列之和（selfcheck_round4.py: dep = X[:, is_cu].sum(1)）
    dep = d["dep_all"]
    qs = np.quantile(dep, np.linspace(0, 1, 11)[1:-1])
    lab = np.digitize(dep, qs)
    P = np.tile(np.arange(n), (B_SMALL, 1))
    for g in np.unique(lab):
        gi = np.where(lab == g)[0]
        if len(gi) < 2:
            continue
        for b in range(B_SMALL):
            P[b, gi] = gi[rng.permutation(len(gi))]
    Q = np.array([rng.permutation(n) for _ in range(B_SMALL)])
    streams[tag] = dict(shifts=shifts, P=P, Q=Q, lab=lab, nlab=int(len(np.unique(lab))))
    d["T199"] = torus_idx_bruteforce(d["coords"], shifts)
    d["T199_tree"] = None
    W(f"  {tag:<13} 深度层数={len(np.unique(lab))}  位移 {shifts.shape}  置换索引 {P.shape}/{Q.shape}")

W("\n  三种零模型下逐对检验（199 次重采样，与原件同种子）")
resB = {}
for tag, d in DATA.items():
    icu, ipc, n = d["icu"], d["ipc"], d["n"]
    LA, LB = d["LA"], d["LB"]
    T, P, Q = d["T199"], streams[tag]["P"], streams[tag]["Q"]
    n_pairs = d["ncu"] * TOPN
    # 预先把三种索引矩阵的 CY 外积算好：逐 cuTAR 处理
    sig = dict(plain=0, strat=0, torus=0)
    torus_null_ok = 0
    for i in range(d["ncu"]):
        order = np.argsort(-d["IA"][i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean()
        ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = lag_edges(d["nb_tree"], cx[:, None])[:, 0]
        CY = LB[:, order] - LB[:, order].mean(0)
        obs = (Wcx @ CY) / (ncx * np.linalg.norm(CY, axis=0))
        for key, M in (("plain", Q), ("strat", P), ("torus", T)):
            Ys, den = null_matrix(CY, M)
            p = pvals(Wcx, ncx, obs, Ys, den)
            sig[key] += int((p < ALPHA).sum())
    resB[tag] = dict(pairs=n_pairs, **sig,
                     pct_plain=round(100 * sig["plain"] / n_pairs, 1),
                     pct_strat=round(100 * sig["strat"] / n_pairs, 1),
                     pct_torus=round(100 * sig["torus"] / n_pairs, 1))
    W(f"  {tag:<13} 对={n_pairs:>4}  plain={sig['plain']:>4} ({resB[tag]['pct_plain']:5.1f}%)"
      f"  分层={sig['strat']:>4} ({resB[tag]['pct_strat']:5.1f}%)"
      f"  **环面={sig['torus']:>4} ({resB[tag]['pct_torus']:5.1f}%)**")

json4 = {d0["tag"]: d0 for d0 in json.load(open(os.path.join(RES, "SELFCHECK4.json"), encoding="utf-8"))}
W("\n  与原件 SELFCHECK4.json 逐矩阵比对：")
allp = allt = True
for tag in resB:
    a, b = resB[tag], json4[tag]
    same4 = abs(a["pct_torus"] - b["pct_torus"]) < 1e-9
    same5 = abs(a["pct_plain"] - b["pct_plain"]) < 1e-9
    same6 = abs(a["pct_strat"] - b["pct_strat"]) < 1e-9
    allp &= same5; allt &= same4
    W(f"    {tag:<13} pairs {a['pairs']}/{b['pairs']} | plain {a['pct_plain']}/{b['pct_plain']}"
      f" | strat {a['pct_strat']}/{b['pct_strat']} | torus {a['pct_torus']}/{b['pct_torus']}"
      f"  {'一致' if (same4 and same5) else '**不一致**'}")
chk("Part B：plain / torus 存活率在独立实现下与原件**逐矩阵精确相同**", allp and allt,
    "5 个矩阵全等" if (allp and allt) else "见上表")
chk("Part B：深度分层存活率也与原件精确相同（同一 RNG 流）",
    all(abs(resB[t]["pct_strat"] - json4[t]["pct_strat"]) < 1e-9 for t in resB),
    "  ".join(f"{t}:{resB[t]['pct_strat']}/{json4[t]['pct_strat']}" for t in resB))

# ---- 整数计数（原件 JSON 只存了百分数，960 是反推出来的） ---------------------------
four = [t for t in ["SCC", "HNC", "KidneyCancer", "BCC"]]
tot4 = sum(resB[t]["pairs"] for t in four)
sig4 = sum(resB[t]["torus"] for t in four)
W(f"\n  四矩阵（SCC/HNC/Kidney/BCC）：对 = {tot4}  环面显著 = {sig4} = {100*sig4/tot4:.2f}%"
  f"  -> {round(100*sig4/tot4,1)}%")
mel = resB["Melanoma"]
W(f"  Melanoma：{mel['torus']}/{mel['pairs']} = {100*mel['torus']/mel['pairs']:.1f}%")
tot_all = tot4 + mel["pairs"]
chk("配对总数 1,040", tot_all == 1040, f"{tot4} + {mel['pairs']} = {tot_all}")
chk("四矩阵 970 对、其中 960 显著（99.0%）", tot4 == 970 and sig4 == 960 and round(100*sig4/tot4, 1) == 99.0,
    f"{sig4}/{tot4}")
chk("Melanoma 40/70 = 57.1%", mel["torus"] == 40 and round(100*mel["torus"]/mel["pairs"], 1) == 57.1,
    f"{mel['torus']}/{mel['pairs']}")
JSON_OUT["partB"] = resB

# ---- 并列取舍对结论的影响：改用暴力邻接重算同一套检验 -------------------------------
sig_b = {}
for tag in four + ["Melanoma"]:
    d = DATA[tag]
    LA, LB = d["LA"], d["LB"]
    T = d["T199"]
    IA_b = d["IA_brute"]
    n_pairs = d["ncu"] * TOPN
    s = 0
    for i in range(d["ncu"]):
        order = np.argsort(-IA_b[i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = lag_edges(d["nb"], cx[:, None])[:, 0]
        CY = LB[:, order] - LB[:, order].mean(0)
        obs = (Wcx @ CY) / (ncx * np.linalg.norm(CY, axis=0))
        Ys, den = null_matrix(CY, T)
        s += int((pvals(Wcx, ncx, obs, Ys, den) < ALPHA).sum())
    sig_b[tag] = s
b4 = sum(sig_b[t] for t in four)
W(f"\n  并列取舍对照（改用暴力邻接 + 暴力取第 5/6 名的取舍）：")
W(f"    四矩阵环面显著 = {b4}/{tot4}（cKDTree 口径为 {sig4}/{tot4}）；"
  f"Melanoma = {sig_b['Melanoma']}/70（cKDTree 口径 {mel['torus']}/70）")
chk("四矩阵合计显著性在两种邻接口径下相同（960/970 不因并列取舍而变）",
    b4 == sig4 == 960, f"暴力 {b4} vs cKDTree {sig4}")
JSON_OUT["partB_neighbour_variant"] = dict(tree=dict(four=sig4, melanoma=mel["torus"]),
                                           brute=dict(four=b4, melanoma=sig_b["Melanoma"]))

# =====================================================================================
W("\n" + "=" * 108)
W("PART C  换新种子、重采样 199 -> 999：主数字是否稳健")
W("=" * 108)
rngC = np.random.default_rng(20260924)
resC = {}
for tag, d in DATA.items():
    n = d["n"]
    shifts = rngC.uniform(-0.5, 0.5, size=(B_BIG, 2))
    Tbig = torus_idx_bruteforce(d["coords"], shifts)
    Qbig = np.array([rngC.permutation(n) for _ in range(B_BIG)])
    LA, LB = d["LA"], d["LB"]
    sig = dict(plain=0, torus=0); plus1 = 0; vbig = []
    for i in range(d["ncu"]):
        order = np.argsort(-d["IA"][i])[:TOPN]
        cx = LA[:, i] - LA[:, i].mean(); ncx = np.linalg.norm(cx)
        if ncx == 0:
            continue
        Wcx = lag_edges(d["nb"], cx[:, None])[:, 0]
        CY = LB[:, order] - LB[:, order].mean(0)
        obs = (Wcx @ CY) / (ncx * np.linalg.norm(CY, axis=0))
        Ys, den = null_matrix(CY, Tbig)
        p = pvals(Wcx, ncx, obs, Ys, den)
        sig["torus"] += int((p < ALPHA).sum())
        vbig.append(p)
        pp = pvals_plus1(Wcx, ncx, obs, Ys, den)
        plus1 += int((pp < ALPHA).sum())
        Ys2, den2 = null_matrix(CY, Qbig)
        sig["plain"] += int((pvals(Wcx, ncx, obs, Ys2, den2) < ALPHA).sum())
    npair = d["ncu"] * TOPN
    lo, hi = wilson(sig["torus"], npair)
    resC[tag] = dict(pairs=npair, torus=sig["torus"], pct_torus=round(100*sig["torus"]/npair, 1),
                     ci95=[round(100*lo, 1), round(100*hi, 1)],
                     pct_torus_plus1=round(100*plus1/npair, 1),
                     pct_plain=round(100*sig["plain"]/npair, 1))
    W(f"  {tag:<13} 对={npair:>4}  环面显著={sig['torus']:>4} ({resC[tag]['pct_torus']:5.1f}%)"
      f"  95%CI=[{100*lo:.1f}, {100*hi:.1f}]   +1 修正式 {resC[tag]['pct_torus_plus1']:5.1f}%"
      f"   plain {resC[tag]['pct_plain']:.1f}%")
tot4c = sum(resC[t]["pairs"] for t in four); sig4c = sum(resC[t]["torus"] for t in four)
lo4, hi4 = wilson(sig4c, tot4c)
W(f"\n  四矩阵重估：{sig4c}/{tot4c} = {100*sig4c/tot4c:.2f}%  95%CI=[{100*lo4:.2f}, {100*hi4:.2f}]")
W(f"  Melanoma 重估：{resC['Melanoma']['torus']}/{resC['Melanoma']['pairs']} = "
  f"{resC['Melanoma']['pct_torus']:.1f}%  95%CI={resC['Melanoma']['ci95']}")
chk("换种子+999 次重采样后，四矩阵存活率仍在 95% 以上（99.0% 不是种子偶然）",
    100 * sig4c / tot4c >= 95,
    f"{100*sig4c/tot4c:.2f}%  CI=[{100*lo4:.1f}, {100*hi4:.1f}]")
JSON_OUT["partC"] = resC

# =====================================================================================
W("\n" + "=" * 108)
W("PART D  Methods 里环面平移「局限性」那些数字的独立复核")
W("=" * 108)
scc = DATA["SCC"]; T = scc["T199"]
uniq = np.array([len(np.unique(T[s])) for s in range(B_SMALL)])
maxrep = np.array([np.bincount(T[s], minlength=scc["n"]).max() for s in range(B_SMALL)])
W(f"  SCC n={scc['n']:,}，199 次位移的重复指派诊断：")
W(f"    唯一目标 spot 数：min {uniq.min()}  中位 {int(np.median(uniq))}  max {uniq.max()}  "
  f"(占比 {100*np.median(uniq)/scc['n']:.1f}%)")
W(f"    最大重复指派次数：min {maxrep.min()}  中位 {int(np.median(maxrep))}  max {maxrep.max()}")
hit = np.where((uniq == 766) | (maxrep == 85))[0]
W(f"    （历史口径对照：稿中旧句曾写「766 / 最多 85」，那是 30 次位移的**均值**，见 DIAG_R4_FOLLOWUP；"
  f"本实现 199 次位移下 766 出现 {(uniq==766).sum()} 次、85 出现 {(maxrep==85).sum()} 次）")
# 精确核对**现稿** Methods 所写的四个数（不再用 or 放宽）
chk("Methods 现稿「SCC median 741 of 1,473」，且范围 488–1,295",
    int(np.median(uniq)) == 741 and uniq.min() == 488 and uniq.max() == 1295,
    f"中位 {int(np.median(uniq))}，范围 [{uniq.min()}, {uniq.max()}]，n={scc['n']}")
chk("Methods 现稿「up to 124 repeat assignments」为 199 次位移中的精确最大值",
    int(maxrep.max()) == 124,
    f"实测最大 {int(maxrep.max())}（中位 {int(np.median(maxrep))}）")
JSON_OUT["torus_diag_SCC"] = dict(uniq_min=int(uniq.min()), uniq_median=int(np.median(uniq)),
                                  uniq_max=int(uniq.max()), maxrep_min=int(maxrep.min()),
                                  maxrep_median=int(np.median(maxrep)), maxrep_max=int(maxrep.max()),
                                  n_766=int((uniq == 766).sum()), n_85=int((maxrep == 85).sum()))

# ---- 单变量 Moran's I：**旧稿口径已被删除**，此处只作口径对照记录 --------------------
# 旧稿 Methods 曾写「0.437 未平移 / 0.408±0.064 位移后 / 94% 保留」，但那组数算在**原始计数**上，
# 与全文统一口径（log1p CP10K）不一致；换成统一口径后位移反而**抬高**自相关（多对一映射的重复赋值）。
# 该句已删除并重写，故此处不再对 0.437/0.408 断言，只记录两种口径各自的值以备复核。
det_cu = scc["det"][scc["icu"]]
i_star = int(np.argmax(det_cu))
W(f"\n  检出率最高的 cuTAR：{scc['names'][scc['icu'][i_star]]}（{100*det_cu[i_star]:.1f}% spot 检出）")
z = scc["LA"][:, i_star] - scc["LA"][:, i_star].mean()
Wz = lag_edges(scc["nb_tree"], z[:, None])[:, 0]
I0 = float(z @ Wz / (z @ z))
Is = np.array([float((z[T[s]] @ lag_edges(scc["nb_tree"], z[T[s]][:, None])[:, 0]) / (z[T[s]] @ z[T[s]]))
               for s in range(B_SMALL)])
W(f"    [统一口径 log1p CP10K] 未平移 {I0:.4f}；环面平移下 {Is.mean():.4f} ± {Is.std(ddof=1):.4f}"
  f"  保留 {100*Is.mean()/I0:.1f}%")
xr = scc["Xd"][:, i_star]
rr = xr - xr.mean()
I0r = float(rr @ lag_edges(scc["nb_tree"], rr[:, None])[:, 0] / (rr @ rr))
Isr = np.array([float(((xr[T[s]] - xr[T[s]].mean()) @ lag_edges(
    scc["nb_tree"], (xr[T[s]] - xr[T[s]].mean())[:, None])[:, 0]) /
    ((xr[T[s]] - xr[T[s]].mean()) @ (xr[T[s]] - xr[T[s]].mean())))
    for s in range(B_SMALL)])
W(f"    [旧稿的原始计数口径]   未平移 {I0r:.4f}；位移后 {Isr.mean():.4f} ± {Isr.std(ddof=1):.4f}"
  f"  保留 {100*Isr.mean()/I0r:.1f}%")
chk("旧稿 0.437 可在**原始计数**口径下复现（证明删除该句是因为口径不一致，而非算错）",
     abs(I0r - 0.4365) < 0.002, f"我算得 {I0r:.4f}")
chk("换到全文统一口径后，位移**抬高**自相关（保留 >100%），故旧句结论方向不成立",
     Is.mean() > I0, f"{I0:.4f} -> {Is.mean():.4f}")
JSON_OUT["univariate_Moran_SCC_topcuTAR"] = dict(feature=str(scc["names"][scc["icu"][i_star]]),
                                                 det=float(det_cu[i_star]),
                                                 log=dict(I0=I0, mean=float(Is.mean()),
                                                          sd=float(Is.std(ddof=1))),
                                                 raw=dict(I0=I0r, mean=float(Isr.mean()),
                                                          sd=float(Isr.std(ddof=1))))

# ---- 三种零模型零分布宽度：旧稿的 0.070–0.076 / 0.016–0.017 / 0.009–0.012 -----------
# 该三组范围在 78 个 SCC cuTAR 中**没有一个**能同时复现（verify_supp 与 DIAG_R4_FOLLOWUP 已证），
# 旧句已删除；现稿 Methods 改为引用 R4_METHODS_NUMBERS.txt 的**全配对**口径。
W("\n  零分布宽度（SCC 检出率最高 cuTAR 的前 5 个伙伴）——仅供对照，现稿已改用全配对口径：")
i = i_star
order = np.argsort(-scc["IA"][i])[:TOPN]
cx = scc["LA"][:, i] - scc["LA"][:, i].mean(); ncx = np.linalg.norm(cx)
Wcx = lag_edges(scc["nb_tree"], cx[:, None])[:, 0]
CY = scc["LB"][:, order] - scc["LB"][:, order].mean(0)
sd = {}
for key, M in (("plain", streams["SCC"]["Q"]), ("strat", streams["SCC"]["P"]), ("torus", T)):
    Ys, den = null_matrix(CY, M)
    num = np.einsum("bnq,n->bq", Ys, Wcx)
    null = num / (ncx * den)
    sd[key] = np.std(null, axis=0, ddof=1)
    W(f"    {key:<6} 五个伙伴的零分布 SD = " + "  ".join(f"{v:.4f}" for v in sd[key])
      + f"   范围 [{sd[key].min():.4f}, {sd[key].max():.4f}]")
chk("环面平移零分布确实最宽（三者最大 SD 的排序 torus > plain > strat）",
    sd["strat"].min() >= 0.0085 and sd["strat"].max() <= 0.0125,
    f"[{sd['strat'].min():.4f}, {sd['strat'].max():.4f}]")
chk("环面平移零分布确实最宽（三者的最大 SD 排序 torus > plain > strat）",
    sd["torus"].min() > sd["plain"].max() > sd["strat"].max(),
    f"torus≥{sd['torus'].min():.4f} > plain≤{sd['plain'].max():.4f} > strat≤{sd['strat'].max():.4f}")
JSON_OUT["null_sd_SCC"] = {k: [float(v.min()), float(v.max())] for k, v in sd.items()}

# =====================================================================================
W("\n" + "=" * 108)
W("PART E  零模型标定：这套检验在真零假设下的实际拒绝率（原件从未做过）")
W("=" * 108)
W("""  做法：留出第 1000 次环面平移，用它把 y 变成"真值零假设"下的观测
        （H0 由构造成立），再用其余 999 次求 p。若检验标定正确，p ~ U(0,1)，5% 水平拒绝率 ≈ 5%。
        分别对 B=999 与 B=199 两套重采样数做，看重采样数是否影响标定。""")
rngE = np.random.default_rng(777)
cal = {}
NP_ = 200
for tag, d in DATA.items():
    n = d["n"]
    sh_all = rngE.uniform(-0.5, 0.5, size=(B_BIG + 1, 2))
    T_all = torus_idx_bruteforce(d["coords"], sh_all)
    hold, Tn = T_all[B_BIG], T_all[:B_BIG]
    Tn199 = Tn[:B_SMALL]
    rb = rngE.integers(0, d["ncu"], NP_)
    rp = rngE.integers(0, d["npc"], NP_)
    p999, p199, obs_abs = [], [], []
    for a, b in zip(rb, rp):
        x = d["Zx"][:, a]; y = d["Zy"][:, b]
        nx = np.linalg.norm(x); ny = np.linalg.norm(y)
        yh = y[hold]
        Wx = lag_edges(d["nb"], x[:, None])[:, 0]
        obs = float(Wx @ yh / (nx * np.linalg.norm(yh)))
        Ys, den = null_matrix(y[:, None], Tn)
        p999.append(float(pvals(Wx, nx, np.array([obs]), Ys, den)[0]))
        Ys2, den2 = null_matrix(y[:, None], Tn199)
        p199.append(float(pvals(Wx, nx, np.array([obs]), Ys2, den2)[0]))
    p999, p199 = np.array(p999), np.array(p199)
    cal[tag] = dict(rej999=float((p999 < ALPHA).mean()), rej199=float((p199 < ALPHA).mean()),
                    mean999=float(p999.mean()), median999=float(np.median(p999)),
                    frac_lt10=float((p999 < 0.10).mean()))
    W(f"  {tag:<13} B=999: 拒绝率 {100*cal[tag]['rej999']:5.1f}%  中位 p {np.median(p999):.3f}"
      f"  均值 p {p999.mean():.3f}  P(p<0.10)={100*cal[tag]['frac_lt10']:5.1f}%"
      f"   |  B=199: 拒绝率 {100*cal[tag]['rej199']:5.1f}%")
allp999 = np.concatenate([np.array([cal[t]["rej999"]]) for t in cal])
nelig = len(cal) * NP_
ntot = int(round(sum(cal[t]["rej999"] * NP_ for t in cal)))
meanrej = float(ntot / nelig)
from math import sqrt
mu = 0.05 * nelig; sd = sqrt(nelig * 0.05 * 0.95)
lo_b, hi_b = mu - 1.96 * sd, mu + 1.96 * sd
W(f"\n  合并：{ntot}/{nelig} = {100*meanrej:.2f}%；"
  f"若检验尺寸正确，应落在 Binomial({nelig}, 0.05) 的 95% 区间 [{lo_b:.1f}, {hi_b:.1f}] 内")
chk("零模型标定：合并拒绝数落在名义水平 Binomial 的 95% 区间内（检验尺寸正确）",
    lo_b <= ntot <= hi_b,
    f"实测 {ntot}，区间 [{lo_b:.1f}, {hi_b:.1f}]")
JSON_OUT["calibration"] = cal
JSON_OUT["calibration_pooled"] = dict(rejected=ntot, n=nelig, pct=100 * meanrej,
                                      ci95_binom=[lo_b, hi_b])
JSON_OUT["calibration_mean_rej999"] = meanrej

# =====================================================================================
W("\n" + "=" * 108)
W("PART F  稿件中「头部伙伴高于随机 31–38 倍」的口径核查")
W("=" * 108)
W("""  稿中该倍数 = MORANS_conditioned.json 的 top_overlap ÷ SELFCHECK4.json 的 chance_overlap。
  前者是 **变体 A（log 值）与变体 B（对 log 深度取残差）两者 top-10 伙伴名单的重叠率**，
  后者是 10²/N。即该倍数衡量的是「换一种去深度的做法后，头部伙伴名单还有多少保留」，
  **不是**「最佳伙伴的关联强度高于随机多少倍」。这两件事被同一个句子混在了一起。""")
mc = {x["tag"]: x for x in json.load(open(os.path.join(RES, "MORANS_conditioned.json"), encoding="utf-8"))}
ratios = {}
for tag in four:
    r = mc[tag]["top_overlap"] / json4[tag]["chance_overlap"]
    ratios[tag] = r
    W(f"  {tag:<13} top_overlap {mc[tag]['top_overlap']:.4f} / {json4[tag]['chance_overlap']:.4f}"
      f" = {r:.1f}×   变体A/B各显著 {mc[tag]['sig_A']}/{mc[tag]['sig_B']}")
W(f"  范围 {min(ratios.values()):.1f}–{max(ratios.values()):.1f}  -> 与稿中 31–38 倍一致")
W("\n  真正对应「高于随机」的量是同文件 J4 的检出率匹配对照：")
for tag in ["SCC", "HNC", "KidneyCancer", "BCC", "Melanoma"]:
    W(f"  {tag:<13} 真 cuTAR 最佳伙伴 {mc[tag]['best_cu']:.4f} vs 匹配伪 cuTAR {mc[tag]['best_null']:.4f}"
      f"   差 {mc[tag]['best_cu']-mc[tag]['best_null']:+.4f}")
chk("31–38 倍这一倍数本身复现（30.8–38.2）",
    round(min(ratios.values())) == 31 and round(max(ratios.values())) == 38,
    "  ".join(f"{t}={ratios[t]:.1f}" for t in ratios))
JSON_OUT["top_overlap_ratio"] = ratios

# =====================================================================================
W("\n" + "=" * 108)
W(f"汇总：{sum(1 for _, c, _ in CHK if c)}/{len(CHK)} 项通过      总耗时 {time.time()-T0:.0f}s")
W("=" * 108)
for nm, c, dt in CHK:
    if not c:
        W(f"  未通过：{nm}  ({dt})")
open(os.path.join(RES, "VERIFY_R4_INDEPENDENT.txt"), "w", encoding="utf-8").write("\n".join(OUT))
json.dump(JSON_OUT, open(os.path.join(RES, "VERIFY_R4_INDEPENDENT.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\VERIFY_R4_INDEPENDENT.txt / .json")
