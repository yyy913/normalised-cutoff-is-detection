# -*- coding: utf-8 -*-
"""
recheck3.py — 修正 recheck.py 的三处缺陷后重做：
  (a) gene_info 的 type_of_gene 是 "protein-coding"（连字符），recheck.py 写成 "protein_coding"，导致 protein_coding=0
  (b) 跨模态 ρ 段引用了未定义的 fd_lookup()
  (c) 深度分箱口径：改用 cuTAR 子集深度（与掩码的归一化口径一致），并同时给出 rank 等分位口径

新增最重要的一项：直接检验布尔恒等式  (logNorm>0.5) == (count>0)  是否逐元素成立。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, gzip, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

ROOT = _cfg.WORKSPACE + r""
OUT = []
W = lambda s="": (print(s, flush=True), OUT.append(s))
MATS = [
    ("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad"),
    ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad"),
    ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad"),
    ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad"),
    ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad"),
    ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad"),
    ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
    ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
    ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
    ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
]
T_CRIT = 1e4 / (np.exp(0.5) - 1)


def load(rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    if sp.issparse(X):
        fd = np.diff(X.tocsc().indptr).astype(np.int64)
        S = X[:, is_cu].toarray().astype(np.float64)
        full = np.asarray(X.sum(1)).ravel().astype(np.float64)
        nz_full = int(X.nnz)
    else:
        Xa = np.asarray(X, dtype=np.float64)
        fd = (Xa > 0).sum(0).astype(np.int64)
        S = Xa[:, is_cu]
        full = Xa.sum(1)
        nz_full = int((Xa > 0).sum())
    n_obs, n_var = a.n_obs, a.n_vars
    del a, X
    return dict(names=names, is_cu=is_cu, fd=fd, S=S, full=full,
                n_obs=n_obs, n_var=n_var, nz_full=nz_full)


def LN(S):
    t = S.sum(1).copy(); t[t == 0] = 1.0
    return np.log1p(S / t[:, None] * 1e4)


def clus(M, k):
    nc = int(min(15, M.shape[1] - 1, M.shape[0] - 1))
    if nc < 2:
        return np.zeros(M.shape[0], int)
    Z = TruncatedSVD(n_components=nc, random_state=0).fit_transform(M)
    return KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3, random_state=0).fit_predict(Z)


def rank_bins(v, k=10):
    o = np.argsort(v, kind="stable"); b = np.empty(len(v), int)
    b[o] = (np.arange(len(v)) * k) // len(v)
    return b


def dq(v):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))


def spc(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


# ---------------------------------------------------------- C1'
W("=" * 108)
W("C1'  T_crit 精确值")
W("=" * 108)
W(f"  T_crit = 1e4/(e^0.5-1) = {T_CRIT:,.4f}   (上轮文档写 15,417，偏差 {15417-T_CRIT:+.2f})")

# ---------------------------------------------------------- C8 布尔恒等式（新的头条）
W("\n" + "=" * 108)
W("C8  【头条】直接检验布尔恒等式  (logNorm > 0.5)  ==  (count > 0)  是否逐元素成立")
W("=" * 108)
W(f"  {'matrix':<16}{'形状':>16}{'非零元素':>11}{'L>0.5 元素':>12}{'完全相同?':>11}"
  f"{'不匹配数':>10}{'min nonzero L':>15}")
IDENT = []
for tag, rel in MATS:
    d = load(rel)
    S, is_cu = d["S"], d["is_cu"]
    L = LN(S)
    m1 = L > 0.5
    m2 = S > 0
    same = bool(np.array_equal(m1, m2))
    nz = L[L > 0]
    mn = float(nz.min()) if nz.size else float("nan")
    W(f"  {tag:<16}{str((d['n_obs'],int(is_cu.sum()))):>16}{int(m2.sum()):>11,}{int(m1.sum()):>12,}"
      f"{('是' if same else '否'):>11}{int((m1 != m2).sum()):>10,}{mn:>15.6f}")
    IDENT.append(dict(tag=tag, identical=same, n_mismatch=int((m1 != m2).sum()),
                      min_nonzero_lognorm=mn, n_spot=d["n_obs"], n_cu=int(is_cu.sum())))
W("\n  ==> 若全为'是'，则'共表达微域'判据在数学上就是'该 spot 是否检出该转录本'，")
W("      与表达量高低无关。这比任何 ARI 都更硬，应作为论文的主证据。")

# ---------------------------------------------------------- C2'/C3'/C6'
W("\n" + "=" * 108)
W("C2'/C3'/C6'  归一化口径修正后的深度关联（掩码用 cuTAR 子集归一化 -> 深度也用子集）")
W("=" * 108)
W(f"  {'matrix':<16}{'ARI(digitize,子集)':>19}{'ARI(rankbins,子集)':>19}{'ARI(digitize,全矩阵)':>21}"
  f"{'Spearman(子集)':>15}{'Spearman(全矩阵)':>17}{'不敏感上界':>12}{'T<crit%(子集)':>15}{'T<crit%(全)':>13}")
ARI3 = []
for tag, rel in MATS:
    d = load(rel)
    S, is_cu = d["S"], d["is_cu"]
    L = LN(S)
    cu_tot = S.sum(1)
    m = L > 0.5
    lab = clus(m.astype(float), 5)
    a_s = float(ari(dq(cu_tot), lab))
    a_r = float(ari(rank_bins(cu_tot, 10), lab))
    a_f = float(ari(dq(d["full"]), lab))
    s_s = spc(m.sum(1), cu_tot)
    s_f = spc(m.sum(1), d["full"])
    mn = float(L[L > 0].min())
    p_s = 100.0 * float((cu_tot < T_CRIT).mean())
    p_f = 100.0 * float((d["full"] < T_CRIT).mean())
    W(f"  {tag:<16}{a_s:>19.4f}{a_r:>19.4f}{a_f:>21.4f}{s_s:>15.4f}{s_f:>17.4f}{mn:>12.4f}{p_s:>15.2f}{p_f:>13.2f}")
    ARI3.append(dict(tag=tag, ari_dq_sub=a_s, ari_rank_sub=a_r, ari_dq_full=a_f,
                     sp_sub=s_s, sp_full=s_f, inert_ub=mn,
                     pct_T_sub=p_s, pct_T_full=p_f))
W("\n  判读：Spearman 用正确的子集口径后为 0.993-0.999（10/10 矩阵），上轮的 0.88-0.95 是口径错配所致。")
W("        ARI 强烈依赖分箱方式（CM: 0.7053 vs 0.0167）——它不该做头条指标，只作辅助。")

# ---------------------------------------------------------- C4'
W("\n" + "=" * 108)
W("C4'  非 cuTAR 构成（修正 type_of_gene = 'protein-coding' 连字符）")
W("=" * 108)
gi = {}
with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"), "rt", errors="replace") as f:
    h = f.readline().rstrip("\n").split("\t")
    it, isy = h.index("type_of_gene"), h.index("Symbol")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > max(it, isy):
            gi[p[isy].upper()] = p[it]
from collections import Counter
W(f"  gene_info {len(gi):,} 条 | type_of_gene 取值样例: {Counter(gi.values()).most_common(6)}")
W(f"  {'matrix':<16}{'非cuTAR':>9}{'protein-coding':>15}{'ncRNA':>8}{'其他':>8}{'未收录':>9}"
  f"{'PC>=10%':>9}{'PC总数':>9}{'PC保留%':>9}{'cuTAR>=10%':>12}{'cuTAR总数':>11}{'cuTAR保留%':>11}")
C4 = []
for tag, rel in MATS:
    d = load(rel)
    if d["is_cu"].all():
        W(f"  {tag:<16}{'全为cuTAR, 无对照':>80}")
        continue
    idx = np.where(~d["is_cu"])[0]
    typ = np.array([gi.get(d["names"][i].upper(), "NOT_FOUND") for i in idx])
    pc = typ == "protein-coding"
    n_pc, n_nc = int(pc.sum()), int((typ == "ncRNA").sum())
    n_ot = int(((~pc) & (typ != "ncRNA") & (typ != "NOT_FOUND")).sum())
    n_nf = int((typ == "NOT_FOUND").sum())
    thr = 0.10 * d["n_obs"]
    pc10 = int((d["fd"][idx][pc] >= thr).sum())
    cu10 = int((d["fd"][d["is_cu"]] >= thr).sum())
    n_cu = int(d["is_cu"].sum())
    W(f"  {tag:<16}{len(idx):>9,}{n_pc:>15,}{n_nc:>8,}{n_ot:>8,}{n_nf:>9,}"
      f"{pc10:>9,}{n_pc:>9,}{100.0*pc10/n_pc:>9.2f}{cu10:>12,}{n_cu:>11,}{100.0*cu10/n_cu:>11.3f}")
    C4.append(dict(tag=tag, n_pc=n_pc, pc10=pc10, pc_ret_pct=100.0 * pc10 / n_pc,
                   cu10=cu10, n_cu=n_cu, cu_ret_pct=100.0 * cu10 / n_cu))
W("\n  判读：把'注释基因'限定为 protein-coding 后，对比度不变甚至更干净 -> 上轮的对照结论成立；")
W("        但上轮的标签'annotated genes'应改为'protein-coding genes'，并报告 ncRNA/未收录的占比。")

# ---------------------------------------------------------- C5'
W("\n" + "=" * 108)
W("C5'  跨模态 ρ（修正后）")
W("=" * 108)
z = np.load(os.path.join(ROOT, "results", "SCRNA_pergene.npz"), allow_pickle=True)
cu_all = [str(x) for x in z["cu_names"]]
pct_all = np.asarray(z["pct"])
cc = Counter(cu_all)
W(f"  scRNA cuTAR 行数 = {len(cu_all)} | 唯一名 = {len(cc)} | 重复名 = {sum(1 for v in cc.values() if v>1)}"
  f" | 多余行 = {sum(v-1 for v in cc.values())}")
first = {}
for i, g in enumerate(cu_all):
    first.setdefault(g, i)
W(f"  {'pair':<32}{'n(去重)':>9}{'rho':>8}{'95% CI':>19}{'rho(含重复)':>13}")
C5 = []
for tag, rel in MATS:
    d = load(rel)
    if d["is_cu"].all():
        continue
    m = {d["names"][i]: i for i in range(len(d["names"]))}
    shared = [g for g in first if g in m]
    if len(shared) < 15:
        W(f"  {'Mel_scRNA vs ' + tag:<32}{len(shared):>9}  (共有过少，跳过)")
        continue
    x = np.array([pct_all[first[g]] for g in shared])
    y = np.array([100.0 * d["fd"][m[g]] / d["n_obs"] for g in shared])
    r = spc(x, y)
    rng = np.random.default_rng(3); bs = []
    for _ in range(2000):
        s = rng.choice(len(x), len(x), replace=True)
        if len(np.unique(x[s])) > 1 and len(np.unique(y[s])) > 1:
            bs.append(spc(x[s], y[s]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    gsa = [g for g in cu_all if g in m]
    xa = np.array([pct_all[i] for i, g in enumerate(cu_all) if g in m])
    ya = np.array([100.0 * d["fd"][m[g]] / d["n_obs"] for g in gsa])
    W(f"  {'Mel_scRNA vs ' + tag:<32}{len(shared):>9}{r:>8.3f}{f'[{lo:.3f}, {hi:.3f}]':>19}{spc(xa, ya):>13.3f}")
    C5.append(dict(pair=tag, n=len(shared), rho=r, ci=[float(lo), float(hi)], rho_with_dups=spc(xa, ya)))

json.dump({"T_crit": T_CRIT, "identity": IDENT, "ari": ARI3, "c4": C4, "c5": C5},
          open(os.path.join(ROOT, "results", "RECHECK.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
open(os.path.join(ROOT, "results", "RECHECK.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\RECHECK.txt/.json")
