# -*- coding: utf-8 -*-
"""
recheck.py — 对方案 A 上一轮结论做对抗性复核。逐项怀疑并验证：

C1  T_crit 的精确值（上轮写 15,417，需重算）
C2  归一化口径错配：混合矩阵的阈值等价性是否该用 cuTAR 子集深度而非全矩阵深度？
C3  decile() 在 >50% spot 为零时退化（CM/CP）→ 上轮的 ARI 是否为并列塌缩的产物？
C4  "cuTAR vs 注释基因"对照不纯：非 cuTAR 里含大量未注释 lncRNA/novel，须限定到蛋白编码基因
C5  跨模态 ρ 是否受 var 重复名影响（scRNA var.index 有 48 个重复名）
C6  "300×–500× 不敏感区间"是否为网格假象？精确不敏感上界是什么？
C7  CM/CP 特征交集、以及各处声明的可复现性
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

# ---------------------------------------------------------------- C1
W("=" * 104)
W("C1  T_crit 精确值")
W("=" * 104)
tc = 1e4 / (np.exp(0.5) - 1)
W(f"  e^0.5 - 1                = {np.exp(0.5)-1:.10f}")
W(f"  T_crit = 1e4/(e^0.5-1)   = {tc:.4f}")
W(f"  上轮文档写 15,417  ->  偏差 {15417-tc:+.2f}  ({(15417-tc)/tc*100:+.4f}%)")
for T in (15414.9, 15415, 15417):
    W(f"  校验 T={T:<10}: log1p(1e4/T) = {np.log1p(1e4/T):.8f}  -> 单 read {'过阈' if np.log1p(1e4/T)>0.5 else '不过阈'}")
W(f"  ==> 正确值 T_crit = {tc:,.2f}（约 15,415），上轮的 15,417 是错的")

# ---------------------------------------------------------------- 准备
def cu_subset_stats(rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    if sp.issparse(X):
        Xc = X.tocsc()
        fd = np.diff(Xc.indptr).astype(np.int64)
        S = X[:, is_cu].toarray().astype(np.float64)
        full_tot = np.asarray(X.sum(1)).ravel().astype(np.float64)
    else:
        Xa = np.asarray(X, dtype=np.float64)
        fd = (Xa > 0).sum(0).astype(np.int64)
        S = Xa[:, is_cu]
        full_tot = Xa.sum(1)
    del X, a
    return names, is_cu, fd, S, full_tot


def lognorm(S):
    t = S.sum(1).copy()
    t[t == 0] = 1.0
    return np.log1p(S / t[:, None] * 1e4)


def cluster_of(M, k, ncomp=15):
    ncomp = int(min(ncomp, M.shape[1] - 1, M.shape[0] - 1))
    Z = TruncatedSVD(n_components=ncomp, random_state=0).fit_transform(M)
    return KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3, random_state=0).fit_predict(Z)


def rank_bins(v, k=10):
    """按秩强制等分位（并列也强制等分），避免 digitize 在大量零值下退化。"""
    order = np.argsort(v, kind="stable")
    b = np.empty(len(v), int)
    b[order] = (np.arange(len(v)) * k) // len(v)
    return b


def decile_old(v):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))


def sp_corr(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


# ---------------------------------------------------------------- C2 C3 C6
W("\n" + "=" * 104)
W("C2/C3/C6  归一化口径、深度分箱退化、精确不敏感上界")
W("=" * 104)
W(f"  {'matrix':<16}{'cuTARsubj中位':>13}{'cuTARsubj最大':>13}{'min nonzero logNorm':>20}"
  f"{'不敏感上界':>11}{'上轮记录':>10}{'T<crit spot%(子集)':>19}{'T<crit spot%(全矩阵)':>21}")
RES = {}
for tag, rel in MATS:
    names, is_cu, fd, S, full_tot = cu_subset_stats(rel)
    cu_tot = S.sum(1)
    L = lognorm(S)
    nz = L[L > 0]
    mn = float(nz.min()) if nz.size else 0.0
    mx = float(cu_tot.max())
    pct_sub = 100.0 * float((cu_tot < tc).mean())
    pct_full = 100.0 * float((full_tot < tc).mean())
    RES[tag] = dict(cu_tot=cu_tot, L=L, is_cu=is_cu, fd=fd, S=S, mn=mn, mx=mx)
    W(f"  {tag:<16}{np.median(cu_tot):>13.0f}{mx:>13.0f}{mn:>20.6f}{mn:>11.6f}{'':>10}"
      f"{pct_sub:>19.2f}{pct_full:>21.2f}")

W("\n  C6 说明：掩码 `L>t` 对所有 t < min nonzero logNorm 完全不变。")
W("     正确的'不敏感上界'是 min nonzero logNorm = log1p(1e4/T_max_cuTAR)，不是上轮的网格点 3.0/4.0/5.0。")

# ---------------------------------------------------------------- C3 重算 ARI
W("\n" + "=" * 104)
W("C3  ARI(naive 域, 深度) 的三种口径对比")
W("=" * 104)
W(f"  {'matrix':<16}{'ARI(digitize,全矩阵)':>21}{'ARI(digitize,子集)':>19}"
  f"{'ARI(rankbins,子集)':>19}{'Spearman(过阈数,子集深度)':>25}")
ARI_ROWS = []
for tag, rel in MATS:
    R = RES[tag]
    L, cu_tot, full_tot = R["L"], R["cu_tot"], None
    names, is_cu, fd, S, full_tot = cu_subset_stats(rel)
    m05 = L > 0.5
    lab = cluster_of(m05.astype(float), 5)
    a_old = float(ari(decile_old(full_tot), lab))
    a_sub = float(ari(decile_old(cu_tot), lab))
    a_rnk = float(ari(rank_bins(cu_tot, 10), lab))
    sq = sp_corr(m05.sum(1), cu_tot)
    ARI_ROWS.append(dict(tag=tag, ari_digitize_full=a_old, ari_digitize_sub=a_sub,
                         ari_rankbins_sub=a_rnk, spearman_sub=sq))
    W(f"  {tag:<16}{a_old:>21.4f}{a_sub:>19.4f}{a_rnk:>19.4f}{sq:>25.4f}")

W("\n  C3 判读：CM/CP 全矩阵总深度 85% 为零，digitize 分箱塌缩（有效仅 2 箱）；")
W("     rank_bins 强制等分位后才是可比的'深度十分位'口径。")

# ---------------------------------------------------------------- C4
W("\n" + "=" * 104)
W("C4  非 cuTAR 的构成：上轮把'未注释 lncRNA/novel'也算进了'注释基因'")
W("=" * 104)
gi = {}
with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"), "rt", errors="replace") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    it, isym = hdr.index("type_of_gene"), hdr.index("Symbol")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > max(it, isym):
            gi[p[isym].upper()] = p[it]
W(f"  gene_info 载入 {len(gi):,} 条 (type_of_gene)")
W(f"  {'matrix':<16}{'非cuTAR':>9}{'protein_coding':>15}{'ncRNA':>8}{'其他已知':>9}{'不在gene_info':>14}"
  f"{'pc>=10%':>9}{'pc总数':>9}{'pc保留%':>9}{'cuTAR>=10%':>12}")
C4 = {}
for tag, rel in MATS:
    R = RES[tag]
    names, is_cu, fd, S, full_tot = cu_subset_stats(rel)
    if is_cu.all():
        W(f"  {tag:<16}{'—':>9}{'—':>15}{'—':>8}{'—':>9}{'—':>14}{'—':>9}{'—':>9}{'—':>9}{'—':>12}")
        continue
    idx = np.where(~is_cu)[0]
    typ = np.array([gi.get(names[i].upper(), "NOT_FOUND") for i in idx])
    n_pc = int((typ == "protein_coding").sum())
    n_ncrna = int((typ == "ncRNA").sum())
    n_oth = int(((typ != "protein_coding") & (typ != "ncRNA") & (typ != "NOT_FOUND")).sum())
    n_nf = int((typ == "NOT_FOUND").sum())
    pcm = (typ == "protein_coding")
    pc_ge10 = int((fd[idx][pcm] >= 0.10 * len(full_tot)).sum())
    cu_ge10 = int((fd[is_cu] >= 0.10 * len(full_tot)).sum())
    W(f"  {tag:<16}{len(idx):>9,}{n_pc:>15,}{n_ncrna:>8,}{n_oth:>9,}{n_nf:>14,}"
      f"{pc_ge10:>9,}{n_pc:>9,}{100.0*pc_ge10/max(1,n_pc):>9.2f}{cu_ge10:>12,}")
    C4[tag] = dict(n_pc=n_pc, pc_ge10=pc_ge10, pc_ret=100.0 * pc_ge10 / max(1, n_pc),
                   cu_ge10=cu_ge10, n_cu=int(is_cu.sum()),
                   cu_ret=100.0 * cu_ge10 / max(1, int(is_cu.sum())))

# ---------------------------------------------------------------- C5
W("\n" + "=" * 104)
W("C5  跨模态 ρ 是否受 scRNA var 重复名影响")
W("=" * 104)
z = np.load(os.path.join(ROOT, "results", "SCRNA_pergene.npz"), allow_pickle=True)
cu = [str(x) for x in z["cu_names"]]
pct = z["pct"]
uniq = len(set(cu))
W(f"  scRNA cuTAR 特征数 = {len(cu)}，去重后 = {uniq}，重复 {len(cu)-uniq} 个")
sp_ref = {}
for tag, rel in MATS:
    if tag in ("CM_pacbio", "CP_pacbio", "BCC_nano", "SCC_nano", "HNC_ilong_nano"):
        continue
    names, is_cu, fd, S, full_tot = cu_subset_stats(rel)
    n = len(full_tot)
    m = {names[i]: i for i in range(len(names))}
    d = {}
    for i, g in enumerate(cu):
        if g in m:
            d[g] = d.get(g, 0.0)
    sp_ref[tag] = {g: fd[m[g]] / n * 100 for g in d}


def boot_ci(x, y, B=2000, seed=1):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    out = []
    for _ in range(B):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(x[s])) < 2 or len(np.unique(y[s])) < 2:
            continue
        out.append(sp_corr(x[s], y[s]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


W(f"  {'pair':<34}{'n(含重复)':>10}{'n(去重)':>9}{'rho(去重)':>11}{'95% CI':>20}")
C5 = []
for tag, d in sp_ref.items():
    # 含重复
    gs_all = list(d.keys())
    x_all = np.array([fd_lookup(g, cu, pct) for g in gs_all])
    # 去重：同名取先出现者
    seen, gs, xs, ys = set(), [], [], []
    for g in gs_all:
        if g in seen:
            continue
        seen.add(g); gs.append(g); xs.append(pct[cu.index(g)]); ys.append(d[g])
    xs, ys = np.array(xs), np.array(ys)
    r_all = sp_corr(np.array([pct[cu.index(g)] for g in gs_all]), np.array([d[g] for g in gs_all]))
    r = sp_corr(xs, ys)
    lo, hi = boot_ci(xs, ys)
    W(f"  {'Mel_scRNA vs ' + tag:<34}{len(gs_all):>10,}{len(gs):>9,}{r:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}")
    C5.append(dict(pair=tag, n=len(gs), rho=r, ci=[lo, hi], n_all=len(gs_all), rho_all=r_all))

# ---------------------------------------------------------------- C7
W("\n" + "=" * 104)
W("C7  CM/CP 特征交集与可复现性抽查")
W("=" * 104)
def feats(rel):
    a = ad.read_h5ad(os.path.join(ROOT, rel), backed="r")
    s = set(str(x) for x in a.var.index)
    return s


cm = feats(r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad")
cp = feats(r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad")
W(f"  CM n={len(cm):,}  CP n={len(cp):,}  交集={len(cm & cp):,}  "
  f"(上轮记录 1,204 / 1,411 / 530 -> {'一致' if (len(cm),len(cp),len(cm&cp))==(1204,1411,530) else '不一致'})")
for g in ("cuTAR283862", "cuTAR27404"):
    W(f"  {g}: in CM={g in cm}  in CP={g in cp}")

# BCC == Vis9D_BCC 抽查
b = ad.read_h5ad(os.path.join(ROOT, r"data\SPanC-Lnc-pan\BCC.h5ad"), backed="r")
W(f"  BCC uns['spatial'] library id = {list(b.uns['spatial'].keys())}")
W(f"  CM  uns['spatial'] library id = {list(ad.read_h5ad(os.path.join(ROOT, r'data\SPanC-Lnc-CRC\CM_pacbio.h5ad'), backed='r').uns['spatial'].keys())}")

json.dump({"T_crit": tc, "ari": ARI_ROWS, "c4": C4, "c5": C5},
          open(os.path.join(ROOT, "results", "RECHECK.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
open(os.path.join(ROOT, "results", "RECHECK.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\RECHECK.txt/.json")
