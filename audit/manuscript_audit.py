# -*- coding: utf-8 -*-
"""
manuscript_audit.py — 全量重算审计
对两份稿子中出现的每一个数字，从**原始 h5ad** 重新计算（不读任何先前脚本的 JSON），
再与稿中文字逐条比对。原则：不继承，只重算。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, re, gzip, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

ROOT = _cfg.WORKSPACE + r""
MATS = [("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad"),
        ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad"),
        ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad"),
        ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad"),
        ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
        ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad")]
OUT, CHK = [], []
def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "MANUSCRIPT_AUDIT.txt"), "w", encoding="utf-8").write("\n".join(OUT))
def chk(claim, manuscript, recomputed, match, detail=""):
    CHK.append((claim, match)); W(f"  [{'OK ' if match else '**MISMATCH**'}] {claim}")
    W(f"        稿中={manuscript}   重算={recomputed}   {detail}")

# ---------- 基因型表（重算 protein-coding 分类）
gi = {}
with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"), "rt", errors="replace") as f:
    h = f.readline().rstrip("\n").split("\t")
    it, isy = h.index("type_of_gene"), h.index("Symbol")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if len(p) > max(it, isy):
            gi[p[isy].upper()] = p[it]
W(f"gene_info 载入 {len(gi):,} 条，type_of_gene 取值含 'protein-coding': "
  f"{'protein-coding' in set(gi.values())}")

R = {}
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    Xd = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    n = a.n_obs
    lib = float(Xd.sum())
    fd_all = (Xd > 0).sum(0)
    S = Xd[:, is_cu]
    T = S.sum(1)
    L = np.log1p(S / np.where(T == 0, 1.0, T)[:, None] * 1e4)
    nz = S > 0
    det_cu = nz.mean(0)
    # protein-coding 保留率（10% 检出）
    if (~is_cu).any():
        pc = np.array([gi.get(str(names[i]).upper(), "") == "protein-coding"
                       for i in np.where(~is_cu)[0]])
        pc10 = int((fd_all[np.where(~is_cu)[0]][pc] >= 0.10 * n).sum())
        n_pc = int(pc.sum())
    else:
        pc10, n_pc = None, None
    R[tag] = dict(n=n, n_cu=int(is_cu.sum()), lib=lib,
                  n_nonzero=int(nz.sum()), mismatch=int(((L > 0.5) != nz).sum()),
                  min_nz=float(L[nz].min()) if nz.any() else np.nan,
                  Tmax=float(T.max()),
                  med_det=float(np.median(det_cu)),
                  best_det=float(det_cu.max()),
                  cu10=int((det_cu >= 0.10).sum()),
                  pc10=pc10, n_pc=n_pc, all_cu=bool(is_cu.all()))
    del a, X, Xd, S, L

TC = 1e4 / (np.exp(0.5) - 1)
W("\n" + "="*100); W("A 组：引言"); W("="*100)
meds = [R[t]["med_det"] * 100 for t, _ in MATS]
chk("cuTAR 中位检出率 0.08–0.80% spot", "0.08–0.80%",
    f"{min(meds):.3f}–{max(meds):.3f}%", round(min(meds), 2) == 0.08 and round(max(meds), 2) == 0.80)
tot_cu = sum(R[t]["n_cu"] for t, _ in MATS)
# 「均值 < 最小非零值」从原始 h5ad 重算（不再读 ATLAS_quadrant_test.json）
q_tot = q_det = q_ok = 0
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel)); X = a.X
    Xd = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    ic = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    S = Xd[:, ic]; T = S.sum(1)
    Lx = np.log1p(S / np.where(T == 0, 1.0, T)[:, None] * 1e4)
    nz = Lx > 0
    mn = np.where(nz.any(0), np.where(nz, Lx, np.inf).min(0), np.nan)
    mu = Lx.mean(0)
    v = np.isfinite(mn)
    q_tot += int(ic.sum()); q_det += int(v.sum()); q_ok += int((mn[v] > mu[v]).sum())
    del a, X, Xd, S, Lx
chk("至少在 1 个 spot 检出过的 cuTAR 中 33,657 / 33,659 满足『均值 < 最小非零值』",
    "33,657 / 33,659（另有 2,632 条无任何计数）",
    f"{q_ok:,} / {q_det:,} = {100*q_ok/q_det:.3f}%（总 cuTAR {q_tot:,}，无计数 {q_tot-q_det:,}）",
    (q_ok == 33657 and q_det == 33659 and q_tot == 36291),
    "本项已改为从原始 h5ad 重算")

W("\n" + "="*100); W("B 组：判据恒等（R1）"); W("="*100)
mm = sum(R[t]["mismatch"] for t, _ in MATS)
chk("零不匹配（10 矩阵）", "0", str(mm), mm == 0)
nzs = [R[t]["n_nonzero"] for t, _ in MATS]
chk("非零元素 7,326–52,217", "7,326–52,217", f"{min(nzs):,}–{max(nzs):,}",
    min(nzs) == 7326 and max(nzs) == 52217)
mns = [R[t]["min_nz"] for t, _ in MATS]
chk("最小非零 logNorm 3.31–5.53", "3.31–5.53", f"{min(mns):.4f}–{max(mns):.4f}",
    round(min(mns), 2) == 3.31 and round(max(mns), 2) == 5.53)

W("\n" + "="*100); W("C 组：深度界（R2）"); W("="*100)
chk("T_crit = 15,414.9", "15,414.9", f"{TC:.4f}", abs(TC - 15414.94) < 0.01)
tm = [R[t]["Tmax"] for t, _ in MATS]
chk("子集最深 spot 40–380", "40–380", f"{min(tm):.0f}–{max(tm):.0f}", min(tm) == 40 and max(tm) == 380)
rat = [TC / R[t]["Tmax"] for t, _ in MATS]
chk("T_crit/T_max = 41–385×", "41–385×", f"{min(rat):.1f}–{max(rat):.1f}",
    round(min(rat)) == 41 and round(max(rat)) == 385)
libs = [R[t]["lib"] for t, _ in MATS]
chk("库深跨 1,576 倍（8,400→13,234,549）", "1,576×", f"{max(libs)/min(libs):.1f}×",
    abs(max(libs)/min(libs) - 1575.5) < 1)
chk("比值跨度 9.5 倍", "9.5×", f"{max(rat)/min(rat):.3f}×", abs(max(rat)/min(rat) - 9.5) < 0.05)
chk("SCC 库深 859× T_crit", "859×", f"{R['SCC']['lib']/TC:.1f}×", abs(R["SCC"]["lib"]/TC - 858.6) < 1)
chk("SCC 子集最深 380，低 41×", "380 / 41×",
    f"{R['SCC']['Tmax']:.0f} / {TC/R['SCC']['Tmax']:.1f}×", R["SCC"]["Tmax"] == 380)
chk("CM 1,204 个 cuTAR 且不含其他", "1,204 / cuTAR-only",
    f"{R['CM_pacbio']['n_cu']:,} / all_cu={R['CM_pacbio']['all_cu']}",
    R["CM_pacbio"]["n_cu"] == 1204 and R["CM_pacbio"]["all_cu"])
chk("CM 任一 spot 最多 40 计数", "40", f"{R['CM_pacbio']['Tmax']:.0f}", R["CM_pacbio"]["Tmax"] == 40)

W("\n" + "="*100); W("D 组：后果（R3）"); W("="*100)
def sp_corr(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])
spes = []
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    Xd = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    S = Xd[:, is_cu]; T = S.sum(1)
    L = np.log1p(S / np.where(T == 0, 1.0, T)[:, None] * 1e4)
    spes.append(sp_corr((L > 0.5).sum(1), T))
    del a, X, Xd, S, L
chk("Spearman（过阈特征数, 子集深度）ρ 下界 = 0.944（下限来自 CM）", "0.9443",
    f"{min(spes):.4f}–{max(spes):.4f}", abs(min(spes) - 0.9443) < 0.001)

W("\n" + "="*100); W("E 组：特征预算（R5）"); W("="*100)
# 稿中明确写出的分数（用于逐项比对，不再用 True 占位）
MS_PC = {"BCC": (5197, 14735), "SCC": (6570, 14562), "HNC": (7742, 13850), "KidneyCancer": (7592, 14530)}
MS_CU = {"BCC": (5, 3759), "SCC": (18, 5322), "HNC": (26, 3723), "KidneyCancer": (7, 5506)}
for t in ["BCC", "SCC", "HNC", "KidneyCancer"]:
    d = R[t]
    chk(f"{t} 蛋白编码 10% 保留 = 稿中 {MS_PC[t][0]:,}/{MS_PC[t][1]:,}",
        f"{MS_PC[t][0]:,}/{MS_PC[t][1]:,}",
        f"{d['pc10']:,}/{d['n_pc']:,} = {100*d['pc10']/d['n_pc']:.2f}%",
        (d["pc10"], d["n_pc"]) == MS_PC[t])
    chk(f"{t} cuTAR 10% 保留 = 稿中 {MS_CU[t][0]}/{MS_CU[t][1]:,}",
        f"{MS_CU[t][0]}/{MS_CU[t][1]:,}",
        f"{d['cu10']}/{d['n_cu']:,} = {100*d['cu10']/d['n_cu']:.3f}%",
        (d["cu10"], d["n_cu"]) == MS_CU[t])
chk("蛋白编码保留区间 = 35.3–55.9%", "35.3–55.9%",
    f"{min(100*R[t]['pc10']/R[t]['n_pc'] for t in MS_PC):.2f}–{max(100*R[t]['pc10']/R[t]['n_pc'] for t in MS_PC):.2f}%",
    round(min(100*R[t]["pc10"]/R[t]["n_pc"] for t in MS_PC), 1) == 35.3
    and round(max(100*R[t]["pc10"]/R[t]["n_pc"] for t in MS_PC), 1) == 55.9)
chk("cuTAR 保留区间 = 0.13–0.70%", "0.13–0.70%",
    f"{min(100*R[t]['cu10']/R[t]['n_cu'] for t in MS_CU):.3f}–{max(100*R[t]['cu10']/R[t]['n_cu'] for t in MS_CU):.3f}%",
    round(min(100*R[t]["cu10"]/R[t]["n_cu"] for t in MS_CU), 2) == 0.13
    and round(max(100*R[t]["cu10"]/R[t]["n_cu"] for t in MS_CU), 2) == 0.70)
chk("倍数区间 = 80–411", "80–411",
    f"{min(R[t]['pc10']/R[t]['n_pc']/(R[t]['cu10']/R[t]['n_cu']) for t in MS_PC):.0f}–"
    f"{max(R[t]['pc10']/R[t]['n_pc']/(R[t]['cu10']/R[t]['n_cu']) for t in MS_PC):.0f}×",
    round(min(R[t]["pc10"]/R[t]["n_pc"]/(R[t]["cu10"]/R[t]["n_cu"]) for t in MS_PC)) == 80
    and round(max(R[t]["pc10"]/R[t]["n_pc"]/(R[t]["cu10"]/R[t]["n_cu"]) for t in MS_PC)) == 411)

W("\n" + "="*100); W("F 组：单细胞对象"); W("="*100)
scp = os.path.join(ROOT, r"data\SPanC-Lnc-pan\Melanoma_scRNA.h5ad")
a = ad.read_h5ad(scp, backed="r")
names = [str(x) for x in a.var.index]
ic = np.where(np.array([n.upper().startswith("CUTAR") for n in names], bool))[0]
ncell = a.n_obs
det = np.zeros(len(ic), np.int64); cth = {}
ctv = a.obs["cell_types"].astype(str).values
cats = sorted(set(ctv))
for c in cats: cth[c] = np.zeros(len(ic), np.int64)
CH = 50
for s in range(0, len(ic), CH):
    blk = a[:, ic[s:s+CH]].to_memory().X
    blk = blk.toarray() if sp.issparse(blk) else np.asarray(blk)
    blk = blk.astype(np.float64)
    det[s:s+CH] = (blk > 0).sum(0)
    for c in cats:
        cth[c][s:s+CH] = (blk[ctv == c] > 0).sum(0)
pct = 100 * det / ncell
chk("单细胞 cuTAR 数 1,314", "1,314", f"{len(ic):,}", len(ic) == 1314)
chk("中位检出细胞比例 1.05%", "1.05%", f"{np.median(pct):.3f}%", abs(np.median(pct) - 1.05) < 0.01)
chk("≥10% 细胞的 cuTAR = 22", "22", f"{int((pct>=10).sum())}", int((pct >= 10).sum()) == 22)
nf1 = 0
for i in range(len(ic)):
    k = 0
    for c in cats:
        if cth[c][i] / max(1, int((ctv == c).sum())) >= 0.10:
            k += 1
    if k == 1: nf1 += 1
chk("仅 1 类细胞 ≥10% 的 cuTAR = 55", "55", f"{nf1}", nf1 == 55)

W("\n" + "="*100)
W(f"汇总：{sum(1 for _,m in CHK if m)}/{len(CHK)} 项与稿中一致")
W("="*100)
for c, m in CHK:
    if not m: W(f"  不一致：{c}")
open(os.path.join(ROOT, "results", "MANUSCRIPT_AUDIT.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\MANUSCRIPT_AUDIT.txt")
