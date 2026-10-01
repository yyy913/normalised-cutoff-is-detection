# -*- coding: utf-8 -*-
"""
verify_uncovered_claims.py — 核查**从未被任何脚本覆盖**的稿中数字断言

`verify_manuscript_coverage.py` 的反向比对查出一批数字从未被核过。本脚本逐条独立重算：

  A  ¶1「the median cuTAR is detected in 0.08–0.80% of spots」——10 个矩阵
  B  跨平台段：「five further matrices … 507,684 / 487,552 / 4,669 spots」「zero mismatching
     entries」「approximately one million spots」「failed in a single Visium section in which
     7.2% of spots exceeded the bound」——`generality_test.py` **根本没有算过掩码恒等**。
  C  R4 第三条限制里我新写进去的检出率匹配对照（+0.017…+0.037 / −0.019 / −0.060）——
     原件 J4 每个 cuTAR 只抽 **1** 个随机匹配基因，抽样噪声可能让正负号不稳定。
  D  Methods「top 1,500 protein-coding genes by median expression」这一参照划分。

掩码恒等的判据（无需构造稠密 L）：
  L > 0.5  ⟺  c / T * 1e4 > e^0.5 − 1  ⟺  c > T · δ,   δ = (e^0.5 − 1)/1e4 = 1.6487212707e-4
  故「掩码 ≠ 检出」只发生在 c > 0 且 c ≤ T·δ 的**非零元素**上。可只在稀疏非零上精确计数。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, gzip
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from scipy.spatial import cKDTree

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
DELTA = (np.exp(0.5) - 1) / 1e4          # = 1.6487212707e-4
TCRIT = 1e4 / (np.exp(0.5) - 1)          # = 15414.94
OUT, CHK, JS = [], [], {}


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "VERIFY_UNCOVERED.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail))
    W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")


def sparse_X(path, backed=False):
    a = ad.read_h5ad(path, backed="r" if backed else None)
    return a


TEN = [("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad"),
       ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad"),
       ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad"),
       ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad"),
       ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad"),
       ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad"),
       ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
       ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
       ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
       ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad")]

EXT = [("GSE280315 P1CRC (Visium HD)", r"data\GSE280315\h5ad\GSM8594567_P1CRC.h5ad"),
       ("GSE280315 P3NAT (Visium HD)", r"data\GSE280315\h5ad\GSM8594570_P3NAT.h5ad"),
       ("GSE225857 C1 (Visium)", r"data\GSE225857\h5ad\GSM7058756_C1.h5ad"),
       ("GSE225857 L1 (Visium)", r"data\GSE225857\h5ad\GSM7058760_L1.h5ad")]
# 平台归属核查（2026-09-24 更正）：GSE280315 的两个切片曾被写成 Stereo-seq，实为 Visium HD。三条依据：
#   ① 两切片 n_vars 均 = 18,085（Visium HD 人转录组探针集大小；Stereo-seq 无探针、通常 25,000+）；
#   ② 原始下载每个样本都带 *_probe_set.csv.gz（探针法）；
#   ③ GEO 官方标题 "…high definition spatial profiling [VisiumHD]"，经 NCBI eutils 实查。

# =====================================================================================
W("=" * 104)
W("PART A   ¶1「median cuTAR detected in 0.08–0.80% of spots」")
W("=" * 104)
W(f"  {'矩阵':<18}{'cuTAR':>7}{'中位检出率':>12}{'最高检出率':>12}{'spot':>9}")
med, mx, nspot = [], [], []
tot_spot_ten = 0
for tag, rel in TEN:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    names = [str(v) for v in a.var.index]
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    Xc = X[:, is_cu].tocsc()
    det = np.diff(Xc.indptr) / a.n_obs          # 每列非零数 / spot 数
    med.append(float(np.median(det))); mx.append(float(det.max()))
    nspot.append(int(a.n_obs)); tot_spot_ten += int(a.n_obs)
    W(f"  {tag:<18}{int(is_cu.sum()):>7,}{100*np.median(det):>11.3f}%{100*det.max():>11.2f}%"
      f"{int(a.n_obs):>9,}")
    del a, X, Xc
W(f"\n  中位检出率跨 10 矩阵 = {100*min(med):.3f}% – {100*max(med):.3f}%")
W(f"  最高检出率跨 10 矩阵 = {100*min(mx):.2f}% – {100*max(mx):.2f}%")
W(f"  10 矩阵 spot 合计 = {tot_spot_ten:,}")
chk("¶1「median cuTAR detected in 0.08–0.80% of spots」复现",
    round(100 * min(med), 2) == 0.08 and round(100 * max(med), 2) == 0.80,
    f"{100*min(med):.3f}%–{100*max(med):.3f}% → 四舍五入 {100*min(med):.2f}–{100*max(med):.2f}")
JS["partA"] = dict(median_range=[100 * min(med), 100 * max(med)],
                   maxdet_range=[100 * min(mx), 100 * max(mx)], spots_ten=tot_spot_ten)

# =====================================================================================
W("\n" + "=" * 104)
W("PART B   跨平台段：掩码恒等（原件从未计算）")
W("=" * 104)
W("  情形 A = 稀有特征子集按该子集归一化；情形 B = 按全矩阵归一化。")
W(f"  δ = (e^0.5−1)/1e4 = {DELTA:.10e}；T_crit = {TCRIT:.2f}\n")
W(f"  {'数据集':<30}{'spot':>9}{'稀有特征':>9}{'A 不匹配':>10}{'B 不匹配':>10}"
  f"{'B 超界spot%':>12}{'B t*':>8}")
rows = []
for tag, rel in EXT:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n = a.n_obs
    Xc = X.tocsc()
    fd = np.diff(Xc.indptr)
    mask = (fd / n) < 0.02                      # 稀有特征：检出率 <2%
    Xs = X[:, mask].tocsr()
    Tsub = np.asarray(Xs.sum(1)).ravel()
    Tfull = np.asarray(X.sum(1)).ravel()
    coo = Xs.tocoo()
    mmA = int(np.count_nonzero(coo.data <= Tsub[coo.row] * DELTA))
    mmB = int(np.count_nonzero(coo.data <= Tfull[coo.row] * DELTA))
    over = float((Tfull > TCRIT).mean())
    tstar = float(np.log1p(1e4 / Tfull.max()))
    nz = int(coo.nnz)
    rows.append(dict(tag=tag, n_spot=int(n), n_rare=int(mask.sum()), mm_A=mmA, mm_B=mmB,
                     nnz=nz, pct_over_bound=100 * over, t_star_B=tstar,
                     tstar_A=float(np.log1p(1e4 / Tsub.max()))))
    W(f"  {tag:<30}{n:>9,}{int(mask.sum()):>9,}{mmA:>10,}{mmB:>10,}{100*over:>11.1f}%{tstar:>8.3f}")
    del a, X, Xc, Xs, coo
JS["partB"] = rows

W("\n  逐条核对稿中跨平台段的说法：")
okA = [r for r in rows if r["mm_A"] == 0]
okB = [r for r in rows if r["mm_B"] == 0]
byN = {r["n_spot"]: r for r in rows}
W(f"\n    外部矩阵共 {len(rows)} 个；情形 A（本文口径）零不匹配 = {len(okA)} 个；"
  f"情形 B 零不匹配 = {len(okB)} 个")
for r in rows:
    W(f"      {r['tag']:<30} spot={r['n_spot']:>9,}  A不匹配={r['mm_A']:>8,}  B不匹配={r['mm_B']:>8,}"
      f"  B超界={r['pct_over_bound']:>5.1f}%")
for want in (507684, 487552, 4669, 2054):
    W(f"    稿中/原表提到的 spot 数 {want:,}：" +
      (f"命中 {byN[want]['tag']}" if want in byN else "**未出现**"))
chk("稿中「507,684 / 487,552 / 4,669 / 2,054 spots」四个数都对应真实矩阵",
    all(w in byN for w in (507684, 487552, 4669, 2054)), "P1CRC / P3NAT / L1 / C1")
chk("外部矩阵实际为 **4** 个（旧稿写 five further matrices，已更正）", len(rows) == 4,
    f"实测 {len(rows)} 个：P1CRC, P3NAT, C1, L1")
chk("情形 A（本文口径）下四个外部矩阵掩码与检出**全部零不匹配**（现稿写法）",
    len(okA) == 4, f"{len(okA)}/4")
chk("情形 B 下只有 2 个零不匹配（C1 与 L1 失效）——故旧稿「仅一个 Visium 切片失效」不成立",
    len(okB) == 2, f"{len(okB)}/4：{', '.join(r['tag'] for r in okB)}")
sumA4 = sum(r["n_spot"] for r in okA)
W(f"\n    四个（情形 A 通过）矩阵 spot 合计 = {sumA4:,}；加 10 矩阵 {tot_spot_ten:,} = {sumA4+tot_spot_ten:,}")
chk("「together with the ten matrices … approximately 1.02 million spots」（现稿写法）",
    abs((sumA4 + tot_spot_ten) - 1.02e6) < 1.2e4,
    f"{sumA4:,} + {tot_spot_ten:,} = {sumA4+tot_spot_ten:,}")
c1 = [r for r in rows if "C1" in r["tag"]][0]
W(f"\n    C1（旧稿点的失效矩阵）：子集内不匹配 {c1['mm_B']:,}；深度超 T_crit 的 spot 占 "
  f"{c1['pct_over_bound']:.1f}%")
chk("旧稿「7.2% of spots exceeded the bound」无法复现；现稿改为实测的 69.7% 与 32,377",
    abs(c1["pct_over_bound"] - 69.7) < 0.1 and c1["mm_B"] == 32377,
    f"{c1['pct_over_bound']:.1f}% / {c1['mm_B']:,}")
JS["partB_c1"] = c1

# =====================================================================================
W("\n" + "=" * 104)
W("PART C   R4 第三条限制里新写的检出率匹配对照，抽样稳定性")
W("=" * 104)
W("  原件 J4 每个 cuTAR 只抽 1 个随机匹配基因；此处每个 cuTAR 抽 30 个，看均值与离散度。")
K, DET_MIN, NDRAW = 6, 0.05, 30
MIT = [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
       ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
       ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad"),
       ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
       ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad")]


def lag(nb, Z, k=K, chunk=512):
    n = Z.shape[0]; rows = np.repeat(np.arange(n), k); flat = nb.ravel()
    out = np.zeros_like(Z)
    for s in range(0, Z.shape[1], chunk):
        np.add.at(out[:, s:s + chunk], rows, Z[:, s:s + chunk][flat] / k)
    return out


j4 = {}
rngC = np.random.default_rng(20260924)
for tag, rel in MIT:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n = a.n_obs
    names = np.array([str(v) for v in a.var.index])
    is_cu = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    Xc = X.tocsc(); det = np.diff(Xc.indptr) / n
    icu = np.where(is_cu & (det >= DET_MIN))[0]
    ipc = np.where((~is_cu) & (det >= DET_MIN))[0]
    rows_nb = X[:, np.r_[icu, ipc]].toarray().astype(float)
    coords = np.asarray(a.obsm["spatial"], float)
    _, it = cKDTree(coords).query(coords, k=K + 1); nb = it[:, 1:]

    def cp(V):
        t = V.sum(1); t = np.where(t == 0, 1.0, t)
        return np.log1p(np.divide(V * 1e4, t[:, None]))
    LA = cp(rows_nb[:, :len(icu)]); LB = cp(rows_nb[:, len(icu):])
    Zx = LA - LA.mean(0); Zy = LB - LB.mean(0)
    WZy = lag(nb, Zy)                       # 预计算，使任意候选列 j 的滞后即其列
    ny = np.linalg.norm(Zy, axis=0)
    IA = lag(nb, Zx).T @ Zy / np.outer(np.linalg.norm(Zx, axis=0), ny)
    best_obs = IA.max(1)
    det_cu, det_pc = det[icu], det[ipc]
    j_det = {int(c): i for i, c in enumerate(ipc)}
    means, sds, allv = [], [], []
    for i in range(len(icu)):
        d = det_cu[i]
        pool = np.where(np.abs(det_pc - d) <= 0.25 * d)[0]
        if len(pool) < 5:
            pool = np.array([int(np.argmin(np.abs(det_pc - d)))])
        cand = ipc[rngC.choice(pool, size=min(NDRAW, len(pool)), replace=len(pool) < NDRAW)]
        vals = []
        for c in cand:
            z = Zx[:, ]  # 占位，避免误用
            col = cp(rows_nb[:, len(icu) + j_det[int(c)]]) if False else LB[:, j_det[int(c)]]
            zj = col - col.mean(); nzj = np.linalg.norm(zj)
            if nzj == 0:
                continue
            Wzj = lag(nb, zj[:, None])[:, 0]
            row = (Wzj @ Zy) / (nzj * ny)
            keep = np.ones(len(ipc), bool); keep[j_det[int(c)]] = False
            vals.append(float(row[keep].max()))
        means.append(np.mean(vals)); sds.append(np.std(vals, ddof=1)); allv += vals
    bo, bn = float(best_obs.mean()), float(np.mean(means))
    W(f"  {tag:<14} 真 cuTAR 最佳伙伴均值 {bo:.4f}   匹配伪 cuTAR（30 抽样/个）均值 {bn:.4f}"
      f"   差 {bo-bn:+.4f}")
    W(f"                 30 次抽样的**逐 cuTAR 标准差**中位 {np.median(sds):.4f}"
      f"（即单次抽样的典型噪声量级）；两次独立种子下的均值差见下")
    # 第二套种子，检验正负号稳定性
    rngC2 = np.random.default_rng(31337)
    means2 = []
    for i in range(len(icu)):
        d = det_cu[i]
        pool = np.where(np.abs(det_pc - d) <= 0.25 * d)[0]
        if len(pool) < 5:
            pool = np.array([int(np.argmin(np.abs(det_pc - d)))])
        cand = ipc[rngC2.choice(pool, size=min(NDRAW, len(pool)), replace=len(pool) < NDRAW)]
        vals = []
        for c in cand:
            col = LB[:, j_det[int(c)]]
            zj = col - col.mean(); nzj = np.linalg.norm(zj)
            if nzj == 0:
                continue
            Wzj = lag(nb, zj[:, None])[:, 0]
            row = (Wzj @ Zy) / (nzj * ny)
            keep = np.ones(len(ipc), bool); keep[j_det[int(c)]] = False
            vals.append(float(row[keep].max()))
        means2.append(np.mean(vals))
    bn2 = float(np.mean(means2))
    W(f"                 第二套种子下的伪 cuTAR 均值 {bn2:.4f}   差 {bo-bn2:+.4f}")
    j4[tag] = dict(best_cu=bo, best_null_30=bn, diff=bo - bn, diff_seed2=bo - bn2,
                   median_per_cutar_sd=float(np.median(sds)))
    del a, X, Xc, rows_nb, LA, LB, Zx, Zy, WZy
JS["partC_j4"] = j4
W("\n  现稿写「attains 62–105% of the true cuTAR's best-partner strength」（旧稿的 +0.017…−0.060 已废弃）：")
for tag in ["SCC", "HNC", "BCC", "KidneyCancer", "Melanoma"]:
    W(f"    {tag:<14} 差 = {j4[tag]['diff']:+.4f}（第二种子 {j4[tag]['diff_seed2']:+.4f}）"
      f"  单抽样噪声量级 {j4[tag]['median_per_cutar_sd']:.4f}"
      f"  对照/真 = {100*j4[tag]['best_null_30']/j4[tag]['best_cu']:.1f}%")
rat = {t: j4[t]["best_null_30"] / j4[t]["best_cu"] for t in j4}
lo, hi = min(rat.values()), max(rat.values())
chk("匹配对照达到真 cuTAR 最佳伙伴强度的 62–105%（现稿写法）",
    round(100 * lo) == 62 and round(100 * hi) == 105,
    "  ".join(f"{t}:{100*rat[t]:.1f}%" for t in rat))
chk("旧稿「另两个反超（−0.019、−0.060）」不成立：30 抽样下五中四为正，仅 Melanoma 略负且幅度在噪声内",
    sum(1 for t in j4 if j4[t]["diff"] > 0) == 4 and j4["Melanoma"]["diff"] > -0.02,
    "  ".join(f"{t}:{j4[t]['diff']:+.4f}" for t in j4))
chk("正负号在第二套种子下不变",
    all((j4[t]["diff"] > 0) == (j4[t]["diff_seed2"] > 0) for t in j4),
    "  ".join(f"{t}:{j4[t]['diff']:+.4f}/{j4[t]['diff_seed2']:+.4f}" for t in j4))

# =====================================================================================
W("\n" + "=" * 104)
W("PART D   Methods「top 1,500 protein-coding genes by median expression」")
W("=" * 104)
W("  RECHECK.json 未记录参照划分规模，故改为：①核对产生它的源码；②重算该划分规模是否确为 1,500。")
try:
    rc = json.load(open(os.path.join(RES, "RECHECK.json"), encoding="utf-8"))
    ari = {d["tag"]: d for d in rc["ari"]}
    W("  RECHECK.json 的 ari 记录键：" + ", ".join(sorted(ari["SCC"].keys())))
    src = open(os.path.join(ROOT, "audit", "recheck2.py"), encoding="utf-8").read()
    line = [l.strip() for l in src.split("\n") if "[:1500]" in l and "pc_idx" in l]
    W(f"  audit/recheck2.py 中的参照划分定义：{line}")
    chk("源码里的参照划分确为「蛋白编码基因按中位表达取前 1,500」",
        any("pc_idx" in l and "1500" in l for l in line), str(line))
    # 重算：该划分规模是否确实能被 1,500 填满
    gi = {}
    with gzip.open(os.path.join(ROOT, r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"),
                   "rt", errors="replace") as f:
        h = f.readline().rstrip("\n").split("\t")
        it, isy = h.index("type_of_gene"), h.index("Symbol")
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) > max(it, isy):
                gi[p[isy].upper()] = p[it]
    n_pc = {}
    for tag, rel in [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"), ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad"),
                     ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad"),
                     ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad")]:
        a = ad.read_h5ad(os.path.join(ROOT, rel), backed="r")
        nm = [str(v) for v in a.var.index]
        n_pc[tag] = sum(1 for s in nm if not s.upper().startswith("CUTAR")
                        and gi.get(s.upper(), "") == "protein-coding")
        del a
    W("  各矩阵的蛋白编码基因数：" + "  ".join(f"{t}={v:,}" for t, v in n_pc.items()))
    chk("每个矩阵的蛋白编码基因数都 ≥1,500（故「top 1,500」取满且划分规模确为 1,500）",
        all(v >= 1500 for v in n_pc.values()), f"最小 {min(n_pc.values()):,}")
    JS["partD"] = dict(source_line=line, n_pc=n_pc)
except Exception as e:
    chk("PART D 可核", False, f"{type(e).__name__}: {e}")

W("\n" + "=" * 104)
W(f"汇总：{sum(1 for _, c, _ in CHK if c)}/{len(CHK)} 项通过")
W("=" * 104)
for nm, c, dt in CHK:
    if not c:
        W(f"  未通过：{nm}  ({dt})")
open(os.path.join(RES, "VERIFY_UNCOVERED.txt"), "w", encoding="utf-8").write("\n".join(OUT))
json.dump(JS, open(os.path.join(RES, "VERIFY_UNCOVERED.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1, default=str)
print("\n[written] results\\VERIFY_UNCOVERED.txt / .json")
