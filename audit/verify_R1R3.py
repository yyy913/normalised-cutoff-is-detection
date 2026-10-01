# -*- coding: utf-8 -*-
"""
verify_R1R3.py — 对 R1–R3 中每一个数字与断言做独立核查
原则：全部**从原始 h5ad 重新计算**，不读回先前脚本产出的 JSON（避免错误被继承）。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import anndata as ad
import scipy.sparse as sp

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

OUT = []
def W(s=""):
    print(s, flush=True); OUT.append(s)

CHK = []
def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail))
    W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")

R = {}
for tag, rel in MATS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    # cuTAR-only 矩阵里 is_cu 应全为 True
    Xd = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    S = Xd[:, is_cu]
    T = S.sum(1)                      # 子集深度
    L = np.log1p(S / np.where(T == 0, 1.0, T)[:, None] * 1e4)
    nz = S > 0
    R[tag] = dict(
        n_spot=int(a.n_obs), n_feat=int(a.n_vars), n_cu=int(is_cu.sum()),
        all_cu=bool(is_cu.all()),
        n_nonzero=int(nz.sum()),
        m1=int((L > 0.5).sum()), m2=int(nz.sum()),
        mismatch=int(((L > 0.5) != nz).sum()),
        min_nz_lognorm=float(L[nz].min()) if nz.any() else 0.0,
        T_max=float(T.max()), lib_total=float(Xd.sum()),
    )
    del a, X, Xd, S, L

W("=" * 104)
W("R1 核查")
W("=" * 104)
chk("cuTAR 子集恒等对全部 10 个矩阵成立（其中 5 个为混合矩阵，判据施加于子集）",
    all(R[t]["mismatch"] == 0 for t, _ in MATS),
    f"cuTAR-only {sum(R[t]['all_cu'] for t,_ in MATS)}/10，混合 {10-sum(R[t]['all_cu'] for t,_ in MATS)}/10")
tot_mm = sum(R[t]["mismatch"] for t, _ in MATS)
chk("(L>0.5) == (count>0) 逐元素，零不匹配", tot_mm == 0, f"总不匹配 = {tot_mm}")
nzs = [R[t]["n_nonzero"] for t, _ in MATS]
chk("非零元素范围 = 7,326–52,217（下限来自 Melanoma）",
    min(nzs) == 7326 and max(nzs) == 52217, f"{min(nzs):,} – {max(nzs):,}")
mn = [R[t]["min_nz_lognorm"] for t, _ in MATS]
chk("最小非零 logNorm = 3.31–5.53（四舍五入到 2 位）",
    round(min(mn), 2) == 3.31 and round(max(mn), 2) == 5.53,
    f"{min(mn):.4f} – {max(mn):.4f}")
chk("t = 1, 2, 3 均低于所有矩阵的最小非零值（即同样惰性）",
    all(R[t]["min_nz_lognorm"] > 3.0 for t, _ in MATS),
    f"最小者 {min(mn):.4f} > 3")

# 代数式：L > t  <=>  c > T(e^t-1)/1e4  ；检验一个矩阵的全部非零元素
a = ad.read_h5ad(os.path.join(ROOT, MATS[0][1]))
X = np.asarray(a.X, dtype=np.float64) if not sp.issparse(a.X) else a.X.toarray()
T = X.sum(1); L = np.log1p(X / np.where(T == 0, 1.0, T)[:, None] * 1e4)
ok = True
for t in (0.5, 1.0, 2.0, 3.0):
    lhs = (L > t).astype(int)
    rhs = (X > (T * (np.exp(t) - 1) / 1e4)[:, None]).astype(int)
    ok &= bool((lhs == rhs).all())
chk("代数式 L>t ⟺ c > T(e^t−1)/1e4（CM 全部元素，t=0.5/1/2/3）", ok)
del a, X, T, L

W("\n" + "=" * 104)
W("R2 核查")
W("=" * 104)
TC = 1e4 / (np.exp(0.5) - 1)
chk("T_crit = 10^4/(e^0.5−1) = 15,414.9", abs(TC - 15414.9) < 0.1, f"{TC:.4f}")
tmax = [R[t]["T_max"] for t, _ in MATS]
chk("cuTAR 子集最深 spot = 40–380 read", min(tmax) == 40 and max(tmax) == 380,
    f"{min(tmax):.0f} – {max(tmax):.0f}")
ratios = [TC / R[t]["T_max"] for t, _ in MATS]
chk("T_crit / T_max = 41–385×", round(min(ratios)) == 41 and round(max(ratios)) == 385,
    f"{min(ratios):.1f} – {max(ratios):.1f}")
libs = [R[t]["lib_total"] for t, _ in MATS]
chk("库深跨 1,576 倍（8,400 → 13,234,549）",
    min(libs) == 8400 and max(libs) == 13234549,
    f"{min(libs):,.0f} → {max(libs):,.0f}  (倍数 {max(libs)/min(libs):.1f})")
chk("比值跨度仅 9.5 倍",
    abs((max(ratios) / min(ratios)) - 9.5) < 0.1, f"{max(ratios)/min(ratios):.3f}")
chk("SCC 库深 = 859 × T_crit", abs(R["SCC"]["lib_total"] / TC - 859) < 1,
    f"{R['SCC']['lib_total']/TC:.1f}")
chk("SCC 子集最深 = 380 read，低于界 41 倍", abs(TC / R["SCC"]["T_max"] - 40.6) < 0.2,
    f"{TC/R['SCC']['T_max']:.1f}")
chk("CM 为 cuTAR-only 且特征数 = 1,204",
    R["CM_pacbio"]["all_cu"] and R["CM_pacbio"]["n_cu"] == 1204,
    f"n_feat={R['CM_pacbio']['n_feat']}, n_cu={R['CM_pacbio']['n_cu']}")
chk("CM 任意 spot 最多 40 个计数", R["CM_pacbio"]["T_max"] == 40)

W("\n" + "=" * 104)
W("R3 核查")
W("=" * 104)
ari = {d["tag"]: d for d in json.load(open(os.path.join(ROOT, "results", "RECHECK.json"),
                                          encoding="utf-8"))["ari"]}
sp = [ari[t]["sp_sub"] for t, _ in MATS]
chk("Spearman(过阈特征数, 子集深度) ρ 的下界 = 0.944，且最大值 = 0.9995",
    abs(min(sp) - 0.9443) < 0.001 and abs(max(sp) - 0.9995) < 0.001,
    f"{min(sp):.4f} – {max(sp):.4f}   （来源 RECHECK.json，非本次重算）")

W("\n" + "=" * 104)
W(f"汇总：{sum(1 for _,c,_ in CHK if c)}/{len(CHK)} 项通过")
W("=" * 104)
for n, c, d in CHK:
    if not c:
        W(f"  未通过：{n}  ({d})")
open(os.path.join(ROOT, "results", "VERIFY_R1R3.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\VERIFY_R1R3.txt")
