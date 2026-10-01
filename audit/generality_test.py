# -*- coding: utf-8 -*-
"""
generality_test.py — 把论文的代数条件实测到其他数据集/平台

条件：对阈值 t，掩码 L>t 与 count>0 等价  <=>  T_max < 10^4/(e^t - 1)
其中 T 是**归一化所用的分母**：
  情形A 子集归一化 -> T = 该子集在 spot 上的计数和
  情形B 全矩阵归一化 -> T = 该 spot 在全矩阵上的计数和

输出：每个数据集的可容许阈值上界 t* = ln(1 + 10^4 / T_max)，以及 0.5 是否在其中。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, numpy as np, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import anndata as ad, scipy.sparse as sp

ROOT = _cfg.WORKSPACE + r""
OUT = []
def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(ROOT, "results", "GENERALITY_test.txt"), "w", encoding="utf-8").write("\n".join(OUT))

CASES = [
    ("SPanC-Lnc CM (cuTAR 子集)",     r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad",        "cuTAR"),
    ("SPanC-Lnc SCC (cuTAR 子集)",    r"data\SPanC-Lnc-pan\SCC.h5ad",             "cuTAR"),
    ("GSE280315 P1CRC (全矩阵)",      r"data\GSE280315\h5ad\GSM8594567_P1CRC.h5ad", "all"),
    ("GSE280315 P3NAT (全矩阵)",      r"data\GSE280315\h5ad\GSM8594570_P3NAT.h5ad", "all"),
    ("GSE225857 C1 (全矩阵)",         r"data\GSE225857\h5ad\GSM7058756_C1.h5ad",  "all"),
    ("GSE225857 L1 (全矩阵)",         r"data\GSE225857\h5ad\GSM7058760_L1.h5ad",  "all"),
]

def tstar(Tmax):
    """可容许阈值上界：t* = ln(1 + 1e4/Tmax)"""
    return float(np.log1p(1e4 / Tmax)) if Tmax > 0 else np.inf

W("=" * 112)
W("A 组：情形 A —— 稀有特征子集按该子集归一化")
W("=" * 112)
W(f"  {'数据集':<28}{'spot':>9}{'特征':>8}{'子集最深 spot':>14}{'可容许 t*':>12}{'0.5 可容许?':>12}")
A = []
for tag, rel, mode in CASES:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    names = [str(x) for x in a.var.index]
    if mode == "cuTAR":
        mask = np.array([n.upper().startswith("CUTAR") for n in names], bool)
    else:
        # 情形A：以"稀有基因"作为子集 —— 检出率 <2% 的特征
        if sp.issparse(X):
            fd = np.diff(X.tocsc().indptr)
        else:
            fd = (np.asarray(X) > 0).sum(0)
        mask = (fd / a.n_obs) < 0.02
    if sp.issparse(X):
        sub = np.asarray(X[:, mask].sum(1)).ravel()
    else:
        sub = np.asarray(X, dtype=np.float64)[:, mask].sum(1)
    Tmax = float(sub.max())
    ts = tstar(Tmax)
    W(f"  {tag:<28}{a.n_obs:>9,}{int(mask.sum()):>8,}{Tmax:>14.0f}{ts:>12.3f}"
      f"{('是' if 0.5 < ts else '**否**'):>12}")
    A.append(dict(tag=tag, n_spot=int(a.n_obs), n_feat=int(mask.sum()), n_all=int(a.n_vars),
                  Tmax_sub=Tmax, t_star=ts, ok05=bool(0.5 < ts)))
    del a, X

W("\n" + "=" * 112)
W("B 组：情形 B —— 稀有特征按全矩阵归一化")
W("=" * 112)
W(f"  {'数据集':<28}{'spot':>9}{'中位深度':>11}{'最深 spot':>12}{'可容许 t*':>12}{'0.5 可容许?':>12}")
B = []
for tag, rel, mode in CASES:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    if sp.issparse(X):
        tot = np.asarray(X.sum(1)).ravel()
    else:
        tot = np.asarray(X, dtype=np.float64).sum(1)
    Tmax = float(tot.max()); ts = tstar(Tmax)
    W(f"  {tag:<28}{a.n_obs:>9,}{np.median(tot):>11.0f}{Tmax:>12,.0f}{ts:>12.3f}"
      f"{('是' if 0.5 < ts else '**否**'):>12}")
    B.append(dict(tag=tag, median_depth=float(np.median(tot)), Tmax_full=Tmax,
                  t_star=ts, ok05=bool(0.5 < ts)))
    del a, X

W("\n" + "=" * 112)
W("判读")
W("=" * 112)
W("  情形 A（子集归一化）：子集深度必然小 -> 可容许 t* 很高 -> 0.5 一律可容许，")
W("                        即'阈值≡检出'这一恒等在**所有**含稀有特征的零膨胀计数数据上成立。")
W("  情形 B（全矩阵归一化）：取决于最深 spot。深度越浅的平台/矩阵，恒等成立越普遍。")
json.dump({"caseA": A, "caseB": B},
          open(os.path.join(ROOT, "results", "GENERALITY_test.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\GENERALITY_test.txt/.json")
