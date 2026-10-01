# -*- coding: utf-8 -*-
"""
verify_supp_tableS2.py — 逐格重算补充材料表 S2（情形 A / 情形 B）的全部单元格

表 S2 声明（情形 B 块）：
  | 数据集      | 中位深度 | 最深 spot | t*    | 0.5 可容许 | 超界 spot 占比 | 不匹配 |
  | CM          |     0    | 40        | 5.525 | 是        | 100.0%        | 0     |
  | SCC         | 5,493    | 46,963    | 0.193 | 否        | 0.0%          | 0     |
  | P1CRC       |   147    | 2,104     | 1.750 | 是        | 0.0%          | 0     |
  | P3NAT       |    23    | 5,560     | 1.029 | 是        | 0.0%          | 0     |
  | C1          | 25,063   | 213,355   | 0.046 | 否        | 7.2%          | 44,891|
  | L1          | 3,942    | 70,920    | 0.132 | 否        | 0.0%          | 0     |

本脚本对每个数据集、两种「不匹配」口径分别重算：
  口径 S（子集）  : 只在「特征子集」的非零元素上判 c ≤ T·δ
  口径 W（全矩阵）: 在全矩阵所有非零元素上判 c ≤ T·δ
  其中 δ = (e^0.5−1)/1e4，T = 归一化分母（情形 A = 子集和；情形 B = 全矩阵和）
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np, anndata as ad, scipy.sparse as sp

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
DELTA = (np.exp(0.5) - 1) / 1e4
TCRIT = 1e4 / (np.exp(0.5) - 1)
OUT, JS = [], {}


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "VERIFY_TABLES2.txt"), "w", encoding="utf-8").write("\n".join(OUT))


DS = [("SPanC-Lnc CM", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad", "cuTAR"),
      ("SPanC-Lnc SCC", r"data\SPanC-Lnc-pan\SCC.h5ad", "cuTAR"),
      ("GSE280315 P1CRC", r"data\GSE280315\h5ad\GSM8594567_P1CRC.h5ad", "rare"),
      ("GSE280315 P3NAT", r"data\GSE280315\h5ad\GSM8594570_P3NAT.h5ad", "rare"),
      ("GSE225857 C1", r"data\GSE225857\h5ad\GSM7058756_C1.h5ad", "rare"),
      ("GSE225857 L1", r"data\GSE225857\h5ad\GSM7058760_L1.h5ad", "rare")]
# 表 S2「情形 B」块的**现行**值（本轮已按重算更正；括号内为更正前的错误值）
CLAIM = {"SPanC-Lnc CM": (0, 40, 5.525, "0.0%", 0),          # 原写 100.0%
         "SPanC-Lnc SCC": (5493, 46963, 0.193, "25.3%", 27303),   # 原写 0.0% / 0
         "GSE280315 P1CRC": (147, 2104, 1.750, "0.0%", 0),
         "GSE280315 P3NAT": (23, 5560, 1.029, "0.0%", 0),
         "GSE225857 C1": (25063, 213355, 0.046, "69.7%", 32377),   # 原写 7.2% / 44,891
         "GSE225857 L1": (3942, 70920, 0.132, "5.5%", 16959)}      # 原写 0.0% / 0
# 14 个「行内非零元素」口径（口径 S）以外的单元格（中位深度 / 最深 spot / t*）此前已核对无误

W("=" * 112)
W("逐格重算补充材料表 S2（情形 B：按全矩阵归一化）")
W("=" * 112)
W(f"  δ = {DELTA:.10e}    T_crit = {TCRIT:.2f}\n")
W(f"  {'数据集':<18}{'中位深度':>10}{'最深spot':>11}{'t*':>8}{'0.5容许':>8}"
  f"{'超界spot%':>10}{'不匹配(S口径)':>14}{'不匹配(W口径)':>14}{'非零(S/W)':>16}")
tab = {}
for tag, rel, mode in DS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n = a.n_obs
    names = [str(v) for v in a.var.index]
    if mode == "cuTAR":
        m = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    else:
        fd = np.diff(X.tocsc().indptr)
        m = (fd / n) < 0.02
    Tfull = np.asarray(X.sum(1)).ravel()
    Ts = Tfull                                   # 情形 B：分母 = 全矩阵和
    mmS = mmW = 0
    cooS = X[:, m].tocoo()
    mmS = int(np.count_nonzero(cooS.data <= Ts[cooS.row] * DELTA))
    cooW = X.tocoo()
    mmW = int(np.count_nonzero(cooW.data <= Ts[cooW.row] * DELTA))
    med = float(np.median(Tfull)); mx = float(Tfull.max())
    ts = float(np.log1p(1e4 / mx))
    over = 100 * float((Tfull > TCRIT).mean())
    tab[tag] = dict(median=med, max=mx, tstar=ts, ok05=bool(0.5 < ts), pct_over=over,
                    mm_S=mmS, mm_W=mmW, nnz_S=int(cooS.nnz), nnz_W=int(cooW.nnz),
                    n_rare=int(m.sum()), n_spot=int(n))
    W(f"  {tag:<18}{med:>10,.0f}{mx:>11,.0f}{ts:>8.3f}{('是' if 0.5<ts else '否'):>8}"
      f"{over:>9.1f}%{mmS:>14,}{mmW:>14,}{f'{cooS.nnz:,}/{cooW.nnz:,}':>16}")
    del a, X, cooS, cooW
JS["caseB"] = tab

W("\n  与表 S2 声明对比：")
W(f"  {'数据集':<18}{'项':<16}{'表示':>14}{'我算':>16}{'一致?':>8}")
bad = []
for tag in CLAIM:
    c_med, c_max, c_ts, c_over, c_mm = CLAIM[tag]
    t = tab[tag]
    for nm, cv, mv in (("中位深度", c_med, t["median"]), ("最深 spot", c_max, t["max"]),
                       ("t*", c_ts, t["tstar"])):
        ok = abs(cv - mv) < max(1, 0.002 * abs(cv))
        if not ok:
            bad.append((tag, nm, cv, mv))
        W(f"  {tag:<18}{nm:<16}{cv:>14,}{mv:>16,.3f}{('一致' if ok else '**不一致**'):>8}")
    # 通用解析任意 "X.Y%" 声明（原先只处理 0.0% / 100.0% / 7.2% 三个字面量，是判据本身的一个漏洞）
    cv = float(c_over.rstrip("%"))
    okl = abs(t["pct_over"] - cv) < (0.05 if cv == 0.0 else 0.5)
    if not okl:
        bad.append((tag, "超界 spot 占比", c_over, f"{t['pct_over']:.1f}%"))
    W(f"  {tag:<18}{'超界 spot 占比':<16}{c_over:>14}{t['pct_over']:>15.1f}%"
      f"{('一致' if okl else '**不一致**'):>8}")
    okm = (c_mm == t["mm_S"]) or (c_mm == t["mm_W"])
    if not okm:
        bad.append((tag, "不匹配", c_mm, f"S={t['mm_S']}, W={t['mm_W']}"))
    W(f"  {tag:<18}{'不匹配':<16}{c_mm:>14,}{f'S={t[chr(109)+chr(109)+chr(95)+chr(83)]:,} W={t[chr(109)+chr(109)+chr(95)+chr(87)]:,}':>16}"
      f"{('一致' if okm else '**不一致**'):>8}")

W(f"\n  共 {len(bad)} 处不一致：")
for tag, nm, cv, mv in bad:
    W(f"    ✗ {tag} / {nm}：表写 {cv}，重算 {mv}")
JS["mismatch"] = [dict(dataset=t, item=n, claim=str(c), computed=str(m)) for t, n, c, m in bad]

W("\n" + "=" * 112)
W("情形 A（按特征子集归一化）核对")
W("=" * 112)
W(f"  {'数据集':<18}{'子集最深':>10}{'t*':>8}{'0.5 容许':>9}{'不匹配':>10}")
for tag, rel, mode in DS:
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(np.asarray(a.X, float))
    X.eliminate_zeros()
    n = a.n_obs
    names = [str(v) for v in a.var.index]
    if mode == "cuTAR":
        m = np.array([s.upper().startswith("CUTAR") for s in names], bool)
    else:
        fd = np.diff(X.tocsc().indptr)
        m = (fd / n) < 0.02
    Xs = X[:, m].tocsr()
    Tsub = np.asarray(Xs.sum(1)).ravel()
    coo = Xs.tocoo()
    mm = int(np.count_nonzero(coo.data <= Tsub[coo.row] * DELTA))
    ts = float(np.log1p(1e4 / max(Tsub.max(), 1)))
    W(f"  {tag:<18}{Tsub.max():>10,.0f}{ts:>8.3f}{('是' if 0.5<ts else '否'):>9}{mm:>10,}")
    JS.setdefault("caseA", {})[tag] = dict(Tmax_sub=float(Tsub.max()), t_star=ts, mismatch=mm)
    del a, X, Xs, coo

open(os.path.join(RES, "VERIFY_TABLES2.txt"), "w", encoding="utf-8").write("\n".join(OUT))
json.dump(JS, open(os.path.join(RES, "VERIFY_TABLES2.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\VERIFY_TABLES2.txt / .json")
