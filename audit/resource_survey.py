# -*- coding: utf-8 -*-
"""
resource_survey.py — 资源可用性普查（A 路线的主体数据层）。

对每个矩阵产出：
  S1 规模与稀疏性：n_spot / n_feat / 总 reads / 非零率 / 每 spot reads 分布 / 有>=1read 的 spot 比例
  S2 特征可检出性：每特征检出 spot 数分布；>=1% / >=5% / >=10% / >=20% 门槛下保留的特征数（全体 / cuTAR / 非cuTAR）
  S3 阈值代数：T_crit = 15417 之上/之下的 spot 比例；logNorm 阈值扫描（0.01→6）下掩码元素数与过阈 spot 数
  S4 naive 伪影度量：Spearman(过阈特征数, 总深度)、ARI(naive 域, 总深度十分位)

另对 SCC/BCC 产出完整稀释阶梯（1.0→0.10）：
  naive 掩码 Jaccard / corrected cuTAR 域 ARI / 编码基因参照域 ARI
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import anndata as ad
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

ROOT = _cfg.WORKSPACE + r""
T_CRIT = 15417.0          # logNorm>0.5 与 count>=1 等价的深度界限
RNG = np.random.default_rng(20260923)
OUT, ROWS, LADDER = [], [], []


def W(s=""):
    print(s, flush=True)
    OUT.append(s)
    with open(os.path.join(ROOT, "results", "RESOURCE_survey.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


def spot_totals(X):
    if sp.issparse(X):
        return np.asarray(X.sum(1)).ravel().astype(np.float64)
    return np.asarray(X, dtype=np.float64).sum(1)


def feature_det(X):
    if sp.issparse(X):
        Xc = X.tocsc()
        return np.diff(Xc.indptr).astype(np.int64)
    return (np.asarray(X) > 0).sum(0).astype(np.int64)


def spot_det(X):
    if sp.issparse(X):
        Xr = X.tocsr()
        return np.diff(Xr.indptr).astype(np.int64)
    return (np.asarray(X) > 0).sum(1).astype(np.int64)


def lognorm_cu(X, colmask):
    """只对选定列做 logNorm，避免大矩阵全量密集化。返回密集子矩阵。"""
    if sp.issparse(X):
        S = X[:, colmask].toarray().astype(np.float64)
    else:
        S = np.asarray(X)[:, colmask].astype(np.float64)
    t = S.sum(1)
    t[t == 0] = 1.0
    return np.log1p(S / t[:, None] * 1e4)


def cluster_of(M, k, ncomp=15):
    ncomp = int(min(ncomp, M.shape[1] - 1, M.shape[0] - 1))
    if ncomp < 2:
        return np.zeros(M.shape[0], dtype=int)
    Z = TruncatedSVD(n_components=ncomp, random_state=0).fit_transform(M)
    return KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3,
                  random_state=0).fit_predict(Z)


def decile(v):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))


def spearman(a, b):
    return float(np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))[0, 1])


def survey(tag, rel, heavy=True, note=""):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        W(f"\n[{tag}] 文件不存在，跳过：{rel}")
        return None
    a = ad.read_h5ad(p)
    X = a.X
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], dtype=bool)
    n_cu = int(is_cu.sum())

    tot = spot_totals(X)
    fdet = feature_det(X)
    sdet = spot_det(X)
    tot_reads = float(tot.sum())
    n_nonzero = float(fdet.sum())
    n_el = a.n_obs * a.n_vars

    r = {"tag": tag, "path": rel, "note": note,
         "n_spot": int(a.n_obs), "n_feat": int(a.n_vars), "n_cuTAR": n_cu,
         "pct_cuTAR": round(100.0 * n_cu / a.n_vars, 2),
         "total_reads": int(tot_reads),
         "density_pct": round(100.0 * n_nonzero / n_el, 4),
         "median_reads_per_spot": float(np.median(tot)),
         "pct_spot_ge1read": round(100.0 * float((tot > 0).mean()), 2),
         "median_genes_per_spot": float(np.median(sdet)),
         "feat_det_median": int(np.median(fdet)),
         "feat_det_max": int(fdet.max()),
         "pct_spot_below_Tcrit": round(100.0 * float((tot < T_CRIT).mean()), 2)}

    for thr in (0.01, 0.05, 0.10, 0.20):
        r[f"n_feat_ge{int(thr*100)}pct_all"] = int((fdet >= thr * a.n_obs).sum())
        r[f"n_feat_ge{int(thr*100)}pct_cu"] = int((fdet[is_cu] >= thr * a.n_obs).sum())
        if n_cu < a.n_vars:
            r[f"n_feat_ge{int(thr*100)}pct_coding"] = int((fdet[~is_cu] >= thr * a.n_obs).sum())
    r["best_cu_detect_pct"] = round(100.0 * float(fdet[is_cu].max() / a.n_obs), 2) if n_cu else None

    W("\n" + "=" * 104)
    W(f"[{tag}]{('  ' + note) if note else ''}")
    W(f"  {rel}")
    W(f"  S1 规模/稀疏: {a.n_obs:,} spot x {a.n_vars:,} feat | cuTAR {n_cu:,} ({r['pct_cuTAR']}%)")
    W(f"     总 reads = {int(tot_reads):,} | 非零率 = {r['density_pct']}% | 每 spot reads 中位 = {r['median_reads_per_spot']:.0f}")
    W(f"     有>=1 read 的 spot = {r['pct_spot_ge1read']}% | 每 spot 检出特征中位 = {r['median_genes_per_spot']:.0f}")
    W(f"  S2 可检出性: 每特征检出 spot 数 中位={r['feat_det_median']} max={r['feat_det_max']}"
      + (f" | cuTAR 最高检出率 = {r['best_cu_detect_pct']}%" if n_cu else ""))
    W(f"     {'门槛':>6}{'全体':>10}{'cuTAR':>10}{'非cuTAR':>10}")
    for thr in (0.01, 0.05, 0.10, 0.20):
        k = int(thr * 100)
        cod = r.get(f"n_feat_ge{k}pct_coding", "-")
        W(f"     >={k:>3}%{r[f'n_feat_ge{k}pct_all']:>10,}{r[f'n_feat_ge{k}pct_cu']:>10,}{cod if cod == '-' else format(cod, ',>' + '10')}")
    W(f"  S3 阈值代数: 深度 < T_crit={T_CRIT:.0f} 的 spot = {r['pct_spot_below_Tcrit']}%"
      f"  -> 这些 spot 上 logNorm>0.5 等价于 count>=1")

    if n_cu >= 20 and heavy:
        Lcu = lognorm_cu(X, is_cu) if a.n_vars <= 40000 else None
        if Lcu is not None:
            scan = []
            for t in (0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
                m = Lcu > t
                scan.append({"thr": t, "elems": int(m.sum()), "spots": int((m.sum(1) > 0).sum())})
            r["threshold_scan"] = scan
            W(f"     logNorm 阈值扫描（掩码元素 / 过阈 spot）:")
            for s in scan:
                W(f"       >{s['thr']:<5} {s['elems']:>10,}  {s['spots']:>8,}")

            m05 = Lcu > 0.5
            lab = cluster_of(m05.astype(float), 5)
            rho = spearman(m05.sum(1), tot)
            a_ari = float(ari(decile(tot), lab))
            r.update({"mask_elems_05": int(m05.sum()),
                      "spots_any_05": int((m05.sum(1) > 0).sum()),
                      "spearman_maskcount_vs_depth": round(rho, 4),
                      "ari_naive_vs_depthdecile": round(a_ari, 4)})
            W(f"  S4 naive 伪影: logNorm>0.5 掩码元素 = {int(m05.sum()):,} ({100*m05.sum()/m05.size:.4f}%)"
              f" | 过阈 spot = {int((m05.sum(1)>0).sum()):,} ({100*(m05.sum(1)>0).mean():.2f}%)")
            W(f"     Spearman(过阈特征数, 总深度) = {rho:.4f}   ARI(naive 域, 深度十分位) = {a_ari:.4f}")
            del Lcu
    del X
    try:
        del a
    except Exception:
        pass
    ROWS.append(r)
    return r


def dilution_ladder(tag, rel, fracs=(0.75, 0.50, 0.35, 0.25, 0.15, 0.10)):
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    X = a.X
    X = X.toarray().astype(np.float64) if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    names = [str(x) for x in a.var.index]
    is_cu = np.array([n.upper().startswith("CUTAR") for n in names], dtype=bool)
    n_spot = a.n_obs

    Xcu = X[:, is_cu]
    Xcd = X[:, ~is_cu]
    base_mask = lognorm_cu(X, is_cu) > 0.5
    det = (Xcu > 0).mean(0)
    keep = det >= 0.10
    cols_cu_keep = np.where(is_cu)[0][keep]
    s1, lab_c = None, None
    if keep.sum() >= 5:
        s1 = np.where((Xcu[:, keep] > 0).sum(1) > 0)[0]
        lab_c = cluster_of(lognorm_cu(X[s1], cols_cu_keep), 4)

    hvg = np.argsort(-np.median(Xcd, 0))[:min(1500, Xcd.shape[1])]
    s_ref = np.where((Xcd[:, hvg] > 0).sum(1) > 0)[0]
    lab_r = cluster_of(lognorm_cu(X[s_ref], np.where(~is_cu)[0][hvg]), 4)

    W("\n" + "=" * 104)
    W(f"[稀释阶梯] {tag}   cuTAR保留(>=10%检出) = {int(keep.sum())}/{int(is_cu.sum())}"
      f" | corrected spot = {0 if s1 is None else len(s1):,} | reference spot = {len(s_ref):,}")
    W(f"  {'frac':>6}{'naive掩码Jaccard':>18}{'corrected域ARI':>17}{'reference域ARI':>17}{'过阈spot%':>12}")
    for frac in fracs:
        Xd = RNG.binomial(np.round(X).astype(np.int64), frac).astype(np.float64)
        mn = lognorm_cu(Xd, is_cu) > 0.5
        jac = float((mn & base_mask).sum() / max(1, (mn | base_mask).sum()))
        pct = 100.0 * float((mn.sum(1) > 0).mean())
        ca = "n/a"
        if s1 is not None:
            c2 = cluster_of(lognorm_cu(Xd[s1], cols_cu_keep), 4)
            ca = f"{ari(lab_c, c2):.4f}"
        r2 = cluster_of(lognorm_cu(Xd[s_ref], np.where(~is_cu)[0][hvg]), 4)
        ra = f"{ari(lab_r, r2):.4f}"
        W(f"  {frac:>6.2f}{jac:>18.4f}{ca:>17}{ra:>17}{pct:>12.2f}")
        LADDER.append({"tag": tag, "frac": frac, "naive_jaccard": round(jac, 4),
                       "corrected_ari": ca, "reference_ari": ra,
                       "pct_spots_any": round(pct, 2)})
    del X, a


def main():
    jobs = [
        ("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad", "CRC 原发/转移，长读 cuTAR-only"),
        ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad", "CRC 配对，长读 cuTAR-only"),
        ("BCC_nano", r"data\SPanC-Lnc-pan\BCC_nano.h5ad", "纳米孔 cuTAR-only"),
        ("SCC_nano", r"data\SPanC-Lnc-pan\SCC_nano.h5ad", "纳米孔 cuTAR-only"),
        ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad", "纳米孔 cuTAR-only"),
        ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad", "混合（cuTAR + 编码）"),
        ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad", "混合（cuTAR + 编码）"),
        ("HNC", r"data\SPanC-Lnc-pan\HNC.h5ad", "混合（cuTAR + 编码）"),
        ("KidneyCancer", r"data\SPanC-Lnc-pan\KidneyCancer.h5ad", "混合（本体注释指向 Kidney Cancer）"),
        ("Melanoma", r"data\SPanC-Lnc-pan\Melanoma.h5ad", "混合"),
        ("Melanoma_scRNA", r"data\SPanC-Lnc-pan\Melanoma_scRNA.h5ad", "单细胞参考对象"),
    ]
    for tag, rel, note in jobs:
        try:
            survey(tag, rel, heavy=(tag not in ("Melanoma_scRNA",)), note=note)
        except Exception as e:
            import traceback
            W(f"\n[{tag}] FAILED: {type(e).__name__}: {e}")
            W(traceback.format_exc())

    # 外部对照：正常 Visium / Stereo-seq 矩阵长什么样
    for tag, rel, note in [
        ("GSE225857_C1", r"data\GSE225857\h5ad\GSM7058756_C1.h5ad", "外部对照：Visium 正常矩阵"),
        ("GSE225857_L1", r"data\GSE225857\h5ad\GSM7058760_L1.h5ad", "外部对照：Visium 正常矩阵"),
    ]:
        try:
            survey(tag, rel, heavy=True, note=note)
        except Exception as e:
            W(f"\n[{tag}] FAILED: {type(e).__name__}: {e}")

    for tag, rel in [("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                     ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad")]:
        try:
            dilution_ladder(tag, rel)
        except Exception as e:
            import traceback
            W(f"\n[ladder {tag}] FAILED: {type(e).__name__}: {e}")
            W(traceback.format_exc())

    import csv
    if ROWS:
        keys = sorted({k for r in ROWS for k in r if k != "threshold_scan"})
        with open(os.path.join(ROOT, "results", "RESOURCE_survey.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in ROWS:
                w.writerow(r)
    if LADDER:
        with open(os.path.join(ROOT, "results", "dilution_ladder.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(LADDER[0].keys()))
            w.writeheader()
            w.writerows(LADDER)
    with open(os.path.join(ROOT, "results", "RESOURCE_survey.json"), "w", encoding="utf-8") as f:
        json.dump({"matrices": ROWS, "ladder": LADDER}, f, ensure_ascii=False, indent=1)
    W("\n[written] results\\RESOURCE_survey.txt/.csv/.json, results\\dilution_ladder.csv")


if __name__ == "__main__":
    main()
