# -*- coding: utf-8 -*-
"""
naive_vs_corrected2.py — 精简版：只回答「Fig X 三列布局在真实数据上能否成立」。
每个数据集跑完立刻落盘并 flush，便于部分取用。缩减 SVD 维数/重复数以提速。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, time
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import anndata as ad
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari

ROOT = _cfg.WORKSPACE + r""
RNG = np.random.default_rng(20260923)
TXT = os.path.join(ROOT, "results", "NAIVE_vs_CORRECTED.txt")
JSN = os.path.join(ROOT, "results", "NAIVE_vs_CORRECTED.json")
OUT, ALL = [], []


def W(s=""):
    print(s, flush=True)
    OUT.append(s)
    with open(TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))


def flush_json(extra=None):
    with open(JSN, "w", encoding="utf-8") as f:
        json.dump(ALL + ([extra] if extra else []), f, ensure_ascii=False, indent=1)


def lognorm(X, target=1e4):
    X = np.asarray(X, dtype=np.float64)
    s = X.sum(1)
    s[s == 0] = 1.0
    return np.log1p(X / s[:, None] * target)


def cluster_of(M, k, ncomp=15):
    ncomp = int(min(ncomp, M.shape[1] - 1, M.shape[0] - 1))
    if ncomp < 2:
        return np.zeros(M.shape[0], dtype=int)
    Z = TruncatedSVD(n_components=ncomp, random_state=0).fit_transform(M)
    lab = KMeans(n_clusters=int(min(k, M.shape[0] - 1)), n_init=3,
                 random_state=0).fit_predict(Z)
    return lab


def decile(v):
    return np.digitize(v, np.quantile(v, np.linspace(0, 1, 11)[1:-1]))


CU = ("CUTAR", "LNCRNA", "MSTRG", "TCONS", "NONCODE", "LINC", "AC0", "AL0", "AP0")


def analyse(tag, rel, fracs=(0.5, 0.25)):
    t0 = time.time()
    a = ad.read_h5ad(os.path.join(ROOT, rel))
    names = [str(x) for x in a.var.index]
    up = [s.upper() for s in names]
    is_cu = np.array([s.startswith("CUTAR") for s in up], dtype=bool)

    X = a.X
    X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
    X = np.asarray(X, dtype=np.float64)
    n_spot, n_feat = X.shape
    tot = X.sum(1)
    det_spot = (X > 0).sum(1)

    W("\n" + "=" * 100)
    W(f"[{tag}]  {rel}")
    W(f"  {n_spot:,} x {n_feat:,} | cuTAR 特征 {int(is_cu.sum()):,} ({100*is_cu.mean():.1f}%) | 非cuTAR {int((~is_cu).sum()):,}")
    W(f"  spot 总计数 min/中位/max = {tot.min():.0f}/{np.median(tot):.0f}/{tot.max():.0f}")
    W(f"  spot 检出基因数 min/中位/max = {det_spot.min()}/{np.median(det_spot):.0f}/{det_spot.max()}")

    res = {"tag": tag, "n_spot": int(n_spot), "n_feat": int(n_feat),
           "n_cut": int(is_cu.sum()), "median_total": float(np.median(tot)),
           "median_det": float(np.median(det_spot))}

    if is_cu.sum() < 20:
        W("  cuTAR 特征不足，跳过")
        flush_json(); return res

    Xcu = X[:, is_cu]
    Lcu = lognorm(Xcu)
    mask_naive = Lcu > 0.5                      # 旧论文的二值化判据
    n_mask = int(mask_naive.sum())
    spots_any = int((mask_naive.sum(1) > 0).sum())
    lab_naive = cluster_of(mask_naive.astype(float), 5)
    ari_naive_depth = float(ari(decile(tot), lab_naive))
    W(f"\n  --- R0 naive: logNorm>0.5 二值化 ---")
    W(f"    掩码元素 = {n_mask:,} / {mask_naive.size:,} ({100*n_mask/mask_naive.size:.4f}%)")
    W(f"    至少一个特征过阈的 spot = {spots_any:,} ({100*spots_any/n_spot:.2f}%)")
    W(f"    各 spot 过阈特征数 与 总计数 的 Spearman = "
      f"{np.corrcoef(np.argsort(np.argsort(tot)), np.argsort(np.argsort(mask_naive.sum(1))))[0,1]:.3f}")
    W(f"    J1 ARI(naive 域, 总计数十分位) = {ari_naive_depth:.4f}   <- 越接近1说明域就是深度分层")
    res.update({"naive_mask_elems": n_mask, "naive_spots_any": spots_any,
                "naive_ari_depth": round(ari_naive_depth, 4)})

    # R1 corrected：物理检出判据作用在原始 count 上
    det_rate = (Xcu > 0).mean(0)
    keep = det_rate >= 0.10
    W(f"\n  --- R1 corrected: count>=1 且检出率>=10% ---")
    W(f"    保留 {int(keep.sum()):,}/{len(keep):,} 个 cuTAR ({100*keep.mean():.1f}%)")
    s1, lab_corr, ari_corr_depth = None, None, None
    if keep.sum() >= 5:
        Xc1 = Xcu[:, keep]
        s1 = np.where((Xc1 > 0).sum(1) > 0)[0]
        lab_corr = cluster_of(lognorm(Xc1[s1]), 4)
        ari_corr_depth = float(ari(decile(tot[s1]), lab_corr))
        W(f"    有效 spot = {len(s1):,} ({100*len(s1)/n_spot:.2f}%)  簇大小 = {np.bincount(lab_corr).tolist()}")
        W(f"    J1 ARI(corrected 域, 总计数十分位) = {ari_corr_depth:.4f}")
        res.update({"corr_keep": int(keep.sum()), "corr_spots": int(len(s1)),
                    "corr_ari_depth": round(ari_corr_depth, 4)})

    # R2 reference：非 cuTAR（编码基因）聚类
    lab_ref, s_ref, hvg_ref = None, None, None
    if (~is_cu).sum() >= 50:
        Xcd = X[:, ~is_cu]
        hvg_ref = np.argsort(-np.median(Xcd, 0))[:min(1500, Xcd.shape[1])]
        Xr = Xcd[:, hvg_ref]
        s_ref = np.where((Xr > 0).sum(1) > 0)[0]
        lab_ref = cluster_of(lognorm(Xr[s_ref]), 4)
        ari_ref_depth = float(ari(decile(tot[s_ref]), lab_ref))
        W(f"\n  --- R2 reference: 非cuTAR 特征 {int((~is_cu).sum()):,} 中取中位表达 top {len(hvg_ref)} ---")
        W(f"    有效 spot = {len(s_ref):,}  簇大小 = {np.bincount(lab_ref).tolist()}")
        W(f"    J1 ARI(reference 域, 总计数十分位) = {ari_ref_depth:.4f}")
        res.update({"ref_spots": int(len(s_ref)), "ref_ari_depth": round(ari_ref_depth, 4)})
        if s1 is not None:
            common = np.intersect1d(s1, s_ref)
            j = float(ari(np.asarray(lab_corr)[np.isin(s1, common)],
                          np.asarray(lab_ref)[np.isin(s_ref, common)]))
            W(f"    J3 ARI(corrected cuTAR 域, reference 编码基因域) @{len(common):,} spot = {j:.4f}")
            res["J3_ari_corr_vs_ref"] = round(j, 4)

    # J2 稀释稳定性
    W(f"\n  --- J2 二项稀释稳定性 ---")
    W(f"    {'frac':>5} | {'naive掩码Jaccard':>16} | {'corrected域ARI':>15} | {'reference域ARI':>15}")
    d2 = {}
    for frac in fracs:
        jac = None
        Xd_cu = RNG.binomial(np.round(X[:, is_cu]).astype(np.int64), frac).astype(np.float64)
        mn = lognorm(Xd_cu) > 0.5
        jac = float((mn & mask_naive).sum() / max(1, (mn | mask_naive).sum()))
        ca = "n/a"
        if s1 is not None:
            c2 = cluster_of(lognorm(Xd_cu[s1][:, keep]), 4)
            ca = f"{ari(lab_corr, c2):.4f}"
        ra = "n/a"
        if lab_ref is not None:
            Xd_cd = RNG.binomial(np.round(X[:, ~is_cu][:, hvg_ref]).astype(np.int64), frac).astype(np.float64)
            r2 = cluster_of(lognorm(Xd_cd[s_ref]), 4)
            ra = f"{ari(lab_ref, r2):.4f}"
        W(f"    {frac:>5.2f} | {jac:>16.4f} | {ca:>15} | {ra:>15}")
        d2[str(frac)] = {"naive_jaccard": jac, "corr_ari": ca, "ref_ari": ra}
    res["dilution"] = d2

    W(f"\n  --- 阈值失效机制 ---")
    W(f"    cuTAR 检出率: min={det_rate.min():.3f} 中位={np.median(det_rate):.3f} max={det_rate.max():.3f}")
    W(f"    检出率<5% 的特征 = {int((det_rate<0.05).sum()):,} ({100*(det_rate<0.05).mean():.1f}%)")
    c_needed = np.expm1(0.5) / 1e4 * np.median(tot)
    W(f"    中位深度 spot 上，单个 read 的 logNorm = log1p(1e4/{np.median(tot):.0f}) = {np.log1p(1e4/np.median(tot)):.3f}"
      f"  -> 只要 count>={int(np.ceil(c_needed))} 即 > 0.5")
    res["cu_det_median"] = round(float(np.median(det_rate)), 4)
    res["elapsed_s"] = round(time.time() - t0, 1)
    W(f"  [耗时 {res['elapsed_s']}s]")
    del a
    return res


def main():
    for tag, rel in [("CM_pacbio", r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad"),
                     ("CP_pacbio", r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad"),
                     ("SCC", r"data\SPanC-Lnc-pan\SCC.h5ad"),
                     ("BCC", r"data\SPanC-Lnc-pan\BCC.h5ad")]:
        sys.stderr.write(f"=== {tag} ===\n"); sys.stderr.flush()
        try:
            ALL.append(analyse(tag, rel))
        except Exception as e:
            import traceback
            W(f"\n[{tag}] FAILED: {type(e).__name__}: {e}")
            W(traceback.format_exc())
        flush_json()
    W("\n" + "=" * 100)
    W("机器可读汇总")
    W(json.dumps(ALL, ensure_ascii=False, indent=1))
    flush_json()
    print(f"\n[written] {TXT}\n[written] {JSN}")


if __name__ == "__main__":
    main()
