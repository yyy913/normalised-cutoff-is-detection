# -*- coding: utf-8 -*-
"""
make_figures.py — A 路线（资源可用性评估 + 检出性条件化框架）主体图件。
数据来源：results/RESOURCE_survey.json、results/dilution_ladder.csv，以及原始 h5ad 的逐 spot / 逐特征量。
输出：results/figures/Fig1..Fig5  (.png 400dpi + .pdf)
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("NUMBA_CACHE_DIR",
                      _cfg.WORKSPACE + r"\.numba_cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import scipy.sparse as sp
import anndata as ad

ROOT = _cfg.WORKSPACE + r""
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)
CACHE = os.path.join(FIG, "_percells.npz")
T_CRIT = 15417.0

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.2,
    "axes.linewidth": 0.6, "axes.labelsize": 7.2, "axes.titlesize": 7.6,
    "xtick.labelsize": 6.4, "ytick.labelsize": 6.4,
    "legend.fontsize": 6.0, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
MM = 1 / 25.4
W1, W2 = 89 * MM, 183 * MM

# tag -> (relpath, group, label)
MATS = [
    ("CM_pacbio",        r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad",        "PacBio, cuTAR-only", "CM (PacBio)"),
    ("CP_pacbio",        r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad",        "PacBio, cuTAR-only", "CP (PacBio)"),
    ("HNC_ilong_nano",   r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad",    "Nanopore, cuTAR-only", "HNC (ONT)"),
    ("BCC_nano",         r"data\SPanC-Lnc-pan\BCC_nano.h5ad",          "Nanopore, cuTAR-only", "BCC (ONT)"),
    ("SCC_nano",         r"data\SPanC-Lnc-pan\SCC_nano.h5ad",          "Nanopore, cuTAR-only", "SCC (ONT)"),
    ("Melanoma",         r"data\SPanC-Lnc-pan\Melanoma.h5ad",          "Short-read, mixed", "Melanoma"),
    ("BCC",              r"data\SPanC-Lnc-pan\BCC.h5ad",               "Short-read, mixed", "BCC"),
    ("KidneyCancer",     r"data\SPanC-Lnc-pan\KidneyCancer.h5ad",      "Short-read, mixed", "Kidney"),
    ("HNC",              r"data\SPanC-Lnc-pan\HNC.h5ad",               "Short-read, mixed", "HNC"),
    ("SCC",              r"data\SPanC-Lnc-pan\SCC.h5ad",               "Short-read, mixed", "SCC"),
]
GCOL = {"PacBio, cuTAR-only": "#b2182b", "Nanopore, cuTAR-only": "#ef8a62",
        "Short-read, mixed": "#2166ac"}
GLAB = {"PacBio, cuTAR-only": "PacBio, cuTAR-only (n=2)",
        "Nanopore, cuTAR-only": "Nanopore, cuTAR-only (n=3)",
        "Short-read, mixed": "Short-read, mixed (n=5)"}


def build_cache():
    if os.path.exists(CACHE):
        return np.load(CACHE, allow_pickle=True)
    d = {}
    for tag, rel, grp, lab in MATS:
        p = os.path.join(ROOT, rel)
        a = ad.read_h5ad(p)
        X = a.X
        names = [str(x) for x in a.var.index]
        is_cu = np.array([n.upper().startswith("CUTAR") for n in names], bool)
        if sp.issparse(X):
            tot = np.asarray(X.sum(1)).ravel()
            det = np.diff(X.tocsr().indptr)
            fd = np.diff(X.tocsc().indptr)
        else:
            Xa = np.asarray(X)
            tot = Xa.sum(1)
            det = (Xa > 0).sum(1)
            fd = (Xa > 0).sum(0)
        d[tag + "__tot"] = tot.astype(np.float64)
        d[tag + "__det"] = det.astype(np.int64)
        d[tag + "__fd"] = fd.astype(np.int64)
        d[tag + "__iscu"] = is_cu
        print(f"  cached {tag}", flush=True)
        del a, X
    np.savez_compressed(CACHE, **d)
    return np.load(CACHE, allow_pickle=True)


surv = json.load(open(os.path.join(ROOT, "results", "RESOURCE_survey.json"), encoding="utf-8"))
S = {m["tag"]: m for m in surv["matrices"]}
C = build_cache()
ORDER = [t for t, *_ in MATS]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
    print(f"[fig] {name}")


# ---------------------------------------------------------------- Fig 1
def fig1():
    fig = plt.figure(figsize=(W2, 74 * MM))
    gs = fig.add_gridspec(1, 4, wspace=0.52, left=0.055, right=0.995, bottom=0.20, top=0.86)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(ORDER))
    v = np.array([S[t]["total_reads"] for t in ORDER], float)
    ax.barh(y, np.log10(v), color=[GCOL[g] for _, _, g, _ in MATS], height=0.68)
    ax.set_yticks(y); ax.set_yticklabels([l for *_, l in MATS])
    ax.invert_yaxis(); ax.set_xlabel("Total reads (log$_{10}$)")
    ax.set_title("a  Library size", loc="left")
    for i, t in enumerate(ORDER):
        if t in ("CM_pacbio", "CP_pacbio"):
            ax.text(np.log10(v[i]) + 0.12, i, f"{v[i]:,.0f}", va="center", fontsize=5.8, color="#b2182b")
    ax.set_xlim(0, 8.4)
    ax.axvline(np.log10(15417), color="k", ls=":", lw=0.7)
    ax.annotate("$T_{crit}$", (np.log10(15417), -0.85), fontsize=5.8, ha="center")

    ax = fig.add_subplot(gs[0, 1])
    d = np.array([S[t]["density_pct"] for t in ORDER])
    ax.barh(y, d, color=[GCOL[g] for _, _, g, _ in MATS], height=0.68)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("Non-zero entries (%)"); ax.set_title("b  Matrix density", loc="left")
    ax.set_xlim(0, 15)

    ax = fig.add_subplot(gs[0, 2])
    ax.barh(y, [S[t]["pct_spot_ge1read"] for t in ORDER],
            color=[GCOL[g] for _, _, g, _ in MATS], height=0.68)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlabel("Spots with $\\geq$1 read (%)")
    ax.set_title("c  Spot coverage", loc="left"); ax.set_xlim(0, 105)
    ax.axvline(50, color="k", ls=":", lw=0.7)
    for i, t in enumerate(ORDER[:2]):
        ax.text(2, i, f"{S[t]['pct_spot_ge1read']:.1f}%", va="center", fontsize=5.8, color="#b2182b")

    ax = fig.add_subplot(gs[0, 3])
    for t, rel, g, lab in MATS:
        tot = C[t + "__tot"]
        xs = np.sort(tot[tot > 0])
        if len(xs) == 0:
            continue
        ax.plot(xs, 100 * (1 - np.arange(len(xs)) / len(tot)), color=GCOL[g], lw=1.0,
                label=lab if t in ("CM_pacbio", "BCC_nano", "SCC") else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Reads per spot"); ax.set_ylabel("Spots at or above (% of all)")
    ax.set_title("d  Depth distribution", loc="left")
    ax.axvline(15417, color="k", ls=":", lw=0.7)
    ax.legend(loc="lower left")
    h = [mpl.patches.Patch(color=GCOL[k], label=GLAB[k]) for k in GLAB]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3)
    save(fig, "Fig1_resource_sparsity")


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig = plt.figure(figsize=(W2, 76 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.058, right=0.995, bottom=0.18, top=0.87,
                          width_ratios=[1.0, 1.05, 0.95])

    ax = fig.add_subplot(gs[0, 0])
    mixed = [t for t, _, g, _ in MATS if g == "Short-read, mixed"]
    for t, _, g, lab in MATS:
        if t not in mixed:
            continue
        iscu = C[t + "__iscu"]
        fd = C[t + "__fd"]
        n = S[t]["n_spot"]
        for m, c, ls in ((iscu, "#b2182b", "-"), (~iscu, "#2166ac", "--")):
            xs = np.sort(fd[m] / n * 100)[::-1]
            ax.step(xs, 100 * np.arange(1, len(xs) + 1) / len(xs), color=c, ls=ls, lw=0.9)
    ax.set_xscale("log"); ax.set_xlabel("Detection rate per feature (%, spots)")
    ax.set_ylabel("Cumulative % of features")
    ax.set_title("a  Detectability of cuTARs vs\n     annotated genes (mixed matrices)", loc="left")
    ax.axvline(10, color="k", ls=":", lw=0.8)
    ax.text(10, 4, "10%", fontsize=5.8, ha="center", rotation=90)
    h = [mpl.lines.Line2D([], [], color="#b2182b", ls="-", lw=1.0, label="$cu$TARs"),
         mpl.lines.Line2D([], [], color="#2166ac", ls="--", lw=1.0, label="all other features")]
    ax.legend(handles=h, loc="upper right")
    ax.set_ylim(0, 102)

    ax = fig.add_subplot(gs[0, 1])
    thrs = [1, 5, 10, 20]
    M = np.array([[S[t][f"n_feat_ge{k}pct_cu"] / max(1, S[t]["n_cuTAR"]) * 100 for k in thrs]
                  for t in ORDER])
    im = ax.imshow(M, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(thrs))); ax.set_xticklabels([f"$\\geq${k}%" for k in thrs])
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([l for *_, l in MATS])
    ax.set_title("b  cuTARs retained (%) by\n     detectability criterion", loc="left")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=5.4,
                    color="k" if M[i, j] < 40 else "w")
    plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03).ax.tick_params(labelsize=5.6)

    ax = fig.add_subplot(gs[0, 2])
    x = np.array([S[t]["total_reads"] for t in ORDER], float)
    y = np.array([S[t]["best_cu_detect_pct"] for t in ORDER], float)
    for i, t in enumerate(ORDER):
        g = [gg for tt, _, gg, _ in MATS if tt == t][0]
        ax.scatter(x[i], y[i], s=22, color=GCOL[g], edgecolor="k", linewidth=0.4, zorder=3)
        ax.annotate([l for tt, *_, l in MATS if tt == t][0], (x[i], y[i]),
                    textcoords="offset points", xytext=(4, 3), fontsize=5.4)
    ax.set_xscale("log"); ax.axhline(10, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Total reads (log$_{10}$)"); ax.set_ylabel("Best cuTAR detection rate (%)")
    ax.set_title("c  Ceiling of cuTAR detectability", loc="left")
    ax.set_ylim(-4, 95)
    save(fig, "Fig2_detectability")


# ---------------------------------------------------------------- Fig 3
def fig3():
    rc = json.load(open(os.path.join(ROOT, "results", "RECHECK.json"), encoding="utf-8"))
    ident = {d["tag"]: d for d in rc["identity"]}
    ari = {d["tag"]: d for d in rc["ari"]}
    TC = rc["T_crit"]

    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.32, left=0.075, right=0.995, bottom=0.19, top=0.88,
                          width_ratios=[1.05, 1.0, 0.95])

    # (a) 布尔恒等式：min nonzero logNorm 与 0.5 判据
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(ORDER))
    vals = [ident[t]["min_nonzero_lognorm"] for t in ORDER]
    ax.barh(y, vals, color=[GCOL[g] for _, _, g, _ in MATS], height=0.66)
    ax.axvline(0.5, color="k", ls="--", lw=1.0)
    ax.set_yticks(y); ax.set_yticklabels([l for *_, l in MATS]); ax.invert_yaxis()
    ax.set_xlabel("Smallest non-zero $\\log_1p$ CP10K value")
    ax.set_title("a  The 0.5 cut lies below every\n     non-zero value in all 10 matrices", loc="left")
    for i, t in enumerate(ORDER):
        ax.text(vals[i] + 0.08, i, f"{vals[i]:.2f}", va="center", fontsize=5.4)
    ax.text(0.62, len(ORDER) - 0.4, "published cut 0.5", fontsize=5.6, rotation=90, va="bottom")
    ax.set_xlim(0, 6.6)
    ax.text(2.6, 0.15, "$(\\log_1p\\ \\mathrm{CP10K} > 0.5)$\n$\\equiv$ $(\\mathrm{count} > 0)$\n0 mismatches, 10/10",
            fontsize=6.0, va="center")

    # (b) 不等式
    ax = fig.add_subplot(gs[0, 1])
    T = np.logspace(0, 6, 400)
    ax.plot(T, 1e4 / T, color="k", lw=1.2)
    ax.fill_between(T, 0.5, np.clip(1e4 / T, 0.5, None), color="#b2182b", alpha=0.13)
    ax.axvline(TC, color="k", ls=":", lw=0.9)
    ax.axhline(0.5, color="k", ls="--", lw=0.8)
    ax.text(1.6e3, 0.60, "a single read clears 0.5", fontsize=5.8, color="#b2182b")
    ax.text(TC * 1.2, 3.4, f"$T_{{crit}}=\\dfrac{{10^4}}{{e^{{0.5}}-1}}={TC:,.0f}$", fontsize=6.0)
    for t in ORDER:
        tot = C[t + "__tot"]
        med = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1
        g = [gg for tt, _, gg, _ in MATS if tt == t][0]
        ax.plot([med], [1e4 / med], "o", ms=3.4, color=GCOL[g], mec="k", mew=0.35, zorder=4)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1e6); ax.set_ylim(0.45, 30)
    ax.set_xlabel("Total reads per spot, $T$")
    ax.set_ylabel("$\\log_1p$ CP10K of a single read")
    ax.set_title("b  Why: below $T_{crit}$ the cut\n     cannot discriminate", loc="left")

    # (c) 正确的深度口径下 Spearman
    ax = fig.add_subplot(gs[0, 2])
    s = [ari[t]["sp_sub"] for t in ORDER]
    ax.barh(y, s, color=[GCOL[g] for _, _, g, _ in MATS], height=0.66)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.invert_yaxis()
    ax.set_xlim(0.90, 1.004); ax.set_xlabel("Spearman(features above cut, spot depth)")
    ax.set_title("c  Mask count is a monotone\n     function of $cu$TAR depth", loc="left")
    for i, t in enumerate(ORDER):
        ax.text(s[i] + 0.001, i, f"{s[i]:.3f}", va="center", fontsize=5.2)
    ax.set_xticks([0.90, 0.95, 1.00])
    save(fig, "Fig3_threshold_failure")


# ---------------------------------------------------------------- Fig 4
def fig4():
    rc = json.load(open(os.path.join(ROOT, "results", "RECHECK3.json"), encoding="utf-8"))
    fig = plt.figure(figsize=(W2, 66 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.065, right=0.99, bottom=0.20, top=0.85)
    for k, d in enumerate(rc):
        ax = fig.add_subplot(gs[0, k])
        rows = d["variants"]
        fr = [r["frac"] for r in rows]
        ax.fill_between(fr, [r["random_lo"] for r in rows], [r["random_hi"] for r in rows],
                        color="#7f7f7f", alpha=0.22, label="random genes, same size")
        ax.plot(fr, [r["random_mean"] for r in rows], "o--", color="#7f7f7f", ms=2.4, lw=0.9)
        ax.fill_between(fr, [r["matched_lo"] for r in rows], [r["matched_hi"] for r in rows],
                        color="#2166ac", alpha=0.18, label="detectability-matched genes")
        ax.plot(fr, [r["matched_mean"] for r in rows], "s--", color="#2166ac", ms=2.4, lw=0.9)
        ax.plot(fr, [r["cuTAR"] for r in rows], "^-", color="#b2182b", ms=3.2, lw=1.3,
                label=f"the {d['n_cu']} surviving $cu$TARs")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xticks([0.5, 0.25, 0.1]); ax.set_xticklabels(["0.50", "0.25", "0.10"])
        ax.set_xlabel("Retained sequencing depth")
        if k == 0:
            ax.set_ylabel("ARI vs full-depth partition")
        ax.set_title(f"{'abc'[k]}  {d['tag']}  (n$_{{cuTAR}}$={d['n_cu']})", loc="left")
        ax.set_ylim(-0.06, 0.62)
        if k == 0:
            ax.legend(loc="upper right")
    fig.text(0.5, 0.035,
             "Size-matched controls do NOT make the surviving $cu$TARs look worse: their partitions are the most\n"
             "depth-reproducible of the three. Binomial thinning largely preserves depth ordering, so this is expected\n"
             "for any depth-tracking partition and therefore does not by itself argue for or against biological meaning.",
             ha="center", fontsize=5.8)
    save(fig, "Fig4_depth_robustness")


# ---------------------------------------------------------------- Fig 5
def fig5():
    fig = plt.figure(figsize=(W2, 62 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.06, right=0.995, bottom=0.20, top=0.86)
    foc = ["CM_pacbio", "CP_pacbio"]
    ax = fig.add_subplot(gs[0, 0])
    for i, t in enumerate(foc):
        tot = C[t + "__tot"]
        jit = np.random.default_rng(0).normal(0, 0.045, len(tot))
        ax.scatter(np.full(len(tot), i) + jit, tot, s=1.2, alpha=0.35,
                   color=GCOL["PacBio, cuTAR-only"], linewidths=0)
        ax.plot([i - 0.25, i + 0.25], [np.median(tot)] * 2, color="k", lw=1.3)
        ax.text(i, tot.max() * 1.25, f"median = {np.median(tot):.0f}\n{(tot==0).mean()*100:.1f}% spots = 0",
                ha="center", fontsize=5.8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["CM (PacBio)", "CP (PacBio)"])
    ax.set_ylabel("Total cuTAR reads per spot"); ax.set_ylim(-3, 115)
    ax.set_title("a  The two focal matrices", loc="left")

    ax = fig.add_subplot(gs[0, 1])
    for t, rel, g, lab in MATS:
        iscu = C[t + "__iscu"]
        if iscu.sum() == 0 or iscu.all():
            continue
        fd, n = C[t + "__fd"], S[t]["n_spot"]
        q = np.linspace(0, 1, 400)
        xq = np.quantile(fd[iscu] / n * 100, q)
        yq = np.quantile(fd[~iscu] / n * 100, q)
        ax.plot(xq, yq, color=GCOL[g], lw=0.9, alpha=0.9,
                label=lab if t in ("Melanoma", "BCC", "SCC", "HNC", "KidneyCancer") else None)
    ax.plot([1e-3, 200], [1e-3, 200], "k--", lw=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("cuTAR detection rate (%, quantile)")
    ax.set_ylabel("Annotated-gene detection rate (%, same quantile)")
    ax.set_title("b  Detectability gap (QQ)", loc="left")
    ax.set_xlim(0.03, 100); ax.set_ylim(0.3, 200)
    ax.axvline(10, color="k", ls=":", lw=0.8)
    ax.legend(loc="upper left")

    ax = fig.add_subplot(gs[0, 2])
    rc = json.load(open(os.path.join(ROOT, "results", "RECHECK.json"), encoding="utf-8"))
    c4 = {d["tag"]: d for d in rc["c4"]}
    tags4 = [t for t in ORDER if t in c4]
    x4 = np.arange(len(tags4))
    w = 0.36
    ax.bar(x4 - w / 2, [c4[t]["pc_ret_pct"] for t in tags4], w, color="#2166ac",
           label="protein-coding genes")
    ax.bar(x4 + w / 2, [c4[t]["cu_ret_pct"] for t in tags4], w, color="#b2182b",
           label="$cu$TARs")
    ax.set_yscale("log"); ax.set_ylim(0.05, 300)
    ax.set_xticks(x4)
    ax.set_xticklabels([[l for tt, *_, l in MATS if tt == t][0] for t in tags4],
                       rotation=30, ha="right")
    ax.set_ylabel("% retained at $\\geq$10% detection (log)")
    ax.set_title("c  One criterion, two outcomes\n     (protein-coding vs $cu$TAR)", loc="left")
    ax.legend(loc="upper right")
    for i, t in enumerate(tags4):
        ax.text(i - w / 2, c4[t]["pc_ret_pct"] * 1.15, f"{c4[t]['pc_ret_pct']:.1f}", ha="center", fontsize=5.0)
        ax.text(i + w / 2, c4[t]["cu_ret_pct"] * 1.15, f"{c4[t]['cu_ret_pct']:.2f}", ha="center", fontsize=5.0)
    save(fig, "Fig5_positive_control")


def fig6():
    z = np.load(os.path.join(ROOT, "results", "SCRNA_pergene.npz"), allow_pickle=True)
    pct = z["pct"]; cu = z["cu_names"]; topf = z["top_frac"]
    ctm = z["ct_matrix"]; ctn = z["ct_names"]; ctc = z["ct_counts"]
    tags = list(z["sp_tags"]); spn = z["sp_names"]; spp = z["sp_pct"]
    lut = {str(n): i for i, n in enumerate(cu)}

    fig = plt.figure(figsize=(W2, 72 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.34, left=0.055, right=0.99, bottom=0.19, top=0.86,
                          width_ratios=[1.0, 1.0, 1.18])

    ax = fig.add_subplot(gs[0, 0])
    xs = np.sort(pct)
    ax.plot(xs, 100 * np.arange(1, len(xs) + 1) / len(xs), color="#b2182b", lw=1.3)
    ax.set_xscale("log"); ax.set_xlabel("Cells expressing the cuTAR (%)")
    ax.set_ylabel("Cumulative % of cuTARs")
    ax.set_title("a  33,524 single cells:\n     intrinsic detectability of cuTARs", loc="left")
    for t, txt in ((1, "1%"), (5, "5%"), (10, "10%")):
        ax.axvline(t, color="k", ls=":", lw=0.8)
        ax.text(t, 4, "  " + txt, fontsize=5.6, rotation=90)
    ax.set_xlim(0.03, 60); ax.set_ylim(0, 102)
    ax.text(0.05, 96, f"median {np.median(pct):.2f}%\nonly {int((pct>=10).sum())}/"
                      f"{len(pct)} cuTARs $\\geq$10% of cells", fontsize=5.8, va="top")

    ax = fig.add_subplot(gs[0, 1])
    for k, tg in enumerate(tags):
        nm = str(spn[k]).split(","); pv = np.array([float(v) for v in str(spp[k]).split(",")])
        sv = np.array([pct[lut[g]] for g in nm if g in lut])
        pv2 = np.array([pv[i] for i, g in enumerate(nm) if g in lut])
        rho = float(np.corrcoef(np.argsort(np.argsort(sv)), np.argsort(np.argsort(pv2)))[0, 1])
        same = (tg == "Melanoma")
        ax.scatter(sv, pv2, s=5 if same else 2.2, alpha=0.85 if same else 0.32,
                   color="#b2182b" if same else "#2166ac", linewidths=0,
                   zorder=4 if same else 2, label=f"{tg} (n={len(sv)}, $\\rho$={rho:.2f})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Single-cell detection (% of cells)")
    ax.set_ylabel("Spatial detection (% of spots)")
    ax.set_title("b  Cross-modality: spatial detection\n     does not follow cellular expression", loc="left")
    ax.legend(loc="lower right", fontsize=5.4)
    ax.set_xlim(0.03, 60); ax.set_ylim(0.05, 100)

    ax = fig.add_subplot(gs[0, 2])
    ordr = np.argsort(-topf)[:18]
    sub = ctm[:, ordr]
    im = ax.imshow(sub, cmap="magma_r", aspect="auto", vmin=0, vmax=min(45, sub.max()))
    ax.set_yticks(range(len(ctn)))
    ax.set_yticklabels([f"{n}  (n={c:,})" for n, c in zip(ctn, ctc)], fontsize=5.2)
    ax.set_xticks(range(len(ordr)))
    ax.set_xticklabels([str(cu[i]).replace("cuTAR", "") for i in ordr],
                       rotation=90, fontsize=5.0)
    ax.set_title("c  Top cuTARs are cell-type restricted\n     (% of cells within each type)", loc="left")
    plt.colorbar(im, ax=ax, fraction=0.040, pad=0.02).ax.tick_params(labelsize=5.4)
    save(fig, "Fig6_cross_modality")


# ------------------------------------------------- Fig 4 阈值 vs 连续统计（自查修正版）
def fig4b():
    subs = json.load(open(os.path.join(ROOT, "results", "MORANS_stratified.json"), encoding="utf-8"))
    mc = json.load(open(os.path.join(ROOT, "results", "MORANS_conditioned.json"), encoding="utf-8"))
    sc = json.load(open(os.path.join(ROOT, "results", "SELFCHECK4.json"), encoding="utf-8"))
    rc = json.load(open(os.path.join(ROOT, "results", "RECHECK.json"), encoding="utf-8"))
    ari = {d["tag"]: d for d in rc["ari"]}
    SS = {d["tag"]: d for d in subs}
    MC = {d["tag"]: d for d in mc}
    SC = {d["tag"]: d for d in sc}
    tags = [d["tag"] for d in sc]
    short = {"SCC": "SCC", "HNC": "HNC", "KidneyCancer": "Kidney", "BCC": "BCC", "Melanoma": "Melanoma"}

    fig = plt.figure(figsize=(W2, 70 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.36, left=0.070, right=0.99, bottom=0.22, top=0.84,
                          width_ratios=[1.0, 1.08, 1.0])

    # (a) 两条路线
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(tags))
    ax.bar(xs - 0.2, [ari[t]["sp_sub"] for t in tags], 0.38, color="#b2182b", label="threshold route")
    ax.bar(xs + 0.2, [SC[t]["pct_torus"] / 100.0 for t in tags], 0.38, color="#2166ac",
           label="continuous route")
    ax.set_xticks(xs); ax.set_xticklabels([short[t] for t in tags], rotation=25, ha="right")
    ax.set_ylim(0, 1.26); ax.set_ylabel("Signal that is not depth")
    ax.set_title("a  Two routes, identical data", loc="left")
    ax.legend(loc="lower left", fontsize=5.8)
    ax.text(-0.45, 1.19, "threshold: mask count vs depth, $\\rho$", fontsize=5.1, color="#b2182b")
    ax.text(-0.45, 1.11, "continuous: % surviving toroidal null", fontsize=5.1, color="#2166ac")

    # (b) 三种零模型对比
    ax = fig.add_subplot(gs[0, 1])
    yy = np.arange(len(tags))
    h = 0.26
    ax.barh(yy - h, [SC[t]["pct_plain"] for t in tags], h, color="#c6dbef", label="plain permutation")
    ax.barh(yy, [SC[t]["pct_strat"] for t in tags], h, color="#6baed6", label="depth-stratified")
    ax.barh(yy + h, [SC[t]["pct_torus"] for t in tags], h, color="#2166ac",
            label="toroidal shift (strict)")
    ax.axvline(5, color="k", ls="--", lw=1.0)
    ax.text(7, len(tags) - 0.45, "5% = null expectation", fontsize=5.6)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{short[t]}  (n={SC[t]['pairs']})" for t in tags])
    ax.invert_yaxis(); ax.set_xlim(0, 118)
    ax.set_xlabel("% of $cu$TAR–gene pairs significant")
    ax.set_title("b  Robust to the choice of null", loc="left")
    ax.legend(loc="lower right", fontsize=5.2)
    for i, t in enumerate(tags):
        ax.text(SC[t]["pct_torus"] + 2, i + h, f"{SC[t]['pct_torus']:.0f}", va="center", fontsize=5.4)

    # (c) 伙伴高于随机（修正基线 k^2/N）
    ax = fig.add_subplot(gs[0, 2])
    obs = [MC[t]["top_overlap"] for t in tags]
    ch = [SC[t]["chance_overlap"] for t in tags]
    ax.bar(xs - 0.2, obs, 0.38, color="#238b45", label="observed top-10 overlap")
    ax.plot(xs - 0.2, ch, "kv", ms=6.5, label="chance $=10^2/N$ (corrected)")
    ax.bar(xs + 0.2, [MC[t]["best_cu"] for t in tags], 0.38, color="#a1d99b",
           label="$cu$TAR best-partner $I$")
    ax.plot(xs + 0.2, [MC[t]["best_null"] for t in tags], "k_", ms=13,
            label="detectability-matched null")
    for i, t in enumerate(tags):
        ax.text(i - 0.2, obs[i] + 0.012, f"{obs[i]/ch[i]:.0f}$\\times$", ha="center", fontsize=5.2,
                color="#238b45")
    ax.set_xticks(xs); ax.set_xticklabels([short[t] for t in tags], rotation=25, ha="right")
    ax.set_ylim(0, 0.60); ax.set_ylabel("Value")
    ax.set_title("c  Partners above chance", loc="left")
    ax.legend(loc="upper right", fontsize=4.9)
    save(fig, "Fig4_threshold_vs_continuous")


# ------------------------------------------------- FigS1 无判别力对照（补充材料）
def figS1():
    rc = json.load(open(os.path.join(ROOT, "results", "RECHECK3.json"), encoding="utf-8"))
    fig = plt.figure(figsize=(W2, 60 * MM))
    gs = fig.add_gridspec(1, 3, wspace=0.30, left=0.07, right=0.99, bottom=0.26, top=0.84)
    for k, d in enumerate(rc):
        ax = fig.add_subplot(gs[0, k])
        rows = d["variants"]; fr = [r["frac"] for r in rows]
        ax.fill_between(fr, [r["random_lo"] for r in rows], [r["random_hi"] for r in rows],
                        color="#7f7f7f", alpha=0.22, label="random genes, same size")
        ax.plot(fr, [r["random_mean"] for r in rows], "o--", color="#7f7f7f", ms=2.4, lw=0.9)
        ax.plot(fr, [r["matched_mean"] for r in rows], "s--", color="#2166ac", ms=2.4, lw=0.9,
                label="detectability-matched")
        ax.plot(fr, [r["cuTAR"] for r in rows], "^-", color="#b2182b", ms=3.2, lw=1.3,
                label=f"the {d['n_cu']} surviving $cu$TARs")
        ax.set_xscale("log"); ax.invert_xaxis()
        ax.set_xticks([0.5, 0.25, 0.1]); ax.set_xticklabels(["0.50", "0.25", "0.10"])
        ax.set_xlabel("Retained sequencing depth"); ax.set_ylim(-0.06, 0.62)
        if k == 0:
            ax.set_ylabel("ARI vs full-depth partition"); ax.legend(loc="upper right")
        ax.set_title(f"{'abc'[k]}  {d['tag']}  (n$_{{cuTAR}}$={d['n_cu']})", loc="left")
    fig.text(0.5, 0.035,
             "Supplementary, non-discriminating: this experiment does not separate $cu$TARs from size-matched controls.\n"
             "Binomial thinning largely preserves depth ordering, so any depth-tracking partition looks stable here.",
             ha="center", fontsize=5.8)
    save(fig, "FigS1_non_discriminating_controls")


if __name__ == "__main__":
    # --- released-package note ---------------------------------------------
    # fig1()..fig6() and fig4b() build exploratory figures from an earlier
    # route of this project. None of them appears in the manuscript, and the
    # "partners above chance" panel of fig4b() (Fig4_threshold_vs_continuous,
    # panel c) was RETRACTED and must not be reused. They are kept here so that
    # the file is complete, but are deliberately not run.
    # The only figure in this file that the manuscript cites is Fig. S1.
    # ------------------------------------------------------------------------
    figS1()
    print(f"\n[written] {FIG}")
