# -*- coding: utf-8 -*-
r"""
make_fig1.py — 论文主图（四联合成图）

  a  判据恒等：10 个矩阵的最小非零 logNorm 全部位于 0.5 之上（0 不匹配）
  b  深度界：T_crit / T_max —— 阈值在比值降到 1 以下之前都不含信息
  c  后果：掩码计数随子集深度单调（y 轴**从 0 起**，数值标在柱顶）
  d  内对照：同数据、同值、同特征，连续统计量在三种零模型下存活

设计约束（由 audit/audit_fig1.py 的几何检查驱动）：
  · 全图最小字号 ≥ 6.0 pt
  · 任何文字不得越出坐标区、不得压在柱体上、不得互相重叠
  · 参考线必须落在坐标区**内部**（不能与边框重合以致看不见）
  · c 面板 y 轴自 0 起（避免截断轴放大微小差异，与"近乎确定性"的论点相冲突）

本模块对外提供 build()，返回 (fig, axes_meta)；audit_fig1.py **导入同一函数**做几何审计，
以保证「被审计的图」与「被保存的图」是同一份代码产物。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

FS = 6.0          # 全图最小字号
MM = 1 / 25.4

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.5,
    "axes.linewidth": 0.6, "axes.labelsize": 6.5, "axes.titlesize": 7.0,
    "xtick.labelsize": FS, "ytick.labelsize": FS,
    "legend.fontsize": FS, "legend.frameon": False,
    "figure.dpi": 200, "savefig.dpi": 400, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ORDER = [("CM_pacbio", "CM\n(PacBio)"), ("CP_pacbio", "CP\n(PacBio)"),
         ("HNC_ilong_nano", "HNC\n(ONT)"), ("BCC_nano", "BCC\n(ONT)"), ("SCC_nano", "SCC\n(ONT)"),
         ("Melanoma", "Mela-\nnoma"), ("BCC", "BCC"), ("KidneyCancer", "Kid-\nney"),
         ("HNC", "HNC"), ("SCC", "SCC")]
TAGS = [t for t, _ in ORDER]
LAB = [l for _, l in ORDER]
GRP = {"CM_pacbio": "PacBio", "CP_pacbio": "PacBio", "HNC_ilong_nano": "ONT",
       "BCC_nano": "ONT", "SCC_nano": "ONT", "Melanoma": "SR", "BCC": "SR",
       "KidneyCancer": "SR", "HNC": "SR", "SCC": "SR"}
COL = {"PacBio": "#b2182b", "ONT": "#ef8a62", "SR": "#2166ac"}


def load():
    rc = json.load(open(os.path.join(RES, "RECHECK.json"), encoding="utf-8"))
    ident = {d["tag"]: d for d in rc["identity"]}
    ari = {d["tag"]: d for d in rc["ari"]}
    TC = {d["tag"]: d for d in json.load(open(os.path.join(RES, "TC_GAP.json"), encoding="utf-8"))}
    SC = {d["tag"]: d for d in json.load(open(os.path.join(RES, "SELFCHECK4.json"), encoding="utf-8"))}
    return ident, ari, TC, SC


def build():
    """画出四联合成图，返回 (fig, axes_meta)。axes_meta[面板] = (ax, patches, ...)"""
    ident, ari, TC, SC = load()
    fig = plt.figure(figsize=(183 * MM, 110 * MM))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.42, left=0.078, right=0.985,
                          bottom=0.085, top=0.945)
    meta = {}

    # ---------------- a  判据恒等
    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(TAGS))
    vals = [ident[t]["min_nonzero_lognorm"] for t in TAGS]
    bars = ax.bar(xs, vals, 0.62, color=[COL[GRP[t]] for t in TAGS])
    ax.axhline(0.5, color="k", ls="--", lw=1.0)
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=FS)
    ax.set_ylabel("Smallest non-zero $\\log_1p$ CP10K")
    ax.set_ylim(0, 7.2)
    ax.set_yticks([0, 0.5, 1, 2, 3, 4, 5, 6])   # 0.5 直接标在轴上，与虚线对齐
    for i, t in enumerate(TAGS):
        ax.text(i, vals[i] + 0.12, f"{vals[i]:.2f}", ha="center", fontsize=FS)
    # 说明文字放在柱顶以上的空白区（左侧），不与任何柱体相交
    ax.text(0.0, 6.95, "dashed line = cut-off 0.5", ha="left", va="top", fontsize=FS,
            color="k")
    ax.set_title("a  The cut-off lies below every non-zero value\n"
                 "     $(\\log_1p\\,\\mathrm{CP10K} > 0.5)\\equiv(\\mathrm{count}>0)$: 0 mismatches in 10/10",
                 loc="left")
    h = [Patch(color=COL[k], label=v) for k, v in
         [("PacBio", "PacBio, $cu$TAR-only"), ("ONT", "Nanopore, $cu$TAR-only"),
          ("SR", "short-read, mixed")]]
    ax.legend(handles=h, loc="upper right", ncol=1, fontsize=FS)
    ax.set_xlim(-0.65, 9.65)
    meta["a"] = (ax, bars)

    # ---------------- b  深度界
    ax = fig.add_subplot(gs[0, 1])
    ratios = [TC[t]["ratio"] for t in TAGS]
    bars = ax.bar(xs, ratios, 0.62, color=[COL[GRP[t]] for t in TAGS])
    ax.set_yscale("log")
    ax.set_ylim(0.1, 2200)           # 下界 0.1：使 y=1 的参考线远离底边，量级差距一眼可见
    ax.axhline(1, color="k", ls="--", lw=1.0)
    ax.set_yticks([0.1, 1, 10, 100, 1000])
    ax.set_yticklabels(["0.1", "1", "10", "100", "1000"])
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=FS)
    ax.set_ylabel("$T_{crit}\\,/\\,T_{max}$  (feature subset)")
    for i, t in enumerate(TAGS):
        ax.text(i, ratios[i] * 1.15, f"{ratios[i]:.0f}", ha="center", fontsize=FS)
    # 说明文字放在最高柱（385）之上：y>500 全宽度均无柱
    ax.text(0.0, 2050, "dashed line = 1: above it the cut-off is informative", ha="left",
            va="top", fontsize=FS)
    ax.set_title("b  Every matrix is 41-385x below the bound\n"
                 "     deepest $cu$TAR-bearing spot: 40-380 reads", loc="left")
    ax.set_xlim(-0.65, 9.65)
    meta["b"] = (ax, bars)

    # ---------------- c  后果（y 轴自 0 起）
    ax = fig.add_subplot(gs[1, 0])
    sp = [ari[t]["sp_sub"] for t in TAGS]
    bars = ax.bar(xs, sp, 0.62, color=[COL[GRP[t]] for t in TAGS])
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks(xs); ax.set_xticklabels(LAB, fontsize=FS)
    ax.set_ylabel("Spearman(features above cut, subset depth)")
    for i, t in enumerate(TAGS):
        ax.text(i, sp[i] + 0.02, f"{sp[i]:.4f}", ha="center", va="bottom", fontsize=FS)
    ax.text(0.0, 1.20, "bars are nearly equal in height: the mask is close to a\n"
                       "deterministic function of depth in every matrix", ha="left", va="top",
            fontsize=FS)
    ax.set_title("c  The mask counts depth, not expression", loc="left")
    ax.set_xlim(-0.65, 9.65)
    meta["c"] = (ax, bars)

    # ---------------- d  内对照
    ax = fig.add_subplot(gs[1, 1])
    MT = ["SCC", "BCC", "HNC", "KidneyCancer", "Melanoma"]
    mlab = ["SCC", "BCC", "HNC", "Kidney", "Melanoma"]
    yy = np.arange(len(MT)); hh = 0.26
    # 配色刻意**不与 a/b/c 的平台色重叠**：a/b/c 用红/橙/蓝表示平台，
    # d 面板改用绿色系表示三种零模型的严格程度（原先 d 用了 #2166ac，与 a/b/c 的
    # "short-read, mixed" 同色，同一张图里同一颜色表示两件事——已修正）。
    b1 = ax.barh(yy - hh, [SC[t]["pct_plain"] for t in MT], hh, color="#c7e9c0",
                 label="plain permutation")
    b2 = ax.barh(yy, [SC[t]["pct_strat"] for t in MT], hh, color="#74c476",
                 label="depth-stratified")
    b3 = ax.barh(yy + hh, [SC[t]["pct_torus"] for t in MT], hh, color="#238b45",
                 label="toroidal shift (strict)")
    ax.axvline(5, color="k", ls="--", lw=1.0)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{l}  (n={SC[t]['pairs']})" for t, l in zip(MT, mlab)], fontsize=FS)
    ax.invert_yaxis()
    ax.set_ylim(4.75, -0.75)
    # 不再用图例：图例框会与 0–100% 的条形相交（实测最大占柱面积 24%）。
    # 改为**直接把三组标签标在第一行右侧的空白边距**（x>101 处所有行都无柱），
    # 并把 "5%" 的说明并入 x 轴标签，从而彻底消除图内文字与柱体的相交。
    ax.set_xlim(0, 148)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    LBLX = 104.0
    for ypos, txt in ((yy[0] - hh, "plain permutation"),
                      (yy[0], "depth-stratified"),
                      (yy[0] + hh, "toroidal (strict)")):
        ax.text(LBLX, ypos, txt, ha="left", va="center", fontsize=FS)
    ax.set_xlabel("% of $cu$TAR-gene associations significant (bivariate Moran's $I$, same data "
                  "and values)\ndashed line = 5% null expectation")
    ax.set_title("d  Continuous statistics survive the same depths", loc="left")
    meta["d"] = (ax, b1 + b2 + b3)

    return fig, meta


def main():
    fig, meta = build()
    fig.savefig(os.path.join(FIG, "Fig1_composite.png"))
    fig.savefig(os.path.join(FIG, "Fig1_composite.pdf"))
    plt.close(fig)
    ident, ari, TC, SC = load()
    print("[fig] Fig1_composite")
    print("\n数值核对：")
    for t in TAGS:
        print(f"  {t:<16} minNZnlogNorm={ident[t]['min_nonzero_lognorm']:.3f}  "
              f"T_max={TC[t]['tmax_sub']:.0f}  Tcrit/Tmax={TC[t]['ratio']:.1f}  "
              f"Spearman={ari[t]['sp_sub']:.4f}")


if __name__ == "__main__":
    main()
