# -*- coding: utf-8 -*-
r"""
audit_fig1.py — Fig. 1 的**机器可查项**审计（替代目视能查的那一部分）

本机无视觉能力，所以把「能自动查的」全部查掉，只把「必须人眼看的」留给作者。

关键设计：**直接导入 make_fig1.build()**，审计的就是被保存的那张图，
不会出现"审计的代码"与"出图的代码"各写一份而逐渐分叉的问题。

检查项：
  1. 逐面板打印每一个画上去的数值及其数据来源 -> 供与图中读数逐条对照
  2. 文字是否越出坐标区（被裁切）
  3. 文字是否压在柱体上（与柱体 bounding box 相交）
  4. 文字之间是否互相重叠
  5. 参考线是否落在坐标区**内部**（贴边=看不见）
  6. 全图最小字号（期刊排版下限）
  7. y 轴是否被截断

注：柱体 bounding box 用 `transData` 手工换算，不能用 `get_window_extent`——
对数轴上柱体从 y=0 起画，y=0 在对数轴上是 -inf，matplotlib 给出的 bbox 不可靠
（这正是本脚本第一版漏报 b 面板"文字压柱"的原因）。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, re
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_fig1 as M

ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
OUT, CHK = [], []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(os.path.join(RES, "AUDIT_FIG1.txt"), "w", encoding="utf-8").write("\n".join(OUT))


def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail))
    W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")


ident, ari, TC, SC = M.load()
TAGS = M.TAGS

W("=" * 104)
W("PART 1  画上去的数值 —— 请与图中读数逐条对照")
W("=" * 104)
W("  数据来源：results/RECHECK.json（a、c）、results/TC_GAP.json（b）、results/SELFCHECK4.json（d）")

W(f"\n  【a 面板】每矩阵「最小非零 log1p CP10K」——虚线在 0.5")
W(f"  {'矩阵':<16}{'柱高':>8}")
a_vals = []
for t in TAGS:
    v = ident[t]["min_nonzero_lognorm"]; a_vals.append(v)
    W(f"    {t:<16}{v:>8.3f}")
W(f"  范围 {min(a_vals):.3f} – {max(a_vals):.3f}  -> 稿中写 3.31–5.53")
chk("a 面板：10 根柱全部 > 0.5", all(v > 0.5 for v in a_vals), f"最小 {min(a_vals):.3f}")
chk("a 面板：范围与稿中「3.31–5.53」一致",
    round(min(a_vals), 2) == 3.31 and round(max(a_vals), 2) == 5.53,
    f"{min(a_vals):.3f} – {max(a_vals):.3f}")

W(f"\n  【b 面板】T_crit / T_max（对数轴，虚线在 1）")
W(f"  {'矩阵':<16}{'比值':>9}{'子集最深 read':>14}")
b_rat, b_tmax = [], []
for t in TAGS:
    b_rat.append(TC[t]["ratio"]); b_tmax.append(TC[t]["tmax_sub"])
    W(f"    {t:<16}{TC[t]['ratio']:>9.1f}{TC[t]['tmax_sub']:>14.0f}")
W(f"  比值范围 {min(b_rat):.1f} – {max(b_rat):.1f}  -> 稿中写 41–385×")
W(f"  最深 read 范围 {min(b_tmax):.0f} – {max(b_tmax):.0f}  -> 稿中写 40–380")
chk("b 面板：比值范围与稿中「41–385×」一致",
    round(min(b_rat)) == 41 and round(max(b_rat)) == 385, f"{min(b_rat):.1f}–{max(b_rat):.1f}")
chk("b 面板：最深 read 范围与稿中「40–380」一致",
    min(b_tmax) == 40 and max(b_tmax) == 380, f"{min(b_tmax):.0f}–{max(b_tmax):.0f}")
chk("b 面板：所有比值都 > 1", all(r > 1 for r in b_rat), f"最小 {min(b_rat):.1f}")

W(f"\n  【c 面板】Spearman(过阈特征数, 子集深度)（y 轴**自 0 起**）")
W(f"  {'矩阵':<16}{'ρ':>9}")
c_vals = []
for t in TAGS:
    v = ari[t]["sp_sub"]; c_vals.append(v)
    W(f"    {t:<16}{v:>9.4f}")
W(f"  范围 {min(c_vals):.4f} – {max(c_vals):.4f}  -> 稿中写「ρ ≥ 0.944」")
chk("c 面板：范围与稿中「ρ ≥ 0.944」一致", abs(min(c_vals) - 0.9443) < 0.001,
    f"最小 {min(c_vals):.4f}")

W(f"\n  【d 面板】三种零模型下的存活率（x 轴 0–118，虚线在 5%）")
W(f"  {'矩阵':<14}{'对':>6}{'plain':>9}{'strat':>9}{'torus':>9}")
for t in ["SCC", "BCC", "HNC", "KidneyCancer", "Melanoma"]:
    W(f"    {t:<14}{SC[t]['pairs']:>6}{SC[t]['pct_plain']:>9.1f}"
      f"{SC[t]['pct_strat']:>9.1f}{SC[t]['pct_torus']:>9.1f}")
four = ["SCC", "BCC", "HNC", "KidneyCancer"]
s4 = sum(round(SC[t]["pct_torus"] / 100 * SC[t]["pairs"]) for t in four)
p4 = sum(SC[t]["pairs"] for t in four)
chk("d 面板：四矩阵环面存活合计 = 960/970（稿中 99.0%）", s4 == 960 and p4 == 970,
    f"{s4}/{p4} = {100*s4/p4:.2f}%")
chk("d 面板：Melanoma 环面 = 40/70（稿中 57.1%）", SC["Melanoma"]["pct_torus"] == 57.1,
    f"{SC['Melanoma']['pct_torus']}%")

# =====================================================================================
W("\n" + "=" * 104)
W("PART 2  几何检查（审计对象 = make_fig1.build() 产出的同一张图）")
W("=" * 104)
fig, meta = M.build()
fig.canvas.draw()
rend = fig.canvas.get_renderer()


def bbox_text(t):
    return t.get_window_extent(renderer=rend)


def bbox_bar(ax, p):
    """手工换算柱体 bbox：y 下界取轴下界（对数轴上 y=0 是 -inf，不能直接用）。"""
    x0, x1 = p.get_x(), p.get_x() + p.get_width()
    y0d = p.get_y(); y1d = p.get_y() + p.get_height()
    lo, hi = min(y0d, y1d), max(y0d, y1d)
    axlo, axhi = ax.get_ylim()
    lo = max(lo, min(axlo, axhi)); hi = min(hi, max(axlo, axhi))
    pts = ax.transData.transform([(x0, lo), (x1, lo), (x0, hi), (x1, hi)])
    xs, ys = pts[:, 0], pts[:, 1]
    from matplotlib.transforms import Bbox
    return Bbox([[xs.min(), ys.min()], [xs.max(), ys.max()]])


def inter(r1, r2):
    x0 = max(r1.x0, r2.x0); x1 = min(r1.x1, r2.x1)
    y0 = max(r1.y0, r2.y0); y1 = min(r1.y1, r2.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


mins = []
for pname, (ax, patches) in meta.items():
    ab = ax.get_window_extent(renderer=rend)
    texts = list(ax.texts)
    over, onbar, pairs, refs = [], [], [], []
    for t in texts:
        b = bbox_text(t)
        if not (ab.x0 - 0.5 <= b.x0 and b.x1 <= ab.x1 + 0.5
                and ab.y0 - 0.5 <= b.y0 and b.y1 <= ab.y1 + 0.5):
            over.append((t.get_text()[:34], round(b.x1 - ab.x1, 1), round(ab.y0 - b.y0, 1)))
        tb = b
        for p in patches:
            pb = bbox_bar(ax, p)
            ov = inter(tb, pb)
            if ov > 0.30 * tb.width * tb.height:
                onbar.append((t.get_text()[:26], round(100 * ov / (tb.width * tb.height))))
                break
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a_, b_ = bbox_text(texts[i]), bbox_text(texts[j])
            ov = inter(a_, b_)
            if ov > 0.15 * min(a_.width * a_.height, b_.width * b_.height):
                pairs.append((texts[i].get_text()[:18], texts[j].get_text()[:18]))
    for ln in ax.lines:
        yd, xd = ln.get_ydata(), ln.get_xdata()
        y0d, y1d = ax.get_ylim(); x0d, x1d = ax.get_xlim()
        if len(set(yd)) == 1 and len(xd) == 2:
            v = yd[0]
            frac = ((np.log10(v) - np.log10(min(y0d, y1d))) /
                    (np.log10(max(y0d, y1d)) - np.log10(min(y0d, y1d)))
                    if ax.get_yscale() == "log" and min(y0d, y1d) > 0
                    else (v - min(y0d, y1d)) / (max(y0d, y1d) - min(y0d, y1d)))
            refs.append(("hline", v, round(100 * frac, 1)))
        if len(set(xd)) == 1 and len(yd) == 2:
            v = xd[0]
            frac = (v - min(x0d, x1d)) / (max(x0d, x1d) - min(x0d, x1d))
            refs.append(("vline", v, round(100 * frac, 1)))
    fs = [t.get_fontsize() for t in texts]
    fs += [ax.title.get_fontsize()]
    fs += [l.get_fontsize() for l in ax.get_xticklabels()]
    fs += [l.get_fontsize() for l in ax.get_yticklabels()]
    fs += [ax.xaxis.label.get_fontsize(), ax.yaxis.label.get_fontsize()]
    lg = ax.get_legend()
    if lg is not None:
        fs += [t.get_fontsize() for t in lg.get_texts()]
    mins += fs
    W(f"\n  【面板 {pname}】")
    W(f"    越界文字：{len(over)}   " + ("；".join(f"{o[0]}" for o in over) if over else "无"))
    W(f"    压柱文字：{len(onbar)} " + ("；".join(f"{o[0]}({o[1]}%)" for o in onbar) if onbar else "无"))
    W(f"    文字重叠：{len(pairs)} " + ("；".join(f"{a}×{b}" for a, b in pairs) if pairs else "无"))
    W(f"    参考线位置（占轴长百分比）：" +
      ("；".join(f"{k}@{v}={f}%" for k, v, f in refs) if refs else "无"))
    W(f"    最小字号：{min(fs):.1f} pt")
    chk(f"{pname} 面板：没有文字越出坐标区", not over,
        f"{len(over)} 处" + (f"：{over[0][0]}" if over else ""))
    chk(f"{pname} 面板：没有文字压在柱体上", not onbar,
        f"{len(onbar)} 处" + (f"：{onbar[0][0]} {onbar[0][1]}%" if onbar else ""))
    chk(f"{pname} 面板：文字之间没有重叠", not pairs, f"{len(pairs)} 对")
    for k, v, f in refs:
        chk(f"{pname} 面板：参考线 {k}@{v} 落在内部（不贴边）", 0.5 < f < 99.5,
            f"位于轴长 {f}% 处")

W(f"\n  全图最小字号 = {min(mins):.1f} pt")
chk("全图最小字号 ≥ 6.0 pt", min(mins) >= 6.0, f"实测最小 {min(mins):.1f} pt")

W("\n  轴范围（截断轴会放大差异）：")
for pname, (ax, _) in meta.items():
    y0, y1 = sorted(ax.get_ylim())
    tag = "  <- **未从 0 起**" if (y0 > 0 and pname in ("a", "c")) else ""
    W(f"    {pname} 面板 y：{y0:g} – {y1:g}{tag}")
chk("c 面板 y 轴自 0 起（不再截断）", min(meta["c"][0].get_ylim()) <= 0.0,
    f"ylim={meta['c'][0].get_ylim()}")
chk("a 面板 y 轴自 0 起", min(meta["a"][0].get_ylim()) <= 0.0,
    f"ylim={meta['a'][0].get_ylim()}")

W("\n" + "=" * 104)
W("PART 3  图例几何（此前审计的盲区：图例文字**不在** ax.texts 里，故 PART 2 查不到）")
W("=" * 104)
for pname, (ax, patches) in meta.items():
    lg = ax.get_legend()
    if lg is None:
        W(f"  【面板 {pname}】无图例")
        chk(f"{pname} 面板：不使用图例（改用直接标注）则无图例-柱体相交风险", True, "无图例")
        continue
    lb = lg.get_window_extent(renderer=rend)
    ab = ax.get_window_extent(renderer=rend)
    inside = (ab.x0 <= lb.x0 and lb.x1 <= ab.x1 and ab.y0 <= lb.y0 and lb.y1 <= ab.y1)
    hits = []
    for p in patches:
        pb = bbox_bar(ax, p)
        ov = inter(lb, pb)
        if ov > 0:
            hits.append(round(100 * ov / (pb.width * pb.height)))
    gaps = []
    hs = lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles
    for t, h in zip(lg.get_texts(), hs):
        # 图例色块**不是**数据坐标里的柱体，必须直接用 get_window_extent；
        # 用 bbox_bar() 会把它当作数据柱去 transform，得到无意义的 −332 px。
        gaps.append(bbox_text(t).x0 - h.get_window_extent(renderer=rend).x1)
    W(f"  【面板 {pname}】图例框 {lb.width:.0f}×{lb.height:.0f} px，位于轴内={inside}；"
      f"与柱体相交 {len(hits)} 根（最大占柱面积 {max(hits) if hits else 0}%）；"
      f"色块→文字最小间距 {min(gaps):+.1f} px")
    chk(f"{pname} 面板：图例框不与任何柱体相交", not hits,
        f"相交 {len(hits)} 根，最大 {max(hits) if hits else 0}%" if hits else "无相交")
    chk(f"{pname} 面板：图例框完整位于坐标区内", inside, f"位于轴内={inside}")
    chk(f"{pname} 面板：图例色块与文字不重叠（间距 > 2 px）", min(gaps) > 2,
        f"最小间距 {min(gaps):+.1f} px")

W("\n" + "=" * 104)
W("PART 4  配色一致性：同一颜色在**语义上**必须是同一件事")
W("=" * 104)
import matplotlib.colors as mc
# 语义分组：a/b/c 共用「平台」语义（图例在 a），d 自带「零模型」语义（图例在 d）
sem = {}
for pname, (ax, patches) in meta.items():
    hs = []
    for p in patches:
        h = mc.to_hex(p.get_facecolor())
        if h not in hs:
            hs.append(h)
    lg = ax.get_legend()
    labels = [t.get_text() for t in lg.get_texts()] if lg is not None else []
    # 没有图例时（d 面板改为直接标注），语义从面板内的标注文字读取
    if not labels and pname == "d":
        labels = [t.get_text() for t in ax.texts if t.get_text() in
                  ("plain permutation", "depth-stratified", "toroidal (strict)")]
    W(f"  面板 {pname} 用色：{'  '.join(hs)}"
      + (f"\n           语义：{labels}" if labels else "   （沿用 a 面板的平台图例）"))
    if pname == "d":
        for h, lb in zip(hs, labels):
            sem.setdefault(h, set()).add(("d", lb.replace("$", "")))
    else:
        # a/b/c 三色对应平台图例的三项，顺序一致
        for h, lb in zip(hs, labels if labels else
                         ["PacBio, cuTAR-only", "Nanopore, cuTAR-only", "short-read, mixed"]):
            sem.setdefault(h, set()).add(("abc", lb.replace("$", "")))
conflicts = {h: v for h, v in sem.items() if len({m for _, m in v}) > 1}
W("")
for h, v in sem.items():
    W(f"    {h} -> " + " | ".join(f"{g}:{m}" for g, m in sorted(v)))
chk("跨面板没有同一颜色表示两件**不同**的事", not conflicts,
    "冲突：" + "; ".join(f"{h}={sorted(m for _, m in v)}" for h, v in conflicts.items())
    if conflicts else "六个颜色语义各不相同")
abc = {mc.to_hex(p.get_facecolor()) for p in meta["a"][1]}
dfc = {mc.to_hex(p.get_facecolor()) for p in meta["d"][1]}
chk("d 面板配色与 a/b/c 的平台色完全不重合", not (abc & dfc),
    f"交集：{sorted(abc & dfc)}" if (abc & dfc) else f"a/b/c={sorted(abc)} d={sorted(dfc)}")

W("\n" + "=" * 104)
W(f"汇总：{sum(1 for _, c, _ in CHK if c)}/{len(CHK)} 项通过")
W("=" * 104)
for n, c, d in CHK:
    if not c:
        W(f"  未通过：{n}  ({d})")
open(os.path.join(RES, "AUDIT_FIG1.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\AUDIT_FIG1.txt")
