# -*- coding: utf-8 -*-
"""audit_legend.py — 图例几何审计（此前审计的盲区：图例文字不在 ax.texts 里）"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import make_fig1 as M

fig, meta = M.build()
fig.canvas.draw()
rend = fig.canvas.get_renderer()


def bb(a):
    return a.get_window_extent(renderer=rend)


def inter(r1, r2):
    x0 = max(r1.x0, r2.x0); x1 = min(r1.x1, r2.x1)
    y0 = max(r1.y0, r2.y0); y1 = min(r1.y1, r2.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def bbox_bar(ax, p):
    from matplotlib.transforms import Bbox
    x0, x1 = p.get_x(), p.get_x() + p.get_width()
    ylo, yhi = sorted([p.get_y(), p.get_y() + p.get_height()])
    a, b = sorted(ax.get_ylim())
    ylo = max(ylo, a); yhi = min(yhi, b)
    pts = ax.transData.transform([(x0, ylo), (x1, ylo), (x0, yhi), (x1, yhi)])
    xs, ys = pts[:, 0], pts[:, 1]
    return Bbox([[xs.min(), ys.min()], [xs.max(), ys.max()]])


for pname, (ax, patches) in meta.items():
    lg = ax.get_legend()
    if lg is None:
        print(f"\n【面板 {pname}】无图例")
        continue
    lb = bb(lg)
    ab = ax.get_window_extent(renderer=rend)
    print(f"\n【面板 {pname}】图例")
    print(f"  图例框：{lb.width:.0f}×{lb.height:.0f} px   位于轴内? "
          f"{'是' if (ab.x0<=lb.x0 and lb.x1<=ab.x1 and ab.y0<=lb.y0 and lb.y1<=ab.y1) else '**否/部分**'}")
    # 图例 vs 柱体
    hit = []
    for p in patches:
        pb = bbox_bar(ax, p)
        ov = inter(lb, pb)
        if ov > 0: hit.append(round(100*ov/(pb.width*pb.height)))
    print(f"  图例框与柱体相交的柱数：{len(hit)}"
          + (f"  最大相交占柱面积 {max(hit)}%" if hit else ""))
    # 图例内部：每条的文字 vs 色块
    texts = lg.get_texts()
    hs = lg.legend_handles if hasattr(lg, "legend_handles") else lg.legendHandles
    for t, h in zip(texts, hs):
        tb, hb = bb(t), bb(h)
        gap = tb.x0 - hb.x1
        flag = "**重叠**" if gap < 0 else ("**过近**" if gap < 2 else "ok")
        print(f"    「{t.get_text()[:24]:<26}」色块右缘→文字左缘 间距 {gap:+.1f} px   {flag}")
    # 图例文字 vs 面板内其它文字
    for t in texts:
        tb = bb(t)
        for o in ax.texts:
            ob = bb(o)
            ov = inter(tb, ob)
            if ov > 0.10 * tb.width * tb.height:
                print(f"    **图例文字「{t.get_text()[:20]}」与面板文字「{o.get_text()[:20]}」重叠**")
    # 图例文字互相之间
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            a_, b_ = bb(texts[i]), bb(texts[j])
            if inter(a_, b_) > 0:
                print(f"    **图例内文字重叠：「{texts[i].get_text()[:18]}」×「{texts[j].get_text()[:18]}」**")
