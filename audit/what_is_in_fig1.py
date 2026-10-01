# -*- coding: utf-8 -*-
"""what_is_in_fig1.py — 打印**当前磁盘上那张图**的实际属性，用于判定读到的是哪一版"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mc
import make_fig1 as M

ROOT = _cfg.WORKSPACE + r""
for f in ("Fig1_composite.png", "Fig1_composite.pdf"):
    p = os.path.join(ROOT, "results", "figures", f)
    print(f"{f}: {os.path.getsize(p):,} 字节   修改时间 {__import__('time').strftime('%Y-%m-%d %H:%M:%S', __import__('time').localtime(os.path.getmtime(p)))}")

fig, meta = M.build()
fig.canvas.draw()

print("\n" + "=" * 92)
print("当前代码产出的图（= 磁盘上那一版）实际长什么样")
print("=" * 92)

ax, bars = meta["a"]
print(f"\n【a 面板】")
print(f"  y 轴范围 {ax.get_ylim()}   y 刻度 {[t.get_text() for t in ax.get_yticklabels()]}")
print(f"  文字：{[t.get_text() for t in ax.texts]}")
lg = ax.get_legend()
print(f"  图例项：{[t.get_text() for t in lg.get_texts()]}")

ax, bars = meta["b"]
print(f"\n【b 面板】")
print(f"  y 轴范围 {ax.get_ylim()}  （底边 = {min(ax.get_ylim())}）")
print(f"  y 刻度标注：{[t.get_text() for t in ax.get_yticklabels()]}"
      f"   <- 十进制而非 10⁰/10¹/10²")
ln = [l for l in ax.lines if len(set(l.get_ydata())) == 1][0]
print(f"  参考线 y = {ln.get_ydata()[0]}，位于轴长 "
      f"{100*((ln.get_ydata()[0]-min(ax.get_ylim()))/(max(ax.get_ylim())-min(ax.get_ylim()))):.1f}% 处")
print(f"  文字：{[t.get_text()[:40] for t in ax.texts]}")

ax, bars = meta["c"]
print(f"\n【c 面板】")
print(f"  y 轴范围 {ax.get_ylim()}   <- 自 0 起，**不再从 0.85 截断**")

ax, bars = meta["d"]
print(f"\n【d 面板】")
cols = []
for p in bars:
    h = mc.to_hex(p.get_facecolor())
    if h not in cols:
        cols.append(h)
print(f"  三色：{cols}")
print(f"  图例项：{[t.get_text() for t in ax.get_legend().get_texts()]}")
print(f"  -> toroidal shift 的条是 {'绿色' if cols[-1].startswith('#')and cols[-1]!='#2166ac' else '深蓝'} {cols[-1]}"
      f"（旧版是深蓝 #2166ac）")
