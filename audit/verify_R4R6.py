# -*- coding: utf-8 -*-
"""verify_R4R6.py — 核查 R4–R6 中每一个数字，全部从源 JSON / 原始计算结果独立求值"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, json, re
sys.stdout.reconfigure(encoding="utf-8")
ROOT = _cfg.WORKSPACE + r""
RES = os.path.join(ROOT, "results")
OUT, CHK = [], []
def W(s=""):
    print(s, flush=True); OUT.append(s)
def chk(name, cond, detail=""):
    CHK.append((name, bool(cond), detail)); W(f"  [{'PASS' if cond else '**FAIL**'}] {name}   {detail}")
def J(f): return json.load(open(os.path.join(RES, f), encoding="utf-8"))

sc4 = {d["tag"]: d for d in J("MORANS_stratified.json")}
sch = {d["tag"]: d for d in J("SELFCHECK4.json")}
mc  = {d["tag"]: d for d in J("MORANS_conditioned.json")}
rc4 = {d["tag"]: d for d in J("RECHECK.json")["c4"]}
u2  = J("U2_spatial_consistency.json")
scr = J("SCRNA_probe.json")

W("=" * 104); W("R4 核查"); W("=" * 104)
MT = ["SCC", "BCC", "HNC", "KidneyCancer", "Melanoma"]
tot_pairs = sum(sch[t]["pairs"] for t in MT)          # torus 结果在 SELFCHECK4
chk("配对总数 = 1,040", tot_pairs == 1040,
    " + ".join(str(sch[t]["pairs"]) for t in MT) + f" = {tot_pairs}")
chk("MORANS_stratified 与 SELFCHECK4 的配对计数一致",
    all(sch[t]["pairs"] == sc4[t]["pairs"] for t in MT),
    "  ".join(f"{t}:{sch[t]['pairs']}/{sc4[t]['pairs']}" for t in MT))
four = [t for t in MT if t != "Melanoma"]
p4 = sum(sch[t]["pairs"] for t in four)
chk("四个矩阵的配对 = 970", p4 == 970, f"{p4}")
n_sig4 = sum(round(sch[t]["pct_torus"] / 100.0 * sch[t]["pairs"]) for t in four)
chk("其中环面平移显著 = 960（99.0%）", n_sig4 == 960 and round(100 * n_sig4 / p4, 1) == 99.0,
    f"{n_sig4}/{p4} = {100*n_sig4/p4:.1f}%")
m = sch["Melanoma"]
n_m = round(m["pct_torus"] / 100.0 * m["pairs"])
chk("Melanoma 40/70 = 57.1%", n_m == 40 and round(100 * n_m / m["pairs"], 1) == 57.1,
    f"{n_m}/{m['pairs']} = {100*n_m/m['pairs']:.1f}%")

# 头部伙伴倍数（用修正后的 k^2/N 基线）
# 注意口径：该倍数 = top_overlap（变体 A 与变体 B 的 top-10 伙伴「名单重叠率」）÷ 10²/N，
# 衡量「换一种去深度的做法后名单还有多少保留」，**不是**「最佳伙伴强度高于随机多少倍」。
# 正文原先把它写成后者，已删改（见 results/VERIFY_R4_INDEPENDENT.txt PART F 与 DIAG_R4_FOLLOWUP.txt）。
ch = {d["tag"]: d for d in J("SELFCHECK4.json")}
ratios = {t: mc[t]["top_overlap"] / ch[t]["chance_overlap"] for t in four}
lo, hi = min(ratios.values()), max(ratios.values())
chk("top-10 伙伴名单重叠率 ÷ 随机基线 = 31–38 倍（口径为名单重叠，非伙伴强度）",
    round(lo) == 31 and round(hi) == 38,
    "  ".join(f"{t}={ratios[t]:.1f}" for t in four))

# U2
rr = [p["rho"] for p in u2["pairs"]]
chk("空间-空间 45 对，中位 = 0.241",
    len(rr) == 45 and abs(sorted(rr)[len(rr)//2] - 0.241) < 0.002,
    f"n={len(rr)}, median={sorted(rr)[len(rr)//2]:.3f}")
d = {(p["a"], p["b"]): p["rho"] for p in u2["pairs"]}
bcc = d.get(("BCC", "BCC_nano"), d.get(("BCC_nano", "BCC")))
scc = d.get(("SCC", "SCC_nano"), d.get(("SCC_nano", "SCC")))
chk("BCC 同癌种两平台 = 0.629", bcc is not None and abs(bcc - 0.629) < 0.002, f"{bcc}")
chk("SCC 同癌种两平台 = 0.647", scc is not None and abs(scc - 0.647) < 0.002, f"{scc}")
sc_rhos = [v["rho"] for v in u2["spatial_scRNA"].values()]
chk("空间-单细胞 中位 = 0.099", abs(sorted(sc_rhos)[len(sc_rhos)//2] - 0.099) < 0.002,
    f"{sorted(sc_rhos)}")
chk("U2 的跨模态值与 SCRNA_probe 一致",
    all(abs(v["rho"] - next(d["spearman"] for d in scr["cross_platform"] if d["pair"] == k)) < 1e-9
        for k, v in u2["spatial_scRNA"].items()),
    "5/5 逐项比对")

W("\n" + "=" * 104); W("R5 核查"); W("=" * 104)
pc = {t: rc4[t]["pc_ret_pct"] for t in ["BCC", "SCC", "HNC", "KidneyCancer"]}
cu = {t: rc4[t]["cu_ret_pct"] for t in ["BCC", "SCC", "HNC", "KidneyCancer"]}
chk("蛋白编码保留 35.3–55.9%",
    abs(min(pc.values()) - 35.27) < 0.01 and abs(max(pc.values()) - 55.90) < 0.01,
    "  ".join(f"{t}={pc[t]:.2f}" for t in pc))
chk("cuTAR 保留 0.13–0.70%",
    abs(min(cu.values()) - 0.127) < 0.001 and abs(max(cu.values()) - 0.698) < 0.001,
    "  ".join(f"{t}={cu[t]:.3f}" for t in cu))
chk("具体分数与正文一致 (BCC 5,197/14,735; SCC 6,570/14,562; HNC 7,742/13,850; Kidney 7,592/14,530)",
    (rc4["BCC"]["pc10"], rc4["BCC"]["n_pc"]) == (5197, 14735)
    and (rc4["SCC"]["pc10"], rc4["SCC"]["n_pc"]) == (6570, 14562)
    and (rc4["HNC"]["pc10"], rc4["HNC"]["n_pc"]) == (7742, 13850)
    and (rc4["KidneyCancer"]["pc10"], rc4["KidneyCancer"]["n_pc"]) == (7592, 14530))
chk("cuTAR 分数与正文一致 (5/3,759; 18/5,322; 26/3,723; 7/5,506)",
    (rc4["BCC"]["cu10"], rc4["BCC"]["n_cu"]) == (5, 3759)
    and (rc4["SCC"]["cu10"], rc4["SCC"]["n_cu"]) == (18, 5322)
    and (rc4["HNC"]["cu10"], rc4["HNC"]["n_cu"]) == (26, 3723)
    and (rc4["KidneyCancer"]["cu10"], rc4["KidneyCancer"]["n_cu"]) == (7, 5506))
rmin = min(pc[t] / cu[t] for t in pc); rmax = max(pc[t] / cu[t] for t in pc)
chk("倍数 = 80–411", round(rmin) == 80 and round(rmax) == 411, f"{rmin:.1f} – {rmax:.1f}")
chk("Melanoma 保留 2.37% 与 0.50%",
    abs(rc4["Melanoma"]["pc_ret_pct"] - 2.37) < 0.01 and abs(rc4["Melanoma"]["cu_ret_pct"] - 0.499) < 0.001,
    f"{rc4['Melanoma']['pc_ret_pct']:.2f}% / {rc4['Melanoma']['cu_ret_pct']:.3f}%")
chk("单细胞：22/1,314 且 55 个细胞类型受限",
    scr["n_cuTAR"] == 1314 and scr["n_cuTAR_ge10pct_cells"] == 22,
    f"n_cuTAR={scr['n_cuTAR']}, ≥10%细胞={scr['n_cuTAR_ge10pct_cells']}")
# 55：需从 SELFCHECK/SCRNA 重算
try:
    z = __import__("numpy").load(os.path.join(RES, "SCRNA_pergene.npz"), allow_pickle=True)
    n_ct = int((z["n_ct_ge10"] == 1).sum())
    chk("55 个仅在 1 类细胞中 ≥10%", n_ct == 55, f"{n_ct}")
except Exception as e:
    chk("55 个仅在 1 类细胞中 ≥10%", False, f"无法重算：{e}")

W("\n" + "=" * 104); W("R6 核查"); W("=" * 104)
rh = open(os.path.join(RES, "SPANCLNC_review_history.txt"), encoding="utf-8").read()
chk("31.7% 出自审稿档案原文", "31.7% of 3' ends" in re.sub(r"\s+", " ", rh),
    "命中" if "31.7%" in rh else "未命中")
chk("作者降采样论证出自审稿档案", "downsample the original Visium reads" in re.sub(r"\s+", " ", rh))

W("\n" + "=" * 104)
W("R4 底层的独立性说明")
W("=" * 104)
W("  本脚本以 results/*.json 为输入（独立性较弱）。R4 底层的 Moran's I 与环面平移")
W("  已由 audit/verify_R4_independent.py 第二次独立实现：换近邻搜索、换归一化结合序、")
W("  换空间滞后累加方式、把 Moran's I 展开为双重求和定义、换环面平移最近邻算法，")
W("  并逐调用复刻 default_rng(99) 的抽取序列。结果见 results/VERIFY_R4_INDEPENDENT.txt。")
W("  成因诊断（k=6 近邻的并列取舍、dep 口径对 RNG 流的影响）见 results/DIAG_R4_FOLLOWUP.txt。")
for _f in ("VERIFY_R4_INDEPENDENT.txt", "DIAG_R4_FOLLOWUP.txt", "R4_METHODS_NUMBERS.txt"):
    _p = os.path.join(RES, _f)
    chk(f"{_f} 存在（独立性证据已落盘）", os.path.exists(_p),
        f"{os.path.getsize(_p):,} 字节" if os.path.exists(_p) else "缺失")

W("\n" + "=" * 104)
W(f"汇总：{sum(1 for _,c,_ in CHK if c)}/{len(CHK)} 项通过")
W("=" * 104)
for n, c, d in CHK:
    if not c: W(f"  未通过：{n}  ({d})")
open(os.path.join(RES, "VERIFY_R4R6.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("\n[written] results\\VERIFY_R4R6.txt")
