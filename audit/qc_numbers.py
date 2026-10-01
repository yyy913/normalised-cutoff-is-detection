# -*- coding: utf-8 -*-
"""audit/qc_numbers.py —— 质检纪律「规则 1」的执行工具

用途：把文档（图注 / 表注 / 正文）中的每个数字，回到 results/*.txt 中查找出处。
      找不到出处的数字 → 需人工复核（可能是凭记忆写的）。

用法：
    python audit/qc_numbers.py 论文图例设计.md
    python audit/qc_numbers.py 案例研究_元数据缺陷污染分析输入.md
    python audit/qc_numbers.py 论文英文稿_v1.md --min-len 4

输出：屏幕报告 + results/QC_<名字>.txt
"""
from __future__ import annotations

import os
import re
import sys
import glob
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

# 这些数字太常见，不值得查（页码、年份、版本号等）
IGNORE = {
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", "1990", "2000",
    "7.1.0", "0.1", "0.2", "0.5",
}


def load_corpus(results_dir):
    """把所有 results/*.txt 读成一个整体语料（含文件名索引）。"""
    corpus = {}
    for p in glob.glob(os.path.join(results_dir, "*.txt")):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                corpus[os.path.basename(p)] = fh.read()
        except Exception:
            pass
    # json 也纳入（结构化结果）
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                corpus[os.path.basename(p)] = fh.read()
        except Exception:
            pass
    return corpus


def variants(tok):
    """一个数字的多种写法：22,072 / 22072 / 22072.0"""
    out = {tok}
    raw = tok.replace(",", "")
    out.add(raw)
    if raw.isdigit():
        out.add(f"{int(raw):,}")
        out.add(f"{raw}.0")
        out.add(f"{int(raw):,}.0")
    # 小数：保留原样与去尾零
    if "." in raw:
        try:
            f = float(raw)
            out.add(f"{f:g}")
        except Exception:
            pass
    return out


def find_in(corpus, tok):
    """返回 [(文件名, 行号)]，最多 6 条。"""
    hits = []
    vs = variants(tok)
    for fname, text in corpus.items():
        for i, line in enumerate(text.split("\n"), 1):
            if any(v in line for v in vs):
                hits.append((fname, i))
                if len(hits) >= 6:
                    return hits
    return hits


def main():
    ap = argparse.ArgumentParser(
        description="检查文档中的数字是否都能在 results/ 中找到出处")
    ap.add_argument("doc", help="待检查的 markdown 文件")
    ap.add_argument("--min-len", type=int, default=3,
                    help="只检查位数 >= 此值的数字（默认 3）")
    a = ap.parse_args()

    if not os.path.exists(a.doc):
        print(f"[错误] 找不到文件：{a.doc}")
        return 1

    with open(a.doc, encoding="utf-8", errors="replace") as fh:
        doc = fh.read()

    corpus = load_corpus(RESULTS)
    print(f"已载入 {len(corpus)} 个结果文件作为语料\n")

    tag = re.sub(r"\W+", "_", os.path.basename(a.doc))[:40]
    out_path = os.path.join(RESULTS, f"QC_{tag}.txt")

    # 抽取数字（含千分位与小数）
    pat = re.compile(r"\d[\d,]*(?:\.\d+)?")
    found, missing = [], []
    seen = set()
    for m in pat.finditer(doc):
        tok = m.group(0).rstrip(",.")
        raw = tok.replace(",", "")
        if raw in IGNORE or tok in IGNORE:
            continue
        if len(raw.replace(".", "")) < a.min_len:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        ctx = doc[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")
        hits = find_in(corpus, tok)
        if hits:
            found.append((tok, ctx, hits))
        else:
            missing.append((tok, ctx))

    with open(out_path, "w", encoding="utf-8") as fp:
        def W(s=""):
            fp.write(s + "\n")
            print(s)

        W("=" * 78)
        W(f"数字出处检查：{os.path.basename(a.doc)}")
        W(f"语料：results/ 下 {len(corpus)} 个文件")
        W("=" * 78)
        W("")
        W(f"文档中不同数字总数（位数 >= {a.min_len}）：{len(seen)}")
        W(f"  能找到出处 : {len(found)}")
        W(f"  **找不到出处: {len(missing)}**")
        W("")

        if missing:
            W("=" * 78)
            W("【需人工复核】以下数字在 results/ 中找不到出处")
            W("=" * 78)
            for tok, ctx in missing:
                W(f"  {tok:>14}   上下文: ...{ctx}...")
            W("")
            W("  处理方式：")
            W("    · 若来自脚本的即时输出（未落盘）→ 重跑并落盘，或删除该数字")
            W("    · 若来自外部来源（论文/网页）→ 在图注中标注来源 URL")
            W("    · 若是手算的 → 立即用脚本重算")
            W("")
        else:
            W("  ✅ 全部数字均能在 results/ 中找到出处")
            W("")

        W("=" * 78)
        W("【已找到出处】（前 40 条，含出处定位）")
        W("=" * 78)
        for tok, ctx, hits in found[:40]:
            W(f"  {tok:>14}  ->  " + "; ".join(f"{f}:L{l}" for f, l in hits[:3]))
        if len(found) > 40:
            W(f"  ...（共 {len(found)} 条）")

    print("")
    print(f"报告已保存：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
