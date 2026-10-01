# -*- coding: utf-8 -*-
"""
lit_scan2.py — 针对性文献扫描 阶段2：全文模式抓取 + 逐篇核实
思路：不做命中计数，而是取一批空间转录组 OA 论文的**全文**，用正则抓"用归一化表达阈值定义
spot/区域"的原句，保留可核实的逐字引用。

同时记录两个关键区分：
  (a) 阈值施加在**高丰度标记基因**上  -> 常见，且未必有问题
  (b) 阈值施加在**稀有/低检出特征**上 -> 本文所指的问题情形
以及论文是否报告了检出率。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, re, ssl, json, time, urllib.request, urllib.parse, html
sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
ROOT = _cfg.WORKSPACE + r""
MAXTEXT = 160        # 最多抓多少篇全文
OUT = []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(ROOT + r"\results\LITSCAN_stage2.txt", "w", encoding="utf-8").write("\n".join(OUT))


def g(u, t=60, tries=3, raw=False):
    last = None
    for i in range(tries):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t,
                                       context=ctx).read().decode("utf-8", "replace")
            return b if raw else json.loads(b)
        except Exception as e:
            last = e; time.sleep(1.5 + i)
    raise last


def epmc_ids(q, cap=400):
    ids, cur = [], 1
    while len(ids) < cap:
        u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=" + urllib.parse.quote(q)
             + f"&format=json&pageSize=100&cursorMark={'*' if cur == 1 else cur}")
        try:
            j = g(u)
        except Exception as e:
            W(f"  [id fetch] {type(e).__name__}: {e}"); break
        res = j.get("resultList", {}).get("result", []) or []
        for it in res:
            if it.get("pmcid") and it.get("isOpenAccess") == "Y":
                ids.append({"pmcid": it["pmcid"], "doi": it.get("doi"),
                            "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title"),
                            "year": it.get("pubYear"), "title": (it.get("title") or "")[:120]})
        nc = j.get("nextCursorMark")
        if not nc or nc == cur or not res:
            break
        cur = nc
        time.sleep(0.2)
    return ids


# 使用模式（要求"归一化/标度化表达"与"阈值/大于"在同一句内共现）
PATS = [
    re.compile(r"[^.]{0,200}(?:log-?normali[sz]ed|normali[sz]ed expression|log-?transformed|"
               r"scaled expression|CP10K|counts per (?:ten thousand|10,?000))[^.]{0,200}"
               r"(?:greater than|above|exceed\w*|>\s*=?\s*[\d.]|threshold\w*|cut-?off)[^.]{0,200}\.", re.I),
    re.compile(r"[^.]{0,200}(?:threshold\w*|cut-?off|greater than|above|>\s*=?\s*[\d.])[^.]{0,200}"
               r"(?:log-?normali[sz]ed|normali[sz]ed expression|CP10K)[^.]{0,200}\.", re.I),
    re.compile(r"[^.]{0,160}spots?[^.]{0,160}(?:positive|expressing|high[- ]expression)[^.]{0,160}"
               r"(?:log-?normali[sz]ed|normali[sz]ed|threshold|above)[^.]{0,160}\.", re.I),
]
# 稀有/低检出特征线索
RARE = re.compile(r"lncRNA|long non-?coding|unannotated|novel transcript|rare transcript|"
                  r"lowly[- ]expressed|low[- ]abundance|uTAR|cuTAR", re.I)
# 是否报告检出率
DETR = re.compile(r"detect(?:ion|ed) (?:rate|in)|percent(?:age)? of spots|fraction of spots", re.I)

QUERIES = [
    '"spatial transcriptomics" AND ("log-normalized" OR "log-normalised" OR "CP10K") AND OPEN_ACCESS:Y',
    '"spatial transcriptomics" AND ("log-normalized" OR "log-normalised") AND (threshold OR "greater than" OR cutoff) AND OPEN_ACCESS:Y',
    '"spatial transcriptomics" AND ("normalize_total" OR "CP10K") AND OPEN_ACCESS:Y',
]

W("=" * 108)
W("阶段2：全文模式抓取")
W("=" * 108)
pool, seen0 = [], set()
for q in QUERIES:
    W(f"检索式: {q}")
    part = epmc_ids(q, cap=150)
    W(f"  -> 去重前 {len(part)} 篇")
    for x in part:
        if x["pmcid"] not in seen0:
            seen0.add(x["pmcid"]); pool.append(x)
uniq = pool[:MAXTEXT]
W(f"合并去重后候选 = {len(pool)} 篇，本次抓取上限 {MAXTEXT} 篇 -> 实际 {len(uniq)} 篇")

HITS = []
for i, it in enumerate(uniq):
    pid = it["pmcid"]
    try:
        xml = g(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML", raw=True)
    except Exception:
        continue
    t = re.sub(r"(?s)<(ref-list|back)\b.*", " ", xml)
    t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t); t = re.sub(r"\s+", " ", t)
    found = []
    for p in PATS:
        for m in p.finditer(t):
            s = re.sub(r"\s+", " ", m.group(0)).strip()
            if 60 < len(s) < 700:
                found.append(s)
    if not found:
        continue
    # 去重、只留最长的 3 条
    found = sorted(set(found), key=len, reverse=True)[:3]
    HITS.append({**it, "sentences": found,
                 "rare_context": bool(RARE.search(t)),
                 "reports_detection": bool(DETR.search(t)),
                 "n_rare_mentions": len(RARE.findall(t))})
    if (i + 1) % 20 == 0:
        W(f"  ...已处理 {i+1}/{len(uniq)}，累计命中 {len(HITS)}")
    time.sleep(0.1)

W(f"\n扫描完成：{len(uniq)} 篇中 **{len(HITS)} 篇**出现'归一化表达 + 阈值'共现句")
withrare = [h for h in HITS if h["n_rare_mentions"] >= 5]
W(f"其中涉及稀有/未注释转录本的（全文提及 lncRNA/unannotated 等 ≥5 次）: **{len(withrare)} 篇**")
withdet = [h for h in HITS if h["reports_detection"]]
W(f"其中报告了检出率/检出 spot 比例的: **{len(withdet)} 篇**")

W("\n" + "=" * 108)
W("逐篇明细（可核实：pmcid + doi + 逐字原句）")
W("=" * 108)
for h in HITS[:60]:
    W(f"\n[{h['pmcid']}] {h['year']} {(h['journal'] or '?')[:34]}")
    W(f"  {h['title']}")
    W(f"  doi={h['doi']}  稀有转录本提及={h['n_rare_mentions']}  报告检出率={h['reports_detection']}")
    for s in h["sentences"]:
        W(f"   » {s[:420]}")

json.dump({"queries": QUERIES, "n_scanned": len(uniq), "n_hits": len(HITS), "hits": HITS},
          open(ROOT + r"\results\LITSCAN_stage2.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\LITSCAN_stage2.txt/.json")
