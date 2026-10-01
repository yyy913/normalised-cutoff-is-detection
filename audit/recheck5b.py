# -*- coding: utf-8 -*-
"""
recheck5b.py — 修正版扫描：显式统计"尝试数 / 成功数 / 失败数"

上一轮（lit_scan2 与 recheck5）的分母都是**尝试次数**，不是**成功抓取的篇数**：
    进度打印写在 try/except 之外 -> 无论抓取成功与否都会计数
    bare `except: continue` -> 全部失败也静默通过
因此 lit_scan2 的"160 篇中 15 篇命中（9.4%）"与 recheck5 的"0/130"**分母都不可信**，
两个比率都不能用。本脚本重做，并把三类计数全部打印出来。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, re, ssl, json, time, urllib.request, urllib.parse, html
sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
ROOT = _cfg.WORKSPACE + r""
CAP = 130
OUT = []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(ROOT + r"\results\SELFCHECK5b.txt", "w", encoding="utf-8").write("\n".join(OUT))


def g(u, t=60, tries=4, raw=False):
    last = None
    for i in range(tries):
        try:
            b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t,
                                       context=ctx).read().decode("utf-8", "replace")
            return b if raw else json.loads(b)
        except Exception as e:
            last = e
            time.sleep(1.0 + 2.0 * i)
    raise last


def epmc_ids(q, cap=300):
    ids, cur = [], "*"
    while len(ids) < cap:
        try:
            j = g("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
                  + urllib.parse.quote(q) + f"&format=json&pageSize=100&cursorMark={urllib.parse.quote(cur)}")
        except Exception as e:
            W(f"  [ids] {type(e).__name__}: {e}"); break
        res = j.get("resultList", {}).get("result", []) or []
        for it in res:
            if it.get("pmcid") and it.get("isOpenAccess") == "Y":
                ids.append({"pmcid": it["pmcid"], "doi": it.get("doi"),
                            "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title"),
                            "year": it.get("pubYear"), "title": (it.get("title") or "")[:130]})
        nc = j.get("nextCursorMark")
        if not nc or nc == cur or not res:
            break
        cur = nc; time.sleep(0.2)
    return ids


RARE = re.compile(r"lncRNA|long non-?coding|unannotated|novel transcript|uTAR|cuTAR", re.I)
TH = [re.compile(r"[^.]{0,220}(?:log-?normali[sz]ed|normali[sz]ed expression|scaled expression|CP10K)"
                 r"[^.]{0,220}(?:greater than|above|exceed\w*|>\s*=?\s*[\d.]|threshold|cut-?off)[^.]{0,220}\.", re.I),
      re.compile(r"[^.]{0,220}(?:threshold\w*|cut-?off|greater than|above|>\s*=?\s*[\d.])"
                 r"[^.]{0,220}(?:log-?normali[sz]ed|normali[sz]ed expression|CP10K)[^.]{0,220}\.", re.I)]
DEPTH = re.compile(r"sparsit|sparse|low depth|shallow|depth limit|limited sensitivity|below detection|"
                   r"detection rate|undetect", re.I)

Q = ('("spatial transcriptomics") AND (lncRNA OR "long non-coding" OR unannotated OR "novel transcript") '
     'AND OPEN_ACCESS:Y')
W("=" * 108)
W("修正版扫描：显式计数")
W("=" * 108)
W(f"检索式: {Q}")
ids = epmc_ids(Q, cap=400)
W(f"OA + 有 PMCID 候选 = {len(ids)} 篇（hitCount 级池）")

cand = ids[:CAP]
n_att = n_ok = n_fail = 0
rare_topic, with_th, depth_aware = [], [], 0
for i, it in enumerate(cand):
    n_att += 1
    try:
        x = g(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{it['pmcid']}/fullTextXML")
        t = re.sub(r"(?s)<(ref-list|back)\b.*", " ", x)
        t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t); t = re.sub(r"\s+", " ", t)
        n_ok += 1
    except Exception:
        n_fail += 1
        continue
    nrare = len(RARE.findall(t))
    if nrare >= 5:
        rare_topic.append({**it, "n_rare": nrare})
    if DEPTH.search(t):
        depth_aware += 1
    sents = []
    for p in TH:
        for m in p.finditer(t):
            s = re.sub(r"\s+", " ", m.group(0)).strip()
            if 60 < len(s) < 700:
                sents.append(s)
    if sents:
        with_th.append({**it, "n_rare": nrare,
                        "sentences": sorted(set(sents), key=len, reverse=True)[:3]})
    if (i + 1) % 25 == 0:
        W(f"  进度 {i+1}/{len(cand)}  成功={n_ok} 失败={n_fail} 稀有主题={len(rare_topic)} 含阈值句={len(with_th)}")
    time.sleep(0.35)

W("\n" + "=" * 108)
W("★ 真实计数（本脚本的核心产出）")
W("=" * 108)
W(f"  尝试抓取                : {n_att}")
W(f"  **成功抓到全文**        : {n_ok}      <- 这才是有效分母")
W(f"  失败                    : {n_fail}")
W(f"  成功中确认为稀有转录本主题（≥5 次提及）: {len(rare_topic)}")
W(f"  成功中出现'归一化表达+阈值'同句共现    : {len(with_th)}")
W(f"  成功中提到检测率/稀疏/深度限制        : {depth_aware}")

if n_ok:
    W(f"\n  有效比率（以成功数为分母）：")
    W(f"    稀有转录本主题占比       : {100*len(rare_topic)/n_ok:.1f}%  ({len(rare_topic)}/{n_ok})")
    W(f"    含'归一化表达+阈值'共现句 : {100*len(with_th)/n_ok:.1f}%  ({len(with_th)}/{n_ok})")
    W(f"    其中同时为稀有转录本主题  : {len([h for h in with_th if h['n_rare']>=5])}"
      f"  <- 本文真正关心的问题情形")

W("\n" + "=" * 108)
W("逐篇明细：稀有转录本主题 ∩ 含阈值句")
W("=" * 108)
inter = [h for h in with_th if h["n_rare"] >= 5]
if not inter:
    W("\n  （无交集）")
for h in inter[:30]:
    W(f"\n[{h['pmcid']}] {h['year']} {(h['journal'] or '?')[:32]}  稀有提及={h['n_rare']}")
    W(f"  {h['title']}")
    W(f"  doi={h['doi']}")
    for s in h["sentences"]:
        W(f"   » {s[:430]}")

W("\n" + "=" * 108)
W("对照：所有含阈值句的论文（不限主题）—— 用于判断该做法施加在什么特征上")
W("=" * 108)
for h in with_th[:20]:
    W(f"\n[{h['pmcid']}] {h['year']} 稀有提及={h['n_rare']}  {h['title'][:80]}")
    for s in h["sentences"][:1]:
        W(f"   » {s[:300]}")

json.dump({"query": Q, "attempted": n_att, "succeeded": n_ok, "failed": n_fail,
           "n_rare_topic": len(rare_topic), "n_with_threshold": len(with_th),
           "n_intersection": len(inter), "n_depth_aware": depth_aware,
           "rare_topic": rare_topic, "with_threshold": with_th, "intersection": inter},
          open(ROOT + r"\results\SELFCHECK5b.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[written] results\\SELFCHECK5b.txt/.json")
