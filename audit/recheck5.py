# -*- coding: utf-8 -*-
"""
recheck5.py — 两处自查 + 一次补做扫描

怀疑点A【事实错误】: 我近几轮反复写"已发表的 0.5 判据""广泛使用的判据"。
   但 0.5 这个切分来自**我们自己的未发表草稿**（论文①②），
   图谱（Nature Methods 2026）用的是连续型 bivariate Moran's I，从未用过 0.5 阈值。
   把未发表草稿的判据称作"已发表/广泛使用"是硬性事实错误。

怀疑点B【效力不足】: 我据 160 篇里"0 篇涉及稀有转录本"得出"该做法未用于稀有特征"。
   但若那 160 篇里本来就只有极少数涉及 lncRNA，则 0/N 的 N 极小，结论无效。
   正确做法：**直接针对"稀有转录本 + 空间转录组"这一子文献做扫描**——
   这才是真正相关的问题，而我原来的检索式没有瞄准它。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, re, ssl, json, time, urllib.request, urllib.parse, html
sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
ROOT = _cfg.WORKSPACE + r""
MAXTEXT = 130
OUT = []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(ROOT + r"\results\SELFCHECK5.txt", "w", encoding="utf-8").write("\n".join(OUT))


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
            ids.append({"pmcid": it.get("pmcid"), "doi": it.get("doi"), "open": it.get("isOpenAccess"),
                        "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title"),
                        "year": it.get("pubYear"), "title": (it.get("title") or "")[:130]})
        nc = j.get("nextCursorMark")
        if not nc or nc == cur or not res:
            break
        cur = nc; time.sleep(0.2)
    return ids


def fulltext(pid):
    x = g(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pid}/fullTextXML")
    t = re.sub(r"(?s)<(ref-list|back)\b.*", " ", x)
    t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t)
    return re.sub(r"\s+", " ", t)


RARE = re.compile(r"lncRNA|long non-?coding|unannotated|novel transcript|uTAR|cuTAR", re.I)
THRESH = re.compile(
    r"[^.]{0,220}(?:log-?normali[sz]ed|normali[sz]ed expression|scaled expression|CP10K)[^.]{0,220}"
    r"(?:greater than|above|exceed\w*|>\s*=?\s*[\d.]|threshold|cut-?off)[^.]{0,220}\.", re.I)
THRESH2 = re.compile(
    r"[^.]{0,220}(?:threshold\w*|cut-?off|greater than|above|>\s*=?\s*[\d.])[^.]{0,220}"
    r"(?:log-?normali[sz]ed|normali[sz]ed expression|CP10K)[^.]{0,220}\.", re.I)
DETR = re.compile(r"detect(?:ion|ed) (?:rate|in [\d.]+%|in fewer)|percent(?:age)? of spots|"
                  r"fraction of spots|sparsity|sparse|low depth|shallow|limited (?:depth|sensitivity)", re.I)
DEPTHLIM = re.compile(r"[^.]{0,200}(?:sparsit|sparse|low depth|shallow|depth limit|"
                      r"limited sensitivity|undetect\w*|below detection)[^.]{0,200}\.", re.I)

W("=" * 108)
W("第四轮之后的自查（第5轮）")
W("=" * 108)

W("\n【怀疑点A】'已发表的 0.5 判据' —— 事实核对")
W("  0.5 切分的出处：论文①②（**未发表草稿**，`论文初稿_中文_v16.md` / `编写规范`等）")
W("  图谱（Prakrithi et al., Nat Methods 2026）的方法：连续型 bivariate Moran's I（已核，见 SpotSweeper核查报告 与 Figure3 脚本）")
W("  => 0.5 阈值**不是已发表判据、也不是广泛使用的做法**。我在近几轮把它写成'已发表/广泛使用'是")
W("     **事实错误**，必须在正文与摘要中一律改为：'a cut-off used in our own earlier analysis of this resource'")
W("     或更中性：'a cut-off of 0.5 applied to log-normalised expression'（不声称其来源与普遍性）")

Q = ('("spatial transcriptomics") AND (lncRNA OR "long non-coding" OR unannotated OR "novel transcript") '
     'AND OPEN_ACCESS:Y')
W("\n【怀疑点B】补做扫描 —— 直接瞄准稀有转录本子文献")
W(f"  检索式: {Q}")
ids = epmc_ids(Q, cap=MAXTEXT * 2)
oa = [x for x in ids if x["pmcid"] and x["open"] == "Y"]
seen, uniq = set(), []
for x in oa:
    if x["pmcid"] in seen:
        continue
    seen.add(x["pmcid"]); uniq.append(x)
W(f"  候选（OA + 有 PMCID）= {len(uniq)} 篇；本次抓取上限 {MAXTEXT} 篇")

HITS, RA, det_cnt, depth_cnt = [], [], 0, 0
for i, it in enumerate(uniq[:MAXTEXT]):
    try:
        t = fulltext(it["pmcid"])
    except Exception:
        continue
    nrare = len(RARE.findall(t))
    if nrare < 5:
        continue
    RA.append({**it, "n_rare": nrare})
    sents = []
    for p in (THRESH, THRESH2):
        for m in p.finditer(t):
            s = re.sub(r"\s+", " ", m.group(0)).strip()
            if 60 < len(s) < 700:
                sents.append(s)
    if DETR.search(t):
        det_cnt += 1
    if DEPTHLIM.search(t):
        depth_cnt += 1
    if sents:
        HITS.append({**it, "n_rare": nrare, "sentences": sorted(set(sents), key=len, reverse=True)[:3]})
    if (i + 1) % 25 == 0:
        W(f"    ...{i+1}/{min(len(uniq),MAXTEXT)}  已确认稀有转录本文献 {len(RA)} 篇，其中含阈值句 {len(HITS)} 篇")

W(f"\n  === 结果 ===")
W(f"  实际扫描全文            : {min(len(uniq),MAXTEXT)} 篇")
W(f"  **确认为稀有转录本主题**（lncRNA/unannotated ≥5 次）: {len(RA)} 篇  <- 这才是怀疑点B 的分母")
W(f"  其中出现'归一化表达+阈值'同句共现: **{len(HITS)} 篇**")
W(f"  其中提到检测率/稀疏/深度限制的   : {depth_cnt} 篇")
W(f"\n  对比：上一轮在'泛空间转录组'文献里得到 0/160 ——")
W(f"        但那一池子里稀有转录本主题只有少数几篇，**0/160 的效力极低**；")
W(f"        本轮直接在该子文献上做，分母才是有效的。")

W("\n" + "=" * 108)
W("逐篇明细（稀有转录本主题 ∩ 含阈值句）")
W("=" * 108)
for h in HITS[:40]:
    W(f"\n[{h['pmcid']}] {h['year']} {(h['journal'] or '?')[:34]}  稀有提及={h['n_rare']}")
    W(f"  {h['title']}")
    W(f"  doi={h['doi']}")
    for s in h["sentences"]:
        W(f"   » {s[:430]}")

json.dump({"suspectA": "0.5 判据来自未发表草稿，非已发表/广泛使用",
           "query_B": Q, "n_scanned": min(len(uniq), MAXTEXT), "n_rare_topic": len(RA),
           "n_with_threshold_sentence": len(HITS), "n_depth_aware": depth_cnt,
           "rare_topic": RA, "hits": HITS},
          open(ROOT + r"\results\SELFCHECK5.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[written] results\\SELFCHECK5.txt/.json")
