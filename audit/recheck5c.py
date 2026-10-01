# -*- coding: utf-8 -*-
"""
recheck5c.py — 小样本核实：稀有转录本 × log-normalized × 阈值（检索式 S10，共 70 篇）
对每篇抓全文，判定它是否**真的把归一化表达阈值用作判据**（而非仅仅提及两个词）。
显式打印 尝试/成功/失败，避免上一轮的分母 bug。
限速：每篇间隔 0.8s，失败重试 2 次，避免触发限流。
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, os, re, ssl, json, time, urllib.request, urllib.parse, html
sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
ROOT = _cfg.WORKSPACE + r""
CAP = 45
OUT = []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(ROOT + r"\results\SELFCHECK5c.txt", "w", encoding="utf-8").write("\n".join(OUT))


def g(u, t=50, tries=2):
    last = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t,
                                          context=ctx).read().decode("utf-8", "replace")
        except Exception as e:
            last = e; time.sleep(3 + 3 * i)
    raise last


def ids(q, cap=120):
    out, cur = [], "*"
    while len(out) < cap:
        try:
            j = json.loads(g("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
                             + urllib.parse.quote(q) + f"&format=json&pageSize=100&cursorMark={urllib.parse.quote(cur)}"))
        except Exception as e:
            W(f"  [ids] {type(e).__name__}"); break
        res = j.get("resultList", {}).get("result", []) or []
        for it in res:
            out.append({"pmcid": it.get("pmcid"), "open": it.get("isOpenAccess"),
                        "doi": it.get("doi"), "year": it.get("pubYear"),
                        "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title"),
                        "title": (it.get("title") or "")[:130]})
        nc = j.get("nextCursorMark")
        if not nc or nc == cur or not res:
            break
        cur = nc
    return out


RARE = re.compile(r"lncRNA|long non-?coding|unannotated|novel transcript", re.I)
# 真判据型：明确"以…阈值判定/定义 spot 或基因为阳性/入选"
DEFINE = re.compile(
    r"[^.]{0,240}(?:defined?|selected|classified|considered|called|filtered|retained|included|thresholded)"
    r"[^.]{0,240}(?:normalized expression|log-?normali[sz]ed|scaled expression|CP10K)[^.]{0,240}\.", re.I)
DEFINE2 = re.compile(
    r"[^.]{0,240}(?:normalized expression|log-?normali[sz]ed|scaled expression|CP10K)[^.]{0,240}"
    r"(?:greater than|above|exceed\w*|>\s*=?\s*[\d.]|threshold|cut-?off)[^.]{0,60}"
    r"(?:defined?|selected|classified|considered|positive|retained|included)[^.]{0,160}\.", re.I)

Q = '"spatial transcriptomics" AND (lncRNA OR "long non-coding" OR unannotated) AND ("log-normalized" OR "log-normalised") AND (threshold OR cutoff OR "cut-off")'
W("=" * 108)
W("小样本核实：稀有转录本 × log-normalized × 阈值")
W("=" * 108)
W(f"检索式: {Q}")
allids = ids(Q, cap=120)
cand = [x for x in allids if x["pmcid"] and x["open"] == "Y"]
W(f"候选（OA+PMCID）= {len(cand)} 篇；本次核实上限 {CAP} 篇")

n_att = n_ok = n_fail = 0
rare_hit, define_hit, verified = [], [], []
for i, it in enumerate(cand[:CAP]):
    n_att += 1
    try:
        x = g(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{it['pmcid']}/fullTextXML")
        t = re.sub(r"(?s)<(ref-list|back)\b.*", " ", x)
        t = re.sub(r"<[^>]+>", " ", t); t = html.unescape(t); t = re.sub(r"\s+", " ", t)
        n_ok += 1
    except Exception:
        n_fail += 1
        continue
    if RARE.search(t):
        rare_hit.append({**it, "n_rare": len(RARE.findall(t))})
    sents = []
    for p in (DEFINE, DEFINE2):
        for m in p.finditer(t):
            s = re.sub(r"\s+", " ", m.group(0)).strip()
            if 60 < len(s) < 700:
                sents.append(s)
    if sents:
        rec = {**it, "n_rare": len(RARE.findall(t)),
               "sentences": sorted(set(sents), key=len, reverse=True)[:3]}
        define_hit.append(rec)
        if rec["n_rare"] >= 5:
            verified.append(rec)
    if (i + 1) % 10 == 0:
        W(f"  {i+1}/{min(len(cand),CAP)}  成功={n_ok} 失败={n_fail} 含判据句={len(define_hit)}")
    time.sleep(0.8)

W("\n" + "=" * 108)
W("★ 真实计数")
W("=" * 108)
W(f"  尝试 {n_att} | **成功 {n_ok}** | 失败 {n_fail}")
if n_ok:
    W(f"  成功中提及稀有转录本           : {len(rare_hit)} ({100*len(rare_hit)/n_ok:.0f}%)")
    W(f"  成功中含'归一化表达作判据'句    : {len(define_hit)} ({100*len(define_hit)/n_ok:.0f}%)")
    W(f"  **其中确为稀有转录本主题（≥5）**: {len(verified)}  <- 我们真正关心的问题情形")

W("\n" + "=" * 108)
W("逐篇明细（确为稀有转录本主题 ∩ 含判据句）")
W("=" * 108)
if not verified:
    W("  （无交集）")
for h in verified[:25]:
    W(f"\n[{h['pmcid']}] {h['year']} {(h['journal'] or '?')[:30]}  稀有提及={h['n_rare']}")
    W(f"  {h['title']}   doi={h['doi']}")
    for s in h["sentences"]:
        W(f"   » {s[:420]}")

W("\n" + "=" * 108)
W("含判据句但稀有提及<5（供参考）")
W("=" * 108)
for h in [x for x in define_hit if x["n_rare"] < 5][:12]:
    W(f"\n[{h['pmcid']}] {h['year']} 稀有提及={h['n_rare']}  {h['title'][:78]}")
    W(f"   » {h['sentences'][0][:300]}")

json.dump({"query": Q, "attempted": n_att, "succeeded": n_ok, "failed": n_fail,
           "n_rare_mention": len(rare_hit), "n_define_sentence": len(define_hit),
           "n_verified_rare_define": len(verified),
           "verified": verified, "define_sentences": define_hit},
          open(ROOT + r"\results\SELFCHECK5c.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n[written] results\\SELFCHECK5c.txt/.json")
