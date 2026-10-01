# -*- coding: utf-8 -*-
"""
lit_scan1.py — 针对性文献扫描 阶段1：通道测试 + 短语命中统计
目的：确认"用归一化表达阈值定义阳性 spot / 区域 / niche"这一做法在已发表文献中的普遍程度。

通道：
  Europe PMC REST（支持 OA 全文检索）
  PMC eutils（esearch + efetch）
"""
import config as _cfg  # released-code shim: all paths resolve via audit/config.py
import sys, ssl, json, time, urllib.request, urllib.parse
sys.stdout.reconfigure(encoding="utf-8")
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
ROOT = _cfg.WORKSPACE + r""
OUT = []


def W(s=""):
    print(s, flush=True); OUT.append(s)
    open(ROOT + r"\results\LITSCAN_stage1.txt", "w", encoding="utf-8").write("\n".join(OUT))


def g(u, t=60, tries=4, want_json=True):
    last = None
    for i in range(tries):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t, context=ctx)
            b = r.read().decode("utf-8", "replace")
            return json.loads(b) if want_json else b
        except Exception as e:
            last = e; time.sleep(2 + 2 * i)
    raise last


def epmc(q, n=25):
    u = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?query="
         + urllib.parse.quote(q) + f"&format=json&resultType=core&pageSize={n}")
    return g(u)


def pmc_count(q):
    u = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pmc&retmode=json&term="
         + urllib.parse.quote(q))
    return int(g(u)["esearchresult"]["count"])


W("=" * 108)
W("通道测试")
W("=" * 108)
ok_epmc = ok_pmc = False
try:
    r = epmc('"log-normalized expression"', 1)
    W(f"  Europe PMC  : OK  (hitCount={r.get('hitCount')})")
    ok_epmc = True
except Exception as e:
    W(f"  Europe PMC  : FAIL {type(e).__name__}: {e}")
try:
    c = pmc_count('"log-normalized expression"')
    W(f"  PMC eutils  : OK  (count={c})")
    ok_pmc = True
except Exception as e:
    W(f"  PMC eutils  : FAIL {type(e).__name__}: {e}")

QUERIES = [
    ('A1', '"log-normalized expression" AND (threshold OR "greater than" OR cutoff OR "cut-off")'),
    ('A2', '"log-normalized" AND spot AND (positive OR "greater than") AND "spatial transcriptomics"'),
    ('A3', '"normalized expression" AND "spatial transcriptomics" AND (threshold OR cutoff)'),
    ('A4', '"log-normalised" AND (threshold OR "greater than") AND spatial'),
    ('A5', '"log1p" AND "CP10K"'),
    ('A6', '"counts per 10,000" AND spatial AND threshold'),
    ('A7', '"co-expression" AND "microdomain" AND spatial'),
    ('A8', '"spatial domain" AND "log-normalized" AND threshold'),
    ('A9', '("unannotated" OR lncRNA) AND "spatial transcriptomics" AND threshold AND (detect OR detection)'),
    ('A10', '"above a threshold" AND "normalized expression" AND spots'),
    ('B1', '"detection rate" AND "spatial transcriptomics" AND threshold AND artifact'),
    ('B2', '"library size" AND normalization AND "spatial domain" AND (confound OR artifact)'),
    ('B3', '"zero-inflated" AND "spatial transcriptomics" AND (threshold OR cutoff)'),
]

W("\n" + "=" * 108)
W("短语命中统计")
W("=" * 108)
RES = {}
for tag, q in QUERIES:
    row = {"tag": tag, "query": q}
    if ok_epmc:
        try:
            r = epmc(q, 20)
            row["epmc_hits"] = r.get("hitCount")
            row["epmc_top"] = [{
                "title": (it.get("title") or "")[:150],
                "journal": ((it.get("journalInfo") or {}).get("journal") or {}).get("title"),
                "year": it.get("pubYear"), "doi": it.get("doi"), "pmcid": it.get("pmcid"),
                "authors": it.get("authorString", "")[:80],
            } for it in (r.get("resultList", {}).get("result", []) or [])[:8]]
        except Exception as e:
            row["epmc_error"] = f"{type(e).__name__}: {e}"
    if ok_pmc:
        try:
            row["pmc_count"] = pmc_count(q)
        except Exception as e:
            row["pmc_error"] = str(e)[:80]
    RES[tag] = row
    W(f"\n[{tag}] {q}")
    W(f"    EuropePMC hitCount = {row.get('epmc_hits')}   PMC esearch count = {row.get('pmc_count')}")
    for t in row.get("epmc_top", [])[:5]:
        W(f"      - {t['year']} {(t['journal'] or '?')[:26]:<26} {t['title'][:96]}")
        W(f"        doi={t['doi']}  pmcid={t['pmcid']}")

json.dump(RES, open(ROOT + r"\results\LITSCAN_stage1.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("\n[written] results\\LITSCAN_stage1.txt/.json")
