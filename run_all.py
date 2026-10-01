# -*- coding: utf-8 -*-
r"""run_all.py — run the released code in dependency order.

Usage
-----
    python run_all.py                 # everything (long: reads all matrices)
    python run_all.py --list          # just print the order
    python run_all.py --stage verify  # only one stage
    python run_all.py --stage verify --only verify_R1R3.py

Stages
------
    producers   write results/*.json            (must run first)
    figures     Fig. 1 and Fig. S1
    tables      the supplementary tables
    verify      the main-text verification scripts
    audit       figure audits and secondary diagnostics

Every script is run as a separate process with the environment variable
``CUTAR_ROOT`` passed through, so ``audit/config.py`` resolves paths the same
way it does when a script is run by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.join(HERE, "audit")

STAGES = [
    ("producers", [
        "recheck.py",                # -> RECHECK.json
        "resource_survey.py",        # -> RESOURCE_survey.json
        "recheck3.py",               # -> RECHECK.json  (corrected; keep last)
        "ladder_control.py",         # -> RECHECK3.json
        "selfcheck_round4.py",       # -> SELFCHECK4.json
        "morans_conditioned.py",     # -> MORANS_conditioned.json
        "morans_stratified.py",      # -> MORANS_stratified.json
        "generality_test.py",        # -> GENERALITY_test.json
        "scrna_probe.py",            # -> SCRNA_probe.json
        "manuscript_audit.py",       # -> MANUSCRIPT_AUDIT.txt
    ]),
    ("figures", [
        "make_fig1.py",              # Fig. 1
        "make_figures.py",           # Fig. S1
    ]),
    ("tables", [
        "make_supp.py",
        "verify_subset_totals.py",
        "verify_supp_tableS2.py",
        "u2_spatial_consistency.py",
    ]),
    ("verify", [
        "verify_R1R3.py",
        "verify_R4R6.py",
        "verify_R4_independent.py",
        "verify_uncovered_claims.py",
    ]),
    ("audit", [
        "audit_fig1.py",
        "audit_legend.py",
        "what_is_in_fig1.py",
        "r4_methods_numbers.py",
        "atlas_quadrant_test.py",
        "naive_vs_corrected2.py",
        "tie_rows.py",
        "diag_knn.py",
        "diag_r4_followup.py",
    ]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=[s for s, _ in STAGES],
                    help="run only this stage")
    ap.add_argument("--only", help="run only this single script (needs --stage)")
    ap.add_argument("--list", action="store_true", help="print the order and exit")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="abort at the first non-zero exit")
    a = ap.parse_args()

    plan = [(s, ns) for s, ns in STAGES if (a.stage is None or s == a.stage)]
    if a.only:
        plan = [(s, [n for n in ns if n == a.only]) for s, ns in plan]
        plan = [(s, ns) for s, ns in plan if ns]

    if a.list:
        for s, ns in plan:
            print(f"[{s}]")
            for n in ns:
                print("   ", n)
        return 0

    env = dict(os.environ)
    total, failed = 0, []
    for stage, names in plan:
        print(f"\n{'=' * 72}\n[{stage}]\n{'=' * 72}", flush=True)
        for n in names:
            path = os.path.join(AUDIT, n)
            if not os.path.exists(path):
                print(f"  MISSING  {n}")
                failed.append(n)
                continue
            t0 = time.time()
            rc = subprocess.call([sys.executable, path], env=env, cwd=HERE)
            dt = time.time() - t0
            total += 1
            print(f"  {'ok  ' if rc == 0 else 'FAIL'}  {n:<28} {dt:7.1f} s  (exit {rc})",
                  flush=True)
            if rc != 0:
                failed.append(n)
                if a.stop_on_error:
                    print("\nstopping at first failure (--stop-on-error)")
                    return 1

    print(f"\n{total - len(failed)}/{total} scripts exited zero")
    for n in failed:
        print("  failed:", n)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
