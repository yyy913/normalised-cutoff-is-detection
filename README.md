# cutar-detection-ceiling

Analysis code for the manuscript **"A normalised-expression cut-off is a
detection indicator in rare-feature spatial transcriptomics"**
(*Journal of Genetics and Genomics*, Research Communication).

## Summary

This repository contains the analysis code for the manuscript. The manuscript
asks one question: when a rare feature class — cancer-associated unannotated
lncRNAs (cuTARs), detected in a median of 0.08–0.80% of spots — is analysed by
thresholding log-normalised expression, does that threshold select spots that
express the transcript, or only spots where it was detected? Writing
*L* = log1p(10⁴*c*/*T*) for count *c* and feature-subset depth *T*, the mask
*L* > *t* is algebraically identical to `count > 0` whenever
*T*max < 10⁴/(e^*t* − 1). At *t* = 0.5 that bound is 15,414.9 reads, while the
deepest cuTAR-bearing spot in each of the ten released spatial matrices holds
40–380 reads, 41–385× below it. The governing depth is that of the **feature
subset**, not the library: library totals span 1,576-fold, subset totals only
9.1-fold.

The code reproduces the following. Every number the manuscript cites is
recomputed from the distributed count matrices rather than copied from a
previous run; the files under `results/` are what that recomputation is
checked against, and they ship with the package so the chain can be walked
end to end (see §5 for the three exceptions).

- Figure 1 (four panels) and Supplementary Figure S1;
- Supplementary Tables S1–S4, including the ten-matrix detection summary and
  the cross-platform comparison;
- every numerical claim in the main text and Supplementary Methods, each
  printed beside the value given in the manuscript;
- the principal spatial result (bivariate Moran's *I* under a toroidal-shift
  null), re-tested by a second, independent implementation.

Results are deterministic given the inputs apart from the permutation,
stratified and toroidal-shift nulls, which use fixed seeds. All figures are
rendered deterministically from the matrices; no generative model created or
altered any image content.

## Contents

```
.
├── README.md                  this file
├── LICENSE                    MIT
├── requirements.txt
├── data/                      <- YOU supply the matrices here (see data/README_data.md)
├── results/                   outputs; shipped with the intermediates the
│   │                          verification scripts consume
│   └── figures/               Fig1_composite, FigS1
└── audit/                     all scripts, plus config.py
    └── config.py              the single place where paths are defined
```

Scripts must be run from the package root, keeping `audit/` intact: several
import each other, and two read `audit/recheck2.py` as source text.

## 1. Data

Public; see **[`data/README_data.md`](data/README_data.md)** for the download
locations and the exact directory layout expected (~2.4 GB). No new data were
generated and the matrices were used as distributed.

## 2. Install

```
python -m pip install -r requirements.txt
```

Python 3.13.9 with NumPy 2.3.5, SciPy 1.16.3, scikit-learn 1.7.2,
AnnData 0.13.2 and Matplotlib 3.10.6 — the versions reported in the Methods.

## 3. Quick start

```
python audit/make_fig1.py            # Fig. 1
python audit/make_figures.py         # Fig. S1
python audit/verify_R1R3.py          # every number in R1-R3
python audit/verify_R4R6.py          # every number in R4-R6
```

`verify_*` scripts print each recomputed value next to the value printed in
the manuscript and end with a `n/n 项通过` (items passed) summary. Their full
output is written to `results/`.

To keep the code here and the data elsewhere, set `CUTAR_ROOT` to the
directory that contains `data/` and `results/`:

```
set CUTAR_ROOT=D:\cutar           # Windows cmd
$env:CUTAR_ROOT = "D:\cutar"      # PowerShell
export CUTAR_ROOT=/data/cutar     # Linux / macOS
```

## 4. Script reference

Two classes of script: **producers**, which read the matrices and write a
`results/*.json`; and **consumers**, which read those JSON files. Producers
must run before the consumers that depend on them — the "Reads" column below
is the dependency list.

### Figures

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `make_fig1.py` | Fig. 1, four-panel composite (a criterion identity; b *T*crit/*T*max; c mask vs depth; d survival under three nulls) | `RECHECK.json`, `TC_GAP.json`, `SELFCHECK4.json` | `figures/Fig1_composite.{png,pdf}` |
| `audit_fig1.py` | machine-checkable audit of Fig. 1 (34 properties: label collisions, clipping, font floor, colour semantics) | imports `make_fig1` | `AUDIT_FIG1.txt` |
| `audit_legend.py` | legend-geometry audit (`ax.texts` does not cover legend text) | imports `make_fig1` | stdout |
| `make_figures.py` | Fig. S1 (non-discriminating dilution controls). Also holds archived, uncited figure code — see §5 | `RECHECK3.json`, `RESOURCE_survey.json`, `RECHECK.json`, `MORANS_stratified.json`, `MORANS_conditioned.json`, `SELFCHECK4.json` | `figures/FigS1_non_discriminating_controls.{png,pdf}` |
| `what_is_in_fig1.py` | prints the properties of the figure file currently on disk (version forensics) | imports `make_fig1` | stdout |

### Tables

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `make_supp.py` | assembles the supplementary tables (S1 ten matrices, S2 cross-platform, S3 numeric checks, S4 U2 reproducibility) | `GENERALITY_test.json`, `MANUSCRIPT_AUDIT.txt`, matrices | `补充材料_v1.md` |
| `verify_subset_totals.py` | recomputes Table S1's cuTAR-subset-total column and the two spans quoted in R2 | ten matrices | `VERIFY_SUBSET_TOTALS.txt` |
| `verify_supp_tableS2.py` | recomputes every cell of Table S2, case A and case B | ten + four external matrices | `VERIFY_TABLES2.{txt,json}` |
| `u2_spatial_consistency.py` | cross-matrix detection-rate consistency (Table S4) | ten matrices, `SCRNA_probe.json`, `RECHECK.json` | `U2_spatial_consistency.{txt,json}` |

### Analysis

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `morans_conditioned.py` | R4 main result: cuTAR–gene bivariate Moran's *I*, conditioned on detectability | five mixed matrices | `MORANS_conditioned.{txt,json}` |
| `morans_stratified.py` | R4 strengthened conditioning (depth-stratified null) | five mixed matrices | `MORANS_stratified.{txt,json}` |
| `naive_vs_corrected2.py` | naive vs depth-corrected partition comparison | four matrices | `NAIVE_vs_CORRECTED.{txt,json}` |
| `selfcheck_round4.py` | adversarial self-check of the round-4 experiment | five mixed matrices | `SELFCHECK4.{txt,json}` |
| `diag_knn.py` | brute-force kNN vs cKDTree agreement; coordinate-source differences | five mixed matrices | stdout |
| `diag_r4_followup.py` | traces the cause of each item that failed in `verify_R4_independent.py` | `SELFCHECK4.json`, matrices | `DIAG_R4_FOLLOWUP.{txt,json}` |
| `atlas_quadrant_test.py` | tests whether the resource's HH/LL partition is equivalent to detection for rare features | ten matrices | `ATLAS_quadrant_test.{txt,json}` |
| `scrna_probe.py` | the single-cell object as an independent detectability anchor (R5) | `Melanoma_scRNA.h5ad`, matrices | `SCRNA_probe.{txt,json}` |
| `generality_test.py` | applies the algebraic condition to four external matrices, two platforms (R2) | ten + four external matrices | `GENERALITY_test.{txt,json}` |
| `ladder_control.py` | extends the matched control to a dilution ladder | `Homo_sapiens.gene_info.gz`, three matrices | `RECHECK3.{txt,json}` |
| `tie_rows.py` | spots where the 6th and 7th nearest-neighbour distances tie exactly | five mixed matrices | stdout |
| `r4_methods_numbers.py` | numbers used in the Methods paragraph on the limits of the toroidal null | matrices | `R4_METHODS_NUMBERS.{txt,json}` |

### Verification

| Script | Purpose | Reads | Writes |
|---|---|---|---|
| `verify_R1R3.py` | every number and assertion in R1–R3, recomputed from the raw matrices (16 items; 15 recomputed, 1 read — see §5) | ten matrices, `RECHECK.json` | `VERIFY_R1R3.txt` |
| `verify_R4R6.py` | every number in R4–R6 (24 items) | the `results/*.json` listed below, `SPANCLNC_review_history.txt` | `VERIFY_R4R6.txt` |
| `verify_R4_independent.py` | second, independent implementation of Moran's *I* + toroidal shift (18 items) | `SELFCHECK4.json`, `MORANS_conditioned.json`, matrices | `VERIFY_R4_INDEPENDENT.{txt,json}` |
| `verify_uncovered_claims.py` | numerical claims in the manuscript covered by no other script | `RECHECK.json`, `audit/recheck2.py`, `Homo_sapiens.gene_info.gz` | `VERIFY_UNCOVERED.{txt,json}` |
| `manuscript_audit.py` | full recomputation audit | ten matrices, `Melanoma_scRNA.h5ad`, `Homo_sapiens.gene_info.gz` | `MANUSCRIPT_AUDIT.txt` |
| `qc_numbers.py` | traces every number in a document back to `results/*.txt` | a document path (CLI argument) | `QC_<name>.txt` |

### Producers (upstream of the figures and verifiers)

| Script | Purpose | Writes |
|---|---|---|
| `recheck.py` | first-pass adversarial recheck; establishes `T_crit`, the identity and the R3 correlations | `RECHECK.json` |
| `recheck3.py` | redoes `recheck.py` after correcting three defects in it | `RECHECK.json` (same target — run last) |
| `recheck2.py` | the most attackable control of that route; also read as source text by `verify_uncovered_claims.py` | `RECHECK2.json` |
| `resource_survey.py` | per-matrix sparsity and detection census | `RESOURCE_survey.json` |

### Literature search (Supplementary Methods)

| Script | Purpose | Writes |
|---|---|---|
| `lit_scan1.py`, `lit_scan2.py` | Europe PMC search for cut-offs imposed on normalised expression (the search reported so that the negative result can be audited) | `LITSCAN_stage1/2.{txt,json}` |
| `recheck5.py`, `recheck5b.py`, `recheck5c.py` | self-checks of that search, including explicit attempted/succeeded/failed counts and a 70-paper small-sample verification | `SELFCHECK5*.{txt,json}` |

## 5. Provenance and honest notes

1. **`results/` ships the intermediate files**, because several verification
   scripts read them. Everything the manuscript cites is recomputed from the
   raw matrices; the intermediates are what the verification is *checked
   against*, and they are included so the chain can be walked end to end.
2. **Two intermediates have no generating script.** `TC_GAP.json` (panel b of
   Fig. 1) and `SPANCLNC_review_history.txt` (used by `verify_R4R6.py`) were
   produced by ad-hoc commands during the analysis. They are supplied here as
   inputs rather than regenerated. The quantities they contain — *T*crit/*T*max
   per matrix, and the resource's own statement that long-read validation
   recovers 31.7% of Visium-captured uTAR 3′ ends — are each independently
   recomputed elsewhere in the package (`verify_R1R3.py`, the ten-matrix
   loop).
3. **One item in `verify_R1R3.py` is read rather than recomputed.** The R3
   Spearman ρ check reads `RECHECK.json`; the script prints this explicitly,
   and the supplementary table notes it. The other 15 of 16 items are
   recomputed from the raw matrices.
4. **`recheck.py` and `recheck3.py` write the same file.** `recheck3.py` is the
   corrected version; if both are run, `recheck3.py` must be last.
5. **`make_figures.py` also contains archived figure code.** `fig1()`–`fig6()`
   and `fig4b()` build exploratory figures from an earlier route of this
   project. None is cited in the manuscript, and **panel c of
   `fig4b()` (`Fig4_threshold_vs_continuous`) is retracted** and must not be
   reused. The released `__main__` therefore runs `figS1()` only; the archived
   functions are kept so the file is complete but are not executed.
6. **`make_supp.py` emits Chinese table headers**, because it was written
   against the bilingual working copy. The submitted supplementary file is the
   English version of the same tables.
7. **Not included: scripts that operate on the manuscript text itself** (word
   counting, citation and cross-reference checking, bilingual paragraph
   alignment) and the infrastructure of an earlier, abandoned survey route
   (STOmicsDB / SPASCER). They are about the writing process rather than the
   science and cannot run without the manuscript source.

## 6. License and citation

Released under the **MIT License** (see `LICENSE`).

If you use this code, please cite the manuscript and this repository. The
repository DOI is minted on release and will be added here; until then cite
the manuscript DOI once it is assigned.
