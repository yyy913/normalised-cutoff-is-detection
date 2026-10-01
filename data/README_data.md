# Data

The raw count matrices are **not** included in this package. They are public
and were used exactly as distributed; no filtering, re-alignment or
re-normalisation was applied.

## Where to get them

| Item | Source |
|---|---|
| Ten spatial matrices + the single-cell object | SPanC-Lnc download bucket: `https://downloads.gmllab.com/?prefix=SPanC-Lnc/` |
| GSE280315 (two Visium HD sections) | GEO accession **GSE280315** |
| GSE225857 (two Visium sections) | GEO accession **GSE225857** |
| `Homo_sapiens.gene_info.gz` | NCBI Gene: `https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz` |

## Where to put them

Unpack so that the tree looks exactly like this, under `<WORKSPACE>/data/`
(by default `<WORKSPACE>` is the root of this code package):

```
data/
├── SPanC-Lnc-CRC/
│   ├── CM_pacbio.h5ad                (26 MB,  PacBio, cuTAR-only, 1,204 features)
│   └── CP_pacbio.h5ad                (26 MB,  PacBio, cuTAR-only)
├── SPanC-Lnc-pan/
│   ├── HNC_ilong_nano.h5ad           (7 MB,   ONT, cuTAR-only)
│   ├── BCC_nano.h5ad                 (41 MB,  ONT, cuTAR-only)
│   ├── SCC_nano.h5ad                 (40 MB,  ONT, cuTAR-only)
│   ├── Melanoma.h5ad                 (66 MB,  short-read, mixed)
│   ├── BCC.h5ad                      (116 MB, short-read, mixed)
│   ├── KidneyCancer.h5ad             (209 MB, short-read, mixed)
│   ├── HNC.h5ad                      (135 MB, short-read, mixed)
│   ├── SCC.h5ad                      (138 MB, short-read, mixed)
│   └── Melanoma_scRNA.h5ad           (883 MB, single-cell object)
├── GSE280315/h5ad/
│   ├── GSM8594567_P1CRC.h5ad
│   └── GSM8594570_P3NAT.h5ad
├── GSE225857/h5ad/
│   ├── GSM7058756_C1.h5ad
│   └── GSM7058760_L1.h5ad
└── ICI_cohorts/
    └── Homo_sapiens.gene_info.gz
```

Total: about 2.4 GB, of which 883 MB is the single-cell object.

If your data live elsewhere, set `CUTAR_ROOT` to the directory that contains
`data/` and `results/` — see `audit/config.py`.

## Notes

- `CM_pacbio.h5ad` and `CP_pacbio.h5ad` sit under `SPanC-Lnc-CRC/`; the other
  eight matrices sit under `SPanC-Lnc-pan/`. This mirrors the layout of the
  download bucket, not the tissue of origin.
- The five cuTAR-only matrices contain cuTAR features and nothing else; the
  five mixed matrices also contain annotated genes. The distinction matters
  for the argument in the paper, because the depth that governs a cut-off is
  that of the feature subset, not of the library.
- `Melanoma_scRNA.h5ad` is the only single-cell object in the resource and is
  used for the cross-modality comparison (Supplementary Table 4).
