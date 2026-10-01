# -*- coding: utf-8 -*-
r"""config.py — the single place where paths are defined.

Every analysis script in this folder imports this module as::

    import config as _cfg
    ROOT = _cfg.WORKSPACE

and then addresses data and outputs relative to it:

    <WORKSPACE>/data/...      raw count matrices   (see ../README.md)
    <WORKSPACE>/results/...   outputs and intermediates (shipped with this package)

By default WORKSPACE is the root of this code package, i.e. the directory
that contains ``audit/``, ``data/`` and ``results/``.  To keep the code in
one place and the data somewhere else, set the environment variable
``CUTAR_ROOT`` before running anything:

    Windows (cmd)      set CUTAR_ROOT=D:\cutar
    Windows (pwsh)     $env:CUTAR_ROOT = "D:\cutar"
    Linux / macOS      export CUTAR_ROOT=/data/cutar

A writable ``NUMBA_CACHE_DIR`` is set here because some spatial-neighbour
computations are JIT-compiled and fail if the default cache location is
not writable.
"""
from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))   # <package>/audit
_DEFAULT = os.path.dirname(_HERE)                    # <package>

WORKSPACE = os.path.abspath(os.environ.get("CUTAR_ROOT") or _DEFAULT)

DATA = os.path.join(WORKSPACE, "data")
RESULTS = os.path.join(WORKSPACE, "results")
FIGURES = os.path.join(RESULTS, "figures")
CACHE = os.path.join(WORKSPACE, ".cache")

for _d in (RESULTS, FIGURES, CACHE, os.path.join(CACHE, "numba")):
    os.makedirs(_d, exist_ok=True)

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(CACHE, "numba"))

# --------------------------------------------------------------- 十个矩阵
# (tag, path relative to WORKSPACE, class)  — kept here so that the layout
# of the distributed data is documented in exactly one place.
MATRICES = [
    ("CM_pacbio",      r"data\SPanC-Lnc-CRC\CM_pacbio.h5ad",      "cuTAR-only"),
    ("CP_pacbio",      r"data\SPanC-Lnc-CRC\CP_pacbio.h5ad",      "cuTAR-only"),
    ("HNC_ilong_nano", r"data\SPanC-Lnc-pan\HNC_ilong_nano.h5ad", "cuTAR-only"),
    ("BCC_nano",       r"data\SPanC-Lnc-pan\BCC_nano.h5ad",       "cuTAR-only"),
    ("SCC_nano",       r"data\SPanC-Lnc-pan\SCC_nano.h5ad",       "cuTAR-only"),
    ("Melanoma",       r"data\SPanC-Lnc-pan\Melanoma.h5ad",       "short-read mixed"),
    ("BCC",            r"data\SPanC-Lnc-pan\BCC.h5ad",            "short-read mixed"),
    ("KidneyCancer",   r"data\SPanC-Lnc-pan\KidneyCancer.h5ad",   "short-read mixed"),
    ("HNC",            r"data\SPanC-Lnc-pan\HNC.h5ad",            "short-read mixed"),
    ("SCC",            r"data\SPanC-Lnc-pan\SCC.h5ad",            "short-read mixed"),
]

SCRNA = r"data\SPanC-Lnc-pan\Melanoma_scRNA.h5ad"
GENE_INFO = r"data\ICI_cohorts\Homo_sapiens.gene_info.gz"

# ------------------------------------------------------- 论文中的关键常数
T_CUT = 0.5                       # representative log-normalised cut-off
T_CRIT = 1e4 / (2.718281828459045 ** T_CUT - 1.0)   # = 15,414.9 reads


def rel(p):
    """Turn a WORKSPACE-relative path into an absolute one."""
    return os.path.join(WORKSPACE, p)
