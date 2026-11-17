"""Shared constants and helpers for the Team 8 (P8) pipeline."""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "inputs"
TABLES = ROOT / "results" / "tables"
PLOTS = ROOT / "results" / "plots"
for d in (TABLES, PLOTS):
    d.mkdir(parents=True, exist_ok=True)

# Candidate = (platform, format / partition scheme / kernel). Names are used everywhere.
OPENMP = ["OMP-COO", "OMP-CSR", "OMP-ELL"]
MPI = ["MPI-1D-rows", "MPI-1D-nnz", "MPI-2D"]
GPU_OWN = ["GPU-scalar-CSR", "GPU-vector-CSR", "GPU-ELL", "GPU-HYB"]
GPU_VENDOR = ["GPU-cuSPARSE"]
GPU = GPU_OWN + GPU_VENDOR
ALL_OWN = OPENMP + MPI + GPU_OWN
ALL = OPENMP + MPI + GPU
PLATFORM_OF = {**{c: "OpenMP" for c in OPENMP}, **{c: "MPI" for c in MPI},
               **{c: "GPU" for c in GPU}}

BASE_FEATURES = ["rows", "nnz", "mean_nnz_row", "std_nnz_row", "max_nnz_row", "bandwidth"]
DERIVED_FEATURES = ["irregularity", "cv_nnz_row", "rel_bandwidth", "density"]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES
FEATURE_LABELS = {
    "rows": "rows", "nnz": "nnz", "mean_nnz_row": "mean nnz / row",
    "std_nnz_row": "std of nnz / row", "max_nnz_row": "max nnz / row",
    "bandwidth": "bandwidth (max |i-j|)", "irregularity": "irregularity (max / mean)",
    "cv_nnz_row": "CV of nnz / row (std / mean)", "rel_bandwidth": "bandwidth / rows",
    "density": "density (nnz / rows^2)",
}

# Okabe-Ito colour-blind-safe palette
PAL = {"OpenMP": "#0072B2", "MPI": "#E69F00", "GPU": "#009E73",
       "OMP-COO": "#56B4E9", "OMP-CSR": "#0072B2", "OMP-ELL": "#004B73",
       "MPI-1D-rows": "#F0C040", "MPI-1D-nnz": "#E69F00", "MPI-2D": "#B36B00",
       "GPU-scalar-CSR": "#A8D5C2", "GPU-vector-CSR": "#009E73", "GPU-ELL": "#CC79A7",
       "GPU-HYB": "#D55E00", "GPU-cuSPARSE": "#555555"}


def gflops(nnz, ms):
    """GFLOP/s using the project convention 2*nnz flops."""
    return 2.0 * np.asarray(nnz, dtype=float) / (np.asarray(ms, dtype=float) * 1e-3) / 1e9
