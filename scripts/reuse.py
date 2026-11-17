"""Platform model with data-movement cost: OpenMP vs GPU when the matrix is reused R times.

Model (assumptions are deliberately simple and stated in the README):
  OpenMP time for R SpMVs : R * t_omp                      (data already in host memory)
  GPU    time for R SpMVs : T_xfer + R * t_gpu_kernel      (matrix/x copied once, y copied back
                                                           once; later SpMVs reuse device data,
                                                           as in an iterative solver)
  T_xfer = Team 7's `transfer_ms` (H2D + D2H) of the *same* kernel/config.
  The GPU side takes the best (candidate, config) among Team 7's own kernels for that R, so
  a padded ELL layout (more bytes to move) can lose to CSR when R is small.
  MPI is excluded: Team 6 records neither matrix-distribution time nor a single-node baseline
  on the same machine, so a one-shot MPI figure cannot be formed.
  The OpenMP and GPU results were measured on different machines (a Windows laptop and a
  Colab-class NVIDIA GPU host); the break-even point is therefore indicative, not universal.
"""
import numpy as np
import pandas as pd
from common import TABLES, OPENMP, GPU_OWN, gflops


def _tables():
    g = pd.read_csv(TABLES / "unified_config_summary.csv")
    feat = pd.read_csv(TABLES / "matrix_features.csv").set_index("matrix_id")
    omp = g[g.candidate.isin(OPENMP)]
    omp_best = omp.loc[omp.groupby("dataset").kernel_ms_median.idxmin()].set_index("dataset")
    gpu = g[g.candidate.isin(GPU_OWN)].copy()
    return g, feat, omp_best, gpu


def platform_perf(R):
    """Effective GFLOP/s (= 2*nnz*R / total time) for OpenMP and GPU at reuse count R."""
    _, feat, omp_best, gpu = _tables()
    gpu["total_R"] = gpu.transfer_ms_median + R * gpu.kernel_ms_median
    gb = gpu.loc[gpu.groupby("dataset").total_R.idxmin()].set_index("dataset")
    P = pd.DataFrame({"OpenMP": gflops(feat.nnz.reindex(omp_best.index), omp_best.kernel_ms_median),
                      "GPU": 2 * feat.nnz.reindex(gb.index) * R / (gb.total_R * 1e-3) / 1e9})
    P.index.name = "matrix_id"
    return P.loc[feat.index]


def breakeven_table():
    """Per matrix: best OpenMP time, best GPU kernel time + transfer, and N* = reuse count at
    which the GPU overtakes OpenMP (N* = T_xfer / (t_omp - t_gpu))."""
    _, feat, omp_best, gpu = _tables()
    gb = gpu.loc[gpu.groupby("dataset").kernel_ms_median.idxmin()].set_index("dataset")
    df = pd.DataFrame({"nnz": feat.nnz, "omp_candidate": omp_best.candidate,
                       "omp_ms": omp_best.kernel_ms_median, "gpu_candidate": gb.candidate,
                       "gpu_kernel_ms": gb.kernel_ms_median, "gpu_transfer_ms": gb.transfer_ms_median})
    df["kernel_speedup_gpu_over_omp"] = df.omp_ms / df.gpu_kernel_ms
    df["oneshot_speedup_gpu_over_omp"] = df.omp_ms / (df.gpu_kernel_ms + df.gpu_transfer_ms)
    df["transfer_over_omp_time"] = df.gpu_transfer_ms / df.omp_ms
    df["break_even_N"] = df.gpu_transfer_ms / (df.omp_ms - df.gpu_kernel_ms)
    df.index.name = "matrix_id"
    return df.sort_values("nnz")
