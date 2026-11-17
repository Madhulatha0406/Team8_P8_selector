#!/usr/bin/env python3
"""Platform choice (OpenMP vs GPU) once PCIe transfer is counted, as a function of reuse count R.
See reuse.py for the model and its assumptions. Outputs: platform_reuse_breakeven.csv,
platform_reuse_winners.csv, plots/platform_reuse.png"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr
from common import TABLES, PLOTS, PAL
from reuse import platform_perf, breakeven_table

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 160, "axes.titleweight": "bold"})
Rs = [1, 2, 5, 10, 20, 50, 100]
be = breakeven_table(); be.round(3).to_csv(TABLES / "platform_reuse_breakeven.csv")
win = pd.DataFrame({R: platform_perf(R).idxmax(axis=1) for R in Rs})
win.to_csv(TABLES / "platform_reuse_winners.csv")
rho, p = spearmanr(be.nnz, be.break_even_N)
print(be[["nnz", "omp_ms", "gpu_kernel_ms", "gpu_transfer_ms", "kernel_speedup_gpu_over_omp",
          "oneshot_speedup_gpu_over_omp", "break_even_N"]].round(2).to_string())
print(f"\nSpearman(nnz, break-even N) = {rho:+.2f} (p={p:.3f}); N* median {be.break_even_N.median():.1f}, "
      f"range {be.break_even_N.min():.1f}-{be.break_even_N.max():.1f}")
print("GPU wins by R:", {R: int((win[R] == 'GPU').sum()) for R in Rs})

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.1, 1]})
a.barh(be.index, be.break_even_N, color=PAL["GPU"])
a.set_xscale("log"); a.set_xlabel("break-even reuse count N*  (GPU wins if the matrix is used more than N* times)")
for R in (1, 10, 100): a.axvline(R, color="grey", ls=":", lw=1)
for i, v in enumerate(be.break_even_N): a.text(v * 1.05, i, f"{v:.1f}", va="center", fontsize=7.5)
a.set_title("GPU break-even vs OpenMP (rows sorted by nnz, smallest on top)"); a.invert_yaxis(); a.grid(axis="x", alpha=.25)
order = be.index
b.imshow((win.loc[order] == "GPU").astype(int).values, cmap=ListedColormap([PAL["OpenMP"], PAL["GPU"]]), aspect="auto")
for i in range(len(order)):
    for j in range(len(Rs)): b.text(j, i, win.loc[order[i], Rs[j]].replace("OpenMP", "CPU"), ha="center", va="center", color="white", fontsize=7.5)
b.set_xticks(range(len(Rs))); b.set_xticklabels(Rs); b.set_yticks(range(len(order))); b.set_yticklabels(order, fontsize=7.5); b.set_xlabel("reuse count R (SpMVs on the same matrix)")
b.set_title("Winner by R (transfer included)")
fig.suptitle("OpenMP vs GPU with PCIe transfer counted. Different machines: indicative only. MPI excluded (setup cost not measured).", fontsize=9.5)
fig.tight_layout(); fig.savefig(PLOTS / "platform_reuse.png", bbox_inches="tight")
