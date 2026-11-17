#!/usr/bin/env python3
"""Integrate the *secondary* outputs of Teams 5-7 that 02_consolidate.py does not use:
  P6 communication volume  -> does the lowest-communication partition win?
  P5 STREAM roofline       -> effective bandwidth and % of STREAM for each OpenMP format
  P5 first-touch / pinning -> size of the NUMA / pinning effects
  P5 static imbalance      -> relation to irregularity
Team 7's Nsight profiling covers only the 4x4 test_small matrix, so it cannot be integrated (stated in README).
Outputs: integrated_mpi_comm.csv, integrated_effective_bandwidth.csv, integrated_numa_pinning.csv, plots/integration_extras.png"""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import INPUTS, TABLES, PLOTS, PAL, OPENMP, MPI

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 160, "axes.titleweight": "bold", "axes.grid": True, "grid.alpha": .25})
feat = pd.read_csv(TABLES / "matrix_features.csv").set_index("matrix_id")
cfg = pd.read_csv(TABLES / "unified_config_summary.csv")

# ---- A. MPI communication volume vs performance --------------------------------------------
c = pd.read_csv(INPUTS / "p6" / "communication_summary.csv")
c["candidate"] = np.where(c.scheme == "2d", "MPI-2D", np.where(c.partition == "nnz", "MPI-1D-nnz", "MPI-1D-rows"))
m = cfg[cfg.platform == "MPI"].copy()
m["ranks"] = m.config.str.replace("ranks", "").astype(int)
m = m.merge(c[["matrix_id", "candidate", "ranks", "total_bytes_sent", "max_rank_bytes_sent"]],
            left_on=["dataset", "candidate", "ranks"], right_on=["matrix_id", "candidate", "ranks"], how="left")
m["comm_fraction_pct"] = 100 * m.transfer_ms_median / m.kernel_ms_median     # exposed comm / wall time
m["bytes_per_nnz"] = m.total_bytes_sent / m.nnz
m.drop(columns=["matrix_id"]).to_csv(TABLES / "integrated_mpi_comm.csv", index=False)
R = 8
at = m[m.ranks == R]
rows = []
for d, g in at.groupby("dataset"):
    rows.append(dict(matrix_id=d, min_comm_scheme=g.loc[g.total_bytes_sent.idxmin(), "candidate"],
                     fastest_scheme=g.loc[g.gflops.idxmax(), "candidate"],
                     comm_ratio_max_min=g.total_bytes_sent.max() / max(g.total_bytes_sent.min(), 1)))
cmp_ = pd.DataFrame(rows).set_index("matrix_id")
cmp_["agree"] = cmp_.min_comm_scheme == cmp_.fastest_scheme
cmp_.to_csv(TABLES / "integrated_comm_vs_winner_8ranks.csv")
print(f"[P6] at {R} ranks the lowest-communication scheme is also the fastest on {cmp_.agree.sum()}/{len(cmp_)} matrices")
print(cmp_.round(1).to_string())
from scipy.stats import spearmanr
sub = m[(m.ranks == R) & (m.candidate == "MPI-1D-rows")]
print("[P6] Spearman(1D-rows bytes/nnz @8 ranks, features):",
      {f: round(spearmanr(feat.loc[sub.dataset, f], sub.bytes_per_nnz)[0], 2) for f in ["nnz", "mean_nnz_row", "irregularity", "rel_bandwidth"]})
print("[P6] median exposed-comm share of wall time @8 ranks by scheme (%):", at.groupby("candidate").comm_fraction_pct.median().round(1).to_dict())

# ---- B. Effective bandwidth vs STREAM (compulsory-traffic model) ---------------------------
stream = json.load(open(INPUTS / "p5" / "roofline.json"))["stream_bandwidth_gbs"]
best = pd.read_csv(TABLES / "best_per_candidate.csv")
def bytes_moved(r):
    f = feat.loc[r.dataset]
    if r.candidate == "OMP-COO":   return 16 * f.nnz + 8 * f.rows + 8 * f.cols
    if r.candidate == "OMP-ELL":   return 12 * f.rows * f.max_nnz_row + 8 * f.rows + 8 * f.cols
    return 12 * f.nnz + 4 * (f.rows + 1) + 8 * f.rows + 8 * f.cols            # CSR-like (also used for HYB/MPI/GPU)
best["model_bytes"] = best.apply(bytes_moved, axis=1)
best["effective_GBs"] = best.model_bytes / (best.kernel_ms_median * 1e-3) / 1e9
omp = best[best.candidate.isin(OPENMP)].copy()
omp["pct_of_STREAM"] = 100 * omp.effective_GBs / stream
best[["dataset", "candidate", "config", "gflops", "model_bytes", "effective_GBs"]].round(3).to_csv(TABLES / "integrated_effective_bandwidth.csv", index=False)
print(f"\n[P5] STREAM Triad = {stream} GB/s. OMP-CSR effective bandwidth, % of STREAM (>100% = matrix partly cache-resident):")
print(omp[omp.candidate == "OMP-CSR"].set_index("dataset").pct_of_STREAM.round(0).sort_values().to_dict())

# ---- C. first-touch / pinning / imbalance ---------------------------------------------------
ft = pd.read_csv(INPUTS / "p5" / "first_touch.csv")
ftp = ft.pivot_table(index=["size", "threads"], columns="mode", values="parallel_read_ms")
ftp["read_speedup_parallel_init"] = ftp.serial / ftp.parallel
pin = pd.read_csv(INPUTS / "p5" / "thread_pinning_summary.csv")
pp = pin.pivot_table(index=["dataset", "threads_or_ranks"], columns="proc_bind", values="mean_ms")
pp["pinned_speedup"] = pp[False] / pp[True]
sd = pin.pivot_table(index=["dataset", "threads_or_ranks"], columns="proc_bind", values="std_ms").max(axis=1)
pp["noise_cv_pct"] = 100 * sd / ((pp[False] + pp[True]) / 2)
pp.columns = [str(c) for c in pp.columns]
out = pd.concat({"first_touch(synthetic array)": ftp, "pinning": pp}, names=["study"])
out.round(3).to_csv(TABLES / "integrated_numa_pinning.csv")
print("\n[P5] first-touch (parallel vs serial init) read speedup by threads:", ftp.read_speedup_parallel_init.round(2).to_dict())
print("[P5] pinning speedup (unpinned/pinned) vs noise:\n", pp[["pinned_speedup", "noise_cv_pct"]].round(3).to_string())

im = pd.read_csv(INPUTS / "p5" / "static_partition_imbalance.csv")
im8 = im[im.threads == im.threads.max()].set_index("matrix_id")
print(f"\n[P5] static-partition imbalance @ {im.threads.max()} threads: Spearman vs irregularity =",
      round(spearmanr(im8.irregularity, im8.static_partition_imbalance)[0], 2), " max imbalance", round(im8.static_partition_imbalance.max(), 2))

# ---- plot -------------------------------------------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
order = feat.sort_values("nnz").index; x = np.arange(len(order)); w = .27
for k, cnd in enumerate(MPI):
    v = [at[(at.dataset == d) & (at.candidate == cnd)].total_bytes_sent.iloc[0] / 1e6 for d in order]
    a.bar(x + (k - 1) * w, np.maximum(v, 1e-3), w, color=PAL[cnd], label=cnd)
a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(order, rotation=60, ha="right", fontsize=7.5)
a.set_ylabel("total MB sent, 8 ranks"); a.set_title(f"MPI communication volume ({cmp_.agree.sum()}/{len(cmp_)} matrices: least-comm scheme is also fastest)")
a.legend(fontsize=8)
o = omp.pivot(index="dataset", columns="candidate", values="pct_of_STREAM").reindex(order)
for k, cnd in enumerate(OPENMP):
    b.bar(x + (k - 1) * w, o[cnd], w, color=PAL[cnd], label=cnd)
b.axhline(100, color="k", ls="--", lw=1); b.text(len(order) - .5, 103, "STREAM Triad (12.8 GB/s)", ha="right", fontsize=8)
b.set_xticks(x); b.set_xticklabels(order, rotation=60, ha="right", fontsize=7.5)
b.set_ylabel("effective GB/s, % of STREAM"); b.set_title("OpenMP: model-traffic bandwidth vs STREAM"); b.legend(fontsize=8)
fig.tight_layout(); fig.savefig(PLOTS / "integration_extras.png", bbox_inches="tight")
