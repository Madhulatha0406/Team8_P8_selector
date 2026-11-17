#!/usr/bin/env python3
"""Common-baseline graphs of the course brief, built from the merged data (all error bars are standard deviations):
  speedup_efficiency.png   speedup and parallel efficiency vs threads (OpenMP) / ranks (MPI), with the ideal line
  roofline.png             Roofline on Team 5's machine (STREAM bandwidth, estimated peak), SpMV points by format
  time_vs_size.png         kernel time and GFLOP/s vs problem size (nnz), best candidate per platform
  time_breakdown.png       MPI compute / remote / exposed-communication shares; GPU kernel vs transfer shares
  selector_tuning.png      the selector's tuning parameter (tree depth) vs leave-one-out accuracy
Weak scaling is not drawn: no team ran a problem that grows with the resources (the 13 matrices are fixed)."""
import json, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import INPUTS, TABLES, PLOTS, PAL, OPENMP, MPI, GPU_OWN
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 120, "savefig.dpi": 160,
                     "axes.grid": True, "grid.alpha": .25, "axes.titleweight": "bold"})
cfg = pd.read_csv(TABLES / "unified_config_summary.csv"); best = pd.read_csv(TABLES / "best_per_candidate.csv")
feat = pd.read_csv(TABLES / "matrix_features.csv").set_index("matrix_id")

def speedups(cands, key, base_val, vals):
    out = {}
    for c in cands:
        d = cfg[cfg.candidate == c].copy(); d["x"] = d.config.str.extract(key).astype(float)
        base = d[d.x == base_val].groupby("dataset").kernel_ms_median.min()
        rows = []
        for v in vals:
            t = d[d.x == v].groupby("dataset").kernel_ms_median.min(); sp = (base / t).dropna(); rows.append((v, sp.mean(), sp.std(ddof=0), len(sp)))
        out[c] = pd.DataFrame(rows, columns=["x", "mean", "std", "n"])
    return out

def fig_speedup():
    omp = speedups(OPENMP, r"thr(\d+)_", 1, [1, 4, 8]); mpi = speedups(MPI, r"ranks(\d+)", 1, [1, 2, 4, 8, 16])
    fig, ax = plt.subplots(2, 2, figsize=(12, 7))
    for col, (res, name, xs) in enumerate(((omp, "OpenMP threads (Team 5)", [1, 4, 8]), (mpi, "MPI ranks (Team 6)", [1, 2, 4, 8, 16]))):
        for c, d in res.items():
            ax[0, col].errorbar(d.x, d["mean"], yerr=d["std"], marker="o", capsize=3, color=PAL[c], label=f"{c} (n={int(d.n.min())}-{int(d.n.max())} matrices)")
            ax[1, col].errorbar(d.x, d["mean"] / d.x, yerr=d["std"] / d.x, marker="o", capsize=3, color=PAL[c], label=c)
        ax[0, col].plot(xs, xs, "k--", lw=1, label="ideal"); ax[1, col].axhline(1, color="k", ls="--", lw=1)
        ax[0, col].set_title(f"Speedup vs {name}"); ax[1, col].set_title(f"Parallel efficiency vs {name}")
        for r in (0, 1): ax[r, col].set_xticks(xs); ax[r, col].set_xlabel("threads" if col == 0 else "ranks")
        ax[0, col].legend(fontsize=7)
    ax[0, 0].set_ylabel("speedup (mean over matrices)"); ax[1, 0].set_ylabel("speedup / threads (or ranks)")
    fig.suptitle("Baseline = the same code at 1 thread / 1 rank (Team 5's -O3 sequential run; Team 6's single rank).  Error bars: std across matrices.\n"
                 "Team 5's STREAM bandwidth peaks at 6 threads; Team 6's hardware is undocumented, so the 16-rank collapse is reported, not explained.", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "speedup_efficiency.png", bbox_inches="tight"); plt.close(fig)

def fig_roofline():
    bw = json.load(open(INPUTS / "p5/roofline.json"))["stream_bandwidth_gbs"]; peak = 10 * 1.3 * 16   # 10 cores x 1.3 GHz (reported) x 16 DP FLOP/cycle (AVX2 FMA): estimate
    eb = pd.read_csv(TABLES / "integrated_effective_bandwidth.csv"); eb = eb[eb.candidate.isin(OPENMP)].copy(); eb["ai"] = 2 * eb.dataset.map(feat.nnz) / eb.model_bytes
    fig, ax = plt.subplots(figsize=(8.5, 5.5)); x = np.logspace(-2, 2, 200)
    ax.plot(x, np.minimum(peak, bw * x), "k-", lw=2, label=f"roof: min({peak:.0f} GFLOP/s est. peak, {bw:.2f} GB/s x AI)")
    ax.axvline(peak / bw, color="grey", ls=":", lw=1); ax.text(peak / bw * 1.06, 230, f"ridge {peak/bw:.0f} FLOP/B", fontsize=8, color="grey")
    for c in OPENMP:
        d = eb[eb.candidate == c]; ax.scatter(d.ai, d.gflops, s=38, color=PAL[c], edgecolor="white", label=c, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(.05, 60); ax.set_ylim(.1, 400)
    ax.set_xlabel("arithmetic intensity (FLOP per byte moved, compulsory-traffic model)"); ax.set_ylabel("GFLOP/s"); ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Roofline, OpenMP SpMV on Team 5's machine (i5-1235U)\nPoints above the bandwidth roof = matrix partly cache-resident (traffic model is a lower bound). Peak is an estimate.", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOTS / "roofline.png", bbox_inches="tight"); plt.close(fig)

def fig_size():
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for plat, grp in (("OpenMP", OPENMP), ("MPI", MPI), ("GPU", GPU_OWN)):
        d = best[best.candidate.isin(grp)]; d = d.loc[d.groupby("dataset").gflops.idxmax()].sort_values("nnz")
        ax[0].errorbar(d.nnz, d.kernel_ms_median, yerr=d.kernel_ms_std, marker="o", capsize=2, color=PAL[plat], label=plat)
        ax[1].errorbar(d.nnz, d.gflops, yerr=d.gflops * d.cv_pct / 100, marker="o", capsize=2, color=PAL[plat], label=plat)
    for a, yl in zip(ax, ("kernel time per SpMV (ms)", "GFLOP/s")): a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("problem size: nnz (13 matrices, 45k to 2M)"); a.set_ylabel(yl); a.legend()
    fig.suptitle("Time and throughput vs problem size, best candidate per platform; error bars = run-to-run std of that configuration (different machines)", fontsize=9.5)
    fig.tight_layout(); fig.savefig(PLOTS / "time_vs_size.png", bbox_inches="tight"); plt.close(fig)

def fig_breakdown():
    sp = pd.read_csv(INPUTS / "p6/spmv.csv"); sp = sp[(sp.ranks == 8) & (sp.scheme == "row") & (sp.partition == "nnz")]
    sp["dataset"] = sp.matrix.str.replace(".mtx", "", regex=False).replace({"soc-sign-Slashdot090221": "soc_sign_slashdot"})
    m = sp.groupby("dataset")[["local_compute_s", "remote_compute_s", "comm_exposed_s"]].median(); m = m.div(m.sum(axis=1), axis=0) * 100
    m = m.loc[[i for i in feat.sort_values("nnz").index if i in m.index]]
    g = cfg[cfg.candidate.isin(GPU_OWN)]; g = g.loc[g.groupby("dataset").kernel_ms_median.idxmin()].set_index("dataset")
    t = g[["kernel_ms_median", "transfer_ms_median"]]; t = t.div(t.sum(axis=1), axis=0) * 100; t = t.loc[feat.sort_values("nnz").index.intersection(t.index)]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5)); left = np.zeros(len(m))
    for col, c, lab in (("local_compute_s", "#0072B2", "local compute"), ("remote_compute_s", "#56B4E9", "remote compute"), ("comm_exposed_s", "#D55E00", "exposed communication")):
        ax[0].barh(m.index, m[col], left=left, color=c, label=lab); left += m[col].values
    ax[0].set_title("MPI 1D-nnz, 8 ranks: share of time (Team 6)"); ax[0].set_xlabel("% of SpMV time"); ax[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3)
    left = np.zeros(len(t))
    for col, c, lab in (("kernel_ms_median", "#009E73", "GPU kernel"), ("transfer_ms_median", "#CC79A7", "host-device transfer")):
        ax[1].barh(t.index, t[col], left=left, color=c, label=lab); left += t[col].values
    ax[1].set_title("GPU, best kernel: kernel vs one-time transfer (Team 7, 1 SpMV)"); ax[1].set_xlabel("% of time for a single SpMV"); ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1), ncol=2)
    fig.suptitle("Compute / communication / transfer shares (rows sorted by nnz). Synchronisation time was not recorded by any team.", fontsize=9.5)
    fig.tight_layout(); fig.savefig(PLOTS / "time_breakdown.png", bbox_inches="tight"); plt.close(fig)

def fig_tuning():
    d = pd.read_csv(TABLES / "selector_depth_sensitivity.csv"); tasks = ["GPU kernel (own)", "GPU kernel (+cuSPARSE)", "MPI partition", "OpenMP format"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); cols = ["#009E73", "#555555", "#E69F00", "#0072B2"]
    for t, c in zip(tasks, cols):
        for fs, ls in (("Tree-10", "-"), ("Tree-6", "--")):
            x = d[(d.task == t) & (d.features == fs)].sort_values("depth"); ax.plot(x.depth, x.top1, ls, marker="o", color=c, label=f"{t}, {fs}")
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("decision-tree depth (the selector's tuning parameter)"); ax.set_ylabel("leave-one-out top-1 accuracy (%)"); ax.set_ylim(0, 105)
    ax.legend(fontsize=7, ncol=1, loc="lower right"); ax.set_title("Selector accuracy vs tree depth (13 matrices)")
    fig.tight_layout(); fig.savefig(PLOTS / "selector_tuning.png", bbox_inches="tight"); plt.close(fig)

for f in (fig_speedup, fig_roofline, fig_size, fig_breakdown, fig_tuning): f(); print("wrote", f.__name__)
