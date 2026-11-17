#!/usr/bin/env python3
"""Task 3: plot performance against each matrix feature and look for correlations.

Figures (results/plots/):
  feature_vs_performance.png    best GFLOP/s per platform vs the six assignment features
  correlation_heatmap.png       Spearman rho, candidate x feature (absolute and relative perf.)
  relative_perf_vs_features.png % of platform-best vs mean nnz/row and irregularity
  winner_map.png                which candidate wins on each matrix
Tables (results/tables/): correlations_long.csv

Why Spearman: n = 13 matrices, features span 4-6 orders of magnitude and relationships are
monotone-but-not-linear; rank correlation is robust to that.  With n = 13, |rho| > 0.56 is
needed for an (uncorrected) two-sided p < 0.05, and we test ~110 (candidate, feature) pairs, so
a few "significant" cells are expected by chance -- treat the heatmap as hypothesis-generating.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr
from common import (TABLES, PLOTS, PAL, ALL, OPENMP, MPI, GPU_OWN, PLATFORM_OF, BASE_FEATURES,
                    ALL_FEATURES, FEATURE_LABELS)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 160, "axes.grid": True,
                     "grid.alpha": .25, "axes.titleweight": "bold"})


def load():
    feat = pd.read_csv(TABLES / "matrix_features.csv").set_index("matrix_id")
    perf = pd.read_csv(TABLES / "performance_matrix.csv").set_index("matrix_id")
    return feat.loc[perf.index], perf


def platform_best(perf):
    return pd.DataFrame({"OpenMP": perf[OPENMP].max(axis=1), "MPI": perf[MPI].max(axis=1),
                         "GPU": perf[GPU_OWN].max(axis=1)})


def spearman(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5 or y[ok].nunique() < 3:   # degenerate (e.g. a format that wins nearly everywhere)
        return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(x[ok], y[ok])
    return r, p, int(ok.sum())


# ---------------------------------------------------------------------------------------------
def _plain_log_axis(ax, which, ticks=None):
    """Log axis with plain-number tick labels (1k, 10k, 2, 5, ...)."""
    from matplotlib.ticker import FuncFormatter, NullFormatter
    def fmt(v, _):
        if v >= 1e6: return f"{v/1e6:g}M"
        if v >= 1e3: return f"{v/1e3:g}k"
        return f"{v:g}"
    axis = ax.xaxis if which == "x" else ax.yaxis
    if ticks is not None:
        axis.set_ticks(ticks)
    axis.set_major_formatter(FuncFormatter(fmt)); axis.set_minor_formatter(NullFormatter())


def fig_feature_vs_perf(feat, perf):
    pb = platform_best(perf)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), sharey=True)
    outliers = {"qpband", "TSOPF_FS_b39_c19", "soc_sign_slashdot", "appu"}
    short = {"TSOPF_FS_b39_c19": "TSOPF", "soc_sign_slashdot": "slashdot"}
    mk = {"OpenMP": "o", "MPI": "s", "GPU": "^"}
    cvp = pd.read_csv(TABLES / "best_per_candidate.csv").set_index(["dataset", "candidate"]).cv_pct
    groups = {"OpenMP": OPENMP, "MPI": MPI, "GPU": GPU_OWN}
    def perr(plat):      # run-to-run std of the winning configuration, as an error bar
        idx = perf[groups[plat]].idxmax(axis=1)
        return pd.Series({m: pb.loc[m, plat] * cvp.get((m, idx[m]), 0) / 100 for m in pb.index})
    for ax, f in zip(axes.flat, BASE_FEATURES):
        parts = []
        for plat in ["OpenMP", "MPI", "GPU"]:
            r, p, n = spearman(feat[f], pb[plat])
            ax.errorbar(feat[f], pb[plat], yerr=perr(plat).reindex(feat.index), fmt=mk[plat], color=PAL[plat], ms=6, alpha=.9,
                        mec="white", mew=.6, ecolor=PAL[plat], elinewidth=1, capsize=2, label=plat)
            parts.append(f"{plat[:3] if plat != 'OpenMP' else 'OMP'} {r:+.2f}{'*' if p < .05 else ''}")
        ax.set_xscale("log"); ax.set_yscale("log")
        xt = {"rows": [1e4, 2e4, 5e4, 1e5], "nnz": [5e4, 1e5, 3e5, 1e6, 2e6]}.get(f)
        _plain_log_axis(ax, "x", xt); _plain_log_axis(ax, "y", [2, 3, 5, 10, 20, 30])
        ax.set_xlabel(FEATURE_LABELS[f])
        ax.set_title("ρ:  " + "   ".join(parts), fontsize=8.5, fontweight="normal")
        for m in outliers:
            ax.annotate(short.get(m, m), (feat.loc[m, f], pb.loc[m, "GPU"]), fontsize=6.5,
                        xytext=(3, 3), textcoords="offset points", color="#444")
    for ax in axes[:, 0]:
        ax.set_ylabel("best GFLOP/s (tuned, log scale)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.suptitle("Performance vs matrix features — best candidate per platform, 13 matrices\n"
                 "ρ = Spearman rank correlation, * = p < 0.05 uncorrected (|ρ| > 0.56 at n = 13).  Error bars = run-to-run std of the best configuration.  "
                 "Platforms ran on different machines: compare trends, not absolute levels.",
                 fontsize=10, y=1.0)
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(PLOTS / "feature_vs_performance.png", bbox_inches="tight"); plt.close(fig)


def correlation_tables(feat, perf):
    # relative performance: candidate GFLOP/s / best candidate on the same platform & matrix
    rel = perf.copy()
    for grp in (OPENMP, MPI, GPU_OWN):
        rel[grp] = perf[grp].div(perf[grp].max(axis=1), axis=0)
    rel["GPU-cuSPARSE"] = perf["GPU-cuSPARSE"] / perf[GPU_OWN + ["GPU-cuSPARSE"]].max(axis=1)
    rows = []
    for kind, M in (("absolute", perf), ("relative_to_platform_best", rel)):
        for c in ALL:
            for f in ALL_FEATURES:
                r, p, n = spearman(feat[f], M[c])
                rows.append(dict(kind=kind, candidate=c, feature=f, rho=r, p=p, n=n))
    long = pd.DataFrame(rows)
    long.to_csv(TABLES / "correlations_long.csv", index=False)
    return long


def fig_heatmap(long):
    fig = plt.figure(figsize=(14.5, 5.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, .03], wspace=.06)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
    for ax, kind, title in zip(axes[:2], ["absolute", "relative_to_platform_best"],
                               ["Absolute GFLOP/s", "Relative to best candidate on same platform"]):
        t = long[long.kind == kind]
        R = t.pivot(index="candidate", columns="feature", values="rho").loc[ALL, ALL_FEATURES]
        P = t.pivot(index="candidate", columns="feature", values="p").loc[ALL, ALL_FEATURES]
        im = ax.imshow(R.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.grid(False)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                v = R.values[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=6.5, color="#999")
                    continue
                sig = P.values[i, j] < .05
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        fontweight="bold" if sig else "normal",
                        color="white" if abs(v) > .6 else "black")
                if sig:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=1.1))
        ax.set_xticks(range(len(ALL_FEATURES)))
        ax.set_xticklabels([FEATURE_LABELS[f].split(" (")[0] for f in ALL_FEATURES], rotation=40, ha="right")
        ax.set_yticks(range(len(ALL)))
        ax.set_yticklabels(ALL if ax is axes[0] else [])
        for k in (2.5, 5.5):
            ax.axhline(k, color="white", lw=3)
        ax.axvline(5.5, color="white", lw=3)
        ax.set_title(title)
    fig.colorbar(im, cax=axes[2], label="Spearman ρ")
    fig.suptitle("Which features move which candidate?   Boxed = p < 0.05 (uncorrected; ~200 tests, so "
                 "several are expected by chance).  n/a = fewer than 3 distinct values (e.g. OMP-CSR "
                 "is best on 12 of 13 matrices).", fontsize=9.5, y=1.0)
    fig.savefig(PLOTS / "correlation_heatmap.png", bbox_inches="tight"); plt.close(fig)


def fig_relative(feat, perf):
    groups = [("OpenMP (CPU)", OPENMP), ("MPI partitioning", MPI), ("GPU kernels", GPU_OWN)]
    xs = [("mean_nnz_row", "mean nnz / row"), ("irregularity", "irregularity = max / mean nnz per row")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for i, (f, flabel) in enumerate(xs):
        for j, (gname, cands) in enumerate(groups):
            ax = axes[i, j]
            rel = perf[cands].div(perf[cands].max(axis=1), axis=0)
            order = feat[f].sort_values().index
            for c in cands:
                ax.scatter(feat.loc[order, f], rel.loc[order, c], s=34, alpha=.85, color=PAL[c],
                           edgecolor="white", linewidth=.5, label=c)
            ax.set_xscale("log"); _plain_log_axis(ax, "x"); ax.set_ylim(0, 1.05)
            ax.axhline(.95, color="grey", ls=":", lw=1)
            ax.set_xlabel(flabel)
            if j == 0:
                ax.set_ylabel("fraction of best in group")
            if i == 0:
                ax.set_title(gname)
            if i == 1:
                ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(.5, -.28), ncol=len(cands),
                          frameon=False, handletextpad=.1, columnspacing=1)
    fig.suptitle("Relative performance within each platform vs the two features that matter most "
                 "(dotted line = within 5% of best, i.e. the run-to-run noise level)", fontsize=10)
    fig.tight_layout(); fig.savefig(PLOTS / "relative_perf_vs_features.png", bbox_inches="tight")
    plt.close(fig)


def fig_winner_map(feat, perf):
    w = pd.read_csv(TABLES / "winners.csv").set_index("matrix_id")
    order = feat.sort_values("mean_nnz_row").index
    cols = [("best_OpenMP", "OpenMP\nformat"), ("best_MPI", "MPI\npartition"),
            ("best_GPU_own", "GPU kernel\n(own)"), ("best_GPU_incl_cuSPARSE", "GPU kernel\n(+cuSPARSE)")]
    fcols = [("mean_nnz_row", "mean\nnnz/row", "{:.1f}"), ("irregularity", "irregu-\nlarity", "{:.1f}"),
             ("nnz", "nnz", "{:,.0f}")]
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    n = len(order)
    for r, m in enumerate(order):
        y = n - r - 1
        ax.text(-0.05, y + .5, m, ha="right", va="center", fontsize=9)
        for k, (f, _, fmt) in enumerate(fcols):
            ax.text(k + .5, y + .5, fmt.format(feat.loc[m, f]), ha="center", va="center",
                    fontsize=8.5, color="#333")
        for k, (c, _) in enumerate(cols):
            cand = w.loc[m, c]
            ax.add_patch(Rectangle((len(fcols) + k + .04, y + .06), .92, .88, color=PAL[cand], lw=0))
            dark = cand in ("OMP-ELL", "MPI-2D", "GPU-cuSPARSE", "OMP-CSR", "GPU-HYB", "GPU-vector-CSR")
            ax.text(len(fcols) + k + .5, y + .5, cand.replace("GPU-", "").replace("MPI-", "").replace("OMP-", ""),
                    ha="center", va="center", fontsize=8.5, color="white" if dark else "black",
                    fontweight="bold")
    for k, (_, lab, _) in enumerate(fcols):
        ax.text(k + .5, n + .15, lab, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for k, (_, lab) in enumerate(cols):
        ax.text(len(fcols) + k + .5, n + .15, lab, ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xlim(-0.05, len(fcols) + len(cols)); ax.set_ylim(0, n + 1.1)
    ax.axis("off")
    ax.set_title("Winner per matrix (sorted by mean nnz/row) — best tuned configuration within each platform",
                 fontsize=10.5, pad=14)
    fig.savefig(PLOTS / "winner_map.png", bbox_inches="tight"); plt.close(fig)


def main():
    feat, perf = load()
    fig_feature_vs_perf(feat, perf)
    long = correlation_tables(feat, perf)
    fig_heatmap(long)
    fig_relative(feat, perf)
    fig_winner_map(feat, perf)
    # console summary of strongest, significant correlations
    s = long[(long.p < .05)].copy()
    s["abs"] = s.rho.abs()
    print("significant (uncorrected p<0.05) Spearman correlations:")
    print(s.sort_values(["kind", "abs"], ascending=[True, False])
            .drop(columns="abs").round(3).to_string(index=False))
    print("\nplots written to", PLOTS)


if __name__ == "__main__":
    main()
