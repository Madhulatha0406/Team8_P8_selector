#!/usr/bin/env python3
"""Task 4: rule-based and decision-tree selectors for the best format / platform, with accuracy.

Selection tasks (the "label" for a matrix is the candidate with the highest tuned GFLOP/s):
  OpenMP format           COO / CSR / ELL
  MPI partition           1D-rows / 1D-nnz / 2D
  GPU kernel (own)        scalar-CSR / vector-CSR / ELL / HYB
  GPU kernel (+cuSPARSE)  same, with the vendor library as an extra choice
  CPU-side platform       OpenMP vs MPI
  Platform                OpenMP vs MPI vs GPU (best own kernel on each)
  Overall (own)           best format+platform among all 10 own implementations
  Overall (+cuSPARSE)     same, plus cuSPARSE

Methods compared
  Majority    always predict the most common label of the *other* 12 matrices (leave-one-out)
  Tree-6      decision tree (depth 2) on the six assignment features        (leave-one-out)
  Tree-10     decision tree (depth 2) on the six + four derived features    (leave-one-out)
  Rule        hand-written thresholds (textbook values: warp width 32, ELL padding <= 1.5, ...)
  Tree-10 fit training accuracy of the tree fitted on all 13 matrices (in-sample, optimistic)

Metrics
  Top-1      % of matrices where the predicted candidate is the exact best
  Within-5%  % of matrices where the predicted candidate reaches >= 95% of the best GFLOP/s
             (5% is the run-to-run noise level, so a miss inside 5% is not a real miss)
  Mean/Min % of best   average / worst fraction of the best performance the choice delivers
              (an infeasible choice, e.g. ELL where it was not benchmarked, counts as 0)

IMPORTANT caveats (also stated in the README): n = 13 matrices; leave-one-out is the only
unbiased estimate available; the Rule row's thresholds were picked by us after looking at the
performance tables, so treat the Rule row as in-sample, not as a held-out test.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from reuse import platform_perf
from common import (TABLES, PLOTS, PAL, OPENMP, MPI, GPU_OWN, GPU_VENDOR, ALL_OWN, ALL,
                    BASE_FEATURES, ALL_FEATURES, FEATURE_LABELS)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.dpi": 160, "axes.titleweight": "bold"})
DEPTH = 2
TIE = 0.95


# ------------------------------------------------------------------------------------------
def load():
    feat = pd.read_csv(TABLES / "matrix_features.csv").set_index("matrix_id")
    perf = pd.read_csv(TABLES / "performance_matrix.csv").set_index("matrix_id")
    return feat.loc[perf.index], perf


def build_tasks(perf):
    """Return {task: DataFrame(matrices x classes) of GFLOP/s}."""
    plat = lambda gpu_cols: pd.DataFrame({"OpenMP": perf[OPENMP].max(axis=1),
                                          "MPI": perf[MPI].max(axis=1),
                                          "GPU": perf[gpu_cols].max(axis=1)})
    return {
        "OpenMP format": perf[OPENMP],
        "MPI partition": perf[MPI],
        "GPU kernel (own)": perf[GPU_OWN],
        "GPU kernel (+cuSPARSE)": perf[GPU_OWN + GPU_VENDOR],
        "CPU-side platform (OpenMP vs MPI)": plat(GPU_OWN)[["OpenMP", "MPI"]],
        "Platform (OpenMP/MPI/GPU)": plat(GPU_OWN),
        "Platform @R=5 (OpenMP vs GPU, incl. transfer)": platform_perf(5),
        "Platform @R=10 (OpenMP vs GPU, incl. transfer)": platform_perf(10),
        "Overall format+platform (own)": perf[ALL_OWN],
        "Overall format+platform (+cuSPARSE)": perf[ALL],
    }


def gpu_rule(r):                          # Bell & Garland style: one warp per row pays off when rows >= warp width
    return "GPU-vector-CSR" if r.mean_nnz_row >= 32 else "GPU-HYB"


RULES = {
    "OpenMP format": lambda r: "OMP-ELL" if r.irregularity <= 1.5 else "OMP-CSR",   # ELL only if padding <= 50%
    "MPI partition": lambda r: "MPI-1D-nnz" if r.cv_nnz_row >= 0.5 else "MPI-1D-rows",
    "GPU kernel (own)": gpu_rule,
    "GPU kernel (+cuSPARSE)": lambda r: "GPU-cuSPARSE",                              # default to the vendor library
    "CPU-side platform (OpenMP vs MPI)": lambda r: "OpenMP" if r.nnz < 1e5 else "MPI",  # tiny problem: skip MPI overhead
    "Platform (OpenMP/MPI/GPU)": lambda r: "GPU",
    "Platform @R=5 (OpenMP vs GPU, incl. transfer)": lambda r: "OpenMP" if r.nnz < 1e6 else "GPU",
    "Platform @R=10 (OpenMP vs GPU, incl. transfer)": lambda r: "OpenMP" if r.nnz < 1e6 else "GPU",
    "Overall format+platform (own)": gpu_rule,
    "Overall format+platform (+cuSPARSE)": lambda r: "GPU-cuSPARSE",
}
RULE_TEXT = {
    "OpenMP format": "irregularity <= 1.5 -> ELL, else CSR",
    "MPI partition": "CV(nnz/row) >= 0.5 -> 1D equal-nnz, else 1D equal-rows",
    "GPU kernel (own)": "mean nnz/row >= 32 -> vector-CSR, else HYB",
    "GPU kernel (+cuSPARSE)": "always cuSPARSE",
    "CPU-side platform (OpenMP vs MPI)": "nnz < 1e5 -> OpenMP, else MPI",
    "Platform (OpenMP/MPI/GPU)": "always GPU",
    "Platform @R=5 (OpenMP vs GPU, incl. transfer)": "nnz < 1e6 -> OpenMP, else GPU",
    "Platform @R=10 (OpenMP vs GPU, incl. transfer)": "nnz < 1e6 -> OpenMP, else GPU",
    "Overall format+platform (own)": "GPU rule: mean nnz/row >= 32 -> vector-CSR, else HYB",
    "Overall format+platform (+cuSPARSE)": "always GPU cuSPARSE",
}


def majority(labels):
    vc = labels.value_counts()
    top = vc[vc == vc.max()].index.sort_values()
    return top[0]


def tree(depth=DEPTH):
    return DecisionTreeClassifier(max_depth=depth, random_state=0)


def loo_predict(X, y, depth=DEPTH):
    pred = []
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        pred.append(tree(depth).fit(X[tr], y[tr]).predict(X[[i]])[0])
    return np.array(pred)


def score(P, y, pred):
    """P: matrices x classes GFLOP/s; returns the metrics for predictions `pred`."""
    best = P.max(axis=1).values
    got = np.array([np.nan_to_num(P.iloc[i][pred[i]], nan=0.0) for i in range(len(y))])
    rel = got / best
    return dict(top1=100 * np.mean(pred == y.values), within5=100 * np.mean(rel >= TIE),
                mean_rel=100 * rel.mean(), min_rel=100 * rel.min()), rel


def run():
    feat, perf = load()
    tasks = build_tasks(perf)
    acc_rows, pred_rows, depth_rows, tree_txt = [], [], [], []
    fitted = {}
    for name, P in tasks.items():
        y = P.idxmax(axis=1)
        counts = y.value_counts().to_dict()
        Xb, Xe = feat[BASE_FEATURES].values.astype(float), feat[ALL_FEATURES].values.astype(float)
        preds = {
            "Majority (LOO)": np.array([majority(y.drop(m)) for m in y.index]),
            "Tree-6 (LOO)": loo_predict(Xb, y.values),
            "Tree-10 (LOO)": loo_predict(Xe, y.values),
            "Rule (hand-set)": np.array([RULES[name](feat.loc[m]) for m in y.index]),
        }
        fit = tree().fit(Xe, y.values)
        preds["Tree-10 fit on all 13 (in-sample)"] = fit.predict(Xe)
        fitted[name] = (fit, y)
        tree_txt.append(f"### {name}\nlabel counts: {counts}\n" +
                        export_text(fit, feature_names=ALL_FEATURES, show_weights=True) + "\n")
        for method, pr in preds.items():
            s, rel = score(P, y, pr)
            acc_rows.append(dict(task=name, method=method, n=len(y), n_classes=y.nunique(),
                                 label_counts=json.dumps(counts), **s))
            if method != "Tree-10 fit on all 13 (in-sample)":
                for k, m in enumerate(y.index):
                    pred_rows.append(dict(task=name, matrix_id=m, method=method, true_best=y[m],
                                          predicted=pr[k], correct=int(pr[k] == y[m]),
                                          pct_of_best=100 * rel[k]))
        for d in (1, 2, 3):
            for fs, X in (("Tree-6", Xb), ("Tree-10", Xe)):
                pr = loo_predict(X, y.values, d)
                s, _ = score(P, y, pr)
                depth_rows.append(dict(task=name, features=fs, depth=d, **s))
    acc = pd.DataFrame(acc_rows)
    acc["rule_text"] = acc.apply(lambda r: RULE_TEXT[r.task] if r.method == "Rule (hand-set)" else "", axis=1)
    acc.round(1).to_csv(TABLES / "selector_accuracy.csv", index=False)
    pd.DataFrame(pred_rows).round(1).to_csv(TABLES / "selector_predictions.csv", index=False)
    pd.DataFrame(depth_rows).round(1).to_csv(TABLES / "selector_depth_sensitivity.csv", index=False)
    (TABLES / "selector_trees.txt").write_text("\n".join(tree_txt))
    return feat, perf, tasks, acc, fitted


def stump_table(feat, tasks):
    """Per-feature one-split trees. LOO accuracy of each feature used alone.

    NOTE: picking the best row of this table is itself a selection among 10 features on the
    same 13 matrices, so its top LOO figure is mildly optimistic; the unbiased number for
    'let the data pick the feature' is the Tree-10 depth-1 row of selector_depth_sensitivity.csv.
    The table's purpose is to show *how much better* one feature is than the others.
    """
    rows = []
    for name, P in tasks.items():
        y = P.idxmax(axis=1)
        if y.nunique() < 2:
            continue
        for f in ALL_FEATURES:
            X = feat[[f]].values.astype(float)
            pred = loo_predict(X, y.values, 1)
            s, _ = score(P, y, pred)
            full = tree(1).fit(X, y.values)
            lo, hi = full.classes_[full.tree_.value[1].argmax()], full.classes_[full.tree_.value[2].argmax()]
            rows.append(dict(task=name, feature=f, threshold=full.tree_.threshold[0],
                             left_class=lo, right_class=hi,
                             train_top1=100 * np.mean(full.predict(X) == y.values),
                             loo_top1=s["top1"], loo_within5=s["within5"], loo_mean_rel=s["mean_rel"]))
    out = pd.DataFrame(rows).sort_values(["task", "loo_top1", "loo_mean_rel"], ascending=[True, False, False])
    out.round(3).to_csv(TABLES / "selector_single_feature_stumps.csv", index=False)
    return out


# ------------------------------------------------------------------------------------------
def fig_accuracy(acc):
    short = {"OpenMP format": "OpenMP\nformat", "MPI partition": "MPI\npartition",
             "GPU kernel (own)": "GPU kernel\n(own)", "GPU kernel (+cuSPARSE)": "GPU kernel\n(+cuSPARSE)",
             "CPU-side platform (OpenMP vs MPI)": "Platform:\nOpenMP vs MPI",
             "Platform (OpenMP/MPI/GPU)": "Platform:\n3-way",
             "Platform @R=5 (OpenMP vs GPU, incl. transfer)": "OpenMP vs GPU\n+transfer, R=5",
             "Platform @R=10 (OpenMP vs GPU, incl. transfer)": "OpenMP vs GPU\n+transfer, R=10"}
    order = [t for t in dict.fromkeys(acc.task) if t in short]   # 'Overall' tasks == GPU tasks, omitted here
    methods = ["Majority (LOO)", "Tree-6 (LOO)", "Tree-10 (LOO)", "Rule (hand-set)"]
    cols = ["#BBBBBB", "#56B4E9", "#0072B2", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.2))
    w = .2
    for ax, metric, title in zip(axes, ["top1", "mean_rel"],
                                 ["Top-1: predicted candidate is exactly the best (%)",
                                  "Mean performance delivered (% of per-matrix best)"]):
        for k, (m, c) in enumerate(zip(methods, cols)):
            v = [acc[(acc.task == t) & (acc.method == m)][metric].iloc[0] for t in order]
            b = ax.bar(np.arange(len(order)) + (k - 1.5) * w, v, w, color=c, label=m)
            for rect, val in zip(b, v):
                ax.text(rect.get_x() + rect.get_width() / 2, val + 1, f"{val:.0f}", ha="center",
                        fontsize=6.5, rotation=90)
        ax.set_xticks(np.arange(len(order)))
        ax.set_xticklabels([short[t] for t in order], fontsize=8)
        ax.set_ylim(0, 115); ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", alpha=.25)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.02))
    fig.suptitle("Selector accuracy, 13 matrices.  Majority and trees: leave-one-out.  Rule: hand-set thresholds "
                 "(in-sample).\n'Platform: 3-way' is trivial (GPU wins 13/13); 'Overall' tasks are identical to "
                 "the GPU rows and are in selector_accuracy.csv.", fontsize=9.5)
    fig.tight_layout(rect=(0, .05, 1, .93)); fig.savefig(PLOTS / "selector_accuracy.png", bbox_inches="tight")
    plt.close(fig)


def fig_trees(fitted):
    show = ["GPU kernel (own)", "GPU kernel (+cuSPARSE)", "MPI partition",
            "CPU-side platform (OpenMP vs MPI)", "OpenMP format"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.5))
    for ax, name in zip(axes.flat, show):
        fit, y = fitted[name]
        plot_tree(fit, feature_names=[FEATURE_LABELS[f].split(" (")[0] for f in ALL_FEATURES],
                  class_names=list(fit.classes_), filled=True, rounded=True, impurity=False,
                  fontsize=8, ax=ax)
        counts = ", ".join(f"{k.split('-', 1)[-1]}: {int(v)}" for k, v in y.value_counts().items())
        ax.set_title(f"{name}\nbest-candidate counts over 13 matrices — {counts}", fontsize=8.5)
    axes.flat[-1].axis("off")
    axes.flat[-1].text(0, .5, "Reading guide\n\nvalue = [matrices per class, in\nalphabetical class order]\n\n"
                       "Trees fitted on all 13 matrices\nfor interpretation only.\nHonest accuracy: leave-one-out\n"
                       "table / selector_accuracy.png.\n\nLeaves with 1-2 matrices\n(e.g. rows <= 25185 -> vector)\n"
                       "are anecdotes, not rules.", fontsize=9, va="center", transform=axes.flat[-1].transAxes)
    fig.suptitle("Depth-2 decision trees (10 candidate features)", fontsize=11)
    fig.tight_layout(); fig.savefig(PLOTS / "selector_trees.png", bbox_inches="tight"); plt.close(fig)


def main():
    feat, perf, tasks, acc, fitted = run()
    fig_accuracy(acc)
    fig_trees(fitted)
    st = stump_table(feat, tasks)
    print("\nsingle-feature stumps (top 3 per task by LOO top-1):")
    print(st.groupby("task").head(3)[["task", "feature", "threshold", "left_class", "right_class",
                                       "loo_top1", "loo_mean_rel"]].round(2).to_string(index=False))
    dep = pd.read_csv(TABLES / "selector_depth_sensitivity.csv")
    print("\ndepth sensitivity (LOO top-1 / mean %best):")
    print(dep.pivot_table(index=["task", "features"], columns="depth", values=["top1", "mean_rel"]).round(0).to_string())
    show = acc.copy()
    show["task"] = show.task.str.slice(0, 34)
    print(show[["task", "method", "n_classes", "top1", "within5", "mean_rel", "min_rel"]]
          .round(1).to_string(index=False))
    print("\nlabel counts:")
    for t in tasks:
        print(f"  {t:38s} {acc[acc.task == t].label_counts.iloc[0]}")
    print("\n" + (TABLES / "selector_trees.txt").read_text())


if __name__ == "__main__":
    main()
