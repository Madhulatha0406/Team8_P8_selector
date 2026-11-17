#!/usr/bin/env python3
"""Task 1: collect the CSV results of P5 (OpenMP), P6 (MPI) and P7 (CUDA) into one clean table.

Outputs (results/tables/):
  unified_raw.csv            every timed run from all three teams, common CSV schema
  unified_config_summary.csv one row per (matrix, candidate, configuration): median over runs
  best_per_candidate.csv     best configuration per (matrix, candidate)   [long]
  performance_matrix.csv     best GFLOP/s per (matrix x candidate)        [wide]
  winners.csv                winning candidate per matrix, per platform and overall

Conventions
  * GFLOP/s = 2*nnz / kernel_time (project convention). Statistic over repeated runs = MEDIAN
    of kernel time (robust to the occasional OS-noise spike on laptops / shared Colab GPUs).
  * Warm-up runs are dropped (P5: `warmup` flag; P6: already excluded by Team 6's code;
    P7: no warm-up flag in the CSV, rows used as delivered).
  * "best configuration" = max over each team's tuning axes: P5 threads x schedule x chunk,
    P6 ranks, P7 block size / warps-per-block / ELL width.  This is the *tuned* performance
    of each candidate, i.e. what a user who tunes would get.
  * The benchmark set is the 13 matrices in matrices.csv (Team 7's 4x4 `test_small`
    correctness fixture is dropped).
"""
import numpy as np
import pandas as pd
from common import INPUTS, TABLES, ALL, OPENMP, MPI, GPU_OWN, PLATFORM_OF, gflops

COMMON_COLS = ["cluster", "problem", "platform", "implementation", "candidate", "dataset",
               "rows", "cols", "nnz", "config", "threads_or_ranks", "schedule", "run_id",
               "total_ms", "kernel_ms", "transfer_ms", "throughput_gflops", "error", "power_w"]
ERR_TOL = 1e-4          # runs with larger |error| vs the reference would be rejected


def load_matrix_table():
    manifest = pd.read_csv(INPUTS / "matrices.csv")
    meta = pd.read_csv(INPUTS / "matrix_metadata.csv")
    m = meta[["matrix_id", "rows", "cols", "nnz"]].set_index("matrix_id")
    # P6 uses the SuiteSparse file name (e.g. soc-sign-Slashdot090221.mtx) as matrix key
    file_to_id = {p.split("/")[-1]: i for i, p in zip(manifest.matrix_id, manifest.relative_path)}
    return m, file_to_id


def p5(m):
    r = pd.read_csv(INPUTS / "p5" / "team5_suitesparse_raw.csv")
    r = r[r.warmup == 0].copy()
    fmt = r.implementation.str.split("_").str[0].str.upper()
    # 1-thread sequential runs are the 1-thread point of the same format
    r["candidate"] = "OMP-" + fmt
    r["config"] = ("thr" + r.threads_or_ranks.astype(str) + "_" + r.schedule + "_chunk"
                   + r.chunk.astype(str) + "_" + r.implementation.str.split("_").str[1])
    r["cluster"], r["problem"], r["platform"] = "Team5", "P5", "CPU-OpenMP"
    r["error"] = pd.to_numeric(r.error, errors="coerce")
    out = r.rename(columns={"throughput": "throughput_orig"})
    out = out.join(m, on="dataset")
    # sanity: Team 5's own `throughput` column == 2*nnz/kernel_ms
    chk = np.abs(gflops(out.nnz, out.kernel_ms) / out.throughput_orig - 1).max()
    assert chk < 1e-3, f"P5 throughput column inconsistent with 2*nnz convention ({chk})"
    return out


def p6(m, file_to_id):
    r = pd.read_csv(INPUTS / "p6" / "spmv.csv")
    r["dataset"] = r.matrix.map(file_to_id)
    assert r.dataset.notna().all()
    scheme = np.where(r.scheme == "2d", "MPI-2D",
                      np.where(r.partition == "nnz", "MPI-1D-nnz", "MPI-1D-rows"))
    out = pd.DataFrame({
        "cluster": "Team6", "problem": "P6", "platform": "MPI", "implementation": scheme,
        "candidate": scheme, "dataset": r.dataset,
        "config": "ranks" + r.ranks.astype(str), "threads_or_ranks": r.ranks,
        "schedule": "", "run_id": r.iteration, "total_ms": r.elapsed_s * 1e3,
        "kernel_ms": r.elapsed_s * 1e3,                  # max-over-ranks wall time per SpMV
        "transfer_ms": r.comm_exposed_s * 1e3,           # exposed (non-overlapped) comm time
        "error": 0.0,                                    # Team 6 aborts on any failed validation
        "power_w": np.nan})
    out = out.join(m, on="dataset")
    assert (out.nnz == r.nnz.values).all(), "P6 nnz differs from metadata"
    return out


def p7(m):
    r = pd.read_csv(INPUTS / "p7" / "team7_raw.csv")
    r = r[r.dataset != "test_small"].copy()
    name = {"scalar_csr": "GPU-scalar-CSR", "vector_csr": "GPU-vector-CSR", "ell": "GPU-ELL",
            "hyb": "GPU-HYB", "cusparse": "GPU-cuSPARSE", "cpu_sequential": "CPU-seq(T7)"}
    r["candidate"] = r.implementation.map(name)
    r["config"] = r.threads_or_ranks.astype(str)
    r["threads_or_ranks"] = np.where(r.platform == "CPU", 1, np.nan)
    r["cluster"] = "Team7"
    r["error"] = pd.to_numeric(r.error, errors="coerce")
    r["platform"] = r.platform.map({"NVIDIA_GPU": "GPU", "CPU": "CPU-seq(T7)"})
    out = r.rename(columns={"throughput": "throughput_orig"}).join(m, on="dataset")
    chk = np.abs(gflops(out.nnz, out.kernel_ms) / out.throughput_orig - 1).max()
    assert chk < 1e-2, f"P7 throughput column inconsistent with 2*nnz convention ({chk})"
    return out


def main():
    m, file_to_id = load_matrix_table()
    raw = pd.concat([p5(m), p6(m, file_to_id), p7(m)], ignore_index=True, sort=False)
    raw["schedule"] = raw.schedule.fillna("")
    raw["throughput_gflops"] = gflops(raw.nnz, raw.kernel_ms)
    bad = raw[raw.error.abs() > ERR_TOL]
    print(f"rows: {len(raw)}   failing verification (|err|>{ERR_TOL}): {len(bad)}")
    raw = raw[~(raw.error.abs() > ERR_TOL)]
    raw[COMMON_COLS].to_csv(TABLES / "unified_raw.csv", index=False)
    # the exact 14-column merge format required by the course brief
    cf = raw.copy(); cf["size"] = cf.nnz; cf["throughput"] = cf.throughput_gflops
    cf["threads_or_ranks"] = [(str(int(t)) if isinstance(t, (int, float)) and t == t and float(t).is_integer() else str(c)) for t, c in zip(cf.threads_or_ranks, cf.config)]
    RAW = TABLES.parent / "raw"; RAW.mkdir(exist_ok=True)
    cf[["cluster", "problem", "platform", "implementation", "dataset", "size", "threads_or_ranks", "run_id", "total_ms", "kernel_ms",
        "transfer_ms", "throughput", "error", "power_w"]].to_csv(RAW / "unified_raw_common_format.csv", index=False)

    # ---- per-configuration summary (benchmark candidates only; CPU-seq(T7) kept for reference) ----
    keys = ["cluster", "problem", "platform", "candidate", "dataset", "nnz", "config"]
    g = raw.groupby(keys, dropna=False).agg(
        n_runs=("run_id", "size"),
        kernel_ms_median=("kernel_ms", "median"), kernel_ms_mean=("kernel_ms", "mean"),
        kernel_ms_std=("kernel_ms", "std"), total_ms_median=("total_ms", "median"),
        transfer_ms_median=("transfer_ms", "median"), max_error=("error", "max")).reset_index()
    g["gflops"] = gflops(g.nnz, g.kernel_ms_median)
    g["gflops_with_transfer"] = np.where(g.platform == "GPU", gflops(g.nnz, g.total_ms_median), np.nan)
    g["cv_pct"] = 100 * g.kernel_ms_std / g.kernel_ms_mean
    g.to_csv(TABLES / "unified_config_summary.csv", index=False)

    # ---- best configuration per (matrix, candidate) ----
    cand = g[g.candidate.isin(ALL)]
    idx = cand.groupby(["dataset", "candidate"]).gflops.idxmax()
    best = cand.loc[idx].sort_values(["dataset", "candidate"]).reset_index(drop=True)
    best["platform_group"] = best.candidate.map(PLATFORM_OF)
    best.to_csv(TABLES / "best_per_candidate.csv", index=False)

    perf = best.pivot(index="dataset", columns="candidate", values="gflops")[ALL]
    perf.index.name = "matrix_id"
    perf.round(4).to_csv(TABLES / "performance_matrix.csv")

    # ---- winners ----
    w = pd.DataFrame(index=perf.index)
    w["best_OpenMP"] = perf[OPENMP].idxmax(axis=1)
    w["best_MPI"] = perf[MPI].idxmax(axis=1)
    w["best_GPU_own"] = perf[GPU_OWN].idxmax(axis=1)
    w["best_GPU_incl_cuSPARSE"] = perf[GPU_OWN + ["GPU-cuSPARSE"]].idxmax(axis=1)
    w["best_overall_own"] = perf[OPENMP + MPI + GPU_OWN].idxmax(axis=1)
    w["best_overall_incl_cuSPARSE"] = perf.idxmax(axis=1)
    w["best_overall_gflops"] = perf.max(axis=1)
    w.to_csv(TABLES / "winners.csv")

    print("\nbest GFLOP/s per matrix x candidate (tuned):")
    print(perf.round(2).to_string())
    print("\nmissing (not benchmarked):", {c: list(perf.index[perf[c].isna()]) for c in ALL if perf[c].isna().any()})
    print("\nrun counts per config: min", g.n_runs.min(), " median", int(g.n_runs.median()), " max", g.n_runs.max())
    print("median run-to-run CV of kernel time: %.1f%%  (90th pct %.1f%%)" %
          (g.cv_pct.median(), g.cv_pct.quantile(.9)))
    print("\nwinners:\n", w.drop(columns="best_overall_gflops").to_string())


if __name__ == "__main__":
    main()
