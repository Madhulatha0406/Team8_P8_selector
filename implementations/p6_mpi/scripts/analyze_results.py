#!/usr/bin/env python3
"""Summarize MPI SpMV CSVs, plot scaling/communication, and write the report."""

import argparse
import csv
import pathlib
import statistics
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parents[1]
METADATA = ROOT / "datasets" / "metadata" / "matrix_metadata.csv"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def matrix_id_from_filename(filename, matrices_by_file):
    return matrices_by_file.get(pathlib.Path(filename).name, pathlib.Path(filename).stem)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=pathlib.Path, default=ROOT / "results")
    args = parser.parse_args()
    result_dir = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    raw_path, rank_path = result_dir / "spmv.csv", result_dir / "rank_metrics.csv"
    if not raw_path.is_file() or not rank_path.is_file():
        parser.error(f"expected {raw_path} and {rank_path}; run the experiments first")

    matrices = read_csv(METADATA)
    matrices_by_file = {pathlib.Path(row["file"]).name: row["matrix_id"] for row in matrices}
    features = {row["matrix_id"]: row for row in matrices}
    grouped = defaultdict(list)
    for row in read_csv(raw_path):
        key = (matrix_id_from_filename(row["matrix"], matrices_by_file),
               row["scheme"], row["partition"], int(row["ranks"]))
        grouped[key].append(row)

    summary = []
    for (matrix, scheme, partition, ranks), samples in sorted(grouped.items()):
        elapsed = statistics.mean(float(row["elapsed_s"]) for row in samples)
        local = statistics.mean(float(row["local_compute_s"]) for row in samples)
        remote = statistics.mean(float(row["remote_compute_s"]) for row in samples)
        exposed = statistics.mean(float(row["comm_exposed_s"]) for row in samples)
        nnz = int(features[matrix]["nnz"])
        summary.append({
            "matrix_id": matrix, "scheme": scheme, "partition": partition,
            "ranks": ranks, "samples": len(samples), "nnz": nnz,
            "irregularity": features[matrix]["irregularity"],
            "elapsed_s": elapsed, "local_compute_s": local,
            "remote_compute_s": remote, "comm_exposed_s": exposed,
            "gflops": 2.0 * nnz / elapsed / 1e9 if elapsed > 0 else 0.0,
        })

    baselines = {}
    for row in summary:
        if row["ranks"] == 1:
            baselines[(row["matrix_id"], row["scheme"], row["partition"])] = row["elapsed_s"]
    for row in summary:
        baseline = baselines.get((row["matrix_id"], row["scheme"], row["partition"]))
        row["speedup"] = baseline / row["elapsed_s"] if baseline and row["elapsed_s"] else 0.0
    summary_path = result_dir / "analysis_summary.csv"
    write_csv(summary_path, summary, list(summary[0]))

    rank_grouped = defaultdict(list)
    for row in read_csv(rank_path):
        key = (matrix_id_from_filename(row["matrix"], matrices_by_file),
               row["scheme"], row["partition"], int(row["ranks"]), int(row["rank"]))
        rank_grouped[key].append(row)
    rank_stats = defaultdict(lambda: {"sent": 0, "received": 0, "compute": 0.0,
                                      "remote": 0.0, "exposed": 0.0, "samples": 0})
    for (matrix, scheme, partition, ranks, _rank), samples in rank_grouped.items():
        aggregate = rank_stats[(matrix, scheme, partition, ranks)]
        aggregate["sent"] += statistics.mean(int(row["sent_bytes"]) for row in samples)
        aggregate["received"] += statistics.mean(int(row["received_bytes"]) for row in samples)
        aggregate["compute"] = max(
            aggregate["compute"],
            statistics.mean(float(row["local_compute_s"]) for row in samples))
        aggregate["remote"] = max(
            aggregate["remote"],
            statistics.mean(float(row["remote_compute_s"]) for row in samples))
        aggregate["exposed"] = max(
            aggregate["exposed"],
            statistics.mean(float(row["comm_exposed_s"]) for row in samples))
        aggregate["samples"] = len(samples)
    communication = []
    for (matrix, scheme, partition, ranks), values in sorted(rank_stats.items()):
        communication.append({
            "matrix_id": matrix, "scheme": scheme, "partition": partition,
            "ranks": ranks, "total_bytes_sent": int(values["sent"]),
            "total_bytes_received": int(values["received"]),
            "max_rank_bytes_sent": max(
                (int(statistics.mean(int(row["sent_bytes"]) for row in samples))
                 for (m, s, p, r, _), samples in rank_grouped.items()
                 if (m, s, p, r) == (matrix, scheme, partition, ranks)), default=0),
        })
    if communication:
        write_csv(result_dir / "communication_summary.csv", communication,
                  list(communication[0]))

    generate_plots(result_dir, summary, communication)
    generate_report(result_dir, summary, communication, features)
    print(f"Summary: {summary_path}")
    print(f"Figures and report: {result_dir}")


def generate_plots(result_dir, summary, communication):
    matrices = sorted({row["matrix_id"] for row in summary})
    fig, axes = plt.subplots(4, 4, figsize=(18, 14), sharex=True)
    for ax, matrix in zip(axes.flat, matrices):
        for scheme, partition, label in (
            ("row", "rows", "1D rows"), ("row", "nnz", "1D nnz"),
            ("2d", "rows", "2D"),
        ):
            data = sorted((row for row in summary if row["matrix_id"] == matrix
                           and row["scheme"] == scheme and row["partition"] == partition),
                          key=lambda row: row["ranks"])
            if data:
                ax.plot([row["ranks"] for row in data],
                        [row["gflops"] for row in data], marker="o", label=label)
        ax.set_title(matrix)
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
    for ax in axes[-1, :]:
        ax.set_xlabel("MPI ranks")
    for ax in axes[:, 0]:
        ax.set_ylabel("GFLOP/s")
    axes.flat[0].legend(fontsize=8)
    for ax in axes.flat[len(matrices):]:
        ax.set_visible(False)
    fig.suptitle("Distributed SpMV scaling by matrix and partition")
    fig.tight_layout()
    fig.savefig(result_dir / "gflops_scaling.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    styles = {("row", "rows"): "o", ("row", "nnz"): "s", ("2d", "rows"): "^"}
    for (scheme, partition), marker in styles.items():
        data = [row for row in summary if row["ranks"] > 1 and
                row["scheme"] == scheme and row["partition"] == partition]
        if data:
            ax.scatter([float(row["irregularity"]) for row in data],
                       [row["speedup"] for row in data], marker=marker, alpha=0.72,
                       label=f"{scheme}/{partition}")
    ax.set_xscale("log")
    ax.set_xlabel("Maximum row nnz / mean row nnz")
    ax.set_ylabel("Speedup over same scheme/partition at 1 rank")
    ax.set_title("MPI speedup versus matrix row irregularity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_dir / "speedup_vs_irregularity.png", dpi=160)
    plt.close(fig)

    selected = [row for row in summary if row["ranks"] == max(
        item["ranks"] for item in summary)]
    selected.sort(key=lambda row: (row["matrix_id"], row["scheme"], row["partition"]))
    labels = [f"{row['matrix_id']}\n{row['scheme']}/{row['partition']}"
              for row in selected]
    compute = [row["local_compute_s"] + row["remote_compute_s"] for row in selected]
    comm = [row["comm_exposed_s"] for row in selected]
    positions = range(len(selected))
    fig, ax = plt.subplots(figsize=(max(11, len(labels) * 0.55), 6))
    ax.bar(positions, compute, label="local + remote SpMV compute")
    ax.bar(positions, comm, bottom=compute, label="exposed communication/other")
    ax.set_xticks(list(positions), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Max-rank phase time (s)")
    ax.set_title(f"Computation and exposed communication ({max(row['ranks'] for row in summary)} ranks)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(result_dir / "compute_communication_breakdown.png", dpi=160)
    plt.close(fig)

    if communication:
        fig, ax = plt.subplots(figsize=(10, 6))
        data = [row for row in communication if row["ranks"] > 1]
        names = sorted({row["matrix_id"] for row in data})
        schemes = sorted({(row["scheme"], row["partition"]) for row in data})
        width = 0.8 / max(1, len(schemes))
        base_positions = list(range(len(names)))
        for index, scheme in enumerate(schemes):
            values = [next((row["total_bytes_sent"] for row in data
                            if row["matrix_id"] == name
                            and (row["scheme"], row["partition"]) == scheme), 0)
                      for name in names]
            ax.bar([x + index * width for x in base_positions], values,
                   width=width, label="/".join(scheme))
        ax.set_xticks([x + width * (len(schemes) - 1) / 2 for x in base_positions],
                      names, rotation=65, ha="right")
        ax.set_ylabel("Aggregate bytes sent by all ranks")
        ax.set_title("Communication volume by matrix and partition")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(result_dir / "communication_volume.png", dpi=160)
        plt.close(fig)


def generate_report(result_dir, summary, communication, features):
    measured_ids = {row["matrix_id"] for row in summary}
    measured_features = [features[matrix] for matrix in measured_ids]
    most_irregular = max(measured_features, key=lambda row: float(row["irregularity"]))
    most_regular = min(measured_features, key=lambda row: float(row["irregularity"]))
    largest = max(measured_features, key=lambda row: int(row["nnz"]))
    best = max(summary, key=lambda row: row["gflops"])
    scalable = [row for row in summary if row["ranks"] > 1]
    max_ranks = max(row["ranks"] for row in summary)
    best_scaling = max(scalable, key=lambda row: row["speedup"]) if scalable else best
    median_speedup = statistics.median(row["speedup"] for row in scalable) if scalable else 0.0
    communication_total = sum(row["total_bytes_sent"] for row in communication)
    lines = [
        "# Team 6: Distributed Sparse Matrix-Vector Multiplication",
        "",
        "## Objective and approach",
        "",
        "This study implements double-precision `y = A*x` using the SuiteSparse Matrix Market "
        "files in `datasets/original`. The 1D implementation assigns contiguous rows by "
        "equal-row cuts or by cumulative nonzero counts. Each rank identifies distinct "
        "off-rank vector indices, exchanges index requests once during setup, and exchanges "
        "only the requested vector values during each timed iteration using nonblocking "
        "point-to-point MPI. In-flight values overlap the local-nonzero multiply; remote "
        "nonzeros are accumulated after the receive completes. The 2D implementation uses a "
        "near-square Cartesian-like process grid, distributes matrix blocks by row and column, "
        "broadcasts each column-owned vector segment down its process-column communicator, "
        "exchanges remote vector values within process rows, and sums partial output rows "
        "within each process row.",
        "",
        "All runs use the vector files supplied with the dataset and validate the final output "
        "against a serial reference loop on rank 0 before reporting success. Timings are "
        "reported as the maximum rank time; communication bytes count per-iteration MPI "
        "payloads for remote-vector exchange and 2D output reduction (sent/received separately "
        "in `rank_metrics.csv`). One-time matrix/vector distribution is excluded. The "
        "compute/communication figure uses max-rank phase timings; exposed communication "
        "includes synchronization and non-overlapped work, not MPI library internal counters.",
        "",
        "## Dataset and measured results",
        "",
        f"The analyzed run covers {len(measured_ids)} matrices, with row-irregularity "
        "`max(nnz per row)/mean(nnz per row)` ranging from "
        f"{most_regular['matrix_id']} ({float(most_regular['irregularity']):.2f}) to "
        f"{most_irregular['matrix_id']} ({float(most_irregular['irregularity']):.2f}). "
        f"The largest matrix by nnz is {largest['matrix_id']} "
        f"({int(largest['nnz']):,} entries).",
        "",
        f"Across the recorded configurations, the highest observed throughput is "
        f"{best['gflops']:.4f} GFLOP/s for {best['matrix_id']} using "
        f"{best['scheme']}/{best['partition']} on {best['ranks']} rank(s). "
        f"The highest speedup over the matching one-rank configuration is "
        f"{best_scaling['speedup']:.2f}x ({best_scaling['matrix_id']}, "
        f"{best_scaling['scheme']}/{best_scaling['partition']}, "
        f"{best_scaling['ranks']} ranks). Median speedup over all measured multi-rank "
        f"configurations is {median_speedup:.2f}x. These are measurements from this run, "
        "not hardware-independent performance claims.",
        "",
        "Aggregate payload sent across the recorded rank/configuration samples is "
        f"{communication_total:,} bytes. Use `communication_summary.csv` for per-rank "
        "totals and `analysis_summary.csv` for individual measured configurations. Compare "
        "1D row-balanced and nnz-balanced layouts to see the effect of row-length skew; the "
        "2D layout exchanges column-block vector data and reduces partial row outputs, so its "
        "extra communication must be weighed against its parallel work distribution.",
        "",
        "## Figures and interpretation",
        "",
        "- `gflops_scaling.png`: GFLOP/s against rank count for 1D equal-row, 1D equal-nnz, "
        "and 2D layouts.",
        "- `speedup_vs_irregularity.png`: measured speedup against row-length irregularity; "
        "equal-nnz cuts are expected to help most when long rows make equal-row partitions "
        "imbalanced, while communication and row indivisibility can limit that gain.",
        "- `compute_communication_breakdown.png`: max-rank computation and exposed "
        "communication at the largest measured rank count.",
        "- `communication_volume.png`: aggregate point-to-point payload sent by layout. "
        "The raw rank CSV also separates send and receive payloads per rank.",
        "",
        "The experiment runner runs all 13 metadata-listed matrices at 1, 2, 4, 8, and 16 "
        "ranks by default; every configuration is serial-reference validated. For repeatability, "
        "record the MPI implementation/version, compiler flags, host CPU/node count, and MPI "
        "binding options alongside these generated CSVs when running on the target cluster. "
        "MPI is launched without vendor-specific affinity flags by default.",
        "",
        "## Reproducibility",
        "",
        "Build `src/mpi_spmv.cpp` with an MPI C++ compiler, then run "
        "`python scripts/run_experiments.py --executable PATH_TO_BINARY`. The runner creates "
        "`spmv.csv` (raw per-iteration timings) and `rank_metrics.csv` (per-rank bytes and "
        "phase timings). Run `python scripts/analyze_results.py` to generate the summary "
        "tables, plots, and this report.",
        "",
    ]
    (result_dir / "team6_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
