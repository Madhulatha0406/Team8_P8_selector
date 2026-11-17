# Team 6: Distributed Sparse Matrix-Vector Multiplication

## Objective and approach

This study implements double-precision `y = A*x` using the SuiteSparse Matrix Market files in `datasets/original`. The 1D implementation assigns contiguous rows by equal-row cuts or by cumulative nonzero counts. Each rank identifies distinct off-rank vector indices, exchanges index requests once during setup, and exchanges only the requested vector values during each timed iteration using nonblocking point-to-point MPI. In-flight values overlap the local-nonzero multiply; remote nonzeros are accumulated after the receive completes. The 2D implementation uses a near-square Cartesian-like process grid, distributes matrix blocks by row and column, broadcasts each column-owned vector segment down its process-column communicator, exchanges remote vector values within process rows, and sums partial output rows within each process row.

All runs use the vector files supplied with the dataset and validate the final output against a serial reference loop on rank 0 before reporting success. Timings are reported as the maximum rank time; communication bytes count per-iteration MPI payloads for remote-vector exchange and 2D output reduction (sent/received separately in `rank_metrics.csv`). One-time matrix/vector distribution is excluded. The compute/communication figure uses max-rank phase timings; exposed communication includes synchronization and non-overlapped work, not MPI library internal counters.

## Dataset and measured results

The analyzed run covers 13 matrices, with row-irregularity `max(nnz per row)/mean(nnz per row)` ranging from qpband (1.33) to TSOPF_FS_b39_c19 (1465.43). The largest matrix by nnz is mixtank_new (1,995,041 entries).

Across the recorded configurations, the highest observed throughput is 6.4898 GFLOP/s for Goodwin_071 using row/rows on 8 rank(s). The highest speedup over the matching one-rank configuration is 2.81x (Goodwin_071, row/rows, 8 ranks). Median speedup over all measured multi-rank configurations is 1.25x. These are measurements from this run, not hardware-independent performance claims.

Aggregate payload sent across the recorded rank/configuration samples is 127,961,472 bytes. Use `communication_summary.csv` for per-rank totals and `analysis_summary.csv` for individual measured configurations. Compare 1D row-balanced and nnz-balanced layouts to see the effect of row-length skew; the 2D layout exchanges column-block vector data and reduces partial row outputs, so its extra communication must be weighed against its parallel work distribution.

## Figures and interpretation

- `gflops_scaling.png`: GFLOP/s against rank count for 1D equal-row, 1D equal-nnz, and 2D layouts.
- `speedup_vs_irregularity.png`: measured speedup against row-length irregularity; equal-nnz cuts are expected to help most when long rows make equal-row partitions imbalanced, while communication and row indivisibility can limit that gain.
- `compute_communication_breakdown.png`: max-rank computation and exposed communication at the largest measured rank count.
- `communication_volume.png`: aggregate point-to-point payload sent by layout. The raw rank CSV also separates send and receive payloads per rank.

The experiment runner runs all 13 metadata-listed matrices at 1, 2, 4, 8, and 16 ranks by default; every configuration is serial-reference validated. For repeatability, record the MPI implementation/version, compiler flags, host CPU/node count, and MPI binding options alongside these generated CSVs when running on the target cluster. MPI is launched without vendor-specific affinity flags by default.

## Reproducibility

Build `src/mpi_spmv.cpp` with an MPI C++ compiler, then run `python scripts/run_experiments.py --executable PATH_TO_BINARY`. The runner creates `spmv.csv` (raw per-iteration timings) and `rank_metrics.csv` (per-rank bytes and phase timings). Run `python scripts/analyze_results.py` to generate the summary tables, plots, and this report.
