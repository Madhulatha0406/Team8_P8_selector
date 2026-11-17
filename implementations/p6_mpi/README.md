# Team 6 — Distributed MPI SpMV

This project evaluates double-precision `y = A*x` using the 13 SuiteSparse
matrices listed in [`datasets/metadata/matrix_metadata.csv`](datasets/metadata/matrix_metadata.csv).
The small `test_small.mtx` file is a correctness fixture, not part of the benchmark
manifest.

## Implemented variants

- **1D equal rows**: contiguous, approximately equal row counts per rank.
- **1D equal nnz**: contiguous row ranges cut using cumulative nonzero counts.
- **2D block**: a near-square process grid; matrix entries are distributed by
  row and column blocks, vector segments are distributed by column and
  broadcast down process-column communicators, remote vector entries are
  exchanged among process-row peers, and partial output rows are summed over
  each process row.
- **Communication/computation overlap**: nonblocking point-to-point vector
  exchanges are posted before local-nonzero computation; remote-nonzero work
  starts after those receives complete.
- **Reference validation**: the final result is compared on rank 0 with a
  serial coordinate-format SpMV loop for every invocation.

Each timed iteration records the maximum elapsed rank time, the local and
remote multiply phases, exposed communication/other time, and bytes sent and
received by each rank. Byte counters cover iterative remote-vector exchange
and 2D output reduction; one-time matrix/vector distribution is excluded.
They count MPI payload data, not protocol headers or network-link traffic.
Matrix Market real, integer, and pattern coordinate matrices are supported;
symmetric, skew-symmetric, and Hermitian entries are expanded to the full
matrix.

## Build

Use an MPI compiler wrapper on Linux/macOS:

```sh
mkdir -p build
mpicxx -O3 -std=c++17 src/mpi_spmv.cpp -o build/mpi_spmv
```

On Windows, install the Microsoft MPI SDK and build with a compiler configured
to use its headers and libraries, or build/run in an MPI-enabled Linux
environment (for example, the course cluster). With MinGW g++ and the default
Microsoft MPI SDK install location:

```powershell
New-Item -ItemType Directory -Force build
g++ -O3 -std=c++17 `
  -I "C:\Program Files (x86)\Microsoft SDKs\MPI\Include" `
  src\mpi_spmv.cpp `
  "C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64\msmpi.lib" `
  -o build\mpi_spmv.exe
```

Ensure `mpiexec` is on `PATH`.

## Run one correctness/performance case

```sh
mpiexec -n 4 ./build/mpi_spmv \
  --matrix datasets/original/av41092.mtx \
  --vector datasets/vectors/av41092_x.txt \
  --scheme row --partition nnz --iterations 5 --warmup 1 \
  --output results/spmv.csv --rank-output results/rank_metrics.csv
```

Use `--scheme 2d --partition rows` for the 2D layout. The `--partition` option
for this scheme is fixed to `rows`. Outputs append, so use fresh CSV files for a
new experiment or let the runner start a clean run.

## Full experiment and analysis

The runner executes both 1D partition variants and 2D for each matrix at 1, 2,
4, 8, and 16 ranks, with one warmup and five measured iterations by default:

```sh
python scripts/run_experiments.py --executable build/mpi_spmv
python scripts/analyze_results.py
```

Select a subset or adjust the rank counts/iterations when testing:

```sh
python scripts/run_experiments.py --executable build/mpi_spmv \
  --matrices av41092,TSOPF_FS_b39_c19 --ranks 1,2,4 --iterations 3
```

The raw iteration CSV is `results/spmv.csv`; per-rank communication and phase
data are in `results/rank_metrics.csv`. Analysis creates
`analysis_summary.csv`, `communication_summary.csv`, scaling and communication
plots, and `team6_report.md` from measured results. The report deliberately
does not claim timings until experiments have actually run on the target
cluster. For a reproducible cluster report, also record the host CPU/node
configuration, compiler and flags, MPI implementation/version, and any
cluster-specific rank-binding settings. No affinity flags are forced by the
runner.
