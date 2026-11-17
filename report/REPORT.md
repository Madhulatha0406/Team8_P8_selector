# Team 8 report section: Format selection and cross-platform synthesis (Problem 8)

**Goal.** Teams 5 (OpenMP), 6 (MPI) and 7 (CUDA) measured SpMV on the same 13 SuiteSparse matrices. We merged their results, described each matrix by simple features, looked for relationships between features and performance, and built selectors that predict the best format and platform.

**Data.** All three teams used byte-identical matrix files (SHA-256 checked). We mapped each team's CSV to one schema (`results/raw/unified_raw_common_format.csv`, 4,675 validated runs, 14 columns) and 11 candidates: OpenMP COO/CSR/ELL; MPI 1D-rows, 1D-nnz, 2D; GPU scalar-CSR, vector-CSR, ELL, HYB, cuSPARSE. Warm-up and test runs were dropped, GFLOP/s was recomputed as 2*nnz/time (agrees with the teams' own columns within 1%), the median over repeats was taken, and each candidate is scored at its best tuned configuration (threads x schedule, ranks, block size / ELL width). Run-to-run noise is about 6% (median CV), so gaps under 5% are treated as ties.

**Features** (`matrix_features.csv`): rows, nnz, mean / std / max nnz per row, bandwidth, plus irregularity (max/mean), CV, relative bandwidth, density. They match Team 5's metadata exactly.

**Correlations** (`feature_vs_performance.png`, `correlation_heatmap.png`). Vector-CSR throughput rises with mean row length (Spearman rho = 0.995); relative to the best GPU kernel, vector-CSR rises with the spread of row length while HYB, scalar-CSR and ELL fall. About 220 correlations were tested, so some significant cells are expected by chance.

**Selector** (`selector_accuracy.csv`, leave-one-out over 13 matrices):
| Task | Majority | Tree (6 features) | Tree (10 features) | Hand rule |
|---|---|---|---|---|
| GPU kernel (own) | 54% | 92% | 100% | 77% |
| GPU kernel (+cuSPARSE) | 54% | 62% | 85% | 54% |
| MPI partition | 46% | 31% | 46% | 46% |
| OpenMP format | 92% | 92% | 92% | 100% |
The GPU rule `std of nnz/row > 30 -> vector-CSR, else HYB` is correct on all 13 matrices. The MPI partition is not predictable from these features. The OpenMP result equals always choosing CSR (only `qpband` prefers ELL), so that rule cannot be validated.

**Platform.** At kernel time the GPU is fastest on all 13 matrices. Counting the host-to-device copy, it needs 3.4 to 27 repeated multiplications to beat OpenMP (median about 4). The MPI scheme with the least communication is also the fastest on only 4 of 13 matrices. SpMV on Team 5's machine lies on the bandwidth roof of the Roofline (`roofline.png`), i.e. it is memory-bound, and scaling with threads or ranks is poor (`speedup_efficiency.png`).

**Limitations.** Only 13 matrices; the three platforms ran on different machines, so trends are compared, not absolute speed; the hand rules were set after seeing the data; Team 7's profiling covers only a 4x4 test matrix and its CUDA kernels were not re-run by us; synchronisation time and weak scaling were not measured by any team.
