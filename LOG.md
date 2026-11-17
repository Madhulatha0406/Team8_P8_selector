# LOG.md (Team 8, Problem 8)
Entries are listed in the order the work was done. **Add your own dates** before submission; the sequence is reconstructed from the project history.

1. Read the Cluster B briefs; listed what Teams 5, 6, 7 must deliver and what Team 8 must deliver (4 tasks + 2 submit items).
2. Collected the teams' outputs: Team 5 (OpenMP CSV, STREAM, first-touch, pinning, imbalance), Team 6 (MPI CSV, communication volumes), Team 7 (CUDA CSV, profiling, notebook).
3. Checked the matrices are byte-identical across the three packages (SHA-256).
4. Task 2: wrote `01_features.py`; recomputed rows, nnz, mean/std/max nnz per row, bandwidth from the .mtx files; asserted equality with Team 5's metadata.
5. Task 1: wrote `02_consolidate.py`; one schema, 11 candidate names, warm-up and test runs dropped, error check, GFLOP/s recomputed; 4,675 runs; best tuned configuration per candidate.
6. Task 3: wrote `03_correlations_plots.py`; feature-vs-performance plots, Spearman heatmap, relative-performance plots, winner map.
7. Task 4: wrote `04_selector.py`; majority baseline, depth-2 trees on 6 and 10 features, hand rules; leave-one-out; found tree tie-break instability and added single-feature stump and depth-sensitivity tables.
8. Added the transfer-cost (reuse) analysis `05_platform_reuse.py` and the extra data integration `06_integrate_extras.py` (Team 6 communication, STREAM effective bandwidth, first-touch/pinning).
9. Vendored the three teams' source into `implementations/`; wrote `suite.py` to build and run it; patched Team 5's timer for Linux; fixed the matrix ID vs file-name bug (soc_sign_slashdot) and OpenMPI detection.
10. Ran Teams 5, 6, 7 (CPU) live in WSL/Ubuntu on the team laptop (e.g. ns3Da: Team 5 21 configs, Team 6 9 configs, Team 7 CPU ran; CUDA skipped, no NVIDIA GPU).
11. Built the web dashboard and the desktop GUI (Python engine, Teams' real code, Selector verdict, Team requirements).
12. Audited against the common baseline: added `hardware` and `stream` commands (official McCalpin STREAM), `07_baseline_graphs.py` (speedup/efficiency, Roofline, time vs size, time breakdown, selector tuning), error bars on the feature plot, the exact 14-column CSV in `results/raw/`.
