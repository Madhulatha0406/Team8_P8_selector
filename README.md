# Team 8 — Problem 8: Format Selection and Cross-Platform Synthesis

Inputs: Team 5 (OpenMP CPU), Team 6 (MPI), Team 7 (CUDA) raw CSVs, copied to `inputs/`.
All three teams used byte-identical SuiteSparse matrices (same SHA-256). Benchmark set = 13 matrices
(Team 7's 4x4 `test_small` fixture is dropped).

## Common-baseline audit (course brief)
| Requirement | Status | Where |
|---|---|---|
| Describe hardware (CPU, cores, caches, memory, GPU, compilers, MPI, CUDA) | Command ready: **run `python3 suite.py hardware` on your laptop** and keep `results/hardware_team8_machine.md`. Team 5's machine: `inputs/p5/hardware_info.txt` (i5-1235U, 10 cores / 12 threads, 12 MB L3, MSYS2 gcc). Team 6/7 machines are undocumented in their data. | `suite.py` |
| STREAM bandwidth | Official McCalpin `stream.c` vendored; **run `python3 suite.py stream`** on your laptop. Team 5's result (12.79 GB/s Triad at 6 threads) is used in the Roofline. | `implementations/stream/` |
| Estimate peak GFLOP/s and draw a Roofline | Done for Team 5's machine (peak estimate 208 GFLOP/s = 10 cores x 1.3 GHz reported x 16 DP FLOP/cycle; assumption, it does not change the result because SpMV sits far left of the ridge). Not drawn for MPI/GPU machines: no STREAM exists for them. | `results/plots/roofline.png` |
| Verified optimised sequential baseline (-O3) | Team 5's sequential COO/CSR/ELL and Team 7's CPU baseline are the baselines (validated, max error 2.4e-6 and 0). Team 8 has no SpMV kernel of its own except the GUI's Python engine. | `speedup_efficiency.png` |
| >= 5 runs, warm-up ignored, mean and std | Every configuration has at least 5 timed runs; medians are used for scoring and std is stored (`kernel_ms_std`, `cv_pct`). | `results/tables/unified_config_summary.csv` |
| >= 3 problem sizes | 13 sizes, nnz from 45,000 to 1,995,041 | `matrix_features.csv` |
| Error bars on graphs | Added to the feature plot, time-vs-size (run-to-run std) and speedup/efficiency (std across matrices). Heatmap, winner map and accuracy bars have none (not measurements with spread). | `results/plots/` |
| CSV with the 14 common columns | Done; `size` = nnz, `throughput` = GFLOP/s | `results/raw/unified_raw_common_format.csv` |
| Standard graphs that apply | Time vs size: `time_vs_size.png`. Speedup vs threads/ranks with ideal line and parallel efficiency: `speedup_efficiency.png`. Roofline: `roofline.png`. Stacked compute/communication/transfer: `time_breakdown.png`. Tuning parameter: `selector_tuning.png` (tree depth). | `results/plots/` |
| Not done | Weak scaling (no team ran a growing problem); synchronisation time (not recorded by any team); STREAM/hardware of the Team 6 and 7 machines. | |
| Repo layout (src/, scripts/, results/raw/, results/plots/, report/, LOG.md) | All present (`src/README.md` maps to `scripts/`, `suite.py`, `gui.py`, `implementations/`); `LOG.md` entries are undated: add your dates. | |

## Desktop GUI (`python gui.py`)
A tkinter window: pick the data folder and matrices, press **Run**, and watch the execution log stream live; then see a results table, a GFLOP/s chart
(bars = this run, diamonds = the teams' recorded results) and the selector's verdict. The GUI also contains the four Problem 8 tasks, each reading the real result files and each with a button that re-runs that step:
**1 Merged table** (task 1: the 13 x 11 table of the three teams' results with the winner per platform; button re-runs `02_consolidate.py`),
**2 Features** (task 2: rows, nnz, mean/std/max nnz per row, bandwidth, irregularity; button recomputes them from the matrix files with `01_features.py`),
**3 Plots & correlations** (task 3: the feature-vs-performance plots, relative-performance plots, Spearman heatmap and winner map shown in the window, plus a table of the strongest correlations; button re-runs `03_correlations_plots.py`),
**4 Selector accuracy** (task 4: the accuracy table with a per-task filter, the accuracy bars, the decision trees and accuracy-vs-depth plot; button re-runs `04_selector.py`).
Screenshots: `results/plots/gui_tab_merged_table.png`, `gui_tab_selector.png`.
Two modes:
* **Python engine** (works on any PC with Python, no compiler/MPI): really runs COO, CSR and ELL SpMV (numpy/scipy), validates each against Team 5's reference output,
  times it (median of 9 runs) and compares with the recorded OpenMP result of the same format. This is a Python implementation, *not* the teams' C/MPI/CUDA code.
* **Teams' real code:** launches `suite.py run` and streams its output (needs gcc / MPI / nvcc as in the run guide).
Windows: tkinter ships with the python.org installer; Linux: `sudo apt install python3-tk`. Tested headless on Linux (virtual display, see `results/plots/gui_screenshot.png`); not tested on a real Windows desktop.
Bug fixed along the way: matrix IDs and file names differ for `soc_sign_slashdot` (file `soc-sign-Slashdot090221.mtx`); `suite.py` and the GUI now map IDs to files via `config/matrices.csv`.

## Running Team 7's CUDA kernels without your own NVIDIA GPU (Google Colab)
`suite.py` builds and runs Team 7's CUDA kernels automatically whenever it finds `nvcc` and a GPU. Use a free Colab GPU:
1. Rename the files on Google Drive to `Team8_P8_selector.zip` and `shared_with_cluster.zip` (no `(2)`) and upload both to My Drive.
2. colab.research.google.com -> New notebook -> Runtime -> Change runtime type -> **T4 GPU** -> Save.
3. Run these cells:
```
!nvidia-smi
from google.colab import drive; drive.mount('/content/drive')
!cp /content/drive/MyDrive/Team8_P8_selector.zip /content/drive/MyDrive/shared_with_cluster.zip /content/
!unzip -q /content/Team8_P8_selector.zip -d /content && unzip -q /content/shared_with_cluster.zip -d /content
import os; os.environ["PATH"] += ":/usr/local/cuda/bin"
%cd /content/team8
!python3 suite.py doctor
!python3 suite.py run --data-dir /content/shared_with_cluster --matrices qpband,wy2010,appu --teams 7
from google.colab import files; files.download('/content/team8/live_summary.json')
```
4. Rename the downloaded file (e.g. `gpu_summary.json`). In the dashboard's **Live** tab select it **together with** your laptop's `live_summary.json`: they are merged.
Not tested by us on a real GPU (none available): if the build fails, the CUDA flags are in `implementations/p7_cuda/Makefile` (sm_75 is the Colab T4). Kaggle notebooks (free GPU) work the same way.

## Run the combined program (one program runs Teams 5, 6, 7)
`python suite.py doctor`: which toolchains you have.
`python suite.py run --data-dir <Team 5 shared_with_cluster folder> --matrices appu,qpband --ranks 1,2,4 [--teams 5,6,7] [--full]`:
builds and runs Team 5 (OpenMP), Team 6 (MPI) and Team 7 (CPU + CUDA) code, collects their output, compares with the recorded results, applies the selector,
and writes `live_summary.csv/json` (load the JSON in the app's **Live** tab). Other commands: `predict --mtx F.mtx`, `all` (rebuild analysis), `app`.

Needs: gcc with OpenMP (Team 5; macOS: `brew install gcc`, `CC=gcc-14`), an MPI install with `mpicxx`/`mpiexec` (Team 6; Windows: Microsoft MPI SDK), g++ (Team 7 CPU),
CUDA toolkit + NVIDIA GPU (Team 7 CUDA). Missing toolchains are skipped with a clear status line, not an error.

**Tested here (Linux sandbox, 1 CPU core):** Team 5 (after a portable-timer patch), Team 6 with OpenMPI 4.1.6 (18/18 configs pass the code's own validation), Team 7 CPU baseline.
Example output: `examples/sandbox_live_run/`. **Not tested:** Team 7 CUDA path (no GPU/nvcc here; the runner mirrors Team 7's own `run_benchmarks.sh` flags exactly) and Windows/MS-MPI.
A 1-core machine cannot show OpenMP/MPI scaling and is noisy, so live-vs-recorded ratios there are not findings; on your laptop they will be meaningful.

## Run the pipeline step by step
`bash scripts/run_all.sh /path/to/datasets/original` (the `.mtx` files are only needed for step 01;
`SKIP_FEATURES=1` uses the cached `results/tables/matrix_features.csv`). Needs pandas, scipy, scikit-learn, matplotlib.

| Step | Script | Output |
|---|---|---|
| 1 Collect results | `02_consolidate.py` | `unified_raw.csv` (4,675 runs, common CSV schema), `unified_config_summary.csv`, `performance_matrix.csv`, `winners.csv` |
| 2 Features | `01_features.py` | `matrix_features.csv` (rows, nnz, mean/std/max nnz per row, bandwidth + derived; verified against Team 5 metadata) |
| 3 Performance vs features | `03_correlations_plots.py` | `feature_vs_performance.png`, `correlation_heatmap.png`, `relative_perf_vs_features.png`, `winner_map.png`, `correlations_long.csv` |
| 4 Selectors | `04_selector.py` | `selector_accuracy.csv/.png`, `selector_predictions.csv`, `selector_depth_sensitivity.csv`, `selector_single_feature_stumps.csv`, `selector_trees.png/.txt` |
| 5 Integrate P5/P6/P7 secondary outputs | `06_integrate_extras.py` | `integrated_mpi_comm.csv`, `integrated_comm_vs_winner_8ranks.csv`, `integrated_effective_bandwidth.csv`, `integrated_numa_pinning.csv`, `integration_extras.png` |
| Extra: platform with transfer cost | `05_platform_reuse.py`, `reuse.py` | `platform_reuse_*.csv`, `platform_reuse.png` |

## Method
GFLOP/s = 2·nnz / median kernel time. Each candidate (OMP-COO/CSR/ELL, MPI-1D-rows/1D-nnz/2D, GPU scalar-CSR/vector-CSR/ELL/HYB/cuSPARSE)
is scored at its **best tuned configuration** (threads x schedule x chunk; ranks; block size / ELL width). The "label" of a matrix is the
candidate with the highest GFLOP/s. Selectors are evaluated by **leave-one-out** (LOO). Metrics: top-1 accuracy, "within 5%" (reaches >=95% of best,
5% being run-to-run noise) and mean % of best performance delivered.

## Results (13 matrices)
Best candidate counts: OpenMP CSR 12 / ELL 1 (qpband). MPI 1D-nnz 6 / 2D 4 / 1D-rows 3. GPU own kernels HYB 7 / vector-CSR 6;
with cuSPARSE: cuSPARSE 7 / HYB 4 / vector-CSR 2. At kernel time the GPU is the best platform on 13/13.

**Correlations (Spearman, n=13).** Vector-CSR GFLOP/s vs mean nnz/row rho=+0.995. Relative to the best GPU kernel, vector-CSR rises with
std nnz/row (+0.90) while HYB (-0.88), scalar-CSR (-0.82) and ELL vs irregularity (-0.95) fall. ~220 tests were run, so some significant cells are chance.

**Selector accuracy, top-1 % (LOO unless noted)**

| Task | Majority | Tree-6 | Tree-10 | Hand rule* |
|---|---|---|---|---|
| OpenMP format (CSR/ELL) | 92 | 92 | 92 | 100 |
| MPI partition | 46 | 31 | 46 | 46 |
| GPU kernel, own | 54 | 92 | **100** | 77 |
| GPU kernel, + cuSPARSE | 54 | 62 | 85 | 54 |
| Platform OpenMP vs MPI | 92 | 92 | 92 | 100 |
| Platform, 3-way (kernel time) | 100 | 100 | 100 | 100 |
| Platform OpenMP vs GPU incl. transfer, R=5 | 77 | 69 | 62 | 92 |
| Platform OpenMP vs GPU incl. transfer, R=10 | 92 | 92 | 92 | 77 |

*Hand rules were set after seeing the results: treat as in-sample. Tree-6 = six assignment features; Tree-10 adds irregularity, CV, relative bandwidth, density.
"Overall format+platform" equals the GPU rows because the GPU always wins at kernel time.

**What is actually learnable**
* **GPU kernel:** one split, `std nnz/row > 30 -> vector-CSR, else HYB`, separates all 13 (gap between 23.6 and 36.5). Tree-6's 92% comes from one fold where `nnz`
  ties with `std` as a perfect split; this is a tie-break artefact, not a different rule. The textbook rule (mean nnz/row >= 32) gets 77% and fails on TSOPF
  (HYB 2.9 vs vector-CSR 13.6 GFLOP/s: a few huge rows make HYB's COO overflow slow).
* **With cuSPARSE** the choice is not reliably predictable (62-85% depending on depth/features; the depth-2 tree's `rows <= 25185` leaf is 2 matrices).
* **MPI partition:** not predictable. Trees reach 100% in-sample but 31-46% LOO; the schemes are mostly within ~5-15% of each other. Defaulting to 1D equal-nnz costs little.
* **OpenMP format / OpenMP vs MPI:** only one minority case each (qpband), so nothing can be validated; "CSR, and MPI only pays off above ~1e5 nnz" is an observation, not a tested rule.
* **Platform with data movement (extra).** Including Team 7's PCIe transfer, the GPU needs N* = 3.4-27 reuses of the matrix to beat OpenMP (median 4.2; smaller matrices need more,
  Spearman(nnz, N*) = -0.76). Winners by R: GPU wins 0/13 at R=1-2, 10/13 at R=5, 12/13 at R=10-20, 13/13 at R>=50. Table N* uses the kernel-fastest GPU kernel; the winners
  table picks the best total time per R, so the two can differ slightly. No tree beats always-guessing the majority under LOO here.

## What was integrated from Teams 5, 6, 7
**Code** (`implementations/`, see `MANIFEST.md`): all three teams' sources are vendored (3,542 lines C/C++/CUDA). Only Team 7's portable CPU baseline could be
built here (no MPI/nvcc; Team 5's `spmv.c` needs `windows.h`). It passes against **Team 5's reference outputs with max error 0** on appu and qpband.
The same code ran 1.8-2.3x faster on this sandbox than on Team 7's machine: a direct measure of the machine effect in cross-platform comparisons.
The MPI, CUDA and Windows-OpenMP code were **not re-executed**; their numbers enter via their CSVs.

**Data used:** P5 timing CSV, P6 `spmv.csv` + `communication_summary.csv`, P7 `team7_raw.csv`, P5 STREAM/roofline config, first-touch, pinning, imbalance tables.
* **P6 communication (new):** at 8 ranks the partition with the least bytes sent is also the fastest on only **4/13** matrices (about chance for 3 schemes), so volume alone does not
  pick the winner; load balance matters. Exposed (non-overlapped) communication is 28% of wall time for 1D-rows, 35% for 1D-nnz and **83% for 2D** (median, 8 ranks).
* **P5 STREAM roofline:** with a compulsory-traffic model (12 B/nnz CSR + vectors), OMP-CSR "effective bandwidth" is 104-140% of STREAM (12.79 GB/s) on the 12 larger matrices and
  236% on qpband. Above 100% means the matrices are partly cache-resident, so a STREAM-based roofline is not a strict ceiling here. The traffic model is ours, not Team 5's.
* **P5 first-touch / pinning:** first-touch read speedup is 0.75-1.33x with no trend (synthetic array, single-socket laptop, so no NUMA effect expected). Pinning changes time by
  -4% to +11% against 2-7% run noise, on 2 matrices only: no demonstrable effect.
* **P5 static imbalance:** at 16 threads imbalance reaches 4.8x and tracks irregularity (Spearman 0.71).

**Not integrable:** Team 7's Nsight profiling (divergence, coalescing, achieved bandwidth) was collected on the 4x4 `test_small` matrix only, so nothing exists for the 13 matrices.
Team 6's reported per-rank breakdown is used only through `comm_exposed_s`; Team 5's ELL results are missing for 4 matrices (padding limit) and Team 7's for TSOPF.

## Caveats
* **n = 13.** Small differences in LOO accuracy are one or two matrices. In-sample tree accuracy is always 100% and means nothing.
* **Different machines.** P5 = Windows laptop (STREAM Triad 12.8 GB/s); P6 hardware undocumented; P7 = NVIDIA GPU (repo's hardware_info.json claims 320 GB/s, 254 GFLOP/s peak,
  but its hardware_info.txt says nvidia-smi was unavailable on the machine it was collected on). Compare trends, not absolute levels; OpenMP-vs-MPI results largely reflect the machines.
* **Noise.** Median run-to-run CV of kernel time is 5.8% (90th percentile 19%); P6 has 5 samples per configuration.
* **Tuning bias.** "Best of configurations" favours candidates with more tuning options (HYB had the most). Missing: OMP-ELL on 4 matrices (padding limit), GPU-ELL on TSOPF.
* `qpband` is labelled "banded" but its bandwidth is 15,000 of 20,000, so bandwidth is a poor regularity proxy here.
* The transfer model assumes the matrix stays on the GPU after the first copy; MPI is excluded because its distribution cost was not measured.
