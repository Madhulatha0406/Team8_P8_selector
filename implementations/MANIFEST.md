# Vendored implementations (Teams 5, 6, 7) — what is here and what was verified

| Team | Code | Build (from their docs) | Re-run in Team 8's sandbox? |
|---|---|---|---|
| P5 OpenMP | `p5_openmp/src/spmv.c`, `first_touch.c`, `scripts/*.py` | `gcc -O3 -std=c11 -fopenmp -march=native src/spmv.c` (**Windows-only**: includes `windows.h`) | **Yes** after a 10-line portable-timer patch (Windows path kept under `#ifdef _WIN32`) |
| P6 MPI | `p6_mpi/src/mpi_spmv.cpp`, `scripts/` | `mpicxx -O3 -std=c++17 src/mpi_spmv.cpp` | **Yes** (OpenMPI 4.1.6; 1/2/4 ranks, all 3 schemes pass validation) |
| P7 CUDA | `p7_cuda/src/*.cu`, `common/`, `Makefile`, notebook | `make` (nvcc, sm_75/80/89) | CUDA kernels: not tested (no GPU/nvcc); runner uses the same flags as Team 7 run_benchmarks.sh. **CPU baseline `cpu_spmv.cpp`: yes** |

Verification performed (g++ 13-class compiler, Linux sandbox):
`g++ -O3 -std=c++17 src/cpu_spmv.cpp src/common/*.cpp -Isrc/common` then run on Team 5's shared vectors and
reference outputs:

| matrix | result vs Team 5 reference | GFLOP/s here | GFLOP/s Team 7 recorded |
|---|---|---|---|
| appu | PASSED, max_abs_error = 0 | 2.89 | 1.57 |
| qpband | PASSED, max_abs_error = 0 | 3.78 | 1.65 |

Meaning: Team 7's code and Team 5's reference data agree exactly, and the same code differs ~1.8-2.3x between
machines — the size of the machine effect that confounds cross-platform comparison in this project.

Not verifiable: Team 7's Nsight profiling (`results/profiling/*.csv`) was collected on the 4x4 `test_small` matrix only,
so no divergence/coalescing/achieved-bandwidth data exists for the 13 benchmark matrices.
