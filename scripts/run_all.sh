#!/usr/bin/env bash
# Usage: bash run_all.sh /path/to/datasets/original   (the .mtx files; only needed for step 1,
# a cached results/tables/matrix_features.csv is included so you may skip it with SKIP_FEATURES=1)
set -e; cd "$(dirname "$0")"
[ -z "$SKIP_FEATURES" ] && python3 01_features.py --mtx-dir "$1"
python3 02_consolidate.py && python3 03_correlations_plots.py && python3 05_platform_reuse.py && python3 06_integrate_extras.py && python3 04_selector.py && python3 07_baseline_graphs.py
