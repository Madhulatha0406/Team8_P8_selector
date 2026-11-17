#!/usr/bin/env python3
"""Compute per-matrix structural features directly from the Matrix Market files.

Features (assignment): rows, nnz, mean / std / max nnz per row, bandwidth.
Extras used by the selector: irregularity (max/mean, the Cluster-B definition),
CV (std/mean), relative bandwidth, density.

Symmetric / skew-symmetric / hermitian files are expanded to the full matrix
(same convention as Teams 5-7), so nnz here equals the nnz they benchmarked.
The result is cross-checked against Team 5's matrix_metadata.csv.

Usage: python 01_features.py --mtx-dir PATH/TO/datasets/original
"""
import argparse
import numpy as np
import pandas as pd
from common import INPUTS, TABLES


def read_mtx_pattern(path):
    """Return (rows, cols, i, j, symmetry) - indices are 0-based; values are ignored."""
    with open(path, "r") as f:
        header = f.readline().lower().split()
        symmetry = header[4] if len(header) >= 5 else "general"
        skip = 1
        line = f.readline()
        while line.startswith("%"):
            skip += 1
            line = f.readline()
        rows, cols, nnz_file = (int(t) for t in line.split()[:3])
        skip += 1
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip, usecols=[0, 1],
                     dtype=np.int64, engine="c")
    assert len(df) == nnz_file, f"{path}: expected {nnz_file} entries, got {len(df)}"
    return rows, cols, df[0].to_numpy() - 1, df[1].to_numpy() - 1, symmetry


def features_for(path):
    rows, cols, i, j, sym = read_mtx_pattern(path)
    if sym != "general":                      # mirror strictly off-diagonal entries
        off = i != j
        i, j = np.concatenate([i, j[off]]), np.concatenate([j, i[off]])
    per_row = np.bincount(i, minlength=rows)
    nnz = int(per_row.sum())
    mean = nnz / rows
    return dict(rows=rows, cols=cols, nnz=nnz, mean_nnz_row=mean,
                std_nnz_row=float(per_row.std()),          # population std (matches Team 5)
                max_nnz_row=int(per_row.max()), min_nnz_row=int(per_row.min()),
                empty_rows=int((per_row == 0).sum()),
                bandwidth=int(np.abs(i - j).max()),
                irregularity=per_row.max() / mean,
                cv_nnz_row=float(per_row.std()) / mean,
                rel_bandwidth=int(np.abs(i - j).max()) / rows,
                density=nnz / (rows * cols), symmetry=sym)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mtx-dir", required=True,
                    help="directory holding the .mtx files (datasets/original)")
    args = ap.parse_args()
    from pathlib import Path
    mtx_dir = Path(args.mtx_dir)

    manifest = pd.read_csv(INPUTS / "matrices.csv")
    meta = pd.read_csv(INPUTS / "matrix_metadata.csv").set_index("matrix_id")
    out = []
    for _, m in manifest.iterrows():
        f = mtx_dir / Path(m.relative_path).name
        feat = features_for(f)
        feat.update(matrix_id=m.matrix_id, category=m.category, suitesparse_name=m["name"])
        # cross-check against Team 5 metadata
        t5 = meta.loc[m.matrix_id]
        for a, b in [("rows", "rows"), ("nnz", "nnz"), ("max_nnz_row", "max_nnz_per_row")]:
            assert feat[a] == t5[b], (m.matrix_id, a, feat[a], t5[b])
        assert abs(feat["mean_nnz_row"] - t5.mean_nnz_per_row) < 1e-9
        assert abs(feat["std_nnz_row"] - t5.std_nnz_per_row) < 1e-6, m.matrix_id
        assert abs(feat["irregularity"] - t5.irregularity) < 1e-9
        out.append(feat)
        print(f"{m.matrix_id:20s} rows={feat['rows']:>7} nnz={feat['nnz']:>8} "
              f"bw={feat['bandwidth']:>6} irr={feat['irregularity']:8.2f}  [matches Team 5 metadata]")
    cols = ["matrix_id", "category", "suitesparse_name", "rows", "cols", "nnz", "mean_nnz_row",
            "std_nnz_row", "max_nnz_row", "min_nnz_row", "empty_rows", "bandwidth",
            "irregularity", "cv_nnz_row", "rel_bandwidth", "density", "symmetry"]
    df = pd.DataFrame(out)[cols].sort_values("nnz").reset_index(drop=True)
    df.to_csv(TABLES / "matrix_features.csv", index=False)
    print(f"\nwrote {TABLES / 'matrix_features.csv'}  ({len(df)} matrices)")


if __name__ == "__main__":
    main()
