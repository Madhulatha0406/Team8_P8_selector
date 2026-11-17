#!/usr/bin/env python3
"""Run Team 6's 1-D and 2-D MPI SpMV experiments from the dataset manifest."""

import argparse
import csv
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
METADATA = ROOT / "datasets" / "metadata" / "matrix_metadata.csv"


def parse_int_list(value):
    try:
        values = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item < 1 for item in values):
        raise argparse.ArgumentTypeError("values must all be positive")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=pathlib.Path,
                        default=ROOT / "build" / "mpi_spmv.exe")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--ranks", type=parse_int_list, default=[1, 2, 4, 8, 16])
    parser.add_argument("--matrices", help="comma-separated matrix IDs; default is all")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "results")
    parser.add_argument("--append", action="store_true",
                        help="append to existing CSVs instead of starting fresh")
    args = parser.parse_args()

    executable = args.executable if args.executable.is_absolute() else ROOT / args.executable
    if not executable.is_file():
        parser.error(f"MPI executable not found: {executable}")
    if args.iterations < 1 or args.warmup < 0:
        parser.error("--iterations must be positive and --warmup nonnegative")

    with METADATA.open(newline="", encoding="utf-8") as source:
        matrices = list(csv.DictReader(source))
    if args.matrices:
        requested = {name.strip() for name in args.matrices.split(",") if name.strip()}
        unknown = requested - {row["matrix_id"] for row in matrices}
        if unknown:
            parser.error("unknown matrix IDs: " + ", ".join(sorted(unknown)))
        matrices = [row for row in matrices if row["matrix_id"] in requested]
    if not matrices:
        parser.error("no matrices selected")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "spmv.csv"
    rank_csv = output_dir / "rank_metrics.csv"
    if not args.append:
        for path in (output_csv, rank_csv):
            if path.exists():
                path.unlink()

    runs = []
    for matrix in matrices:
        for rank_count in args.ranks:
            runs.extend((matrix, rank_count, scheme, partition)
                        for scheme, partition in (
                            ("row", "rows"), ("row", "nnz"), ("2d", "rows")))

    for index, (matrix, rank_count, scheme, partition) in enumerate(runs, 1):
        command = [
            args.mpiexec, "-n", str(rank_count), str(executable),
            "--matrix", str(ROOT / matrix["file"]),
            "--vector", str(ROOT / matrix["vector_file"]),
            "--scheme", scheme, "--partition", partition,
            "--iterations", str(args.iterations), "--warmup", str(args.warmup),
            "--output", str(output_csv), "--rank-output", str(rank_csv),
        ]
        print(f"[{index}/{len(runs)}] {matrix['matrix_id']} ranks={rank_count} "
              f"scheme={scheme} partition={partition}", flush=True)
        try:
            result = subprocess.run(
                command, cwd=ROOT, check=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True,
            )
        except subprocess.CalledProcessError as error:
            print(error.stdout, file=sys.stderr)
            raise
        if "Validation: PASS" not in result.stdout:
            print(result.stdout, file=sys.stderr)
            raise RuntimeError("run did not report successful reference validation")
        print("  reference validation passed", flush=True)

    print(f"Raw results: {output_csv}")
    print(f"Per-rank communication/timing: {rank_csv}")
    print("Next: python scripts/analyze_results.py")


if __name__ == "__main__":
    main()
