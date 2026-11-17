#!/usr/bin/env python3
"""SpMV Suite: ONE program that builds and runs Team 5 (OpenMP), Team 6 (MPI) and Team 7 (CPU + CUDA) code,
collects every result in the common CSV schema, compares with the recorded results and applies the Team 8 selector.

  python suite.py doctor                         toolchains found + what will run live
  python suite.py run --data-dir DIR [--matrices appu,qpband] [--teams 5,6,7] [--ranks 1,2,4] [--full]
        DIR = Team 5's shared_with_cluster folder (needs datasets/original/*.mtx, datasets/vectors/*_x.txt)
        writes live_results/{p5,p6,p7}/*.csv, live_summary.csv, live_summary.json (load the JSON in the app's Live tab)
  python suite.py predict --mtx F.mtx [-R 10]    features + recommended format/platform for any matrix
  python suite.py all [--mtx-dir DIR]            rebuild every analysis table/plot from the recorded CSVs
  python suite.py hardware                       describe this machine (CPU, caches, memory, compilers, MPI, GPU)\n  python suite.py stream [--stream-size N]       official STREAM benchmark at several thread counts\n  python suite.py app                            open the dashboard"""
import argparse, importlib.util, json, os, platform, shutil, subprocess, sys, webbrowser
from pathlib import Path
R = Path(__file__).resolve().parent; S = R / "scripts"; I = R / "implementations"; EXE = ".exe" if os.name == "nt" else ""
H5 = "cluster,problem,platform,implementation,dataset,size,threads_or_ranks,run_id,total_ms,kernel_ms,transfer_ms,throughput,error,power_w,warmup,schedule,chunk,first_touch,proc_bind"
STATUS = {}

def sh(cmd, **k):
    cmd = [str(c) for c in cmd]; print("   $", " ".join(cmd)[:170])
    try: return subprocess.run(cmd, **k).returncode
    except (OSError, subprocess.TimeoutExpired) as e: print("   !!", e); return 99
def load(name):
    sys.path.insert(0, str(S)); sp = importlib.util.spec_from_file_location(name, S / f"{name}.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
def tc(): return {t: shutil.which(t) for t in ("gcc", "g++", "mpicxx", "mpiexec", "nvcc")} | {"mpiexec": shutil.which("mpiexec") or shutil.which("mpirun")}

def doctor():
    t = tc(); [print(f"  {k:8s}{v or 'NOT FOUND'}") for k, v in t.items()]
    print("\n  Team 5 OpenMP :", "runs live (gcc, needs -fopenmp)" if t["gcc"] else "needs gcc")
    print("  Team 6 MPI    :", "runs live" if t["mpicxx"] and t["mpiexec"] else "needs an MPI install (mpicxx + mpiexec)")
    print("  Team 7 CPU    :", "runs live" if t["g++"] else "needs g++")
    print("  Team 7 CUDA   :", "runs live (needs an NVIDIA GPU too)" if t["nvcc"] else "needs the CUDA toolkit (nvcc) + NVIDIA GPU")

def idmap(d):
    """matrix_id -> .mtx path (the ID and file name differ for e.g. soc_sign_slashdot)."""
    import pandas as pd; d = Path(d); f = d / "config/matrices.csv"
    if f.exists(): return {r.matrix_id: d / r.relative_path for r in pd.read_csv(f).itertuples()}
    return {p.stem: p for p in sorted((d / "datasets/original").glob("*.mtx")) if p.stem != "test_small"}

def paths(d, ids):
    d = Path(d); out = []; mm = idmap(d)
    for i in ids:
        m, v, r = mm.get(i, d / "datasets/original" / f"{i}.mtx"), d / "datasets/vectors" / f"{i}_x.txt", d / "reference_outputs" / f"{i}_y_reference.txt"
        if not (m.exists() and v.exists()): sys.exit(f"missing data for {i}: expected {m} and {v}")
        out.append((i, m, v, r if r.exists() else None))
    return out

def team5(ds, o, a):
    print("\n== Team 5: OpenMP (COO/CSR/ELL x schedules x threads)")
    exe = o / "bin" / f"p5_spmv{EXE}"; exe.parent.mkdir(parents=True, exist_ok=True)
    if sh([os.environ.get("CC", "gcc"), "-O3", "-std=c11", "-fopenmp", "-march=native", I / "p5_openmp/src/spmv.c", "-o", exe, "-lm"]):
        STATUS["Team 5 OpenMP"] = "BUILD FAILED (need gcc with OpenMP; on macOS: brew install gcc and set CC=gcc-14)"; return
    raw = o / "p5" / "team5_suitesparse_raw.csv"; raw.parent.mkdir(parents=True, exist_ok=True); raw.write_text(H5 + "\n")
    cfgf = Path(a.data_dir) / "config/experiment.json"; lim = json.loads(cfgf.read_text()).get("max_ell_padding_ratio", 20.0) if cfgf.exists() else 20.0
    feats = load("01_features"); n = os.cpu_count() or 2; thr = sorted({min(4, n), n} - {1}) or [1]; ok = bad = 0
    for i, m, v, _ in ds:
        pad = feats.features_for(m)["irregularity"]
        for fmt in ("coo", "csr", "ell"):
            if fmt == "ell" and pad > lim: print(f"   (skip ELL for {i}: padding ratio {pad:.1f} > Team 5 limit {lim})"); continue
            for mode, th, sch in [("seq", 1, "static")] + [("omp", t, s) for t in thr for s in ("static", "dynamic", "guided")]:
                rc = sh([exe, "--matrix", m, "--vector", v, "--dataset", i, "--format", fmt, "--mode", mode, "--schedule", sch, "--threads", th,
                         "--chunk", 16, "--runs", a.runs, "--repetitions", a.reps, "--raw", raw], stdout=subprocess.DEVNULL, timeout=300)
                ok, bad = ok + (rc == 0), bad + (rc != 0)
    if n == 1: print("   NOTE: this machine reports 1 CPU core; OpenMP scaling is not measurable here")
    STATUS["Team 5 OpenMP"] = f"ran {ok} configs ({bad} skipped/failed, e.g. ELL padding too large)"

def team6(ds, o, a):
    print("\n== Team 6: MPI (1D-rows, 1D-nnz, 2D)")
    t = tc()
    if not (t["mpicxx"] and t["mpiexec"]): STATUS["Team 6 MPI"] = "SKIPPED (no MPI: install OpenMPI/MPICH, or Microsoft MPI SDK on Windows)"; return
    exe = o / "bin" / f"p6_mpi{EXE}"; exe.parent.mkdir(parents=True, exist_ok=True)
    if sh(["mpicxx", "-O3", "-std=c++17", I / "p6_mpi/src/mpi_spmv.cpp", "-o", exe]): STATUS["Team 6 MPI"] = "BUILD FAILED"; return
    d = o / "p6"; d.mkdir(parents=True, exist_ok=True); [p.unlink() for p in d.glob("*.csv")]
    ver = subprocess.run([t["mpiexec"], "--version"], capture_output=True, text=True).stdout
    extra = ["--oversubscribe"] if ("Open MPI" in ver or "OpenRTE" in ver) else []
    env = os.environ | ({"OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"} if hasattr(os, "geteuid") and os.geteuid() == 0 else {})
    ok = bad = 0
    for i, m, v, _ in ds:
        for r in a.ranks:
            for sch, part in (("row", "rows"), ("row", "nnz"), ("2d", "rows")):
                rc = sh([t["mpiexec"], *extra, "-n", r, exe, "--matrix", m, "--vector", v, "--scheme", sch, "--partition", part, "--iterations", a.runs + 1,
                         "--warmup", 1, "--output", d / "spmv.csv", "--rank-output", d / "rank_metrics.csv"], env=env, stdout=subprocess.DEVNULL, timeout=300)
                ok, bad = ok + (rc == 0), bad + (rc != 0)
    STATUS["Team 6 MPI"] = f"ran {ok} configs ({bad} failed); each run self-validates against a serial reference"

def team7(ds, o, a):
    print("\n== Team 7: CPU baseline + CUDA kernels")
    t = tc(); d = o / "p7"; d.mkdir(parents=True, exist_ok=True); raw, summ = d / "team7_raw.csv", d / "summary.csv"; [p.unlink() for p in (raw, summ) if p.exists()]
    src = I / "p7_cuda/src"; cpu = o / "bin" / f"p7_cpu{EXE}"; cpu.parent.mkdir(parents=True, exist_ok=True)
    def runall(exe, extra=()):
        for i, m, v, r in ds:
            sh([exe, "--matrix", m, "--vector", v, *(["--reference", r] if r else []), "--dataset", i, *extra, "--runs", a.runs + 4, "--warmup", 2,
                "--raw", raw, "--summary", summ, "--quiet"], stdout=subprocess.DEVNULL, timeout=300)
    cpu_ok = t["g++"] and sh(["g++", "-O3", "-std=c++17", src / "cpu_spmv.cpp", *(src / "common").glob("*.cpp"), "-I", src / "common", "-o", cpu]) == 0
    if cpu_ok: runall(cpu)
    if not t["nvcc"]: STATUS["Team 7 CPU"] = "ran" if cpu_ok else "SKIPPED (no g++)"; STATUS["Team 7 CUDA"] = "SKIPPED (no nvcc / NVIDIA GPU)"; return
    STATUS["Team 7 CPU"] = "ran" if cpu_ok else "failed"
    if sh(["make", "-C", I / "p7_cuda", "cuda"]): STATUS["Team 7 CUDA"] = "BUILD FAILED (see make output; Makefile targets sm_75/80/89)"; return
    b = I / "p7_cuda/bin"
    for k, extra in [("scalar_csr", [("--block-size", s) for s in (128, 256, 512)]), ("vector_csr", [("--block-size", s) for s in (64, 128, 256, 512)]),
                     ("ell", [()]), ("hyb", [(), *[("--ell-width", w) for w in (8, 16, 32)]]), ("cusparse_spmv", [()])]:
        for e in extra: runall(b / k, e)
    STATUS["Team 7 CUDA"] = "ran (scalar/vector CSR, ELL, HYB, cuSPARSE)"

def summarize(ds, o):
    import numpy as np, pandas as pd
    f = load("01_features"); nnz = {i: f.features_for(m)["nnz"] for i, m, _, _ in ds}; rows = []
    def add(c, d, ms, cfg): rows.append(dict(dataset=d, candidate=c, kernel_ms=ms, config=cfg))
    p = o / "p5/team5_suitesparse_raw.csv"
    if p.exists() and p.stat().st_size > len(H5) + 2:
        r = pd.read_csv(p); r = r[r.warmup == 0]; r["c"] = "OMP-" + r.implementation.str.split("_").str[0].str.upper()
        for (c, d, th, sc, impl), g in r.groupby(["c", "dataset", "threads_or_ranks", "schedule", "implementation"]): add(c, d, g.kernel_ms.median(), f"thr{th}_{sc}")
    p = o / "p6/spmv.csv"
    if p.exists():
        r = pd.read_csv(p); s2i = {m.stem: i for i, m, _, _ in ds}; r["dataset"] = r.matrix.str.replace(".mtx", "", regex=False).map(lambda z: s2i.get(z, z))
        r["c"] = np.where(r.scheme == "2d", "MPI-2D", np.where(r.partition == "nnz", "MPI-1D-nnz", "MPI-1D-rows"))
        for (c, d, k), g in r.groupby(["c", "dataset", "ranks"]): add(c, d, g.elapsed_s.median() * 1e3, f"ranks{k}")
    p = o / "p7/team7_raw.csv"
    if p.exists():
        r = pd.read_csv(p); nm = {"scalar_csr": "GPU-scalar-CSR", "vector_csr": "GPU-vector-CSR", "ell": "GPU-ELL", "hyb": "GPU-HYB", "cusparse": "GPU-cuSPARSE", "cpu_sequential": "CPU-seq(T7)"}
        r["c"] = r.implementation.map(nm)
        for (c, d, cf), g in r.dropna(subset=["c"]).groupby(["c", "dataset", "threads_or_ranks"]): add(c, d, g.kernel_ms.median(), str(cf))
    df = pd.DataFrame(rows)
    if df.empty: print("no live results collected"); return
    df["gflops"] = [2 * nnz[d] / ms / 1e6 for d, ms in zip(df.dataset, df.kernel_ms)]
    best = df.loc[df.groupby(["dataset", "candidate"]).gflops.idxmax()].sort_values(["dataset", "candidate"])
    best[["dataset", "candidate", "gflops", "config"]].round(4).to_csv(R / "live_summary.csv", index=False)
    rec = pd.read_csv(R / "results/tables/performance_matrix.csv").set_index("matrix_id")
    js = {"machine": platform.platform(), "cpus": os.cpu_count(), "status": STATUS, "results": {}}
    print("\n== LIVE RESULTS (this machine) vs RECORDED (the teams' machines), best GFLOP/s per candidate")
    wide = best.pivot(index="dataset", columns="candidate", values="gflops")
    for d in wide.index:
        js["results"][d] = {c: round(v, 4) for c, v in wide.loc[d].dropna().items()}
        print(f"\n  {d}")
        for c, v in wide.loc[d].dropna().items():
            old = rec.loc[d, c] if c in rec.columns and d in rec.index and not np.isnan(rec.loc[d, c]) else None
            print(f"    {c:16s} live {v:7.3f}" + (f"   recorded {old:7.3f}   ratio {v / old:4.2f}" if old else ""))
    print("\n== SELECTOR vs LIVE MEASUREMENT")
    for i, m, _, _ in ds:
        ft = f.features_for(m); w = wide.loc[i].dropna() if i in wide.index else pd.Series(dtype=float)
        om = w[[c for c in w.index if c.startswith("OMP-")]]; gp = w[[c for c in w.index if c.startswith("GPU-") and "cuSPARSE" not in c]]
        pick_o = "OMP-ELL" if ft["irregularity"] <= 1.5 else "OMP-CSR"; pick_g = "GPU-vector-CSR" if ft["std_nnz_row"] > 30 else "GPU-HYB"
        s = f"  {i:20s} OpenMP: rule {pick_o}, measured best {om.idxmax() if len(om) else 'n/a'}"
        s += f" | GPU: rule {pick_g}, measured best {gp.idxmax()}" if len(gp) else " | GPU: not run"; print(s)
    (R / "live_summary.json").write_text(json.dumps(js)); print(f"\nwrote {R/'live_summary.csv'} and {R/'live_summary.json'} (load the JSON in the app's Live tab)")


def hardware():
    """Describe THIS machine (common-baseline requirement): CPU, cores, caches, memory, compilers, MPI, GPU."""
    run = lambda c: (lambda r: (r.stdout + r.stderr).strip())(subprocess.run(c, capture_output=True, text=True, timeout=30)) if shutil.which(c[0]) else "not found"
    info = {"os": platform.platform(), "python": platform.python_version(), "logical_cpus": os.cpu_count()}
    ls = run(["lscpu"]); 
    for k in ("Model name", "Core(s) per socket", "Thread(s) per core", "Socket(s)", "CPU max MHz", "L1d cache", "L2 cache", "L3 cache"):
        for line in ls.splitlines():
            if line.strip().startswith(k + ":"): info[k] = line.split(":", 1)[1].strip()
    try: info["memory_GB"] = round(int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1]) / 1048576, 1)
    except Exception: pass
    info["gcc"] = run(["gcc", "--version"]).splitlines()[0] if shutil.which("gcc") else "not found"
    info["mpicxx"] = run(["mpicxx", "--version"]).splitlines()[0] if shutil.which("mpicxx") else "not found"
    info["mpi"] = (run(["ompi_info", "--version"]).splitlines() or ["not found"])[0] if shutil.which("ompi_info") else "not found"
    info["gpu"] = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]) if shutil.which("nvidia-smi") else "no NVIDIA GPU / nvidia-smi not found"
    info["nvcc"] = run(["nvcc", "--version"]).splitlines()[-1] if shutil.which("nvcc") else "not found"
    out = R / "results"; out.mkdir(exist_ok=True); (out / "hardware_team8_machine.json").write_text(json.dumps(info, indent=2))
    md = "# Hardware and software of the machine that produced the live runs\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in info.items()) + "\n"
    (out / "hardware_team8_machine.md").write_text(md); print(md)

def stream(a):
    """Official McCalpin STREAM at several thread counts -> results/tables/stream_team8_machine.csv"""
    import re, csv
    exe = R / "live_results/bin" / f"stream{EXE}"; exe.parent.mkdir(parents=True, exist_ok=True)
    if sh(["gcc", "-O3", "-fopenmp", f"-DSTREAM_ARRAY_SIZE={a.stream_size}", "-DNTIMES=10", I / "stream/stream.c", "-o", exe]): sys.exit("STREAM build failed (needs gcc with OpenMP)")
    n = os.cpu_count() or 1; rows = []
    for th in sorted({1, max(1, n // 2), n}):
        o = subprocess.run([str(exe)], capture_output=True, text=True, env=os.environ | {"OMP_NUM_THREADS": str(th)}).stdout
        r = {k: float(re.search(rf"^{k}:\s+([\d.]+)", o, re.M).group(1)) for k in ("Copy", "Scale", "Add", "Triad")}; r["threads"] = th; rows.append(r)
        print(f"  threads {th:3d}: " + "  ".join(f"{k} {v:9.1f}" for k, v in r.items() if k != "threads") + "  MB/s")
    out = R / "results/tables"; out.mkdir(parents=True, exist_ok=True)
    with open(out / "stream_team8_machine.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threads", "Copy", "Scale", "Add", "Triad"]); w.writeheader(); w.writerows(rows)
    best = max(r["Triad"] for r in rows); print(f"\nHighest Triad bandwidth: {best/1000:.2f} GB/s (array = {a.stream_size:,} doubles; must exceed 4x the last-level cache)")

def run(a):
    ids = a.matrices.split(","); ds = paths(a.data_dir, ids); o = R / "live_results"; o.mkdir(exist_ok=True)
    teams = set(a.teams.split(","))
    for t, fn in (("5", team5), ("6", team6), ("7", team7)):
        if t in teams: fn(ds, o, a)
    print("\n== STATUS"); [print(f"  {k:15s}{v}") for k, v in STATUS.items()]; summarize(ds, o)

def predict(mtx, Rn):
    f = load("01_features").features_for(mtx); print({k: (round(float(v), 3) if not isinstance(v, str) else v) for k, v in f.items()})
    print("GPU kernel     :", "GPU-vector-CSR" if f["std_nnz_row"] > 30 else "GPU-HYB", "(rule std nnz/row > 30; LOO 13/13)")
    print("OpenMP format  :", "OMP-ELL" if f["irregularity"] <= 1.5 else "OMP-CSR"); print("MPI partition  : MPI-1D-nnz (low confidence)")
    n = f["nnz"] / 1e6; print(f"Platform @R={Rn}:", "GPU" if .3 + 2.8 * n + Rn * (.07 * n + .01) < Rn * (.8 * n + .01) else "OpenMP", "(rough timing estimate)")

def main():
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["doctor", "run", "predict", "all", "app", "hardware", "stream"]); a.add_argument("--mtx"); a.add_argument("--mtx-dir")
    a.add_argument("--data-dir"); a.add_argument("--matrices", default="qpband,wy2010"); a.add_argument("--teams", default="5,6,7"); a.add_argument("-R", type=int, default=10)
    a.add_argument("--ranks", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4]); a.add_argument("--full", action="store_true"); a.add_argument("--stream-size", type=int, default=20000000); x = a.parse_args()
    x.runs, x.reps = (6, 100) if x.full else (4, 30)
    if x.cmd == "doctor": doctor()
    elif x.cmd == "run":
        if not x.data_dir: sys.exit("run needs --data-dir <Team 5 shared_with_cluster folder>")
        run(x)
    elif x.cmd == "hardware": hardware()
    elif x.cmd == "stream": stream(x)
    elif x.cmd == "predict": predict(x.mtx, x.R)
    elif x.cmd == "all":
        if x.mtx_dir: sh([sys.executable, S / "01_features.py", "--mtx-dir", x.mtx_dir])
        for s in ("02_consolidate", "03_correlations_plots", "05_platform_reuse", "06_integrate_extras", "04_selector", "07_baseline_graphs"): sh([sys.executable, S / f"{s}.py"], cwd=S)
    else: webbrowser.open((R / "app/spmv_suite.html").as_uri())
if __name__ == "__main__": main()
