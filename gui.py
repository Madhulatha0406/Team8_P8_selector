#!/usr/bin/env python3
"""SpMV Suite GUI (tkinter; no extra install beyond pandas/numpy/scipy/matplotlib).
Pick matrices, press Run, watch the execution log stream live, see GFLOP/s results, a chart vs the recorded results, and the selector's verdict.
Mode 1 'Python engine': really executes COO / CSR / ELL SpMV (numpy/scipy) and validates against Team 5's reference output. Works on any machine.
Mode 2 'Teams real code': launches `suite.py run` (Team 5 OpenMP, Team 6 MPI, Team 7 CPU/CUDA) and streams it; needs gcc / MPI / nvcc where available."""
import importlib.util, json, os, platform, queue, shutil, webbrowser, subprocess, sys, threading, time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.image as mpimg
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
R = Path(__file__).resolve().parent; sys.path.insert(0, str(R)); import suite
COL = {"PY": "#6a51a3", "OMP": "#0072B2", "MPI": "#E69F00", "GPU": "#009E73", "CPU": "#777777"}
ELL_LIMIT = 8.0   # engine memory guard (Team 5 used 20 for its C code)

def engine_run(i, mtx, vec, ref, q, stopped):
    from scipy.io import mmread
    log = lambda s: q.put(("log", s))
    log(f"--- {i}: loading {mtx.name} ..."); t0 = time.time(); A = mmread(str(mtx)).tocoo(); rows, cols = A.shape; nnz = A.nnz
    x = np.loadtxt(vec); y_ref = np.loadtxt(ref) if ref else None; csr = A.tocsr(); ri, ci, v = A.row, A.col, A.data
    per = np.diff(csr.indptr); mean, std, mx = nnz / rows, float(per.std()), int(per.max()); irr = mx / mean; bw = int(np.abs(ri - ci).max())
    log(f"    {rows} x {cols}, nnz={nnz}, loaded in {time.time()-t0:.1f}s | mean {mean:.1f}  std {std:.1f}  max {mx}  irregularity {irr:.1f}  bandwidth {bw}")
    K = {"PY-COO": lambda: np.bincount(ri, weights=v * x[ci], minlength=rows), "PY-CSR": lambda: csr.dot(x)}
    if mx * rows / nnz <= ELL_LIMIT:
        pos = np.arange(nnz) - np.repeat(csr.indptr[:-1], per); rr = np.repeat(np.arange(rows), per)
        ec = np.zeros((rows, mx), np.int32); ev = np.zeros((rows, mx)); ec[rr, pos] = csr.indices; ev[rr, pos] = csr.data
        K["PY-ELL"] = lambda: (ev * x[ec]).sum(axis=1)
    else: log(f"    skip ELL: padding ratio {irr:.1f} > {ELL_LIMIT} (same idea as Team 5's padding limit)")
    for name, f in K.items():
        if stopped(): return
        log(f"    running {name}: 1 warm-up + 9 timed repetitions"); y = f(); ts = []
        for _ in range(9): t = time.perf_counter(); y = f(); ts.append(time.perf_counter() - t)
        ms = float(np.median(ts)) * 1e3; err = float(np.abs(y - y_ref).max()) if y_ref is not None else float(np.abs(y - csr.dot(x)).max())
        ok = err < 1e-6; gf = 2 * nnz / (ms * 1e-3) / 1e9
        log(f"      {name}: {ms:.3f} ms  {gf:.3f} GFLOP/s  max|err|={err:.1e}  {'VALID' if ok else 'MISMATCH'}")
        q.put(("row", dict(matrix=i, cand=name, gflops=gf, ms=ms, check="valid" if ok else "MISMATCH")))
    q.put(("sel", dict(matrix=i, std=std, irr=irr, mean=mean, nnz=nnz)))



class ImgView:
    """Shows a PNG inside the window (matplotlib's Tk canvas; no extra imaging package needed)."""
    def __init__(s, parent):
        s.fig = Figure(figsize=(8, 4.5), dpi=100); s.ax = s.fig.add_axes([0, 0, 1, 1]); s.ax.axis("off"); s.path = None
        s.cv = FigureCanvasTkAgg(s.fig, parent); s.cv.get_tk_widget().pack(fill="both", expand=True)
    def show(s, path):
        s.path = Path(path); s.ax.clear(); s.ax.axis("off")
        if s.path.exists(): s.ax.imshow(mpimg.imread(str(s.path)))
        else: s.ax.text(.5, .5, f"{s.path.name} not found: use the Rebuild button", ha="center")
        s.cv.draw_idle()

ALLC = ["OMP-COO", "OMP-CSR", "OMP-ELL", "MPI-1D-rows", "MPI-1D-nnz", "MPI-2D", "GPU-scalar-CSR", "GPU-vector-CSR", "GPU-ELL", "GPU-HYB", "GPU-cuSPARSE"]
PLOTS3 = {"Performance vs each feature (6 assignment features)": "feature_vs_performance.png", "Relative performance vs the key features": "relative_perf_vs_features.png",
          "Correlation heatmap (Spearman)": "correlation_heatmap.png", "Winner per matrix": "winner_map.png"}
PLOTS4 = {"Selector accuracy": "selector_accuracy.png", "Decision trees": "selector_trees.png", "Accuracy vs tree depth": "selector_tuning.png"}

def table(parent, cols, widths, height=10):
    fr = ttk.Frame(parent); tv = ttk.Treeview(fr, columns=cols, show="headings", height=height)
    ys = ttk.Scrollbar(fr, orient="vertical", command=tv.yview); xs = ttk.Scrollbar(fr, orient="horizontal", command=tv.xview); tv.configure(yscroll=ys.set, xscroll=xs.set)
    for c, w in zip(cols, widths): tv.heading(c, text=c); tv.column(c, width=w, minwidth=w, anchor="w" if c in (cols[0], "task", "method", "rule") else "center")
    ys.pack(side="right", fill="y"); xs.pack(side="bottom", fill="x"); tv.pack(side="left", fill="both", expand=True); return fr, tv

def fill(tv, rows):
    tv.delete(*tv.get_children())
    for i, r in enumerate(rows): tv.insert("", "end", values=r, tags=("odd",) if i % 2 else ())
    tv.tag_configure("odd", background="#f1f3f5")

class App:
    def __init__(s, root):
        s.root = root; root.title("SpMV Suite: live runner"); root.geometry("1320x800"); s.q = queue.Queue(); s.rows = []; s.proc = None; s.halt = False; s.sel = {}
        p = R / "results/tables/performance_matrix.csv"; s.rec = pd.read_csv(p).set_index("matrix_id") if p.exists() else None
        L = ttk.Frame(root, padding=8); L.pack(side="left", fill="y")
        s.dd = tk.StringVar(); fr = ttk.Frame(L); fr.pack(fill="x")
        ttk.Entry(fr, textvariable=s.dd, width=34).pack(side="left"); ttk.Button(fr, text="Browse", command=s.browse).pack(side="left", padx=3)
        for c in (R.parent / "shared_with_cluster", Path.home() / "shared_with_cluster", R / "shared_with_cluster"):
            if (c / "datasets/original").exists(): s.dd.set(str(c)); break
        ttk.Button(L, text="Load matrix list", command=s.load).pack(anchor="w", pady=4); s.lb = tk.Listbox(L, selectmode="extended", height=13, exportselection=False); s.lb.pack(fill="x")
        s.mode = tk.StringVar(value="engine"); ttk.Label(L, text="Mode", font=("", 10, "bold")).pack(anchor="w", pady=(8, 0))
        ttk.Radiobutton(L, text="Python engine (real execution, any PC)", variable=s.mode, value="engine").pack(anchor="w")
        ttk.Radiobutton(L, text="Teams' real code (suite.py run)", variable=s.mode, value="teams").pack(anchor="w")
        s.tv = {t: tk.BooleanVar(value=True) for t in "567"}; tf = ttk.Frame(L); tf.pack(anchor="w", padx=18)
        for t, n in (("5", "T5 OpenMP"), ("6", "T6 MPI"), ("7", "T7 CPU/CUDA")): ttk.Checkbutton(tf, text=n, variable=s.tv[t]).pack(side="left")
        rf = ttk.Frame(L); rf.pack(anchor="w", padx=18, pady=2); ttk.Label(rf, text="MPI ranks").pack(side="left"); s.rk = tk.StringVar(value="1,2,4"); ttk.Entry(rf, textvariable=s.rk, width=10).pack(side="left", padx=4)
        bf = ttk.Frame(L); bf.pack(fill="x", pady=8); s.run_b = ttk.Button(bf, text="▶ Run", command=s.start); s.run_b.pack(side="left"); ttk.Button(bf, text="■ Stop", command=s.stop).pack(side="left", padx=6)
        s.pb = ttk.Progressbar(L, mode="indeterminate"); s.pb.pack(fill="x"); s.st = tk.StringVar(value="Idle"); ttk.Label(L, textvariable=s.st, wraplength=290).pack(anchor="w", pady=6)
        s.nb = ttk.Notebook(root); s.nb.pack(side="right", fill="both", expand=True, padx=6, pady=6)
        s.log = scrolledtext.ScrolledText(s.nb, font=("Consolas", 9), bg="#101418", fg="#d7e0ea"); s.nb.add(s.log, text=" Execution log ")
        rf = ttk.Frame(s.nb); s.nb.add(rf, text=" Results "); cols = ("matrix", "candidate", "GFLOP/s", "ms", "check")
        s.tree = ttk.Treeview(rf, columns=cols, show="headings", height=8)
        for c in cols: s.tree.heading(c, text=c); s.tree.column(c, width=110)
        s.tree.pack(fill="x"); cf = ttk.Frame(rf); cf.pack(anchor="w", pady=4); ttk.Label(cf, text="Chart for matrix:").pack(side="left"); s.cm = ttk.Combobox(cf, state="readonly", width=24); s.cm.pack(side="left", padx=4); s.cm.bind("<<ComboboxSelected>>", lambda e: s.draw())
        s.fig = Figure(figsize=(7, 3.6), dpi=100); s.ax = s.fig.add_subplot(111); s.cv = FigureCanvasTkAgg(s.fig, rf); s.cv.get_tk_widget().pack(fill="both", expand=True)
        s.stx = scrolledtext.ScrolledText(s.nb, font=("Segoe UI", 10), wrap="word"); s.nb.add(s.stx, text=" Selector verdict ")
        s.build_tabs()
        if s.dd.get(): s.load()
        root.after(80, s.poll)
    def build_tabs(s):
        s.T, s.P = R / "results/tables", R / "results/plots"
        def tab(title):
            f = ttk.Frame(s.nb); s.nb.add(f, text=title); return f
        def bar(f, text, buttons):
            lab = ttk.Label(f, text=text, wraplength=780, justify="left"); lab.pack(anchor="w", padx=6, pady=(6, 2)); b = ttk.Frame(f); b.pack(anchor="w", padx=6, pady=2)
            for t, cmd in buttons: ttk.Button(b, text=t, command=cmd).pack(side="left", padx=(0, 6))
            return lab
        def picker(f, plots, view):
            row = ttk.Frame(f); row.pack(anchor="w", padx=6, pady=2); ttk.Label(row, text="Plot:").pack(side="left"); cb = ttk.Combobox(row, state="readonly", width=52, values=list(plots)); cb.pack(side="left", padx=4); cb.set(list(plots)[0])
            ttk.Button(row, text="Open full size", command=lambda: s.open_file(s.P / plots[cb.get()])).pack(side="left", padx=4); cb.bind("<<ComboboxSelected>>", lambda e: view.show(s.P / plots[cb.get()])); return cb
        f1 = tab(" 1 Merged table "); s.l1 = bar(f1, "", [("Rebuild from the teams' CSVs", lambda: s.rerun("02_consolidate.py")), ("Open unified_raw.csv", lambda: s.open_file(s.T / "unified_raw.csv"))])
        fr, s.t1 = table(f1, ["matrix", "best OpenMP", "best MPI", "best GPU"] + ALLC, [150, 100, 105, 135] + [82] * 3 + [105, 105, 70] + [105, 105, 70, 80, 95]); fr.pack(fill="both", expand=True, padx=6, pady=4)
        f2 = tab(" 2 Features "); s.l2 = bar(f2, "", [("Recompute from the matrix files", s.recompute_features), ("Open matrix_features.csv", lambda: s.open_file(s.T / "matrix_features.csv"))])
        fr, s.t2 = table(f2, ["matrix", "rows", "nnz", "mean nnz/row", "std nnz/row", "max nnz/row", "bandwidth", "irregularity (max/mean)", "category"], [160, 80, 90, 100, 100, 100, 90, 140, 90], 16); fr.pack(fill="both", expand=True, padx=6, pady=4)
        f3 = tab(" 3 Plots & correlations "); bar(f3, "Performance against each matrix feature, and the strongest correlations (Spearman rank correlation; * = p < 0.05, uncorrected; error bars = run-to-run std).", [("Rebuild plots", lambda: s.rerun("03_correlations_plots.py"))])
        h3 = ttk.Frame(f3); s.v3 = ImgView(h3); s.pk3 = picker(f3, PLOTS3, s.v3); h3.pack(fill="both", expand=True)
        ttk.Label(f3, text="Strongest significant correlations (absolute GFLOP/s vs feature):").pack(anchor="w", padx=6)
        fr, s.t3 = table(f3, ["candidate", "feature", "rho", "p-value", "n"], [200, 220, 80, 100, 50], 4); fr.pack(fill="x", padx=6, pady=4)
        f4 = tab(" 4 Selector accuracy "); s.l4 = bar(f4, "", [("Re-run the selector", lambda: s.rerun("04_selector.py")), ("Open selector_accuracy.csv", lambda: s.open_file(s.T / "selector_accuracy.csv"))])
        fl = ttk.Frame(f4); fl.pack(anchor="w", padx=6); ttk.Label(fl, text="Show task:").pack(side="left"); s.tf = ttk.Combobox(fl, state="readonly", width=44, values=["All tasks"]); s.tf.set("All tasks"); s.tf.pack(side="left", padx=4); s.tf.bind("<<ComboboxSelected>>", lambda e: s.fill_acc())
        fr, s.t4 = table(f4, ["task", "method", "top-1 %", "within 5 %", "mean % of best", "min % of best", "rule"], [200, 175, 60, 70, 90, 90, 330], 9); fr.pack(fill="x", padx=6, pady=4)
        h4 = ttk.Frame(f4); s.v4 = ImgView(h4); s.pk4 = picker(f4, PLOTS4, s.v4); h4.pack(fill="both", expand=True)
        s.load_tables()
    def load_tables(s):
        T = s.T
        try:
            perf = pd.read_csv(T / "performance_matrix.csv"); win = pd.read_csv(T / "winners.csv").set_index("matrix_id"); n = len(pd.read_csv(T / "unified_raw.csv", usecols=["run_id"]))
            fill(s.t1, [[r[0], win.loc[r[0], "best_OpenMP"], win.loc[r[0], "best_MPI"], win.loc[r[0], "best_GPU_own"]] + ["" if pd.isna(v) else f"{v:.2f}" for v in r[1:]] for r in perf.itertuples(index=False)])
            s.l1.config(text=f"Task 1: results of Teams 5 (OpenMP), 6 (MPI) and 7 (CUDA) in one clean table: {n:,} validated runs merged; each cell is the best tuned GFLOP/s of that candidate on that matrix (blank = not benchmarked). The first columns name the winner within each platform; scroll right for all 11 candidates.")
            ft = pd.read_csv(T / "matrix_features.csv").sort_values("nnz")
            fill(s.t2, [[r.matrix_id, f"{r.rows:,}", f"{r.nnz:,}", f"{r.mean_nnz_row:.2f}", f"{r.std_nnz_row:.2f}", r.max_nnz_row, f"{r.bandwidth:,}", f"{r.irregularity:.2f}", r.category] for r in ft.itertuples()])
            s.l2.config(text=f"Task 2: features computed directly from the {len(ft)} matrix files (symmetric matrices expanded): rows, nnz, mean and standard deviation of nnz per row, maximum nnz per row, bandwidth, plus irregularity = max / mean.")
            cl = pd.read_csv(T / "correlations_long.csv"); a = cl[(cl.kind == "absolute") & (cl.p < .05)].copy(); a["ab"] = a.rho.abs()
            fill(s.t3, [[r.candidate, r.feature, f"{r.rho:+.2f}", f"{r.p:.4f}", int(r.n)] for r in a.sort_values("ab", ascending=False).head(12).itertuples()])
            ac = pd.read_csv(T / "selector_accuracy.csv"); s.acc = ac; s.tf.configure(values=["All tasks"] + list(dict.fromkeys(ac.task))); s.fill_acc()
            g = ac[(ac.task == "GPU kernel (own)") & (ac.method == "Tree-10 (LOO)")].top1.iloc[0]; b = ac[(ac.task == "GPU kernel (own)") & (ac.method == "Majority (LOO)")].top1.iloc[0]
            s.l4.config(text=f"Task 4: selector accuracy = percentage of the 13 matrices where the best choice is predicted (leave-one-out: each matrix is predicted by a model trained on the other 12). Headline: GPU kernel, std nnz/row > 30 -> vector-CSR else HYB: {g:.0f}% vs {b:.0f}% for always guessing the most common winner. Rule rows were set by hand after seeing the data (in-sample).")
        except Exception as e: s.st.set(f"Could not read result tables: {e}")
        s.v3.show(s.P / PLOTS3[s.pk3.get()]); s.v4.show(s.P / PLOTS4[s.pk4.get()])
    def fill_acc(s):
        ac = s.acc; t = s.tf.get(); ac = ac if t == "All tasks" else ac[ac.task == t]
        fill(s.t4, [[r.task, r.method, f"{r.top1:.1f}", f"{r.within5:.1f}", f"{r.mean_rel:.1f}", f"{r.min_rel:.1f}", ("" if pd.isna(r.rule_text) else r.rule_text)] for r in ac.itertuples()])
    def open_file(s, p):
        try:
            if os.name == "nt": os.startfile(str(p))
            elif "microsoft" in platform.release().lower() and shutil.which("explorer.exe"): subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", str(p)], text=True).strip()])
            else: webbrowser.open(Path(p).as_uri())
        except Exception as e: s.st.set(f"Could not open {Path(p).name}: {e}")
    def recompute_features(s):
        d = Path(s.dd.get()) / "datasets/original"
        if not d.exists(): s.st.set("Choose the data folder first (needs datasets/original with the .mtx files)."); return
        s.rerun("01_features.py", "--mtx-dir", str(d))
    def rerun(s, script, *args):
        if getattr(s, "busy", False): s.st.set("Busy: wait for the current job to finish."); return
        s.busy = True; s.st.set(f"Running {script} ..."); s.pb.start(12); s.q.put(("log", f"\n$ python scripts/{script} {' '.join(args)}"))
        def work():
            try:
                p = subprocess.Popen([sys.executable, "-u", str(R / "scripts" / script), *args], cwd=R / "scripts", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ | {"PYTHONUTF8": "1"})
                for line in p.stdout: s.q.put(("log", line.rstrip()))
                p.wait(); s.q.put(("status", f"{script} finished (exit code {p.returncode}); tables and plots reloaded. Details are in the Execution log tab."))
            except Exception as e: s.q.put(("log", f"ERROR: {e!r}")); s.q.put(("status", f"{script} failed: {e}"))
            s.busy = False; s.q.put(("reload", None))
        threading.Thread(target=work, daemon=True).start()
    def browse(s):
        d = filedialog.askdirectory()
        if d: s.dd.set(d); s.load()
    def load(s):
        s.lb.delete(0, "end")
        if not (Path(s.dd.get()) / "datasets/original").exists(): s.st.set("Folder not found: expected datasets/original inside it"); return
        s.map = suite.idmap(s.dd.get())
        for i in sorted(s.map): s.lb.insert("end", i)
        s.st.set(f"{s.lb.size()} matrices found. Pick one or more (small ones: qpband, wy2010).")
    def put(s, t, txt): t.configure(state="normal"); t.insert("end", txt); t.see("end")
    def start(s):
        ids = [s.lb.get(i) for i in s.lb.curselection()]
        if not ids: s.st.set("Select at least one matrix first."); return
        s.rows.clear(); s.tree.delete(*s.tree.get_children()); s.sel.clear(); s.stx.delete("1.0", "end"); s.log.delete("1.0", "end"); s.halt = False; s.nb.select(0); s.run_b.state(["disabled"]); s.pb.start(12); s.st.set("Running ...")
        threading.Thread(target=s.work, args=(ids,), daemon=True).start()
    def stop(s):
        s.halt = True
        if s.proc and s.proc.poll() is None: s.proc.terminate()
        s.st.set("Stopping ...")
    def work(s, ids):
        d = Path(s.dd.get()); q = s.q
        try:
            if s.mode.get() == "engine":
                for i in ids:
                    if s.halt: break
                    ref = d / "reference_outputs" / f"{i}_y_reference.txt"
                    engine_run(i, s.map[i], d / f"datasets/vectors/{i}_x.txt", ref if ref.exists() else None, q, lambda: s.halt)
            else:
                teams = ",".join(t for t in "567" if s.tv[t].get())
                cmd = [sys.executable, "-u", str(R / "suite.py"), "run", "--data-dir", str(d), "--matrices", ",".join(ids), "--teams", teams, "--ranks", s.rk.get()]
                q.put(("log", "$ " + " ".join(cmd) + "\n")); s.proc = subprocess.Popen(cmd, cwd=R, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ | {"PYTHONUTF8": "1"})
                for line in s.proc.stdout: q.put(("log", line.rstrip()))
                s.proc.wait(); f = R / "live_summary.csv"
                if f.exists():
                    for r in pd.read_csv(f).itertuples(): q.put(("row", dict(matrix=r.dataset, cand=r.candidate, gflops=r.gflops, ms=float("nan"), check="live")))
        except Exception as e: q.put(("log", f"ERROR: {e!r}"))
        q.put(("done", None))
    def poll(s):
        try:
            while True:
                k, v = s.q.get_nowait()
                if k == "log": s.put(s.log, v + "\n")
                elif k == "row": s.rows.append(v); s.tree.insert("", "end", values=(v["matrix"], v["cand"], f"{v['gflops']:.3f}", "" if v["ms"] != v["ms"] else f"{v['ms']:.3f}", v["check"])); s.cm["values"] = sorted({r["matrix"] for r in s.rows}); s.cm.set(v["matrix"]); s.draw()
                elif k == "sel": s.sel[v["matrix"]] = v; s.verdict(v)
                elif k == "status": s.st.set(v)
                elif k == "reload": s.pb.stop(); s.load_tables()
                elif k == "done": s.pb.stop(); s.run_b.state(["!disabled"]); s.st.set("Finished. See Results and Selector tabs." if not s.halt else "Stopped."); s.nb.select(1)
        except queue.Empty: pass
        s.root.after(80, s.poll)
    def draw(s):
        m = s.cm.get(); rs = [r for r in s.rows if r["matrix"] == m]; s.ax.clear()
        if rs:
            names = [r["cand"] for r in rs]; s.ax.barh(names, [r["gflops"] for r in rs], color=[COL.get(n.split("-")[0], "#555") for n in names])
            if s.rec is not None and m in s.rec.index:
                first = True
                for j, n in enumerate(names):
                    rn = "OMP-" + n[3:] if n.startswith("PY-") else n       # Python engine vs the recorded OpenMP result of the same format
                    if rn in s.rec.columns and pd.notna(s.rec.loc[m, rn]): s.ax.plot(s.rec.loc[m, rn], j, "kD", ms=6, label="recorded (teams' machines)" if first else None); first = False
                if not first: s.ax.legend(fontsize=8, loc="lower right")
            s.ax.set_xlabel("GFLOP/s  (bars = this run, diamonds = recorded)"); s.ax.set_title(m)
        s.fig.tight_layout(); s.cv.draw_idle()
    def verdict(s, v):
        i = v["matrix"]; fmt = "ELL" if v["irr"] <= 1.5 else "CSR"; gpu = "vector-CSR" if v["std"] > 30 else "HYB"
        mine = [r for r in s.rows if r["matrix"] == i]; best = max(mine, key=lambda r: r["gflops"])["cand"] if mine else "n/a"
        t = f"{i}   (nnz {v['nnz']:,}, mean {v['mean']:.1f}/row, std {v['std']:.1f}, irregularity {v['irr']:.1f})\n  Selector: format = {fmt}  |  GPU kernel = {gpu}  |  MPI = 1D-nnz (low confidence)\n  Measured best format in this run: {best}"
        if s.rec is not None and i in s.rec.index:
            r = s.rec.loc[i]; t += "\n  Recorded best (the teams' own runs on their machines):"
            for g, cs in (("OpenMP", ["OMP-COO", "OMP-CSR", "OMP-ELL"]), ("MPI", ["MPI-1D-rows", "MPI-1D-nnz", "MPI-2D"]), ("GPU", ["GPU-scalar-CSR", "GPU-vector-CSR", "GPU-ELL", "GPU-HYB"])): t += f"\n     {g}: {r[cs].astype(float).idxmax()} ({r[cs].astype(float).max():.1f} GFLOP/s)"
        s.put(s.stx, t + "\n\n")

def main():
    root = tk.Tk(); app = App(root)
    if "--selftest" in sys.argv:           # headless test: python gui.py --selftest DATA_DIR MATRIX OUT.png
        d, m, out = sys.argv[-3:]; app.dd.set(d); app.load(); app.lb.selection_set(list(app.lb.get(0, "end")).index(m)); app.start()
        app.root_ = root
        def chk():
            if str(app.run_b.state()) and "disabled" not in app.run_b.state(): subprocess.run(["import", "-window", "root", out]); print(app.stx.get("1.0", "end")); root.destroy()
            else: root.after(500, chk)
        root.after(1500, chk)
    root.mainloop()
if __name__ == "__main__": main()
