"""cmp_common.py -- shared helpers of the comparison scripts (cmp_a1..): run-file discovery, subject-level statistics (paired Wilcoxon, Holm, bootstrap CI over subjects, Cliff's delta)."""
import glob, os, re
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

RUN_RE = re.compile(r"baseline_all_subjects_(\d{8}_\d{6})_seed(\d+)_([0-9a-f]{10})\.(csv|json)$")


def find_runs(roots):
    """{(hash, seed): csv_path}; the CSV is preferred, the JSON used only if no CSV exists; the newest stamp wins when a run was repeated"""
    found = {}
    for root in roots:
        for p in glob.glob(os.path.join(root, "**", "baseline_all_subjects_*.*"), recursive=True):
            m = RUN_RE.search(os.path.basename(p))
            if not m: continue
            stamp, seed, h, ext = m.group(1), int(m.group(2)), m.group(3), m.group(4)
            key = (h, seed); cur = found.get(key)
            if cur is None or (ext == "csv" and cur[2] == "json") or (ext == cur[2] and stamp > cur[0]):
                found[key] = (stamp, p, ext)
    return {k: v[1] for k, v in found.items()}


def load_run(path):
    if path.endswith(".json"):
        d = pd.DataFrame(pd.read_json(path).T).reset_index(drop=True)
    else:
        d = pd.read_csv(path)
    if "status" in d.columns: d = d[d["status"] == "ok"]
    return d


def holm(p):
    p = np.asarray(p, float); n = len(p); order = np.argsort(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def cliffs_delta(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(((a[:, None] > b[None, :]).sum() - (a[:, None] < b[None, :]).sum()) / (len(a) * len(b)))


def delta_label(d):
    d = abs(d); return "negligible" if d < 0.147 else "small" if d < 0.33 else "medium" if d < 0.474 else "large"


def boot_ci(x, n=10000, seed=0):
    rng = np.random.RandomState(seed); x = np.asarray(x, float)
    m = x[rng.randint(0, len(x), (n, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, n_boot=10000, seed=0):
    """a, b: per-subject vectors in the same subject order; difference = a - b (in accuracy units, 0-1)"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = a - b; nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) else 1.0
    return dict(n=len(d), mean_a=float(a.mean()), mean_b=float(b.mean()), ci_a=boot_ci(a, n_boot, seed), ci_b=boot_ci(b, n_boot, seed), diff=float(d.mean()),
                ci_diff=boot_ci(d, n_boot, seed), cliffs=cliffs_delta(a, b), p_raw=p, n_better=int((d > 0).sum()), n_worse=int((d < 0).sum()))
