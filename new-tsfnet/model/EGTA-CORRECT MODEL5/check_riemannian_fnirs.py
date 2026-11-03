"""
check_riemannian_fnirs.py -- cheap CPU probe: do Riemannian (covariance) features of the fNIRS signal carry load information that simple level/slope features do not?
Same protocol as check_riemannian_eeg.py (per-session Frechet re-centering, tangent vectors, StandardScaler + LogReg(C=0.1), leave-one-session-out, 3-class and 2v3).
DATA: the 36-channel channel-space fNIRS is NOT saved per window (only a QC .fif exists), so this works on the 16x16 interpolated grid: 2x2 average pooling -> 8x8 cells,
cells with no signal dropped, K cells per chromophore (~30-40).  The 23 s continuous trace of each window is rebuilt from its 21 overlapping 3 s segments.  This is an
APPROXIMATION of channel covariance (interpolated neighbours are strongly correlated; the SPD matrix is ill-conditioned, hence the trace-scaled ridge 1e-2).
Feature sets (all per-session domain handling is label-free and transductive, like AdaBN):
  riem_sep      tangent vectors of the K x K covariance of HbO and of HbR, concatenated, Frechet re-centred per session
  riem_joint    tangent vector of the 2K x 2K joint covariance (adds HbO-HbR cross-covariance), re-centred
  simple_global mean and slope of the global (all-cell) HbO and HbR trace                     (4 features)
  simple_cells  mean and slope per cell and chromophore                                       (4K features; the strong simple competitor)
  logvar_cells  log-variance per cell and chromophore                                         (2K features; the cheap analogue of the covariance diagonal)
PRE-REGISTERED DECISION (before running): Riemannian fNIRS is worth adding as an ensemble member only if the better of riem_sep / riem_joint beats simple_cells by
>= +0.02 mean 3-class accuracy AND paired Wilcoxon p < 0.05 over subjects (two-sided). Otherwise it is reported as a negative result and NOT added.
Expectation, stated before the run: the slow fNIRS information sits in level/slope, so simple_cells >= riem is the likely outcome; chance 0.333.
Reference only (from earlier runs, not recomputed here): deep fNIRS head ~0.58.
  python check_riemannian_fnirs.py --data "...\\TSFNET_PAPER_EXACT_nback" --subjects VP001 VP002 ... [--device cuda]
  python check_riemannian_fnirs.py --self-test
"""
import os, argparse, sys
import numpy as np
from scipy.stats import wilcoxon
import check_riemannian_eeg as R

STEP, SEG = 10, 30
RIDGE_F = 1e-2


def rebuild_pooled(win, mask=None):
    """win (n,S,H,W,30,C) -> (n,C,K,L): continuous trace, 2x2-pooled cells, cells outside the signal mask dropped"""
    n, S, H, W, T, C = win.shape; L = (S - 1) * STEP + T
    tr = np.zeros((n, H, W, L, C), np.float32)
    for k in range(S): tr[:, :, :, k * STEP:k * STEP + T, :] = win[:, k]
    tr = tr.reshape(n, H // 2, 2, W // 2, 2, L, C).mean((2, 4)).reshape(n, -1, L, C)
    if mask is None: mask = np.abs(tr[0]).sum((1, 2)) > 0
    return np.transpose(tr[:, mask], (0, 3, 1, 2)), mask                     # (n, C, K, L)


def load_subject(sd, sid, chunk=64):
    win = np.load(os.path.join(sd, f"{sid}_fnirs_windows.npy"), mmap_mode="r"); out = []; mask = None
    for i in range(0, len(win), chunk):
        a, mask = rebuild_pooled(np.asarray(win[i:i + chunk], dtype=np.float32), mask); out.append(a)
    return np.concatenate(out)


def slope(x):
    t = np.arange(x.shape[-1]) - (x.shape[-1] - 1) / 2; return (x * t).sum(-1) / (t ** 2).sum()


def feature_sets(tr, ses):
    """tr (N,C,K,L) -> dict of (N,d) feature matrices"""
    N, C, K, L = tr.shape; f = {}
    f["simple_global"] = np.concatenate([tr.mean(2).mean(-1), slope(tr.mean(2))], 1)
    f["simple_cells"] = np.concatenate([tr.mean(-1).reshape(N, -1), slope(tr).reshape(N, -1)], 1)
    f["logvar_cells"] = np.log(tr.var(-1).reshape(N, -1) + 1e-8)
    f["riem_sep"] = np.hstack([R.domain_features(R.covs(tr[:, c].astype(np.float64), RIDGE_F), ses, "riem") for c in range(C)])
    f["riem_joint"] = R.domain_features(R.covs(tr.reshape(N, C * K, L).astype(np.float64), RIDGE_F), ses, "riem")
    return f


def evaluate(tr, y, ses):
    return {k: (R.loso(X, y, ses), R.loso(X, y, ses, (1, 2))) for k, X in feature_sets(tr, ses).items()}


def decide(rows, keys):
    s = np.array([r["simple_cells"][0] for r in rows]); best = max(("riem_sep", "riem_joint"), key=lambda k: np.mean([r[k][0] for r in rows]))   # best by mean, reported openly
    d = np.array([r[best][0] for r in rows]) - s; p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    ok = d.mean() >= 0.02 and p < 0.05
    return best, float(d.mean()), p, ok


def report(rows, ids):
    keys = list(rows[0]); print(f"\nMEAN over {len(rows)} subjects (chance 3-class 0.333, 2v3 0.500)")
    print(f"{'features':16} {'3-class':>8} {'2-vs-3':>8} | {'d3 vs simple_cells':>19} {'p':>7} | subjects better")
    base = np.array([r["simple_cells"][0] for r in rows])
    for k in keys:
        v3 = np.array([r[k][0] for r in rows]); v2 = np.array([r[k][1] for r in rows]); d = v3 - base; p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"{k:16} {v3.mean():8.4f} {v2.mean():8.4f} | {d.mean():+19.4f} {p:7.4f} | {int((d > 0).sum())}/{len(d)}" if k != "simple_cells" else f"{k:16} {v3.mean():8.4f} {v2.mean():8.4f} | {'-':>19} {'-':>7} | -")
    best, dm, p, ok = decide(rows, keys)
    print(f"\nPRE-REGISTERED DECISION: best Riemannian variant = {best}; minus simple_cells = {dm:+.4f}, Wilcoxon p = {p:.4f}")
    print("  -> " + ("ADD as an ensemble member candidate (weight pre-registered, not tuned on test data)" if ok else "DO NOT ADD: report as a negative / non-superior result"))
    print("Limits: grid-pooled approximation of channel covariance; linear probe; transductive per-session centring; probes in this project have over-predicted deep-model gains.")


def _synth(rng, kind, n=60, K=8, L=230):
    """kind 'cov': classes differ only in cross-cell covariance (no level/slope difference); 'level': classes differ only in mean level"""
    y = np.tile([0, 1, 2], n // 3); ses = np.repeat([1, 2, 3], n // 3 * 3 // 3)[:n]; ses = np.sort(np.tile([1, 2, 3], n // 3))[:n]; y = np.tile([0, 1, 2], n // 3)[:n]
    tr = np.zeros((n, 2, K, L))
    for i in range(n):
        z = rng.randn(2, K, L)
        if kind == "cov":
            g = rng.randn(2, 1, L); mix = [0.0, 0.8, 1.6][y[i]]; z = z + mix * g * np.linspace(0.5, 1.5, K)[None, :, None]
        else:
            z = z + [0.0, 0.8, 1.6][y[i]] * 0.5
        tr[i] = z
    return tr, y, ses


def self_test():
    rng = np.random.RandomState(0)
    win = rng.randn(2, 21, 8, 8, 30, 2).astype(np.float32); win[:, :, :2, :, :, :] = 0
    a, m = rebuild_pooled(win); assert a.shape == (2, 2, int(m.sum()), 230) and m.sum() < 16, (a.shape, m.sum())
    print("[1] rebuild + 2x2 pooling + empty-cell mask:", a.shape)
    assert abs(slope(np.arange(230.0)[None]) [0] - 1.0) < 1e-9; print("[2] slope exact")
    res = {}
    for kind in ("cov", "level"):
        rows = []
        for s in range(8):
            tr, y, ses = _synth(np.random.RandomState(s), kind, n=90); rows.append(evaluate(tr, y, ses))
        res[kind] = rows
    c = {k: np.mean([r[k][0] for r in res["cov"]]) for k in res["cov"][0]}; l = {k: np.mean([r[k][0] for r in res["level"]]) for k in res["level"][0]}
    assert c["riem_sep"] > c["simple_cells"] + 0.1 and l["simple_cells"] > l["riem_sep"] + 0.1, (c, l)
    print("[3] covariance-only classes: Riemannian wins; level-only classes: simple wins (the probe distinguishes the two)")
    report(res["cov"], list(range(8))); assert decide(res["cov"], None)[3]
    _, _, _, ok = decide(res["level"], None); assert not ok; print("[4] decision rule: ADD for covariance structure, DO NOT ADD for level structure\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data"); ap.add_argument("--self-test", action="store_true"); ap.add_argument("--device", default=None)
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)]); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.data: sys.exit("need --data")
    print("eigendecomposition backend:", R.set_eigh_backend(a.device)); rows, ids = [], []
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        if not os.path.exists(os.path.join(d, f"{sid}_fnirs_windows.npy")): continue
        tr = load_subject(d, sid); y = np.load(os.path.join(d, f"{sid}_labels.npy")); ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        r = evaluate(tr, y, ses); rows.append(r); ids.append(sid)
        print(f"{sid}  K={tr.shape[2]} | " + " | ".join(f"{k} {v[0]:.3f}" for k, v in r.items()), flush=True)
    if len(rows) < 6: sys.exit(f"{len(rows)} subjects; need >= 6 for the group test")
    report(rows, ids)


if __name__ == "__main__":
    main()
