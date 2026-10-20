"""
check_fnirs_alignment.py -- cheap probe: does Euclidean-Alignment-style covariance whitening of the
fNIRS signal, per session, change cross-session accuracy of the simple linear probe?

Why fNIRS: the 2-back vs 3-back bottleneck lives in fNIRS (EEG is at chance for that pair). The
EEG version of this idea (--eeg-cov-align) gave +1.4 on the EEG head and nothing at the output.

What it does (per subject, same split as check_feature_baseline_tsfnet.py: leave-one-session-out)
  features : per grid cell and chromophore, mean and linear slope over time of the segment-averaged
             window (identical to check_feature_baseline_tsfnet.py) -> 16*16*C*2 features
  EA       : for each session and each chromophore, treat the 256 grid cells as "channels". The
             grid is a LINEAR interpolation of ~36 real channels, so the cell covariance has rank
             ~36: R^-1/2 is therefore a truncated pseudo-inverse square root (eigenvalues above
             --rel-eig of the largest; the rest are projected out). Two references:
               second-moment   R = mean_over_windows(X X^T / t)     (paper form, Wu 2025 Eq. 1)
               centred         R = mean_over_windows(Xc Xc^T / t)   (Xc = window-mean removed, so the
                               reference describes temporal fluctuation, not the level the mean
                               feature carries)
             then X~ = R^-1/2 X is applied to the window (mean and slope are linear in X, so the
             segment average, the transform and the feature extraction commute).
  metrics  : 3-class LOSO accuracy and 2-vs-3 LOSO accuracy (train and test on classes 2/3 only),
             raw vs EA, paired over subjects with Wilcoxon.

HONEST LIMITS (printed again at the end): it runs on the interpolated grid, not on channel space
(the channel-space fNIRS windows are not saved); the probe is linear; and every earlier probe in
this project was MORE optimistic than the deep model (grid alignment, covariance alignment,
neurovascular coupling...). A positive result here is necessary, never sufficient.

usage:
  python check_fnirs_alignment.py --self-test
  python check_fnirs_alignment.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import os
import argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _sqrt_inv_trunc(R, rel_eig):
    w, v = np.linalg.eigh(R)
    keep = w > rel_eig * w.max()
    d = np.zeros_like(w)
    d[keep] = w[keep] ** -0.5
    return (v * d) @ v.T, int(keep.sum())


def align_session(X, mode, rel_eig):
    """X: (n, cells, T) one chromophore, one session. returns aligned X and the retained rank."""
    n, c, t = X.shape
    if mode == "centred":
        Xr = X - X.mean(-1, keepdims=True)
    else:
        Xr = X
    R = np.einsum("nct,ndt->cd", Xr, Xr) / (n * t)
    T, k = _sqrt_inv_trunc(R, rel_eig)
    return np.einsum("cd,ndt->nct", T, X), k


def features_from(f):
    """f: (N, 16, 16, T, C) -> mean and slope per cell/chromophore, flattened (N, 16*16*C*2)"""
    t = np.arange(f.shape[-2], dtype=np.float64)
    t -= t.mean()
    slope = (f * t[None, None, None, :, None]).sum(-2) / (t ** 2).sum()
    mean_ = f.mean(-2)
    N = f.shape[0]
    return np.concatenate([mean_.reshape(N, -1), slope.reshape(N, -1)], axis=1)


def align_windows(f, ses, mode, rel_eig):
    """f: (N,16,16,T,C) segment-averaged windows -> aligned copy"""
    N, H, W, T, C = f.shape
    out = np.empty_like(f, dtype=np.float64)
    ranks = []
    for s in np.unique(ses):
        m = ses == s
        for ch in range(C):
            X = f[m][..., ch].reshape(m.sum(), H * W, T).astype(np.float64)
            Xa, k = align_session(X, mode, rel_eig)
            out[m, ..., ch] = Xa.reshape(m.sum(), H, W, T)
            ranks.append(k)
    return out, int(np.median(ranks))


def loso(X, y, ses, classes=None):
    accs = []
    for s in np.unique(ses):
        tr, te = ses != s, ses == s
        if classes is not None:
            tr = tr & np.isin(y, classes)
            te = te & np.isin(y, classes)
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else float("nan")


def evaluate(f, y, ses, rel_eig):
    out = {}
    out["raw"] = (loso(features_from(f), y, ses), loso(features_from(f), y, ses, (1, 2)))
    for mode in ("second", "centred"):
        fa, k = align_windows(f, ses, mode, rel_eig)
        Xa = features_from(fa)
        out[mode] = (loso(Xa, y, ses), loso(Xa, y, ses, (1, 2)), k)
    return out


def self_test():
    """planted: the 256 'cells' are a linear map of 12 latent sources; every session rescales the
    latents with its own gains (+ mild mixing): a COVARIANCE shift of the kind EA is built for. The
    class effect is a latent-mean shift. EA removes covariance/gain shifts but NOT a remaining
    orthogonal rotation, so a large random mixing would (correctly) not be recoverable -- we plant
    only what EA can fix."""
    rng = np.random.RandomState(0)
    rows = []
    for rep in range(6):
        Wm = rng.randn(256, 12)
        N_per, T = 60, 30
        f, y, ses = [], [], []
        for s in (1, 2, 3):
            M = np.diag(np.exp(0.9 * rng.randn(12))) @ (np.eye(12) + 0.15 * rng.randn(12, 12))   # per-latent gain + mild mixing
            for i in range(N_per):
                cls = i % 3
                lat = rng.randn(12, T) * 0.6
                lat[cls] += 0.8                    # class: shift of latent 'cls'
                x = Wm @ (M @ lat)                 # (256, T)
                f.append(x.reshape(16, 16, T)[..., None])
                y.append(cls)
                ses.append(s)
        f, y, ses = np.array(f), np.array(y), np.array(ses)
        rows.append(evaluate(f, y, ses, 1e-3))
    g = lambda k, j: np.mean([r[k][j] for r in rows])
    print("planted session-specific latent mixing (chance 3-class 0.333, 2v3 0.5):")
    print(f"  raw      3-class {g('raw', 0):.3f}  2v3 {g('raw', 1):.3f}")
    print(f"  EA 2nd   3-class {g('second', 0):.3f}  2v3 {g('second', 1):.3f}  (rank kept {rows[0]['second'][2]})")
    print(f"  EA cent. 3-class {g('centred', 0):.3f}  2v3 {g('centred', 1):.3f}")
    good = max(g("second", 0), g("centred", 0)) > g("raw", 0) + 0.08 and rows[0]["second"][2] <= 14
    print("SELFTEST", "PASSED" if good else "FAILED")
    return 0 if good else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--rel-eig", type=float, default=1e-3,
                    help="keep eigenvalues above this fraction of the largest (truncated pseudo-inverse)")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    rows = []
    print(f"{'subject':8} {'raw3':>6} {'EA2nd3':>7} {'EAcen3':>7} | {'raw2v3':>7} {'EA2nd':>6} {'EAcen':>6} | rank")
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        if not os.path.exists(os.path.join(d, f"{sid}_labels.npy")):
            continue
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        fn = np.load(os.path.join(d, f"{sid}_fnirs_windows.npy"), mmap_mode="r")      # (N,S,16,16,T,C)
        f = np.asarray(fn, dtype=np.float32).mean(axis=1)                              # (N,16,16,T,C)
        r = evaluate(f, y, ses, a.rel_eig)
        rows.append(r)
        print(f"{sid:8} {r['raw'][0]:6.3f} {r['second'][0]:7.3f} {r['centred'][0]:7.3f} | "
              f"{r['raw'][1]:7.3f} {r['second'][1]:6.3f} {r['centred'][1]:6.3f} | {r['second'][2]}", flush=True)
    if not rows:
        raise SystemExit("no subjects found under --data")
    print("\nMEAN over subjects (3-class | 2-vs-3):")
    for k, nm in (("raw", "raw"), ("second", "EA second-moment"), ("centred", "EA centred")):
        print(f"  {nm:18} {np.mean([r[k][0] for r in rows]):.4f} | {np.mean([r[k][1] for r in rows]):.4f}")
    print("\npaired differences vs raw (Wilcoxon, subjects as units):")
    for k, nm in (("second", "EA second-moment"), ("centred", "EA centred")):
        for j, lab in ((0, "3-class"), (1, "2-vs-3")):
            d = np.array([r[k][j] - r["raw"][j] for r in rows])
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  {nm:18} {lab:8} diff {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    print("\nLimits: runs on the interpolated grid (truncated pseudo-inverse, rank ~36), linear probe, "
          "no AdaBN. Earlier probes in this project over-predicted deep-model gains; treat a positive result "
          "as 'not destroyed', not as 'will help'.")


if __name__ == "__main__":
    main()
