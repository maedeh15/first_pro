"""
check_fnirs_2v3.py -- the 2-back vs 3-back bottleneck, probed on fNIRS.

Why fNIRS and why this pair. After late fusion, 74.3% of the remaining window errors are 2-back
confused with 3-back, and that pair is the one place the EEG side does NOT help: the aligned
covariance member scores 0.5818 on 2-vs-3 against 0.5996 for the deep model, i.e. below it. What
little 2-vs-3 signal the model has comes from the fNIRS side. So the question this script asks is
narrow on purpose: does any cheap fNIRS feature family separate 2-back from 3-back across sessions?

The primary metric is 2-VS-3 (train and test on those two classes only). 3-class accuracy is printed
beside it and is NOT the reading: a feature set can lift 3-class by sharpening 0-back, which is
already solved, while leaving the bottleneck exactly where it was.

Feature families, all per grid cell and chromophore, each session re-centred independently (the same
transductive per-session treatment the baseline already applies through E3/AdaBN):

  mean        window mean                                       the level feature already in use
  slope       linear trend over the window                      the shape feature already in use
  mean_slope  both                                              = check_feature_baseline_tsfnet.py
  hbdiff      HbO - HbR and HbO + HbR, mean and slope           the standard contrast pair; HbT is
                                                                dominated by systemic physiology,
                                                                HbO-HbR less so
  seg         the per-segment means, kept separate              the within-window time course, which
                                                                mean and slope both throw away
  cov         covariance between grid cells, EA-aligned,        the family that worked on EEG, asked
              tangent space                                     of fNIRS

WHAT THIS CANNOT DO. There is no short-separation channel in this dataset, so systemic physiology
(cardiac, respiratory, blood pressure) cannot be regressed out, and every family here still carries
it. A null result is therefore evidence that these features do not separate the pair, not that fNIRS
cannot -- the systemic-regression route (rest periods as a baseline scan for a PCA filter) is a
separate experiment. A positive result is necessary, never sufficient: every earlier probe in this
project was more optimistic than the deep model that followed it.

usage:
  python check_fnirs_2v3.py --self-test
  python check_fnirs_2v3.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" [--device cuda]
"""
import os
import argparse
import numpy as np
from scipy.stats import wilcoxon

import check_riemannian_eeg as R

FAMILIES = ("mean", "slope", "mean_slope", "hbdiff", "seg", "cov")


def _center_per_session(X, ses):
    """remove each session's own mean and scale, column-wise. Label-free, same transductive
    assumption as E3/AdaBN; without it a session offset alone can separate the folds."""
    out = np.empty_like(X, dtype=np.float64)
    for s in np.unique(ses):
        m = ses == s
        blk = X[m].astype(np.float64)
        out[m] = (blk - blk.mean(0)) / (blk.std(0) + 1e-8)
    return out


def _mean_slope(f):
    """f: (N, H, W, T, C) -> mean over time, slope over time, each (N, H*W*C)"""
    t = np.arange(f.shape[-2], dtype=np.float64)
    t = t - t.mean()
    N = f.shape[0]
    mean_ = f.mean(-2).reshape(N, -1)
    slope = ((f * t[None, None, None, :, None]).sum(-2) / (t ** 2).sum()).reshape(N, -1)
    return mean_, slope


def features(name, fn, ses):
    """fn: (N, S, H, W, T, C) raw windows -> (N, D) features, per-session centred"""
    N, S, H, W, T, C = fn.shape
    f = fn.mean(axis=1)                                    # segment-averaged, (N,H,W,T,C)
    if name == "seg":
        X = fn.mean(axis=4).reshape(N, -1)                 # per-segment means, (N, S*H*W*C)
    elif name == "hbdiff":
        d = np.stack([f[..., 0] - f[..., 1], f[..., 0] + f[..., 1]], axis=-1)
        m, s = _mean_slope(d)
        X = np.hstack([m, s])
    elif name == "cov":
        g = f.transpose(0, 4, 1, 2, 3).reshape(N * C, H * W, T)
        live = np.abs(g).sum(axis=(0, 2)) > 0
        g = g[:, live, :]
        flat = g.transpose(1, 0, 2).reshape(g.shape[1], -1)
        w, v = np.linalg.eigh(flat @ flat.T / flat.shape[1])
        B = v[:, ::-1][:, :20]                             # fNIRS has ~36 real channels; 20 keeps
        g = np.einsum("cr,nct->nrt", B, g)                 # the covariance well conditioned
        cov = R.covs(g)
        ses_rep = np.repeat(ses[:, None], C, axis=1).reshape(-1) + 1000 * np.tile(np.arange(C), N)
        X = R.domain_features(cov, ses_rep, "EA").reshape(N, -1)
    else:
        m, s = _mean_slope(f)
        X = {"mean": m, "slope": s, "mean_slope": np.hstack([m, s])}[name]
    return _center_per_session(X, ses)


def run_subject(fn, y, ses, fams):
    out = {}
    for name in fams:
        X = features(name, fn, ses)
        out[name] = (R.loso(X, y, ses), R.loso(X, y, ses, (1, 2)))
    return out


def self_test():
    """plant a 2-vs-3 effect that lives ONLY in the within-window time course: classes 1 and 2 have
    the same window mean and the same linear trend, and differ in the shape between them. 'mean',
    'slope' and 'mean_slope' must stay near chance on 2-vs-3; 'seg' must find it. If the script
    cannot see a planted 2-vs-3 effect, a null on the real data says nothing."""
    rng = np.random.RandomState(0)
    N_per, S, H, W, T, C = 40, 6, 4, 4, 10, 2
    ses = np.repeat([1, 2, 3], N_per)
    y = np.tile([0, 1, 2], N_per)
    fn = rng.randn(3 * N_per, S, H, W, T, C) * 0.3
    hump = np.zeros(S)
    hump[S // 2] = 1.0                                   # a bump in the middle segment
    hump = hump - hump.mean()                            # zero mean AND zero linear trend
    hump = hump - np.polyval(np.polyfit(np.arange(S), hump, 1), np.arange(S))
    for i in range(len(y)):
        if y[i] == 2:
            fn[i] += 1.2 * hump.reshape(S, 1, 1, 1, 1)
        if y[i] == 0:
            fn[i] += 0.9                                 # 0-back is a plain level shift
    r = run_subject(fn, y, ses, ("mean", "mean_slope", "seg"))
    for k in ("mean", "mean_slope", "seg"):
        print(f"  {k:11} 3-class {r[k][0]:.3f}  2v3 {r[k][1]:.3f}")
    ok = r["seg"][1] > 0.75 and r["mean"][1] < 0.65 and r["mean_slope"][1] < 0.65
    print(f"[timecourse] planted 2v3 found by 'seg', missed by mean/slope -> {'OK' if ok else 'FAIL'}")
    good = r["mean"][0] > 0.55
    print(f"[level] the planted 0-back level shift is still found by 'mean' -> {'OK' if good else 'FAIL'}")
    ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default=None, help="torch device for the eigendecompositions (cov only)")
    ap.add_argument("--fams", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    if a.device:
        print(f"eigendecomposition backend: {R.set_eigh_backend(a.device)}")
    rows = []
    hdr = "  ".join(f"{f:>10}" for f in a.fams)
    print(f"{'subject':8} {'2-VS-3:':>10} {hdr}   | 3-class: {hdr}")
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        p = os.path.join(d, f"{sid}_fnirs_windows.npy")
        if not os.path.exists(p):
            continue
        fn = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        r = run_subject(fn, y, ses, a.fams)
        rows.append(r)
        print(f"{sid:8} {'':>10} " + "  ".join(f"{r[f][1]:10.3f}" for f in a.fams) + "   | "
              + "  ".join(f"{r[f][0]:10.3f}" for f in a.fams), flush=True)
    if not rows:
        raise SystemExit("no subject found under --data")
    print("\nMEAN over subjects:")
    print(f"  {'family':12} {'2-VS-3':>9} {'3-class':>9}")
    for f in a.fams:
        print(f"  {f:12} {np.mean([r[f][1] for r in rows]):9.4f} {np.mean([r[f][0] for r in rows]):9.4f}")
    base = "mean_slope" if "mean_slope" in a.fams else a.fams[0]
    print(f"\n2-VS-3, paired against '{base}' (the features the pipeline already uses):")
    for f in a.fams:
        if f == base:
            continue
        d = np.array([r[f][1] - r[base][1] for r in rows])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  {f:12} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    print(f"\nReference points for 2-vs-3: deep model 0.5996, EEG covariance member 0.5818, late "
          f"fusion 0.6135, chance 0.5. A family has to beat 0.5996 to be worth a GPU run, and the "
          f"p-values above are uncorrected over {len(a.fams)} families -- apply Holm before reading "
          f"any of them as a result. No short-separation channel exists in this dataset, so systemic "
          f"physiology is present in every row; a null here is a null for THESE features.")


if __name__ == "__main__":
    main()
