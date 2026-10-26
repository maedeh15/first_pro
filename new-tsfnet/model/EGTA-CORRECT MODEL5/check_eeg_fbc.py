"""
check_eeg_fbc.py -- EEG functional brain connectivity (FBC) as a feature family, probed on the
2-back vs 3-back bottleneck under THIS project's protocol.

Why. Cao, Garro & Zhao (Sensors 2022, 22, 7623) use bivariate FBC from EEG plus fNIRS HbO/HbR on
the SAME Shin et al. dataset and report 77% for 0-back vs 2-back and 83% for 0-back vs 3-back,
which is where chapter 7 lists this as future work. Four estimators in three bands:
  PCC  Pearson correlation          (linear, time domain)
  MI   mutual information           (nonlinear, time domain; their best)
  MSC  magnitude-squared coherence  (linear, frequency domain)
  PLV  phase-locking value          (nonlinear, frequency domain)
PSD per channel is computed too, because their claim is that FBC beats univariate PSD by ~5 points.

READ THEIR 2-vs-3 NUMBERS BEFORE SPENDING A GPU RUN ON THIS. Their own Table 4 gives 2-vs-3 as
62% for the best FBC (alpha) and 59% for the EEG+fNIRS hybrid -- the hybrid is WORSE than FBC
alone on this pair, and 62% is barely above this project's late fusion (0.6135) and deep model
(0.5996). Two further differences make even 62% optimistic here:
  * Their top-10 features are chosen by a one-way ANOVA computed on the whole dataset, BEFORE the
    5-fold cross-validation that produces the accuracy. Selection therefore sees the test folds.
    This script selects inside the training sessions only (--topk, ANOVA on train), so its numbers
    are not comparable to theirs and are expected to be lower.
  * Their 5 folds are drawn from 702 samples pooled over sessions AND participants, so train and
    test share both. This project's protocol is leave-one-session-out within a subject.
So a null here does not contradict their paper; it says FBC does not transfer to this protocol.

Bands are theirs (delta 0.5-4, theta 4-7, alpha 8-15 Hz), which are NOT this project's bands
(theta 4-8, alpha 8-13, beta 13-30). Kept as published so the comparison is to their claim.

Volume conduction inflates every zero-lag connectivity estimate and is the limitation their own
discussion names. This pipeline applies CAR before saving, which changes the estimates further;
nothing here separates genuine coupling from either effect.
"""
import argparse, os, sys
import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import wilcoxon, f_oneway
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

SFREQ = 200.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 7.0), "alpha": (8.0, 15.0)}
METHODS = ("pcc", "mi", "msc", "plv", "psd")
REF = {"deep model": 0.5996, "late fusion": 0.6135, "EEG cov member": 0.5818, "chance": 0.5}


def bandpass(x, lo, hi, order=3):
    """x: (N, C, T). Their third-order Butterworth, zero-phase."""
    ny = SFREQ / 2
    b, a = butter(order, [max(lo / ny, 1e-6), min(hi / ny, 0.999)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _upper(M):
    """the C*(C-1)/2 upper-triangle entries of a per-window (N, C, C) stack"""
    C = M.shape[-1]
    i, j = np.triu_indices(C, k=1)
    return M[:, i, j]


def fbc_pcc(xb):
    """one matmul per window: correlation of the band-filtered signal"""
    z = xb - xb.mean(-1, keepdims=True)
    s = np.sqrt(np.einsum("nct,nct->nc", z, z)) + 1e-12
    z = z / s[:, :, None]
    return _upper(np.einsum("nct,ndt->ncd", z, z))


def fbc_plv(xb):
    """phases from the Hilbert transform, then |mean exp(i d_phi)| as one complex matmul"""
    ph = np.angle(hilbert(xb, axis=-1))
    u = np.exp(1j * ph)
    T = xb.shape[-1]
    M = np.einsum("nct,ndt->ncd", u, u.conj()) / T
    return _upper(np.abs(M))


def fbc_msc(xb, lo, hi):
    """band-averaged magnitude-squared coherence.

    Coherence needs averaging: with ONE window and no averaging, |S_xy|^2/(S_xx S_yy) is
    identically 1 at every frequency and carries nothing. Averaging over the band's rfft bins
    (equivalent to Welch over frequency) is what makes it informative."""
    X = np.fft.rfft(xb, axis=-1)
    f = np.fft.rfftfreq(xb.shape[-1], 1.0 / SFREQ)
    k = (f >= lo) & (f <= hi)
    if k.sum() < 2:
        raise ValueError(f"band {lo}-{hi} Hz covers {k.sum()} rfft bins; coherence needs >= 2")
    Xb = X[:, :, k]
    S = np.einsum("ncf,ndf->ncd", Xb, Xb.conj())
    d = np.real(np.einsum("ncf,ncf->nc", Xb, Xb.conj())) + 1e-20
    return _upper(np.abs(S) ** 2 / (d[:, :, None] * d[:, None, :]))


def fbc_mi(xb, bins=5, decim=4):
    """binned plug-in mutual information for every channel pair, as ONE matmul per window.

    One-hot each channel into `bins` equal-count bins -> O of shape (C*bins, T); then
    O @ O.T / T is a (C*bins, C*bins) matrix whose (i, j) block of size bins x bins is the joint
    distribution of channels i and j. That turns ~C^2/2 histograms into a single BLAS call.
    `decim` subsamples time first: the plug-in estimator is biased upward when T/bins^2 is small,
    and the bias is what a classifier would otherwise latch onto."""
    x = xb[:, :, ::decim]
    N, C, T = x.shape
    # equal-count (quantile) binning per window and channel, so the marginals are flat by
    # construction and the MI comes from the joint structure rather than from differing marginals
    q = np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1], axis=-1)        # (bins-1, N, C)
    idx = (x[None, :, :, :] > q[:, :, :, None]).sum(0)                     # (N, C, T) in 0..bins-1
    out = np.empty((N, C * (C - 1) // 2))
    ii, jj = np.triu_indices(C, k=1)
    eye = np.eye(bins)
    for n in range(N):
        O = eye[idx[n]].transpose(0, 2, 1).reshape(C * bins, T)            # (C*bins, T)
        J = (O @ O.T) / T
        J = J.reshape(C, bins, C, bins)
        # exact marginals from the diagonal blocks (a block's row sums are p_i)
        marg = np.einsum("ibic->ib", J)
        Jp = J[ii, :, jj, :]                                               # (P, bins, bins)
        mi = Jp * (np.log(Jp + 1e-12)
                   - np.log(marg[ii][:, :, None] + 1e-12)
                   - np.log(marg[jj][:, None, :] + 1e-12))
        out[n] = mi.sum((1, 2))
    return out


def psd_band(xb):
    """log band power per channel -- the univariate baseline their FBC is claimed to beat"""
    z = xb - xb.mean(-1, keepdims=True)
    return np.log(np.einsum("nct,nct->nc", z, z) / xb.shape[-1] + 1e-12)


def blocks_from_windows(x, trial_ids, step_s=1.0, sfreq=SFREQ):
    """Rebuild each n-back block's continuous signal from its overlapping windows.

    The preprocessing saves 10 s windows with a 1 s step, so window w of a block starts 1 s after
    window w-1 and the first `step_s` seconds of consecutive windows tile the block exactly. The
    block is therefore the concatenation of each window's first second, plus the whole last window
    to recover the tail: for 35 windows that is 34 s + 10 s = 44 s, which is the 40 s scale the
    paper's own window-size analysis reports as the EEG optimum (its Fig. 3B), against the 10 s
    this project's model uses.

    Returns (blocks, keep_index) where blocks is (n_blocks, C, T) truncated to the shortest block
    so the feature dimension is fixed, and keep_index gives, per block, the index of its first
    window so labels and session ids can be taken from the window arrays."""
    stepn = int(round(step_s * sfreq))
    out, first = [], []
    for t in np.unique(trial_ids):
        idx = np.flatnonzero(trial_ids == t)
        if len(idx) < 2:
            continue
        idx = idx[np.argsort(idx)]                 # windows are saved in time order within a trial
        head = [x[i][:, :stepn] for i in idx[:-1]]
        seg = np.concatenate(head + [x[idx[-1]]], axis=-1)
        out.append(seg)
        first.append(idx[0])
    if not out:
        raise ValueError("no trial had at least 2 windows")
    T = min(o.shape[-1] for o in out)
    return np.stack([o[:, :T] for o in out]), np.array(first)


def features_from_band(xb, method, lo, hi, mi_bins=5, mi_decim=4):
    """the per-method feature, given the ALREADY band-filtered signal. Split out from features()
    so the main loop filters once per band instead of once per (band, method): filtfilt on
    (850, 28, 2000) dominates the cost of pcc and plv."""
    if method == "pcc":
        return fbc_pcc(xb)
    if method == "plv":
        return fbc_plv(xb)
    if method == "msc":
        return fbc_msc(xb, lo, hi)
    if method == "mi":
        return fbc_mi(xb, mi_bins, mi_decim)
    if method == "psd":
        return psd_band(xb)
    raise ValueError(method)


def features(x, method, band, mi_bins=5, mi_decim=4):
    lo, hi = BANDS[band]
    xb = bandpass(x, lo, hi)
    if method == "pcc":
        return fbc_pcc(xb)
    if method == "plv":
        return fbc_plv(xb)
    if method == "msc":
        return fbc_msc(xb, lo, hi)
    if method == "mi":
        return fbc_mi(xb, mi_bins, mi_decim)
    if method == "psd":
        return psd_band(xb)
    raise ValueError(method)


def loso(X, y, ses, classes=None, topk=10, clf="svm"):
    """leave-one-session-out. ANOVA feature selection and scaling are fitted on the TRAINING
    sessions only -- the difference from the paper, and the reason these numbers are lower."""
    accs = []
    for s in np.unique(ses):
        tr, te = ses != s, ses == s
        if classes is not None:
            tr, te = tr & np.isin(y, classes), te & np.isin(y, classes)
        if te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X[tr], y[tr]
        if topk and topk < Xtr.shape[1]:
            groups = [Xtr[ytr == c] for c in np.unique(ytr)]
            with np.errstate(invalid="ignore", divide="ignore"):
                F, _p = f_oneway(*groups, axis=0)
            F = np.nan_to_num(F, nan=-np.inf)
            keep = np.argsort(F)[::-1][:topk]
        else:
            keep = np.arange(Xtr.shape[1])
        model = (SVC(kernel="rbf", C=1.0, gamma="scale") if clf == "svm"
                 else LogisticRegression(C=0.1, max_iter=3000))
        pipe = make_pipeline(StandardScaler(), model)
        pipe.fit(Xtr[:, keep], ytr)
        accs.append(float((pipe.predict(X[te][:, keep]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def holm(ps):
    ps = np.asarray(ps, dtype=float); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    ap.add_argument("--topk", type=int, default=10, help="features kept, ANOVA on train only (0 = all)")
    ap.add_argument("--clf", default="svm", choices=["svm", "logreg"])
    ap.add_argument("--mi-bins", type=int, default=5)
    ap.add_argument("--mi-decim", type=int, default=4)
    ap.add_argument("--level", default="window", choices=["window", "block"],
                    help="window (default): one feature vector per 10 s window, this project's "
                         "unit. block: one per 40 s n-back block, rebuilt from its windows, which "
                         "is the scale the paper measured at -- but then each fold has only 18 "
                         "training blocks, so the ANOVA selection is fitted on 18 samples and "
                         "both the selection and the accuracy are far noisier.")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.data:
        sys.exit("need --data (the preprocessed root)")

    combos = [(m, b) for m in a.methods for b in a.bands]
    hdr = "  ".join(f"{m}:{b[:3]}".rjust(10) for m, b in combos)
    print(f"{'subject':8} {'n':>7}  2-VS-3: {hdr}   | 3-class: {hdr}")
    rows23, rows3 = {c: [] for c in combos}, {c: [] for c in combos}
    subs = []
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        f = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(f):
            continue
        x = np.asarray(np.load(f, mmap_mode="r"), dtype=np.float64)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        if a.level == "block":
            tri = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
            x, keep = blocks_from_windows(x, tri)
            y, ses = y[keep], ses[keep]
            # every window of a block carries the block's label, so taking the first window's is
            # exact -- verified rather than assumed
            for b, i0 in enumerate(keep):
                m = tri == tri[i0]
                if len(np.unique(np.asarray(y_all := np.load(os.path.join(d, f"{sid}_labels.npy"))[m]))) != 1:
                    raise SystemExit(f"{sid}: block {b} has mixed labels {np.unique(y_all)}")
        r23, r3 = {}, {}
        for band in a.bands:                      # filter once per band, reuse for every method
            lo, hi = BANDS[band]
            xb = bandpass(x, lo, hi)
            for m in a.methods:
                c = (m, band)
                if c not in r23 and c in rows23:
                    F = features_from_band(xb, m, lo, hi, a.mi_bins, a.mi_decim)
                    r23[c] = loso(F, y, ses, (1, 2), a.topk, a.clf)
                    r3[c] = loso(F, y, ses, None, a.topk, a.clf)
                    rows23[c].append(r23[c]); rows3[c].append(r3[c])
            del xb
        subs.append(sid)
        print(f"{sid:8} [{len(y):5d}w] " + " " * 9
              + "  ".join(f"{r23[c]:10.3f}" for c in combos) + "   | " + " " * 10
              + "  ".join(f"{r3[c]:10.3f}" for c in combos), flush=True)
    if not subs:
        sys.exit(f"no subject under {a.data} has {'{sid}'}_eeg_channel_windows.npy")

    print(f"\nMEAN over {len(subs)} subjects   (classifier: {a.clf}, top-{a.topk} by ANOVA on the "
          f"training sessions only)")
    print(f"  {'method':8} {'band':7} {'2-VS-3':>8} {'3-class':>8}")
    for c in combos:
        print(f"  {c[0]:8} {c[1]:7} {np.mean(rows23[c]):8.4f} {np.mean(rows3[c]):8.4f}")

    best_psd = max((c for c in combos if c[0] == "psd"), key=lambda c: np.mean(rows23[c]), default=None)
    if best_psd and len(combos) > 1:
        print(f"\n2-VS-3, paired against the best PSD band ({best_psd[1]}, "
              f"{np.mean(rows23[best_psd]):.4f}) -- their claim is FBC beats PSD by ~5 points:")
        fbc = [c for c in combos if c[0] != "psd"]
        ps = []
        for c in fbc:
            d = np.array(rows23[c]) - np.array(rows23[best_psd])
            ps.append(1.0 if np.allclose(d, 0) else
                      float(wilcoxon(rows23[best_psd], rows23[c]).pvalue))
        adj = holm(ps)
        for c, p, q in zip(fbc, ps, adj):
            d = np.array(rows23[c]) - np.array(rows23[best_psd])
            print(f"  {c[0]:4} {c[1]:6} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  "
                  f"p={p:.4f}  Holm={q:.4f}")
    print("\nReference points for 2-vs-3 in this project: "
          + ", ".join(f"{k} {v}" for k, v in REF.items()))
    print("A family has to beat the deep model (0.5996) to be worth a GPU run. The paper's own best")
    print("2-vs-3 is 62% (FBC alpha) and its EEG+fNIRS hybrid is 59%, both measured with feature")
    print("selection done on all data and 5-fold CV pooled over sessions and subjects, so they are")
    print("upper bounds, not targets. Volume conduction inflates every estimate here.")


def self_test():
    """Plant coupling that lives ONLY in the phase relationship, built in the FREQUENCY domain so
    that band power is EXACTLY equal in every class and PSD provably carries nothing.

    Every channel gets unit magnitude at every in-band bin, so the band power of every channel in
    every window is identical; the class sets only how tightly channel 1's phase tracks channel
    0's. An earlier version of this test mixed two time-domain signals instead, which leaked:
    mixing narrows the SPREAD of the sample variance across windows even with the mean unchanged,
    and PSD scored 0.43-0.46 against a 0.333 chance level. The planted-data artifact, not the
    estimators, produced that -- hence this construction."""
    rng = np.random.RandomState(0)
    C, T, N = 8, 1000, 180
    ses = np.repeat([1, 2, 3], N // 3)
    y = np.tile([0, 1, 2], N // 3)
    f = np.fft.rfftfreq(T, 1.0 / SFREQ)
    inb = (f >= BANDS["alpha"][0]) & (f <= BANDS["alpha"][1])
    nb = int(inb.sum())
    x = np.zeros((N, C, T))
    for i_ in range(N):
        kappa = (0.0, 1.5, 12.0)[y[i_]]              # class = phase-locking concentration
        X = np.zeros((C, len(f)), dtype=complex)
        base = rng.uniform(-np.pi, np.pi, nb)
        for c in range(C):
            if c == 1:
                ph = base + rng.vonmises(0.0, kappa, nb)     # tracks channel 0
            else:
                ph = rng.uniform(-np.pi, np.pi, nb)          # independent
            if c == 0:
                ph = base
            X[c, inb] = np.exp(1j * ph)                      # |X| == 1 for EVERY channel/class
        x[i_] = np.fft.irfft(X, n=T, axis=-1)
    # band power is identical across classes by construction -- verify, do not assume
    pw = psd_band(bandpass(x, *BANDS["alpha"]))
    spread = max(abs(pw[y == a].mean() - pw[y == b].mean()) for a in range(3) for b in range(3))
    assert spread < 0.02, f"planted data leaks into band power (max class gap {spread:.4f})"

    acc_psd = loso(features(x, "psd", "alpha"), y, ses, topk=0, clf="logreg")
    accs = {m: loso(features(x, m, "alpha"), y, ses, topk=0, clf="logreg")
            for m in ("pcc", "mi", "msc", "plv")}
    print(f"[planted phase-coupling only] psd {acc_psd:.3f} | "
          + " | ".join(f"{m} {v:.3f}" for m, v in accs.items()) + "   (3-class, chance 0.333)")
    ok = acc_psd < 0.45 and accs["plv"] > 0.70 and accs["pcc"] > 0.60

    # bounds and non-degeneracy
    msc = features(x, "msc", "alpha")
    assert msc.std() > 1e-6 and (msc <= 1.0 + 1e-9).all() and (msc >= -1e-9).all(), "MSC degenerate"
    plv = features(x, "plv", "alpha")
    assert (plv <= 1.0 + 1e-9).all() and (plv >= -1e-9).all(), "PLV out of [0,1]"
    assert (features(x, "mi", "alpha") >= -1e-9).all(), "MI negative"
    # the one-matmul MI must agree with per-pair histograms on the same binning
    xb_full = bandpass(x[:1], *BANDS["alpha"])
    xb = xb_full[:, :, ::4]
    q = np.quantile(xb[0], np.linspace(0, 1, 6)[1:-1], axis=-1)
    idx = (xb[0][None] > q[:, :, None]).sum(0)
    direct = []
    for a_ in range(C):
        for b_ in range(a_ + 1, C):
            J = np.histogram2d(idx[a_], idx[b_], bins=[range(6), range(6)])[0] / idx.shape[1]
            pi, pj = J.sum(1), J.sum(0)
            direct.append((J * (np.log(J + 1e-12) - np.log(pi[:, None] + 1e-12)
                                - np.log(pj[None, :] + 1e-12))).sum())
    assert np.allclose(fbc_mi(xb_full, 5, 4)[0], direct, atol=1e-9), \
        "the one-matmul MI disagrees with per-pair histograms"
    # feature counts: C*(C-1)/2 pairs for the connectivity families, C for psd
    for m in ("pcc", "mi", "msc", "plv"):
        assert features(x, m, "alpha").shape[1] == C * (C - 1) // 2
    assert features(x, "psd", "alpha").shape[1] == C
    # ANOVA selection must never see the held-out session: a feature that separates the classes
    # ONLY in session 3 must not be picked when session 3 is the test fold
    Xl = np.zeros((N, 4)); Xl[ses == 3, 0] = y[ses == 3] * 10.0
    leak = loso(Xl, y, ses, topk=1, clf="logreg")
    assert leak < 0.45, f"selection leaked the test session (acc {leak:.3f})"
    print("MI matmul == per-pair histograms; MSC and PLV in [0,1]; MI >= 0; feature counts right;"
          f" a session-3-only feature does not get selected when session 3 is held out ({leak:.3f})")
    # block reconstruction must return the ORIGINAL continuous signal, not a reshuffle of it
    C2, T2, win, stepn = 4, 200 * 10, 10, 200
    cont = np.random.RandomState(1).randn(C2, 200 * 44)          # 44 s at 200 Hz
    nwin = (cont.shape[-1] - win * 200) // stepn + 1
    wins = np.stack([cont[:, k * stepn:k * stepn + win * 200] for k in range(nwin)])
    tri = np.zeros(nwin, dtype=int)
    blk, keep = blocks_from_windows(wins, tri)
    assert keep.tolist() == [0] and blk.shape[0] == 1
    assert np.allclose(blk[0], cont[:, :blk.shape[-1]]), "block reconstruction is not the original"
    # two blocks -> two rows, truncated to the shorter one, first-window index per block
    tri2 = np.array([0] * nwin + [1] * (nwin - 3))
    wins2 = np.concatenate([wins, wins[:nwin - 3]])
    blk2, keep2 = blocks_from_windows(wins2, tri2)
    assert blk2.shape[0] == 2 and keep2.tolist() == [0, nwin]
    assert blk2.shape[-1] == min(200 * 44, (nwin - 4) * stepn + win * 200)
    print(f"block reconstruction exact over {blk.shape[-1] / 200:.0f} s; two blocks truncate to "
          f"{blk2.shape[-1] / 200:.0f} s with first-window indices {keep2.tolist()}")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
