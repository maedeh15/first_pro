"""
check_riemannian_eeg.py -- cheap CPU probe: EEG-only Riemannian tangent-space features with
per-session (domain-specific) re-centering + logistic regression, cross-session LOSO.

Motivation (the two papers you sent + TSMNet):
  * Roy et al. 2022 (passive-BCI cross-session workload competition): the best pipelines were
    Riemannian covariance methods; deep models overfit to chance.
  * Kobler et al. (TSMNet / SPDDSMBN): normalise each DOMAIN on the SPD manifold -- transport its
    Frechet mean to the identity and (optionally) rescale its dispersion (Eq. 10:
    Z~ = Gamma_{I->G_phi}( Gamma_{G_k->I}(Z)^{nu_phi / (nu_k+eps)} ), here G_phi = I, nu_phi = 1).
  Here a "domain" is one recording session. This is the closed-form, non-learned part only.

Feature sets (all on eeg_channel_windows.npy, N x 28 x T, same LOSO + StandardScaler + LR(C=0.1) as
check_feature_baseline_tsfnet.py, so the numbers are comparable with its EEG row):
  logvar        log-variance per channel                                  (reference probe row)
  tan_raw       tangent vector at the identity of the window covariance, no domain handling
  tan_EA        arithmetic mean-covariance whitening per session  (Euclidean Alignment, Wu 2020)
  tan_riem      Frechet-mean (affine-invariant) re-centering per session
  tan_riem_disp tan_riem + per-session dispersion normalisation (tangent vectors / nu_session)
each on  broadband (the stored 0.5-50 Hz signal)  and on  multiband (theta 4-8, alpha 8-13, beta
13-30 Hz and broadband, tangent vectors concatenated).
Metrics: 3-class LOSO accuracy and 2-vs-3 LOSO (train and test on 2-/3-back only); paired
Wilcoxon over subjects against logvar and against tan_raw.

Transductive like E3 / AdaBN / EA: the held-out session's UNLABELED windows define its own centre.
It uses no labels and no other session's statistics.

HONEST LIMITS (printed again at the end): linear probe, not the deep model; every earlier probe in
this project (grid alignment, covariance alignment, neurovascular coupling, band ratios) was MORE
optimistic than the deep model; the deep EEG head sits at ~0.487-0.50 while the log-variance probe is
already ~0.558, so beating the head is easy and beating the probe is the real test.

usage:
  python check_riemannian_eeg.py --self-test
  python check_riemannian_eeg.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import os
import argparse
import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
FS = 200.0
RIDGE = 1e-3


# ----------------------------------------------------------------------------- SPD maths
_EIGH_DEVICE = None      # None = numpy, exactly what every number in this project was computed with


def set_eigh_backend(device=None):
    """Route the eigendecomposition through torch on `device` instead of numpy.

    Every probe here spends almost all of its time in np.linalg.eigh: measured, 74 s of the 80 s a
    26-channel 4-band subject takes goes into 6800 decompositions of 28x28 matrices, and the einsums
    around them cost 0.3 s. numpy runs those one at a time; a GPU does the batch at once. torch on
    the CPU is not faster (85 s measured), so this is only worth passing a real device.

    float64 is forced: the tangent vector is a log of eigenvalues, and in float32 the small
    eigenvalues -- which is where the ridge lives -- lose the precision the features depend on.

    device=None restores numpy. The numpy path stays the default everywhere so published numbers are
    reproduced bit for bit; self_test() checks the two paths agree.
    """
    global _EIGH_DEVICE
    if device is None:
        _EIGH_DEVICE = None
        return "numpy"
    import torch
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda asked for, but torch.cuda.is_available() is False")
    _EIGH_DEVICE = dev
    return str(dev)


def _eigh(C):
    if _EIGH_DEVICE is None:
        return np.linalg.eigh(C)
    import torch
    t = torch.as_tensor(np.ascontiguousarray(C), dtype=torch.float64, device=_EIGH_DEVICE)
    w, v = torch.linalg.eigh(t)
    return w.cpu().numpy(), v.cpu().numpy()


def _eigfun(C, f, clamp=True):
    """apply f to the eigenvalues of a symmetric matrix. clamp=True floors eigenvalues at 1e-12 (for
    SPD inputs: log, sqrt, inverse sqrt); clamp=False is REQUIRED for exp of a symmetric matrix whose
    eigenvalues are legitimately negative (the Karcher update)."""
    w, v = _eigh(C)
    return np.einsum("...ij,...j,...kj->...ik", v, f(np.maximum(w, 1e-12) if clamp else w), v)


def covs(x, ridge=RIDGE):
    """x (N,C,T) -> (N,C,C) SPD covariances (window-mean removed, ridge on the trace)"""
    x = x - x.mean(-1, keepdims=True)
    c = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
    tr = np.trace(c, axis1=1, axis2=2)[:, None, None] / c.shape[1]
    return c + ridge * tr * np.eye(c.shape[1])[None]


def karcher_mean(C, iters=30, tol=1e-9):
    """affine-invariant Frechet mean of SPD matrices C (N,d,d)"""
    M = C.mean(0)
    for _ in range(iters):
        Mh = _eigfun(M, np.sqrt)
        Mi = _eigfun(M, lambda w: w ** -0.5)
        S = _eigfun(np.einsum("ij,njk,kl->nil", Mi, C, Mi), np.log).mean(0)
        M = Mh @ _eigfun(S, np.exp, clamp=False) @ Mh
        if np.linalg.norm(S) < tol:
            break
    return M


def whiten(C, M):
    Mi = _eigfun(M, lambda w: w ** -0.5)
    return np.einsum("ij,njk,kl->nil", Mi, C, Mi)


def tangent_vec(C):
    """log map at the identity, isometric vectorisation (off-diagonals * sqrt(2))"""
    L = _eigfun(C, np.log)
    d = C.shape[1]
    iu = np.triu_indices(d)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))
    return L[:, iu[0], iu[1]] * w


def domain_features(C, ses, mode):
    """mode in {raw, EA, riem, riem_disp}; returns tangent features (N, d(d+1)/2)"""
    out = None
    for s in np.unique(ses):
        m = ses == s
        Cs = C[m]
        if mode == "raw":
            T = tangent_vec(Cs)
        elif mode == "EA":
            T = tangent_vec(whiten(Cs, Cs.mean(0)))
        else:
            M = karcher_mean(Cs)
            Cw = whiten(Cs, M)
            T = tangent_vec(Cw)
            if mode == "riem_disp":
                nu = np.sqrt(np.mean(np.sum(_eigfun(Cw, np.log) ** 2, axis=(1, 2))))   # Frechet std
                T = T / (nu + 1e-8)
        if out is None:
            out = np.empty((len(C), T.shape[1]))
        out[m] = T
    return out


# ----------------------------------------------------------------------------- probe
def bandpass(x, lo, hi):
    sos = butter(4, [lo, hi], btype="band", fs=FS, output="sos")
    return sosfiltfilt(sos, x, axis=-1)


def build_covs(x, multiband):
    out = {"broad": covs(x)}
    if multiband:
        for name, (lo, hi) in BANDS.items():
            out[name] = covs(bandpass(x, lo, hi))
    return out


def loso(X, y, ses, classes=None):
    accs = []
    for s in np.unique(ses):
        tr, te = ses != s, ses == s
        if classes is not None:
            tr = tr & np.isin(y, classes)
            te = te & np.isin(y, classes)
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else float("nan")


MODES = ("raw", "EA", "riem", "riem_disp")


def evaluate(x, y, ses, multiband=True):
    cov_bb = covs(x)
    lv = np.log(np.einsum("nct,nct->nc", x - x.mean(-1, keepdims=True), x - x.mean(-1, keepdims=True)) / x.shape[-1] + 1e-8)
    res = {"logvar": (loso(lv, y, ses), loso(lv, y, ses, (1, 2)))}
    bands = build_covs(x, multiband)
    for mode in MODES:
        Xb = domain_features(cov_bb, ses, mode)
        res[f"{mode}|broad"] = (loso(Xb, y, ses), loso(Xb, y, ses, (1, 2)))
        if multiband:
            Xm = np.hstack([domain_features(bands[b], ses, mode) for b in bands])
            res[f"{mode}|multi"] = (loso(Xm, y, ses), loso(Xm, y, ses, (1, 2)))
    return res


# ----------------------------------------------------------------------------- self-test
def _synth(rng, gain_sd, n=40, C=8, T=400):
    L = rng.randn(C, C) * 0.3 + np.eye(C)
    X, y, ses = [], [], []
    for s in (1, 2, 3):
        A = np.diag(np.exp(gain_sd * rng.randn(C))) @ (np.eye(C) + 0.1 * rng.randn(C, C))
        for i in range(3 * n):
            cls = i % 3
            z = rng.randn(C, T)
            z[cls] *= 2.2                                   # class: variance of one latent source
            X.append(A @ L @ z)
            y.append(cls)
            ses.append(s)
    return np.array(X), np.array(y), np.array(ses)


def self_test():
    print("planted session-specific per-source gains (+ mild mixing); class = variance of one source")
    rng = np.random.RandomState(0)
    rows = {"none": [], "shifted": []}
    for rep in range(6):
        for lab, gsd in (("none", 0.0), ("shifted", 0.8)):
            X, y, ses = _synth(rng, gsd)
            rows[lab].append(evaluate(X, y, ses, multiband=False))
    ok = True
    for lab in rows:
        print(f"  [{lab:7}] " + "  ".join(
            f"{k.split('|')[0]} {np.mean([r[k][0] for r in rows[lab]]):.3f}"
            for k in ("logvar", "raw|broad", "EA|broad", "riem|broad", "riem_disp|broad")))
    g = lambda lab, k: np.mean([r[k][0] for r in rows[lab]])
    good = g("shifted", "riem|broad") > g("shifted", "raw|broad") + 0.15
    print(f"  shifted: riem {g('shifted', 'riem|broad'):.3f} vs raw {g('shifted', 'raw|broad'):.3f} "
          f"(want riem > raw + 0.15) -> {'OK' if good else 'FAIL'}")
    ok &= good
    # Karcher mean sanity: for commuting matrices it equals the geometric mean
    d = np.array([[2.0, 0.0], [0.0, 1.0]])
    Cs = np.stack([d, np.diag([8.0, 4.0]), np.diag([0.5, 0.25])])
    M = karcher_mean(Cs)
    expect = np.diag([np.exp(np.log([2, 8, 0.5]).mean()), np.exp(np.log([1, 4, 0.25]).mean())])
    good = np.allclose(M, expect, atol=1e-6)
    print(f"  Karcher mean of commuting matrices == geometric mean -> {'OK' if good else 'FAIL'}")
    ok &= good
    # after riemannian whitening the Frechet mean must be the identity
    W = whiten(Cs, M)
    good = np.allclose(karcher_mean(W), np.eye(2), atol=1e-6)
    print(f"  Frechet mean after recentering == I -> {'OK' if good else 'FAIL'}")
    ok &= good
    # the torch backend must not move the features. Checked against torch on the CPU, which is the
    # same code path as a GPU run bar the device; a GPU run also prints this check.
    try:
        import torch  # noqa: F401
        rng2 = np.random.RandomState(7)
        xb = rng2.randn(60, 10, 300)
        sesb = np.repeat([1, 2, 3], 20)
        ref = {k: domain_features(build_covs(xb, True)[k], sesb, "riem_disp") for k in ("broad", "alpha")}
        set_eigh_backend("cpu")
        try:
            got = {k: domain_features(build_covs(xb, True)[k], sesb, "riem_disp") for k in ("broad", "alpha")}
        finally:
            set_eigh_backend(None)
        dmax = max(np.abs(ref[k] - got[k]).max() for k in ref)
        good = dmax < 1e-8
        print(f"[backend] torch path reproduces the numpy features, max diff {dmax:.2e} -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
    except ImportError:
        print("[backend] torch not installed -- numpy path only, skipped")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-multiband", action="store_true", help="broadband covariance only (faster)")
    ap.add_argument("--device", default=None,
                    help="route the eigendecompositions through torch on this device (e.g. cuda). "
                         "Default: numpy, which is what every published number here used.")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    print(f"eigendecomposition backend: {set_eigh_backend(a.device)}")
    rows, ids = [], []
    keys = None
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        p = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(p):
            continue
        x = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        r = evaluate(x, y, ses, multiband=not a.no_multiband)
        keys = list(r)
        rows.append(r)
        ids.append(sid)
        print(f"{sid}  logvar {r['logvar'][0]:.3f} | tan_raw {r['raw|broad'][0]:.3f} | EA {r['EA|broad'][0]:.3f} | "
              f"riem {r['riem|broad'][0]:.3f} | riem+disp {r['riem_disp|broad'][0]:.3f}"
              + (f" || multi: raw {r['raw|multi'][0]:.3f} riem {r['riem|multi'][0]:.3f} "
                 f"riem+disp {r['riem_disp|multi'][0]:.3f}" if not a.no_multiband else ""), flush=True)
    if not rows:
        raise SystemExit("no subjects found (need *_eeg_channel_windows.npy)")
    print("\nMEAN over %d subjects (chance 3-class 0.333, 2v3 0.500)" % len(rows))
    print(f"{'features':22} {'3-class':>8} {'2-vs-3':>8} | {'d3 vs logvar':>13} {'p':>7} | {'d3 vs raw':>10} {'p':>7} | wins vs logvar")
    base = np.array([r["logvar"][0] for r in rows])
    for k in keys:
        v3 = np.array([r[k][0] for r in rows])
        v2 = np.array([r[k][1] for r in rows])
        dl = v3 - base
        pl = wilcoxon(dl).pvalue if np.any(dl != 0) else 1.0
        rawk = "raw|" + k.split("|")[1] if "|" in k else None
        if rawk and rawk in keys and k != rawk:
            dr = v3 - np.array([r[rawk][0] for r in rows])
            pr = wilcoxon(dr).pvalue if np.any(dr != 0) else 1.0
            rtxt = f"{dr.mean():+10.4f} {pr:7.4f}"
        else:
            rtxt = f"{'-':>10} {'-':>7}"
        wl = f"{int((dl > 0).sum())}/{len(dl)}" if k != "logvar" else "-"
        print(f"{k:22} {v3.mean():8.4f} {v2.mean():8.4f} | {dl.mean():+13.4f} {pl:7.4f} | {rtxt} | {wl}")
    print("\nReference points: deep EEG head ~0.487-0.50 (3-class); log-variance probe ~0.558; "
          "earlier covariance+EA probe 0.585 (trace-normalised covariance, broadband).")
    print("Limits: linear probe, not the deep model; transductive per-session centring; probes in this "
          "project have repeatedly over-predicted deep-model gains. Beating the log-variance probe is the "
          "bar; beating the deep head only shows the head is weak.")


if __name__ == "__main__":
    main()
