"""
check_covariance_alignment.py -- cheap, parameter-free pre-screen of idea #4
(DGIFE, Jin et al. 2026: covariance alignment across domains), EEG only.

Question: is there cross-session COVARIANCE shift in the EEG channel data,
and does closed-form per-session alignment (Euclidean Alignment: whiten every
window by the session's mean covariance^-1/2, unlabeled, transductive like E3)
recover cross-session accuracy?  No learned parameters, no GPU.

Feature sets (same cross-session LOSO + logistic regression as
check_feature_baseline_tsfnet.py):
  logvar : log-variance per channel                  (the baseline EEG row)
  cov    : log-Euclidean upper-triangle of window covariance, raw
  cov+EA : same after per-session Euclidean Alignment
Also reports shift = mean pairwise Frobenius distance between per-session
mean covariances (the DGIFE Lalign quantity, on normalised covariances).

usage:
  python check_covariance_alignment.py --self-test
  python check_covariance_alignment.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import os, argparse
import numpy as np
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REG = 1e-3


def covs(x):                                   # x (N,C,T) -> (N,C,C), trace-normalised
    x = x - x.mean(-1, keepdims=True)
    c = np.einsum("nct,ndt->ncd", x, x) / x.shape[-1]
    c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
    return c + REG * np.eye(c.shape[1]) / c.shape[1]


def _fn(c, f):
    w, v = np.linalg.eigh(c)
    return np.einsum("...ij,...j,...kj->...ik", v, f(np.maximum(w, 1e-12)), v)


def align(c, ses):
    out = np.empty_like(c)
    for s in np.unique(ses):
        m = ses == s
        r = _fn(c[m].mean(0), lambda w: w ** -0.5)
        out[m] = r @ c[m] @ r
    return out


def tangent(c):
    l = _fn(c, np.log)
    iu = np.triu_indices(c.shape[1])
    return l[:, iu[0], iu[1]]


def shift(c, ses):
    ms = [c[ses == s].mean(0) for s in np.unique(ses)]
    d = [np.linalg.norm(ms[i] - ms[j]) for i in range(len(ms)) for j in range(i + 1, len(ms))]
    return float(np.mean(d))


def loso(X, y, ses, cls=None):
    a = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        if not te.sum() or not tr.sum():
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        mask = np.ones(te.sum(), bool) if cls is None else np.isin(y[te], cls)
        a.append((p[mask] == y[te][mask]).mean())
    return float(np.mean(a))


def evaluate(x, y, ses):
    c = covs(x.astype(np.float64))
    lv = np.log(np.einsum("nct,nct->nc", x, x) / x.shape[-1] + 1e-8)
    raw, ea = tangent(c), tangent(align(c, ses))
    return dict(logvar=loso(lv, y, ses), cov=loso(raw, y, ses), covEA=loso(ea, y, ses),
                shift_raw=shift(c, ses), shift_EA=shift(align(c, ses), ses))


def synth(seed, shifted, n=90, C=12, T=400, mix_sd=0.0):
    r = np.random.RandomState(seed)
    y = np.tile(np.arange(3), n // 3 * 3)[: n * 3 // 3 * 1] if False else r.randint(0, 3, 3 * n)
    ses = np.repeat([1, 2, 3], n)
    base = r.randn(C, C) * 0.3 + np.eye(C)
    x = np.empty((3 * n, C, T))
    mixes = {s: np.eye(C) + mix_sd * r.randn(C, C) for s in (1, 2, 3)}
    for i in range(3 * n):
        z = r.randn(C, T)
        z[y[i] % C] *= 1.0                         # class effect: covariance coupling
        a = 0.6 * (y[i] - 1)
        z[0] += a * z[1]
        x[i] = (mixes[ses[i]] if shifted else np.eye(C)) @ base @ z
    return x, y, ses


def self_test():
    print("planted session-specific channel mixing (EA should help), and none (EA ~ neutral):")
    for sd, lab in ((0.0, "no shift"), (0.35, "session mixing")):
        rs = [evaluate(*synth(s, sd > 0, mix_sd=sd)) for s in range(8)]
        for k in ("logvar", "cov", "covEA"):
            print(f"  [{lab:14}] {k:7} {np.mean([r[k] for r in rs]):.3f}")
        print(f"  [{lab:14}] shift raw {np.mean([r['shift_raw'] for r in rs]):.4f} -> EA {np.mean([r['shift_EA'] for r in rs]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    rows = []
    print(f"{'subject':8} {'logvar':>7} {'cov':>7} {'cov+EA':>7} {'shiftRaw':>9} {'shiftEA':>8}")
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        p = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(p):
            continue
        x = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float32)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        r = evaluate(x, y, ses)
        rows.append(r)
        print(f"{sid:8} {r['logvar']:7.3f} {r['cov']:7.3f} {r['covEA']:7.3f} {r['shift_raw']:9.4f} {r['shift_EA']:8.4f}", flush=True)
    if not rows:
        return print("no subjects found (need *_eeg_channel_windows.npy)")
    g = lambda k: np.array([r[k] for r in rows])
    print(f"MEAN     {g('logvar').mean():7.3f} {g('cov').mean():7.3f} {g('covEA').mean():7.3f}   (chance 0.333)")
    for a_, b_ in (("covEA", "cov"), ("covEA", "logvar")):
        d = g(a_) - g(b_)
        pv = wilcoxon(d).pvalue if np.any(d) else 1.0
        print(f"{a_} - {b_}: mean {d.mean():+.4f}  wins {(d>0).sum()}/{len(d)}  paired Wilcoxon p={pv:.4f}")
    rho, pr = spearmanr(g("shift_raw"), g("cov"))
    print(f"Spearman(shift_raw, cov acc) rho={rho:+.3f} p={pr:.3f}  (negative => shift hurts)")


if __name__ == "__main__":
    main()
