"""
member_nested_tuning.py -- can the closed-form EEG member itself be made better, without touching the
test sessions to choose how?  (CPU only, no network trained.)

Why this is the next test.  The member (per-session EA-recentred multiband covariance -> tangent vector ->
StandardScaler -> LogisticRegression(C=0.1)) is the component that carries the headline result of the
thesis (late fusion 0.671).  Its two knobs were never tuned: the band set (broad + theta + alpha + beta) and
C = 0.1 were fixed by hand.  Multiband itself was worth ~+4 points over broadband, and the in-model
weight-decay run suggests regularisation matters.  Improving the member improves the fused system.

How selection is done WITHOUT leakage.  For every outer fold (subject i, held-out session s) the choice of
(band set, C) is made by an INNER leave-one-session-out inside the two training sessions only: train on
one of them, validate on the other, and the other way round; pick the configuration with the best mean
inner accuracy (ties go to the earlier configuration in the list, i.e. the current member first).  The
held-out session is never read during selection (asserted).  Then the member is refitted on both training
sessions with the chosen configuration and scored on the held-out session.

A known weakness of that design, stated here because it biases the result downward rather than up: the
inner selection fits on ONE session (~295 windows) while the outer refit uses TWO (~590).  The best C for
half the data is not the best C for all of it -- it is systematically stronger -- so the selected C tends
to over-regularise the outer fit.  With 3 sessions there is no way to both select and refit at the same
training size.

PRE-REGISTERED reading (written before any real run):
  primary contrast : nested-selected member  minus  the current member (S0_current, C=0.1),
                     window accuracy, paired Wilcoxon over the 26 subjects
  counted as an improvement only if >= +1.0 pp with p < 0.05.
Everything else (the table of every fixed configuration, which configurations were chosen, block and
fused columns) is descriptive.  The "best fixed configuration in hindsight" row is optimistic by
construction and is printed only so it cannot be mistaken for a result.

usage
  python member_nested_tuning.py --self-test
  python member_nested_tuning.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" --cache-dir bandfeats ^
        [--posteriors-dir "E:\\...\\ensemble_pure_baseline_40b7d8ea53\\posteriors"] [--device cuda] ^
        [--bandsets S0_current S1_fine] [--Cs 0.03 0.1 0.3] [--out nested_folds.csv]
"""
import os
import csv
import argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import check_riemannian_eeg as R
import late_fusion_riemann as L
import build_riem_features as B

BANDS_ALL = {"broad": None, "theta": (4.0, 8.0), "alpha": (8.0, 13.0), "alpha_lo": (8.0, 10.0),
             "alpha_hi": (10.0, 13.0), "beta": (13.0, 30.0), "beta_lo": (13.0, 20.0),
             "beta_hi": (20.0, 30.0), "gamma": (30.0, 45.0)}
BAND_SETS = {
    "S0_current": ["broad", "theta", "alpha", "beta"],                 # the member used everywhere so far
    "S1_fine": ["broad", "theta", "alpha_lo", "alpha_hi", "beta_lo", "beta_hi"],
    "S2_plus_gamma": ["broad", "theta", "alpha", "beta", "gamma"],
    "S3_no_broad": ["theta", "alpha", "beta"],
}
CS = (0.1, 0.01, 0.03, 0.3, 1.0)            # 0.1 first: the current value wins ties
BASE = ("S0_current", 0.1)
MIN_GAIN = 0.01


def band_features(x, ses, name):
    lo_hi = BANDS_ALL[name]
    xs = x if lo_hi is None else R.bandpass(x, *lo_hi)
    return R.domain_features(R.covs(xs), ses, "EA").astype(np.float32)


def load_subject(data, sid, names, cache_dir=None):
    d = os.path.join(data, sid)
    pe = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
    if not os.path.exists(pe):
        return None
    y = np.load(os.path.join(d, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
    tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
    # The cache is keyed by subject only, so it has to carry a fingerprint of the window set it was
    # built from. Without one, re-running the preprocessing (new windowing, new bad-channel decision,
    # a different normalisation) leaves a cache that loads fine and is wrong, and every number
    # downstream is quietly wrong with it. This project has already been bitten by a stale artefact.
    fp = B.fingerprint(y, tid, ses)
    cp = os.path.join(cache_dir, f"{sid}_bands_EA.npz") if cache_dir else None
    have = {}
    if cp and os.path.exists(cp):
        z = np.load(cp)
        cached_fp = str(z["__fingerprint__"]) if "__fingerprint__" in z.files else None
        if cached_fp != fp:
            print(f"[cache] {sid}: fingerprint mismatch -- the cached features were built from a "
                  f"different window set. Recomputing (delete {cp} to silence this).", flush=True)
        else:
            have = {k: z[k] for k in z.files if k != "__fingerprint__" and len(z[k]) == len(y)}
    need = [n for n in names if n not in have]
    if need:
        x = np.asarray(np.load(pe, mmap_mode="r"), dtype=np.float64)
        for n in need:
            have[n] = band_features(x, ses, n)
        if cp:
            os.makedirs(cache_dir, exist_ok=True)
            np.savez(cp, __fingerprint__=np.array(fp), **have)
    print(f"[features] {sid}: {len(y)} windows, bands {sorted(set(names))}", flush=True)
    return {n: have[n] for n in names}, y, ses, tid


class _CfgView:
    """X[cfg] -> the feature matrix of cfg's band set. Keeps the call sites unchanged while storing
    one matrix per band set instead of one per (band set, C)."""

    def __init__(self, by_bandset):
        self._b = by_bandset

    def __getitem__(self, cfg):
        return self._b[cfg[0]]


def fit_pred(Xtr, ytr, Xte, C):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(Xtr, ytr)
    P = np.zeros((len(Xte), 3))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(Xte)
    return P


def assert_inner_clean(mask, te_mask):
    if np.any(mask & te_mask):
        raise AssertionError("LEAK: the inner selection touched the held-out session")


def select_config(X, y, ses, test_s, configs):
    """X: {config: (N, D)}; returns (best config, {config: inner score}); reads ONLY sessions != test_s"""
    te = ses == test_s
    tr_sessions = sorted(set(ses[~te]))
    scores = {}
    for cfg in configs:
        accs = []
        for v in tr_sessions:
            itr = (~te) & (ses != v)
            iva = (~te) & (ses == v)
            assert_inner_clean(itr, te)
            assert_inner_clean(iva, te)
            if len(np.unique(y[itr])) < 2 or iva.sum() == 0:
                continue
            P = fit_pred(X[cfg][itr], y[itr], X[cfg][iva], cfg[1])
            accs.append(float((P.argmax(1) == y[iva]).mean()))
        scores[cfg] = float(np.mean(accs)) if accs else float("-inf")
    best = configs[0]
    for cfg in configs:                       # strict '>' keeps the earlier configuration on ties
        if scores[cfg] > scores[best]:
            best = cfg
    return best, scores


def run_subject(sid, Fb, y, ses, tid, bandsets, Cs, posteriors_dir=None, w=0.5):
    configs = [(b, c) for b in bandsets for c in Cs]
    if BASE in configs:                       # make sure the current member is first
        configs.remove(BASE)
        configs.insert(0, BASE)
    # keyed by band set: C changes the classifier, not the features, so building one matrix per
    # (band set, C) duplicated every matrix five times over
    Xb = {b: np.hstack([Fb[n] for n in BAND_SETS[b]]) for b in {c[0] for c in configs}}
    X = _CfgView(Xb)
    rows = []
    for s in sorted(set(ses)):
        te = ses == s
        tr = ~te
        best, inner = select_config(X, y, ses, s, configs)
        outer = {cfg: fit_pred(X[cfg][tr], y[tr], X[cfg][te], cfg[1]) for cfg in configs}
        yte, tte = y[te], tid[te]
        Pd = None
        if posteriors_dir:
            p = os.path.join(posteriors_dir, f"{sid}_session{s}.npz")
            if os.path.exists(p):
                z = np.load(p)
                if np.array_equal(z["y"], yte) and np.array_equal(z["trial_ids"], tte):
                    Pd = L.norm_rows(z["P_final"].astype(np.float64))
        row = dict(subject=sid, session=int(s), n=int(te.sum()), selected=f"{best[0]}|C={best[1]}",
                   inner_best=inner[best])
        for name, cfg in (("fixed", BASE), ("nested", best)):
            if cfg not in outer:
                continue
            P = outer[cfg]
            row[f"{name}_win"] = L.acc_window(P, yte)
            row[f"{name}_tri"] = L.acc_trial(P, yte, tte)
            if Pd is not None:
                Pf = L.fuse(Pd.mean(0), P, w)
                row[f"{name}_fused_win"] = L.acc_window(Pf, yte)
                row[f"{name}_fused_tri"] = L.acc_trial(Pf, yte, tte)
        for cfg, P in outer.items():
            row[f"cfg_{cfg[0]}|C={cfg[1]}"] = L.acc_window(P, yte)
        rows.append(row)
    return rows


def per_subject(rows, key):
    subs = sorted({r["subject"] for r in rows})
    return subs, np.array([np.mean([r[key] for r in rows if r["subject"] == s]) for s in subs])


def summarise(rows):
    subs, fx = per_subject(rows, "fixed_win")
    _, nt = per_subject(rows, "nested_win")
    print(f"\n{len(subs)} subjects.  member window accuracy (mean over subjects)")
    print(f"  current member ({BASE[0]}, C={BASE[1]})  : {fx.mean():.4f}")
    print(f"  nested-selected member                 : {nt.mean():.4f}")
    d = nt - fx
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"  nested - current                       : {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}"
          f"   <== PRE-REGISTERED PRIMARY")
    for k, lab in (("tri", "block"), ("fused_win", "fused window"), ("fused_tri", "fused block")):
        if f"nested_{k}" in rows[0] and f"fixed_{k}" in rows[0]:
            _, a = per_subject(rows, f"nested_{k}")
            _, b = per_subject(rows, f"fixed_{k}")
            dd = a - b
            pp = wilcoxon(dd).pvalue if np.any(dd != 0) else 1.0
            print(f"  {lab:14} current {b.mean():.4f} -> nested {a.mean():.4f}   {dd.mean():+.4f}  "
                  f"wins {int((dd > 0).sum())}/{len(dd)}  p={pp:.4f}  (descriptive)")
    big = d.mean() >= MIN_GAIN
    sig = p < 0.05
    ok = big and sig
    print("\nPRE-REGISTERED READING (both conditions must hold):")
    print(f"  [{'PASS' if big else 'FAIL'}] mean difference >= {MIN_GAIN:+.4f}   (observed {d.mean():+.4f})")
    print(f"  [{'PASS' if sig else 'FAIL'}] p < 0.05                   (observed {p:.4f})")
    print("  -> " + ("improvement by the rule: adopt the nested-selected member and re-run the fusion "
                     "report with it" if ok else "NOT an improvement by the rule: keep the current "
                     "member."))
    if big and not sig:
        print(f"     note: the effect met the size threshold and missed only significance, so this is "
              f"'not established', not 'no effect'.")
    from collections import Counter
    cnt = Counter(r["selected"] for r in rows)
    print("\nconfigurations chosen by the inner selection (folds):")
    for k, v in cnt.most_common():
        print(f"  {k:28} {v}")
    cfgs = sorted({k for k in rows[0] if k.startswith("cfg_")})
    print("\nevery FIXED configuration on the outer folds (descriptive; picking the best here would be a "
          "test-set choice):")
    best = None
    for k in cfgs:
        m = per_subject(rows, k)[1].mean()
        best = max(best, (m, k)) if best else (m, k)
        print(f"  {k[4:]:28} {m:.4f}")
    print(f"  best fixed configuration in HINDSIGHT: {best[1][4:]} {best[0]:.4f}  (optimistic by construction)")
    print("\nCaveats: transductive per-session centring (unlabeled test windows only); selection uses only the "
          "two training sessions, so it is noisy; the primary test is one comparison, everything else is "
          "descriptive.")


# ----------------------------------------------------------------------------- self-test
def _synth_subject(rng, n_trials=12, wpt=3, C=8, T=400, amp=0.12, band=(30.0, 45.0)):
    """class lives in ONE band only (default 30-45 Hz, which the current member only sees diluted inside
    the broadband covariance): a latent source with a fixed spatial pattern whose power grows with the
    class, on a white-noise floor with a session-specific channel gain."""
    u = rng.randn(C)
    u /= np.linalg.norm(u)
    gains = {s: np.exp(0.3 * rng.randn(C)) for s in (1, 2, 3)}
    X, y, ses, tid = [], [], [], []
    t = 0
    for s in (1, 2, 3):
        for k in range(n_trials):
            cls = k % 3
            for _ in range(wpt):
                x = rng.randn(C, T)
                src = R.bandpass(rng.randn(1, T), band[0], band[1])[0]
                src = src / (src.std() + 1e-12)
                x = x + amp * (0.0, 1.0, 2.0)[cls] * np.outer(u, src) * 2.0
                X.append(gains[s][:, None] * x)
                y.append(cls); ses.append(s); tid.append(t)
            t += 1
    return np.array(X), np.array(y), np.array(ses), np.array(tid)


def self_test():
    rng = np.random.RandomState(0)
    ok = True
    names = sorted({b for bs in ("S0_current", "S2_plus_gamma") for b in BAND_SETS[bs]})
    allrows = []
    first = None
    for i in range(4):
        x, y, ses, tid = _synth_subject(rng)
        Fb = {n: band_features(x, ses, n) for n in names}
        rows = run_subject(f"VP{i + 1:03d}", Fb, y, ses, tid, ("S0_current", "S2_plus_gamma"), (0.1, 0.03, 0.3))
        allrows += rows
        if first is None:
            first = (Fb, y, ses)
    # 1. the 'fixed' row must equal the earlier member (same features, same LOSO, same classifier)
    Fb, y, ses = first
    X0 = np.hstack([Fb[b] for b in BAND_SETS["S0_current"]]).astype(np.float64)
    ref = R.loso(X0, y, ses)
    mine = np.mean([r["fixed_win"] for r in allrows if r["subject"] == "VP001"])
    good = abs(ref - mine) < 1e-9
    print(f"[fixed==R.loso] {mine:.6f} vs {ref:.6f} -> {'OK' if good else 'FAIL'}")
    ok &= good
    # 2. selection must find the informative fine band and gain over the coarse set
    fx = per_subject(allrows, "fixed_win")[1].mean()
    nt = per_subject(allrows, "nested_win")[1].mean()
    fine = np.mean([r["selected"].startswith("S2_plus_gamma") for r in allrows])
    print(f"[planted 30-45 Hz] current {fx:.3f} | nested {nt:.3f} | share of folds choosing S2_plus_gamma {fine:.2f}")
    good = nt > fx + 0.03 and fine > 0.5
    print(f"  nested selection finds the band the current member misses and gains > 0.03 -> {'OK' if good else 'FAIL'}")
    ok &= good
    # 3. the held-out session must not influence the selection: corrupt it and the choice must not change
    configs = [(b, c) for b in ("S0_current", "S2_plus_gamma") for c in (0.1, 0.03)]
    Xc = {cfg: np.hstack([Fb[b] for b in BAND_SETS[cfg[0]]]).astype(np.float64) for cfg in configs}
    best_a, _ = select_config(Xc, y, ses, 2, configs)
    Xbad = {cfg: M.copy() for cfg, M in Xc.items()}
    for M in Xbad.values():
        M[ses == 2] = np.nan
    best_b, _ = select_config(Xbad, y, ses, 2, configs)
    good = best_a == best_b
    print(f"[leak] corrupting the held-out session leaves the selection unchanged -> {'OK' if good else 'FAIL'}")
    ok &= good
    # 4. the guard itself must fire
    try:
        assert_inner_clean(np.array([True, False]), np.array([True, True]))
        good = False
    except AssertionError:
        good = True
    print(f"[guard] a mask overlapping the held-out session is refused -> {'OK' if good else 'FAIL'}")
    ok &= good
    summarise(allrows)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--posteriors-dir")
    ap.add_argument("--cache-dir", help="cache the per-band tangent features per subject (recommended)")
    ap.add_argument("--bandsets", nargs="+", default=list(BAND_SETS), choices=list(BAND_SETS))
    ap.add_argument("--Cs", nargs="+", type=float, default=list(CS))
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--device", default=None, help="torch device for the eigendecompositions")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    if "S0_current" not in a.bandsets:
        a.bandsets = ["S0_current"] + a.bandsets
    if 0.1 not in a.Cs:
        a.Cs = [0.1] + a.Cs
    if a.device and hasattr(R, "set_eigh_backend"):
        print(f"eigendecomposition backend: {R.set_eigh_backend(a.device)}")
    names = sorted({b for bs in a.bandsets for b in BAND_SETS[bs]})
    rows = []
    for sid in a.subjects:
        got = load_subject(a.data, sid, names, a.cache_dir)
        if got is None:
            continue
        Fb, y, ses, tid = got
        rows += run_subject(sid, Fb, y, ses, tid, a.bandsets, a.Cs, a.posteriors_dir)
        print(f"[{sid}] done", flush=True)
    if not rows:
        raise SystemExit("no subject found")
    if a.out:
        keys = sorted({k for r in rows for k in r})
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print("per-fold rows ->", a.out)
    fx = per_subject(rows, "fixed_win")[1].mean()
    print(f"\nsanity: the current member should land near 0.6449 on the real data; got {fx:.4f}")
    summarise(rows)


if __name__ == "__main__":
    main()
