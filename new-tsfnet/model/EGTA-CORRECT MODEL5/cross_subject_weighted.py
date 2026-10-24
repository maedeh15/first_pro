"""
cross_subject_weighted.py -- does the closed-form EEG member improve when it ALSO learns from the other
25 subjects, once the target subject's own sessions are given a realistic weight?  (CPU only.)

NAME NOTE: this is NOT the three-condition cross_subject_member.py you already ran (within / cross /
cross+within).  That run showed equal-weight pooling is swamped (own data are only 2.6% of the training
rows, 591 of 22780).  This script adds the weighted variants that were missing, so the only open question
on this branch -- does population data help at all when the own data are not drowned out -- gets answered.

Why: the diagnosis of this project is "very little independent training data": 2 training sessions =
18 blocks per fold.  The EEG member (per-session re-centred multiband covariance -> tangent vector ->
standardise -> logistic regression, see check_riemannian_eeg.py / late_fusion_riemann.py) is cheap, so
it can be fitted on far more data.  Every (subject, session) is its own domain and is centred on its
OWN unlabeled windows, so sessions of different subjects are put on a common footing before pooling.

Folds: exactly the 78 folds of the deep model (subject i, held-out session s).
Training sets per fold (the held-out session of subject i is NEVER in any of them; asserted):
  own        the two other sessions of subject i                    (== the member reported so far)
  pooled     own + all sessions of the other subjects, equal weights  <- PRE-REGISTERED main variant
  others     all sessions of the other subjects only (no own data)    (pure cross-subject)
  pooled_w5  pooled with the own sessions weighted x5                 (descriptive sensitivity)
  pooled_w20 same, x20   (own data = 35% of the total weight)         (descriptive sensitivity)
  pooled_w50 same, x50   (own data = 57% of the total weight)         (descriptive sensitivity)
Scoring: window and block accuracy of the member alone; if --posteriors-dir is given, also of the late
fusion P = 0.5 * P_deep + 0.5 * P_member (deep = K-seed ensemble, and each seed alone), i.e. the system
reported in the thesis chapter.  Per subject = mean over its 3 folds; paired Wilcoxon over subjects.

PRE-REGISTERED (written before any real run): primary contrast = member window accuracy,
pooled - own.  Reading rule: counted as an improvement only if >= +1.0 pp with Wilcoxon p < 0.05; if not,
cross-subject pre-training of the full TSFNet (GPU, expensive) is NOT justified by this evidence.
Everything else (others, pooled_w5, block, fused columns) is descriptive.

Protocol note: 'pooled' and 'others' use the LABELS of other subjects (never of the test session).  It is
a different protocol from the strictly within-subject one and must be reported separately.

usage
  python cross_subject_weighted.py --self-test
  python cross_subject_weighted.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" ^
        [--posteriors-dir "E:\\...\\ensemble_pure_baseline_40b7d8ea53\\posteriors"] ^
        [--mode EA] [--cache-dir feats] [--subjects VP001 VP002] [--out cross_subject_folds.csv]
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

VARIANTS = {            # name: (use own sessions, use other subjects, weight of own rows)
    "own": (True, False, 1.0),
    "pooled": (True, True, 1.0),
    "others": (False, True, 1.0),
    "pooled_w5": (True, True, 5.0),
    "pooled_w20": (True, True, 20.0),
    "pooled_w50": (True, True, 50.0),
}
PRIMARY = ("pooled", "own")
MIN_GAIN = 0.01


def subject_features(x, ses, mode, multiband=True):
    return L.eeg_member_features(x, ses, multiband, mode).astype(np.float32)


def load_all(data, subjects, mode, cache_dir=None, multiband=True):
    """-> dict sid -> (X, y, ses, tid)"""
    out = {}
    for sid in subjects:
        d = os.path.join(data, sid)
        pe = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(pe):
            continue
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        # the cache carries a fingerprint of the window set: a cache left over from an earlier
        # preprocessing run loads fine and is wrong, and nothing downstream would notice
        fp = B.fingerprint(y, tid, ses)
        stem = f"{sid}_{mode}_{'mb' if multiband else 'bb'}"
        cp = os.path.join(cache_dir, stem + ".npz") if cache_dir else None
        X = None
        if cp and os.path.exists(cp):
            z = np.load(cp)
            if str(z["__fingerprint__"]) == fp and len(z["X"]) == len(y):
                X = z["X"]
            else:
                print(f"[cache] {sid}: fingerprint mismatch, recomputing (delete {cp} to silence)",
                      flush=True)
        if X is None:
            x = np.asarray(np.load(pe, mmap_mode="r"), dtype=np.float64)
            X = subject_features(x, ses, mode, multiband)
            if cp:
                os.makedirs(cache_dir, exist_ok=True)
                np.savez(cp, X=X, __fingerprint__=np.array(fp))
        out[sid] = (X, y, ses, tid)
        print(f"[features] {sid}: {X.shape}", flush=True)
    return out


def assert_no_leak(train_mask, subj, ses, sid, s):
    """the held-out session of the held-out subject must not be in the training rows"""
    if np.any(train_mask & (subj == sid) & (ses == s)):
        raise AssertionError(f"LEAK: training rows contain held-out session {s} of {sid}")


def fit_predict(Xtr, ytr, wtr, Xte):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    clf.fit(Xtr, ytr, logisticregression__sample_weight=wtr)
    P = np.zeros((len(Xte), 3))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(Xte)
    return P


def run_cv(data, variants=tuple(VARIANTS), posteriors_dir=None, w=0.5, verbose=True):
    sids = sorted(data)
    X = np.concatenate([data[s][0] for s in sids])
    y = np.concatenate([data[s][1] for s in sids])
    ses = np.concatenate([data[s][2] for s in sids])
    subj = np.concatenate([[s] * len(data[s][1]) for s in sids])
    rows = []
    for sid in sids:
        Xs, ys, ss, ts = data[sid]
        for s in sorted(np.unique(ss)):
            te_local = ss == s
            Xte, yte, tid = Xs[te_local], ys[te_local], ts[te_local]
            Pdeep = None
            if posteriors_dir:
                p = os.path.join(posteriors_dir, f"{sid}_session{s}.npz")
                if os.path.exists(p):
                    z = np.load(p)
                    if np.array_equal(z["y"], yte) and np.array_equal(z["trial_ids"], tid):
                        Pdeep = L.norm_rows(z["P_final"].astype(np.float64))
                    elif verbose:
                        print(f"[skip fusion] {sid} s{s}: labels/trial ids differ from the saved fold")
            for v in variants:
                use_own, use_oth, w_own = VARIANTS[v]
                own = (subj == sid) & (ses != s)
                oth = subj != sid
                mask = (own if use_own else np.zeros_like(own)) | (oth if use_oth else np.zeros_like(oth))
                assert_no_leak(mask, subj, ses, sid, s)
                wt = np.where(own & mask, w_own, 1.0)[mask]
                Pe = fit_predict(X[mask], y[mask], wt, Xte)
                row = dict(subject=sid, session=int(s), variant=v, n=len(yte), n_train=int(mask.sum()),
                           member_win=L.acc_window(Pe, yte), member_tri=L.acc_trial(Pe, yte, tid))
                if Pdeep is not None:
                    row["fused_ens_win"] = L.acc_window(L.fuse(Pdeep.mean(0), Pe, w), yte)
                    row["fused_ens_tri"] = L.acc_trial(L.fuse(Pdeep.mean(0), Pe, w), yte, tid)
                    row["fused_single_win"] = float(np.mean([L.acc_window(L.fuse(Pdeep[k], Pe, w), yte)
                                                             for k in range(len(Pdeep))]))
                rows.append(row)
        if verbose:
            print(f"[{sid}] done", flush=True)
    return rows


def per_subject(rows, variant, key):
    subs = sorted({r["subject"] for r in rows})
    return subs, np.array([np.mean([r[key] for r in rows if r["subject"] == s and r["variant"] == variant])
                           for s in subs])


def summarise(rows):
    keys = [k for k in ("member_win", "member_tri", "fused_ens_win", "fused_ens_tri", "fused_single_win")
            if k in rows[0]]
    variants = [v for v in VARIANTS if any(r["variant"] == v for r in rows)]
    print(f"\n{len({r['subject'] for r in rows})} subjects.  mean over subjects, then paired Wilcoxon vs 'own'.")
    print(f"{'variant':10} " + " ".join(f"{k:>16}" for k in keys) + "   n_train(mean)")
    for v in variants:
        ntr = np.mean([r["n_train"] for r in rows if r["variant"] == v])
        print(f"{v:10} " + " ".join(f"{per_subject(rows, v, k)[1].mean():16.4f}" for k in keys) + f"   {ntr:9.0f}")
    print("\npaired difference vs 'own' (per-subject means):")
    print(f"{'variant':10} {'metric':18} {'diff':>9} {'wins':>7} {'p':>8}")
    out = {}
    for v in variants:
        if v == "own":
            continue
        for k in keys:
            _, a = per_subject(rows, v, k)
            _, b = per_subject(rows, "own", k)
            d = a - b
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            mark = "   <== PRE-REGISTERED PRIMARY" if (v, k) == (PRIMARY[0], "member_win") else ""
            print(f"{v:10} {k:18} {d.mean():+9.4f} {int((d > 0).sum()):3d}/{len(d):<3} {p:8.4f}{mark}")
            out[(v, k)] = (d.mean(), p)
    if ("pooled", "member_win") in out:
        d, p = out[("pooled", "member_win")]
        ok = d >= MIN_GAIN and p < 0.05
        print(f"\nPRE-REGISTERED READING: pooled - own = {d:+.4f} (p={p:.4f}) -> "
              + ("improvement by the rule: cross-subject pre-training is worth testing on the deep model"
                 if ok else "NOT an improvement by the rule: do not spend GPU on cross-subject pre-training "
                            "on this evidence"))
    print("\nCaveats: uses labels of OTHER subjects (different protocol, report separately); transductive "
          "per-session centring (unlabeled test windows only); only pooled-own/member_win is pre-registered, "
          "every other row is descriptive and uncorrected.")
    return out


# ----------------------------------------------------------------------------- self-test
def _synth(rng, n_sub=6, n_trials=8, wpt=3, C=8, T=120, effect=1.08):
    L0 = rng.randn(C, C) * 0.3 + np.eye(C)
    data = {}
    for i in range(n_sub):
        Ls = L0 + 0.05 * rng.randn(C, C)                    # mild subject-specific mixing
        X, y, ses, tid = [], [], [], []
        t = 0
        for s in (1, 2, 3):
            A = np.diag(np.exp(0.6 * rng.randn(C)))          # session-specific per-source gains
            for k in range(n_trials):
                cls = k % 3
                for _ in range(wpt):
                    z = rng.randn(C, T)
                    z[cls] *= effect
                    X.append(A @ Ls @ z)
                    y.append(cls); ses.append(s); tid.append(t)
                t += 1
        data[f"VP{i + 1:03d}"] = (np.array(X), np.array(y), np.array(ses), np.array(tid))
    return data


def self_test():
    rng = np.random.RandomState(0)
    raw = _synth(rng)
    feats = {sid: (subject_features(x, ses, "EA", multiband=False), y, ses, tid)
             for sid, (x, y, ses, tid) in raw.items()}
    ok = True
    rows = run_cv(feats, verbose=False)
    _, own = per_subject(rows, "own", "member_win")
    _, pooled = per_subject(rows, "pooled", "member_win")
    # 1. 'own' must reproduce the earlier member exactly (same features, same LOSO, same classifier)
    sid = sorted(feats)[0]
    X, y, ses, _ = feats[sid]
    ref = R.loso(X.astype(np.float64), y, ses)
    mine = np.mean([r["member_win"] for r in rows if r["subject"] == sid and r["variant"] == "own"])
    good = abs(ref - mine) < 1e-9
    print(f"[own==R.loso] {mine:.6f} vs {ref:.6f} -> {'OK' if good else 'FAIL'}")
    ok &= good
    # 2. pooling more subjects must help when own data are tiny and the class structure is shared
    good = pooled.mean() > own.mean() + 0.05
    print(f"[pooling] own {own.mean():.3f} | pooled {pooled.mean():.3f} (want > own + 0.05) -> "
          f"{'OK' if good else 'FAIL'}")
    ok &= good
    # 3. training sizes: pooled uses own + others, 'others' excludes own
    n = {v: np.mean([r["n_train"] for r in rows if r["variant"] == v]) for v in VARIANTS}
    good = n["pooled"] > n["own"] and n["others"] > n["own"] and abs(n["pooled"] - n["others"] - n["own"]) < 1e-9
    print(f"[sizes] own {n['own']:.0f}, others {n['others']:.0f}, pooled {n['pooled']:.0f} -> "
          f"{'OK' if good else 'FAIL'}")
    ok &= good
    # 4. leak guard must fire when the held-out session is put into the training rows
    subj = np.array(["A"] * 6 + ["B"] * 6)
    sess = np.array([1, 1, 2, 2, 3, 3] * 2)
    bad = np.ones(12, bool)
    try:
        assert_no_leak(bad, subj, sess, "A", 2)
        good = False
    except AssertionError:
        good = True
    print(f"[guard] held-out session in the training rows is refused -> {'OK' if good else 'FAIL'}")
    ok &= good
    # 5. the reading rule fires on the planted effect, and the report renders
    summarise(rows)
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--posteriors-dir")
    ap.add_argument("--mode", default="EA", choices=["EA", "riem_disp", "riem", "raw"])
    ap.add_argument("--cache-dir", help="cache the per-subject feature matrices here (recommended)")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)],
                    help="subjects to EVALUATE; features of all listed subjects are used as the 'others' pool")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--no-multiband", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    if "own" not in a.variants:
        a.variants = ["own"] + a.variants
    data = load_all(a.data, a.subjects, a.mode, a.cache_dir, not a.no_multiband)
    if len(data) < 2:
        raise SystemExit("need at least 2 subjects with features")
    print(f"member mode = {a.mode}; subjects = {len(data)}")
    if len(data) < 10 and any(v != "own" for v in a.variants):
        print(f"WARNING: --subjects gives both the evaluated subjects AND the 'others' pool, and only "
              f"{len(data)} subjects were loaded, so every pooled row is trained on {len(data) - 1} "
              f"other subjects rather than 25. Those rows are not comparable to a full run.")
    rows = run_cv(data, tuple(a.variants), a.posteriors_dir)
    if a.out:
        keys = list(rows[0])
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(rows)
        print("per-fold rows ->", a.out)
    summarise(rows)


if __name__ == "__main__":
    main()
