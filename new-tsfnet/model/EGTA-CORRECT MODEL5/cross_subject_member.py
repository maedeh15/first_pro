"""
cross_subject_member.py -- does the EA covariance member transfer to a SUBJECT it never saw?

Every number in this project comes from a subject-dependent protocol: leave one SESSION out, with
the other two sessions of the same person in training. A reviewer will ask whether the member is
anything more than a per-subject classifier. Nothing in the project answers that, so this does.

Three training sets, each evaluated on the SAME held-out session of the same target subject, so the
three readings are paired per fold:

  within        the target's own two remaining sessions            the protocol used everywhere else
  cross         all 3 sessions of the other 25 subjects, and
                NO data from the target at all                     the subject-independent question
  cross+within  both pooled, every window weighted equally        NOT "the two together": the target
                                                                   contributes about 591 of 22780
                                                                   rows, i.e. 2.6%, so this row sits
                                                                   near the cross row by construction
                                                                   and says nothing about whether the
                                                                   target's data would help if it
                                                                   counted for more
  cross+within(w) the same with the target's windows weighted      the row that actually asks it.
                x --target-weight, so they carry a                 --target-weight 1 reproduces the
                comparable share of the loss                       equal-weight row

Why the features are comparable across people at all: each session is whitened by its own mean
covariance, so every session -- whoever it belongs to -- is mapped to a neighbourhood of the
identity before the log map. That is the Euclidean-Alignment argument, and the `cross` row is a
direct test of it on this dataset. Without that step the rows would differ in head size, electrode
impedance and amplitude scale, and `cross` would mean nothing.

What it does NOT test: this is the linear member alone, not TSFNet and not the fused system. A
positive `cross` row says the member carries population-level structure; it does not say the deep
model or the fusion transfers.

LEAKAGE. The per-session whitening of the TARGET's sessions uses that session's own unlabelled
windows, the same transductive step disclosed for the within-subject results. No label of the
target subject enters the `cross` condition, and the target's held-out session never enters any
training set in any condition.

It reads {sid}_eeg_riem.npy when build_riem_features.py has written it (mode EA), and computes the
features otherwise.

usage:
  python cross_subject_member.py --self-test
  python cross_subject_member.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" [--device cuda]
"""
import os
import json
import argparse
import tempfile
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import check_riemannian_eeg as R

CONDITIONS = ("within", "cross", "cross+within")
WEIGHTED = "cross+within(w)"


def fit_eval(Xtr, ytr, Xte, yte, classes=None, sw=None):
    if classes is not None:
        mtr, mte = np.isin(ytr, classes), np.isin(yte, classes)
        Xtr, ytr, Xte, yte = Xtr[mtr], ytr[mtr], Xte[mte], yte[mte]
        if sw is not None:
            sw = sw[mtr]
    if len(Xte) == 0 or len(np.unique(ytr)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    if sw is None:
        clf.fit(Xtr, ytr)
    else:
        clf.fit(Xtr, ytr, logisticregression__sample_weight=sw)
    return float((clf.predict(Xte) == yte).mean())


def load_subject(data, sid, verbose=True):
    """-> (F, y, ses) with F the EA multiband tangent features, from the cached file when present"""
    d = os.path.join(data, sid)
    y = np.load(os.path.join(d, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
    tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
    cache = os.path.join(d, f"{sid}_eeg_riem.npy")
    meta = os.path.join(d, f"{sid}_eeg_riem_meta.json")
    if os.path.exists(cache) and os.path.exists(meta):
        import build_riem_features as B
        m = json.load(open(meta, encoding="utf-8"))
        if m.get("mode") != "EA":
            raise SystemExit(f"{sid}: cached features were built with mode={m.get('mode')!r}, not 'EA'. "
                             f"Rebuild with build_riem_features.py --mode EA --overwrite, or delete the "
                             f"cache so this script computes them.")
        if m.get("fingerprint") != B.fingerprint(y, tid, ses):
            raise SystemExit(f"{sid}: cached features do not match the current labels/trials/sessions. "
                             f"Rebuild with build_riem_features.py --mode EA --overwrite.")
        F = np.load(cache)
        if verbose:
            print(f"  {sid}: cached features {F.shape}")
    else:
        x = np.asarray(np.load(os.path.join(d, f"{sid}_eeg_channel_windows.npy"), mmap_mode="r"),
                       dtype=np.float64)
        bands = R.build_covs(x, True)
        F = np.hstack([R.domain_features(bands[k], ses, "EA") for k in bands]).astype(np.float32)
        if verbose:
            print(f"  {sid}: computed features {F.shape}")
    if len(F) != len(y):
        raise SystemExit(f"{sid}: {len(F)} feature rows for {len(y)} labels")
    return F, y, ses


def run(data, subjects, verbose=True, target_weight=None):
    feats = {}
    for sid in subjects:
        if not os.path.exists(os.path.join(data, sid, f"{sid}_labels.npy")):
            continue
        feats[sid] = load_subject(data, sid, verbose)
    if len(feats) < 3:
        raise SystemExit("need at least 3 subjects")
    dims = {sid: F.shape[1] for sid, (F, _, _) in feats.items()}
    if len(set(dims.values())) != 1:
        raise SystemExit(f"feature dimension differs between subjects, so they cannot be pooled: {dims}")
    rows = []
    for sid, (F, y, ses) in feats.items():
        others = [s for s in feats if s != sid]
        Xo = np.vstack([feats[s][0] for s in others])
        yo = np.concatenate([feats[s][1] for s in others])
        for s in np.unique(ses):
            te = ses == s
            tr = ~te
            Xp = np.vstack([Xo, F[tr]])
            yp = np.concatenate([yo, y[tr]])
            sets = {"within": (F[tr], y[tr], None),
                    "cross": (Xo, yo, None),
                    "cross+within": (Xp, yp, None)}
            conds = list(CONDITIONS)
            if target_weight is not None:
                sw = np.concatenate([np.ones(len(yo)), np.full(len(y[tr]), float(target_weight))])
                sets[WEIGHTED] = (Xp, yp, sw)
                conds.append(WEIGHTED)
            row = dict(subject=sid, session=int(s), n_test=int(te.sum()))
            for c in conds:
                Xtr, ytr, sw_c = sets[c]
                row[f"{c}_3cls"] = fit_eval(Xtr, ytr, F[te], y[te], sw=sw_c)
                row[f"{c}_2v3"] = fit_eval(Xtr, ytr, F[te], y[te], (1, 2), sw=sw_c)
                row[f"{c}_ntrain"] = int(len(ytr))
            row["target_share"] = float(len(y[tr]) / len(yp))
            rows.append(row)
            if verbose:
                print(f"  {sid} session {s}: " + "  ".join(
                    f"{c} {row[f'{c}_3cls']:.3f}" for c in conds), flush=True)
    return rows


def summarise(rows):
    subs = sorted({r["subject"] for r in rows})
    conds = [c for c in list(CONDITIONS) + [WEIGHTED] if f"{c}_3cls" in rows[0]]

    def per_subject(key):
        return np.array([np.mean([r[key] for r in rows if r["subject"] == s]) for s in subs])

    print(f"\n{len(subs)} target subjects, {len(rows)} folds. Mean over subjects:")
    print(f"  {'training set':14} {'3-class':>9} {'2-vs-3':>9} {'train windows':>15}")
    for c in conds:
        n = int(np.mean([r[f"{c}_ntrain"] for r in rows]))
        print(f"  {c:14} {per_subject(f'{c}_3cls').mean():9.4f} {per_subject(f'{c}_2v3').mean():9.4f} "
              f"{n:15d}")
    share = float(np.mean([r.get("target_share", float("nan")) for r in rows]))
    print(f"\nthe target contributes {100 * share:.1f}% of the pooled training rows, so the "
          f"equal-weight 'cross+within' row sits near 'cross' by construction.")
    print("\npaired over subjects (Wilcoxon):")
    pairs = [("cross", "within"), ("cross+within", "within"), ("cross+within", "cross")]
    if WEIGHTED in conds:
        pairs += [(WEIGHTED, "within"), (WEIGHTED, "cross+within")]
    for a, b in pairs:
        for lab, k in (("3-class", "3cls"), ("2-vs-3", "2v3")):
            d = per_subject(f"{a}_{k}") - per_subject(f"{b}_{k}")
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  {a:13} - {b:13} {lab:8} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  "
                  f"p={p:.4f}")
    w = per_subject("within_3cls")
    c = per_subject("cross_3cls")
    print(f"\nsubjects where the subject-independent model beats the subject-specific one: "
          f"{int((c > w).sum())}/{len(w)}")
    print(f"chance is 0.333 (3-class) and 0.500 (2-vs-3). The within row should land near the "
          f"0.6449 reported for this member; a cross row far above chance means the aligned "
          f"covariance carries population-level structure, and one near chance means the member is "
          f"subject-specific. Transductive per-session whitening applies to the target's sessions in "
          f"every row; no target label enters the cross row.")


def self_test():
    """Two planted worlds, same machinery. In the SHARED world the class effect is the same
    direction for everyone, so a model trained on other subjects must transfer. In the IDIOSYNCRATIC
    world each subject's effect is in its own random direction, so it must not. If the script cannot
    tell those apart, its reading of the real data is worthless."""
    ok = True
    for world in ("shared", "idiosyncratic"):
        rng = np.random.RandomState(0)
        C, T, nper = 8, 300, 24
        with tempfile.TemporaryDirectory() as td:
            subs = [f"VP{i:03d}" for i in range(1, 7)]
            shared_dir = rng.randn(C)
            for sid in subs:
                d = os.path.join(td, sid)
                os.makedirs(d)
                direction = shared_dir if world == "shared" else rng.randn(C)
                direction = direction / np.linalg.norm(direction)
                N = 3 * nper
                ses = np.repeat([1, 2, 3], nper)
                y = np.tile([0, 1, 2], N // 3)
                x = rng.randn(N, C, T)
                for i in range(N):
                    # class = power along `direction`, i.e. a covariance effect
                    x[i] += (0.9 * y[i]) * np.outer(direction, rng.randn(T))
                    x[i] *= np.exp(0.4 * rng.randn(C, 1))        # per-subject channel gains
                np.save(os.path.join(d, f"{sid}_eeg_channel_windows.npy"), x.astype(np.float32))
                for nm, a in (("labels", y), ("session_ids", ses),
                              ("trial_ids", np.repeat(np.arange(N // 3), 3))):
                    np.save(os.path.join(d, f"{sid}_{nm}.npy"), a)
            rows = run(td, subs, verbose=False, target_weight=40.0)
            g = lambda c: float(np.mean([r[f"{c}_3cls"] for r in rows]))
            print(f"[{world:14}] within {g('within'):.3f} | cross {g('cross'):.3f} | "
                  f"cross+within {g('cross+within'):.3f}")
            if world == "shared":
                good = g("cross") > 0.60
                print(f"  a shared effect transfers across subjects -> {'OK' if good else 'FAIL'}")
            else:
                good = g("cross") < 0.45 and g("within") > 0.60
                print(f"  a per-subject effect does NOT transfer, while within still works -> "
                      f"{'OK' if good else 'FAIL'}")
                ok &= good
                # and the weighted row must rescue what equal-weight pooling drowns: with a
                # per-subject effect, upweighting the target has to beat the equal-weight pool
                good = g(WEIGHTED) > g("cross+within") + 0.05
                print(f"  upweighting the target beats equal-weight pooling "
                      f"({g(WEIGHTED):.3f} vs {g('cross+within'):.3f}) -> {'OK' if good else 'FAIL'}")
            ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--device", default=None, help="torch device for the eigendecompositions")
    ap.add_argument("--out", help="per-fold csv")
    ap.add_argument("--target-weight", type=float, default=None,
                    help="also fit a pooled model with the target's own windows carrying this "
                         "sample weight. The target is ~2.6%% of the pooled rows, so a weight near "
                         "38 makes its share of the loss comparable to the other 25 subjects "
                         "combined; 1 reproduces the equal-weight row.")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    if a.device:
        print(f"eigendecomposition backend: {R.set_eigh_backend(a.device)}")
    rows = run(a.data, a.subjects, target_weight=a.target_weight)
    if a.out:
        import csv
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"per-fold rows -> {a.out}")
    summarise(rows)


if __name__ == "__main__":
    main()
