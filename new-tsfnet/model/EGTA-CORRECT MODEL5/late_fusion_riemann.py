"""
late_fusion_riemann.py -- OFFLINE late fusion of a non-deep Riemannian EEG classifier with the deep
TSFNet posteriors. No training of any network, no GPU.

Pipeline (per subject, per held-out session s -- the same 78 folds the deep model was scored on)
  EEG member : multiband (broadband + theta 4-8 + alpha 8-13 + beta 13-30 Hz) window covariances,
               per-SESSION Frechet re-centring + dispersion normalisation (TSMNet's SPDDSMBN idea,
               closed form), tangent-space vector, StandardScaler + LogisticRegression(C=0.1)
               fitted ONLY on the two training sessions; predict_proba on session s.
               (identical to the best row of check_riemannian_eeg.py: riem_disp|multi)
  deep member: posteriors of the 'final' head saved by  ensemble_eval.py --save-posteriors
               (K seeds). Used as (a) each seed alone and (b) the K-seed arithmetic ensemble.
  fusion     : P = (1-w) * P_deep + w * P_eeg   (arithmetic; rows normalised first).
               w = 0.5 is the PRE-REGISTERED primary weight. The sweep over w is descriptive only:
               no weight is selected on test data, and nothing below changes after seeing it.
  scoring    : window accuracy and block (trial) accuracy (mean posterior over the block, argmax),
               per subject = mean over its 3 folds; paired Wilcoxon over subjects.

Alignment guard: the saved labels and trial ids of every fold must equal the ones recomputed here,
otherwise that fold is refused (a silent misalignment would pair windows with another window's
posterior).

HONEST CAVEATS (printed again at the end)
  * transductive per-session centring of the EEG member (unlabeled test windows only), like E3/AdaBN;
  * the fused model is a different, larger system: report it as "TSFNet + Riemannian EEG member";
  * only w = 0.5 is the primary test; the other weights are a sensitivity curve, not a search;
  * this does not retrain TSFNet: a model that learns to use the EEG information jointly could do
    better or worse.

usage
  python late_fusion_riemann.py --self-test
  python late_fusion_riemann.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" ^
        --posteriors-dir "E:\\...\\ensemble_eval_...\\posteriors" [--out late_fusion_folds.csv]
"""
import os
import re
import csv
import glob
import argparse
import tempfile
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import check_riemannian_eeg as R

WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
W_PRIMARY = 0.5


def norm_rows(P):
    return P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)


def acc_window(P, y):
    return float((P.argmax(-1) == y).mean())


def acc_trial(P, y, tid):
    ts = np.unique(tid)
    return float(np.mean([P[tid == t].mean(0).argmax() == y[tid == t][0] for t in ts]))


def acc_2v3(P, y):
    m = np.isin(y, (1, 2))
    if m.sum() == 0:
        return float("nan")
    return float((np.where(P[m][:, 1] >= P[m][:, 2], 1, 2) == y[m]).mean())


MEMBER_MODE = "riem_disp"      # set by --mode; 'EA' = per-session arithmetic-mean whitening (Wu 2020)


def eeg_member_features(x, ses, multiband=True, mode=None):
    """multiband tangent features for all windows of one subject (each session centred independently);
    mode in {riem_disp (default, the pre-registered main member), riem, EA, raw}"""
    mode = mode or MEMBER_MODE
    bands = R.build_covs(x, multiband)
    return np.hstack([R.domain_features(bands[b], ses, mode) for b in bands])


def eeg_posteriors(X, y, ses, test_session):
    tr, te = ses != test_session, ses == test_session
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))
    clf.fit(X[tr], y[tr])
    P = np.zeros((te.sum(), 3))
    P[:, clf.classes_.astype(int)] = clf.predict_proba(X[te])
    return P


def fuse(Pd, Pe, w):
    return (1.0 - w) * Pd + w * Pe


def fold_rows(sid, s, Pdeep_seeds, Pe, y, tid):
    """Pdeep_seeds (K,N,3) normalised. returns list of dict rows (one per deep variant x weight)"""
    rows = []
    variants = {"ensemble": Pdeep_seeds.mean(0)}
    for w in WEIGHTS:
        Pf = fuse(variants["ensemble"], Pe, w)
        rows.append(dict(subject=sid, session=s, deep="ensemble", w=w, win=acc_window(Pf, y),
                         tri=acc_trial(Pf, y, tid), v23=acc_2v3(Pf, y)))
        # single seeds fused, averaged over seeds (the fair 'one trained model' number)
        sw = [acc_window(fuse(Pdeep_seeds[k], Pe, w), y) for k in range(len(Pdeep_seeds))]
        st = [acc_trial(fuse(Pdeep_seeds[k], Pe, w), y, tid) for k in range(len(Pdeep_seeds))]
        s2 = [acc_2v3(fuse(Pdeep_seeds[k], Pe, w), y) for k in range(len(Pdeep_seeds))]
        rows.append(dict(subject=sid, session=s, deep="single", w=w, win=float(np.mean(sw)),
                         tri=float(np.mean(st)), v23=float(np.mean(s2))))
    return rows


def run(data, post_dir, multiband=True, subjects=None, verbose=True, save_member=None):
    files = sorted(glob.glob(os.path.join(post_dir, "*_session*.npz")))
    if not files:
        raise SystemExit("no *_session*.npz in --posteriors-dir (run ensemble_eval.py --save-posteriors)")
    by_sid = {}
    for f in files:
        m = re.match(r"(VP\d+)_session(\d+)\.npz$", os.path.basename(f))
        if m:
            by_sid.setdefault(m.group(1), {})[int(m.group(2))] = f
    rows, skipped = [], []
    for sid in sorted(by_sid):
        if subjects and sid not in subjects:
            continue
        d = os.path.join(data, sid)
        pe = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(pe):
            skipped.append((sid, "no eeg_channel_windows.npy"))
            continue
        x = np.asarray(np.load(pe, mmap_mode="r"), dtype=np.float64)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        X = eeg_member_features(x, ses, multiband)
        for s, f in sorted(by_sid[sid].items()):
            z = np.load(f)
            te = ses == s
            if not (np.array_equal(z["y"], y[te]) and np.array_equal(z["trial_ids"], tid[te])):
                skipped.append((sid, f"session {s}: labels/trial ids differ from the saved fold -- refused"))
                continue
            Pd = norm_rows(z["P_final"].astype(np.float64))
            Pe = eeg_posteriors(X, y, ses, s)
            if save_member:                        # the offline member's own window posteriors, in the
                os.makedirs(save_member, exist_ok=True)   # posteriors/ layout, for decompose_integrated --member
                np.savez(os.path.join(save_member, f"{sid}_session{s}.npz"), P_final=np.asarray(Pe, dtype=np.float32)[None],
                         y=z["y"], trial_ids=z["trial_ids"], pos_in_trial=z["pos_in_trial"])
            rows += fold_rows(sid, s, Pd, Pe, y[te], tid[te])
        if verbose:
            print(f"[{sid}] done ({len(by_sid[sid])} folds)", flush=True)
    return rows, skipped


def summarise(rows):
    subs = sorted(set(r["subject"] for r in rows))

    def per_subject(deep, w, key):
        return np.array([np.mean([r[key] for r in rows if r["subject"] == s and r["deep"] == deep and r["w"] == w])
                         for s in subs])

    out = {}
    print(f"\n{len(subs)} subjects. mean over subjects; weight w = share of the Riemannian EEG member.")
    for deep, lab in (("ensemble", "K-seed ensemble of deep 'final' head"), ("single", "single deep seed (avg over seeds)")):
        print("\n" + "=" * 92)
        print(f"deep member: {lab}")
        print("=" * 92)
        print(f"{'w':>5} {'window':>8} {'block':>8} {'2v3':>8} | {'dWin vs deep':>13} {'wins':>6} {'p':>8} | {'dBlk vs deep':>13} {'p':>8}")
        base_w = per_subject(deep, 0.0, "win")
        base_t = per_subject(deep, 0.0, "tri")
        for w in WEIGHTS:
            a, t, v = (per_subject(deep, w, k) for k in ("win", "tri", "v23"))
            dw, dt = a - base_w, t - base_t
            pw = wilcoxon(dw).pvalue if np.any(dw != 0) else 1.0
            pt = wilcoxon(dt).pvalue if np.any(dt != 0) else 1.0
            mark = "  <== PRIMARY (pre-registered)" if w == W_PRIMARY else ""
            print(f"{w:5.2f} {a.mean():8.4f} {t.mean():8.4f} {np.nanmean(v):8.4f} | {dw.mean():+13.4f} "
                  f"{int((dw > 0).sum()):3d}/{len(dw):<2} {pw:8.4f} | {dt.mean():+13.4f} {pt:8.4f}{mark}")
            out[(deep, w)] = (a.mean(), t.mean(), dw.mean(), pw, int((dw > 0).sum()), len(dw))
    # fused vs the EEG member alone (w = 1)
    print("\nPrimary comparisons (w = 0.5, window level):")
    for deep in ("ensemble", "single"):
        a = per_subject(deep, W_PRIMARY, "win")
        for other, nm in ((0.0, "deep alone"), (1.0, "Riemannian EEG alone")):
            b = per_subject(deep, other, "win")
            d = a - b
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  fused({deep}) - {nm:22}: {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    print("\nCaveats: transductive per-session centring of the EEG member; the fused system is "
          "'TSFNet + Riemannian EEG member' (report it as such); w=0.5 is the only pre-registered test, the "
          "rest is a sensitivity curve; TSFNet itself is not retrained.")
    return out


# ----------------------------------------------------------------------------- self-test
def self_test():
    """planted: EEG covariance carries the class (with a per-session gain shift), the 'deep' posteriors carry
    it independently and weakly. Fusion should beat the deep member; guard must refuse a misaligned fold."""
    rng = np.random.RandomState(0)
    C, T, n = 8, 400, 36
    with tempfile.TemporaryDirectory() as td:
        dd, pdir = os.path.join(td, "data"), os.path.join(td, "post")
        os.makedirs(os.path.join(dd, "VP001")), os.makedirs(pdir)
        Lm = rng.randn(C, C) * 0.3 + np.eye(C)
        X, y, ses, tid = [], [], [], []
        t = 0
        for s in (1, 2, 3):
            A = np.diag(np.exp(0.8 * rng.randn(C)))
            for k in range(n // 3):
                cls = k % 3
                for _ in range(3):
                    z = rng.randn(C, T)
                    z[cls] *= 1.25
                    X.append(A @ Lm @ z)
                    y.append(cls); ses.append(s); tid.append(t)
                t += 1
        X, y, ses, tid = np.array(X), np.array(y), np.array(ses), np.array(tid)
        np.save(os.path.join(dd, "VP001", "VP001_eeg_channel_windows.npy"), X.astype(np.float32))
        for nm, a in (("labels", y), ("session_ids", ses), ("trial_ids", tid)):
            np.save(os.path.join(dd, "VP001", f"VP001_{nm}.npy"), a)
        for s in (1, 2, 3):
            m = ses == s
            K = 3
            P = np.empty((K, m.sum(), 3))
            for k in range(K):
                L = rng.randn(m.sum(), 3)
                L[np.arange(m.sum()), y[m]] += 0.5            # weak deep evidence
                e = np.exp(L)
                P[k] = e / e.sum(1, keepdims=True)
            np.savez(os.path.join(pdir, f"VP001_session{s}.npz"), seeds=np.array([1, 2, 3]), y=y[m],
                     trial_ids=tid[m], pos_in_trial=np.zeros(m.sum(), np.int32),
                     P_final=P, P_eeg=P, P_fusion=P, P_fnirs=P)
        rows, skipped = run(dd, pdir, multiband=False, verbose=False)
        mdir = os.path.join(dd, "member_out")          # --save-member writes aligned, normalised window posteriors
        run(dd, pdir, multiband=False, verbose=False, save_member=mdir)
        for fn in sorted(os.listdir(mdir)):
            zm, zd = np.load(os.path.join(mdir, fn)), np.load(os.path.join(pdir, fn))
            assert np.array_equal(zm["y"], zd["y"]) and np.array_equal(zm["trial_ids"], zd["trial_ids"])
            assert zm["P_final"].shape == (1, len(zd["y"]), 3), zm["P_final"].shape
            assert np.allclose(zm["P_final"].sum(-1), 1.0, atol=1e-4)
        print("[save-member] aligned with the deep folds, rows sum to 1 -> OK")
        ok = not skipped
        get = lambda deep, w: np.mean([r["win"] for r in rows if r["deep"] == deep and r["w"] == w])
        d0, d5, d1 = get("ensemble", 0.0), get("ensemble", 0.5), get("ensemble", 1.0)
        print(f"[fusion] deep alone {d0:.3f} | fused(0.5) {d5:.3f} | EEG member alone {d1:.3f}")
        good = d5 > d0 + 0.05
        print(f"  fused beats the weak deep member by > 0.05 -> {'OK' if good else 'FAIL'}")
        ok &= good
        # guard: corrupt the saved labels of one fold -> must be refused
        z = dict(np.load(os.path.join(pdir, "VP001_session2.npz")))
        z["y"] = (z["y"] + 1) % 3
        np.savez(os.path.join(pdir, "VP001_session2.npz"), **z)
        rows2, skipped2 = run(dd, pdir, multiband=False, verbose=False)
        good = len(skipped2) == 1 and "refused" in skipped2[0][1]
        print(f"[guard] misaligned fold refused -> {'OK' if good else 'FAIL'}")
        ok &= good
        # normalisation: unnormalised deep rows must not change argmax-based scores
        P = rng.rand(5, 3) * 4
        good = np.allclose(norm_rows(P).sum(1), 1.0)
        print(f"[norm] rows sum to 1 -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--posteriors-dir")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-multiband", action="store_true")
    ap.add_argument("--mode", default="riem_disp", choices=["riem_disp", "riem", "EA", "raw"],
                    help="domain handling of the EEG member. riem_disp = Frechet centring + dispersion "
                         "(pre-registered main member); EA = per-session arithmetic-mean whitening (simpler, "
                         "equivalent in check_riemannian_eeg / fusion_controls: 0.6760 vs 0.6765 fused window).")
    ap.add_argument("--subjects", nargs="+")
    ap.add_argument("--out", help="write per-fold rows to this csv")
    ap.add_argument("--save-member", help="write the EEG member's own window posteriors to this folder")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not (a.data and a.posteriors_dir):
        ap.error("--data and --posteriors-dir are required unless --self-test")
    global MEMBER_MODE
    MEMBER_MODE = a.mode
    print(f"EEG member: multiband={'no' if a.no_multiband else 'yes'}, domain mode = {a.mode}")
    rows, skipped = run(a.data, a.posteriors_dir, multiband=not a.no_multiband, subjects=a.subjects,
                        save_member=a.save_member)
    for sid, why in skipped:
        print(f"[skipped] {sid}: {why}")
    if not rows:
        raise SystemExit("no fold could be fused")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("per-fold rows ->", a.out)
    summarise(rows)


if __name__ == "__main__":
    main()
