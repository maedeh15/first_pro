"""
fnirs_lda_check.py -- fast, model-free sanity check: "does the fNIRS tensor carry CROSS-SESSION class
information at all?"  (CPU, seconds per root for 26 subjects; no deep model, no GPU)

For every subject:
  1. take the lag-0 fNIRS segment (3 s, 16x16x30x2) of every window whose START lies in [t0, t1] s
     after the trial marker, and average over those windows and over time -> ONE feature map per TRIAL
     (HbO and HbR);
  2. shrinkage-LDA, leave-one-session-out (train = 2 sessions = 18 trials, test = 1 session = 9 trials).
(--unit window instead trains/tests on single WINDOWS -- the unit the network sees -- and also reports the trial-level
accuracy obtained by averaging the window posteriors inside each test trial; --lags L averages the first L lag
segments of each window, L=11 = exactly the fNIRS input of one STA-Net sample, pooled over time.)
Reports per-subject mean accuracy (chance = 33.3 %), the pooled trial-level accuracy with a 95 % Wilson
interval, and paired Wilcoxon tests of the FIRST root against every other root.

Typical use -- compare STA-Net's current tensors, your TSFNet tensors and STA-Net fNIRS variants:
  python fnirs_lda_check.py ^
      --root STA=E:\\...\\STANET_PAPER_EXACT_nback ^
      --root TSF=E:\\...\\TSFNET_PAPER_EXACT_nback ^
      --root STA_bp_off=E:\\...\\fv_bpoff --root STA_z=E:\\...\\fv_z --root STA_sess=E:\\...\\fv_sess ^
      --root STA_tsflike=E:\\...\\fv_tsflike
Interpretation: if TSF is clearly above 33 % and STA is at chance, the STA-Net fNIRS PREPROCESSING loses the
information (and the variant roots show which step brings it back). If all roots sit at chance, the problem is
not the preprocessing; look at the model.

Layout notes
  * Features default to a 2x2-average-pooled grid (8x8x2 = 128 features), valid for ANY of the 16x16 layouts
    (STA-Net hard-coded pixels or TSFNet azimuthal projection). `--features grid` = full 16x16x2,
    `--features channels` = the 36 channel pixels x 2 (ONLY meaningful for STA-Net-layout roots).
  * Window start offsets: taken from {sid}_window_ids.npy when present (STA-Net files), otherwise estimated as
    first_offset + rank of the window inside its trial (TSFNet files: -2, -1, ... assumes no window was dropped;
    a warning is printed when the per-trial window count differs from --expect-windows for that root).
  * --norm session  standardises every feature within each session (TRANSDUCTIVE, unlabeled inputs only; a cheap
    emulation of per-session normalisation on tensors that were saved without it). --norm subject = over all trials.
"""
import argparse
import os
import sys
import csv

import numpy as np

# Official 16x16 pixel positions ([row, col]) of the 36 fNIRS channels (same table as Preproc_stanet_nback.py).
# Embedded here so this script has NO dependency on stanet_common.py (only used by --features channels).
FNIRS_PIXELS = np.array([
    [2, 4], [3, 4], [1, 5], [2, 5], [1, 7], [3, 6], [2, 7], [3, 8], [1, 8], [2, 9], [1, 9], [6, 4], [7, 5],
    [7, 3], [8, 4], [10, 5], [11, 6], [11, 4], [12, 5], [3, 10], [2, 11], [3, 12], [1, 11], [2, 12], [6, 12],
    [7, 13], [7, 11], [8, 12], [10, 11], [11, 12], [11, 10], [12, 11], [12, 8], [13, 7], [13, 9], [14, 8]])

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--root", action="append", required=True,
               help="LABEL=PATH to a preprocessing output folder (holding VPxxx/). Repeat; first one is the reference.")
p.add_argument("--subjects", default="1-26")
p.add_argument("--t0", type=float, default=6.0, help="first window START (s after marker) to include")
p.add_argument("--t1", type=float, default=30.0, help="last window START (s after marker) to include")
p.add_argument("--first-offset", type=float, default=-2.0, help="start (s) of window rank 0 when no window_ids file")
p.add_argument("--expect-windows", type=int, default=None, help="warn if a trial has another window count")
p.add_argument("--features", choices=["pooled", "grid", "channels"], default="pooled")
p.add_argument("--unit", choices=["trial", "window"], default="trial",
               help="trial: one averaged sample per trial; window: one sample per window (what the network classifies)")
p.add_argument("--lags", type=int, default=1, help="--unit window: average the first L lag segments of each window (L=11: full STA-Net input)")
p.add_argument("--norm", choices=["none", "session", "subject"], default="none")
p.add_argument("--csv", default=None, help="write the per-subject table here")
args = p.parse_args()


def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-"); ids += list(range(int(a), int(b) + 1))
        elif part.strip():
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


_missing_reported = set()


def load_samples(root, sid):
    """returns X (n, d), y (n,), session (n,), group/trial id (n,) -- or None if the subject is unavailable."""
    d = os.path.join(root, sid)
    f = lambda k: np.load(os.path.join(d, f"{sid}_{k}.npy"))
    try:
        labels, sess, trial = f("labels"), f("session_ids"), f("trial_ids")
        fn = np.load(os.path.join(d, f"{sid}_fnirs_windows.npy"), mmap_mode="r")
    except FileNotFoundError as e:
        if root not in _missing_reported:        # say it once per root, not once per subject
            _missing_reported.add(root)
            print(f"  !! [{os.path.basename(root.rstrip(chr(92) + '/'))}] {e}\n"
                  f"  !! this root has no usable files for {sid} (further subjects of this root are skipped silently). "
                  f"Check the folder contents with:  dir \"{d}\"")
        return None
    wid_path = os.path.join(d, f"{sid}_window_ids.npy")
    if os.path.exists(wid_path):
        offset = args.first_offset + np.load(wid_path)
    else:
        offset = np.empty(len(trial))
        for t in np.unique(trial):
            m = np.where(trial == t)[0]
            offset[m] = args.first_offset + np.arange(len(m))
            if args.expect_windows is not None and len(m) != args.expect_windows:
                print(f"  [{sid}] WARNING trial {t} has {len(m)} windows (expected {args.expect_windows}); offsets are ranks")
    L = min(args.lags, fn.shape[1])

    def feat(m):                                       # m: (16, 16, 2) -> feature vector
        if args.features == "pooled":
            m = m.reshape(8, 2, 8, 2, 2).mean(axis=(1, 3))
        elif args.features == "channels":
            m = m[FNIRS_PIXELS[:, 0], FNIRS_PIXELS[:, 1]]
        return m.reshape(-1)

    X, y, s, g = [], [], [], []
    for t in np.unique(trial):
        idx = np.where((trial == t) & (offset >= args.t0) & (offset <= args.t1))[0]
        if len(idx) == 0:
            continue
        if args.unit == "trial":
            seg = np.asarray(fn[idx, 0], dtype=np.float32)                   # (n, 16, 16, 30, 2) lag-0 segments
            X.append(feat(seg.mean(axis=(0, 3)))); y.append(labels[idx[0]]); s.append(sess[idx[0]]); g.append(t)
        else:
            for i in idx:                                                      # one sample per window
                seg = np.asarray(fn[i, :L], dtype=np.float32)                  # (L, 16, 16, 30, 2)
                X.append(feat(seg.mean(axis=(0, 3)))); y.append(labels[i]); s.append(sess[i]); g.append(t)
    if not X:
        return None
    return np.array(X), np.array(y), np.array(s), np.array(g)


def loso_lda(X, y, s, g):
    """returns (sample_hits, n_samples, trial_hits, n_trials); trial decision = argmax of the mean posterior of the
    test trial's samples (identical to the sample level when --unit trial)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = X.copy()
    if args.norm == "session":
        for k in np.unique(s):
            m = s == k
            X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    elif args.norm == "subject":
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    sh = sn = th = tn = 0
    for k in np.unique(s):
        tr, te = s != k, s == k
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        pred = clf.classes_[proba.argmax(1)]
        sh += int((pred == y[te]).sum()); sn += int(te.sum())
        for t in np.unique(g[te]):
            m = g[te] == t
            th += int(clf.classes_[proba[m].mean(0).argmax()] == y[te][m][0]); tn += 1
    return sh, sn, th, tn


def wilson(h, n, z=1.96):
    ph = h / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - w, c + w


def main():
    roots = [r.split("=", 1) for r in args.root]
    subs = parse_subjects(args.subjects)
    print(f"features={args.features} unit={args.unit}" + (f" lags={args.lags}" if args.unit == "window" else "")
          + f" norm={args.norm} windows with start in [{args.t0}, {args.t1}] s; chance = 0.333")
    acc = {lab: {} for lab, _ in roots}          # per-subject sample-level accuracy
    acct = {lab: {} for lab, _ in roots}         # per-subject trial-level accuracy
    tot = {lab: [0, 0, 0, 0] for lab, _ in roots}
    for lab, root in roots:
        for sid in subs:
            r = load_samples(root, sid)
            if r is None:
                continue
            sh, sn, th, tn = loso_lda(*r)
            acc[lab][sid] = sh / sn; acct[lab][sid] = th / tn
            for i, v in enumerate((sh, sn, th, tn)):
                tot[lab][i] += v
        print(f"  [{lab}] done ({len(acc[lab])} subjects)")

    labs = [l for l, _ in roots]
    print("\n" + "subject".ljust(8) + "".join(l[:14].rjust(15) for l in labs))
    for sid in subs:
        if any(sid in acc[l] for l in labs):
            print(sid.ljust(8) + "".join((f"{acc[l][sid]:.3f}" if sid in acc[l] else "  -  ").rjust(15) for l in labs))
    print("\n" + "=" * 78 + f"\nSUMMARY (LOSO LDA on fNIRS only, unit={args.unit})\n" + "=" * 78)
    ref = next((l for l in labs if acc[l]), None)
    for l in labs:
        if not acc[l]:
            print(f"  {l:<14} NO DATA (no subject could be loaded -- see the '!!' message above)")
            continue
        v = np.array(list(acc[l].values()))
        h, n, th, tn = tot[l]
        lo, hi = wilson(h, n)
        line = (f"  {l:<14} mean-over-subjects {v.mean():.4f} +- {v.std():.4f} | pooled {h / n:.4f} "
                f"[{lo:.3f}, {hi:.3f}] (n={n}) | subjects > 0.40: {(v > 0.40).sum()}/{len(v)}")
        if args.unit == "window":
            vt = np.array(list(acct[l].values()))
            line += f"\n  {'':<14} trial-level (mean posterior over a trial's windows): {vt.mean():.4f} +- {vt.std():.4f} | pooled {th / tn:.4f} (n={tn} trials)"
        if l != ref and ref is not None:
            common = [x for x in acc[l] if x in acc[ref]]
            if len(common) >= 6:
                from scipy.stats import wilcoxon
                a_, b_ = np.array([acc[ref][x] for x in common]), np.array([acc[l][x] for x in common])
                try:
                    line += f"\n  {'':<14} vs {ref}: diff {np.mean(b_ - a_):+.4f}, Wilcoxon p={wilcoxon(a_, b_).pvalue:.3g}"
                except ValueError:
                    pass
        print(line)
    print("\nNOTE: the pooled 95% interval ignores that samples of one subject are not independent (and windows of one trial "
          "are highly correlated); treat it as optimistic. With 9 test trials per fold a single subject is very noisy -- "
          "read the cohort numbers.")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["subject"] + labs)
            for sid in subs:
                w.writerow([sid] + [acc[l].get(sid, "") for l in labs])
        print(f"[written] {args.csv}")


if __name__ == "__main__":
    main()
