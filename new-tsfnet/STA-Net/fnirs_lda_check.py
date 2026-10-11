"""
fnirs_lda_check.py -- fast, model-free sanity check: "does the fNIRS tensor carry CROSS-SESSION class
information at all?"  (CPU, seconds per root for 26 subjects; no deep model, no GPU)

For every subject:
  1. take the lag-0 fNIRS segment (3 s, 16x16x30x2) of every window whose START lies in [t0, t1] s
     after the trial marker, and average over those windows and over time -> ONE feature map per TRIAL
     (HbO and HbR);
  2. shrinkage-LDA, leave-one-session-out (train = 2 sessions = 18 trials, test = 1 session = 9 trials).
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


def trial_features(root, sid):
    """returns X (n_trials, d), y, session (n_trials,) or None if unavailable."""
    d = os.path.join(root, sid)
    f = lambda k: np.load(os.path.join(d, f"{sid}_{k}.npy"))
    try:
        labels, sess, trial = f("labels"), f("session_ids"), f("trial_ids")
        fn = np.load(os.path.join(d, f"{sid}_fnirs_windows.npy"), mmap_mode="r")
    except FileNotFoundError as e:
        print(f"  [{sid}] skipped ({e})")
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
    X, y, s = [], [], []
    for t in np.unique(trial):
        idx = np.where((trial == t) & (offset >= args.t0) & (offset <= args.t1))[0]
        if len(idx) == 0:
            continue
        seg = np.asarray(fn[idx, 0], dtype=np.float32)              # (n, 16, 16, 30, 2) lag-0 segments
        m = seg.mean(axis=(0, 3))                                   # (16, 16, 2)
        if args.features == "pooled":
            m = m.reshape(8, 2, 8, 2, 2).mean(axis=(1, 3))
        elif args.features == "channels":
            m = m[FNIRS_PIXELS[:, 0], FNIRS_PIXELS[:, 1]]           # (36, 2)
        X.append(m.reshape(-1)); y.append(labels[idx[0]]); s.append(sess[idx[0]])
    if not X:
        return None
    return np.array(X), np.array(y), np.array(s)


def loso_lda(X, y, s):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    X = X.copy()
    if args.norm == "session":
        for k in np.unique(s):
            m = s == k
            X[m] = (X[m] - X[m].mean(0)) / (X[m].std(0) + 1e-8)
    elif args.norm == "subject":
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    hits, n = 0, 0
    for k in np.unique(s):
        tr, te = s != k, s == k
        clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        hits += int((pred == y[te]).sum()); n += int(te.sum())
    return hits / n, hits, n


def wilson(h, n, z=1.96):
    ph = h / n
    c = (ph + z * z / (2 * n)) / (1 + z * z / n)
    w = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - w, c + w


def main():
    roots = [r.split("=", 1) for r in args.root]
    subs = parse_subjects(args.subjects)
    print(f"features={args.features} norm={args.norm} windows with start in [{args.t0}, {args.t1}] s; chance = 0.333")
    acc = {lab: {} for lab, _ in roots}
    tot = {lab: [0, 0] for lab, _ in roots}
    for lab, root in roots:
        for sid in subs:
            r = trial_features(root, sid)
            if r is None:
                continue
            a, h, n = loso_lda(*r)
            acc[lab][sid] = a; tot[lab][0] += h; tot[lab][1] += n
        print(f"  [{lab}] done ({len(acc[lab])} subjects)")

    labs = [l for l, _ in roots]
    print("\n" + "subject".ljust(8) + "".join(l[:14].rjust(15) for l in labs))
    for sid in subs:
        if any(sid in acc[l] for l in labs):
            print(sid.ljust(8) + "".join((f"{acc[l][sid]:.3f}" if sid in acc[l] else "  -  ").rjust(15) for l in labs))
    print("\n" + "=" * 78 + "\nSUMMARY (trial-level LOSO LDA on fNIRS only)\n" + "=" * 78)
    ref = labs[0]
    for l in labs:
        v = np.array(list(acc[l].values()))
        h, n = tot[l]
        lo, hi = wilson(h, n)
        line = (f"  {l:<14} mean-over-subjects {v.mean():.4f} +- {v.std():.4f} | pooled {h / n:.4f} "
                f"[{lo:.3f}, {hi:.3f}] (n={n} trials) | subjects > 0.40: {(v > 0.40).sum()}/{len(v)}")
        if l != ref:
            common = [s for s in acc[l] if s in acc[ref]]
            if len(common) >= 6:
                from scipy.stats import wilcoxon
                a, b = np.array([acc[ref][s] for s in common]), np.array([acc[l][s] for s in common])
                try:
                    line += f" | vs {ref}: diff {np.mean(b - a):+.4f}, Wilcoxon p={wilcoxon(a, b).pvalue:.3g}"
                except ValueError:
                    pass
        print(line)
    print("\nNOTE: the pooled 95% interval ignores that trials of one subject are not independent; treat it as "
          "optimistic. With 9 test trials per fold a single subject is very noisy -- read the cohort numbers.")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["subject"] + labs)
            for sid in subs:
                w.writerow([sid] + [acc[l].get(sid, "") for l in labs])
        print(f"[written] {args.csv}")


if __name__ == "__main__":
    main()
