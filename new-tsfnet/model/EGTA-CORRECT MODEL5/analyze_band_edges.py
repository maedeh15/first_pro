"""
analyze_band_edges.py -- what did --eeg-filterbank learnable actually learn, across
every subject and fold?

WHY
===
A spot check on VP001's 3 checkpoints showed something specific: delta/alpha/beta/gamma
barely moved (<1 Hz in every fold), but theta consistently NARROWED and shifted down in
all 3 folds -- 4-8 Hz became roughly 2.5-5.5 Hz. That could be:
  (a) a real, subject-general signal the model found (theta narrower/lower matters), or
  (b) VP001-specific overfitting that happens to look consistent across its own 3 folds.
Only (a) is distinguishable from (b) by checking many subjects: if the same narrowing
shows up everywhere, it is (a); if different subjects move theta in different or opposite
directions, it is (b) -- i.e. the model is fitting fold-specific noise, consistent with
this project's repeated finding that added freedom hurts (run73-learnable itself scored
-0.19pp vs baseline, -1.40pp vs the fixed-filter variant).

WHAT IT DOES
============
Reads every VPxxx_session{1,2,3}_best.pt checkpoint in --results-dir, pulls out
eeg_filterbank.low_raw / band_raw (NO model construction, NO torch model needed -- these
are plain tensors in the state_dict), converts them to Hz edges with the exact formula in
GridFilterBank.band_edges_hz() (tsfnet_model_notebook.py), and reports, per band:
  * mean/SD low-edge and high-edge shift from the physiological initialisation, pooled
    over every fold of every subject;
  * SIGN CONSISTENCY: the fraction of folds whose shift has the majority sign, for low
    and high edges separately -- this is the number that distinguishes (a) from (b) above.
    Close to 1.0 = most folds agree on a direction (real/general signal). Close to 0.5 =
    folds disagree (idiosyncratic overfitting, no shared direction);
  * a per-subject table (mean shift per band, SD across that subject's 3 folds);
  * bandwidth change (high - low) per band, since narrowing is a distinct effect from
    shifting;
  * optionally, if --fixed-csv and --baseline-csv are given (the per-subject mean_acc
    columns from two baseline_all_subjects*.csv runs, e.g. the learnable run and the
    run60 baseline), the per-subject accuracy change is correlated against each
    subject's total band-edge movement -- to see whether subjects whose filters moved
    more lost more accuracy (overfitting) or gained more (learned signal).

USAGE
=====
    python analyze_band_edges.py --results-dir "E:\\...\\results_1798f4db95_seed42"
    python analyze_band_edges.py --results-dir "...\\results_1798f4db95_seed42" ^
        --fixed-csv "...\\baseline_all_subjects_..._1798f4db95.csv" ^
        --baseline-csv "...\\baseline_all_subjects_..._40b7d8ea53.csv"

No GPU, no preprocessed data, no tsfnet_model_notebook.py needed -- only the .pt files.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
INIT_LOW = np.array([0.0, 4.0, 8.0, 13.0, 30.0])
INIT_HIGH = np.array([4.0, 8.0, 13.0, 30.0, 50.0])
HZ_SCALE = 10.0     # GridFilterBank.__init__ default
SFREQ = 200.0       # EEG sampling rate this project uses throughout
NYQ = SFREQ / 2.0

p = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--results-dir", required=True,
              help="Folder holding VPxxx_session{1,2,3}_best.pt (one results_<hash>_seed<N> run).")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-csv", default=None,
              help="Per-fold band-edge CSV. Default: band_edges.csv inside --results-dir.")
p.add_argument("--fixed-csv", default=None,
              help="baseline_all_subjects*.csv of the learnable run (subject_id,...,mean_acc,...).")
p.add_argument("--baseline-csv", default=None,
              help="baseline_all_subjects*.csv of the run to compare against (e.g. run60). "
                   "Both --fixed-csv and --baseline-csv must be given together.")
ARGS = p.parse_args()


def band_edges_hz(low_raw: torch.Tensor, band_raw: torch.Tensor):
    """Exact reimplementation of GridFilterBank.band_edges_hz() (mode='learnable'),
    operating directly on the two parameter tensors -- no model construction."""
    lo = (low_raw * HZ_SCALE).abs().clamp(max=NYQ - 1.0)
    hi = torch.maximum(lo + (band_raw * HZ_SCALE).abs(), lo + 0.5)
    hi = torch.minimum(hi, torch.full_like(lo, NYQ))
    return lo.numpy(), hi.numpy()


def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


def load_acc_csv(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id") and row.get("mean_acc"):
                d[row["subject_id"]] = float(row["mean_acc"])
    return d


def main():
    if not os.path.isdir(ARGS.results_dir):
        sys.exit(f"ERROR: --results-dir does not exist: {ARGS.results_dir}")
    subjects = parse_subjects(ARGS.subjects)
    out_csv = ARGS.out_csv or os.path.join(ARGS.results_dir, "band_edges.csv")

    rows = []          # one row per (subject, session, band)
    missing_fb = []     # checkpoints with no learnable filterbank (wrong --eeg-filterbank)
    missing_ckpt = []

    for sid in subjects:
        for sess in (1, 2, 3):
            ckpt_path = os.path.join(ARGS.results_dir, f"{sid}_session{sess}_best.pt")
            if not os.path.isfile(ckpt_path):
                missing_ckpt.append(ckpt_path)
                continue
            sd = torch.load(ckpt_path, map_location="cpu")
            prefix = None
            for k in sd:
                if k.endswith("low_raw"):
                    prefix = k[: -len(".low_raw")]
                    break
            if prefix is None:
                missing_fb.append(ckpt_path)
                continue
            lo, hi = band_edges_hz(sd[f"{prefix}.low_raw"], sd[f"{prefix}.band_raw"])
            for b, il, ih, l, h in zip(BANDS, INIT_LOW, INIT_HIGH, lo, hi):
                rows.append({"subject_id": sid, "session": sess, "band": b,
                            "init_low": il, "init_high": ih, "learned_low": float(l),
                            "learned_high": float(h), "low_shift": float(l - il),
                            "high_shift": float(h - ih), "bandwidth_init": il - ih,
                            "bandwidth_learned": float(h - l),
                            "bandwidth_shift": float((h - l) - (ih - il))})

    if missing_ckpt:
        print(f"[skip] {len(missing_ckpt)} checkpoint(s) not found (e.g. {missing_ckpt[0]})")
    if missing_fb:
        print(f"[skip] {len(missing_fb)} checkpoint(s) have no learnable filterbank params "
              f"-- were they trained with --eeg-filterbank learnable? (e.g. {missing_fb[0]})")
    if not rows:
        sys.exit("ERROR: no checkpoint in --results-dir had learnable filterbank parameters.")

    n_subjects = len(set(r["subject_id"] for r in rows))
    n_folds = len(set((r["subject_id"], r["session"]) for r in rows))
    print("=" * 78)
    print(f"LEARNED BAND-EDGE ANALYSIS | {n_subjects} subjects, {n_folds} folds "
          f"(from {ARGS.results_dir})")
    print("=" * 78)

    print()
    print(f"{'band':8s} {'init Hz':>12s}  | {'low shift (Hz)':^22s} | {'high shift (Hz)':^22s} "
          f"| {'bandwidth':>16s}")
    print(f"{'':8s} {'':>12s}  | {'mean':>8s}{'SD':>7s}{'sign%':>7s} | "
          f"{'mean':>8s}{'SD':>7s}{'sign%':>7s} | {'init->learned':>16s}")
    for b, il, ih in zip(BANDS, INIT_LOW, INIT_HIGH):
        ls = np.array([r["low_shift"] for r in rows if r["band"] == b])
        hs = np.array([r["high_shift"] for r in rows if r["band"] == b])
        bw0 = np.array([r["bandwidth_init"] for r in rows if r["band"] == b]) * -1  # init_low - init_high is negative; want width
        bw1 = np.array([r["bandwidth_learned"] for r in rows if r["band"] == b])
        sign_l = max((ls > 0).mean(), (ls < 0).mean()) if (ls != 0).any() else float("nan")
        sign_h = max((hs > 0).mean(), (hs < 0).mean()) if (hs != 0).any() else float("nan")
        print(f"{b:8s} [{il:4.1f},{ih:4.1f}]  | {ls.mean():+8.2f}{ls.std(ddof=1):7.2f}"
              f"{sign_l*100:6.0f}% | {hs.mean():+8.2f}{hs.std(ddof=1):7.2f}{sign_h*100:6.0f}% "
              f"| {(ih-il):6.1f} -> {bw1.mean():5.1f}")

    print()
    print("sign% = fraction of folds agreeing on the shift's direction (majority sign).")
    print("        ~100% across many SUBJECTS (not just one subject's own folds) would mean")
    print("        the model found a shared, subject-general signal; ~50% means folds")
    print("        disagree on direction -- consistent with fold-specific overfitting.")

    # ---- per-subject summary (mean shift per band, collapsed over that subject's folds) --
    print()
    print("PER-SUBJECT mean shift (Hz), low edge / high edge, one row per band is too wide --")
    print("showing THETA only here (the band that moved most in the VP001 spot check); full")
    print("detail for all bands is in the CSV.")
    subj_list = sorted(set(r["subject_id"] for r in rows))
    print(f"  {'subject':8s} {'theta low':>10s} {'theta high':>11s} {'theta width':>12s}  "
          f"{'n folds':>8s}")
    theta_subj_low, theta_subj_high = {}, {}
    for sid in subj_list:
        sub = [r for r in rows if r["subject_id"] == sid and r["band"] == "theta"]
        l = np.mean([r["low_shift"] for r in sub])
        h = np.mean([r["high_shift"] for r in sub])
        theta_subj_low[sid], theta_subj_high[sid] = l, h
        print(f"  {sid:8s} {l:+10.2f} {h:+11.2f} {np.mean([r['bandwidth_learned'] for r in sub]):12.2f}  "
              f"{len(sub):8d}")
    ls_all = np.array(list(theta_subj_low.values()))
    hs_all = np.array(list(theta_subj_high.values()))
    print()
    print(f"ACROSS SUBJECTS (theta, {len(subj_list)} subjects, each averaged over its own folds):")
    print(f"  low  edge: mean {ls_all.mean():+.2f} Hz, SD {ls_all.std(ddof=1):.2f}, "
          f"{int((ls_all < 0).sum())}/{len(ls_all)} subjects shifted DOWN, "
          f"{int((ls_all > 0).sum())} shifted up")
    print(f"  high edge: mean {hs_all.mean():+.2f} Hz, SD {hs_all.std(ddof=1):.2f}, "
          f"{int((hs_all < 0).sum())}/{len(hs_all)} subjects shifted DOWN, "
          f"{int((hs_all > 0).sum())} shifted up")
    if len(ls_all) >= 3:
        from scipy import stats
        tlo = stats.ttest_1samp(ls_all, 0.0)
        thi = stats.ttest_1samp(hs_all, 0.0)
        print(f"  one-sample t-test (mean shift != 0, across subjects): "
              f"low p={tlo.pvalue:.4f}  high p={thi.pvalue:.4f}")

    # ---- optional: correlate with accuracy -----------------------------------------------
    if ARGS.fixed_csv and ARGS.baseline_csv:
        print()
        print("=" * 78)
        print("BAND-EDGE MOVEMENT vs ACCURACY CHANGE (learnable - baseline, per subject)")
        print("=" * 78)
        fixed_acc = load_acc_csv(ARGS.fixed_csv)
        base_acc = load_acc_csv(ARGS.baseline_csv)
        common = [s for s in subj_list if s in fixed_acc and s in base_acc]
        if len(common) < 3:
            print(f"  only {len(common)} subjects in common between the two CSVs -- skipping.")
        else:
            acc_diff = np.array([fixed_acc[s] - base_acc[s] for s in common])
            # total movement = sum of |low_shift| + |high_shift| over all 5 bands, averaged over folds
            total_move = []
            for sid in common:
                sub = [r for r in rows if r["subject_id"] == sid]
                total_move.append(np.mean([abs(r["low_shift"]) + abs(r["high_shift"]) for r in sub]))
            total_move = np.array(total_move)
            r = float(np.corrcoef(total_move, acc_diff)[0, 1])
            print(f"  subjects compared: {len(common)}")
            print(f"  corr(total |edge shift| across all bands, accuracy change) = {r:+.3f}")
            print("  negative => subjects whose filters moved more LOST more accuracy "
                  "(consistent with overfitting)")
            print("  positive => subjects whose filters moved more GAINED more accuracy "
                  "(consistent with a useful learned signal)")
            theta_move = np.array([abs(theta_subj_low[s]) + abs(theta_subj_high[s]) for s in common])
            r2 = float(np.corrcoef(theta_move, acc_diff)[0, 1])
            print(f"  corr(theta-only |edge shift|, accuracy change)            = {r2:+.3f}")
    elif ARGS.fixed_csv or ARGS.baseline_csv:
        print()
        print("[note] --fixed-csv and --baseline-csv must both be given to run the "
              "accuracy-correlation section; skipping it.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"[written] {out_csv}  ({len(rows)} rows: subject x session x band)")


if __name__ == "__main__":
    main()
