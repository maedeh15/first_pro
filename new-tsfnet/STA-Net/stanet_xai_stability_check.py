"""
stanet_xai_stability_check.py -- is STA-Net's EGTA attention explanation REPRODUCIBLE across
random seeds, or does every seed tell a different story about the same input?

STA-Net counterpart of your xai_stability_check.py.

WHY THIS EXISTS
STA-Net, MGFNet and MSTFDN each show attention-map figures from a SINGLE trained run. None asks
whether a second run -- same subject, same data, only the random seed changed -- would produce
the same explanation. If the attention pattern is mostly an artifact of one initialisation and
optimisation trajectory rather than a stable property of the subject's data, then "the model
attends to lagged window X" is not a scientific claim, it is a coin flip that landed on X this
time. This measures it directly, reusing seeds you have ALREADY trained (42, 999, 2024) -- no
new training needed.

WHAT IT MEASURES
For each subject and each held-out session, every seed's checkpoint is run on the EXACT same
held-out windows. WindowLoader iterates in index order and never shuffles, so window i is the
same physical window for every seed, and the attention vectors can be compared window for
window rather than as a vague pooled picture:

  1. PER-WINDOW pairwise Spearman rho between two seeds' attention vectors over the l lagged
     fNIRS windows, for that same window. Averaged -> one within-window cross-seed consistency
     number per subject.
  2. ICC(2,1) (Shrout & Fleiss 1979, two-way random effects, single rater, absolute agreement)
     on each subject's SUBJECT-LEVEL attention profile (mean attention per lagged window,
     pooled over all that subject's windows), treating the l lagged windows as "items" and the
     seeds as "raters". The standard reliability statistic for "do independent measurements
     agree", applied to seeds instead of human raters. Koo & Li (2016): >= 0.75 excellent,
     0.5-0.75 moderate, < 0.5 poor.

Both are reported per subject -- so a committee member can see exactly which subjects'
explanations are trustworthy -- and at the group level.

A CAVEAT WORTH REPORTING: ICC(2,1) here is computed over the l lagged windows as items. With
the paper-exact preprocessing l = 11, which is a small number of items, so the ICC has wide
sampling uncertainty; with the TSFNet-comparable preprocessing l = 21 it is better conditioned.
Interpret a single subject's ICC loosely; the group mean is the reportable number.

USAGE
=====
    python stanet_xai_stability_check.py ^
        --data-root "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_nback" ^
        --results-dirs "E:\\...\\STANET_TSFCOMP_seed42" "E:\\...\\_seed999" "E:\\...\\_seed2024" ^
        --seeds 42 999 2024 --fnirs-scale 1.0

All seeds must come from runs on the SAME preprocessing -- the script refuses to continue if
the seeds disagree on a session's window count, since that means mismatched runs. Each subject
is loaded from disk ONCE and reused for every seed and fold. No GPU required, much faster with
one.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

from stanet_xai_signals import (SubjectData, add_common_args, collect_fold_signals,
                                parse_subjects, run_header)

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dirs", nargs="+", required=True,
               help="One results folder per seed, SAME ORDER as --seeds.")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 999, 2024])
p.add_argument("--out-dir", default=None, help="default: <data-root>/xai_stability_check")
add_common_args(p)
ARGS = p.parse_args()


def icc_2_1(data):
    """Shrout & Fleiss (1979) ICC(2,1): two-way random effects, single rater, absolute
    agreement. data: (n_items, k_raters)."""
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = data.mean()
    ss_total = ((data - grand) ** 2).sum()
    ss_rows = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((data.mean(axis=0) - grand) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denom) if denom > 0 else float("nan")


def subject_stability(data, results_dirs, seeds, seed_pairs, device, batch_size, adabn):
    """Returns (per-window pairwise rhos, per-seed subject-level profile) or None if any
    checkpoint is missing / the seeds disagree on window counts."""
    rhos = []
    per_seed_windows = [[] for _ in seeds]
    for session in (1, 2, 3):
        mats = []
        for seed, rdir in zip(seeds, results_dirs):
            ckpt = os.path.join(rdir, f"{data.sid}_session{session}_best.pt")
            if not os.path.isfile(ckpt):
                print(f"[{data.sid} s{session} seed{seed}] SKIP -- checkpoint not found")
                return None
            _, att = collect_fold_signals(data, data.session_idx(session), ckpt, device,
                                          batch_size, adabn, return_att=True)
            mats.append(att)
        n_win = mats[0].shape[0]
        if any(m.shape[0] != n_win for m in mats):
            print(f"[{data.sid} s{session}] SKIP -- seeds disagree on window count "
                  f"({[m.shape[0] for m in mats]}); checkpoints are from mismatched runs.")
            return None
        for i, m in enumerate(mats):
            per_seed_windows[i].append(m)
        # Spearman per window, vectorised over windows: rank within each row, then correlate.
        ranks = [np.apply_along_axis(lambda r: np.argsort(np.argsort(r)), 1, m).astype(float)
                 for m in mats]
        for a, b in seed_pairs:
            ra, rb = ranks[a], ranks[b]
            ra = ra - ra.mean(axis=1, keepdims=True)
            rb = rb - rb.mean(axis=1, keepdims=True)
            den = np.sqrt((ra ** 2).sum(axis=1) * (rb ** 2).sum(axis=1))
            with np.errstate(invalid="ignore", divide="ignore"):
                r = (ra * rb).sum(axis=1) / den
            rhos.extend(r[~np.isnan(r)].tolist())
    profiles = [np.concatenate(w, axis=0).mean(axis=0) for w in per_seed_windows]
    return rhos, profiles


def main():
    if len(ARGS.results_dirs) != len(ARGS.seeds):
        sys.exit(f"ERROR: --results-dirs has {len(ARGS.results_dirs)} entries but --seeds has "
                 f"{len(ARGS.seeds)}. Pass one folder per seed, in the same order.")
    out_dir = ARGS.out_dir or os.path.join(ARGS.data_root, "xai_stability_check")
    os.makedirs(out_dir, exist_ok=True)
    run_header("STA-Net CROSS-SEED EXPLANATION-STABILITY CHECK", ARGS.data_root, ARGS.results_dirs,
               ARGS.adabn, ARGS.fnirs_scale, extra=f"seeds = {ARGS.seeds}")
    for s, d in zip(ARGS.seeds, ARGS.results_dirs):
        print(f"  seed {s:>5}: {d}")

    seed_pairs = list(itertools.combinations(range(len(ARGS.seeds)), 2))
    per_subject, seg_lens = {}, set()
    for sid in parse_subjects(ARGS.subjects):
        try:
            data = SubjectData(ARGS.data_root, sid, ARGS.fnirs_scale)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            continue
        seg_lens.add(data.seg_len)
        try:
            got = subject_stability(data, ARGS.results_dirs, ARGS.seeds, seed_pairs,
                                    ARGS.device, ARGS.batch_size, ARGS.adabn)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            got = None
        finally:
            del data
        if got is None:
            print(f"[{sid}] SKIP -- incomplete checkpoint set across seeds/sessions.")
            continue
        rhos, profiles = got
        icc = icc_2_1(np.stack(profiles, axis=1))          # (l lagged windows, k seeds)
        mean_rho = float(np.mean(rhos)) if rhos else np.nan
        sd_rho = float(np.std(rhos, ddof=1)) if len(rhos) > 1 else np.nan
        per_subject[sid] = {"n_windows": len(rhos) // max(1, len(seed_pairs)),
                            "mean_pairwise_rho": mean_rho, "sd_pairwise_rho": sd_rho,
                            "icc_2_1": icc}
        print(f"[{sid}] n_windows={per_subject[sid]['n_windows']:4d}  "
              f"mean cross-seed rho={mean_rho:+.3f} (SD {sd_rho:.3f})  ICC(2,1)={icc:+.3f}")

    ids = sorted(per_subject)
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects produced usable cross-seed data.")

    rhos = np.array([per_subject[s]["mean_pairwise_rho"] for s in ids])
    iccs = np.array([per_subject[s]["icc_2_1"] for s in ids])
    rhos_ok, iccs_ok = rhos[~np.isnan(rhos)], iccs[~np.isnan(iccs)]
    l = sorted(seg_lens)

    print("\n" + "=" * 100)
    print(f"CROSS-SEED EXPLANATION-STABILITY SUMMARY -- {len(ids)} subjects, seeds={ARGS.seeds}, "
          f"l={l} lagged fNIRS windows")
    print("=" * 100)
    print(f"Per-window pairwise Spearman rho ({len(seed_pairs)} seed-pairs, same exact held-out window): "
          f"mean={rhos_ok.mean():+.3f}, SD={rhos_ok.std(ddof=1):.3f}, median={np.median(rhos_ok):+.3f}")
    print(f"ICC(2,1) on the subject-level attention profile ({l} lagged windows x {len(ARGS.seeds)} seeds): "
          f"mean={iccs_ok.mean():+.3f}, SD={iccs_ok.std(ddof=1):.3f}")
    excellent = int((iccs_ok >= 0.75).sum())
    moderate = int(((iccs_ok >= 0.5) & (iccs_ok < 0.75)).sum())
    poor = int((iccs_ok < 0.5).sum())
    print(f"ICC categories (Koo & Li 2016): excellent (>=0.75) {excellent}/{len(iccs_ok)}, "
          f"moderate (0.5-0.75) {moderate}/{len(iccs_ok)}, poor (<0.5) {poor}/{len(iccs_ok)}")

    print("\nREADING")
    if iccs_ok.mean() >= 0.75:
        print("  EGTA's attention explanation is HIGHLY REPRODUCIBLE across independent training runs: which\n"
              "  lagged fNIRS windows it attends to for a given subject is a stable property of that subject's\n"
              "  data, not an artifact of one random initialisation. This is a reliability claim none of the\n"
              "  three reference papers can make, since each shows a single trained run.")
    elif iccs_ok.mean() >= 0.5:
        print("  MODERATE cross-seed reproducibility: broadly consistent, but with enough seed-to-seed\n"
              "  variation that any single-run attention figure (as in STA-Net/MGFNet/MSTFDN) should be read\n"
              "  as indicative rather than a precise, re-derivable measurement.")
    else:
        print("  EGTA's attention explanation is NOT reliably reproducible across seeds: a different random\n"
              "  initialisation gives a materially different story about which lagged fNIRS windows 'matter'\n"
              "  for the same subject. This is an important and honest finding -- single-run attention figures,\n"
              "  the only kind the three reference papers show, should not be trusted as stable\n"
              "  characterisations of subject-level physiology, and this check is itself a methodological\n"
              "  contribution the literature is missing.")
    if max(l) <= 11:
        print(f"  Caveat: ICC here is computed over only {max(l)} items (lagged windows), so per-subject ICCs\n"
              "  are noisy -- quote the group mean, not an individual subject's value.")

    with open(os.path.join(out_dir, "xai_stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": ARGS.seeds, "results_dirs": ARGS.results_dirs, "seg_len": l,
                   "per_subject": per_subject,
                   "group": {"n_subjects": len(ids),
                             "mean_pairwise_rho": float(rhos_ok.mean()),
                             "sd_pairwise_rho": float(rhos_ok.std(ddof=1)),
                             "mean_icc": float(iccs_ok.mean()), "sd_icc": float(iccs_ok.std(ddof=1)),
                             "n_excellent": excellent, "n_moderate": moderate, "n_poor": poor}},
                  f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_stability_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_windows", "mean_pairwise_rho", "sd_pairwise_rho", "icc_2_1"])
        for s in ids:
            r = per_subject[s]
            w.writerow([s, r["n_windows"], r["mean_pairwise_rho"], r["sd_pairwise_rho"], r["icc_2_1"]])
    print(f"\n[written] {out_dir}: xai_stability_summary.json, xai_stability_per_subject.csv")


if __name__ == "__main__":
    main()
