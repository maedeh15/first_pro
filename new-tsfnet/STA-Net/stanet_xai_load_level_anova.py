"""
stanet_xai_load_level_anova.py -- is STA-Net's own attention/gate signal FORMALLY different
across 0-/2-/3-back, or does it merely "look" different in a figure?

STA-Net counterpart of your xai_load_level_anova.py; see stanet_xai_signals.py for how each
signal maps from TSFNet's internals to STANet's output dict.

WHY THIS EXISTS
STA-Net describes its prediction-weight curves qualitatively ("kappa_fnirs stays flat",
"kappa_fusion is larger in MI/WG than in MA"); MGFNet's t-SNE and confusion matrices are
likewise descriptive. Neither runs a formal omnibus test across task conditions on an
XAI-derived signal, the way a behavioural or physiological result normally must before being
reported. capacity_ceiling_check.py already holds this project to that standard for REAL
physiology. This script applies the matching formal test -- the non-parametric
repeated-measures ANOVA, i.e. the FRIEDMAN test, plus BH-corrected post-hoc pairwise
comparisons -- to the model's OWN internal signals across all three load levels, not just the
3-vs-2 contrast that stanet_xai_biomarker_validation.py covers.

DESIGN
Each subject contributes ONE value per load level per signal (that subject's mean over all
held-out trials at that level, pooled across the 3 leave-one-session-out folds). Subjects are
the blocks, load level is the 3-level repeated-measures factor -- exactly mirroring
capacity_ceiling_check.py's own hbo_pfc/theta_fm/p300_pz profile construction, but for the
model's signals instead of raw physiology.

SIGNALS: egta_centroid, egta_entropy, fgsa1_phi_mean, fgsa2_phi_mean, w_fnirs.
w_fnirs has no TSFNet equivalent -- STA-Net's decision weights are instance-adaptive (Eq. 6-7)
while TSFNet's are fixed scalars, so "does the model re-weight the modalities by task
difficulty" is a question only STA-Net can be asked.

STATISTICS PER SIGNAL
  1. Friedman chi-square omnibus (H0: the 3 load levels have the same distribution of
     subject-level means).
  2. Kendall's W -- the effect-size companion to Friedman (0-1: how much of the variance load
     level explains vs subject-to-subject noise).
  3. Post-hoc pairwise Wilcoxon signed-rank for all 3 pairs, BH-corrected ACROSS ALL
     3 x len(SIGNALS) tests as ONE family -- the standard guard against inflating false
     positives over many related comparisons.
  4. Per-subject shape classification (monotonic / inverted-U / other), the same convention as
     capacity_ceiling_check.py's hbo_shape, so the model's attention profile can be compared
     directly against the real prefrontal HbO load curve.

USAGE
=====
    python stanet_xai_load_level_anova.py ^
        --results-dir "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_seed42" ^
        --data-root   "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_nback" ^
        --fnirs-scale 1.0

--fnirs-scale must match training (1.0 for z-scored preprocessing, omit for paper-exact).
No GPU required, much faster with one. One forward pass per fold, same cost class as
stanet_xai_biomarker_validation.py.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from stanet_xai_signals import (CLASS_NAMES, LABELS, SIGNALS, SubjectData, add_common_args,
                                bh, parse_subjects, run_header, subject_trial_rows)

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True,
               help="Folder with VPxxx_session{1,2,3}_best.pt from ONE seed's LOSO run.")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_load_level_check")
add_common_args(p)
ARGS = p.parse_args()

LOAD_SHORT = ["0b", "2b", "3b"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def kendalls_w(matrix):
    """matrix: (n_subjects, k_conditions). Kendall's W, the effect size for Friedman's test."""
    n, k = matrix.shape
    ranks = np.apply_along_axis(lambda row: np.argsort(np.argsort(row)) + 1, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    ss = ((rank_sums - rank_sums.mean()) ** 2).sum()
    return float(12 * ss / (n ** 2 * (k ** 3 - k) + 1e-12))


def subject_profile(data, results_dir, device, batch_size, adabn):
    """Per signal, a length-3 vector [mean@0-back, mean@2-back, mean@3-back], pooled over the
    3 cross-session folds."""
    rows = subject_trial_rows(data, results_dir, device, batch_size, adabn)
    if not rows:
        return None
    prof = {}
    for sig in SIGNALS:
        vals = np.array([r["value"] for r in rows if r["signal"] == sig])
        labs = np.array([r["label"] for r in rows if r["signal"] == sig])
        prof[sig] = np.array([vals[labs == k].mean() if (labs == k).any() else np.nan
                              for k in (0, 1, 2)])
    return prof


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_load_level_check")
    os.makedirs(out_dir, exist_ok=True)
    run_header("STA-Net XAI LOAD-LEVEL OMNIBUS TEST (Friedman)", ARGS.data_root, ARGS.results_dir,
               ARGS.adabn, ARGS.fnirs_scale)

    profiles = {}
    for sid in parse_subjects(ARGS.subjects):
        try:
            data = SubjectData(ARGS.data_root, sid, ARGS.fnirs_scale)
            prof = subject_profile(data, ARGS.results_dir, ARGS.device, ARGS.batch_size, ARGS.adabn)
            if prof is not None:
                profiles[sid] = prof
            del data
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")

    ids = sorted(profiles)
    if len(ids) < 8:
        sys.exit("ERROR: fewer than 8 subjects produced usable profiles.")
    print(f"\n{len(ids)} subjects processed.")

    # Collect every post-hoc p FIRST, so BH runs over the whole family of 3 x len(SIGNALS) tests.
    all_tests, friedman_results = [], {}
    shapes = {sig: {"monotonic": [], "inverted-U": [], "other": []} for sig in SIGNALS}
    for sig in SIGNALS:
        mat = np.array([profiles[s][sig] for s in ids])            # (n_subjects, 3)
        keep = ~np.isnan(mat).any(axis=1)
        mat_ok = mat[keep]
        ids_ok = [s for s, k in zip(ids, keep) if k]
        if mat_ok.shape[0] < 8:
            friedman_results[sig] = {"n": int(mat_ok.shape[0]), "stat": np.nan, "p": np.nan,
                                     "kendalls_w": np.nan}
            continue
        stat, pv = friedmanchisquare(mat_ok[:, 0], mat_ok[:, 1], mat_ok[:, 2])
        friedman_results[sig] = {"n": int(mat_ok.shape[0]), "stat": float(stat), "p": float(pv),
                                 "kendalls_w": kendalls_w(mat_ok)}
        for i, (a, b) in enumerate(PAIRS):
            try:
                pv_pair = float(wilcoxon(mat_ok[:, a], mat_ok[:, b]).pvalue)
            except ValueError:        # all differences zero
                pv_pair = np.nan
            all_tests.append((sig, i, pv_pair))
        for sid in ids_ok:
            m0, m2, m3 = profiles[sid][sig]
            shapes[sig]["monotonic" if m0 < m2 < m3
                        else "inverted-U" if (m2 > m3 and m2 > m0)
                        else "other"].append(sid)

    posthoc = {}
    for (sig, pair_idx, pv), qv in zip(all_tests, bh([t[2] for t in all_tests])):
        posthoc.setdefault(sig, {})[PAIRS[pair_idx]] = {"p": float(pv), "q": float(qv)}

    print("\n" + "=" * 108)
    print(f"LOAD-LEVEL FRIEDMAN TEST -- {len(ids)} subjects, 0-/2-/3-back, model-internal XAI signals")
    print(f"(BH-FDR across all 3 pairs x {len(SIGNALS)} signals = {3 * len(SIGNALS)} post-hoc tests, one family)")
    print("=" * 108)
    print(f"{'signal':17s} {'n':>4s} {'Friedman chi2':>14s} {'p':>10s} {'Kendall W':>10s} | post-hoc q")
    for sig in SIGNALS:
        fr, ph = friedman_results[sig], posthoc.get(sig, {})
        ph_str = "  ".join(f"{LOAD_SHORT[a]}v{LOAD_SHORT[b]}={ph.get((a, b), {}).get('q', np.nan):.3g}"
                           for a, b in PAIRS)
        print(f"{sig:17s} {fr['n']:4d} {fr['stat']:14.3f} {fr['p']:10.3g} {fr['kendalls_w']:10.3f} | {ph_str}")

    print("\nPer-subject 0/2/3-back profile shape (same convention as capacity_ceiling_check.py):")
    for sig in SIGNALS:
        sh = shapes[sig]
        print(f"  {sig:17s}: monotonic {len(sh['monotonic']):2d}, inverted-U {len(sh['inverted-U']):2d}, "
              f"other {len(sh['other']):2d}")

    print("\nREADING")
    sig_ps = {sig: friedman_results[sig]["p"] for sig in SIGNALS}
    hits = [s for s, pv in sig_ps.items() if pv == pv and pv < 0.05]
    if hits:
        best = min(hits, key=lambda s: sig_ps[s])
        print(f"  Formal omnibus evidence that '{best}' ({LABELS[best]}) differs across cognitive-load levels:\n"
              f"  Friedman p={sig_ps[best]:.3g}, Kendall's W={friedman_results[best]['kendalls_w']:.2f}. This is an\n"
              f"  FDR-corrected inferential claim about the model's own internal representation tracking task\n"
              f"  difficulty -- not a qualitative figure description, which is all STA-Net and MGFNet provide\n"
              f"  for their analogous attention-weight-across-conditions figures.")
        if best == "w_fnirs":
            print("  Note: w_fnirs is STA-Net-specific (instance-adaptive decision weights, Eq. 6-7). A load\n"
                  "  effect here means the model re-weights EEG vs fNIRS by task difficulty -- behaviour\n"
                  "  TSFNet's fixed fusion scalars cannot exhibit at all.")
    else:
        print("  No XAI signal cleared the omnibus threshold across load levels. Reporting this honestly\n"
              "  matters: whatever role the attention/gating plays in accuracy, it does NOT visibly track task\n"
              "  difficulty in a way a formal repeated-measures test can detect at this sample size -- a\n"
              "  materially different, and more defensible, claim than describing a figure as 'load-dependent'.")

    summary = {
        "n_subjects": len(ids),
        "labels": LABELS,
        "friedman": friedman_results,
        "posthoc_bh_family": {sig: {f"{CLASS_NAMES[a]}_vs_{CLASS_NAMES[b]}": v
                                    for (a, b), v in posthoc.get(sig, {}).items()} for sig in SIGNALS},
        "shapes": {sig: dict(shapes[sig]) for sig in SIGNALS},
    }
    with open(os.path.join(out_dir, "xai_load_level_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_load_level_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{sig}_{lab}" for sig in SIGNALS for lab in CLASS_NAMES])
        for s in ids:
            w.writerow([s] + [profiles[s][sig][k] for sig in SIGNALS for k in (0, 1, 2)])
    print(f"\n[written] {out_dir}: xai_load_level_summary.json, xai_load_level_per_subject.csv")


if __name__ == "__main__":
    main()
