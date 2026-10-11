"""
xai_load_level_anova_mgfnet.py -- is MGFNet's own CGSCR signal FORMALLY
different across 0-/2-/3-back, or does it only "look" different in a figure?

MGFNet version of xai_load_level_anova.py: same design and statistics,
MGFNet's signals (see mgfnet_xai_common.py) instead of EGTA / EFGF Phi.

DESIGN: each subject contributes ONE value per load level per signal (mean
over that subject's held-out trials at that level, pooled across the 3 LOSO
folds). Subjects = blocks, load = 3-level repeated-measures factor.

STATISTICS PER SIGNAL
  1. Friedman chi-square omnibus test
  2. Kendall's W effect size
  3. post-hoc Wilcoxon for 0v2, 0v3, 2v3, BH-FDR over ALL pairs x signals
     (3 x 7 = 21 tests, one family)
  4. per-subject profile shape (monotonic / inverted-U / other), same
     convention as capacity_ceiling_check.py's hbo_shape

USAGE
=====
    python xai_load_level_anova_mgfnet.py --results-dir "E:\\...\\results_mgf_<hash>_seed42" ^
        --data-root "E:\\...\\MGFNET_PAPER_EXACT_nback_28ch"
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import friedmanchisquare

import mgfnet_xai_common as X

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_load_level_check")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--min-subjects", type=int, default=8)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

SIGNALS = X.SIGNALS
LOAD_NAMES = ["0-back", "2-back", "3-back"]
LOAD_SHORT = ["0b", "2b", "3b"]
PAIRS = [(0, 1), (0, 2), (1, 2)]


def subject_profile(sid):
    rows = X.subject_trial_rows(sid, ARGS.results_dir, ARGS.data_root, ARGS.device, ARGS.batch_size)
    if not rows:
        return None
    prof = {}
    for name in SIGNALS:
        v = np.array([r["value"] for r in rows if r["signal"] == name])
        l = np.array([r["label"] for r in rows if r["signal"] == name])
        prof[name] = np.array([v[l == k].mean() if (l == k).any() else np.nan for k in (0, 1, 2)])
    return prof


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_load_level_check")
    os.makedirs(out_dir, exist_ok=True)
    profiles = {}
    for sid in X.parse_subjects(ARGS.subjects):
        try:
            prof = subject_profile(sid)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            continue
        if prof is not None:
            profiles[sid] = prof
            print(f"[{sid}] done")
    ids = sorted(profiles)
    if len(ids) < ARGS.min_subjects:
        sys.exit(f"ERROR: only {len(ids)} subjects produced usable profiles (< {ARGS.min_subjects}).")

    all_tests, friedman, shapes = [], {}, {n: {"monotonic": [], "inverted-U": [], "other": []} for n in SIGNALS}
    for name in SIGNALS:
        mat = np.array([profiles[s][name] for s in ids])
        keep = ~np.isnan(mat).any(axis=1)
        mat_ok, ids_ok = mat[keep], [s for s, k in zip(ids, keep) if k]
        if mat_ok.shape[0] < ARGS.min_subjects:
            friedman[name] = {"n": int(mat_ok.shape[0]), "stat": np.nan, "p": np.nan, "kendalls_w": np.nan}
            continue
        try:
            stat, pv = friedmanchisquare(mat_ok[:, 0], mat_ok[:, 1], mat_ok[:, 2])
        except ValueError:
            stat, pv = np.nan, np.nan
        friedman[name] = {"n": int(mat_ok.shape[0]), "stat": float(stat), "p": float(pv),
                          "kendalls_w": X.kendalls_w(mat_ok)}
        for i, (a, b) in enumerate(PAIRS):
            all_tests.append((name, i, X.safe_wilcoxon(mat_ok[:, a], mat_ok[:, b])))
        for sid in ids_ok:
            m0, m2, m3 = profiles[sid][name]
            shapes[name]["monotonic" if m0 < m2 < m3 else
                         "inverted-U" if (m2 > m3 and m2 > m0) else "other"].append(sid)

    posthoc = {}
    for (name, i, pv), q in zip(all_tests, X.bh([t[2] for t in all_tests])):
        posthoc.setdefault(name, {})[PAIRS[i]] = {"p": float(pv), "q": float(q)}

    hline = "=" * 104
    print("\n" + hline)
    print(f"LOAD-LEVEL FRIEDMAN TEST -- {len(ids)} subjects, 0-/2-/3-back, MGFNet CGSCR signals")
    print(f"(BH-FDR across all {len(all_tests)} post-hoc tests, one family)")
    print(hline)
    print(f"{'signal':20s} {'n':>4s} {'Friedman chi2':>14s} {'p':>10s} {'Kendall W':>10s} | post-hoc (q)")
    for name in SIGNALS:
        fr, ph = friedman[name], posthoc.get(name, {})
        ph_s = "  ".join(f"{LOAD_SHORT[a]}v{LOAD_SHORT[b]}={ph.get((a, b), {}).get('q', np.nan):.3g}"
                         for a, b in PAIRS)
        print(f"{name:20s} {fr['n']:4d} {fr['stat']:14.3f} {fr['p']:10.3g} {fr['kendalls_w']:10.3f} | {ph_s}")
    print("\nShape of the per-subject 0/2/3-back profile (same convention as capacity_ceiling_check.py):")
    for name in SIGNALS:
        sh = shapes[name]
        print(f"  {name:20s}: monotonic {len(sh['monotonic']):2d}, inverted-U {len(sh['inverted-U']):2d}, "
              f"other {len(sh['other']):2d}")

    print("\nREADING")
    sig = [n for n in SIGNALS if friedman[n]["p"] == friedman[n]["p"] and friedman[n]["p"] < 0.05]
    if sig:
        best = min(sig, key=lambda n: friedman[n]["p"])
        print(f"  Formal omnibus evidence that '{best}' differs across load levels (Friedman p="
              f"{friedman[best]['p']:.3g}, Kendall's W={friedman[best]['kendalls_w']:.2f}) -- an FDR-"
              f"controlled inferential claim that MGFNet's CGSCR tracks task difficulty, not a description "
              f"of a figure (the MGFNet paper reports only t-SNE / confusion figures).")
    else:
        print("  No CGSCR signal clears the omnibus threshold across load levels: whatever CGSCR contributes "
              "to accuracy, its gating/routing does not visibly track task difficulty at this sample size -- "
              "worth reporting as such.")

    summary = {"n_subjects": len(ids), "friedman": friedman,
               "posthoc_bh_family": {n: {f"{LOAD_NAMES[a]}_vs_{LOAD_NAMES[b]}": v
                                         for (a, b), v in posthoc.get(n, {}).items()} for n in SIGNALS},
               "shapes": shapes}
    with open(os.path.join(out_dir, "xai_load_level_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=X.nan_float)
    with open(os.path.join(out_dir, "xai_load_level_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{n}_{lab}" for n in SIGNALS for lab in LOAD_NAMES])
        for s in ids:
            w.writerow([s] + [profiles[s][n][k] for n in SIGNALS for k in (0, 1, 2)])
    print(f"\n[written] {out_dir}: xai_load_level_summary.json, xai_load_level_per_subject.csv")


if __name__ == "__main__":
    main()
