"""
xai_stability_check_mgfnet.py -- are MGFNet's CGSCR explanations REPRODUCIBLE
across random seeds, or does every seed tell a different story about the same
input?

MGFNet version of xai_stability_check.py. Same design and statistics; the
explanation vectors are MGFNet's (TSFNet's 21-segment EGTA attention does not
exist here):
  gate      6-way CGSCR branch gate              (items = 6 branches)
  coupling  coupling matrix C, flattened         (items = 15 band x scale cells)
  routing   router distribution softmax(scores)  (items = 15 tokens)

For every subject and held-out session, each seed's checkpoint is run on the
EXACT same held-out windows (unshuffled loader), so explanations are compared
window for window:
  1. per-window Spearman between every seed pair, averaged per subject
  2. ICC(2,1) (Shrout & Fleiss) on the subject-level profile (mean vector over
     all held-out windows), items x seeds; Koo & Li (2016) categories:
     >= 0.75 excellent, 0.5-0.75 moderate, < 0.5 poor.

USAGE
=====
    python xai_stability_check_mgfnet.py --data-root "E:\\...\\MGFNET_PAPER_EXACT_nback_28ch" ^
        --variant-hash <hash> --seeds 42 999 2024
(results_mgf_<hash>_seed<N> folders, as written by run_all_subjects_MGFNET.py;
or pass --results-dirs explicitly, one per seed, in --seeds order.)
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

import mgfnet_xai_common as X

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--data-root", required=True)
p.add_argument("--variant-hash", default=None, help="hash in results_mgf_<hash>_seed<N>")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 999, 2024])
p.add_argument("--results-dirs", nargs="*", default=None)
p.add_argument("--results-root", default=None, help="where results_mgf_* live (default: --data-root)")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-root>/xai_stability_check")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--min-subjects", type=int, default=6)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

VECTORS = ("gate", "coupling", "routing")


def resolve_results_dirs():
    if ARGS.results_dirs:
        if len(ARGS.results_dirs) != len(ARGS.seeds):
            sys.exit(f"ERROR: {len(ARGS.results_dirs)} --results-dirs for {len(ARGS.seeds)} --seeds.")
        return list(ARGS.results_dirs)
    if not ARGS.variant_hash:
        sys.exit("ERROR: pass --results-dirs (one per seed) or --variant-hash.")
    root = ARGS.results_root or ARGS.data_root
    dirs = [os.path.join(root, f"results_mgf_{ARGS.variant_hash}_seed{s}") for s in ARGS.seeds]
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        sys.exit(f"ERROR: not found: {missing}. Pass --results-dirs or --results-root.")
    return dirs


def main():
    if len(ARGS.seeds) < 2:
        sys.exit("ERROR: need at least 2 seeds.")
    rdirs = resolve_results_dirs()
    for s, d in zip(ARGS.seeds, rdirs):
        print(f"  seed {s:>5}: {d}")
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_root or ARGS.data_root, "xai_stability_check")
    os.makedirs(out_dir, exist_ok=True)
    pairs = list(itertools.combinations(range(len(ARGS.seeds)), 2))

    per_subject = {}
    for sid in X.parse_subjects(ARGS.subjects):
        try:
            full = X.load_subject(ARGS.data_root, sid)
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
            continue
        win_rhos = {v: [] for v in VECTORS}
        seed_mats = {v: [[] for _ in ARGS.seeds] for v in VECTORS}
        ok = True
        try:
            for test_session in (1, 2, 3):
                loader, _ = X.test_loader(full, test_session, ARGS.batch_size)
                per_seed = []
                for seed, rdir in zip(ARGS.seeds, rdirs):
                    fold = X.load_fold(rdir, sid, test_session, ARGS.device)
                    if fold is None:
                        ok = False
                        break
                    model, norm = fold
                    _, vec = X.collect_fold(model, norm, loader, ARGS.device)
                    per_seed.append(vec)
                    del model
                if not ok:
                    break
                n_win = per_seed[0]["gate"].shape[0]
                for v in VECTORS:
                    for i, vec in enumerate(per_seed):
                        seed_mats[v][i].append(vec[v])
                    for w in range(n_win):
                        for a, b in pairs:
                            x, y = per_seed[a][v][w], per_seed[b][v][w]
                            if np.allclose(x, x[0]) or np.allclose(y, y[0]):
                                continue
                            rho, _ = spearmanr(x, y)
                            if not np.isnan(rho):
                                win_rhos[v].append(rho)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            continue
        if not ok:
            print(f"[{sid}] SKIP -- incomplete checkpoint set across seeds/sessions")
            continue

        rec = {}
        for v in VECTORS:
            prof = np.stack([np.concatenate(seed_mats[v][i]).mean(0) for i in range(len(ARGS.seeds))], 1)
            rec[v] = {"mean_pairwise_rho": float(np.mean(win_rhos[v])) if win_rhos[v] else np.nan,
                      "sd_pairwise_rho": float(np.std(win_rhos[v], ddof=1)) if len(win_rhos[v]) > 1 else np.nan,
                      "icc_2_1": X.icc_2_1(prof)}
        per_subject[sid] = rec
        print(f"[{sid}] " + " | ".join(f"{v}: rho={rec[v]['mean_pairwise_rho']:+.3f} ICC={rec[v]['icc_2_1']:+.3f}"
                                       for v in VECTORS))

    ids = sorted(per_subject)
    if len(ids) < ARGS.min_subjects:
        sys.exit(f"ERROR: only {len(ids)} subjects produced cross-seed data (< {ARGS.min_subjects}).")

    hline = "=" * 104
    print("\n" + hline)
    print(f"CROSS-SEED EXPLANATION-STABILITY SUMMARY -- {len(ids)} subjects, seeds={ARGS.seeds}")
    print(hline)
    group = {}
    for v in VECTORS:
        rh = np.array([per_subject[s][v]["mean_pairwise_rho"] for s in ids]); rh = rh[~np.isnan(rh)]
        ic = np.array([per_subject[s][v]["icc_2_1"] for s in ids]); ic = ic[~np.isnan(ic)]
        exc, mod, poor = int((ic >= 0.75).sum()), int(((ic >= 0.5) & (ic < 0.75)).sum()), int((ic < 0.5).sum())
        group[v] = {"mean_pairwise_rho": float(rh.mean()) if len(rh) else np.nan,
                    "sd_pairwise_rho": float(rh.std(ddof=1)) if len(rh) > 1 else np.nan,
                    "mean_icc": float(ic.mean()) if len(ic) else np.nan,
                    "sd_icc": float(ic.std(ddof=1)) if len(ic) > 1 else np.nan,
                    "n_excellent": exc, "n_moderate": mod, "n_poor": poor}
        print(f"{v:9s} per-window rho: mean={group[v]['mean_pairwise_rho']:+.3f} "
              f"(SD {group[v]['sd_pairwise_rho']:.3f}) | ICC(2,1): mean={group[v]['mean_icc']:+.3f} | "
              f"excellent {exc}/{len(ic)}, moderate {mod}/{len(ic)}, poor {poor}/{len(ic)}")

    print("\nREADING")
    for v in VECTORS:
        m = group[v]["mean_icc"]
        verdict = ("HIGHLY reproducible: a stable property of each subject's data, not of one initialisation"
                   if m >= 0.75 else
                   "MODERATELY reproducible: single-run figures are indicative, not precise" if m >= 0.5 else
                   "NOT reliably reproducible: a different seed gives a materially different explanation")
        print(f"  {v:9s}: {verdict}.")
    print("  (The MGFNet paper shows explanations from a single run only; this is the reliability check it lacks.)")

    with open(os.path.join(out_dir, "xai_stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"seeds": ARGS.seeds, "results_dirs": rdirs, "per_subject": per_subject, "group": group},
                  f, indent=2, default=X.nan_float)
    with open(os.path.join(out_dir, "xai_stability_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [f"{v}_{k}" for v in VECTORS for k in ("mean_pairwise_rho", "sd_pairwise_rho", "icc_2_1")]
        w.writerow(["subject"] + cols)
        for s in ids:
            w.writerow([s] + [per_subject[s][v][k] for v in VECTORS
                              for k in ("mean_pairwise_rho", "sd_pairwise_rho", "icc_2_1")])
    print(f"\n[written] {out_dir}: xai_stability_summary.json, xai_stability_per_subject.csv")


if __name__ == "__main__":
    main()
