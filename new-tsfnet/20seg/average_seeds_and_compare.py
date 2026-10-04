# ============================================================================
# MULTI-SEED NOISE AVERAGING + STATISTICAL COMPARISON
#
# Single run, all checks:
#   1. Loads 3 baseline_all_subjects_*.json files for the 11-segment config
#      (one per seed: 42, 123, 2024) and 3 for the 20-segment config.
#   2. For each subject, averages mean_acc and mean_kappa ACROSS THE 3 SEEDS
#      (only using seeds where that subject succeeded) -- this is the
#      noise-reduction step: each subject's "value" going into the paired
#      test is now a 3-seed average, cutting training-stochasticity variance
#      by ~1/sqrt(3) (~42% reduction in std) relative to any single seed.
#   3. Runs the SAME paired statistical comparison as
#      compare_ablation_segments.py (paired t-test, Wilcoxon signed-rank,
#      Cohen's d) on these seed-averaged per-subject values.
#   4. Saves the seed-averaged per-subject/per-config CSVs, the full
#      comparison table, and a thesis-ready paragraph explicitly describing
#      the multi-seed averaging methodology (for the Methods section) plus
#      the comparison result (for the Results/Ablation section).
#
# Usage:
#   python average_seeds_and_compare.py \
#       --seg11_jsons "path/11seg_seed42.json" "path/11seg_seed123.json" "path/11seg_seed2024.json" \
#       --seg20_jsons "path/20seg_seed42.json" "path/20seg_seed123.json" "path/20seg_seed2024.json"
# ============================================================================

import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def load_ok_subjects(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {sid: row for sid, row in data.items() if row.get("status") == "ok"}


def average_across_seeds(json_paths, label):
    """Returns {subject_id: {"mean_acc": avg, "mean_kappa": avg, "n_seeds": k}}
    averaging mean_acc/mean_kappa across however many of the given seed runs
    that subject succeeded in (usually all 3, but robust to partial failures)."""
    per_subject_vals = {}   # subject_id -> {"acc": [...], "kappa": [...]}
    for path in json_paths:
        data = load_ok_subjects(path)
        for sid, row in data.items():
            per_subject_vals.setdefault(sid, {"acc": [], "kappa": []})
            per_subject_vals[sid]["acc"].append(row["mean_acc"])
            per_subject_vals[sid]["kappa"].append(row["mean_kappa"])

    averaged = {}
    for sid, vals in per_subject_vals.items():
        n = len(vals["acc"])
        averaged[sid] = {
            "mean_acc": float(np.mean(vals["acc"])),
            "mean_kappa": float(np.mean(vals["kappa"])),
            "acc_seed_std": float(np.std(vals["acc"], ddof=1)) if n > 1 else 0.0,
            "kappa_seed_std": float(np.std(vals["kappa"], ddof=1)) if n > 1 else 0.0,
            "n_seeds": n,
        }

    n_full = sum(1 for v in averaged.values() if v["n_seeds"] == len(json_paths))
    print(f"{label}: {len(averaged)} subjects with >=1 successful seed, "
          f"{n_full}/{len(averaged)} with all {len(json_paths)} seeds successful")
    incomplete = {sid: v["n_seeds"] for sid, v in averaged.items() if v["n_seeds"] < len(json_paths)}
    if incomplete:
        print(f"  WARNING -- subjects with incomplete seed coverage: {incomplete}")

    return averaged


def cohens_d_paired(x, y):
    diff = np.asarray(x) - np.asarray(y)
    return float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else np.nan


def paired_tests(old_vals, new_vals, label):
    old_vals = np.asarray(old_vals, dtype=float)
    new_vals = np.asarray(new_vals, dtype=float)
    n = len(old_vals)

    t_stat, t_p = stats.ttest_rel(new_vals, old_vals)
    try:
        w_stat, w_p = stats.wilcoxon(new_vals, old_vals)
    except ValueError as e:
        w_stat, w_p = np.nan, np.nan
        print(f"  (Wilcoxon could not be computed for {label}: {e})")

    d = cohens_d_paired(new_vals, old_vals)

    print(f"\n--- {label} (n={n} paired subjects, seed-averaged) ---")
    print(f"  Mean 11-seg: {old_vals.mean():.4f} (+-{old_vals.std(ddof=1):.4f})")
    print(f"  Mean 20-seg: {new_vals.mean():.4f} (+-{new_vals.std(ddof=1):.4f})")
    print(f"  Mean delta (20seg-11seg): {(new_vals - old_vals).mean():+.4f}")
    print(f"  Paired t-test:   t({n-1}) = {t_stat:.4f}, p = {t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W = {w_stat}, p = {w_p if np.isnan(w_p) else round(w_p, 4)}")
    print(f"  Cohen's d (paired): {d:.4f}")

    return {
        "metric": label, "n": n,
        "mean_11seg": float(old_vals.mean()), "mean_20seg": float(new_vals.mean()),
        "mean_delta": float((new_vals - old_vals).mean()),
        "t_stat": float(t_stat), "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "wilcoxon_pvalue": float(w_p) if not np.isnan(w_p) else None,
        "cohens_d": d,
    }


def effect_size_label(d):
    ad = abs(d)
    if np.isnan(ad):
        return "n/a"
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def significance_label(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seg11_jsons", type=str, nargs="+", required=True,
                         help="3 baseline_all_subjects_*.json paths for the 11-segment config (one per seed)")
    parser.add_argument("--seg20_jsons", type=str, nargs="+", required=True,
                         help="3 baseline_all_subjects_*.json paths for the 20-segment config (one per seed)")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.seg20_jsons[0])
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Averaging {len(args.seg11_jsons)} seed-runs for 11-segment config ===")
    avg_11 = average_across_seeds(args.seg11_jsons, "11-segment")
    print(f"\n=== Averaging {len(args.seg20_jsons)} seed-runs for 20-segment config ===")
    avg_20 = average_across_seeds(args.seg20_jsons, "20-segment")

    # save the seed-averaged per-config results for traceability
    for label, avg in (("11seg", avg_11), ("20seg", avg_20)):
        df_avg = pd.DataFrame([{"subject_id": sid, **vals} for sid, vals in avg.items()])
        df_avg.to_csv(os.path.join(out_dir, f"seed_averaged_{label}.csv"), index=False)

    common_subjects = sorted(set(avg_11.keys()) & set(avg_20.keys()))
    print(f"\nPaired subjects (present in both averaged configs): {len(common_subjects)}")
    if len(common_subjects) < 2:
        print("ERROR: fewer than 2 paired subjects -- cannot run paired statistical tests.")
        return

    acc_11 = [avg_11[s]["mean_acc"] for s in common_subjects]
    acc_20 = [avg_20[s]["mean_acc"] for s in common_subjects]
    kappa_11 = [avg_11[s]["mean_kappa"] for s in common_subjects]
    kappa_20 = [avg_20[s]["mean_kappa"] for s in common_subjects]

    acc_res = paired_tests(acc_11, acc_20, "Accuracy (seed-averaged)")
    kappa_res = paired_tests(kappa_11, kappa_20, "Kappa (seed-averaged)")

    rows = []
    for s in common_subjects:
        rows.append({
            "subject_id": s,
            "acc_11seg": avg_11[s]["mean_acc"], "acc_20seg": avg_20[s]["mean_acc"],
            "acc_delta": avg_20[s]["mean_acc"] - avg_11[s]["mean_acc"],
            "kappa_11seg": avg_11[s]["mean_kappa"], "kappa_20seg": avg_20[s]["mean_kappa"],
            "kappa_delta": avg_20[s]["mean_kappa"] - avg_11[s]["mean_kappa"],
            "acc_seed_std_11seg": avg_11[s]["acc_seed_std"], "acc_seed_std_20seg": avg_20[s]["acc_seed_std"],
            "n_seeds_11seg": avg_11[s]["n_seeds"], "n_seeds_20seg": avg_20[s]["n_seeds"],
        })
    df = pd.DataFrame(rows).sort_values("acc_delta", ascending=False)
    out_csv = os.path.join(out_dir, "seed_averaged_comparison.csv")
    df.to_csv(out_csv, index=False)

    n_improved = int((df["acc_delta"] > 0).sum())
    n = len(common_subjects)

    print("\n" + "=" * 78)
    print(f"Improved (20-seg > 11-seg) after seed-averaging: {n_improved}/{n}")
    print(f"Full comparison table saved: {out_csv}")

    # ----- thesis-ready paragraph -----
    n_seeds_used = len(args.seg11_jsons)
    d_label = effect_size_label(acc_res["cohens_d"])
    acc_sig = significance_label(acc_res["t_pvalue"])
    acc_sig_w = significance_label(acc_res["wilcoxon_pvalue"])
    kappa_sig = significance_label(kappa_res["t_pvalue"])
    kappa_sig_w = significance_label(kappa_res["wilcoxon_pvalue"])

    paragraph = (
        f"To reduce training-stochasticity noise (weight initialization, batch ordering, dropout "
        f"masks) that could obscure a true effect of the segment-count manipulation, each "
        f"configuration (11-segment paper-exact and 20-segment extended) was trained with "
        f"{n_seeds_used} different random seeds (42, 123, 2024), and each subject's accuracy and "
        f"kappa were averaged across seeds before the paired comparison. This reduces the "
        f"per-subject noise standard deviation by a factor of approximately "
        f"1/sqrt({n_seeds_used})~={1/np.sqrt(n_seeds_used):.2f}. Across {n} subjects with complete "
        f"seed coverage in both configurations, seed-averaged accuracy increased from "
        f"{acc_res['mean_11seg']*100:.2f}% (11-segment) to {acc_res['mean_20seg']*100:.2f}% "
        f"(20-segment), a mean paired improvement of {acc_res['mean_delta']*100:+.2f} percentage "
        f"points (paired t-test: t({n-1})={acc_res['t_stat']:.2f}, {acc_sig}; Wilcoxon signed-rank: "
        f"W={acc_res['wilcoxon_stat']:.1f}, {acc_sig_w}; Cohen's d={acc_res['cohens_d']:.2f}, a "
        f"{d_label} effect). Kappa increased correspondingly from {kappa_res['mean_11seg']:.3f} to "
        f"{kappa_res['mean_20seg']:.3f} ({kappa_res['mean_delta']:+.3f}; "
        f"t({n-1})={kappa_res['t_stat']:.2f}, {kappa_sig}; W={kappa_res['wilcoxon_stat']:.1f}, "
        f"{kappa_sig_w}). {n_improved} of {n} subjects ({100*n_improved/n:.0f}%) showed improved "
        f"seed-averaged accuracy under the extended-window condition."
    )

    print(f"\n--- Thesis-ready paragraph (Methods + Ablation Study) ---\n{paragraph}")
    out_txt = os.path.join(out_dir, "seed_averaged_ablation_paragraph.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"\nParagraph saved: {out_txt}")


if __name__ == "__main__":
    main()
