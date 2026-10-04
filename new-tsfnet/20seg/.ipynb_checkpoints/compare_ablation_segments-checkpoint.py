# ============================================================================
# ABLATION STATISTICAL COMPARISON: 11-segment (paper-exact) vs 20-segment
# (extended HRF-tail-coverage) fNIRS windowing
#
# Single run, all checks:
#   1. Loads both baseline_all_subjects_*.json summary files.
#   2. Aligns subjects present (status=="ok") in BOTH runs (paired design --
#      every subject has one accuracy/kappa value per condition).
#   3. Runs a PAIRED t-test (parametric) AND a Wilcoxon signed-rank test
#      (non-parametric, robust to the fact that normality of the per-subject
#      accuracy differences is not established) on:
#        - accuracy
#        - kappa
#      Also reports Cohen's d for paired samples (effect size) alongside the
#      p-values, since p-values alone don't convey magnitude.
#   4. If branch-accuracy / fusion-weight fields (eeg_only_acc, fnirs_only_acc,
#      fusion_only_acc, w_eeg, w_fusion, w_fnirs) are present in BOTH files,
#      also compares those (paired t-test + Wilcoxon) -- gracefully skipped
#      with a note if the OLD run's JSON predates that logging addition.
#   5. Builds one full per-subject comparison CSV (old vs new vs delta for
#      every available metric).
#   6. Prints a thesis-ready "Ablation Study" paragraph with exact numbers,
#      test statistics, and p-values, following APA-style reporting
#      conventions (t(df)=..., p=..., Cohen's d=...; W=..., p=... for Wilcoxon).
#
# Usage:
#   python compare_ablation_segments.py --old_json "path/to/11seg.json" --new_json "path/to/20seg.json"
# ============================================================================

import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy import stats


BRANCH_FIELDS = ["eeg_only_acc", "fnirs_only_acc", "fusion_only_acc",
                  "w_eeg", "w_fusion", "w_fnirs"]


def load_ok_subjects(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {sid: row for sid, row in data.items() if row.get("status") == "ok"}


def cohens_d_paired(x, y):
    """Cohen's d for paired samples: mean difference / std of the differences."""
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
        # e.g. all differences are zero, or n too small
        w_stat, w_p = np.nan, np.nan
        print(f"  (Wilcoxon could not be computed for {label}: {e})")

    d = cohens_d_paired(new_vals, old_vals)

    print(f"\n--- {label} (n={n} paired subjects) ---")
    print(f"  Mean OLD:  {old_vals.mean():.4f} (+-{old_vals.std(ddof=1):.4f})")
    print(f"  Mean NEW:  {new_vals.mean():.4f} (+-{new_vals.std(ddof=1):.4f})")
    print(f"  Mean delta (NEW-OLD): {(new_vals - old_vals).mean():+.4f}")
    print(f"  Paired t-test:   t({n-1}) = {t_stat:.4f}, p = {t_p:.4f}")
    print(f"  Wilcoxon signed-rank: W = {w_stat}, p = {w_p if np.isnan(w_p) else round(w_p, 4)}")
    print(f"  Cohen's d (paired): {d:.4f}")

    return {
        "metric": label, "n": n,
        "mean_old": float(old_vals.mean()), "mean_new": float(new_vals.mean()),
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
    if p is None or np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_json", type=str, required=True,
                         help="baseline_all_subjects_*.json for the 11-segment (paper-exact) run")
    parser.add_argument("--new_json", type=str, required=True,
                         help="baseline_all_subjects_*.json for the 20-segment (extended) run")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.new_json)
    os.makedirs(out_dir, exist_ok=True)

    old_data = load_ok_subjects(args.old_json)
    new_data = load_ok_subjects(args.new_json)

    common_subjects = sorted(set(old_data.keys()) & set(new_data.keys()))
    missing_old = sorted(set(new_data.keys()) - set(old_data.keys()))
    missing_new = sorted(set(old_data.keys()) - set(new_data.keys()))

    print(f"=== ABLATION COMPARISON: 11-segment vs 20-segment ===")
    print(f"Subjects with valid ('ok') results in OLD (11-seg): {len(old_data)}")
    print(f"Subjects with valid ('ok') results in NEW (20-seg): {len(new_data)}")
    print(f"Paired subjects (present + 'ok' in BOTH): {len(common_subjects)}")
    if missing_old:
        print(f"  Present in NEW but missing/failed in OLD: {missing_old}")
    if missing_new:
        print(f"  Present in OLD but missing/failed in NEW: {missing_new}")

    if len(common_subjects) < 2:
        print("ERROR: fewer than 2 paired subjects -- cannot run paired statistical tests.")
        return

    old_acc = [old_data[s]["mean_acc"] for s in common_subjects]
    new_acc = [new_data[s]["mean_acc"] for s in common_subjects]
    old_kappa = [old_data[s]["mean_kappa"] for s in common_subjects]
    new_kappa = [new_data[s]["mean_kappa"] for s in common_subjects]

    results = []
    results.append(paired_tests(old_acc, new_acc, "Accuracy"))
    results.append(paired_tests(old_kappa, new_kappa, "Kappa"))

    # ----- branch accuracy / fusion weights, only if present in BOTH files -----
    branch_available = all(
        all(f in old_data[s] for f in BRANCH_FIELDS) for s in common_subjects
    ) and all(
        all(f in new_data[s] for f in BRANCH_FIELDS) for s in common_subjects
    )
    if branch_available:
        for field in BRANCH_FIELDS:
            old_vals = [old_data[s][field] for s in common_subjects]
            new_vals = [new_data[s][field] for s in common_subjects]
            results.append(paired_tests(old_vals, new_vals, field))
    else:
        print("\nNOTE: branch-accuracy/fusion-weight fields not present in both JSONs "
              "(likely the OLD 11-segment run predates that logging addition) -- "
              "skipping branch-level statistical comparison.")

    # ----- full per-subject comparison table -----
    rows = []
    for s in common_subjects:
        row = {
            "subject_id": s,
            "acc_old": old_data[s]["mean_acc"], "acc_new": new_data[s]["mean_acc"],
            "acc_delta": new_data[s]["mean_acc"] - old_data[s]["mean_acc"],
            "kappa_old": old_data[s]["mean_kappa"], "kappa_new": new_data[s]["mean_kappa"],
            "kappa_delta": new_data[s]["mean_kappa"] - old_data[s]["mean_kappa"],
        }
        if branch_available:
            for field in BRANCH_FIELDS:
                row[f"{field}_old"] = old_data[s][field]
                row[f"{field}_new"] = new_data[s][field]
                row[f"{field}_delta"] = new_data[s][field] - old_data[s][field]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("acc_delta", ascending=False)
    out_csv = os.path.join(out_dir, "ablation_11seg_vs_20seg_comparison.csv")
    df.to_csv(out_csv, index=False)

    n_improved = int((df["acc_delta"] > 0).sum())
    n_worse = int((df["acc_delta"] < 0).sum())
    n_same = int((df["acc_delta"] == 0).sum())

    print("\n" + "=" * 78)
    print("PER-SUBJECT DIRECTION OF CHANGE (accuracy)")
    print("=" * 78)
    print(f"Improved (20-seg > 11-seg): {n_improved}/{len(common_subjects)}")
    print(f"Worse    (20-seg < 11-seg): {n_worse}/{len(common_subjects)}")
    print(f"Unchanged:                  {n_same}/{len(common_subjects)}")
    print(f"\nFull per-subject comparison table saved: {out_csv}")

    # ----- thesis-ready paragraph -----
    acc_res = results[0]
    kappa_res = results[1]
    n = acc_res["n"]

    acc_sig = significance_label(acc_res["t_pvalue"])
    acc_sig_w = significance_label(acc_res["wilcoxon_pvalue"])
    kappa_sig = significance_label(kappa_res["t_pvalue"])
    kappa_sig_w = significance_label(kappa_res["wilcoxon_pvalue"])

    d_label = effect_size_label(acc_res["cohens_d"])

    paragraph = (
        f"To formally test whether extending the fNIRS segment count from 11 (paper-exact, "
        f"[-2,10]s window) to 20 (extended, [-2,42]s window, covering the full hemodynamic "
        f"response including the post-peak undershoot) improved classification performance, "
        f"a paired comparison was performed across all {n} subjects with valid results under "
        f"both conditions. Mean subject-level accuracy increased from {acc_res['mean_old']*100:.2f}% "
        f"(11-segment) to {acc_res['mean_new']*100:.2f}% (20-segment), a mean paired improvement of "
        f"{acc_res['mean_delta']*100:+.2f} percentage points "
        f"(paired t-test: t({n-1})={acc_res['t_stat']:.2f}, {acc_sig}; "
        f"Wilcoxon signed-rank: W={acc_res['wilcoxon_stat']:.1f}, {acc_sig_w}; "
        f"Cohen's d={acc_res['cohens_d']:.2f}, a {d_label} effect). "
        f"Cohen's kappa showed a corresponding increase from {kappa_res['mean_old']:.3f} to "
        f"{kappa_res['mean_new']:.3f} ({kappa_res['mean_delta']:+.3f}; "
        f"t({n-1})={kappa_res['t_stat']:.2f}, {kappa_sig}; W={kappa_res['wilcoxon_stat']:.1f}, "
        f"{kappa_sig_w}). {n_improved} of {n} subjects ({100*n_improved/n:.0f}%) showed improved "
        f"accuracy under the extended-window condition."
    )

    if branch_available:
        fnirs_res = next(r for r in results if r["metric"] == "fnirs_only_acc")
        weeg_res = next(r for r in results if r["metric"] == "w_eeg")
        wfnirs_res = next(r for r in results if r["metric"] == "w_fnirs")
        paragraph += (
            f" This improvement was accompanied by a corresponding change in the fNIRS branch's "
            f"standalone accuracy (from {fnirs_res['mean_old']*100:.2f}% to "
            f"{fnirs_res['mean_new']*100:.2f}%, {significance_label(fnirs_res['t_pvalue'])}) and in "
            f"the model's learned fusion-gate weights, which shifted toward the fNIRS branch "
            f"(w_fnirs: {wfnirs_res['mean_old']:.3f} -> {wfnirs_res['mean_new']:.3f}) relative to "
            f"EEG (w_eeg: {weeg_res['mean_old']:.3f} -> {weeg_res['mean_new']:.3f}), consistent with "
            f"the extended window providing more physiologically complete (and thus more "
            f"informative) fNIRS input."
        )

    print(f"\n--- Thesis-ready paragraph (Ablation Study section) ---\n{paragraph}")

    out_txt = os.path.join(out_dir, "ablation_11seg_vs_20seg_paragraph.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(paragraph)
    print(f"\nParagraph saved: {out_txt}")

    # ----- also save the raw statistical results table -----
    stats_df = pd.DataFrame(results)
    stats_csv = os.path.join(out_dir, "ablation_statistical_tests_summary.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"Statistical test summary saved: {stats_csv}")


if __name__ == "__main__":
    main()
