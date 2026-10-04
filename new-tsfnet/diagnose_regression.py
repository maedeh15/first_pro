# ============================================================================
# REGRESSION DIAGNOSIS -- one run, all checks
#
# Compares OLD vs NEW per-subject accuracy/kappa (from two baseline_all_subjects_
# *.csv files), flags subjects that got WORSE, then automatically pulls every
# preprocessing diagnostic available in the NEW run's log for those subjects
# (and, for contrast, the same diagnostics for subjects that IMPROVED), so you
# can see at a glance whether the regressed subjects share a common cause:
#   - unusually few windows extracted (e.g. artifact-rejection threshold issue)
#   - a manual EOG-threshold or IC-exclusion override that was applied
#   - an unusually high/low number of ICA components excluded
#   - a fNIRS/EEG sync issue
#
# Usage:
#   python diagnose_regression.py --old_csv "baseline_all_subjects_OLD.csv" \
#                                   --new_csv "baseline_all_subjects_NEW.csv" \
#                                   --log "full_preprocessing_log.txt"
# ============================================================================

import os
import re
import argparse

import pandas as pd


def load_baseline(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["mean_acc"] = pd.to_numeric(df["mean_acc"], errors="coerce")
    df["mean_kappa"] = pd.to_numeric(df["mean_kappa"], errors="coerce")
    return df[["subject_id", "mean_acc", "mean_kappa"]]


def parse_log_diagnostics(log_path):
    """Extracts per-subject diagnostic lines from the preprocessing log:
    window counts, EOG threshold used, manual overrides, ICA exclusion counts."""
    diag = {}   # subject_id -> dict of collected info

    if not os.path.exists(log_path):
        print(f"WARNING: log file not found at {log_path} -- diagnostics will be empty.")
        return diag

    patterns = {
        "eeg_windows_shape": re.compile(r"(VP\d+).*?EEG \((\d+), \d+, \d+, \d+\)"),
        "fnirs_windows_shape": re.compile(r"(VP\d+).*?fNIRS \((\d+),"),
        "eog_threshold": re.compile(r"(VP\d+): EOG threshold used = ([\d.]+) \((\w+)"),
        "manual_override": re.compile(r"(VP\d+): MANUAL OVERRIDE -- removed IC\(s\) (\[.*?\])"),
        "removed_ic_count": re.compile(r"(VP\d+): removed (\d+) ocular ICA component"),
        "no_windows": re.compile(r"(VP\d+): no windows extracted"),
        "error": re.compile(r"\[(VP\d+)\].*?FAILED|ERROR processing (VP\d+)"),
    }

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = patterns["eeg_windows_shape"].search(line)
            if m:
                diag.setdefault(m.group(1), {})["n_eeg_windows"] = int(m.group(2))
                continue
            m = patterns["fnirs_windows_shape"].search(line)
            if m:
                diag.setdefault(m.group(1), {})["n_fnirs_windows"] = int(m.group(2))
                continue
            m = patterns["eog_threshold"].search(line)
            if m:
                d = diag.setdefault(m.group(1), {})
                d["eog_threshold"] = float(m.group(2))
                d["eog_threshold_source"] = m.group(3)
                continue
            m = patterns["manual_override"].search(line)
            if m:
                diag.setdefault(m.group(1), {})["manual_ic_override"] = m.group(2)
                continue
            m = patterns["removed_ic_count"].search(line)
            if m:
                diag.setdefault(m.group(1), {})["n_ic_removed"] = int(m.group(2))
                continue
            m = patterns["no_windows"].search(line)
            if m:
                diag.setdefault(m.group(1), {})["zero_windows_flag"] = True
                continue
            m = patterns["error"].search(line)
            if m:
                sid = m.group(1) or m.group(2)
                if sid:
                    diag.setdefault(sid, {})["error_in_log"] = line.strip()

    return diag


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_csv", type=str, required=True)
    parser.add_argument("--new_csv", type=str, required=True)
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.new_csv)
    os.makedirs(out_dir, exist_ok=True)

    old_df = load_baseline(args.old_csv).rename(columns={"mean_acc": "acc_old", "mean_kappa": "kappa_old"})
    new_df = load_baseline(args.new_csv).rename(columns={"mean_acc": "acc_new", "mean_kappa": "kappa_new"})

    merged = pd.merge(old_df, new_df, on="subject_id", how="outer")
    merged["delta_acc"] = merged["acc_new"] - merged["acc_old"]
    merged["delta_kappa"] = merged["kappa_new"] - merged["kappa_old"]
    merged = merged.sort_values("delta_acc")

    regressed = merged[merged["delta_acc"] < 0].copy()
    improved = merged[merged["delta_acc"] > 0].copy()

    print("=" * 90)
    print(f"OVERALL: old mean acc = {old_df['acc_old'].mean():.4f} -> "
          f"new mean acc = {new_df['acc_new'].mean():.4f} "
          f"(delta = {new_df['acc_new'].mean() - old_df['acc_old'].mean():+.4f})")
    print(f"Regressed subjects: {len(regressed)} | Improved subjects: {len(improved)}")
    print("=" * 90)

    diag = parse_log_diagnostics(args.log)

    diag_cols = ["n_eeg_windows", "n_fnirs_windows", "eog_threshold", "eog_threshold_source",
                 "manual_ic_override", "n_ic_removed", "zero_windows_flag", "error_in_log"]

    def attach_diag(df):
        for col in diag_cols:
            df[col] = df["subject_id"].map(lambda sid: diag.get(sid, {}).get(col, ""))
        return df

    regressed = attach_diag(regressed)
    improved = attach_diag(improved)

    print("\n--- REGRESSED subjects (worse after new preprocessing) ---")
    print(regressed[["subject_id", "acc_old", "acc_new", "delta_acc"] + diag_cols].to_string(index=False))

    print("\n--- IMPROVED subjects (for contrast, same diagnostics) ---")
    print(improved[["subject_id", "acc_old", "acc_new", "delta_acc"] + diag_cols].to_string(index=False))

    # ----- automated pattern check: does any diagnostic differ systematically -----
    print("\n" + "=" * 90)
    print("AUTOMATED PATTERN CHECK (regressed vs improved group)")
    print("=" * 90)

    if not regressed.empty and not improved.empty:
        for col in ["n_eeg_windows", "n_fnirs_windows", "n_ic_removed"]:
            reg_vals = pd.to_numeric(regressed[col], errors="coerce").dropna()
            imp_vals = pd.to_numeric(improved[col], errors="coerce").dropna()
            if len(reg_vals) and len(imp_vals):
                print(f"{col}: regressed mean={reg_vals.mean():.1f} (n={len(reg_vals)})  "
                      f"vs improved mean={imp_vals.mean():.1f} (n={len(imp_vals)})")

        n_override_regressed = (regressed["manual_ic_override"] != "").sum()
        n_override_improved = (improved["manual_ic_override"] != "").sum()
        print(f"manual_ic_override present: regressed={n_override_regressed}/{len(regressed)}  "
              f"vs improved={n_override_improved}/{len(improved)}")

        n_zero_windows_regressed = (regressed["zero_windows_flag"] == True).sum()
        print(f"zero_windows_flag (extraction failure) among regressed: {n_zero_windows_regressed}/{len(regressed)}")

        n_error_regressed = (regressed["error_in_log"] != "").sum()
        print(f"error_in_log present among regressed: {n_error_regressed}/{len(regressed)}")
    else:
        print("Not enough data in one of the groups to compare.")

    out_csv = os.path.join(out_dir, "regression_diagnosis_report.csv")
    merged_full = attach_diag(merged.copy())
    merged_full.to_csv(out_csv, index=False)
    print(f"\nFull comparison + diagnostics saved: {out_csv}")


if __name__ == "__main__":
    main()
