# =============================================================================
# COMPREHENSIVE AUDIT SCRIPT -- zero training, zero GPU
# =============================================================================
# This script pulls together every remaining zero-cost check from the
# existing preprocessed data and result files. Each section is independent --
# if a required file is missing for one section, that section is skipped and
# the rest still runs.
#
# Sections:
#   1. Final confirmation of the trial structure (why [6,6,6] in the earlier
#      audit was expected, not a bug)
#   2. Extraction of the apples-to-apples baseline recall from the existing
#      results_*.json files (full 26 subjects, no retraining)
#   3. Audit C2 -- count of subjects with a trial-marker count != 27
#   4. Audit C3 -- count of fNIRS/EEG-clock fallback occurrences
#   5. Audit B4 -- indirect estimate of window-rejection severity
#
# Paste each block into its own Jupyter cell (cell boundaries marked with # %%).
# =============================================================================


# %% CELL 0 -- shared configuration
import os
import re
import json
import glob
import numpy as np
import pandas as pd

DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
PREPROC_LOG_PATH = os.path.join(DATA_ROOT, "preprocessing_log.txt")

NUM_CLASSES = 3
CLASS_NAMES = ["0-back", "2-back", "3-back"]

print("DATA_ROOT =", DATA_ROOT)
print("PREPROC_LOG_PATH =", PREPROC_LOG_PATH, "| exists?", os.path.exists(PREPROC_LOG_PATH))


# %% CELL 1 -- confirm trial structure (explains the earlier [6,6,6] result)
# =============================================================================
# Per the preprocessing code (line 334): TRIALS_PER_SESSION = 9, i.e. each
# session has exactly 9 trials. Per the comment on lines 325-326, those 9
# trials are 3 repetitions of each of the 3 conditions (0-back/2-back/3-back)
# -- i.e. 3 trials per class per session. With 2 sessions in train+val:
# 2 x 3 = 6 unique trials per class -- exactly matching the earlier audit
# result. This cell verifies that on your actual data.

subjects = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(DATA_ROOT, "VP*"))
    if os.path.isdir(p)
)

confirm_rows = []
for subject_id in subjects[:3]:  # a few sample subjects are enough to check
    trial_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_trial_ids.npy")
    session_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_session_ids.npy")
    labels_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_labels.npy")
    if not (os.path.exists(trial_path) and os.path.exists(session_path) and os.path.exists(labels_path)):
        continue
    trial_ids = np.load(trial_path)
    session_ids = np.load(session_path)
    labels = np.load(labels_path)

    for s in sorted(np.unique(session_ids)):
        mask = session_ids == s
        n_unique_trials_this_session = len(np.unique(trial_ids[mask]))
        per_class = {
            CLASS_NAMES[c]: len(np.unique(trial_ids[mask & (labels == c)]))
            for c in range(NUM_CLASSES)
        }
        confirm_rows.append({
            "subject_id": subject_id, "session": int(s),
            "n_unique_trials_total": n_unique_trials_this_session,
            **per_class,
        })

confirm_df = pd.DataFrame(confirm_rows)
print("Expected: n_unique_trials_total=9 per session, 3 per class")
print(confirm_df)
print()
if not confirm_df.empty and (confirm_df["n_unique_trials_total"] == 9).all():
    print("CONFIRMED: every session has exactly 9 unique trials (3 per class), "
          "so [6,6,6] in the earlier audit (2 sessions x 3 = 6) was correct and "
          "expected -- not a bug.")
else:
    print("WARNING: at least one session has a trial count other than 9 -- "
          "inspect the table above; this may reflect a real marker-count "
          "problem (see the C2 audit below).")


# %% CELL 2 -- extract apples-to-apples baseline recall (no new training run)
# =============================================================================
# Per the original code (detailed_test_report function, lines 709-750), every
# results_{subject}_session{N}.json contains a "confusion_matrix_final" key
# (row = true class, column = predicted class). This cell aggregates all of
# these matrices from the reference (no-asymmetric-cost) result bundle to get
# the fully apples-to-apples recall across all 26 subjects -- exactly what a
# fresh training run was previously going to be used for.
#
# If you have multiple result variants (baseline vs. asymmetric), point this
# at whichever variant's folder you want to summarize. This assumes the
# baseline results sit directly under DATA_ROOT/{subject}/results_{subject}
# _session{N}.json -- if you use variant-tagged subfolders, change
# BASELINE_RESULTS_ROOT accordingly.

BASELINE_RESULTS_ROOT = DATA_ROOT   # <<< change if needed

cm_total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
found_files = 0
missing_subjects = []

for subject_id in subjects:
    subj_found = False
    for session in [1, 2, 3]:
        json_path = os.path.join(BASELINE_RESULTS_ROOT, subject_id, f"results_{subject_id}_session{session}.json")
        if not os.path.exists(json_path):
            continue
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        cm = report.get("confusion_matrix_final")
        if cm is None:
            print(f"[warning] {json_path}: 'confusion_matrix_final' key not found.")
            continue
        cm_total += np.array(cm, dtype=np.int64)
        found_files += 1
        subj_found = True
    if not subj_found:
        missing_subjects.append(subject_id)

print(f"{found_files} results_*.json files processed (out of {len(subjects)} subjects).")
if missing_subjects:
    print(f"[warning] {len(missing_subjects)} subjects had no result file at all: {missing_subjects}")

print()
print("Aggregated confusion matrix (row = true, column = predicted):")
cm_df = pd.DataFrame(cm_total, index=[f"true_{c}" for c in CLASS_NAMES],
                      columns=[f"pred_{c}" for c in CLASS_NAMES])
print(cm_df)
print()

total_acc = np.trace(cm_total) / cm_total.sum()
print(f"Overall accuracy (from aggregated matrix): {total_acc:.4%}")
print()
print("Per-class recall:")
for i, cname in enumerate(CLASS_NAMES):
    row_sum = cm_total[i].sum()
    recall = cm_total[i, i] / row_sum if row_sum > 0 else float("nan")
    print(f"  {cname}: recall = {recall:.4%}  (n={row_sum})")

total_errors = cm_total.sum() - np.trace(cm_total)
pair_2_3_errors = cm_total[1, 2] + cm_total[2, 1]
print()
print(f"Total errors: {total_errors}")
if total_errors > 0:
    print(f"2-back<->3-back pair errors: {pair_2_3_errors} ({pair_2_3_errors/total_errors:.2%} of all errors)")


# %% CELL 3 -- Audit C2: count of subjects with trial-marker count != 27
# =============================================================================
# Per the code (lines 1292-1295 and line 334), if a subject's target_events
# count isn't exactly 27, the session mapping (trial_idx // 9 + 1) silently
# breaks. The current code does not assert this. This cell counts the actual
# number of unique trials directly from each subject's trial_ids.npy -- the
# most reliable method since it doesn't depend on the log -- and also pulls
# any related message from the preprocessing log.

c2_rows = []
for subject_id in subjects:
    trial_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_trial_ids.npy")
    if not os.path.exists(trial_path):
        continue
    trial_ids = np.load(trial_path)
    n_unique_total = len(np.unique(trial_ids))
    c2_rows.append({"subject_id": subject_id, "n_unique_trials_total": n_unique_total})

c2_df = pd.DataFrame(c2_rows)
print("Distribution of total unique trials per subject (expected: all = 27):")
print(c2_df["n_unique_trials_total"].value_counts().sort_index())
print()
anomalous = c2_df[c2_df["n_unique_trials_total"] != 27]
if anomalous.empty:
    print("OK: every subject has exactly 27 unique trials -- the session mapping is sound.")
else:
    print(f"WARNING: {len(anomalous)} subject(s) with a trial count other than 27:")
    print(anomalous)

if os.path.exists(PREPROC_LOG_PATH):
    with open(PREPROC_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()
    marker_warnings = re.findall(r".*no trial markers found.*", log_text)
    print(f"\n'no trial markers found' messages in the log: {len(marker_warnings)}")
    for w in marker_warnings[:10]:
        print(" ", w.strip())


# %% CELL 4 -- Audit C3: count of EEG-clock fallback occurrences for fNIRS
# =============================================================================
# Exact messages in the code (lines 1316-1320):
#   "WARNING - fNIRS/EEG trial markers mismatch, falling back to EEG-clock timing..."
#   "WARNING - fNIRS trial markers unavailable, falling back to EEG-clock timing..."
# This cell counts both patterns in the log and identifies which subjects
# were affected by potential fNIRS/EEG clock drift.

if os.path.exists(PREPROC_LOG_PATH):
    with open(PREPROC_LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        log_text = f.read()

    fallback_lines = re.findall(r".*falling back to EEG-clock timing.*", log_text)
    print(f"Total EEG-clock fallback lines: {len(fallback_lines)}")
    for line in fallback_lines:
        print(" ", line.strip())

    affected_subjects = []
    for line in fallback_lines:
        m = re.match(r"\s*(\S+):\s*WARNING", line)
        if m:
            affected_subjects.append(m.group(1))
    if affected_subjects:
        print(f"\nAffected subjects ({len(set(affected_subjects))} unique): {sorted(set(affected_subjects))}")
    else:
        print("\nNo affected subject could be extracted (the message format may differ slightly -- "
              "inspect the lines above manually).")
else:
    print(f"Preprocessing log not found at {PREPROC_LOG_PATH} -- this section skipped.")


# %% CELL 5 -- Audit B4: indirect estimate of window-rejection severity
# =============================================================================
# Three possible reasons a window gets dropped: out-of-range signal (line
# 1353/1374), the |z|>8 threshold (line 1366), and AutoReject (if enabled).
# The current code does not log these separately. This cell provides an
# indirect signal via the windows-per-trial distribution -- for an exact
# three-way breakdown, three separate counters would need to be added to the
# preprocessing script and it would need to be rerun once on these subjects.

b4_rows = []
for subject_id in subjects:
    labels_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_labels.npy")
    trial_path = os.path.join(DATA_ROOT, subject_id, f"{subject_id}_trial_ids.npy")
    if not (os.path.exists(labels_path) and os.path.exists(trial_path)):
        continue
    labels = np.load(labels_path)
    trial_ids = np.load(trial_path)

    windows_per_trial = pd.Series(trial_ids).value_counts()
    b4_rows.append({
        "subject_id": subject_id,
        "n_windows_total": len(labels),
        "n_trials": len(windows_per_trial),
        "mean_windows_per_trial": round(windows_per_trial.mean(), 2),
        "min_windows_per_trial": int(windows_per_trial.min()),
        "max_windows_per_trial": int(windows_per_trial.max()),
    })

b4_df = pd.DataFrame(b4_rows)
print("Windows-per-trial distribution across all subjects (indirect rejection-severity signal):")
print(b4_df.describe())
print()
print("5 subjects with the lowest mean windows-per-trial (highest relative rejection):")
print(b4_df.sort_values("mean_windows_per_trial").head())

print()
print("=" * 70)
print("Note: this is a rough indirect estimate. For an exact breakdown of the")
print("three rejection reasons, add three separate counters to the")
print("preprocessing script and rerun it once.")
print("=" * 70)
