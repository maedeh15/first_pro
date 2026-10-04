# ============================================================================
# VERIFY TRIALS_PER_SESSION ASSUMPTION — n-back dataset (Shin et al., 2018)
#
# WHY: Preproc_tsfnet_nback_badch.py hard-codes TRIALS_PER_SESSION = 9 and
# derives session_id as:
#       session_id = trial_idx // TRIALS_PER_SESSION + 1
# This was verified directly only against VP001's mrk_nback.mat (27 events,
# codes 7/8/9, in three clean groups of 9 separated by a ~148s gap). If ANY
# other subject deviates from exactly 27 trial markers in three clean groups
# of 9, session_id (NOT the class labels) will silently be wrong for that
# subject -- trials near a session boundary would be assigned to the wrong
# session, which corrupts any session-wise split/normalization downstream
# (see the per-session normalization step in the main script).
#
# WHAT THIS SCRIPT DOES, per subject (VP001..VP026), using the SAME .mat
# parsing conventions as the main script (struct_as_record=False,
# squeeze_me=True, same mrk_nback.mat field-name fallbacks):
#   1. Loads EEG mrk_nback.mat -> extracts event codes + times (ms).
#   2. Keeps only the 3 session/series-onset codes (112/128/144 = 0-/2-/3-back).
#   3. Checks: total count == 27 ?
#   4. Splits events into contiguous session groups by looking for large time
#      gaps between consecutive trial onsets (a session boundary should look
#      like a ~100+ s gap vs. the ~62s 2s-instruction+40s-task+20s-rest
#      within-session spacing). Reports the actual group sizes found, e.g.
#      [9, 9, 9] (expected) vs. e.g. [8, 10, 9] (would break the naive
#      trial_idx // 9 assumption).
#   5. Cross-checks the SAME thing against fNIRS mrk_nback.mat (codes 7/8/9)
#      if present, since the main script also relies on these for timing
#      sync and they should agree in count/order with the EEG markers.
#   6. Writes a per-subject PASS/FAIL table + a global summary to stdout and
#      to trials_per_session_report.csv / .txt in OUTPUT_PATH.
#
# This script does NOT modify or depend on the main preprocessing script; it
# only reads the raw .mat files, so it's safe to run before/independently of
# a full preprocessing run.
#
# USAGE:
#   python verify_trials_per_session.py
#   python verify_trials_per_session.py --subjects 1,2,21
#   python verify_trials_per_session.py --data-dir "E:\kntu\Thesis\codes\DATA"
# ============================================================================

import os
import csv
import argparse
import numpy as np
import scipy.io as sio

# ----------------------------------------------------------------------------
# CONFIG (mirrors the main script's constants -- change here if yours differ)
# ----------------------------------------------------------------------------
DEFAULT_BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"   # folder holding VPxxx-EEG / VPxxx-NIRS
DEFAULT_OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"

EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}     # 0-back / 2-back / 3-back (EEG session/series onset)
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}         # matching fNIRS session/series onset codes

EXPECTED_TRIALS_PER_SESSION = 9
EXPECTED_N_SESSIONS = 3
EXPECTED_TOTAL_TRIALS = EXPECTED_TRIALS_PER_SESSION * EXPECTED_N_SESSIONS   # 27

# A within-session trial-to-trial gap is ~ 2s instruction + 40s task + 20s
# rest = ~62s. A between-session gap is much larger (~148s was observed for
# VP001). We use a conservative midpoint threshold to split trials into
# session groups purely from their timestamps, independent of any assumed
# fixed count.
SESSION_GAP_THRESHOLD_S = 90.0


# ============================================================================
# .mat LOADING (same convention as the main script)
# ============================================================================
def _load_struct(mat_path):
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


def load_eeg_trial_events(eeg_folder):
    """Returns (times_s, codes) for ALL markers in mrk_nback.mat (EEG),
    sorted by time. Mirrors load_eeg_data()'s event-extraction logic in the
    main script, but we don't need cnt_nback.mat here at all -- only mrk."""
    mrk_struct = _load_struct(os.path.join(eeg_folder, "mrk_nback.mat"))

    if hasattr(mrk_struct, "time"):
        times_ms = np.asarray(mrk_struct.time, dtype=float).flatten()
    elif hasattr(mrk_struct, "pos"):
        # 'pos' is in samples, not ms; without cnt.fs we cannot convert here,
        # but this dataset's real files use 'time' (ms) for mrk_nback.mat, so
        # this branch is just a defensive fallback matching the main script.
        times_ms = np.asarray(mrk_struct.pos, dtype=float).flatten()
    else:
        raise ValueError("Could not find 'time' or 'pos' in EEG marker struct")

    if hasattr(mrk_struct, "event") and hasattr(mrk_struct.event, "desc"):
        codes = np.asarray(mrk_struct.event.desc, dtype=int).flatten()
    elif hasattr(mrk_struct, "y"):
        y = mrk_struct.y
        codes = (np.argmax(y, axis=0) + 1) if y.ndim > 1 else y.flatten()
    else:
        raise ValueError("Could not find 'event.desc' or 'y' in EEG marker struct")

    order = np.argsort(times_ms)
    return times_ms[order] / 1000.0, codes[order]


def load_fnirs_trial_events(fnirs_folder):
    """Returns (times_s, codes) for ALL markers in mrk_nback.mat (fNIRS),
    sorted by time, or (None, None) if the file/fields are not present."""
    mrk_path = os.path.join(fnirs_folder, "mrk_nback.mat")
    if not os.path.exists(mrk_path):
        return None, None
    try:
        mrk_struct = _load_struct(mrk_path)
    except Exception:
        return None, None

    times_ms = None
    for attr in ("time",):
        if hasattr(mrk_struct, attr):
            times_ms = np.asarray(getattr(mrk_struct, attr), dtype=float).flatten()
            break
    if times_ms is None:
        return None, None

    codes = None
    if hasattr(mrk_struct, "event") and hasattr(mrk_struct.event, "desc"):
        codes = np.asarray(mrk_struct.event.desc, dtype=int).flatten()
    elif hasattr(mrk_struct, "y"):
        y = mrk_struct.y
        codes = (np.argmax(y, axis=0) + 1) if y.ndim > 1 else y.flatten()
    if codes is None:
        return None, None

    order = np.argsort(times_ms)
    return times_ms[order] / 1000.0, codes[order]


# ============================================================================
# SESSION-GROUPING PURELY FROM TIMESTAMPS (no assumed fixed count)
# ============================================================================
def split_into_session_groups(times_s, gap_threshold_s=SESSION_GAP_THRESHOLD_S):
    """Given sorted trial-onset times, split into contiguous groups wherever
    the gap to the next trial exceeds gap_threshold_s. Returns a list of
    group sizes, e.g. [9, 9, 9], and the list of (start_idx, end_idx) spans."""
    if len(times_s) == 0:
        return [], []
    diffs = np.diff(times_s)
    boundary_idx = np.where(diffs > gap_threshold_s)[0]  # index i means boundary AFTER trial i
    group_sizes, spans = [], []
    start = 0
    for b in boundary_idx:
        spans.append((start, int(b)))
        group_sizes.append(int(b - start + 1))
        start = b + 1
    spans.append((start, len(times_s) - 1))
    group_sizes.append(int(len(times_s) - start))
    return group_sizes, spans


# ============================================================================
# PER-SUBJECT CHECK
# ============================================================================
def check_subject(subject_id, base_dataset_path):
    eeg_folder = os.path.join(base_dataset_path, f"{subject_id}-EEG")
    fnirs_folder = os.path.join(base_dataset_path, f"{subject_id}-NIRS")

    result = {
        "subject_id": subject_id,
        "status": "ERROR",
        "n_eeg_trial_events": None,
        "eeg_group_sizes": None,
        "eeg_classes_per_group": None,
        "n_fnirs_trial_events": None,
        "fnirs_group_sizes": None,
        "eeg_fnirs_count_match": None,
        "eeg_fnirs_class_sequence_match": None,
        "notes": "",
    }

    # ---- EEG ----
    try:
        times_s, codes = load_eeg_trial_events(eeg_folder)
    except Exception as e:
        result["notes"] = f"EEG mrk_nback.mat load failed: {e}"
        return result

    mask = np.isin(codes, list(EEG_TRIAL_LABEL_MAP.keys()))
    trial_times = times_s[mask]
    trial_codes = codes[mask]
    classes = np.array([EEG_TRIAL_LABEL_MAP[c] for c in trial_codes])

    result["n_eeg_trial_events"] = int(len(trial_times))
    group_sizes, spans = split_into_session_groups(trial_times)
    result["eeg_group_sizes"] = group_sizes
    result["eeg_classes_per_group"] = [
        classes[s:e + 1].tolist() for (s, e) in spans
    ]

    # ---- fNIRS (cross-check only; not fatal if missing) ----
    try:
        f_times_s, f_codes = load_fnirs_trial_events(fnirs_folder)
    except Exception as e:
        f_times_s, f_codes = None, None
        result["notes"] += f" | fNIRS mrk_nback.mat load failed: {e}"

    if f_times_s is not None:
        f_mask = np.isin(f_codes, list(FNIRS_TRIAL_LABEL_MAP.keys()))
        f_trial_times = f_times_s[f_mask]
        f_trial_codes = f_codes[f_mask]
        f_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in f_trial_codes])

        result["n_fnirs_trial_events"] = int(len(f_trial_times))
        f_group_sizes, _ = split_into_session_groups(f_trial_times)
        result["fnirs_group_sizes"] = f_group_sizes

        result["eeg_fnirs_count_match"] = bool(len(trial_times) == len(f_trial_times))
        n_common = min(len(classes), len(f_classes))
        result["eeg_fnirs_class_sequence_match"] = bool(
            n_common > 0 and np.array_equal(classes[:n_common], f_classes[:n_common])
        )
    else:
        result["notes"] += " | fNIRS markers unavailable for cross-check"

    # ---- Verdict ----
    is_total_ok = (result["n_eeg_trial_events"] == EXPECTED_TOTAL_TRIALS)
    is_grouping_ok = (group_sizes == [EXPECTED_TRIALS_PER_SESSION] * EXPECTED_N_SESSIONS)

    if is_total_ok and is_grouping_ok:
        result["status"] = "PASS"
    elif is_total_ok and not is_grouping_ok:
        result["status"] = "FAIL (right total, wrong grouping)"
    else:
        result["status"] = "FAIL (wrong total)"

    return result


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Verify TRIALS_PER_SESSION=9 assumption against real mrk_nback.mat files."
    )
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject numbers, e.g. '1,2,21'. "
                              "Default: all 26 (VP001..VP026).")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_BASE_DATASET_PATH,
                         help="Folder holding VPxxx-EEG / VPxxx-NIRS subfolders.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUTPUT_PATH,
                         help="Folder to write the report into.")
    args = parser.parse_args()

    if args.subjects is not None:
        sub_nums = [int(s.strip()) for s in args.subjects.split(",")]
    else:
        sub_nums = list(range(1, 27))

    os.makedirs(args.out_dir, exist_ok=True)
    results = []

    print("=" * 100)
    print(f"Verifying TRIALS_PER_SESSION={EXPECTED_TRIALS_PER_SESSION} "
          f"(expected total={EXPECTED_TOTAL_TRIALS}) against: {args.data_dir}")
    print("=" * 100)

    for n in sub_nums:
        subject_id = f"VP{n:03d}"
        r = check_subject(subject_id, args.data_dir)
        results.append(r)
        print(f"{subject_id}: {r['status']:30s} "
              f"n_eeg_trials={r['n_eeg_trial_events']}  "
              f"eeg_groups={r['eeg_group_sizes']}  "
              f"n_fnirs_trials={r['n_fnirs_trial_events']}  "
              f"fnirs_groups={r['fnirs_group_sizes']}  "
              f"eeg_fnirs_count_match={r['eeg_fnirs_count_match']}  "
              f"eeg_fnirs_class_seq_match={r['eeg_fnirs_class_sequence_match']}"
              + (f"  NOTES: {r['notes']}" if r["notes"] else ""))

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"].startswith("FAIL"))
    n_error = sum(1 for r in results if r["status"] == "ERROR")

    print("-" * 100)
    print(f"SUMMARY: {n_pass} PASS / {n_fail} FAIL / {n_error} ERROR out of {len(results)} subjects")
    if n_fail or n_error:
        print("  -> TRIALS_PER_SESSION=9 (and the resulting trial_idx // 9 session_id formula)")
        print("     is NOT safe to assume for every subject. Fix the main script's session_id")
        print("     assignment for the flagged subject(s) before trusting per-session")
        print("     normalization / splits for them -- e.g. derive session_id directly from the")
        print("     timestamp-based grouping this script computes, instead of integer division.")
    else:
        print("  -> Confirmed for every requested subject: 27 trial markers/subject, in three")
        print("     clean groups of 9. trial_idx // 9 + 1 is safe to use as-is.")
    print("-" * 100)

    # ---- write CSV ----
    csv_path = os.path.join(args.out_dir, "trials_per_session_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id", "status", "n_eeg_trial_events", "eeg_group_sizes",
            "n_fnirs_trial_events", "fnirs_group_sizes",
            "eeg_fnirs_count_match", "eeg_fnirs_class_sequence_match", "notes"
        ])
        for r in results:
            writer.writerow([
                r["subject_id"], r["status"], r["n_eeg_trial_events"], r["eeg_group_sizes"],
                r["n_fnirs_trial_events"], r["fnirs_group_sizes"],
                r["eeg_fnirs_count_match"], r["eeg_fnirs_class_sequence_match"], r["notes"]
            ])
    print(f"Report written to: {csv_path}")


if __name__ == "__main__":
    main()
