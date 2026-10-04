# ============================================================================
# count_affected_windows.py -- exact sub-window overlap count vs. whole-trial
# ============================================================================
# WHY THIS EXISTS: a subject has only TRIALS_PER_SESSION x n_sessions trials
# total (27 for this dataset), but extract_windows() slices each trial into
# many overlapping 10s sub-windows via a sliding offset (SLIDE_TMIN to
# SLIDE_TMAX, step WINDOW_STEP -- currently ~34 sub-windows per trial). A
# burst artifact lasting only a few seconds does NOT necessarily contaminate
# all 34 sub-windows of the trial it falls in -- only the ones whose 10s span
# actually overlaps the burst. Counting "how many trials have a burst nearby"
# (as detect_bad_segments.py's task_events_overlapping does) drastically
# OVER-estimates the true cost if you were to discard whole trials, and
# UNDER-estimates it if you were to naively assume "1 trial = 1 window".
# This script replicates extract_windows()'s EXACT per-trial offset loop
# (same constants: WINDOW_LEN_EEG, SLIDE_TMIN, SLIDE_TMAX, WINDOW_STEP,
# TRIALS_PER_SESSION) -- reusing them directly from the pipeline module, not
# retyped -- and checks each individual sub-window's [start,end] time range
# against the bad segment(s), so you get the TRUE fraction of the dataset at
# stake before deciding whether/how to act on a burst.
#
# NOTE: this only needs the raw EEG timing and trial event markers, not the
# full filter/PyPREP/ICA chain, so it runs in a couple of seconds per subject.
#
# USAGE:
#   # Using an existing detect_bad_segments.py output directly:
#   python count_affected_windows.py --subjects 1 --pipeline-script "..." \
#       --bad-segments-json "...\QC_VALIDATION\VP001\VP001_bad_segments.json"
#
#   # Or specify burst window(s) manually:
#   python count_affected_windows.py --subjects 1 --pipeline-script "..." \
#       --burst-windows "656,962"
#
# OUTPUT: printed per-trial breakdown + overall totals, and a JSON:
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_affected_windows.json
# ============================================================================

import os
import sys
import json
import argparse
import importlib.util
import datetime

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=str, required=True)
parser.add_argument("--pipeline-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Preproc_tsfnet_nback_badch.py"))
parser.add_argument("--bad-segments-json", type=str, default=None,
                     help="Path to a {subject}_bad_segments.json (from detect_bad_segments.py) "
                          "whose 'segments' start_sec/end_sec are used as the bad windows to check "
                          "against. Mutually usable alongside --burst-windows (both are unioned).")
parser.add_argument("--burst-windows", type=str, default=None,
                     help="Optional manual 'start,end;start,end' in seconds, e.g. '656,962'. "
                          "Unioned with --bad-segments-json if both are given.")
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]

if not os.path.isfile(args.pipeline_script):
    sys.exit(f"ERROR: pipeline script not found at {args.pipeline_script!r}.")

_spec = importlib.util.spec_from_file_location("tsfnet_pipeline", args.pipeline_script)
pp = importlib.util.module_from_spec(_spec)
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(pp)

print(f"[count_affected_windows] Loaded pipeline module from: {args.pipeline_script}")
print(f"[count_affected_windows] WINDOW_LEN_EEG={pp.WINDOW_LEN_EEG}s, SLIDE_TMIN={pp.SLIDE_TMIN}s, "
      f"SLIDE_TMAX={pp.SLIDE_TMAX}s, WINDOW_STEP={pp.WINDOW_STEP}s, TRIALS_PER_SESSION={pp.TRIALS_PER_SESSION}")
QC_VALIDATION_ROOT = os.path.join(pp.OUTPUT_PATH, "QC_VALIDATION")


def load_bad_windows():
    """Returns a list of (start_sec, end_sec) tuples, unioning --bad-segments-json and
    --burst-windows if both are given."""
    windows = []
    if args.bad_segments_json:
        if not os.path.isfile(args.bad_segments_json):
            sys.exit(f"ERROR: --bad-segments-json file not found: {args.bad_segments_json!r}")
        with open(args.bad_segments_json) as f:
            data = json.load(f)
        for seg in data.get("segments", []):
            windows.append((float(seg["start_sec"]), float(seg["end_sec"])))
    if args.burst_windows:
        for chunk in args.burst_windows.split(";"):
            chunk = chunk.strip()
            if not chunk:
                continue
            start, end = [float(x.strip()) for x in chunk.split(",")]
            windows.append((start, end))
    if not windows:
        sys.exit("ERROR: no bad windows given -- pass --bad-segments-json and/or --burst-windows.")
    return windows


def overlaps(a_start, a_end, b_start, b_end):
    return a_start < b_end and b_start < a_end


def enumerate_all_subwindows(events, sfreq_eeg, total_duration_sec):
    """Exactly replicates extract_windows()'s per-trial offset loop (timing only -- no signal
    data needed), reusing the pipeline's own constants so this can never silently drift out of
    sync with the real extraction logic. Returns a list of dicts, one per sub-window that WOULD
    be extracted (i.e. respects the same in-bounds check extract_windows() applies), each with
    its trial index, label, and exact [start_sec, end_sec]."""
    target_events = events[np.isin(events[:, 2], list(pp.EEG_TRIAL_LABEL_MAP.keys()))]
    offsets = np.arange(pp.SLIDE_TMIN, pp.SLIDE_TMAX - pp.WINDOW_LEN_EEG + 1e-6, pp.WINDOW_STEP)

    subwindows = []
    for trial_idx, ev in enumerate(target_events):
        label = pp.EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1)
        if label == -1:
            continue
        session_id = trial_idx // pp.TRIALS_PER_SESSION + 1
        eeg_event_time = ev[0] / sfreq_eeg

        for offset in offsets:
            w_start = eeg_event_time + offset
            w_end = w_start + pp.WINDOW_LEN_EEG
            # Same in-bounds check as extract_windows() (w_start_eeg < 0 or w_end_eeg > n_times).
            if w_start < 0 or w_end > total_duration_sec:
                continue
            subwindows.append({
                "trial_idx": int(trial_idx),
                "session_id": int(session_id),
                "label": int(label),
                "trial_onset_sec": float(eeg_event_time),
                "window_start_sec": float(w_start),
                "window_end_sec": float(w_end),
            })
    return subwindows, len(target_events)


def main():
    bad_windows = load_bad_windows()
    print(f"Checking against {len(bad_windows)} bad window(s): {bad_windows}\n")

    for sub_num in SUBJECT_NUMS:
        subject_id = f"VP{sub_num:03d}"
        eeg_folder = os.path.join(pp.BASE_DATASET_PATH, f"{subject_id}-EEG")
        if not os.path.isdir(eeg_folder):
            print(f"{subject_id}: SKIPPED (folder not found: {eeg_folder})")
            continue

        print(f"=== {subject_id} ===")
        try:
            raw, events, sfreq_eeg, eeg_mnt_struct = pp.load_eeg_data(eeg_folder)
            total_duration_sec = raw.n_times / sfreq_eeg

            subwindows, n_trials = enumerate_all_subwindows(events, sfreq_eeg, total_duration_sec)
            for sw in subwindows:
                sw["overlaps_bad_window"] = any(
                    overlaps(sw["window_start_sec"], sw["window_end_sec"], bs, be) for bs, be in bad_windows
                )

            n_total = len(subwindows)
            n_affected = sum(sw["overlaps_bad_window"] for sw in subwindows)
            pct_affected = 100.0 * n_affected / n_total if n_total > 0 else 0.0

            trials_with_any_affected_subwindow = sorted(set(
                sw["trial_idx"] for sw in subwindows if sw["overlaps_bad_window"]
            ))
            trials_fully_affected = sorted(
                t for t in trials_with_any_affected_subwindow
                if all(sw["overlaps_bad_window"] for sw in subwindows if sw["trial_idx"] == t)
            )

            print(f"  n_trials (raw event markers): {n_trials}")
            print(f"  n_subwindows extracted (in-bounds, matches extract_windows()): {n_total}")
            print(f"  n_subwindows overlapping a bad window: {n_affected} ({pct_affected:.2f}% of this subject's windows)")
            print(f"  n_trials with >=1 affected sub-window: {len(trials_with_any_affected_subwindow)} "
                  f"(out of {n_trials} trials, {100.0 * len(trials_with_any_affected_subwindow) / max(n_trials,1):.1f}%)")
            print(f"  n_trials FULLY affected (every sub-window overlaps): {len(trials_fully_affected)}")

            # Per-trial breakdown, only for trials that have at least one affected sub-window
            # (keeps the printout short -- full detail is still in the saved JSON).
            print(f"  Per-trial breakdown (only trials with >=1 affected sub-window):")
            for t in trials_with_any_affected_subwindow:
                trial_subs = [sw for sw in subwindows if sw["trial_idx"] == t]
                n_aff_this_trial = sum(sw["overlaps_bad_window"] for sw in trial_subs)
                print(f"    trial {t:>2} (label={trial_subs[0]['label']}, onset={trial_subs[0]['trial_onset_sec']:.1f}s): "
                      f"{n_aff_this_trial}/{len(trial_subs)} sub-windows affected")

            out_dir = os.path.join(QC_VALIDATION_ROOT, subject_id)
            os.makedirs(out_dir, exist_ok=True)
            result = {
                "subject_id": subject_id,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "bad_windows_checked": bad_windows,
                "n_trials": n_trials,
                "n_subwindows_total": n_total,
                "n_subwindows_affected": n_affected,
                "pct_subwindows_affected": pct_affected,
                "n_trials_with_any_affected_subwindow": len(trials_with_any_affected_subwindow),
                "n_trials_fully_affected": len(trials_fully_affected),
                "subwindows": subwindows,
            }
            out_path = os.path.join(out_dir, f"{subject_id}_affected_windows.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  -> {out_path}\n")

        except Exception as e:
            import traceback
            print(f"{subject_id}: ERROR ({e})")
            traceback.print_exc()


if __name__ == "__main__":
    main()
