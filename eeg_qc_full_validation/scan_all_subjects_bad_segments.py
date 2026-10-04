# ============================================================================
# scan_all_subjects_bad_segments.py -- RECONNAISSANCE ONLY, no decisions made
# ============================================================================
# Runs the exact same burst-segment detection as detect_bad_segments.py across
# a list of subjects and prints/saves ONE summary table. This script does NOT
# decide anything and does NOT annotate/modify anything -- it exists purely
# so you can see, at a glance, which subjects even have a pattern worth a
# closer per-subject look (their own detect_bad_segments.py run + visual
# inspection of the overlay plot + a case-by-case decision), and which
# subjects clearly don't. Different subjects are expected to need different
# treatment (or none at all) -- this table is a triage step, not a verdict.
#
# USAGE:
#   python scan_all_subjects_bad_segments.py --subjects 1,2,3,...,26
#   (same --zscore-threshold / --min-channels / --window-sec as
#    detect_bad_segments.py, applied identically to every subject purely so
#    the numbers in the table are comparable to each other -- NOT because
#    the same threshold is assumed to be the right final answer for all of
#    them.)
#
# OUTPUT:
#   {OUTPUT_PATH}/QC_VALIDATION/_all_subjects_bad_segments_summary.json
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_bad_segments.json          (per-subject, same as detect_bad_segments.py)
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_bad_segments_overlay.png   (per-subject, same as detect_bad_segments.py)
#   Printed summary table, sorted by pct_recording_flagged descending, so the
#   subjects most worth a closer look float to the top.
# ============================================================================

import os
import sys
import json
import glob
import argparse
import importlib.util
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=str, required=True,
                     help="Comma-separated subject numbers, e.g. '1,2,3,...,26'.")
parser.add_argument("--pipeline-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Preproc_tsfnet_nback_badch.py"))
parser.add_argument("--detect-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "detect_bad_segments.py"),
                     help="Path to detect_bad_segments.py, whose functions are reused as-is.")
parser.add_argument("--window-sec", type=float, default=2.0)
parser.add_argument("--zscore-threshold", type=float, default=4.0)
parser.add_argument("--min-channels", type=int, default=6)
parser.add_argument("--merge-gap-sec", type=float, default=2.0)
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]

if not os.path.isfile(args.pipeline_script):
    sys.exit(f"ERROR: pipeline script not found at {args.pipeline_script!r}.")
if not os.path.isfile(args.detect_script):
    sys.exit(f"ERROR: detect_bad_segments.py not found at {args.detect_script!r}. "
              f"Pass --detect-script <full path to detect_bad_segments.py>.")

# Import the pipeline module first (detect_bad_segments.py's own CLI parsing expects to do this
# itself when run standalone, but here we import its FUNCTIONS as a library, so we replicate the
# same loading it does, then hand it an already-loaded `pp` via monkeypatch-free direct calls).
_pp_spec = importlib.util.spec_from_file_location("tsfnet_pipeline", args.pipeline_script)
pp = importlib.util.module_from_spec(_pp_spec)
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
_pp_spec.loader.exec_module(pp)

# Import detect_bad_segments.py as a library. IMPORTANT: that script runs argparse and loads the
# pipeline module itself at import time (it's written to double as a standalone CLI tool), so we
# temporarily set sys.argv to what IT expects, let it load its own (separate) `pp` instance, then
# reuse its already-defined functions (clean_up_to_car, detect_burst_segments,
# count_events_in_segments, plot_overlay) directly -- no logic is duplicated/reimplemented here.
sys.argv = [sys.argv[0], "--subjects", "1", "--pipeline-script", args.pipeline_script]
_detect_spec = importlib.util.spec_from_file_location("detect_bad_segments_lib", args.detect_script)
detect_lib = importlib.util.module_from_spec(_detect_spec)
_detect_spec.loader.exec_module(detect_lib)
sys.argv = _saved_argv

print(f"[scan_all_subjects] Reusing detect_bad_segments.py functions from: {args.detect_script}")
QC_VALIDATION_ROOT = os.path.join(pp.OUTPUT_PATH, "QC_VALIDATION")


def print_table(title, rows_sorted):
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)
    print(f"{'subject':>8} | {'n_segments':>10} | {'pct_flagged':>11} | {'flagged_sec':>11} | {'task_events_hit':>15}")
    print("-" * 90)
    for r in rows_sorted:
        print(f"{r['subject_id']:>8} | {r['n_segments_flagged']:>10} | {r['pct_recording_flagged']:>10.2f}% | "
              f"{r['total_flagged_sec']:>11.1f} | {r['n_task_events_overlapping_total']:>15}")


def build_cumulative_summary(qc_validation_root):
    """Reads EVERY {subject}_bad_segments.json currently on disk under QC_VALIDATION -- including
    ones written by earlier, separate batch runs of this script (or of detect_bad_segments.py
    directly) -- and returns one combined, de-duplicated (by subject_id, keeping the most
    recently generated file per subject) summary. This is what lets you run subjects in small
    batches across multiple sessions and still get a single final table at the end, without
    having to re-run everyone in one giant command.
    """
    pattern = os.path.join(qc_validation_root, "VP*", "VP*_bad_segments.json")
    cumulative = {}
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                data = json.load(f)
            subject_id = data.get("subject_id")
            if subject_id is None:
                continue
            # Normalize: files written directly by detect_bad_segments.py (run standalone,
            # outside this batch script) don't have a top-level n_task_events_overlapping_total
            # -- derive it from the segment list so both sources merge into one consistent table.
            if "n_task_events_overlapping_total" not in data:
                data["n_task_events_overlapping_total"] = sum(
                    s.get("n_task_events_overlapping", 0) for s in data.get("segments", [])
                )
            existing = cumulative.get(subject_id)
            if existing is None or data.get("generated_at", "") >= existing.get("generated_at", ""):
                cumulative[subject_id] = data
        except Exception as e:
            print(f"  WARNING: could not read {path} ({e}), skipping it in the cumulative summary.")
    return sorted(cumulative.values(), key=lambda r: r["pct_recording_flagged"], reverse=True)


def main():
    rows = []
    for sub_num in SUBJECT_NUMS:
        subject_id = f"VP{sub_num:03d}"
        eeg_folder = os.path.join(pp.BASE_DATASET_PATH, f"{subject_id}-EEG")
        if not os.path.isdir(eeg_folder):
            print(f"{subject_id}: SKIPPED (folder not found: {eeg_folder})")
            continue

        print(f"=== {subject_id} ===")
        try:
            raw, events, sfreq_eeg, eeg_mnt_struct = pp.load_eeg_data(eeg_folder)
            filtered = detect_lib.clean_up_to_car(raw, subject_id)
            segments, n_channels_elevated, rms_per_ch_win = detect_lib.detect_burst_segments(
                filtered, args.window_sec, args.zscore_threshold, args.merge_gap_sec, args.min_channels,
            )
            detect_lib.count_events_in_segments(events, sfreq_eeg, segments)

            total_duration_sec = filtered.n_times / filtered.info["sfreq"]
            total_flagged_sec = sum(s["duration_sec"] for s in segments)
            pct_flagged = 100.0 * total_flagged_sec / total_duration_sec if total_duration_sec > 0 else 0.0
            n_events_overlapping = sum(s.get("n_task_events_overlapping", 0) for s in segments)

            out_dir = os.path.join(QC_VALIDATION_ROOT, subject_id)
            os.makedirs(out_dir, exist_ok=True)
            result = {
                "subject_id": subject_id,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "window_sec": args.window_sec, "zscore_threshold": args.zscore_threshold,
                "min_channels": args.min_channels,
                "total_duration_sec": float(total_duration_sec),
                "n_segments_flagged": len(segments),
                "total_flagged_sec": float(total_flagged_sec),
                "pct_recording_flagged": float(pct_flagged),
                "n_task_events_overlapping_total": int(n_events_overlapping),
                "segments": segments,
            }
            with open(os.path.join(out_dir, f"{subject_id}_bad_segments.json"), "w") as f:
                json.dump(result, f, indent=2)
            detect_lib.plot_overlay(filtered, subject_id,
                                     segments, os.path.join(out_dir, f"{subject_id}_bad_segments_overlay.png"))

            rows.append(result)
            print(f"  {len(segments)} segment(s), {pct_flagged:.2f}% flagged, "
                  f"{n_events_overlapping} task event(s) overlapping")

        except Exception as e:
            import traceback
            print(f"{subject_id}: ERROR ({e})")
            traceback.print_exc()

    rows_sorted = sorted(rows, key=lambda r: r["pct_recording_flagged"], reverse=True)
    print_table("THIS BATCH ONLY (triage -- decide per-subject before acting on any of this)", rows_sorted)

    batch_summary_path = os.path.join(QC_VALIDATION_ROOT, "_this_batch_bad_segments_summary.json")
    os.makedirs(QC_VALIDATION_ROOT, exist_ok=True)
    with open(batch_summary_path, "w") as f:
        json.dump(rows_sorted, f, indent=2)
    print(f"\n-> {batch_summary_path}")

    # CUMULATIVE summary: re-reads every {subject}_bad_segments.json on disk (this batch AND any
    # earlier batches, from earlier runs of this script or of detect_bad_segments.py directly),
    # so subjects can be scanned across multiple separate sessions and still end up in one final
    # combined table -- no need to run all subjects in a single command.
    cumulative_sorted = build_cumulative_summary(QC_VALIDATION_ROOT)
    print_table(f"CUMULATIVE ACROSS ALL RUNS FOUND ON DISK ({len(cumulative_sorted)} subject(s) total)",
                cumulative_sorted)

    cumulative_path = os.path.join(QC_VALIDATION_ROOT, "_cumulative_bad_segments_summary.json")
    with open(cumulative_path, "w") as f:
        json.dump(cumulative_sorted, f, indent=2)
    print(f"\n-> {cumulative_path}")


if __name__ == "__main__":
    main()
