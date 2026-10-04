# ============================================================================
# detect_bad_segments.py -- Auto-detect bad TIME SEGMENTS (not bad channels,
# not bad ICs) in continuous EEG, for cases like VP001 where a burst artifact
# survives filter+PyPREP+CAR+ICA because it's not a single stable spatial
# component ICA can linearly separate (confirmed empirically: forcing IC3
# out for VP001 barely changed the burst and made CAR->ICA / line-noise worse
# -- see VP001_full_validation.json test-vs-baseline comparison).
# ============================================================================
# METHOD (standard PREP-pipeline-style segment QC, applied on the
# post-filter+PyPREP+CAR signal -- i.e. AFTER the steps that are known to
# behave well, so this only catches what's left):
#   1. Split the continuous signal into non-overlapping windows (default 2s).
#   2. For each window, compute the RMS amplitude PER CHANNEL, then take the
#      MAX across channels (a burst that hits several channels at once, like
#      VP001's, will dominate this even if only some channels spike).
#   3. Convert each window's max-channel RMS to a robust z-score across ALL
#      windows in the recording, using the MEDIAN and MAD (median absolute
#      deviation) instead of mean/std, so the burst windows themselves don't
#      inflate the baseline they're being compared against.
#   4. Flag any window whose z-score exceeds --zscore-threshold (default 5).
#   5. Merge flagged windows that are within --merge-gap-sec (default 2s) of
#      each other into a single segment, so a burst that spans several
#      consecutive windows is reported as ONE segment with a start/end time,
#      not a scattered list of individual windows.
#   6. Cross-reference each flagged segment against the task's event markers
#      (from load_eeg_data's `events`) so you know whether the burst overlaps
#      actual n-back trials (matters a lot more) or falls in a rest/inter-
#      trial period (matters less).
#
# OUTPUT (per subject):
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_bad_segments.json
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_bad_segments_overlay.png
#     (full-duration stacked-channel plot, same style as the other QC plots,
#      with flagged segments shaded in red so you can confirm by eye)
#
# USAGE:
#   python detect_bad_segments.py --subjects 1
#   python detect_bad_segments.py --subjects 1,2,3 --zscore-threshold 4 --window-sec 2
#
# NOTE: read-only w.r.t. your dataset and pipeline module; writes only into
# the QC_VALIDATION subfolder, same as eeg_qc_full_validation.py. Detecting a
# segment here does NOT remove it from anything automatically -- it's a
# reconnaissance tool. Acting on a bad segment (annotating it as BAD in the
# real pipeline so it's excluded from window extraction) is a separate,
# deliberate edit to Preproc_tsfnet_nback_badch.py you'd make after reviewing
# these results, the same way IC overrides were only added after review.
# ============================================================================

import os
import sys
import json
import argparse
import importlib.util
import datetime

import numpy as np
import mne

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=str, required=True,
                     help="Comma-separated subject numbers, e.g. '1,2,3'.")
parser.add_argument("--pipeline-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Preproc_tsfnet_nback_badch.py"))
parser.add_argument("--window-sec", type=float, default=2.0,
                     help="Window length (seconds) for the sliding burst-detection scan (default 2.0).")
parser.add_argument("--zscore-threshold", type=float, default=5.0,
                     help="Robust z-score threshold (median/MAD-based) above which a window is "
                          "flagged as an amplitude burst (default 5.0 -- deliberately conservative; "
                          "lower it to 3-4 if bursts are being missed, raise it if too many benign "
                          "windows get flagged).")
parser.add_argument("--merge-gap-sec", type=float, default=2.0,
                     help="Flagged windows within this many seconds of each other are merged into "
                          "one reported segment (default 2.0).")
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]

if not os.path.isfile(args.pipeline_script):
    sys.exit(f"ERROR: pipeline script not found at {args.pipeline_script!r}. "
              f"Pass --pipeline-script <full path to Preproc_tsfnet_nback_badch.py>.")

_spec = importlib.util.spec_from_file_location("tsfnet_pipeline", args.pipeline_script)
pp = importlib.util.module_from_spec(_spec)
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(pp)

print(f"[detect_bad_segments] Loaded pipeline module from: {args.pipeline_script}")
QC_VALIDATION_ROOT = os.path.join(pp.OUTPUT_PATH, "QC_VALIDATION")


# ----------------------------------------------------------------------------
# Re-run filter -> PyPREP -> CAR only (NOT ICA -- segment detection is meant
# to catch what survives the channel-level cleaning steps, independent of
# which ICA components get excluded, since VP001 showed the burst isn't a
# clean single-IC phenomenon).
# ----------------------------------------------------------------------------
def clean_up_to_car(raw, subject_id):
    filtered = raw.copy()
    if pp.APPLY_NOTCH:
        filtered.notch_filter(freqs=pp.EEG_NOTCH_FREQ, picks="eeg", verbose=False)
    filtered.filter(
        l_freq=pp.EEG_BP_LOW, h_freq=pp.EEG_BP_HIGH, method="iir",
        iir_params=dict(order=pp.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    if pp._PYPREP_AVAILABLE:
        try:
            nd = pp.NoisyChannels(filtered, random_state=42)
            try:
                nd.find_all_bads(ransac=True)
            except Exception:
                nd.find_all_bads(ransac=False)
            bads = nd.get_bads()
            filtered.info["bads"] = bads
            if bads:
                filtered.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:
            print(f"  {subject_id}: PyPREP step FAILED ({e}).")
    filtered.set_eeg_reference("average", projection=False, verbose=False)
    return filtered


def detect_burst_segments(filtered, window_sec, zscore_threshold, merge_gap_sec):
    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    data = filtered.get_data(picks=picks_eeg) * 1e6  # -> uV
    sfreq = filtered.info["sfreq"]
    win_samp = int(round(window_sec * sfreq))
    n_ch, n_times = data.shape
    n_windows = n_times // win_samp
    if n_windows < 5:
        return [], np.array([]), np.array([])

    trimmed = data[:, : n_windows * win_samp].reshape(n_ch, n_windows, win_samp)
    rms_per_ch_win = np.sqrt((trimmed ** 2).mean(axis=2))  # (n_ch, n_windows)
    max_rms_per_win = rms_per_ch_win.max(axis=0)  # (n_windows,) -- worst channel in each window

    median = np.median(max_rms_per_win)
    mad = np.median(np.abs(max_rms_per_win - median))
    # 1.4826x scales MAD to be a consistent estimator of the standard deviation for Gaussian data
    # -- the standard robust-z-score convention (same convention PyPREP/Automagic-style tools use).
    robust_std = 1.4826 * mad if mad > 0 else np.std(max_rms_per_win)
    z_scores = (max_rms_per_win - median) / max(robust_std, 1e-12)

    flagged_windows = np.where(z_scores > zscore_threshold)[0]

    # merge contiguous/near-contiguous flagged windows into segments
    segments = []
    if len(flagged_windows) > 0:
        merge_gap_windows = max(1, int(round(merge_gap_sec / window_sec)))
        start_idx = flagged_windows[0]
        prev_idx = flagged_windows[0]
        for idx in flagged_windows[1:]:
            if idx - prev_idx > merge_gap_windows:
                segments.append((start_idx, prev_idx))
                start_idx = idx
            prev_idx = idx
        segments.append((start_idx, prev_idx))

    segment_times = [
        {
            "start_sec": float(s * window_sec),
            "end_sec": float((e + 1) * window_sec),
            "duration_sec": float((e - s + 1) * window_sec),
            "max_zscore_in_segment": float(z_scores[s:e + 1].max()),
            "worst_channel": filtered.ch_names[picks_eeg[int(np.argmax(rms_per_ch_win[:, s:e + 1].max(axis=1)))]],
        }
        for s, e in segments
    ]
    return segment_times, z_scores, max_rms_per_win


def count_events_in_segments(events, sfreq, segments):
    """How many task event markers (n-back trial onsets) fall inside each flagged segment --
    tells you whether the burst overlaps actual trials (worse) or a rest period (less bad)."""
    if events is None or len(events) == 0:
        return
    event_times_sec = np.asarray(events)[:, 0] / sfreq if events.ndim == 2 else np.asarray(events) / sfreq
    for seg in segments:
        n_events = int(np.sum((event_times_sec >= seg["start_sec"]) & (event_times_sec <= seg["end_sec"])))
        seg["n_task_events_overlapping"] = n_events


def plot_overlay(filtered, subject_id, segments, out_path):
    raw_eeg_only = filtered.copy().pick(picks="eeg")
    sfreq = raw_eeg_only.info["sfreq"]
    data = raw_eeg_only.get_data()
    t = np.arange(data.shape[1]) / sfreq
    ch_names_plot = raw_eeg_only.ch_names
    spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 30 + 1e-12
    n_ch_plot = data.shape[0]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * n_ch_plot)))
    for seg in segments:
        ax.axvspan(seg["start_sec"], seg["end_sec"], color="red", alpha=0.15, zorder=0)
    for i in range(n_ch_plot):
        offset = (n_ch_plot - 1 - i) * spread
        ax.plot(t, data[i] + offset, color="tab:blue", linewidth=0.4, zorder=1)
    ax.set_yticks([(n_ch_plot - 1 - i) * spread for i in range(n_ch_plot)])
    ax.set_yticklabels(ch_names_plot, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{subject_id} - post-filter+PyPREP+CAR, flagged burst segments shaded red "
                 f"({len(segments)} segment(s))")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    for sub_num in SUBJECT_NUMS:
        subject_id = f"VP{sub_num:03d}"
        eeg_folder = os.path.join(pp.BASE_DATASET_PATH, f"{subject_id}-EEG")
        if not os.path.isdir(eeg_folder):
            print(f"{subject_id}: SKIPPED (folder not found: {eeg_folder})")
            continue

        print(f"\n=== {subject_id} ===")
        try:
            raw, events, sfreq_eeg, eeg_mnt_struct = pp.load_eeg_data(eeg_folder)
            filtered = clean_up_to_car(raw, subject_id)
            segments, z_scores, max_rms_per_win = detect_burst_segments(
                filtered, args.window_sec, args.zscore_threshold, args.merge_gap_sec,
            )
            count_events_in_segments(events, sfreq_eeg, segments)

            total_duration_sec = filtered.n_times / filtered.info["sfreq"]
            total_flagged_sec = sum(s["duration_sec"] for s in segments)
            pct_flagged = 100.0 * total_flagged_sec / total_duration_sec if total_duration_sec > 0 else 0.0

            out_dir = os.path.join(QC_VALIDATION_ROOT, subject_id)
            os.makedirs(out_dir, exist_ok=True)

            result = {
                "subject_id": subject_id,
                "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "window_sec": args.window_sec,
                "zscore_threshold": args.zscore_threshold,
                "total_duration_sec": float(total_duration_sec),
                "n_segments_flagged": len(segments),
                "total_flagged_sec": float(total_flagged_sec),
                "pct_recording_flagged": float(pct_flagged),
                "segments": segments,
            }
            out_json = os.path.join(out_dir, f"{subject_id}_bad_segments.json")
            with open(out_json, "w") as f:
                json.dump(result, f, indent=2)

            out_png = os.path.join(out_dir, f"{subject_id}_bad_segments_overlay.png")
            plot_overlay(filtered, subject_id, segments, out_png)

            print(f"  {len(segments)} segment(s) flagged, {pct_flagged:.2f}% of recording ({total_flagged_sec:.1f}s / {total_duration_sec:.1f}s)")
            for seg in segments:
                print(f"    {seg['start_sec']:.1f}s - {seg['end_sec']:.1f}s "
                      f"(z={seg['max_zscore_in_segment']:.1f}, worst_ch={seg['worst_channel']}, "
                      f"task_events_overlapping={seg.get('n_task_events_overlapping', 'n/a')})")
            print(f"  -> {out_json}")
            print(f"  -> {out_png}")

        except Exception as e:
            import traceback
            print(f"{subject_id}: ERROR ({e})")
            traceback.print_exc()

    print(f"\nDone. Output root: {QC_VALIDATION_ROOT}")


if __name__ == "__main__":
    main()
