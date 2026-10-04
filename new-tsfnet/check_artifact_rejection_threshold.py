# ============================================================================
# ARTIFACT-REJECTION THRESHOLD VALIDATION (single run, all checks)
#
# Answers, in one run:
#   1. What is the REAL effect of the original 120uV fixed amplitude threshold?
#      -> Histogram of per-window max-amplitude (in real uV, computed on the
#         post-filter+CAR+ICA signal, BEFORE the z-score normalization that
#         the main pipeline currently applies) across ALL final windows for
#         several subjects.
#      -> Exact percentage of windows that would be rejected by the 120uV rule.
#      -> If effectively nothing is rejected, this is stated explicitly and
#         numerically (not just "the pipeline works fine").
#   2. How does an ADAPTIVE per-subject threshold (mean + 5*std of that
#      subject's own window max-amplitudes) compare to the fixed 120uV rule?
#      -> Side-by-side rejection percentages for both rules, per subject and
#         aggregated.
#   3. For completeness/traceability, also reports what the CURRENTLY
#      IMPLEMENTED z-score-based threshold (8.0, applied post-normalization,
#      see preprocess_eeg()'s corrected-ordering z-score + extract_windows()'s
#      artifact check) would reject on the SAME underlying windows, so all
#      three rules (fixed-uV, adaptive-uV, current z-score) are directly
#      comparable in one table.
#
# This script does NOT modify the main pipeline -- it is purely diagnostic/
# reporting, replicating extract_windows()'s exact window-slicing logic
# (WINDOW_LEN, WINDOW_STEP, SLIDE_TMIN/TMAX) on a handful of subjects to
# answer "what would each rule actually have done".
#
# Usage:
#   python check_artifact_rejection_threshold.py --n_subjects 5
#   python check_artifact_rejection_threshold.py --subjects VP001,VP010,VP020
# ============================================================================

import os
import sys
import argparse

import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_NAME = "Preprocessing_TSFNet_paper_nback_v2_without_override2210"
if MAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MAIN_SCRIPT_DIR)
main_pipeline = __import__(MAIN_SCRIPT_NAME)

BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
OUTPUT_PATH = main_pipeline.OUTPUT_PATH
ALL_SUBJECTS = [f"VP{n:03d}" for n in main_pipeline.SUBJECTS]

FIXED_THRESHOLD_UV = 120.0     # the original literature-standard threshold, in microvolts
ADAPTIVE_N_STD = 5.0           # mean + N*std, computed per subject from its own window distribution
CURRENT_ZSCORE_THRESHOLD = 8.0 # the threshold actually implemented in extract_windows() right now


def preprocess_eeg_no_zscore(subject_id):
    """Replicates preprocess_eeg() EXACTLY (bandpass -> CAR -> ICA with the same
    EOG-threshold-override + MANUAL_IC_OVERRIDES logic) but returns the signal
    in real physical VOLTS, i.e. BEFORE the per-channel z-score normalization
    step -- this is the signal whose amplitude the original 120uV threshold
    was meant to act on."""
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    raw, events, sfreq_eeg, eeg_mnt_struct = main_pipeline.load_eeg_data(eeg_folder)
    raw_before = raw.copy()
    filtered = raw.copy()
    if main_pipeline.APPLY_NOTCH:
        filtered.notch_filter(freqs=main_pipeline.EEG_NOTCH_FREQ, picks="eeg", verbose=False)
    filtered.filter(
        l_freq=main_pipeline.EEG_BP_LOW, h_freq=main_pipeline.EEG_BP_HIGH, method="iir",
        iir_params=dict(order=main_pipeline.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True), random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    eog_threshold = main_pipeline._get_subject_eog_threshold(subject_id)
    eog_indices = []
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=eog_threshold, verbose=False)
                eog_indices.extend(idx)
            except Exception:
                pass
    try:
        import mne_icalabel
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        iclabel_ocular = []
    exclude_set = set(eog_indices) | set(iclabel_ocular)
    exclude_set, _ = main_pipeline._apply_manual_ic_overrides(subject_id, exclude_set)
    ica.exclude = sorted(exclude_set)
    if ica.exclude:
        ica.apply(filtered, verbose=False)

    return filtered, events, sfreq_eeg   # filtered here is in VOLTS (no z-score applied)


def get_window_max_amplitudes(filtered, events, sfreq_eeg, scale=1.0):
    """Replicates extract_windows()'s exact window-slicing (same offsets,
    same window length/step) and returns, for each valid window, the max
    absolute amplitude (multiplied by `scale`, e.g. 1e6 for volts->uV)."""
    eeg_full = filtered.get_data(picks=mne.pick_types(filtered.info, eeg=True, eog=False))
    win_samp_eeg = int(main_pipeline.WINDOW_LEN * sfreq_eeg)

    max_amps = []
    for eeg_event_time, _ in events:
        offsets = np.arange(main_pipeline.SLIDE_TMIN,
                             main_pipeline.SLIDE_TMAX - main_pipeline.WINDOW_LEN + 1e-6,
                             main_pipeline.WINDOW_STEP)
        for offset in offsets:
            w_start = int(np.round((eeg_event_time + offset) * sfreq_eeg))
            w_end = w_start + win_samp_eeg
            if w_start < 0 or w_end > eeg_full.shape[1]:
                continue
            eeg_slice = eeg_full[:, w_start:w_end] * scale
            max_amps.append(float(np.max(np.abs(eeg_slice))))
    return np.array(max_amps)


def get_events_list(events, sfreq_eeg):
    """Filters the raw (N,3) MNE events array to trial-marker events only,
    EXACTLY like extract_windows() does via EEG_TRIAL_LABEL_MAP, and converts
    each to (time_seconds, code)."""
    target_events = events[np.isin(events[:, 2], list(main_pipeline.EEG_TRIAL_LABEL_MAP.keys()))]
    return [(ev[0] / sfreq_eeg, int(ev[2])) for ev in target_events]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--n_subjects", type=int, default=5)
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(OUTPUT_PATH, "artifact_rejection_validation"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        step = max(1, len(ALL_SUBJECTS) // args.n_subjects)
        subjects = ALL_SUBJECTS[::step][:args.n_subjects]

    print(f"\n=== Artifact-rejection threshold validation on {subjects} ===\n")

    per_subject_rows = []
    all_max_amps_uv = []
    all_max_amps_z = []

    for subject_id in subjects:
        print(f"--- {subject_id} ---")
        try:
            filtered, events_arr, sfreq_eeg = preprocess_eeg_no_zscore(subject_id)
            events = get_events_list(events_arr, sfreq_eeg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        max_amps_uv = get_window_max_amplitudes(filtered, events, sfreq_eeg, scale=1e6)
        if len(max_amps_uv) == 0:
            print("  No valid windows found, skipping.")
            continue

        n_windows = len(max_amps_uv)
        n_rejected_fixed = int(np.sum(max_amps_uv > FIXED_THRESHOLD_UV))
        pct_rejected_fixed = 100.0 * n_rejected_fixed / n_windows

        subj_mean, subj_std = max_amps_uv.mean(), max_amps_uv.std()
        adaptive_threshold_uv = subj_mean + ADAPTIVE_N_STD * subj_std
        n_rejected_adaptive = int(np.sum(max_amps_uv > adaptive_threshold_uv))
        pct_rejected_adaptive = 100.0 * n_rejected_adaptive / n_windows

        # for comparison: what the CURRENTLY implemented z-score-based rule would reject,
        # computed on a per-channel z-scored version of the SAME data (matches
        # preprocess_eeg()'s actual normalization step)
        picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
        data_v = filtered.get_data(picks=picks_eeg)
        ch_mean = data_v.mean(axis=1, keepdims=True)
        ch_std = data_v.std(axis=1, keepdims=True) + 1e-8
        data_z = (data_v - ch_mean) / ch_std
        filtered_z = filtered.copy()
        filtered_z._data[picks_eeg, :] = data_z
        max_amps_z = get_window_max_amplitudes(filtered_z, events, sfreq_eeg, scale=1.0)
        n_rejected_zscore = int(np.sum(max_amps_z > CURRENT_ZSCORE_THRESHOLD))
        pct_rejected_zscore = 100.0 * n_rejected_zscore / max(len(max_amps_z), 1)

        print(f"  n_windows={n_windows} | max-amp range: {max_amps_uv.min():.1f}-{max_amps_uv.max():.1f} uV, "
              f"mean={subj_mean:.1f}, std={subj_std:.1f}")
        print(f"  FIXED 120uV:      rejected {n_rejected_fixed}/{n_windows} ({pct_rejected_fixed:.2f}%)")
        print(f"  ADAPTIVE ({adaptive_threshold_uv:.1f}uV): rejected {n_rejected_adaptive}/{n_windows} ({pct_rejected_adaptive:.2f}%)")
        print(f"  CURRENT z-score (>{CURRENT_ZSCORE_THRESHOLD}): rejected {n_rejected_zscore}/{n_windows} ({pct_rejected_zscore:.2f}%)")

        per_subject_rows.append({
            "subject_id": subject_id, "n_windows": n_windows,
            "mean_max_amp_uv": round(subj_mean, 2), "std_max_amp_uv": round(subj_std, 2),
            "adaptive_threshold_uv": round(adaptive_threshold_uv, 2),
            "n_rejected_fixed_120uv": n_rejected_fixed, "pct_rejected_fixed_120uv": round(pct_rejected_fixed, 3),
            "n_rejected_adaptive": n_rejected_adaptive, "pct_rejected_adaptive": round(pct_rejected_adaptive, 3),
            "n_rejected_current_zscore": n_rejected_zscore, "pct_rejected_current_zscore": round(pct_rejected_zscore, 3),
        })
        all_max_amps_uv.append(max_amps_uv)
        all_max_amps_z.append(max_amps_z)

    if not per_subject_rows:
        print("No subjects processed successfully.")
        return

    df = pd.DataFrame(per_subject_rows)
    out_csv = os.path.join(args.out_dir, "artifact_rejection_summary.csv")
    df.to_csv(out_csv, index=False)

    pooled = np.concatenate(all_max_amps_uv)
    pooled_z = np.concatenate(all_max_amps_z)
    n_total = len(pooled)
    n_rejected_fixed_total = int(np.sum(pooled > FIXED_THRESHOLD_UV))
    pct_fixed_total = 100.0 * n_rejected_fixed_total / n_total
    n_rejected_zscore_total = int(np.sum(pooled_z > CURRENT_ZSCORE_THRESHOLD))
    pct_zscore_total = 100.0 * n_rejected_zscore_total / n_total

    print("\n" + "=" * 78)
    print(f"POOLED ACROSS ALL {len(per_subject_rows)} SUBJECTS ({n_total} total windows)")
    print("=" * 78)
    print(f"FIXED 120uV threshold:        {n_rejected_fixed_total}/{n_total} windows rejected ({pct_fixed_total:.3f}%)")
    print(f"CURRENT z-score(>{CURRENT_ZSCORE_THRESHOLD}) threshold: {n_rejected_zscore_total}/{n_total} windows rejected ({pct_zscore_total:.3f}%)")
    print(f"Mean adaptive threshold across subjects: {df['adaptive_threshold_uv'].mean():.1f} uV "
          f"(range {df['adaptive_threshold_uv'].min():.1f}-{df['adaptive_threshold_uv'].max():.1f})")

    if n_rejected_fixed_total == 0:
        print("\n*** The fixed 120uV threshold rejects ZERO windows across all sampled subjects. ***")
        print("*** It is effectively a no-op safety net at this data scale, not an active filter. ***")

    # ----- histogram -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(pooled, bins=80, color="steelblue", edgecolor="none")
    axes[0].axvline(FIXED_THRESHOLD_UV, color="red", linestyle="--", label=f"Fixed threshold ({FIXED_THRESHOLD_UV} uV)")
    axes[0].axvline(df["adaptive_threshold_uv"].mean(), color="orange", linestyle="--",
                     label=f"Mean adaptive threshold ({df['adaptive_threshold_uv'].mean():.1f} uV)")
    axes[0].set_xlabel("Per-window max amplitude (uV)")
    axes[0].set_ylabel("Number of windows")
    axes[0].set_title(f"Distribution of window max-amplitude (all {len(per_subject_rows)} subjects pooled)")
    axes[0].legend()

    axes[1].hist(pooled, bins=80, color="steelblue", edgecolor="none")
    axes[1].set_xlim(0, min(FIXED_THRESHOLD_UV * 1.5, pooled.max()))
    axes[1].axvline(FIXED_THRESHOLD_UV, color="red", linestyle="--", label=f"Fixed threshold ({FIXED_THRESHOLD_UV} uV)")
    axes[1].set_xlabel("Per-window max amplitude (uV)")
    axes[1].set_title("Zoomed view near the 120 uV threshold")
    axes[1].legend()

    fig.tight_layout()
    hist_path = os.path.join(args.out_dir, "artifact_rejection_histogram.png")
    fig.savefig(hist_path, dpi=150)
    plt.close(fig)

    print(f"\nHistogram saved: {hist_path}")
    print(f"Per-subject summary: {out_csv}")

    # ----- thesis-ready paragraph -----
    if n_rejected_fixed_total == 0:
        fixed_sentence = (
            f"The fixed {FIXED_THRESHOLD_UV:.0f}uV amplitude threshold rejected 0 of {n_total} "
            f"({pct_fixed_total:.2f}%) final windows across all sampled subjects, i.e. it did not "
            f"actively reject any window at this data scale and functioned as an inactive safety net "
            f"rather than a meaningful artifact filter."
        )
    else:
        fixed_sentence = (
            f"The fixed {FIXED_THRESHOLD_UV:.0f}uV amplitude threshold rejected {n_rejected_fixed_total} "
            f"of {n_total} ({pct_fixed_total:.2f}%) final windows across all sampled subjects."
        )
    paragraph = (
        f"To assess the real effect of the amplitude-based artifact-rejection threshold, the maximum "
        f"absolute amplitude of every final window (post-filter+CAR+ICA, before z-score normalization) "
        f"was computed for {len(per_subject_rows)} subjects ({n_total} windows total). {fixed_sentence} "
        f"An adaptive per-subject alternative (mean + {ADAPTIVE_N_STD:.0f} standard deviations of that "
        f"subject's own window-amplitude distribution, mean threshold across subjects = "
        f"{df['adaptive_threshold_uv'].mean():.1f}uV) was also tested for comparison. For reference, the "
        f"threshold actually implemented in the pipeline (a z-score-space equivalent, >{CURRENT_ZSCORE_THRESHOLD} "
        f"standard deviations after per-channel normalization) rejected {n_rejected_zscore_total} of "
        f"{n_total} ({pct_zscore_total:.2f}%) windows on the same underlying data."
    )
    print(f"\n--- Thesis-ready paragraph ---\n{paragraph}")
    with open(os.path.join(args.out_dir, "artifact_rejection_paragraph.txt"), "w", encoding="utf-8") as f:
        f.write(paragraph)


if __name__ == "__main__":
    main()
