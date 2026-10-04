# ============================================================================
# eeg_qc_metrics.py -- Numeric QC diagnostics for the TSFNet EEG pipeline
# ============================================================================
# WHY THIS SCRIPT EXISTS
# -----------------------
# The main pipeline (Preproc_tsfnet_nback_badch.py) saves ONLY the PNG QC
# plots and a z-score-NORMALIZED {subject}_eeg_clean_raw.fif. It does not
# currently write out the numeric signal-level metrics used in your ablation
# report (max_variance_ratio_after, still_bad_after_cleaning, EMG-band
# residual power, EEG-band power-retention ratio).
#
# CRITICAL GOTCHA THIS SCRIPT WORKS AROUND:
#   preprocess_eeg() in the main script applies a PER-CHANNEL z-score
#   BEFORE returning `filtered`, and *that* normalized object is what gets
#   saved to {subject}_eeg_clean_raw.fif. Per-channel z-scoring forces every
#   channel's variance to ~1.0 by construction -- so computing
#   max_variance_ratio_after from the saved .fif would ALWAYS return ~1.0,
#   silently hiding exactly the bad channels you're trying to find.
#
#   To avoid this, this script does NOT read the saved .fif for the ratio
#   metrics. Instead it re-runs the identical filter -> PyPREP -> CAR -> ICA
#   (+ your manual per-subject overrides) chain used inside preprocess_eeg(),
#   using the SAME helper functions imported directly from the main
#   pipeline module (so there is exactly one copy of the actual cleaning
#   logic -- this script does not re-implement it, it re-uses it), and stops
#   BEFORE the z-score step. That intermediate signal is still in real
#   physical volts, exactly matching what your Ablation Study report (Section
#   2.2, "Signal-level QC" metrics) is defined on.
#
# WHAT IT COMPUTES (definitions per your own ablation report, Sec. 2.2):
#   - max_variance_ratio_after : max(per-channel variance) / median(...)
#                                 AFTER cleaning. ~1 = healthy energy spread.
#   - still_bad_after_cleaning : channel NAMES whose variance AFTER cleaning
#                                 exceeds 4x the across-channel median
#                                 (same 4x threshold your report already uses).
#   - emg_band_power_db        : residual power in the 30-49 Hz band AFTER
#                                 cleaning, per channel (dB) via Welch PSD,
#                                 plus the single noisiest channel.
#   - eeg_band_retention_ratio : ratio of 4-30 Hz band power AFTER / BEFORE
#                                 cleaning, per channel and mean. Values far
#                                 below 1 flag possible over-cleaning.
#   - pyprep_bads_detected     : channel(s) PyPREP itself flagged+interpolated
#                                 (cross-check -- if a channel shows up in
#                                 still_bad_after_cleaning but NOT here, PyPREP
#                                 missed it and interpolation never ran on it).
#   - n_ica_excluded           : number of ICA components removed (+ their
#                                 indices), for reference.
#
# OUTPUT:
#   {OUTPUT_PATH}/{subject}/{subject}_qc_metrics.json   (one file per subject)
#   + a printed summary table across all requested subjects, with any
#     still-bad channel or borderline retention ratio flagged with "<-- FLAG".
#
# USAGE (run from the same folder as Preproc_tsfnet_nback_badch.py, or pass
# --pipeline-script with the full path):
#   python eeg_qc_metrics.py --subjects 1,2
#   python eeg_qc_metrics.py --subjects 1,2,3,4,5 --pipeline-script "E:\kntu\Thesis\codes\Preproc_tsfnet_nback_badch.py"
#
# NOTE: this script only READS your dataset and the pipeline module's
# functions/constants. It does not modify Preproc_tsfnet_nback_badch.py and
# does not touch any of its saved outputs.
# ============================================================================

import os
import sys
import json
import argparse
import importlib.util
import datetime

import numpy as np
import mne
from scipy.signal import welch

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=str, required=True,
                     help="Comma-separated subject numbers, e.g. '1,2' or '1,2,3,4,5'.")
parser.add_argument("--pipeline-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Preproc_tsfnet_nback_badch.py"),
                     help="Path to Preproc_tsfnet_nback_badch.py (default: same folder as this script).")
parser.add_argument("--var-ratio-threshold", type=float, default=4.0,
                     help="Variance-ratio threshold for 'still_bad_after_cleaning' (default 4.0, "
                          "matches your ablation report's definition).")
parser.add_argument("--retention-flag-threshold", type=float, default=0.5,
                     help="Flag a channel if its EEG-band (4-30Hz) retention ratio drops below this "
                          "(default 0.5 = lost more than half the in-band power).")
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]

# ----------------------------------------------------------------------------
# Import the pipeline module BY FILE PATH (so this works regardless of CWD),
# re-using its constants and helper functions rather than duplicating them.
# The pipeline script's own RUN section is guarded by `if __name__ ==
# "__main__":`, so importing it here does NOT launch the full 26-subject run.
# ----------------------------------------------------------------------------
if not os.path.isfile(args.pipeline_script):
    sys.exit(f"ERROR: pipeline script not found at {args.pipeline_script!r}. "
              f"Pass --pipeline-script <full path to Preproc_tsfnet_nback_badch.py>.")

_spec = importlib.util.spec_from_file_location("tsfnet_pipeline", args.pipeline_script)
pp = importlib.util.module_from_spec(_spec)
sys.argv = [sys.argv[0]]  # prevent the pipeline module's own argparse from seeing OUR --subjects/etc.
_spec.loader.exec_module(pp)

print(f"[eeg_qc_metrics] Loaded pipeline module from: {args.pipeline_script}")
print(f"[eeg_qc_metrics] BASE_DATASET_PATH = {pp.BASE_DATASET_PATH}")
print(f"[eeg_qc_metrics] OUTPUT_PATH       = {pp.OUTPUT_PATH}")
print(f"[eeg_qc_metrics] PyPREP available  = {pp._PYPREP_AVAILABLE}")


# ----------------------------------------------------------------------------
# Re-run filter -> PyPREP -> CAR -> ICA(+overrides), stopping BEFORE the
# z-score step. Mirrors preprocess_eeg() in the pipeline module line-for-line
# (same functions/constants imported from `pp`), MINUS the final
# normalization block, so the returned signal is directly comparable (in real
# physical volts) to raw_before for variance/power-ratio metrics.
# ----------------------------------------------------------------------------
def clean_eeg_pre_normalization(raw, subject_id):
    raw_before = raw.copy()
    filtered = raw.copy()

    if pp.APPLY_NOTCH:
        filtered.notch_filter(freqs=pp.EEG_NOTCH_FREQ, picks="eeg", verbose=False)

    filtered.filter(
        l_freq=pp.EEG_BP_LOW, h_freq=pp.EEG_BP_HIGH,
        method="iir",
        iir_params=dict(order=pp.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )

    pyprep_bads = []
    if pp._PYPREP_AVAILABLE:
        try:
            nd = pp.NoisyChannels(filtered, random_state=42)
            try:
                nd.find_all_bads(ransac=True)
            except Exception as e:
                print(f"  {subject_id}: PyPREP RANSAC failed ({e}); retrying without RANSAC.")
                nd.find_all_bads(ransac=False)
            pyprep_bads = nd.get_bads()
            filtered.info["bads"] = pyprep_bads
            if pyprep_bads:
                filtered.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:
            print(f"  {subject_id}: PyPREP step FAILED ({e}); continuing without bad-channel interpolation.")
    else:
        print(f"  {subject_id}: pyprep not installed -- skipping bad-channel detection/interpolation.")

    filtered.set_eeg_reference("average", projection=False, verbose=False)

    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    eog_indices = []
    eog_threshold = pp._get_subject_eog_threshold(subject_id)
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=eog_threshold, verbose=False)
                eog_indices.extend(idx)
            except Exception as e:
                print(f"  {subject_id}: find_bads_eog({eog_ch}) failed: {e}")

    try:
        ic_labels = pp.mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        iclabel_ocular = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    exclude_set, override_logs = pp._apply_manual_ic_overrides(subject_id, set(ica.exclude))
    for msg in override_logs:
        print(f"  {msg}")
    ica.exclude = sorted(exclude_set)

    if ica.exclude:
        ica.apply(filtered, verbose=False)

    # Cast to plain Python int/str here (not numpy int64/str_) so the metrics dict built from
    # these is safely json.dump()-able downstream without a custom encoder.
    ica_excluded_plain = [int(x) for x in ica.exclude]
    pyprep_bads_plain = [str(x) for x in pyprep_bads]

    return raw_before, filtered, pyprep_bads_plain, ica_excluded_plain, eog_threshold


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def _band_power_db(raw_obj, fmin, fmax, picks_eeg):
    """Mean PSD power (dB, 10log10 uV^2/Hz) per channel within [fmin, fmax]."""
    data = raw_obj.get_data(picks=picks_eeg)  # (n_ch, n_times), volts
    sfreq = raw_obj.info["sfreq"]
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    # psd is in V^2/Hz; convert to uV^2/Hz for readability, matching the pipeline's own PSD plots.
    band_power_uv2 = psd[:, band_mask].mean(axis=1) * (1e6 ** 2)
    band_power_db = 10 * np.log10(np.maximum(band_power_uv2, 1e-20))
    return band_power_db, band_power_uv2


def compute_qc_metrics(subject_id, raw_before, filtered_pre_zscore, pyprep_bads, ica_excluded,
                        eog_threshold, var_ratio_threshold, retention_flag_threshold):
    picks_eeg = mne.pick_types(filtered_pre_zscore.info, eeg=True, eog=False)
    ch_names = [filtered_pre_zscore.ch_names[i] for i in picks_eeg]

    data_after = filtered_pre_zscore.get_data(picks=picks_eeg)   # volts
    data_before = raw_before.get_data(picks=picks_eeg)           # volts

    # --- variance-ratio metrics (AFTER cleaning) ---
    var_after = data_after.var(axis=1)
    median_var_after = float(np.median(var_after))
    max_variance_ratio_after = float(var_after.max() / median_var_after) if median_var_after > 0 else float("nan")
    var_ratio_per_channel = (var_after / median_var_after).tolist() if median_var_after > 0 else [float("nan")] * len(ch_names)
    still_bad_after_cleaning = [
        ch_names[i] for i in range(len(ch_names))
        if median_var_after > 0 and (var_after[i] / median_var_after) > var_ratio_threshold
    ]

    # --- EMG residual band (30-49 Hz), AFTER cleaning ---
    emg_db_after, emg_uv2_after = _band_power_db(filtered_pre_zscore, 30.0, 49.0, picks_eeg)
    noisiest_emg_idx = int(np.argmax(emg_db_after))

    # --- EEG in-band (4-30 Hz) retention ratio, AFTER / BEFORE ---
    eeg_db_before, eeg_uv2_before = _band_power_db(raw_before, 4.0, 30.0, picks_eeg)
    eeg_db_after, eeg_uv2_after = _band_power_db(filtered_pre_zscore, 4.0, 30.0, picks_eeg)
    retention_ratio = eeg_uv2_after / np.maximum(eeg_uv2_before, 1e-20)
    low_retention_channels = [
        ch_names[i] for i in range(len(ch_names)) if retention_ratio[i] < retention_flag_threshold
    ]

    metrics = {
        "subject_id": subject_id,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_channels": len(ch_names),
        "channel_names": ch_names,
        "eog_threshold_used": eog_threshold,
        "pyprep_bads_detected": pyprep_bads,
        "n_ica_excluded": len(ica_excluded),
        "ica_excluded_indices": ica_excluded,
        "max_variance_ratio_after": max_variance_ratio_after,
        "var_ratio_threshold_used": var_ratio_threshold,
        "still_bad_after_cleaning": still_bad_after_cleaning,
        "per_channel_variance_ratio_after": dict(zip(ch_names, var_ratio_per_channel)),
        "emg_band_30_49hz_db_after": dict(zip(ch_names, emg_db_after.tolist())),
        "emg_band_noisiest_channel": ch_names[noisiest_emg_idx],
        "emg_band_noisiest_channel_db": float(emg_db_after[noisiest_emg_idx]),
        "eeg_band_4_30hz_retention_ratio": dict(zip(ch_names, retention_ratio.tolist())),
        "eeg_band_retention_ratio_mean": float(retention_ratio.mean()),
        "eeg_band_retention_flag_threshold": retention_flag_threshold,
        "eeg_band_low_retention_channels": low_retention_channels,
    }
    return metrics


def print_summary_row(metrics):
    flags = []
    if metrics["still_bad_after_cleaning"]:
        flags.append(f"STILL-BAD: {metrics['still_bad_after_cleaning']}")
    if metrics["eeg_band_low_retention_channels"]:
        flags.append(f"LOW-RETENTION: {metrics['eeg_band_low_retention_channels']}")
    flag_str = "  <-- FLAG (" + "; ".join(flags) + ")" if flags else "  OK"
    print(
        f"{metrics['subject_id']:>6} | "
        f"max_var_ratio={metrics['max_variance_ratio_after']:6.2f} | "
        f"emg_noisiest={metrics['emg_band_noisiest_channel']:>6} "
        f"({metrics['emg_band_noisiest_channel_db']:6.2f} dB) | "
        f"eeg_retention_mean={metrics['eeg_band_retention_ratio_mean']:5.3f} | "
        f"pyprep_bads={metrics['pyprep_bads_detected']}"
        f"{flag_str}"
    )


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    print(f"\n{'subject':>6} | {'max_var_ratio':>18} | {'emg_noisiest':>20} | "
          f"{'eeg_retention_mean':>19} | pyprep_bads / flags")
    print("-" * 110)

    all_metrics = []
    for sub_num in SUBJECT_NUMS:
        subject_id = f"VP{sub_num:03d}"
        eeg_folder = os.path.join(pp.BASE_DATASET_PATH, f"{subject_id}-EEG")
        if not os.path.isdir(eeg_folder):
            print(f"{subject_id}: SKIPPED (folder not found: {eeg_folder})")
            continue

        try:
            raw, events, sfreq_eeg, eeg_mnt_struct = pp.load_eeg_data(eeg_folder)
            raw_before, filtered_pre_zscore, pyprep_bads, ica_excluded, eog_threshold = \
                clean_eeg_pre_normalization(raw, subject_id)

            metrics = compute_qc_metrics(
                subject_id, raw_before, filtered_pre_zscore, pyprep_bads, ica_excluded,
                eog_threshold, args.var_ratio_threshold, args.retention_flag_threshold,
            )

            sub_out_dir = os.path.join(pp.OUTPUT_PATH, subject_id)
            os.makedirs(sub_out_dir, exist_ok=True)
            out_path = os.path.join(sub_out_dir, f"{subject_id}_qc_metrics.json")
            with open(out_path, "w") as f:
                json.dump(metrics, f, indent=2)

            print_summary_row(metrics)
            all_metrics.append(metrics)

        except Exception as e:
            import traceback
            print(f"{subject_id}: ERROR ({e})")
            traceback.print_exc()

    print("-" * 110)
    print(f"Done. {len(all_metrics)}/{len(SUBJECT_NUMS)} subjects processed successfully.")
    print(f"Per-subject JSON written to: {pp.OUTPUT_PATH}\\{{subject}}\\{{subject}}_qc_metrics.json")


if __name__ == "__main__":
    main()
