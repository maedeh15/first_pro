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
# v2 UPDATE (retention-ratio robustness fixes, per VP001/VP002 numeric
# results): the whole-recording Welch retention ratio in v1 was dragged down
# by large transient artifact bursts present in the RAW signal (e.g. VP001's
# ~750-950s burst) inflating the "before" baseline, which made post-cleaning
# retention look artificially low even on channels PyPREP/CAR/ICA cleaned
# correctly. Three fixes, all additive (nothing removed, so v1 fields are
# still present for continuity):
#   1. eeg_band_retention_ratio_robust : band power computed per-channel as
#      the MEDIAN over non-overlapping 2s epochs (both before and after),
#      instead of one Welch average over the full recording. A few
#      artifact-contaminated epochs no longer dominate the "before" estimate.
#   2. Channels PyPREP itself interpolated (pyprep_bads_detected) are
#      EXCLUDED from eeg_band_retention_ratio_mean_non_interpolated and from
#      low_retention_channels, since "before" vs "after" isn't a fair
#      apples-to-apples comparison for a channel whose "after" signal is a
#      spatial interpolation of its neighbors, not a cleaned version of
#      itself.
#   3. Both the naive whole-recording metric (renamed *_naive_wholerecording
#      for clarity) and the new robust epoch-median metric are kept side by
#      side in the JSON, so a large gap between them is itself diagnostic
#      (it means a burst artifact, not gradual over-cleaning, explains most
#      of the naive number).
#
# v3 UPDATE (three-checkpoint breakdown, per VP001/VP002 v2 results): the v2
# robust retention ratio stayed low (~0.3) even after fixing the burst-bias
# issue, so v1/v2's raw-vs-final comparison was CONFLATING two very different
# effects: (a) CAR/re-referencing, which is EXPECTED to remove a large,
# healthy chunk of common-mode power shared across all channels, and (b) ICA,
# which should only remove ocular-artifact components and ideally leaves
# in-band brain power close to unchanged. A snapshot is now taken right after
# filter+PyPREP+CAR but BEFORE ICA (`post_filter_car`), so the pipeline can
# report each stage's power ratio separately:
#   - eeg_band_ratio_raw_to_postCAR       : raw -> post-CAR (expected LOW, healthy)
#   - eeg_band_ratio_postCAR_to_postICA   : post-CAR -> post-ICA (expected ~1;
#                                            THIS is the one actually worth
#                                            investigating if it's low, since a
#                                            low value here means ICA is
#                                            removing more than eye-movement
#                                            components)
#   - eeg_band_4_30hz_retention_ratio     : raw -> final (= product of the two
#                                            stages above; kept for continuity)
# `eeg_band_low_retention_channels_ica_stage_non_interpolated` flags channels
# where the ICA STAGE specifically drops below the retention threshold --
# this is the field to look at now, not the raw-to-final one.
#
# v4 UPDATE (per-IC diagnostics, per VP001's suspicious broad ICA-stage
# power drop across C3/Cz/CP5/CP1/CP2/CP6 -- non-frontal channels an ocular
# component shouldn't touch): for every EXCLUDED ICA component, `ic_diagnostics`
# now reports:
#   - explained_variance_ratio_eeg : how much of the total EEG variance this
#     component alone accounts for (from ica.get_explained_variance_ratio()).
#   - eog_correlation               : correlation score vs each EOG channel
#     from ica.find_bads_eog() (weak correlation despite being excluded is a
#     red flag -- it means the component was NOT selected primarily because
#     it looks like an eye movement/blink).
#   - iclabel_label / iclabel_probability : ICLabel's own classification and
#     confidence for this component.
#   - selected_by_eog_correlation / selected_by_iclabel / selected_by_manual_override :
#     which mechanism(s) actually flagged this component for removal, so a
#     component excluded ONLY via a weak automatic path (or a stale manual
#     override) is easy to spot.
# The printed summary line now also raises a WEAK-EOG-IC<n> flag for any
# excluded, non-manually-overridden component whose EOG correlation is
# < 0.3 on every EOG channel -- a candidate for `MANUAL_IC_OVERRIDES` review.
#
# v5 UPDATE (all-component + burst-window diagnostics, per direct visual
# re-inspection of VP001_eeg_after_allchannels.png and VP001_psd_before_after.png:
# a large multi-channel burst around 750-950s survives the FULL pipeline --
# attenuated, not removed -- and the post-cleaning PSD shows uneven,
# broadband residual noise across channels). v4 only diagnosed components
# ICA had ALREADY decided to exclude, which can't explain "why did an
# artifact survive" -- that requires looking at the components that were
# KEPT. This update:
#   1. Runs diagnostics (explained variance, EOG correlation, ICLabel) on
#      ALL fitted components, not just the excluded ones -- see
#      `all_components_diagnostics`.
#   2. Accepts an optional --burst-window "start,end" (seconds) CLI arg. When
#      given, every component's source time course (via ica.get_sources()) is
#      scored by `burst_vs_rest_rms_ratio` = RMS activity INSIDE the burst
#      window / RMS activity OUTSIDE it. A ratio >> 1 means that component is
#      disproportionately active during exactly the suspect window --
#      `all_components_ranked_by_burst_ratio` sorts all components by this
#      score, and the printed summary flags the top suspect if it scores
#      > 3x AND was NOT excluded by the pipeline (i.e. a real candidate to
#      add to MANUAL_IC_OVERRIDES).
# Usage: python eeg_qc_metrics.py --subjects 1 --burst-window 750,950
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
parser.add_argument("--burst-window", type=str, default=None,
                     help="Optional 'start,end' in seconds (e.g. '750,950') marking a visually-identified "
                          "artifact burst in the RAW recording. When given, ALL ICA components (not just "
                          "the excluded ones) are additionally scored by how much MORE active they are "
                          "inside this window vs. outside it, to find which component(s) -- if any -- "
                          "actually carry the burst and were left in.")
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]

BURST_WINDOW = None
if args.burst_window:
    try:
        _start, _end = [float(x.strip()) for x in args.burst_window.split(",")]
        BURST_WINDOW = (_start, _end)
    except Exception:
        sys.exit(f"ERROR: --burst-window must be 'start,end' in seconds, e.g. '750,950'. "
                  f"Got: {args.burst_window!r}")

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


def _burst_vs_rest_ratio(source_raw, burst_window):
    """RMS activity of each ICA source component INSIDE `burst_window` (start_s, end_s) vs
    OUTSIDE it, as a ratio (>1 means the component is more active during the burst than during
    the rest of the recording -- a strong candidate for carrying the artifact)."""
    sfreq = source_raw.info["sfreq"]
    n_times = source_raw.n_times
    start_samp = max(0, int(round(burst_window[0] * sfreq)))
    end_samp = min(n_times, int(round(burst_window[1] * sfreq)))
    if start_samp >= end_samp:
        return None
    data = source_raw.get_data()  # (n_components, n_times)
    inside = data[:, start_samp:end_samp]
    outside_mask = np.ones(n_times, dtype=bool)
    outside_mask[start_samp:end_samp] = False
    outside = data[:, outside_mask]
    rms_inside = np.sqrt((inside ** 2).mean(axis=1))
    rms_outside = np.sqrt((outside ** 2).mean(axis=1)) if outside.shape[1] > 0 else np.full(data.shape[0], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rms_inside / np.maximum(rms_outside, 1e-20)
    return ratio  # (n_components,)


# ----------------------------------------------------------------------------
# Re-run filter -> PyPREP -> CAR -> ICA(+overrides), stopping BEFORE the
# z-score step. Mirrors preprocess_eeg() in the pipeline module line-for-line
# (same functions/constants imported from `pp`), MINUS the final
# normalization block, so the returned signal is directly comparable (in real
# physical volts) to raw_before for variance/power-ratio metrics.
# ----------------------------------------------------------------------------
def clean_eeg_pre_normalization(raw, subject_id, burst_window=None):
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

    # v3 CHECKPOINT: snapshot the signal HERE -- after filter+PyPREP+CAR, BEFORE ICA -- so the
    # retention-ratio metric can separate how much power drop comes from re-referencing (CAR,
    # expected/healthy: removes common-mode signal shared across all channels) vs. from ICA
    # (which should only remove ocular-artifact components; a large drop here would mean ICA is
    # over-removing real brain signal). Without this checkpoint, raw-vs-final conflates both
    # effects into one number.
    post_filter_car = filtered.copy()

    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    eog_indices = []
    eog_scores_by_ch = {}  # ch_name -> {ic_index: correlation_score}, for per-IC diagnostics below
    eog_threshold = pp._get_subject_eog_threshold(subject_id)
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, scores = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=eog_threshold, verbose=False)
                eog_indices.extend(idx)
                eog_scores_by_ch[eog_ch] = [float(s) for s in np.asarray(scores).ravel()]
            except Exception as e:
                print(f"  {subject_id}: find_bads_eog({eog_ch}) failed: {e}")

    try:
        ic_labels = pp.mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        ic_labels = None
        iclabel_ocular = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    exclude_set, override_logs = pp._apply_manual_ic_overrides(subject_id, set(ica.exclude))
    for msg in override_logs:
        print(f"  {msg}")
    ica.exclude = sorted(exclude_set)

    # --- ALL-COMPONENT diagnostics (added per your request after visually re-inspecting VP001's
    # after-cleaning plot and PSD, which show the 750-950s burst was only ATTENUATED, not removed).
    # For EVERY fitted component -- not just the ones ICA decided to exclude -- this reports
    # explained variance, ICLabel's full label + probability, EOG correlation, and (if
    # --burst-window was given) how much more active the component is DURING the burst window vs.
    # the rest of the recording. A component with a high burst ratio that was NOT excluded is the
    # prime suspect for "why does the burst survive the full pipeline" and a candidate to add to
    # MANUAL_IC_OVERRIDES for forced exclusion.
    source_raw = ica.get_sources(post_filter_car)  # source time courses for ALL components, order-preserved
    burst_ratio = _burst_vs_rest_ratio(source_raw, burst_window) if burst_window is not None else None

    all_components_diagnostics = []
    for ic_idx in range(ica.n_components_):
        try:
            explained_var = ica.get_explained_variance_ratio(filtered, components=[ic_idx], ch_type="eeg")
            explained_var_ic = float(explained_var.get("eeg", float("nan")))
        except Exception:
            explained_var_ic = None
        entry = {
            "ic_index": int(ic_idx),
            "explained_variance_ratio_eeg": explained_var_ic,
            "eog_correlation": {
                ch: (float(scores[ic_idx]) if ic_idx < len(scores) else None)
                for ch, scores in eog_scores_by_ch.items()
            },
            "iclabel_label": ic_labels["labels"][ic_idx] if ic_labels is not None else None,
            "iclabel_probability": (
                float(ic_labels["y_pred_proba"][ic_idx])
                if ic_labels is not None and "y_pred_proba" in ic_labels else None
            ),
            "burst_vs_rest_rms_ratio": (
                float(burst_ratio[ic_idx]) if burst_ratio is not None and np.isfinite(burst_ratio[ic_idx]) else None
            ),
            "selected_by_eog_correlation": ic_idx in eog_indices,
            "selected_by_iclabel": ic_idx in iclabel_ocular,
            "final_excluded": ic_idx in exclude_set,
        }
        all_components_diagnostics.append(entry)

    # Rank by burst ratio (highest = most likely to be carrying the burst artifact) so the
    # top suspects are easy to read off without scanning all ~9-24 components manually.
    if burst_window is not None:
        all_components_diagnostics_by_burst = sorted(
            [e for e in all_components_diagnostics if e["burst_vs_rest_rms_ratio"] is not None],
            key=lambda e: e["burst_vs_rest_rms_ratio"], reverse=True,
        )
    else:
        all_components_diagnostics_by_burst = []

    # --- per-IC diagnostics for every EXCLUDED component (kept from v4, now just a filtered view
    # of all_components_diagnostics above for backward-compatible field names/shape) ---
    ic_diagnostics = [dict(e, selected_by_manual_override=(
        e["ic_index"] not in (set(eog_indices) | set(iclabel_ocular)) and e["final_excluded"]
    )) for e in all_components_diagnostics if e["final_excluded"]]
    for e in ic_diagnostics:
        e.pop("final_excluded", None)

    if ica.exclude:
        ica.apply(filtered, verbose=False)

    # Cast to plain Python int/str here (not numpy int64/str_) so the metrics dict built from
    # these is safely json.dump()-able downstream without a custom encoder.
    ica_excluded_plain = [int(x) for x in ica.exclude]
    pyprep_bads_plain = [str(x) for x in pyprep_bads]

    return (raw_before, post_filter_car, filtered, pyprep_bads_plain, ica_excluded_plain,
            eog_threshold, ic_diagnostics, all_components_diagnostics, all_components_diagnostics_by_burst)


# ----------------------------------------------------------------------------
# Metric helpers
# ----------------------------------------------------------------------------
def _band_power_db(raw_obj, fmin, fmax, picks_eeg):
    """Mean PSD power (dB, 10log10 uV^2/Hz) per channel within [fmin, fmax], via a SINGLE
    Welch estimate over the whole recording. Kept for the naive/original metric -- note this
    average is NOT robust to a few large-amplitude artifact epochs dominating the estimate
    (see _band_power_db_robust_median below for the fix used in the main retention metric)."""
    data = raw_obj.get_data(picks=picks_eeg)  # (n_ch, n_times), volts
    sfreq = raw_obj.info["sfreq"]
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    # psd is in V^2/Hz; convert to uV^2/Hz for readability, matching the pipeline's own PSD plots.
    band_power_uv2 = psd[:, band_mask].mean(axis=1) * (1e6 ** 2)
    band_power_db = 10 * np.log10(np.maximum(band_power_uv2, 1e-20))
    return band_power_db, band_power_uv2


def _band_power_db_robust_median(raw_obj, fmin, fmax, picks_eeg, epoch_sec=2.0):
    """Per-channel band power (dB, 10log10 uV^2/Hz) computed as the MEDIAN of per-epoch Welch
    band-power estimates across non-overlapping `epoch_sec` windows, instead of one Welch
    average over the whole recording.

    WHY: a single whole-recording Welch average is a MEAN across the whole duration, so a
    handful of large-amplitude artifact epochs (movement bursts, remaining EMG, electrode pops)
    can dominate the estimate even though most of the recording is clean. Taking the per-epoch
    MEDIAN instead makes the estimate robust to that -- it reflects the "typical" epoch, not the
    artifact-inflated average. This matters most for the "before cleaning" baseline used in the
    retention ratio: without this fix, a channel that PyPREP/CAR/ICA cleaned correctly can look
    like it "lost most of its power" simply because the RAW baseline was inflated by a transient
    burst that isn't representative of the channel's real ongoing EEG amplitude.
    """
    data = raw_obj.get_data(picks=picks_eeg)  # (n_ch, n_times), volts
    sfreq = raw_obj.info["sfreq"]
    n_ch, n_times = data.shape
    win_samp = int(round(epoch_sec * sfreq))
    n_epochs = n_times // win_samp
    if n_epochs < 3:
        # Recording too short to epoch meaningfully -- fall back to the whole-recording estimate
        # so the function still returns something usable instead of raising.
        return _band_power_db(raw_obj, fmin, fmax, picks_eeg)

    trimmed = data[:, : n_epochs * win_samp].reshape(n_ch, n_epochs, win_samp)
    nperseg = min(int(sfreq * 1.0), win_samp)  # 1s sub-windows inside each epoch for Welch
    freqs, psd = welch(trimmed, fs=sfreq, nperseg=nperseg, axis=2)  # (n_ch, n_epochs, n_freqs)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    per_epoch_band_power_uv2 = psd[:, :, band_mask].mean(axis=2) * (1e6 ** 2)  # (n_ch, n_epochs)
    band_power_uv2 = np.median(per_epoch_band_power_uv2, axis=1)  # (n_ch,) -- robust to outlier epochs
    band_power_db = 10 * np.log10(np.maximum(band_power_uv2, 1e-20))
    return band_power_db, band_power_uv2


def compute_qc_metrics(subject_id, raw_before, post_filter_car, filtered_pre_zscore, pyprep_bads,
                        ica_excluded, ic_diagnostics, all_components_diagnostics, all_components_diagnostics_by_burst,
                        eog_threshold, var_ratio_threshold, retention_flag_threshold, burst_window):
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

    # --- EEG in-band (4-30 Hz) retention ratio: THREE-CHECKPOINT breakdown ---
    # (a) NAIVE whole-recording Welch version (v1 metric, raw -> final, kept for continuity).
    eeg_db_before, eeg_uv2_before = _band_power_db(raw_before, 4.0, 30.0, picks_eeg)
    eeg_db_after, eeg_uv2_after = _band_power_db(filtered_pre_zscore, 4.0, 30.0, picks_eeg)
    retention_ratio_naive = eeg_uv2_after / np.maximum(eeg_uv2_before, 1e-20)
    low_retention_channels_naive = [
        ch_names[i] for i in range(len(ch_names)) if retention_ratio_naive[i] < retention_flag_threshold
    ]

    # (b) ROBUST epoch-median version, computed at THREE checkpoints: raw -> post_filter_car
    #     (isolates the CAR/re-referencing effect, which is expected/healthy) -> post_ica
    #     (isolates the ICA effect, which should be small/ocular-only). Not dominated by
    #     transient artifact bursts in the raw baseline (see _band_power_db_robust_median).
    eeg_uv2_raw_r = _band_power_db_robust_median(raw_before, 4.0, 30.0, picks_eeg)[1]
    eeg_uv2_car_r = _band_power_db_robust_median(post_filter_car, 4.0, 30.0, picks_eeg)[1]
    eeg_uv2_ica_r = _band_power_db_robust_median(filtered_pre_zscore, 4.0, 30.0, picks_eeg)[1]

    # Stage-wise ratios: how much of the 4-30Hz power survives EACH step, not just the total.
    ratio_car_stage = eeg_uv2_car_r / np.maximum(eeg_uv2_raw_r, 1e-20)     # raw -> post-CAR (expected: <1, healthy)
    ratio_ica_stage = eeg_uv2_ica_r / np.maximum(eeg_uv2_car_r, 1e-20)     # post-CAR -> post-ICA (expected: ~1, only ocular removed)
    retention_ratio = eeg_uv2_ica_r / np.maximum(eeg_uv2_raw_r, 1e-20)     # raw -> final (= ratio_car_stage * ratio_ica_stage)
    low_retention_channels = [
        ch_names[i] for i in range(len(ch_names)) if retention_ratio[i] < retention_flag_threshold
    ]
    # Channels where ICA ALONE (not CAR) is responsible for most of the drop -- these are the
    # ones actually worth investigating (possible over-aggressive component removal), since a
    # low ratio_car_stage is normal re-referencing behavior, not a cleaning problem.
    ica_stage_flag_threshold = retention_flag_threshold  # same threshold, applied to the ICA-only stage
    low_retention_channels_ica_stage = [
        ch_names[i] for i in range(len(ch_names)) if ratio_ica_stage[i] < ica_stage_flag_threshold
    ]

    # (c) Exclude PyPREP-interpolated channels from the summary mean/flag list: for those
    #     channels "after" is a spatial interpolation of neighbors, not a cleaned version of the
    #     same electrode, so before/after isn't an apples-to-apples comparison. Per-channel values
    #     for ALL channels (including interpolated ones) are still kept in the full dict below --
    #     only the summary mean and the flag list are filtered.
    non_interp_mask = np.array([ch not in pyprep_bads for ch in ch_names])
    if non_interp_mask.any():
        retention_ratio_mean_non_interp = float(retention_ratio[non_interp_mask].mean())
        ratio_car_stage_mean_non_interp = float(ratio_car_stage[non_interp_mask].mean())
        ratio_ica_stage_mean_non_interp = float(ratio_ica_stage[non_interp_mask].mean())
    else:
        retention_ratio_mean_non_interp = float("nan")
        ratio_car_stage_mean_non_interp = float("nan")
        ratio_ica_stage_mean_non_interp = float("nan")
    low_retention_channels_non_interp = [
        ch for ch in low_retention_channels if ch not in pyprep_bads
    ]
    low_retention_channels_ica_stage_non_interp = [
        ch for ch in low_retention_channels_ica_stage if ch not in pyprep_bads
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
        "ic_diagnostics": ic_diagnostics,
        "burst_window_seconds": list(burst_window) if burst_window is not None else None,
        "all_components_diagnostics": all_components_diagnostics,
        "all_components_ranked_by_burst_ratio": all_components_diagnostics_by_burst,
        "max_variance_ratio_after": max_variance_ratio_after,
        "var_ratio_threshold_used": var_ratio_threshold,
        "still_bad_after_cleaning": still_bad_after_cleaning,
        "per_channel_variance_ratio_after": dict(zip(ch_names, var_ratio_per_channel)),
        "emg_band_30_49hz_db_after": dict(zip(ch_names, emg_db_after.tolist())),
        "emg_band_noisiest_channel": ch_names[noisiest_emg_idx],
        "emg_band_noisiest_channel_db": float(emg_db_after[noisiest_emg_idx]),
        # --- retention: THREE-CHECKPOINT robust epoch-median breakdown (primary metrics) ---
        "eeg_band_4_30hz_retention_ratio": dict(zip(ch_names, retention_ratio.tolist())),
        "eeg_band_retention_ratio_mean": float(retention_ratio.mean()),
        "eeg_band_retention_ratio_mean_non_interpolated": retention_ratio_mean_non_interp,
        "eeg_band_retention_flag_threshold": retention_flag_threshold,
        "eeg_band_low_retention_channels": low_retention_channels,
        "eeg_band_low_retention_channels_non_interpolated": low_retention_channels_non_interp,
        "interpolated_channels_excluded_from_retention_summary": [c for c in ch_names if c in pyprep_bads],
        # stage (a): raw -> post-filter+PyPREP+CAR (expected <1, healthy -- common-mode removal)
        "eeg_band_ratio_raw_to_postCAR": dict(zip(ch_names, ratio_car_stage.tolist())),
        "eeg_band_ratio_raw_to_postCAR_mean_non_interpolated": ratio_car_stage_mean_non_interp,
        # stage (b): post-CAR -> post-ICA (expected ~1, only ocular components should be removed
        # here -- this is the stage actually worth investigating if it's low)
        "eeg_band_ratio_postCAR_to_postICA": dict(zip(ch_names, ratio_ica_stage.tolist())),
        "eeg_band_ratio_postCAR_to_postICA_mean_non_interpolated": ratio_ica_stage_mean_non_interp,
        "eeg_band_low_retention_channels_ica_stage_non_interpolated": low_retention_channels_ica_stage_non_interp,
        # --- retention: NAIVE whole-recording version (v1 metric, kept for comparison only --
        #     a large gap vs. the robust numbers above means a transient burst artifact, not
        #     real over-cleaning, explains most of the naive drop) ---
        "eeg_band_4_30hz_retention_ratio_naive_wholerecording": dict(zip(ch_names, retention_ratio_naive.tolist())),
        "eeg_band_retention_ratio_mean_naive_wholerecording": float(retention_ratio_naive.mean()),
        "eeg_band_low_retention_channels_naive_wholerecording": low_retention_channels_naive,
    }
    return metrics


def print_summary_row(metrics):
    flags = []
    if metrics["still_bad_after_cleaning"]:
        flags.append(f"STILL-BAD: {metrics['still_bad_after_cleaning']}")
    if metrics["eeg_band_low_retention_channels_ica_stage_non_interpolated"]:
        flags.append(f"ICA-STAGE-DROP: {metrics['eeg_band_low_retention_channels_ica_stage_non_interpolated']}")
    for ic in metrics["ic_diagnostics"]:
        weak_eog = all(
            (v is None or abs(v) < 0.3) for v in ic["eog_correlation"].values()
        ) if ic["eog_correlation"] else True
        if weak_eog and not ic["selected_by_manual_override"]:
            flags.append(
                f"WEAK-EOG-IC{ic['ic_index']}: eog_corr={ic['eog_correlation']}, "
                f"iclabel={ic['iclabel_label']}({ic['iclabel_probability']})"
            )
    if metrics["all_components_ranked_by_burst_ratio"]:
        top = metrics["all_components_ranked_by_burst_ratio"][0]
        if top["burst_vs_rest_rms_ratio"] and top["burst_vs_rest_rms_ratio"] > 3.0 and not top["final_excluded"]:
            flags.append(
                f"BURST-SUSPECT-IC{top['ic_index']} (NOT excluded): "
                f"burst_ratio={top['burst_vs_rest_rms_ratio']:.1f}x, label={top['iclabel_label']}"
            )
    flag_str = "  <-- FLAG (" + "; ".join(flags) + ")" if flags else "  OK"
    print(
        f"{metrics['subject_id']:>6} | "
        f"max_var_ratio={metrics['max_variance_ratio_after']:6.2f} | "
        f"emg_noisiest={metrics['emg_band_noisiest_channel']:>6} "
        f"({metrics['emg_band_noisiest_channel_db']:6.2f} dB) | "
        f"raw->CAR={metrics['eeg_band_ratio_raw_to_postCAR_mean_non_interpolated']:5.3f} | "
        f"CAR->ICA={metrics['eeg_band_ratio_postCAR_to_postICA_mean_non_interpolated']:5.3f} | "
        f"total(robust)={metrics['eeg_band_retention_ratio_mean_non_interpolated']:5.3f} | "
        f"pyprep_bads={metrics['pyprep_bads_detected']}"
        f"{flag_str}"
    )
    if metrics["all_components_ranked_by_burst_ratio"]:
        print("  Top burst-window suspects (all components, ranked by RMS-inside/RMS-outside burst window):")
        for e in metrics["all_components_ranked_by_burst_ratio"][:5]:
            print(
                f"    IC{e['ic_index']:>2} | burst_ratio={e['burst_vs_rest_rms_ratio']:6.2f}x | "
                f"explained_var={e['explained_variance_ratio_eeg']} | "
                f"iclabel={e['iclabel_label']}({e['iclabel_probability']}) | "
                f"excluded={e['final_excluded']}"
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
            (raw_before, post_filter_car, filtered_pre_zscore, pyprep_bads, ica_excluded,
             eog_threshold, ic_diagnostics, all_components_diagnostics, all_components_diagnostics_by_burst) = \
                clean_eeg_pre_normalization(raw, subject_id, burst_window=BURST_WINDOW)

            metrics = compute_qc_metrics(
                subject_id, raw_before, post_filter_car, filtered_pre_zscore, pyprep_bads, ica_excluded,
                ic_diagnostics, all_components_diagnostics, all_components_diagnostics_by_burst,
                eog_threshold, args.var_ratio_threshold, args.retention_flag_threshold, BURST_WINDOW,
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
