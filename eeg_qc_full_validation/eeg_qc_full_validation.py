# ============================================================================
# eeg_qc_full_validation.py -- One-stop QC + IC-override A/B test, all subjects
# ============================================================================
# WHAT THIS SCRIPT DOES (all in a single run, per subject):
#
#   1. Runs the pipeline's OWN cleaning chain (filter -> PyPREP -> CAR -> ICA
#      with automatic + your existing manual overrides) -- this is the
#      "BASELINE" -- reusing the exact same functions from
#      Preproc_tsfnet_nback_badch.py that eeg_qc_metrics.py used.
#   2. If you pass --test-ic-overrides for a subject (e.g. to test "what if
#      IC3 were also excluded for VP001"), it ALSO runs a second "TEST" clean
#      with that extra component force-excluded, so you get a true A/B
#      comparison, not just a hypothesis.
#   3. For EACH variant (baseline, and test if requested) it saves the two
#      key visual QC plots side-by-side-comparable:
#        {subject}_eeg_after_allchannels[_test].png   (full-duration stacked traces)
#        {subject}_psd_before_after[_test].png         (PSD RAW vs PREPROCESSED)
#      into a dedicated QC_VALIDATION output folder, so you can open BASELINE
#      and TEST side by side and see with your own eyes whether the burst
#      actually goes away.
#   4. Computes the FULL metric suite from eeg_qc_metrics.py v1-v5 (variance
#      ratio, still-bad channels, EMG residual, 3-stage retention breakdown,
#      per-IC and all-IC diagnostics, burst-window ranking) for both variants,
#      PLUS three literature-standard metrics that were missing before (see
#      "NEW METRICS" section below), and writes ONE combined JSON per subject
#      with both variants side by side for direct numeric comparison.
#
# NEW METRICS ADDED (based on a literature check of standard EEG QC practice
# -- PREP pipeline, the "CLEAN" cross-site pipeline, Automagic, and the
# "EEG is better left alone" over-cleaning study):
#   - var_art_pct            : "Percentage of Total Variance Explained by
#                               Artifacts" (the CLEAN pipeline's own term) --
#                               the % of total EEG variance accounted for by
#                               the EXCLUDED ICA components combined. A very
#                               low VarArt with a big visible leftover
#                               artifact (like VP001's burst) is itself a
#                               red flag: it means whatever component(s) ARE
#                               carrying the artifact were NOT excluded.
#   - kurtosis_after          : per-channel kurtosis of the cleaned signal.
#                               Real EEG has near-Gaussian kurtosis (~0);
#                               residual artifacts (spikes, muscle bursts)
#                               show up as strongly non-Gaussian (high
#                               positive) kurtosis on the channels they hit --
#                               this is a classic PREP/Automagic-style bad
#                               channel/artifact indicator applied HERE to the
#                               already-cleaned signal, i.e. "what's still
#                               non-Gaussian after the pipeline claims to be
#                               done".
#   - epoch_high_amp_reject_pct : the standard "epoch rejection rate" QC
#                               metric (a common benchmark in EEG pipelines is
#                               ~5-15% of epochs flagged) computed post-hoc by
#                               splitting the cleaned signal into 2s epochs
#                               and flagging any epoch where peak-to-peak
#                               amplitude on any channel exceeds
#                               --epoch-reject-uv (default 150 uV, a standard
#                               clinical/ERP threshold). A subject whose
#                               cleaned signal STILL has a much-higher-than-
#                               normal rejection rate has a leftover-artifact
#                               problem, independent of any single metric
#                               above.
#   - line_noise_ratio_after  : residual 49-51Hz power relative to the
#                               49-51Hz band's immediate neighbors (47-49 +
#                               51-53Hz). The CLEAN pipeline reports this
#                               explicitly and calls ~1.0 "optimal" (no
#                               leftover line-noise peak).
#
# USAGE:
#   # Just validate current pipeline behavior for subjects 1-5 (no A/B test):
#   python eeg_qc_full_validation.py --subjects 1,2,3,4,5
#
#   # A/B test: also try excluding IC3 for VP001 and IC5 for VP002, and score
#   # both against a known burst window per subject:
#   python eeg_qc_full_validation.py --subjects 1,2 \
#       --test-ic-overrides "1:3;2:5" \
#       --burst-windows "1:750,950"
#
# OUTPUT (per subject):
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_eeg_after_allchannels.png
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_psd_before_after.png
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_eeg_after_allchannels_test.png   (if tested)
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_psd_before_after_test.png       (if tested)
#   {OUTPUT_PATH}/QC_VALIDATION/{subject}/{subject}_full_validation.json
#
# NOTE: this script only READS your dataset and the pipeline module's
# functions/constants, and WRITES ONLY into the separate QC_VALIDATION
# subfolder -- it never touches your production {subject}_eeg_clean_raw.fif
# or production QC plots.
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
from scipy.stats import kurtosis as scipy_kurtosis

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--subjects", type=str, required=True,
                     help="Comma-separated subject numbers, e.g. '1,2' or '1,2,3,4,5'.")
parser.add_argument("--pipeline-script", type=str,
                     default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           "Preproc_tsfnet_nback_badch.py"),
                     help="Path to Preproc_tsfnet_nback_badch.py.")
parser.add_argument("--var-ratio-threshold", type=float, default=4.0)
parser.add_argument("--retention-flag-threshold", type=float, default=0.5)
parser.add_argument("--epoch-reject-uv", type=float, default=150.0,
                     help="Peak-to-peak threshold (uV) per 2s epoch for the epoch_high_amp_reject_pct "
                          "metric (default 150uV, a standard ERP/clinical rejection threshold).")
parser.add_argument("--test-ic-overrides", type=str, default=None,
                     help="Optional 'subjectnum:ic1,ic2;subjectnum2:ic3' -- extra ICA component "
                          "indices to FORCE-EXCLUDE on top of automatic detection, for A/B testing "
                          "a hypothesis (e.g. '1:3' tests excluding IC3 for subject 1/VP001).")
parser.add_argument("--burst-windows", type=str, default=None,
                     help="Optional 'subjectnum:start,end;subjectnum2:start,end' in seconds -- a "
                          "visually-identified artifact burst per subject, used to rank ALL ICA "
                          "components by in-burst vs. out-of-burst activity (see eeg_qc_metrics.py v5).")
args = parser.parse_args()

SUBJECT_NUMS = [int(s.strip()) for s in args.subjects.split(",")]


def _parse_per_subject_map(spec, value_parser):
    """Parses 'sub1:val;sub2:val' into {sub_num: parsed_value}. Returns {} if spec is None."""
    result = {}
    if not spec:
        return result
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        sub_str, val_str = chunk.split(":")
        result[int(sub_str.strip())] = value_parser(val_str.strip())
    return result


TEST_IC_OVERRIDES = _parse_per_subject_map(
    args.test_ic_overrides, lambda v: [int(x.strip()) for x in v.split(",")]
)
BURST_WINDOWS = _parse_per_subject_map(
    args.burst_windows, lambda v: tuple(float(x.strip()) for x in v.split(","))
)

if not os.path.isfile(args.pipeline_script):
    sys.exit(f"ERROR: pipeline script not found at {args.pipeline_script!r}. "
              f"Pass --pipeline-script <full path to Preproc_tsfnet_nback_badch.py>.")

_spec = importlib.util.spec_from_file_location("tsfnet_pipeline", args.pipeline_script)
pp = importlib.util.module_from_spec(_spec)
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(pp)

print(f"[eeg_qc_full_validation] Loaded pipeline module from: {args.pipeline_script}")
print(f"[eeg_qc_full_validation] BASE_DATASET_PATH = {pp.BASE_DATASET_PATH}")
QC_VALIDATION_ROOT = os.path.join(pp.OUTPUT_PATH, "QC_VALIDATION")


# ----------------------------------------------------------------------------
# Reusable plotting (mirrors the pipeline's own QC plot style so BASELINE and
# TEST images -- and the production ones you already reviewed -- are directly
# visually comparable). _plot_psd_before_after / save_psd_qc_plots are already
# module-level in the pipeline and are reused as-is; the stacked all-channels
# plot is nested inside preprocess_eeg() in the pipeline so it isn't
# importable, and is reproduced here to match.
# ----------------------------------------------------------------------------
def plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
    raw_eeg_only = raw_obj.copy().pick(picks="eeg")
    sfreq_plot = raw_eeg_only.info["sfreq"]
    if tmin is not None or tmax is not None:
        start_sample = int((tmin or 0.0) * sfreq_plot)
        stop_sample = int(tmax * sfreq_plot) if tmax is not None else None
        data = raw_eeg_only.get_data(start=start_sample, stop=stop_sample)
        t = (np.arange(data.shape[1]) + start_sample) / sfreq_plot
    else:
        data = raw_eeg_only.get_data()
        t = np.arange(data.shape[1]) / sfreq_plot
    ch_names_plot = raw_eeg_only.ch_names
    spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 30 + 1e-12
    n_ch_plot = data.shape[0]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * n_ch_plot)))
    for i in range(n_ch_plot):
        offset = (n_ch_plot - 1 - i) * spread
        ax.plot(t, data[i] + offset, color=color, linewidth=0.4)
    ax.set_yticks([(n_ch_plot - 1 - i) * spread for i in range(n_ch_plot)])
    ax.set_yticklabels(ch_names_plot, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def save_validation_plots(raw_before, filtered, subject_id, out_dir, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    fig_after = plot_all_channels_stacked(
        filtered, f"{subject_id} - EEG PREPROCESSED{' [TEST]' if suffix else ''} - All Channels", "tab:blue")
    fig_after.savefig(os.path.join(out_dir, f"{subject_id}_eeg_after_allchannels{suffix}.png"), dpi=150)
    plt.close(fig_after)

    fig_psd = pp._plot_psd_before_after(raw_before, filtered, subject_id)
    fig_psd.savefig(os.path.join(out_dir, f"{subject_id}_psd_before_after{suffix}.png"), dpi=150)
    plt.close(fig_psd)


# ----------------------------------------------------------------------------
# NEW literature-standard metrics (see header docstring for sources/rationale)
# ----------------------------------------------------------------------------
def compute_kurtosis_after(filtered, picks_eeg):
    data = filtered.get_data(picks=picks_eeg)
    ch_names = [filtered.ch_names[i] for i in picks_eeg]
    k = scipy_kurtosis(data, axis=1, fisher=True, bias=False)
    return dict(zip(ch_names, [float(x) for x in k]))


def compute_epoch_high_amp_reject_pct(filtered, picks_eeg, threshold_uv, epoch_sec=2.0):
    data = filtered.get_data(picks=picks_eeg) * 1e6  # -> uV
    sfreq = filtered.info["sfreq"]
    win_samp = int(round(epoch_sec * sfreq))
    n_epochs = data.shape[1] // win_samp
    if n_epochs == 0:
        return None
    trimmed = data[:, : n_epochs * win_samp].reshape(data.shape[0], n_epochs, win_samp)
    ptp_per_epoch = trimmed.max(axis=2) - trimmed.min(axis=2)  # (n_ch, n_epochs)
    bad_epochs = (ptp_per_epoch > threshold_uv).any(axis=0)  # epoch is bad if ANY channel exceeds threshold
    return float(100.0 * bad_epochs.sum() / n_epochs)


def compute_line_noise_ratio(filtered, picks_eeg, line_freq=50.0):
    data = filtered.get_data(picks=picks_eeg)
    sfreq = filtered.info["sfreq"]
    freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 4), data.shape[1]), axis=1)
    line_mask = (freqs >= line_freq - 1) & (freqs <= line_freq + 1)
    neighbor_mask = ((freqs >= line_freq - 3) & (freqs < line_freq - 1)) | \
                    ((freqs > line_freq + 1) & (freqs <= line_freq + 3))
    if not line_mask.any() or not neighbor_mask.any():
        return None
    line_power = psd[:, line_mask].mean()
    neighbor_power = psd[:, neighbor_mask].mean()
    if neighbor_power <= 0:
        return None
    return float(line_power / neighbor_power)


def compute_var_art_pct(ica, filtered, exclude_indices):
    """'Percentage of Total Variance Explained by Artifacts' (VarArt), as defined by the CLEAN
    pipeline: the fraction of total EEG variance accounted for by the EXCLUDED components,
    computed jointly (this IS the correct use of get_explained_variance_ratio's joint mode --
    unlike the per-IC diagnostics elsewhere, VarArt is specifically about the pooled/joint total)."""
    if not exclude_indices:
        return 0.0
    try:
        explained = ica.get_explained_variance_ratio(filtered, components=list(exclude_indices), ch_type="eeg")
        return float(explained.get("eeg", 0.0)) * 100.0
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Re-run filter -> PyPREP -> CAR -> ICA(+overrides [+extra test excludes]),
# reusing the identical building blocks from the pipeline module. Returns
# everything needed for both plotting and the full metric suite.
# ----------------------------------------------------------------------------
def run_clean_variant(raw, subject_id, extra_exclude=None, burst_window=None):
    extra_exclude = set(extra_exclude or [])
    raw_before = raw.copy()
    filtered = raw.copy()

    if pp.APPLY_NOTCH:
        filtered.notch_filter(freqs=pp.EEG_NOTCH_FREQ, picks="eeg", verbose=False)
    filtered.filter(
        l_freq=pp.EEG_BP_LOW, h_freq=pp.EEG_BP_HIGH, method="iir",
        iir_params=dict(order=pp.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )

    pyprep_bads = []
    if pp._PYPREP_AVAILABLE:
        try:
            nd = pp.NoisyChannels(filtered, random_state=42)
            try:
                nd.find_all_bads(ransac=True)
            except Exception:
                nd.find_all_bads(ransac=False)
            pyprep_bads = nd.get_bads()
            filtered.info["bads"] = pyprep_bads
            if pyprep_bads:
                filtered.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:
            print(f"  {subject_id}: PyPREP step FAILED ({e}).")

    filtered.set_eeg_reference("average", projection=False, verbose=False)
    post_filter_car = filtered.copy()

    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard", fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    eog_indices = []
    eog_scores_by_ch = {}
    eog_threshold = pp._get_subject_eog_threshold(subject_id)
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, scores = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=eog_threshold, verbose=False)
                eog_indices.extend(idx)
                eog_scores_by_ch[eog_ch] = [float(s) for s in np.asarray(scores).ravel()]
            except Exception:
                pass

    try:
        ic_labels = pp.mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        ic_labels = None
        iclabel_ocular = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    exclude_set, override_logs = pp._apply_manual_ic_overrides(subject_id, set(ica.exclude))
    # Layer the TEST override on top of the pipeline's normal automatic+manual selection, so the
    # comparison is "current behavior" vs. "current behavior + this one extra component".
    exclude_set = exclude_set | extra_exclude
    ica.exclude = sorted(exclude_set)

    source_raw = ica.get_sources(post_filter_car)
    burst_ratio = None
    if burst_window is not None:
        sfreq = source_raw.info["sfreq"]
        start_samp = max(0, int(round(burst_window[0] * sfreq)))
        end_samp = min(source_raw.n_times, int(round(burst_window[1] * sfreq)))
        data = source_raw.get_data()
        inside = data[:, start_samp:end_samp]
        outside_mask = np.ones(source_raw.n_times, dtype=bool)
        outside_mask[start_samp:end_samp] = False
        outside = data[:, outside_mask]
        rms_in = np.sqrt((inside ** 2).mean(axis=1))
        rms_out = np.sqrt((outside ** 2).mean(axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            burst_ratio = rms_in / np.maximum(rms_out, 1e-20)

    all_components_diagnostics = []
    for ic_idx in range(ica.n_components_):
        try:
            ev = ica.get_explained_variance_ratio(filtered, components=[ic_idx], ch_type="eeg")
            ev_ic = float(ev.get("eeg", float("nan")))
        except Exception:
            ev_ic = None
        all_components_diagnostics.append({
            "ic_index": int(ic_idx),
            "explained_variance_ratio_eeg": ev_ic,
            "eog_correlation": {ch: (float(s[ic_idx]) if ic_idx < len(s) else None)
                                for ch, s in eog_scores_by_ch.items()},
            "iclabel_label": ic_labels["labels"][ic_idx] if ic_labels is not None else None,
            "iclabel_probability": (float(ic_labels["y_pred_proba"][ic_idx])
                                     if ic_labels is not None and "y_pred_proba" in ic_labels else None),
            "burst_vs_rest_rms_ratio": (float(burst_ratio[ic_idx])
                                        if burst_ratio is not None and np.isfinite(burst_ratio[ic_idx]) else None),
            "final_excluded": ic_idx in exclude_set,
            "excluded_via_test_override": ic_idx in extra_exclude,
        })

    var_art_pct = compute_var_art_pct(ica, filtered, exclude_set)

    if ica.exclude:
        ica.apply(filtered, verbose=False)

    pyprep_bads_plain = [str(x) for x in pyprep_bads]
    ica_excluded_plain = [int(x) for x in ica.exclude]

    return {
        "raw_before": raw_before,
        "post_filter_car": post_filter_car,
        "filtered": filtered,
        "pyprep_bads": pyprep_bads_plain,
        "ica_excluded": ica_excluded_plain,
        "all_components_diagnostics": all_components_diagnostics,
        "var_art_pct": var_art_pct,
        "picks_eeg": picks_eeg,
    }


# ----------------------------------------------------------------------------
# Robust epoch-median band power (reused from eeg_qc_metrics.py v2/v3 logic)
# ----------------------------------------------------------------------------
def band_power_robust_median(raw_obj, fmin, fmax, picks_eeg, epoch_sec=2.0):
    data = raw_obj.get_data(picks=picks_eeg)
    sfreq = raw_obj.info["sfreq"]
    n_ch, n_times = data.shape
    win_samp = int(round(epoch_sec * sfreq))
    n_epochs = n_times // win_samp
    if n_epochs < 3:
        freqs, psd = welch(data, fs=sfreq, nperseg=min(int(sfreq * 4), n_times), axis=1)
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        return psd[:, band_mask].mean(axis=1) * (1e6 ** 2)
    trimmed = data[:, : n_epochs * win_samp].reshape(n_ch, n_epochs, win_samp)
    nperseg = min(int(sfreq * 1.0), win_samp)
    freqs, psd = welch(trimmed, fs=sfreq, nperseg=nperseg, axis=2)
    band_mask = (freqs >= fmin) & (freqs <= fmax)
    per_epoch = psd[:, :, band_mask].mean(axis=2) * (1e6 ** 2)
    return np.median(per_epoch, axis=1)


def summarize_variant(subject_id, variant_name, result, raw_before, eog_threshold,
                       var_ratio_threshold, retention_flag_threshold, epoch_reject_uv, burst_window):
    filtered = result["filtered"]
    post_filter_car = result["post_filter_car"]
    picks_eeg = result["picks_eeg"]
    ch_names = [filtered.ch_names[i] for i in picks_eeg]
    pyprep_bads = result["pyprep_bads"]

    var_after = filtered.get_data(picks=picks_eeg).var(axis=1)
    median_var_after = float(np.median(var_after))
    max_var_ratio = float(var_after.max() / median_var_after) if median_var_after > 0 else float("nan")
    still_bad = [ch_names[i] for i in range(len(ch_names))
                 if median_var_after > 0 and (var_after[i] / median_var_after) > var_ratio_threshold]

    eeg_uv2_raw = band_power_robust_median(raw_before, 4.0, 30.0, picks_eeg)
    eeg_uv2_car = band_power_robust_median(post_filter_car, 4.0, 30.0, picks_eeg)
    eeg_uv2_ica = band_power_robust_median(filtered, 4.0, 30.0, picks_eeg)
    ratio_car_stage = eeg_uv2_car / np.maximum(eeg_uv2_raw, 1e-20)
    ratio_ica_stage = eeg_uv2_ica / np.maximum(eeg_uv2_car, 1e-20)
    non_interp_mask = np.array([ch not in pyprep_bads for ch in ch_names])

    kurt = compute_kurtosis_after(filtered, picks_eeg)
    epoch_reject_pct = compute_epoch_high_amp_reject_pct(filtered, picks_eeg, epoch_reject_uv)
    line_noise_ratio = compute_line_noise_ratio(filtered, picks_eeg)

    ranked_by_burst = []
    if burst_window is not None:
        ranked_by_burst = sorted(
            [e for e in result["all_components_diagnostics"] if e["burst_vs_rest_rms_ratio"] is not None],
            key=lambda e: e["burst_vs_rest_rms_ratio"], reverse=True,
        )

    return {
        "variant": variant_name,
        "pyprep_bads_detected": pyprep_bads,
        "ica_excluded_indices": result["ica_excluded"],
        "var_art_pct": result["var_art_pct"],
        "max_variance_ratio_after": max_var_ratio,
        "still_bad_after_cleaning": still_bad,
        "eeg_band_ratio_raw_to_postCAR_mean_non_interpolated": float(ratio_car_stage[non_interp_mask].mean()) if non_interp_mask.any() else None,
        "eeg_band_ratio_postCAR_to_postICA_mean_non_interpolated": float(ratio_ica_stage[non_interp_mask].mean()) if non_interp_mask.any() else None,
        "kurtosis_after": kurt,
        "kurtosis_after_max_channel": max(kurt, key=kurt.get),
        "kurtosis_after_max_value": kurt[max(kurt, key=kurt.get)],
        "epoch_high_amp_reject_pct": epoch_reject_pct,
        "line_noise_ratio_after": line_noise_ratio,
        "all_components_diagnostics": result["all_components_diagnostics"],
        "all_components_ranked_by_burst_ratio": ranked_by_burst,
    }


def print_variant_summary(subject_id, s):
    print(
        f"  [{s['variant']:>8}] max_var_ratio={s['max_variance_ratio_after']:.2f} | "
        f"VarArt={s['var_art_pct']:.1f}% | raw->CAR={s['eeg_band_ratio_raw_to_postCAR_mean_non_interpolated']:.3f} | "
        f"CAR->ICA={s['eeg_band_ratio_postCAR_to_postICA_mean_non_interpolated']:.3f} | "
        f"epoch_reject={s['epoch_high_amp_reject_pct']:.1f}% | "
        f"line_noise_ratio={s['line_noise_ratio_after']:.2f} | "
        f"max_kurtosis={s['kurtosis_after_max_channel']}={s['kurtosis_after_max_value']:.1f} | "
        f"still_bad={s['still_bad_after_cleaning']}"
    )
    if s["all_components_ranked_by_burst_ratio"]:
        top = s["all_components_ranked_by_burst_ratio"][0]
        print(f"    top burst suspect: IC{top['ic_index']} ratio={top['burst_vs_rest_rms_ratio']:.2f}x "
              f"label={top['iclabel_label']} excluded={top['final_excluded']}")


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
            eog_threshold = pp._get_subject_eog_threshold(subject_id)
            burst_window = BURST_WINDOWS.get(sub_num)
            out_dir = os.path.join(QC_VALIDATION_ROOT, subject_id)

            baseline = run_clean_variant(raw, subject_id, extra_exclude=None, burst_window=burst_window)
            save_validation_plots(baseline["raw_before"], baseline["filtered"], subject_id, out_dir, suffix="")
            baseline_summary = summarize_variant(
                subject_id, "baseline", baseline, baseline["raw_before"], eog_threshold,
                args.var_ratio_threshold, args.retention_flag_threshold, args.epoch_reject_uv, burst_window,
            )
            print_variant_summary(subject_id, baseline_summary)

            combined = {"subject_id": subject_id, "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                        "burst_window_seconds": list(burst_window) if burst_window else None,
                        "baseline": baseline_summary, "test": None}

            if sub_num in TEST_IC_OVERRIDES:
                extra = TEST_IC_OVERRIDES[sub_num]
                test = run_clean_variant(raw, subject_id, extra_exclude=extra, burst_window=burst_window)
                save_validation_plots(test["raw_before"], test["filtered"], subject_id, out_dir, suffix="_test")
                test_summary = summarize_variant(
                    subject_id, f"test(+IC{extra})", test, test["raw_before"], eog_threshold,
                    args.var_ratio_threshold, args.retention_flag_threshold, args.epoch_reject_uv, burst_window,
                )
                print_variant_summary(subject_id, test_summary)
                combined["test"] = test_summary
                combined["test_extra_excluded_ics"] = extra

            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{subject_id}_full_validation.json")
            with open(out_path, "w") as f:
                json.dump(combined, f, indent=2)
            print(f"  -> {out_path}")
            print(f"  -> plots in {out_dir}")

        except Exception as e:
            import traceback
            print(f"{subject_id}: ERROR ({e})")
            traceback.print_exc()

    print(f"\nDone. Output root: {QC_VALIDATION_ROOT}")


if __name__ == "__main__":
    main()
