# ============================================================================
# EEG FULL-DATASET QC SCAN — runs your VALIDATED final preprocessing method
# (C1: filter 0.5-50Hz -> PyPREP bad-channel interpolation -> CAR -> ICA
# [find_bads_eog + ICLabel-argmax ocular only] -> epoch-level rejection
# diagnostic) on ALL 26 subjects, and produces:
#
#   1) The same 5 QC images per subject you've been reviewing manually
#      (before/after, zoomed, all-channels) -- unchanged in content, just
#      run once for every subject instead of one at a time.
#   2) A per-subject metrics JSON (same fields as the earlier ablation
#      scripts: bad_channels_detected, still_bad_after_cleaning,
#      autoreject_reject_fraction, eeg_band_retention, etc.)
#   3) all_subjects_metrics.csv -- one row per subject, all 26.
#   4) flagged_subjects_report.txt -- a PRIORITIZED list of subjects that
#      need your visual attention, sorted by severity, so you review the
#      handful of problem subjects first instead of blindly opening 26
#      folders of images. A subject is flagged if ANY of:
#        - still_bad_after_cleaning is non-empty (a channel behaves like an
#          unresolved bad channel even after PyPREP+CAR+ICA -- this is
#          exactly the VP025 Fp1/Fp2 pattern)
#        - autoreject_reject_fraction is high (>0.30 by default -- more
#          than ~30% of 3s pseudo-epochs still look artifactual after
#          cleaning, a sign of pervasive residual noise)
#        - max_variance_ratio_after is unusually high (>6.0 by default --
#          one channel still dominates the others)
#      Subjects that flag on NONE of these are listed separately at the
#      bottom as "clean, no manual review needed" so you know at a glance
#      how many of the 26 actually require your attention.
#
# WHAT THIS SCRIPT DOES NOT DO: it does not decide FOR you what to do with
# a flagged subject (interpolate a specific channel, etc.) -- that decision
# still needs your visual review of that subject's QC images, exactly like
# we did together for VP001/015/018/021/025/005. This script's only job is
# to make sure you spend that manual-review time on the RIGHT subjects
# first, instead of scanning all 26 blindly.
#
# HOW TO RUN:
#   python eeg_full_dataset_qc_scan.py
#   python eeg_full_dataset_qc_scan.py --subjects 1-26 --workers 2
#   python eeg_full_dataset_qc_scan.py --subjects 1,2,3,25,26 --workers 1
#
# REQUIREMENTS: pip install pyprep autoreject pandas --break-system-packages
# ============================================================================

import os
import sys
import json
import time
import argparse
import traceback
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mne
import mne_icalabel

import warnings
warnings.filterwarnings("ignore")


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


# ----------------------------------------------------------------------------
# 1) IMPORT THE MAIN PIPELINE MODULE (for load_eeg_data + shared constants)
# ----------------------------------------------------------------------------
MAIN_PIPELINE_PATH = os.environ.get(
    "MAIN_PIPELINE_PATH",
    r"E:\kntu\Thesis\codes\HOPE-LAST-CODE-FILE\new-tsfnet\eeg10s\eeg10s+fnirs21segment\fnirs-nofilter\Preprocessing_TSFNet_paper_nback_v2_EEG10s_21seg_nofilt.py",
)

if not os.path.isfile(MAIN_PIPELINE_PATH):
    raise FileNotFoundError(
        f"Could not find your main pipeline script at:\n  {MAIN_PIPELINE_PATH}\n"
        "Set MAIN_PIPELINE_PATH above or via environment variable, e.g.:\n"
        '  set MAIN_PIPELINE_PATH=E:\\path\\to\\your\\script.py'
    )

_spec = importlib.util.spec_from_file_location("main_pipeline", MAIN_PIPELINE_PATH)
main_pipeline = importlib.util.module_from_spec(_spec)

# main_pipeline.py has its OWN argparse for --subjects that only accepts
# comma-separated integers (e.g. "1,2,3"), not range syntax like "1-26".
# Since it reads sys.argv directly at import time (top-level code, not
# inside a function), passing --subjects 1-26 to THIS script also gets
# parsed by main_pipeline.py's argparse and crashes with
# "invalid literal for int() with base 10: '1-26'". We only need
# main_pipeline's functions/constants (load_eeg_data, BASE_DATASET_PATH,
# etc.), not its own CLI behavior, so sys.argv is temporarily cleared to
# just the program name during the import, then restored right after.
_saved_argv = sys.argv
sys.argv = [sys.argv[0]]
try:
    _spec.loader.exec_module(main_pipeline)
finally:
    sys.argv = _saved_argv

load_eeg_data = main_pipeline.load_eeg_data
BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
EEG_BP_LOW = main_pipeline.EEG_BP_LOW
EEG_BP_HIGH = main_pipeline.EEG_BP_HIGH
EEG_BP_ORDER = main_pipeline.EEG_BP_ORDER

# ----------------------------------------------------------------------------
# 2) CONFIG
# ----------------------------------------------------------------------------
def _parse_subjects_arg(s):
    """Accepts '1-26', '1,2,3', or a mix like '1-5,25,26'."""
    subjects = set()
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            subjects.update(range(int(a), int(b) + 1))
        elif part:
            subjects.add(int(part))
    return sorted(subjects)


TEST_SUBJECTS_DEFAULT = "1-26"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--subjects", type=str, default=TEST_SUBJECTS_DEFAULT,
                      help="e.g. '1-26' (default, all subjects) or '1,2,3,25,26'.")
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT,
                      help="Parallel subject workers. Keep at 1 in Jupyter. "
                           "In terminal, 2-3 is usually safe depending on RAM.")
_parser.add_argument("--still_bad_flags_if_any", action="store_true", default=True,
                      help="(default on) flag a subject if still_bad_after_cleaning "
                           "is non-empty.")
_parser.add_argument("--reject_fraction_threshold", type=float, default=0.30,
                      help="Flag a subject if autoreject_reject_fraction exceeds this.")
_parser.add_argument("--variance_ratio_threshold", type=float, default=6.0,
                      help="Flag a subject if max_variance_ratio_after exceeds this.")
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = _parse_subjects_arg(_args.subjects)
N_WORKERS = _args.workers
REJECT_FRACTION_THRESHOLD = _args.reject_fraction_threshold
VARIANCE_RATIO_THRESHOLD = _args.variance_ratio_threshold

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_FULL_DATASET_QC_SCAN")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_ROOT, "qc_scan_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------------------
# 3) QC PLOTTING (identical to previous ablation scripts)
# ----------------------------------------------------------------------------
def plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
    raw_eeg_only = raw_obj.copy().pick("eeg")
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
    fig_h = max(6, 0.5 * n_ch_plot)
    fig, ax = plt.subplots(figsize=(14, fig_h))
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


def save_qc_plots(raw_before, filtered, subject_id, qc_dir):
    os.makedirs(qc_dir, exist_ok=True)
    try:
        ch0 = filtered.ch_names[0]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(raw_before.times, raw_before.get_data(picks=[ch0])[0],
                     color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - EEG RAW ({ch0})")
        axes[1].plot(filtered.times, filtered.get_data(picks=[ch0])[0],
                     color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} - EEG PREPROCESSED ({ch0})")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_after.png"), dpi=150)
        plt.close(fig)

        fig_before = plot_all_channels_stacked(
            raw_before, f"{subject_id} - EEG RAW - All Channels", "tab:red")
        fig_after = plot_all_channels_stacked(
            filtered, f"{subject_id} - EEG PREPROCESSED - All Channels", "tab:blue")
        fig_before_zoom = plot_all_channels_stacked(
            raw_before, f"{subject_id} - EEG RAW (10-30s)", "tab:red", tmin=10.0, tmax=30.0)
        fig_after_zoom = plot_all_channels_stacked(
            filtered, f"{subject_id} - EEG PREPROCESSED (10-30s)", "tab:blue", tmin=10.0, tmax=30.0)

        fig_before.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels.png"), dpi=150)
        fig_after.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels.png"), dpi=150)
        fig_before_zoom.savefig(
            os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels_zoom10_30.png"), dpi=150)
        fig_after_zoom.savefig(
            os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels_zoom10_30.png"), dpi=150)
        for f_ in (fig_before, fig_after, fig_before_zoom, fig_after_zoom):
            plt.close(f_)
    except Exception as e:
        log(f"  {subject_id}: QC plot save failed: {e}")


# ----------------------------------------------------------------------------
# 3b) PSD (frequency-domain) QC PLOTS -- new addition.
#
# WHY THESE THREE PLOTS: the time-domain plots above (stacked traces) are
# good at catching spikes, blinks, and continuous noisy channels by eye, but
# several things are much clearer in the frequency domain:
#   - line noise (50/60Hz) -- nearly invisible in a time trace, a sharp
#     unmistakable spike in PSD
#   - whether the gamma band (30-49Hz) actually carries real signal power
#     for a given channel/subject, or is dominated by EMG -- directly
#     relevant to the filter-band-vs-method confound we found earlier
#     (C3/C4's narrower 30Hz low-pass "solving" VP025 was actually just
#     removing this band, not genuinely cleaning it)
#   - exactly WHICH channel and WHICH frequency range is abnormal, which a
#     28-line overlaid PSD plot makes hard to read but a heatmap makes obvious
# ----------------------------------------------------------------------------
def _compute_psd(raw_obj, fmin=0.5, fmax=49.0):
    """Returns (ch_names, freqs, psd_db) where psd_db has shape
    (n_channels, n_freqs), in dB (10*log10(uV^2/Hz)) for readability."""
    raw_eeg = raw_obj.copy().pick("eeg")
    spectrum = raw_eeg.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                    n_fft=int(raw_eeg.info["sfreq"] * 4),
                                    verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)  # psd shape (n_ch, n_freqs), V^2/Hz
    psd_db = 10 * np.log10(np.maximum(psd, 1e-24) * 1e12)  # V^2 -> uV^2, then dB
    return raw_eeg.ch_names, freqs, psd_db


def plot_psd_before_after(raw_before, filtered, subject_id):
    """Plot 1: all channels overlaid, before (red) vs after (blue), side by
    side. A sharp spike at 50Hz that disappears after processing confirms
    line-noise removal; broad shifts show the filter's effect on each band."""
    ch_before, freqs_before, psd_before = _compute_psd(raw_before)
    ch_after, freqs_after, psd_after = _compute_psd(filtered)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i in range(psd_before.shape[0]):
        axes[0].plot(freqs_before, psd_before[i], linewidth=0.6, alpha=0.6)
    axes[0].set_title(f"{subject_id} - PSD RAW (all channels overlaid)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    axes[0].axvline(50, color="k", linestyle="--", linewidth=0.8, label="50Hz line noise")
    axes[0].legend(fontsize=8)

    for i in range(psd_after.shape[0]):
        axes[1].plot(freqs_after, psd_after[i], linewidth=0.6, alpha=0.6)
    axes[1].set_title(f"{subject_id} - PSD PREPROCESSED (all channels overlaid)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].axvline(50, color="k", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    return fig


def plot_psd_heatmap(raw_obj, subject_id, title_suffix):
    """Plot 2: channel x frequency heatmap. Makes it immediately obvious
    WHICH channel is abnormal in WHICH frequency range (e.g. a horizontal
    bright stripe across all frequencies for one channel = broadband noise
    like the VP015 Cz case; a bright region only above 30Hz for one channel
    = EMG-like noise concentrated in gamma, like VP025's Fp1/Fp2)."""
    ch_names, freqs, psd_db = _compute_psd(raw_obj)

    fig, ax = plt.subplots(figsize=(12, max(6, 0.3 * len(ch_names))))
    im = ax.imshow(psd_db, aspect="auto", origin="lower", cmap="viridis",
                    extent=[freqs[0], freqs[-1], 0, len(ch_names)])
    ax.set_yticks(np.arange(len(ch_names)) + 0.5)
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Channel")
    ax.set_title(f"{subject_id} - PSD Heatmap ({title_suffix})")
    fig.colorbar(im, ax=ax, label="Power (dB, 10log10 uV^2/Hz)")
    plt.tight_layout()
    return fig


def plot_band_power_bars(raw_obj, subject_id, title_suffix):
    """Plot 3 (bonus): per-channel power in each classic EEG band (delta,
    theta, alpha, beta, gamma), as grouped bars. Directly answers "does this
    channel actually carry real gamma-band signal, or is it dominated by
    noise" -- relevant to your workload/n-back analysis since beta/gamma
    over prefrontal channels (Fp1/Fp2) is specifically what you care about
    (per the literature search on cognitive-workload-relevant channels)."""
    ch_names, freqs, psd_db = _compute_psd(raw_obj, fmin=0.5, fmax=49.0)
    bands = {"delta(0.5-4)": (0.5, 4), "theta(4-8)": (4, 8), "alpha(8-13)": (8, 13),
             "beta(13-30)": (13, 30), "gamma(30-49)": (30, 49)}

    band_means = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_means[band_name] = psd_db[:, mask].mean(axis=1) if mask.any() else \
            np.full(len(ch_names), np.nan)

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(ch_names)), 6))
    x = np.arange(len(ch_names))
    n_bands = len(bands)
    width = 0.8 / n_bands
    for i, (band_name, vals) in enumerate(band_means.items()):
        ax.bar(x + i * width, vals, width=width, label=band_name)
    ax.set_xticks(x + width * (n_bands - 1) / 2)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    ax.set_title(f"{subject_id} - Per-channel band power ({title_suffix})")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    return fig


def save_psd_plots(raw_before, filtered, subject_id, qc_dir):
    try:
        fig1 = plot_psd_before_after(raw_before, filtered, subject_id)
        fig1.savefig(os.path.join(qc_dir, f"{subject_id}_psd_before_after.png"), dpi=150)
        plt.close(fig1)

        fig2 = plot_psd_heatmap(raw_before, subject_id, "RAW")
        fig2.savefig(os.path.join(qc_dir, f"{subject_id}_psd_heatmap_before.png"), dpi=150)
        plt.close(fig2)

        fig3 = plot_psd_heatmap(filtered, subject_id, "PREPROCESSED")
        fig3.savefig(os.path.join(qc_dir, f"{subject_id}_psd_heatmap_after.png"), dpi=150)
        plt.close(fig3)

        fig4 = plot_band_power_bars(filtered, subject_id, "PREPROCESSED")
        fig4.savefig(os.path.join(qc_dir, f"{subject_id}_band_power_after.png"), dpi=150)
        plt.close(fig4)
    except Exception as e:
        log(f"  {subject_id}: PSD plot save failed: {e}")


# ----------------------------------------------------------------------------
# 4) METRICS
# ----------------------------------------------------------------------------
def _bandpower(data, sfreq, fmin, fmax):
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 4), verbose=False,
    )
    return psd.mean(axis=1)


def compute_metrics(subject_id, raw_before, filtered, bad_channels_detected,
                     n_ica_excluded, ica_excluded_ids, runtime_s,
                     n_rejected_epochs, n_total_epochs):
    picks = mne.pick_types(filtered.info, eeg=True)
    ch_names = [filtered.ch_names[i] for i in picks]

    data_before = raw_before.get_data(picks=picks)
    data_after = filtered.get_data(picks=picks)
    sfreq = filtered.info["sfreq"]

    var_before = data_before.var(axis=1)
    var_after = data_after.var(axis=1)

    emg_power_after = _bandpower(data_after, sfreq, 30, 49)
    eeg_power_before = _bandpower(data_before, sfreq, 4, 30)
    eeg_power_after = _bandpower(data_after, sfreq, 4, 30)
    eeg_power_ratio = eeg_power_after / (eeg_power_before + 1e-20)

    median_var_after = np.median(var_after)
    still_bad_mask = var_after > 4 * median_var_after
    still_bad_channels = [ch_names[i] for i in range(len(ch_names)) if still_bad_mask[i]]

    reject_frac = round(n_rejected_epochs / n_total_epochs, 3) if n_total_epochs else None

    metrics = {
        "subject": subject_id,
        "runtime_s": round(runtime_s, 1),
        "n_channels_used": len(ch_names),
        "bad_channels_detected": sorted(bad_channels_detected),
        "n_bad_channels_detected": len(bad_channels_detected),
        "n_ica_components_excluded": n_ica_excluded,
        "ica_excluded_ids": list(ica_excluded_ids),
        "still_bad_after_cleaning": still_bad_channels,
        "median_variance_before": float(np.median(var_before)),
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
        "autoreject_epochs_rejected": n_rejected_epochs,
        "autoreject_epochs_total": n_total_epochs,
        "autoreject_reject_fraction": reject_frac,
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) THE VALIDATED FINAL PIPELINE (C1)
# ----------------------------------------------------------------------------
def _filter_only(raw):
    filtered = raw.copy()
    filtered.filter(
        l_freq=EEG_BP_LOW, h_freq=EEG_BP_HIGH, method="iir",
        iir_params=dict(order=EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    return filtered


def _run_pyprep(raw_filtered, subject_id):
    from pyprep import NoisyChannels

    tmp = raw_filtered.copy()
    nd = NoisyChannels(tmp, random_state=42)
    try:
        nd.find_all_bads(ransac=True)
    except Exception as e:
        log(f"  {subject_id}: PyPREP RANSAC failed ({e}); retrying without RANSAC.")
        nd.find_all_bads(ransac=False)

    bads = nd.get_bads()
    tmp.info["bads"] = bads
    if bads:
        tmp.interpolate_bads(reset_bads=True, verbose=False)
    return tmp, bads


def _run_ica_argmax_ocular(filtered, raw_before, subject_id):
    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    eog_indices = []
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch,
                                            threshold=3.0, verbose=False)
                eog_indices.extend(idx)
            except Exception as e:
                log(f"  {subject_id}: find_bads_eog({eog_ch}) failed: {e}")

    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"])
                           if lab in ("eye blink", "eye movement")]
    except Exception:
        iclabel_ocular = []

    exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    ica.exclude = exclude
    if exclude:
        ica.apply(filtered, verbose=False)
    return filtered, exclude


def _run_autoreject_qc_epochs(filtered, subject_id):
    from autoreject import AutoReject

    events = mne.make_fixed_length_events(filtered, duration=3.0)
    epochs = mne.Epochs(filtered, events, tmin=0, tmax=3.0, baseline=None,
                         preload=True, verbose=False, picks="eeg")
    if len(epochs) < 5:
        log(f"  {subject_id}: too few pseudo-epochs ({len(epochs)}) for AutoReject, skipping.")
        return 0, len(epochs)

    ar = AutoReject(random_state=42, verbose=False, n_jobs=1)
    _, reject_log = ar.fit_transform(epochs, return_log=True)
    n_rejected = int(reject_log.bad_epochs.sum())
    return n_rejected, len(epochs)


def process_one_subject(subject_num):
    subject_id = f"VP{subject_num:03d}"
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    log(f"Processing {subject_id} from {eeg_folder} ...")
    t0 = time.time()
    try:
        raw, events, sfreq, mnt_struct = load_eeg_data(eeg_folder)
    except Exception as e:
        log(f"FAILED to load {subject_id}: {e}")
        traceback.print_exc()
        return None

    try:
        raw_before = raw.copy()
        filtered = _filter_only(raw)
        filtered, bads_detected = _run_pyprep(filtered, subject_id)
        filtered.set_eeg_reference("average", projection=False, verbose=False)
        filtered, excluded = _run_ica_argmax_ocular(filtered, raw_before, subject_id)
        n_rejected_epochs, n_total_epochs = _run_autoreject_qc_epochs(filtered, subject_id)
        runtime_s = time.time() - t0

        qc_dir = os.path.join(OUTPUT_ROOT, subject_id)
        save_qc_plots(raw_before, filtered, subject_id, qc_dir)
        save_psd_plots(raw_before, filtered, subject_id, qc_dir)

        metrics = compute_metrics(subject_id, raw_before, filtered, bads_detected,
                                   len(excluded), excluded, runtime_s,
                                   n_rejected_epochs, n_total_epochs)

        try:
            with open(os.path.join(qc_dir, f"{subject_id}_metrics.json"), "w",
                      encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
        except Exception as e:
            log(f"  {subject_id}: WARNING could not save metrics JSON ({e}); "
                f"metrics were still computed and will appear in all_subjects_metrics.csv.")

        log(f"  {subject_id} done in {runtime_s:.1f}s | bads={bads_detected} | "
            f"ICA excluded={len(excluded)} | still_bad_after={metrics['still_bad_after_cleaning']} | "
            f"max_var_ratio={metrics['max_variance_ratio_after']:.2f} | "
            f"autoreject_reject_frac={metrics['autoreject_reject_fraction']}")

        return metrics

    except Exception as e:
        log(f"FAILED processing {subject_id}: {e}")
        traceback.print_exc()
        return None


# ----------------------------------------------------------------------------
# 6) MAIN LOOP
# ----------------------------------------------------------------------------
def main():
    log("===== EEG FULL-DATASET QC SCAN STARTED =====")
    log(f"Subjects: {TEST_SUBJECTS} ({len(TEST_SUBJECTS)} total)")
    log(f"Pipeline: filter(0.5-50Hz) -> PyPREP -> CAR -> ICA(find_bads_eog+ICLabel-ocular) "
        f"-> AutoReject (diagnostic)")
    log(f"Flag thresholds: reject_fraction>{REJECT_FRACTION_THRESHOLD}, "
        f"variance_ratio>{VARIANCE_RATIO_THRESHOLD}, or any still_bad_after_cleaning")
    log(f"Output root: {OUTPUT_ROOT}")

    all_rows = []
    if N_WORKERS <= 1:
        for sub in TEST_SUBJECTS:
            m = process_one_subject(sub)
            if m is not None:
                all_rows.append(m)
    else:
        import multiprocessing as mp
        with mp.Pool(processes=N_WORKERS) as pool:
            for m in pool.imap_unordered(process_one_subject, TEST_SUBJECTS):
                if m is not None:
                    all_rows.append(m)

    if not all_rows:
        log("No metrics collected -- all subjects failed. Check qc_scan_log.txt.")
        return

    df = pd.DataFrame(all_rows)
    df = df.sort_values("subject").reset_index(drop=True)
    csv_path = os.path.join(OUTPUT_ROOT, "all_subjects_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"Saved combined metrics table: {csv_path}")

    build_flagged_report(df)
    log("===== EEG FULL-DATASET QC SCAN FINISHED =====")


# ----------------------------------------------------------------------------
# 7) PRIORITIZED FLAGGED-SUBJECTS REPORT
# ----------------------------------------------------------------------------
def build_flagged_report(df):
    def severity_score(row):
        score = 0.0
        score += len(row["still_bad_after_cleaning"]) * 10   # heaviest weight -- this is
                                                                # the VP025-style pattern
        rf = row["autoreject_reject_fraction"]
        if rf is not None and rf > REJECT_FRACTION_THRESHOLD:
            score += rf * 5
        if row["max_variance_ratio_after"] > VARIANCE_RATIO_THRESHOLD:
            score += row["max_variance_ratio_after"] * 0.5
        return score

    df = df.copy()
    df["severity_score"] = df.apply(severity_score, axis=1)

    def is_flagged(row):
        rf = row["autoreject_reject_fraction"]
        return (
            len(row["still_bad_after_cleaning"]) > 0
            or (rf is not None and rf > REJECT_FRACTION_THRESHOLD)
            or row["max_variance_ratio_after"] > VARIANCE_RATIO_THRESHOLD
        )

    df["flagged"] = df.apply(is_flagged, axis=1)
    flagged = df[df["flagged"]].sort_values("severity_score", ascending=False)
    clean = df[~df["flagged"]].sort_values("subject")

    lines = []
    lines.append("=" * 78)
    lines.append("EEG FULL-DATASET QC SCAN — FLAGGED SUBJECTS REPORT")
    lines.append("=" * 78)
    lines.append(f"Total subjects scanned: {len(df)}")
    lines.append(f"Flagged for manual review: {len(flagged)}")
    lines.append(f"Clean (no manual review needed): {len(clean)}")
    lines.append("")
    lines.append(f"Flag thresholds: still_bad_after_cleaning non-empty (any), "
                 f"autoreject_reject_fraction > {REJECT_FRACTION_THRESHOLD}, "
                 f"OR max_variance_ratio_after > {VARIANCE_RATIO_THRESHOLD}")
    lines.append("")

    lines.append("--- REVIEW THESE FIRST, MOST SEVERE AT TOP ---")
    if len(flagged) == 0:
        lines.append("(none -- every subject passed all thresholds)")
    for _, row in flagged.iterrows():
        lines.append(
            f"\n{row['subject']}  (severity score {row['severity_score']:.1f})"
        )
        if row["still_bad_after_cleaning"]:
            lines.append(f"  ** still_bad_after_cleaning={row['still_bad_after_cleaning']} "
                         f"-- pattern matches VP025's Fp1/Fp2 case. Open this subject's QC "
                         f"images and decide whether manual channel interpolation is needed.")
        rf = row["autoreject_reject_fraction"]
        if rf is not None and rf > REJECT_FRACTION_THRESHOLD:
            lines.append(f"  ** autoreject_reject_fraction={rf} -- more than "
                         f"{REJECT_FRACTION_THRESHOLD*100:.0f}% of pseudo-epochs still look "
                         f"artifactual after cleaning.")
        if row["max_variance_ratio_after"] > VARIANCE_RATIO_THRESHOLD:
            lines.append(f"  ** max_variance_ratio_after={row['max_variance_ratio_after']:.2f} "
                         f"-- one channel still dominates the others.")
        lines.append(f"  bads_detected_by_pyprep={row['bad_channels_detected']}")
        lines.append(f"  QC images: {OUTPUT_ROOT}\\{row['subject']}\\")
    lines.append("")

    lines.append("--- CLEAN, NO MANUAL REVIEW NEEDED (passed all thresholds) ---")
    if len(clean) == 0:
        lines.append("(none)")
    for _, row in clean.iterrows():
        lines.append(f"  {row['subject']}: bads={row['bad_channels_detected']}, "
                     f"reject_frac={row['autoreject_reject_fraction']}, "
                     f"max_var_ratio={row['max_variance_ratio_after']:.2f}")
    lines.append("")

    lines.append("NEXT STEP: open the QC images (5 per subject, same format you've reviewed")
    lines.append("before) for every subject in the flagged list above, starting from the top.")
    lines.append("For each, decide (same way we did for VP025): does this need manual channel")
    lines.append("interpolation, or is the automated PyPREP+CAR+ICA+AutoReject result actually")
    lines.append("fine and just tripped a threshold without a real visible problem?")

    report_path = os.path.join(OUTPUT_ROOT, "flagged_subjects_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Saved flagged-subjects report: {report_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
