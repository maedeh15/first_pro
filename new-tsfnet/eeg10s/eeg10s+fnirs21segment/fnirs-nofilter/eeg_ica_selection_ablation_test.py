# ============================================================================
# EEG ICA-SELECTION ABLATION TEST — compares how ICA components are chosen
# for exclusion, both built on the already-validated A1 ordering (filter ->
# PyPREP -> CAR -> ICA -> epoch rejection). Tests whether the
# probability-threshold approach from your exploratory notebook
# (Preprocessing_v2_run_a1b_10s.ipynb) catches the persistent, "diluted"
# EMG-like noise on Fp1/Fp2 in VP025 that plain argmax-based ICLabel and
# find_bads_eog (the current final pipeline) do not.
#
# WHAT IS COMPARED (2 variants, both after filter -> PyPREP -> CAR):
#   B1_argmax_only     : CURRENT FINAL PIPELINE'S selection method.
#                          exclude = find_bads_eog(VEOG/HEOG) UNION
#                                    ICLabel-argmax in {eye blink, eye movement}
#   B2_prob_threshold   : exploratory-notebook's selection method.
#                          exclude = find_bads_eog(VEOG/HEOG) UNION
#                                    ICLabel-argmax in {eye blink, eye movement,
#                                    muscle artifact, line noise} UNION
#                                    (ICLabel probability >= 0.30 for any of
#                                    those 4 classes, even if argmax says
#                                    something else, e.g. 'brain')
#                          MINUS a rescue: components flagged ONLY by
#                          find_bads_eog (no ICLabel support at all) are kept
#                          if ICLabel's own 'brain' probability >= 0.50 for
#                          that component (avoids discarding real brain
#                          signal that merely correlates with EOG).
#
# NOTE: B2 is intentionally broader than the plain "V3_prep_iclabel_all"
# tested earlier (which only added muscle/channel-noise via argmax). B2 adds
# the probability-threshold mechanism itself, which is the part of the
# notebook's logic specifically designed to catch a real source "diluted"
# across several under-scored components -- exactly the failure mode
# hypothesized for VP025's Fp1/Fp2.
#
# WHAT IS SAVED: identical structure to the previous ablation scripts -- same
# 5 QC images per (subject, variant), a per-(subject,variant) metrics JSON,
# and global all_metrics.csv / ranking_report.txt / ranking_summary.png
# under Results/EEG_ICA_SELECTION_ABLATION_TEST/.
#
# HOW TO RUN:
#   python eeg_ica_selection_ablation_test.py
#   python eeg_ica_selection_ablation_test.py --subjects 25,1,15,18,21,5 --workers 1
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
        "Set MAIN_PIPELINE_PATH above or via environment variable."
    )

_spec = importlib.util.spec_from_file_location("main_pipeline", MAIN_PIPELINE_PATH)
main_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main_pipeline)

load_eeg_data = main_pipeline.load_eeg_data
BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
SFREQ_EEG = main_pipeline.SFREQ_EEG
EEG_BP_LOW = main_pipeline.EEG_BP_LOW
EEG_BP_HIGH = main_pipeline.EEG_BP_HIGH
EEG_BP_ORDER = main_pipeline.EEG_BP_ORDER

# ----------------------------------------------------------------------------
# 2) CONFIG (edit directly for Jupyter; --flags work from terminal too)
# ----------------------------------------------------------------------------
# VP025 is listed FIRST since it's the specific case this test targets, but
# the other 5 subjects (already covered by earlier ablation tests) are kept
# too, so we can also confirm B2 does NOT over-clean subjects that were
# already handled well by B1.
TEST_SUBJECTS_DEFAULT = "25,1,15,18,21,5"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--subjects", type=str, default=TEST_SUBJECTS_DEFAULT)
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = [int(s.strip()) for s in _args.subjects.split(",")]
N_WORKERS = _args.workers

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_ICA_SELECTION_ABLATION_TEST")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_ROOT, "ablation_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


VARIANTS = ["B1_argmax_only", "B2_prob_threshold"]

# Known problem channels per subject, from the manual QC review across all
# previous ablation tests -- used only to report whether each variant
# resolved them (still_bad_after_cleaning), same reporting purpose as
# KNOWN_BAD_CHANNELS in the first ablation script.
KNOWN_PERSISTENT_NOISE_CHANNELS = {
    25: ["Fp1", "Fp2"],   # persistent EMG-like noise, unresolved by PyPREP/B1 ICA/AutoReject so far
}


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


def save_qc_plots(raw_before, filtered, subject_id, variant_name, qc_dir):
    os.makedirs(qc_dir, exist_ok=True)
    try:
        ch0 = filtered.ch_names[0]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(raw_before.times, raw_before.get_data(picks=[ch0])[0],
                     color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} [{variant_name}] - EEG RAW ({ch0})")
        axes[1].plot(filtered.times, filtered.get_data(picks=[ch0])[0],
                     color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} [{variant_name}] - EEG PREPROCESSED ({ch0})")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_after.png"), dpi=150)
        plt.close(fig)

        fig_before = plot_all_channels_stacked(
            raw_before, f"{subject_id} [{variant_name}] - EEG RAW - All Channels", "tab:red")
        fig_after = plot_all_channels_stacked(
            filtered, f"{subject_id} [{variant_name}] - EEG PREPROCESSED - All Channels", "tab:blue")
        fig_before_zoom = plot_all_channels_stacked(
            raw_before, f"{subject_id} [{variant_name}] - EEG RAW (10-30s)", "tab:red",
            tmin=10.0, tmax=30.0)
        fig_after_zoom = plot_all_channels_stacked(
            filtered, f"{subject_id} [{variant_name}] - EEG PREPROCESSED (10-30s)", "tab:blue",
            tmin=10.0, tmax=30.0)

        fig_before.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels.png"), dpi=150)
        fig_after.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels.png"), dpi=150)
        fig_before_zoom.savefig(
            os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels_zoom10_30.png"), dpi=150)
        fig_after_zoom.savefig(
            os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels_zoom10_30.png"), dpi=150)
        for f_ in (fig_before, fig_after, fig_before_zoom, fig_after_zoom):
            plt.close(f_)
    except Exception as e:
        log(f"  {subject_id} [{variant_name}]: QC plot save failed: {e}")


# ----------------------------------------------------------------------------
# 4) METRICS
# ----------------------------------------------------------------------------
def _bandpower(data, sfreq, fmin, fmax):
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 4), verbose=False,
    )
    return psd.mean(axis=1)


def compute_metrics(subject_num, variant_name, raw_before, filtered,
                     bad_channels_detected, n_ica_excluded, ica_excluded_ids,
                     runtime_s):
    subject_id = f"VP{subject_num:03d}"
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

    known_target = set(KNOWN_PERSISTENT_NOISE_CHANNELS.get(subject_num, []))
    still_bad_set = set(still_bad_channels)
    known_persistent_still_present = sorted(known_target & still_bad_set)
    # A slightly softer check than exact variance-outlier: also report the
    # EMG-band power specifically ON the known-problem channels, in case the
    # noise reduced but didn't drop the variance below the 4x-median cutoff.
    target_ch_emg_power = {}
    for ch in known_target:
        if ch in ch_names:
            idx = ch_names.index(ch)
            target_ch_emg_power[ch] = float(emg_power_after[idx])

    metrics = {
        "subject": subject_id,
        "variant": variant_name,
        "runtime_s": round(runtime_s, 1),
        "n_channels_used": len(ch_names),
        "bad_channels_detected": sorted(bad_channels_detected),
        "n_bad_channels_detected": len(bad_channels_detected),
        "n_ica_components_excluded": n_ica_excluded,
        "ica_excluded_ids": list(ica_excluded_ids),
        "still_bad_after_cleaning": still_bad_channels,
        "known_persistent_noise_channels": sorted(known_target),
        "known_persistent_still_present": known_persistent_still_present,
        "target_channel_emg_band_power": target_ch_emg_power,
        "median_variance_before": float(np.median(var_before)),
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) SHARED STEPS (filter -> PyPREP -> CAR, identical for both variants)
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


# ----------------------------------------------------------------------------
# 6) THE TWO ICA-COMPONENT-SELECTION VARIANTS
# ----------------------------------------------------------------------------
# Standard ICLabel class order (verified from ica.labels_scores_ shape=(n,7)):
ARTIFACT_CLASSES_IDX = {
    "brain": 0, "muscle artifact": 1, "eye blink": 2,
    "heart beat": 3, "line noise": 4, "channel noise": 5, "other": 6,
}


def _fit_ica(filtered, subject_id):
    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)
    return ica


def _select_B1_argmax_only(ica, filtered, raw_before, subject_id):
    """Current final pipeline: find_bads_eog UNION ICLabel-argmax ocular."""
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
    return exclude


def _select_B2_prob_threshold(ica, filtered, raw_before, subject_id):
    """Exploratory-notebook method: argmax over a wider target set UNION
    probability-threshold (>=0.30) over the same target set UNION
    find_bads_eog, MINUS an EOG-only rescue for components ICLabel itself
    scores as likely 'brain' (>=0.50)."""
    ARTIFACT_TARGETS = ["eye blink", "eye movement", "muscle artifact", "line noise"]
    PROB_THRESHOLD = 0.30
    BRAIN_PROB_CEILING = 0.5

    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
    except Exception as e:
        log(f"  {subject_id}: ICLabel failed ({e}), falling back to B1 selection.")
        return _select_B1_argmax_only(ica, filtered, raw_before, subject_id)

    proba_matrix = ica.labels_scores_  # shape (n_components, 7)

    artifact_idx_argmax = [i for i, lab in enumerate(ic_labels["labels"])
                            if lab in ARTIFACT_TARGETS]

    artifact_idx_prob = []
    for comp_idx in range(proba_matrix.shape[0]):
        for cls_name in ARTIFACT_TARGETS:
            cls_col = ARTIFACT_CLASSES_IDX.get(cls_name)
            if cls_col is not None and proba_matrix[comp_idx, cls_col] >= PROB_THRESHOLD:
                artifact_idx_prob.append(comp_idx)
                break

    eog_indices_veog, eog_indices_heog = [], []
    try:
        if "VEOG" in filtered.ch_names:
            eog_indices_veog, _ = ica.find_bads_eog(raw_before, ch_name="VEOG",
                                                      threshold=2.0, verbose=False)
        if "HEOG" in filtered.ch_names:
            eog_indices_heog, _ = ica.find_bads_eog(raw_before, ch_name="HEOG",
                                                      threshold=2.0, verbose=False)
    except Exception as e:
        log(f"  {subject_id}: find_bads_eog failed: {e}")

    BRAIN_CLASS_COL = ARTIFACT_CLASSES_IDX["brain"]
    eog_only_idx = (set(eog_indices_veog) | set(eog_indices_heog)) - \
                   (set(artifact_idx_argmax) | set(artifact_idx_prob))
    eog_only_rescued = {idx for idx in eog_only_idx
                         if proba_matrix[idx, BRAIN_CLASS_COL] >= BRAIN_PROB_CEILING}
    if eog_only_rescued:
        log(f"  {subject_id}: rescued from EOG-only rejection (brain-prob>=0.5): "
            f"{sorted(eog_only_rescued)}")

    exclude = sorted(
        (set(artifact_idx_argmax) | set(artifact_idx_prob) |
         set(eog_indices_veog) | set(eog_indices_heog)) - eog_only_rescued
    )
    return exclude


def run_variant(variant_name, subject_num, raw):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    raw_before = raw.copy()

    filtered = _filter_only(raw)
    filtered, bads_detected = _run_pyprep(filtered, subject_id)
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    ica = _fit_ica(filtered, subject_id)

    if variant_name == "B1_argmax_only":
        excluded = _select_B1_argmax_only(ica, filtered, raw_before, subject_id)
    elif variant_name == "B2_prob_threshold":
        excluded = _select_B2_prob_threshold(ica, filtered, raw_before, subject_id)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    ica.exclude = excluded
    if ica.exclude:
        ica.apply(filtered, verbose=False)

    n_rejected_epochs, n_total_epochs = _run_autoreject_qc_epochs(filtered, subject_id)
    runtime_s = time.time() - t0

    qc_dir = os.path.join(OUTPUT_ROOT, variant_name, subject_id)
    save_qc_plots(raw_before, filtered, subject_id, variant_name, qc_dir)

    metrics = compute_metrics(subject_num, variant_name, raw_before, filtered,
                               bads_detected, len(excluded), excluded, runtime_s)
    metrics["autoreject_epochs_rejected"] = n_rejected_epochs
    metrics["autoreject_epochs_total"] = n_total_epochs
    metrics["autoreject_reject_fraction"] = (
        round(n_rejected_epochs / n_total_epochs, 3) if n_total_epochs else None
    )

    try:
        with open(os.path.join(qc_dir, f"{subject_id}_{variant_name}_metrics.json"), "w",
                  encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False, default=_json_default)
    except Exception as e:
        log(f"  {subject_id} [{variant_name}]: WARNING could not save metrics JSON ({e}); "
            f"metrics were still computed and will appear in all_metrics.csv.")

    log(f"  {subject_id} [{variant_name}] done in {runtime_s:.1f}s | "
        f"bads={bads_detected} | ICA excluded={len(excluded)} {excluded} | "
        f"still_bad_after={metrics['still_bad_after_cleaning']} | "
        f"known_persistent_still_present={metrics['known_persistent_still_present']}")

    return metrics


# ----------------------------------------------------------------------------
# 7) MAIN LOOP
# ----------------------------------------------------------------------------
def process_one_subject(subject_num):
    subject_id = f"VP{subject_num:03d}"
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    log(f"Loading {subject_id} from {eeg_folder} ...")
    try:
        raw, events, sfreq, mnt_struct = load_eeg_data(eeg_folder)
    except Exception as e:
        log(f"FAILED to load {subject_id}: {e}")
        traceback.print_exc()
        return []

    all_metrics = []
    for variant_name in VARIANTS:
        try:
            m = run_variant(variant_name, subject_num, raw)
            all_metrics.append(m)
        except Exception as e:
            log(f"FAILED variant {variant_name} for {subject_id}: {e}")
            traceback.print_exc()
    return all_metrics


def main():
    log("===== EEG ICA-SELECTION ABLATION TEST STARTED (B1: argmax-only vs B2: prob-threshold) =====")
    log(f"Test subjects: {TEST_SUBJECTS} (VP025 first -- the targeted case)")
    log(f"Variants: {VARIANTS}")
    log(f"Output root: {OUTPUT_ROOT}")

    all_rows = []
    if N_WORKERS <= 1:
        for sub in TEST_SUBJECTS:
            all_rows.extend(process_one_subject(sub))
    else:
        import multiprocessing as mp
        with mp.Pool(processes=N_WORKERS) as pool:
            for rows in pool.imap_unordered(process_one_subject, TEST_SUBJECTS):
                all_rows.extend(rows)

    if not all_rows:
        log("No metrics collected -- all subjects failed. Check ablation_log.txt.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUTPUT_ROOT, "all_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log(f"Saved combined metrics table: {csv_path}")

    build_ranking_report(df)
    build_ranking_chart(df)
    log("===== EEG ICA-SELECTION ABLATION TEST FINISHED =====")


# ----------------------------------------------------------------------------
# 8) RANKING / FINAL TEXT REPORT
# ----------------------------------------------------------------------------
def build_ranking_report(df):
    lines = []
    lines.append("=" * 78)
    lines.append("EEG ICA-SELECTION ABLATION — RANKING REPORT (B1: argmax-only vs B2: prob-threshold)")
    lines.append("=" * 78)
    lines.append(f"Subjects tested: {sorted(df['subject'].unique().tolist())}")
    lines.append(f"Variants tested: {VARIANTS}")
    lines.append("")

    summary = df.groupby("variant").agg(
        mean_still_bad_count=("still_bad_after_cleaning", lambda s: np.mean([len(x) for x in s])),
        mean_max_variance_ratio_after=("max_variance_ratio_after", "mean"),
        mean_emg_band_power_after=("median_emg_band_power_after_30_49hz", "mean"),
        mean_eeg_band_retention=("median_eeg_band_power_retention_ratio_4_30hz", "mean"),
        mean_ica_excluded=("n_ica_components_excluded", "mean"),
        mean_autoreject_reject_fraction=("autoreject_reject_fraction", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    ).reset_index()

    lines.append("--- Per-variant averages across all test subjects ---")
    lines.append(summary.to_string(index=False))
    lines.append("")

    lines.append("--- THE KEY QUESTION FOR THIS TEST: VP025's Fp1/Fp2 ---")
    vp025 = df[df["subject"] == "VP025"]
    for _, row in vp025.iterrows():
        lines.append(f"  {row['variant']:<20s} still_bad={row['still_bad_after_cleaning']} "
                     f"known_persistent_still_present={row['known_persistent_still_present']} "
                     f"target_channel_emg_power={row['target_channel_emg_band_power']} "
                     f"ica_excluded_ids={row['ica_excluded_ids']}")
    lines.append("")
    lines.append("If B2's known_persistent_still_present is EMPTY (or its target_channel_emg_band_power")
    lines.append("values are clearly lower than B1's) while B1's still shows Fp1/Fp2, that is direct")
    lines.append("evidence the probability-threshold mechanism caught the diluted noise source B1 missed.")
    lines.append("")

    lines.append("--- How to read the rest ---")
    lines.append("mean_still_bad_count            : LOWER is better (channels still >4x median variance).")
    lines.append("mean_emg_band_power_after       : LOWER is better (less residual muscle noise), BUT")
    lines.append("                                   check eeg_band_retention below for over-cleaning.")
    lines.append("mean_eeg_band_retention         : should stay close to B1's value for the OTHER 5")
    lines.append("                                   subjects -- if B2's retention drops sharply there")
    lines.append("                                   (not just on VP025), it is over-cleaning real brain")
    lines.append("                                   signal by being too aggressive with the probability")
    lines.append("                                   threshold.")
    lines.append("mean_ica_excluded               : B2 removing noticeably MORE components across ALL")
    lines.append("                                   subjects (not just VP025) without a clear metric")
    lines.append("                                   benefit is a red flag for false positives.")
    lines.append("")

    lines.append("IMPORTANT: as with previous ablation tests, this is a starting point -- open the QC")
    lines.append(f"images for VP025 under both:")
    lines.append(f"  {OUTPUT_ROOT}\\B1_argmax_only\\VP025\\")
    lines.append(f"  {OUTPUT_ROOT}\\B2_prob_threshold\\VP025\\")
    lines.append("and visually confirm whether Fp1/Fp2's persistent noise actually looks cleaner in B2,")
    lines.append("and that the other 5 subjects were NOT over-cleaned by B2.")
    lines.append("")

    lines.append("--- Per-subject detail (all subjects) ---")
    for sub in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == sub]
        lines.append(f"\n{sub}:")
        for _, row in sub_df.iterrows():
            lines.append(
                f"  {row['variant']:<20s} bads={row['bad_channels_detected']} "
                f"still_bad={row['still_bad_after_cleaning']} "
                f"ica_excluded={row['n_ica_components_excluded']} {row['ica_excluded_ids']} "
                f"emg_power={row['median_emg_band_power_after_30_49hz']:.3e} "
                f"eeg_retention={row['median_eeg_band_power_retention_ratio_4_30hz']:.3f}"
            )

    report_path = os.path.join(OUTPUT_ROOT, "ranking_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Saved ranking report: {report_path}")
    print("\n".join(lines))


def build_ranking_chart(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics_to_plot = [
        ("still_bad_after_cleaning", "Channels still bad after cleaning (count, lower=better)",
         lambda s: s.apply(len)),
        ("n_ica_components_excluded", "Mean ICA components excluded", None),
        ("median_emg_band_power_after_30_49hz", "Residual EMG-band power 30-49Hz (lower=better)", None),
        ("median_eeg_band_power_retention_ratio_4_30hz",
         "EEG-band (4-30Hz) power retention ratio (~stable=better)", None),
    ]

    for ax, (col, title, transform) in zip(axes.flat, metrics_to_plot):
        vals = df[col] if transform is None else transform(df[col])
        plot_df = df.copy()
        plot_df["_val"] = vals
        grouped = plot_df.groupby("variant")["_val"].mean().reindex(VARIANTS)
        ax.bar(grouped.index, grouped.values, color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=15, labelsize=8)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_ROOT, "ranking_summary.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    log(f"Saved ranking chart: {chart_path}")


if __name__ == "__main__":
    main()
