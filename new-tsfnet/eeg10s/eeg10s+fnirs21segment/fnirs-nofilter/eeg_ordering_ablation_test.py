# ============================================================================
# EEG ORDERING ABLATION TEST — compares CAR-vs-ICA ORDER (both AFTER PyPREP
# bad-channel fix), to settle whether ICA should run before or after CAR.
#
# WHAT IS COMPARED (2 variants, both built on the already-validated
# PyPREP-before-CAR fix from eeg_cleaning_ablation_test.py's V1_prep):
#   A1_car_then_ica : filter -> PyPREP -> CAR -> ICA(+ICLabel ocular)
#                      -> epoch-level rejection (QC-only here, see note below)
#   A2_ica_then_car : filter -> PyPREP -> ICA(+ICLabel ocular) -> CAR
#                      -> epoch-level rejection (QC-only here, see note below)
#
# NOTE ON ICLabel: ICLabel is a CLASSIFIER that labels ALREADY-FITTED ICA
# components (ocular / muscle / channel noise / brain / ...). It cannot run
# as a standalone step before ICA exists -- there would be nothing to label.
# So "ICLabel before CAR" in A2, as a literal separate step, is not
# something that can be implemented; the only meaningful reading is "run
# ICA (fit + ICLabel-assisted component selection + apply) before CAR
# instead of after it" -- which is what A2_ica_then_car implements below.
# If this is not what you meant, tell me and I will adjust.
#
# NOTE ON epoch-level rejection: exactly like eeg_cleaning_ablation_test.py,
# this script runs AutoReject on FIXED-LENGTH 3s pseudo-epochs purely for
# QC/diagnostic metrics (how much residual artifact remains after each
# ordering). It does NOT feed back into extract_windows() in your real
# pipeline -- wiring real epoch-level rejection into your trial-based
# windowing is a separate, later step once A1 vs A2 is decided.
#
# WHAT IS SAVED: identical structure to eeg_cleaning_ablation_test.py --
# same 5 QC images per (subject, variant), a per-(subject,variant) metrics
# JSON, and global all_metrics.csv / ranking_report.txt / ranking_summary.png
# under Results/EEG_ORDERING_ABLATION_TEST/.
#
# HOW TO RUN:
#   python eeg_ordering_ablation_test.py
#   python eeg_ordering_ablation_test.py --subjects 1,15,18,21,5 --workers 1
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
    """See eeg_cleaning_ablation_test.py for full rationale: converts numpy
    scalar types (np.int64/float64/bool_/ndarray) that PyPREP/MNE/ICA return
    mixed with plain Python types, which json.dump() cannot serialize
    natively."""
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
# 1) IMPORT THE MAIN PIPELINE MODULE (for load_eeg_data + shared constants
#    only -- same approach as eeg_cleaning_ablation_test.py).
# ----------------------------------------------------------------------------
MAIN_PIPELINE_PATH = os.environ.get(
    "MAIN_PIPELINE_PATH",
    r"E:\kntu\Thesis\codes\HOPE-LAST-CODE-FILE\new-tsfnet\eeg10s\eeg10s+fnirs21segment\fnirs-nofilter\Preprocessing_TSFNet_paper_nback_v2_EEG10s_21seg_nofilt.py",
)

if not os.path.isfile(MAIN_PIPELINE_PATH):
    raise FileNotFoundError(
        f"Could not find your main pipeline script at:\n  {MAIN_PIPELINE_PATH}\n"
        "Set the correct path either by editing MAIN_PIPELINE_PATH above, or by "
        "setting the environment variable MAIN_PIPELINE_PATH before running this "
        "script."
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
# 2) CONFIG (edit these two lines directly when running from Jupyter --
#    same reasoning as eeg_cleaning_ablation_test.py: a notebook kernel
#    passes its own argv, so parse_known_args is used to stay safe in both
#    Jupyter and terminal).
# ----------------------------------------------------------------------------
TEST_SUBJECTS_DEFAULT = "1,15,18,21,5"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--subjects", type=str, default=TEST_SUBJECTS_DEFAULT,
                      help="Comma-separated subject numbers. Default matches "
                           "the previous ablation test's 5 test subjects so "
                           "results are directly comparable.")
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT,
                      help="Parallel subject workers. Keep at 1 in Jupyter.")
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = [int(s.strip()) for s in _args.subjects.split(",")]
N_WORKERS = _args.workers

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_ORDERING_ABLATION_TEST")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_ROOT, "ablation_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


VARIANTS = ["A1_car_then_ica", "A2_ica_then_car"]


# ----------------------------------------------------------------------------
# 3) QC PLOTTING (identical to eeg_cleaning_ablation_test.py)
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
# 4) METRICS (identical logic to eeg_cleaning_ablation_test.py)
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

    # Inter-channel similarity: after a GOOD cleaning, channels that are
    # topographic neighbours should still correlate reasonably (real brain
    # signal is spatially smooth); a ordering that badly distorts spatial
    # structure would show unusually LOW average pairwise correlation.
    corr_matrix = np.corrcoef(data_after)
    off_diag = corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]
    mean_abs_interchannel_corr = float(np.mean(np.abs(off_diag)))

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
        "median_variance_before": float(np.median(var_before)),
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
        "mean_abs_interchannel_correlation": mean_abs_interchannel_corr,
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) SHARED STEPS
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


def _fit_and_apply_ica(filtered, raw_before_ref, subject_id):
    """ICA fit -> EOG-correlation + ICLabel ocular detection -> apply.
    Returns (cleaned_raw, excluded_component_ids)."""
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
                idx, _ = ica.find_bads_eog(raw_before_ref, ch_name=eog_ch,
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

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    if ica.exclude:
        ica.apply(filtered, verbose=False)
    return filtered, ica.exclude


def _run_autoreject_qc_epochs(filtered, subject_id):
    """QC-only diagnostic, identical purpose/limits as in
    eeg_cleaning_ablation_test.py -- see module docstring above."""
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
# 6) THE 2 ORDERING VARIANTS
# ----------------------------------------------------------------------------
def run_variant(variant_name, subject_num, raw):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    raw_before = raw.copy()

    filtered = _filter_only(raw)
    filtered, bads_detected = _run_pyprep(filtered, subject_id)

    if variant_name == "A1_car_then_ica":
        filtered.set_eeg_reference("average", projection=False, verbose=False)
        filtered, excluded = _fit_and_apply_ica(filtered, raw_before, subject_id)

    elif variant_name == "A2_ica_then_car":
        filtered, excluded = _fit_and_apply_ica(filtered, raw_before, subject_id)
        filtered.set_eeg_reference("average", projection=False, verbose=False)

    else:
        raise ValueError(f"Unknown variant: {variant_name}")

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
        f"bads={bads_detected} | ICA excluded={len(excluded)} | "
        f"still_bad_after={metrics['still_bad_after_cleaning']} | "
        f"interchannel_corr={metrics['mean_abs_interchannel_correlation']:.3f}")

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
    log("===== EEG ORDERING ABLATION TEST STARTED (A1: CAR->ICA vs A2: ICA->CAR) =====")
    log(f"Test subjects: {TEST_SUBJECTS}")
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
    log("===== EEG ORDERING ABLATION TEST FINISHED =====")


# ----------------------------------------------------------------------------
# 8) RANKING / FINAL TEXT REPORT
# ----------------------------------------------------------------------------
def build_ranking_report(df):
    lines = []
    lines.append("=" * 78)
    lines.append("EEG ORDERING ABLATION — RANKING REPORT (A1: CAR before ICA vs A2: ICA before CAR)")
    lines.append("=" * 78)
    lines.append(f"Subjects tested: {sorted(df['subject'].unique().tolist())}")
    lines.append(f"Variants tested: {VARIANTS}")
    lines.append("")

    summary = df.groupby("variant").agg(
        mean_still_bad_count=("still_bad_after_cleaning", lambda s: np.mean([len(x) for x in s])),
        mean_max_variance_ratio_after=("max_variance_ratio_after", "mean"),
        mean_emg_band_power_after=("median_emg_band_power_after_30_49hz", "mean"),
        mean_eeg_band_retention=("median_eeg_band_power_retention_ratio_4_30hz", "mean"),
        mean_interchannel_corr=("mean_abs_interchannel_correlation", "mean"),
        mean_ica_excluded=("n_ica_components_excluded", "mean"),
        mean_autoreject_reject_fraction=("autoreject_reject_fraction", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    ).reset_index()

    lines.append("--- Per-variant averages across all test subjects ---")
    lines.append(summary.to_string(index=False))
    lines.append("")

    lines.append("--- How to read this ---")
    lines.append("mean_still_bad_count            : LOWER is better (channels still >4x median")
    lines.append("                                   variance after cleaning).")
    lines.append("mean_max_variance_ratio_after    : LOWER is better, closer to 1-3 is healthy.")
    lines.append("mean_emg_band_power_after        : LOWER is better (less residual muscle noise),")
    lines.append("                                   check against eeg_band_retention for over-cleaning.")
    lines.append("mean_eeg_band_retention          : should stay close between A1/A2 -- large gap =")
    lines.append("                                   one ordering is destroying more real EEG signal.")
    lines.append("mean_interchannel_corr           : this is the ordering-specific check. If ICA runs")
    lines.append("                                   BEFORE CAR (A2), CAR afterwards re-introduces a")
    lines.append("                                   common signal that ICA never saw/could remove --")
    lines.append("                                   watch for A2 showing unusually different (often")
    lines.append("                                   higher, less physiologically structured) correlation")
    lines.append("                                   than A1.")
    lines.append("mean_autoreject_reject_fraction  : LOWER is better -- fraction of 3s pseudo-epochs")
    lines.append("                                   AutoReject would still flag as bad after cleaning.")
    lines.append("")

    a1 = summary[summary["variant"] == "A1_car_then_ica"]
    a2 = summary[summary["variant"] == "A2_ica_then_car"]

    lines.append("--- Automatic note (heuristic, please confirm visually) ---")
    if len(a1) and len(a2):
        a1_row, a2_row = a1.iloc[0], a2.iloc[0]
        reasons_a1 = []
        reasons_a2 = []
        if a1_row["mean_emg_band_power_after"] < a2_row["mean_emg_band_power_after"]:
            reasons_a1.append("lower residual EMG-band power")
        else:
            reasons_a2.append("lower residual EMG-band power")
        if a1_row["mean_autoreject_reject_fraction"] < a2_row["mean_autoreject_reject_fraction"]:
            reasons_a1.append("fewer epochs still flagged bad by AutoReject")
        else:
            reasons_a2.append("fewer epochs still flagged bad by AutoReject")
        if abs(a1_row["mean_interchannel_corr"] - a2_row["mean_interchannel_corr"]) > 0.05:
            worse = "A2_ica_then_car" if a2_row["mean_interchannel_corr"] > a1_row["mean_interchannel_corr"] else "A1_car_then_ica"
            lines.append(f"NOTE: inter-channel correlation differs notably between orderings "
                         f"(diff={abs(a1_row['mean_interchannel_corr']-a2_row['mean_interchannel_corr']):.3f}); "
                         f"{worse} shows the higher value, which can indicate CAR reintroducing shared "
                         f"noise that ICA (run before CAR) never had the chance to remove. Inspect this "
                         f"ordering's QC images closely.")
            lines.append("")
        lines.append(f"A1_car_then_ica favoured by: {reasons_a1 if reasons_a1 else 'no metric'}")
        lines.append(f"A2_ica_then_car favoured by: {reasons_a2 if reasons_a2 else 'no metric'}")
    lines.append("")
    lines.append("IMPORTANT: as with the previous ablation test, this is a starting point, not a")
    lines.append("final verdict -- open the QC images for both variants under:")
    lines.append(f"  {OUTPUT_ROOT}\\A1_car_then_ica\\")
    lines.append(f"  {OUTPUT_ROOT}\\A2_ica_then_car\\")
    lines.append("and compare them visually before deciding, the same way we did for the previous")
    lines.append("PyPREP ablation test.")
    lines.append("")

    lines.append("--- Per-subject detail ---")
    for sub in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == sub]
        lines.append(f"\n{sub}:")
        for _, row in sub_df.iterrows():
            lines.append(
                f"  {row['variant']:<18s} bads={row['bad_channels_detected']} "
                f"still_bad={row['still_bad_after_cleaning']} "
                f"ica_excluded={row['n_ica_components_excluded']} "
                f"emg_power={row['median_emg_band_power_after_30_49hz']:.3e} "
                f"eeg_retention={row['median_eeg_band_power_retention_ratio_4_30hz']:.3f} "
                f"interchannel_corr={row['mean_abs_interchannel_correlation']:.3f} "
                f"autoreject_reject_frac={row['autoreject_reject_fraction']}"
            )

    report_path = os.path.join(OUTPUT_ROOT, "ranking_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Saved ranking report: {report_path}")
    print("\n".join(lines))


def build_ranking_chart(df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ("still_bad_after_cleaning", "Channels still bad after cleaning (count, lower=better)",
         lambda s: s.apply(len)),
        ("max_variance_ratio_after", "Max channel variance ratio after cleaning (lower=better)", None),
        ("median_emg_band_power_after_30_49hz", "Residual EMG-band power 30-49Hz (lower=better)", None),
        ("median_eeg_band_power_retention_ratio_4_30hz",
         "EEG-band (4-30Hz) power retention ratio (~stable=better)", None),
        ("mean_abs_interchannel_correlation", "Mean |inter-channel correlation| (ordering check)", None),
        ("autoreject_reject_fraction", "Fraction of pseudo-epochs still flagged bad (lower=better)", None),
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
