# ============================================================================
# EEG ARTIFACT-REMOVAL METHOD ABLATION TEST — compares our own
# already-validated final pipeline (C1) against the artifact-removal
# approach used by 3 published works found via GitHub search for
# cognitive-workload / n-back EEG preprocessing:
#
#   C1_our_final_pipeline : filter(0.5-50Hz) -> PyPREP -> CAR ->
#                            ICA (find_bads_eog + ICLabel-argmax ocular only)
#                            -> epoch-level rejection (diagnostic)
#                            This is the pipeline validated across the V1/V3,
#                            A1/A2, and B1/B2 ablation tests earlier.
#
#   C2_pyprep_filter_only  : filter(1-45Hz) -> PyPREP -> CAR -> NO ICA
#                            -> epoch-level rejection
#                            From: "Recurrent and Convolutional Neural
#                            Networks in Classification of EEG Signal for
#                            Guided Imagery and Mental Workload Detection"
#                            (arXiv:2405.16901). This paper uses PyPREP/
#                            RANSAC bad-channel interpolation + a 1-45Hz
#                            bandpass filter and does NOT report any
#                            component-based (ICA) ocular/muscle artifact
#                            removal step -- epoch-level handling is the
#                            only additional cleaning. Implemented here as:
#                            same PyPREP+CAR foundation as C1, but skipping
#                            the ICA step entirely, relying on epoch
#                            rejection alone for whatever ocular/muscle
#                            contamination remains.
#
#   C3_bemobil_keep_brain  : filter(0.1-30Hz) -> PyPREP -> CAR -> ICA ->
#                            ICLabel, KEEP ONLY components whose argmax
#                            label is 'brain' (exclude everything else:
#                            eye, muscle, heart, line noise, channel noise,
#                            other) -> epoch-level rejection
#                            From: "Manipulating Drivers' Mental Workload:
#                            ... N-Back Task ..." (arXiv:2405.18099), which
#                            follows the BeMoBil pipeline: AMICA + ICLabel,
#                            retaining only components most likely to be
#                            brain activity. We substitute Picard for AMICA
#                            (AMICA has no maintained open Python
#                            implementation; Picard is the standard
#                            MNE-native equivalent already used in C1),
#                            everything else follows the paper's described
#                            "keep only brain" logic exactly.
#
#   C4_ssp_autoreject       : filter(1-30Hz) -> PyPREP -> CAR -> SSP-based
#                            EOG projection (NOT ICA) -> AutoReject epoch
#                            rejection
#                            From: teanijarv/EEG-pyline, whose preprocessing
#                            notebooks explicitly use SSP (Signal Space
#                            Projection) for EOG artifact removal instead of
#                            ICA, followed by the AutoReject algorithm for
#                            epoch-level artifact rejection.
#
# DESIGN NOTE (read before interpreting results): each variant keeps the
# already-validated PyPREP-bad-channel + CAR foundation constant (since 4
# earlier independent ablation tests confirmed this step is necessary
# regardless of what comes after it). Only the OCULAR/MUSCLE ARTIFACT
# REMOVAL METHOD differs between variants, matching what each source
# actually reports for that step. Filter bands are kept as each source
# paper specifies, since changing the band is part of "the exact method"
# requested -- this does mean C2/C3/C4 differ from C1 in more than one
# dimension at once (band AND artifact-removal method), which is disclosed
# here rather than hidden, and the ranking report below reminds you of it.
#
# WHAT IS SAVED: same 5 QC images per (subject, variant), a per-
# (subject,variant) metrics JSON, and global all_metrics.csv /
# ranking_report.txt / ranking_summary.png under
# Results/EEG_METHOD_COMPARISON_ABLATION_TEST/.
#
# HOW TO RUN:
#   python eeg_method_comparison_ablation_test.py
#   python eeg_method_comparison_ablation_test.py --subjects 25,1,15,18,21,5 --workers 1
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
EEG_BP_ORDER = main_pipeline.EEG_BP_ORDER

# ----------------------------------------------------------------------------
# 2) CONFIG (edit directly for Jupyter; --flags work from terminal too)
# ----------------------------------------------------------------------------
TEST_SUBJECTS_DEFAULT = "25,1,15,18,21,5"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--subjects", type=str, default=TEST_SUBJECTS_DEFAULT)
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = [int(s.strip()) for s in _args.subjects.split(",")]
N_WORKERS = _args.workers

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_METHOD_COMPARISON_ABLATION_TEST")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_ROOT, "ablation_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


VARIANTS = ["C1_our_final_pipeline", "C2_pyprep_filter_only",
            "C3_bemobil_keep_brain", "C4_ssp_autoreject"]

# Filter bands per variant, matching each source's reported method
# (see module docstring for citations).
FILTER_BANDS = {
    "C1_our_final_pipeline": (0.5, 50.0),
    "C2_pyprep_filter_only": (1.0, 45.0),
    "C3_bemobil_keep_brain": (0.1, 30.0),
    "C4_ssp_autoreject": (1.0, 30.0),
}

KNOWN_PERSISTENT_NOISE_CHANNELS = {
    25: ["Fp1", "Fp2"],  # confirmed unresolved by PyPREP/ICA(argmax)/ICA(prob-thresh)/AutoReject
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

    emg_power_after = _bandpower(data_after, sfreq, 30, min(49, sfreq / 2 - 1))
    eeg_power_before = _bandpower(data_before, sfreq, 4, 30)
    eeg_power_after = _bandpower(data_after, sfreq, 4, 30)
    eeg_power_ratio = eeg_power_after / (eeg_power_before + 1e-20)

    median_var_after = np.median(var_after)
    still_bad_mask = var_after > 4 * median_var_after
    still_bad_channels = [ch_names[i] for i in range(len(ch_names)) if still_bad_mask[i]]

    known_target = set(KNOWN_PERSISTENT_NOISE_CHANNELS.get(subject_num, []))
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
        "known_persistent_still_present": sorted(known_target & set(still_bad_channels)),
        "target_channel_emg_band_power": target_ch_emg_power,
        "median_variance_before": float(np.median(var_before)),
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) SHARED STEPS (bad-channel + CAR foundation, identical for all variants)
# ----------------------------------------------------------------------------
def _filter_only(raw, l_freq, h_freq):
    filtered = raw.copy()
    filtered.filter(
        l_freq=l_freq, h_freq=h_freq, method="iir",
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


ARTIFACT_CLASSES_IDX = {
    "brain": 0, "muscle artifact": 1, "eye blink": 2,
    "heart beat": 3, "line noise": 4, "channel noise": 5, "other": 6,
}


# ----------------------------------------------------------------------------
# 6) THE 4 ARTIFACT-REMOVAL METHODS
# ----------------------------------------------------------------------------
def _method_C1_ica_argmax_ocular(filtered, raw_before, subject_id):
    """Our own already-validated method (find_bads_eog + ICLabel-argmax
    ocular only)."""
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


def _method_C2_no_ica(filtered, raw_before, subject_id):
    """arXiv:2405.16901 method: no component-based artifact removal at all
    -- PyPREP bad-channel handling is the only channel-level cleaning; ocular
    contamination is left to whatever epoch-level rejection catches later."""
    return filtered, []


def _method_C3_bemobil_keep_brain_only(filtered, raw_before, subject_id):
    """arXiv:2405.18099 / BeMoBil-style method: fit ICA, label with ICLabel,
    KEEP ONLY components classified as 'brain' (exclude everything else --
    eye, muscle, heart, line noise, channel noise, other), regardless of
    class."""
    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)

    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        exclude = [i for i, lab in enumerate(ic_labels["labels"]) if lab != "brain"]
    except Exception as e:
        log(f"  {subject_id}: ICLabel failed ({e}), keeping all components (no exclusion).")
        exclude = []

    ica.exclude = exclude
    if exclude:
        ica.apply(filtered, verbose=False)
    return filtered, exclude


def _method_C4_ssp_eog(filtered, raw_before, subject_id):
    """teanijarv/EEG-pyline method: SSP-based EOG projection instead of ICA.
    Uses mne.preprocessing.compute_proj_eog on the recorded EOG channel(s),
    then applies the resulting projector(s) to the EEG data. Returns an
    empty ica_excluded_ids list since this method is NOT component-based --
    the "n_projectors_applied" is logged separately for transparency."""
    eog_ch_used = None
    for cand in ("VEOG", "HEOG"):
        if cand in filtered.ch_names:
            eog_ch_used = cand
            break

    if eog_ch_used is None:
        log(f"  {subject_id}: no EOG channel found for SSP, skipping ocular correction.")
        return filtered, []

    try:
        projs, _ = mne.preprocessing.compute_proj_eog(
            raw_before, n_eeg=1, n_mag=0, n_grad=0,
            ch_name=eog_ch_used, reject=None, no_proj=True, verbose=False,
        )
    except Exception as e:
        log(f"  {subject_id}: compute_proj_eog failed ({e}), skipping SSP.")
        return filtered, []

    if not projs:
        log(f"  {subject_id}: compute_proj_eog found no usable EOG events, skipping SSP.")
        return filtered, []

    filtered.add_proj(projs)
    filtered.apply_proj()
    # Report projector count via the ica_excluded_ids slot for consistent
    # logging/metrics plumbing, even though these are SSP vectors, not ICA
    # component indices.
    return filtered, list(range(len(projs)))


def run_variant(variant_name, subject_num, raw):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    raw_before = raw.copy()

    l_freq, h_freq = FILTER_BANDS[variant_name]
    filtered = _filter_only(raw, l_freq, h_freq)
    filtered, bads_detected = _run_pyprep(filtered, subject_id)
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    if variant_name == "C1_our_final_pipeline":
        filtered, excluded = _method_C1_ica_argmax_ocular(filtered, raw_before, subject_id)
    elif variant_name == "C2_pyprep_filter_only":
        filtered, excluded = _method_C2_no_ica(filtered, raw_before, subject_id)
    elif variant_name == "C3_bemobil_keep_brain":
        filtered, excluded = _method_C3_bemobil_keep_brain_only(filtered, raw_before, subject_id)
    elif variant_name == "C4_ssp_autoreject":
        filtered, excluded = _method_C4_ssp_eog(filtered, raw_before, subject_id)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    n_rejected_epochs, n_total_epochs = _run_autoreject_qc_epochs(filtered, subject_id)
    runtime_s = time.time() - t0

    qc_dir = os.path.join(OUTPUT_ROOT, variant_name, subject_id)
    save_qc_plots(raw_before, filtered, subject_id, variant_name, qc_dir)

    metrics = compute_metrics(subject_num, variant_name, raw_before, filtered,
                               bads_detected, len(excluded), excluded, runtime_s)
    metrics["filter_band_hz"] = list(FILTER_BANDS[variant_name])
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
        f"band={FILTER_BANDS[variant_name]} | bads={bads_detected} | "
        f"excluded={len(excluded)} | still_bad_after={metrics['still_bad_after_cleaning']} | "
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
    log("===== EEG METHOD COMPARISON ABLATION TEST STARTED =====")
    log(f"Test subjects: {TEST_SUBJECTS} (VP025 first -- the targeted case)")
    log(f"Variants: {VARIANTS}")
    log(f"Filter bands: {FILTER_BANDS}")
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
    log("===== EEG METHOD COMPARISON ABLATION TEST FINISHED =====")


# ----------------------------------------------------------------------------
# 8) RANKING / FINAL TEXT REPORT
# ----------------------------------------------------------------------------
def build_ranking_report(df):
    lines = []
    lines.append("=" * 78)
    lines.append("EEG METHOD COMPARISON ABLATION — RANKING REPORT")
    lines.append("C1 (our validated pipeline) vs C2/C3/C4 (methods found via GitHub search)")
    lines.append("=" * 78)
    lines.append(f"Subjects tested: {sorted(df['subject'].unique().tolist())}")
    lines.append(f"Variants tested: {VARIANTS}")
    lines.append("")
    lines.append("REMINDER: C2/C3/C4 differ from C1 in BOTH filter band AND artifact-removal")
    lines.append("method (see script docstring) -- a difference here is not attributable")
    lines.append("purely to the artifact-removal method by itself. Use this test to see how")
    lines.append("each PUBLISHED METHOD AS A WHOLE performs on your data, not as a clean")
    lines.append("single-variable ablation.")
    lines.append("")

    summary = df.groupby("variant").agg(
        mean_still_bad_count=("still_bad_after_cleaning", lambda s: np.mean([len(x) for x in s])),
        mean_max_variance_ratio_after=("max_variance_ratio_after", "mean"),
        mean_emg_band_power_after=("median_emg_band_power_after_30_49hz", "mean"),
        mean_eeg_band_retention=("median_eeg_band_power_retention_ratio_4_30hz", "mean"),
        mean_n_excluded=("n_ica_components_excluded", "mean"),
        mean_autoreject_reject_fraction=("autoreject_reject_fraction", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    ).reset_index()
    summary["variant"] = pd.Categorical(summary["variant"], categories=VARIANTS, ordered=True)
    summary = summary.sort_values("variant")

    lines.append("--- Per-variant averages across all test subjects ---")
    lines.append(summary.to_string(index=False))
    lines.append("")

    lines.append("--- VP025's Fp1/Fp2 across all 4 methods (the case none of our own ---")
    lines.append("--- ablation tests could resolve) ---")
    vp025 = df[df["subject"] == "VP025"]
    for variant in VARIANTS:
        row = vp025[vp025["variant"] == variant]
        if len(row):
            row = row.iloc[0]
            lines.append(f"  {variant:<24s} still_bad={row['still_bad_after_cleaning']} "
                         f"known_persistent_still_present={row['known_persistent_still_present']} "
                         f"target_channel_emg_power={row['target_channel_emg_band_power']}")
    lines.append("")

    lines.append("--- How to read the rest ---")
    lines.append("mean_still_bad_count            : LOWER is better.")
    lines.append("mean_emg_band_power_after       : LOWER is better, but check eeg_band_retention.")
    lines.append("mean_eeg_band_retention         : large drops vs C1 = likely over-cleaning")
    lines.append("                                   (relevant for C3, which excludes ALL")
    lines.append("                                   non-brain components, not just ocular).")
    lines.append("mean_autoreject_reject_fraction : for C2 (no ICA at all), a HIGH value here is")
    lines.append("                                   expected/consistent -- it means blink/muscle")
    lines.append("                                   contamination is being caught downstream by")
    lines.append("                                   epoch rejection instead of ICA, at the cost")
    lines.append("                                   of discarding more trials outright.")
    lines.append("")

    lines.append("IMPORTANT: open the QC images for VP025 (and at least one other subject) under")
    for v in VARIANTS:
        lines.append(f"  {OUTPUT_ROOT}\\{v}\\VP025\\")
    lines.append("and visually confirm before adopting any variant over C1.")
    lines.append("")

    lines.append("--- Per-subject detail (all subjects) ---")
    for sub in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == sub]
        lines.append(f"\n{sub}:")
        for variant in VARIANTS:
            row = sub_df[sub_df["variant"] == variant]
            if len(row):
                row = row.iloc[0]
                lines.append(
                    f"  {variant:<24s} band={row['filter_band_hz']} "
                    f"bads={row['bad_channels_detected']} "
                    f"still_bad={row['still_bad_after_cleaning']} "
                    f"n_excluded={row['n_ica_components_excluded']} "
                    f"emg_power={row['median_emg_band_power_after_30_49hz']:.3e} "
                    f"eeg_retention={row['median_eeg_band_power_retention_ratio_4_30hz']:.3f} "
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
        ("n_ica_components_excluded", "Mean components/projectors excluded", None),
        ("autoreject_reject_fraction", "Fraction of pseudo-epochs still flagged bad (lower=better)", None),
    ]

    for ax, (col, title, transform) in zip(axes.flat, metrics_to_plot):
        vals = df[col] if transform is None else transform(df[col])
        plot_df = df.copy()
        plot_df["_val"] = vals
        grouped = plot_df.groupby("variant")["_val"].mean().reindex(VARIANTS)
        ax.bar(grouped.index, grouped.values, color="tab:blue")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", rotation=20, labelsize=7)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_ROOT, "ranking_summary.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    log(f"Saved ranking chart: {chart_path}")


if __name__ == "__main__":
    main()
