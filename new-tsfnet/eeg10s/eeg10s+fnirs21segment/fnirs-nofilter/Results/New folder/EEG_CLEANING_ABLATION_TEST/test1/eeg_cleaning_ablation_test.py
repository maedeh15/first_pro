# ============================================================================
# EEG CLEANING ABLATION TEST — compares candidate bad-channel/ICA/epoch-
# rejection COMBINATIONS on a small subset of subjects, to decide which
# combination becomes the FINAL pipeline for the full 26-subject dataset.
#
# This script does NOT modify your main pipeline
# (Preprocessing_TSFNet_paper_nback_v2_EEG10s_21seg_nofilt.py). It re-uses
# ONLY the pure data-loading logic from it (load_eeg_data, path/constant
# config) via direct import, so the .mat parsing / channel-count logic stays
# in exactly one place and can't drift out of sync between the two scripts.
#
# WHAT IS COMPARED (4 variants, all built on your existing baseline):
#   V0_baseline        : EXACTLY your current pipeline
#                        (filter -> CAR -> ICA[EOG-corr + ICLabel ocular])
#   V1_prep            : + PyPREP bad-channel detection/interpolation
#                        BEFORE CAR
#   V2_prep_autoreject : V1 + AutoReject on fixed-length pseudo-epochs
#                        (epoch-level residual-artifact cleanup, QC-only
#                        here — does not change how your real pipeline
#                        epochs trial windows, see NOTE in run_variant())
#   V3_prep_iclabel_all: V1 + ICLabel used to ALSO exclude muscle/channel-
#                        noise ICA components (not just ocular), to test
#                        whether the residual EMG noise seen in VP001's QC
#                        plots can be reduced without over-cleaning
#
# WHAT IS SAVED PER (subject x variant):
#   - the exact same 5 QC images your main pipeline already produces
#     ({subject}_eeg_before_after.png, _after_allchannels.png,
#      _after_allchannels_zoom10_30.png, _before_allchannels.png,
#      _before_allchannels_zoom10_30.png) so they can be visually compared
#     side-by-side with the ones you already reviewed with me.
#   - a per-subject-per-variant metrics JSON (variance/kurtosis per channel,
#     detected bad channels, ICA components excluded, EMG-band residual
#     power, EEG-band power retained, etc. -- see compute_metrics())
#
# WHAT IS SAVED GLOBALLY (after all subjects x variants finish):
#   - all_metrics.csv           one row per (subject, variant)
#   - ranking_report.txt        plain-language ranking + recommendation
#   - ranking_summary.png       bar chart comparing variants across subjects
#
# HOW TO RUN:
#   python eeg_cleaning_ablation_test.py
#   python eeg_cleaning_ablation_test.py --subjects 1,15,18,21,5 --workers 1
#
# REQUIREMENTS (in addition to what your main pipeline already needs):
#   pip install pyprep autoreject --break-system-packages
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
    """json.dump() cannot serialize numpy scalar types (np.int64, np.float64,
    np.bool_, np.ndarray, ...) that PyPREP/MNE/ICA frequently return mixed in
    with plain Python types (e.g. ica.find_bads_eog() indices, PyPREP's
    internal bad-channel bookkeeping). Passing this as json.dump(..., default=
    _json_default) converts any such value on the fly instead of crashing --
    this was the actual cause of V1/V2/V3 silently losing their VP005 results
    (the exception happened INSIDE run_variant(), so process_one_subject()'s
    except-block discarded the whole already-computed metrics dict along with
    it, even though PyPREP/ICA themselves had run successfully)."""
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
#    only -- we do NOT call its preprocess_eeg()/process_subject(), so this
#    script's cleaning logic is fully independent and safe to experiment on).
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
        "script, e.g. (Windows cmd):\n"
        '  set MAIN_PIPELINE_PATH=E:\\path\\to\\your\\script.py\n'
        "  python eeg_cleaning_ablation_test.py"
    )

_spec = importlib.util.spec_from_file_location("main_pipeline", MAIN_PIPELINE_PATH)
main_pipeline = importlib.util.module_from_spec(_spec)
# main_pipeline.py parses its own argv on import (argparse.parse_known_args),
# which is safe even when THIS script is run with different flags, since it
# uses parse_known_args (ignores unrecognized args) not parse_args.
_spec.loader.exec_module(main_pipeline)

load_eeg_data = main_pipeline.load_eeg_data
BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
SFREQ_EEG = main_pipeline.SFREQ_EEG
EEG_BP_LOW = main_pipeline.EEG_BP_LOW
EEG_BP_HIGH = main_pipeline.EEG_BP_HIGH
EEG_BP_ORDER = main_pipeline.EEG_BP_ORDER

# ----------------------------------------------------------------------------
# 2) CONFIG
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# EDIT THESE TWO LINES DIRECTLY WHEN RUNNING FROM JUPYTER (a notebook kernel
# passes its OWN argv, e.g. "-f /path/to/kernel.json", so relying on
# command-line flags here would break/misparse in Jupyter). When running this
# file from an actual terminal instead, --subjects/--workers below still work
# too (parse_known_args ignores Jupyter's/anything else's extra argv safely
# either way, so this file works unmodified in BOTH environments).
# ----------------------------------------------------------------------------
TEST_SUBJECTS_DEFAULT = "1,15,18,21,5"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument(
    "--subjects", type=str, default=TEST_SUBJECTS_DEFAULT,
    help="Comma-separated subject numbers for the ablation test (default: "
         "1,15,18,21,5 -- the 4 subjects with known bad channels/EMG noise "
         "identified during manual QC review, plus one presumed-clean subject "
         "as a control against over-cleaning/false-positive bad-channel calls).",
)
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT,
                      help="Parallel subject workers. Keep at 1 unless you have "
                           "verified RAM headroom -- ICA + PyPREP RANSAC together "
                           "are memory-heavy per subject. STRONGLY recommended to "
                           "leave at 1 when running from Jupyter -- multiprocessing "
                           "Pool + Jupyter kernels on Windows is a common source of "
                           "silent hangs/pickling errors.")
# parse_known_args (not parse_args): ignores Jupyter's own "-f kernel.json"
# argv instead of erroring out on it, so this cell/script runs unmodified in
# both a plain terminal AND a Jupyter kernel.
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = [int(s.strip()) for s in _args.subjects.split(",")]
N_WORKERS = _args.workers

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_CLEANING_ABLATION_TEST")
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LOG_PATH = os.path.join(OUTPUT_ROOT, "ablation_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# Channels independently flagged as bad-by-eye during our manual QC review,
# used ONLY to sanity-check that PyPREP's automatic detection actually finds
# them (see compute_metrics() -> "known_bad_recall"). This does NOT feed back
# into the cleaning itself -- it's a ground-truth check for THIS ablation
# test only.
KNOWN_BAD_CHANNELS = {
    1: [],                       # VP001: no full bad channel, EMG-type noise instead
    15: ["Cz"],
    18: ["Cz", "POz", "O2"],
    21: ["CP2"],
    5: [],                       # presumed clean control subject; adjust if your own
                                  # QC review found something for VP005
}

VARIANTS = ["V0_baseline", "V1_prep", "V2_prep_autoreject", "V3_prep_iclabel_all"]


# ----------------------------------------------------------------------------
# 3) SHARED QC PLOTTING (copied verbatim in spirit from your main pipeline's
#    preprocess_eeg() so the images are pixel-for-pixel comparable to the
#    ones you already reviewed)
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
# 4) METRICS  (the quantitative "did we remove noise without destroying
#    signal" check -- this is what the visual QC alone cannot tell you)
# ----------------------------------------------------------------------------
def _bandpower(data, sfreq, fmin, fmax):
    """Average PSD power (Welch) in [fmin, fmax] Hz, per channel."""
    psd, freqs = mne.time_frequency.psd_array_welch(
        data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=int(sfreq * 4),
        verbose=False,
    )
    return psd.mean(axis=1)  # (n_channels,)


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

    # EMG-band proxy: 30-49 Hz power (top of the passband, where muscle
    # activity dominates and true cortical EEG contributes comparatively
    # little) -- lower is better (less residual EMG), but NOT if it comes at
    # the cost of also killing the EEG-band power below.
    emg_power_after = _bandpower(data_after, sfreq, 30, 49)

    # Physiological EEG-band proxy: 4-30 Hz (theta+alpha+beta, where the
    # task-relevant n-back signal is expected to live) -- this should be
    # RETAINED, not reduced, between before/after. We report it as a ratio
    # (after/before) per channel; ~1.0 = fully preserved, <<1 = signal loss,
    # note some reduction IS expected/correct here since 'before' still
    # contains blink/EMG contamination riding on top of the true EEG band.
    eeg_power_before = _bandpower(data_before, sfreq, 4, 30)
    eeg_power_after = _bandpower(data_after, sfreq, 4, 30)
    eeg_power_ratio = eeg_power_after / (eeg_power_before + 1e-20)

    # Outlier-variance channels remaining AFTER cleaning: a channel whose
    # variance is >4x the median across channels is still behaving like an
    # unaddressed bad channel.
    median_var_after = np.median(var_after)
    still_bad_mask = var_after > 4 * median_var_after
    still_bad_channels = [ch_names[i] for i in range(len(ch_names)) if still_bad_mask[i]]

    known_bad = set(KNOWN_BAD_CHANNELS.get(subject_num, []))
    detected_set = set(bad_channels_detected)
    known_bad_recall = (
        len(known_bad & detected_set) / len(known_bad) if known_bad else None
    )

    metrics = {
        "subject": subject_id,
        "variant": variant_name,
        "runtime_s": round(runtime_s, 1),
        "n_channels_used": len(ch_names),
        "bad_channels_detected": sorted(bad_channels_detected),
        "n_bad_channels_detected": len(bad_channels_detected),
        "known_bad_channels": sorted(known_bad),
        "known_bad_recall": known_bad_recall,   # 1.0 = found all manually-identified
                                                  # bad channels, 0.0 = found none, None
                                                  # = subject had no known-bad channel
        "n_ica_components_excluded": n_ica_excluded,
        "ica_excluded_ids": ica_excluded_ids,
        "still_bad_after_cleaning": still_bad_channels,   # SHOULD be empty/short --
                                                            # non-empty = cleaning failed
                                                            # for that channel
        "median_variance_before": float(np.median(var_before)),
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
                                                            # closer to 1 = more uniform
                                                            # across channels = healthier
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
                                                            # closer to expected physiological
                                                            # range (not near-zero) = less
                                                            # over-cleaning / info loss
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) THE 4 PIPELINE VARIANTS
# ----------------------------------------------------------------------------
def _base_filter_and_ica(raw, subject_id, use_iclabel_muscle=False):
    """Shared filter -> (CAR happens in caller, AFTER optional PyPREP step)
    -> ICA logic used by every variant, factored out so V0..V3 differ ONLY
    in the bad-channel/epoch-rejection steps wrapped around this core."""
    filtered = raw.copy()
    filtered.filter(
        l_freq=EEG_BP_LOW, h_freq=EEG_BP_HIGH, method="iir",
        iir_params=dict(order=EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    return filtered


def _run_car_and_ica(filtered, raw_before_ref, subject_id, use_iclabel_muscle=False):
    filtered.set_eeg_reference("average", projection=False, verbose=False)

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
        ocular_labels = ("eye blink", "eye movement")
        extra_labels = ("muscle artifact", "channel noise") if use_iclabel_muscle else ()
        wanted_labels = ocular_labels + extra_labels
        iclabel_flagged = [i for i, lab in enumerate(ic_labels["labels"]) if lab in wanted_labels]
    except Exception:
        iclabel_flagged = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_flagged))
    if ica.exclude:
        ica.apply(filtered, verbose=False)

    return filtered, ica.exclude


def _run_pyprep(raw_filtered_no_car, subject_id):
    """Detect + interpolate bad channels BEFORE CAR. Returns the cleaned raw
    and the list of channel names that were flagged bad."""
    from pyprep import NoisyChannels

    tmp = raw_filtered_no_car.copy()
    nd = NoisyChannels(tmp, random_state=42)
    try:
        nd.find_all_bads(ransac=True)
    except Exception as e:
        log(f"  {subject_id}: PyPREP RANSAC failed ({e}); retrying without RANSAC "
            f"(slower-but-safer fallback, still uses deviation/HF-noise/correlation).")
        nd.find_all_bads(ransac=False)

    bads = nd.get_bads()
    tmp.info["bads"] = bads
    if bads:
        tmp.interpolate_bads(reset_bads=True, verbose=False)
    return tmp, bads


def _run_autoreject_qc_epochs(filtered, subject_id):
    """AutoReject needs epochs. Since your real pipeline windows trials
    separately downstream, we build FIXED-LENGTH 3s pseudo-epochs here
    PURELY to let AutoReject flag/interpolate residual per-epoch artifacts
    for the QC metrics of this ablation test. We do NOT feed these epochs
    back into extract_windows() -- this function only reports how much
    AutoReject WOULD reject, as a diagnostic signal for choosing the winning
    variant; wiring AutoReject into the real trial-windowing step (if V2/V3
    wins) is a separate follow-up change to the main pipeline, not done
    here."""
    from autoreject import AutoReject

    events = mne.make_fixed_length_events(filtered, duration=3.0)
    epochs = mne.Epochs(filtered, events, tmin=0, tmax=3.0, baseline=None,
                         preload=True, verbose=False, picks="eeg")
    if len(epochs) < 5:
        log(f"  {subject_id}: too few pseudo-epochs ({len(epochs)}) for AutoReject, skipping.")
        return filtered, 0, len(epochs)

    ar = AutoReject(random_state=42, verbose=False, n_jobs=1)
    epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)
    n_rejected = int(reject_log.bad_epochs.sum())
    return filtered, n_rejected, len(epochs)


def run_variant(variant_name, subject_num, raw, events, mnt_struct):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    raw_before = raw.copy()

    if variant_name == "V0_baseline":
        filtered = _base_filter_and_ica(raw, subject_id)
        filtered, excluded = _run_car_and_ica(filtered, raw_before, subject_id)
        bads_detected = []
        n_rejected_epochs, n_total_epochs = None, None

    elif variant_name == "V1_prep":
        filtered = _base_filter_and_ica(raw, subject_id)
        filtered, bads_detected = _run_pyprep(filtered, subject_id)
        filtered, excluded = _run_car_and_ica(filtered, raw_before, subject_id)
        n_rejected_epochs, n_total_epochs = None, None

    elif variant_name == "V2_prep_autoreject":
        filtered = _base_filter_and_ica(raw, subject_id)
        filtered, bads_detected = _run_pyprep(filtered, subject_id)
        filtered, excluded = _run_car_and_ica(filtered, raw_before, subject_id)
        filtered, n_rejected_epochs, n_total_epochs = _run_autoreject_qc_epochs(filtered, subject_id)

    elif variant_name == "V3_prep_iclabel_all":
        filtered = _base_filter_and_ica(raw, subject_id)
        filtered, bads_detected = _run_pyprep(filtered, subject_id)
        filtered, excluded = _run_car_and_ica(filtered, raw_before, subject_id,
                                               use_iclabel_muscle=True)
        n_rejected_epochs, n_total_epochs = None, None

    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    runtime_s = time.time() - t0

    qc_dir = os.path.join(OUTPUT_ROOT, variant_name, subject_id)
    save_qc_plots(raw_before, filtered, subject_id, variant_name, qc_dir)

    metrics = compute_metrics(subject_num, variant_name, raw_before, filtered,
                               bads_detected, len(excluded), list(excluded), runtime_s)
    if n_rejected_epochs is not None:
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
        # IMPORTANT: a serialization failure here must NOT discard the metrics
        # dict itself (that was the VP005 bug) -- log it and keep going, the
        # caller still gets `metrics` back for all_metrics.csv/ranking_report.txt.
        log(f"  {subject_id} [{variant_name}]: WARNING could not save metrics JSON "
            f"({e}); metrics were still computed and will appear in all_metrics.csv.")

    log(f"  {subject_id} [{variant_name}] done in {runtime_s:.1f}s | "
        f"bads={bads_detected} | ICA excluded={len(excluded)} | "
        f"still_bad_after={metrics['still_bad_after_cleaning']}")

    return metrics


# ----------------------------------------------------------------------------
# 6) MAIN LOOP (subject x variant)
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
            m = run_variant(variant_name, subject_num, raw, events, mnt_struct)
            all_metrics.append(m)
        except Exception as e:
            log(f"FAILED variant {variant_name} for {subject_id}: {e}")
            traceback.print_exc()
    return all_metrics


def main():
    log("===== EEG CLEANING ABLATION TEST STARTED =====")
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
    log("===== EEG CLEANING ABLATION TEST FINISHED =====")


# ----------------------------------------------------------------------------
# 7) RANKING / FINAL TEXT REPORT
# ----------------------------------------------------------------------------
def build_ranking_report(df):
    lines = []
    lines.append("=" * 78)
    lines.append("EEG CLEANING ABLATION — RANKING REPORT")
    lines.append("=" * 78)
    lines.append(f"Subjects tested: {sorted(df['subject'].unique().tolist())}")
    lines.append(f"Variants tested: {VARIANTS}")
    lines.append("")

    summary = df.groupby("variant").agg(
        mean_still_bad_count=("still_bad_after_cleaning", lambda s: np.mean([len(x) for x in s])),
        mean_known_bad_recall=("known_bad_recall", lambda s: s.dropna().mean() if s.notna().any() else np.nan),
        mean_max_variance_ratio_after=("max_variance_ratio_after", "mean"),
        mean_emg_band_power_after=("median_emg_band_power_after_30_49hz", "mean"),
        mean_eeg_band_retention=("median_eeg_band_power_retention_ratio_4_30hz", "mean"),
        mean_ica_excluded=("n_ica_components_excluded", "mean"),
        mean_runtime_s=("runtime_s", "mean"),
    ).reset_index()

    lines.append("--- Per-variant averages across all test subjects ---")
    lines.append(summary.to_string(index=False))
    lines.append("")

    lines.append("--- How to read this (lower/higher = better, per column) ---")
    lines.append("mean_still_bad_count           : LOWER is better. Channels still >4x median")
    lines.append("                                  variance AFTER cleaning = cleaning failed for")
    lines.append("                                  that channel. Should be ~0 for a good variant.")
    lines.append("mean_known_bad_recall          : HIGHER is better (max 1.0). Fraction of the")
    lines.append("                                  manually-identified bad channels (Cz/POz/O2/CP2)")
    lines.append("                                  that were automatically caught.")
    lines.append("mean_max_variance_ratio_after  : LOWER is better, closer to 1-3 is healthy.")
    lines.append("                                  Very high = one channel still dominates.")
    lines.append("mean_emg_band_power_after      : LOWER is better = less residual muscle noise")
    lines.append("                                  (30-49Hz), BUT check against eeg_band_retention")
    lines.append("                                  below -- if that also crashed, the variant is")
    lines.append("                                  over-cleaning, not genuinely fixing EMG.")
    lines.append("mean_eeg_band_retention         : Should stay reasonably close across variants.")
    lines.append("                                  A variant that is MUCH lower than V0_baseline's")
    lines.append("                                  value here is destroying real EEG signal, even")
    lines.append("                                  if its other numbers look 'better'.")
    lines.append("")

    # --- Automatic recommendation heuristic -------------------------------
    baseline_row = summary[summary["variant"] == "V0_baseline"]
    baseline_eeg_retention = (
        baseline_row["mean_eeg_band_retention"].values[0] if len(baseline_row) else np.nan
    )

    def info_preserved(row):
        # Flag variants that lose >15% of the baseline's EEG-band power on
        # average as likely over-cleaning (arbitrary but transparent
        # threshold -- adjust if your own judgment differs after reviewing
        # the QC images).
        if np.isnan(baseline_eeg_retention) or baseline_eeg_retention == 0:
            return True
        return row["mean_eeg_band_retention"] >= 0.85 * baseline_eeg_retention

    candidates = summary[summary.apply(info_preserved, axis=1)].copy()
    if candidates.empty:
        candidates = summary.copy()
        lines.append("WARNING: every variant lost >15% EEG-band power vs baseline on average -- "
                      "the 0.85 threshold above may be too strict for your data, or the true "
                      "explanation is that bad-channel interpolation itself legitimately changes "
                      "band power (since interpolated channels stop contributing independent "
                      "noise). Re-check manually before trusting the automatic pick below.")
        lines.append("")

    # Among variants that preserved info, prefer the one with lowest
    # still_bad_count, then highest known_bad_recall, then lowest EMG power.
    candidates = candidates.sort_values(
        by=["mean_still_bad_count", "mean_known_bad_recall", "mean_emg_band_power_after"],
        ascending=[True, False, True],
    )
    winner = candidates.iloc[0]["variant"] if len(candidates) else "V0_baseline"

    lines.append("--- Automatic recommendation (heuristic, please confirm visually) ---")
    lines.append(f"Suggested winning variant: {winner}")
    lines.append("")
    lines.append("This pick is based on: (1) fewest channels still behaving badly after")
    lines.append("cleaning, (2) highest recall of your manually-identified bad channels, (3)")
    lines.append("lowest residual EMG-band power -- AMONG variants that did not lose more than")
    lines.append("~15% of baseline EEG-band power (i.e. did not over-clean).")
    lines.append("")
    lines.append("IMPORTANT: this is a starting point, not a final verdict. Before committing")
    lines.append("this variant to the full 26-subject run, open the QC images for all 5 test")
    lines.append(f"subjects under:\n  {OUTPUT_ROOT}\\{winner}\\")
    lines.append("and visually confirm (same way we reviewed VP001/VP015/VP018/VP021 together)")
    lines.append("that the previously-bad channels (Cz, POz, O2, CP2) now look like normal EEG,")
    lines.append("and that the presumed-clean control subject (VP005 by default) was NOT altered")
    lines.append("in a way that removes real brain signal.")
    lines.append("")

    lines.append("--- Per-subject detail ---")
    for sub in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == sub]
        lines.append(f"\n{sub}:")
        for _, row in sub_df.iterrows():
            lines.append(
                f"  {row['variant']:<20s} bads={row['bad_channels_detected']} "
                f"still_bad={row['still_bad_after_cleaning']} "
                f"known_bad_recall={row['known_bad_recall']} "
                f"ica_excluded={row['n_ica_components_excluded']} "
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
        ("max_variance_ratio_after", "Max channel variance ratio after cleaning (lower=better)", None),
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
        ax.tick_params(axis="x", rotation=30, labelsize=8)

    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_ROOT, "ranking_summary.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    log(f"Saved ranking chart: {chart_path}")


if __name__ == "__main__":
    main()
