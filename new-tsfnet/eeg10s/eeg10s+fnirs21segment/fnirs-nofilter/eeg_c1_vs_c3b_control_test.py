# ============================================================================
# EEG C1-vs-C3b CONTROL TEST — minimal script, only 2 variants, to answer ONE
# question: did C3's earlier apparent success on VP025's persistent Fp1/Fp2
# noise come from its "keep only ICLabel='brain' components" ICA method, or
# just from its narrower 0.1-30Hz filter band cutting off the very 30-49Hz
# range our EMG-noise metric measures?
#
#   C1_our_final_pipeline         : filter(0.5-50Hz) -> PyPREP -> CAR ->
#                                    ICA (find_bads_eog + ICLabel-argmax
#                                    ocular only) -> epoch rejection
#                                    (our already-validated pipeline)
#
#   C3b_bemobil_keep_brain_c1band : filter(0.5-50Hz) -> PyPREP -> CAR ->
#                                    ICA + ICLabel, KEEP ONLY 'brain'-labeled
#                                    components -> epoch rejection
#                                    (C3's exact artifact-removal method,
#                                    but with C1's filter band instead of
#                                    C3's own 0.1-30Hz -- this isolates the
#                                    ICA method as the single variable left)
#
# INTERPRETING THE RESULT (see "VP025's Fp1/Fp2" section of the report):
#   - If C3b ALSO shows still_bad=[] for VP025 (like C3 did) -> the
#     keep-brain-only method itself is genuinely effective, independent of
#     filter band.
#   - If C3b shows still_bad=['Fp1','Fp2'] again (like C1 did) -> C3's
#     earlier apparent success was an artifact of its narrower filter band,
#     not the ICA method.
#
# WHAT IS SAVED: same 5 QC images per (subject, variant), a per-
# (subject,variant) metrics JSON, and global all_metrics.csv /
# ranking_report.txt / ranking_summary.png under
# Results/EEG_C1_VS_C3B_CONTROL_TEST/.
#
# HOW TO RUN:
#   python eeg_c1_vs_c3b_control_test.py
#   python eeg_c1_vs_c3b_control_test.py --subjects 25,1 --workers 1
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
_spec.loader.exec_module(main_pipeline)

load_eeg_data = main_pipeline.load_eeg_data
BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
EEG_BP_ORDER = main_pipeline.EEG_BP_ORDER

# ----------------------------------------------------------------------------
# 2) CONFIG (edit directly for Jupyter; --flags work from terminal too)
# ----------------------------------------------------------------------------
# VP025 is the actual test case. VP001 is included as a quick sanity check
# (it's the subject where C3's full version caused a catastrophic 97.3%
# epoch-rejection rate -- worth re-checking here too, since it's cheap to
# include and tells us whether that problem was ALSO filter-band-related).
TEST_SUBJECTS_DEFAULT = "25,1"
N_WORKERS_DEFAULT = 1

_parser = argparse.ArgumentParser()
_parser.add_argument("--subjects", type=str, default=TEST_SUBJECTS_DEFAULT)
_parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
_args, _unknown = _parser.parse_known_args()

TEST_SUBJECTS = [int(s.strip()) for s in _args.subjects.split(",")]
N_WORKERS = _args.workers

OUTPUT_ROOT = os.path.join(os.path.dirname(MAIN_PIPELINE_PATH), "Results",
                            "EEG_C1_VS_C3B_CONTROL_TEST")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_ROOT, "ablation_log.txt")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


VARIANTS = ["C1_our_final_pipeline", "C3b_bemobil_keep_brain_c1band"]

# BOTH variants use C1's filter band -- that is the entire point of this
# control test (isolate the ICA method as the only remaining variable).
FILTER_BAND = (0.5, 50.0)

KNOWN_PERSISTENT_NOISE_CHANNELS = {
    25: ["Fp1", "Fp2"],
}


# ----------------------------------------------------------------------------
# 3) QC PLOTTING
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

    emg_power_after = _bandpower(data_after, sfreq, 30, 49)  # both variants use
                                                                # the same 0.5-50Hz
                                                                # band here, so this
                                                                # is now an apples-
                                                                # to-apples comparison
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
        "bad_channels_detected": sorted(bad_channels_detected),
        "n_ica_components_excluded": n_ica_excluded,
        "ica_excluded_ids": list(ica_excluded_ids),
        "still_bad_after_cleaning": still_bad_channels,
        "known_persistent_noise_channels": sorted(known_target),
        "known_persistent_still_present": sorted(known_target & set(still_bad_channels)),
        "target_channel_emg_band_power": target_ch_emg_power,
        "median_variance_after": float(np.median(var_after)),
        "max_variance_ratio_after": float(var_after.max() / (median_var_after + 1e-20)),
        "median_emg_band_power_after_30_49hz": float(np.median(emg_power_after)),
        "median_eeg_band_power_retention_ratio_4_30hz": float(np.median(eeg_power_ratio)),
    }
    return metrics


# ----------------------------------------------------------------------------
# 5) SHARED STEPS
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


# ----------------------------------------------------------------------------
# 6) THE 2 ICA METHODS
# ----------------------------------------------------------------------------
def _method_C1_ica_argmax_ocular(filtered, raw_before, subject_id):
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


def _method_C3b_keep_brain_only(filtered, raw_before, subject_id):
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


def run_variant(variant_name, subject_num, raw):
    subject_id = f"VP{subject_num:03d}"
    t0 = time.time()
    raw_before = raw.copy()

    l_freq, h_freq = FILTER_BAND
    filtered = _filter_only(raw, l_freq, h_freq)
    filtered, bads_detected = _run_pyprep(filtered, subject_id)
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    if variant_name == "C1_our_final_pipeline":
        filtered, excluded = _method_C1_ica_argmax_ocular(filtered, raw_before, subject_id)
    elif variant_name == "C3b_bemobil_keep_brain_c1band":
        filtered, excluded = _method_C3b_keep_brain_only(filtered, raw_before, subject_id)
    else:
        raise ValueError(f"Unknown variant: {variant_name}")

    n_rejected_epochs, n_total_epochs = _run_autoreject_qc_epochs(filtered, subject_id)
    runtime_s = time.time() - t0

    qc_dir = os.path.join(OUTPUT_ROOT, variant_name, subject_id)
    save_qc_plots(raw_before, filtered, subject_id, variant_name, qc_dir)

    metrics = compute_metrics(subject_num, variant_name, raw_before, filtered,
                               bads_detected, len(excluded), excluded, runtime_s)
    metrics["filter_band_hz"] = list(FILTER_BAND)
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
        f"bads={bads_detected} | excluded={len(excluded)} | "
        f"still_bad_after={metrics['still_bad_after_cleaning']} | "
        f"known_persistent_still_present={metrics['known_persistent_still_present']} | "
        f"autoreject_reject_frac={metrics['autoreject_reject_fraction']}")

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
    log("===== C1 vs C3b CONTROL TEST STARTED (same filter band, only ICA method differs) =====")
    log(f"Test subjects: {TEST_SUBJECTS}")
    log(f"Variants: {VARIANTS}")
    log(f"Filter band (BOTH variants): {FILTER_BAND}")
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
    log("===== C1 vs C3b CONTROL TEST FINISHED =====")


# ----------------------------------------------------------------------------
# 8) REPORT (short and to the point -- this is a yes/no control test)
# ----------------------------------------------------------------------------
def build_ranking_report(df):
    lines = []
    lines.append("=" * 78)
    lines.append("C1 vs C3b CONTROL TEST — both use filter band 0.5-50Hz.")
    lines.append("Only difference: ICA component-selection method.")
    lines.append("=" * 78)
    lines.append("")

    lines.append("--- THE ANSWER (VP025's Fp1/Fp2) ---")
    vp025 = df[df["subject"] == "VP025"]
    for variant in VARIANTS:
        row = vp025[vp025["variant"] == variant]
        if len(row):
            row = row.iloc[0]
            lines.append(f"  {variant:<32s} still_bad={row['still_bad_after_cleaning']} "
                         f"known_persistent_still_present={row['known_persistent_still_present']} "
                         f"target_channel_emg_power={row['target_channel_emg_band_power']} "
                         f"n_excluded={row['n_ica_components_excluded']}")
    lines.append("")
    lines.append("If C3b's known_persistent_still_present is EMPTY -> keep-brain-only ICA is")
    lines.append("genuinely effective, independent of filter band -- consider adopting it.")
    lines.append("If C3b's known_persistent_still_present still shows Fp1/Fp2 (like C1) ->")
    lines.append("C3's earlier apparent success on VP025 was solely due to its narrower")
    lines.append("0.1-30Hz filter band cutting off the 30-49Hz band our metric measures, NOT")
    lines.append("the keep-brain-only ICA method itself. In that case, do not adopt C3/C3b --")
    lines.append("stick with C1, and VP025's Fp1/Fp2 remains a case needing the manual")
    lines.append("decision discussed earlier (e.g. dropping the whole subject).")
    lines.append("")

    lines.append("--- Sanity check on VP001 (catastrophic 97.3% epoch-rejection under full C3) ---")
    vp001 = df[df["subject"] == "VP001"]
    for variant in VARIANTS:
        row = vp001[vp001["variant"] == variant]
        if len(row):
            row = row.iloc[0]
            lines.append(f"  {variant:<32s} autoreject_reject_frac={row['autoreject_reject_fraction']} "
                         f"n_excluded={row['n_ica_components_excluded']} "
                         f"eeg_retention={row['median_eeg_band_power_retention_ratio_4_30hz']:.3f}")
    lines.append("")
    lines.append("If C3b's autoreject_reject_frac for VP001 is still very high (close to C3's")
    lines.append("0.973), the keep-brain-only method is inherently too aggressive regardless of")
    lines.append("filter band, and should not be adopted even if it happened to help VP025.")
    lines.append("")

    lines.append("--- Full detail ---")
    for sub in sorted(df["subject"].unique()):
        sub_df = df[df["subject"] == sub]
        lines.append(f"\n{sub}:")
        for variant in VARIANTS:
            row = sub_df[sub_df["variant"] == variant]
            if len(row):
                row = row.iloc[0]
                lines.append(
                    f"  {variant:<32s} bads={row['bad_channels_detected']} "
                    f"still_bad={row['still_bad_after_cleaning']} "
                    f"n_excluded={row['n_ica_components_excluded']} "
                    f"emg_power={row['median_emg_band_power_after_30_49hz']:.3e} "
                    f"eeg_retention={row['median_eeg_band_power_retention_ratio_4_30hz']:.3f} "
                    f"autoreject_reject_frac={row['autoreject_reject_fraction']}"
                )

    report_path = os.path.join(OUTPUT_ROOT, "ranking_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"Saved report: {report_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
