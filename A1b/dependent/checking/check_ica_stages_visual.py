# ============================================================================
# check_ica_stages_visual.py
#
# DIAGNOSTIC SCRIPT -- does NOT modify the main pipeline or any saved output.
#
# Purpose: apply the 3 ICA artifact-rejection layers PROGRESSIVELY, one at a
# time, to real EEG recordings, and save a complete visual record at each
# stage so the effect of each layer can be judged by eye:
#
#   STAGE 1: argmax only            (ICLabel's top-1 class is an artifact class)
#   STAGE 2: argmax + prob_threshold (adds: any artifact-class probability >= 0.30)
#   STAGE 3: full pipeline           (adds: VEOG/HEOG correlation, with a
#                                     brain-probability rescue gate)
#
# For EACH stage, for EACH subject, this script saves:
#   - one inspection figure (topomap + 60s time series + full-recording time
#     series + PSD + printed probability vector) for every IC that is
#     NEWLY excluded at this stage (i.e. was NOT excluded at the previous
#     stage) -- this isolates exactly what each additional layer is doing.
#   - one "all channels" plot of the EEG signal AFTER applying this stage's
#     cleaning, for the full recording (downsampled for display).
#   - one "all channels, zoomed 10-30s" plot, same signal, full resolution,
#     for close visual inspection of a representative window.
#
# Subjects processed: the noisy-suspect subjects flagged in your own earlier
# project reports (VP004, VP015, VP018, VP019, VP020), plus one subject with
# comparatively clean data (VP001) as a reference/contrast case.
#
# OUTPUT LAYOUT (created next to this script):
#   check-ica/
#     VP001/
#       stage1_argmax/
#         IC{n}_inspection.png   (one per newly-excluded IC this stage)
#         all_channels_full.png
#         all_channels_zoom_10-30s.png
#         probs_log.txt
#       stage2_argmax_prob/
#         ...
#       stage3_full/
#         ...
#     VP004/
#       ...
#     ... etc
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH below.
#   2. Run: python check_ica_stages_visual.py
#   3. Browse check-ica/{subject}/{stage}/ folders and compare progressively:
#      does each stage remove something visibly different in the all-channel
#      plots? Do the newly-excluded ICs at stage2/stage3 look like genuine
#      artifacts (per the topomap/time-series/PSD criteria) or borderline/
#      brain-like components?
# ============================================================================

import os
import numpy as np
import scipy.io as sio
import mne
import mne_icalabel
import matplotlib.pyplot as plt
from scipy.signal import welch
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THIS
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"

# noisy-suspect subjects (per your earlier project reports) + one clean reference subject
SUBJECT_IDS = ["VP004", "VP015", "VP018", "VP019", "VP020", "VP001"]

SFREQ_EEG = 200.0
ZOOM_START_SEC = 10.0
ZOOM_END_SEC = 30.0

ARTIFACT_CLASSES_IDX = {
    'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
    'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6
}
ALL_CLASS_NAMES = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
                    'line noise', 'channel noise', 'other']
ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
PROB_THRESHOLD = 0.30
BRAIN_PROB_CEILING = 0.5

OUTPUT_ROOT_NAME = "check-ica"


# ----------------------------------------------------------------------------
# Loading + preprocessing up to ICA fit (mirrors preprocess_eeg(), independent copy)
# ----------------------------------------------------------------------------
def load_eeg_raw(subject_id):
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt_struct = mat_cnt[real_keys[0]]

    data = cnt_struct.x.T
    sfreq = float(cnt_struct.fs)
    ch_names = [str(ch).strip() for ch in cnt_struct.clab]

    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6
    if sfreq != SFREQ_EEG:
        from scipy.signal import resample
        n_samples = int(data.shape[1] * SFREQ_EEG / sfreq)
        data = resample(data, n_samples, axis=1)
        sfreq = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    eog_channels = [ch for ch in ['VEOG', 'HEOG'] if ch in ch_names]
    if eog_channels:
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
    raw.set_montage('standard_1005', match_case=False, on_missing='ignore')
    return raw


def fit_ica(subject_id):
    raw = load_eeg_raw(subject_id)
    raw_original = raw.copy()   # kept unfiltered, for find_bads_eog() (matches main pipeline)

    raw_filtered = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    raw_filtered.filter(l_freq=1.0, h_freq=40.0, method='fir', verbose=False)

    raw_temp = raw_filtered.copy().filter(l_freq=10.0, h_freq=40.0, method='fir', verbose=False)
    eeg_picks = mne.pick_types(raw_temp.info, eeg=True, eog=False)
    eeg_names = [raw_temp.ch_names[idx] for idx in eeg_picks]
    temp_data = raw_temp.get_data(picks=eeg_picks)
    stds = np.std(temp_data, axis=1)
    median_std = np.median(stds)
    mad = np.median(np.abs(stds - median_std)) if np.median(np.abs(stds - median_std)) > 0 else 1e-6
    corr_matrix = np.corrcoef(temp_data)
    mean_corrs = np.mean(corr_matrix, axis=1)

    bad_channels = []
    for i, ch_name in enumerate(eeg_names):
        if ch_name.upper() in ['VEOG', 'HEOG', 'TP9']:
            continue
        if stds[i] > (median_std + 4.5 * mad) or stds[i] < (median_std - 4.5 * mad) or mean_corrs[i] < 0.30:
            bad_channels.append(ch_name)
    if bad_channels:
        raw_filtered.info['bads'] = bad_channels
    if raw_filtered.info['bads']:
        raw_filtered.interpolate_bads(reset_bads=True, verbose=False)

    raw_filtered.set_eeg_reference('average', projection=False, verbose=False)

    picks_for_ica = mne.pick_types(raw_filtered.info, eeg=True, eog=False, exclude='bads')
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard',
                                 fit_params=dict(ortho=False, extended=True),
                                 random_state=42, verbose=False)
    ica.fit(raw_filtered, picks=picks_for_ica)

    ic_labels = mne_icalabel.label_components(raw_filtered, ica, method='iclabel')

    eog_indices_veog, eog_indices_heog = [], []
    try:
        if 'VEOG' in raw_original.ch_names:
            eog_indices_veog, _ = ica.find_bads_eog(raw_original, ch_name='VEOG', threshold=2.0, verbose=False)
        if 'HEOG' in raw_original.ch_names:
            eog_indices_heog, _ = ica.find_bads_eog(raw_original, ch_name='HEOG', threshold=2.0, verbose=False)
    except Exception as e:
        print(f"    find_bads_eog failed: {e}")

    return raw_filtered, ica, ic_labels, set(eog_indices_veog), set(eog_indices_heog)


# ----------------------------------------------------------------------------
# Stage-wise exclusion sets
# ----------------------------------------------------------------------------
def compute_stage_exclusions(ic_labels, proba_matrix, eog_veog, eog_heog):
    n_components = proba_matrix.shape[0]

    argmax_idx = {idx for idx, label in enumerate(ic_labels['labels']) if label in ARTIFACT_TARGETS}

    prob_idx = set()
    for comp_idx in range(n_components):
        for cls_name in ARTIFACT_TARGETS:
            cls_col = ARTIFACT_CLASSES_IDX.get(cls_name)
            if cls_col is not None and proba_matrix[comp_idx, cls_col] >= PROB_THRESHOLD:
                prob_idx.add(comp_idx)
                break

    eog_idx = eog_veog | eog_heog
    brain_col = ARTIFACT_CLASSES_IDX['brain']
    eog_only_idx = eog_idx - (argmax_idx | prob_idx)
    eog_only_rescued = {idx for idx in eog_only_idx if proba_matrix[idx, brain_col] >= BRAIN_PROB_CEILING}

    stage1 = set(argmax_idx)
    stage2 = set(argmax_idx | prob_idx)
    stage3 = (set(argmax_idx | prob_idx | eog_idx)) - eog_only_rescued

    return [
        ("stage1_argmax", stage1),
        ("stage2_argmax_prob", stage2),
        ("stage3_full", stage3),
    ]


# ----------------------------------------------------------------------------
# Per-IC inspection plot (topomap + time series + PSD), mirrors inspect_single_ic.py
# ----------------------------------------------------------------------------
def plot_ic_inspection(raw_filtered, ica, ic_labels, proba_matrix, ic_index, subject_id, stage_name, out_dir, log_lines):
    sources = ica.get_sources(raw_filtered).get_data()
    ic_timeseries = sources[ic_index]
    sfreq = raw_filtered.info['sfreq']
    probs = proba_matrix[ic_index]
    argmax_label = ic_labels['labels'][ic_index]

    log_lines.append(f"\nIC{ic_index} ({subject_id}, {stage_name}):")
    log_lines.append(f"  argmax label: '{argmax_label}'")
    for cname, p in zip(ALL_CLASS_NAMES, probs):
        marker = "  <-- argmax" if cname == argmax_label else ""
        log_lines.append(f"    {cname:18s}: {p:.3f}{marker}")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax_topo = fig.add_subplot(gs[0, 0])
    ica.plot_components(picks=[ic_index], axes=ax_topo, show=False, colorbar=False)
    ax_topo.set_title(f"IC{ic_index} topography")

    ax_time = fig.add_subplot(gs[0, 1])
    n_show = min(int(60 * sfreq), len(ic_timeseries))
    t = np.arange(n_show) / sfreq
    ax_time.plot(t, ic_timeseries[:n_show], linewidth=0.5)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude (a.u.)")
    ax_time.set_title(f"IC{ic_index} time series (first 60s)")

    ax_time_full = fig.add_subplot(gs[1, 0])
    step = max(1, len(ic_timeseries) // 20000)
    t_full = np.arange(0, len(ic_timeseries), step) / sfreq
    ax_time_full.plot(t_full, ic_timeseries[::step], linewidth=0.3)
    ax_time_full.set_xlabel("Time (s)")
    ax_time_full.set_ylabel("Amplitude (a.u.)")
    ax_time_full.set_title(f"IC{ic_index} time series (full recording, downsampled)")

    ax_psd = fig.add_subplot(gs[1, 1])
    freqs, psd = welch(ic_timeseries, fs=sfreq, nperseg=min(4096, len(ic_timeseries)))
    ax_psd.semilogy(freqs, psd)
    ax_psd.set_xlim(0, 60)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (log scale)")
    ax_psd.set_title(f"IC{ic_index} power spectrum")

    fig.suptitle(f"{subject_id} -- {stage_name} -- NEWLY EXCLUDED IC{ic_index}  "
                 f"(argmax='{argmax_label}', brain={probs[0]:.2f} muscle={probs[1]:.2f} "
                 f"eye={probs[2]:.2f} other={probs[6]:.2f})", fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"IC{ic_index}_inspection.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# All-channels plots (full + zoomed)
# ----------------------------------------------------------------------------
def plot_all_channels(raw_clean_stage, subject_id, stage_name, out_dir):
    data = raw_clean_stage.get_data(picks='eeg') * 1e6   # convert to uV for readability
    ch_names = [raw_clean_stage.ch_names[i] for i in mne.pick_types(raw_clean_stage.info, eeg=True, eog=False)]
    sfreq = raw_clean_stage.info['sfreq']
    n_ch = data.shape[0]

    # --- full recording, downsampled for display ---
    step = max(1, data.shape[1] // 20000)
    t_full = np.arange(0, data.shape[1], step) / sfreq
    fig, ax = plt.subplots(figsize=(16, max(8, n_ch * 0.3)))
    offset_step = np.percentile(np.abs(data), 99) * 2.5
    for i in range(n_ch):
        ax.plot(t_full, data[i, ::step] + i * offset_step, linewidth=0.3)
    ax.set_yticks([i * offset_step for i in range(n_ch)])
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{subject_id} -- {stage_name} -- ALL CHANNELS (full recording, after this stage's cleaning)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "all_channels_full.png"), dpi=130)
    plt.close(fig)

    # --- zoomed 10-30s, full resolution ---
    z_start = int(ZOOM_START_SEC * sfreq)
    z_end = int(ZOOM_END_SEC * sfreq)
    z_end = min(z_end, data.shape[1])
    t_zoom = np.arange(z_start, z_end) / sfreq
    fig, ax = plt.subplots(figsize=(16, max(8, n_ch * 0.3)))
    for i in range(n_ch):
        ax.plot(t_zoom, data[i, z_start:z_end] + i * offset_step, linewidth=0.6)
    ax.set_yticks([i * offset_step for i in range(n_ch)])
    ax.set_yticklabels(ch_names, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"{subject_id} -- {stage_name} -- ALL CHANNELS, zoomed {ZOOM_START_SEC:.0f}-{ZOOM_END_SEC:.0f}s")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"all_channels_zoom_{int(ZOOM_START_SEC)}-{int(ZOOM_END_SEC)}s.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main per-subject processing
# ----------------------------------------------------------------------------
def process_subject(subject_id, output_root):
    print(f"\n{'='*70}\n{subject_id}\n{'='*70}")
    print(f"  fitting ICA (slow step)...")
    raw_filtered, ica, ic_labels, eog_veog, eog_heog = fit_ica(subject_id)
    proba_matrix = ica.labels_scores_
    n_components = proba_matrix.shape[0]
    print(f"  n_components={n_components}")

    stages = compute_stage_exclusions(ic_labels, proba_matrix, eog_veog, eog_heog)

    subject_out_dir = os.path.join(output_root, subject_id)
    os.makedirs(subject_out_dir, exist_ok=True)

    previous_excluded = set()
    for stage_name, excluded_set in stages:
        stage_out_dir = os.path.join(subject_out_dir, stage_name)
        os.makedirs(stage_out_dir, exist_ok=True)

        newly_excluded = sorted(excluded_set - previous_excluded)
        print(f"  [{stage_name}] total excluded={len(excluded_set)}  newly excluded this stage={newly_excluded}")

        log_lines = [f"Subject: {subject_id}   Stage: {stage_name}",
                      f"Total excluded so far: {sorted(excluded_set)}",
                      f"Newly excluded this stage: {newly_excluded}"]

        for ic_idx in newly_excluded:
            plot_ic_inspection(raw_filtered, ica, ic_labels, proba_matrix, ic_idx,
                                subject_id, stage_name, stage_out_dir, log_lines)

        with open(os.path.join(stage_out_dir, "probs_log.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

        # apply this stage's exclusion to a fresh copy and plot all-channels result
        raw_stage_clean = raw_filtered.copy()
        ica.exclude = sorted(excluded_set)
        if ica.exclude:
            ica.apply(raw_stage_clean, verbose=False)
        plot_all_channels(raw_stage_clean, subject_id, stage_name, stage_out_dir)

        previous_excluded = excluded_set

    print(f"  done -- outputs saved under {subject_out_dir}")


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_root = os.path.join(out_dir, OUTPUT_ROOT_NAME)
    os.makedirs(output_root, exist_ok=True)

    for sid in SUBJECT_IDS:
        try:
            process_subject(sid, output_root)
        except Exception as e:
            print(f"  FAILED for {sid}: {e}")

    print(f"\n{'='*70}")
    print(f"DONE. All outputs saved under: {output_root}")
    print(f"{'='*70}")
    print("Browse check-ica/{subject}/{stage}/ and compare progressively:")
    print("  stage1_argmax/        -> after argmax-only cleaning")
    print("  stage2_argmax_prob/   -> after argmax + prob_threshold cleaning")
    print("  stage3_full/          -> after full pipeline (+ EOG-correlation + rescue)")
    print("Each stage folder contains: one IC{n}_inspection.png per newly-excluded IC,")
    print("all_channels_full.png, all_channels_zoom_10-30s.png, and probs_log.txt.")
