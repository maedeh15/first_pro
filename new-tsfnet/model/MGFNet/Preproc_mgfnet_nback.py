# ============================================================================
# MGFNet PREPROCESSING PIPELINE -- n-back dataset (Shin et al., 2018)
#
# Reproduces Section 2.2 ("Data Preprocessing and Sample Construction") of:
#   Zhang, Y.; Gong, X.; Yuan, X. "MGFNet: A Multi-Granularity Fusion Network
#   with Coupling-Guided Sparse Routing for Hybrid EEG-fNIRS Decoding."
#   Sensors 2026, 26, 3402.
#
# NO OFFICIAL CODE RELEASE EXISTS. This is a from-scratch, paper-faithful
# re-implementation of the PREPROCESSING ONLY. Every place where the paper is
# underdetermined is marked # AMBIGUITY-Pn and listed at the bottom.
#
# PAPER (Sec. 2.2)                              HERE
#   EEG -> 200 Hz                               cnt.fs is already 200 Hz
#   EEG 1-40 Hz band-pass                       zero-phase Butterworth, 1-40 Hz
#   EOG "standard correction algorithm"         ICA + automatic EOG detection
#   X_EEG in R^{30 x 1000}                      28 scalp channels (default);
#                                               --include-eog gives the paper's 30
#   fNIRS MBLL -> HbO/HbR                       dataset ships them converted
#   fNIRS -> 10 Hz, low-pass                    10 Hz, 0.2 Hz low-pass
#   X_HbO/HbR in R^{36 x 50}                    36 x 50
#   5 s window, 1 s step, task periods          task period = onset+2 .. +42 s
#   no normalisation mentioned                  none here; per-channel z-score
#                                               is fitted on TRAINING windows
#                                               only, inside the LOSO loop
#                                               (mgfnet_dataset.ChannelNormalizer)
#
# EVALUATION PROTOCOL (the only deliberate deviation from the paper): the
# split is leave-one-session-out, applied downstream on the session_ids saved
# here -- NOT the paper's within-subject 80/20 trial split.
#
# OUTPUTS (per subject)
#   {subject}_mgf_eeg.npy      (N, 28, 1000)  float32   EEG   @200 Hz, 5 s, volts
#   {subject}_mgf_hbo.npy      (N, 36,   50)  float32   HbO   @10  Hz, 5 s
#   {subject}_mgf_hbr.npy      (N, 36,   50)  float32   HbR   @10  Hz, 5 s
#   {subject}_labels.npy       (N,)  int64    0/1/2  (0-back / 2-back / 3-back)
#   {subject}_session_ids.npy  (N,)  int64    1/2/3
#   {subject}_trial_ids.npy    (N,)  int64    0..26
#   {subject}_mgf_metadata.json           config + shapes + checks
# ============================================================================
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import multiprocessing as mp
import traceback

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

mne.set_log_level("ERROR")

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\MGFNET_PAPER_EXACT_nback_28ch"
OUTPUT_PATH_WITH_EOG = r"E:\kntu\Thesis\codes\Results\MGFNET_PAPER_EXACT_nback_30ch"

SUBJECTS = list(range(1, 27))

# --- EEG ---------------------------------------------------------------------
SFREQ_EEG = 200.0
EEG_BP_LOW = 1.0
EEG_BP_HIGH = 40.0
# The paper states the passband only. Design order N=6 (same design as this
# project's TSFNet pipeline). NOTE: scipy/MNE double N for a band-pass, so
# this is a 12th-order band-pass filter, and phase="zero" runs it forward +
# backward, so the magnitude response is that of a 24th-order filter.
# # AMBIGUITY-P1
EEG_BP_DESIGN_ORDER = 6
EEG_BP_EFFECTIVE_ORDER = EEG_BP_DESIGN_ORDER * 2 * 2

# The paper's EEG tensor is 30 x 1000. This recording's cnt.clab has exactly
# 30 entries: 28 scalp electrodes + HEOG + VEOG. The paper's own feature
# dims (x1 = 120 = 4 x 30, x12 = 264 = 4 x 66) confirm a 30-channel input,
# i.e. the EOG channels were fed to the network. They are kept here (after
# the same 1-40 Hz band-pass; ICA is applied to the scalp channels only).
# Set False for the 28-channel scalp-only variant. # AMBIGUITY-P5
EEG_INCLUDE_EOG_CHANNELS = False

# --- fNIRS -------------------------------------------------------------------
SFREQ_FNIRS = 10.0
# Cutoff not given ("conventions commonly adopted [52]"). 0.2 Hz: above the
# Mayer-wave band, below the cardiac band.  # AMBIGUITY-P2
FNIRS_LP_HIGH = 0.2
FNIRS_LP_DESIGN_ORDER = 2
FNIRS_LP_EFFECTIVE_ORDER = FNIRS_LP_DESIGN_ORDER * 2      # filtfilt
# The paper says LOW-pass: no high-pass corner, no baseline correction.
# # AMBIGUITY-P3
DATASET_ALREADY_BEER_LAMBERT = True

# --- Windowing ---------------------------------------------------------------
WINDOW_LEN = 5.0
WINDOW_STEP = 1.0
# "Task periods": each block is 2 s instruction + 40 s task + 20 s rest, and
# the trial marker sits at the START OF THE INSTRUCTION. Windows therefore
# cover onset+2 .. onset+42 s. marker_onset_diagnostic() below checks this
# assumption on every subject and records the result in the metadata; use
# --task-tmin to override if it fails.  # AMBIGUITY-P7
TASK_TMIN = 2.0
TASK_LEN = 40.0

# --- Verified dataset constants ----------------------------------------------
EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}
TRIALS_PER_SESSION = 9
EXPECTED_TRIALS = 27
EXPECTED_EEG_SCALP_CHANNELS = 28
EXPECTED_EOG_CHANNELS = ("HEOG", "VEOG")
EXPECTED_FNIRS_CHANNELS = 36

_LOG_PATH = None
_LOG_LOCK = None

# QC figures -- the same set Preproc_tsfnet_nback_badch.py saves per subject
# (into the subject's output folder): EEG single-channel before/after, all-
# channels stacked before/after (full + 10-30 s zoom), PSD before/after, PSD
# heatmaps before/after, per-channel band power, fNIRS before/after.
# (TSFNet's AutoReject figures have no counterpart: this pipeline has no
# AutoReject step.)  --no-qc disables them.
SAVE_QC = True

# Subject-level parallelism, as in the TSFNet script (its default is 4).
# ICA dominates the run time; each worker needs ~1.5-2.5 GB RAM.
N_WORKERS = 4


def log_message(msg):
    print(msg, flush=True)
    if _LOG_PATH:
        if _LOG_LOCK is not None:
            with _LOG_LOCK:
                with open(_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
        else:
            with open(_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


# ============================================================================
# LOADING
# ============================================================================
def _load_struct(mat_path):
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


def load_eeg_data(eeg_folder):
    cnt = _load_struct(os.path.join(eeg_folder, "cnt_nback.mat"))
    mrk = _load_struct(os.path.join(eeg_folder, "mrk_nback.mat"))

    data = np.asarray(cnt.x, dtype=np.float64).T      # (n_channels, n_samples)
    sfreq = float(cnt.fs)
    ch_names = [str(c).strip() for c in np.atleast_1d(cnt.clab)]

    if np.max(np.abs(data)) > 1.0:                   # uV -> V
        data = data * 1e-6
    if sfreq != SFREQ_EEG:                           # expected no-op
        data = resample(data, int(round(data.shape[1] * SFREQ_EEG / sfreq)), axis=1)
        sfreq = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    eog = [c for c in ch_names if c.upper() in EXPECTED_EOG_CHANNELS]
    if eog:
        raw.set_channel_types({c: "eog" for c in eog})
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore")

    if hasattr(mrk, "time"):
        pos = np.round(np.atleast_1d(np.asarray(mrk.time, dtype=float)) * sfreq / 1000).astype(int)
    elif hasattr(mrk, "pos"):
        pos = np.atleast_1d(np.asarray(mrk.pos, dtype=int)) - 1
    else:
        raise ValueError("no 'time' or 'pos' in EEG marker struct")

    if hasattr(mrk, "event") and hasattr(mrk.event, "desc"):
        codes = np.asarray(mrk.event.desc, dtype=int).flatten()
    elif hasattr(mrk, "y"):
        y = mrk.y
        codes = (np.argmax(y, axis=0) + 1) if np.ndim(y) > 1 else np.asarray(y).flatten()
    else:
        raise ValueError("no event codes in EEG marker struct")

    events = np.zeros((len(pos), 3), dtype=int)
    events[:, 0] = pos
    events[:, 2] = codes
    return raw, events, sfreq


def load_fnirs_data(subject_id):
    folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt = _load_struct(os.path.join(folder, "cnt_nback.mat"))

    ev_time = ev_code = None
    mrk_path = os.path.join(folder, "mrk_nback.mat")
    if os.path.exists(mrk_path):
        mrk = _load_struct(mrk_path)
        if hasattr(mrk, "time"):
            ev_time = np.atleast_1d(np.asarray(mrk.time, dtype=float))
        elif hasattr(mrk, "pos"):
            fs_tmp = float(getattr(cnt, "fs", SFREQ_FNIRS))
            ev_time = np.atleast_1d(np.asarray(mrk.pos, dtype=float)) / fs_tmp * 1000.0
        if hasattr(mrk, "event") and hasattr(mrk.event, "desc"):
            ev_code = np.asarray(mrk.event.desc, dtype=int).flatten()

    oxy = np.asarray(cnt.oxy.x, dtype=np.float64)        # (n_samples, 36)
    deoxy = np.asarray(cnt.deoxy.x, dtype=np.float64)
    fs = float(getattr(cnt.oxy, "fs", getattr(cnt, "fs", SFREQ_FNIRS)))
    return oxy, deoxy, fs, ev_time, ev_code


# ============================================================================
# CHECKS
# ============================================================================
def marker_onset_diagnostic(events, sfreq):
    """Checks the TASK_TMIN assumption (trial marker = instruction onset).

    For every trial marker, finds the first NON-trial marker (n-back stimulus)
    within the following 15 s. If the trial marker sits at instruction onset,
    the first stimulus follows ~2 s later; if it sits at task onset, ~0 s.
    Returns a dict for the metadata. Purely diagnostic: nothing is changed.
    """
    trial_mask = np.isin(events[:, 2], list(EEG_TRIAL_LABEL_MAP.keys()))
    t_trial = events[trial_mask, 0] / sfreq
    other = events[~trial_mask]
    if len(other) == 0:
        return {"status": "no_stimulus_markers", "median_delay_s": None,
                "stimulus_codes": []}
    t_other = other[:, 0] / sfreq
    delays, codes = [], []
    for t in t_trial:
        nxt = np.where((t_other > t + 1e-6) & (t_other <= t + 15.0))[0]
        if len(nxt):
            delays.append(t_other[nxt[0]] - t)
            codes.append(int(other[nxt[0], 2]))
    if not delays:
        return {"status": "no_stimulus_within_15s", "median_delay_s": None,
                "stimulus_codes": []}
    med = float(np.median(delays))
    status = "ok" if abs(med - TASK_TMIN) <= 0.6 else "MISMATCH"
    return {"status": status, "median_delay_s": round(med, 3),
            "stimulus_codes": sorted(set(codes))}



# ============================================================================
# QC FIGURES (same plots as Preproc_tsfnet_nback_badch.py)
# ============================================================================
def _compute_psd_from_raw(raw_obj, fmin=0.5, fmax=49.0):
    """(ch_names, freqs, psd_db) for the EEG channels, dB = 10log10(uV^2/Hz)."""
    raw_eeg = raw_obj.copy().pick("eeg")
    spectrum = raw_eeg.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                   n_fft=int(raw_eeg.info["sfreq"] * 4), verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)
    return raw_eeg.ch_names, freqs, 10 * np.log10(np.maximum(psd, 1e-24) * 1e12)


def _plot_psd_before_after(raw_before, filtered, subject_id):
    _, f_b, p_b = _compute_psd_from_raw(raw_before)
    _, f_a, p_a = _compute_psd_from_raw(filtered)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i in range(p_b.shape[0]):
        axes[0].plot(f_b, p_b[i], linewidth=0.6, alpha=0.6)
    axes[0].set_title(f"{subject_id} - PSD RAW (all channels overlaid)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    axes[0].axvline(50, color="k", linestyle="--", linewidth=0.8, label="50Hz line noise")
    axes[0].legend(fontsize=8)
    for i in range(p_a.shape[0]):
        axes[1].plot(f_a, p_a[i], linewidth=0.6, alpha=0.6)
    axes[1].set_title(f"{subject_id} - PSD PREPROCESSED (all channels overlaid)")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].axvline(50, color="k", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    return fig


def _plot_psd_heatmap(raw_obj, subject_id, title_suffix):
    ch_names, freqs, psd_db = _compute_psd_from_raw(raw_obj)
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


def _plot_band_power_bars(raw_obj, subject_id, title_suffix):
    ch_names, freqs, psd_db = _compute_psd_from_raw(raw_obj, fmin=0.5, fmax=49.0)
    bands = {"delta(0.5-4)": (0.5, 4), "theta(4-8)": (4, 8), "alpha(8-13)": (8, 13),
             "beta(13-30)": (13, 30), "gamma(30-49)": (30, 49)}
    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(ch_names)), 6))
    x = np.arange(len(ch_names))
    width = 0.8 / len(bands)
    for i, (name, (lo, hi)) in enumerate(bands.items()):
        m = (freqs >= lo) & (freqs < hi)
        vals = psd_db[:, m].mean(axis=1) if m.any() else np.full(len(ch_names), np.nan)
        ax.bar(x + i * width, vals, width=width, label=name)
    ax.set_xticks(x + width * (len(bands) - 1) / 2)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    ax.set_title(f"{subject_id} - Per-channel band power ({title_suffix})")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    return fig


def _plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
    """One row per EEG channel (EOG excluded, as in the TSFNet QC plots)."""
    raw_eeg = raw_obj.copy().pick("eeg")
    sf = raw_eeg.info["sfreq"]
    start = int((tmin or 0.0) * sf)
    stop = int(tmax * sf) if tmax is not None else None
    data = raw_eeg.get_data(start=start, stop=stop)
    t = (np.arange(data.shape[1]) + start) / sf
    spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 30 + 1e-12
    n = data.shape[0]
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * n)))
    for i in range(n):
        ax.plot(t, data[i] + (n - 1 - i) * spread, color=color, linewidth=0.4)
    ax.set_yticks([(n - 1 - i) * spread for i in range(n)])
    ax.set_yticklabels(raw_eeg.ch_names, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def save_eeg_qc_plots(raw_before, filtered, subject_id, qc_dir):
    """Every EEG figure TSFNet's preprocess_eeg() saves, for this pipeline's
    1-40 Hz + ICA cleaning. Never raises: a plotting failure is logged only."""
    desc = f"{EEG_BP_LOW:g}-{EEG_BP_HIGH:g}Hz + ICA"
    n_ch = len(mne.pick_types(filtered.info, eeg=True, eog=False))
    try:
        os.makedirs(qc_dir, exist_ok=True)
        ch0 = filtered.copy().pick("eeg").ch_names[0]
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(raw_before.times, raw_before.get_data(picks=[ch0])[0], color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - EEG RAW ({ch0})")
        axes[1].plot(filtered.times, filtered.get_data(picks=[ch0])[0], color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} - EEG PREPROCESSED ({desc}) ({ch0})")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_after.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save EEG single-channel QC plot: {e}")
    try:
        for tag, obj, col, lab in (("before", raw_before, "tab:red", "RAW"),
                                   ("after", filtered, "tab:blue", f"PREPROCESSED ({desc})")):
            fig = _plot_all_channels_stacked(
                obj, f"{subject_id} - EEG {lab} - All {n_ch} Channels, EOG excluded", col)
            fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_{tag}_allchannels.png"), dpi=150)
            plt.close(fig)
            fig = _plot_all_channels_stacked(
                obj, f"{subject_id} - EEG {lab} - All {n_ch} Channels, EOG excluded (10-30s)",
                col, tmin=10.0, tmax=30.0)
            fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_{tag}_allchannels_zoom10_30.png"), dpi=150)
            plt.close(fig)
        log_message(f"  {subject_id}: saved EEG all-channels QC plots (before/after, full+zoomed)")
    except Exception as e:
        log_message(f"  {subject_id}: could not save EEG all-channels QC plots: {e}")
    try:
        for fig, name in ((_plot_psd_before_after(raw_before, filtered, subject_id), "psd_before_after"),
                          (_plot_psd_heatmap(raw_before, subject_id, "RAW"), "psd_heatmap_before"),
                          (_plot_psd_heatmap(filtered, subject_id, "PREPROCESSED"), "psd_heatmap_after"),
                          (_plot_band_power_bars(filtered, subject_id, "PREPROCESSED"), "band_power_after")):
            fig.savefig(os.path.join(qc_dir, f"{subject_id}_{name}.png"), dpi=150)
            plt.close(fig)
        log_message(f"  {subject_id}: saved PSD QC plots (before/after, heatmaps, band power)")
    except Exception as e:
        log_message(f"  {subject_id}: could not save PSD QC plots: {e}")


def save_fnirs_qc_plot(oxy_raw, fs_raw, hbo, subject_id, qc_dir):
    try:
        os.makedirs(qc_dir, exist_ok=True)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(np.arange(oxy_raw.shape[0]) / fs_raw, oxy_raw[:, 0], color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - fNIRS RAW HbO (ch 0)")
        axes[1].plot(np.arange(hbo.shape[0]) / SFREQ_FNIRS, hbo[:, 0], color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} - fNIRS PREPROCESSED HbO (Beer-Lambert + "
                          f"{FNIRS_LP_HIGH:g}Hz low-pass)")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save fNIRS QC plot: {e}")


# ============================================================================
# FILTERING
# ============================================================================
def fnirs_lowpass(data, fs, high=FNIRS_LP_HIGH, order=FNIRS_LP_DESIGN_ORDER):
    """Zero-phase Butterworth low-pass along axis 0 (time)."""
    nyq = fs / 2.0
    wn = min(high / nyq, 0.99)
    b, a = butter(order, wn, btype="low")
    padlen = 3 * max(len(a), len(b))
    if data.shape[0] <= padlen:
        raise ValueError(f"fNIRS recording too short to filter: {data.shape[0]} samples")
    return filtfilt(b, a, data, axis=0)


def preprocess_eeg(raw, subject_id, qc_dir=None):
    """1-40 Hz band-pass (scalp + EOG channels), then ICA-based EOG
    correction of the scalp channels.  # AMBIGUITY-P6

    Returns (data, ch_names, n_ica_excluded). data is in volts; no
    normalisation is applied here (see header).
    """
    filt = raw.copy()
    filt.filter(l_freq=EEG_BP_LOW, h_freq=EEG_BP_HIGH, method="iir",
                iir_params=dict(order=EEG_BP_DESIGN_ORDER, ftype="butter", output="sos"),
                picks=["eeg", "eog"], phase="zero", verbose=False)

    picks_eeg = mne.pick_types(filt.info, eeg=True, eog=False)
    eog_names = [filt.ch_names[i] for i in mne.pick_types(filt.info, eeg=False, eog=True)]
    n_excluded = 0
    if eog_names:
        try:
            ica = mne.preprocessing.ICA(n_components=0.99, method="picard",
                                        fit_params=dict(ortho=False, extended=True),
                                        random_state=42, verbose=False)
            ica.fit(filt, picks=picks_eeg)
            bad, _ = ica.find_bads_eog(filt, ch_name=eog_names, verbose=False)
            ica.exclude = bad
            n_excluded = len(bad)
            ica.apply(filt, verbose=False)
        except Exception as e:
            log_message(f"  {subject_id}: ICA EOG correction FAILED ({e}); "
                        f"continuing with band-passed but UNCORRECTED EEG. "
                        f"This subject deviates from the paper -- record it.")
            n_excluded = -1
    else:
        log_message(f"  {subject_id}: no VEOG/HEOG channels found; EOG correction SKIPPED.")
        n_excluded = -1
    log_message(f"  {subject_id}: EEG {EEG_BP_LOW}-{EEG_BP_HIGH} Hz (Butterworth, "
                f"effective order {EEG_BP_EFFECTIVE_ORDER}, zero-phase); "
                f"ICA removed {n_excluded} ocular component(s).")

    if SAVE_QC and qc_dir:
        save_eeg_qc_plots(raw, filt, subject_id, qc_dir)

    picks = mne.pick_types(filt.info, eeg=True, eog=EEG_INCLUDE_EOG_CHANNELS)
    ch_names = [filt.ch_names[i] for i in picks]
    return filt.get_data(picks=picks), ch_names, n_excluded


def preprocess_fnirs(subject_id, qc_dir=None):
    oxy, deoxy, fs, ev_time, ev_code = load_fnirs_data(subject_id)
    oxy_raw, fs_raw = oxy.copy(), fs
    if not DATASET_ALREADY_BEER_LAMBERT:
        raise NotImplementedError(
            "This dataset copy ships oxy/deoxy already Beer-Lambert converted. "
            "Add an MBLL step here if your copy really is optical density.")

    if abs(fs - SFREQ_FNIRS) > 0.05:                    # expected no-op
        n = int(round(oxy.shape[0] * SFREQ_FNIRS / fs))
        oxy, deoxy = resample(oxy, n, axis=0), resample(deoxy, n, axis=0)
        log_message(f"  {subject_id}: fNIRS resampled {fs:.3g} -> {SFREQ_FNIRS} Hz")

    hbo = fnirs_lowpass(oxy, SFREQ_FNIRS)
    hbr = fnirs_lowpass(deoxy, SFREQ_FNIRS)
    log_message(f"  {subject_id}: fNIRS low-pass {FNIRS_LP_HIGH} Hz "
                f"(design order {FNIRS_LP_DESIGN_ORDER}, effective {FNIRS_LP_EFFECTIVE_ORDER}).")
    if SAVE_QC and qc_dir:
        save_fnirs_qc_plot(oxy_raw, fs_raw, hbo, subject_id, qc_dir)
    # returned as (36, n_samples) to match the EEG (C, n_times) convention
    return hbo.T, hbr.T, ev_time, ev_code


# ============================================================================
# WINDOWING
# ============================================================================
def extract_windows(eeg, sfreq_eeg, events, hbo, hbr, fnirs_ev_time,
                    fnirs_ev_code, subject_id):
    """5 s / 1 s sliding windows over the 40 s task period of each trial.

    fNIRS windows are timed from fNIRS's OWN trial markers whenever they
    agree with EEG's (the two devices are independently clocked).
    Sets extract_windows.last_sync_ok so the caller can record which clock
    was used.
    """
    extract_windows.last_sync_ok = False
    empty = (np.empty((0, eeg.shape[0], 0)),) * 3 + (np.empty(0, dtype=np.int64),) * 3

    target = events[np.isin(events[:, 2], list(EEG_TRIAL_LABEL_MAP.keys()))]
    target = target[np.argsort(target[:, 0], kind="stable")]
    if len(target) == 0:
        log_message(f"  {subject_id}: NO trial markers (codes seen: {np.unique(events[:, 2])})")
        return empty
    if len(target) != EXPECTED_TRIALS:
        log_message(f"  {subject_id}: WARNING - {len(target)} trial markers, expected "
                    f"{EXPECTED_TRIALS}. Session assignment (trial_idx // "
                    f"{TRIALS_PER_SESSION}) may be wrong.")

    # --- fNIRS clock sync ---------------------------------------------------
    sync_ok, fnirs_times = False, None
    if fnirs_ev_time is not None and fnirs_ev_code is not None:
        mask = np.isin(fnirs_ev_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
        t_ms = np.asarray(fnirs_ev_time)[mask].astype(float)
        codes = np.asarray(fnirs_ev_code)[mask]
        order = np.argsort(t_ms, kind="stable")
        fnirs_times = t_ms[order] / 1000.0
        f_cls = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in codes[order]])
        e_cls = np.array([EEG_TRIAL_LABEL_MAP[int(ev[2])] for ev in target])
        if len(f_cls) == len(e_cls) and np.array_equal(f_cls, e_cls):
            sync_ok = True
        else:
            log_message(f"  {subject_id}: WARNING - fNIRS/EEG marker mismatch; "
                        f"falling back to the EEG clock for fNIRS windows.")
    else:
        log_message(f"  {subject_id}: WARNING - no fNIRS markers; "
                    f"falling back to the EEG clock for fNIRS windows.")
    extract_windows.last_sync_ok = sync_ok

    n_eeg = int(round(WINDOW_LEN * sfreq_eeg))       # 1000
    n_nir = int(round(WINDOW_LEN * SFREQ_FNIRS))     # 50
    offsets = np.arange(TASK_TMIN, TASK_TMIN + TASK_LEN - WINDOW_LEN + 1e-6, WINDOW_STEP)

    E, O, R, lab, sess, trl = [], [], [], [], [], []
    for trial_idx, ev in enumerate(target):
        label = EEG_TRIAL_LABEL_MAP[int(ev[2])]
        session_id = trial_idx // TRIALS_PER_SESSION + 1
        t_eeg = ev[0] / sfreq_eeg
        t_nir = fnirs_times[trial_idx] if sync_ok else t_eeg

        for off in offsets:
            s_e = int(round((t_eeg + off) * sfreq_eeg))
            s_f = int(round((t_nir + off) * SFREQ_FNIRS))
            if s_e < 0 or s_e + n_eeg > eeg.shape[1]:
                continue
            if s_f < 0 or s_f + n_nir > hbo.shape[1]:
                continue
            E.append(eeg[:, s_e:s_e + n_eeg])
            O.append(hbo[:, s_f:s_f + n_nir])
            R.append(hbr[:, s_f:s_f + n_nir])
            lab.append(label); sess.append(session_id); trl.append(trial_idx)

    if not E:
        return empty
    return (np.asarray(E, dtype=np.float32), np.asarray(O, dtype=np.float32),
            np.asarray(R, dtype=np.float32), np.asarray(lab, dtype=np.int64),
            np.asarray(sess, dtype=np.int64), np.asarray(trl, dtype=np.int64))


# ============================================================================
# PER-SUBJECT DRIVER
# ============================================================================
def process_subject(subject_num):
    subject_id = f"VP{subject_num:03d}"
    log_message(f"\n===== {subject_id} =====")

    raw, events, sfreq = load_eeg_data(os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG"))
    n_scalp = len(mne.pick_types(raw.info, eeg=True, eog=False))
    n_eog = len(mne.pick_types(raw.info, eeg=False, eog=True))
    if n_scalp != EXPECTED_EEG_SCALP_CHANNELS or n_eog != len(EXPECTED_EOG_CHANNELS):
        log_message(f"  {subject_id}: WARNING - {n_scalp} scalp + {n_eog} EOG channels, "
                    f"expected {EXPECTED_EEG_SCALP_CHANNELS} + {len(EXPECTED_EOG_CHANNELS)}.")

    onset_check = marker_onset_diagnostic(events, sfreq)
    log_message(f"  {subject_id}: trial-marker -> first stimulus delay: "
                f"{onset_check['median_delay_s']} s (TASK_TMIN={TASK_TMIN}) "
                f"[{onset_check['status']}]")
    if onset_check["status"] == "MISMATCH":
        log_message(f"  {subject_id}: *** WARNING - the trial marker does not appear "
                    f"to sit {TASK_TMIN} s before the first stimulus. Check TASK_TMIN "
                    f"(--task-tmin) before training.")

    out_dir = os.path.join(OUTPUT_PATH, subject_id)
    os.makedirs(out_dir, exist_ok=True)
    eeg, ch_names, n_ica = preprocess_eeg(raw, subject_id, qc_dir=out_dir)
    hbo, hbr, ev_t, ev_c = preprocess_fnirs(subject_id, qc_dir=out_dir)
    if hbo.shape[0] != EXPECTED_FNIRS_CHANNELS:
        log_message(f"  {subject_id}: WARNING - {hbo.shape[0]} fNIRS channels, "
                    f"expected {EXPECTED_FNIRS_CHANNELS}.")

    E, O, R, lab, sess, trl = extract_windows(eeg, sfreq, events, hbo, hbr,
                                              ev_t, ev_c, subject_id)
    fnirs_clock_sync = bool(extract_windows.last_sync_ok)
    if not fnirs_clock_sync:
        log_message(f"  {subject_id}: *** fNIRS windows timed on the EEG CLOCK -- this "
                    f"subject's fNIRS is misaligned and MUST be reported.")
    if E.shape[0] == 0:
        log_message(f"  {subject_id}: NO windows produced -- nothing saved.")
        return {"subject_id": subject_id, "status": "no_windows"}

    np.save(os.path.join(out_dir, f"{subject_id}_mgf_eeg.npy"), E)
    np.save(os.path.join(out_dir, f"{subject_id}_mgf_hbo.npy"), O)
    np.save(os.path.join(out_dir, f"{subject_id}_mgf_hbr.npy"), R)
    np.save(os.path.join(out_dir, f"{subject_id}_labels.npy"), lab)
    np.save(os.path.join(out_dir, f"{subject_id}_session_ids.npy"), sess)
    np.save(os.path.join(out_dir, f"{subject_id}_trial_ids.npy"), trl)

    per_sess = {int(s): int((sess == s).sum()) for s in np.unique(sess)}
    per_cls = {int(c): int((lab == c).sum()) for c in np.unique(lab)}
    meta = {
        "subject_id": subject_id, "pipeline": "MGFNet Sensors 2026 26 3402 Sec. 2.2",
        "n_windows": int(E.shape[0]),
        "eeg_shape": list(E.shape), "hbo_shape": list(O.shape), "hbr_shape": list(R.shape),
        "eeg_channels": ch_names, "eeg_includes_eog": EEG_INCLUDE_EOG_CHANNELS,
        "windows_per_session": per_sess, "windows_per_class": per_cls,
        "eeg_bandpass_hz": [EEG_BP_LOW, EEG_BP_HIGH],
        "eeg_bp_effective_order": EEG_BP_EFFECTIVE_ORDER,
        "ica_components_removed": n_ica,
        "fnirs_lowpass_hz": FNIRS_LP_HIGH,
        "fnirs_lp_effective_order": FNIRS_LP_EFFECTIVE_ORDER,
        "window_len_s": WINDOW_LEN, "window_step_s": WINDOW_STEP,
        "task_span_s": [TASK_TMIN, TASK_TMIN + TASK_LEN],
        "marker_onset_check": onset_check,
        "normalization": "none in preprocessing (train-fold z-score in training)",
        "baseline_correction": False, "window_amplitude_rejection": False,
        "label_map": {"0-back": 0, "2-back": 1, "3-back": 2},
        "trials_per_session": TRIALS_PER_SESSION,
        "fnirs_clock_sync": fnirs_clock_sync,
    }
    with open(os.path.join(out_dir, f"{subject_id}_mgf_metadata.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log_message(f"  {subject_id}: saved N={E.shape[0]}  EEG{tuple(E.shape)} "
                f"HbO{tuple(O.shape)} HbR{tuple(R.shape)}")
    log_message(f"  {subject_id}: per-session {per_sess} | per-class {per_cls}")
    return {"subject_id": subject_id, "status": "ok", **meta}


_CONFIG_KEYS = ("BASE_DATASET_PATH", "OUTPUT_PATH", "TASK_TMIN", "EEG_INCLUDE_EOG_CHANNELS",
                "SAVE_QC", "_LOG_PATH")


def _init_worker(config, lock):
    """Spawned workers (Windows) re-import this file with its DEFAULT globals;
    copy the parent's CLI-adjusted configuration into each worker."""
    global _LOG_LOCK
    globals().update(config)
    _LOG_LOCK = lock


def _process_subject_safe(n):
    try:
        return process_subject(n)
    except Exception:
        log_message(f"  VP{n:03d}: FAILED\n{traceback.format_exc()}")
        return {"subject_id": f"VP{n:03d}", "status": "failed"}


def main():
    global _LOG_PATH, SUBJECTS, BASE_DATASET_PATH, OUTPUT_PATH, TASK_TMIN, SAVE_QC, N_WORKERS, EEG_INCLUDE_EOG_CHANNELS
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default=None,
                    help="comma-separated subject numbers, e.g. 1 or 1,2,3")
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--include-eog", action="store_true",
                    help="keep HEOG/VEOG as network inputs (30 channels, the paper's "
                         "literal setting). Default: 28 scalp channels only.")
    ap.add_argument("--workers", type=int, default=None,
                    help=f"parallel subject workers (default {N_WORKERS}, as in the TSFNet script; "
                         f"use 1-2 if RAM is short)")
    ap.add_argument("--no-qc", action="store_true", help="skip the QC figures")
    ap.add_argument("--task-tmin", type=float, default=None,
                    help="seconds from trial marker to task onset (default 2.0)")
    args = ap.parse_args()

    if args.include_eog:
        EEG_INCLUDE_EOG_CHANNELS = True
        if not args.out:
            OUTPUT_PATH = OUTPUT_PATH_WITH_EOG
    if args.data_root:
        BASE_DATASET_PATH = args.data_root
    if args.out:
        OUTPUT_PATH = args.out
    if args.subjects:
        SUBJECTS = [int(s) for s in args.subjects.split(",")]
    if args.task_tmin is not None:
        TASK_TMIN = args.task_tmin
    if args.workers is not None:
        N_WORKERS = max(1, args.workers)
    if args.no_qc:
        SAVE_QC = False

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    _LOG_PATH = os.path.join(OUTPUT_PATH, "mgfnet_preprocessing_log.txt")
    n_workers = min(N_WORKERS, len(SUBJECTS))
    log_message(f"MGFNet preprocessing | subjects={SUBJECTS} | out={OUTPUT_PATH} | "
                f"workers={n_workers} | QC figures={SAVE_QC}")

    if n_workers <= 1:
        results = [_process_subject_safe(n) for n in SUBJECTS]
    else:
        config = {k: globals()[k] for k in _CONFIG_KEYS}
        lock = mp.Lock()
        with mp.Pool(processes=n_workers, initializer=_init_worker,
                     initargs=(config, lock)) as pool:
            results = list(pool.imap(_process_subject_safe, SUBJECTS))

    ok = [r for r in results if r.get("status") == "ok"]
    log_message(f"\n===== DONE: {len(ok)}/{len(results)} subjects =====")
    if ok:
        ns = [r["n_windows"] for r in ok]
        log_message(f"windows per subject: min={min(ns)} max={max(ns)} mean={np.mean(ns):.1f}")
        bad_sync = [r["subject_id"] for r in ok if not r["fnirs_clock_sync"]]
        bad_onset = [r["subject_id"] for r in ok
                     if r["marker_onset_check"]["status"] == "MISMATCH"]
        if bad_sync:
            log_message(f"*** fNIRS clock NOT synced for: {bad_sync}")
        if bad_onset:
            log_message(f"*** marker-onset MISMATCH for: {bad_onset}")
    with open(os.path.join(OUTPUT_PATH, "mgfnet_preprocessing_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


MGFNET_PREPROC_AMBIGUITIES = {
    "P1": "EEG filter order/type not stated; only the 1-40 Hz passband. "
          "Zero-phase Butterworth (design N=6 -> 12th-order band-pass, "
          "effective 24th order after forward-backward filtering).",
    "P2": "fNIRS low-pass cutoff not stated. 0.2 Hz used: above the "
          "Mayer-wave band, below cardiac.",
    "P3": "The paper says low-pass, not band-pass, and mentions no baseline "
          "correction. Neither is applied.",
    "P4": "No amplitude normalisation is mentioned. None is applied here; a "
          "per-channel z-score is fitted on the training windows of each "
          "LOSO fold only (no statistics from the held-out session).",
    "P5": "The paper's EEG tensor is 30 x 1000 and its feature dims are "
          "4 x 30 / 4 x 66, so the 30 cnt.clab channels (28 scalp + HEOG + "
          "VEOG) were fed to the network. DEFAULT HERE: 28 scalp channels, because "
          "n-back load modulates blinks/eye movements and EOG inputs would let "
          "the model decode ocular rather than cortical activity. --include-eog "
          "reproduces the paper's 30-channel input.",
    "P6": "'Standard correction algorithm' for EOG is not named. ICA + "
          "automatic EOG-component detection from VEOG/HEOG is used.",
    "P7": "'Task periods' read as trial onset + 2 s .. + 42 s (instruction "
          "excluded). Verified per subject by marker_onset_diagnostic().",
}

if __name__ == "__main__":
    main()
