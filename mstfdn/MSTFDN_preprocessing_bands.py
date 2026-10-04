"""
MSTFDN Preprocessing Pipeline - N-back Dataset (Shin et al. 2018)
=================================================================
Implements the exact preprocessing described in:
  Ding et al., "MSTFDN: An EEG-fNIRS multimodal spatial-temporal fusion
  decoding network for personalized multi-task scenarios",
  Information Fusion 131 (2026) 104187.

Dataset: Shin et al. 2018 (https://doi.org/10.1038/sdata.2018.3)
  - 26 subjects (VP001-VP026)
  - Tasks: 0-back, 2-back, 3-back (3-class MWL)
  - EEG: 30 channels, 200 Hz
  - fNIRS: 36 channels (HbO+HbR), 10 Hz

Key MSTFDN preprocessing decisions (from paper Sec. 2.2 & 2.3):
  1. EEG: 1-40 Hz bandpass + 50 Hz notch, ICA artifact removal,
     average reference, azimuthal-equidistant spatial mapping → 16x16 grid
  2. fNIRS: Beer-Lambert already applied by dataset authors (oxy=HbO, deoxy=HbR),
     NO separate wavelet/MARA/lowpass (model's large-kernel convolutions act as LPF),
     baseline correction (-5 to -2 s before trial), spatial mapping → 16x16 grid
  3. Large time window: EEG 10 s, fNIRS 21 overlapping 3 s segments (step 1 s)
     → fNIRS input shape per window: (21, 16, 16, 30, 2)
  4. Synchronized asymmetric windowing: EEG uses EEG's own trial-onset markers
     (codes 112/128/144), fNIRS uses fNIRS's own trial-onset markers (codes 7/8/9)
     to eliminate cross-session clock drift (~46-93 s across 3 sessions)
  5. Per-subject z-score normalization in grid space BEFORE windowing

Saved outputs per subject (in OUTPUT_PATH/VPxxx/):
  - {id}_eeg_windows.npy          shape (N, 5, 16, 16, 2000)  float32  [5 real freq bands]
  - {id}_fnirs_windows.npy        shape (N, 21, 16, 16, 30, 2)  float32
  - {id}_labels.npy               shape (N,)  int  [0=0-back, 1=2-back, 2=3-back]
  - {id}_session_ids.npy          shape (N,)  int  [1,2,3]
  - {id}_trial_ids.npy            shape (N,)  int  trial index per subject (0..26)
  - {id}_eeg_clean_raw.fif        cleaned continuous EEG (MNE Raw, for QC)
  - {id}_fnirs_before_after.png   fNIRS QC plot
  - {id}_eeg_before_allchannels.png  EEG QC before
  - {id}_eeg_after_allchannels.png   EEG QC after
  - {id}_eeg_before_allchannels_zoom10_30.png
  - {id}_eeg_after_allchannels_zoom10_30.png
  - {id}_metadata.json            pipeline parameters & shapes
  - processing_log.txt            timestamped log

Set BASE_DATASET_PATH and OUTPUT_PATH below before running.
"""

import os
import sys
import gc       # ← Fix 1: explicit garbage collection
import json
import time
import datetime
import warnings

import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import CloughTocher2DInterpolator, griddata
from scipy.spatial import Delaunay
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import mne
import mne_icalabel
import pywt
from scipy.interpolate import UnivariateSpline

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION - adjust paths before running
# ============================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"   # root: DATA/VPxxx-EEG, DATA/VPxxx-NIRS
OUTPUT_PATH       = r"E:\kntu\Thesis\codes\Results\MSTFDN_PREPROCESSED_NBACK"

# --- Window / epoch parameters (MSTFDN paper Table 1 / Sec. 2.2) ---
EEG_WINDOW_LEN   = 10.0   # seconds  (paper sweep: 8/9/10/11 s, optimal = 10 s)
EEG_STEP         = 1.0    # sliding-window step
EEG_EPOCH_TMIN   = 0.0
EEG_EPOCH_TMAX   = 40.0   # full N-back task period per trial (Shin 2018: 2s instruct + 40s task + 20s rest)

FNIRS_WINDOW_LEN  = 3.0   # length of each fNIRS segment
FNIRS_NUM_SEGMENTS = 21   # segments span t=0..23 s → covers 5-10 s HRF delay for 10 s task window
FNIRS_STEP        = 1.0   # segment step (adjacent segments overlap by 2 s)

BASELINE_TMIN    = -5.0   # fNIRS baseline window start (s before trial onset)
BASELINE_TMAX    = -2.0   # fNIRS baseline window end

FNIRS_EPOCH_TMIN = 0.0
FNIRS_EPOCH_TMAX = 18.0   # (kept for reference; fNIRS coverage governed by FNIRS_NUM_SEGMENTS)

SFREQ_EEG   = 200.0       # Hz
SFREQ_FNIRS = 10.0        # Hz

# --- EEG frequency bands (E-TFEM groups=5, paper Sec. 2.2.1) ---
# Real bandpass filtering - NOT a broadcast copy. Each band must carry
# genuinely different spectral information for the grouped dilated
# convolution in E-TFEM to learn distinct band-specific temporal patterns.
EEG_BANDS = {
    'delta': (1.0,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 40.0),
}
EEG_BAND_NAMES = list(EEG_BANDS.keys())   # order preserved for stacking

# --- Marker codes (Shin 2018 Dataset_description_MATLAB.pdf) ---
# EEG  mrk_nback: 112=0-back session, 128=2-back session, 144=3-back session
# fNIRS mrk_nback:   7=0-back session,   8=2-back session,   9=3-back session
EEG_TRIAL_LABEL_MAP   = {112: 0, 128: 1, 144: 2}
FNIRS_TRIAL_LABEL_MAP = {7: 0,   8: 1,   9: 2}

TRIALS_PER_SESSION = 9   # 27 trials per subject: 3 sessions x 9 trials each

# ============================================================
#  LOGGING
# ============================================================
LOG_FILE = os.path.join(OUTPUT_PATH, "processing_log.txt")

def log_message(msg):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
#  SPATIAL MAPPING - azimuthal equidistant projection → 16x16
# ============================================================
def spatial_mapping(data, mnt):
    """
    Project sensor data onto a 16x16 2-D grid using azimuthal equidistant
    projection (paper Sec. 2.3, mirrors TSFNet methodology).

    Parameters
    ----------
    data : ndarray (n_samples, n_channels)
    mnt  : struct with .pos_3d (3, n_channels)

    Returns
    -------
    mapped : ndarray (n_samples, 16, 16)  float32
    """
    pos_3d = np.asarray(mnt.pos_3d, dtype=np.float64)   # (3, n_channels)
    x3d, y3d, z3d = pos_3d[0], pos_3d[1], pos_3d[2]

    r      = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    r_safe = np.where(r > 0, r, 1.0)
    theta  = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))  # angle from vertex
    phi    = np.arctan2(y3d, x3d)                           # azimuthal angle
    x_proj = theta * np.cos(phi)
    y_proj = theta * np.sin(phi)

    grid_x, grid_y = np.mgrid[x_proj.min():x_proj.max():16j,
                               y_proj.min():y_proj.max():16j]
    points = np.column_stack((x_proj, y_proj))

    # Batched CloughTocher interpolation - triangulation computed ONCE
    tri    = Delaunay(points)
    values = data.T                  # (n_channels, n_samples)
    interp = CloughTocher2DInterpolator(tri, values, fill_value=0)
    result = interp((grid_x, grid_y))              # (16, 16, n_samples)
    mapped = np.transpose(result, (2, 0, 1)).astype(np.float32)  # (n_samples,16,16)

    # Fix 2: delete large intermediate arrays before return
    del result, values, interp, tri, grid_x, grid_y, points
    return mapped


# ============================================================
#  EEG MONTAGE ADAPTER
# ============================================================
class _EEGMontageAdapter:
    """Minimal wrapper exposing .pos_3d (3, n_channels) for spatial_mapping()."""
    def __init__(self, pos_3d):
        self.pos_3d = pos_3d


def get_eeg_mnt(raw, eeg_mnt_struct):
    """
    Build an _EEGMontageAdapter from the dataset's own EEG mnt_nback.mat,
    selecting channels by name to match raw.ch_names (post-drop order).
    Must be called AFTER channel drops (VEOG/HEOG/TP9 removed in preprocess_eeg).
    """
    full_clab  = [str(c).strip() for c in np.asarray(eeg_mnt_struct.clab).flatten()]
    full_pos3d = np.asarray(eeg_mnt_struct.pos_3d, dtype=np.float64)   # (3, 30)
    name2idx   = {n: i for i, n in enumerate(full_clab)}
    name2idx_u = {n.upper(): i for i, n in enumerate(full_clab)}        # case-insensitive fallback

    pos_list = []
    missing  = []
    for ch in raw.ch_names:
        idx = name2idx.get(ch, name2idx_u.get(ch.upper(), None))
        if idx is None:
            missing.append(ch)
            pos_list.append([0.0, 0.0, 0.0])     # fallback → grid centre
        else:
            pos_list.append(full_pos3d[:, idx])

    if missing:
        print(f"  WARNING: {len(missing)} EEG channel(s) not in mnt_nback.mat, "
              f"placed at grid centre: {missing}")

    pos_3d = np.array(pos_list, dtype=np.float64).T   # (3, n_channels)
    return _EEGMontageAdapter(pos_3d)


# ============================================================
#  LOAD EEG DATA
# ============================================================
def load_eeg_data(eeg_folder):
    """
    Load cnt_nback.mat + mrk_nback.mat + mnt_nback.mat from eeg_folder.
    Returns (raw, events, sfreq, eeg_mnt_struct).
    """
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(eeg_folder, 'mrk_nback.mat')
    mnt_path = os.path.join(eeg_folder, 'mnt_nback.mat')

    print(f"Loading EEG from: {cnt_path}")

    # --- Load signal ---
    mat_cnt   = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    cnt_key   = [k for k in mat_cnt if not k.startswith('__')][0]
    cnt       = mat_cnt[cnt_key]

    data     = cnt.x.T                                    # (n_channels, n_times)
    sfreq    = float(cnt.fs)
    ch_names = [str(ch).strip() for ch in cnt.clab]

    if np.max(np.abs(data)) > 1.0:    # convert μV → V if needed
        data = data * 1e-6

    if sfreq != SFREQ_EEG:
        print(f"Resampling EEG from {sfreq} to {SFREQ_EEG} Hz")
        n_samp = int(data.shape[1] * SFREQ_EEG / sfreq)
        data   = resample(data, n_samp, axis=1)
        sfreq  = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw  = mne.io.RawArray(data, info, verbose=False)

    # Mark EOG channels
    eog_ch = [ch for ch in ch_names if ch.upper() in ('VEOG', 'HEOG')]
    if eog_ch:
        raw.set_channel_types({ch: 'eog' for ch in eog_ch})

    # Standard montage for interpolate_bads() internal use only
    raw.set_montage('standard_1005', match_case=False, on_missing='ignore')

    # --- Load markers ---
    mat_mrk  = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    mrk_key  = [k for k in mat_mrk if not k.startswith('__')][0]
    mrk      = mat_mrk[mrk_key]

    if hasattr(mrk, 'time'):
        ev_pos = np.round(mrk.time * sfreq / 1000).astype(int)
    elif hasattr(mrk, 'pos'):
        ev_pos = np.array(mrk.pos, dtype=int) - 1
    else:
        raise ValueError("Could not find 'time' or 'pos' in EEG marker struct")

    if hasattr(mrk, 'event') and hasattr(mrk.event, 'desc'):
        desc = np.asarray(mrk.event.desc)
        if desc.ndim > 1:
            desc = desc.squeeze()
        ev_codes = desc.flatten().astype(int)
    elif hasattr(mrk, 'y'):
        y = mrk.y
        ev_codes = y.flatten() if y.ndim == 1 else np.argmax(y, axis=0) + 1
    else:
        ev_codes = ev_pos.copy()

    events           = np.zeros((len(ev_pos), 3), dtype=int)
    events[:, 0]     = ev_pos
    events[:, 2]     = ev_codes

    print(f"EEG loaded: {raw.n_times} samples, {sfreq} Hz, {len(events)} markers")
    print(f"unique markers: {np.unique(ev_codes)}")

    # --- Load real electrode 3D positions ---
    mat_mnt  = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)
    mnt_key  = [k for k in mat_mnt if not k.startswith('__')][0]
    eeg_mnt  = mat_mnt[mnt_key]
    print(f"EEG mnt_nback.mat loaded: pos_3d shape {np.asarray(eeg_mnt.pos_3d).shape}")

    return raw, events, sfreq, eeg_mnt


# ============================================================
#  LOAD & PREPROCESS fNIRS DATA
# ============================================================
def load_and_preprocess_fnirs(subject_id):
    """
    Load fNIRS data, apply spatial mapping to 16x16 grid.
    Returns (fnirs_data, fnirs_events_time, fnirs_events_code).

    MSTFDN paper Sec. 2.2 fNIRS preprocessing:
      - Beer-Lambert already applied by dataset authors → use oxy/deoxy directly as HbO/HbR
      - NO wavelet / MARA / separate lowpass (model's large-kernel convolutions = implicit LPF)
      - Baseline correction performed inside extract_synchronized_windows (per trial)
      - z-score normalization (per-subject, in grid space, before windowing) - done in process_subject
    """
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')
    mnt_path = os.path.join(fnirs_folder, 'mnt_nback.mat')

    if not os.path.exists(cnt_path):
        raise FileNotFoundError(f"fNIRS file not found: {cnt_path}")

    print(f"Loading raw fNIRS from: {cnt_path}")

    mat_cnt  = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    cnt_key  = [k for k in mat_cnt if not k.startswith('__')][0]
    cnt      = mat_cnt[cnt_key]

    mat_mnt  = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)
    mnt_key  = [k for k in mat_mnt if not k.startswith('__')][0]
    mnt      = mat_mnt[mnt_key]

    # --- Load fNIRS trial markers (for cross-device synchronization) ---
    fnirs_events_time = None
    fnirs_events_code = None
    if os.path.exists(mrk_path):
        mat_fmrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
        fmrk_key = [k for k in mat_fmrk if not k.startswith('__')][0]
        fmrk     = mat_fmrk[fmrk_key]
        if hasattr(fmrk, 'time') and hasattr(fmrk, 'event') and hasattr(fmrk.event, 'desc'):
            fnirs_events_time = np.asarray(fmrk.time).flatten()
            fnirs_events_code = np.asarray(fmrk.event.desc).flatten()

    # --- Extract sampling rate ---
    sfreq_fnirs_raw = float(cnt.oxy.fs) if hasattr(cnt, 'oxy') and hasattr(cnt.oxy, 'fs') else 10.0
    print(f"raw fNIRS sampling rate: {sfreq_fnirs_raw} Hz")

    # --- Extract HbO / HbR (Beer-Lambert already applied by Shin et al. 2018) ---
    oxy_raw   = np.asarray(cnt.oxy.x,   dtype=np.float64)   # (n_samples, 36)
    deoxy_raw = np.asarray(cnt.deoxy.x, dtype=np.float64)
    print(f"Raw fNIRS: oxy {oxy_raw.shape}, deoxy {deoxy_raw.shape}")
    print("Using oxy/deoxy fields directly as HbO/HbR "
          "(Beer-Lambert already applied by dataset authors; see Shin et al. 2018 Data Descriptor)...")
    hbo_raw, hbr_raw = oxy_raw, deoxy_raw

    # --- No separate filtering (paper rationale: large-kernel conv = implicit LPF) ---
    print("Skipping wavelet/MARA/lowpass filtering "
          "(MSTFDN-style: relying on the model's own large-kernel convolutions for implicit smoothing)...")
    hbo_filtered, hbr_filtered = hbo_raw, hbr_raw

    # --- Resample to SFREQ_FNIRS (usually already 10 Hz, kept for generality) ---
    print(f"Resampling from {sfreq_fnirs_raw} Hz to {SFREQ_FNIRS} Hz...")
    n_samp      = int(hbo_filtered.shape[0] * SFREQ_FNIRS / sfreq_fnirs_raw)
    hbo_res     = resample(hbo_filtered, n_samp, axis=0)
    hbr_res     = resample(hbr_filtered, n_samp, axis=0)

    # --- Spatial mapping → (n_samples, 16, 16) per hemoglobin type ---
    print("Spatial mapping to 16x16...")
    hbo_mapped  = spatial_mapping(hbo_res, mnt)   # (n_samples, 16, 16)
    hbr_mapped  = spatial_mapping(hbr_res, mnt)

    # Stack: (n_samples, 16, 16, 2)   axis-3: [HbO, HbR]
    fnirs_data  = np.stack([hbo_mapped, hbr_mapped], axis=-1)
    print(f"fNIRS preprocessed: {fnirs_data.shape}")

    # --- QC plot: fNIRS before vs after ---
    _save_fnirs_qc_plot(subject_id, oxy_raw, hbo_res, sfreq_fnirs_raw)

    # Fix 3: delete all raw/intermediate arrays - only fnirs_data is needed downstream
    del mat_cnt, mat_mnt, oxy_raw, deoxy_raw
    del hbo_raw, hbr_raw, hbo_filtered, hbr_filtered
    del hbo_res, hbr_res, hbo_mapped, hbr_mapped
    try:
        del mat_fmrk, fmrk
    except NameError:
        pass
    gc.collect()

    return fnirs_data, fnirs_events_time, fnirs_events_code


def _save_fnirs_qc_plot(subject_id, oxy_raw, hbo_resampled, sfreq_raw):
    """Save before/after fNIRS QC plot for one representative channel."""
    try:
        ch = 0
        t_raw  = np.arange(oxy_raw.shape[0])      / sfreq_raw
        t_proc = np.arange(hbo_resampled.shape[0]) / SFREQ_FNIRS
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        axes[0].plot(t_raw,  oxy_raw[:, ch],        color='tab:red',  lw=0.6)
        axes[0].set_title(f"{subject_id} - fNIRS RAW HbO (ch {ch}, before preprocessing)")
        axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("HbO (raw units)")
        axes[1].plot(t_proc, hbo_resampled[:, ch],  color='tab:blue', lw=0.8)
        axes[1].set_title(f"{subject_id} - fNIRS PREPROCESSED HbO (ch {ch}, after resample+spatial map)")
        axes[1].set_xlabel("Time (s)"); axes[1].set_ylabel("HbO")
        plt.tight_layout()
        qc_dir = os.path.join(OUTPUT_PATH, subject_id)
        os.makedirs(qc_dir, exist_ok=True)
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=150)
        plt.close(fig)
        print(f"Saved fNIRS before/after QC plot for {subject_id}")
    except Exception as e:
        print(f"Could not save fNIRS QC plot for {subject_id}: {e}")


# ============================================================
#  EEG FREQUENCY-BAND DECOMPOSITION (E-TFEM real 5-band input)
# ============================================================
def bandpass_filter_eeg_bands(raw_clean: "mne.io.Raw"):
    """
    Filter the cleaned continuous EEG into 5 canonical bands
    (delta, theta, alpha, beta, gamma) using zero-phase FIR filters.

    IMPORTANT: This must run on raw_clean (post-ICA, post-CAR) so that
    each band inherits the artifact-free signal. Filtering is done on
    a fresh copy per band - raw_clean itself is left untouched.

    Implemented as a GENERATOR (not returning a dict of all 5 bands at
    once) so that only ONE band's array (~120 MB) exists in memory at a
    time - the caller spatial-maps and discards each band before the
    next is filtered. This keeps peak RAM per worker bounded, which
    matters when running 6 parallel workers on a laptop.

    Yields
    ------
    (band_name, band_array) : (str, np.ndarray)
        band_array shape (n_ch, n_times), float64, Volts
    """
    for band_name, (lo, hi) in EEG_BANDS.items():
        # Fresh copy per band - MNE filter() is in-place, so copy first
        raw_band = raw_clean.copy().filter(
            l_freq=lo, h_freq=hi,
            method='fir', phase='zero',
            verbose=False)
        band_array = raw_band.get_data()   # (n_ch, n_times)
        del raw_band   # free the MNE Raw copy immediately, keep only ndarray
        yield band_name, band_array
        # caller must not hold a reference to band_array past its loop
        # iteration - the next yield reuses the memory slot conceptually


# ============================================================
#  PREPROCESS EEG
# ============================================================
def preprocess_eeg(raw, events):
    """
    Full EEG preprocessing chain (MSTFDN paper Sec. 2.2):
      1. 50 Hz notch filter
      2. 1-40 Hz FIR bandpass
      3. Bad-channel detection (std + correlation criterion, 4.5xMAD)
      4. Spherical-spline interpolation of bad channels
      5. Common Average Reference (CAR)
      6. ICA with picard (99% variance) + ICLabel + EOG correlation
      7. Drop VEOG / HEOG / TP9

    Returns cleaned MNE Raw object.
    """
    print("Preprocessing EEG...")
    raw_before_qc = raw.copy()   # kept for QC plot

    # 1. Notch + bandpass
    raw_f = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    raw_f.filter(l_freq=1.0, h_freq=40.0, method='fir', verbose=False)

    # 2. Bad-channel detection
    raw_tmp = raw_f.copy().filter(l_freq=10.0, h_freq=40.0, method='fir', verbose=False)
    eeg_picks = mne.pick_types(raw_tmp.info, eeg=True, eog=False)
    eeg_names = [raw_tmp.ch_names[i] for i in eeg_picks]
    temp_data = raw_tmp.get_data(picks=eeg_picks)

    stds       = np.std(temp_data, axis=1)
    med_std    = np.median(stds)
    mad        = np.median(np.abs(stds - med_std))
    mad        = mad if mad > 0 else 1e-6
    corr_mat   = np.corrcoef(temp_data)
    mean_corrs = np.mean(corr_mat, axis=1)

    bad_ch = []
    for i, name in enumerate(eeg_names):
        if name.upper() in ('VEOG', 'HEOG', 'TP9'):
            continue
        if (stds[i] > med_std + 4.5 * mad or
                stds[i] < med_std - 4.5 * mad or
                mean_corrs[i] < 0.30):
            bad_ch.append(name)

    if bad_ch:
        print(f"Bad channels: {bad_ch}")
        raw_f.info['bads'] = bad_ch

    interpolated_ch = list(raw_f.info['bads'])

    if raw_f.info['bads']:
        raw_f.interpolate_bads(reset_bads=True, verbose=False)

    # 3. CAR
    raw_f.set_eeg_reference('average', projection=False, verbose=False)

    # 4. ICA (picard, 99% variance - automatically handles rank reduction
    #         from interpolated channels without manual counting)
    picks_ica = mne.pick_types(raw_f.info, eeg=True, eog=False, exclude='bads')
    print(f"ICA: using variance-based component selection (99% explained variance), "
          f"picks={len(picks_ica)}, interpolated={len(interpolated_ch)}")
    ica = mne.preprocessing.ICA(
        n_components=0.99, method='picard',
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False)
    ica.fit(raw_f, picks=picks_ica)

    # 5. ICLabel classification
    ic_labels = mne_icalabel.label_components(raw_f, ica, method='iclabel')

    ARTIFACT_CLASSES = {
        'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
        'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6}
    ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
    PROB_THRESH      = 0.30
    BRAIN_CEIL       = 0.50    # rescue EOG-only hits if ICLabel brain-prob >= 0.5

    proba = ica.labels_scores_    # (n_components, 7)

    # argmax-based
    idx_argmax = {i for i, lbl in enumerate(ic_labels['labels'])
                  if lbl in ARTIFACT_TARGETS}
    # probability-threshold based
    idx_prob = set()
    for ci in range(proba.shape[0]):
        for cls in ARTIFACT_TARGETS:
            col = ARTIFACT_CLASSES.get(cls)
            if col is not None and proba[ci, col] >= PROB_THRESH:
                idx_prob.add(ci)
                break

    # EOG correlation
    idx_veog, idx_heog = [], []
    try:
        if 'VEOG' in raw.ch_names:
            idx_veog, _ = ica.find_bads_eog(raw, ch_name='VEOG', threshold=2.0, verbose=False)
            print(f"  find_bads_eog(VEOG) flagged components: {idx_veog}")
        if 'HEOG' in raw.ch_names:
            idx_heog, _ = ica.find_bads_eog(raw, ch_name='HEOG', threshold=2.0, verbose=False)
            print(f"  find_bads_eog(HEOG) flagged components: {idx_heog}")
    except Exception as e:
        print(f"  find_bads_eog failed: {e}")

    # EOG-only rescue: if a component is flagged ONLY by EOG correlation (no ICLabel support)
    # but ICLabel thinks it's mostly brain signal, keep it
    all_flagged = idx_argmax | idx_prob
    eog_only    = (set(idx_veog) | set(idx_heog)) - all_flagged
    rescued     = {i for i in eog_only if proba[i, ARTIFACT_CLASSES['brain']] >= BRAIN_CEIL}
    if rescued:
        print(f"  Rescued from EOG-only rejection (ICLabel brain-prob >= {BRAIN_CEIL}): {sorted(rescued)}")

    artifact_idx = sorted((idx_argmax | idx_prob | set(idx_veog) | set(idx_heog)) - rescued)

    # Detailed per-component log
    class_names = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
                   'line noise', 'channel noise', 'other']
    for ci in artifact_idx:
        probs_str = ", ".join(f"{class_names[c]}={proba[ci, c]:.2f}" for c in range(7))
        reasons = []
        if ci in idx_argmax: reasons.append(f"argmax='{ic_labels['labels'][ci]}'")
        if ci in idx_prob:   reasons.append(f"prob>={PROB_THRESH}")
        if ci in idx_veog:   reasons.append("VEOG-correlation")
        if ci in idx_heog:   reasons.append("HEOG-correlation")
        print(f"  IC{ci} flagged [{', '.join(reasons)}]: {probs_str}")

    ica.exclude = artifact_idx
    if ica.exclude:
        ica.apply(raw_f, verbose=False)
        print(f"Removed {len(ica.exclude)} ICA components")
    else:
        print("Removed 0 ICA components (no artifact ICs detected)")

    # Fix 4: ICA object holds a full copy of training data internally (~hundreds MB).
    # Delete immediately after apply - it is never needed again.
    del ica, ic_labels, proba
    gc.collect()

    # 6. Drop EOG + TP9
    drop_ch = [ch for ch in raw_f.ch_names if ch.upper() in ('VEOG', 'HEOG', 'TP9')]
    if drop_ch:
        raw_f.drop_channels(drop_ch)
        print(f"Dropped channels: {drop_ch}")

    # 7. QC plots (all channels, stacked)
    _save_eeg_qc_plots(raw_before_qc, raw_f)

    return raw_f


def _plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
    """Stacked EEG montage plot for all channels."""
    sfreq_p = raw_obj.info['sfreq']
    if tmin is not None or tmax is not None:
        s0 = int((tmin or 0.0) * sfreq_p)
        s1 = int(tmax * sfreq_p) if tmax is not None else None
        data = raw_obj.get_data(start=s0, stop=s1)
        t    = (np.arange(data.shape[1]) + s0) / sfreq_p
    else:
        data = raw_obj.get_data()
        t    = np.arange(data.shape[1]) / sfreq_p
    n_ch   = data.shape[0]
    spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 30 + 1e-12
    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * n_ch)))
    for i in range(n_ch):
        ax.plot(t, data[i] + (n_ch - 1 - i) * spread, color=color, lw=0.4)
    ax.set_yticks([(n_ch - 1 - i) * spread for i in range(n_ch)])
    ax.set_yticklabels(raw_obj.ch_names, fontsize=6)
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def _save_eeg_qc_plots(raw_before, raw_after):
    """Save all-channel before/after EEG QC plots to a temp dir;
    process_subject renames them to per-subject files."""
    try:
        tmp = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc")
        os.makedirs(tmp, exist_ok=True)
        figs = [
            (_plot_all_channels_stacked(raw_before, "EEG RAW - All Channels (before)", 'tab:red'),
             "last_eeg_before_allchannels.png"),
            (_plot_all_channels_stacked(raw_after,  "EEG PREPROCESSED - All Channels (after)", 'tab:blue'),
             "last_eeg_after_allchannels.png"),
            (_plot_all_channels_stacked(raw_before, "EEG RAW - All Channels (10-30 s)", 'tab:red', 10, 30),
             "last_eeg_before_allchannels_zoom10_30.png"),
            (_plot_all_channels_stacked(raw_after,  "EEG PREPROCESSED - All Channels (10-30 s)", 'tab:blue', 10, 30),
             "last_eeg_after_allchannels_zoom10_30.png"),
        ]
        for fig, fname in figs:
            fig.savefig(os.path.join(tmp, fname), dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"Could not save EEG QC plots: {e}")


# ============================================================
#  SYNCHRONIZED ASYMMETRIC SLIDING-WINDOW EXTRACTION
# ============================================================
def extract_synchronized_windows(eeg_raw, eeg_spatial_data, fnirs_data, events,
                                  sfreq_eeg, fnirs_events_time, fnirs_events_code):
    """
    Extract paired (EEG, fNIRS) windows for every sliding position within each trial.

    EEG window  : (5_bands, 16, 16, EEG_WINDOW_LENxsfreq_eeg)  ← from pre-computed
                  5-band grid (delta/theta/alpha/beta/gamma), NOT a broadcast copy
    fNIRS window: (FNIRS_NUM_SEGMENTS, 16, 16, FNIRS_WINDOW_LENxSFREQ_FNIRS, 2)
                  i.e. (21, 16, 16, 30, 2)

    Key synchronization fix: fNIRS windows are anchored to fNIRS's OWN trial
    markers (codes 7/8/9) not to EEG's clock, eliminating ~46-93 s cross-session drift.

    Returns
    -------
    eeg_windows  : ndarray (N, 5, 16, 16, 2000)
    fnirs_windows: ndarray (N, 21, 16, 16, 30, 2)
    labels       : ndarray (N,)
    session_ids  : ndarray (N,)  values 1/2/3
    trial_ids    : ndarray (N,)  trial index 0..26 per subject
    """
    print("Extracting synchronized asymmetric sliding windows...")
    print(f"EEG window: {EEG_WINDOW_LEN}s, step: {EEG_STEP}s")
    print(f"fNIRS window: {FNIRS_WINDOW_LEN}s, step: {FNIRS_STEP}s")

    # --- EEG trial-onset events ---
    label_map    = EEG_TRIAL_LABEL_MAP
    valid_codes  = list(label_map.keys())
    target_ev    = events[np.isin(events[:, 2], valid_codes)]
    print(f"found {len(target_ev)} trial markers")

    if len(target_ev) == 0:
        print(f"No trial markers found. Available: {np.unique(events[:, 2])}")
        return (np.array([]), np.array([]), np.array([]),
                np.array([]), np.array([]))

    # --- Build fNIRS's own trial-onset time array (independent clock) ---
    fnirs_sync_ok    = False
    fnirs_trial_times = None
    if fnirs_events_time is not None and fnirs_events_code is not None:
        f_valid   = np.isin(fnirs_events_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
        f_times   = np.asarray(fnirs_events_time)[f_valid].astype(float)
        f_codes   = np.asarray(fnirs_events_code)[f_valid]
        f_order   = np.argsort(f_times)
        f_times   = f_times[f_order] / 1000.0          # ms → s
        f_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in f_codes[f_order]])

        eeg_classes = np.array([label_map.get(int(ev[2]), -1) for ev in target_ev])
        n_common    = min(len(f_classes), len(eeg_classes))

        if (len(f_times) == len(target_ev) and
                np.array_equal(f_classes[:n_common], eeg_classes[:n_common])):
            fnirs_sync_ok     = True
            fnirs_trial_times = f_times
            print(f"fNIRS-own-marker sync check: OK ({len(f_times)} trials, class order matches EEG)")
        else:
            print(f"  WARNING: fNIRS own markers do not match EEG 1:1 "
                  f"(fNIRS n={len(f_times)}, EEG n={len(target_ev)}) - "
                  f"falling back to EEG-clock-derived fNIRS timing.")
    else:
        print("  WARNING: fNIRS event markers unavailable - falling back to EEG-clock timing.")

    win_samp_eeg   = int(EEG_WINDOW_LEN  * sfreq_eeg)
    win_samp_fnirs = int(FNIRS_WINDOW_LEN * SFREQ_FNIRS)
    max_slide      = EEG_EPOCH_TMAX - EEG_WINDOW_LEN

    eeg_windows   = []
    fnirs_windows = []
    labels        = []
    session_ids   = []
    trial_ids     = []

    for trial_idx, ev in enumerate(target_ev):
        ev_samp_eeg = ev[0]
        raw_code    = ev[2]
        label       = label_map.get(raw_code, -1)
        if label == -1:
            continue

        session_id = trial_idx // TRIALS_PER_SESSION + 1

        eeg_t   = ev_samp_eeg / sfreq_eeg
        fnirs_t = fnirs_trial_times[trial_idx] if fnirs_sync_ok else eeg_t

        # Per-trial fNIRS baseline correction (-5 to -2 s before trial onset)
        bs_start = int(np.round((fnirs_t + BASELINE_TMIN)  * SFREQ_FNIRS))
        bs_end   = int(np.round((fnirs_t + BASELINE_TMAX)  * SFREQ_FNIRS))
        if bs_start >= 0 and bs_end > bs_start:
            fnirs_baseline = fnirs_data[bs_start:bs_end].mean(axis=0, keepdims=True)  # (1,16,16,2)
        else:
            fnirs_baseline = 0.0   # first trial: no pre-trial data

        for offset in np.arange(0.0, max_slide + 0.001, EEG_STEP):

            # --- EEG window ---
            w0_eeg = int(ev_samp_eeg + np.round(offset * sfreq_eeg))
            w1_eeg = w0_eeg + win_samp_eeg
            if w1_eeg > eeg_raw.n_times:
                continue

            # Amplitude-based artifact rejection (channel-space, 120 μV threshold)
            raw_slice = eeg_raw.get_data(start=w0_eeg, stop=w1_eeg)
            _reject = np.max(np.abs(raw_slice)) > 120e-6
            del raw_slice   # Fix 5a: free immediately - (n_ch x 2000) float64 per window
            if _reject:
                continue

            # Slice from pre-computed 5-band spatial grid:
            # eeg_spatial_data shape (n_times, 5, 16, 16) → (T,5,16,16) → (5,16,16,T)
            eeg_grid = eeg_spatial_data[w0_eeg:w1_eeg]          # (2000, 5, 16, 16)
            # Cast to float16 immediately: with real 5-band data the window
            # list would otherwise hold ~8 GB of float32 slices simultaneously
            # before the final np.array() conversion. float16 halves that.
            eeg_slice = eeg_grid.transpose(1, 2, 3, 0).astype(np.float16)   # (5, 16, 16, 2000)

            # --- fNIRS segments (FNIRS_NUM_SEGMENTS x FNIRS_WINDOW_LEN) ---
            segments   = []
            valid_segs = True
            for si in range(FNIRS_NUM_SEGMENTS):
                seg_off  = offset + si * FNIRS_STEP
                w0_f     = int(np.round((fnirs_t + seg_off) * SFREQ_FNIRS))
                w1_f     = w0_f + win_samp_fnirs
                if w0_f < 0 or w1_f > fnirs_data.shape[0]:
                    valid_segs = False
                    break
                seg = fnirs_data[w0_f:w1_f] - fnirs_baseline   # (30, 16, 16, 2)
                segments.append(seg)

            if not valid_segs:
                continue

            # Stack segments: (21, 30, 16, 16, 2) → transpose → (21, 16, 16, 30, 2)
            fnirs_stack = np.stack(segments, axis=0)             # (21, 30, 16, 16, 2)
            fnirs_slice = fnirs_stack.transpose(0, 2, 3, 1, 4).copy()  # (21, 16, 16, 30, 2)
            # Fix 5b: free segments list and intermediate stack immediately
            del segments, fnirs_stack

            eeg_windows.append(eeg_slice)
            fnirs_windows.append(fnirs_slice)
            labels.append(label)
            session_ids.append(session_id)
            trial_ids.append(trial_idx)

    # Fix 6: convert lists to arrays one at a time and immediately free the list
    # to avoid the peak of holding BOTH list + array simultaneously in RAM
    # Fix 6b: eeg_windows list already holds float16 slices (cast at creation
    # above) - np.array() here just concatenates, no extra precision loss.
    # fnirs stays float32 (smaller total volume, precision matters more for
    # the smaller-amplitude hemoglobin signal).
    _eeg_list    = eeg_windows;    eeg_windows   = np.array(_eeg_list,   dtype=np.float16);  del _eeg_list
    _fnirs_list  = fnirs_windows;  fnirs_windows = np.array(_fnirs_list, dtype=np.float32);  del _fnirs_list
    _lbl_list    = labels;         labels        = np.array(_lbl_list,   dtype=np.int64);    del _lbl_list
    _sess_list   = session_ids;    session_ids   = np.array(_sess_list,  dtype=np.int32);    del _sess_list
    _trial_list  = trial_ids;      trial_ids     = np.array(_trial_list, dtype=np.int32);    del _trial_list
    gc.collect()

    print(f"Extracted {len(labels)} synchronized windows")
    if len(labels) > 0:
        print(f"Label distribution: {np.bincount(labels)}")

    return eeg_windows, fnirs_windows, labels, session_ids, trial_ids


# ============================================================
#  MAIN SUBJECT PROCESSING
# ============================================================
def process_subject(subject_id):
    t0 = time.time()
    log_message(f"START processing {subject_id}")
    print(f"Processing {subject_id}")

    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    if not os.path.exists(eeg_folder):
        log_message(f"SKIPPED {subject_id}: EEG folder not found")
        print(f"EEG folder not found: {eeg_folder}")
        return

    # Fix 7: pre-declare ALL heavy locals as None so the finally block
    # can safely del them even if an exception happened before they were assigned
    raw_eeg = events = eeg_mnt_struct = None
    fnirs_data = fnirs_events_time = fnirs_events_code = None
    raw_eeg_clean = eeg_full = eeg_mnt = eeg_spatial = None
    eeg_windows = fnirs_windows = labels = session_ids = trial_ids = None

    try:
        # ── 1. Load EEG ──────────────────────────────────────────────────────
        print("\n Loading EEG...")
        raw_eeg, events, sfreq_eeg, eeg_mnt_struct = load_eeg_data(eeg_folder)

        # ── 2. Load & preprocess fNIRS ────────────────────────────────────────
        print("\n Loading and preprocessing fNIRS...")
        fnirs_data, fnirs_events_time, fnirs_events_code = \
            load_and_preprocess_fnirs(subject_id)

        # ── 3. Normalize fNIRS BEFORE windowing (per-subject, grid space) ─────
        #       mirrors per-subject z-score used for EEG below
        print("\n Normalizing fNIRS (subject-level, before windowing)...")
        fnirs_data = ((fnirs_data - fnirs_data.mean(axis=0, keepdims=True)) /
                      (fnirs_data.std(axis=0, keepdims=True) + 1e-8))

        # ── 4. Preprocess EEG ─────────────────────────────────────────────────
        print("\n Preprocessing EEG...")
        raw_eeg_clean = preprocess_eeg(raw_eeg, events)

        # Save cleaned continuous EEG (for QC / debugging)
        sub_out = os.path.join(OUTPUT_PATH, subject_id)
        os.makedirs(sub_out, exist_ok=True)
        raw_eeg_clean.save(
            os.path.join(sub_out, f"{subject_id}_eeg_clean_raw.fif"),
            overwrite=True)

        # EEG per-subject stats (informational only; normalization happens on grid)
        eeg_full = raw_eeg_clean.get_data()   # (n_ch, n_times) - broadband, for logging only
        print(f"\n EEG per-subject normalization stats computed: "
              f"mean range [{eeg_full.mean(axis=-1).min():.2e}, "
              f"{eeg_full.mean(axis=-1).max():.2e}], "
              f"std range [{eeg_full.std(axis=-1).min():.2e}, "
              f"{eeg_full.std(axis=-1).max():.2e}]")
        del eeg_full
        eeg_full = None

        # ── 5. Real 5-band frequency decomposition + spatial mapping ─────────
        # Paper E-TFEM (Sec. 2.2.1) requires 5 SPECTRALLY DISTINCT input
        # channels (delta/theta/alpha/beta/gamma) - NOT 5 identical copies
        # of the broadband signal. Each band is filtered from raw_eeg_clean
        # independently, then spatially mapped to its own 16x16 grid.
        print("\n Filtering EEG into 5 frequency bands "
              f"(delta/theta/alpha/beta/gamma)...")
        eeg_mnt = get_eeg_mnt(raw_eeg_clean, eeg_mnt_struct)

        band_spatial_list = []   # will hold 5 arrays, each (n_times, 16, 16)
        for band_name, band_arr in bandpass_filter_eeg_bands(raw_eeg_clean):
            print(f"   Spatial mapping band '{band_name}' "
                  f"({EEG_BANDS[band_name][0]}-{EEG_BANDS[band_name][1]} Hz) to 16x16...")
            band_grid = spatial_mapping(band_arr.T, eeg_mnt)      # (n_times, 16, 16)
            del band_arr   # free this band's channel-space array (~120 MB) NOW

            # Per-subject, per-band z-score in grid space (before windowing)
            band_grid = ((band_grid - band_grid.mean(axis=0, keepdims=True)) /
                         (band_grid.std(axis=0, keepdims=True) + 1e-8))

            band_spatial_list.append(band_grid.astype(np.float32))
            gc.collect()   # only ONE band's raw array alive at any time

        # Stack into (n_times, n_bands, 16, 16)
        eeg_spatial = np.stack(band_spatial_list, axis=1)   # (n_times, 5, 16, 16)
        del band_spatial_list
        gc.collect()
        print(f"EEG spatially mapped (5-band): {eeg_spatial.shape}")

        # Rename temp EEG QC plots to per-subject files
        _rename_eeg_qc_plots(subject_id, sub_out)


        # ── 6. Synchronized window extraction ────────────────────────────────
        print("\n Extracting synchronized asymmetric windows...")
        eeg_windows, fnirs_windows, labels, session_ids, trial_ids = \
            extract_synchronized_windows(
                raw_eeg_clean, eeg_spatial, fnirs_data,
                events, sfreq_eeg,
                fnirs_events_time, fnirs_events_code)

        if len(labels) == 0:
            log_message(f"WARNING {subject_id}: no windows extracted")
            print(f"No windows extracted for {subject_id}")
            return

        # Fix 7c: eeg_spatial (~123 MB) and fnirs_data (~12 MB) are no longer
        # needed after windowing - free them before allocating save buffers
        del eeg_spatial, fnirs_data, raw_eeg_clean, raw_eeg
        eeg_spatial = fnirs_data = raw_eeg_clean = raw_eeg = None
        gc.collect()

        # ── 7. Save outputs ───────────────────────────────────────────────────
        print("\n Saving...")
        np.save(os.path.join(sub_out, f"{subject_id}_eeg_windows.npy"),   eeg_windows)
        np.save(os.path.join(sub_out, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)
        np.save(os.path.join(sub_out, f"{subject_id}_labels.npy"),        labels)
        np.save(os.path.join(sub_out, f"{subject_id}_session_ids.npy"),   session_ids)
        np.save(os.path.join(sub_out, f"{subject_id}_trial_ids.npy"),     trial_ids)

        # Metadata JSON
        metadata = {
            'subject_id':             subject_id,
            'eeg_shape':              list(eeg_windows.shape),
            'eeg_bands':              EEG_BAND_NAMES,
            'eeg_band_ranges_hz':     EEG_BANDS,
            'fnirs_shape':            list(fnirs_windows.shape),
            'labels_distribution':    np.bincount(labels).tolist(),
            'eeg_window_len':         EEG_WINDOW_LEN,
            'eeg_step':               EEG_STEP,
            'eeg_epoch_tmax':         EEG_EPOCH_TMAX,
            'fnirs_num_segments':     FNIRS_NUM_SEGMENTS,
            'fnirs_window_len':       FNIRS_WINDOW_LEN,
            'fnirs_step':             FNIRS_STEP,
            'sfreq_eeg':              SFREQ_EEG,
            'sfreq_fnirs':            SFREQ_FNIRS,
            'beer_lambert_applied':   False,
            'fnirs_sync_fix_applied': True,
            'normalization':          'per-subject z-score (EEG grid + fNIRS grid, before windowing)',
            'trial_ids_saved':        True,
        }
        with open(os.path.join(sub_out, f"{subject_id}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n {subject_id} completed!")
        print(f"EEG windows:   {eeg_windows.shape}")
        print(f"fNIRS windows: {fnirs_windows.shape}")
        print(f"Labels:        {np.bincount(labels)}")

        elapsed = (time.time() - t0) / 60
        log_message(
            f"DONE {subject_id} in {elapsed:.1f} min | "
            f"EEG: {eeg_windows.shape} | fNIRS: {fnirs_windows.shape} | "
            f"Labels: {np.bincount(labels)}")

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        log_message(f"ERROR processing {subject_id} after {elapsed:.1f} min: {str(e)}")
        print(f"Error processing {subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # Fix 7d: GUARANTEED cleanup - runs even if an exception occurred mid-subject.
        # Prevents a failed subject's half-built arrays from accumulating in RAM
        # and crashing the NEXT subject's processing.
        # All variables were pre-declared as None at function entry, so each
        # del is always safe - no NameError can occur.
        # NOTE: we use explicit 'del varname' (NOT del locals()[name]) because
        # locals() in CPython is a snapshot; mutating it has NO effect on actual memory.
        del raw_eeg
        del raw_eeg_clean
        del eeg_full
        del eeg_spatial
        del fnirs_data
        del eeg_windows
        del fnirs_windows
        del labels
        del session_ids
        del trial_ids
        del eeg_mnt
        del eeg_mnt_struct
        del events
        del fnirs_events_time
        del fnirs_events_code
        gc.collect()


def _rename_eeg_qc_plots(subject_id, sub_out):
    """Move temp EEG QC plots produced inside preprocess_eeg() to subject folder."""
    try:
        tmp = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc")
        name_pairs = [
            ("last_eeg_before_allchannels.png",
             f"{subject_id}_eeg_before_allchannels.png"),
            ("last_eeg_after_allchannels.png",
             f"{subject_id}_eeg_after_allchannels.png"),
            ("last_eeg_before_allchannels_zoom10_30.png",
             f"{subject_id}_eeg_before_allchannels_zoom10_30.png"),
            ("last_eeg_after_allchannels_zoom10_30.png",
             f"{subject_id}_eeg_after_allchannels_zoom10_30.png"),
        ]
        for src_name, dst_name in name_pairs:
            src = os.path.join(tmp, src_name)
            dst = os.path.join(sub_out, dst_name)
            if os.path.exists(src):
                os.replace(src, dst)
        print(f"Saved EEG before/after (all-channels) QC plots for {subject_id}")
    except Exception as e:
        print(f"Could not finalize EEG QC plot for {subject_id}: {e}")


# ============================================================
#  PARALLEL ENTRY POINT
#  Tuned for: i7-12700H (14 physical cores) + 20.5 GB RAM free
#  Strategy : spawn-based Pool - safe on Windows, no fork issues
#
#  N_WORKERS = 2  →  2 x ~9 GB peak = 18 GB  (fits in 20.5 GB free)
#
#  IMPORTANT: with REAL 5-band EEG filtering (not a broadcast copy),
#  eeg_windows is 5x larger than the single-band version. Peak RAM per
#  subject is now ~9 GB (was ~3 GB), so N_WORKERS was reduced from 6 to 2
#  to stay within 20.5 GB free RAM. Do not raise this without more RAM -
#  3 workers x 9 GB = 27 GB would very likely OOM-crash mid-run.
#  Each worker processes one subject fully and independently.
#  MNE / ICA are NOT thread-safe but ARE process-safe (spawn).
#
#  HOW TO RUN:
#    python MSTFDN_preprocessing_parallel.py
#  (must be run as a script, NOT inside Jupyter - see note below)
#
#  If you want to run from Jupyter, use the companion notebook cell
#  at the bottom of this file.
# ============================================================

N_WORKERS = 2   # ← reduced from 6: real 5-band EEG data ~9 GB peak/subject


def _worker_wrapper(subject_id: str) -> str:
    """
    Top-level function required by multiprocessing on Windows (spawn).
    Must be defined at module level (not inside __main__) so each
    worker process can import it cleanly.

    Returns a short status string for the progress bar.
    """
    import mne
    mne.set_log_level('WARNING')   # suppress per-subject MNE verbosity

    t0 = time.time()
    try:
        process_subject(subject_id)
        elapsed = (time.time() - t0) / 60
        return f"OK  {subject_id}  {elapsed:.1f} min"
    except Exception as e:
        elapsed = (time.time() - t0) / 60
        return f"ERR {subject_id}  {elapsed:.1f} min  - {e}"


def _progress_callback(result: str):
    """Called in the MAIN process when a worker finishes one subject."""
    tag = "[OK ]" if result.startswith("OK") else "[ERR]"
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"[{ts}] {tag} {result}"
    # print with ASCII-safe encoding for Windows cp1252 terminal
    print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding), flush=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_parallel(subjects: list, n_workers: int = N_WORKERS):
    """
    Launch preprocessing for all subjects using a spawn-based Pool.

    Parameters
    ----------
    subjects  : list of subject IDs, e.g. ['VP001', 'VP002', ...]
    n_workers : number of parallel processes (default = N_WORKERS = 2)

    Notes
    -----
    - Uses 'spawn' context - required on Windows, also safer on Linux/Mac
      because MNE uses OpenMP internally and fork + OpenMP = deadlocks.
    - imap_unordered: subjects finish in any order; progress printed as
      they complete so you can monitor without waiting for all 26.
    - chunksize=1: each worker gets one subject at a time - correct because
      subjects vary widely in processing time (ICA is the bottleneck).
    """
    import multiprocessing as mp

    total_t0 = time.time()
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    log_message("=" * 55)
    log_message(f"PARALLEL PIPELINE START - {len(subjects)} subjects, {n_workers} workers")
    log_message(f"System: i7-12700H, 14 physical cores, ~20.5 GB RAM free")
    log_message(f"RAM budget: {n_workers} workers x ~9 GB peak = ~{n_workers*9} GB")
    log_message("=" * 55)

    print(f"\n{'='*55}")
    print(f"  MSTFDN Parallel Preprocessing")
    print(f"  Subjects : {len(subjects)}")
    print(f"  Workers  : {n_workers}")
    print(f"  Est. time: {len(subjects) * 15 / n_workers:.0f}-"
          f"{len(subjects) * 22 / n_workers:.0f} min  "
          f"(vs {len(subjects)*15}-{len(subjects)*22} min serial)")
    print(f"{'='*55}\n")

    # ── spawn context (safe on Windows + MNE) ────────────────────────
    ctx  = mp.get_context('spawn')
    done = 0

    with ctx.Pool(
        processes=n_workers,
        maxtasksperchild=1,   # restart worker process after each subject
                               # → prevents memory leaks between subjects
    ) as pool:
        for result in pool.imap_unordered(
            _worker_wrapper,
            subjects,
            chunksize=1,
        ):
            done += 1
            _progress_callback(result)
            print(f"  Progress: {done}/{len(subjects)} subjects done", flush=True)

    total_elapsed = (time.time() - total_t0) / 60
    log_message(f"PARALLEL PIPELINE FINISHED in {total_elapsed:.1f} min")
    print(f"\n[OK] All {len(subjects)} subjects finished in {total_elapsed:.1f} min")


# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # All 26 subjects
    ALL_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]

    # ── Quick single-subject test (uncomment to verify before full run) ──
    # run_parallel(["VP001"], n_workers=1)

    # ── Full parallel run ────────────────────────────────────────────────
    run_parallel(ALL_SUBJECTS, n_workers=N_WORKERS)


# ============================================================
#  HOW TO CALL FROM JUPYTER NOTEBOOK
#  (copy this into a notebook cell)
# ============================================================
#
#  import subprocess, sys
#
#  result = subprocess.run(
#      [sys.executable,
#       r"E:\kntu\Thesis\codes\HOPE-LAST-CODE-FILE\A1b\New folder\MSTFDN_preprocessing_parallel.py"],
#      capture_output=True, text=True
#  )
#  print(result.stdout)
#  if result.returncode != 0:
#      print("STDERR:", result.stderr)
#
# ── OR: monitor the log file while it runs in a separate terminal ────────
#
#  import time, os
#  log = r"E:\kntu\Thesis\codes\Results\MSTFDN_PREPROCESSED_NBACK\processing_log.txt"
#  seen = 0
#  for _ in range(3600):          # watch for up to 1 hour
#      if os.path.exists(log):
#          lines = open(log, encoding='utf-8').readlines()
#          for line in lines[seen:]:
#              print(line, end='')
#          seen = len(lines)
#      time.sleep(10)             # refresh every 10 s
