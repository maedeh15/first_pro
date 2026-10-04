"""
MSTFDN Preprocessing Pipeline v3 -- N-back Dataset (Shin et al. 2018)
=======================================================================
CORRECTED to match the EXACT protocol described in paper Sec. 3.2:

  "For the N-back sub-dataset (MBI dataset-A), we extracted 42 s task data
   segment containing both 'Instruction' and 'Task' based on task markers
   and the corresponding 21 s (~20 s) rest data segment. ... a sliding
   window operation with a 1 s step size and a 10 s window size was applied
   to two multimodal data segments. The number of windows per 42 s task
   data segment was n=33, and for a rest data segment, the number of
   windows was n=11. Each session ... consists of 9 trials, with 3 trials
   for each of the three N-back classes and 9 trials for rest class.
   Therefore, each session generates a total of 99 samples for each class.
   The total sample size across the 3 sessions is 1188 ...
   The individual fNIRS sample size for the N-back sub-dataset is
   3 x 36 x 100 (hemoglobin type x channel x time), ... the EEG sample
   size is 5 x 28 x 2000 (band type x channel x time)."

KEY CORRECTIONS vs the previous (v2/bands) implementation:
  1. N-back is a 4-CLASS problem: 0-back, 2-back, 3-back, REST
     (previous version only extracted the 3 task classes, dropping REST
     entirely -- this alone changes the class distribution and the
     model's decision boundary).
  2. Task segment = 42 s (2 s instruction + 40 s task), not 40 s.
  3. Rest segment = ~20 s, taken immediately after each task trial's
     42 s segment (before the next trial begins). ALL 9 trials
     contribute a rest segment (not just 3).
  4. Samples are stored in raw CHANNEL SPACE, NOT spatially mapped to a
     16x16 grid. The paper's reported sample shapes (5x28x2000 for EEG,
     3x36x100 for fNIRS) are channel x time -- spatial mapping into a
     2-D grid, if used at all, happens INSIDE the model (E-SFEM/F-SFEM),
     not as a stored preprocessing artifact.
  5. fNIRS sample = ONE 10 s window (100 samples @ 10 Hz) with 3
     hemoglobin channels (HbO, HbR, HbT = HbO+HbR), matching the EEG
     window 1:1 -- NOT 21 stacked overlapping 3 s segments.
  6. Expected total samples per subject: 1188 (99 x 4 classes x 3
     sessions), before any artifact-rejection dropouts.

Dataset: Shin et al. 2018 -- 26 subjects (VP001-VP026), 30 EEG ch @200Hz,
         36 fNIRS ch @10Hz, 3 sessions, 9 trials/session.

Saved outputs per subject (in OUTPUT_PATH/VPxxx/):
  - {id}_eeg_windows.npy     shape (N, 5, 28, 2000)  float16
  - {id}_fnirs_windows.npy   shape (N, 3, 36, 100)    float32
  - {id}_labels.npy          shape (N,)  int  [0=0back,1=2back,2=3back,3=rest]
  - {id}_session_ids.npy     shape (N,)  int  [1,2,3]
  - {id}_trial_ids.npy       shape (N,)  int  trial index 0..8 within session
  - {id}_eeg_clean_raw.fif   cleaned continuous EEG (MNE Raw, for QC)
  - QC plots (same as before)
  - {id}_metadata.json
  - processing_log.txt
"""

import os
import gc
import sys
import json
import time
import datetime
import warnings

import numpy as np
import scipy.io as sio
from scipy.signal import resample
import matplotlib.pyplot as plt
import mne
import mne_icalabel

warnings.filterwarnings('ignore')

# ============================================================
#  CONFIGURATION
# ============================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
OUTPUT_PATH        = r"E:\kntu\Thesis\codes\Results\MSTFDN_PREPROCESSED_NBACK_V3"

SFREQ_EEG   = 200.0
SFREQ_FNIRS = 10.0

# --- EEG frequency bands (E-TFEM groups=5) ---
EEG_BANDS = {
    'delta': (0.5,  4.0),
    'theta': (4.0,  8.0),
    'alpha': (8.0,  13.0),
    'beta':  (13.0, 30.0),
    'gamma': (30.0, 50.0),
}
EEG_BAND_NAMES = list(EEG_BANDS.keys())

# --- Window / segment parameters (paper Sec. 3.2, EXACT) ---
WINDOW_LEN      = 10.0   # s -- shared by EEG and fNIRS
WINDOW_STEP     = 1.0    # s
TASK_SEG_LEN    = 42.0   # s (2 s instruction + 40 s task)  -> 33 windows
REST_SEG_LEN    = 20.0   # s (immediately following the task segment) -> 11 windows
REST_SEG_OFFSET = TASK_SEG_LEN   # rest starts right where the task segment ends

N_TASK_WINDOWS_EXPECTED = int((TASK_SEG_LEN - WINDOW_LEN) / WINDOW_STEP) + 1   # 33
N_REST_WINDOWS_EXPECTED = int((REST_SEG_LEN - WINDOW_LEN) / WINDOW_STEP) + 1   # 11

# --- Marker codes (Shin 2018) ---
EEG_TRIAL_LABEL_MAP   = {112: 0, 128: 1, 144: 2}   # 0-back, 2-back, 3-back
FNIRS_TRIAL_LABEL_MAP = {7: 0,   8: 1,   9: 2}
REST_LABEL = 3   # 4th class: rest

TRIALS_PER_SESSION = 9

AMPLITUDE_REJECT_THRESH = 120e-6   # V, EEG artifact rejection

# ============================================================
#  LOGGING
# ============================================================
LOG_FILE = os.path.join(OUTPUT_PATH, "processing_log.txt")

def log_message(msg):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
#  EEG LOADING
# ============================================================
def load_eeg_data(eeg_folder):
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(eeg_folder, 'mrk_nback.mat')

    mat_cnt  = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    cnt_key  = [k for k in mat_cnt if not k.startswith('__')][0]
    cnt      = mat_cnt[cnt_key]

    data     = cnt.x.T
    sfreq    = float(cnt.fs)
    ch_names = [str(ch).strip() for ch in cnt.clab]

    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6

    if sfreq != SFREQ_EEG:
        n_samp = int(data.shape[1] * SFREQ_EEG / sfreq)
        data   = resample(data, n_samp, axis=1)
        sfreq  = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw  = mne.io.RawArray(data, info, verbose=False)

    eog_ch = [ch for ch in ch_names if ch.upper() in ('VEOG', 'HEOG')]
    if eog_ch:
        raw.set_channel_types({ch: 'eog' for ch in eog_ch})
    raw.set_montage('standard_1005', match_case=False, on_missing='ignore')

    mat_mrk  = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    mrk_key  = [k for k in mat_mrk if not k.startswith('__')][0]
    mrk      = mat_mrk[mrk_key]

    if hasattr(mrk, 'time'):
        ev_pos = np.round(mrk.time * sfreq / 1000).astype(int)
    elif hasattr(mrk, 'pos'):
        ev_pos = np.array(mrk.pos, dtype=int) - 1
    else:
        raise ValueError("No 'time'/'pos' in EEG marker struct")

    if hasattr(mrk, 'event') and hasattr(mrk.event, 'desc'):
        desc = np.asarray(mrk.event.desc)
        if desc.ndim > 1:
            desc = desc.squeeze()
        ev_codes = desc.flatten().astype(int)
    else:
        ev_codes = ev_pos.copy()

    events       = np.zeros((len(ev_pos), 3), dtype=int)
    events[:, 0] = ev_pos
    events[:, 2] = ev_codes

    del mat_cnt, mat_mrk
    return raw, events, sfreq


# ============================================================
#  fNIRS LOADING (channel-space, no spatial mapping)
# ============================================================
def load_fnirs_data(subject_id):
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    cnt_key = [k for k in mat_cnt if not k.startswith('__')][0]
    cnt     = mat_cnt[cnt_key]

    fnirs_events_time = fnirs_events_code = None
    if os.path.exists(mrk_path):
        mat_fmrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
        fmrk_key = [k for k in mat_fmrk if not k.startswith('__')][0]
        fmrk     = mat_fmrk[fmrk_key]
        if hasattr(fmrk, 'time') and hasattr(fmrk, 'event') and hasattr(fmrk.event, 'desc'):
            fnirs_events_time = np.asarray(fmrk.time).flatten()
            fnirs_events_code = np.asarray(fmrk.event.desc).flatten()
        del mat_fmrk

    sfreq_raw = float(cnt.oxy.fs) if hasattr(cnt, 'oxy') and hasattr(cnt.oxy, 'fs') else 10.0

    # Beer-Lambert already applied by dataset authors -> oxy=HbO, deoxy=HbR directly
    hbo_raw = np.asarray(cnt.oxy.x,   dtype=np.float64)   # (n_samples, 36)
    hbr_raw = np.asarray(cnt.deoxy.x, dtype=np.float64)

    if sfreq_raw != SFREQ_FNIRS:
        n_samp = int(hbo_raw.shape[0] * SFREQ_FNIRS / sfreq_raw)
        hbo_raw = resample(hbo_raw, n_samp, axis=0)
        hbr_raw = resample(hbr_raw, n_samp, axis=0)

    hbt_raw = hbo_raw + hbr_raw   # 3rd hemoglobin channel, per paper "3 x 36 x 100"

    # Stack: (n_samples, 3, 36)  -- [HbO, HbR, HbT]
    fnirs_data = np.stack([hbo_raw, hbr_raw, hbt_raw], axis=1).astype(np.float32)

    del mat_cnt, hbo_raw, hbr_raw, hbt_raw
    gc.collect()

    return fnirs_data, fnirs_events_time, fnirs_events_code


# ============================================================
#  EEG PREPROCESSING (cleaning: notch, bandpass, bad-ch, ICA, CAR)
#  -- unchanged from v2, this part of the pipeline was correct --
# ============================================================
def preprocess_eeg(raw, events):
    raw_before_qc = raw.copy()

    raw_f = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    # Fix: widened from (1.0, 40.0) to (0.5, 50.0) -- the narrower range was
    # truncating the delta band's 0.5-1 Hz portion and the ENTIRE 40-50 Hz
    # portion of the gamma band before band decomposition even started,
    # since gamma is defined as 30-50 Hz per paper Sec 2.1 (not 30-40 Hz).
    raw_f.filter(l_freq=0.5, h_freq=50.0, method='fir', verbose=False)

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
        # TP9=reference electrode, TP10=ground electrode (paper Sec 3.1.2:
        # "additional electrodes placed at TP9 as the reference electrode
        # and at TP10 as the ground electrode") -- neither is a real signal
        # channel, both must be excluded from bad-channel stats and dropped
        # later, or the final channel count is 29 instead of the paper's 28.
        if name.upper() in ('VEOG', 'HEOG', 'TP9', 'TP10'):
            continue
        if (stds[i] > med_std + 4.5 * mad or
                stds[i] < med_std - 4.5 * mad or
                mean_corrs[i] < 0.30):
            bad_ch.append(name)

    if bad_ch:
        raw_f.info['bads'] = bad_ch
    if raw_f.info['bads']:
        raw_f.interpolate_bads(reset_bads=True, verbose=False)

    raw_f.set_eeg_reference('average', projection=False, verbose=False)

    picks_ica = mne.pick_types(raw_f.info, eeg=True, eog=False, exclude='bads')
    ica = mne.preprocessing.ICA(
        n_components=0.99, method='picard',
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False)
    ica.fit(raw_f, picks=picks_ica)

    ic_labels = mne_icalabel.label_components(raw_f, ica, method='iclabel')
    ARTIFACT_CLASSES = {'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
                        'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6}
    ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
    PROB_THRESH = 0.30
    BRAIN_CEIL  = 0.50

    proba = ica.labels_scores_
    idx_argmax = {i for i, lbl in enumerate(ic_labels['labels']) if lbl in ARTIFACT_TARGETS}
    idx_prob = set()
    for ci in range(proba.shape[0]):
        for cls in ARTIFACT_TARGETS:
            col = ARTIFACT_CLASSES.get(cls)
            if col is not None and proba[ci, col] >= PROB_THRESH:
                idx_prob.add(ci)
                break

    idx_veog, idx_heog = [], []
    try:
        if 'VEOG' in raw.ch_names:
            idx_veog, _ = ica.find_bads_eog(raw, ch_name='VEOG', threshold=2.0, verbose=False)
        if 'HEOG' in raw.ch_names:
            idx_heog, _ = ica.find_bads_eog(raw, ch_name='HEOG', threshold=2.0, verbose=False)
    except Exception:
        pass

    all_flagged = idx_argmax | idx_prob
    eog_only    = (set(idx_veog) | set(idx_heog)) - all_flagged
    rescued     = {i for i in eog_only if proba[i, ARTIFACT_CLASSES['brain']] >= BRAIN_CEIL}
    artifact_idx = sorted((idx_argmax | idx_prob | set(idx_veog) | set(idx_heog)) - rescued)

    ica.exclude = artifact_idx
    if ica.exclude:
        ica.apply(raw_f, verbose=False)

    del ica, ic_labels, proba
    gc.collect()

    # Fix: drop TP10 (ground) as well as TP9 (reference) -- neither is a
    # real EEG signal channel. Dropping only TP9 leaves 29 channels instead
    # of the paper's 28 (Sec 3.1.2 channel list has exactly 28 real
    # electrodes; TP9=reference, TP10=ground are separate from those 28).
    drop_ch = [ch for ch in raw_f.ch_names if ch.upper() in ('VEOG', 'HEOG', 'TP9', 'TP10')]
    if drop_ch:
        raw_f.drop_channels(drop_ch)

    _save_eeg_qc_plots(raw_before_qc, raw_f)
    del raw_before_qc
    return raw_f


def _plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
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
    try:
        tmp = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc")
        os.makedirs(tmp, exist_ok=True)
        figs = [
            (_plot_all_channels_stacked(raw_before, "EEG RAW - All Channels (before)", 'tab:red'),
             "last_eeg_before_allchannels.png"),
            (_plot_all_channels_stacked(raw_after,  "EEG PREPROCESSED - All Channels (after)", 'tab:blue'),
             "last_eeg_after_allchannels.png"),
        ]
        for fig, fname in figs:
            fig.savefig(os.path.join(tmp, fname), dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"Could not save EEG QC plots: {e}")


def _rename_eeg_qc_plots(subject_id, sub_out):
    try:
        tmp = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc")
        pairs = [
            ("last_eeg_before_allchannels.png", f"{subject_id}_eeg_before_allchannels.png"),
            ("last_eeg_after_allchannels.png",  f"{subject_id}_eeg_after_allchannels.png"),
        ]
        for src_name, dst_name in pairs:
            src = os.path.join(tmp, src_name)
            dst = os.path.join(sub_out, dst_name)
            if os.path.exists(src):
                os.replace(src, dst)
    except Exception as e:
        print(f"Could not finalize EEG QC plot for {subject_id}: {e}")


# ============================================================
#  EEG 5-BAND FILTERING (channel space, no spatial mapping)
# ============================================================
def bandpass_filter_eeg_bands(raw_clean):
    """Generator: yields (band_name, (n_ch, n_times) float64) one at a time."""
    for band_name, (lo, hi) in EEG_BANDS.items():
        raw_band = raw_clean.copy().filter(
            l_freq=lo, h_freq=hi, method='fir', phase='zero', verbose=False)
        band_array = raw_band.get_data()
        del raw_band
        yield band_name, band_array


# ============================================================
#  WINDOW EXTRACTION -- task (42s, 33 win) + rest (20s, 11 win)
# ============================================================
def _zscore(arr, axis=-1, eps=1e-8):
    """z-score normalize along the given axis (typically time)."""
    mu  = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    return (arr - mu) / (std + eps)


def extract_windows_for_subject(eeg_band_data, raw_eeg_clean,
                                fnirs_data, events, sfreq_eeg,
                                fnirs_events_time, fnirs_events_code):
    """
    eeg_band_data : ndarray (5, n_ch, n_times) -- 5-band filtered EEG, channel space
    fnirs_data    : ndarray (n_fnirs_times, 3, 36) -- HbO/HbR/HbT, channel space

    Implements the paper's EXACT segment-level preprocessing (Sec 3.1.2/3.2)
    for N-back, applied BEFORE sliding the 10s window (this step was
    completely missing in the earlier v3/v4 rewrite when spatial mapping
    was removed):

      TASK segment (42s, labels 0/1/2):
        - fNIRS: baseline-corrected using the mean of the -5s to 0s window
          BEFORE the segment onset (same convention as HBCI dataset,
          reused per Sec 3.1.2: "other preprocessing operations are the
          same as ... HBCI dataset"), then z-score normalized.
        - EEG (each of the 5 bands independently): z-score normalized
          over the 42s segment.

      REST segment (20s "effective" window count -> n=11, immediately
      after the task segment, labels 3):
        - fNIRS: baseline-corrected using the mean of the 35s-40s mark of
          the SAME trial's 42s task segment (per paper: "we performed
          baseline correction from 35 s to 40 s on the 21 s rest data
          segment" -- 35-40s falls inside the preceding task period, i.e.
          the baseline is taken from the last 5s of the task, not from
          before the rest segment itself), then z-score normalized.
        - EEG: z-score normalized over the 20s rest segment.

    Returns eeg_windows (N,5,n_ch,2000), fnirs_windows (N,3,36,100),
            labels (N,), session_ids (N,), trial_ids (N,)
    """
    label_map = EEG_TRIAL_LABEL_MAP
    valid_codes = list(label_map.keys())
    target_ev = events[np.isin(events[:, 2], valid_codes)]

    if len(target_ev) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    fnirs_sync_ok = False
    fnirs_trial_times = None
    if fnirs_events_time is not None and fnirs_events_code is not None:
        f_valid = np.isin(fnirs_events_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
        f_times = np.asarray(fnirs_events_time)[f_valid].astype(float)
        f_codes = np.asarray(fnirs_events_code)[f_valid]
        f_order = np.argsort(f_times)
        f_times = f_times[f_order] / 1000.0
        f_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in f_codes[f_order]])
        eeg_classes = np.array([label_map.get(int(ev[2]), -1) for ev in target_ev])
        n_common = min(len(f_classes), len(eeg_classes))
        if (len(f_times) == len(target_ev) and
                np.array_equal(f_classes[:n_common], eeg_classes[:n_common])):
            fnirs_sync_ok = True
            fnirs_trial_times = f_times

    win_samp_eeg   = int(WINDOW_LEN * sfreq_eeg)
    win_samp_fnirs = int(WINDOW_LEN * SFREQ_FNIRS)

    eeg_windows, fnirs_windows, labels, session_ids, trial_ids = [], [], [], [], []

    def _slide_windows(eeg_seg, fnirs_seg, eeg_t0_samp, seg_len_s, label,
                       session_id, trial_idx):
        """
        eeg_seg   : (5, n_ch, seg_samples_eeg)   -- already z-scored
        fnirs_seg : (seg_samples_fnirs, 3, 36)    -- already baseline+z-scored
        eeg_t0_samp : absolute sample index in raw_eeg_clean where eeg_seg
                      starts (needed for the raw-signal amplitude-reject check)
        """
        max_offset = seg_len_s - WINDOW_LEN
        offset = 0.0
        while offset <= max_offset + 1e-6:
            w0_eeg_local = int(round(offset * sfreq_eeg))
            w1_eeg_local = w0_eeg_local + win_samp_eeg
            if w1_eeg_local > eeg_seg.shape[-1]:
                offset += WINDOW_STEP
                continue

            # Amplitude-based artifact rejection on the ORIGINAL (non
            # band-filtered, non z-scored) cleaned signal, same as before
            w0_abs = eeg_t0_samp + w0_eeg_local
            w1_abs = eeg_t0_samp + w1_eeg_local
            raw_slice = raw_eeg_clean.get_data(start=w0_abs, stop=w1_abs)
            reject = np.max(np.abs(raw_slice)) > AMPLITUDE_REJECT_THRESH
            del raw_slice
            if reject:
                offset += WINDOW_STEP
                continue

            eeg_slice = eeg_seg[:, :, w0_eeg_local:w1_eeg_local].astype(np.float16)

            w0_f_local = int(round(offset * SFREQ_FNIRS))
            w1_f_local = w0_f_local + win_samp_fnirs
            if w1_f_local > fnirs_seg.shape[0]:
                offset += WINDOW_STEP
                continue
            fnirs_slice = fnirs_seg[w0_f_local:w1_f_local].transpose(1, 2, 0).astype(np.float32)

            eeg_windows.append(eeg_slice)
            fnirs_windows.append(fnirs_slice)
            labels.append(label)
            session_ids.append(session_id)
            trial_ids.append(trial_idx)
            offset += WINDOW_STEP

    for trial_idx, ev in enumerate(target_ev):
        ev_samp_eeg = ev[0]
        raw_code    = ev[2]
        label       = label_map.get(raw_code, -1)
        if label == -1:
            continue

        session_id = trial_idx // TRIALS_PER_SESSION + 1
        eeg_t   = ev_samp_eeg / sfreq_eeg
        fnirs_t = fnirs_trial_times[trial_idx] if fnirs_sync_ok else eeg_t

        # ============================================================
        # TASK segment (42s) -- labels 0/1/2
        # ============================================================
        eeg_t0_samp = int(round(eeg_t * sfreq_eeg))
        eeg_t1_samp = eeg_t0_samp + int(round(TASK_SEG_LEN * sfreq_eeg))
        f_t0_samp   = int(round(fnirs_t * SFREQ_FNIRS))
        f_t1_samp   = f_t0_samp + int(round(TASK_SEG_LEN * SFREQ_FNIRS))

        if eeg_t1_samp > eeg_band_data.shape[-1] or f_t1_samp > fnirs_data.shape[0] or f_t0_samp < 0:
            continue   # trial runs past end of recording, skip entirely

        eeg_task_seg = eeg_band_data[:, :, eeg_t0_samp:eeg_t1_samp]   # (5, n_ch, 8400)
        eeg_task_seg = _zscore(eeg_task_seg, axis=-1)                  # z-score per band, per channel

        fnirs_task_raw = fnirs_data[f_t0_samp:f_t1_samp]               # (420, 3, 36)

        # fNIRS baseline: mean of [-5s, 0s] BEFORE the task segment onset
        bs0 = f_t0_samp - int(round(5.0 * SFREQ_FNIRS))
        bs1 = f_t0_samp
        if bs0 >= 0:
            task_baseline = fnirs_data[bs0:bs1].mean(axis=0, keepdims=True)   # (1,3,36)
        else:
            task_baseline = 0.0   # first trial in recording: no pre-trial data available
        fnirs_task_seg = fnirs_task_raw - task_baseline
        fnirs_task_seg = _zscore(fnirs_task_seg, axis=0)                # z-score over time

        _slide_windows(eeg_task_seg, fnirs_task_seg, eeg_t0_samp,
                       TASK_SEG_LEN, label, session_id, trial_idx)

        del eeg_task_seg, fnirs_task_raw, fnirs_task_seg

        # ============================================================
        # REST segment (effective 20s -> n=11 windows) -- label REST_LABEL
        # Baseline: mean of the 35s-40s mark of THIS SAME trial's task
        # segment (paper Sec 3.1.2), i.e. the last 5s of the task period,
        # NOT relative to the rest segment's own onset.
        # ============================================================
        rest_eeg_t0_samp = eeg_t1_samp   # right after the 42s task segment
        rest_eeg_t1_samp = rest_eeg_t0_samp + int(round(REST_SEG_LEN * sfreq_eeg))
        rest_f_t0_samp   = f_t1_samp
        rest_f_t1_samp   = rest_f_t0_samp + int(round(REST_SEG_LEN * SFREQ_FNIRS))

        if rest_eeg_t1_samp > eeg_band_data.shape[-1] or rest_f_t1_samp > fnirs_data.shape[0]:
            continue   # not enough data for the rest segment, skip

        eeg_rest_seg = eeg_band_data[:, :, rest_eeg_t0_samp:rest_eeg_t1_samp]
        eeg_rest_seg = _zscore(eeg_rest_seg, axis=-1)

        fnirs_rest_raw = fnirs_data[rest_f_t0_samp:rest_f_t1_samp]

        # Rest baseline = mean of fNIRS data from t=35s to t=40s of the
        # SAME trial (absolute segment coordinates -- inside the task
        # period, 5s before it ends), not before the rest segment itself.
        rb0 = f_t0_samp + int(round(35.0 * SFREQ_FNIRS))
        rb1 = f_t0_samp + int(round(40.0 * SFREQ_FNIRS))
        if rb1 <= fnirs_data.shape[0]:
            rest_baseline = fnirs_data[rb0:rb1].mean(axis=0, keepdims=True)
        else:
            rest_baseline = 0.0
        fnirs_rest_seg = fnirs_rest_raw - rest_baseline
        fnirs_rest_seg = _zscore(fnirs_rest_seg, axis=0)

        _slide_windows(eeg_rest_seg, fnirs_rest_seg, rest_eeg_t0_samp,
                       REST_SEG_LEN, REST_LABEL, session_id, trial_idx)

        del eeg_rest_seg, fnirs_rest_raw, fnirs_rest_seg

    eeg_windows   = np.array(eeg_windows,   dtype=np.float16) if eeg_windows else np.array([])
    fnirs_windows = np.array(fnirs_windows, dtype=np.float32) if fnirs_windows else np.array([])
    labels        = np.array(labels,        dtype=np.int64)
    session_ids   = np.array(session_ids,   dtype=np.int32)
    trial_ids     = np.array(trial_ids,     dtype=np.int32)

    return eeg_windows, fnirs_windows, labels, session_ids, trial_ids


# ============================================================
#  MAIN SUBJECT PROCESSING
# ============================================================
def process_subject(subject_id):
    t0 = time.time()
    log_message(f"START processing {subject_id}")

    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    if not os.path.exists(eeg_folder):
        log_message(f"SKIPPED {subject_id}: EEG folder not found")
        return

    raw_eeg = events = None
    fnirs_data = fnirs_events_time = fnirs_events_code = None
    raw_eeg_clean = None
    eeg_band_stack = None
    eeg_windows = fnirs_windows = labels = session_ids = trial_ids = None

    try:
        print(f"\n[{subject_id}] Loading EEG...")
        raw_eeg, events, sfreq_eeg = load_eeg_data(eeg_folder)

        print(f"[{subject_id}] Loading fNIRS (channel-space, HbO/HbR/HbT)...")
        fnirs_data, fnirs_events_time, fnirs_events_code = load_fnirs_data(subject_id)

        print(f"[{subject_id}] Preprocessing EEG (notch/bandpass/ICA/CAR)...")
        raw_eeg_clean = preprocess_eeg(raw_eeg, events)

        sub_out = os.path.join(OUTPUT_PATH, subject_id)
        os.makedirs(sub_out, exist_ok=True)
        raw_eeg_clean.save(
            os.path.join(sub_out, f"{subject_id}_eeg_clean_raw.fif"), overwrite=True)
        _rename_eeg_qc_plots(subject_id, sub_out)

        print(f"[{subject_id}] Filtering into 5 frequency bands "
              "(delta/theta/alpha/beta/gamma), channel space...")
        band_list = []
        for band_name, band_arr in bandpass_filter_eeg_bands(raw_eeg_clean):
            band_list.append(band_arr.astype(np.float32))
            del band_arr
            gc.collect()
        eeg_band_stack = np.stack(band_list, axis=0)   # (5, n_ch, n_times)
        del band_list
        gc.collect()
        print(f"[{subject_id}] EEG band stack: {eeg_band_stack.shape}")

        print(f"[{subject_id}] Extracting task (42s->33win) + rest (20s->11win) windows...")
        eeg_windows, fnirs_windows, labels, session_ids, trial_ids = \
            extract_windows_for_subject(
                eeg_band_stack, raw_eeg_clean,
                fnirs_data, events, sfreq_eeg,
                fnirs_events_time, fnirs_events_code)

        if len(labels) == 0:
            log_message(f"WARNING {subject_id}: no windows extracted")
            return

        print(f"[{subject_id}] Extracted {len(labels)} windows. "
              f"Class distribution: {np.bincount(labels, minlength=4)}")
        print(f"[{subject_id}] Expected 1188 total (99 x 4 classes x 3 sessions) "
              f"before artifact-based dropout.")

        del eeg_band_stack, fnirs_data, raw_eeg_clean, raw_eeg
        eeg_band_stack = fnirs_data = raw_eeg_clean = raw_eeg = None
        gc.collect()

        np.save(os.path.join(sub_out, f"{subject_id}_eeg_windows.npy"),   eeg_windows)
        np.save(os.path.join(sub_out, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)
        np.save(os.path.join(sub_out, f"{subject_id}_labels.npy"),        labels)
        np.save(os.path.join(sub_out, f"{subject_id}_session_ids.npy"),   session_ids)
        np.save(os.path.join(sub_out, f"{subject_id}_trial_ids.npy"),     trial_ids)

        metadata = {
            'subject_id':          subject_id,
            'eeg_shape':           list(eeg_windows.shape),
            'fnirs_shape':         list(fnirs_windows.shape),
            'eeg_bands':           EEG_BAND_NAMES,
            'fnirs_hemo_channels': ['HbO', 'HbR', 'HbT'],
            'n_classes':           4,
            'class_labels':        {'0': '0-back', '1': '2-back', '2': '3-back', '3': 'rest'},
            'labels_distribution': np.bincount(labels, minlength=4).tolist(),
            'expected_total':      1188,
            'window_len_s':        WINDOW_LEN,
            'window_step_s':       WINDOW_STEP,
            'task_segment_s':      TASK_SEG_LEN,
            'rest_segment_s':      REST_SEG_LEN,
            'sfreq_eeg':           SFREQ_EEG,
            'sfreq_fnirs':         SFREQ_FNIRS,
            'spatial_mapping_in_preprocessing': False,
            'note': 'Matches paper Sec 3.2 exactly: 4-class N-back, '
                    'channel-space samples (no 16x16 grid stored), '
                    'fNIRS=HbO/HbR/HbT single 10s window (not 21-segment stack).',
        }
        with open(os.path.join(sub_out, f"{subject_id}_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)

        elapsed = (time.time() - t0) / 60
        log_message(
            f"DONE {subject_id} in {elapsed:.1f} min | "
            f"EEG: {eeg_windows.shape} | fNIRS: {fnirs_windows.shape} | "
            f"Labels: {np.bincount(labels, minlength=4)}")

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        log_message(f"ERROR processing {subject_id} after {elapsed:.1f} min: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        del raw_eeg
        del raw_eeg_clean
        del eeg_band_stack
        del fnirs_data
        del eeg_windows
        del fnirs_windows
        del labels
        del session_ids
        del trial_ids
        del events
        del fnirs_events_time
        del fnirs_events_code
        gc.collect()


# ============================================================
#  PARALLEL ENTRY POINT
#  Memory footprint is MUCH smaller now (channel-space samples,
#  no 16x16 spatial grid, no 21-segment fNIRS stacking):
#    EEG windows ~1188 x 5x28x2000 x2bytes(f16)  ~= 0.66 GB
#    fNIRS windows ~1188 x 3x36x100 x4bytes(f32) ~= 0.05 GB
#    5-band continuous EEG stack (before windowing) ~= 0.6 GB
#  Peak per subject ~= 1.5-2 GB -> back up to 6 workers is safe.
# ============================================================
N_WORKERS = 6


def _worker_wrapper(subject_id: str) -> str:
    import mne
    mne.set_log_level('WARNING')
    t0 = time.time()
    try:
        process_subject(subject_id)
        elapsed = (time.time() - t0) / 60
        return f"OK  {subject_id}  {elapsed:.1f} min"
    except Exception as e:
        elapsed = (time.time() - t0) / 60
        return f"ERR {subject_id}  {elapsed:.1f} min - {e}"


def _progress_callback(result: str):
    tag = "[OK ]" if result.startswith("OK") else "[ERR]"
    ts  = datetime.datetime.now().strftime("%H:%M:%S")
    msg = f"[{ts}] {tag} {result}"
    print(msg.encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding), flush=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_parallel(subjects: list, n_workers: int = N_WORKERS):
    import multiprocessing as mp
    total_t0 = time.time()
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    log_message("=" * 55)
    log_message(f"PARALLEL PIPELINE v3 START - {len(subjects)} subjects, {n_workers} workers")
    log_message("Protocol: 4-class N-back (0/2/3-back + rest), channel-space samples")
    log_message("=" * 55)

    print(f"\n{'='*55}")
    print(f"  MSTFDN Preprocessing v3 (paper-exact protocol)")
    print(f"  Subjects : {len(subjects)}")
    print(f"  Workers  : {n_workers}")
    print(f"  Expected : ~1188 samples/subject, 4 classes")
    print(f"{'='*55}\n")

    ctx  = mp.get_context('spawn')
    done = 0
    with ctx.Pool(processes=n_workers, maxtasksperchild=1) as pool:
        for result in pool.imap_unordered(_worker_wrapper, subjects, chunksize=1):
            done += 1
            _progress_callback(result)
            print(f"  Progress: {done}/{len(subjects)} subjects done", flush=True)

    total_elapsed = (time.time() - total_t0) / 60
    log_message(f"PARALLEL PIPELINE FINISHED in {total_elapsed:.1f} min")
    print(f"\n[OK] All {len(subjects)} subjects finished in {total_elapsed:.1f} min")


if __name__ == "__main__":
    ALL_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]
    run_parallel(ALL_SUBJECTS, n_workers=N_WORKERS)
