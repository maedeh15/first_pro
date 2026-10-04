#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Preprocessing v2 run a1b 10s


# In[ ]:


#version2 - session_id
# ============================================================================
# RUN A1b (isolated experiment, derived from preprocessing_v2_run_a1_10s.py)
# THE ONLY SCIENTIFIC VARIABLE CHANGED vs Run A1: FNIRS_NUM_SEGMENTS 11 -> 21.
# Rationale: with a 10s EEG window, 11 segments (3s each, 1s step, anchored at t=0)
# only span t=0..13s relative to the window start. The fNIRS hemodynamic response
# peaks ~5-10s after stimulus onset, so a 10s task window needs coverage out to
# ~t=23s to contain the full response. 21 segments span t=0..23s exactly.
# This preserves the TSFNet philosophy: the model still learns the effective
# EEG-fNIRS offset from the data (CAFE attends over the segment sequence); it is
# only given a segment span wide enough to contain the response in the first place.
# The only other change is OUTPUT_PATH (new, separate folder) so Run A1's validated
# data is never overwritten. NOTHING else in this file differs from Run A1.
# ============================================================================
# ============================================================================
# RUN A1-REVISED (isolated experiment): EEG_WINDOW_LEN 3.0 -> 10.0 seconds, EVERYTHING ELSE
# UNCHANGED from the validated preprocessing_v2_fixed.py (same EEG_STEP=1.0, same
# EEG_EPOCH_TMAX=40.0, same fNIRS windowing (FNIRS_WINDOW_LEN=3.0, 11 segments, 1s step) --
# only ONE variable changed, per the project's single-variable-isolation discipline.
# Rationale: Ding et al. (MSTFDN, 2026), a very similar 3-branch fusion architecture evaluated
# on this SAME dataset, ran a systematic sweep of window lengths (8s/9s/10s/11s) and found 10s
# optimal. This is the single strongest, most directly transferable empirical precedent found
# in the literature review for this dataset.
# OUTPUT_PATH below points to a NEW, SEPARATE folder (not the validated 3s-window
# MULTIMODAL_PREPROCESSED folder) so this experiment never overwrites or corrupts the existing
# baseline data -- both can coexist and be compared side by side.
# ============================================================================
# MULTIMODAL PREPROCESSING: EEG + fNIRS 
# Set the paths based on your system before running the code
# you can adjust which subjects to process at the bottom of this cell

import os
import gc  # NEW (memory leak fix): needed for explicit cleanup in compute_ml_anomaly_intervals (D2)
import psutil  # NEW (parallel execution): needed for memory-aware worker count calculation
from concurrent.futures import ProcessPoolExecutor, as_completed  # NEW (parallel execution)
import numpy as np
import scipy.io as sio
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, resample
from scipy.signal import welch          # NEW (D2): needed for ML-anomaly-filter band-power features
from scipy.stats import kurtosis, skew  # NEW (D2): needed for ML-anomaly-filter feature extraction
from scipy.interpolate import griddata
from scipy.interpolate import CloughTocher2DInterpolator  # NEW: used by the vectorized spatial_mapping()
from scipy.spatial import Delaunay  # NEW: used by the vectorized spatial_mapping()
import mne
# NEW (D2): sklearn models for the epoch-level ML-anomaly filter -- same ensemble
# (Isolation Forest + LOF + Elliptic Envelope, majority vote) used and validated
# in eeg_artifact_detection_26subj_v3_english.py / epoch_filter_calibration_test.py
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
import mne_icalabel
import pywt
import scipy.stats as stats
import pandas as pd
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt  # NEW: needed for before/after preprocessing plots
import time
import datetime
import json  # NEW: for per-subject metadata.json export (adopted from reviewing a third-party implementation, see chat history)
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
# BASE_DATASET_PATH = r"D:\Research\EEG-fNIRS\project\project\Dataset"
# OUTPUT_PATH = r"D:\Research\EEG-fNIRS\project\project\Results\MULTIMODAL_PREPROCESSED"
# FIX (path update per user's actual folder layout): the user's dataset is laid out as
# DATA\VP001-EEG, DATA\VP001-NIRS, ... (i.e. WITHOUT an intermediate numeric subfolder like
# DATA\001\VP001-EEG that the original code assumed). Paths below updated accordingly; see the
# matching fix in load_and_preprocess_fnirs() and process_subject() further down where
# subject_path/subject_num are removed for the same reason.
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"  # NEW: matches user's actual dataset folder
OUTPUT_PATH = r"F:\preprocessing-history-thesis\D1"  # CHANGED (Run A1b): new separate folder, does not touch Run A1's validated 11-segment data

# window parameters
EEG_WINDOW_LEN = 10.0     # CHANGED (Run A1-revised): 3.0 -> 10.0, per MSTFDN's validated optimum on this dataset. THE ONLY intentional change in this file vs preprocessing_v2_fixed.py.
EEG_STEP = 1.0            # 66% overlap

# NEW (D1 -- isolated data-level fix): VP004 shows a chronic, session-independent
# EEG artifact pattern (Z-score-flagged channel count 21-26/28, artifact score
# 80-83/100, in ALL 3 sessions -- see eeg_artifact_detection_26subj_v3_english.py
# summary). This is a structural/recording-quality issue for this subject, not a
# transient artifact, so the subject is excluded entirely rather than filtered
# at the epoch level.
REMOVE_SUBJECTS = {"VP004"}  # NEW (D1)

# NEW (isolation toggle): the combined D1+D2 run showed a large Session-3 drop
# (59.00% -> 53.81%) versus the D1/D2-free baseline. Per the project's
# one-variable-at-a-time methodology, D1 and D2 must be tested in isolation
# to determine which change is responsible before drawing any conclusion.
# This run: ENABLE_D2_EPOCH_FILTER = False -- D1 (VP004 removal) stays active,
# D2 (epoch-level ML-anomaly filter) is fully disabled, so its window-skip
# check below never triggers and every window that passes the existing
# amplitude QC check is kept, exactly as in the pre-D1/D2 baseline.
ENABLE_D2_EPOCH_FILTER = False  # NEW (isolation toggle) -- set True to re-enable D2

# NEW (D2 -- isolated data-level fix): epoch-level ML-anomaly filter. A sliding
# EEG window is dropped if the fraction of its duration overlapping a detected
# ML-anomaly interval is >= this threshold. Calibrated on VP001/VP006/VP008
# (temporal_emg artifact pattern subjects) via epoch_filter_calibration_test.py:
# thresholds 0.10-0.20 removed 40-50% of windows (too aggressive, likely
# discarding usable data); 0.30 removed ~20-24% (the smallest threshold at which
# every tested subject stayed below the 30% "too much data lost" guideline from
# the improvement checklist); 0.40-0.50 removed too little to reliably exclude
# the known burst artifacts visible in the raw EEG plots. 0.30 was chosen as the
# largest (most conservative, least data-discarding) threshold that still keeps
# every calibration subject at or under the 30% ceiling.
# NOTE: only takes effect when ENABLE_D2_EPOCH_FILTER = True (see toggle above).
EPOCH_FILTER_OVERLAP_THRESHOLD = 0.30  # NEW (D2)
EPOCH_FILTER_ANOMALY_EPOCH_LEN = 2.0   # NEW (D2): matches epoch_len used by the ML feature extractor below
EPOCH_FILTER_CONTAMINATION     = 0.10  # NEW (D2): same contamination fraction as the calibration script

EEG_EPOCH_TMIN = 0.0
# EEG_EPOCH_TMAX = 8.0
# FIX (Problem 2): the n-back task period is 40 s long (2 s instruction + 40 s task + 20 s rest,
# Shin et al. 2018). The previous value of 8 s only let the sliding window traverse the first
# 8 s of each trial, discarding ~80% of the task period and, more importantly, was being applied
# on top of STIMULUS-level markers (every ~2 s) rather than TRIAL-level markers, which produced
# massive window overlap across different stimuli instead of covering the full cognitive-load trial.
EEG_EPOCH_TMAX = 40.0      # NEW: cover the full 40 s task period of a single n-back trial

# FNIRS_WINDOW_LEN = 10.0
# CHANGED (EXPERIMENTAL -- 11-segment fNIRS windowing, per chat history): matches the TSFNet
# paper's actual approach more closely than the single continuous 10s window used previously --
# instead of one 10s fNIRS window per EEG window, 11 overlapping 3s segments (stepped by
# FNIRS_STEP=1s each) are extracted and stacked, giving the model an explicit per-segment temporal
# axis (matching Table 1's fNIRS branch input) rather than one pre-averaged continuous block.
FNIRS_WINDOW_LEN = 3.0    # CHANGED: 10.0 -> 3.0 (length of EACH of the 11 segments, not the whole span)
FNIRS_NUM_SEGMENTS = 21   # CHANGED (Run A1b): 11 -> 21. THE ONLY scientific variable changed vs Run A1. Segment k covers [k, k+3]s relative to the EEG window start, so 21 segments span t=0..23s (vs t=0..13s with 11), fully covering the 5-10s hemodynamic delay of a 10s task window.
FNIRS_STEP = 1.0          # 90% overlap

# NEW (isolated baseline-correction test, per chat history): the paper's Sec. 2.2 also specifies
# "baseline correction was performed by subtracting the average signal within the -5 to -2 s
# interval before each trial" -- a step that was previously only ever tested BUNDLED together
# with the paper's bandpass filter (Run C/E, both rejected), so its effect was never isolated.
# Testing it alone here, with the current (already-validated) 0.2 Hz lowpass filter kept UNCHANGED.
BASELINE_TMIN = -5.0  # NEW: seconds before trial onset (start of baseline window)
BASELINE_TMAX = -2.0  # NEW: seconds before trial onset (end of baseline window)
FNIRS_EPOCH_TMIN = 0.0
FNIRS_EPOCH_TMAX = 18.0   # bacause of the hemodynamic delay
# NOTE: FNIRS_EPOCH_TMAX is no longer used directly to bound the slide range (the EEG trial length
# now governs how many windows are produced, see extract_synchronized_windows below), but it is left
# untouched here exactly as before so no other part of the code/structure changes.

SFREQ_EEG = 200.0
SFREQ_FNIRS = 10.0      

# NEW (Problem 1 fix support): explicit trial-level marker maps based on the official dataset
# documentation (Shin et al. 2018, Dataset_description_MATLAB.pdf):
#   EEG  mrk_nback.event.desc : 112 = 0-back session, 128 = 2-back session, 144 = 3-back session
#   fNIRS mrk_nback.event.desc:   7 = 0-back session,   8 = 2-back session,   9 = 3-back session
# These TRIAL-onset codes (one per ~40 s task period) replace the old STIMULUS-level codes
# (16/48/64/80/96 for EEG; 1/2/3/4/5 placeholders) that fired every ~2 s within a trial.
EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}      # NEW: EEG session/trial-onset codes -> class
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}          # NEW: fNIRS session/trial-onset codes -> class


# LOAD EEG DATA
def load_eeg_data(eeg_folder):
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(eeg_folder, 'mrk_nback.mat')

    print(f"Loading EEG from: {cnt_path}")

    # Load signal
    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_cnt_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt_key = real_cnt_keys[0]
    cnt_struct = mat_cnt[cnt_key]

    data = cnt_struct.x.T
    sfreq = float(cnt_struct.fs)
    ch_names = [str(ch).strip() for ch in cnt_struct.clab]

    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6

    # making sure the sampling rate of 200
    if sfreq != SFREQ_EEG:
        print(f"Resampling from {sfreq} to {SFREQ_EEG} Hz")
        from scipy.signal import resample
        n_samples = int(data.shape[1] * SFREQ_EEG / sfreq)
        data = resample(data, n_samples, axis=1)
        sfreq = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    # EOG channels
    eog_channels = [ch for ch in ['VEOG', 'HEOG'] if ch in ch_names]
    if eog_channels:
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})

    # NOTE: this generic standard_1005 template montage is kept ONLY for MNE-internal purposes that
    # need SOME position estimate for every channel (e.g. interpolate_bads()'s spherical-spline
    # interpolation in preprocess_eeg()). It is NOT used for the spatial_mapping() grid anymore --
    # get_eeg_mnt() (further below) uses the REAL, dataset-provided positions from mnt_nback.mat
    # (loaded at the end of this function) for that purpose instead.
    raw.set_montage('standard_1005', match_case=False, on_missing='ignore')

    # Load markers 
    mat_mrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    real_mrk_keys = [k for k in mat_mrk.keys() if not k.startswith('__')]
    mrk_key = real_mrk_keys[0]
    mrk_struct = mat_mrk[mrk_key]

    if hasattr(mrk_struct, 'time'):
        event_positions = np.round(mrk_struct.time * sfreq / 1000).astype(int)
    elif hasattr(mrk_struct, 'pos'):
        event_positions = np.array(mrk_struct.pos, dtype=int) - 1
    else:
        raise ValueError("Could not find 'time' or 'pos' in marker struct")

    # Event extraction
    if hasattr(mrk_struct, 'event') and hasattr(mrk_struct.event, 'desc'):
        desc = mrk_struct.event.desc
        if desc.ndim > 1:
            desc = desc.squeeze()
        event_codes = np.asarray(desc, dtype=int).flatten()
    elif hasattr(mrk_struct, 'y'):
        y = mrk_struct.y
        if y.ndim > 1:
            if y.shape[0] == 1 or y.shape[1] == 1:
                event_codes = y.flatten()
            else:
                event_codes = np.argmax(y, axis=0) + 1
        else:
            event_codes = y.flatten()
    else:
        event_codes = event_positions.copy()

    events = np.zeros((len(event_positions), 3), dtype=int)
    events[:, 0] = event_positions
    events[:, 2] = event_codes

    print(f"EEG loaded: {raw.n_times} samples, {sfreq} Hz, {len(events)} markers")
    print(f"unique markers: {np.unique(event_codes)}")

    # NEW: load the EEG mnt_nback.mat file -- REAL electrode 3D positions from this dataset,
    # confirmed via user-provided inspection script output: same struct format as the fNIRS
    # mnt_nback.mat already used in load_and_preprocess_fnirs (fields ['x','y','pos_3d','clab'],
    # pos_3d shape (3, 30), clab shape (30,)). This CORRECTS an earlier (wrong) assumption that EEG
    # had no equivalent .mat position file and had to fall back on MNE's generic standard_1005
    # template positions (raw.set_montage() below, which is KEPT unchanged -- it is still needed
    # for interpolate_bads()'s spherical-spline interpolation, an unrelated MNE-internal use). For
    # the azimuthal-equidistant spatial_mapping() grid, get_eeg_mnt() (defined further below) now
    # uses THIS real, dataset-provided mnt_nback.mat instead, exactly mirroring how fNIRS's own
    # mnt_nback.mat is already used for its spatial grid.
    mnt_path = os.path.join(eeg_folder, 'mnt_nback.mat')  # NEW
    mat_eeg_mnt = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)  # NEW
    real_eeg_mnt_keys = [k for k in mat_eeg_mnt.keys() if not k.startswith('__')]  # NEW
    eeg_mnt_struct = mat_eeg_mnt[real_eeg_mnt_keys[0]]  # NEW: has .pos_3d (3,30) and .clab (30,)
    print(f"EEG mnt_nback.mat loaded: pos_3d shape {np.asarray(eeg_mnt_struct.pos_3d).shape}")  # NEW

    # return raw, events, sfreq
    return raw, events, sfreq, eeg_mnt_struct  # CHANGED: also return the real electrode-position struct


# LOAD AND PREPROCESS FNIRS DATA
def load_and_preprocess_fnirs(subject_id):
    # subject_num = subject_id[2:]
    # subject_path = os.path.join(BASE_DATASET_PATH, subject_num)
    # fnirs_folder = os.path.join(subject_path, f"{subject_id}-NIRS")
    # FIX: removed the intermediate numeric subfolder (e.g. "001") which the user's actual dataset
    # layout does not have; fNIRS data sits directly under BASE_DATASET_PATH\VPxxx-NIRS.
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")  # NEW: flat folder structure

    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')  # NEW: fNIRS trial markers needed for the quality-check plot and (optionally) downstream use
    mnt_path = os.path.join(fnirs_folder, 'mnt_nback.mat')

    if not os.path.exists(cnt_path):
        raise FileNotFoundError(f"fNIRS file not found: {cnt_path}")

    print(f"Loading raw fNIRS from: {cnt_path}")

    # cnt_data = loadmat(cnt_path)
    # cnt = cnt_data['cnt_nback'][0, 0]
    #
    # mnt_data = loadmat(mnt_path)
    # mnt = mnt_data['mnt_nback'][0, 0]
    # FIX (Problem 1): the lines above used the default scipy.io.loadmat() call (struct_as_record=True),
    # while the rest of the pipeline (load_eeg_data) consistently uses
    # struct_as_record=False, squeeze_me=True. Mixing the two access styles is fragile: under the
    # default mode, struct fields are accessed via numpy structured-array indexing (e.g. cnt['oxy'][0,0]['x']),
    # while under struct_as_record=False they are accessed as plain Python attributes (e.g. cnt.oxy.x).
    # Verified against the actual VP001 .mat structure (inspect_data.py output): cnt_nback has fields
    # ['oxy','deoxy'], each of which is itself a struct with field 'x' holding the (n_samples, 36) signal,
    # and mnt_nback has fields ['x','y','pos_3d','clab'] holding (36,) coordinate vectors. Using the
    # unified squeeze_me=True / struct_as_record=False loader removes the [0,0] nested-indexing style and
    # avoids a silent shape mismatch in spatial_mapping() if the indexing assumption is ever wrong.
    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)  # NEW: unified loader (matches load_eeg_data)
    real_cnt_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]     # NEW
    cnt = mat_cnt[real_cnt_keys[0]]                                          # NEW: cnt is now a MATLAB-struct object with .oxy / .deoxy attributes

    mat_mnt = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)  # NEW: unified loader (matches load_eeg_data)
    real_mnt_keys = [k for k in mat_mnt.keys() if not k.startswith('__')]     # NEW
    mnt = mat_mnt[real_mnt_keys[0]]                                          # NEW: mnt is now a MATLAB-struct object with .x / .y attributes

    # Load fNIRS trial markers (NEW: needed only for the before/after QC plot at the end of this
    # function; segmentation itself happens later using EEG_TRIAL_LABEL_MAP / FNIRS_TRIAL_LABEL_MAP
    # inside extract_synchronized_windows, which keeps this function's return signature unchanged).
    fnirs_events_time = None  # NEW
    fnirs_events_code = None  # NEW
    if os.path.exists(mrk_path):  # NEW
        mat_fmrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)  # NEW
        real_fmrk_keys = [k for k in mat_fmrk.keys() if not k.startswith('__')]    # NEW
        fmrk = mat_fmrk[real_fmrk_keys[0]]                                         # NEW
        if hasattr(fmrk, 'time') and hasattr(fmrk, 'event') and hasattr(fmrk.event, 'desc'):  # NEW
            fnirs_events_time = np.asarray(fmrk.time).flatten()                     # NEW
            fnirs_events_code = np.asarray(fmrk.event.desc).flatten()               # NEW

    # Extract actual sampling rate (making sure it's 10Hz)
    # sfreq_fnirs_raw = float(cnt['fs'][0, 0]) if 'fs' in cnt.dtype.names else 10.0
    # FIX (Problem 1, continued): 'fs' is a field of cnt.oxy / cnt.deoxy (each a sub-struct), not of
    # cnt itself (verified: cnt_nback.oxy.fs == 10, cnt_nback.deoxy.fs == 10 in the inspected VP001 file).
    # The original line would raise an AttributeError under the new loader (cnt has no .dtype/.fs), so
    # it must read fs from the oxy sub-struct.
    sfreq_fnirs_raw = float(cnt.oxy.fs) if hasattr(cnt, 'oxy') and hasattr(cnt.oxy, 'fs') else 10.0  # NEW
    print(f"raw fNIRS sampling rate: {sfreq_fnirs_raw} Hz")

    # Extract raw oxy/deoxy
    # if 'oxy' in cnt.dtype.names:
    #     oxy_raw = cnt['oxy'][0, 0]['x']
    #     deoxy_raw = cnt['deoxy'][0, 0]['x']
    # else:
    #     oxy_raw = cnt['oxy']
    #     deoxy_raw = cnt['deoxy']
    # FIX (Problem 1, continued): with struct_as_record=False, squeeze_me=True, the signal matrix is
    # accessed as cnt.oxy.x / cnt.deoxy.x (plain attribute access), matching the (n_samples, 36) shape
    # confirmed in inspect_data.py for VP001 (21178, 36).
    oxy_raw = np.asarray(cnt.oxy.x, dtype=np.float64)     # NEW
    deoxy_raw = np.asarray(cnt.deoxy.x, dtype=np.float64)  # NEW

    print(f"Raw fNIRS: oxy {oxy_raw.shape}, deoxy {deoxy_raw.shape}")

    # Convert to HbO/HbR
    # print("Converting to HbO/HbR...")
    # hbo_raw, hbr_raw = calculate_hbo_hbr(oxy_raw, deoxy_raw)
    # CHANGED (EXPERIMENTAL -- Beer-Lambert removal test, per chat history): the official dataset
    # Data Descriptor (Shin et al. 2018) explicitly states "cnt files contain deoxy/oxy-hemoglobin
    # data as separate fields... The raw NIRS optical intensity data converted to deoxy-(HbR) and
    # oxy hemoglobin (HbO) concentration changes... using the modified Beer-Lambert law" BEFORE
    # distribution -- confirmed further by the dataset description PDF itself, which explicitly
    # labels these fields "oxy hemoglobin data" / "deoxy hemoglobin data" (not raw optical density).
    # oxy_raw/deoxy_raw are therefore already ΔHbO/ΔHbR; calling calculate_hbo_hbr() on them
    # re-applies Beer-Lambert to already-converted concentration data, which is physically incorrect.
    # calculate_hbo_hbr() itself is left defined-but-unused below (not deleted), per the instruction
    # not to touch other parts of the file.
    print("Using oxy/deoxy fields directly as HbO/HbR (Beer-Lambert already applied by dataset authors; see Shin et al. 2018 Data Descriptor)...")  # NEW
    hbo_raw, hbr_raw = oxy_raw, deoxy_raw  # NEW: no conversion -- oxy_raw IS HbO, deoxy_raw IS HbR

    # Wavelet denoising 
    # print("Wavelet denoising...")
    # hbo_wv = wavelet_denoising(hbo_raw)
    # hbr_wv = wavelet_denoising(hbr_raw)
    #
    # MARA on full signal
    # print("MARA...")
    # hbo_mara = apply_mara(hbo_wv, fs=sfreq_fnirs_raw)
    # hbr_mara = apply_mara(hbr_wv, fs=sfreq_fnirs_raw)
    #
    # Low-pass filter 
    # FNIRS_LOWPASS_CUTOFF_HZ = 0.2
    # hbo_filtered = lowpass_filter(hbo_mara, fs=sfreq_fnirs_raw, cutoff=FNIRS_LOWPASS_CUTOFF_HZ)
    # hbr_filtered = lowpass_filter(hbr_mara, fs=sfreq_fnirs_raw, cutoff=FNIRS_LOWPASS_CUTOFF_HZ)
    # CHANGED (EXPERIMENTAL -- isolated fNIRS-filtering-removal test, Priority 2, per chat history):
    # Ding et al. (2026, Information Fusion, "MSTFDN"), using this SAME dataset, applies NO
    # wavelet denoising, NO MARA, and NO separate lowpass/bandpass filtering to fNIRS at all --
    # their stated rationale: "since the fNIRS branch uses convolution kernels with a large size
    # and receptive field (large size temporal convolution is equivalent to low-pass filtering in
    # the frequency domain), the model naturally focuses on low-frequency components during
    # feature learning. Therefore, we did not perform separate filtering preprocessing." Their
    # fNIRS pipeline is simply: Beer-Lambert (already done by dataset authors) -> baseline
    # correction -> z-score normalization. This test skips wavelet_denoising()/apply_mara()/
    # lowpass_filter() entirely (functions left defined-but-unused, not deleted) to test this
    # simplification hypothesis directly against our own hand-designed filtering pipeline, on the
    # SAME CONFIRMED baseline-corrected Run H pipeline (isolated from Priority 1's baseline-window
    # change and the still-unconfirmed EEG filter change).
    print("Skipping wavelet/MARA/lowpass filtering (MSTFDN-style: relying on the model's own large-kernel convolutions for implicit smoothing)...")  # NEW
    hbo_filtered, hbr_filtered = hbo_raw, hbr_raw  # NEW: pass-through, no denoising/artifact-removal/filtering applied

    # dynamic downsampling to 10 Hz
    print(f"Resampling from {sfreq_fnirs_raw} Hz to {SFREQ_FNIRS} Hz...")
    from scipy.signal import resample
    n_samples = int(hbo_filtered.shape[0] * SFREQ_FNIRS / sfreq_fnirs_raw)
    hbo_resampled = resample(hbo_filtered, n_samples, axis=0)
    hbr_resampled = resample(hbr_filtered, n_samples, axis=0)

    # spatial mapping to 16x16
    print("Spatial mapping to 16x16...")
    hbo_mapped = spatial_mapping(hbo_resampled, mnt)
    hbr_mapped = spatial_mapping(hbr_resampled, mnt)

    # Stack as channels: (n_samples, 16, 16, 2)
    fnirs_data = np.stack([hbo_mapped, hbr_mapped], axis=-1)

    print(f"fNIRS preprocessed: {fnirs_data.shape}")

    # NEW (user request): save a before/after preprocessing QC plot for fNIRS (one representative
    # channel, raw HbO vs fully preprocessed/resampled HbO). Saved per-subject next to that subject's
    # output folder. Does not alter any returned value or downstream logic.
    try:  # NEW
        qc_channel = 0  # NEW: first fNIRS channel, arbitrary but fixed for reproducibility
        t_raw = np.arange(oxy_raw.shape[0]) / sfreq_fnirs_raw  # NEW
        t_proc = np.arange(hbo_resampled.shape[0]) / SFREQ_FNIRS  # NEW
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)  # NEW
        axes[0].plot(t_raw, oxy_raw[:, qc_channel], color='tab:red', linewidth=0.6)  # NEW
        axes[0].set_title(f"{subject_id} - fNIRS RAW HbO (channel {qc_channel}, before preprocessing)")  # NEW
        axes[0].set_xlabel("Time (s)")  # NEW
        axes[0].set_ylabel("HbO (raw units)")  # NEW
        axes[1].plot(t_proc, hbo_resampled[:, qc_channel], color='tab:blue', linewidth=0.8)  # NEW
        axes[1].set_title(f"{subject_id} - fNIRS PREPROCESSED HbO (channel {qc_channel}, after wavelet+MARA+lowpass+resample)")  # NEW
        axes[1].set_xlabel("Time (s)")  # NEW
        axes[1].set_ylabel("HbO (mmol/L)")  # NEW
        plt.tight_layout()  # NEW
        qc_dir = os.path.join(OUTPUT_PATH, subject_id)  # NEW
        os.makedirs(qc_dir, exist_ok=True)  # NEW
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=150)  # NEW
        plt.close(fig)  # NEW
        print(f"Saved fNIRS before/after QC plot for {subject_id}")  # NEW
    except Exception as e:  # NEW
        print(f"Could not save fNIRS QC plot for {subject_id}: {e}")  # NEW

    # return fnirs_data
    # CHANGED (CRITICAL SYNCHRONIZATION FIX): diagnostic testing (see chat history) directly compared
    # EEG's own trial markers (codes 112/128/144) against fNIRS's own trial markers (codes 7/8/9),
    # both loaded independently from their respective mrk_nback.mat files. This revealed that the
    # two devices do NOT share a common time origin: within each of the 3 class "sessions" the two
    # devices agree almost exactly (delta ~0.1-0.3s, pure jitter), but ACROSS session boundaries a
    # near-constant ~46-47s jump appears (e.g. VP001: session 1 delta=16.6s, session 2 delta=62.7s,
    # session 3 delta=110.0s), consistently reproduced across all 5 subjects tested. Since
    # extract_synchronized_windows() previously computed the fNIRS window position purely by
    # converting EEG's own event time to an fNIRS sample index (assuming a shared t=0), this means
    # for the 2-back and 3-back sessions the fNIRS window was being read from a position roughly
    # 46-93 seconds away from where it should be -- i.e. NOT synchronized with the EEG window it was
    # being paired with. fnirs_events_time and fnirs_events_code (already loaded above, previously
    # used only for the QC plot) are now also returned so extract_synchronized_windows() can use
    # fNIRS's OWN trial markers as the anchor for fNIRS windowing, instead of a value derived from
    # EEG's clock.
    return fnirs_data, fnirs_events_time, fnirs_events_code  # CHANGED: also return fNIRS's own trial markers

# fNIRS HELPER FUNCTIONS
def calculate_hbo_hbr(oxy, deoxy):
    lambda1, lambda2 = 760, 850
    L = 3.0

    def get_dpf(lmbda, age=25):
        alpha, beta, gamma = 223.3, 0.05624, 0.8493
        delta, eps, zeta = -5.723e-7, 0.001245, -0.9025
        return alpha + beta * (age ** gamma) + delta * (lmbda ** 3) + eps * (lmbda ** 2) + zeta * lmbda

    dpf1 = get_dpf(lambda1)
    dpf2 = get_dpf(lambda2)

    e1_hbo, e1_hbr = 0.586, 1.54852
    e2_hbo, e2_hbr = 1.058, 0.69132

    denom = L * (e1_hbr * e2_hbo - e2_hbr * e1_hbo)
    hbo = (e1_hbr * (deoxy / dpf2) - e2_hbr * (oxy / dpf1)) / denom
    hbr = (e1_hbo * (oxy / dpf1) - e2_hbo * (deoxy / dpf2)) / denom

    return hbo, hbr

def wavelet_denoising(data, alpha=0.1, wavelet='db5'):
    cleaned = np.zeros_like(data)
    n_samples, n_channels = data.shape

    for ch in range(n_channels):
        signal = data[:, ch]
        level = min(int(np.floor(np.log2(n_samples))), 6)
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            d_coeffs = coeffs[i].copy()
            sigma = np.median(np.abs(d_coeffs - np.median(d_coeffs))) / 0.6745
            if sigma > 0:
                threshold = sigma * stats.norm.ppf(1 - alpha / 2)
                d_coeffs[np.abs(d_coeffs) > threshold] = 0
            new_coeffs.append(d_coeffs)
        cleaned[:, ch] = pywt.waverec(new_coeffs, wavelet)[:n_samples]

    return cleaned

def apply_mara(data, fs=10.0, window_size=41, threshold_factor=1.8):
    n_samples, n_channels = data.shape
    corrected = np.zeros_like(data)
    edge_len = int(0.5 * fs)

    for ch in range(n_channels):
        signal = data[:, ch].copy()
        msd = pd.Series(signal).rolling(window=window_size, center=True, min_periods=1).std().to_numpy()
        threshold = np.median(msd) + (threshold_factor * np.std(msd))
        ma_mask = (msd > threshold).astype(int)

        diff_mask = np.diff(ma_mask, prepend=0, append=0)
        starts = np.where(diff_mask == 1)[0]
        ends = np.where(diff_mask == -1)[0]
        indices = sorted(list(set([0, n_samples] + list(starts) + list(ends))))
        segments = [signal[indices[i]:indices[i+1]].copy() for i in range(len(indices) - 1)]

        if not segments:
            corrected[:, ch] = signal
            continue

        final_parts = [segments[0]]
        for i in range(1, len(segments)):
            prev_seg = final_parts[-1]
            curr_seg = segments[i]
            val_prev = np.median(prev_seg[-edge_len:]) if len(prev_seg) >= edge_len else prev_seg[-1]
            val_curr = np.median(curr_seg[:edge_len]) if len(curr_seg) >= edge_len else curr_seg[0]
            shift = val_prev - val_curr

            if ma_mask[indices[i]] == 1:
                x = np.arange(len(curr_seg))
                s_val = max(np.var(curr_seg) * len(curr_seg), 1e-6)
                try:
                    spline = UnivariateSpline(x, curr_seg, s=s_val)
                    curr_seg = curr_seg - spline(x)
                    val_curr_new = np.median(curr_seg[:edge_len]) if len(curr_seg) >= edge_len else curr_seg[0]
                    curr_seg += (val_prev - val_curr_new)
                except:
                    curr_seg += shift
            else:
                if abs(shift) > 2 * np.std(msd):
                    curr_seg += shift
            final_parts.append(curr_seg)

        full_signal = np.concatenate(final_parts)[:n_samples]
        t_idx = np.arange(n_samples)
        original_trend = np.polyval(np.polyfit(t_idx, data[:, ch], 1), t_idx)
        corrected_trend = np.polyval(np.polyfit(t_idx, full_signal, 1), t_idx)
        full_signal += (original_trend - corrected_trend)
        corrected[:, ch] = full_signal

    return corrected

def lowpass_filter(data, fs=10.0, cutoff=0.15):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(3, normal_cutoff, btype='low')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered

def spatial_mapping(data, mnt):
    n_samples, n_channels = data.shape
    # x_coords = np.asarray(mnt.x).flatten()
    # y_coords = np.asarray(mnt.y).flatten()
    # FIX (projection accuracy): mnt.x / mnt.y are pre-computed 2D coordinates from the BBCI
    # toolbox's own projection, which is not necessarily the azimuthal equidistant projection
    # described in the TSFNet reference paper (Zhang, Yin, Yuan, Sensors 2025). Using the raw 3D
    # channel positions (mnt.pos_3d, shape (3, n_channels), confirmed present in the fNIRS mnt_nback
    # struct via inspect_data.py) and projecting them with a genuine azimuthal equidistant
    # projection (angular distance from the vertex is preserved: theta = angle from north pole,
    # phi = azimuthal angle) more faithfully reproduces the paper's methodology than reusing the
    # toolbox's built-in 2D coordinates.
    pos_3d = np.asarray(mnt.pos_3d, dtype=np.float64)  # NEW: shape (3, n_channels)
    x3d, y3d, z3d = pos_3d[0, :], pos_3d[1, :], pos_3d[2, :]  # NEW
    r = np.sqrt(x3d**2 + y3d**2 + z3d**2)  # NEW
    r_safe = np.where(r > 0, r, 1.0)  # NEW: avoid division by zero for any degenerate channel position
    theta = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))  # NEW: angle from the vertex (pole)
    phi = np.arctan2(y3d, x3d)  # NEW: azimuthal angle
    x_coords = theta * np.cos(phi)  # NEW: azimuthal equidistant projection
    y_coords = theta * np.sin(phi)  # NEW

    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):16j, 
                               min(y_coords):max(y_coords):16j]
    points = np.column_stack((x_coords, y_coords))

    # mapped = np.zeros((n_samples, 16, 16), dtype=np.float32)
    # for t in range(n_samples):
    #     grid_z = griddata(points, data[t, :], (grid_x, grid_y), method='cubic', fill_value=0)
    #     mapped[t] = grid_z
    # PERF FIX: the loop above calls griddata() once per time sample, re-computing the Delaunay
    # triangulation of `points` from scratch every single time -- even though `points` (electrode
    # positions) NEVER changes across time samples for a given subject; only the values (data[t,:])
    # change. Confirmed via VP001/VP002 real runs, EEG spatial mapping (394930 / 395238 samples at
    # 200 Hz) was the dominant cost of the ~28-32 min per-subject runtime. griddata's 'cubic' 2D
    # path internally builds scipy.interpolate.CloughTocher2DInterpolator(points, values, ...),
    # which officially supports (a) being given a pre-built scipy.spatial.Delaunay triangulation
    # instead of raw points, and (b) vector-valued `values` of shape (n_points, ...trailing_dims),
    # interpolating all trailing "columns" (here: all time samples) in one batched call. Verified
    # numerically bit-identical (max abs diff = 0.0) against the old per-sample loop on synthetic
    # data of the same shape before applying this change (see chat history for the verification
    # script). This is a pure performance refactor -- the interpolation math is unchanged.
    tri = Delaunay(points)  # NEW: triangulation depends only on `points`, computed ONCE instead of n_samples times
    values = data.T  # NEW: (n_samples, n_channels) -> (n_channels, n_samples); CloughTocher2DInterpolator
                      # NEW: requires values.shape[0] == points.shape[0] (== n_channels here)
    interp = CloughTocher2DInterpolator(tri, values, fill_value=0)  # NEW: gradient estimation + patch setup done once, for all time samples together
    result = interp((grid_x, grid_y))  # NEW: shape (16, 16, n_samples) -- one batched evaluation instead of n_samples separate ones
    mapped = np.transpose(result, (2, 0, 1)).astype(np.float32)  # NEW: (16, 16, n_samples) -> (n_samples, 16, 16), matching the original return shape/dtype exactly

    return mapped

# NEW (EEG spatial mapping support): spatial_mapping() above was written for fNIRS, which reads
# 3D channel positions from mnt.pos_3d (a field already present in the dataset's mnt_nback.mat
# struct). EEG has no equivalent .mat mnt struct -- its electrode positions instead live inside
# the MNE montage set in load_eeg_data() via raw.set_montage('standard_1005', match_case=False,
# on_missing='ignore'). _EEGMontageAdapter + get_eeg_mnt() below extract those positions, in the
# SAME channel order as raw.ch_names (so they line up 1:1 with data columns passed to
# spatial_mapping), into a lightweight object exposing .pos_3d with shape (3, n_channels) --
# exactly the interface spatial_mapping() already expects -- so spatial_mapping() itself does not
# need to change at all and is reused UNCHANGED for EEG, with the same azimuthal equidistant
# projection methodology already used for fNIRS.
# NEW (EEG spatial mapping support): spatial_mapping() above was written for fNIRS, which reads
# 3D channel positions from mnt.pos_3d (a field present in the dataset's fNIRS mnt_nback.mat
# struct). The EEG folder has its OWN mnt_nback.mat with the exact same format (pos_3d shape
# (3, 30), clab shape (30,)), confirmed via direct inspection of VP001-EEG/mnt_nback.mat -- so EEG
# positions come from the REAL, dataset-provided file, not from MNE's generic standard_1005
# template (an earlier, incorrect assumption; see the comment above raw.set_montage() in
# load_eeg_data(), where that template montage is kept only for interpolate_bads()'s internal use).
# _EEGMontageAdapter + get_eeg_mnt() below select, from the full 30-channel mnt_nback.mat struct,
# only the positions of the channels that SURVIVED preprocessing (raw.ch_names, after
# VEOG/HEOG/TP9 are dropped in preprocess_eeg), matched BY NAME against mnt_nback.mat's own clab
# list (channel order in cnt_nback.mat and mnt_nback.mat is not guaranteed to be identical, so
# matching by name rather than by raw positional index is required for correctness). The result is
# a lightweight object exposing .pos_3d with shape (3, n_channels) -- exactly the interface
# spatial_mapping() already expects -- so spatial_mapping() itself needs no changes at all.
class _EEGMontageAdapter:  # NEW
    """Minimal stand-in for the fNIRS .mat mnt struct, exposing only the .pos_3d attribute that
    spatial_mapping() reads."""
    def __init__(self, pos_3d):  # NEW
        self.pos_3d = pos_3d  # NEW: shape (3, n_channels), matching mnt.pos_3d's shape for fNIRS

def get_eeg_mnt(raw, eeg_mnt_struct):  # CHANGED: now takes the real mnt_nback.mat struct (loaded once in load_eeg_data) instead of deriving positions from raw's MNE montage
    """
    Builds an _EEGMontageAdapter (pos_3d shape (3, n_channels)) from the REAL electrode 3D
    positions in the dataset's own EEG mnt_nback.mat (eeg_mnt_struct.pos_3d / .clab), selecting
    and ordering them to match raw.ch_names exactly. Must be called AFTER any channel drops
    (VEOG/HEOG/TP9 in preprocess_eeg) so the returned position order matches the channel order of
    the data that will be passed to spatial_mapping().
    """
    full_clab = [str(c).strip() for c in np.asarray(eeg_mnt_struct.clab).flatten()]  # NEW: e.g. 30 names, same cleanup style as cnt_struct.clab in load_eeg_data
    full_pos_3d = np.asarray(eeg_mnt_struct.pos_3d, dtype=np.float64)  # NEW: shape (3, 30)
    name_to_idx = {name: i for i, name in enumerate(full_clab)}  # NEW
    name_to_idx_upper = {name.upper(): i for i, name in enumerate(full_clab)}  # NEW: case-insensitive fallback, matches match_case=False used elsewhere in this pipeline

    pos_list = []  # NEW
    missing = []  # NEW
    for ch in raw.ch_names:  # NEW
        idx = name_to_idx.get(ch, name_to_idx_upper.get(ch.upper(), None))  # NEW
        if idx is None:  # NEW: channel present in raw but not found in mnt_nback.mat's clab list
            missing.append(ch)  # NEW
            pos_list.append([0.0, 0.0, 0.0])  # NEW: origin fallback (theta=0 -> maps to grid center)
        else:  # NEW
            pos_list.append(full_pos_3d[:, idx])  # NEW

    if missing:  # NEW
        print(f"  WARNING: {len(missing)} EEG channel(s) not found in mnt_nback.mat's clab "  # NEW
              f"(spatial_mapping will place them at the grid center): {missing}")  # NEW

    pos_3d = np.array(pos_list, dtype=np.float64).T  # NEW: shape (3, n_channels)
    return _EEGMontageAdapter(pos_3d)  # NEW

# PREPROCESS EEG
def preprocess_eeg(raw, events):
    print("Preprocessing EEG...")

    # NEW (user request): keep a copy of the raw signal (one representative channel) for the
    # before/after preprocessing QC plot saved at the end of this function.
    raw_before_qc = raw.copy()  # NEW

    raw_filtered = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    raw_filtered.filter(l_freq=1.0, h_freq=40.0, method='fir', verbose=False)

    # Detect bad channels
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
        print(f"Bad channels: {bad_channels}")
        raw_filtered.info['bads'] = bad_channels

    # NEW (fix support): remember which channels were flagged as bad BEFORE interpolate_bads()
    # resets info['bads']. This list is needed further below to correctly compute the ICA rank,
    # since interpolate_bads(reset_bads=True) clears info['bads'] and the original code's
    # exclude='bads' filter therefore silently included the interpolated channels in the ICA fit.
    interpolated_channels = list(raw_filtered.info['bads'])  # NEW

    if raw_filtered.info['bads']:
        raw_filtered.interpolate_bads(reset_bads=True, verbose=False)

    raw_filtered.set_eeg_reference('average', projection=False, verbose=False)

    # ICA
    picks_for_ica = mne.pick_types(raw_filtered.info, eeg=True, eog=False, exclude='bads')
    # corrected_rank = max(1, len(picks_for_ica) - 1)
    # FIX (rank problem): interpolate_bads(reset_bads=True) above already cleared info['bads'],
    # so exclude='bads' here has NOTHING to exclude and picks_for_ica silently includes every
    # interpolated channel. Interpolated channels are linear combinations of their neighbours and
    # add NO new degrees of freedom to the data, so the true data rank is
    # (n_good_channels - 1 for CAR - n_interpolated_channels), not just (n_channels - 1).
    # Asking ICA (via n_components=corrected_rank) for more components than the true rank forces
    # the algorithm to spread a single real source (e.g. the eye-blink dipole) across multiple
    # weak, "diluted" components, none of which individually scores high enough in ICLabel
    # (whether by argmax or by the probability threshold added above) to be flagged as
    # 'eye blink'. This is very likely the main reason eye-blink artifacts survive in this
    # pipeline's output while code 1 (which never interpolates channels) does not have this issue.

###########

    # corrected_rank = max(1, len(picks_for_ica) - 1 - len(interpolated_channels))
    # print(f"ICA rank: {corrected_rank} (picks={len(picks_for_ica)}, interpolated={len(interpolated_channels)})")
    # ica = mne.preprocessing.ICA(n_components=corrected_rank, method='infomax',
    #                             fit_params=dict(extended=True), random_state=42, verbose=False)
    # FIX (rank problem, cleaner solution): the manual rank correction above (subtracting the number
    # of interpolated channels) is a reasonable workaround, but n_components=0.99 solves the same
    # problem more robustly and automatically: instead of a FIXED component count, ICA's internal PCA
    # step keeps only as many components as needed to explain 99% of the data's variance. Interpolated
    # channels contribute (near-)zero additional variance since they are linear combinations of their
    # neighbours, so PCA naturally discards that redundant dimensionality without needing to manually
    # count and subtract interpolated channels. This avoids asking ICA for more components than the
    # true data rank, which is what caused the eye-blink source to be diluted across several weak,
    # under-scored components in the original pipeline (see comment above find_bads_eog for the full
    # rationale of that issue).
    # method='picard' (Ablin, Cardoso & Gramfort, 2018, IEEE Signal Processing Letters) is mathematically
    # equivalent to extended infomax but converges faster and more reliably; it is the ICA algorithm
    # used in the reference implementation (code 1) and is increasingly the default recommendation in
    # recent BCI/EEG preprocessing literature.
    print(f"ICA: using variance-based component selection (99% explained variance), "  # NEW
          f"picks={len(picks_for_ica)}, interpolated={len(interpolated_channels)}")     # NEW
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard',                     # NEW
                                fit_params=dict(ortho=False, extended=True),            # NEW
                                random_state=42, verbose=False)                          # NEW
    ica.fit(raw_filtered, picks=picks_for_ica)


#########    

####
    ic_labels = mne_icalabel.label_components(raw_filtered, ica, method='iclabel')

    ARTIFACT_CLASSES_IDX = {  # NEW: standard ICLabel class order (verified from ica.labels_scores_ shape=(n,7))
        'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
        'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6
    }
    ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']  # NEW
    PROB_THRESHOLD = 0.30  # NEW

    # Original argmax-based selection (unchanged logic)
    artifact_idx_argmax = [idx for idx, label in enumerate(ic_labels['labels'])
                           if label in ARTIFACT_TARGETS]  # CHANGED: renamed

    # NEW: probability-threshold based selection using ica.labels_scores_ (shape: n_components x 7)
    artifact_idx_prob = []  # NEW
    proba_matrix = ica.labels_scores_  # NEW: verified shape (27, 7) for this dataset
    for comp_idx in range(proba_matrix.shape[0]):  # NEW
        for cls_name in ARTIFACT_TARGETS:  # NEW
            cls_col = ARTIFACT_CLASSES_IDX.get(cls_name)  # NEW
            if cls_col is not None and proba_matrix[comp_idx, cls_col] >= PROB_THRESHOLD:  # NEW
                artifact_idx_prob.append(comp_idx)  # NEW
                break  # NEW: one match per component is enough

    # eog_indices, eog_scores = None, None  (did not exist in the previous version of this file)
    # FIX (Problem: no direct EOG-correlation check): code 1 (the reference implementation) relies
    # primarily on mne.preprocessing.ICA.find_bads_eog(), which correlates each ICA component's time
    # course directly against the RECORDED VEOG channel -- a ground-truth signal, independent of
    # ICLabel's scalp-topography/spectrum-based classifier. Code 2 previously relied ONLY on
    # ICLabel (via argmax and/or probability threshold), which never consults the VEOG/HEOG channels
    # that are already loaded and typed as 'eog' in load_eeg_data(). Adding find_bads_eog() here
    # gives ICA exclusion a second, independent detection path so that a genuine eye-blink component
    # that ICLabel under-scores (e.g. due to the rank/dilution issue fixed above) can still be
    # caught via its correlation with the real VEOG signal, matching code 1's methodology.
    eog_indices_veog, eog_scores_veog = [], None  # NEW: defaults in case VEOG channel is missing
    eog_indices_heog, eog_scores_heog = [], None  # NEW: defaults in case HEOG channel is missing
    try:  # NEW
        if 'VEOG' in raw.ch_names:  # NEW: raw is the ORIGINAL (unfiltered) signal passed into this function, matching code 1's use of raw_input
            eog_indices_veog, eog_scores_veog = ica.find_bads_eog(raw, ch_name='VEOG', threshold=2.0, verbose=False)  # NEW
            print(f"  find_bads_eog(VEOG) flagged components: {eog_indices_veog}")  # NEW
        if 'HEOG' in raw.ch_names:  # NEW: also check HEOG (horizontal eye movements), which code 1 did not check but is available in this dataset
            eog_indices_heog, eog_scores_heog = ica.find_bads_eog(raw, ch_name='HEOG', threshold=2.0, verbose=False)  # NEW
            print(f"  find_bads_eog(HEOG) flagged components: {eog_indices_heog}")  # NEW
    except Exception as e:  # NEW
        print(f"  find_bads_eog failed: {e}")  # NEW
#########


    # NEW (methodological decision, not a named standard algorithm -- see justification below):
    # Gate EOG-correlation-only hits (components flagged ONLY via find_bads_eog, with no ICLabel
    # support) behind an ICLabel 'brain' probability ceiling. Rationale: EOG-correlation is a
    # well-established artifact-detection method (Jung et al. 1998/2000; implemented in EEGLAB,
    # FieldTrip, MNE), and combining multiple independent artifact detectors is itself a recognized
    # practice (e.g. SASICA, Chaumon et al. 2015, J. Neurosci. Methods, which combines EOG-correlation,
    # autocorrelation, topography focus, and discontinuity as independent criteria). However, the
    # literature also documents that EOG-correlation alone can produce false positives, since some
    # genuine brain sources (e.g. frontal activity related to eye-movement planning / corollary
    # discharge) can correlate with EOG without being ocular artifacts themselves. The specific
    # threshold below (BRAIN_PROB_CEILING = 0.5) is NOT a published/validated cutoff from any single
    # paper -- it is a conservative engineering choice made for this thesis to reduce the risk of
    # discarding real brain signal when a component's only supporting evidence is EOG-correlation.
    # Any component that ALSO has independent ICLabel support (argmax or probability threshold) is
    # NOT affected by this gate and is still excluded regardless of its 'brain' probability.
    BRAIN_PROB_CEILING = 0.5  # NEW: conservative, thesis-specific choice -- not a literature-standard value
    BRAIN_CLASS_COL = ARTIFACT_CLASSES_IDX['brain']  # NEW

    eog_only_idx = (set(eog_indices_veog) | set(eog_indices_heog)) - (set(artifact_idx_argmax) | set(artifact_idx_prob))  # NEW: components flagged ONLY by EOG-correlation, with no ICLabel support at all
    eog_only_rescued = {idx for idx in eog_only_idx if proba_matrix[idx, BRAIN_CLASS_COL] >= BRAIN_PROB_CEILING}  # NEW: rescue if ICLabel itself thinks it's likely brain signal
    if eog_only_rescued:  # NEW
        print(f"  Rescued from EOG-only rejection (ICLabel brain-prob >= {BRAIN_PROB_CEILING}): {sorted(eog_only_rescued)}")  # NEW

    artifact_idx = sorted(
        (set(artifact_idx_argmax) | set(artifact_idx_prob) | set(eog_indices_veog) | set(eog_indices_heog))
        - eog_only_rescued
    )  # CHANGED: same union as before, minus the conservative brain-probability rescue applied only to EOG-only hits



###########
    # NEW: detailed log for each flagged component
    all_class_names = ['brain', 'muscle artifact', 'eye blink', 'heart beat', 'line noise', 'channel noise', 'other']  # NEW
    for comp_idx in artifact_idx:  # NEW
        probs_str = ", ".join([f"{all_class_names[c]}={proba_matrix[comp_idx, c]:.2f}"  # NEW
                               for c in range(7)])  # NEW
        reason = []  # NEW
        if comp_idx in artifact_idx_argmax:  # NEW
            reason.append(f"argmax='{ic_labels['labels'][comp_idx]}'")  # NEW
        if comp_idx in artifact_idx_prob:  # NEW
            reason.append(f"prob>={PROB_THRESHOLD}")  # NEW
        if comp_idx in eog_indices_veog:  # NEW
            reason.append("VEOG-correlation")  # NEW
        if comp_idx in eog_indices_heog:  # NEW
            reason.append("HEOG-correlation")  # NEW
        print(f"  IC{comp_idx} flagged [{', '.join(reason)}]: {probs_str}")  # NEW

    ica.exclude = artifact_idx
    if len(ica.exclude) > 0:
        ica.apply(raw_filtered, verbose=False)
        print(f"Removed {len(ica.exclude)} ICA components")
    else:
        print("Removed 0 ICA components (no artifact ICs detected)")  # NEW


####


    # Case-insensitive channel drop
    channels_to_drop = [ch for ch in raw_filtered.ch_names if ch.upper() in ['VEOG', 'HEOG', 'TP9']]
    if channels_to_drop:
        raw_filtered.drop_channels(channels_to_drop)
        print(f"Dropped channels: {channels_to_drop}")

    # NEW (user request, updated): save before/after preprocessing QC plots for EEG showing ALL
    # channels at once (not just one representative channel). Two SEPARATE figures are produced:
    # one figure with all RAW channels stacked, one figure with all PREPROCESSED channels stacked.
    # Each channel is plotted on its own horizontal row using a fixed vertical offset (the standard
    # "butterfly/stacked montage" view used by EEG viewers such as MNE's raw.plot()), so all channels
    # remain visually separable in a single image instead of being overlaid on top of each other.
    try:  # NEW
        def _plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):  # CHANGED: added tmin/tmax for zoomed view
            sfreq_plot = raw_obj.info['sfreq']               # NEW
            if tmin is not None or tmax is not None:         # NEW: optional time-window crop
                start_sample = int((tmin or 0.0) * sfreq_plot)             # NEW
                stop_sample = int((tmax * sfreq_plot)) if tmax is not None else None  # NEW
                data = raw_obj.get_data(start=start_sample, stop=stop_sample)         # NEW
                t = (np.arange(data.shape[1]) + start_sample) / sfreq_plot            # NEW
            else:                                            # NEW
                data = raw_obj.get_data()                      # NEW: (n_channels, n_times)
                t = np.arange(data.shape[1]) / sfreq_plot        # NEW
            ch_names_plot = raw_obj.ch_names                # NEW

            # Use the robust (median absolute deviation based) per-channel spread to set a fixed
            # vertical offset between channels, so that channels with very different amplitudes
            # (e.g. after CAR/ICA vs raw) are still readable and non-overlapping within the same figure.
            #spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 8 + 1e-12  # NEW
            spread = np.median(np.abs(data - np.median(data, axis=1, keepdims=True))) * 30 + 1e-12  # NEW
            n_ch_plot = data.shape[0]                        # NEW
            #fig_h = max(6, 0.28 * n_ch_plot)  # NEW: scale figure height with channel count
            fig_h = max(6, 0.5 * n_ch_plot)                 # NEW: scale figure height with channel count
            fig, ax = plt.subplots(figsize=(14, fig_h))      # NEW
            for i in range(n_ch_plot):                       # NEW
                offset = (n_ch_plot - 1 - i) * spread         # NEW: top channel = first channel name
                ax.plot(t, data[i] + offset, color=color, linewidth=0.4)  # NEW
            ax.set_yticks([(n_ch_plot - 1 - i) * spread for i in range(n_ch_plot)])  # NEW
            ax.set_yticklabels(ch_names_plot, fontsize=6)    # NEW
            ax.set_xlabel("Time (s)")                        # NEW
            ax.set_ylabel("Channel")                         # NEW
            ax.set_title(title)                              # NEW
            plt.tight_layout()                                # NEW
            return fig                                        # NEW

        fig_before = _plot_all_channels_stacked(  # NEW
            raw_before_qc, "EEG RAW - All Channels (before preprocessing)", 'tab:red'  # NEW
        )  # NEW
        fig_after = _plot_all_channels_stacked(  # NEW
            raw_filtered, "EEG PREPROCESSED - All Channels (after notch+bandpass+CAR+ICA)", 'tab:blue'  # NEW
        )  # NEW
        fig_before_zoom = _plot_all_channels_stacked(  # NEW: zoomed 10-30s view
            raw_before_qc, "EEG RAW - All Channels (before preprocessing, 10-30s)", 'tab:red', tmin=10.0, tmax=30.0  # NEW
        )  # NEW
        fig_after_zoom = _plot_all_channels_stacked(  # NEW: zoomed 10-30s view
            raw_filtered, "EEG PREPROCESSED - All Channels (after notch+bandpass+CAR+ICA, 10-30s)", 'tab:blue', tmin=10.0, tmax=30.0  # NEW
        )  # NEW


        # The subject_id is not directly available inside this function (it only receives raw/events),
        # so the figures are saved using generic names inside OUTPUT_PATH/_tmp_eeg_qc and renamed by
        # the caller (process_subject) right after this function returns. This keeps this function's
        # signature and call sites unchanged.
        tmp_qc_dir = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc")  # NEW
        os.makedirs(tmp_qc_dir, exist_ok=True)  # NEW
        fig_before.savefig(os.path.join(tmp_qc_dir, "last_eeg_before_allchannels.png"), dpi=150)  # NEW
        fig_after.savefig(os.path.join(tmp_qc_dir, "last_eeg_after_allchannels.png"), dpi=150)  # NEW
        fig_before_zoom.savefig(os.path.join(tmp_qc_dir, "last_eeg_before_allchannels_zoom10_30.png"), dpi=150)  # NEW
        fig_after_zoom.savefig(os.path.join(tmp_qc_dir, "last_eeg_after_allchannels_zoom10_30.png"), dpi=150)  # NEW
        plt.close(fig_before)  # NEW
        plt.close(fig_after)  # NEW
        plt.close(fig_before_zoom)  # NEW
        plt.close(fig_after_zoom)  # NEW
    except Exception as e:  # NEW
        print(f"Could not save EEG QC plot: {e}")  # NEW

    return raw_filtered


# SYNCHRONIZED SLIDING WINDOW EXTRACTION 
# def extract_synchronized_windows(eeg_raw, fnirs_data, events, sfreq_eeg):
# CHANGED: added eeg_spatial_data parameter -- the pre-computed, spatially-mapped (16x16 grid)
# continuous EEG signal for this subject (shape (n_times, 16, 16), produced once in
# process_subject() via spatial_mapping(), mirroring how fNIRS is spatially mapped once before
# windowing rather than per-window). eeg_raw (the MNE Raw object, channel x time) is still passed
# in and still used, but now ONLY for the amplitude-based artifact rejection check below, since
# that threshold (120 uV) is calibrated in electrode voltage units and is not a meaningful check
# on interpolated grid values.
# def extract_synchronized_windows(eeg_raw, eeg_spatial_data, fnirs_data, events, sfreq_eeg):
# CHANGED (CRITICAL SYNCHRONIZATION FIX, see the comment above load_and_preprocess_fnirs()'s return
# statement for the full diagnostic rationale): added fnirs_events_time / fnirs_events_code so each
# EEG trial can be paired with fNIRS's OWN trial marker instead of a position derived from EEG's
# clock, which diagnostic testing showed drifts by ~46-93s across the dataset's 3 class sessions.
# ══════════════════════════════════════════════════════════════════════════
# NEW (D2 -- isolated data-level fix): epoch-level ML-anomaly filter helpers
#
# These reproduce, exactly, the ML ensemble anomaly detector (Isolation
# Forest + LOF + Elliptic Envelope, majority vote) and the interval-merge
# fix already validated in eeg_artifact_detection_26subj_v3_english.py and
# epoch_filter_calibration_test.py. Nothing else in this file is touched.
# ══════════════════════════════════════════════════════════════════════════

def _qc_bandpower(x, sfreq, band):  # NEW (D2)
    f, pxx = welch(x, fs=sfreq, nperseg=min(512, len(x)))
    idx = (f >= band[0]) & (f <= band[1])
    return float(np.trapz(pxx[idx], f[idx]))


def _qc_extract_epoch_features(epoch, sfreq):  # NEW (D2)
    feats = []
    for sig in epoch:
        ch_f = [
            float(np.std(sig)), float(kurtosis(sig)), float(skew(sig)),
            float(np.ptp(sig)), float(np.sqrt(np.mean(sig ** 2))),
            _qc_bandpower(sig, sfreq, (0.5, 4)), _qc_bandpower(sig, sfreq, (4, 8)),
            _qc_bandpower(sig, sfreq, (8, 13)), _qc_bandpower(sig, sfreq, (13, 30)),
            _qc_bandpower(sig, sfreq, (30, min(90, sfreq / 2 - 1))),
        ]
        feats.extend(ch_f)
    return np.array(feats, dtype=np.float32)


def _qc_epoch_signal(data, sfreq, epoch_len=2.0, overlap=0.5):  # NEW (D2)
    step = int(epoch_len * sfreq * (1 - overlap))
    win  = int(epoch_len * sfreq)
    epochs, starts = [], []
    for s in range(0, data.shape[1] - win + 1, step):
        epochs.append(data[:, s:s + win])
        starts.append(s / sfreq)
    return np.array(epochs), np.array(starts)


def _qc_merge_anomaly_intervals(anomaly_epoch_starts, anomaly_epoch_len):  # NEW (D2)
    """
    Merges consecutive/overlapping raw anomaly epochs (2s long, 1s apart)
    into continuous [start, end] intervals before overlap is computed
    against sliding windows. Without this merge step, a long anomalous
    stretch spanning several consecutive flagged epochs would never
    accumulate more than a single epoch's overlap with any one window --
    see the bug found and fixed in epoch_filter_calibration_test.py.
    """
    if not anomaly_epoch_starts:
        return []
    intervals = sorted([(s, s + anomaly_epoch_len) for s in anomaly_epoch_starts])
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_ml_anomaly_intervals(eeg_raw, sfreq_eeg):  # NEW (D2)
    """
    Runs the Isolation Forest + LOF + Elliptic Envelope majority-vote
    ensemble on the FULL cleaned continuous EEG signal (same signal saved
    as {subject_id}_eeg_clean_raw.fif) and returns merged, continuous
    anomaly intervals in absolute time (seconds from the start of the
    recording). Computed ONCE per subject, before the trial/window loop,
    since it operates on the whole recording rather than per-window.
    """
    data_uv = eeg_raw.get_data() * 1e6   # Volt -> microvolt, same fix as the QC script
    epochs, times = _qc_epoch_signal(data_uv, sfreq_eeg, epoch_len=2.0, overlap=0.5)
    if len(epochs) < 5:
        return []

    X = np.array([_qc_extract_epoch_features(ep, sfreq_eeg) for ep in epochs], dtype=np.float32)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    X_sc = StandardScaler().fit_transform(X)

    iso  = IsolationForest(contamination=EPOCH_FILTER_CONTAMINATION, n_estimators=150, random_state=42)
    p_if = iso.fit_predict(X_sc)

    lof  = LocalOutlierFactor(n_neighbors=min(20, len(X_sc) - 1), contamination=EPOCH_FILTER_CONTAMINATION)
    p_lof = lof.fit_predict(X_sc)

    pca_x, ee = None, None  # NEW: pre-declared so cleanup below is always safe
    try:
        pca_x = PCA(n_components=min(10, X_sc.shape[1], X_sc.shape[0] - 1)).fit_transform(X_sc)
        ee    = EllipticEnvelope(contamination=EPOCH_FILTER_CONTAMINATION, random_state=42, support_fraction=0.9)
        p_ee  = ee.fit_predict(pca_x)
    except Exception:
        p_ee = np.ones(len(X_sc), dtype=int)

    votes = (p_if == -1).astype(int) + (p_lof == -1).astype(int) + (p_ee == -1).astype(int)
    majority_starts = times[votes >= 2].tolist()
    result = _qc_merge_anomaly_intervals(majority_starts, EPOCH_FILTER_ANOMALY_EPOCH_LEN)

    # NEW (memory leak fix): this function runs once per subject (26 times
    # total across the pipeline). Without explicit cleanup, the temporary
    # arrays below (each several hundred MB for a ~2000s recording) and the
    # fitted sklearn estimators (which each hold internal references to their
    # training data) were not being released promptly, causing memory to
    # accumulate across subjects and crash with MemoryError starting partway
    # through the 26-subject batch (observed starting at VP019).
    del data_uv, epochs, X, X_sc, iso, lof, p_if, p_lof, p_ee, votes, pca_x, ee  # NEW
    gc.collect()  # NEW: force immediate release back to the OS

    return result


def window_overlaps_anomaly(w_start_s, w_end_s, anomaly_intervals, overlap_threshold):  # NEW (D2)
    """True if the fraction of [w_start_s, w_end_s] covered by any merged
    anomaly interval is >= overlap_threshold."""
    window_dur = w_end_s - w_start_s
    if window_dur <= 0 or not anomaly_intervals:
        return False
    for a_start, a_end in anomaly_intervals:
        overlap = max(0.0, min(w_end_s, a_end) - max(w_start_s, a_start))
        if overlap / window_dur >= overlap_threshold:
            return True
    return False


def extract_synchronized_windows(eeg_raw, eeg_spatial_data, fnirs_data, events, sfreq_eeg,  # CHANGED
                                  fnirs_events_time, fnirs_events_code):  # CHANGED

    print("Extracting synchronized asymmetric sliding windows...")
    print(f"EEG window: {EEG_WINDOW_LEN}s, step: {EEG_STEP}s")
    print(f"fNIRS window: {FNIRS_WINDOW_LEN}s, step: {FNIRS_STEP}s")

    # NEW (D2): compute ML-anomaly intervals ONCE per subject, on the full
    # continuous cleaned EEG signal, before the trial/window loop below.
    # Each candidate window is checked against these intervals inside the
    # loop (see the epoch-level filter check next to the existing amplitude
    # QC check).
    #
    # NEW (isolation toggle): skipped entirely when ENABLE_D2_EPOCH_FILTER
    # is False -- this computation (Isolation Forest + LOF + Elliptic
    # Envelope on the whole recording) is only useful when D2 is active, so
    # skipping it also saves meaningful per-subject runtime during this
    # isolated D1-only ablation run.
    if ENABLE_D2_EPOCH_FILTER:  # NEW (isolation toggle)
        print("Computing epoch-level ML-anomaly intervals (D2 filter)...")  # NEW (D2)
        ml_anomaly_intervals = compute_ml_anomaly_intervals(eeg_raw, sfreq_eeg)  # NEW (D2)
        print(f"  {len(ml_anomaly_intervals)} merged anomaly interval(s) found "  # NEW (D2)
              f"(overlap threshold={EPOCH_FILTER_OVERLAP_THRESHOLD})")  # NEW (D2)
    else:
        print("D2 epoch-level filter disabled (isolation test) -- skipping ML-anomaly computation")  # NEW (isolation toggle)
        ml_anomaly_intervals = []  # NEW (isolation toggle): unused when disabled, kept for the finally-block cleanup

    # support both marker formats 
    # label_map = {
    #     1: 0, 2: 1, 3: 1, 4: 2, 5: 2,
    #     16: 0, 48: 1, 64: 1, 80: 2, 96: 2
    # }
    #
    # valid_codes = [1, 2, 3, 4, 5, 16, 48, 64, 80, 96]
    # target_events = events[np.isin(events[:, 2], valid_codes)]
    # FIX (Problem 2): the codes above (1-5, 16/48/64/80/96) are STIMULUS-level markers that fire
    # roughly every 2 s within a 40 s trial (per Shin et al. 2018 / Dataset_description_MATLAB.pdf,
    # codes 16/48/64/80/96 correspond to individual target/non-target number presentations, not to
    # the start of a cognitive-load trial). Sliding a window from each of these stimulus markers
    # caused massive, redundant overlap between windows anchored to different stimuli within the same
    # trial, rather than a clean traversal of the full 40 s task period for that trial. The correct
    # trial-onset codes, confirmed directly from the VP001 .mat structure (inspect_data.py output:
    # mrk_nback.event.desc contains 144 etc., and the dataset documentation confirms 112/128/144 =
    # 0-/2-/3-back SESSION start), are used instead: each trial-onset marker anchors exactly one
    # 40 s task period, which is then traversed by the (unchanged) sliding-window loop below.
    label_map = EEG_TRIAL_LABEL_MAP  # NEW: trial-onset codes (112/128/144) -> class label, defined at top of script
    valid_codes = list(EEG_TRIAL_LABEL_MAP.keys())  # NEW
    target_events = events[np.isin(events[:, 2], valid_codes)]  # NEW: now selects one event per ~40s trial instead of one event per ~2s stimulus

    print(f"found {len(target_events)} trial markers")
    if len(target_events) == 0:
        print(f"No trial markers found. Available markers: {np.unique(events[:, 2])}")
        # FIX (trial_ids support): keep the empty-return tuple consistent with the new 5-value
        # return signature (added trial_ids), so callers unpacking 5 values never get a
        # "not enough values to unpack" error on this early-exit path.
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])  # CHANGED: was 3 empty arrays, now 5 (added session_ids, trial_ids placeholders)

    # NEW (CRITICAL SYNCHRONIZATION FIX -- see comment above load_and_preprocess_fnirs()'s return
    # statement for the full diagnostic rationale): build fNIRS's OWN trial-onset time array (in
    # seconds), independently of EEG's clock, from fnirs_events_time/fnirs_events_code (loaded in
    # load_and_preprocess_fnirs()). Diagnostic testing confirmed that, for every subject checked,
    # EEG's trial-class sequence and fNIRS's trial-class sequence match exactly 1:1 in order, so
    # trial i in target_events corresponds to trial i in fnirs_trial_times/fnirs_trial_classes --
    # this is what allows pairing them by simple array position below.
    fnirs_sync_ok = False  # NEW
    fnirs_trial_times = None  # NEW
    if fnirs_events_time is not None and fnirs_events_code is not None:  # NEW
        fnirs_valid_codes = list(FNIRS_TRIAL_LABEL_MAP.keys())  # NEW
        fnirs_mask = np.isin(fnirs_events_code, fnirs_valid_codes)  # NEW
        fnirs_trial_times_ms = np.asarray(fnirs_events_time)[fnirs_mask].astype(float)  # NEW
        fnirs_trial_codes = np.asarray(fnirs_events_code)[fnirs_mask]  # NEW
        fnirs_order = np.argsort(fnirs_trial_times_ms)  # NEW
        fnirs_trial_times = fnirs_trial_times_ms[fnirs_order] / 1000.0  # NEW: seconds, fNIRS's own clock
        fnirs_trial_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in fnirs_trial_codes[fnirs_order]])  # NEW

        eeg_trial_classes = np.array([label_map.get(int(ev[2]), -1) for ev in target_events])  # NEW
        n_common = min(len(fnirs_trial_classes), len(eeg_trial_classes))  # NEW
        if (len(fnirs_trial_times) == len(target_events)  # NEW: same trial count in both devices
                and np.array_equal(fnirs_trial_classes[:n_common], eeg_trial_classes[:n_common])):  # NEW: same class order
            fnirs_sync_ok = True  # NEW
            print(f"fNIRS-own-marker sync check: OK ({len(fnirs_trial_times)} trials, class order matches EEG)")  # NEW
        else:
            print(f"  WARNING: fNIRS own markers do not match EEG 1:1 "  # NEW
                  f"(fNIRS n={len(fnirs_trial_times)}, EEG n={len(target_events)}) -- "  # NEW
                  f"falling back to EEG-clock-derived fNIRS timing for this subject.")  # NEW
    else:
        print("  WARNING: fNIRS event markers unavailable -- falling back to EEG-clock-derived fNIRS timing.")  # NEW

    window_samples_eeg = int(EEG_WINDOW_LEN * sfreq_eeg)
    window_samples_fnirs = int(FNIRS_WINDOW_LEN * SFREQ_FNIRS)

    max_slide_seconds = EEG_EPOCH_TMAX - EEG_WINDOW_LEN
    step_seconds = EEG_STEP

    eeg_windows = []
    fnirs_windows = []
    labels = []
    session_ids = []   # CROSS-SESSION: session number (1,2,3) برای هر window
    trial_ids = []     # NEW (trial_ids fix): trial index (0..N-1, per-subject) for each window -- required by
                        # NEW: Training's trial-level 4:1 val split (run_training reads
                        # NEW: {subject}_trial_ids.npy). Previously this array was never computed or saved,
                        # NEW: which made Training crash with FileNotFoundError the moment it ran.

    for trial_idx, ev in enumerate(target_events):  # CHANGED: enumerate() added to index into fnirs_trial_times
        event_sample_eeg = ev[0]
        raw_code = ev[2]

        label = label_map.get(raw_code, -1)
        if label == -1:
            continue
        # CROSS-SESSION: session_id از ترتیب زمانی trial استنتاج می‌شود
        # هر subject 27 trial دارد (3 session × 9 trial per session)
        # trial_idx 0..8 → session 1، trial_idx 9..17 → session 2، trial_idx 18..26 → session 3
        TRIALS_PER_SESSION = 9
        session_id = trial_idx // TRIALS_PER_SESSION + 1

        # event_time = event_sample_eeg / sfreq_eeg
        # CHANGED (CRITICAL SYNCHRONIZATION FIX): event_time used to be derived purely from EEG's
        # own clock and then reused to index INTO fNIRS's array too, silently assuming a shared
        # t=0 between the two independently-recorded devices. eeg_event_time (below) is still used
        # for the EEG window and for the baseline window (both correctly indexed into EEG-derived
        # arrays / fnirs_data using EEG's own timeline as before -- baseline correction was already
        # working correctly within a session since the drift is BETWEEN sessions, not within one,
        # and baseline windows are always taken from shortly before the SAME trial). fnirs_event_time
        # (used only for the fNIRS window's actual position below) now comes from fNIRS's OWN trial
        # marker when the sync check above passed, eliminating the ~46-93s cross-session drift.
        eeg_event_time = event_sample_eeg / sfreq_eeg  # CHANGED: renamed from event_time for clarity
        if fnirs_sync_ok:  # NEW
            fnirs_event_time = fnirs_trial_times[trial_idx]  # NEW: fNIRS's own clock for this trial
        else:  # NEW
            fnirs_event_time = eeg_event_time  # NEW: fallback -- old (drift-prone) behavior, only used if sync check failed

        # NEW (isolated baseline-correction test): compute this trial's fNIRS baseline ONCE per
        # trial (not per segment/offset), from the continuous fnirs_data signal, using the
        # BASELINE_TMIN/BASELINE_TMAX window (-5 to -2 s before this trial's onset, on the SAME
        # sync-fixed fnirs_event_time anchor used for the segments themselves). Subtracted from
        # every fNIRS segment belonging to this trial below. If the pre-trial baseline window
        # falls before the start of the recording (only possible for a trial very close to time
        # 0), no correction is applied for that trial (fnirs_baseline stays 0) rather than
        # raising an error or using an out-of-bounds/negative index.
        baseline_start_fnirs = int(np.round((fnirs_event_time + BASELINE_TMIN) * SFREQ_FNIRS))  # NEW
        baseline_end_fnirs = int(np.round((fnirs_event_time + BASELINE_TMAX) * SFREQ_FNIRS))    # NEW
        if baseline_start_fnirs >= 0 and baseline_end_fnirs > baseline_start_fnirs:  # NEW
            fnirs_baseline = fnirs_data[baseline_start_fnirs:baseline_end_fnirs].mean(axis=0, keepdims=True)  # NEW: shape (1,16,16,2)
        else:  # NEW
            fnirs_baseline = 0.0  # NEW: not enough pre-trial data (e.g. the very first trial) -- skip correction for this trial

        for offset_sec in np.arange(0.0, max_slide_seconds + 0.001, step_seconds):

            # EEG window
            w_start_eeg = int(event_sample_eeg + np.round(offset_sec * sfreq_eeg))
            w_end_eeg = w_start_eeg + window_samples_eeg

            # fNIRS window 
            # w_start_fnirs = int(np.round((event_time + offset_sec) * SFREQ_FNIRS))
            # w_start_fnirs = int(np.round((fnirs_event_time + offset_sec) * SFREQ_FNIRS))  # CHANGED: uses fNIRS's own clock (fnirs_event_time), not EEG-derived event_time
            # w_end_fnirs = w_start_fnirs + window_samples_fnirs
            #
            # if w_end_eeg > eeg_raw.n_times:
            #     continue
            #
            # if w_end_fnirs > fnirs_data.shape[0]:
            #     continue
            # CHANGED (EXPERIMENTAL -- 11-segment fNIRS windowing): instead of ONE continuous
            # 10s fNIRS window anchored at fnirs_event_time+offset_sec, extract FNIRS_NUM_SEGMENTS=11
            # overlapping 3s segments, each shifted by FNIRS_STEP=1s from the previous one (matching
            # the TSFNet paper's approach, and the same segment structure used in the other expert's
            # extract_tsfnet_windows()). The sync-fix anchor (fnirs_event_time, from fNIRS's OWN
            # trial marker -- completely UNCHANGED from above) is still used as the base time for
            # segment 0; this is the one improvement over the other expert's version, which anchored
            # segments to EEG's clock directly and was therefore subject to the ~46-93s cross-session
            # drift this project already diagnosed and fixed.
            if w_end_eeg > eeg_raw.n_times:  # CHANGED: moved this check up, unchanged logic
                continue

            fnirs_segments = []  # NEW
            valid_segments = True  # NEW
            for seg_idx in range(FNIRS_NUM_SEGMENTS):  # NEW
                seg_offset = offset_sec + seg_idx * FNIRS_STEP  # NEW: each segment shifted by 1s from the previous
                w_start_fnirs_seg = int(np.round((fnirs_event_time + seg_offset) * SFREQ_FNIRS))  # NEW: still uses the sync-fixed anchor
                w_end_fnirs_seg = w_start_fnirs_seg + window_samples_fnirs  # NEW: window_samples_fnirs now = 3s worth of samples (FNIRS_WINDOW_LEN=3.0)

                if w_end_fnirs_seg > fnirs_data.shape[0] or w_start_fnirs_seg < 0:  # NEW
                    valid_segments = False  # NEW
                    break  # NEW

                fnirs_segments.append(fnirs_data[w_start_fnirs_seg:w_end_fnirs_seg] - fnirs_baseline)  # CHANGED: subtract this trial's baseline, isolated from the filter change

            if not valid_segments:  # NEW
                continue  # NEW

            # Extract EEG
            # eeg_slice = eeg_raw.get_data(start=w_start_eeg, stop=w_end_eeg)
            # if np.max(np.abs(eeg_slice)) > 120e-6:
            #     continue
            # CHANGED (EEG spatial mapping): the SAVED EEG window must now be the spatially-mapped
            # (16, 16, 600) grid representation, not the raw (28, 600) channel-space slice. The
            # artifact-rejection amplitude check, however, stays on the raw channel-space slice
            # (eeg_slice_raw) since the 120 uV threshold is only meaningful in electrode voltage
            # units, not on cubic-interpolated grid values.
            eeg_slice_raw = eeg_raw.get_data(start=w_start_eeg, stop=w_end_eeg)  # CHANGED: renamed, QC-only now
            if np.max(np.abs(eeg_slice_raw)) > 120e-6:  # CHANGED: uses eeg_slice_raw
                continue

            # NEW (D2 -- isolated data-level fix): epoch-level ML-anomaly filter.
            # Drops this window if its overlap fraction with a detected
            # ML-anomaly interval (computed once per subject above) is >=
            # EPOCH_FILTER_OVERLAP_THRESHOLD. This removes only the specific
            # windows that fall inside a flagged burst, instead of discarding
            # an entire channel or subject -- preserving usable trials/windows
            # from subjects (e.g. VP001, VP006, VP008) whose artifacts are
            # localized in time rather than continuous.
            #
            # NEW (isolation toggle): gated behind ENABLE_D2_EPOCH_FILTER so
            # D1 (VP004 removal) and D2 (this filter) can be tested in
            # isolation. When False, this check is skipped entirely -- every
            # window that passes the existing amplitude QC check above is
            # kept, exactly as in the pre-D1/D2 baseline.
            if ENABLE_D2_EPOCH_FILTER:  # NEW (isolation toggle)
                w_start_eeg_s = w_start_eeg / sfreq_eeg  # NEW (D2)
                w_end_eeg_s   = w_end_eeg / sfreq_eeg    # NEW (D2)
                if window_overlaps_anomaly(w_start_eeg_s, w_end_eeg_s,           # NEW (D2)
                                           ml_anomaly_intervals,                  # NEW (D2)
                                           EPOCH_FILTER_OVERLAP_THRESHOLD):       # NEW (D2)
                    continue  # NEW (D2)

            # NEW: slice the pre-computed spatially-mapped EEG (n_times, 16, 16) at the SAME sample
            # range as eeg_slice_raw (both index the same continuous recording at sfreq_eeg=200Hz),
            # then transpose (600, 16, 16) -> (16, 16, 600) i.e. grid-H x grid-W x time, per spec.
            eeg_slice_spatial = eeg_spatial_data[w_start_eeg:w_end_eeg]  # NEW: (600, 16, 16)
            eeg_slice = eeg_slice_spatial.transpose(1, 2, 0)             # NEW: (16, 16, 600) -- this is what gets saved

            # Extract fNIRS
            # fnirs_slice = fnirs_data[w_start_fnirs:w_end_fnirs]
            # fnirs_slice = fnirs_slice.transpose(3, 0, 1, 2) 
            # CHANGED (EXPERIMENTAL -- 11-segment fNIRS windowing): stack the 11 already-extracted
            # segments (each (30,16,16,2)) into (11,30,16,16,2), then transpose to put the segment
            # axis first and channel (HbO/HbR) axis last, matching the shape convention used by the
            # other expert's implementation of this same approach: (11, 16, 16, 30, 2).
            fnirs_slice = np.stack(fnirs_segments, axis=0)      # NEW: (11, 30, 16, 16, 2)
            fnirs_slice = fnirs_slice.transpose(0, 2, 3, 1, 4)  # NEW: (11, 16, 16, 30, 2)

            eeg_windows.append(eeg_slice)
            fnirs_windows.append(fnirs_slice)
            labels.append(label)
            session_ids.append(session_id)   # CROSS-SESSION
            trial_ids.append(trial_idx)      # NEW (trial_ids fix): records which trial (0..26) this window came from,
                                              # NEW: needed by Training's trial-level 4:1 split to guarantee that no
                                              # NEW: window from a given trial leaks across train/val.

    eeg_windows = np.array(eeg_windows)
    fnirs_windows = np.array(fnirs_windows)
    labels = np.array(labels)
    session_ids = np.array(session_ids, dtype=np.int32)   # CROSS-SESSION
    trial_ids = np.array(trial_ids, dtype=np.int32)       # NEW (trial_ids fix)

    print(f"Extracted {len(labels)} synchronized windows")
    if len(labels) > 0:
        print(f"Label distribution: {np.bincount(labels)}")

    return eeg_windows, fnirs_windows, labels, session_ids, trial_ids   # CHANGED: trial_ids added (was: ..., session_ids only)



LOG_FILE = os.path.join(OUTPUT_PATH, "processing_log.txt")  # NEW: log file path

def log_message(msg):  # NEW: helper to print + append to log file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")



# MAIN PROCESSING FUNCTION
def process_subject(subject_id):
    t0 = time.time()  # NEW
    log_message(f"START processing {subject_id}")  # NEW
    print(f"Processing {subject_id}")

    # NEW (D1 -- isolated data-level fix): skip subjects with a confirmed
    # chronic, session-independent EEG artifact pattern. See REMOVE_SUBJECTS
    # definition above for the evidence (VP004: artifact score 80-83/100 in
    # all 3 sessions, eeg_artifact_detection_26subj_v3_english.py).
    if subject_id in REMOVE_SUBJECTS:  # NEW (D1)
        log_message(f"SKIPPED {subject_id}: excluded via REMOVE_SUBJECTS (D1)")  # NEW (D1)
        print(f"{subject_id} is in REMOVE_SUBJECTS -- skipping (D1 exclusion)")  # NEW (D1)
        return  # NEW (D1)

    # subject_num = subject_id[2:]
    # subject_path = os.path.join(BASE_DATASET_PATH, subject_num)
    # eeg_folder = os.path.join(subject_path, f"{subject_id}-EEG")
    # FIX: same flat-folder-structure fix as in load_and_preprocess_fnirs() above; EEG data sits
    # directly under BASE_DATASET_PATH\VPxxx-EEG, with no intermediate numeric subfolder.
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")  # NEW: flat folder structure

    if not os.path.exists(eeg_folder):
        log_message(f"SKIPPED {subject_id}: EEG folder not found")  # NEW
        print(f"EEG folder not found: {eeg_folder}")
        return

    try:
        print("\n Loading EEG...")
        # raw_eeg, events, sfreq_eeg = load_eeg_data(eeg_folder)
        raw_eeg, events, sfreq_eeg, eeg_mnt_struct = load_eeg_data(eeg_folder)  # CHANGED: also receive the real EEG mnt_nback.mat struct

        print("\n Loading and preprocessing fNIRS...")
        # fnirs_data = load_and_preprocess_fnirs(subject_id)
        fnirs_data, fnirs_events_time, fnirs_events_code = load_and_preprocess_fnirs(subject_id)  # CHANGED: also receive fNIRS's own trial markers

        # NEW (bug fix): normalization must happen BEFORE window extraction, not after.
        # extract_synchronized_windows() below slices fnirs_windows directly out of fnirs_data,
        # so if fnirs_data is normalized AFTER that call, the windows already saved are un-normalized
        # and the later normalization line has no effect on the saved output (confirmed by
        # VP011's quality report: fNIRS std=0.000367 instead of ~1.0). Moving the normalization
        # here, right after fnirs_data is produced and before any windowing happens, fixes this.
        print("\n Normalizing fNIRS (subject-level, before windowing)...")  # NEW
        fnirs_data = (fnirs_data - fnirs_data.mean(axis=0, keepdims=True)) / (fnirs_data.std(axis=0, keepdims=True) + 1e-8)  # NEW: moved here from below

        print("\n Preprocessing EEG...")
        raw_eeg_clean = preprocess_eeg(raw_eeg, events)

        # NEW (save cleaned continuous EEG for QC): raw_eeg_clean is the final, fully cleaned
        # channel-space signal (notch + 1-40Hz bandpass + bad-channel interpolation + average
        # reference + ICA artifact removal + VEOG/HEOG/TP9 dropped), saved here IMMEDIATELY after
        # preprocess_eeg() returns -- i.e. before spatial mapping / normalization / windowing
        # convert it into the 16x16 grid tensor representation below, at which point it is no
        # longer available in this channel x time form as a regular mne.Raw object. This .fif file
        # is what load_clean_from_pipeline() in eeg_preprocessing_qc.py looks for first, before
        # falling back to a lossy approximation reconstructed from the windowed data.
        sub_out_dir = os.path.join(OUTPUT_PATH, subject_id)  # NEW: defined early so the .fif can be saved now (re-defined harmlessly later via os.makedirs(exist_ok=True) before the .npy saves)
        os.makedirs(sub_out_dir, exist_ok=True)  # NEW
        raw_eeg_clean.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_clean_raw.fif"), overwrite=True)  # NEW

        # NEW (EEG per-subject normalization fix support): compute per-channel mean/std from the
        # FULL cleaned, continuous EEG signal (after filtering/CAR/ICA) BEFORE any windowing happens.
        # FIX (per-window vs per-subject normalization): the previous EEG normalization (further
        # below, in the "Normalizing..." section) computed a separate z-score for EACH extracted
        # window independently (axis=-1 over the window's 600 time samples). With a window as short
        # as 3 s, this local z-scoring is noisy/unstable and, more importantly, completely erases any
        # systematic difference in absolute EEG amplitude/power between subjects or between cognitive
        # load levels within a subject -- the model can then only see the RELATIVE waveform shape
        # inside each isolated 3 s window, which is a much weaker and more subject-specific signal.
        # This is very likely a major contributor to the severe overfitting observed during model
        # training (train acc ~96% vs val/test acc near chance level): the model latches onto
        # subject-specific residual noise patterns within each locally-renormalized window instead of
        # learning generalizable cognitive-load-related features. Computing mean/std once per subject
        # over the full continuous recording (mirroring the fNIRS per-subject normalization fix above)
        # and applying those FIXED per-channel statistics to every window extracted from that subject
        # preserves the relative amplitude structure across windows and across load levels, which
        # should generalize better to unseen subjects.
        eeg_full_data = raw_eeg_clean.get_data()  # NEW: shape (n_channels, n_times), fully preprocessed continuous signal
        eeg_subject_mean = eeg_full_data.mean(axis=-1, keepdims=True)  # NEW: shape (n_channels, 1)
        eeg_subject_std = eeg_full_data.std(axis=-1, keepdims=True) + 1e-8  # NEW: shape (n_channels, 1)
        print(f"\n EEG per-subject normalization stats computed: "  # NEW
              f"mean range [{eeg_subject_mean.min():.2e}, {eeg_subject_mean.max():.2e}], "  # NEW
              f"std range [{eeg_subject_std.min():.2e}, {eeg_subject_std.max():.2e}]")  # NEW

        # NEW (EEG spatial mapping, TSFNet paper Sec. 2.3): project the EEG electrodes onto a 16x16
        # spatial grid via azimuthal equidistant projection, exactly as already done for fNIRS
        # (spatial_mapping() + mnt), instead of leaving EEG as raw (channel x time). This is applied
        # ONCE to the full continuous, already-cleaned recording (not per-window) for the same
        # efficiency reason fNIRS is spatially mapped once in load_and_preprocess_fnirs() before
        # windowing: with 66% window overlap, per-window mapping would re-run cubic griddata
        # interpolation on largely the same timepoints hundreds of times over.
        print("\n Spatial mapping EEG to 16x16...")  # NEW
        # eeg_mnt = get_eeg_mnt(raw_eeg_clean)  # electrode positions in raw_eeg_clean.ch_names order (post channel-drop)
        eeg_mnt = get_eeg_mnt(raw_eeg_clean, eeg_mnt_struct)  # CHANGED: uses REAL mnt_nback.mat positions, matched by clab name, not MNE's guessed standard_1005
        eeg_spatial_full = spatial_mapping(eeg_full_data.T, eeg_mnt)  # NEW: (n_channels,n_times).T -> (n_times, n_channels) in, (n_times, 16, 16) out
        print(f"EEG spatially mapped: {eeg_spatial_full.shape}")  # NEW

        # NEW: per-subject normalization of the spatially-mapped EEG, applied to the FULL continuous
        # grid signal before windowing -- mirrors exactly how fNIRS is normalized in this same
        # function (fnirs_data = (fnirs_data - mean)/std over axis=0, i.e. per grid-cell across
        # time). This supersedes the old per-channel normalization (eeg_subject_mean/eeg_subject_std
        # above), which no longer applies now that EEG windows are (16,16,600) grid data rather than
        # (28,600) channel data; eeg_subject_mean/eeg_subject_std are kept above only for the
        # informational print (channel-space signal scale), not used for normalization below anymore.
        eeg_spatial_full = (eeg_spatial_full - eeg_spatial_full.mean(axis=0, keepdims=True)) / (  # NEW
            eeg_spatial_full.std(axis=0, keepdims=True) + 1e-8)  # NEW

        # NEW (user request): finalize the EEG before/after QC plots (now two all-channel figures)
        # for this specific subject by renaming the temp files produced inside preprocess_eeg()
        # (which has no access to subject_id).
        try:  # NEW
            tmp_qc_before = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc", "last_eeg_before_allchannels.png")  # NEW
            tmp_qc_after = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc", "last_eeg_after_allchannels.png")    # NEW
            tmp_qc_before_zoom = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc", "last_eeg_before_allchannels_zoom10_30.png")  # NEW
            tmp_qc_after_zoom = os.path.join(OUTPUT_PATH, "_tmp_eeg_qc", "last_eeg_after_allchannels_zoom10_30.png")    # NEW
            if os.path.exists(tmp_qc_before) and os.path.exists(tmp_qc_after):  # NEW
                final_qc_dir = os.path.join(OUTPUT_PATH, subject_id)  # NEW
                os.makedirs(final_qc_dir, exist_ok=True)  # NEW
                final_qc_before = os.path.join(final_qc_dir, f"{subject_id}_eeg_before_allchannels.png")  # NEW
                final_qc_after = os.path.join(final_qc_dir, f"{subject_id}_eeg_after_allchannels.png")    # NEW
                os.replace(tmp_qc_before, final_qc_before)  # NEW
                os.replace(tmp_qc_after, final_qc_after)    # NEW
                if os.path.exists(tmp_qc_before_zoom) and os.path.exists(tmp_qc_after_zoom):  # NEW
                    final_qc_before_zoom = os.path.join(final_qc_dir, f"{subject_id}_eeg_before_allchannels_zoom10_30.png")  # NEW
                    final_qc_after_zoom = os.path.join(final_qc_dir, f"{subject_id}_eeg_after_allchannels_zoom10_30.png")    # NEW
                    os.replace(tmp_qc_before_zoom, final_qc_before_zoom)  # NEW
                    os.replace(tmp_qc_after_zoom, final_qc_after_zoom)    # NEW
                print(f"Saved EEG before/after (all-channels) QC plots for {subject_id}")  # NEW
        except Exception as e:  # NEW
            print(f"Could not finalize EEG QC plot for {subject_id}: {e}")  # NEW

        print("\n Extracting synchronized asymmetric windows...")
        # eeg_windows, fnirs_windows, labels = extract_synchronized_windows(
        #     raw_eeg_clean, fnirs_data, events, sfreq_eeg
        # )
        # CHANGED: pass eeg_spatial_full (the pre-computed 16x16-grid-mapped continuous EEG) in
        # addition to raw_eeg_clean (still needed inside for the channel-space QC amplitude check).
        # eeg_windows, fnirs_windows, labels = extract_synchronized_windows(
        #     raw_eeg_clean, eeg_spatial_full, fnirs_data, events, sfreq_eeg
        # )
        # CHANGED (CRITICAL SYNCHRONIZATION FIX): also pass fnirs_events_time/fnirs_events_code so
        # fNIRS windows are anchored to fNIRS's OWN trial markers, not EEG-clock-derived timing.
        # CHANGED (trial_ids fix): also receive trial_ids, now returned by extract_synchronized_windows.
        eeg_windows, fnirs_windows, labels, session_ids, trial_ids = extract_synchronized_windows(   # CHANGED: trial_ids added
            raw_eeg_clean, eeg_spatial_full, fnirs_data, events, sfreq_eeg,
            fnirs_events_time, fnirs_events_code
        )

        if len(labels) == 0:
            log_message(f"WARNING {subject_id}: no windows extracted")  # NEW
            print(f"No windows extracted for {subject_id}")
            return

        print("\n Normalizing...")
        # EEG: (n_windows, n_channels, n_times) 
        # eeg_windows = (eeg_windows - eeg_windows.mean(axis=-1, keepdims=True)) / (eeg_windows.std(axis=-1, keepdims=True) + 1e-8)
        # FIX (per-window -> per-subject EEG normalization): the line above computed a fresh z-score
        # for every single window independently (per-window mean/std over its own 600 time samples).
        # As explained in the comment above where eeg_subject_mean/eeg_subject_std are computed, this
        # erases systematic amplitude differences between cognitive-load levels and between subjects,
        # which is very likely contributing to the poor generalization (near-chance val/test accuracy)
        # observed during model training. Applying the FIXED per-subject statistics computed once
        # from the full continuous recording (broadcast across all windows and all time samples via
        # reshape to (1, n_channels, 1)) fixes this while still correcting for each subject's overall
        # signal scale, mirroring the per-subject normalization already used for fNIRS.
        # eeg_mean_broadcast = eeg_subject_mean.reshape(1, -1, 1)  # (1, n_channels, 1) for broadcasting over (N, n_channels, n_times)
        # eeg_std_broadcast = eeg_subject_std.reshape(1, -1, 1)
        # eeg_windows = (eeg_windows - eeg_mean_broadcast) / eeg_std_broadcast
        # CHANGED (EEG spatial mapping): eeg_windows is now (n_windows, 16, 16, 600) grid data, not
        # (n_windows, 28, 600) channel data, so the per-channel normalization above no longer applies
        # and is removed. EEG normalization now happens EARLIER, on the full continuous spatially-
        # mapped signal (eeg_spatial_full) before windowing -- see the "NEW: per-subject
        # normalization of the spatially-mapped EEG" block above, right after spatial_mapping() is
        # called -- exactly mirroring how fNIRS is normalized in grid space before windowing. No
        # further EEG normalization is needed here; eeg_windows already carries normalized values.

        # fNIRS: (n_windows, 2, 100, 16, 16)
        # fnirs_windows = (fnirs_windows - fnirs_windows.mean(axis=2, keepdims=True)) / (fnirs_windows.std(axis=2, keepdims=True) + 1e-8)
        # FIX (Problem 3): ...
        # fnirs_data = (fnirs_data - fnirs_data.mean(axis=0, keepdims=True)) / (fnirs_data.std(axis=0, keepdims=True) + 1e-8)
        # BUG FIX (moved): this line was dead code -- fnirs_windows had already been sliced from the
        # un-normalized fnirs_data by extract_synchronized_windows() before this point ran. The actual
        # normalization now happens earlier, right after load_and_preprocess_fnirs() returns and
        # before extract_synchronized_windows() is called. See that block above.

        print("\n Saving...")
        sub_out_dir = os.path.join(OUTPUT_PATH, subject_id)
        os.makedirs(sub_out_dir, exist_ok=True)

        # np.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_windows.npy"), eeg_windows)
        # np.save(os.path.join(sub_out_dir, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)
        # np.save(os.path.join(sub_out_dir, f"{subject_id}_labels.npy"), labels)
        # NEW (user request): for the first verification run, only save outputs for the very first
        # subject so correctness can be checked before running the full 26-subject batch. The save
        # calls below are commented out for all subjects except VP001; uncomment the three np.save
        # lines below (and remove/relax this condition) once the pipeline has been verified.
#        #if subject_id == "VP001":  # NEW: save outputs only for the first subject during verification
        np.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_windows.npy"), eeg_windows)    # NEW
        np.save(os.path.join(sub_out_dir, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)  # NEW
        np.save(os.path.join(sub_out_dir, f"{subject_id}_labels.npy"), labels)               # NEW
        np.save(os.path.join(sub_out_dir, f"{subject_id}_session_ids.npy"), session_ids)   # CROSS-SESSION
        np.save(os.path.join(sub_out_dir, f"{subject_id}_trial_ids.npy"), trial_ids)       # NEW (trial_ids fix): required by Training's trial-level 4:1 split
        # else:                                                                                    # NEW
        #     # Saving suppressed for subjects other than VP001 during this verification run.        # NEW
        #     pass                                                                                  # NEW

        # NEW (adopted from reviewing a third-party implementation, see chat history): a small
        # per-subject metadata.json sidecar file, so the exact parameters/shapes a given
        # preprocessing run used can be checked at a glance without loading the (large) .npy
        # arrays -- useful given this project has gone through many preprocessing variants
        # (Run A-H) and it has repeatedly been unclear afterward exactly which run produced a
        # given set of output files.
        metadata = {  # NEW
            'subject_id': subject_id,  # NEW
            'eeg_shape': list(eeg_windows.shape),  # NEW
            'fnirs_shape': list(fnirs_windows.shape),  # NEW
            'labels_distribution': np.bincount(labels).tolist(),  # NEW
            'eeg_window_len': EEG_WINDOW_LEN,  # NEW
            'eeg_step': EEG_STEP,  # NEW
            'eeg_epoch_tmax': EEG_EPOCH_TMAX,  # NEW: important given this exact parameter caused a major bug in the reviewed third-party code
            'fnirs_num_segments': FNIRS_NUM_SEGMENTS,  # NEW
            'fnirs_window_len': FNIRS_WINDOW_LEN,  # NEW
            'fnirs_step': FNIRS_STEP,  # NEW
            'sfreq_eeg': SFREQ_EEG,  # NEW
            'sfreq_fnirs': SFREQ_FNIRS,  # NEW
            'beer_lambert_applied': False,  # NEW: this pipeline uses oxy_raw/deoxy_raw directly (Run F fix), not calculate_hbo_hbr()
            'fnirs_sync_fix_applied': True,  # NEW: fNIRS windows anchored to fNIRS's own trial markers, not EEG's clock
            'normalization': 'per-subject z-score (EEG grid + fNIRS grid, before windowing)',  # NEW
            'trial_ids_saved': True,  # NEW (trial_ids fix): flags that this run produced {subject_id}_trial_ids.npy
        }
        with open(os.path.join(sub_out_dir, f"{subject_id}_metadata.json"), 'w') as f:  # NEW
            json.dump(metadata, f, indent=2)  # NEW

        print(f"\n {subject_id} completed!")
        print(f"EEG windows: {eeg_windows.shape}")
        print(f"fNIRS windows: {fnirs_windows.shape}")
        print(f"Labels: {np.bincount(labels)}")
        elapsed = (time.time() - t0) / 60  # NEW
        log_message(f"DONE {subject_id} in {elapsed:.1f} min | EEG: {eeg_windows.shape} | fNIRS: {fnirs_windows.shape} | Labels: {np.bincount(labels)}")  # NEW

    except Exception as e:
        elapsed = (time.time() - t0) / 60  # NEW
        log_message(f"ERROR processing {subject_id} after {elapsed:.1f} min: {str(e)}")  # NEW
        print(f"Error processing {subject_id}: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # NEW (memory fix, part 2): explicit cleanup of the large per-subject
        # arrays on EVERY exit path (success or exception). The earlier fix
        # inside compute_ml_anomaly_intervals (D2) only addressed memory
        # leaked by that one function; it did not address memory retained by
        # process_subject's own large local arrays (oxy_raw/deoxy_raw,
        # eeg_full_data, eeg_spatial_full, eeg_windows, fnirs_windows, etc.),
        # which is why a MemoryError could still occur late in the 26-subject
        # loop (observed at VP026, the very last subject) even after the D2
        # fix.
        #
        # IMPORTANT: `del locals()[name]` does NOT reliably delete a local
        # variable binding in CPython -- locals() inside a function returns a
        # snapshot dict, not a live view, so mutating it silently does
        # nothing. Each variable below is therefore deleted with an explicit
        # `del <name>` statement, individually guarded with try/except
        # NameError since not all of them exist on every code path (e.g. an
        # early exception may leave later-assigned variables undefined).
        try:    del oxy_raw            # NEW
        except NameError: pass         # NEW
        try:    del deoxy_raw          # NEW
        except NameError: pass         # NEW
        try:    del fnirs_data         # NEW
        except NameError: pass         # NEW
        try:    del raw_eeg            # NEW
        except NameError: pass         # NEW
        try:    del raw_eeg_clean      # NEW
        except NameError: pass         # NEW
        try:    del eeg_full_data      # NEW
        except NameError: pass         # NEW
        try:    del eeg_spatial_full   # NEW
        except NameError: pass         # NEW
        try:    del eeg_windows        # NEW
        except NameError: pass         # NEW
        try:    del fnirs_windows      # NEW
        except NameError: pass         # NEW
        try:    del ml_anomaly_intervals  # NEW
        except NameError: pass            # NEW
        gc.collect()  # NEW: force immediate release back to the OS

# RUN
# ══════════════════════════════════════════════════════════════════════════
# NEW (parallel execution): memory-safe multiprocessing across subjects
#
# Subjects are fully independent (each loads its own raw files, processes,
# and saves its own outputs), so this is embarrassingly parallel. However,
# using ALL physical CPU cores at once would exceed available RAM (observed
# peak per-subject memory: ~1.5-3 GB from earlier MemoryError crashes), so
# the worker count is calculated from available RAM, not just CPU count.
# ══════════════════════════════════════════════════════════════════════════

RAM_SAFETY_MARGIN_GB = 4.0             # NEW: always leave this much RAM free for the OS
ESTIMATED_PEAK_GB_PER_SUBJECT = 3.0    # NEW: conservative estimate from observed crashes


def compute_safe_worker_count() -> int:  # NEW (parallel execution)
    """
    Returns a worker count that is safe for BOTH available RAM and physical
    CPU core count -- whichever is more restrictive. Recalculated fresh each
    run since available RAM changes depending on what else is running on
    the machine.
    """
    physical_cores = psutil.cpu_count(logical=False)
    available_gb   = psutil.virtual_memory().available / (1024 ** 3)
    usable_gb      = max(0.0, available_gb - RAM_SAFETY_MARGIN_GB)
    mem_limited    = max(1, int(usable_gb // ESTIMATED_PEAK_GB_PER_SUBJECT))
    safe_workers   = min(physical_cores, mem_limited)

    print("=" * 60)
    print("  Parallel preprocessing -- worker count calculation")
    print("=" * 60)
    print(f"  Physical cores available    : {physical_cores}")
    print(f"  RAM available right now     : {available_gb:.1f} GB")
    print(f"  Safety margin reserved      : {RAM_SAFETY_MARGIN_GB:.1f} GB")
    print(f"  Usable RAM for workers      : {usable_gb:.1f} GB")
    print(f"  Estimated peak RAM/subject  : {ESTIMATED_PEAK_GB_PER_SUBJECT:.1f} GB")
    print(f"  Memory-limited worker count : {mem_limited}")
    print(f"  CPU-limited worker count    : {physical_cores}")
    print(f"  ==> Using {safe_workers} parallel workers")
    print("=" * 60)
    return safe_workers


def _run_one_subject_worker(subject_id: str) -> dict:  # NEW (parallel execution)
    """
    Runs inside a separate OS process (separate memory space). Wraps the
    existing, already memory-fixed process_subject() (with its finally-block
    cleanup) and reports timing + peak memory for monitoring. Must be a
    module-level function (not nested/lambda) so it can be pickled and sent
    to worker processes under Windows' spawn start method.
    """
    t0 = time.time()
    proc = psutil.Process(os.getpid())
    try:
        process_subject(subject_id)
        status, error_msg = "OK", None
    except Exception as e:
        status, error_msg = "ERROR", str(e)
    finally:
        gc.collect()
    return {
        "subject": subject_id,
        "status": status,
        "error": error_msg,
        "elapsed_min": round((time.time() - t0) / 60, 2),
        "peak_rss_gb": round(proc.memory_info().rss / (1024 ** 3), 2),
    }


def run_parallel(subject_ids: list, max_workers: int = None):  # NEW (parallel execution)
    """
    Dispatches process_subject() across subject_ids in parallel, using a
    memory-safe worker count computed automatically unless max_workers is
    given explicitly.
    """
    if max_workers is None:
        max_workers = compute_safe_worker_count()

    print(f"\nStarting parallel preprocessing of {len(subject_ids)} subjects "
          f"with {max_workers} workers...\n")

    results = []
    t_total0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one_subject_worker, sid): sid for sid in subject_ids}
        for future in as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
            except Exception as e:
                # a crash severe enough to kill the worker process itself
                result = {"subject": sid, "status": "WORKER_CRASHED", "error": str(e),
                          "elapsed_min": None, "peak_rss_gb": None}
            results.append(result)
            tag = "[OK]" if result["status"] == "OK" else "[FAIL]"
            print(f"  {tag} {result['subject']:8s} status={result['status']:14s} "
                  f"time={result.get('elapsed_min', '?')} min "
                  f"peak_rss={result.get('peak_rss_gb', '?')} GB")

    total_elapsed = (time.time() - t_total0) / 60
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"\n{'='*60}")
    print(f"  Done -- {n_ok}/{len(results)} subjects succeeded, "
          f"total time: {total_elapsed:.1f} min")
    print(f"{'='*60}")

    n_fail = len(results) - n_ok
    if n_fail > 0:
        print("\nFailed subjects:")
        for r in results:
            if r["status"] != "OK":
                print(f"  {r['subject']}: {r['status']} -- {r['error']}")
        print("\nRe-run just the failed subjects (sequentially, via process_subject(),")
        print("or with fewer workers) after checking the error messages above.")

    return results


if __name__ == "__main__":
    total_t0 = time.time()  # NEW
    log_message("===== PIPELINE STARTED =====")  # NEW
    print("MULTIMODAL PREPROCESSING PIPELINE")
    print(f"EEG window: {EEG_WINDOW_LEN}s, step: {EEG_STEP}s")
    print(f"fNIRS window: {FNIRS_WINDOW_LEN}s, step: {FNIRS_STEP}s")
    print(f"fNIRS epoch: {FNIRS_EPOCH_TMAX}s")
    print(f"fNIRS sampling rate: {SFREQ_FNIRS} Hz")

    # NEW (parallel execution): replaces the previous sequential for-loop.
    # ALL_SUBJECTS already excludes REMOVE_SUBJECTS via the D1 check inside
    # process_subject(), so VP004 is still skipped automatically, just like
    # in the sequential version.
    #
    # To run only a subset (e.g. for verification), pass an explicit list:
    #   run_parallel(["VP001"])
    # To force a specific worker count instead of the automatic RAM-based
    # calculation:
    #   run_parallel(ALL_SUBJECTS, max_workers=4)
    ALL_SUBJECT_IDS = [f"VP{i:03d}" for i in range(1, 27)]  # NEW
    run_parallel(ALL_SUBJECT_IDS)  # NEW

    print("ALL THE SELECTED SUBJECTS PROCESSED")


# In[ ]:




