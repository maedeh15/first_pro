# ============================================================================
# TSFNet-EXACT PREPROCESSING PIPELINE — n-back dataset (Shin et al., 2018)
# Reproduces Section 2.2 (Data Preprocessing) and Section 2.3 (3D Tensor
# Transformation) of:
#   Zhang, Y.; Yin, B.; Yuan, X. "TSFNet: Temporal-Spatial Fusion Network for
#   Hybrid Brain-Computer Interface." Sensors 2025, 25, 6111.
#
# No official code release for this paper could be found on GitHub or
# elsewhere (checked: github.com search, Sensors/MDPI page, ResearchGate,
# Semantic Scholar). This script is therefore a from-scratch, paper-faithful
# re-implementation of the preprocessing pipeline only (the model/architecture
# will be implemented in the next step, per your request).
#
# WHAT IS "PAPER-EXACT" HERE (Sec. 2.2 / 2.3), as opposed to the exploratory
# run-script you sent me (Preprocessing_v2_run_a1b_10s.ipynb), which
# deliberately deviates from the paper in several places for experimentation:
#   EEG : 6th-order zero-phase Butterworth bandpass, 0.5-50 Hz -> CAR ->
#         ICA ocular-artifact removal using the recorded EOG channels.
#   fNIRS: modified Beer-Lambert law -> HbO/HbR -> Butterworth bandpass,
#         0.01-0.1 Hz -> baseline correction (mean of -5..-2 s pre-trial).
#   Windows: EEG 3 s sliding window, step 1 s, spanning -2..10 s relative to
#         trial onset (instruction + first part of task) -> X_eeg in
#         R^(16x16x600) after azimuthal-equidistant projection + cubic
#         interpolation to a 16x16 grid.
#         fNIRS: matching 3 s segments, step 1 s; each EEG window is paired
#         with the CURRENT + subsequent 10 fNIRS segments (11 total) to
#         absorb the hemodynamic delay -> X_fnirs in R^(11x16x16x30x2)
#         after the same spatial-mapping method applied to the fNIRS
#         physiological (source-detector) channel positions.
#
# Engineering details that are NOT stated in the paper but are necessary to
# make ANY implementation work on the real dataset (.mat parsing conventions,
# EEG/fNIRS cross-device clock synchronization via each device's own trial
# markers, vectorized cubic-spline spatial mapping, bad-channel handling)
# are reused/adapted from the debugged logic already present in your
# uploaded reference notebook, since that groundwork is independent of which
# preprocessing PARAMETERS are used.
#
# All outputs your reference pipeline saves are reproduced here too:
#   {subject}_eeg_windows.npy      (N, 16, 16, 600)               float32
#   {subject}_fnirs_windows.npy    (N, 11, 16, 16, 30, 2)         float32
#   {subject}_labels.npy           (N,)  0=0-back, 1=2-back, 2=3-back
#   {subject}_session_ids.npy      (N,)  1/2/3
#   {subject}_trial_ids.npy        (N,)  0..n_trials-1 (per subject)
#   {subject}_metadata.json        run configuration + shapes
#   {subject}_eeg_before_after.png / {subject}_fnirs_before_after.png  (QC)
#   preprocessing_log.txt          global run log
# ============================================================================

import os
# ----------------------------------------------------------------------------
# MUST be set BEFORE numpy/scipy/mne are imported: pins each worker process's
# BLAS/OpenMP thread pool to 1 thread. Without this, when N_WORKERS subject
# processes run in parallel (see PARALLEL RUN section near the bottom), every
# process ALSO tries to multithread its own numpy/scipy calls internally,
# causing N_WORKERS x N_THREADS-way oversubscription of your 14 physical
# cores -- this is the single most common reason "parallel" preprocessing
# ends up SLOWER than serial. Pinning each worker to 1 BLAS thread and
# parallelizing across SUBJECTS instead (embarrassingly parallel: subjects
# are fully independent) gives the real speedup.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import time
import json
import datetime
import warnings
import multiprocessing as mp

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import mne_icalabel

warnings.filterwarnings("ignore")

# ============================================================================
# PARALLELIZATION (tuned for your machine: 14 physical / 20 logical cores,
# i7-12700H, 34.1 GB RAM / ~20.5 GB available, RTX 3070 8.6GB unused here)
# ============================================================================
# Preprocessing is embarrassingly parallel across SUBJECTS (each subject's
# EEG/fNIRS load -> filter -> ICA -> spatial-mapping -> windowing chain is
# fully independent of every other subject's). This is the only kind of
# parallelism used here -- NOT multithreading inside a single subject's
# pipeline, which is what the OMP/MKL/OPENBLAS pins above intentionally
# disable per-worker.
#
# Sizing N_WORKERS: 14 physical cores -> ICA (the CPU-heaviest single step,
# ~picard/infomax on 30 channels) scales roughly linearly with worker count
# up to core count, so more workers = faster wall-clock, UP TO the point RAM
# runs out. Rough per-worker peak memory for this pipeline (dominated by
# holding the full-recording spatially-mapped EEG grid, ~(n_times,16,16)
# float32, n_times up to ~450,000 samples at 200Hz for a ~35-40 min raw
# recording) is ~1.5-2.5 GB. With ~20.5 GB currently available:
#   20.5 GB / ~2 GB per worker ≈ 10 workers safely, with headroom left for
#   Windows/IDE/other processes. Using all 14 physical cores risks swapping
#   (you only have 14.5 GB page file, and swap is already at 0.8 GB used).
# N_WORKERS=10 is therefore a safe default; drop to 6-8 if you see swapping
# (Task Manager -> Performance -> Memory) or raise toward 12-14 if you close
# other RAM-heavy apps first and watch memory stay comfortably under ~90%.
N_WORKERS = 10


# ============================================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"            # folder holding VPxxx-EEG / VPxxx-NIRS
OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"

# NOTE (confirmed from Dataset_description_MATLAB.pdf / Shin et al. 2018, Sci Data): 26 subjects,
# 3 sessions/subject, 9 trials/session (3 repeats x 0-/2-/3-back, counterbalanced order) -> 27
# trial-onset markers/subject total. VERIFIED directly against VP001's real mrk_nback.mat (27
# events, codes 7/8/9, in three clean groups of 9 separated by a ~148s inter-session gap).
# Each series = 2s instruction + 40s task + 20s rest. EEG: 28 real scalp electrodes (per official
# doc, the cap has 30 incl. TP9/TP10, but CONFIRMED via direct inspection of VP001's real
# cnt_nback.mat that TP9/TP10 are NOT recorded as separate channels -- cnt.clab has 28 scalp
# electrodes + HEOG + VEOG = 30 total), recorded at 1000Hz and shipped in cnt_nback.mat already
# downsampled to 200Hz, UNFILTERED, values in uV (confirmed range ~[-866, 1657]). fNIRS: 16
# sources x 16 detectors -> 36 physiological channels, shipped already downsampled to 10Hz with
# oxy/deoxy already Beer-Lambert converted to HbO/HbR concentration changes (confirmed:
# cnt.oxy.yUnit='mmol/L'; see DATASET_ALREADY_BEER_LAMBERT below), also otherwise unfiltered.
SUBJECTS = list(range(1, 27))  # VP001..VP026 (Shin et al. 2018 n-back cohort)

# --- EEG filtering (paper Sec. 2.2) ----------------------------------------
EEG_BP_LOW = 0.5
EEG_BP_HIGH = 50.0
EEG_BP_ORDER = 6          # 6th-order zero-phase Butterworth
EEG_NOTCH_FREQ = 50.0     # mains notch is not mentioned in the paper text but is standard
                          # practice for this dataset's recording site; kept OFF by default
                          # to stay literal to the paper. Set APPLY_NOTCH=True to enable.
APPLY_NOTCH = False

# --- fNIRS filtering (paper Sec. 2.2) ---------------------------------------
FNIRS_BP_LOW = 0.01
FNIRS_BP_HIGH = 0.1
FNIRS_BP_ORDER = 3
BASELINE_TMIN = -5.0       # seconds before trial onset
BASELINE_TMAX = -2.0
# Shin et al. (2018)'s released .mat files already contain oxy/deoxy as
# hemoglobin CONCENTRATION CHANGES (the modified Beer-Lambert law has already
# been applied by the dataset authors, matching what the reference notebook
# found empirically). Per the paper's Sec. 2.2 wording, the Beer-Lambert step
# is still listed here explicitly and is applied IF the raw arrays look like
# optical densities rather than concentrations; set this flag manually if you
# know your copy of the dataset is not yet Beer-Lambert converted.
DATASET_ALREADY_BEER_LAMBERT = True

# --- Sampling rates ----------------------------------------------------------
SFREQ_EEG = 200.0
SFREQ_FNIRS = 10.0        # after resampling (matches 3 s segment == 30 samples, Table in Sec. 2.3)

# --- Windowing (paper Sec. 2.3) ---------------------------------------------
WINDOW_LEN = 3.0           # seconds, both modalities
WINDOW_STEP = 1.0          # seconds, both modalities
SLIDE_TMIN = -2.0          # seconds relative to trial onset (instruction period)
SLIDE_TMAX = 10.0          # seconds relative to trial onset (early task period)
FNIRS_NUM_SEGMENTS = 11    # current + subsequent 10 segments, absorbs hemodynamic delay

GRID_SIZE = 16             # 16x16 spatial grid for both modalities

# --- n-back trial-level marker codes (Shin et al. 2018 documentation) -------
EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}     # 0-back / 2-back / 3-back (EEG session/series onset)
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}         # matching fNIRS session/series onset codes
# VERIFIED against VP001's real mrk_nback.mat (ground truth, not just the paper's prose):
# the actual file contains 27 trial-onset markers per subject (event.desc has 27 entries, codes
# 7/8/9), split into three clean groups of 9 by an inter-group time gap of ~148s (vs. ~69.5s
# between markers WITHIN a group) -- i.e. 9 trials/session x 3 sessions = 27 trials/subject, each
# session containing 3 repeats of each of the 3 n-back conditions. This directly contradicts a
# literal reading of the Sci Data paper's prose ("Nine series ... for each participant"), which
# is ambiguous/misleading here -- "series" there evidently refers to a group-of-3 unit (one
# instance of each condition), not a single trial. Ground truth from the actual .mat file wins
# over the ambiguous prose. (This also matches the exploratory reference notebook's original
# TRIALS_PER_SESSION=9 assumption, which an earlier revision of this script incorrectly "fixed"
# to 3 based on a literal, mistaken reading of the paper text alone -- reverted here.)
TRIALS_PER_SESSION = 9                             # 9 trials/session x 3 sessions = 27 trials/subject

# --- Official electrode/optode 3D positions (Cartesian), per Shin et al. 2018 dataset
# documentation (Dataset_description_MATLAB.pdf). Used as the AUTHORITATIVE position source for
# spatial_mapping() (azimuthal-equidistant projection), since they are fixed across all subjects
# and come directly from the dataset's own official description rather than being re-derived from
# a per-subject mnt_nback.mat parse (whose exact field layout/order is easy to get subtly wrong).
# mnt_nback.mat is still read (get_eeg_positions/get_fnirs_positions) and used only as a fallback
# for any channel name not found in these tables.
EEG_POSITIONS_3D = {
    "Fp1": (-0.3090, 0.9511, 0.0001), "Fp2": (0.3091, 0.9511, 0.0000),
    "AFF5h": (-0.5417, 0.7777, 0.3163), "AFF6h": (0.5417, 0.7777, 0.3163),
    "AFz": (0.0000, 0.9230, 0.3824),
    "F1": (-0.2888, 0.6979, 0.6542), "F2": (0.2888, 0.6979, 0.6542),
    "FC1": (-0.3581, 0.3770, 0.8532), "FC2": (0.3581, 0.3770, 0.8532),
    "FC5": (-0.8709, 0.3373, 0.3549), "FC6": (0.8709, 0.3373, 0.3549),
    "Cz": (0.0000, 0.0002, 1.0000),
    "C3": (-0.7066, 0.0001, 0.7066), "C4": (0.7066, 0.0001, 0.7066),
    "T7": (-1.0000, 0.0000, 0.0000), "T8": (1.0000, 0.0000, 0.0000),
    "CP1": (-0.3580, -0.3767, 0.8534), "CP2": (0.3580, -0.3767, 0.8534),
    "CP5": (-0.8712, -0.3372, 0.3552), "CP6": (0.8712, -0.3372, 0.3552),
    "Pz": (0.0000, -0.7063, 0.7065),
    "P3": (-0.5401, -0.6724, 0.5045), "P4": (0.5401, -0.6724, 0.5045),
    "P7": (-0.8090, -0.5878, -0.0001), "P8": (0.8090, -0.5878, -0.0001),
    "POz": (0.0000, -0.9230, 0.3824),
    "O1": (-0.3090, -0.9511, 0.0000), "O2": (0.3090, -0.9511, 0.0000),
    "TP9": (-0.8777, -0.2852, -0.3826), "TP10": (0.8777, -0.2853, -0.3826),
}

NIRS_POSITIONS_3D = {
    "AF7": (-0.5878, 0.809, 0), "AF8": (0.5878, 0.809, 0),
    "AFF5": (-0.6149, 0.7564, 0.2206), "AFF6": (0.6149, 0.7564, 0.2206),
    "AFp7": (-0.454, 0.891, 0), "AFp8": (0.454, 0.891, 0),
    "AF5h": (-0.4284, 0.875, 0.2213), "AF6h": (0.4284, 0.875, 0.2212),
    "AFp3": (-0.2508, 0.9565, 0.1438), "AFp4": (0.2508, 0.9565, 0.1437),
    "AFF3h": (-0.352, 0.8111, 0.4658), "AFF4h": (0.352, 0.8111, 0.4658),
    "AF1": (-0.1857, 0.915, 0.3558), "AF2": (0.1857, 0.915, 0.3558),
    "AFFz": (0, 0.8312, 0.5554), "AFpz": (0, 0.9799, 0.1949),
    "FCC3": (-0.6957, 0.1838, 0.6933), "FCC4": (0.6957, 0.1838, 0.6933),
    "C3h": (-0.555, 0.0002, 0.8306), "C4h": (0.555, 0.0002, 0.8306),
    "C5h": (-0.8311, 0.0001, 0.5552), "C6h": (0.8311, 0.0001, 0.5552),
    "CCP3": (-0.6959, -0.1836, 0.6936), "CCP4": (0.6959, -0.1836, 0.6936),
    "CPP3": (-0.6109, -0.5259, 0.5904), "CPP4": (0.6109, -0.5258, 0.5904),
    "P3h": (-0.4217, -0.6869, 0.5912), "P4h": (0.4216, -0.687, 0.5912),
    "P5h": (-0.6411, -0.6546, 0.3985), "P6h": (0.6411, -0.6546, 0.3985),
    "PPO3": (-0.4537, -0.796, 0.3995), "PPO4": (0.4537, -0.796, 0.3995),
    "PPOz": (0, -0.8306, 0.5551),
    "PO1": (-0.1858, -0.9151, 0.3559), "PO2": (0.1859, -0.9151, 0.3559),
    "POOz": (0, -0.9797, 0.1949),
}

# ============================================================================
# LOGGING (multiprocessing-safe: a global Lock, set once per worker via the
# Pool initializer below, serializes writes to the shared log file so lines
# from different subject-processes never interleave/corrupt each other)
# ============================================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_PATH, "preprocessing_log.txt")
_LOG_LOCK = None   # set in each worker by _init_worker() when running via mp.Pool


def _init_worker(lock):
    """Pool(initializer=...) target: stashes the shared Lock in this worker
    process's global namespace so log_message() below can use it."""
    global _LOG_LOCK
    _LOG_LOCK = lock


def log_message(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pid_tag = f"PID{os.getpid()}"
    line = f"[{stamp}] [{pid_tag}] {msg}"
    print(line)
    if _LOG_LOCK is not None:
        with _LOG_LOCK:
            with open(LOG_PATH, "a") as f:
                f.write(line + "\n")
    else:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


# ============================================================================
# .mat LOADING HELPERS
# ============================================================================
def _load_struct(mat_path):
    """Uniform loader: struct_as_record=False, squeeze_me=True everywhere so
    fields are accessed as plain attributes (e.g. s.oxy.x) consistently
    across EEG and fNIRS files."""
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


# ============================================================================
# LOAD EEG
# ============================================================================
def load_eeg_data(eeg_folder):
    cnt_struct = _load_struct(os.path.join(eeg_folder, "cnt_nback.mat"))
    mrk_struct = _load_struct(os.path.join(eeg_folder, "mrk_nback.mat"))
    mnt_struct = _load_struct(os.path.join(eeg_folder, "mnt_nback.mat"))

    data = cnt_struct.x.T                       # (n_channels, n_samples)
    sfreq = float(cnt_struct.fs)
    ch_names = [str(ch).strip() for ch in cnt_struct.clab]

    if np.max(np.abs(data)) > 1.0:               # heuristically in microvolts -> convert to volts
        data = data * 1e-6

    if sfreq != SFREQ_EEG:
        n_samples = int(data.shape[1] * SFREQ_EEG / sfreq)
        data = resample(data, n_samples, axis=1)
        sfreq = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    eog_channels = [ch for ch in ["VEOG", "HEOG"] if ch in ch_names]
    if eog_channels:
        raw.set_channel_types({ch: "eog" for ch in eog_channels})
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore")

    # CONFIRMED (directly verified against VP001's real cnt_nback.mat via inspect_eeg_cnt.py):
    # cnt.clab has exactly 30 channels = 28 real scalp electrodes (Fp1..O2) + HEOG + VEOG. TP9
    # (reference) and TP10 (ground) are NOT present as recorded data channels at all -- confirming
    # what mnt_nback.mat's missing entries for them had already suggested. cnt.fs=200Hz (already
    # downsampled, matches SFREQ_EEG) and cnt.yUnit='uV' with values in the hundreds (confirmed:
    # min=-866, max=1657.5) -- i.e. genuinely microvolts, matching the ">1.0 -> assume uV, x1e-6"
    # heuristic conversion below.
    # CONCLUSION: this dataset's actual EEG tensor has 28 real (non-EOG) channels, NOT the 30
    # stated literally in the paper's Sec. 2.3 ("30 x 600"). The paper's number was evidently
    # written for the general 30-electrode cap layout (including the reference/ground positions),
    # not for the channel count that ends up in the classifier's input tensor once VEOG/HEOG are
    # split out as separate EOG channels and TP9/TP10 turn out not to be separately recorded. This
    # script proceeds with the REAL, verified 28 channels (the log line below records the count
    # per subject, since a different subject's cnt.clab could in principle differ, though this is
    # not expected given all subjects share the same recording montage/protocol).
    n_real_channels = len([c for c in ch_names if c not in ("VEOG", "HEOG")])
    has_tp9_tp10 = "TP9" in ch_names or "TP10" in ch_names
    log_message(
        f"  EEG channel check: {len(ch_names)} total in cnt.clab, "
        f"{n_real_channels} non-EOG (scalp) channels, TP9/TP10 present={has_tp9_tp10} "
        f"(expected: 28 channels, False -- confirmed via VP001)."
    )

    # Trial-level events (112/128/144 = 0-/2-/3-back session onset)
    if hasattr(mrk_struct, "time"):
        event_positions = np.round(mrk_struct.time * sfreq / 1000).astype(int)
    elif hasattr(mrk_struct, "pos"):
        event_positions = np.array(mrk_struct.pos, dtype=int) - 1
    else:
        raise ValueError("Could not find 'time' or 'pos' in EEG marker struct")

    if hasattr(mrk_struct, "event") and hasattr(mrk_struct.event, "desc"):
        desc = mrk_struct.event.desc
        event_codes = np.asarray(desc, dtype=int).flatten()
    elif hasattr(mrk_struct, "y"):
        y = mrk_struct.y
        event_codes = (np.argmax(y, axis=0) + 1) if y.ndim > 1 else y.flatten()
    else:
        event_codes = event_positions.copy()

    events = np.zeros((len(event_positions), 3), dtype=int)
    events[:, 0] = event_positions
    events[:, 2] = event_codes

    return raw, events, sfreq, mnt_struct


# ============================================================================
# LOAD fNIRS
# ============================================================================
def load_fnirs_data(subject_id):
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_struct = _load_struct(os.path.join(fnirs_folder, "cnt_nback.mat"))
    mnt_struct = _load_struct(os.path.join(fnirs_folder, "mnt_nback.mat"))

    mrk_path = os.path.join(fnirs_folder, "mrk_nback.mat")
    fnirs_events_time, fnirs_events_code = None, None
    if os.path.exists(mrk_path):
        mrk_struct = _load_struct(mrk_path)
        if hasattr(mrk_struct, "time"):
            fnirs_events_time = np.asarray(mrk_struct.time, dtype=float)
        elif hasattr(mrk_struct, "pos"):
            sf_tmp = float(getattr(cnt_struct, "fs", SFREQ_FNIRS))
            fnirs_events_time = np.asarray(mrk_struct.pos, dtype=float) / sf_tmp * 1000.0
        if hasattr(mrk_struct, "event") and hasattr(mrk_struct.event, "desc"):
            fnirs_events_code = np.asarray(mrk_struct.event.desc, dtype=int).flatten()

    oxy_raw = np.asarray(cnt_struct.oxy.x, dtype=np.float64)     # (n_samples, n_channels)
    deoxy_raw = np.asarray(cnt_struct.deoxy.x, dtype=np.float64)
    # NOTE: per Dataset_description_MATLAB.pdf, the released cnt_nback.mat is already
    # downsampled to 10 Hz (from the NIRScout's native 10.4 Hz) and already carries oxy/deoxy
    # as hemoglobin CONCENTRATION CHANGES (i.e. the modified Beer-Lambert law has already been
    # applied by the dataset authors -- see DATASET_ALREADY_BEER_LAMBERT above). The fs read
    # below is therefore expected to already read ~10 Hz; resampling further downstream is a
    # no-op safety net, not a real downsampling step.
    sfreq_fnirs_raw = float(getattr(cnt_struct.oxy, "fs", getattr(cnt_struct, "fs", SFREQ_FNIRS)))

    # 36 physiological (source-detector) channel names, needed to look each channel's position
    # up in NIRS_POSITIONS_3D (order in cnt.oxy.x's columns is assumed to match mnt.clab, both
    # coming from the same subject's files).
    try:
        ch_names = [str(c).strip() for c in np.asarray(mnt_struct.clab).flatten()]
    except Exception:
        ch_names = [f"ch{i}" for i in range(oxy_raw.shape[1])]   # fallback: positions will come
                                                                    # entirely from mnt_struct's
                                                                    # own pos_3d/ordering, if present

    return oxy_raw, deoxy_raw, sfreq_fnirs_raw, mnt_struct, ch_names, fnirs_events_time, fnirs_events_code


# ============================================================================
# fNIRS: modified Beer-Lambert law (paper Sec. 2.2)
# ============================================================================
def modified_beer_lambert(od_hi, od_lo, dpf_age=25.0):
    """Convert raw optical-density (or wavelength intensity) changes into
    HbO/HbR concentration changes using the modified Beer-Lambert law.
    Only used when DATASET_ALREADY_BEER_LAMBERT is False."""
    lambda1, lambda2 = 760, 850
    path_length = 3.0

    def get_dpf(lmbda, age):
        alpha, beta, gamma = 223.3, 0.05624, 0.8493
        delta, eps, zeta = -5.723e-7, 0.001245, -0.9025
        return alpha + beta * (age ** gamma) + delta * (lmbda ** 3) + eps * (lmbda ** 2) + zeta * lmbda

    dpf1, dpf2 = get_dpf(lambda1, dpf_age), get_dpf(lambda2, dpf_age)
    e1_hbo, e1_hbr = 0.586, 1.54852
    e2_hbo, e2_hbr = 1.058, 0.69132
    denom = path_length * (e1_hbr * e2_hbo - e2_hbr * e1_hbo)
    hbo = (e1_hbr * (od_lo / dpf2) - e2_hbr * (od_hi / dpf1)) / denom
    hbr = (e1_hbo * (od_hi / dpf1) - e2_hbo * (od_lo / dpf2)) / denom
    return hbo, hbr


def fnirs_bandpass(data, fs, low=FNIRS_BP_LOW, high=FNIRS_BP_HIGH, order=FNIRS_BP_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    filtered = np.zeros_like(data)
    for ch in range(data.shape[1]):
        filtered[:, ch] = filtfilt(b, a, data[:, ch])
    return filtered


# ============================================================================
# SPATIAL MAPPING: azimuthal equidistant projection + cubic interpolation
# (paper Sec. 2.3) — shared by EEG and fNIRS
# ============================================================================
def spatial_mapping(data, pos_3d, grid_size=GRID_SIZE):
    """
    data:    (n_samples, n_channels)
    pos_3d:  (3, n_channels) real 3D sensor positions
    returns: (n_samples, grid_size, grid_size) float32
    """
    x3d, y3d, z3d = pos_3d[0], pos_3d[1], pos_3d[2]
    r = np.sqrt(x3d ** 2 + y3d ** 2 + z3d ** 2)
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))   # angle from vertex
    phi = np.arctan2(y3d, x3d)                            # azimuthal angle
    x_coords = theta * np.cos(phi)                        # azimuthal-equidistant projection
    y_coords = theta * np.sin(phi)

    grid_x, grid_y = np.mgrid[
        min(x_coords):max(x_coords):complex(0, grid_size),
        min(y_coords):max(y_coords):complex(0, grid_size),
    ]
    points = np.column_stack((x_coords, y_coords))

    tri = Delaunay(points)                                # triangulation computed once
    values = data.T                                       # (n_channels, n_samples)
    interp = CloughTocher2DInterpolator(tri, values, fill_value=0)  # cubic spline interpolation
    result = interp((grid_x, grid_y))                     # (grid, grid, n_samples)
    mapped = np.transpose(result, (2, 0, 1)).astype(np.float32)
    return mapped


def _mnt_struct_position_lookup(mnt_struct):
    """Build a {channel_name -> (x,y,z)} dict from a per-subject mnt_nback.mat
    struct, used only as a fallback when a channel name is missing from the
    official EEG_POSITIONS_3D / NIRS_POSITIONS_3D tables above."""
    try:
        clab = [str(c).strip() for c in np.asarray(mnt_struct.clab).flatten()]
        pos_3d = np.asarray(mnt_struct.pos_3d, dtype=np.float64)
        return {name: pos_3d[:, i] for i, name in enumerate(clab)}
    except Exception:
        return {}


def get_eeg_positions(raw, eeg_mnt_struct):
    """Positions for the real (non-EOG) EEG channels present in this dataset.
    NOTE: the paper's Sec. 2.3 literally states a 30 x 600 (channel x time)
    EEG tensor, but this was CONFIRMED (via direct inspection of VP001's
    cnt_nback.mat, see inspect_eeg_cnt.py) to not match this dataset's actual
    recorded channels: cnt.clab has 28 real scalp electrodes (Fp1..O2) + HEOG
    + VEOG = 30 total, but TP9 (reference) and TP10 (ground) are simply not
    recorded as separate data channels at all. So the real, usable channel
    count for spatial mapping is 28, not 30 -- positions for the channels
    that DO exist are looked up primarily from the fixed, dataset-documented
    EEG_POSITIONS_3D table (identical across all subjects); falls back to
    the subject's own mnt_nback.mat for any name not found there (in
    practice, unused here, since mnt_nback.mat's own clab is a subset of
    EEG_POSITIONS_3D's keys)."""
    mnt_fallback = _mnt_struct_position_lookup(eeg_mnt_struct)
    pos_list, missing = [], []
    for ch in raw.ch_names:
        if ch in EEG_POSITIONS_3D:
            pos_list.append(EEG_POSITIONS_3D[ch])
        elif ch in mnt_fallback:
            pos_list.append(mnt_fallback[ch])
        else:
            missing.append(ch)
            pos_list.append((0.0, 0.0, 0.0))
    if missing:
        log_message(f"  WARNING: EEG channel(s) not found in official position table or "
                    f"mnt_nback.mat, placed at grid center: {missing}")
    return np.array(pos_list, dtype=np.float64).T          # (3, n_channels)


def get_fnirs_positions(ch_names, mnt_struct):
    """Positions of the fNIRS PHYSIOLOGICAL (source-detector midpoint)
    channels, as required by the paper ('the spatial projection was
    performed based on the positions of the physiological channels formed
    by source-detector pairs, rather than their actual positions on the
    scalp'). Looked up primarily from the fixed NIRS_POSITIONS_3D table;
    falls back to the subject's own mnt_nback.mat for any name not found."""
    mnt_fallback = _mnt_struct_position_lookup(mnt_struct)
    pos_list, missing = [], []
    for ch in ch_names:
        if ch in NIRS_POSITIONS_3D:
            pos_list.append(NIRS_POSITIONS_3D[ch])
        elif ch in mnt_fallback:
            pos_list.append(mnt_fallback[ch])
        else:
            missing.append(ch)
            pos_list.append((0.0, 0.0, 0.0))
    if missing:
        log_message(f"  WARNING: fNIRS channel(s) not found in official position table or "
                    f"mnt_nback.mat, placed at grid center: {missing}")
    return np.array(pos_list, dtype=np.float64).T          # (3, n_channels)


# ============================================================================
# EEG PREPROCESSING (paper Sec. 2.2)
# ============================================================================
def preprocess_eeg(raw, subject_id, qc_dir):
    raw_before = raw.copy()
    filtered = raw.copy()

    if APPLY_NOTCH:
        filtered.notch_filter(freqs=EEG_NOTCH_FREQ, picks="eeg", verbose=False)

    # 6th-order zero-phase Butterworth bandpass, 0.5-50 Hz
    filtered.filter(
        l_freq=EEG_BP_LOW, h_freq=EEG_BP_HIGH,
        method="iir",
        iir_params=dict(order=EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )

    # Common average reference (CAR)
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    # ICA-based ocular-artifact removal using EOG channels
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
                idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=2.0, verbose=False)
                eog_indices.extend(idx)
            except Exception as e:
                log_message(f"  {subject_id}: find_bads_eog({eog_ch}) failed: {e}")

    # Cross-check with ICLabel so weak/ambiguous EOG-correlation hits are
    # corroborated by an independent topography/spectrum-based classifier.
    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        iclabel_ocular = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))
    if ica.exclude:
        ica.apply(filtered, verbose=False)
    log_message(f"  {subject_id}: removed {len(ica.exclude)} ocular ICA component(s): {ica.exclude}")

    # QC plot: one representative channel, raw vs preprocessed
    try:
        os.makedirs(qc_dir, exist_ok=True)
        ch0 = filtered.ch_names[0]
        t_raw = raw_before.times
        t_proc = filtered.times
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(t_raw, raw_before.get_data(picks=[ch0])[0], color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - EEG RAW ({ch0})")
        axes[1].plot(t_proc, filtered.get_data(picks=[ch0])[0], color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} - EEG PREPROCESSED (0.5-50Hz + CAR + ICA) ({ch0})")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_after.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save EEG QC plot: {e}")

    return filtered


# ============================================================================
# fNIRS PREPROCESSING (paper Sec. 2.2)
# ============================================================================
def preprocess_fnirs(subject_id, qc_dir):
    oxy_raw, deoxy_raw, sfreq_raw, mnt_struct, ch_names, ev_time, ev_code = load_fnirs_data(subject_id)

    if DATASET_ALREADY_BEER_LAMBERT:
        hbo, hbr = oxy_raw, deoxy_raw
    else:
        hbo, hbr = modified_beer_lambert(oxy_raw, deoxy_raw)

    hbo_f = fnirs_bandpass(hbo, sfreq_raw)
    hbr_f = fnirs_bandpass(hbr, sfreq_raw)

    n_samples_out = int(hbo_f.shape[0] * SFREQ_FNIRS / sfreq_raw)
    hbo_rs = resample(hbo_f, n_samples_out, axis=0)
    hbr_rs = resample(hbr_f, n_samples_out, axis=0)

    pos_3d = get_fnirs_positions(ch_names, mnt_struct)

    hbo_mapped = spatial_mapping(hbo_rs, pos_3d)   # (n_samples, 16, 16)
    hbr_mapped = spatial_mapping(hbr_rs, pos_3d)

    fnirs_grid = np.stack([hbo_mapped, hbr_mapped], axis=-1)   # (n_samples, 16, 16, 2)

    try:
        os.makedirs(qc_dir, exist_ok=True)
        t_raw = np.arange(oxy_raw.shape[0]) / sfreq_raw
        t_proc = np.arange(hbo_rs.shape[0]) / SFREQ_FNIRS
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(t_raw, oxy_raw[:, 0], color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - fNIRS RAW HbO (ch 0)")
        axes[1].plot(t_proc, hbo_rs[:, 0], color="tab:blue", linewidth=0.5)
        axes[1].set_title(f"{subject_id} - fNIRS PREPROCESSED HbO (Beer-Lambert + 0.01-0.1Hz + resample)")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save fNIRS QC plot: {e}")

    return fnirs_grid, ev_time, ev_code


# ============================================================================
# WINDOW EXTRACTION (paper Sec. 2.3)
# ============================================================================
def extract_windows(eeg_raw_filtered, eeg_pos_3d, events, fnirs_grid, fnirs_ev_time, fnirs_ev_code, subject_id):
    sfreq_eeg = eeg_raw_filtered.info["sfreq"]

    # Spatially-map the FULL continuous EEG signal once (avoids re-triangulating per window)
    eeg_full = eeg_raw_filtered.get_data()             # (n_channels, n_times)
    eeg_spatial_full = spatial_mapping(eeg_full.T, eeg_pos_3d)   # (n_times, 16, 16)

    target_events = events[np.isin(events[:, 2], list(EEG_TRIAL_LABEL_MAP.keys()))]
    if len(target_events) == 0:
        log_message(f"  {subject_id}: no trial markers found (codes seen: {np.unique(events[:, 2])})")
        return [np.array([])] * 5

    # Synchronize fNIRS windowing to fNIRS's OWN trial markers (EEG and
    # fNIRS are independently-clocked devices; using EEG's clock for both
    # produces tens of seconds of drift across session boundaries).
    fnirs_sync_ok = False
    fnirs_trial_times = None
    if fnirs_ev_time is not None and fnirs_ev_code is not None:
        mask = np.isin(fnirs_ev_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
        times_ms = np.asarray(fnirs_ev_time)[mask].astype(float)
        codes = np.asarray(fnirs_ev_code)[mask]
        order = np.argsort(times_ms)
        fnirs_trial_times = times_ms[order] / 1000.0
        fnirs_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in codes[order]])
        eeg_classes = np.array([EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1) for ev in target_events])
        n_common = min(len(fnirs_classes), len(eeg_classes))
        if len(fnirs_trial_times) == len(target_events) and np.array_equal(
            fnirs_classes[:n_common], eeg_classes[:n_common]
        ):
            fnirs_sync_ok = True
        else:
            log_message(f"  {subject_id}: WARNING - fNIRS/EEG trial markers mismatch, "
                        f"falling back to EEG-clock timing for fNIRS windows.")
    else:
        log_message(f"  {subject_id}: WARNING - fNIRS trial markers unavailable, "
                    f"falling back to EEG-clock timing for fNIRS windows.")

    win_samp_eeg = int(WINDOW_LEN * sfreq_eeg)
    win_samp_fnirs = int(WINDOW_LEN * SFREQ_FNIRS)
    offsets = np.arange(SLIDE_TMIN, SLIDE_TMAX - WINDOW_LEN + 1e-6, WINDOW_STEP)

    eeg_windows, fnirs_windows, labels, session_ids, trial_ids = [], [], [], [], []

    for trial_idx, ev in enumerate(target_events):
        eeg_event_sample = ev[0]
        label = EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1)
        if label == -1:
            continue
        session_id = trial_idx // TRIALS_PER_SESSION + 1

        eeg_event_time = eeg_event_sample / sfreq_eeg
        fnirs_event_time = fnirs_trial_times[trial_idx] if fnirs_sync_ok else eeg_event_time

        # Baseline: mean fNIRS signal in [-5, -2] s before trial onset
        b_start = int(np.round((fnirs_event_time + BASELINE_TMIN) * SFREQ_FNIRS))
        b_end = int(np.round((fnirs_event_time + BASELINE_TMAX) * SFREQ_FNIRS))
        if b_start >= 0 and b_end > b_start:
            fnirs_baseline = fnirs_grid[b_start:b_end].mean(axis=0, keepdims=True)   # (1,16,16,2)
        else:
            fnirs_baseline = 0.0

        for offset in offsets:
            w_start_eeg = int(np.round((eeg_event_time + offset) * sfreq_eeg))
            w_end_eeg = w_start_eeg + win_samp_eeg
            if w_start_eeg < 0 or w_end_eeg > eeg_spatial_full.shape[0]:
                continue

            eeg_slice_raw = eeg_full[:, w_start_eeg:w_end_eeg]
            if np.max(np.abs(eeg_slice_raw)) > 120e-6:     # simple artifact rejection
                continue

            fnirs_segments, valid = [], True
            for seg_idx in range(FNIRS_NUM_SEGMENTS):
                seg_offset = offset + seg_idx * WINDOW_STEP
                w_start_f = int(np.round((fnirs_event_time + seg_offset) * SFREQ_FNIRS))
                w_end_f = w_start_f + win_samp_fnirs
                if w_start_f < 0 or w_end_f > fnirs_grid.shape[0]:
                    valid = False
                    break
                fnirs_segments.append(fnirs_grid[w_start_f:w_end_f] - fnirs_baseline)
            if not valid:
                continue

            eeg_slice = eeg_spatial_full[w_start_eeg:w_end_eeg].transpose(1, 2, 0)   # (16,16,600)
            fnirs_slice = np.stack(fnirs_segments, axis=0).transpose(0, 2, 3, 1, 4)   # (11,16,16,30,2)

            eeg_windows.append(eeg_slice)
            fnirs_windows.append(fnirs_slice)
            labels.append(label)
            session_ids.append(session_id)
            trial_ids.append(trial_idx)

    if not eeg_windows:
        return [np.array([])] * 5

    return (
        np.array(eeg_windows, dtype=np.float32),
        np.array(fnirs_windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(session_ids, dtype=np.int64),
        np.array(trial_ids, dtype=np.int64),
    )


# ============================================================================
# PER-SUBJECT PIPELINE
# ============================================================================
def process_subject(subject_id):
    t0 = time.time()
    log_message(f"===== START {subject_id} =====")
    sub_out_dir = os.path.join(OUTPUT_PATH, subject_id)
    os.makedirs(sub_out_dir, exist_ok=True)

    try:
        eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
        raw, events, sfreq_eeg, eeg_mnt_struct = load_eeg_data(eeg_folder)
        log_message(f"  {subject_id}: EEG loaded, {raw.n_times} samples @ {sfreq_eeg}Hz, {len(events)} markers")

        filtered = preprocess_eeg(raw, subject_id, sub_out_dir)
        eeg_pos_3d = get_eeg_positions(filtered, eeg_mnt_struct)
        n_eeg_channels_used = eeg_pos_3d.shape[1]   # real record of how many channels actually went into spatial_mapping()

        fnirs_grid, fnirs_ev_time, fnirs_ev_code = preprocess_fnirs(subject_id, sub_out_dir)
        log_message(f"  {subject_id}: fNIRS preprocessed, shape {fnirs_grid.shape}")

        eeg_windows, fnirs_windows, labels, session_ids, trial_ids = extract_windows(
            filtered, eeg_pos_3d, events, fnirs_grid, fnirs_ev_time, fnirs_ev_code, subject_id
        )

        if eeg_windows.size == 0:
            log_message(f"  {subject_id}: no windows extracted, skipping save.")
            return

        # Per-subject z-score normalization (grid-space, before saving)
        eeg_windows = (eeg_windows - eeg_windows.mean()) / (eeg_windows.std() + 1e-8)
        fnirs_windows = (fnirs_windows - fnirs_windows.mean()) / (fnirs_windows.std() + 1e-8)

        np.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_windows.npy"), eeg_windows)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_labels.npy"), labels)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_session_ids.npy"), session_ids)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_trial_ids.npy"), trial_ids)

        metadata = {
            "subject_id": subject_id,
            "eeg_shape": list(eeg_windows.shape),
            "fnirs_shape": list(fnirs_windows.shape),
            "labels_distribution": np.bincount(labels).tolist(),
            "window_len_s": WINDOW_LEN,
            "window_step_s": WINDOW_STEP,
            "slide_tmin_s": SLIDE_TMIN,
            "slide_tmax_s": SLIDE_TMAX,
            "fnirs_num_segments": FNIRS_NUM_SEGMENTS,
            "sfreq_eeg": SFREQ_EEG,
            "sfreq_fnirs": SFREQ_FNIRS,
            "eeg_bandpass_hz": [EEG_BP_LOW, EEG_BP_HIGH],
            "eeg_bandpass_order": EEG_BP_ORDER,
            "fnirs_bandpass_hz": [FNIRS_BP_LOW, FNIRS_BP_HIGH],
            "baseline_window_s": [BASELINE_TMIN, BASELINE_TMAX],
            "trials_per_session": TRIALS_PER_SESSION,
            "n_eeg_channels_used_for_spatial_mapping": n_eeg_channels_used,   # CONFIRMED 28 (not the
            # paper's literal 30, Sec. 2.3 "30 x 600") via direct inspection of VP001's real
            # cnt_nback.mat: 28 real scalp electrodes (Fp1..O2) + HEOG + VEOG = 30 total channels,
            # but TP9 (reference)/TP10 (ground) are not recorded as separate data channels at all.
            # Recorded per-subject here in case any subject's montage ever differs (not expected).
            "dataset_already_beer_lambert": DATASET_ALREADY_BEER_LAMBERT,
            "grid_size": GRID_SIZE,
            "normalization": "per-subject global z-score (grid space)",
            "spec_source": "TSFNet, Zhang/Yin/Yuan, Sensors 2025, 25, 6111, Sec. 2.2-2.3",
        }
        with open(os.path.join(sub_out_dir, f"{subject_id}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        elapsed = (time.time() - t0) / 60
        log_message(
            f"DONE {subject_id} in {elapsed:.1f} min | EEG {eeg_windows.shape} | "
            f"fNIRS {fnirs_windows.shape} | labels {np.bincount(labels)}"
        )

    except Exception as e:
        elapsed = (time.time() - t0) / 60
        log_message(f"ERROR processing {subject_id} after {elapsed:.1f} min: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# RUN
# ============================================================================
def _process_subject_safe(sub_num):
    """Thin wrapper so exceptions in one worker are logged and don't kill the
    whole pool (process_subject() already catches internally, but this is a
    second safety net for anything unexpected, e.g. a crashed MNE/ICA call)."""
    subject_id = f"VP{sub_num:03d}"
    try:
        process_subject(subject_id)
    except Exception as e:
        log_message(f"FATAL (uncaught) error on {subject_id}: {e}")
        import traceback
        traceback.print_exc()
    return subject_id


if __name__ == "__main__":
    log_message("===== PIPELINE STARTED (TSFNet paper-exact, n-back) =====")
    log_message(f"EEG: {EEG_BP_LOW}-{EEG_BP_HIGH}Hz order-{EEG_BP_ORDER} Butterworth, CAR, ICA-EOG")
    log_message(f"fNIRS: {FNIRS_BP_LOW}-{FNIRS_BP_HIGH}Hz bandpass, baseline [{BASELINE_TMIN},{BASELINE_TMAX}]s")
    log_message(f"Windows: {WINDOW_LEN}s step {WINDOW_STEP}s, slide [{SLIDE_TMIN},{SLIDE_TMAX}]s, "
                f"fNIRS segments={FNIRS_NUM_SEGMENTS}")
    log_message(f"Parallel run: N_WORKERS={N_WORKERS} (subject-level parallelism, "
                f"1 BLAS/OMP thread pinned per worker)")

    t_total0 = time.time()
    lock = mp.Lock()
    # NOTE (Windows): multiprocessing on Windows uses the 'spawn' start method,
    # which re-imports this script fresh in every worker process -- this is
    # exactly why the OMP/MKL/OPENBLAS env vars are set at the very top of the
    # file (before numpy/scipy/mne are imported), not just once in the parent.
    with mp.Pool(processes=N_WORKERS, initializer=_init_worker, initargs=(lock,)) as pool:
        for subject_id in pool.imap_unordered(_process_subject_safe, SUBJECTS):
            pass  # progress already reported by log_message() inside process_subject()

    total_elapsed = (time.time() - t_total0) / 60
    log_message(f"===== PIPELINE FINISHED in {total_elapsed:.1f} min "
                f"({len(SUBJECTS)} subjects, {N_WORKERS} parallel workers) =====")
