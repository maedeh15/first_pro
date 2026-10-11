# ============================================================================
# STA-Net PREPROCESSING, TSFNet-COMPARABLE WINDOWING — n-back dataset (Shin et al., 2018)
# ----------------------------------------------------------------------------
# PURPOSE: produce the SECOND row of the comparison table — STA-Net fed with the same windows as
# TSFNet — so that the only difference between the two trained models is the ARCHITECTURE.
# This is Preproc_stanet_nback_norm.py with the segmentation constants and the preprocessing
# defaults changed to match Preproc_tsfnet_nback_badch.py:
#
#                               paper-exact STA-Net      THIS SCRIPT (= TSFNet)
#   EEG window length           3 s  (600 samples)       10 s (2000 samples)
#   fNIRS segment length        3 s  (30 samples)        3 s  (30 samples)   <- unchanged
#   fNIRS segments per sample   l = 11                   l = 21
#   window step                 1 s                      1 s                 <- unchanged
#   first/last window start     -2 .. +39 s              -2 .. +32 s
#   windows per trial           42                       35
#   windows per subject         1134 (378/session)       945  (315/session)
#   EEG sample shape            (16,16,600)              (16,16,2000)
#   fNIRS sample shape          (11,16,16,30,2)          (21,16,16,30,2)
#   50 Hz notch                 ON                       OFF   (TSFNet: APPLY_NOTCH = False)
#   fNIRS 0.01-0.1 Hz bandpass  ON                       OFF   (TSFNet: ENABLE_FNIRS_BANDPASS = False)
#   EEG / fNIRS z-score         off                      ON, scope=subject (TSFNet-exact)
#   per-session normalisation   off                      ON    (TSFNet: ENABLE_PER_SESSION_NORM = True)
#
# The last window starts at SLIDE_TMAX - EEG window length = 42 - 10 = 32 s, exactly as TSFNet's
#   offsets = np.arange(SLIDE_TMIN, SLIDE_TMAX - WINDOW_LEN_EEG + 1e-6, WINDOW_STEP)
# and for each such window the 21 fNIRS segments start at offset, offset+1, ..., offset+20 s, so the
# fNIRS epoch must reach 32 + 20 + 3 = 55 s past onset (within the 62 s trial incl. rest).
#
# LEAKAGE: --zscore-scope defaults to 'subject' HERE (not 'session'), because that is what TSFNet does:
# the per-channel mean/std is taken over the whole recording, which includes the held-out session. It is
# label-free but NOT leak-free in the strict sense. It is the default here only so that both models get
# IDENTICAL inputs — a comparison in which one model is handicapped by a stricter normalisation is not a
# comparison of architectures. Pass --zscore-scope session to get the leak-free variant (then run TSFNet
# the same way, or the comparison breaks again).
#
# NOT made identical to TSFNet (and why):
#   * The 16x16 spatial mapping is STA-Net's official cubic-griddata interpolation, not TSFNet's
#     spatial_mapping(). Both place the same 28 EEG / 36 fNIRS channels on a 16x16 grid; the interpolation
#     of the remaining cells differs. Changing it would mean reimplementing TSFNet's mapping, and it is a
#     property of the DATA REPRESENTATION both models consume, so it stays the STA-Net one here.
#   * AutoReject window rejection stays OFF (TSFNet's ENABLE_AUTOREJECT_WINDOWS is False in the script you
#     sent), so no windows are dropped and both models see all 945 windows of every subject.
#   * PyPREP bad-channel interpolation is ON in both.
#
# NOTE ON THE MODEL: nothing in stanet_model.py needs changing. STANet reads seg_len from the data and
# collapses the temporal axis with a mean before the first Linear, so (16,16,2000) EEG and 21 lagged fNIRS
# windows run unchanged. The temporal kernels (13/stride 6, then 5/stride 2) were designed for the 3 s
# window and are kept as they are — say so in the thesis.
#
# OUTPUT SIZE: ~3.2 GB per subject (1.9 GB EEG + 1.2 GB fNIRS), ~83 GB for all 26. Check free disk first.
#
# CLI (all optional):
#   --eeg-window <s>          EEG window length in seconds          (default 10)
#   --fnirs-segments <n>      number of lagged fNIRS windows l      (default 21)
#   --notch on|off            50 Hz mains notch                     (default off = TSFNet)
#   --fnirs-bandpass on|off   0.01-0.1 Hz on fNIRS                  (default off = TSFNet)
#   --eeg-zscore on|off  --fnirs-zscore on|off  --per-session-norm on|off      (default on = TSFNet)
#   --zscore-scope session|subject                                  (default subject = TSFNet)
#   --subjects 5,6,7   --workers 4   --base <dir>   --out <dir>
# ----------------------------------------------------------------------------
# STA-Net-EXACT PREPROCESSING PIPELINE — n-back dataset (Shin et al., 2018)
# Reproduces Sec. 2.2 (Data Preprocessing), Sec. 2.3 (Data Segmentation) and
# Sec. 2.4 (3D EEG-fNIRS tensor) of:
#   Liu, M.; Yang, B.; et al. "STA-Net: Spatial-temporal alignment network for
#   hybrid EEG-fNIRS decoding." Information Fusion 119 (2025) 103023.
# Official code: https://github.com/MutianLiu-SHU/STA-Net  (preprocessing/ folder:
#   load_mat -> preprocessing -> epoch -> to3d -> window). Every numeric choice
#   below that the paper text leaves open was taken from that code, NOT guessed.
#
# PIPELINE (paper + official code):
#   EEG  : 50 Hz notch (official code only) -> 6th-order zero-phase Butterworth
#          0.5-50 Hz -> CAR -> ICA (fit on a 1 Hz high-passed copy, n_components=20,
#          applied to the 0.5 Hz data) removing ocular components.
#          NO z-score, NO bad-channel interpolation, NO window rejection (flags below
#          keep your extra TSFNet-style steps available but OFF by default).
#   fNIRS: dataset already holds HbO/HbR (modified Beer-Lambert applied by the
#          dataset authors) -> 6th-order zero-phase Butterworth 0.01-0.1 Hz ->
#          per-trial baseline correction (mean of -5..-2 s). No z-score.
#   Epochs: EEG [-5, TASK_PERIOD] s, fNIRS [-5, TASK_PERIOD+10] s around trial onset.
#   3D   : 16x16 grid with the official HARD-CODED electrode / channel pixel
#          coordinates (identical montage in the WG and n-back recordings),
#          cubic interpolation, then nearest-neighbour fill for pixels outside the
#          convex hull -> X_eeg (16,16,T), X_fnirs (16,16,T,2).
#   Windows: 3 s windows, step 1 s, from -2 s to TASK_PERIOD s (paper: 10, here 42 like TSFNet).
#          EEG sample = 1 window (16,16,600). fNIRS sample = the SAME-start window
#          plus the next 10 windows (l = 11) -> (11,16,16,30,2).
#
# OUTPUTS per subject (VPxxx), in OUTPUT_PATH/VPxxx/ :
#   {subject}_eeg_windows.npy    (N, 16, 16, 600)          float32  (uV)
#   {subject}_fnirs_windows.npy  (N, 11, 16, 16, 30, 2)    float32  (mmol/L, channel 0=HbO 1=HbR)
#   {subject}_labels.npy         (N,)  0=0-back, 1=2-back, 2=3-back
#   {subject}_session_ids.npy    (N,)  1/2/3   <- use this for leave-one-session-out
#   {subject}_trial_ids.npy      (N,)  0..26
#   {subject}_window_ids.npy     (N,)  0..9 (window index inside its trial)
#   {subject}_metadata.json, {subject}_eeg_clean_raw.fif, QC pngs, preprocessing_log.txt
# NOTE: the official training script multiplies fNIRS by 1e3 before feeding the
#       network (`fnirs *= 1e3`). It is NOT applied here (files stay in mmol/L);
#       do it in the training script (FNIRS_TRAIN_SCALE in metadata).
# NOTE: eeg_windows has no trailing channel axis; the official model input is
#       (16,16,600,1) -> add it in the Dataset class (np.expand_dims(..., -1)).
# ============================================================================

import os
import sys
# Must be set BEFORE numpy/scipy/mne are imported (1 BLAS thread per subject-worker).
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
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
try:
    import mne_icalabel
    _ICLABEL_AVAILABLE = True
except ImportError:
    _ICLABEL_AVAILABLE = False
try:
    from pyprep import NoisyChannels
    _PYPREP_AVAILABLE = True
except ImportError:
    _PYPREP_AVAILABLE = False

warnings.filterwarnings("ignore")

import argparse as _argparse
_arg_parser = _argparse.ArgumentParser()
_arg_parser.add_argument("--subjects", type=str, default=None,
                          help="Comma-separated subject numbers, e.g. '21' or '21,14,9'. "
                               "Omit to process all 26 subjects.")
_arg_parser.add_argument("--workers", type=int, default=None, help="Override N_WORKERS.")
_arg_parser.add_argument("--base", type=str, default=None, help="Override BASE_DATASET_PATH.")
_arg_parser.add_argument("--out", type=str, default=None, help="Override OUTPUT_PATH.")
_arg_parser.add_argument("--eeg-channel-space", choices=["on", "off"], default="on",
                          help="also write {sid}_eeg_channel_windows.npy, the SAME windows in channel space "
                               "(N,28,samples) before the 16x16 mapping -- TSFNet's ENABLE_EEG_CHANNEL_SPACE. "
                               "STA-Net does not read it; it is there so TSFNet can be trained from this very "
                               "folder. Costs ~0.4 GB per subject; pass 'off' to skip it.")
_arg_parser.add_argument("--eeg-window", type=float, default=10.0,
                          help="EEG window length in seconds (TSFNet WINDOW_LEN_EEG = 10; STA-Net paper = 3)")
_arg_parser.add_argument("--fnirs-segments", type=int, default=21,
                          help="number of lagged fNIRS windows per sample, l (TSFNet FNIRS_NUM_SEGMENTS = 21; paper = 11)")
_arg_parser.add_argument("--notch", choices=["on", "off"], default="off",
                          help="50 Hz mains notch on EEG (TSFNet APPLY_NOTCH = False -> default off here; "
                               "the STA-Net official code has it on)")
_arg_parser.add_argument("--fnirs-bandpass", choices=["on", "off"], default="off",
                          help="0.01-0.1 Hz zero-phase Butterworth on fNIRS (paper: on). The LDA diagnosis on the n-back data "
                               "found that switching it OFF restores the class information the 40 s blocks carry.")
_arg_parser.add_argument("--eeg-zscore", choices=["on", "off"], default="on", help="layer 1 (see header)")
_arg_parser.add_argument("--fnirs-zscore", choices=["on", "off"], default="on", help="layer 2 (see header)")
_arg_parser.add_argument("--per-session-norm", choices=["on", "off"], default="on", help="layer 3 (see header)")
_arg_parser.add_argument("--per-session-modalities", choices=["both", "eeg", "fnirs"], default="both",
                          help="which modality layer 3 acts on")
_arg_parser.add_argument("--zscore-scope", choices=["session", "subject"], default="subject",
                          help="statistics scope of layers 1/2: 'session' = leak-free per-session; 'subject' = whole "
                               "recording incl. the held-out session (TSFNet-exact, NOT leak-free)")
_cli_args, _ = _arg_parser.parse_known_args()

N_WORKERS = 4
if _cli_args.workers is not None:
    N_WORKERS = _cli_args.workers

# ============================================================================
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"            # folder holding VPxxx-EEG / VPxxx-NIRS
OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\STANET_TSFCOMP_nback"    # own folder: nothing else is overwritten
if _cli_args.base is not None:
    BASE_DATASET_PATH = _cli_args.base
if _cli_args.out is not None:
    OUTPUT_PATH = _cli_args.out

SUBJECTS = list(range(1, 27))  # VP001..VP026
if _cli_args.subjects is not None:
    SUBJECTS = [int(s.strip()) for s in _cli_args.subjects.split(",")]

# --- Sampling rates (the released .mat files are already 200 Hz / 10 Hz) ----
SFREQ_EEG = 200
SFREQ_FNIRS = 10

# --- EEG filtering (paper Sec. 2.2 + official preprocessing.py) --------------
EEG_BP_LOW = 0.5
EEG_BP_HIGH = 50.0
EEG_BP_ORDER = 6                 # mne iir_params=dict(order=6, ftype='butter'), zero-phase (default)
APPLY_NOTCH = (_cli_args.notch == "on")   # TSFNet applies NO notch -> default off. Official STA-Net code: raw.notch_filter(np.arange(50, 100, 50)) = 50 Hz only.
                                 # (Not mentioned in the paper text; present in the released code.)
EEG_NOTCH_FREQ = 50.0
ICA_N_COMPONENTS = 20            # official: ICA(n_components=20)
ICA_FIT_HP = 1.0                 # official: ICA fitted on a copy high-passed at 1 Hz, applied to 0.5 Hz data
ICA_METHOD = "picard"            # paper only says "ICA"; official uses mne default (fastica). picard(extended)
                                 # is the same choice as in your TSFNet script (needed by ICLabel).
DEFAULT_EOG_THRESHOLD = 2.0      # same default as your TSFNet script (find_bads_eog z-threshold)
USE_ICLABEL = True               # official code excludes ICs MANUALLY (input()); this automates it with
                                 # EOG-correlation UNION ICLabel(eye blink / eye movement), as in your script.

# --- fNIRS filtering (paper Sec. 2.2 + official code) ------------------------
FNIRS_BP_LOW = 0.01
FNIRS_BP_HIGH = 0.1
FNIRS_BP_ORDER = 6               # same mne call as EEG: iir order=6 butter, zero-phase
DATASET_ALREADY_BEER_LAMBERT = True   # Shin's .mat already contains HbO/HbR concentration changes

# --- normalisation layers (NEW; the only difference to the paper-exact script) --------------------------
ENABLE_FNIRS_BANDPASS = (_cli_args.fnirs_bandpass == "on")
ENABLE_EEG_ZSCORE = (_cli_args.eeg_zscore == "on")
ENABLE_FNIRS_ZSCORE = (_cli_args.fnirs_zscore == "on")
ENABLE_PER_SESSION_NORM = (_cli_args.per_session_norm == "on")
PER_SESSION_MODALITIES = _cli_args.per_session_modalities
ZSCORE_SCOPE = _cli_args.zscore_scope
SESSION_PRE_S = 5.0       # a session's statistics window starts this long before its first trial marker
SESSION_POST_S = 62.0     # ... and ends this long after its last one (2 s instruction + 40 s task + 20 s rest)
FNIRS_IS_STANDARDISED = ENABLE_FNIRS_ZSCORE or (ENABLE_PER_SESSION_NORM and PER_SESSION_MODALITIES in ("both", "fnirs"))
FNIRS_TRAIN_SCALE = 1e3          # official run_sta_net.py: fnirs *= 1e3 (applied in TRAINING, not here)

# --- Epoching, seconds relative to trial onset ------------------------------
# TASK_PERIOD = 42 = TSFNet's SLIDE_TMAX (2 s instruction + 40 s task). Windows start at SLIDE_TMIN = -2 s
# and the LAST one starts at TASK_PERIOD - WIN_LENGTH_EEG, so it still ends inside the task period.
TASK_PERIOD = 42
SLIDE_TMIN = -2                  # TSFNet SLIDE_TMIN: first window starts 2 s before the trial marker
EEG_PRE_ONSET = 5
EEG_POST_ONSET = TASK_PERIOD
BASELINE_N_SAMPLES = 31          # mne EpochsArray(baseline=(None, 3.)) with tmin=0 -> samples 0..30
                                 # INCLUSIVE = -5.0 ... -2.0 s (31 samples @10 Hz)

# --- Windowing (TSFNet-comparable; see the table in the header) --------------
WIN_LENGTH_EEG = int(round(_cli_args.eeg_window))   # s   -- TSFNet WINDOW_LEN_EEG      (10)
WIN_LENGTH_FNIRS = 3             # s   -- TSFNet WINDOW_LEN_FNIRS_SEGMENT (3, unchanged from the paper)
WIN_STEP = 1                     # s   -- TSFNet WINDOW_STEP
FNIRS_LAG_LENGTH = int(_cli_args.fnirs_segments)    # l   -- TSFNet FNIRS_NUM_SEGMENTS (21)
GRID_SIZE = 16

# window starts, relative to onset: SLIDE_TMIN, SLIDE_TMIN+1, ..., TASK_PERIOD - WIN_LENGTH_EEG
# (identical to TSFNet's np.arange(SLIDE_TMIN, SLIDE_TMAX - WINDOW_LEN_EEG + 1e-6, WINDOW_STEP))
N_EEG_WINDOWS = (TASK_PERIOD - WIN_LENGTH_EEG) - SLIDE_TMIN + 1          # 42-10+2+1 = 35
WIN_FIRST_START = EEG_PRE_ONSET + SLIDE_TMIN                             # s from epoch start -> 3
assert N_EEG_WINDOWS >= 1, "EEG window longer than the task period"

# the l fNIRS segments of the LAST EEG window start at (TASK_PERIOD - WIN_LENGTH_EEG) + (l-1) s and are
# WIN_LENGTH_FNIRS long, so the fNIRS epoch has to reach that far past onset:
FNIRS_PRE_ONSET = 5
FNIRS_POST_ONSET = (TASK_PERIOD - WIN_LENGTH_EEG) + (FNIRS_LAG_LENGTH - 1) + WIN_LENGTH_FNIRS   # 32+20+3 = 55
assert FNIRS_POST_ONSET <= 62, ("fNIRS epoch would run past the 62 s trial (2 s instruction + 40 s task + "
                                "20 s rest); reduce --fnirs-segments or --eeg-window")
N_FNIRS_SEGMENTS = N_EEG_WINDOWS - 1 + FNIRS_LAG_LENGTH                  # 34+21 = 55 segments to cut

# --- Optional extras from YOUR TSFNet pipeline. ALL OFF = paper/official-exact. ----
ENABLE_EEG_CHANNEL_SPACE = (_cli_args.eeg_channel_space == "on")   # TSFNet writes this too (priority-G4)
ENABLE_PYPREP_BADCH = True       # PyPREP bad-channel interpolation BEFORE CAR (not in STA-Net; ON to match your
                                 # TSFNet pipeline so both models get identical bad-channel handling)

# --- n-back marker codes (Shin et al. 2018) ----------------------------------
EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}     # 0-back / 2-back / 3-back
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}
TRIALS_PER_SESSION = 9                              # 3 sessions x 9 trials = 27 trials/subject
N_SESSIONS = 3

# --- Channel orders = official preprocessing.py (the code's pixel tables follow THIS order) ----
EEG_CHN_NAMES = ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3', 'Pz',
                 'POz', 'O1', 'Fp2', 'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2']
FNIRS_CHN_NAMES = ['AF7', 'AFF5', 'AFp7', 'AF5h', 'AFp3', 'AFF3h', 'AF1', 'AFFz', 'AFpz', 'AF2', 'AFp4', 'FCC3',
                   'C3h', 'C5h', 'CCP3', 'CPP3', 'P3h', 'P5h', 'PPO3', 'AFF4h', 'AF6h', 'AFF6', 'AFp8', 'AF8',
                   'FCC4', 'C6h', 'C4h', 'CCP4', 'CPP4', 'P6h', 'P4h', 'PPO4', 'PPOz', 'PO1', 'PO2', 'POOz']

# --- Official hard-coded 16x16 pixel coordinates, [row, col] (official to3d.py) -----------------
EEG_PIXELS = np.array([
    [0, 6], [2, 5], [2, 8], [3, 7], [5, 2], [5, 6], [7, 1], [7, 4], [7, 8], [9, 2], [9, 6], [11, 2], [11, 5],
    [11, 8], [13, 8], [14, 6], [0, 10], [2, 11], [3, 9], [5, 10], [5, 14], [7, 12], [7, 15], [9, 10], [9, 14],
    [11, 11], [11, 14], [14, 10]], dtype=float)
FNIRS_PIXELS = np.array([
    [2, 4], [3, 4], [1, 5], [2, 5], [1, 7], [3, 6], [2, 7], [3, 8], [1, 8], [2, 9], [1, 9], [6, 4], [7, 5],
    [7, 3], [8, 4], [10, 5], [11, 6], [11, 4], [12, 5], [3, 10], [2, 11], [3, 12], [1, 11], [2, 12], [6, 12],
    [7, 13], [7, 11], [8, 12], [10, 11], [11, 12], [11, 10], [12, 11], [12, 8], [13, 7], [13, 9], [14, 8]],
    dtype=float)
assert len(EEG_PIXELS) == len(EEG_CHN_NAMES) == 28 and len(FNIRS_PIXELS) == len(FNIRS_CHN_NAMES) == 36
N_EEG_CHANNELS = len(EEG_CHN_NAMES)      # 28 scalp electrodes (TP9/TP10 are reference/ground, not data)

# ============================================================================
# LOGGING (multiprocessing-safe)
# ============================================================================
os.makedirs(OUTPUT_PATH, exist_ok=True)
LOG_PATH = os.path.join(OUTPUT_PATH, "preprocessing_log.txt")
_LOG_LOCK = None


def _init_worker(lock):
    global _LOG_LOCK
    _LOG_LOCK = lock


def log_message(msg):
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] [PID{os.getpid()}] {msg}"
    print(line)
    if _LOG_LOCK is not None:
        with _LOG_LOCK:
            with open(LOG_PATH, "a") as f:
                f.write(line + "\n")
    else:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


# ============================================================================
# .mat LOADING
# ============================================================================
def _load_struct(mat_path):
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


def _read_markers(mrk_struct, sfreq_hint):
    """Returns (times_ms, codes) of ALL markers in a BBCI mrk struct."""
    if hasattr(mrk_struct, "time"):
        times_ms = np.atleast_1d(np.asarray(mrk_struct.time, dtype=float))
    elif hasattr(mrk_struct, "pos"):
        times_ms = np.atleast_1d(np.asarray(mrk_struct.pos, dtype=float)) / sfreq_hint * 1000.0
    else:
        raise ValueError("marker struct has neither 'time' nor 'pos'")
    codes = np.atleast_1d(np.asarray(mrk_struct.event.desc, dtype=int))
    return times_ms, codes


def _reorder(data, have_names, want_names, what):
    """data: (n_ch, n_t) with rows named `have_names` -> rows in `want_names` order. Fails loudly."""
    missing = [c for c in want_names if c not in have_names]
    if missing:
        raise ValueError(f"{what}: channels missing from file: {missing} (file has {have_names})")
    return data[[have_names.index(c) for c in want_names]]


def load_eeg(eeg_folder):
    cnt = _load_struct(os.path.join(eeg_folder, "cnt_nback.mat"))
    mrk = _load_struct(os.path.join(eeg_folder, "mrk_nback.mat"))
    data = np.asarray(cnt.x, dtype=np.float64).T                       # (30, n) in uV
    sfreq = float(cnt.fs)
    ch_names = [str(c).strip() for c in cnt.clab]
    if int(round(sfreq)) != SFREQ_EEG:
        raise ValueError(f"EEG fs={sfreq}, expected {SFREQ_EEG} (released data is already 200 Hz)")
    eog_present = [c for c in ("VEOG", "HEOG") if c in ch_names]
    eeg = _reorder(data, ch_names, EEG_CHN_NAMES, "EEG")               # (28, n) official order, uV
    eog = np.array([data[ch_names.index(c)] for c in eog_present]) if eog_present else np.zeros((0, data.shape[1]))
    t_ms, codes = _read_markers(mrk, sfreq)
    return eeg, eog, eog_present, t_ms, codes


def load_fnirs(subject_id):
    folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt = _load_struct(os.path.join(folder, "cnt_nback.mat"))
    mrk = _load_struct(os.path.join(folder, "mrk_nback.mat"))
    hbo = np.asarray(cnt.oxy.x, dtype=np.float64).T                    # (36, n) mmol/L
    hbr = np.asarray(cnt.deoxy.x, dtype=np.float64).T
    sfreq = float(getattr(cnt.oxy, "fs", getattr(cnt, "fs", SFREQ_FNIRS)))
    if int(round(sfreq)) != SFREQ_FNIRS:
        raise ValueError(f"fNIRS fs={sfreq}, expected {SFREQ_FNIRS} (released data is already 10 Hz)")
    ch_names = [str(c).strip() for c in np.asarray(cnt.oxy.clab).flatten()]
    # Shin's cnt.oxy.clab may carry a ' oxy' suffix or be a list of source-detector pairs; if the names
    # cannot be matched to the 36 official channel names, fall back to the dataset's native column
    # order (which is the order the official preprocessing.py assumes) and log it.
    ch_names_clean = [c.replace(" oxy", "").replace(" deoxy", "").strip() for c in ch_names]
    if all(c in ch_names_clean for c in FNIRS_CHN_NAMES):
        hbo = _reorder(hbo, ch_names_clean, FNIRS_CHN_NAMES, "fNIRS HbO")
        hbr = _reorder(hbr, ch_names_clean, FNIRS_CHN_NAMES, "fNIRS HbR")
    else:
        if hbo.shape[0] != 36:
            raise ValueError(f"fNIRS has {hbo.shape[0]} channels, expected 36")
        log_message(f"  {subject_id}: WARNING fNIRS clab names do not match the official list "
                    f"({ch_names_clean[:5]}...); assuming native column order == official order.")
    t_ms, codes = _read_markers(mrk, sfreq)
    return hbo, hbr, t_ms, codes


# ============================================================================
# EEG PREPROCESSING (official preprocessing.py)
# ============================================================================
def _get_subject_eog_threshold(subject_id):
    """Optional per-subject override file, same convention as your TSFNet script
    (OUTPUT_PATH/subject_specific_eog_thresholds.json). Default 2.0."""
    path = os.path.join(OUTPUT_PATH, "subject_specific_eog_thresholds.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                cfg = json.load(f)
            entry = cfg.get("overrides", {}).get(subject_id)
            if entry is not None:
                return float(entry["suggested_threshold"] if isinstance(entry, dict) else entry)
            return float(cfg.get("default_threshold", DEFAULT_EOG_THRESHOLD))
        except Exception as e:
            log_message(f"WARNING: could not read {path}: {e}")
    return DEFAULT_EOG_THRESHOLD


def preprocess_eeg(eeg_uv, eog_uv, eog_names, subject_id, qc_dir):
    """eeg_uv: (28, n) microvolts, official channel order. Returns (28, n) uV, filtered+CAR+ICA."""
    info = mne.create_info(ch_names=list(EEG_CHN_NAMES) + list(eog_names), sfreq=SFREQ_EEG,
                           ch_types=["eeg"] * len(EEG_CHN_NAMES) + ["eog"] * len(eog_names))
    # mne works in volts (needed for ICLabel); results are converted back to uV at the end.
    raw = mne.io.RawArray(np.vstack([eeg_uv, eog_uv]) * 1e-6, info, verbose=False)
    raw.set_montage("standard_1005", match_case=False, on_missing="ignore", verbose=False)
    raw_before = raw.copy()

    if APPLY_NOTCH:
        raw.notch_filter(freqs=EEG_NOTCH_FREQ, picks="eeg", verbose=False)
    raw.filter(l_freq=EEG_BP_LOW, h_freq=EEG_BP_HIGH, method="iir", picks="eeg",
               iir_params=dict(order=EEG_BP_ORDER, ftype="butter", output="sos"), verbose=False)

    if ENABLE_PYPREP_BADCH and _PYPREP_AVAILABLE:
        try:
            nd = NoisyChannels(raw, random_state=42)
            try:
                nd.find_all_bads(ransac=True)
            except Exception:
                nd.find_all_bads(ransac=False)
            bads = nd.get_bads()
            raw.info["bads"] = bads
            if bads:
                raw.interpolate_bads(reset_bads=True, verbose=False)
            log_message(f"  {subject_id}: PyPREP bad channels interpolated: {bads}")
        except Exception as e:
            log_message(f"  {subject_id}: PyPREP failed ({e}); continuing without it.")

    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ICA: fitted on a 1 Hz high-passed copy, applied to the 0.5 Hz data (official).
    picks_eeg = mne.pick_types(raw.info, eeg=True, eog=False)
    filt_ica = raw.copy().filter(l_freq=ICA_FIT_HP, h_freq=None, picks="eeg", verbose=False)
    ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, method=ICA_METHOD,
                                fit_params=dict(ortho=False, extended=True) if ICA_METHOD == "picard" else None,
                                random_state=42, verbose=False)
    ica.fit(filt_ica, picks=picks_eeg)

    eog_idx = []
    thr = _get_subject_eog_threshold(subject_id)
    for ch in eog_names:
        try:
            idx, _ = ica.find_bads_eog(raw_before, ch_name=ch, threshold=thr, verbose=False)
            eog_idx.extend(idx)
        except Exception as e:
            log_message(f"  {subject_id}: find_bads_eog({ch}) failed: {e}")
    iclabel_idx = []
    if USE_ICLABEL and _ICLABEL_AVAILABLE:
        try:
            lab = mne_icalabel.label_components(filt_ica, ica, method="iclabel")
            iclabel_idx = [i for i, l in enumerate(lab["labels"]) if l in ("eye blink", "eye movement")]
        except Exception as e:
            log_message(f"  {subject_id}: ICLabel failed ({e}); using EOG correlation only.")
    ica.exclude = sorted(set(eog_idx) | set(iclabel_idx))
    if ica.exclude:
        ica.apply(raw, verbose=False)
    log_message(f"  {subject_id}: ICA removed {len(ica.exclude)} component(s) {ica.exclude} "
                f"(EOG thr={thr}, EOG-corr={sorted(set(eog_idx))}, ICLabel={iclabel_idx})")
    del filt_ica

    try:
        os.makedirs(qc_dir, exist_ok=True)
        raw.save(os.path.join(qc_dir, f"{subject_id}_eeg_clean_raw.fif"), overwrite=True, verbose=False)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(raw_before.times, raw_before.get_data(picks=[0])[0] * 1e6, "tab:red", lw=0.5)
        axes[0].set_title(f"{subject_id} EEG RAW ({EEG_CHN_NAMES[0]}) uV")
        axes[1].plot(raw.times, raw.get_data(picks=[0])[0] * 1e6, "tab:blue", lw=0.5)
        axes[1].set_title(f"{subject_id} EEG PREPROCESSED (notch+0.5-50Hz+CAR+ICA) uV")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_after.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save EEG QC outputs: {e}")

    return raw.get_data(picks=picks_eeg) * 1e6, ica.exclude


# ============================================================================
# fNIRS PREPROCESSING (official preprocessing.py)
# ============================================================================
def _mne_bandpass(data, sfreq, low, high, order):
    """Exactly the official call: mne .filter(low, high, method='iir', order=6 Butterworth), zero-phase."""
    info = mne.create_info(ch_names=[f"c{i}" for i in range(data.shape[0])], sfreq=sfreq, ch_types="eeg")
    r = mne.io.RawArray(data.copy(), info, verbose=False)
    r.filter(low, high, method="iir", iir_params=dict(order=order, ftype="butter", output="sos"), verbose=False)
    return r.get_data()


def preprocess_fnirs(hbo, hbr, subject_id, qc_dir):
    """hbo/hbr: (36, n) mmol/L. Returns band-passed HbO, HbR (36, n)."""
    if not DATASET_ALREADY_BEER_LAMBERT:
        raise NotImplementedError("Shin's released .mat files are already HbO/HbR; modified Beer-Lambert "
                                  "conversion from raw intensity is not implemented here.")
    # NOTE: STA-Net's mne call scales data by 1 (mmol/L). mne treats the values as volts; for 'eeg'
    # channels filtering is linear, so scale does not matter.
    if ENABLE_FNIRS_BANDPASS:
        hbo_f = _mne_bandpass(hbo, SFREQ_FNIRS, FNIRS_BP_LOW, FNIRS_BP_HIGH, FNIRS_BP_ORDER)
        hbr_f = _mne_bandpass(hbr, SFREQ_FNIRS, FNIRS_BP_LOW, FNIRS_BP_HIGH, FNIRS_BP_ORDER)
    else:
        hbo_f, hbr_f = hbo.copy(), hbr.copy()
    try:
        t = np.arange(hbo.shape[1]) / SFREQ_FNIRS
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(t, hbo[0], "tab:red", lw=0.5)
        axes[0].set_title(f"{subject_id} fNIRS RAW HbO ({FNIRS_CHN_NAMES[0]})")
        axes[1].plot(t, hbo_f[0], "tab:blue", lw=0.5)
        axes[1].set_title(f"{subject_id} fNIRS PREPROCESSED HbO (0.01-0.1 Hz)")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=120)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save fNIRS QC plot: {e}")
    return hbo_f, hbr_f


# ============================================================================
# EPOCHING (official epoch.py)
# ============================================================================
def _trial_times(subject_id, eeg_t_ms, eeg_codes, f_t_ms, f_codes):
    """Per-trial onset times (s) for EEG and fNIRS, each from its OWN device's markers,
    plus labels. Falls back to EEG clock for fNIRS if the two marker sets do not match."""
    em = np.isin(eeg_codes, list(EEG_TRIAL_LABEL_MAP))
    eo = np.argsort(eeg_t_ms[em])
    eeg_t = eeg_t_ms[em][eo] / 1000.0
    labels = np.array([EEG_TRIAL_LABEL_MAP[int(c)] for c in eeg_codes[em][eo]])
    fm = np.isin(f_codes, list(FNIRS_TRIAL_LABEL_MAP))
    fo = np.argsort(f_t_ms[fm])
    f_t = f_t_ms[fm][fo] / 1000.0
    f_labels = np.array([FNIRS_TRIAL_LABEL_MAP[int(c)] for c in f_codes[fm][fo]])
    n_expected = TRIALS_PER_SESSION * N_SESSIONS
    if len(eeg_t) != n_expected:
        raise ValueError(f"{subject_id}: found {len(eeg_t)} EEG trial markers, expected {n_expected}")
    if len(f_t) != len(eeg_t) or not np.array_equal(f_labels, labels):
        log_message(f"  {subject_id}: WARNING fNIRS/EEG trial markers mismatch; using EEG clock for fNIRS.")
        f_t = eeg_t.copy()
    return eeg_t, f_t, labels


def epoch_trials(eeg, hbo, hbr, eeg_t, f_t, subject_id):
    n_trials = len(eeg_t)
    e_len = (EEG_PRE_ONSET + EEG_POST_ONSET) * SFREQ_EEG
    f_len = (FNIRS_PRE_ONSET + FNIRS_POST_ONSET) * SFREQ_FNIRS
    eeg_ep = np.zeros((n_trials, eeg.shape[0], e_len))
    hbo_ep = np.zeros((n_trials, hbo.shape[0], f_len))
    hbr_ep = np.zeros((n_trials, hbr.shape[0], f_len))
    for t in range(n_trials):
        s = int((eeg_t[t] - EEG_PRE_ONSET) * SFREQ_EEG)          # int() truncation as in the official code
        if s < 0 or s + e_len > eeg.shape[1]:
            raise ValueError(f"{subject_id}: trial {t} EEG epoch [{s}:{s + e_len}] outside recording ({eeg.shape[1]})")
        eeg_ep[t] = eeg[:, s:s + e_len]
        fs_ = int((f_t[t] - FNIRS_PRE_ONSET) * SFREQ_FNIRS)
        if fs_ < 0 or fs_ + f_len > hbo.shape[1]:
            raise ValueError(f"{subject_id}: trial {t} fNIRS epoch [{fs_}:{fs_ + f_len}] outside recording ({hbo.shape[1]})")
        hbo_ep[t] = hbo[:, fs_:fs_ + f_len]
        hbr_ep[t] = hbr[:, fs_:fs_ + f_len]
    # fNIRS baseline correction: subtract mean of -5..-2 s (31 samples, inclusive, as mne baseline=(None,3.))
    hbo_ep -= hbo_ep[:, :, :BASELINE_N_SAMPLES].mean(axis=2, keepdims=True)
    hbr_ep -= hbr_ep[:, :, :BASELINE_N_SAMPLES].mean(axis=2, keepdims=True)
    return eeg_ep, hbo_ep, hbr_ep


# ============================================================================
# 3D TENSOR (official to3d.py)
# ============================================================================
def _build_interp_matrix(pixels, grid_size=GRID_SIZE):
    """The official per-sample procedure is LINEAR in the channel values:
         1) cubic (Clough-Tocher) griddata at all non-electrode pixels,
         2) electrode pixels set to their measured values,
         3) remaining NaN pixels (outside convex hull) <- nearest non-NaN pixel (nearest 'griddata').
    Feeding the identity matrix through the same calls therefore yields the exact (256 x n_ch)
    operator M, so mapping a whole recording is one matrix product instead of one griddata call per
    time sample (identical result, orders of magnitude faster). Verified against the verbatim
    official loop in test_stanet_preproc.py."""
    n_ch = len(pixels)
    xx, yy = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    all_points = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
    known = {tuple(p) for p in pixels.tolist()}
    unknown = np.array([p for p in all_points if tuple(p.tolist()) not in known])
    eye = np.eye(n_ch)
    img = np.ones((grid_size, grid_size, n_ch))
    cub = griddata(points=pixels, values=eye, xi=unknown, method="cubic")          # (n_unknown, n_ch)
    for k in range(n_ch):
        img[int(pixels[k, 0]), int(pixels[k, 1])] = eye[k]
    for u in range(len(unknown)):
        img[int(unknown[u, 0]), int(unknown[u, 1])] = cub[u]
    nan_mask = np.isnan(img).any(axis=2)
    if nan_mask.any():
        good = np.argwhere(~nan_mask)
        bad = np.argwhere(nan_mask)
        near = griddata(points=good, values=img[~nan_mask], xi=bad, method="nearest")
        for i in range(len(bad)):
            img[bad[i, 0], bad[i, 1]] = near[i]
    return img.reshape(grid_size * grid_size, n_ch)                                  # (256, n_ch)


_EEG_M = None
_FNIRS_M = None


def to_3d(x, kind):
    """x: (n_trials, n_ch, T) -> (n_trials, 16, 16, T)"""
    global _EEG_M, _FNIRS_M
    if kind == "eeg":
        if _EEG_M is None:
            _EEG_M = _build_interp_matrix(EEG_PIXELS)
        M = _EEG_M
    else:
        if _FNIRS_M is None:
            _FNIRS_M = _build_interp_matrix(FNIRS_PIXELS)
        M = _FNIRS_M
    out = np.einsum("pc,nct->npt", M, x)
    return out.reshape(x.shape[0], GRID_SIZE, GRID_SIZE, x.shape[2])


# ============================================================================
# WINDOWING (official window.py)
# ============================================================================
def make_windows(eeg3d, hbo3d, hbr3d):
    """eeg3d: (n,16,16,Te); hbo3d/hbr3d: (n,16,16,Tf).
    returns eeg (n*W,16,16,Te), fnirs (n*W,l,16,16,Tf,2), trial_idx (n*W,), window_idx (n*W,);
    with the TSFNet-comparable defaults Te = 2000 (10 s), Tf = 30 (3 s), l = 21, W = 35."""
    n = eeg3d.shape[0]
    w_e = WIN_LENGTH_EEG * SFREQ_EEG
    w_f = WIN_LENGTH_FNIRS * SFREQ_FNIRS
    eeg_w = np.zeros((n, N_EEG_WINDOWS, GRID_SIZE, GRID_SIZE, w_e), dtype=np.float32)
    for w in range(N_EEG_WINDOWS):
        s = (WIN_FIRST_START + w * WIN_STEP) * SFREQ_EEG
        eeg_w[:, w] = eeg3d[..., s:s + w_e]
    seg = np.zeros((n, N_FNIRS_SEGMENTS, GRID_SIZE, GRID_SIZE, w_f, 2), dtype=np.float32)
    for fw in range(N_FNIRS_SEGMENTS):
        s = (WIN_FIRST_START + fw * WIN_STEP) * SFREQ_FNIRS
        seg[:, fw, ..., 0] = hbo3d[..., s:s + w_f]
        seg[:, fw, ..., 1] = hbr3d[..., s:s + w_f]
    fnirs_w = np.zeros((n, N_EEG_WINDOWS, FNIRS_LAG_LENGTH, GRID_SIZE, GRID_SIZE, w_f, 2), dtype=np.float32)
    for w in range(N_EEG_WINDOWS):
        fnirs_w[:, w] = seg[:, w:w + FNIRS_LAG_LENGTH]
    trial_idx = np.repeat(np.arange(n), N_EEG_WINDOWS)
    window_idx = np.tile(np.arange(N_EEG_WINDOWS), n)
    return (eeg_w.reshape(n * N_EEG_WINDOWS, GRID_SIZE, GRID_SIZE, w_e),
            fnirs_w.reshape(n * N_EEG_WINDOWS, FNIRS_LAG_LENGTH, GRID_SIZE, GRID_SIZE, w_f, 2),
            trial_idx, window_idx)


def make_channel_windows(eeg_ep):
    """The SAME EEG windows as make_windows(), but in CHANNEL space (before the 16x16 mapping).
    eeg_ep: (n_trials, n_ch, Te) -> (n_trials*W, n_ch, WIN_LENGTH_EEG*SFREQ_EEG), window order identical to
    make_windows()'s, so index i refers to the same window in both arrays (TSFNet's eeg_channel_space_windows)."""
    n, n_ch = eeg_ep.shape[0], eeg_ep.shape[1]
    w_e = WIN_LENGTH_EEG * SFREQ_EEG
    out = np.zeros((n, N_EEG_WINDOWS, n_ch, w_e), dtype=np.float32)
    for w in range(N_EEG_WINDOWS):
        st = (WIN_FIRST_START + w * WIN_STEP) * SFREQ_EEG
        out[:, w] = eeg_ep[..., st:st + w_e]
    return out.reshape(n * N_EEG_WINDOWS, n_ch, w_e)


# ============================================================================
# NORMALISATION LAYERS
# ============================================================================
def session_slices(trial_times_s, sfreq, n_samples):
    """Per-session [start, stop) sample ranges used to compute session statistics, from the trial markers
    (first trial - SESSION_PRE_S .. last trial + SESSION_POST_S), plus the 'owned' ranges the statistics are
    applied to (the gaps between sessions are assigned to the preceding session; never used by any epoch).
    Returns (core, owned): lists of (a, b) per session."""
    core = []
    for k in range(N_SESSIONS):
        t = trial_times_s[k * TRIALS_PER_SESSION:(k + 1) * TRIALS_PER_SESSION]
        a = max(0, int(np.floor((t[0] - SESSION_PRE_S) * sfreq)))
        b = min(n_samples, int(np.ceil((t[-1] + SESSION_POST_S) * sfreq)))
        core.append([a, b])
    for k in range(N_SESSIONS - 1):                      # the sessions must not overlap
        core[k][1] = min(core[k][1], core[k + 1][0])
    owned = [(0 if k == 0 else core[k][0], core[k + 1][0] if k < N_SESSIONS - 1 else n_samples)
             for k in range(N_SESSIONS)]
    return [tuple(c) for c in core], owned


def zscore_channels(data, core, owned, scope, name, subject_id):
    """data: (n_ch, n_t). Per-channel z-score over TIME. scope='session': each session uses only its own
    samples' mean/std; scope='subject': whole recording (TSFNet-exact; includes the held-out session)."""
    out = np.empty_like(data, dtype=np.float64)
    if scope == "subject":
        log_message(f"  {subject_id}: WARNING {name} z-score scope='subject' -> mean/std use ALL sessions, incl. the "
                    f"held-out one (not leak-free; TSFNet-exact).")
        out[:] = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
        return out
    for (a, b), (oa, ob) in zip(core, owned):
        mu = data[:, a:b].mean(axis=1, keepdims=True)
        sd = data[:, a:b].std(axis=1, keepdims=True) + 1e-8
        out[:, oa:ob] = (data[:, oa:ob] - mu) / sd
    return out


def apply_session_norm(eeg_w, fnirs_w, session_ids, which, chan_w=None):
    """Layer 3 (TSFNet priority-E3): each session standardised with ITS OWN statistics, per modality.
      EEG  (n,16,16,T)       : per 16x16 cell over (windows, time)
      fNIRS (n,l,16,16,T,2)  : per 16x16 cell and chromophore over (windows, lags, time)
      chan_w (n,n_ch,T)      : per channel over (windows, time) -- the channel-space copy, normalised the
                               same way TSFNet normalises eeg_channel_space_windows, so it stays consistent
                               with the grid tensor it was cut alongside.
    Sessions never see each other's statistics. In place."""
    for sess in np.unique(session_ids):
        m = np.where(session_ids == sess)[0]
        if which in ("fnirs", "both"):
            f = fnirs_w[m]
            fnirs_w[m] = (f - f.mean(axis=(0, 1, 4), keepdims=True)) / (f.std(axis=(0, 1, 4), keepdims=True) + 1e-8)
        if which in ("eeg", "both") and eeg_w is not None:
            e = eeg_w[m]
            eeg_w[m] = (e - e.mean(axis=(0, 3), keepdims=True)) / (e.std(axis=(0, 3), keepdims=True) + 1e-8)
            if chan_w is not None:
                c = chan_w[m]
                chan_w[m] = (c - c.mean(axis=(0, 2), keepdims=True)) / (c.std(axis=(0, 2), keepdims=True) + 1e-8)
    return eeg_w, fnirs_w, chan_w


# ============================================================================
# PER-SUBJECT PIPELINE
# ============================================================================
def process_subject(subject_id):
    t0 = time.time()
    log_message(f"===== START {subject_id} =====")
    sub_out_dir = os.path.join(OUTPUT_PATH, subject_id)
    os.makedirs(sub_out_dir, exist_ok=True)
    try:
        eeg, eog, eog_names, eeg_t_ms, eeg_codes = load_eeg(os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG"))
        hbo, hbr, f_t_ms, f_codes = load_fnirs(subject_id)
        log_message(f"  {subject_id}: EEG {eeg.shape} (+{len(eog_names)} EOG), fNIRS HbO {hbo.shape}, "
                    f"{len(eeg_t_ms)} EEG / {len(f_t_ms)} fNIRS markers")

        eeg_t, f_t, trial_labels = _trial_times(subject_id, eeg_t_ms, eeg_codes, f_t_ms, f_codes)   # markers only

        # Fail BEFORE the expensive steps (PyPREP + ICA, ~2 min) if a trial's epoch would run past the end of
        # the recording. The TSFNet-comparable windowing needs the fNIRS epoch to reach TASK_PERIOD -
        # WIN_LENGTH_EEG + FNIRS_LAG_LENGTH - 1 + WIN_LENGTH_FNIRS seconds past onset (55 s with the defaults,
        # vs 52 s for the paper-exact setting), so the last trial of a recording is the one at risk.
        _need_e = int((eeg_t[-1] - EEG_PRE_ONSET) * SFREQ_EEG) + (EEG_PRE_ONSET + EEG_POST_ONSET) * SFREQ_EEG
        _need_f = int((f_t[-1] - FNIRS_PRE_ONSET) * SFREQ_FNIRS) + (FNIRS_PRE_ONSET + FNIRS_POST_ONSET) * SFREQ_FNIRS
        if _need_e > eeg.shape[1] or _need_f > hbo.shape[1]:
            raise ValueError(
                f"{subject_id}: the recording ends before the last trial's epoch does "
                f"(EEG needs {_need_e} of {eeg.shape[1]} samples, fNIRS {_need_f} of {hbo.shape[1]}). "
                f"The fNIRS epoch reaches +{FNIRS_POST_ONSET}s past onset with --eeg-window {WIN_LENGTH_EEG} "
                f"--fnirs-segments {FNIRS_LAG_LENGTH}; lower one of them, or lower TASK_PERIOD.")

        eeg_p, ica_excl = preprocess_eeg(eeg, eog, eog_names, subject_id, sub_out_dir)
        if ENABLE_EEG_ZSCORE:        # layer 1: after ICA, before epoching / spatial mapping
            core, owned = session_slices(eeg_t, SFREQ_EEG, eeg_p.shape[1])
            eeg_p = zscore_channels(eeg_p, core, owned, ZSCORE_SCOPE, "EEG", subject_id)
            log_message(f"  {subject_id}: EEG z-score applied (scope={ZSCORE_SCOPE}); per-channel std after "
                        f"norm over the whole recording ~{float(eeg_p.std(axis=1).mean()):.3f}")
        hbo_p, hbr_p = preprocess_fnirs(hbo, hbr, subject_id, sub_out_dir)     # band-pass only, as in the paper
        if ENABLE_FNIRS_ZSCORE:      # layer 2: HbO and HbR separately, after band-pass, BEFORE baseline correction
            core_f, owned_f = session_slices(f_t, SFREQ_FNIRS, hbo_p.shape[1])
            hbo_p = zscore_channels(hbo_p, core_f, owned_f, ZSCORE_SCOPE, "HbO", subject_id)
            hbr_p = zscore_channels(hbr_p, core_f, owned_f, ZSCORE_SCOPE, "HbR", subject_id)
            log_message(f"  {subject_id}: fNIRS z-score applied (scope={ZSCORE_SCOPE}), before baseline correction")
        eeg_ep, hbo_ep, hbr_ep = epoch_trials(eeg_p, hbo_p, hbr_p, eeg_t, f_t, subject_id)
        chan_w = make_channel_windows(eeg_ep) if ENABLE_EEG_CHANNEL_SPACE else None
        eeg3d = to_3d(eeg_ep, "eeg")
        hbo3d = to_3d(hbo_ep, "fnirs")
        hbr3d = to_3d(hbr_ep, "fnirs")
        del eeg_ep, hbo_ep, hbr_ep

        eeg_w, fnirs_w, trial_idx, window_idx = make_windows(eeg3d, hbo3d, hbr3d)
        labels = trial_labels[trial_idx].astype(np.int64)
        session_ids = (trial_idx // TRIALS_PER_SESSION + 1).astype(np.int64)
        if ENABLE_PER_SESSION_NORM:  # layer 3: last step, on the final windowed tensors
            eeg_w, fnirs_w, chan_w = apply_session_norm(eeg_w, fnirs_w, session_ids, PER_SESSION_MODALITIES,
                                                        chan_w=chan_w)
            log_message(f"  {subject_id}: per-session normalisation applied to {PER_SESSION_MODALITIES}")

        pre = f"{subject_id}_"
        np.save(os.path.join(sub_out_dir, pre + "eeg_windows.npy"), eeg_w)
        np.save(os.path.join(sub_out_dir, pre + "fnirs_windows.npy"), fnirs_w)
        np.save(os.path.join(sub_out_dir, pre + "labels.npy"), labels)
        np.save(os.path.join(sub_out_dir, pre + "session_ids.npy"), session_ids)
        np.save(os.path.join(sub_out_dir, pre + "trial_ids.npy"), trial_idx.astype(np.int64))
        np.save(os.path.join(sub_out_dir, pre + "window_ids.npy"), window_idx.astype(np.int64))
        if chan_w is not None:
            np.save(os.path.join(sub_out_dir, pre + "eeg_channel_windows.npy"), chan_w)
            log_message(f"  {subject_id}: saved channel-space EEG windows {chan_w.shape} "
                        f"(same window order as eeg_windows.npy)")

        metadata = {
            "subject_id": subject_id,
            "eeg_shape": list(eeg_w.shape), "fnirs_shape": list(fnirs_w.shape),
            "labels_distribution": np.bincount(labels).tolist(),
            "samples_per_session": [int((session_ids == s).sum()) for s in range(1, N_SESSIONS + 1)],
            "task_period_s": TASK_PERIOD, "eeg_epoch_s": [-EEG_PRE_ONSET, EEG_POST_ONSET],
            "fnirs_epoch_s": [-FNIRS_PRE_ONSET, FNIRS_POST_ONSET],
            "win_length_eeg_s": WIN_LENGTH_EEG, "win_length_fnirs_segment_s": WIN_LENGTH_FNIRS,
            "win_step_s": WIN_STEP, "first_window_start_s": WIN_FIRST_START - EEG_PRE_ONSET,
            "last_window_start_s": TASK_PERIOD - WIN_LENGTH_EEG,
            "windows_per_trial": N_EEG_WINDOWS, "fnirs_lag_length": FNIRS_LAG_LENGTH,
            # TSFNet metadata key names, so the two folders can be diffed directly:
            "window_len_eeg_s": WIN_LENGTH_EEG, "window_len_fnirs_segment_s": WIN_LENGTH_FNIRS,
            "slide_tmin_s": float(SLIDE_TMIN), "slide_tmax_s": float(TASK_PERIOD),
            "fnirs_num_segments": FNIRS_LAG_LENGTH, "window_step_s": float(WIN_STEP),
            "autoreject_windows_enabled": False,
            "eeg_channel_space_saved": ENABLE_EEG_CHANNEL_SPACE,
            "eeg_channel_space_shape": (list(chan_w.shape) if chan_w is not None else None),
            "eeg_connectivity_saved": False,   # TSFNet's ENABLE_EEG_CONNECTIVITY is False too
            "variant": "STA-Net preprocessing with TSFNet-comparable segmentation",
            "eeg_filter": f"notch50={APPLY_NOTCH}, butter order {EEG_BP_ORDER} zero-phase {EEG_BP_LOW}-{EEG_BP_HIGH} Hz, CAR, ICA",
            "ica": {"n_components": ICA_N_COMPONENTS, "method": ICA_METHOD, "fit_highpass_hz": ICA_FIT_HP,
                    "excluded": [int(i) for i in ica_excl]},
            "fnirs_filter": (f"butter order {FNIRS_BP_ORDER} zero-phase {FNIRS_BP_LOW}-{FNIRS_BP_HIGH} Hz"
                             if ENABLE_FNIRS_BANDPASS else "NONE (bandpass off)"),
            "fnirs_baseline_s": [-5.0, -2.0],
            "FNIRS_TRAIN_SCALE": (1.0 if FNIRS_IS_STANDARDISED else FNIRS_TRAIN_SCALE),   # read by stanet_common.load_subject
            "eeg_units": "uV" if not (ENABLE_EEG_ZSCORE or (ENABLE_PER_SESSION_NORM and PER_SESSION_MODALITIES in ("both", "eeg")))
                        else "standardised (unitless)",
            "fnirs_units": "mmol/L (NOT scaled)" if not FNIRS_IS_STANDARDISED else "standardised (unitless)",
            "normalization": {
                "ENABLE_EEG_ZSCORE": ENABLE_EEG_ZSCORE, "ENABLE_FNIRS_ZSCORE": ENABLE_FNIRS_ZSCORE,
                "ENABLE_PER_SESSION_NORM": ENABLE_PER_SESSION_NORM, "per_session_modalities": PER_SESSION_MODALITIES,
                "zscore_scope": ZSCORE_SCOPE,
                "leak_free_wrt_heldout_session": (not (ENABLE_EEG_ZSCORE or ENABLE_FNIRS_ZSCORE) or ZSCORE_SCOPE == "session"),
                "order": "EEG: filters->PyPREP->CAR->ICA->[EEG z-score]->epoch->grid->windows->[per-session norm]; "
                         "fNIRS: band-pass->[z-score]->epoch->baseline(-5..-2s)->grid->windows->[per-session norm]",
            },
            "pyprep_badch": ENABLE_PYPREP_BADCH,
            "spec_source": "STA-Net, Liu et al., Information Fusion 119 (2025) 103023, Sec. 2.2-2.4 + official repo; "
                           "segmentation and normalisation matched to Preproc_tsfnet_nback_badch.py",
        }
        with open(os.path.join(sub_out_dir, pre + "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        log_message(f"DONE {subject_id} in {(time.time() - t0) / 60:.1f} min | EEG {eeg_w.shape} | "
                    f"fNIRS {fnirs_w.shape} | labels {np.bincount(labels)} | per-session {metadata['samples_per_session']}")
    except Exception as e:
        log_message(f"ERROR processing {subject_id} after {(time.time() - t0) / 60:.1f} min: {e}")
        import traceback
        traceback.print_exc()


def _process_subject_safe(sub_num):
    subject_id = f"VP{sub_num:03d}"
    try:
        process_subject(subject_id)
    except Exception as e:
        log_message(f"FATAL (uncaught) error on {subject_id}: {e}")
    return subject_id


if __name__ == "__main__":
    log_message("===== PIPELINE STARTED (STA-Net preprocessing, TSFNet-COMPARABLE windowing, n-back) =====")
    log_message(f"WINDOWING: EEG {WIN_LENGTH_EEG}s ({WIN_LENGTH_EEG * SFREQ_EEG} samples) / fNIRS segment "
                f"{WIN_LENGTH_FNIRS}s ({WIN_LENGTH_FNIRS * SFREQ_FNIRS} samples), step {WIN_STEP}s, "
                f"starts {SLIDE_TMIN}..{TASK_PERIOD - WIN_LENGTH_EEG}s -> {N_EEG_WINDOWS} windows/trial "
                f"({N_EEG_WINDOWS * TRIALS_PER_SESSION} per session), l={FNIRS_LAG_LENGTH} lagged fNIRS windows, "
                f"fNIRS epoch -{FNIRS_PRE_ONSET}..{FNIRS_POST_ONSET}s | notch50={APPLY_NOTCH}")
    log_message(f"NORMALISATION: eeg_zscore={ENABLE_EEG_ZSCORE} fnirs_zscore={ENABLE_FNIRS_ZSCORE} "
                f"per_session_norm={ENABLE_PER_SESSION_NORM}({PER_SESSION_MODALITIES}) zscore_scope={ZSCORE_SCOPE} "
                f"fnirs_bandpass={ENABLE_FNIRS_BANDPASS} -> fNIRS training scale {'1.0 (standardised)' if FNIRS_IS_STANDARDISED else '1e3'}")
    log_message(f"SUBJECTS={SUBJECTS}, N_WORKERS={N_WORKERS}, TASK_PERIOD={TASK_PERIOD}s, "
                f"EEG windows/trial={N_EEG_WINDOWS}, fNIRS lag l={FNIRS_LAG_LENGTH}")
    _gb = (N_EEG_WINDOWS * TRIALS_PER_SESSION * N_SESSIONS
           * (GRID_SIZE ** 2 * WIN_LENGTH_EEG * SFREQ_EEG
              + FNIRS_LAG_LENGTH * GRID_SIZE ** 2 * WIN_LENGTH_FNIRS * SFREQ_FNIRS * 2
              + (N_EEG_CHANNELS * WIN_LENGTH_EEG * SFREQ_EEG if ENABLE_EEG_CHANNEL_SPACE else 0)) * 4) / 1024 ** 3
    log_message(f"DISK: about {_gb:.1f} GB per subject, {_gb * len(SUBJECTS):.0f} GB for this run. "
                f"Make sure that much is free -- a full disk fails with 'N requested and 0 written'.")
    t_total0 = time.time()
    # Jupyter/IPython `%run` leaves __main__ without __spec__, which makes multiprocessing's Windows "spawn" pool crash
    # (AttributeError: module '__main__' has no attribute '__spec__'). Detect that and process the subjects one after the
    # other in this process instead (same results, just not parallel). From a terminal (python script.py) or `!python
    # script.py` in a notebook the parallel pool is used as before.
    if N_WORKERS > 1 and not hasattr(sys.modules.get("__main__"), "__spec__"):
        log_message("NOTE: running inside an IPython/Jupyter %run -> multiprocessing is not possible here; processing "
                    "the subjects SEQUENTIALLY (use a terminal or `!python` for the parallel run).")
        for _n in SUBJECTS:
            _process_subject_safe(_n)
    else:
        lock = mp.Lock()
        with mp.Pool(processes=N_WORKERS, initializer=_init_worker, initargs=(lock,)) as pool:
            for _ in pool.imap_unordered(_process_subject_safe, SUBJECTS):
                pass
    log_message(f"===== PIPELINE FINISHED in {(time.time() - t_total0) / 60:.1f} min =====")
