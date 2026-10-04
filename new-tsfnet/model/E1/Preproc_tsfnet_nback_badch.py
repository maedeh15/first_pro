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
#   {subject}_eeg_clean_raw.fif                (fully-preprocessed EEG, channel-space, MNE format --
#                                                reloadable for QC without re-running ICA)
#   {subject}_eeg_before_after.png            (QC, 1 representative channel)
#   {subject}_eeg_before_allchannels.png      (QC, all 28 EEG channels stacked, raw)
#   {subject}_eeg_after_allchannels.png       (QC, all 28 EEG channels stacked, preprocessed)
#   {subject}_eeg_before_allchannels_zoom10_30.png  (QC, same, zoomed to 10-30s)
#   {subject}_eeg_after_allchannels_zoom10_30.png   (QC, same, zoomed to 10-30s)
#   {subject}_fnirs_clean_raw.fif              (fully-preprocessed fNIRS, channel-space, MNE
#                                                format, hbo/hbr channel types -- reloadable
#                                                without re-running Beer-Lambert/bandpass/resample)
#   {subject}_fnirs_before_after.png          (QC, 1 representative channel)
#   preprocessing_log.txt          global run log
# ============================================================================

import os
import sys
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
try:
    from autoreject import AutoReject
    _AUTOREJECT_AVAILABLE = True
except ImportError:
    _AUTOREJECT_AVAILABLE = False
try:
    from pyprep import NoisyChannels
    _PYPREP_AVAILABLE = True
except ImportError:
    _PYPREP_AVAILABLE = False

warnings.filterwarnings("ignore")

# ADDED: optional CLI overrides for SUBJECTS/N_WORKERS, so a single subject (e.g. a
# transient-OOM retry like VP021) can be re-run WITHOUT re-processing all 26 subjects
# and WITHOUT hand-editing SUBJECTS/N_WORKERS below every time. If neither flag is
# passed, behavior is 100% unchanged: all 26 subjects, N_WORKERS as configured below.
# Placed here (right after imports, before N_WORKERS/SUBJECTS are defined below) so
# both of those can read _cli_args without a definition-order problem.
#   python preprocessing_....py --subjects 21 --workers 1
#   python preprocessing_....py --subjects 21,14,9 --workers 2
import argparse as _argparse
_arg_parser = _argparse.ArgumentParser()
_arg_parser.add_argument("--subjects", type=str, default=None,
                          help="Comma-separated subject numbers to (re-)process, e.g. '21' or "
                               "'21,14,9'. Omit to process all 26 subjects (default, unchanged "
                               "behavior).")
_arg_parser.add_argument("--workers", type=int, default=None,
                          help="Override N_WORKERS below (e.g. 1 or 2 for a small single-subject "
                               "re-run, to avoid the concurrent-memory-pressure OOM seen with "
                               "N_WORKERS=6 on the full 26-subject run). Omit to use the default "
                               "N_WORKERS defined below (unchanged behavior).")
_cli_args, _ = _arg_parser.parse_known_args()

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
# N_WORKERS=10 caused MemoryError crashes in practice (observed: VP012, VP014,
# VP017, VP019, VP021 all failed with "Unable to allocate ... MiB" during a real
# run) -- actual peak per-worker memory during simultaneous ICA/QC-plot steps
# across 10 concurrent subjects exceeded the ~2GB/worker estimate above. Reduced
# to 6 as a safer default; raise back toward 8-10 only if you've confirmed
# memory headroom (Task Manager -> Performance -> Memory staying well under 90%
# throughout a full run) on your specific machine.
#
# IMPORTANT UPDATE (post-AutoReject-on-windows): apply_autoreject_to_windows()
# (see ENABLE_AUTOREJECT_WINDOWS further below) keeps an additional
# (28, win_samp_eeg) x n_windows channel-space array in memory per subject AND
# runs AutoReject's own cross-validated fitting on top of everything ICA
# already needed -- this pushes each worker's peak memory noticeably above
# the estimate the N_WORKERS=10 crash above was based on. The N_WORKERS=4
# default below already accounts for this; if you raise it, start from 2-3
# and watch free RAM in Task Manager on your FIRST run with
# ENABLE_AUTOREJECT_WINDOWS=True before going higher -- do not assume the old
# ceiling (6-10) still applies now that this step is enabled.
N_WORKERS = 4
if _cli_args.workers is not None:
    N_WORKERS = _cli_args.workers


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

# _cli_args parsed once, right after imports (see top of file) -- reused here.
if _cli_args.subjects is not None:
    SUBJECTS = [int(s.strip()) for s in _cli_args.subjects.split(",")]

# --- EEG filtering (paper Sec. 2.2) ----------------------------------------
EEG_BP_LOW = 0.5
EEG_BP_HIGH = 50.0
EEG_BP_ORDER = 6          # 6th-order zero-phase Butterworth
EEG_NOTCH_FREQ = 50.0     # mains notch is not mentioned in the paper text but is standard
                          # practice for this dataset's recording site; kept OFF by default
                          # to stay literal to the paper. Set APPLY_NOTCH=True to enable.
APPLY_NOTCH = False

# --- fNIRS filtering (paper Sec. 2.2) ---------------------------------------
# CHANGED per ablation item #5: the paper's 0.01-0.1Hz Butterworth low-pass is now
# DISABLED by default (ENABLE_FNIRS_BANDPASS=False), per the MSTFDN paper's argument
# that large convolutional kernels in the model itself already act as an implicit
# low-pass filter, making a hand-designed one potentially redundant -- but this is a
# genuinely riskier hypothesis than #1/#3 (unlike those, it's not a physiological
# consistency requirement, it's a bet on the architecture's own inductive bias): removing
# it can let physiological noise (heart rate ~1Hz -- aliased into the 10Hz-sampled fNIRS
# signal since no anti-aliasing low-pass remains -- respiration, ~0.1Hz Mayer waves) leak
# directly into the 0.01-0.1Hz band the target hemodynamic response lives in. Tested here
# in ISOLATION from #1/#3 (same WINDOW_LEN_EEG=10s/FNIRS_NUM_SEGMENTS=21 pipeline as the
# already-validated run) so the filtering effect alone can be attributed. The
# fnirs_bandpass() function itself is left completely unchanged below (not removed, not
# rewritten) -- only whether it's CALLED is gated by this flag -- so ablation #6 (isolated
# Baseline-correction test) can later toggle this same flag independently to get its
# filter-on/off x baseline-on/off 2x2 factorial without any further code changes here.
ENABLE_FNIRS_BANDPASS = False
FNIRS_BP_LOW = 0.01
FNIRS_BP_HIGH = 0.1
# EFFECTIVE (documented/intended) order of the zero-phase fNIRS bandpass. The paper does not
# specify an order for fNIRS (only EEG's is stated as 6). We intentionally use a lower order than
# EEG (see rationale below) because fNIRS has a much lower sampling rate (10Hz) and a very narrow
# passband (0.01-0.1Hz), where a high-order filter risks numerical instability/ringing.
#
# IMPORTANT (same fix as applied to the EEG filter): fnirs_bandpass() below uses filtfilt for
# zero-phase filtering, which applies the filter forward AND backward -> this SQUARES the
# magnitude response, doubling the effective filter order/roll-off steepness relative to the
# order passed into scipy.signal.butter(). Previously FNIRS_BP_ORDER=3 was passed DIRECTLY into
# butter(), silently producing an effective order of 6 (double the intended/documented value) --
# the exact same class of bug flagged for the EEG filter in the implementation notes.
# FNIRS_BP_ORDER below is now the DESIGN order actually passed to butter(); it is derived from the
# documented effective order via ceil(effective/2), and FNIRS_BP_EFFECTIVE_ORDER records the true
# resulting effective order for logging/metadata so the two numbers are never confused again.
FNIRS_BP_EFFECTIVE_ORDER_TARGET = 3   # documented/intended zero-phase order (fNIRS-appropriate, lower than EEG's 6)
FNIRS_BP_ORDER = int(np.ceil(FNIRS_BP_EFFECTIVE_ORDER_TARGET / 2))   # design order passed to butter() -> 2
FNIRS_BP_EFFECTIVE_ORDER = FNIRS_BP_ORDER * 2   # true effective order after filtfilt -> 4 (closest even
                                                  # value achievable to the odd target of 3, since
                                                  # zero-phase filtfilt doubling can only produce even
                                                  # effective orders)
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

# --- Windowing (paper Sec. 2.3, ADAPTED for this dataset's actual task length) ---------------
# The paper's [-2, 10]s window was sized for its own MI/MA/WG tasks. This n-back dataset's actual
# task duration is 40s (confirmed: "2s instruction + 40s task + 20s rest", per Dataset_description_
# MATLAB.pdf / Shin et al. 2018 -- see SUBJECTS note below), so the slide range is extended here to
# actually cover the real task, not just its first ~10s.
WINDOW_LEN_EEG = 10.0      # seconds -- EEG sliding-window duration.

# ----------------------------------------------------------------------------
# REAL epoch-level artifact rejection on the trial windows themselves (not the
# QC-only diagnostic version used in the earlier ablation-test scripts, which
# ran AutoReject on throwaway fixed-length 3s pseudo-epochs and never touched
# the actual saved windows). This is the piece discussed with the user: after
# extract_windows() builds each (16,16,2000) EEG window, AutoReject is fit on
# the CHANNEL-SPACE version of those same windows to decide, per window, per
# channel: good / needs interpolation / unrecoverable.
#   - "needs interpolation" windows: the few bad channels in that window are
#     replaced with AutoReject's own interpolation (same principle as
#     PyPREP's channel interpolation, but per-window instead of per-recording
#     -- catches TRANSIENT noise like VP001's 750-1000s EMG burst that a
#     whole-recording bad-channel check cannot see, since it only affects a
#     handful of windows, not the channel's overall statistics).
#     The window is corrected and KEPT -- both modalities stay in sync.
#   - "unrecoverable" windows (too many channels bad at once in that window):
#     the window is DROPPED from eeg_windows, and the matching fnirs_windows/
#     labels/session_ids/trial_ids entries are dropped too, so both
#     modalities stay perfectly synchronized (see extract_windows() sync
#     logic already in place for fNIRS trial-marker alignment).
# Set to False to fully restore the previous behavior (no AutoReject on
# windows at all) for A/B comparison against a baseline run.
ENABLE_AUTOREJECT_WINDOWS = False
# priority-E1: compute and save per-window EEG functional-connectivity (PLV,
# theta+alpha) features alongside the usual eeg_windows.npy/fnirs_windows.npy/
# etc. See eeg_connectivity.py for the math and the block right after the
# np.save() calls below for where this is invoked. Independent of
# ENABLE_AUTOREJECT_WINDOWS -- connectivity is computed on whichever
# eeg_channel_space_final extract_windows() returns either way (corrected if
# AutoReject ran, raw-sliced if it didn't). Default False: a subject
# directory produced with this flag off is missing
# "{subject_id}_eeg_connectivity.npy", and NBackDataset (see
# tsfnet_model_notebook.py) transparently falls back to connectivity=None in
# that case, so existing runs/results are entirely unaffected by adding this
# flag -- it only takes effect when explicitly turned on for a NEW run.
ENABLE_EEG_CONNECTIVITY = True
# AutoReject's own internal cross-validated thresholding needs a reasonable
# number of epochs to fit reliably; below this, we skip AutoReject for that
# subject (log it) rather than let it fit noisily on too few windows.
AUTOREJECT_MIN_WINDOWS = 20
                           # CHANGED per ablation item #1: 3.0 -> 10.0 s. Rationale (Ding et al.
                           # MSTFDN, 2026, same dataset): a 3 s window only captures a small slice of
                           # the n-back task's sustained cognitive-load dynamics (40 s task duration);
                           # a 10 s window lets the model see much more of the alpha/theta working-
                           # memory-related dynamics. This directly changes the EEG tensor shape from
                           # (16,16,600) to (16,16,2000) at SFREQ_EEG=200Hz.
WINDOW_LEN_FNIRS_SEGMENT = 3.0   # seconds -- fNIRS PER-SEGMENT duration. Deliberately kept separate
                           # from WINDOW_LEN_EEG (previously both were driven by one shared
                           # WINDOW_LEN variable, which would have silently also stretched each
                           # fNIRS segment to 10 s -- an unrelated, undiscussed change). Left at the
                           # paper's original 3 s here; FNIRS_NUM_SEGMENTS (total temporal coverage)
                           # is a separate ablation item (#3), out of scope for this edit.
WINDOW_STEP = 1.0          # seconds, both modalities
SLIDE_TMIN = -2.0          # seconds relative to trial onset (instruction period)
SLIDE_TMAX = 42.0          # covers the full 2s instruction + 40s task for this dataset
FNIRS_NUM_SEGMENTS = 21    # CHANGED per ablation item #3: 11 -> 21. Direct consequence of #1
                           # (WINDOW_LEN_EEG: 3->10s): with 11 segments, fNIRS coverage was
                           # (11-1)*WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT = 10+3 = 13s past the
                           # window offset -- fine when EEG window itself was 3s, but no longer
                           # covers a 10s EEG window plus the fNIRS hemodynamic response lag
                           # (HRF can peak up to ~10s after task onset). With 21 segments, coverage
                           # becomes (21-1)*1 + 3 = 23s, which spans the full 10s EEG window AND the
                           # HRF delay after it, letting CAFE actually see the relevant hemodynamic
                           # response instead of missing its tail. This was deliberately NOT changed
                           # together with #1 (kept at the paper's original 11 here previously) so
                           # the two effects could be attributed separately in the ablation.

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
    the subject's own mnt_nback.mat for any name not found there.

    BUGFIX: this must iterate over EEG-TYPE channels only (mne.pick_types),
    NOT raw.ch_names directly. raw.ch_names still includes VEOG/HEOG (their
    ch_type is set to 'eog' in load_eeg_data(), but they are never dropped
    from the Raw object's channel list). VEOG/HEOG are absent from
    EEG_POSITIONS_3D, and mnt_nback.mat's own pos_3d entries for them are
    NaN (EOG electrodes have no standard scalp position) -- so the previous
    version of this function silently produced NaN rows for VEOG/HEOG,
    which downstream crashed scipy.spatial.Delaunay in spatial_mapping()
    with "Points cannot contain NaN" for every single subject. Restricting
    to EEG-type channels here (and matching that in extract_windows()'s
    eeg_full = eeg_raw_filtered.get_data(picks="eeg")) fixes this."""
    eeg_ch_names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, eog=False)]
    mnt_fallback = _mnt_struct_position_lookup(eeg_mnt_struct)
    pos_list, missing = [], []
    for ch in eeg_ch_names:
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
# EOG THRESHOLD OVERRIDE (optional, produced by qc_eog_threshold_validation.py)
# ============================================================================
DEFAULT_EOG_THRESHOLD = 2.0
_EOG_THRESHOLD_CONFIG_PATH = os.path.join(OUTPUT_PATH, "subject_specific_eog_thresholds.json")
_EOG_THRESHOLD_CONFIG = None   # lazy-loaded once per process


def _get_subject_eog_threshold(subject_id):
    """Returns the per-subject EOG threshold if a manually-reviewed override file exists
    (see qc_eog_threshold_validation.py), otherwise DEFAULT_EOG_THRESHOLD=2.0. Overrides are
    ONLY meant to be added after visual QC confirms a genuine ocular IC was missed by the
    default threshold -- never applied blindly/globally (see dataset_implementation_notes.md
    discussion on why this must stay per-subject)."""
    global _EOG_THRESHOLD_CONFIG
    if _EOG_THRESHOLD_CONFIG is None:
        if os.path.exists(_EOG_THRESHOLD_CONFIG_PATH):
            try:
                with open(_EOG_THRESHOLD_CONFIG_PATH, "r") as f:
                    _EOG_THRESHOLD_CONFIG = json.load(f)
                log_message(f"Loaded subject-specific EOG threshold overrides from "
                            f"{_EOG_THRESHOLD_CONFIG_PATH}: "
                            f"{list(_EOG_THRESHOLD_CONFIG.get('overrides', {}).keys())}")
            except Exception as e:
                log_message(f"WARNING: could not load {_EOG_THRESHOLD_CONFIG_PATH}: {e} "
                            f"-- using default threshold for all subjects.")
                _EOG_THRESHOLD_CONFIG = {"default_threshold": DEFAULT_EOG_THRESHOLD, "overrides": {}}
        else:
            _EOG_THRESHOLD_CONFIG = {"default_threshold": DEFAULT_EOG_THRESHOLD, "overrides": {}}

    overrides = _EOG_THRESHOLD_CONFIG.get("overrides", {})
    if subject_id in overrides:
        entry = overrides[subject_id]
        # supports both the QC script's {"suggested_threshold": x, "reason": ...} shape
        # and a bare numeric override for convenience
        value = entry["suggested_threshold"] if isinstance(entry, dict) else entry
        if value is not None:
            return float(value)
    return _EOG_THRESHOLD_CONFIG.get("default_threshold", DEFAULT_EOG_THRESHOLD)


# ============================================================================
# MANUAL IC EXCLUSION OVERRIDES (from full 26-subject visual/spectral QC review)
# ============================================================================
# Result of running qc_eog_fast_full_check.py (automated spectral veto) across ALL
# 26 subjects, followed by manual visual review (topography + time-course + PSD,
# 2-of-3 criterion) of every IC the automated veto could not resolve on its own
# (or, for VP010, where the veto's automatic verdict conflicted with an earlier
# manual read of the plots). Only 5/26 subjects had any CORR_ONLY (correlation-
# flagged, ICLabel-unconfirmed) IC at all; the other 21 needed no review.
#
# "unexclude": IC indices that the DEFAULT pipeline (correlation ∪ ICLabel) would
#   exclude, but that visual review determined to be false positives (genuine
#   brain/muscle activity, not ocular) -- these are REMOVED from ica.exclude.
# "keep_excluded": IC indices visual review explicitly CONFIRMED as genuinely
#   ocular -- listed here only for documentation/traceability; no code action
#   needed since they are already excluded by the default logic.
#
# EMPIRICAL VALIDATION (2026-08-15, full 26-subject baseline run comparison):
# comparing accuracy before/after these overrides revealed that VP010 and VP022 --
# 2 of the 3 subjects that had an "unexclude" override applied -- were the WORST
# and 3rd-worst accuracy regressions in the entire dataset (VP010: -13.7%, VP022:
# -7.1%), while VP011 (the third overridden subject) was essentially flat (+0.3%,
# not a real improvement). This is strong evidence that the visual "false positive"
# calls for VP010 IC8 and VP022 IC6 were WRONG -- these ICs likely did carry some
# genuine ocular contamination despite not cleanly meeting the 2-of-3 visual
# criterion, and un-excluding them reintroduced that noise into the signal. Both
# overrides are therefore DISABLED below (commented out, not deleted, for
# traceability) so these two subjects revert to the DEFAULT exclusion logic
# (correlation ∪ ICLabel, both ICs correctly excluded again).
MANUAL_IC_OVERRIDES = {
    # "VP010": {
    #     "unexclude": [8],
    #     "reason": "No frontal topography, continuous (non-transient) time-course, "
    #                "PSD bump near 30Hz (muscle-like) instead of low-freq dominance -- "
    #                "false positive. (Automated spectral veto returned KEEP_EXCLUDED "
    #                "here; this manual visual read overrides it -- see dual-filter ICA "
    #                "test which showed the alternative fit was unstable/unreliable for "
    #                "this subject and could not be used to arbitrate.)",
    #     "DISABLED": "Empirically caused -13.7% accuracy regression (worst in dataset) -- "
    #                  "the visual false-positive call was likely wrong. Reverted to default "
    #                  "(IC8 stays excluded).",
    # },
    "VP011": {
        "unexclude": [6],
        "reason": "Clear 8-10Hz alpha peak, lateralized (non-frontal) topography, "
                   "continuous oscillatory time-course -- genuine brain activity, "
                   "not ocular. (Automated spectral veto: AUTO_UNEXCLUDE.)",
    },
    # "VP022": {
    #     "unexclude": [6],
    #     "reason": "No frontal topography (lateralized, asymmetric bipolar pattern), "
    #                "continuous non-transient time-course, flat 1/f PSD with no "
    #                "low-freq or alpha peak -- same false-positive pattern as VP011 "
    #                "IC6. (Automated spectral veto: AMBIGUOUS -> resolved manually.)",
    #     "DISABLED": "Empirically caused -7.1% accuracy regression (3rd-worst in dataset) -- "
    #                  "the visual false-positive call was likely wrong. Reverted to default "
    #                  "(IC6 stays excluded).",
    # },
    "VP018": {
        "unexclude": [],
        "reason": "CONFIRMED genuinely ocular (3/3 criteria: frontal topography, "
                   "sparse transient spikes, low-frequency-dominant PSD). No change "
                   "-- stays excluded by default logic. (Automated spectral veto: "
                   "AMBIGUOUS -> resolved manually as KEEP.)",
    },
    "VP012": {
        "unexclude": [],
        "reason": "CONFIRMED genuinely ocular (2/3 criteria: frontal topography, "
                   "sparse blink-like transients; PSD atypical -- 15-30Hz bump "
                   "instead of low-freq dominance -- but explained by the transient, "
                   "non-rhythmic nature of infrequent blinks rather than indicating "
                   "a non-ocular source). No change -- stays excluded by default "
                   "logic. (Automated spectral veto: AMBIGUOUS -> resolved manually as KEEP.)",
    },
}


def _apply_manual_ic_overrides(subject_id, exclude_set):
    """Applies the hardcoded MANUAL_IC_OVERRIDES table to an automatically-computed
    exclusion set. Returns (new_exclude_set, list_of_log_messages)."""
    logs = []
    override = MANUAL_IC_OVERRIDES.get(subject_id)
    if override is None:
        return exclude_set, logs
    to_unexclude = set(override.get("unexclude", []))
    removed = to_unexclude & exclude_set
    if removed:
        exclude_set = exclude_set - to_unexclude
        logs.append(f"  {subject_id}: MANUAL OVERRIDE -- removed IC(s) {sorted(removed)} "
                    f"from exclusion (false positive per visual QC). Reason: {override['reason']}")
    elif to_unexclude:
        logs.append(f"  {subject_id}: MANUAL OVERRIDE configured for IC(s) {sorted(to_unexclude)} "
                     f"but none were in the automatically-computed exclusion set (nothing to remove).")
    return exclude_set, logs


# ============================================================================
# FREQUENCY-DOMAIN (PSD) QC PLOTS -- same helpers as eeg_full_dataset_qc_scan.py,
# now saved for every subject directly by the main pipeline (not just the
# separate QC-scan script), so before/after PSD comparisons are always
# available without a second run. Three plots per raw/filtered pair:
#   1. all-channels-overlaid PSD, before vs after, side by side -- shows
#      line-noise removal and the overall shape/level change.
#   2. channel x frequency heatmap -- makes it obvious WHICH channel is
#      abnormal in WHICH frequency band (broadband noise = full-row bright
#      stripe; EMG-like noise = bright only above ~30Hz for one channel).
#   3. per-channel band-power bar chart (delta/theta/alpha/beta/gamma) --
#      answers "does this channel/subject actually carry real gamma-band
#      signal, or is it dominated by noise", directly relevant to the
#      workload/n-back analysis (prefrontal beta/gamma is specifically what
#      matters for cognitive load, per the literature search done earlier).
# ============================================================================
def _compute_psd_from_raw(raw_obj, fmin=0.5, fmax=49.0):
    """Returns (ch_names, freqs, psd_db) where psd_db has shape
    (n_channels, n_freqs), in dB (10*log10(uV^2/Hz)) for readability."""
    raw_eeg = raw_obj.copy().pick("eeg")
    spectrum = raw_eeg.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                                    n_fft=int(raw_eeg.info["sfreq"] * 4),
                                    verbose=False)
    psd, freqs = spectrum.get_data(return_freqs=True)  # (n_ch, n_freqs), V^2/Hz
    psd_db = 10 * np.log10(np.maximum(psd, 1e-24) * 1e12)  # V^2 -> uV^2, then dB
    return raw_eeg.ch_names, freqs, psd_db


def _plot_psd_before_after(raw_before, filtered, subject_id):
    ch_before, freqs_before, psd_before = _compute_psd_from_raw(raw_before)
    ch_after, freqs_after, psd_after = _compute_psd_from_raw(filtered)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for i in range(psd_before.shape[0]):
        axes[0].plot(freqs_before, psd_before[i], linewidth=0.6, alpha=0.6)
    axes[0].set_title(f"{subject_id} - PSD RAW (all channels overlaid)")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    axes[0].axvline(50, color="k", linestyle="--", linewidth=0.8, label="50Hz line noise")
    axes[0].legend(fontsize=8)

    for i in range(psd_after.shape[0]):
        axes[1].plot(freqs_after, psd_after[i], linewidth=0.6, alpha=0.6)
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

    band_means = {}
    for band_name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        band_means[band_name] = psd_db[:, mask].mean(axis=1) if mask.any() else \
            np.full(len(ch_names), np.nan)

    fig, ax = plt.subplots(figsize=(max(12, 0.35 * len(ch_names)), 6))
    x = np.arange(len(ch_names))
    n_bands = len(bands)
    width = 0.8 / n_bands
    for i, (band_name, vals) in enumerate(band_means.items()):
        ax.bar(x + i * width, vals, width=width, label=band_name)
    ax.set_xticks(x + width * (n_bands - 1) / 2)
    ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
    ax.set_ylabel("Power (dB, 10log10 uV^2/Hz)")
    ax.set_title(f"{subject_id} - Per-channel band power ({title_suffix})")
    ax.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    plt.tight_layout()
    return fig


def save_psd_qc_plots(raw_before, filtered, subject_id, qc_dir):
    """Saves the before/after PSD comparison for filter+PyPREP+CAR+ICA (the
    continuous-signal cleaning steps). For the AutoReject-on-windows
    before/after PSD comparison, see save_autoreject_psd_plots() below --
    that one operates on the extracted windows, not the continuous signal,
    since AutoReject's corrections are per-window."""
    try:
        os.makedirs(qc_dir, exist_ok=True)
        fig1 = _plot_psd_before_after(raw_before, filtered, subject_id)
        fig1.savefig(os.path.join(qc_dir, f"{subject_id}_psd_before_after.png"), dpi=150)
        plt.close(fig1)

        fig2 = _plot_psd_heatmap(raw_before, subject_id, "RAW")
        fig2.savefig(os.path.join(qc_dir, f"{subject_id}_psd_heatmap_before.png"), dpi=150)
        plt.close(fig2)

        fig3 = _plot_psd_heatmap(filtered, subject_id, "PREPROCESSED")
        fig3.savefig(os.path.join(qc_dir, f"{subject_id}_psd_heatmap_after.png"), dpi=150)
        plt.close(fig3)

        fig4 = _plot_band_power_bars(filtered, subject_id, "PREPROCESSED")
        fig4.savefig(os.path.join(qc_dir, f"{subject_id}_band_power_after.png"), dpi=150)
        plt.close(fig4)
        log_message(f"  {subject_id}: saved PSD QC plots (before/after, heatmaps, band power)")
    except Exception as e:
        log_message(f"  {subject_id}: could not save PSD QC plots: {e}")


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

    # ------------------------------------------------------------------
    # PyPREP bad-channel detection + interpolation -- BEFORE CAR, per the
    # V1_prep vs V0_baseline and A1_car_then_ica vs A2_ica_then_car ablation
    # tests run on this dataset. Those tests showed:
    #   - without this step, channels with a fully dead/dominant electrode
    #     (e.g. VP015's Cz, VP021's CP2, VP018's Cz/POz/O2) are never
    #     resolved by CAR+ICA alone and their noise leaks into every other
    #     channel through the CAR average.
    #   - running this BEFORE CAR (not after) is what the A1/A2 test
    #     confirmed as correct: doing it after CAR lets the still-bad
    #     channel's noise already contaminate the reference average.
    # NOT a "paper-exact" step (the paper's own protocol doesn't include
    # it) -- this is the empirically best-performing option found by
    # testing candidate methods against THIS dataset's actual QC images,
    # not a literature-fidelity choice.
    # ------------------------------------------------------------------
    pyprep_bads = []
    if _PYPREP_AVAILABLE:
        try:
            nd = NoisyChannels(filtered, random_state=42)
            try:
                nd.find_all_bads(ransac=True)
            except Exception as e:
                log_message(f"  {subject_id}: PyPREP RANSAC failed ({e}); retrying without RANSAC.")
                nd.find_all_bads(ransac=False)
            pyprep_bads = nd.get_bads()
            filtered.info["bads"] = pyprep_bads
            if pyprep_bads:
                filtered.interpolate_bads(reset_bads=True, verbose=False)
            log_message(f"  {subject_id}: PyPREP bad channels detected+interpolated: {pyprep_bads}")
        except Exception as e:
            log_message(f"  {subject_id}: PyPREP step FAILED ({e}); continuing WITHOUT "
                        f"bad-channel interpolation for this subject (CAR/ICA below will run "
                        f"on the uncorrected channels).")
    else:
        log_message(f"  {subject_id}: pyprep not installed (pip install pyprep) -- skipping "
                    f"bad-channel detection/interpolation entirely for this subject.")

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
    eog_threshold = _get_subject_eog_threshold(subject_id)
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch, threshold=eog_threshold, verbose=False)
                eog_indices.extend(idx)
            except Exception as e:
                log_message(f"  {subject_id}: find_bads_eog({eog_ch}) failed: {e}")
    log_message(f"  {subject_id}: EOG threshold used = {eog_threshold} "
                f"({'OVERRIDE (see subject_specific_eog_thresholds.json)' if eog_threshold != DEFAULT_EOG_THRESHOLD else 'default'})")

    # Cross-check with ICLabel so weak/ambiguous EOG-correlation hits are
    # corroborated by an independent topography/spectrum-based classifier.
    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_ocular = [i for i, lab in enumerate(ic_labels["labels"]) if lab in ("eye blink", "eye movement")]
    except Exception:
        iclabel_ocular = []

    ica.exclude = sorted(set(eog_indices) | set(iclabel_ocular))

    # Apply the manual IC overrides from the full 26-subject QC review (see
    # MANUAL_IC_OVERRIDES above) -- removes confirmed false positives for the
    # 3 subjects (VP010, VP011, VP022) where visual review overturned the
    # automatic correlation/ICLabel decision.
    exclude_set, override_logs = _apply_manual_ic_overrides(subject_id, set(ica.exclude))
    for msg in override_logs:
        log_message(msg)
    ica.exclude = sorted(exclude_set)

    if ica.exclude:
        ica.apply(filtered, verbose=False)
    log_message(f"  {subject_id}: removed {len(ica.exclude)} ocular ICA component(s): {ica.exclude}")

    # ------------------------------------------------------------------
    # QC PLOTS -- reproduces every plot the exploratory reference notebook
    # (Preprocessing_v2_run_a1b_10s.ipynb) saved per subject, so runs from
    # this paper-exact script and the old exploratory runs can be visually
    # compared side by side when writing up results:
    #   1. Single-channel before/after (kept, unchanged from before)
    #   2-5. ALL-CHANNELS STACKED before/after, full duration + 10-30s zoom
    #        (butterfly-style view, one row per channel with a fixed
    #        vertical offset, so all 28 real EEG channels -- see the
    #        confirmed channel-count note in load_eeg_data() -- are visible
    #        in a single image instead of only one representative channel).
    # ------------------------------------------------------------------
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
        log_message(f"  {subject_id}: could not save EEG single-channel QC plot: {e}")

    try:
        def _plot_all_channels_stacked(raw_obj, title, color, tmin=None, tmax=None):
            # EOG channels (VEOG/HEOG) are EXCLUDED from these plots: (1) they never reach the
            # model (extract_windows() uses picks="eeg" only, per the get_eeg_positions() fix
            # above), so a QC plot should show exactly the 28 channels the model actually sees,
            # and (2) EOG amplitude is typically much larger than EEG, which skews the shared
            # MAD-based spread/offset below and makes the real EEG channels look flat/unreadable
            # by comparison when EOG is mixed in.
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

            # Robust (MAD-based) per-channel spread for a fixed vertical offset between
            # channels, so channels stay readable/non-overlapping whether raw (large,
            # artifact-heavy amplitudes) or preprocessed (smaller, cleaned amplitudes).
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

        fig_before = _plot_all_channels_stacked(
            raw_before, f"{subject_id} - EEG RAW - All 28 Channels, EOG excluded (before preprocessing)", "tab:red")
        fig_after = _plot_all_channels_stacked(
            filtered, f"{subject_id} - EEG PREPROCESSED - All 28 Channels, EOG excluded (0.5-50Hz+CAR+ICA)", "tab:blue")
        fig_before_zoom = _plot_all_channels_stacked(
            raw_before, f"{subject_id} - EEG RAW - All 28 Channels, EOG excluded (10-30s)", "tab:red", tmin=10.0, tmax=30.0)
        fig_after_zoom = _plot_all_channels_stacked(
            filtered, f"{subject_id} - EEG PREPROCESSED - All 28 Channels, EOG excluded (10-30s)", "tab:blue", tmin=10.0, tmax=30.0)

        fig_before.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels.png"), dpi=150)
        fig_after.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels.png"), dpi=150)
        fig_before_zoom.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_before_allchannels_zoom10_30.png"), dpi=150)
        fig_after_zoom.savefig(os.path.join(qc_dir, f"{subject_id}_eeg_after_allchannels_zoom10_30.png"), dpi=150)
        plt.close(fig_before); plt.close(fig_after)
        plt.close(fig_before_zoom); plt.close(fig_after_zoom)
        log_message(f"  {subject_id}: saved EEG all-channels QC plots (before/after, full+zoomed)")
    except Exception as e:
        log_message(f"  {subject_id}: could not save EEG all-channels QC plots: {e}")

    # PSD (frequency-domain) QC plots -- before/after filter+PyPREP+CAR+ICA.
    # Uses raw_before/filtered as-is (still in real physical units at this
    # point, PRE z-score normalization below), same as the time-domain plots
    # above.
    save_psd_qc_plots(raw_before, filtered, subject_id, qc_dir)

    # ------------------------------------------------------------------
    # PER-CHANNEL, PER-SUBJECT Z-SCORE NORMALIZATION (corrected ordering)
    # ------------------------------------------------------------------
    # Applied HERE (on the continuous, filtered+CAR+ICA-cleaned signal, BEFORE spatial mapping
    # and windowing) rather than globally on the final saved window tensor. Rationale:
    #   - Per-CHANNEL statistics (mean/std computed independently for each of the 28 electrodes,
    #     over the full recording) preserve the relative amplitude differences BETWEEN channels,
    #     which spatial_mapping() below depends on for a meaningful 16x16 topographic grid. A
    #     single global mean/std across the whole flattened tensor (the previous approach, applied
    #     in process_subject() after windows were already saved) would distort/wash out exactly
    #     that inter-channel spatial information.
    #   - Doing this BEFORE windowing (not after, as the old code did) avoids the previously
    #     documented bug where z-scoring a window tensor that had already been through further
    #     transposes/copies produced a near-zero apparent std (std ~ 0.0003 instead of ~1.0) --
    #     normalizing once, early, on the clean channel-space signal sidesteps that failure mode
    #     entirely.
    #   - The z-score is intentionally computed AFTER ICA (so ocular-artifact removal isn't
    #     distorted by a prior rescaling) and BEFORE spatial_mapping/windowing (so every extracted
    #     window already carries normalized amplitudes; no separate post-hoc normalization step is
    #     needed downstream in extract_windows()/process_subject()).
    #   - QC plots above intentionally use the PRE-normalization data (real physical uV units) so
    #     they remain visually interpretable; only the returned `filtered` object (and therefore
    #     the saved .fif and all downstream tensors) is normalized.
    picks_eeg_norm = mne.pick_types(filtered.info, eeg=True, eog=False)
    eeg_data_norm = filtered.get_data(picks=picks_eeg_norm)          # (n_channels, n_times)
    ch_mean = eeg_data_norm.mean(axis=1, keepdims=True)
    ch_std = eeg_data_norm.std(axis=1, keepdims=True) + 1e-8
    eeg_data_norm = (eeg_data_norm - ch_mean) / ch_std
    filtered._data[picks_eeg_norm, :] = eeg_data_norm
    log_message(f"  {subject_id}: applied per-channel z-score to EEG "
                f"(sanity check, mean per-channel std AFTER norm, should be ~1.0: "
                f"{float(np.mean(filtered.get_data(picks=picks_eeg_norm).std(axis=1))):.6g})")

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

    if ENABLE_FNIRS_BANDPASS:
        hbo_f = fnirs_bandpass(hbo, sfreq_raw)
        hbr_f = fnirs_bandpass(hbr, sfreq_raw)
    else:
        # ablation #5: filter disabled -- pass the (Beer-Lambert-resolved, per the branch
        # above) HbO/HbR straight through unfiltered. Resample step right below is
        # UNCHANGED either way (resampling is not part of the "filtering" ablation).
        hbo_f, hbr_f = hbo, hbr

    n_samples_out = int(hbo_f.shape[0] * SFREQ_FNIRS / sfreq_raw)
    hbo_rs = resample(hbo_f, n_samples_out, axis=0)
    hbr_rs = resample(hbr_f, n_samples_out, axis=0)

    pos_3d = get_fnirs_positions(ch_names, mnt_struct)

    # Save the fully-preprocessed fNIRS in CHANNEL-SPACE (36 physiological channels x 2
    # chromophores x time), as a standard MNE .fif file, using MNE's native 'hbo'/'hbr' channel
    # types -- same rationale as the EEG .fif save in preprocess_eeg(): this is the only point
    # where the cleaned HbO/HbR signal still exists per-channel; spatial_mapping() right above
    # already converted it into the 16x16 grid tensor, after which the channel-space form is
    # not recoverable. Lets you reload with mne.io.read_raw_fif() later (raw.plot(), PSD,
    # topomaps of individual source-detector channels, etc.) without re-running the
    # Beer-Lambert/bandpass/resample steps.
    try:
        os.makedirs(qc_dir, exist_ok=True)
        fnirs_ch_names, fnirs_ch_types, fnirs_data_rows = [], [], []
        for i, ch in enumerate(ch_names):
            fnirs_ch_names += [f"{ch} hbo", f"{ch} hbr"]
            fnirs_ch_types += ["hbo", "hbr"]
            fnirs_data_rows += [hbo_rs[:, i], hbr_rs[:, i]]
        fnirs_info = mne.create_info(ch_names=fnirs_ch_names, sfreq=SFREQ_FNIRS, ch_types=fnirs_ch_types)
        fnirs_raw = mne.io.RawArray(np.array(fnirs_data_rows), fnirs_info, verbose=False)
        fnirs_raw.save(os.path.join(qc_dir, f"{subject_id}_fnirs_clean_raw.fif"), overwrite=True, verbose=False)
        log_message(f"  {subject_id}: saved cleaned channel-space fNIRS ({subject_id}_fnirs_clean_raw.fif)")
    except Exception as e:
        log_message(f"  {subject_id}: could not save fNIRS .fif: {e}")

    try:
        os.makedirs(qc_dir, exist_ok=True)
        t_raw = np.arange(oxy_raw.shape[0]) / sfreq_raw
        t_proc = np.arange(hbo_rs.shape[0]) / SFREQ_FNIRS
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].plot(t_raw, oxy_raw[:, 0], color="tab:red", linewidth=0.5)
        axes[0].set_title(f"{subject_id} - fNIRS RAW HbO (ch 0)")
        axes[1].plot(t_proc, hbo_rs[:, 0], color="tab:blue", linewidth=0.5)
        _filt_desc = f"{FNIRS_BP_LOW}-{FNIRS_BP_HIGH}Hz" if ENABLE_FNIRS_BANDPASS else "NO bandpass (ablation #5)"
        axes[1].set_title(f"{subject_id} - fNIRS PREPROCESSED HbO (Beer-Lambert + {_filt_desc} + resample)")
        plt.tight_layout()
        fig.savefig(os.path.join(qc_dir, f"{subject_id}_fnirs_before_after.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        log_message(f"  {subject_id}: could not save fNIRS QC plot: {e}")

    # ------------------------------------------------------------------
    # PER-CHANNEL, PER-SUBJECT Z-SCORE NORMALIZATION (corrected ordering) -- mirrors the EEG fix
    # in preprocess_eeg(). Applied HERE, on the continuous filtered+resampled channel-space signal
    # (separately for HbO and HbR, each channel normalized by its own mean/std over the full
    # recording), BEFORE spatial_mapping/windowing and BEFORE baseline correction.
    #   - Per-channel (not global-tensor) statistics preserve the inter-channel spatial contrast
    #     that spatial_mapping() needs for a meaningful 16x16 grid.
    #   - Doing this pre-windowing avoids the previously documented std~0.0003 bug from
    #     normalizing an already-transposed/reshaped window tensor.
    #   - Critically, this happens BEFORE baseline correction (which is applied later, in
    #     extract_windows(), directly on fnirs_grid) so that baseline correction subtracts the
    #     pre-trial mean of the ALREADY-normalized signal -- baseline correction on top of z-score
    #     is meaningful and does not get cancelled out, whereas the reverse order (z-score AFTER
    #     baseline correction) would largely undo the baseline subtraction.
    #   - The .fif save and QC plot above intentionally use the PRE-normalization hbo_rs/hbr_rs
    #     (real physical mmol/L units) so they stay visually/physically interpretable; only the
    #     grid tensor built below (and everything downstream of it) is normalized.
    hbo_mean, hbo_std = hbo_rs.mean(axis=0, keepdims=True), hbo_rs.std(axis=0, keepdims=True) + 1e-8
    hbr_mean, hbr_std = hbr_rs.mean(axis=0, keepdims=True), hbr_rs.std(axis=0, keepdims=True) + 1e-8
    hbo_rs_norm = (hbo_rs - hbo_mean) / hbo_std
    hbr_rs_norm = (hbr_rs - hbr_mean) / hbr_std
    log_message(f"  {subject_id}: applied per-channel z-score to fNIRS HbO/HbR "
                f"(sanity check, mean per-channel std AFTER norm, should be ~1.0: "
                f"HbO={float(np.mean(hbo_rs_norm.std(axis=0))):.6g}, "
                f"HbR={float(np.mean(hbr_rs_norm.std(axis=0))):.6g})")

    hbo_mapped = spatial_mapping(hbo_rs_norm, pos_3d)   # (n_samples, 16, 16), normalized
    hbr_mapped = spatial_mapping(hbr_rs_norm, pos_3d)

    fnirs_grid = np.stack([hbo_mapped, hbr_mapped], axis=-1)   # (n_samples, 16, 16, 2), normalized;
                                                                 # baseline correction (per-trial,
                                                                 # -5..-2s) is applied to THIS grid
                                                                 # downstream in extract_windows(),
                                                                 # i.e. after normalization -- correct order.

    return fnirs_grid, ev_time, ev_code


# ============================================================================
# WINDOW EXTRACTION (paper Sec. 2.3)
# ============================================================================
def extract_windows(eeg_raw_filtered, eeg_pos_3d, events, fnirs_grid, fnirs_ev_time, fnirs_ev_code, subject_id, qc_dir=None):
    sfreq_eeg = eeg_raw_filtered.info["sfreq"]

    # BUGFIX (see get_eeg_positions() docstring): must use picks="eeg" here, matching exactly
    # the EEG-type-only channel selection used to build eeg_pos_3d. Without picks="eeg",
    # get_data() returns ALL channels including VEOG/HEOG, so eeg_full's channel axis (30) would
    # no longer align with eeg_pos_3d's channel axis (28) even after the get_eeg_positions() fix
    # above -- this alignment is required for spatial_mapping()'s per-channel (points, values)
    # pairing to be meaningful, not just to avoid a shape-mismatch crash.
    eeg_full = eeg_raw_filtered.get_data(picks="eeg")   # (28, n_times) -- EOG channels excluded
    eeg_spatial_full = spatial_mapping(eeg_full.T, eeg_pos_3d)   # (n_times, 16, 16)

    target_events = events[np.isin(events[:, 2], list(EEG_TRIAL_LABEL_MAP.keys()))]
    if len(target_events) == 0:
        log_message(f"  {subject_id}: no trial markers found (codes seen: {np.unique(events[:, 2])})")
        return [np.array([])] * 6

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

    win_samp_eeg = int(WINDOW_LEN_EEG * sfreq_eeg)
    win_samp_fnirs = int(WINDOW_LEN_FNIRS_SEGMENT * SFREQ_FNIRS)
    offsets = np.arange(SLIDE_TMIN, SLIDE_TMAX - WINDOW_LEN_EEG + 1e-6, WINDOW_STEP)

    eeg_windows, fnirs_windows, labels, session_ids, trial_ids = [], [], [], [], []
    eeg_channel_space_windows = []   # parallel list: (28, win_samp_eeg) per window, RAW channel-
                                       # space (pre-spatial_mapping) slices, needed as AutoReject's
                                       # input below since it operates on channels, not the 16x16
                                       # spatially-interpolated grid.

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
            # NOTE: eeg_full comes from `filtered`, which is now Z-SCORE NORMALIZED
            # (per-channel, in preprocess_eeg() -- see the corrected-ordering notes there),
            # NOT raw microvolts anymore. The old 120e-6 (120 uV) threshold was calibrated
            # for raw-volts data; comparing it against z-scored values (typically in the
            # -5..+5 range) made EVERY window fail this check, silently producing zero
            # windows for every subject. The threshold below is the z-score equivalent:
            # reject a window only if some sample is an extreme outlier (>8 standard
            # deviations from that channel's mean), which is the standardized-data analogue
            # of the original amplitude-based rejection intent.
            if np.max(np.abs(eeg_slice_raw)) > 8.0:     # z-score-based artifact rejection
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

            eeg_slice = eeg_spatial_full[w_start_eeg:w_end_eeg].transpose(1, 2, 0)   # (16,16,2000) with WINDOW_LEN_EEG=10s @ 200Hz
            fnirs_slice = np.stack(fnirs_segments, axis=0).transpose(0, 2, 3, 1, 4)   # (21,16,16,30,2) with FNIRS_NUM_SEGMENTS=21

            eeg_windows.append(eeg_slice)
            eeg_channel_space_windows.append(eeg_slice_raw.copy())   # (28, win_samp_eeg), same
                                                                        # index as eeg_windows[-1]
            fnirs_windows.append(fnirs_slice)
            labels.append(label)
            session_ids.append(session_id)
            trial_ids.append(trial_idx)

    if not eeg_windows:
        return [np.array([])] * 6

    eeg_windows = np.array(eeg_windows, dtype=np.float32)
    fnirs_windows = np.array(fnirs_windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    session_ids = np.array(session_ids, dtype=np.int64)
    trial_ids = np.array(trial_ids, dtype=np.int64)
    eeg_channel_space_windows = np.array(eeg_channel_space_windows, dtype=np.float32)

    if ENABLE_AUTOREJECT_WINDOWS:
        eeg_windows, fnirs_windows, labels, session_ids, trial_ids, eeg_channel_space_windows = \
            apply_autoreject_to_windows(
                eeg_windows, eeg_channel_space_windows, fnirs_windows, labels, session_ids, trial_ids,
                eeg_pos_3d, sfreq_eeg, eeg_raw_filtered, subject_id, qc_dir,
            )
        if eeg_windows.size == 0:
            return [np.array([])] * 6

    return (
        eeg_windows,
        fnirs_windows,
        labels,
        session_ids,
        trial_ids,
        eeg_channel_space_windows,   # priority-E1: final, AutoReject-corrected,
                                       # per-window channel-space EEG (pre-spatial_mapping),
                                       # the input eeg_connectivity.py's PLV extraction needs.
    )


def apply_autoreject_to_windows(eeg_windows, eeg_channel_space_windows, fnirs_windows,
                                 labels, session_ids, trial_ids, eeg_pos_3d, sfreq_eeg,
                                 eeg_raw_filtered, subject_id, qc_dir):
    """REAL epoch-level artifact rejection on the actual saved trial windows
    (not the QC-only diagnostic version from the ablation-test scripts).

    Fits AutoReject on the channel-space (28, win_samp_eeg) version of every
    extracted window, treating each window as one "epoch". For each window,
    AutoReject decides per-channel: good / needs interpolation / bad. Windows
    where too many channels are simultaneously bad are DROPPED (removed from
    ALL FIVE output arrays together, keeping EEG/fNIRS/labels/session_ids/
    trial_ids in sync by index). Windows needing only a few channels fixed
    are corrected in place: AutoReject's own interpolated channel-space data
    replaces the affected channels, then spatial_mapping() is re-run on just
    that window's corrected channel data so eeg_windows keeps using the same
    16x16 spatially-interpolated representation as everywhere else in this
    pipeline.

    Returns the (possibly shorter, possibly locally-corrected) five arrays.
    On any failure (AutoReject not installed, too few windows, or any
    exception), logs a warning and returns the ORIGINAL arrays unchanged --
    this step is a refinement on top of the already-validated
    filter->PyPREP->CAR->ICA pipeline, not a step that should ever be able to
    crash a subject's entire run or silently produce zero windows.
    """
    n_windows = eeg_windows.shape[0]

    if not _AUTOREJECT_AVAILABLE:
        log_message(f"  {subject_id}: AutoReject not installed (pip install autoreject) -- "
                    f"skipping window-level rejection, keeping all {n_windows} windows as-is.")
        return eeg_windows, fnirs_windows, labels, session_ids, trial_ids, eeg_channel_space_windows

    if n_windows < AUTOREJECT_MIN_WINDOWS:
        log_message(f"  {subject_id}: only {n_windows} windows (<{AUTOREJECT_MIN_WINDOWS}), "
                    f"too few for reliable AutoReject cross-validation -- skipping, keeping "
                    f"all windows as-is.")
        return eeg_windows, fnirs_windows, labels, session_ids, trial_ids, eeg_channel_space_windows

    try:
        eeg_picks_idx = mne.pick_types(eeg_raw_filtered.info, eeg=True, eog=False)
        eeg_only_info = mne.pick_info(eeg_raw_filtered.info, eeg_picks_idx)   # MEMORY: does
                                                                                # NOT duplicate the
                                                                                # continuous recording's
                                                                                # data array, unlike
                                                                                # eeg_raw_filtered.copy()
                                                                                # would.
        epochs = mne.EpochsArray(eeg_channel_space_windows, eeg_only_info, verbose=False)

        ar = AutoReject(random_state=42, verbose=False, n_jobs=1)
        epochs_clean, reject_log = ar.fit_transform(epochs, return_log=True)

        # reject_log.labels: (n_epochs, n_channels) array, values per autoreject's own
        # convention: 0 = good, 1 = interpolated, 2 = bad-and-not-interpolated (only set
        # for epochs autoreject decided to drop entirely, i.e. reject_log.bad_epochs=True
        # for that row).
        bad_epoch_mask = reject_log.bad_epochs   # bool, shape (n_windows,) -- True = DROP
        n_dropped = int(bad_epoch_mask.sum())
        n_interpolated_windows = int(((reject_log.labels == 1).any(axis=1) & ~bad_epoch_mask).sum())
        n_untouched = n_windows - n_dropped - n_interpolated_windows

        log_message(f"  {subject_id}: AutoReject on {n_windows} trial windows -> "
                    f"{n_untouched} untouched, {n_interpolated_windows} channel-interpolated "
                    f"(kept), {n_dropped} dropped entirely.")

        # --- Rebuild eeg_windows: for kept-but-interpolated windows, redo spatial_mapping()
        # on the corrected channel-space data so the 16x16 grid reflects the fix.
        # MEMORY: mutates eeg_windows IN PLACE (no .copy()) -- the caller (extract_windows)
        # does not need the pre-correction version after this call returns, so duplicating
        # this array (often the single largest array per subject, since it's the full
        # 16x16 spatial grid) would only waste memory for no benefit. Same reasoning for
        # eeg_channel_space_after below: only allocated when qc_dir is actually provided,
        # since it exists solely to feed the optional before/after PSD comparison plots. ---
        # priority-E1: this corrected channel-space array is now ALWAYS built
        # (previously it was `eeg_channel_space_after`, allocated ONLY when
        # qc_dir was provided, since its sole purpose used to be feeding the
        # optional before/after PSD comparison plots). It is the single
        # authoritative "final, AutoReject-corrected, per-window channel-space
        # EEG" array -- the exact input eeg_connectivity.py's PLV extraction
        # needs, and it must exist regardless of whether qc_dir happens to be
        # set, since connectivity extraction is not a QC-only feature.
        eeg_windows_corrected = eeg_windows
        eeg_channel_space_corrected = eeg_channel_space_windows.copy()
        eeg_channel_space_after = eeg_channel_space_corrected if qc_dir is not None else None
        cleaned_data = epochs_clean.get_data()  # (n_kept_by_epochs_object, 28, win_samp) --
                                                  # NOTE: mne.Epochs/EpochsArray + AutoReject's
                                                  # fit_transform by default DROPS bad epochs from
                                                  # its OWN returned object, so cleaned_data's first
                                                  # axis does NOT directly index-align with the
                                                  # original n_windows if any were dropped. We handle
                                                  # this by iterating the ORIGINAL indices and only
                                                  # pulling from cleaned_data for indices that survived
                                                  # (i.e. not bad_epoch_mask), tracking our own
                                                  # running pointer into cleaned_data.
        clean_ptr = 0
        for orig_idx in range(n_windows):
            if bad_epoch_mask[orig_idx]:
                continue  # this window will be dropped below; no need to correct it
            if reject_log.labels[orig_idx].any() and (reject_log.labels[orig_idx] == 1).any():
                # this window had >=1 channel interpolated by AutoReject -- rebuild its
                # spatial-grid representation from the corrected channel-space data
                corrected_channel_slice = cleaned_data[clean_ptr]   # (28, win_samp_eeg)
                corrected_spatial = spatial_mapping(corrected_channel_slice.T, eeg_pos_3d)
                eeg_windows_corrected[orig_idx] = corrected_spatial.transpose(1, 2, 0).astype(np.float32)
                eeg_channel_space_corrected[orig_idx] = corrected_channel_slice
                if eeg_channel_space_after is not None:
                    eeg_channel_space_after[orig_idx] = corrected_channel_slice
            clean_ptr += 1

        # Free large intermediates no longer needed before continuing -- epochs/epochs_clean
        # each internally hold their own (n_windows, 28, win_samp) copy, and are not needed
        # past this point. Explicit del + gc.collect() encourages the memory back to the OS
        # promptly within THIS worker process, rather than waiting for Python's garbage
        # collector to get around to it -- matters more here than usual since multiple
        # worker processes are running this same function concurrently.
        del epochs, epochs_clean, cleaned_data
        import gc
        gc.collect()

        keep_mask = ~bad_epoch_mask
        eeg_windows_final = eeg_windows_corrected[keep_mask]
        fnirs_windows_final = fnirs_windows[keep_mask]
        labels_final = labels[keep_mask]
        session_ids_final = session_ids[keep_mask]
        trial_ids_final = trial_ids[keep_mask]
        eeg_channel_space_final = eeg_channel_space_corrected[keep_mask]  # priority-E1

        # --- Save verification artifacts (per the 4-point check discussed with the user) ---
        if qc_dir is not None:
            try:
                os.makedirs(qc_dir, exist_ok=True)
                # 1) reject_log heatmap (windows x channels: 0=good,1=interpolated,2=dropped)
                fig, ax = plt.subplots(figsize=(10, max(4, 0.02 * n_windows)))
                im = ax.imshow(reject_log.labels, aspect="auto", cmap="viridis",
                                vmin=0, vmax=2, interpolation="nearest")
                ax.set_xlabel("Channel index")
                ax.set_ylabel("Window index")
                ax.set_title(f"{subject_id} - AutoReject window x channel labels "
                             f"(0=good, 1=interpolated, 2=dropped)")
                fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
                plt.tight_layout()
                fig.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_window_heatmap.png"), dpi=150)
                plt.close(fig)

                # 3) class-balance check: label distribution before vs after
                labels_before_counts = np.bincount(labels).tolist()
                labels_after_counts = np.bincount(labels_final, minlength=len(labels_before_counts)).tolist()
                summary = {
                    "subject_id": subject_id,
                    "n_windows_before": int(n_windows),
                    "n_windows_after": int(eeg_windows_final.shape[0]),
                    "n_dropped": n_dropped,
                    "n_channel_interpolated": n_interpolated_windows,
                    "n_untouched": n_untouched,
                    "labels_distribution_before": labels_before_counts,
                    "labels_distribution_after": labels_after_counts,
                }
                with open(os.path.join(qc_dir, f"{subject_id}_autoreject_windows_summary.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

                # 5) PSD before/after AutoReject, averaged across all KEPT windows -- lets you
                # compare the frequency content of the windows AutoReject actually corrected
                # (channel-interpolated) against their original, uncorrected version, directly
                # answering "did this step actually change the frequency content, and how".
                # Only kept windows are used (dropped windows have nothing to compare -- they
                # no longer exist in the output).
                kept_before = eeg_channel_space_windows[keep_mask]   # (n_kept, 28, win_samp)
                kept_after = eeg_channel_space_after[keep_mask]
                ch_names_eeg = eeg_only_info["ch_names"]
                sfreq = float(sfreq_eeg)
                n_fft = min(int(sfreq * 4), kept_before.shape[-1])

                def _avg_psd_db(window_stack):
                    # window_stack: (n_windows, n_channels, n_times) -> mean PSD in dB per channel
                    psd, freqs = mne.time_frequency.psd_array_welch(
                        window_stack.reshape(-1, window_stack.shape[-1]),
                        sfreq=sfreq, fmin=0.5, fmax=min(49.0, sfreq / 2 - 1),
                        n_fft=n_fft, verbose=False,
                    )
                    n_ch = window_stack.shape[1]
                    psd = psd.reshape(window_stack.shape[0], n_ch, -1).mean(axis=0)
                    return freqs, 10 * np.log10(np.maximum(psd, 1e-24) * 1e12)

                freqs_win, psd_before_db = _avg_psd_db(kept_before)
                _, psd_after_db = _avg_psd_db(kept_after)

                fig, axes = plt.subplots(1, 2, figsize=(14, max(6, 0.3 * len(ch_names_eeg))),
                                          sharey=True)
                im0 = axes[0].imshow(psd_before_db, aspect="auto", origin="lower", cmap="viridis",
                                      extent=[freqs_win[0], freqs_win[-1], 0, len(ch_names_eeg)])
                axes[0].set_yticks(np.arange(len(ch_names_eeg)) + 0.5)
                axes[0].set_yticklabels(ch_names_eeg, fontsize=6)
                axes[0].set_xlabel("Frequency (Hz)")
                axes[0].set_ylabel("Channel")
                axes[0].set_title(f"{subject_id} - avg window PSD BEFORE AutoReject")

                im1 = axes[1].imshow(psd_after_db, aspect="auto", origin="lower", cmap="viridis",
                                      extent=[freqs_win[0], freqs_win[-1], 0, len(ch_names_eeg)],
                                      vmin=im0.get_clim()[0], vmax=im0.get_clim()[1])
                axes[1].set_yticks(np.arange(len(ch_names_eeg)) + 0.5)
                axes[1].set_yticklabels(ch_names_eeg, fontsize=6)
                axes[1].set_xlabel("Frequency (Hz)")
                axes[1].set_title(f"{subject_id} - avg window PSD AFTER AutoReject "
                                  f"(same color scale as BEFORE)")
                fig.colorbar(im1, ax=axes, label="Power (dB, 10log10 uV^2/Hz)")
                fig.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_psd_before_after.png"),
                           dpi=150, bbox_inches="tight")
                plt.close(fig)
                log_message(f"  {subject_id}: saved AutoReject before/after PSD comparison "
                            f"({kept_before.shape[0]} kept windows averaged)")

                # --- Remaining frequency-domain plots, same style/set as the per-recording
                # PSD QC plots saved earlier by save_psd_qc_plots(), but computed on the
                # window-averaged before/after-AutoReject data instead of the continuous
                # recording -- lets you compare the SAME kind of plot at both pipeline
                # stages (continuous-signal cleaning vs window-level AutoReject). ---
                fig_hb, ax_hb = plt.subplots(figsize=(12, max(6, 0.3 * len(ch_names_eeg))))
                im_hb = ax_hb.imshow(psd_before_db, aspect="auto", origin="lower", cmap="viridis",
                                      extent=[freqs_win[0], freqs_win[-1], 0, len(ch_names_eeg)])
                ax_hb.set_yticks(np.arange(len(ch_names_eeg)) + 0.5)
                ax_hb.set_yticklabels(ch_names_eeg, fontsize=6)
                ax_hb.set_xlabel("Frequency (Hz)")
                ax_hb.set_ylabel("Channel")
                ax_hb.set_title(f"{subject_id} - avg window PSD Heatmap (BEFORE AutoReject)")
                fig_hb.colorbar(im_hb, ax=ax_hb, label="Power (dB, 10log10 uV^2/Hz)")
                plt.tight_layout()
                fig_hb.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_psd_heatmap_before.png"), dpi=150)
                plt.close(fig_hb)

                fig_ha, ax_ha = plt.subplots(figsize=(12, max(6, 0.3 * len(ch_names_eeg))))
                im_ha = ax_ha.imshow(psd_after_db, aspect="auto", origin="lower", cmap="viridis",
                                      extent=[freqs_win[0], freqs_win[-1], 0, len(ch_names_eeg)])
                ax_ha.set_yticks(np.arange(len(ch_names_eeg)) + 0.5)
                ax_ha.set_yticklabels(ch_names_eeg, fontsize=6)
                ax_ha.set_xlabel("Frequency (Hz)")
                ax_ha.set_ylabel("Channel")
                ax_ha.set_title(f"{subject_id} - avg window PSD Heatmap (AFTER AutoReject)")
                fig_ha.colorbar(im_ha, ax=ax_ha, label="Power (dB, 10log10 uV^2/Hz)")
                plt.tight_layout()
                fig_ha.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_psd_heatmap_after.png"), dpi=150)
                plt.close(fig_ha)

                bands = {"delta(0.5-4)": (0.5, 4), "theta(4-8)": (4, 8), "alpha(8-13)": (8, 13),
                         "beta(13-30)": (13, 30), "gamma(30-49)": (30, 49)}
                band_means_after = {}
                for band_name, (lo, hi) in bands.items():
                    mask_b = (freqs_win >= lo) & (freqs_win < hi)
                    band_means_after[band_name] = psd_after_db[:, mask_b].mean(axis=1) if mask_b.any() \
                        else np.full(len(ch_names_eeg), np.nan)
                fig_bp, ax_bp = plt.subplots(figsize=(max(12, 0.35 * len(ch_names_eeg)), 6))
                x_bp = np.arange(len(ch_names_eeg))
                width_bp = 0.8 / len(bands)
                for i, (band_name, vals) in enumerate(band_means_after.items()):
                    ax_bp.bar(x_bp + i * width_bp, vals, width=width_bp, label=band_name)
                ax_bp.set_xticks(x_bp + width_bp * (len(bands) - 1) / 2)
                ax_bp.set_xticklabels(ch_names_eeg, rotation=90, fontsize=6)
                ax_bp.set_ylabel("Power (dB, 10log10 uV^2/Hz)")
                ax_bp.set_title(f"{subject_id} - avg window per-channel band power (AFTER AutoReject)")
                ax_bp.legend(fontsize=8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.15))
                plt.tight_layout()
                fig_bp.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_band_power_after.png"), dpi=150)
                plt.close(fig_bp)

                # --- Time-domain all-channels plots, before/after AutoReject, full + zoom.
                # NOTE: windows are concatenated end-to-end in trial/window order (NOT the
                # original continuous recording timeline -- consecutive windows can overlap
                # or skip time in the real recording due to the sliding-window extraction).
                # This is clearly labeled "concatenated kept windows" below, not "raw
                # recording time", so it isn't mistaken for the continuous-signal QC plots
                # saved earlier by save_psd_qc_plots()/preprocess_eeg(). It still shows
                # exactly what AutoReject changed, window by window, which is the point of
                # this comparison. ---
                def _plot_windows_concatenated(window_stack, ch_names_ts, sfreq_ts, title, color,
                                                 max_seconds=None):
                    concat = window_stack.transpose(1, 0, 2).reshape(len(ch_names_ts), -1)  # (ch, n_kept*win_samp)
                    if max_seconds is not None:
                        n_samp = int(max_seconds * sfreq_ts)
                        concat = concat[:, :n_samp]
                    t_ts = np.arange(concat.shape[1]) / sfreq_ts
                    spread_ts = np.median(np.abs(concat - np.median(concat, axis=1, keepdims=True))) * 30 + 1e-12
                    n_ch_ts = concat.shape[0]
                    fig_ts, ax_ts = plt.subplots(figsize=(14, max(6, 0.5 * n_ch_ts)))
                    for i in range(n_ch_ts):
                        offset_ts = (n_ch_ts - 1 - i) * spread_ts
                        ax_ts.plot(t_ts, concat[i] + offset_ts, color=color, linewidth=0.4)
                    ax_ts.set_yticks([(n_ch_ts - 1 - i) * spread_ts for i in range(n_ch_ts)])
                    ax_ts.set_yticklabels(ch_names_ts, fontsize=6)
                    ax_ts.set_xlabel("Time (s, concatenated kept windows -- not continuous recording time)")
                    ax_ts.set_ylabel("Channel")
                    ax_ts.set_title(title)
                    plt.tight_layout()
                    return fig_ts

                fig_tb = _plot_windows_concatenated(
                    kept_before, ch_names_eeg, sfreq,
                    f"{subject_id} - EEG BEFORE AutoReject - All Channels (concatenated kept windows)",
                    "tab:red")
                fig_tb.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_eeg_before_allchannels.png"), dpi=150)
                plt.close(fig_tb)

                fig_ta = _plot_windows_concatenated(
                    kept_after, ch_names_eeg, sfreq,
                    f"{subject_id} - EEG AFTER AutoReject - All Channels (concatenated kept windows)",
                    "tab:blue")
                fig_ta.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_eeg_after_allchannels.png"), dpi=150)
                plt.close(fig_ta)

                fig_tbz = _plot_windows_concatenated(
                    kept_before, ch_names_eeg, sfreq,
                    f"{subject_id} - EEG BEFORE AutoReject - All Channels (first 20s of concatenated kept windows)",
                    "tab:red", max_seconds=20.0)
                fig_tbz.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_eeg_before_allchannels_zoom.png"), dpi=150)
                plt.close(fig_tbz)

                fig_taz = _plot_windows_concatenated(
                    kept_after, ch_names_eeg, sfreq,
                    f"{subject_id} - EEG AFTER AutoReject - All Channels (first 20s of concatenated kept windows)",
                    "tab:blue", max_seconds=20.0)
                fig_taz.savefig(os.path.join(qc_dir, f"{subject_id}_autoreject_eeg_after_allchannels_zoom.png"), dpi=150)
                plt.close(fig_taz)

                log_message(f"  {subject_id}: saved remaining AutoReject frequency-domain plots "
                            f"(separate heatmaps, band power) and time-domain plots "
                            f"(full + zoom, before/after)")
            except Exception as e:
                log_message(f"  {subject_id}: WARNING could not save AutoReject verification "
                            f"artifacts ({e}); rejection itself still applied normally.")

        return (eeg_windows_final, fnirs_windows_final, labels_final, session_ids_final,
                trial_ids_final, eeg_channel_space_final)

    except Exception as e:
        log_message(f"  {subject_id}: AutoReject on windows FAILED ({e}); falling back to "
                    f"keeping all {n_windows} windows uncorrected (pre-AutoReject state).")
        return eeg_windows, fnirs_windows, labels, session_ids, trial_ids, eeg_channel_space_windows


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

        # Save the fully-preprocessed (0.5-50Hz + CAR + ICA) EEG in CHANNEL-SPACE (28 channels x
        # time), as a standard MNE .fif file, immediately after preprocess_eeg() returns -- this
        # is the only point in the pipeline where the cleaned signal still exists in this form;
        # spatial_mapping()/extract_windows() below convert it into the 16x16 grid tensor
        # representation, after which the channel-space Raw object is no longer recoverable.
        # Lets you reload with mne.io.read_raw_fif() later to re-inspect/QC the cleaned signal
        # (raw.plot(), PSD, topomaps, custom epoching, etc.) WITHOUT re-running ICA (the slowest
        # step, ~1-8 min/subject per the run log) or any other preprocessing step.
        try:
            filtered.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_clean_raw.fif"),
                          overwrite=True, verbose=False)
            log_message(f"  {subject_id}: saved cleaned channel-space EEG ({subject_id}_eeg_clean_raw.fif)")
        except Exception as e:
            log_message(f"  {subject_id}: could not save .fif: {e}")

        eeg_pos_3d = get_eeg_positions(filtered, eeg_mnt_struct)
        n_eeg_channels_used = eeg_pos_3d.shape[1]   # real record of how many channels actually went into spatial_mapping()

        fnirs_grid, fnirs_ev_time, fnirs_ev_code = preprocess_fnirs(subject_id, sub_out_dir)
        log_message(f"  {subject_id}: fNIRS preprocessed, shape {fnirs_grid.shape}")

        eeg_windows, fnirs_windows, labels, session_ids, trial_ids, eeg_channel_space_final = extract_windows(
            filtered, eeg_pos_3d, events, fnirs_grid, fnirs_ev_time, fnirs_ev_code, subject_id,
            qc_dir=sub_out_dir,
        )

        if eeg_windows.size == 0:
            log_message(f"  {subject_id}: no windows extracted, skipping save.")
            return

        # NOTE: no normalization is applied here anymore. Per-channel, per-subject z-score
        # normalization is now applied earlier in the pipeline -- inside preprocess_eeg() (on the
        # continuous, filtered+CAR+ICA-cleaned EEG signal) and inside preprocess_fnirs() (on the
        # continuous, filtered+resampled HbO/HbR signal) -- BEFORE spatial_mapping/windowing, so
        # eeg_windows/fnirs_windows arriving here are already normalized per-channel. This replaces
        # the previous global-tensor z-score that was applied here (see corrected-ordering notes in
        # preprocess_eeg()/preprocess_fnirs() for the full rationale).

        np.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_windows.npy"), eeg_windows)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_fnirs_windows.npy"), fnirs_windows)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_labels.npy"), labels)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_session_ids.npy"), session_ids)
        np.save(os.path.join(sub_out_dir, f"{subject_id}_trial_ids.npy"), trial_ids)

        # priority-E1: EEG functional-connectivity (PLV, theta+alpha) features,
        # computed on eeg_channel_space_final -- the SAME final, AutoReject-
        # corrected, per-window channel-space array (28 channels x 2000
        # samples @ 200Hz per window) that eeg_windows.npy's 16x16 spatial
        # grid was itself built from, so window indices/order/count are
        # GUARANTEED to line up 1:1 with eeg_windows.npy/labels.npy/etc. --
        # no separate re-derivation, no risk of index drift. See
        # eeg_connectivity.py for the PLV math (numerically verified against
        # phase-locked/independent-noise synthetic signals in
        # test_e1_plv_connectivity.py) and ENABLE_EEG_CONNECTIVITY below to
        # disable this step entirely (e.g. for a quick re-run where E1 isn't
        # needed) without touching anything else in this function.
        if ENABLE_EEG_CONNECTIVITY:
            try:
                from eeg_connectivity import batch_connectivity_features
                conn_feats = batch_connectivity_features(eeg_channel_space_final, sfreq_eeg)
                np.save(os.path.join(sub_out_dir, f"{subject_id}_eeg_connectivity.npy"), conn_feats)
                log_message(f"  {subject_id}: saved EEG connectivity features "
                            f"(priority-E1), shape {conn_feats.shape}")
            except Exception as e:
                log_message(f"  {subject_id}: WARNING could not compute/save EEG connectivity "
                            f"features ({e}); {subject_id}_eeg_connectivity.npy NOT written -- "
                            f"NBackDataset will fall back to use_fc_features=False behavior for "
                            f"this subject if this flag is later requested.")

        metadata = {
            "subject_id": subject_id,
            "eeg_shape": list(eeg_windows.shape),
            "fnirs_shape": list(fnirs_windows.shape),
            "labels_distribution": np.bincount(labels).tolist(),
            "window_len_eeg_s": WINDOW_LEN_EEG,
            "window_len_fnirs_segment_s": WINDOW_LEN_FNIRS_SEGMENT,
            "autoreject_windows_enabled": ENABLE_AUTOREJECT_WINDOWS,
            "autoreject_min_windows_threshold": AUTOREJECT_MIN_WINDOWS,
            "window_step_s": WINDOW_STEP,
            "slide_tmin_s": SLIDE_TMIN,
            "slide_tmax_s": SLIDE_TMAX,
            "fnirs_num_segments": FNIRS_NUM_SEGMENTS,
            "sfreq_eeg": SFREQ_EEG,
            "sfreq_fnirs": SFREQ_FNIRS,
            "eeg_bandpass_hz": [EEG_BP_LOW, EEG_BP_HIGH],
            "eeg_bandpass_order": EEG_BP_ORDER,
            "fnirs_bandpass_enabled": ENABLE_FNIRS_BANDPASS,      # ablation #5 flag
            "fnirs_bandpass_hz": [FNIRS_BP_LOW, FNIRS_BP_HIGH] if ENABLE_FNIRS_BANDPASS else None,
            "fnirs_bandpass_design_order": FNIRS_BP_ORDER if ENABLE_FNIRS_BANDPASS else None,
            "fnirs_bandpass_effective_order": FNIRS_BP_EFFECTIVE_ORDER,  # true zero-phase order after filtfilt
            "baseline_window_s": [BASELINE_TMIN, BASELINE_TMAX],
            "trials_per_session": TRIALS_PER_SESSION,
            "n_eeg_channels_used_for_spatial_mapping": n_eeg_channels_used,   # CONFIRMED 28 (not the
            # paper's literal 30, Sec. 2.3 "30 x 600") via direct inspection of VP001's real
            # cnt_nback.mat: 28 real scalp electrodes (Fp1..O2) + HEOG + VEOG = 30 total channels,
            # but TP9 (reference)/TP10 (ground) are not recorded as separate data channels at all.
            # Recorded per-subject here in case any subject's montage ever differs (not expected).
            "dataset_already_beer_lambert": DATASET_ALREADY_BEER_LAMBERT,
            "grid_size": GRID_SIZE,
            "normalization": (
                "per-channel, per-subject z-score, applied on the continuous channel-space signal "
                "BEFORE spatial mapping/windowing (EEG: after ICA; fNIRS: after bandpass+resample, "
                "before baseline correction) -- NOT a global z-score over the final window tensor"
            ),
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
    log_message("===== PIPELINE STARTED (TSFNet paper-exact, n-back, extended windows) =====")
    # ADDED: make it unmistakable whether --subjects/--workers were actually received (vs.
    # silently falling back to the full-26-subject/N_WORKERS=6 default because the script was
    # launched from an IDE "Run" button, a bare double-click, or a %run without the flags on
    # the same line -- none of which forward sys.argv the way a terminal invocation does).
    if _cli_args.subjects is not None or _cli_args.workers is not None:
        log_message(f"[CLI] Flags RECEIVED: --subjects={_cli_args.subjects!r} "
                    f"--workers={_cli_args.workers!r} -> running SUBJECTS={SUBJECTS}, "
                    f"N_WORKERS={N_WORKERS}")
    else:
        log_message(f"[CLI] No --subjects/--workers flags received (running with sys.argv="
                    f"{sys.argv!r}) -> using DEFAULTS: all {len(SUBJECTS)} subjects, "
                    f"N_WORKERS={N_WORKERS}. If you intended to run a subset, the flags were "
                    f"not forwarded by however this script was launched (e.g. an IDE Run "
                    f"button, a bare double-click, or a %run without the flags appended) -- "
                    f"re-run from a terminal/cmd as: python <this_script>.py --subjects 21 "
                    f"--workers 1")
    log_message(f"EEG: {EEG_BP_LOW}-{EEG_BP_HIGH}Hz order-{EEG_BP_ORDER} Butterworth, CAR, ICA-EOG")
    if ENABLE_FNIRS_BANDPASS:
        log_message(f"fNIRS: {FNIRS_BP_LOW}-{FNIRS_BP_HIGH}Hz bandpass, design order={FNIRS_BP_ORDER} "
                    f"(effective zero-phase order={FNIRS_BP_EFFECTIVE_ORDER}), "
                    f"baseline [{BASELINE_TMIN},{BASELINE_TMAX}]s")
    else:
        log_message(f"fNIRS: bandpass DISABLED (ablation #5 -- MSTFDN large-kernel argument), "
                    f"baseline [{BASELINE_TMIN},{BASELINE_TMAX}]s")
    log_message(f"Windows: EEG {WINDOW_LEN_EEG}s / fNIRS-segment {WINDOW_LEN_FNIRS_SEGMENT}s, "
                f"step {WINDOW_STEP}s, slide [{SLIDE_TMIN},{SLIDE_TMAX}]s, "
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
