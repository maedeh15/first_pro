"""
priority-E1: EEG functional-connectivity (PLV) feature extraction.

Motivated by Cao et al. 2022 (Sensors, 22(19):7623 -- the SAME Shin et al.
2018 dataset this project uses) and the subject-specific PTE/MI/PLV n-back
study (PMC8541420): bivariate functional-connectivity features between EEG
channel PAIRS, computed per frequency band, substantially improved
0-back-vs-2-back/3-back separation relative to univariate per-channel power.

This module is architecture-agnostic: it operates on RAW CHANNEL-SPACE EEG
(channels x time), which in this project's pipeline exists ONLY transiently
inside extract_windows() (`eeg_channel_space_windows`, shape (N, 28, 2000)
at 200Hz for a 10s window) BEFORE spatial_mapping() interpolates it onto the
16x16 grid that TSFNet actually consumes. It is never saved to disk today --
see extract_connectivity_features.py for how to hook in and save it as a new,
parallel .npy file WITHOUT re-running PyPREP/ICA (the slow preprocessing
steps), by reusing the already-cached `{subject_id}_eeg_clean_raw.fif`.

Metric: Phase-Locking Value (PLV) with a linear FIR band-pass filter (via
scipy) + the analytic signal (Hilbert transform), the standard formulation
(Lachaux et al. 1999):

    PLV_ij = | (1/T) * sum_t exp( i * (phi_i(t) - phi_j(t)) ) |

computed separately for theta (4-8 Hz) and alpha (8-13 Hz) -- the two bands
motivating B3/E1 and directly implicated in cognitive-load level separation
in this project's cited literature (Raufi & Longo [33]; Cao et al. [8]).
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

THETA_BAND = (4.0, 8.0)
ALPHA_BAND = (8.0, 13.0)


def _bandpass(x, sfreq, band, order=4):
    """Zero-phase Butterworth band-pass filter along the last axis.
    x: (..., T). Returns array of the same shape."""
    nyq = sfreq / 2.0
    low, high = band[0] / nyq, band[1] / nyq
    b, a = butter(order, [low, high], btype="band")
    # filtfilt needs the signal length to comfortably exceed the filter's
    # padding requirement (default padlen ~3*(max(len(a),len(b))-1)); with
    # order=4 bandpass (a,b length 9) and a 2000-sample (10s @ 200Hz) window
    # this is never close to being an issue, but guard explicitly rather
    # than let scipy raise an opaque ValueError deep in a subject loop.
    padlen = 3 * (max(len(a), len(b)) - 1)
    if x.shape[-1] <= padlen:
        raise ValueError(f"signal length {x.shape[-1]} too short for filtfilt "
                          f"padlen={padlen} (band={band}, order={order})")
    return filtfilt(b, a, x, axis=-1)


def plv_matrix(window, sfreq, band):
    """
    window: (C, T) raw channel-space EEG for ONE window (one trial/offset).
    sfreq: sampling rate in Hz.
    band: (low_hz, high_hz) tuple.
    Returns: (C, C) symmetric PLV matrix, diagonal exactly 1.0.
    """
    C, T = window.shape
    filtered = _bandpass(window, sfreq, band)          # (C, T)
    analytic = hilbert(filtered, axis=-1)               # (C, T) complex
    phase = np.angle(analytic)                          # (C, T)
    # Pairwise phase differences via outer subtraction, vectorized: build
    # (C, C, T) is memory-heavy for large C; instead use the standard trick
    # PLV_ij = |mean_t( exp(i*phi_i) * exp(-i*phi_j) )| = |mean_t(u_i * conj(u_j))|
    # where u = exp(i*phase), so it reduces to a normalized complex "coherence"
    # computable via a (C,T) x (T,C) matrix product -- O(C^2 T) time, O(C^2)
    # memory, no O(C^2 T) intermediate array.
    u = np.exp(1j * phase)                               # (C, T)
    plv = np.abs(u @ u.conj().T) / T                     # (C, C)
    # Numerical symmetrization + exact unit diagonal (should already be ~1.0
    # and ~symmetric up to float error; enforce exactly for downstream code
    # that may assume it, e.g. taking only the upper triangle).
    plv = (plv + plv.T) / 2.0
    np.fill_diagonal(plv, 1.0)
    return plv


def upper_triangle_vector(mat):
    """Vectorize the strict upper triangle of a (C, C) matrix -> (C*(C-1)/2,).
    Used to avoid feeding the redundant symmetric half + trivial diagonal
    (always 1.0 for PLV) into the network."""
    C = mat.shape[0]
    iu = np.triu_indices(C, k=1)
    return mat[iu]


def window_connectivity_features(window, sfreq, bands=(THETA_BAND, ALPHA_BAND)):
    """
    window: (C, T) raw channel-space EEG for one window.
    Returns: (len(bands), C*(C-1)/2) float32 array -- one vectorized
    upper-triangle PLV vector per band, stacked.
    """
    feats = [upper_triangle_vector(plv_matrix(window, sfreq, band)) for band in bands]
    return np.stack(feats, axis=0).astype(np.float32)


def batch_connectivity_features(windows, sfreq, bands=(THETA_BAND, ALPHA_BAND)):
    """
    windows: (N, C, T) raw channel-space EEG, N windows.
    Returns: (N, len(bands), C*(C-1)/2) float32 array.
    This is the array `extract_connectivity_features.py` saves as
    `{subject_id}_eeg_connectivity.npy`.
    """
    return np.stack([window_connectivity_features(w, sfreq, bands) for w in windows], axis=0)
