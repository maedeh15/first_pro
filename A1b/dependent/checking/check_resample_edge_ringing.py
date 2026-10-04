# ============================================================================
# check_resample_edge_ringing.py
#
# DIAGNOSTIC SCRIPT -- does NOT touch or import the main preprocessing pipeline.
#
# Purpose: check whether scipy.signal.resample() (FFT-based, assumes periodicity)
# introduces detectable edge-ringing (Gibbs-type) artifacts near the start/end
# of the continuous EEG and fNIRS recordings, compared to scipy.signal.resample_poly
# (polyphase FIR, does NOT assume periodicity) run on the SAME raw data.
#
# Method:
#   1. Load raw EEG (cnt_nback.mat) and raw fNIRS (cnt_nback.mat oxy/deoxy) for one
#      or more subjects, exactly as the main pipeline's loaders do (same struct
#      access), WITHOUT importing the main pipeline.
#   2. Resample with BOTH scipy.signal.resample (current pipeline method) AND
#      scipy.signal.resample_poly (reference / cleaner method).
#   3. For each resampled version, take a short window near the START edge, a
#      short window near the END edge, and a short window from the MIDDLE, and
#      compute each window's power spectral density (Welch's method).
#   4. Compare high-frequency energy (top ~25% of the Nyquist band) at edges vs
#      middle, for BOTH resampling methods. If scipy.signal.resample shows a
#      much bigger edge-vs-middle high-frequency energy ratio than resample_poly
#      does, that is direct evidence of edge ringing caused specifically by the
#      FFT-periodicity assumption (not just generic edge effects that any
#      windowed PSD estimate would show).
#   5. Also directly measure the "wrap-around jump" size (|first_sample - last_sample|)
#      of the ORIGINAL raw signal per channel, since larger jumps predict worse
#      ringing (this is what makes the artifact SUBJECT-DEPENDENT).
#
# Outputs:
#   - printed summary table
#   - check_resample_edge_ringing_report.csv
#   - check_resample_edge_ringing_report.png  (PSD comparison plots)
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH and SUBJECT_IDS below.
#   2. Run: python check_resample_edge_ringing.py
#   3. Read the printed table: look at the "edge_ratio" columns. If
#      edge_ratio(resample) >> edge_ratio(resample_poly) (e.g. more than ~2x),
#      that subject/modality shows a real, method-specific ringing artifact.
# ============================================================================

import os
import numpy as np
import scipy.io as sio
from scipy.signal import resample, resample_poly, welch
import matplotlib.pyplot as plt
import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_IDS = ["VP001", "VP002", "VP003"]   # start small; extend once you trust the script

SFREQ_EEG_TARGET = 200.0
SFREQ_FNIRS_TARGET = 10.0

EDGE_WINDOW_SEC = 2.0     # length of the "near edge" window to analyze, in seconds (post-resample)
MID_WINDOW_SEC = 2.0      # length of the "middle" window to analyze, in seconds (post-resample)

# high-frequency band = top HIGH_FREQ_FRACTION of the Nyquist range, where ringing
# artifacts concentrate (they are, by construction, spurious HIGH-frequency energy)
HIGH_FREQ_FRACTION = 0.25


# ----------------------------------------------------------------------------
# Minimal, standalone loaders (mirror the main pipeline's .mat access pattern,
# but kept independent so this script never imports/depends on the pipeline file)
# ----------------------------------------------------------------------------
def load_raw_eeg_signal(subject_id):
    """Returns (data (n_channels, n_samples), sfreq_original, ch_names)."""
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt_struct = mat_cnt[real_keys[0]]
    data = np.asarray(cnt_struct.x, dtype=np.float64).T   # (n_channels, n_samples)
    sfreq = float(cnt_struct.fs)
    ch_names = [str(ch).strip() for ch in cnt_struct.clab]
    return data, sfreq, ch_names


def load_raw_fnirs_signal(subject_id):
    """Returns (oxy_raw (n_samples, n_channels), sfreq_original)."""
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt = mat_cnt[real_keys[0]]
    oxy_raw = np.asarray(cnt.oxy.x, dtype=np.float64)      # (n_samples, n_channels)
    sfreq = float(cnt.oxy.fs) if hasattr(cnt.oxy, 'fs') else 10.4
    return oxy_raw, sfreq


# ----------------------------------------------------------------------------
# Core diagnostic
# ----------------------------------------------------------------------------
def resample_both_ways(data_1d, sfreq_orig, sfreq_target):
    """data_1d: shape (n_samples,). Returns (sig_fft, sig_poly), both resampled to sfreq_target."""
    n_samples_target = int(round(len(data_1d) * sfreq_target / sfreq_orig))

    # Method A: current pipeline's method (FFT-based, assumes periodicity)
    sig_fft = resample(data_1d, n_samples_target)

    # Method B: polyphase FIR (reference method, does not assume periodicity)
    from fractions import Fraction
    frac = Fraction(sfreq_target / sfreq_orig).limit_denominator(1000)
    sig_poly = resample_poly(data_1d, frac.numerator, frac.denominator)
    # resample_poly's output length can differ slightly from resample()'s; trim to common length
    n_common = min(len(sig_fft), len(sig_poly))
    return sig_fft[:n_common], sig_poly[:n_common]


def high_freq_energy_ratio(segment, fs, high_freq_fraction=HIGH_FREQ_FRACTION):
    """Welch PSD of `segment`; returns fraction of total power lying in the
    top `high_freq_fraction` of the [0, fs/2] band."""
    nperseg = min(len(segment), 256)
    if nperseg < 8:
        return np.nan
    freqs, psd = welch(segment, fs=fs, nperseg=nperseg)
    nyq = fs / 2.0
    cutoff = nyq * (1 - high_freq_fraction)
    total_power = np.trapz(psd, freqs)
    if total_power <= 0:
        return np.nan
    hf_power = np.trapz(psd[freqs >= cutoff], freqs[freqs >= cutoff])
    return hf_power / total_power


def analyze_signal(data_1d, sfreq_orig, sfreq_target, label):
    """Runs the full edge-vs-middle comparison for one 1D channel signal.
    Returns a dict of results."""
    sig_fft, sig_poly = resample_both_ways(data_1d, sfreq_orig, sfreq_target)
    n = len(sig_fft)

    edge_n = int(EDGE_WINDOW_SEC * sfreq_target)
    mid_n = int(MID_WINDOW_SEC * sfreq_target)
    mid_start = n // 2 - mid_n // 2

    results = {}
    for name, sig in [('resample_FFT', sig_fft), ('resample_poly', sig_poly)]:
        start_seg = sig[:edge_n]
        end_seg = sig[-edge_n:]
        mid_seg = sig[mid_start: mid_start + mid_n]

        hf_start = high_freq_energy_ratio(start_seg, sfreq_target)
        hf_end = high_freq_energy_ratio(end_seg, sfreq_target)
        hf_mid = high_freq_energy_ratio(mid_seg, sfreq_target)

        edge_ratio = (max(hf_start, hf_end) / hf_mid) if (hf_mid and hf_mid > 0) else np.nan

        results[f'{name}_hf_start'] = hf_start
        results[f'{name}_hf_end'] = hf_end
        results[f'{name}_hf_mid'] = hf_mid
        results[f'{name}_edge_ratio'] = edge_ratio

    # wrap-around jump of the ORIGINAL (pre-resample) signal: bigger jump -> predicts worse FFT ringing
    wrap_jump = abs(data_1d[0] - data_1d[-1])
    signal_std = np.std(data_1d)
    results['wrap_jump_over_std'] = wrap_jump / signal_std if signal_std > 0 else np.nan
    results['label'] = label
    return results, sig_fft, sig_poly


def run_for_subject(subject_id, rows, plot_data):
    print(f"\n--- {subject_id} ---")

    # EEG: analyze a representative subset of channels (not all, to keep this fast);
    # edit `eeg_channels_to_check` if you want specific channels
    try:
        eeg_data, eeg_sfreq_orig, eeg_ch_names = load_raw_eeg_signal(subject_id)
        n_check = min(3, eeg_data.shape[0])
        for ch_idx in range(n_check):
            ch_name = eeg_ch_names[ch_idx]
            res, sig_fft, sig_poly = analyze_signal(
                eeg_data[ch_idx], eeg_sfreq_orig, SFREQ_EEG_TARGET,
                label=f"{subject_id} EEG ch={ch_name}")
            res['subject'] = subject_id
            res['modality'] = 'EEG'
            res['channel'] = ch_name
            rows.append(res)
            if ch_idx == 0:
                plot_data.append((f"{subject_id} EEG ({ch_name})", sig_fft, sig_poly, SFREQ_EEG_TARGET))
            print(f"  EEG {ch_name}: edge_ratio(FFT)={res['resample_FFT_edge_ratio']:.2f}  "
                  f"edge_ratio(poly)={res['resample_poly_edge_ratio']:.2f}  "
                  f"wrap_jump/std={res['wrap_jump_over_std']:.2f}")
    except Exception as e:
        print(f"  EEG load/analysis failed for {subject_id}: {e}")

    # fNIRS: check a couple of oxy channels
    try:
        oxy_raw, fnirs_sfreq_orig = load_raw_fnirs_signal(subject_id)
        n_check = min(2, oxy_raw.shape[1])
        for ch_idx in range(n_check):
            res, sig_fft, sig_poly = analyze_signal(
                oxy_raw[:, ch_idx], fnirs_sfreq_orig, SFREQ_FNIRS_TARGET,
                label=f"{subject_id} fNIRS ch={ch_idx}")
            res['subject'] = subject_id
            res['modality'] = 'fNIRS'
            res['channel'] = f'oxy_{ch_idx}'
            rows.append(res)
            if ch_idx == 0:
                plot_data.append((f"{subject_id} fNIRS (oxy ch{ch_idx})", sig_fft, sig_poly, SFREQ_FNIRS_TARGET))
            print(f"  fNIRS oxy_{ch_idx}: edge_ratio(FFT)={res['resample_FFT_edge_ratio']:.2f}  "
                  f"edge_ratio(poly)={res['resample_poly_edge_ratio']:.2f}  "
                  f"wrap_jump/std={res['wrap_jump_over_std']:.2f}")
    except Exception as e:
        print(f"  fNIRS load/analysis failed for {subject_id}: {e}")


def plot_edge_zoom(plot_data, out_path, n_show=4):
    """For a few (label, sig_fft, sig_poly, fs) entries, plot the first ~2s of both
    resampling methods side by side so ringing is visible by eye."""
    plot_data = plot_data[:n_show]
    if not plot_data:
        return
    fig, axes = plt.subplots(len(plot_data), 2, figsize=(12, 3 * len(plot_data)), squeeze=False)
    for i, (label, sig_fft, sig_poly, fs) in enumerate(plot_data):
        n_show_samples = int(EDGE_WINDOW_SEC * fs)
        t = np.arange(n_show_samples) / fs

        axes[i, 0].plot(t, sig_fft[:n_show_samples], color='tab:red', linewidth=0.8)
        axes[i, 0].set_title(f"{label} -- scipy.signal.resample (current pipeline)\nfirst {EDGE_WINDOW_SEC}s")
        axes[i, 0].set_xlabel("Time (s)")

        axes[i, 1].plot(t, sig_poly[:n_show_samples], color='tab:blue', linewidth=0.8)
        axes[i, 1].set_title(f"{label} -- resample_poly (reference)\nfirst {EDGE_WINDOW_SEC}s")
        axes[i, 1].set_xlabel("Time (s)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved edge zoom-in comparison to: {out_path}")


if __name__ == "__main__":
    rows = []
    plot_data = []

    for sid in SUBJECT_IDS:
        run_for_subject(sid, rows, plot_data)

    if not rows:
        print("\nNo results collected -- check BASE_DATASET_PATH / SUBJECT_IDS.")
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(__file__))

    cols_to_show = ['subject', 'modality', 'channel',
                     'resample_FFT_edge_ratio', 'resample_poly_edge_ratio',
                     'wrap_jump_over_std']
    print(f"\n{'='*90}")
    print("SUMMARY (edge_ratio = high-freq energy near edges / high-freq energy in middle)")
    print(f"{'='*90}")
    print(df[cols_to_show].to_string(index=False))

    csv_path = os.path.join(out_dir, "check_resample_edge_ringing_report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved full numeric report to: {csv_path}")

    plot_path = os.path.join(out_dir, "check_resample_edge_ringing_report.png")
    plot_edge_zoom(plot_data, plot_path)

    print("\n" + "="*90)
    print("HOW TO READ THIS:")
    print("- edge_ratio close to 1.0 means edges look spectrally like the middle -- no ringing.")
    print("- edge_ratio >> 1.0 means high-frequency energy is concentrated near the edges --")
    print("  consistent with ringing.")
    print("- THE KEY COMPARISON: look at resample_FFT_edge_ratio vs resample_poly_edge_ratio")
    print("  for the SAME channel. Both methods can show *some* edge ratio > 1 just from normal")
    print("  edge effects of a short analysis window -- that is not itself meaningful.")
    print("  What IS meaningful is resample_FFT_edge_ratio being SUBSTANTIALLY larger than")
    print("  resample_poly_edge_ratio (e.g. 2x or more) for the same channel -- that gap is")
    print("  specifically caused by the FFT method's periodicity assumption, since resample_poly")
    print("  does not make that assumption and processes the exact same raw input.")
    print("- wrap_jump_over_std is the (|first_sample - last_sample of ORIGINAL signal|) divided by")
    print("  the signal's own std. Larger values predict worse FFT-resample ringing, and if this")
    print("  varies a lot ACROSS SUBJECTS, that supports the concern that the resulting artifact")
    print("  is subject-dependent (i.e. adds systematic, non-random noise correlated with subject")
    print("  identity) rather than uniform, harmless noise.")
    print("- Also open the .png: if the red (FFT) traces show visible oscillation near t=0 that")
    print("  the blue (poly) traces do not, that is the artifact visible by eye.")
    print("="*90)
