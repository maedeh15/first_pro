# ============================================================================
# DIAGNOSTIC (read-only): was the fNIRS data already filtered BEFORE our
# pipeline ever touches it (e.g. by the NIRScout device itself or by the
# dataset authors), such that our own 0.01-0.1Hz Butterworth (now disabled in
# ablation #5) was actually mostly redundant / not doing much?
#
# This does NOT modify or depend on the preprocessing pipeline in any way --
# it independently re-loads the RAW cnt_nback.mat file (identical loading
# logic to load_fnirs_data() in the preprocessing script, copied here on
# purpose rather than imported, so this script has zero side effects and
# can't accidentally trigger the pipeline's own __main__ block / argparse /
# multiprocessing machinery just by running).
#
# METHOD: compute the power spectral density (Welch's method) of the RAW
# oxy/deoxy signal, before ANY of our processing (before Beer-Lambert check,
# before resampling, before our bandpass, before anything). Then look at:
#   1. How much power exists ABOVE 0.1Hz (our filter's cutoff) -- physiological
#      noise sources live there: respiration ~0.2-0.4Hz, heart rate ~0.8-1.5Hz
#      (aliased down from its true ~1Hz if sfreq is close to it, but the
#      device here samples at ~10Hz so heart rate itself is still resolvable).
#      If there's essentially NO power above 0.1Hz already in the RAW data,
#      that's strong evidence the device/dataset authors already low-pass
#      filtered it upstream -- meaning our own filter (and its removal) was
#      likely close to a no-op, and the big accuracy jump we saw must have a
#      DIFFERENT explanation (not "physiological noise got let through").
#   2. Whether there's a visible/sharp spectral roll-off right around 0.1-0.2Hz
#      in the RAW data (a tell-tale sign of an existing anti-alias/low-pass
#      filter already baked in), vs. a smooth natural 1/f-like decay (which
#      would indicate no prior filtering, i.e. genuinely raw physiological
#      data going into our pipeline).
#
# USAGE:
#   python check_fnirs_prefiltered.py                 # checks VP001 (default)
#   python check_fnirs_prefiltered.py --subjects 1,5,17,21   # checks a few
#
# OUTPUT: printed band-power breakdown per subject + a saved PNG per subject
# (PSD plot with the 0.01-0.1Hz target band and the 0.1Hz cutoff both marked)
# so you can visually confirm the roll-off shape yourself.
# ============================================================================
import os
import argparse

import numpy as np
import scipy.io as sio
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# numpy >=2.0 renamed trapz -> trapezoid and removed the old name; numpy <2.0 (and some
# 2.x builds) only have trapz. This works on both without knowing which numpy is installed.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Must match preprocessing's BASE_DATASET_PATH exactly (copied, not imported,
# per the "zero side effects" design above -- if you change the path in the
# preprocessing script, update it here too).
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
OUTPUT_DIR = r"E:\kntu\Thesis\codes\Results\fnirs_prefilter_check"

_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--subjects", type=str, default="1",
                          help="Comma-separated subject numbers to check, e.g. '1' or '1,5,17,21'.")
_cli_args, _ = _arg_parser.parse_known_args()
SUBJECTS = [int(s.strip()) for s in _cli_args.subjects.split(",")]


def _load_struct(mat_path):
    """Identical to preprocessing's _load_struct(): struct_as_record=False,
    squeeze_me=True so fields are accessed as plain attributes (s.oxy.x)."""
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


def load_raw_fnirs(subject_id):
    """Identical loading logic to preprocessing's load_fnirs_data(), but we
    only need oxy_raw/deoxy_raw/sfreq here -- markers/positions are irrelevant
    for a pure spectral-content check."""
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_struct = _load_struct(os.path.join(fnirs_folder, "cnt_nback.mat"))
    oxy_raw = np.asarray(cnt_struct.oxy.x, dtype=np.float64)      # (n_samples, n_channels)
    deoxy_raw = np.asarray(cnt_struct.deoxy.x, dtype=np.float64)
    sfreq_raw = float(getattr(cnt_struct.oxy, "fs", getattr(cnt_struct, "fs", 10.0)))
    return oxy_raw, deoxy_raw, sfreq_raw


def band_power(freqs, psd, lo, hi):
    """Integrate PSD over [lo, hi) Hz (trapezoidal), per channel already
    averaged into psd's shape (n_freqs,)."""
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return 0.0
    return float(_trapz(psd[mask], freqs[mask]))


def check_subject(sub_num):
    subject_id = f"VP{sub_num:03d}"
    print(f"\n{'='*70}")
    print(f"{subject_id}")
    print(f"{'='*70}")

    oxy_raw, deoxy_raw, sfreq_raw = load_raw_fnirs(subject_id)
    print(f"  Raw fNIRS: {oxy_raw.shape[0]} samples x {oxy_raw.shape[1]} channels, "
          f"fs={sfreq_raw}Hz (device-reported)")

    # PSD averaged across all channels (oxy, since that's what's usually most
    # affected by motion/respiration artifacts in fNIRS literature).
    nperseg = min(1024, oxy_raw.shape[0])
    freqs, psd_per_ch = welch(oxy_raw, fs=sfreq_raw, nperseg=nperseg, axis=0)
    psd = psd_per_ch.mean(axis=1)   # average across channels -> (n_freqs,)

    bands = [
        ("< 0.01 Hz (below target, slow drift)", 0.0, 0.01),
        ("0.01-0.1 Hz (TARGET hemodynamic band)", 0.01, 0.1),
        ("0.1-0.3 Hz (respiration range)", 0.1, 0.3),
        ("0.3-1.0 Hz (upper physiological)", 0.3, 1.0),
        ("1.0 Hz+ (heart rate range, if resolvable)", 1.0, sfreq_raw / 2),
    ]
    powers = {}
    for label, lo, hi in bands:
        p = band_power(freqs, psd, lo, min(hi, sfreq_raw / 2))
        powers[label] = p
        print(f"    {label:45s}: power={p:.6g}")

    total_above_target = sum(powers[l] for l, lo, hi in bands if lo >= 0.1)
    total_all = sum(powers.values())
    frac_above_target = total_above_target / total_all if total_all > 0 else 0.0
    print(f"  --> Fraction of total power ABOVE 0.1Hz (our filter's cutoff): "
          f"{frac_above_target*100:.2f}%")

    if frac_above_target < 0.01:
        verdict = ("Essentially NO power above 0.1Hz in the RAW data -- consistent with "
                   "the device/dataset already having a low-pass filter baked in upstream. "
                   "Our own filter (and its removal) was likely close to a no-op; the #5 "
                   "accuracy jump probably has a DIFFERENT cause -- do not attribute it to "
                   "'physiological noise now let through' without further checking.")
    elif frac_above_target < 0.05:
        verdict = ("Small but non-negligible power above 0.1Hz -- some prior band-limiting "
                   "may exist, but not total. Our filter's removal plausibly had SOME real "
                   "effect, though probably not enough alone to explain a very large jump.")
    else:
        verdict = ("Meaningful power above 0.1Hz already present in the RAW data -- the raw "
                   "signal genuinely contains content our filter was removing (respiration/"
                   "motion/cardiac range). Our filter's removal is a plausible real "
                   "contributor to the #5 accuracy change; consistent with the MSTFDN "
                   "large-kernel hypothesis being genuinely tested here.")
    print(f"  VERDICT: {verdict}")

    # Plot for visual roll-off inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(freqs, psd, color="#2a78d6")
    ax.axvspan(0.01, 0.1, color="#1baf7a", alpha=0.15, label="Target band (0.01-0.1Hz)")
    ax.axvline(0.1, color="#e24b4a", linestyle="--", label="Our filter's cutoff (0.1Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (log scale)")
    ax.set_title(f"{subject_id} - RAW fNIRS (oxy) PSD, before ANY preprocessing")
    ax.set_xlim(0, min(2.0, sfreq_raw / 2))
    ax.legend()
    out_path = os.path.join(OUTPUT_DIR, f"{subject_id}_raw_fnirs_psd.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {out_path}")

    return frac_above_target


if __name__ == "__main__":
    print(f"Checking {len(SUBJECTS)} subject(s) for pre-existing fNIRS filtering: {SUBJECTS}")
    results = {}
    for sub_num in SUBJECTS:
        try:
            results[sub_num] = check_subject(sub_num)
        except Exception as e:
            print(f"  FAILED for VP{sub_num:03d}: {e}")

    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY across subjects")
        print(f"{'='*70}")
        for sub_num, frac in results.items():
            print(f"  VP{sub_num:03d}: {frac*100:.2f}% of power above 0.1Hz")
