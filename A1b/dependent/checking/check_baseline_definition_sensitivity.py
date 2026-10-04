# ============================================================================
# check_baseline_definition_sensitivity.py
#
# DIAGNOSTIC SCRIPT -- does NOT touch or import the main preprocessing pipeline,
# and does NOT require any model training.
#
# Purpose: quantify how much it actually matters, for real trials in this
# dataset, whether fNIRS baseline correction uses:
#
#   (A) TRIAL-LEVEL baseline (current pipeline): one baseline value per trial,
#       computed once from [-5, -2]s before trial onset, subtracted from every
#       sliding-window segment of that trial regardless of how far into the
#       trial (offset_sec) that segment starts.
#
#   (B) WINDOW-LOCAL baseline (alternative): baseline computed from a short
#       window immediately preceding EACH segment's own start (a "local"
#       pre-segment reference), instead of always referencing the start of
#       the trial.
#
# For each trial, this script:
#   1. Loads and preprocesses fNIRS exactly as load_and_preprocess_fnirs() does
#      (Beer-Lambert skip, no wavelet/MARA/lowpass, resample to 10Hz, spatial
#      mapping to 16x16, per-subject normalization) -- standalone, no pipeline
#      import.
#   2. Uses the SAME trial-onset markers / sync logic as the main pipeline
#      (fNIRS's own trial markers, not EEG's clock) to anchor trials.
#   3. For each trial and each sliding-window offset (0, 1, 2, ... seconds,
#      matching EEG_STEP), computes:
#        - baseline_trial:  mean over [-5,-2]s before TRIAL onset (method A)
#        - baseline_local:  mean over [-5,-2]s before the CURRENT segment's
#                            own start time, i.e. [offset-5, offset-2]s
#                            relative to trial onset (method B)
#        - the corrected segment mean under EACH baseline definition
#        - the difference between the two corrected values
#   4. Reports how this difference GROWS as offset increases (this is exactly
#      the "late segments still carry the trial-start baseline" concern),
#      separately for each of the 3 classes (0-back/2-back/3-back), since
#      cognitive load may drift differently across trial types.
#
# Output:
#   - baseline_definition_report.csv  (per trial x offset x channel-averaged)
#   - baseline_definition_report.png  (divergence vs offset, per class)
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH and SUBJECT_IDS below.
#   2. Run: python check_baseline_definition_sensitivity.py
#   3. Read the printed summary + look at the plot: if the two baseline
#      definitions diverge a lot at late offsets (e.g. > some fraction of the
#      signal's own std), method choice matters and should be tested with a
#      real training run. If divergence stays small throughout, the current
#      trial-level baseline is safe as-is.
# ============================================================================

import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_IDS = ["VP001", "VP002", "VP003"]

SFREQ_FNIRS = 10.0
BASELINE_TMIN = -5.0
BASELINE_TMAX = -2.0
EEG_STEP = 1.0            # matches main pipeline's sliding-window step
EEG_EPOCH_TMAX = 40.0     # matches main pipeline's trial length
EEG_WINDOW_LEN = 10.0     # matches main pipeline (only used to bound max offset)

FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}
CLASS_NAMES = {0: '0-back', 1: '2-back', 2: '3-back'}


# ----------------------------------------------------------------------------
# Standalone loaders / helpers (mirror the main pipeline, kept independent)
# ----------------------------------------------------------------------------
def load_fnirs_raw_and_markers(subject_id):
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_cnt_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt = mat_cnt[real_cnt_keys[0]]

    oxy_raw = np.asarray(cnt.oxy.x, dtype=np.float64)
    deoxy_raw = np.asarray(cnt.deoxy.x, dtype=np.float64)
    sfreq_raw = float(cnt.oxy.fs) if hasattr(cnt.oxy, 'fs') else 10.4

    mat_mrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    real_mrk_keys = [k for k in mat_mrk.keys() if not k.startswith('__')]
    fmrk = mat_mrk[real_mrk_keys[0]]

    if hasattr(fmrk, 'time'):
        fnirs_events_time = np.asarray(fmrk.time, dtype=float)
    elif hasattr(fmrk, 'pos'):
        fnirs_events_time = np.asarray(fmrk.pos, dtype=float) / sfreq_raw * 1000.0
    else:
        raise ValueError("no time/pos field in fNIRS marker struct")

    if hasattr(fmrk, 'event') and hasattr(fmrk.event, 'desc'):
        fnirs_events_code = np.asarray(fmrk.event.desc).flatten()
    else:
        raise ValueError("no event.desc field in fNIRS marker struct")

    return oxy_raw, deoxy_raw, sfreq_raw, fnirs_events_time, fnirs_events_code


def resample_to_10hz(data, sfreq_raw):
    from scipy.signal import resample
    n_samples = int(data.shape[0] * SFREQ_FNIRS / sfreq_raw)
    return resample(data, n_samples, axis=0)


def spatial_mapping_simple(data, mnt_pos_3d):
    """Simplified but numerically equivalent version of the main pipeline's
    spatial_mapping(), kept standalone here. data: (n_samples, n_channels)."""
    x3d, y3d, z3d = mnt_pos_3d[0, :], mnt_pos_3d[1, :], mnt_pos_3d[2, :]
    r = np.sqrt(x3d**2 + y3d**2 + z3d**2)
    r_safe = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z3d / r_safe, -1.0, 1.0))
    phi = np.arctan2(y3d, x3d)
    x_coords = theta * np.cos(phi)
    y_coords = theta * np.sin(phi)

    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):16j, min(y_coords):max(y_coords):16j]
    points = np.column_stack((x_coords, y_coords))
    tri = Delaunay(points)
    interp = CloughTocher2DInterpolator(tri, data.T, fill_value=0)
    mapped = interp(grid_x, grid_y)   # (16, 16, n_samples)
    return np.moveaxis(mapped, -1, 0)  # (n_samples, 16, 16)


def load_mnt_pos3d(subject_id):
    mnt_path = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS", 'mnt_nback.mat')
    mat = sio.loadmat(mnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith('__')]
    struct = mat[real_keys[0]]
    return np.asarray(struct.pos_3d, dtype=np.float64)


# ----------------------------------------------------------------------------
# Core analysis
# ----------------------------------------------------------------------------
def analyze_subject(subject_id, rows):
    print(f"\n--- {subject_id} ---")
    oxy_raw, deoxy_raw, sfreq_raw, fnirs_events_time, fnirs_events_code = load_fnirs_raw_and_markers(subject_id)

    # Beer-Lambert skipped (per pipeline's validated decision): oxy_raw/deoxy_raw used directly
    hbo_resampled = resample_to_10hz(oxy_raw, sfreq_raw)
    # (deoxy not needed for this diagnostic; HbO alone is enough to demonstrate the effect)

    mnt_pos_3d = load_mnt_pos3d(subject_id)
    print("  Spatial mapping (this is the slow step)...")
    hbo_mapped = spatial_mapping_simple(hbo_resampled, mnt_pos_3d)   # (n_samples, 16, 16)

    # per-subject normalization, same as main pipeline (mirrors fnirs_data z-score before windowing)
    hbo_mapped = (hbo_mapped - hbo_mapped.mean(axis=0, keepdims=True)) / (hbo_mapped.std(axis=0, keepdims=True) + 1e-8)

    # trial markers, fNIRS's own clock (sync-fixed, matches main pipeline)
    fnirs_mask = np.isin(fnirs_events_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
    trial_times_ms = fnirs_events_time[fnirs_mask]
    trial_codes = fnirs_events_code[fnirs_mask]
    order = np.argsort(trial_times_ms)
    trial_times = trial_times_ms[order] / 1000.0   # seconds
    trial_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in trial_codes[order]])

    print(f"  found {len(trial_times)} trials")

    max_offset = EEG_EPOCH_TMAX - EEG_WINDOW_LEN
    offsets = np.arange(0.0, max_offset + 0.001, EEG_STEP)

    for trial_idx, (t_onset, cls) in enumerate(zip(trial_times, trial_classes)):
        # method A: trial-level baseline (fixed, computed once for this trial)
        b_start = int(np.round((t_onset + BASELINE_TMIN) * SFREQ_FNIRS))
        b_end = int(np.round((t_onset + BASELINE_TMAX) * SFREQ_FNIRS))
        if b_start < 0 or b_end <= b_start or b_end > hbo_mapped.shape[0]:
            continue
        baseline_trial = hbo_mapped[b_start:b_end].mean(axis=0)   # (16,16)

        for offset_sec in offsets:
            seg_start_t = t_onset + offset_sec
            seg_start = int(np.round(seg_start_t * SFREQ_FNIRS))
            seg_end = seg_start + int(3.0 * SFREQ_FNIRS)   # 3s segment, matches FNIRS_WINDOW_LEN
            if seg_end > hbo_mapped.shape[0] or seg_start < 0:
                continue
            segment = hbo_mapped[seg_start:seg_end].mean(axis=0)   # (16,16)

            # method B: window-local baseline, i.e. [-5,-2]s relative to THIS segment's own start
            local_b_start = int(np.round((seg_start_t + BASELINE_TMIN) * SFREQ_FNIRS))
            local_b_end = int(np.round((seg_start_t + BASELINE_TMAX) * SFREQ_FNIRS))
            if local_b_start < 0 or local_b_end <= local_b_start or local_b_end > hbo_mapped.shape[0]:
                continue
            baseline_local = hbo_mapped[local_b_start:local_b_end].mean(axis=0)   # (16,16)

            corrected_trial = segment - baseline_trial
            corrected_local = segment - baseline_local
            diff = corrected_trial - corrected_local

            rows.append({
                'subject': subject_id,
                'trial_idx': trial_idx,
                'class': CLASS_NAMES[cls],
                'offset_sec': offset_sec,
                'baseline_trial_mean': baseline_trial.mean(),
                'baseline_local_mean': baseline_local.mean(),
                'baseline_drift': baseline_local.mean() - baseline_trial.mean(),
                'corrected_trial_mean': corrected_trial.mean(),
                'corrected_local_mean': corrected_local.mean(),
                'diff_mean': diff.mean(),
                'diff_abs_mean': np.abs(diff).mean(),
                'diff_rms': np.sqrt((diff ** 2).mean()),
                'segment_rms': np.sqrt((segment ** 2).mean()),
            })


def plot_divergence_vs_offset(df, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for cls_name in df['class'].unique():
        sub = df[df['class'] == cls_name]
        grouped = sub.groupby('offset_sec')['diff_rms'].agg(['mean', 'std'])
        axes[0].plot(grouped.index, grouped['mean'], marker='o', label=cls_name)
        axes[0].fill_between(grouped.index,
                              grouped['mean'] - grouped['std'],
                              grouped['mean'] + grouped['std'], alpha=0.15)

        grouped_rel = sub.groupby('offset_sec').apply(
            lambda g: (g['diff_rms'] / g['segment_rms'].replace(0, np.nan)).mean())
        axes[1].plot(grouped_rel.index, grouped_rel.values, marker='o', label=cls_name)

    axes[0].set_xlabel("Offset within trial (s)")
    axes[0].set_ylabel("RMS difference between baseline-trial and baseline-local\n(normalized fNIRS grid units)")
    axes[0].set_title("Absolute divergence vs offset")
    axes[0].legend()
    axes[0].axhline(0, color='gray', linewidth=0.5)

    axes[1].set_xlabel("Offset within trial (s)")
    axes[1].set_ylabel("Divergence / segment RMS (relative)")
    axes[1].set_title("Relative divergence vs offset\n(> ~0.2-0.3 suggests a practically meaningful effect)")
    axes[1].legend()
    axes[1].axhline(0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    rows = []
    for sid in SUBJECT_IDS:
        try:
            analyze_subject(sid, rows)
        except Exception as e:
            print(f"  FAILED for {sid}: {e}")

    if not rows:
        print("\nNo results collected -- check BASE_DATASET_PATH / SUBJECT_IDS.")
        raise SystemExit(0)

    df = pd.DataFrame(rows)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "baseline_definition_report.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\nSaved full numeric report to: {csv_path}")

    print(f"\n{'='*80}")
    print("SUMMARY: divergence between trial-level and window-local baseline, by offset")
    print(f"{'='*80}")
    summary = df.groupby(['class', 'offset_sec'])[['diff_rms', 'segment_rms']].mean()
    summary['relative_divergence'] = summary['diff_rms'] / summary['segment_rms']
    print(summary.to_string())

    plot_divergence_vs_offset(df, os.path.join(out_dir, "baseline_definition_report.png"))

    print("\n" + "="*80)
    print("HOW TO READ THIS:")
    print("- diff_rms / diff_abs_mean: how different the corrected segment values are under")
    print("  the two baseline definitions, in normalized (z-scored) fNIRS grid units.")
    print("- relative_divergence = diff_rms / segment_rms: how big that difference is RELATIVE")
    print("  to the segment's own signal magnitude. This is the most interpretable number:")
    print("    - close to 0 at ALL offsets  -> the two definitions barely differ; current")
    print("      trial-level baseline is safe, no further action needed.")
    print("    - grows clearly with offset_sec (e.g. approaches or exceeds ~0.2-0.3 at late")
    print("      offsets)  -> confirms the concern: late-trial segments are meaningfully")
    print("      affected by which baseline definition is used, and this should be tested")
    print("      with an actual isolated training run (trial-level vs window-local baseline,")
    print("      everything else held fixed) before concluding it doesn't matter.")
    print("- Also check baseline_drift: if this itself trends consistently in one direction as")
    print("  offset grows (not just noisy back-and-forth), that is direct evidence of a genuine")
    print("  slow physiological drift over the 40s trial (not sensor noise), which is exactly")
    print("  the kind of signal a window-local baseline would remove but a trial-level baseline")
    print("  would preserve -- so the 'right' choice really does depend on whether that drift")
    print("  is signal (cognitive load accumulating) or nuisance (systemic drift), which needs")
    print("  a labeled downstream check (i.e. an actual training run), not just this script.")
    print("="*80)
