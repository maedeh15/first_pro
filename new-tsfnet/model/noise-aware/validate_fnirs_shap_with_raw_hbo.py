"""
validate_fnirs_shap_with_raw_hbo.py -- item 4.2's missing piece: does the
model's fNIRS segment-importance pattern (SHAP/EGTA, monotonically rising
with segment index -- seen in all 3 of VP001's sessions) line up with the
REAL hemodynamic response function (HRF) in the raw, pre-model HbO signal?

DATA SOURCES (both already produced by Preproc_tsfnet_nback_badch.py, no
new preprocessing run needed):
  1. "{subject_id}_fnirs_clean_raw.fif" in the subject's output folder --
     the CONTINUOUS, cleaned (Beer-Lambert + bandpass + resample) but
     UN-normalized HbO/HbR signal, in real mmol/L units (see the comment
     block above hbo_mapped=spatial_mapping(...) in preprocess_fnirs: this
     file is saved BEFORE spatial_mapping/windowing/z-score on purpose for
     exactly this kind of validation).
  2. Trial onset times/codes, re-derived with the SAME function the real
     pipeline uses (load_fnirs_data), from the subject's raw .mat files --
     NOT re-implemented by hand, to avoid the exact kind of "reconstructed
     from memory" timing bug the project's engineering rules warn about.

KEY SIMPLIFYING ASSUMPTION (stated explicitly, not hidden):
The SHAP/EGTA segment-importance plots aggregate over EVERY sliding-window
offset (SLIDE_TMIN=-2s to SLIDE_TMAX=42s, step 1s) pooled together, so
"segment index i" does not correspond to one fixed absolute post-trial time
across the whole plot. This script instead uses the offset=0 window (the
window starting exactly at trial onset) as a representative reference,
since that is the single most interpretable case: here segment i covers
[i*WINDOW_STEP, i*WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT] seconds after
trial onset. This is an approximation, not an exact reproduction of what
the SHAP plot averaged over -- stated here so it isn't mistaken for more
than it is.

USAGE
    %run validate_fnirs_shap_with_raw_hbo.py
Run this in the same kernel/folder as Preproc_tsfnet_nback_badch.py (it
imports load_fnirs_data, FNIRS_TRIAL_LABEL_MAP, and the window-timing
constants directly from it, rather than re-typing them).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import mne

from Preproc_tsfnet_nback_badch import (
    load_fnirs_data, FNIRS_TRIAL_LABEL_MAP, EEG_TRIAL_LABEL_MAP,
    SFREQ_FNIRS, WINDOW_STEP, WINDOW_LEN_FNIRS_SEGMENT, FNIRS_NUM_SEGMENTS,
    BASELINE_TMIN, BASELINE_TMAX, TRIALS_PER_SESSION,
)
# BUGFIX: TRIALS_PER_SESSION was a manually-set None/placeholder before, which silently
# pooled ALL 27 trials (all 3 sessions) into one curve instead of the 9 trials/session the
# real pipeline uses (session_id = trial_idx // TRIALS_PER_SESSION + 1, TRIALS_PER_SESSION=9
# in Preproc_tsfnet_nback_badch.py). Importing it directly means this can never silently
# drift out of sync with the real preprocessing pipeline again.

# ---- EDIT THESE ----
SUBJECT_ID = "VP001"
SUBJECT_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
SESSIONS_TO_PLOT = [1, 2, 3]   # all three, one subplot each, for direct comparison against
                                # the three SHAP/EGTA segment-importance plots you already have
HBO_CHANNEL_INDEX = None       # None = average across ALL HbO channels; or an int for one specific channel
# ----------------------

fif_path = os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_fnirs_clean_raw.fif")
if not os.path.exists(fif_path):
    raise FileNotFoundError(
        f"{fif_path} not found -- this file is written by preprocess_fnirs() during the normal "
        f"preprocessing run (see its 'saved cleaned channel-space fNIRS' log line). If it's "
        f"missing, preprocessing for this subject may predate that save step, or qc_dir differs "
        f"from SUBJECT_DIR -- check your preprocessing log for the actual path it used."
    )

raw = mne.io.read_raw_fif(fif_path, preload=True, verbose=False)
hbo_picks = mne.pick_types(raw.info, fnirs="hbo") if "hbo" in raw.get_channel_types() else \
            [i for i, t in enumerate(raw.get_channel_types()) if t == "hbo"]
hbo_data = raw.get_data(picks=hbo_picks)   # (n_hbo_channels, n_times), real mmol/L
sfreq = raw.info["sfreq"]
assert abs(sfreq - SFREQ_FNIRS) < 1e-6, f"raw.fif sfreq ({sfreq}) != SFREQ_FNIRS ({SFREQ_FNIRS}) -- check for a stale .fif from an older preprocessing run."

if HBO_CHANNEL_INDEX is not None:
    hbo_signal = hbo_data[HBO_CHANNEL_INDEX]
    chan_desc = f"channel {HBO_CHANNEL_INDEX} ({raw.ch_names[hbo_picks[HBO_CHANNEL_INDEX]]})"
else:
    hbo_signal = hbo_data.mean(axis=0)
    chan_desc = f"mean across all {len(hbo_picks)} HbO channels"

print(f"Loaded raw HbO: {chan_desc}, {hbo_signal.shape[0]} samples @ {sfreq} Hz "
      f"({hbo_signal.shape[0]/sfreq:.1f} s total)")

# ---- Re-derive trial onset times/codes with the REAL pipeline function (not re-typed) ----
_, _, _, _, _, fnirs_ev_time, fnirs_ev_code = load_fnirs_data(SUBJECT_ID)
mask = np.isin(fnirs_ev_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
times_s = np.asarray(fnirs_ev_time)[mask].astype(float) / 1000.0
codes = np.asarray(fnirs_ev_code)[mask]
order = np.argsort(times_s)
trial_times = times_s[order]
trial_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in codes[order]])
print(f"Found {len(trial_times)} fNIRS trial onsets total.")

# ---- Epoch the raw HbO, offset=0 window (trial-onset-aligned), baseline-corrected ----
# segment i covers [i*WINDOW_STEP, i*WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT] s post-onset
# (see module docstring for why offset=0 is used as the representative case).
win_len_samples = int(round(WINDOW_LEN_FNIRS_SEGMENT * SFREQ_FNIRS))
task_window_samples = int(round((FNIRS_NUM_SEGMENTS * WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT) * SFREQ_FNIRS))
class_names = {0: "0-back", 1: "2-back", 2: "3-back"}
colors = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}

session_ids_all = np.arange(len(trial_times)) // TRIALS_PER_SESSION + 1

fig, axes = plt.subplots(1, len(SESSIONS_TO_PLOT), figsize=(6 * len(SESSIONS_TO_PLOT), 5), sharey=True)
if len(SESSIONS_TO_PLOT) == 1:
    axes = [axes]

all_seg_mean_hbo = {}   # session -> {class: seg_vals}

for ax, session in zip(axes, SESSIONS_TO_PLOT):
    sess_mask = session_ids_all == session
    trial_times_sess = trial_times[sess_mask]
    trial_classes_sess = trial_classes[sess_mask]
    print(f"\nSession {session}: {len(trial_times_sess)} trials "
          f"(TRIALS_PER_SESSION={TRIALS_PER_SESSION}).")

    per_trial_curves = {0: [], 1: [], 2: []}
    for t0, cls in zip(trial_times_sess, trial_classes_sess):
        b_start = int(round((t0 + BASELINE_TMIN) * SFREQ_FNIRS))
        b_end = int(round((t0 + BASELINE_TMAX) * SFREQ_FNIRS))
        onset_sample = int(round(t0 * SFREQ_FNIRS))
        end_sample = onset_sample + task_window_samples
        if b_start < 0 or end_sample > len(hbo_signal):
            continue
        baseline = hbo_signal[b_start:b_end].mean()
        trial_curve = hbo_signal[onset_sample:end_sample] - baseline
        per_trial_curves[int(cls)].append(trial_curve)

    seg_mean_hbo = {}
    for cls, curves in per_trial_curves.items():
        if not curves:
            print(f"  [warn] class {class_names[cls]}: 0 usable trials in session {session}")
            continue
        curves = np.stack(curves)
        mean_curve = curves.mean(axis=0)
        seg_vals = []
        for seg_idx in range(FNIRS_NUM_SEGMENTS):
            s_start = int(round(seg_idx * WINDOW_STEP * SFREQ_FNIRS))
            s_end = s_start + win_len_samples
            seg_vals.append(mean_curve[s_start:s_end].mean())
        seg_mean_hbo[cls] = np.array(seg_vals)
        print(f"  class {class_names[cls]}: {len(curves)} trials averaged")
        ax.plot(range(FNIRS_NUM_SEGMENTS), seg_vals, "o-", color=colors[cls], label=class_names[cls])

    all_seg_mean_hbo[session] = seg_mean_hbo
    ax.set_xlabel("fNIRS segment index")
    ax.set_title(f"Session {session} (n per class shown above)")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend()

axes[0].set_ylabel("Mean raw HbO, baseline-corrected (mmol/L)")
fig.suptitle(f"RAW HbO per segment -- {SUBJECT_ID}, all requested sessions\n"
             f"Compare each panel's SHAPE directly against that session's SHAP/EGTA segment-importance plot "
             f"(offset=0 window; see script docstring for this approximation)")
plt.tight_layout()
out_path = os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_all_sessions_raw_hbo_validation.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_path}")

print("\n===== HOW TO READ THIS AGAINST YOUR SHAP/EGTA PLOTS =====")
print("The model's SHAP/EGTA segment-importance rose monotonically with segment index in all")
print("3 sessions (later segments = more important). The biologically expected pattern is a")
print("delayed rise-then-plateau (HRF typically peaks ~4-8s after task onset, i.e. around")
print(f"segment {int(5/WINDOW_STEP)}-{int(8/WINDOW_STEP)} here), NOT a rise across the entire 21-segment window.")
print("If the raw HbO curve above peaks early (segments ~4-8) while SHAP kept rising through")
print("segment 20, the model's importance pattern is NOT fully explained by the HRF shape alone")
print("-- it may also be picking up slower trends (sustained hemodynamic response during a long")
print("task block, or baseline drift) rather than purely the classic transient HRF peak. State")
print("this nuance explicitly rather than claiming a clean HRF match either way.")
