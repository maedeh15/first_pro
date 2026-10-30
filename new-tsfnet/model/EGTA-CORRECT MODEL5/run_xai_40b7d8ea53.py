"""
run_xai_40b7d8ea53.py -- run the full XAI suite (xai_extensions.py) on checkpoints
from the ACCEPTED configuration, tag 40b7d8ea53 (AdaBN + EGTA + EFGF-sigmoid,
no noise-aware, no GroupNorm). Run this in the same kernel/notebook where
tsfnet_model_notebook.py and xai_extensions.py have already been %run / imported,
since it imports both by name (matches the project's existing convention).

USAGE (edit the three paths below, then):
    %run run_xai_40b7d8ea53.py

WHAT THIS DOES
For one subject (default VP001, change SUBJECT_ID below) and its held-out-session
checkpoint from results_40b7d8ea53_seed42:
  1. Rebuilds TSFNet with the EXACT architecture flags used for that checkpoint
     (fusion_combination=paper_mean, fnirs_temporal_align=egta,
     efgf_gate_mode=sigmoid) -- same fix as evaluate_noise_robustness.py; getting
     this wrong raises a state_dict mismatch, not a silent wrong answer.
  2. Parses the held-out session from the checkpoint filename
     ("..._session<N>_best.pt"), same bugfix as evaluate_noise_robustness.py,
     so the test_loader is ONLY the true held-out session, not train+test mixed.
  3. Calls every function from xai_extensions.py's own usage example:
     SHAP, fusion-gating-weight plot, EFGF attention map, CAFE attention,
     Grad-CAM. (run_group_level_tsne is a separate, later step -- it needs ALL
     26 subjects' checkpoints, not one -- see bottom of this file for that call,
     commented out until the per-subject pass below looks sane.)

HONEST NOTE: this is a template, not a finished one-click script. The
SUBJECT_ID loop and paths below must be filled in with your real layout.
Run it on VP001 first (single-subject convention from the engineering rules),
read the output, THEN loop over all 26 subjects.
"""
import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import mne
import torch
from torch.utils.data import DataLoader

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch
from xai_trunk import trunk_state                       # drops 'riem_head.*' so a member-branch checkpoint loads into the plain trunk
from xai_extensions import (
    run_shap_analysis, plot_fusion_gating_weights,
    plot_efgf_gating_scalars, plot_efgf_attention_map,
    run_cafe_attention_analysis, run_gradcam_analysis,
    run_single_subject_tsne,  # was available in xai_extensions.py but never
                               # called anywhere in the original script -- the
                               # t-SNE(trained-state)+confusion-matrix analogue
                               # of MSTFDN's Fig. 16, missing until now.
)
# Merged in from validate_fnirs_shap_with_raw_hbo.py: the raw-HbO biomarker
# validation (item 4.2) now runs as part of the same per-session loop below,
# using the SAME checkpoint/session/loader already built for the rest of XAI,
# instead of being a separate script with its own path edits.
from Preproc_tsfnet_nback_badch import (
    load_fnirs_data, FNIRS_TRIAL_LABEL_MAP, SFREQ_FNIRS, WINDOW_STEP,
    WINDOW_LEN_FNIRS_SEGMENT, FNIRS_NUM_SEGMENTS as PREPROC_FNIRS_NUM_SEGMENTS,
    BASELINE_TMIN, BASELINE_TMAX, TRIALS_PER_SESSION,
)

CLASS_NAMES = {0: "0-back", 1: "2-back", 2: "3-back"}
CLASS_COLORS = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}

# ---- EDIT THESE THREE PATHS ------------------------------------------------
RESULTS_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\EGTA\6\results_40b7d8ea53_seed42"
DATA_ROOT   = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
OUTPUT_DIR  = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\xai_40b7d8ea53"
# -----------------------------------------------------------------------------

SUBJECT_ID = "VP001"          # single-subject test first, per project convention
BATCH_SIZE = 16
FNIRS_NUM_SEGMENTS = 21        # must match the accepted config's preprocessing
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Loaded once: raw continuous HbO signal + the real trial onset times/codes (re-derived with
# the pipeline's OWN load_fnirs_data, not re-typed -- see validate_fnirs_shap_with_raw_hbo.py's
# original docstring for why this matters).
_fif_path = os.path.join(DATA_ROOT, SUBJECT_ID, f"{SUBJECT_ID}_fnirs_clean_raw.fif")
if not os.path.exists(_fif_path):
    raise FileNotFoundError(
        f"{_fif_path} not found -- written by preprocess_fnirs() during the normal "
        f"preprocessing run. Needed for the raw-HbO biomarker validation merged into this script."
    )
_raw = mne.io.read_raw_fif(_fif_path, preload=True, verbose=False)
_hbo_picks = [i for i, t in enumerate(_raw.get_channel_types()) if t == "hbo"]
HBO_SIGNAL = _raw.get_data(picks=_hbo_picks).mean(axis=0)   # mean across all HbO channels, mmol/L
assert abs(_raw.info["sfreq"] - SFREQ_FNIRS) < 1e-6, "raw.fif sfreq mismatch -- stale file?"

_, _, _, _, _, _fnirs_ev_time, _fnirs_ev_code = load_fnirs_data(SUBJECT_ID)
_mask = np.isin(_fnirs_ev_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
_times_s = np.asarray(_fnirs_ev_time)[_mask].astype(float) / 1000.0
_codes = np.asarray(_fnirs_ev_code)[_mask]
_order = np.argsort(_times_s)
TRIAL_TIMES = _times_s[_order]
TRIAL_CLASSES = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in _codes[_order]])
SESSION_IDS_ALL = np.arange(len(TRIAL_TIMES)) // TRIALS_PER_SESSION + 1


def compute_raw_hbo_segment_curve(session):
    """Per-class, per-segment mean raw HbO (baseline-corrected), offset=0 window --
    same approximation documented in validate_fnirs_shap_with_raw_hbo.py: segment i
    covers [i*WINDOW_STEP, i*WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT] s post-onset."""
    win_len_samples = int(round(WINDOW_LEN_FNIRS_SEGMENT * SFREQ_FNIRS))
    task_window_samples = int(round(
        (PREPROC_FNIRS_NUM_SEGMENTS * WINDOW_STEP + WINDOW_LEN_FNIRS_SEGMENT) * SFREQ_FNIRS))
    sess_mask = SESSION_IDS_ALL == session
    curves_by_class = {0: [], 1: [], 2: []}
    for t0, cls in zip(TRIAL_TIMES[sess_mask], TRIAL_CLASSES[sess_mask]):
        b_start = int(round((t0 + BASELINE_TMIN) * SFREQ_FNIRS))
        b_end = int(round((t0 + BASELINE_TMAX) * SFREQ_FNIRS))
        onset_sample = int(round(t0 * SFREQ_FNIRS))
        end_sample = onset_sample + task_window_samples
        if b_start < 0 or end_sample > len(HBO_SIGNAL):
            continue
        baseline = HBO_SIGNAL[b_start:b_end].mean()
        curves_by_class[int(cls)].append(HBO_SIGNAL[onset_sample:end_sample] - baseline)

    seg_mean = {}
    for cls, curves in curves_by_class.items():
        if not curves:
            continue
        mean_curve = np.stack(curves).mean(axis=0)
        seg_vals = [mean_curve[int(round(s * WINDOW_STEP * SFREQ_FNIRS)):
                                int(round(s * WINDOW_STEP * SFREQ_FNIRS)) + win_len_samples].mean()
                    for s in range(PREPROC_FNIRS_NUM_SEGMENTS)]
        seg_mean[cls] = np.array(seg_vals)
    return seg_mean


# Accumulated across sessions for the final combined comparison figure.
_shap_fnirs_curves_by_session = {}
_hbo_curves_by_session = {}

# Find this subject's checkpoint(s) -- there is one *_best.pt per held-out
# session/fold, same as evaluate_noise_robustness.py.
ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{SUBJECT_ID}_*_best.pt")))
if not ckpts:
    raise FileNotFoundError(f"No checkpoints for {SUBJECT_ID} in {RESULTS_DIR}")

for ck_path in ckpts:
    name = os.path.basename(ck_path)

    # BUGFIX (same as evaluate_noise_robustness.py): held-out session lives in
    # the filename, never inside the state_dict itself.
    m = re.search(r"_session(\d+)_best\.pt$", name)
    if not m:
        print(f"[skip] cannot parse held-out session from {name}")
        continue
    test_session = int(m.group(1))

    ckpt = torch.load(ck_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    # Architecture MUST match the accepted config exactly, or load_state_dict
    # raises a key-mismatch error (egta adds cafe.egta_q/egta_k; sigmoid gate
    # stores gamma_raw/alpha_raw instead of gamma/alpha).
    model = TSFNet(
        num_classes=3,
        fusion_combination="paper_mean",
        fnirs_num_segments=FNIRS_NUM_SEGMENTS,
        fnirs_temporal_align="egta",
        efgf_gate_mode="sigmoid",
    ).to(DEVICE)
    model.load_state_dict(trunk_state(sd), strict=True)
    model.eval()

    subject_dir = os.path.join(DATA_ROOT, SUBJECT_ID)
    ds = NBackDataset(subject_dir, SUBJECT_ID)
    import numpy as np
    idx = np.where(ds.session_ids == test_session)[0]
    ds = NBackDataset(subject_dir, SUBJECT_ID, indices=idx)
    test_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    tag = f"{SUBJECT_ID}_session{test_session}"
    print(f"\n===== XAI for {tag} (checkpoint: {name}) =====")

    # 1. SHAP on raw EEG/fNIRS inputs
    run_shap_analysis(model, test_loader, DEVICE, session_id=tag,
                       output_dir=OUTPUT_DIR, n_background=32, n_explain=64)

    # 1b. Pull the SAME SHAP fNIRS segment-importance numbers the plot above was drawn
    #     from (saved to disk by run_shap_analysis itself) -- re-deriving them here from the
    #     saved .npz rather than re-running SHAP, and compute the raw-HbO curve for the SAME
    #     session, for the combined comparison figure built after the loop.
    _npz = np.load(os.path.join(OUTPUT_DIR, f"S{tag}_shap_raw.npz"))
    _shap_fnirs, _labels = _npz["shap_fnirs"], _npz["labels"]
    _shap_seg_curve = {}
    for c in range(3):
        _m = _labels == c
        if _m.sum() == 0:
            continue
        _shap_seg_curve[c] = np.abs(_shap_fnirs[c][_m]).mean(axis=(0, 2, 3, 4, 5))
    _shap_fnirs_curves_by_session[test_session] = _shap_seg_curve
    _hbo_curves_by_session[test_session] = compute_raw_hbo_segment_curve(test_session)

    # 2. Fixed fusion scalars (w_eeg/w_fusion/w_fnirs) -- honest reporting,
    #    not instance-adaptive gating (TSFNet has none in this config).
    plot_fusion_gating_weights(model, output_dir=OUTPUT_DIR, session_id=tag)

    # 3. EFGF gate scalars report -- this is the V-2 prerequisite for the
    #    still-open item 1.6 (feature-dependent weighting instead of fixed
    #    gamma/alpha), worth capturing now that the pipeline is already running.
    plot_efgf_gating_scalars(model, output_dir=OUTPUT_DIR, session_id=tag)

    # 4. EFGF hybrid attention map Phi (16x16 grid -- NOT per-electrode, see
    #    the honest limitation noted at the top of xai_extensions.py)
    plot_efgf_attention_map(model, test_loader, DEVICE, session_id=tag,
                             output_dir=OUTPUT_DIR)

    # 5. CAFE cross-modal attention (fNIRS segment importance)
    run_cafe_attention_analysis(model, test_loader, DEVICE, session_id=tag,
                                 output_dir=OUTPUT_DIR)

    # 6. Grad-CAM (cheaper than SHAP, run on more samples)
    run_gradcam_analysis(model, test_loader, DEVICE, session_id=tag,
                          output_dir=OUTPUT_DIR, n_samples=200)

    # 7. Single-subject t-SNE (trained-state fusion features) + confusion matrix
    #    -- direct analogue of MSTFDN Fig. 16. `train_loader` is accepted by the
    #    function's signature but not actually used in its body (it builds the
    #    "initial state" panel from a PCA of test_loader's raw EEG instead) --
    #    passing test_loader again here is harmless, not a mistake.
    run_single_subject_tsne(model, test_loader, test_loader, DEVICE,
                             subject_id=SUBJECT_ID, session_id=test_session,
                             output_dir=OUTPUT_DIR)

print(f"\nDone. Check {OUTPUT_DIR} for {SUBJECT_ID}'s plots before looping over "
      f"all 26 subjects (engineering rule: single-subject sanity check first).")

# -----------------------------------------------------------------------------
# COMBINED COMPARISON FIGURE: SHAP fNIRS segment importance vs. raw HbO per segment,
# one panel per session, twin y-axes (units differ: SHAP has no physical unit, HbO is
# mmol/L) -- this formalizes the earlier eyeballed side-by-side comparison into one
# real figure instead of two separate PNGs read next to each other.
# -----------------------------------------------------------------------------
sessions_done = sorted(_shap_fnirs_curves_by_session.keys())
if sessions_done:
    fig, axes = plt.subplots(1, len(sessions_done), figsize=(6 * len(sessions_done), 5), sharey=False)
    if len(sessions_done) == 1:
        axes = [axes]
    for ax_left, session in zip(axes, sessions_done):
        ax_right = ax_left.twinx()
        shap_curve = _shap_fnirs_curves_by_session[session]
        hbo_curve = _hbo_curves_by_session[session]
        n_seg = max((len(v) for v in shap_curve.values()), default=PREPROC_FNIRS_NUM_SEGMENTS)
        for c in range(3):
            if c in shap_curve:
                ax_left.plot(range(len(shap_curve[c])), shap_curve[c], "-o", ms=3,
                             color=CLASS_COLORS[c], label=f"{CLASS_NAMES[c]} (SHAP)")
            if c in hbo_curve:
                ax_right.plot(range(len(hbo_curve[c])), hbo_curve[c], "--s", ms=3,
                              color=CLASS_COLORS[c], alpha=0.55, label=f"{CLASS_NAMES[c]} (raw HbO)")
        ax_left.set_xlabel("fNIRS segment index")
        ax_left.set_ylabel("Mean |SHAP| (solid)")
        ax_right.set_ylabel("Raw HbO, baseline-corrected, mmol/L (dashed)")
        ax_right.axhline(0, color="gray", linewidth=0.5)
        ax_left.set_title(f"Session {session}")
        # Combine both axes' legend entries into one box (twin axes keep separate
        # legends by default, which is confusing here since the comparison IS the point).
        h1, l1 = ax_left.get_legend_handles_labels()
        h2, l2 = ax_right.get_legend_handles_labels()
        ax_left.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    fig.suptitle(
        f"{SUBJECT_ID} -- SHAP fNIRS segment importance (solid) vs. raw HbO per segment (dashed)\n"
        "Approximation: raw HbO uses the offset=0 window as a representative case (SHAP pools "
        "over all window offsets) -- see validate_fnirs_shap_with_raw_hbo.py's original docstring.",
        fontsize=10,
    )
    plt.tight_layout()
    combined_path = os.path.join(OUTPUT_DIR, f"{SUBJECT_ID}_all_sessions_shap_vs_hbo_comparison.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[combined comparison] saved -> {combined_path}")
else:
    print("\n[combined comparison] skipped -- no sessions were processed above.")

# -----------------------------------------------------------------------------
# GROUP-LEVEL t-SNE (run separately, once ALL 26 subjects' checkpoints exist for
# tag 40b7d8ea53 -- NOT part of the per-subject loop above, since it needs every
# subject's held-out-session checkpoint at once, not just VP001's). Disabled by
# default (RUN_GROUP_LEVEL_TSNE = False) -- flip it to True once you've expanded
# SUBJECT_ID above (or looped this script) across all 26 subjects and are ready
# for the full-population pass. Uses loader_builder (not subject_test_loaders) so
# at most one subject's raw arrays sit in RAM at a time, per the function's own
# memory-safety guidance.
# -----------------------------------------------------------------------------
RUN_GROUP_LEVEL_TSNE = False

if RUN_GROUP_LEVEL_TSNE:
    from xai_extensions import run_group_level_tsne

    ALL_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]

    def model_builder():
        return TSFNet(
            num_classes=3, fusion_combination="paper_mean",
            fnirs_num_segments=FNIRS_NUM_SEGMENTS, fnirs_temporal_align="egta",
            efgf_gate_mode="sigmoid",
        ).to(DEVICE)

    # One representative session per subject (the first checkpoint found) --
    # see the function's own docstring for how to include all 3 sessions/subject
    # instead (key each entry "{subject_id}_S{session}").
    subject_checkpoints = {}
    for sid in ALL_SUBJECTS:
        ckpts_s = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{sid}_*_best.pt")))
        if ckpts_s:
            subject_checkpoints[sid] = ckpts_s[0]
        else:
            print(f"[group-tsne] [skip] no checkpoint found for {sid}")

    def loader_builder(key):
        ck_path = subject_checkpoints[key]
        m = re.search(r"_session(\d+)_best\.pt$", os.path.basename(ck_path))
        sess = int(m.group(1))
        sdir = os.path.join(DATA_ROOT, key)
        ds = NBackDataset(sdir, key)
        idx = np.where(ds.session_ids == sess)[0]
        ds = NBackDataset(sdir, key, indices=idx)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n===== Group-level t-SNE across {len(subject_checkpoints)} subjects =====")
    run_group_level_tsne(
        model_builder=model_builder,
        subject_checkpoints=subject_checkpoints,
        loader_builder=loader_builder,
        device=DEVICE, output_dir=OUTPUT_DIR, max_subjects=26,
    )
# -----------------------------------------------------------------------------
