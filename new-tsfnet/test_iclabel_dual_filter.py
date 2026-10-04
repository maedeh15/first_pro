# ============================================================================
# TEST: dual-filter ICA (fit on 1-100Hz for ICLabel) vs current single-filter
# approach (fit on 0.5-50Hz) -- side-by-side comparison on selected subjects.
#
# For each subject, this script builds TWO parallel pipelines and reports what
# each one would exclude, so you can compare before deciding which to adopt:
#
#   OLD (current pipeline): ICA fit on the 0.5-50Hz analysis-filtered signal;
#     exclude = union(find_bads_eog correlation, ICLabel eye-only on 0.5-50Hz data)
#
#   NEW (dual-filter): ICA fit on a SEPARATE 1-100Hz copy (matches ICLabel's
#     documented spec); ICLabel run on that same 1-100Hz copy using ALL 6
#     artifact classes (eye/muscle/heart/line/channel noise) with a probability
#     threshold (default 0.80); the resulting ica.exclude is then applied to the
#     actual 0.5-50Hz analysis data (ica.apply() is filter-agnostic -- it only
#     needs the same channels/samples as what it was fit on... actually it needs
#     the SAME raw length/channels, so here we apply the same `ica` object,
#     fit on 1-100Hz data, directly to the 0.5-50Hz `filtered` copy of the SAME
#     recording).
#
# Usage:
#   python test_iclabel_dual_filter.py --subjects VP010,VP011,VP018
#   python test_iclabel_dual_filter.py --subjects VP010,VP011,VP018 --n_workers 3
# (Run via terminal for --n_workers > 1, per the earlier Jupyter/Windows caveat.)
# ============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import sys
import json
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import mne
import mne_icalabel

MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_NAME = "Preprocessing_TSFNet_paper_nback_v2"
if MAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MAIN_SCRIPT_DIR)
main_pipeline = __import__(MAIN_SCRIPT_NAME)

BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
OUTPUT_PATH = main_pipeline.OUTPUT_PATH
DEFAULT_EOG_THRESHOLD = 2.0

# --- NEW pipeline settings ---
ICLABEL_FILTER_LOW = 1.0
ICLABEL_FILTER_HIGH_TARGET = 100.0   # ICLabel's documented spec upper edge
ICLABEL_FILTER_ORDER = 6
PROB_THRESHOLD = 0.80
ARTIFACT_CLASSES = ("eye blink", "eye movement", "muscle artifact", "heart beat",
                     "line noise", "channel noise")
EYE_ONLY_CLASSES = ("eye blink", "eye movement")


def safe_iclabel_high_freq(sfreq, target=ICLABEL_FILTER_HIGH_TARGET, margin=1.0):
    """MNE requires h_freq to be STRICTLY below the Nyquist frequency (sfreq/2), not
    equal to it. For this dataset sfreq=200Hz -> Nyquist=100Hz exactly, so the target
    100Hz upper edge (ICLabel's documented spec) sits exactly ON the boundary and gets
    rejected. This clips the requested upper edge to at most (Nyquist - margin)."""
    nyquist = sfreq / 2.0
    return min(target, nyquist - margin)


def build_filtered_copy(raw, l_freq, h_freq, order):
    out = raw.copy()
    out.filter(
        l_freq=l_freq, h_freq=h_freq, method="iir",
        iir_params=dict(order=order, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    out.set_eeg_reference("average", projection=False, verbose=False)
    return out


def run_old_pipeline(raw_before, filtered_analysis, picks_eeg):
    """Current pipeline: ICA fit on the 0.5-50Hz analysis-filtered signal."""
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True), random_state=42, verbose=False,
    )
    ica.fit(filtered_analysis, picks=picks_eeg)

    corr_flagged = set()
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered_analysis.ch_names:
            idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch,
                                        threshold=DEFAULT_EOG_THRESHOLD, verbose=False)
            corr_flagged.update(idx)

    ic_labels = mne_icalabel.label_components(filtered_analysis, ica, method="iclabel")
    iclabel_flagged = {i for i, lab in enumerate(ic_labels["labels"]) if lab in EYE_ONLY_CLASSES}

    excluded = sorted(corr_flagged | iclabel_flagged)
    return {
        "n_ics": ica.n_components_,
        "excluded": excluded,
        "n_excluded": len(excluded),
        "pct_excluded": round(100.0 * len(excluded) / max(ica.n_components_, 1), 1),
        "corr_flagged": sorted(corr_flagged),
        "iclabel_eye_flagged": sorted(iclabel_flagged),
    }


def run_new_pipeline(raw_before, filtered_analysis, picks_eeg):
    """Dual-filter pipeline: ICA fit on a separate 1-100Hz copy, ICLabel run with
    all 6 artifact classes + probability threshold, then applied to the analysis data."""
    filtered_iclabel = build_filtered_copy(
        raw_before, ICLABEL_FILTER_LOW, safe_iclabel_high_freq(raw_before.info["sfreq"]),
        ICLABEL_FILTER_ORDER
    )

    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True), random_state=42, verbose=False,
    )
    ica.fit(filtered_iclabel, picks=picks_eeg)

    # correlation scoring still uses the raw (unfiltered) EOG reference, same as before
    corr_flagged = set()
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered_iclabel.ch_names:
            idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch,
                                        threshold=DEFAULT_EOG_THRESHOLD, verbose=False)
            corr_flagged.update(idx)

    ic_labels = mne_icalabel.label_components(filtered_iclabel, ica, method="iclabel")
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]

    iclabel_flagged_all = {
        i for i, (lab, p) in enumerate(zip(labels, probs))
        if lab in ARTIFACT_CLASSES and p >= PROB_THRESHOLD
    }
    iclabel_flagged_eye_only = {
        i for i, (lab, p) in enumerate(zip(labels, probs))
        if lab in EYE_ONLY_CLASSES and p >= PROB_THRESHOLD
    }
    per_ic_labels = {i: {"label": lab, "prob": round(float(p), 3)} for i, (lab, p) in enumerate(zip(labels, probs))}

    excluded = sorted(corr_flagged | iclabel_flagged_all)   # full 6-class exclusion (the "aggressive" option)
    excluded_eye_only = sorted(corr_flagged | iclabel_flagged_eye_only)  # eye-only exclusion (closer to old scope)

    return {
        "n_ics": ica.n_components_,
        "excluded_all_classes": excluded,
        "n_excluded_all_classes": len(excluded),
        "pct_excluded_all_classes": round(100.0 * len(excluded) / max(ica.n_components_, 1), 1),
        "excluded_eye_only": excluded_eye_only,
        "n_excluded_eye_only": len(excluded_eye_only),
        "pct_excluded_eye_only": round(100.0 * len(excluded_eye_only) / max(ica.n_components_, 1), 1),
        "corr_flagged": sorted(corr_flagged),
        "iclabel_flagged_all_classes": sorted(iclabel_flagged_all),
        "iclabel_flagged_eye_only": sorted(iclabel_flagged_eye_only),
        "per_ic_labels": per_ic_labels,
        # NOTE: `ica` here was fit on filtered_iclabel; to actually USE this in the
        # real pipeline you would do: ica.exclude = excluded_eye_only (or _all_classes);
        # ica.apply(filtered_analysis)  -- applying the SAME ica object (fit on 1-100Hz)
        # to the 0.5-50Hz analysis copy of the SAME recording.
    }


def process_subject_compare(subject_id):
    print(f"--- {subject_id} (PID {os.getpid()}) ---")
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    try:
        raw, events, sfreq_eeg, eeg_mnt_struct = main_pipeline.load_eeg_data(eeg_folder)
    except Exception as e:
        return {"subject_id": subject_id, "error": f"load failed: {e}"}

    raw_before = raw.copy()
    filtered_analysis = build_filtered_copy(
        raw_before, main_pipeline.EEG_BP_LOW, main_pipeline.EEG_BP_HIGH, main_pipeline.EEG_BP_ORDER
    )
    picks_eeg = mne.pick_types(filtered_analysis.info, eeg=True, eog=False)

    try:
        old_result = run_old_pipeline(raw_before, filtered_analysis, picks_eeg)
    except Exception as e:
        print(f"  [{subject_id}] OLD pipeline FAILED: {e}")
        old_result = {"error": f"OLD pipeline failed: {e}"}

    try:
        new_result = run_new_pipeline(raw_before, filtered_analysis, picks_eeg)
    except Exception as e:
        import traceback
        print(f"  [{subject_id}] NEW pipeline FAILED: {e}")
        print(f"  [{subject_id}] traceback:\n{traceback.format_exc()}")
        new_result = {"error": f"NEW pipeline failed: {e}"}

    print(f"  [{subject_id}] OLD: excluded={old_result.get('excluded')} "
          f"({old_result.get('pct_excluded')}%)")
    print(f"  [{subject_id}] NEW (eye-only):    excluded={new_result.get('excluded_eye_only')} "
          f"({new_result.get('pct_excluded_eye_only')}%)")
    print(f"  [{subject_id}] NEW (all classes): excluded={new_result.get('excluded_all_classes')} "
          f"({new_result.get('pct_excluded_all_classes')}%)")

    return {"subject_id": subject_id, "error": None, "old": old_result, "new": new_result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default="VP010,VP011,VP018")
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--out_dir", type=str,
                         default=os.path.join(OUTPUT_PATH, "iclabel_dual_filter_test"))
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"\n=== Comparing OLD vs NEW (dual-filter) ICA pipeline on: {subjects} ===\n")

    if args.n_workers <= 1:
        results = [process_subject_compare(s) for s in subjects]
    else:
        with mp.Pool(processes=args.n_workers) as pool:
            results = pool.map(process_subject_compare, subjects)

    rows = []
    for r in results:
        if r.get("error"):
            print(f"ERROR: {r['subject_id']}: {r['error']}")
            continue
        old, new = r["old"], r["new"]
        rows.append({
            "subject_id": r["subject_id"],
            "n_ics": old.get("n_ics"),
            "OLD_excluded": old.get("excluded"),
            "OLD_n_excluded": old.get("n_excluded"),
            "OLD_pct_excluded": old.get("pct_excluded"),
            "NEW_eyeonly_excluded": new.get("excluded_eye_only"),
            "NEW_eyeonly_n_excluded": new.get("n_excluded_eye_only"),
            "NEW_eyeonly_pct_excluded": new.get("pct_excluded_eye_only"),
            "NEW_allclasses_excluded": new.get("excluded_all_classes"),
            "NEW_allclasses_n_excluded": new.get("n_excluded_all_classes"),
            "NEW_allclasses_pct_excluded": new.get("pct_excluded_all_classes"),
            "NEW_per_ic_labels": json.dumps(new.get("per_ic_labels", {})),
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(args.out_dir, "old_vs_new_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nComparison table saved: {out_path}")
    if not df.empty:
        print(df[["subject_id", "n_ics", "OLD_pct_excluded",
                   "NEW_eyeonly_pct_excluded", "NEW_allclasses_pct_excluded"]].to_string(index=False))
    print("\nNext step: open old_vs_new_comparison.csv and NEW_per_ic_labels to see WHICH extra "
          "classes (muscle/heart/line/channel noise) the all-classes version catches beyond eye-only, "
          "then decide: keep OLD, switch to NEW eye-only (dual-filter fix only), or switch to NEW "
          "all-classes (dual-filter + broader artifact removal).")


if __name__ == "__main__":
    main()
