# ============================================================================
# EOG THRESHOLD VISUAL QC — per-subject validation of ica.find_bads_eog()
#
# Implements the 4-step protocol discussed:
#   1. Parse preprocessing_log.txt to find subjects with ZERO ocular ICs removed
#      (these are the priority candidates for a threshold override).
#   2. For a stratified sample of subjects, re-run EEG loading + filtering + CAR +
#      ICA-fit (identical to preprocess_eeg() in the main pipeline, but WITHOUT
#      excluding/applying anything) and score every IC against every EOG channel.
#   3. Save three kinds of QC evidence per candidate IC: topography, time-course +
#      PSD ("properties" plot), and the raw correlation score — plus a CSV summary
#      table so all borderline ICs across all sampled subjects can be scanned at a
#      glance instead of opening dozens of images blindly.
#   4. Produce a starter subject_specific_eog_thresholds.json that you fill in by
#      hand after visually judging the plots against the 2-out-of-3 criterion
#      (frontal topography + time-locked-to-EOG spikes + low-frequency-dominant PSD).
#
# This script does NOT modify the main pipeline or re-run the full 26-subject
# preprocessing. It only re-does the (fast) filter+CAR+ICA-fit step for a handful
# of subjects, purely for visual inspection.
#
# PARALLELIZATION: subjects are embarrassingly parallel (identical rationale to
# the main preprocessing script's N_WORKERS section) -- each subject's
# load->filter->CAR->ICA-fit->score->plot chain is fully independent of every
# other subject's. Same 1-BLAS-thread-per-worker pinning is applied below for the
# same reason (avoid N_WORKERS x N_THREADS oversubscription during ICA).
#
# IMPORTANT (Jupyter/Windows caveat): multiprocessing.Pool uses the 'spawn' start
# method on Windows, which re-imports this script fresh in every worker process.
# This works cleanly when the script is run from a terminal (`python
# qc_eog_threshold_validation.py`) or via `%run` from a .py file. Running it after
# pasting code directly into notebook cells (rather than `%run`-ing the .py file)
# is NOT supported with parallel workers on Windows, because spawned processes
# cannot re-import a notebook cell. If you hit pickling/spawn errors in Jupyter,
# either run with --n_workers 1 (falls back to the original serial loop) or run
# from an actual terminal instead of %run.
# ============================================================================

import os
# MUST be set BEFORE numpy/scipy/mne are imported -- same reasoning as the main
# preprocessing script: pins each worker's BLAS/OMP thread pool to 1 thread so
# N_WORKERS subject-processes don't each ALSO multithread internally.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import re
import sys
import json
import argparse
import multiprocessing as mp

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

# ----------------------------------------------------------------------------
# Import the main preprocessing script as a module, so we reuse its EXACT
# load_eeg_data() / filtering / ICA-fit logic instead of duplicating it (and
# risking the two versions drifting apart over time).
# ----------------------------------------------------------------------------
MAIN_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT_NAME = "Preprocessing_TSFNet_paper_nback_v2"   # no .py extension, for import

if MAIN_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, MAIN_SCRIPT_DIR)

try:
    main_pipeline = __import__(MAIN_SCRIPT_NAME)
except ImportError as e:
    raise ImportError(
        f"Could not import {MAIN_SCRIPT_NAME}.py -- make sure this QC script sits in the SAME "
        f"folder as your main preprocessing script (or edit MAIN_SCRIPT_NAME/MAIN_SCRIPT_DIR "
        f"above to point at it). Original error: {e}"
    )

BASE_DATASET_PATH = main_pipeline.BASE_DATASET_PATH
OUTPUT_PATH = main_pipeline.OUTPUT_PATH
DEFAULT_EOG_THRESHOLD = 2.0
BORDERLINE_LOW, BORDERLINE_HIGH = 1.5, 2.5   # window around threshold=2.0 to flag as "worth a look"


# ============================================================================
# STEP 1: find subjects with ZERO ocular ICs removed in the last full run
# ============================================================================
def find_zero_removal_subjects(log_path):
    """
    Parses lines like:
      [...] [PID...]   VP0XX: removed 0 ocular ICA component(s): []
    and returns the list of subject_ids that removed exactly 0 components.
    """
    zero_removal, all_seen = [], {}
    pattern = re.compile(r"(VP\d+): removed (\d+) ocular ICA component")
    if not os.path.exists(log_path):
        print(f"WARNING: log file not found at {log_path} -- cannot auto-detect zero-removal "
              f"subjects. You will need to pass --subjects manually.")
        return []
    with open(log_path, "r") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                sid, n_removed = m.group(1), int(m.group(2))
                all_seen[sid] = n_removed   # keep last occurrence per subject
    zero_removal = [sid for sid, n in all_seen.items() if n == 0]
    print(f"Parsed {len(all_seen)} subjects from log. "
          f"{len(zero_removal)} removed ZERO ocular components: {zero_removal}")
    return zero_removal, all_seen


def pick_stratified_sample(all_seen, zero_removal, n_total=5):
    """
    Builds a stratified sample:
      - ALL zero-removal subjects (priority; capped at n_total-1 so at least one
        control subject is always included)
      - subject with the MOST components removed (opposite extreme, sanity check)
      - remaining slots filled with subjects near the median removal count
    """
    if not all_seen:
        return []
    sorted_by_count = sorted(all_seen.items(), key=lambda kv: kv[1])
    most_removed_subject = sorted_by_count[-1][0]

    sample = list(dict.fromkeys(zero_removal[: max(0, n_total - 1)] + [most_removed_subject]))
    if len(sample) < n_total:
        median_idx = len(sorted_by_count) // 2
        for sid, _ in sorted_by_count[median_idx:]:
            if sid not in sample:
                sample.append(sid)
            if len(sample) >= n_total:
                break
    return sample[:n_total]


# ============================================================================
# STEP 2+3: refit ICA (no exclusion applied) and generate QC evidence
# ============================================================================
def refit_ica_for_qc(subject_id):
    """
    Re-runs load -> bandpass -> CAR -> ICA.fit() IDENTICALLY to preprocess_eeg()
    in the main pipeline, but stops BEFORE any component exclusion/application,
    so we can inspect the raw scores for every IC (not just the ones that already
    passed the default threshold).
    """
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    raw, events, sfreq_eeg, eeg_mnt_struct = main_pipeline.load_eeg_data(eeg_folder)
    raw_before = raw.copy()
    filtered = raw.copy()

    if main_pipeline.APPLY_NOTCH:
        filtered.notch_filter(freqs=main_pipeline.EEG_NOTCH_FREQ, picks="eeg", verbose=False)

    filtered.filter(
        l_freq=main_pipeline.EEG_BP_LOW, h_freq=main_pipeline.EEG_BP_HIGH,
        method="iir",
        iir_params=dict(order=main_pipeline.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    filtered.set_eeg_reference("average", projection=False, verbose=False)

    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True),
        random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)
    return ica, filtered, raw_before


def score_all_components(ica, filtered, raw_before, eog_channels=("VEOG", "HEOG")):
    """Returns {eog_ch: np.array(scores per IC)} for EVERY IC, not just the excluded ones."""
    all_scores = {}
    for eog_ch in eog_channels:
        if eog_ch in filtered.ch_names:
            try:
                _, scores = ica.find_bads_eog(
                    raw_before, ch_name=eog_ch, threshold=DEFAULT_EOG_THRESHOLD, verbose=False
                )
                all_scores[eog_ch] = np.asarray(scores)
            except Exception as e:
                print(f"  WARNING: find_bads_eog({eog_ch}) failed: {e}")
    return all_scores


def find_borderline_ics(all_scores, low=BORDERLINE_LOW, high=BORDERLINE_HIGH):
    """ICs whose |score| falls in [low, high] for ANY EOG channel -- these are the
    ones worth a human look, whether or not they currently get excluded."""
    borderline = []
    for eog_ch, scores in all_scores.items():
        for ic_idx, score in enumerate(scores):
            if low <= abs(score) <= high:
                borderline.append((ic_idx, eog_ch, float(score)))
    return sorted(set(borderline), key=lambda x: -abs(x[2]))


def generate_qc_plots(ica, filtered, subject_id, ics_to_plot, qc_out_dir):
    os.makedirs(qc_out_dir, exist_ok=True)
    if not ics_to_plot:
        print(f"  {subject_id}: no borderline/candidate ICs to plot.")
        return

    try:
        fig = ica.plot_components(picks=ics_to_plot, show=False)
        figs = fig if isinstance(fig, list) else [fig]
        for i, f in enumerate(figs):
            f.savefig(os.path.join(qc_out_dir, f"{subject_id}_ica_topo_{i}.png"), dpi=150)
            plt.close(f)
    except Exception as e:
        print(f"  {subject_id}: topography plot failed: {e}")

    for ic_idx in ics_to_plot:
        try:
            figs = ica.plot_properties(filtered, picks=[ic_idx], show=False, psd_args={"fmax": 50})
            for f in figs:
                f.savefig(os.path.join(qc_out_dir, f"{subject_id}_ica_ic{ic_idx}_properties.png"), dpi=150)
                plt.close(f)
        except Exception as e:
            print(f"  {subject_id}: properties plot for IC{ic_idx} failed: {e}")


# ============================================================================
# PER-SUBJECT WORKER (used by both the serial loop and the parallel Pool)
# ============================================================================
def process_subject_qc(subject_id, qc_out_dir):
    """
    Runs the full per-subject QC chain (refit ICA -> score -> plot) and returns
    ONLY plain, picklable Python data (no MNE/ICA objects), since this is exactly
    what needs to cross the process boundary when run under multiprocessing.Pool.

    Returns: (subject_id, csv_row_strings, borderline_list, default_excluded_sorted, error_or_None)
    """
    print(f"--- {subject_id} (PID {os.getpid()}) ---")
    try:
        ica, filtered, raw_before = refit_ica_for_qc(subject_id)
    except Exception as e:
        return subject_id, [], [], [], f"ERROR loading/fitting {subject_id}: {e}"

    all_scores = score_all_components(ica, filtered, raw_before)
    borderline = find_borderline_ics(all_scores)

    default_excluded = set()
    for eog_ch in all_scores:
        idx, _ = ica.find_bads_eog(raw_before, ch_name=eog_ch,
                                    threshold=DEFAULT_EOG_THRESHOLD, verbose=False)
        default_excluded.update(idx)

    print(f"  [{subject_id}] Components with default threshold={DEFAULT_EOG_THRESHOLD}: "
          f"{sorted(default_excluded)}")
    print(f"  [{subject_id}] Borderline ICs (|score| in [{BORDERLINE_LOW},{BORDERLINE_HIGH}]): "
          f"{[(i, ch, round(s, 3)) for i, ch, s in borderline]}")

    csv_rows = []
    for eog_ch, scores in all_scores.items():
        for ic_idx, score in enumerate(scores):
            csv_rows.append(
                f"{subject_id},{ic_idx},{eog_ch},{score:.4f},{abs(score):.4f},"
                f"{ic_idx in default_excluded}"
            )

    ics_to_plot = sorted(set([i for i, _, _ in borderline]) | default_excluded)
    subj_qc_dir = os.path.join(qc_out_dir, subject_id)
    generate_qc_plots(ica, filtered, subject_id, ics_to_plot, subj_qc_dir)
    print(f"  [{subject_id}] QC plots saved to: {subj_qc_dir}\n")

    return subject_id, csv_rows, borderline, sorted(default_excluded), None


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="EOG threshold visual QC")
    parser.add_argument("--n_subjects", type=int, default=5,
                         help="How many subjects to sample if --subjects is not given (default 5).")
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated explicit subject list, e.g. VP001,VP014. "
                              "Overrides automatic stratified sampling.")
    parser.add_argument("--log_path", type=str,
                         default=os.path.join(OUTPUT_PATH, "preprocessing_log.txt"))
    parser.add_argument("--qc_out_dir", type=str,
                         default=os.path.join(OUTPUT_PATH, "eog_threshold_qc"))
    parser.add_argument("--n_workers", type=int, default=1,
                         help="Number of parallel subject-level workers (default 1 = serial, "
                              "same behavior as before). Set e.g. 4-5 to parallelize across "
                              "subjects, matching the main pipeline's subject-level parallelism. "
                              "NOTE: only reliable when run via terminal or `%%run script.py` on "
                              "a saved .py file -- not from pasted notebook-cell code on Windows.")
    args = parser.parse_args()

    zero_removal, all_seen = find_zero_removal_subjects(args.log_path)

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    else:
        subjects = pick_stratified_sample(all_seen, zero_removal, n_total=args.n_subjects)

    if not subjects:
        print("No subjects to process. Pass --subjects VP001,VP002,... manually, "
              "or make sure preprocessing_log.txt exists and has been generated by a prior run.")
        return

    os.makedirs(args.qc_out_dir, exist_ok=True)
    print(f"\n=== Running EOG threshold QC on: {subjects} "
          f"(n_workers={args.n_workers}) ===\n")

    csv_rows = ["subject_id,ic_index,eog_channel,score,abs_score,currently_excluded"]
    all_results = []   # (subject_id, borderline, default_excluded)

    if args.n_workers <= 1:
        # --- SERIAL (original behavior, always safe incl. inside Jupyter) ---
        for subject_id in subjects:
            sid, rows, borderline, default_excluded, err = process_subject_qc(subject_id, args.qc_out_dir)
            if err:
                print(f"  {err}")
                continue
            csv_rows.extend(rows)
            all_results.append((sid, borderline, default_excluded))
    else:
        # --- PARALLEL (subject-level, mirrors the main pipeline's Pool usage) ---
        with mp.Pool(processes=args.n_workers) as pool:
            for sid, rows, borderline, default_excluded, err in pool.starmap(
                process_subject_qc, [(s, args.qc_out_dir) for s in subjects]
            ):
                if err:
                    print(f"  {err}")
                    continue
                csv_rows.extend(rows)
                all_results.append((sid, borderline, default_excluded))

    csv_path = os.path.join(args.qc_out_dir, "eog_ic_scores_summary.csv")
    with open(csv_path, "w") as f:
        f.write("\n".join(csv_rows))
    print(f"Score summary table saved to: {csv_path}")

    # Starter JSON for manual decisions (step 4) -- pre-filled with subjects+ICs
    # that showed at least one borderline score, so you don't have to retype IDs.
    starter = {
        "_comment": (
            "Fill in 'overrides' ONLY for subjects where visual inspection (topography + "
            "time-course + PSD, in the per-subject QC folders) confirmed a genuine ocular IC "
            "that the default threshold=2.0 missed. Decision rule: at least 2 of 3 criteria "
            "(frontal topography, spikes time-locked to EOG channel, low-frequency-dominant PSD)."
        ),
        "default_threshold": DEFAULT_EOG_THRESHOLD,
        "overrides": {
            sid: {"suggested_threshold": None, "reason": "TODO: fill in after visual review"}
            for sid, _, _ in all_results if sid in zero_removal
        },
    }
    starter_path = os.path.join(args.qc_out_dir, "subject_specific_eog_thresholds.STARTER.json")
    with open(starter_path, "w") as f:
        json.dump(starter, f, indent=2, ensure_ascii=False)
    print(f"Starter decision file saved to: {starter_path}")
    print("\nNext step: open the per-subject QC PNGs + the CSV, decide per subject, "
          "fill in 'suggested_threshold' + 'reason' in the STARTER json, then rename it to "
          "subject_specific_eog_thresholds.json for the main pipeline to consume.")


if __name__ == "__main__":
    main()
