# ============================================================================
# FAST CONSOLIDATED EOG QC — one run, all checks, minimal manual review
#
# Does everything in a single pass across ALL subjects:
#   1. Re-fits ICA (no exclusion applied) for every subject.
#   2. For every IC, records: correlation score (per EOG channel), whether
#      find_bads_eog flags it (score-based), whether ICLabel independently
#      flags it ("eye blink"/"eye movement"), and which of the 3 categories it
#      falls into: CORR_ONLY (risky -- likely false positive, e.g. frontal brain
#      activity mistaken for EOG), ICLABEL_ONLY, or BOTH (high-confidence, safe).
#   3. AUTOMATED SPECTRAL VETO on every CORR_ONLY IC (the risky category), using
#      the exact feature definitions ICLabel itself is documented to rely on:
#        - Eye components: power dominated by <5 Hz
#        - Brain components: a distinct alpha-band (8-13 Hz) peak
#      This yields one of three verdicts per CORR_ONLY IC:
#        - KEEP_EXCLUDED   -> low-freq dominant, consistent with genuine ocular
#                             activity -> stays excluded, no human review needed
#        - AUTO_UNEXCLUDE  -> alpha peak present (no low-freq dominance) ->
#                             very likely genuine brain activity -> automatically
#                             REMOVED from the exclusion list, no human review needed
#        - AMBIGUOUS       -> neither pattern is clear -> only THESE get plotted
#                             for manual visual review (the true bottleneck subset)
#   4. Computes per-subject exclusion rate (post-veto) and auto-flags subjects
#      that still need a human look, using objective rules:
#        - pct_excluded > 25% (post-veto)  -> unusually high removal
#        - n_ambiguous > 0                 -> at least one IC the spectral veto
#                                             could not resolve automatically
#        - n_excluded == 0                 -> possible missed ocular component
#   5. Generates plots ONLY for AMBIGUOUS ICs of flagged subjects -- the minimum
#      possible set that still requires a human decision.
#
# Usage (run via terminal or %run on the saved .py file, NOT pasted cells,
# if using --n_workers > 1 on Windows):
#   python qc_eog_fast_full_check.py --n_workers 5
#   python qc_eog_fast_full_check.py --subjects VP001,VP002 --n_workers 1
# ============================================================================

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import sys
import json
import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
ALL_SUBJECTS = [f"VP{n:03d}" for n in main_pipeline.SUBJECTS]

# --- Auto-flagging rules (objective, no manual triage to pick which subject) ---
PCT_EXCLUDED_FLAG = 25.0     # flag if >25% of ICs excluded (post-veto)
N_AMBIGUOUS_FLAG = 0         # flag if any IC could not be auto-resolved (>0 -> flagged)

# --- Automated spectral veto thresholds (mirrors ICLabel's own documented feature
# definitions: "Eye components ... dominate <5 Hz"; "Brain components show ... alpha-range
# peaks") -- see mne-icalabel docs / Pion-Tonachini et al. 2019 ---
VETO_LOW_FREQ_CUTOFF = 5.0        # Hz; below this = "low-frequency" band for the eye check
VETO_ALPHA_BAND = (8.0, 13.0)     # Hz; standard alpha band for the brain check
VETO_ALPHA_PEAK_RATIO = 1.5       # alpha power must exceed this x the whole-spectrum mean power
VETO_LOWFREQ_DOMINANCE_RATIO = 2.0  # <5Hz power must exceed this x the whole-spectrum mean power


def auto_veto_corr_only(filtered, ica, ic_idx):
    """
    Automated decision for a single CORR_ONLY IC (flagged by correlation, NOT by
    ICLabel), using ICLabel's own documented spectral feature definitions:
      - Eye:   power dominated by <5 Hz
      - Brain: a distinct alpha-band (8-13 Hz) peak
    Returns one of: "KEEP_EXCLUDED", "AUTO_UNEXCLUDE", "AMBIGUOUS", plus a reason string.
    """
    sources = ica.get_sources(filtered).get_data(picks=[ic_idx])[0]
    psd, freqs = mne.time_frequency.psd_array_welch(
        sources, sfreq=filtered.info["sfreq"], fmin=1, fmax=50, verbose=False
    )
    psd = np.asarray(psd).squeeze()

    low_power = psd[(freqs >= 1) & (freqs < VETO_LOW_FREQ_CUTOFF)].mean()
    alpha_power = psd[(freqs >= VETO_ALPHA_BAND[0]) & (freqs <= VETO_ALPHA_BAND[1])].mean()
    total_power = psd.mean()

    has_alpha_peak = alpha_power > VETO_ALPHA_PEAK_RATIO * total_power
    is_low_freq_dominant = low_power > VETO_LOWFREQ_DOMINANCE_RATIO * total_power

    if is_low_freq_dominant and not has_alpha_peak:
        return "KEEP_EXCLUDED", f"low-freq dominant (<{VETO_LOW_FREQ_CUTOFF}Hz), consistent with ocular pattern"
    if has_alpha_peak and not is_low_freq_dominant:
        return "AUTO_UNEXCLUDE", f"alpha peak ({VETO_ALPHA_BAND[0]}-{VETO_ALPHA_BAND[1]}Hz) detected, likely genuine brain activity"
    return "AMBIGUOUS", "neither pattern clearly dominant -- needs manual visual check"


def refit_ica(subject_id):
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    raw, events, sfreq_eeg, eeg_mnt_struct = main_pipeline.load_eeg_data(eeg_folder)
    raw_before = raw.copy()
    filtered = raw.copy()
    if main_pipeline.APPLY_NOTCH:
        filtered.notch_filter(freqs=main_pipeline.EEG_NOTCH_FREQ, picks="eeg", verbose=False)
    filtered.filter(
        l_freq=main_pipeline.EEG_BP_LOW, h_freq=main_pipeline.EEG_BP_HIGH,
        method="iir", iir_params=dict(order=main_pipeline.EEG_BP_ORDER, ftype="butter", output="sos"),
        picks="eeg", phase="zero", verbose=False,
    )
    filtered.set_eeg_reference("average", projection=False, verbose=False)
    picks_eeg = mne.pick_types(filtered.info, eeg=True, eog=False)
    ica = mne.preprocessing.ICA(
        n_components=0.99, method="picard",
        fit_params=dict(ortho=False, extended=True), random_state=42, verbose=False,
    )
    ica.fit(filtered, picks=picks_eeg)
    return ica, filtered, raw_before


def process_subject_full(subject_id, qc_out_dir):
    """One-pass, per-subject: score every IC via BOTH correlation and ICLabel,
    classify each into CORR_ONLY / ICLABEL_ONLY / BOTH / NEITHER, decide if the
    subject needs review, and plot only the CORR_ONLY (risky) ICs if flagged."""
    print(f"--- {subject_id} (PID {os.getpid()}) ---")
    try:
        ica, filtered, raw_before = refit_ica(subject_id)
    except Exception as e:
        return {"subject_id": subject_id, "error": f"{e}"}

    n_ics = ica.n_components_

    # correlation-based flags (union across EOG channels, matching main pipeline logic)
    corr_flagged = set()
    corr_scores = {}   # ic_idx -> max |score| across channels
    for eog_ch in ("VEOG", "HEOG"):
        if eog_ch in filtered.ch_names:
            try:
                idx, scores = ica.find_bads_eog(raw_before, ch_name=eog_ch,
                                                  threshold=DEFAULT_EOG_THRESHOLD, verbose=False)
                corr_flagged.update(idx)
                for i, s in enumerate(scores):
                    corr_scores[i] = max(corr_scores.get(i, 0.0), abs(float(s)))
            except Exception as e:
                print(f"  [{subject_id}] find_bads_eog({eog_ch}) failed: {e}")

    # ICLabel-based flags
    try:
        ic_labels = mne_icalabel.label_components(filtered, ica, method="iclabel")
        iclabel_flagged = {i for i, lab in enumerate(ic_labels["labels"])
                            if lab in ("eye blink", "eye movement")}
        iclabel_labels = ic_labels["labels"]
    except Exception as e:
        print(f"  [{subject_id}] ICLabel failed: {e}")
        iclabel_flagged, iclabel_labels = set(), ["n/a"] * n_ics

    both = corr_flagged & iclabel_flagged
    corr_only = corr_flagged - iclabel_flagged
    iclabel_only = iclabel_flagged - corr_flagged

    # --- AUTOMATED SPECTRAL VETO on every CORR_ONLY IC (the risky category) ---
    veto_results = {}
    auto_unexcluded, ambiguous = set(), set()
    for ic_idx in corr_only:
        try:
            verdict, reason = auto_veto_corr_only(filtered, ica, ic_idx)
        except Exception as e:
            verdict, reason = "AMBIGUOUS", f"veto computation failed: {e}"
        veto_results[ic_idx] = {"verdict": verdict, "reason": reason}
        print(f"  [{subject_id}] IC{ic_idx} (CORR_ONLY) -> {verdict}: {reason}")
        if verdict == "AUTO_UNEXCLUDE":
            auto_unexcluded.add(ic_idx)
        elif verdict == "AMBIGUOUS":
            ambiguous.add(ic_idx)
        # KEEP_EXCLUDED needs no action -- it simply stays in excluded_union below

    # post-veto exclusion set: union minus the automatically-vetoed false positives
    excluded_union = (corr_flagged | iclabel_flagged) - auto_unexcluded

    pct_excluded = 100.0 * len(excluded_union) / max(n_ics, 1)
    needs_review = (
        pct_excluded > PCT_EXCLUDED_FLAG
        or len(ambiguous) > N_AMBIGUOUS_FLAG
        or len(excluded_union) == 0
    )

    print(f"  [{subject_id}] n_ics={n_ics}, excluded(post-veto)={len(excluded_union)} "
          f"({pct_excluded:.1f}%), corr_only={len(corr_only)} "
          f"(auto_unexcluded={len(auto_unexcluded)}, ambiguous={len(ambiguous)}), "
          f"iclabel_only={len(iclabel_only)}, both={len(both)}, FLAGGED={needs_review}")

    plotted = []
    if ambiguous:
        # ONLY the ambiguous ICs get plotted -- the true minimum bottleneck for
        # human review, regardless of the subject-level needs_review flag (an
        # ambiguous IC always deserves a look, even if the subject's overall
        # exclusion rate looks otherwise unremarkable).
        subj_dir = os.path.join(qc_out_dir, subject_id)
        os.makedirs(subj_dir, exist_ok=True)
        ics_to_plot = sorted(ambiguous)
        try:
            fig = ica.plot_components(picks=ics_to_plot, show=False)
            figs = fig if isinstance(fig, list) else [fig]
            for i, f in enumerate(figs):
                f.savefig(os.path.join(subj_dir, f"{subject_id}_AMBIGUOUS_topo_{i}.png"), dpi=150)
                plt.close(f)
        except Exception as e:
            print(f"  [{subject_id}] topo plot failed: {e}")
        for ic_idx in ics_to_plot:
            try:
                figs = ica.plot_properties(filtered, picks=[ic_idx], show=False, psd_args={"fmax": 50})
                for f in figs:
                    f.savefig(os.path.join(subj_dir, f"{subject_id}_AMBIGUOUS_ic{ic_idx}_properties.png"), dpi=150)
                    plt.close(f)
            except Exception as e:
                print(f"  [{subject_id}] properties plot IC{ic_idx} failed: {e}")
        plotted = ics_to_plot
        print(f"  [{subject_id}] plotted AMBIGUOUS ICs {plotted} -> {subj_dir}")

    return {
        "subject_id": subject_id, "error": None, "n_ics": n_ics,
        "n_excluded": len(excluded_union), "pct_excluded": round(pct_excluded, 1),
        "n_corr_only": len(corr_only), "n_iclabel_only": len(iclabel_only), "n_both": len(both),
        "n_auto_unexcluded": len(auto_unexcluded), "n_ambiguous": len(ambiguous),
        "needs_review": needs_review,
        "corr_only_ics": sorted(corr_only), "iclabel_only_ics": sorted(iclabel_only),
        "both_ics": sorted(both), "auto_unexcluded_ics": sorted(auto_unexcluded),
        "ambiguous_ics": sorted(ambiguous), "final_excluded_ics": sorted(excluded_union),
        "veto_details": veto_results, "plotted_ics": plotted,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=str, default=None,
                         help="Comma-separated subject list. Default: ALL subjects in the dataset.")
    parser.add_argument("--n_workers", type=int, default=5)
    parser.add_argument("--qc_out_dir", type=str,
                         default=os.path.join(OUTPUT_PATH, "eog_fast_full_check"))
    args = parser.parse_args()

    subjects = [s.strip() for s in args.subjects.split(",")] if args.subjects else ALL_SUBJECTS
    os.makedirs(args.qc_out_dir, exist_ok=True)
    print(f"\n=== FAST FULL EOG CHECK on {len(subjects)} subjects, n_workers={args.n_workers} ===\n")

    if args.n_workers <= 1:
        results = [process_subject_full(s, args.qc_out_dir) for s in subjects]
    else:
        with mp.Pool(processes=args.n_workers) as pool:
            results = pool.starmap(process_subject_full, [(s, args.qc_out_dir) for s in subjects])

    df = pd.DataFrame([r for r in results if r.get("error") is None])
    errors = [r for r in results if r.get("error") is not None]
    for e in errors:
        print(f"ERROR: {e['subject_id']}: {e['error']}")

    if not df.empty and "veto_details" in df.columns:
        df["veto_details"] = df["veto_details"].apply(json.dumps)   # dict -> JSON string for CSV

    summary_path = os.path.join(args.qc_out_dir, "full_check_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"\nFull summary saved: {summary_path}")

    total_auto_unexcluded = int(df["n_auto_unexcluded"].sum()) if not df.empty else 0
    total_ambiguous = int(df["n_ambiguous"].sum()) if not df.empty else 0
    print(f"\n=== Spectral veto totals across all subjects: "
          f"{total_auto_unexcluded} IC(s) auto-unexcluded (false positives caught), "
          f"{total_ambiguous} IC(s) remain ambiguous (need a human look) ===")

    flagged = df[df["needs_review"]] if not df.empty else df
    print(f"\n=== {len(flagged)} / {len(df)} subjects FLAGGED for review ===")
    if not flagged.empty:
        print(flagged[["subject_id", "n_ics", "n_excluded", "pct_excluded",
                        "n_auto_unexcluded", "n_ambiguous"]].to_string(index=False))
        print(f"\nPlots for AMBIGUOUS ICs only saved under: {args.qc_out_dir}/<subject_id>/")
    else:
        print("No subjects flagged -- after the automated spectral veto, exclusion rates and "
              "IC counts look consistent for all subjects. No manual visual review strictly "
              "required, though a spot-check is still good practice for the thesis writeup.")

    # Starter decision file, pre-populated ONLY with subjects that have ambiguous ICs
    ambiguous_subjects = df[df["n_ambiguous"] > 0] if not df.empty else df
    starter = {
        "_comment": (
            "Only subjects with at least one AMBIGUOUS IC (i.e. the automated spectral veto "
            "could not confidently classify it as ocular vs. brain) are listed here. Everything "
            "else was already resolved automatically: KEEP_EXCLUDED (genuine ocular) or "
            "AUTO_UNEXCLUDE (false positive, already removed from the exclusion set -- see "
            "final_excluded_ics / auto_unexcluded_ics in full_check_summary.csv). Fill in "
            "suggested_threshold/reason after reviewing the AMBIGUOUS plots for that subject."
        ),
        "default_threshold": DEFAULT_EOG_THRESHOLD,
        "overrides": {
            row["subject_id"]: {"suggested_threshold": None,
                                 "reason": f"AMBIGUOUS ICs: {row['ambiguous_ics']}, TODO: review plots"}
            for _, row in ambiguous_subjects.iterrows()
        } if not ambiguous_subjects.empty else {},
    }
    starter_path = os.path.join(args.qc_out_dir, "subject_specific_eog_thresholds.STARTER.json")
    with open(starter_path, "w") as f:
        json.dump(starter, f, indent=2, ensure_ascii=False)
    print(f"Starter decision file: {starter_path}")


if __name__ == "__main__":
    main()
