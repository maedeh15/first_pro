# ============================================================================
# check_ica_attribution_sensitivity.py
#
# DIAGNOSTIC SCRIPT -- does NOT touch or import the main preprocessing pipeline.
#
# Purpose: for the 3-layer ICA artifact-rejection logic used in preprocess_eeg()
# (argmax ICLabel + probability-threshold ICLabel + EOG-correlation, with a
# "brain-probability rescue" gate on EOG-only hits), answer three questions
# PER SUBJECT without needing any model training:
#
#   (1) ATTRIBUTION: for each excluded IC, which criterion (or combination)
#       caused its exclusion? (argmax-only / prob-threshold-only / EOG-only /
#       some combination)
#   (2) THRESHOLD SENSITIVITY: how many ICs get excluded/rescued as
#       PROB_THRESHOLD and BRAIN_PROB_CEILING are swept over a small grid of
#       nearby values? (tests whether 0.30 / 0.50 are load-bearing choices or
#       arbitrary-but-harmless)
#   (3) SIMPLIFIED-VS-FULL COMPARISON: how many ICs differ between the FULL
#       3-layer logic and a SIMPLE argmax-only baseline (the most common,
#       easiest-to-defend approach in the literature)?
#
# This script re-fits ICA itself (ICA fitting is required to get labels/scores/
# EOG-correlations at all), using the EXACT same filtering / bad-channel /
# referencing / ICA-fit settings as preprocess_eeg() in the main pipeline, so
# the ICA solution being analyzed here is representative. It does NOT apply
# ICA, does NOT save any cleaned data, and does NOT write anything back to the
# main pipeline's output folders -- it only reports numbers about IC labels.
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH and SUBJECT_IDS below.
#   2. Run: python check_ica_attribution_sensitivity.py
#   3. Read the printed tables and the saved CSVs:
#        - ica_attribution_report.csv       (per-IC, per-subject attribution)
#        - ica_threshold_sensitivity.csv    (IC count vs threshold value)
#        - ica_simple_vs_full_comparison.csv
# ============================================================================

import os
import numpy as np
import scipy.io as sio
import mne
import mne_icalabel
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_IDS = ["VP001", "VP002", "VP003", "VP004", "VP005"]   # start with ~5, extend once trusted

SFREQ_EEG = 200.0

# current pipeline's operating point (from preprocess_eeg())
PROB_THRESHOLD_CURRENT = 0.30
BRAIN_PROB_CEILING_CURRENT = 0.5

# sweep grids for sensitivity analysis
PROB_THRESHOLD_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
BRAIN_PROB_CEILING_GRID = [0.35, 0.45, 0.50, 0.55, 0.65]

ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
ARTIFACT_CLASSES_IDX = {
    'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
    'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6
}
ALL_CLASS_NAMES = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
                    'line noise', 'channel noise', 'other']


# ----------------------------------------------------------------------------
# Loading + preprocessing up to (and including) ICA fit -- mirrors
# load_eeg_data() + the pre-ICA portion of preprocess_eeg() in the main
# pipeline, kept standalone/independent here.
# ----------------------------------------------------------------------------
def load_eeg_raw(subject_id):
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt_struct = mat_cnt[real_keys[0]]

    data = cnt_struct.x.T
    sfreq = float(cnt_struct.fs)
    ch_names = [str(ch).strip() for ch in cnt_struct.clab]

    if np.max(np.abs(data)) > 1.0:
        data = data * 1e-6

    if sfreq != SFREQ_EEG:
        from scipy.signal import resample
        n_samples = int(data.shape[1] * SFREQ_EEG / sfreq)
        data = resample(data, n_samples, axis=1)
        sfreq = SFREQ_EEG

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info, verbose=False)

    eog_channels = [ch for ch in ['VEOG', 'HEOG'] if ch in ch_names]
    if eog_channels:
        raw.set_channel_types({ch: 'eog' for ch in eog_channels})
    raw.set_montage('standard_1005', match_case=False, on_missing='ignore')

    return raw


def fit_ica_for_subject(subject_id):
    """Reproduces preprocess_eeg()'s pipeline up to and including ICA fit +
    ICLabel + EOG-correlation scoring. Returns a dict with everything needed
    for attribution/sensitivity analysis. Does NOT apply ICA to the data."""
    raw = load_eeg_raw(subject_id)
    raw_original = raw.copy()   # kept for find_bads_eog(), matches main pipeline's use of the unfiltered `raw`

    raw_filtered = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    raw_filtered.filter(l_freq=1.0, h_freq=40.0, method='fir', verbose=False)

    # bad channel detection (same heuristic as main pipeline)
    raw_temp = raw_filtered.copy().filter(l_freq=10.0, h_freq=40.0, method='fir', verbose=False)
    eeg_picks = mne.pick_types(raw_temp.info, eeg=True, eog=False)
    eeg_names = [raw_temp.ch_names[idx] for idx in eeg_picks]
    temp_data = raw_temp.get_data(picks=eeg_picks)
    stds = np.std(temp_data, axis=1)
    median_std = np.median(stds)
    mad = np.median(np.abs(stds - median_std)) if np.median(np.abs(stds - median_std)) > 0 else 1e-6
    corr_matrix = np.corrcoef(temp_data)
    mean_corrs = np.mean(corr_matrix, axis=1)

    bad_channels = []
    for i, ch_name in enumerate(eeg_names):
        if ch_name.upper() in ['VEOG', 'HEOG', 'TP9']:
            continue
        if stds[i] > (median_std + 4.5 * mad) or stds[i] < (median_std - 4.5 * mad) or mean_corrs[i] < 0.30:
            bad_channels.append(ch_name)

    if bad_channels:
        raw_filtered.info['bads'] = bad_channels

    interpolated_channels = list(raw_filtered.info['bads'])
    if raw_filtered.info['bads']:
        raw_filtered.interpolate_bads(reset_bads=True, verbose=False)

    raw_filtered.set_eeg_reference('average', projection=False, verbose=False)

    # ICA fit (same settings as main pipeline)
    picks_for_ica = mne.pick_types(raw_filtered.info, eeg=True, eog=False, exclude='bads')
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard',
                                 fit_params=dict(ortho=False, extended=True),
                                 random_state=42, verbose=False)
    ica.fit(raw_filtered, picks=picks_for_ica)

    ic_labels = mne_icalabel.label_components(raw_filtered, ica, method='iclabel')
    proba_matrix = ica.labels_scores_   # shape (n_components, 7)

    eog_indices_veog, eog_indices_heog = [], []
    try:
        if 'VEOG' in raw_original.ch_names:
            eog_indices_veog, _ = ica.find_bads_eog(raw_original, ch_name='VEOG', threshold=2.0, verbose=False)
        if 'HEOG' in raw_original.ch_names:
            eog_indices_heog, _ = ica.find_bads_eog(raw_original, ch_name='HEOG', threshold=2.0, verbose=False)
    except Exception as e:
        print(f"  [{subject_id}] find_bads_eog failed: {e}")

    return {
        'subject_id': subject_id,
        'n_components': proba_matrix.shape[0],
        'labels': ic_labels['labels'],
        'proba_matrix': proba_matrix,
        'eog_indices_veog': set(eog_indices_veog),
        'eog_indices_heog': set(eog_indices_heog),
        'interpolated_channels': interpolated_channels,
    }


# ----------------------------------------------------------------------------
# Decision logic -- mirrors preprocess_eeg()'s exclusion logic exactly, but
# parameterized by (prob_threshold, brain_prob_ceiling) so it can be swept.
# ----------------------------------------------------------------------------
def compute_exclusion(fit_result, prob_threshold, brain_prob_ceiling):
    labels = fit_result['labels']
    proba_matrix = fit_result['proba_matrix']
    eog_veog = fit_result['eog_indices_veog']
    eog_heog = fit_result['eog_indices_heog']
    n_components = fit_result['n_components']

    argmax_idx = {idx for idx, label in enumerate(labels) if label in ARTIFACT_TARGETS}

    prob_idx = set()
    for comp_idx in range(n_components):
        for cls_name in ARTIFACT_TARGETS:
            cls_col = ARTIFACT_CLASSES_IDX.get(cls_name)
            if cls_col is not None and proba_matrix[comp_idx, cls_col] >= prob_threshold:
                prob_idx.add(comp_idx)
                break

    eog_idx = eog_veog | eog_heog
    brain_col = ARTIFACT_CLASSES_IDX['brain']

    eog_only_idx = eog_idx - (argmax_idx | prob_idx)
    eog_only_rescued = {idx for idx in eog_only_idx if proba_matrix[idx, brain_col] >= brain_prob_ceiling}

    final_excluded = (argmax_idx | prob_idx | eog_idx) - eog_only_rescued

    return {
        'argmax_idx': argmax_idx,
        'prob_idx': prob_idx,
        'eog_idx': eog_idx,
        'eog_veog': eog_veog,
        'eog_heog': eog_heog,
        'eog_only_rescued': eog_only_rescued,
        'final_excluded': final_excluded,
    }


def attribution_reason(comp_idx, decision):
    reasons = []
    if comp_idx in decision['argmax_idx']:
        reasons.append('argmax')
    if comp_idx in decision['prob_idx']:
        reasons.append('prob_threshold')
    if comp_idx in decision['eog_veog']:
        reasons.append('VEOG_corr')
    if comp_idx in decision['eog_heog']:
        reasons.append('HEOG_corr')
    return '+'.join(reasons) if reasons else 'none'


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    attribution_rows = []
    sensitivity_rows = []
    simple_vs_full_rows = []

    for sid in SUBJECT_IDS:
        print(f"\n{'='*70}\nProcessing {sid}\n{'='*70}")
        try:
            fit_result = fit_ica_for_subject(sid)
        except Exception as e:
            print(f"  FAILED to fit ICA for {sid}: {e}")
            continue

        # ---- (1) ATTRIBUTION at the CURRENT operating point ----
        decision_current = compute_exclusion(fit_result, PROB_THRESHOLD_CURRENT, BRAIN_PROB_CEILING_CURRENT)
        print(f"  n_components={fit_result['n_components']}  "
              f"excluded (current settings)={len(decision_current['final_excluded'])}")

        for comp_idx in sorted(decision_current['final_excluded']):
            reason = attribution_reason(comp_idx, decision_current)
            probs = fit_result['proba_matrix'][comp_idx]
            row = {
                'subject': sid,
                'ic_index': comp_idx,
                'iclabel_argmax': fit_result['labels'][comp_idx],
                'reason': reason,
                'rescued': comp_idx in decision_current['eog_only_rescued'],
            }
            for i, cname in enumerate(ALL_CLASS_NAMES):
                row[f'prob_{cname}'] = probs[i]
            attribution_rows.append(row)
            print(f"    IC{comp_idx}: argmax='{fit_result['labels'][comp_idx]}'  reason=[{reason}]")

        # summarize attribution categories for this subject
        excluded = decision_current['final_excluded']
        only_argmax = sum(1 for c in excluded if attribution_reason(c, decision_current) == 'argmax')
        only_prob = sum(1 for c in excluded if attribution_reason(c, decision_current) == 'prob_threshold')
        only_eog = sum(1 for c in excluded
                        if attribution_reason(c, decision_current) in ('VEOG_corr', 'HEOG_corr', 'VEOG_corr+HEOG_corr'))
        combo = len(excluded) - only_argmax - only_prob - only_eog
        print(f"  breakdown: argmax-only={only_argmax}  prob-only={only_prob}  "
              f"EOG-only={only_eog}  combination={combo}  "
              f"rescued={len(decision_current['eog_only_rescued'])}")

        # ---- (2) THRESHOLD SENSITIVITY ----
        for pt in PROB_THRESHOLD_GRID:
            decision = compute_exclusion(fit_result, pt, BRAIN_PROB_CEILING_CURRENT)
            sensitivity_rows.append({
                'subject': sid, 'swept_param': 'PROB_THRESHOLD', 'value': pt,
                'n_excluded': len(decision['final_excluded']),
                'excluded_ics': sorted(decision['final_excluded']),
            })
        for bc in BRAIN_PROB_CEILING_GRID:
            decision = compute_exclusion(fit_result, PROB_THRESHOLD_CURRENT, bc)
            sensitivity_rows.append({
                'subject': sid, 'swept_param': 'BRAIN_PROB_CEILING', 'value': bc,
                'n_excluded': len(decision['final_excluded']),
                'excluded_ics': sorted(decision['final_excluded']),
            })

        # ---- (3) SIMPLE (argmax-only) VS FULL comparison ----
        simple_excluded = decision_current['argmax_idx']   # argmax-only baseline
        full_excluded = decision_current['final_excluded']
        only_in_full = full_excluded - simple_excluded
        only_in_simple = simple_excluded - full_excluded   # should be empty by construction, but check anyway
        shared = full_excluded & simple_excluded
        simple_vs_full_rows.append({
            'subject': sid,
            'n_simple_argmax_only': len(simple_excluded),
            'n_full_pipeline': len(full_excluded),
            'n_shared': len(shared),
            'n_only_in_full': len(only_in_full),
            'only_in_full_ics': sorted(only_in_full),
            'n_only_in_simple': len(only_in_simple),
            'pct_agreement': 100 * len(shared) / max(len(simple_excluded | full_excluded), 1),
        })
        print(f"  simple(argmax-only)={len(simple_excluded)}  full={len(full_excluded)}  "
              f"shared={len(shared)}  only_in_full={sorted(only_in_full)}")

    # ---- Save reports ----
    out_dir = os.path.dirname(os.path.abspath(__file__))

    df_attr = pd.DataFrame(attribution_rows)
    df_attr.to_csv(os.path.join(out_dir, "ica_attribution_report.csv"), index=False, encoding='utf-8-sig')

    df_sens = pd.DataFrame(sensitivity_rows)
    df_sens.to_csv(os.path.join(out_dir, "ica_threshold_sensitivity.csv"), index=False, encoding='utf-8-sig')

    df_simple = pd.DataFrame(simple_vs_full_rows)
    df_simple.to_csv(os.path.join(out_dir, "ica_simple_vs_full_comparison.csv"), index=False, encoding='utf-8-sig')

    print(f"\n{'='*70}")
    print("SAVED REPORTS:")
    print(f"  {os.path.join(out_dir, 'ica_attribution_report.csv')}")
    print(f"  {os.path.join(out_dir, 'ica_threshold_sensitivity.csv')}")
    print(f"  {os.path.join(out_dir, 'ica_simple_vs_full_comparison.csv')}")
    print(f"{'='*70}")

    print("\nHOW TO READ ica_threshold_sensitivity.csv:")
    print("  Group by subject + swept_param, look at how n_excluded changes across the grid.")
    print("  If n_excluded is IDENTICAL (or off by 0-1) across the whole PROB_THRESHOLD grid")
    print("  (0.20 to 0.40), the exact value 0.30 is not load-bearing -- defensible as-is.")
    print("  If n_excluded changes by 2+ at every step, the threshold is a fragile, sensitive")
    print("  choice and should be validated (e.g. against manual expert IC labeling on a subset)")
    print("  rather than asserted.")

    print("\nHOW TO READ ica_simple_vs_full_comparison.csv:")
    print("  pct_agreement close to 100% means the 3-layer logic barely changes anything vs")
    print("  plain argmax -- you can simplify the pipeline (and the thesis defense) with no cost.")
    print("  Low pct_agreement means the extra layers are doing real, substantial work, and you")
    print("  need independent evidence (e.g. visual IC inspection, or the accuracy-impact run in")
    print("  step 4) that this extra work is correct rather than noise.")


if __name__ == "__main__":
    main()
