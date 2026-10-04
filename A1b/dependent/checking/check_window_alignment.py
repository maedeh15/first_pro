# ============================================================================
# check_window_alignment.py
#
# DIAGNOSTIC / ALIGNMENT SCRIPT -- does NOT modify the main pipeline or its
# saved outputs. Read-only with respect to both.
#
# PROBLEM THIS SOLVES:
# The main pipeline's extract_synchronized_windows() builds windows by looping
# trial_idx -> offset_sec (0..30s, step 1s) and SKIPS a window if:
#   (a) the EEG window would run past the end of the recording (w_end_eeg > n_times), or
#   (b) the raw EEG slice's max abs amplitude exceeds 120 uV (artifact rejection), or
#   (c) any of the FNIRS_NUM_SEGMENTS fNIRS segments would run out of bounds.
# The saved outputs only contain trial_ids.npy and session_ids.npy (one entry per
# SURVIVING window, in loop order) -- NOT which offset_sec each surviving window
# came from. Meanwhile, generate_baseline_ablation_datasets.py (the fNIRS-only
# ablation generator) only checks condition (c) -- it has no access to the EEG
# amplitude check (a)/(b), so it may generate MORE windows than the main
# pipeline actually kept for training.
#
# WHAT THIS SCRIPT DOES:
#   1. Independently reconstructs, per subject, exactly which (trial_id,
#      offset_sec) pairs survive ALL THREE checks -- by re-running the EEG
#      amplitude check on the real raw EEG data, using the exact same
#      thresholds/logic as extract_synchronized_windows().
#   2. VALIDATES this reconstruction: groups the reconstructed surviving
#      windows by trial_id and compares the per-trial COUNT against the
#      per-trial count found in the real, saved trial_ids.npy from the main
#      pipeline. If counts match for every trial, the reconstruction is
#      almost certainly correct (offsets are visited in strictly increasing
#      order per trial with no gaps in the trial_ids array, so an identical
#      per-trial count is strong evidence the exact same offsets were kept).
#   3. Saves a per-subject alignment key CSV: (trial_id, offset_sec, kept_by_main_pipeline)
#      that generate_baseline_ablation_datasets.py's output can be filtered
#      against, so downstream training only ever sees windows the main
#      pipeline itself would have produced.
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH, MAIN_PIPELINE_OUTPUT_PATH, SUBJECT_IDS below.
#   2. Run: python check_window_alignment.py
#   3. Check the printed validation summary: "counts match" should be True
#      for every subject/trial. If any mismatch is reported, DO NOT trust the
#      alignment key for that subject until investigated (see the printed
#      diagnostic detail for that subject).
#   4. Use the saved alignment_key_{subject}.csv files as a filter when
#      loading the ablation-generated fNIRS .npz files for training.
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
# Path where the main pipeline's per-subject outputs (trial_ids.npy etc.) now live
# (moved by user from the original OUTPUT_PATH location).
MAIN_PIPELINE_OUTPUT_PATH = r"F:\preprocessing-history-thesis\MULTIMODAL_PREPROCESSED_RUN_A1b_10s-onetime ica"
SUBJECT_IDS = ["VP001", "VP002", "VP003"]   # extend after validating on a few

SFREQ_EEG = 200.0
SFREQ_FNIRS = 10.0
EEG_WINDOW_LEN = 10.0
EEG_EPOCH_TMAX = 40.0
EEG_STEP = 1.0
FNIRS_SEGMENT_LEN_SEC = 3.0
FNIRS_STEP = 1.0
FNIRS_NUM_SEGMENTS = 21
TRIALS_PER_SESSION = 9
EEG_AMPLITUDE_THRESHOLD = 120e-6   # matches main pipeline's 120 uV artifact rejection

EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}   # trial-onset codes -> class (EDIT if your codes differ)
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}       # fNIRS trial-onset codes -> class

# ICA / artifact-rejection settings (mirror preprocess_eeg() exactly, since the 120uV amplitude
# check in the main pipeline runs on the FULLY CLEANED signal -- filtered, referenced, ICA-applied
# -- not on the raw signal. Skipping this step was an earlier bug in this script: it produced 0/837
# windows kept because raw, unfiltered EEG routinely exceeds 120uV from drift/DC-offset alone.)
ARTIFACT_CLASSES_IDX = {
    'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
    'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6
}
ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
PROB_THRESHOLD = 0.30
BRAIN_PROB_CEILING = 0.5


# ----------------------------------------------------------------------------
# Standalone loaders (mirror main pipeline, independent copy, read-only)
# ----------------------------------------------------------------------------
def load_eeg_raw_and_events(subject_id):
    """Loads raw EEG + events, exactly as load_eeg_data() does in the main pipeline
    (unfiltered -- filtering/cleaning happens in preprocess_eeg_clean() below)."""
    eeg_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-EEG")
    cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(eeg_folder, 'mrk_nback.mat')

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

    mat_mrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    real_mrk_keys = [k for k in mat_mrk.keys() if not k.startswith('__')]
    emrk = mat_mrk[real_mrk_keys[0]]

    if hasattr(emrk, 'time'):
        event_times_ms = np.asarray(emrk.time, dtype=float)
    elif hasattr(emrk, 'pos'):
        event_times_ms = np.asarray(emrk.pos, dtype=float) / sfreq * 1000.0
    else:
        raise ValueError("no time/pos field in EEG marker struct")
    event_codes = np.asarray(emrk.event.desc).flatten()
    event_samples = np.round(event_times_ms / 1000.0 * sfreq).astype(int)

    events = np.column_stack([event_samples, np.zeros_like(event_samples), event_codes]).astype(int)
    return raw, events


def preprocess_eeg_clean(raw):
    """Reproduces preprocess_eeg() from the main pipeline in full: notch+bandpass filter,
    bad-channel detection/interpolation, average reference, ICA fit+apply (argmax + prob
    threshold + EOG-correlation with brain-probability rescue gate). Returns the FULLY
    CLEANED raw object -- this is what the 120uV amplitude check must run on, matching
    the main pipeline's eeg_slice_raw = raw_eeg_clean.get_data(...) exactly."""
    raw_filtered = raw.copy().notch_filter(freqs=50.0, picks='eeg', verbose=False)
    raw_filtered.filter(l_freq=1.0, h_freq=40.0, method='fir', verbose=False)

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
    if raw_filtered.info['bads']:
        raw_filtered.interpolate_bads(reset_bads=True, verbose=False)

    raw_filtered.set_eeg_reference('average', projection=False, verbose=False)

    picks_for_ica = mne.pick_types(raw_filtered.info, eeg=True, eog=False, exclude='bads')
    ica = mne.preprocessing.ICA(n_components=0.99, method='picard',
                                 fit_params=dict(ortho=False, extended=True),
                                 random_state=42, verbose=False)
    ica.fit(raw_filtered, picks=picks_for_ica)

    ic_labels = mne_icalabel.label_components(raw_filtered, ica, method='iclabel')
    proba_matrix = ica.labels_scores_

    artifact_idx_argmax = [idx for idx, label in enumerate(ic_labels['labels']) if label in ARTIFACT_TARGETS]
    artifact_idx_prob = []
    for comp_idx in range(proba_matrix.shape[0]):
        for cls_name in ARTIFACT_TARGETS:
            cls_col = ARTIFACT_CLASSES_IDX.get(cls_name)
            if cls_col is not None and proba_matrix[comp_idx, cls_col] >= PROB_THRESHOLD:
                artifact_idx_prob.append(comp_idx)
                break

    eog_indices_veog, eog_indices_heog = [], []
    try:
        if 'VEOG' in raw.ch_names:
            eog_indices_veog, _ = ica.find_bads_eog(raw, ch_name='VEOG', threshold=2.0, verbose=False)
        if 'HEOG' in raw.ch_names:
            eog_indices_heog, _ = ica.find_bads_eog(raw, ch_name='HEOG', threshold=2.0, verbose=False)
    except Exception as e:
        print(f"    find_bads_eog failed: {e}")

    eog_only_idx = (set(eog_indices_veog) | set(eog_indices_heog)) - (set(artifact_idx_argmax) | set(artifact_idx_prob))
    eog_only_rescued = {idx for idx in eog_only_idx if proba_matrix[idx, ARTIFACT_CLASSES_IDX['brain']] >= BRAIN_PROB_CEILING}

    artifact_idx = sorted(
        (set(artifact_idx_argmax) | set(artifact_idx_prob) | set(eog_indices_veog) | set(eog_indices_heog))
        - eog_only_rescued
    )
    ica.exclude = artifact_idx
    if len(ica.exclude) > 0:
        ica.apply(raw_filtered, verbose=False)

    # CORRECTED (was missing): drop VEOG/HEOG/TP9 before returning, exactly matching
    # preprocess_eeg()'s channels_to_drop step. Without this, get_data() below (used for the
    # 120uV amplitude check) still includes the raw, un-cleaned EOG channels -- eye-blink/movement
    # amplitude routinely exceeds 120uV and is NEVER touched by ICA (fit with eog=False), so
    # leaving these channels in caused nearly every window to fail the check incorrectly.
    channels_to_drop = [ch for ch in raw_filtered.ch_names if ch.upper() in ['VEOG', 'HEOG', 'TP9']]
    if channels_to_drop:
        raw_filtered.drop_channels(channels_to_drop)

    return raw_filtered


def load_fnirs_n_samples_and_markers(subject_id):
    """Only what's needed to determine fNIRS segment validity: total sample count
    (post-resample-to-10Hz) and trial marker times."""
    fnirs_folder = os.path.join(BASE_DATASET_PATH, f"{subject_id}-NIRS")
    cnt_path = os.path.join(fnirs_folder, 'cnt_nback.mat')
    mrk_path = os.path.join(fnirs_folder, 'mrk_nback.mat')

    mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
    real_cnt_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
    cnt = mat_cnt[real_cnt_keys[0]]
    oxy_raw = np.asarray(cnt.oxy.x, dtype=np.float64)
    sfreq_raw = float(cnt.oxy.fs) if hasattr(cnt.oxy, 'fs') else 10.4
    n_samples_10hz = int(round(oxy_raw.shape[0] * SFREQ_FNIRS / sfreq_raw))

    mat_mrk = sio.loadmat(mrk_path, struct_as_record=False, squeeze_me=True)
    real_mrk_keys = [k for k in mat_mrk.keys() if not k.startswith('__')]
    fmrk = mat_mrk[real_mrk_keys[0]]
    if hasattr(fmrk, 'time'):
        fnirs_events_time = np.asarray(fmrk.time, dtype=float)
    elif hasattr(fmrk, 'pos'):
        fnirs_events_time = np.asarray(fmrk.pos, dtype=float) / sfreq_raw * 1000.0
    else:
        raise ValueError("no time/pos field in fNIRS marker struct")
    fnirs_events_code = np.asarray(fmrk.event.desc).flatten()

    fnirs_mask = np.isin(fnirs_events_code, list(FNIRS_TRIAL_LABEL_MAP.keys()))
    trial_times_ms = fnirs_events_time[fnirs_mask]
    trial_codes = fnirs_events_code[fnirs_mask]
    order = np.argsort(trial_times_ms)
    fnirs_trial_times = trial_times_ms[order] / 1000.0
    fnirs_trial_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in trial_codes[order]])

    return n_samples_10hz, fnirs_trial_times, fnirs_trial_classes


# ----------------------------------------------------------------------------
# Reconstruction of the main pipeline's exact keep/skip decision per window
# ----------------------------------------------------------------------------
def reconstruct_valid_offsets(subject_id):
    raw, events = load_eeg_raw_and_events(subject_id)
    print(f"  [{subject_id}] running full EEG cleaning pipeline (filter+ICA) -- this is the slow step...")
    raw_clean = preprocess_eeg_clean(raw)
    n_times_eeg = raw_clean.n_times
    sfreq_eeg = raw_clean.info['sfreq']

    valid_codes = list(EEG_TRIAL_LABEL_MAP.keys())
    target_events = events[np.isin(events[:, 2], valid_codes)]
    print(f"  [{subject_id}] found {len(target_events)} EEG trial markers")

    n_samples_fnirs_10hz, fnirs_trial_times, fnirs_trial_classes = load_fnirs_n_samples_and_markers(subject_id)

    eeg_trial_classes = np.array([EEG_TRIAL_LABEL_MAP.get(int(ev[2]), -1) for ev in target_events])
    n_common = min(len(fnirs_trial_times), len(eeg_trial_classes))
    fnirs_sync_ok = (len(fnirs_trial_times) == len(target_events) and
                     np.array_equal(fnirs_trial_classes[:n_common], eeg_trial_classes[:n_common]))
    if not fnirs_sync_ok:
        print(f"  [{subject_id}] WARNING: fNIRS/EEG trial sync check failed -- "
              f"reconstruction may not match the main pipeline for this subject "
              f"(main pipeline would have fallen back to EEG-clock timing too, "
              f"so this script mirrors that fallback below).")

    window_samples_eeg = int(EEG_WINDOW_LEN * sfreq_eeg)
    window_samples_fnirs = int(FNIRS_SEGMENT_LEN_SEC * SFREQ_FNIRS)
    max_slide_seconds = EEG_EPOCH_TMAX - EEG_WINDOW_LEN
    offsets = np.arange(0.0, max_slide_seconds + 0.001, EEG_STEP)

    rows = []
    for trial_idx, ev in enumerate(target_events):
        event_sample_eeg = ev[0]
        raw_code = ev[2]
        label = EEG_TRIAL_LABEL_MAP.get(raw_code, -1)
        if label == -1:
            continue

        eeg_event_time = event_sample_eeg / sfreq_eeg
        if fnirs_sync_ok and trial_idx < len(fnirs_trial_times):
            fnirs_event_time = fnirs_trial_times[trial_idx]
        else:
            fnirs_event_time = eeg_event_time

        for offset_sec in offsets:
            w_start_eeg = int(event_sample_eeg + np.round(offset_sec * sfreq_eeg))
            w_end_eeg = w_start_eeg + window_samples_eeg

            reason = None
            if w_end_eeg > n_times_eeg:
                reason = 'eeg_out_of_bounds'
            else:
                eeg_slice_raw = raw_clean.get_data(start=w_start_eeg, stop=w_end_eeg)
                if np.max(np.abs(eeg_slice_raw)) > EEG_AMPLITUDE_THRESHOLD:
                    reason = 'eeg_amplitude_artifact'

            if reason is None:
                # also check fNIRS segment bounds (mirrors condition (c))
                fnirs_ok = True
                for seg_idx in range(FNIRS_NUM_SEGMENTS):
                    seg_offset = offset_sec + seg_idx * FNIRS_STEP
                    w_start_fnirs_seg = int(np.round((fnirs_event_time + seg_offset) * SFREQ_FNIRS))
                    w_end_fnirs_seg = w_start_fnirs_seg + window_samples_fnirs
                    if w_start_fnirs_seg < 0 or w_end_fnirs_seg > n_samples_fnirs_10hz:
                        fnirs_ok = False
                        break
                if not fnirs_ok:
                    reason = 'fnirs_out_of_bounds'

            rows.append({
                'subject': subject_id,
                'trial_id': trial_idx,
                'offset_sec': offset_sec,
                'kept_by_main_pipeline': reason is None,
                'skip_reason': reason if reason else 'kept',
            })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Validation against the real saved trial_ids.npy from the main pipeline
# ----------------------------------------------------------------------------
def validate_against_saved_outputs(subject_id, df_reconstructed):
    trial_ids_path = os.path.join(MAIN_PIPELINE_OUTPUT_PATH, subject_id, f"{subject_id}_trial_ids.npy")
    if not os.path.exists(trial_ids_path):
        print(f"  [{subject_id}] SKIP validation: {trial_ids_path} not found "
              f"(has the main pipeline been run for this subject yet?)")
        return None

    real_trial_ids = np.load(trial_ids_path)
    real_counts = pd.Series(real_trial_ids).value_counts().sort_index()

    recon_counts = (df_reconstructed[df_reconstructed['kept_by_main_pipeline']]
                    .groupby('trial_id').size())

    all_trials = sorted(set(real_counts.index) | set(recon_counts.index))
    mismatches = []
    for t in all_trials:
        real_n = int(real_counts.get(t, 0))
        recon_n = int(recon_counts.get(t, 0))
        if real_n != recon_n:
            mismatches.append((t, real_n, recon_n))

    n_match = len(all_trials) - len(mismatches)
    print(f"  [{subject_id}] validation: {n_match}/{len(all_trials)} trials have matching "
          f"window counts between reconstruction and real saved output.")
    if mismatches:
        print(f"    MISMATCHES (trial_id, real_count, reconstructed_count):")
        for t, real_n, recon_n in mismatches[:20]:
            print(f"      trial {t}: real={real_n}  reconstructed={recon_n}")
        if len(mismatches) > 20:
            print(f"      ... and {len(mismatches) - 20} more")
        print(f"    -> DO NOT TRUST the alignment key for this subject until this is resolved.")
        print(f"    Common causes: EEG_TRIAL_LABEL_MAP codes wrong for this subject/dataset variant,")
        print(f"    EEG_AMPLITUDE_THRESHOLD differs from what the main pipeline actually used, or")
        print(f"    the main pipeline's saved output is from a DIFFERENT run/config than the constants")
        print(f"    set at the top of this script.")
    else:
        print(f"    All trial window counts match exactly -- reconstruction is validated for {subject_id}.")

    return len(mismatches) == 0


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    all_validations = {}

    for sid in SUBJECT_IDS:
        print(f"\n{'='*70}\n{sid}\n{'='*70}")
        try:
            df = reconstruct_valid_offsets(sid)
        except Exception as e:
            print(f"  FAILED to reconstruct for {sid}: {e}")
            continue

        is_valid = validate_against_saved_outputs(sid, df)
        all_validations[sid] = is_valid

        out_csv = os.path.join(out_dir, f"alignment_key_{sid}.csv")
        df.to_csv(out_csv, index=False, encoding='utf-8-sig')
        n_kept = df['kept_by_main_pipeline'].sum()
        print(f"  saved alignment key -> {out_csv}  ({n_kept}/{len(df)} offsets kept)")

        reason_counts = df.loc[~df['kept_by_main_pipeline'], 'skip_reason'].value_counts()
        if len(reason_counts):
            print(f"  skip reasons: {dict(reason_counts)}")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for sid, is_valid in all_validations.items():
        status = "VALIDATED" if is_valid else ("MISMATCH -- do not trust" if is_valid is False else "not checked (no saved output found)")
        print(f"  {sid}: {status}")

    print("\nHOW TO USE THE alignment_key_{subject}.csv FILES:")
    print("  When loading a window from generate_baseline_ablation_datasets.py's output")
    print("  (matched by trial_id + offset_sec), only keep it if the alignment key marks")
    print("  kept_by_main_pipeline=True for that (trial_id, offset_sec) pair. This guarantees")
    print("  your baseline-ablation training set contains EXACTLY the same windows (by trial")
    print("  and offset) that the main pipeline's EEG branch would have -- i.e. you are only")
    print("  ever comparing the effect of the baseline definition, not accidentally comparing")
    print("  different sets of windows.")
