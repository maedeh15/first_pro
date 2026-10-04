# ============================================================================
# check_eeg_amplitude_diagnostic.py
#
# QUICK DIAGNOSTIC -- run this BEFORE trusting check_window_alignment.py's
# results. It prints amplitude statistics (raw vs after full cleaning) so we
# can see exactly why almost every window exceeds the 120uV threshold.
#
# EDIT BASE_DATASET_PATH and SUBJECT_ID below, then run.
# ============================================================================

import numpy as np
import scipy.io as sio
import mne
import mne_icalabel
import os
import warnings
warnings.filterwarnings('ignore')

BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_ID = "VP001"
SFREQ_EEG = 200.0

eeg_folder = os.path.join(BASE_DATASET_PATH, f"{SUBJECT_ID}-EEG")
cnt_path = os.path.join(eeg_folder, 'cnt_nback.mat')

mat_cnt = sio.loadmat(cnt_path, struct_as_record=False, squeeze_me=True)
real_keys = [k for k in mat_cnt.keys() if not k.startswith('__')]
cnt_struct = mat_cnt[real_keys[0]]

data = cnt_struct.x.T
sfreq = float(cnt_struct.fs)
ch_names = [str(ch).strip() for ch in cnt_struct.clab]

print(f"Raw cnt.x stats BEFORE any scaling:")
print(f"  shape: {data.shape}")
print(f"  dtype: {data.dtype}")
print(f"  max abs value: {np.max(np.abs(data)):.6f}")
print(f"  mean abs value: {np.mean(np.abs(data)):.6f}")
print(f"  median abs value: {np.median(np.abs(data)):.6f}")
print(f"  99th percentile abs value: {np.percentile(np.abs(data), 99):.6f}")
print(f"  per-channel max (first 10 channels): {np.max(np.abs(data), axis=1)[:10]}")

# apply the SAME conversion logic as the main pipeline
if np.max(np.abs(data)) > 1.0:
    print("\n  -> max abs > 1.0, applying *1e-6 (assuming input was in microvolts)")
    data_scaled = data * 1e-6
else:
    print("\n  -> max abs <= 1.0, NOT scaling (assuming input already in volts)")
    data_scaled = data.copy()

print(f"\nAfter scaling (should now be in VOLTS, so ~1e-5 to 1e-4 range is normal EEG):")
print(f"  max abs value: {np.max(np.abs(data_scaled)):.8f}  ({np.max(np.abs(data_scaled))*1e6:.2f} uV)")
print(f"  mean abs value: {np.mean(np.abs(data_scaled)):.8f}  ({np.mean(np.abs(data_scaled))*1e6:.2f} uV)")
print(f"  99th percentile: {np.percentile(np.abs(data_scaled), 99)*1e6:.2f} uV")
print(f"  fraction of ALL samples exceeding 120uV: "
      f"{100 * np.mean(np.abs(data_scaled) > 120e-6):.2f}%")

if sfreq != SFREQ_EEG:
    from scipy.signal import resample
    n_samples = int(data_scaled.shape[1] * SFREQ_EEG / sfreq)
    data_scaled = resample(data_scaled, n_samples, axis=1)
    sfreq = SFREQ_EEG

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
raw = mne.io.RawArray(data_scaled, info, verbose=False)
eog_channels = [ch for ch in ['VEOG', 'HEOG'] if ch in ch_names]
if eog_channels:
    raw.set_channel_types({ch: 'eog' for ch in eog_channels})
raw.set_montage('standard_1005', match_case=False, on_missing='ignore')

print("\n--- Running full cleaning pipeline (filter + bad-channel + ICA) ---")
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
print(f"Bad channels detected: {bad_channels}")
if bad_channels:
    raw_filtered.info['bads'] = bad_channels
if raw_filtered.info['bads']:
    raw_filtered.interpolate_bads(reset_bads=True, verbose=False)

raw_filtered.set_eeg_reference('average', projection=False, verbose=False)

data_after_filter_ref = raw_filtered.get_data(picks='eeg')
print(f"\nAfter filter+bad-channel-interp+average-reference (BEFORE ICA):")
print(f"  max abs value: {np.max(np.abs(data_after_filter_ref))*1e6:.2f} uV")
print(f"  mean abs value: {np.mean(np.abs(data_after_filter_ref))*1e6:.2f} uV")
print(f"  99th percentile: {np.percentile(np.abs(data_after_filter_ref), 99)*1e6:.2f} uV")
print(f"  fraction of samples exceeding 120uV: "
      f"{100 * np.mean(np.abs(data_after_filter_ref) > 120e-6):.2f}%")

picks_for_ica = mne.pick_types(raw_filtered.info, eeg=True, eog=False, exclude='bads')
ica = mne.preprocessing.ICA(n_components=0.99, method='picard',
                             fit_params=dict(ortho=False, extended=True),
                             random_state=42, verbose=False)
ica.fit(raw_filtered, picks=picks_for_ica)
ic_labels = mne_icalabel.label_components(raw_filtered, ica, method='iclabel')
proba_matrix = ica.labels_scores_

ARTIFACT_TARGETS = ['eye blink', 'eye movement', 'muscle artifact', 'line noise']
artifact_idx_argmax = [idx for idx, label in enumerate(ic_labels['labels']) if label in ARTIFACT_TARGETS]
print(f"\nICA: {proba_matrix.shape[0]} components fit, argmax-flagged for exclusion: {artifact_idx_argmax}")
for idx in range(proba_matrix.shape[0]):
    print(f"  IC{idx}: label={ic_labels['labels'][idx]}  probs={np.round(proba_matrix[idx], 2)}")

ica.exclude = artifact_idx_argmax
raw_clean = raw_filtered.copy()
if artifact_idx_argmax:
    ica.apply(raw_clean, verbose=False)

data_clean = raw_clean.get_data(picks='eeg')
print(f"\nAfter full ICA cleaning (argmax-only exclusion, for this quick diagnostic):")
print(f"  max abs value: {np.max(np.abs(data_clean))*1e6:.2f} uV")
print(f"  mean abs value: {np.mean(np.abs(data_clean))*1e6:.2f} uV")
print(f"  99th percentile: {np.percentile(np.abs(data_clean), 99)*1e6:.2f} uV")
print(f"  fraction of samples exceeding 120uV: "
      f"{100 * np.mean(np.abs(data_clean) > 120e-6):.2f}%")
print(f"  per-channel max abs (uV): {np.round(np.max(np.abs(data_clean), axis=1) * 1e6, 1)}")
print(f"  per-channel 99th pct (uV): {np.round(np.percentile(np.abs(data_clean), 99, axis=1) * 1e6, 1)}")

# find which channel(s)/times drive the worst violations
worst_ch_idx = np.argmax(np.max(np.abs(data_clean), axis=1))
worst_ch_name = raw_clean.ch_names[[i for i in range(len(raw_clean.ch_names)) if raw_clean.get_channel_types()[i]=='eeg'][worst_ch_idx]] if False else eeg_names[worst_ch_idx] if worst_ch_idx < len(eeg_names) else "?"
print(f"\nWorst channel by max abs amplitude: index {worst_ch_idx}")
worst_trace = data_clean[worst_ch_idx]
worst_time_idx = np.argmax(np.abs(worst_trace))
print(f"  peak value: {worst_trace[worst_time_idx]*1e6:.1f} uV at sample {worst_time_idx} "
      f"(t={worst_time_idx/sfreq:.1f}s)")
