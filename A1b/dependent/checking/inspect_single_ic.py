# ============================================================================
# inspect_single_ic.py
#
# DIAGNOSTIC SCRIPT -- does NOT modify the main pipeline or any saved output.
#
# Purpose: visually inspect ONE specific IC (by default, VP018's IC6 -- the
# component argmax labeled 'other' but which prob_threshold flagged and
# excluded) so you can judge by eye whether it looks like a real artifact
# (eye movement / muscle / drift) or a plausible brain component that was
# a false positive.
#
# Produces, for the target IC:
#   1. Scalp topography (topomap) -- artifacts often have a very
#      characteristic spatial pattern (e.g. frontal for eye blinks, temporal
#      for muscle) whereas brain components usually look more diffuse/
#      dipolar over central-parietal areas.
#   2. Time series of the IC's activation over the whole recording -- artifacts
#      often show sparse large spikes (blinks) or continuous high-frequency
#      buzz (muscle), while brain components look more like ongoing oscillatory
#      activity.
#   3. Power spectral density (PSD) of the IC -- muscle artifacts show a
#      broadband high-frequency hump (>20-30Hz), eye movements show low-frequency
#      power, brain components typically show the classic 1/f + alpha bump shape.
#   4. The full ICLabel probability vector for this IC, printed as text.
#
# Output: inspect_ic_{subject}_IC{ic_index}.png (all 3 plots in one figure)
#
# HOW TO USE:
#   1. Edit BASE_DATASET_PATH, SUBJECT_ID, IC_INDEX_TO_INSPECT below.
#      (defaults are set to the VP018 / IC6 case from the sensitivity report)
#   2. Run: python inspect_single_ic.py
#   3. Look at the saved .png and judge: does IC6 look like a genuine artifact
#      (sharp frontal blink-like topography, sparse large time-domain spikes,
#      or a broadband high-frequency PSD hump) or does it look brain-like
#      (smoother topography, oscillatory time series, 1/f + alpha-bump PSD)?
# ============================================================================

import os
import numpy as np
import scipy.io as sio
import mne
import mne_icalabel
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# CONFIGURATION -- EDIT THESE
# ----------------------------------------------------------------------------
BASE_DATASET_PATH = r"E:\kntu\Thesis\codes\DATA"
SUBJECT_ID = "VP018"
IC_INDEX_TO_INSPECT = 6   # the IC flagged by prob_threshold but not argmax

SFREQ_EEG = 200.0

ARTIFACT_CLASSES_IDX = {
    'brain': 0, 'muscle artifact': 1, 'eye blink': 2,
    'heart beat': 3, 'line noise': 4, 'channel noise': 5, 'other': 6
}
ALL_CLASS_NAMES = ['brain', 'muscle artifact', 'eye blink', 'heart beat',
                    'line noise', 'channel noise', 'other']


# ----------------------------------------------------------------------------
# Loading + preprocessing up to ICA fit (mirrors preprocess_eeg(), independent copy)
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


def fit_ica(subject_id):
    raw = load_eeg_raw(subject_id)

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
    return raw_filtered, ica, ic_labels


# ----------------------------------------------------------------------------
# Visual inspection
# ----------------------------------------------------------------------------
def inspect_ic(raw_filtered, ica, ic_labels, ic_index, subject_id, out_path):
    sources = ica.get_sources(raw_filtered).get_data()   # (n_components, n_times)
    ic_timeseries = sources[ic_index]
    sfreq = raw_filtered.info['sfreq']
    proba_matrix = ica.labels_scores_
    probs = proba_matrix[ic_index]
    argmax_label = ic_labels['labels'][ic_index]

    print(f"\nIC{ic_index} ({subject_id}):")
    print(f"  argmax label: '{argmax_label}'")
    for cname, p in zip(ALL_CLASS_NAMES, probs):
        marker = "  <-- argmax" if cname == argmax_label else ""
        print(f"    {cname:18s}: {p:.3f}{marker}")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    # 1. topomap
    ax_topo = fig.add_subplot(gs[0, 0])
    ica.plot_components(picks=[ic_index], axes=ax_topo, show=False, colorbar=False)
    ax_topo.set_title(f"IC{ic_index} topography")

    # 2. time series (first 60s, for readability -- full recording is long)
    ax_time = fig.add_subplot(gs[0, 1])
    n_show = min(int(60 * sfreq), len(ic_timeseries))
    t = np.arange(n_show) / sfreq
    ax_time.plot(t, ic_timeseries[:n_show], linewidth=0.5)
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude (a.u.)")
    ax_time.set_title(f"IC{ic_index} time series (first 60s)\n"
                       f"look for: sparse large spikes = eye blink, continuous high-freq buzz = muscle")

    # 3. full-recording time series (downsampled for plotting speed)
    ax_time_full = fig.add_subplot(gs[1, 0])
    step = max(1, len(ic_timeseries) // 20000)
    t_full = np.arange(0, len(ic_timeseries), step) / sfreq
    ax_time_full.plot(t_full, ic_timeseries[::step], linewidth=0.3)
    ax_time_full.set_xlabel("Time (s)")
    ax_time_full.set_ylabel("Amplitude (a.u.)")
    ax_time_full.set_title(f"IC{ic_index} time series (full recording, downsampled for display)")

    # 4. PSD
    ax_psd = fig.add_subplot(gs[1, 1])
    from scipy.signal import welch
    freqs, psd = welch(ic_timeseries, fs=sfreq, nperseg=min(4096, len(ic_timeseries)))
    ax_psd.semilogy(freqs, psd)
    ax_psd.set_xlim(0, 60)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (log scale)")
    ax_psd.set_title(f"IC{ic_index} power spectrum\n"
                      f"look for: broadband hump >20-30Hz = muscle, 1/f + alpha bump ~8-12Hz = brain-like")

    fig.suptitle(f"{subject_id} -- IC{ic_index}  (argmax='{argmax_label}', "
                 f"probs: brain={probs[0]:.2f} muscle={probs[1]:.2f} eye={probs[2]:.2f} "
                 f"other={probs[6]:.2f})", fontsize=11)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved inspection plot to: {out_path}")


if __name__ == "__main__":
    print(f"Fitting ICA for {SUBJECT_ID} (this is the slow step)...")
    raw_filtered, ica, ic_labels = fit_ica(SUBJECT_ID)

    n_components = ica.labels_scores_.shape[0]
    if IC_INDEX_TO_INSPECT >= n_components:
        print(f"ERROR: IC_INDEX_TO_INSPECT={IC_INDEX_TO_INSPECT} but this subject only has "
              f"{n_components} components (0..{n_components-1}). Check the component count "
              f"printed during ICA fit matches what you expect for {SUBJECT_ID}.")
        raise SystemExit(1)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"inspect_ic_{SUBJECT_ID}_IC{IC_INDEX_TO_INSPECT}.png")
    inspect_ic(raw_filtered, ica, ic_labels, IC_INDEX_TO_INSPECT, SUBJECT_ID, out_path)

    print("\n" + "="*70)
    print("HOW TO JUDGE:")
    print("- Topomap: a tight frontal (near-eyes) pattern suggests eye movement/blink;")
    print("  a temporal/peripheral pattern suggests muscle; a diffuse or dipolar")
    print("  central-parietal pattern suggests genuine brain activity.")
    print("- Time series: isolated large sparse deflections = eye blink; continuous")
    print("  jittery high-amplitude noise = muscle; smooth ongoing oscillation = brain.")
    print("- PSD: a hump rising above baseline at high frequencies (>20-30Hz) is the")
    print("  classic muscle-artifact signature; a clean 1/f decay with an alpha bump")
    print("  around 8-12Hz is the classic brain-component signature.")
    print("- If it looks like a genuine artifact -> the prob_threshold exclusion for")
    print("  this IC was correct, and this documents a concrete case where the extra")
    print("  layer earns its place in the pipeline.")
    print("- If it looks brain-like -> this may be a false positive from prob_threshold,")
    print("  worth reconsidering (e.g. is 0.30 too aggressive for 'other'-labeled ICs")
    print("  specifically?).")
    print("="*70)
