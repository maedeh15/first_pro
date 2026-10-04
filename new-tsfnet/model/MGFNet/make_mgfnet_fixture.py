"""
make_mgfnet_fixture.py -- builds a synthetic VP001 that reproduces the REAL
.mat contract of Shin et al. (2018), so Preproc_mgfnet_nback.py can be run
end-to-end without the actual dataset.

Rule 8: the fixture reproduces the contract as VERIFIED in
Preproc_tsfnet_nback_badch.py's comments -- not the format the paper's prose
implies. Specifically:
  * cnt.x is (n_samples, n_channels), NOT (n_channels, n_samples)
  * cnt.clab has 30 entries: 28 scalp + HEOG + VEOG. TP9/TP10 absent.
  * cnt.fs is already 200 (EEG) / 10 (fNIRS); values in MICROVOLTS
  * mrk.time is in MILLISECONDS; mrk.event.desc holds the codes
  * EEG codes are 112/128/144; fNIRS codes are 7/8/9 -- DIFFERENT maps
  * exactly 27 trials, three groups of 9 separated by a long gap
  * the fNIRS clock is deliberately OFFSET from the EEG clock, so the
    marker-sync path is actually exercised rather than accidentally passing
  * oxy/deoxy are already Beer-Lambert concentrations, 36 channels
  * n-back stimulus markers every 2 s from task onset (trial marker + 2 s),
    so Preproc's marker_onset_diagnostic is exercised. Their code
    (FIXTURE_STIM_CODE) is a fixture placeholder, not a verified dataset code.
"""
import os
import sys

import numpy as np
import scipy.io as sio

rng = np.random.default_rng(0)

SCALP = ["Fp1", "Fp2", "AFF5h", "AFF6h", "AFz", "F1", "F2", "FC1", "FC2",
         "FC5", "FC6", "Cz", "C3", "C4", "T7", "T8", "CP1", "CP2", "CP5",
         "CP6", "Pz", "P3", "P4", "P7", "P8", "POz", "O1", "O2"]   # 28
assert len(SCALP) == 28
CLAB = SCALP + ["HEOG", "VEOG"]                                     # 30

FS_EEG, FS_NIR = 200.0, 10.0
TRIAL_SPACING = 62.0          # 2 s instruction + 40 s task + 20 s rest
SESSION_GAP = 148.0           # verified inter-session gap
FNIRS_CLOCK_OFFSET = 3.7      # seconds of drift between the two devices
FIXTURE_STIM_CODE = 16        # placeholder stimulus code (fixture only)
N_STIM_PER_TRIAL = 20         # one digit every 2 s over the 40 s task


def trial_onsets():
    t, out = 30.0, []
    for s in range(3):
        for k in range(9):
            out.append(t)
            t += TRIAL_SPACING
        t += SESSION_GAP
    return np.asarray(out)                       # 27 onsets


def build(out_root, subject_id="VP001"):
    onsets = trial_onsets()
    total_s = onsets[-1] + 200.0

    # labels: 3 repeats of each class per session, shuffled within session
    labels = []
    for s in range(3):
        blk = np.repeat([0, 1, 2], 3)
        rng.shuffle(blk)
        labels.extend(blk.tolist())
    labels = np.asarray(labels)

    # ---- EEG: 30 channels in MICROVOLTS, (n_samples, 30) -------------------
    n_e = int(total_s * FS_EEG)
    t_e = np.arange(n_e) / FS_EEG
    eeg = rng.normal(0, 12.0, size=(n_e, 30))
    eeg += 8.0 * np.sin(2 * np.pi * 10.0 * t_e)[:, None]        # alpha
    eeg += 40.0 * np.sin(2 * np.pi * 0.15 * t_e)[:, None]       # slow drift (killed by 1 Hz HP)
    # class-dependent task modulation on the 28 SCALP channels only (not on
    # HEOG/VEOG: a task signal shared with the EOG channels is removed by the
    # ICA EOG correction, which would make the data-level task-period check
    # in verify_mgfnet_preproc.py meaningless).
    for on, lab in zip(onsets, labels):
        a, b = int((on + 2) * FS_EEG), int((on + 42) * FS_EEG)
        eeg[a:b, :28] += (3.0 * (lab + 1)) * np.sin(
            2 * np.pi * (6.0 + 2.0 * lab) * t_e[a:b])[:, None]
    # ocular bursts, strongest on the frontal + EOG channels
    ocular = np.zeros(n_e)
    for c in rng.choice(n_e, size=400, replace=False):
        ocular[c:c + 100] += rng.normal(60, 10)
    eeg[:, [0, 1, 28, 29]] += ocular[:, None] * np.array([0.6, 0.6, 1.0, 1.0])

    eeg_mrk_codes = np.array([{0: 112, 1: 128, 2: 144}[l] for l in labels])

    eeg_dir = os.path.join(out_root, f"{subject_id}-EEG")
    os.makedirs(eeg_dir, exist_ok=True)
    sio.savemat(os.path.join(eeg_dir, "cnt_nback.mat"),
                {"cnt": {"x": eeg.astype(np.float32), "fs": FS_EEG,
                         "clab": np.array(CLAB, dtype=object),
                         "yUnit": "uV"}}, do_compression=True)
    stim_t = np.concatenate([on + 2.0 + 2.0 * np.arange(N_STIM_PER_TRIAL) for on in onsets])
    all_t = np.concatenate([onsets, stim_t])
    all_c = np.concatenate([eeg_mrk_codes, np.full(len(stim_t), FIXTURE_STIM_CODE)])
    order = np.argsort(all_t, kind="stable")
    sio.savemat(os.path.join(eeg_dir, "mrk_nback.mat"),
                {"mrk": {"time": all_t[order] * 1000.0,
                         "event": {"desc": all_c[order]}}})
    # mnt is loaded by the TSFNet pipeline but NOT by the MGFNet one (no
    # spatial mapping); written anyway so the folder matches the real layout.
    sio.savemat(os.path.join(eeg_dir, "mnt_nback.mat"),
                {"mnt": {"clab": np.array(CLAB, dtype=object)}})

    # ---- fNIRS: 36 channels, already HbO/HbR concentrations ----------------
    n_f = int(total_s * FS_NIR)
    t_f = np.arange(n_f) / FS_NIR
    hbo = rng.normal(0, 0.02, size=(n_f, 36))
    hbo += 0.05 * np.sin(2 * np.pi * 0.1 * t_f)[:, None]        # Mayer waves
    hbo += 0.03 * np.sin(2 * np.pi * 1.1 * t_f)[:, None]        # cardiac -> low-pass must kill this
    for on, lab in zip(onsets, labels):
        # haemodynamic response, peaking ~6 s after TASK onset, on the
        # fNIRS device's own (offset) clock
        a = int((on + FNIRS_CLOCK_OFFSET + 2) * FS_NIR)
        b = int((on + FNIRS_CLOCK_OFFSET + 42) * FS_NIR)
        tt = np.arange(b - a) / FS_NIR
        hrf = (tt ** 2) * np.exp(-tt / 3.0)
        hbo[a:b, :] += 0.04 * (lab + 1) * (hrf / (hrf.max() + 1e-9))[:, None]
    hbr = -0.45 * hbo + rng.normal(0, 0.01, size=hbo.shape)

    nir_codes = np.array([{0: 7, 1: 8, 2: 9}[l] for l in labels])
    nir_dir = os.path.join(out_root, f"{subject_id}-NIRS")
    os.makedirs(nir_dir, exist_ok=True)
    sio.savemat(os.path.join(nir_dir, "cnt_nback.mat"),
                {"cnt": {"oxy": {"x": hbo.astype(np.float32), "fs": FS_NIR,
                                 "yUnit": "mmol/L"},
                         "deoxy": {"x": hbr.astype(np.float32), "fs": FS_NIR}}},
                do_compression=True)
    sio.savemat(os.path.join(nir_dir, "mrk_nback.mat"),
                {"mrk": {"time": (onsets + FNIRS_CLOCK_OFFSET) * 1000.0,
                         "event": {"desc": nir_codes}}})
    sio.savemat(os.path.join(nir_dir, "mnt_nback.mat"),
                {"mnt": {"clab": np.array([f"ch{i}" for i in range(36)],
                                          dtype=object)}})

    print(f"fixture written to {out_root}")
    print(f"  trials=27  labels/session={[labels[i*9:(i+1)*9].tolist() for i in range(3)]}")
    print(f"  EEG {eeg.shape} @{FS_EEG}Hz  fNIRS {hbo.shape} @{FS_NIR}Hz")
    print(f"  fNIRS clock offset vs EEG = {FNIRS_CLOCK_OFFSET}s (sync path exercised)")
    return labels, onsets


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixture_data"))
