"""
verify_mgfnet_preproc.py -- checks the OUTPUT of Preproc_mgfnet_nback.py run
on the synthetic fixture (make_mgfnet_fixture.py) against the fixture's known
ground truth. Each check targets a specific way the pipeline could be
silently wrong.

USAGE
    python make_mgfnet_fixture.py  <fixture_data_dir>
    python Preproc_mgfnet_nback.py --subjects 1 --data-root <fixture_data_dir> --out <out_dir>
    python verify_mgfnet_preproc.py <out_dir> <fixture_data_dir>
"""
import json
import os
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "fixture_out")
DATA = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "fixture_data")
SUB = "VP001"
D = os.path.join(OUT, SUB)

E = np.load(os.path.join(D, f"{SUB}_mgf_eeg.npy"))
O = np.load(os.path.join(D, f"{SUB}_mgf_hbo.npy"))
R = np.load(os.path.join(D, f"{SUB}_mgf_hbr.npy"))
lab = np.load(os.path.join(D, f"{SUB}_labels.npy"))
sess = np.load(os.path.join(D, f"{SUB}_session_ids.npy"))
trl = np.load(os.path.join(D, f"{SUB}_trial_ids.npy"))
with open(os.path.join(D, f"{SUB}_mgf_metadata.json"), encoding="utf-8") as f:
    meta = json.load(f)

OFFS = np.arange(2.0, 42.0 - 5.0 + 1e-6, 1.0)          # 36 window offsets
results = []


def rec(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --- 1. tensor shapes match MGFNet Sec. 2.2 (30 x 1000 / 36 x 50) ---------
n = E.shape[0]
rec("shapes", E.shape == (n, 30, 1000) and O.shape == (n, 36, 50) and R.shape == (n, 36, 50),
    f"EEG{E.shape} HbO{O.shape} HbR{R.shape}")
rec("channel list", len(meta["eeg_channels"]) == 30
    and meta["eeg_channels"][-2:] == ["HEOG", "VEOG"],
    f"28 scalp + {meta['eeg_channels'][-2:]}")

# --- 2. window count is exactly what the sliding rule predicts -------------
rec("window count", n == 27 * len(OFFS), f"N={n} = 27 trials x {len(OFFS)} offsets")

# --- 3. labels / sessions match the fixture's marker file ------------------
mrk = sio.loadmat(os.path.join(DATA, f"{SUB}-EEG", "mrk_nback.mat"),
                  struct_as_record=False, squeeze_me=True)["mrk"]
codes = np.asarray(mrk.event.desc, dtype=int).flatten()
trial_codes = codes[np.isin(codes, [112, 128, 144])]
gt = np.array([{112: 0, 128: 1, 144: 2}[c] for c in trial_codes])
rec("label mapping", all((lab[trl == t] == gt[t]).all() for t in np.unique(trl)),
    "every window carries its trial's true class")
rec("session mapping", all((sess[trl == t] == t // 9 + 1).all() for t in np.unique(trl)),
    "session_id == trial_idx // 9 + 1")

# --- 4. LOSO is well-formed: no trial spans two sessions ------------------
spans = [int(t) for t in np.unique(trl) if len(np.unique(sess[trl == t])) > 1]
rec("LOSO integrity", not spans, f"offenders: {spans}")

# --- 5. class balance ------------------------------------------------------
cnt = np.bincount(lab, minlength=3)
rec("class balance", len(set(cnt.tolist())) == 1, f"per-class counts {cnt.tolist()}")

# --- 6. the EEG band-pass removed the 0.15 Hz drift ------------------------
f, P = welch(E.reshape(-1, 1000), fs=200.0, nperseg=512, axis=-1)
Pm = P.mean(axis=0)
p_drift = Pm[(f > 0.0) & (f < 0.6)].max()
p_alpha = Pm[(f > 8) & (f < 12)].max()
rec("EEG band-pass", p_drift < 0.2 * p_alpha,
    f"0-0.6Hz peak={p_drift:.3e} vs 8-12Hz peak={p_alpha:.3e}")

# --- 7. the fNIRS low-pass removed the 1.1 Hz cardiac component -----------
f2, P2 = welch(O.reshape(-1, 50), fs=10.0, nperseg=50, axis=-1)
P2m = P2.mean(axis=0)
p_card = P2m[(f2 > 0.9) & (f2 < 1.4)].max()
p_hemo = P2m[(f2 >= 0.0) & (f2 < 0.15)].max()
rec("fNIRS low-pass", p_card < 0.05 * p_hemo,
    f"1.1Hz cardiac peak={p_card:.3e} vs <0.15Hz hemodynamic={p_hemo:.3e}")

# --- 8. fNIRS windows were timed from the fNIRS clock --------------------
rec("fNIRS clock sync (reported)", meta.get("fnirs_clock_sync") is True,
    f"metadata fnirs_clock_sync={meta.get('fnirs_clock_sync')}")

# --- 9. windows start at TASK onset (data-level timing check) --------------
# Tracing mean |HbO| across the 36 window offsets of a trial reconstructs the
# fixture's HRF, which starts exactly at task onset (fNIRS clock) and peaks
# ~6 s later. Labelling windows with offsets 2..37 s therefore puts the peak
# at ~6 s ONLY if both the clock and the task onset are right:
#   correct                       -> ~6 s
#   TASK_TMIN wrong by 2 s        -> ~8 s
#   EEG-clock fallback (3.7 s)    -> ~10 s
# (An EEG-based check is not usable on this fixture: the ICA EOG correction
# also removes its rank-1 task sinusoid.)
peaks = []
for t in np.unique(trl):
    w = O[trl == t]
    if len(w) == len(OFFS):
        peaks.append(OFFS[np.argmax(np.abs(w).mean(axis=(1, 2)))])
peak = float(np.median(peaks))
rec("window timing (HRF lag)", 5.0 <= peak <= 7.0,
    f"median HRF peak at window offset {peak:.1f}s "
    f"(correct ~6s; onset off by 2 s -> ~8s; EEG-clock fallback -> ~10s)")
rec("task span", meta["task_span_s"] == [2.0, 42.0], f"span={meta['task_span_s']}")
rec("marker-onset check", meta["marker_onset_check"]["status"] == "ok",
    f"{meta['marker_onset_check']}")

# --- 10. no NaN/Inf anywhere ---------------------------------------------
rec("finite values", all(np.isfinite(a).all() for a in (E, O, R)), "no NaN/Inf")

# --- 11. no normalisation in preprocessing (z-score happens per LOSO fold) --
# Fixture EEG is ~12 uV of noise (+ components) stored in uV -> must be in
# VOLTS here (~1e-5), not z-scored (~1).
rec("units preserved (no transductive z-score)", 1e-7 < E.std() < 1e-3,
    f"EEG std={E.std():.3e} V (z-scored data would be ~1)")

# --- 12. HbO and HbR are NOT the same array ------------------------------
rec("HbO != HbR", not np.allclose(O, R),
    f"corr={np.corrcoef(O.ravel(), R.ravel())[0, 1]:.3f}")

print()
if all(results):
    print(">>> All preprocessing checks PASSED.")
else:
    print(">>> FAILURES present. Do not preprocess the real dataset yet.")
    sys.exit(1)
