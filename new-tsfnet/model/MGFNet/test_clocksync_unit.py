"""
test_clocksync_unit.py -- fast, targeted test of extract_windows()'s fNIRS
clock-sync logic and the fnirs_clock_sync reporting flag, WITHOUT paying for
ICA/filtering on a full-length recording (that already passed once; this
isolates just the logic touched by the last patch).
"""
import numpy as np
import Preproc_mgfnet_nback as P

rng = np.random.default_rng(0)

FS_EEG, FS_NIR = 200.0, 10.0
onsets = []
t = 30.0
for s in range(3):
    for k in range(9):
        onsets.append(t)
        t += 62.0
    t += 148.0
onsets = np.asarray(onsets)
total_s = onsets[-1] + 200.0

labels = []
for s in range(3):
    blk = np.repeat([0, 1, 2], 3)
    rng.shuffle(blk)
    labels.extend(blk.tolist())
labels = np.asarray(labels)

n_e = int(total_s * FS_EEG)
eeg = rng.normal(0, 1.0, size=(30, n_e)).astype(np.float32)   # (30, n_times) -- matches
                                                                # preprocess_eeg's return
n_f = int(total_s * FS_NIR)
t_f = np.arange(n_f) / FS_NIR
hbo = rng.normal(0, 0.02, size=(36, n_f))
hbr = rng.normal(0, 0.01, size=(36, n_f))
HRF_OFFSET = 3.7   # fNIRS device clock is 3.7s AHEAD of EEG's

for on, lab in zip(onsets, labels):
    a = int((on + HRF_OFFSET + 2) * FS_NIR)
    b = int((on + HRF_OFFSET + 42) * FS_NIR)
    tt = np.arange(b - a) / FS_NIR
    hrf = (tt ** 2) * np.exp(-tt / 3.0)
    hbo[:, a:b] += 0.04 * (lab + 1) * (hrf / (hrf.max() + 1e-9))[None, :]

eeg_codes = np.array([{0: 112, 1: 128, 2: 144}[l] for l in labels])
events = np.zeros((27, 3), dtype=int)
events[:, 0] = (onsets * FS_EEG).astype(int)
events[:, 2] = eeg_codes

fnirs_codes = np.array([{0: 7, 1: 8, 2: 9}[l] for l in labels])
fnirs_ev_time_correct = (onsets + HRF_OFFSET) * 1000.0   # ms, matches fixture convention


def run(ev_time, ev_code, label):
    P.extract_windows.last_sync_ok = None
    E, O, R, lab_out, sess_out, trl_out = P.extract_windows(
        eeg, FS_EEG, events, hbo, hbr, ev_time, ev_code, "VP_UNIT")
    sync_flag = P.extract_windows.last_sync_ok
    offs = np.arange(2.0, 42.0 - 5.0 + 1e-6, 1.0)
    peaks = []
    for tr in np.unique(trl_out):
        w = O[trl_out == tr]
        if len(w) != len(offs):
            continue
        peaks.append(offs[np.argmax(np.abs(w).mean(axis=(1, 2)))])
    peak = float(np.median(peaks)) if peaks else float("nan")
    print(f"[{label}] reported sync_ok={sync_flag}  N={E.shape[0]}  "
          f"HRF peak offset={peak:.2f}s")
    return sync_flag, peak


print("=== case A: correct fNIRS markers present (clocks differ by 3.7s, "
      "sync logic must detect & correct for it) ===")
sync_a, peak_a = run(fnirs_ev_time_correct, fnirs_codes, "A-correct-markers")

print("\n=== case B: fNIRS markers ABSENT -> forced EEG-clock fallback ===")
sync_b, peak_b = run(None, None, "B-forced-eeg-clock")

print()
ok1 = sync_a is True
ok2 = sync_b is False
ok3 = 3.0 <= peak_a <= 8.0
ok4 = peak_b > peak_a + 2.0          # must be measurably later when misaligned
print(f"[check] A reports sync_ok=True:  {'PASS' if ok1 else 'FAIL'}")
print(f"[check] B reports sync_ok=False: {'PASS' if ok2 else 'FAIL'}")
print(f"[check] A HRF peak in [3,8]s (correct sync): {'PASS' if ok3 else 'FAIL'}")
print(f"[check] B HRF peak shifted later than A by >2s: {'PASS' if ok4 else 'FAIL'}  "
      f"(A={peak_a:.2f}s B={peak_b:.2f}s)")

if all([ok1, ok2, ok3, ok4]):
    print("\n>>> Clock-sync fix VERIFIED: the metadata flag and the lag-recovery "
          "check both discriminate correctly.")
else:
    print("\n>>> FAIL -- clock-sync logic or its reporting is still broken.")
    import sys
    sys.exit(1)
