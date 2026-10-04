"""
Numeric verification for session_shift_diagnostic.py (priority-G3).

The whole value of this script is that its numbers are TRUSTWORTHY enough to
put in a thesis, so the tests check the metrics against known ground truth
rather than merely checking that they run:

  1. MMD of a distribution against ITSELF sits at ~0 (unbiased estimator),
     i.e. there is a real noise floor and "no shift" is distinguishable from
     "small shift". A biased estimator would fail this.
  2. MMD is symmetric.
  3. MMD increases MONOTONICALLY with injected shift magnitude.
  4. MMD detects a pure VARIANCE change (same mean) -- a mean-only metric
     would miss this, and session drift is not only a mean offset.
  5. mean_wasserstein matches the closed form for a known 1-D shift:
     W1(N(0,1), N(mu,1)) = |mu| for well-separated mu.
  6. mean_wasserstein is ~0 for identical distributions and grows with shift.
  7. Determinism: same inputs + same seed -> bit-identical results.
  8. End-to-end on synthetic .npy files laid out exactly like the real
     pipeline's output, including the case that matters: a directory WITH
     injected per-session drift must report a larger shift than the same
     data after per-session normalization (the E3 before/after claim).
"""
import os
import shutil
import tempfile

import numpy as np

from session_shift_diagnostic import (
    mmd_rbf, mean_wasserstein, load_subject, analyze_subject, summarize, run,
)

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Check 1 & 2: noise floor and symmetry
# ---------------------------------------------------------------------------
A = rng.standard_normal((150, 10))
B = rng.standard_normal((150, 10))   # same distribution, different sample

mmd_same = mmd_rbf(A, B, rng=np.random.default_rng(1))
mmd_self = mmd_rbf(A, A[::-1].copy(), rng=np.random.default_rng(1))

assert abs(mmd_same) < 0.02, f"MMD between same-distribution samples too large: {mmd_same:.5f}"
print(f"[PASS] check 1: MMD between two samples of the SAME distribution "
      f"≈ 0 ({mmd_same:+.5f}) -> unbiased estimator, real noise floor.")

mmd_ab = mmd_rbf(A, B, sigma_sq=2.0)
mmd_ba = mmd_rbf(B, A, sigma_sq=2.0)
assert abs(mmd_ab - mmd_ba) < 1e-10, "MMD is not symmetric"
print(f"[PASS] check 2: MMD symmetric ({mmd_ab:.6f} vs {mmd_ba:.6f}).")

# ---------------------------------------------------------------------------
# Check 3: monotonic in shift magnitude
# ---------------------------------------------------------------------------
shifts = [0.0, 0.25, 0.5, 1.0, 2.0]
vals = []
for s in shifts:
    Bs = rng.standard_normal((150, 10)) + s
    vals.append(mmd_rbf(A, Bs, sigma_sq=10.0))
assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), \
    f"MMD not monotonic in shift: {vals}"
print(f"[PASS] check 3: MMD increases monotonically with injected mean shift "
      f"({', '.join(f'{v:.4f}' for v in vals)} for shifts {shifts}).")

# ---------------------------------------------------------------------------
# Check 4: detects a pure VARIANCE change (identical means)
# ---------------------------------------------------------------------------
C = rng.standard_normal((150, 10)) * 3.0   # same mean 0, 3x std
mmd_var = mmd_rbf(A, C, sigma_sq=10.0)
assert abs(C.mean() - A.mean()) < 0.15, "test setup: means should be ~equal"
assert mmd_var > 10 * abs(mmd_same), \
    f"MMD failed to detect a variance-only difference ({mmd_var:.5f})"
print(f"[PASS] check 4: MMD detects a VARIANCE-only shift with equal means "
      f"({mmd_var:.5f} vs noise floor {abs(mmd_same):.5f}) -> not a mean-only metric.")

# ---------------------------------------------------------------------------
# Check 5 & 6: Wasserstein against the closed form
# ---------------------------------------------------------------------------
MU = 5.0
X1 = rng.standard_normal((4000, 6))
X2 = rng.standard_normal((4000, 6)) + MU
w = mean_wasserstein(X1, X2)
assert abs(w - MU) < 0.15, f"W1 for a shift of {MU} should be ~{MU}, got {w:.4f}"
print(f"[PASS] check 5: mean 1-D Wasserstein matches the closed form for a "
      f"pure shift (expected {MU:.1f}, got {w:.4f}).")

w0 = mean_wasserstein(X1, rng.standard_normal((4000, 6)))
assert w0 < 0.1, f"W1 for identical distributions should be ~0, got {w0:.4f}"
assert w0 < w
print(f"[PASS] check 6: W1 ≈ 0 for identical distributions ({w0:.4f}) and "
      f"grows with shift.")

# ---------------------------------------------------------------------------
# Check 7: determinism
# ---------------------------------------------------------------------------
r1 = mmd_rbf(A, B, rng=np.random.default_rng(7))
r2 = mmd_rbf(A, B, rng=np.random.default_rng(7))
assert r1 == r2, "MMD not deterministic under a fixed seed"
print(f"[PASS] check 7: identical seed -> bit-identical MMD ({r1:.8f}).")

# ---------------------------------------------------------------------------
# Check 8: end-to-end on synthetic files matching the real pipeline layout,
# WITH vs WITHOUT per-session drift. This is the exact claim the thesis needs.
# ---------------------------------------------------------------------------
tmp_drift = tempfile.mkdtemp()
tmp_norm = tempfile.mkdtemp()
try:
    # Small stand-ins for the real shapes: (N,16,16,T) and (N,S,16,16,T,2).
    N_PER, G, T_E = 25, 16, 60
    S_SEG, T_F = 6, 10
    sessions = np.repeat([1, 2, 3], N_PER).astype(np.int64)
    N = len(sessions)

    eeg = rng.standard_normal((N, G, G, T_E)).astype(np.float32)
    fnirs = rng.standard_normal((N, S_SEG, G, G, T_F, 2)).astype(np.float32)
    labels = rng.integers(0, 3, N).astype(np.int64)

    # Inject a per-session offset+gain: this is the "before E3" state.
    drift = {1: (0.0, 1.0), 2: (4.0, 2.5), 3: (-3.0, 0.4)}
    eeg_d, fn_d = eeg.copy(), fnirs.copy()
    for s, (off, gain) in drift.items():
        m = sessions == s
        eeg_d[m] = eeg_d[m] * gain + off
        fn_d[m] = fn_d[m] * gain + off

    # Per-session normalization: the "after E3" state.
    eeg_n, fn_n = eeg_d.copy(), fn_d.copy()
    for s in drift:
        m = sessions == s
        e = eeg_n[m]
        eeg_n[m] = (e - e.mean(axis=(0, 3), keepdims=True)) / (e.std(axis=(0, 3), keepdims=True) + 1e-8)
        f = fn_n[m]
        fn_n[m] = (f - f.mean(axis=(0, 1, 4), keepdims=True)) / (f.std(axis=(0, 1, 4), keepdims=True) + 1e-8)

    for d, (e_arr, f_arr) in [(tmp_drift, (eeg_d, fn_d)), (tmp_norm, (eeg_n, fn_n))]:
        np.save(os.path.join(d, "VP001_eeg_windows.npy"), e_arr)
        np.save(os.path.join(d, "VP001_fnirs_windows.npy"), f_arr)
        np.save(os.path.join(d, "VP001_session_ids.npy"), sessions)
        np.save(os.path.join(d, "VP001_labels.npy"), labels)

    kw = dict(eeg_time_stride=2, fnirs_seg_stride=1, fnirs_time_stride=1, n_components=15)
    fd = load_subject(tmp_drift, "VP001", **kw)
    fn_ = load_subject(tmp_norm, "VP001", **kw)
    assert fd is not None and fn_ is not None, "load_subject failed to read synthetic files"

    rd = analyze_subject(fd, np.random.default_rng(3))
    rn = analyze_subject(fn_, np.random.default_rng(3))
    sd = summarize({"VP001": rd})
    sn = summarize({"VP001": rn})

    for mod in ("eeg", "fnirs"):
        assert sn[mod]["mmd_mean"] < sd[mod]["mmd_mean"], \
            f"{mod}: per-session normalization did NOT reduce measured MMD"
        assert sn[mod]["emd_mean"] < sd[mod]["emd_mean"], \
            f"{mod}: per-session normalization did NOT reduce measured EMD"
        print(f"[PASS] check 8 ({mod}): drift MMD={sd[mod]['mmd_mean']:.5f} -> "
              f"normalized MMD={sn[mod]['mmd_mean']:.5f}; EMD "
              f"{sd[mod]['emd_mean']:.4f} -> {sn[mod]['emd_mean']:.4f} (both reduced).")

    # Missing subject must be skipped cleanly, not crash.
    assert load_subject(tmp_drift, "VP999") is None
    res, summ = run(tmp_drift, ["VP001", "VP999"])
    assert "VP001" in res and "VP999" not in res
    print("[PASS] check 8b: run() skips absent subjects cleanly and summarizes the rest.")

    # ---------------------------------------------------------------------
    # Check 9: per-subject SUBDIRECTORY layout -- the layout the real
    # pipeline actually writes (<data_dir>/VP001/VP001_*.npy). It must give
    # BIT-IDENTICAL results to the flat layout holding the same arrays;
    # otherwise the on-disk layout would silently change the reported
    # numbers.
    # ---------------------------------------------------------------------
    tmp_nested = tempfile.mkdtemp()
    try:
        sub = os.path.join(tmp_nested, "VP001")
        os.makedirs(sub)
        for name in ("eeg_windows", "fnirs_windows", "session_ids", "labels"):
            shutil.copy(os.path.join(tmp_drift, f"VP001_{name}.npy"),
                        os.path.join(sub, f"VP001_{name}.npy"))

        fnest = load_subject(tmp_nested, "VP001", **kw)
        assert fnest is not None, "nested per-subject layout was not detected"
        assert np.array_equal(fnest["eeg"], fd["eeg"]), \
            "nested layout produced different EEG features than the flat layout"
        assert np.array_equal(fnest["sessions"], fd["sessions"])

        rnest = analyze_subject(fnest, np.random.default_rng(3))
        snest = summarize({"VP001": rnest})
        for mod in ("eeg", "fnirs"):
            assert snest[mod]["mmd_mean"] == sd[mod]["mmd_mean"], \
                f"{mod}: nested vs flat layout gave different MMD"
        print("[PASS] check 9: per-subject subdirectory layout (<dir>/VP001/VP001_*.npy) "
              "is detected and yields BIT-IDENTICAL results to the flat layout.")

        # A directory containing an incomplete subject folder is still skipped.
        os.makedirs(os.path.join(tmp_nested, "VP002"))
        assert load_subject(tmp_nested, "VP002") is None
        print("[PASS] check 9b: an empty/incomplete subject folder is skipped, not crashed on.")
    finally:
        shutil.rmtree(tmp_nested, ignore_errors=True)

finally:
    shutil.rmtree(tmp_drift, ignore_errors=True)
    shutil.rmtree(tmp_norm, ignore_errors=True)

print("\nALL CHECKS PASSED.")
