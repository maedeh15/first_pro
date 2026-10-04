"""
Numeric verification for eeg_connectivity.py (priority-E1 PLV extraction).

Checks:
  1. Two channels with IDENTICAL phase (same-frequency sine, zero lag) in the
     target band -> PLV ~= 1.0 (perfectly phase-locked).
  2. Two channels with a FIXED, non-zero phase offset in-band -> PLV still
     ~= 1.0 (PLV is phase-DIFFERENCE invariant, not amplitude/phase-value
     invariant -- this is the defining property of PLV vs. plain correlation).
  3. Two independent white-noise channels -> PLV close to a small value
     (not exactly 0 for finite T, but well below the phase-locked case).
  4. Band selectivity: a pure 6 Hz (theta) sine pair phase-locked in theta
     gives high theta-band PLV; the SAME pair evaluated in the alpha band
     (10-13Hz, no signal there -- only filtered noise/residual) gives much
     lower PLV, confirming the band-pass actually isolates frequency
     content rather than just returning ~1 regardless of band.
  5. Matrix-level checks: symmetry, unit diagonal, shape.
  6. upper_triangle_vector length matches C*(C-1)/2 and pulls the right
     values (spot-check against the full matrix).
  7. batch_connectivity_features output shape matches (N, n_bands, n_pairs).
"""
import numpy as np
from eeg_connectivity import (
    plv_matrix, upper_triangle_vector, window_connectivity_features,
    batch_connectivity_features, THETA_BAND, ALPHA_BAND,
)

rng = np.random.default_rng(0)
sfreq = 200.0
T = 2000  # 10s @ 200Hz, matches this project's WINDOW_LEN_EEG=10s

t = np.arange(T) / sfreq

# ---------------------------------------------------------------------------
# Check 1 & 2: phase-locked sines (zero and nonzero phase offset) -> PLV~1
# ---------------------------------------------------------------------------
freq = 6.0  # Hz, inside theta band
noise_amp = 0.05

ch_a = np.sin(2 * np.pi * freq * t) + noise_amp * rng.standard_normal(T)
ch_b_zero_lag = np.sin(2 * np.pi * freq * t) + noise_amp * rng.standard_normal(T)
phase_offset = 1.3  # radians, arbitrary fixed nonzero offset
ch_b_offset = np.sin(2 * np.pi * freq * t + phase_offset) + noise_amp * rng.standard_normal(T)

window_zero = np.stack([ch_a, ch_b_zero_lag])
window_offset = np.stack([ch_a, ch_b_offset])

plv_zero = plv_matrix(window_zero, sfreq, THETA_BAND)
plv_offset = plv_matrix(window_offset, sfreq, THETA_BAND)

assert plv_zero[0, 1] > 0.95, f"zero-lag phase-locked PLV too low: {plv_zero[0,1]:.4f}"
assert plv_offset[0, 1] > 0.95, f"fixed-offset phase-locked PLV too low: {plv_offset[0,1]:.4f}"
print(f"[PASS] check 1: zero-lag phase-locked sines -> PLV={plv_zero[0,1]:.4f} (>0.95)")
print(f"[PASS] check 2: fixed nonzero-offset phase-locked sines -> PLV={plv_offset[0,1]:.4f} "
      f"(>0.95, confirms PLV is offset-invariant, not just correlation)")

# ---------------------------------------------------------------------------
# Check 3: independent white noise -> low PLV
# ---------------------------------------------------------------------------
noise_a = rng.standard_normal(T)
noise_b = rng.standard_normal(T)
window_noise = np.stack([noise_a, noise_b])
plv_noise = plv_matrix(window_noise, sfreq, THETA_BAND)
assert plv_noise[0, 1] < 0.5, f"independent noise PLV unexpectedly high: {plv_noise[0,1]:.4f}"
print(f"[PASS] check 3: independent white noise -> PLV={plv_noise[0,1]:.4f} "
      f"(well below phase-locked case's >0.95)")

# ---------------------------------------------------------------------------
# Check 4: band selectivity -- same theta-locked pair, evaluated in alpha
# band, should show much lower PLV (no genuine alpha-band phase relationship,
# since the only signal present is at 6 Hz).
# ---------------------------------------------------------------------------
plv_offset_alpha = plv_matrix(window_offset, sfreq, ALPHA_BAND)
print(f"[INFO] check 4: same signal pair -- theta-band PLV={plv_offset[0,1]:.4f}, "
      f"alpha-band PLV={plv_offset_alpha[0,1]:.4f}")
assert plv_offset_alpha[0, 1] < plv_offset[0, 1] - 0.3, \
    "alpha-band PLV on a purely theta-locked signal should be much lower than theta-band PLV"
print("[PASS] check 4: band-pass correctly isolates frequency content "
      "(alpha-band PLV collapses when no true alpha-band phase relationship exists).")

# ---------------------------------------------------------------------------
# Check 5: matrix properties (symmetry, unit diagonal, shape) on a
# multi-channel window.
# ---------------------------------------------------------------------------
C = 6
multi_window = np.stack([
    np.sin(2 * np.pi * freq * t + 0.4 * i) + noise_amp * rng.standard_normal(T)
    for i in range(C)
])
mat = plv_matrix(multi_window, sfreq, THETA_BAND)
assert mat.shape == (C, C)
assert np.allclose(mat, mat.T), "PLV matrix should be exactly symmetric"
assert np.allclose(np.diag(mat), 1.0), "PLV matrix diagonal should be exactly 1.0"
print(f"[PASS] check 5: PLV matrix shape={mat.shape}, symmetric, unit diagonal.")

# ---------------------------------------------------------------------------
# Check 6: upper_triangle_vector length and content spot-check.
# ---------------------------------------------------------------------------
vec = upper_triangle_vector(mat)
expected_len = C * (C - 1) // 2
assert vec.shape == (expected_len,), f"expected length {expected_len}, got {vec.shape}"
# spot-check: first element of triu_indices(k=1) is (0,1)
assert np.isclose(vec[0], mat[0, 1])
print(f"[PASS] check 6: upper_triangle_vector length={vec.shape[0]} "
      f"(expected C*(C-1)/2={expected_len}), values match source matrix.")

# ---------------------------------------------------------------------------
# Check 7: full per-window and batch feature extraction shapes.
# ---------------------------------------------------------------------------
feats = window_connectivity_features(multi_window, sfreq)
assert feats.shape == (2, expected_len), f"got {feats.shape}"
print(f"[PASS] check 7a: window_connectivity_features shape={feats.shape} "
      f"(2 bands x {expected_len} pairs).")

N = 5
windows_batch = np.stack([multi_window + 0.01 * rng.standard_normal(multi_window.shape)
                           for _ in range(N)])
batch_feats = batch_connectivity_features(windows_batch, sfreq)
assert batch_feats.shape == (N, 2, expected_len), f"got {batch_feats.shape}"
assert batch_feats.dtype == np.float32
print(f"[PASS] check 7b: batch_connectivity_features shape={batch_feats.shape}, dtype=float32.")

print("\nALL CHECKS PASSED.")
