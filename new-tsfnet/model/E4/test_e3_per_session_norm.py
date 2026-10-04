"""
Numeric verification for priority-E3 (stage 1) per-session normalization
(apply_per_session_normalization in Preproc_tsfnet_nback_badch.py).

The load-bearing question this test answers: does the function actually
REMOVE session-to-session drift that the existing per-subject z-score leaves
behind? A shape/no-crash test would not answer that, so the core checks
INJECT a known per-session offset and gain and verify they are eliminated.

Checks:
  1. Injected per-session offsets (different mean per session) are removed:
     after normalization every session's per-grid-cell mean is ~0.
  2. Injected per-session gains (different std per session) are removed:
     after normalization every session's per-grid-cell std is ~1.
  3. Between-location contrast is PRESERVED in the sense that each grid cell
     is normalized independently (a cell that was constant-offset relative
     to another still gets its own stats) -- verified by confirming the
     normalization is per-cell, not a single global scalar.
  4. Sessions are normalized INDEPENDENTLY: changing session 3's data does
     not alter the normalized output of sessions 1 and 2 at all. This is the
     concrete, testable form of the "no cross-session leakage" claim in the
     docstring.
  5. fNIRS: HbO and HbR are normalized SEPARATELY (per chromophore), not
     pooled -- injected chromophore-specific offsets are each removed.
  6. Channel-space array is normalized per channel per session.
  7. Edge case: a session with a single window is skipped (not divided by
     ~0) and leaves NaN/inf out of the output.
"""
import numpy as np
import sys
import types

# apply_per_session_normalization uses log_message from the preprocessing
# module. Import the module's function without executing the full pipeline
# by loading the source and exec'ing just what's needed would be fragile;
# instead import the module and rely on it being import-safe (it guards its
# main with __name__ == "__main__" / CLI parsing at import is harmless here).
import importlib.util
import ast

# The preprocessing module imports mne/pyprep etc. at module level, which are
# heavy (and unnecessary for testing a pure-numpy function). So instead of
# importing the whole module, extract THIS EXACT FUNCTION's source from the
# file and exec it in a minimal namespace. This still tests the real shipped
# code (not a copy that could drift), while staying dependency-free.
_src = open("Preproc_tsfnet_nback_badch.py", encoding="utf-8").read()
_tree = ast.parse(_src)
_fn_node = next(n for n in _tree.body
                if isinstance(n, ast.FunctionDef)
                and n.name == "apply_per_session_normalization")
_fn_src = ast.get_source_segment(_src, _fn_node)

_ns = {"np": np, "log_message": lambda msg: print(f"    [log] {msg}")}
exec(_fn_src, _ns)
apply_per_session_normalization = _ns["apply_per_session_normalization"]

rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# Build synthetic windows with KNOWN, injected per-session drift.
# Shapes mirror the real pipeline but smaller: T reduced, S reduced.
# ---------------------------------------------------------------------------
N_PER_SESS = 8
SESSIONS = [1, 2, 3]
G = 4          # grid size (real pipeline: 16)
T_EEG = 50     # real: 2000
S_SEG = 3      # real: 21
T_FN = 6       # real: 30
C = 5          # channels (real: 28)

session_ids = np.repeat(SESSIONS, N_PER_SESS).astype(np.int64)
N = len(session_ids)

# Per-session offset and gain to inject (the "drift" we want removed).
SESS_OFFSET = {1: 0.0, 2: 5.0, 3: -3.0}
SESS_GAIN = {1: 1.0, 2: 4.0, 3: 0.25}

eeg = rng.standard_normal((N, G, G, T_EEG)).astype(np.float32)
fnirs = rng.standard_normal((N, S_SEG, G, G, T_FN, 2)).astype(np.float32)
chan = rng.standard_normal((N, C, T_EEG)).astype(np.float32)

for s in SESSIONS:
    m = session_ids == s
    eeg[m] = eeg[m] * SESS_GAIN[s] + SESS_OFFSET[s]
    fnirs[m] = fnirs[m] * SESS_GAIN[s] + SESS_OFFSET[s]
    chan[m] = chan[m] * SESS_GAIN[s] + SESS_OFFSET[s]

# Additionally give HbO and HbR DIFFERENT offsets so check 5 is meaningful.
fnirs[..., 0] += 2.0    # HbO
fnirs[..., 1] += -7.0   # HbR

eeg_n, fnirs_n, chan_n = apply_per_session_normalization(
    eeg, fnirs, chan, session_ids, "TESTVP")

# ---------------------------------------------------------------------------
# Check 1 & 2: per-session, per-grid-cell mean ~0 and std ~1 afterwards.
# ---------------------------------------------------------------------------
for s in SESSIONS:
    m = session_ids == s
    cell_means = eeg_n[m].mean(axis=(0, 3))   # (G, G)
    cell_stds = eeg_n[m].std(axis=(0, 3))     # (G, G)
    assert np.allclose(cell_means, 0.0, atol=1e-5), \
        f"session {s} EEG per-cell mean not ~0: max |mean|={np.abs(cell_means).max():.2e}"
    assert np.allclose(cell_stds, 1.0, atol=1e-4), \
        f"session {s} EEG per-cell std not ~1: {cell_stds.min():.4f}..{cell_stds.max():.4f}"
print("[PASS] check 1+2: injected per-session offsets AND gains removed from EEG "
      "(every session's per-grid-cell mean ~0, std ~1).")

# Confirm the drift really WAS there before (otherwise the test is vacuous).
pre_means = [eeg[session_ids == s].mean() for s in SESSIONS]
assert max(pre_means) - min(pre_means) > 1.0, \
    "test setup failure: no session drift was actually injected"
post_means = [eeg_n[session_ids == s].mean() for s in SESSIONS]
print(f"[INFO]  pre-norm per-session grand means:  {[f'{v:+.3f}' for v in pre_means]}")
print(f"[INFO]  post-norm per-session grand means: {[f'{v:+.3f}' for v in post_means]}")

# ---------------------------------------------------------------------------
# Check 3: normalization is per-cell, not one global scalar. Give two cells
# very different scales and confirm BOTH end up at std ~1 (a global scalar
# normalization could not achieve that).
# ---------------------------------------------------------------------------
eeg2 = rng.standard_normal((N, G, G, T_EEG)).astype(np.float32)
eeg2[:, 0, 0, :] *= 100.0     # one cell wildly larger
eeg2[:, 1, 1, :] *= 0.01      # another wildly smaller
e2n, _, _ = apply_per_session_normalization(
    eeg2, fnirs, chan, session_ids, "TESTVP")
for s in SESSIONS:
    m = session_ids == s
    assert abs(e2n[m][:, 0, 0, :].std() - 1.0) < 1e-3
    assert abs(e2n[m][:, 1, 1, :].std() - 1.0) < 1e-3
print("[PASS] check 3: normalization is PER GRID CELL (cells differing by 10^4 in scale "
      "both land at std ~1), so between-location structure is handled per location, "
      "not washed out by one global scalar.")

# ---------------------------------------------------------------------------
# Check 4: sessions are normalized INDEPENDENTLY -- the concrete form of the
# "no cross-session information flow" claim. Perturb session 3 drastically
# and confirm sessions 1 and 2 outputs are BIT-IDENTICAL.
# ---------------------------------------------------------------------------
eeg_pert = eeg.copy()
eeg_pert[session_ids == 3] *= 1000.0
eeg_pert[session_ids == 3] += 500.0

eeg_pn, _, _ = apply_per_session_normalization(
    eeg_pert, fnirs, chan, session_ids, "TESTVP")

m12 = np.isin(session_ids, [1, 2])
assert np.array_equal(eeg_n[m12], eeg_pn[m12]), \
    "perturbing session 3 changed sessions 1/2 output -- sessions are NOT independent!"
print("[PASS] check 4: perturbing session 3 by 1000x leaves sessions 1 and 2 output "
      "BIT-IDENTICAL -> sessions normalized independently, no cross-session leakage.")

# ---------------------------------------------------------------------------
# Check 5: fNIRS HbO and HbR normalized separately (per chromophore).
# ---------------------------------------------------------------------------
for s in SESSIONS:
    m = session_ids == s
    for chrom, name in [(0, "HbO"), (1, "HbR")]:
        cm = fnirs_n[m][..., chrom].mean()
        cs = fnirs_n[m][..., chrom].std()
        assert abs(cm) < 1e-4, f"session {s} {name} mean not ~0: {cm:.2e}"
        assert abs(cs - 1.0) < 1e-2, f"session {s} {name} std not ~1: {cs:.4f}"
print("[PASS] check 5: HbO and HbR normalized SEPARATELY (each chromophore's own "
      "injected offset removed independently; +2.0 and -7.0 both eliminated).")

# ---------------------------------------------------------------------------
# Check 6: channel-space array normalized per channel per session.
# ---------------------------------------------------------------------------
for s in SESSIONS:
    m = session_ids == s
    ch_means = chan_n[m].mean(axis=(0, 2))   # (C,)
    ch_stds = chan_n[m].std(axis=(0, 2))
    assert np.allclose(ch_means, 0.0, atol=1e-5)
    assert np.allclose(ch_stds, 1.0, atol=1e-4)
print("[PASS] check 6: channel-space EEG normalized per channel per session "
      "(keeps priority-E1 PLV features on the same normalization state).")

# ---------------------------------------------------------------------------
# Check 7: single-window session is skipped safely (no NaN/inf).
# ---------------------------------------------------------------------------
sess_single = np.array([1] * 6 + [2], dtype=np.int64)
n2 = len(sess_single)
eeg_s = rng.standard_normal((n2, G, G, T_EEG)).astype(np.float32)
fn_s = rng.standard_normal((n2, S_SEG, G, G, T_FN, 2)).astype(np.float32)
ch_s = rng.standard_normal((n2, C, T_EEG)).astype(np.float32)

e_s, f_s, c_s = apply_per_session_normalization(eeg_s, fn_s, ch_s, sess_single, "TESTVP")
for arr, name in [(e_s, "eeg"), (f_s, "fnirs"), (c_s, "chan")]:
    assert np.isfinite(arr).all(), f"{name} contains NaN/inf after single-window session"
# The single-window session should be left untouched (not zeroed/NaN).
assert np.array_equal(e_s[sess_single == 2], eeg_s[sess_single == 2]), \
    "single-window session should be left as-is, not modified"
print("[PASS] check 7: single-window session skipped safely (finite output, left unmodified).")

print("\nALL CHECKS PASSED.")
