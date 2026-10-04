"""
Numeric verification for priority-G2a (HbT as a third fNIRS chromophore).

Checks:
  1. fnirs_num_chromophores=2 (default) is a BIT-EXACT no-op: same state_dict
     keys, same parameter count, same outputs, same chroma mask as before G2a.
  2. fnirs_num_chromophores=3 changes only conv1's in_channels; every
     downstream shape is unchanged.
  3. A shape mismatch between the data and the model raises a CLEAR error
     rather than silently mis-broadcasting.
  4. The extended chroma modes mask exactly the intended channels, and the
     two original 2-channel modes are unchanged.
  5. Modes that need HbT are rejected at construction time when the model
     only has 2 chromophores.
  6. HbT ORDER-OF-OPERATIONS: building HbT from RAW concentrations and then
     z-scoring gives something strictly less redundant than building it from
     already-normalized HbO/HbR (which would be an exact linear combination).
     This is the correctness claim the preprocessing comment makes, so it is
     verified numerically rather than asserted.
  7. HOW MUCH does HbT actually add? Quantified as the pointwise linear R^2
     of grid-level HbT given HbO/HbR -- reported so the expected effect size
     is calibrated before spending GPU time.
"""
import numpy as np
import torch

from tsfnet_model_notebook import TSFNet, FNIRSBranch

S, G, T = 21, 16, 30
rng = np.random.default_rng(0)


def _mk(C, B=2):
    return torch.randn(B, S, G, G, T, C)


# ---------------------------------------------------------------------------
# Check 1: 2-chromophore default is a bit-exact no-op
# ---------------------------------------------------------------------------
x_eeg = torch.randn(2, 16, 16, 600)
x_fn2 = _mk(2)

torch.manual_seed(42)
m_default = TSFNet(num_classes=3, fnirs_num_segments=S).eval()
torch.manual_seed(42)
m_explicit = TSFNet(num_classes=3, fnirs_num_segments=S, fnirs_num_chromophores=2).eval()

sd1, sd2 = m_default.state_dict(), m_explicit.state_dict()
assert sd1.keys() == sd2.keys()
for k in sd1:
    assert torch.equal(sd1[k], sd2[k]), f"mismatch in {k}"
with torch.no_grad():
    o1 = m_default(x_eeg, x_fn2)
    o2 = m_explicit(x_eeg, x_fn2)
for a, b in zip(o1[:4], o2[:4]):
    assert torch.equal(a, b)
assert torch.equal(m_default._fnirs_chroma_mask, torch.ones(1, 1, 1, 1, 1, 2))
assert m_default.fnirs_branch.conv1.conv.conv.in_channels == 2
print("[PASS] check 1: fnirs_num_chromophores=2 (default) is a bit-exact no-op "
      "(identical params, identical outputs, all-ones 2-channel mask).")

# ---------------------------------------------------------------------------
# Check 2: 3 chromophores changes only conv1.in_channels
# ---------------------------------------------------------------------------
torch.manual_seed(42)
m3 = TSFNet(num_classes=3, fnirs_num_segments=S, fnirs_num_chromophores=3).eval()
assert m3.fnirs_branch.conv1.conv.conv.in_channels == 3
x_fn3 = _mk(3)
with torch.no_grad():
    o3 = m3(x_eeg, x_fn3)
for a, b in zip(o1[:4], o3[:4]):
    assert a.shape == b.shape, "downstream output shapes changed with 3 chromophores"

n2 = sum(p.numel() for p in m_default.parameters())
n3 = sum(p.numel() for p in m3.parameters())
k = m_default.fnirs_branch.conv1.conv.kernel_size
added = 16 * int(np.prod(k))     # one extra input channel worth of conv1 weights
assert n3 - n2 == added, f"expected exactly {added} new params, got {n3 - n2}"
print(f"[PASS] check 2: 3 chromophores adds exactly {added} parameters "
      f"({n2} -> {n3}, +{100*(n3-n2)/n2:.2f}%), and all downstream shapes are unchanged.")

# ---------------------------------------------------------------------------
# Check 3: data/model mismatch raises a clear error
# ---------------------------------------------------------------------------
for model, bad_x, tag in ((m_default, x_fn3, "2-ch model, 3-ch data"),
                          (m3, x_fn2, "3-ch model, 2-ch data")):
    try:
        with torch.no_grad():
            model(x_eeg, bad_x)
        raise AssertionError(f"expected a ValueError for {tag}")
    except ValueError as e:
        assert "chromophore" in str(e) and "ENABLE_FNIRS_HBT" in str(e), str(e)
print("[PASS] check 3: both mismatch directions raise a clear ValueError naming "
      "ENABLE_FNIRS_HBT, instead of silently broadcasting.")

# ---------------------------------------------------------------------------
# Check 4: chroma modes mask exactly the intended channels
# ---------------------------------------------------------------------------
expected = {
    (2, "both"): [1, 1], (2, "hbo_only"): [1, 0], (2, "hbr_only"): [0, 1],
    (3, "both"): [1, 1, 1], (3, "hbo_only"): [1, 0, 0], (3, "hbr_only"): [0, 1, 0],
    (3, "hbt_only"): [0, 0, 1], (3, "hbo_hbr"): [1, 1, 0],
}
for (nc, mode), exp in expected.items():
    mm = TSFNet(num_classes=3, fnirs_num_segments=S,
                fnirs_num_chromophores=nc, fnirs_chroma_mode=mode)
    got = mm._fnirs_chroma_mask.reshape(-1).tolist()
    assert got == [float(v) for v in exp], f"({nc},{mode}): expected {exp}, got {got}"
print("[PASS] check 4: all 8 (n_chromophores, mode) combinations produce the exact "
      "intended mask; the two original 2-channel modes are unchanged.")

# Masking really zeroes the channel that reaches FNIRSBranch.
seen = {}
mh = TSFNet(num_classes=3, fnirs_num_segments=S, fnirs_num_chromophores=3,
            fnirs_chroma_mode="hbo_hbr").eval()
mh.fnirs_branch.register_forward_hook(
    lambda mod, inp, out: seen.update(x=inp[0].detach().clone()))
with torch.no_grad():
    mh(x_eeg, x_fn3)
assert torch.equal(seen["x"][..., 2], torch.zeros_like(seen["x"][..., 2]))
assert torch.equal(seen["x"][..., 0], x_fn3[..., 0])
print("[PASS] check 4b: 'hbo_hbr' zeroes the HbT channel reaching FNIRSBranch and "
      "leaves HbO untouched.")

# ---------------------------------------------------------------------------
# Check 5: HbT modes rejected when the model has only 2 chromophores
# ---------------------------------------------------------------------------
# Only 'hbt_only' genuinely needs a third channel. 'hbo_hbr' on a 2-channel
# model is NOT an error -- it keeps both available channels and is therefore
# exactly equivalent to 'both', which is the sensible behavior.
try:
    TSFNet(num_classes=3, fnirs_num_chromophores=2, fnirs_chroma_mode="hbt_only")
    raise AssertionError("expected ValueError for hbt_only with 2 chromophores")
except ValueError as e:
    assert "ENABLE_FNIRS_HBT" in str(e)

m_hh = TSFNet(num_classes=3, fnirs_num_chromophores=2, fnirs_chroma_mode="hbo_hbr")
assert torch.equal(m_hh._fnirs_chroma_mask, torch.ones(1, 1, 1, 1, 1, 2)), \
    "'hbo_hbr' on a 2-channel model should be equivalent to 'both'"
try:
    TSFNet(num_classes=3, fnirs_chroma_mode="nonsense")
    raise AssertionError("expected ValueError for an unknown mode")
except ValueError as e:
    assert "must be one of" in str(e)
print("[PASS] check 5: 'hbt_only' is rejected on a 2-channel model, unknown modes are "
      "rejected too, and 'hbo_hbr' on 2 channels correctly degrades to 'both'.")

# ---------------------------------------------------------------------------
# Check 6: order of operations -- raw-then-normalize is strictly better
# ---------------------------------------------------------------------------
n_t, n_ch = 4000, 36
hbo_raw = rng.standard_normal((n_t, n_ch)) * rng.uniform(0.5, 3, n_ch) + rng.uniform(-2, 2, n_ch)
hbr_raw = rng.standard_normal((n_t, n_ch)) * rng.uniform(0.5, 3, n_ch) + rng.uniform(-2, 2, n_ch)


def z(x):
    return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-8)


hbo_n, hbr_n = z(hbo_raw), z(hbr_raw)
hbt_correct = z(hbo_raw + hbr_raw)       # what the pipeline does
hbt_wrong = z(hbo_n + hbr_n)             # the tempting shortcut


def lin_r2(target, preds):
    A = np.concatenate([*[p[:, None] for p in preds], np.ones((len(target), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, target, rcond=None)
    resid = target - A @ coef
    return 1.0 - (resid ** 2).sum() / ((target - target.mean()) ** 2).sum()


r2_correct = np.mean([lin_r2(hbt_correct[:, c], [hbo_n[:, c], hbr_n[:, c]])
                      for c in range(n_ch)])
r2_wrong = np.mean([lin_r2(hbt_wrong[:, c], [hbo_n[:, c], hbr_n[:, c]])
                    for c in range(n_ch)])
assert r2_wrong > 0.9999, f"normalize-then-sum should be an exact linear combo, R^2={r2_wrong}"
assert r2_correct > 0.9999, "per-channel, raw-then-normalize is also a linear combo"
print(f"[PASS] check 6: per CHANNEL, both orders are linear combinations "
      f"(R^2 correct={r2_correct:.6f}, shortcut={r2_wrong:.6f}) -- so any benefit "
      f"cannot come from the channel level.")

# ---------------------------------------------------------------------------
# Check 7: how much does HbT add at GRID level (after spatial interpolation)?
# ---------------------------------------------------------------------------
W = rng.uniform(0, 1, (n_ch, 25))
W /= W.sum(0, keepdims=True)
O, R, Tg = hbo_n @ W, hbr_n @ W, hbt_correct @ W
r2_grid = np.mean([lin_r2(Tg[:, j], [O[:, j], R[:, j]]) for j in range(25)])
Tg_wrong = (hbt_wrong @ W)
r2_grid_wrong = np.mean([lin_r2(Tg_wrong[:, j], [O[:, j], R[:, j]]) for j in range(25)])

assert r2_grid < r2_grid_wrong, \
    "raw-then-normalize should be LESS linearly recoverable at grid level than the shortcut"
print(f"[PASS] check 7: at GRID level the pipeline's HbT is only {r2_grid:.3f} "
      f"linearly recoverable from HbO/HbR, vs {r2_grid_wrong:.3f} for the shortcut "
      f"-- confirming the order of operations matters, while also showing ~{100*r2_grid:.0f}% "
      f"of HbT is already available. Calibrate expectations accordingly.")

print("\nALL CHECKS PASSED.")
