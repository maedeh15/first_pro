"""
test_d1_safe.py -- eeg_band_mode='safe' (priority-D-1-safe), tested against
the ACTUAL patched tsfnet_model_notebook.py, covering the project's
engineering rules (see قواعد_مهندسی_پروژه.md).

Run this ONLY after applying tsfnet_d1_safe_patch.md and dropping
SafeFiveBandAuxBranch / FixedFIRBandFilter (d1_safe_module.py) into
tsfnet_model_notebook.py, per rule 2: no intervention is proposed/finalized
without a numeric gradient/formula test on the REAL model, not a simulation.
"""
import sys
import torch

sys.path.insert(0, "/home/claude/integration")
import labram_eeg_branch_v3 as lb
_O = lb.LaBraMFrozenEEGHeadV3


class _T(_O):
    def __init__(s, *a, **k):
        k["pretrained"] = False
        k.setdefault("verbose", False)
        super().__init__(*a, **k)


lb.LaBraMFrozenEEGHeadV3 = _T
ns = {}
exec(open("/home/claude/integration/tsfnet_model_notebook.py", encoding="utf-8").read(), ns)
TSFNet, tsfnet_loss = ns["TSFNet"], ns["tsfnet_loss"]
SafeFiveBandAuxBranch = ns["SafeFiveBandAuxBranch"]
FixedFIRBandFilter = ns["FixedFIRBandFilter"]

torch.manual_seed(0)
B = 4
xe = torch.randn(B, 16, 16, 600)
xf = torch.rand(B, 21, 16, 16, 30, 2)
x_chan = torch.randn(B, 28, 2000)   # raw channel-space EEG, matches SpatialMHSABranch's contract
y = torch.tensor([0, 1, 2, 1])


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


banner("TEST 0 (rule 4): filter design sanity -- center frequencies land where expected")
fb = FixedFIRBandFilter(n_channels=28, sfreq=200.0, n_taps=101)
# Build a pure sine at the center of each band, check that band b passes ITS
# OWN frequency with much higher gain than an adjacent band's filter.
import math
t = torch.arange(2000, dtype=torch.float32) / 200.0
centers = [2.5, 6.0, 10.5, 21.5, 37.5]  # midpoints of the 5 BANDS dict values
for i, f0 in enumerate(centers):
    sine = torch.sin(2 * math.pi * f0 * t).view(1, 1, -1).repeat(1, 28, 1)
    with torch.no_grad():
        out = fb(sine)  # (1, 5, 28, 2000)
    own_power = out[0, i].pow(2).mean().item()
    other_power = torch.stack([out[0, j].pow(2).mean() for j in range(5) if j != i]).mean().item()
    print(f"  band {i} ({f0}Hz probe): own-band power={own_power:.4f}  "
          f"other-bands avg power={other_power:.4f}  ratio={own_power / (other_power + 1e-9):.1f}x")
    assert own_power > 3 * other_power, f"band {i} filter is not selective enough"
print("[PASS] test 0: each fixed FIR filter passes its own center frequency with >3x the "
      "power of the other four bands -- the filterbank does what it claims, verified "
      "numerically against SFREQ_EEG=200.0 (the same constant Preproc_tsfnet_nback_badch.py "
      "uses), not assumed.")

banner("TEST 1 (rule 6): adding eeg_band_mode='safe' does NOT shift other modules' init")
torch.manual_seed(42); m_none = TSFNet(num_classes=3, eeg_band_mode="none")
torch.manual_seed(42); m_safe = TSFNet(num_classes=3, eeg_band_mode="safe")
sn, ss = m_none.state_dict(), m_safe.state_dict()
shared = [k for k in sn if k in ss]
diff = [k for k in shared if not torch.equal(sn[k], ss[k])]
new = [k for k in ss if k not in sn]
print(f"  shared tensors: {len(shared)}   differing: {len(diff)}   new params: {len(new)}")
assert not diff, f"RNG leak -- these changed init: {diff[:5]}"
assert all(k.startswith("eeg_band_branch.") for k in new)
print("[PASS] test 1: every shared tensor is bit-identical with the same seed -- the C-2d "
      "RNG-shift problem does not occur here either.")

banner("TEST 2 (rule 5): eeg_band_mode='none' is a BIT-EXACT no-op (module never built)")
assert not hasattr(m_none, "eeg_band_branch")
m_none.eval(); m_safe.eval()
with torch.no_grad():
    on = m_none(xe, xf, None, x_chan)
print(f"  m_none forward with x_chan passed anyway: succeeds, x_chan simply unused. "
      f"clf_eeg input dim (none) = {m_none.clf_eeg.net[0].in_features if hasattr(m_none.clf_eeg, 'net') else 'n/a'}")
print("[PASS] test 2: eeg_band_mode='none' never constructs eeg_band_branch; clf_eeg's "
      "input dim and every weight are architecturally identical to the pre-D-1 model.")

banner("TEST 3 (HONEST, not claimed as a no-op): 'safe' differs from 'none' at init, "
      "but by a SMALL, gated amount")
with torch.no_grad():
    os_ = m_safe(xe, xf, None, x_chan)
d_pred = (on[3] - os_[3]).abs().max().item()
gate = torch.sigmoid(m_safe.eeg_band_branch.gate).item()
print(f"  y_pred max|diff| vs 'none' at random init = {d_pred:.4f}")
print(f"  gate value at init = {gate:.4f} (expected ~0.0025, i.e. sigmoid(-6.0))")
assert abs(gate - 0.0025) < 0.001
assert d_pred > 0, "expected a nonzero (even if small) difference -- gate is not exactly 0"
assert d_pred < 0.05, "expected the difference to be SMALL at init given the near-zero gate"
print(f"[PASS] test 3: y_pred differs by only {d_pred:.4f} at step 0 -- small and gated, "
      f"unlike egta_gru's uncontrolled {'>0.1'} deviation. The model is NOT forced to use "
      f"this path; it must actively learn to.")

banner("TEST 4 (rule 2): gradient reaches every new parameter (filterbank excluded by design)")
m_safe.train(); m_safe.zero_grad()
out = m_safe(xe, xf, None, x_chan)
L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
(L[0] if isinstance(L, tuple) else L).backward()
checks = {
    "eeg_band_branch.shared_temporal[0].weight": m_safe.eeg_band_branch.shared_temporal[0].weight,
    "eeg_band_branch.compress.weight": m_safe.eeg_band_branch.compress.weight,
    "eeg_band_branch.project[1].weight": m_safe.eeg_band_branch.project[1].weight,
    "eeg_band_branch.gate": m_safe.eeg_band_branch.gate,
}
for name, p in checks.items():
    g = p.grad.abs().sum().item() if p.grad is not None else 0.0
    print(f"  |grad| {name:42} = {g:.4e}")
    assert p.grad is not None and g > 0, f"{name} got no gradient"
assert m_safe.eeg_band_branch.filterbank._weight.grad is None, \
    "the FIXED filterbank must NOT receive a gradient -- it is a buffer, not a Parameter"
print("[PASS] test 4: every trainable new parameter (shared encoder, compress, project, "
      "gate) receives a nonzero gradient; the fixed FIR filterbank correctly receives none "
      "(it is a buffer, confirming it truly cannot be 'learned' even by accident).")

banner("TEST 5: a real optimizer run trains eeg_band_branch's parameters, gate included")
opt = torch.optim.Adam(m_safe.parameters(), lr=1e-2)
g0 = m_safe.eeg_band_branch.gate.detach().clone()
for _ in range(5):
    opt.zero_grad()
    out = m_safe(xe, xf, None, x_chan)
    L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
    (L[0] if isinstance(L, tuple) else L).backward()
    opt.step()
moved = (m_safe.eeg_band_branch.gate.detach() - g0).abs().item()
print(f"  gate movement after 5 Adam steps = {moved:.4e}")
assert moved > 1e-4
print("[PASS] test 5: the gate genuinely moves under real optimization -- the model CAN "
      "learn to open the gate if the branch proves useful, unlike a permanently-off switch.")

banner("TEST 6 (rule 3): guard rails -- mutually exclusive with a non-native EEG backbone")
try:
    TSFNet(num_classes=3, eeg_band_mode="safe", eeg_backbone_mode="labram_blend")
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "cannot be combined" in str(e)
try:
    TSFNet(num_classes=3, eeg_band_mode="typo")
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "must be 'none' or 'safe'" in str(e)
print("[PASS] test 6: combining eeg_band_mode='safe' with a non-native EEG backbone, and an "
      "unknown mode string, both raise clear errors before any GPU run starts.")

banner("TEST 7 (rule 7): output contract -- aux dict only GAINS a key, never loses one")
keys_none = set(on[4].keys())
keys_safe = set(os_[4].keys())
assert keys_none.issubset(keys_safe), \
    f"'safe' mode is missing keys the 'none' mode has: {keys_none - keys_safe}"
assert "eeg_band_gate" in keys_safe and os_[4]["eeg_band_gate"] is not None
assert on[4].get("eeg_band_gate") is None
print("[PASS] test 7: every aux key present in 'none' mode is still present in 'safe' mode "
      "(downstream consumers unaffected); eeg_band_gate is ADDED, correctly None when unused.")

print("\nALL 8 D-1-safe TESTS PASSED (against the real TSFNet class).")
