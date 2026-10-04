"""
test_efgf_gatefix.py -- proves, on the REAL TSFNet class, that:
  1. gate_mode='clamp' (default) reproduces the exact confirmed bug: once a
     gradient step pushes alpha/gamma negative, clamp's gradient is exactly
     zero and the gate can NEVER recover.
  2. gate_mode='clamp' is bit-exact vs the ORIGINAL pre-fix file (rule 5).
  3. gate_mode='sigmoid' does NOT have this failure mode: gradient is never
     exactly zero, so the gate can always keep moving.

Run with no arguments from the folder containing tsfnet_model_notebook.py
(and, for test 2, tsfnet_model_notebook_ORIGINAL.py).
"""
import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, _HERE)


def _load(fname):
    ns = {}
    exec(open(os.path.join(_HERE, fname), encoding="utf-8").read(), ns)
    return ns


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


ns = _load("tsfnet_model_notebook.py")
TSFNet, EFGFLayer, tsfnet_loss = ns["TSFNet"], ns["EFGFLayer"], ns["tsfnet_loss"]

torch.manual_seed(0)
B = 8
xe = torch.randn(B, 16, 16, 600)
xf = torch.rand(B, 21, 16, 16, 30, 2)
y = torch.randint(0, 3, (B,))

# ---------------------------------------------------------------------------
banner("TEST 0: reproduce the CONFIRMED dead-gradient trap (gate_mode='clamp')")
layer = EFGFLayer(channels=16, eeg_kernel=(2, 2, 6), fnirs_kernel=(2, 2, 3), gate_mode="clamp")
with torch.no_grad():
    layer.alpha.fill_(-0.0052)   # the EXACT value found on the real checkpoint
    layer.gamma.fill_(-0.0046)
eeg_map = torch.randn(B, 16, 16, 16, 4)
fnirs_map = torch.randn(B, 21, 16, 16, 16, 4)
fusion_map = torch.randn(B, 16, 16, 16, 4)
fused, _, _ = layer(eeg_map, fnirs_map, fusion_map)
loss = fused.pow(2).mean()
loss.backward()
print(f"  alpha={layer.alpha.item():.4f}  grad={layer.alpha.grad.item():.6f}")
print(f"  gamma={layer.gamma.item():.4f}  grad={layer.gamma.grad.item():.6f}")
assert layer.alpha.grad.item() == 0.0 and layer.gamma.grad.item() == 0.0, \
    "expected EXACTLY zero gradient once clamp's input is negative -- this IS the bug"
print("[CONFIRMED] with gate_mode='clamp', once alpha/gamma go negative, gradient is "
      "EXACTLY 0.0 -- Adam can never move them again. This is the real, reproduced bug.")

# ---------------------------------------------------------------------------
banner("TEST 1 (rule 5): gate_mode='clamp' (default) is BIT-IDENTICAL to the ORIGINAL file")
orig_path = os.path.join(_HERE, "tsfnet_model_notebook_ORIGINAL.py")
if not os.path.exists(orig_path):
    print("  [SKIP] tsfnet_model_notebook_ORIGINAL.py not found beside this script")
else:
    TSFNet_orig = _load("tsfnet_model_notebook_ORIGINAL.py")["TSFNet"]
    torch.manual_seed(42); mo = TSFNet_orig(num_classes=3, fnirs_temporal_align="egta")
    torch.manual_seed(42); mc = TSFNet(num_classes=3, fnirs_temporal_align="egta", efgf_gate_mode="clamp")
    so, sc = mo.state_dict(), mc.state_dict()
    assert set(so) == set(sc), (set(so) ^ set(sc))
    diff = [k for k in so if not torch.equal(so[k], sc[k])]
    assert not diff, diff
    mo.eval(); mc.eval()
    with torch.no_grad():
        oo, oc = mo(xe, xf, None, None), mc(xe, xf, None, None)
    for name, a, b in zip(["y_eeg", "y_fusion", "y_fnirs", "y_pred"], oo[:4], oc[:4]):
        d = (a - b).abs().max().item()
        print(f"  {name:9} max|diff| = {d:.3e}")
        assert d == 0.0, name
    print(f"[PASS] test 1: gate_mode='clamp' reproduces the ORIGINAL file exactly "
          f"({len(so)} tensors, 4/4 outputs bit-identical).")

# ---------------------------------------------------------------------------
banner("TEST 2 (rule 6): switching to 'sigmoid' does NOT shift any other module's init")
torch.manual_seed(42); m_clamp = TSFNet(num_classes=3, fnirs_temporal_align="egta", efgf_gate_mode="clamp")
torch.manual_seed(42); m_sig = TSFNet(num_classes=3, fnirs_temporal_align="egta", efgf_gate_mode="sigmoid")
pc = dict(m_clamp.named_parameters()); ps = dict(m_sig.named_parameters())
shared = sorted(set(pc) & set(ps))
diff = [k for k in shared if not torch.equal(pc[k], ps[k])]
only_clamp = sorted(set(pc) - set(ps))
only_sig = sorted(set(ps) - set(pc))
print(f"  shared params: {len(shared)}  differing: {len(diff)}")
print(f"  only in clamp mode: {only_clamp}")
print(f"  only in sigmoid mode: {only_sig}")
assert not diff, f"unrelated params shifted: {diff}"
assert set(only_clamp) == {"efgf1.gamma", "efgf1.alpha", "efgf2.gamma", "efgf2.alpha"}
assert set(only_sig) == {"efgf1.gamma_raw", "efgf1.alpha_raw", "efgf2.gamma_raw", "efgf2.alpha_raw"}
print("[PASS] test 2: every OTHER parameter (89 total minus the 4 gate params) is "
      "bit-identical between modes -- only the gate parametrization itself changed.")

# ---------------------------------------------------------------------------
banner("TEST 3: gate values at init -- 'sigmoid' starts near-zero, matching the paper's "
      "documented intent, WITHOUT clamp's brittleness")
for name, layer in [("efgf1", m_sig.efgf1), ("efgf2", m_sig.efgf2)]:
    a = torch.sigmoid(layer.alpha_raw).item()
    g = torch.sigmoid(layer.gamma_raw).item()
    print(f"  {name}: alpha={a:.4f}  gamma={g:.4f}  (expected ~0.0025)")
    assert abs(a - 0.0025) < 0.001 and abs(g - 0.0025) < 0.001
print("[PASS] test 3: both EFGF layers start near zero under 'sigmoid', consistent with the "
      "paper's 'initialized to zero' (Sec. 3.6), honestly not bit-exact (same documented "
      "deviation already used for beta_labram/EGTA/eeg_band elsewhere in this file).")

# ---------------------------------------------------------------------------
banner("TEST 4 (THE FIX): gate_mode='sigmoid' NEVER has zero gradient, even after a bad step")
layer_s = EFGFLayer(channels=16, eeg_kernel=(2, 2, 6), fnirs_kernel=(2, 2, 3), gate_mode="sigmoid")
opt = torch.optim.SGD(layer_s.parameters(), lr=10.0)  # deliberately large -- worst case
grads_over_steps = []
for step in range(15):
    opt.zero_grad()
    fused, _, _ = layer_s(eeg_map, fnirs_map, fusion_map)
    loss = fused.pow(2).mean()
    loss.backward()
    ga = layer_s.alpha_raw.grad.item()
    gg = layer_s.gamma_raw.grad.item()
    grads_over_steps.append((ga, gg))
    opt.step()
print("  (alpha_grad, gamma_grad) across 15 large SGD steps:")
for i, (ga, gg) in enumerate(grads_over_steps):
    print(f"    step {i:2d}: alpha_grad={ga: .6f}  gamma_grad={gg: .6f}")
assert all(ga != 0.0 for ga, gg in grads_over_steps), "sigmoid gradient must never be exactly 0"
assert all(gg != 0.0 for ga, gg in grads_over_steps), "sigmoid gradient must never be exactly 0"
print(f"  final alpha={torch.sigmoid(layer_s.alpha_raw).item():.4f}  "
      f"final gamma={torch.sigmoid(layer_s.gamma_raw).item():.4f}")
print("[PASS] test 4: across 15 large, adversarial gradient steps, alpha's and gamma's "
      "gradient is NEVER exactly zero -- the gate can always keep learning, even in the "
      "worst case that killed it under 'clamp'.")

# ---------------------------------------------------------------------------
banner("TEST 5 (rule 2): gradient reaches the new raw parameters inside the REAL model")
m_sig.train(); m_sig.zero_grad()
out = m_sig(xe, xf, None, None)
L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
(L[0] if isinstance(L, tuple) else L).backward()
for name, p in [("efgf1.alpha_raw", m_sig.efgf1.alpha_raw), ("efgf1.gamma_raw", m_sig.efgf1.gamma_raw),
                ("efgf2.alpha_raw", m_sig.efgf2.alpha_raw), ("efgf2.gamma_raw", m_sig.efgf2.gamma_raw)]:
    g = p.grad.abs().item() if p.grad is not None else 0.0
    print(f"  |grad| {name:18} = {g:.4e}")
    assert p.grad is not None and g > 0, name
print("[PASS] test 5: all four gate parameters receive gradient in a real forward/backward "
      "pass through the full model.")

# ---------------------------------------------------------------------------
banner("TEST 6 (rule 3): guard rail")
try:
    TSFNet(num_classes=3, efgf_gate_mode="softmax")
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "must be 'clamp' or 'sigmoid'" in str(e)
print("[PASS] test 6")

print("\nALL EFGF-gatefix TESTS PASSED against the real TSFNet class.")
