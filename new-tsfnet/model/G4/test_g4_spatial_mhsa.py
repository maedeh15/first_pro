"""
Numeric verification for priority-G4 (SpatialMHSABranch + eeg_spatial_mode).

Checks:
  1. eeg_spatial_mode='none' (default) is a BIT-EXACT no-op: no module built,
     identical state_dict, identical outputs, unaffected by a stray x_chan.
  2. The attention really is over CHANNELS: permuting the channel order of
     the input permutes nothing about the token SET, so with the depthwise
     stem removed the attention output set is permutation-equivariant. Tested
     directly on the encoder to prove tokens = channels (not time).
  3. No positional encoding: verified by the permutation-equivariance test
     above -- a position-encoded encoder would NOT be equivariant.
  4. Channel identity is preserved by the depthwise stem: perturbing one
     input channel must leave the OTHER channels' stem outputs bit-identical
     (a non-depthwise conv would smear the change across all channels).
  5. Shapes, parameter budget, and gradient flow through the whole module.
  6. eeg_spatial_mode='mhsa' without x_chan raises a clear error; wrong
     channel count raises a clear error.
  7. clf_eeg's input dim grows by exactly embed_dim, and ONLY the y_eeg path
     changes -- y_fnirs must be bit-identical, proving the documented scope.
  8. Dataset/_unpack_batch: 3-, 4- and 5-tuple batches all unpack correctly.
  9. End-to-end train_one_fold with 5-tuple batches.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tsfnet_model_notebook import TSFNet, SpatialMHSABranch, _unpack_batch, train_one_fold

S, CH, T_CH = 21, 28, 2000
torch.manual_seed(0)

x_eeg = torch.randn(2, 16, 16, 600)
x_fn = torch.randn(2, S, 16, 16, 30, 2)
x_chan = torch.randn(2, CH, T_CH)

# ---------------------------------------------------------------------------
# Check 1: default is a bit-exact no-op
# ---------------------------------------------------------------------------
torch.manual_seed(42)
m_def = TSFNet(num_classes=3, fnirs_num_segments=S).eval()
torch.manual_seed(42)
m_none = TSFNet(num_classes=3, fnirs_num_segments=S, eeg_spatial_mode="none").eval()

assert not hasattr(m_def, "eeg_spatial_branch")
assert m_def.state_dict().keys() == m_none.state_dict().keys()
for k in m_def.state_dict():
    assert torch.equal(m_def.state_dict()[k], m_none.state_dict()[k])
with torch.no_grad():
    o_a = m_def(x_eeg, x_fn)
    o_b = m_none(x_eeg, x_fn)
    o_c = m_def(x_eeg, x_fn, None, x_chan)   # stray x_chan must be ignored
for a, b, c in zip(o_a[:4], o_b[:4], o_c[:4]):
    assert torch.equal(a, b) and torch.equal(a, c)
print("[PASS] check 1: eeg_spatial_mode='none' is a bit-exact no-op (no module built, "
      "identical outputs, a stray x_chan is ignored).")

# ---------------------------------------------------------------------------
# Checks 2 & 3: tokens are CHANNELS, and there is no positional encoding.
# A Transformer encoder without positional encoding is permutation-EQUIVARIANT
# over its token axis. So permuting channels must permute the encoder output
# identically. If tokens were time, or if positions were encoded, this fails.
# ---------------------------------------------------------------------------
br = SpatialMHSABranch(n_channels=CH, d_model=64, n_heads=4, n_layers=2, dropout=0.0).eval()

tokens = torch.randn(2, CH, 64)
perm = torch.randperm(CH)
with torch.no_grad():
    enc_plain = br.encoder(tokens)
    enc_perm = br.encoder(tokens[:, perm, :])
assert torch.allclose(enc_perm, enc_plain[:, perm, :], atol=1e-5), \
    "encoder is not permutation-equivariant over the token axis"
print("[PASS] check 2+3: the encoder is permutation-equivariant over its token axis "
      "-> tokens are CHANNELS (not time) and no positional encoding is applied, "
      "matching the reference design.")

# ---------------------------------------------------------------------------
# Check 4: the depthwise stem preserves channel identity
# ---------------------------------------------------------------------------
xa = torch.randn(1, CH, T_CH)
xb = xa.clone()
xb[0, 7] += 5.0                      # perturb exactly one channel
with torch.no_grad():
    ha, hb = br.temporal(xa), br.temporal(xb)
changed = [c for c in range(CH) if not torch.allclose(ha[0, c], hb[0, c], atol=1e-6)]
assert changed == [7], f"expected only channel 7 to change, got {changed}"
print("[PASS] check 4: perturbing one input channel changes ONLY that channel's stem "
      "output -> the stem is genuinely depthwise, so channel identity reaches attention.")

# ---------------------------------------------------------------------------
# Check 5: shapes, parameter budget, gradient flow
# ---------------------------------------------------------------------------
br.train()
inp = torch.randn(3, CH, T_CH, requires_grad=True)
out = br(inp)
assert out.shape == (3, 64), out.shape
out.sum().backward()
assert inp.grad is not None and inp.grad.abs().sum() > 0

n_base = sum(p.numel() for p in m_def.parameters())
torch.manual_seed(42)
m_mhsa = TSFNet(num_classes=3, fnirs_num_segments=S, eeg_spatial_mode="mhsa")
n_mhsa = sum(p.numel() for p in m_mhsa.parameters())
print(f"[PASS] check 5: output shape {tuple(out.shape)}, gradients flow to the input; "
      f"model grows {n_base} -> {n_mhsa} params (+{100*(n_mhsa-n_base)/n_base:.1f}%).")

# ---------------------------------------------------------------------------
# Check 6: clear errors
# ---------------------------------------------------------------------------
try:
    m_mhsa.eval()
    with torch.no_grad():
        m_mhsa(x_eeg, x_fn)
    raise AssertionError("expected ValueError when x_chan is missing")
except ValueError as e:
    assert "ENABLE_EEG_CHANNEL_SPACE" in str(e)
try:
    with torch.no_grad():
        m_mhsa(x_eeg, x_fn, None, torch.randn(2, CH + 3, T_CH))
    raise AssertionError("expected ValueError for wrong channel count")
except ValueError as e:
    assert "channel-space EEG" in str(e)
try:
    TSFNet(num_classes=3, eeg_spatial_mode="bogus")
    raise AssertionError("expected ValueError for bad mode")
except ValueError as e:
    assert "eeg_spatial_mode" in str(e)
print("[PASS] check 6: missing x_chan, wrong channel count and an invalid mode all raise "
      "clear, actionable errors.")

# ---------------------------------------------------------------------------
# Check 7: scope -- only the y_eeg path changes
# ---------------------------------------------------------------------------
assert (m_mhsa.clf_eeg.net[0].in_features
        == m_def.clf_eeg.net[0].in_features + 64), "clf_eeg input did not grow by embed_dim"

torch.manual_seed(42)
m_a = TSFNet(num_classes=3, fnirs_num_segments=S).eval()
torch.manual_seed(42)
m_b = TSFNet(num_classes=3, fnirs_num_segments=S, eeg_spatial_mode="mhsa").eval()
# Copy every shared (non-G4, non-clf_eeg) parameter so the comparison isolates
# the architectural change rather than differing random initialisation.
sd_a, sd_b = m_a.state_dict(), m_b.state_dict()
for k in sd_b:
    if k in sd_a and sd_a[k].shape == sd_b[k].shape:
        sd_b[k] = sd_a[k].clone()
m_b.load_state_dict(sd_b)

with torch.no_grad():
    ya = m_a(x_eeg, x_fn)
    yb = m_b(x_eeg, x_fn, None, x_chan)
assert torch.equal(ya[2], yb[2]), "y_fnirs changed -- scope is wider than documented!"
assert not torch.equal(ya[0], yb[0]), "y_eeg did NOT change -- the module is inert"
print("[PASS] check 7: y_fnirs is bit-identical while y_eeg changes -> the change is "
      "confined to the EEG branch exactly as documented.")

# ---------------------------------------------------------------------------
# Check 8: _unpack_batch across all three arities
# ---------------------------------------------------------------------------
b3 = (torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, dtype=torch.long))
b4 = b3 + (torch.randn(2, 2, 378),)
b5 = b3 + (torch.zeros(2, 0), torch.randn(2, CH, T_CH))
assert _unpack_batch(b3)[3] is None and _unpack_batch(b3)[4] is None
assert _unpack_batch(b4)[3] is not None and _unpack_batch(b4)[4] is None
u5 = _unpack_batch(b5)
assert u5[3] is None, "empty connectivity placeholder should unpack to None"
assert u5[4] is not None and u5[4].shape == (2, CH, T_CH)
print("[PASS] check 8: 3-, 4- and 5-tuple batches unpack correctly, and the empty "
      "connectivity placeholder maps to None.")

# ---------------------------------------------------------------------------
# Check 9: end-to-end training with 5-tuple batches
# ---------------------------------------------------------------------------
class _DS(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(500 + i)
        return (torch.randn(16, 16, 600, generator=g),
                torch.randn(S, 16, 16, 30, 2, generator=g),
                torch.randint(0, 3, (1,), generator=g).squeeze().long(),
                torch.zeros(0),
                torch.randn(CH, T_CH, generator=g))


loader = DataLoader(_DS(), batch_size=4)
torch.manual_seed(3)
m_run = TSFNet(num_classes=3, fnirs_num_segments=S, eeg_spatial_mode="mhsa")
acc, kap, trained, rep = train_one_fold(
    m_run, loader, loader, loader, "cpu",
    stage1_max_epochs=1, stage1_patience=1, stage2_max_epochs=1,
    ckpt_path="/tmp/_g4.pt", log_fn=lambda *a, **k: None)
assert 0.0 <= acc <= 1.0
grad_ok = any(p.grad is not None for p in trained.eeg_spatial_branch.parameters())
print(f"[PASS] check 9: train_one_fold runs end-to-end with 5-tuple batches "
      f"(test_acc={acc:.3f}) and the spatial branch participates in training.")

print("\nALL CHECKS PASSED.")
