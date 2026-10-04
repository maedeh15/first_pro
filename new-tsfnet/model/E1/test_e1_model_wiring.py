"""
Numeric verification for priority-E1 model-side wiring:
  - ConnectivityBranch module correctness (shape, gradient flow).
  - TSFNet(use_fc_features=False) (default) is a BIT-EXACT no-op vs. the
    pre-E1 model: same state_dict keys, same output, for the same seed.
  - TSFNet(use_fc_features=True) requires x_conn (raises a clear error if
    omitted), produces the expected output shapes, and the connectivity
    embedding actually influences y_fusion/y_pred (gradient reaches
    connectivity_branch's parameters).
  - NBackDataset's variable-arity __getitem__ + _unpack_batch: a 3-tuple
    dataset (no connectivity file) is handled by _unpack_batch exactly as
    before (connectivity=None), and a 4-tuple (synthetic, connectivity
    present) is unpacked correctly.
  - evaluate()/train_one_fold() run end-to-end with use_fc_features=True on
    tiny synthetic data (plumbing smoke test), matching the pattern already
    used in test_b1_hsic.py's check 5.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tsfnet_model_notebook import (
    TSFNet, ConnectivityBranch, _unpack_batch, evaluate, train_one_fold,
)

torch.manual_seed(0)

N_CHANNELS = 28
N_PAIRS = N_CHANNELS * (N_CHANNELS - 1) // 2   # 378
S = 21  # fnirs_num_segments

# ---------------------------------------------------------------------------
# Check 1: ConnectivityBranch shape + gradient flow in isolation.
# ---------------------------------------------------------------------------
branch = ConnectivityBranch(n_channels=N_CHANNELS, n_bands=2, embed_dim=64, dropout=0.0).train()
x_conn = torch.randn(4, 2, N_PAIRS, requires_grad=True)
out = branch(x_conn)
assert out.shape == (4, 64), out.shape
out.sum().backward()
assert x_conn.grad is not None and x_conn.grad.abs().sum() > 0
print(f"[PASS] check 1: ConnectivityBranch output shape={tuple(out.shape)}, "
      f"gradient reaches input.")

# Wrong-shaped input raises a clear error, not an opaque matmul crash.
try:
    branch(torch.randn(4, 2, N_PAIRS + 5))
    raise AssertionError("expected ValueError for mismatched connectivity input size")
except ValueError as e:
    assert "ConnectivityBranch expected" in str(e)
    print("[PASS] check 1b: mismatched input size raises a clear ValueError.")

# ---------------------------------------------------------------------------
# Check 2: use_fc_features=False (default) is a bit-exact no-op.
# ---------------------------------------------------------------------------
B = 3
x_eeg = torch.randn(B, 16, 16, 600)
x_fnirs = torch.randn(B, S, 16, 16, 30, 2)

torch.manual_seed(42)
model_default = TSFNet(num_classes=3, fnirs_num_segments=S).eval()
torch.manual_seed(42)
model_explicit_false = TSFNet(num_classes=3, fnirs_num_segments=S, use_fc_features=False).eval()

sd1, sd2 = model_default.state_dict(), model_explicit_false.state_dict()
assert sd1.keys() == sd2.keys(), "use_fc_features=False should introduce no new parameters/buffers"
for k in sd1:
    assert torch.equal(sd1[k], sd2[k]), f"mismatch in {k}"
assert not hasattr(model_default, "connectivity_branch"), \
    "connectivity_branch should not even be constructed when use_fc_features=False"

with torch.no_grad():
    out1 = model_default(x_eeg, x_fnirs)
    out2 = model_explicit_false(x_eeg, x_fnirs)
    # Also confirm passing a (harmless, ignored) x_conn doesn't change anything.
    out3 = model_default(x_eeg, x_fnirs, torch.randn(B, 2, N_PAIRS))

for a, b, c in zip(out1[:4], out2[:4], out3[:4]):
    assert torch.equal(a, b)
    assert torch.equal(a, c)
print("[PASS] check 2: use_fc_features=False (default) is a bit-exact no-op "
      "(no new params, identical output, unaffected by a stray x_conn argument).")

# ---------------------------------------------------------------------------
# Check 3: use_fc_features=True requires x_conn (clear error if missing).
# ---------------------------------------------------------------------------
torch.manual_seed(1)
model_fc = TSFNet(num_classes=3, fnirs_num_segments=S, use_fc_features=True,
                   fc_num_channels=N_CHANNELS, fc_embed_dim=64)
assert hasattr(model_fc, "connectivity_branch")

try:
    model_fc(x_eeg, x_fnirs, x_conn=None)
    raise AssertionError("expected ValueError when use_fc_features=True and x_conn=None")
except ValueError as e:
    assert "use_fc_features=True" in str(e)
    print("[PASS] check 3: use_fc_features=True with x_conn=None raises a clear ValueError "
          "(no silent zero-fill).")

# ---------------------------------------------------------------------------
# Check 4: use_fc_features=True with real x_conn runs, correct shapes, and
# the connectivity embedding actually influences y_fusion/y_pred (gradient
# check on connectivity_branch's own parameters).
# ---------------------------------------------------------------------------
x_conn_batch = torch.randn(B, 2, N_PAIRS)
model_fc.train()
y_eeg, y_fusion, y_fnirs, y_pred, aux = model_fc(x_eeg, x_fnirs, x_conn_batch)
assert y_fusion.shape == (B, 3) and y_pred.shape == (B, 3)
loss = y_pred.sum()
loss.backward()
conn_grad_norm = sum(p.grad.abs().sum().item() for p in model_fc.connectivity_branch.parameters()
                      if p.grad is not None)
assert conn_grad_norm > 0, "connectivity_branch parameters received no gradient from y_pred"
print(f"[PASS] check 4: use_fc_features=True forward/backward runs correctly, "
      f"connectivity_branch gradient norm={conn_grad_norm:.4f} (>0, confirms it's wired in).")

# clf_fusion's input dim should have grown by fc_embed_dim relative to default.
assert model_fc.clf_fusion.net[0].in_features == model_default.clf_fusion.net[0].in_features + 64, \
    "clf_fusion input dim should grow by exactly fc_embed_dim when use_fc_features=True"
print("[PASS] check 4b: clf_fusion input dim grew by exactly fc_embed_dim (64) as expected.")

# ---------------------------------------------------------------------------
# Check 5: NBackDataset-style variable-arity dataset + _unpack_batch.
# ---------------------------------------------------------------------------
class _FakeDataset3(Dataset):
    """Mimics NBackDataset with NO connectivity file (the default/legacy case)."""
    def __len__(self): return 6
    def __getitem__(self, idx):
        return torch.randn(16, 16, 600), torch.randn(S, 16, 16, 30, 2), torch.randint(0, 3, ()).long()


class _FakeDataset4(Dataset):
    """Mimics NBackDataset WITH a connectivity file present."""
    def __len__(self): return 6
    def __getitem__(self, idx):
        return (torch.randn(16, 16, 600), torch.randn(S, 16, 16, 30, 2),
                torch.randint(0, 3, ()).long(), torch.randn(2, N_PAIRS))


loader3 = DataLoader(_FakeDataset3(), batch_size=2)
loader4 = DataLoader(_FakeDataset4(), batch_size=2)

for batch in loader3:
    xe, xf, y, xc = _unpack_batch(batch)
    assert xc is None, "_unpack_batch should return None for a 3-tuple batch"
for batch in loader4:
    xe, xf, y, xc = _unpack_batch(batch)
    assert xc is not None and xc.shape == (2, 2, N_PAIRS)
print("[PASS] check 5: _unpack_batch correctly handles both 3-tuple (connectivity=None) "
      "and 4-tuple (real connectivity) dataset outputs.")

# ---------------------------------------------------------------------------
# Check 6: end-to-end evaluate()/train_one_fold() smoke test with
# use_fc_features=True on tiny synthetic 4-tuple data.
# ---------------------------------------------------------------------------
n_samples = 12
ds = _FakeDataset4()
# Need exactly n_samples items with real (non-random-length) content; reuse
# _FakeDataset4 via a thin wrapper with the right length.
class _FakeDataset4N(Dataset):
    def __init__(self, n): self.n = n
    def __len__(self): return self.n
    def __getitem__(self, idx):
        return (torch.randn(16, 16, 600), torch.randn(S, 16, 16, 30, 2),
                torch.randint(0, 3, (1,)).squeeze().long(), torch.randn(2, N_PAIRS))

loader = DataLoader(_FakeDataset4N(n_samples), batch_size=4)

avg_loss, acc, kappa = evaluate(model_fc, loader, "cpu", 0.2, 0.2)
assert isinstance(avg_loss, float) and avg_loss == avg_loss
print(f"[PASS] check 6a: evaluate() runs end-to-end with use_fc_features=True model "
      f"and 4-tuple batches (avg_loss={avg_loss:.4f}).")

model_fc2 = TSFNet(num_classes=3, fnirs_num_segments=S, use_fc_features=True,
                    fc_num_channels=N_CHANNELS, fc_embed_dim=64)
test_acc, test_kappa, trained_model, report = train_one_fold(
    model_fc2, loader, loader, loader, "cpu",
    stage1_max_epochs=2, stage1_patience=1, stage2_max_epochs=1,
    ckpt_path="/tmp/_e1_test_ckpt.pt", log_fn=lambda *a, **k: None)
assert 0.0 <= test_acc <= 1.0
print(f"[PASS] check 6b: train_one_fold() runs end-to-end with use_fc_features=True "
      f"(test_acc={test_acc:.3f}).")

print("\nALL CHECKS PASSED.")
