"""
test_groupnorm.py -- priority-GN (--norm-mode group), tested against the ACTUAL
patched tsfnet_model_notebook.py AND the ACTUAL pre-patch ORIGINAL file.

Run with NO arguments from the folder that contains:
    tsfnet_model_notebook.py            (patched)
    tsfnet_model_notebook_ORIGINAL.py   (pre-D1 reference)
"""
import os
import sys
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, _HERE)


def _load(fname):
    ns = {}
    exec(open(os.path.join(_HERE, fname), encoding="utf-8").read(), ns)
    return ns


ns = _load("tsfnet_model_notebook.py")
TSFNet, tsfnet_loss = ns["TSFNet"], ns["tsfnet_loss"]
run_subject_cross_session = ns["run_subject_cross_session"]
_orig_path = os.path.join(_HERE, "tsfnet_model_notebook_ORIGINAL.py")
TSFNet_orig = _load("tsfnet_model_notebook_ORIGINAL.py")["TSFNet"] if os.path.exists(_orig_path) else None


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


torch.manual_seed(0)
B = 16
xe = torch.randn(B, 16, 16, 600)
xf = torch.rand(B, 21, 16, 16, 30, 2)
y = torch.randint(0, 3, (B,))

# ---------------------------------------------------------------------------
banner("TEST 1 (rule 5): norm_mode='batch' is BIT-IDENTICAL to the ORIGINAL file")
if TSFNet_orig is None:
    print("  [SKIP] tsfnet_model_notebook_ORIGINAL.py not found beside this script")
else:
    torch.manual_seed(42); mo = TSFNet_orig(num_classes=3, fnirs_temporal_align="egta")
    torch.manual_seed(42); mb = TSFNet(num_classes=3, fnirs_temporal_align="egta", norm_mode="batch")
    so, sb = mo.state_dict(), mb.state_dict()
    assert set(so) == set(sb), (set(so) ^ set(sb))
    diff = [k for k in so if not torch.equal(so[k], sb[k])]
    assert not diff, diff
    mo.eval(); mb.eval()
    with torch.no_grad():
        oo, ob = mo(xe, xf, None, None), mb(xe, xf, None, None)
    for nm, a, b in zip(["y_eeg", "y_fusion", "y_fnirs", "y_pred"], oo[:4], ob[:4]):
        d = (a - b).abs().max().item()
        print(f"  {nm:9} max|diff| = {d:.3e}")
        assert d == 0.0, nm
    print(f"[PASS] test 1: all {len(so)} tensors and all four outputs bit-identical.")

# ---------------------------------------------------------------------------
banner("TEST 2 (rule 6): 'group' shifts NO parameter init; only the norm layers change")
torch.manual_seed(42); m_b = TSFNet(num_classes=3, fnirs_temporal_align="egta", norm_mode="batch")
torch.manual_seed(42); m_g = TSFNet(num_classes=3, fnirs_temporal_align="egta", norm_mode="group")
pb = dict(m_b.named_parameters()); pg = dict(m_g.named_parameters())
assert set(pb) == set(pg), set(pb) ^ set(pg)      # BN affine and GN affine share names
diff = [k for k in pb if not torch.equal(pb[k], pg[k])]
assert not diff, f"init shifted: {diff[:5]}"
n_bn = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in m_g.modules())
n_gn = sum(isinstance(m, nn.GroupNorm) for m in m_g.modules())
n_bn_ref = sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in m_b.modules())
print(f"  parameters: {len(pb)} identical by name AND value")
print(f"  BatchNorm layers: batch-mode={n_bn_ref}  group-mode={n_bn}   GroupNorm layers: {n_gn}")
assert n_bn == 0 and n_gn == n_bn_ref
print(f"  trainable params: batch={sum(p.numel() for p in m_b.parameters())} "
      f"group={sum(p.numel() for p in m_g.parameters())}")
assert sum(p.numel() for p in m_b.parameters()) == sum(p.numel() for p in m_g.parameters())
print("[PASS] test 2: every one of the model's parameters starts from the SAME value; "
      "every BatchNorm became GroupNorm; parameter count identical (not added capacity).")

# ---------------------------------------------------------------------------
banner("TEST 3 (THE POINT OF THIS CHANGE): output is independent of batch composition")
# In TRAIN mode a BatchNorm model's output for sample 0 depends on which other
# samples share its batch. That is the structural defect. GroupNorm removes it.
def train_mode_no_dropout(m):
    """Train mode for normalization layers, but every STOCHASTIC module off.
    Includes nn.MultiheadAttention, whose internal dropout (CAFE layer) draws
    a batch-size-dependent number of random numbers -- that is random noise,
    not a batch-STATISTICS dependency, and must be excluded to isolate the
    effect this test is about (found by module-by-module isolation: MHA alone
    in train mode gave 5.6e-05 even with GroupNorm)."""
    m.train()
    for mod in m.modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
                            nn.MultiheadAttention)):
            mod.eval()
for nm, m in [("batch", m_b), ("group", m_g)]:
    train_mode_no_dropout(m)
    with torch.no_grad():
        full = m(xe, xf, None, None)[3][0]            # sample 0 inside a batch of 16
        small = m(xe[:2], xf[:2], None, None)[3][0]   # sample 0 inside a batch of 2
    d = (full - small).abs().max().item()
    print(f"  {nm:5}: |pred(sample0 | batch=16) - pred(sample0 | batch=2)| = {d:.3e}")
    if nm == "batch":
        d_batch = d
    else:
        d_group = d
# The claim is qualitative: BN > 0, GN == 0. The BN magnitude at RANDOM INIT is
# small (~4e-4) because y_pred is a sigmoid-gated sum of heads; it is not a
# measure of how large the effect is after training, and is not reported as one.
assert d_batch > 1e-6, "sanity: BatchNorm SHOULD depend on batch composition"
assert d_group == 0.0, "GroupNorm must NOT depend on batch composition"
print("[PASS] test 3: with BatchNorm, a sample's training-time output changes with its "
      "batch-mates; with GroupNorm it does not. This is exactly the small-last-batch "
      "defect (no drop_last, batch_size=16) that this change targets.")

# ---------------------------------------------------------------------------
banner("TEST 4: train-mode and eval-mode agree (no running stats to mis-estimate)")
m_g.eval()
with torch.no_grad():
    ev = m_g(xe, xf, None, None)[3]
train_mode_no_dropout(m_g)
with torch.no_grad():
    tr = m_g(xe, xf, None, None)[3]
d = (ev - tr).abs().max().item()
print(f"  group: |eval - train(no dropout)| = {d:.3e}")
assert d < 1e-5
print("[PASS] test 4: GroupNorm behaves identically in train and eval.")

# ---------------------------------------------------------------------------
banner("TEST 5 (rule 2): gradients reach every GroupNorm affine parameter")
m_g.train(); m_g.zero_grad()
out = m_g(xe, xf, None, None)
L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
(L[0] if isinstance(L, tuple) else L).backward()
for n, mod in m_g.named_modules():
    if isinstance(mod, nn.GroupNorm):
        gw = mod.weight.grad.abs().sum().item()
        print(f"  |grad| {n + '.weight':34} = {gw:.3e}   (groups={mod.num_groups}, C={mod.num_channels})")
        assert gw > 0, n
print("[PASS] test 5: every GroupNorm layer is trained.")

# ---------------------------------------------------------------------------
banner("TEST 6: batch of ONE works in train mode (BN degenerates; GN must not)")
train_mode_no_dropout(m_g)
with torch.no_grad():
    o1 = m_g(xe[:1], xf[:1], None, None)[3]
assert torch.isfinite(o1).all()
print(f"  group, batch=1: y_pred finite = True, values = {o1[0].tolist()}")
print("[PASS] test 6")

# ---------------------------------------------------------------------------
banner("TEST 7 (rule 3): guard rails")
try:
    TSFNet(num_classes=3, norm_mode="layer"); raise AssertionError("expected ValueError")
except ValueError as e:
    assert "must be 'batch' or 'group'" in str(e)
for kw, frag in [(dict(adabn=True), "adabn=True cannot be combined"),
                 (dict(tta_mode="tent"), "tta_mode='tent' cannot be combined")]:
    try:
        run_subject_cross_session("/nonexistent", "VP000", norm_mode="group", **kw)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert frag in str(e), str(e)
    print(f"  [PASS] norm_mode='group' + {kw} -> ValueError BEFORE any data loading")
print("[PASS] test 7: invalid mode and both BatchNorm-only companions fail fast.")

print("\nALL GroupNorm TESTS PASSED (real TSFNet class"
      + (", real ORIGINAL file)." if TSFNet_orig else "; ORIGINAL file not found, test 1 skipped)."))
