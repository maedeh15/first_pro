"""
test_egta.py -- fnirs_temporal_align='egta' (roadmap priority-1), tested
against the ACTUAL patched tsfnet_model_notebook.py, covering the project's
engineering rules (see قواعد_مهندسی_پروژه.md).
"""
import sys
import torch
import torch.nn.functional as F

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
TSFNet, CAFELayer, tsfnet_loss = ns["TSFNet"], ns["CAFELayer"], ns["tsfnet_loss"]

torch.manual_seed(0)
B = 4
xe = torch.randn(B, 16, 16, 600)
xf = torch.rand(B, 21, 16, 16, 30, 2)
y = torch.tensor([0, 1, 2, 1])


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


banner("TEST 1 (rule 6): adding EGTA does NOT shift any other module's init")
torch.manual_seed(42); m_none = TSFNet(num_classes=3, fnirs_temporal_align="none")
torch.manual_seed(42); m_egta = TSFNet(num_classes=3, fnirs_temporal_align="egta")
sn, se = m_none.state_dict(), m_egta.state_dict()
shared = [k for k in sn if k in se]
diff = [k for k in shared if not torch.equal(sn[k], se[k])]
new = [k for k in se if k not in sn]
print(f"  shared tensors: {len(shared)}   differing: {len(diff)}   new: {new}")
assert not diff, f"RNG leak -- these changed init: {diff[:5]}"
assert set(new) == {"cafe.egta_q.weight", "cafe.egta_q.bias",
                    "cafe.egta_k.weight", "cafe.egta_k.bias"}
print("[PASS] test 1: every one of the shared tensors is bit-identical with the same "
      "seed -- the C-2d RNG-shift problem does not occur here.")

banner("TEST 2 (rule 5): 'egta' is a no-op at step 0 (uniform weights == mean)")
m_none.eval(); m_egta.eval()
with torch.no_grad():
    on = m_none(xe, xf, None, None)
    oe = m_egta(xe, xf, None, None)
for name, a, b in zip(["y_eeg", "y_fusion", "y_fnirs", "y_pred"], on[:4], oe[:4]):
    d = (a - b).abs().max().item()
    print(f"  {name:9} max|diff| = {d:.2e}")
    assert d < 1e-6, name
w = oe[4]["egta_weights"]
assert w.shape == (B, 4, 21)
assert torch.allclose(w, torch.full_like(w, 1 / 21), atol=1e-7)
assert on[4]["egta_weights"] is None
print("[PASS] test 2: all four outputs match 'none' to <1e-6; attention weights are "
      "exactly 1/21 everywhere at init; 'none' exposes egta_weights=None.")

banner("TEST 3: 'none' fNIRS path is order-INVARIANT; trained 'egta' is order-SENSITIVE")
perm = torch.randperm(21)
cafe_n = CAFELayer(n_segments=21, fnirs_temporal_align="none").eval()
cafe_e = CAFELayer(n_segments=21, fnirs_temporal_align="egta").eval()
with torch.no_grad():
    cafe_e.egta_k.weight.normal_(0, 0.5)   # simulate a trained, non-zero key projection
fu = torch.randn(B, 32, 4, 4); fn = torch.randn(B, 21, 32, 4, 4)
with torch.no_grad():
    dn = (cafe_n(fu, fn)[1] - cafe_n(fu, fn[:, perm])[1]).abs().max().item()
    de = (cafe_e(fu, fn)[1] - cafe_e(fu, fn[:, perm])[1]).abs().max().item()
print(f"  'none' enhanced_fnirs, original vs shuffled segments: {dn:.2e}")
print(f"  'egta' enhanced_fnirs, original vs shuffled segments: {de:.2e}")
assert dn < 1e-6 and de > 1e-4
print("[PASS] test 3: the existing path is temporally blind (a bag of segments); EGTA, "
      "once its keys are non-zero, uses segment POSITION -- the property it exists to add.")

banner("TEST 4 (rule 2): gradient reaches the zero-initialized keys (no deadlock)")
m_egta.train(); m_egta.zero_grad()
out = m_egta(xe, xf, None, None)
L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
(L[0] if isinstance(L, tuple) else L).backward()
gk = m_egta.cafe.egta_k.weight.grad.abs().sum().item()
gq = m_egta.cafe.egta_q.weight.grad.abs().sum().item()
print(f"  |grad| egta_k.weight = {gk:.3e}   |grad| egta_q.weight = {gq:.3e}")
assert gk > 0, "keys got no gradient -- training would be stuck at the uniform mean"
print("[PASS] test 4: the zero-initialized key projection receives a nonzero gradient, "
      "so the model CAN learn to move away from the uniform mean. (egta_q is exactly 0 at "
      "step 0 as expected -- its gradient is proportional to the zero keys -- and becomes "
      "nonzero once egta_k moves; verified in test 5.)")

banner("TEST 5: a real optimizer run actually moves the weights away from uniform")
opt = torch.optim.Adam(m_egta.parameters(), lr=1e-2)
for _ in range(5):
    opt.zero_grad()
    out = m_egta(xe, xf, None, None)
    L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
    (L[0] if isinstance(L, tuple) else L).backward()
    opt.step()
m_egta.eval()
with torch.no_grad():
    w = m_egta(xe, xf, None, None)[4]["egta_weights"]
dev = (w - 1 / 21).abs().max().item()
gq_after = m_egta.cafe.egta_q.weight.grad.abs().sum().item()
print(f"  max deviation of attention weights from uniform after 5 steps: {dev:.2e}")
print(f"  |grad| egta_q.weight on the last step: {gq_after:.3e}")
assert dev > 1e-5 and gq_after > 0
print("[PASS] test 5: after real Adam steps the segment weights are no longer uniform and "
      "the query projection is now also training.")

banner("TEST 6 (rule 7): PCC regularizer now sees the aligned fNIRS feature")
m_egta.train()
out = m_egta(xe, xf, None, None)
assert out[4]["enhanced_fnirs"].shape == (B, 512)
assert out[4]["enhanced_fusion"].shape == (B, 512)
print("[PASS] test 6: aux['enhanced_fnirs'] keeps shape (B,512), so cafe_regularization_loss "
      "(the project's existing PCC term -- STA-Net's `1 - pearson_r(q_eeg, fnirs_weighted)`) "
      "and every downstream consumer work unchanged.")

banner("TEST 7 (rule 3): guard rails")
for kw, msg in [(dict(fnirs_temporal_align="gru"), "must be 'none' or 'egta'")]:
    try:
        CAFELayer(**kw); raise AssertionError("expected ValueError")
    except ValueError as e:
        assert msg in str(e)
try:
    CAFELayer(fnirs_dim=510, n_heads=4, fnirs_temporal_align="egta")
    raise AssertionError("expected ValueError for non-divisible dims")
except ValueError as e:
    assert "divisible" in str(e)
print("[PASS] test 7: an unknown mode and a head count that does not divide the feature "
      "dims both raise clear errors.")

print("\nALL 7 EGTA TESTS PASSED (against the real TSFNet class).")
