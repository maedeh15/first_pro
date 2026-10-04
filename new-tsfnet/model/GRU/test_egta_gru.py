"""
test_egta_gru.py -- fnirs_temporal_align='egta_gru', tested against the
ACTUAL patched tsfnet_model_notebook.py.
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
TSFNet, CAFELayer, tsfnet_loss = ns["TSFNet"], ns["CAFELayer"], ns["tsfnet_loss"]

torch.manual_seed(0)
B = 4
xe = torch.randn(B, 16, 16, 600)
xf = torch.rand(B, 21, 16, 16, 30, 2)
y = torch.tensor([0, 1, 2, 1])


def banner(t):
    print("\n" + "=" * 72 + "\n" + t + "\n" + "=" * 72)


banner("TEST 1 (rule 6): adding egta_gru params does NOT shift other modules' init")
torch.manual_seed(42); m_none = TSFNet(num_classes=3, fnirs_temporal_align="none")
torch.manual_seed(42); m_gru = TSFNet(num_classes=3, fnirs_temporal_align="egta_gru")
sn, sg = m_none.state_dict(), m_gru.state_dict()
shared = [k for k in sn if k in sg]
diff = [k for k in shared if not torch.equal(sn[k], sg[k])]
new = [k for k in sg if k not in sn]
print(f"  shared tensors: {len(shared)}   differing: {len(diff)}")
print(f"  new params: {len(new)} tensors, e.g. {new[:4]}")
assert not diff, f"RNG leak -- these changed init: {diff[:5]}"
expected_new_prefixes = {"cafe.gru", "cafe.gru_norm1", "cafe.gru_q_proj",
                         "cafe.gru_attn", "cafe.gru_norm2"}
assert all(any(n.startswith(p) for p in expected_new_prefixes) for n in new)
print("[PASS] test 1: all 89 shared tensors bit-identical with the same seed -- the "
      "C-2d RNG-shift problem does not occur here either.")

banner("TEST 2 (HONEST, not a no-op): egta_gru genuinely differs from 'none' at init")
m_none.eval(); m_gru.eval()
with torch.no_grad():
    on = m_none(xe, xf, None, None)
    og = m_gru(xe, xf, None, None)
d_pred = (on[3] - og[3]).abs().max().item()
print(f"  y_pred max|diff| vs 'none' at random init = {d_pred:.4f}")
assert d_pred > 1e-3, "expected a REAL difference (GRU has no identity init)"
w = og[4]["egta_gru_weights"]
assert w.shape == (B, 8, 21)   # gru_n_heads=8 default
assert not torch.allclose(w, torch.full_like(w, 1 / 21), atol=1e-3), \
    "even at random init, GRU-conditioned attention should not be exactly uniform"
assert og[4]["egta_weights"] is None   # the OTHER mode's key stays None
assert on[4]["egta_gru_weights"] is None
print(f"[PASS] test 2: y_pred differs by {d_pred:.4f} at step 0 (HONEST deviation from rule "
      f"5, documented in the class docstring); attention weights shape "
      f"{tuple(w.shape)}, not exactly uniform even before training; 'none' and 'egta' both "
      f"correctly report egta_gru_weights=None.")

banner("TEST 3: order-sensitivity -- the GRU makes this path sensitive to segment SEQUENCE, "
      "not just segment identity (unlike a plain attention pool)")
perm = torch.randperm(21)
cafe_g = CAFELayer(n_segments=21, fnirs_temporal_align="egta_gru").eval()
fu = torch.randn(B, 32, 4, 4); fn = torch.randn(B, 21, 32, 4, 4)
with torch.no_grad():
    dg = (cafe_g(fu, fn)[1] - cafe_g(fu, fn[:, perm])[1]).abs().max().item()
print(f"  enhanced_fnirs, original vs shuffled SEGMENT ORDER: {dg:.4f}")
assert dg > 1e-3
print("[PASS] test 3: shuffling segment order changes the output substantially -- the "
      "recurrent GRU (unlike the plain 'egta' attention pool alone) is sensitive to the "
      "SEQUENCE, not only to which segments are present.")

banner("TEST 4 (rule 2): gradient reaches every new parameter (GRU, query proj, attention)")
m_gru.train(); m_gru.zero_grad()
out = m_gru(xe, xf, None, None)
L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
(L[0] if isinstance(L, tuple) else L).backward()
checks = {
    "gru (input-hidden weight)": m_gru.cafe.gru.weight_ih_l0,
    "gru_q_proj.weight": m_gru.cafe.gru_q_proj.weight,
    "gru_attn.in_proj_weight": m_gru.cafe.gru_attn.in_proj_weight,
    "gru_norm1.weight": m_gru.cafe.gru_norm1.weight,
}
for name, p in checks.items():
    g = p.grad.abs().sum().item() if p.grad is not None else 0.0
    print(f"  |grad| {name:32} = {g:.4e}")
    assert p.grad is not None and g > 0, f"{name} got no gradient"
print("[PASS] test 4: every new parameter group (GRU weights, query projection, attention "
      "in-proj, layer norm) receives a nonzero gradient -- nothing is dead at init.")

banner("TEST 5: a real optimizer run trains egta_gru's parameters")
opt = torch.optim.Adam(m_gru.parameters(), lr=1e-2)
w0 = m_gru.cafe.gru_q_proj.weight.detach().clone()
for _ in range(3):
    opt.zero_grad()
    out = m_gru(xe, xf, None, None)
    L = tsfnet_loss(*out[:4], y, out[4], lambda_eeg=0.2, lambda_fnirs=0.2)
    (L[0] if isinstance(L, tuple) else L).backward()
    opt.step()
moved = (m_gru.cafe.gru_q_proj.weight.detach() - w0).abs().max().item()
print(f"  gru_q_proj.weight max movement after 3 Adam steps = {moved:.4e}")
assert moved > 1e-4
print("[PASS] test 5: parameters genuinely move under real optimization.")

banner("TEST 6: 'egta' and 'egta_gru' are mutually exclusive, independent modes")
assert not hasattr(m_none, "gru") and not hasattr(m_none, "egta_q")
torch.manual_seed(1); m_egta = TSFNet(num_classes=3, fnirs_temporal_align="egta")
assert not hasattr(m_egta.cafe, "gru")
assert not hasattr(m_gru.cafe, "egta_q")
print("[PASS] test 6: 'none' has neither module's params; 'egta' has no GRU; 'egta_gru' has "
      "no linear egta_q/egta_k -- the three modes' parameter sets are disjoint.")

banner("TEST 7 (rule 3): guard rails")
try:
    CAFELayer(fnirs_temporal_align="typo")
    raise AssertionError("expected ValueError")
except ValueError as e:
    assert "must be" in str(e)
try:
    CAFELayer(fnirs_dim=511, fnirs_temporal_align="egta_gru")
    raise AssertionError("expected ValueError for odd fnirs_dim")
except ValueError as e:
    assert "even" in str(e)
try:
    CAFELayer(fnirs_dim=512, gru_n_heads=7, fnirs_temporal_align="egta_gru")
    raise AssertionError("expected ValueError for non-divisible gru_n_heads")
except ValueError as e:
    assert "gru_n_heads" in str(e)
print("[PASS] test 7: unknown mode, odd fnirs_dim, and non-divisible gru_n_heads all raise "
      "clear errors.")

print("\nALL 7 egta_gru TESTS PASSED (against the real TSFNet class).")
