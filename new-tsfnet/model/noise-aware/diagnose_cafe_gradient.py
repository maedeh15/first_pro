"""
diagnose_cafe_gradient.py -- localizes the ~300x gradient drop found between
cafe's output (enhanced_fusion, |grad|=0.000712) and efgf2's output
(fused2, |grad|=0.0000022) in trace_gradient_chain.py.

THE EXACT FORMULA (CAFELayer.forward, line 704):
    enhanced_fusion = g_fusion * fusion_enhanced_full + (1 - g_fusion) * fusion_flat
    g_fusion = sigmoid(self.gate_fusion), gate_fusion initialized to 0 -> g_fusion=0.5

And UPSTREAM of fusion_flat (TSFNet.forward):
    fusion_pooled_final = fused2.mean(dim=-1)   # GAP over the TIME dimension
    fusion_flat = fusion_pooled_final.reshape(B, -1)

TWO STRUCTURAL (non-bug) candidates this script checks BEFORE concluding
anything is actually wrong:
  (1) The skip-gate split: at g_fusion~0.5, only HALF of dL/d(enhanced_fusion)
      reaches fusion_flat via the skip term -- the other half goes through the
      cross-attention branch (fusion_proj -> attention -> fusion_out_proj),
      which could amplify OR shrink it further depending on that branch's own
      Jacobian.
  (2) The GAP-over-time division: fusion_pooled_final = fused2.mean(dim=-1)
      divides the gradient by T (fused2's time dimension) when backpropagating
      -- a STANDARD, EXPECTED effect of mean-pooling, not a bug. If fused2 has
      T=50 timesteps, this alone predicts roughly a 50x drop.

This script measures BOTH effects directly and checks whether (1)*(2) alone
already accounts for the ~300x drop (in which case the "collapse" is just
ordinary backprop through architecture, not a pathology) or whether there is
an ADDITIONAL, unexplained shrinkage (in which case something inside the
cross-attention/HSIC machinery is the real problem).

USAGE
    %run diagnose_cafe_gradient.py
(edit SUBJECT_DIR below; run in the same kernel as tsfnet_model_notebook.py)
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from tsfnet_model_notebook import TSFNet, _unpack_batch, tsfnet_loss


def load_minibatch_mmap(subject_dir, subject_id, n=16, seed=42):
    """Lightweight replacement for NBackDataset for diagnostics ONLY.

    NBackDataset.__init__ calls plain np.load() on the FULL array before any
    indices slicing happens -- fine normally, but this script only needs `n`
    samples, and if a real training run is ALSO active (consuming most RAM),
    loading the full ~1.6GB array on top of that can raise MemoryError (as it
    did). mmap_mode='r' here means the OS only pages in the `n` rows we
    actually index, instead of materializing the whole file in RAM first.

    This intentionally skips connectivity/channel-space files (x_conn/x_chan
    stay None) since the current accepted config's TSFNet doesn't require
    them (use_fc_features=False, eeg_spatial_mode='native', eeg_band_mode=
    'none', eeg_backbone_mode='native') -- if your config needs them, say so
    and this needs a couple more mmap'd loads, same pattern.
    """
    eeg_mm = np.load(os.path.join(subject_dir, f"{subject_id}_eeg_windows.npy"), mmap_mode="r")
    fnirs_mm = np.load(os.path.join(subject_dir, f"{subject_id}_fnirs_windows.npy"), mmap_mode="r")
    labels_mm = np.load(os.path.join(subject_dir, f"{subject_id}_labels.npy"), mmap_mode="r")

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(labels_mm), size=min(n, len(labels_mm)), replace=False)
    idx.sort()  # sequential access is friendlier to mmap paging than random order

    x_eeg = torch.from_numpy(np.array(eeg_mm[idx])).float()
    x_fnirs = torch.from_numpy(np.array(fnirs_mm[idx])).float()
    y = torch.from_numpy(np.array(labels_mm[idx])).long()
    return x_eeg, x_fnirs, y

# ---- EDIT THIS ----
SUBJECT_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
SUBJECT_ID = "VP001"
FNIRS_NUM_SEGMENTS = 21
BATCH_SIZE = 16
SEED = 42
# --------------------

torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TSFNet(
    num_classes=3, fusion_combination="paper_mean",
    fnirs_num_segments=FNIRS_NUM_SEGMENTS, fnirs_temporal_align="egta",
    efgf_gate_mode="sigmoid", efgf_gate_init=-2.0,
).to(device)
model.train()

x_eeg, x_fnirs, y = load_minibatch_mmap(SUBJECT_DIR, SUBJECT_ID, n=BATCH_SIZE, seed=SEED)
x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
x_conn, x_chan = None, None  # accepted config doesn't use connectivity/channel-space inputs

# ---- Hooks ----
captured = {}

def hook_cafe_output(module, inputs, output):
    # inputs = (fusion_feat, fnirs_feat) to CAFELayer.forward
    captured["cafe_fusion_feat_in"] = inputs[0].detach().clone()  # this IS fusion_flat's source

def hook_cafe_bwd(module, grad_input, grad_output):
    captured["grad_enhanced_fusion"] = grad_output[0].detach().clone()

def hook_fusion_proj_bwd(module, grad_input, grad_output):
    # fusion_proj's INPUT is fusion_flat -- grad_input[0] is dL/d(fusion_flat)
    # arriving via the cross-attention branch ONLY (the branch fusion_proj feeds).
    if grad_input[0] is not None:
        captured["grad_fusion_flat_via_attn"] = grad_input[0].detach().clone()

def hook_efgf2_bwd(module, grad_input, grad_output):
    captured["grad_fused2"] = grad_output[0].detach().clone()

def hook_efgf2_fwd(module, inputs, output):
    fused2 = output[0]
    captured["fused2_shape"] = tuple(fused2.shape)

handles = [
    model.cafe.register_forward_hook(hook_cafe_output),
    model.cafe.register_full_backward_hook(hook_cafe_bwd),
    model.cafe.fusion_proj.register_full_backward_hook(hook_fusion_proj_bwd),
    model.efgf2.register_full_backward_hook(hook_efgf2_bwd),
    model.efgf2.register_forward_hook(hook_efgf2_fwd),
]

model.zero_grad()
y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)
loss.backward()

for h in handles:
    h.remove()

print(f"Loss: {loss.item():.4f}\n")

g_fusion = torch.sigmoid(model.cafe.gate_fusion).item()
g_fnirs = torch.sigmoid(model.cafe.gate_fnirs).item()
print(f"g_fusion = sigmoid(gate_fusion) = {g_fusion:.4f}  (0.5 expected at init, gate_fusion=0)")
print(f"g_fnirs  = sigmoid(gate_fnirs)  = {g_fnirs:.4f}\n")

T = captured["fused2_shape"][-1]
print(f"fused2 shape = {captured['fused2_shape']}  ->  T (time dim being GAP-pooled) = {T}\n")

grad_enhanced = captured["grad_enhanced_fusion"]
grad_enhanced_mag = grad_enhanced.abs().mean().item()
print(f"[measured]   |dL/d(enhanced_fusion)|       = {grad_enhanced_mag:.6g}  (from trace_gradient_chain.py's 'cafe output' row)")

# Skip-path contribution: (1 - g_fusion) * grad(enhanced_fusion), elementwise exact
grad_fusion_flat_via_skip = (1 - g_fusion) * grad_enhanced
skip_mag = grad_fusion_flat_via_skip.abs().mean().item()
print(f"[predicted]  |dL/d(fusion_flat)| via SKIP path  = (1-g_fusion) * above = {skip_mag:.6g}")

attn_grad = captured.get("grad_fusion_flat_via_attn")
if attn_grad is not None:
    attn_mag = attn_grad.abs().mean().item()
    print(f"[measured]   |dL/d(fusion_flat)| via ATTENTION path (fusion_proj backward) = {attn_mag:.6g}")
else:
    attn_mag = None
    print("[measured]   ATTENTION path gradient: hook did not fire (check fusion_proj wiring)")

# Total gradient reaching fusion_flat = sum of both paths (they add, since
# fusion_flat feeds both the skip term AND fusion_tok -> attention).
# NOTE: these are different elements' average magnitudes, not a single
# tensor sum, so we report both the naive sum-of-magnitudes (upper bound,
# ignores sign cancellation) as context.
if attn_mag is not None:
    naive_total = skip_mag + attn_mag
    print(f"[predicted]  naive upper-bound total |dL/d(fusion_flat)| (sum of both paths) = {naive_total:.6g}")
    predicted_fused2_grad = naive_total / T
    print(f"\n[predicted]  |dL/d(fused2)| if GAP-over-T(={T}) division is the ONLY extra factor "
          f"= {naive_total:.6g} / {T} = {predicted_fused2_grad:.6g}")

actual_fused2_grad = captured["grad_fused2"].abs().mean().item()
print(f"[measured]   |dL/d(fused2)| (actual, from efgf2 backward hook) = {actual_fused2_grad:.6g}")

if attn_mag is not None:
    ratio = actual_fused2_grad / predicted_fused2_grad if predicted_fused2_grad > 0 else float("nan")
    print(f"\n===== VERDICT =====")
    print(f"actual / predicted-from-known-structural-factors = {ratio:.3f}")
    if 0.1 <= ratio <= 10:
        print("Within an order of magnitude: the ~300x drop is MOSTLY explained by the")
        print("skip-gate split (g_fusion~0.5) and the GAP-over-time mean division (T="
              f"{T}) -- i.e. ORDINARY backprop through this architecture's pooling and")
        print("gating, NOT a separate pathology inside cross-attention/HSIC. The weak")
        print("gradient reaching efgf1/efgf2 is then an architectural property (deep,")
        print("pooling-heavy path from the loss) worth noting as a limitation, but not a")
        print("bug to 'fix' before attempting item #2.")
    else:
        print("Off by MORE than an order of magnitude: the known structural factors (skip")
        print("gate + GAP-over-time) do NOT fully explain the drop -- something ELSE inside")
        print("the cross-attention/HSIC computation, or a path this script didn't account")
        print("for, is shrinking the gradient further. Worth hooking cross_attn_fusion_to_fnirs")
        print("and fusion_out_proj individually next to isolate it.")
