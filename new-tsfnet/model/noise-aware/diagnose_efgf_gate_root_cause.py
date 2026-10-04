"""
diagnose_efgf_gate_root_cause.py -- WHY is grad_norm on efgf1/efgf2's
gamma_raw/alpha_raw weak (confirmed in diagnose_efgf_gate_gradient.py:
efgf1.gamma_raw was ~100x below the model's p10)? This decomposes the
gradient analytically from EFGFLayer.forward's actual formula, instead of
guessing, so the answer is a number, not a theory.

THE EXACT FORMULA (from EFGFLayer.forward, tsfnet_model_notebook.py line 380+):
    phi   = alpha * fnirs_spatial_att + (1 - alpha) * eeg_temporal_att
    fused = gamma * eeg_map + (1 - gamma) * fusion_map + phi * fusion_map

Chain rule gives, EXACTLY (not approximately):
    dL/dgamma_raw = sigmoid'(gamma_raw) * sum[ dL/dfused * (eeg_map - fusion_map) ]
    dL/dalpha_raw = sigmoid'(alpha_raw) * sum[ dL/dphi   * (fnirs_spatial_att - eeg_temporal_att) ]

So grad_norm on gamma_raw can ONLY be small for one of three reasons, and
this script measures all three SEPARATELY so we know which one it is instead
of guessing:
    (A) sigmoid'(gamma_raw) is tiny               -- parametrization/init issue
    (B) (eeg_map - fusion_map) is tiny everywhere  -- the two branches already
        produce near-identical features at this stage (a DATA/architecture
        fact, not a bug)
    (C) dL/dfused (the upstream gradient reaching this layer's output) is
        tiny                                       -- a genuine dead/vanishing
        upstream gradient (a real problem, possibly connected to the
        project's previously-documented C-2b auxiliary-gradient bug)

We verify the formula itself against real autograd (not just trust the
derivation) via full backward hooks on efgf1/efgf2, then recombine (A)*(B)
summed against (C) and check it equals the REAL .grad on gamma_raw/alpha_raw
to float precision -- if it doesn't match, something else entirely is going
on (e.g. an unexpected second gradient path) and that mismatch itself is the
finding.

USAGE
    %run diagnose_efgf_gate_root_cause.py
(edit SUBJECT_DIR below; run in the same kernel as tsfnet_model_notebook.py)
"""
import torch
from torch.utils.data import DataLoader

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, tsfnet_loss

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

ds = NBackDataset(SUBJECT_DIR, SUBJECT_ID)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                     generator=torch.Generator().manual_seed(SEED))
batch = next(iter(loader))
x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
x_conn = x_conn.to(device) if x_conn is not None else None
x_chan = x_chan.to(device) if x_chan is not None else None

# ---- Capture EFGFLayer forward INPUTS (eeg_map, fnirs_map, fusion_map) ----
captured_inputs = {}
def make_fwd_hook(name):
    def hook(module, inputs, output):
        eeg_map, fnirs_map, fusion_map = inputs
        captured_inputs[name] = {
            "eeg_map": eeg_map.detach().clone(),
            "fusion_map": fusion_map.detach().clone(),
        }
    return hook

# ---- Capture gradient reaching the EFGFLayer's OUTPUT (dL/dfused) ----
captured_grad_output = {}
def make_bwd_hook(name):
    def hook(module, grad_input, grad_output):
        # output is a tuple (fused, eeg_pooled_for_pcc, phi_for_pcc); we want
        # the gradient on `fused` specifically, which is grad_output[0].
        captured_grad_output[name] = grad_output[0].detach().clone()
    return hook

handles = []
for name, layer in (("efgf1", model.efgf1), ("efgf2", model.efgf2)):
    handles.append(layer.register_forward_hook(make_fwd_hook(name)))
    handles.append(layer.register_full_backward_hook(make_bwd_hook(name)))

model.zero_grad()
y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)
loss.backward()

for h in handles:
    h.remove()

print(f"Loss: {loss.item():.4f}\n")

for name, layer in (("efgf1", model.efgf1), ("efgf2", model.efgf2)):
    print(f"===== {name} =====")
    eeg_map = captured_inputs[name]["eeg_map"]
    fusion_map = captured_inputs[name]["fusion_map"]
    grad_out = captured_grad_output[name]   # dL/dfused

    # --- Factor B: branch similarity (eeg_map vs fusion_map) ---
    diff = eeg_map - fusion_map
    diff_mag = diff.abs().mean().item()
    eeg_mag = eeg_map.abs().mean().item()
    fusion_mag = fusion_map.abs().mean().item()
    rel_diff = diff_mag / (eeg_mag + fusion_mag + 1e-12)
    print(f"  [Factor B] |eeg_map - fusion_map| mean = {diff_mag:.6g}  "
          f"(|eeg_map| mean = {eeg_mag:.6g}, |fusion_map| mean = {fusion_mag:.6g})")
    print(f"             relative branch difference = {rel_diff:.4%}  "
          f"({'NEAR-IDENTICAL branches -> this alone explains a weak gamma gradient' if rel_diff < 0.02 else 'branches are meaningfully different'})")

    # --- Factor C: upstream gradient magnitude reaching this layer's output ---
    grad_out_mag = grad_out.abs().mean().item()
    print(f"  [Factor C] |dL/dfused| mean (upstream gradient reaching this layer) = {grad_out_mag:.6g}")

    # --- Factor A: sigmoid derivative at current gamma_raw ---
    raw = layer.gamma_raw.detach()
    gamma = torch.sigmoid(raw)
    sig_prime = (gamma * (1 - gamma)).item()
    print(f"  [Factor A] gamma_raw={raw.item():.4f} -> gamma={gamma.item():.4f}, "
          f"sigmoid'(gamma_raw) = {sig_prime:.4f}  "
          f"({'this IS a bottleneck' if sig_prime < 0.01 else 'not a bottleneck (not saturated)'})")

    # --- Hand-derived prediction vs. REAL autograd .grad (sanity check) ---
    predicted_grad = sig_prime * (grad_out * diff).sum().item()
    actual_grad = layer.gamma_raw.grad.item()
    match = abs(predicted_grad - actual_grad) < 1e-4 * (abs(actual_grad) + 1e-8)
    print(f"  [Verify]  hand-derived dL/dgamma_raw = {predicted_grad:.6g}  |  "
          f"actual autograd .grad = {actual_grad:.6g}  |  "
          f"{'MATCH (formula confirmed, no hidden gradient path)' if match else 'MISMATCH -- something else is going on, investigate before trusting the factors above'}")
    print()

print("===== CONCLUSION GUIDE =====")
print("Whichever factor (A/B/C) is small is the actual bottleneck for that layer's")
print("gamma gradient. If Factor B (branch similarity) is the culprit, this is a")
print("property of what EEG/fusion branches have learned to represent at this network")
print("depth, NOT a plumbing bug -- the practical fix is architectural (e.g. encourage")
print("branch divergence) or accept it as a documented limitation, not a gate-mode fix.")
print("If Factor C is tiny for BOTH efgf1 and efgf2, that points to a genuine vanishing-")
print("gradient problem somewhere downstream, worth comparing against the project's")
print("earlier C-2b auxiliary-gradient finding.")
print("\n(The same three factors apply to alpha_raw via fnirs_spatial_att/eeg_temporal_att")
print("and dL/dphi -- not printed here to keep the output readable; ask if you want that")
print("breakdown too, it's the same hook data, just a different few lines of arithmetic.)")
