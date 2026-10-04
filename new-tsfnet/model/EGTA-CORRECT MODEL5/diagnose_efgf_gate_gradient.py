"""
diagnose_efgf_gate_gradient.py -- measures the REAL data gradient magnitude on
efgf1/efgf2's gamma_raw/alpha_raw, on one real batch of VP001, BEFORE deciding
whether item #2 (feature-dependent gamma/alpha) is worth implementing.

WHY A SINGLE BATCH IS ENOUGH FOR THIS QUESTION (and why no optimizer.step() here)
We are not trying to reproduce training dynamics -- we're asking one narrow
question: "when there IS a data gradient reaching gamma_raw/alpha_raw, how big
is it, relative to gradients reaching other parameters in the same model?"
.backward() alone gives us that number with zero weight_decay contamination
(weight_decay only enters via optimizer.step(), which we never call here) --
exactly the clean reading the project's own weight_decay-drift finding needed
and didn't have at the time.

This uses a FRESH, randomly-initialized model (gate_init=-2.0, same as your
accepted config), NOT a loaded checkpoint -- on purpose. A loaded checkpoint's
gradient would already reflect wherever training left it (possibly already
in a low-gradient region); a fresh model's gradient tells you what signal was
available from the very first step, which is the more basic diagnostic. If you
also want the SAME question answered at a late-training point, say so and I'll
add a loaded-checkpoint variant -- it's a one-line change (load_state_dict
instead of fresh TSFNet(...)), but is a different and complementary question.

USAGE
    %run diagnose_efgf_gate_gradient.py
(edit SUBJECT_DIR below first; run in the same kernel as tsfnet_model_notebook.py)
"""
import torch
import torch.nn.functional as F
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

# Fresh model, SAME architecture/init flags as the accepted config.
model = TSFNet(
    num_classes=3,
    fusion_combination="paper_mean",
    fnirs_num_segments=FNIRS_NUM_SEGMENTS,
    fnirs_temporal_align="egta",
    efgf_gate_mode="sigmoid",
    efgf_gate_init=-2.0,
).to(device)
model.train()

ds = NBackDataset(SUBJECT_DIR, SUBJECT_ID)   # all sessions; we just need ONE real batch
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
batch = next(iter(loader))
x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
x_conn = x_conn.to(device) if x_conn is not None else None
x_chan = x_chan.to(device) if x_chan is not None else None

print(f"Batch: x_eeg={tuple(x_eeg.shape)} x_fnirs={tuple(x_fnirs.shape)} y={tuple(y.shape)}")

# One clean forward + backward. NO optimizer, NO optimizer.step() -- so
# weight_decay never enters the picture. model.zero_grad() first in case
# TSFNet's __init__ left any stray .grad from buffer construction (defensive,
# should be a no-op on a fresh model).
model.zero_grad()
y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux,
                       normalize_ypred=False)  # match accepted config's actual flag if different
loss.backward()

print(f"\nLoss (fresh init, one batch): {loss.item():.4f}\n")

# ---- Per-parameter gradient norms: EFGF gates vs. everything else ----
efgf_names = {"efgf1.gamma_raw", "efgf1.alpha_raw", "efgf2.gamma_raw", "efgf2.alpha_raw"}
efgf_grads, other_grad_norms = {}, []
for name, p in model.named_parameters():
    if p.grad is None:
        continue
    g = p.grad.detach().abs().norm().item()
    if name in efgf_names:
        efgf_grads[name] = g
    else:
        other_grad_norms.append(g)

print("===== EFGF gate gradients (THE question) =====")
for name in sorted(efgf_names):
    val = efgf_grads.get(name)
    print(f"  {name:>20}: grad_norm = {val if val is not None else 'NO GRADIENT (None) -- dead path!'}")

import numpy as np
other = np.array(other_grad_norms)
print(f"\n===== Context: gradient norms on the OTHER {len(other)} parameters =====")
print(f"  median = {np.median(other):.6g}, mean = {other.mean():.6g}, "
      f"p10 = {np.percentile(other,10):.6g}, p90 = {np.percentile(other,90):.6g}")

print("\n===== READING GUIDE =====")
print("If an efgf gate's grad_norm falls BELOW the p10 of 'other' parameters, its")
print("gradient signal is genuinely weak relative to the rest of the model -- a")
print("feature-dependent replacement (gamma_fc) would likely inherit the same weak")
print("signal through the same upstream path, and may not move much either.")
print("If it falls within the typical range of 'other' parameters, the near-init")
print("convergence seen in report_efgf_gate_values.py is more likely a LOSS-LANDSCAPE")
print("preference (the data genuinely wants gamma/alpha near that value) rather than")
print("a plumbing problem -- in which case an adaptive map COULD still learn a more")
print("refined, per-sample version of that same preference.")
