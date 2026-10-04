"""
trace_gradient_chain.py -- finds WHERE along the path
    y_pred <- y_fusion <- clf_fusion <- enhanced_fusion <- cafe <- fusion_pooled_final
            <- fused2 <- efgf2 <- fus_s2 <- fusion_branch.conv2 <- fused1 <- efgf1
the gradient magnitude collapses from "normal" to the ~1e-6 level measured at
efgf1/efgf2's output in diagnose_efgf_gate_root_cause.py.

METHOD: register a forward hook (to timestamp the module in topological order)
and a full backward hook (to capture |grad_output| reaching that module's
output) on each module boundary along this exact path -- using REAL module
boundaries already in TSFNet, not inserted retain_grad() calls, so this
cannot accidentally change the computation graph.

Printed in FORWARD order (input side first) but the gradient NUMBERS are from
the BACKWARD pass -- read the ratio between consecutive rows to find the
layer where the drop actually happens, rather than assuming it is uniform.

USAGE
    %run trace_gradient_chain.py
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

# Real module boundaries along the fusion path, in FORWARD order (input side
# first). fus_s2's producer is fusion_branch.conv2; its hook's grad_output IS
# dL/d(fus_s2), i.e. the gradient arriving at efgf2's SECOND input -- a
# slightly different quantity than efgf1/efgf2's own outputs (which we already
# have from the previous script), included here to localize the conv2 step
# specifically.
CHAIN = [
    ("efgf1 output (fused1)",            model.efgf1),
    ("fusion_branch.conv2 output (fus_s2)", model.fusion_branch.conv2),
    ("efgf2 output (fused2)",            model.efgf2),
    ("cafe output (enhanced_fusion)",    model.cafe),
    ("clf_fusion output (y_fusion)",     model.clf_fusion),
]

captured = {}
handles = []
def make_bwd_hook(name):
    def hook(module, grad_input, grad_output):
        g = grad_output[0] if isinstance(grad_output, tuple) else grad_output
        if g is None:
            captured[name] = None
        else:
            captured[name] = g.detach().abs().mean().item()
    return hook

for name, module in CHAIN:
    handles.append(module.register_full_backward_hook(make_bwd_hook(name)))

model.zero_grad()
y_eeg, y_fnirs, y_fusion, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)
loss, _ = tsfnet_loss(y_eeg, y_fusion, y_fnirs, y_pred, y, aux, normalize_ypred=False)

# Also track the two outputs that aren't module outputs (y_pred IS the model's
# final output, so retain_grad isn't needed -- it already gets one implicitly
# via loss.backward(); we grab it from y_pred.grad after backward instead).
y_pred.retain_grad()
y_fusion.retain_grad()

loss.backward()

for h in handles:
    h.remove()

print(f"Loss: {loss.item():.4f}\n")
print(f"{'stage':<42} {'|grad| mean':>14}")
print("-" * 58)
print(f"{'y_pred (model final output)':<42} {y_pred.grad.detach().abs().mean().item():>14.6g}")
print(f"{'y_fusion (direct tensor, via retain_grad)':<42} {y_fusion.grad.detach().abs().mean().item():>14.6g}")
for name, _ in CHAIN:
    val = captured.get(name)
    shown = f"{val:.6g}" if val is not None else "None (no gradient reached this module!)"
    print(f"{name:<42} {shown:>14}")

print("\n===== HOW TO READ THIS =====")
print("Numbers are |dL/d(output)| at each stage, in topological (forward) order.")
print("A roughly CONSTANT or slowly-shrinking sequence is normal backprop decay.")
print("Find the row where the value drops by orders of magnitude (not just 2-3x)")
print("compared to the row above it -- THAT module's backward computation is where")
print("the gradient actually collapses. Compare that module's operations (pooling,")
print("averaging, a bottleneck dimension, a detach, a loss-weighting constant like")
print("lambda_eeg/lambda_fnirs if this stage is where branch losses combine) against")
print("the surrounding rows to form a concrete hypothesis BEFORE changing any code.")
