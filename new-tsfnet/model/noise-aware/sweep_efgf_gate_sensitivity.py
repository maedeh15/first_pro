"""
sweep_efgf_gate_sensitivity.py -- answers the ACTUAL question behind item #2
("is it worth fixing the weak gradient on EFGF's gamma/alpha?") directly and
cheaply: on an ALREADY-TRAINED checkpoint, manually force each of the four
scalars (efgf1.gamma, efgf1.alpha, efgf2.gamma, efgf2.alpha) across its full
[0,1] range ONE AT A TIME (holding the other three at their trained/converged
values), and measure held-out accuracy at each forced value. No training, no
gradient, no optimizer -- pure forward-pass inference, so this takes seconds
per sweep point.

WHY "ONE AT A TIME" (isolated-variable, per the project's own engineering
convention): forcing all four simultaneously would confound which scalar's
value is responsible for any accuracy change observed.

HOW THE OVERRIDE WORKS
gate_mode="sigmoid" means gamma = sigmoid(gamma_raw). To force gamma to a
target value v in (0,1), we set gamma_raw.data = logit(v) = log(v/(1-v))
directly (no backward pass involved -- this is a plain tensor write, not a
training step). v is swept over a fine grid EXCLUDING exact 0/1 (logit is
infinite there); we approximate v=0 and v=1 with raw=-15/+15
(sigmoid(+-15) is already closer to 0/1 than float32 can distinguish from
the true endpoint).

READING THE RESULT
  - FLAT curve (accuracy barely moves across the whole sweep) for a scalar:
    confirms that scalar's weak gradient cost NOTHING -- the model's
    predictions are genuinely insensitive to it, so fixing the gradient
    (item #2) would have had nothing to gain. Settle this definitively.
  - PEAKED curve (accuracy clearly better at some v than at the trained
    value): the model is NOT at this scalar's optimum, and a working
    gradient (if it existed) could plausibly have found a better value --
    THIS is the one outcome that would justify revisiting item #2, now with
    concrete evidence of how much headroom exists (the gap between the
    trained-value accuracy and the sweep's peak).

USAGE
    %run sweep_efgf_gate_sensitivity.py
(edit CHECKPOINT_PATH / SUBJECT_DIR / SUBJECT_ID below first)
"""
import os
import re
import numpy as np
import torch

from tsfnet_model_notebook import TSFNet

# ---- EDIT THESE ----
CHECKPOINT_PATH = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\EGTA\6\results_40b7d8ea53_seed42\VP001_session1_best.pt"
SUBJECT_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\VP001"
SUBJECT_ID = "VP001"
FNIRS_NUM_SEGMENTS = 21
BATCH_SIZE = 32
# ----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Parse held-out session from filename (same bugfix as evaluate_noise_robustness.py) ----
ckpt_name = os.path.basename(CHECKPOINT_PATH)
m = re.search(r"_session(\d+)_best\.pt$", ckpt_name)
if not m:
    raise ValueError(f"Cannot parse held-out session from checkpoint filename: {ckpt_name}")
TEST_SESSION = int(m.group(1))
print(f"Held-out session for this checkpoint: {TEST_SESSION}")

# ---- Load held-out session's data with mmap (avoid loading the whole subject
#      into RAM if a real training run is concurrently active -- same fix as
#      diagnose_cafe_gradient.py needed) ----
session_ids = np.load(os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_session_ids.npy"))
idx = np.where(session_ids == TEST_SESSION)[0]
print(f"Held-out samples: {len(idx)}")

eeg_mm = np.load(os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_eeg_windows.npy"), mmap_mode="r")
fnirs_mm = np.load(os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_fnirs_windows.npy"), mmap_mode="r")
labels_mm = np.load(os.path.join(SUBJECT_DIR, f"{SUBJECT_ID}_labels.npy"), mmap_mode="r")

x_eeg_all = torch.from_numpy(np.array(eeg_mm[idx])).float()
x_fnirs_all = torch.from_numpy(np.array(fnirs_mm[idx])).float()
y_all = torch.from_numpy(np.array(labels_mm[idx])).long()

# ---- Build model with the EXACT accepted-config architecture flags and load checkpoint ----
model = TSFNet(
    num_classes=3, fusion_combination="paper_mean",
    fnirs_num_segments=FNIRS_NUM_SEGMENTS, fnirs_temporal_align="egta",
    efgf_gate_mode="sigmoid",
).to(DEVICE)
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
model.load_state_dict(sd, strict=True)
model.eval()

# ---- Record the TRAINED (converged) values, to restore after each sweep and
#      to mark them on the printed curve for reference ----
TRAINED_RAW = {
    "efgf1.gamma_raw": model.efgf1.gamma_raw.data.clone(),
    "efgf1.alpha_raw": model.efgf1.alpha_raw.data.clone(),
    "efgf2.gamma_raw": model.efgf2.gamma_raw.data.clone(),
    "efgf2.alpha_raw": model.efgf2.alpha_raw.data.clone(),
}
TRAINED_V = {k: torch.sigmoid(v).item() for k, v in TRAINED_RAW.items()}


def get_param(name):
    layer, field = name.split(".")
    return getattr(getattr(model, layer), field)


def logit(v, eps=1e-6):
    v = min(max(v, eps), 1 - eps)
    return float(np.log(v / (1 - v)))


@torch.no_grad()
def evaluate_accuracy():
    correct, total = 0, 0
    for i in range(0, len(y_all), BATCH_SIZE):
        xb_eeg = x_eeg_all[i:i + BATCH_SIZE].to(DEVICE)
        xb_fnirs = x_fnirs_all[i:i + BATCH_SIZE].to(DEVICE)
        yb = y_all[i:i + BATCH_SIZE].to(DEVICE)
        _, _, _, y_pred, _ = model(xb_eeg, xb_fnirs, None, None)
        pred_class = y_pred.argmax(dim=-1)
        correct += (pred_class == yb).sum().item()
        total += yb.shape[0]
    return correct / total


# Sweep grid: fine resolution, endpoints approximated (see module docstring).
SWEEP_V = [0.0] + list(np.round(np.linspace(0.05, 0.95, 19), 3)) + [1.0]

print(f"\nBaseline (all 4 params at their TRAINED values): "
      f"acc = {evaluate_accuracy():.4f}")
for name, v in TRAINED_V.items():
    print(f"  trained {name}: {v:.4f}")

results = {}
for sweep_name in ("efgf1.gamma_raw", "efgf1.alpha_raw", "efgf2.gamma_raw", "efgf2.alpha_raw"):
    print(f"\n===== Sweeping {sweep_name} (other 3 held at trained values) =====")
    param = get_param(sweep_name)
    curve = []
    for v in SWEEP_V:
        raw_v = logit(v) if 0.0 < v < 1.0 else (-15.0 if v == 0.0 else 15.0)
        param.data.fill_(raw_v)
        acc = evaluate_accuracy()
        curve.append((v, acc))
        marker = "  <-- closest to trained value" if abs(v - TRAINED_V[sweep_name]) < 0.03 else ""
        print(f"  {sweep_name} forced to {v:.2f}: acc = {acc:.4f}{marker}")
    param.data.copy_(TRAINED_RAW[sweep_name])   # restore before moving to the next sweep
    results[sweep_name] = curve

print("\n===== SUMMARY: sensitivity range per parameter =====")
for name, curve in results.items():
    accs = [a for _, a in curve]
    v_at_best = curve[int(np.argmax(accs))][0]
    print(f"  {name}: acc range [{min(accs):.4f}, {max(accs):.4f}]  "
          f"(spread = {max(accs)-min(accs):.4f}), best at forced value {v_at_best:.2f}, "
          f"trained value was {TRAINED_V[name]:.4f}")

print("\n===== READING GUIDE =====")
print("spread < ~0.02-0.03 (2-3 accuracy points) across the WHOLE [0,1] range for a")
print("parameter: the model is genuinely insensitive to it -- confirms item #2 would")
print("have bought nothing for that scalar, closing the question for good.")
print("spread clearly larger, with the trained value NOT near the peak: there was real")
print("headroom the weak gradient left on the table -- worth reconsidering a fix, now")
print("with a concrete accuracy-gap number to justify the GPU time.")
print("\nNote: this is a SINGLE checkpoint (VP001, session {}). A real decision needs".format(TEST_SESSION))
print("this repeated across at least a handful of subjects/seeds before generalizing --")
print("this script answers 'is it worth investigating further', not 'is it accepted'.")
