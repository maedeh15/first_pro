"""
report_efgf_gate_values.py -- V-2: report the CONVERGED gamma/alpha scalar values
of both EFGF layers, across all checkpoints of the accepted configuration
(tag 40b7d8ea53, all 3 seeds). Pure read, no GPU compute needed beyond a
state_dict load.

WHY THIS MUST RUN BEFORE WRITING THE ADAPTIVE GAMMA/ALPHA CODE
If gamma/alpha are already saturated near 0 or 1 for every checkpoint, the
fixed scalar was never a meaningful free parameter in practice -- replacing
it with a feature-dependent map would be "fixing" something that wasn't
doing much work to begin with, and any accuracy change from item #2 could be
coming from the EXTRA CAPACITY of the new gamma_fc/alpha_fc layers rather
than from the adaptivity itself. This script settles that question with
real numbers before any GPU time is spent.

USAGE
    %run report_efgf_gate_values.py
(edit RESULTS_ROOT below first)
"""
import os
import re
import glob
import numpy as np
import torch

from tsfnet_model_notebook import TSFNet, _collect_gate_values

# ---- EDIT THIS ----
RESULTS_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\EGTA\6"
TAG = "40b7d8ea53"
SEEDS = [42, 999, 2024]
FNIRS_NUM_SEGMENTS = 21
# --------------------

DEVICE = "cpu"  # no training/inference needed, just reading parameters

rows = []
for seed in SEEDS:
    results_dir = os.path.join(RESULTS_ROOT, f"results_{TAG}_seed{seed}")
    ckpts = sorted(glob.glob(os.path.join(results_dir, "*_best.pt")))
    if not ckpts:
        print(f"[warn] no checkpoints found in {results_dir} -- check RESULTS_ROOT/TAG")
        continue
    for ck_path in ckpts:
        name = os.path.basename(ck_path)
        subject_id = next((p for p in name.split("_") if p.startswith("VP")), None)
        m = re.search(r"_session(\d+)_best\.pt$", name)
        test_session = int(m.group(1)) if m else None

        ckpt = torch.load(ck_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

        # Architecture MUST match the accepted config exactly (same bugfix as
        # evaluate_noise_robustness.py), or load_state_dict raises a key mismatch.
        model = TSFNet(
            num_classes=3,
            fusion_combination="paper_mean",
            fnirs_num_segments=FNIRS_NUM_SEGMENTS,
            fnirs_temporal_align="egta",
            efgf_gate_mode="sigmoid",
        ).to(DEVICE)
        model.load_state_dict(sd, strict=True)
        model.eval()

        gates = _collect_gate_values(model)
        rows.append({
            "seed": seed, "subject_id": subject_id, "test_session": test_session,
            "efgf1_gamma": gates.get("efgf1_gamma"), "efgf1_alpha": gates.get("efgf1_alpha"),
            "efgf2_gamma": gates.get("efgf2_gamma"), "efgf2_alpha": gates.get("efgf2_alpha"),
        })

if not rows:
    raise RuntimeError("No checkpoints read at all -- fix RESULTS_ROOT/TAG before continuing.")

print(f"Read {len(rows)} checkpoints across {len(SEEDS)} seeds.\n")

# ---- Per-checkpoint table ----
print(f"{'seed':>5} {'subject':>8} {'sess':>4} {'efgf1_gamma':>12} {'efgf1_alpha':>12} "
      f"{'efgf2_gamma':>12} {'efgf2_alpha':>12}")
for r in rows:
    print(f"{r['seed']:>5} {r['subject_id']:>8} {r['test_session']:>4} "
          f"{r['efgf1_gamma']:>12.4f} {r['efgf1_alpha']:>12.4f} "
          f"{r['efgf2_gamma']:>12.4f} {r['efgf2_alpha']:>12.4f}")

# ---- Summary stats + saturation check ----
print("\n===== SUMMARY (is this parameter actually doing anything?) =====")
SATURATION_LOW, SATURATION_HIGH = 0.05, 0.95  # outside [0.05, 0.95] counts as "saturated"
for key in ("efgf1_gamma", "efgf1_alpha", "efgf2_gamma", "efgf2_alpha"):
    vals = np.array([r[key] for r in rows if r[key] is not None])
    n_sat = int(((vals < SATURATION_LOW) | (vals > SATURATION_HIGH)).sum())
    print(f"{key:>12}: mean={vals.mean():.4f} std={vals.std():.4f} "
          f"min={vals.min():.4f} max={vals.max():.4f} "
          f"saturated(<{SATURATION_LOW} or >{SATURATION_HIGH})={n_sat}/{len(vals)} "
          f"({n_sat/len(vals)*100:.0f}%)")

print("\nHONEST READING GUIDE:")
print("  - If a parameter's saturated-% is HIGH (e.g. >70%) across checkpoints, it has")
print("    converged to near-constant behavior (near-pure residual bypass, or near-pure")
print("    gate-off) for most subjects -- a feature-dependent replacement has little fixed")
print("    behavior left to meaningfully improve upon FOR THOSE SUBJECTS specifically.")
print("  - If std is LARGE across subjects (even if no single subject is saturated), that")
print("    itself is evidence FOR item #2: it means a single global scalar is already a")
print("    poor compromise across the population, which an adaptive map could fix.")
print("  - Look at BOTH signals together, not just the mean -- a dead mean with huge std")
print("    (bimodal across subjects) is a very different story from a dead mean with tiny std.")
