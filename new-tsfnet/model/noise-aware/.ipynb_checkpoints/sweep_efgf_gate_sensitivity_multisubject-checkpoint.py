"""
sweep_efgf_gate_sensitivity_multisubject.py -- same sensitivity-sweep idea as
sweep_efgf_gate_sensitivity.py, generalized across MANY subjects (default:
the first 14 of 26, i.e. just over half) and ALL of each subject's fold
checkpoints, then averaged -- so a conclusion ("this scalar matters" /
"doesn't matter") is based on the population, not one subject's quirk.

COST NOTE: this runs len(SUBJECTS) x 3 folds x len(PARAMS_TO_SWEEP) x 21
sweep points x (a handful of mini-batch forward passes each). With the
default PARAMS_TO_SWEEP (gamma only, alpha already closed by the single-
subject result) and 14 subjects, that's 14*3*2*21 = 1176 short forward-pass
evaluations -- still pure inference, no training, but budget real wall-clock
time (likely tens of minutes depending on your GPU), not seconds. Reduce
SUBJECTS or PARAMS_TO_SWEEP if you want a faster first look.

WHY ALPHA IS EXCLUDED BY DEFAULT: the single-subject (VP001) sweep already
showed efgf1.alpha completely flat (spread=0.0000) and efgf2.alpha nearly flat
(spread=0.0033) -- re-running that across 14 more subjects is unlikely to
change the conclusion and would double the runtime for no real benefit. Add
"efgf1.alpha_raw"/"efgf2.alpha_raw" back to PARAMS_TO_SWEEP below if you want
the population-level confirmation anyway.

USAGE
    %run sweep_efgf_gate_sensitivity_multisubject.py
(edit RESULTS_DIR / DATA_ROOT / SUBJECTS below first)
"""
import os
import re
import glob
import numpy as np
import torch

from tsfnet_model_notebook import TSFNet

# ---- EDIT THESE ----
RESULTS_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\EGTA\6\results_40b7d8ea53_seed42"
DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
SUBJECTS = [f"VP{i:03d}" for i in range(1, 15)]   # first 14 of 26 (>half); extend to range(1,27) for all
FNIRS_NUM_SEGMENTS = 21
BATCH_SIZE = 32
PARAMS_TO_SWEEP = ["efgf1.gamma_raw", "efgf2.gamma_raw"]   # alpha already closed, see docstring
# ----------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SWEEP_V = [0.0] + list(np.round(np.linspace(0.05, 0.95, 19), 3)) + [1.0]


def get_param(model, name):
    layer, field = name.split(".")
    return getattr(getattr(model, layer), field)


def logit(v, eps=1e-6):
    v = min(max(v, eps), 1 - eps)
    return float(np.log(v / (1 - v)))


@torch.no_grad()
def evaluate_accuracy(model, x_eeg_all, x_fnirs_all, y_all):
    correct, total = 0, 0
    for i in range(0, len(y_all), BATCH_SIZE):
        xb_eeg = x_eeg_all[i:i + BATCH_SIZE].to(DEVICE)
        xb_fnirs = x_fnirs_all[i:i + BATCH_SIZE].to(DEVICE)
        yb = y_all[i:i + BATCH_SIZE].to(DEVICE)
        _, _, _, y_pred, _ = model(xb_eeg, xb_fnirs, None, None)
        correct += (y_pred.argmax(dim=-1) == yb).sum().item()
        total += yb.shape[0]
    return correct / total


# per-param: list of (fold_curve) where fold_curve is a list of (v, acc) --
# one fold_curve per (subject, checkpoint) processed successfully.
all_curves = {p: [] for p in PARAMS_TO_SWEEP}
trained_values = {p: [] for p in PARAMS_TO_SWEEP}   # the converged value at each (subject, fold)
baseline_accs = []
n_processed = 0

for subject_id in SUBJECTS:
    subject_dir = os.path.join(DATA_ROOT, subject_id)
    ckpts = sorted(glob.glob(os.path.join(RESULTS_DIR, f"{subject_id}_*_best.pt")))
    if not ckpts:
        print(f"[skip] no checkpoints for {subject_id} in {RESULTS_DIR}")
        continue

    session_ids = np.load(os.path.join(subject_dir, f"{subject_id}_session_ids.npy"))
    eeg_mm = np.load(os.path.join(subject_dir, f"{subject_id}_eeg_windows.npy"), mmap_mode="r")
    fnirs_mm = np.load(os.path.join(subject_dir, f"{subject_id}_fnirs_windows.npy"), mmap_mode="r")
    labels_mm = np.load(os.path.join(subject_dir, f"{subject_id}_labels.npy"), mmap_mode="r")

    for ck_path in ckpts:
        name = os.path.basename(ck_path)
        m = re.search(r"_session(\d+)_best\.pt$", name)
        if not m:
            print(f"  [skip] cannot parse session from {name}")
            continue
        test_session = int(m.group(1))
        idx = np.where(session_ids == test_session)[0]
        if len(idx) == 0:
            print(f"  [skip] {name}: no samples for session {test_session}")
            continue

        x_eeg_all = torch.from_numpy(np.array(eeg_mm[idx])).float()
        x_fnirs_all = torch.from_numpy(np.array(fnirs_mm[idx])).float()
        y_all = torch.from_numpy(np.array(labels_mm[idx])).long()

        model = TSFNet(
            num_classes=3, fusion_combination="paper_mean",
            fnirs_num_segments=FNIRS_NUM_SEGMENTS, fnirs_temporal_align="egta",
            efgf_gate_mode="sigmoid",
        ).to(DEVICE)
        ckpt = torch.load(ck_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd, strict=True)
        model.eval()

        baseline_acc = evaluate_accuracy(model, x_eeg_all, x_fnirs_all, y_all)
        baseline_accs.append(baseline_acc)
        n_processed += 1
        print(f"[{subject_id} session{test_session}] baseline acc = {baseline_acc:.4f} "
              f"({n_processed} checkpoints processed so far)")

        for pname in PARAMS_TO_SWEEP:
            param = get_param(model, pname)
            trained_raw = param.data.clone()
            trained_v = torch.sigmoid(trained_raw).item()
            trained_values[pname].append(trained_v)

            curve = []
            for v in SWEEP_V:
                raw_v = logit(v) if 0.0 < v < 1.0 else (-15.0 if v == 0.0 else 15.0)
                param.data.fill_(raw_v)
                acc = evaluate_accuracy(model, x_eeg_all, x_fnirs_all, y_all)
                curve.append((v, acc))
            param.data.copy_(trained_raw)
            all_curves[pname].append(curve)

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

print(f"\n{'='*70}\nProcessed {n_processed} (subject, fold) checkpoints across "
      f"{len(SUBJECTS)} requested subjects.\nMean baseline accuracy = {np.mean(baseline_accs):.4f}\n{'='*70}")

for pname in PARAMS_TO_SWEEP:
    curves = all_curves[pname]
    if not curves:
        print(f"\n[{pname}] no data collected -- check RESULTS_DIR/DATA_ROOT paths.")
        continue
    n_v = len(SWEEP_V)
    mean_curve = []
    for vi in range(n_v):
        v = curves[0][vi][0]
        accs_at_v = [c[vi][1] for c in curves]
        mean_curve.append((v, np.mean(accs_at_v), np.std(accs_at_v)))

    mean_trained_v = np.mean(trained_values[pname])
    print(f"\n===== {pname}: MEAN curve across {len(curves)} (subject, fold) checkpoints =====")
    print(f"  mean trained (converged) value = {mean_trained_v:.4f}")
    for v, m, s in mean_curve:
        print(f"  forced to {v:.2f}: mean acc = {m:.4f}  (std across checkpoints = {s:.4f})")

    means_only = [m for _, m, _ in mean_curve]
    best_v = mean_curve[int(np.argmax(means_only))][0]
    spread = max(means_only) - min(means_only)
    print(f"\n  SPREAD of mean curve = {spread:.4f}  |  best forced value (population-level) = {best_v:.2f} "
          f"|  mean trained value = {mean_trained_v:.4f}")

print("\n===== POPULATION-LEVEL READING GUIDE =====")
print("A small spread (<~0.02) in the MEAN curve, even if individual subjects vary, means")
print("the effect averages out / is not a consistent population-level pattern -- do not")
print("invest in fixing this parameter's gradient.")
print("A spread that is both clearly larger AND one-directional (trained value")
print("consistently below the population's best forced value, like VP001 alone showed)")
print("is the only outcome that would justify revisiting item #2, now with a population-")
print("level effect size instead of a single subject's anecdote.")
