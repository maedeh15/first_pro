# %% CELL 0 -- shared configuration
# =============================================================================
# A2: Modality-level dissociation of the 2-back/3-back confusion.
#
# Question: is the EEG branch or the fNIRS branch (or both) responsible for
# confusing 2-back with 3-back? This matters directly for A3 (asymmetric
# lambda_eeg/lambda_fnirs) and for interpreting the fNIRS-filter hypothesis:
# if fNIRS is the branch that confuses this pair, a systemic (e.g. cardiac)
# fNIRS signal could plausibly explain part of the bottleneck.
#
# Zero training: this script only loads the existing best-checkpoint (.pt)
# for every subject x session fold, reconstructs that exact test split from
# the existing preprocessed .npy files, and runs a forward pass (inference
# only, no gradient updates) to get each branch's OWN prediction (EEG-only,
# fNIRS-only, fusion-only, final) -- exactly the same per-branch classifier
# heads already used to report eeg_only_acc/fnirs_only_acc/fusion_only_acc,
# except here every individual prediction is kept (not just the aggregate
# accuracy) so a full per-branch confusion matrix can be pooled across all
# 26 subjects x 3 sessions = 78 folds.
# =============================================================================

import os
import glob
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader

# --- paths -------------------------------------------------------------
DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
# Folder holding this run's checkpoints, e.g. the plain N=21 baseline variant
# you want to analyze. Copy the exact "results_..._seed{N}" folder name from
# your run's own printed "[RESULTS] Per-subject session-level ... go to:" line.
RESULTS_DIR_VARIANT = os.path.join(
    DATA_ROOT, "results_paper_mean_noclip_noclassweight_nosmote_gap_cafe64_noasym_seed42"
)

# Must match what that run actually used.
FNIRS_NUM_SEGMENTS = 21
NUM_CLASSES = 3
CLASS_NAMES = ["0-back", "2-back", "3-back"]
BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Path to the model definitions file (same one used for training).
MODEL_FILE_PATH = r"E:\kntu\Thesis\codes\HOPE-LAST-CODE-FILE\new-tsfnet\model18\New folder\tsfnet_model_notebook.py"

print("DATA_ROOT             =", DATA_ROOT)
print("RESULTS_DIR_VARIANT   =", RESULTS_DIR_VARIANT, "| exists?", os.path.exists(RESULTS_DIR_VARIANT))
print("MODEL_FILE_PATH        =", MODEL_FILE_PATH, "| exists?", os.path.exists(MODEL_FILE_PATH))
print("device                 =", DEVICE)


# %% CELL 1 -- load model definitions (TSFNet, NBackDataset, session_holdout_split)
# =============================================================================
_ns = {}
with open(MODEL_FILE_PATH, encoding="utf-8") as f:
    _src = f.read()
exec(compile(_src, MODEL_FILE_PATH, "exec"), _ns)

TSFNet = _ns["TSFNet"]
NBackDataset = _ns["NBackDataset"]
session_holdout_split = _ns["session_holdout_split"]
print("Model definitions loaded OK.")


# %% CELL 2 -- enumerate subjects and folds that have a saved checkpoint
# =============================================================================
subjects = sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(DATA_ROOT, "VP*"))
    if os.path.isdir(p)
)

fold_specs = []  # list of (subject_id, test_session, ckpt_path)
for subject_id in subjects:
    for test_session in (1, 2, 3):
        ckpt_path = os.path.join(RESULTS_DIR_VARIANT, f"{subject_id}_session{test_session}_best.pt")
        if os.path.exists(ckpt_path):
            fold_specs.append((subject_id, test_session, ckpt_path))

print(f"Found {len(fold_specs)} fold checkpoints across {len(subjects)} candidate subjects "
      f"(expected up to {len(subjects)*3}).")
missing = len(subjects) * 3 - len(fold_specs)
if missing > 0:
    print(f"[note] {missing} fold(s) have no checkpoint in this folder -- they will simply be skipped.")


# %% CELL 3 -- run inference (no training) for every fold, pool per-branch confusion matrices
# =============================================================================

def confusion_matrix_np(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


cm_eeg = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
cm_fnirs = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
cm_fusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
cm_final = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)

per_fold_rows = []
n_ok, n_failed = 0, 0

for subject_id, test_session, ckpt_path in fold_specs:
    subject_dir = os.path.join(DATA_ROOT, subject_id)
    try:
        full = NBackDataset(subject_dir, subject_id)
        _, test_idx = session_holdout_split(full.session_ids, test_session)
        test_ds = Subset(full, test_idx)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = TSFNet(num_classes=NUM_CLASSES, fnirs_num_segments=FNIRS_NUM_SEGMENTS).to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()

        all_true, all_eeg, all_fnirs, all_fusion, all_final = [], [], [], [], []
        with torch.no_grad():
            for x_eeg, x_fnirs, y in test_loader:
                x_eeg, x_fnirs = x_eeg.to(DEVICE), x_fnirs.to(DEVICE)
                y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs)
                all_true.append(y.numpy())
                all_eeg.append(y_eeg.argmax(dim=-1).cpu().numpy())
                all_fnirs.append(y_fnirs.argmax(dim=-1).cpu().numpy())
                all_fusion.append(y_fusion.argmax(dim=-1).cpu().numpy())
                all_final.append(y_pred.argmax(dim=-1).cpu().numpy())

        y_true = np.concatenate(all_true)
        y_eeg = np.concatenate(all_eeg)
        y_fnirs = np.concatenate(all_fnirs)
        y_fusion = np.concatenate(all_fusion)
        y_final = np.concatenate(all_final)

        fold_cm_eeg = confusion_matrix_np(y_true, y_eeg)
        fold_cm_fnirs = confusion_matrix_np(y_true, y_fnirs)
        fold_cm_fusion = confusion_matrix_np(y_true, y_fusion)
        fold_cm_final = confusion_matrix_np(y_true, y_final)

        cm_eeg += fold_cm_eeg
        cm_fnirs += fold_cm_fnirs
        cm_fusion += fold_cm_fusion
        cm_final += fold_cm_final

        per_fold_rows.append({
            "subject_id": subject_id, "test_session": test_session,
            "n_test_windows": len(y_true),
            "eeg_acc": float((y_eeg == y_true).mean()),
            "fnirs_acc": float((y_fnirs == y_true).mean()),
            "fusion_acc": float((y_fusion == y_true).mean()),
            "final_acc": float((y_final == y_true).mean()),
        })
        n_ok += 1
    except Exception as e:
        print(f"[FAILED] {subject_id} session {test_session}: {e}")
        n_failed += 1

print(f"\n{n_ok} folds processed successfully, {n_failed} failed.")


# %% CELL 4 -- per-branch pooled confusion matrices and 2-back/3-back error share
# =============================================================================

def report_branch(name, cm):
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES],
                          columns=[f"pred_{c}" for c in CLASS_NAMES])
    print(cm_df)
    total = cm.sum()
    acc = np.trace(cm) / total
    print(f"Overall accuracy (pooled): {acc:.4%}")
    for i, cname in enumerate(CLASS_NAMES):
        row_sum = cm[i].sum()
        recall = cm[i, i] / row_sum if row_sum > 0 else float("nan")
        print(f"  recall {cname}: {recall:.4%}  (n={row_sum})")
    total_errors = total - np.trace(cm)
    pair_23_errors = cm[1, 2] + cm[2, 1]
    if total_errors > 0:
        print(f"  2-back<->3-back pair errors: {pair_23_errors} "
              f"({pair_23_errors/total_errors:.2%} of all errors)")
    return acc, pair_23_errors / total_errors if total_errors > 0 else float("nan")


results_summary = {}
for name, cm in [("EEG branch (own classifier head)", cm_eeg),
                  ("fNIRS branch (own classifier head)", cm_fnirs),
                  ("Fusion branch (own classifier head)", cm_fusion),
                  ("Final (paper_mean-weighted combination)", cm_final)]:
    acc, pair_share = report_branch(name, cm)
    results_summary[name] = {"acc": acc, "pair_23_share_of_errors": pair_share}


# %% CELL 5 -- side-by-side summary and interpretation
# =============================================================================
print("\n" + "=" * 70)
print("SIDE-BY-SIDE SUMMARY")
print("=" * 70)
summary_df = pd.DataFrame(results_summary).T
summary_df["acc"] = (summary_df["acc"] * 100).round(2)
summary_df["pair_23_share_of_errors"] = (summary_df["pair_23_share_of_errors"] * 100).round(2)
summary_df.columns = ["accuracy_%", "pct_errors_that_are_2back_3back_pair"]
print(summary_df)

print()
print("Interpretation guide:")
print("- If fNIRS's pair-error share is close to or higher than EEG's, the")
print("  2-back/3-back bottleneck is not uniquely an EEG problem -- this is")
print("  relevant context for the asymmetric-lambda (A3) and fNIRS-filter")
print("  hypotheses.")
print("- If one branch's pair-error share is much lower than the other's,")
print("  that branch is comparatively better at separating this pair, which")
print("  is exactly the kind of evidence A3 (lambda_eeg/lambda_fnirs) would")
print("  act on.")


# %% CELL 6 (optional) -- save per-fold branch accuracies to CSV for further analysis
# =============================================================================
per_fold_df = pd.DataFrame(per_fold_rows)
out_csv = os.path.join(RESULTS_DIR_VARIANT, "A2_per_branch_confusion_analysis.csv")
per_fold_df.to_csv(out_csv, index=False)
print(f"\nPer-fold branch accuracies saved to: {out_csv}")
