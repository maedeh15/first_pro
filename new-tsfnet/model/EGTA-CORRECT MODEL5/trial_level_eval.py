"""
trial_level_eval.py -- trial-level (block-level) decision aggregation for the
accepted run60-onestage configuration.

WHY THIS SCRIPT EXISTS
======================
Each n-back trial in this dataset is a single 40 s block carrying ONE label.
The preprocessing pipeline turns that block into 35 sliding windows
(Preproc_tsfnet_nback_badch.py: SLIDE_TMIN=-2, SLIDE_TMAX=42,
WINDOW_LEN_EEG=10, WINDOW_STEP=1 -> np.arange(-2, 32+eps, 1) = 35 offsets),
each overlapping its neighbour by 90%.

Every accuracy this project has reported so far is a PER-WINDOW accuracy
(the grand 61.02% over 23076 windows). But the decision unit in any
cognitive-load or BCI application is the BLOCK, not an arbitrary 10 s crop of
it. Aggregating the 35 window posteriors of a trial into one trial-level
decision is therefore not a trick to inflate a number -- it is reporting the
metric that matches the task. The held-out session stays entirely held out:
aggregation happens WITHIN each test trial, using only that trial's own
window posteriors, never any label.

This script computes BOTH metrics side by side so the window-level number
stays available for comparison against TSFNet / MSTFDN (which also report
window-level), and the trial-level number is reported as the applied metric.

WHY IT RE-RUNS THE MODEL INSTEAD OF READING THE SAVED JSON
==========================================================
detailed_test_report() (tsfnet_model_notebook.py) calls .argmax(dim=-1)
immediately on every head and never records trial_ids, so the saved
per-session JSON contains only the confusion matrix and aggregate accuracies.
The per-window POSTERIORS -- which is exactly what trial-level aggregation
needs -- are not persisted anywhere. So this script reloads each saved
checkpoint and does one forward pass per fold. No training, no gradients:
26 subjects x 3 folds x ~300 windows is a few minutes on a GPU.

CRITICAL: AdaBN IS NOT IN THE CHECKPOINT
========================================
In run_subject_cross_session the order is:

    model.load_state_dict(torch.load(ckpt_path))     # tsfnet_model_notebook.py:3324
    adapt_batchnorm_to_target(model, test_loader, ...)  #                     :3349
    report = detailed_test_report(model, test_loader, ...)  #                 :3350

and the last torch.save() happens at :3314, i.e. BEFORE AdaBN runs. So
`VPxxx_session{k}_best.pt` does NOT contain the AdaBN-adapted BatchNorm
running statistics -- it holds the pre-adaptation ones. Evaluating the
checkpoint without re-applying AdaBN therefore reproduces a DIFFERENT (and
lower) window-level accuracy than the 61.02% on record, and that discrepancy
would silently contaminate the window-vs-trial comparison this script exists
to make.

This script therefore re-applies adapt_batchnorm_to_target() exactly as the
training script does (--adabn, on by default here because run60 used it), and
PRINTS the reproduced window-level accuracy against the value on record as a
self-check before reporting anything else. If that check does not match,
nothing below it should be trusted -- the message says so explicitly.

adapt_batchnorm_to_target is order-independent by construction
(reset_running_stats() then momentum=None -> exact cumulative average), so
re-running it here reproduces the same statistics the original run used, not
merely similar ones.

TRANSDUCTIVE-PROTOCOL DISCLOSURE (unchanged from the accepted baseline)
=======================================================================
AdaBN uses the held-out session's UNLABELED inputs. That is this project's
existing, documented, accepted choice (priority-E4) and this script does not
change it -- it only reproduces it. The thesis must disclose it, as the
docstring of adapt_batchnorm_to_target already states. Trial-level
aggregation itself adds NO new transduction: it pools posteriors within one
trial and touches no label.

WHAT IT DOES *NOT* MODIFY
=========================
Nothing. No existing file in this folder is touched. Definitions are loaded
from tsfnet_model_notebook.py via exec(), the same mechanism
run_all_subjects.py:755 already uses -- that file contains only defs plus
three harmless module-level constants, so loading it starts no training.

Every behavioural choice is behind a flag whose default reproduces the
run60-onestage configuration, per this project's rule 5.

USAGE
=====
    python trial_level_eval.py --seed 2024 --results-dir <...>

    # all three seeds of run60, then compare
    python trial_level_eval.py --seed 42   --variant-hash 40b7d8ea53
    python trial_level_eval.py --seed 999  --variant-hash 40b7d8ea53
    python trial_level_eval.py --seed 2024 --variant-hash 40b7d8ea53

Known small-sample caveat, also printed in the output: each test session
holds only 9 trials (3 per class), so a trial-level accuracy is quantized to
multiples of 1/9 and a single subject-fold estimate is very noisy. Only the
pooled estimate (26 subjects x 3 folds x 3 seeds = 702 trial decisions) is
worth interpreting.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


# ---------------------------------------------------------------------------
# CLI -- every architecture flag mirrors run_all_subjects.py's spelling, and
# every default reproduces run60-onestage:
#   --seed {42,999,2024} --adabn --fnirs-temporal-align egta
#   --efgf-gate-mode sigmoid --efgf-gate-init -2.0 --normalize-ypred
#   --no-decay-scalars --stage2-stop-mode none
# (--normalize-ypred / --no-decay-scalars / --stage2-stop-mode are TRAINING-
# time only: they affect the loss and the optimizer, not the forward pass or
# the module structure, so they cannot and need not be replayed at eval time.
# The architecture-relevant subset is what appears below.)
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser(
    description="Trial-level (block-level) decision aggregation from saved checkpoints.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

p.add_argument("--data-root", type=str,
               default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback",
               help="Folder holding the per-subject VPxxx/ preprocessed .npy folders. "
                    "Same DATA_ROOT as run_all_subjects.py.")
p.add_argument("--seed", type=int, required=True,
               help="Which seed's results folder to evaluate (42, 999 or 2024 for run60).")
p.add_argument("--variant-hash", type=str, default="40b7d8ea53",
               help="The 10-char variant hash in the results folder name "
                    "(run60-onestage = 40b7d8ea53).")
p.add_argument("--results-dir", type=str, default=None,
               help="Explicit path to the results folder holding the .pt checkpoints. "
                    "Overrides --variant-hash/--seed folder derivation.")
p.add_argument("--subjects", type=str, default="1-26",
               help="Subject range or comma list, e.g. '1-26' or '1,5,17'.")
p.add_argument("--batch-size", type=int, default=16,
               help="Must match the training run so AdaBN sees the same batch "
                    "partitioning (run_all_subjects.py passes 16).")
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out-prefix", type=str, default=None,
               help="Prefix for the CSV/JSON written into --results-dir. "
                    "Defaults to 'trial_level_eval_seed{SEED}'.")
p.add_argument("--expected-window-acc", type=float, default=None,
               help="Grand-mean window-level accuracy on record for this run+seed "
                    "(run60 seed 2024 = 0.6091). When given, the script prints a "
                    "PASS/FAIL self-check against it.")
p.add_argument("--reproduce-tolerance", type=float, default=0.002,
               help="Absolute tolerance for the --expected-window-acc self-check.")

# --- AdaBN: ON by default because run60 used --adabn -----------------------
p.add_argument("--adabn", dest="adabn", action="store_true", default=True,
               help="Re-apply adapt_batchnorm_to_target before scoring (run60 default). "
                    "REQUIRED to reproduce the reported accuracy -- see module docstring.")
p.add_argument("--no-adabn", dest="adabn", action="store_false",
               help="Score the raw checkpoint without AdaBN. Will NOT reproduce the "
                    "reported window-level accuracy; for diagnosis only.")

# --- architecture flags (TSFNet ctor), defaults = run60 -------------------
p.add_argument("--fnirs-temporal-align", type=str, default="egta", choices=["none", "egta"])
p.add_argument("--efgf-gate-mode", type=str, default="sigmoid", choices=["clamp", "sigmoid"])
p.add_argument("--efgf-gate-init", type=float, default=-2.0)
p.add_argument("--fusion-combination", type=str, default="paper_mean",
               choices=["paper_mean", "wsum", "gated", "stanet2", "stanet3"])
p.add_argument("--fnirs-num-segments", type=int, default=21)
p.add_argument("--cafe-d-model", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--cafe-dropout", type=float, default=0.3)
p.add_argument("--eeg-temporal-mode", type=str, default="gap", choices=["gap", "gap_logvar", "bilstm"])
p.add_argument("--fusion-fnirs-inject", type=str, default="none", choices=["none", "stage1", "both"])
p.add_argument("--erp-branch", type=str, default="none", choices=["none", "eeg", "eeg_fusion"])
p.add_argument("--eeg-filterbank", type=str, default="none", choices=["none", "fixed", "learnable"])
p.add_argument("--eeg-filterbank-scope", type=str, default="both", choices=["both", "eeg"])
p.add_argument("--fnirs-chroma-split", type=str, default="none",
               choices=["none", "separate", "separate_pair"])
p.add_argument("--eeg-lstm-hidden", type=int, default=64)
p.add_argument("--eeg-readout-mode", type=str, default="h_n",
               choices=["h_n", "last_step", "mean_pool"])
p.add_argument("--fnirs-chroma-mode", type=str, default="both")
p.add_argument("--fnirs-num-chromophores", type=int, default=2, choices=[2, 3])
p.add_argument("--eeg-spatial-mode", type=str, default="none")
p.add_argument("--bandtok-norm", type=str, default="none", choices=["none", "relative"])
p.add_argument("--bandtok-d-model", type=int, default=32)
p.add_argument("--bandtok-heads", type=int, default=4)
p.add_argument("--eeg-band-mode", type=str, default="none")
p.add_argument("--eeg-backbone", type=str, default="native")
p.add_argument("--norm-mode", type=str, default="batch", choices=["batch", "group"])
p.add_argument("--use-fc-features", action="store_true", default=False)

ARGS = p.parse_args()


# ---------------------------------------------------------------------------
# Load the project's definitions. Same mechanism as run_all_subjects.py:755 --
# tsfnet_model_notebook.py holds only defs (plus CLASS_NAMES/COLORS/
# EEG_WINDOW_SECONDS), so this starts nothing.
# ---------------------------------------------------------------------------
try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:                                  # %run / exec context
    _this_dir = os.getcwd()

_notebook_path = os.path.join(_this_dir, "tsfnet_model_notebook.py")
if not os.path.isfile(_notebook_path):
    sys.exit(f"ERROR: tsfnet_model_notebook.py not found next to this script "
             f"({_notebook_path}). Put trial_level_eval.py in the same folder.")

if "TSFNet" not in globals():                      # already loaded by a prior %run?
    print(f"[init] loading definitions from {_notebook_path}")
    exec(open(_notebook_path, encoding="utf-8").read(), globals())


# ---------------------------------------------------------------------------
# Aggregation rules
# ---------------------------------------------------------------------------
def aggregate_mean_prob(probs, trial_ids):
    """PRIMARY rule: average the posterior over a trial's windows, then argmax.

    Uses the model's confidence, so a window the model is unsure about
    contributes less than a confident one -- strictly more information than a
    hard vote. This is the standard choice for pooling overlapping-crop
    posteriors.

    probs: (N, C) per-window posteriors (already softmax -- every head of this
           model returns probabilities, see BranchClassifier.forward).
    trial_ids: (N,) trial id per window.
    Returns (trials_sorted, preds) with one prediction per unique trial.
    """
    trials = np.unique(trial_ids)
    preds = np.empty(len(trials), dtype=np.int64)
    for i, t in enumerate(trials):
        preds[i] = probs[trial_ids == t].mean(axis=0).argmax()
    return trials, preds


def aggregate_majority_vote(probs, trial_ids, num_classes=3):
    """SECONDARY rule, reported for comparison: hard per-window argmax, then
    the modal class over the trial. Ties are broken toward the class with the
    larger summed posterior, which keeps the rule deterministic (np.bincount's
    argmax alone would silently favour the lowest class index)."""
    trials = np.unique(trial_ids)
    preds = np.empty(len(trials), dtype=np.int64)
    hard = probs.argmax(axis=1)
    for i, t in enumerate(trials):
        m = trial_ids == t
        counts = np.bincount(hard[m], minlength=num_classes)
        top = np.flatnonzero(counts == counts.max())
        if len(top) == 1:
            preds[i] = top[0]
        else:
            summed = probs[m].sum(axis=0)
            preds[i] = top[np.argmax(summed[top])]
    return trials, preds


def trial_labels(labels, trial_ids):
    """One label per trial. Every window of a trial shares its block's label
    by construction; this asserts that rather than assuming it, because a
    silent violation here would invalidate every trial-level number below."""
    trials = np.unique(trial_ids)
    out = np.empty(len(trials), dtype=np.int64)
    for i, t in enumerate(trials):
        lab = np.unique(labels[trial_ids == t])
        if len(lab) != 1:
            raise AssertionError(
                f"trial {t} carries {len(lab)} distinct labels ({lab.tolist()}). "
                f"Trial-level aggregation is only meaningful if each trial has one "
                f"label -- check the preprocessing's trial_ids/labels alignment.")
        out[i] = lab[0]
    return out


HEADS = ("final", "eeg", "fusion", "fnirs")


# ---------------------------------------------------------------------------
# One fold: load checkpoint, re-apply AdaBN, collect posteriors
# ---------------------------------------------------------------------------
def collect_fold_posteriors(full, test_idx, ckpt_path, device, batch_size, adabn):
    """Returns dict head -> (N, 3) posteriors, plus the labels actually seen
    in loader order.

    The loader is built with shuffle=False over Subset(full, test_idx), exactly
    as run_subject_cross_session does, so posterior row i corresponds to
    test_idx[i]. The returned labels are compared against full.labels[test_idx]
    by the caller -- an ordering bug would otherwise corrupt the trial grouping
    without raising anything.
    """
    model = TSFNet(
        num_classes=3,
        dropout=ARGS.dropout, cafe_dropout=ARGS.cafe_dropout,
        cafe_d_model=ARGS.cafe_d_model,
        fusion_combination=ARGS.fusion_combination,
        eeg_temporal_mode=ARGS.eeg_temporal_mode,
        eeg_lstm_hidden=ARGS.eeg_lstm_hidden,
        eeg_readout_mode=ARGS.eeg_readout_mode,
        fnirs_num_segments=ARGS.fnirs_num_segments,
        fnirs_chroma_mode=ARGS.fnirs_chroma_mode,
        fnirs_num_chromophores=ARGS.fnirs_num_chromophores,
        use_fc_features=ARGS.use_fc_features,
        eeg_spatial_mode=ARGS.eeg_spatial_mode,
        eeg_backbone_mode=ARGS.eeg_backbone,
        fnirs_temporal_align=ARGS.fnirs_temporal_align,
        eeg_band_mode=ARGS.eeg_band_mode,
        norm_mode=ARGS.norm_mode,
        efgf_gate_mode=ARGS.efgf_gate_mode,
        efgf_gate_init=ARGS.efgf_gate_init,
        fusion_fnirs_inject=ARGS.fusion_fnirs_inject,
        erp_branch=ARGS.erp_branch,
        eeg_filterbank=ARGS.eeg_filterbank,
        eeg_filterbank_scope=ARGS.eeg_filterbank_scope,
        fnirs_chroma_split=ARGS.fnirs_chroma_split,
        bandtok_d_model=ARGS.bandtok_d_model, bandtok_heads=ARGS.bandtok_heads,
        bandtok_norm=ARGS.bandtok_norm,
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    # strict=True on purpose: a key mismatch means the --efgf-gate-mode /
    # --fnirs-temporal-align / --eeg-* flags do not match the run that produced
    # this checkpoint, and silently tolerating that would score a partly
    # randomly-initialized model. Fail loudly with the offending keys instead.
    model.load_state_dict(state, strict=True)
    model.eval()

    test_ds = Subset(full, test_idx)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if adabn:
        # Reproduces the training script's own pre-scoring step
        # (tsfnet_model_notebook.py:3349). Order-independent by construction.
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)

    out = {h: [] for h in HEADS}
    seen_labels = []
    with torch.no_grad():
        for _batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            y_eeg, y_fusion, y_fnirs, y_pred, _aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            out["final"].append(y_pred.cpu().numpy())
            out["eeg"].append(y_eeg.cpu().numpy())
            out["fusion"].append(y_fusion.cpu().numpy())
            out["fnirs"].append(y_fnirs.cpu().numpy())
            seen_labels.append(y.numpy())

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return ({h: np.concatenate(v, axis=0) for h, v in out.items()},
            np.concatenate(seen_labels))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


def locate_results_dir(folder_name, search_roots, max_depth=4):
    """Breadth-limited search for a results folder that was moved or renamed.

    Results folders are named results_<hash>_seed<N> by run_all_subjects.py, so
    the folder NAME is a reliable fingerprint even when the parent path has
    changed (e.g. the code tree was copied somewhere new). Returns every match,
    so an ambiguous situation is reported instead of silently picking one.
    """
    matches, seen = [], set()
    for root in search_roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        base_depth = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _files in os.walk(root):
            if dirpath.count(os.sep) - base_depth >= max_depth:
                dirnames[:] = []                     # stop descending
                continue
            if folder_name in dirnames:
                matches.append(os.path.join(dirpath, folder_name))
            # skip heavy/irrelevant trees
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in ("__pycache__", "visualizations")]
    return sorted(set(matches))


def main():
    device = torch.device(ARGS.device)
    folder_name = f"results_{ARGS.variant_hash}_seed{ARGS.seed}"
    results_dir = ARGS.results_dir or os.path.join(ARGS.data_root, folder_name)

    if not os.path.isdir(results_dir):
        print(f"[locate] not found at: {results_dir}")
        if ARGS.results_dir:
            sys.exit("ERROR: the --results-dir you passed does not exist.")
        # The default path is derived from --data-root. When the code tree has
        # been moved, look for the folder by name around the most likely roots:
        # the configured data root's ancestors and this script's own ancestors.
        roots = []
        for start in (ARGS.data_root, _this_dir):
            p = os.path.abspath(start)
            for _ in range(4):                       # p, parent, grandparent, ...
                roots.append(p)
                p = os.path.dirname(p)
        print(f"[locate] searching for '{folder_name}' near the data root and this "
              f"script (this can take a few seconds) ...")
        found = locate_results_dir(folder_name, roots)
        if len(found) == 1:
            results_dir = found[0]
            print(f"[locate] FOUND: {results_dir}")
            # The per-subject .npy folders normally sit next to the results folder.
            # Only follow it automatically if --data-root itself is missing.
            if not os.path.isdir(ARGS.data_root):
                ARGS.data_root = os.path.dirname(results_dir)
                print(f"[locate] --data-root did not exist; using the results folder's "
                      f"parent instead: {ARGS.data_root}")
        elif len(found) > 1:
            print("[locate] the folder exists in MORE THAN ONE place:")
            for f in found:
                print(f"           {f}")
            sys.exit("ERROR: ambiguous -- pass the right one with --results-dir \"...\"")
        else:
            msg = [f"ERROR: could not find '{folder_name}' anywhere near:",
                   f"         {ARGS.data_root}",
                   f"         {_this_dir}"]
            if os.path.isdir(ARGS.data_root):
                existing = sorted(d for d in os.listdir(ARGS.data_root)
                                  if d.startswith("results_"))
                if existing:
                    msg.append(f"  results_* folders that DO exist in --data-root "
                               f"({len(existing)}):")
                    msg += [f"         {d}" for d in existing[:30]]
                    same_seed = [d for d in existing if d.endswith(f"_seed{ARGS.seed}")]
                    if same_seed:
                        msg.append(f"  For seed {ARGS.seed}, candidates are: "
                                   f"{', '.join(same_seed)} -- check --variant-hash.")
                else:
                    msg.append("  --data-root exists but contains no results_* folders.")
            else:
                msg.append(f"  --data-root itself does not exist: {ARGS.data_root}")
            msg.append("  Fix: find the folder in Explorer and pass it with "
                       "--results-dir \"<full path>\" (and --data-root \"<folder holding "
                       "VP001..VP026>\" if that moved too).")
            sys.exit("\n".join(msg))

    # The checkpoints alone are not enough: the per-subject .npy inputs are read
    # from --data-root/VPxxx. Check up front instead of skipping all 26 subjects.
    if not os.path.isdir(os.path.join(ARGS.data_root, "VP001")):
        sys.exit(f"ERROR: --data-root has no VP001 folder: {ARGS.data_root}\n"
                 f"  Pass --data-root \"<folder that contains VP001..VP026>\".")

    out_prefix = ARGS.out_prefix or f"trial_level_eval_seed{ARGS.seed}"
    subjects = parse_subjects(ARGS.subjects)

    print("=" * 78)
    print(f"TRIAL-LEVEL EVALUATION | seed={ARGS.seed} | device={device}")
    print(f"  results_dir : {results_dir}")
    print(f"  data_root   : {ARGS.data_root}")
    print(f"  AdaBN       : {'ON (reproduces the reported condition)' if ARGS.adabn else 'OFF -- will NOT reproduce reported accuracy'}")
    print(f"  subjects    : {len(subjects)} ({subjects[0]}..{subjects[-1]})")
    print("=" * 78)

    rows = []
    # Pooled decisions across every subject and fold, for one grand estimate
    # that is not an average-of-averages over 9-trial folds.
    trial_rows = []   # one row per block, final head -- consumed by capacity_ceiling_check.py
    pooled = {h: {"win_true": [], "win_pred": [], "tri_true": [], "tri_pred": [],
                  "tri_pred_vote": []} for h in HEADS}

    for subject_id in subjects:
        subject_dir = os.path.join(ARGS.data_root, subject_id)
        if not os.path.isdir(subject_dir):
            print(f"[{subject_id}] SKIP -- no preprocessed folder at {subject_dir}")
            continue

        try:
            full = NBackDataset(subject_dir, subject_id, load_erp=(ARGS.erp_branch != "none"))
        except Exception as exc:
            print(f"[{subject_id}] SKIP -- could not load dataset: {exc}")
            continue

        session_ids = full.session_ids
        all_trial_ids = full.trial_ids
        labels_np = full.labels.numpy()

        for test_session in sorted(np.unique(session_ids)):
            _train_val_idx, test_idx = session_holdout_split(session_ids, test_session)
            ckpt_path = os.path.join(results_dir, f"{subject_id}_session{test_session}_best.pt")
            if not os.path.isfile(ckpt_path):
                print(f"[{subject_id} s{test_session}] SKIP -- no checkpoint at {ckpt_path}")
                continue

            try:
                posteriors, seen_labels = collect_fold_posteriors(
                    full, test_idx, ckpt_path, device, ARGS.batch_size, ARGS.adabn)
            except Exception as exc:
                print(f"[{subject_id} s{test_session}] FAILED -- {type(exc).__name__}: {exc}")
                continue

            y_true_win = labels_np[test_idx]
            tri_ids_win = all_trial_ids[test_idx]

            # Ordering guard: DataLoader(shuffle=False) must have handed back
            # the windows in test_idx order. If it did not, tri_ids_win would
            # be misaligned with the posteriors and every trial-level number
            # would be quietly wrong.
            if len(seen_labels) != len(y_true_win) or not np.array_equal(seen_labels, y_true_win):
                print(f"[{subject_id} s{test_session}] FAILED -- loader order does not "
                      f"match test_idx; refusing to aggregate.")
                continue

            y_true_tri = trial_labels(y_true_win, tri_ids_win)

            row = {"subject_id": subject_id, "test_session": int(test_session),
                   "n_windows": int(len(y_true_win)), "n_trials": int(len(y_true_tri))}

            for h in HEADS:
                probs = posteriors[h]
                win_pred = probs.argmax(axis=1)
                _tr, tri_pred = aggregate_mean_prob(probs, tri_ids_win)
                _tr2, tri_pred_vote = aggregate_majority_vote(probs, tri_ids_win)

                row[f"{h}_win_acc"] = float((win_pred == y_true_win).mean())
                row[f"{h}_win_kappa"] = float(cohen_kappa(y_true_win, win_pred))
                row[f"{h}_trial_acc"] = float((tri_pred == y_true_tri).mean())
                row[f"{h}_trial_kappa"] = float(cohen_kappa(y_true_tri, tri_pred))
                row[f"{h}_trial_acc_vote"] = float((tri_pred_vote == y_true_tri).mean())

                pooled[h]["win_true"].append(y_true_win)
                pooled[h]["win_pred"].append(win_pred)
                pooled[h]["tri_true"].append(y_true_tri)
                pooled[h]["tri_pred"].append(tri_pred)
                pooled[h]["tri_pred_vote"].append(tri_pred_vote)

            rows.append(row)
            fp = posteriors["final"]
            for t in np.unique(tri_ids_win):
                mp = fp[tri_ids_win == t].mean(axis=0)
                trial_rows.append({"subject_id": subject_id, "test_session": int(test_session),
                                   "trial_id": int(t), "label": int(y_true_win[tri_ids_win == t][0]),
                                   "pred": int(mp.argmax()), "p0": float(mp[0]),
                                   "p1": float(mp[1]), "p2": float(mp[2])})
            print(f"[{subject_id} s{test_session}] "
                  f"window acc={row['final_win_acc']:.4f} -> "
                  f"trial acc={row['final_trial_acc']:.4f} "
                  f"(vote {row['final_trial_acc_vote']:.4f}) "
                  f"| {row['n_windows']} win / {row['n_trials']} trials")

    if not rows:
        sys.exit("ERROR: no fold produced a result. Nothing to report.")

    # ---------------- per-subject means (the unit of the Wilcoxon test) -----
    by_subject = {}
    for r in rows:
        by_subject.setdefault(r["subject_id"], []).append(r)

    subj_win_acc, subj_tri_acc, subj_win_kap, subj_tri_kap, subj_ids = [], [], [], [], []
    for sid, rs in sorted(by_subject.items()):
        subj_ids.append(sid)
        subj_win_acc.append(np.mean([r["final_win_acc"] for r in rs]))
        subj_tri_acc.append(np.mean([r["final_trial_acc"] for r in rs]))
        subj_win_kap.append(np.mean([r["final_win_kappa"] for r in rs]))
        subj_tri_kap.append(np.mean([r["final_trial_kappa"] for r in rs]))
    subj_win_acc = np.array(subj_win_acc)
    subj_tri_acc = np.array(subj_tri_acc)
    subj_win_kap = np.array(subj_win_kap)
    subj_tri_kap = np.array(subj_tri_kap)

    # ---------------- grand-mean self-check against the record -------------
    grand_win = float(subj_win_acc.mean())
    print()
    print("=" * 78)
    print("SELF-CHECK -- does the checkpoint replay reproduce the reported run?")
    print("=" * 78)
    print(f"  reproduced grand-mean window accuracy : {grand_win:.4f}")
    if ARGS.expected_window_acc is not None:
        delta = abs(grand_win - ARGS.expected_window_acc)
        ok = delta <= ARGS.reproduce_tolerance
        print(f"  on record for this run+seed           : {ARGS.expected_window_acc:.4f}")
        print(f"  |difference|                          : {delta:.4f} "
              f"(tolerance {ARGS.reproduce_tolerance})")
        print(f"  VERDICT: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  *** The replay does NOT reproduce the reported window-level "
                  "accuracy. ***")
            print("  Every trial-level number below is therefore NOT comparable to the "
                  "reported baseline. Most likely causes, in order:")
            print("    1. --adabn omitted (the checkpoint does not contain the "
                  "AdaBN-adapted BN statistics -- see module docstring).")
            print("    2. --batch-size differs from training (16), changing AdaBN's "
                  "batch partitioning.")
            print("    3. An architecture flag differs from the run that wrote these "
                  "checkpoints.")
            print("    4. The preprocessed .npy files under --data-root were "
                  "regenerated after this run.")
    else:
        print("  (pass --expected-window-acc to check this automatically; "
              "run60 seed 2024 = 0.6091)")

    # ---------------- pooled metrics ---------------------------------------
    pooled_summary = {}
    for h in HEADS:
        wt = np.concatenate(pooled[h]["win_true"])
        wp = np.concatenate(pooled[h]["win_pred"])
        tt = np.concatenate(pooled[h]["tri_true"])
        tp = np.concatenate(pooled[h]["tri_pred"])
        tv = np.concatenate(pooled[h]["tri_pred_vote"])
        pooled_summary[h] = {
            "n_windows": int(len(wt)),
            "n_trials": int(len(tt)),
            "window_acc_pooled": float((wp == wt).mean()),
            "window_kappa_pooled": float(cohen_kappa(wt, wp)),
            "trial_acc_pooled_meanprob": float((tp == tt).mean()),
            "trial_kappa_pooled_meanprob": float(cohen_kappa(tt, tp)),
            "trial_acc_pooled_vote": float((tv == tt).mean()),
            "trial_kappa_pooled_vote": float(cohen_kappa(tt, tv)),
            "window_confusion_pooled": confusion_matrix_np(wt, wp, 3).tolist(),
            "trial_confusion_pooled_meanprob": confusion_matrix_np(tt, tp, 3).tolist(),
        }

    # ---------------- Wilcoxon, matching the project's acceptance protocol --
    # MIN_N: the Wilcoxon signed-rank test cannot reach p<0.05 two-sided with
    # fewer than 6 non-tied pairs no matter how one-sided the data is (the
    # smallest attainable two-sided p at n=5 is 2/2^5 = 0.0625). Reporting a
    # p-value below that threshold would be reporting a number that cannot
    # mean what a reader will take it to mean -- the n=1 case in this script's
    # own smoke test printed a confident-looking "p=1". So below MIN_N the
    # descriptive statistics are still reported and p_value is explicitly None
    # with a note, rather than a figure that invites over-reading.
    MIN_N_WILCOXON = 6
    stats_block = {}
    for name, a, b in (("accuracy", subj_win_acc, subj_tri_acc),
                       ("kappa", subj_win_kap, subj_tri_kap)):
        diff = b - a
        n_nonzero = int((diff != 0).sum())
        block = {
            "n_subjects": int(len(a)),
            "n_nonzero_pairs": n_nonzero,
            "window_level_mean": float(a.mean()),
            "trial_level_mean": float(b.mean()),
            "mean_improvement": float(diff.mean()),
            "n_subjects_improved": int((diff > 0).sum()),
            "n_subjects_worsened": int((diff < 0).sum()),
            "n_subjects_tied": int((diff == 0).sum()),
            "wilcoxon_statistic": None,
            "p_value": None,
        }
        if n_nonzero == 0:
            block["note"] = ("every subject's window- and trial-level values are "
                             "identical; the test is undefined")
        elif n_nonzero < MIN_N_WILCOXON:
            block["note"] = (f"only {n_nonzero} non-tied pair(s); the Wilcoxon "
                             f"signed-rank test cannot reach p<0.05 two-sided below "
                             f"{MIN_N_WILCOXON}, so no p-value is reported")
        else:
            # Any scipy failure is caught: it must not discard a completed
            # evaluation sweep that already cost every forward pass above.
            try:
                from scipy.stats import wilcoxon
                stat, pval = wilcoxon(a, b)
                block["wilcoxon_statistic"] = float(stat)
                block["p_value"] = float(pval)
            except ImportError:
                block["note"] = "scipy not installed; Wilcoxon test skipped"
            except Exception as exc:
                block["note"] = f"Wilcoxon test failed ({type(exc).__name__}: {exc})"
        stats_block[name] = block

    # ---------------- report ------------------------------------------------
    print()
    print("=" * 78)
    print("WINDOW-LEVEL vs TRIAL-LEVEL (final fused head)")
    print("=" * 78)
    print(f"  subjects                      : {len(subj_ids)}")
    print(f"  per-subject mean, window level: acc={subj_win_acc.mean():.4f} "
          f"+-{subj_win_acc.std():.4f}  kappa={subj_win_kap.mean():.4f}")
    print(f"  per-subject mean, trial level : acc={subj_tri_acc.mean():.4f} "
          f"+-{subj_tri_acc.std():.4f}  kappa={subj_tri_kap.mean():.4f}")
    print(f"  improvement                   : "
          f"{(subj_tri_acc.mean() - subj_win_acc.mean()) * 100:+.2f} pp accuracy, "
          f"{subj_tri_kap.mean() - subj_win_kap.mean():+.4f} kappa")
    sa = stats_block["accuracy"]
    if sa.get("p_value") is not None:
        print(f"  Wilcoxon (paired, {sa['n_subjects']} subjects) : "
              f"p={sa['p_value']:.3g} | improved in "
              f"{sa['n_subjects_improved']}/{sa['n_subjects']} subjects")
    else:
        print(f"  Wilcoxon: not reported -- {sa.get('note', 'unavailable')}")
        print(f"  (descriptive only: improved in {sa['n_subjects_improved']}, "
              f"worsened in {sa['n_subjects_worsened']}, "
              f"tied in {sa['n_subjects_tied']})")

    print()
    print("POOLED OVER ALL SUBJECTS AND FOLDS (not an average of averages)")
    for h in HEADS:
        s = pooled_summary[h]
        print(f"  [{h:>6}] window {s['window_acc_pooled']:.4f} "
              f"(n={s['n_windows']})  ->  trial {s['trial_acc_pooled_meanprob']:.4f} "
              f"(n={s['n_trials']}, vote {s['trial_acc_pooled_vote']:.4f})")

    print()
    print("CAVEATS (carry these into the thesis text)")
    print("  * 9 trials per test session -> trial accuracy is quantized to 1/9")
    print("    steps. A single subject-fold value is very noisy; interpret only")
    print("    the pooled estimate, and only after merging all three seeds.")
    print("  * AdaBN is transductive (unlabeled target inputs). That is this")
    print("    project's existing accepted choice, reproduced here unchanged, and")
    print("    must be disclosed. Trial aggregation itself adds no transduction.")
    print("  * Report BOTH levels: window level for comparability with TSFNet /")
    print("    MSTFDN, trial level as the applied (block) decision metric.")

    # ---------------- write -------------------------------------------------
    csv_path = os.path.join(results_dir, f"{out_prefix}.csv")
    json_path = os.path.join(results_dir, f"{out_prefix}.json")

    fieldnames = ["subject_id", "test_session", "n_windows", "n_trials"]
    for h in HEADS:
        fieldnames += [f"{h}_win_acc", f"{h}_win_kappa", f"{h}_trial_acc",
                       f"{h}_trial_kappa", f"{h}_trial_acc_vote"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "config": {
            "seed": ARGS.seed,
            "results_dir": results_dir,
            "data_root": ARGS.data_root,
            "adabn": ARGS.adabn,
            "batch_size": ARGS.batch_size,
            "aggregation_primary": "mean posterior over a trial's windows, then argmax",
            "aggregation_secondary": "per-window argmax, then modal class "
                                     "(ties broken by summed posterior)",
            "architecture_flags": {
                "fnirs_temporal_align": ARGS.fnirs_temporal_align,
                "efgf_gate_mode": ARGS.efgf_gate_mode,
                "efgf_gate_init": ARGS.efgf_gate_init,
                "fusion_combination": ARGS.fusion_combination,
                "fnirs_num_segments": ARGS.fnirs_num_segments,
                "cafe_d_model": ARGS.cafe_d_model,
                "eeg_temporal_mode": ARGS.eeg_temporal_mode,
                "norm_mode": ARGS.norm_mode,
            },
        },
        "self_check": {
            "reproduced_grand_mean_window_acc": grand_win,
            "expected_window_acc": ARGS.expected_window_acc,
            "tolerance": ARGS.reproduce_tolerance,
            "passed": (None if ARGS.expected_window_acc is None
                       else bool(abs(grand_win - ARGS.expected_window_acc)
                                 <= ARGS.reproduce_tolerance)),
        },
        "per_subject_mean_final_head": {
            "subjects": subj_ids,
            "window_acc": subj_win_acc.tolist(),
            "trial_acc": subj_tri_acc.tolist(),
            "window_kappa": subj_win_kap.tolist(),
            "trial_kappa": subj_tri_kap.tolist(),
            "grand_window_acc": float(subj_win_acc.mean()),
            "grand_trial_acc": float(subj_tri_acc.mean()),
            "grand_window_kappa": float(subj_win_kap.mean()),
            "grand_trial_kappa": float(subj_tri_kap.mean()),
        },
        "pooled": pooled_summary,
        "wilcoxon_window_vs_trial": stats_block,
        "caveats": [
            "9 trials per test session: trial-level accuracy is quantized to 1/9 "
            "steps and a single subject-fold estimate is very noisy. Interpret the "
            "pooled estimate, after merging all seeds.",
            "AdaBN is transductive (uses the held-out session's unlabeled inputs). "
            "Pre-existing accepted project choice, reproduced unchanged here; must "
            "be disclosed in the thesis.",
            "Trial-level aggregation uses only each test trial's own window "
            "posteriors and no labels; the held-out session remains held out.",
            "Report window-level alongside trial-level: TSFNet and MSTFDN report "
            "window-level, so only that number is directly comparable to them.",
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[written] {csv_path}")
    print(f"[written] {json_path}")
    trials_path = os.path.join(results_dir, f"{out_prefix}_trials.csv")
    with open(trials_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
        w.writeheader()
        w.writerows(trial_rows)
    print(f"[written] {trials_path}")


if __name__ == "__main__":
    main()
