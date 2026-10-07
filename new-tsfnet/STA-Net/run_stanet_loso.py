# ============================================================================
# STA-Net -- subject-dependent, CROSS-SESSION (leave-one-session-out) evaluation
# on the n-back dataset, 3 classes (0-back / 2-back / 3-back).
#
# Protocol (per subject, 3 folds): test = one whole session, train = the other two.
# Training follows the STA-Net paper / official run_sta_net.py:
#   stage 1: Adam lr=1e-3, batch 32, early stopping on validation class-CE
#            (patience 50, max 300 epochs), best weights restored;
#   stage 2: continue (same optimizer) on ALL training data until the epoch train
#            class-CE drops to the stage-1 train class-CE at the best epoch (max 200).
# Validation split (only used for stage-1 early stopping), --val-mode:
#   trial          : 20 % of the training TRIALS, stratified by class (no window leakage).
#                    The official script instead draws 80 random WINDOWS, which leaks
#                    because windows of one trial overlap -- not reproduced on purpose.
#   cross_session  : the lower-numbered training session is the validation set
#                    (same rule as your TSFNet run_all_subjects.py --val-mode cross_session).
# Test data is never used for early stopping, model selection or normalisation.
#
# Run:  python run_stanet_loso.py --data-root <Preproc output> --subjects 1,2,3 --seed 42
# ============================================================================
import os
import sys
import csv
import json
import time
import copy
import random
import datetime
import argparse
import traceback

import numpy as np
import torch

from stanet_model import STANet, sta_loss
from stanet_common import (NUM_CLASSES, CLASS_NAMES, load_subject, WindowLoader, cohen_kappa,
                           confusion_matrix_np as confusion, adapt_batchnorm_to_target)

try:
    from crash_safe_logger import CrashSafeLogger
except ImportError:
    class CrashSafeLogger:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

p = argparse.ArgumentParser()
p.add_argument("--data-root", default=r"E:\kntu\Thesis\codes\Results\STANET_PAPER_EXACT_nback")
p.add_argument("--results-dir", default=None, help="default: <data-root>/results_stanet_<val-mode>_seed<seed>")
p.add_argument("--subjects", default=None, help="e.g. '1,2,3' (default: 1..26)")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--val-mode", choices=["trial", "cross_session"], default="trial")
p.add_argument("--val-ratio", type=float, default=0.2, help="trial val-mode: fraction of training trials")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--stage1-max-epochs", type=int, default=300)
p.add_argument("--stage1-patience", type=int, default=50)
p.add_argument("--stage2-max-epochs", type=int, default=200)
p.add_argument("--stage2-stop", choices=["loss", "epoch_count", "none"], default="loss",
               help="loss = official; epoch_count = stage 2 runs best_epoch more epochs; none = skip stage 2")
p.add_argument("--adabn", action="store_true", default=False,
               help="AdaBN on the held-out session's UNLABELED windows before scoring (transductive; NOT part of "
                    "STA-Net). Only for matching a TSFNet run that used --adabn. The saved checkpoint is pre-AdaBN.")
p.add_argument("--diagnostics", action="store_true", default=False,
               help="run stanet_diagnostics.run_fold_diagnostics for the whitelisted subjects")
p.add_argument("--diagnostics-subjects", default=None, help="override whitelist, e.g. 'VP001,VP006' or 'ALL'")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args, _ = p.parse_known_args()

SUBJECTS = [f"VP{int(s):03d}" for s in (args.subjects.split(",") if args.subjects else range(1, 27))]
RESULTS_DIR = args.results_dir or os.path.join(args.data_root, f"results_stanet_{args.val_mode}_seed{args.seed}")
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}_{args.val_mode}"


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- data
def batches(eeg, fnirs, labels, idx, bs, shuffle, device):
    idx = np.random.permutation(idx) if shuffle else idx
    for i in range(0, len(idx), bs):
        b = torch.from_numpy(idx[i:i + bs])
        y = torch.nn.functional.one_hot(torch.from_numpy(labels[idx[i:i + bs]]), NUM_CLASSES).float()
        yield eeg[b].to(device), fnirs[b].to(device), y.to(device)


def trial_level_split(trial_ids, labels, idx, ratio, seed):
    """stratified-by-class split of whole trials (all windows of a trial stay together)."""
    rng = np.random.default_rng(seed)
    trials = np.unique(trial_ids[idx])
    t2l = {t: labels[idx][trial_ids[idx] == t][0] for t in trials}
    val_t = []
    for c in np.unique(list(t2l.values())):
        ct = [t for t in trials if t2l[t] == c]
        rng.shuffle(ct)
        val_t += ct[:max(1, int(round(len(ct) * ratio)))]
    m = np.isin(trial_ids[idx], val_t)
    return idx[~m], idx[m]


# ---------------------------------------------------------------- train / eval
@torch.no_grad()
def evaluate(model, eeg, fnirs, labels, idx, device, bs=64):
    model.eval()
    n, ce_sum, preds, branch, wsum = 0, 0.0, [], {"eeg": [], "fusion": [], "fnirs": []}, np.zeros(2)
    for xe, xf, y in batches(eeg, fnirs, labels, idx, bs, False, device):
        out = model(xe, xf)
        _, c = sta_loss(out, y)
        ce_sum += float(c) * len(y); n += len(y)
        preds.append(out["p_final"].argmax(1).cpu().numpy())
        branch["eeg"].append(out["p_eeg"].argmax(1).cpu().numpy())
        branch["fusion"].append(out["p_fusion"].argmax(1).cpu().numpy())
        branch["fnirs"].append(out["p_fnirs"].argmax(1).cpu().numpy())
        wsum += out["w"].sum(0).cpu().numpy()
    y_true = labels[idx]
    return ce_sum / n, np.concatenate(preds), {k: np.concatenate(v) for k, v in branch.items()}, wsum / n, y_true


def train_epoch(model, opt, eeg, fnirs, labels, idx, device, bs):
    model.train()
    n, cls_sum = 0, 0.0
    for xe, xf, y in batches(eeg, fnirs, labels, idx, bs, True, device):
        out = model(xe, xf)
        loss, c = sta_loss(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        cls_sum += c.item() * len(y); n += len(y)          # size-weighted epoch mean, as Keras
    return cls_sum / n


def run_fold(eeg, fnirs, labels, sessions, trials, test_session, subject_id):
    test_session = int(test_session)
    dev = args.device
    tr_all = np.where(sessions != test_session)[0]
    te = np.where(sessions == test_session)[0]
    if args.val_mode == "trial":
        tr, va = trial_level_split(trials, labels, tr_all, args.val_ratio, args.seed + test_session)
    else:
        vs = int(sessions[tr_all].min())
        tr, va = tr_all[sessions[tr_all] != vs], tr_all[sessions[tr_all] == vs]
    print(f"  fold test=S{test_session}: train {len(tr)} / val {len(va)} / test {len(te)} windows "
          f"(stage 2 uses all {len(tr_all)})")

    set_seed(args.seed * 100 + test_session)
    model = STANet(NUM_CLASSES, seg_len=fnirs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)      # keras Adam epsilon

    best_val, best_ep, best_state, best_train_ce, wait = float("inf"), 0, None, None, 0
    for ep in range(1, args.stage1_max_epochs + 1):
        tr_ce = train_epoch(model, opt, eeg, fnirs, labels, tr, dev, args.batch_size)
        va_ce = evaluate(model, eeg, fnirs, labels, va, dev)[0]
        if va_ce < best_val:
            best_val, best_ep, best_train_ce, wait = va_ce, ep, tr_ce, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    [s1] ep {ep:3d} train_ce {tr_ce:.4f} val_ce {va_ce:.4f} (best {best_val:.4f}@{best_ep})")
        if wait >= args.stage1_patience:
            print(f"    [s1] early stop at epoch {ep}; restoring epoch {best_ep}")
            break
    model.load_state_dict(best_state)

    s2_epochs = 0
    if args.stage2_stop != "none":
        budget = best_ep if args.stage2_stop == "epoch_count" else args.stage2_max_epochs
        for ep in range(1, budget + 1):
            tr_ce = train_epoch(model, opt, eeg, fnirs, labels, tr_all, dev, args.batch_size)
            s2_epochs = ep
            if args.stage2_stop == "loss" and tr_ce <= best_train_ce:
                break
        print(f"    [s2] {s2_epochs} epoch(s), last train_ce {tr_ce:.4f} (target {best_train_ce:.4f})")

    # checkpoint = weights after stage 2, BEFORE any AdaBN (so replays can re-apply AdaBN identically)
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f"{subject_id}_session{test_session}_best.pt"))
    if args.adabn:
        adapt_batchnorm_to_target(model, WindowLoader(eeg, fnirs, labels, te, 64), dev)
    _, pred, branch, w, y_true = evaluate(model, eeg, fnirs, labels, te, dev)
    acc, kappa = float((pred == y_true).mean()), float(cohen_kappa(y_true, pred))
    res = {"test_session": test_session, "acc": acc, "kappa": kappa,
           "cm": confusion(y_true, pred).tolist(),
           "eeg_only_acc": float((branch["eeg"] == y_true).mean()),
           "fusion_only_acc": float((branch["fusion"] == y_true).mean()),
           "fnirs_only_acc": float((branch["fnirs"] == y_true).mean()),
           "w_fusion": float(w[0]), "w_fnirs": float(w[1]),
           "best_epoch_s1": best_ep, "epochs_s2": s2_epochs,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te))}
    print(f"    TEST S{test_session}: acc {acc * 100:.2f}%  kappa {kappa:.3f}")
    if args.diagnostics:
        try:
            import stanet_diagnostics as D
            if args.diagnostics_subjects:
                D.DIAGNOSTIC_SUBJECTS = "ALL" if args.diagnostics_subjects == "ALL" else args.diagnostics_subjects.split(",")
            d = D.run_fold_diagnostics(model, WindowLoader(eeg, fnirs, labels, tr_all, 32),
                                       WindowLoader(eeg, fnirs, labels, te, 32), dev,
                                       subject_id, test_session, RESULTS_DIR)
            if d:
                res["diagnostics"] = {k: v for k, v in d.items() if k != "figures"}
        except Exception as e:
            print(f"    [diagnostics] skipped: {e}")
    return res


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"stanet_all_subjects_{RUN_TAG}.csv")
    fields = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa", "eeg_only_acc",
              "fnirs_only_acc", "fusion_only_acc", "w_fusion", "w_fnirs",
              "recall_0back", "recall_2back", "recall_3back", "elapsed_min", "error"]
    rows, total_cm = {}, np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    with CrashSafeLogger(os.path.join(RESULTS_DIR, f"stanet_run_log_{RUN_TAG}.txt"), experiment_tag="STANET_LOSO"):
        print(f"===== STA-Net LOSO started {RUN_TAG} | device {args.device} | args {vars(args)}")
        for i, sid in enumerate(SUBJECTS, 1):
            sdir = os.path.join(args.data_root, sid)
            print(f"\n{'=' * 70}\n[{i}/{len(SUBJECTS)}] {sid}\n{'=' * 70}")
            t0 = time.time()
            row = {k: "" for k in fields}; row["subject_id"] = sid
            try:
                eeg, fnirs, labels, sessions, trials = load_subject(sdir, sid)
                folds = [run_fold(eeg, fnirs, labels, sessions, trials, s, sid) for s in sorted(np.unique(sessions))]
                accs, kaps = [f["acc"] for f in folds], [f["kappa"] for f in folds]
                cm = np.sum([np.array(f["cm"]) for f in folds], axis=0)
                total_cm += cm
                rec = [cm[c, c] / cm[c].sum() if cm[c].sum() else float("nan") for c in range(NUM_CLASSES)]
                row.update(status="ok", mean_acc=np.mean(accs), std_acc=np.std(accs), mean_kappa=np.mean(kaps),
                           std_kappa=np.std(kaps), eeg_only_acc=np.mean([f["eeg_only_acc"] for f in folds]),
                           fnirs_only_acc=np.mean([f["fnirs_only_acc"] for f in folds]),
                           fusion_only_acc=np.mean([f["fusion_only_acc"] for f in folds]),
                           w_fusion=np.mean([f["w_fusion"] for f in folds]), w_fnirs=np.mean([f["w_fnirs"] for f in folds]),
                           recall_0back=rec[0], recall_2back=rec[1], recall_3back=rec[2])
                with open(os.path.join(RESULTS_DIR, f"stanet_{sid}_summary.json"), "w") as fh:
                    json.dump({"subject_id": sid, "args": vars(args), "folds": folds,
                               "mean_acc": row["mean_acc"], "mean_kappa": row["mean_kappa"]}, fh, indent=2)
                print(f"  {sid}: mean acc {row['mean_acc'] * 100:.2f}%  kappa {row['mean_kappa']:.3f}")
                del eeg, fnirs
            except Exception as e:
                row.update(status="error", error=str(e)); traceback.print_exc()
            row["elapsed_min"] = round((time.time() - t0) / 60, 2)
            rows[sid] = row
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                if new: w.writeheader()
                w.writerow(row)

        ok = [r for r in rows.values() if r["status"] == "ok"]
        if ok:
            a, k = np.array([r["mean_acc"] for r in ok]), np.array([r["mean_kappa"] for r in ok])
            print(f"\n===== OVERALL ({len(ok)} subjects): acc {a.mean() * 100:.2f} ± {a.std() * 100:.2f} %, "
                  f"kappa {k.mean():.3f} ± {k.std():.3f}")
            print("Summed confusion matrix (rows=true, cols=pred):\n", total_cm)


if __name__ == "__main__":
    main()
