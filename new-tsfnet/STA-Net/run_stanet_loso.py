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
    """returns (total_loss, class_ce, acc): size-weighted epoch means over the (dropout-on) training pass.
    total_loss = CE(final) + CE(eeg) + (1-PCC_fgsa1) + (1-PCC_fgsa2) + (1-PCC_egta); class_ce is the term
    used for early stopping / the stage-2 target (Keras `class_output_loss`)."""
    model.train()
    n, tot_sum, ce_sum, hit = 0, 0.0, 0.0, 0
    for xe, xf, y in batches(eeg, fnirs, labels, idx, bs, True, device):
        out = model(xe, xf)
        loss, c = sta_loss(out, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot_sum += loss.item() * len(y); ce_sum += c.item() * len(y); n += len(y)
        hit += int((out["p_final"].argmax(1) == y.argmax(1)).sum())
    return tot_sum / n, ce_sum / n, hit / n


def run_fold(eeg, fnirs, labels, sessions, trials, test_session, subject_id):
    test_session = int(test_session)
    dev = args.device
    tr_all = np.where(sessions != test_session)[0]
    te = np.where(sessions == test_session)[0]
    print(f"\n=== {subject_id} | test session {test_session} ===")
    if args.val_mode == "trial":
        tr, va = trial_level_split(trials, labels, tr_all, args.val_ratio, args.seed + test_session)
        print(f"  [val_mode] 'trial' -- {args.val_ratio:.0%} of the training TRIALS (stratified) held out for "
              f"early stopping: train {len(tr)} / val {len(va)} / test {len(te)} windows (stage 2 uses all {len(tr_all)})")
    else:
        vs = int(sessions[tr_all].min())
        tr, va = tr_all[sessions[tr_all] != vs], tr_all[sessions[tr_all] == vs]
        print(f"  [val_mode] 'cross_session' -- training session {vs} is the validation set: "
              f"train {len(tr)} / val {len(va)} / test {len(te)} windows (stage 2 uses all {len(tr_all)})")
    print("  [stanet] EGTA temporal alignment (cross-attention over the 11 lagged fNIRS windows) + FGSA spatial "
          "alignment; final decision = softmax-weighted fusion of the fusion and fNIRS heads")

    set_seed(args.seed * 100 + test_session)
    model = STANet(NUM_CLASSES, seg_len=fnirs.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)      # keras Adam epsilon

    best_val, best_ep, best_state, best_train_ce, wait = float("inf"), 0, None, None, 0
    stopped_early = False
    for ep in range(1, args.stage1_max_epochs + 1):
        tr_loss, tr_ce, tr_acc = train_epoch(model, opt, eeg, fnirs, labels, tr, dev, args.batch_size)
        va_ce, va_pred, _, _, va_true = evaluate(model, eeg, fnirs, labels, va, dev)
        va_acc, va_kap = float((va_pred == va_true).mean()), float(cohen_kappa(va_true, va_pred))
        if va_ce < best_val:
            best_val, best_ep, best_train_ce, wait = va_ce, ep, tr_ce, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        print(f"  [stage1] epoch {ep:3d} | train_loss={tr_loss:.4f} train_ce={tr_ce:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_ce:.4f} val_acc={va_acc:.4f} val_kappa={va_kap:.4f}")
        if wait >= args.stage1_patience:
            print(f"  [stage1] early stop at epoch {ep} (no improvement for {args.stage1_patience} epochs)")
            stopped_early = True
            break
    if not stopped_early:
        print(f"  [stage1] reached max epochs ({args.stage1_max_epochs}) without early stopping")
    model.load_state_dict(best_state)
    print(f"  [stage1] best validation class-CE {best_val:.4f} at epoch {best_ep} restored "
          f"(train class-CE there = {best_train_ce:.4f}, the stage-2 target)")

    s2_epochs = 0
    if args.stage2_stop == "none":
        print(f"  [stage2] skipped (stage2_stop='none') -- using stage 1's best checkpoint (epoch {best_ep}) directly "
              f"as the final model.")
    else:
        budget = best_ep if args.stage2_stop == "epoch_count" else args.stage2_max_epochs
        reached = False
        for ep in range(1, budget + 1):
            tr_loss, tr_ce, tr_acc = train_epoch(model, opt, eeg, fnirs, labels, tr_all, dev, args.batch_size)
            s2_epochs = ep
            print(f"  [stage2] epoch {ep:3d} | train_loss={tr_loss:.4f} train_ce={tr_ce:.4f} train_acc={tr_acc:.4f} "
                  f"(target train_ce <= {best_train_ce:.4f})")
            if args.stage2_stop == "loss" and tr_ce <= best_train_ce:
                reached = True
                break
        why = ("train class-CE reached the stage-1 target" if reached else
               f"epoch budget exhausted ({budget})" if args.stage2_stop == "loss" else
               f"ran {budget} epochs (= stage-1 best epoch, stage2_stop='epoch_count')")
        print(f"  [stage2] stopped after {s2_epochs} epoch(s): {why}")

    # checkpoint = weights after stage 2, BEFORE any AdaBN (so replays can re-apply AdaBN identically)
    ckpt_path = os.path.join(RESULTS_DIR, f"{subject_id}_session{test_session}_best.pt")
    torch.save(model.state_dict(), ckpt_path)
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
    print(f"  -> test_acc={acc:.4f} test_kappa={kappa:.4f} | ablation: eeg={res['eeg_only_acc']:.3f} "
          f"fnirs={res['fnirs_only_acc']:.3f} fusion={res['fusion_only_acc']:.3f} | "
          f"weights: w_fusion={res['w_fusion']:.3f} w_fnirs={res['w_fnirs']:.3f}")
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
            print(f"  [diagnostics] skipped: {e}")
    fold_json = os.path.join(RESULTS_DIR, f"results_{subject_id}_session{test_session}.json")
    with open(fold_json, "w") as fh:
        json.dump({"subject_id": subject_id, "args": vars(args), **res}, fh, indent=2)
    print(f"  results saved to: {fold_json}")
    print(f"  checkpoint saved to: {ckpt_path}")
    return res


def _fmt_cm(cm):
    """3x3 confusion matrix as aligned text with per-class recall (same layout as run_all_subjects.py)."""
    out = ["          " + "".join(f"{('pred ' + n):>12}" for n in CLASS_NAMES)]
    for i, n in enumerate(CLASS_NAMES):
        tot = int(sum(cm[i]))
        rec = (cm[i][i] / tot * 100) if tot else float("nan")
        out.append(f"  true {n:<4}" + "".join(f"{int(v):>12d}" for v in cm[i]) + f"   | recall {rec:5.1f}%")
    return "\n".join(out)


def _prep_line(subject_dir, sid):
    """One-line provenance from the preprocessing run ({sid}_metadata.json), so the training log is
    self-describing: window counts, task period, ICA exclusions."""
    try:
        with open(os.path.join(subject_dir, f"{sid}_metadata.json")) as fh:
            m = json.load(fh)
        return (f"  [prep] eeg {m['eeg_shape']} fnirs {m['fnirs_shape']} | labels {m['labels_distribution']} "
                f"| per-session {m['samples_per_session']} | task_period {m['task_period_s']}s | "
                f"ICA excluded {m['ica']['excluded']} | PyPREP={m.get('pyprep_badch')}")
    except Exception as e:
        return f"  [prep] metadata unavailable ({type(e).__name__})"


def main():
    import statistics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"stanet_all_subjects_{RUN_TAG}.csv")
    json_path = os.path.join(RESULTS_DIR, f"stanet_all_subjects_{RUN_TAG}.json")
    fields = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa", "eeg_only_acc",
              "fnirs_only_acc", "fusion_only_acc", "w_fusion", "w_fnirs",
              "recall_0back", "recall_2back", "recall_3back", "elapsed_min", "error"]
    rows, all_cms = {}, []
    with CrashSafeLogger(os.path.join(RESULTS_DIR, f"stanet_run_log_{RUN_TAG}.txt"), experiment_tag="STANET_LOSO"):
        print(f"===== STA-Net LOSO started {RUN_TAG} (seed={args.seed}) =====")
        print(f"[CONFIG] val_mode={args.val_mode!r} (val_ratio={args.val_ratio} if 'trial') | batch={args.batch_size} "
              f"lr={args.lr} (Adam eps=1e-7) | stage1 max/patience={args.stage1_max_epochs}/{args.stage1_patience} "
              f"| stage2 max={args.stage2_max_epochs} stop={args.stage2_stop!r} | adabn={args.adabn} "
              f"(transductive; STA-Net default OFF) | diagnostics={args.diagnostics}")
        print(f"[PROTOCOL] subject-dependent, leave-one-session-out (3 folds/subject), 3 classes, "
              f"test session never used for early stopping / model selection / normalisation")
        print(f"Subjects: {len(SUBJECTS)} | device={args.device} | data_root={args.data_root}")
        print(f"Summary CSV : {csv_path}\nSummary JSON: {json_path}\nCheckpoints/per-subject JSON: {RESULTS_DIR}\n")
        run_t0 = time.time()
        for i, sid in enumerate(SUBJECTS, 1):
            sdir = os.path.join(args.data_root, sid)
            print(f"\n{'=' * 70}\n[{i}/{len(SUBJECTS)}] {sid}  ({time.time() - run_t0:.1f}s elapsed so far)\n{'=' * 70}")
            t0 = time.time()
            row = {k: "" for k in fields}; row["subject_id"] = sid
            if not os.path.isdir(sdir):
                print(f"  SKIP: {sdir} does not exist (preprocessing not run for this subject?)")
                row["status"] = "skipped_no_data"
                rows[sid] = row
            else:
                print(_prep_line(sdir, sid))
                try:
                    eeg, fnirs, labels, sessions, trials = load_subject(sdir, sid)
                    folds = [run_fold(eeg, fnirs, labels, sessions, trials, s, sid) for s in sorted(np.unique(sessions))]
                    accs, kaps = [f["acc"] for f in folds], [f["kappa"] for f in folds]
                    cm = np.sum([np.array(f["cm"]) for f in folds], axis=0)
                    all_cms.append(cm)
                    rec = [cm[c, c] / cm[c].sum() if cm[c].sum() else float("nan") for c in range(NUM_CLASSES)]
                    row.update(status="ok", mean_acc=np.mean(accs), std_acc=np.std(accs), mean_kappa=np.mean(kaps),
                               std_kappa=np.std(kaps), eeg_only_acc=np.mean([f["eeg_only_acc"] for f in folds]),
                               fnirs_only_acc=np.mean([f["fnirs_only_acc"] for f in folds]),
                               fusion_only_acc=np.mean([f["fusion_only_acc"] for f in folds]),
                               w_fusion=np.mean([f["w_fusion"] for f in folds]),
                               w_fnirs=np.mean([f["w_fnirs"] for f in folds]),
                               recall_0back=rec[0], recall_2back=rec[1], recall_3back=rec[2])
                    row["elapsed_min"] = round((time.time() - t0) / 60, 2)
                    sd = lambda k: float(np.std([f[k] for f in folds]))
                    summ_path = os.path.join(RESULTS_DIR, f"cross_session_{sid}_summary.json")
                    with open(summ_path, "w") as fh:
                        json.dump({"subject_id": sid, "args": vars(args), "session_results": folds,
                                   "final_average_accuracy": row["mean_acc"], "final_std_accuracy": row["std_acc"],
                                   "final_average_kappa": row["mean_kappa"], "final_std_kappa": row["std_kappa"],
                                   "branch_accuracy": {k: float(np.mean([f[k] for f in folds])) for k in
                                                       ("eeg_only_acc", "fnirs_only_acc", "fusion_only_acc")},
                                   "decision_weights": {"w_fusion_mean": row["w_fusion"], "w_fnirs_mean": row["w_fnirs"]},
                                   "confusion_matrix_summed": cm.tolist(), "per_class_recall": rec}, fh, indent=2)
                    print(f"\n{sid} FINAL: acc={row['mean_acc']:.4f}+-{row['std_acc']:.4f}  "
                          f"kappa={row['mean_kappa']:.4f}+-{row['std_kappa']:.4f}")
                    print(f"{sid} BRANCH ACCURACY (mean over 3 folds): eeg={row['eeg_only_acc']:.4f}+-{sd('eeg_only_acc'):.4f}  "
                          f"fnirs={row['fnirs_only_acc']:.4f}+-{sd('fnirs_only_acc'):.4f}  "
                          f"fusion={row['fusion_only_acc']:.4f}+-{sd('fusion_only_acc'):.4f}")
                    print(f"{sid} LEARNED DECISION WEIGHTS (mean over 3 folds, instance-adaptive softmax kappa, EEG head is "
                          f"not part of the decision): w_fusion={row['w_fusion']:.4f}  w_fnirs={row['w_fnirs']:.4f}")
                    print(f"Full cross-session summary saved to: {summ_path}")
                    print(f"\n[{sid}] DONE in {row['elapsed_min']:.1f} min | "
                          f"acc={row['mean_acc']:.4f}+-{row['std_acc']:.4f}  kappa={row['mean_kappa']:.4f}+-{row['std_kappa']:.4f} | "
                          f"branch acc: eeg={row['eeg_only_acc']:.4f} fnirs={row['fnirs_only_acc']:.4f} "
                          f"fusion={row['fusion_only_acc']:.4f} | "
                          f"decision weights: w_fusion={row['w_fusion']:.4f} w_fnirs={row['w_fnirs']:.4f}")
                    print("  confusion matrix (summed over 3 session-folds):")
                    print(_fmt_cm(cm))
                    del eeg, fnirs
                except Exception as e:
                    row.update(status="error", error=str(e), elapsed_min=round((time.time() - t0) / 60, 2))
                    print(f"\n[{sid}] FAILED after {row['elapsed_min']:.1f} min: {e}")
                    traceback.print_exc()
                rows[sid] = row
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as fh:     # written immediately: survives a crash
                w = csv.DictWriter(fh, fieldnames=fields)
                if new: w.writeheader()
                w.writerow(row)
            with open(json_path, "w") as fh:
                json.dump(rows, fh, indent=2, default=float)

        ok = [r for r in rows.values() if r["status"] == "ok"]
        print(f"\n\n{'=' * 70}\nSTA-Net RUN COMPLETE: {len(ok)}/{len(SUBJECTS)} subjects finished successfully\n{'=' * 70}")
        if ok:
            col = lambda k: [r[k] for r in ok]
            print(f"Grand mean accuracy across subjects: {statistics.mean(col('mean_acc')):.4f} (+-{statistics.pstdev(col('mean_acc')):.4f})")
            print(f"Grand mean kappa across subjects:    {statistics.mean(col('mean_kappa')):.4f} (+-{statistics.pstdev(col('mean_kappa')):.4f})")
            print(f"Grand mean branch accuracy:  eeg={statistics.mean(col('eeg_only_acc')):.4f}  "
                  f"fnirs={statistics.mean(col('fnirs_only_acc')):.4f}  fusion={statistics.mean(col('fusion_only_acc')):.4f}")
            print(f"Grand mean learned decision weights:  w_fusion={statistics.mean(col('w_fusion')):.4f}  "
                  f"w_fnirs={statistics.mean(col('w_fnirs')):.4f}")
        if all_cms:
            g = np.sum(all_cms, axis=0)
            tot, diag = int(g.sum()), int(np.trace(g))
            print(f"\n{'=' * 70}\nGRAND CONFUSION MATRIX (summed over {len(all_cms)} subjects x 3 folds)\n{'=' * 70}")
            print(_fmt_cm(g))
            print(f"\n  overall accuracy from matrix: {diag / tot * 100:.2f}%  (n={tot} windows)")
            errs, hard = tot - diag, int(g[1, 2] + g[2, 1])
            if errs > 0:
                print(f"  2-back <-> 3-back confusions: {hard} of {errs} total errors ({hard / errs * 100:.1f}% of all errors)")
        failed = [sid for sid, r in rows.items() if r["status"] not in ("ok",)]
        if failed:
            print(f"\nSubjects that did NOT complete: {failed}")
        print(f"\n{'=' * 70}\nPER-SUBJECT RESULTS ({len(ok)} completed / {len(SUBJECTS)} total)\n{'=' * 70}")
        for sid in SUBJECTS:
            r = rows.get(sid)
            if r is None:
                print(f"{sid}: NOT RUN")
            elif r["status"] == "ok":
                print(f"{sid}: acc={r['mean_acc']:.4f} +- {r['std_acc']:.4f}  kappa={r['mean_kappa']:.4f} +- {r['std_kappa']:.4f}  |  "
                      f"branch acc: eeg={r['eeg_only_acc']:.4f} fnirs={r['fnirs_only_acc']:.4f} fusion={r['fusion_only_acc']:.4f}  |  "
                      f"decision weights: w_fusion={r['w_fusion']:.4f} w_fnirs={r['w_fnirs']:.4f}")
            elif r["status"] == "skipped_no_data":
                print(f"{sid}: SKIPPED (no preprocessed data)")
            else:
                print(f"{sid}: FAILED ({r.get('error', 'unknown error')})")
        print(f"\nFull results: {csv_path}\n             {json_path}")


if __name__ == "__main__":
    main()
