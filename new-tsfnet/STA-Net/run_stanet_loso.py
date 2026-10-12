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
p.add_argument("--weight-decay", type=float, default=0.0,
               help="L2 weight decay on the Adam optimiser. STA-Net's official Keras code uses NONE (0.0, the "
                    "default here). Set it to your TSFNet value (1e-4) for a matched-optimiser comparison.")
p.add_argument("--decay-scalars", choices=["on", "off"], default="off",
               help="Whether weight decay also applies to 1-D parameters: biases, BatchNorm weight/bias, and "
                    "STA-Net's own learned scalars (FGSA residual_para, EGTA positional embedding). 'off' (the "
                    "default, and the usual practice) exempts them, matching a TSFNet run with --no-decay-scalars. "
                    "Ignored when --weight-decay is 0.")
p.add_argument("--stop-metric", choices=["ce", "acc"], default="ce",
               help="Stage-1 early-stopping / model-selection criterion on the validation set. 'ce' = validation "
                    "class cross-entropy (the default; what STA-Net's own code tracks). 'acc' = validation "
                    "accuracy, to match a TSFNet run that selected on accuracy. Ties on 'acc' are broken by the "
                    "lower val CE, so the choice is deterministic.")
p.add_argument("--val-seed", type=int, default=None,
               help="Seed for the stratified trial-level validation split, overriding the default "
                    "(--seed + test_session). Set it to whatever formula your TSFNet run used so BOTH models hold "
                    "out the SAME trials. The log prints the chosen validation trial ids and their checksum for "
                    "every fold -- compare those two lines between the runs to VERIFY the splits are identical "
                    "instead of assuming it.")
p.add_argument("--fnirs-scale", type=float, default=None,
               help="Explicit fNIRS multiplier applied at load time, overriding the metadata's "
                    "FNIRS_TRAIN_SCALE (default 1e3 for mmol/L paper-exact tensors). Pass 1.0 when the "
                    "fNIRS is ALREADY z-scored by the preprocessing -- e.g. the TSFNet pipeline, whose "
                    "metadata.json has no FNIRS_TRAIN_SCALE key and would otherwise be scaled by 1000.")
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
def evaluate(model, eeg, fnirs, labels, idx, device, bs=64, return_total=False):
    model.eval()
    n, ce_sum, tot_sum, preds, branch, wsum = 0, 0.0, 0.0, [], {"eeg": [], "fusion": [], "fnirs": []}, np.zeros(2)
    for xe, xf, y in batches(eeg, fnirs, labels, idx, bs, False, device):
        out = model(xe, xf)
        tl, c = sta_loss(out, y)
        ce_sum += float(c) * len(y); tot_sum += float(tl) * len(y); n += len(y)
        preds.append(out["p_final"].argmax(1).cpu().numpy())
        branch["eeg"].append(out["p_eeg"].argmax(1).cpu().numpy())
        branch["fusion"].append(out["p_fusion"].argmax(1).cpu().numpy())
        branch["fnirs"].append(out["p_fnirs"].argmax(1).cpu().numpy())
        wsum += out["w"].sum(0).cpu().numpy()
    y_true = labels[idx]
    res = (ce_sum / n, np.concatenate(preds), {k: np.concatenate(v) for k, v in branch.items()}, wsum / n, y_true)
    return res + (tot_sum / n,) if return_total else res


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


def _split_checksum(trial_ids):
    """Short, order-independent fingerprint of a validation split, so the SAME line printed by the
    TSFNet run can be eyeballed against this one. Two runs that hold out the same trials print the
    same checksum; anything else means the splits differ and the comparison is confounded."""
    import hashlib
    return hashlib.sha1(",".join(str(int(t)) for t in sorted(trial_ids)).encode()).hexdigest()[:8]


def param_groups(model):
    """Adam parameter groups honouring --weight-decay / --decay-scalars.

    With --decay-scalars off (the default) every 1-D parameter is exempt from L2: biases,
    BatchNorm weight/bias, and STA-Net's two learned scalars -- FGSA's residual_para (the
    EEG-vs-guided-EEG balance, Eq. 3) and EGTA's positional embedding `pos`. Decaying those
    does not regularise anything; it just drags a learned balance toward zero and shrinks the
    positional code, which is why the usual practice -- and your TSFNet --no-decay-scalars run --
    exempts them. With weight decay 0 (STA-Net's official Keras setting) the split is a no-op and
    a single group is returned."""
    wd = args.weight_decay
    if wd == 0.0:
        return model.parameters()
    if args.decay_scalars == "on":
        return [{"params": list(model.parameters()), "weight_decay": wd}]
    # Exempt: every 1-D parameter (biases, BatchNorm weight/bias, FGSA's scalar residual_para) AND the
    # EGTA positional embedding, which is 3-D (1, seg_len, 512) and so would NOT be caught by the ndim
    # rule -- positional embeddings are conventionally decay-exempt for the same reason biases are: they
    # are a learned code, not a weight whose magnitude controls capacity, and decaying one just erases
    # the lag ordering EGTA needs.
    exempt_names = {"egta.pos"}
    decay, no_decay = [], []
    for name, prm in model.named_parameters():
        if not prm.requires_grad:
            continue
        (no_decay if (prm.ndim <= 1 or name in exempt_names) else decay).append(prm)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trainable



def run_fold(eeg, fnirs, labels, sessions, trials, test_session, subject_id):
    test_session = int(test_session)
    dev = args.device
    tr_all = np.where(sessions != test_session)[0]
    te = np.where(sessions == test_session)[0]
    print(f"\n=== {subject_id} | test session {test_session} ===")
    if args.val_mode == "trial":
        vseed = args.seed + test_session if args.val_seed is None else args.val_seed
        tr, va = trial_level_split(trials, labels, tr_all, args.val_ratio, vseed)
        print(f"  [val_mode] 'trial' -- {args.val_ratio:.0%} of the training TRIALS (stratified) held out for "
              f"early stopping: train {len(tr)} / val {len(va)} / test {len(te)} windows (stage 2 uses all {len(tr_all)})")
        vt = sorted(int(t) for t in np.unique(trials[va]))
        print(f"  [val_split] seed={vseed} | val trials {vt} | checksum {_split_checksum(vt)}   "
              f"<-- compare this line with the TSFNet run to verify both models hold out the SAME trials")
    else:
        vs = int(sessions[tr_all].min())
        tr, va = tr_all[sessions[tr_all] != vs], tr_all[sessions[tr_all] == vs]
        print(f"  [val_mode] 'cross_session' -- training session {vs} is the validation set: "
              f"train {len(tr)} / val {len(va)} / test {len(te)} windows (stage 2 uses all {len(tr_all)})")
    print(f"  [stanet] EGTA temporal alignment (cross-attention over the {fnirs.shape[1]} lagged fNIRS windows) "
          f"+ FGSA spatial alignment; final decision = softmax-weighted fusion of the fusion and fNIRS heads")

    set_seed(args.seed * 100 + test_session)
    model = STANet(NUM_CLASSES, seg_len=fnirs.shape[1]).to(dev)
    opt = torch.optim.Adam(param_groups(model), lr=args.lr, eps=1e-7)     # keras Adam epsilon

    # stage-1 model selection: 'ce' minimises val class-CE, 'acc' maximises val accuracy (ties -> lower CE).
    # Scoring as a tuple to MINIMISE keeps one comparison for both, with a deterministic tie-break either way.
    score_of = (lambda ce, acc: (ce,)) if args.stop_metric == "ce" else (lambda ce, acc: (-acc, ce))
    best_val, best_ep, best_state, best_train_ce, wait = (float("inf"),), 0, None, None, 0
    stopped_early = False
    for ep in range(1, args.stage1_max_epochs + 1):
        tr_loss, tr_ce, tr_acc = train_epoch(model, opt, eeg, fnirs, labels, tr, dev, args.batch_size)
        va_ce, va_pred, _, _, va_true, va_loss = evaluate(model, eeg, fnirs, labels, va, dev, return_total=True)
        va_acc, va_kap = float((va_pred == va_true).mean()), float(cohen_kappa(va_true, va_pred))
        if score_of(va_ce, va_acc) < best_val:
            best_val, best_ep, best_train_ce, wait = score_of(va_ce, va_acc), ep, tr_ce, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
        print(f"  [stage1] epoch {ep:3d} | train_loss={tr_loss:.4f} train_ce={tr_ce:.4f} train_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_ce={va_ce:.4f} val_acc={va_acc:.4f} val_kappa={va_kap:.4f}")
        if wait >= args.stage1_patience:
            print(f"  [stage1] early stop at epoch {ep} (no improvement for {args.stage1_patience} epochs)")
            stopped_early = True
            break
    if not stopped_early:
        print(f"  [stage1] reached max epochs ({args.stage1_max_epochs}) without early stopping")
    model.load_state_dict(best_state)
    _bs = f"val_ce {best_val[0]:.4f}" if args.stop_metric == "ce" else f"val_acc {-best_val[0]:.4f}"
    print(f"  [stage1] best {_bs} at epoch {best_ep} restored (selection metric = {args.stop_metric!r}; "
          f"train_ce there = {best_train_ce:.4f}, the stage-2 target)")

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


def _norm_str(n):
    """compact description of the preprocessing normalisation recorded in metadata.json (str or dict)."""
    if isinstance(n, dict):
        on = [k.replace("ENABLE_", "") for k in ("ENABLE_EEG_ZSCORE", "ENABLE_FNIRS_ZSCORE", "ENABLE_PER_SESSION_NORM") if n.get(k)]
        return (("+".join(on) if on else "none") + f" (scope={n.get('zscore_scope')}, per-session on {n.get('per_session_modalities')},"
                f" leak-free={n.get('leak_free_wrt_heldout_session')})")
    return str(n)


def _prep_line(subject_dir, sid):
    """One-line provenance from the preprocessing run ({sid}_metadata.json), so the training log is
    self-describing: window counts, task period, ICA exclusions. Tolerates metadata written by the
    TSFNet pipeline, whose key names differ and which has no FNIRS_TRAIN_SCALE / task_period_s."""
    try:
        with open(os.path.join(subject_dir, f"{sid}_metadata.json")) as fh:
            m = json.load(fh)
        scale = args.fnirs_scale if args.fnirs_scale is not None else m.get("FNIRS_TRAIN_SCALE", 1000.0)
        bits = [f"eeg {m['eeg_shape']} fnirs {m['fnirs_shape']}", f"labels {m['labels_distribution']}"]
        if "samples_per_session" in m:
            bits.append(f"per-session {m['samples_per_session']}")
        if "task_period_s" in m:                              # STA-Net preprocessing
            bits.append(f"task_period {m['task_period_s']}s")
        elif "slide_tmin_s" in m:                             # TSFNet preprocessing
            bits.append(f"slide [{m['slide_tmin_s']},{m['slide_tmax_s']}]s "
                        f"eeg_win {m.get('window_len_eeg_s')}s segs {m.get('fnirs_num_segments')}")
        if isinstance(m.get("ica"), dict):
            bits.append(f"ICA excluded {m['ica']['excluded']}")
        bits.append(f"PyPREP={m.get('pyprep_badch')}")
        bits.append(f"fNIRS train scale x{float(scale):g}"
                    + ("" if args.fnirs_scale is None else " (--fnirs-scale)"))
        bits.append(f"normalisation {_norm_str(m.get('normalization'))}")
        return "  [prep] " + " | ".join(bits)
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
        print(f"[OPTIM] weight_decay={args.weight_decay:g}"
              + (f" (decay_scalars={args.decay_scalars}: 1-D params -- biases, BatchNorm, FGSA residual_para, "
                 f"EGTA pos -- are {'INCLUDED' if args.decay_scalars == 'on' else 'EXEMPT'})"
                 if args.weight_decay else " (STA-Net official setting: no L2)")
              + f" | stage-1 selection metric = val_{args.stop_metric}"
              + (" (STA-Net default)" if args.stop_metric == "ce" else " (matched to a TSFNet accuracy-selected run)")
              + f" | val-split seed = {'seed + test_session' if args.val_seed is None else args.val_seed}")
        print(f"[LOSS] STA-Net's own objective, deliberately NOT replaced: CE(p_final) + CE(p_eeg) + "
              f"3x(1 - |PCC|). The comparison equalises the DATA and the PROTOCOL, never each method's "
              f"own objective -- swapping those would compare two different methods, not two architectures.")
        print(f"[PROTOCOL] subject-dependent, leave-one-session-out (3 folds/subject), 3 classes, "
              f"test session never used for early stopping / model selection / normalisation")
        print(f"Subjects: {len(SUBJECTS)} | device={args.device} | data_root={args.data_root}")
        try:   # parameter count, for the comparison table (seg_len does not change it except for `pos`)
            import stanet_common as _sc
            _probe = _sc.build_model(seg_len=11, device="cpu")
            _tot, _trn = count_params(_probe)
            print(f"[MODEL] STANet parameters: {_tot:,} total ({_trn:,} trainable) at seg_len=11; the EGTA "
                  f"positional embedding adds 512 per extra lagged window, so seg_len=21 is {_tot + 512 * 10:,}.")
            del _probe
        except Exception as _e:
            print(f"[MODEL] parameter count unavailable ({type(_e).__name__})")
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
                    eeg, fnirs, labels, sessions, trials = load_subject(sdir, sid,
                                                                        scale_override=args.fnirs_scale)
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
        # --- comparison manifest: every knob that must match the TSFNet run, in one file, so the
        # --- thesis table is filled from a record rather than from memory. Diff the two manifests.
        ok_rows = [r for r in rows.values() if r.get("status") == "ok"]
        manifest = {
            "model": "STA-Net", "run_tag": RUN_TAG, "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "data": {"data_root": args.data_root, "n_subjects_requested": len(SUBJECTS),
                     "n_subjects_ok": len(ok_rows), "fnirs_scale": args.fnirs_scale},
            "protocol": {"scheme": "subject-dependent leave-one-session-out", "folds_per_subject": 3,
                         "n_classes": NUM_CLASSES, "val_mode": args.val_mode, "val_ratio": args.val_ratio,
                         "val_seed": args.val_seed, "seed": args.seed,
                         "per_fold_seed_formula": "seed * 100 + test_session",
                         "adabn": args.adabn},
            "optimisation": {"optimizer": "Adam", "lr": args.lr, "adam_eps": 1e-7,
                             "batch_size": args.batch_size, "weight_decay": args.weight_decay,
                             "decay_scalars": args.decay_scalars,
                             "stage1_max_epochs": args.stage1_max_epochs,
                             "stage1_patience": args.stage1_patience,
                             "stage1_selection_metric": f"val_{args.stop_metric}",
                             "stage2_max_epochs": args.stage2_max_epochs, "stage2_stop": args.stage2_stop},
            "method_kept_unchanged": {
                "loss": "CE(p_final) + CE(p_eeg) + 3 * (1 - |PCC|)  -- STA-Net's own objective",
                "architecture": "STANet, unmodified",
                "temporal_kernels": "as published (13/stride 6, then 5/stride 2) -- designed for a 3 s EEG "
                                    "window and deliberately NOT retuned for a longer one",
            },
            "results": {
                "mean_acc": (statistics.mean(r["mean_acc"] for r in ok_rows) if ok_rows else None),
                "sd_acc_across_subjects": (statistics.stdev([r["mean_acc"] for r in ok_rows])
                                           if len(ok_rows) > 1 else None),
                "mean_kappa": (statistics.mean(r["mean_kappa"] for r in ok_rows) if ok_rows else None),
                "mean_eeg_only_acc": (statistics.mean(r["eeg_only_acc"] for r in ok_rows) if ok_rows else None),
                "mean_fnirs_only_acc": (statistics.mean(r["fnirs_only_acc"] for r in ok_rows) if ok_rows else None),
                "mean_fusion_only_acc": (statistics.mean(r["fusion_only_acc"] for r in ok_rows) if ok_rows else None),
            },
            "caveats_to_report": [
                "The EEG window is longer than the one STA-Net's kernels were designed for; the architecture "
                "was run unchanged, which is a deliberate choice and a limitation to state.",
                "Preprocessing z-score with scope=subject uses statistics that include the held-out session, "
                "and the per-session normalisation is transductive. Both are label-free and IDENTICAL for the "
                "two models, so the comparison is fair, but neither is strictly leak-free in absolute terms.",
                "Single seed unless repeated; report the seed and say so under limitations.",
            ],
        }
        man_path = os.path.join(RESULTS_DIR, f"comparison_manifest_{RUN_TAG}.json")
        with open(man_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"\nFull results: {csv_path}\n             {json_path}\n             {man_path}")


if __name__ == "__main__":
    main()
