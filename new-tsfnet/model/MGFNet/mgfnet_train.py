"""
mgfnet_train.py -- training + reporting for MGFNet under the
leave-one-session-out (LOSO) cross-session protocol.

TRAINING = PAPER (Sec. 3.4-3.5)
  * Adam, lr = 1e-3, weight decay = 1e-3, constant LR (no decay, no warm-up)
  * batch size 64, 60 epochs for n-back, NO early stopping
  * deep-supervision loss: 0.4 * CE(hybrid) + 0.1 * CE(each of 6 branches)
  * five-fold cross-validation at the TRIAL level inside the training data,
    used only for validation / model selection; the test data are never seen

EVALUATION = LOSO (the one deliberate deviation from the paper)
  For each subject and each held-out session s in {1,2,3}:
    training data = the two other sessions, test data = session s.

MODEL SELECTION ("--selection", AMBIGUITY-T1)
  The paper says the 5-fold CV inside the training subset was used "for
  validation and model selection" but not WHAT was selected. With a fixed LR
  and a fixed epoch budget, the only free training choice is how long to
  train, so:
    cv5   (default): train one model per CV fold for the full 60 epochs,
          average the validation accuracy curves over the 5 folds, pick the
          epoch with the best mean validation accuracy (ties -> lower mean
          validation loss), then retrain ONE model on ALL training data for
          that many epochs and evaluate it once on the held-out session.
    fixed: train on all training data for exactly 60 epochs and evaluate the
          final model (no validation at all).

NORMALISATION: per-channel z-score fitted on the training windows of the
model being trained (CV-fold train part, or all training data for the final
model) and applied unchanged to validation / test windows.

REPRODUCIBILITY: every model gets its own seed derived from
(base seed, subject, held-out session, CV fold), so a subject run alone
reproduces exactly what it gives inside the full 26-subject sweep.
"""
import json
import math
import os
import zlib

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from mgfnet_model import MGFNet, mgfnet_deep_supervision_loss
from mgfnet_dataset import (MGFNBackDataset, ChannelNormalizer, _unpack_mgf_batch,
                            session_holdout_split, stratified_trial_kfold)

BRANCH_KEYS = ("1", "2", "3", "12", "13", "23")
CLASS_NAMES = ("0-back", "2-back", "3-back")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def derive_seed(base_seed, subject_id, test_session, fold):
    """Deterministic (not Python's salted hash) per-model seed.
    fold = 0..4 for CV models, -1 for the final model."""
    key = f"{base_seed}|{subject_id}|{int(test_session)}|{int(fold)}".encode()
    return zlib.crc32(key) & 0x7FFFFFFF


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(ds, idx, batch_size, shuffle, seed):
    sub = Subset(ds, np.asarray(idx))
    g = torch.Generator()
    g.manual_seed(seed)
    # Drop the last batch only if it would contain a single sample, which
    # BatchNorm cannot normalise in training mode.
    drop_last = shuffle and (len(sub) % batch_size == 1)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      generator=g if shuffle else None, drop_last=drop_last)


# ---------------------------------------------------------------------------
# Metrics (paper Sec. 3.5: Accuracy, Precision, Recall, F1 + ITR)
# ---------------------------------------------------------------------------
def cohen_kappa(y_true, y_pred):
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    if n == 0:
        return float("nan")
    po = (y_true == y_pred).mean()
    pe = 0.0
    for c in classes:
        pe += (y_true == c).mean() * (y_pred == c).mean()
    denom = 1 - pe
    return float((po - pe) / denom) if denom > 1e-8 else 0.0


def confusion_matrix_np(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def macro_prf(cm):
    """Macro-averaged precision / recall / F1 from a confusion matrix
    (rows = true, cols = predicted). A class with no predictions (or no
    samples) contributes 0 to that metric."""
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    pred_tot = cm.sum(axis=0)
    true_tot = cm.sum(axis=1)
    prec = np.divide(tp, pred_tot, out=np.zeros_like(tp), where=pred_tot > 0)
    rec = np.divide(tp, true_tot, out=np.zeros_like(tp), where=true_tot > 0)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(prec.mean()), float(rec.mean()), float(f1.mean())


def itr_bits_per_min(acc, n_classes=3, decision_time_s=5.0):
    """Paper Eqs. 6-7 (Wolpaw ITR). Below chance level the ITR is 0."""
    n, p = n_classes, float(acc)
    if p <= 1.0 / n:
        return 0.0
    if p >= 1.0:
        bits = math.log2(n)
    else:
        bits = math.log2(n) + p * math.log2(p) + (1 - p) * math.log2((1 - p) / (n - 1))
    return bits * 60.0 / decision_time_s


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def _to_device(batch, device, norm):
    x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
    x_eeg, x_hbo, x_hbr, y = (x_eeg.to(device), x_hbo.to(device),
                              x_hbr.to(device), y.to(device))
    x_eeg, x_hbo, x_hbr = norm(x_eeg, x_hbo, x_hbr)
    return x_eeg, x_hbo, x_hbr, y


@torch.no_grad()
def evaluate_loss_acc(model, loader, device, norm, lambda_hyb, lambda_aux):
    model.eval()
    tot_loss, n, correct = 0.0, 0, 0
    for batch in loader:
        x_eeg, x_hbo, x_hbr, y = _to_device(batch, device, norm)
        y_hyb, y_aux, _ = model(x_eeg, x_hbo, x_hbr)
        loss = mgfnet_deep_supervision_loss(y_hyb, y_aux, y, lambda_hyb, lambda_aux)
        tot_loss += loss.item() * y.size(0)
        n += y.size(0)
        correct += (y_hyb.argmax(dim=-1) == y).sum().item()
    return tot_loss / max(n, 1), correct / max(n, 1)


def train_model(model, train_loader, device, norm, n_epochs, lr=1e-3,
                weight_decay=1e-3, lambda_hyb=0.4, lambda_aux=0.1,
                val_loader=None, log_fn=print, log_prefix=""):
    """Fixed-budget training (paper: constant LR, no early stopping).
    Returns the per-epoch history; validation is only MONITORED."""
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, n_seen, n_correct = 0.0, 0, 0
        for batch in train_loader:
            x_eeg, x_hbo, x_hbr, y = _to_device(batch, device, norm)
            opt.zero_grad(set_to_none=True)
            y_hyb, y_aux, _ = model(x_eeg, x_hbo, x_hbr)
            loss = mgfnet_deep_supervision_loss(y_hyb, y_aux, y, lambda_hyb, lambda_aux)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{log_prefix} non-finite loss at epoch {epoch}")
            loss.backward()
            opt.step()
            ep_loss += loss.item() * y.size(0)
            n_seen += y.size(0)
            n_correct += (y_hyb.argmax(dim=-1) == y).sum().item()
        rec = {"epoch": epoch, "train_loss": ep_loss / max(n_seen, 1),
               "train_acc": n_correct / max(n_seen, 1)}
        msg = (f"  {log_prefix}[epoch {epoch:3d}] train_loss={rec['train_loss']:.4f} "
               f"train_acc={rec['train_acc']:.4f}")
        if val_loader is not None:
            rec["val_loss"], rec["val_acc"] = evaluate_loss_acc(
                model, val_loader, device, norm, lambda_hyb, lambda_aux)
            msg += f" val_loss={rec['val_loss']:.4f} val_acc={rec['val_acc']:.4f}"
        history.append(rec)
        log_fn(msg)
    return history


def _branch_ensemble_acc(y_true, logits_list):
    probs = sum(torch.softmax(l, dim=-1) for l in logits_list) / len(logits_list)
    return float((probs.argmax(dim=-1).numpy() == y_true).mean())


@torch.no_grad()
def detailed_test_report_mgfnet(model, test_loader, device, norm, num_classes=3,
                                decision_time_s=5.0):
    model.eval()
    all_true, all_hyb = [], []
    branch_logits = {k: [] for k in model.active_keys}
    all_gate, all_sel, all_C = [], [], []
    for batch in test_loader:
        x_eeg, x_hbo, x_hbr, y = _to_device(batch, device, norm)
        y_hyb, y_aux, aux = model(x_eeg, x_hbo, x_hbr)
        all_true.append(y.cpu().numpy())
        all_hyb.append(y_hyb.argmax(dim=-1).cpu().numpy())
        for k in model.active_keys:
            branch_logits[k].append(y_aux[k].cpu())
        if aux.get("gate_weights") is not None:
            all_gate.append(aux["gate_weights"].cpu().numpy())
            all_sel.append(aux["component_selection"].cpu().numpy())
            all_C.append(aux["coupling"].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_hyb)
    acc = float((y_pred == y_true).mean())
    cm = confusion_matrix_np(y_true, y_pred, num_classes)
    prec, rec, f1 = macro_prf(cm)

    branch_cat = {k: torch.cat(v, dim=0) for k, v in branch_logits.items()}
    branch_acc = {k: float((v.argmax(dim=-1).numpy() == y_true).mean())
                  for k, v in branch_cat.items()}
    fnirs_parts = [branch_cat[k] for k in ("2", "3") if k in branch_cat]
    cross_parts = [branch_cat[k] for k in ("12", "13", "23") if k in branch_cat]

    cgscr = {f"cgscr_w_{k}": None for k in BRANCH_KEYS}
    cgscr.update({"cgscr_max_cv": None, "cgscr_is_per_sample": None,
                  "cgscr_selection_hist": None, "cgscr_selection_entropy_bits": None,
                  "cgscr_coupling_mean": None, "cgscr_tau": None})
    if all_gate:
        gate = np.concatenate(all_gate, axis=0)              # (N, 6)
        g_mean, g_std = gate.mean(axis=0), gate.std(axis=0)
        g_cv = g_std / np.maximum(g_mean, 1e-8)
        for i, k in enumerate(BRANCH_KEYS):
            cgscr[f"cgscr_w_{k}"] = float(g_mean[i])
        cgscr["cgscr_max_cv"] = float(g_cv.max())
        cgscr["cgscr_is_per_sample"] = bool(g_cv.max() > 0.01)
        sel = np.concatenate(all_sel, axis=0)                # (N, 15) one-hot
        hist = sel.mean(axis=0)
        nz = hist[hist > 0]
        cgscr["cgscr_selection_hist"] = hist.round(4).tolist()
        cgscr["cgscr_selection_entropy_bits"] = float(-(nz * np.log2(nz)).sum())
        cgscr["cgscr_coupling_mean"] = np.concatenate(all_C, axis=0).mean(axis=0).round(4).tolist()
        cgscr["cgscr_tau"] = float(model.cgscr.tau.item())

    return {
        "test_accuracy_final": acc,
        "precision_macro": prec, "recall_macro": rec, "f1_macro": f1,
        "kappa_final": cohen_kappa(y_true, y_pred),
        "itr_bits_per_min": itr_bits_per_min(acc, num_classes, decision_time_s),
        "confusion_matrix_final": cm.tolist(),
        "n_test_windows": int(len(y_true)),
        "ablation": {
            "eeg_only": branch_acc.get("1"),
            "fnirs_only": _branch_ensemble_acc(y_true, fnirs_parts) if fnirs_parts else None,
            "fusion_only": _branch_ensemble_acc(y_true, cross_parts) if cross_parts else None,
            "final_weighted": acc,
            "per_branch_raw": branch_acc,
        },
        "learned_weights": {"w_eeg": None, "w_fusion": None, "w_fnirs": None, **cgscr},
    }


def select_epoch_cv(curves_acc, curves_loss):
    """Best epoch (1-based) from the fold-averaged validation curves:
    highest mean accuracy; ties broken by the lower mean loss, then by the
    earlier epoch."""
    mean_acc = np.mean(np.asarray(curves_acc), axis=0)
    mean_loss = np.mean(np.asarray(curves_loss), axis=0)
    best = np.flatnonzero(np.isclose(mean_acc, mean_acc.max(), rtol=0, atol=1e-12))
    best = best[np.argmin(mean_loss[best])]
    return int(best) + 1, mean_acc, mean_loss


# ---------------------------------------------------------------------------
# LOSO driver
# ---------------------------------------------------------------------------
def run_fold_mgfnet(full, train_idx, test_idx, subject_id, test_session, device,
                    model_kwargs, base_seed=42, batch_size=64, max_epochs=60,
                    lr=1e-3, weight_decay=1e-3, lambda_hyb=0.4, lambda_aux=0.1,
                    selection="cv5", n_cv_folds=5, ckpt_path=None, log_fn=print):
    hp = dict(lr=lr, weight_decay=weight_decay, lambda_hyb=lambda_hyb, lambda_aux=lambda_aux)
    labels_np = full.labels.numpy()
    cv_info = None

    if selection == "cv5":
        folds = stratified_trial_kfold(full.trial_ids[train_idx], labels_np[train_idx],
                                       n_splits=n_cv_folds,
                                       seed=derive_seed(base_seed, subject_id, test_session, 99))
        curves_acc, curves_loss = [], []
        for k, (tr_pos, va_pos) in enumerate(folds):
            seed = derive_seed(base_seed, subject_id, test_session, k)
            set_seed(seed)
            tr, va = train_idx[tr_pos], train_idx[va_pos]
            norm = ChannelNormalizer.fit(full, tr)
            model = MGFNet(**model_kwargs)
            hist = train_model(model, _make_loader(full, tr, batch_size, True, seed), device,
                               norm, max_epochs,
                               val_loader=_make_loader(full, va, batch_size, False, seed),
                               log_fn=log_fn, log_prefix=f"cv{k + 1}/{n_cv_folds} ", **hp)
            curves_acc.append([h["val_acc"] for h in hist])
            curves_loss.append([h["val_loss"] for h in hist])
            del model
        n_epochs, mean_acc, mean_loss = select_epoch_cv(curves_acc, curves_loss)
        cv_info = {"selected_epoch": n_epochs,
                   "cv_mean_val_acc_at_selected": float(mean_acc[n_epochs - 1]),
                   "cv_mean_val_acc_curve": mean_acc.round(5).tolist(),
                   "cv_mean_val_loss_curve": mean_loss.round(5).tolist()}
        log_fn(f"  [cv] selected epoch {n_epochs} "
               f"(mean val acc {mean_acc[n_epochs - 1]:.4f} over {n_cv_folds} folds)")
    elif selection == "fixed":
        n_epochs = max_epochs
    else:
        raise ValueError(f"unknown selection mode {selection!r}")

    # Final model: ALL training sessions, selected epoch budget, tested once.
    seed = derive_seed(base_seed, subject_id, test_session, -1)
    set_seed(seed)
    norm = ChannelNormalizer.fit(full, train_idx)
    model = MGFNet(**model_kwargs)
    hist = train_model(model, _make_loader(full, train_idx, batch_size, True, seed), device,
                       norm, n_epochs, log_fn=log_fn, log_prefix="final ", **hp)
    if ckpt_path:
        torch.save({"model_state_dict": model.state_dict(),
                    "normalizer": norm.state_dict(),
                    "model_kwargs": model_kwargs,
                    "test_session": int(test_session),
                    "n_epochs": n_epochs}, ckpt_path)

    report = detailed_test_report_mgfnet(
        model, _make_loader(full, test_idx, batch_size, False, seed), device, norm)
    report.update({"selection": selection, "epochs_trained": n_epochs,
                   "final_train_acc": hist[-1]["train_acc"] if hist else None,
                   "final_train_loss": hist[-1]["train_loss"] if hist else None,
                   "cv": cv_info, "history_final": hist})
    return report, model


def run_subject_loso_mgfnet(subject_dir, subject_id, device, batch_size=64,
                            results_dir=None, model_kwargs=None, train_kwargs=None,
                            base_seed=42, log_fn=print):
    """Leave-one-session-out loop for one subject."""
    train_kwargs = dict(train_kwargs or {})
    if results_dir is None:
        results_dir = subject_dir
    os.makedirs(results_dir, exist_ok=True)

    full = MGFNBackDataset(subject_dir, subject_id)
    # Channel counts always come from the data on disk, so the model can
    # never be built for a different channel layout than the one it gets.
    model_kwargs = dict(model_kwargs or {})
    model_kwargs.update(eeg_channels=full.eeg_channels, nirs_channels=full.nirs_channels)
    model_kwargs.setdefault("num_classes", 3)

    session_ids = full.session_ids
    sessions = sorted(np.unique(session_ids).tolist())
    if len(sessions) < 2:
        raise ValueError(f"{subject_id}: LOSO needs >= 2 sessions, found {sessions}")

    fold_reports = []
    for test_session in sessions:
        log_fn(f"\n=== {subject_id} | test session {test_session} (MGFNet, LOSO) ===")
        train_idx, test_idx = session_holdout_split(session_ids, test_session)
        assert not set(full.trial_ids[train_idx]) & set(full.trial_ids[test_idx]), \
            "a trial appears in both the training and the held-out session"
        ckpt_path = os.path.join(results_dir, f"{subject_id}_session{test_session}_mgf_final.pt")
        report, model = run_fold_mgfnet(full, train_idx, test_idx, subject_id, test_session,
                                        device, model_kwargs, base_seed=base_seed,
                                        batch_size=batch_size, ckpt_path=ckpt_path,
                                        log_fn=log_fn, **train_kwargs)
        report["test_session"] = int(test_session)
        report["train_sessions"] = [s for s in sessions if s != test_session]
        report["n_train_windows"] = int(len(train_idx))
        log_fn(f"  -> session {test_session}: acc={report['test_accuracy_final']:.4f} "
               f"P={report['precision_macro']:.4f} R={report['recall_macro']:.4f} "
               f"F1={report['f1_macro']:.4f} kappa={report['kappa_final']:.4f} "
               f"ITR={report['itr_bits_per_min']:.2f} bits/min")

        results_path = os.path.join(results_dir, f"results_{subject_id}_session{test_session}_mgf.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        fold_reports.append(report)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _ms(key):
        v = np.array([r[key] for r in fold_reports], dtype=float)
        return float(v.mean()), float(v.std())

    def _mean_or_none(path):
        vals = []
        for r in fold_reports:
            v = r
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    cm_summed = np.sum([np.asarray(r["confusion_matrix_final"]) for r in fold_reports], axis=0)
    row_tot = cm_summed.sum(axis=1)
    per_class_recall = [float(cm_summed[i, i] / row_tot[i]) if row_tot[i] else None
                        for i in range(cm_summed.shape[0])]

    mean_acc, std_acc = _ms("test_accuracy_final")
    summary = {
        "subject_id": subject_id,
        "protocol": "leave-one-session-out (cross-session)",
        "model_kwargs": model_kwargs, "train_kwargs": train_kwargs,
        "batch_size": batch_size, "base_seed": base_seed,
        "session_results": fold_reports,
        "final_average_accuracy": mean_acc, "final_std_accuracy": std_acc,
        "final_average_kappa": _ms("kappa_final")[0], "final_std_kappa": _ms("kappa_final")[1],
        "final_average_precision": _ms("precision_macro")[0],
        "final_average_recall": _ms("recall_macro")[0],
        "final_average_f1": _ms("f1_macro")[0],
        "final_average_itr": _ms("itr_bits_per_min")[0],
        "branch_accuracy": {
            "eeg_only_mean": _mean_or_none(["ablation", "eeg_only"]),
            "fnirs_only_mean": _mean_or_none(["ablation", "fnirs_only"]),
            "fusion_only_mean": _mean_or_none(["ablation", "fusion_only"]),
        },
        "cgscr_routing_stats": {
            **{f"cgscr_w_{k}_mean": _mean_or_none(["learned_weights", f"cgscr_w_{k}"])
               for k in BRANCH_KEYS},
            "cgscr_max_cv_max": (max(r["learned_weights"]["cgscr_max_cv"] for r in fold_reports)
                                 if model_kwargs.get("use_cgscr", True) else None),
        },
        "confusion_matrix_summed": cm_summed.tolist(),
        "per_class_recall": per_class_recall,
    }
    log_fn(f"\n{subject_id} FINAL (MGFNet, LOSO): acc={mean_acc:.4f}+-{std_acc:.4f}  "
           f"F1={summary['final_average_f1']:.4f}  kappa={summary['final_average_kappa']:.4f}")
    summary_path = os.path.join(results_dir, f"cross_session_{subject_id}_mgf_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log_fn(f"Full cross-session summary saved to: {summary_path}")
    return summary
