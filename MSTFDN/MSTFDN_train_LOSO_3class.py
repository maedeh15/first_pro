"""
MSTFDN — Cross-Session LOSO Training
=====================================
Leave-One-Session-Out (LOSO) strategy as described in paper Sec. 3.3.1:

  "One strategy was leave-one-session-out (LOSO) testing. Since each task
   consists of three sessions, the data from one session for each participant
   were used as the test set, while the remaining two sessions served as the
   training set."

3 folds per subject:
  Fold 1: train=[S2,S3]  test=[S1]
  Fold 2: train=[S1,S3]  test=[S2]
  Fold 3: train=[S1,S2]  test=[S3]

Metrics per subject: Accuracy, Recall, Precision, F1-score (average of 3 folds)
Final result: average across all subjects

Output files saved in OUTPUT_PATH/LOSO/:
  results_per_subject.csv   — per-subject per-fold metrics
  results_summary.csv       — per-subject average + all-subject average
  best_model_VP{id}_fold{f}.pt — best checkpoint per subject per fold
  training_log.txt          — timestamped log
"""

import os
import gc
import sys
import json
import time
import datetime
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings('ignore')

# ── import model ──────────────────────────────────────────────────────────────
# فایل MSTFDN_model.py باید در همین پوشه باشد
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from MSTFDN_model_v2 import MSTFDN, MSTFDNConfig, MSTFDNLoss

# ===========================================================================
#  CONFIGURATION — تنظیم کن قبل از اجرا
# ===========================================================================
# آدرس پوشه‌ای که preprocessing خروجی‌ها را ذخیره کرده
# مثال Windows: r"E:\kntu\Thesis\codes\Results\MSTFDN_PREPROCESSED_NBACK"
# مثال Linux:   "/media/hassan-hossein/B660FEE360FEA8EF/Mahi/preprocessed"
PREPROCESSED_PATH = r"E:\kntu\Thesis\codes\Results\MSTFDN_PREPROCESSED_NBACK_V3"

# پوشه ذخیره نتایج
OUTPUT_PATH = r"E:\kntu\Thesis\codes\Results\MSTFDN_LOSO_V3_3CLASS"

# لیست همه subject ها
ALL_SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]

# اگر می‌خواهی فقط چند subject تست کنی:
# ALL_SUBJECTS = ["VP001", "VP002"]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================================================================
#  MODEL CONFIG (N-back dataset — از مقاله Table A.1)
# ===========================================================================
def get_config() -> MSTFDNConfig:
    cfg = MSTFDNConfig()
    cfg.n_eeg_ch    = 28
    cfg.n_fnirs_ch  = 36
    cfg.T_e         = 2000    # 10s x 200Hz
    cfg.T_f         = 100     # 10s x 10Hz -- single window, matches paper exactly
                              # (Sec 3.2: fNIRS sample = 3x36x100, no 21-segment stack)
    cfg.n_bands     = 5
    cfg.n_hemo      = 3       # HbO, HbR, HbT -- all supplied as model input
    cfg.n_class     = 3       # 0-back, 2-back, 3-back ONLY -- 3 cognitive
                              # workload levels. REST (label=3, from paper's
                              # 4-class protocol) is EXCLUDED here, filtered
                              # out at the dataset level (see
                              # MSTFDNSubjectDataset below) -- preprocessing
                              # v3 still extracts all 4 classes on disk, we
                              # just drop the rest-class windows at load time.
    cfg.He          = 8       # T_e/He = 2000/8 = 250 (exact)
    cfg.Hf          = 5       # T_f/Hf = 100/5 = 20 (exact)
    cfg.Hef         = 5
    cfg.Ne          = 1
    cfg.Nf          = 1
    cfg.Nef         = 1
    cfg.KSS         = 4
    cfg.ffn_dim     = 256
    cfg.dropout     = 0.4
    cfg.lambda_nw   = 0.02
    cfg.lambda_mhs  = 0.05
    cfg.lr          = 3e-4
    cfg.weight_decay = 1e-3
    cfg.batch_size  = 16
    cfg.max_epochs  = 1000
    cfg.warmup_epochs = 10
    cfg.eta_min_ratio = 0.05
    cfg.patience    = 50
    return cfg

# ===========================================================================
#  DATASET
# ===========================================================================
class MSTFDNSubjectDataset(Dataset):
    """
    یک subject را load می‌کند -- خروجی MSTFDN_preprocessing_v3.py.

    EEG:   (N, 5, 28, 2000) float16  -- 5 باند فرکانسی واقعی، channel-space
           (بدون 16x16 grid -- مقاله اصلاً spatial mapping ندارد)
    fNIRS: (N, 3, 36, 100)  float32  -- HbO/HbR/HbT، یک window 10 ثانیه‌ای
           (بدون 21-segment stack -- طبق Sec 3.2 مقاله: X_fnirs در R^(3xCHfxTf))
    labels روی دیسک: 0=0-back, 1=2-back, 2=3-back, 3=rest  (4 کلاس، خروجی preprocessing)
    session_ids: (N,)  values 1/2/3

    exclude_rest=True (پیش‌فرض): نمونه‌های rest (label==3) در همین‌جا حذف
    می‌شوند و فقط ۳ سطح بار شناختی (0/2/3-back) باقی می‌ماند -- بدون نیاز
    به اجرای دوباره preprocessing.

    حافظه: eeg به‌صورت float16 نگه داشته می‌شود، تبدیل به float32 فقط در
    __getitem__ برای یک window انجام می‌شود -- نه برای کل subject.
    """
    REST_LABEL_ON_DISK = 3   # must match REST_LABEL in MSTFDN_preprocessing_v3.py

    def __init__(self, data_dir: str, subject_id: str, exclude_rest: bool = True):
        sub_dir = os.path.join(data_dir, subject_id)
        eeg    = np.load(os.path.join(sub_dir, f"{subject_id}_eeg_windows.npy"))   # (N,5,28,2000) float16
        fnirs  = np.load(os.path.join(sub_dir, f"{subject_id}_fnirs_windows.npy")) # (N,3,36,100)  float32
        labels = np.load(os.path.join(sub_dir, f"{subject_id}_labels.npy"))
        sess   = np.load(os.path.join(sub_dir, f"{subject_id}_session_ids.npy"))

        # Sanity check: reject old-format files (16x16 grid or 21-segment
        # stack) loudly instead of silently reshaping garbage.
        if eeg.ndim != 4 or eeg.shape[1] != 5:
            raise ValueError(
                f"{subject_id}: expected eeg_windows shape (N,5,CH,T_e) from "
                f"MSTFDN_preprocessing_v3.py, got {eeg.shape}. Old preprocessing "
                f"versions (16x16 grid or broadcast-copy bands) are not compatible."
            )
        if fnirs.ndim != 4 or fnirs.shape[1] != 3:
            raise ValueError(
                f"{subject_id}: expected fnirs_windows shape (N,3,CH,T_f) from "
                f"MSTFDN_preprocessing_v3.py, got {fnirs.shape}. Old preprocessing "
                f"versions (21-segment stack, 16x16 grid) are not compatible."
            )

        # Drop REST windows -> 3-class problem (0-back, 2-back, 3-back).
        # Labels stay 0/1/2 unchanged since rest was already the LAST label
        # index (3); no relabeling needed for the remaining 3 classes.
        if exclude_rest:
            keep = labels != self.REST_LABEL_ON_DISK
            eeg    = eeg[keep]
            fnirs  = fnirs[keep]
            labels = labels[keep]
            sess   = sess[keep]

        # Keep EEG as float16 in host RAM; cast to float32 per-window in
        # __getitem__ instead of for the whole subject.
        self.eeg         = torch.from_numpy(eeg).half()
        self.fnirs       = torch.from_numpy(fnirs).float()
        self.labels      = torch.from_numpy(labels).long()
        self.session_ids = sess   # numpy array, kept for splitting
        # FIX 1 support: trial_ids are needed for val_mode='trial' (trial-level
        # split), so that sliding windows from the SAME trial can never
        # straddle the train/val boundary -- windows overlap ~90%, so a
        # window-level split would leak almost-identical windows into
        # validation and make it meaningless. Preprocessing already writes
        # this file; it simply was not being loaded here.
        _trial_path = os.path.join(sub_dir, f"{subject_id}_trial_ids.npy")
        if os.path.exists(_trial_path):
            _tr = np.load(_trial_path)
            if exclude_rest:
                _tr = _tr[keep]
            self.trial_ids = _tr
        else:
            self.trial_ids = None   # val_mode='trial' will raise a clear error

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Cast EEG to float32 here (single window, cheap) -- the model
        # expects float32 input; keeping the full dataset in float16 in
        # RAM is what actually saves memory.
        return self.eeg[idx].float(), self.fnirs[idx], self.labels[idx]


# ===========================================================================
#  LOGGING
# ===========================================================================
LOG_FILE = None

def log(msg: str):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ===========================================================================
#  TRAINING UTILITIES
# ===========================================================================
def warmup_lr(optimizer, epoch: int, cfg: MSTFDNConfig):
    if epoch < cfg.warmup_epochs:
        scale = (epoch + 1) / cfg.warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = cfg.lr * scale


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> dict:
    model.train()
    total_loss = ce_sum = l2_sum = mhs_sum = 0.0
    correct = n = 0

    for eeg, fnirs, labels in loader:
        eeg    = eeg.to(device)
        fnirs  = fnirs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, he, hf, hef = model(eeg, fnirs)
        loss, ce, l2, mhs   = loss_fn(logits, labels, he, hf, hef)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        B = labels.size(0)
        total_loss += loss.item() * B
        ce_sum     += ce.item()   * B
        l2_sum     += l2.item()   * B
        mhs_sum    += mhs.item()  * B
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += B

    return {'loss': total_loss/n, 'ce': ce_sum/n,
            'l2': l2_sum/n, 'mhs': mhs_sum/n, 'acc': correct/n}


def trial_level_train_val_split(trial_ids_tv, labels_tv, val_ratio=0.2, seed=42):
    """val_mode='trial' -- OPTION 1, matched to TSFNet's own split.

    Holds out `val_ratio` of the TRIALS (not windows) from each class, pooled
    across both training sessions. Splitting at the trial level is essential:
    windows slide with ~90% overlap, so a window-level split would put
    near-duplicate windows on both sides and validation would be
    meaningless.

    Use this when the GOAL IS A FAIR COMPARISON against TSFNet, because it
    gives MSTFDN exactly the same kind of validation signal TSFNet gets --
    adding no extra differing variable to the comparison.
    Known weakness (shared with TSFNet): with ~6 trials per class available,
    round(6*0.2)=1 trial per class -> a 3-trial validation set, drawn from
    the SAME sessions as training even though the test is cross-session.
    """
    rng = np.random.default_rng(seed)
    trial_ids_tv = np.asarray(trial_ids_tv); labels_tv = np.asarray(labels_tv)
    val_trials = []
    for cls in np.unique(labels_tv):
        cls_trials = np.unique(trial_ids_tv[labels_tv == cls])
        n_val = max(1, int(round(len(cls_trials) * val_ratio)))
        val_trials.extend(rng.choice(cls_trials, size=n_val, replace=False).tolist())
    val_mask = np.isin(trial_ids_tv, val_trials)
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def cross_session_train_val_split(session_ids_tv, val_session):
    """val_mode='cross_session' -- OPTION 2, scientifically stronger.

    Holds out one WHOLE training session as validation. Two advantages over
    option 1: (a) ~9 validation trials instead of ~3, and (b) validation is
    itself cross-session, matching the regime the model is actually SCORED
    under, so early stopping optimizes the right thing.

    Honest cost: the model then trains on ONE session instead of two --
    roughly half the training data. Which effect dominates is empirical,
    which is why this is a flag rather than a silent default.
    The validation session is chosen DETERMINISTICALLY (lowest-numbered of
    the two remaining), not randomly, so the data partition is identical
    across seeds and seed-to-seed variance reflects only model init / batch
    order -- never a different split.
    """
    session_ids_tv = np.asarray(session_ids_tv)
    val_mask = session_ids_tv == val_session
    tr, va = np.where(~val_mask)[0], np.where(val_mask)[0]
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(
            f"cross_session split empty for val_session={val_session} "
            f"(train={len(tr)}, val={len(va)}); sessions present: "
            f"{np.unique(session_ids_tv).tolist()}")
    return tr, va


def cohen_kappa(y_true, y_pred, num_classes=3):
    """Cohen's kappa, computed EXACTLY as TSFNet's own cohen_kappa does, so
    the two models' kappa numbers are directly comparable.

    NOTE: this is the real Cohen's kappa (uses the marginal distributions of
    both raters), NOT the simplified (acc - p0)/(1 - p0) with p0 = chance
    level that some papers in this area report. Mixing the two would make
    any cross-paper kappa comparison invalid.
    """
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    cm = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, pr in zip(y_true, y_pred):
        cm[int(t), int(pr)] += 1
    po = np.trace(cm) / n
    pe = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (n * n)
    return 0.0 if abs(1.0 - pe) < 1e-12 else float((po - pe) / (1.0 - pe))


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for eeg, fnirs, labels in loader:
        eeg    = eeg.to(device)
        fnirs  = fnirs.to(device)
        labels = labels.to(device)

        logits, he, hf, hef = model(eeg, fnirs)
        loss, *_ = loss_fn(logits, labels, he, hf, hef)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(logits.argmax(1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    n      = len(all_labels)
    acc    = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    prec   = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    f1     = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'loss': total_loss / n,
        'acc':  acc,
        'kappa': cohen_kappa(all_labels, all_preds),   # FIX 2: second acceptance metric
        'recall': recall,
        'precision': prec,
        'f1': f1,
        'preds':  all_preds,
        'labels': all_labels,
    }


def train_fold(model, cfg, train_loader, val_loader,
               checkpoint_path: str, device: str) -> dict:
    """
    Train one LOSO fold. Returns best val metrics.
    """
    loss_fn   = MSTFDNLoss(cfg, model)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs - cfg.warmup_epochs,
        eta_min=cfg.lr * cfg.eta_min_ratio)

    best_val_loss = float('inf')
    best_metrics  = {}
    patience_cnt  = 0

    for epoch in range(cfg.max_epochs):
        warmup_lr(optimizer, epoch, cfg)
        train_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics   = evaluate(model, val_loader, loss_fn, device)

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        if epoch % 20 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            log(f"    Ep{epoch:4d} | LR={lr_now:.2e} | "
                f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.4f} "
                f"F1={val_metrics['f1']:.4f}")

        # early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_metrics  = val_metrics.copy()
            patience_cnt  = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'val_metrics': val_metrics,
            }, checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                log(f"    Early stop at epoch {epoch}")
                break

    return best_metrics


# ===========================================================================
#  LOSO TRAINING FOR ONE SUBJECT
# ===========================================================================
def run_loso_subject(subject_id: str, cfg: MSTFDNConfig,
                     out_dir: str, device: str,
                     val_mode: str = "trial", seed: int = 42) -> list:
    """
    Run 3-fold LOSO for one subject.
    Returns list of 3 dicts (one per fold) with metrics.
    """
    log(f"\n{'='*55}")
    log(f"Subject: {subject_id}  |  LOSO (3 folds)")
    log(f"{'='*55}")

    # Load dataset
    dataset = MSTFDNSubjectDataset(
        data_dir=PREPROCESSED_PATH,
        subject_id=subject_id,
    )

    session_ids = dataset.session_ids   # numpy (N,) values 1/2/3
    sessions    = [1, 2, 3]
    fold_results = []

    for fold_idx, test_session in enumerate(sessions, 1):
        log(f"\n  Fold {fold_idx}/3 — Test session: S{test_session} | "
            f"Train sessions: {[s for s in sessions if s != test_session]}")

        # Split indices by session
        test_idx  = np.where(session_ids == test_session)[0]
        train_idx = np.where(session_ids != test_session)[0]

        log(f"  Train windows: {len(train_idx)} | Test windows: {len(test_idx)}")

        if len(train_idx) == 0 or len(test_idx) == 0:
            log(f"  WARNING: empty split for fold {fold_idx}, skipping")
            continue

        # ===== FIX 1 (CRITICAL -- DATA LEAK): build a REAL validation set =====
        # Previously test_loader was passed to train_fold as the validation
        # loader, so the best epoch was selected by evaluating on the TEST
        # set every epoch and keeping the best one. The reported number was
        # therefore "max test accuracy reached during training", not
        # performance on unseen data -- systematically optimistic and NOT
        # comparable to TSFNet's number (which uses a held-out validation
        # set). Validation is now carved out of the TRAINING sessions only;
        # the test session is never looked at until the final evaluation.
        if val_mode == "trial":
            if dataset.trial_ids is None:
                raise ValueError(
                    f"val_mode='trial' needs {subject_id}_trial_ids.npy, which was not "
                    f"found. Either re-run preprocessing (it does write this file) or "
                    f"use --val-mode cross_session.")
            tv_trials = dataset.trial_ids[train_idx]
            tv_labels = dataset.labels.numpy()[train_idx]
            sub_tr, sub_va = trial_level_train_val_split(tv_trials, tv_labels, seed=seed)
            log(f"  [val_mode] 'trial' -- {len(np.unique(tv_trials[sub_va]))} validation "
                f"trial(s) held out from the pooled training sessions "
                f"({len(sub_tr)} train / {len(sub_va)} val windows). Matches TSFNet's "
                f"own split, so the comparison adds no extra differing variable.")
        else:  # 'cross_session'
            tv_sessions = np.unique(session_ids[train_idx])
            val_session = int(tv_sessions.min())
            sub_tr, sub_va = cross_session_train_val_split(session_ids[train_idx], val_session)
            log(f"  [val_mode] 'cross_session' -- validating on session {val_session}, "
                f"training on {[int(s) for s in tv_sessions if s != val_session]} "
                f"({len(sub_tr)} train / {len(sub_va)} val windows). Validation is itself "
                f"cross-session, matching the test regime; training data is halved.")

        fit_idx = train_idx[sub_tr]
        val_idx = train_idx[sub_va]

        train_ds = Subset(dataset, fit_idx)
        val_ds   = Subset(dataset, val_idx)
        test_ds  = Subset(dataset, test_idx)

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)

        # Fresh model for each fold (paper: "trained from scratch")
        model    = MSTFDN(cfg).to(device)
        ckpt_path = os.path.join(out_dir, f"best_model_{subject_id}_fold{fold_idx}.pt")

        t0 = time.time()
        # Model selection uses VALIDATION only...
        train_fold(model, cfg, train_loader, val_loader, ckpt_path, device)
        # ...then the best checkpoint is scored ONCE on the untouched test set.
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        best_metrics = evaluate(model, test_loader, MSTFDNLoss(cfg, model), device)
        elapsed = (time.time() - t0) / 60

        # FIX 6: PER-CLASS recall, so the TSFNet-format CSV can carry the
        # recall_0back / recall_2back / recall_3back columns. The existing
        # 'recall' field is the MACRO average and hides which class the model
        # actually fails on -- on the TSFNet side the per-class pattern turned
        # out to differ sharply between subjects (sometimes 2-back collapses
        # into 3-back, sometimes 0-back collapses), and that comparison is
        # impossible without these three numbers.
        _yt = np.asarray(best_metrics.get('labels', []))
        _yp = np.asarray(best_metrics.get('preds', []))
        _per_class_recall = []
        for _c in range(3):
            _m = (_yt == _c)
            _per_class_recall.append(
                float((_yp[_m] == _c).mean() * 100.0) if _m.any() else float('nan'))

        fold_results.append({
            'subject':   subject_id,
            'fold':      fold_idx,
            'test_session': test_session,
            'train_sessions': str([s for s in sessions if s != test_session]),
            'n_train':   len(train_idx),
            'n_test':    len(test_idx),
            'accuracy':  round(best_metrics.get('acc', 0) * 100, 2),
            'kappa':     round(best_metrics.get('kappa', 0), 4),   # FIX 2
            'recall':    round(best_metrics.get('recall', 0) * 100, 2),
            'precision': round(best_metrics.get('precision', 0) * 100, 2),
            'f1':        round(best_metrics.get('f1', 0) * 100, 2),
            'recall_0back': round(_per_class_recall[0], 2),
            'recall_2back': round(_per_class_recall[1], 2),
            'recall_3back': round(_per_class_recall[2], 2),
            'time_min':  round(elapsed, 2),
        })

        log(f"  Fold {fold_idx} done in {elapsed:.1f} min | "
            f"Acc={best_metrics.get('acc',0)*100:.2f}% "
            f"Kappa={best_metrics.get('kappa',0):.4f} "
            f"F1={best_metrics.get('f1',0)*100:.2f}%")

        # ===== diagnostics + XAI, whitelist-gated (same design as TSFNet) =====
        # Runs only for the handful of subjects in DIAGNOSTIC_SUBJECTS, since
        # rendering these for 26 subjects x 3 folds is slow and produces files
        # nobody reads. Both are wrapped so a plotting failure can never kill a
        # multi-hour training run, and ImportError is tolerated so the trainer
        # still works if these optional modules are absent.
        try:
            from mstfdn_model_diagnostics import run_fold_diagnostics, _should_run
            if _should_run(subject_id):
                run_fold_diagnostics(model, train_loader, test_loader, device,
                                     subject_id=subject_id, fold_idx=fold_idx,
                                     out_dir=out_dir)
                try:
                    from mstfdn_xai_extensions import run_mstfdn_xai
                    run_mstfdn_xai(model, test_loader, device,
                                   os.path.join(out_dir, "xai"),
                                   subject_id=subject_id, fold_idx=fold_idx)
                except ImportError:
                    pass
        except ImportError:
            pass
        except Exception as _diag_exc:
            log(f"  [diagnostics] WARNING: skipped for {subject_id} fold {fold_idx}: {_diag_exc}")

        # Free memory between folds
        del model, train_ds, val_ds, test_ds, train_loader, val_loader, test_loader
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    return fold_results


# ===========================================================================
#  MAIN — run all subjects
# ===========================================================================
def main():
    global LOG_FILE

    # ===== FIX 3: reproducible seed control, matching TSFNet's own protocol =====
    import argparse, random
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed. TSFNet's protocol uses 42/999/2024; use the same "
                         "three here so the comparison is seed-matched.")
    ap.add_argument("--val-mode", type=str, default="trial",
                    choices=["trial", "cross_session"],
                    help="How the validation set is carved out of the two TRAINING "
                         "sessions (the test session is never touched either way). "
                         "'trial' (default): hold out ~20%% of TRIALS pooled across both "
                         "training sessions -- identical in kind to TSFNet's own split, so "
                         "a TSFNet-vs-MSTFDN comparison adds no extra differing variable. "
                         "'cross_session': hold out one WHOLE training session -> ~3x more "
                         "validation trials AND cross-session validation matching the test "
                         "regime, at the cost of halving the training data.")
    args = ap.parse_args()

    SEED = args.seed
    VAL_MODE = args.val_mode
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # FIX 5 support: results of different configs must never share a folder.
    out_root = os.path.join(OUTPUT_PATH, f"seed{SEED}_val{VAL_MODE}")
    os.makedirs(out_root, exist_ok=True)
    LOG_FILE = os.path.join(out_root, "training_log.txt")

    log("=" * 55)
    log("MSTFDN — Cross-Session LOSO Training")
    log(f"Device : {DEVICE}")
    if DEVICE == 'cuda':
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
        log(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log(f"Subjects: {len(ALL_SUBJECTS)}")
    log(f"Preprocessed data: {PREPROCESSED_PATH}")
    log(f"Output: {out_root}")
    log(f"[SEED CONTROL] seed={SEED} (python/numpy/torch seeded, cuDNN deterministic)")
    log(f"[CONFIG] val_mode={VAL_MODE!r}  "
        f"(validation carved from TRAINING sessions only -- the test session is never "
        f"used for model selection; this is the fix for the earlier test-as-validation leak)")
    log("=" * 55)

    cfg = get_config()
    all_fold_results = []

    total_t0 = time.time()

    for subject_id in ALL_SUBJECTS:
        sub_dir = os.path.join(PREPROCESSED_PATH, subject_id)
        if not os.path.exists(sub_dir):
            log(f"SKIPPED {subject_id}: data not found")
            continue

        try:
            fold_results = run_loso_subject(subject_id, cfg, out_root, DEVICE,
                                            val_mode=VAL_MODE, seed=SEED)
            all_fold_results.extend(fold_results)

            # per-subject average across 3 folds
            if fold_results:
                avg_acc  = np.mean([r['accuracy']  for r in fold_results])
                avg_rec  = np.mean([r['recall']    for r in fold_results])
                avg_prec = np.mean([r['precision'] for r in fold_results])
                avg_f1   = np.mean([r['f1']        for r in fold_results])
                log(f"\n  {subject_id} LOSO average → "
                    f"Acc={avg_acc:.2f}% Recall={avg_rec:.2f}% "
                    f"Precision={avg_prec:.2f}% F1={avg_f1:.2f}%")

        except Exception as e:
            log(f"ERROR {subject_id}: {e}")
            import traceback; traceback.print_exc()

        finally:
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────────
    if not all_fold_results:
        log("No results to save.")
        return

    df_folds = pd.DataFrame(all_fold_results)
    df_folds.to_csv(os.path.join(out_root, "results_per_subject.csv"), index=False)
    log(f"\nSaved: results_per_subject.csv")

    # Per-subject summary (mean ± std across 3 folds)
    summary_rows = []
    for sub in df_folds['subject'].unique():
        sub_df = df_folds[df_folds['subject'] == sub]
        summary_rows.append({
            'subject':       sub,
            'acc_mean':      round(sub_df['accuracy'].mean(), 2),
            'acc_std':       round(sub_df['accuracy'].std(), 2),
            'recall_mean':   round(sub_df['recall'].mean(), 2),
            'recall_std':    round(sub_df['recall'].std(), 2),
            'precision_mean':round(sub_df['precision'].mean(), 2),
            'precision_std': round(sub_df['precision'].std(), 2),
            'f1_mean':       round(sub_df['f1'].mean(), 2),
            'f1_std':        round(sub_df['f1'].std(), 2),
            'kappa_mean':    round(sub_df['kappa'].mean(), 4),   # FIX 2
            'kappa_std':     round(sub_df['kappa'].std(), 4),
        })

    df_summary = pd.DataFrame(summary_rows)

    # All-subject average (final reported number — paper Table 2)
    overall = {
        'subject':       'ALL_SUBJECTS_AVG',
        'acc_mean':      round(df_summary['acc_mean'].mean(), 2),
        'acc_std':       round(df_summary['acc_mean'].std(), 2),
        'recall_mean':   round(df_summary['recall_mean'].mean(), 2),
        'recall_std':    round(df_summary['recall_mean'].std(), 2),
        'precision_mean':round(df_summary['precision_mean'].mean(), 2),
        'precision_std': round(df_summary['precision_mean'].std(), 2),
        'f1_mean':       round(df_summary['f1_mean'].mean(), 2),
        'f1_std':        round(df_summary['f1_mean'].std(), 2),
        'kappa_mean':    round(df_summary['kappa_mean'].mean(), 4),
        'kappa_std':     round(df_summary['kappa_mean'].std(), 4),
    }
    df_summary = pd.concat([df_summary, pd.DataFrame([overall])], ignore_index=True)
    df_summary.to_csv(os.path.join(out_root, "results_summary.csv"), index=False)
    log(f"Saved: results_summary.csv")

    # ===== FIX 5: emit a CSV in TSFNet's OWN format =====
    # retroactive_significance_check.py / ensemble tooling expect exactly the
    # columns subject_id, mean_acc, mean_kappa, with accuracy as a FRACTION
    # (0-1), not a percentage. Writing this file means MSTFDN results can be
    # fed straight into the same paired Wilcoxon test used for every TSFNet
    # comparison -- no manual reformatting, no risk of a unit mismatch
    # (percent vs fraction) silently corrupting the statistics.
    # FIX 6: emit the FULL 17-column TSFNet contract, not just three columns.
    #
    # Three of those columns cannot be filled for this architecture, and are
    # written EMPTY rather than dropped (project rule 7: a parallel function
    # must keep the consumer's key set, putting None where a value is
    # meaningless, instead of silently removing the key):
    #
    #   eeg_only_acc / fnirs_only_acc / fusion_only_acc
    #       TSFNet has THREE separate classifier heads (y_eeg, y_fnirs,
    #       y_fusion), so each branch has its own accuracy and an oracle
    #       accuracy can be computed over them. MSTFDN's forward returns
    #       (logits, he, hf, hef) where he/hf/hef are hidden FEATURE
    #       representations feeding a SINGLE classifier -- there is no
    #       per-branch prediction to score. Filling these with a proxy
    #       (e.g. a linear probe trained post-hoc on he/hf) would produce a
    #       different quantity from TSFNet's and make the comparison
    #       misleading, so they are deliberately left blank.
    #
    #   w_eeg / w_fusion / w_fnirs
    #       Decision-level fusion weights are specific to TSFNet's Eq.4
    #       combination. MSTFDN fuses at the FEATURE level, so no equivalent
    #       scalar exists.
    #
    # Consequence to state in the thesis: the oracle-accuracy and
    # branch-agreement analyses reported for TSFNet cannot be replicated for
    # MSTFDN. Accuracy, kappa and the per-class recalls ARE fully comparable.
    tsfnet_rows = []
    for sub in df_folds['subject'].unique():
        sub_df = df_folds[df_folds['subject'] == sub]
        tsfnet_rows.append({
            'subject_id':      sub,
            'status':          'ok',
            'mean_acc':        sub_df['accuracy'].mean() / 100.0,   # percent -> fraction
            'std_acc':         sub_df['accuracy'].std() / 100.0,
            'mean_kappa':      sub_df['kappa'].mean(),
            'std_kappa':       sub_df['kappa'].std(),
            'eeg_only_acc':    '',   # architecturally unavailable -- see note above
            'fnirs_only_acc':  '',   # architecturally unavailable
            'fusion_only_acc': '',   # architecturally unavailable
            'w_eeg':           '',   # feature-level fusion: no decision weight
            'w_fusion':        '',
            'w_fnirs':         '',
            'recall_0back':    sub_df['recall_0back'].mean() / 100.0,
            'recall_2back':    sub_df['recall_2back'].mean() / 100.0,
            'recall_3back':    sub_df['recall_3back'].mean() / 100.0,
            'elapsed_min':     sub_df['time_min'].sum(),
            'error':           '',
        })
    compat_path = os.path.join(
        out_root, f"baseline_all_subjects_MSTFDN_seed{SEED}_val{VAL_MODE}.csv")
    pd.DataFrame(tsfnet_rows).to_csv(compat_path, index=False)
    log(f"Saved (TSFNet-compatible, for direct Wilcoxon comparison): "
        f"{os.path.basename(compat_path)}")

    total_elapsed = (time.time() - total_t0) / 60
    log(f"\n{'='*55}")
    log(f"LOSO FINISHED in {total_elapsed:.1f} min")
    log(f"Final — Acc: {overall['acc_mean']} ± {overall['acc_std']}%  "
        f"F1: {overall['f1_mean']} ± {overall['f1_std']}%")
    log(f"{'='*55}")

    print("\n=== FINAL RESULTS ===")
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
