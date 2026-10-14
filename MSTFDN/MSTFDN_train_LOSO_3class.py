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


CLASS_NAMES = ["0-back", "2-back", "3-back"]


def train_fold(model, cfg, train_loader, val_loader,
               checkpoint_path: str, device: str) -> dict:
    """
    Train one LOSO fold. Logs every epoch in the same layout as the TSFNet
    log. Returns {'best_epoch', 'stopped_epoch', 'early_stopped', 'best_val'}.
    """
    loss_fn   = MSTFDNLoss(cfg, model)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs - cfg.warmup_epochs,
        eta_min=cfg.lr * cfg.eta_min_ratio)

    best_val_loss = float('inf')
    best_epoch    = 0
    best_metrics  = {}
    patience_cnt  = 0
    early_stopped = False
    epoch_no      = 0

    for epoch in range(cfg.max_epochs):
        epoch_no = epoch + 1
        warmup_lr(optimizer, epoch, cfg)
        tr = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = evaluate(model, val_loader, loss_fn, device)

        if epoch >= cfg.warmup_epochs:
            scheduler.step()

        log(f"[train] epoch {epoch_no} | train_loss={tr['loss']:.4f} "
            f"train_acc={tr['acc']:.4f} | val_loss={va['loss']:.4f} "
            f"val_acc={va['acc']:.4f} val_kappa={va['kappa']:.4f}")

        # early stopping on validation loss
        if va['loss'] < best_val_loss:
            best_val_loss = va['loss']
            best_metrics  = va.copy()
            best_epoch    = epoch_no
            patience_cnt  = 0
            torch.save({
                'epoch':       epoch_no,
                'model_state': model.state_dict(),
                'val_metrics': va,
            }, checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.patience:
                early_stopped = True
                log(f"[train] early stop at epoch {epoch_no} "
                    f"(no improvement for {cfg.patience} epochs)")
                break

    if not early_stopped:
        log(f"[train] reached max_epochs={cfg.max_epochs} without early stop")
    return {'best_epoch': best_epoch, 'stopped_epoch': epoch_no,
            'early_stopped': early_stopped, 'best_val': best_metrics}


@torch.no_grad()
def modality_ablation(model, loader, device) -> dict:
    """Test accuracy with ONE modality's input set to zero (inputs are
    z-scored, so zero = the per-segment mean).

    MSTFDN has a single classifier head on the fused features, so TSFNet's
    per-branch head accuracies do not exist here. This zero-input ablation
    is the closest MSTFDN-native measurement of how much each modality
    contributes. It is a DIFFERENT quantity from TSFNet's branch accuracy
    and is labelled as such in the log, CSV and JSON.
      eeg   = EEG kept, fNIRS zeroed
      fnirs = fNIRS kept, EEG zeroed
      fusion = both inputs (= the normal test accuracy)
    """
    model.eval()
    hits = {'eeg': 0, 'fnirs': 0, 'fusion': 0}
    n = 0
    for eeg, fnirs, labels in loader:
        eeg, fnirs, labels = eeg.to(device), fnirs.to(device), labels.to(device)
        hits['eeg']    += (model(eeg, torch.zeros_like(fnirs))[0].argmax(1) == labels).sum().item()
        hits['fnirs']  += (model(torch.zeros_like(eeg), fnirs)[0].argmax(1) == labels).sum().item()
        hits['fusion'] += (model(eeg, fnirs)[0].argmax(1) == labels).sum().item()
        n += labels.size(0)
    return {k: v / max(n, 1) for k, v in hits.items()}


def confusion(y_true, y_pred, k=3):
    cm = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def format_confusion(cm) -> str:
    lines = ["            " + " ".join(f"pred {c:>7}" for c in CLASS_NAMES)]
    for i, c in enumerate(CLASS_NAMES):
        row_n = cm[i].sum()
        rec = 100.0 * cm[i, i] / row_n if row_n else float('nan')
        lines.append(f"true {c:>7} " + " ".join(f"{v:>12d}" for v in cm[i])
                     + f"   | recall {rec:.1f}%")
    return "\n".join(lines)


def _pm(values):
    v = np.asarray(values, dtype=float)
    return float(np.mean(v)), float(np.std(v))


# ===========================================================================
#  LOSO TRAINING FOR ONE SUBJECT
# ===========================================================================
def run_loso_subject(subject_id: str, cfg: MSTFDNConfig,
                     out_dir: str, device: str,
                     val_mode: str = "trial", seed: int = 42,
                     print_config: bool = False) -> dict:
    """
    Run 3-fold LOSO for one subject.
    Returns a subject summary dict (with a 'folds' list), or None if the
    subject produced no folds.
    """
    dataset = MSTFDNSubjectDataset(data_dir=PREPROCESSED_PATH, subject_id=subject_id)

    session_ids  = dataset.session_ids   # numpy (N,) values 1/2/3
    sessions     = [1, 2, 3]
    fold_results = []
    subj_cm      = np.zeros((3, 3), dtype=np.int64)
    t_subj       = time.time()

    for fold_idx, test_session in enumerate(sessions, 1):
        log(f"=== {subject_id} | test session {test_session} ===")
        if print_config and fold_idx == 1:
            log(f"[config] val_mode={val_mode!r} -- validation is carved from the two "
                f"TRAINING sessions only; the test session is never used for model "
                f"selection")
            log(f"[config] early stopping on val_loss, patience={cfg.patience}, "
                f"max_epochs={cfg.max_epochs}, warmup={cfg.warmup_epochs}, "
                f"lr={cfg.lr}, weight_decay={cfg.weight_decay}, batch={cfg.batch_size}")
            log(f"[config] no AdaBN / no test-time adaptation -- BatchNorm uses "
                f"training-set running statistics at test time (paper protocol)")
            log(f"[config] ablation = zero-input modality ablation (MSTFDN has one "
                f"fused classifier, no per-branch heads); fusion weights = n/a "
                f"(feature-level fusion, no Eq.4 gates)")

        test_idx  = np.where(session_ids == test_session)[0]
        train_idx = np.where(session_ids != test_session)[0]
        if len(train_idx) == 0 or len(test_idx) == 0:
            log(f"WARNING: empty split for fold {fold_idx}, skipping")
            continue

        # Validation is carved out of the TRAINING sessions only.
        if val_mode == "trial":
            if dataset.trial_ids is None:
                raise ValueError(
                    f"val_mode='trial' needs {subject_id}_trial_ids.npy, which was not "
                    f"found. Either re-run preprocessing (it does write this file) or "
                    f"use --val-mode cross_session.")
            tv_trials = dataset.trial_ids[train_idx]
            tv_labels = dataset.labels.numpy()[train_idx]
            sub_tr, sub_va = trial_level_train_val_split(tv_trials, tv_labels, seed=seed)
            split_msg = (f"{len(np.unique(tv_trials[sub_va]))} val trial(s) held out "
                         f"from the pooled training sessions")
        else:  # 'cross_session'
            tv_sessions = np.unique(session_ids[train_idx])
            val_session = int(tv_sessions.min())
            sub_tr, sub_va = cross_session_train_val_split(session_ids[train_idx], val_session)
            split_msg = f"validating on whole session {val_session}"

        fit_idx = train_idx[sub_tr]
        val_idx = train_idx[sub_va]
        log(f"[split] train={len(fit_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"windows ({split_msg})")

        train_loader = DataLoader(Subset(dataset, fit_idx), batch_size=cfg.batch_size,
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader   = DataLoader(Subset(dataset, val_idx), batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)
        test_loader  = DataLoader(Subset(dataset, test_idx), batch_size=cfg.batch_size,
                                  shuffle=False, num_workers=0, pin_memory=True)

        # Fresh model for each fold (paper: "trained from scratch")
        model     = MSTFDN(cfg).to(device)
        ckpt_path = os.path.join(out_dir, f"{subject_id}_session{test_session}_best.pt")

        t0 = time.time()
        # Model selection uses VALIDATION only...
        tr_info = train_fold(model, cfg, train_loader, val_loader, ckpt_path, device)
        # ...then the best checkpoint is scored ONCE on the untouched test set.
        # The checkpoint is a dict {'epoch','model_state','val_metrics'};
        # weights_only=False because val_metrics holds numpy scalars
        # (PyTorch >= 2.6 defaults to weights_only=True).
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        log(f"[select] using best checkpoint (epoch {ckpt['epoch']}, lowest val_loss) "
            f"as the final model")
        test_m  = evaluate(model, test_loader, MSTFDNLoss(cfg, model), device)
        abl     = modality_ablation(model, test_loader, device)
        elapsed = (time.time() - t0) / 60

        cm = confusion(test_m['labels'], test_m['preds'])
        subj_cm += cm
        per_class_recall = [float(cm[c, c] / cm[c].sum()) if cm[c].sum() else float('nan')
                            for c in range(3)]

        log(f"-> test_acc={test_m['acc']:.4f} test_kappa={test_m['kappa']:.4f} "
            f"| ablation (zero-input): eeg={abl['eeg']:.3f} fnirs={abl['fnirs']:.3f} "
            f"fusion={abl['fusion']:.3f} | weights: n/a (feature-level fusion)")

        fold_rec = {
            'subject':        subject_id,
            'fold':           fold_idx,
            'test_session':   test_session,
            'train_sessions': str([s for s in sessions if s != test_session]),
            'n_train':        len(fit_idx),
            'n_val':          len(val_idx),
            'n_test':         len(test_idx),
            'best_epoch':     int(ckpt['epoch']),
            'stopped_epoch':  tr_info['stopped_epoch'],
            'early_stopped':  tr_info['early_stopped'],
            'best_val_loss':  float(tr_info['best_val'].get('loss', float('nan'))),
            'best_val_acc':   float(tr_info['best_val'].get('acc', float('nan'))),
            'best_val_kappa': float(tr_info['best_val'].get('kappa', float('nan'))),
            'accuracy':       round(test_m['acc'] * 100, 2),
            'kappa':          round(test_m['kappa'], 4),
            'recall':         round(test_m['recall'] * 100, 2),
            'precision':      round(test_m['precision'] * 100, 2),
            'f1':             round(test_m['f1'] * 100, 2),
            'recall_0back':   round(per_class_recall[0] * 100, 2),
            'recall_2back':   round(per_class_recall[1] * 100, 2),
            'recall_3back':   round(per_class_recall[2] * 100, 2),
            'ablation_eeg_acc':    round(abl['eeg'], 4),
            'ablation_fnirs_acc':  round(abl['fnirs'], 4),
            'ablation_fusion_acc': round(abl['fusion'], 4),
            'time_min':       round(elapsed, 2),
        }
        fold_results.append(fold_rec)

        res_path = os.path.join(out_dir, f"results_{subject_id}_session{test_session}.json")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump({**fold_rec, 'confusion_matrix': cm.tolist(),
                       'class_names': CLASS_NAMES}, f, indent=2)
        log(f"results saved to: {res_path}")
        log(f"checkpoint saved to: {ckpt_path}")

        # ===== diagnostics + XAI, whitelist-gated (same design as TSFNet) =====
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
            log(f"[diagnostics] WARNING: skipped for {subject_id} fold {fold_idx}: {_diag_exc}")

        del model, train_loader, val_loader, test_loader
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

    if not fold_results:
        return None

    acc_m, acc_s = _pm([r['accuracy'] / 100 for r in fold_results])
    kap_m, kap_s = _pm([r['kappa'] for r in fold_results])
    abl_stats = {k: _pm([r[f'ablation_{k}_acc'] for r in fold_results])
                 for k in ('eeg', 'fnirs', 'fusion')}

    log(f"{subject_id} FINAL: acc={acc_m:.4f}+-{acc_s:.4f} kappa={kap_m:.4f}+-{kap_s:.4f}")
    log(f"{subject_id} ABLATION ACCURACY (zero-input, mean over {len(fold_results)} folds): "
        + " ".join(f"{k}={m:.4f}+-{s:.4f}" for k, (m, s) in abl_stats.items()))
    log(f"{subject_id} LEARNED FUSION WEIGHTS: n/a (MSTFDN fuses at feature level)")

    summary = {
        'subject_id': subject_id,
        'mean_acc': acc_m, 'std_acc': acc_s,
        'mean_kappa': kap_m, 'std_kappa': kap_s,
        'ablation': {k: {'mean': m, 'std': s} for k, (m, s) in abl_stats.items()},
        'confusion_matrix': subj_cm.tolist(),
        'class_names': CLASS_NAMES,
        'elapsed_min': (time.time() - t_subj) / 60,
        'folds': fold_results,
    }
    sum_path = os.path.join(out_dir, f"cross_session_{subject_id}_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Full cross-session summary saved to: {sum_path}")

    log(f"[{subject_id}] DONE in {summary['elapsed_min']:.1f} min | "
        f"acc={acc_m:.4f}+-{acc_s:.4f} kappa={kap_m:.4f}+-{kap_s:.4f} | "
        f"ablation acc: eeg={abl_stats['eeg'][0]:.4f} fnirs={abl_stats['fnirs'][0]:.4f} "
        f"fusion={abl_stats['fusion'][0]:.4f} | fusion weights: n/a")
    log(f"confusion matrix (summed over {len(fold_results)} session-folds):\n"
        + format_confusion(subj_cm))
    return summary


# ===========================================================================
#  MAIN — run all subjects
# ===========================================================================
def main():
    global LOG_FILE

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
                         "training sessions -- identical in kind to TSFNet's own split. "
                         "'cross_session': hold out one WHOLE training session.")
    args = ap.parse_args()

    SEED = args.seed
    VAL_MODE = args.val_mode
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Results of different configs must never share a folder.
    out_root = os.path.join(OUTPUT_PATH, f"seed{SEED}_val{VAL_MODE}")
    os.makedirs(out_root, exist_ok=True)
    LOG_FILE = os.path.join(out_root, "training_log.txt")
    run_tag = f"MSTFDN_LOSO_3CLASS_PAPER_EXACT_VAL{VAL_MODE.upper()}_SEED{SEED}"
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log(f"===== CrashSafeLogger started [{run_tag}] "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log("MSTFDN -- Cross-Session LOSO Training (3-class N-back)")
    log(f"Device : {DEVICE}")
    if DEVICE == 'cuda':
        log(f"GPU    : {torch.cuda.get_device_name(0)}")
        log(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    log(f"Subjects: {len(ALL_SUBJECTS)}")
    log(f"Preprocessed data: {PREPROCESSED_PATH}")
    log(f"Output: {out_root}")
    log(f"[SEED CONTROL] seed={SEED} (python/numpy/torch seeded, cuDNN deterministic)")

    cfg = get_config()
    summaries, failed = [], []
    total_t0 = time.time()
    n_sub = len(ALL_SUBJECTS)

    for i, subject_id in enumerate(ALL_SUBJECTS, 1):
        log("=" * 70)
        log(f"[{i}/{n_sub}] {subject_id} ({time.time() - total_t0:.1f}s elapsed so far)")
        sub_dir = os.path.join(PREPROCESSED_PATH, subject_id)
        if not os.path.exists(sub_dir):
            log(f"SKIPPED {subject_id}: data not found")
            failed.append((subject_id, "data not found"))
            continue
        try:
            s = run_loso_subject(subject_id, cfg, out_root, DEVICE,
                                 val_mode=VAL_MODE, seed=SEED, print_config=(i == 1))
            if s is None:
                failed.append((subject_id, "no folds"))
            else:
                summaries.append(s)
        except Exception as e:
            log(f"ERROR {subject_id}: {e}")
            import traceback; traceback.print_exc()
            failed.append((subject_id, str(e)))
        finally:
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

    log("=" * 70)
    if not summaries:
        log("No results to save.")
        log(f"===== CrashSafeLogger stopped [{run_tag}] "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
        return

    log(f"BASELINE RUN COMPLETE: {len(summaries)}/{n_sub} subjects finished successfully")
    for sid, why in failed:
        log(f"  FAILED {sid}: {why}")

    accs   = [s['mean_acc'] for s in summaries]
    kappas = [s['mean_kappa'] for s in summaries]
    log(f"Grand mean accuracy across subjects: {np.mean(accs):.4f} (+-{np.std(accs):.4f})")
    log(f"Grand mean kappa across subjects: {np.mean(kappas):.4f} (+-{np.std(kappas):.4f})")
    log("Grand mean ablation accuracy (zero-input): " + " ".join(
        f"{k}={np.mean([s['ablation'][k]['mean'] for s in summaries]):.4f}"
        for k in ('eeg', 'fnirs', 'fusion')))
    log("Grand mean learned fusion weights: n/a (feature-level fusion)")

    grand_cm = sum(np.asarray(s['confusion_matrix']) for s in summaries)
    n_win = int(grand_cm.sum())
    n_err = n_win - int(np.trace(grand_cm))
    c23 = int(grand_cm[1, 2] + grand_cm[2, 1])
    log("=" * 70)
    log(f"GRAND CONFUSION MATRIX (summed over {len(summaries)} subjects x 3 folds)\n"
        + format_confusion(grand_cm))
    log(f"overall accuracy from matrix: {100.0 * np.trace(grand_cm) / max(n_win, 1):.2f}% "
        f"(n={n_win} windows)")
    log(f"2-back <-> 3-back confusions: {c23} of {n_err} total errors "
        f"({100.0 * c23 / max(n_err, 1):.1f}% of all errors)")

    log("=" * 70)
    log(f"PER-SUBJECT RESULTS ({len(summaries)} completed / {n_sub} total)")
    for s in summaries:
        a = s['ablation']
        log(f"{s['subject_id']}: acc={s['mean_acc']:.4f} +- {s['std_acc']:.4f} "
            f"kappa={s['mean_kappa']:.4f} +- {s['std_kappa']:.4f} | ablation acc: "
            f"eeg={a['eeg']['mean']:.4f} fnirs={a['fnirs']['mean']:.4f} "
            f"fusion={a['fusion']['mean']:.4f} | fusion weights: n/a")

    # ── Save results ──────────────────────────────────────────────────────
    all_folds = [r for s in summaries for r in s['folds']]
    df_folds = pd.DataFrame(all_folds)
    p_folds = os.path.join(out_root, "results_per_subject.csv")
    df_folds.to_csv(p_folds, index=False)

    summary_rows = []
    for sub in df_folds['subject'].unique():
        sub_df = df_folds[df_folds['subject'] == sub]
        summary_rows.append({
            'subject':        sub,
            'acc_mean':       round(sub_df['accuracy'].mean(), 2),
            'acc_std':        round(sub_df['accuracy'].std(), 2),
            'recall_mean':    round(sub_df['recall'].mean(), 2),
            'recall_std':     round(sub_df['recall'].std(), 2),
            'precision_mean': round(sub_df['precision'].mean(), 2),
            'precision_std':  round(sub_df['precision'].std(), 2),
            'f1_mean':        round(sub_df['f1'].mean(), 2),
            'f1_std':         round(sub_df['f1'].std(), 2),
            'kappa_mean':     round(sub_df['kappa'].mean(), 4),
            'kappa_std':      round(sub_df['kappa'].std(), 4),
        })
    df_summary = pd.DataFrame(summary_rows)
    overall = {'subject': 'ALL_SUBJECTS_AVG'}
    for col in ('acc', 'recall', 'precision', 'f1', 'kappa'):
        nd = 4 if col == 'kappa' else 2
        overall[f'{col}_mean'] = round(df_summary[f'{col}_mean'].mean(), nd)
        overall[f'{col}_std']  = round(df_summary[f'{col}_mean'].std(), nd)
    df_summary = pd.concat([df_summary, pd.DataFrame([overall])], ignore_index=True)
    p_summary = os.path.join(out_root, "results_summary.csv")
    df_summary.to_csv(p_summary, index=False)

    # TSFNet-format CSV (same 17 columns, accuracy as a FRACTION) so MSTFDN
    # results feed straight into the same paired Wilcoxon test.
    # eeg/fnirs/fusion_only_acc and w_* are left EMPTY on purpose: MSTFDN has
    # no per-branch heads and no Eq.4 gates. The zero-input ablation is a
    # different quantity and lives in its own columns in
    # results_per_subject.csv and the JSON, never in TSFNet's columns.
    tsfnet_rows = []
    for s in summaries:
        f = pd.DataFrame(s['folds'])
        tsfnet_rows.append({
            'subject_id':      s['subject_id'],
            'status':          'ok',
            'mean_acc':        s['mean_acc'],
            'std_acc':         s['std_acc'],
            'mean_kappa':      s['mean_kappa'],
            'std_kappa':       s['std_kappa'],
            'eeg_only_acc':    '',
            'fnirs_only_acc':  '',
            'fusion_only_acc': '',
            'w_eeg':           '',
            'w_fusion':        '',
            'w_fnirs':         '',
            'recall_0back':    f['recall_0back'].mean() / 100.0,
            'recall_2back':    f['recall_2back'].mean() / 100.0,
            'recall_3back':    f['recall_3back'].mean() / 100.0,
            'elapsed_min':     s['elapsed_min'],
            'error':           '',
        })
    for sid, why in failed:
        tsfnet_rows.append({'subject_id': sid, 'status': 'failed', 'error': why})
    base = f"baseline_all_subjects_{stamp}_MSTFDN_seed{SEED}_val{VAL_MODE}"
    p_csv  = os.path.join(out_root, base + ".csv")
    p_json = os.path.join(out_root, base + ".json")
    pd.DataFrame(tsfnet_rows).to_csv(p_csv, index=False)
    with open(p_json, "w", encoding="utf-8") as fh:
        json.dump({
            'run_tag': run_tag, 'seed': SEED, 'val_mode': VAL_MODE,
            'n_completed': len(summaries), 'n_total': n_sub,
            'failed': [{'subject_id': a, 'error': b} for a, b in failed],
            'grand_mean_acc': float(np.mean(accs)), 'grand_std_acc': float(np.std(accs)),
            'grand_mean_kappa': float(np.mean(kappas)), 'grand_std_kappa': float(np.std(kappas)),
            'grand_confusion_matrix': grand_cm.tolist(), 'class_names': CLASS_NAMES,
            'subjects': summaries,
        }, fh, indent=2)

    log(f"Full results: {p_csv}")
    log(f"              {p_json}")
    log(f"              {p_folds}")
    log(f"              {p_summary}")
    log(f"Total time: {(time.time() - total_t0) / 60:.1f} min")
    log(f"===== CrashSafeLogger stopped [{run_tag}] "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")


if __name__ == "__main__":
    main()
