# ============================================================================
# Shared helpers for the STA-Net scripts (training, diagnostics, XAI, trial-level eval).
# Import-safe: no argparse, no file or GPU side effects at import time.
# ============================================================================
import os
import numpy as np
import torch

NUM_CLASSES = 3
CLASS_NAMES = ["0-back", "2-back", "3-back"]
COLORS = ["#2a78d6", "#e24b4a", "#1baf7a"]
FNIRS_TRAIN_SCALE = 1e3          # official run_sta_net.py: fnirs *= 1e3
HEADS = ("final", "eeg", "fusion", "fnirs")   # keys p_<head> of the model output dict

# Duration bookkeeping for plot axes (must match Preproc_stanet_nback.py)
WINDOW_SECONDS = 3.0             # one EEG / fNIRS window
LAG_STEP_SECONDS = 1.0           # consecutive fNIRS windows of one sample start 1 s apart


def load_subject(subject_dir, subject_id, scale_fnirs=True):
    """Returns eeg (N,16,16,600) tensor, fnirs (N,11,16,16,30,2) tensor (x1e3, in place),
    labels (N,), session_ids (N,), trial_ids (N,) numpy arrays."""
    f = lambda k: np.load(os.path.join(subject_dir, f"{subject_id}_{k}.npy"))
    eeg = f("eeg_windows").astype(np.float32, copy=False)
    fnirs = f("fnirs_windows").astype(np.float32, copy=False)
    if scale_fnirs:
        fnirs *= FNIRS_TRAIN_SCALE
    return (torch.from_numpy(eeg), torch.from_numpy(fnirs), f("labels").astype(np.int64),
            f("session_ids"), f("trial_ids"))


def session_holdout_split(session_ids, test_session):
    """Leave-one-session-out: (train_val_idx, test_idx)."""
    return np.where(session_ids != test_session)[0], np.where(session_ids == test_session)[0]


class WindowLoader:
    """Minimal re-iterable 'DataLoader' over a subset of one subject's windows, in index order
    (shuffle=False). Yields CPU tensors (x_eeg, x_fnirs, y_long); the caller moves them to a device.
    Indexing the big tensors per batch avoids copying a whole subject subset into RAM."""

    def __init__(self, eeg, fnirs, labels, idx, batch_size=32):
        self.eeg, self.fnirs, self.labels = eeg, fnirs, labels
        self.idx, self.batch_size = np.asarray(idx), batch_size

    def __len__(self):
        return -(-len(self.idx) // self.batch_size)

    def __iter__(self):
        for i in range(0, len(self.idx), self.batch_size):
            ii = self.idx[i:i + self.batch_size]
            b = torch.from_numpy(ii)
            yield self.eeg[b], self.fnirs[b], torch.from_numpy(self.labels[ii])


def cohen_kappa(y_true, y_pred):
    po = (y_true == y_pred).mean()
    pe = sum((y_true == c).mean() * (y_pred == c).mean()
             for c in np.unique(np.concatenate([y_true, y_pred])))
    return (po - pe) / (1 - pe) if 1 - pe > 1e-8 else 0.0


def confusion_matrix_np(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, q in zip(y_true, y_pred):
        cm[int(t), int(q)] += 1
    return cm


def build_model(seg_len=11, num_classes=NUM_CLASSES, device="cpu"):
    from stanet_model import STANet
    return STANet(num_classes, seg_len=seg_len).to(device)


def adapt_batchnorm_to_target(model, loader, device, log_fn=print):
    """AdaBN (Li et al. 2017), same implementation as your TSFNet adapt_batchnorm_to_target:
    replace BN running stats by the exact cumulative average over the held-out session's UNLABELED
    windows; weights untouched, dropout stays off. TRANSDUCTIVE -- STA-Net itself does NOT do this;
    only use it to match a TSFNet run that used --adabn."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    moms = [m.momentum for m in bns]
    model.eval()
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
        m.train()
    n = 0
    with torch.no_grad():
        for xe, xf, _ in loader:
            model(xe.to(device), xf.to(device))
            n += 1
    for m, mom in zip(bns, moms):
        m.momentum = mom
    model.eval()
    log_fn(f"  [adabn] adapted {len(bns)} BatchNorm layer(s) on {n} unlabeled batch(es)")
    return model


@torch.no_grad()
def collect_outputs(model, loader, device, keys=("p_final", "p_eeg", "p_fusion", "p_fnirs", "w", "att",
                                                 "fusion_emb", "fnirs_emb", "eeg_emb"),
                    max_batches=None):
    """One eval-mode pass. Returns dict key -> concatenated numpy array, plus 'y_true'."""
    was_training = model.training
    model.eval()
    store, ys = {k: [] for k in keys}, []
    for i, (xe, xf, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        out = model(xe.to(device), xf.to(device))
        for k in keys:
            store[k].append(out[k].detach().cpu().numpy())
        ys.append(y.numpy())
    if was_training:
        model.train()
    res = {k: np.concatenate(v) for k, v in store.items()}
    res["y_true"] = np.concatenate(ys)
    return res


# Official 16x16 pixel positions ([row, col]) of the 28 EEG electrodes (same table as Preproc_stanet_nback.py).
# At these pixels the grid holds the MEASURED value; all other pixels are interpolated.
EEG_CHN_NAMES = ['Fp1', 'AFF5h', 'AFz', 'F1', 'FC5', 'FC1', 'T7', 'C3', 'Cz', 'CP5', 'CP1', 'P7', 'P3', 'Pz',
                 'POz', 'O1', 'Fp2', 'AFF6h', 'F2', 'FC2', 'FC6', 'C4', 'T8', 'CP2', 'CP6', 'P4', 'P8', 'O2']
EEG_PIXELS = np.array([
    [0, 6], [2, 5], [2, 8], [3, 7], [5, 2], [5, 6], [7, 1], [7, 4], [7, 8], [9, 2], [9, 6], [11, 2], [11, 5],
    [11, 8], [13, 8], [14, 6], [0, 10], [2, 11], [3, 9], [5, 10], [5, 14], [7, 12], [7, 15], [9, 10], [9, 14],
    [11, 11], [11, 14], [14, 10]])
