"""
mgfnet_dataset.py -- one subject's MGFNet-preprocessed windows
(Preproc_mgfnet_nback.py output) as a PyTorch Dataset, plus the split and
normalisation helpers used by the leave-one-session-out loop.

Leakage rules enforced here:
  * every split is at the TRIAL level, so overlapping sliding windows of one
    trial never end up on both sides of a split;
  * ChannelNormalizer is fitted on the training indices only and then
    applied unchanged to validation / test data.
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class MGFNBackDataset(Dataset):
    def __init__(self, subject_dir, subject_id, indices=None):
        eeg = np.load(os.path.join(subject_dir, f"{subject_id}_mgf_eeg.npy"))
        hbo = np.load(os.path.join(subject_dir, f"{subject_id}_mgf_hbo.npy"))
        hbr = np.load(os.path.join(subject_dir, f"{subject_id}_mgf_hbr.npy"))
        labels = np.load(os.path.join(subject_dir, f"{subject_id}_labels.npy"))
        session_ids = np.load(os.path.join(subject_dir, f"{subject_id}_session_ids.npy"))
        trial_ids = np.load(os.path.join(subject_dir, f"{subject_id}_trial_ids.npy"))

        n = len(labels)
        for name, arr in (("eeg", eeg), ("hbo", hbo), ("hbr", hbr),
                          ("session_ids", session_ids), ("trial_ids", trial_ids)):
            if len(arr) != n:
                raise ValueError(f"{subject_id}: {name} has {len(arr)} rows, labels has {n}")

        if indices is not None:
            eeg, hbo, hbr, labels = eeg[indices], hbo[indices], hbr[indices], labels[indices]
            session_ids, trial_ids = session_ids[indices], trial_ids[indices]

        self.eeg = torch.from_numpy(np.ascontiguousarray(eeg)).float()
        self.hbo = torch.from_numpy(np.ascontiguousarray(hbo)).float()
        self.hbr = torch.from_numpy(np.ascontiguousarray(hbr)).float()
        self.labels = torch.from_numpy(np.ascontiguousarray(labels)).long()
        self.session_ids = np.asarray(session_ids)
        self.trial_ids = np.asarray(trial_ids)

    @property
    def eeg_channels(self):
        return int(self.eeg.shape[1])

    @property
    def nirs_channels(self):
        return int(self.hbo.shape[1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.eeg[idx], self.hbo[idx], self.hbr[idx], self.labels[idx]


def _unpack_mgf_batch(batch):
    """(x_eeg, x_hbo, x_hbr, y)"""
    return batch[0], batch[1], batch[2], batch[3]


class ChannelNormalizer:
    """Per-channel z-score for each modality, fitted on TRAINING windows only.

    Mean/std are taken over all training windows and time points of each
    channel. Applied on-device to every batch (train, val and test) with the
    SAME statistics, so nothing from the held-out session leaks into training.
    """

    def __init__(self, stats):
        self.stats = stats          # {"eeg": (mean, std), "hbo": ..., "hbr": ...}, (C, 1) tensors
        self._dev = {}

    @classmethod
    def fit(cls, ds, indices):
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        stats = {}
        for name in ("eeg", "hbo", "hbr"):
            x = getattr(ds, name)[idx].double()                  # (N, C, T)
            mean = x.mean(dim=(0, 2))
            std = x.std(dim=(0, 2), unbiased=False)
            std = torch.where(std > 0, std, torch.ones_like(std))
            stats[name] = (mean.float().unsqueeze(-1), std.float().unsqueeze(-1))
        return cls(stats)

    def _on(self, device):
        key = str(device)
        if key not in self._dev:
            self._dev[key] = {k: (m.to(device), s.to(device)) for k, (m, s) in self.stats.items()}
        return self._dev[key]

    def __call__(self, x_eeg, x_hbo, x_hbr):
        st = self._on(x_eeg.device)
        return ((x_eeg - st["eeg"][0]) / st["eeg"][1],
                (x_hbo - st["hbo"][0]) / st["hbo"][1],
                (x_hbr - st["hbr"][0]) / st["hbr"][1])

    def state_dict(self):
        return {k: (m.clone(), s.clone()) for k, (m, s) in self.stats.items()}

    @classmethod
    def from_state_dict(cls, sd):
        return cls({k: (m.clone(), s.clone()) for k, (m, s) in sd.items()})


def session_holdout_split(session_ids, test_session):
    """Leave-one-session-out: test = one session, train = the other two."""
    train_idx = np.where(session_ids != test_session)[0]
    test_idx = np.where(session_ids == test_session)[0]
    return train_idx, test_idx


def stratified_trial_kfold(trial_ids, labels, n_splits=5, seed=42):
    """Trial-level stratified K-fold (paper Sec. 3.5: "the fold assignment was
    conducted at the trial/block level rather than at the window level").

    Returns a list of (train_positions, val_positions) into the given arrays.
    Each class's trials are shuffled and dealt round-robin to the folds, so
    every fold sees every class as long as each class has >= n_splits trials.
    """
    trial_ids = np.asarray(trial_ids)
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_ids)
    trial_to_label = {}
    for t in unique_trials:
        lab_t = np.unique(labels[trial_ids == t])
        if len(lab_t) != 1:
            raise ValueError(f"trial {t} carries more than one label: {lab_t}")
        trial_to_label[t] = int(lab_t[0])

    fold_of_trial = {}
    start = 0
    for cls in sorted(set(trial_to_label.values())):
        cls_trials = np.array([t for t in unique_trials if trial_to_label[t] == cls])
        rng.shuffle(cls_trials)
        for j, t in enumerate(cls_trials):
            # rotate the starting fold per class so fold sizes stay balanced
            fold_of_trial[t] = (start + j) % n_splits
        start += len(cls_trials)

    folds = []
    win_fold = np.array([fold_of_trial[t] for t in trial_ids])
    for k in range(n_splits):
        val_pos = np.where(win_fold == k)[0]
        tr_pos = np.where(win_fold != k)[0]
        if len(val_pos) == 0 or len(tr_pos) == 0:
            raise ValueError(f"fold {k} is empty -- too few trials for {n_splits}-fold CV")
        folds.append((tr_pos, val_pos))
    return folds


def trial_level_train_val_split(trial_ids, labels, val_ratio=0.2, seed=42):
    """VERBATIM copy of tsfnet_model_notebook.py's function of the same name
    (TSFNet Sec. 3.6): a single 4:1 TRIAL-level train/val split, stratified by
    class, fixed seed 42. With the same trial_ids/labels it selects exactly
    the same validation trials as the TSFNet runs, so both models are trained
    and validated on identical partitions of every LOSO fold."""
    rng = np.random.default_rng(seed)
    unique_trials = np.unique(trial_ids)
    trial_to_label = {}
    for t in unique_trials:
        trial_to_label[t] = labels[trial_ids == t][0]
    val_trials = []
    for cls in np.unique(labels):
        cls_trials = [t for t in unique_trials if trial_to_label[t] == cls]
        rng.shuffle(cls_trials)
        n_val = max(1, int(round(len(cls_trials) * val_ratio)))
        val_trials.extend(cls_trials[:n_val])
    val_trials = set(val_trials)
    val_mask = np.isin(trial_ids, list(val_trials))
    return np.where(~val_mask)[0], np.where(val_mask)[0]
