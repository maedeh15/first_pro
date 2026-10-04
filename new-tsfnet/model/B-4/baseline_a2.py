"""
baseline_a2.py -- Priority A-2: classical (band-power + SVM/LDA) and
shallow deep-learning (EEGNet) baselines, evaluated under the SAME
subject-dependent, leave-one-session-out protocol as TSFNet, so their
numbers are directly comparable to the project's main results table.

Two baseline families, matching what A-2 in the roadmap asked for:

1. Classical ML on hand-crafted features (log band-power per channel per
   canonical EEG band, via Welch's method) + SVM(RBF) and LDA -- the
   standard, decades-old BCI pipeline. This is the correct "how much is
   deep learning even buying us" reference point.

2. EEGNet (Lawhern et al., 2018, J. Neural Eng., "EEGNet: A Compact
   Convolutional Neural Network for EEG-based Brain-Computer Interfaces")
   -- a small, EEG-specific deep architecture, distinct from TSFNet's much
   larger multimodal 3D-CNN. This answers a different question than (1):
   not "does deep learning help" but "does TSFNet's added complexity
   (multimodal fusion, CAFE, EFGF, ...) help beyond a minimal EEG-only
   deep baseline".

Both baselines use ONLY the EEG channel-space data (`{subject}_eeg_channel_
windows.npy`, shape (N, 28, 2000) at 200 Hz for a 10 s window) -- not the
fNIRS data and not the 16x16 interpolated grid -- because that is exactly
what a classical BCI pipeline or EEGNet would consume; comparing them
against TSFNet's fused decision isolates the value of the fusion itself,
not an unfair data advantage.

Prerequisite: pip install scikit-learn (already a project dependency, but
confirmed here in case a fresh environment is missing it).

USAGE (per fold, exactly mirroring the project's LOSO loop):

    from baseline_a2 import run_svm_lda_baseline, train_eegnet_one_fold

    results = run_svm_lda_baseline(x_train, y_train, x_test, y_test)
    # results = {"svm_acc":..., "svm_kappa":..., "lda_acc":..., "lda_kappa":...}

    acc, kappa = train_eegnet_one_fold(
        x_train, y_train, x_val, y_val, x_test, y_test,
        n_channels=28, n_samples=2000, device=device, seed=999)
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score

# Canonical EEG bands used by essentially every classical BCI band-power
# pipeline (matches the band definitions used in the TSFNet/MSTFDN papers'
# own preprocessing description).
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
         "beta": (13, 30), "gamma": (30, 45)}


# ============================================================================
# 1. Classical baseline: log band-power features + SVM(RBF) / LDA
# ============================================================================

def extract_band_power_features(x_chan, fs=200):
    """
    x_chan : (N, C, T) array of raw multichannel EEG (float).
    fs     : sampling rate in Hz (200 Hz for this project's 2000-sample,
             10-second channel-space window: 2000/10 = 200).

    Returns (N, C * len(BANDS)) log band-power features -- one value per
    channel per canonical band, via Welch's method. This is the exact
    feature family used by classical BCI pipelines (and by Adaptive
    Fusion's classical baselines).
    """
    x_chan = np.asarray(x_chan, dtype=np.float64)
    N, C, T = x_chan.shape
    n_bands = len(BANDS)
    feats = np.zeros((N, C * n_bands), dtype=np.float64)
    nperseg = min(256, T)
    for i in range(N):
        for c in range(C):
            f, pxx = welch(x_chan[i, c], fs=fs, nperseg=nperseg)
            for b, (lo, hi) in enumerate(BANDS.values()):
                mask = (f >= lo) & (f < hi)
                power = pxx[mask].sum() if mask.any() else 0.0
                feats[i, c * n_bands + b] = np.log(power + 1e-12)
    return feats


def run_svm_lda_baseline(x_train, y_train, x_test, y_test, fs=200, seed=42):
    """
    Trains SVM(RBF) and LDA on log band-power features extracted from raw
    channel-space EEG, and evaluates on the held-out session.

    Note on `seed`: SVC's decision for a convex, deterministic optimization
    problem does not meaningfully depend on random_state (it only affects
    internal tie-breaking in probability estimates, which are not used
    here), and LDA has a closed-form solution with no randomness at all.
    So unlike TSFNet (where 3 independent seeds are required because SGD
    training is stochastic), a single run per fold is sufficient and
    faithful for this baseline -- re-running with different seeds would
    not produce meaningfully different numbers, and reporting several
    identical runs as "3 seeds" would misrepresent the baseline's actual
    (lack of) stochasticity.
    """
    feat_train = extract_band_power_features(x_train, fs=fs)
    feat_test = extract_band_power_features(x_test, fs=fs)

    scaler = StandardScaler().fit(feat_train)
    feat_train_s = scaler.transform(feat_train)
    feat_test_s = scaler.transform(feat_test)

    results = {}

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=seed)
    svm.fit(feat_train_s, y_train)
    pred_svm = svm.predict(feat_test_s)
    results["svm_acc"] = accuracy_score(y_test, pred_svm)
    results["svm_kappa"] = cohen_kappa_score(y_test, pred_svm)

    lda = LinearDiscriminantAnalysis()
    lda.fit(feat_train_s, y_train)
    pred_lda = lda.predict(feat_test_s)
    results["lda_acc"] = accuracy_score(y_test, pred_lda)
    results["lda_kappa"] = cohen_kappa_score(y_test, pred_lda)

    return results


# ============================================================================
# 2. EEGNet (Lawhern et al. 2018) -- compact EEG-specific deep baseline
# ============================================================================

class EEGNet(nn.Module):
    """
    Faithful compact reimplementation of EEGNet's four-block structure:
    temporal conv -> depthwise spatial conv (one filter per channel,
    groups=F1) -> separable conv -> linear classifier. Operates directly
    on raw (channels, time) EEG, exactly the architecture family this
    baseline is meant to represent.
    """

    def __init__(self, n_channels, n_samples, n_classes=3,
                 F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.depthwiseconv = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separableconv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            flat_dim = self._forward_features(dummy).numel()
        self.classify = nn.Linear(flat_dim, n_classes)

    def _forward_features(self, x):
        x = self.firstconv(x)
        x = self.depthwiseconv(x)
        x = self.separableconv(x)
        return x

    def forward(self, x):
        # x: (B, C, T) -> (B, 1, C, T) for Conv2d
        x = x.unsqueeze(1)
        x = self._forward_features(x)
        x = x.flatten(1)
        return self.classify(x)


def train_eegnet_one_fold(x_train, y_train, x_val, y_val, x_test, y_test,
                           n_channels, n_samples, device, seed=42,
                           max_epochs=100, patience=20, lr=1e-3,
                           batch_size=16, verbose=False):
    """
    One LOSO fold: trains EEGNet with early stopping on a held-out
    validation split (same idea as TSFNet's stage1 early stopping), then
    evaluates on the held-out test session. Returns (accuracy, kappa).
    """
    torch.manual_seed(seed)
    model = EEGNet(n_channels, n_samples).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    x_train_t = torch.tensor(np.asarray(x_train), dtype=torch.float32)
    y_train_t = torch.tensor(np.asarray(y_train), dtype=torch.long)
    x_val_t = torch.tensor(np.asarray(x_val), dtype=torch.float32).to(device)
    y_val_t = torch.tensor(np.asarray(y_val), dtype=torch.long).to(device)
    x_test_t = torch.tensor(np.asarray(x_test), dtype=torch.float32).to(device)
    y_test_np = np.asarray(y_test)

    train_ds = torch.utils.data.TensorDataset(x_train_t, y_train_t)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True)

    best_val_acc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(x_val_t).argmax(1) == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  [EEGNet] early stop at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test_t).argmax(1).cpu().numpy()

    acc = accuracy_score(y_test_np, test_pred)
    kappa = cohen_kappa_score(y_test_np, test_pred)
    return acc, kappa


# ============================================================================
# Numeric debug / verification script -- run this file directly
# (`python baseline_a2.py`) before trusting it on real data, per this
# project's standing methodology rule.
# ============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    fs = 200
    T = 2000
    C = 28
    t = np.arange(T) / fs

    # ---- Check 1: band-power extraction correctly localizes a pure sine
    # wave's energy to the right band, for every channel independently. ----
    def make_sine_batch(freq, n=10, noise=0.05):
        x = np.zeros((n, C, T))
        for i in range(n):
            for c in range(C):
                phase = rng.uniform(0, 2 * np.pi)
                x[i, c] = np.sin(2 * np.pi * freq * t + phase) + noise * rng.standard_normal(T)
        return x

    alpha_batch = make_sine_batch(10.0)  # 10 Hz -> alpha band (8-13 Hz)
    feats = extract_band_power_features(alpha_batch, fs=fs)
    n_bands = len(BANDS)
    band_names = list(BANDS.keys())
    alpha_idx = band_names.index("alpha")
    # reshape to (N, C, n_bands) to check per-channel, per-band values
    feats_r = feats.reshape(alpha_batch.shape[0], C, n_bands)
    mean_power_per_band = feats_r.mean(axis=(0, 1))
    assert np.argmax(mean_power_per_band) == alpha_idx, (
        f"Expected peak power in 'alpha' band, got peak in "
        f"'{band_names[np.argmax(mean_power_per_band)]}' -- band extraction is wrong."
    )
    print(f"[PASS] check 1: a pure 10 Hz sine wave correctly shows peak log-power "
          f"in the alpha band (mean log-power per band: "
          f"{dict(zip(band_names, np.round(mean_power_per_band, 2)))}).")

    # ---- Check 2: SVM/LDA baseline runs end-to-end and clears chance on a
    # separable synthetic problem (sanity check the whole pipeline, not
    # just the feature extractor). ----
    def make_labeled_batch(n_per_class=20):
        freqs = {0: 5.0, 1: 10.0, 2: 20.0}  # theta / alpha / beta -> 3 classes
        xs, ys = [], []
        for label, f0 in freqs.items():
            xs.append(make_sine_batch(f0, n=n_per_class, noise=0.3))
            ys.append(np.full(n_per_class, label))
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)

    x_train, y_train = make_labeled_batch(30)
    x_test, y_test = make_labeled_batch(10)
    res = run_svm_lda_baseline(x_train, y_train, x_test, y_test, fs=fs)
    assert res["svm_acc"] > 0.9, f"SVM failed an easily separable synthetic task: {res}"
    assert res["lda_acc"] > 0.9, f"LDA failed an easily separable synthetic task: {res}"
    print(f"[PASS] check 2: SVM/LDA baseline correctly solves an easily-separable "
          f"synthetic 3-class problem (svm_acc={res['svm_acc']:.2f}, "
          f"lda_acc={res['lda_acc']:.2f}).")

    # ---- Check 3: EEGNet forward pass shape + parameter/gradient flow ----
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EEGNet(n_channels=C, n_samples=T, n_classes=3).to(device)
    x_dummy = torch.randn(4, C, T, device=device, requires_grad=True)
    out = model(x_dummy)
    assert out.shape == (4, 3), f"unexpected EEGNet output shape {out.shape}"
    out.sum().backward()
    assert x_dummy.grad is not None and x_dummy.grad.abs().sum() > 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PASS] check 3: EEGNet forward+backward OK, output shape {tuple(out.shape)}, "
          f"{n_params} parameters (compact, as expected for this architecture family).")

    # ---- Check 4: train_eegnet_one_fold runs end-to-end on tiny synthetic
    # data and clears chance (sanity check the whole training loop). ----
    x_tr, y_tr = make_labeled_batch(15)
    x_va, y_va = make_labeled_batch(5)
    x_te, y_te = make_labeled_batch(5)
    acc, kappa = train_eegnet_one_fold(
        x_tr, y_tr, x_va, y_va, x_te, y_te,
        n_channels=C, n_samples=T, device=device, seed=0,
        max_epochs=30, patience=10, batch_size=8)
    assert 0.0 <= acc <= 1.0
    assert acc > 0.5, f"EEGNet training loop failed to clear chance on an easy task (acc={acc:.2f})"
    print(f"[PASS] check 4: train_eegnet_one_fold runs end-to-end and solves an easy "
          f"synthetic task (acc={acc:.2f}, kappa={kappa:.2f}).")

    print("\nALL CHECKS PASSED.")
