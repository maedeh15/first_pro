"""
stanet_xai.py -- Explainability for STA-Net (rewrite of your TSFNet xai_extensions.py).

TSFNet component  ->  STA-Net counterpart used here
--------------------------------------------------------------------------------------------
SHAP GradientExplainer on raw EEG / fNIRS        ->  same (run_shap_analysis); EEG window = 3 s
                                                     (16x16x600), fNIRS = 11 lagged windows.
fixed fusion weights w_eeg/w_fusion/w_fnirs      ->  STA-Net's kappa_fusion/kappa_fnirs are computed PER
                                                     SAMPLE (softmax of an FC layer), so they are
                                                     plotted as DISTRIBUTIONS (plot_fusion_gating_weights).
EFGF gamma/alpha scalars                         ->  FGSA gamma = sigmoid(residual_para) per FGSA layer
                                                     (plot_fgsa_gating_scalars). STA-Net has no alpha.
EFGF attention map Phi                           ->  FGSA spatial attention Phi (plot_fgsa_attention_map).
                                                     Phi is derived from fNIRS ONLY (Eq. 2), at 8x8 / 4x4.
CAFE cross-attention hooks (degenerate)          ->  NOT needed: STA-Net's EGTA attention is real
                                                     (query = fusion token, keys = 11 lagged fNIRS
                                                     windows) and is returned by the model as out['att'].
run_egta_attention_analysis                      ->  same name; x-axis = lag of the fNIRS window behind
                                                     the EEG window start (not time since trial onset).
evaluate_segment_pruning                         ->  prunes the l=11 lagged windows (not 21 segments).
run_single_subject_tsne / run_group_level_tsne   ->  same, on the 256-d EEG / fNIRS / fusion features.
GradCAM3D                                        ->  same formula; targets = last conv block of each branch
                                                     (BN output before ELU), fusion target = FGSA-2 output.

Differences worth knowing
  * STA-Net has NO AdaBN / test-time adaptation. Checkpoints come from run_stanet_loso.py
    (<results>/VPxxx_session{k}_best.pt).
  * Grad-CAM FIX vs the original: retained activation gradients accumulate across successive
    .backward() calls, so the class loop in the original leaked class 0's gradient into class 1/2.
    GradCAM3D.compute() here clears act.grad after use.
  * 16x16 grid: unlike TSFNet's projection grid, STA-Net's EEG grid has the 28 electrodes at KNOWN pixels
    (stanet_common.EEG_PIXELS) -- these are marked on the EEG maps; all other cells are interpolated.

Prerequisite: pip install shap
"""
import os
import gc
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import StandardScaler

from stanet_common import (CLASS_NAMES, COLORS, WINDOW_SECONDS, LAG_STEP_SECONDS, EEG_PIXELS, EEG_CHN_NAMES,
                           WindowLoader, collect_outputs)

warnings.filterwarnings("ignore")


def _mark_electrodes(ax, names=False):
    ax.scatter(EEG_PIXELS[:, 1], EEG_PIXELS[:, 0], s=14, facecolors="none", edgecolors="white", lw=0.8)
    if names:
        for (r, c), n in zip(EEG_PIXELS, EEG_CHN_NAMES):
            ax.text(c, r, n, color="white", fontsize=5, ha="center", va="bottom")


# ============================================================================
# Feature extraction (replaces the hook-based _extract_features of TSFNet)
# ============================================================================
def extract_features(model, loader, device):
    d = collect_outputs(model, loader, device,
                        keys=("p_final", "eeg_emb", "fnirs_emb", "fusion_emb"))
    return {"feat_eeg": d["eeg_emb"], "feat_fnirs": d["fnirs_emb"], "feat_fusion": d["fusion_emb"],
            "feat_combined": np.concatenate([d["eeg_emb"], d["fnirs_emb"], d["fusion_emb"]], axis=1),
            "probs": d["p_final"], "labels": d["y_true"]}


# ============================================================================
# 1. SHAP -- GradientExplainer on raw EEG and fNIRS inputs
# ============================================================================
class _WrappedBranchModel(torch.nn.Module):
    """shap needs single-input -> single-output. Freezes one modality at a (background-mean) tensor and
    explains the other; output = p_final (B,3)."""

    def __init__(self, model, fixed_fnirs=None, fixed_eeg=None):
        super().__init__()
        self.model, self.fixed_fnirs, self.fixed_eeg = model, fixed_fnirs, fixed_eeg

    def forward(self, x):
        if self.fixed_fnirs is not None:
            fn = self.fixed_fnirs.expand(x.shape[0], *self.fixed_fnirs.shape[1:])
            return self.model(x, fn)["p_final"]
        eg = self.fixed_eeg.expand(x.shape[0], *self.fixed_eeg.shape[1:])
        return self.model(eg, x)["p_final"]


def _gather(loader, n):
    eeg, fn, lab = [], [], []
    for xe, xf, y in loader:
        eeg.append(xe); fn.append(xf); lab.append(y)
        if sum(x.shape[0] for x in eeg) >= n:
            break
    return torch.cat(eeg).float(), torch.cat(fn).float(), torch.cat(lab)


def run_shap_analysis(model, test_loader, device, session_id, output_dir, n_background=32, n_explain=64):
    """SHAP (GradientExplainer) for the EEG and fNIRS inputs, per class. Cost: heavy for the 3-D tensors;
    reduce n_explain if needed."""
    try:
        import shap
    except ImportError:
        raise ImportError("pip install shap")
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    eeg_all, fnirs_all, lbl_all = _gather(test_loader, n_background + n_explain)
    n_avail = eeg_all.shape[0]
    n_bg = min(n_background, n_avail // 2)
    n_ex = min(n_explain, n_avail - n_bg)
    idx_bg = np.random.choice(n_avail, n_bg, replace=False)
    idx_ex = np.random.choice(np.setdiff1d(np.arange(n_avail), idx_bg), n_ex, replace=False)
    eeg_bg, fnirs_bg = eeg_all[idx_bg].to(device), fnirs_all[idx_bg].to(device)
    eeg_ex, fnirs_ex = eeg_all[idx_ex].to(device), fnirs_all[idx_ex].to(device)
    lbl_ex = lbl_all[idx_ex].numpy()
    print(f"  [SHAP] background={n_bg}, explain={n_ex} ...")

    w_eeg = _WrappedBranchModel(model, fixed_fnirs=fnirs_bg.mean(0, keepdim=True)).to(device)
    sv_eeg = _normalize_shap_output(shap.GradientExplainer(w_eeg, eeg_bg).shap_values(eeg_ex), 3)
    w_fn = _WrappedBranchModel(model, fixed_eeg=eeg_bg.mean(0, keepdim=True)).to(device)
    sv_fn = _normalize_shap_output(shap.GradientExplainer(w_fn, fnirs_bg).shap_values(fnirs_ex), 3)

    _plot_shap_eeg_spatial(sv_eeg, lbl_ex, session_id, output_dir)
    _plot_shap_eeg_temporal(sv_eeg, lbl_ex, session_id, output_dir)
    _plot_shap_fnirs_segment_importance(sv_fn, lbl_ex, session_id, output_dir)
    np.savez(os.path.join(output_dir, f"S{session_id}_shap_raw.npz"),
             shap_eeg=sv_eeg, shap_fnirs=sv_fn, labels=lbl_ex)
    print(f"  [SHAP] saved raw values + plots -> {output_dir}")
    return {"eeg": sv_eeg, "fnirs": sv_fn}


def _normalize_shap_output(shap_values, n_classes):
    """-> (n_classes, N, *input_dims) regardless of shap version."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)
    arr = np.asarray(shap_values)
    return np.moveaxis(arr, -1, 0) if arr.shape[-1] == n_classes else arr


def _plot_shap_eeg_spatial(sv, labels, session_id, output_dir):
    """mean |SHAP| over time -> 16x16 map per class; white circles = the 28 real electrode pixels."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, ax in enumerate(axes):
        m = labels == c
        if m.sum() == 0:
            continue
        v = np.abs(sv[c][m]).mean(axis=(0, -1))
        im = ax.imshow(v, cmap="inferno")
        _mark_electrodes(ax)
        ax.set_title(f"{CLASS_NAMES[c]} (n={m.sum()})"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"SHAP |importance| -- EEG spatial map -- Session {session_id}\n"
                 "(white circles = real electrode pixels; other cells are interpolated)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_eeg_spatial.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    [saved] {path}")


def _plot_shap_eeg_temporal(sv, labels, session_id, output_dir, window_seconds=WINDOW_SECONDS):
    fig, ax = plt.subplots(figsize=(12, 5))
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        m = labels == c
        if m.sum() == 0:
            continue
        v = np.abs(sv[c][m]).mean(axis=(0, 1, 2))
        ax.plot(np.linspace(0, window_seconds, len(v)), v, color=col, label=cname, lw=1.3)
    ax.set_xlabel("Time within the 3 s EEG window (s)"); ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"SHAP -- EEG temporal profile -- Session {session_id}"); ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_eeg_temporal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    [saved] {path}")


def _plot_shap_fnirs_segment_importance(sv, labels, session_id, output_dir):
    """mean |SHAP| over (H,W,T,chromophore) -> importance of each of the l=11 lagged fNIRS windows."""
    fig, ax = plt.subplots(figsize=(10, 5))
    n_seg = sv.shape[2]
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        m = labels == c
        if m.sum() == 0:
            continue
        ax.plot(np.arange(n_seg) * LAG_STEP_SECONDS, np.abs(sv[c][m]).mean(axis=(0, 2, 3, 4, 5)),
                marker="o", ms=3, color=col, label=cname)
    ax.set_xlabel("fNIRS window lag behind the EEG window start (s)"); ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"SHAP -- fNIRS lag-window importance -- Session {session_id}"); ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_fnirs_segments.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    [saved] {path}")


# ============================================================================
# 2. Decision weights kappa and FGSA gamma
# ============================================================================
def plot_fusion_gating_weights(model, loader, device, output_dir, session_id=None):
    """STA-Net's kappa_fusion/kappa_fnirs are INSTANCE-ADAPTIVE (softmax of an FC layer on the two branch
    features), unlike TSFNet's three fixed scalars -- so they are shown as distributions, not bars."""
    from stanet_diagnostics import plot_prediction_weights
    os.makedirs(output_dir, exist_ok=True)
    d = collect_outputs(model, loader, device, keys=("p_final", "w"))
    path, w = plot_prediction_weights(d, "xai", session_id if session_id is not None else "model", output_dir)
    print(f"  [saved] {path}  (kappa_fusion={w['kappa_fusion_mean']:.3f}, kappa_fnirs={w['kappa_fnirs_mean']:.3f})")
    return w


def plot_fgsa_gating_scalars(model, output_dir, session_id=None):
    """gamma of FGSA-1/2 = sigmoid(residual_para) (weight of the EEG-branch residual in Eq. 1; official code
    initialises residual_para = 0 -> gamma starts at 0.5, the paper text says 0). The fusion branch keeps
    (1-gamma) of its own feature plus the Phi-gated term."""
    os.makedirs(output_dir, exist_ok=True)
    vals = {"gamma1 (FGSA-1)": torch.sigmoid(model.b1.fga.residual_para).item(),
            "gamma2 (FGSA-2)": torch.sigmoid(model.b2.fga.residual_para).item()}
    fig, ax = plt.subplots(figsize=(5, 5))
    bars = ax.bar(vals.keys(), vals.values(), color=["#2a78d6", "#5aa8e0"])
    for b, v in zip(bars, vals.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.1); ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("sigmoid(residual_para) (init 0.5)")
    ax.set_title("FGSA residual gamma" + (f" -- Session {session_id}" if session_id is not None else ""), fontsize=10)
    plt.tight_layout()
    tag = f"S{session_id}_" if session_id is not None else ""
    path = os.path.join(output_dir, f"{tag}fgsa_gamma.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}  ({', '.join(f'{k}={v:.3f}' for k, v in vals.items())})")
    return vals


# ============================================================================
# 3. FGSA spatial attention maps Phi
# ============================================================================
@torch.no_grad()
def plot_fgsa_attention_map(model, loader, device, session_id, output_dir, stage="phi2", n_samples=300):
    """stage 'phi1' (8x8, FGSA-1) or 'phi2' (4x4, FGSA-2). Phi = sigmoid(mean_t conv(F_fnirs)) averaged over
    the l lagged windows (Eq. 2) -> an fNIRS-derived spatial importance of the fusion branch. Resolution is
    the strided conv grid (each cell covers 2x2 / 4x4 pixels of the 16x16 input)."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    store = {c: [] for c in range(3)}
    collected = 0
    for xe, xf, y in loader:
        if collected >= n_samples:
            break
        phi = model(xe.to(device), xf.to(device))[stage].cpu()[:, 0, :, :, 0]     # (B,H,W)
        for c in range(3):
            m = y.numpy() == c
            if m.any():
                store[c].append(phi[m])
        collected += len(y)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, ax in enumerate(axes):
        if not store[c]:
            ax.axis("off"); continue
        v = torch.cat(store[c]).mean(0).numpy()
        im = ax.imshow(v, cmap="viridis")
        ax.set_title(f"{CLASS_NAMES[c]} (n={sum(x.shape[0] for x in store[c])})"); ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"FGSA spatial attention Phi (Eq. 2, from fNIRS) -- {stage} -- Session {session_id}", fontsize=11)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_fgsa_attention_{stage}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ============================================================================
# 4. EGTA attention (+ qualitative HRF overlay) and pruning ablation
# ============================================================================
def _canonical_hrf_curve(t_seconds, peak=6.0, undershoot=16.0, peak_disp=1.0, under_disp=1.0, ratio=6.0):
    """Qualitative double-gamma HRF normalised to peak 1 -- NOT fitted, by-eye reference only."""
    from scipy.stats import gamma as _gamma
    t = np.clip(np.asarray(t_seconds, dtype=float), 1e-6, None)
    hrf = _gamma.pdf(t, peak / peak_disp, scale=peak_disp) - _gamma.pdf(t, undershoot / under_disp,
                                                                         scale=under_disp) / ratio
    m = hrf.max()
    return hrf / m if m > 0 else hrf


def run_egta_attention_analysis(model, test_loader, device, session_id, output_dir, n_samples=300,
                                plot_hrf_overlay=True):
    """Mean EGTA attention (out['att'], heads+query averaged) per fNIRS lag window and class. x-axis =
    midpoint of fNIRS window k relative to the EEG window start (k*1 s + 1.5 s). Returns {class: (l,) curve},
    reusable as `segment_importance` in evaluate_segment_pruning."""
    os.makedirs(output_dir, exist_ok=True)
    d = collect_outputs(model, test_loader, device, keys=("att",), max_batches=-(-n_samples // test_loader.batch_size))
    W, y = d["att"], d["y_true"]
    S = W.shape[1]
    mid = np.arange(S) * LAG_STEP_SECONDS + WINDOW_SECONDS / 2.0
    curves = {}
    fig, ax = plt.subplots(figsize=(10, 5))
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        m = y == c
        if m.sum() == 0:
            continue
        curves[c] = W[m].mean(0)
        ax.plot(mid, curves[c], "-o", ms=3, color=col, label=f"{cname} (n={m.sum()})")
    ax.axhline(1.0 / S, color="gray", linestyle=":", label=f"uniform (1/{S}={1.0 / S:.3f})")
    ax.set_xlabel("fNIRS window midpoint after the EEG window start (s)")
    ax.set_ylabel("Mean EGTA attention weight")
    if plot_hrf_overlay:
        ax2 = ax.twinx()
        t = np.linspace(mid.min(), mid.max(), 200)
        ax2.plot(t, _canonical_hrf_curve(t), "--", color="black", alpha=0.4, lw=1.5,
                 label="canonical HRF (qualitative, NOT fit)")
        ax2.set_ylabel("canonical HRF (normalised)")
        h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8)
    ax.set_title(f"STA-Net EGTA attention over lagged fNIRS windows -- Session {session_id}\n"
                 "(dashed: textbook HRF shape, by-eye reference only; delay is relative to the EEG window)",
                 fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_egta_attention_vs_hrf.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")
    return curves


@torch.no_grad()
def evaluate_segment_pruning(model, test_loader, device, segment_importance, keep_k_list=(1, 3, 5, 7, 9, 11)):
    """Inference-only: keep the top-k lag windows (by `segment_importance`), zero the rest, re-score the
    FINAL head. NOT a substitute for retraining with fewer windows. Returns {k: {'acc','kappa'}}."""
    model.eval()
    order = np.argsort(-np.asarray(segment_importance))
    results = {}
    for k in keep_k_list:
        keep = set(order[:k].tolist())
        P, Y = [], []
        for xe, xf, y in test_loader:
            S = xf.shape[1]
            if k < S:
                xf = xf.clone()
                xf[:, [i for i in range(S) if i not in keep]] = 0.0
            P.append(model(xe.to(device), xf.to(device))["p_final"].argmax(-1).cpu().numpy())
            Y.append(y.numpy())
        P, Y = np.concatenate(P), np.concatenate(Y)
        results[k] = {"acc": float((P == Y).mean()), "kappa": float(cohen_kappa_score(Y, P))}
        print(f"  [prune] keep top-{k:>2d}/{len(order)} -> acc={results[k]['acc'] * 100:.2f}%  "
              f"kappa={results[k]['kappa']:.4f}")
    return results


def plot_segment_pruning_curve(results, session_id, output_dir, full_k=11):
    os.makedirs(output_dir, exist_ok=True)
    ks = sorted(results)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(ks, [results[k]["acc"] * 100 for k in ks], "-o", color="#2a78d6")
    ax1.set_xlabel("lag windows kept (top-k by importance, rest zeroed)")
    ax1.set_ylabel("Accuracy (%)", color="#2a78d6")
    ax2 = ax1.twinx()
    ax2.plot(ks, [results[k]["kappa"] for k in ks], "-s", color="#e24b4a")
    ax2.set_ylabel("Cohen's kappa", color="#e24b4a")
    if full_k in results:
        ax1.axhline(results[full_k]["acc"] * 100, color="#2a78d6", ls=":", alpha=0.5)
    ax1.set_title(f"Hard-pruning ablation (inference-only, no retrain) -- Session {session_id}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_segment_pruning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ============================================================================
# 5. t-SNE: single subject and group level
# ============================================================================
def run_single_subject_tsne(model, train_loader, test_loader, device, subject_id, session_id, output_dir,
                            feature_extractor=None, perplexity=30):
    """Analogue of MSTFDN Fig 16: PCA of the raw EEG ('initial state') vs t-SNE of the trained fusion
    feature, plus the confusion matrix. train_loader is accepted for signature parity (unused)."""
    feature_extractor = feature_extractor or extract_features
    os.makedirs(output_dir, exist_ok=True)
    data = feature_extractor(model, test_loader, device)
    labels, preds = data["labels"], np.argmax(data["probs"], axis=1)

    raw = np.concatenate([xe[..., ::10].reshape(len(xe), -1).numpy() for xe, _, _ in test_loader])  # 20 Hz view
    from sklearn.decomposition import PCA
    Z0 = PCA(n_components=2, random_state=42).fit_transform(StandardScaler().fit_transform(raw))
    Xs = StandardScaler().fit_transform(data["feat_fusion"])
    Z1 = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(Xs) // 4)), random_state=42,
              max_iter=1000).fit_transform(Xs)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, Z, ttl in ((axes[0], Z0, "Initial state (PCA on raw EEG input)"),
                       (axes[1], Z1, "Trained state (t-SNE, fusion feature)")):
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            m = labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.45, color=col, label=cname)
        ax.set_title(ttl); ax.legend(markerscale=3, fontsize=8)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    axes[2].imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    axes[2].set_xticks(range(3)); axes[2].set_yticks(range(3))
    axes[2].set_xticklabels(CLASS_NAMES); axes[2].set_yticklabels(CLASS_NAMES)
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{cmn[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center",
                         color="white" if cmn[i, j] > 55 else "black", fontsize=8)
    axes[2].set_title(f"Confusion Matrix (accuracy={cm.diagonal().sum() / cm.sum() * 100:.1f}%)")
    fig.suptitle(f"Single-subject t-SNE -- {subject_id}, Session {session_id} (STA-Net)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{subject_id}_S{session_id}_single_subject_tsne.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"    [saved] {path}")


def run_group_level_tsne(model_builder, subject_checkpoints, loader_builder, device="cpu", output_dir=".",
                         feature_extractor=None, max_subjects=26, perplexity=30):
    """model_builder: () -> STANet (e.g. lambda: STANet(3)). subject_checkpoints: {key: .pt path}
    (`VPxxx_session{k}_best.pt` from run_stanet_loso.py). loader_builder: key -> WindowLoader for that
    fold's TEST windows (built on demand so only one subject is in RAM at a time)."""
    feature_extractor = feature_extractor or extract_features
    os.makedirs(output_dir, exist_ok=True)
    keys = list(subject_checkpoints)[:max_subjects]
    feats = {"eeg": [], "fnirs": [], "fusion": []}
    labels_l, preds_l = [], []
    for i, key in enumerate(keys):
        model = model_builder().to(device)
        model.load_state_dict(torch.load(subject_checkpoints[key], map_location=device))
        model.eval()
        loader = loader_builder(key)
        d = feature_extractor(model, loader, device)
        for k in feats:
            feats[k].append(d[f"feat_{k}"])
        labels_l.append(d["labels"]); preds_l.append(np.argmax(d["probs"], axis=1))
        print(f"    [{i + 1}/{len(keys)}] {key}: {len(d['labels'])} windows")
        del model, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    labels, preds = np.concatenate(labels_l), np.concatenate(preds_l)
    for k, title in (("eeg", "EEG branch"), ("fnirs", "fNIRS branch (EGTA-aligned)"), ("fusion", "Fusion branch")):
        X = StandardScaler().fit_transform(np.concatenate(feats[k]))
        n_max = min(6000, len(X))
        idx = np.random.choice(len(X), n_max, replace=False)
        Z = TSNE(n_components=2, perplexity=min(perplexity, max(5, n_max // 10)), random_state=42,
                 max_iter=1000).fit_transform(X[idx])
        yp, pp = labels[idx], preds[idx]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            m = yp == c
            axes[0].scatter(Z[m, 0], Z[m, 1], s=5, alpha=0.35, color=col, label=cname)
        axes[0].set_title(f"true class (pooled over {len(keys)} entries)"); axes[0].legend(markerscale=3)
        wrong = pp != yp
        axes[1].scatter(Z[~wrong, 0], Z[~wrong, 1], s=4, alpha=0.3, color="#1baf7a", label="correct")
        axes[1].scatter(Z[wrong, 0], Z[wrong, 1], s=8, alpha=0.6, color="#e24b4a", marker="x", label="misclassified")
        axes[1].set_title(f"correct vs misclassified (error={wrong.mean() * 100:.1f}%)"); axes[1].legend(markerscale=3)
        fig.suptitle(f"Group-level t-SNE -- {title}", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"group_tsne_{k}.png"), dpi=150, bbox_inches="tight"); plt.close()
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 55 else "black", fontsize=9)
    ax.set_title(f"Pooled confusion matrix (accuracy={cm.diagonal().sum() / cm.sum() * 100:.1f}%)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "group_confusion_matrix.png"), dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n  Group-level analyses saved -> {output_dir}\n")


# ============================================================================
# 6. Grad-CAM (Selvaraju et al. 2017), generalised to 3-D activation maps
# ============================================================================
class GradCAM3D:
    """Hook a module whose output (or output[0]) is the activation A (B,C,H,W,T). compute():
        alpha_k = GAP over (H,W,T) of dY/dA_k ;  CAM = ReLU(sum_k alpha_k A^k).
    FIX vs the TSFNet version: retained activation grads ACCUMULATE across backward() calls, so
    compute() clears act.grad -> every class gets its own, uncontaminated gradient."""

    def __init__(self, model, target_module):
        self.model, self.activation = model, None
        self._h = target_module.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        act.retain_grad()
        self.activation = act

    def compute(self):
        act = self.activation
        if act is None or act.grad is None:
            raise RuntimeError("run a forward pass and score.backward() first")
        alpha = act.grad.mean(dim=tuple(range(2, act.grad.dim())), keepdim=True)
        cam = F.relu((alpha * act).sum(dim=1)).detach()
        act.grad = None
        return cam

    def remove(self):
        self._h.remove()


def run_gradcam_analysis(model, test_loader, device, session_id, output_dir, n_samples=200,
                         window_seconds=WINDOW_SECONDS):
    """Grad-CAM on the three branches for the three classes. Targets: EEG = b2.eeg_bn output (last conv
    block, pre-ELU), fNIRS = b2.fn_bn output (averaged over the 11 lag windows), fusion = FGSA-2 output.
    Score = p_final[:, c] (the model exposes probabilities, not logits)."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    sp = {b: {c: [] for c in range(3)} for b in ("eeg", "fnirs", "fusion")}
    tp = {b: {c: [] for c in range(3)} for b in ("eeg", "fusion")}
    collected = 0
    for xe, xf, y in test_loader:
        if collected >= n_samples:
            break
        xe, xf = xe.to(device), xf.to(device)
        B, S = xf.shape[:2]
        cams = {"eeg": GradCAM3D(model, model.b2.eeg_bn), "fnirs": GradCAM3D(model, model.b2.fn_bn),
                "fusion": GradCAM3D(model, model.b2.fga)}
        out = model(xe, xf)
        for c in range(3):
            m = y.numpy() == c
            if not m.any():
                continue
            model.zero_grad()
            out["p_final"][m, c].sum().backward(retain_graph=True)
            ce = cams["eeg"].compute()[m]                                           # (n,4,4,50)
            cf = cams["fnirs"].compute()                                            # (B*S,4,4,8)
            cf = cf.reshape(B, S, *cf.shape[1:]).mean(dim=1)[m]
            cu = cams["fusion"].compute()[m]
            for b, v in (("eeg", ce), ("fnirs", cf), ("fusion", cu)):
                sp[b][c].append(v.mean(dim=-1).cpu())
            tp["eeg"][c].append(ce.mean(dim=(1, 2)).cpu()); tp["fusion"][c].append(cu.mean(dim=(1, 2)).cpu())
        for cm_ in cams.values():
            cm_.remove()
        collected += B

    for b, bt in (("eeg", "EEG"), ("fnirs", "fNIRS"), ("fusion", "Fusion")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, ax in enumerate(axes):
            if not sp[b][c]:
                ax.axis("off"); continue
            im = ax.imshow(torch.cat(sp[b][c]).mean(0).numpy(), cmap="jet")
            ax.set_title(f"{CLASS_NAMES[c]} (n={sum(x.shape[0] for x in sp[b][c])})"); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"Grad-CAM -- {bt} branch spatial map (4x4 feature grid) -- Session {session_id}", fontsize=11)
        plt.tight_layout()
        path = os.path.join(output_dir, f"S{session_id}_gradcam_{b}_spatial.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")
    for b, bt in (("eeg", "EEG"), ("fusion", "Fusion")):
        fig, ax = plt.subplots(figsize=(12, 5))
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            if tp[b][c]:
                v = torch.cat(tp[b][c]).mean(0).numpy()
                ax.plot(np.linspace(0, window_seconds, len(v)), v, color=col, label=cname, lw=1.3)
        ax.set_xlabel("Time within the 3 s window (s)"); ax.set_ylabel("Grad-CAM (averaged over space)")
        ax.set_title(f"Grad-CAM -- {bt} branch temporal profile -- Session {session_id}"); ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, f"S{session_id}_gradcam_{b}_temporal.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")
    print(f"\n  Grad-CAM analysis complete -> {output_dir}\n")
