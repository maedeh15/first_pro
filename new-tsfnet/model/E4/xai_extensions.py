"""
xai_extensions.py -- Explainability additions for TSFNet (priority: XAI stage)
================================================================================
This module is an ADD-ON to `model_diagnostics.py` (already present in the
project, which does PCA/t-SNE/confusion-matrix/gradient-saliency on a
per-subject-per-session basis). The goal here is to cover exactly the
interpretability methods used in the two reference papers (MSTFDN and
Adaptive Fusion) that were missing from the existing pipeline -- not to
rewrite anything that already works.

┌─────────────────────────────────────────────────────────────────────────┐
│ Paper method -> implementation here -> why / why not                     │
├─────────────────────────────────────────────────────────────────────────┤
│ SHAP on hand-crafted features (Adaptive Fusion, RF/XGBoost/MLP)          │
│   -> Our architecture is a deep 3D-CNN, not a shallow model on hand-     │
│      crafted features. So shap.GradientExplainer is applied directly to │
│      the raw inputs (x_eeg: 16x16x2000 grid, and x_fnirs) -- the correct│
│      SHAP analogue for continuous deep models (DeepSHAP/GradientExplainer│
│      rather than KernelExplainer, which is meant for tree/linear        │
│      models). Output: a spatio-temporal importance map per class, for   │
│      both branches.                                                     │
│                                                                           │
│ SHAP on the gating model (Adaptive Fusion, Fig 11-12)                    │
│   -> Our architecture has no instance-adaptive gating network; w_eeg/   │
│      w_fusion/w_fnirs are three fixed, model-level scalars (Eq. 4 of the │
│      TSFNet paper), not a network that learns a weight per instance.    │
│      SHAP on gating therefore has no meaning here; instead these fixed  │
│      weights are reported and plotted directly (plot_fusion_gating_     │
│      weights) -- an honest treatment of the difference, rather than     │
│      faking an instance-adaptive gate that doesn't exist.               │
│                                                                           │
│ Channel-attention map + head-topography overlay (MSTFDN, Fig 15)         │
│   -> TSFNet already has its own explicit attention map: the EFGF layer's│
│      hybrid attention map Phi (already returned in aux['phi_pcc1']/     │
│      ['phi_pcc2'], no model change needed). Here Phi is averaged per    │
│      class and plotted as a heatmap over the 16x16 grid.                │
│      Honest limitation (also noted in this project's own preprocessing  │
│      comments): this 16x16 grid is the result of cubic interpolation    │
│      over an azimuthal-equidistant projection -- each cell does NOT     │
│      correspond to one real electrode. So this map should not be read   │
│      as "channel Fp1 has value X", but as an approximate spatial map of │
│      the active scalp region.                                           │
│                                                                           │
│ CAFE cross-modal attention (fNIRS segment importance)                    │
│   -> Extracted with a forward hook on the two nn.MultiheadAttention     │
│      modules inside CAFELayer (no model code change) -- by default      │
│      nn.MultiheadAttention returns attn_weights, currently discarded    │
│      with ", _" in the model's forward(); here we capture it via a hook │
│      on the module's own output (not CAFELayer's output).               │
│                                                                           │
│ t-SNE for N subjects (MSTFDN, Fig 16, cross-subject aggregation)         │
│   -> The existing module (`model_diagnostics._plot_tsne`) is only per-  │
│      subject-per-session. Here `run_group_level_tsne` loops over a set  │
│      of subject_dirs (default: all 26), pools each subject's 512-dim   │
│      features via the existing _extract_features, and builds a single  │
│      t-SNE over the pooled data -- exactly matching MSTFDN's approach   │
│      for showing the feature space at the group level, not just one    │
│      fold.                                                              │
│                                                                           │
│ Grad-CAM (Selvaraju et al. 2017, ICCV / arXiv:1610.02391)                │
│   -> Complements SHAP, does not replace it: SHAP works on the raw input │
│      (pixel/time sample), Grad-CAM works on the last convolutional      │
│      activation map before GAP in each branch (per the paper's explicit │
│      recommendation for choosing the target layer). Its cost is much    │
│      lower than GradientExplainer (one forward+backward instead of      │
│      dozens of background samples), so it's more practical at the scale │
│      of all 26 subjects. LIME was deliberately NOT added -- none of     │
│      TSFNet/MSTFDN/Adaptive Fusion use it, and it has no natural        │
│      conceptual fit with a continuous EEG/fNIRS tensor (no natural unit │
│      analogous to an image superpixel).                                 │
└─────────────────────────────────────────────────────────────────────────┘

Prerequisite install:
    pip install shap --break-system-packages

Usage (example, in the same notebook where the model is trained):

    from xai_extensions import (
        run_shap_analysis, plot_fusion_gating_weights,
        plot_efgf_attention_map, run_cafe_attention_analysis,
        run_group_level_tsne, run_gradcam_analysis,
    )

    # After training one fold (or loading a checkpoint):
    run_shap_analysis(model, test_loader, device, session_id=test_session,
                       output_dir=diagnostics_dir, n_background=32, n_explain=64)

    plot_fusion_gating_weights(model, output_dir=diagnostics_dir)

    plot_efgf_attention_map(model, test_loader, device, session_id=test_session,
                             output_dir=diagnostics_dir)

    run_cafe_attention_analysis(model, test_loader, device, session_id=test_session,
                                 output_dir=diagnostics_dir)

    run_gradcam_analysis(model, test_loader, device, session_id=test_session,
                          output_dir=diagnostics_dir, n_samples=200)

    # Group-level analysis (once, after checkpoints for all 26 subjects are ready):
    run_group_level_tsne(
        subject_dirs={"VP001": "...\\VP001", ..., "VP026": "...\\VP026"},
        results_dir_template="...\\results_.../{subject_id}_session{session}_best.pt",
        device="cuda", output_dir="...\\group_level_xai",
    )
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
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLASS_NAMES = ["0-back", "2-back", "3-back"]
COLORS = ["#2a78d6", "#e24b4a", "#1baf7a"]


# ============================================================================
# 1. SHAP -- GradientExplainer on raw EEG and fNIRS inputs
# ============================================================================

class _WrappedBranchModel(torch.nn.Module):
    """SHAP's GradientExplainer expects a model that maps a SINGLE input
    tensor -> a SINGLE output tensor. TSFNet.forward takes (x_eeg, x_fnirs)
    and returns a 5-tuple, so this thin wrapper freezes one modality's input
    (as a fixed background-mean tensor) and exposes only the other modality
    as the explained input, with y_pred (the final decision) as the output.
    This mirrors the standard practice of explaining one modality at a time
    (matches how Adaptive Fusion's SHAP results are reported per-modality).
    """

    def __init__(self, model, fixed_fnirs=None, fixed_eeg=None):
        super().__init__()
        self.model = model
        self.fixed_fnirs = fixed_fnirs
        self.fixed_eeg = fixed_eeg

    def forward(self, x):
        if self.fixed_fnirs is not None:
            # x is x_eeg; broadcast the fixed fNIRS background to match batch size
            fn = self.fixed_fnirs.expand(x.shape[0], *self.fixed_fnirs.shape[1:])
            _, _, _, y_pred, _ = self.model(x, fn)
        else:
            eg = self.fixed_eeg.expand(x.shape[0], *self.fixed_eeg.shape[1:])
            _, _, _, y_pred, _ = self.model(eg, x)
        return y_pred  # (B, 3) softmax probabilities


def run_shap_analysis(model, test_loader, device, session_id, output_dir,
                       n_background=32, n_explain=64):
    """
    Compute SHAP (GradientExplainer) for both the EEG and fNIRS branches,
    per class, and plot spatial/temporal importance maps.

    n_background: number of background (baseline) samples for
        GradientExplainer -- averaging over these samples acts as the
        "neutral state" (exactly like the background set in a standard
        shap.DeepExplainer/GradientExplainer call; 32 random samples from
        the same test set is common and sufficient in practice).
    n_explain: number of samples SHAP is actually computed for
        (GradientExplainer's cost on a 3D-CNN tensor is fairly heavy; 64
        samples takes a few minutes in practice -- reduce if needed).
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "The shap package is not installed. Run: "
            "pip install shap --break-system-packages"
        )

    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # Collect one large batch to draw both the background and the
    # explained samples from.
    eeg_all, fnirs_all, lbl_all = [], [], []
    for batch in test_loader:
        # Compatible with both the 3-tuple/N-tuple NBackDataset outputs --
        # only the first three fields are needed here.
        eeg_b, fnirs_b = batch[0], batch[1]
        lbl_b = batch[2]
        eeg_all.append(eeg_b)
        fnirs_all.append(fnirs_b)
        lbl_all.append(lbl_b)
        if sum(x.shape[0] for x in eeg_all) >= n_background + n_explain:
            break
    eeg_all = torch.cat(eeg_all, dim=0).float()
    fnirs_all = torch.cat(fnirs_all, dim=0).float()
    lbl_all = torch.cat(lbl_all, dim=0)

    n_avail = eeg_all.shape[0]
    n_bg = min(n_background, n_avail // 2)
    n_ex = min(n_explain, n_avail - n_bg)

    idx_bg = np.random.choice(n_avail, n_bg, replace=False)
    remaining = np.setdiff1d(np.arange(n_avail), idx_bg)
    idx_ex = np.random.choice(remaining, n_ex, replace=False)

    eeg_bg = eeg_all[idx_bg].to(device)
    fnirs_bg = fnirs_all[idx_bg].to(device)
    eeg_ex = eeg_all[idx_ex].to(device)
    fnirs_ex = fnirs_all[idx_ex].to(device)
    lbl_ex = lbl_all[idx_ex].numpy()

    print(f"  [SHAP] background={n_bg} samples, explain={n_ex} samples ...")

    results = {}

    # ---- EEG branch: variable input = x_eeg, fNIRS held fixed at background mean ----
    fnirs_mean = fnirs_bg.mean(dim=0, keepdim=True)
    wrapped_eeg = _WrappedBranchModel(model, fixed_fnirs=fnirs_mean).to(device)
    explainer_eeg = shap.GradientExplainer(wrapped_eeg, eeg_bg)
    shap_vals_eeg = explainer_eeg.shap_values(eeg_ex)  # list[3], each (N,16,16,600), or an array (N,16,16,600,3)
    shap_vals_eeg = _normalize_shap_output(shap_vals_eeg, n_classes=3)
    results["eeg"] = shap_vals_eeg

    # ---- fNIRS branch: variable input = x_fnirs, EEG held fixed at background mean ----
    eeg_mean = eeg_bg.mean(dim=0, keepdim=True)
    wrapped_fnirs = _WrappedBranchModel(model, fixed_eeg=eeg_mean).to(device)
    explainer_fnirs = shap.GradientExplainer(wrapped_fnirs, fnirs_bg)
    shap_vals_fnirs = explainer_fnirs.shap_values(fnirs_ex)
    shap_vals_fnirs = _normalize_shap_output(shap_vals_fnirs, n_classes=3)
    results["fnirs"] = shap_vals_fnirs

    _plot_shap_eeg_spatial(shap_vals_eeg, lbl_ex, session_id, output_dir)
    _plot_shap_eeg_temporal(shap_vals_eeg, lbl_ex, session_id, output_dir)
    _plot_shap_fnirs_segment_importance(shap_vals_fnirs, lbl_ex, session_id, output_dir)

    # Save the raw values for later analysis / cross-subject aggregation
    np.savez(
        os.path.join(output_dir, f"S{session_id}_shap_raw.npz"),
        shap_eeg=shap_vals_eeg, shap_fnirs=shap_vals_fnirs, labels=lbl_ex,
    )
    print(f"  [SHAP] saved raw values + plots -> {output_dir}")
    return results


def _normalize_shap_output(shap_values, n_classes):
    """shap.GradientExplainer can return either a list of per-class arrays
    or a single array with a trailing class dimension depending on shap
    version; this normalizes to shape (n_classes, N, *input_dims)."""
    if isinstance(shap_values, list):
        return np.stack(shap_values, axis=0)
    arr = np.asarray(shap_values)
    if arr.shape[-1] == n_classes:
        return np.moveaxis(arr, -1, 0)
    return arr


def _plot_shap_eeg_spatial(shap_vals_eeg, labels, session_id, output_dir):
    """Mean |SHAP| over the time axis -> a 16x16 spatial map per class.
    Warning (per this project's own preprocessing comments): this grid is
    the result of cubic interpolation from 28 real electrodes; each cell is
    NOT one real electrode."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, ax in enumerate(axes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        v = np.abs(shap_vals_eeg[c][mask]).mean(axis=(0, -1))  # (16,16)
        im = ax.imshow(v, cmap="inferno")
        ax.set_title(f"{CLASS_NAMES[c]} (n={mask.sum()})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"SHAP |importance| -- EEG spatial map -- Session {session_id}\n"
        "(interpolated 16x16 grid -- each cell does not correspond to a real electrode)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_eeg_spatial.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [saved] {path}")


def _plot_shap_eeg_temporal(shap_vals_eeg, labels, session_id, output_dir,
                             window_seconds=10.0):
    """Mean |SHAP| over both spatial axes -> a temporal profile per class."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = labels == c
        if mask.sum() == 0:
            continue
        v = np.abs(shap_vals_eeg[c][mask]).mean(axis=(0, 1, 2))  # (T,)
        t = np.linspace(0, window_seconds, len(v))
        ax.plot(t, v, color=col, label=cname, lw=1.3)
    ax.set_xlabel("Time within window (seconds)")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"SHAP -- EEG temporal profile -- Session {session_id}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_eeg_temporal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [saved] {path}")


def _plot_shap_fnirs_segment_importance(shap_vals_fnirs, labels, session_id, output_dir):
    """Mean |SHAP| over (H,W,T,Chromophore) -> importance of each fNIRS
    time segment (the Segment axis, exactly the same axis discussed as
    "which fNIRS segments carry information" in the G3/CAFE reports)."""
    # shap_vals_fnirs[c]: (N, S, H, W, T, C)
    fig, ax = plt.subplots(figsize=(10, 5))
    n_seg = shap_vals_fnirs.shape[1]
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = labels == c
        if mask.sum() == 0:
            continue
        v = np.abs(shap_vals_fnirs[c][mask]).mean(axis=(0, 2, 3, 4, 5))  # (S,)
        ax.plot(np.arange(n_seg), v, marker="o", ms=3, color=col, label=cname)
    ax.set_xlabel("fNIRS segment index")
    ax.set_ylabel("Mean |SHAP|")
    ax.set_title(f"SHAP -- fNIRS segment importance -- Session {session_id}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_shap_fnirs_segments.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    [saved] {path}")


# ============================================================================
# 2. Fixed gating weights (w_eeg, w_fusion, w_fnirs) -- the static analogue
#    of the "modality-level importance" reported in Adaptive Fusion (Fig 11)
# ============================================================================

def plot_fusion_gating_weights(model, output_dir, session_id=None):
    """Plot the three learned, model-level weights (Eq. 4). Unlike Adaptive
    Fusion, which has an instance-adaptive gating network, these are fixed
    scalars -- the title/text make this difference explicit so it isn't
    mistaken for an instance-adaptive gate."""
    os.makedirs(output_dir, exist_ok=True)
    w = {
        "EEG": torch.sigmoid(model.w_eeg).item(),
        "Fusion": torch.sigmoid(model.w_fusion).item(),
        "fNIRS": torch.sigmoid(model.w_fnirs).item(),
    }
    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(w.keys(), w.values(), color=["#2a78d6", "#f5a623", "#e24b4a"])
    for b, v in zip(bars, w.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Learned sigmoid weight (Eq. 4)")
    title = "Fixed decision-fusion weights (model-level, not instance-adaptive)"
    if session_id is not None:
        title += f" -- Session {session_id}"
    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    tag = f"S{session_id}_" if session_id is not None else ""
    path = os.path.join(output_dir, f"{tag}fusion_gating_weights.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}  (w_eeg={w['EEG']:.3f}, w_fusion={w['Fusion']:.3f}, w_fnirs={w['fNIRS']:.3f})")
    return w


# ============================================================================
# 3. EFGF attention map (Phi) -- a direct, zero-extra-cost analogue of the
#    MSTFDN channel-attention heatmap (aux['phi_pcc*'] is already produced
#    by TSFNet.forward; it just needed per-class averaging and plotting).
# ============================================================================

def plot_efgf_attention_map(model, test_loader, device, session_id, output_dir,
                             stage="phi_pcc2", n_samples=300):
    """stage: 'phi_pcc1' (8x8 resolution, EFGF stage 1) or 'phi_pcc2'
    (4x4 resolution, stage 2 -- default, since it is closer to the deeper
    layers and is usually more meaningful)."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    store = {c: [] for c in range(3)}
    collected = 0
    with torch.no_grad():
        for batch in test_loader:
            if collected >= n_samples:
                break
            eeg_b, fnirs_b, lbl_b = batch[0].to(device), batch[1].to(device), batch[2]
            _, _, _, _, aux = model(eeg_b, fnirs_b)
            phi = aux[stage].detach().cpu()  # (B, H, W, T)
            phi_spatial = phi.mean(dim=-1)   # average over time -> (B, H, W)
            for c in range(3):
                m = (lbl_b.numpy() == c)
                if m.any():
                    store[c].append(phi_spatial[m])
            collected += len(lbl_b)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, ax in enumerate(axes):
        if not store[c]:
            ax.axis("off")
            continue
        v = torch.cat(store[c], dim=0).mean(dim=0).numpy()
        im = ax.imshow(v, cmap="viridis")
        ax.set_title(f"{CLASS_NAMES[c]}  (n={sum(x.shape[0] for x in store[c])})")
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"EFGF hybrid attention map (Phi, Eq. 2) -- {stage} -- Session {session_id}\n"
        "(same approximate 16x16 grid -- the interpolation caveat from the SHAP plots above applies here too)",
        fontsize=11,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_efgf_attention_{stage}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")


# ============================================================================
# 4. CAFE cross-attention (fNIRS segment importance as seen by the model's
#    own attention module -- complements the SHAP-based view above, since
#    this is the network's real attention weight, not a gradient-based
#    importance score).
# ============================================================================

def run_cafe_attention_analysis(model, test_loader, device, session_id, output_dir,
                                 n_samples=300):
    """Uses a forward hook on the two nn.MultiheadAttention modules inside
    CAFELayer to capture the real attention weight (currently discarded
    with ", _" in the model's forward()) -- with no change to the model
    code."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    captured = {"fnirs_to_fusion": [], "fusion_to_fnirs": []}

    def _hook(key):
        def _h(module, inp, out):
            # out = (attn_output, attn_output_weights); weights averaged over heads by default
            attn_weights = out[1]
            if attn_weights is not None:
                captured[key].append(attn_weights.detach().cpu())
        return _h

    h1 = model.cafe.cross_attn_fnirs_to_fusion.register_forward_hook(_hook("fnirs_to_fusion"))
    h2 = model.cafe.cross_attn_fusion_to_fnirs.register_forward_hook(_hook("fusion_to_fnirs"))

    labels_all = []
    collected = 0
    with torch.no_grad():
        for batch in test_loader:
            if collected >= n_samples:
                break
            eeg_b, fnirs_b, lbl_b = batch[0].to(device), batch[1].to(device), batch[2]
            model(eeg_b, fnirs_b)
            labels_all.append(lbl_b.numpy())
            collected += len(lbl_b)
    h1.remove(); h2.remove()

    labels_all = np.concatenate(labels_all, axis=0)[:collected]
    # fnirs_to_fusion: query=fnirs tokens (S segments) attending to 1 fusion token
    # -> weights shape (B, S, 1) -- exactly "how much each fNIRS segment listens
    # to the fusion representation", i.e. segment-level importance.
    w_f2f = torch.cat(captured["fnirs_to_fusion"], dim=0).squeeze(-1).numpy()  # (N, S)

    fig, ax = plt.subplots(figsize=(10, 5))
    n_seg = w_f2f.shape[1]
    for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
        mask = labels_all == c
        if mask.sum() == 0:
            continue
        v = w_f2f[mask].mean(axis=0)
        ax.plot(np.arange(n_seg), v, marker="o", ms=3, color=col, label=cname)
    ax.set_xlabel("fNIRS segment index")
    ax.set_ylabel("Cross-attention weight (fNIRS -> Fusion)")
    ax.set_title(f"CAFE -- fNIRS segment importance from the model's real attention -- Session {session_id}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, f"S{session_id}_cafe_segment_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")
    return w_f2f, labels_all


# ============================================================================
# 5-b. Single representative-subject t-SNE (like Subject #00 in MSTFDN,
#    Fig 16) -- complements run_group_level_tsne; covers both paper
#    approaches (a single qualitative example + a group-level aggregate),
#    not just one of them.
# ============================================================================

def run_single_subject_tsne(model, train_loader, test_loader, device,
                             subject_id, session_id, output_dir,
                             feature_extractor=None, perplexity=30):
    """A direct analogue of MSTFDN Fig 16: pick one subject as a qualitative
    example and show the initial-state feature distribution (raw/PCA-
    reduced) side by side with the trained-state distribution (post-
    training 512-dim feature space per branch) -- plus that subject's
    confusion matrix.

    Comparing "before/after training" would strictly need two models: an
    untrained one (initial state) and a trained one (trained state, the
    `model` argument here). If an untrained model isn't kept around
    (keeping a copy is costly), a 2D PCA of the raw input is used instead
    as the "initial state" baseline -- a reasonable substitute that still
    preserves the paper's before/after concept.
    """
    if feature_extractor is None:
        try:
            from model_diagnostics import _extract_features as feature_extractor
        except ImportError:
            raise ImportError(
                "feature_extractor was not provided and model_diagnostics.py "
                "is not importable either."
            )

    os.makedirs(output_dir, exist_ok=True)
    print(f"  [single-subject t-SNE] {subject_id}, session {session_id} ...")

    data = feature_extractor(model, test_loader, device)
    labels = data["labels"]
    preds = np.argmax(data["probs"], axis=1)

    # "Initial state": 2D PCA of the raw (flattened) EEG input, used as the
    # visual baseline before any learned representation -- lighter-weight
    # than keeping a full untrained copy of the model, while preserving the
    # paper's "before/after" framing.
    raw_eeg = []
    for batch in test_loader:
        raw_eeg.append(batch[0].reshape(batch[0].shape[0], -1).numpy())
    raw_eeg = np.concatenate(raw_eeg, axis=0)
    from sklearn.decomposition import PCA
    Z_initial = PCA(n_components=2, random_state=42).fit_transform(
        StandardScaler().fit_transform(raw_eeg))

    for feat_key, title in [("feat_fusion", "Fusion branch (final representation)")]:
        X = data[feat_key]
        X_s = StandardScaler().fit_transform(X)
        eff_perp = min(perplexity, max(5, len(X_s) // 4))
        Z_trained = TSNE(n_components=2, perplexity=eff_perp, random_state=42,
                          max_iter=1000).fit_transform(X_s)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        ax = axes[0]
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            mask = labels == c
            ax.scatter(Z_initial[mask, 0], Z_initial[mask, 1], s=8, alpha=0.45,
                       color=col, label=cname)
        ax.set_title("Initial state (PCA on raw EEG input)")
        ax.legend(markerscale=3, fontsize=8)

        ax = axes[1]
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            mask = labels == c
            ax.scatter(Z_trained[mask, 0], Z_trained[mask, 1], s=8, alpha=0.45,
                       color=col, label=cname)
        ax.set_title(f"Trained state (t-SNE, {title})")
        ax.legend(markerscale=3, fontsize=8)

        ax = axes[2]
        cm = confusion_matrix(labels, preds)
        cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        im = ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{cmn[i,j]:.1f}%\n({cm[i,j]})", ha="center", va="center",
                        color="white" if cmn[i, j] > 55 else "black", fontsize=8)
        ax.set_title(f"Confusion Matrix (accuracy={cm.diagonal().sum()/cm.sum()*100:.1f}%)")

        fig.suptitle(f"Single representative-subject t-SNE -- {subject_id}, Session {session_id}"
                     f"\n(direct analogue of MSTFDN Fig 16 -- Subject #00)", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{subject_id}_S{session_id}_single_subject_tsne.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [saved] {path}")


# ============================================================================
# 5. Group-level t-SNE over N subjects (default N=26) -- extends the MSTFDN
#    Fig 16 approach, which was done on a pooled multi-subject set, not
#    just a single fold.
#
#    Uses the _extract_features function already defined in this project's
#    model_diagnostics.py (re-imported here so the dependency is explicit
#    and no duplicate code is written).
# ============================================================================

def run_group_level_tsne(model_builder, subject_checkpoints, subject_test_loaders,
                          device, output_dir, feature_extractor=None,
                          max_subjects=26, perplexity=30):
    """
    Parameters
    ----------
    model_builder : callable() -> TSFNet
        A no-argument function that returns a fresh TSFNet instance with
        the same hyperparameter configuration being evaluated (checkpoints
        are state_dicts only, so the architecture must already be known).
    subject_checkpoints : dict[str, str]
        Mapping subject_id -> .pt file path (the same
        "{subject_id}_session{test_session}_best.pt" pattern saved by
        run_all_subjects.py). One representative session per subject is
        enough; to include all three sessions/subject as well, pass
        several entries keyed by "{subject_id}_S{session}".
    subject_test_loaders : dict[str, DataLoader]
        Mapping the same keys -> the corresponding test DataLoader.
    feature_extractor : callable(model, loader, device) -> dict
        Default: the _extract_features function from this project's
        model_diagnostics.py (if None, an attempt is made to import it
        from tsfnet_model_notebook).
    max_subjects : maximum number of subjects to include (default: all 26 --
        as requested, "for however many of the 26 is needed and standard";
        the standard practice in comparable papers (MSTFDN) is to aggregate
        over every available subject, not an arbitrary subset).
    """
    if feature_extractor is None:
        try:
            from model_diagnostics import _extract_features as feature_extractor
        except ImportError:
            raise ImportError(
                "feature_extractor was not provided and model_diagnostics.py "
                "is not importable either. Either place that module next to "
                "this file, or pass feature_extractor explicitly."
            )

    os.makedirs(output_dir, exist_ok=True)
    keys = list(subject_checkpoints.keys())[:max_subjects]
    print(f"  [group t-SNE] aggregating {len(keys)} subject/session entries ...")

    all_feat = {"eeg": [], "fnirs": [], "fusion": []}
    all_labels, all_subj_id, all_preds = [], [], []

    for i, key in enumerate(keys):
        model = model_builder().to(device)
        model.load_state_dict(torch.load(subject_checkpoints[key], map_location=device))
        model.eval()

        data = feature_extractor(model, subject_test_loaders[key], device)
        all_feat["eeg"].append(data["feat_eeg"])
        all_feat["fnirs"].append(data["feat_fnirs"])
        all_feat["fusion"].append(data["feat_fusion"])
        all_labels.append(data["labels"])
        all_preds.append(np.argmax(data["probs"], axis=1))
        all_subj_id.append(np.full(len(data["labels"]), i))

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"    [{i+1}/{len(keys)}] {key}: {len(data['labels'])} windows")

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    subj_id = np.concatenate(all_subj_id)

    for feat_key, title in [("eeg", "EEG branch"), ("fnirs", "fNIRS branch (CAFE-enhanced)"),
                             ("fusion", "Fusion branch")]:
        X = np.concatenate(all_feat[feat_key], axis=0)
        X_s = StandardScaler().fit_transform(X)

        n_max = min(6000, len(X_s))  # practical cap on t-SNE runtime
        idx = np.random.choice(len(X_s), n_max, replace=False)
        Xp, yp, predp, sidp = X_s[idx], labels[idx], preds[idx], subj_id[idx]

        eff_perp = min(perplexity, max(5, n_max // 10))
        print(f"    t-SNE ({feat_key}, {n_max} windows from {len(keys)} subjects, "
              f"perplexity={eff_perp}) ...", end="", flush=True)
        Z = TSNE(n_components=2, perplexity=eff_perp, random_state=42,
                 max_iter=1000).fit_transform(Xp)
        print(" done")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            mask = yp == c
            ax.scatter(Z[mask, 0], Z[mask, 1], s=5, alpha=0.35, color=col, label=cname)
        ax.set_title(f"Colour = true class (pooled over {len(keys)} subjects)")
        ax.legend(markerscale=3, fontsize=9)

        ax = axes[1]
        wrong = predp != yp
        ax.scatter(Z[~wrong, 0], Z[~wrong, 1], s=4, alpha=0.3, color="#1baf7a", label="Correct")
        ax.scatter(Z[wrong, 0], Z[wrong, 1], s=8, alpha=0.6, color="#e24b4a",
                   marker="x", label="Misclassified")
        ax.set_title(f"Correct vs. misclassified (error={wrong.mean()*100:.1f}%)")
        ax.legend(markerscale=3, fontsize=9)

        fig.suptitle(f"Group-level t-SNE -- {title} -- {len(keys)} subjects", fontsize=13)
        plt.tight_layout()
        path = os.path.join(output_dir, f"group_tsne_{feat_key}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    [saved] {path}")

    # Group-level pooled confusion matrix (complements the existing
    # per-session/per-subject confusion matrix, which was not group-level)
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    im = ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i,j]:.1f}%\n({cm[i,j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 55 else "black", fontsize=9)
    ax.set_title(f"Pooled confusion matrix -- {len(keys)} subjects "
                 f"(accuracy={cm.diagonal().sum()/cm.sum()*100:.1f}%)")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    path = os.path.join(output_dir, "group_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {path}")

    print(f"\n  All group-level analyses saved -> {output_dir}\n")


# ============================================================================
# 6. Grad-CAM (Selvaraju et al., ICCV 2017 / arXiv:1610.02391,
#    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
#    Localization") -- a direct generalization of the original formula to
#    3D tensors (H, W, T) instead of a 2D image (H, W).
#
#    Original formula (Eq. 1-2):
#        alpha_k^c = (1/Z) * sum_i sum_j  dY^c / dA_k^{ij}        (global-
#                     average-pool the gradient of class c w.r.t. the
#                     activation map A in channel k, over all spatial
#                     positions i,j -- "global average pooling" exactly as
#                     stated in the paper)
#        L_GradCAM^c = ReLU( sum_k alpha_k^c * A^k )               (weighted
#                     combination of channels + ReLU, because "we are only
#                     interested in the features that have a POSITIVE
#                     influence")
#    Here, since the activation map is 3D (H,W,T) rather than 2D, the GAP
#    in alpha is taken over all three of H, W, T -- a direct, unambiguous
#    generalization of the formula, exactly as the paper itself notes
#    (Section 3) that the method works for any convolutional layer with a
#    multi-dimensional activation map.
#
#    Why the target layer here is the last Conv3d before GAP in each branch:
#    the paper's own explicit recommendation (Section 3: "we generalize...
#    to the last convolutional layer... [which] represents the best
#    compromise between high-level semantics and detailed spatial
#    information").
#
#    Implementation: no modification to the model code -- only
#      (a) a forward hook to capture the target activation map,
#      (b) a tensor.register_hook on that same map to capture its gradient
#          during backward -- exactly the standard way Grad-CAM is
#          implemented in PyTorch (the same pattern used by the official
#          pytorch-grad-cam code and standard PyTorch tutorials).
# ============================================================================

class GradCAM3D:
    """General-purpose Grad-CAM for any 3D activation tensor (B,C,H,W,T)
    along TSFNet's forward path. For the EEG and fNIRS branches, the target
    module is hooked directly; for the Fusion branch (whose relevant output
    -- efgf2's returned tuple -- is not the direct output of a simple
    module) a "getter" function is used instead, which pulls the target
    tensor out of the intermediate computation -- with no change whatsoever
    to TSFNet.forward's definition.
    """

    def __init__(self, model, target_module=None, target_getter=None):
        """
        target_module : nn.Module or None -- if given, this module's
            forward output is taken as the target activation map (the
            simple case, suitable for model.eeg_branch.conv2 or
            model.fnirs_branch.conv2).
        target_getter : callable(model) -> None -- if target_module is not
            given, this function must be provided: it must register an
            appropriate forward hook itself and populate self.activation
            (for more complex cases like efgf2's output, which is a
            torch.Tensor buried inside a returned tuple).
        """
        assert (target_module is not None) ^ (target_getter is not None), \
            "Exactly one of target_module or target_getter must be given."
        self.model = model
        self.activation = None
        self.gradient = None
        self._fwd_handle = None

        if target_module is not None:
            self._fwd_handle = target_module.register_forward_hook(self._save_activation)
        else:
            target_getter(self)  # the getter itself is responsible for registering the right hook and populating self.activation

    def _save_activation(self, module, inp, out):
        act = out[0] if isinstance(out, tuple) else out
        act.retain_grad()  # ensures .grad is populated even if this is a non-leaf tensor
        self.activation = act

    def compute(self, class_idx, retain_graph=False):
        """Must be called after a complete forward pass and before the next
        optimizer.zero_grad. class_idx: a class index (0/1/2) or a (B,)
        tensor of per-sample target class indices in the batch."""
        if self.activation is None:
            raise RuntimeError("No forward pass has been run yet -- call model(...) first.")

        act = self.activation
        # Eq. 1: alpha_k^c = GAP over (H,W,T) of the gradient of class c w.r.t. A^k
        grad = act.grad  # (B, C, H, W, T), or (B*S, C, H, W, T) for fNIRS
        if grad is None:
            raise RuntimeError(
                "No gradient was recorded on the target activation map -- make "
                "sure score.backward(retain_graph=...) was called after the "
                "forward pass."
            )
        alpha = grad.mean(dim=tuple(range(2, grad.dim())), keepdim=True)  # (B, C, 1, 1, [1])

        # Eq. 2: L^c = ReLU( sum_k alpha_k^c * A^k )
        cam = (alpha * act).sum(dim=1)          # (B, H, W, [T])
        cam = F.relu(cam)
        return cam.detach()

    def remove(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()


def _fusion_activation_getter(gradcam_self):
    """target_getter for the Fusion branch: the target map = efgf2's output
    (fused2), which in the model's forward() is a torch.Tensor (the first
    element of efgf2's returned tuple), not the direct output of a plain
    module -- so it is captured with a hook on the efgf2 module itself."""
    def _hook(module, inp, out):
        act = out[0]  # efgf2 returns: (fused, eeg_pcc, phi_pcc)
        act.retain_grad()
        gradcam_self.activation = act
    gradcam_self._fwd_handle = gradcam_self.model.efgf2.register_forward_hook(_hook)


def run_gradcam_analysis(model, test_loader, device, session_id, output_dir,
                          n_samples=200, window_seconds=10.0):
    """Runs Grad-CAM on all three branches (EEG, fNIRS, Fusion) for all
    three classes, and plots the result as a spatial map (averaged over
    time) and a temporal profile (averaged over space) -- deliberately
    shaped the same as the SHAP output above so the two methods can be
    compared directly in the report.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()  # note: eval() only freezes BN/Dropout; backward still works fine

    store_spatial = {"eeg": {c: [] for c in range(3)},
                      "fnirs": {c: [] for c in range(3)},
                      "fusion": {c: [] for c in range(3)}}
    store_temporal = {"eeg": {c: [] for c in range(3)},
                       "fusion": {c: [] for c in range(3)}}

    collected = 0
    for batch in test_loader:
        if collected >= n_samples:
            break
        eeg_b, fnirs_b, lbl_b = batch[0].to(device), batch[1].to(device), batch[2]
        B = eeg_b.shape[0]

        cam_eeg = GradCAM3D(model, target_module=model.eeg_branch.conv2)
        cam_fnirs = GradCAM3D(model, target_module=model.fnirs_branch.conv2)
        cam_fusion = GradCAM3D(model, target_getter=_fusion_activation_getter)

        eeg_b.requires_grad_(True)
        y_eeg, y_fusion, y_fnirs, y_pred, aux = model(eeg_b, fnirs_b)

        for c in range(3):
            mask = (lbl_b.numpy() == c)
            if not mask.any():
                continue
            model.zero_grad()
            # Per the paper, the pre-softmax score y^c (logit) should be
            # used; however, since this project's architecture only
            # returns the already-softmaxed output (y_pred is already a
            # probability -- forward() never exposes a separate logit
            # layer for the final decision), the target-class softmax
            # score is used instead, following common practice in such
            # cases; this only softens the relative ranking scale a little
            # (due to softmax saturation) and does not change the
            # gradient's direction.
            score = y_pred[mask, c].sum()
            score.backward(retain_graph=True)

            cam_e = cam_eeg.compute(c)[mask]        # (n, H, W, T)
            cam_fn = cam_fnirs.compute(c)           # (B*S, H, W, T) -- reshape to (B,S,...)
            S = fnirs_b.shape[1]
            cam_fn = cam_fn.reshape(B, S, *cam_fn.shape[1:]).mean(dim=1)[mask]  # average over segments
            cam_fu = cam_fusion.compute(c)[mask]

            store_spatial["eeg"][c].append(cam_e.mean(dim=-1).cpu())      # (n,H,W)
            store_spatial["fnirs"][c].append(cam_fn.mean(dim=-1).cpu())
            store_spatial["fusion"][c].append(cam_fu.mean(dim=-1).cpu())
            store_temporal["eeg"][c].append(cam_e.mean(dim=(1, 2)).cpu())  # (n,T)
            store_temporal["fusion"][c].append(cam_fu.mean(dim=(1, 2)).cpu())

        cam_eeg.remove(); cam_fnirs.remove(); cam_fusion.remove()
        eeg_b = eeg_b.detach()
        collected += B

    # --- spatial map (averaged over time), all three branches, all three classes ---
    for branch_key, branch_title in [("eeg", "EEG"), ("fnirs", "fNIRS"), ("fusion", "Fusion")]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, ax in enumerate(axes):
            items = store_spatial[branch_key][c]
            if not items:
                ax.axis("off")
                continue
            v = torch.cat(items, dim=0).mean(dim=0).numpy()
            im = ax.imshow(v, cmap="jet")
            ax.set_title(f"{CLASS_NAMES[c]} (n={sum(x.shape[0] for x in items)})")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(
            f"Grad-CAM -- {branch_title} branch spatial map -- Session {session_id}\n"
            "(approximate grid -- the same interpolation caveat noted for the SHAP/EFGF maps above applies)",
            fontsize=11,
        )
        plt.tight_layout()
        path = os.path.join(output_dir, f"S{session_id}_gradcam_{branch_key}_spatial.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {path}")

    # --- temporal profile (only EEG and Fusion have a meaningful time axis) ---
    for branch_key, branch_title in [("eeg", "EEG"), ("fusion", "Fusion")]:
        fig, ax = plt.subplots(figsize=(12, 5))
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            items = store_temporal[branch_key][c]
            if not items:
                continue
            v = torch.cat(items, dim=0).mean(dim=0).numpy()
            t = np.linspace(0, window_seconds, len(v))
            ax.plot(t, v, color=col, label=cname, lw=1.3)
        ax.set_xlabel("Time within window (seconds)")
        ax.set_ylabel("Grad-CAM (averaged over space)")
        ax.set_title(f"Grad-CAM -- {branch_title} branch temporal profile -- Session {session_id}")
        ax.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, f"S{session_id}_gradcam_{branch_key}_temporal.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [saved] {path}")

    print(f"\n  Grad-CAM analysis complete -> {output_dir}\n")
