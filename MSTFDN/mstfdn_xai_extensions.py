"""
mstfdn_xai_extensions.py -- post-hoc explainability for MSTFDN, mirroring
xai_extensions.py (the TSFNet version) so both architectures are explained
with the SAME methods and the results can be compared directly.

WHEN TO RUN THIS: after training finishes, on a saved checkpoint. It is not
part of the architecture and adds no trainable parameters -- it only reads a
frozen model. Runs fine on CPU.

WHY THE CROSS-ARCHITECTURE COMPARISON MATTERS (the reason this file exists):
if two architectures as different as TSFNet and MSTFDN converge on the SAME
channels / frequency bands / time windows, that pattern is not an artifact of
either architecture -- it is evidence about the underlying neurophysiology.
That is a much stronger claim than either model's attention map alone, and
it is exactly what a biomarker claim needs.

WHAT DIFFERS FROM THE TSFNet VERSION (and why)
----------------------------------------------
  * MSTFDN takes (eeg, fnirs) and returns (logits, he, hf, hef). Its head
    emits LOGITS; every probability below therefore comes from an explicit
    softmax (the project's documented probability/logit trap, rule 9).
  * MSTFDN's EEG input is (B, 5, CH, T) -- five FIXED frequency bands in
    CHANNEL space, not a 16x16 interpolated grid. This makes one analysis
    possible here that TSFNet cannot do at all: per-BAND attribution, i.e.
    which frequency band the model actually relies on. That is reported as
    a distinct panel rather than being forced into the TSFNet plot format.
  * MSTFDN has no EFGF gates and no EGTA temporal attention, so those
    panels are dropped rather than faked.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
CHROMO_NAMES = ["HbO", "HbR", "HbT"]
CLASS_NAMES = ["0-back", "2-back", "3-back"]


# ---------------------------------------------------------------------------
# 1. Occlusion-based attribution -- model-agnostic, works on any architecture
# ---------------------------------------------------------------------------
@torch.no_grad()
def occlusion_attribution(model, loader, device, axis="eeg_band", max_batches=None):
    """
    Measures how much the model's correct-class probability DROPS when one
    input component is zeroed out. Higher drop = the model relied on it more.

    Occlusion is used here instead of gradient-based saliency on purpose: it
    is model-agnostic, so the identical procedure can be run on TSFNet and on
    MSTFDN and the two results compared without worrying that a difference
    came from the attribution method rather than the model.

    axis:
      'eeg_band'    -> one of the 5 EEG frequency bands (MSTFDN-only; TSFNet's
                       grid input has no band dimension)
      'eeg_channel' -> one EEG channel
      'fnirs_chromo'-> one of HbO / HbR / HbT
      'fnirs_channel'-> one fNIRS channel
    Returns (labels, mean_prob_drop) -- drop in the TRUE class's probability,
    averaged over windows.
    """
    model.eval()
    baseline_probs, occluded_probs, all_y = [], None, []

    # First pass: baseline true-class probability per window.
    batches = []
    for i, (eeg, fnirs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        eeg, fnirs = eeg.to(device), fnirs.to(device)
        logits, *_ = model(eeg, fnirs)
        p = F.softmax(logits, dim=-1)
        baseline_probs.append(p[torch.arange(len(labels)), labels].cpu().numpy())
        all_y.append(labels.numpy())
        batches.append((eeg.cpu(), fnirs.cpu(), labels))
    baseline_probs = np.concatenate(baseline_probs)

    # Decide what to iterate over.
    eeg0, fnirs0, _ = batches[0]
    if axis == "eeg_band":
        n_units, names = eeg0.shape[1], BAND_NAMES[:eeg0.shape[1]]
    elif axis == "eeg_channel":
        n_units, names = eeg0.shape[2], [f"E{c}" for c in range(eeg0.shape[2])]
    elif axis == "fnirs_chromo":
        n_units, names = fnirs0.shape[1], CHROMO_NAMES[:fnirs0.shape[1]]
    elif axis == "fnirs_channel":
        n_units, names = fnirs0.shape[2], [f"N{c}" for c in range(fnirs0.shape[2])]
    else:
        raise ValueError(f"unknown axis {axis!r}")

    drops = np.zeros(n_units)
    for u in range(n_units):
        probs_u = []
        for eeg_c, fnirs_c, labels in batches:
            eeg = eeg_c.clone().to(device)
            fnirs = fnirs_c.clone().to(device)
            if axis == "eeg_band":
                eeg[:, u] = 0
            elif axis == "eeg_channel":
                eeg[:, :, u] = 0
            elif axis == "fnirs_chromo":
                fnirs[:, u] = 0
            else:
                fnirs[:, :, u] = 0
            logits, *_ = model(eeg, fnirs)
            p = F.softmax(logits, dim=-1)
            probs_u.append(p[torch.arange(len(labels)), labels].cpu().numpy())
        drops[u] = float(baseline_probs.mean() - np.concatenate(probs_u).mean())

    return names, drops


def plot_band_attribution(model, loader, device, out_dir, subject_id=None,
                          fold_idx=None, max_batches=20):
    """Which EEG frequency band does MSTFDN actually use?

    This is the panel TSFNet structurally cannot produce (its EEG input is a
    spatially-interpolated grid with no band axis), and it speaks directly to
    the cognitive-load literature: frontal theta increase and parietal alpha
    decrease are the canonical workload markers. If the bands the model
    relies on match that prior, it is evidence the model learned physiology
    rather than an artifact.
    """
    os.makedirs(out_dir, exist_ok=True)
    names, drops = occlusion_attribution(model, loader, device,
                                         axis="eeg_band", max_batches=max_batches)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    colors = ["#4C72B0" if d >= 0 else "#C44E52" for d in drops]
    ax.bar(names, drops, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("drop in true-class probability when occluded")
    tag = ""
    if subject_id is not None:
        tag = f"{subject_id}_fold{fold_idx}_"
        ax.set_title(f"{subject_id} fold {fold_idx} -- MSTFDN EEG band attribution")
    else:
        ax.set_title("MSTFDN EEG band attribution (occlusion)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}mstfdn_band_attribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [saved] {path}  ({', '.join(f'{n}={d:+.4f}' for n, d in zip(names, drops))})")
    return path, dict(zip(names, drops.tolist()))


def plot_chromophore_attribution(model, loader, device, out_dir, subject_id=None,
                                 fold_idx=None, max_batches=20):
    """Which fNIRS chromophore (HbO / HbR / HbT) does the model rely on?
    Directly comparable to this project's own G-2a ablation on the TSFNet
    side, which tested adding HbT as a third chromophore."""
    os.makedirs(out_dir, exist_ok=True)
    names, drops = occlusion_attribution(model, loader, device,
                                         axis="fnirs_chromo", max_batches=max_batches)
    fig, ax = plt.subplots(figsize=(6, 4.3))
    ax.bar(names, drops, color=["#DD8452", "#55A868", "#8172B2"][:len(names)])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("drop in true-class probability when occluded")
    tag = ""
    if subject_id is not None:
        tag = f"{subject_id}_fold{fold_idx}_"
        ax.set_title(f"{subject_id} fold {fold_idx} -- MSTFDN chromophore attribution")
    else:
        ax.set_title("MSTFDN fNIRS chromophore attribution (occlusion)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}mstfdn_chromophore_attribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [saved] {path}  ({', '.join(f'{n}={d:+.4f}' for n, d in zip(names, drops))})")
    return path, dict(zip(names, drops.tolist()))


def plot_channel_attribution(model, loader, device, out_dir, modality="eeg",
                             subject_id=None, fold_idx=None, max_batches=20, top_k=10):
    """Per-channel attribution, the panel most directly comparable with the
    TSFNet-side SHAP spatial analysis. If the two architectures highlight the
    same channels, that is cross-architecture evidence for a spatial
    biomarker rather than an architectural artifact."""
    os.makedirs(out_dir, exist_ok=True)
    axis = "eeg_channel" if modality == "eeg" else "fnirs_channel"
    names, drops = occlusion_attribution(model, loader, device,
                                         axis=axis, max_batches=max_batches)
    order = np.argsort(drops)[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.bar([names[i] for i in order], drops[order], color="#4C72B0")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("drop in true-class probability")
    tag = ""
    ttl = f"MSTFDN {modality.upper()} per-channel attribution (occlusion)"
    if subject_id is not None:
        tag = f"{subject_id}_fold{fold_idx}_"
        ttl = f"{subject_id} fold {fold_idx} -- " + ttl
    ax.set_title(ttl)
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}mstfdn_{modality}_channel_attribution.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    top = [(names[i], float(drops[i])) for i in order[:top_k]]
    print(f"  [saved] {path}  top-{top_k}: {top}")
    return path, {"all": dict(zip(names, drops.tolist())), "top_k": top}


# ---------------------------------------------------------------------------
# 2. Per-class attribution -- does the model use different evidence per class?
# ---------------------------------------------------------------------------
@torch.no_grad()
def plot_band_attribution_per_class(model, loader, device, out_dir,
                                    subject_id=None, fold_idx=None, max_batches=20):
    """Band attribution computed separately for each true class.

    This is the panel that can actually support a cognitive claim: if the
    bands the model relies on SHIFT between 0-back and 3-back, that is
    load-dependent spectral reliance -- which is what the workload
    literature predicts (frontal theta rising with load). A flat profile
    across classes would mean the model is using a load-independent cue,
    which is an equally important (negative) finding.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    batches = []
    for i, (eeg, fnirs, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batches.append((eeg, fnirs, labels))
    n_bands = batches[0][0].shape[1]
    n_cls = len(CLASS_NAMES)
    drops = np.full((n_cls, n_bands), np.nan)

    base_by_cls = {c: [] for c in range(n_cls)}
    for eeg_c, fnirs_c, labels in batches:
        eeg, fnirs = eeg_c.to(device), fnirs_c.to(device)
        p = F.softmax(model(eeg, fnirs)[0], dim=-1)
        tp = p[torch.arange(len(labels)), labels].cpu().numpy()
        for c in range(n_cls):
            m = labels.numpy() == c
            if m.any():
                base_by_cls[c].append(tp[m])
    base_mean = {c: (np.concatenate(v).mean() if v else np.nan)
                 for c, v in base_by_cls.items()}

    for b in range(n_bands):
        occ_by_cls = {c: [] for c in range(n_cls)}
        for eeg_c, fnirs_c, labels in batches:
            eeg = eeg_c.clone().to(device); eeg[:, b] = 0
            fnirs = fnirs_c.to(device)
            p = F.softmax(model(eeg, fnirs)[0], dim=-1)
            tp = p[torch.arange(len(labels)), labels].cpu().numpy()
            for c in range(n_cls):
                m = labels.numpy() == c
                if m.any():
                    occ_by_cls[c].append(tp[m])
        for c in range(n_cls):
            if occ_by_cls[c] and not np.isnan(base_mean[c]):
                drops[c, b] = base_mean[c] - np.concatenate(occ_by_cls[c]).mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(n_bands); w = 0.8 / n_cls
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for c in range(n_cls):
        ax.bar(x + c * w - 0.4 + w / 2, drops[c], width=w,
               label=CLASS_NAMES[c], color=colors[c])
    ax.set_xticks(x); ax.set_xticklabels(BAND_NAMES[:n_bands])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("drop in true-class probability")
    ttl = "MSTFDN band attribution per cognitive-load level"
    tag = ""
    if subject_id is not None:
        tag = f"{subject_id}_fold{fold_idx}_"
        ttl = f"{subject_id} fold {fold_idx} -- " + ttl
    ax.set_title(ttl); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, f"{tag}mstfdn_band_attribution_per_class.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  [saved] {path}")
    return path, {CLASS_NAMES[c]: dict(zip(BAND_NAMES[:n_bands], drops[c].tolist()))
                  for c in range(n_cls)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_mstfdn_xai(model, loader, device, out_dir, subject_id=None, fold_idx=None,
                   max_batches=20):
    """Runs the full MSTFDN XAI battery on a TRAINED checkpoint.
    Every step is individually guarded so one failing panel cannot lose the rest."""
    os.makedirs(out_dir, exist_ok=True)
    results = {"architecture": "MSTFDN", "subject_id": subject_id,
               "fold_idx": fold_idx, "figures": []}
    steps = [
        ("band", plot_band_attribution),
        ("chromophore", plot_chromophore_attribution),
        ("band_per_class", plot_band_attribution_per_class),
    ]
    for name, fn in steps:
        try:
            path, vals = fn(model, loader, device, out_dir,
                            subject_id=subject_id, fold_idx=fold_idx,
                            max_batches=max_batches)
            results["figures"].append(path)
            results[name] = vals
        except Exception as exc:
            import traceback
            print(f"  [xai] WARNING: {name} failed: {exc}")
            traceback.print_exc()
            results[f"{name}_error"] = str(exc)
    for modality in ("eeg", "fnirs"):
        try:
            path, vals = plot_channel_attribution(
                model, loader, device, out_dir, modality=modality,
                subject_id=subject_id, fold_idx=fold_idx, max_batches=max_batches)
            results["figures"].append(path)
            results[f"{modality}_channels"] = vals
        except Exception as exc:
            import traceback
            print(f"  [xai] WARNING: {modality} channel attribution failed: {exc}")
            traceback.print_exc()
            results[f"{modality}_channels_error"] = str(exc)
    return results
