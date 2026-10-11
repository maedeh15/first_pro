"""
mgfnet_xai.py -- explainability for MGFNet (the MGFNet counterpart of the
TSFNet project's xai_extensions.py). Works post hoc on the checkpoints that
run_all_subjects_MGFNET.py saves ({subject}_session{s}_mgf_final.pt, which
contain the weights, the train-fold normaliser and the model kwargs).

┌──────────────────────────────────────────────────────────────────────────┐
│ TSFNet xai_extensions.py            -> here (MGFNet)                       │
├──────────────────────────────────────────────────────────────────────────┤
│ SHAP on 16x16-grid EEG / fNIRS      -> SHAP (GradientExplainer) on the    │
│   segments                             raw EEG / HbO / HbR inputs. MGFNet │
│                                        uses REAL channels (no interpolated│
│                                        grid), so the EEG map is a true    │
│                                        electrode topomap.                 │
│ fixed w_eeg/w_fusion/w_fnirs bars   -> gone: MGFNet's gate is per-sample. │
│ "SHAP on gating has no meaning"     -> now it DOES (Adaptive-Fusion       │
│                                        Fig. 11-12 analogue): SHAP of the  │
│                                        6 CGSCR gate weights w.r.t. each   │
│                                        modality (run_gate_shap_analysis). │
│ EFGF gamma/alpha, Phi map, CAFE,    -> no counterpart in MGFNet. The      │
│   EGTA attention                       CGSCR internals (gate, coupling C, │
│                                        token selection, Morlet bands) are │
│                                        in mgfnet_diagnostics.py.          │
│ segment hard-pruning                -> modality OCCLUSION (inference-only)│
│                                        + EEG AWGN robustness sweep = the  │
│                                        paper's Sec. 4.3 clean-train /     │
│                                        noisy-test protocol, incl. a       │
│                                        multi-variant comparison (full vs  │
│                                        w/o CGSCR, paper Fig. 6).          │
│ Grad-CAM 3D (H,W,T)                 -> Grad-CAM 1D over time on the last  │
│                                        CBR layer of each of the 6 branches│
│ single-subject / group t-SNE        -> same, from MGFNet checkpoints, plus│
│                                        group-level CGSCR summaries.       │
└──────────────────────────────────────────────────────────────────────────┘

All attributions target the hybrid head's LOGIT (MGFNet exposes logits, so
Grad-CAM uses the pre-softmax score exactly as Selvaraju et al. prescribe).

Install for SHAP:  pip install shap

CLI
    # per-fold analyses for one subject/session from its checkpoint
    python mgfnet_xai.py --results-dir <results_mgf_...> --data-root <MGFNET_..._28ch> \
        --subject VP001 --session 1
    # group level (t-SNE + pooled confusion + CGSCR summaries + noise curve)
    python mgfnet_xai.py --results-dir <...> --data-root <...> --group
    # paper Fig. 6: full vs w/o CGSCR under EEG noise
    python mgfnet_xai.py --data-root <...> --compare-noise full=<dir1> noCGSCR=<dir2>
"""
import argparse
import gc
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mgfnet_model import MGFNet
from mgfnet_dataset import MGFNBackDataset, ChannelNormalizer, _unpack_mgf_batch
from mgfnet_train import cohen_kappa
from mgfnet_diagnostics import (extract_outputs, CLASS_NAMES, CLASS_COLORS, BRANCH_KEYS,
                                BRANCH_NAMES, EEG_BAND_NAMES, NIRS_SCALE_NAMES, token_labels)

MODALITIES = ("eeg", "hbo", "hbr")
MOD_NAMES = {"eeg": "EEG", "hbo": "HbO", "hbr": "HbR"}
EEG_FS = 200.0      # window length in seconds = n_samples / EEG_FS (5 s or 10 s)


# ============================================================================
# Loading helpers
# ============================================================================
def checkpoint_path(results_dir, subject_id, session):
    return os.path.join(results_dir, f"{subject_id}_session{session}_mgf_final.pt")


def load_mgfnet_checkpoint(path, device="cpu"):
    """-> (model in eval mode, ChannelNormalizer, checkpoint dict)"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MGFNet(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ChannelNormalizer.from_state_dict(ckpt["normalizer"]), ckpt


def build_split_loaders(data_root, subject_id, test_session, batch_size=64):
    """Test loader (held-out session) and train loader (other sessions) --
    the exact LOSO split the checkpoint was trained/tested on."""
    full = MGFNBackDataset(os.path.join(data_root, subject_id), subject_id)
    test_idx = np.where(full.session_ids == test_session)[0]
    train_idx = np.where(full.session_ids != test_session)[0]
    mk = lambda idx: DataLoader(torch.utils.data.Subset(full, idx), batch_size=batch_size,
                                shuffle=False)
    return mk(train_idx), mk(test_idx)


def read_eeg_channel_names(data_root, subject_id):
    p = os.path.join(data_root, subject_id, f"{subject_id}_mgf_metadata.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)["eeg_channels"]
    except (OSError, KeyError, ValueError):
        return None


def _collect(loader, n_max):
    xs = {m: [] for m in MODALITIES}
    ys, n = [], 0
    for batch in loader:
        x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
        for m, x in zip(MODALITIES, (x_eeg, x_hbo, x_hbr)):
            xs[m].append(x)
        ys.append(y)
        n += len(y)
        if n >= n_max:
            break
    return {m: torch.cat(v)[:n_max] for m, v in xs.items()}, torch.cat(ys)[:n_max]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    [saved] {path}")
    return path


# ============================================================================
# Topomap (real electrodes)
# ============================================================================
def _plot_eeg_channel_map(ax, values, ch_names, title):
    """Scalp topomap via MNE when available, else a bar chart. EOG channels
    (30-channel variant) have no scalp position and are left out of the map."""
    try:
        import mne
        keep = [i for i, n in enumerate(ch_names) if n.upper() not in ("HEOG", "VEOG")]
        info = mne.create_info([ch_names[i] for i in keep], 200.0, "eeg")
        info.set_montage("standard_1005", match_case=False, on_missing="ignore")
        im, _ = mne.viz.plot_topomap(np.asarray(values)[keep], info, axes=ax, show=False,
                                     cmap="inferno", contours=0)
        ax.set_title(title, fontsize=9)
        return im
    except Exception:
        ax.bar(range(len(values)), values, color="#C44E52")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(ch_names or range(len(values)), rotation=90, fontsize=6)
        ax.set_title(title, fontsize=9)
        return None


# ============================================================================
# 1. SHAP on the inputs (one modality at a time)
# ============================================================================
class _ModalityWrapper(torch.nn.Module):
    """Single-input -> single-output view of MGFNet for shap: the explained
    modality varies, the other two are held at their background mean.
    output='logits' -> hybrid-head logits (B,3); output='gate' -> CGSCR gate (B,6).
    Inputs are already NORMALISED."""

    def __init__(self, model, modality, fixed, output="logits"):
        super().__init__()
        self.model, self.modality, self.fixed, self.output = model, modality, fixed, output

    def forward(self, x):
        B = x.shape[0]
        args = {m: (x if m == self.modality else self.fixed[m].expand(B, *self.fixed[m].shape[1:]))
                for m in MODALITIES}
        y_hyb, _, aux = self.model(args["eeg"], args["hbo"], args["hbr"])
        return y_hyb if self.output == "logits" else aux["gate_weights"]


def _normalize_shap_output(sv, n_out):
    """-> (n_out, N, *input_dims) whatever the shap version returned."""
    if isinstance(sv, list):
        return np.stack([np.asarray(s) for s in sv], axis=0)
    arr = np.asarray(sv)
    if arr.shape[-1] == n_out:
        return np.moveaxis(arr, -1, 0)
    return arr


def _shap_per_modality(model, norm, test_loader, device, output, n_background,
                       n_explain, nsamples, seed):
    try:
        import shap
    except ImportError as e:
        raise ImportError("shap is not installed: pip install shap") from e
    model.eval()
    # draw from the whole held-out session (class blocks are contiguous in
    # time, so the first N windows could all be one class)
    xs, ys = _collect(test_loader, 10 ** 9)
    n = len(ys)
    n_bg = min(n_background, n // 2)
    n_ex = min(n_explain, n - n_bg)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    idx_bg, rest = perm[:n_bg], perm[n_bg:]
    # explained samples: stratified by class, so every class gets a map
    y_rest = ys.numpy()[rest]
    per_cls = [rest[y_rest == c] for c in range(3)]
    idx_ex = []
    while len(idx_ex) < n_ex and any(len(p) for p in per_cls):
        for c in range(3):
            if len(per_cls[c]) and len(idx_ex) < n_ex:
                idx_ex.append(per_cls[c][0]); per_cls[c] = per_cls[c][1:]
    idx_ex = np.array(idx_ex)
    xe, xo, xr = norm(xs["eeg"].to(device), xs["hbo"].to(device), xs["hbr"].to(device))
    xn = {"eeg": xe, "hbo": xo, "hbr": xr}
    fixed = {m: xn[m][idx_bg].mean(0, keepdim=True) for m in MODALITIES}
    n_out = 3 if output == "logits" else 6
    out = {}
    for m in MODALITIES:
        wrapped = _ModalityWrapper(model, m, fixed, output).to(device)
        expl = shap.GradientExplainer(wrapped, xn[m][idx_bg], batch_size=16)
        sv = expl.shap_values(xn[m][idx_ex], nsamples=nsamples)
        out[m] = _normalize_shap_output(sv, n_out)          # (n_out, N, C, T)
        print(f"    [SHAP-{output}] {MOD_NAMES[m]} done {out[m].shape}")
    return out, ys[idx_ex].numpy()


def run_shap_analysis(model, norm, test_loader, device, subject_id, session_id, output_dir,
                      eeg_channel_names=None, n_background=32, n_explain=48, nsamples=50,
                      seed=0):
    """GradientExplainer SHAP of the hybrid logit w.r.t. each modality.
    Saves: EEG electrode topomap per class, HbO/HbR channel importance per
    class, temporal importance profiles, and the raw values (.npz)."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [SHAP] {subject_id} session {session_id} ...")
    sv, labels = _shap_per_modality(model, norm, test_loader, device, "logits",
                                    n_background, n_explain, nsamples, seed)
    tag = f"{subject_id}_S{session_id}"
    paths = []

    # EEG spatial: mean |SHAP| of the true class over samples and time
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    names = eeg_channel_names or [f"ch{i}" for i in range(sv["eeg"].shape[2])]
    for c, ax in enumerate(axes):
        m = labels == c
        if not m.any():
            ax.axis("off"); ax.set_title(f"{CLASS_NAMES[c]} (no samples)", fontsize=9); continue
        v = np.abs(sv["eeg"][c][m]).mean(axis=(0, 2))
        _plot_eeg_channel_map(ax, v, names, f"{CLASS_NAMES[c]} (n={m.sum()})")
    fig.suptitle(f"SHAP |importance| -- EEG electrodes -- {tag}", fontsize=11)
    paths.append(_save(fig, os.path.join(output_dir, f"{tag}_shap_eeg_topomap.png")))

    # fNIRS channel importance
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=True)
    for ax, mod in zip(axes, ("hbo", "hbr")):
        w = 0.27
        for c in range(3):
            m = labels == c
            if m.any():
                v = np.abs(sv[mod][c][m]).mean(axis=(0, 2))
                ax.bar(np.arange(len(v)) + (c - 1) * w, v, w, color=CLASS_COLORS[c],
                       label=CLASS_NAMES[c])
        ax.set_ylabel(f"{MOD_NAMES[mod]} mean |SHAP|"); ax.legend(fontsize=8)
    axes[1].set_xlabel("fNIRS channel index")
    fig.suptitle(f"SHAP -- fNIRS channel importance -- {tag}", fontsize=11)
    paths.append(_save(fig, os.path.join(output_dir, f"{tag}_shap_fnirs_channels.png")))

    # temporal profiles within the 5 s window
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, mod in zip(axes, MODALITIES):
        for c in range(3):
            m = labels == c
            if m.any():
                v = np.abs(sv[mod][c][m]).mean(axis=(0, 1))
                win_s = sv["eeg"].shape[-1] / EEG_FS
                ax.plot(np.linspace(0, win_s, len(v)), v, color=CLASS_COLORS[c],
                        label=CLASS_NAMES[c], lw=1.2)
        ax.set_xlabel("time in window (s)"); ax.set_title(MOD_NAMES[mod]); ax.legend(fontsize=8)
    axes[0].set_ylabel("mean |SHAP|")
    fig.suptitle(f"SHAP -- temporal importance -- {tag}", fontsize=11)
    paths.append(_save(fig, os.path.join(output_dir, f"{tag}_shap_temporal.png")))

    # modality share (how much each modality drives the decision)
    share = {MOD_NAMES[m]: float(np.abs(sv[m][labels, np.arange(len(labels))]).sum(axis=(1, 2)).mean())
             for m in MODALITIES}
    np.savez_compressed(os.path.join(output_dir, f"{tag}_shap_raw.npz"),
                        **{f"shap_{m}": sv[m].astype(np.float32) for m in MODALITIES},
                        labels=labels)
    print(f"  [SHAP] total |SHAP| per modality (true class): {share}")
    return {"figures": paths, "modality_total_abs_shap": share}


# ============================================================================
# 2. SHAP of the CGSCR gate (instance-adaptive gating -> now meaningful)
# ============================================================================
def run_gate_shap_analysis(model, norm, test_loader, device, subject_id, session_id,
                           output_dir, n_background=32, n_explain=48, nsamples=50, seed=0):
    """Which input modality drives each of the 6 branch-gate weights?
    Heatmap of mean |SHAP| (rows: EEG/HbO/HbR, cols: gate slots), plus the
    per-class version for the EEG row. Analogue of Adaptive Fusion Fig. 11-12."""
    if not getattr(model, "use_cgscr", False):
        print("  [gate-SHAP] skipped: model has no CGSCR (w/o CGSCR variant).")
        return None
    os.makedirs(output_dir, exist_ok=True)
    print(f"  [gate-SHAP] {subject_id} session {session_id} ...")
    sv, labels = _shap_per_modality(model, norm, test_loader, device, "gate",
                                    n_background, n_explain, nsamples, seed)
    M = np.array([[np.abs(sv[m][g]).sum(axis=(1, 2)).mean() for g in range(6)]
                  for m in MODALITIES])
    tag = f"{subject_id}_S{session_id}"
    cols = [BRANCH_NAMES[k] for k in BRANCH_KEYS]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(M, cmap="viridis")
    for i in range(3):
        for j in range(6):
            ax.text(j, i, f"{M[i, j]:.2e}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xticks(range(6)); ax.set_xticklabels([f"gate[{c}]" for c in cols], rotation=30)
    ax.set_yticks(range(3)); ax.set_yticklabels([MOD_NAMES[m] for m in MODALITIES])
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"SHAP of CGSCR branch-gate weights w.r.t. each modality -- {tag}", fontsize=10)
    path = _save(fig, os.path.join(output_dir, f"{tag}_shap_gate.png"))
    return {"figures": [path],
            "gate_shap": {MOD_NAMES[m]: dict(zip(cols, M[i].tolist()))
                          for i, m in enumerate(MODALITIES)}}


# ============================================================================
# 3. Grad-CAM 1D on every branch
# ============================================================================
def _branch_target_modules(model):
    mods = {}
    if model.use_intra_modal:
        mods.update({"1": model.enc_eeg.blocks[-1], "2": model.enc_hbo.blocks[-1],
                     "3": model.enc_hbr.blocks[-1]})
    if model.use_cross_modal:
        mods.update({"12": model.enc_12.blocks[-1], "13": model.enc_13.blocks[-1],
                     "23": model.enc_23.blocks[-1]})
    return mods


def run_gradcam_analysis(model, norm, test_loader, device, subject_id, session_id,
                         output_dir, n_samples=300, n_points=100):
    """Grad-CAM (Selvaraju et al. 2017) on the last CBR activation (B, C, T')
    before GAP in each branch: alpha_k = mean_t dy_c/dA_k, CAM = ReLU(sum_k
    alpha_k A_k). Target = hybrid-head LOGIT of each sample's TRUE class.
    Each CAM is resampled to n_points over the window and normalised per
    sample (max = 1) before averaging, so branches are comparable."""
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    targets = _branch_target_modules(model)
    acts = {}
    hooks = []
    for k, mod in targets.items():
        def _h(m, inp, out, key=k):
            out.retain_grad()
            acts[key] = out
        hooks.append(mod.register_forward_hook(_h))

    store = {k: {c: [] for c in range(3)} for k in targets}
    seen = 0
    try:
        for batch in test_loader:
            if seen >= n_samples:
                break
            x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
            window_s = x_eeg.shape[-1] / EEG_FS
            x_eeg, x_hbo, x_hbr = norm(x_eeg.to(device), x_hbo.to(device), x_hbr.to(device))
            y = y.to(device)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                y_hyb, _, _ = model(x_eeg, x_hbo, x_hbr)
                y_hyb.gather(1, y[:, None]).sum().backward()
            for k, A in acts.items():
                alpha = A.grad.mean(dim=2, keepdim=True)                 # (B, C, 1)
                cam = F.relu((alpha * A).sum(dim=1, keepdim=True))       # (B, 1, T')
                cam = F.interpolate(cam, size=n_points, mode="linear", align_corners=False)[:, 0]
                cam = cam / cam.amax(dim=1, keepdim=True).clamp_min(1e-12)
                cam = cam.detach().cpu().numpy()
                for c in range(3):
                    m = (y == c).cpu().numpy()
                    if m.any():
                        store[k][c].append(cam[m])
            seen += len(y)
    finally:
        for h in hooks:
            h.remove()
        model.zero_grad(set_to_none=True)

    tag = f"{subject_id}_S{session_id}"
    keys = list(targets)
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True)
    t = np.linspace(0, window_s, n_points)
    profiles = {}
    for ax, k in zip(axes.ravel(), BRANCH_KEYS):
        if k not in store:
            ax.axis("off"); continue
        profiles[BRANCH_NAMES[k]] = {}
        for c in range(3):
            if store[k][c]:
                v = np.concatenate(store[k][c]).mean(0)
                profiles[BRANCH_NAMES[k]][CLASS_NAMES[c]] = v.round(4).tolist()
                ax.plot(t, v, color=CLASS_COLORS[c], label=CLASS_NAMES[c], lw=1.3)
        ax.set_title(f"{BRANCH_NAMES[k]} branch"); ax.legend(fontsize=7)
    for ax in axes[1]:
        ax.set_xlabel("time in window (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Grad-CAM (normalised)")
    fig.suptitle(f"Grad-CAM over time, per branch (target: true-class logit) -- {tag}",
                 fontsize=11)
    path = _save(fig, os.path.join(output_dir, f"{tag}_gradcam_branches.png"))
    return {"figures": [path], "gradcam_profiles": profiles, "branches": keys}


# ============================================================================
# 4. Modality occlusion + EEG noise robustness (inference-only)
# ============================================================================
@torch.no_grad()
def _evaluate(model, norm, loader, device, transform=None, seed=0):
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    preds, labels = [], []
    for batch in loader:
        x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
        if transform is not None:
            x_eeg, x_hbo, x_hbr = transform(x_eeg, x_hbo, x_hbr, gen, norm)
        x_eeg, x_hbo, x_hbr = norm(x_eeg.to(device), x_hbo.to(device), x_hbr.to(device))
        y_hyb, _, _ = model(x_eeg, x_hbo, x_hbr)
        preds.append(y_hyb.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    p, l = np.concatenate(preds), np.concatenate(labels)
    return {"acc": float((p == l).mean()), "kappa": float(cohen_kappa(l, p))}


def _occlude(*mods):
    """Replace modality inputs by their TRAINING mean (= 0 after the
    normaliser), i.e. 'no information', without changing the input scale."""
    def tf(x_eeg, x_hbo, x_hbr, gen, norm):
        xs = {"eeg": x_eeg, "hbo": x_hbo, "hbr": x_hbr}
        for m in mods:
            mean = norm.stats[m][0]
            xs[m] = mean.unsqueeze(0).expand_as(xs[m]).clone()
        return xs["eeg"], xs["hbo"], xs["hbr"]
    return tf


def evaluate_modality_occlusion(model, norm, test_loader, device):
    """Inference-only: accuracy with each modality occluded. NOT the paper's
    Fig. 5 modality ablation (which RETRAINS EEG-only / fNIRS-only models);
    it measures how much the trained model RELIES on each modality."""
    variants = {"full": None, "no EEG": _occlude("eeg"), "no HbO": _occlude("hbo"),
                "no HbR": _occlude("hbr"), "no fNIRS": _occlude("hbo", "hbr")}
    res = {k: _evaluate(model, norm, test_loader, device, tf) for k, tf in variants.items()}
    for k, v in res.items():
        print(f"  [occlusion] {k:<9} acc={v['acc']:.4f} kappa={v['kappa']:.4f}")
    return res


def _eeg_awgn(snr_db):
    """Paper Eqs. 8-9 on the RAW (pre-normalisation) EEG of every sample:
    x + N(0, sigma^2), sigma^2 = P_signal / 10^(SNR/10), P_signal = variance
    of that sample's EEG. fNIRS untouched."""
    def tf(x_eeg, x_hbo, x_hbr, gen, norm):
        p = x_eeg.var(dim=(1, 2), keepdim=True, unbiased=False)
        sigma = torch.sqrt(p / (10.0 ** (snr_db / 10.0)))
        return x_eeg + torch.randn(x_eeg.shape, generator=gen) * sigma, x_hbo, x_hbr
    return tf


def evaluate_eeg_noise_robustness(model, norm, test_loader, device,
                                  snr_list=(10, 5, 0, -5, -10), seed=0):
    """Clean-training / noisy-testing (paper Sec. 4.3): accuracy on the held-
    out session with AWGN injected into EEG only, at each SNR."""
    res = {"clean": _evaluate(model, norm, test_loader, device)}
    for snr in snr_list:
        res[f"{snr:+g} dB"] = _evaluate(model, norm, test_loader, device, _eeg_awgn(snr), seed)
    print("  [noise] " + "  ".join(f"{k}: {v['acc']:.4f}" for k, v in res.items()))
    return res


def plot_noise_curves(curves, output_dir, title, fname="eeg_noise_robustness.png"):
    """curves: {label: {"clean": {"acc":..}, "+10 dB": {...}, ...}} or the
    same with lists of per-subject dicts (mean +- SEM is then plotted)."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for label, res in curves.items():
        keys = list(res)
        if isinstance(res[keys[0]], list):
            vals = np.array([[r["acc"] for r in res[k]] for k in keys]) * 100
            mean = vals.mean(1)
            sem = vals.std(1) / np.sqrt(vals.shape[1])
            ax.errorbar(range(len(keys)), mean, yerr=sem, marker="o", capsize=3, label=label)
        else:
            ax.plot(range(len(keys)), [res[k]["acc"] * 100 for k in keys], "-o", label=label)
    ax.axhline(100 / 3, color="gray", ls="--", label="chance")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    ax.set_xlabel("EEG condition (AWGN, fNIRS clean)"); ax.set_ylabel("accuracy (%)")
    ax.set_title(title, fontsize=10); ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, os.path.join(output_dir, fname))


# ============================================================================
# 5. Single-subject and group-level t-SNE
# ============================================================================
def run_single_subject_tsne(model, norm, test_loader, device, subject_id, session_id,
                            output_dir, perplexity=30):
    """Before/after view (MSTFDN Fig. 16 analogue): PCA of the raw normalised
    input vs t-SNE of the trained fused representation, + confusion matrix."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    os.makedirs(output_dir, exist_ok=True)
    d = extract_outputs(model, test_loader, device, norm)
    labels, preds = d["y_true"], d["logits_hyb"].argmax(1)

    raw = []
    for batch in test_loader:
        x_eeg, x_hbo, x_hbr, _ = _unpack_mgf_batch(batch)
        xe, xo, xr = norm(x_eeg, x_hbo, x_hbr)
        raw.append(torch.cat([xe.flatten(1), xo.flatten(1), xr.flatten(1)], 1).numpy())
    Z0 = PCA(n_components=2, random_state=0).fit_transform(np.concatenate(raw))
    X = StandardScaler().fit_transform(d["fused"])
    Z1 = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(X) // 4)),
              init="pca", random_state=0).fit_transform(X)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for ax, Z, title in ((axes[0], Z0, "Initial state (PCA of raw EEG+HbO+HbR input)"),
                         (axes[1], Z1, "Trained state (t-SNE of fused representation)")):
        for c in range(3):
            m = labels == c
            ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.5, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        ax.set_title(title); ax.legend(markerscale=3, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    cm = np.zeros((3, 3), dtype=int)
    for t_, p_ in zip(labels, preds):
        cm[t_, p_] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    axes[2].imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, f"{cmn[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center",
                         color="white" if cmn[i, j] > 55 else "black", fontsize=8)
    axes[2].set_xticks(range(3)); axes[2].set_xticklabels(CLASS_NAMES)
    axes[2].set_yticks(range(3)); axes[2].set_yticklabels(CLASS_NAMES)
    axes[2].set_title(f"Confusion (accuracy={np.trace(cm) / cm.sum() * 100:.1f}%)")
    fig.suptitle(f"Single-subject view -- {subject_id}, held-out session {session_id}", fontsize=12)
    return _save(fig, os.path.join(output_dir, f"{subject_id}_S{session_id}_single_subject_tsne.png"))


def run_group_level_analysis(results_dir, data_root, output_dir, subjects=None,
                             sessions=(1, 2, 3), device="cpu", perplexity=30,
                             max_points=6000, with_noise=True, seed=0):
    """Loads every available checkpoint ONE AT A TIME (memory-friendly), and
    pools test-set outputs over subjects/sessions. Saves:
      * group t-SNE of the fused representation (by class / correct vs wrong /
        by subject) and of the EEG and EEG-HbO branch features
      * pooled confusion matrix
      * group CGSCR summary: gate per class, coupling C, token selection
      * group EEG-noise robustness curve (mean +- SEM over subjects)
    """
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    os.makedirs(output_dir, exist_ok=True)
    subjects = subjects or [f"VP{n:03d}" for n in range(1, 27)]

    pooled = {"fused": [], "feat_1": [], "feat_12": [], "y": [], "pred": [], "subj": []}
    cg = {"gate": [], "coupling": [], "selection": [], "y": []}
    noise_runs = {}
    tok_names, n_loaded = None, 0
    for si, sid in enumerate(subjects):
        for s in sessions:
            path = checkpoint_path(results_dir, sid, s)
            if not os.path.exists(path):
                continue
            model, norm, _ = load_mgfnet_checkpoint(path, device)
            _, test_loader = build_split_loaders(data_root, sid, s)
            d = extract_outputs(model, test_loader, device, norm)
            pooled["fused"].append(d["fused"])
            for k in ("feat_1", "feat_12"):
                if k in d:
                    pooled[k].append(d[k])
            pooled["y"].append(d["y_true"])
            pooled["pred"].append(d["logits_hyb"].argmax(1))
            pooled["subj"].append(np.full(len(d["y_true"]), si))
            if "gate" in d:
                for k in ("gate", "coupling", "selection"):
                    cg[k].append(d[k])
                cg["y"].append(d["y_true"])
                tok_names = tok_names or token_labels()
            if with_noise:
                r = evaluate_eeg_noise_robustness(model, norm, test_loader, device, seed=seed)
                for k, v in r.items():
                    noise_runs.setdefault(k, []).append(v)
            n_loaded += 1
            print(f"    [{n_loaded}] {sid} session {s}: {len(d['y_true'])} windows")
            del model, test_loader, d
            gc.collect()
    if n_loaded == 0:
        raise FileNotFoundError(f"no *_mgf_final.pt checkpoints found in {results_dir}")

    y = np.concatenate(pooled["y"]); pred = np.concatenate(pooled["pred"])
    subj = np.concatenate(pooled["subj"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), min(max_points, len(y)), replace=False)
    figs = []
    for key, title in (("fused", "fused representation"), ("feat_1", "EEG branch"),
                       ("feat_12", "EEG-HbO branch")):
        if not pooled[key]:
            continue
        X = StandardScaler().fit_transform(np.concatenate(pooled[key])[idx])
        Z = TSNE(n_components=2, perplexity=min(perplexity, max(5, len(X) // 10)),
                 init="pca", random_state=seed).fit_transform(X)
        yp, pp, sp = y[idx], pred[idx], subj[idx]
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for c in range(3):
            m = yp == c
            axes[0].scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.35, color=CLASS_COLORS[c], label=CLASS_NAMES[c])
        axes[0].set_title("colour = true class"); axes[0].legend(markerscale=3)
        wrong = pp != yp
        axes[1].scatter(Z[~wrong, 0], Z[~wrong, 1], s=4, alpha=0.3, color="#55A868", label="correct")
        axes[1].scatter(Z[wrong, 0], Z[wrong, 1], s=7, alpha=0.6, color="#C44E52", marker="x",
                        label="misclassified")
        axes[1].set_title(f"correct vs misclassified (error={wrong.mean() * 100:.1f}%)")
        axes[1].legend(markerscale=3)
        axes[2].scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.5, c=sp, cmap="tab20")
        axes[2].set_title("colour = subject (clusters by subject = subject-specific space)")
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"Group-level t-SNE -- {title} -- {n_loaded} held-out sessions", fontsize=12)
        figs.append(_save(fig, os.path.join(output_dir, f"group_tsne_{key}.png")))

    cm = np.zeros((3, 3), dtype=int)
    for t_, p_ in zip(y, pred):
        cm[t_, p_] += 1
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.1f}%\n({cm[i, j]})", ha="center", va="center",
                    color="white" if cmn[i, j] > 55 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Pooled confusion -- accuracy {np.trace(cm) / cm.sum() * 100:.1f}%")
    fig.colorbar(im, ax=ax, fraction=0.046)
    figs.append(_save(fig, os.path.join(output_dir, "group_confusion_matrix.png")))

    summary = {"n_sessions": n_loaded, "pooled_accuracy": float(np.trace(cm) / cm.sum())}
    if cg["gate"]:
        G = np.concatenate(cg["gate"]); C = np.concatenate(cg["coupling"])
        S = np.concatenate(cg["selection"]); yc = np.concatenate(cg["y"])
        fig, axes = plt.subplots(1, 3, figsize=(20, 4.8))
        w = 0.27
        for c in range(3):
            m = yc == c
            axes[0].bar(np.arange(6) + (c - 1) * w, G[m].mean(0), w, color=CLASS_COLORS[c],
                        label=CLASS_NAMES[c])
            axes[2].bar(np.arange(15) + (c - 1) * w, S[m].mean(0), w, color=CLASS_COLORS[c])
        axes[0].axhline(1 / 6, color="gray", ls="--")
        axes[0].set_xticks(range(6)); axes[0].set_xticklabels([BRANCH_NAMES[k] for k in BRANCH_KEYS],
                                                             rotation=30)
        axes[0].set_title("gate weight per class"); axes[0].legend(fontsize=8)
        im = axes[1].imshow(C.mean(0), cmap="magma", vmin=0, vmax=1)
        axes[1].set_xticks(range(3)); axes[1].set_xticklabels(NIRS_SCALE_NAMES)
        axes[1].set_yticks(range(5)); axes[1].set_yticklabels(EEG_BAND_NAMES)
        axes[1].set_title("mean coupling C"); fig.colorbar(im, ax=axes[1], fraction=0.046)
        axes[2].axhline(1 / 15, color="gray", ls="--")
        axes[2].set_xticks(range(15)); axes[2].set_xticklabels(tok_names, rotation=60, ha="right",
                                                              fontsize=7)
        hist = S.mean(0); nz = hist[hist > 0]
        ent = float(-(nz * np.log2(nz)).sum())
        axes[2].set_title(f"token selection (entropy {ent:.2f} / {np.log2(15):.2f} bits)")
        fig.suptitle(f"Group-level CGSCR summary -- {n_loaded} held-out sessions", fontsize=12)
        figs.append(_save(fig, os.path.join(output_dir, "group_cgscr_summary.png")))
        summary.update({"gate_mean": dict(zip([BRANCH_NAMES[k] for k in BRANCH_KEYS],
                                              G.mean(0).round(4).tolist())),
                        "coupling_mean": C.mean(0).round(4).tolist(),
                        "selection_entropy_bits": ent})
    if noise_runs:
        figs.append(plot_noise_curves({"MGFNet": noise_runs}, output_dir,
                                      f"EEG noise robustness, mean +- SEM over {n_loaded} sessions",
                                      "group_eeg_noise_robustness.png"))
        summary["noise_mean_acc"] = {k: float(np.mean([r["acc"] for r in v]))
                                     for k, v in noise_runs.items()}
    summary["figures"] = figs
    with open(os.path.join(output_dir, "group_xai_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Group-level analyses saved -> {output_dir}")
    return summary


def compare_noise_robustness(variant_dirs, data_root, output_dir, subjects=None,
                             sessions=(1, 2, 3), device="cpu", snr_list=(10, 5, 0, -5, -10),
                             seed=0):
    """Paper Fig. 6: clean-trained variants (e.g. {'MGFNet': dir_full,
    'w/o CGSCR': dir_noCGSCR}) tested with EEG-only AWGN. Only (subject,
    session) pairs present for EVERY variant are used, so the curves are paired."""
    os.makedirs(output_dir, exist_ok=True)
    subjects = subjects or [f"VP{n:03d}" for n in range(1, 27)]
    pairs = [(sid, s) for sid in subjects for s in sessions
             if all(os.path.exists(checkpoint_path(d, sid, s)) for d in variant_dirs.values())]
    if not pairs:
        raise FileNotFoundError("no (subject, session) checkpoint present in every variant dir")
    curves = {}
    for label, d in variant_dirs.items():
        runs = {}
        for sid, s in pairs:
            model, norm, _ = load_mgfnet_checkpoint(checkpoint_path(d, sid, s), device)
            _, test_loader = build_split_loaders(data_root, sid, s)
            for k, v in evaluate_eeg_noise_robustness(model, norm, test_loader, device,
                                                      snr_list, seed).items():
                runs.setdefault(k, []).append(v)
            del model, test_loader
            gc.collect()
        curves[label] = runs
    path = plot_noise_curves(curves, output_dir,
                             f"Clean-train / noisy-test (EEG AWGN), {len(pairs)} paired sessions",
                             "compare_eeg_noise_robustness.png")
    out = {label: {k: float(np.mean([r["acc"] for r in v])) for k, v in runs.items()}
           for label, runs in curves.items()}
    with open(os.path.join(output_dir, "compare_eeg_noise_robustness.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_pairs": len(pairs), "mean_acc": out}, f, indent=2)
    return path, out


# ============================================================================
# One-call per-fold driver
# ============================================================================
def run_fold_xai(results_dir, data_root, subject_id, session, output_dir=None, device="cpu",
                 with_shap=True, shap_kwargs=None):
    """Every per-fold analysis for one checkpoint. SHAP is the slowest step
    (and needs `pip install shap`); pass with_shap=False to skip it."""
    output_dir = output_dir or os.path.join(results_dir, "xai")
    os.makedirs(output_dir, exist_ok=True)
    model, norm, _ = load_mgfnet_checkpoint(checkpoint_path(results_dir, subject_id, session),
                                            device)
    _, test_loader = build_split_loaders(data_root, subject_id, session)
    names = read_eeg_channel_names(data_root, subject_id)
    out = {"subject_id": subject_id, "session": session}
    out["occlusion"] = evaluate_modality_occlusion(model, norm, test_loader, device)
    out["eeg_noise"] = evaluate_eeg_noise_robustness(model, norm, test_loader, device)
    plot_noise_curves({"MGFNet": out["eeg_noise"]}, output_dir,
                      f"EEG noise robustness -- {subject_id} session {session}",
                      f"{subject_id}_S{session}_eeg_noise_robustness.png")
    out["gradcam"] = run_gradcam_analysis(model, norm, test_loader, device, subject_id,
                                          session, output_dir)
    out["single_tsne"] = run_single_subject_tsne(model, norm, test_loader, device,
                                                 subject_id, session, output_dir)
    if with_shap:
        kw = shap_kwargs or {}
        out["shap"] = run_shap_analysis(model, norm, test_loader, device, subject_id, session,
                                        output_dir, eeg_channel_names=names, **kw)
        out["gate_shap"] = run_gate_shap_analysis(model, norm, test_loader, device,
                                                  subject_id, session, output_dir, **kw)
    with open(os.path.join(output_dir, f"{subject_id}_S{session}_xai_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str)
    ap.add_argument("--data-root", type=str, required=True)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--subject", type=str)
    ap.add_argument("--session", type=int)
    ap.add_argument("--group", action="store_true")
    ap.add_argument("--subjects", type=str, default=None, help="comma-separated, default all 26")
    ap.add_argument("--compare-noise", nargs="+", metavar="LABEL=DIR")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    subjects = a.subjects.split(",") if a.subjects else None

    if a.compare_noise:
        dirs = dict(item.split("=", 1) for item in a.compare_noise)
        out = a.out or os.path.join(a.data_root, "xai_compare_noise")
        compare_noise_robustness(dirs, a.data_root, out, subjects, device=a.device)
    elif a.group:
        run_group_level_analysis(a.results_dir, a.data_root,
                                 a.out or os.path.join(a.results_dir, "group_level_xai"),
                                 subjects, device=a.device)
    else:
        if not (a.results_dir and a.subject and a.session):
            ap.error("per-fold mode needs --results-dir, --subject and --session")
        run_fold_xai(a.results_dir, a.data_root, a.subject, a.session, a.out, a.device,
                     with_shap=not a.no_shap)


if __name__ == "__main__":
    main()
