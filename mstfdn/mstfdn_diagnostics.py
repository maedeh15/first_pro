"""
mstfdn_diagnostics.py
================================
Comprehensive diagnostic visualization for MSTFDN (MSTFDN_model_v2.py).

REWRITTEN from the TSFNet_With_fNIRS_2D diagnostics script to match the
ACTUAL MSTFDN architecture:

  - EEG branch:    E-TFEM -> E-SFEM -> E-STFEM -> f_eeg_st (KSS*CH_e, T_f)
  - fNIRS branch:  F-TFEM -> F-SFEM -> F-STFEM -> f_fnirs_st (KSS*CH_f, T_f)
  - Fusion branch: concat -> EFF-SFEM -> f_fusion_s (KSS*(CH_e+CH_f), T_f)
  - Classifier:    cls_conv1 -> ReLU -> cls_conv2 -> ReLU -> GAP -> Softmax
                   ONE classifier only (unlike TSFNet's three separate
                   fc_eeg/fc_fnirs/fc_fusion heads) -- so there is only a
                   SINGLE "Final" confusion matrix, not four per-branch ones.
  - forward() returns (logits, heads_e, heads_f, heads_ef) -- a 4-tuple,
    not TSFNet's 8-tuple.
  - 4 classes for N-back: 0-back, 2-back, 3-back, rest (paper Sec 3.2).

KEY DIFFERENCES vs the original TSFNet diagnostics script:
  1. No per-branch classifier heads exist -> only ONE confusion matrix
     ("Final"), and per-class accuracy is computed from the single final
     classifier, not compared across 4 separate branch heads.
  2. Branch FEATURES still exist (f_eeg_st, f_fnirs_st, f_fusion_s) and are
     extracted via forward hooks on e_stfem / f_stfem / eff_sfem, then
     GAP-pooled over the time dimension T_f to get a fixed-size vector
     per branch for PCA/t-SNE (these modules output (B, C, T_f), not a
     ready-made (B, 64) vector like TSFNet's Linear layers).
  3. No BiLSTM/cudnn RNN in MSTFDN -> gradient saliency does NOT need the
     model.train()-with-frozen-BN workaround; plain model.eval() works.
  4. EEG input shape is (B, 5, 28, 2000) [band, channel, time], fNIRS is
     (B, 3, 36, 100) [hemo, channel, time] -- saliency plots are adapted
     to these shapes (band-averaged EEG saliency over the 10s window).
  5. CLASS_NAMES updated to 4 classes.

HOW TO USE
----------
    from mstfdn_diagnostics import run_diagnostics

    model = MSTFDN(cfg)
    model.load_state_dict(torch.load(checkpoint_path)['model_state'])

    run_diagnostics(
        model        = model,
        train_loader = train_loader,
        test_loader  = test_loader,
        device       = device,
        subject_id   = "VP001",
        fold_id      = 1,
        output_dir   = os.path.join(OUTPUT_PATH, "diagnostics", "VP001_fold1"),
    )

This is a POST-HOC analysis tool -- run it AFTER training finishes (e.g.
right after loading a saved best-model checkpoint from
MSTFDN_train_LOSO_v2.py), not during the training loop itself. To run it
for several subjects for a report, loop over subjects and call
run_diagnostics() once per (subject, fold) with a fresh output_dir.
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (confusion_matrix, silhouette_score,
                              davies_bouldin_score, classification_report)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CLASS_NAMES = ["0-back", "2-back", "3-back", "rest"]   # paper Sec 3.2, 4 classes
N_CLASS     = len(CLASS_NAMES)
COLORS      = ["#2a78d6", "#e24b4a", "#1baf7a", "#9b59b6"]


# ---------------------------------------------------------------------------
# 1. Feature extraction via forward hooks on the 3 branch output modules
#    (e_stfem, f_stfem, eff_sfem), GAP-pooled over T_f to a fixed vector.
# ---------------------------------------------------------------------------

def _extract_features(model, loader, device):
    """
    One forward pass over `loader`, collecting:
      feat_eeg    : (N, KSS*CH_e)          -- GAP-pooled E-STFEM output
      feat_fnirs  : (N, KSS*CH_f)          -- GAP-pooled F-STFEM output
      feat_fusion : (N, KSS*(CH_e+CH_f))   -- GAP-pooled EFF-SFEM output
      feat_combined : concat of the three
      logits      : (N, N_CLASS)           -- final classifier output
      labels      : (N,)
    """
    model.eval()
    feat_store = {"eeg": [], "fnirs": [], "fusion": []}

    def _hook_factory(key):
        def _hook(module, inp, out):
            # out shape: (B, C, T_f) for e_stfem/f_stfem/eff_sfem -> GAP over T_f
            pooled = out.mean(dim=-1).detach().cpu().numpy()   # (B, C)
            feat_store[key].append(pooled)
        return _hook

    h1 = model.e_stfem.register_forward_hook(_hook_factory("eeg"))
    h2 = model.f_stfem.register_forward_hook(_hook_factory("fnirs"))
    # eff_sfem returns (x, head_outputs) as a tuple -- hook sees the whole
    # tuple as `out`, so index [0] for the feature tensor
    def _fusion_hook(module, inp, out):
        pooled = out[0].mean(dim=-1).detach().cpu().numpy()
        feat_store["fusion"].append(pooled)
    h3 = model.eff_sfem.register_forward_hook(_fusion_hook)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for eeg_b, nirs_b, lbl_b in loader:
            logits, _, _, _ = model(eeg_b.to(device), nirs_b.to(device))
            all_logits.append(logits.cpu().numpy())
            all_labels.append(lbl_b.numpy())

    h1.remove(); h2.remove(); h3.remove()

    feat_eeg    = np.concatenate(feat_store["eeg"],    axis=0)
    feat_fnirs  = np.concatenate(feat_store["fnirs"],  axis=0)
    feat_fusion = np.concatenate(feat_store["fusion"], axis=0)
    logits      = np.concatenate(all_logits, axis=0)
    labels      = np.concatenate(all_labels, axis=0)
    feat_combined = np.concatenate([feat_eeg, feat_fnirs, feat_fusion], axis=1)

    del feat_store, all_logits, all_labels
    gc.collect()

    return {
        "feat_eeg": feat_eeg, "feat_fnirs": feat_fnirs, "feat_fusion": feat_fusion,
        "feat_combined": feat_combined, "logits": logits, "labels": labels,
    }


# ---------------------------------------------------------------------------
# 2. Confusion matrix -- ONLY ONE (MSTFDN has a single classifier, unlike
#    TSFNet's separate per-branch heads)
# ---------------------------------------------------------------------------

def _plot_confusion(logits, labels, tag, output_dir):
    preds = np.argmax(logits, axis=1)
    cm    = confusion_matrix(labels, preds, labels=list(range(N_CLASS)))
    cmn   = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cmn, vmin=0, vmax=100, cmap="Blues")
    ax.set_xticks(range(N_CLASS)); ax.set_yticks(range(N_CLASS))
    ax.set_xticklabels(CLASS_NAMES, fontsize=9, rotation=30)
    ax.set_yticklabels(CLASS_NAMES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (acc={cm.diagonal().sum()/cm.sum()*100:.1f}%)", fontsize=10)
    for i in range(N_CLASS):
        for j in range(N_CLASS):
            ax.text(j, i, f"{cmn[i,j]:.1f}%\n({cm[i,j]})",
                    ha="center", va="center",
                    color="white" if cmn[i,j] > 55 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{tag} -- Final Classifier", fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{tag}_01_confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 3. Per-class accuracy (single classifier -- no per-branch comparison
#    possible since MSTFDN has no separate branch heads)
# ---------------------------------------------------------------------------

def _plot_per_class_acc(logits, labels, tag, output_dir):
    preds = np.argmax(logits, axis=1)
    accs  = [(preds[labels == c] == c).mean() * 100 if (labels == c).sum() > 0 else 0
             for c in range(N_CLASS)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(CLASS_NAMES, accs, color=COLORS, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.5,
                f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.axhline(100.0 / N_CLASS, ls="--", color="gray", lw=0.8,
               label=f"Chance ({100.0/N_CLASS:.1f}%)")
    ax.legend(fontsize=9)
    ax.set_title(f"{tag} -- Per-class Accuracy")
    plt.tight_layout()
    path = os.path.join(output_dir, f"{tag}_02_per_class_acc.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 4. PCA -- per branch feature space (eeg_st, fnirs_st, fusion)
# ---------------------------------------------------------------------------

def _plot_pca(train_data, test_data, tag, output_dir):
    for feat_key, title in [("feat_eeg",      "EEG branch (E-STFEM, GAP-pooled)"),
                              ("feat_fnirs",    "fNIRS branch (F-STFEM, GAP-pooled)"),
                              ("feat_fusion",   "Fusion branch (EFF-SFEM, GAP-pooled)"),
                              ("feat_combined", "All branches concatenated")]:
        X_tr, y_tr = train_data[feat_key], train_data["labels"]
        X_te, y_te = test_data[feat_key],  test_data["labels"]

        scaler = StandardScaler().fit(X_tr)
        pca    = PCA(n_components=2, random_state=42).fit(scaler.transform(X_tr))
        Z_tr   = pca.transform(scaler.transform(X_tr))
        Z_te   = pca.transform(scaler.transform(X_te))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, (Z, y, split) in zip(axes, [(Z_tr, y_tr, "Train"), (Z_te, y_te, "Test")]):
            for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
                mask = y == c
                ax.scatter(Z[mask,0], Z[mask,1], s=8, alpha=0.5, color=col, label=cname)
            ax.set_title(f"PCA -- {split} | {title}")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.legend(markerscale=3, fontsize=8)

        plt.suptitle(f"{tag} -- PCA Feature Space", fontsize=12)
        plt.tight_layout()
        fkey = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"{tag}_03_pca_{fkey}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 5. t-SNE + misclassified overlay
# ---------------------------------------------------------------------------

def _plot_tsne(test_data, tag, output_dir, perplexity=30):
    for feat_key, title in [("feat_eeg",      "EEG branch (E-STFEM)"),
                              ("feat_fnirs",    "fNIRS branch (F-STFEM)"),
                              ("feat_fusion",   "Fusion branch (EFF-SFEM)")]:
        X    = test_data[feat_key]
        y    = test_data["labels"]
        pred = np.argmax(test_data["logits"], axis=1)

        X_s = StandardScaler().fit_transform(X)
        n_max = min(3000, len(X_s))
        idx   = np.random.choice(len(X_s), n_max, replace=False)
        X_s, y_s, pred_s = X_s[idx], y[idx], pred[idx]

        print(f"    t-SNE {feat_key} ({n_max} samples)...", end="", flush=True)
        Z = TSNE(n_components=2, perplexity=min(perplexity, max(5, n_max // 4)),
                 random_state=42, max_iter=1000).fit_transform(X_s)
        print(" done")

        wrong = pred_s != y_s
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        ax = axes[0]
        for c, (cname, col) in enumerate(zip(CLASS_NAMES, COLORS)):
            mask = y_s == c
            ax.scatter(Z[mask,0], Z[mask,1], s=7, alpha=0.45, color=col, label=cname)
        ax.set_title("Colour = true class"); ax.legend(markerscale=3, fontsize=9)

        ax = axes[1]
        ax.scatter(Z[~wrong,0], Z[~wrong,1], s=6, alpha=0.35, color="#1baf7a", label="Correct")
        ax.scatter(Z[wrong,0], Z[wrong,1], s=10, alpha=0.7, color="#e24b4a",
                   label="Misclassified", marker="x")
        ax.set_title(f"Correct vs Misclassified (error {wrong.mean()*100:.1f}%)")
        ax.legend(markerscale=3, fontsize=9)

        fig.suptitle(f"{tag} -- t-SNE -- {title}", fontsize=12)
        plt.tight_layout()
        fkey = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"{tag}_04_tsne_{fkey}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 6. Train vs Test distribution shift
# ---------------------------------------------------------------------------

def _plot_distribution_shift(train_data, test_data, tag, output_dir):
    for feat_key, title in [("feat_eeg",    "EEG branch (E-STFEM)"),
                              ("feat_fnirs",  "fNIRS branch (F-STFEM)")]:
        X_tr = train_data[feat_key]; X_te = test_data[feat_key]
        mu_tr, mu_te   = X_tr.mean(0), X_te.mean(0)
        std_tr, std_te = X_tr.std(0),  X_te.std(0)
        dims = np.arange(X_tr.shape[1])

        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax = axes[0]
        ax.plot(dims, mu_tr, color="#2a78d6", lw=1.0, label="Train mean")
        ax.plot(dims, mu_te, color="#e24b4a", lw=1.0, label="Test mean")
        ax.fill_between(dims, mu_tr-std_tr, mu_tr+std_tr, color="#2a78d6", alpha=0.15)
        ax.fill_between(dims, mu_te-std_te, mu_te+std_te, color="#e24b4a", alpha=0.15)
        ax.set_xlabel("Feature dimension (KSS * channels)"); ax.set_ylabel("Activation")
        ax.set_title(f"Mean +/- Std -- {title}"); ax.legend()

        ax = axes[1]
        shift = np.abs(mu_tr - mu_te)
        ax.bar(dims, shift, color="#ef9f27", width=0.8)
        ax.set_xlabel("Feature dimension")
        ax.set_ylabel("|Train mean - Test mean|")
        ax.set_title("Distribution shift per dimension (high = session-to-session shift)")
        top5 = np.argsort(shift)[-5:]
        for d in top5:
            ax.axvline(d, color="red", lw=0.8, ls="--")

        plt.suptitle(f"{tag} -- Distribution Shift", fontsize=12)
        plt.tight_layout()
        fkey = feat_key.replace("feat_", "")
        path = os.path.join(output_dir, f"{tag}_05_dist_shift_{fkey}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 7. Class centroid distance matrix (4x4 now)
# ---------------------------------------------------------------------------

def _plot_centroid_distance(train_data, test_data, tag, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (data, split) in zip(axes, [(train_data, "Train"), (test_data, "Test")]):
        X = data["feat_combined"]; y = data["labels"]
        centroids = np.array([
            X[y == c].mean(0) if (y == c).sum() > 0 else np.zeros(X.shape[1])
            for c in range(N_CLASS)
        ])
        dist = np.array([[np.linalg.norm(centroids[i]-centroids[j]) for j in range(N_CLASS)]
                         for i in range(N_CLASS)])
        im = ax.imshow(dist, cmap="YlOrRd")
        ax.set_xticks(range(N_CLASS)); ax.set_yticks(range(N_CLASS))
        ax.set_xticklabels(CLASS_NAMES, rotation=30); ax.set_yticklabels(CLASS_NAMES)
        for i in range(N_CLASS):
            for j in range(N_CLASS):
                ax.text(j, i, f"{dist[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"{split} -- Centroid Distance")

    plt.suptitle(f"{tag} -- Class Centroid Distance (combined features)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{tag}_06_centroid_dist.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 8. Confidence & entropy
# ---------------------------------------------------------------------------

def _plot_confidence(test_data, tag, output_dir):
    logits = test_data["logits"]; labels = test_data["labels"]
    probs   = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds   = np.argmax(logits, axis=1)
    correct = preds == labels

    max_prob = probs.max(axis=1)
    entropy  = -(probs * np.log(probs + 1e-9)).sum(axis=1)
    max_ent  = np.log(N_CLASS)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.hist(max_prob[correct],  bins=40, alpha=0.7, color="#1baf7a",
            label=f"Correct ({correct.sum()})", density=True)
    ax.hist(max_prob[~correct], bins=40, alpha=0.7, color="#e24b4a",
            label=f"Wrong ({(~correct).sum()})", density=True)
    ax.set_xlabel("Max softmax probability"); ax.set_ylabel("Density")
    ax.set_title("Confidence Distribution"); ax.legend()
    ax.axvline(1.0 / N_CLASS, color="gray", ls="--", lw=0.8)

    ax = axes[1]
    ax.hist(entropy[correct]/max_ent,  bins=40, alpha=0.7, color="#1baf7a",
            label="Correct", density=True)
    ax.hist(entropy[~correct]/max_ent, bins=40, alpha=0.7, color="#e24b4a",
            label="Wrong", density=True)
    ax.set_xlabel("Normalised entropy (0=certain, 1=uniform)")
    ax.set_ylabel("Density")
    ax.set_title("Prediction Uncertainty (Entropy)"); ax.legend()

    plt.suptitle(f"{tag} -- Confidence & Uncertainty", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{tag}_07_confidence.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 9. Cluster metrics
# ---------------------------------------------------------------------------

def _compute_cluster_metrics(data):
    results = {}
    for feat_key in ["feat_eeg", "feat_fnirs", "feat_fusion", "feat_combined"]:
        X = data[feat_key]; y = data["labels"]
        n_classes_present = len(np.unique(y))
        if n_classes_present < 2:
            results[feat_key] = {"silhouette": float("nan"), "davies_bouldin": float("nan")}
            continue
        n   = min(2000, len(X))
        idx = np.random.choice(len(X), n, replace=False)
        Xs, ys = StandardScaler().fit_transform(X[idx]), y[idx]
        results[feat_key] = {
            "silhouette":     silhouette_score(Xs, ys, metric="euclidean"),
            "davies_bouldin": davies_bouldin_score(Xs, ys),
        }
    return results


# ---------------------------------------------------------------------------
# 10. Input gradient saliency (EEG) -- NO cudnn RNN issue in MSTFDN
#     (no LSTM anywhere), so plain model.eval() works for backward().
# ---------------------------------------------------------------------------

def _plot_saliency(model, test_data_loader, device, tag, output_dir, n_samples=200):
    model.eval()

    _grads = {c: [] for c in range(N_CLASS)}
    collected = 0

    for eeg_b, nirs_b, lbl_b in test_data_loader:
        if collected >= n_samples:
            break
        eeg_b  = eeg_b.to(device).float()
        nirs_b = nirs_b.to(device).float()
        lbl_np = lbl_b.numpy()
        eeg_b.requires_grad_(True)

        logits, _, _, _ = model(eeg_b, nirs_b)   # (B, N_CLASS)

        for c in range(N_CLASS):
            mask = (lbl_np == c)
            if mask.sum() == 0:
                continue
            if eeg_b.grad is not None:
                eeg_b.grad.zero_()
            model.zero_grad()
            score = logits[mask, c].sum()
            score.backward(retain_graph=(c < N_CLASS - 1))
            if eeg_b.grad is not None:
                grad = eeg_b.grad.detach().cpu().abs()   # (B, 5, 28, 2000)
                _grads[c].append(grad[mask].mean(0))       # (5, 28, 2000)

        eeg_b = eeg_b.detach()
        collected += len(lbl_np)

    if not any(_grads[c] for c in range(N_CLASS)):
        print("  [saliency] no gradients collected -- skipped.")
        return

    fig, axes = plt.subplots(N_CLASS, 1, figsize=(14, 3 * N_CLASS), sharex=True)
    for c, ax in enumerate(axes):
        if not _grads[c]:
            ax.set_ylabel(f"|Gradient|\n{CLASS_NAMES[c]}", fontsize=9)
            continue
        g = torch.stack(_grads[c]).mean(0)          # (5, 28, 2000) -- avg over collected samples
        g_time = g.mean(dim=(0, 1)).numpy()          # avg over band & channel -> (2000,)
        t = np.linspace(0, 10, len(g_time))          # 10s EEG window
        ax.plot(t, g_time, color=COLORS[c % len(COLORS)], lw=1.0)
        ax.fill_between(t, 0, g_time, color=COLORS[c % len(COLORS)], alpha=0.3)
        ax.set_ylabel(f"|Gradient|\n{CLASS_NAMES[c]}", fontsize=9)
        ax.set_xlim(0, 10)

    axes[-1].set_xlabel("Time within window (seconds)")
    plt.suptitle(f"{tag} -- Input Gradient Saliency (EEG, band+channel avg)\n"
                 "(which time points drive predictions)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f"{tag}_08_saliency.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  [saved] {path}")


# ---------------------------------------------------------------------------
# 11. Text report
# ---------------------------------------------------------------------------

def _write_report(test_data, cluster_metrics, tag, output_dir):
    logits = test_data["logits"]; labels = test_data["labels"]
    preds  = np.argmax(logits, axis=1)

    lines = [
        f"DIAGNOSTIC REPORT -- {tag}", "=" * 60,
        "\n[Classification Report -- Final Classifier]",
        classification_report(labels, preds, target_names=CLASS_NAMES,
                              labels=list(range(N_CLASS)), zero_division=0),
        "\n[Cluster Separability Metrics]",
        f"{'Feature':20s}  {'Silhouette':>12}  {'Davies-Bouldin':>15}",
        "-" * 52,
    ]
    for k, v in cluster_metrics.items():
        tag2 = k.replace("feat_", "")
        lines.append(f"{tag2:20s}  {v['silhouette']:12.4f}  {v['davies_bouldin']:15.4f}")
    lines += [
        "\nNote: Silhouette in [-1,1] -- higher is better separable.",
        "      Davies-Bouldin >= 0    -- lower is better separable.",
        "\n[Model Architecture Notes -- MSTFDN]",
        "  EEG branch:   E-TFEM (4-scale dilated conv, groups=5 bands) ->",
        "                E-SFEM (Transformer, He heads, BN in FFN) ->",
        "                E-STFEM (depthwise stride=20 + pointwise KSS)",
        "  fNIRS branch: F-TFEM (2-scale dilated conv, groups=3 hemo) ->",
        "                F-SFEM (Transformer, Hf heads, LN in FFN) ->",
        "                F-STFEM (depthwise kernel=11 + pointwise KSS)",
        "  Fusion:       concat(eeg_st, fnirs_st) -> EFF-SFEM (Hef heads, LN) ->",
        "                Classifier: Conv1d->ReLU->Conv1d->ReLU->GAP->Softmax",
        "  Loss:         CrossEntropy + L2-norm(lambda_nw) + HSIC(lambda_mhs)",
        "  NOTE: MSTFDN has a SINGLE classifier (unlike TSFNet's 3 separate",
        "        branch heads), so only one confusion matrix/accuracy exists;",
        "        PCA/t-SNE are still shown per-branch using the intermediate",
        "        E-STFEM / F-STFEM / EFF-SFEM feature representations.",
    ]

    path = os.path.join(output_dir, f"{tag}_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [saved] {path}")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def run_diagnostics(model, train_loader, test_loader, device,
                    subject_id: str, fold_id: int, output_dir: str):
    """
    Parameters
    ----------
    model        : trained MSTFDN instance (already .to(device))
    train_loader : DataLoader for the training sessions of this fold
    test_loader  : DataLoader for the test session of this fold
    device       : 'cuda' or 'cpu'
    subject_id   : e.g. 'VP001'
    fold_id      : 1, 2, or 3 (LOSO fold number)
    output_dir   : folder where all PNGs and the report are saved

    Call this AFTER training (e.g. right after loading a saved checkpoint
    from MSTFDN_train_LOSO_v2.py), once per (subject, fold). To generate a
    report across several subjects, loop over subjects and call this once
    per subject with a distinct output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{subject_id}_fold{fold_id}"
    print(f"\n{'='*60}")
    print(f"  MSTFDN DIAGNOSTICS -- {tag}")
    print(f"{'='*60}")

    print("\n[1/9] Extracting features -- train set...")
    train_data = _extract_features(model, train_loader, device)

    print("[2/9] Extracting features -- test set...")
    test_data  = _extract_features(model, test_loader, device)

    labels = test_data["labels"]
    logits = test_data["logits"]

    print("[3/9] Confusion matrix...")
    _plot_confusion(logits, labels, tag, output_dir)

    print("[4/9] Per-class accuracy...")
    _plot_per_class_acc(logits, labels, tag, output_dir)

    print("[5/9] PCA...")
    _plot_pca(train_data, test_data, tag, output_dir)

    print("[6/9] t-SNE (1-2 min)...")
    _plot_tsne(test_data, tag, output_dir)

    print("[7/9] Distribution shift...")
    _plot_distribution_shift(train_data, test_data, tag, output_dir)

    _plot_centroid_distance(train_data, test_data, tag, output_dir)
    _plot_confidence(test_data, tag, output_dir)

    print("[+] Cluster metrics...")
    cluster_metrics = _compute_cluster_metrics(test_data)

    print("[+] Gradient saliency (EEG)...")
    _plot_saliency(model, test_loader, device, tag, output_dir)

    print("[+] Writing report...")
    _write_report(test_data, cluster_metrics, tag, output_dir)

    del train_data, test_data, logits, labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n  All diagnostics saved -> {output_dir}\n")


# ---------------------------------------------------------------------------
# EXAMPLE: run diagnostics for multiple subjects (for a report)
# ---------------------------------------------------------------------------
#
#   import torch
#   from MSTFDN_model_v2 import MSTFDN
#   from MSTFDN_train_LOSO_v2 import get_config, MSTFDNSubjectDataset
#   from torch.utils.data import DataLoader, Subset
#   import numpy as np, os
#
#   PREPROCESSED_PATH = r"E:\...\MSTFDN_PREPROCESSED_NBACK_V3"
#   OUTPUT_PATH        = r"E:\...\MSTFDN_LOSO_V3"
#   device = 'cuda' if torch.cuda.is_available() else 'cpu'
#   cfg = get_config()
#
#   subjects_for_report = ["VP001", "VP002", "VP003"]
#   fold_id = 1   # test session S1, matches best_model_VPxxx_fold1.pt
#
#   for subject_id in subjects_for_report:
#       dataset = MSTFDNSubjectDataset(PREPROCESSED_PATH, subject_id)
#       sess = dataset.session_ids
#       test_idx  = np.where(sess == fold_id)[0]
#       train_idx = np.where(sess != fold_id)[0]
#       train_loader = DataLoader(Subset(dataset, train_idx), batch_size=16, shuffle=False)
#       test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=16, shuffle=False)
#
#       model = MSTFDN(cfg).to(device)
#       ckpt = torch.load(
#           os.path.join(OUTPUT_PATH, f"best_model_{subject_id}_fold{fold_id}.pt"),
#           map_location=device)
#       model.load_state_dict(ckpt['model_state'])
#
#       run_diagnostics(
#           model=model, train_loader=train_loader, test_loader=test_loader,
#           device=device, subject_id=subject_id, fold_id=fold_id,
#           output_dir=os.path.join(OUTPUT_PATH, "diagnostics", subject_id),
#       )
