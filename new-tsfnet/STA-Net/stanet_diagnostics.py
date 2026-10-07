"""
stanet_diagnostics.py -- behavioural / representational diagnostics for STA-Net
(rewrite of your TSFNet model_diagnostics.py; same figure set, adapted to STA-Net's outputs).

What changed vs. model_diagnostics.py, and why
----------------------------------------------
* Embeddings: STA-Net has no 512-d "enhanced_fusion". The representations the heads see are the
  256-d EGTA outputs: `fusion_emb` (spatially-aligned EEG-fusion feature) and `fnirs_emb`
  (temporally-aligned fNIRS feature), plus the EEG-branch feature `eeg_emb`.
* Final probabilities: p_final = kappa_fusion*p_fusion + kappa_fnirs*p_fnirs with
  kappa = softmax(.) -> a convex combination of two softmaxes, so it already sums to 1.
  (TSFNet needed the display-only renormalisation; here it is a harmless no-op, kept for safety.)
* "Four heads" are final / eeg / fusion / fnirs, as before. NOTE: in STA-Net the EEG head is NOT part
  of the final decision (only fusion and fNIRS are) -- it only receives an auxiliary loss.
* Plot 7 (EGTA attention): STA-Net's EGTA attends over the l=11 LAGGED fNIRS windows of ONE sample
  (window k starts k s after the EEG window starts), uniform = 1/11. This is the model's learned
  hemodynamic-delay profile; x-axis = lag k in seconds. (TSFNet: 21 segments.)
* New plot 8: instance-adaptive prediction weights kappa_fusion / kappa_fnirs per class (STA-Net's
  analogue of TSFNet's fixed w_eeg/w_fusion/w_fnirs -- here they are computed PER SAMPLE).
* Models are fed through `WindowLoader` (stanet_common.py) instead of a torch DataLoader/_unpack_batch.

Usage
-----
    from stanet_diagnostics import run_fold_diagnostics
    run_fold_diagnostics(model, train_loader, test_loader, device, subject_id, fold_idx, out_dir)
or run_stanet_loso.py --diagnostics [--diagnostics-subjects VP001,VP006|ALL]
Only whitelisted subjects are processed (cost control, identical idea to the original).
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stanet_common import (CLASS_NAMES, COLORS, LAG_STEP_SECONDS, collect_outputs)

# Same whitelist as model_diagnostics.py (strong / outlier / weak / mid / strong subjects).
DIAGNOSTIC_SUBJECTS = ["VP001", "VP006", "VP012", "VP017", "VP025"]

_KEYS = ("p_final", "p_eeg", "p_fusion", "p_fnirs", "w", "att", "fusion_emb", "fnirs_emb", "eeg_emb")


def _should_run(subject_id):
    return DIAGNOSTIC_SUBJECTS == "ALL" or subject_id in DIAGNOSTIC_SUBJECTS


def extract_embeddings(model, loader, device, max_batches=None):
    """One eval-mode pass; dict of numpy arrays (see _KEYS) + 'y_true'."""
    return collect_outputs(model, loader, device, keys=_KEYS, max_batches=max_batches)


# ---------------------------------------------------------------------------
# 1 + 2. t-SNE / PCA with the train-vs-test overlay (cross-session shift)
# ---------------------------------------------------------------------------
def plot_embedding_space(train_d, test_d, subject_id, fold_idx, out_dir,
                         key="fusion_emb", method="tsne", max_points=1500):
    """LEFT: test embedding by true class. RIGHT: train (grey) vs test (colour) in ONE shared projection
    -> a displaced test cloud is the visual signature of the cross-session domain shift."""
    from sklearn.decomposition import PCA

    Xtr, Xte = train_d[key], test_d[key]
    rng = np.random.default_rng(0)
    if len(Xtr) > max_points:
        Xtr = Xtr[rng.choice(len(Xtr), max_points, replace=False)]
    idx_te = np.arange(len(Xte))
    if len(Xte) > max_points:
        idx_te = rng.choice(len(Xte), max_points, replace=False)
    Xte_s, yte_s = Xte[idx_te], test_d["y_true"][idx_te]

    X = np.concatenate([Xtr, Xte_s])
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=0).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = max(5, min(30, (len(X) - 1) // 4))
        Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0,
                 max_iter=600).fit_transform(X)
    else:
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    Ztr, Zte = Z[:len(Xtr)], Z[len(Xtr):]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for c in range(len(CLASS_NAMES)):
        m = yte_s == c
        axes[0].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.7, color=COLORS[c], label=CLASS_NAMES[c])
    axes[0].set_title(f"{subject_id} fold {fold_idx} -- TEST embedding by class")
    axes[0].legend(markerscale=2)
    axes[1].scatter(Ztr[:, 0], Ztr[:, 1], s=7, alpha=0.30, color="#999999", label="train (2 sessions)")
    for c in range(len(CLASS_NAMES)):
        m = yte_s == c
        axes[1].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.8, color=COLORS[c], label=f"test {CLASS_NAMES[c]}")
    axes[1].set_title("TRAIN vs TEST -- cross-session shift")
    axes[1].legend(markerscale=2, fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method.upper()} of {key} (STA-Net)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_{key}_{method}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Normalized confusion heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(test_d, subject_id, fold_idx, out_dir):
    y_true, y_pred = test_d["y_true"], test_d["p_final"].argmax(1)
    cm = np.zeros((3, 3))
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_n = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_n[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center",
                    color="white" if cm_n[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{subject_id} fold {fold_idx} -- recall-normalized")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_confusion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 4 + 5. Confidence, entropy, calibration
# ---------------------------------------------------------------------------
def plot_confidence_and_calibration(test_d, subject_id, fold_idx, out_dir, n_bins=10):
    """Confidence / entropy histograms split correct-vs-incorrect, and a reliability diagram with ECE.
    p_final already sums to 1 in STA-Net (convex combination of two softmaxes); the renorm below is a
    safe no-op kept for parity with the TSFNet version."""
    p = test_d["p_final"].astype(np.float64)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)
    y = test_d["y_true"]
    pred, conf = p.argmax(1), p.max(1)
    correct = pred == y
    ent = -(p * np.log(p + 1e-12)).sum(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axes[0].hist(conf[correct], bins=20, alpha=0.65, label="correct", color="#55A868")
    axes[0].hist(conf[~correct], bins=20, alpha=0.65, label="incorrect", color="#C44E52")
    axes[0].set_xlabel("max predicted probability"); axes[0].set_ylabel("count")
    axes[0].set_title("confidence"); axes[0].legend()
    axes[1].hist(ent[correct], bins=20, alpha=0.65, label="correct", color="#55A868")
    axes[1].hist(ent[~correct], bins=20, alpha=0.65, label="incorrect", color="#C44E52")
    axes[1].set_xlabel("predictive entropy (nats)"); axes[1].set_title("uncertainty"); axes[1].legend()

    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, ece = [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            accs.append(np.nan); confs.append((lo + hi) / 2); continue
        a, c = correct[m].mean(), conf[m].mean()
        accs.append(a); confs.append(c)
        ece += (m.sum() / len(conf)) * abs(a - c)
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes[2].plot(confs, accs, "o-", color="#4C72B0", label="model")
    axes[2].set_xlabel("confidence"); axes[2].set_ylabel("accuracy")
    axes[2].set_title(f"reliability (ECE={ece:.3f})"); axes[2].legend()
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    fig.suptitle(f"{subject_id} fold {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_confidence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(ece)


# ---------------------------------------------------------------------------
# 6. Branch agreement -- complementary or redundant?
# ---------------------------------------------------------------------------
def plot_branch_agreement(test_d, subject_id, fold_idx, out_dir):
    """LEFT: pairwise agreement of the four heads. RIGHT: accuracy per head + 'oracle'
    (any of eeg/fnirs/fusion correct). NOTE: in STA-Net the final decision = fusion + fNIRS only; the
    EEG head is auxiliary, so 'recoverable_by_fusion' counts the EEG head as an outside expert."""
    y = test_d["y_true"]
    names = ["eeg", "fnirs", "fusion", "final"]
    preds = {k: test_d[f"p_{k}"].argmax(1) for k in names}
    A = np.array([[(preds[a] == preds[b]).mean() for b in names] for a in names])
    any_correct = np.zeros(len(y), dtype=bool)
    for k in ["eeg", "fnirs", "fusion"]:
        any_correct |= (preds[k] == y)
    recoverable = float((any_correct & ~(preds["final"] == y)).mean())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    im = axes[0].imshow(A, cmap="viridis", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                         color="white" if A[i, j] < 0.6 else "black", fontsize=9)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(names, rotation=45)
    axes[0].set_yticks(range(4)); axes[0].set_yticklabels(names)
    axes[0].set_title("pairwise prediction agreement")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    accs = [float((preds[k] == y).mean()) for k in names]
    oracle = float(any_correct.mean())
    axes[1].bar(names + ["oracle"], accs + [oracle], color=["#4C72B0"] * 4 + ["#55A868"])
    axes[1].axhline(1 / 3, color="gray", linestyle="--", label="chance")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title(f"branch accuracy -- recoverable by better fusion: {recoverable:.1%}")
    axes[1].legend(); plt.setp(axes[1].get_xticklabels(), rotation=45)
    fig.suptitle(f"{subject_id} fold {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_branch_agreement.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"branch_acc": dict(zip(names, accs)), "oracle_acc": oracle,
                  "recoverable_by_fusion": recoverable}


# ---------------------------------------------------------------------------
# 7. EGTA temporal attention over the l lagged fNIRS windows
# ---------------------------------------------------------------------------
def plot_egta_attention(test_d, subject_id, fold_idx, out_dir):
    """Mean EGTA attention per fNIRS lag window, per TRUE class, uniform = 1/l. Lag k = fNIRS window that
    starts k s after the EEG window starts. A flat curve on the uniform line = EGTA learned no delay
    preference; a consistent peak = the model's own estimate of the hemodynamic delay (reportable)."""
    W = test_d.get("att")
    if W is None:
        return None
    y = test_d["y_true"]
    S = W.shape[-1]
    lags = np.arange(S) * LAG_STEP_SECONDS
    fig, ax = plt.subplots(figsize=(8, 4.3))
    for c in range(len(CLASS_NAMES)):
        m = y == c
        if m.sum() == 0:
            continue
        prof = W[m].mean(0)
        sem = W[m].std(0) / max(np.sqrt(m.sum()), 1)
        ax.plot(lags, prof, "-o", ms=3, color=COLORS[c], label=CLASS_NAMES[c])
        ax.fill_between(lags, prof - sem, prof + sem, color=COLORS[c], alpha=0.18)
    ax.axhline(1 / S, color="gray", linestyle="--", label=f"uniform (1/{S})")
    ax.set_xlabel("fNIRS window lag behind the EEG window start (s)")
    ax.set_ylabel("mean EGTA attention weight (heads & query averaged)")
    ax.set_title(f"{subject_id} fold {fold_idx} -- EGTA temporal profile")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_egta_attention.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 8. NEW: instance-adaptive prediction weights kappa_fusion / kappa_fnirs
# ---------------------------------------------------------------------------
def plot_prediction_weights(test_d, subject_id, fold_idx, out_dir):
    """kappa = softmax(fc([F_fusion, F_fnirs])) is computed PER SAMPLE in STA-Net (TSFNet has three fixed
    scalars). Shows kappa_fnirs per true class and for correct vs incorrect windows (kappa_fusion = 1-kappa_fnirs)."""
    k_fn = test_d["w"][:, 1]
    y = test_d["y_true"]
    correct = test_d["p_final"].argmax(1) == y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    data = [k_fn[y == c] for c in range(3)]
    bp = axes[0].boxplot(data, patch_artist=True, showfliers=False)
    axes[0].set_xticks([1, 2, 3]); axes[0].set_xticklabels(CLASS_NAMES)
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col); patch.set_alpha(0.6)
    axes[0].axhline(0.5, color="gray", ls="--"); axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("kappa_fnirs (1 - kappa_fusion)"); axes[0].set_title("by true class")
    axes[1].hist(k_fn[correct], bins=20, alpha=0.65, color="#55A868", label="correct")
    axes[1].hist(k_fn[~correct], bins=20, alpha=0.65, color="#C44E52", label="incorrect")
    axes[1].set_xlabel("kappa_fnirs"); axes[1].legend(); axes[1].set_title("correct vs incorrect")
    fig.suptitle(f"{subject_id} fold {fold_idx} -- instance-adaptive decision weights "
                 f"(mean kappa_fnirs={k_fn.mean():.3f})", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_prediction_weights.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"kappa_fusion_mean": float(1 - k_fn.mean()), "kappa_fnirs_mean": float(k_fn.mean())}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_fold_diagnostics(model, train_loader, test_loader, device, subject_id, fold_idx, out_dir,
                         save_embeddings=True, method="tsne"):
    """Runs the whole battery for ONE fold if subject_id is whitelisted; returns a dict of paths + scalars
    (or None when skipped). Wrapped in try/except: a plot failure must never kill a training run."""
    if not _should_run(subject_id):
        return None
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    summary = {"subject_id": subject_id, "fold_idx": int(fold_idx), "figures": []}
    try:
        print(f"  [diagnostics] extracting embeddings for {subject_id} fold {fold_idx} ...")
        test_d = extract_embeddings(model, test_loader, device)
        train_d = extract_embeddings(model, train_loader, device, max_batches=60)

        summary["figures"].append(plot_embedding_space(train_d, test_d, subject_id, fold_idx, diag_dir,
                                                       key="fusion_emb", method=method))
        summary["figures"].append(plot_embedding_space(train_d, test_d, subject_id, fold_idx, diag_dir,
                                                       key="fnirs_emb", method=method))
        summary["figures"].append(plot_confusion_heatmap(test_d, subject_id, fold_idx, diag_dir))
        p, ece = plot_confidence_and_calibration(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary["ece"] = ece
        p, agree = plot_branch_agreement(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary.update(agree)
        p = plot_egta_attention(test_d, subject_id, fold_idx, diag_dir)
        if p:
            summary["figures"].append(p)
            summary["egta_max_dev_from_uniform"] = float(
                np.abs(test_d["att"].mean(0) - 1.0 / test_d["att"].shape[-1]).max())
        p, kw = plot_prediction_weights(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary.update(kw)

        if save_embeddings:
            npz = os.path.join(diag_dir, f"{subject_id}_fold{fold_idx}_embeddings.npz")
            np.savez_compressed(
                npz,
                test_fusion=test_d["fusion_emb"].astype(np.float32),
                test_fnirs=test_d["fnirs_emb"].astype(np.float32),
                test_eeg=test_d["eeg_emb"].astype(np.float32),
                test_y=test_d["y_true"], test_p_final=test_d["p_final"].astype(np.float32),
                test_att=test_d["att"].astype(np.float32), test_w=test_d["w"].astype(np.float32),
                train_fusion=train_d["fusion_emb"].astype(np.float32),
                train_fnirs=train_d["fnirs_emb"].astype(np.float32), train_y=train_d["y_true"])
            summary["embeddings_npz"] = npz
        with open(os.path.join(diag_dir, f"{subject_id}_fold{fold_idx}_diagnostics.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [diagnostics] saved {len(summary['figures'])} figure(s) to {diag_dir}")
    except Exception as exc:
        import traceback
        print(f"  [diagnostics] WARNING: failed for {subject_id} fold {fold_idx}: {exc}")
        traceback.print_exc()
        summary["error"] = str(exc)
    return summary
