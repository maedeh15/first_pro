"""
model_diagnostics.py -- behavioural / representational diagnostics for TSFNet.

WHY THIS EXISTS
---------------
The project currently saves only scalars (acc, kappa, confusion matrix) and
per-epoch curves. That tells you WHETHER a fold worked, never WHY it failed.
This module saves the things that make failure modes visible:

  1. t-SNE / PCA of the learned fusion embedding, coloured by class
     -> are the three n-back levels separable at all in feature space, or is
        the classifier head the bottleneck?
  2. TRAIN vs TEST embedding overlay (the single most important plot for this
     project) -> directly visualizes the cross-session domain shift that E3,
     E4 and EGTA all exist to fix. If test points sit in a displaced cloud,
     the problem is distribution shift, not class separability.
  3. Per-class confusion heatmap, normalized by true class
     -> makes the documented 2-back/3-back bottleneck legible at a glance.
  4. Confidence / entropy histograms, split by correct vs incorrect
     -> an overconfident-and-wrong model needs different fixes (calibration,
        label noise) than an unconfident one (capacity, features).
  5. Reliability (calibration) diagram + ECE
     -> quantifies whether predicted probabilities mean anything.
  6. Per-branch agreement matrix (EEG vs fNIRS vs fusion vs final)
     -> shows whether the modalities are complementary (disagree but each
        right sometimes) or redundant (always agree). Directly informs
        whether fusion work is worth continuing.
  7. EGTA attention weights over the 21 fNIRS segments, per class
     -> does the adaptive temporal alignment actually learn a non-uniform,
        physiologically plausible profile, or stay near the uniform mean?

COST CONTROL
------------
Everything here runs on a WHITELIST of subjects only (DIAGNOSTIC_SUBJECTS).
For 26 subjects x 3 folds, saving embeddings + rendering ~8 figures each is
slow and produces hundreds of files nobody reads. Five representative
subjects is enough to characterize behaviour: pick a strong one, a weak one,
the known outlier, and two mid-range ones.

Cheap aggregate diagnostics (confusion matrices, per-subject distributions)
are computed for ALL subjects from the CSV that every run already writes --
see plot_grand_kappa_distribution in training_diagnostics.py.

USAGE (see the patch in diagnostics_patch.md for exact wiring)
--------------------------------------------------------------
    from model_diagnostics import DIAGNOSTIC_SUBJECTS, run_fold_diagnostics

    if subject_id in DIAGNOSTIC_SUBJECTS:
        run_fold_diagnostics(model, train_loader, test_loader, device,
                             subject_id=subject_id, fold_idx=test_session,
                             out_dir=results_dir, unpack_batch=_unpack_batch)
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


# ---------------------------------------------------------------------------
# WHITELIST -- edit this list to control cost.
# ---------------------------------------------------------------------------
# Chosen to span the observed performance range rather than at random:
#   VP017 -- best subject (~87% acc): what "working" looks like
#   VP006 -- known outlier, negative kappa: what total failure looks like
#   VP001 -- weak but not degenerate, high fold-to-fold variance
#   VP012 -- mid-range, stable
#   VP025 -- strong but not extreme
# Set to [] to disable all per-subject diagnostics, or use "ALL" for every one.
DIAGNOSTIC_SUBJECTS = ["VP001", "VP006", "VP012", "VP017", "VP025"]

CLASS_NAMES = ["0-back", "2-back", "3-back"]


def _should_run(subject_id):
    if DIAGNOSTIC_SUBJECTS == "ALL":
        return True
    return subject_id in DIAGNOSTIC_SUBJECTS


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(model, loader, device, unpack_batch, max_batches=None):
    """
    One forward pass collecting, per window:
      - enhanced_fusion (B,512) : the fused representation the final head sees
      - enhanced_fnirs  (B,512) : the CAFE-filtered fNIRS representation
      - probabilities of all four heads
      - true label
      - egta attention weights (B,h,S) when the EGTA path is active

    Returns a dict of numpy arrays. No gradient, model left in eval mode.
    """
    was_training = model.training
    model.eval()
    fusion_emb, fnirs_emb = [], []
    p_final, p_eeg, p_fnirs_h, p_fusion_h = [], [], [], []
    labels, egta = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x_eeg, x_fnirs, y, x_conn, x_chan = unpack_batch(batch)
        x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
        x_conn = x_conn.to(device) if x_conn is not None else None
        x_chan = x_chan.to(device) if x_chan is not None else None

        y_eeg, y_fusion, y_fnirs, y_pred, aux = model(x_eeg, x_fnirs, x_conn, x_chan)

        fusion_emb.append(aux["enhanced_fusion"].detach().cpu().numpy())
        fnirs_emb.append(aux["enhanced_fnirs"].detach().cpu().numpy())
        p_final.append(y_pred.detach().cpu().numpy())
        p_eeg.append(y_eeg.detach().cpu().numpy())
        p_fnirs_h.append(y_fnirs.detach().cpu().numpy())
        p_fusion_h.append(y_fusion.detach().cpu().numpy())
        labels.append(y.numpy())
        if aux.get("egta_weights") is not None:
            egta.append(aux["egta_weights"].detach().cpu().numpy())

    if was_training:
        model.train()

    out = {
        "fusion_emb": np.concatenate(fusion_emb),
        "fnirs_emb": np.concatenate(fnirs_emb),
        "p_final": np.concatenate(p_final),
        "p_eeg": np.concatenate(p_eeg),
        "p_fnirs": np.concatenate(p_fnirs_h),
        "p_fusion": np.concatenate(p_fusion_h),
        "y_true": np.concatenate(labels),
    }
    if egta:
        out["egta_weights"] = np.concatenate(egta)
    return out


# ---------------------------------------------------------------------------
# 1 + 2. t-SNE / PCA, with the crucial train-vs-test overlay
# ---------------------------------------------------------------------------
def plot_embedding_space(train_d, test_d, subject_id, fold_idx, out_dir,
                         key="fusion_emb", method="tsne", max_points=1500):
    """
    Two panels:
      LEFT  -- test embedding coloured by TRUE CLASS. Answers: are the three
               cognitive-load levels separable in the learned space?
      RIGHT -- train (grey) vs test (coloured) in the SAME projection.
               Answers: is there a cross-session domain shift? A test cloud
               displaced from the train cloud is the visual signature of the
               exact problem E3/E4/EGTA target. This is the most informative
               single figure in this module.

    Both panels share one projection fitted on the CONCATENATION of train and
    test, otherwise the two clouds are not comparable.
    """
    from sklearn.decomposition import PCA

    Xtr, Xte = train_d[key], test_d[key]
    # Subsample for speed; t-SNE is O(n^2)-ish and 2000+ points adds nothing.
    rng = np.random.default_rng(0)
    if len(Xtr) > max_points:
        Xtr = Xtr[rng.choice(len(Xtr), max_points, replace=False)]
    idx_te = np.arange(len(Xte))
    if len(Xte) > max_points:
        idx_te = rng.choice(len(Xte), max_points, replace=False)
    Xte_s, yte_s = Xte[idx_te], test_d["y_true"][idx_te]

    X = np.concatenate([Xtr, Xte_s])
    # PCA to 50d first: standard practice, denoises and makes t-SNE stable.
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=0).fit_transform(X)

    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = max(5, min(30, (len(X) - 1) // 4))
        Z = TSNE(n_components=2, perplexity=perp, init="pca",
                 random_state=0, max_iter=600).fit_transform(X)
    else:
        Z = PCA(n_components=2, random_state=0).fit_transform(X)

    Ztr, Zte = Z[:len(Xtr)], Z[len(Xtr):]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#4C72B0", "#DD8452", "#C44E52"]

    for c in range(len(CLASS_NAMES)):
        m = yte_s == c
        axes[0].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.7,
                        color=colors[c], label=CLASS_NAMES[c])
    axes[0].set_title(f"{subject_id} fold {fold_idx} -- TEST embedding by class")
    axes[0].legend(markerscale=2)

    axes[1].scatter(Ztr[:, 0], Ztr[:, 1], s=7, alpha=0.30, color="#999999",
                    label="train (2 sessions)")
    for c in range(len(CLASS_NAMES)):
        m = yte_s == c
        axes[1].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.8,
                        color=colors[c], label=f"test {CLASS_NAMES[c]}")
    axes[1].set_title("TRAIN vs TEST -- cross-session shift")
    axes[1].legend(markerscale=2, fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method.upper()} of {key}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_{key}_{method}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Normalized confusion heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(test_d, subject_id, fold_idx, out_dir):
    """Row-normalized (recall) confusion matrix. Row-normalizing is what makes
    the 2-back/3-back confusion legible regardless of class counts."""
    y_true = test_d["y_true"]
    y_pred = test_d["p_final"].argmax(1)
    cm = np.zeros((3, 3))
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    cm_n = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_n[i, j]:.2f}\n({int(cm[i, j])})",
                    ha="center", va="center",
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
    """
    LEFT   -- max-probability histogram, split correct vs incorrect. A model
              that is confidently wrong looks very different from one that is
              merely uncertain, and needs a different fix.
    MIDDLE -- predictive entropy, same split.
    RIGHT  -- reliability diagram with ECE. NOTE: TSFNet's y_pred is a
              weighted sum of three independent sigmoid-gated heads and, by
              DEFAULT (--normalize-ypred not set), does NOT sum to 1 (the
              documented probability/logit trap, rule 9) -- so it is
              renormalized here purely for this plot; the renormalization is
              display-only. If the model being diagnosed was trained WITH
              --normalize-ypred (the priority-ypred-normfix, kept as project
              policy alongside --efgf-gate-mode sigmoid), y_pred already
              sums to ~1 and this renorm is a safe no-op (idempotent) --
              this line is correct and needs no change either way.
    """
    p = test_d["p_final"].astype(np.float64)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-12)   # safe whether or not the model
                                                          # was trained with --normalize-ypred
    y = test_d["y_true"]
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    ent = -(p * np.log(p + 1e-12)).sum(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

    axes[0].hist(conf[correct], bins=20, alpha=0.65, label="correct", color="#55A868")
    axes[0].hist(conf[~correct], bins=20, alpha=0.65, label="incorrect", color="#C44E52")
    axes[0].set_xlabel("max predicted probability"); axes[0].set_ylabel("count")
    axes[0].set_title("confidence"); axes[0].legend()

    axes[1].hist(ent[correct], bins=20, alpha=0.65, label="correct", color="#55A868")
    axes[1].hist(ent[~correct], bins=20, alpha=0.65, label="incorrect", color="#C44E52")
    axes[1].set_xlabel("predictive entropy (nats)")
    axes[1].set_title("uncertainty"); axes[1].legend()

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
# 6. Branch agreement -- are the modalities complementary or redundant?
# ---------------------------------------------------------------------------
def plot_branch_agreement(test_d, subject_id, fold_idx, out_dir):
    """
    LEFT  -- pairwise agreement rate between the four heads. High agreement
             everywhere means the branches are redundant and fusion work has
             little headroom left; low agreement with similar accuracies means
             they are complementary and better fusion could still pay off.
    RIGHT -- "oracle" analysis: how often is AT LEAST ONE branch correct while
             the final prediction is wrong? That gap is the upper bound on
             what any smarter decision-fusion rule could recover.
    """
    y = test_d["y_true"]
    preds = {k: test_d[f"p_{k}"].argmax(1) for k in ["eeg", "fnirs", "fusion", "final"]}
    names = list(preds)

    A = np.zeros((4, 4))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            A[i, j] = (preds[a] == preds[b]).mean()

    any_correct = np.zeros(len(y), dtype=bool)
    for k in ["eeg", "fnirs", "fusion"]:
        any_correct |= (preds[k] == y)
    final_correct = preds["final"] == y
    recoverable = float((any_correct & ~final_correct).mean())

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
    axes[1].bar(names + ["oracle"], accs + [oracle],
                color=["#4C72B0"] * 4 + ["#55A868"])
    axes[1].axhline(1 / 3, color="gray", linestyle="--", label="chance")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1)
    axes[1].set_title(f"branch accuracy -- recoverable by better fusion: {recoverable:.1%}")
    axes[1].legend()
    plt.setp(axes[1].get_xticklabels(), rotation=45)

    fig.suptitle(f"{subject_id} fold {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_branch_agreement.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"branch_acc": dict(zip(names, accs)),
                  "oracle_acc": oracle, "recoverable_by_fusion": recoverable}


# ---------------------------------------------------------------------------
# 7. EGTA temporal attention profile
# ---------------------------------------------------------------------------
def plot_egta_attention(test_d, subject_id, fold_idx, out_dir):
    """
    Mean EGTA attention over the 21 fNIRS segments, per true class, with the
    uniform baseline (1/21) drawn for reference. If the curves sit on the
    uniform line, EGTA learned nothing and its small gain came from elsewhere;
    if they peak in a consistent window, that window is the model's learned
    estimate of the hemodynamic response delay -- a directly reportable,
    physiologically interpretable result for the thesis.
    """
    if "egta_weights" not in test_d:
        return None
    W = test_d["egta_weights"]          # (N, heads, S)
    y = test_d["y_true"]
    S = W.shape[-1]
    Wm = W.mean(axis=1)                 # average over attention heads -> (N, S)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    colors = ["#4C72B0", "#DD8452", "#C44E52"]
    for c in range(len(CLASS_NAMES)):
        m = y == c
        if m.sum() == 0:
            continue
        prof = Wm[m].mean(0)
        sem = Wm[m].std(0) / max(np.sqrt(m.sum()), 1)
        ax.plot(range(S), prof, "-o", ms=3, color=colors[c], label=CLASS_NAMES[c])
        ax.fill_between(range(S), prof - sem, prof + sem, color=colors[c], alpha=0.18)
    ax.axhline(1 / S, color="gray", linestyle="--", label=f"uniform (1/{S})")
    ax.set_xlabel("fNIRS segment index"); ax.set_ylabel("mean attention weight")
    ax.set_title(f"{subject_id} fold {fold_idx} -- EGTA temporal profile")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_egta_attention.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_fold_diagnostics(model, train_loader, test_loader, device, subject_id,
                         fold_idx, out_dir, unpack_batch,
                         save_embeddings=True, method="tsne"):
    """
    Runs the whole battery for ONE fold, but only if subject_id is on the
    whitelist. Returns a dict of paths + scalar summaries (safe to merge into
    the fold's report), or None when skipped.

    Deliberately wrapped in a broad try/except: a plotting failure must never
    kill a multi-hour training run. Any error is printed and the run goes on.
    """
    if not _should_run(subject_id):
        return None

    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    summary = {"subject_id": subject_id, "fold_idx": int(fold_idx), "figures": []}

    try:
        print(f"  [diagnostics] extracting embeddings for {subject_id} fold {fold_idx} ...")
        test_d = extract_embeddings(model, test_loader, device, unpack_batch)
        train_d = extract_embeddings(model, train_loader, device, unpack_batch,
                                     max_batches=60)   # cap: train set is ~2x larger

        summary["figures"].append(
            plot_embedding_space(train_d, test_d, subject_id, fold_idx, diag_dir,
                                 key="fusion_emb", method=method))
        summary["figures"].append(
            plot_confusion_heatmap(test_d, subject_id, fold_idx, diag_dir))

        p, ece = plot_confidence_and_calibration(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary["ece"] = ece

        p, agree = plot_branch_agreement(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary.update(agree)

        p = plot_egta_attention(test_d, subject_id, fold_idx, diag_dir)
        if p:
            summary["figures"].append(p)
            W = test_d["egta_weights"].mean(axis=1)          # (N, S)
            S = W.shape[-1]
            y = test_d["y_true"]

            # Pooled deviation (all classes mixed together). Kept for
            # backward compatibility with anything already reading this key,
            # but NOTE: this is diluted whenever a near-uniform class (e.g.
            # 0-back, which has no sustained cognitive load and correctly
            # produces a near-flat profile) is averaged in with classes that
            # deviate strongly (2-back/3-back). Prefer the per-class
            # breakdown below for reporting.
            summary["egta_max_dev_from_uniform_pooled"] = float(
                np.abs(W.mean(0) - 1.0 / S).max())
            # Back-compat alias (old key name) -- same pooled value.
            summary["egta_max_dev_from_uniform"] = summary[
                "egta_max_dev_from_uniform_pooled"]

            # Per-class deviation: matches exactly what plot_egta_attention
            # actually draws (one profile per true class), so the scalar and
            # the figure tell the same story instead of the scalar quietly
            # underselling the class-specific signal.
            per_class_dev = {}
            for c in range(len(CLASS_NAMES)):
                m = y == c
                if m.sum() == 0:
                    continue
                prof = W[m].mean(0)
                per_class_dev[CLASS_NAMES[c]] = float(np.abs(prof - 1.0 / S).max())
            summary["egta_max_dev_from_uniform_per_class"] = per_class_dev
            if per_class_dev:
                summary["egta_max_dev_from_uniform_worst_class"] = float(
                    max(per_class_dev.values()))

        if save_embeddings:
            # Raw arrays, so any further analysis (UMAP, silhouette, MMD between
            # train and test, ...) can be done later WITHOUT re-running the GPU.
            npz = os.path.join(diag_dir, f"{subject_id}_fold{fold_idx}_embeddings.npz")
            np.savez_compressed(
                npz,
                test_fusion=test_d["fusion_emb"].astype(np.float32),
                test_fnirs=test_d["fnirs_emb"].astype(np.float32),
                test_y=test_d["y_true"],
                test_p_final=test_d["p_final"].astype(np.float32),
                train_fusion=train_d["fusion_emb"].astype(np.float32),
                train_y=train_d["y_true"])
            summary["embeddings_npz"] = npz

        with open(os.path.join(diag_dir,
                               f"{subject_id}_fold{fold_idx}_diagnostics.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [diagnostics] saved {len(summary['figures'])} figure(s) to {diag_dir}")

    except Exception as exc:          # never let a plot kill a training run
        import traceback
        print(f"  [diagnostics] WARNING: failed for {subject_id} fold {fold_idx}: {exc}")
        traceback.print_exc()
        summary["error"] = str(exc)

    return summary
