"""
mstfdn_model_diagnostics.py -- behavioural / representational diagnostics for
MSTFDN, mirroring model_diagnostics.py (the TSFNet version) so the two
architectures are diagnosed by the SAME set of plots and the same definitions.
That parity is the point: any difference you see between the two should come
from the architectures, not from one being analysed more generously.

WHAT DIFFERS FROM THE TSFNet VERSION (and why)
----------------------------------------------
  * Forward contract: MSTFDN returns (logits, he, hf, hef) from
    model(eeg, fnirs) -- two inputs, four outputs -- whereas TSFNet returns
    (y_eeg, y_fusion, y_fnirs, y_pred, aux) from
    model(x_eeg, x_fnirs, x_conn, x_chan).
  * MSTFDN's heads emit LOGITS (softmax applied in the loss), while TSFNet's
    y_* are already post-softmax probabilities. Every probability here is
    therefore obtained with an explicit softmax -- getting this wrong is the
    project's documented probability/logit trap (rule 9), and it would
    silently corrupt the confidence/ECE plots rather than crash.
  * MSTFDN has no per-branch classifier heads exposed at inference, so the
    "branch agreement / oracle" panel of the TSFNet version has no direct
    analogue. Instead, the three embeddings (he = EEG, hf = fNIRS,
    hef = fused) are compared by how separable each one is on its own,
    which answers the same underlying question -- are the modalities
    complementary or redundant -- without inventing heads that do not exist.
  * MSTFDN has no EGTA temporal-attention weights, so that panel is dropped
    rather than faked.

COST CONTROL
------------
Same whitelist discipline as the TSFNet version: per-subject plots run only
for DIAGNOSTIC_SUBJECTS. Everything is wrapped so a plotting failure can
never kill a multi-hour training run.
"""
import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Keep this list identical to model_diagnostics.py's so the two architectures
# are inspected on the SAME subjects -- spanning the observed range rather
# than chosen at random.
# ---------------------------------------------------------------------------
DIAGNOSTIC_SUBJECTS = ["VP001", "VP006", "VP012", "VP017", "VP025"]
CLASS_NAMES = ["0-back", "2-back", "3-back"]


def _should_run(subject_id):
    if DIAGNOSTIC_SUBJECTS == "ALL":
        return True
    return subject_id in DIAGNOSTIC_SUBJECTS


# ---------------------------------------------------------------------------
# Embedding / probability extraction
# ---------------------------------------------------------------------------
def _heads_to_embedding(heads):
    """MSTFDN returns each SFEM's last-layer MHSA output as a LIST of per-head
    tensors (B, CH, d_k). Average over the CH tokens and concatenate heads ->
    (B, H*d_k) = (B, d_spatial), one vector per window."""
    if torch.is_tensor(heads):
        return heads.flatten(1)
    return torch.cat([h.mean(dim=1) if h.dim() == 3 else h.flatten(1)
                      for h in heads], dim=1)


@torch.no_grad()
def extract_embeddings(model, loader, device, max_batches=None):
    """One forward pass collecting, per window:
      - he  : EEG-branch embedding
      - hf  : fNIRS-branch embedding
      - hef : fused embedding (what the classifier actually sees)
      - probabilities (softmax of logits -- MSTFDN emits LOGITS, not probs)
      - true label
    """
    was_training = model.training
    model.eval()
    he_all, hf_all, hef_all, prob_all, y_all = [], [], [], [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        eeg, fnirs, labels = batch
        eeg, fnirs = eeg.to(device), fnirs.to(device)
        logits, he, hf, hef = model(eeg, fnirs)
        # EXPLICIT softmax: MSTFDN's head returns logits (see MSTFDNLoss,
        # which applies cross-entropy). Treating them as probabilities would
        # silently produce meaningless confidence/ECE numbers.
        probs = F.softmax(logits, dim=-1)

        he_all.append(_heads_to_embedding(he).detach().float().cpu().numpy())
        hf_all.append(_heads_to_embedding(hf).detach().float().cpu().numpy())
        hef_all.append(_heads_to_embedding(hef).detach().float().cpu().numpy())
        prob_all.append(probs.detach().float().cpu().numpy())
        y_all.append(labels.numpy())

    if was_training:
        model.train()

    return {
        "he": np.concatenate(he_all),
        "hf": np.concatenate(hf_all),
        "hef": np.concatenate(hef_all),
        "probs": np.concatenate(prob_all),
        "y_true": np.concatenate(y_all),
    }


# ---------------------------------------------------------------------------
# 1 + 2. t-SNE of the fused embedding, and the train-vs-test overlay
# ---------------------------------------------------------------------------
def plot_embedding_space(train_d, test_d, subject_id, fold_idx, out_dir,
                         key="hef", method="tsne", max_points=1500):
    """
    LEFT  -- test embedding coloured by TRUE CLASS: are the three cognitive
             load levels separable in MSTFDN's learned space at all?
    RIGHT -- train (grey) vs test (coloured) in the SAME projection: the
             visual signature of cross-session shift, the problem this whole
             project is about. A displaced test cloud means distribution
             shift, not poor class separability.
    Both panels share one projection fitted on train+test together; fitting
    them separately would make the two clouds incomparable by construction.
    """
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
    n_pc = min(50, X.shape[0], X.shape[1])
    if X.shape[1] > n_pc:
        X = PCA(n_components=n_pc, random_state=0).fit_transform(X)

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
    axes[0].set_title(f"{subject_id} fold {fold_idx} -- MSTFDN TEST embedding by class")
    axes[0].legend(markerscale=2)

    axes[1].scatter(Ztr[:, 0], Ztr[:, 1], s=7, alpha=0.30, color="#999999",
                    label="train")
    for c in range(len(CLASS_NAMES)):
        m = yte_s == c
        axes[1].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.8,
                        color=colors[c], label=f"test {CLASS_NAMES[c]}")
    axes[1].set_title("TRAIN vs TEST -- cross-session shift")
    axes[1].legend(markerscale=2, fontsize=8)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method.upper()} of MSTFDN {key}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_{key}_{method}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Recall-normalized confusion heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(test_d, subject_id, fold_idx, out_dir):
    """Row-normalized (recall) confusion matrix -- row-normalizing is what
    makes the documented 2-back/3-back bottleneck legible regardless of how
    many windows each class happens to contribute."""
    y_true = test_d["y_true"]
    y_pred = test_d["probs"].argmax(1)
    k = len(CLASS_NAMES)
    cm = np.zeros((k, k))
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    cm_n = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(5, 4.3))
    im = ax.imshow(cm_n, cmap="Blues", vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{cm_n[i, j]:.2f}\n({int(cm[i, j])})",
                    ha="center", va="center",
                    color="white" if cm_n[i, j] > 0.5 else "black", fontsize=9)
    ax.set_xticks(range(k)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(k)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{subject_id} fold {fold_idx} -- MSTFDN, recall-normalized")
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
    LEFT   -- max-probability histogram, correct vs incorrect. A confidently
              wrong model needs a different fix than a merely uncertain one.
    MIDDLE -- predictive entropy, same split.
    RIGHT  -- reliability diagram + ECE.
    NOTE: probs here are already a proper softmax of MSTFDN's logits (see
    extract_embeddings), so unlike the TSFNet version no renormalization
    workaround is needed -- MSTFDN's output is a valid distribution by
    construction, which is itself a noteworthy architectural difference.
    """
    p = test_d["probs"].astype(np.float64)
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
        a, cf = correct[m].mean(), conf[m].mean()
        accs.append(a); confs.append(cf)
        ece += (m.sum() / len(conf)) * abs(a - cf)
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes[2].plot(confs, accs, "o-", color="#4C72B0", label="model")
    axes[2].set_xlabel("confidence"); axes[2].set_ylabel("accuracy")
    axes[2].set_title(f"reliability (ECE={ece:.3f})"); axes[2].legend()
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)

    fig.suptitle(f"{subject_id} fold {fold_idx} -- MSTFDN", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_confidence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(ece)


# ---------------------------------------------------------------------------
# 6. Per-modality separability -- MSTFDN's analogue of TSFNet's branch-agreement
# ---------------------------------------------------------------------------
def plot_modality_separability(test_d, subject_id, fold_idx, out_dir):
    """
    MSTFDN exposes no per-branch classifier heads at inference, so TSFNet's
    "which branch votes what" analysis has no direct analogue. The underlying
    question -- are EEG and fNIRS complementary or redundant? -- is answered
    here instead by asking how separable each EMBEDDING is on its own:

    LEFT  -- silhouette score of each embedding w.r.t. the true labels.
             Higher = that modality alone already groups the classes.
    RIGHT -- accuracy of a simple linear probe (logistic regression, fit and
             scored on the same test fold) on each embedding. This is NOT a
             generalization estimate -- it is a measure of how linearly
             decodable the class is from each representation, which is
             exactly what "is this modality carrying the signal" means here.
             Labelled as such on the plot so it is never misread as a test
             accuracy.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    y = test_d["y_true"]
    names = [("he", "EEG"), ("hf", "fNIRS"), ("hef", "fused")]
    sils, probes = [], []
    rng = np.random.default_rng(0)
    for key, _ in names:
        X = test_d[key]
        if len(X) > 2000:                      # silhouette is O(n^2)
            sub = rng.choice(len(X), 2000, replace=False)
            Xs, ys = X[sub], y[sub]
        else:
            Xs, ys = X, y
        try:
            sils.append(float(silhouette_score(Xs, ys)))
        except Exception:
            sils.append(float("nan"))
        # NOTE: no multi_class= argument -- it was removed in scikit-learn 1.7+
        # and sklearn infers multinomial automatically from the label count.
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=300))
        clf.fit(Xs, ys)
        probes.append(float((clf.predict(Xs) == ys).mean()))

    labels = [lbl for _, lbl in names]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    axes[0].bar(labels, sils, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("silhouette score")
    axes[0].set_title("class structure within each embedding")
    axes[1].bar(labels, probes, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].axhline(1 / len(CLASS_NAMES), color="gray", ls="--", label="chance")
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("linear-probe accuracy (in-sample)")
    axes[1].set_title("linear decodability (NOT a generalization estimate)")
    axes[1].legend()
    fig.suptitle(f"{subject_id} fold {fold_idx} -- MSTFDN per-modality separability",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_fold{fold_idx}_modality_separability.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"silhouette": dict(zip(labels, sils)),
                  "linear_probe_insample": dict(zip(labels, probes))}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_fold_diagnostics(model, train_loader, test_loader, device, subject_id,
                         fold_idx, out_dir, save_embeddings=True, method="tsne"):
    """Runs the whole battery for ONE fold, only for whitelisted subjects.
    Returns a dict of paths + scalar summaries (safe to merge into the fold's
    result record), or None when skipped.

    Deliberately wrapped in a broad try/except: a plotting failure must never
    kill a multi-hour training run.
    """
    if not _should_run(subject_id):
        return None

    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    summary = {"subject_id": subject_id, "fold_idx": int(fold_idx),
               "architecture": "MSTFDN", "figures": []}

    try:
        print(f"  [diagnostics] extracting MSTFDN embeddings for {subject_id} "
              f"fold {fold_idx} ...")
        test_d = extract_embeddings(model, test_loader, device)
        train_d = extract_embeddings(model, train_loader, device, max_batches=60)

        summary["figures"].append(
            plot_embedding_space(train_d, test_d, subject_id, fold_idx, diag_dir,
                                 key="hef", method=method))
        summary["figures"].append(
            plot_confusion_heatmap(test_d, subject_id, fold_idx, diag_dir))

        p, ece = plot_confidence_and_calibration(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary["ece"] = ece

        p, sep = plot_modality_separability(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary.update(sep)

        if save_embeddings:
            # Raw arrays, so any further analysis (UMAP, MMD between train and
            # test, cross-architecture comparison against the TSFNet .npz files)
            # can be done later WITHOUT re-running the GPU.
            npz = os.path.join(diag_dir, f"{subject_id}_fold{fold_idx}_embeddings.npz")
            np.savez_compressed(
                npz,
                test_hef=test_d["hef"].astype(np.float32),
                test_he=test_d["he"].astype(np.float32),
                test_hf=test_d["hf"].astype(np.float32),
                test_y=test_d["y_true"],
                test_probs=test_d["probs"].astype(np.float32),
                train_hef=train_d["hef"].astype(np.float32),
                train_y=train_d["y_true"])
            summary["embeddings_npz"] = npz

        with open(os.path.join(diag_dir,
                               f"{subject_id}_fold{fold_idx}_diagnostics.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [diagnostics] saved {len(summary['figures'])} figure(s) to {diag_dir}")

    except Exception as exc:
        import traceback
        print(f"  [diagnostics] WARNING: failed for {subject_id} fold {fold_idx}: {exc}")
        traceback.print_exc()
        summary["error"] = str(exc)

    return summary
