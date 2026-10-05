"""
mgfnet_diagnostics.py -- behavioural / representational diagnostics for MGFNet
(the MGFNet counterpart of the TSFNet project's model_diagnostics.py).

WHAT IS SAVED PER (subject, held-out session)
  1. t-SNE / PCA of the fused representation (the classifier's input), test
     coloured by class + TRAIN vs TEST overlay -> cross-session shift.
  2. Recall-normalised confusion heatmap.
  3. Confidence / entropy histograms + reliability diagram with ECE.
     (MGFNet heads return LOGITS, so softmax gives proper probabilities --
     no renormalisation trick needed, unlike TSFNet's y_pred.)
  4. Branch agreement over the 6 deep-supervision heads + the hybrid head,
     with the "oracle" upper bound for a better fusion rule.
  5. CGSCR -- the module the paper's central claim rests on:
       a. branch gate weights per class + per-sample spread (is the gate
          really sample-specific, or collapsed to a constant?)
       b. mean coupling matrix C (5 EEG bands x 3 fNIRS scales) per class
       c. which of the 15 cross-modal tokens the sparse router selects, per
          class, with the selection entropy (log2(15)=3.91 bits = uniform,
          i.e. the router learned nothing)
       d. learned Morlet centre frequencies / bandwidths vs. their init
  6. Raw arrays (.npz) so any later analysis needs no GPU re-run.

COST CONTROL: only subjects in DIAGNOSTIC_SUBJECTS are processed.

WIRING: mgfnet_train.run_subject_loso_mgfnet calls run_fold_diagnostics_mgfnet
after each final model, if this file is importable. A failure here is printed
and never stops a training run.
"""
import json
import os
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from mgfnet_dataset import _unpack_mgf_batch
from mgfnet_model import EEG_F0_INIT, EEG_BW_INIT, NIRS_F0_INIT, NIRS_BW_INIT

# ---------------------------------------------------------------------------
# WHITELIST -- edit to control cost. "ALL" = every subject, [] = disabled.
# Pick a strong, a weak and some mid-range subjects AFTER a first sweep.
# ---------------------------------------------------------------------------
DIAGNOSTIC_SUBJECTS = ["VP001", "VP006", "VP012", "VP017", "VP025"]

CLASS_NAMES = ["0-back", "2-back", "3-back"]
CLASS_COLORS = ["#4C72B0", "#DD8452", "#C44E52"]
BRANCH_KEYS = ("1", "2", "3", "12", "13", "23")
BRANCH_NAMES = {"1": "EEG", "2": "HbO", "3": "HbR",
                "12": "EEG-HbO", "13": "EEG-HbR", "23": "HbO-HbR"}
EEG_BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
NIRS_SCALE_NAMES = ["slow", "mid", "fast"]


def should_run(subject_id):
    if DIAGNOSTIC_SUBJECTS == "ALL":
        return True
    return subject_id in DIAGNOSTIC_SUBJECTS


def _softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_outputs(model, loader, device, norm, max_batches=None):
    """One eval-mode pass collecting, per window:
         fused (B, D)        classifier input (gated branches + x_scr)
         feat_<k> (B, d_k)   each branch's feature vector
         logits_hyb (B, 3), logits_<k> (B, 3)
         gate (B, 6), coupling (B, 5, 3), selection (B, 15), scores (B, 15)
         y_true (B,)
    """
    was_training = model.training
    model.eval()
    store = {"fused": [], "y_true": [], "logits_hyb": []}
    for k in model.active_keys:
        store[f"feat_{k}"] = []
        store[f"logits_{k}"] = []
    for k in ("gate", "coupling", "selection", "scores"):
        store[k] = []

    captured = {}
    hooks = [model.classifier[1].register_forward_pre_hook(
        lambda m, inp: captured.__setitem__("fused", inp[0]))]
    for k in model.active_keys:
        hooks.append(model.aux_heads[k].register_forward_pre_hook(
            (lambda key: lambda m, inp: captured.__setitem__(key, inp[0]))(f"feat_{k}")))

    try:
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
            x_eeg, x_hbo, x_hbr = norm(x_eeg.to(device), x_hbo.to(device), x_hbr.to(device))
            y_hyb, y_aux, aux = model(x_eeg, x_hbo, x_hbr)
            store["fused"].append(captured["fused"].cpu().numpy())
            store["y_true"].append(y.numpy())
            store["logits_hyb"].append(y_hyb.cpu().numpy())
            for k in model.active_keys:
                store[f"feat_{k}"].append(captured[f"feat_{k}"].cpu().numpy())
                store[f"logits_{k}"].append(y_aux[k].cpu().numpy())
            if aux["gate_weights"] is not None:
                store["gate"].append(aux["gate_weights"].cpu().numpy())
                store["coupling"].append(aux["coupling"].cpu().numpy())
                store["selection"].append(aux["component_selection"].cpu().numpy())
                store["scores"].append(aux["importance_scores"].cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return {k: np.concatenate(v) for k, v in store.items() if v}


# ---------------------------------------------------------------------------
# 1. Embedding space with train-vs-test overlay
# ---------------------------------------------------------------------------
def plot_embedding_space(train_d, test_d, subject_id, fold_idx, out_dir,
                         key="fused", method="tsne", max_points=1500):
    """LEFT: test embedding by class. RIGHT: train (grey) vs test (coloured)
    in ONE projection fitted on both -- a displaced test cloud is the visual
    signature of cross-session shift."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    Xtr, Xte, yte = train_d[key], test_d[key], test_d["y_true"]
    if len(Xtr) > max_points:
        Xtr = Xtr[rng.choice(len(Xtr), max_points, replace=False)]
    if len(Xte) > max_points:
        idx = rng.choice(len(Xte), max_points, replace=False)
        Xte, yte = Xte[idx], yte[idx]

    X = StandardScaler().fit_transform(np.concatenate([Xtr, Xte]))
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=0).fit_transform(X)
    if method == "tsne":
        from sklearn.manifold import TSNE
        perp = max(5, min(30, (len(X) - 1) // 4))
        Z = TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)
    else:
        Z = PCA(n_components=2, random_state=0).fit_transform(X)
    Ztr, Zte = Z[:len(Xtr)], Z[len(Xtr):]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for c, name in enumerate(CLASS_NAMES):
        m = yte == c
        axes[0].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.7, color=CLASS_COLORS[c], label=name)
    axes[0].set_title(f"{subject_id} session {fold_idx} -- TEST embedding by class")
    axes[0].legend(markerscale=2)
    axes[1].scatter(Ztr[:, 0], Ztr[:, 1], s=7, alpha=0.3, color="#999999", label="train (2 sessions)")
    for c, name in enumerate(CLASS_NAMES):
        m = yte == c
        axes[1].scatter(Zte[m, 0], Zte[m, 1], s=9, alpha=0.8, color=CLASS_COLORS[c],
                        label=f"test {name}")
    axes[1].set_title("TRAIN vs TEST -- cross-session shift")
    axes[1].legend(markerscale=2, fontsize=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{method.upper()} of MGFNet {key} representation", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_{key}_{method}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 2. Confusion heatmap
# ---------------------------------------------------------------------------
def plot_confusion_heatmap(test_d, subject_id, fold_idx, out_dir):
    y_true = test_d["y_true"]
    y_pred = test_d["logits_hyb"].argmax(1)
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
    ax.set_title(f"{subject_id} session {fold_idx} -- recall-normalised")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_confusion.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# 3. Confidence and calibration
# ---------------------------------------------------------------------------
def plot_confidence_and_calibration(test_d, subject_id, fold_idx, out_dir, n_bins=10):
    p = _softmax(test_d["logits_hyb"].astype(np.float64))
    y = test_d["y_true"]
    conf = p.max(1)
    correct = p.argmax(1) == y
    ent = -(p * np.log(p + 1e-12)).sum(1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for ax, vals, xlabel, title in ((axes[0], conf, "max predicted probability", "confidence"),
                                    (axes[1], ent, "predictive entropy (nats)", "uncertainty")):
        ax.hist(vals[correct], bins=20, alpha=0.65, label="correct", color="#55A868")
        ax.hist(vals[~correct], bins=20, alpha=0.65, label="incorrect", color="#C44E52")
        ax.set_xlabel(xlabel); ax.set_title(title); ax.legend()
    axes[0].set_ylabel("count")

    bins = np.linspace(0, 1, n_bins + 1)
    accs, confs, ece = [], [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if not m.any():
            continue
        a, c = correct[m].mean(), conf[m].mean()
        accs.append(a); confs.append(c)
        ece += m.mean() * abs(a - c)
    axes[2].plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    axes[2].plot(confs, accs, "o-", color="#4C72B0", label="model")
    axes[2].set_xlabel("confidence"); axes[2].set_ylabel("accuracy")
    axes[2].set_title(f"reliability (ECE={ece:.3f})"); axes[2].legend()
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    fig.suptitle(f"{subject_id} session {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_confidence.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, float(ece)


# ---------------------------------------------------------------------------
# 4. Branch agreement over the deep-supervision heads
# ---------------------------------------------------------------------------
def plot_branch_agreement(test_d, subject_id, fold_idx, out_dir, branch_keys):
    """LEFT: pairwise agreement of the 6 branch heads + hybrid head.
    RIGHT: per-head accuracy + oracle (>=1 branch right). 'Recoverable' =
    oracle-right but hybrid-wrong: the head-room for a better fusion rule."""
    y = test_d["y_true"]
    preds = {BRANCH_NAMES[k]: test_d[f"logits_{k}"].argmax(1) for k in branch_keys}
    preds["hybrid"] = test_d["logits_hyb"].argmax(1)
    names = list(preds)
    n = len(names)
    A = np.array([[(preds[a] == preds[b]).mean() for b in names] for a in names])

    any_correct = np.zeros(len(y), dtype=bool)
    for k in branch_keys:
        any_correct |= preds[BRANCH_NAMES[k]] == y
    hyb_correct = preds["hybrid"] == y
    recoverable = float((any_correct & ~hyb_correct).mean())
    accs = [float((preds[k] == y).mean()) for k in names]
    oracle = float(any_correct.mean())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    im = axes[0].imshow(A, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            axes[0].text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                         color="white" if A[i, j] < 0.6 else "black", fontsize=8)
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_yticks(range(n)); axes[0].set_yticklabels(names)
    axes[0].set_title("pairwise prediction agreement")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[1].bar(names + ["oracle"], accs + [oracle], color=["#4C72B0"] * n + ["#55A868"])
    axes[1].axhline(1 / 3, color="gray", linestyle="--", label="chance")
    axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0, 1); axes[1].legend()
    axes[1].set_title(f"head accuracy -- recoverable by better fusion: {recoverable:.1%}")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    fig.suptitle(f"{subject_id} session {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_branch_agreement.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"head_acc": dict(zip(names, accs)), "oracle_acc": oracle,
                  "recoverable_by_fusion": recoverable}


# ---------------------------------------------------------------------------
# 5. CGSCR
# ---------------------------------------------------------------------------
def token_labels(model=None):
    """Names of the 15 cross-modal tokens (token n = EEG comp n//3, fNIRS
    comp n%3), using the LEARNED centre frequencies when a model is given."""
    if model is not None and getattr(model, "use_cgscr", False):
        fe = model.cgscr.eeg_bank.f0.detach().cpu().numpy()
        fn = model.cgscr.hbo_bank.f0.detach().cpu().numpy()
        return [f"{EEG_BAND_NAMES[i]}({fe[i]:.1f}Hz)x{NIRS_SCALE_NAMES[j]}({fn[j]:.2f}Hz)"
                for i in range(5) for j in range(3)]
    return [f"{EEG_BAND_NAMES[i]}x{NIRS_SCALE_NAMES[j]}" for i in range(5) for j in range(3)]


def plot_gate_weights(test_d, subject_id, fold_idx, out_dir):
    """LEFT: mean gate weight per branch and class (uniform = 1/6).
    RIGHT: per-sample distribution -- a collapsed gate shows zero spread."""
    G, y = test_d["gate"], test_d["y_true"]
    names = [BRANCH_NAMES[k] for k in BRANCH_KEYS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.27
    for c, cname in enumerate(CLASS_NAMES):
        m = y == c
        if m.any():
            axes[0].bar(np.arange(6) + (c - 1) * w, G[m].mean(0), w,
                        color=CLASS_COLORS[c], label=cname)
    axes[0].axhline(1 / 6, color="gray", ls="--", label="uniform (1/6)")
    axes[0].set_xticks(range(6)); axes[0].set_xticklabels(names, rotation=30)
    axes[0].set_ylabel("mean gate weight"); axes[0].legend(fontsize=8)
    axes[0].set_title("CGSCR branch gate per class")
    axes[1].boxplot([G[:, i] for i in range(6)], showfliers=False)
    axes[1].set_xticks(range(1, 7)); axes[1].set_xticklabels(names, rotation=30)
    cv = G.std(0) / np.maximum(G.mean(0), 1e-8)
    axes[1].set_title(f"per-sample spread (max CV={cv.max():.3f}; <0.01 = collapsed)")
    fig.suptitle(f"{subject_id} session {fold_idx}", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_cgscr_gate.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"gate_mean": dict(zip(names, G.mean(0).round(4).tolist())),
                  "gate_max_cv": float(cv.max())}


def plot_coupling_matrices(test_d, subject_id, fold_idx, out_dir):
    """Mean coupling matrix C (rows: EEG bands, cols: fNIRS scales; each row
    sums to 1) per class, plus the class-difference map (3-back - 0-back)."""
    C, y = test_d["coupling"], test_d["y_true"]
    mats = [C[y == c].mean(0) if (y == c).any() else np.full((5, 3), np.nan) for c in range(3)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for c in range(3):
        im = axes[c].imshow(mats[c], cmap="magma", vmin=0, vmax=1)
        axes[c].set_title(CLASS_NAMES[c])
        for i in range(5):
            for j in range(3):
                axes[c].text(j, i, f"{mats[c][i, j]:.2f}", ha="center", va="center",
                             color="white" if mats[c][i, j] < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    diff = mats[2] - mats[0]
    lim = max(np.nanmax(np.abs(diff)), 1e-6)
    im = axes[3].imshow(diff, cmap="RdBu_r", vmin=-lim, vmax=lim)
    axes[3].set_title("3-back minus 0-back")
    fig.colorbar(im, ax=axes[3], fraction=0.046)
    for ax in axes:
        ax.set_xticks(range(3)); ax.set_xticklabels(NIRS_SCALE_NAMES)
        ax.set_yticks(range(5)); ax.set_yticklabels(EEG_BAND_NAMES)
    fig.suptitle(f"{subject_id} session {fold_idx} -- CGSCR coupling C (EEG band -> fNIRS scale)",
                 fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_cgscr_coupling.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"coupling_mean": C.mean(0).round(4).tolist(),
                  "coupling_row_entropy_bits": float(
                      (-(C * np.log2(C + 1e-12)).sum(-1)).mean())}


def plot_token_selection(test_d, subject_id, fold_idx, out_dir, model=None):
    """How often each of the 15 cross-modal tokens is routed, per class.
    Entropy of the selection histogram: 3.91 bits = uniform (router not
    discriminating), 0 = always the same token."""
    S, y = test_d["selection"], test_d["y_true"]
    labels = token_labels(model)
    hist_all = S.mean(0)
    nz = hist_all[hist_all > 0]
    ent = float(-(nz * np.log2(nz)).sum())
    fig, ax = plt.subplots(figsize=(14, 4.8))
    w = 0.27
    for c, cname in enumerate(CLASS_NAMES):
        m = y == c
        if m.any():
            ax.bar(np.arange(15) + (c - 1) * w, S[m].mean(0), w, color=CLASS_COLORS[c], label=cname)
    ax.axhline(1 / 15, color="gray", ls="--", label="uniform (1/15)")
    ax.set_xticks(range(15)); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("selection frequency")
    ax.set_title(f"{subject_id} session {fold_idx} -- sparse router token selection "
                 f"(entropy {ent:.2f} bits; uniform = {np.log2(15):.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_cgscr_selection.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, {"selection_hist": hist_all.round(4).tolist(),
                  "selection_entropy_bits": ent,
                  "most_selected_token": labels[int(hist_all.argmax())]}


def plot_morlet_learned(model, subject_id, fold_idx, out_dir):
    """Learned centre frequency +- bandwidth of every Morlet component vs
    its initial value. Large moves = the data pulled the band elsewhere."""
    c = model.cgscr
    banks = [("EEG", c.eeg_bank, EEG_F0_INIT, EEG_BW_INIT, EEG_BAND_NAMES),
             ("HbO", c.hbo_bank, NIRS_F0_INIT, NIRS_BW_INIT, NIRS_SCALE_NAMES),
             ("HbR", c.hbr_bank, NIRS_F0_INIT, NIRS_BW_INIT, NIRS_SCALE_NAMES)]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    out = {}
    for ax, (name, bank, f0i, bwi, names) in zip(axes, banks):
        f0 = bank.f0.detach().cpu().numpy()
        bw = bank.bw.detach().cpu().numpy()
        k = np.arange(len(f0))
        ax.errorbar(k - 0.1, f0i, yerr=bwi, fmt="o", color="#999999", capsize=4, label="init")
        ax.errorbar(k + 0.1, f0, yerr=bw, fmt="o", color="#C44E52", capsize=4, label="learned")
        ax.set_xticks(k); ax.set_xticklabels(names)
        ax.set_ylabel("Hz (centre +- bandwidth)")
        ax.set_title(f"{name} Morlet bank")
        ax.legend(fontsize=8)
        out[name] = {"f0": f0.round(4).tolist(), "bw": bw.round(4).tolist()}
    fig.suptitle(f"{subject_id} session {fold_idx} -- learned Morlet components", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{subject_id}_S{fold_idx}_morlet_learned.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_fold_diagnostics_mgfnet(model, train_loader, test_loader, device, norm,
                                subject_id, fold_idx, out_dir, method="tsne",
                                save_arrays=True, force=False):
    """Whole battery for ONE held-out session (if subject is whitelisted, or
    force=True). Returns a JSON-safe summary dict, or None when skipped.
    Never raises: a plotting failure must not kill a training sweep."""
    if not (force or should_run(subject_id)):
        return None
    diag_dir = os.path.join(out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)
    summary = {"subject_id": subject_id, "test_session": int(fold_idx), "figures": []}
    try:
        print(f"  [diagnostics] {subject_id} session {fold_idx} ...")
        test_d = extract_outputs(model, test_loader, device, norm)
        train_d = extract_outputs(model, train_loader, device, norm, max_batches=30)

        summary["figures"].append(plot_embedding_space(train_d, test_d, subject_id, fold_idx,
                                                       diag_dir, method=method))
        summary["figures"].append(plot_confusion_heatmap(test_d, subject_id, fold_idx, diag_dir))
        p, ece = plot_confidence_and_calibration(test_d, subject_id, fold_idx, diag_dir)
        summary["figures"].append(p); summary["ece"] = ece
        p, agree = plot_branch_agreement(test_d, subject_id, fold_idx, diag_dir, model.active_keys)
        summary["figures"].append(p); summary.update(agree)

        if getattr(model, "use_cgscr", False):
            for fn in (plot_gate_weights, plot_coupling_matrices):
                p, info = fn(test_d, subject_id, fold_idx, diag_dir)
                summary["figures"].append(p); summary.update(info)
            p, info = plot_token_selection(test_d, subject_id, fold_idx, diag_dir, model)
            summary["figures"].append(p); summary.update(info)
            p, info = plot_morlet_learned(model, subject_id, fold_idx, diag_dir)
            summary["figures"].append(p); summary["morlet"] = info

        if save_arrays:
            npz = os.path.join(diag_dir, f"{subject_id}_S{fold_idx}_outputs.npz")
            np.savez_compressed(
                npz, **{f"test_{k}": v.astype(np.float32) if v.dtype.kind == "f" else v
                        for k, v in test_d.items()},
                train_fused=train_d["fused"].astype(np.float32), train_y=train_d["y_true"])
            summary["arrays_npz"] = npz

        with open(os.path.join(diag_dir, f"{subject_id}_S{fold_idx}_diagnostics.json"),
                  "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [diagnostics] saved {len(summary['figures'])} figure(s) to {diag_dir}")
    except Exception as exc:
        print(f"  [diagnostics] WARNING: failed for {subject_id} session {fold_idx}: {exc}")
        traceback.print_exc()
        summary["error"] = str(exc)
    return summary
