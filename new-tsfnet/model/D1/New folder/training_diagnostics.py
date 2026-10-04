"""
training_diagnostics.py -- post-hoc visual analysis of train_one_fold's
`history` dict (already collected by the project; this module only reads
and plots it -- zero effect on training itself).

USAGE (after a run, per subject, per fold -- `result` is whatever
run_subject_cross_session / train_one_fold returns per fold):

    from training_diagnostics import (
        plot_fold_curves, plot_subject_fold_overlay,
        plot_kappa_oscillation, plot_gate_evolution,
        plot_grand_kappa_distribution,
    )

    plot_fold_curves(fold_result["history"], subject_id="VP006", fold_idx=0,
                      save_dir=RESULTS_DIR)

Every function saves a PNG (matplotlib, non-interactive backend) and also
returns the Figure, so it works fine inside or outside Jupyter.
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless-safe; works in Jupyter too
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------
# 1. Per-fold train/val loss + val acc/kappa curves (what you asked for).
# ----------------------------------------------------------------------------
def plot_fold_curves(history, subject_id, fold_idx, save_dir=None, show_stage_boundary=True):
    """
    Two-panel figure for ONE fold:
      left:  train_loss vs val_loss over epochs
      right: val_acc vs val_kappa over epochs
    If history["stage"] is present (see train_one_fold_diagnostics_patch.md),
    a vertical line marks the stage-1 -> stage-2 transition, since epoch
    numbering restarts at 0 for stage 2 in this project's training loop --
    without this marker the x-axis can look like a single, misleadingly
    continuous curve.
    """
    epochs = np.arange(len(history["train_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="train_loss", color="#4C72B0")
    axes[0].plot(epochs, history["val_loss"], label="val_loss", color="#DD8452")
    axes[0].set_xlabel("epoch (concatenated across stages)")
    axes[0].set_ylabel("loss")
    axes[0].set_title(f"{subject_id} fold {fold_idx} -- loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_acc"], label="val_acc", color="#55A868")
    axes[1].plot(epochs, history["val_kappa"], label="val_kappa", color="#C44E52")
    axes[1].set_xlabel("epoch (concatenated across stages)")
    axes[1].set_ylabel("value")
    axes[1].set_title(f"{subject_id} fold {fold_idx} -- val acc / kappa")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    if show_stage_boundary and "stage" in history and history["stage"]:
        stages = np.array(history["stage"])
        boundary = np.where(np.diff(stages) != 0)[0]
        for b in boundary:
            for ax in axes:
                ax.axvline(b + 0.5, color="gray", linestyle="--", alpha=0.6,
                           label="stage1->stage2" if b == boundary[0] else None)
        if len(boundary):
            axes[0].legend()

    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"{subject_id}_fold{fold_idx}_curves.png")
        fig.savefig(path, dpi=130)
        print(f"saved: {path}")
    return fig


# ----------------------------------------------------------------------------
# 2. Overlay val_kappa across a subject's 3 folds -- fold-to-fold consistency.
# ----------------------------------------------------------------------------
def plot_subject_fold_overlay(fold_histories, subject_id, save_dir=None):
    """
    fold_histories: list of history dicts, one per fold (typically 3, one
    per held-out session). Overlays val_kappa curves so you can see whether
    a subject's folds behave similarly (good sign) or wildly differently
    (a subject-specific instability worth flagging, e.g. VP006).
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(fold_histories)))
    for i, h in enumerate(fold_histories):
        ax.plot(h["val_kappa"], label=f"fold {i}", color=colors[i], alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("epoch (concatenated across stages)")
    ax.set_ylabel("val_kappa")
    ax.set_title(f"{subject_id} -- val_kappa across folds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"{subject_id}_fold_overlay_kappa.png")
        fig.savefig(path, dpi=130)
        print(f"saved: {path}")
    return fig


# ----------------------------------------------------------------------------
# 3. Oscillation diagnostic -- directly targets the "TENT val_kappa jumps
#    around a lot between epochs" observation.
# ----------------------------------------------------------------------------
def plot_kappa_oscillation(history, subject_id, fold_idx, save_dir=None, window=5):
    """
    Plots |val_kappa[e] - val_kappa[e-1]| per epoch, plus a rolling std-dev
    band. A run where this stays small and flat is stable; a run with large,
    persistent spikes (not just early-training noise that settles down) is
    the pattern worth flagging as "possibly a different problem" -- e.g. a
    batch-size-driven BatchNorm issue (rule 1) rather than the intervention
    itself, exactly the kind of thing raised about the TENT logs.
    """
    kappa = np.array(history["val_kappa"])
    if len(kappa) < 2:
        raise ValueError("need at least 2 epochs to compute oscillation")
    deltas = np.abs(np.diff(kappa))
    roll_std = np.array([
        deltas[max(0, i - window):i + 1].std() for i in range(len(deltas))
    ])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(deltas, label="|Δ val_kappa| per epoch", color="#8172B2", alpha=0.8)
    ax.plot(roll_std, label=f"rolling std (window={window})", color="#C44E52",
            linestyle="--")
    ax.set_xlabel("epoch transition")
    ax.set_ylabel("|Δ val_kappa|")
    ax.set_title(f"{subject_id} fold {fold_idx} -- epoch-to-epoch kappa oscillation")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"{subject_id}_fold{fold_idx}_oscillation.png")
        fig.savefig(path, dpi=130)
        print(f"saved: {path}")

    # A simple scalar summary worth printing/logging alongside the figure:
    late_std = deltas[len(deltas) // 2:].std()  # oscillation in training's second half
    early_std = deltas[:len(deltas) // 2].std()
    print(f"  {subject_id} fold {fold_idx}: early-half |Δkappa| std={early_std:.4f}, "
          f"late-half std={late_std:.4f} "
          f"({'still unstable late in training' if late_std > 0.5 * early_std else 'settles down as expected'})")
    return fig


# ----------------------------------------------------------------------------
# 4. Gate evolution -- did the model actually open a zero-init gate, or did
#    it stay near-zero the whole time? Directly answers "did the safe-
#    capacity branch (EGTA/LaBraM/D-1-safe) get used at all".
# ----------------------------------------------------------------------------
def plot_gate_evolution(history, subject_id, fold_idx, save_dir=None):
    """
    Requires history["gates"] (see train_one_fold_diagnostics_patch.md).
    Plots every gate name found across epochs on one figure. A gate that
    stays flat near its init value the whole run means the model chose NOT
    to use that capacity (a valid, informative outcome -- not a bug); a
    gate that rises steadily means it's being actively adopted.
    """
    if "gates" not in history or not history["gates"]:
        print(f"  {subject_id} fold {fold_idx}: no gate history recorded (patch not applied, "
              f"or this configuration constructs no gated modules) -- skipping.")
        return None
    gate_names = sorted({k for epoch_gates in history["gates"] for k in epoch_gates})
    if not gate_names:
        print(f"  {subject_id} fold {fold_idx}: history['gates'] present but empty every "
              f"epoch -- this configuration has no gated modules (e.g. eeg_band_mode='none' "
              f"and eeg_backbone_mode='native'). Nothing to plot.")
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    epochs = np.arange(len(history["gates"]))
    for name in gate_names:
        values = [g.get(name, np.nan) for g in history["gates"]]
        ax.plot(epochs, values, label=name, marker=".", markersize=3)
    ax.set_xlabel("epoch (concatenated across stages)")
    ax.set_ylabel("sigmoid(raw gate)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{subject_id} fold {fold_idx} -- gate evolution")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"{subject_id}_fold{fold_idx}_gates.png")
        fig.savefig(path, dpi=130)
        print(f"saved: {path}")
    return fig


# ----------------------------------------------------------------------------
# 5. Grand summary across all 26 subjects -- reads the run's final CSV
#    (baseline_all_subjects_*.csv), no history needed. Complements the
#    per-fold plots above with the project-wide picture in one glance.
# ----------------------------------------------------------------------------
def plot_grand_kappa_distribution(csv_path, save_dir=None, reference_mean_kappa=None):
    """
    Reads a baseline_all_subjects_*.csv (the format every run_all_subjects.py
    run already produces) and plots per-subject mean_kappa sorted, with an
    optional horizontal reference line (e.g. the accepted E3+E4+EGTA
    baseline's mean_kappa) so a new intervention's per-subject pattern is
    immediately comparable, not just its single aggregate number.
    """
    import csv as csv_module
    rows = list(csv_module.DictReader(open(csv_path, encoding="utf-8")))
    subjects = [r["subject_id"] for r in rows]
    kappas = [float(r["mean_kappa"]) for r in rows]
    order = np.argsort(kappas)
    subjects_sorted = [subjects[i] for i in order]
    kappas_sorted = [kappas[i] for i in order]

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#C44E52" if k < 0 else "#4C72B0" for k in kappas_sorted]
    ax.bar(subjects_sorted, kappas_sorted, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    if reference_mean_kappa is not None:
        ax.axhline(reference_mean_kappa, color="#55A868", linestyle="--",
                   label=f"reference mean κ = {reference_mean_kappa:.4f}")
        ax.legend()
    ax.set_ylabel("mean_kappa (3 folds)")
    ax.set_title(f"per-subject kappa, sorted -- {os.path.basename(csv_path)}")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    fig.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, f"{os.path.basename(csv_path).replace('.csv', '')}_kappa_dist.png")
        fig.savefig(path, dpi=130)
        print(f"saved: {path}")
    return fig
