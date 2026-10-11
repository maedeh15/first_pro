# ============================================================================
# Shared machinery for the four STA-Net XAI VALIDATION scripts
#   stanet_xai_biomarker_validation.py   does the attention track real physiology?
#   stanet_xai_load_level_anova.py       does it differ formally across 0/2/3-back?
#   stanet_xai_multimethod_consensus.py  do independent XAI methods agree?
#   stanet_xai_stability_check.py        is the explanation reproducible across seeds?
#
# These are the STA-Net counterparts of your TSFNet xai_* scripts. The statistics are
# unchanged; what changes is where the model-internal signals come from.
#
#   TSFNet                                        STA-Net (this file)
#   model.cafe._last_egta_weights (B,h,S)         out["att"] (B,l)   -- already head-averaged
#                                                 by STANet.forward (att.mean over heads+query)
#   forward hook on efgf1/efgf2, out[2] = phi     out["phi1"] (B,1,8,8,1), out["phi2"] (B,1,4,4,1)
#   y_pred (4th of a 5-tuple)                     out["p_final"] (B,3)
#   fixed scalars w_eeg/w_fusion/w_fnirs          out["w"] (B,2) -- INSTANCE-ADAPTIVE, see below
#   NBackDataset + DataLoader + _unpack_batch     load_subject + WindowLoader (stanet_common)
#
# No forward hooks are needed anywhere: STANet.forward already returns att, phi1 and phi2 in
# its output dict, so every signal is read directly.
#
# ONE SIGNAL IS NEW, AND IT IS STA-NET-SPECIFIC ------------------------------
# TSFNet's three fusion weights are fixed model-level scalars, which is why your
# xai_extensions.py says SHAP-on-gating "has no meaning here". STA-Net's are NOT: Eq. 6-7
# compute w = softmax([kappa_fusion, kappa_fnirs]) from EACH SAMPLE's own fusion and fNIRS
# embeddings, so every window gets its own pair of weights. That makes w_fnirs a per-window
# XAI signal in its own right -- "how much did the model trust the fNIRS head for THIS window"
# -- and it is included below as a fifth signal. It is the one place where STA-Net admits a
# question TSFNet's architecture cannot be asked.
#
# SEGMENT COUNT: l is read from the data (11 for the paper-exact preprocessing, 21 for the
# TSFNet-comparable one), so these scripts work on either without a flag.
# ============================================================================
import argparse
import os

import numpy as np
import torch

from stanet_common import (NUM_CLASSES, load_subject, WindowLoader, build_model,
                           adapt_batchnorm_to_target, fnirs_train_scale)

# --- the model-internal signals these scripts test -------------------------
SIGNALS = ["egta_centroid", "egta_entropy", "fgsa1_phi_mean", "fgsa2_phi_mean", "w_fnirs"]

# Only signals with a literature-grounded DIRECTIONAL prediction get a sign; the rest are
# reported descriptively (an imposed sign you cannot defend is worse than none).
EXPECTED_SIGN = {
    # Higher working-memory load -> slower/later haemodynamic response -> EGTA should have to
    # look FURTHER DOWN the lagged-window sequence to find the task-relevant fNIRS response.
    # Same directional logic as hbo_pfc in your capacity_ceiling_check.py.
    "egta_centroid": +1,
}

LABELS = {
    "egta_centroid": "EGTA attention centroid (lagged fNIRS window index)",
    "egta_entropy": "EGTA attention entropy (selectivity)",
    "fgsa1_phi_mean": "FGSA-1 spatial attention Phi (mean)",
    "fgsa2_phi_mean": "FGSA-2 spatial attention Phi (mean)",
    "w_fnirs": "decision weight on the fNIRS head (instance-adaptive, Eq. 6-7)",
}

CLASS_NAMES = ["0-back", "2-back", "3-back"]


# ---------------------------------------------------------------------------
# CLI pieces shared by all four scripts
# ---------------------------------------------------------------------------
def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def add_common_args(p):
    """--data-root / --subjects / --batch-size / --adabn / --fnirs-scale / --device."""
    p.add_argument("--data-root", required=True, help="Folder with VP001..VP026 preprocessed .npy files.")
    p.add_argument("--subjects", default="1-26")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--adabn", action="store_true", default=False,
                   help="Transductive BN adaptation on the held-out session. STA-Net does NOT do this; "
                        "default OFF. Turn it on only to match a run that was TRAINED that way -- the "
                        "signals must be extracted under the same conditions the checkpoint was evaluated.")
    p.add_argument("--fnirs-scale", type=float, default=None,
                   help="Override the fNIRS multiplier (metadata FNIRS_TRAIN_SCALE, default 1e3). Pass 1.0 "
                        "for preprocessing whose fNIRS is already z-scored. MUST match what training used.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


# ---------------------------------------------------------------------------
# Loading one subject + one fold's checkpoint
# ---------------------------------------------------------------------------
class SubjectData:
    """One subject's windows, loaded once and reused for all 3 folds (and, in the stability
    check, for all seeds) -- the arrays are hundreds of MB, so reloading per fold is wasteful."""

    def __init__(self, data_root, sid, fnirs_scale=None):
        d = os.path.join(data_root, sid)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"{d} does not exist (preprocessing not run for {sid}?)")
        self.dir, self.sid = d, sid
        (self.eeg, self.fnirs, self.labels,
         self.session_ids, self.trial_ids) = load_subject(d, sid, scale_override=fnirs_scale)
        self.seg_len = self.fnirs.shape[1]
        self.scale = fnirs_scale if fnirs_scale is not None else fnirs_train_scale(d, sid)

    def session_idx(self, session):
        return np.flatnonzero(self.session_ids == session)

    def loader(self, idx, batch_size):
        return WindowLoader(self.eeg, self.fnirs, self.labels, idx, batch_size)


def load_fold_model(ckpt_path, seg_len, device):
    model = build_model(seg_len=seg_len, num_classes=NUM_CLASSES, device=device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()
    return model


def free(model, device):
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# The extraction itself
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_fold_signals(data, test_idx, ckpt_path, device, batch_size, adabn, return_att=False):
    """One eval pass over a held-out session. Returns {signal_name: (n_windows,)}, in loader
    order (never shuffled), so window i is the same physical window for every checkpoint.
    With return_att, also returns the full (n_windows, l) EGTA attention matrix."""
    model = load_fold_model(ckpt_path, data.seg_len, device)
    loader = data.loader(test_idx, batch_size)
    if adabn:
        adapt_batchnorm_to_target(model, loader, device, log_fn=lambda *a, **k: None)

    store = {k: [] for k in SIGNALS}
    att_rows = []
    for xe, xf, _y in loader:
        out = model(xe.to(device), xf.to(device))
        a = out["att"]                                           # (B, l), softmax over the l windows
        a = a / a.sum(dim=1, keepdim=True).clamp_min(1e-12)      # renormalise: it is a head mean
        idx = torch.arange(a.shape[1], device=a.device, dtype=a.dtype)
        store["egta_centroid"].append((a * idx).sum(dim=1).cpu().numpy())
        store["egta_entropy"].append((-(a * (a + 1e-12).log()).sum(dim=1)).cpu().numpy())
        store["fgsa1_phi_mean"].append(out["phi1"].flatten(1).mean(dim=1).cpu().numpy())
        store["fgsa2_phi_mean"].append(out["phi2"].flatten(1).mean(dim=1).cpu().numpy())
        store["w_fnirs"].append(out["w"][:, 1].cpu().numpy())    # w = [w_fusion, w_fnirs]
        if return_att:
            att_rows.append(a.cpu().numpy())

    free(model, device)
    sig = {k: np.concatenate(v) for k, v in store.items()}
    return (sig, np.concatenate(att_rows, axis=0)) if return_att else sig


def trial_mean(values, trial_ids):
    """Window-level values -> one value per trial (the unit of analysis in all four scripts:
    windows inside a trial are overlapping and not independent, trials are)."""
    trials = np.unique(trial_ids)
    return trials, np.array([values[trial_ids == t].mean() for t in trials])


def fold_checkpoint(results_dir, sid, session):
    return os.path.join(results_dir, f"{sid}_session{session}_best.pt")


def subject_trial_rows(data, results_dir, device, batch_size, adabn, sessions=(1, 2, 3)):
    """Per-trial rows for every fold of one subject: [{session, label, signal, value}, ...].
    Each trial is held out exactly once (leave-one-session-out), so no trial is counted twice."""
    rows = []
    for session in sessions:
        ckpt = fold_checkpoint(results_dir, data.sid, session)
        if not os.path.isfile(ckpt):
            print(f"[{data.sid} s{session}] SKIP -- checkpoint not found: {ckpt}")
            continue
        test_idx = data.session_idx(session)
        if test_idx.size == 0:
            print(f"[{data.sid} s{session}] SKIP -- no windows for this session")
            continue
        sig = collect_fold_signals(data, test_idx, ckpt, device, batch_size, adabn)
        lab_w, tid_w = data.labels[test_idx], data.trial_ids[test_idx]
        _, lab_t = trial_mean(lab_w, tid_w)                 # constant within a trial
        lab_t = np.round(lab_t).astype(int)
        for name in SIGNALS:
            _, v_t = trial_mean(sig[name], tid_w)
            for lab, v in zip(lab_t, v_t):
                rows.append({"session": session, "label": int(lab), "signal": name, "value": float(v)})
    return rows


# ---------------------------------------------------------------------------
# Statistics shared by several of the scripts (same engine as your TSFNet versions)
# ---------------------------------------------------------------------------
def bh(pvals):
    """Benjamini-Hochberg FDR. NaNs pass through."""
    p_ = np.asarray(pvals, float)
    q = np.full_like(p_, np.nan)
    ok = ~np.isnan(p_)
    m = int(ok.sum())
    if m:
        idx = np.argsort(p_[ok])
        ranked = p_[ok][idx] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[idx] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    if len(values) < 3:
        return float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    arr = np.asarray(values, float)
    boots = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def run_header(title, data_root, results_dirs, adabn, scale, extra=""):
    print("=" * 100)
    print(title)
    print("=" * 100)
    print(f"data_root = {data_root}")
    for d in ([results_dirs] if isinstance(results_dirs, str) else results_dirs):
        print(f"checkpoints = {d}")
    print(f"AdaBN = {adabn} (STA-Net default OFF) | fNIRS scale = "
          f"{'from metadata' if scale is None else f'x{scale:g} (--fnirs-scale)'}"
          + (f" | {extra}" if extra else ""))
    print()
