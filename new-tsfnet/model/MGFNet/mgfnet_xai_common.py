"""
mgfnet_xai_common.py -- shared code for the four MGFNet XAI validation scripts
(xai_biomarker_validation / xai_load_level_anova / xai_multimethod_consensus /
xai_stability_check). The statistics are the SAME engine the TSFNet versions
use (copied from capacity_ceiling_check.py there); only the model-internal
signals change, because MGFNet has no EGTA segment attention and no EFGF Phi.

MGFNet'S MODEL-INTERNAL SIGNALS (eval mode, one value per window)
  From the CGSCR module -- the part of MGFNet the paper's claim rests on:
  gate_eeg           gate weight of the EEG intra-modal branch
  gate_fnirs         HbO + HbR intra-modal gate weights
  gate_cross         EEG-HbO + EEG-HbR + HbO-HbR cross-modal gate weights
  gate_entropy       entropy of the 6-way gate (low = selective, ln6 = uniform)
  coupling_entropy   mean row entropy of the coupling matrix C (5 EEG bands x
                     3 fNIRS scales); low = each EEG band couples to ONE scale
  routing_theta_mass probability mass the sparse router puts on the 3 tokens
                     built from the THETA Morlet band. Router distribution =
                     softmax(importance scores): the Gumbel-max sampling
                     distribution the training-time selection draws from.
                     HYPOTHESIS (+): frontal-midline theta rises with working-
                     memory load (Jensen & Tesche 2002; Gevins et al. 1997), so
                     the router should rely more on theta tokens at 3-back than
                     2-back -- the MGFNet counterpart of TSFNet's EGTA-centroid
                     hypothesis, testable against capacity_ceiling_check's
                     theta_fm per-subject effect sizes.
  routing_entropy    entropy of that router distribution (log 15 = uniform).

Checkpoints: run_all_subjects_MGFNET.py's {subject}_session{s}_mgf_final.pt,
which carry weights + train-fold normaliser + model kwargs. They are
evaluated exactly as in training (no AdaBN: the reported MGFNet results do not
use it, so the explanations describe the same model the accuracies describe).
"""
import itertools
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from mgfnet_dataset import MGFNBackDataset, _unpack_mgf_batch
from mgfnet_xai import checkpoint_path, load_mgfnet_checkpoint

SIGNALS = ["gate_eeg", "gate_fnirs", "gate_cross", "gate_entropy",
           "coupling_entropy", "routing_theta_mass", "routing_entropy"]
EXPECTED_SIGN = {"routing_theta_mass": +1}     # the only literature-grounded direction
LABELS = {"gate_eeg": "CGSCR gate: EEG branch",
          "gate_fnirs": "CGSCR gate: HbO + HbR branches",
          "gate_cross": "CGSCR gate: 3 cross-modal branches",
          "gate_entropy": "CGSCR gate entropy",
          "coupling_entropy": "coupling matrix C, mean row entropy",
          "routing_theta_mass": "router mass on theta-band tokens",
          "routing_entropy": "router distribution entropy"}
THETA_BAND = 1                      # EEG Morlet components: delta, THETA, alpha, beta, gamma
N_NIRS_COMP = 3


# ---------------------------------------------------------------------------
# Subjects / data / checkpoints
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


def load_subject(data_root, sid):
    return MGFNBackDataset(os.path.join(data_root, sid), sid)


def load_fold(results_dir, sid, test_session, device):
    """-> (model, normaliser) or None if the checkpoint is missing.
    Refuses a w/o-CGSCR checkpoint: it has no gate/coupling/router."""
    path = checkpoint_path(results_dir, sid, test_session)
    if not os.path.isfile(path):
        print(f"[{sid} s{test_session}] SKIP -- checkpoint not found: {path}")
        return None
    model, norm, _ = load_mgfnet_checkpoint(path, device)
    if not model.use_cgscr:
        raise ValueError(f"{path} is a w/o-CGSCR model: no CGSCR signals to analyse")
    return model, norm


def test_loader(full, test_session, batch_size):
    idx = np.flatnonzero(full.session_ids == test_session)
    return DataLoader(Subset(full, idx), batch_size=batch_size, shuffle=False), idx


def to_device(batch, device, norm):
    x_eeg, x_hbo, x_hbr, y = _unpack_mgf_batch(batch)
    x_eeg, x_hbo, x_hbr = norm(x_eeg.to(device), x_hbo.to(device), x_hbr.to(device))
    return x_eeg, x_hbo, x_hbr, y.to(device)


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------
def _entropy(p, dim=-1):
    return -(p * (p + 1e-12).log()).sum(dim=dim)


def signals_from_aux(aux):
    """Per-window scalar signals (dict of (B,) tensors) + the raw vectors."""
    g, C, s = aux["gate_weights"], aux["coupling"], aux["importance_scores"]
    p = F.softmax(s, dim=-1)                                  # router distribution (B, 15)
    theta_tokens = [THETA_BAND * N_NIRS_COMP + j for j in range(N_NIRS_COMP)]
    return {
        "gate_eeg": g[:, 0], "gate_fnirs": g[:, 1] + g[:, 2], "gate_cross": g[:, 3:].sum(1),
        "gate_entropy": _entropy(g), "coupling_entropy": _entropy(C).mean(1),
        "routing_theta_mass": p[:, theta_tokens].sum(1), "routing_entropy": _entropy(p),
    }, {"gate": g, "coupling": C.flatten(1), "routing": p}


@torch.no_grad()
def collect_fold(model, norm, loader, device):
    """-> (scalar signals {name: (N,)}, vectors {gate (N,6), coupling (N,15),
    routing (N,15)}), in loader order (shuffle=False)."""
    model.eval()
    sc, vec = {k: [] for k in SIGNALS}, {"gate": [], "coupling": [], "routing": []}
    for batch in loader:
        x_eeg, x_hbo, x_hbr, _ = to_device(batch, device, norm)
        _, _, aux = model(x_eeg, x_hbo, x_hbr)
        s, v = signals_from_aux(aux)
        for k in SIGNALS:
            sc[k].append(s[k].cpu().numpy())
        for k in vec:
            vec[k].append(v[k].cpu().numpy())
    return ({k: np.concatenate(v) for k, v in sc.items()},
            {k: np.concatenate(v) for k, v in vec.items()})


def trial_mean(values, trial_ids):
    trials = np.unique(trial_ids)
    return trials, np.array([values[trial_ids == t].mean() for t in trials])


def subject_trial_rows(sid, results_dir, data_root, device, batch_size):
    """One row per (held-out trial, signal): the trial-mean of the window
    signal, from the LOSO fold that held that trial's session out."""
    full = load_subject(data_root, sid)
    rows = []
    for test_session in (1, 2, 3):
        fold = load_fold(results_dir, sid, test_session, device)
        if fold is None:
            continue
        model, norm = fold
        loader, idx = test_loader(full, test_session, batch_size)
        sig, _ = collect_fold(model, norm, loader, device)
        labels_w = full.labels.numpy()[idx]
        trial_w = full.trial_ids[idx]
        _, lab_t = trial_mean(labels_w, trial_w)
        lab_t = np.round(lab_t).astype(int)
        for name in SIGNALS:
            _, v_t = trial_mean(sig[name], trial_w)
            for lab, v in zip(lab_t, v_t):
                rows.append({"session": test_session, "label": int(lab), "signal": name, "value": float(v)})
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Forward pass with an externally supplied coupling matrix (for attribution)
# ---------------------------------------------------------------------------
def forward_parts(model, x_eeg, x_hbo, x_hbr):
    """Everything up to the CGSCR coupling: branch features + projected
    component embeddings + the model's own coupling matrix C."""
    T_eeg = x_eeg.shape[-1]
    feats = {}
    if model.use_intra_modal:
        feats["1"], feats["2"], feats["3"] = model.enc_eeg(x_eeg), model.enc_hbo(x_hbo), model.enc_hbr(x_hbr)
    if model.use_cross_modal:
        hbo_up = F.interpolate(x_hbo, size=T_eeg, mode="linear", align_corners=False)
        hbr_up = F.interpolate(x_hbr, size=T_eeg, mode="linear", align_corners=False)
        feats["12"] = model.enc_12(torch.cat([x_eeg, hbo_up], 1))
        feats["13"] = model.enc_13(torch.cat([x_eeg, hbr_up], 1))
        feats["23"] = model.enc_23(torch.cat([hbo_up, hbr_up], 1))
    c = model.cgscr
    E = c.eeg_enc(c.eeg_bank(x_eeg))
    Fc = 0.5 * (c.nirs_enc(c.hbo_bank(x_hbo)) + c.nirs_enc(c.hbr_bank(x_hbr)))
    Ep, Fp = c.proj_e(E), c.proj_f(Fc)
    return feats, Ep, Fp, c.coupling(Ep, Fp)


def logits_from_parts(model, feats, Ep, Fp, C=None, gate=None):
    """Hybrid-head logits with C and/or the 6-way gate OVERRIDDEN (eval mode:
    discrete argmax routing exactly as at test time). Returns (logits, gate)."""
    x_scr, _, _ = model.cgscr.route(Ep, Fp, C)
    g = F.softmax(model.cgscr.routing_gate(x_scr), -1) if gate is None else gate
    parts = [feats[k] * g[:, i:i + 1] for k, i in zip(model.active_keys, model._gate_idx)]
    return model.classifier(torch.cat(parts + [x_scr], 1)), g


# ---------------------------------------------------------------------------
# Statistics engine (verbatim from the TSFNet scripts / capacity_ceiling_check)
# ---------------------------------------------------------------------------
def exact_perm_p(v, lab, sess):
    """Exact within-session permutation p for mean(3-back) - mean(2-back)."""
    m = np.isin(lab, [1, 2]) & ~np.isnan(v)
    v, lab, sess = v[m], lab[m], sess[m]
    n3 = (lab == 2).sum(); n2 = (lab == 1).sum()
    if n3 == 0 or n2 == 0:
        return np.nan
    per_session = []
    for s in np.unique(sess):
        vs = v[sess == s]; k3 = int((lab[sess == s] == 2).sum())
        sums = np.array([vs[list(c)].sum() for c in itertools.combinations(range(len(vs)), k3)])
        per_session.append(sums)
    tot3 = per_session[0]
    for ps in per_session[1:]:
        tot3 = (tot3[:, None] + ps[None, :]).ravel()
    total = v.sum()
    stats = tot3 / n3 - (total - tot3) / n2
    obs = v[lab == 2].mean() - v[lab == 1].mean()
    return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))


def bh(pvals):
    p_ = np.asarray(pvals, float); q = np.full_like(p_, np.nan)
    ok = ~np.isnan(p_); idx = np.argsort(p_[ok]); m = ok.sum()
    if m:
        ranked = p_[ok][idx] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m); out[idx] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def kendalls_w(matrix):
    n, k = matrix.shape
    ranks = np.apply_along_axis(lambda row: np.argsort(np.argsort(row)) + 1, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    ss = ((rank_sums - rank_sums.mean()) ** 2).sum()
    return float(12 * ss / (n ** 2 * (k ** 3 - k) + 1e-12))


def icc_2_1(data):
    """Shrout & Fleiss ICC(2,1). data: (n_items, k_raters)."""
    n, k = data.shape
    gm = data.mean()
    ss_total = ((data - gm) ** 2).sum()
    ss_rows = k * ((data.mean(1) - gm) ** 2).sum()
    ss_cols = n * ((data.mean(0) - gm) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows, ms_cols = ss_rows / (n - 1), ss_cols / (k - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    return float((ms_rows - ms_err) / denom) if denom > 0 else float("nan")


def bootstrap_ci(values, n_boot, alpha=0.05, seed=0):
    if len(values) < 3:
        return float("nan"), float("nan")
    arr = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = [rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def safe_wilcoxon(x, y=None, **kw):
    """scipy.wilcoxon, returning nan instead of raising (all-zero differences,
    too few samples)."""
    from scipy.stats import wilcoxon
    try:
        return float(wilcoxon(x, y, **kw).pvalue)
    except ValueError:
        return float("nan")


def nan_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan
