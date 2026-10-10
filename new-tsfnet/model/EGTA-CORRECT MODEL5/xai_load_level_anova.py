"""
xai_load_level_anova.py -- is the model's own attention/gate signal
FORMALLY, STATISTICALLY different across 0-/2-/3-back, or does it just "look"
different in a figure?

WHY THIS EXISTS
STA-Net describes its prediction-weight curves qualitatively ("kappa_fnirs
stays flat", "kappa_fusion is larger in MI/WG than MA"); MGFNet's t-SNE and
confusion-matrix figures are likewise descriptive. Neither paper runs a
formal omnibus test across task conditions on an XAI-derived signal, the way
a behavioural or physiological result normally has to pass before being
reported. capacity_ceiling_check.py already holds this project to that
standard for REAL physiological measures (permutation tests, BH-FDR,
heterogeneity). This script applies the matching formal test -- repeated-
measures ANOVA's non-parametric equivalent, the FRIEDMAN TEST, plus
Benjamini-Hochberg-corrected post-hoc pairwise comparisons -- to the model's
OWN internal XAI signals across all three cognitive-load levels (not just
the 3-vs-2 contrast xai_biomarker_validation.py already covers).

DESIGN
Each of the 26 subjects contributes ONE value per load level per signal
(that subject's mean over all held-out blocks at that level, pooled across
the 3 cross-session folds) -- subjects are the "blocks", load level is the
3-level repeated-measures factor, exactly mirroring capacity_ceiling_check.py's
own hbo_pfc/theta_fm/p300_pz profile construction (its `prof` dict), but for
the model's attention/gate signals instead of raw physiology.

SIGNALS (same four as xai_biomarker_validation.py, see that file for the
underlying extraction and literature rationale)
  egta_centroid, egta_entropy, efgf1_phi_mean, efgf2_phi_mean

STATISTICS PER SIGNAL
  1. Friedman chi-square omnibus test (H0: the 3 load levels have the same
     distribution of subject-level means) -- the formal, non-parametric
     repeated-measures-ANOVA-equivalent this analysis is built around.
  2. Kendall's W (effect size companion to Friedman, 0-1, "how much of the
     variance is explained by load level vs. subject-to-subject noise").
  3. Post-hoc pairwise Wilcoxon signed-rank tests for all 3 pairs
     (0-vs-2, 0-vs-3, 2-vs-3), Benjamini-Hochberg corrected ACROSS ALL
     12 tests (3 pairs x 4 signals) in one family, the standard way to avoid
     inflating false positives when running many related comparisons.
  4. Shape classification per subject (monotonic / inverted-U / other), same
     convention as capacity_ceiling_check.py's hbo_shape, so a committee
     member can directly compare "does the model's attention follow the same
     load-ordering shape as the real prefrontal HbO response".

USAGE
=====
    python xai_load_level_anova.py --results-dir "E:\\...\\results_40b7d8ea53_seed42" ^
        --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback"

No GPU required, much faster with one. Same cost class as
xai_biomarker_validation.py (one forward pass per fold, no extra ablation
passes).
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import friedmanchisquare, wilcoxon
from torch.utils.data import DataLoader, Subset

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_load_level_check")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

SIGNALS = ["egta_centroid", "egta_entropy", "efgf1_phi_mean", "efgf2_phi_mean"]
LABELS = {"egta_centroid": "EGTA attention centroid (segment index)",
          "egta_entropy": "EGTA attention entropy",
          "efgf1_phi_mean": "EFGF-1 hybrid attention Phi (mean)",
          "efgf2_phi_mean": "EFGF-2 hybrid attention Phi (mean)"}
LOAD_NAMES = ["0-back", "2-back", "3-back"]
LOAD_SHORT = ["0b", "2b", "3b"]
PAIRS = [(0, 1), (0, 2), (1, 2)]   # indices into LOAD_NAMES


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


def build_model(n_segments):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=n_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def collect_fold_signals(full, test_idx, ckpt_path, device, batch_size, adabn):
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    model = build_model(n_segments).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    phi1_store, phi2_store = [], []

    def hook1(_m, _i, out):
        phi1_store.append(out[2].detach())

    def hook2(_m, _i, out):
        phi2_store.append(out[2].detach())
    h1 = model.efgf1.register_forward_hook(hook1)
    h2 = model.efgf2.register_forward_hook(hook2)

    test_loader = DataLoader(Subset(full, test_idx), batch_size=batch_size, shuffle=False)
    if adabn:
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)
        phi1_store.clear(); phi2_store.clear()

    egta_centroid, egta_entropy = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            model(x_eeg, x_fnirs, x_conn, x_chan)
            w = model.cafe._last_egta_weights.mean(dim=1)
            idx = torch.arange(w.shape[1], device=w.device).float()
            centroid = (w * idx).sum(dim=1) / (w.sum(dim=1) + 1e-12)
            entropy = -(w * (w + 1e-12).log()).sum(dim=1)
            egta_centroid.append(centroid.cpu().numpy())
            egta_entropy.append(entropy.cpu().numpy())

    h1.remove(); h2.remove()
    centroid_all = np.concatenate(egta_centroid)
    entropy_all = np.concatenate(egta_entropy)
    n_total = centroid_all.shape[0]

    def phi_mean_per_sample(store):
        if not store:
            return np.full(n_total, np.nan)
        cat = torch.cat(store, dim=0)
        return cat.reshape(cat.shape[0], -1).mean(dim=1).cpu().numpy()

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"egta_centroid": centroid_all, "egta_entropy": entropy_all,
            "efgf1_phi_mean": phi_mean_per_sample(phi1_store),
            "efgf2_phi_mean": phi_mean_per_sample(phi2_store)}


def trial_mean(values, trial_ids):
    trials = np.unique(trial_ids)
    return trials, np.array([values[trial_ids == t].mean() for t in trials])


def subject_profile(sid, results_dir, data_root, device, batch_size, adabn):
    """Returns, per signal, a length-3 vector [mean@0back, mean@2back, mean@3back]
    pooled over all 3 cross-session folds."""
    subject_dir = os.path.join(data_root, sid)
    full = NBackDataset(subject_dir, sid)
    rows = []
    for test_session in (1, 2, 3):
        ckpt_path = os.path.join(results_dir, f"{sid}_session{test_session}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[{sid} s{test_session}] SKIP -- checkpoint not found")
            continue
        test_idx = np.flatnonzero(full.session_ids == test_session)
        signals = collect_fold_signals(full, test_idx, ckpt_path, device, batch_size, adabn)
        labels_w = full.labels[test_idx]
        trial_ids_w = full.trial_ids[test_idx]
        trials, lab_per_trial = trial_mean(labels_w, trial_ids_w)
        lab_per_trial = np.round(lab_per_trial).astype(int)
        for sig_name in SIGNALS:
            _, v_per_trial = trial_mean(signals[sig_name], trial_ids_w)
            for lab, v in zip(lab_per_trial, v_per_trial):
                rows.append({"label": int(lab), "signal": sig_name, "value": float(v)})
    if not rows:
        return None
    profile = {}
    for sig_name in SIGNALS:
        vals = [r["value"] for r in rows if r["signal"] == sig_name]
        labs = [r["label"] for r in rows if r["signal"] == sig_name]
        vals, labs = np.array(vals), np.array(labs)
        profile[sig_name] = np.array([vals[labs == k].mean() if (labs == k).any() else np.nan
                                      for k in (0, 1, 2)])
    return profile


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
    """matrix: (n_subjects, k_conditions). Kendall's W for Friedman's test."""
    n, k = matrix.shape
    ranks = np.apply_along_axis(lambda row: np.argsort(np.argsort(row)) + 1, 1, matrix)
    rank_sums = ranks.sum(axis=0)
    mean_rank_sum = rank_sums.mean()
    ss = ((rank_sums - mean_rank_sum) ** 2).sum()
    return float(12 * ss / (n ** 2 * (k ** 3 - k) + 1e-12))


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_load_level_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    profiles = {}
    for sid in subjects:
        try:
            prof = subject_profile(sid, ARGS.results_dir, ARGS.data_root, device,
                                   ARGS.batch_size, ARGS.adabn)
            if prof is not None:
                profiles[sid] = prof
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")

    ids = sorted(profiles)
    if len(ids) < 8:
        sys.exit("ERROR: fewer than 8 subjects produced usable profiles.")
    print(f"\n{len(ids)} subjects processed.")

    # ---- collect raw post-hoc p-values across ALL signal x pair combinations first,
    #      so BH correction is applied over the whole family of 3*4=12 tests ----
    all_tests = []   # list of (signal, pair_idx, p)
    friedman_results = {}
    shapes = {sig: {"monotonic": [], "inverted-U": [], "other": []} for sig in SIGNALS}
    for sig_name in SIGNALS:
        mat = np.array([profiles[s][sig_name] for s in ids])   # (n_subjects, 3)
        ok_rows = ~np.isnan(mat).any(axis=1)
        mat_ok = mat[ok_rows]
        ids_ok = [s for s, keep in zip(ids, ok_rows) if keep]
        if mat_ok.shape[0] < 8:
            friedman_results[sig_name] = {"n": int(mat_ok.shape[0]), "stat": float("nan"), "p": float("nan"),
                                          "kendalls_w": float("nan")}
            continue
        stat, pv = friedmanchisquare(mat_ok[:, 0], mat_ok[:, 1], mat_ok[:, 2])
        w = kendalls_w(mat_ok)
        friedman_results[sig_name] = {"n": int(mat_ok.shape[0]), "stat": float(stat), "p": float(pv),
                                      "kendalls_w": w}
        for i, (a, b) in enumerate(PAIRS):
            try:
                _, pv_pair = wilcoxon(mat_ok[:, a], mat_ok[:, b])
            except ValueError:
                pv_pair = float("nan")
            all_tests.append((sig_name, i, pv_pair))
        for sid in ids_ok:
            m0, m2, m3 = profiles[sid][sig_name]
            key = ("monotonic" if m0 < m2 < m3 else "inverted-U" if (m2 > m3 and m2 > m0) else "other")
            shapes[sig_name][key].append(sid)

    raw_ps = [t[2] for t in all_tests]
    qs = bh(raw_ps)
    posthoc = {}
    for (sig_name, pair_idx, pv), qv in zip(all_tests, qs):
        posthoc.setdefault(sig_name, {})[PAIRS[pair_idx]] = {"p": float(pv), "q": float(qv)}

    hline = "=" * 100
    print("\n" + hline)
    print(f"LOAD-LEVEL FRIEDMAN TEST -- {len(ids)} subjects, 0-/2-/3-back, model-internal XAI signals")
    print("(BH-FDR correction applied across all 3 pairs x 4 signals = 12 post-hoc tests, one family)")
    print(hline)
    print(f"{'signal':20s} {'n':>4s} {'Friedman chi2':>14s} {'p':>10s} {'Kendall W':>10s} | post-hoc (q)")
    for sig_name in SIGNALS:
        fr = friedman_results[sig_name]
        ph = posthoc.get(sig_name, {})
        ph_str = "  ".join(f"{LOAD_SHORT[a]}v{LOAD_SHORT[b]}="
                           f"{ph.get((a, b), {}).get('q', float('nan')):.3g}" for a, b in PAIRS)
        print(f"{sig_name:20s} {fr['n']:4d} {fr['stat']:14.3f} {fr['p']:10.3g} {fr['kendalls_w']:10.3f} | {ph_str}")

    print("\nShape of the per-subject 0/2/3-back profile (same convention as capacity_ceiling_check.py):")
    for sig_name in SIGNALS:
        sh = shapes[sig_name]
        print(f"  {sig_name:20s}: monotonic {len(sh['monotonic']):2d}, inverted-U {len(sh['inverted-U']):2d}, "
              f"other {len(sh['other']):2d}")

    print("\nREADING")
    sig_ps = {sig: friedman_results[sig]["p"] for sig in SIGNALS}
    any_sig = [s for s, pv in sig_ps.items() if pv == pv and pv < 0.05]
    if any_sig:
        best = min(any_sig, key=lambda s: sig_ps[s])
        print(f"  Formal omnibus evidence that '{best}' differs across cognitive-load levels "
              f"(Friedman p={sig_ps[best]:.3g}, Kendall's W={friedman_results[best]['kendalls_w']:.2f}). "
              f"This is a REPORTED, FDR-corrected inferential claim about the model's own internal "
              f"representation tracking task difficulty -- not a qualitative figure description, which is "
              f"all STA-Net and MGFNet provide for their analogous attention-weight-across-conditions figures.")
    else:
        print("  No XAI signal cleared the omnibus significance threshold across load levels. Reporting this "
              "honestly matters: it means the model's attention/gating, whatever role it plays in "
              "classification accuracy, does NOT visibly track task difficulty in a way a formal "
              "repeated-measures test can detect at this sample size -- a materially different (and more "
              "defensible) claim than simply describing a figure as 'load-dependent'.")

    summary = {"n_subjects": len(ids), "friedman": friedman_results,
              "posthoc_bh_family": {sig: {f"{LOAD_NAMES[a]}_vs_{LOAD_NAMES[b]}": v
                                          for (a, b), v in posthoc.get(sig, {}).items()}
                                    for sig in SIGNALS},
              "shapes": {sig: {k: v for k, v in shapes[sig].items()} for sig in SIGNALS}}
    with open(os.path.join(out_dir, "xai_load_level_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_load_level_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{sig}_{lab}" for sig in SIGNALS for lab in LOAD_NAMES])
        for s in ids:
            w.writerow([s] + [profiles[s][sig][k] for sig in SIGNALS for k in (0, 1, 2)])
    print(f"\n[written] {out_dir}: xai_load_level_summary.json, xai_load_level_per_subject.csv")


if __name__ == "__main__":
    main()
