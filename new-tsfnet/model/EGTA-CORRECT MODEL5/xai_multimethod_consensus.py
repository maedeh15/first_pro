"""
xai_multimethod_consensus.py -- do THREE INDEPENDENT explanation methods agree
on which fNIRS segments matter, or does each one tell a different story?

WHY THIS EXISTS
STA-Net, MGFNet and MSTFDN each report exactly ONE explanation method
(attention weight) and treat its output as ground truth. This project's own
thesis proposal cites a published example (SHAP DeepExplainer + GradientExplainer
run TOGETHER, precisely because a single XAI method's output is known to be
unreliable on its own -- Lundberg & Lee 2017; Molnar's "Interpretable ML" book,
ch. on method disagreement). None of the three reference papers apply that
same discipline to their own attention-only explanations. This script does:
three independent, architecturally-unrelated methods are computed on the EXACT
SAME input, and their agreement is measured directly, not assumed.

THE THREE METHODS (all reduced to one importance value per fNIRS SEGMENT, so
they are directly comparable to each other and to EGTA's own granularity)
  1. attention        EGTA's own softmax attention weight over the 21
                       segments (model.cafe._last_egta_weights, mean over
                       heads) -- what the model's architecture CLAIMS it uses.
  2. input_gradient    plain saliency: |d(true-class prob)/d(x_fnirs)|,
                       averaged over each segment's (H, W, T, chromophore)
                       volume. Cheapest gradient-based method, one backward
                       pass; sensitive to local curvature noise (Smilkov et
                       al. 2017) but a standard, widely used baseline.
  3. integrated_gradients   Sundararajan et al. 2017: integrates the
                       input-gradient along a straight path from an
                       all-zero baseline to the real input (N_STEPS Riemann
                       steps), which removes much of plain saliency's
                       sensitivity to local gradient noise and satisfies a
                       completeness axiom plain saliency does not. More
                       expensive (N_STEPS extra forward+backward passes) but
                       architecturally independent of both attention (not a
                       learned weight at all) and plain saliency (integrates
                       rather than reading one local gradient).

These three differ in EVERY relevant way a method can differ: attention is a
LEARNED, architecture-internal weight; input-gradient and integrated
gradients are POST-HOC, computed from the trained network's gradient field
with no learned parameters of their own; input-gradient uses one local
gradient while integrated gradients integrates over a path. If all three
still agree on which segments matter, that is strong, method-independent
evidence the finding is real rather than an artifact of any one method's
particular inductive bias.

AGREEMENT METRIC
For every held-out window, Spearman correlation between every pair of the
three methods' 21-segment importance vectors (3 pairs: attention-saliency,
attention-IG, saliency-IG), averaged per subject, with the same bootstrap-CI
and group-level reporting used in attention_faithfulness_check.py and
xai_stability_check.py.

USAGE
=====
    python xai_multimethod_consensus.py --results-dir "E:\\...\\results_40b7d8ea53_seed42" ^
        --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback"

No GPU required, much faster with one. Integrated gradients (N_STEPS=20 by
default) dominates the cost -- lower --ig-steps for a quicker, slightly
noisier pass.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader, Subset

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
from xai_trunk import trunk_state                       # drops 'riem_head.*' so a member-branch checkpoint loads into the plain trunk

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_consensus_check")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--ig-steps", type=int, default=20)
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--n-bootstrap", type=int, default=2000)
p.add_argument("--seed", type=int, default=0, help="seed of the bootstrap (it used the global, unseeded numpy RNG before)")
p.add_argument("--target", choices=["true", "pred"], default="true",
               help="class explained: the TRUE label (default, as before) or the class the model PREDICTS on the real input "
                    "(an explanation of the model's own decision; the class is fixed from the real input, not re-chosen along the IG path)")
p.add_argument("--objective", choices=["prob", "logprob"], default="prob",
               help="prob (default, as before; saturates when the model is confident) or log-probability (does not saturate)")
p.add_argument("--rho-threshold", type=float, default=0.5,
               help="Spearman agreement counted as 'strong' in READING. It was a hard-coded 0.5; state it BEFORE looking at the result.")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

METHOD_PAIRS = [("attention", "input_gradient"), ("attention", "integrated_gradients"),
                ("input_gradient", "integrated_gradients")]


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


def true_class_prob(y_pred, y):
    return y_pred.gather(1, y.view(-1, 1)).squeeze(1)


def target_value(y_pred, cls):
    """quantity differentiated by saliency / IG: P(cls) or log P(cls), per --objective"""
    p = y_pred.gather(1, cls.view(-1, 1)).squeeze(1)
    return p if ARGS.objective == "prob" else torch.log(p.clamp_min(1e-8))


def per_segment_mean(vol):
    """vol: (B, S, H, W, T, C) -> (B, S), mean |.| over the non-segment dims."""
    return vol.abs().mean(dim=tuple(range(2, vol.dim())))


def three_methods(model, x_eeg, x_fnirs, y, x_conn, x_chan, ig_steps):
    # 1. attention (no grad needed, just read the module's own stored weights)
    with torch.no_grad():
        y_pred0 = model(x_eeg, x_fnirs, x_conn, x_chan)[3]
        attn = model.cafe._last_egta_weights.mean(dim=1)   # (B, S)
    cls = y if ARGS.target == "true" else y_pred0.argmax(1)      # fixed ONCE, from the real input

    # 2. plain input-gradient saliency
    x_fnirs_g = x_fnirs.clone().requires_grad_(True)
    _, _, _, y_pred, _ = model(x_eeg, x_fnirs_g, x_conn, x_chan)
    target = target_value(y_pred, cls).sum()
    grad = torch.autograd.grad(target, x_fnirs_g, retain_graph=False)[0]
    saliency = per_segment_mean(grad.detach())

    # 3. integrated gradients (Sundararajan et al. 2017), all-zero baseline
    baseline = torch.zeros_like(x_fnirs)
    total_grad = torch.zeros_like(x_fnirs)
    for step in range(ig_steps):
        alpha = (step + 0.5) / ig_steps
        x_interp = (baseline + alpha * (x_fnirs - baseline)).clone().requires_grad_(True)
        _, _, _, y_pred_i, _ = model(x_eeg, x_interp, x_conn, x_chan)
        target_i = target_value(y_pred_i, cls).sum()
        g = torch.autograd.grad(target_i, x_interp, retain_graph=False)[0]
        total_grad = total_grad + g.detach()
    avg_grad = total_grad / ig_steps
    ig = (x_fnirs - baseline) * avg_grad
    ig_vec = per_segment_mean(ig)

    return {"attention": attn.detach().cpu().numpy(),
            "input_gradient": saliency.cpu().numpy(),
            "integrated_gradients": ig_vec.cpu().numpy()}


_BOOT_RNG = np.random.RandomState(ARGS.seed)


def bootstrap_ci(values, n_boot, alpha=0.05):
    if len(values) < 3:
        return float("nan"), float("nan")
    arr = np.array(values)
    boots = [_BOOT_RNG.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def subject_consensus(sid, results_dir, data_root, device, batch_size, adabn, ig_steps):
    subject_dir = os.path.join(data_root, sid)
    full = NBackDataset(subject_dir, sid)
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    pair_rhos = {pair: [] for pair in METHOD_PAIRS}

    for test_session in (1, 2, 3):
        ckpt_path = os.path.join(results_dir, f"{sid}_session{test_session}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[{sid} s{test_session}] SKIP -- checkpoint not found: {ckpt_path}")
            continue
        model = build_model(n_segments).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(trunk_state(state), strict=True)
        model.eval()

        test_idx = np.flatnonzero(full.session_ids == test_session)
        test_loader = DataLoader(Subset(full, test_idx), batch_size=batch_size, shuffle=False)
        if adabn:
            adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)

        for batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs, y = x_eeg.to(device), x_fnirs.to(device), y.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            vecs = three_methods(model, x_eeg, x_fnirs, y, x_conn, x_chan, ig_steps)
            B = vecs["attention"].shape[0]
            for b in range(B):
                for pair in METHOD_PAIRS:
                    a, b_ = vecs[pair[0]][b], vecs[pair[1]][b]
                    # scale-free constancy test (np.allclose's absolute tolerance 1e-8 would discard small probability gradients)
                    if np.ptp(a) <= 1e-9 * (np.abs(a).max() + 1e-300) or np.ptp(b_) <= 1e-9 * (np.abs(b_).max() + 1e-300):
                        continue
                    rho, _ = spearmanr(a, b_)
                    if not np.isnan(rho):
                        pair_rhos[pair].append(rho)

        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return pair_rhos


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_consensus_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    per_subject = {}
    for sid in subjects:
        try:
            pair_rhos = subject_consensus(sid, ARGS.results_dir, ARGS.data_root, device,
                                          ARGS.batch_size, ARGS.adabn, ARGS.ig_steps)
            if not pair_rhos[METHOD_PAIRS[0]]:
                print(f"[{sid}] SKIP -- no usable samples.")
                continue
            per_subject[sid] = pair_rhos
            msg = "  ".join(f"{a[:4]}-{b[:4]}={np.mean(pair_rhos[(a, b)]):+.2f}" for a, b in METHOD_PAIRS)
            print(f"[{sid}] n={len(pair_rhos[METHOD_PAIRS[0]]):4d}  {msg}")
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")

    ids = sorted(per_subject)
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects produced usable consensus data.")

    hline = "=" * 100
    print("\n" + hline)
    print(f"MULTI-METHOD XAI CONSENSUS SUMMARY -- {len(ids)} subjects")
    print(hline)
    print(f"{'pair':28s} {'mean rho':>9s} {'SD':>7s} {'95% CI (bootstrap, subj. means)':>34s} "
          f"{'Wilcoxon p (rho>0)':>20s}")
    summary = {"per_subject": {}, "pairs": {}}
    for pair in METHOD_PAIRS:
        subj_means = []
        for s in ids:
            v = per_subject[s][pair]
            if v:
                subj_means.append(float(np.mean(v)))
        subj_means = np.array(subj_means)
        lo, hi = bootstrap_ci(subj_means, ARGS.n_bootstrap)
        stat = wilcoxon(subj_means, alternative="greater") if len(subj_means) >= 6 else None
        pname = f"{pair[0]} vs {pair[1]}"
        summary["pairs"][pname] = {
            "n_subjects": len(subj_means), "mean_rho": float(subj_means.mean()),
            "sd_rho": float(subj_means.std(ddof=1)), "ci": [lo, hi],
            "wilcoxon_p_greater": float(stat.pvalue) if stat is not None else float("nan")}
        print(f"{pname:28s} {subj_means.mean():9.3f} {subj_means.std(ddof=1):7.3f} "
              f"[{lo:6.3f}, {hi:6.3f}]{'':10s} "
              f"{(stat.pvalue if stat is not None else float('nan')):20.4g}")

    for s in ids:
        summary["per_subject"][s] = {f"{a}_{b}_mean_rho": float(np.mean(per_subject[s][(a, b)]))
                                     if per_subject[s][(a, b)] else float("nan")
                                     for a, b in METHOD_PAIRS}

    print("\nREADING")
    all_means = [summary["pairs"][f"{a} vs {b}"]["mean_rho"] for a, b in METHOD_PAIRS]
    all_ps = [summary["pairs"][f"{a} vs {b}"]["wilcoxon_p_greater"] for a, b in METHOD_PAIRS]
    thr = ARGS.rho_threshold
    frac = {f"{a} vs {b}": float(np.mean([np.mean(per_subject[s][(a, b)]) > thr for s in ids if per_subject[s][(a, b)]])) for a, b in METHOD_PAIRS}
    print(f"  threshold for 'strong' agreement: rho > {thr} (set with --rho-threshold; state it before looking). Fraction of subjects above it: "
          + "; ".join(f"{k} {v:.2f}" for k, v in frac.items()))
    print(f"  target = {ARGS.target} label, objective = {ARGS.objective}; the Wilcoxon test is one-sided (rho>0) and with ~26 subjects it is passed by "
          f"almost any positive correlation, so the MAGNITUDE and the fraction above threshold are the informative numbers.")
    if all(m > thr for m in all_means) and all(p < 0.05 for p in all_ps):
        print("  All three architecturally-independent methods (a LEARNED attention weight and two POST-HOC "
              "gradient methods, one local and one path-integrated) agree STRONGLY on which fNIRS segments "
              "matter, and each pairwise agreement is statistically significant. This is a form of "
              "triangulated evidence for the explanation's validity that none of the three reference papers "
              "provide -- they report attention alone and never check it against an independent method.")
    elif all(p < 0.05 for p in all_ps):
        print("  All three methods agree significantly better than chance, but only moderately -- report "
              "this honestly as partial convergence: the broad picture (which segments matter most) is "
              "shared, but the fine-grained ranking differs by method, so any single-method figure "
              "overstates precision.")
    else:
        print("  At least one method pair shows no significant agreement. This directly surfaces the "
              "'attention is not explanation' risk this check was designed to catch: if the LEARNED "
              "attention weight disagrees with the POST-HOC gradient methods, the attention map alone "
              "(the ONLY thing STA-Net/MGFNet/MSTFDN show) is not a trustworthy stand-alone explanation. "
              "Report which pair disagrees -- that pinpoints whether the issue is the learned attention "
              "itself or the gradient-based methods' known sensitivity to noise.")

    with open(os.path.join(out_dir, "xai_consensus_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_consensus_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["subject"] + [f"{a}_{b}_mean_rho" for a, b in METHOD_PAIRS]
        w.writerow(header)
        for s in ids:
            w.writerow([s] + [summary["per_subject"][s][f"{a}_{b}_mean_rho"] for a, b in METHOD_PAIRS])
    print(f"\n[written] {out_dir}: xai_consensus_summary.json, xai_consensus_per_subject.csv")


if __name__ == "__main__":
    main()
