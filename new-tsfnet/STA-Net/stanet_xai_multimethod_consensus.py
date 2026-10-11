"""
stanet_xai_multimethod_consensus.py -- do THREE INDEPENDENT explanation methods agree on which
lagged fNIRS windows matter, or does each tell a different story?

STA-Net counterpart of your xai_multimethod_consensus.py.

WHY THIS EXISTS
STA-Net, MGFNet and MSTFDN each report exactly ONE explanation method (an attention weight) and
treat its output as ground truth. The thesis proposal's own cited practice is the opposite --
run several methods together, precisely because a single XAI method is known to be unreliable
alone (Lundberg & Lee 2017; Molnar, Interpretable ML, ch. on method disagreement; Jain &
Wallace 2019 "Attention is not Explanation"). This script computes three architecturally
unrelated methods on the EXACT same inputs and measures their agreement instead of assuming it.

THE THREE METHODS (each reduced to one importance value per lagged fNIRS window, so all three
are directly comparable to each other and to EGTA's own granularity)
  1. attention            EGTA's own softmax weight over the l lagged fNIRS windows
                          (STANet's out["att"]) -- what the architecture CLAIMS it uses.
  2. input_gradient       plain saliency |d p_final[true class] / d x_fnirs|, averaged over each
                          window's (H, W, T, chromophore) volume. One backward pass; sensitive
                          to local gradient noise (Smilkov et al. 2017) but a standard baseline.
  3. integrated_gradients Sundararajan et al. 2017: integrates the input gradient along a
                          straight path from an all-zero baseline to the real input
                          (--ig-steps Riemann midpoints). Removes much of plain saliency's
                          local-noise sensitivity and satisfies a completeness axiom it does not.

These differ in every way a method can: attention is a LEARNED, architecture-internal weight;
the other two are POST-HOC with no parameters of their own; one reads a single local gradient,
the other integrates along a path. If all three still agree on which lagged windows matter,
that is triangulated, method-independent evidence -- which none of the three reference papers
provide.

NOTE ON THE BASELINE: integrated gradients uses an all-zero fNIRS tensor as its reference. For
the z-scored preprocessing, zero IS the per-channel mean, so this is the natural "no deviation
from baseline" reference. For the paper-exact mmol/L preprocessing the data is already
baseline-corrected per trial (-5..-2 s), so zero is again the trial's own baseline. The choice
is defensible in both cases, but it is a choice -- state it when reporting.

AGREEMENT METRIC
For each held-out window, Spearman correlation between every pair of the three methods'
l-element importance vectors, averaged per subject; then bootstrap CI and a one-sided group
Wilcoxon across subjects, the same reporting as stanet_xai_stability_check.py.

USAGE
=====
    python stanet_xai_multimethod_consensus.py ^
        --results-dir "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_seed42" ^
        --data-root   "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_nback" ^
        --fnirs-scale 1.0

--fnirs-scale must match training. Integrated gradients dominates the cost (--ig-steps forward
+ backward passes per batch); lower --ig-steps for a quicker, slightly noisier pass, and
--max-windows-per-fold to cap the work per fold. A GPU is strongly recommended here, unlike the
other three scripts.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from stanet_xai_signals import (SubjectData, add_common_args, bootstrap_ci, free,
                                load_fold_model, parse_subjects, run_header)

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True,
               help="Folder with VPxxx_session{1,2,3}_best.pt from ONE seed's LOSO run.")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_consensus_check")
p.add_argument("--ig-steps", type=int, default=20, help="Riemann steps for integrated gradients.")
p.add_argument("--max-windows-per-fold", type=int, default=None,
               help="Cap the windows analysed per fold (loader order, no shuffling). "
                    "Use it to keep the IG cost bounded; omit to use every held-out window.")
p.add_argument("--n-bootstrap", type=int, default=2000)
p.add_argument("--seed", type=int, default=0, help="Seed for the bootstrap only.")
add_common_args(p)
ARGS = p.parse_args()

METHOD_PAIRS = [("attention", "input_gradient"),
                ("attention", "integrated_gradients"),
                ("input_gradient", "integrated_gradients")]


def per_window_mean(vol):
    """vol: (B, l, H, W, T, C) -> (B, l): mean |.| over everything but the lag axis."""
    return vol.abs().mean(dim=tuple(range(2, vol.dim())))


def three_methods(model, xe, xf, y, ig_steps):
    """All three importance vectors for one batch. xf must NOT require grad on entry."""
    # 1. attention -- just read what the model already returns
    with torch.no_grad():
        out = model(xe, xf)
        a = out["att"]
        attn = (a / a.sum(dim=1, keepdim=True).clamp_min(1e-12)).cpu().numpy()

    pick = lambda pf: pf.gather(1, y.view(-1, 1)).squeeze(1).sum()

    # 2. plain input-gradient saliency on the true-class probability
    xg = xf.clone().requires_grad_(True)
    g = torch.autograd.grad(pick(model(xe, xg)["p_final"]), xg)[0]
    saliency = per_window_mean(g.detach()).cpu().numpy()

    # 3. integrated gradients from an all-zero baseline (midpoint Riemann sum)
    total = torch.zeros_like(xf)
    for step in range(ig_steps):
        alpha = (step + 0.5) / ig_steps
        xi = (alpha * xf).clone().requires_grad_(True)
        total += torch.autograd.grad(pick(model(xe, xi)["p_final"]), xi)[0].detach()
    ig = per_window_mean(xf * total / ig_steps).cpu().numpy()

    return {"attention": attn, "input_gradient": saliency, "integrated_gradients": ig}


def subject_consensus(data, results_dir, device, batch_size, adabn, ig_steps, cap):
    pair_rhos = {pair: [] for pair in METHOD_PAIRS}
    for session in (1, 2, 3):
        ckpt = os.path.join(results_dir, f"{data.sid}_session{session}_best.pt")
        if not os.path.isfile(ckpt):
            print(f"[{data.sid} s{session}] SKIP -- checkpoint not found")
            continue
        test_idx = data.session_idx(session)
        if cap is not None:
            test_idx = test_idx[:cap]
        if test_idx.size == 0:
            continue
        model = load_fold_model(ckpt, data.seg_len, device)
        loader = data.loader(test_idx, batch_size)
        if adabn:
            from stanet_common import adapt_batchnorm_to_target
            adapt_batchnorm_to_target(model, loader, device, log_fn=lambda *a, **k: None)

        for xe, xf, y in loader:
            vecs = three_methods(model, xe.to(device), xf.to(device), y.to(device), ig_steps)
            for b in range(vecs["attention"].shape[0]):
                for pair in METHOD_PAIRS:
                    u, v = vecs[pair[0]][b], vecs[pair[1]][b]
                    if np.allclose(u, u[0]) or np.allclose(v, v[0]):
                        continue            # a constant vector has no rank correlation
                    rho, _ = spearmanr(u, v)
                    if not np.isnan(rho):
                        pair_rhos[pair].append(float(rho))
        free(model, device)
    return pair_rhos


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_consensus_check")
    os.makedirs(out_dir, exist_ok=True)
    run_header("STA-Net MULTI-METHOD XAI CONSENSUS", ARGS.data_root, ARGS.results_dir,
               ARGS.adabn, ARGS.fnirs_scale,
               extra=f"IG steps = {ARGS.ig_steps}"
                     + ("" if ARGS.max_windows_per_fold is None
                        else f", capped at {ARGS.max_windows_per_fold} windows/fold"))

    per_subject, seg_lens = {}, set()
    for sid in parse_subjects(ARGS.subjects):
        try:
            data = SubjectData(ARGS.data_root, sid, ARGS.fnirs_scale)
            seg_lens.add(data.seg_len)
            rhos = subject_consensus(data, ARGS.results_dir, ARGS.device, ARGS.batch_size,
                                     ARGS.adabn, ARGS.ig_steps, ARGS.max_windows_per_fold)
            del data
            if not rhos[METHOD_PAIRS[0]]:
                print(f"[{sid}] SKIP -- no usable samples.")
                continue
            per_subject[sid] = rhos
            msg = "  ".join(f"{a[:4]}-{b[:4]}={np.mean(rhos[(a, b)]):+.2f}" for a, b in METHOD_PAIRS)
            print(f"[{sid}] n={len(rhos[METHOD_PAIRS[0]]):5d}  {msg}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")

    ids = sorted(per_subject)
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects produced usable consensus data.")

    rng = np.random.default_rng(ARGS.seed)
    print("\n" + "=" * 104)
    print(f"MULTI-METHOD XAI CONSENSUS -- {len(ids)} subjects, l = {sorted(seg_lens)} lagged fNIRS windows")
    print("=" * 104)
    print(f"{'pair':44s} {'mean rho':>9s} {'SD':>7s} {'95% CI (bootstrap)':>22s} {'Wilcoxon p (rho>0)':>19s}")
    summary = {"n_subjects": len(ids), "seg_len": sorted(seg_lens), "pairs": {}, "per_subject": {}}
    for pair in METHOD_PAIRS:
        means = np.array([float(np.mean(per_subject[s][pair])) for s in ids if per_subject[s][pair]])
        lo, hi = bootstrap_ci(means, ARGS.n_bootstrap, rng=rng)
        pv = float(wilcoxon(means, alternative="greater").pvalue) if len(means) >= 6 else np.nan
        name = f"{pair[0]} vs {pair[1]}"
        summary["pairs"][name] = {"n_subjects": int(len(means)), "mean_rho": float(means.mean()),
                                  "sd_rho": float(means.std(ddof=1)), "ci": [lo, hi],
                                  "wilcoxon_p_greater": pv}
        print(f"{name:44s} {means.mean():9.3f} {means.std(ddof=1):7.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22s} {pv:19.4g}")
    for s in ids:
        summary["per_subject"][s] = {f"{a}_{b}_mean_rho": (float(np.mean(per_subject[s][(a, b)]))
                                                           if per_subject[s][(a, b)] else np.nan)
                                     for a, b in METHOD_PAIRS}

    print("\nREADING")
    means = [summary["pairs"][f"{a} vs {b}"]["mean_rho"] for a, b in METHOD_PAIRS]
    ps = [summary["pairs"][f"{a} vs {b}"]["wilcoxon_p_greater"] for a, b in METHOD_PAIRS]
    if all(m > 0.5 for m in means) and all(p < 0.05 for p in ps):
        print("  All three architecturally independent methods -- a LEARNED attention weight and two POST-HOC\n"
              "  gradient methods, one local and one path-integrated -- agree STRONGLY on which lagged fNIRS\n"
              "  windows matter, each pairwise agreement significant. This is triangulated evidence for the\n"
              "  explanation's validity that none of the three reference papers provide: they report attention\n"
              "  alone and never check it against an independent method.")
    elif all(p < 0.05 for p in ps):
        print("  All three agree significantly better than chance, but only moderately. Report this as partial\n"
              "  convergence: the broad picture (which lagged windows matter most) is shared, but the\n"
              "  fine-grained ranking differs by method, so any single-method figure overstates its precision.")
    else:
        bad = [f"{a} vs {b}" for (a, b), pv in zip(METHOD_PAIRS, ps) if not (pv < 0.05)]
        print(f"  No significant agreement for: {', '.join(bad)}.")
        print("  This is exactly the 'attention is not explanation' risk the check was designed to catch. If the\n"
              "  LEARNED attention disagrees with the POST-HOC gradient methods, then STA-Net's attention map --\n"
              "  the only thing its paper shows -- is not a trustworthy stand-alone explanation. Which pair\n"
              "  disagrees pinpoints whether the problem is the learned attention itself or the gradient\n"
              "  methods' known noise sensitivity: if saliency-IG also disagrees, suspect the gradients.")

    with open(os.path.join(out_dir, "xai_consensus_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_consensus_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [f"{a}_{b}_mean_rho" for a, b in METHOD_PAIRS]
        w.writerow(["subject"] + cols)
        for s in ids:
            w.writerow([s] + [summary["per_subject"][s][c] for c in cols])
    print(f"\n[written] {out_dir}: xai_consensus_summary.json, xai_consensus_per_subject.csv")


if __name__ == "__main__":
    main()
