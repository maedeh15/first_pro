"""
xai_multimethod_consensus_mgfnet.py -- do THREE INDEPENDENT explanation
methods agree on what matters inside MGFNet, or does each tell a different story?

MGFNet version of xai_multimethod_consensus.py. TSFNet's unit of comparison
(21 fNIRS segments, EGTA attention) does not exist in MGFNet. MGFNet has two
LEARNED, architecture-internal weightings, and agreement is tested on both:

  LEVEL "coupling" -- 15 cells of the CGSCR coupling matrix C
                      (5 EEG Morlet bands x 3 fNIRS scales)
  LEVEL "gate"     -- the 6 CGSCR branch-gate weights
                      (EEG, HbO, HbR, EEG-HbO, EEG-HbR, HbO-HbR)

THE THREE METHODS (one importance value per unit, per held-out window)
  1. attention              the model's own learned weight (C_ij, or gate_k):
                            what the architecture CLAIMS it uses.
  2. input_gradient         |d logit_true / d unit|, one backward pass
                            (plain saliency, local).
  3. integrated_gradients   Sundararajan et al. 2017: |(u - u0) * mean grad|
                            along the straight path from a NEUTRAL baseline u0
                            (C0 = 1/3 per cell = uniform coupling, which is
                            the identity token scale; gate0 = 1/6 per branch =
                            uniform gating) to the real value, N_STEPS Riemann
                            steps. A zero baseline is NOT used: C = 0 would
                            delete every token, which is not "no information".
Both gradient methods target the hybrid head's true-class LOGIT and run the
model in eval mode exactly as at test time (discrete argmax routing).

NOTE on the gate level: given the window's branch features, the logit is
linear in the gate, so IG there equals (g - 1/6) * gradient exactly; it still
differs from plain saliency through the (g - 1/6) factor.

AGREEMENT METRIC: per-window Spearman correlation for the 3 method pairs at
each level, averaged per subject; group mean, SD, bootstrap CI of subject
means, one-sided Wilcoxon (rho > 0).

USAGE
=====
    python xai_multimethod_consensus_mgfnet.py --results-dir "E:\\...\\results_mgf_<hash>_seed42" ^
        --data-root "E:\\...\\MGFNET_PAPER_EXACT_nback_28ch"
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr

import mgfnet_xai_common as X

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_consensus_check")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--ig-steps", type=int, default=20)
p.add_argument("--n-bootstrap", type=int, default=2000)
p.add_argument("--min-subjects", type=int, default=6)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

METHODS = ("attention", "input_gradient", "integrated_gradients")
METHOD_PAIRS = [("attention", "input_gradient"), ("attention", "integrated_gradients"),
                ("input_gradient", "integrated_gradients")]
LEVELS = ("coupling", "gate")


def _target(logits, y):
    return logits.gather(1, y.view(-1, 1)).sum()


def three_methods(model, x_eeg, x_hbo, x_hbr, y, ig_steps):
    with torch.no_grad():
        feats, Ep, Fp, C = X.forward_parts(model, x_eeg, x_hbo, x_hbr)
        _, gate = X.logits_from_parts(model, feats, Ep, Fp, C)
    B = C.shape[0]
    out = {}

    # ---- level "coupling": unit = C_ij -----------------------------------
    Cg = C.clone().requires_grad_(True)
    lg, _ = X.logits_from_parts(model, feats, Ep, Fp, Cg)
    gC = torch.autograd.grad(_target(lg, y), Cg)[0]
    C0 = torch.full_like(C, 1.0 / C.shape[-1])
    tot = torch.zeros_like(C)
    for k in range(ig_steps):
        a = (k + 0.5) / ig_steps
        Ci = (C0 + a * (C - C0)).requires_grad_(True)
        li, _ = X.logits_from_parts(model, feats, Ep, Fp, Ci)
        tot += torch.autograd.grad(_target(li, y), Ci)[0]
    out["coupling"] = {"attention": C.reshape(B, -1),
                       "input_gradient": gC.abs().reshape(B, -1),
                       "integrated_gradients": ((C - C0) * tot / ig_steps).abs().reshape(B, -1)}

    # ---- level "gate": unit = gate_k --------------------------------------
    gg = gate.clone().requires_grad_(True)
    lg, _ = X.logits_from_parts(model, feats, Ep, Fp, C, gate=gg)
    gG = torch.autograd.grad(_target(lg, y), gg)[0]
    g0 = torch.full_like(gate, 1.0 / gate.shape[-1])
    tot = torch.zeros_like(gate)
    for k in range(ig_steps):
        a = (k + 0.5) / ig_steps
        gi = (g0 + a * (gate - g0)).requires_grad_(True)
        li, _ = X.logits_from_parts(model, feats, Ep, Fp, C, gate=gi)
        tot += torch.autograd.grad(_target(li, y), gi)[0]
    out["gate"] = {"attention": gate, "input_gradient": gG.abs(),
                   "integrated_gradients": ((gate - g0) * tot / ig_steps).abs()}
    return {lv: {m: v.detach().cpu().numpy() for m, v in d.items()} for lv, d in out.items()}


def subject_consensus(sid):
    full = X.load_subject(ARGS.data_root, sid)
    rhos = {lv: {pair: [] for pair in METHOD_PAIRS} for lv in LEVELS}
    for test_session in (1, 2, 3):
        fold = X.load_fold(ARGS.results_dir, sid, test_session, ARGS.device)
        if fold is None:
            continue
        model, norm = fold
        model.eval()
        loader, _ = X.test_loader(full, test_session, ARGS.batch_size)
        for batch in loader:
            x_eeg, x_hbo, x_hbr, y = X.to_device(batch, ARGS.device, norm)
            vecs = three_methods(model, x_eeg, x_hbo, x_hbr, y, ARGS.ig_steps)
            for lv in LEVELS:
                for b in range(y.shape[0]):
                    for m1, m2 in METHOD_PAIRS:
                        a, c = vecs[lv][m1][b], vecs[lv][m2][b]
                        if np.allclose(a, a[0]) or np.allclose(c, c[0]):
                            continue                      # constant vector: rank correlation undefined
                        rho, _ = spearmanr(a, c)
                        if not np.isnan(rho):
                            rhos[lv][(m1, m2)].append(rho)
        del model
        if str(ARGS.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return rhos


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_consensus_check")
    os.makedirs(out_dir, exist_ok=True)
    per_subject = {}
    for sid in X.parse_subjects(ARGS.subjects):
        try:
            r = subject_consensus(sid)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            continue
        if not any(r[lv][pair] for lv in LEVELS for pair in METHOD_PAIRS):
            print(f"[{sid}] SKIP -- no usable windows")
            continue
        per_subject[sid] = r
        msg = " | ".join(f"{lv}: " + " ".join(f"{a[:4]}-{b[:4]}={np.mean(r[lv][(a, b)]) if r[lv][(a, b)] else np.nan:+.2f}"
                                               for a, b in METHOD_PAIRS) for lv in LEVELS)
        print(f"[{sid}] {msg}")
    ids = sorted(per_subject)
    if len(ids) < ARGS.min_subjects:
        sys.exit(f"ERROR: only {len(ids)} subjects produced consensus data (< {ARGS.min_subjects}).")

    hline = "=" * 104
    summary = {"n_subjects": len(ids), "levels": {}, "per_subject": {}}
    for lv in LEVELS:
        print("\n" + hline)
        print(f"MULTI-METHOD XAI CONSENSUS -- level '{lv}' ({15 if lv == 'coupling' else 6} units), {len(ids)} subjects")
        print(hline)
        print(f"{'pair':44s} {'mean rho':>9s} {'SD':>7s} {'95% CI (subj. means)':>24s} {'Wilcoxon p (rho>0)':>20s}")
        summary["levels"][lv] = {}
        for pair in METHOD_PAIRS:
            sm = np.array([np.mean(per_subject[s][lv][pair]) for s in ids if per_subject[s][lv][pair]])
            lo, hi = X.bootstrap_ci(sm, ARGS.n_bootstrap)
            pv = X.safe_wilcoxon(sm, alternative="greater") if len(sm) >= 6 else np.nan
            name = f"{pair[0]} vs {pair[1]}"
            summary["levels"][lv][name] = {"n_subjects": int(len(sm)),
                                           "mean_rho": float(sm.mean()) if len(sm) else np.nan,
                                           "sd_rho": float(sm.std(ddof=1)) if len(sm) > 1 else np.nan,
                                           "ci": [lo, hi], "wilcoxon_p_greater": pv}
            print(f"{name:44s} {summary['levels'][lv][name]['mean_rho']:9.3f} "
                  f"{summary['levels'][lv][name]['sd_rho']:7.3f} [{lo:6.3f}, {hi:6.3f}]{'':8s} {pv:20.4g}")

        means = [summary["levels"][lv][f"{a} vs {b}"]["mean_rho"] for a, b in METHOD_PAIRS]
        ps = [summary["levels"][lv][f"{a} vs {b}"]["wilcoxon_p_greater"] for a, b in METHOD_PAIRS]
        print("READING")
        if all(m > 0.5 for m in means) and all(q < 0.05 for q in ps):
            print(f"  All three methods agree STRONGLY on the '{lv}' ranking, every pair significant: "
                  f"triangulated, method-independent evidence the learned {lv} weights are a valid explanation.")
        elif all(q < 0.05 for q in ps):
            print(f"  All three agree better than chance on '{lv}' but only moderately: the broad picture is "
                  f"shared, the fine ranking is method-dependent -- a single-method figure overstates precision.")
        else:
            bad = [f"{a} vs {b}" for (a, b), q in zip(METHOD_PAIRS, ps) if not (q < 0.05)]
            print(f"  No significant agreement for: {bad}. If 'attention' is in that pair, the learned {lv} "
                  f"weight alone is NOT a trustworthy stand-alone explanation ('attention is not explanation').")

    for s in ids:
        summary["per_subject"][s] = {f"{lv}_{a}_{b}_mean_rho": (float(np.mean(per_subject[s][lv][(a, b)]))
                                                                 if per_subject[s][lv][(a, b)] else np.nan)
                                     for lv in LEVELS for a, b in METHOD_PAIRS}
    with open(os.path.join(out_dir, "xai_consensus_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=X.nan_float)
    with open(os.path.join(out_dir, "xai_consensus_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [f"{lv}_{a}_{b}_mean_rho" for lv in LEVELS for a, b in METHOD_PAIRS]
        w.writerow(["subject"] + cols)
        for s in ids:
            w.writerow([s] + [summary["per_subject"][s][c] for c in cols])
    print(f"\n[written] {out_dir}: xai_consensus_summary.json, xai_consensus_per_subject.csv")


if __name__ == "__main__":
    main()
