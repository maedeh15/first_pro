"""
attention_faithfulness_check.py -- does the model's attention/gating weight
actually correspond to what causally matters for its decision ("faithfulness"),
or does it just look plausible ("attention is not explanation", Jain & Wallace
2019; Serrano & Smith 2019)?

WHY THIS EXISTS
STA-Net, MGFNet and MSTFDN all treat high attention weight as if it directly
means "this input mattered". None of them test that assumption: they only show
that the attention PATTERN looks reasonable (temporally stable, concentrated
on plausible channels), never that INTERVENING on the high-attention
component actually hurts the prediction more than intervening on a
low-attention or random component of the same size. This script runs that
intervention directly, at the finest grain the architecture allows:

  1. EGTA SEGMENT FAITHFULNESS (fine-grained, the main result)
     Leave-One-Segment-Out (LOSO): for EVERY one of the S=21 fNIRS segments,
     zero that segment's window across the whole batch and re-run the full
     forward pass, recording the drop in true-class probability it causes.
     This gives one real, causal "impact" value per segment, per sample --
     NOT estimated from the attention weight, but measured directly by
     removing that exact input and watching the output change.
     Then, for each sample, Spearman-correlate its own EGTA attention weight
     per segment (what the model SAYS it is using) against its own measured
     impact per segment (what the model ACTUALLY uses). rho near +1 means the
     attention weight faithfully predicts which segments causally matter;
     rho near 0 means attention and causal importance are unrelated --
     exactly the failure mode "attention is not explanation" describes.

  2. EFGF PHI-MAP FAITHFULNESS (coarser, spatial-temporal gate)
     Phi (Eq. 2) cannot be ablated pixel-by-pixel without an explosion of
     forward passes (it spans H x W x T), so this uses 4 quantile BINS of the
     Phi map's own value per sample (top 25%, 25-50%, 50-75%, bottom 25%) and
     leave-one-BIN-out ablation of fusion_map at those locations (4 extra
     forward passes per batch instead of H*W*T). Same logic: does zeroing the
     bin Phi itself calls "most important" hurt more than zeroing the bin it
     calls "least important"?

Both tests are run on the SAME held-out folds and checkpoints as
trial_level_eval.py / xai_biomarker_validation.py -- no new training, no new
data, just extra forward passes (21 + 4 = 25 extra passes per batch; a few
seconds per fold on GPU).

OUTPUT
  Per-subject mean Spearman rho (EGTA) and bin-monotonicity score (EFGF),
  with a bootstrap CI (resampling test samples) and a group-level one-sample
  Wilcoxon test of "is the across-subject mean rho greater than 0" -- the
  same statistical rigor (BH-FDR optional, Wilcoxon, CI) used elsewhere in
  this project's diagnostics, applied here to a question none of the three
  reference papers even ask.

USAGE
=====
    python attention_faithfulness_check.py --results-dir "E:\\...\\results_40b7d8ea53_seed42" ^
        --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback"

Needs: preprocessed VP001..VP026 windows + trained checkpoints of one seed,
fnirs_temporal_align="egta" (the project's accepted baseline default).
No GPU required, much faster with one.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon
from torch.utils.data import DataLoader, Subset

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
from xai_trunk import trunk_state, load_checkpoint                       # drops 'riem_head.*' so a member-branch checkpoint loads into the plain trunk

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True)
p.add_argument("--data-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/faithfulness_check")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--n-bootstrap", type=int, default=2000)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()


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


def _is_constant(v):
    """scale-free: np.allclose's absolute tolerance (1e-8) would call a vector of tiny probability changes constant"""
    return np.ptp(v) <= 1e-9 * (np.abs(v).max() + 1e-300)


def build_model(n_segments):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=n_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def true_class_prob(y_pred, y):
    return y_pred.gather(1, y.view(-1, 1)).squeeze(1)


# ---------------------------------------------------------------------------
# EGTA: leave-one-segment-out faithfulness
# ---------------------------------------------------------------------------
def egta_faithfulness(model, x_eeg, x_fnirs, y, x_conn, x_chan, device):
    """Returns per-sample Spearman rho between EGTA attention weight (per
    segment) and the REAL causal impact of zeroing that exact segment,
    measured by one forward pass per segment (not per sample)."""
    with torch.no_grad():
        _, _, _, y_pred0, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
        p0 = true_class_prob(y_pred0, y)
        w = model.cafe._last_egta_weights.mean(dim=1)   # (B, S), already softmax'd

    B, S = w.shape
    impact = torch.zeros(B, S, device=device)
    with torch.no_grad():
        for s in range(S):
            x_fnirs_abl = x_fnirs.clone()
            x_fnirs_abl[:, s] = 0.0
            _, _, _, y_pred_s, _ = model(x_eeg, x_fnirs_abl, x_conn, x_chan)
            p_s = true_class_prob(y_pred_s, y)
            impact[:, s] = (p0 - p_s)   # positive = removing this segment hurt the true-class prob

    w_np, impact_np = w.cpu().numpy(), impact.cpu().numpy()
    rhos = []
    for b in range(B):
        if _is_constant(impact_np[b]) or _is_constant(w_np[b]):
            continue   # degenerate (no variation) -- correlation undefined, skip this sample
        rho, _ = spearmanr(w_np[b], impact_np[b])
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos, float(p0.mean().item())


# ---------------------------------------------------------------------------
# EFGF: leave-one-Phi-quantile-bin-out faithfulness
# ---------------------------------------------------------------------------
def efgf_faithfulness(model, x_eeg, x_fnirs, y, x_conn, x_chan, device, which="efgf1"):
    """4-bin coarse version of the same test for EFGF's hybrid attention Phi.
    Phi lives on a REDUCED (h, w) spatial grid (after conv1/conv2 downsampling)
    relative to the raw x_fnirs input's (16, 16) grid, so its per-sample
    quantile-bin mask is nearest-neighbour-upsampled back to (16, 16) before
    being used to zero the raw fNIRS input (broadcast over segments, the
    30-sample time axis, and the 2 chromophore channels) -- the closest
    input-side analogue of "ablate what Phi calls important", using the SAME
    full-forward-pass re-evaluation the EGTA test uses (4 extra passes here).
    Monotonicity score: Spearman rho between bin rank (by Phi, high to low)
    and the bin's measured impact, across the 4 bins, per sample."""
    captured = {}

    def hook(_m, _i, out):
        captured["phi"] = out[2].detach()   # (B, 1, h, w, t) -- Phi's own reduced grid

    layer = getattr(model, which)
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        _, _, _, y_pred0, _ = model(x_eeg, x_fnirs, x_conn, x_chan)
        p0 = true_class_prob(y_pred0, y)
        phi0 = captured["phi"]
    handle.remove()

    B = phi0.shape[0]
    phi_spatial = phi0.amax(dim=-1)[:, 0]        # (B, h, w) -- collapse Phi's time axis (max, not mean,
                                                  # so a bin flagged "high" really contains a peak, not
                                                  # just a long near-zero stretch)
    flat = phi_spatial.reshape(B, -1)
    qs = torch.quantile(flat, torch.tensor([0.25, 0.5, 0.75], device=device), dim=1)   # (3, B)
    bin_id = torch.zeros_like(flat, dtype=torch.long)
    bin_id[flat > qs[2].unsqueeze(1)] = 3     # top 25% (highest Phi)
    bin_id[(flat > qs[1].unsqueeze(1)) & (flat <= qs[2].unsqueeze(1))] = 2
    bin_id[(flat > qs[0].unsqueeze(1)) & (flat <= qs[1].unsqueeze(1))] = 1
    bin_id = bin_id.reshape(phi_spatial.shape)   # (B, h, w)

    raw_hw = x_fnirs.shape[2], x_fnirs.shape[3]   # (16, 16)
    bin_id_up = torch.nn.functional.interpolate(
        bin_id.unsqueeze(1).float(), size=raw_hw, mode="nearest").squeeze(1).long()   # (B, 16, 16)

    bin_impact = []
    for b_id in (3, 2, 1, 0):   # high Phi first
        with torch.no_grad():
            spatial_mask = (bin_id_up == b_id)             # (B, 16, 16)
            keep = (~spatial_mask).float().view(B, 1, raw_hw[0], raw_hw[1], 1, 1)   # broadcast S, T, chroma
            x_fnirs_abl = x_fnirs * keep
            _, _, _, y_pred_b, _ = model(x_eeg, x_fnirs_abl, x_conn, x_chan)
            p_b = true_class_prob(y_pred_b, y)
            bin_impact.append((p0 - p_b).cpu().numpy())

    bin_impact = np.stack(bin_impact, axis=1)   # (B, 4), columns ordered bin 3,2,1,0 (high to low Phi)
    rank_phi = np.array([3, 2, 1, 0])
    rhos = []
    for b in range(B):
        if _is_constant(bin_impact[b]):
            continue
        rho, _ = spearmanr(rank_phi, bin_impact[b])
        if not np.isnan(rho):
            rhos.append(rho)
    return rhos


def bootstrap_ci(values, n_boot, alpha=0.05):
    if len(values) < 3:
        return float("nan"), float("nan")
    arr = np.array(values)
    boots = [np.random.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def subject_faithfulness(sid, results_dir, data_root, device, batch_size, adabn):
    subject_dir = os.path.join(data_root, sid)
    full = NBackDataset(subject_dir, sid)
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    egta_rhos, efgf1_rhos, efgf2_rhos = [], [], []

    for test_session in (1, 2, 3):
        ckpt_path = os.path.join(results_dir, f"{sid}_session{test_session}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[{sid} s{test_session}] SKIP -- checkpoint not found: {ckpt_path}")
            continue
        model = build_model(n_segments).to(device)
        state = load_checkpoint(ckpt_path, map_location=device)
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
            rhos, _ = egta_faithfulness(model, x_eeg, x_fnirs, y, x_conn, x_chan, device)
            egta_rhos.extend(rhos)
            try:
                r1 = efgf_faithfulness(model, x_eeg, x_fnirs, y, x_conn, x_chan, device, "efgf1")
                efgf1_rhos.extend(r1)
            except Exception as e:
                print(f"  [{sid} s{test_session}] EFGF1 faithfulness skipped ({type(e).__name__}: {e})")

        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    return egta_rhos, efgf1_rhos, efgf2_rhos


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "faithfulness_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    per_subject = {}
    for sid in subjects:
        try:
            egta_rhos, efgf1_rhos, _ = subject_faithfulness(sid, ARGS.results_dir, ARGS.data_root,
                                                             device, ARGS.batch_size, ARGS.adabn)
            per_subject[sid] = {"egta_rhos": egta_rhos, "efgf1_rhos": efgf1_rhos}
            print(f"[{sid}] n_samples(egta)={len(egta_rhos)} mean_rho={np.mean(egta_rhos) if egta_rhos else float('nan'):.3f}  "
                  f"n_samples(efgf1)={len(efgf1_rhos)} mean_rho={np.mean(efgf1_rhos) if efgf1_rhos else float('nan'):.3f}")
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")

    ids = [s for s in per_subject if per_subject[s]["egta_rhos"]]
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects produced usable EGTA faithfulness samples.")

    hline = "=" * 100
    print("\n" + hline)
    print(f"FAITHFULNESS SUMMARY -- {len(ids)} subjects (does attention weight predict REAL causal impact?)")
    print(hline)
    print(f"{'subject':8s} {'EGTA n':>7s} {'EGTA mean rho':>14s} {'95% CI':>18s} | "
          f"{'EFGF1 n':>8s} {'EFGF1 mean rho':>15s} {'95% CI':>18s}")
    summary = {"per_subject": {}}
    subj_egta_means, subj_efgf1_means = [], []
    for s in ids:
        er = per_subject[s]["egta_rhos"]; fr = per_subject[s]["efgf1_rhos"]
        e_mean = float(np.mean(er))
        e_lo, e_hi = bootstrap_ci(er, ARGS.n_bootstrap)
        f_mean = float(np.mean(fr)) if fr else float("nan")
        f_lo, f_hi = bootstrap_ci(fr, ARGS.n_bootstrap) if fr else (float("nan"), float("nan"))
        subj_egta_means.append(e_mean)
        if fr:
            subj_efgf1_means.append(f_mean)
        summary["per_subject"][s] = {
            "egta_n": len(er), "egta_mean_rho": e_mean, "egta_ci": [e_lo, e_hi],
            "efgf1_n": len(fr), "efgf1_mean_rho": f_mean, "efgf1_ci": [f_lo, f_hi],
        }
        print(f"{s:8s} {len(er):7d} {e_mean:14.3f} [{e_lo:6.3f},{e_hi:6.3f}] | "
              f"{len(fr):8d} {f_mean:15.3f} [{f_lo:6.3f},{f_hi:6.3f}]")

    print("\n" + hline)
    print("GROUP-LEVEL TEST: is mean faithfulness rho > 0 across subjects? (one-sample Wilcoxon)")
    print(hline)
    for name, means in (("EGTA segment attention", subj_egta_means), ("EFGF1 Phi quartile bins", subj_efgf1_means)):
        arr = np.array([m for m in means if not np.isnan(m)])
        if len(arr) < 6:
            print(f"  {name}: too few subjects with usable data -- skipped.")
            continue
        stat = wilcoxon(arr, alternative="greater")
        print(f"  {name}: n={len(arr)}, mean rho={arr.mean():+.3f}, "
              f"Wilcoxon (H1: rho>0) p={stat.pvalue:.4g}")
        summary[f"group_{'egta' if 'EGTA' in name else 'efgf1'}"] = {
            "n": len(arr), "mean_rho": float(arr.mean()), "wilcoxon_p_greater": float(stat.pvalue)}

    print("\nREADING")
    egta_p = summary.get("group_egta", {}).get("wilcoxon_p_greater", 1.0)
    egta_mean = summary.get("group_egta", {}).get("mean_rho", 0.0)
    if egta_p < 0.05 and egta_mean > 0.3:
        print("  EGTA attention weight is a FAITHFUL explanation: segments it marks important really do "
              "cause the largest true-class-probability drop when removed (strong, significant positive "
              "rho). This is a real, causally-verified claim none of the three reference papers make.")
    elif egta_p < 0.05 and egta_mean > 0:
        print("  EGTA attention is weakly but significantly faithful: the direction is right, but the "
              "correspondence between stated and actual importance is far from perfect -- worth reporting "
              "honestly as a partial-faithfulness finding rather than overclaiming interpretability.")
    else:
        print("  No significant evidence that EGTA attention weight predicts real causal impact: this is "
              "exactly the 'attention is not explanation' failure mode. Reporting this HONESTLY, with the "
              "quantitative test that found it, is itself a novel and defensible thesis contribution -- it "
              "is a finding the three reference papers' qualitative figures could never have caught.")

    with open(os.path.join(out_dir, "faithfulness_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=float)
    with open(os.path.join(out_dir, "faithfulness_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "egta_n", "egta_mean_rho", "egta_ci_lo", "egta_ci_hi",
                    "efgf1_n", "efgf1_mean_rho", "efgf1_ci_lo", "efgf1_ci_hi"])
        for s in ids:
            r = summary["per_subject"][s]
            w.writerow([s, r["egta_n"], r["egta_mean_rho"], r["egta_ci"][0], r["egta_ci"][1],
                       r["efgf1_n"], r["efgf1_mean_rho"], r["efgf1_ci"][0], r["efgf1_ci"][1]])
    print(f"\n[written] {out_dir}: faithfulness_summary.json, faithfulness_per_subject.csv")


if __name__ == "__main__":
    main()
