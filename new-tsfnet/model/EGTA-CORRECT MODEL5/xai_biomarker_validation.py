"""
xai_biomarker_validation.py -- is the model's own attention/gating a valid
cognitive-load biomarker, or just a plausible-looking picture?

WHY THIS EXISTS (the gap in STA-Net / MGFNet / MSTFDN's XAI sections)
All three published baselines this project re-implements show attention maps
and say they "look reasonable" (e.g. STA-Net: "kappa_fnirs stays flat over
time -- EGTA is working"). None of them run a single formal statistical test
against an established neuroscience ground truth. This script applies EXACTLY
the same statistical machinery already used in capacity_ceiling_check.py
(per-subject exact permutation test, Cohen's d, BH-FDR across subjects,
cross-session trait-consistency, group Wilcoxon, Cochran's Q / I^2
heterogeneity) to the MODEL'S OWN internal signals instead of raw
physiological features -- and then correlates the two, subject by subject.
If the model's attention tracks the same subjects' same direction of effect
as the real HbO/theta/P300 biomarkers, that is formal, quantitative evidence
the network has learned something neuroscientifically real, not just a
classification shortcut. If it does not, that is equally worth reporting --
either way this closes the "does XAI output mean anything" gap the three
reference papers leave open, which is the thesis's own stated differentiator
("EEG-fNIRS biomarker extraction" + "explainable deep learning").

MODEL-DERIVED SIGNALS EXTRACTED (per window, averaged to per-block/trial via
NBackDataset's trial_ids -- same unit of analysis as trial_level_eval.py)
  egta_centroid   attention-weighted mean fNIRS-segment index (0..S-1) from
                  CAFELayer._egta_pool's softmax weights (model.cafe's
                  _last_egta_weights, averaged over heads). HYPOTHESIS: higher
                  cognitive load -> slower/delayed hemodynamic onset -> the
                  model should need to attend LATER in the segment sequence to
                  find the task-relevant fNIRS response, i.e. centroid UP with
                  load -- directly analogous to, and testable against, this
                  project's own hbo_pfc/theta_fm/p300_pz literature predictions
                  in capacity_ceiling_check.py.
  egta_entropy    Shannon entropy of that same attention distribution
                  (selectivity: peaked vs. near-uniform). EXPLORATORY -- no
                  literature sign is imposed; reported descriptively only.
  efgf1_phi_mean / efgf2_phi_mean   mean value of EFGF's hybrid attention map
                  Phi (Eq. 2, the THIRD return value of EFGFLayer.forward,
                  captured via a forward hook -- no model-code change needed)
                  at stage 1 / stage 2. EXPLORATORY, no imposed sign.

Only works on checkpoints trained with fnirs_temporal_align="egta" (the
project's accepted baseline default) -- egta_centroid/entropy need
model.cafe._last_egta_weights, which only exists in that mode.

STATISTICS -- IDENTICAL engine to capacity_ceiling_check.py (same functions,
copied rather than imported so this file stays runnable standalone):
  per subject: delta = mean(3-back) - mean(2-back) of the model signal, Cohen's
  d, exact permutation p (all within-session relabelings enumerated, not
  sampled), session-trait-consistency (how many of 3 sessions agree in sign).
  BH-FDR q across subjects. Group: Wilcoxon on delta, Cochran's Q / I^2.

CROSS-VALIDATION AGAINST REAL PHYSIOLOGY (the key novel step) -- optionally
pass --capacity-summary pointing at capacity_ceiling_check.py's own
capacity_summary.json for the SAME subjects; this script then Spearman-
correlates, per subject, the model's egta_centroid effect size against that
subject's real hbo_pfc / theta_fm / p300_pz effect sizes. A significant
positive correlation (for hbo_pfc/theta_fm; negative for p300_pz, matching
their literature signs) is direct evidence the attention mechanism is
tracking real inter-subject physiological variation, not an artifact.

USAGE
=====
    python xai_biomarker_validation.py --results-dir "E:\\...\\results_40b7d8ea53_seed42" ^
        --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback" ^
        --capacity-summary "E:\\...\\capacity_check\\capacity_summary.json"

Needs: preprocessed VP001..VP026 windows + the trained checkpoints of ONE
seed (any seed; repeat with other seeds for the cross-seed stability check
that is a separate, complementary piece of work). No GPU required, much
faster with one -- dominated by one forward pass per fold, same cost as
trial_level_eval.py.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest, chi2, spearmanr, wilcoxon
from torch.utils.data import DataLoader, Subset

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
from xai_trunk import trunk_state, load_checkpoint                       # drops 'riem_head.*' so a member-branch checkpoint loads into the plain trunk

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True,
              help="Folder with VPxxx_session{1,2,3}_best.pt (one results_<hash>_seed<N> run, "
                   "fnirs_temporal_align='egta').")
p.add_argument("--data-root", required=True, help="Folder with VP001..VP026 preprocessed .npy files.")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_biomarker_check")
p.add_argument("--capacity-summary", default=None,
              help="capacity_ceiling_check.py's capacity_summary.json, for the physiology cross-check.")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()

SIGNALS = ["egta_centroid", "egta_entropy", "efgf1_phi_mean", "efgf2_phi_mean"]
EXPECTED_SIGN = {"egta_centroid": +1}   # only this one has a literature-grounded directional prediction
LABELS = {"egta_centroid": "EGTA attention centroid (segment index)",
          "egta_entropy": "EGTA attention entropy",
          "efgf1_phi_mean": "EFGF-1 hybrid attention Phi (mean)",
          "efgf2_phi_mean": "EFGF-2 hybrid attention Phi (mean)"}


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


# ---------------------------------------------------------------------------
# Statistics engine -- copied verbatim from capacity_ceiling_check.py so this
# file stays runnable standalone (see that file for the full rationale).
# ---------------------------------------------------------------------------
def exact_perm_p(v, lab, sess):
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


# ---------------------------------------------------------------------------
# Model forward pass with hooks to capture EGTA weights + EFGF phi
# ---------------------------------------------------------------------------
def build_model(fnirs_num_segments):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=fnirs_num_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def collect_fold_signals(full, test_idx, ckpt_path, device, batch_size, adabn):
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    model = build_model(n_segments).to(device)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(trunk_state(state), strict=True)
    model.eval()

    phi1_store, phi2_store = [], []

    def hook1(_m, _i, out):
        phi1_store.append(out[2].detach())

    def hook2(_m, _i, out):
        phi2_store.append(out[2].detach())
    h1 = model.efgf1.register_forward_hook(hook1)
    h2 = model.efgf2.register_forward_hook(hook2)

    test_ds = Subset(full, test_idx)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    if adabn:
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)
        phi1_store.clear(); phi2_store.clear()   # AdaBN's own forward passes don't count

    egta_centroid, egta_entropy = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            model(x_eeg, x_fnirs, x_conn, x_chan)
            w = model.cafe._last_egta_weights          # (B, h, S), already softmax'd + detached
            w_mean = w.mean(dim=1)                       # average over heads -> (B, S)
            idx = torch.arange(w_mean.shape[1], device=w_mean.device).float()
            centroid = (w_mean * idx).sum(dim=1) / (w_mean.sum(dim=1) + 1e-12)
            entropy = -(w_mean * (w_mean + 1e-12).log()).sum(dim=1)
            egta_centroid.append(centroid.cpu().numpy())
            egta_entropy.append(entropy.cpu().numpy())

    h1.remove(); h2.remove()
    centroid_all = np.concatenate(egta_centroid)
    entropy_all = np.concatenate(egta_entropy)
    n_total = centroid_all.shape[0]

    def phi_mean_per_sample(store):
        if not store:
            return np.full(n_total, np.nan)
        cat = torch.cat(store, dim=0)                      # (N, 1, H, W, T)
        return cat.reshape(cat.shape[0], -1).mean(dim=1).cpu().numpy()

    phi1 = phi_mean_per_sample(phi1_store)
    phi2 = phi_mean_per_sample(phi2_store)

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    return {"egta_centroid": centroid_all, "egta_entropy": entropy_all,
            "efgf1_phi_mean": phi1, "efgf2_phi_mean": phi2}


def trial_mean(values, trial_ids):
    trials = np.unique(trial_ids)
    return trials, np.array([values[trial_ids == t].mean() for t in trials])


def subject_blocks(sid, results_dir, data_root, device, batch_size, adabn):
    subject_dir = os.path.join(data_root, sid)
    full = NBackDataset(subject_dir, sid)
    rows = []
    for test_session in (1, 2, 3):
        ckpt_path = os.path.join(results_dir, f"{sid}_session{test_session}_best.pt")
        if not os.path.isfile(ckpt_path):
            print(f"[{sid} s{test_session}] SKIP -- checkpoint not found: {ckpt_path}")
            continue
        test_idx = np.flatnonzero(full.session_ids == test_session)
        signals = collect_fold_signals(full, test_idx, ckpt_path, device, batch_size, adabn)
        labels_w = full.labels[test_idx]
        trial_ids_w = full.trial_ids[test_idx]
        trials, lab_per_trial = trial_mean(labels_w, trial_ids_w)   # labels are constant per trial
        lab_per_trial = np.round(lab_per_trial).astype(int)
        for sig_name in SIGNALS:
            _, v_per_trial = trial_mean(signals[sig_name], trial_ids_w)
            for lab, v in zip(lab_per_trial, v_per_trial):
                rows.append({"session": test_session, "label": int(lab), "signal": sig_name, "value": float(v)})
    return rows


def load_capacity_summary(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for m in ("hbo_pfc", "theta_fm", "p300_pz"):
        out[m] = {s: v["d"] for s, v in d["per_subject"][m].items()}
    return out


def main():
    device = ARGS.device
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_biomarker_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    all_rows = {}
    for sid in subjects:
        try:
            all_rows[sid] = subject_blocks(sid, ARGS.results_dir, ARGS.data_root, device,
                                           ARGS.batch_size, ARGS.adabn)
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    ids = sorted(s for s in all_rows if all_rows[s])
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects processed -- check --results-dir / --data-root.")

    res = {sig: {} for sig in SIGNALS}
    for s in ids:
        rows = all_rows[s]
        lab = np.array([r["label"] for r in rows])
        sess = np.array([r["session"] for r in rows])
        sig_arr = np.array([r["signal"] for r in rows])
        val = np.array([r["value"] for r in rows])
        for sig_name in SIGNALS:
            m = sig_arr == sig_name
            v, l, sv = val[m], lab[m], sess[m]
            v2, v3 = v[(l == 1)], v[(l == 2)]
            if len(v2) < 2 or len(v3) < 2 or np.isnan(v2).any() or np.isnan(v3).any():
                res[sig_name][s] = {"delta": np.nan, "d": np.nan, "se": np.nan, "p": np.nan, "sessions": 0}
                continue
            delta = v3.mean() - v2.mean()
            sp = np.sqrt(((len(v2) - 1) * v2.var(ddof=1) + (len(v3) - 1) * v3.var(ddof=1))
                         / (len(v2) + len(v3) - 2))
            per_sess = [np.nanmean(v[(l == 2) & (sv == k)]) - np.nanmean(v[(l == 1) & (sv == k)])
                       for k in np.unique(sv)]
            res[sig_name][s] = {
                "delta": float(delta), "d": float(delta / (sp + 1e-30)),
                "se": float(np.sqrt(v2.var(ddof=1) / len(v2) + v3.var(ddof=1) / len(v3))),
                "p": exact_perm_p(v, l, sv),
                "sessions": int(sum(np.sign(x) == np.sign(delta) for x in per_sess if x == x)),
            }
    for sig_name in SIGNALS:
        q = bh([res[sig_name][s]["p"] for s in ids])
        for s, qq in zip(ids, q):
            res[sig_name][s]["q"] = float(qq)

    summary = {"n_subjects": len(ids), "signals": {}}
    print("\n" + "=" * 100)
    print(f"XAI-AS-BIOMARKER VALIDATION -- model-internal attention/gate signals, 3-back minus 2-back, "
          f"{len(ids)} subjects")
    print("(statistics identical to capacity_ceiling_check.py: exact within-session permutation, BH-FDR, "
          "Wilcoxon, Cochran's Q/I^2)")
    print("=" * 100)
    print(f"{'signal':20s} {'expect':>6s} | {'as exp.':>7s} {'reversed':>8s} | "
          f"{'sig as exp.':>11s} {'sig rev.':>8s} | {'group p':>8s} | {'I^2':>5s} {'Q p':>8s} | "
          f"{'stable 3/3':>10s}")
    for sig_name in SIGNALS:
        d_ = np.array([res[sig_name][s]["delta"] for s in ids]); se = np.array([res[sig_name][s]["se"] for s in ids])
        q_ = np.array([res[sig_name][s]["q"] for s in ids]); ok = ~np.isnan(d_)
        exp = EXPECTED_SIGN.get(sig_name, 0)
        if exp == 0:
            as_exp = rev = sig_exp = sig_rev = int(-1)   # not applicable, no literature sign to check against
        else:
            as_exp = int((np.sign(d_[ok]) == exp).sum()); rev = int((np.sign(d_[ok]) == -exp).sum())
            sig_exp = int(((q_ < 0.05) & (np.sign(d_) == exp)).sum())
            sig_rev = int(((q_ < 0.05) & (np.sign(d_) == -exp)).sum())
        gp = wilcoxon(d_[ok]).pvalue if ok.sum() >= 6 else np.nan
        w = 1 / se[ok] ** 2; mu = (w * d_[ok]).sum() / w.sum()
        Q = float((w * (d_[ok] - mu) ** 2).sum()); df = ok.sum() - 1
        I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0; Qp = float(chi2.sf(Q, df)) if df > 0 else np.nan
        stable = int(sum(res[sig_name][s]["sessions"] == 3 for s in ids))
        summary["signals"][sig_name] = {
            "expected_sign": exp, "n_as_expected": as_exp, "n_reversed": rev,
            "n_sig_as_expected": sig_exp, "n_sig_reversed": sig_rev,
            "sign_test_p": (float(binomtest(as_exp, as_exp + rev, 0.5).pvalue)
                           if exp != 0 and as_exp + rev else np.nan),
            "group_wilcoxon_p": float(gp), "I2": I2, "Q": Q, "Q_p": Qp,
            "n_sign_stable_all_3_sessions": stable,
        }
        exp_str = "+" if exp > 0 else "-" if exp < 0 else "n/a"
        print(f"{sig_name:20s} {exp_str:>6s} | {as_exp:7d} {rev:8d} | {sig_exp:11d} {sig_rev:8d} | "
              f"{gp:8.3g} | {I2:5.2f} {Qp:8.3g} | {stable:6d}/{len(ids)}")

    print("\nREADING (egta_centroid, the only signal with a literature-grounded directional hypothesis):")
    ec = summary["signals"]["egta_centroid"]
    if ec["n_sig_as_expected"] >= 3 and ec["group_wilcoxon_p"] < 0.05:
        print("  EGTA attention shifts to LATER fNIRS segments at 3-back vs 2-back, matching the predicted "
              "slower hemodynamic onset under higher working-memory load -- the model's learned attention "
              "is tracking a real, literature-consistent physiological pattern, not an arbitrary shortcut.")
    elif ec["n_sig_reversed"] >= 3:
        print("  EGTA attention shifts EARLIER at higher load for several subjects -- opposite to the "
              "literature prediction; worth checking whether those are the same subjects showing capacity-"
              "ceiling reversals in capacity_ceiling_check.py's hbo_pfc measure (see the cross-check below).")
    else:
        print("  No reliable group-level shift in EGTA attention centroid with load: the mechanism may be "
              "contributing to accuracy through a route other than a literature-matched temporal shift.")

    if ARGS.capacity_summary:
        print("\n" + "=" * 100)
        print("CROSS-CHECK vs REAL PHYSIOLOGY (capacity_ceiling_check.py's own per-subject effect sizes)")
        print("=" * 100)
        phys = load_capacity_summary(ARGS.capacity_summary)
        summary["physiology_crosscheck"] = {}
        for sig_name in SIGNALS:
            summary["physiology_crosscheck"][sig_name] = {}
            x_all = {s: res[sig_name][s]["d"] for s in ids}
            for phys_name, phys_d in phys.items():
                common = [s for s in ids if s in phys_d and not np.isnan(x_all[s]) and not np.isnan(phys_d[s])]
                if len(common) < 6:
                    continue
                x = np.array([x_all[s] for s in common]); y = np.array([phys_d[s] for s in common])
                rho, pv = spearmanr(x, y)
                summary["physiology_crosscheck"][sig_name][phys_name] = {
                    "n": len(common), "rho": float(rho), "p": float(pv)}
                print(f"  {sig_name:18s} vs {phys_name:9s} (n={len(common):2d}): "
                      f"Spearman rho={rho:+.2f}, p={pv:.3g}"
                      + ("  <-- model attention tracks real physiology" if pv < 0.05 else ""))

    with open(os.path.join(out_dir, "xai_biomarker_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": {sig: res[sig] for sig in SIGNALS}, **summary}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_biomarker_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{sig}_{k}" for sig in SIGNALS for k in ("delta", "d", "p", "q", "sessions")])
        for s in ids:
            w.writerow([s] + [res[sig][s][k] for sig in SIGNALS for k in ("delta", "d", "p", "q", "sessions")])
    print(f"\n[written] {out_dir}: xai_biomarker_summary.json, xai_biomarker_per_subject.csv")


if __name__ == "__main__":
    main()
