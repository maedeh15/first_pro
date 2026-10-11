"""
stanet_xai_biomarker_validation.py -- is STA-Net's own attention/gating a valid cognitive-load
biomarker, or just a plausible-looking picture?

STA-Net counterpart of your xai_biomarker_validation.py. Same question, same statistics; the
signals now come from STANet's output dict instead of TSFNet's CAFE/EFGF internals (see
stanet_xai_signals.py for the full mapping).

WHY THIS EXISTS (the gap in STA-Net's own XAI section)
STA-Net shows attention maps and prediction-weight curves and says they "look reasonable"
(Sec. 3.5: kappa_fnirs stays flat over time, so EGTA "is working"). It runs no formal
statistical test of that claim against any neuroscience ground truth. This script applies the
same machinery your capacity_ceiling_check.py already applies to REAL physiology -- per-subject
exact permutation test, Cohen's d, BH-FDR across subjects, cross-session trait consistency,
group Wilcoxon, Cochran's Q / I^2 -- to the MODEL'S OWN internal signals, and then correlates
the two subject by subject.

MODEL-DERIVED SIGNALS (per window -> averaged per trial via trial_ids, the same unit of
analysis as stanet_trial_level_eval.py)
  egta_centroid    attention-weighted mean index over the l lagged fNIRS windows, from
                   STANet's out["att"]. HYPOTHESIS, and the only signal here with a
                   literature-grounded direction: higher load -> slower haemodynamic onset ->
                   the model has to attend LATER to find the task-relevant fNIRS response, so
                   the centroid should go UP from 2-back to 3-back. Directly comparable to the
                   hbo_pfc/theta_fm/p300_pz predictions in capacity_ceiling_check.py.
  egta_entropy     Shannon entropy of the same distribution (peaked vs near-uniform).
                   EXPLORATORY, no sign imposed.
  fgsa1_phi_mean   mean of FGSA's fNIRS-derived spatial attention map Phi (Eq. 2) at stage 1
  fgsa2_phi_mean   ... and stage 2. EXPLORATORY, no sign imposed.
  w_fnirs          the decision weight STA-Net puts on the fNIRS head for THIS window (Eq. 6-7).
                   EXPLORATORY. This signal does not exist in TSFNet, whose fusion weights are
                   fixed model-level scalars -- STA-Net computes them per sample, so "how much
                   did the model trust fNIRS for this trial" is a question only askable here.

IMPORTANT -- the 3-back-minus-2-back contrast ignores 0-back on purpose: it is the contrast
with a clean literature prediction (both are n-back with a memory load, differing only in
degree). Use stanet_xai_load_level_anova.py for the omnibus test across all three levels.

CROSS-VALIDATION AGAINST REAL PHYSIOLOGY (the key step)
Pass --capacity-summary pointing at capacity_ceiling_check.py's capacity_summary.json for the
same subjects, and this script Spearman-correlates, per subject, each model signal's effect
size against that subject's real hbo_pfc / theta_fm / p300_pz effect sizes. A significant
correlation means the attention mechanism tracks real inter-subject physiological variation
rather than an artifact.

USAGE
=====
    python stanet_xai_biomarker_validation.py ^
        --results-dir "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_seed42" ^
        --data-root   "E:\\kntu\\Thesis\\codes\\Results\\STANET_TSFCOMP_nback" ^
        --fnirs-scale 1.0 ^
        --capacity-summary "E:\\...\\capacity_check\\capacity_summary.json"

--fnirs-scale MUST match what training used (1.0 for z-scored preprocessing, omit for
paper-exact mmol/L). The signals are read from the network, so feeding it differently scaled
inputs than it was trained on gives meaningless attention.

Needs the preprocessed windows + one seed's checkpoints. No GPU required, much faster with
one -- one forward pass per fold, same cost as stanet_trial_level_eval.py.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import binomtest, chi2, spearmanr, wilcoxon

from stanet_xai_signals import (SIGNALS, EXPECTED_SIGN, LABELS, SubjectData, add_common_args,
                                bh, parse_subjects, run_header, subject_trial_rows)

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True,
               help="Folder with VPxxx_session{1,2,3}_best.pt from ONE seed's LOSO run.")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_biomarker_check")
p.add_argument("--capacity-summary", default=None,
               help="capacity_ceiling_check.py's capacity_summary.json, for the physiology cross-check.")
add_common_args(p)
ARGS = p.parse_args()


# ---------------------------------------------------------------------------
# Statistics -- same engine as capacity_ceiling_check.py / your TSFNet version
# ---------------------------------------------------------------------------
def exact_perm_p(v, lab, sess):
    """Exact two-sided permutation p for mean(3-back) - mean(2-back), enumerating every
    WITHIN-SESSION relabelling (not sampling). Permuting within session keeps any session-level
    drift out of the null, exactly as in capacity_ceiling_check.py."""
    m = np.isin(lab, [1, 2]) & ~np.isnan(v)
    v, lab, sess = v[m], lab[m], sess[m]
    n3, n2 = int((lab == 2).sum()), int((lab == 1).sum())
    if n3 == 0 or n2 == 0:
        return np.nan
    per_session = []
    for s in np.unique(sess):
        vs = v[sess == s]
        k3 = int((lab[sess == s] == 2).sum())
        per_session.append(np.array([vs[list(c)].sum()
                                     for c in itertools.combinations(range(len(vs)), k3)]))
    tot3 = per_session[0]
    for ps in per_session[1:]:
        tot3 = (tot3[:, None] + ps[None, :]).ravel()
    total = v.sum()
    stats = tot3 / n3 - (total - tot3) / n2
    obs = v[lab == 2].mean() - v[lab == 1].mean()
    return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))


def load_capacity_summary(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {m: {s: v["d"] for s, v in d["per_subject"][m].items()}
            for m in ("hbo_pfc", "theta_fm", "p300_pz") if m in d.get("per_subject", {})}


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_biomarker_check")
    os.makedirs(out_dir, exist_ok=True)
    run_header("STA-Net XAI-AS-BIOMARKER VALIDATION", ARGS.data_root, ARGS.results_dir,
               ARGS.adabn, ARGS.fnirs_scale)

    all_rows, seg_lens = {}, set()
    for sid in parse_subjects(ARGS.subjects):
        try:
            data = SubjectData(ARGS.data_root, sid, ARGS.fnirs_scale)
            seg_lens.add(data.seg_len)
            all_rows[sid] = subject_trial_rows(data, ARGS.results_dir, ARGS.device,
                                               ARGS.batch_size, ARGS.adabn)
            del data
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    ids = sorted(s for s in all_rows if all_rows[s])
    if len(ids) < 6:
        sys.exit("ERROR: fewer than 6 subjects processed -- check --results-dir / --data-root.")
    print(f"\n{len(ids)} subjects processed, l = {sorted(seg_lens)} lagged fNIRS windows.")

    # ---- per subject, per signal: delta, d, exact p, session-sign-consistency ----
    res = {sig: {} for sig in SIGNALS}
    for s in ids:
        rows = all_rows[s]
        lab = np.array([r["label"] for r in rows])
        sess = np.array([r["session"] for r in rows])
        which = np.array([r["signal"] for r in rows])
        val = np.array([r["value"] for r in rows])
        for sig in SIGNALS:
            m = which == sig
            v, l, sv = val[m], lab[m], sess[m]
            v2, v3 = v[l == 1], v[l == 2]
            if len(v2) < 2 or len(v3) < 2 or np.isnan(v2).any() or np.isnan(v3).any():
                res[sig][s] = {"delta": np.nan, "d": np.nan, "se": np.nan, "p": np.nan, "sessions": 0}
                continue
            delta = v3.mean() - v2.mean()
            sp = np.sqrt(((len(v2) - 1) * v2.var(ddof=1) + (len(v3) - 1) * v3.var(ddof=1))
                         / (len(v2) + len(v3) - 2))
            per_sess = [np.nanmean(v[(l == 2) & (sv == k)]) - np.nanmean(v[(l == 1) & (sv == k)])
                        for k in np.unique(sv)]
            res[sig][s] = {
                "delta": float(delta), "d": float(delta / (sp + 1e-30)),
                "se": float(np.sqrt(v2.var(ddof=1) / len(v2) + v3.var(ddof=1) / len(v3))),
                "p": exact_perm_p(v, l, sv),
                "sessions": int(sum(np.sign(x) == np.sign(delta) for x in per_sess if x == x)),
            }
    for sig in SIGNALS:
        for s, qq in zip(ids, bh([res[sig][s]["p"] for s in ids])):
            res[sig][s]["q"] = float(qq)

    # ---- group level ----
    summary = {"n_subjects": len(ids), "seg_len": sorted(seg_lens), "signals": {}}
    print("\n" + "=" * 112)
    print(f"Model-internal signals, 3-back minus 2-back, {len(ids)} subjects")
    print("(exact within-session permutation, BH-FDR across subjects, group Wilcoxon, Cochran Q / I^2 "
          "-- identical to capacity_ceiling_check.py)")
    print("=" * 112)
    print(f"{'signal':17s} {'expect':>6s} | {'as exp.':>7s} {'rev.':>5s} | {'sig as exp.':>11s} {'sig rev.':>8s} | "
          f"{'group p':>9s} | {'I^2':>5s} {'Q p':>9s} | {'stable 3/3':>10s}")
    for sig in SIGNALS:
        d_ = np.array([res[sig][s]["delta"] for s in ids])
        se = np.array([res[sig][s]["se"] for s in ids])
        q_ = np.array([res[sig][s]["q"] for s in ids])
        ok = ~np.isnan(d_)
        exp = EXPECTED_SIGN.get(sig, 0)
        if exp == 0:
            as_exp = rev = sig_exp = sig_rev = -1          # no literature sign to test against
        else:
            as_exp = int((np.sign(d_[ok]) == exp).sum())
            rev = int((np.sign(d_[ok]) == -exp).sum())
            sig_exp = int(((q_ < 0.05) & (np.sign(d_) == exp)).sum())
            sig_rev = int(((q_ < 0.05) & (np.sign(d_) == -exp)).sum())
        gp = float(wilcoxon(d_[ok]).pvalue) if ok.sum() >= 6 else np.nan
        w = 1 / np.clip(se[ok], 1e-12, None) ** 2
        mu = (w * d_[ok]).sum() / w.sum()
        Q = float((w * (d_[ok] - mu) ** 2).sum())
        df = int(ok.sum()) - 1
        I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
        Qp = float(chi2.sf(Q, df)) if df > 0 else np.nan
        stable = int(sum(res[sig][s]["sessions"] == 3 for s in ids))
        summary["signals"][sig] = {
            "label": LABELS[sig], "expected_sign": exp,
            "n_as_expected": as_exp, "n_reversed": rev,
            "n_sig_as_expected": sig_exp, "n_sig_reversed": sig_rev,
            "sign_test_p": (float(binomtest(as_exp, as_exp + rev, 0.5).pvalue)
                            if exp != 0 and as_exp + rev else np.nan),
            "group_wilcoxon_p": gp, "I2": I2, "Q": Q, "Q_p": Qp,
            "n_sign_stable_all_3_sessions": stable,
        }
        es = "+" if exp > 0 else "-" if exp < 0 else "n/a"
        fmt = lambda x: f"{x:d}" if x >= 0 else "-"
        print(f"{sig:17s} {es:>6s} | {fmt(as_exp):>7s} {fmt(rev):>5s} | {fmt(sig_exp):>11s} {fmt(sig_rev):>8s} | "
              f"{gp:9.3g} | {I2:5.2f} {Qp:9.3g} | {stable:6d}/{len(ids)}")

    print("\nREADING (egta_centroid -- the only signal with a literature-grounded direction):")
    ec = summary["signals"]["egta_centroid"]
    if ec["n_sig_as_expected"] >= 3 and ec["group_wilcoxon_p"] < 0.05:
        print("  EGTA attends to LATER lagged fNIRS windows at 3-back than at 2-back, matching the predicted\n"
              "  slower haemodynamic onset under higher working-memory load. The learned temporal alignment is\n"
              "  tracking a real, literature-consistent physiological pattern, not an arbitrary shortcut.")
    elif ec["n_sig_reversed"] >= 3:
        print("  EGTA attends EARLIER at higher load for several subjects -- opposite to the prediction. Worth\n"
              "  checking whether those are the same subjects showing capacity-ceiling reversals in\n"
              "  capacity_ceiling_check.py's hbo_pfc (the cross-check below answers this directly).")
    else:
        print("  No reliable group-level shift in the EGTA centroid with load. EGTA may be contributing to\n"
              "  accuracy by a route other than a literature-matched temporal shift -- which is worth reporting\n"
              "  as-is, since STA-Net's own paper asserts the temporal-alignment story without testing it.")
    wf = summary["signals"]["w_fnirs"]
    print(f"\n  w_fnirs (STA-Net-only signal, no TSFNet equivalent): group Wilcoxon p={wf['group_wilcoxon_p']:.3g}, "
          f"sign stable across all 3 sessions in {wf['n_sign_stable_all_3_sessions']}/{len(ids)} subjects.")
    print("  A load-dependent shift here would mean the model re-weights EEG vs fNIRS by task difficulty;\n"
          "  no shift means the decision weights are effectively a per-subject constant.")

    # ---- cross-check against real physiology ----
    if ARGS.capacity_summary:
        print("\n" + "=" * 100)
        print("CROSS-CHECK vs REAL PHYSIOLOGY (capacity_ceiling_check.py's per-subject effect sizes)")
        print("=" * 100)
        phys = load_capacity_summary(ARGS.capacity_summary)
        summary["physiology_crosscheck"] = {}
        for sig in SIGNALS:
            summary["physiology_crosscheck"][sig] = {}
            x_all = {s: res[sig][s]["d"] for s in ids}
            for pname, pd_ in phys.items():
                common = [s for s in ids if s in pd_ and not np.isnan(x_all[s]) and not np.isnan(pd_[s])]
                if len(common) < 6:
                    continue
                rho, pv = spearmanr([x_all[s] for s in common], [pd_[s] for s in common])
                summary["physiology_crosscheck"][sig][pname] = {"n": len(common), "rho": float(rho),
                                                                "p": float(pv)}
                print(f"  {sig:17s} vs {pname:9s} (n={len(common):2d}): Spearman rho={rho:+.2f}, p={pv:.3g}"
                      + ("   <-- model attention tracks real physiology" if pv < 0.05 else ""))

    with open(os.path.join(out_dir, "xai_biomarker_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": {sig: res[sig] for sig in SIGNALS}, **summary}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_biomarker_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ("delta", "d", "p", "q", "sessions")
        w.writerow(["subject"] + [f"{sig}_{k}" for sig in SIGNALS for k in keys])
        for s in ids:
            w.writerow([s] + [res[sig][s][k] for sig in SIGNALS for k in keys])
    print(f"\n[written] {out_dir}: xai_biomarker_summary.json, xai_biomarker_per_subject.csv")


if __name__ == "__main__":
    main()
