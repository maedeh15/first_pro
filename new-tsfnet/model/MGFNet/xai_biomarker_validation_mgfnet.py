"""
xai_biomarker_validation_mgfnet.py -- is MGFNet's own CGSCR routing/gating a
valid cognitive-load biomarker, or just a plausible-looking picture?

MGFNet version of xai_biomarker_validation.py. The question and the statistics
are unchanged; the model signals are MGFNet's (see mgfnet_xai_common.py):
gate_eeg / gate_fnirs / gate_cross / gate_entropy / coupling_entropy /
routing_theta_mass / routing_entropy. TSFNet's egta_centroid and EFGF Phi do
not exist in MGFNet.

DIRECTIONAL HYPOTHESIS (the counterpart of TSFNet's egta_centroid):
  routing_theta_mass UP with load -- frontal-midline theta rises with working-
  memory load, so CGSCR's router should draw more on theta-band tokens at
  3-back than at 2-back. All other signals are exploratory (no imposed sign).

STATISTICS (identical engine): per subject delta = mean(3-back) - mean(2-back)
over held-out trials, Cohen's d, exact within-session permutation p, session
sign-consistency, BH-FDR across subjects; group Wilcoxon, sign test,
Cochran's Q / I^2.

PHYSIOLOGY CROSS-CHECK: --capacity-summary = capacity_ceiling_check.py's
capacity_summary.json; per-subject Spearman between each model signal's d and
the subject's real hbo_pfc / theta_fm / p300_pz d. The key test for MGFNet:
routing_theta_mass vs theta_fm (expected rho > 0).

USAGE
=====
    python xai_biomarker_validation_mgfnet.py --results-dir "E:\\...\\results_mgf_<hash>_seed42" ^
        --data-root "E:\\...\\MGFNET_PAPER_EXACT_nback_28ch" ^
        --capacity-summary "E:\\...\\capacity_check\\capacity_summary.json"
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from scipy.stats import binomtest, chi2, spearmanr

import mgfnet_xai_common as X

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--results-dir", required=True, help="results_mgf_<hash>_seed<N> folder (full MGFNet)")
p.add_argument("--data-root", required=True, help="folder with the MGFNet-preprocessed VPxxx/ subjects")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <results-dir>/xai_biomarker_check")
p.add_argument("--capacity-summary", default=None)
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--min-subjects", type=int, default=6)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ARGS = p.parse_args()
SIGNALS, EXPECTED_SIGN = X.SIGNALS, X.EXPECTED_SIGN


def load_capacity_summary(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return {m: {s: v["d"] for s, v in d["per_subject"][m].items()}
            for m in ("hbo_pfc", "theta_fm", "p300_pz") if m in d.get("per_subject", {})}


def per_subject_stats(rows):
    lab = np.array([r["label"] for r in rows]); sess = np.array([r["session"] for r in rows])
    sig = np.array([r["signal"] for r in rows]); val = np.array([r["value"] for r in rows])
    out = {}
    for name in SIGNALS:
        m = sig == name
        v, l, sv = val[m], lab[m], sess[m]
        v2, v3 = v[l == 1], v[l == 2]
        if len(v2) < 2 or len(v3) < 2 or np.isnan(v2).any() or np.isnan(v3).any():
            out[name] = {"delta": np.nan, "d": np.nan, "se": np.nan, "p": np.nan, "sessions": 0}
            continue
        delta = v3.mean() - v2.mean()
        sp = np.sqrt(((len(v2) - 1) * v2.var(ddof=1) + (len(v3) - 1) * v3.var(ddof=1)) / (len(v2) + len(v3) - 2))
        per_sess = [np.nanmean(v[(l == 2) & (sv == k)]) - np.nanmean(v[(l == 1) & (sv == k)]) for k in np.unique(sv)]
        out[name] = {"delta": float(delta), "d": float(delta / (sp + 1e-30)),
                     "se": float(np.sqrt(v2.var(ddof=1) / len(v2) + v3.var(ddof=1) / len(v3))),
                     "p": X.exact_perm_p(v, l, sv),
                     "sessions": int(sum(np.sign(x) == np.sign(delta) for x in per_sess if x == x))}
    return out


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.results_dir, "xai_biomarker_check")
    os.makedirs(out_dir, exist_ok=True)
    res = {name: {} for name in SIGNALS}
    ids = []
    for sid in X.parse_subjects(ARGS.subjects):
        try:
            rows = X.subject_trial_rows(sid, ARGS.results_dir, ARGS.data_root, ARGS.device, ARGS.batch_size)
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
            continue
        if not rows:
            continue
        for name, r in per_subject_stats(rows).items():
            res[name][sid] = r
        ids.append(sid)
        print(f"[{sid}] done ({len(rows) // len(SIGNALS)} held-out trials)")
    if len(ids) < ARGS.min_subjects:
        sys.exit(f"ERROR: only {len(ids)} subjects processed (< {ARGS.min_subjects}).")
    for name in SIGNALS:
        for s, q in zip(ids, X.bh([res[name][s]["p"] for s in ids])):
            res[name][s]["q"] = float(q)

    summary = {"n_subjects": len(ids), "signals": {}}
    print("\n" + "=" * 104)
    print(f"XAI-AS-BIOMARKER VALIDATION (MGFNet CGSCR signals) -- 3-back minus 2-back, {len(ids)} subjects")
    print("(exact within-session permutation, BH-FDR, Wilcoxon, Cochran's Q/I^2 -- same engine as TSFNet)")
    print("=" * 104)
    print(f"{'signal':20s} {'expect':>6s} | {'as exp.':>7s} {'reversed':>8s} | {'sig as exp.':>11s} "
          f"{'sig rev.':>8s} | {'group p':>8s} | {'I^2':>5s} {'Q p':>8s} | {'stable 3/3':>10s}")
    for name in SIGNALS:
        d_ = np.array([res[name][s]["delta"] for s in ids]); se = np.array([res[name][s]["se"] for s in ids])
        q_ = np.array([res[name][s]["q"] for s in ids]); ok = ~np.isnan(d_) & (se > 0)
        exp = EXPECTED_SIGN.get(name, 0)
        if exp:
            as_exp = int((np.sign(d_[ok]) == exp).sum()); rev = int((np.sign(d_[ok]) == -exp).sum())
            sig_exp = int(((q_ < 0.05) & (np.sign(d_) == exp)).sum())
            sig_rev = int(((q_ < 0.05) & (np.sign(d_) == -exp)).sum())
        else:
            as_exp = rev = sig_exp = sig_rev = -1
        gp = X.safe_wilcoxon(d_[ok]) if ok.sum() >= 6 else np.nan
        if ok.sum() >= 2:
            w = 1 / se[ok] ** 2; mu = (w * d_[ok]).sum() / w.sum()
            Q = float((w * (d_[ok] - mu) ** 2).sum()); df = int(ok.sum()) - 1
            I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
            Qp = float(chi2.sf(Q, df))
        else:
            Q = I2 = Qp = np.nan
        stable = int(sum(res[name][s]["sessions"] == 3 for s in ids))
        summary["signals"][name] = {
            "label": X.LABELS[name], "expected_sign": exp, "n_as_expected": as_exp, "n_reversed": rev,
            "n_sig_as_expected": sig_exp, "n_sig_reversed": sig_rev,
            "sign_test_p": float(binomtest(as_exp, as_exp + rev, 0.5).pvalue) if exp and as_exp + rev else np.nan,
            "group_wilcoxon_p": gp, "I2": I2, "Q": Q, "Q_p": Qp, "n_sign_stable_all_3_sessions": stable}
        es = "+" if exp > 0 else "-" if exp < 0 else "n/a"
        print(f"{name:20s} {es:>6s} | {as_exp:7d} {rev:8d} | {sig_exp:11d} {sig_rev:8d} | "
              f"{gp:8.3g} | {I2:5.2f} {Qp:8.3g} | {stable:6d}/{len(ids)}")

    print("\nREADING (routing_theta_mass, the only signal with a literature-grounded direction):")
    rt = summary["signals"]["routing_theta_mass"]
    if rt["n_sig_as_expected"] >= 3 and rt["group_wilcoxon_p"] < 0.05:
        print("  CGSCR's router draws MORE on theta-band tokens at 3-back than 2-back, matching the "
              "frontal-midline theta increase with working-memory load: the routing is tracking a real, "
              "literature-consistent physiological pattern, not an arbitrary shortcut.")
    elif rt["n_sig_reversed"] >= 3:
        print("  The router uses theta tokens LESS at higher load for several subjects -- opposite to the "
              "prediction; check whether these subjects show capacity-ceiling reversals in theta_fm.")
    else:
        print("  No reliable group-level load effect on theta routing: CGSCR may help accuracy through a "
              "route other than a literature-matched theta emphasis.")

    if ARGS.capacity_summary:
        print("\n" + "=" * 104)
        print("CROSS-CHECK vs REAL PHYSIOLOGY (capacity_ceiling_check.py per-subject effect sizes)")
        print("=" * 104)
        phys = load_capacity_summary(ARGS.capacity_summary)
        summary["physiology_crosscheck"] = {}
        for name in SIGNALS:
            summary["physiology_crosscheck"][name] = {}
            xd = {s: res[name][s]["d"] for s in ids}
            for pname, pd in phys.items():
                common = [s for s in ids if s in pd and not np.isnan(xd[s]) and not np.isnan(pd[s])]
                if len(common) < 6:
                    continue
                rho, pv = spearmanr([xd[s] for s in common], [pd[s] for s in common])
                summary["physiology_crosscheck"][name][pname] = {"n": len(common), "rho": float(rho), "p": float(pv)}
                print(f"  {name:20s} vs {pname:9s} (n={len(common):2d}): Spearman rho={rho:+.2f}, p={pv:.3g}"
                      + ("  <-- tracks real physiology" if pv < 0.05 else ""))

    with open(os.path.join(out_dir, "xai_biomarker_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": res, **summary}, f, indent=2, default=X.nan_float)
    with open(os.path.join(out_dir, "xai_biomarker_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        keys = ("delta", "d", "p", "q", "sessions")
        w.writerow(["subject"] + [f"{n}_{k}" for n in SIGNALS for k in keys])
        for s in ids:
            w.writerow([s] + [res[n][s][k] for n in SIGNALS for k in keys])
    print(f"\n[written] {out_dir}: xai_biomarker_summary.json, xai_biomarker_per_subject.csv")


if __name__ == "__main__":
    main()
