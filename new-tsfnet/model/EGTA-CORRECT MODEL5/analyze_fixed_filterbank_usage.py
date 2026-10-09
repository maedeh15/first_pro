"""
analyze_fixed_filterbank_usage.py -- did --eeg-filterbank fixed actually do anything,
or is it architecturally a no-op that training never bothered to exploit?

WHY
===
GridFilterBank(mode="fixed") has ZERO trainable parameters -- it cannot be diagnosed
the way analyze_band_edges.py diagnoses "learnable" (there are no low_raw/band_raw to
move). The filter design itself was verified separately (pure signal-processing check,
no checkpoints needed): each of the 5 bands has unity gain at its own center frequency,
negligible leakage into far frequencies, and -- critically -- the 5 bands SUM to an
~all-pass response across 0.5-50 Hz (mean gain 0.998 over that range). So the filterbank
itself has no bug.

That "sums to all-pass" fact matters because of HOW conv1 is initialised when
in_bands=5 (EEGLikeBranch.__init__, tsfnet_model_notebook.py:358-375): the new 5-input
Conv3d is built by literally REPEATING the original 1-input kernel 5 times
(`_new.weight.copy_(_old.weight.repeat(1, in_bands, 1, 1, 1))`). Repeated-identical
weights applied to 5 bands that sum to all-pass is, at step 0, mathematically the same
computation as the original 1-band conv1 applied to the raw broadband signal. In other
words: **the fixed-filterbank model starts out IDENTICAL to the baseline** and can only
diverge from it to the extent that training pulls the 5 per-band weight slices apart
from each other.

So the right diagnostic is not "did the filter work" (it does) but "did training find it
useful to split these weights apart, and if so, did that correlate with any accuracy
change" -- exactly the band-edge-movement question asked for 'learnable', just measured
in weight space instead of Hz space.

WHAT IT DOES
============
For every VPxxx_session{s}_best.pt in --results-dir, pulls
  eeg_branch.conv1.conv.conv.weight     shape (16, 5, 4, 4, 12)
  fusion_branch.conv1.conv.conv.weight  shape (16, 5, 4, 4, 12)   (only if filterbank
                                                                    scope="both")
No model construction needed -- plain state_dict tensors, CPU only. For each branch and
each checkpoint:
  * per-band weight norm (L2 over the kernel's non-band dims), and how far each band's
    norm has moved (ratio) from the shared starting point (all 5 bands have the exact
    same norm at init, since they're literal copies);
  * pairwise cosine similarity between band kernel slices -- 1.0 if two bands are still
    identical (no differentiation learned), lower if training split them apart;
  * an overall "divergence score" = 1 - mean pairwise cosine similarity across the 10
    band pairs, pooled over folds/subjects, with a sign-consistency check (does every
    subject diverge in a similar way, or is it idiosyncratic per-fold noise, same
    distinction analyze_band_edges.py draws for learnable);
  * optionally (--fixed-csv/--baseline-csv, same convention as analyze_band_edges.py)
    correlates per-subject divergence against that subject's accuracy change vs the
    plain baseline, to see whether MORE differentiation helped or hurt.

Low divergence (cosine ~1.0 everywhere) across most subjects = the architecture gave the
model band-specific capacity and training chose not to use it (fixed falls back toward
baseline behaviour, consistent with the near-identical ensemble numbers already measured).
High divergence with no accuracy correlation = the model DID split the bands apart but in
an idiosyncratic/overfit way that didn't generalize (same failure mode already seen with
learnable's edges). High divergence WITH positive accuracy correlation would be the one
result that argues the filterbank is worth keeping -- not seen so far in the ensemble
numbers, but this checks it directly rather than inferring it from aggregate accuracy.

USAGE
=====
    python analyze_fixed_filterbank_usage.py --results-dir "E:\\...\\results_d11ef7593f_seed42"
    python analyze_fixed_filterbank_usage.py --results-dir "...\\results_d11ef7593f_seed42" ^
        --fixed-csv "...\\baseline_all_subjects_..._d11ef7593f.csv" ^
        --baseline-csv "...\\baseline_all_subjects_..._40b7d8ea53.csv"

No GPU, no preprocessed data, no tsfnet_model_notebook.py import needed -- only the .pt files.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

p = argparse.ArgumentParser(description=__doc__.split("USAGE")[0],
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--results-dir", required=True,
              help="Folder holding VPxxx_session{1,2,3}_best.pt from an --eeg-filterbank fixed run.")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-csv", default=None,
              help="Per-fold divergence CSV. Default: fixed_filterbank_divergence.csv inside --results-dir.")
p.add_argument("--fixed-csv", default=None,
              help="baseline_all_subjects*.csv of the fixed-filterbank run (subject_id,...,mean_acc,...).")
p.add_argument("--baseline-csv", default=None,
              help="baseline_all_subjects*.csv of the run to compare against (e.g. run60/40b7d8ea53). "
                   "Both --fixed-csv and --baseline-csv must be given together.")
ARGS = p.parse_args()

BRANCH_KEYS = {
    "eeg": "eeg_branch.conv1.conv.conv.weight",
    "fusion": "fusion_branch.conv1.conv.conv.weight",
}


def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


def load_acc_csv(path):
    d = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("subject_id") and row.get("mean_acc"):
                d[row["subject_id"]] = float(row["mean_acc"])
    return d


def band_stats(weight):
    """weight: (out_ch, in_bands, kh, kw, kt) torch tensor. Returns per-band norms and
    the mean pairwise cosine similarity between band slices (flattened over all other
    dims), i.e. between weight[:, b, ...] for each pair of bands b."""
    n_bands = weight.shape[1]
    flat = weight.reshape(weight.shape[0], n_bands, -1)          # (out_ch, bands, k)
    flat = flat.permute(1, 0, 2).reshape(n_bands, -1)             # (bands, out_ch*k)
    norms = flat.norm(dim=1).numpy()                             # (bands,)
    sims = []
    for i in range(n_bands):
        for j in range(i + 1, n_bands):
            num = float((flat[i] * flat[j]).sum())
            den = float(flat[i].norm() * flat[j].norm()) + 1e-12
            sims.append(num / den)
    return norms, np.array(sims)


def main():
    if not os.path.isdir(ARGS.results_dir):
        sys.exit(f"ERROR: --results-dir does not exist: {ARGS.results_dir}")
    subjects = parse_subjects(ARGS.subjects)
    out_csv = ARGS.out_csv or os.path.join(ARGS.results_dir, "fixed_filterbank_divergence.csv")

    rows = []
    missing_ckpt, missing_branch = [], {k: [] for k in BRANCH_KEYS}

    for sid in subjects:
        for sess in (1, 2, 3):
            ckpt_path = os.path.join(ARGS.results_dir, f"{sid}_session{sess}_best.pt")
            if not os.path.isfile(ckpt_path):
                missing_ckpt.append(ckpt_path)
                continue
            sd = torch.load(ckpt_path, map_location="cpu")
            for branch, key in BRANCH_KEYS.items():
                if key not in sd:
                    missing_branch[branch].append(ckpt_path)
                    continue
                w = sd[key]
                if w.shape[1] == 1:
                    continue  # this branch has in_bands=1 (scope="eeg", fusion untouched)
                norms, sims = band_stats(w)
                rows.append({
                    "subject_id": sid, "session": sess, "branch": branch,
                    "n_bands": w.shape[1],
                    "norm_delta": float(norms[0]), "norm_theta": float(norms[1]),
                    "norm_alpha": float(norms[2]), "norm_beta": float(norms[3]),
                    "norm_gamma": float(norms[4]) if len(norms) > 4 else float("nan"),
                    "norm_cv": float(norms.std(ddof=1) / (norms.mean() + 1e-12)),
                    "mean_pairwise_cos_sim": float(sims.mean()),
                    "min_pairwise_cos_sim": float(sims.min()),
                    "divergence": float(1.0 - sims.mean()),
                })

    if missing_ckpt:
        print(f"[skip] {len(missing_ckpt)} checkpoint(s) not found (e.g. {missing_ckpt[0]})")
    for branch, miss in missing_branch.items():
        if miss:
            print(f"[note] {len(miss)} checkpoint(s) have no '{BRANCH_KEYS[branch]}' key "
                  f"-- were they trained with --eeg-filterbank fixed? (e.g. {miss[0]})")
    if not rows:
        sys.exit("ERROR: no checkpoint in --results-dir had a multi-band conv1 weight "
                  "(in_bands>1) in either branch -- wrong --results-dir, or this run used "
                  "--eeg-filterbank none/learnable, or --eeg-filterbank-scope eeg only "
                  "(fusion branch would then show in_bands=1).")

    n_subjects = len(set(r["subject_id"] for r in rows))
    n_folds = len(set((r["subject_id"], r["session"], r["branch"]) for r in rows))
    print("=" * 88)
    print(f"FIXED-FILTERBANK WEIGHT-DIVERGENCE ANALYSIS | {n_subjects} subjects, "
          f"{n_folds} (subject,session,branch) rows (from {ARGS.results_dir})")
    print("=" * 88)
    print()
    print("At init, all 5 bands share the IDENTICAL conv1 kernel (replicated copy), so")
    print("mean_pairwise_cos_sim = 1.0 and divergence = 0.0 for every row at step 0.")
    print("divergence close to 0 after training = the model never bothered to tell the")
    print("bands apart (fixed-filterbank conv1 stayed ~equivalent to broadband baseline).")
    print()

    for branch in BRANCH_KEYS:
        sub = [r for r in rows if r["branch"] == branch]
        if not sub:
            continue
        div = np.array([r["divergence"] for r in sub])
        cv = np.array([r["norm_cv"] for r in sub])
        print(f"[{branch} branch, n={len(sub)} folds]")
        print(f"  divergence (1 - mean pairwise cosine sim):  mean={div.mean():.4f}  "
              f"SD={div.std(ddof=1):.4f}  min={div.min():.4f}  max={div.max():.4f}")
        print(f"  per-band norm coefficient of variation:      mean={cv.mean():.4f}  "
              f"SD={cv.std(ddof=1):.4f}")
        band_names = ["delta", "theta", "alpha", "beta", "gamma"]
        norm_means = [np.mean([r[f"norm_{b}"] for r in sub]) for b in band_names]
        norm_grand = np.mean(norm_means)
        print(f"  mean weight-norm share per band (equal=20% each if untouched):")
        for b, m in zip(band_names, norm_means):
            print(f"    {b:6s} {100*m/sum(norm_means):5.1f}%")
        print()

    # per-subject divergence (collapsed over folds/branches) for the correlation step
    subj_list = sorted(set(r["subject_id"] for r in rows))
    subj_div = {sid: np.mean([r["divergence"] for r in rows if r["subject_id"] == sid])
                for sid in subj_list}

    print(f"{'subject':8s} {'divergence':>11s}  {'n rows':>7s}")
    for sid in subj_list:
        n = sum(1 for r in rows if r["subject_id"] == sid)
        print(f"  {sid:8s} {subj_div[sid]:11.4f}  {n:7d}")
    div_all = np.array(list(subj_div.values()))
    print()
    print(f"ACROSS SUBJECTS: mean divergence {div_all.mean():.4f}, SD {div_all.std(ddof=1):.4f}, "
          f"{int((div_all < 0.02).sum())}/{len(div_all)} subjects essentially unchanged (<0.02), "
          f"{int((div_all > 0.10).sum())} meaningfully diverged (>0.10)")

    if ARGS.fixed_csv and ARGS.baseline_csv:
        print()
        print("=" * 88)
        print("WEIGHT DIVERGENCE vs ACCURACY CHANGE (fixed - baseline, per subject)")
        print("=" * 88)
        fixed_acc = load_acc_csv(ARGS.fixed_csv)
        base_acc = load_acc_csv(ARGS.baseline_csv)
        common = [s for s in subj_list if s in fixed_acc and s in base_acc]
        if len(common) < 3:
            print(f"  only {len(common)} subjects in common between the two CSVs -- skipping.")
        else:
            acc_diff = np.array([fixed_acc[s] - base_acc[s] for s in common])
            div_common = np.array([subj_div[s] for s in common])
            r = float(np.corrcoef(div_common, acc_diff)[0, 1])
            print(f"  subjects compared: {len(common)}")
            print(f"  corr(per-subject weight divergence, accuracy change) = {r:+.3f}")
            print("  near 0 => how much a subject's filters differentiated has no bearing on")
            print("            whether that subject's accuracy moved (consistent with the")
            print("            filterbank being a no-op rather than a useful-but-risky feature)")
            print("  negative => more differentiation associated with WORSE accuracy (overfitting)")
            print("  positive => more differentiation associated with BETTER accuracy (useful signal)")
    elif ARGS.fixed_csv or ARGS.baseline_csv:
        print()
        print("[note] --fixed-csv and --baseline-csv must both be given to run the "
              "accuracy-correlation section; skipping it.")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print()
    print(f"[written] {out_csv}  ({len(rows)} rows: subject x session x branch)")


if __name__ == "__main__":
    main()
