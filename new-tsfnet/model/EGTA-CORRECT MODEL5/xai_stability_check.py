"""
xai_stability_check.py -- is EGTA's attention explanation REPRODUCIBLE across
random seeds, or does every seed tell a different story about the same input?

WHY THIS EXISTS
STA-Net, MGFNet and MSTFDN each show attention-map figures from a SINGLE
trained run. None of them ask whether a second training run (same subject,
same data, only the random seed changed) would produce the same explanation.
If the attention pattern is mostly an artifact of a particular
initialization/optimization trajectory rather than a stable property of the
subject's data, then "the model attends to segment X" is not a reliable
scientific claim -- it is a coin flip that happened to land on X this time.
This script measures that directly, using the SAME 3 seeds (42, 999, 2024)
already trained and evaluated elsewhere in this project (ensemble_eval.py,
analyze_fixed_filterbank_usage.py) -- no new training needed.

WHAT IT MEASURES
For each subject and each held-out test session, the three seeds' checkpoints
are run on the EXACT SAME held-out windows (same subject, same session, same
preprocessing -> identical input order since the test loader is never
shuffled), so the three seeds' EGTA attention vectors over the 21 fNIRS
segments can be compared WINDOW FOR WINDOW, not just as a vague "looks
similar" pooled picture:

  1. PER-WINDOW pairwise Spearman correlation (3 seed pairs: 42-999, 42-2024,
     999-2024) between the two seeds' attention-weight vector over segments,
     for that exact same window. Averaged over all windows -> one
     "within-window cross-seed consistency" number per subject.
  2. ICC(2,1) (Shrout & Fleiss 1979 two-way random-effects, single rater) on
     each subject's SUBJECT-LEVEL attention profile (mean attention per
     segment, pooled over all that subject's windows) treating the 21
     segments as "items" and the 3 seeds as "raters". This is the standard
     reliability statistic for "do independent raters/measurements agree",
     applied here to seeds instead of human raters -- ICC > 0.75 is
     conventionally "excellent" agreement, 0.5-0.75 "moderate", < 0.5 "poor"
     (Koo & Li 2016).

Both metrics are reported per subject (so a committee member can see exactly
which subjects' explanations are trustworthy and which are not) and at the
group level (mean +/- SD across subjects, and the fraction of subjects
clearing the "moderate"/"excellent" ICC thresholds).

USAGE
=====
    python xai_stability_check.py --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback" ^
        --variant-hash 40b7d8ea53 --seeds 42 999 2024

(same --variant-hash / --seeds / --data-root convention as ensemble_eval.py;
pass --results-dirs explicitly, one path per seed, if auto-locating the
results_<hash>_seed<N> folders fails.) No GPU required, much faster with one.
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr

from tsfnet_model_notebook import TSFNet, NBackDataset, _unpack_batch, adapt_batchnorm_to_target
from xai_trunk import trunk_state, load_checkpoint                       # drops 'riem_head.*' so a member-branch checkpoint loads into the plain trunk

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                            description=__doc__.split("USAGE")[0])
p.add_argument("--data-root", required=True)
p.add_argument("--variant-hash", default=None, help="Hash in results_<hash>_seed<N> (e.g. run60 = 40b7d8ea53).")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 999, 2024])
p.add_argument("--results-dirs", nargs="*", default=None,
              help="Explicit results dir per seed, same order as --seeds. Overrides --variant-hash lookup.")
p.add_argument("--subjects", default="1-26")
p.add_argument("--out-dir", default=None, help="default: <data-root>/xai_stability_check")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
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


def locate_results_dir(folder_name, search_roots, max_depth=4):
    matches, seen = [], set()
    for root in search_roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root) or root in seen:
            continue
        seen.add(root)
        base = root.rstrip(os.sep).count(os.sep)
        for dirpath, dirnames, _f in os.walk(root):
            if dirpath.count(os.sep) - base >= max_depth:
                dirnames[:] = []
                continue
            if folder_name in dirnames:
                matches.append(os.path.join(dirpath, folder_name))
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("__pycache__", "visualizations")]
    return sorted(set(matches))


def resolve_results_dirs():
    if ARGS.results_dirs is not None:
        if len(ARGS.results_dirs) != len(ARGS.seeds):
            sys.exit(f"ERROR: --results-dirs has {len(ARGS.results_dirs)} entries but --seeds has {len(ARGS.seeds)}.")
        return list(ARGS.results_dirs)
    if not ARGS.variant_hash:
        sys.exit("ERROR: pass either --results-dirs (one per seed) or --variant-hash.")
    dirs = []
    for s in ARGS.seeds:
        name = f"results_{ARGS.variant_hash}_seed{s}"
        d = os.path.join(ARGS.data_root, name)
        if not os.path.isdir(d):
            roots, q = [], os.path.abspath(ARGS.data_root)
            for _ in range(4):
                roots.append(q); q = os.path.dirname(q)
            found = locate_results_dir(name, roots)
            if len(found) == 1:
                d = found[0]
            elif len(found) > 1:
                sys.exit(f"ERROR: '{name}' exists in several places: {found}. Pass --results-dirs.")
            else:
                sys.exit(f"ERROR: could not find '{name}' under {ARGS.data_root}. Pass --results-dirs.")
        dirs.append(d)
    return dirs


def build_model(n_segments):
    return TSFNet(num_classes=3, fusion_combination="paper_mean",
                  fnirs_num_segments=n_segments, fnirs_temporal_align="egta",
                  efgf_gate_mode="sigmoid", efgf_gate_init=-2.0)


def collect_egta_profile(sid, test_session, ckpt_path, full, device, batch_size, adabn):
    """Returns (n_windows, S) EGTA attention matrix for this subject/session's
    held-out windows, IN LOADER ORDER (shuffle=False), so window i is the
    SAME physical window regardless of which seed's checkpoint produced it."""
    n_segments = full.fnirs.shape[1] if full.fnirs.ndim == 5 else 21
    model = build_model(n_segments).to(device)
    state = load_checkpoint(ckpt_path, map_location=device)
    model.load_state_dict(trunk_state(state), strict=True)
    model.eval()

    test_idx = np.flatnonzero(full.session_ids == test_session)
    test_loader = DataLoader(Subset(full, test_idx), batch_size=batch_size, shuffle=False)
    if adabn:
        adapt_batchnorm_to_target(model, test_loader, device, log_fn=lambda *a, **k: None)

    weights = []
    with torch.no_grad():
        for batch in test_loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            model(x_eeg, x_fnirs, x_conn, x_chan)
            w = model.cafe._last_egta_weights.mean(dim=1)   # (B, S)
            weights.append(w.cpu().numpy())
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(weights, axis=0)


def icc_2_1(data):
    """Shrout & Fleiss (1979) ICC(2,1): two-way random effects, single
    rater/measurement, absolute agreement. data: (n_items, k_raters)."""
    n, k = data.shape
    grand_mean = data.mean()
    item_means = data.mean(axis=1)
    rater_means = data.mean(axis=0)
    ss_total = ((data - grand_mean) ** 2).sum()
    ss_rows = k * ((item_means - grand_mean) ** 2).sum()
    ss_cols = n * ((rater_means - grand_mean) ** 2).sum()
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom <= 0:
        return float("nan")
    return float((ms_rows - ms_error) / denom)


def main():
    device = ARGS.device
    results_dirs = resolve_results_dirs()
    for s, d in zip(ARGS.seeds, results_dirs):
        print(f"  seed {s:>5}: {d}")
    out_dir = ARGS.out_dir or os.path.join(ARGS.data_root, "xai_stability_check")
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)
    seed_pairs = list(itertools.combinations(range(len(ARGS.seeds)), 2))

    per_subject = {}
    for sid in subjects:
        subject_dir = os.path.join(ARGS.data_root, sid)
        try:
            full = NBackDataset(subject_dir, sid)
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
            continue

        window_pairwise_rhos = []       # per-window, per-pair rho, pooled over all sessions
        pooled_profiles = []            # one (S,) vector per seed, pooled over all this subject's windows

        seed_window_mats = [[] for _ in ARGS.seeds]   # per seed: list of (n_win_session, S), concatenated across sessions
        ok = True
        for test_session in (1, 2, 3):
            mats = []
            for seed, rdir in zip(ARGS.seeds, results_dirs):
                ckpt_path = os.path.join(rdir, f"{sid}_session{test_session}_best.pt")
                if not os.path.isfile(ckpt_path):
                    print(f"[{sid} s{test_session} seed{seed}] SKIP -- checkpoint not found")
                    ok = False
                    break
                mats.append(collect_egta_profile(sid, test_session, ckpt_path, full, device,
                                                 ARGS.batch_size, ARGS.adabn))
            if not ok or len(mats) != len(ARGS.seeds):
                ok = False
                break
            n_win = mats[0].shape[0]
            if any(m.shape[0] != n_win for m in mats):
                print(f"[{sid} s{test_session}] SKIP -- seeds disagree on window count "
                      f"({[m.shape[0] for m in mats]}); checkpoints likely from mismatched runs.")
                ok = False
                break
            for i, m in enumerate(mats):
                seed_window_mats[i].append(m)
            for w in range(n_win):
                for (a, b) in seed_pairs:
                    rho, _ = spearmanr(mats[a][w], mats[b][w])
                    if not np.isnan(rho):
                        window_pairwise_rhos.append(rho)

        if not ok:
            print(f"[{sid}] SKIP -- incomplete checkpoint set across seeds/sessions.")
            continue

        for i in range(len(ARGS.seeds)):
            cat = np.concatenate(seed_window_mats[i], axis=0)   # (n_windows_all_sessions, S)
            pooled_profiles.append(cat.mean(axis=0))             # (S,) subject-level profile for this seed

        profile_matrix = np.stack(pooled_profiles, axis=1)   # (S, k_seeds)
        icc = icc_2_1(profile_matrix)
        mean_rho = float(np.mean(window_pairwise_rhos)) if window_pairwise_rhos else float("nan")
        sd_rho = float(np.std(window_pairwise_rhos, ddof=1)) if len(window_pairwise_rhos) > 1 else float("nan")
        per_subject[sid] = {"n_windows": len(window_pairwise_rhos) // len(seed_pairs),
                            "mean_pairwise_rho": mean_rho, "sd_pairwise_rho": sd_rho, "icc_2_1": icc}
        print(f"[{sid}] n_windows={per_subject[sid]['n_windows']:4d}  "
              f"mean cross-seed rho={mean_rho:+.3f} (SD {sd_rho:.3f})  ICC(2,1)={icc:+.3f}")

    ids = sorted(per_subject)
    if len(ids) < 6:
        sys.exit(f"ERROR: only {len(ids)} subject(s) usable; fewer than 6 subjects produced usable cross-seed data.. The group tests (Wilcoxon, bootstrap) need at least 6 subjects: run with --subjects 1-6 or more (the per-subject lines above are already valid).")

    rhos = np.array([per_subject[s]["mean_pairwise_rho"] for s in ids])
    iccs = np.array([per_subject[s]["icc_2_1"] for s in ids])
    rhos_ok = rhos[~np.isnan(rhos)]
    iccs_ok = iccs[~np.isnan(iccs)]

    hline = "=" * 100
    print("\n" + hline)
    print(f"CROSS-SEED EXPLANATION-STABILITY SUMMARY -- {len(ids)} subjects, "
          f"seeds={ARGS.seeds}")
    print(hline)
    print(f"Per-window pairwise Spearman rho (3 seed-pairs, same exact held-out window): "
          f"mean={rhos_ok.mean():+.3f}, SD={rhos_ok.std(ddof=1):.3f}, "
          f"median={np.median(rhos_ok):+.3f}")
    print(f"ICC(2,1) on subject-level attention profile (21 segments x 3 seeds): "
          f"mean={iccs_ok.mean():+.3f}, SD={iccs_ok.std(ddof=1):.3f}")
    excellent = int((iccs_ok >= 0.75).sum()); moderate = int(((iccs_ok >= 0.5) & (iccs_ok < 0.75)).sum())
    poor = int((iccs_ok < 0.5).sum())
    print(f"ICC categories (Koo & Li 2016): excellent (>=0.75): {excellent}/{len(iccs_ok)}, "
          f"moderate (0.5-0.75): {moderate}/{len(iccs_ok)}, poor (<0.5): {poor}/{len(iccs_ok)}")

    print("\nREADING")
    if iccs_ok.mean() >= 0.75:
        print("  EGTA's attention explanation is HIGHLY REPRODUCIBLE across independent training runs: "
              "the segments it attends to for a given subject are a stable property of that subject's "
              "data, not an artifact of one particular random initialization. This is a reliability claim "
              "none of the three reference papers can make, since each shows only one trained run.")
    elif iccs_ok.mean() >= 0.5:
        print("  EGTA's attention explanation shows MODERATE cross-seed reproducibility: broadly consistent, "
              "but with enough seed-to-seed variation that any single-run attention figure (as in STA-Net/"
              "MGFNet/MSTFDN) should be read as indicative rather than a precise, re-derivable measurement.")
    else:
        print("  EGTA's attention explanation is NOT reliably reproducible across seeds: a different random "
              "initialization gives a materially different story about which fNIRS segments 'matter' for "
              "the same subject. This is an important, honest finding -- it means single-run attention "
              "figures (the ONLY kind the three reference papers show) should not be trusted as stable "
              "characterizations of subject-level physiology, and this quantitative check is itself a "
              "methodological contribution the literature is missing.")

    with open(os.path.join(out_dir, "xai_stability_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": per_subject,
                  "group": {"mean_pairwise_rho": float(rhos_ok.mean()), "sd_pairwise_rho": float(rhos_ok.std(ddof=1)),
                            "mean_icc": float(iccs_ok.mean()), "sd_icc": float(iccs_ok.std(ddof=1)),
                            "n_excellent": excellent, "n_moderate": moderate, "n_poor": poor}},
                 f, indent=2, default=float)
    with open(os.path.join(out_dir, "xai_stability_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "n_windows", "mean_pairwise_rho", "sd_pairwise_rho", "icc_2_1"])
        for s in ids:
            r = per_subject[s]
            w.writerow([s, r["n_windows"], r["mean_pairwise_rho"], r["sd_pairwise_rho"], r["icc_2_1"]])
    print(f"\n[written] {out_dir}: xai_stability_summary.json, xai_stability_per_subject.csv")


if __name__ == "__main__":
    main()
