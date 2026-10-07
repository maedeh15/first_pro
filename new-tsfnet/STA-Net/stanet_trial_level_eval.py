"""
stanet_trial_level_eval.py -- trial-level (block-level) decision aggregation for STA-Net
(rewrite of your TSFNet trial_level_eval.py).

WHY: every n-back trial is ONE block with ONE label, but the pipeline turns it into many overlapping
3 s windows (TASK_PERIOD=42 -> 42 windows per trial). The per-window accuracy that run_stanet_loso.py
reports is therefore not the decision unit of an application; pooling the window posteriors of a trial
into one decision is. The held-out session stays held out: aggregation happens WITHIN each test trial,
using only that trial's own posteriors, never a label. Both metrics are reported side by side so the
window-level number stays comparable with TSFNet / MSTFDN.

HOW: run_stanet_loso.py stores per-fold weights (<results>/VPxxx_session{k}_best.pt, written after
stage 2, before any AdaBN) but not the per-window posteriors, so this script reloads each checkpoint,
does one forward pass over the held-out session, aggregates per trial_id and compares against the
window-level accuracy the training run itself recorded (self-check, see below).

Differences from the TSFNet version
  * No exec() of a notebook: the model comes from stanet_model.py / stanet_common.py.
  * STA-Net has NO AdaBN. `--adabn` exists only to replay a run that was TRAINED/SCORED with
    `run_stanet_loso.py --adabn`; the flag must match that run or the self-check fails.
  * The self-check reads the run's own stanet_all_subjects_*.csv automatically (per-subject and grand
    mean window accuracy); --expected-window-acc overrides it.
  * Heads: final / eeg / fusion / fnirs. Remember the EEG head is auxiliary in STA-Net (it is not part of
    the final decision), so "eeg" here is a diagnostic branch, not an ensemble member.
  * 9 trials per test session (3 per class) -> trial accuracy is quantised to 1/9 steps; only the pooled
    estimate (26 subjects x 3 folds x seeds) is worth interpreting.

Usage
    python stanet_trial_level_eval.py --seed 42 --val-mode trial
    python stanet_trial_level_eval.py --results-dir <...> --data-root <...> --subjects 1-26
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

from stanet_common import (NUM_CLASSES, HEADS, load_subject, WindowLoader, session_holdout_split,
                           cohen_kappa, confusion_matrix_np, build_model, adapt_batchnorm_to_target,
                           collect_outputs)

p = argparse.ArgumentParser(description="Trial-level aggregation from saved STA-Net checkpoints.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data-root", default=r"E:\kntu\Thesis\codes\Results\STANET_PAPER_EXACT_nback",
               help="folder holding the per-subject VPxxx/ preprocessed .npy folders")
p.add_argument("--seed", type=int, required=True, help="seed of the training run to evaluate")
p.add_argument("--val-mode", default="trial", choices=["trial", "cross_session"],
               help="val-mode of the training run (selects results_stanet_<val-mode>_seed<seed>)")
p.add_argument("--results-dir", default=None, help="explicit results folder (overrides the two flags above)")
p.add_argument("--subjects", default="1-26", help="'1-26' or '1,5,17'")
p.add_argument("--batch-size", type=int, default=64)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out-prefix", default=None, help="default: stanet_trial_level_eval_seed{SEED}")
p.add_argument("--adabn", action="store_true", default=False,
               help="re-apply AdaBN (ONLY if the training run used --adabn)")
p.add_argument("--expected-window-acc", type=float, default=None,
               help="grand-mean window accuracy on record; default: read from the run's CSV")
p.add_argument("--reproduce-tolerance", type=float, default=0.002)
ARGS = p.parse_args()

# ---------------------------------------------------------------------------
# Aggregation rules (unchanged logic)
# ---------------------------------------------------------------------------
def aggregate_mean_prob(probs, trial_ids):
    """PRIMARY: mean posterior over a trial's windows, then argmax (confidence-weighted)."""
    trials = np.unique(trial_ids)
    return trials, np.array([probs[trial_ids == t].mean(axis=0).argmax() for t in trials], dtype=np.int64)


def aggregate_majority_vote(probs, trial_ids, num_classes=NUM_CLASSES):
    """SECONDARY: hard per-window argmax then modal class; ties -> larger summed posterior."""
    trials = np.unique(trial_ids)
    preds = np.empty(len(trials), dtype=np.int64)
    hard = probs.argmax(axis=1)
    for i, t in enumerate(trials):
        m = trial_ids == t
        counts = np.bincount(hard[m], minlength=num_classes)
        top = np.flatnonzero(counts == counts.max())
        preds[i] = top[0] if len(top) == 1 else top[np.argmax(probs[m].sum(axis=0)[top])]
    return trials, preds


def trial_labels(labels, trial_ids):
    trials = np.unique(trial_ids)
    out = np.empty(len(trials), dtype=np.int64)
    for i, t in enumerate(trials):
        lab = np.unique(labels[trial_ids == t])
        if len(lab) != 1:
            raise AssertionError(f"trial {t} carries {len(lab)} distinct labels {lab.tolist()}; "
                                 f"check the preprocessing's trial_ids/labels alignment")
        out[i] = lab[0]
    return out


def collect_fold_posteriors(eeg, fnirs, labels, test_idx, ckpt_path, device, batch_size, adabn):
    model = build_model(seg_len=fnirs.shape[1], device=device)
    # strict=True on purpose: a mismatch means the checkpoint belongs to a different architecture
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()
    loader = WindowLoader(eeg, fnirs, labels, test_idx, batch_size)        # index order, no shuffling
    if adabn:
        adapt_batchnorm_to_target(model, loader, device, log_fn=lambda *a, **k: None)
    d = collect_outputs(model, loader, device, keys=tuple(f"p_{h}" for h in HEADS))
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {h: d[f"p_{h}"] for h in HEADS}, d["y_true"]


def parse_subjects(spec):
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-"); ids.extend(range(int(a), int(b) + 1))
        elif part:
            ids.append(int(part))
    return [f"VP{i:03d}" for i in ids]


def find_results_dir():
    if ARGS.results_dir:
        return ARGS.results_dir
    d = os.path.join(ARGS.data_root, f"results_stanet_{ARGS.val_mode}_seed{ARGS.seed}")
    if os.path.isdir(d):
        return d
    cands = sorted(glob.glob(os.path.join(ARGS.data_root, f"results_stanet_*_seed{ARGS.seed}")))
    sys.exit(f"ERROR: {d} not found. Candidates for seed {ARGS.seed}: {cands or 'none'}. "
             f"Pass --results-dir.")


def recorded_window_acc(results_dir):
    """per-subject mean accuracy recorded by the training run (latest stanet_all_subjects_*.csv)."""
    files = sorted(glob.glob(os.path.join(results_dir, "stanet_all_subjects_*.csv")))
    if not files:
        return None
    rec = {}
    with open(files[-1], newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["status"] == "ok" and r["mean_acc"] != "":
                rec[r["subject_id"]] = float(r["mean_acc"])
    return rec or None


def main():
    device = torch.device(ARGS.device)
    results_dir = find_results_dir()
    if not os.path.isdir(os.path.join(ARGS.data_root, "VP001")):
        sys.exit(f"ERROR: --data-root has no VP001 folder: {ARGS.data_root}")
    out_prefix = ARGS.out_prefix or f"stanet_trial_level_eval_seed{ARGS.seed}"
    subjects = parse_subjects(ARGS.subjects)
    print("=" * 78)
    print(f"STA-Net TRIAL-LEVEL EVALUATION | seed={ARGS.seed} | device={device}")
    print(f"  results_dir : {results_dir}\n  data_root   : {ARGS.data_root}")
    print(f"  AdaBN       : {'ON (must match the training run)' if ARGS.adabn else 'OFF (STA-Net default)'}")
    print("=" * 78)

    rows = []
    pooled = {h: {k: [] for k in ("win_true", "win_pred", "tri_true", "tri_pred", "tri_pred_vote")}
              for h in HEADS}
    for sid in subjects:
        sdir = os.path.join(ARGS.data_root, sid)
        if not os.path.isdir(sdir):
            print(f"[{sid}] SKIP -- no preprocessed folder"); continue
        try:
            eeg, fnirs, labels, sessions, trials = load_subject(sdir, sid)
        except Exception as exc:
            print(f"[{sid}] SKIP -- could not load: {exc}"); continue
        for ts in sorted(np.unique(sessions)):
            _, test_idx = session_holdout_split(sessions, ts)
            ckpt = os.path.join(results_dir, f"{sid}_session{ts}_best.pt")
            if not os.path.isfile(ckpt):
                print(f"[{sid} s{ts}] SKIP -- no checkpoint {ckpt}"); continue
            try:
                post, seen = collect_fold_posteriors(eeg, fnirs, labels, test_idx, ckpt, device,
                                                     ARGS.batch_size, ARGS.adabn)
            except Exception as exc:
                print(f"[{sid} s{ts}] FAILED -- {type(exc).__name__}: {exc}"); continue
            y_win, tri_win = labels[test_idx], trials[test_idx]
            if len(seen) != len(y_win) or not np.array_equal(seen, y_win):
                print(f"[{sid} s{ts}] FAILED -- loader order != test_idx; refusing to aggregate"); continue
            y_tri = trial_labels(y_win, tri_win)
            row = {"subject_id": sid, "test_session": int(ts), "n_windows": int(len(y_win)),
                   "n_trials": int(len(y_tri))}
            for h in HEADS:
                pr = post[h]
                wp = pr.argmax(1)
                _, tp = aggregate_mean_prob(pr, tri_win)
                _, tv = aggregate_majority_vote(pr, tri_win)
                row[f"{h}_win_acc"] = float((wp == y_win).mean())
                row[f"{h}_win_kappa"] = float(cohen_kappa(y_win, wp))
                row[f"{h}_trial_acc"] = float((tp == y_tri).mean())
                row[f"{h}_trial_kappa"] = float(cohen_kappa(y_tri, tp))
                row[f"{h}_trial_acc_vote"] = float((tv == y_tri).mean())
                for k, v in (("win_true", y_win), ("win_pred", wp), ("tri_true", y_tri),
                             ("tri_pred", tp), ("tri_pred_vote", tv)):
                    pooled[h][k].append(v)
            rows.append(row)
            print(f"[{sid} s{ts}] window acc={row['final_win_acc']:.4f} -> trial acc={row['final_trial_acc']:.4f} "
                  f"(vote {row['final_trial_acc_vote']:.4f}) | {row['n_windows']} win / {row['n_trials']} trials")
        del eeg, fnirs

    if not rows:
        sys.exit("ERROR: no fold produced a result.")

    by_subj = {}
    for r in rows:
        by_subj.setdefault(r["subject_id"], []).append(r)
    ids = sorted(by_subj)
    m = lambda key: np.array([np.mean([r[key] for r in by_subj[s]]) for s in ids])
    s_wa, s_ta, s_wk, s_tk = m("final_win_acc"), m("final_trial_acc"), m("final_win_kappa"), m("final_trial_kappa")

    # ---------------- self-check vs the training run's own record ----------------
    grand_win = float(s_wa.mean())
    print("\n" + "=" * 78 + "\nSELF-CHECK -- does the checkpoint replay reproduce the training run?\n" + "=" * 78)
    print(f"  reproduced grand-mean window accuracy : {grand_win:.4f}")
    rec = recorded_window_acc(results_dir)
    expected, passed = ARGS.expected_window_acc, None
    if expected is None and rec:
        common = [s for s in ids if s in rec]
        if common:
            expected = float(np.mean([rec[s] for s in common]))
            worst = max(abs(np.mean([r["final_win_acc"] for r in by_subj[s]]) - rec[s]) for s in common)
            print(f"  per-subject max |replay - recorded|   : {worst:.4f} over {len(common)} subject(s)")
            grand_win = float(np.mean([np.mean([r["final_win_acc"] for r in by_subj[s]]) for s in common]))
    if expected is not None:
        delta = abs(grand_win - expected)
        passed = delta <= ARGS.reproduce_tolerance
        print(f"  on record for this run                : {expected:.4f}\n  |difference| : {delta:.4f} "
              f"(tolerance {ARGS.reproduce_tolerance})\n  VERDICT: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print("  *** replay does NOT reproduce the recorded window accuracy; trial-level numbers are not "
                  "comparable. Likely causes: --adabn differs from the training run; the .npy files were "
                  "regenerated; checkpoints belong to another seed/val-mode. ***")
    else:
        print("  (no recorded CSV found and no --expected-window-acc given: self-check skipped)")

    # ---------------- pooled ----------------
    pooled_summary = {}
    for h in HEADS:
        wt, wp = np.concatenate(pooled[h]["win_true"]), np.concatenate(pooled[h]["win_pred"])
        tt, tp = np.concatenate(pooled[h]["tri_true"]), np.concatenate(pooled[h]["tri_pred"])
        tv = np.concatenate(pooled[h]["tri_pred_vote"])
        pooled_summary[h] = {
            "n_windows": int(len(wt)), "n_trials": int(len(tt)),
            "window_acc_pooled": float((wp == wt).mean()), "window_kappa_pooled": float(cohen_kappa(wt, wp)),
            "trial_acc_pooled_meanprob": float((tp == tt).mean()),
            "trial_kappa_pooled_meanprob": float(cohen_kappa(tt, tp)),
            "trial_acc_pooled_vote": float((tv == tt).mean()), "trial_kappa_pooled_vote": float(cohen_kappa(tt, tv)),
            "window_confusion_pooled": confusion_matrix_np(wt, wp).tolist(),
            "trial_confusion_pooled_meanprob": confusion_matrix_np(tt, tp).tolist()}

    # ---------------- Wilcoxon (paired by subject; not reported below 6 non-tied pairs) ----------------
    MIN_N = 6
    stats = {}
    for name, a, b in (("accuracy", s_wa, s_ta), ("kappa", s_wk, s_tk)):
        diff = b - a
        nz = int((diff != 0).sum())
        blk = {"n_subjects": int(len(a)), "n_nonzero_pairs": nz, "window_level_mean": float(a.mean()),
               "trial_level_mean": float(b.mean()), "mean_improvement": float(diff.mean()),
               "n_subjects_improved": int((diff > 0).sum()), "n_subjects_worsened": int((diff < 0).sum()),
               "n_subjects_tied": int((diff == 0).sum()), "wilcoxon_statistic": None, "p_value": None}
        if nz == 0:
            blk["note"] = "all differences are zero; test undefined"
        elif nz < MIN_N:
            blk["note"] = f"only {nz} non-tied pair(s); p<0.05 unreachable below {MIN_N}, no p-value reported"
        else:
            try:
                from scipy.stats import wilcoxon
                st, pv = wilcoxon(a, b)
                blk["wilcoxon_statistic"], blk["p_value"] = float(st), float(pv)
            except Exception as exc:
                blk["note"] = f"Wilcoxon failed ({type(exc).__name__}: {exc})"
        stats[name] = blk

    print("\n" + "=" * 78 + "\nWINDOW-LEVEL vs TRIAL-LEVEL (final head)\n" + "=" * 78)
    print(f"  subjects: {len(ids)}")
    print(f"  per-subject mean, window level: acc={s_wa.mean():.4f} +-{s_wa.std():.4f}  kappa={s_wk.mean():.4f}")
    print(f"  per-subject mean, trial level : acc={s_ta.mean():.4f} +-{s_ta.std():.4f}  kappa={s_tk.mean():.4f}")
    print(f"  improvement: {(s_ta.mean() - s_wa.mean()) * 100:+.2f} pp accuracy, {s_tk.mean() - s_wk.mean():+.4f} kappa")
    sa = stats["accuracy"]
    if sa["p_value"] is not None:
        print(f"  Wilcoxon (paired, {sa['n_subjects']} subjects): p={sa['p_value']:.3g} | improved in "
              f"{sa['n_subjects_improved']}/{sa['n_subjects']}")
    else:
        print(f"  Wilcoxon: not reported -- {sa.get('note', 'unavailable')}")
    print("\nPOOLED OVER ALL SUBJECTS AND FOLDS")
    for h in HEADS:
        s = pooled_summary[h]
        print(f"  [{h:>6}] window {s['window_acc_pooled']:.4f} (n={s['n_windows']}) -> trial "
              f"{s['trial_acc_pooled_meanprob']:.4f} (n={s['n_trials']}, vote {s['trial_acc_pooled_vote']:.4f})")
    print("\nCAVEATS\n  * 9 trials per test session -> trial accuracy is quantised to 1/9; interpret the pooled "
          "estimate only.\n  * Report BOTH levels: window level for comparability with TSFNet/MSTFDN, trial level "
          "as the applied block decision.\n  * Aggregation uses only each test trial's own posteriors (no labels).")

    csv_path, json_path = (os.path.join(results_dir, f"{out_prefix}.{e}") for e in ("csv", "json"))
    fields = ["subject_id", "test_session", "n_windows", "n_trials"]
    for h in HEADS:
        fields += [f"{h}_win_acc", f"{h}_win_kappa", f"{h}_trial_acc", f"{h}_trial_kappa", f"{h}_trial_acc_vote"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"config": {"seed": ARGS.seed, "val_mode": ARGS.val_mode, "results_dir": results_dir,
                              "data_root": ARGS.data_root, "adabn": ARGS.adabn,
                              "aggregation_primary": "mean posterior over a trial's windows, then argmax",
                              "aggregation_secondary": "per-window argmax, modal class (ties: summed posterior)"},
                   "self_check": {"reproduced_grand_mean_window_acc": grand_win, "expected_window_acc": expected,
                                  "tolerance": ARGS.reproduce_tolerance, "passed": passed},
                   "per_subject_mean_final_head": {"subjects": ids, "window_acc": s_wa.tolist(),
                                                   "trial_acc": s_ta.tolist(), "window_kappa": s_wk.tolist(),
                                                   "trial_kappa": s_tk.tolist()},
                   "pooled": pooled_summary, "wilcoxon_window_vs_trial": stats}, f, indent=2)
    print(f"\n[written] {csv_path}\n[written] {json_path}")


if __name__ == "__main__":
    main()
