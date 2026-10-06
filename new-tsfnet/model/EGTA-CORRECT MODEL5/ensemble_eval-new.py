"""
ensemble_eval-new.py -- seed-ensemble evaluation from SAVED checkpoints (no training).

WHY
===
Across the three run60 seeds the per-fold accuracy moves a lot while the
subject ranking barely does:
    within-fold SD across seeds  = 6.6 points
    between-fold SD              = 13.4 points
    subject-rank correlation     = 0.88
So a large part of the error is MODEL VARIANCE (which minimum the optimiser
landed in) on top of a stable, data-determined difficulty. Averaging the
posteriors of independently trained models removes the variance part (and only
that part). This script measures how much that is worth, using the checkpoints
that already exist -- one forward pass per model per fold, no gradients.

WHAT IT DOES
============
For every subject and every held-out session:
  1. for every seed: load VPxxx_session{k}_best.pt, RE-APPLY AdaBN on the held-out
     session's unlabeled inputs (AdaBN is NOT stored in the checkpoint -- see
     trial_level_eval-new.py, which this script mirrors), collect the posteriors
     of the four heads (final / eeg / fusion / fnirs);
  2. combine the seeds' posteriors: ARITHMETIC mean (primary) and GEOMETRIC mean
     (mean of log-probabilities, for comparison);
  3. score window level and block (trial) level -- the trial decision is the mean
     posterior over a block's windows, then argmax (same rule as
     trial_level_eval-new.py), plus the majority-vote variant.
Then, over subjects (the unit of the paired tests):
  * per-seed single-model accuracy, the mean of the singles, the ensemble, and
    the PAIRED difference ensemble - mean(singles) with a 95% CI, the number of
    subjects improved, and Wilcoxon / paired-t p-values;
  * an ensemble-size curve (all subsets of 1..K seeds), to see whether a 4th or
    5th seed would still pay;
  * the same table for each head.

NORMALISATION
=============
The "final" head is a sigmoid-weighted MEAN of three softmax outputs and does not
sum to 1 (it is normalised only inside the loss, --normalize-ypred). Averaging
un-normalised posteriors across models would let a model with a larger sum
dominate, so by default every posterior row is divided by its sum before it is
combined. argmax is unchanged, so the window-level numbers of a single model are
identical to trial_level_eval-new.py; block-level singles can differ in the 3rd
decimal. --raw-posteriors disables the division (for an exact cross-check).

HONEST CAVEATS (they are printed again at the end)
==================================================
* AdaBN is transductive (unlabeled target-session inputs). Same accepted project
  choice as the baseline, applied to each ensemble member separately.
* The ensemble is K trained networks: report it as such (K x the parameters and
  K x the inference cost). It is a legitimate final model, not a free lunch.
* Ensembling reduces variance, not the data-determined part (e.g. VP006 session 2
  fails in every seed). Do not expect it to move failing folds.
* Each test session has 9 blocks, so block accuracy is quantised to 1/9 per fold.
* Seeds 42/999/2024 are the only models available; the K-curve for K=3 has only
  3 pairs and 1 triple -- treat it as indicative.

USAGE
=====
    python ensemble_eval-new.py ^
        --data-root "E:\\kntu\\Thesis\\codes\\Results\\TSFNET_PAPER_EXACT_nback" ^
        --variant-hash 40b7d8ea53 --seeds 42 999 2024 ^
        --expected-window-accs 0.6074 0.5961 0.6091 --save-posteriors

Every architecture flag defaults to run60. For another configuration pass the same
architecture flags that were used for ITS training (as for trial_level_eval-new.py);
all ensemble members must share one architecture.
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

p = argparse.ArgumentParser(
    description="Seed-ensemble evaluation from saved checkpoints.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--data-root", type=str,
               default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback",
               help="Folder with VP001..VP026 (preprocessed .npy) and the results_* folders.")
p.add_argument("--seeds", type=int, nargs="+", default=[42, 999, 2024],
               help="Seeds whose checkpoints form the ensemble.")
p.add_argument("--variant-hash", type=str, default="40b7d8ea53",
               help="Hash in results_<hash>_seed<N> (run60-onestage = 40b7d8ea53).")
p.add_argument("--results-dirs", type=str, nargs="+", default=None,
               help="Explicit results folders, one per seed IN THE SAME ORDER as --seeds. "
                    "Overrides --variant-hash.")
p.add_argument("--subjects", type=str, default="1-26")
p.add_argument("--batch-size", type=int, default=16,
               help="Must equal the training batch size so AdaBN partitions the same way.")
p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--out-dir", type=str, default=None,
               help="Where to write outputs. Default: <data-root>/ensemble_eval_<hash>_seeds-<...>.")
p.add_argument("--expected-window-accs", type=float, nargs="+", default=None,
               help="Grand-mean window accuracy on record for each seed, same order as --seeds "
                    "(run60: 0.6074 0.5961 0.6091). Enables a PASS/FAIL replay self-check.")
p.add_argument("--reproduce-tolerance", type=float, default=0.002)
p.add_argument("--adabn", dest="adabn", action="store_true", default=True)
p.add_argument("--no-adabn", dest="adabn", action="store_false")
p.add_argument("--raw-posteriors", action="store_true", default=False,
               help="Do not divide posterior rows by their sum before combining (see docstring).")
p.add_argument("--save-posteriors", action="store_true", default=False,
               help="Also save every fold's per-seed posteriors, labels, trial ids and window "
                    "position within the block (npz) for later analysis.")

# --- architecture flags (TSFNet ctor), defaults = run60 (same as trial_level_eval-new.py) ---
p.add_argument("--fnirs-temporal-align", type=str, default="egta", choices=["none", "egta"])
p.add_argument("--efgf-gate-mode", type=str, default="sigmoid", choices=["clamp", "sigmoid"])
p.add_argument("--efgf-gate-init", type=float, default=-2.0)
p.add_argument("--fusion-combination", type=str, default="paper_mean",
               choices=["paper_mean", "wsum", "gated", "stanet2", "stanet3"])
p.add_argument("--fnirs-num-segments", type=int, default=21)
p.add_argument("--cafe-d-model", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--cafe-dropout", type=float, default=0.3)
p.add_argument("--eeg-temporal-mode", type=str, default="gap", choices=["gap", "gap_logvar", "bilstm"])
p.add_argument("--fusion-fnirs-inject", type=str, default="none", choices=["none", "stage1", "both"])
p.add_argument("--erp-branch", type=str, default="none", choices=["none", "eeg", "eeg_fusion"])
p.add_argument("--eeg-filterbank", type=str, default="none", choices=["none", "fixed", "learnable"])
p.add_argument("--eeg-filterbank-scope", type=str, default="both", choices=["both", "eeg"])
p.add_argument("--fnirs-chroma-split", type=str, default="none",
               choices=["none", "separate", "separate_pair"])
p.add_argument("--eeg-lstm-hidden", type=int, default=64)
p.add_argument("--eeg-readout-mode", type=str, default="h_n", choices=["h_n", "last_step", "mean_pool"])
p.add_argument("--fnirs-chroma-mode", type=str, default="both")
p.add_argument("--fnirs-num-chromophores", type=int, default=2, choices=[2, 3])
p.add_argument("--eeg-spatial-mode", type=str, default="none")
p.add_argument("--bandtok-norm", type=str, default="none", choices=["none", "relative"])
p.add_argument("--bandtok-d-model", type=int, default=32)
p.add_argument("--bandtok-heads", type=int, default=4)
p.add_argument("--eeg-band-mode", type=str, default="none")
p.add_argument("--eeg-backbone", type=str, default="native")
p.add_argument("--norm-mode", type=str, default="batch", choices=["batch", "group"])
p.add_argument("--use-fc-features", action="store_true", default=False)

ARGS = p.parse_args()

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
_notebook_path = os.path.join(_this_dir, "tsfnet_model_notebook-new.py")
if not os.path.isfile(_notebook_path):
    sys.exit(f"ERROR: tsfnet_model_notebook-new.py not found next to this script ({_notebook_path}).")
if "TSFNet" not in globals():
    print(f"[init] loading definitions from {_notebook_path}")
    exec(open(_notebook_path, encoding="utf-8").read(), globals())

HEADS = ("final", "eeg", "fusion", "fnirs")


# ---------------------------------------------------------------------------
# Decision rules (identical to trial_level_eval-new.py)
# ---------------------------------------------------------------------------
def aggregate_mean_prob(probs, trial_ids):
    trials = np.unique(trial_ids)
    preds = np.empty(len(trials), dtype=np.int64)
    for i, t in enumerate(trials):
        preds[i] = probs[trial_ids == t].mean(axis=0).argmax()
    return trials, preds


def aggregate_majority_vote(probs, trial_ids, num_classes=3):
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
            raise AssertionError(f"trial {t} carries {len(lab)} distinct labels")
        out[i] = lab[0]
    return out


# ---------------------------------------------------------------------------
# Posterior combination.  P has shape (K, N, C): K models, N windows, C classes.
# ---------------------------------------------------------------------------
def combine_arith(P):
    return P.mean(axis=0)


def combine_geo(P):
    L = np.log(np.clip(P, 1e-8, 1.0)).mean(axis=0)
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def score(probs, y_win, tri_ids, y_tri):
    """window acc, trial acc (mean-posterior rule), trial acc (majority vote)."""
    win = float((probs.argmax(axis=1) == y_win).mean())
    _t, tp = aggregate_mean_prob(probs, tri_ids)
    _t2, tv = aggregate_majority_vote(probs, tri_ids)
    return win, float((tp == y_tri).mean()), float((tv == y_tri).mean())


def fmt_stats(st):
    """One-line summary of a paired_stats dict; tolerates n=1 (no CI / p-values)."""
    ci = ("n/a" if st["ci95"] is None
          else f"[{st['ci95'][0] * 100:+.2f}, {st['ci95'][1] * 100:+.2f}]")
    pw = "n/a" if st["p_wilcoxon"] is None else f"{st['p_wilcoxon']:.3g}"
    pt = "n/a" if st["p_ttest"] is None else f"{st['p_ttest']:.3g}"
    return (f"{st['mean_diff'] * 100:+.2f} pp  95% CI {ci}  improved {st['n_improved']}/{st['n']} "
            f"(worse {st['n_worsened']})  Wilcoxon p={pw}  t-test p={pt}")


def paired_stats(a, b):
    """Paired comparison b - a over subjects (arrays of equal length)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    n = len(d)
    out = {"n": int(n), "mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)) if n > 1 else None,
           "n_improved": int((d > 0).sum()), "n_worsened": int((d < 0).sum()), "n_tied": int((d == 0).sum()),
           "se": None, "ci95": None, "p_wilcoxon": None, "p_ttest": None}
    if n > 1:
        se = float(d.std(ddof=1) / np.sqrt(n))
        out["se"] = se
        try:
            from scipy import stats
            tcrit = float(stats.t.ppf(0.975, n - 1))
            out["ci95"] = [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)]
            if (d != 0).sum() >= 6:
                out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
            if se > 0:
                out["p_ttest"] = float(stats.ttest_rel(b, a).pvalue)
        except Exception:                                              # scipy missing -> descriptive only
            out["ci95"] = [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]
    return out


# ---------------------------------------------------------------------------
# One model on one fold: load checkpoint, re-apply AdaBN, collect posteriors
# ---------------------------------------------------------------------------
def build_model(device):
    return TSFNet(
        num_classes=3, dropout=ARGS.dropout, cafe_dropout=ARGS.cafe_dropout,
        cafe_d_model=ARGS.cafe_d_model, fusion_combination=ARGS.fusion_combination,
        eeg_temporal_mode=ARGS.eeg_temporal_mode, eeg_lstm_hidden=ARGS.eeg_lstm_hidden,
        eeg_readout_mode=ARGS.eeg_readout_mode, fnirs_num_segments=ARGS.fnirs_num_segments,
        fnirs_chroma_mode=ARGS.fnirs_chroma_mode, fnirs_num_chromophores=ARGS.fnirs_num_chromophores,
        use_fc_features=ARGS.use_fc_features, eeg_spatial_mode=ARGS.eeg_spatial_mode,
        eeg_backbone_mode=ARGS.eeg_backbone, fnirs_temporal_align=ARGS.fnirs_temporal_align,
        eeg_band_mode=ARGS.eeg_band_mode, norm_mode=ARGS.norm_mode,
        efgf_gate_mode=ARGS.efgf_gate_mode, efgf_gate_init=ARGS.efgf_gate_init,
        fusion_fnirs_inject=ARGS.fusion_fnirs_inject, erp_branch=ARGS.erp_branch,
        eeg_filterbank=ARGS.eeg_filterbank, eeg_filterbank_scope=ARGS.eeg_filterbank_scope,
        fnirs_chroma_split=ARGS.fnirs_chroma_split,
        bandtok_d_model=ARGS.bandtok_d_model, bandtok_heads=ARGS.bandtok_heads,
        bandtok_norm=ARGS.bandtok_norm,
    ).to(device)


def collect_fold_posteriors(full, test_idx, ckpt_path, device):
    model = build_model(device)
    # strict=True: a key mismatch means the architecture flags differ from the run that
    # wrote this checkpoint; scoring a partly random model silently would be worse.
    model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    model.eval()
    loader = DataLoader(Subset(full, test_idx), batch_size=ARGS.batch_size, shuffle=False)
    if ARGS.adabn:
        adapt_batchnorm_to_target(model, loader, device, log_fn=lambda *a, **k: None)
    out = {h: [] for h in HEADS}
    seen = []
    with torch.no_grad():
        for _batch in loader:
            x_eeg, x_fnirs, y, x_conn, x_chan = _unpack_batch(_batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            y_eeg, y_fusion, y_fnirs, y_pred, _aux = model(x_eeg, x_fnirs, x_conn, x_chan)
            for h, v in zip(HEADS, (y_pred, y_eeg, y_fusion, y_fnirs)):
                out[h].append(v.cpu().numpy())
            seen.append(y.numpy())
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    res = {h: np.concatenate(v, axis=0).astype(np.float64) for h, v in out.items()}
    if not ARGS.raw_posteriors:
        for h in HEADS:
            res[h] = res[h] / np.clip(res[h].sum(axis=1, keepdims=True), 1e-12, None)
    return res, np.concatenate(seen)


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
    seeds = ARGS.seeds
    if ARGS.results_dirs is not None:
        if len(ARGS.results_dirs) != len(seeds):
            sys.exit(f"ERROR: --results-dirs has {len(ARGS.results_dirs)} entries but --seeds has {len(seeds)}.")
        dirs = list(ARGS.results_dirs)
    else:
        dirs = []
        for s in seeds:
            name = f"results_{ARGS.variant_hash}_seed{s}"
            d = os.path.join(ARGS.data_root, name)
            if not os.path.isdir(d):
                roots, q = [], os.path.abspath(ARGS.data_root)
                for _ in range(4):
                    roots.append(q)
                    q = os.path.dirname(q)
                roots.append(_this_dir)
                found = locate_results_dir(name, roots)
                if len(found) == 1:
                    d = found[0]
                elif len(found) > 1:
                    sys.exit(f"ERROR: '{name}' exists in several places: {found}. Pass --results-dirs.")
                else:
                    sys.exit(f"ERROR: could not find '{name}' under {ARGS.data_root}. "
                             f"Pass --results-dirs (one per seed, same order as --seeds).")
            dirs.append(d)
    for s, d in zip(seeds, dirs):
        if not os.path.isdir(d):
            sys.exit(f"ERROR: results folder for seed {s} does not exist: {d}")
    return dirs


def main():
    device = torch.device(ARGS.device)
    seeds, K = ARGS.seeds, len(ARGS.seeds)
    if len(set(seeds)) != K:
        sys.exit("ERROR: duplicate seeds in --seeds.")
    if ARGS.expected_window_accs is not None and len(ARGS.expected_window_accs) != K:
        sys.exit("ERROR: --expected-window-accs needs one value per seed.")
    dirs = resolve_results_dirs()
    if not os.path.isdir(os.path.join(ARGS.data_root, "VP001")) and ARGS.subjects == "1-26":
        sys.exit(f"ERROR: --data-root has no VP001 folder: {ARGS.data_root}")
    out_dir = ARGS.out_dir or os.path.join(
        ARGS.data_root, f"ensemble_eval_{ARGS.variant_hash}_seeds-" + "-".join(map(str, seeds)))
    os.makedirs(out_dir, exist_ok=True)
    subjects = parse_subjects(ARGS.subjects)

    print("=" * 78)
    print(f"SEED-ENSEMBLE EVALUATION | K={K} seeds={seeds} | device={device}")
    for s, d in zip(seeds, dirs):
        print(f"  seed {s:>5}: {d}")
    print(f"  data_root : {ARGS.data_root}")
    print(f"  out_dir   : {out_dir}")
    print(f"  AdaBN     : {'ON' if ARGS.adabn else 'OFF (will not reproduce the record)'} | "
          f"posterior normalisation: {'OFF (raw)' if ARGS.raw_posteriors else 'ON (rows sum to 1)'}")
    print("=" * 78)

    folds = []            # one dict per (subject, session)
    trial_rows = []
    combos = {k: list(itertools.combinations(range(K), k)) for k in range(1, K + 1)}

    for sid in subjects:
        subject_dir = os.path.join(ARGS.data_root, sid)
        if not os.path.isdir(subject_dir):
            print(f"[{sid}] SKIP -- no preprocessed folder")
            continue
        try:
            full = NBackDataset(subject_dir, sid, load_erp=(ARGS.erp_branch != "none"))
        except Exception as exc:
            print(f"[{sid}] SKIP -- could not load dataset: {exc}")
            continue
        labels_np = full.labels.numpy()
        for test_session in sorted(np.unique(full.session_ids)):
            _tr, test_idx = session_holdout_split(full.session_ids, test_session)
            y_win = labels_np[test_idx]
            tri_ids = full.trial_ids[test_idx]
            ckpts = [os.path.join(d, f"{sid}_session{test_session}_best.pt") for d in dirs]
            missing = [c for c in ckpts if not os.path.isfile(c)]
            if missing:
                print(f"[{sid} s{test_session}] SKIP -- missing checkpoint(s): {missing}")
                continue
            try:
                post, ok = [], True
                for c in ckpts:
                    pst, seen = collect_fold_posteriors(full, test_idx, c, device)
                    if len(seen) != len(y_win) or not np.array_equal(seen, y_win):
                        print(f"[{sid} s{test_session}] FAILED -- loader order != test_idx; refusing to aggregate.")
                        ok = False
                        break
                    post.append(pst)
                if not ok:
                    continue
            except Exception as exc:
                print(f"[{sid} s{test_session}] FAILED -- {type(exc).__name__}: {exc}")
                continue

            y_tri = trial_labels(y_win, tri_ids)
            fold = {"subject_id": sid, "test_session": int(test_session),
                    "n_windows": int(len(y_win)), "n_trials": int(len(y_tri))}
            # per head: singles, arithmetic ensemble, geometric ensemble
            for h in HEADS:
                P = np.stack([post[k][h] for k in range(K)], axis=0)          # (K, N, 3)
                fold[h] = {
                    "single": [score(P[k], y_win, tri_ids, y_tri) for k in range(K)],
                    "arith": score(combine_arith(P), y_win, tri_ids, y_tri),
                    "geo": score(combine_geo(P), y_win, tri_ids, y_tri),
                }
            # ensemble-size curve (final head, arithmetic): average over all subsets of each size
            P = np.stack([post[k]["final"] for k in range(K)], axis=0)
            fold["curve"] = {k: np.mean([score(combine_arith(P[list(c)]), y_win, tri_ids, y_tri)
                                         for c in combos[k]], axis=0).tolist() for k in range(1, K + 1)}
            folds.append(fold)

            ens_p = combine_arith(P)
            for t in np.unique(tri_ids):
                m = tri_ids == t
                mp = ens_p[m].mean(axis=0)
                row = {"subject_id": sid, "test_session": int(test_session), "trial_id": int(t),
                       "label": int(y_win[m][0]), "pred_ensemble": int(mp.argmax()),
                       "p0": float(mp[0]), "p1": float(mp[1]), "p2": float(mp[2])}
                for k, s in enumerate(seeds):
                    row[f"pred_seed{s}"] = int(P[k][m].mean(axis=0).argmax())
                trial_rows.append(row)

            if ARGS.save_posteriors:
                pos = np.zeros(len(tri_ids), dtype=np.int32)
                for t in np.unique(tri_ids):
                    idx = np.flatnonzero(tri_ids == t)
                    pos[idx] = np.arange(len(idx))
                pdir = os.path.join(out_dir, "posteriors")
                os.makedirs(pdir, exist_ok=True)
                np.savez_compressed(
                    os.path.join(pdir, f"{sid}_session{test_session}.npz"),
                    seeds=np.array(seeds), y=y_win, trial_ids=tri_ids, pos_in_trial=pos,
                    **{f"P_{h}": np.stack([post[k][h] for k in range(K)], axis=0).astype(np.float32) for h in HEADS})

            f = fold["final"]
            print(f"[{sid} s{test_session}] singles(win)=" +
                  "/".join(f"{x[0]:.3f}" for x in f["single"]) +
                  f" -> ensemble win={f['arith'][0]:.4f} trial={f['arith'][1]:.4f} "
                  f"| {fold['n_windows']} win / {fold['n_trials']} trials")

    if not folds:
        sys.exit("ERROR: no fold produced a result.")

    # ------------------------- per-subject means -----------------------------
    subj = sorted(set(f["subject_id"] for f in folds))

    def subj_mean(getter):
        return np.array([np.mean([getter(f) for f in folds if f["subject_id"] == s]) for s in subj])

    def metric_table(head):
        """Per-subject arrays for each of {window, trial, vote}: singles (K), mean_single, arith, geo."""
        T = {}
        for mi, mname in enumerate(("window", "trial", "vote")):
            singles = [subj_mean(lambda f, k=k, mi=mi: f[head]["single"][k][mi]) for k in range(K)]
            T[mname] = {"singles": singles, "mean_single": np.mean(singles, axis=0),
                        "arith": subj_mean(lambda f, mi=mi: f[head]["arith"][mi]),
                        "geo": subj_mean(lambda f, mi=mi: f[head]["geo"][mi])}
        return T

    tables = {h: metric_table(h) for h in HEADS}
    fin = tables["final"]

    # ------------------------- self-check ------------------------------------
    print()
    print("=" * 78)
    print("SELF-CHECK -- does the checkpoint replay reproduce each seed's record?")
    print("=" * 78)
    all_ok = True
    for k, s in enumerate(seeds):
        g = float(fin["window"]["singles"][k].mean())
        line = f"  seed {s:>5}: reproduced grand window acc {g:.4f}"
        if ARGS.expected_window_accs is not None:
            exp = ARGS.expected_window_accs[k]
            ok = abs(g - exp) <= ARGS.reproduce_tolerance
            all_ok &= ok
            line += f" | on record {exp:.4f} | |diff| {abs(g-exp):.4f} -> {'PASS' if ok else 'FAIL'}"
        print(line)
    if ARGS.expected_window_accs is None:
        print("  (pass --expected-window-accs 0.6074 0.5961 0.6091 for an automatic check)")
    elif not all_ok:
        print("  *** FAIL: the replay does not reproduce the record; every number below is then NOT "
              "comparable to the baseline. Likely causes: --adabn missing, --batch-size != 16, an "
              "architecture flag that differs from training, or regenerated .npy files. ***")

    # ------------------------- main report -----------------------------------
    print()
    print("=" * 78)
    print(f"ENSEMBLE vs SINGLE MODELS (final head, mean over {len(subj)} subjects)")
    print("=" * 78)
    print(f"  {'':<26}{'window':>10}{'block(mean-p)':>16}{'block(vote)':>14}")
    for k, s in enumerate(seeds):
        print(f"  {'single, seed ' + str(s):<26}" + "".join(
            f"{fin[m]['singles'][k].mean():>{w}.4f}" for m, w in (("window", 10), ("trial", 16), ("vote", 14))))
    print(f"  {'MEAN of the singles':<26}" + "".join(
        f"{fin[m]['mean_single'].mean():>{w}.4f}" for m, w in (("window", 10), ("trial", 16), ("vote", 14))))
    print(f"  {'ENSEMBLE (arithmetic)':<26}" + "".join(
        f"{fin[m]['arith'].mean():>{w}.4f}" for m, w in (("window", 10), ("trial", 16), ("vote", 14))))
    print(f"  {'ensemble (geometric)':<26}" + "".join(
        f"{fin[m]['geo'].mean():>{w}.4f}" for m, w in (("window", 10), ("trial", 16), ("vote", 14))))

    summary = {"config": {"seeds": seeds, "results_dirs": dirs, "adabn": ARGS.adabn,
                          "normalised_posteriors": not ARGS.raw_posteriors, "batch_size": ARGS.batch_size,
                          "n_subjects": len(subj), "n_folds": len(folds)},
               "self_check": {"reproduced": {str(s): float(fin['window']['singles'][k].mean())
                                             for k, s in enumerate(seeds)},
                              "expected": (None if ARGS.expected_window_accs is None
                                           else {str(s): v for s, v in zip(seeds, ARGS.expected_window_accs)}),
                              "passed": (None if ARGS.expected_window_accs is None else bool(all_ok))},
               "paired": {}, "size_curve": {}, "heads": {}}

    print()
    print("PAIRED over subjects: ensemble(arithmetic) - mean(singles)   [positive = ensemble better]")
    for mname, label in (("window", "window"), ("trial", "block (mean-posterior)"), ("vote", "block (vote)")):
        st = paired_stats(fin[mname]["mean_single"], fin[mname]["arith"])
        st_geo = paired_stats(fin[mname]["mean_single"], fin[mname]["geo"])
        summary["paired"][mname] = {"arith": st, "geo": st_geo}
        print(f"  {label:<24} {fmt_stats(st)}")
    print("  vs each single seed (window):")
    for k, s in enumerate(seeds):
        st = paired_stats(fin["window"]["singles"][k], fin["window"]["arith"])
        summary["paired"][f"window_vs_seed{s}"] = st
        print(f"    seed {s:>5}: {st['mean_diff'] * 100:+.2f} pp, improved {st['n_improved']}/{st['n']}")

    print()
    print("ENSEMBLE-SIZE CURVE (final head, arithmetic; average over all subsets of that size)")
    print(f"  {'K':>3}{'#subsets':>10}{'window':>10}{'block(mean-p)':>16}{'block(vote)':>14}")
    for k in range(1, K + 1):
        vals = np.array([[np.mean([f["curve"][k][mi] for f in folds if f["subject_id"] == s])
                          for mi in range(3)] for s in subj]).mean(axis=0)
        summary["size_curve"][str(k)] = {"n_subsets": len(combos[k]), "window": float(vals[0]),
                                         "trial": float(vals[1]), "vote": float(vals[2])}
        print(f"  {k:>3}{len(combos[k]):>10}{vals[0]:>10.4f}{vals[1]:>16.4f}{vals[2]:>14.4f}")
    if K >= 3:
        print("  (compare the increments: a shrinking gain from K-1 to K suggests more seeds would add little)")

    print()
    print("PER HEAD (window acc, mean over subjects): mean of singles -> ensemble (arithmetic)")
    for h in HEADS:
        w = tables[h]["window"]
        st = paired_stats(w["mean_single"], w["arith"])
        summary["heads"][h] = {"mean_single": float(w["mean_single"].mean()), "ensemble": float(w["arith"].mean()),
                               "paired": st}
        print(f"  [{h:>6}] {w['mean_single'].mean():.4f} -> {w['arith'].mean():.4f}  "
              f"({st['mean_diff'] * 100:+.2f} pp, improved {st['n_improved']}/{st['n']})")

    # failing folds do not improve with ensembling: show it
    fold_single = np.array([np.mean([x[0] for x in f["final"]["single"]]) for f in folds])
    fold_ens = np.array([f["final"]["arith"][0] for f in folds])
    low = fold_single < 0.45
    if low.any():
        summary["low_folds"] = {"n": int(low.sum()), "mean_single": float(fold_single[low].mean()),
                                "mean_ensemble": float(fold_ens[low].mean())}
        print()
        print(f"FOLDS WITH MEAN-SINGLE WINDOW ACC < 0.45 ({int(low.sum())} folds): "
              f"single {fold_single[low].mean():.3f} -> ensemble {fold_ens[low].mean():.3f}  "
              f"(shows whether variance reduction helps the data-determined failures)")

    print()
    print("CAVEATS")
    print("  * The ensemble is K networks: K x parameters and K x inference cost. Report it as an ensemble.")
    print("  * AdaBN is transductive (unlabeled target inputs); applied to each member separately.")
    print("  * Ensembling removes MODEL variance only; subject/fold-specific failures stay.")
    print("  * 9 blocks per test session: block accuracy is quantised to 1/9 per fold; trust pooled numbers.")
    print("  * Window level is the figure comparable with TSFNet/MSTFDN; block level is the applied metric.")

    # ------------------------- write -----------------------------------------
    with open(os.path.join(out_dir, "ensemble_folds.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["subject_id", "test_session", "n_windows", "n_trials"]
        for s in seeds:
            cols += [f"win_acc_seed{s}", f"block_acc_seed{s}"]
        cols += ["ens_win_acc", "ens_block_acc", "ens_block_acc_vote",
                 "ens_geo_win_acc", "ens_geo_block_acc"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for fo in folds:
            row = {c: fo[c] for c in cols[:4]}
            for k, s in enumerate(seeds):
                row[f"win_acc_seed{s}"], row[f"block_acc_seed{s}"] = fo["final"]["single"][k][:2]
            row["ens_win_acc"], row["ens_block_acc"], row["ens_block_acc_vote"] = fo["final"]["arith"]
            row["ens_geo_win_acc"], row["ens_geo_block_acc"] = fo["final"]["geo"][:2]
            w.writerow(row)
    with open(os.path.join(out_dir, "ensemble_trials.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
        w.writeheader()
        w.writerows(trial_rows)
    with open(os.path.join(out_dir, "ensemble_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f"[written] {out_dir}{os.sep}ensemble_folds.csv | ensemble_trials.csv | ensemble_summary.json"
          + (" | posteriors/*.npz" if ARGS.save_posteriors else ""))


if __name__ == "__main__":
    main()
