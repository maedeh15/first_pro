"""
ensemble_seeds.py -- priority-2 of the final roadmap: 3-seed ensemble of the
ACCEPTED configuration (E3 + E4/AdaBN + EGTA), using the checkpoints that
already exist. No training. GPU optional (CPU works, just slower).

WHAT IT DOES, PER SUBJECT x TEST SESSION
----------------------------------------
  1. Rebuilds the held-out test set EXACTLY as run_subject_cross_session does:
     NBackDataset -> session_holdout_split -> DataLoader(batch=16, shuffle=False).
     (E3 per-session normalization is already baked into the saved .npy
     windows by preprocessing, so loading them reproduces the exact inputs.)
  2. For each seed: builds TSFNet(fnirs_temporal_align='egta') with every
     other argument at its default (= the reference config), loads that
     seed's <subject>_session<k>_best.pt with strict=True (any architecture
     mismatch raises instead of silently mis-loading), then RE-APPLIES AdaBN
     on the test loader. The .pt deliberately stores the UNADAPTED model
     (see the comment above tta_mode in train_one_fold), so skipping this
     step would ensemble a different model from the one that was reported.
  3. REPRODUCTION GATE (mandatory): each seed's recomputed single-model
     accuracy is compared with test_accuracy_final in that fold's saved
     results_<subject>_session<k>.json. If they disagree beyond REPRO_TOL the
     fold is flagged, and if any fold fails the script refuses to present the
     ensemble as a valid result. An ensemble built on models that do not
     reproduce their own reported numbers means nothing.
  4. Ensembles the three seeds by averaging y_pred (the model's final
     weighted output) and taking argmax; also reports majority vote.

OUTPUTS (written to OUT_DIR)
----------------------------
  ensemble_per_subject.csv   per-subject: each seed's acc/kappa + ensemble
  ensemble_summary.json      macro means, reproduction status, Wilcoxon tests
  and a printed summary.

The seed-to-seed comparison uses the Wilcoxon signed-rank test across the 26
subjects (paired, non-parametric) -- the statistic the final roadmap marks
as required for the thesis defence.

USAGE
-----
    %run ensemble_seeds.py
Edit the CONFIG block below if your folders differ.
"""
import os
import sys
import csv
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ============================== CONFIG ======================================
MODEL_FILE = "tsfnet_model_notebook.py"   # the current (patched) model file; its
                                          # defaults reproduce the reference exactly,
                                          # verified bit-for-bit by test_groupnorm.py
DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
CKPT_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\EGTA"
SEED_DIRS = {
    42:   "results_b5e27ee74d_seed42",
    999:  "results_b5e27ee74d_seed999",
    2024: "results_b5e27ee74d_seed2024",
}
SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]
OUT_DIR = os.path.join(CKPT_ROOT, "ensemble_3seed")
BATCH_SIZE = 16          # must match the training runs (run_all_subjects.py)
REPRO_TOL = 1e-3         # max allowed |recomputed - reported| single-seed accuracy
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================================


def _load_model_namespace(path):
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    full = path if os.path.isabs(path) else os.path.join(here, path)
    if not os.path.exists(full):
        full = path  # fall back to cwd-relative
    sys.path.insert(0, os.path.dirname(os.path.abspath(full)))
    ns = {}
    exec(open(full, encoding="utf-8").read(), ns)
    return ns


@torch.no_grad()
def _predict(model, loader, unpack_batch, device):
    """Final weighted output y_pred for every test window, in loader order."""
    model.eval()
    outs, ys = [], []
    for batch in loader:
        x_eeg, x_fnirs, y, x_conn, x_chan = unpack_batch(batch)
        x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
        x_conn = x_conn.to(device) if x_conn is not None else None
        x_chan = x_chan.to(device) if x_chan is not None else None
        y_pred = model(x_eeg, x_fnirs, x_conn, x_chan)[3]
        outs.append(y_pred.float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(outs), np.concatenate(ys)


def main():
    ns = _load_model_namespace(MODEL_FILE)
    TSFNet = ns["TSFNet"]
    NBackDataset = ns["NBackDataset"]
    session_holdout_split = ns["session_holdout_split"]
    adapt_bn = ns["adapt_batchnorm_to_target"]
    unpack_batch = ns["_unpack_batch"]
    cohen_kappa = ns["cohen_kappa"]
    confusion_matrix_np = ns["confusion_matrix_np"]

    os.makedirs(OUT_DIR, exist_ok=True)
    seeds = list(SEED_DIRS)
    quiet = lambda *a, **k: None

    rows = []
    repro_failures = []
    total_cm_ens = np.zeros((3, 3), dtype=int)

    print(f"device={DEVICE}  seeds={seeds}  subjects={len(SUBJECTS)}\n")

    for sid in SUBJECTS:
        subject_dir = os.path.join(DATA_ROOT, sid)
        full = NBackDataset(subject_dir, sid)
        session_ids = full.session_ids if hasattr(full, "session_ids") else None
        if session_ids is None:
            raise RuntimeError("NBackDataset has no session_ids attribute -- cannot rebuild splits")

        per_seed_fold_acc = {s: [] for s in seeds}
        per_seed_fold_kap = {s: [] for s in seeds}
        ens_fold_acc, ens_fold_kap, vote_fold_acc = [], [], []

        for test_session in sorted(np.unique(session_ids)):
            _, test_idx = session_holdout_split(session_ids, test_session)
            loader = DataLoader(Subset(full, test_idx), batch_size=BATCH_SIZE, shuffle=False)

            seed_outputs, y_true = {}, None
            for s in seeds:
                folder = os.path.join(CKPT_ROOT, SEED_DIRS[s])
                ckpt = os.path.join(folder, f"{sid}_session{test_session}_best.pt")
                rjson = os.path.join(folder, f"results_{sid}_session{test_session}.json")

                model = TSFNet(num_classes=3, fnirs_temporal_align="egta").to(DEVICE)
                state = torch.load(ckpt, map_location=DEVICE)
                model.load_state_dict(state, strict=True)       # loud on any mismatch
                adapt_bn(model, loader, DEVICE, log_fn=quiet)   # AdaBN, exactly as in the run
                y_pred, y = _predict(model, loader, unpack_batch, DEVICE)
                if y_true is None:
                    y_true = y
                elif not np.array_equal(y_true, y):
                    raise RuntimeError(f"{sid} s{test_session}: label order differs between "
                                       f"seeds -- the test loader is not deterministic")
                seed_outputs[s] = y_pred

                acc = float((y_pred.argmax(1) == y).mean())
                kap = float(cohen_kappa(y, y_pred.argmax(1)))
                per_seed_fold_acc[s].append(acc)
                per_seed_fold_kap[s].append(kap)

                # ---- reproduction gate ----
                reported = None
                if os.path.exists(rjson):
                    with open(rjson, encoding="utf-8") as f:
                        reported = json.load(f).get("test_accuracy_final")
                if reported is None or abs(acc - reported) > REPRO_TOL:
                    repro_failures.append({"subject": sid, "session": int(test_session),
                                           "seed": s, "recomputed": acc, "reported": reported})

            stack = np.stack([seed_outputs[s] for s in seeds])       # (S, N, 3)
            ens_pred = stack.mean(0).argmax(1)
            votes = stack.argmax(2)                                    # (S, N)
            vote_pred = np.array([np.bincount(votes[:, i], minlength=3).argmax()
                                  for i in range(votes.shape[1])])
            ens_fold_acc.append(float((ens_pred == y_true).mean()))
            ens_fold_kap.append(float(cohen_kappa(y_true, ens_pred)))
            vote_fold_acc.append(float((vote_pred == y_true).mean()))
            total_cm_ens += np.asarray(confusion_matrix_np(y_true, ens_pred, 3), dtype=int)

        row = {"subject_id": sid}
        for s in seeds:
            row[f"acc_seed{s}"] = np.mean(per_seed_fold_acc[s])
            row[f"kappa_seed{s}"] = np.mean(per_seed_fold_kap[s])
        row["acc_seedmean"] = np.mean([row[f"acc_seed{s}"] for s in seeds])
        row["kappa_seedmean"] = np.mean([row[f"kappa_seed{s}"] for s in seeds])
        row["acc_ensemble"] = np.mean(ens_fold_acc)
        row["kappa_ensemble"] = np.mean(ens_fold_kap)
        row["acc_vote"] = np.mean(vote_fold_acc)
        rows.append(row)
        print(f"{sid}: seed-mean acc={row['acc_seedmean']:.4f} k={row['kappa_seedmean']:.4f}  "
              f"|  ensemble acc={row['acc_ensemble']:.4f} k={row['kappa_ensemble']:.4f}  "
              f"|  vote acc={row['acc_vote']:.4f}")

    # ------------------------------------------------------------ write CSV
    csv_path = os.path.join(OUT_DIR, "ensemble_per_subject.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------ summary + stats
    def col(k):
        return np.array([r[k] for r in rows])

    summary = {"n_subjects": len(rows), "seeds": seeds,
               "reproduction_failures": repro_failures,
               "reproduction_ok": len(repro_failures) == 0}
    for k in ["acc_seedmean", "kappa_seedmean", "acc_ensemble", "kappa_ensemble", "acc_vote"] + \
             [f"acc_seed{s}" for s in seeds] + [f"kappa_seed{s}" for s in seeds]:
        summary[f"macro_{k}"] = float(col(k).mean())
    summary["confusion_matrix_ensemble_summed"] = total_cm_ens.tolist()

    try:
        from scipy.stats import wilcoxon
        tests = {}
        for metric in ["acc", "kappa"]:
            ens = col(f"{metric}_ensemble")
            tests[f"{metric}_ensemble_vs_seedmean"] = _wx(wilcoxon, ens, col(f"{metric}_seedmean"))
            for s in seeds:
                tests[f"{metric}_ensemble_vs_seed{s}"] = _wx(wilcoxon, ens, col(f"{metric}_seed{s}"))
        summary["wilcoxon"] = tests
    except ImportError:
        summary["wilcoxon"] = "scipy not installed"

    with open(os.path.join(OUT_DIR, "ensemble_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------ print
    print("\n" + "=" * 72)
    if not summary["reproduction_ok"]:
        print(f"REPRODUCTION GATE FAILED on {len(repro_failures)} fold(s) -- the re-loaded "
              f"single-seed models do NOT reproduce their reported accuracy.")
        for r in repro_failures[:10]:
            print(f"   {r}")
        print("The ensemble numbers below are therefore NOT a valid result. Do not report them.")
        print("=" * 72)
    else:
        print("REPRODUCTION GATE PASSED: every seed/fold reproduces its reported accuracy "
              f"to within {REPRO_TOL}.")
        print("=" * 72)
    for s in seeds:
        print(f"  seed {s:<5}  acc={summary[f'macro_acc_seed{s}']*100:.2f}%  "
              f"kappa={summary[f'macro_kappa_seed{s}']:.4f}")
    print(f"  seed-mean   acc={summary['macro_acc_seedmean']*100:.2f}%  "
          f"kappa={summary['macro_kappa_seedmean']:.4f}   (the currently reported 3-seed figure)")
    print(f"  ENSEMBLE    acc={summary['macro_acc_ensemble']*100:.2f}%  "
          f"kappa={summary['macro_kappa_ensemble']:.4f}   (mean of y_pred, argmax)")
    print(f"  vote        acc={summary['macro_acc_vote']*100:.2f}%")
    if isinstance(summary.get("wilcoxon"), dict):
        w = summary["wilcoxon"]
        print(f"\n  Wilcoxon, ensemble vs seed-mean ({len(rows)} subjects): "
              f"acc p={w['acc_ensemble_vs_seedmean']['p']:.4f}   "
              f"kappa p={w['kappa_ensemble_vs_seedmean']['p']:.4f}")
    print(f"\n  written: {csv_path}")
    print(f"           {os.path.join(OUT_DIR, 'ensemble_summary.json')}")


def _wx(wilcoxon, a, b):
    d = a - b
    if np.allclose(d, 0):
        return {"statistic": None, "p": 1.0, "mean_diff": 0.0, "n_better": 0, "n_worse": 0}
    stat, p = wilcoxon(a, b)
    return {"statistic": float(stat), "p": float(p), "mean_diff": float(d.mean()),
            "n_better": int((d > 0).sum()), "n_worse": int((d < 0).sum())}


if __name__ == "__main__":
    main()
