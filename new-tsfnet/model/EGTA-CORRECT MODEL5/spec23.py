"""
spec23.py -- priority-SPEC23: a dedicated 2-back vs 3-back specialist with its
own model selection, combined hierarchically with an existing 3-class model.

WHY
===
In run60, 64% of block-level errors are 2-back <-> 3-back swaps. One model with
one early stopping on 3-class validation accuracy decides everything, so the
checkpoint chosen is the one that is best on average over all three classes --
not necessarily the best at separating 2 from 3. Every loss-side lever on that
pair (asymmetric cost, class weights, focal, SMOTE, ordinal, CORN) has already
been tried and failed; this changes MODEL SELECTION instead of loss weights.

DESIGN
======
  main model   : an existing 3-class run (default run60 checkpoints, untouched)
  specialist   : the same architecture with num_classes=2, trained ONLY on the
                 2-back/3-back windows of the training sessions, early-stopped
                 on 2-vs-3 validation accuracy
  decision     : the MAIN model decides "0-back or not" (unchanged, by
                 construction); only where it says "not 0-back" does the
                 specialist choose 2 vs 3. Window: argmax(P_main) == 0 -> 0-back,
                 else 3-back if q > 0.5 else 2-back (q = P_spec(3-back)). Block:
                 the same rule on the block-averaged P_main and q.
                 (Taking argmax of H = [P0, (1-P0)(1-q), (1-P0)q] instead would
                 let the specialist overturn 0-back calls whenever P0 < 0.5 --
                 that changes two things at once and was rejected.)

The specialist uses EXACTLY the main model's per-fold split (same
session_holdout_split, same trial-level train/val split) restricted to classes
2 and 3, so both models see the same training sessions and the held-out
session stays held out.

--val-mode trial (default, = main model) validates on 1 trial per class,
i.e. only TWO 2/3 trials -- noisy model selection, which is the very thing this
step is about. --val-mode cross_session validates on a whole training session
(6 trials of 2/3) at the cost of halving training data. Screen both.

ADABN (transductive, as in the accepted baseline): the main model adapts on all
windows of the held-out session (exactly as run60 did). The specialist adapts
on the held-out windows the MAIN model predicts as not-0-back -- the
distribution it was trained on -- selected from model predictions, never from
labels.

USAGE
    # 1) train the specialist (GPU; ~2/3 of a run60 seed, 2-class data only)
    python spec23.py train --seed 42
    # 2) combine with run60's checkpoints and evaluate
    python spec23.py eval  --seed 42 --main-hash 40b7d8ea53 --expected-window-acc 0.6074
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("phase", choices=["train", "eval"])
p.add_argument("--seed", type=int, required=True)
p.add_argument("--data-root", default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback")
p.add_argument("--subjects", default="1-26")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--val-mode", default="trial", choices=["trial", "cross_session"])
p.add_argument("--spec-dir", default=None,
               help="specialist results folder; default <data-root>/results_spec23_<id>_seed<SEED>")
# eval only
p.add_argument("--main-hash", default="40b7d8ea53", help="variant hash of the 3-class run (run60)")
p.add_argument("--main-results-dir", default=None, help="explicit folder of the 3-class checkpoints")
p.add_argument("--expected-window-acc", type=float, default=None,
               help="main model's grand window accuracy on record, for the replay self-check")
# training (defaults = run60)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--weight-decay", type=float, default=1e-4)
p.add_argument("--max-epochs", type=int, default=100)
p.add_argument("--patience", type=int, default=20)
p.add_argument("--lambda-eeg", type=float, default=0.2)
p.add_argument("--lambda-fnirs", type=float, default=0.2)
# architecture (defaults = run60; the main model must use the same)
p.add_argument("--fnirs-temporal-align", default="egta", choices=["none", "egta"])
p.add_argument("--efgf-gate-mode", default="sigmoid", choices=["clamp", "sigmoid"])
p.add_argument("--efgf-gate-init", type=float, default=-2.0)
p.add_argument("--fusion-combination", default="paper_mean", choices=["paper_mean", "wsum", "gated"])
p.add_argument("--fnirs-num-segments", type=int, default=21)
p.add_argument("--cafe-d-model", type=int, default=64)
p.add_argument("--dropout", type=float, default=0.5)
p.add_argument("--cafe-dropout", type=float, default=0.3)
p.add_argument("--conv-dropout", type=float, default=None)
p.add_argument("--conv-dropout-type", default="channel", choices=["channel", "element"])
p.add_argument("--eeg-temporal-mode", default="gap", choices=["gap", "gap_logvar", "bilstm"])
p.add_argument("--fusion-fnirs-inject", default="none", choices=["none", "stage1", "both"])
p.add_argument("--erp-branch", default="none", choices=["none", "eeg", "eeg_fusion"])
ARGS = p.parse_args()

try:
    _this_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _this_dir = os.getcwd()
if "TSFNet" not in globals():
    exec(open(os.path.join(_this_dir, "tsfnet_model_notebook.py"), encoding="utf-8").read(), globals())

ARCH = dict(dropout=ARGS.dropout, cafe_dropout=ARGS.cafe_dropout, cafe_d_model=ARGS.cafe_d_model,
            fusion_combination=ARGS.fusion_combination, eeg_temporal_mode=ARGS.eeg_temporal_mode,
            fnirs_num_segments=ARGS.fnirs_num_segments, fnirs_temporal_align=ARGS.fnirs_temporal_align,
            efgf_gate_mode=ARGS.efgf_gate_mode, efgf_gate_init=ARGS.efgf_gate_init,
            conv_dropout=ARGS.conv_dropout, conv_dropout_type=ARGS.conv_dropout_type,
            fusion_fnirs_inject=ARGS.fusion_fnirs_inject, erp_branch=ARGS.erp_branch)
TRAIN = dict(lr=ARGS.lr, weight_decay=ARGS.weight_decay, max_epochs=ARGS.max_epochs,
             patience=ARGS.patience, lambda_eeg=ARGS.lambda_eeg, lambda_fnirs=ARGS.lambda_fnirs,
             val_mode=ARGS.val_mode, batch_size=ARGS.batch_size)
SPEC_ID = hashlib.md5(json.dumps({"arch": ARCH, "train": TRAIN}, sort_keys=True).encode()).hexdigest()[:8]
SPEC_DIR = ARGS.spec_dir or os.path.join(ARGS.data_root, f"results_spec23_{SPEC_ID}_seed{ARGS.seed}")
LOAD_ERP = ARGS.erp_branch != "none"


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


class Remap23(Dataset):
    """Windows of classes 1 (2-back) / 2 (3-back) relabelled 0 / 1 for the binary model."""

    def __init__(self, base, indices):
        self.base, self.indices = base, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        b = list(self.base[int(self.indices[i])])
        if int(b[2]) not in (1, 2):
            raise ValueError("Remap23 received a 0-back window")
        b[2] = b[2] - 1
        return tuple(b)


def fold_split(full, test_session):
    """The main model's split, verbatim (run_subject_cross_session)."""
    session_ids, trial_ids, labels = full.session_ids, full.trial_ids, full.labels.numpy()
    train_val_idx, test_idx = session_holdout_split(session_ids, test_session)
    if ARGS.val_mode == "trial":
        tr_sub, va_sub = trial_level_train_val_split(trial_ids[train_val_idx], labels[train_val_idx])
    else:
        val_session = int(np.unique(session_ids[train_val_idx]).min())
        tr_sub, va_sub = cross_session_train_val_split(session_ids[train_val_idx], val_session)
    return train_val_idx[tr_sub], train_val_idx[va_sub], test_idx


# ---------------------------------------------------------------------------
def train():
    random.seed(ARGS.seed)
    np.random.seed(ARGS.seed)
    torch.manual_seed(ARGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(ARGS.seed)
    os.makedirs(SPEC_DIR, exist_ok=True)
    man_path = os.path.join(SPEC_DIR, "spec23_manifest.json")
    manifest = {"id": SPEC_ID, "seed": ARGS.seed, "arch": ARCH, "train": TRAIN, "folds": {}}
    if os.path.isfile(man_path):
        with open(man_path, encoding="utf-8") as f:
            manifest["folds"] = json.load(f).get("folds", {})
    device = torch.device(ARGS.device)
    print(f"SPEC23 TRAIN | id={SPEC_ID} seed={ARGS.seed} val_mode={ARGS.val_mode} -> {SPEC_DIR}")

    for sid in parse_subjects(ARGS.subjects):
        sdir = os.path.join(ARGS.data_root, sid)
        if not os.path.isdir(sdir):
            continue
        full = NBackDataset(sdir, sid, load_erp=LOAD_ERP)
        labels = full.labels.numpy()
        for test_session in sorted(np.unique(full.session_ids)):
            key = f"{sid}_s{int(test_session)}"
            ckpt = os.path.join(SPEC_DIR, f"{sid}_session{int(test_session)}_spec23.pt")
            if key in manifest["folds"] and os.path.isfile(ckpt):
                continue
            tr, va, te = fold_split(full, test_session)
            tr23, va23 = tr[labels[tr] >= 1], va[labels[va] >= 1]
            # leakage guards: nothing from the held-out session, nothing 0-back
            assert not np.isin(full.session_ids[np.r_[tr23, va23]], [test_session]).any()
            assert set(np.unique(labels[np.r_[tr23, va23]])) <= {1, 2}
            if len(np.unique(labels[va23])) < 2:
                print(f"[{key}] SKIP -- validation lacks one of the two classes")
                continue
            tl = DataLoader(Remap23(full, tr23), batch_size=ARGS.batch_size, shuffle=True)
            vl = DataLoader(Remap23(full, va23), batch_size=ARGS.batch_size, shuffle=False)
            model = TSFNet(num_classes=2, **ARCH).to(device)
            t0 = time.time()
            val_acc, val_kappa, _m, _r = train_one_fold(
                model, tl, vl, vl, device,
                lambda_eeg=ARGS.lambda_eeg, lambda_fnirs=ARGS.lambda_fnirs,
                lr=ARGS.lr, weight_decay=ARGS.weight_decay,
                stage1_max_epochs=ARGS.max_epochs, stage1_patience=ARGS.patience,
                stage2_max_epochs=0, stage2_stop_mode="none", ckpt_path=ckpt, adabn=False,
                normalize_ypred=True, no_decay_scalars=True, log_fn=lambda *a, **k: None)
            manifest["folds"][key] = {"val_2v3_acc": float(val_acc), "n_train": int(len(tr23)),
                                      "n_val": int(len(va23)), "minutes": round((time.time() - t0) / 60, 2)}
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            print(f"[{key}] specialist trained on {len(tr23)} windows | val 2v3 acc {val_acc:.3f} "
                  f"({len(va23)} windows) | {manifest['folds'][key]['minutes']} min")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    print(f"\nDONE. Evaluate with:\n  python spec23.py eval --seed {ARGS.seed} "
          f"--spec-dir \"{SPEC_DIR}\" --main-hash {ARGS.main_hash}")


# ---------------------------------------------------------------------------
def posteriors(model, ds, device):
    model.eval()
    out = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=ARGS.batch_size, shuffle=False):
            x_eeg, x_fnirs, _y, x_conn, x_chan = _unpack_batch(batch)
            x_eeg, x_fnirs = x_eeg.to(device), x_fnirs.to(device)
            x_conn = x_conn.to(device) if x_conn is not None else None
            x_chan = x_chan.to(device) if x_chan is not None else None
            yp = model(x_eeg, x_fnirs, x_conn, x_chan)[3]
            out.append((yp / yp.sum(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def find_dir(name):
    for root, dirs, _ in os.walk(ARGS.data_root):
        if name in dirs:
            return os.path.join(root, name)
        if root[len(ARGS.data_root):].count(os.sep) >= 3:
            dirs[:] = []
    return None


def run_eval():   # not "evaluate": that name belongs to the notebook (train_one_fold calls it)
    from scipy.stats import wilcoxon
    device = torch.device(ARGS.device)
    main_dir = ARGS.main_results_dir or find_dir(f"results_{ARGS.main_hash}_seed{ARGS.seed}")
    if not main_dir or not os.path.isdir(main_dir):
        sys.exit(f"ERROR: main results folder not found; pass --main-results-dir")
    if not os.path.isdir(SPEC_DIR):
        sys.exit(f"ERROR: specialist folder not found: {SPEC_DIR} (run the train phase or pass --spec-dir)")
    print(f"SPEC23 EVAL | main: {main_dir}\n            spec: {SPEC_DIR}")

    rows, pooled = [], {"main": [], "hier": [], "true": []}
    for sid in parse_subjects(ARGS.subjects):
        sdir = os.path.join(ARGS.data_root, sid)
        if not os.path.isdir(sdir):
            continue
        full = NBackDataset(sdir, sid, load_erp=LOAD_ERP)
        labels, trial_ids = full.labels.numpy(), full.trial_ids
        for ts in sorted(np.unique(full.session_ids)):
            mck = os.path.join(main_dir, f"{sid}_session{int(ts)}_best.pt")
            sck = os.path.join(SPEC_DIR, f"{sid}_session{int(ts)}_spec23.pt")
            if not (os.path.isfile(mck) and os.path.isfile(sck)):
                print(f"[{sid} s{int(ts)}] SKIP -- checkpoint missing")
                continue
            _tr, _va, te = fold_split(full, ts)
            test_ds = Subset(full, te)
            main = TSFNet(num_classes=3, **ARCH).to(device)
            main.load_state_dict(torch.load(mck, map_location=device), strict=True)
            adapt_batchnorm_to_target(main, DataLoader(test_ds, batch_size=ARGS.batch_size),
                                      device, log_fn=lambda *a, **k: None)
            P = posteriors(main, test_ds, device)                         # (n, 3)
            non0 = P.argmax(1) != 0
            spec = TSFNet(num_classes=2, **ARCH).to(device)
            spec.load_state_dict(torch.load(sck, map_location=device), strict=True)
            adapt_on = Subset(full, te[non0]) if non0.any() else test_ds   # predictions, not labels
            adapt_batchnorm_to_target(spec, DataLoader(adapt_on, batch_size=ARGS.batch_size),
                                      device, log_fn=lambda *a, **k: None)
            q = posteriors(spec, test_ds, device)[:, 1]                   # P(3-back | 2 or 3)
            y, tid = labels[te], trial_ids[te]
            win_main = P.argmax(1)
            win_hier = np.where(win_main == 0, 0, np.where(q > 0.5, 2, 1))
            blocks = np.unique(tid)
            yt = np.array([y[tid == t][0] for t in blocks])
            pm = np.array([P[tid == t].mean(0).argmax() for t in blocks])
            qb = np.array([q[tid == t].mean() for t in blocks])
            ph = np.where(pm == 0, 0, np.where(qb > 0.5, 2, 1))
            assert np.array_equal(pm == 0, ph == 0) and np.array_equal(win_main == 0, win_hier == 0)
            m23 = np.isin(yt, [1, 2]) & (pm != 0)                         # 2/3 blocks main kept
            row = {"subject": sid, "test_session": int(ts),
                   "main_win_acc": float((P.argmax(1) == y).mean()),
                   "hier_win_acc": float((win_hier == y).mean()),
                   "main_trial_acc": float((pm == yt).mean()), "hier_trial_acc": float((ph == yt).mean()),
                   "n_23_blocks_kept": int(m23.sum()),
                   "main_23_correct": int((pm[m23] == yt[m23]).sum()),
                   "hier_23_correct": int((ph[m23] == yt[m23]).sum())}
            rows.append(row)
            pooled["main"].append(pm); pooled["hier"].append(ph); pooled["true"].append(yt)
            print(f"[{sid} s{int(ts)}] window {row['main_win_acc']:.3f} -> {row['hier_win_acc']:.3f} | "
                  f"trial {row['main_trial_acc']:.3f} -> {row['hier_trial_acc']:.3f} | 2v3 on kept 2/3 "
                  f"blocks {row['main_23_correct']}/{row['n_23_blocks_kept']} -> "
                  f"{row['hier_23_correct']}/{row['n_23_blocks_kept']}")
            del main, spec
    if not rows:
        sys.exit("ERROR: nothing evaluated")

    subj = sorted({r["subject"] for r in rows})
    agg = lambda k: np.array([np.mean([r[k] for r in rows if r["subject"] == s]) for s in subj])
    out = {"main_dir": main_dir, "spec_dir": SPEC_DIR, "n_subjects": len(subj)}
    print("\n" + "=" * 80)
    for lvl in ("win", "trial"):
        a, b = agg(f"main_{lvl}_acc"), agg(f"hier_{lvl}_acc")
        d = b - a
        pv = wilcoxon(a, b).pvalue if (d != 0).sum() >= 6 else float("nan")
        out[lvl] = {"main": float(a.mean()), "hier": float(b.mean()), "diff": float(d.mean()),
                    "improved": int((d > 0).sum()), "worsened": int((d < 0).sum()), "p": pv}
        print(f"{lvl:5s}-level acc: main {a.mean():.4f} -> hierarchical {b.mean():.4f} "
              f"({100 * d.mean():+.2f} pp) | better in {(d > 0).sum()}/{len(subj)} | Wilcoxon p={pv:.3g}")
    k = sum(r["n_23_blocks_kept"] for r in rows)
    mc = sum(r["main_23_correct"] for r in rows); hc = sum(r["hier_23_correct"] for r in rows)
    out["2v3_on_kept_blocks"] = {"n": k, "main": mc / max(k, 1), "hier": hc / max(k, 1)}
    print(f"2-vs-3 on 2/3 blocks the main model did not call 0-back: main {mc}/{k} = {mc / max(k, 1):.3f} "
          f"-> specialist {hc}/{k} = {hc / max(k, 1):.3f}")
    ym, yh, yt = (np.concatenate(pooled[x]) for x in ("main", "hier", "true"))
    out["trial_confusion_main"] = confusion_matrix_np(yt, ym, 3).tolist()
    out["trial_confusion_hier"] = confusion_matrix_np(yt, yh, 3).tolist()
    print("trial confusion main:", out["trial_confusion_main"])
    print("trial confusion hier:", out["trial_confusion_hier"])
    if ARGS.expected_window_acc is not None:
        rep = float(agg("main_win_acc").mean())
        ok = abs(rep - ARGS.expected_window_acc) <= 0.002
        out["self_check"] = {"reproduced": rep, "expected": ARGS.expected_window_acc, "passed": ok}
        print(f"SELF-CHECK main model replay: {rep:.4f} vs {ARGS.expected_window_acc:.4f} -> "
              f"{'PASS' if ok else 'FAIL (results not comparable to the record)'}")
    with open(os.path.join(SPEC_DIR, f"spec23_eval_seed{ARGS.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(SPEC_DIR, f"spec23_eval_seed{ARGS.seed}.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"[written] {SPEC_DIR}")


if __name__ == "__main__":
    train() if ARGS.phase == "train" else run_eval()
