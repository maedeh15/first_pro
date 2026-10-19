"""
leak_test_mstfdn.py -- controlled test of the "test-set early stopping"
hypothesis for the MSTFDN LOSO gap (paper 69.02% vs honest re-implementation
38.34%).

The paper (Sec 3.3.1/3.3.2) uses early stopping with max 1000 epochs but
defines only two training sessions and one test session -- no validation
set. This script reproduces that LEAKY protocol on purpose, with everything
else identical to MSTFDN_train_LOSO_3class.py (same data, model, loss,
optimizer, scheduler, seed, patience):

  * train on ALL windows of the two training sessions (594, no val split)
  * evaluate the TEST session after every epoch
  * early-stop on TEST loss (patience = cfg.patience) and report:
      leak_es_acc   -- test acc at the epoch with the lowest TEST loss
                       (= "early stopping on the test set")
      leak_max_acc  -- the highest test acc seen in any epoch
                       (= "pick the best epoch on the test set")
      last_acc      -- test acc at the last epoch trained
  * the per-epoch test-accuracy curve is saved for plotting

If --honest-csv points at results_per_subject.csv from the normal (honest)
run, the honest per-fold accuracy is put next to the leaky numbers.

THIS IS A DIAGNOSTIC, NOT A RESULT: the leaky numbers are optimistic by
construction and must never be reported as model performance.

usage (one subject first, ~1 h on an RTX 3070):
  python leak_test_mstfdn.py --subjects VP001 ^
      --honest-csv E:\\...\\MSTFDN_LOSO_V3_3CLASS\\seed42_valtrial\\results_per_subject.csv
"""
import os
import sys
import json
import time
import random
import argparse
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import MSTFDN_train_LOSO_3class as T
from MSTFDN_model_v2 import MSTFDN, MSTFDNLoss

LOG_FILE = None


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    if LOG_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def leaky_fold(cfg, dataset, train_idx, test_idx, device):
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=cfg.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=cfg.batch_size,
                             shuffle=False, num_workers=0, pin_memory=True)
    model = MSTFDN(cfg).to(device)
    loss_fn = MSTFDNLoss(cfg, model)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.max_epochs - cfg.warmup_epochs,
                              eta_min=cfg.lr * cfg.eta_min_ratio)

    curve = []                                # (epoch, test_loss, test_acc, train_acc)
    best_loss, best_epoch, best_acc_at_loss, patience = float("inf"), 0, 0.0, 0
    for epoch in range(cfg.max_epochs):
        T.warmup_lr(opt, epoch, cfg)
        tr = T.train_one_epoch(model, train_loader, opt, loss_fn, device)
        te = T.evaluate(model, test_loader, loss_fn, device)   # <-- TEST used every epoch
        if epoch >= cfg.warmup_epochs:
            sched.step()
        curve.append((epoch + 1, te["loss"], te["acc"], tr["acc"]))
        log(f"[leak] epoch {epoch + 1} | train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} "
            f"| TEST_loss={te['loss']:.4f} TEST_acc={te['acc']:.4f}")
        if te["loss"] < best_loss:
            best_loss, best_epoch, best_acc_at_loss, patience = te["loss"], epoch + 1, te["acc"], 0
        else:
            patience += 1
            if patience >= cfg.patience:
                log(f"[leak] early stop on TEST loss at epoch {epoch + 1}")
                break

    accs = [c[2] for c in curve]
    i_max = int(np.argmax(accs))
    out = {
        "leak_es_acc": float(best_acc_at_loss), "leak_es_epoch": best_epoch,
        "leak_max_acc": float(accs[i_max]), "leak_max_epoch": curve[i_max][0],
        "last_acc": float(accs[-1]), "stopped_epoch": curve[-1][0],
        "curve": curve,
    }
    del model, train_loader, test_loader
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    global LOG_FILE
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=T.PREPROCESSED_PATH)
    ap.add_argument("--out", default=os.path.join(T.OUTPUT_PATH, "LEAK_TEST"))
    ap.add_argument("--subjects", nargs="+", default=["VP001"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--honest-csv", default=None,
                    help="results_per_subject.csv of the honest run, for side-by-side numbers")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = os.path.join(a.out, f"seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    LOG_FILE = os.path.join(out_dir, "leak_test_log.txt")
    T.PREPROCESSED_PATH = a.data
    cfg = T.get_config()

    honest = None
    if a.honest_csv and os.path.exists(a.honest_csv):
        honest = pd.read_csv(a.honest_csv)

    log("=" * 70)
    log("MSTFDN LEAK TEST -- early stopping / epoch selection on the TEST session")
    log("DIAGNOSTIC ONLY: these numbers are optimistic by construction")
    log(f"device={device} seed={a.seed} data={a.data} patience={cfg.patience} "
        f"max_epochs={cfg.max_epochs}")
    log("=" * 70)

    rows = []
    for sid in a.subjects:
        ds = T.MSTFDNSubjectDataset(a.data, sid)
        ses = ds.session_ids
        for fold, test_s in enumerate((1, 2, 3), 1):
            # resume: skip folds already done
            fpath = os.path.join(out_dir, f"leak_{sid}_session{test_s}.json")
            if os.path.exists(fpath):
                with open(fpath, encoding="utf-8") as f:
                    r = json.load(f)
                log(f"[resume] {sid} session {test_s} already done")
            else:
                log(f"=== {sid} | test session {test_s} (leaky) ===")
                tr_idx, te_idx = np.where(ses != test_s)[0], np.where(ses == test_s)[0]
                log(f"[split] train={len(tr_idx)} (ALL windows of both training sessions, "
                    f"no validation) test={len(te_idx)}")
                t0 = time.time()
                r = leaky_fold(cfg, ds, tr_idx, te_idx, device)
                r.update({"subject": sid, "fold": fold, "test_session": test_s,
                          "n_train": len(tr_idx), "n_test": len(te_idx),
                          "time_min": round((time.time() - t0) / 60, 2)})
                with open(fpath + ".tmp", "w", encoding="utf-8") as f:
                    json.dump(r, f, indent=2)
                os.replace(fpath + ".tmp", fpath)
            h = np.nan
            if honest is not None:
                m = honest[(honest["subject"] == sid) & (honest["test_session"] == test_s)]
                if len(m):
                    h = float(m["accuracy"].iloc[0]) / 100.0
            log(f"-> {sid} S{test_s}: honest={h:.4f} | leak_es={r['leak_es_acc']:.4f} "
                f"(ep {r['leak_es_epoch']}) | leak_max={r['leak_max_acc']:.4f} "
                f"(ep {r['leak_max_epoch']}) | last={r['last_acc']:.4f} (ep {r['stopped_epoch']})")
            rows.append({k: v for k, v in r.items() if k != "curve"} | {"honest_acc": h})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "leak_test_results.csv"), index=False)
    log("=" * 70)
    log(f"LEAK TEST SUMMARY ({df['subject'].nunique()} subject(s), {len(df)} folds)")
    for col, name in (("honest_acc", "honest (val from training sessions)"),
                      ("leak_es_acc", "early stopping on TEST loss"),
                      ("leak_max_acc", "best epoch picked on TEST acc"),
                      ("last_acc", "last epoch")):
        per_sub = df.groupby("subject")[col].mean()
        log(f"  {name:38s}: {per_sub.mean():.4f} (+-{per_sub.std(ddof=0):.4f})")
    log("  paper (Table 2, LOSO 3-class)        : 0.6902")

    # The quantity the test exists to measure: what epoch selection on the
    # test session BUYS over the honest protocol, fold by fold.
    # Caveat printed with it: the two runs are not paired on training data --
    # the leaky run trains on all 594 windows, the honest one on 495 after the
    # validation split, and the RNG stream differs. The extra training data
    # favours the leaky run, so a gain at or below zero is the stronger result.
    g = df.dropna(subset=["honest_acc"])
    if len(g):
        log(f"  leak gain over honest, per fold (n={len(g)}):")
        for col, name in (("leak_es_acc", "early stopping on TEST loss"),
                          ("leak_max_acc", "best epoch picked on TEST acc")):
            d = (g[col] - g["honest_acc"]).to_numpy()
            # honest_acc comes from a CSV rounded to 0.01 percentage points,
            # so anything under 1e-3 is a tie, not an improvement.
            log(f"    {name:34s}: {d.mean():+.4f} (+-{d.std(ddof=0):.4f}) "
                f"| improved in {(d > 1e-3).sum()}/{len(d)} folds "
                f"| worst {d.min():+.4f} best {d.max():+.4f}")
        log("    (not paired: leaky trains on 594 windows, honest on 495; "
            "the extra data favours the leaky run)")
    log(f"saved: {os.path.join(out_dir, 'leak_test_results.csv')}")

    # test-accuracy curves, one panel per subject
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for sid in df["subject"].unique():
            fig, ax = plt.subplots(figsize=(8, 4))
            for s in (1, 2, 3):
                p = os.path.join(out_dir, f"leak_{sid}_session{s}.json")
                if not os.path.exists(p):
                    continue
                c = np.array(json.load(open(p, encoding="utf-8"))["curve"])
                ax.plot(c[:, 0], c[:, 2], label=f"test session {s}")
            ax.axhline(1 / 3, ls="--", color="gray", label="chance")
            ax.axhline(0.6902, ls=":", color="black", label="paper 69.02%")
            ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_ylim(0, 1)
            ax.set_title(f"{sid} -- test accuracy per epoch (leaky protocol)")
            ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"leak_curve_{sid}.png"), dpi=130)
            plt.close(fig)
    except Exception as exc:
        log(f"[plot] skipped: {exc}")


if __name__ == "__main__":
    main()
