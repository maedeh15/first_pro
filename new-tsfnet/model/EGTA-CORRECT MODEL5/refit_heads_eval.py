"""
refit_heads_eval.py -- replace the trained EEG head with one fitted on CLEAN features, and measure
what that does to the branch AND to the final decision. No network is trained.

Why. probe_eeg_trunk.py established that a logistic regression fitted on exactly the features
clf_eeg is given beats clf_eeg itself, on held-out data and on the training rows alike:

    head              test 0.4958   train 0.6861
    fresh LR, C=0.001 test 0.5285   train 0.8019     26/26 subjects on the train row, p=3e-8
    without AdaBN     the advantage is larger, not smaller (+0.0548), and head_train stays 0.6736

so the head is UNDER-fitting its own features and AdaBN is not the cause. The mechanism this script
acts on is visible in the architecture: clf_eeg is an MLP (Linear 512->128, ELU, Dropout(0.5),
Linear 128->3) whose input comes from a ConvBlock3D ending in Dropout3d(0.5). During training it
therefore sees doubly dropout-corrupted features, and at evaluation both dropouts are off. A
classifier fitted directly on the clean evaluation-time features has no such handicap.

What it does, per fold (subject i, held-out session s):
  1. load the fold's checkpoint, apply AdaBN on the held-out session exactly as ensemble_eval does
  2. in eval mode (both dropouts off), extract the features clf_eeg receives for ALL windows
  3. fit StandardScaler + LogisticRegression on the TWO TRAINING sessions' features and labels
  4. write that fit into the model as a drop-in replacement for clf_eeg, exactly
  5. run the model again on the held-out session and read every head

Step 4 is what makes this more than another probe: the decision combination, the learned fusion
weights, --normalize-ypred and everything else downstream are computed BY THE MODEL, so y_pred after
the swap is the real thing, not a reconstruction.

LEAKAGE. Only the two training sessions' labels are used, which the protocol already allows. The
held-out session's unlabeled windows enter through AdaBN alone, the transductive step already
disclosed for E3/AdaBN. Run with --no-adabn to drop even that.

WHAT IT CANNOT SHOW. The trunk is the one trained under the joint objective; this does not say what
a retrained branch would reach. And a better y_eeg need not mean a better y_pred -- the
--lambda-eeg 1.0 run left the head unchanged and cost 1.94 points of overall accuracy, so the
decision combination is not a monotone function of its parts. That is exactly why this script
reports y_pred and not just y_eeg.

usage
  python refit_heads_eval.py --self-test
  python refit_heads_eval.py --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback" \
        --results-dir "E:\\...\\results_40b7d8ea53_seed42" [--device cuda] [--C 0.001] [--out refit.csv]
"""
import os
import csv
import sys
import argparse
import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HEADS = ("y_pred", "y_eeg", "y_fusion", "y_fnirs")


def fit_linear_head(Xtr, ytr, C, num_classes=3):
    """Fit StandardScaler + LogisticRegression and fold them into ONE affine map on the raw
    features, so the result can stand in for clf_eeg with no scaler at inference:

        z = (x - mu) / sigma ;  logits = W z + b  =  (W/sigma) x + (b - W mu/sigma)

    Returns (A, c) with logits = x @ A.T + c, and the classes the rows belong to. A class missing
    from the training rows gets a -inf row so softmax assigns it probability 0 rather than silently
    shifting the column order."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(Xtr, ytr)
    sc, lr = clf.named_steps["standardscaler"], clf.named_steps["logisticregression"]
    mu, sd = sc.mean_, np.sqrt(sc.var_) + 1e-12
    W, b = lr.coef_ / sd[None, :], lr.intercept_ - (lr.coef_ * (mu / sd)[None, :]).sum(1)
    if len(lr.classes_) == 2:          # sklearn gives one row for two classes
        W, b = np.vstack([-W[0], W[0]]), np.array([-b[0], b[0]])
    A = np.full((num_classes, Xtr.shape[1]), 0.0)
    c = np.full(num_classes, -1e9)
    for i, k in enumerate(lr.classes_.astype(int)):
        A[k], c[k] = W[i], b[i]
    return A, c, clf


def make_head_module(A, c):
    import torch
    import torch.nn as nn

    class FittedHead(nn.Module):
        """stand-in for BranchClassifier: softmax of an affine map, so the model's own decision
        combination is applied to it unchanged"""

        def __init__(self, A, c):
            super().__init__()
            self.register_buffer("A", torch.as_tensor(A, dtype=torch.float32))
            self.register_buffer("c", torch.as_tensor(c, dtype=torch.float32))

        def forward(self, x):
            return torch.softmax(x @ self.A.t() + self.c, dim=-1)

    return FittedHead(A, c)


def self_test():
    import torch
    ok = True
    rng = np.random.RandomState(0)
    n, d = 400, 30
    y = rng.randint(0, 3, n)
    X = rng.randn(n, d) * 2.5 + 7.0
    X[np.arange(n), y % d] += 2.0

    # 1. the folded affine map must reproduce sklearn's own probabilities
    A, c, clf = fit_linear_head(X, y, C=0.1)
    head = make_head_module(A, c)
    with torch.no_grad():
        mine = head(torch.as_tensor(X, dtype=torch.float32)).numpy()
    ref = clf.predict_proba(X)
    dmax = np.abs(mine - ref).max()
    good = dmax < 2e-5
    print(f"[affine] folded scaler+LR reproduces predict_proba, max diff {dmax:.2e} -> "
          f"{'OK' if good else 'FAIL'}")
    ok &= good

    # 2. a class absent from training must get probability ~0, not a shifted column
    m = y != 1
    A2, c2, _ = fit_linear_head(X[m], y[m], C=0.1)
    with torch.no_grad():
        p2 = make_head_module(A2, c2)(torch.as_tensor(X, dtype=torch.float32)).numpy()
    good = p2[:, 1].max() < 1e-6 and abs(p2.sum(1).mean() - 1.0) < 1e-5
    print(f"[missing class] absent class gets p=0 and rows still sum to 1 -> "
          f"{'OK' if good else 'FAIL'}")
    ok &= good

    # 3. swapping a head must not touch the others. A stand-in model with three heads: replace one,
    #    check the other two are bit-identical.
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.clf_eeg = nn.Linear(d, 3)
            self.clf_other = nn.Linear(d, 3)

        def forward(self, x):
            return torch.softmax(self.clf_eeg(x), -1), torch.softmax(self.clf_other(x), -1)

    t = Tiny()
    xb = torch.randn(7, d)
    with torch.no_grad():
        e0, o0 = t(xb)
    t.clf_eeg = make_head_module(A, c)
    with torch.no_grad():
        e1, o1 = t(xb)
    good = torch.allclose(o0, o1) and not torch.allclose(e0, e1)
    print(f"[swap] replacing one head changes only that head -> {'OK' if good else 'FAIL'}")
    ok &= good

    # 4. the fit must use only the rows it is given: refitting on a subset that excludes a block of
    #    rows must not depend on those rows' labels
    keep = np.arange(n) < 300
    A3, c3, _ = fit_linear_head(X[keep], y[keep], C=0.1)
    y_tampered = y.copy()
    y_tampered[~keep] = (y_tampered[~keep] + 1) % 3
    A4, c4, _ = fit_linear_head(X[keep], y_tampered[keep], C=0.1)
    good = np.allclose(A3, A4) and np.allclose(c3, c4)
    print(f"[no leak] changing held-out labels leaves the fit identical -> {'OK' if good else 'FAIL'}")
    ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root")
    ap.add_argument("--results-dir", help="folder holding VPxxx_session{k}_best.pt for ONE seed")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--adabn", dest="adabn", action="store_true", default=True)
    ap.add_argument("--no-adabn", dest="adabn", action="store_false")
    ap.add_argument("--C", type=float, default=0.001,
                    help="L2 of the refitted head. 0.001 is the strongest value in the probe's sweep "
                         "and the one with the best train/test balance there; the sweep showed test "
                         "accuracy varies by under 1.2 points across five orders of magnitude.")
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--cafe-dropout", type=float, default=0.3)
    ap.add_argument("--cafe-d-model", type=int, default=64)
    ap.add_argument("--fnirs-num-segments", type=int, default=21)
    ap.add_argument("--fnirs-temporal-align", default="egta")
    ap.add_argument("--efgf-gate-mode", default="sigmoid")
    ap.add_argument("--efgf-gate-init", type=float, default=-2.0)
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not (a.data_root and a.results_dir):
        ap.error("--data-root and --results-dir are required unless --self-test")

    import torch
    from torch.utils.data import DataLoader, Subset
    here = os.path.dirname(os.path.abspath(__file__))
    nb = os.path.join(here, "tsfnet_model_notebook.py")
    if not os.path.isfile(nb):
        sys.exit(f"ERROR: tsfnet_model_notebook.py not found next to this script ({nb})")
    print(f"[init] loading definitions from {nb}")
    exec(open(nb, encoding="utf-8").read(), globals())

    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device = {device}, adabn = {a.adabn}, C = {a.C}")
    rows = []
    for sid in a.subjects:
        sub_dir = os.path.join(a.data_root, sid)
        if not os.path.isdir(sub_dir):
            continue
        try:
            full = NBackDataset(sub_dir, sid)                                     # noqa: F821
        except Exception as exc:
            print(f"[{sid}] SKIP -- {exc}")
            continue
        labels = full.labels.numpy()
        ses_all = np.asarray(full.session_ids)
        for test_s in sorted(np.unique(ses_all)):
            ckpt = os.path.join(a.results_dir, f"{sid}_session{test_s}_best.pt")
            if not os.path.isfile(ckpt):
                print(f"[{sid} s{test_s}] SKIP -- no checkpoint")
                continue
            model = TSFNet(num_classes=3, dropout=a.dropout, cafe_dropout=a.cafe_dropout,  # noqa: F821
                           cafe_d_model=a.cafe_d_model, fnirs_num_segments=a.fnirs_num_segments,
                           fnirs_temporal_align=a.fnirs_temporal_align,
                           efgf_gate_mode=a.efgf_gate_mode,
                           efgf_gate_init=a.efgf_gate_init).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
            model.eval()
            te_idx = np.where(ses_all == test_s)[0]
            tr_idx = np.where(ses_all != test_s)[0]
            te_loader = DataLoader(Subset(full, te_idx), batch_size=a.batch_size, shuffle=False)
            if a.adabn:
                adapt_batchnorm_to_target(model, te_loader, device,                # noqa: F821
                                          log_fn=lambda *x, **k: None)

            # the features clf_eeg receives: the branch's pooled map, flattened. Taken in eval mode,
            # so both the Dropout3d inside ConvBlock3D and the Dropout inside BranchClassifier are off.
            grabbed = {}
            hook = model.eeg_branch.register_forward_hook(
                lambda m, i, o: grabbed.__setitem__("pooled", o[2].detach()))

            def run(idx):
                out = {h: [] for h in HEADS}
                feats = []
                loader = DataLoader(Subset(full, idx.tolist()), batch_size=a.batch_size, shuffle=False)
                with torch.no_grad():
                    for batch in loader:
                        x_eeg, x_fnirs = batch[0].to(device), batch[1].to(device)
                        y_eeg, y_fus, y_fn, y_pred, _aux = model(x_eeg, x_fnirs, None, None)
                        feats.append(grabbed["pooled"].reshape(grabbed["pooled"].shape[0], -1)
                                     .cpu().numpy())
                        for h, v in zip(HEADS, (y_pred, y_eeg, y_fus, y_fn)):
                            out[h].append(v.cpu().numpy())
                return {h: np.concatenate(v) for h, v in out.items()}, np.concatenate(feats)

            before_tr, Xtr = run(tr_idx)
            before_te, _Xte = run(te_idx)
            A, c, _ = fit_linear_head(Xtr, labels[tr_idx], C=a.C)
            orig_head = model.clf_eeg
            model.clf_eeg = make_head_module(A, c).to(device)
            after_te, _ = run(te_idx)
            after_tr, _ = run(tr_idx)
            model.clf_eeg = orig_head
            hook.remove()
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            acc = lambda P, idx: float((P.argmax(1) == labels[idx]).mean())
            row = dict(subject=sid, session=int(test_s), n_test=len(te_idx))
            for h in HEADS:
                row[f"{h}_before"] = acc(before_te[h], te_idx)
                row[f"{h}_after"] = acc(after_te[h], te_idx)
            row["y_eeg_before_train"] = acc(before_tr["y_eeg"], tr_idx)
            row["y_eeg_after_train"] = acc(after_tr["y_eeg"], tr_idx)
            rows.append(row)
            print(f"[{sid} s{test_s}] y_eeg {row['y_eeg_before']:.3f} -> {row['y_eeg_after']:.3f} | "
                  f"y_pred {row['y_pred_before']:.3f} -> {row['y_pred_after']:.3f}", flush=True)
    if not rows:
        raise SystemExit("no fold could be evaluated")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("per-fold rows ->", a.out)

    subs = sorted({r["subject"] for r in rows})
    per = lambda k: np.array([np.mean([r[k] for r in rows if r["subject"] == s]) for s in subs])
    print(f"\n{len(subs)} subjects, {len(rows)} folds. mean over subjects, before -> after the swap:")
    print(f"  {'head':10} {'before':>8} {'after':>8} {'diff':>9} {'wins':>7} {'p':>8}")
    for h in HEADS + ("y_eeg_before_train",):
        if h == "y_eeg_before_train":
            b, A_ = per("y_eeg_before_train"), per("y_eeg_after_train")
            name = "y_eeg TRAIN"
        else:
            b, A_ = per(f"{h}_before"), per(f"{h}_after")
            name = h
        d = A_ - b
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  {name:10} {b.mean():8.4f} {A_.mean():8.4f} {d.mean():+9.4f} "
              f"{int((d > 0).sum()):3d}/{len(d):<3} {p:8.4f}")
    print("\ny_eeg is the head that was replaced; y_fusion and y_fnirs must be unchanged (they share "
          "no parameters with clf_eeg) and serve as a control. y_pred is the question that matters: "
          "the decision combination is not a monotone function of its parts, so a better y_eeg can "
          "leave it flat or hurt it -- the --lambda-eeg 1.0 run cost 1.94 points of overall accuracy "
          "without moving the head at all.")


if __name__ == "__main__":
    main()
