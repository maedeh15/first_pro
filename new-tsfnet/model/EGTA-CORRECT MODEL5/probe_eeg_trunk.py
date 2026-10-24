"""
probe_eeg_trunk.py (v2) -- the last untested cause of the EEG branch's 0.487.

v2 ADDS two diagnostics, and changes nothing in the existing columns (the default run prints the same
numbers as v1):
  * head_train      accuracy of the deployed head on the TRAINING windows of the fold. The point is the
                    comparison with the fresh classifier's own train accuracy, not the size of the
                    train/test gap on its own: a head that trails the fresh classifier on held-out data
                    BECAUSE it over-fits must beat it on the training windows. MEASURED: it does not. The
                    head reaches 0.6861 on its training windows while the most heavily regularised fresh
                    classifier reaches 0.8019 on the same rows, in 26/26 subjects. So the head is not
                    over-fitting its features; it is under-fitting them.
                    CAVEAT: under --adabn these training windows pass the HELD-OUT session's BN statistics,
                    so head_train is not the accuracy the head had at the end of training. The --no-adabn
                    run is what separates under-fitting from AdaBN disabling the head.
  * --gap-Cs        a sweep of the fresh classifier's regularisation C on the `gap` features (test AND train
                    accuracy per C). It separates two explanations of the fresh classifier's +4 to +5 points:
                    REGULARISATION (the advantage should vanish as C grows) versus something about what the
                    deployed head was fitted to (the advantage should persist at every C).

Where the question stands. A linear classifier on the same EEG reaches 0.6449 with full covariance
features and 0.5872 with the rectified band power that a conv+ReLU+GAP branch CAN represent, while
the branch's own head sits at 0.487 (diag_eeg_branch_gap.py). Of the three candidate causes, one is
settled and two are not:

  (a) the auxiliary weight of 0.2 on l_eeg   -- RULED OUT: --lambda-eeg 1.0 left the head at 0.4846
                                                while overall accuracy fell 0.6074 -> 0.5880
  (b) the shared trunk is shaped by what EFGF/CAFE/fusion need, not by y_eeg   -- untested
  (c) too little independent training data (18 blocks per fold)                -- untested

This script separates (b) from (c) WITHOUT training anything. It loads each fold's saved checkpoint,
takes the EEG branch's own stage-2 feature map, and fits a FRESH logistic regression on the two
training sessions, scoring the held-out session:

  gap          mean over time of the stage-2 map, i.e. exactly what clf_eeg is given
  logvar       log variance over time of the same map -- the band power a mean discards
  gap+logvar   both

Reading it. If `gap` on a fresh classifier lands near the head's own 0.487, the trunk's features
really do carry only that much and the cause is (c) or the trunk itself; the head is not wasting
anything. If `gap` lands near 0.587, the features were adequate and the deployed head is the loss,
which is a fixable training problem rather than a representational limit. And if `logvar` is well
above `gap`, the band power survives into the trunk and only the GAP read-out discards it -- which
would have to be reconciled with the real --eeg-temporal-mode gap_logvar run, which gained +0.8
points on the head with p=0.67.

AdaBN. Applied exactly as ensemble_eval.py does, on the held-out session's unlabeled windows, before
any feature is extracted -- so the features come from the same model state the reported 0.487 comes
from. Features of the two TRAINING sessions are then taken from that same adapted state, which is
what a deployed system would do; it also means the training features carry the test session's
normalisation, the same transductive assumption disclosed for E3/AdaBN throughout.

This probe cannot say what a RETRAINED branch would reach. It reads the trunk of a model trained
under the joint objective, which is the thing being diagnosed.

usage
  python probe_eeg_trunk.py --self-test
  python probe_eeg_trunk.py --data-root "E:\\...\\TSFNET_PAPER_EXACT_nback" \
        --results-dir "E:\\...\\results_40b7d8ea53_seed42" [--device cuda] [--out trunk_folds.csv]
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

FEATS = ("gap", "logvar", "gap+logvar")
# reference points measured elsewhere in this project, printed with the result so the numbers are
# read against the same scale every time
REF = {"branch head (y_eeg)": 0.487, "linear probe, rectified band power": 0.5872,
       "linear probe, full covariance": 0.6449}


def pool_features(s2, kind):
    """s2: (B, C, H, W, T) stage-2 map -> (B, D). 'gap' reproduces what clf_eeg is given."""
    import torch
    B = s2.shape[0]
    gap = s2.mean(dim=-1).reshape(B, -1)
    if kind == "gap":
        return gap
    lv = torch.log(torch.clamp(s2.var(dim=-1, unbiased=False), min=1e-6)).reshape(B, -1)
    if kind == "logvar":
        return lv
    if kind == "gap+logvar":
        return torch.cat([gap, lv], dim=-1)
    raise ValueError(kind)


def fit_eval(Xtr, ytr, Xte, yte, C=0.1):
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean())


def fit_eval2(Xtr, ytr, Xte, yte, C=0.1):
    """like fit_eval, also returns the training accuracy (the over-fitting read-out)"""
    if len(np.unique(ytr)) < 2 or len(yte) == 0:
        return float("nan"), float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    clf.fit(Xtr, ytr)
    return float((clf.predict(Xte) == yte).mean()), float((clf.predict(Xtr) == ytr).mean())


def self_test():
    """The measurement chain is: a forward hook grabs the stage-2 map, pool_features reduces it, a
    fresh classifier is fitted. Each link is checked on data where the right answer is known.

    The planted map has a class effect ONLY in its variance over time, with the mean identical
    across classes, so 'gap' must be at chance and 'logvar' must find it. If that contrast cannot be
    measured here, the reading of the real trunk means nothing."""
    import torch
    import torch.nn as nn
    ok = True
    rng = np.random.RandomState(0)
    N, C, H, W, T = 240, 8, 4, 4, 50
    y = np.tile([0, 1, 2], N // 3)
    ses = np.repeat([1, 2, 3], N // 3)
    amp = np.array([1.0, 1.6, 2.2])[y]
    s2 = rng.randn(N, C, H, W, T) * amp[:, None, None, None, None]
    s2 = s2 - s2.mean(axis=-1, keepdims=True)          # variance carries the class, mean is zero
    # A class-independent offset per (window, unit) is added on purpose. Without it the time-mean is
    # zero only to floating-point precision, and the size of THAT residue still scales with the
    # planted amplitude, so StandardScaler -- which divides each column by its own spread -- blows the
    # residue up into a usable signal and 'gap' scores 0.51 instead of chance. The offset dominates
    # the residue and makes the mean genuinely class-independent, which is what the test needs.
    s2 = s2 + rng.randn(N, C, H, W, 1)
    s2 = torch.from_numpy(s2)

    got = {}
    for k in FEATS:
        X = pool_features(s2, k).numpy()
        tr, te = ses != 3, ses == 3
        got[k] = fit_eval(X[tr], y[tr], X[te], y[te])
    print(f"[planted variance-only] " + " | ".join(f"{k} {got[k]:.3f}" for k in FEATS))
    good = got["gap"] < 0.45 and got["logvar"] > 0.85
    print(f"  the time-mean is blind to it, the log-variance finds it -> {'OK' if good else 'FAIL'}")
    ok &= good

    good = pool_features(s2, "gap").shape[1] == C * H * W == 128
    print(f"[dims] gap gives one feature per stage-2 unit -> {'OK' if good else 'FAIL'}")
    ok &= good
    good = pool_features(s2, "gap+logvar").shape[1] == 2 * C * H * W
    print(f"[dims] gap+logvar concatenates both -> {'OK' if good else 'FAIL'}")
    ok &= good

    # the hook must capture the branch's SECOND return value, not the first or third
    class FakeBranch(nn.Module):
        def forward(self, x):
            return x * 0.0, x, x.mean(dim=-1)

    grabbed = {}
    fb = FakeBranch()
    h = fb.register_forward_hook(lambda m, i, o: grabbed.__setitem__("s2", o[1].detach()))
    probe = torch.randn(5, C, H, W, T, dtype=torch.float64)
    fb(probe)
    h.remove()
    good = "s2" in grabbed and torch.allclose(grabbed["s2"], probe)
    print(f"[hook] the forward hook captures the stage-2 map -> {'OK' if good else 'FAIL'}")
    ok &= good

    # a classifier fitted on the training sessions must never see the held-out session
    tr_idx = np.where(ses != 3)[0]
    good = not np.any(ses[tr_idx] == 3)
    print(f"[split] the training rows exclude the held-out session -> {'OK' if good else 'FAIL'}")
    ok &= good
    # the C sweep must expose memorisation in its TRAIN column and report a separate TEST column.
    # (In this synthetic world weak regularisation costs only ~1-2 points of test accuracy even though it
    # memorises the training rows -- so the sweep is a MEASURE of how much C matters, and a flat result on
    # the real trunk would be informative in itself, not a failure of the tool.)
    def world(seed, n=150, d=400, k=40, shift=0.35):
        r = np.random.RandomState(seed)
        mu = r.randn(3, k) * shift
        yy = np.tile([0, 1, 2], n // 3)
        X = r.randn(n, d)
        X[:, :k] += mu[yy]
        yt = np.tile([0, 1, 2], 300)
        Xt = r.randn(900, d)
        Xt[:, :k] += mu[yt]
        return X, yy, Xt, yt

    res = []
    for sd in range(6):
        X2, y2, Xte, yte = world(sd)
        res.append((fit_eval2(X2, y2, Xte, yte, C=0.001), fit_eval2(X2, y2, Xte, yte, C=100.0)))
    lo_te = np.mean([r[0][0] for r in res]); lo_tr = np.mean([r[0][1] for r in res])
    hi_te = np.mean([r[1][0] for r in res]); hi_tr = np.mean([r[1][1] for r in res])
    print(f"[C sweep] C=0.001 test {lo_te:.3f} train {lo_tr:.3f} | C=100 test {hi_te:.3f} train {hi_tr:.3f}")
    good = hi_tr > 0.98 and hi_tr >= lo_tr and lo_te >= hi_te - 0.005 and lo_te > 0.5
    print(f"  train column shows memorisation at weak regularisation, test column is separate and above chance "
          f"-> {'OK' if good else 'FAIL'}")
    ok &= good
    X2, y2, Xte, yte = world(0)
    good = abs(fit_eval(X2, y2, Xte, yte) - fit_eval2(X2, y2, Xte, yte)[0]) < 0.02
    print(f"[v1 == v2] the default classifier path agrees with v1 (up to the max_iter setting) -> "
          f"{'OK' if good else 'FAIL'}")
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
    ap.add_argument("--out")
    ap.add_argument("--gap-Cs", nargs="+", type=float, default=None,
                    help="extra: regularisation sweep of the fresh classifier on `gap` (e.g. 0.001 0.01 0.1 1 100)")
    ap.add_argument("--self-test", action="store_true")
    # architecture flags: defaults are the accepted baseline of this project
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
    print(f"[init] device = {device}, adabn = {a.adabn}")
    rows = []
    for sid in a.subjects:
        sub_dir = os.path.join(a.data_root, sid)
        if not os.path.isdir(sub_dir):
            continue
        try:
            full = NBackDataset(sub_dir, sid)                                    # noqa: F821
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
            model = TSFNet(num_classes=3, dropout=a.dropout, cafe_dropout=a.cafe_dropout,   # noqa: F821
                           cafe_d_model=a.cafe_d_model, fnirs_num_segments=a.fnirs_num_segments,
                           fnirs_temporal_align=a.fnirs_temporal_align,
                           efgf_gate_mode=a.efgf_gate_mode, efgf_gate_init=a.efgf_gate_init).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device), strict=True)
            model.eval()
            te_idx = np.where(ses_all == test_s)[0]
            te_loader = DataLoader(Subset(full, te_idx), batch_size=a.batch_size, shuffle=False)
            if a.adabn:
                adapt_batchnorm_to_target(model, te_loader, device, log_fn=lambda *x, **k: None)  # noqa: F821

            grabbed = {}
            hook = model.eeg_branch.register_forward_hook(
                lambda m, i, o: grabbed.__setitem__("s2", o[1].detach()))
            feats, yhead = {k: [] for k in FEATS}, []
            order = np.concatenate([np.where(ses_all != test_s)[0], te_idx])
            loader = DataLoader(Subset(full, order.tolist()), batch_size=a.batch_size, shuffle=False)
            with torch.no_grad():
                for batch in loader:
                    x_eeg, x_fnirs = batch[0].to(device), batch[1].to(device)
                    y_eeg, _f, _n, _p, _aux = model(x_eeg, x_fnirs, None, None)
                    s2 = grabbed["s2"]
                    for k in FEATS:
                        feats[k].append(pool_features(s2, k).cpu().numpy())
                    yhead.append(y_eeg.cpu().numpy())
            hook.remove()
            del model
            if str(device).startswith("cuda"):
                torch.cuda.empty_cache()

            y_ord = labels[order]
            is_te = np.concatenate([np.zeros(len(order) - len(te_idx), bool), np.ones(len(te_idx), bool)])
            head_acc = float((np.concatenate(yhead).argmax(1)[is_te] == y_ord[is_te]).mean())
            head_train = float((np.concatenate(yhead).argmax(1)[~is_te] == y_ord[~is_te]).mean())
            row = dict(subject=sid, session=int(test_s), n_test=int(is_te.sum()), head=head_acc,
                       head_train=head_train)
            for k in FEATS:
                X = np.concatenate(feats[k])
                row[k] = fit_eval(X[~is_te], y_ord[~is_te], X[is_te], y_ord[is_te])
            if a.gap_Cs:
                Xg = np.concatenate(feats["gap"])
                for Cv in a.gap_Cs:
                    te_acc, tr_acc = fit_eval2(Xg[~is_te], y_ord[~is_te], Xg[is_te], y_ord[is_te], C=Cv)
                    row[f"gap@C{Cv:g}"] = te_acc
                    row[f"gap_train@C{Cv:g}"] = tr_acc
            rows.append(row)
            print(f"[{sid} s{test_s}] head {head_acc:.3f} (train {head_train:.3f}) | "
                  + " ".join(f"{k} {row[k]:.3f}" for k in FEATS), flush=True)
    if not rows:
        raise SystemExit("no fold could be probed")
    if a.out:
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("per-fold rows ->", a.out)

    subs = sorted({r["subject"] for r in rows})
    per = lambda k: np.array([np.mean([r[k] for r in rows if r["subject"] == s]) for s in subs])
    print(f"\n{len(subs)} subjects, {len(rows)} folds. mean over subjects:")
    print(f"  {'the branch head itself':34} {per('head').mean():.4f}")
    for k in FEATS:
        print(f"  {'fresh classifier on ' + k:34} {per(k).mean():.4f}")
    print("\nreference points measured elsewhere:")
    for k, v in REF.items():
        print(f"  {k:34} {v:.4f}")
    print("\npaired against the branch head (Wilcoxon over subjects):")
    h = per("head")
    for k in FEATS:
        d = per(k) - h
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  {k:12} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    d = per("logvar") - per("gap")
    p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    print(f"  {'logvar - gap':12} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    ht = per("head_train")
    print(f"\nthe deployed head: accuracy on its TRAINING windows {ht.mean():.4f} vs held-out "
          f"{h.mean():.4f}. Read this against the fresh classifier's train column below, not on its "
          f"own: over-fitting would mean the head BEATS the fresh classifier on the training rows.")
    if a.gap_Cs:
        print("\nfresh classifier on `gap`, regularisation sweep (test | train | paired vs the head):")
        for Cv in a.gap_Cs:
            te = per(f"gap@C{Cv:g}")
            tr = per(f"gap_train@C{Cv:g}")
            d = te - h
            p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            print(f"  C={Cv:<8g} test {te.mean():.4f} | train {tr.mean():.4f} | vs head {d.mean():+.4f} "
                  f"wins {int((d > 0).sum())}/{len(d)} p={p:.4f}")
        Cmin = min(a.gap_Cs)
        dt = per(f"gap_train@C{Cmin:g}") - ht
        pt = wilcoxon(dt).pvalue if np.any(dt != 0) else 1.0
        print(f"  TRAIN accuracy, the most regularised fresh classifier (C={Cmin:g}) minus the head: "
              f"{dt.mean():+.4f} wins {int((dt > 0).sum())}/{len(dt)} p={pt:.2g}")
        print("  reading: if the advantage over the head shrinks toward 0 as C grows, the head's deficit is "
              "regularisation; if it persists at every C, it is not. And if the head also trails on the "
              "TRAIN row, it is under-fitting its own features rather than over-fitting them.")
        low = [r for r in rows if r["head_train"] < 0.55]
        if low:
            hi = [r for r in rows if r["head_train"] >= 0.55]
            g = lambda rs, k: float(np.mean([r[k] for r in rs]))
            print(f"\n{len(low)} of {len(rows)} folds have the head below 0.55 on its own training windows. "
                  f"There its held-out accuracy is {g(low, 'head'):.4f} against {g(hi, 'head'):.4f} "
                  f"elsewhere, and the fresh classifier's advantage is "
                  f"{g(low, 'gap') - g(low, 'head'):+.4f} against {g(hi, 'gap') - g(hi, 'head'):+.4f}. "
                  f"Under --adabn this can be the held-out session's BN statistics disabling the head "
                  f"rather than under-fitting; the --no-adabn run separates the two.")
    print("\nThis reads the trunk of a model trained under the joint objective; it does not say what a "
          "retrained EEG-only branch would reach. AdaBN was applied on the held-out session before "
          "any extraction, so the training-session features carry that session's normalisation -- the "
          "same transductive assumption disclosed for E3/AdaBN.")


if __name__ == "__main__":
    main()
