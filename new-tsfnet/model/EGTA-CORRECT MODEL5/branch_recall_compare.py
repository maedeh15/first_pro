"""
branch_recall_compare.py -- where did a branch's gain land: 0-back, or the 2-vs-3 pair?

Question. Several interventions raised a single branch's accuracy (EEG head +2 to +3 points) and
left the final decision unchanged. One hypothesis: the gain sits where the fused system was already
right (0-back), not on the 2-vs-3 pair that holds ~71% of the errors. This reads the per-window
posteriors that ensemble_eval.py --save-posteriors writes and splits each head's gain by class.

Input: two posterior folders (A = reference run, B = intervention), each from
    ensemble_eval.py ... --save-posteriors          (one seed per call is fine; seeds are averaged)
Output, per head (final / eeg / fnirs / fusion): per-class recall A, B, diff, subject wins, paired
Wilcoxon p, plus the 2-vs-3 accuracy (train-free: restricted to windows whose true class is 2 or 3,
predictions restricted to the two columns). Subjects are the unit; seeds averaged per subject first.

A reading, not a test of the hypothesis: it says where the gain is, not why the fusion ignores it.
Seed 42 selected nothing here, but the head/intervention pairs were chosen after seeing totals --
treat p-values as descriptive. Holm over the 4 heads x 3 classes is printed for the recall rows.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

HEADS = ("final", "eeg", "fusion", "fnirs")


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        sid = os.path.basename(f).split("_session")[0]
        out.setdefault(sid, []).append({"y": z["y"], **{h: z[f"P_{h}"] for h in HEADS}})
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def subject_stats(folds, head):
    """per-subject: recall of classes 0/1/2 and 2-vs-3 accuracy, pooled over the subject's folds,
    seeds averaged first (each seed scored on its own argmax, then averaged)"""
    rec, p23 = np.zeros(3), []
    cnt = np.zeros(3)
    rc = np.zeros(3)
    c23, n23 = 0.0, 0
    for f in folds:
        y = f["y"]; P = f[head]                               # P: (K, N, 3)
        for k in range(P.shape[0]):
            pred = P[k].argmax(1)
            for c in range(3):
                m = y == c
                rc[c] += (pred[m] == c).sum() / P.shape[0]; cnt[c] += m.sum() / P.shape[0]
            m = (y == 1) | (y == 2)
            sub = P[k][:, 1:3].argmax(1) + 1
            c23 += (sub[m] == y[m]).sum() / P.shape[0]; n23 += m.sum() / P.shape[0]
    return rc / np.maximum(cnt, 1), c23 / max(n23, 1)


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj


def compare(A, B, heads=HEADS):
    subs = sorted(set(A) & set(B))
    if len(subs) < 5:
        raise SystemExit(f"only {len(subs)} common subjects")
    rows = []
    for h in heads:
        a = [subject_stats(A[s], h) for s in subs]; b = [subject_stats(B[s], h) for s in subs]
        ra, rb = np.array([x[0] for x in a]), np.array([x[0] for x in b])
        a23, b23 = np.array([x[1] for x in a]), np.array([x[1] for x in b])
        for name, xa, xb in (("recall 0-back", ra[:, 0], rb[:, 0]), ("recall 2-back", ra[:, 1], rb[:, 1]),
                             ("recall 3-back", ra[:, 2], rb[:, 2]), ("acc 2-vs-3", a23, b23)):
            d = xb - xa
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(xa, xb).pvalue)
            rows.append((h, name, xa.mean(), xb.mean(), d.mean(), int((d > 0).sum()), len(d), p))
    return subs, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--self-test", action="store_true")
    ar = ap.parse_args()
    if ar.self_test:
        return self_test()
    if not (ar.a and ar.b):
        sys.exit("need --a and --b (posterior folders)")
    subs, rows = compare(load_dir(ar.a), load_dir(ar.b))
    rec_p = [r[7] for r in rows if r[1].startswith("recall")]
    adj = iter(holm(rec_p))
    print(f"{len(subs)} subjects, seeds averaged per subject\n")
    print(f"{'head':7s} {'metric':14s} {'A':>7s} {'B':>7s} {'diff':>8s} {'wins':>7s} {'p':>8s} {'Holm(recall)':>12s}")
    for h, name, ma, mb, d, w, n, p in rows:
        ha = f"{next(adj):.4f}" if name.startswith("recall") else "-"
        print(f"{h:7s} {name:14s} {ma:7.4f} {mb:7.4f} {d:+8.4f} {w:3d}/{n:<3d} {p:8.4f} {ha:>12s}")
        if name == "acc 2-vs-3":
            print()
    print("Reading: if a head's gain is on '0-back' and 'acc 2-vs-3' does not move, the gain sits where")
    print("the fused system was already right. Descriptive; the hypothesis explains WHERE, not WHY.")


def self_test():
    rng = np.random.default_rng(0)
    def mk(gain0, n_sub=12):
        out = {}
        for s in range(n_sub):
            folds = []
            for ses in (1, 2, 3):
                y = rng.integers(0, 3, 300)
                P = np.full((1, 300, 3), 0.1) + rng.random((1, 300, 3)) * 0.05
                # base head: 0-back ok 70%, 2/3 ok 55%; gain only raises 0-back
                ok = np.where(y == 0, rng.random(300) < 0.70 + gain0, rng.random(300) < 0.55)
                pred = np.where(ok, y, (y + rng.integers(1, 3, 300)) % 3)
                P[0, np.arange(300), pred] += 1.0
                folds.append({"y": y, **{h: P.copy() for h in HEADS}})
            out[f"VP{s:03d}"] = folds
        return out
    A, B = mk(0.0), mk(0.20)
    _, rows = compare(A, B, heads=("eeg",))
    r = {x[1]: x for x in rows}
    assert r["recall 0-back"][4] > 0.12 and r["recall 0-back"][7] < 0.01, r["recall 0-back"]
    assert abs(r["acc 2-vs-3"][4]) < 0.06 and abs(r["recall 2-back"][4]) < 0.06
    _, same = compare(A, A, heads=("eeg",))
    assert all(abs(x[4]) < 1e-12 and x[7] == 1.0 for x in same)
    print("self-test OK: a planted 0-back-only gain is found there (+%.2f, p=%.4f) and 2-vs-3 stays flat (%+.3f);"
          " A vs A gives diff 0, p=1" % (r["recall 0-back"][4], r["recall 0-back"][7], r["acc 2-vs-3"][4]))


if __name__ == "__main__":
    main()
