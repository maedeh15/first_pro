"""
block_window_weighting.py -- the 35 windows of a block are NOT equally informative. Does weighting
them by position inside the block raise the BLOCK accuracy, which is this project's applied metric?

Why there is reason to expect something. A block is 2 s instruction + 40 s task + 20 s rest, and
the pipeline slides a 10 s window with a 1 s step over [-2, +42] s, so window w starts at
(-2 + w) s and ends at (8 + w) s:
  * window 0 covers [-2, 8]: two of its ten seconds are the INSTRUCTION, before the task begins;
  * window 34 covers [32, 42]: two of its ten seconds are AFTER the 40 s task ended;
  * the fNIRS haemodynamic response lags the stimulus by about 6 s and does not return to
    baseline until roughly 10 s after it, so the earliest windows carry little fNIRS signal;
  * Cao et al. (Sensors 2022), on THIS dataset, slid a window over the series and found fNIRS
    accuracy peaked in the 45-50 s interval, selecting 45-54 s -- i.e. the far end, into the rest
    period. Their time axis starts at the series onset, so the comparable region here is the
    latest windows this pipeline produces.
Block aggregation currently averages (or votes over) all 35 equally.

Why it is nearly free: ensemble_eval.py --save-posteriors already writes every window's posterior
AND its position in the block (pos_in_trial). This is a re-analysis of saved output, not a run.

What is and is not honest here. A cut-off chosen to maximise accuracy on the fold it is then
scored on is worthless, so three numbers are reported, and only the second is an estimate of what
this would give in practice:
  * BASELINE     -- all windows, the current pipeline;
  * NESTED       -- for each fold, the cut-off is chosen on the SUBJECT'S OTHER FOLDS, whose
                    labels are not the held-out session's. Also a leave-one-subject-out variant
                    that picks one global cut-off from the other 25 subjects;
  * HINDSIGHT    -- the best cut-off per fold, knowing the answer. An UPPER BOUND, never a result.
This project has measured before (nested band/C tuning of the member: +0.0003 nested against
+0.41 in hindsight) that per-subject selection at this data scale buys almost nothing, so expect
the nested column to sit far below the hindsight one.

Window-level accuracy cannot change: this only alters how windows are pooled into a block.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

HEADS = ("final", "eeg", "fusion", "fnirs")


def load_dir(d, head="final"):
    """-> {subject: {session: dict(y, trial, pos, P)}}; P is the seed-ensembled posterior"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        base = os.path.basename(f)[:-4]
        sid, ses = base.split("_session")
        key = f"P_{head}"
        if key not in z:
            raise SystemExit(f"{f} has no {key}; heads present: "
                             f"{[k[2:] for k in z.files if k.startswith('P_')]}")
        P = z[key].astype(np.float64)                       # (K, N, 3)
        P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        out.setdefault(sid, {})[int(ses)] = {
            "y": z["y"], "trial": z["trial_ids"], "pos": z["pos_in_trial"],
            "P": P.mean(0)}                                  # arithmetic seed ensemble
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def profiles(max_pos):
    """one-parameter families, so the nested choice has a single degree of freedom.

    drop<k>  : ignore the first k windows of the block (the instruction/pre-response end)
    tail<k>  : keep only the last k windows (Cao et al.'s late interval)
    The identity (drop0) is the current pipeline and is always the first candidate."""
    names = ["drop0"]
    names += [f"drop{k}" for k in range(1, max_pos + 1, 2)]
    names += [f"tail{k}" for k in range(2, max_pos + 1, 2)]
    return names


def weight_of(name, pos, max_pos):
    if name.startswith("drop"):
        return (pos >= int(name[4:])).astype(np.float64)
    if name.startswith("tail"):
        k = int(name[4:])
        return (pos > max_pos - k).astype(np.float64)
    raise ValueError(name)


def block_acc(fold, name, max_pos, how="mean"):
    """block decision from the weighted windows of each block. Returns NaN if a profile leaves a
    block with no window at all, so an empty selection can never be scored as if it had worked."""
    w = weight_of(name, fold["pos"], max_pos)
    ok = []
    for t in np.unique(fold["trial"]):
        m = (fold["trial"] == t) & (w > 0)
        if not m.any():
            return float("nan")
        if how == "mean":
            pred = fold["P"][m].mean(0).argmax()
        else:
            pred = np.bincount(fold["P"][m].argmax(1), minlength=3).argmax()
        ok.append(pred == fold["y"][fold["trial"] == t][0])
    return float(np.mean(ok))


def evaluate(data, how="mean"):
    max_pos = max(int(f["pos"].max()) for s in data.values() for f in s.values())
    names = profiles(max_pos)
    # acc[subject][session][profile]
    acc = {s: {k: {n: block_acc(f, n, max_pos, how) for n in names} for k, f in d.items()}
           for s, d in data.items()}
    subs = sorted(data)

    def mean_over_folds(s, n, skip=None):
        v = [acc[s][k][n] for k in acc[s] if k != skip]
        return float(np.nanmean(v)) if v else float("nan")

    base = np.array([mean_over_folds(s, "drop0") for s in subs])

    # NESTED, per subject: for each fold, pick the profile on that subject's OTHER folds
    nested = []
    picked = []
    for s in subs:
        per_fold = []
        for k in acc[s]:
            cand = [(mean_over_folds(s, n, skip=k), n) for n in names]
            cand = [c for c in cand if not np.isnan(c[0])]
            best = max(cand)[1]                   # ties -> the name that sorts last; deterministic
            picked.append(best)
            per_fold.append(acc[s][k][best])
        nested.append(float(np.nanmean(per_fold)))
    nested = np.array(nested)

    # NESTED, global: one cut-off from the other 25 subjects
    glob_nested = []
    for s in subs:
        others = [x for x in subs if x != s]
        cand = [(float(np.nanmean([mean_over_folds(o, n) for o in others])), n) for n in names]
        cand = [c for c in cand if not np.isnan(c[0])]
        best = max(cand)[1]
        glob_nested.append(mean_over_folds(s, best))
    glob_nested = np.array(glob_nested)

    # HINDSIGHT upper bound: best profile per fold, knowing the answer
    hind = np.array([float(np.nanmean([max(acc[s][k][n] for n in names) for k in acc[s]]))
                     for s in subs])
    # and the single best FIXED profile over everything, also in hindsight
    fixed = {n: float(np.nanmean([mean_over_folds(s, n) for s in subs])) for n in names}
    return subs, names, acc, base, nested, glob_nested, hind, fixed, picked, max_pos


def report(tag, base, x, subs):
    d = x - base
    ok = ~np.isnan(d)
    p = 1.0 if not np.any(d[ok] != 0) else float(wilcoxon(base[ok], x[ok]).pvalue)
    print(f"  {tag:34} {np.nanmean(x):.4f}  {np.nanmean(d):+.4f}  "
          f"wins {int((d[ok] > 0).sum())}/{int(ok.sum())}  p={p:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posteriors", help="the posteriors/ folder from ensemble_eval.py")
    ap.add_argument("--head", default="final", choices=list(HEADS))
    ap.add_argument("--how", default="mean", choices=["mean", "vote"],
                    help="block decision: mean posterior (default) or majority vote")
    ap.add_argument("--curve", action="store_true", help="print block accuracy for every profile")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.posteriors:
        sys.exit("need --posteriors")
    data = load_dir(a.posteriors, a.head)
    subs, names, acc, base, nested, gnest, hind, fixed, picked, max_pos = evaluate(data, a.how)
    print(f"{len(subs)} subjects, {sum(len(v) for v in data.values())} folds, "
          f"window positions 0..{max_pos}, head={a.head}, block decision={a.how}\n")
    print(f"  {'':34} {'block':>6}  {'vs all-windows':>14}")
    report("BASELINE (all windows)", base, base, subs)
    report("NESTED, per subject", base, nested, subs)
    report("NESTED, leave-one-subject-out", base, gnest, subs)
    report("HINDSIGHT per fold (UPPER BOUND)", base, hind, subs)
    bestfix = max(fixed.items(), key=lambda kv: kv[1])
    report(f"best FIXED profile in hindsight ({bestfix[0]})", base,
           np.array([np.nanmean([float(np.nanmean([acc[s][k][bestfix[0]]])) for k in acc[s]])
                     for s in subs]), subs)
    from collections import Counter
    print(f"\n  profiles the nested rule picked: {Counter(picked).most_common(5)}")
    if a.curve:
        print(f"\n  {'profile':9} {'block':>7}")
        for n in names:
            print(f"  {n:9} {fixed[n]:7.4f}")
    print("\nOnly the two NESTED rows are estimates of what this would give in practice. The")
    print("hindsight rows are upper bounds and must never be reported as results. Window-level")
    print("accuracy is unchanged by construction: this changes only how windows pool into a block.")


def self_test():
    rng = np.random.RandomState(0)
    # planted: the LAST 10 windows of every block are informative, the first 25 are pure noise.
    # The nested rule must find that, and dropping nothing must be clearly worse.
    nsub, nses, ntri, nwin = 12, 3, 9, 35
    data = {}
    for i in range(nsub):
        data[f"VP{i:03d}"] = {}
        for s in range(1, nses + 1):
            y, tri, pos, P = [], [], [], []
            for t in range(ntri):
                c = t % 3
                for w in range(nwin):
                    y.append(c); tri.append(t); pos.append(w)
                    p = np.full(3, 1 / 3) + rng.randn(3) * 0.02
                    if w >= nwin - 10:
                        p[c] += 0.9                       # informative tail
                    else:
                        # a CONSISTENTLY MISLEADING head, not noise: pure noise averages out over
                        # 25 windows and the baseline would already score 0.99, leaving nothing to
                        # detect. The instruction period plausibly looks like one particular class.
                        p[(c + 1) % 3] += 0.6
                    P.append(np.clip(p, 1e-9, None))
            P = np.array(P); P /= P.sum(1, keepdims=True)
            data[f"VP{i:03d}"][s] = {"y": np.array(y), "trial": np.array(tri),
                                     "pos": np.array(pos), "P": P}
    subs, names, acc, base, nested, gnest, hind, fixed, picked, mx = evaluate(data)
    print(f"[planted: only the last 10 of 35 windows are informative]")
    print(f"  all windows {base.mean():.3f} | nested per-subject {nested.mean():.3f} | "
          f"nested LOSO-subject {gnest.mean():.3f} | hindsight {hind.mean():.3f}")
    assert nested.mean() > base.mean() + 0.15, (base.mean(), nested.mean())
    assert gnest.mean() > base.mean() + 0.15
    assert hind.mean() >= nested.mean() - 1e-9, "hindsight must bound nested"
    kept = [p for p in picked if p.startswith("tail") or (p.startswith("drop") and int(p[4:]) >= 20)]
    assert len(kept) > 0.8 * len(picked), set(picked)
    print("  the nested rule recovers a late-window profile and the hindsight bound holds")

    # a profile that empties a block must score NaN, never a silent 0 or a lucky guess
    one = data["VP000"][1]
    assert np.isnan(block_acc(one, f"drop{mx + 1}", mx)), "an empty selection must be NaN"
    # with NO structure, nested must not invent a gain
    flat = {}
    for i in range(nsub):
        flat[f"VP{i:03d}"] = {}
        for s in range(1, nses + 1):
            y, tri, pos, P = [], [], [], []
            for t in range(ntri):
                c = t % 3
                for w in range(nwin):
                    y.append(c); tri.append(t); pos.append(w)
                    # weak per-window signal in heavy noise, so block accuracy lands in the
                    # middle rather than at the ceiling -- otherwise "nested must not invent a
                    # gain" would pass trivially at 1.000
                    p = np.full(3, 1 / 3) + rng.randn(3) * 0.9
                    p[c] += 0.12
                    P.append(np.clip(p, 1e-9, None))
            P = np.array(P); P /= P.sum(1, keepdims=True)
            flat[f"VP{i:03d}"][s] = {"y": np.array(y), "trial": np.array(tri),
                                     "pos": np.array(pos), "P": P}
    _, _, _, b2, n2, g2, h2, _, _, _ = evaluate(flat)
    print(f"[planted: every window equally informative] all {b2.mean():.3f} | nested {n2.mean():.3f}"
          f" | hindsight {h2.mean():.3f}")
    assert n2.mean() <= b2.mean() + 1e-9, "nested must not invent a gain on position-free data"
    print("  on position-free data the nested rule adds nothing, as it must")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
