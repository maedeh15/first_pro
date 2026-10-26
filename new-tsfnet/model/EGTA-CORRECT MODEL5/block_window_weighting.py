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
import argparse, glob, os, re, sys
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


FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
RAMPS = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)


def profiles():
    """One-parameter families, chosen so that (a) no two names denote the same window set and
    (b) no selection can ever be empty, whatever a block's length.

    An earlier version used ABSOLUTE positions, which was wrong twice over. Blocks here hold
    between 26 and 35 windows, so a fixed cut-off such as "keep positions >= 33" empties the
    shorter blocks: those folds scored NaN, the NaN propagated to the subject's mean, and the
    paired test silently dropped 10 of 26 subjects (it printed "wins 0/16"). And the two families
    were redundant: keeping positions >= k is the same set as keeping the last 35-k, so drop33 and
    tail2 were one profile under two names and the candidate list was half the size it looked.

    Positions are therefore normalised BY EACH BLOCK'S OWN LENGTH, so every profile keeps at least
    the boundary window of every block and all rows are computed on all 26 subjects:
      first<f>  drop the first fraction f  (the instruction / pre-response end)
      last<f>   drop the last fraction f   (the windows reaching past the task)
      mid<f>    drop f/2 from BOTH ends
      ramp<a>   a soft weight 1 + a*rel instead of a cut-off, so nothing is discarded at all
    """
    names = ["all"]
    names += [f"first{f:g}" for f in FRACTIONS]
    names += [f"last{f:g}" for f in FRACTIONS]
    names += [f"mid{f:g}" for f in FRACTIONS]
    names += [f"ramp{a:+g}" for a in RAMPS]
    return names


def weight_of(name, rel):
    """rel: each window's position as a fraction of its own block's span, in [0, 1]."""
    if name == "all":
        return np.ones_like(rel)
    if name.startswith("ramp"):
        return np.clip(1.0 + float(name[4:]) * rel, 0.0, None)
    f = float(re.match(r"[a-z]+([0-9.]+)", name).group(1))
    if name.startswith("first"):
        return (rel >= f).astype(np.float64)
    if name.startswith("last"):
        return (rel <= 1.0 - f).astype(np.float64)
    if name.startswith("mid"):
        return ((rel >= f / 2) & (rel <= 1.0 - f / 2)).astype(np.float64)
    raise ValueError(name)


def block_acc(fold, name, how="mean"):
    """Weighted block decision. Every profile keeps at least one window of every block by
    construction; if one ever does not, that is a bug, so it raises instead of returning NaN."""
    ok = []
    for t in np.unique(fold["trial"]):
        m = fold["trial"] == t
        pos = fold["pos"][m].astype(np.float64)
        span = max(pos.max() - pos.min(), 1.0)
        rel = (pos - pos.min()) / span
        w = weight_of(name, rel)
        if w.sum() <= 0:
            raise AssertionError(f"profile {name!r} left block {t} with no weight "
                                 f"({m.sum()} windows) -- the profile families must never do this")
        P = fold["P"][m]
        if how == "mean":
            pred = (P * w[:, None]).sum(0).argmax()
        else:
            cnt = np.zeros(P.shape[1])
            np.add.at(cnt, P.argmax(1), w)
            pred = cnt.argmax()
        ok.append(pred == fold["y"][m][0])
    return float(np.mean(ok))


def evaluate(data, how="mean"):
    names = profiles()
    acc = {s: {k: {n: block_acc(f, n, how) for n in names} for k, f in d.items()}
           for s, d in data.items()}
    subs = sorted(data)

    def mean_over_folds(s, n, skip=None):
        v = [acc[s][k][n] for k in acc[s] if k != skip]
        if not v:
            raise AssertionError(f"{s}: no fold left after skipping {skip}")
        return float(np.mean(v))

    base = np.array([mean_over_folds(s, "all") for s in subs])

    # NESTED, per subject: for each fold the profile is chosen on that subject's OTHER folds
    nested, picked = [], []
    for s in subs:
        per_fold = []
        for k in acc[s]:
            best = max(names, key=lambda n: (mean_over_folds(s, n, skip=k), n))
            picked.append(best)
            per_fold.append(acc[s][k][best])
        nested.append(float(np.mean(per_fold)))
    nested = np.array(nested)

    # NESTED, global: one profile chosen from the OTHER 25 subjects
    glob_nested, picked_glob = [], []
    for s in subs:
        others = [x for x in subs if x != s]
        best = max(names, key=lambda n: (float(np.mean([mean_over_folds(o, n) for o in others])), n))
        picked_glob.append(best)
        glob_nested.append(mean_over_folds(s, best))
    glob_nested = np.array(glob_nested)

    # HINDSIGHT upper bounds, never results
    hind = np.array([float(np.mean([max(acc[s][k][n] for n in names) for k in acc[s]]))
                     for s in subs])
    fixed = {n: float(np.mean([mean_over_folds(s, n) for s in subs])) for n in names}
    bestfix = max(names, key=lambda n: (fixed[n], n))
    hind_fixed = np.array([mean_over_folds(s, bestfix) for s in subs])
    return (subs, names, acc, base, nested, glob_nested, hind, fixed, bestfix, hind_fixed,
            picked, picked_glob)


def report(tag, base, x):
    d = x - base
    if np.isnan(d).any():
        raise AssertionError(f"{tag}: NaN reached the comparison; every profile must score "
                             f"every fold")
    p = 1.0 if not np.any(d != 0) else float(wilcoxon(base, x).pvalue)
    print(f"  {tag:36} {x.mean():.4f}  {d.mean():+.4f}  "
          f"wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")


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
    (subs, names, acc, base, nested, gnest, hind, fixed, bestfix, hind_fixed,
     picked, picked_glob) = evaluate(data, a.how)
    mx = max(int(f["pos"].max()) for d in data.values() for f in d.values())
    lens = sorted({int((f["trial"] == t).sum()) for d in data.values() for f in d.values()
                   for t in np.unique(f["trial"])})
    print(f"{len(subs)} subjects, {sum(len(v) for v in data.values())} folds, "
          f"head={a.head}, block decision={a.how}")
    print(f"window positions 0..{mx}; blocks hold {lens[0]} to {lens[-1]} windows, so every "
          f"profile is defined on each block's OWN span\n")
    print(f"  {'':36} {'block':>6}  {'vs all-windows':>14}")
    report("BASELINE (all windows)", base, base)
    report("NESTED, per subject", base, nested)
    report("NESTED, leave-one-subject-out", base, gnest)
    report("HINDSIGHT per fold (UPPER BOUND)", base, hind)
    report(f"HINDSIGHT best fixed ({bestfix}) (BOUND)", base, hind_fixed)
    from collections import Counter
    print(f"\n  profiles the per-subject rule picked : {Counter(picked).most_common(5)}")
    print(f"  profiles the leave-one-subject rule picked: {Counter(picked_glob).most_common(3)}")
    if a.curve:
        print(f"\n  {'profile':9} {'block':>7}")
        for n in names:
            print(f"  {n:9} {fixed[n]:7.4f}")
    print("\nOnly the two NESTED rows are estimates of what this would give in practice. The")
    print("hindsight rows are upper bounds and must never be reported as results. Window-level")
    print("accuracy is unchanged by construction: this changes only how windows pool into a block.")


def _make(nsub, nses, ntri, nwin_of, filler):
    rng = np.random.RandomState(0)
    data = {}
    for i_ in range(nsub):
        data[f"VP{i_:03d}"] = {}
        for s_ in range(1, nses + 1):
            y, tri, pos, P = [], [], [], []
            for t in range(ntri):
                c = t % 3
                nw = nwin_of(i_, t)
                for w in range(nw):
                    y.append(c); tri.append(t); pos.append(w)
                    P.append(filler(rng, c, w, nw))
            P = np.array(P); P /= P.sum(1, keepdims=True)
            data[f"VP{i_:03d}"][s_] = {"y": np.array(y), "trial": np.array(tri),
                                       "pos": np.array(pos), "P": P}
    return data


def self_test():
    # 1) planted: only the last 30% of each block is informative, and BLOCK LENGTHS DIFFER
    #    (26..35), which is what broke the absolute-cut-off version of this script.
    def tail_filler(rng, c, w, nw):
        p = np.full(3, 1 / 3) + rng.randn(3) * 0.02
        if w >= 0.7 * (nw - 1):
            p[c] += 0.9
        else:
            p[(c + 1) % 3] += 0.6          # consistently misleading head, not noise
        return np.clip(p, 1e-9, None)
    varlen = lambda i_, t: 26 + (i_ + t) % 10
    data = _make(12, 3, 9, varlen, tail_filler)
    (subs, names, acc, base, nested, gnest, hind, fixed, bestfix, hfix,
     picked, picked_glob) = evaluate(data)
    print(f"[planted: only the last 30% informative, block lengths 26..35]")
    print(f"  all {base.mean():.3f} | nested/subject {nested.mean():.3f} | "
          f"nested/LOSO {gnest.mean():.3f} | hindsight {hind.mean():.3f}")
    assert nested.mean() > base.mean() + 0.15 and gnest.mean() > base.mean() + 0.15
    assert hind.mean() >= nested.mean() - 1e-9, "hindsight must bound nested"
    assert all(p.startswith("first") or p.startswith("ramp+") for p in picked), set(picked)
    print("  recovered by BOTH nested rules despite unequal block lengths")

    # 2) THE BUG THE REAL RUN EXPOSED: every row must use all subjects, no NaN anywhere
    for arr, nm in ((base, "base"), (nested, "nested"), (gnest, "LOSO"), (hind, "hindsight"),
                    (hfix, "hindsight-fixed")):
        assert len(arr) == len(subs) and not np.isnan(arr).any(), nm
    assert all(not np.isnan(v) for v in fixed.values())
    print(f"  all {len(names)} profiles score all {len(subs)} subjects; no NaN, no shrunken n")

    # 3) no two profile names denote the same window set (the drop33 == tail2 collapse)
    rel = np.linspace(0, 1, 31)
    sig = {}
    for n in names:
        key = tuple(np.round(weight_of(n, rel), 9))
        assert key not in sig, f"{n} and {sig[key]} are the same profile"
        sig[key] = n
    print(f"  all {len(names)} profiles are distinct as window weightings")

    # 4) no profile can empty a block, even the shortest
    short = {"VP000": {1: data["VP000"][1]}}
    for n in names:
        block_acc(short["VP000"][1], n)          # raises if a block is left with no weight
    print("  no profile empties a block (checked on the shortest blocks present)")

    # 5) position-free data: the nested rules must not invent a gain, hindsight will
    def flat_filler(rng, c, w, nw):
        p = np.full(3, 1 / 3) + rng.randn(3) * 0.9
        p[c] += 0.12
        return np.clip(p, 1e-9, None)
    flat = _make(12, 3, 9, varlen, flat_filler)
    _, _, _, b2, n2, g2, h2, _, _, _, _, _ = evaluate(flat)
    print(f"[planted: every window equally informative] all {b2.mean():.3f} | "
          f"nested {n2.mean():.3f} | LOSO {g2.mean():.3f} | hindsight {h2.mean():.3f}")
    assert n2.mean() <= b2.mean() + 0.01 and g2.mean() <= b2.mean() + 0.01
    assert h2.mean() > b2.mean() + 0.05, "hindsight should visibly over-promise here"
    print("  nested adds nothing on position-free data while hindsight invents a gain")

    # 6) 'all' must reproduce a plain unweighted mean, and ramp0 is not offered as a duplicate
    f0 = data["VP000"][1]
    plain = []
    for t in np.unique(f0["trial"]):
        m = f0["trial"] == t
        plain.append(f0["P"][m].mean(0).argmax() == f0["y"][m][0])
    assert abs(block_acc(f0, "all") - float(np.mean(plain))) < 1e-12
    print("  'all' equals the current unweighted pipeline exactly")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
