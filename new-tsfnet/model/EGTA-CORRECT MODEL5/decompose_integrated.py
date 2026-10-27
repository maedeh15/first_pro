"""
decompose_integrated.py -- WHY does the in-network member (--riem-branch member) fall short of the
offline late fusion?  Inference-only, from saved posteriors.

The integrated network returns y_final = 0.5*y_deep + 0.5*y_riem. ensemble_eval.py --save-posteriors on
that run now also stores y_riem and y_deep (P_riem, P_deep). With the separately trained baseline
(P_final of the plain run) and, optionally, the offline member's own posteriors, four competing
explanations become four numbers. They are hypotheses, not findings, until this has run.

  H-head   the in-network head is a WORSE classifier than the offline logistic regression
           -> acc(P_riem) is below the offline member's (0.6449 on this data, window level).
  H-trunk  joint training leaves a WORSE deep half than a separately trained one
           -> acc(P_deep) is below acc(baseline). (Nothing forces them equal: the trunk is trained under
           the blended loss, so it can learn to lean on the head.)
  H-corr   joint training makes the two halves' ERRORS MORE ALIKE, so the average gains less
           -> the phi correlation of correctness between deep and riem is higher inside the network
           than between the baseline and the same head.
  H-swap   (the decisive one) late fusion of the SEPARATE baseline with the in-network head
           -> if this is near 0.67 the loss is in the trunk; if it stays near 0.65 the loss is in the head.

All figures are window level and block level (mean-posterior pooling), per subject, paired Wilcoxon.
Seeds inside a folder are averaged first (as combine_posterior_dirs does), so give matching seeds.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon


def load(d, head):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        if f"P_{head}" not in z:
            raise SystemExit(f"{f} has no P_{head} (was it written before --riem-branch member saved it?)")
        sid, ses = os.path.basename(f)[:-4].split("_session")
        P = z[f"P_{head}"].astype(np.float64)
        P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        out[(sid, int(ses))] = dict(P=P.mean(0), y=z["y"], tri=z["trial_ids"], pos=z["pos_in_trial"])
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def check_aligned(a, b, la, lb):
    if set(a) != set(b):
        raise SystemExit(f"'{la}' and '{lb}' cover different folds, e.g. {sorted(set(a) ^ set(b))[:3]}")
    for k in a:
        if not all(np.array_equal(a[k][f], b[k][f]) for f in ("y", "tri", "pos")):
            raise SystemExit(f"fold {k}: '{la}' and '{lb}' disagree on labels / trials / window order")


def per_subject(parts, fn):
    """fn(fold_dict_of_P, y, tri) -> tuple of floats; returns {subject: mean over folds}."""
    acc = {}
    for k in sorted(parts[0]):
        v = fn([p[k]["P"] for p in parts], parts[0][k]["y"], parts[0][k]["tri"])
        acc.setdefault(k[0], []).append(v)
    return {s: np.mean(v, axis=0) for s, v in acc.items()}


def accs(P, y, tri):
    w = float((P.argmax(1) == y).mean())
    b = float(np.mean([P[tri == t].mean(0).argmax() == y[tri == t][0] for t in np.unique(tri)]))
    return w, b


def phi(P, Q, y):
    """phi coefficient between 'A is correct' and 'B is correct' over windows (1 = identical errors)."""
    a, b = (P.argmax(1) == y).astype(float), (Q.argmax(1) == y).astype(float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def row(label, d):
    S = sorted(d)
    return label, np.array([d[s][0] for s in S]), np.array([d[s][1] for s in S])


def compare(base, x):
    dd = x - base
    p = 1.0 if not np.any(dd != 0) else float(wilcoxon(base, x).pvalue)
    return dd.mean(), int((dd > 0).sum()), len(dd), p


def run(integ, baseline, member, blend):
    deep, riem, fin = load(integ, "deep"), load(integ, "riem"), load(integ, "final")
    base = load(baseline, "final")
    for a, b, la, lb in ((deep, riem, "deep", "riem"), (deep, fin, "deep", "final"), (deep, base, "integrated", "baseline")):
        check_aligned(a, b, la, lb)
    # algebraic identity the whole analysis rests on -- refuse to continue if it fails
    for k in deep:
        rec = (1 - blend) * deep[k]["P"] + blend * riem[k]["P"]
        if not np.allclose(rec, fin[k]["P"], atol=1e-4):
            raise SystemExit(f"fold {k}: (1-b)*deep + b*riem != final (max err "
                             f"{np.abs(rec - fin[k]['P']).max():.2e}); wrong --blend or a stale posteriors folder")
    mix = lambda Ps: (1 - blend) * Ps[0] + blend * Ps[1]
    rows = [row("integrated FINAL (what was reported)", per_subject([fin], lambda P, y, t: accs(P[0], y, t))),
            row("  its deep half alone", per_subject([deep], lambda P, y, t: accs(P[0], y, t))),
            row("  its Riemannian head alone", per_subject([riem], lambda P, y, t: accs(P[0], y, t))),
            row("baseline deep (separately trained)", per_subject([base], lambda P, y, t: accs(P[0], y, t))),
            row("SWAP: baseline deep + in-network head", per_subject([base, riem], lambda P, y, t: accs(mix(P), y, t)))]
    if member:
        mem = load(member, "final")
        check_aligned(deep, mem, "integrated", "member")
        rows += [row("offline member alone", per_subject([mem], lambda P, y, t: accs(P[0], y, t))),
                 row("SWAP: in-network deep + offline member", per_subject([deep, mem], lambda P, y, t: accs(mix(P), y, t))),
                 row("late fusion (baseline + offline member)", per_subject([base, mem], lambda P, y, t: accs(mix(P), y, t)))]
    ref = rows[0]
    print(f"{len(ref[1])} subjects, blend={blend}\n")
    print(f"  {'':44} {'window':>8} {'block':>8}   vs integrated FINAL (window)")
    for lab, w, b in rows:
        if lab is ref[0]:
            print(f"  {lab:44} {w.mean():8.4f} {b.mean():8.4f}")
        else:
            d, win, n, p = compare(ref[1], w)
            print(f"  {lab:44} {w.mean():8.4f} {b.mean():8.4f}   {d:+.4f}  {win}/{n}  p={p:.4f}")
    cor_in = per_subject([deep, riem], lambda P, y, t: (phi(P[0], P[1], y),))
    cor_out = per_subject([base, riem], lambda P, y, t: (phi(P[0], P[1], y),))
    ci = np.array([cor_in[s][0] for s in sorted(cor_in)]); co = np.array([cor_out[s][0] for s in sorted(cor_out)])
    ok = ~(np.isnan(ci) | np.isnan(co))
    d, win, n, p = compare(co[ok], ci[ok])
    print(f"\n  error correlation (phi of 'correct') between the two halves:")
    print(f"    inside the network (deep, head)            {ci[ok].mean():.4f}")
    print(f"    separate baseline vs the same head         {co[ok].mean():.4f}")
    print(f"    inside minus separate                      {d:+.4f}  {win}/{n}  p={p:.4f}")
    print("\nREADING (decide BEFORE looking): the largest gap among")
    print("  head-alone vs offline member | deep-half vs baseline | SWAP vs FINAL | error correlation")
    print("names the cause. If two of them move, the cause is not single and no one-flag fix is implied.")


def self_test():
    rng = np.random.RandomState(0)
    import tempfile
    tmp = tempfile.mkdtemp()
    nbl, nw = 9, 12
    y = np.repeat(np.arange(nbl) % 3, nw); tri = np.repeat(np.arange(nbl), nw); pos = np.tile(np.arange(nw), nbl)

    def post(acc, shared=None):
        right = (rng.rand(len(y)) < acc) if shared is None else shared
        cls = np.where(right, y, (y + 1 + rng.randint(0, 2, len(y))) % 3)
        P = np.full((len(y), 3), 0.1); P[np.arange(len(y)), cls] += 0.8
        return P / P.sum(1, keepdims=True)

    for sub in range(8):
        for ses in (1, 2, 3):
            deepP, riemP = post(0.6), post(0.6)
            basP = post(0.6)
            fin = 0.5 * deepP + 0.5 * riemP
            for d, kw in (("i", dict(P_deep=deepP, P_riem=riemP, P_final=fin)), ("b", dict(P_final=basP))):
                os.makedirs(os.path.join(tmp, d), exist_ok=True)
                np.savez(os.path.join(tmp, d, f"VP{sub:03d}_session{ses}.npz"), y=y, trial_ids=tri, pos_in_trial=pos,
                         **{k: v[None] for k, v in kw.items()})
    run(os.path.join(tmp, "i"), os.path.join(tmp, "b"), None, 0.5)
    # the identity guard must fire on a wrong blend
    try:
        run(os.path.join(tmp, "i"), os.path.join(tmp, "b"), None, 0.3); raise SystemExit("wrong blend NOT refused")
    except SystemExit as e:
        assert "!= final" in str(e), e
    print("\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated", help="posteriors/ of the --riem-branch member run (needs P_deep, P_riem)")
    ap.add_argument("--baseline", help="posteriors/ of the plain run with the SAME seeds")
    ap.add_argument("--member", default=None, help="optional: posteriors/ holding the offline member as P_final")
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.integrated and a.baseline):
        sys.exit("need --integrated and --baseline")
    run(a.integrated, a.baseline, a.member, a.blend)


if __name__ == "__main__":
    main()
