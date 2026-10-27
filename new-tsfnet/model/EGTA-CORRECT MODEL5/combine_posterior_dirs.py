"""
combine_posterior_dirs.py -- an ensemble of runs with DIFFERENT configurations, not different seeds.

Why. The seed-ensemble size curve saturated almost immediately on this data: 0.6495 for one seed,
0.6556 for two, 0.6562 for three. Seeds of one architecture are highly correlated, and an ensemble
lives on diversity. Several fully-trained 26-subject runs already exist on disk from interventions
that failed INDIVIDUALLY (each within about a point of the baseline), and they differ by
architecture or training flags rather than by seed, so they are more diverse than seeds. Combining
them costs inference only -- no training.

On the order of operations, one thing is settled by algebra rather than by experiment and is
recorded here so nobody spends a run on it: with ARITHMETIC combination and mean-posterior block
pooling, combining members per window and then averaging over a block is EXACTLY the same number
as averaging each member over the block and then combining, because
    (1/W) sum_w (1/M) sum_m P = (1/M) sum_m (1/W) sum_w P.
The script asserts that equality as a correctness check. The order only matters where the two
operations fail to commute: GEOMETRIC combination (--how geo), or majority-vote block pooling.
Those are the only versions of the question worth measuring.

Inputs are posteriors/ folders written by ensemble_eval.py --save-posteriors, one per
configuration (run ensemble_eval once per configuration WITH THAT CONFIGURATION'S OWN
architecture flags; a single call cannot mix architectures because it builds one model).

Alignment is checked, not assumed: every folder must cover the same subjects and sessions, and
within each fold the labels, trial ids and window order must match exactly. A mismatch pairs one
window's posterior with another window's, which would be invisible in the final number, so it is
refused instead.

Honest reporting:
  * this is K networks: K times the parameters and K times the inference cost. Report it as an
    ensemble of K configurations, never as "the model";
  * which folders go in must be fixed BEFORE looking at the combined score. Pass them all. The
    per-member column is printed so a reader can see none was cherry-picked, and the leave-one-out
    column shows what each member contributes;
  * the size curve averages over ALL subsets of each size, so it is not a search either.
"""
import argparse, glob, itertools, os, sys
import numpy as np
from scipy.stats import wilcoxon

HEADS = ("final", "eeg", "fusion", "fnirs", "riem", "deep")


def load_dir(d, head):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        sid, ses = os.path.basename(f)[:-4].split("_session")
        key = f"P_{head}"
        if key not in z:
            raise SystemExit(f"{f} has no {key}")
        P = z[key].astype(np.float64)
        P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        out[(sid, int(ses))] = {"y": z["y"], "trial": z["trial_ids"], "pos": z["pos_in_trial"],
                                "P": P.mean(0)}            # seeds inside a folder are averaged
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def align(members, labels):
    keys = set(members[0])
    for lab, m in zip(labels[1:], members[1:]):
        if set(m) != keys:
            only = sorted(set(m) ^ keys)[:4]
            raise SystemExit(f"'{lab}' covers different folds; e.g. {only}")
    for k in sorted(keys):
        ref = members[0][k]
        for lab, m in zip(labels[1:], members[1:]):
            f = m[k]
            if not (np.array_equal(f["y"], ref["y"]) and np.array_equal(f["trial"], ref["trial"])
                    and np.array_equal(f["pos"], ref["pos"])):
                raise SystemExit(f"fold {k}: '{lab}' disagrees with '{labels[0]}' on the labels, "
                                 f"trial ids or window order -- refusing to combine")
            if f["P"].shape != ref["P"].shape:
                raise SystemExit(f"fold {k}: '{lab}' has {f['P'].shape} windows, "
                                 f"'{labels[0]}' has {ref['P'].shape}")
    return sorted(keys)


def comb(Ps, how):
    if how == "arith":
        return np.mean(Ps, axis=0)
    g = np.exp(np.mean(np.log(np.clip(Ps, 1e-12, None)), axis=0))
    return g / g.sum(-1, keepdims=True)


def block_from_windows(P, trial, y):
    ok = []
    for t in np.unique(trial):
        m = trial == t
        ok.append(P[m].mean(0).argmax() == y[m][0])
    return np.array(ok, dtype=float)


def block_posteriors(P, trial):
    return np.array([P[trial == t].mean(0) for t in np.unique(trial)])


def score(members, keys, idx, how):
    """-> per-subject arrays for: window acc, block acc combining windows first,
    block acc pooling each member to blocks first and then combining"""
    per = {}
    for (sid, ses) in keys:
        ref = members[idx[0]][(sid, ses)]
        y, tri = ref["y"], ref["trial"]
        Ps = [members[i][(sid, ses)]["P"] for i in idx]
        Pw = comb(Ps, how)
        win = float((Pw.argmax(1) == y).mean())
        blk_wf = float(block_from_windows(Pw, tri, y).mean())
        Bs = [block_posteriors(p, tri) for p in Ps]
        yb = np.array([y[tri == t][0] for t in np.unique(tri)])
        blk_bf = float((comb(Bs, how).argmax(1) == yb).mean())
        per.setdefault(sid, []).append((win, blk_wf, blk_bf))
    subs = sorted(per)
    a = np.array([np.mean(per[s], axis=0) for s in subs])
    return subs, a[:, 0], a[:, 1], a[:, 2]


def paired(base, x):
    d = x - base
    p = 1.0 if not np.any(d != 0) else float(wilcoxon(base, x).pvalue)
    return d.mean(), int((d > 0).sum()), len(d), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", help="one posteriors/ folder per configuration")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--head", default="final", choices=list(HEADS))
    ap.add_argument("--how", default="arith", choices=["arith", "geo"])
    ap.add_argument("--save-combined", default=None,
                    help="write the combined window posteriors of ALL members to this folder, in the "
                         "same *_session<k>.npz layout ensemble_eval.py --save-posteriors writes, so "
                         "balanced_assignment_eval.py (or a second combine) can read it")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.dirs or len(a.dirs) < 2:
        sys.exit("need --dirs with at least two posteriors folders")
    labels = a.labels or [os.path.basename(os.path.dirname(d.rstrip("/\\"))) or d for d in a.dirs]
    if len(labels) != len(a.dirs):
        sys.exit("--labels must match --dirs")
    members = [load_dir(d, a.head) for d in a.dirs]
    keys = align(members, labels)
    K = len(members)
    print(f"{len(set(k[0] for k in keys))} subjects, {len(keys)} folds, {K} configurations, "
          f"head={a.head}, combination={a.how}\n")

    print(f"  {'member':34} {'window':>8} {'block':>8}")
    singles = []
    for i, lab in enumerate(labels):
        subs, w, b, _ = score(members, keys, [i], a.how)
        singles.append((w, b))
        print(f"  {lab[:34]:34} {w.mean():8.4f} {b.mean():8.4f}")
    mean_single_w = np.mean([s[0] for s in singles], axis=0)
    mean_single_b = np.mean([s[1] for s in singles], axis=0)
    print(f"  {'MEAN of the members':34} {mean_single_w.mean():8.4f} {mean_single_b.mean():8.4f}")
    best_single_w = max(singles, key=lambda s: s[0].mean())[0]

    subs, w_all, bwf, bbf = score(members, keys, list(range(K)), a.how)
    print(f"  {'ENSEMBLE of all ' + str(K):34} {w_all.mean():8.4f} {bwf.mean():8.4f}")

    print("\npaired over subjects, ensemble minus the mean of the members:")
    for tag, base, x in (("window", mean_single_w, w_all),
                         ("block (windows combined first)", mean_single_b, bwf)):
        d, win, n, p = paired(base, x)
        print(f"  {tag:34} {d:+.4f}  wins {win}/{n}  p={p:.4f}")
    d, win, n, p = paired(best_single_w, w_all)
    print(f"  {'window, vs the BEST single member':34} {d:+.4f}  wins {win}/{n}  p={p:.4f}"
          f"   (the honest comparison if one member were shipped alone)")

    print("\norder of operations at block level:")
    if a.how == "arith":
        if not np.allclose(bwf, bbf, atol=1e-12):
            sys.exit("BUG: with arithmetic combination the two orders must be identical by algebra")
        print(f"  identical by algebra ({bwf.mean():.4f}), and verified equal to 1e-12. The order")
        print(f"  can only matter for --how geo or for vote pooling; pass --how geo to measure it.")
    else:
        d, win, n, p = paired(bwf, bbf)
        print(f"  pool-to-block THEN combine        {bbf.mean():.4f} vs {bwf.mean():.4f}  "
              f"{d:+.4f}  wins {win}/{n}  p={p:.4f}")

    if K > 2:
        print(f"\nsize curve (average over ALL subsets of each size, so it is not a search):")
        print(f"  {'K':>2} {'#subsets':>9} {'window':>8} {'block':>8}")
        for k in range(1, K + 1):
            subsets = list(itertools.combinations(range(K), k))
            ws = np.mean([score(members, keys, list(c), a.how)[1].mean() for c in subsets])
            bs = np.mean([score(members, keys, list(c), a.how)[2].mean() for c in subsets])
            print(f"  {k:2d} {len(subsets):9d} {ws:8.4f} {bs:8.4f}")
        print("\nleave-one-member-out (what each configuration contributes):")
        for i, lab in enumerate(labels):
            rest = [j for j in range(K) if j != i]
            _, w_r, _, _ = score(members, keys, rest, a.how)
            print(f"  without {lab[:28]:28} {w_r.mean():8.4f}  ({w_all.mean() - w_r.mean():+.4f})")

    if a.save_combined:
        os.makedirs(a.save_combined, exist_ok=True)
        for (sid, ses) in keys:
            ref = members[0][(sid, ses)]
            Pw = comb([m[(sid, ses)]["P"] for m in members], a.how)
            np.savez(os.path.join(a.save_combined, f"{sid}_session{ses}.npz"),
                     **{f"P_{a.head}": Pw[None].astype(np.float32)},
                     y=ref["y"], trial_ids=ref["trial"], pos_in_trial=ref["pos"])
        print(f"\nwrote {len(keys)} combined folds ({a.how}, head={a.head}) to {a.save_combined}")

    print("\nThis is K networks: K times the parameters and K times the inference cost. Report it")
    print("as an ensemble of K configurations. The member folders must be fixed before looking at")
    print("the combined score; the per-member and leave-one-out columns exist so a reader can see")
    print("that none was selected after the fact.")


def self_test():
    rng = np.random.RandomState(0)
    nsub, nses, ntri, nwin = 10, 3, 9, 20

    def make(err_bias, n_members, correlated):
        """n_members posterior sets. correlated=True -> they make the SAME mistakes (like seeds of
        one architecture); False -> independent mistakes (like different architectures)."""
        mem = [dict() for _ in range(n_members)]
        for i in range(nsub):
            for s in range(1, nses + 1):
                y = np.repeat(np.arange(ntri) % 3, nwin)
                tri = np.repeat(np.arange(ntri), nwin)
                pos = np.tile(np.arange(nwin), ntri)
                shared = rng.rand(len(y)) < err_bias
                for m in range(n_members):
                    wrong = shared if correlated else (rng.rand(len(y)) < err_bias)
                    cls = np.where(wrong, (y + 1 + rng.randint(0, 2, len(y))) % 3, y)
                    P = np.full((len(y), 3), 0.2) + rng.rand(len(y), 3) * 0.02
                    P[np.arange(len(y)), cls] += 1.0
                    P /= P.sum(1, keepdims=True)
                    mem[m][(f"VP{i:03d}", s)] = {"y": y, "trial": tri, "pos": pos, "P": P}
        return mem

    labs = ["m0", "m1", "m2"]
    # 1) diverse members must beat correlated ones at the same single-member accuracy
    div = make(0.45, 3, correlated=False)
    cor = make(0.45, 3, correlated=True)
    kd, kc = align(div, labs), align(cor, labs)
    _, wd1, _, _ = score(div, kd, [0], "arith")
    _, wd3, _, _ = score(div, kd, [0, 1, 2], "arith")
    _, wc1, _, _ = score(cor, kc, [0], "arith")
    _, wc3, _, _ = score(cor, kc, [0, 1, 2], "arith")
    print(f"[diverse]    single {wd1.mean():.3f} -> ensemble of 3 {wd3.mean():.3f}  "
          f"(+{wd3.mean()-wd1.mean():.3f})")
    print(f"[correlated] single {wc1.mean():.3f} -> ensemble of 3 {wc3.mean():.3f}  "
          f"(+{wc3.mean()-wc1.mean():.3f})")
    assert wd3.mean() - wd1.mean() > (wc3.mean() - wc1.mean()) + 0.05, "diversity must pay more"
    assert abs(wd1.mean() - wc1.mean()) < 0.05, "the two cases must start from the same accuracy"

    # 2) identical members: the ensemble must equal the member exactly, and both block orders agree
    same = [div[0], div[0], div[0]]
    ks = align(same, labs)
    _, w1, b1, bb1 = score(same, ks, [0], "arith")
    _, w3, b3, bb3 = score(same, ks, [0, 1, 2], "arith")
    assert np.allclose(w1, w3) and np.allclose(b1, b3) and np.allclose(bb1, bb3)
    print("identical members -> the ensemble reproduces the member exactly")

    # 3) misalignment is refused, not silently combined
    bad = [div[0], {k: dict(v) for k, v in div[1].items()}]
    k0 = sorted(bad[1])[0]
    bad[1][k0] = dict(bad[1][k0]); bad[1][k0]["y"] = bad[1][k0]["y"][::-1].copy()
    try:
        align(bad, ["a", "b"]); raise SystemExit("misaligned labels were NOT refused")
    except SystemExit as e:
        assert "disagrees" in str(e), e
    short = [div[0], {k: dict(v) for k, v in div[1].items()}]
    short[1].pop(sorted(short[1])[0])
    try:
        align(short, ["a", "b"]); raise SystemExit("a missing fold was NOT refused")
    except SystemExit as e:
        assert "different folds" in str(e), e
    print("a relabelled fold and a missing fold are both refused")

    # 4) ARITHMETIC: the two block orders are the same number by algebra -- assert it, because a
    #    difference here would mean the pooling code is wrong, not that the order matters
    # a HARDER fixture: with err_bias 0.45 the block vote is already perfect and both orders read
    # 1.0000, which would make the comparison vacuous rather than informative
    hard = make(0.66, 3, correlated=False)
    kh = align(hard, labs)
    _, _, bwf, bbf = score(hard, kh, [0, 1, 2], "arith")
    assert 0.1 < bwf.mean() < 0.95, f"fixture is degenerate at {bwf.mean():.3f}"
    assert np.allclose(bwf, bbf, atol=1e-12), (bwf.mean(), bbf.mean())
    print(f"arith: the two block orders are identical by algebra, verified ({bwf.mean():.4f})")
    # GEOMETRIC: they do NOT commute, so there the question is real
    _, _, gwf, gbf = score(hard, kh, [0, 1, 2], "geo")
    assert not np.allclose(gwf, gbf), "geometric combination must NOT commute with block averaging"
    print(f"geo:   windows-first {gwf.mean():.4f} vs blocks-first {gbf.mean():.4f} -- a real question")
    print("SELFTEST PASSED")


if __name__ == "__main__":
    main()
