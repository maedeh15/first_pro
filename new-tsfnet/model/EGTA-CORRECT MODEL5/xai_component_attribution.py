"""
xai_component_attribution.py -- WHO decides in the ~0.67 system (deep trunk + linear member), and where does each one help?

The final posterior is P = (1-w) * P_trunk + w * P_member with w = 0.5. Because it is a fixed average of two probability vectors, the
split of every decision between the two components is exact (no post-hoc approximation). From saved posteriors only (CPU, seconds):

  1. correctness table      per subject: both right / only the trunk / only the member / both wrong; the oracle ceiling (either right)
                            and the complementarity (phi, Yule's Q) -- how much room a blend has at all;
  2. rescue and harm        windows the blend gets right that the trunk got wrong (rescued by the member) and the reverse, per true class,
                            net per subject;
  3. who wins disagreements where the two argmaxes differ, how often the blended decision follows each, and the blended accuracy there;
  4. contribution share     of the predicted class' probability, the share carried by each component (0.5*P_k(c*) / P(c*));
  5. per-class recall       of trunk, member and blend, and the same correctness table at BLOCK level (mean-posterior pooling);
  6. subject structure      are the subjects hard for the trunk the same ones hard for the member? (Spearman across subjects.)

Inputs: either one posteriors/ folder of a member-branch run (ensemble_eval.py --riem-branch member --save-posteriors; P_deep, P_riem),
or --deep-dir and --member-dir (each with P_final), e.g. the plain baseline and late_fusion_riemann.py --save-member. Seeds inside a folder
are averaged first. Windows of one block overlap by 90%, so window-level counts are descriptive; every test below is across SUBJECTS.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon, spearmanr

CLASSES = ("0-back", "2-back", "3-back")


def load(d, key):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        if f"P_{key}" not in z:
            raise SystemExit(f"{f} has no P_{key} (keys: {sorted(z.files)})")
        sid, ses = os.path.basename(f)[:-4].split("_session")
        P = z[f"P_{key}"].astype(np.float64)
        P = (P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)).mean(0) if P.ndim == 3 else P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)
        out[(sid, int(ses))] = (P, z["y"], z["trial_ids"])
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def phi_q(a, b):
    """a, b boolean correctness vectors -> (phi, Yule's Q)"""
    n11, n10, n01, n00 = (a & b).sum(), (a & ~b).sum(), (~a & b).sum(), (~a & ~b).sum()
    den = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    phi = (n11 * n00 - n10 * n01) / den if den else np.nan
    q = (n11 * n00 - n10 * n01) / float(n11 * n00 + n10 * n01) if (n11 * n00 + n10 * n01) else np.nan
    return float(phi), float(q)


def fold_stats(Pd, Pm, y, tid, w):
    Pf = (1 - w) * Pd + w * Pm
    out = {}
    for lvl in ("win", "blk"):
        if lvl == "win":
            d, m, f, yy = Pd, Pm, Pf, y
        else:
            ub = np.unique(tid)
            d = np.array([Pd[tid == t].mean(0) for t in ub]); m = np.array([Pm[tid == t].mean(0) for t in ub])
            f = np.array([Pf[tid == t].mean(0) for t in ub]); yy = np.array([y[tid == t][0] for t in ub])
        cd, cm, cf = d.argmax(1) == yy, m.argmax(1) == yy, f.argmax(1) == yy
        n = len(yy)
        out[lvl] = dict(n=n, acc_d=cd.mean(), acc_m=cm.mean(), acc_f=cf.mean(),
                        both=(cd & cm).mean(), only_d=(cd & ~cm).mean(), only_m=(~cd & cm).mean(), neither=(~cd & ~cm).mean(),
                        rescued=(cf & ~cd).mean(), harmed=(cd & ~cf).mean(),
                        recall_d=np.array([cd[yy == c].mean() if (yy == c).any() else np.nan for c in range(3)]),
                        recall_m=np.array([cm[yy == c].mean() if (yy == c).any() else np.nan for c in range(3)]),
                        recall_f=np.array([cf[yy == c].mean() if (yy == c).any() else np.nan for c in range(3)]),
                        rescued_c=np.array([(cf & ~cd)[yy == c].sum() for c in range(3)]),
                        harmed_c=np.array([(cd & ~cf)[yy == c].sum() for c in range(3)]),
                        counts=np.array([(yy == c).sum() for c in range(3)]),
                        phi=phi_q(cd, cm)[0], q=phi_q(cd, cm)[1])
        if lvl == "win":
            dis = d.argmax(1) != m.argmax(1)
            nd = max(int(dis.sum()), 1)
            cs = f.argmax(1)
            cstar = f.argmax(1)
            out[lvl].update(dis=dis.mean(), follow_d=(cs[dis] == d.argmax(1)[dis]).sum() / nd, follow_m=(cs[dis] == m.argmax(1)[dis]).sum() / nd,
                            acc_f_dis=(cs[dis] == yy[dis]).sum() / nd,
                            share_d=float(np.mean((1 - w) * d[np.arange(n), cstar] / np.clip(f[np.arange(n), cstar], 1e-12, None))),
                            share_d_right=float(np.mean(((1 - w) * d[np.arange(n), cstar] / np.clip(f[np.arange(n), cstar], 1e-12, None))[cf])) if cf.any() else np.nan,
                            share_d_wrong=float(np.mean(((1 - w) * d[np.arange(n), cstar] / np.clip(f[np.arange(n), cstar], 1e-12, None))[~cf])) if (~cf).any() else np.nan)
    return out


def run(deep, member, w):
    if set(deep) != set(member):
        raise SystemExit(f"the folders cover different folds, e.g. {sorted(set(deep) ^ set(member))[:3]}")
    per = {}
    for k in sorted(deep):
        Pd, yd, td = deep[k]; Pm, ym, tm = member[k]
        if not (np.array_equal(yd, ym) and np.array_equal(td, tm)):
            raise SystemExit(f"fold {k}: labels or trial ids differ between the components -- refusing")
        per.setdefault(k[0], []).append(fold_stats(Pd, Pm, yd, td, w))
    return per


def agg(per, lvl, key):
    subs = sorted(per)
    return np.array([np.nanmean([f[lvl][key] for f in per[s]], axis=0) for s in subs])


def paired(a, b):
    d = a - b
    return d.mean(), int((d > 0).sum()), len(d), (1.0 if not np.any(d != 0) else float(wilcoxon(a, b).pvalue))


def report(per, w):
    subs = sorted(per)
    print(f"{len(subs)} subjects, blend weight on the member w={w}\n")
    for lvl, nm in (("win", "WINDOW"), ("blk", "BLOCK (mean-posterior pooling)")):
        print(f"=== {nm} level")
        A = {k: agg(per, lvl, k) for k in ("acc_d", "acc_m", "acc_f", "both", "only_d", "only_m", "neither", "rescued", "harmed", "phi", "q")}
        print(f"  accuracy   trunk {A['acc_d'].mean():.4f} | member {A['acc_m'].mean():.4f} | blend {A['acc_f'].mean():.4f}")
        print(f"  correctness table (share of {lvl}s): both right {A['both'].mean():.3f} | only trunk {A['only_d'].mean():.3f} | only member {A['only_m'].mean():.3f} | both wrong {A['neither'].mean():.3f}")
        print(f"  oracle ceiling (either right) {1 - A['neither'].mean():.4f}; blend captures {100 * (A['acc_f'].mean() - max(A['acc_d'].mean(), A['acc_m'].mean())) / max(1e-9, (1 - A['neither'].mean()) - max(A['acc_d'].mean(), A['acc_m'].mean())):.0f}% of the room above the better single component")
        print(f"  complementarity: phi {np.nanmean(A['phi']):.3f}, Yule's Q {np.nanmean(A['q']):.3f}  (phi near 0 = independent errors = most to gain)")
        m, wn, n, p = paired(A["only_m"], A["only_d"])
        print(f"  'only member right' minus 'only trunk right': {m:+.3f}  larger in {wn}/{n} subjects  p={p:.4f}")
        print(f"  rescued by the member {A['rescued'].mean():.3f} | harmed {A['harmed'].mean():.3f} | net {(A['rescued'] - A['harmed']).mean():+.3f}")
        rc = np.array([f[lvl]["rescued_c"] for s in subs for f in per[s]]).sum(0); hc = np.array([f[lvl]["harmed_c"] for s in subs for f in per[s]]).sum(0)
        cn = np.array([f[lvl]["counts"] for s in subs for f in per[s]]).sum(0)
        print("  per true class (pooled counts):  " + "  ".join(f"{CLASSES[c]}: rescued {rc[c]}, harmed {hc[c]}, net {rc[c] - hc[c]:+d} of {cn[c]}" for c in range(3)))
        R = {k: agg(per, lvl, k) for k in ("recall_d", "recall_m", "recall_f")}
        print("  recall      " + "  ".join(f"{CLASSES[c]}: trunk {R['recall_d'][:, c].mean():.3f} member {R['recall_m'][:, c].mean():.3f} blend {R['recall_f'][:, c].mean():.3f}" for c in range(3)))
        if lvl == "win":
            B = {k: agg(per, lvl, k) for k in ("dis", "follow_d", "follow_m", "acc_f_dis", "share_d", "share_d_right", "share_d_wrong")}
            print(f"  windows where trunk and member disagree: {B['dis'].mean():.3f}; there the blend follows the trunk {B['follow_d'].mean():.3f}, the member {B['follow_m'].mean():.3f} (blend accuracy there {B['acc_f_dis'].mean():.3f})")
            print(f"  share of the predicted class' probability carried by the trunk: {B['share_d'].mean():.3f} overall | when the blend is right {np.nanmean(B['share_d_right']):.3f} | when wrong {np.nanmean(B['share_d_wrong']):.3f}")
        print()
    ad, am = agg(per, "win", "acc_d"), agg(per, "win", "acc_m")
    rho, p = spearmanr(ad, am)
    print(f"=== subject structure: Spearman between a subject's trunk accuracy and member accuracy rho={rho:+.3f} (p={p:.4f})")
    print("    positive = the same subjects are hard for both (shared, data-determined failures); near 0 = different failure sets.")
    print("\nWindow counts overlap by 90% inside a block, so only the across-subject tests carry inferential weight.")


def self_test():
    rng = np.random.RandomState(0)
    nb, nw = 9, 20
    y = np.repeat(np.arange(nb) % 3, nw); tid = np.repeat(np.arange(nb), nw)
    n = len(y)
    def post(good_classes, acc_good, acc_bad):
        P = np.full((n, 3), 0.1)
        for c in range(3):
            m = y == c; ok = rng.rand(m.sum()) < (acc_good if c in good_classes else acc_bad)
            cls = np.where(ok, c, (c + 1 + rng.randint(0, 2, m.sum())) % 3)
            idx = np.flatnonzero(m)
            # confident when right, hesitant when wrong (as in real posteriors); otherwise a blend of two equally peaked
            # components has nothing to arbitrate disagreements with and cannot beat either of them
            P[idx[ok], c] += 0.8
            P[idx[~ok], cls[~ok]] += 0.42
            P[idx[~ok], c] += 0.30
        return P / P.sum(1, keepdims=True)
    deep, mem = {}, {}
    for s in range(8):
        for t in (1, 2, 3):
            deep[(f"VP{s:03d}", t)] = (post([1, 2], 0.8, 0.45), y, tid)    # trunk strong on 2- and 3-back
            mem[(f"VP{s:03d}", t)] = (post([0], 0.9, 0.45), y, tid)        # member strong on 0-back
    per = run(deep, mem, 0.5)
    A = lambda k: agg(per, "win", k)
    assert A("acc_f").mean() > max(A("acc_d").mean(), A("acc_m").mean()), "complementary components must beat both"
    assert abs(A("both").mean() + A("only_d").mean() + A("only_m").mean() + A("neither").mean() - 1.0) < 1e-9
    rc = np.array([f["win"]["rescued_c"] for s in per for f in per[s]]).sum(0)
    assert rc[0] > rc[1] and rc[0] > rc[2], "the member (strong on 0-back) must rescue mostly 0-back windows"
    # identical components: nothing to rescue or harm, no disagreement
    same = run(deep, deep, 0.5)
    assert agg(same, "win", "rescued").max() == 0 and agg(same, "win", "harmed").max() == 0 and agg(same, "win", "dis").max() == 0
    # shares sum to one across components
    sd = agg(per, "win", "share_d").mean()
    assert 0.0 < sd < 1.0
    # mismatched labels are refused
    bad = dict(mem); k0 = sorted(bad)[0]; bad[k0] = (bad[k0][0], y[::-1].copy(), tid)
    try:
        run(deep, bad, 0.5); raise SystemExit("mismatched labels were NOT refused")
    except SystemExit as e:
        assert "refusing" in str(e)
    print(f"[1] complementary components: blend {A('acc_f').mean():.3f} > trunk {A('acc_d').mean():.3f}, member {A('acc_m').mean():.3f}; table sums to 1")
    print(f"[2] rescues land on the member's strong class: rescued per class {rc}")
    print("[3] identical components -> no rescue, no harm, no disagreement; mismatched labels refused")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated", help="posteriors/ folder with P_deep and P_riem (a --riem-branch member run)")
    ap.add_argument("--deep-dir", help="posteriors/ of the trunk (P_final)")
    ap.add_argument("--member-dir", help="posteriors/ of the member (P_final)")
    ap.add_argument("--w", type=float, default=0.5)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if a.integrated:
        deep, member = load(a.integrated, "deep"), load(a.integrated, "riem")
    elif a.deep_dir and a.member_dir:
        deep, member = load(a.deep_dir, "final"), load(a.member_dir, "final")
    else:
        sys.exit("give --integrated, or --deep-dir and --member-dir")
    report(run(deep, member, a.w), a.w)


if __name__ == "__main__":
    main()
