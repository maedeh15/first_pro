"""
confidence_check.py -- is the deep trunk much more CONFIDENT than the linear member, so that the 0.5/0.5 blend is really a
trunk-dominated blend?  Inference-only, CPU, from saved posteriors.

Why. The trunk reaches training accuracy 1.0 within two epochs and its posteriors are probably peaked; the member is a
logistic regression with C=0.1 and is softer. In an arithmetic average the more confident component wins disagreements, so the
nominal weight 0.5 may not be the effective one. This script measures it; it does not search for a better weight.

Per fold (window level) and then averaged per subject:
  confidence   mean max-probability, mean entropy (nats)
  calibration  mean NLL of the true class
  blend        among windows where the two argmaxes DISAGREE: how often the 0.5/0.5 fused argmax follows the trunk vs the
               member, and the fused accuracy there
  exploratory  ONE label-free rule, no search: temper the trunk (p^(1/T), renormalised) so its mean max-probability on the
               fold's test windows equals the member's; fuse 0.5/0.5. Uses unlabeled test posteriors only (transductive, like
               AdaBN); a descriptive probe, not a model change.

Reading. If the trunk is NOT clearly more confident (mean max-prob within ~0.05 of the member's) and the blend follows both about
equally, the confidence hypothesis is dead and --label-smoothing as a calibration fix has no rationale.
"""
import argparse, glob, os, sys
import numpy as np
from scipy.stats import wilcoxon

EPS = 1e-12


def load(d, key="final"):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_session*.npz"))):
        z = np.load(f)
        sid, ses = os.path.basename(f)[:-4].split("_session")
        if f"P_{key}" not in z:
            raise SystemExit(f"{f} has no P_{key} (keys: {sorted(z.files)})")
        P = z[f"P_{key}"].astype(np.float64)
        P = P / np.clip(P.sum(-1, keepdims=True), EPS, None)
        out[(sid, int(ses))] = (P, z["y"])
    if not out:
        raise SystemExit(f"no *_session*.npz in {d}")
    return out


def temper(P, T):
    Q = np.power(np.clip(P, EPS, None), 1.0 / T)
    return Q / Q.sum(-1, keepdims=True)


def match_temperature(Pd, target_conf):
    """T in [0.05, 50] with mean max-prob of temper(Pd, T) == target_conf (monotone in T); clipped at the ends."""
    lo, hi = 0.05, 50.0
    f = lambda T: temper(Pd, T).max(1).mean()
    if f(lo) < target_conf:
        return lo
    if f(hi) > target_conf:
        return hi
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if f(mid) > target_conf else (lo, mid)
    return float(np.sqrt(lo * hi))


def fold_stats(Pd, Pm, y):
    nll = lambda P: float(-np.log(np.clip(P[np.arange(len(y)), y], EPS, None)).mean())
    ent = lambda P: float(-(P * np.log(np.clip(P, EPS, None))).sum(1).mean())
    fused = 0.5 * Pd + 0.5 * Pm
    dis = Pd.argmax(1) != Pm.argmax(1)
    T = match_temperature(Pd, Pm.max(1).mean())
    fusedT = 0.5 * temper(Pd, T) + 0.5 * Pm
    acc = lambda P: float((P.argmax(1) == y).mean())
    nd = max(int(dis.sum()), 1)
    return dict(conf_d=Pd.max(1).mean(), conf_m=Pm.max(1).mean(), ent_d=ent(Pd), ent_m=ent(Pm), nll_d=nll(Pd), nll_m=nll(Pm),
                acc_d=acc(Pd), acc_m=acc(Pm), acc_f=acc(fused), acc_fT=acc(fusedT), T=T,
                dis=float(dis.mean()),
                follow_d=float((fused.argmax(1)[dis] == Pd.argmax(1)[dis]).sum() / nd),
                follow_m=float((fused.argmax(1)[dis] == Pm.argmax(1)[dis]).sum() / nd),
                acc_f_dis=float((fused.argmax(1)[dis] == y[dis]).sum() / nd))


def run(deep_dir, member_dir, seed_index=0, deep_key="final", member_key="final"):
    D, M = load(deep_dir, deep_key), load(member_dir, member_key)
    if set(D) != set(M):
        raise SystemExit(f"the two folders cover different folds, e.g. {sorted(set(D) ^ set(M))[:3]}")
    rows = {}
    for k in sorted(D):
        Pd_all, yd = D[k]; Pm_all, ym = M[k]
        if not np.array_equal(yd, ym):
            raise SystemExit(f"fold {k}: labels differ between the folders -- refusing")
        Pd = Pd_all if Pd_all.ndim == 2 else Pd_all[seed_index]
        Pm = Pm_all if Pm_all.ndim == 2 else Pm_all[0]
        if Pd.shape != Pm.shape:
            raise SystemExit(f"fold {k}: window counts differ {Pd.shape} vs {Pm.shape}")
        rows.setdefault(k[0], []).append(fold_stats(Pd, Pm, yd))
    subs = sorted(rows)
    agg = {key: np.array([np.mean([r[key] for r in rows[s]]) for s in subs]) for key in rows[subs[0]][0]}
    return subs, agg


def report(subs, a):
    print(f"{len(subs)} subjects (mean over each subject's folds)\n")
    print(f"  {'':34} {'trunk':>8} {'member':>8}")
    print(f"  {'mean max-probability':34} {a['conf_d'].mean():8.3f} {a['conf_m'].mean():8.3f}")
    print(f"  {'mean entropy (nats)':34} {a['ent_d'].mean():8.3f} {a['ent_m'].mean():8.3f}")
    print(f"  {'mean NLL of the true class':34} {a['nll_d'].mean():8.3f} {a['nll_m'].mean():8.3f}")
    print(f"  {'window accuracy':34} {a['acc_d'].mean():8.4f} {a['acc_m'].mean():8.4f}")
    d = a['conf_d'] - a['conf_m']
    p = 1.0 if not np.any(d != 0) else float(wilcoxon(a['conf_d'], a['conf_m']).pvalue)
    print(f"\n  trunk minus member confidence: {d.mean():+.3f}  trunk more confident in {int((d > 0).sum())}/{len(d)} subjects  p={p:.4f}")
    print(f"\n  windows where the two disagree: {a['dis'].mean():.3f} of all windows")
    print(f"  there, the 0.5/0.5 fused decision follows the trunk {a['follow_d'].mean():.3f}, the member {a['follow_m'].mean():.3f}"
          f"  (fused accuracy on those windows {a['acc_f_dis'].mean():.3f})")
    print(f"\n  EXPLORATORY, label-free, one rule, no search: trunk tempered to the member's mean confidence "
          f"(median T {np.median(a['T']):.2f})")
    dd = a['acc_fT'] - a['acc_f']
    p = 1.0 if not np.any(dd != 0) else float(wilcoxon(a['acc_f'], a['acc_fT']).pvalue)
    print(f"    fused window accuracy {a['acc_f'].mean():.4f} -> {a['acc_fT'].mean():.4f}  ({dd.mean():+.4f}, better "
          f"{int((dd > 0).sum())}/{len(dd)}, p={p:.4f})  descriptive only; transductive")
    print("\n  Reading: a confidence gap below ~0.05 with the blend following both about equally means the hypothesis is dead.")


def self_test():
    rng = np.random.RandomState(0)
    # temperature matching: monotone and exact on a synthetic peaked posterior
    Pd = rng.dirichlet(np.ones(3) * 0.2, size=500)
    target = 0.6
    T = match_temperature(Pd, target)
    got = temper(Pd, T).max(1).mean()
    assert abs(got - target) < 1e-4 and T > 1.0, (T, got)
    assert np.allclose(temper(Pd, 1.0), Pd, atol=1e-9)
    print(f"[1] temperature matching is exact ({got:.4f} vs target {target}) and >1 for a peaked trunk -> OK")
    # a peaky trunk vs a soft member: the script must see it, and the fused decision must follow the trunk
    n = 600; y = rng.randint(0, 3, n)
    def post(acc, sharp):
        cls = np.where(rng.rand(n) < acc, y, (y + 1 + rng.randint(0, 2, n)) % 3)
        L = rng.randn(n, 3) * 0.3; L[np.arange(n), cls] += sharp
        e = np.exp(L - L.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    s = fold_stats(post(0.6, 6.0), post(0.6, 0.8), y)
    assert s["conf_d"] > s["conf_m"] + 0.2 and s["follow_d"] > s["follow_m"] + 0.2, s
    print(f"[2] peaky trunk vs soft member detected (conf {s['conf_d']:.2f} vs {s['conf_m']:.2f}); blend follows the trunk "
          f"{s['follow_d']:.2f} vs {s['follow_m']:.2f} -> OK")
    # equal confidence: no gap, the blend splits roughly evenly
    s2 = fold_stats(post(0.6, 2.0), post(0.6, 2.0), y)
    assert abs(s2["conf_d"] - s2["conf_m"]) < 0.05 and abs(s2["follow_d"] - s2["follow_m"]) < 0.25, s2
    print("[3] equally confident components -> no gap, blend splits evenly -> OK")
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", help="posteriors/ folder of the trunk (P_final), e.g. the seed-42 baseline")
    ap.add_argument("--member", help="posteriors/ folder of the offline member (late_fusion_riemann.py --save-member)")
    ap.add_argument("--deep-key", default="final", help="posterior key of the trunk: final (a plain run) or deep (a member-branch run)")
    ap.add_argument("--member-key", default="final", help="posterior key of the member: final (offline folder) or riem (in-network head)")
    ap.add_argument("--seed-index", type=int, default=0, help="which seed of the trunk folder to use (default the first)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not (a.deep and a.member):
        sys.exit("need --deep and --member")
    report(*run(a.deep, a.member, a.seed_index, a.deep_key, a.member_key))


if __name__ == "__main__":
    main()
