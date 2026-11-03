"""
xai_pair_2v3.py -- H4 as a PAIRWISE test: where do the trunk and the Riemannian member separate the load levels?
For every pair of levels (0 vs 2, 0 vs 3, 2 vs 3) only the windows (or blocks) of those two classes are scored; the decision is the larger of the two class posteriors
(the third class is ignored) and the score is BALANCED accuracy (chance 0.5). Components: trunk ensemble (P_deep), member (P_riem), the 0.5/0.5 blend (P_final) and
the trunk's own heads (EEG-only, fusion, fNIRS-only). Fold -> subject mean -> n = 26 subjects.
Input: <ensemble_eval out dir>/posteriors/*.npz (run ensemble_eval.py ... --riem-branch member --riem-head-fit post --save-posteriors); seeds are averaged first (ensemble).
Statistics: bootstrap 95% CI over subjects; one-sided Wilcoxon against 0.5 per component; paired two-sided Wilcoxon member-vs-trunk, blend-vs-trunk, blend-vs-member per pair,
Holm over the 9 comparisons of a level (window and block reported separately; window is primary); interaction: (member - trunk) on 2v3 against (member - trunk) on 0v2 and on 0v3.
PRE-STATED reading of H4 ("2v3 stays in the trunk, the member only solves 0-back"), fixed before the run:
  H4 strong  : trunk > member on 2v3 (Holm p < 0.05)  AND  member > trunk on 0v2 and on 0v3 (Holm p < 0.05)  AND  blend - trunk on 2v3 <= +1.0 point.
  H4 partial : (member - trunk) is larger on 0v2 and 0v3 than on 2v3 (interaction p < 0.05) without the strong form  ->  "complementary specialisation".
  otherwise  : not supported.
  python xai_pair_2v3.py --posteriors-dir "...\\ensemble_eval_cf64a71127_seeds-42-999-2024\\posteriors" --out-dir "...\\pair_2v3"
"""
import argparse, glob, json, os, sys, tempfile
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from cmp_common import holm, boot_ci

PAIRS = {"0v2": (0, 1), "0v3": (0, 2), "2v3": (1, 2)}
COMP = {"trunk": "deep", "member": "riem", "blend": "final", "trunk EEG head": "eeg", "trunk fusion head": "fusion", "trunk fNIRS head": "fnirs"}
COLORS = {"trunk": "#0072B2", "member": "#D55E00", "blend": "#009E73", "trunk EEG head": "#CC79A7", "trunk fusion head": "#E69F00", "trunk fNIRS head": "#56B4E9"}


def load_fold(f):
    z = np.load(f); out = {"y": z["y"], "tid": z["trial_ids"]}
    for name, key in COMP.items():
        if f"P_{key}" not in z.files: raise SystemExit(f"{f} lacks P_{key} (keys {sorted(z.files)}); rerun ensemble_eval with --riem-branch member --save-posteriors")
        P = z[f"P_{key}"].astype(np.float64); P = P / np.clip(P.sum(-1, keepdims=True), 1e-12, None); out[name] = P.mean(0)
    return out


def balanced(score, truth_is_b):
    pa, pb = truth_is_b.sum(), (~truth_is_b).sum()
    if pa == 0 or pb == 0: return np.nan
    pred_b = score > 0
    return 0.5 * (pred_b[truth_is_b].mean() + (~pred_b)[~truth_is_b].mean())


def fold_scores(d):
    res = {}; y, tid = d["y"], d["tid"]
    for pair, (a, b) in PAIRS.items():
        for name in COMP:
            P = d[name]; m = (y == a) | (y == b)
            res[("window", pair, name)] = balanced(P[m, b] - P[m, a], y[m] == b)
            ub = [t for t in np.unique(tid) if y[tid == t][0] in (a, b)]
            if ub:
                Pb = np.array([P[tid == t].mean(0) for t in ub]); yb = np.array([y[tid == t][0] for t in ub])
                res[("block", pair, name)] = balanced(Pb[:, b] - Pb[:, a], yb == b)
    return res


def collect(pdir):
    rows = []
    for f in sorted(glob.glob(os.path.join(pdir, "*_session*.npz"))):
        sid, ses = os.path.basename(f)[:-4].split("_session"); sc = fold_scores(load_fold(f))
        for (lvl, pair, name), v in sc.items(): rows.append(dict(subject=sid, session=int(ses), level=lvl, pair=pair, component=name, acc=v))
    if not rows: sys.exit(f"no *_session*.npz in {pdir}")
    D = pd.DataFrame(rows).groupby(["subject", "level", "pair", "component"], as_index=False)["acc"].mean()
    return D


def analyse(D, verbose=True):
    out = {}; verdicts = {}
    for lvl in ("window", "block"):
        W = D[D.level == lvl].pivot_table(index="subject", columns=["pair", "component"], values="acc")
        n = len(W); tests = []
        if verbose: print(f"\n=== {lvl}-level balanced accuracy (%), n = {n} subjects; chance 50 ===")
        for pair in PAIRS:
            if verbose: print(f"  {pair}: " + " | ".join(f"{c} {100*W[(pair,c)].mean():5.1f}" for c in COMP))
            for c in COMP:
                v = W[(pair, c)].dropna().values; ci = boot_ci(v); p = wilcoxon(v - 0.5, alternative="greater").pvalue if np.any(v != 0.5) else 1.0
                out[(lvl, pair, c)] = dict(mean=float(v.mean()), ci=ci, p_above_chance=float(p))
            for a_, b_ in (("member", "trunk"), ("blend", "trunk"), ("blend", "member")):
                d = (W[(pair, a_)] - W[(pair, b_)]).dropna().values; p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
                tests.append(dict(pair=pair, contrast=f"{a_} - {b_}", diff=float(d.mean()), ci=boot_ci(d), p_raw=float(p), better=int((d > 0).sum()), n=len(d)))
        adj = holm([t["p_raw"] for t in tests])
        for t, a in zip(tests, adj): t["p_holm"] = float(a)
        T = pd.DataFrame(tests); out[(lvl, "tests")] = T
        if verbose:
            print(f"  paired comparisons ({lvl}; Holm over {len(T)}):")
            for _, r in T.iterrows(): print(f"    {r.pair} {r.contrast:16s} {100*r['diff']:+6.2f} pp [{100*r.ci[0]:+.2f}, {100*r.ci[1]:+.2f}]  better in {r.better}/{r.n}  p={r.p_raw:.3g}  Holm {r.p_holm:.3g}")
        dm = {p: (W[(p, "member")] - W[(p, "trunk")]).values for p in PAIRS}
        inter = {k: float(wilcoxon(dm[k] - dm["2v3"]).pvalue) if np.any(dm[k] != dm["2v3"]) else 1.0 for k in ("0v2", "0v3")}
        inter_d = {k: float((dm[k] - dm["2v3"]).mean()) for k in ("0v2", "0v3")}
        if verbose: print("  interaction (member - trunk on the 0-pair minus on 2v3): " + ", ".join(f"{k}: {100*inter_d[k]:+.2f} pp (p={inter[k]:.3g})" for k in inter))
        g = lambda pair, con: T[(T.pair == pair) & (T.contrast == con)].iloc[0]
        strong = (g("2v3", "member - trunk")["diff"] < 0 and g("2v3", "member - trunk")["p_holm"] < 0.05 and all(g(p, "member - trunk")["diff"] > 0 and g(p, "member - trunk")["p_holm"] < 0.05 for p in ("0v2", "0v3"))
                  and g("2v3", "blend - trunk")["diff"] <= 0.01)
        partial = (not strong) and all(inter_d[k] > 0 and inter[k] < 0.05 for k in inter)
        verdicts[lvl] = "H4 STRONG" if strong else ("H4 PARTIAL (complementary specialisation)" if partial else "H4 NOT SUPPORTED")
        if verbose: print(f"  verdict ({lvl}, pre-stated rule): {verdicts[lvl]}")
    return out, verdicts


def figure(D, out_dir, level="window"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    W = D[D.level == level].pivot_table(index="subject", columns=["pair", "component"], values="acc"); names = list(COMP)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True); rng = np.random.RandomState(0)
    for ax, pair in zip(axs, PAIRS):
        for i, c in enumerate(names):
            v = W[(pair, c)].dropna().values; lo, hi = boot_ci(v)
            ax.bar(i, 100 * v.mean(), color=COLORS[c], alpha=0.85, width=0.7); ax.errorbar(i, 100 * v.mean(), yerr=[[100 * (v.mean() - lo)], [100 * (hi - v.mean())]], color="k", capsize=3, lw=1)
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), 100 * v, s=6, color="k", alpha=0.35, zorder=3)
        ax.axhline(50, color="k", ls="--", lw=0.8); ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
        ax.set_title(f"{pair.replace('v', ' vs ')}-back", fontsize=9); ax.set_ylim(30, 100)
    axs[0].set_ylabel(f"balanced accuracy (%), {level} level"); fig.suptitle(f"Pairwise discrimination by component (n={len(W)} subjects, mean ± 95% bootstrap CI; dashed = chance)", fontsize=9)
    fig.tight_layout(); os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"): fig.savefig(os.path.join(out_dir, f"fig_pair_2v3_{level}.{ext}"), dpi=300)
    plt.close(fig)


def self_test():
    rng = np.random.RandomState(0); d = tempfile.mkdtemp(); K = 2
    M_member = np.array([[2, 0, 0], [-1, .2, -.2], [-1, -.2, .2]], float)       # strong on "0 vs the rest", weak on 2 vs 3
    M_trunk = np.array([[.3, -.3, -.3], [-.2, 1, -1], [-.2, -1, 1]], float)     # moderate on 0, strong on 2 vs 3
    def post(y, M):
        z = rng.randn(len(y), 3) + M[y]; e = np.exp(z - z.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
    os.makedirs(os.path.join(d, "posteriors"))
    for s in range(14):
        for ses in (1, 2, 3):
            tid = np.repeat(np.arange(12), 30); y = np.repeat(rng.permutation(np.tile([0, 1, 2], 4)), 30)
            Pdeep = np.stack([post(y, M_trunk) for _ in range(K)]); Priem = np.stack([post(y, M_member) for _ in range(K)]); Pfin = 0.5 * Pdeep + 0.5 * Priem
            np.savez(os.path.join(d, "posteriors", f"VP{s+1:03d}_session{ses}.npz"), seeds=np.arange(K), y=y, trial_ids=tid, pos_in_trial=np.zeros(360, int),
                     P_deep=Pdeep, P_riem=Priem, P_final=Pfin, P_eeg=Pdeep, P_fusion=Pdeep, P_fnirs=Pdeep)
    D = collect(os.path.join(d, "posteriors")); out, v = analyse(D); figure(D, d)
    assert out[("window", "0v2", "member")]["mean"] > out[("window", "0v2", "trunk")]["mean"] and out[("window", "2v3", "trunk")]["mean"] > out[("window", "2v3", "member")]["mean"]
    assert v["window"] == "H4 STRONG", v
    assert os.path.isfile(os.path.join(d, "fig_pair_2v3_window.png"))
    # a case where both components are equally good everywhere must NOT be called supported
    for fz in os.listdir(os.path.join(d, "posteriors")):
        z = dict(np.load(os.path.join(d, "posteriors", fz))); z["P_riem"] = z["P_deep"]; z["P_final"] = z["P_deep"]; np.savez(os.path.join(d, "posteriors", fz), **z)
    _, v2 = analyse(collect(os.path.join(d, "posteriors")), verbose=False); assert v2["window"] == "H4 NOT SUPPORTED", v2
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--posteriors-dir"); ap.add_argument("--out-dir", default="pair_2v3_out"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.posteriors_dir: sys.exit("need --posteriors-dir")
    D = collect(a.posteriors_dir); os.makedirs(a.out_dir, exist_ok=True); D.to_csv(os.path.join(a.out_dir, "pair_2v3_per_subject.csv"), index=False)
    out, verdicts = analyse(D)
    pd.concat([out[(l, "tests")].assign(level=l) for l in ("window", "block")]).to_csv(os.path.join(a.out_dir, "pair_2v3_tests.csv"), index=False)
    json.dump({"verdict": verdicts}, open(os.path.join(a.out_dir, "pair_2v3_verdict.json"), "w"), indent=2)
    figure(D, a.out_dir, "window"); figure(D, a.out_dir, "block"); print("\nwritten ->", a.out_dir)


if __name__ == "__main__":
    main()
