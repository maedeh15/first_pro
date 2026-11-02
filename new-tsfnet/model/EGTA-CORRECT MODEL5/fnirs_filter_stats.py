"""
fnirs_filter_stats.py -- SYMMETRIC paired test of "fNIRS filter off vs on" from per-subject CSVs (CPU, seconds).
The earlier comparison was asymmetric (one side the mean of 2 seeds, the other a single seed). Here every seed present on BOTH sides is
compared separately (paired over subjects), and the seed-averaged curves are compared too, each with the exact p, rank-biserial r,
number of subjects improved and a bootstrap CI over subjects.

  python fnirs_filter_stats.py --a-label filtered --a 42=P3_seed42.csv 2024=P3_seed2024.csv ^
                               --b-label unfiltered --b 42=P5_seed42.csv 2024=P5_seed2024.csv
Each CSV needs columns subject_id and mean_acc (run_all_subjects.py's baseline_all_subjects_*.csv). Difference reported = B minus A.
"""
import argparse, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel


def load(spec):
    out = {}
    for item in spec:
        seed, path = item.split("=", 1)
        d = pd.read_csv(path)
        if "status" in d.columns:
            d = d[d["status"] == "ok"]
        out[int(seed)] = d.set_index("subject_id")["mean_acc"].astype(float)
    return out


def paired(a, b, rng, n_boot=5000):
    ids = sorted(set(a.index) & set(b.index)); x, y = a.loc[ids].values, b.loc[ids].values; d = y - x; n = len(d)
    nz = d[d != 0]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    ranks = pd.Series(np.abs(nz)).rank().values
    rb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
    boots = [d[rng.randint(0, n, n)].mean() for _ in range(n_boot)]
    return dict(n=n, a=float(x.mean()), b=float(y.mean()), diff=float(d.mean()), ci=(float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))),
                improved=int((d > 0).sum()), worsened=int((d < 0).sum()), p_wilcoxon=p, rank_biserial=rb, p_t=float(ttest_rel(y, x).pvalue))


def show(tag, r):
    print(f"  {tag:22s} n={r['n']}  A {100*r['a']:6.2f}  B {100*r['b']:6.2f}  B-A {100*r['diff']:+6.2f}  95% CI [{100*r['ci'][0]:+.2f}, {100*r['ci'][1]:+.2f}]  "
          f"improved {r['improved']}/{r['n']} (worse {r['worsened']})  Wilcoxon p={r['p_wilcoxon']:.3g}  rank-biserial r={r['rank_biserial']:+.2f}  t-test p={r['p_t']:.3g}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", nargs="+", required=True, help="seed=csv ...  (reference side)"); ap.add_argument("--b", nargs="+", required=True, help="seed=csv ...")
    ap.add_argument("--a-label", default="A"); ap.add_argument("--b-label", default="B")
    a = ap.parse_args(argv); rng = np.random.RandomState(0)
    A, B = load(a.a), load(a.b); common = sorted(set(A) & set(B))
    print(f"A = {a.a_label}, B = {a.b_label}; seeds in A {sorted(A)}, in B {sorted(B)}; compared on common seeds {common}")
    if not common:
        sys.exit("no seed present on both sides")
    res = {}
    for s in common:
        res[s] = paired(A[s], B[s], rng); show(f"seed {s}", res[s])
    am = pd.concat([A[s] for s in common], axis=1).mean(axis=1); bm = pd.concat([B[s] for s in common], axis=1).mean(axis=1)
    res["mean"] = paired(am, bm, rng); show(f"mean of {len(common)} seeds", res["mean"])
    print("\n  Same seeds on both sides => the comparison is symmetric. Every p is over the subjects of ONE seed (or of the seed mean): with 2 seeds it says"
          "\n  the effect is not a single-seed accident, it does not estimate seed-to-seed variance. Report B-A, the CI and r, not only p.")
    return res


def self_test():
    import tempfile, os
    rng = np.random.RandomState(1); ids = [f"VP{i:03d}" for i in range(1, 27)]; d = tempfile.mkdtemp()
    base = rng.uniform(0.3, 0.6, 26); paths = {}
    for tag, shift in (("a", 0.0), ("b", 0.09)):
        for s in (42, 2024):
            p = os.path.join(d, f"{tag}{s}.csv"); pd.DataFrame({"subject_id": ids, "status": "ok", "mean_acc": base + shift + rng.normal(0, 0.02, 26)}).to_csv(p, index=False); paths[(tag, s)] = p
    r = main(["--a", f"42={paths[('a',42)]}", f"2024={paths[('a',2024)]}", "--b", f"42={paths[('b',42)]}", f"2024={paths[('b',2024)]}"])
    assert abs(r["mean"]["diff"] - 0.09) < 0.02 and r["mean"]["p_wilcoxon"] < 1e-4 and r["mean"]["improved"] >= 24
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
