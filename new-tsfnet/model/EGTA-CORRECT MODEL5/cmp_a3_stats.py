"""
cmp_a3_stats.py (A3) -- paired subject-level statistics of the reference model against the other runs (layout of MSTFDN's Tables 3 and 4).
Input: per_subject_long.csv from cmp_a1_registry.py (columns hash, seed, subject_id, mean_acc) and, optionally, --long-results (model, subject, session, seed, win_acc) for
models run elsewhere. Unit = subject (n = 26). For every run the SEEDS COMMON to the reference and the comparator are averaged inside each subject first, then the
paired Wilcoxon signed-rank test (two-sided), bootstrap 95% CI over subjects (10000, seed 0), Cliff's delta and the difference in percentage points are computed; Holm
correction inside ONE family = one call of this script (the family is printed). --per-seed adds the single-seed sensitivity rows. Window-level accuracy only; fewer than 6
paired subjects are reported descriptively. Published numbers of other papers never enter here (no per-subject values).
  python cmp_a3_stats.py --long out\\per_subject_long.csv --ref cf64a71127 --vs 40b7d8ea53 a1621a1ecc 84c18210ab --out-dir out [--per-seed]
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd
from cmp_common import paired, holm, delta_label


def load(a):
    L = pd.read_csv(a.long)[["hash", "seed", "subject_id", "mean_acc"]].rename(columns={"hash": "run", "subject_id": "subject", "mean_acc": "acc"})
    if a.long_results:
        X = pd.read_csv(a.long_results); X = X.groupby(["model", "seed", "subject"], as_index=False)["win_acc"].mean().rename(columns={"model": "run", "win_acc": "acc"}); L = pd.concat([L, X], ignore_index=True)
    return L


def vec(L, run, seeds):
    g = L[(L["run"] == run) & (L["seed"].isin(seeds))]; return g.groupby("subject")["acc"].mean()


def compare(L, ref, others, per_seed=False):
    rows = []; seeds_ref = set(L[L.run == ref]["seed"])
    if not seeds_ref: raise SystemExit(f"reference {ref!r} not found")
    todo = []
    for o in others:
        common = sorted(seeds_ref & set(L[L.run == o]["seed"]))
        if not common: print(f"[skip] {o}: no seed in common with {ref} (ref seeds {sorted(seeds_ref)}, {o} seeds {sorted(set(L[L.run == o]['seed']))})"); continue
        todo.append((o, common, f"seeds {common} averaged"))
        if per_seed: todo += [(o, [s], f"seed {s} only") for s in common]
    for o, seeds, note in todo:
        a, b = vec(L, ref, seeds), vec(L, o, seeds); ids = sorted(set(a.index) & set(b.index))
        r = paired(a.loc[ids].values, b.loc[ids].values); r.update(comparator=o, note=note, descriptive=len(ids) < 6); rows.append(r)
    fam = [i for i, r in enumerate(rows) if not r["note"].endswith("only")]
    adj = holm([rows[i]["p_raw"] for i in fam]) if fam else []
    for r in rows: r["p_holm"] = np.nan
    for i, p in zip(fam, adj): rows[i]["p_holm"] = float(p)
    return rows


def show(ref, rows):
    print(f"\nreference = {ref}; family for Holm = the {sum(1 for r in rows if not r['note'].endswith('only'))} seed-averaged comparisons below; accuracy in %, difference = reference minus comparator\n")
    print("| comparator | basis | n | reference [95% CI] | comparator [95% CI] | difference [95% CI] (pp) | Cliff's δ | p (raw) | p (Holm) | reference better in |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ph = "--" if np.isnan(r["p_holm"]) else f"{r['p_holm']:.3g}"
        print(f"| {r['comparator']} | {r['note']} | {r['n']} | {100*r['mean_a']:.2f} [{100*r['ci_a'][0]:.2f}, {100*r['ci_a'][1]:.2f}] | {100*r['mean_b']:.2f} [{100*r['ci_b'][0]:.2f}, {100*r['ci_b'][1]:.2f}] | "
              f"{100*r['diff']:+.2f} [{100*r['ci_diff'][0]:+.2f}, {100*r['ci_diff'][1]:+.2f}] | {r['cliffs']:+.2f} ({delta_label(r['cliffs'])}) | {r['p_raw']:.3g} | {ph} | {r['n_better']}/{r['n']} |" + (" descriptive (n<6)" if r["descriptive"] else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", required=True); ap.add_argument("--long-results", default=None); ap.add_argument("--ref", required=True); ap.add_argument("--vs", nargs="+", required=True)
    ap.add_argument("--per-seed", action="store_true"); ap.add_argument("--out-dir", default="cmp_out"); a = ap.parse_args(argv); os.makedirs(a.out_dir, exist_ok=True)
    rows = compare(load(a), a.ref, a.vs, a.per_seed); show(a.ref, rows)
    pd.DataFrame([{k: (str(v) if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]).to_csv(os.path.join(a.out_dir, f"stats_{a.ref}.csv"), index=False); return rows


def self_test():
    rng = np.random.RandomState(0); ids = [f"VP{i:03d}" for i in range(1, 27)]; base = rng.uniform(0.4, 0.7, 26); r = []
    for run, shift in (("ref", 0.06), ("low", 0.0), ("same", 0.06)):
        for seed in (42, 999, 2024):
            for i, s in enumerate(ids): r.append(dict(hash=run, seed=seed, subject_id=s, mean_acc=base[i] + shift + rng.normal(0, .015)))
    d = tempfile.mkdtemp(); p = os.path.join(d, "l.csv"); pd.DataFrame(r).to_csv(p, index=False)
    rows = main(["--long", p, "--ref", "ref", "--vs", "low", "same", "--out-dir", d, "--per-seed"]); by = {(x["comparator"], x["note"]): x for x in rows}
    lo = by[("low", "seeds [42, 999, 2024] averaged")]; sa = by[("same", "seeds [42, 999, 2024] averaged")]
    assert lo["p_holm"] < 1e-4 and abs(lo["diff"] - 0.06) < 0.01 and lo["n_better"] >= 24 and sa["p_holm"] > 0.05
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
