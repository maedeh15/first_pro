"""
cmp_a2_sessions.py (A2) -- cross-session table in the layout of MSTFDN's Table 2 (Strategy 1 = leave-one-session-out): test session 1 / 2 / 3 and "average of the subjects",
each as mean over subjects +/- SD over subjects, for the WINDOW and the BLOCK level separately.
Source for the final model: ensemble_folds.csv (78 fold rows). Official number = per fold the mean of the seeds' accuracies, then over folds of a subject (for the
"average" column), then over subjects; the ensemble columns (ens_*) are a separate K-network system and are labelled as such. Other models: --long-results CSV
(columns model, subject, session, seed, win_acc[, block_acc]) -> same table; each model's per-subject mean is taken over seeds first.
PAPER rows: only MSTFDN-early (Ding et al. 2026, Table 10 and 11, 3-category N-back) -- the variant whose size matches this model -- are added, with the paper's own
numbers and NO statistics; the paper gives its Strategy-1 average only (no per-session split). --include-full-mstfdn adds the full MSTFDN row (Table 2: 68.57 / 70.68 /
67.82, average 69.02 +/- 5.16; 34.47 M parameters), which a reader will ask about.
Protocol caveat printed with the table: preprocessing, window/band definitions, checkpoint selection and the transductive steps (AdaBN, member centring) differ from the paper.
  python cmp_a2_sessions.py --ensemble-folds ensemble_folds.csv --out-dir out [--long-results long_results.csv] [--include-full-mstfdn]
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd

PAPER_EARLY = ("MSTFDN-early (paper, Table 10; 0.403 M parameters)", None, None, None, (56.24, 3.36))     # CS1 average +/- SD only
PAPER_FULL = ("MSTFDN (paper, Table 2; 34.47 M parameters)", (68.57, 5.05), (70.68, 7.66), (67.82, 5.32), (69.02, 5.16))


def table_from_folds(df, cols, level):
    """cols: fold-level accuracy columns to average (the seeds). returns per-subject 3 session values and their mean"""
    d = df.copy(); d["acc"] = d[cols].mean(axis=1)
    P = d.pivot(index="subject_id", columns="test_session", values="acc").dropna()
    return P


def summarise(P):
    out = {}
    for s in sorted(P.columns): out[f"session{s}"] = (100 * P[s].mean(), 100 * P[s].std(ddof=1))
    avg = P.mean(axis=1); out["average"] = (100 * avg.mean(), 100 * avg.std(ddof=1)); return out, P


def fmt(t): return "--" if t is None else f"{t[0]:.2f} ± {t[1]:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble-folds", required=True); ap.add_argument("--long-results", default=None); ap.add_argument("--out-dir", default="cmp_out")
    ap.add_argument("--include-full-mstfdn", action="store_true"); a = ap.parse_args(argv); os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.ensemble_folds); seeds = sorted({c.split("seed")[1] for c in df.columns if c.startswith("win_acc_seed")})
    rows = {}
    for level, pre in (("window", "win_acc_seed"), ("block", "block_acc_seed")):
        tab = []
        for sd in seeds:
            s, _ = summarise(table_from_folds(df, [f"{pre}{sd}"], level)); tab.append((f"final, seed {sd}", s))
        s3, P3 = summarise(table_from_folds(df, [f"{pre}{sd}" for sd in seeds], level)); tab.append((f"final, mean of {len(seeds)} seeds (official)", s3))
        ens, _ = summarise(table_from_folds(df, ["ens_win_acc" if level == "window" else "ens_block_acc"], level)); tab.append((f"final, ensemble of {len(seeds)} networks (3x cost)", ens))
        rows[level] = tab
    if a.long_results:
        L = pd.read_csv(a.long_results)
        for level, col in (("window", "win_acc"), ("block", "block_acc")):
            if col not in L.columns: continue
            for m, g in L.groupby("model"):
                P = g.groupby(["subject", "session"])[col].mean().unstack("session").dropna(); s, _ = summarise(P); rows[level].append((f"{m} (this work)", s))
    md = []
    for level in ("window", "block"):
        md.append(f"\n**{level.capitalize()}-level accuracy (%), leave-one-session-out, mean ± SD over subjects (n = {df['subject_id'].nunique()})**\n")
        md.append("| model | test session 1 | test session 2 | test session 3 | average of the subjects |\n|---|---|---|---|---|")
        for name, s in rows[level]: md.append(f"| {name} | {fmt(s.get('session1'))} | {fmt(s.get('session2'))} | {fmt(s.get('session3'))} | {fmt(s['average'])} |")
        if level == "window":
            md.append("| *reported in the paper, not re-run:* | | | | |")
            for nm, s1, s2, s3, av in ([PAPER_EARLY] + ([PAPER_FULL] if a.include_full_mstfdn else [])): md.append(f"| {nm} | {fmt(s1)} | {fmt(s2)} | {fmt(s3)} | {fmt(av)} |")
    md.append("\nPaper rows: window-level, their protocol (Cross-Session Strategy 1), no statistical test. Preprocessing, window/band definitions, checkpoint selection and the transductive "
              "steps (AdaBN, member centring) differ from this work; do not read the difference as a controlled comparison.")
    txt = "\n".join(md); print(txt); open(os.path.join(a.out_dir, "table_cross_session.md"), "w", encoding="utf-8").write(txt)
    pd.DataFrame([dict(level=l, model=n, **{k: (f"{v[0]:.4f}±{v[1]:.4f}") for k, v in s.items()}) for l, t in rows.items() for n, s in t]).to_csv(os.path.join(a.out_dir, "table_cross_session.csv"), index=False)
    return rows


def self_test():
    rng = np.random.RandomState(0); ids = [f"VP{i:03d}" for i in range(1, 27)]; base = rng.uniform(0.4, 0.8, 26)
    r = [dict(subject_id=s, test_session=t, **{f"win_acc_seed{sd}": base[i] + rng.normal(0, .03) for sd in (42, 999)}, **{f"block_acc_seed{sd}": base[i] + .03 for sd in (42, 999)},
              ens_win_acc=base[i], ens_block_acc=base[i] + .04) for i, s in enumerate(ids) for t in (1, 2, 3)]
    d = tempfile.mkdtemp(); p = os.path.join(d, "f.csv"); pd.DataFrame(r).to_csv(p, index=False)
    rows = main(["--ensemble-folds", p, "--out-dir", d, "--include-full-mstfdn"]); off = dict(rows["window"])["final, mean of 2 seeds (official)"]["average"]
    assert abs(off[0] - 100 * base.mean()) < 3.0
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
