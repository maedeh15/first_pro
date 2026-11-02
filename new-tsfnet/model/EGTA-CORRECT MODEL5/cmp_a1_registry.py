"""
cmp_a1_registry.py (A1) -- registry of all runs found under one or more folders, and the per-subject long table.
Reads baseline_all_subjects_<stamp>_seed<N>_<hash>.csv (or .json). Per (hash, seed): subjects (status ok), mean window accuracy and kappa over subjects, class recalls,
mean elapsed minutes, complete = 26 subjects. Per hash: the seed-mean (per subject first, then over subjects) when >= 2 seeds, and the list of seeds.
Window accuracy here is the subject mean over his 3 folds (as written by run_all_subjects.py); this file has NO fold-level accuracy.
Acceptance check: --check HASH:ACC:KAPPA (e.g. cf64a71127:0.6667:0.5001) compares the 3-seed mean to 4 decimals.
  python cmp_a1_registry.py --roots "E:\\...\\TSFNET_PAPER_EXACT_nback" "E:\\...\\TSFNET_PAPER_EXACT_nback_baseline" --out-dir out --check cf64a71127:0.6667:0.5001
"""
import argparse, os, sys, tempfile
import numpy as np, pandas as pd
from cmp_common import find_runs, load_run

COLS = ["mean_acc", "mean_kappa", "eeg_only_acc", "fnirs_only_acc", "fusion_only_acc", "w_eeg", "w_fusion", "w_fnirs", "recall_0back", "recall_2back", "recall_3back", "elapsed_min"]


def build(roots):
    runs = find_runs(roots); long, rows = [], []
    for (h, s), path in sorted(runs.items()):
        d = load_run(path)
        if "subject_id" not in d.columns: print(f"[skip] {path}: no subject_id column"); continue
        for c in COLS:
            if c not in d.columns: d[c] = np.nan
        d = d.assign(hash=h, seed=s, source_file=os.path.basename(path)); long.append(d[["hash", "seed", "subject_id"] + COLS + ["source_file"]])
        rows.append(dict(hash=h, seed=s, n_subjects=len(d), complete=len(d) == 26, **{c: float(pd.to_numeric(d[c], errors="coerce").mean()) for c in COLS}))
    L = pd.concat(long, ignore_index=True) if long else pd.DataFrame(); R = pd.DataFrame(rows)
    seedmean = []
    for h, g in L.groupby("hash") if len(L) else []:
        sm = g.groupby("subject_id")[["mean_acc", "mean_kappa"]].mean(); full = sorted(g.groupby("seed")["subject_id"].nunique().loc[lambda x: x == 26].index)
        sub = g[g["seed"].isin(full)].groupby("subject_id")[["mean_acc", "mean_kappa"]].mean() if len(full) >= 2 else None
        seedmean.append(dict(hash=h, seeds_all=sorted(g["seed"].unique().tolist()), seeds_complete_26=full, n_seeds_complete=len(full),
                             acc_seedmean=float(sub["mean_acc"].mean()) if sub is not None else np.nan, kappa_seedmean=float(sub["mean_kappa"].mean()) if sub is not None else np.nan))
    return L, R, pd.DataFrame(seedmean)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", required=True); ap.add_argument("--out-dir", default="cmp_out"); ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args(argv); os.makedirs(a.out_dir, exist_ok=True)
    L, R, S = build(a.roots)
    if R.empty: sys.exit("no baseline_all_subjects_* files found under --roots")
    L.to_csv(os.path.join(a.out_dir, "per_subject_long.csv"), index=False); R.to_csv(os.path.join(a.out_dir, "runs_registry.csv"), index=False); S.to_csv(os.path.join(a.out_dir, "seedmean_registry.csv"), index=False)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500)
    print(f"{len(R)} runs, {R['hash'].nunique()} distinct hashes, {int(R['complete'].sum())} complete (26 subjects)")
    print(R[["hash", "seed", "n_subjects", "mean_acc", "mean_kappa", "elapsed_min"]].round(4).to_string(index=False))
    print("\nseed means (only seeds with all 26 subjects; per subject first, then over subjects):"); print(S.round(4).to_string(index=False))
    ok = True
    for c in a.check:
        h, acc, kap = c.split(":"); row = S[S["hash"] == h]
        if row.empty: print(f"CHECK {h}: not found"); ok = False; continue
        da, dk = abs(float(row["acc_seedmean"].iloc[0]) - float(acc)), abs(float(row["kappa_seedmean"].iloc[0]) - float(kap))
        good = da < 5e-5 and dk < 5e-5; ok &= good
        print(f"CHECK {h}: acc {row['acc_seedmean'].iloc[0]:.4f} (expected {acc}), kappa {row['kappa_seedmean'].iloc[0]:.4f} (expected {kap}) -> {'PASS' if good else 'FAIL'}")
    return ok


def self_test():
    rng = np.random.RandomState(0); d = tempfile.mkdtemp(); ids = [f"VP{i:03d}" for i in range(1, 27)]
    for h, base in (("aaaaaaaaaa", 0.60), ("bbbbbbbbbb", 0.66)):
        for seed in (42, 999, 2024):
            sub = os.path.join(d, f"results_{h}_seed{seed}"); os.makedirs(sub)
            pd.DataFrame({"subject_id": ids, "status": "ok", "mean_acc": base + rng.normal(0, 0.03, 26), "mean_kappa": 0.4 + rng.normal(0, 0.03, 26), "elapsed_min": 6.0}).to_csv(
                os.path.join(sub, f"baseline_all_subjects_20260101_000000_seed{seed}_{h}.csv"), index=False)
    L, R, S = build([d]); assert len(R) == 6 and set(S["hash"]) == {"aaaaaaaaaa", "bbbbbbbbbb"} and (S["n_seeds_complete"] == 3).all()
    g = L[L.hash == "bbbbbbbbbb"].groupby("subject_id")["mean_acc"].mean().mean(); assert abs(S[S.hash == "bbbbbbbbbb"]["acc_seedmean"].iloc[0] - g) < 1e-12
    assert main(["--roots", d, "--out-dir", os.path.join(d, "o"), "--check", f"bbbbbbbbbb:{g:.4f}:{S[S.hash == 'bbbbbbbbbb']['kappa_seedmean'].iloc[0]:.4f}"]) is True
    print("SELFTEST PASSED")


if __name__ == "__main__":
    self_test() if "--self-test" in sys.argv else main()
