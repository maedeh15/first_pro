"""
compare_runs.py -- paired comparison of two full runs of run_all_subjects.py (A = reference,
B = intervention), over subjects, with seeds averaged first.

Every GPU intervention in this project has been judged by the same four numbers, computed by hand
each time: the mean over subjects, the paired difference, the number of subjects that improved, and
a Wilcoxon p-value. This does that from the per-run csv files, so a reading cannot drift between
experiments, and adds the two things the hand version kept omitting: a bootstrap CI over subjects
(the spread that matters, since subjects are the unit) and the per-class recall columns, which are
already in the csv and are where a flat accuracy difference usually hides a real change.

Seeds are averaged PER SUBJECT before the test: the unit of analysis is the subject, not the run,
and 26 subjects x k seeds would otherwise be treated as 26k independent observations.

Subjects that did not complete in either run (status != 'ok') are dropped from BOTH sides and
listed, because a run that fails on the hard subjects would otherwise look better than one that
does not.

usage:
  python compare_runs.py --self-test
  python compare_runs.py --a <csv> [<csv> ...] --b <csv> [<csv> ...] [--label-a ...] [--label-b ...]
e.g.
  python compare_runs.py --label-a baseline --label-b "riem member" \
    --a "...\\results_40b7d8ea53_seed42\\baseline_all_subjects_*.csv" ... \
    --b "...\\results_a1621a1ecc_seed42\\baseline_all_subjects_*.csv" ...
"""
import csv
import glob
import argparse
import tempfile
import os
import numpy as np
from scipy.stats import wilcoxon

METRICS = [("mean_acc", "accuracy"), ("mean_kappa", "kappa"),
           ("recall_0back", "recall 0-back"), ("recall_2back", "recall 2-back"),
           ("recall_3back", "recall 3-back")]


def load(patterns):
    """-> {metric: {subject: mean over the runs that have it}}, plus the set of incomplete subjects"""
    files = [f for p in patterns for f in sorted(glob.glob(p))]
    if not files:
        raise SystemExit(f"no csv matched {patterns}")
    vals, bad = {}, set()
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                sid = r["subject_id"]
                if r.get("status") != "ok":
                    bad.add(sid)
                    continue
                for k, _ in METRICS:
                    if r.get(k) in (None, ""):
                        continue
                    vals.setdefault(k, {}).setdefault(sid, []).append(float(r[k]))
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in vals.items()}, bad, files


def boot_ci(d, n=10000, seed=0):
    rng = np.random.RandomState(seed)
    m = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def movers(A, B, subs, la, lb, k=5):
    """the subjects that moved most in each direction, and every subject at or below chance in either
    run -- a mean difference hides both, and a subject below 1/3 in a 3-class task is a failed fit,
    not a hard subject."""
    d = {s: B["mean_acc"][s] - A["mean_acc"][s] for s in subs}
    order = sorted(subs, key=lambda s: d[s])
    print(f"\nlargest moves in accuracy ({lb} - {la}):")
    for s in order[:k]:
        print(f"  {s}  {A['mean_acc'][s]:.4f} -> {B['mean_acc'][s]:.4f}  {d[s]:+.4f}")
    print("  ...")
    for s in order[-k:]:
        print(f"  {s}  {A['mean_acc'][s]:.4f} -> {B['mean_acc'][s]:.4f}  {d[s]:+.4f}")
    low = [s for s in subs if min(A["mean_acc"][s], B["mean_acc"][s]) <= 1.0 / 3]
    if low:
        print("\nat or below 3-class chance (0.3333) in at least one run -- inspect these fits:")
        for s in low:
            print(f"  {s}  {la} {A['mean_acc'][s]:.4f} | {lb} {B['mean_acc'][s]:.4f}")
    else:
        print("\nno subject at or below chance in either run.")


def compare(A, B, bad, la, lb, boot=10000):
    subs = sorted(set(A.get("mean_acc", {})) & set(B.get("mean_acc", {})) - bad)
    if not subs:
        raise SystemExit("no subject completed in both runs")
    print(f"\n{len(subs)} subjects compared (seeds averaged per subject first).")
    if bad:
        print(f"dropped from BOTH sides (incomplete in at least one run): {sorted(bad)}")
    print(f"\n{'metric':15} {la[:12]:>12} {lb[:12]:>12} {'diff':>9} {'wins':>7} {'p':>8} {'95% CI of diff':>22}")
    out = {}
    for k, name in METRICS:
        if k not in A or k not in B:
            continue
        a = np.array([A[k][s] for s in subs])
        b = np.array([B[k][s] for s in subs])
        d = b - a
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        ci = f"[{boot_ci(d, boot)[0]:+.4f}, {boot_ci(d, boot)[1]:+.4f}]" if boot else "-"
        print(f"{name:15} {a.mean():12.4f} {b.mean():12.4f} {d.mean():+9.4f} {int((d > 0).sum()):3d}/{len(d):<3} "
              f"{p:8.4f} {ci:>22}")
        out[k] = (d.mean(), p)
    movers(A, B, subs, la, lb)
    print(f"\nThe accuracy row is the primary reading; the recall rows are descriptive. p-values are "
          f"uncorrected -- if this intervention is one of several tested, correct across them (Holm).")
    return out, subs


def self_test():
    ok = True
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.RandomState(0)
        subs = [f"VP{i:03d}" for i in range(1, 27)]
        base = {s: 0.55 + 0.1 * rng.rand() for s in subs}

        def write(path, delta, seeds=3, fail=()):
            paths = []
            for k in range(seeds):
                p = os.path.join(td, f"{path}_s{k}.csv")
                with open(p, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["subject_id", "status", "mean_acc", "mean_kappa",
                                                      "recall_0back", "recall_2back", "recall_3back"])
                    w.writeheader()
                    for s in subs:
                        if s in fail:
                            w.writerow(dict(subject_id=s, status="failed", mean_acc="", mean_kappa="",
                                            recall_0back="", recall_2back="", recall_3back=""))
                            continue
                        a = base[s] + delta + 0.01 * rng.randn()
                        w.writerow(dict(subject_id=s, status="ok", mean_acc=a, mean_kappa=1.5 * a - 0.5,
                                        recall_0back=a + 0.2, recall_2back=a - 0.1, recall_3back=a - 0.1))
                paths.append(p)
            return paths

        pa, pb = write("A", 0.0), write("B", 0.04, fail=("VP006",))
        A, badA, _ = load(pa)
        B, badB, _ = load(pb)
        good = badA == set() and badB == {"VP006"}
        print(f"[load] incomplete subject detected on the B side only -> {'OK' if good else 'FAIL'}")
        ok &= good
        res, subs_used = compare(A, B, badA | badB, "A", "B", boot=2000)
        good = "VP006" not in subs_used and len(subs_used) == 25
        print(f"[drop] VP006 excluded from BOTH sides, 25 compared -> {'OK' if good else 'FAIL'}")
        ok &= good
        d, p = res["mean_acc"]
        good = abs(d - 0.04) < 0.008 and p < 0.001
        print(f"[effect] planted +0.0400 recovered as {d:+.4f} p={p:.4f} -> {'OK' if good else 'FAIL'}")
        ok &= good
        # null case: same data on both sides must not be called a difference
        A2, b2, _ = load(pa)
        res0, _ = compare(A2, A2, set(), "A", "A", boot=0)
        good = res0["mean_acc"][1] == 1.0 and abs(res0["mean_acc"][0]) < 1e-12
        print(f"[null] identical runs give diff 0, p=1 -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", help="reference run csv(s), one per seed (globs allowed)")
    ap.add_argument("--b", nargs="+", help="intervention run csv(s), one per seed")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not (a.a and a.b):
        ap.error("--a and --b are required unless --self-test")
    A, badA, fa = load(a.a)
    B, badB, fb = load(a.b)
    print(f"{a.label_a}: {len(fa)} csv  |  {a.label_b}: {len(fb)} csv")
    compare(A, B, badA | badB, a.label_a, a.label_b, a.boot)


if __name__ == "__main__":
    main()
