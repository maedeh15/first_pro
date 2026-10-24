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
  # no path typed by hand: list every run found under these folders
  python compare_runs.py --discover "E:\\...\\Results" "E:\\...\\Results_baseline"
  # then build the whole intervention table against one of the hashes it listed
  python compare_runs.py --discover "E:\\...\\Results" --baseline-hash 40b7d8ea53 [--labels names.json]
  python compare_runs.py --a <csv> [<csv> ...] --b <csv> [<csv> ...] [--label-a ...] [--label-b ...]
e.g.
  python compare_runs.py --label-a baseline --label-b "riem member" \
    --a "...\\results_40b7d8ea53_seed42\\baseline_all_subjects_*.csv" ... \
    --b "...\\results_a1621a1ecc_seed42\\baseline_all_subjects_*.csv" ...
"""
import csv
import re
import json
import glob
import argparse
import tempfile
import os
import numpy as np
from scipy.stats import wilcoxon

METRICS = [("mean_acc", "accuracy"), ("mean_kappa", "kappa"),
           ("recall_0back", "recall 0-back"), ("recall_2back", "recall 2-back"),
           ("recall_3back", "recall 3-back")]


def seed_of(path):
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return m.group(1) if m else None


def load(patterns, label="", allow_dup=False, exclude_seeds=()):
    """-> {metric: {subject: mean over the runs that have it}}, plus the set of incomplete subjects

    A glob such as results_*_seed*/baseline_all_subjects_*.csv happily matches an ABANDONED run of a
    seed alongside its good one (this project has a seed-42 run that failed on 7 subjects). Averaging
    the two silently halves those subjects' weight, and the failed subjects are then dropped from
    both sides, so the comparison quietly shrinks to 19 subjects. Two csv of the same seed are
    therefore refused unless --allow-duplicate-seeds says they are meant to be pooled."""
    files, empty = [], []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if not hits:
            empty.append(pat)
        files.extend(hits)
    if exclude_seeds:
        drop = [f for f in files if seed_of(f) in {str(x) for x in exclude_seeds}]
        files = [f for f in files if f not in drop]
        for f in drop:
            print(f"{label or 'run'}: EXCLUDED seed {seed_of(f)} -- {os.path.basename(f)}")
        if not files:
            raise SystemExit(f"{label}: every csv was excluded by --exclude-seed")
    if empty:
        raise SystemExit(f"{label}: these patterns matched no csv: {empty}\n"
                         f"  a pattern that silently matches nothing drops a whole seed from the comparison.")
    print(f"{label or 'run'}: {len(files)} csv")
    for f in files:
        print(f"    seed {seed_of(f) or '?':>5}  {os.path.basename(f)}")
    seen = {}
    for f in files:
        seen.setdefault(seed_of(f), []).append(os.path.basename(f))
    dup = {k: v for k, v in seen.items() if len(v) > 1}
    if dup and not allow_dup:
        raise SystemExit(f"{label}: more than one csv for the same seed: {dup}\n"
                         f"  pass the intended files explicitly, or --allow-duplicate-seeds to pool them.")
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
    return ({k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in vals.items()},
            bad, len({seed_of(f) for f in files}))


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


def per_seed_signs(pa, pb, exclude):
    """sign of the mean accuracy difference within each seed present on both sides"""
    fa = {seed_of(f): f for pat in pa for f in sorted(glob.glob(pat))}
    fb = {seed_of(f): f for pat in pb for f in sorted(glob.glob(pat))}
    ex = {str(x) for x in exclude}
    signs = []
    for sd in sorted(set(fa) & set(fb) - ex):
        a1, _, _ = load([fa[sd]], f"seed {sd} A")
        b1, _, _ = load([fb[sd]], f"seed {sd} B")
        subs = sorted(set(a1["mean_acc"]) & set(b1["mean_acc"]))
        d = np.mean([b1["mean_acc"][s] - a1["mean_acc"][s] for s in subs])
        signs.append(int(np.sign(d)))
        print(f"  seed {sd}: mean difference {d:+.4f}")
    return signs


def verdict(diff, p, per_seed_signs, rule):
    """Print a PASS/FAIL against a rule the caller wrote down BEFORE seeing this output.

    The point is not the arithmetic -- it is that the threshold is in the command line, so it cannot
    be chosen after the fact. A rule given here is only pre-registered if the command was written
    before the runs finished; the tool cannot check that, so it says so."""
    if not any(v is not None and v is not False for v in rule.values()):
        return
    print("\nPRE-REGISTERED DECISION RULE (as given on the command line):")
    checks = []
    if rule["min_diff"] is not None:
        checks.append((f"mean difference >= {rule['min_diff']:+.4f}", diff >= rule["min_diff"],
                       f"{diff:+.4f}"))
    if rule["max_p"] is not None:
        checks.append((f"p <= {rule['max_p']:.4f}", p <= rule["max_p"], f"{p:.4f}"))
    if rule["same_sign"] and per_seed_signs is not None:
        same = len(set(per_seed_signs)) == 1 and per_seed_signs[0] != 0
        checks.append(("every seed moves in the same direction", same, str(per_seed_signs)))
    for text, passed, got in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {text}   (observed {got})")
    ok = all(c[1] for c in checks)
    print(f"  -> {'SUPPORTED' if ok else 'NOT SUPPORTED'} under this rule.")
    print("  This is only a pre-registration if the rule was written before the runs finished; "
          "nothing here can verify that.")


def compare(A, B, bad, la, lb, boot=10000, rule=None, per_seed_signs=None):
    subs = sorted(set(A.get("mean_acc", {})) & set(B.get("mean_acc", {})) - bad)
    if not subs:
        raise SystemExit("no subject completed in both runs")
    print(f"\n{len(subs)} subjects compared (seeds averaged per subject first).")
    if bad:
        print(f"*** dropped from BOTH sides, incomplete in at least one run: {sorted(bad)}\n"
              f"*** the numbers below are over {len(subs)} subjects, not 26 -- say so wherever they are reported.")
    print(f"\n{'metric':15} {la[:12]:>12} {lb[:12]:>12} {'diff':>9} {'wins':>7} {'p':>8} {'95% CI of diff':>22}")
    out = {}
    for k, name in METRICS:
        if k not in A or k not in B:
            continue
        a = np.array([A[k][s] for s in subs])
        b = np.array([B[k][s] for s in subs])
        d = b - a
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        if boot:
            lo, hi = boot_ci(d, boot)
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
        else:
            ci = "-"
        print(f"{name:15} {a.mean():12.4f} {b.mean():12.4f} {d.mean():+9.4f} {int((d > 0).sum()):3d}/{len(d):<3} "
              f"{p:8.4f} {ci:>22}")
        out[k] = (d.mean(), p)
    movers(A, B, subs, la, lb)
    if rule:
        verdict(out["mean_acc"][0], out["mean_acc"][1], per_seed_signs, rule)
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
                p = os.path.join(td, f"{path}_seed{k}.csv")
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
        A, badA, _ = load(pa, "A")
        B, badB, _ = load(pb, "B")
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
        # the wildcard trap: two csv of the same seed must be refused
        dupdir = os.path.join(td, "dup")
        os.makedirs(dupdir)
        for nm in ("run_a_seed42.csv", "run_b_seed42.csv"):
            with open(os.path.join(dupdir, nm), "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["subject_id", "status", "mean_acc"])
                w.writeheader()
                w.writerow(dict(subject_id="VP001", status="ok", mean_acc=0.6))
        try:
            load([os.path.join(dupdir, "*.csv")], "dup")
            good = False
        except SystemExit as e:
            good = "same seed" in str(e)
        print(f"[dupguard] two csv of one seed refused -> {'OK' if good else 'FAIL'}")
        ok &= good

        # a pattern that matches nothing must not pass silently
        try:
            load([os.path.join(td, "A_seed*.csv"), os.path.join(td, "nope_*.csv")], "A")
            good = False
        except SystemExit as e:
            good = "matched no csv" in str(e)
        print(f"[emptyglob] a pattern matching nothing is refused -> {'OK' if good else 'FAIL'}")
        ok &= good

        # --table mode: rows, seed counts and Holm across the family
        cfgp = os.path.join(td, "cfg.json")
        import json as _json
        _json.dump({"baseline": {"label": "base", "csv": pa},
                    "rows": [{"label": "up", "csv": pb},
                             {"label": "same", "csv": pa},
                             {"label": "one seed", "csv": pb[:1], "baseline": pa[:1]}]},
                   open(cfgp, "w"))
        import io as _io
        import contextlib as _ctx
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            run_table(_json.load(open(cfgp, encoding="utf-8")), 0)
        txt = buf.getvalue()
        good = ("| up |" in txt and "| same |" in txt and "| Holm |" in txt
                and "+0.0000" in txt and "Holm family" in txt)
        print(f"[table] rows, the reference column and the Holm families rendered -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        good = "| base | 3 | 3 | 25 |" in txt
        print(f"[table] reference, seed counts and reduced n reported per row -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        seg = txt.split("| one seed |")[1]
        good = "| 1 | 1 | 25 |" in seg[:60] and "A_seed0.csv" in seg[:60]
        print(f"[table] a row with its own one-seed baseline is read 1-vs-1 and names it -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good

        # --discover: the layout is read, and an abandoned run of a seed loses to its complete one
        dd = os.path.join(td, "disc")
        # ("hash", seed, stamp, n failed, n subjects written)
        for h, seed, stamp, nfail, ntot in (("aaaa111111", "42", "20260101_000000", 0, 26),
                                            ("aaaa111111", "999", "20260102_000000", 0, 26),
                                            ("bbbb222222", "42", "20260103_000000", 7, 26),
                                            ("bbbb222222", "42", "20260104_000000", 0, 26),
                                            # a run stopped after 2 subjects: no failures at all,
                                            # and the NEWEST -- it must still lose to the full run
                                            ("bbbb222222", "42", "20260105_000000", 0, 2),
                                            ("cccc333333", "42", "20260106_000000", 0, 2)):
            d2 = os.path.join(dd, f"results_{h}_seed{seed}")
            os.makedirs(d2, exist_ok=True)
            with open(os.path.join(d2, f"baseline_all_subjects_{stamp}_seed{seed}_{h}.csv"),
                      "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["subject_id", "status", "mean_acc"])
                w.writeheader()
                for i, sid in enumerate(subs[:ntot]):
                    okrow = i >= nfail
                    w.writerow(dict(subject_id=sid, status="ok" if okrow else "failed",
                                    mean_acc=(base[sid] + (0.03 if h == "bbbb222222" else 0.0)) if okrow else ""))
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            runs = discover([dd])
        good = (sorted(runs) == ["aaaa111111", "bbbb222222"]
                and sorted(runs["aaaa111111"]) == ["42", "999"]
                and "20260104" in runs["bbbb222222"]["42"])
        print(f"[discover] full run beats both the 7-failure run and the newer 2-subject run -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        good = "cccc333333" not in runs and "PARTIAL" in buf.getvalue()
        print(f"[discover] variant with only a 2-subject run excluded and flagged -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        try:
            table_discovered([dd], "aaaa111111", {}, None, 26)
            good = False
        except SystemExit as e:
            good = "not a family" in str(e)
        print(f"[discover] a table over every variant is refused without --rows -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            run_table(table_discovered([dd], "aaaa111111", {"bbbb222222": "intervention"},
                                       ["bbbb222222"], 26), 0)
        txt2 = buf.getvalue()
        good = "| intervention |" in txt2 and "+0.03" in txt2
        print(f"[discover] table built against the chosen baseline hash -> {'OK' if good else 'FAIL'}")
        ok &= good

        # --exclude-seed must drop the named seed from both sides
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            Ax, _, nax = load(pa, "A", exclude_seeds=("1",))
        good = nax == 2 and "EXCLUDED seed 1" in buf.getvalue()
        print(f"[exclude] --exclude-seed drops that seed and says so -> {'OK' if good else 'FAIL'}")
        ok &= good

        # the decision rule must fail a result that misses one of its conditions
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            verdict(0.004, 0.2, [1, 1], dict(min_diff=0.01, max_p=0.05, same_sign=True))
        txt3 = buf.getvalue()
        good = "NOT SUPPORTED" in txt3 and txt3.count("[FAIL]") == 2 and "[PASS]" in txt3
        print(f"[rule] a result below threshold and above p fails exactly those two checks -> "
              f"{'OK' if good else 'FAIL'}")
        ok &= good
        buf = _io.StringIO()
        with _ctx.redirect_stdout(buf):
            verdict(0.02, 0.01, [1, 1], dict(min_diff=0.01, max_p=0.05, same_sign=True))
        good = "SUPPORTED" in buf.getvalue() and "NOT SUPPORTED" not in buf.getvalue()
        print(f"[rule] a result meeting every condition passes -> {'OK' if good else 'FAIL'}")
        ok &= good

        # null case: same data on both sides must not be called a difference
        A2, b2, _ = load(pa, "A")
        res0, _ = compare(A2, A2, set(), "A", "A", boot=0)
        good = res0["mean_acc"][1] == 1.0 and abs(res0["mean_acc"][0]) < 1e-12
        print(f"[null] identical runs give diff 0, p=1 -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def discover(roots, min_subjects=26):
    """Find every complete run under <roots> without any path being typed by hand.

    Runs live at <root>/results_<hash>_seed<N>/baseline_all_subjects_<stamp>_seed<N>_<hash>.csv, so
    the variant hash and the seed come from the layout itself. Transcribing those paths into a config
    by hand is how the first version of the chapter-5 table ended up disagreeing with the data in
    every row.

    Where one seed of one variant has several csv (this project has both an abandoned seed-42 run
    that failed on 7 subjects and several runs stopped after a couple of subjects), the run with the
    MOST COMPLETED subjects wins, then the one with fewest failures, then the latest timestamp. The
    completed count has to come first: a run stopped after two subjects has no failed rows at all,
    so a rule that only counted failures preferred it over the full 26-subject run.

    -> {hash: {seed: path}}
    """
    found = {}
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "baseline_all_subjects_*.csv"), recursive=True)):
            m = re.search(r"baseline_all_subjects_(\d+_\d+)_seed(\d+)_([0-9a-f]+)\.csv$", os.path.basename(f))
            if not m:
                continue
            stamp, seed, h = m.groups()
            nbad = 0
            with open(f, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            nbad = sum(1 for r in rows if r.get("status") != "ok")
            nok = sum(1 for r in rows if r.get("status") == "ok")
            found.setdefault(h, {}).setdefault(seed, []).append((nbad, stamp, f, len(rows), nok))
    out = {}
    for h, seeds in sorted(found.items()):
        for seed, cands in sorted(seeds.items()):
            cands.sort(key=lambda c: (-c[4], c[0], [-int(x) for x in c[1].split("_")]))
            nbad, stamp, f, n, nok = cands[0]
            note = ""
            if len(cands) > 1:
                note = (f"  <- chosen over {len(cands) - 1} other csv of this seed "
                        f"({', '.join(f'{c[1]}: {c[4]} ok/{c[0]} failed' for c in cands[1:])})")
            flag = f"  [{nbad} failed]" if nbad else ""
            if nok < min_subjects:
                flag += f"  [PARTIAL: only {nok} subjects -- excluded from tables]"
            print(f"  {h}  seed {seed:>4}  {nok:3d} ok{flag}{note}")
            if nok >= min_subjects:
                out.setdefault(h, {})[seed] = f
    return out


def table_discovered(roots, baseline_hash, labels, row_hashes, min_subjects):
    runs = discover(roots, min_subjects)
    if baseline_hash not in runs:
        raise SystemExit(f"baseline hash {baseline_hash} not found under {roots}; "
                         f"available: {sorted(runs)}")
    if not row_hashes:
        raise SystemExit(
            f"{len(runs)} variants have a run of at least {min_subjects} subjects. A table over all of "
            f"them is not a family of interventions: pass --rows <hash> ... with the ones the table "
            f"should contain, and state the inclusion rule in the text.")
    missing = [h for h in row_hashes if h not in runs]
    if missing:
        raise SystemExit(f"no complete run found for: {missing}")
    return {"baseline": {"label": labels.get(baseline_hash, baseline_hash),
                         "csv": sorted(runs[baseline_hash].values())},
            "rows": [{"label": labels.get(h, h), "csv": sorted(runs[h].values())} for h in row_hashes]}


def run_table(cfg, boot):
    """Build the intervention table from a json config, instead of assembling it by hand.

    {"baseline": {"label": "baseline", "csv": ["...seed42...csv", "...seed999...csv"]},
     "rows": [{"label": "cafe-lambda 2", "csv": ["...seed42...csv"],
               "baseline": ["...baseline seed42 only...csv"]}, ...]}

    A row may carry its own "baseline". A one-seed intervention has to be read against the SAME
    seed of the reference run, not against the mean of three: seed-to-seed spread in this project is
    up to about one point, which is the size of most of the effects being judged.

    Each row is compared against the same baseline with the same reading, the number of seeds on
    each side is printed (a one-seed row and a three-seed row are not equally strong evidence), and
    Holm is applied ACROSS THE ROWS, which is the family that matters: these are the interventions
    tried on one dataset. Hand-assembled versions of this table in this project disagreed with the
    csv files in every row, which is why it exists."""
    base = cfg["baseline"]
    A0, badA0, nA0 = load(base["csv"], base.get("label", "baseline"))
    rows, ps = [], []
    for r in cfg["rows"]:
        if r.get("baseline"):
            A, badA, nA = load(r["baseline"], f"baseline for {r['label']}")
        else:
            A, badA, nA = A0, badA0, nA0
        B, badB, nB = load(r["csv"], r["label"])
        subs = sorted((set(A["mean_acc"]) & set(B["mean_acc"])) - (badA | badB))
        a = np.array([A["mean_acc"][s] for s in subs])
        b = np.array([B["mean_acc"][s] for s in subs])
        d = b - a
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        ci = boot_ci(d, boot) if boot else (float("nan"), float("nan"))
        ref = r.get("baseline_label") or (base.get("label", "baseline") if not r.get("baseline")
                                          else " ".join(os.path.basename(c) for c in r["baseline"]))
        rows.append(dict(label=r["label"], seeds=nB, base_seeds=nA, n=len(subs), diff=d.mean(),
                         wins=int((d > 0).sum()), p=float(p), ci=ci, ref=ref,
                         dropped=sorted(badA | badB)))
        ps.append(float(p))
    # Holm is applied WITHIN each group of rows that share a reference run. Rows measured against a
    # different reference are not the same family: correcting them together would mix comparisons
    # that answer different questions.
    adj = [None] * len(rows)
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r["ref"], []).append(i)
    for ref, idx in groups.items():
        for i, h in zip(idx, holm([ps[i] for i in idx])):
            adj[i] = h
    print(f"\n| intervention | reference | seeds | base seeds | n | diff | wins | p | Holm | 95% CI |")
    print("|---|---|---|---|---|---:|---|---:|---:|---|")
    for r, h in zip(rows, adj):
        lo, hi = r["ci"]
        ci = "-" if np.isnan(lo) else f"[{lo:+.4f}, {hi:+.4f}]"
        mark = " **" if r["seeds"] != r["base_seeds"] else ""
        print(f"| {r['label']} | {r['ref']} | {r['seeds']} | {r['base_seeds']}{mark} | {r['n']} | "
              f"{r['diff']:+.4f} | {r['wins']}/{r['n']} | {r['p']:.4f} | {h:.4f} | {ci} |")
    if any(r["seeds"] != r["base_seeds"] for r in rows):
        print("\n** this row compares a different number of seeds on the two sides. Seed-to-seed spread "
              "here reaches about one point, so give such a row its own same-seed baseline.")
    print(f"\ndefault baseline: {nA0} seed(s), {len(A0['mean_acc'])} subjects.")
    drop = {tuple(r["dropped"]) for r in rows if r["dropped"]}
    for d_ in drop:
        print(f"*** some rows dropped subjects: {list(d_)} -- their n differs, say so in the caption.")
    for ref, idx in groups.items():
        print(f"Holm family '{ref}': {len(idx)} row(s).")
    print("Holm is applied WITHIN each reference group, because rows measured against different "
          "reference runs answer different questions. If other interventions were tried against the "
          "same reference and are not listed, that family is larger than shown and these corrected "
          "values are optimistic: list every intervention, or state the selection rule.")


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    out = np.zeros(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i]))
        out[i] = run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", help="reference run csv(s), one per seed (globs allowed)")
    ap.add_argument("--b", nargs="+", help="intervention run csv(s), one per seed")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--exclude-seed", nargs="+", default=(), metavar="SEED",
                    help="drop these seeds from BOTH sides. Use it to keep the seed a variant was "
                         "CHOSEN on out of its own confirmation: a variant picked as the best of "
                         "several on seed 42 is biased upward on seed 42, and including it makes the "
                         "p-value optimistic.")
    ap.add_argument("--rule-min-diff", type=float, default=None,
                    help="pre-registered decision rule: the minimum mean accuracy difference")
    ap.add_argument("--rule-max-p", type=float, default=None,
                    help="pre-registered decision rule: the maximum p-value")
    ap.add_argument("--rule-same-sign", action="store_true",
                    help="pre-registered decision rule: every seed must move in the same direction")
    ap.add_argument("--allow-duplicate-seeds", action="store_true",
                    help="pool several csv of the same seed instead of refusing them")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--table", help="json config: build the whole intervention table")
    ap.add_argument("--discover", nargs="+", metavar="ROOT",
                    help="find every run under these folders automatically -- no path typed by hand")
    ap.add_argument("--baseline-hash", help="with --discover: the variant hash of the reference run")
    ap.add_argument("--labels", help="with --discover: optional json {hash: readable name}")
    ap.add_argument("--rows", nargs="+", metavar="HASH",
                    help="with --discover: the variant hashes that become rows of the table")
    ap.add_argument("--min-subjects", type=int, default=26,
                    help="a run with fewer completed subjects is treated as partial and excluded")
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if a.discover:
        if not a.baseline_hash:
            print("no --baseline-hash given; listing what was found and stopping.\n")
            discover(a.discover, a.min_subjects)
            raise SystemExit(0)
        labels = json.load(open(a.labels, encoding="utf-8")) if a.labels else {}
        raise SystemExit(run_table(table_discovered(a.discover, a.baseline_hash, labels,
                                                    a.rows, a.min_subjects), a.boot) or 0)
    if a.table:
        with open(a.table, encoding="utf-8") as f:
            raise SystemExit(run_table(json.load(f), a.boot) or 0)
    if not (a.a and a.b):
        ap.error("--a and --b are required unless --self-test")
    A, badA, _ = load(a.a, a.label_a, a.allow_duplicate_seeds)
    B, badB, _ = load(a.b, a.label_b, a.allow_duplicate_seeds)
    rule = dict(min_diff=a.rule_min_diff, max_p=a.rule_max_p, same_sign=a.rule_same_sign)
    signs = per_seed_signs(a.a, a.b, a.exclude_seed) if a.rule_same_sign else None
    compare(A, B, badA | badB, a.label_a, a.label_b, a.boot, rule, signs)


if __name__ == "__main__":
    main()
