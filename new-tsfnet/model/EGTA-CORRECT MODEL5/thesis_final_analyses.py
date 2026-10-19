"""
thesis_final_analyses.py -- CPU-only analyses for the thesis write-up, in one file.

Subcommands
-----------
  describe     scan a folder of run logs and print one row per run (seed, hash, #subjects,
               grand mean accuracy, and the non-default flags found in the [ABLATION] line).
               Use it to FIND the Deep CORAL run and the cardiac-band-stop run.
  holm         paired Wilcoxon of every intervention against the matched baseline, per seed
               and on the seed-averaged per-subject accuracy, then Holm-Bonferroni over the
               whole family (plus any extra p-values you pass in).
  heads        fixed (equal-weight, NOT learned) average of the EEG / fusion / fNIRS head
               posteriors versus the model's own 'final' head, from ensemble_eval.py
               --save-posteriors output.
  splitprobe   how much does the SPLIT PROTOCOL alone inflate accuracy? Same features and
               classifier as check_feature_baseline_tsfnet.py under three protocols:
               A cross-session LOSO, B random window K-fold (overlapping windows leak),
               C trial-grouped K-fold (no overlap leak, but sessions pooled).
  selftest     planted-effect checks of all four on synthetic data.

Rules kept: nothing here trains a model; no number is chosen by looking at test accuracy;
every comparison is paired over subjects; Holm is applied to a family that is printed in full
(including the null results), so nothing is cherry-picked.

usage
-----
  python thesis_final_analyses.py selftest
  python thesis_final_analyses.py describe --logs-root "E:\\...\\TSFNET_PAPER_EXACT_nback"
  python thesis_final_analyses.py holm --logs-root "E:\\...\\TSFNET_PAPER_EXACT_nback" [--config cfg.json]
         [--extra-p ensemble_window=0.000745 ...]
  python thesis_final_analyses.py heads --posteriors-dir "...\\ensemble_eval_...\\posteriors"
  python thesis_final_analyses.py splitprobe --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import os
import re
import sys
import json
import glob
import argparse
import tempfile
import numpy as np
from scipy.stats import wilcoxon

SEEDS = ("42", "999", "2024")

# ----- default family: file NAMES only (found recursively under --logs-root; a leading
# "<uuid>-" prefix, as added by upload tools, is ignored). Edit with --config.
DEFAULT_CONFIG = {
    "baseline": {
        "old": {"42": "baseline_run_log_20260929_004912_seed42_40b7d8ea53.txt",
                "999": "baseline_run_log_20260929_041913_seed999_40b7d8ea53.txt",
                "2024": "baseline_run_log_20260929_075912_seed2024_40b7d8ea53.txt"},
        "new": {"42": "baseline_run_log_20261009_235146_seed42_40b7d8ea53.txt",
                "999": "baseline_run_log_20261010_091132_seed999_40b7d8ea53.txt",
                "2024": "baseline_run_log_20261010_125034_seed2024_40b7d8ea53.txt"},
    },
    "interventions": {
        "fnirs_grid_bbox_eeg": {
            "42": "baseline_run_log_20261018_010207_seed42_1ac25c39c5.txt",
            "999": "baseline_run_log_20261018_075508_seed999_1ac25c39c5.txt",
            "2024": "baseline_run_log_20261018_042509_seed2024_1ac25c39c5.txt"},
        "eeg_cov_align": {
            "42": "baseline_run_log_20261018_171550_seed42_641b91a59c.txt",
            "999": "baseline_run_log_20261019_011754_seed999_641b91a59c.txt",
            "2024": "baseline_run_log_20261018_212052_seed2024_641b91a59c.txt"},
        "fnirs_input_gate_soft_prior": {
            "42": "baseline_run_log_20261018_111457_seed42_31ba9f1e11.txt"},
        "stage1_fixed_epochs_12": {
            "42": "baseline_run_log_20261019_163649_seed42_b4d8862e80.txt"},
    },
    "extra_p": {},
}


# =============================================================================
# log parsing
# =============================================================================
ACC_RE = re.compile(r"^(VP\d+): acc=([\d.]+)", re.M)


def find_file(root, name):
    hits = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if f == name or f.endswith("-" + name):
                hits.append(os.path.join(dp, f))
    return sorted(hits)[0] if hits else None


def read_acc(path):
    """per-subject mean accuracy from the final 'PER-SUBJECT RESULTS' block of a run log"""
    with open(path, encoding="utf-8", errors="ignore") as f:
        t = f.read()
    d = {}
    for m in ACC_RE.finditer(t):
        d[m.group(1)] = float(m.group(2))
    return d


def paired(a, b):
    """a, b: dict subject->acc. returns (mean diff, wins, n, p)"""
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks])
    p = wilcoxon(d).pvalue if len(d) > 5 and np.any(d != 0) else float("nan")
    return float(d.mean()), int((d > 0).sum()), len(d), float(p)


def mean_dict(ds):
    ks = sorted(set.intersection(*[set(d) for d in ds]))
    return {k: float(np.mean([d[k] for d in ds])) for k in ks}


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)"""
    p = np.array(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


# =============================================================================
# describe
# =============================================================================
def cmd_describe(a):
    rows = []
    for dp, _dn, fn in os.walk(a.logs_root):
        for f in fn:
            if "baseline_run_log_" not in f or not f.endswith(".txt"):
                continue
            p = os.path.join(dp, f)
            with open(p, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
            ms = re.search(r"seed(\d+)_([0-9a-f]{10})\.txt$", f)
            g = re.search(r"Grand mean accuracy across subjects: ([\d.]+)", t)
            n = len(ACC_RE.findall(t))
            ab = re.search(r"\[ABLATION\][^\n]*", t)
            flags = []
            if ab:
                for k in ("domain_adapt_lambda", "tta_mode", "fnirs_bandpass", "label_smoothing",
                          "val_mode", "stage2_stop_mode", "cafe_lambda", "efgf_lambda",
                          "fnirs_input_gate", "swa_enabled", "eeg_filterbank", "fnirs_chroma_mode",
                          "norm_mode", "mhs_lambda"):
                    m = re.search(k + r"=([^\s|,)]+)", ab.group(0))
                    if m:
                        flags.append(f"{k}={m.group(1)}")
            rows.append((re.sub(r"^[0-9a-f]{8}-", "", f), ms.group(1) if ms else "?",
                         ms.group(2) if ms else "?", n, g.group(1) if g else "-", " ".join(flags)))
    rows.sort(key=lambda r: r[0])
    print(f"{'file':72} {'seed':>5} {'hash':>10} {'n':>3} {'grand':>7}  flags")
    for r in rows:
        print(f"{r[0]:72} {r[1]:>5} {r[2]:>10} {r[3]:>3} {r[4]:>7}  {r[5]}")
    print(f"\n{len(rows)} logs. Look for: Deep CORAL -> domain_adapt_lambda>0; Tent -> tta_mode=tent; "
          f"cardiac band-stop / fNIRS bandpass -> fnirs_bandpass or the run named in your notes.")


# =============================================================================
# holm
# =============================================================================
def load_cfg(a):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if a.config:
        with open(a.config, encoding="utf-8") as f:
            user = json.load(f)
        for k in ("baseline", "interventions", "extra_p"):
            if k in user:
                cfg[k] = user[k]
    for kv in a.extra_p or []:
        k, v = kv.split("=")
        cfg["extra_p"][k] = float(v)
    return cfg


def cmd_holm(a, quiet=False):
    cfg = load_cfg(a)
    root = a.logs_root
    base = {}
    for era in ("old", "new"):
        base[era] = {}
        for s, name in cfg["baseline"][era].items():
            p = find_file(root, name)
            if p is None:
                print(f"[warn] baseline {era} seed {s}: file not found ({name})")
                continue
            base[era][s] = read_acc(p)
    fam = []
    print("=" * 100)
    print(f"{'intervention':32} {'seeds':>9} {'vs':>5} {'base':>7} {'interv':>7} {'diff':>8} {'wins':>6} {'p':>8}")
    print("=" * 100)
    for name, files in cfg["interventions"].items():
        iv = {}
        for s, fname in files.items():
            p = find_file(root, fname)
            if p is None:
                print(f"[warn] {name} seed {s}: file not found ({fname})")
                continue
            iv[s] = read_acc(p)
        if not iv:
            continue
        seeds = sorted(iv)
        iv_mean = mean_dict([iv[s] for s in seeds])
        prim = None
        for lab in ("old", "new", "avg"):
            bl = []
            for s in seeds:
                if lab == "avg":
                    if s in base["old"] and s in base["new"]:
                        bl.append(mean_dict([base["old"][s], base["new"][s]]))
                elif s in base[lab]:
                    bl.append(base[lab][s])
            if len(bl) != len(seeds):
                continue
            bm = mean_dict(bl)
            d, w, n, p = paired(iv_mean, bm)
            bmean = float(np.mean(list(bm.values())))
            imean = float(np.mean([iv_mean[k] for k in bm]))
            print(f"{name:32} {','.join(seeds):>9} {lab:>5} {bmean:7.4f} {imean:7.4f} {d:+8.4f} {w:3d}/{n:<2} {p:8.4f}")
            if lab == "avg":
                prim = (name, d, w, n, p)
        if len(seeds) > 1:
            for s in seeds:
                if s in base["old"] and s in base["new"]:
                    bm = mean_dict([base["old"][s], base["new"][s]])
                    d, w, n, p = paired(iv[s], bm)
                    print(f"{'   seed ' + s:32} {s:>9} {'avg':>5} {np.mean(list(bm.values())):7.4f} "
                          f"{np.mean([iv[s][k] for k in bm]):7.4f} {d:+8.4f} {w:3d}/{n:<2} {p:8.4f}")
        if prim:
            fam.append(prim)
    for k, p in cfg["extra_p"].items():
        fam.append((k + " (supplied)", float("nan"), 0, 0, float(p)))
    if not fam:
        print("no complete intervention found -- check --logs-root / --config")
        return None
    adj = holm([f[4] for f in fam])
    print("\n" + "=" * 100)
    print("HOLM-BONFERRONI over the family below (primary test = seed-averaged per-subject accuracy vs the "
          "average of the OLD and NEW baseline era; two-sided Wilcoxon)")
    print("=" * 100)
    print(f"{'test':42} {'diff':>8} {'wins':>7} {'p_raw':>9} {'p_holm':>9}  verdict @0.05")
    out = []
    for (name, d, w, n, p), pa in sorted(zip(fam, adj), key=lambda x: x[0][4]):
        dd = f"{d:+8.4f}" if d == d else "       -"
        ww = f"{w:3d}/{n:<2}" if n else "      -"
        print(f"{name:42} {dd} {ww:>7} {p:9.5f} {pa:9.5f}  {'SIGNIFICANT' if pa < 0.05 else 'not significant'}")
        out.append((name, d, p, pa))
    print(f"\nfamily size m = {len(fam)}. A result that is significant raw but not after Holm must NOT be "
          f"called 'significant' in the thesis. Add every other tested intervention with --extra-p name=p "
          f"(or --config) so the family is complete; a smaller family makes Holm too lenient.")
    return out


# =============================================================================
# heads
# =============================================================================
def _norm(P):
    return P / np.clip(P.sum(-1, keepdims=True), 1e-12, None)


def _acc_win(P, y):
    return float((P.argmax(-1) == y).mean())


def _acc_trial(P, y, tid):
    c = 0
    ts = np.unique(tid)
    for t in ts:
        m = tid == t
        c += int(P[m].mean(0).argmax() == y[m][0])
    return c / len(ts)


def cmd_heads(a):
    files = sorted(glob.glob(os.path.join(a.posteriors_dir, "*_session*.npz")))
    if not files:
        sys.exit("no *.npz under --posteriors-dir (run ensemble_eval.py with --save-posteriors)")
    per = {}
    for f in files:
        sid = os.path.basename(f).split("_")[0]
        z = np.load(f)
        y, tid = z["y"], z["trial_ids"]
        Pf, Pe, Pu, Pn = (_norm(z["P_final"]), _norm(z["P_eeg"]), _norm(z["P_fusion"]), _norm(z["P_fnirs"]))
        eq = (Pe + Pu + Pn) / 3.0                     # FIXED equal weights, nothing learned or tuned
        K = Pf.shape[0]
        r = {}
        for lab, P in (("final", Pf), ("equal3", eq)):
            r[lab + "_single_win"] = np.mean([_acc_win(P[k], y) for k in range(K)])
            r[lab + "_single_tri"] = np.mean([_acc_trial(P[k], y, tid) for k in range(K)])
            r[lab + "_ens_win"] = _acc_win(P.mean(0), y)
            r[lab + "_ens_tri"] = _acc_trial(P.mean(0), y, tid)
        per.setdefault(sid, []).append(r)
    subs = sorted(per)
    M = {k: np.array([np.mean([x[k] for x in per[s]]) for s in subs]) for k in per[subs[0]][0]}
    print(f"{len(subs)} subjects, {len(files)} folds, seeds per fold = {K}")
    print(f"{'metric':22} {'final':>8} {'equal3':>8} {'diff':>8} {'wins':>7} {'p':>8}")
    for lvl, nm in (("single_win", "window, single model"), ("ens_win", "window, ensemble"),
                    ("single_tri", "block, single model"), ("ens_tri", "block, ensemble")):
        a_, b_ = M["final_" + lvl], M["equal3_" + lvl]
        d = b_ - a_
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"{nm:22} {a_.mean():8.4f} {b_.mean():8.4f} {d.mean():+8.4f} {int((d > 0).sum()):3d}/{len(d):<2} {p:8.4f}")
    print("\nequal3 = (eeg + fusion + fnirs head posteriors)/3, rows normalised, weights fixed a priori. "
          "No weight was chosen on test data. Report as a post-hoc decision-rule comparison, not a tuned result.")


# =============================================================================
# splitprobe
# =============================================================================
def _protocols(fe, ff, y, ses, tid, seed=0):
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    def mk():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))

    out = {}
    for lab, X in (("EEG", fe), ("fNIRS", ff), ("both", np.hstack([fe, ff]))):
        accs = {}
        a = []
        for s in np.unique(ses):
            tr, te = ses != s, ses == s
            m = mk().fit(X[tr], y[tr])
            a.append((m.predict(X[te]) == y[te]).mean())
        accs["A_cross_session"] = float(np.mean(a))
        b = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
            m = mk().fit(X[tr], y[tr])
            b.append((m.predict(X[te]) == y[te]).mean())
        accs["B_random_windows"] = float(np.mean(b))
        c = []
        for tr, te in GroupKFold(5).split(X, y, groups=tid):
            m = mk().fit(X[tr], y[tr])
            c.append((m.predict(X[te]) == y[te]).mean())
        accs["C_trial_grouped"] = float(np.mean(c))
        out[lab] = accs
    return out


def cmd_splitprobe(a):
    import check_feature_baseline_tsfnet as C
    rows = []
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        if not os.path.exists(os.path.join(d, f"{sid}_labels.npy")):
            continue
        fe, ff, y, ses = C.features(d, sid)
        tid = np.load(os.path.join(d, f"{sid}_trial_ids.npy"))
        r = _protocols(fe, ff, y, ses, tid)
        rows.append(r)
        print(f"{sid}  both: A {r['both']['A_cross_session']:.3f} | B {r['both']['B_random_windows']:.3f} | "
              f"C {r['both']['C_trial_grouped']:.3f}", flush=True)
    if not rows:
        sys.exit("no subjects found under --data")
    print("\nMEAN over subjects")
    print(f"{'features':8} {'A cross-session':>16} {'B random windows':>17} {'C trial-grouped':>16} {'B-A':>7} {'p(B>A)':>9}")
    for lab in ("EEG", "fNIRS", "both"):
        A = np.array([r[lab]["A_cross_session"] for r in rows])
        B = np.array([r[lab]["B_random_windows"] for r in rows])
        Cc = np.array([r[lab]["C_trial_grouped"] for r in rows])
        d = B - A
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"{lab:8} {A.mean():16.4f} {B.mean():17.4f} {Cc.mean():16.4f} {d.mean():+7.3f} {p:9.5f}")
    print("\nB uses random window folds: windows are 90% overlapping, so neighbours of a test window sit in the "
          "training fold (leakage). C keeps whole trials together. A is the project protocol. A large B-A (and "
          "C-A) means the split protocol alone can explain numbers far above cross-session accuracy.")


# =============================================================================
# selftest (planted effects)
# =============================================================================
def cmd_selftest(_a):
    ok = True
    # --- holm
    adj = holm([0.001, 0.04, 0.5, 0.03])
    exp = np.array([0.004, 0.09, 0.5, 0.09])
    good = np.allclose(adj, exp)
    print(f"[holm] adjusted {np.round(adj, 4).tolist()} expected {exp.tolist()} -> {'OK' if good else 'FAIL'}")
    ok &= good
    # --- holm / log parser on synthetic logs
    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as td:
        def wlog(name, accs):
            with open(os.path.join(td, name), "w", encoding="utf-8") as f:
                f.write("[ABLATION] domain_adapt_lambda=0.0 val_mode='trial'\n")
                for i, v in enumerate(accs, 1):
                    f.write(f"VP{i:03d}: acc={v:.4f} +- 0.01  kappa=0.3 |\n")
                f.write(f"Grand mean accuracy across subjects: {np.mean(accs):.4f} (+-0.0)\n")
        base = 0.6 + 0.08 * rng.randn(26)
        cfg = {"baseline": {"old": {}, "new": {}}, "interventions": {"planted_plus3": {}, "null": {}}}
        for s in SEEDS:
            wlog(f"b_old_{s}.txt", base + 0.005 * rng.randn(26))
            wlog(f"b_new_{s}.txt", base + 0.005 * rng.randn(26))
            wlog(f"plus3_{s}.txt", base + 0.03 + 0.005 * rng.randn(26))
            wlog(f"null_{s}.txt", base + 0.005 * rng.randn(26))
            cfg["baseline"]["old"][s] = f"b_old_{s}.txt"
            cfg["baseline"]["new"][s] = f"b_new_{s}.txt"
            cfg["interventions"]["planted_plus3"][s] = f"plus3_{s}.txt"
            cfg["interventions"]["null"][s] = f"null_{s}.txt"
        cp = os.path.join(td, "cfg.json")
        json.dump(cfg, open(cp, "w"))
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        ns = argparse.Namespace(logs_root=td, config=cp, extra_p=None)
        with redirect_stdout(buf):
            out = cmd_holm(ns)
        res = {n: (d, p, pa) for n, d, p, pa in out}
        good = res["planted_plus3"][2] < 0.05 and res["null"][2] > 0.05 and abs(res["planted_plus3"][0] - 0.03) < 0.01
        print(f"[holm pipeline] planted +0.03 -> p_holm {res['planted_plus3'][2]:.5f} (want <0.05), "
              f"null -> p_holm {res['null'][2]:.3f} (want >0.05) -> {'OK' if good else 'FAIL'}")
        ok &= good
    # --- heads: plant a case where the equal mean of 3 heads beats a noisy 'final'
    with tempfile.TemporaryDirectory() as td:
        for s in range(1, 4):
            for ses in (1, 2, 3):
                N = 120
                y = rng.randint(0, 3, N)
                tid = np.repeat(np.arange(N // 10), 10)
                y = np.repeat(rng.randint(0, 3, N // 10), 10)

                def head(snr):
                    L = rng.randn(2, N, 3)
                    L[:, np.arange(N), y] += snr
                    e = np.exp(L)
                    return (e / e.sum(-1, keepdims=True)).astype(np.float32)
                np.savez(os.path.join(td, f"VP{s:03d}_session{ses}.npz"), seeds=np.array([1, 2]), y=y,
                         trial_ids=tid, pos_in_trial=np.zeros(N, np.int32),
                         P_final=head(0.3), P_eeg=head(0.9), P_fusion=head(0.9), P_fnirs=head(0.9))
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            cmd_heads(argparse.Namespace(posteriors_dir=td))
        txt = buf.getvalue()
        m = re.search(r"window, single model\s+([\d.]+)\s+([\d.]+)", txt)
        good = bool(m) and float(m.group(2)) > float(m.group(1)) + 0.05
        print(f"[heads] weak 'final' vs equal3 of strong heads: {m.groups() if m else None} -> {'OK' if good else 'FAIL'}")
        ok &= good
    # --- splitprobe: overlapping windows + session shift => B >> A, C ~ A-ish
    N_per_trial, trials_per_session = 30, 9
    X, y, ses, tid = [], [], [], []
    t = 0
    base_means = rng.randn(3, 20)
    for s in (1, 2, 3):
        shift = 1.5 * rng.randn(20)
        for k in range(trials_per_session):
            cls = k % 3
            trial_offset = 0.8 * rng.randn(20)      # trial-specific nuisance shared by its windows
            for w in range(N_per_trial):
                X.append(0.3 * base_means[cls] + shift + trial_offset + 0.15 * rng.randn(20))
                y.append(cls); ses.append(s); tid.append(t)
            t += 1
    X, y, ses, tid = np.array(X), np.array(y), np.array(ses), np.array(tid)
    r = _protocols(X, X[:, :5], y, ses, tid)["both"]
    good = r["B_random_windows"] > r["A_cross_session"] + 0.2
    print(f"[splitprobe] A {r['A_cross_session']:.3f} B {r['B_random_windows']:.3f} C {r['C_trial_grouped']:.3f} "
          f"(planted overlap leak: want B-A > 0.2) -> {'OK' if good else 'FAIL'}")
    ok &= good
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("describe"); s.add_argument("--logs-root", required=True)
    s = sub.add_parser("holm"); s.add_argument("--logs-root", required=True)
    s.add_argument("--config"); s.add_argument("--extra-p", nargs="*")
    s = sub.add_parser("heads"); s.add_argument("--posteriors-dir", required=True)
    s = sub.add_parser("splitprobe"); s.add_argument("--data", required=True)
    s.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "describe":
        cmd_describe(a)
    elif a.cmd == "holm":
        cmd_holm(a)
    elif a.cmd == "heads":
        cmd_heads(a)
    elif a.cmd == "splitprobe":
        cmd_splitprobe(a)
    else:
        sys.exit(cmd_selftest(a))


if __name__ == "__main__":
    main()
