"""
map_fnirs_prior.py -- map WHERE the fNIRS 2-back-vs-3-back signal sits, so a
spatial/chromophore prior can be set deliberately instead of guessed.

WHY THIS EXISTS
check_chromophore_region.py found, per-subject cross-session, 2-vs-3:
    HbO anterior half (256 feat)   0.601   <- best
    HbO mid-frontal    (60 feat)   0.595
    HbO all grid      (512 feat)   0.585
    HbO+HbR all grid (1024 feat)   0.579   <- what the model currently gets
i.e. RESTRICTING the input improves the hard class pair, while for 3-class
the full grid with both chromophores was best (0.581). Before spending a
4-hour training run on a prior, this script answers three questions:
  1. along the anterior-posterior axis (w), where exactly is the signal?
  2. where should the cut go -- and is there a broad PLATEAU (robust) or a
     single spike (noise)?
  3. what does the restriction COST on 3-class? If HbO-only or
     anterior-only hurts 0-back separation, the prior must be a soft
     weighting, not a hard mask.

HOW TO READ THE SWEEPS -- the selection-bias trap
The best cut is chosen on the same 26 subjects it is scored on, so its
margin is optimistically biased. Do NOT take the argmax at face value. What
matters is whether neighbouring cuts agree: a plateau of several adjacent
cuts all beating the baseline is evidence; one cut spiking while its
neighbours sit at baseline is noise. The tables print every cut for exactly
this reason.

Effect sizes here are small (~0.02), so every decision-relevant comparison
also gets PAIRED statistics across the 26 subjects: mean paired difference,
how many subjects improve, and a Wilcoxon signed-rank p-value. That is the
project's established bar, and it is what turns "suggestive" into something
quotable.

AXIS CONVENTIONS (verified in check_chromophore_region.py's --self-test)
  fnirs_windows.npy is (N, S, 16, 16, T, C):  axis 3 = w, HIGH w = FRONTAL;
  axis 5: C=0 is HbO, C=1 is HbR.

usage
  python map_fnirs_prior.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def grid_mean_slope(path, chunk=80):
    """fnirs_windows.npy -> (mean, slope), each (N, H, W, C)."""
    X = np.load(path, mmap_mode="r")
    means, slopes = [], []
    for a in range(0, X.shape[0], chunk):
        x = np.asarray(X[a:a + chunk], dtype=np.float32).mean(axis=1)   # (n,H,W,T,C)
        t = np.arange(x.shape[3], dtype=np.float32)
        t -= t.mean()
        means.append(x.mean(axis=3))
        slopes.append((x * t[None, None, None, :, None]).sum(axis=3) / (t ** 2).sum())
    return np.concatenate(means), np.concatenate(slopes)


def feats(mean, slope, h, w, chans):
    n = mean.shape[0]
    parts = []
    for c in chans:
        parts.append(mean[:, h, w, c].reshape(n, -1))
        parts.append(slope[:, h, w, c].reshape(n, -1))
    return np.hstack(parts)


def cross_session(Xf, y, ses):
    accs = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=5000)).fit(Xf[tr], y[tr])
        accs.append(float((m.predict(Xf[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def paired_report(name_a, name_b, per_sub):
    """Paired comparison of two definitions across subjects."""
    a = np.asarray(per_sub[name_a], dtype=float)
    b = np.asarray(per_sub[name_b], dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    better = int((d > 0).sum())
    try:
        p = float(wilcoxon(a, b).pvalue)
    except Exception:
        p = float("nan")
    return {"mean_diff": float(d.mean()), "n_better": better, "n": int(len(d)), "p": p}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--chunk", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not os.path.isdir(args.data):
        sys.exit(f"--data does not exist: {args.data}")

    # ---- one pass over the big .npy files, keep only the small grids
    store = []
    print(f"{'subject':9} {'windows':>8}")
    for sid in args.subjects:
        d = os.path.join(args.data, sid)
        fp = os.path.join(d, f"{sid}_fnirs_windows.npy")
        if not os.path.exists(fp):
            print(f"{sid:9} SKIP -- fnirs_windows.npy missing")
            continue
        mean, slope = grid_mean_slope(fp, args.chunk)
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        store.append((sid, mean, slope, y, ses))
        print(f"{sid:9} {len(y):8d}", flush=True)
    if not store:
        sys.exit("No subjects processed.")
    H, W, C = store[0][1].shape[1:]

    # ---- the definitions to sweep
    allh = slice(None)
    defs = {
        "HbO+HbR all grid  [model input]": (allh, slice(None), [0, 1]),
        "HbO all grid": (allh, slice(None), [0]),
        "HbR all grid": (allh, slice(None), [1]),
    }
    for k in range(0, W, 4):                       # 4 posterior->anterior bands
        lbl = f"w[{k}:{k + 4}]"
        defs[f"HbO band {lbl}"] = (allh, slice(k, k + 4), [0])
        defs[f"HbR band {lbl}"] = (allh, slice(k, k + 4), [1])
    for k in range(0, W - 1, 2):                   # cumulative anterior cuts
        defs[f"HbO w>={k:2d} (anterior)"] = (allh, slice(k, W), [0])
        defs[f"HbO+HbR w>={k:2d}"] = (allh, slice(k, W), [0, 1])
    for k in range(0, H, 4):                       # left-right bands, anterior HbO
        defs[f"HbO h[{k}:{k + 4}] w>=8"] = (slice(k, k + 4), slice(8, W), [0])

    # ---- per-subject accuracies for every definition and both tasks
    per_sub = {"2v3": {}, "3-class": {}}
    nfeat = {}
    for sid, mean, slope, y, ses in store:
        for name, (h, w, ch) in defs.items():
            Xf = feats(mean, slope, h, w, ch)
            nfeat[name] = Xf.shape[1]
            for task, mask in (("3-class", np.ones(len(y), bool)),
                                ("2v3", np.isin(y, [1, 2]))):
                per_sub[task].setdefault(name, []).append(
                    cross_session(Xf[mask], y[mask], ses[mask]))

    base = "HbO+HbR all grid  [model input]"
    line = "=" * 88
    print("\n" + line)
    print(f"PER-SUBJECT CROSS-SESSION ACCURACY, {len(store)} subjects")
    print("chance: 2v3 = 0.500, 3-class = 0.333.  Delta = 2v3 minus the model-input row.")
    print(line)
    print(f"{'definition':34} {'n_feat':>7} {'2v3':>7} {'3-class':>8} {'d(2v3)':>8} {'n_up':>6}")
    base_2v3 = float(np.nanmean(per_sub["2v3"][base]))
    rows = {}
    for name in defs:
        a2 = float(np.nanmean(per_sub["2v3"][name]))
        a3 = float(np.nanmean(per_sub["3-class"][name]))
        st = paired_report(name, base, per_sub["2v3"])
        rows[name] = {"acc_2v3": a2, "acc_3class": a3, "n_features": nfeat[name],
                      "vs_model_input_2v3": st}
        mark = "" if name == base else f"{a2 - base_2v3:+8.3f} {st['n_better']:4d}/{st['n']}"
        print(f"{name:34} {nfeat[name]:7d} {a2:7.3f} {a3:8.3f} {mark}")

    # ---- paired statistics for the decision-relevant comparisons
    print("\n" + line)
    print("PAIRED STATISTICS vs the current model input (Wilcoxon signed-rank)")
    print(line)
    cands = [n for n in defs if n != base]
    cands.sort(key=lambda n: -rows[n]["acc_2v3"])
    print(f"{'definition':34} {'task':>8} {'mean d':>8} {'n_up':>7} {'p':>9}")
    stats_out = {}
    for name in cands[:6] + ["HbO all grid", "HbR all grid"]:
        if name in stats_out:
            continue
        for task in ("2v3", "3-class"):
            st = paired_report(name, base, per_sub[task])
            stats_out[f"{name} | {task}"] = st
            print(f"{name if task == '2v3' else '':34} {task:>8} "
                  f"{st['mean_diff']:+8.3f} {st['n_better']:4d}/{st['n']} {st['p']:9.4f}")

    # ---- plateau check on the cumulative anterior cuts
    print("\n" + line)
    print("PLATEAU CHECK -- cumulative anterior cuts, HbO only")
    print(line)
    cuts = [(k, rows[f"HbO w>={k:2d} (anterior)"]["acc_2v3"]) for k in range(0, W - 1, 2)]
    for k, a in cuts:
        bar = "#" * max(0, int(round((a - 0.50) * 200)))
        flag = " <- beats model input" if a > base_2v3 else ""
        print(f"  w>={k:2d}  2v3={a:.3f}  {bar}{flag}")
    wins = [k for k, a in cuts if a > base_2v3]
    print()
    if len(wins) >= 3 and max(wins) - min(wins) >= 4:
        print(f"  PLATEAU: cuts {wins} all beat the model input, over a contiguous-ish range.\n"
              f"  That is robust to where exactly the boundary goes, so a SOFT anterior\n"
              f"  weighting is the right form -- no need to pick an exact cut, and no risk of\n"
              f"  having tuned the boundary to these 26 subjects.")
    elif len(wins) >= 1:
        print(f"  NO PLATEAU: only cuts {wins} beat the model input. A single winning cut\n"
              f"  surrounded by non-winners is what selection noise looks like. Do not build\n"
              f"  a prior around one boundary; treat this as a null unless the paired p for\n"
              f"  that cut is convincing on its own.")
    else:
        print("  No anterior cut beats the current model input. The earlier 0.601 was within\n"
              "  the noise of this sweep -- no spatial prior is indicated.")

    print("\n" + line)
    print("READING")
    print(line)
    best = cands[0]
    bst = rows[best]["vs_model_input_2v3"]
    cost = paired_report(best, base, per_sub["3-class"])
    print(f"  best 2v3 definition: {best}")
    print(f"    2v3     {rows[best]['acc_2v3']:.3f} vs {base_2v3:.3f}  "
          f"(d={bst['mean_diff']:+.3f}, {bst['n_better']}/{bst['n']} subjects up, p={bst['p']:.4f})")
    print(f"    3-class {rows[best]['acc_3class']:.3f} vs "
          f"{float(np.nanmean(per_sub['3-class'][base])):.3f}  "
          f"(d={cost['mean_diff']:+.3f}, p={cost['p']:.4f})")
    if bst["p"] < 0.05 and cost["mean_diff"] < -0.01:
        print("  => The restriction helps 2-vs-3 but COSTS 3-class. So a hard mask is wrong:\n"
              "     the prior must be a SOFT weighting the model can learn to override, e.g. an\n"
              "     anterior/chromophore gate initialised toward the anterior-HbO side, with\n"
              "     the full grid still reachable.")
    elif bst["p"] < 0.05:
        print("  => The restriction helps 2-vs-3 at no measurable 3-class cost. A hard\n"
              "     restriction of the fNIRS input is defensible, and it also shrinks the\n"
              "     input tensor, which is a capacity reduction rather than an addition.")
    else:
        print("  => Not significant across subjects. The ordering seen in\n"
              "     check_chromophore_region.py does not survive paired testing at this\n"
              "     effect size, so do not spend a training run on it. Record the chromophore\n"
              "     result (HbO > HbR, consistent 4/4) which was the sturdier finding.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "rows": rows, "paired": stats_out},
                      fh, indent=2, default=float)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
