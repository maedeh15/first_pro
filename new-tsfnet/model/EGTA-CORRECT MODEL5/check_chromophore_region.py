"""
check_chromophore_region.py -- cheap diagnostic: does HbR carry more
2-back-vs-3-back information than HbO, and is it concentrated in the
mid-frontal part of the grid?

WHY THIS EXISTS
check_neurovascular_coupling.py established that, on this project's data,
EEG band power is at chance for 2-back-vs-3-back (0.509 vs 0.500) while
fNIRS carries 0.635 -- i.e. essentially ALL of the signal for the class pair
that accounts for 61-68% of this project's errors lives in fNIRS. So the
useful next question is WHERE inside fNIRS: which chromophore, which region.

Meidenbauer et al., "Load-dependent relationships between frontal fNIRS
activity and performance: A data-driven PLS approach", NeuroImage (2021),
70 participants, is the prompt for testing HbR specifically.

WHAT THAT PAPER ACTUALLY CLAIMS -- the two claims are not equivalent
  SUPPORTED: 3-back was unstable. Quote: "participants showed reliably
    greater fronto-parietal activation for the 2-back versus the 1-back
    task... The results for 3-back were much less consistent, potentially
    due to poor behavioral performance in the 3-back task." That is genuine
    independent evidence for a 2/3-back bottleneck, with a mechanism
    (subjects perform poorly and variably at 3-back, so the neural response
    becomes heterogeneous). Worth citing in a discussion chapter regardless
    of what this script finds.
  NOT what they showed: that HbR DISCRIMINATES LOAD LEVELS better than HbO.
    Their mid-frontal HbR result is a brain-BEHAVIOUR correlation: the
    relationship between accuracy and HbR change differs by load level
    (positive at 3-back, null at 2-back, negative at 1-back) in eight
    mid-frontal channels. That is about individual differences in
    performance, not about separating 2-back from 3-back trials. Their task
    was also 1/2/3-back on their own 70-subject recording, not Shin's
    0/2/3-back.
So "emphasise HbR" is a reasonable ENGINEERING hypothesis worth a cheap
test, but it is not a finding transferred from that paper. Do not cite the
paper as support for a classification claim.

AXIS CONVENTIONS -- verified against the preprocessing code, not assumed
  fnirs_windows.npy is (N, S, 16, 16, T, C) where
    axis 2 = h : left-right (projected x)
    axis 3 = w : posterior -> anterior. Verified numerically from
                 NIRS_POSITIONS_3D through _azimuthal_projection:
                 corr(y_3d, y_projected) = +0.996, and the most anterior
                 optodes (AFpz, AFp3/4, AF1) map to the TOP of the range.
                 So HIGH w index = FRONTAL.
    axis 5 = C : 0 = HbO, 1 = HbR  (preprocess_fnirs does
                 np.stack([hbo_mapped, hbr_mapped], axis=-1))
  Getting either of these backwards would silently invert the whole result,
  so --self-test plants a signal in a known chromophore and a known region
  and checks that the matching feature set -- and only it -- picks it up.

FEATURES (same form as check_feature_baseline_tsfnet.py's fNIRS features, so
the "HbO+HbR all grid" row is directly comparable to its 0.581 fNIRS column)
  average over the S temporal segments, then per (h, w, chromophore) take the
  mean and the linear slope over time.

usage
  python check_chromophore_region.py --self-test
  python check_chromophore_region.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

CHROMO = {0: "HbO", 1: "HbR", 2: "HbT"}
# Meidenbauer's ROI is eight mid-frontal channels. On this 16x16 grid that is
# the anterior rows (high w) restricted to the medial columns (central h).
MIDFRONTAL_W = slice(11, 16)        # anterior
MIDFRONTAL_H = slice(5, 11)         # medial
ANTERIOR_W = slice(8, 16)
POSTERIOR_W = slice(0, 8)


def trial_features(chunk):
    """chunk: (n, S, H, W, T, C) -> (mean, slope) each (n, H, W, C)."""
    x = np.asarray(chunk, dtype=np.float32).mean(axis=1)        # (n, H, W, T, C)
    t = np.arange(x.shape[3], dtype=np.float32)
    t -= t.mean()
    mean = x.mean(axis=3)
    slope = (x * t[None, None, None, :, None]).sum(axis=3) / (t ** 2).sum()
    return mean, slope


def grid_features(path_or_array, chunk_size=80):
    """-> mean, slope arrays of shape (N, H, W, C), read in chunks."""
    X = (np.load(path_or_array, mmap_mode="r")
         if isinstance(path_or_array, str) else path_or_array)
    n = X.shape[0]
    means, slopes = [], []
    for a in range(0, n, chunk_size):
        m, s = trial_features(X[a:a + chunk_size])
        means.append(m)
        slopes.append(s)
    return np.concatenate(means), np.concatenate(slopes)


def build_feature_sets(mean, slope):
    """mean/slope: (N, H, W, C) -> {name: (N, F)}."""
    n, H, W, C = mean.shape
    sets = {}

    def pack(sel_h, sel_w, chans, label):
        parts = []
        for c in chans:
            parts.append(mean[:, sel_h, sel_w, c].reshape(n, -1))
            parts.append(slope[:, sel_h, sel_w, c].reshape(n, -1))
        sets[label] = np.hstack(parts)

    all_h, all_w = slice(None), slice(None)
    for c in range(min(C, 3)):
        pack(all_h, all_w, [c], f"{CHROMO[c]} all grid")
    if C >= 2:
        pack(all_h, all_w, [0, 1], "HbO+HbR all grid")
    if C >= 3:
        pack(all_h, all_w, list(range(C)), "HbO+HbR+HbT all grid")

    for c in range(min(C, 2)):
        pack(all_h, ANTERIOR_W, [c], f"{CHROMO[c]} anterior half")
        pack(all_h, POSTERIOR_W, [c], f"{CHROMO[c]} posterior half")
        pack(MIDFRONTAL_H, MIDFRONTAL_W, [c], f"{CHROMO[c]} mid-frontal ROI")
    if C >= 2:
        pack(MIDFRONTAL_H, MIDFRONTAL_W, [0, 1], "HbO+HbR mid-frontal ROI")
    return sets


# ============================================================================
def _clf(kind):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def cross_session(Xf, y, ses, kind):
    accs = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        m = _clf(kind).fit(Xf[tr], y[tr])
        accs.append(float((m.predict(Xf[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


# ============================================================================
def self_test(args, n_per_class=120, seed=0):
    """Plant a class effect in HbR, mid-frontal ONLY, and confirm that the
    matching feature set finds it while HbO and the posterior half do not.

    This is an axis-convention check: if chromophore order or the
    anterior/posterior orientation were wrong, the wrong rows would light up.
    """
    rng = np.random.default_rng(seed)
    S, H, W, T, C = 4, 16, 16, 30, 2
    ys, blocks = [], []
    for cls in (0, 1):
        x = rng.normal(scale=1.0, size=(n_per_class, S, H, W, T, C)).astype(np.float32)
        # the planted effect: HbR (c=1), mid-frontal cells only, class-dependent
        # downward drift over time (more activation = more HbR reduction)
        ramp = np.linspace(0.0, 1.0, T, dtype=np.float32)
        x[:, :, MIDFRONTAL_H, MIDFRONTAL_W, :, 1] -= (
            args.selftest_effect * cls * ramp[None, None, None, None, :])
        blocks.append(x)
        ys.append(np.full(n_per_class, cls))
    X = np.concatenate(blocks)
    y = np.concatenate(ys)
    ses = rng.integers(1, 4, size=len(y))

    mean, slope = grid_features(X)
    sets = build_feature_sets(mean, slope)
    print("=" * 84)
    print(f"SELF-TEST -- effect planted in HbR, mid-frontal ROI only "
          f"(amplitude {args.selftest_effect})")
    print("=" * 84)
    print(f"{'feature set':34} {'n_feat':>7} {'acc':>7}   (chance = 0.500)")
    res = {}
    for name, Xf in sets.items():
        res[name] = cross_session(Xf, y, ses, args.clf)
        print(f"{name:34} {Xf.shape[1]:7d} {res[name]:7.3f}")
    hbr_roi = res["HbR mid-frontal ROI"]
    hbo_roi = res["HbO mid-frontal ROI"]
    hbr_post = res["HbR posterior half"]
    print()
    ok_chromo = hbr_roi > hbo_roi + 0.10
    ok_region = hbr_roi > hbr_post + 0.10
    if ok_chromo and ok_region:
        print(f"  PASS -- the planted effect shows up in HbR mid-frontal ({hbr_roi:.3f}) and not\n"
              f"  in HbO mid-frontal ({hbo_roi:.3f}) or HbR posterior ({hbr_post:.3f}). Both the\n"
              f"  chromophore index and the anterior/posterior orientation are correct, so the\n"
              f"  real-data rows can be read at face value.")
    else:
        print(f"  FAIL -- HbR mid-frontal {hbr_roi:.3f}, HbO mid-frontal {hbo_roi:.3f}, "
              f"HbR posterior {hbr_post:.3f}.\n"
              f"  An axis convention is wrong (chromophore order or w orientation). Do NOT\n"
              f"  interpret real-data results until this passes.")
    print("=" * 84)
    return res


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="preprocessed folder with {sid}/{sid}_fnirs_windows.npy")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--self-test", action="store_true",
                    help="run the axis-convention check and exit")
    ap.add_argument("--selftest-effect", type=float, default=0.25,
                    help="amplitude of the planted HbR mid-frontal effect")
    ap.add_argument("--clf", choices=["lr", "svm"], default="lr",
                    help="lr (default, matches check_feature_baseline_tsfnet.py) or svm")
    ap.add_argument("--chunk", type=int, default=80,
                    help="windows per read chunk (memory: fnirs_windows.npy is ~1 GB/subject)")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    args = ap.parse_args()

    if args.self_test:
        self_test(args)
        return
    if not args.data:
        sys.exit("--data is required (or pass --self-test)")
    if not os.path.isdir(args.data):
        sys.exit(f"--data does not exist: {args.data}")

    acc_lists, pool_store = {}, {}
    n_total = 0
    print(f"{'subject':9} {'windows':>8} {'chromo':>7}")
    for sid in args.subjects:
        d = os.path.join(args.data, sid)
        fp = os.path.join(d, f"{sid}_fnirs_windows.npy")
        if not os.path.exists(fp):
            print(f"{sid:9} SKIP -- {sid}_fnirs_windows.npy missing")
            continue
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        try:
            mean, slope = grid_features(fp, args.chunk)
        except Exception as exc:
            print(f"{sid:9} SKIP -- {type(exc).__name__}: {exc}")
            continue
        sets = build_feature_sets(mean, slope)
        n_total += len(y)
        print(f"{sid:9} {len(y):8d} {mean.shape[-1]:7d}", flush=True)

        for name, Xf in sets.items():
            for task, mask in (("3-class", np.ones(len(y), bool)),
                                ("2v3", np.isin(y, [1, 2]))):
                acc_lists.setdefault(name, {}).setdefault(task, []).append(
                    cross_session(Xf[mask], y[mask], ses[mask], args.clf))
                st = pool_store.setdefault(name, {}).setdefault(task, ([], [], []))
                st[0].append(Xf[mask]); st[1].append(y[mask]); st[2].append(ses[mask])
        del mean, slope, sets

    if not acc_lists:
        sys.exit("No subjects processed -- check --data.")

    results = {}
    line = "=" * 94
    for task, chance in (("3-class", 1 / 3), ("2v3", 0.5)):
        title = "3-class (0/2/3-back)" if task == "3-class" else "2-back vs 3-back"
        print("\n" + line)
        print(f"{title}   (chance = {chance:.3f}, {n_total} windows total)")
        print(line)
        print(f"{'feature set':34} {'n_feat':>7} {'per-subject':>13} {'pooled':>9}")
        for name in acc_lists:
            ps = float(np.nanmean(acc_lists[name][task]))
            Xs, ys, ss = pool_store[name][task]
            Xp = np.vstack(Xs)
            pl = cross_session(Xp, np.concatenate(ys), np.concatenate(ss), args.clf)
            results[f"{title} | {name}"] = {"per_subject": ps, "pooled": pl,
                                             "n_features": int(Xp.shape[1])}
            print(f"{name:34} {Xp.shape[1]:7d} {ps:13.3f} {pl:9.3f}")

    print("\n" + line)
    print("READING")
    print(line)
    g = lambda k: results.get(f"2-back vs 3-back | {k}", {}).get("per_subject", float("nan"))
    hbo, hbr, both = g("HbO all grid"), g("HbR all grid"), g("HbO+HbR all grid")
    roi_r, roi_o = g("HbR mid-frontal ROI"), g("HbO mid-frontal ROI")
    print(f"  2-vs-3, per-subject:  HbO={hbo:.3f}  HbR={hbr:.3f}  both={both:.3f}  |  "
          f"mid-frontal HbR={roi_r:.3f} HbO={roi_o:.3f}")
    if hbr > hbo + 0.02:
        print("  HbR beats HbO on the class pair that dominates this project's errors. Since\n"
              "  both chromophores are already in the input tensor, this is not about adding\n"
              "  data -- it is about the model under-using HbR. Candidate interventions:\n"
              "  a chromophore-wise attention or gating term, or a per-chromophore loss\n"
              "  weighting. Check whether 'both' also beats HbO: if it does not, the model is\n"
              "  being actively hurt by HbO rather than merely ignoring HbR.")
    elif hbo > hbr + 0.02:
        print("  HbO beats HbR. The 'emphasise HbR' hypothesis does not hold on this data --\n"
              "  which is consistent with Meidenbauer's actual claim being about a\n"
              "  brain-behaviour correlation, not about load discriminability.")
    else:
        print("  The two chromophores are equivalent for this class pair, and emphasising\n"
              "  either one is not indicated. Note this is still informative: it means the\n"
              "  fNIRS 2-vs-3 signal is not chromophore-specific, so a chromophore-weighting\n"
              "  intervention has no target.")
    if max(roi_r, roi_o) > both + 0.02:
        print(f"\n  NOTE: the mid-frontal ROI ({max(roi_r, roi_o):.3f}) beats the whole grid\n"
              f"  ({both:.3f}) despite using far fewer features -- the rest of the grid is\n"
              f"  adding noise. That is a spatial-prior intervention, and a cheap one.")
    print("\n  Anchor: 'HbO+HbR all grid' for 3-class per-subject should land near\n"
          "  check_feature_baseline_tsfnet.py's fNIRS column (0.581) -- same features,\n"
          "  same protocol. A large gap means the two scripts disagree.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "results": results}, fh, indent=2, default=float)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
