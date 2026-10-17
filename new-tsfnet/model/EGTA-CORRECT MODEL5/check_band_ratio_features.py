"""
check_band_ratio_features.py -- cheap diagnostic: do EEG alpha/theta BAND
RATIOS carry 2-back-vs-3-back information that ABSOLUTE band powers do not?

WHY THIS EXISTS
Raufi & Longo, "An Evaluation of the EEG Alpha-to-Theta and Theta-to-Alpha
Band Ratios as Indexes of Mental Workload", Frontiers in Neuroinformatics
16:861967 (2022), report that alpha-to-theta / theta-to-alpha ratio indexes
discriminate self-reported mental workload better than the individual bands.

WHAT THAT PAPER ACTUALLY SHOWS -- read before quoting it
  * Different dataset and task: STEW (48 subjects, 14-channel Emotiv, 128 Hz),
    resting vs. SIMKAP multitasking. Not n-back, not three load levels.
  * Different target: SELF-REPORTED workload ratings (1-9, binarised, 5s
    discarded), explicitly chosen INSTEAD of the task condition.
  * The widely-quoted "18.4-30.2% better than the individual bands" is a
    misreading, including by the paper's own Discussion. Those two numbers
    are the at-3 and at-1 entries of Table 5's "Avg. Increase" row, whose
    caption defines it as the gain from ADDING SYNTHETIC DATA, per index --
    not a ratio-vs-band comparison. Computed properly from the same table
    (Orig.+Synth. row), only 2 of the 6 ratio indexes beat their own
    components: at-1 +10.3/+21.7 and at-2 +17.6/+19.4, while at-3 (-5.7),
    ta-1 (-3.9) and ta-3 (-10.1) are WORSE than the band they came from.
  * On the real data alone the ratios reached ~70% and the single bands
    <=60%, with the paper reporting underfitting and poor generalisation.
    The headline ~84% appears only after adding 180 SYNTHETIC subjects
    generated from the same 48 real ones, split randomly against them --
    a leakage pathway, since the generator saw every real subject.
  * Correctly read: their Table 3 does show no significant difference
    between Logistic Regression, SVM and Decision Trees (p = 0.13-0.96).

SO WHY TEST IT HERE ANYWAY -- two reasons from THIS project's own data,
not from that paper:
  1. check_neurovascular_coupling.py found EEG band power to be at exactly
     chance for 2-back-vs-3-back (0.509 vs 0.500), while fNIRS carried
     0.635. Anything that lifts EEG above chance on that specific pair is
     new information for the fusion -- and that pair is 61-68% of this
     project's errors.
  2. A ratio is scale-invariant: log(alpha/theta) = log alpha - log theta
     cancels per-channel gain and session-level amplitude drift, which is
     exactly the nuisance a leave-one-session-out protocol suffers from.
     That is a mechanistic reason ratios might survive where the absolute
     5-band filterbank (+0.87 points, sub-threshold) did not.

THE KEY MECHANISM (worked out with the --self-test, not assumed)
It is tempting to argue that ratios add nothing to a linear model, since
log(alpha/theta) = log alpha - log theta already lies in the span of the log
powers. That argument is wrong in practice, and the self-test shows why.
A per-window amplitude gain g enters every log power as +log g. When
var(log g) dominates the class effect -- which is exactly the situation
under session-to-session amplitude drift -- the useful direction is a tiny
antisymmetric contrast sitting on top of a huge shared-variance direction.
An L2-regularised model shrinks toward zero and does not recover that
contrast, even though it is representable. Handing it the ratio directly
removes the nuisance by construction. Measured on planted data (see
--self-test defaults): absolute band power 0.528, ratio 0.914.
So ratios can genuinely win, and they win as a CONSTRAINT rather than as
extra capacity -- the one intervention pattern this project has had success
with. They can also be fed to the deep model as an extra input channel.

WHAT IS COMPARED (all on the EXISTING preprocessed data, no re-run needed)
  log power theta+alpha        per channel                      -> 2*C
  log power 5 bands            per channel                      -> 5*C
  log alpha/theta ratio        per channel                      -> C
  log theta/alpha ratio        per channel (= sign flip of above)
  Raufi cluster ratios         frontal theta vs parietal alpha  -> 3
  ratio + log power            concatenation
Evaluated under this project's cross-session protocol, for 3-class AND for
the 2-back-vs-3-back binary subtask, at WINDOW level -- so the "log power"
rows are directly comparable to check_feature_baseline_tsfnet.py's EEG
column (0.558 for 3-class, per-subject).

REQUIREMENTS
  {sid}_eeg_channel_windows.npy  (written when ENABLE_EEG_CHANNEL_SPACE=True)
  {sid}_labels.npy, {sid}_session_ids.npy
  {sid}_eeg_clean_raw.fif        (only for the channel NAMES, so the
                                  frontal/parietal clusters can be located;
                                  without it the cluster rows are skipped)

usage
  python check_band_ratio_features.py --self-test
  python check_band_ratio_features.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SFREQ_EEG = 200.0
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
# Raufi & Longo build the index from a frontal theta cluster and a parietal
# alpha cluster. Their electrodes come from a 14-channel Emotiv montage; these
# are the closest equivalents in THIS dataset's 28-channel montage.
FRONTAL_CLUSTERS = {
    "c1": ["Fp1", "Fp2", "AFF5h", "AFF6h", "AFz", "F1", "F2"],   # broad frontal
    "c2": ["F1", "F2"],                                            # narrow
    "c3": ["F1", "F2", "AFF5h", "AFF6h"],                          # intermediate
}
PARIETAL_CLUSTER = ["P7", "P8"]      # same electrodes as the paper's c-alpha


# ============================================================================
# band power
# ============================================================================
def band_powers(X, fs=SFREQ_EEG, chunk=200):
    """X: (N, C, T) -> {band: (N, C)} mean PSD in that band.

    Chunked over windows so the complex FFT of a full subject (~870 x 28 x
    2000) never materialises at once.
    """
    n, c, t = X.shape
    win = np.hanning(t)
    freqs = np.fft.rfftfreq(t, 1.0 / fs)
    masks = {b: (freqs >= lo) & (freqs < hi) for b, (lo, hi) in BANDS.items()}
    out = {b: np.empty((n, c), dtype=np.float64) for b in BANDS}
    for a in range(0, n, chunk):
        z = np.asarray(X[a:a + chunk], dtype=np.float64)
        z = z - z.mean(axis=-1, keepdims=True)
        psd = np.abs(np.fft.rfft(z * win, axis=-1)) ** 2
        for b, m in masks.items():
            out[b][a:a + chunk] = psd[..., m].mean(axis=-1)
    return out


def _log(p):
    return np.log(p + 1e-20)


def build_feature_sets(X, ch_names):
    """-> dict of {name: (N, F) feature matrix}."""
    P = band_powers(X)
    lp = {b: _log(P[b]) for b in BANDS}

    sets = {
        "log power theta+alpha": np.hstack([lp["theta"], lp["alpha"]]),
        "log power 5 bands": np.hstack([lp[b] for b in BANDS]),
        "log alpha/theta (per ch)": lp["alpha"] - lp["theta"],
        "log theta/alpha (per ch)": lp["theta"] - lp["alpha"],
        "ratio + log power th+al": np.hstack(
            [lp["alpha"] - lp["theta"], lp["theta"], lp["alpha"]]),
    }

    if ch_names is not None:
        idx = {c: i for i, c in enumerate(ch_names)}
        par = [idx[c] for c in PARIETAL_CLUSTER if c in idx]
        cols = []
        for cname, chans in FRONTAL_CLUSTERS.items():
            fro = [idx[c] for c in chans if c in idx]
            if not fro or not par:
                continue
            # paper: at = avg(alpha over parietal) / avg(theta over frontal)
            at = _log(P["alpha"][:, par].mean(1)) - _log(P["theta"][:, fro].mean(1))
            cols.append(at)
        if cols:
            sets["Raufi cluster ratios"] = np.column_stack(cols)
            sets["cluster ratios + per-ch ratio"] = np.hstack(
                [np.column_stack(cols), lp["alpha"] - lp["theta"]])
    return sets


# ============================================================================
# evaluation (same protocol as check_feature_baseline_tsfnet.py)
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
# loading
# ============================================================================
def load_subject(data_root, sid):
    d = os.path.join(data_root, sid)
    win = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
    if not os.path.exists(win):
        return None
    X = np.load(win, mmap_mode="r")                      # (N, C, T)
    y = np.load(os.path.join(d, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))

    ch_names = None
    fif = os.path.join(d, f"{sid}_eeg_clean_raw.fif")
    if os.path.exists(fif):
        try:
            import mne
            raw = mne.io.read_raw_fif(fif, preload=False, verbose=False)
            picks = mne.pick_types(raw.info, eeg=True)
            ch_names = [raw.ch_names[i] for i in picks]
            if len(ch_names) != X.shape[1]:
                ch_names = None          # order/count mismatch -> don't guess
        except Exception:
            ch_names = None
    return X, y, ses, ch_names


# ============================================================================
# instrument self-test
# ============================================================================
def _narrowband(n, t, fs, lo, hi, rng):
    x = rng.normal(size=(n, t))
    b, a = butter(4, [lo / (fs / 2), hi / (fs / 2)], btype="band")
    x = filtfilt(b, a, x, axis=-1)
    return x / (x.std(axis=-1, keepdims=True) + 1e-12)


def self_test(args, n_per_class=150, n_ch=8, seed=0):
    """Plant a RATIO-ONLY effect under a large per-window gain.

    theta power scales up with class, alpha power scales down, and both are
    multiplied by the same random per-window gain g. log(alpha/theta)
    cancels g exactly, while log theta and log alpha individually sit on top
    of var(log g). The planted class effect is deliberately SMALL relative
    to that gain (0.07 vs SD 1.5 in log units) so the run is not saturated
    and the two feature families actually separate -- which is the regime
    session drift creates on real data.
    """
    rng = np.random.default_rng(seed)
    fs, t = SFREQ_EEG, 2000
    ys, blocks = [], []
    for cls in (0, 1):
        shift = 1.0 + args.selftest_shift * cls
        th = _narrowband(n_per_class * n_ch, t, fs, 4.0, 8.0, rng).reshape(n_per_class, n_ch, t)
        al = _narrowband(n_per_class * n_ch, t, fs, 8.0, 13.0, rng).reshape(n_per_class, n_ch, t)
        g = np.exp(rng.normal(0.0, args.selftest_gain_sd, size=(n_per_class, 1, 1)))
        sig = g * (shift * th + (1.0 / shift) * al)
        sig = sig + 2.0 * g * rng.normal(size=sig.shape)       # broadband background
        blocks.append(sig)
        ys.append(np.full(n_per_class, cls))
    X = np.concatenate(blocks).astype(np.float32)
    y = np.concatenate(ys)
    ses = rng.integers(1, 4, size=len(y))                       # 3 pseudo-sessions

    sets = build_feature_sets(X, None)
    print("=" * 82)
    print(f"SELF-TEST -- ratio-only effect: class shift {args.selftest_shift} under a "
          f"per-window\ngain of SD {args.selftest_gain_sd} (log units), "
          f"{n_per_class} windows/class, {n_ch} channels")
    print("=" * 82)
    print(f"{'feature set':30} {'cross-session acc':>18}   (chance = 0.500)")
    res = {}
    for name, Xf in sets.items():
        acc = cross_session(Xf, y, ses, args.clf)
        res[name] = acc
        print(f"{name:30} {acc:18.3f}")
    r = res["log alpha/theta (per ch)"]
    p = res["log power theta+alpha"]
    print()
    if r > 0.70 and r > p:
        print(f"  PASS -- the ratio recovers the planted effect ({r:.3f}) and beats absolute\n"
              f"  power ({p:.3f}). The instrument can detect a ratio-only advantage, so a null\n"
              f"  on real data is meaningful.")
    elif r > 0.70:
        print(f"  PARTIAL -- the ratio recovers the effect ({r:.3f}) but absolute power does\n"
              f"  too ({p:.3f}); at this gain SD the linear model can already undo the\n"
              f"  nuisance. Raise --selftest-gain-sd to separate them.")
    else:
        print(f"  FAIL -- the ratio does not recover a planted ratio-only effect ({r:.3f}).\n"
              f"  Do not interpret a null on real data; the feature pipeline is broken.")
    print("=" * 82)
    return res


# ============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="preprocessed folder with {sid}/{sid}_eeg_channel_windows.npy")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--self-test", action="store_true",
                    help="run the planted-ratio instrument check and exit")
    ap.add_argument("--selftest-gain-sd", type=float, default=1.5,
                    help="SD of the log per-window nuisance gain used by --self-test")
    ap.add_argument("--selftest-shift", type=float, default=0.07,
                    help="planted per-class band shift used by --self-test; kept small on "
                         "purpose so the run is not saturated")
    ap.add_argument("--clf", choices=["lr", "svm"], default="lr",
                    help="lr (default, matches check_feature_baseline_tsfnet.py) or svm")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    args = ap.parse_args()

    if args.self_test:
        self_test(args)
        return
    if not args.data:
        sys.exit("--data is required (or pass --self-test)")
    if not os.path.isdir(args.data):
        sys.exit(f"--data does not exist: {args.data}")

    per_subject = {}      # {set_name: {task: [acc per subject]}}
    pooled = {}           # {set_name: {task: (X, y, ses) accumulators}}
    acc_lists, pool_store = {}, {}
    n_win_total, no_names = 0, []

    print(f"{'subject':9} {'windows':>8}")
    for sid in args.subjects:
        got = load_subject(args.data, sid)
        if got is None:
            print(f"{sid:9} SKIP -- {sid}_eeg_channel_windows.npy missing")
            continue
        X, y, ses, ch_names = got
        if ch_names is None:
            no_names.append(sid)
        sets = build_feature_sets(X, ch_names)
        n_win_total += len(y)
        print(f"{sid:9} {len(y):8d}", flush=True)

        for name, Xf in sets.items():
            for task, mask in (("3-class", np.ones(len(y), bool)),
                                ("2v3", np.isin(y, [1, 2]))):
                acc_lists.setdefault(name, {}).setdefault(task, []).append(
                    cross_session(Xf[mask], y[mask], ses[mask], args.clf))
                st = pool_store.setdefault(name, {}).setdefault(task, ([], [], []))
                st[0].append(Xf[mask]); st[1].append(y[mask]); st[2].append(ses[mask])

    if not acc_lists:
        sys.exit("No subjects processed -- check --data.")
    if no_names:
        print(f"\nNOTE: channel names unavailable for {len(no_names)} subject(s) "
              f"({', '.join(no_names[:4])}{'...' if len(no_names) > 4 else ''}) -- "
              f"the cluster-ratio rows are computed only from subjects that had them.")

    results = {}
    line = "=" * 92
    for task, chance in (("3-class", 1 / 3), ("2v3", 0.5)):
        print("\n" + line)
        title = "3-class (0/2/3-back)" if task == "3-class" else "2-back vs 3-back"
        print(f"{title}   (chance = {chance:.3f}, {n_win_total} windows total)")
        print(line)
        print(f"{'feature set':32} {'n_feat':>7} {'per-subject':>13} {'pooled':>9}")
        for name in acc_lists:
            if task not in acc_lists[name]:
                continue
            ps = float(np.nanmean(acc_lists[name][task]))
            Xs, ys, ss = pool_store[name][task]
            Xp, yp, sp = np.vstack(Xs), np.concatenate(ys), np.concatenate(ss)
            pl = cross_session(Xp, yp, sp, args.clf)
            results[f"{title} | {name}"] = {"per_subject": ps, "pooled": pl,
                                             "n_features": int(Xp.shape[1])}
            print(f"{name:32} {Xp.shape[1]:7d} {ps:13.3f} {pl:9.3f}")

    print("\n" + line)
    print("READING")
    print(line)
    base = results.get("2-back vs 3-back | log power theta+alpha", {}).get("per_subject", float("nan"))
    rat = results.get("2-back vs 3-back | log alpha/theta (per ch)", {}).get("per_subject", float("nan"))
    comb = results.get("2-back vs 3-back | ratio + log power th+al", {}).get("per_subject", float("nan"))
    print(f"  2-vs-3, per-subject cross-session:  abs power={base:.3f}  "
          f"ratio={rat:.3f}  ratio+power={comb:.3f}")
    if max(rat, comb) > base + 0.02:
        print("  The ratio beats absolute band power on the class pair that dominates this\n"
              "  project's errors. Since a linear model on log powers could already form the\n"
              "  ratio, the gain is coming from the CONSTRAINT (fewer features, nuisance gain\n"
              "  removed structurally) -- which is the one intervention pattern this project\n"
              "  has had success with. Worth wiring into the model as an extra input.")
    elif abs(max(rat, comb) - base) <= 0.02:
        print("  The ratio neither helps nor hurts versus absolute band power. Given the\n"
              "  caveat that a linear model can already form log(alpha/theta) from log powers,\n"
              "  this is the expected outcome and a genuine null: the ratio carries no\n"
              "  information the bands do not, and its regularising effect is not worth a\n"
              "  model change here.")
    else:
        print("  The ratio is WORSE than absolute band power -- discarding the absolute levels\n"
              "  throws away usable signal. No case for this direction.")
    print("\n  Anchor: the 'log power 5 bands' 3-class per-subject number should land near\n"
          "  check_feature_baseline_tsfnet.py's EEG column (0.558), which used broadband\n"
          "  log-variance. A large gap means the two scripts disagree and one is wrong.")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "results": results}, fh, indent=2, default=float)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
