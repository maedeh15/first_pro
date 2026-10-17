"""
check_neurovascular_coupling.py -- cheap diagnostic: does DIRECTED EEG-fNIRS
neurovascular coupling carry 2-back-vs-3-back information that the separate
per-modality features do NOT?

WHY THIS EXISTS
Gao et al., "Classification of working memory loads via assessing broken
detailed balance of EEG-fNIRS neurovascular coupling measures", IEEE TBME
70(3):877-887, 2023, report 82.48% 3-class accuracy on THIS EXACT dataset
(Shin et al. 2018, 26 subjects, 0-/2-/3-back) using: permutation transfer
entropy (PTE) for bidirectional coupling in theta/alpha/beta -> generalized
relative entropy (GRE) of the coarse-grained coupling state transitions ->
SVM.

PROTOCOL WARNING -- verified against the paper's own Sec. II-H, quote:
  "The selected GREs of each series were concatenated into feature vectors
   and ALL SERIES OF ALL SUBJECTS FORMED THE SAMPLE SET. Support vector
   machine (SVM) was chosen as the classifier and 5-FOLD CROSS VALIDATION
   was implemented."
Their 82.48% therefore comes from RANDOM 5-fold CV over a pool that mixes
subjects AND sessions across train/test. It is neither cross-session nor
cross-subject, so it is NOT comparable to this project's
leave-one-session-out numbers -- do not quote the two side by side without
saying so. This script evaluates under THIS PROJECT's cross-session
protocol, so every comparison it prints is internal and self-consistent.

WHAT THIS SCRIPT TESTS
On the SAME trials, under the SAME cross-session split: do coupling features
beat plain per-modality power features at separating 2-back from 3-back --
the class pair responsible for ~61-68% of this project's errors? If coupling
adds nothing, the direction is closed cheaply. If it adds something, a
model-level coupling branch is justified.

!! READ THIS BEFORE TRUSTING ANY NULL RESULT !!
Run --self-test first. It plants a known, strong EEG->fNIRS coupling in
synthetic trials and checks that each feature family recovers it. Measured
behaviour of this implementation:
  * PTE magnitude / asymmetry features  -- PASS, separate planted coupling
    from no coupling at p < 1e-4.
  * GRE features (Gao's own quantity)   -- FAIL at Gao's 5 s window. GRE
    discards coupling MAGNITUDE by construction: it measures only the
    temporal irreversibility of the clustered coupling-state trajectory, and
    at 5 s / 50 samples per window with m=3 (216 joint symbol states) the
    per-window PTE estimates are dominated by estimator bias, so the
    clustering is essentially arbitrary and every trial returns GRE ~ 0.45
    regardless of condition.
  => A null from the GRE rows alone means nothing. Judge this direction on
     the PTE magnitude/asymmetry rows, which are instrument-validated.
This is also why the defaults below (--pte-m 2, --win 20) depart from Gao's
(m unstated, 5 s): m=2 has 8 joint states instead of 216 and a 20 s window
has 200 samples instead of 50, which is what makes the estimate usable here.
Pass --pte-m 3 --win 5 to reproduce the Gao-faithful (and, per the
self-test, insensitive) configuration.

PIPELINE (per subject, per trial)
  1. EEG side (Gao's "expected hemoglobin concentration change" regressor):
     prefrontal electrodes -> band-pass (theta/alpha/beta) -> first temporal
     PC across those channels -> Hilbert power envelope -> decimated to the
     fNIRS rate -> convolved with the canonical HRF.
     DEVIATION, stated explicitly: Gao convolve only the detected power
     PEAKS/latencies with the HRF; this convolves the full envelope, the
     standard EEG-informed-regressor form and a superset of peaks-only.
  2. fNIRS side: continuous channel-space HbO/HbR from the preprocessed
     .fif, epoched on the fNIRS DEVICE'S OWN trial markers (the two devices
     are independently clocked and drift 46-93 s across sessions on this
     dataset -- using EEG's clock for fNIRS would destroy any coupling
     estimate), then baseline-corrected on [-5, -2] s as elsewhere here.
  3. Coupling, per sliding window: PTE in BOTH directions
     (EEG->fNIRS "forward", fNIRS->EEG "backward").
  4. Two feature families per (band, chromophore):
       PTE   : mean forward, mean backward, forward-minus-backward asymmetry
               -> 3 bands x 2 chromophores x 3 = 18 features   [VALIDATED]
       GRE   : k-means coarse-grain the per-window bidirectional coupling
               vectors of that trial, then GRE of the state-transition
               matrix -> 6 features                            [SEE WARNING]
     Clustering is per-trial, so nothing crosses trials.
  5. The same trials also get plain baselines: per-band per-channel EEG log
     power, and per-channel fNIRS mean+slope. This is what makes the
     comparison apples-to-apples at TRIAL granularity.
     (check_feature_baseline_tsfnet.py works on ~870 sliding WINDOWS per
     subject; its numbers are NOT comparable to this script's 27 trials per
     subject. Compare only the rows this script prints, against each other.)

EVALUATION
  pooled cross-session      : train on every subject's sessions != s, test
                              on session == s, averaged over s. The headline
                              number -- still strictly cross-session, but
                              ~468 training trials instead of 18, so it can
                              actually resolve a small effect.
  per-subject cross-session : this project's own protocol, averaged over
                              subjects. Honest but very noisy here: 18 train
                              / 9 test trials per fold.
Both for 3-class AND for the 2-back-vs-3-back binary subtask.

REQUIREMENTS
  - a preprocessed folder with, per subject, {sid}_eeg_clean_raw.fif and
    {sid}_fnirs_clean_raw.fif (both written by Preproc_tsfnet_nback_badch.py;
    the fNIRS one is pre-normalization, in physical concentration units,
    which is what we want)
  - the ORIGINAL dataset folder, for both devices' trial markers
    ({sid}-EEG/mrk_nback.mat and {sid}-NIRS/mrk_nback.mat)

usage
  python check_neurovascular_coupling.py --self-test
  python check_neurovascular_coupling.py ^
      --data "E:\\kntu\\Thesis\\codes\\Results\\TSFNET_PAPER_EXACT_nback" ^
      --dataset "E:\\kntu\\Thesis\\codes\\DATA" --subjects VP001 VP002
  python check_neurovascular_coupling.py --data ... --dataset ...
"""
import argparse
import json
import math
import os
import sys
import warnings
from itertools import permutations

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import gamma as _gamma_dist
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

SFREQ_FNIRS = 10.0
EEG_TRIAL_LABEL_MAP = {112: 0, 128: 1, 144: 2}      # 0-/2-/3-back, EEG clock
FNIRS_TRIAL_LABEL_MAP = {7: 0, 8: 1, 9: 2}          # same trials, fNIRS clock
TRIALS_PER_SESSION = 9
BASELINE_TMIN, BASELINE_TMAX = -5.0, -2.0           # project convention
BANDS = {"theta": (4.0, 8.0), "alpha": (8.0, 13.0), "beta": (13.0, 30.0)}
# Gao build the regressor from 7 prefrontal EEG channels; these are exactly
# the 7 prefrontal electrodes in this dataset's montage.
FRONTAL_CHANNELS = ["Fp1", "Fp2", "AFF5h", "AFF6h", "AFz", "F1", "F2"]


# ============================================================================
# markers (both device clocks)
# ============================================================================
def _load_struct(mat_path):
    """Same loader convention as Preproc_tsfnet_nback_badch.py."""
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    return mat[real_keys[0]]


def load_trial_onsets(dataset_root, sid):
    """Trial onsets in SECONDS on each device's own clock.

    Returns (eeg_times, fnirs_times, labels, session_ids, synced). If the
    fNIRS markers cannot be matched 1:1 to the EEG ones, fnirs_times falls
    back to the EEG clock and `synced` is False -- the same fallback as the
    preprocessing script, and the caller warns, because the cross-device
    drift on this dataset is 46-93 s.
    """
    mrk_eeg = _load_struct(os.path.join(dataset_root, f"{sid}-EEG", "mrk_nback.mat"))
    if not hasattr(mrk_eeg, "time"):
        raise ValueError(f"{sid}: EEG marker struct has no 'time' field")
    eeg_t = np.asarray(mrk_eeg.time, dtype=float) / 1000.0
    codes = np.asarray(mrk_eeg.event.desc, dtype=int).flatten()

    keep = np.isin(codes, list(EEG_TRIAL_LABEL_MAP.keys()))
    eeg_t, codes = eeg_t[keep], codes[keep]
    order = np.argsort(eeg_t)
    eeg_t, codes = eeg_t[order], codes[order]
    labels = np.array([EEG_TRIAL_LABEL_MAP[c] for c in codes], dtype=np.int64)
    sessions = np.arange(len(labels)) // TRIALS_PER_SESSION + 1

    fnirs_t, synced = eeg_t.copy(), False
    f_path = os.path.join(dataset_root, f"{sid}-NIRS", "mrk_nback.mat")
    if os.path.exists(f_path):
        mrk_f = _load_struct(f_path)
        if hasattr(mrk_f, "time") and hasattr(mrk_f, "event"):
            ft = np.asarray(mrk_f.time, dtype=float)
            fc = np.asarray(mrk_f.event.desc, dtype=int).flatten()
            m = np.isin(fc, list(FNIRS_TRIAL_LABEL_MAP.keys()))
            ft, fc = ft[m], fc[m]
            o = np.argsort(ft)
            ft, fc = ft[o] / 1000.0, fc[o]
            f_classes = np.array([FNIRS_TRIAL_LABEL_MAP[c] for c in fc])
            if len(ft) == len(labels) and np.array_equal(f_classes, labels):
                fnirs_t, synced = ft, True
    return eeg_t, fnirs_t, labels, sessions, synced


# ============================================================================
# signal helpers
# ============================================================================
def _bandpass(x, fs, lo, hi, order=4):
    b, a = butter(order, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def _canonical_hrf(fs, dur=32.0):
    """SPM-style double-gamma canonical HRF, unit L1 norm (peak ~5 s)."""
    t = np.arange(0.0, dur, 1.0 / fs)
    h = _gamma_dist.pdf(t, 6.0) - _gamma_dist.pdf(t, 16.0) / 6.0
    return h / (np.abs(h).sum() + 1e-12)


def _decimate_to(x, fs_in, fs_out):
    """Average-bin x from fs_in down to fs_out (averaging = anti-aliasing)."""
    factor = int(round(fs_in / fs_out))
    n = (len(x) // factor) * factor
    return x[:n].reshape(-1, factor).mean(axis=1)


def _zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _first_pc(arr):
    """First temporal principal component time course of arr (n_times, n_ch)."""
    c = arr - arr.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    return u[:, 0] * s[0]


def eeg_band_regressor(eeg_epoch, fs_eeg, lo, hi, use_hrf, fs_out=SFREQ_FNIRS):
    """Gao's EEG-derived 'expected hemoglobin change' regressor for one band.

    eeg_epoch: (n_frontal_channels, n_times) at fs_eeg.
    band-pass -> first temporal PC -> Hilbert power -> decimate -> HRF -> z.
    """
    filt = _bandpass(eeg_epoch, fs_eeg, lo, hi)
    pc = _first_pc(filt.T)
    power = _decimate_to(np.abs(hilbert(pc)) ** 2, fs_eeg, fs_out)
    if use_hrf:
        power = np.convolve(power, _canonical_hrf(fs_out), mode="same")
    return _zscore(power)


# ============================================================================
# permutation transfer entropy
# ============================================================================
def ordinal_symbols(x, m):
    """Ordinal (permutation) pattern sequence of x, embedding dimension m."""
    n = len(x) - (m - 1)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    idx = np.arange(n)[:, None] + np.arange(m)[None, :]
    order = np.argsort(x[idx], axis=1, kind="stable")
    table = {p: i for i, p in enumerate(permutations(range(m)))}
    return np.fromiter((table[tuple(o)] for o in order), dtype=np.int64, count=n)


def transfer_entropy_symbols(sx, sy, n_sym):
    """TE from Y to X on symbolized sequences, history length 1.

    TE_{Y->X} = sum p(x_{t+1},x_t,y_t) log[p(x_{t+1}|x_t,y_t)/p(x_{t+1}|x_t)]
    """
    if len(sx) < 2 or len(sy) < 2:
        return 0.0
    n = min(len(sx), len(sy)) - 1
    xn, xc, yc = sx[1:n + 1], sx[:n], sy[:n]
    j3 = np.zeros((n_sym, n_sym, n_sym), dtype=np.float64)
    np.add.at(j3, (xn, xc, yc), 1.0)
    j3 /= n
    p_xc_yc = j3.sum(axis=0)
    p_xn_xc = j3.sum(axis=2)
    p_xc = p_xn_xc.sum(axis=0)
    i, j, k = np.nonzero(j3)
    p = j3[i, j, k]
    return float((p * np.log((p / p_xc_yc[j, k]) / (p_xn_xc[i, j] / p_xc[j]))).sum())


def generalized_relative_entropy(labels, n_states, k=2.0):
    """GRE of the state-transition matrix (Gao et al. eq. 4).

    The paper's trailing term is data-independent, so it is omitted: it
    shifts every GRE by the same constant and cannot affect a standardized
    classifier. With k=2 this reduces to a scaled Jensen-Shannon divergence
    between P_ij and P_ji -- finite upper bound, which is the whole reason
    GRE replaces plain RE. Verified: 0 for a reversible chain, 2*ln(2) for a
    maximally directed cycle.
    """
    if len(labels) < 2:
        return 0.0
    P = np.zeros((n_states, n_states), dtype=np.float64)
    np.add.at(P, (labels[:-1], labels[1:]), 1.0)
    total = P.sum()
    if total <= 0:
        return 0.0
    P /= total
    Pt = P.T
    out = 0.0
    m = P > 0
    out += float((P[m] * np.log(k * P[m] / ((k - 1.0) * P[m] + Pt[m]))).sum())
    m = Pt > 0
    out += float((Pt[m] * np.log(k * Pt[m] / (P[m] + (k - 1.0) * Pt[m]))).sum())
    return out


# ============================================================================
# per-trial features
# ============================================================================
def coupling_features(eeg_epoch, fs_eeg, hbo, hbr, args):
    """-> (pte_features, gre_features) for one trial.

    pte_features: per (band, chromophore): mean forward PTE, mean backward
                  PTE, forward-backward asymmetry  -> 18 values.
    gre_features: per (band, chromophore): GRE of the clustered coupling
                  state trajectory                 -> 6 values.
    """
    n_sym = math.factorial(args.pte_m)
    chromo = {}
    for name, arr in (("HbO", hbo), ("HbR", hbr)):
        chromo[name] = _first_pc(arr)[:, None] if args.fnirs_mode == "pc" else arr

    win = int(round(args.win * SFREQ_FNIRS))
    step = max(1, int(round(args.step * SFREQ_FNIRS)))

    pte_feats, gre_feats = [], []
    for lo, hi in BANDS.values():
        x = eeg_band_regressor(eeg_epoch, fs_eeg, lo, hi, args.hrf == "on")
        for cname in ("HbO", "HbR"):
            y_all = chromo[cname]
            t_end = min(len(x), y_all.shape[0])
            sx = ordinal_symbols(x[:t_end], args.pte_m)
            if len(sx) < win + 1:
                pte_feats += [0.0, 0.0, 0.0]
                gre_feats.append(0.0)
                continue
            sy_all = [ordinal_symbols(y_all[:t_end, c], args.pte_m)
                      for c in range(y_all.shape[1])]

            vecs = []
            for a in range(0, len(sx) - win + 1, step):
                b = a + win
                row = []
                for sy in sy_all:
                    row.append(transfer_entropy_symbols(sy[a:b], sx[a:b], n_sym))  # EEG->fNIRS
                    row.append(transfer_entropy_symbols(sx[a:b], sy[a:b], n_sym))  # fNIRS->EEG
                vecs.append(row)
            V = np.asarray(vecs, dtype=np.float64)

            fwd = V[:, 0::2].mean()
            bwd = V[:, 1::2].mean()
            pte_feats += [fwd, bwd, fwd - bwd]

            if len(V) < 2 or not np.isfinite(V).all() or V.std() == 0:
                gre_feats.append(0.0)
            else:
                k_eff = int(min(args.k_clusters, len(V)))
                lab = KMeans(n_clusters=k_eff, n_init=10, random_state=0).fit_predict(V)
                gre_feats.append(generalized_relative_entropy(lab, k_eff, k=args.gre_k))

    return np.asarray(pte_feats), np.asarray(gre_feats)


def plain_features(eeg_epoch, fs_eeg, hbo, hbr):
    """Per-modality baselines on the same trial, at the same granularity."""
    eeg_feats = np.concatenate([
        np.log((_bandpass(eeg_epoch, fs_eeg, lo, hi) ** 2).mean(axis=1) + 1e-12)
        for lo, hi in BANDS.values()
    ])
    n_t = min(hbo.shape[0], hbr.shape[0])
    t = np.arange(n_t, dtype=np.float64)
    t -= t.mean()
    fn = []
    for arr in (hbo[:n_t], hbr[:n_t]):
        fn.append(arr.mean(axis=0))
        fn.append((arr * t[:, None]).sum(axis=0) / (t ** 2).sum())
    return eeg_feats, np.concatenate(fn)


def subject_matrices(data_root, dataset_root, sid, args):
    """All trials of one subject -> feature matrices, labels, session ids."""
    import mne

    sub_dir = os.path.join(data_root, sid)
    eeg_fif = os.path.join(sub_dir, f"{sid}_eeg_clean_raw.fif")
    fnirs_fif = os.path.join(sub_dir, f"{sid}_fnirs_clean_raw.fif")
    if not (os.path.exists(eeg_fif) and os.path.exists(fnirs_fif)):
        return None

    raw_e = mne.io.read_raw_fif(eeg_fif, preload=True, verbose=False)
    raw_f = mne.io.read_raw_fif(fnirs_fif, preload=True, verbose=False)
    fs_eeg = float(raw_e.info["sfreq"])

    picks = [raw_e.ch_names.index(c) for c in FRONTAL_CHANNELS if c in raw_e.ch_names]
    if len(picks) < 3:
        print(f"  {sid}: SKIP -- only {len(picks)} of 7 prefrontal channels present")
        return None
    eeg_all = raw_e.get_data(picks=picks)
    hbo_all = raw_f.get_data(picks=[c for c in raw_f.ch_names if c.endswith(" hbo")]).T
    hbr_all = raw_f.get_data(picks=[c for c in raw_f.ch_names if c.endswith(" hbr")]).T

    eeg_t, fnirs_t, labels, sessions, synced = load_trial_onsets(dataset_root, sid)
    if not synced:
        print(f"  {sid}: WARNING -- fNIRS markers did not match EEG 1:1; using the EEG "
              f"clock. Cross-device drift (46-93 s here) makes this subject's coupling "
              f"estimates unreliable.")

    P, G, E, F, Y, S = [], [], [], [], [], []
    for i in range(len(labels)):
        e0 = int(round((eeg_t[i] + args.tmin) * fs_eeg))
        e1 = int(round((eeg_t[i] + args.tmax) * fs_eeg))
        f0 = int(round((fnirs_t[i] + args.tmin) * SFREQ_FNIRS))
        f1 = int(round((fnirs_t[i] + args.tmax) * SFREQ_FNIRS))
        if e0 < 0 or f0 < 0 or e1 > eeg_all.shape[1] or f1 > hbo_all.shape[0]:
            continue

        b0 = int(round((fnirs_t[i] + BASELINE_TMIN) * SFREQ_FNIRS))
        b1 = int(round((fnirs_t[i] + BASELINE_TMAX) * SFREQ_FNIRS))
        if b0 >= 0 and b1 > b0:
            base_o = hbo_all[b0:b1].mean(axis=0, keepdims=True)
            base_r = hbr_all[b0:b1].mean(axis=0, keepdims=True)
        else:
            base_o = base_r = 0.0
        eeg_ep = eeg_all[:, e0:e1]
        hbo_ep, hbr_ep = hbo_all[f0:f1] - base_o, hbr_all[f0:f1] - base_r

        p, g = coupling_features(eeg_ep, fs_eeg, hbo_ep, hbr_ep, args)
        e, f = plain_features(eeg_ep, fs_eeg, hbo_ep, hbr_ep)
        P.append(p); G.append(g); E.append(e); F.append(f)
        Y.append(labels[i]); S.append(sessions[i])

    if not Y:
        return None
    return tuple(np.asarray(v) for v in (P, G, E, F, Y, S))


# ============================================================================
# evaluation
# ============================================================================
def _clf(kind):
    if kind == "svm":
        return make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf", gamma="scale"))
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))


def cross_session(X, y, ses, kind):
    """Train on sessions != s, test on session == s; mean over s."""
    accs = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        if te.sum() == 0 or tr.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        model = _clf(kind).fit(X[tr], y[tr])
        accs.append(float((model.predict(X[te]) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


# ============================================================================
# instrument self-test (plants a known coupling)
# ============================================================================
def self_test(args, n_trials=12, seed=1):
    """Plant a strong, explicit EEG->fNIRS coupling and check each feature
    family recovers it. A family that fails here cannot be trusted to
    produce a meaningful NULL on real data."""
    from scipy.stats import mannwhitneyu
    rng = np.random.default_rng(seed)
    fs, dur = 200.0, args.tmax - args.tmin
    n_e, n_f = int(dur * fs), int(dur * SFREQ_FNIRS)
    hrf = _canonical_hrf(SFREQ_FNIRS)

    def make(coupled):
        t = np.arange(n_e) / fs
        env = 1.0 + 0.8 * np.sin(2 * np.pi * 0.05 * t) + 0.3 * rng.normal(size=n_e)
        theta = env * np.sin(2 * np.pi * 6.0 * t)
        eeg = rng.normal(scale=0.5, size=(7, n_e)) + theta[None, :]
        if coupled:
            drive = np.convolve(_decimate_to(theta ** 2, fs, SFREQ_FNIRS), hrf, mode="same")
            drive = _zscore(drive)
            hbo = drive[:, None] * np.ones((1, 36)) + 0.5 * rng.normal(size=(n_f, 36))
        else:
            hbo = rng.normal(size=(n_f, 36))
        return eeg, hbo, -0.6 * hbo + 0.4 * rng.normal(size=(n_f, 36))

    out = {True: {"pte": [], "gre": []}, False: {"pte": [], "gre": []}}
    for cond in (True, False):
        for _ in range(n_trials):
            e, o, r = make(cond)
            p, g = coupling_features(e, fs, o, r, args)
            out[cond]["pte"].append(p)
            out[cond]["gre"].append(g)

    print("=" * 78)
    print(f"SELF-TEST -- planted EEG->fNIRS coupling, {n_trials} trials per condition")
    print(f"config: pte_m={args.pte_m} win={args.win}s step={args.step}s "
          f"fnirs_mode={args.fnirs_mode} hrf={args.hrf} k_clusters={args.k_clusters}")
    print("=" * 78)
    verdict = {}
    for fam, idx, label in (("pte", 2, "theta-HbO asymmetry (fwd-bwd)"),
                            ("pte", 0, "theta-HbO forward PTE"),
                            ("gre", 0, "theta-HbO GRE")):
        a = np.array([v[idx] for v in out[True][fam]])
        b = np.array([v[idx] for v in out[False][fam]])
        p = float(mannwhitneyu(a, b)[1])
        ok = p < 0.01
        verdict.setdefault(fam, []).append(ok)
        print(f"  {label:32} coupled={a.mean():9.4f}  none={b.mean():9.4f}  p={p:.2e}  "
              f"{'PASS' if ok else 'FAIL'}")
    print()
    for fam, name in (("pte", "PTE magnitude/asymmetry"), ("gre", "GRE (Gao)")):
        if any(verdict[fam]):
            msg = "USABLE -- a null from these rows is meaningful"
        else:
            msg = "INSENSITIVE -- a null from these rows means NOTHING"
        print(f"  {name:26}: {msg}")
    print("=" * 78)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("usage")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="preprocessed folder with {sid}/{sid}_eeg_clean_raw.fif "
                                   "and {sid}_fnirs_clean_raw.fif")
    ap.add_argument("--dataset", default=r"E:\kntu\Thesis\codes\DATA",
                    help="ORIGINAL dataset folder (needs {sid}-EEG / {sid}-NIRS mrk_nback.mat)")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    ap.add_argument("--self-test", action="store_true",
                    help="run the planted-coupling instrument check and exit")
    ap.add_argument("--tmin", type=float, default=-5.0, help="epoch start rel. onset (Gao: -5)")
    ap.add_argument("--tmax", type=float, default=60.0, help="epoch end rel. onset (Gao: 60)")
    ap.add_argument("--win", type=float, default=20.0,
                    help="coupling window, s. Default 20 (200 samples) for a usable PTE "
                         "estimate; Gao use 5, which the self-test shows is too short here.")
    ap.add_argument("--step", type=float, default=2.0, help="coupling window step, s")
    ap.add_argument("--pte-m", type=int, default=2,
                    help="PTE embedding dimension. Default 2 (8 joint symbol states); m=3 "
                         "gives 216 states, which 50-200-sample windows cannot populate.")
    ap.add_argument("--k-clusters", type=int, default=10, help="coarse-grained states (Gao: 10)")
    ap.add_argument("--gre-k", type=float, default=2.0, help="GRE control parameter (Gao: 2)")
    ap.add_argument("--hrf", choices=["on", "off"], default="on",
                    help="convolve the EEG power regressor with the canonical HRF")
    ap.add_argument("--fnirs-mode", choices=["pc", "channels"], default="pc",
                    help="'pc' (default, ~0.5 s/trial) couples EEG to the first PC across the "
                         "36 fNIRS channels; 'channels' couples to all 36 separately, keeping "
                         "Gao's spatial detail but ~36x slower (hours for 26 subjects -- use "
                         "with --subjects)")
    ap.add_argument("--clf", choices=["svm", "lr"], default="svm", help="classifier (Gao: SVM)")
    ap.add_argument("--out", default=None, help="write a JSON summary here")
    args = ap.parse_args()

    if args.self_test:
        self_test(args)
        return
    if not args.data:
        sys.exit("--data is required (or pass --self-test)")
    if not os.path.isdir(args.data):
        sys.exit(f"--data does not exist: {args.data}")
    if not os.path.isdir(args.dataset):
        sys.exit(f"--dataset does not exist: {args.dataset}")

    allP, allG, allE, allF, allY, allS, allsub = [], [], [], [], [], [], []
    print(f"{'subject':9} {'trials':>6}   (extracting coupling features)")
    for sid in args.subjects:
        try:
            got = subject_matrices(args.data, args.dataset, sid, args)
        except Exception as exc:
            print(f"{sid:9} SKIP -- {type(exc).__name__}: {exc}")
            continue
        if got is None:
            print(f"{sid:9} SKIP -- required files missing")
            continue
        P, G, E, F, Y, S = got
        allP.append(P); allG.append(G); allE.append(E); allF.append(F)
        allY.append(Y); allS.append(S); allsub.append(np.full(len(Y), sid))
        print(f"{sid:9} {len(Y):6d}", flush=True)

    if not allY:
        sys.exit("No subjects processed -- check --data / --dataset paths.")

    P = np.vstack(allP); G = np.vstack(allG)
    E = np.vstack(allE); F = np.vstack(allF)
    y = np.concatenate(allY); ses = np.concatenate(allS); sub = np.concatenate(allsub)

    sets = {
        "EEG power": E,
        "fNIRS mean+slope": F,
        "EEG + fNIRS": np.hstack([E, F]),
        "coupling PTE [validated]": P,
        "coupling GRE [see warning]": G,
        "PTE + EEG + fNIRS": np.hstack([P, E, F]),
        "everything": np.hstack([P, G, E, F]),
    }
    tasks = {
        "3-class (0/2/3-back)": np.ones(len(y), dtype=bool),
        "2-back vs 3-back": np.isin(y, [1, 2]),
    }

    results = {}
    line = "=" * 88
    for tname, tmask in tasks.items():
        chance = 1.0 / len(np.unique(y[tmask]))
        print("\n" + line)
        print(f"{tname}   (chance = {chance:.3f}, n = {int(tmask.sum())} trials)")
        print(line)
        print(f"{'feature set':28} {'pooled cross-session':>21} {'per-subject cross-session':>26}")
        for fname, X in sets.items():
            Xm, ym, sm, subm = X[tmask], y[tmask], ses[tmask], sub[tmask]
            pooled = cross_session(Xm, ym, sm, args.clf)
            per_sub = float(np.nanmean([
                cross_session(Xm[subm == s], ym[subm == s], sm[subm == s], args.clf)
                for s in np.unique(subm)]))
            results[f"{tname} | {fname}"] = {"pooled": pooled, "per_subject": per_sub}
            print(f"{fname:28} {pooled:21.3f} {per_sub:26.3f}")

    print("\n" + line)
    print("READING")
    print(line)
    base = results["2-back vs 3-back | EEG + fNIRS"]["pooled"]
    cpl = results["2-back vs 3-back | coupling PTE [validated]"]["pooled"]
    both = results["2-back vs 3-back | PTE + EEG + fNIRS"]["pooled"]
    print(f"  2-vs-3, pooled cross-session:  EEG+fNIRS={base:.3f}  "
          f"coupling PTE={cpl:.3f}  PTE+EEG+fNIRS={both:.3f}")
    if both > base + 0.02 or cpl > base + 0.02:
        print("  Coupling carries 2-vs-3 information the per-modality features do not -- on the\n"
              "  class pair that dominates this project's errors. A model-level coupling branch\n"
              "  is justified. That is a real project (new input tensor + architecture change),\n"
              "  not a flag, so scope it before starting.")
    elif abs(both - base) <= 0.02 and abs(cpl - base) <= 0.02:
        print("  Coupling neither helps nor hurts beyond the per-modality features. Because the\n"
              "  PTE rows are instrument-validated (see --self-test), this is a genuine null on\n"
              "  702 trials, not small-n noise: the cheap version of this direction is closed.\n"
              "  Before calling it final, re-run with --fnirs-mode channels on a few subjects --\n"
              "  the default PC reduction discards exactly the spatial detail Gao keep.")
    else:
        print("  Coupling is WORSE than the per-modality features here. Together with the\n"
              "  protocol mismatch (Gao: random 5-fold over pooled subjects AND sessions; this:\n"
              "  strict cross-session), there is no cheap evidence for this direction.")
    print("\n  Reminder: these are TRIAL-level numbers (27 per subject) and are NOT comparable\n"
          "  to check_feature_baseline_tsfnet.py's window-level numbers (~870 per subject).\n"
          "  Compare only the rows above against each other.")

    if args.out:
        d = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(d, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "results": results}, fh, indent=2, default=float)
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main()
