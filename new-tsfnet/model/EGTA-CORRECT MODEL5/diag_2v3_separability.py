"""
diag_2v3_separability.py -- is 2-back vs 3-back separable in this data at all?

WHY
===
Across three seeds, 64% of the run60 model's block-level errors are 2-back <->
3-back swaps, and every loss-side intervention aimed at that pair (asymmetric
cost, class weights, focal loss, SMOTE, ordinal regularization, CORN) failed.
Before building new architecture, this script measures the CEILING that simple,
physiologically motivated features reach on the same question, with the same
leave-one-session-out protocol:

  * if even these features sit at chance -> 2/3 is not separable at the
    single-subject level in this dataset; that is a thesis finding (ceiling
    analysis) and engineering effort should go elsewhere;
  * if some feature group clearly beats the deep model's 2-vs-3 accuracy ->
    that group carries information the network is not extracting, and tells
    you which branch to build.

UNIT OF ANALYSIS: the n-back BLOCK (one 40 s series, one label), 27 per subject,
9 per session -- the same unit as trial_level_eval.py's block-level verdict.

FEATURE GROUPS (each computed per block)
  erp_p300   mean amplitude 0.30-0.50 s at Pz, Cz, CP1, CP2, P3, P4 and P300
             peak latency at Pz (0.25-0.60 s), from the average of all ~20
             digit-locked epochs of the block                        (7 feats)
  erp_full   the block-average ERP itself, 0-0.8 s in 50 ms bins, same six
             channels (classic shrinkage-LDA ERP decoding)           (96 feats)
  band_roi   log theta (4-8 Hz) over AFz/F1/F2/FC1/FC2, log alpha (8-13 Hz)
             over Pz/P3/P4/POz, and their difference                  (3 feats)
  band_all   log theta and log alpha at every EEG channel           (2 x 28)
  fnirs_roi  prefrontal (all "AF*" optodes, 16 ch) HbO and HbR: mean over the
             task period minus pre-block baseline, and slope         (4 feats)
  fnirs_all  baseline-corrected task-period HbO mean per channel     (36 feats)
  all_roi    erp_p300 + band_roi + fnirs_roi                         (14 feats)

CLASSIFIER: shrinkage LDA (sklearn, solver="lsqr", shrinkage="auto"), the
standard for small-n neural decoding; StandardScaler and mean imputation are
fit on the training fold only.

PROTOCOL (per subject): for each held-out session, train on the 2-back and
3-back blocks of the other two sessions (12 blocks), test on the held-out
session's 6. Reported per subject = mean over its 3 folds.

SANITY CHECK: the identical pipeline is run on 0-back vs 2-back, which the
deep model separates well (~79% 0-back recall). If a feature group cannot
separate 0 from 2 either, its 2-vs-3 result says nothing about the data --
it says the features are broken.

LEAKAGE: stimulus marker codes ENCODE THE CLASS (16 = 0-back, 48/64 = 2-back,
80/96 = 3-back). Here they are used ONLY to obtain digit onset TIMES, which are
the same 2 s grid in every class; a stimulus's code never enters a feature.
The block label comes from the block-onset code (112/128/144), as in
preprocessing. Inputs are the cleaned continuous recordings preprocessing
already saved ({VP}_eeg_clean_raw.fif, {VP}_fnirs_clean_raw.fif) plus the raw
marker files -- the same sources Run D0.1b used. No GPU, no model.

CAPACITY-CEILING CHECK (also printed): per subject, the sign of
(3-back - 2-back) for prefrontal HbO and for P300 amplitude. Literature
predicts HbO rising and P300 falling with load; subjects with the opposite
sign may be past their capacity (inverted U, Callicott et al. 1999).

USAGE
    python diag_2v3_separability.py
    python diag_2v3_separability.py --clean-root "<folder with VPxxx\\*_clean_raw.fif>" ^
                                    --raw-root   "<folder with VPxxx-EEG / VPxxx-NIRS>"
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import binomtest, wilcoxon

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--raw-root", default=r"E:\kntu\Thesis\codes\DATA",
               help="folder holding VPxxx-EEG and VPxxx-NIRS (raw .mat, for markers)")
p.add_argument("--clean-root", default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback",
               help="folder holding VPxxx\\VPxxx_eeg_clean_raw.fif and _fnirs_clean_raw.fif")
p.add_argument("--out-dir", default=None, help="default: <clean-root>/diag_2v3")
p.add_argument("--subjects", default="1-26")
p.add_argument("--deep-ref", type=float, nargs="*", default=[259 / 425, 240 / 414],
               help="deep model 2-vs-3 block accuracy (among 2/3 blocks not called 0-back) "
                    "for comparison; defaults = run60 seeds 42 and 999 from trial_level_eval")
ARGS = p.parse_args()

SFREQ_EEG = 200.0
SFREQ_FNIRS = 10.0
BLOCK_CODES_EEG = {112: 0, 128: 1, 144: 2}          # 0-/2-/3-back block onset (as preprocessing)
BLOCK_CODES_FNIRS = {7: 0, 8: 1, 9: 2}
STIM_CODES = {16, 48, 64, 80, 96}                    # used for TIMING ONLY (see docstring)
BLOCKS_PER_SESSION = 9
P300_CH = ["Pz", "Cz", "CP1", "CP2", "P3", "P4"]
THETA_CH = ["AFz", "F1", "F2", "FC1", "FC2"]
ALPHA_CH = ["Pz", "P3", "P4", "POz"]
ERP_TMIN, ERP_TMAX, ERP_BASE = -0.2, 0.8, (-0.2, 0.0)
FNIRS_BASE = (-5.0, -2.0)                            # same pre-block baseline as preprocessing
FNIRS_TASK = (2.0, 42.0)                             # task period after the 2 s instruction
FNIRS_HRF_LAG = 5.0                                  # mean is taken from task start + lag

GROUPS = ["erp_p300", "erp_full", "band_roi", "band_all", "fnirs_roi", "fnirs_all", "all_roi"]


def parse_subjects(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return [f"VP{i:03d}" for i in out]


def _first_struct(path):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return m[[k for k in m if not k.startswith("__")][0]]


def load_eeg_events(eeg_folder):
    """Same logic as Run D0.1b's load_eeg_markers: (sample @200 Hz, code)."""
    cnt = _first_struct(os.path.join(eeg_folder, "cnt_nback.mat"))
    fs_orig = float(cnt.fs)
    fs = SFREQ_EEG if fs_orig != SFREQ_EEG else fs_orig
    mrk = _first_struct(os.path.join(eeg_folder, "mrk_nback.mat"))
    if hasattr(mrk, "time"):
        pos = np.round(np.asarray(mrk.time, dtype=float) * fs / 1000).astype(int)
    else:
        pos = np.asarray(mrk.pos, dtype=int) - 1
    codes = np.asarray(mrk.event.desc, dtype=int).flatten()
    return pos, codes


def load_fnirs_events(nirs_folder):
    """Block-onset times in seconds and codes, as preprocessing's load_fnirs_data."""
    cnt = _first_struct(os.path.join(nirs_folder, "cnt_nback.mat"))
    mrk = _first_struct(os.path.join(nirs_folder, "mrk_nback.mat"))
    if hasattr(mrk, "time"):
        t = np.asarray(mrk.time, dtype=float) / 1000.0
    else:
        t = np.asarray(mrk.pos, dtype=float) / float(getattr(cnt, "fs", SFREQ_FNIRS))
    codes = np.asarray(mrk.event.desc, dtype=int).flatten()
    return t, codes


def block_table(pos, codes, block_codes):
    """Chronological blocks: list of (onset, label, session)."""
    idx = np.flatnonzero(np.isin(codes, list(block_codes)))
    blocks = [(pos[i], block_codes[int(codes[i])]) for i in idx]
    return [(on, lab, k // BLOCKS_PER_SESSION + 1) for k, (on, lab) in enumerate(blocks)]


def eeg_block_features(raw, pos, codes, blocks):
    """Per block: ERP (P300 summary + binned), band power. Returns list of dicts."""
    import mne
    data = raw.get_data()
    names = raw.ch_names
    ch = {n: i for i, n in enumerate(names)}
    n_t = data.shape[1]
    stim_idx = np.flatnonzero(np.isin(codes, list(STIM_CODES)))
    stim_pos = pos[stim_idx]
    bad = np.zeros(n_t, dtype=bool)                     # honour BAD_* annotations like mne.Epochs
    for a in raw.annotations:
        if str(a["description"]).upper().startswith("BAD"):
            s = int(round((a["onset"] - raw.first_time) * SFREQ_EEG))
            bad[max(s, 0):min(s + int(round(a["duration"] * SFREQ_EEG)) + 1, n_t)] = True
    e0, e1 = int(round(ERP_TMIN * SFREQ_EEG)), int(round(ERP_TMAX * SFREQ_EEG))
    b0, b1 = 0, int(round((ERP_BASE[1] - ERP_TMIN) * SFREQ_EEG))
    times = np.arange(e0, e1 + 1) / SFREQ_EEG
    p300_ch = [ch[c] for c in P300_CH if c in ch]
    out = []
    for k, (on, lab, sess) in enumerate(blocks):
        end = blocks[k + 1][0] if k + 1 < len(blocks) else n_t
        sp = stim_pos[(stim_pos > on) & (stim_pos < end)]
        epochs = []
        for s in sp:
            a, b = s + e0, s + e1 + 1
            if a < 0 or b > n_t or bad[a:b].any():
                continue
            ep = data[:, a:b]
            epochs.append(ep - ep[:, b0:b1].mean(axis=1, keepdims=True))
        f = {"label": lab, "session": sess, "n_epochs": len(epochs), "n_stim": len(sp)}
        if epochs and p300_ch:
            erp = np.mean(epochs, axis=0)[p300_ch]                    # (6, T)
            w = (times >= 0.30) & (times <= 0.50)
            lat_w = (times >= 0.25) & (times <= 0.60)
            pz = P300_CH.index("Pz") if "Pz" in ch else 0
            f["erp_p300"] = np.r_[erp[:, w].mean(axis=1),
                                  times[lat_w][np.argmax(erp[pz, lat_w])]]
            bins = [(times >= t0) & (times < t0 + 0.05) for t0 in np.arange(0, 0.8, 0.05)]
            f["erp_full"] = np.stack([erp[:, m].mean(axis=1) for m in bins], axis=1).ravel()
        else:
            f["erp_p300"] = np.full(len(P300_CH) + 1, np.nan)
            f["erp_full"] = np.full(len(P300_CH) * 16, np.nan)
        # band power over the span actually covered by this block's digits
        if len(sp) >= 2:
            seg = data[:, sp[0]:min(sp[-1] + int(2 * SFREQ_EEG), n_t)]
            fr, pxx = welch(seg, fs=SFREQ_EEG, nperseg=int(2 * SFREQ_EEG), axis=1)
            th = np.log(pxx[:, (fr >= 4) & (fr < 8)].mean(axis=1) + 1e-30)
            al = np.log(pxx[:, (fr >= 8) & (fr < 13)].mean(axis=1) + 1e-30)
            tr = np.mean([th[ch[c]] for c in THETA_CH if c in ch])
            ar = np.mean([al[ch[c]] for c in ALPHA_CH if c in ch])
            f["band_roi"] = np.array([tr, ar, tr - ar])
            f["band_all"] = np.r_[th, al]
        else:
            f["band_roi"] = np.full(3, np.nan)
            f["band_all"] = np.full(2 * len(names), np.nan)
        out.append(f)
    return out


def fnirs_block_features(raw, t_on, blocks):
    data = raw.get_data()
    names = raw.ch_names
    hbo = [i for i, n in enumerate(names) if n.endswith("hbo")]
    hbr = [i for i, n in enumerate(names) if n.endswith("hbr")]
    pfc_o = [i for i in hbo if names[i].startswith("AF")]
    pfc_r = [i for i in hbr if names[i].startswith("AF")]
    n_t = data.shape[1]
    tt = np.arange(n_t) / SFREQ_FNIRS
    out = []
    for (on, lab, sess) in blocks:
        t0 = on                                                 # seconds
        base = (tt >= t0 + FNIRS_BASE[0]) & (tt < t0 + FNIRS_BASE[1])
        task = (tt >= t0 + FNIRS_TASK[0]) & (tt < t0 + FNIRS_TASK[1])
        late = (tt >= t0 + FNIRS_TASK[0] + FNIRS_HRF_LAG) & (tt < t0 + FNIRS_TASK[1])
        f = {"label": lab, "session": sess}
        if base.sum() < 5 or task.sum() < 20 or not pfc_o:
            f["fnirs_roi"] = np.full(4, np.nan)
            f["fnirs_all"] = np.full(len(hbo), np.nan)
        else:
            bc = data[:, late].mean(axis=1) - data[:, base].mean(axis=1)
            x = tt[task] - tt[task][0]
            slope = np.polyfit(x, data[:, task].T, 1)[0]            # per channel
            f["fnirs_roi"] = np.array([bc[pfc_o].mean(), slope[pfc_o].mean(),
                                       bc[pfc_r].mean(), slope[pfc_r].mean()])
            f["fnirs_all"] = bc[hbo]
        out.append(f)
    return out


def classify(X, y, sess, pos_label, neg_label):
    """LOSO over sessions, binary pos vs neg. Returns (per-fold acc list, n_test)."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = np.isin(y, [pos_label, neg_label])
    X, y, sess = X[m], (y[m] == pos_label).astype(int), sess[m]
    accs, correct, n = [], 0, 0
    for s in np.unique(sess):
        tr, te = sess != s, sess == s
        if len(np.unique(y[tr])) < 2 or te.sum() == 0:
            continue
        if np.isnan(X[tr]).all(axis=0).any():               # a feature missing in all train rows
            keep = ~np.isnan(X[tr]).all(axis=0)
        else:
            keep = np.ones(X.shape[1], bool)
        clf = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X[tr][:, keep], y[tr])
            pr = clf.predict(X[te][:, keep])
        accs.append(float((pr == y[te]).mean()))
        correct += int((pr == y[te]).sum())
        n += int(te.sum())
    return accs, correct, n


def main():
    import mne
    mne.set_log_level("ERROR")
    out_dir = ARGS.out_dir or os.path.join(ARGS.clean_root, "diag_2v3")
    os.makedirs(out_dir, exist_ok=True)

    per_subject, ceiling = {}, {}
    pooled = {g: {"2v3": [0, 0], "0v2": [0, 0]} for g in GROUPS}
    for sid in parse_subjects(ARGS.subjects):
        eeg_fif = os.path.join(ARGS.clean_root, sid, f"{sid}_eeg_clean_raw.fif")
        nirs_fif = os.path.join(ARGS.clean_root, sid, f"{sid}_fnirs_clean_raw.fif")
        eeg_dir = os.path.join(ARGS.raw_root, f"{sid}-EEG")
        nirs_dir = os.path.join(ARGS.raw_root, f"{sid}-NIRS")
        missing = [p for p in (eeg_fif, nirs_fif, eeg_dir, nirs_dir) if not os.path.exists(p)]
        if missing:
            print(f"[{sid}] SKIP -- missing: {missing}")
            continue
        try:
            pos, codes = load_eeg_events(eeg_dir)
            eblocks = block_table(pos, codes, BLOCK_CODES_EEG)
            t_on, fcodes = load_fnirs_events(nirs_dir)
            fblocks = block_table(t_on, fcodes, BLOCK_CODES_FNIRS)
        except Exception as e:
            print(f"[{sid}] SKIP -- markers: {type(e).__name__}: {e}")
            continue
        if len(eblocks) != 27 or len(fblocks) != 27:
            print(f"[{sid}] SKIP -- expected 27 blocks, got EEG {len(eblocks)} / fNIRS {len(fblocks)}")
            continue
        if [b[1] for b in eblocks] != [b[1] for b in fblocks]:
            print(f"[{sid}] SKIP -- EEG and fNIRS block label sequences disagree")
            continue

        ef = eeg_block_features(mne.io.read_raw_fif(eeg_fif, preload=True), pos, codes, eblocks)
        ff = fnirs_block_features(mne.io.read_raw_fif(nirs_fif, preload=True), t_on, fblocks)
        y = np.array([b[1] for b in eblocks])
        sess = np.array([b[2] for b in eblocks])
        feats = {g: np.array([f[g] for f in ef]) for g in ("erp_p300", "erp_full", "band_roi", "band_all")}
        feats.update({g: np.array([f[g] for f in ff]) for g in ("fnirs_roi", "fnirs_all")})
        feats["all_roi"] = np.hstack([feats["erp_p300"], feats["band_roi"], feats["fnirs_roi"]])

        row = {"n_epochs_per_block": float(np.mean([f["n_epochs"] for f in ef])),
               "n_stim_per_block": float(np.mean([f["n_stim"] for f in ef]))}
        for g in GROUPS:
            for name, (a, b) in (("2v3", (2, 1)), ("0v2", (1, 0))):
                accs, c, n = classify(feats[g], y, sess, a, b)
                row[f"{g}_{name}"] = float(np.mean(accs)) if accs else float("nan")
                pooled[g][name][0] += c
                pooled[g][name][1] += n
        per_subject[sid] = row

        # capacity-ceiling signs (raw feature means, no model)
        m2, m3 = y == 1, y == 2
        ceiling[sid] = {
            "pfc_hbo_3minus2": float(np.nanmean(feats["fnirs_roi"][m3, 0]) - np.nanmean(feats["fnirs_roi"][m2, 0])),
            "p300_pz_3minus2": float(np.nanmean(feats["erp_p300"][m3, 0]) - np.nanmean(feats["erp_p300"][m2, 0])),
        }
        print(f"[{sid}] {row['n_stim_per_block']:.0f} digits/block ({row['n_epochs_per_block']:.1f} clean) | "
              f"2v3: erp {row['erp_p300_2v3']:.2f} band {row['band_roi_2v3']:.2f} "
              f"fnirs {row['fnirs_roi_2v3']:.2f} all {row['all_roi_2v3']:.2f} | "
              f"0v2 all {row['all_roi_0v2']:.2f}")

    if not per_subject:
        sys.exit("ERROR: no subject could be processed -- check --clean-root / --raw-root.")

    ids = sorted(per_subject)
    summary = {"n_subjects": len(ids), "deep_model_reference_2v3": ARGS.deep_ref, "groups": {}}
    print("\n" + "=" * 92)
    print(f"2-BACK vs 3-BACK, block level, leave-one-session-out, {len(ids)} subjects "
          f"(chance = 0.50; deep model reference = {', '.join(f'{r:.3f}' for r in ARGS.deep_ref)})")
    print("=" * 92)
    print(f"{'features':10s} | {'2v3 mean':>8s} {'sd':>5s} {'>0.5':>5s} {'p(W)':>7s} {'pooled':>7s} "
          f"{'p(binom)':>8s} || {'0v2 sanity':>10s}")
    for g in GROUPS:
        a = np.array([per_subject[s][f"{g}_2v3"] for s in ids])
        s0 = np.array([per_subject[s][f"{g}_0v2"] for s in ids])
        a_ok = a[~np.isnan(a)]
        d = a_ok - 0.5
        pw = (wilcoxon(d, alternative="greater").pvalue
              if (d != 0).sum() >= 6 else float("nan"))
        c, n = pooled[g]["2v3"]
        pb = binomtest(c, n, 0.5, alternative="greater").pvalue if n else float("nan")
        c0, n0 = pooled[g]["0v2"]
        summary["groups"][g] = {
            "acc_2v3_mean": float(np.nanmean(a)), "acc_2v3_sd": float(np.nanstd(a)),
            "n_subjects_above_chance": int((a_ok > 0.5).sum()), "wilcoxon_p_vs_chance": pw,
            "pooled_2v3": [c, n], "binomial_p_vs_chance": pb,
            "acc_0v2_mean": float(np.nanmean(s0)), "pooled_0v2": [c0, n0]}
        print(f"{g:10s} | {np.nanmean(a):8.3f} {np.nanstd(a):5.3f} {int((a_ok > 0.5).sum()):3d}/{len(a_ok):<2d}"
              f"{pw:7.3g} {c / max(n, 1):7.3f} {pb:8.3g} || {np.nanmean(s0):10.3f}")

    # ---- Holm correction across the feature groups (7 tests at once) -------
    raw_p = [(g, summary["groups"][g]["wilcoxon_p_vs_chance"]) for g in GROUPS]
    valid = sorted([(g, pv) for g, pv in raw_p if pv == pv], key=lambda t: t[1])
    m, running = len(valid), 0.0
    for i, (g, pv) in enumerate(valid):
        running = max(running, min(1.0, (m - i) * pv))
        summary["groups"][g]["wilcoxon_p_holm"] = running
    for g, pv in raw_p:
        summary["groups"][g].setdefault("wilcoxon_p_holm", float("nan"))
    print("  (Holm-corrected Wilcoxon p across the 7 groups: "
          + ", ".join(f"{g} {summary['groups'][g]['wilcoxon_p_holm']:.3g}" for g in GROUPS) + ")")

    # ---- verdict -----------------------------------------------------------
    # Only groups that separate 0-back from 2-back are trusted for 2-vs-3.
    ref = float(np.mean(ARGS.deep_ref))
    sanity_ok = [g for g in GROUPS if summary["groups"][g]["acc_0v2_mean"] >= 0.65]
    best = max(sanity_ok or GROUPS, key=lambda g: summary["groups"][g]["acc_2v3_mean"])
    bg = summary["groups"][best]
    print("\nVERDICT")
    if not sanity_ok:
        verdict = ("PIPELINE SUSPECT: no feature group separates 0-back from 2-back (>= 0.65). "
                   "Do not interpret the 2-vs-3 numbers; check paths, markers and channel names.")
    elif bg["acc_2v3_mean"] < 0.55 or not (bg["wilcoxon_p_holm"] < 0.05):
        verdict = (f"NOT SEPARABLE with these features: best group '{best}' reaches "
                   f"{bg['acc_2v3_mean']:.3f} (Holm p={bg['wilcoxon_p_holm']:.3g} vs chance) while the "
                   f"same features separate 0 vs 2 ({', '.join(sanity_ok)}). Report as a ceiling finding; "
                   f"engineering effort on the 2/3 pair is unlikely to pay off.")
    elif bg["acc_2v3_mean"] >= ref + 0.03:
        verdict = (f"INFORMATION THE NETWORK MISSES: '{best}' reaches {bg['acc_2v3_mean']:.3f}, above the "
                   f"deep model's ~{ref:.3f}. Build a branch for this feature group.")
    else:
        verdict = (f"SEPARABLE BUT NOT BEYOND THE NETWORK: '{best}' reaches {bg['acc_2v3_mean']:.3f} "
                   f"(above chance, Holm p={bg['wilcoxon_p_holm']:.3g}) but not clearly above the deep "
                   f"model's ~{ref:.3f}. The network already extracts about as much as these features hold.")
    print("  " + verdict)
    summary["verdict"] = verdict

    # ---- capacity-ceiling signs --------------------------------------------
    h = np.array([ceiling[s]["pfc_hbo_3minus2"] for s in ids])
    q = np.array([ceiling[s]["p300_pz_3minus2"] for s in ids])
    print("\nCAPACITY CHECK (3-back minus 2-back, per subject; literature: HbO up, P300 down)")
    print(f"  prefrontal HbO : higher in 3-back for {(h > 0).sum()}/{len(h)} subjects, "
          f"lower for {(h < 0).sum()}  -> opposite-sign subjects: "
          f"{[s for s, v in zip(ids, h) if v < 0]}")
    print(f"  P300 at Pz     : lower in 3-back for {(q < 0).sum()}/{len(q)} subjects, "
          f"higher for {(q > 0).sum()}")
    summary["capacity_check"] = ceiling

    with open(os.path.join(out_dir, "diag_2v3_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": per_subject, **summary}, f, indent=2)
    import csv
    with open(os.path.join(out_dir, "diag_2v3_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = list(per_subject[ids[0]])
        w.writerow(["subject"] + cols + ["pfc_hbo_3minus2", "p300_pz_3minus2"])
        for s in ids:
            w.writerow([s] + [per_subject[s][c] for c in cols]
                       + [ceiling[s]["pfc_hbo_3minus2"], ceiling[s]["p300_pz_3minus2"]])
    print(f"\n[written] {out_dir}\\diag_2v3_summary.json and diag_2v3_per_subject.csv")


if __name__ == "__main__":
    main()
