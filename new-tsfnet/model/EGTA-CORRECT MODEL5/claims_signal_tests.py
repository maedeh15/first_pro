"""
claims_signal_tests.py -- data-only tests of literature claims on the SIGNALS (no model, CPU). Subject = unit; two-sided paired Wilcoxon; Holm within family.
PRE-STATED PREDICTIONS (written before looking at the output; direction is recorded, a wrong direction is a reportable negative result):
 [H-hemo]  fNIRS, spatial-mean (global) trace, mean over the task window 10-40 s after onset, baseline-corrected as saved by the preprocessing:
           HbO(2-back) > HbO(0-back) and HbO(3-back) > HbO(0-back) (load increases oxygenated haemoglobin); HbR moves the opposite way;
           inverted-U: HbO(3-back) < HbO(2-back) AND HbO(2-back) > HbO(0-back).  Global only: the 16x16 map has no verified channel identity.
 [H-eeg]   EEG, mean over the 28 channels (global; channel names are unverified, so NO frontal-theta / parietal-alpha claim), Welch 2 s segments, per-session z-scored
           band power: theta (4-8 Hz) up with load, alpha (8-13 Hz) down with load, theta/alpha log-ratio up with load.  Effect = pairwise AUC (0v2, 0v3, 2v3)
           per subject (windows pooled over sessions after per-session z-scoring), AUC != 0.5 tested over subjects.  Extra: |AUC-0.5| of the ratio vs of alpha alone
           (does the ratio add anything over its parts?).
[H-region] (needs --channel-names = channels.txt of check_channel_order.py, order CONFIRMED there; run with --regional-only) EEG, REGIONAL band power, same AUC/Holm machinery, pre-stated:
           frontal theta UP with load (AUC > 0.5 for 0v2, 0v3), parietal alpha DOWN with load (AUC < 0.5).  Regions (fixed in this file BEFORE any result, 10-05 names of the 28 channels):
           frontal = Fp1, AFF5h, AFz, F1, Fp2, AFF6h, F2 ; parietal = P7, P3, Pz, POz, P4, P8.  Holm over these 2 features x 3 pairs = 6 tests.
           Exploratory controls (reported, not part of the Holm family): parietal theta and frontal alpha, and the SPECIFICITY contrasts frontal-theta minus parietal-theta and
           parietal-alpha minus frontal-alpha (is the effect regional, or just the global effect seen at every site?).  Positions are standard 10-05 (not subject-specific); band power is of the
           channel-averaged PSD of the region; per-session z-scored; windows overlap, AUC is per subject.
Windows overlap strongly (1 s step); AUC is computed per subject so the group test is not inflated by that. fNIRS window rank within a trial gives its time
(offset = -2 s + rank x 1 s); rejected windows would shift this slightly (state it).
  python claims_signal_tests.py --data-root "..." --subjects 1-26 --out-dir "...\\claims_signal"
"""
import argparse, csv, json, os, sys
import numpy as np
from scipy.signal import welch
from scipy.stats import wilcoxon, rankdata

STEP, SEG, FS_F, FS_E = 10, 30, 10.0, 200.0
T0 = -2.0
BANDS = {"theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
PAIRS = {"0v2": (0, 1), "0v3": (0, 2), "2v3": (1, 2)}
CLS = {0: "0-back", 1: "2-back", 2: "3-back"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o): run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def auc(a, b):
    """P(a > b) + 0.5 P(tie)"""
    r = rankdata(np.concatenate([a, b])); return float((r[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def wil(d):
    d = np.asarray(d, float); return float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0


# ---------------------------------------------------------------- fNIRS time course
def rebuild_global(win):
    """win (n,S,H,W,30,C) -> (n,C,L) spatial mean over cells that carry signal (same rule as fnirs_psd_check.py)"""
    n, S, H, W, T, C = win.shape; L = (S - 1) * STEP + T
    grid = np.abs(win[0]).sum(axis=(0, 3, 4)) > 0; out = np.zeros((n, C, L))
    for k in range(S):
        seg = win[:, k][:, grid] if grid.any() else win[:, k].reshape(n, -1, T, C)
        out[:, :, k * STEP:k * STEP + T] = np.transpose(seg.mean(axis=1), (0, 2, 1))
    return out


def stitch(traces, ranks):
    """traces (n,C,L) windows of ONE trial, ranks (n,) -> (C, Tgrid) at 0.1 s, time t = T0 + rank + j/10; overlap-averaged; returns (t, trace)"""
    n, C, L = traces.shape; Tn = int(max(ranks)) * STEP + L; acc = np.zeros((C, Tn)); cnt = np.zeros(Tn)
    for tr, r in zip(traces, ranks):
        acc[:, int(r) * STEP:int(r) * STEP + L] += tr; cnt[int(r) * STEP:int(r) * STEP + L] += 1
    return T0 + np.arange(Tn) / FS_F, acc / np.maximum(cnt, 1)


def hemo_subject(sd, sid, task=(10, 40), chunk=32):
    win = np.load(os.path.join(sd, f"{sid}_fnirs_windows.npy"), mmap_mode="r"); y = np.load(os.path.join(sd, f"{sid}_labels.npy")); tid = np.load(os.path.join(sd, f"{sid}_trial_ids.npy"))
    ses = np.load(os.path.join(sd, f"{sid}_session_ids.npy")); tr = np.concatenate([rebuild_global(np.asarray(win[i:i + chunk], dtype=np.float32)) for i in range(0, len(y), chunk)])
    key = ses * 100000 + tid; curves = {c: [] for c in CLS}; t = None; metr = {c: [] for c in CLS}
    for k in np.unique(key):
        ii = np.flatnonzero(key == k)
        if len(set(y[ii])) != 1: continue
        t, st = stitch(tr[ii], np.arange(len(ii))); c = int(y[ii][0]); curves[c].append(st)
        m = (t >= task[0]) & (t <= task[1]); metr[c].append(st[:, m].mean(1) if m.any() else np.full(st.shape[0], np.nan))
    Tm = min(min(len(x[0]) for x in v) for v in curves.values() if v)
    mean_curve = {c: np.mean([x[:, :Tm] for x in v], 0) for c, v in curves.items() if v}
    return t[:Tm], mean_curve, {c: np.nanmean(np.array(v), 0) for c, v in metr.items() if v}      # metric: (C,) mean HbO, HbR in the task window


def hemo_group(per):
    ids = list(per); out = {}; rows = []
    for ci, ch in enumerate(("HbO", "HbR")):
        for nm, (a, b) in (("2back-0back", (1, 0)), ("3back-0back", (2, 0)), ("3back-2back", (2, 1))):
            d = np.array([per[s][a][ci] - per[s][b][ci] for s in ids]); rows.append((ch, nm, float(d.mean()), int((d > 0).sum()), wil(d)))
    ph = holm([r[4] for r in rows])
    for r, p in zip(rows, ph): out[f"{r[0]}:{r[1]}"] = dict(mean_diff=r[2], subjects_positive=r[3], n=len(ids), p=r[4], p_holm=float(p))
    o = lambda: out["HbO:3back-2back"]["mean_diff"] < 0 and out["HbO:3back-2back"]["p_holm"] < 0.05 and out["HbO:2back-0back"]["mean_diff"] > 0 and out["HbO:2back-0back"]["p_holm"] < 0.05
    out["inverted_U_supported"] = bool(o()); return out


# ---------------------------------------------------------------- EEG
def band_features(x):
    """x (n,28,2000) -> dict of (n,) global features: log band power theta/alpha/beta and theta/alpha log-ratio"""
    f, p = welch(x, fs=FS_E, nperseg=400, noverlap=200, axis=-1); p = p.mean(axis=1)                    # mean PSD over channels (n,F)
    bp = {k: np.log(p[:, (f >= lo) & (f < hi)].sum(1) + 1e-30) for k, (lo, hi) in BANDS.items()}
    bp["theta/alpha"] = bp["theta"] - bp["alpha"]; return bp


def eeg_subject(sd, sid, chunk=64):
    x = np.load(os.path.join(sd, f"{sid}_eeg_channel_windows.npy"), mmap_mode="r"); y = np.load(os.path.join(sd, f"{sid}_labels.npy")); ses = np.load(os.path.join(sd, f"{sid}_session_ids.npy"))
    F = {}
    for i in range(0, len(y), chunk):
        for k, v in band_features(np.asarray(x[i:i + chunk], dtype=np.float64)).items(): F.setdefault(k, []).append(v)
    F = {k: np.concatenate(v) for k, v in F.items()}
    for k in F:                                                                                           # per-session z-score
        for s in np.unique(ses): m = ses == s; F[k][m] = (F[k][m] - F[k][m].mean()) / (F[k][m].std() + 1e-12)
    return {k: {p: auc(F[k][y == b], F[k][y == a]) for p, (a, b) in PAIRS.items() if (y == a).any() and (y == b).any()} for k in F}


def eeg_group(per):
    ids = list(per); feats = list(next(iter(per.values()))); rows = []; out = {}
    for k in feats:
        for p in PAIRS:
            v = np.array([per[s][k][p] for s in ids]); rows.append((k, p, float(v.mean()), int((v > 0.5).sum()), wil(v - 0.5)))
    ph = holm([r[4] for r in rows])
    for r, q in zip(rows, ph): out[f"{r[0]}:{r[1]}"] = dict(mean_auc=r[2], subjects_above_half=r[3], n=len(ids), p=r[4], p_holm=float(q))
    for p in PAIRS:
        d = np.array([abs(per[s]["theta/alpha"][p] - .5) - abs(per[s]["alpha"][p] - .5) for s in ids]); out[f"ratio_minus_alpha_abs_effect:{p}"] = dict(mean=float(d.mean()), p=wil(d))
    return out


# ---------------------------------------------------------------- regional EEG
FRONTAL = ["Fp1", "AFF5h", "AFz", "F1", "Fp2", "AFF6h", "F2"]
PARIETAL = ["P7", "P3", "Pz", "POz", "P4", "P8"]
REG_FEATS = ("frontal_theta", "frontal_alpha", "parietal_theta", "parietal_alpha")
PRESTATED = (("frontal_theta", "up"), ("parietal_alpha", "down"))


def region_groups(names):
    miss = [n for n in FRONTAL + PARIETAL if n not in names]
    if miss: raise SystemExit(f"channel names missing from the channel list: {miss}; the list must be the 28 names of check_channel_order.py in data order")
    return {"frontal": [names.index(n) for n in FRONTAL], "parietal": [names.index(n) for n in PARIETAL]}


def region_band_logs(x, groups):
    """x (n,28,2000) -> {feature: (n,)} log band power of the region-averaged PSD"""
    f, p = welch(x, fs=FS_E, nperseg=400, noverlap=200, axis=-1); out = {}
    for reg, idx in groups.items():
        pr = p[:, idx, :].mean(axis=1)
        for b in ("theta", "alpha"):
            lo, hi = BANDS[b]; out[f"{reg}_{b}"] = np.log(pr[:, (f >= lo) & (f < hi)].sum(1) + 1e-30)
    return out


def regional_subject(sd, sid, groups, chunk=64):
    x = np.load(os.path.join(sd, f"{sid}_eeg_channel_windows.npy"), mmap_mode="r"); y = np.load(os.path.join(sd, f"{sid}_labels.npy")); ses = np.load(os.path.join(sd, f"{sid}_session_ids.npy")); F = {}
    for i in range(0, len(y), chunk):
        for k, v in region_band_logs(np.asarray(x[i:i + chunk], dtype=np.float64), groups).items(): F.setdefault(k, []).append(v)
    F = {k: np.concatenate(v) for k, v in F.items()}
    for k in F:
        for s_ in np.unique(ses): m = ses == s_; F[k][m] = (F[k][m] - F[k][m].mean()) / (F[k][m].std() + 1e-12)
    return {k: {p: auc(F[k][y == b], F[k][y == a]) for p, (a, b) in PAIRS.items() if (y == a).any() and (y == b).any()} for k in F}


def regional_group(per):
    ids = list(per); out = {}; rows = []
    for k, _ in PRESTATED:
        for p in PAIRS:
            v = np.array([per[s][k][p] for s in ids]); rows.append((k, p, float(v.mean()), int((v > 0.5).sum()), wil(v - 0.5)))
    ph = holm([r[4] for r in rows])
    for r, q in zip(rows, ph): out[f"{r[0]}:{r[1]}"] = dict(family="pre-stated", mean_auc=r[2], subjects_above_half=r[3], n=len(ids), p=r[4], p_holm=float(q))
    for k in ("parietal_theta", "frontal_alpha"):
        for p in PAIRS:
            v = np.array([per[s][k][p] for s in ids]); out[f"{k}:{p}"] = dict(family="exploratory control", mean_auc=float(v.mean()), subjects_above_half=int((v > 0.5).sum()), n=len(ids), p=wil(v - 0.5))
    for p in PAIRS:
        d1 = np.array([per[s]["frontal_theta"][p] - per[s]["parietal_theta"][p] for s in ids]); d2 = np.array([per[s]["parietal_alpha"][p] - per[s]["frontal_alpha"][p] for s in ids])
        out[f"specificity_theta(frontal-parietal):{p}"] = dict(mean=float(d1.mean()), p=wil(d1)); out[f"specificity_alpha(parietal-frontal):{p}"] = dict(mean=float(d2.mean()), p=wil(d2))
    return out


def regional_report(per, out_dir=None):
    g = regional_group(per); print(f"\nEEG REGIONAL BAND POWER, n = {len(per)}; AUC = P(second class of the pair higher). Pre-stated: frontal theta > 0.5 (up), parietal alpha < 0.5 (down)")
    for k, v in g.items():
        if "mean_auc" in v: print(f"  {k:24s} AUC {v['mean_auc']:.3f}  >0.5 in {v['subjects_above_half']}/{v['n']}  p {v['p']:.3g}" + (f"  Holm {v['p_holm']:.3g}" if "p_holm" in v else "  (exploratory)"))
        else: print(f"  {k:46s} mean {v['mean']:+.4f}  p {v['p']:.3g}")
    print("  reading: a pre-stated effect counts only if it has the predicted direction and Holm < 0.05; 'specificity' rows say whether the region differs from the other region (paired Wilcoxon, exploratory; expected sign: theta(frontal-parietal) POSITIVE, alpha(parietal-frontal) NEGATIVE).")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True); json.dump({"eeg_regional": g}, open(os.path.join(out_dir, "claims_signal_regional.json"), "w"), indent=2)
        import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt
        ids = list(per); lab = {"frontal_theta": "frontal\ntheta", "frontal_alpha": "frontal\nalpha", "parietal_theta": "parietal\ntheta", "parietal_alpha": "parietal\nalpha"}
        col = {"frontal_theta": fs.C["red"], "parietal_alpha": fs.C["blue"], "frontal_alpha": fs.C["grey"], "parietal_theta": fs.C["grey"]}
        fig, ax = plt.subplots(1, 3, figsize=(fs.W2, 2.7), sharey=True)
        for a_, p in zip(ax, PAIRS):
            for j, k in enumerate(REG_FEATS):
                v = np.array([per[s][k][p] for s in ids]); fs.strip(a_, j, v, color=col[k], width=0.15); fs.mean_ci(a_, j, v, color="k")
            a_.axhline(0.5, ls="--", color="k", lw=0.6); a_.set_xticks(range(4)); a_.set_xticklabels([lab[k] for k in REG_FEATS], fontsize=6.3); a_.set_title(p, fontsize=8)
        ax[0].set_ylabel("AUC (second class of the pair higher)"); [fs.panel(a_, l) for a_, l in zip(ax, "abc")]
        fig.suptitle(f"Regional EEG band power as load marker (n = {len(ids)} subjects; coloured = pre-stated)", fontsize=8); fig.tight_layout(); print("figure:", fs.save(fig, out_dir, "fig_eeg_regional"))


# ---------------------------------------------------------------- figures
def figures(hemo, eeg, curves, out_dir):
    import fig_style as fs; fs.apply(); import matplotlib.pyplot as plt; res = []
    if hemo:
        fig, ax = plt.subplots(1, 3, figsize=(fs.W2, 2.7), gridspec_kw={"width_ratios": [2, 2, 1.6]}); cc = {0: fs.C["blue"], 1: fs.C["orange"], 2: fs.C["red"]}
        ids = list(curves); Tc = min(len(curves[s][0]) for s in ids)                                  # subjects differ in how many windows (time points) they have: common length
        for ci, a in enumerate(ax[:2]):
            for c in CLS:
                A = np.array([curves[s][1][c][ci][:Tc] for s in ids if c in curves[s][1]]); t = curves[ids[0]][0][:Tc]
                sub = slice(None, None, 10); lo, hi = np.array([fs.boot_ci(A[:, j], n=500) for j in range(0, A.shape[1], 10)]).T
                a.fill_between(t[sub], lo, hi, color=cc[c], alpha=0.2, lw=0); a.plot(t[sub], A.mean(0)[sub], color=cc[c], label=CLS[c])
            a.axvline(0, color="k", lw=0.5); a.set_xlabel("Time from block onset (s)"); a.set_ylabel(("HbO", "HbR")[ci] + " (global mean, a.u.)"); a.set_title(("HbO", "HbR")[ci], fontsize=8)
        ax[0].legend(frameon=False, fontsize=6)
        names = ["2b-0b", "3b-0b", "3b-2b"]; keys = ["2back-0back", "3back-0back", "3back-2back"]
        for j, k in enumerate(keys):
            d = np.array([ids_ for ids_ in []]) if False else np.array([hemo_pers[s][a_][0] - hemo_pers[s][b_][0] for s in ids for hemo_pers in [HEMO_CACHE] for a_, b_ in [dict(zip(keys, ((1, 0), (2, 0), (2, 1))))[k]]])
            fs.strip(ax[2], j, d, color=fs.C["blue"]); fs.mean_ci(ax[2], j, d, color="k")
        ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xticks(range(3)); ax[2].set_xticklabels(names); ax[2].set_ylabel("HbO difference, task window")
        for a, l in zip(ax, "abc"): fs.panel(a, l)
        fig.suptitle(f"Global haemodynamic response by load (n = {len(ids)} subjects)", fontsize=8); fig.tight_layout(); res.append(fs.save(fig, out_dir, "fig_hemodynamics"))
    if eeg:
        ids = list(eeg); feats = list(eeg[ids[0]]); fig, ax = plt.subplots(1, 3, figsize=(fs.W2, 2.7), sharey=True)
        for a, p in zip(ax, PAIRS):
            for j, k in enumerate(feats):
                v = np.array([eeg[s][k][p] for s in ids]); fs.strip(a, j, v, color=fs.C["blue"], horizontal=False); fs.mean_ci(a, j, v, color="k")
            a.axhline(0.5, ls="--", color="k", lw=0.6); a.set_xticks(range(len(feats))); a.set_xticklabels(feats, rotation=30, fontsize=6.5); a.set_title(p.replace("v", " vs ") + " (AUC > 0.5: higher in the higher load)" if False else p, fontsize=8)
        ax[0].set_ylabel("AUC (second class of the pair higher)"); [fs.panel(a, l) for a, l in zip(ax, "abc")]
        fig.suptitle(f"EEG band power as load marker, global over channels (n = {len(ids)} subjects)", fontsize=8); fig.tight_layout(); res.append(fs.save(fig, out_dir, "fig_eeg_theta_alpha"))
    return res


HEMO_CACHE = {}


def report(hemo_per, curves, eeg_per, out_dir):
    HEMO_CACHE.clear(); HEMO_CACHE.update(hemo_per or {}); out = {}
    if hemo_per and curves:
        ends = np.array([float(curves[s][0][-1]) for s in curves]); print(f"\nTIME COVERAGE of the stitched traces: last time point min {ends.min():.1f} s, median {np.median(ends):.1f} s, max {ends.max():.1f} s -> the task window actually averaged is 10 s to min(40 s, that end) per subject")
    if hemo_per:
        g = hemo_group(hemo_per); out["hemo"] = g; print(f"\nHAEMODYNAMICS, n = {len(hemo_per)}, global mean, task window; positive = first minus second")
        for k, v in g.items():
            if isinstance(v, dict): print(f"  {k:18s} diff {v['mean_diff']:+.4f}  positive in {v['subjects_positive']}/{v['n']}  p {v['p']:.3g}  Holm {v['p_holm']:.3g}")
        print(f"  inverted-U (2b>0b and 3b<2b, both Holm<0.05): {'SUPPORTED' if g['inverted_U_supported'] else 'NOT SUPPORTED'}")
    if eeg_per:
        g = eeg_group(eeg_per); out["eeg"] = g; print(f"\nEEG BAND POWER (global), n = {len(eeg_per)}; AUC = P(second class of the pair higher), predicted: theta>0.5, alpha<0.5, ratio>0.5 for 0v2/0v3")
        for k, v in g.items():
            if "mean_auc" in v: print(f"  {k:18s} AUC {v['mean_auc']:.3f}  >0.5 in {v['subjects_above_half']}/{v['n']}  p {v['p']:.3g}  Holm {v['p_holm']:.3g}")
            else: print(f"  {k:36s} mean {v['mean']:+.4f}  p {v['p']:.3g}  (positive = ratio carries more than alpha alone)")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True); json.dump(out, open(os.path.join(out_dir, "claims_signal_tests.json"), "w"), indent=2)
        print("figures:", figures(hemo_per, eeg_per, curves, out_dir))


# ---------------------------------------------------------------- self-test
def _fake_subject(d, sid, rng, amp, R=8):
    S, H, L = 21, 4, (21 - 1) * STEP + SEG; os.makedirs(d, exist_ok=True); fw, y, ses, tid, eeg = [], [], [], [], []
    t_all = lambda r: T0 + r + np.arange(L) / FS_F
    for s in (1, 2):
        for k, c in enumerate([0, 1, 2, 0, 1, 2]):
            for r in range(R):
                t = t_all(r); g = amp[c] * np.clip(t / 5, 0, 1) * (t > 0) + 0.01 * rng.randn(L)
                w = np.zeros((S, H, H, SEG, 2), np.float32)
                for q in range(S): w[q, :, :, :, 0] = g[q * STEP:q * STEP + SEG]; w[q, :, :, :, 1] = -0.5 * g[q * STEP:q * STEP + SEG]
                fw.append(w); y.append(c); ses.append(s); tid.append(k)
                tt = np.arange(2000) / FS_E; th = [0.5, 1.0, 1.5][c]; al = [1.5, 1.0, 0.5][c]
                eeg.append((th * np.sin(2 * np.pi * 6 * tt)[None] + al * np.sin(2 * np.pi * 10 * tt + 1)[None] + 0.3 * rng.randn(28, 2000)).astype(np.float32))
    for n, a in (("fnirs_windows", np.stack(fw)), ("labels", np.array(y)), ("session_ids", np.array(ses)), ("trial_ids", np.array(tid)), ("eeg_channel_windows", np.stack(eeg))): np.save(os.path.join(d, f"{sid}_{n}.npy"), a)


def self_test():
    import tempfile
    rng = np.random.RandomState(0)
    tr = rng.randn(3, 2, 230); t, st = stitch(np.stack([np.concatenate([np.zeros((2, 0)), np.ones((2, 230))], 1) * (i + 1) for i in range(3)]), np.arange(3)); assert st.shape[1] == 2 * STEP + 230 and abs(st[0, 5] - 1) < 1e-9 and abs(st[0, -1] - 3) < 1e-9
    print("[1] stitching places windows on one time axis and averages overlaps")
    a, b = rng.randn(200) + 2, rng.randn(200); assert auc(a, b) > 0.85 and abs(auc(b, b.copy()) - .5) < .01; assert np.allclose(holm([.01, .04, .03]), [.03, .06, .06]); print("[2] AUC and Holm")
    d = tempfile.mkdtemp(); hp, ep, cv = {}, {}, {}
    for i in range(10):
        sid = f"VP{i:03d}"; _fake_subject(os.path.join(d, sid), sid, rng, amp=[0.0, 1.0 + .1 * rng.randn(), 0.5 + .1 * rng.randn()], R=8 if i % 2 else 7)   # unequal lengths across subjects, as in the real data
        tt, mc, me = hemo_subject(os.path.join(d, sid), sid, task=(5, 20)); hp[sid] = me; cv[sid] = (tt, mc); ep[sid] = eeg_subject(os.path.join(d, sid), sid)
    g = hemo_group(hp); assert g["HbO:2back-0back"]["mean_diff"] > 0.5 and g["HbO:3back-2back"]["mean_diff"] < 0 and g["inverted_U_supported"], g
    assert g["HbR:2back-0back"]["mean_diff"] < 0; print("[3] injected inverted-U in HbO recovered, HbR opposite")
    e = eeg_group(ep); assert e["theta:0v3"]["mean_auc"] > 0.9 and e["alpha:0v3"]["mean_auc"] < 0.1 and e["theta/alpha:0v3"]["p_holm"] < 0.05; print("[4] injected theta-up / alpha-down recovered")
    report(hp, cv, ep, tempfile.mkdtemp()); print("[5] report and figures written")
    names = ["Fp1", "AFF5h", "AFz", "F1", "FC5", "FC1", "T7", "C3", "Cz", "CP5", "CP1", "P7", "P3", "Pz", "POz", "O1", "Fp2", "AFF6h", "F2", "FC2", "FC6", "C4", "T8", "CP2", "CP6", "P4", "P8", "O2"]
    gr = region_groups(names); fr, pa = gr["frontal"], gr["parietal"]; rp = {}
    for i in range(10):
        sid = f"VQ{i:03d}"; sd = os.path.join(d, sid); os.makedirs(sd, exist_ok=True); y, ses, eeg = [], [], []; tt = np.arange(2000) / FS_E
        for s_ in (1, 2):
            for c in [0, 1, 2] * 4:
                th = np.zeros(28); th[fr] = [0.0, 1.0, 1.6][c]; th += 0.3; al = np.zeros(28); al[pa] = [1.6, 1.0, 0.0][c]; al += 0.3
                eeg.append((th[:, None] * np.sin(2 * np.pi * 6 * tt)[None] + al[:, None] * np.sin(2 * np.pi * 10 * tt + 1)[None] + 0.3 * rng.randn(28, 2000)).astype(np.float32)); y.append(c); ses.append(s_)
        np.save(os.path.join(sd, f"{sid}_eeg_channel_windows.npy"), np.stack(eeg)); np.save(os.path.join(sd, f"{sid}_labels.npy"), np.array(y)); np.save(os.path.join(sd, f"{sid}_session_ids.npy"), np.array(ses))
        rp[sid] = regional_subject(sd, sid, gr)
    g = regional_group(rp); assert g["frontal_theta:0v3"]["mean_auc"] > 0.9 and g["parietal_alpha:0v3"]["mean_auc"] < 0.1 and g["frontal_theta:0v3"]["p_holm"] < 0.05, g
    assert abs(g["parietal_theta:0v3"]["mean_auc"] - 0.5) < 0.2 and g["specificity_theta(frontal-parietal):0v3"]["mean"] > 0.2, g
    print("[6] regional test recovers an injected frontal-theta rise and parietal-alpha fall, with controls near 0.5 and positive specificity")
    regional_report(rp, tempfile.mkdtemp()); print("[7] regional report and figure written\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--out-dir", default=None); ap.add_argument("--task-start", type=float, default=10); ap.add_argument("--task-end", type=float, default=40)
    ap.add_argument("--skip", choices=("none", "hemo", "eeg"), default="none"); ap.add_argument("--channel-names", default=None, help="channels.txt (28 names, data order, confirmed by check_channel_order.py)"); ap.add_argument("--regional-only", action="store_true", help="only the regional EEG test (needs --channel-names)"); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.data_root: sys.exit("need --data-root")
    lo, hi = a.subjects.split("-") if "-" in a.subjects else (a.subjects, a.subjects); hp, ep, cv = {}, {}, {}
    if a.regional_only or a.channel_names:
        if not a.channel_names: sys.exit("--regional-only needs --channel-names")
        names = [t.strip() for t in open(a.channel_names, encoding="utf-8").read().replace(",", "\n").split("\n") if t.strip()]
        if len(names) != 28: sys.exit(f"--channel-names has {len(names)} names, expected 28")
        gr = region_groups(names); rp = {}
        for i in range(int(lo), int(hi) + 1):
            sid = f"VP{i:03d}"
            try: rp[sid] = regional_subject(os.path.join(a.data_root, sid), sid, gr); print(f"[{sid}] regional done", flush=True)
            except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
        if len(rp) < 6: sys.exit("fewer than 6 subjects")
        regional_report(rp, a.out_dir)
        if a.regional_only: return
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"; sd = os.path.join(a.data_root, sid)
        try:
            if a.skip != "hemo": t, mc, me = hemo_subject(sd, sid, (a.task_start, a.task_end)); hp[sid] = me; cv[sid] = (t, mc)
            if a.skip != "eeg": ep[sid] = eeg_subject(sd, sid)
            print(f"[{sid}] done", flush=True)
        except Exception as e: print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if max(len(hp), len(ep)) < 6: sys.exit("fewer than 6 subjects; group tests need >= 6")
    report(hp, cv, ep, a.out_dir)


if __name__ == "__main__":
    main()
