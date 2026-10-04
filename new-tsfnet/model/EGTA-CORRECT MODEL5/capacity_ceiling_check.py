"""
capacity_ceiling_check.py -- does 3-back push some subjects past their working-
memory capacity, and does that explain the 2-back / 3-back decoding errors?

HYPOTHESIS (cognitive neuroscience)
Prefrontal activity follows an inverted U with working-memory load: it rises
up to a subject's capacity and falls once they overload or disengage
(Callicott et al., 1999, Cereb. Cortex). If some subjects pass capacity at
3-back, their prefrontal response to 3-back is equal to or BELOW their 2-back
response -- the opposite sign to subjects who are still within capacity. A
classifier then faces a 2/3 boundary whose direction differs between subjects.

WHAT IS MEASURED (per n-back block, from the cleaned recordings preprocessing
saved, same sources and windows as diag_2v3_separability.py)
  hbo_pfc   prefrontal ("AF*" optodes) HbO, task period (from +7 s, HRF lag)
            minus pre-block baseline [-5, -2) s       literature: UP with load
  hbr_pfc   the same for HbR                          literature: DOWN with load
  hbo_left / hbo_right   HbO over left (odd) / right (even) prefrontal optodes
  p300_pz   digit-locked P300, mean 0.30-0.50 s at Pz  literature: DOWN with load
  theta_fm  log 4-8 Hz power, AFz/F1/F2/FC1/FC2       literature: UP with load

STATISTICS (per subject, per measure)
  delta     mean(3-back) - mean(2-back) over the 9 + 9 blocks
  d         Cohen's d of the same contrast (pooled SD)
  p_perm    EXACT permutation test: within each session the 3 + 3 2/3 blocks
            are exchangeable under H0, so all C(6,3)^3 = 8000 relabelings are
            enumerated (no random sampling); two-sided
  q         Benjamini-Hochberg across subjects, per measure
  sessions  in how many of the 3 sessions delta has the subject's overall sign
            (a trait should repeat across sessions; noise does not)
  shape     using 0/2/3 means: "monotonic" (0<2<3), "inverted-U" (2 highest),
            or "other"  -- for the measures where load raises the signal
GROUP
  Wilcoxon on delta across subjects (is there a group-level 3 vs 2 effect?),
  sign counts, and heterogeneity: Cochran's Q and I^2 of the per-subject deltas
  (is the between-subject spread larger than within-subject noise explains?).
  High I^2 with mixed signs = subjects genuinely follow different patterns,
  the warning for cross-subject pretraining (step 8).
LINK TO THE CLASSIFIER (optional, needs trial_level_eval *_trials.csv files)
  per subject, run60's 2-vs-3 accuracy on the 2/3 blocks it did not call
  0-back (pooled over the seeds given), Spearman-correlated with the subject's
  signed and absolute HbO / P300 effect sizes.

FIGURES (PNG, for the thesis)
  capacity_effects.png   per-subject Cohen's d (3-back minus 2-back) for four
                         measures, same subject order in every panel; blue =
                         direction the literature predicts, red = reversed;
                         filled = q < 0.05, hollow = not significant
  capacity_profiles.png  per-subject prefrontal HbO across 0/2/3-back
                         (within-subject z), group mean in black; significant
                         reversals drawn in red
  capacity_vs_decoding.png  HbO effect size vs run60 2-vs-3 decoding accuracy
                         (only with --trials-csv)

USAGE
    python capacity_ceiling_check.py
    python capacity_ceiling_check.py --trials-csv "<...>\\trial_level_eval_seed42_trials.csv" ^
                                                  "<...>\\trial_level_eval_seed999_trials.csv" ^
                                                  "<...>\\trial_level_eval_seed2024_trials.csv"
"""

import argparse
import csv
import itertools
import json
import os
import re
import sys

import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import binomtest, chi2, spearmanr, wilcoxon

p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
p.add_argument("--raw-root", default=r"E:\kntu\Thesis\codes\DATA")
p.add_argument("--clean-root", default=r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback")
p.add_argument("--out-dir", default=None, help="default: <clean-root>/capacity_check")
p.add_argument("--subjects", default="1-26")
p.add_argument("--trials-csv", nargs="*", default=[],
               help="trial_level_eval_seed*_trials.csv of run60 (any number of seeds)")
ARGS = p.parse_args()

FS_EEG, FS_NIRS = 200.0, 10.0
BLOCK_EEG = {112: 0, 128: 1, 144: 2}
BLOCK_NIRS = {7: 0, 8: 1, 9: 2}
STIM = {16, 48, 64, 80, 96}                      # timing only
PER_SESSION = 9
THETA_CH = ["AFz", "F1", "F2", "FC1", "FC2"]
MEASURES = ["hbo_pfc", "hbr_pfc", "hbo_left", "hbo_right", "p300_pz", "theta_fm"]
EXPECTED_SIGN = {"hbo_pfc": +1, "hbr_pfc": -1, "hbo_left": +1, "hbo_right": +1,
                 "p300_pz": -1, "theta_fm": +1}
PLOT_MEASURES = ["hbo_pfc", "hbr_pfc", "p300_pz", "theta_fm"]
LABELS = {"hbo_pfc": "Prefrontal HbO", "hbr_pfc": "Prefrontal HbR", "p300_pz": "P300 at Pz",
          "theta_fm": "Frontal theta", "hbo_left": "Left PFC HbO", "hbo_right": "Right PFC HbO"}


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


def _first(path):
    m = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    return m[[k for k in m if not k.startswith("__")][0]]


def eeg_markers(folder):
    cnt = _first(os.path.join(folder, "cnt_nback.mat"))
    fs = FS_EEG if float(cnt.fs) != FS_EEG else float(cnt.fs)
    mrk = _first(os.path.join(folder, "mrk_nback.mat"))
    pos = (np.round(np.asarray(mrk.time, float) * fs / 1000).astype(int) if hasattr(mrk, "time")
           else np.asarray(mrk.pos, int) - 1)
    return pos, np.asarray(mrk.event.desc, int).flatten()


def nirs_markers(folder):
    cnt = _first(os.path.join(folder, "cnt_nback.mat"))
    mrk = _first(os.path.join(folder, "mrk_nback.mat"))
    t = (np.asarray(mrk.time, float) / 1000 if hasattr(mrk, "time")
         else np.asarray(mrk.pos, float) / float(getattr(cnt, "fs", FS_NIRS)))
    return t, np.asarray(mrk.event.desc, int).flatten()


def blocks_of(pos, codes, table):
    idx = np.flatnonzero(np.isin(codes, list(table)))
    return [(pos[i], table[int(codes[i])], k // PER_SESSION + 1) for k, i in enumerate(idx)]


def side(name):
    """10-5 convention: odd number = left, even = right, z = midline."""
    m = re.search(r"(\d+)", name.split()[0])
    return None if not m else ("left" if int(m.group(1)) % 2 else "right")


def subject_blocks(sid):
    import mne
    mne.set_log_level("ERROR")
    d = os.path.join(ARGS.clean_root, sid)
    eeg = mne.io.read_raw_fif(os.path.join(d, f"{sid}_eeg_clean_raw.fif"), preload=True)
    nir = mne.io.read_raw_fif(os.path.join(d, f"{sid}_fnirs_clean_raw.fif"), preload=True)
    pos, codes = eeg_markers(os.path.join(ARGS.raw_root, f"{sid}-EEG"))
    t_on, ncodes = nirs_markers(os.path.join(ARGS.raw_root, f"{sid}-NIRS"))
    eb, nb = blocks_of(pos, codes, BLOCK_EEG), blocks_of(t_on, ncodes, BLOCK_NIRS)
    if len(eb) != 27 or len(nb) != 27 or [b[1] for b in eb] != [b[1] for b in nb]:
        raise ValueError(f"block tables disagree (EEG {len(eb)}, fNIRS {len(nb)})")

    X = eeg.get_data(picks="eeg"); ch = {n: i for i, n in enumerate(eeg.copy().pick("eeg").ch_names)}
    N = nir.get_data(); names = nir.ch_names
    hbo = [i for i, n in enumerate(names) if n.endswith("hbo") and n.startswith("AF")]
    hbr = [i for i, n in enumerate(names) if n.endswith("hbr") and n.startswith("AF")]
    left = [i for i in hbo if side(names[i]) == "left"]
    right = [i for i in hbo if side(names[i]) == "right"]
    tt = np.arange(N.shape[1]) / FS_NIRS
    stims = np.sort(pos[np.isin(codes, list(STIM))])
    pz = ch.get("Pz")
    th = [ch[c] for c in THETA_CH if c in ch]

    rows = []
    for k, ((on, lab, sess), (ton, _, _)) in enumerate(zip(eb, nb)):
        end = eb[k + 1][0] if k + 1 < len(eb) else X.shape[1]
        base = (tt >= ton - 5) & (tt < ton - 2)
        late = (tt >= ton + 7) & (tt < ton + 42)
        r = {"label": lab, "session": sess}
        for key, idx in (("hbo_pfc", hbo), ("hbr_pfc", hbr), ("hbo_left", left), ("hbo_right", right)):
            r[key] = (float(N[idx][:, late].mean() - N[idx][:, base].mean())
                      if idx and base.any() and late.any() else np.nan)
        sp = stims[(stims > on) & (stims < end)]
        eps = []
        for s in sp:
            if s - 40 >= 0 and s + 160 <= X.shape[1] and pz is not None:
                e = X[pz, s - 40:s + 160]
                eps.append(e - e[:40].mean())
        r["p300_pz"] = float(np.mean(eps, axis=0)[100:140].mean()) if eps else np.nan   # 0.30-0.50 s
        if len(sp) >= 2 and th:
            seg = X[th, sp[0]:min(sp[-1] + int(2 * FS_EEG), X.shape[1])]
            f, pxx = welch(seg, fs=FS_EEG, nperseg=int(2 * FS_EEG), axis=1)
            r["theta_fm"] = float(np.log(pxx[:, (f >= 4) & (f < 8)].mean() + 1e-30))
        else:
            r["theta_fm"] = np.nan
        rows.append(r)
    return rows


def exact_perm_p(v, lab, sess):
    """Two-sided exact permutation p for mean(3) - mean(2), labels exchanged
    only within session (all C(n,n3) relabelings per session, full product)."""
    m = np.isin(lab, [1, 2]) & ~np.isnan(v)
    v, lab, sess = v[m], lab[m], sess[m]
    n3 = (lab == 2).sum(); n2 = (lab == 1).sum()
    if n3 == 0 or n2 == 0:
        return np.nan
    per_session = []
    for s in np.unique(sess):
        vs = v[sess == s]; k3 = int((lab[sess == s] == 2).sum())
        sums = np.array([vs[list(c)].sum() for c in itertools.combinations(range(len(vs)), k3)])
        per_session.append(sums)                   # sum of the values labelled 3-back
    tot3 = per_session[0]
    for ps in per_session[1:]:
        tot3 = (tot3[:, None] + ps[None, :]).ravel()
    total = v.sum()
    stats = tot3 / n3 - (total - tot3) / n2
    obs = v[lab == 2].mean() - v[lab == 1].mean()
    return float(np.mean(np.abs(stats) >= abs(obs) - 1e-12))


def bh(pvals):
    p_ = np.asarray(pvals, float); q = np.full_like(p_, np.nan)
    ok = ~np.isnan(p_); idx = np.argsort(p_[ok]); m = ok.sum()
    if m:
        ranked = p_[ok][idx] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m); out[idx] = np.minimum(ranked, 1)
        q[ok] = out
    return q


def decoding_2v3(paths):
    acc = {}
    for path in paths:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            lab, pred = int(r["label"]), int(r["pred"])
            if lab in (1, 2) and pred != 0:
                c = acc.setdefault(r["subject_id"], [0, 0])
                c[0] += int(lab == pred); c[1] += 1
    return {s: c[0] / c[1] for s, c in acc.items() if c[1]}


def figures(out_dir, ids, res, prof, dec):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    BLUE, RED, GRID, INK, MUTED = "#2a78d6", "#e34948", "#e4e3df", "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10})

    order = sorted(ids, key=lambda s: res["hbo_pfc"][s]["d"] * EXPECTED_SIGN["hbo_pfc"])
    fig, axes = plt.subplots(1, len(PLOT_MEASURES), figsize=(11, 0.24 * len(ids) + 1.6), sharey=True)
    for ax, mname in zip(axes, PLOT_MEASURES):
        ax.axvline(0, color="#b4b2ab", lw=1, zorder=1)
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
        for yi, s in enumerate(order):
            d_ = res[mname][s]["d"]; q_ = res[mname][s]["q"]
            if np.isnan(d_):
                continue
            col = BLUE if np.sign(d_) == EXPECTED_SIGN[mname] else RED
            sig = q_ < 0.05
            ax.plot([0, d_], [yi, yi], color=col, lw=1.2, alpha=0.5, zorder=2)
            ax.scatter(d_, yi, s=42, zorder=3, facecolor=col if sig else "white",
                       edgecolor=col, linewidth=1.6)
        arrow = "↑" if EXPECTED_SIGN[mname] > 0 else "↓"
        ax.set_title(f"{LABELS[mname]}\n(literature: {arrow} with load)", color=INK)
        ax.set_xlabel("Cohen's d, 3-back − 2-back")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(order)), order)
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, markersize=7, label="direction as literature predicts"),
               Line2D([], [], marker="o", ls="", color=RED, markersize=7, label="reversed direction"),
               Line2D([], [], marker="o", ls="", markerfacecolor="#777", markeredgecolor="#777", markersize=7,
                      label="filled: q < 0.05 (exact permutation, BH)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777", markersize=7,
                      label="hollow: not significant")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-subject 3-back vs 2-back effect (subjects ordered by prefrontal HbO effect)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(out_dir, "capacity_effects.png"), dpi=200, facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    zs = []
    for s in ids:
        m = np.array(prof[s])
        if np.isnan(m).any():
            continue
        z = (m - m.mean()) / (m.std() + 1e-12); zs.append(z)
        rev = res["hbo_pfc"][s]["q"] < 0.05 and res["hbo_pfc"][s]["d"] < 0
        ax.plot([0, 1, 2], z, color=RED if rev else "#b4b2ab", lw=2 if rev else 1,
                alpha=1 if rev else 0.8, marker="o", markersize=4 if not rev else 6, zorder=3 if rev else 2)
    if zs:
        ax.plot([0, 1, 2], np.mean(zs, axis=0), color=INK, lw=2.5, marker="o", markersize=8, zorder=4)
    ax.set_xticks([0, 1, 2], ["0-back", "2-back", "3-back"])
    ax.set_ylabel("prefrontal HbO (within-subject z)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(handles=[Line2D([], [], color=INK, lw=2.5, marker="o", label="group mean"),
                       Line2D([], [], color="#b4b2ab", lw=1, label="subject"),
                       Line2D([], [], color=RED, lw=2, label="significant 3 < 2 (q < 0.05)")],
              frameon=False, fontsize=8, loc="best")
    ax.set_title("Prefrontal HbO across load levels", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "capacity_profiles.png"), dpi=200, facecolor="white")
    plt.close(fig)

    if dec:
        common = [s for s in ids if s in dec and not np.isnan(res["hbo_pfc"][s]["d"])]
        if len(common) >= 5:
            fig, ax = plt.subplots(figsize=(5.2, 4.2))
            x = np.array([res["hbo_pfc"][s]["d"] for s in common]); y = np.array([dec[s] for s in common])
            ax.axvline(0, color="#b4b2ab", lw=1); ax.axhline(0.5, color="#b4b2ab", lw=1, ls=":")
            ax.scatter(x, y, s=48, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3)
            for s, xi, yi in zip(common, x, y):
                if yi < 0.45 or abs(xi) > np.percentile(np.abs(x), 85):
                    ax.annotate(s, (xi, yi), xytext=(4, 3), textcoords="offset points", fontsize=7, color=MUTED)
            rho, pv = spearmanr(x, y)
            ax.set_xlabel("prefrontal HbO Cohen's d, 3-back − 2-back")
            ax.set_ylabel("run60 2-vs-3 block accuracy")
            ax.set_title(f"Physiological contrast vs decoding (Spearman ρ={rho:.2f}, p={pv:.3g})", color=INK)
            ax.grid(color=GRID, lw=0.6)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "capacity_vs_decoding.png"), dpi=200, facecolor="white")
            plt.close(fig)


def main():
    out_dir = ARGS.out_dir or os.path.join(ARGS.clean_root, "capacity_check")
    os.makedirs(out_dir, exist_ok=True)
    data = {}
    for sid in parse_subjects(ARGS.subjects):
        try:
            data[sid] = subject_blocks(sid)
        except FileNotFoundError as e:
            print(f"[{sid}] SKIP -- {e}")
        except Exception as e:
            print(f"[{sid}] SKIP -- {type(e).__name__}: {e}")
    if len(data) < 6:
        sys.exit("ERROR: fewer than 6 subjects processed -- check --clean-root / --raw-root")
    ids = sorted(data)

    res = {m: {} for m in MEASURES}
    prof = {}
    for s in ids:
        lab = np.array([r["label"] for r in data[s]]); ses = np.array([r["session"] for r in data[s]])
        for m in MEASURES:
            v = np.array([r[m] for r in data[s]], float)
            v2, v3 = v[(lab == 1) & ~np.isnan(v)], v[(lab == 2) & ~np.isnan(v)]
            if len(v2) < 2 or len(v3) < 2:
                res[m][s] = {"delta": np.nan, "d": np.nan, "se": np.nan, "p": np.nan, "sessions": 0}
                continue
            delta = v3.mean() - v2.mean()
            sp = np.sqrt(((len(v2) - 1) * v2.var(ddof=1) + (len(v3) - 1) * v3.var(ddof=1)) / (len(v2) + len(v3) - 2))
            per_sess = [np.nanmean(v[(lab == 2) & (ses == k)]) - np.nanmean(v[(lab == 1) & (ses == k)])
                        for k in np.unique(ses)]
            res[m][s] = {"delta": float(delta), "d": float(delta / (sp + 1e-30)),
                         "se": float(np.sqrt(v2.var(ddof=1) / len(v2) + v3.var(ddof=1) / len(v3))),
                         "p": exact_perm_p(v, lab, ses),
                         "sessions": int(sum(np.sign(x) == np.sign(delta) for x in per_sess if x == x))}
            if m == "hbo_pfc":
                prof[s] = [float(np.nanmean(v[lab == c])) for c in (0, 1, 2)]
    for m in MEASURES:
        q = bh([res[m][s]["p"] for s in ids])
        for s, qq in zip(ids, q):
            res[m][s]["q"] = float(qq)

    dec = decoding_2v3(ARGS.trials_csv) if ARGS.trials_csv else {}
    summary = {"n_subjects": len(ids), "measures": {}}
    print("\n" + "=" * 96)
    print(f"CAPACITY CEILING CHECK -- 3-back minus 2-back, {len(ids)} subjects "
          f"(per-subject exact permutation over 8000 within-session relabelings, BH across subjects)")
    print("=" * 96)
    print(f"{'measure':10s} {'expect':>6s} | {'as exp.':>7s} {'reversed':>8s} | {'sig as exp.':>11s} {'sig rev.':>8s} | "
          f"{'group p':>8s} | {'I^2':>5s} {'Q p':>8s} | {'stable 3/3':>10s}")
    for m in MEASURES:
        d_ = np.array([res[m][s]["delta"] for s in ids]); se = np.array([res[m][s]["se"] for s in ids])
        q_ = np.array([res[m][s]["q"] for s in ids]); ok = ~np.isnan(d_)
        exp = EXPECTED_SIGN[m]
        as_exp = int((np.sign(d_[ok]) == exp).sum()); rev = int((np.sign(d_[ok]) == -exp).sum())
        sig_exp = int(((q_ < 0.05) & (np.sign(d_) == exp)).sum()); sig_rev = int(((q_ < 0.05) & (np.sign(d_) == -exp)).sum())
        gp = wilcoxon(d_[ok]).pvalue if ok.sum() >= 6 else np.nan
        w = 1 / se[ok] ** 2; mu = (w * d_[ok]).sum() / w.sum()
        Q = float((w * (d_[ok] - mu) ** 2).sum()); df = ok.sum() - 1
        I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0; Qp = float(chi2.sf(Q, df))
        stable = int(sum(res[m][s]["sessions"] == 3 for s in ids))
        summary["measures"][m] = {"expected_sign": exp, "n_as_expected": as_exp, "n_reversed": rev,
                                  "n_sig_as_expected": sig_exp, "n_sig_reversed": sig_rev,
                                  "sign_test_p": float(binomtest(as_exp, as_exp + rev, 0.5).pvalue) if as_exp + rev else np.nan,
                                  "group_wilcoxon_p": float(gp), "I2": I2, "Q": Q, "Q_p": Qp,
                                  "n_sign_stable_all_3_sessions": stable}
        print(f"{m:10s} {'+' if exp > 0 else '-':>6s} | {as_exp:7d} {rev:8d} | {sig_exp:11d} {sig_rev:8d} | "
              f"{gp:8.3g} | {I2:5.2f} {Qp:8.3g} | {stable:6d}/{len(ids)}")

    shapes = {"monotonic": [], "inverted-U": [], "other": []}
    for s in ids:
        m0, m2, m3 = prof.get(s, [np.nan] * 3)
        key = ("monotonic" if m0 < m2 < m3 else "inverted-U" if (m2 > m3 and m2 > m0) else "other")
        shapes[key].append(s)
    summary["hbo_shape"] = shapes
    print(f"\nPrefrontal HbO shape across 0/2/3-back: monotonic {len(shapes['monotonic'])}, "
          f"inverted-U {len(shapes['inverted-U'])} {shapes['inverted-U']}, other {len(shapes['other'])}")
    rev_sig = [s for s in ids if res["hbo_pfc"][s]["q"] < 0.05 and np.sign(res["hbo_pfc"][s]["d"]) == -1]
    print(f"Significant prefrontal-HbO reversals (3 < 2, q < 0.05): {rev_sig or 'none'}")

    if dec:
        common = [s for s in ids if s in dec]
        summary["decoding_link"] = {}
        print(f"\nLINK TO run60 2-vs-3 decoding ({len(common)} subjects, from {len(ARGS.trials_csv)} trials file(s)):")
        for m in ("hbo_pfc", "p300_pz", "theta_fm"):
            x = np.array([res[m][s]["d"] for s in common]); y = np.array([dec[s] for s in common])
            ok = ~np.isnan(x)
            r_s, p_s = spearmanr(x[ok], y[ok]); r_a, p_a = spearmanr(np.abs(x[ok]), y[ok])
            summary["decoding_link"][m] = {"rho_signed": float(r_s), "p_signed": float(p_s),
                                           "rho_abs": float(r_a), "p_abs": float(p_a)}
            print(f"  {m:9s}: Spearman rho(signed d, acc) = {r_s:+.2f} (p={p_s:.3g}) | "
                  f"rho(|d|, acc) = {r_a:+.2f} (p={p_a:.3g})")

    hb = summary["measures"]["hbo_pfc"]
    print("\nREADING")
    if hb["n_sig_reversed"] >= 2 and hb["n_sig_as_expected"] >= 2:
        print("  Subjects split into significant opposite directions on prefrontal HbO: consistent with an "
              "inverted-U / capacity ceiling in a subset. Cross-subject pretraining may average the 2/3 "
              "direction toward zero -- check step 8's 2/3 cells specifically.")
    elif hb["I2"] >= 0.5 and hb["Q_p"] < 0.05:
        print("  Strong between-subject heterogeneity (I^2 >= 0.5) without many individually significant "
              "reversals: subjects differ more than noise explains, but most single subjects lack the "
              "power (9 vs 9 blocks) to show it alone.")
    elif hb["group_wilcoxon_p"] < 0.05:
        print("  A consistent group-level 3-vs-2 prefrontal effect with limited heterogeneity: no evidence "
              "that subjects follow opposite patterns; the 2/3 difficulty is not explained by overload.")
    else:
        print("  No reliable 3-vs-2 prefrontal HbO effect at group or subject level: the hemodynamic "
              "response barely distinguishes 2-back from 3-back in this data (a ceiling, not a reversal).")

    with open(os.path.join(out_dir, "capacity_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"per_subject": {m: res[m] for m in MEASURES}, "hbo_profile_0_2_3": prof,
                   "decoding_2v3": dec, **summary}, f, indent=2, default=float)
    with open(os.path.join(out_dir, "capacity_per_subject.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject"] + [f"{m}_{k}" for m in MEASURES for k in ("delta", "d", "p", "q", "sessions")]
                   + ["decoding_2v3"])
        for s in ids:
            w.writerow([s] + [res[m][s][k] for m in MEASURES for k in ("delta", "d", "p", "q", "sessions")]
                       + [dec.get(s, "")])
    figures(out_dir, ids, res, prof, dec)
    print(f"\n[written] {out_dir}: capacity_summary.json, capacity_per_subject.csv, "
          f"capacity_effects.png, capacity_profiles.png" + (", capacity_vs_decoding.png" if dec else ""))


if __name__ == "__main__":
    main()
