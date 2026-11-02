"""
fnirs_psd_check.py -- spectrum of the fNIRS the model actually sees, for ALL subjects, and whether it differs between loads (CPU).
From {sid}_fnirs_windows.npy (N, S, 16, 16, 30, 2): the S segments are 3 s (30 samples @ 10 Hz) with a 1 s step, so overlapping samples are
identical and each window is rebuilt into one continuous trace of (S-1)*10+30 samples (= 23 s for S = 21), spatial mean over the active grid cells,
per chromophore. Welch PSD per window (one Hann segment, mean removed), averaged within subject x class.
Per subject and class: power fraction above 0.1 Hz, cardiac peak frequency (0.7-2.0 Hz), cardiac band (0.8-2.0 Hz) power fraction.
Group: mean +/- SD over subjects, Friedman across 0/2/3-back, pairwise Wilcoxon (exploratory, uncorrected, say so), figure + CSV.
LIMITS: windows are 23 s, so the frequency step is 0.043 Hz and a slow drift below ~0.05 Hz cannot be resolved here -- that needs the raw trace
(the preprocessing script); windows are baseline-subtracted per trial and from the interpolated 16x16 map, not the 36 raw channels; windows of one
trial overlap, so they are averaged inside subject and class before any test (the unit of inference is the subject).
  python fnirs_psd_check.py --data-root "...\\TSFNET_PAPER_EXACT_nback" --subjects 1-26 --out-dir "...\\fnirs_psd"
"""
import argparse, csv, os, sys
import numpy as np
from scipy.signal import welch
from scipy.stats import friedmanchisquare, wilcoxon

FS, STEP, SEG = 10.0, 10, 30
CLASSES = ("0-back", "2-back", "3-back")


def rebuild(win):
    """win (n, S, H, W, 30, C) -> (n, C, L): spatial mean of the continuous trace"""
    n, S, H, W, T, C = win.shape
    L = (S - 1) * STEP + T
    grid = np.abs(win[0]).sum(axis=(0, 3, 4)) > 0 if n else None            # cells that carry signal
    out = np.zeros((n, C, L), dtype=np.float64)
    for k in range(S):
        seg = win[:, k][:, grid] if grid is not None and grid.any() else win[:, k].reshape(n, -1, T, C)
        seg = seg.mean(axis=1) if seg.ndim == 4 else seg                      # (n, T, C)
        out[:, :, k * STEP:k * STEP + T] = np.transpose(seg, (0, 2, 1))
    return out


def psd_windows(tr):
    f, p = welch(tr, fs=FS, nperseg=tr.shape[-1], noverlap=0, window="hann", detrend="constant", nfft=2048, axis=-1)
    return f, p


def metrics(f, P):
    """P (..., F) -> high-frequency fraction, cardiac peak frequency, cardiac band fraction"""
    pos = f > 0
    tot = P[..., pos].sum(-1) + 1e-30
    high = P[..., f > 0.1].sum(-1) / tot
    cb = (f >= 0.8) & (f <= 2.0)
    card = P[..., cb].sum(-1) / tot
    pk = f[(f >= 0.7) & (f <= 2.0)][np.argmax(P[..., (f >= 0.7) & (f <= 2.0)], axis=-1)]
    return high, pk, card


def subject_summary(sd, sid, stride):
    win = np.load(os.path.join(sd, f"{sid}_fnirs_windows.npy"), mmap_mode="r"); y = np.load(os.path.join(sd, f"{sid}_labels.npy"))
    idx = np.arange(0, len(y), stride); out = {}
    P = {c: [] for c in range(3)}; f = None
    for s in range(0, len(idx), 32):
        ii = idx[s:s + 32]; tr = rebuild(np.asarray(win[ii], dtype=np.float32)); f, p = psd_windows(tr)
        for c in range(3):
            m = y[ii] == c
            if m.any(): P[c].append(p[m])
    return f, {c: np.concatenate(P[c]).mean(0) for c in range(3) if P[c]}      # (chromophore, F) per class


def group(per, f, out_dir=None):
    subs = sorted(per); rows = []
    print(f"\nfNIRS spectrum of the model input, {len(subs)} subjects (window = {(len(f) and 23)} s trace; frequency step 0.043 Hz)")
    stats = {}
    for ch, chn in enumerate(("HbO", "HbR")):
        for mi, mn in enumerate(("power above 0.1 Hz", "cardiac peak (Hz)", "cardiac band 0.8-2 Hz fraction")):
            M = np.array([[metrics(f, per[s][c][ch])[mi] for c in range(3)] for s in subs])     # (subjects, 3)
            stats[(chn, mn)] = M
            fr = (float(friedmanchisquare(M[:, 0], M[:, 1], M[:, 2]).pvalue) if (len(subs) >= 6 and np.ptp(M, axis=1).max() > 1e-12) else 1.0)
            pw = {n: (float(wilcoxon(M[:, a] - M[:, b]).pvalue) if np.any(M[:, a] != M[:, b]) else 1.0) for n, (a, b) in (("2v0", (1, 0)), ("3v0", (2, 0)), ("3v2", (2, 1)))}
            fmt = ".3f" if "peak" not in mn else ".3f"
            print(f"  {chn} {mn:32s} " + "  ".join(f"{CLASSES[c]} {M[:, c].mean():{fmt}}±{M[:, c].std(ddof=1):.3f}" for c in range(3)) + f" | Friedman p={fr:.3g} | pairwise p " + " ".join(f"{k} {v:.3g}" for k, v in pw.items()))
            for i, s in enumerate(subs): rows.append([s, chn, mn] + [float(v) for v in M[i]])
    allm = np.array([[metrics(f, per[s][c][0])[0] for c in range(3)] for s in subs]).mean(1)
    print(f"  HbO power above 0.1 Hz over subjects (mean of classes): mean {allm.mean():.3f}, min {allm.min():.3f} ({subs[int(allm.argmin())]}), max {allm.max():.3f} ({subs[int(allm.argmax())]})")
    print("  Pairwise p are exploratory and uncorrected. A class difference in cardiac frequency/power would mean the input carries a systemic signal that separates loads;"
          "\n  no difference means the model cannot read the load from heart rate in this representation.")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "fnirs_psd_per_subject.csv"), "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["subject", "chromophore", "metric"] + list(CLASSES)); w.writerows(rows)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
            for ch, ax in enumerate(axs):
                for c in range(3):
                    A = np.array([per[s][c][ch] / per[s][c][ch][f > 0].sum() for s in subs]); m, se = A.mean(0), A.std(0, ddof=1) / np.sqrt(len(subs))
                    ax.semilogy(f[(f > 0) & (f < 3)], m[(f > 0) & (f < 3)], label=CLASSES[c]); ax.fill_between(f[(f > 0) & (f < 3)], (m - se)[(f > 0) & (f < 3)], (m + se)[(f > 0) & (f < 3)], alpha=0.2)
                ax.axvline(0.1, color="k", ls=":"); ax.set_title(("HbO", "HbR")[ch]); ax.set_xlabel("Hz")
            axs[0].set_ylabel("relative PSD"); axs[0].legend(); fig.tight_layout(); fig.savefig(os.path.join(out_dir, "fnirs_psd_group.png"), dpi=150); plt.close(fig)
            print("figure + csv ->", out_dir)
        except Exception as e:
            print("figure skipped:", e)
    return stats


def self_test():
    rng = np.random.RandomState(0); S = 21; L = (S - 1) * STEP + SEG; t = np.arange(L) / FS
    tr = np.stack([np.sin(2 * np.pi * 1.2 * t + p) for p in rng.uniform(0, 6, 5)])[:, None, :] + 0.1 * rng.randn(5, 1, L)
    win = np.zeros((5, S, 4, 4, SEG, 2), dtype=np.float32)
    for k in range(S):
        win[:, k, :, :, :, 0] = tr[:, 0, k * STEP:k * STEP + SEG][:, None, None, :]; win[:, k, :, :, :, 1] = win[:, k, :, :, :, 0]
    rb = rebuild(win); assert np.allclose(rb[:, 0, :], tr[:, 0, :], atol=1e-5), "rebuild must reproduce the continuous trace"
    f, p = psd_windows(rb); hi, pk, cd = metrics(f, p)
    assert np.all(np.abs(pk - 1.2) < 0.08) and np.all(cd > 0.5), (pk, cd)
    print("[1] overlapping segments rebuild the continuous trace exactly; a 1.2 Hz test tone gives peak", round(float(pk.mean()), 3), "Hz and cardiac fraction", round(float(cd.mean()), 2))
    per = {f"VP{i:03d}": {c: p.mean(0) * (1 + 0.1 * c * (i % 2)) for c in range(3)} for i in range(8)}
    group(per, f); print("[2] group summary runs\nSELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("  python")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root"); ap.add_argument("--subjects", default="1-26"); ap.add_argument("--stride", type=int, default=3, help="use every k-th window (90%% overlap makes neighbours redundant)")
    ap.add_argument("--out-dir", default=None); ap.add_argument("--self-test", action="store_true"); a = ap.parse_args()
    if a.self_test: return self_test()
    if not a.data_root: sys.exit("need --data-root")
    lo, hi = (a.subjects.split("-") + [a.subjects])[:2] if "-" in a.subjects else (a.subjects, a.subjects); per = {}; f = None
    for i in range(int(lo), int(hi) + 1):
        sid = f"VP{i:03d}"
        try:
            f, per[sid] = subject_summary(os.path.join(a.data_root, sid), sid, a.stride); print(f"[{sid}] done", flush=True)
        except OSError as e:
            print(f"[{sid}] skipped: {e}")
    if len(per) < 6: sys.exit(f"{len(per)} subjects; group statistics need >= 6")
    group({s: v for s, v in per.items() if len(v) == 3}, f, a.out_dir)


if __name__ == "__main__":
    main()
