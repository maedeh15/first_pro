"""
priority-G3: quantify the CROSS-SESSION DISTRIBUTION SHIFT in this project's
preprocessed EEG/fNIRS windows, with no model training of any kind.

WHY THIS EXISTS
---------------
Two interventions in this project improved leave-one-session-out performance
where five model-side interventions did not:
    E3  per-session input normalization      58.91% -> 60.09%
    E4  AdaBN test-time BN-stat adaptation   60.09% -> 61.20%
Both are DOMAIN-ADAPTATION interventions: they assume a measurable
distribution shift between sessions and try to remove it. But so far that
assumption is inferred backwards, from the fact that the interventions
happened to work. This script measures the shift DIRECTLY, so the claim
becomes "the shift was measured, then removed, and accuracy rose" rather
than "accuracy rose, so presumably there was a shift".

It mirrors the diagnostic Ding et al. 2026 (MSTFDN, Information Fusion)
report for this same MBI/Shin N-back dataset: Maximum Mean Discrepancy (MMD)
and Wasserstein/Earth Mover's Distance between the train and test session
distributions, plus a t-SNE view. Their finding -- that the N-back fNIRS
modality shows the largest cross-session shift of any task/modality they
tested -- is directly checkable against your own data here.

WHAT IT REPORTS
---------------
Per subject and modality (EEG, fNIRS), for every session PAIR:
    MMD^2 (unbiased, RBF kernel, median-heuristic bandwidth)
    mean 1-D Wasserstein distance across the retained PCA components
plus a grand summary across subjects, and optional t-SNE figures.

USAGE
-----
One directory (report the shift present in that dataset):
    python session_shift_diagnostic.py --data-dir /path/to/preprocessed

Two directories (the important one: BEFORE vs AFTER per-session
normalization, showing E3 actually reduced the measured shift):
    python session_shift_diagnostic.py \
        --data-dir /path/to/baseline_preprocessed \
        --compare-dir /path/to/E3_preprocessed

NOTE ON THE "BEFORE" DIRECTORY: running the preprocessing with
ENABLE_PER_SESSION_NORM=True overwrites the .npy files in place. If no
pre-E3 backup exists, run this on the current (E3) directory alone -- the
absolute shift levels and the EEG-vs-fNIRS comparison are still reportable
and still support the argument; only the before/after delta needs both.

DIMENSIONALITY: a single EEG window is 16x16x2000 = 512k values, so windows
are strided in time (and, for fNIRS, in segments) and then projected with a
PCA fitted on that subject's POOLED sessions. Pooling the PCA fit across
sessions is deliberate: a per-session PCA would define a different basis per
session and make the resulting distances incomparable, which would silently
manufacture or hide shift.
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Distance metrics
# ---------------------------------------------------------------------------
def _median_bandwidth(X, Y, rng, max_pairs=2000):
    """Median-heuristic RBF bandwidth: sigma^2 = median of squared pairwise
    distances over the pooled sample. Subsampled for tractability, with an
    explicit rng so the result is reproducible."""
    Z = np.vstack([X, Y])
    n = Z.shape[0]
    if n > max_pairs:
        idx = rng.choice(n, size=max_pairs, replace=False)
        Z = Z[idx]
    sq = _sq_dists(Z, Z)
    iu = np.triu_indices(sq.shape[0], k=1)
    med = np.median(sq[iu])
    # Guard the degenerate case of identical points (median 0 -> divide by 0).
    return med if med > 1e-12 else 1.0


def _sq_dists(A, B):
    """Pairwise squared Euclidean distances, (n_a, n_b)."""
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    d = a2 + b2 - 2.0 * (A @ B.T)
    return np.maximum(d, 0.0)   # clamp tiny negatives from float error


def mmd_rbf(X, Y, sigma_sq=None, rng=None, unbiased=True):
    """
    Squared Maximum Mean Discrepancy with an RBF kernel.

        MMD^2 = E[k(x,x')] + E[k(y,y')] - 2 E[k(x,y)]

    UNBIASED estimator (default): the within-sample terms EXCLUDE the
    diagonal k(x_i, x_i)=1. This matters -- the biased estimator includes
    those self-similarities and is therefore strictly positive even when X
    and Y are drawn from the identical distribution, which would make "no
    shift" indistinguishable from "small shift" in this report.

    X: (n_x, d), Y: (n_y, d). Returns a float (can be slightly negative for
    the unbiased estimator when the true MMD is 0 -- that is expected and is
    reported as-is rather than clipped, so the reader can see the noise floor.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    if sigma_sq is None:
        sigma_sq = _median_bandwidth(X, Y, rng)

    Kxx = np.exp(-_sq_dists(X, X) / (2.0 * sigma_sq))
    Kyy = np.exp(-_sq_dists(Y, Y) / (2.0 * sigma_sq))
    Kxy = np.exp(-_sq_dists(X, Y) / (2.0 * sigma_sq))

    n, m = X.shape[0], Y.shape[0]
    if unbiased:
        if n < 2 or m < 2:
            raise ValueError("unbiased MMD needs at least 2 samples per group")
        sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
        sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    else:
        sum_xx = Kxx.mean()
        sum_yy = Kyy.mean()
    return float(sum_xx + sum_yy - 2.0 * Kxy.mean())


def mean_wasserstein(X, Y):
    """
    Mean 1-D Wasserstein (Earth Mover's) distance across feature dimensions.

    A true multivariate optimal-transport distance is expensive and, for the
    sample sizes here, dominated by estimation noise. Averaging the exact 1-D
    W1 per PCA component is the standard tractable surrogate; because the
    components are the same basis for both groups (pooled PCA fit), the
    comparison stays meaningful. Reported as a marginal-wise average, not
    claimed to be the multivariate W1.
    """
    return float(np.mean([wasserstein_distance(X[:, j], Y[:, j])
                          for j in range(X.shape[1])]))


# ---------------------------------------------------------------------------
# Data loading / reduction
# ---------------------------------------------------------------------------
def _reduce(arr, n_components, rng_seed=42):
    """Flatten to (N, D) and PCA-project to n_components, fitted on the POOLED
    input (all sessions together) -- see module docstring for why."""
    N = arr.shape[0]
    flat = arr.reshape(N, -1).astype(np.float64)
    k = min(n_components, N - 1, flat.shape[1])
    pca = PCA(n_components=k, random_state=rng_seed)
    return pca.fit_transform(flat), pca.explained_variance_ratio_.sum()


def _resolve_subject_paths(data_dir, subject_id):
    """
    Locate a subject's three .npy files, supporting BOTH on-disk layouts this
    project produces:

      (a) per-subject subdirectory (what Preproc_tsfnet_nback_badch.py
          actually writes, via sub_out_dir):
              <data_dir>/VP001/VP001_eeg_windows.npy
      (b) flat, all subjects in one directory:
              <data_dir>/VP001_eeg_windows.npy

    Returns (eeg_path, fnirs_path, sessions_path) or None if neither layout
    has all three files. Layout (a) is checked first because it is the
    pipeline's default; falling back to (b) keeps the script usable if the
    files are ever collected into one folder.
    """
    names = (f"{subject_id}_eeg_windows.npy",
             f"{subject_id}_fnirs_windows.npy",
             f"{subject_id}_session_ids.npy")
    for base in (os.path.join(data_dir, subject_id), data_dir):
        paths = tuple(os.path.join(base, n) for n in names)
        if all(os.path.exists(p) for p in paths):
            return paths
    return None


def load_subject(data_dir, subject_id, eeg_time_stride=20,
                 fnirs_seg_stride=3, fnirs_time_stride=5, n_components=50):
    """Load one subject's windows and return PCA-reduced EEG/fNIRS features
    plus session ids. Returns None if the subject's files are absent."""
    resolved = _resolve_subject_paths(data_dir, subject_id)
    if resolved is None:
        return None
    eeg_p, fn_p, ses_p = resolved

    eeg = np.load(eeg_p, mmap_mode="r")      # (N, 16, 16, T)
    fnirs = np.load(fn_p, mmap_mode="r")     # (N, S, 16, 16, T, 2)
    sessions = np.load(ses_p)

    eeg_s = np.asarray(eeg[:, :, :, ::eeg_time_stride])
    fnirs_s = np.asarray(fnirs[:, ::fnirs_seg_stride, :, :, ::fnirs_time_stride, :])

    eeg_feat, eeg_evr = _reduce(eeg_s, n_components)
    fn_feat, fn_evr = _reduce(fnirs_s, n_components)
    return {"eeg": eeg_feat, "fnirs": fn_feat, "sessions": sessions,
            "eeg_evr": eeg_evr, "fnirs_evr": fn_evr}


def analyze_subject(feats, rng):
    """Pairwise session MMD/EMD for both modalities of one subject."""
    out = {}
    sessions = feats["sessions"]
    uniq = sorted(np.unique(sessions))
    for mod in ("eeg", "fnirs"):
        X = feats[mod]
        pairs = {}
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = uniq[i], uniq[j]
                Xa, Xb = X[sessions == a], X[sessions == b]
                if len(Xa) < 2 or len(Xb) < 2:
                    continue
                pairs[f"{a}->{b}"] = {
                    "mmd": mmd_rbf(Xa, Xb, rng=rng),
                    "emd": mean_wasserstein(Xa, Xb),
                }
        out[mod] = pairs
    return out


def summarize(all_results):
    """Grand mean/std across subjects, per modality."""
    summary = {}
    for mod in ("eeg", "fnirs"):
        mmds, emds = [], []
        for subj in all_results.values():
            for pair in subj[mod].values():
                mmds.append(pair["mmd"])
                emds.append(pair["emd"])
        if mmds:
            summary[mod] = {
                "mmd_mean": float(np.mean(mmds)), "mmd_std": float(np.std(mmds)),
                "emd_mean": float(np.mean(emds)), "emd_std": float(np.std(emds)),
                "n_pairs": len(mmds),
            }
    return summary


def make_tsne_figure(feats, subject_id, out_path, seed=42):
    """t-SNE of the two modalities coloured by session (MSTFDN Fig. 5b style)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    sessions = feats["sessions"]
    for ax, mod in zip(axes, ("eeg", "fnirs")):
        X = feats[mod]
        perp = min(30, max(5, (len(X) - 1) // 3))
        emb = TSNE(n_components=2, random_state=seed, perplexity=perp,
                   init="pca").fit_transform(X)
        for s in sorted(np.unique(sessions)):
            m = sessions == s
            ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, label=f"session {s}")
        ax.set_title(f"{subject_id} — {mod.upper()}")
        ax.legend(fontsize=8)
    fig.suptitle("Cross-session distribution (t-SNE)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(data_dir, subjects, rng_seed=42, tsne_subject=None, out_dir=None):
    rng = np.random.default_rng(rng_seed)
    results = {}
    for sid in subjects:
        feats = load_subject(data_dir, sid)
        if feats is None:
            print(f"  {sid}: files not found, skipped")
            continue
        results[sid] = analyze_subject(feats, rng)
        e = results[sid]["eeg"]
        f = results[sid]["fnirs"]
        print(f"  {sid}: EEG mmd={np.mean([v['mmd'] for v in e.values()]):.5f} "
              f"emd={np.mean([v['emd'] for v in e.values()]):.4f} | "
              f"fNIRS mmd={np.mean([v['mmd'] for v in f.values()]):.5f} "
              f"emd={np.mean([v['emd'] for v in f.values()]):.4f}")
        if tsne_subject == sid and out_dir:
            make_tsne_figure(feats, sid, os.path.join(out_dir, f"{sid}_tsne_sessions.png"))
    return results, summarize(results)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--compare-dir", default=None,
                    help="Optional second directory (e.g. the pre-E3 baseline) to "
                         "contrast against --data-dir.")
    ap.add_argument("--subjects", default=None,
                    help="Comma-separated ids, e.g. VP001,VP002. Default VP001..VP026.")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--tsne-subject", default=None)
    args = ap.parse_args()

    subjects = ([s.strip() for s in args.subjects.split(",")] if args.subjects
                else [f"VP{i:03d}" for i in range(1, 27)])
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"=== {args.data_dir} ===")
    res_a, sum_a = run(args.data_dir, subjects, tsne_subject=args.tsne_subject,
                       out_dir=args.out_dir)
    payload = {"data_dir": args.data_dir, "summary": sum_a, "per_subject": res_a}

    print("\nSUMMARY (mean over all subjects x session pairs)")
    for mod, s in sum_a.items():
        print(f"  {mod:6s} MMD={s['mmd_mean']:.5f} (+-{s['mmd_std']:.5f})  "
              f"EMD={s['emd_mean']:.4f} (+-{s['emd_std']:.4f})  n={s['n_pairs']}")

    if args.compare_dir:
        print(f"\n=== {args.compare_dir} ===")
        res_b, sum_b = run(args.compare_dir, subjects)
        payload.update({"compare_dir": args.compare_dir, "compare_summary": sum_b,
                        "compare_per_subject": res_b})
        print("\nSUMMARY (compare dir)")
        for mod, s in sum_b.items():
            print(f"  {mod:6s} MMD={s['mmd_mean']:.5f}  EMD={s['emd_mean']:.4f}")
        print("\nCHANGE (data-dir relative to compare-dir); negative = LESS shift")
        for mod in sum_a:
            if mod in sum_b:
                dm = sum_a[mod]["mmd_mean"] - sum_b[mod]["mmd_mean"]
                de = sum_a[mod]["emd_mean"] - sum_b[mod]["emd_mean"]
                rm = 100.0 * dm / sum_b[mod]["mmd_mean"] if sum_b[mod]["mmd_mean"] else float("nan")
                print(f"  {mod:6s} dMMD={dm:+.5f} ({rm:+.1f}%)  dEMD={de:+.4f}")

    out_json = os.path.join(args.out_dir, "session_shift_diagnostic.json")
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote {out_json}")


if __name__ == "__main__":
    main()
