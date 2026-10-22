"""
diag_eeg_branch_gap.py -- why is the EEG branch's own classifier at 48.7% when a linear classifier
on the SAME EEG reaches 64.97%?

The 16-point gap has three candidate causes, and this script separates the one that can be settled
offline. All of them are properties of y_eeg, the EEG branch's own head:

  (a) y_eeg is AUXILIARY. tsfnet_loss uses l_pred + 0.2*l_eeg + 0.2*l_fnirs + l_fusion, so the EEG
      head carries weight 0.2 and the shared trunk is shaped mainly by what EFGF/CAFE/fusion need.
      Its accuracy is a deep-supervision read-out, not "what EEG can give". Needs a GPU run to test
      (raise lambda_eeg), so it is NOT tested here.
  (b) REPRESENTATION. The branch sees the 16x16 interpolated grid, not the 28 channels. The grid is
      a rank-28 linear map of them, so a linear read-out loses nothing in principle -- but the
      conditioning may differ. Tested here: the same covariance probe in channel space and in grid
      space.
  (c) READ-OUT ORDER. `pooled` is mean over time of the conv activations (GAP), and the final layer
      is linear on it. Band power -- frontal-midline theta up, parietal alpha down with load -- is
      a SECOND-order statistic. A mean over time destroys it unless a nonlinearity rectifies first.
      Tested here, and this is the point of the script.

What it computes, per subject, leave-one-session-out, same classifier as the other probes
(StandardScaler + LogisticRegression(C=0.1)):

  mean        mean over time per channel/band                 first order, no rectification
  rect        mean over time of |x| per channel/band          first order AFTER rectification: a
                                                              band-power proxy that conv+ReLU+GAP
                                                              CAN represent
  logvar      log variance over time per channel/band         second order, diagonal only
  cov         EA-aligned covariance, tangent space            second order, full (the 64.97% member)
  cov_grid    the same, on the saved 16x16 grid windows        (b): does the grid representation cost
                                                              anything a linear read-out can see?

Reading it: if `rect` lands near the branch's 48.7% while `cov` lands near 65%, then what the branch
is missing is the CROSS-CHANNEL second-order structure, which no amount of time-averaging recovers,
and (c) is the dominant cause. If `rect` is already near 65%, the branch's features are not the
problem and the cause is (a) or the architecture above the features.

NOTE: an `--eeg-temporal-mode gap_logvar` flag that adds ShallowConvNet-style log-variance beside
the GAP already exists in the model (priority-LOGVAR). Whatever this probe says, that flag's own run
is the GPU-side evidence; this script only says which cause to spend GPU time on.

usage:
  python diag_eeg_branch_gap.py --self-test
  python diag_eeg_branch_gap.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback"
  python diag_eeg_branch_gap.py --data "..." --device cuda        # same numbers, much faster
"""
import os
import argparse
import tempfile
import numpy as np
from scipy.stats import wilcoxon

import check_riemannian_eeg as R

FEATS = ("mean", "rect", "logvar", "cov", "cov_grid")


def band_signals(x):
    """-> {band: (N, C, T)} broadband plus theta/alpha/beta, the same filters as the cov member"""
    out = {"broad": np.asarray(x, dtype=np.float64)}
    for name, (lo, hi) in R.BANDS.items():
        out[name] = np.asarray(R.bandpass(out["broad"], lo, hi), dtype=np.float64)
    return out


def first_order(bands, rectify):
    """mean over time, per channel and band -> (N, C*n_bands). rectify=True takes |x| first."""
    cols = []
    for b in sorted(bands):
        s = np.abs(bands[b]) if rectify else bands[b]
        cols.append(s.mean(axis=-1))
    return np.hstack(cols)


def diag_logvar(bands):
    cols = [np.log(np.maximum(bands[b].var(axis=-1), 1e-12)) for b in sorted(bands)]
    return np.hstack(cols)


def cov_features(x, ses, multiband=True):
    b = R.build_covs(np.asarray(x, dtype=np.float64), multiband)
    return np.hstack([R.domain_features(b[k], ses, "EA") for k in b])


def grid_signals(grid_windows, rank=28):
    """{sid}_eeg_windows.npy is the 16x16 interpolated grid the model actually eats: (N, 16, 16, T).
    Flatten to (N, cells, T), drop cells that are zero in every window (outside the projected head
    they are structurally zero), then project the live cells onto their leading `rank` directions.

    The projection is not an approximation. The grid is W @ channels with W fixed by the electrode
    positions, so the cells span a subspace of dimension at most the channel count: a 160x160 cell
    covariance has rank ~28 and its 12880-element tangent vector holds only ~406 independent
    numbers. Computing it at full size makes the ridge, not the data, decide the whitening, and costs
    minutes per subject. The basis is the row space of W -- sensor geometry, not labels, and the same
    for every window and session -- so taking it from the pooled windows uses no label information.
    rank=None keeps every live cell (slow, ill-conditioned; kept only to check the claim above)."""
    g = np.asarray(grid_windows, dtype=np.float64)
    if g.ndim != 4:
        raise ValueError(f"expected (N, 16, 16, T) grid windows, got {g.shape}")
    N, H, W_, T = g.shape
    g = g.reshape(N, H * W_, T)
    keep = np.abs(g).sum(axis=(0, 2)) > 0
    g = g[:, keep, :]
    live = int(keep.sum())
    if rank is None or rank >= live:
        return g, live, live
    flat = g.transpose(1, 0, 2).reshape(live, -1)                 # (cells, N*T)
    S = flat @ flat.T / flat.shape[1]
    w, v = np.linalg.eigh(S)
    B = v[:, ::-1][:, :rank]                                      # (cells, rank)
    return np.einsum("cr,nct->nrt", B, g), live, rank


def features_for(name, x, ses, grid, grid_rank=28):
    if name in ("mean", "rect", "logvar"):
        b = band_signals(x)
        if name == "logvar":
            return diag_logvar(b)
        return first_order(b, rectify=(name == "rect"))
    if name == "cov":
        return cov_features(x, ses)
    if name == "cov_grid":
        if grid is None:
            raise ValueError("cov_grid needs the grid windows")
        g, _, _ = grid_signals(grid, rank=grid_rank)
        return cov_features(g, ses)
    raise ValueError(name)


def run_subject(x, y, ses, grid, which, grid_rank=28):
    out = {}
    for name in which:
        F = features_for(name, x, ses, grid, grid_rank)
        out[name] = (R.loso(F, y, ses), R.loso(F, y, ses, (1, 2)))
    return out


def self_test():
    """planted: the class lives ONLY in band power (variance), with the time-mean carrying nothing.
    A first-order mean must stay at chance, a rectified mean and the variance must find it. This is
    the exact contrast the script is built to measure, so if it cannot measure it on planted data
    its reading of the real data means nothing."""
    rng = np.random.RandomState(0)
    ok = True
    C, T, N = 8, 400, 300
    ses = np.repeat([1, 2, 3], N // 3)
    y = np.tile([0, 1, 2], N // 3)
    x = np.empty((N, C, T))
    for i in range(N):
        amp = 1.0 + 0.6 * y[i]                      # class = amplitude (power), not level
        x[i] = amp * rng.randn(C, T)
        x[i, :2] += 0.0                             # no mean shift anywhere
    b = band_signals(x)
    a_mean = R.loso(first_order(b, False), y, ses)
    a_rect = R.loso(first_order(b, True), y, ses)
    a_lv = R.loso(diag_logvar(b), y, ses)
    print(f"[planted power-only] mean {a_mean:.3f} | rect {a_rect:.3f} | logvar {a_lv:.3f}")
    good = a_mean < 0.45 and a_rect > 0.80 and a_lv > 0.80
    print(f"  time-mean blind to power, rectified mean and logvar find it -> {'OK' if good else 'FAIL'}")
    ok &= good

    # and the converse: a class that is a pure LEVEL shift must be found by the mean.
    # N matters here: the feature vector holds 4 bands x C channels and only one of them carries the
    # shift, so with few windows per fold the informative column is diluted by the regularisation
    # (measured: 0.63 at N=60, 0.74 at 120, 0.80 at 180, 0.93 at 300). Real subjects have on the
    # order of 1700 windows, far past that, but the per-subject window count is printed so a thin
    # subject can be recognised rather than read as "the feature is absent".
    x2 = rng.randn(N, C, T)
    for i in range(N):
        x2[i, 0] += 0.8 * y[i]
    a_mean2 = R.loso(first_order(band_signals(x2), False), y, ses)
    print(f"[planted level-only] mean {a_mean2:.3f}")
    good = a_mean2 > 0.80
    print(f"  time-mean finds a level shift -> {'OK' if good else 'FAIL'}")
    ok &= good
    # the grid claim: send channels through a fixed linear map into many cells; the rank-reduced
    # grid probe must recover what the channel-space probe finds, since the map is invertible on its
    # row space. If it does not, --grid-rank is losing something and cov_grid cannot be read.
    Cch, Ncells = 8, 120
    Wmap = rng.randn(Ncells, Cch)
    xs = np.empty((N, Cch, T))
    for i in range(N):
        xs[i] = (1.0 + 0.6 * y[i]) * rng.randn(Cch, T)
    cells = np.einsum("gc,nct->ngt", Wmap, xs)
    side = int(np.ceil(np.sqrt(Ncells)))
    grid = np.zeros((N, side, side, T))
    grid.reshape(N, side * side, T)[:, :Ncells, :] = cells
    a_ch = R.loso(cov_features(xs, ses), y, ses)
    gr, live, kept = grid_signals(grid, rank=Cch)
    a_gr = R.loso(cov_features(gr, ses), y, ses)
    print(f"[grid] channel-space {a_ch:.3f} | rank-{kept} grid {a_gr:.3f} (of {live} live cells)")
    good = kept == Cch and abs(a_ch - a_gr) < 0.05
    print(f"  rank-reduced grid probe matches the channel-space probe -> {'OK' if good else 'FAIL'}")
    ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--feats", nargs="+", default=list(FEATS), choices=list(FEATS))
    ap.add_argument("--grid-rank", type=int, default=28,
                    help="cov_grid: project the live grid cells onto this many directions. The grid "
                         "is a rank-<n_channels> map of the channels, so 28 loses nothing and keeps "
                         "the covariance well conditioned. 0 = keep every cell (slow).")
    ap.add_argument("--device", default=None,
                    help="run the eigendecompositions on this torch device (e.g. cuda). Almost all "
                         "the runtime is thousands of 28x28 eigh calls, which numpy does one at a "
                         "time; a GPU does the batch at once. Default: numpy.")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    a.grid_rank = a.grid_rank or None
    if not a.data:
        ap.error("--data is required unless --self-test")
    print(f"eigendecomposition backend: {R.set_eigh_backend(a.device)}")
    rows = []
    hdr = "  ".join(f"{f:>9}" for f in a.feats)
    print(f"{'subject':8} {'n':>6} {hdr}   | 2-vs-3: {hdr}")
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        f_eeg = os.path.join(d, f"{sid}_eeg_channel_windows.npy")
        if not os.path.exists(f_eeg):
            continue
        x = np.load(f_eeg, mmap_mode="r")
        y = np.load(os.path.join(d, f"{sid}_labels.npy"))
        ses = np.load(os.path.join(d, f"{sid}_session_ids.npy"))
        grid = None
        if "cov_grid" in a.feats:
            p = os.path.join(d, f"{sid}_eeg_windows.npy")
            if not os.path.exists(p):
                raise SystemExit(f"{sid}: no eeg_windows.npy, so cov_grid cannot be built. "
                                 f"Re-run with --feats mean rect logvar cov")
            grid = np.load(p, mmap_mode="r")
        r = run_subject(np.asarray(x, dtype=np.float64), y, ses, grid, a.feats, a.grid_rank)
        rows.append(r)
        print(f"{sid:8} [{len(y):4d}w] " + "  ".join(f"{r[f][0]:9.3f}" for f in a.feats) + "   | "
              + "  ".join(f"{r[f][1]:9.3f}" for f in a.feats), flush=True)
    if not rows:
        raise SystemExit("no subject found under --data")
    print("\nMEAN over subjects:")
    print(f"  {'feature':10} {'3-class':>8} {'2-vs-3':>8}")
    for f in a.feats:
        print(f"  {f:10} {np.mean([r[f][0] for r in rows]):8.4f} {np.mean([r[f][1] for r in rows]):8.4f}")
    print("\npaired differences vs 'rect' (the band-power proxy a conv+ReLU+GAP branch can represent):")
    for f in a.feats:
        if f == "rect" or "rect" not in a.feats:
            continue
        d = np.array([r[f][0] - r["rect"][0] for r in rows])
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
        print(f"  {f:10} {d.mean():+.4f}  wins {int((d > 0).sum())}/{len(d)}  p={p:.4f}")
    print("\nReading: the EEG branch's own head sits at 0.487. If 'rect' is near that and 'cov' is "
          "near 0.650, what the branch cannot represent is the CROSS-CHANNEL second-order structure, "
          "and raising lambda_eeg will not fix it. If 'rect' is already near 0.650, the features are "
          "fine and the cause is the auxiliary 0.2 weight or the layers above them. Linear probe, no "
          "learned temporal filters, no AdaBN: necessary evidence, never sufficient.")


if __name__ == "__main__":
    main()
