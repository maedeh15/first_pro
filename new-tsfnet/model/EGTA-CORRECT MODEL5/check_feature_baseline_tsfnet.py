"""
check_feature_baseline_tsfnet.py -- fast sanity check of a preprocessed
TSFNet folder, independent of the deep model. Adapted from
check_feature_baseline.py (MSTFDN) to this project's actual saved shapes.

For every subject it runs the SAME cross-session LOSO split as the trainer
(train on 2 sessions, test on the 3rd, 3 classes 0/2/3-back) with a plain
logistic regression on hand-made features:
  EEG   : log-variance per channel over time.
          Uses {sid}_eeg_channel_windows.npy (N, 28, T) if that file exists
          (ENABLE_EEG_CHANNEL_SPACE=True) -> 28 features; otherwise falls
          back to {sid}_eeg_windows.npy (N, 16, 16, T), the spatially-mapped
          grid -> 256 features (one per grid cell, not a real electrode).
  fNIRS : mean and linear slope over time, per grid cell and per chromophore,
          AVERAGED FIRST over the S temporal segments (collapses the 21-
          segment redundancy before building features, so the feature count
          stays in the hundreds, not the thousands) -> 16*16*C*2 features
          (C=2 or 3 chromophores depending on ENABLE_FNIRS_HBT).
If these simple features are at chance, the information your new
preprocessing variant removed is not recoverable by ANY network -- a
negative result here rules out wasting a 4h training run on that variant.
A POSITIVE result here is necessary but not sufficient: it does not
guarantee TSFNet's actual accuracy improves, only that the raw signal
hasn't been destroyed.

Runs on CPU in a few minutes, no GPU needed.

usage:
  python check_feature_baseline_tsfnet.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback_normscopesegment"
  python check_feature_baseline_tsfnet.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" --subjects VP001 VP002
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def features(sub_dir, sid):
    y = np.load(os.path.join(sub_dir, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(sub_dir, f"{sid}_session_ids.npy"))

    # --- EEG: prefer real per-channel data (priority-G4) over the 16x16
    # spatial-interpolation grid, which smears real electrodes across
    # artificial grid cells and would understate how much signal survives.
    chan_path = os.path.join(sub_dir, f"{sid}_eeg_channel_windows.npy")
    if os.path.exists(chan_path):
        eeg = np.load(chan_path, mmap_mode="r")          # (N, 28, T)
        fe = np.log(np.asarray(eeg, dtype=np.float32).var(-1) + 1e-8)   # (N, 28)
    else:
        eeg = np.load(os.path.join(sub_dir, f"{sid}_eeg_windows.npy"), mmap_mode="r")   # (N,16,16,T)
        fe = np.log(np.asarray(eeg, dtype=np.float32).var(-1) + 1e-8).reshape(len(y), -1)  # (N, 256)

    # --- fNIRS: (N, S, 16, 16, T, C) -- average over segments S first (they
    # overlap heavily by construction, see FNIRS_NUM_SEGMENTS/step=1s in the
    # preprocessing script), THEN take mean/slope over time T per grid cell
    # and chromophore. Keeps feature count at 16*16*C*2, not S times that.
    fn = np.load(os.path.join(sub_dir, f"{sid}_fnirs_windows.npy"))   # (N, S, 16, 16, T, C)
    f = fn.mean(axis=1)                                               # (N, 16, 16, T, C)
    t = np.arange(f.shape[-2], dtype=np.float32)
    t -= t.mean()
    slope = (f * t[None, None, None, :, None]).sum(-2) / (t ** 2).sum()   # (N, 16, 16, C)
    mean_ = f.mean(-2)                                                    # (N, 16, 16, C)
    ff = np.concatenate([mean_.reshape(len(y), -1), slope.reshape(len(y), -1)], axis=1)

    return fe, ff, y, ses


def loso(X, y, ses):
    accs = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        if te.sum() == 0 or tr.sum() == 0:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs)) if accs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                     help="a TSFNet preprocessed output folder, e.g. "
                          "TSFNET_PAPER_EXACT_nback or ..._normscopesegment")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    rows = []
    print(f"{'subject':8} {'EEG':>7} {'fNIRS':>7} {'both':>7}")
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        if not os.path.exists(os.path.join(d, f"{sid}_labels.npy")):
            continue
        fe, ff, y, ses = features(d, sid)
        r = (loso(fe, y, ses), loso(ff, y, ses), loso(np.hstack([fe, ff]), y, ses))
        rows.append(r)
        print(f"{sid:8} {r[0]:7.3f} {r[1]:7.3f} {r[2]:7.3f}", flush=True)
    if not rows:
        print("No subjects found under --data (check the path).")
        return
    m = np.nanmean(rows, axis=0)
    print(f"{'MEAN':8} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f}   (chance = 0.333)")


if __name__ == "__main__":
    main()
