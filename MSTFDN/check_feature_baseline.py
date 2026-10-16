"""
check_feature_baseline.py -- fast sanity check of a preprocessed MSTFDN
folder, independent of the deep model.

For every subject it runs the SAME cross-session LOSO split as the trainer
(train on 2 sessions, test on the 3rd, 3 classes 0/2/3-back) with a plain
logistic regression on hand-made features:
  EEG   : log-variance of each (band, channel) in the 10 s window  -> 140 features
  fNIRS : mean and linear slope of each (chromophore, channel)      -> 216 features
If these simple features are at chance, the information is not in the data
as preprocessed (e.g. removed by the normalization) and no network can
recover it. Runs on CPU in a few minutes.

usage:
  python check_feature_baseline.py --data E:\\...\\MSTFDN_PREPROCESSED_NBACK_V3
"""
import os
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def features(sub_dir, sid):
    eeg = np.load(os.path.join(sub_dir, f"{sid}_eeg_windows.npy"), mmap_mode="r")
    fn = np.load(os.path.join(sub_dir, f"{sid}_fnirs_windows.npy"))
    y = np.load(os.path.join(sub_dir, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(sub_dir, f"{sid}_session_ids.npy"))
    keep = np.where(y != 3)[0]                      # drop rest -> 3 classes
    fe = np.stack([np.log(np.asarray(eeg[i], dtype=np.float32).var(-1) + 1e-8).ravel()
                   for i in keep])
    f = fn[keep]                                    # (N, 3, 36, 100)
    t = np.arange(f.shape[-1], dtype=np.float32); t -= t.mean()
    slope = (f * t).sum(-1) / (t ** 2).sum()
    ff = np.concatenate([f.mean(-1).reshape(len(keep), -1),
                         slope.reshape(len(keep), -1)], 1)
    return fe, ff, y[keep], ses[keep]


def loso(X, y, ses):
    accs = []
    for s in (1, 2, 3):
        tr, te = ses != s, ses == s
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        clf.fit(X[tr], y[tr])
        accs.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
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
    m = np.mean(rows, 0)
    print(f"{'MEAN':8} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f}   (chance = 0.333)")


if __name__ == "__main__":
    main()
