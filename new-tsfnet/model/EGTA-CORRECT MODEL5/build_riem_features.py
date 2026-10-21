"""
build_riem_features.py -- precompute the per-session tangent-space EEG features used by TSFNet's
riem_branch ('member'), once per subject, next to the other preprocessed .npy files.

For every subject folder <data>/VPxxx it reads {sid}_eeg_channel_windows.npy (N, 28, T) plus labels /
session_ids / trial_ids and writes
    {sid}_eeg_riem.npy        (N, D) float32, D = n_bands * C(C+1)/2  (4 bands x 406 = 1624 for C=28)
    {sid}_eeg_riem_meta.json  fingerprint of the window set (labels, trial_ids, session_ids), mode, bands, D

Features (identical to the best members of check_riemannian_eeg.py): covariance of the broadband signal
and of theta 4-8, alpha 8-13, beta 13-30 Hz; every SESSION is re-centred independently with
  EA        arithmetic-mean whitening  (Euclidean Alignment; default, simplest, equivalent in the probes), or
  riem_disp Frechet-mean re-centring + dispersion normalisation, or  riem  /  raw;
then the log map at the identity, isometrically vectorised.

TRANSDUCTIVE, like E3 / AdaBN: a session's centre is computed from that session's own UNLABELED windows only;
no label and no other session enters. Because every session is processed independently the file does not
depend on which session is the test session, so one file serves all three folds.

usage:
  python build_riem_features.py --self-test
  python build_riem_features.py --data "E:\\...\\TSFNET_PAPER_EXACT_nback" [--mode EA] [--overwrite]
"""
import os
import json
import hashlib
import argparse
import tempfile
import numpy as np

import check_riemannian_eeg as R


def fingerprint(labels, trial_ids, session_ids):
    h = hashlib.sha1()
    for a in (labels, trial_ids, session_ids):
        h.update(np.ascontiguousarray(a).astype(np.int64).tobytes())
    return h.hexdigest()


def riem_features(x, ses, mode="EA"):
    bands = R.build_covs(x, True)
    return np.hstack([R.domain_features(bands[b], ses, mode) for b in bands]).astype(np.float32)


def build_subject(sub_dir, sid, mode="EA", overwrite=False, verbose=True):
    out = os.path.join(sub_dir, f"{sid}_eeg_riem.npy")
    meta = os.path.join(sub_dir, f"{sid}_eeg_riem_meta.json")
    if os.path.exists(out) and os.path.exists(meta) and not overwrite:
        if verbose:
            print(f"[{sid}] exists -- skipped (use --overwrite)")
        return False
    x = np.asarray(np.load(os.path.join(sub_dir, f"{sid}_eeg_channel_windows.npy"), mmap_mode="r"), dtype=np.float64)
    y = np.load(os.path.join(sub_dir, f"{sid}_labels.npy"))
    ses = np.load(os.path.join(sub_dir, f"{sid}_session_ids.npy"))
    tid = np.load(os.path.join(sub_dir, f"{sid}_trial_ids.npy"))
    F = riem_features(x, ses, mode)
    assert F.shape[0] == len(y) and np.isfinite(F).all(), "feature matrix misaligned or non-finite"
    np.save(out, F)
    with open(meta, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint(y, tid, ses), "mode": mode, "dim": int(F.shape[1]),
                   "n_windows": int(F.shape[0]), "bands": ["broad"] + list(R.BANDS),
                   "channels": int(x.shape[1])}, f, indent=2)
    if verbose:
        print(f"[{sid}] wrote {F.shape} mode={mode}")
    return True


def self_test():
    rng = np.random.RandomState(0)
    ok = True
    with tempfile.TemporaryDirectory() as td:
        sid = "VP001"
        d = os.path.join(td, sid)
        os.makedirs(d)
        C, T, N = 6, 400, 36
        ses = np.repeat([1, 2, 3], N // 3)
        y = np.tile([0, 1, 2], N // 3)
        tid = np.repeat(np.arange(N // 3), 3)
        np.save(os.path.join(d, f"{sid}_eeg_channel_windows.npy"), rng.randn(N, C, T).astype(np.float32))
        for nm, a in (("labels", y), ("session_ids", ses), ("trial_ids", tid)):
            np.save(os.path.join(d, f"{sid}_{nm}.npy"), a)
        build_subject(d, sid, "EA", verbose=False)
        F = np.load(os.path.join(d, f"{sid}_eeg_riem.npy"))
        D = 4 * C * (C + 1) // 2
        good = F.shape == (N, D) and F.dtype == np.float32
        print(f"[shape] {F.shape} == ({N}, {D}) float32 -> {'OK' if good else 'FAIL'}")
        ok &= good
        meta = json.load(open(os.path.join(d, f"{sid}_eeg_riem_meta.json")))
        good = meta["fingerprint"] == fingerprint(y, tid, ses) and meta["dim"] == D
        print(f"[meta] fingerprint/dim -> {'OK' if good else 'FAIL'}")
        ok &= good
        # session independence: changing another session's data must not change this session's features
        x2 = np.load(os.path.join(d, f"{sid}_eeg_channel_windows.npy")).copy()
        x2[ses == 3] = rng.randn(int((ses == 3).sum()), C, T)
        F2 = riem_features(x2.astype(np.float64), ses, "EA")
        good = np.allclose(F[ses != 3], F2[ses != 3], atol=1e-4)
        print(f"[independence] sessions 1-2 unchanged when session 3 changes -> {'OK' if good else 'FAIL'}")
        ok &= good
        # skip-existing
        good = build_subject(d, sid, "EA", verbose=False) is False
        print(f"[skip] existing file not rebuilt without --overwrite -> {'OK' if good else 'FAIL'}")
        ok &= good
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--mode", default="EA", choices=["EA", "riem_disp", "riem", "raw"])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subjects", nargs="+", default=[f"VP{i:03d}" for i in range(1, 27)])
    a = ap.parse_args()
    if a.self_test:
        raise SystemExit(self_test())
    if not a.data:
        ap.error("--data is required unless --self-test")
    n = 0
    for sid in a.subjects:
        d = os.path.join(a.data, sid)
        if not os.path.exists(os.path.join(d, f"{sid}_eeg_channel_windows.npy")):
            print(f"[{sid}] no eeg_channel_windows.npy -- skipped")
            continue
        n += int(build_subject(d, sid, a.mode, a.overwrite))
    print(f"done: {n} subject(s) written")


if __name__ == "__main__":
    main()
