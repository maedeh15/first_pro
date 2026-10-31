"""
test_xai_groundtruth.py -- the two ground-truth devices for XAI (priority-PLANT, priority-SHUFFLE). CPU, synthetic.

  1. parse_plant_spec: valid specs, defaults, and refusals;
  2. plant_fnirs: changes ONLY the chosen segment, ONLY inside the patch, scaled -1/0/+1 by class, zero for 2-back, HbR opposite to HbO,
     and leaves the array untouched when nothing is planted (env var unset => NBackDataset unchanged);
  3. NBackDataset reads TSFNET_PLANT (and ignores it when unset), on a tiny synthetic subject folder;
  4. block_shuffled_labels: class counts preserved, every window of a block shares one label, most blocks change, windows outside idx untouched.
"""
import os, tempfile
import numpy as np
import torch

import tsfnet_model_notebook as M


def main():
    # 1
    d = M.parse_plant_spec("seg=3,strength=0.5")
    assert d == dict(seg=3, strength=0.5, row=6, col=6, size=4) and M.parse_plant_spec("") is None
    for bad in ("strength=1", "seg=2,bogus=1"):
        try:
            M.parse_plant_spec(bad); raise SystemExit(f"{bad!r} was NOT refused")
        except ValueError:
            pass
    print("[1] plant spec parsed; missing seg and unknown keys refused -> OK")
    # 2
    rng = np.random.RandomState(0)
    N, S, H, W, T, C = 30, 5, 16, 16, 30, 2
    labels = np.tile([0, 1, 2], N // 3)
    f0 = rng.randn(N, S, H, W, T, C).astype(np.float32)
    f1 = M.plant_fnirs(f0.copy(), labels, seg=2, strength=1.0)
    diff = f1 - f0
    assert np.abs(diff[:, [0, 1, 3, 4]]).max() == 0, "other segments must be untouched"
    outside = diff[:, 2].copy(); outside[:, 6:10, 6:10] = 0
    assert np.abs(outside).max() == 0, "only the patch may change"
    assert np.abs(diff[labels == 1]).max() == 0, "2-back must get no planted response"
    p0, p3 = diff[labels == 0][:, 2, 6:10, 6:10], diff[labels == 2][:, 2, 6:10, 6:10]
    assert np.allclose(p0, -p3[:len(p0)] if len(p0) == len(p3) else -p3, atol=1e-6) or np.allclose(p0.mean(0), -p3.mean(0), atol=1e-6)
    assert np.abs(diff[labels == 2][:, 2, 6:10, 6:10, :, 1]).max() > 0
    hbo, hbr = diff[labels == 2][0, 2, 6, 6, :, 0], diff[labels == 2][0, 2, 6, 6, :, 1]
    assert np.allclose(hbr, -0.5 * hbo, atol=1e-6), "HbR must be -0.5 x HbO"
    try:
        M.plant_fnirs(f0.copy(), labels, seg=9, strength=1.0); raise SystemExit("out-of-range segment was NOT refused")
    except ValueError:
        pass
    print("[2] planting touches one segment, inside the patch, -1/0/+1 by class, HbR = -0.5 HbO -> OK")
    # 3
    sub = tempfile.mkdtemp(); sid = "VP999"; n = 18
    np.save(os.path.join(sub, f"{sid}_eeg_windows.npy"), rng.randn(n, 16, 16, 600).astype(np.float32))
    np.save(os.path.join(sub, f"{sid}_fnirs_windows.npy"), f0[:n, :, :, :, :, :].astype(np.float32))
    np.save(os.path.join(sub, f"{sid}_labels.npy"), labels[:n]); np.save(os.path.join(sub, f"{sid}_session_ids.npy"), np.repeat([1, 2, 3], 6))
    np.save(os.path.join(sub, f"{sid}_trial_ids.npy"), np.arange(n))
    os.environ.pop("TSFNET_PLANT", None)
    a = M.NBackDataset(sub, sid).fnirs.numpy()
    assert np.array_equal(a, f0[:n]), "with TSFNET_PLANT unset the data must be bit-identical"
    os.environ["TSFNET_PLANT"] = "seg=1,strength=1.0"
    b = M.NBackDataset(sub, sid).fnirs.numpy()
    os.environ.pop("TSFNET_PLANT", None)
    assert not np.array_equal(a, b) and np.abs((b - a)[:, [0, 2, 3, 4]]).max() == 0
    print("[3] NBackDataset: unset env var => bit-identical data; set => only the chosen segment changes -> OK")
    # 4
    nb, nw = 12, 8
    lab = np.repeat(np.tile([0, 1, 2], nb // 3), nw).astype(np.int64)
    ses = np.repeat([1, 2], nb * nw // 2); tid = np.repeat(np.arange(nb) % (nb // 2), nw)
    idx = np.flatnonzero(ses == 1)
    new = M.block_shuffled_labels(lab, ses, tid, idx, seed=3)
    assert np.bincount(new, minlength=3).tolist() == np.bincount(lab[idx], minlength=3).tolist(), "class counts must be preserved"
    for t in np.unique(tid[idx]):
        assert len(np.unique(new[tid[idx] == t])) == 1, "a block must keep ONE label"
    assert (new != lab[idx]).mean() > 0.3, "most windows should change class"
    print("[4] block shuffle: class counts preserved, one label per block, labels actually changed -> OK")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
