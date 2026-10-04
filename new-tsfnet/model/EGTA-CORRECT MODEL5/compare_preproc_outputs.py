"""
compare_preproc_outputs.py -- verify that an alternative preprocessing run
changed ONLY what it was supposed to change.

priority-GRIDALIGN re-runs the full preprocessing with
`--fnirs-grid-bbox eeg`. ICA, PyPREP and AutoReject are all seeded
(random_state=42), so on the same machine with the same package versions the
EEG side should come out bit-identical. That is an expectation, not a
guarantee (a different MNE/scipy/BLAS build can change floating-point
results), and if the EEG differs, the comparison against run60 is no longer a
single-variable ablation. This script checks it before any GPU time is spent.

For every subject it requires, between the baseline and the alternative root:
  * eeg_windows.npy, labels, session_ids, trial_ids (and eeg_channel_windows
    if present)  -> bit-identical
  * fnirs_windows.npy -> same shape (same windows kept), different content
and reports the fraction of fNIRS grid cells that are dead (always zero).

USAGE
    python compare_preproc_outputs.py ^
        --baseline-root "E:\\kntu\\Thesis\\codes\\Results\\TSFNET_PAPER_EXACT_nback" ^
        --new-root      "E:\\kntu\\Thesis\\codes\\Results\\TSFNET_PAPER_EXACT_nback_fnirsbboxEEG"

Exit code 0 = every subject passed; 1 = at least one failed (do not train).
"""

import argparse
import os
import sys

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--baseline-root", required=True)
p.add_argument("--new-root", required=True)
p.add_argument("--subjects", default="1-26")
p.add_argument("--chunk", type=int, default=16,
               help="windows compared per step (memory bound; the EEG files are large)")
args = p.parse_args()


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


def identical(a_path, b_path, chunk):
    """Bit-identical comparison of two .npy files without loading either fully."""
    a = np.load(a_path, mmap_mode="r")
    b = np.load(b_path, mmap_mode="r")
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
    if a.ndim == 0 or a.shape[0] <= chunk:
        ok = np.array_equal(np.asarray(a), np.asarray(b))
        return ok, "" if ok else "values differ"
    for i in range(0, a.shape[0], chunk):
        x, y = np.asarray(a[i:i + chunk]), np.asarray(b[i:i + chunk])
        if not np.array_equal(x, y):
            diff = np.abs(x.astype(np.float64) - y.astype(np.float64)).max()
            return False, f"values differ from window {i} (max |diff| {diff:.3g})"
    return True, ""


def dead_fraction(path, chunk):
    """Share of 16x16 cells that are zero in every window/segment/time/chromophore."""
    f = np.load(path, mmap_mode="r")                    # (N, S, 16, 16, T, C)
    alive = np.zeros(f.shape[2:4], dtype=bool)
    for i in range(0, f.shape[0], chunk):
        x = np.asarray(f[i:i + chunk])
        alive |= (x != 0).any(axis=(0, 1, 4, 5))
    return 1.0 - alive.mean()


MUST_MATCH = ["eeg_windows", "labels", "session_ids", "trial_ids", "eeg_channel_windows"]

failed = []
print(f"baseline: {args.baseline_root}\nnew     : {args.new_root}\n")
for sid in parse_subjects(args.subjects):
    a_dir = os.path.join(args.baseline_root, sid)
    b_dir = os.path.join(args.new_root, sid)
    if not os.path.isdir(b_dir):
        print(f"[{sid}] MISSING in new root -- preprocessing not run for it?")
        failed.append(sid)
        continue
    problems = []
    for name in MUST_MATCH:
        fa = os.path.join(a_dir, f"{sid}_{name}.npy")
        fb = os.path.join(b_dir, f"{sid}_{name}.npy")
        if not os.path.exists(fa) and not os.path.exists(fb):
            continue                                     # optional file absent in both
        if os.path.exists(fa) != os.path.exists(fb):
            problems.append(f"{name}: present in only one root")
            continue
        ok, why = identical(fa, fb, args.chunk)
        if not ok:
            problems.append(f"{name}: {why}")

    fa = os.path.join(a_dir, f"{sid}_fnirs_windows.npy")
    fb = os.path.join(b_dir, f"{sid}_fnirs_windows.npy")
    sa = np.load(fa, mmap_mode="r").shape
    sb = np.load(fb, mmap_mode="r").shape
    if sa != sb:
        problems.append(f"fnirs_windows shape {sa} vs {sb}")
    same_fnirs, _ = identical(fa, fb, args.chunk) if sa == sb else (False, "")
    if same_fnirs:
        problems.append("fnirs_windows is IDENTICAL -- the bbox option did not take effect")

    dead_a, dead_b = dead_fraction(fa, args.chunk), dead_fraction(fb, args.chunk)
    status = "PASS" if not problems else "FAIL"
    print(f"[{sid}] {status} | fNIRS dead cells {100*dead_a:.1f}% -> {100*dead_b:.1f}%"
          + ("" if not problems else "\n        " + "\n        ".join(problems)))
    if problems:
        failed.append(sid)

print()
if failed:
    print(f"VERDICT: FAIL for {len(failed)} subject(s): {', '.join(failed)}")
    print("Do NOT train on the new root: the EEG (or the window set) is not identical to the")
    print("baseline, so a run on it would change more than the fNIRS grid. Likely cause: a")
    print("different package version than the one that produced the baseline data.")
    sys.exit(1)
print("VERDICT: PASS -- EEG, labels and window set are bit-identical; only the fNIRS grid changed.")
