"""
inspect_fnirs_mnt.py -- does this dataset's fNIRS montage file contain SOURCE and
DETECTOR coordinates, or only one position per channel?

WHY THIS MATTERS
FGANet (Kwak et al., IEEE TNSRE 2022) builds the fNIRS image by filling the PATH
between each source and its detector with the channel's value before interpolating.
That needs two 3-D points per channel. The position table used by
Preproc_tsfnet_nback_badch.py (NIRS_POSITIONS_3D) has ONE point per channel (the
S-D midpoint). The path cannot be drawn from midpoints alone: the 30 mm segment's
ORIENTATION is unknown. So before implementing it, we must know whether the .mat
files carry the optode coordinates.

USAGE (pure Python + scipy, takes seconds, touches nothing):
    python inspect_fnirs_mnt.py "E:\\...\\BASE_DATASET_PATH" VP001
(BASE_DATASET_PATH is the same folder that contains VP001-NIRS, as in the preprocessing script.)
Send me the full printed output.
"""
import os, sys
import numpy as np
import scipy.io as sio

KEYS = ("source", "src", "detector", "det", "optode", "opt", "sd", "pair", "emitter", "pos", "coord", "loc", "xyz")

def load(path):
    mat = sio.loadmat(path, struct_as_record=False, squeeze_me=True)
    keys = [k for k in mat if not k.startswith("__")]
    return keys, mat[keys[0]]

def describe(obj, prefix="", depth=0, out=None):
    out = [] if out is None else out
    if depth > 3:
        return out
    if isinstance(obj, sio.matlab.mat_struct):
        for f in obj._fieldnames:
            describe(getattr(obj, f), f"{prefix}.{f}" if prefix else f, depth + 1, out)
        return out
    arr = None
    try:
        arr = np.asarray(obj)
    except Exception:
        pass
    if arr is None:
        out.append((prefix, "unreadable", "", False)); return out
    kind = f"shape={arr.shape} dtype={arr.dtype}"
    sample = ""
    if arr.dtype.kind in "fiu" and arr.size <= 6:
        sample = f"value={np.round(arr.astype(float), 4).tolist()}"
    elif arr.dtype.kind in "OUS" and arr.size <= 4:
        sample = f"value={arr.tolist()}"
    hit = any(k in prefix.lower() for k in KEYS)
    out.append((prefix, kind, sample, hit))
    return out

def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    base, subj = sys.argv[1], sys.argv[2]
    folder = os.path.join(base, f"{subj}-NIRS")
    found_geometry = False
    for fname in ("mnt_nback.mat", "cnt_nback.mat"):
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            print(f"[missing] {path}"); continue
        key, obj = load(path)
        print("=" * 78); print(f"{fname}  (top-level variable: {key[0]!r})"); print("=" * 78)
        for name, kind, sample, hit in describe(obj):
            if name.endswith(".x") and "cnt" in fname:      # the big signal arrays: skip noise
                continue
            print(f"  {'*' if hit else ' '} {name:<34} {kind:<34} {sample}")
            if hit and any(k in name.lower() for k in ("source", "src", "detector", "det", "optode", "emitter")):
                found_geometry = True
    print("\n" + "-" * 78)
    if found_geometry:
        print("RESULT: fields that look like source/detector/optode geometry EXIST (marked * above).")
        print("        Check their shapes: a path fill needs 16 source and 16 detector points (x,y,z), plus the")
        print("        36-row list saying which source pairs with which detector.")
    else:
        print("RESULT: NO field named like source/detector/optode was found.")
        print("        Then only channel positions are available and FGANet's path fill cannot be reproduced")
        print("        faithfully from these files (orientation of each 30 mm segment unknown).")

if __name__ == "__main__":
    main()
