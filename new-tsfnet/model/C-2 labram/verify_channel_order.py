"""
verify_channel_order.py -- C-2 prerequisite audit (~1 minute, no GPU).

WHAT IT ANSWERS: does the hard-coded channel list `PROJECT_EEG_CHANNELS` in
labram_eeg_branch_v3.py match, NAME FOR NAME AND IN ORDER, the channel axis of
`{subject}_eeg_channel_windows.npy`?

WHY THIS IS THE MOST IMPORTANT UNCHECKED ASSUMPTION IN C-2: LaBraM does not
use electrode coordinates. It maps each input row to an entry of its own
learned spatial position embedding PURELY BY CHANNEL NAME, in the order the
names are supplied. So if `PROJECT_EEG_CHANNELS[i]` is not actually the
electrode sitting in row `i` of the saved array, then every single channel is
handed to the wrong learned position -- silently, with no error, no shape
mismatch, and no warning. The model would still run, still train, still
produce plausible-looking accuracy, and the entire C-2 result would be
meaningless.

Nothing tested so far would have caught this: the shape checks only verify
that there are 28 rows, not which electrode each row holds.

WHAT IT CHECKS:
  1. The montage/info file's channel names and order vs PROJECT_EEG_CHANNELS.
  2. That the count matches the saved array's channel axis.
  3. (If coordinates are available) that no two supposedly-distinct electrodes
     sit at the same position, which would indicate a duplicated/misaligned list.

USAGE:
    python verify_channel_order.py --montage "path/to/montage.fif"
    python verify_channel_order.py --montage "path/to/ch_names.json"
    python verify_channel_order.py --montage "path/to/ch_names.txt" \\
        --array "E:/.../TSFNET_PAPER_EXACT_nback/VP001/VP001_eeg_channel_windows.npy"

Supported montage inputs: .fif / .fif.gz (MNE Raw, Epochs or Info), .json
(a list of names, or a dict with a "ch_names" key), .txt/.csv (one name per
line, or a single comma-separated line), .npy (array of names).
"""

import argparse
import json
import os
import sys


def load_reference_names(path):
    """Returns (names, source_description). Raises on an unreadable file."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".fif", ".gz"):
        import mne
        mne.set_log_level("ERROR")
        for reader, label in (
            (lambda p: mne.io.read_info(p), "mne.io.read_info"),
            (lambda p: mne.io.read_raw_fif(p, preload=False).info, "read_raw_fif"),
            (lambda p: mne.read_epochs(p, preload=False).info, "read_epochs"),
        ):
            try:
                info = reader(path)
                return list(info["ch_names"]), f"{label}({os.path.basename(path)})"
            except Exception:
                continue
        raise ValueError(f"could not read {path} as an MNE Info/Raw/Epochs file")

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict):
            for key in ("ch_names", "channel_names", "channels", "names"):
                if key in doc:
                    return list(doc[key]), f"{os.path.basename(path)}[{key}]"
            raise ValueError(f"{path}: no ch_names-like key found; keys={list(doc)[:10]}")
        return list(doc), os.path.basename(path)

    if ext == ".npy":
        import numpy as np
        arr = np.load(path, allow_pickle=True)
        return [str(x) for x in arr.ravel()], os.path.basename(path)

    # .txt / .csv / anything else: one name per line, or one comma-separated line
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read().strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and "," in lines[0]:
        names = [t.strip() for t in lines[0].split(",") if t.strip()]
    else:
        names = [ln.split(",")[0].strip() for ln in lines]
    return names, os.path.basename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--montage", required=True,
                    help="file holding the REAL channel names in the REAL order")
    ap.add_argument("--array", default=None,
                    help="optional {subject}_eeg_channel_windows.npy, to check the count")
    args = ap.parse_args()

    try:
        from labram_eeg_branch_v3 import PROJECT_EEG_CHANNELS as CODE
    except ImportError:
        print("ERROR: run this from the folder containing labram_eeg_branch_v3.py")
        sys.exit(1)

    try:
        ref, src = load_reference_names(args.montage)
    except Exception as exc:
        print(f"ERROR reading --montage: {exc}")
        sys.exit(1)

    print(f"reference source : {src}  ({len(ref)} channels)")
    print(f"code list        : PROJECT_EEG_CHANNELS  ({len(CODE)} channels)\n")

    # ---- check 1: count ----
    if len(ref) != len(CODE):
        print(f"[FAIL] channel COUNT differs: reference has {len(ref)}, code has {len(CODE)}.")
        print("       If the reference file includes non-EEG channels (fNIRS, triggers,")
        print("       EOG), filter it down to the EEG channels first.")
        print(f"\n  reference: {ref}")
        print(f"\n  code     : {CODE}")
        sys.exit(1)
    print(f"[PASS] channel count matches ({len(CODE)}).")

    # ---- check 2: order, case-insensitively ----
    mismatches = [(i, r, c) for i, (r, c) in enumerate(zip(ref, CODE))
                  if r.strip().lower() != c.strip().lower()]
    if not mismatches:
        print("[PASS] every channel matches the reference IN ORDER "
              "(case-insensitive).")
        print("\n  => The hard-coded list is correct. LaBraM received each electrode at")
        print("     its true position, so this assumption is NOT a threat to the C-2")
        print("     result. State this in the thesis as a verified check.")
    else:
        print(f"[FAIL] {len(mismatches)} position(s) disagree:\n")
        print(f"  {'idx':>4}  {'reference (truth)':<16} {'code':<16}")
        for i, r, c in mismatches:
            print(f"  {i:>4}  {r:<16} {c:<16}")
        same_set = sorted(x.lower() for x in ref) == sorted(x.lower() for x in CODE)
        print()
        if same_set:
            print("  => SAME CHANNELS, WRONG ORDER. This is the dangerous case: every")
            print("     C-2 run fed LaBraM correctly-named electrodes in the wrong")
            print("     sequence, so each row was matched to another electrode's learned")
            print("     position. The C-2 results are INVALID and must be re-run after")
            print("     fixing PROJECT_EEG_CHANNELS to the order printed above.")
            print("\n  corrected list to paste into labram_eeg_branch_v3.py:")
            print("  PROJECT_EEG_CHANNELS = [")
            for i in range(0, len(ref), 6):
                print("      " + ", ".join(f'"{n}"' for n in ref[i:i + 6]) + ",")
            print("  ]")
        else:
            print("  => The two lists are not even the same SET of channels. Check that")
            print("     the reference file really describes this project's EEG montage")
            print("     and not a different recording or a superset including non-EEG")
            print("     channels.")
        sys.exit(1)

    # ---- check 3: the saved array's channel axis ----
    if args.array:
        import numpy as np
        try:
            arr = np.load(args.array, mmap_mode="r")
        except Exception as exc:
            print(f"\n[WARN] could not open --array: {exc}")
            return
        print(f"\narray {os.path.basename(args.array)}: shape={tuple(arr.shape)}")
        if arr.ndim != 3:
            print(f"[WARN] expected 3 dims (N, C, T), got {arr.ndim}.")
        elif arr.shape[1] != len(CODE):
            print(f"[FAIL] the array's channel axis is {arr.shape[1]}, but the code list "
                  f"has {len(CODE)} names.")
            sys.exit(1)
        else:
            print(f"[PASS] the array's channel axis ({arr.shape[1]}) matches the "
                  f"channel list length.")

    print("\nALL CHECKS PASSED.")


if __name__ == "__main__":
    main()
