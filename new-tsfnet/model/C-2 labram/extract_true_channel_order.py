"""
extract_true_channel_order.py -- produces the GROUND-TRUTH channel order for
`_eeg_channel_windows.npy`, read straight from the raw dataset.

WHY A DEDICATED SCRIPT: the channel order of the saved channel-space array is
not written down anywhere as a list. It is the end of a three-step chain
inside Preproc_tsfnet_nback_badch.py:

  1. `load_eeg_data()` reads `cnt_nback.mat` and takes
         ch_names = [str(ch).strip() for ch in cnt_struct.clab]
     -- 30 entries: 28 scalp electrodes plus HEOG and VEOG.
  2. The same function marks HEOG/VEOG as EOG-type channels:
         raw.set_channel_types({ch: "eog" for ch in ["VEOG", "HEOG"]})
  3. `extract_windows()` then pulls the data with
         eeg_full = eeg_raw_filtered.get_data(picks="eeg")
     which returns ONLY the EEG-type channels, in `raw.ch_names` order --
     i.e. `clab` order with the two EOG entries dropped.

So the truth is simply: clab, minus VEOG/HEOG, order preserved. This script
reproduces exactly that, with no other preprocessing, then (optionally) hands
the result to verify_channel_order.py.

WHY IT MATTERS: LaBraM does not use electrode coordinates. It maps each input
row to an entry of its learned spatial position embedding PURELY BY NAME, in
the order supplied. If `PROJECT_EEG_CHANNELS` is not this exact list in this
exact order, every channel is silently bound to the wrong learned position --
no error, no shape mismatch, and a completely meaningless C-2 result.

USAGE:
    python extract_true_channel_order.py --eeg-folder "E:/.../VP001/EEG"
    python extract_true_channel_order.py --eeg-folder "..." --save ch_order.txt
"""

import argparse
import os
import sys

import numpy as np
import scipy.io as sio


EOG_NAMES = ("VEOG", "HEOG")


def load_clab(eeg_folder):
    """Replicates Preproc_tsfnet_nback_badch.py's `_load_struct` + clab read."""
    mat_path = os.path.join(eeg_folder, "cnt_nback.mat")
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(
            f"cnt_nback.mat not found in {eeg_folder!r}. Point --eeg-folder at the "
            f"folder that directly contains cnt_nback.mat / mrk_nback.mat / "
            f"mnt_nback.mat for one subject.")
    mat = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)
    real_keys = [k for k in mat.keys() if not k.startswith("__")]
    cnt = mat[real_keys[0]]
    return [str(ch).strip() for ch in np.asarray(cnt.clab).ravel()]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eeg-folder", required=True,
                    help="folder containing cnt_nback.mat for ONE subject")
    ap.add_argument("--save", default=None,
                    help="optional .txt to write the result to (one name per line), "
                         "ready to feed verify_channel_order.py --montage")
    ap.add_argument("--array", default=None,
                    help="optional {subject}_eeg_channel_windows.npy, to cross-check "
                         "the channel count")
    args = ap.parse_args()

    try:
        clab = load_clab(args.eeg_folder)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"cnt.clab                    : {len(clab)} entries")
    print(f"  {clab}\n")

    eog_present = [c for c in clab if c.upper() in EOG_NAMES]
    true_order = [c for c in clab if c.upper() not in EOG_NAMES]

    print(f"EOG entries dropped         : {eog_present if eog_present else 'none found'}")
    print(f"TRUE EEG channel order      : {len(true_order)} channels")
    print(f"  {true_order}\n")

    if not eog_present:
        print("[WARN] Neither VEOG nor HEOG was found in clab. The preprocessing script")
        print("       expects both and marks them as EOG so that picks='eeg' drops them.")
        print("       If this subject genuinely has no EOG channels the list above is")
        print("       still correct, but confirm the count below matches the saved array.")

    # ---- cross-check against the saved array ----
    if args.array:
        try:
            arr = np.load(args.array, mmap_mode="r")
            print(f"saved array {os.path.basename(args.array)}: shape={tuple(arr.shape)}")
            if arr.ndim == 3 and arr.shape[1] == len(true_order):
                print(f"[PASS] the array's channel axis ({arr.shape[1]}) matches the "
                      f"{len(true_order)} channels derived above.\n")
            else:
                print(f"[FAIL] the array's channel axis does not match: array has "
                      f"{arr.shape[1] if arr.ndim == 3 else '?'} , derived list has "
                      f"{len(true_order)}.\n")
        except Exception as exc:
            print(f"[WARN] could not read --array: {exc}\n")

    # ---- compare against the hard-coded list, if importable ----
    try:
        from labram_eeg_branch_v3 import PROJECT_EEG_CHANNELS as CODE
    except ImportError:
        CODE = None
        print("(labram_eeg_branch_v3.py not importable from here -- skipping the "
              "direct comparison; run verify_channel_order.py from its folder instead.)")

    if CODE is not None:
        same_len = len(CODE) == len(true_order)
        same_order = same_len and all(
            a.strip().lower() == b.strip().lower() for a, b in zip(true_order, CODE))
        print(f"PROJECT_EEG_CHANNELS        : {len(CODE)} channels")
        if same_order:
            print("\n[PASS] PROJECT_EEG_CHANNELS matches the true order exactly "
                  "(case-insensitive).")
            print("  => The C-2 channel binding is correct. Record this as a verified")
            print("     check in the thesis.")
        else:
            print("\n[FAIL] PROJECT_EEG_CHANNELS does NOT match the true order.")
            if same_len and sorted(c.lower() for c in CODE) == sorted(
                    c.lower() for c in true_order):
                print("  => Same channels, WRONG ORDER -- every electrode was bound to")
                print("     another electrode's learned position. All C-2 results are")
                print("     invalid until this is fixed and the runs repeated.")
            else:
                print("  => The two lists differ in content, not just order.")
            print("\n  corrected list to paste into labram_eeg_branch_v3.py:")
            print("  PROJECT_EEG_CHANNELS = [")
            for i in range(0, len(true_order), 6):
                print("      " + ", ".join(f'"{n}"' for n in true_order[i:i + 6]) + ",")
            print("  ]")

    if args.save:
        with open(args.save, "w", encoding="utf-8") as fh:
            fh.write("\n".join(true_order) + "\n")
        print(f"\nwritten to: {args.save}")
        print(f"  next: python verify_channel_order.py --montage \"{args.save}\"")


if __name__ == "__main__":
    main()
