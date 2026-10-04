"""
check_checkpoint_contamination.py -- run this in a SEPARATE terminal/Jupyter
kernel from whatever GPU run is currently in progress. Pure CPU, torch.load
only, takes a few seconds. Does NOT touch or interfere with a running job.

USAGE:
    python check_checkpoint_contamination.py "E:\\path\\to\\results_b5e27ee74d_seed42"

Or with no argument, edit RESULTS_DIR below and run directly / %run in Jupyter.
"""
import sys
import os
import glob
import torch

RESULTS_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\results_b5e27ee74d_seed42"


def check_one(pt_path):
    try:
        ckpt = torch.load(pt_path, map_location="cpu")
    except Exception as e:
        return {"path": pt_path, "error": f"could not load: {e}"}

    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        return {"path": pt_path, "error": "unrecognized checkpoint format"}

    has_band_branch = any("eeg_band_branch" in k for k in sd.keys())
    clf_eeg_shape = None
    for k in ("clf_eeg.net.0.weight",):
        if k in sd:
            clf_eeg_shape = tuple(sd[k].shape)
            break

    return {
        "path": pt_path,
        "contaminated_with_D1_safe": has_band_branch,
        "clf_eeg_in_features": clf_eeg_shape[1] if clf_eeg_shape else None,
        "n_tensors": len(sd),
    }


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    pt_files = sorted(glob.glob(os.path.join(results_dir, "*_best.pt")))

    if not pt_files:
        print(f"No .pt files found in: {results_dir}")
        print("Check the path is correct.")
        return

    print(f"Checking {len(pt_files)} checkpoint(s) in:\n  {results_dir}\n")
    n_clean, n_contaminated, n_error = 0, 0, 0

    for p in pt_files:
        r = check_one(p)
        name = os.path.basename(p)
        if "error" in r:
            print(f"  [ERROR] {name}: {r['error']}")
            n_error += 1
            continue
        if r["contaminated_with_D1_safe"]:
            print(f"  [CONTAMINATED] {name}  clf_eeg_in={r['clf_eeg_in_features']}  "
                  f"(expected 512 for a clean reference checkpoint)")
            n_contaminated += 1
        else:
            print(f"  [clean]         {name}  clf_eeg_in={r['clf_eeg_in_features']}")
            n_clean += 1

    print("\n" + "=" * 60)
    print(f"clean: {n_clean}   contaminated: {n_contaminated}   errors: {n_error}")
    if n_contaminated > 0:
        print("\n>>> These reference checkpoints were overwritten by a later D1-safe run.")
        print(">>> They cannot be used for an E3+E4+EGTA ensemble as-is.")
        print(">>> You will need to re-run the clean reference config to regenerate them:")
        print('    %run run_all_subjects.py --seed <SEED> --adabn --fnirs-temporal-align egta')
    elif n_clean > 0:
        print("\n>>> All checkpoints in this folder are clean reference checkpoints.")
        print(">>> Safe to use for the 3-seed ensemble.")


if __name__ == "__main__":
    main()
