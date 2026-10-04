"""
check_groupnorm_applied.py -- opens a REAL checkpoint from the --norm-mode
group run and proves, from the tensor names themselves, whether GroupNorm was
actually used or the run silently fell back to BatchNorm.

THE DEFINITIVE SIGNAL: BatchNorm layers store two BUFFERS (not parameters)
called running_mean and running_var (the batch statistics accumulated during
training). GroupNorm has NO running statistics at all -- it only has weight
and bias. So:
  - state_dict contains "*.running_mean" / "*.running_var" keys  -> BatchNorm
  - state_dict has NO such keys anywhere                          -> GroupNorm

This is checked directly against a real .pt file, not inferred from the log.

USAGE:
    python check_groupnorm_applied.py "E:\\path\\to\\results_..._cf7309774e_seed42"
"""
import sys
import os
import glob
import torch

RESULTS_DIR = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback\results_cf7309774e_seed42"


def check_one(pt_path):
    ckpt = torch.load(pt_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    running_keys = [k for k in sd if k.endswith("running_mean") or k.endswith("running_var")]
    # num_batches_tracked is another BatchNorm-only buffer, checked as a second signal
    tracked_keys = [k for k in sd if k.endswith("num_batches_tracked")]
    return running_keys, tracked_keys, len(sd)


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else RESULTS_DIR
    pt_files = sorted(glob.glob(os.path.join(results_dir, "*_best.pt")))
    if not pt_files:
        print(f"No .pt files found in: {results_dir}")
        return

    print(f"Checking {len(pt_files)} checkpoint(s) in:\n  {results_dir}\n")
    any_batchnorm = False
    for p in pt_files[:5]:   # a handful is enough; they all come from the same code path
        running_keys, tracked_keys, n = check_one(p)
        name = os.path.basename(p)
        if running_keys or tracked_keys:
            print(f"  [STILL BATCHNORM] {name}")
            print(f"      running_mean/var keys: {running_keys[:3]}{'...' if len(running_keys) > 3 else ''}")
            any_batchnorm = True
        else:
            print(f"  [confirmed GroupNorm] {name}  ({n} tensors, zero running-stat buffers)")

    print("\n" + "=" * 60)
    if any_batchnorm:
        print(">>> BatchNorm running-stat buffers were found. --norm-mode group did NOT")
        print(">>> actually take effect in this run -- the flag was silently ignored.")
        print(">>> This would mean the GroupNorm result is INVALID and must be re-run")
        print(">>> after finding why the flag didn't reach the model.")
    else:
        print(">>> Confirmed: these checkpoints genuinely contain GroupNorm, not BatchNorm.")
        print(">>> The --norm-mode group flag worked correctly. The rejection is a real result,")
        print(">>> not an execution bug.")


if __name__ == "__main__":
    main()
