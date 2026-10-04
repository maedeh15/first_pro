"""
check_beta.py -- C-2 post-hoc audit #1 (costs ~2 minutes, no GPU).

WHAT IT ANSWERS: during the C-2d ('labram_blend') runs, did the model ever
actually GIVE the frozen LaBraM head any weight?

WHY IT MATTERS: in C-2d the two EEG-only predictions are blended BEFORE the
paper's Eq. 4, as

    y_eeg = (1 - beta) * y_eeg_native + beta * y_labram

with `beta = sigmoid(beta_labram)` and `beta_labram` initialized to -6.0, so
beta starts at ~0.0025 -- a deliberate near-no-op the model has to actively
train away from. `beta_labram` was written into every per-session result
JSON under learned_weights, but it was never printed to the run log, so it
has not been inspected yet.

HOW TO READ THE ANSWER:
  * beta stays near 0.0025  -> the model LEARNED TO IGNORE the foundation
    model. This is the single strongest possible piece of evidence for the
    C-2 rejection, and belongs in the thesis: the rejection is not "we could
    not make it work", it is "the model, given a free learnable gate, chose
    not to use it".
  * beta rises well above 0.0025 but accuracy did not improve -> the model
    DID try to use LaBraM and it did not help. A different, weaker claim,
    and one that leaves more of the hypothesis space open.

USAGE:
    python check_beta.py
    python check_beta.py --root "E:/path/to/Results/TSFNET_PAPER_EXACT_nback"
"""

import argparse
import glob
import json
import os
import statistics


DEFAULT_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
INIT_BETA = 1.0 / (1.0 + 2.718281828459045 ** 6.0)   # sigmoid(-6.0) ~ 0.002473


def collect(root):
    """Walks the results tree and pulls every stored beta_labram / w_* value."""
    pattern = os.path.join(root, "**", "results_VP*_session*.json")
    rows = []
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [skip] unreadable: {os.path.basename(path)} ({exc})")
            continue
        weights = doc.get("learned_weights") or {}
        beta = weights.get("beta_labram")
        if beta is None:
            continue          # a run from before C-2d, or a non-labram run
        rows.append({
            "run": os.path.basename(os.path.dirname(path)),
            "file": os.path.basename(path),
            "beta": float(beta),
            "w_eeg": weights.get("w_eeg"),
            "w_fusion": weights.get("w_fusion"),
            "w_fnirs": weights.get("w_fnirs"),
            "test_acc": doc.get("test_acc"),
        })
    return rows


def summarize(rows):
    betas = [r["beta"] for r in rows]
    print(f"\nfound {len(rows)} session-level results carrying beta_labram")
    if not rows:
        print("\n  Nothing to report. Either the C-2d runs are not under --root, or")
        print("  the runs predate the beta_labram field. Check the folder name:")
        print("  C-2d results use the '_labramblend' sub-tag.")
        return

    print(f"\n  initial value (sigmoid(-6.0))      : {INIT_BETA:.6f}")
    print(f"  min                                : {min(betas):.6f}")
    print(f"  mean                               : {statistics.mean(betas):.6f}")
    print(f"  median                             : {statistics.median(betas):.6f}")
    print(f"  max                                : {max(betas):.6f}")
    if len(betas) > 1:
        print(f"  std                                : {statistics.stdev(betas):.6f}")

    # Per-run breakdown, so the three seeds can be compared.
    runs = sorted({r["run"] for r in rows})
    if len(runs) > 1:
        print("\n  per results-folder:")
        for run in runs:
            sub = [r["beta"] for r in rows if r["run"] == run]
            print(f"    {run:<48} n={len(sub):<3} mean beta={statistics.mean(sub):.6f}")

    # The interpretation, stated explicitly so it is not left to the reader.
    grew = statistics.mean(betas) / INIT_BETA
    print("\n  --- interpretation ---")
    print(f"  mean beta is {grew:.1f}x its initial value.")
    if grew < 2.0:
        print("  => The model essentially NEVER opened the gate. It was handed a free,")
        print("     trainable scalar that could have let the pretrained representation in,")
        print("     and it left that scalar closed. Report this verbatim in the thesis:")
        print("     the rejection of C-2 is a decision the MODEL made, not only a")
        print("     conclusion drawn from aggregate accuracy.")
    elif grew < 20.0:
        print("  => The gate opened somewhat but stayed small. The model gave the")
        print("     foundation model a minor, non-decisive role. Consistent with the")
        print("     observed near-zero effect on accuracy.")
    else:
        print("  => The gate opened substantially: the model DID try to use the")
        print("     pretrained representation. Since accuracy did not improve, the")
        print("     honest claim is 'it was used and did not help', which is weaker")
        print("     than 'the model ignored it'. Adjust the discussion accordingly.")

    # Sanity: beta should never exceed 1 or fall below 0 (it is a sigmoid).
    bad = [r for r in rows if not (0.0 <= r["beta"] <= 1.0)]
    if bad:
        print(f"\n  [WARNING] {len(bad)} value(s) outside [0, 1] -- beta is a sigmoid "
              f"output and cannot legitimately land there. Inspect: "
              f"{[b['file'] for b in bad][:5]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="results root to search recursively")
    ap.add_argument("--csv", default=None,
                    help="optional path to write the per-session table to")
    args = ap.parse_args()

    print(f"searching: {args.root}")
    if not os.path.isdir(args.root):
        print(f"\n  ERROR: not a directory. Pass the correct path with --root.")
        return

    rows = collect(args.root)
    summarize(rows)

    if args.csv and rows:
        import csv as _csv
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  per-session table written to: {args.csv}")


if __name__ == "__main__":
    main()
