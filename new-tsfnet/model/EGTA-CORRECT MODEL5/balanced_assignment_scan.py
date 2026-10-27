"""
balanced_assignment_scan.py -- run balanced_assignment_eval on EVERY posteriors folder under a root
and print one line per folder, so the folder behind an unexplained number can be identified by name.

The numbers are descriptive (transductive, non-causal; see balanced_assignment_eval.py). A folder that
does not hold the 3/3/3 composition is reported as skipped, not silently dropped.
"""
import argparse, glob, os
import numpy as np
from scipy.stats import wilcoxon
from balanced_assignment_eval import load, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder to search recursively")
    ap.add_argument("--head", default="final", choices=["final", "eeg", "fusion", "fnirs"])
    a = ap.parse_args()
    dirs = sorted({os.path.dirname(f) for f in
                   glob.glob(os.path.join(a.root, "**", "*_session*.npz"), recursive=True)})
    if not dirs:
        raise SystemExit(f"no *_session*.npz under {a.root}")
    print(f"{len(dirs)} folders\n")
    print(f"{'argmax':>7} {'balanced':>8} {'gain':>8} {'better/worse':>13} {'p':>7}  folder")
    for d in dirs:
        try:
            subs, A, B, pc, fb, fw, nf = evaluate(load(d, a.head))
        except SystemExit as e:
            print(f"{'skipped':>7} {'':>8} {'':>8} {'':>13} {'':>7}  {d}  ({str(e)[:60]})")
            continue
        dd = B - A
        p = 1.0 if not np.any(dd != 0) else float(wilcoxon(A, B).pvalue)
        print(f"{A.mean():7.4f} {B.mean():8.4f} {dd.mean():+8.4f} "
              f"{int((dd > 0).sum()):>6}/{int((dd < 0).sum()):<6} {p:7.4f}  {d}")


if __name__ == "__main__":
    main()
