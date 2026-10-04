# ============================================================================
# Compute per-class recall + grand confusion matrix from ALREADY-SAVED results.
#
# WHY THIS EXISTS
#   The recall_0back/recall_2back/recall_3back columns were only added to
#   run_all_subjects.py recently, so older runs' CSVs lack them. But nothing
#   needs to be retrained to recover those numbers: every run already wrote
#   {results_dir}/cross_session_VPxxx_summary.json containing
#   session_results[i]["confusion_matrix_final"] for all three folds. Summing
#   those gives exactly the same recalls the new code prints.
#
#   Point it at any old results folder to make it directly comparable with a
#   new run, in seconds instead of a multi-hour retrain.
#
# USAGE
#   python recall_from_existing_results.py --results-dir "<path to results_..._seedNN>"
#   (optionally --compare-dir "<another results folder>" for a side-by-side)
# ============================================================================
import os
import json
import glob
import argparse

import numpy as np

CLASS_NAMES = ["0-back", "2-back", "3-back"]


def collect(results_dir):
    """Sum confusion matrices across every subject/fold in a results folder."""
    pattern = os.path.join(results_dir, "cross_session_*_summary.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No cross_session_*_summary.json found in {results_dir!r}. "
            f"Check the path -- it should be the variant+seed-tagged results folder, "
            f"e.g. .../results_paper_mean_noclip_effective_numberbeta0.9_..._seed42")

    grand = np.zeros((3, 3), dtype=np.int64)
    per_subject = {}
    accs, kappas = [], []
    n_folds = 0
    skipped = []

    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            skipped.append((os.path.basename(f), f"unreadable: {e}"))
            continue
        sid = d.get("subject_id", os.path.basename(f))
        subj_cm = np.zeros((3, 3), dtype=np.int64)
        for r in d.get("session_results", []):
            cm = np.asarray(r.get("confusion_matrix_final", []), dtype=np.int64)
            if cm.shape == (3, 3):
                subj_cm += cm
                n_folds += 1
        if subj_cm.sum() == 0:
            skipped.append((sid, "no usable confusion matrices"))
            continue
        grand += subj_cm
        per_subject[sid] = subj_cm
        if d.get("final_average_accuracy") is not None:
            accs.append(d["final_average_accuracy"])
        if d.get("final_average_kappa") is not None:
            kappas.append(d["final_average_kappa"])

    return grand, per_subject, accs, kappas, n_folds, skipped


def recalls(cm):
    tot = cm.sum(axis=1)
    return [float(cm[i, i] / tot[i]) if tot[i] > 0 else float("nan") for i in range(3)]


def fmt_cm(cm):
    out = ["          " + "".join(f"{('pred ' + n):>12}" for n in CLASS_NAMES)]
    for i, n in enumerate(CLASS_NAMES):
        rt = cm[i].sum()
        rec = (cm[i, i] / rt * 100) if rt else float("nan")
        out.append(f"  true {n:<4}" + "".join(f"{v:>12d}" for v in cm[i]) +
                   f"   | recall {rec:5.1f}%")
    return "\n".join(out)


def report(label, results_dir):
    grand, per_subject, accs, kappas, n_folds, skipped = collect(results_dir)
    print("=" * 72)
    print(f"{label}")
    print(f"  {results_dir}")
    print("=" * 72)
    print(f"subjects: {len(per_subject)} | folds: {n_folds}")
    if skipped:
        print(f"skipped: {skipped}")
    print()
    print(fmt_cm(grand))
    tot = grand.sum()
    diag = np.trace(grand)
    print()
    print(f"  accuracy from matrix : {diag / tot * 100:.2f}%  (n={tot} windows)")
    if accs:
        print(f"  mean_acc (per-fold avg): {np.mean(accs) * 100:.2f}%")
    if kappas:
        print(f"  mean_kappa            : {np.mean(kappas):.4f}")
    errs = tot - diag
    hard = grand[1, 2] + grand[2, 1]
    if errs > 0:
        print(f"  2-back <-> 3-back confusions: {hard} of {errs} errors "
              f"({hard / errs * 100:.1f}% of all errors)")
    return grand, recalls(grand), accs, kappas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True)
    p.add_argument("--compare-dir", default=None,
                   help="Optional second results folder for a side-by-side diff.")
    p.add_argument("--label", default="RUN A")
    p.add_argument("--compare-label", default="RUN B")
    a = p.parse_args()

    gA, rA, accA, kapA = report(a.label, a.results_dir)

    if a.compare_dir:
        print()
        gB, rB, accB, kapB = report(a.compare_label, a.compare_dir)

        print()
        print("=" * 72)
        print(f"SIDE-BY-SIDE  ({a.label}  ->  {a.compare_label})")
        print("=" * 72)
        print(f"{'metric':>22} {a.label:>12} {a.compare_label:>12} {'diff':>10}")
        accuracyA = np.trace(gA) / gA.sum()
        accuracyB = np.trace(gB) / gB.sum()
        rows = [("accuracy (matrix)", accuracyA, accuracyB)]
        if accA and accB:
            rows.append(("mean_acc", np.mean(accA), np.mean(accB)))
        if kapA and kapB:
            rows.append(("mean_kappa", np.mean(kapA), np.mean(kapB)))
        for i, n in enumerate(CLASS_NAMES):
            rows.append((f"recall {n}", rA[i], rB[i]))
        for name, va, vb in rows:
            print(f"{name:>22} {va:>12.4f} {vb:>12.4f} {vb - va:>+10.4f}")

        # The metric that actually motivated the asymmetric-loss experiment.
        errA = gA.sum() - np.trace(gA)
        errB = gB.sum() - np.trace(gB)
        hardA = gA[1, 2] + gA[2, 1]
        hardB = gB[1, 2] + gB[2, 1]
        print()
        print(f"  share of errors that are 2<->3 confusions: "
              f"{hardA / errA * 100:.1f}%  ->  {hardB / errB * 100:.1f}%")
        print("  (this is the number the asymmetric loss was designed to move; "
              "a drop here with flat accuracy means a better-balanced model, "
              "not a worse one)")


if __name__ == "__main__":
    main()
