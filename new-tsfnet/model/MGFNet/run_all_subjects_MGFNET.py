"""
run_all_subjects_MGFNET.py -- 26-subject x 3-session leave-one-session-out
(cross-session) sweep for MGFNet on the n-back task.

Everything follows the MGFNet paper (Sensors 2026, 26, 3402) except the
evaluation protocol, which is LOSO instead of the paper's within-subject
80/20 split. See mgfnet_train.py for the training / model-selection details.

ALWAYS TEST VP001 ALONE FIRST:
    python run_all_subjects_MGFNET.py --subjects VP001
Then the full sweep:
    python run_all_subjects_MGFNET.py --seed 42
Table 4 ablations:
    python run_all_subjects_MGFNET.py --use-cgscr 0
    python run_all_subjects_MGFNET.py --use-cross-modal 0
    python run_all_subjects_MGFNET.py --use-intra-modal 0
"""
import os

# Deterministic cuBLAS must be configured before torch initialises CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import datetime
import hashlib
import json
import time
import traceback

import numpy as np
import torch

from crash_safe_logger import CrashSafeLogger
from mgfnet_train import run_subject_loso_mgfnet

# ============================================================================
# CONFIG
# ============================================================================
DATA_ROOT = r"E:\kntu\Thesis\codes\Results\MGFNET_PAPER_EXACT_nback"
ALL_SUBJECTS = [f"VP{n:03d}" for n in range(1, 27)]

PAPER_DEFAULTS = dict(lr=1e-3, weight_decay=1e-3, max_epochs=60, batch_size=64,
                      selection="cv5", cv_folds=5)


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=str, default=None,
                    help="comma-separated subject IDs, e.g. VP001 or VP001,VP002. Default: all 26.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", type=str, default=None)
    ap.add_argument("--results-root", type=str, default=None,
                    help="where results_mgf_* folders go (default: --data-root)")
    ap.add_argument("--batch-size", type=int, default=PAPER_DEFAULTS["batch_size"])
    ap.add_argument("--max-epochs", type=int, default=PAPER_DEFAULTS["max_epochs"])
    ap.add_argument("--lr", type=float, default=PAPER_DEFAULTS["lr"])
    ap.add_argument("--weight-decay", type=float, default=PAPER_DEFAULTS["weight_decay"])
    ap.add_argument("--selection", choices=["cv5", "fixed"], default=PAPER_DEFAULTS["selection"],
                    help="cv5 (default): 5-fold trial-level CV inside the training sessions "
                         "selects the epoch budget, then retrain on all training data. "
                         "fixed: train exactly --max-epochs on all training data.")
    ap.add_argument("--cv-folds", type=int, default=PAPER_DEFAULTS["cv_folds"])
    ap.add_argument("--use-cgscr", type=int, default=1, choices=[0, 1],
                    help="1 = full MGFNet. 0 = Table 4 'w/o CGSCR' (static concatenation).")
    ap.add_argument("--use-cross-modal", type=int, default=1, choices=[0, 1])
    ap.add_argument("--use-intra-modal", type=int, default=1, choices=[0, 1])
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--no-deterministic", action="store_true",
                    help="allow non-deterministic cuDNN kernels (faster, not bit-reproducible)")
    return ap.parse_args(argv)


def variant_tag(args):
    """EVERY flag that changes model/training behaviour enters this tag, so
    two different configurations can never write into the same folder."""
    tag = "mgfnet"
    if not args.use_cgscr:
        tag += "_noCGSCR"
    if not args.use_cross_modal:
        tag += "_noCrossModal"
    if not args.use_intra_modal:
        tag += "_noIntraModal"
    parts = []
    for name, val, default in (("lr", args.lr, PAPER_DEFAULTS["lr"]),
                               ("wd", args.weight_decay, PAPER_DEFAULTS["weight_decay"]),
                               ("ep", args.max_epochs, PAPER_DEFAULTS["max_epochs"]),
                               ("bs", args.batch_size, PAPER_DEFAULTS["batch_size"]),
                               ("sel", args.selection, PAPER_DEFAULTS["selection"]),
                               ("cvk", args.cv_folds, PAPER_DEFAULTS["cv_folds"])):
        if val != default:
            parts.append(f"{name}{val}")
    if args.no_deterministic:
        parts.append("nondet")
    return tag + (("_" + "_".join(parts)) if parts else "")


CSV_FIELDS = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa",
              "eeg_only_acc", "fnirs_only_acc", "fusion_only_acc",
              "w_eeg", "w_fusion", "w_fnirs",
              "recall_0back", "recall_2back", "recall_3back",
              "elapsed_min", "error",
              # MGFNet-specific (appended, never reordered)
              "cgscr_w_1", "cgscr_w_2", "cgscr_w_3",
              "cgscr_w_12", "cgscr_w_13", "cgscr_w_23", "cgscr_max_cv",
              "mean_precision", "mean_recall", "mean_f1", "mean_itr_bits_min"]


def _fmt_cm(cm):
    names = ["0-back", "2-back", "3-back"]
    out = ["          " + "".join(f"{('pred ' + n):>12}" for n in names)]
    for i, n in enumerate(names):
        row_total = sum(cm[i])
        rec = (cm[i][i] / row_total * 100) if row_total else float("nan")
        out.append(f"  true {n:<4}" + "".join(f"{v:>12d}" for v in cm[i])
                   + f"   | recall {rec:5.1f}%")
    return "\n".join(out)


def _blank(v):
    return "" if v is None else v


def main(argv=None):
    args = parse_args(argv)
    data_root = args.data_root or DATA_ROOT
    results_root = args.results_root or data_root
    subjects = args.subjects.split(",") if args.subjects else ALL_SUBJECTS
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if not args.no_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # warn_only: a few CUDA backward kernels (e.g. linear interpolation)
        # have no deterministic implementation; warn instead of crashing.
        torch.use_deterministic_algorithms(True, warn_only=True)

    model_kw = dict(use_cgscr=bool(args.use_cgscr),
                    use_cross_modal=bool(args.use_cross_modal),
                    use_intra_modal=bool(args.use_intra_modal))
    # eeg_channels / nirs_channels are read from the data on disk inside
    # run_subject_loso_mgfnet, so they can never disagree with preprocessing.
    train_kw = dict(lr=args.lr, weight_decay=args.weight_decay, max_epochs=args.max_epochs,
                    selection=args.selection, n_cv_folds=args.cv_folds)

    tag_full = variant_tag(args)
    tag_hash = hashlib.md5(tag_full.encode()).hexdigest()[:10]
    run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}_{tag_hash}"
    results_dir = os.path.join(results_root, f"results_mgf_{tag_hash}_seed{args.seed}")
    os.makedirs(results_dir, exist_ok=True)
    summary_csv = os.path.join(results_dir, f"mgfnet_all_subjects_{run_tag}.csv")
    summary_json = os.path.join(results_dir, f"mgfnet_all_subjects_{run_tag}.json")
    with open(os.path.join(results_dir, "variant.json"), "w", encoding="utf-8") as f:
        json.dump({"variant_tag": tag_full, "hash": tag_hash, "args": vars(args)}, f, indent=2)

    def append_row(row):
        exists = os.path.exists(summary_csv)
        with open(summary_csv, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(row)

    all_results = {}
    with CrashSafeLogger(os.path.join(results_dir, f"mgfnet_run_log_{run_tag}.txt"),
                         experiment_tag=f"MGFNET_{tag_full.upper()}"):
        print(f"===== MGFNet LOSO RUN started {run_tag} (seed={args.seed}) =====")
        print(f"[CONFIG] variant={tag_full} | {model_kw} | {train_kw} | "
              f"batch_size={args.batch_size} | deterministic={not args.no_deterministic}")
        print(f"Subjects: {len(subjects)} | device={device}")
        print(f"Results dir: {results_dir}\n")

        run_t0 = time.time()
        for i, subject_id in enumerate(subjects, 1):
            subject_dir = os.path.join(data_root, subject_id)
            print(f"\n{'=' * 70}\n[{i}/{len(subjects)}] {subject_id}  "
                  f"({time.time() - run_t0:.1f}s elapsed)\n{'=' * 70}")
            row = {k: "" for k in CSV_FIELDS}
            row["subject_id"] = subject_id

            if not os.path.isdir(subject_dir):
                print(f"  SKIP: {subject_dir} does not exist (preprocessing not run?)")
                row["status"] = "skipped_no_data"
                all_results[subject_id] = row
                append_row(row)
                continue

            t0 = time.time()
            try:
                s = run_subject_loso_mgfnet(subject_dir, subject_id, device,
                                            batch_size=args.batch_size,
                                            results_dir=results_dir, model_kwargs=model_kw,
                                            train_kwargs=train_kw, base_seed=args.seed,
                                            log_fn=print)
                cm = s["confusion_matrix_summed"]
                print(f"\n{subject_id} BRANCH ACCURACY (mean over folds): "
                      f"eeg={s['branch_accuracy']['eeg_only_mean']} "
                      f"fnirs(HbO+HbR)={s['branch_accuracy']['fnirs_only_mean']} "
                      f"cross-modal={s['branch_accuracy']['fusion_only_mean']}")
                print(f"{subject_id} CGSCR routing: {s['cgscr_routing_stats']}")
                print(f"{subject_id} confusion matrix (summed over session-folds):")
                print(_fmt_cm(cm))

                cg = s["cgscr_routing_stats"]
                row.update({
                    "status": "ok",
                    "mean_acc": round(s["final_average_accuracy"], 4),
                    "std_acc": round(s["final_std_accuracy"], 4),
                    "mean_kappa": round(s["final_average_kappa"], 4),
                    "std_kappa": round(s["final_std_kappa"], 4),
                    "eeg_only_acc": _blank(s["branch_accuracy"]["eeg_only_mean"]),
                    "fnirs_only_acc": _blank(s["branch_accuracy"]["fnirs_only_mean"]),
                    "fusion_only_acc": _blank(s["branch_accuracy"]["fusion_only_mean"]),
                    "recall_0back": _blank(s["per_class_recall"][0]),
                    "recall_2back": _blank(s["per_class_recall"][1]),
                    "recall_3back": _blank(s["per_class_recall"][2]),
                    "elapsed_min": round((time.time() - t0) / 60.0, 2),
                    **{f"cgscr_w_{k}": _blank(cg[f"cgscr_w_{k}_mean"])
                       for k in ("1", "2", "3", "12", "13", "23")},
                    "cgscr_max_cv": _blank(cg["cgscr_max_cv_max"]),
                    "mean_precision": round(s["final_average_precision"], 4),
                    "mean_recall": round(s["final_average_recall"], 4),
                    "mean_f1": round(s["final_average_f1"], 4),
                    "mean_itr_bits_min": round(s["final_average_itr"], 3),
                })
            except Exception as e:
                print(f"  ERROR on {subject_id}: {e}")
                traceback.print_exc()
                row.update({"status": "error", "elapsed_min": round((time.time() - t0) / 60.0, 2),
                            "error": str(e)})
            all_results[subject_id] = row
            append_row(row)

        ok = [r for r in all_results.values() if r["status"] == "ok"]
        if ok:
            accs = np.array([r["mean_acc"] for r in ok], dtype=float)
            f1s = np.array([r["mean_f1"] for r in ok], dtype=float)
            print(f"\n===== ACROSS {len(ok)} SUBJECTS: acc={accs.mean():.4f}+-{accs.std():.4f} "
                  f"F1={f1s.mean():.4f} =====")
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n===== MGFNet LOSO RUN finished in {(time.time() - run_t0) / 60.0:.1f} min =====")
        print(f"Summary CSV: {summary_csv}")
        print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
