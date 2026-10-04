# ============================================================================
# FULL BASELINE RUN -- all 26 subjects, subject-dependent (paper-exact
# protocol: Sec. 3.6), for later comparison against any changes you make.
#
# Paste this as CELL 9 in tsfnet_model_notebook.py (after CELL 8's imports/
# setup are already in the kernel), or run standalone with %run after
# %run tsfnet_model_notebook.py and %run crash_safe_logger.py.
#
# WHY THIS SHAPE:
#   - run_diagnostics_flag=False by default here: PCA/t-SNE/saliency for
#     26 subjects x 3 sessions = 78 diagnostic runs would take a very long
#     time and isn't needed for a baseline comparison run (accuracy/kappa
#     numbers are what you compare across changes). Diagnostics are for
#     drilling into ONE subject's behavior, not for a 26-subject sweep.
#     Set RUN_DIAGNOSTICS=True below if you want them anyway.
#   - Per-subject try/except: one subject crashing (e.g. a corrupted .npy,
#     an out-of-memory spike) does NOT stop the other 25 -- this matters a
#     lot for an unattended multi-hour run.
#   - CrashSafeLogger wraps the WHOLE run: every subject's full training
#     log is on disk in real time, so even a full kernel crash partway
#     through subject 15 still leaves you with subjects 1-14's complete
#     logs (and the results/checkpoints already written by
#     run_subject_cross_session for each finished subject).
#   - A combined summary (baseline_all_subjects_<timestamp>.csv/.json) is
#     written incrementally (after EACH subject, not just at the end), so
#     partial results are never lost even if the run stops partway.
# ============================================================================
import os
import csv
import json
import time
import datetime
import traceback
import argparse
import random

# Make this script self-sufficient regardless of what was run in earlier
# cells: explicitly import CrashSafeLogger here rather than relying on it
# already being in the notebook's namespace from a previous %run. Requires
# crash_safe_logger.py to be in the same folder as this script (or on
# sys.path). If CrashSafeLogger truly isn't available for any other reason,
# a plain no-op fallback context manager is used instead of crashing --
# real-time crash-safe logging is a nice-to-have, not something that should
# ever block your 26-subject run from starting.
try:
    from crash_safe_logger import CrashSafeLogger
except ImportError:
    print("WARNING: could not import CrashSafeLogger from crash_safe_logger.py "
          "(is it in the same folder as this script?). Falling back to plain "
          "print() with no crash-safe file logging for this run.")

    class CrashSafeLogger:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

import torch   # needed for torch.cuda.is_available() below; harmless re-import if already loaded

# ----------------------------------------------------------------------------
# SEED CONTROL -- for the multi-seed ablation-vs-baseline statistical comparison.
# By default this script previously had NO fixed seed at all, meaning every run
# was a random draw from the training-stochasticity distribution (weight
# initialization, batch order, dropout masks, etc.) -- this is exactly the
# noise source that made the 11-seg vs 20-seg paired comparison statistically
# non-significant despite a consistent positive mean effect. Running the SAME
# preprocessing output with 3 different fixed seeds and averaging each
# subject's result across those 3 runs reduces this noise component by
# ~1/sqrt(3) (~42% reduction in std), making the true effect (if any) easier
# to detect without collecting any new data.
#
# Usage:
#   python run_all_subjects.py --seed 42
#   python run_all_subjects.py --seed 123
#   python run_all_subjects.py --seed 2024
# (Run three times, once per seed value, for both the 11-segment and
# 20-segment pipelines -- 6 total runs -- then average_seeds_and_compare.py
# aggregates all of them.)
_arg_parser = argparse.ArgumentParser()
_arg_parser.add_argument("--seed", type=int, default=42,
                          help="Random seed for numpy/torch/python random, for reproducibility "
                               "and for the multi-seed noise-averaging ablation comparison.")
# RESULT OF PRIORITY-1 SUB-EXPERIMENT (gradient clipping, norm=1.0): REJECTED.
# Both seeds (42, 2024) showed a consistent drop vs. the no-clip baseline
# (overall acc -1.57pp, kappa -0.023, fusion-branch acc -2.06pp), matching
# the same "EEG branch improves alone, fusion/overall degrades" pattern seen
# in every preprocessing ablation (AutoReject, ICLabel, SSP). Default is now
# False (no clipping) so this script reproduces the accepted baseline unless
# explicitly re-enabled with --grad-clip (e.g. to revisit with a larger norm
# such as 5.0 later, if time permits).
_arg_parser.add_argument("--grad-clip", dest="grad_clip", action="store_true",
                          help="Re-enable gradient clipping (norm=1.0), which was tested "
                               "and REJECTED in the priority-1 sub-experiment. Off by default.")
_arg_parser.set_defaults(grad_clip=False)

# ISOLATED ABLATION FLAG (priority-2 sub-experiment: fusion combination formula).
# Default is now "paper_mean" (Eq. 4 exactly: independent Sigmoid per weight,
# plain mean / divide-by-3, no renormalization) -- this is the ONLY variable
# this script changes relative to the accepted priority-1 baseline (no grad
# clipping); everything else (epochs, patience, lr, weight_decay, class
# weights, CAFE dim) is untouched. Pass --fusion-combination wsum to
# reproduce the prior (renormalized-by-sum) behavior for a direct A/B run.
_arg_parser.add_argument("--fusion-combination", type=str, default="paper_mean",
                          choices=["paper_mean", "wsum"],
                          help="Final decision-level combination formula for y_eeg/y_fusion/"
                               "y_fnirs. 'paper_mean' (default): Eq.4 as written, sigmoid+mean/3. "
                               "'wsum': prior renormalized-by-sum behavior, for comparison only.")

# ISOLATED ABLATION FLAG (priority-3 sub-experiment: inverse-frequency class
# weighting). Default False so a plain `python run_all_subjects.py --seed X`
# still reproduces the accepted priority-1+2 baseline exactly; pass
# --class-weight to enable it. This is the ONLY variable this script changes
# relative to that baseline when the flag is used; everything else (epochs,
# patience, lr, weight_decay, combination formula, CAFE dim) is untouched.
_arg_parser.add_argument("--class-weight", dest="use_class_weighting", action="store_true",
                          help="Enable inverse-frequency class weighting in the loss "
                               "(priority-3 sub-experiment). Off by default.")
_arg_parser.set_defaults(use_class_weighting=False)
SEED_ARGS, _ = _arg_parser.parse_known_args()
SEED = SEED_ARGS.seed
GRAD_CLIP_NORM = 1.0 if SEED_ARGS.grad_clip else None
FUSION_COMBINATION = SEED_ARGS.fusion_combination
USE_CLASS_WEIGHTING = SEED_ARGS.use_class_weighting

random.seed(SEED)
import numpy as np
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
# Deterministic cuDNN (slightly slower, but removes another source of
# run-to-run noise beyond just the initial seed).
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"[SEED CONTROL] Using seed={SEED} (numpy/torch/python random all seeded, cuDNN deterministic)")

# Same self-sufficiency pattern for run_subject_cross_session: if it's already in
# the notebook's namespace (from an earlier %run tsfnet_model_notebook.py), that
# version is used as-is. Otherwise, try importing it from the installed
# tsfnet_model PACKAGE (folder) if that's what you're using; if that package
# isn't present either (e.g. you're only using the single-file
# tsfnet_model_notebook.py, not the tsfnet_model/ package folder), fall back
# to running tsfnet_model_notebook.py itself, found next to THIS script --
# this covers all three ways this project can be set up without silently
# depending on cell execution order in any of them.
try:
    run_subject_cross_session
except NameError:
    try:
        from tsfnet_model.train import run_subject_cross_session
    except ModuleNotFoundError:
        try:
            _this_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            # __file__ may not be defined in some exec/%run contexts -- fall back
            # to the current working directory (Jupyter's default is usually the
            # notebook's own folder, which is where these files normally live).
            _this_dir = os.getcwd()
        _notebook_path = os.path.join(_this_dir, "tsfnet_model_notebook.py")
        if not os.path.isfile(_notebook_path):
            raise ModuleNotFoundError(
                "Could not find run_subject_cross_session anywhere: no "
                "tsfnet_model package installed, and tsfnet_model_notebook.py "
                f"not found next to this script ({_notebook_path}). Put "
                "tsfnet_model_notebook.py in the same folder as "
                "run_all_subjects.py, or run it first via %run.")
        print(f"NOTE: tsfnet_model package not found; loading definitions from "
              f"{_notebook_path} instead.")
        exec(open(_notebook_path, encoding="utf-8").read(), globals())

DATA_ROOT = r"E:\kntu\Thesis\codes\Results\TSFNET_PAPER_EXACT_nback"
SUBJECTS = [f"VP{i:03d}" for i in range(1, 27)]     # VP001..VP026
RUN_DIAGNOSTICS = False                              # True = also make PCA/t-SNE/saliency plots per subject (slow)
# Variant tag distinguishes this isolated ablation (fusion combination formula,
# plus the now-accepted no-clip setting from priority-1) from any other
# variant in every filename, so conditions can NEVER be accidentally mixed up
# or overwrite each other on disk.
_CLIP_SUBTAG = "gradclip1p0" if GRAD_CLIP_NORM is not None else "noclip"
_CW_SUBTAG = "classweight" if USE_CLASS_WEIGHTING else "noclassweight"
_VARIANT_TAG = f"{FUSION_COMBINATION}_{_CLIP_SUBTAG}_{_CW_SUBTAG}"
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{SEED}_{_VARIANT_TAG}"   # unique per run+seed+variant, so a later comparison run never overwrites this baseline
SUMMARY_CSV = os.path.join(DATA_ROOT, f"baseline_all_subjects_{RUN_TAG}.csv")
SUMMARY_JSON = os.path.join(DATA_ROOT, f"baseline_all_subjects_{RUN_TAG}.json")

_csv_fields = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa",
               "eeg_only_acc", "fnirs_only_acc", "fusion_only_acc",
               "w_eeg", "w_fusion", "w_fnirs",
               "elapsed_min", "error"]


def _append_csv_row(row):
    file_exists = os.path.exists(SUMMARY_CSV)
    with open(SUMMARY_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_csv_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


all_results = {}

with CrashSafeLogger(os.path.join(DATA_ROOT, f"baseline_run_log_{RUN_TAG}.txt"),
                      experiment_tag=f"11SEG_BASELINE_{_VARIANT_TAG.upper()}") as log:
    print(f"===== FULL BASELINE RUN started {RUN_TAG} (seed={SEED}) =====")
    print(f"[ABLATION] use_class_weighting={USE_CLASS_WEIGHTING!r} (priority-3 isolated variable) "
          f"| fusion_combination={FUSION_COMBINATION!r} (priority-2 result: accepted) "
          f"| grad_clip_norm={GRAD_CLIP_NORM!r} (priority-1 result: rejected, fixed at None)")
    print(f"Subjects: {len(SUBJECTS)} | diagnostics={RUN_DIAGNOSTICS} | device="
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Summary CSV: {SUMMARY_CSV}")
    print(f"Summary JSON: {SUMMARY_JSON}\n")

    run_t0 = time.time()
    for i, subject_id in enumerate(SUBJECTS, 1):
        subject_dir = os.path.join(DATA_ROOT, subject_id)
        print(f"\n{'='*70}")
        print(f"[{i}/{len(SUBJECTS)}] {subject_id}  ({time.time()-run_t0:.1f}s elapsed so far)")
        print(f"{'='*70}")

        if not os.path.isdir(subject_dir):
            print(f"  SKIP: {subject_dir} does not exist (preprocessing not run for this subject?)")
            row = {"subject_id": subject_id, "status": "skipped_no_data", "mean_acc": "",
                   "std_acc": "", "mean_kappa": "", "std_kappa": "",
                   "eeg_only_acc": "", "fnirs_only_acc": "", "fusion_only_acc": "",
                   "w_eeg": "", "w_fusion": "", "w_fnirs": "",
                   "elapsed_min": "", "error": ""}
            all_results[subject_id] = row
            _append_csv_row(row)
            continue

        t0 = time.time()
        try:
            mean_acc, std_acc, mean_kappa, std_kappa, summary = run_subject_cross_session(
                subject_dir, subject_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=16,
                run_diagnostics_flag=RUN_DIAGNOSTICS,
                fusion_combination=FUSION_COMBINATION,   # priority-2 result: accepted, fixed at "paper_mean"
                use_class_weighting=USE_CLASS_WEIGHTING, # ISOLATED ABLATION (priority-3): only this differs vs. the accepted priority-1+2 baseline
                grad_clip_norm=GRAD_CLIP_NORM,           # priority-1 result: rejected -> default None
            )
            elapsed = (time.time() - t0) / 60
            branch = summary["branch_accuracy"]
            weights = summary["learned_fusion_weights"]
            row = {"subject_id": subject_id, "status": "ok",
                   "mean_acc": mean_acc, "std_acc": std_acc,
                   "mean_kappa": mean_kappa, "std_kappa": std_kappa,
                   "eeg_only_acc": round(branch["eeg_only_mean"], 4),
                   "fnirs_only_acc": round(branch["fnirs_only_mean"], 4),
                   "fusion_only_acc": round(branch["fusion_only_mean"], 4),
                   "w_eeg": round(weights["w_eeg_mean"], 4),
                   "w_fusion": round(weights["w_fusion_mean"], 4),
                   "w_fnirs": round(weights["w_fnirs_mean"], 4),
                   "elapsed_min": round(elapsed, 2), "error": ""}
            print(f"\n[{subject_id}] DONE in {elapsed:.1f} min | "
                  f"acc={mean_acc:.4f}+-{std_acc:.4f}  kappa={mean_kappa:.4f}+-{std_kappa:.4f} | "
                  f"branch acc: eeg={branch['eeg_only_mean']:.4f} fnirs={branch['fnirs_only_mean']:.4f} "
                  f"fusion={branch['fusion_only_mean']:.4f} | "
                  f"fusion weights: w_eeg={weights['w_eeg_mean']:.4f} w_fusion={weights['w_fusion_mean']:.4f} "
                  f"w_fnirs={weights['w_fnirs_mean']:.4f}")

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n[{subject_id}] FAILED after {elapsed:.1f} min: {e}")
            traceback.print_exc()
            row = {"subject_id": subject_id, "status": "error", "mean_acc": "",
                   "std_acc": "", "mean_kappa": "", "std_kappa": "",
                   "eeg_only_acc": "", "fnirs_only_acc": "", "fusion_only_acc": "",
                   "w_eeg": "", "w_fusion": "", "w_fnirs": "",
                   "elapsed_min": round(elapsed, 2), "error": str(e)}

        all_results[subject_id] = row
        _append_csv_row(row)               # written immediately -- survives a crash on the NEXT subject
        with open(SUMMARY_JSON, "w") as f:  # full JSON rewritten each time (cheap at 26 subjects)
            json.dump(all_results, f, indent=2)

    # ----- final aggregate across all successfully completed subjects -----
    ok_rows = [r for r in all_results.values() if r["status"] == "ok"]
    print(f"\n\n{'='*70}")
    print(f"BASELINE RUN COMPLETE: {len(ok_rows)}/{len(SUBJECTS)} subjects finished successfully")
    print(f"{'='*70}")
    if ok_rows:
        import statistics
        accs = [r["mean_acc"] for r in ok_rows]
        kappas = [r["mean_kappa"] for r in ok_rows]
        print(f"Grand mean accuracy across subjects: {statistics.mean(accs):.4f} "
              f"(+-{statistics.pstdev(accs):.4f})")
        print(f"Grand mean kappa across subjects:    {statistics.mean(kappas):.4f} "
              f"(+-{statistics.pstdev(kappas):.4f})")

        eeg_accs = [r["eeg_only_acc"] for r in ok_rows]
        fnirs_accs = [r["fnirs_only_acc"] for r in ok_rows]
        fusion_accs = [r["fusion_only_acc"] for r in ok_rows]
        w_eeg = [r["w_eeg"] for r in ok_rows]
        w_fusion = [r["w_fusion"] for r in ok_rows]
        w_fnirs = [r["w_fnirs"] for r in ok_rows]
        print(f"Grand mean branch accuracy:  eeg={statistics.mean(eeg_accs):.4f}  "
              f"fnirs={statistics.mean(fnirs_accs):.4f}  fusion={statistics.mean(fusion_accs):.4f}")
        print(f"Grand mean learned fusion weights:  w_eeg={statistics.mean(w_eeg):.4f}  "
              f"w_fusion={statistics.mean(w_fusion):.4f}  w_fnirs={statistics.mean(w_fnirs):.4f}")
    failed = [sid for sid, r in all_results.items() if r["status"] != "ok"]
    if failed:
        print(f"\nSubjects that did NOT complete: {failed}")

    # ----- per-subject listing: mean accuracy/kappa for every subject, one line each -----
    print(f"\n{'='*70}")
    print(f"PER-SUBJECT RESULTS ({len(ok_rows)} completed / {len(SUBJECTS)} total)")
    print(f"{'='*70}")
    for subject_id in SUBJECTS:   # preserves VP001..VP026 order regardless of completion order
        r = all_results.get(subject_id)
        if r is None:
            print(f"{subject_id}: NOT RUN")
        elif r["status"] == "ok":
            print(f"{subject_id}: acc={r['mean_acc']:.4f} +- {r['std_acc']:.4f}  "
                  f"kappa={r['mean_kappa']:.4f} +- {r['std_kappa']:.4f}  |  "
                  f"branch acc: eeg={r['eeg_only_acc']:.4f} fnirs={r['fnirs_only_acc']:.4f} "
                  f"fusion={r['fusion_only_acc']:.4f}  |  "
                  f"fusion weights: w_eeg={r['w_eeg']:.4f} w_fusion={r['w_fusion']:.4f} "
                  f"w_fnirs={r['w_fnirs']:.4f}")
        elif r["status"] == "skipped_no_data":
            print(f"{subject_id}: SKIPPED (no preprocessed data)")
        else:
            print(f"{subject_id}: FAILED ({r.get('error', 'unknown error')})")

    print(f"\nFull results: {SUMMARY_CSV}")
    print(f"             {SUMMARY_JSON}")
