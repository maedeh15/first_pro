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
RUN_TAG = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")   # unique per run, so a later comparison run never overwrites this baseline
SUMMARY_CSV = os.path.join(DATA_ROOT, f"baseline_all_subjects_{RUN_TAG}.csv")
SUMMARY_JSON = os.path.join(DATA_ROOT, f"baseline_all_subjects_{RUN_TAG}.json")

_csv_fields = ["subject_id", "status", "mean_acc", "std_acc", "mean_kappa", "std_kappa",
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
                      experiment_tag="11SEG_BASELINE") as log:
    print(f"===== FULL BASELINE RUN started {RUN_TAG} =====")
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
                   "std_acc": "", "mean_kappa": "", "std_kappa": "", "elapsed_min": "", "error": ""}
            all_results[subject_id] = row
            _append_csv_row(row)
            continue

        t0 = time.time()
        try:
            mean_acc, std_acc, mean_kappa, std_kappa = run_subject_cross_session(
                subject_dir, subject_id,
                device="cuda" if torch.cuda.is_available() else "cpu",
                batch_size=16,
                run_diagnostics_flag=RUN_DIAGNOSTICS,
            )
            elapsed = (time.time() - t0) / 60
            row = {"subject_id": subject_id, "status": "ok",
                   "mean_acc": mean_acc, "std_acc": std_acc,
                   "mean_kappa": mean_kappa, "std_kappa": std_kappa,
                   "elapsed_min": round(elapsed, 2), "error": ""}
            print(f"\n[{subject_id}] DONE in {elapsed:.1f} min | "
                  f"acc={mean_acc:.4f}+-{std_acc:.4f}  kappa={mean_kappa:.4f}+-{std_kappa:.4f}")

        except Exception as e:
            elapsed = (time.time() - t0) / 60
            print(f"\n[{subject_id}] FAILED after {elapsed:.1f} min: {e}")
            traceback.print_exc()
            row = {"subject_id": subject_id, "status": "error", "mean_acc": "",
                   "std_acc": "", "mean_kappa": "", "std_kappa": "",
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
                  f"kappa={r['mean_kappa']:.4f} +- {r['std_kappa']:.4f}")
        elif r["status"] == "skipped_no_data":
            print(f"{subject_id}: SKIPPED (no preprocessed data)")
        else:
            print(f"{subject_id}: FAILED ({r.get('error', 'unknown error')})")

    print(f"\nFull results: {SUMMARY_CSV}")
    print(f"             {SUMMARY_JSON}")
