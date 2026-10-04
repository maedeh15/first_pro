# ============================================================================
# run_parallel_preprocessing.py
#
# HOW TO USE:
# 1. Put this file in the SAME folder as your main preprocessing script
#    (the one converted from Preprocessing_v2_run_a1b_10s.ipynb).
# 2. Rename your main script file to e.g. "preprocessing_pipeline.py"
#    (it must contain BASE_DATASET_PATH, OUTPUT_PATH, LOG_FILE, and the
#    process_subject() function -- exactly as in your notebook cell).
# 3. In that script, make sure the code under `if __name__ == "__main__":`
#    at the bottom ONLY runs the loop -- do NOT delete process_subject or
#    the constants above it. This wrapper imports everything else from it.
# 4. Run: python run_parallel_preprocessing.py
#
# WHY THIS STRUCTURE:
# - Thread-limiting env vars MUST be set BEFORE numpy/scipy/mne are
#   imported anywhere in the process, otherwise MKL/OpenBLAS grabs
#   multiple threads per worker and 14 processes end up fighting each
#   other for the same cores (oversubscription -> slower than serial).
# - process_subject() is already fully self-contained per subject, so
#   ProcessPoolExecutor can call it directly without touching your
#   pipeline logic.
# ============================================================================

import os

# --- MUST happen before importing numpy/scipy/mne (done here AND inside
# each worker via the initializer, since the main process may also import
# numpy indirectly) ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import time
import traceback
import psutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- EDIT THIS: name of your renamed pipeline script (without .py) ----
PIPELINE_MODULE_NAME = "preprocessing_pipeline"

# ---- EDIT THIS: subjects to process ----
SUBJECT_IDS = [f"VP{n:03d}" for n in range(18, 27)]

# ---- Worker count ----
# Physical cores on your machine = 14, but RAM (20.5 GB free) is the real
# ceiling here, not CPU. Start conservative and raise it once you've
# watched peak RAM per subject (see README note printed at the end).
N_WORKERS = 4


def _worker_init():
    """Runs once inside each worker process before any task executes."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

    # Non-interactive backend: matplotlib will crash/hang in worker
    # processes on Windows if it tries to open a GUI window.
    import matplotlib
    matplotlib.use("Agg")


def _run_one_subject(subject_id):
    """Runs in a worker process. Imports the pipeline module lazily so the
    thread-limiting env vars above are already in place before numpy/mne
    load inside it."""
    t0 = time.time()
    try:
        pipeline = __import__(PIPELINE_MODULE_NAME)
        pipeline.process_subject(subject_id)
        return subject_id, True, time.time() - t0, None
    except Exception as e:
        tb = traceback.format_exc()
        return subject_id, False, time.time() - t0, tb


def main():
    print(f"Physical cores available: {psutil.cpu_count(logical=False)}")
    print(f"Free RAM at start: {psutil.virtual_memory().available / 1e9:.1f} GB")
    print(f"Using {N_WORKERS} worker processes for {len(SUBJECT_IDS)} subjects")
    print("-" * 60)

    total_t0 = time.time()
    results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_worker_init) as ex:
        futures = {ex.submit(_run_one_subject, sid): sid for sid in SUBJECT_IDS}
        for fut in as_completed(futures):
            subject_id, ok, elapsed, err = fut.result()
            status = "OK" if ok else "FAILED"
            print(f"[{status}] {subject_id} in {elapsed/60:.1f} min "
                  f"(free RAM: {psutil.virtual_memory().available / 1e9:.1f} GB)")
            if not ok:
                print(err)
            results.append((subject_id, ok, elapsed))

    total_elapsed = time.time() - total_t0
    n_ok = sum(1 for _, ok, _ in results if ok)
    print("-" * 60)
    print(f"Done: {n_ok}/{len(SUBJECT_IDS)} subjects succeeded")
    print(f"Total wall time: {total_elapsed/60:.1f} min")
    print(
        "\nNOTE: watch Task Manager RAM usage during this run. "
        "If free RAM drops near 0 or swap usage climbs a lot, lower N_WORKERS. "
        "If RAM stayed comfortably above ~4-5 GB free the whole time, "
        "you can raise N_WORKERS (try 6, then 8) on the next run."
    )


if __name__ == "__main__":
    main()
